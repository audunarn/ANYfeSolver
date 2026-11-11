"""Private routing/authority regressions; no broader parity claim."""
from copy import deepcopy
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import _ge_beam3_g3c_so3_numerics as kernel
from anyfesolver import _ge_beam3_mixed_ad as shared
from anyfesolver import _native_material_protocol as protocol
from anyfesolver._ge_beam3_g3c_stable import authority, owner, element, operator
from anyfesolver._ge_beam3_g3c_stable import beam, shell, joint, chart, boundary, line_work, fit, compensated


def load():
    return dict(kind='LOAD_STAGE', load_factor=0., force_scale=.01, root_stage=0)


def test_exact_derivative_routes_and_authentic_shared_types():
    for module in (owner, operator, beam, shell, joint):
        assert module.so3_exp is kernel.so3_exp and module.so3_log is kernel.so3_log
        assert module.Jet2 is shared.Jet2
    assert chart.so3_exp is kernel.so3_exp and line_work.so3_exp is kernel.so3_exp
    assert element.NativeMaterialValidator is protocol.NativeMaterialValidator
    assert element.NativeMaterialContext is protocol.NativeMaterialContext
    assert element.chart_pullback is boundary.chart_pullback
    assert boundary.exp_chart_terms is chart.exp_chart_terms
    assert operator.line_work is line_work.evaluate
    assert beam._exp_terms is joint._exp_terms and shell._exp_terms is joint._exp_terms
    assert shell.rotation_jets is fit.rotation_jets


def test_no_uncaptured_construction_or_recapture(monkeypatch):
    with pytest.raises(ValueError, match='already captured'):
        authority.capture(b'{}\n')
    monkeypatch.setattr(authority, '_LEASE', None)
    with pytest.raises(ValueError, match='lease required'):
        owner.MixedGraphOwner()


def test_invalid_coefficient_type_or_values_rejected_before_capture(monkeypatch):
    for name in ('SINC', 'COSC', 'LOG'):
        original = getattr(kernel, name)
        for changed in (list(original), (original[0]+.01, *original[1:])):
            with monkeypatch.context() as patch:
                patch.setattr(kernel, name, changed)
                with pytest.raises(ValueError, match='coefficient authority'):
                    owner.MixedGraphOwner()


def test_equal_tuple_replacement_poisoned(monkeypatch):
    graph = owner.MixedGraphOwner(); published = graph._published
    original = kernel.LOG
    monkeypatch.setattr(kernel, 'LOG', tuple(list(original)))
    with pytest.raises(ValueError, match='dispatch'):
        graph.snapshot_bytes()
    assert graph._published is published and not graph._owned_lock.locked()
    monkeypatch.setattr(kernel, 'LOG', original)
    with pytest.raises(ValueError, match='captured'):
        graph.snapshot_bytes()


def test_kernel_coefficient_helper_replacement_poisoned(monkeypatch):
    graph = owner.MixedGraphOwner(); published = graph._published
    original = kernel._log_factor
    monkeypatch.setattr(kernel, '_log_factor', lambda value: original(value))
    with pytest.raises(ValueError, match='dispatch'):
        graph.solve(load())
    assert graph._published is published and not graph._owned_lock.locked()
    monkeypatch.setattr(kernel, '_log_factor', original)
    with pytest.raises(ValueError, match='captured'):
        graph.snapshot_bytes()


def test_transitive_derivative_helper_replacements_poisoned(monkeypatch):
    for module, name in ((chart, 'exp_chart_terms'), (boundary, 'chart_pullback'),
                         (line_work, 'evaluate'), (fit, 'rotation_jets'), (compensated, 'sum_jets')):
        graph = owner.MixedGraphOwner(); published = graph._published
        original = getattr(module, name)
        with monkeypatch.context() as patch:
            patch.setattr(module, name, lambda *a, **kw: original(*a, **kw))
            with pytest.raises(ValueError, match='dispatch'):
                graph.snapshot_bytes()
            assert graph._published is published
        with pytest.raises(ValueError, match='captured'):
            graph.snapshot_bytes()


def test_leased_runtime_hash_mutation_rejected_without_writes():
    path, size, digest = authority._LEASE[1][0]
    with pytest.raises(ValueError, match='leased runtime source'):
        authority._verify_rows(((path, size, '0'*64),))


def test_foreign_predecessor_native_identity_rejected():
    from anyfesolver._ge_beam3_g1_element import ElasticElement as OldElement
    graph = owner.MixedGraphOwner()
    model, elements, states = graph._native_model(None)
    native = elements[0]
    old = OldElement(native.element_id, native.node_ids, native.operator.reference, native.section)
    assert native.identity != old.identity
    foreign = deepcopy(states[native.element_id])
    foreign['element_identity'] = old.identity
    with pytest.raises(ValueError, match='definition/history'):
        native._validate(foreign, np.zeros(18))
    assert element.SCHEMA == 'GE_BEAM3_G1_ELASTIC_STATE_V1'


def test_authoritative_native_rotations_not_reconstructed():
    graph = owner.MixedGraphOwner()
    graph.solve(load())
    state = json.loads(graph.snapshot_bytes())['state']
    for row in state['native_rows']:
        assert row['payload']['epoch'] == 1
        assert np.array_equal(row['payload']['committed_nodal_rotation_matrices'],
                              np.tile(np.eye(3), (3, 1, 1)))
    from anyfesolver import _native_rotation_state as original_rotation
    assert owner.rotation_module is original_rotation


def test_nonzero_curved_line_work_directional_derivatives():
    # A load-only analytic regression, not a new graph qualification fixture.
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    coordinates = np.array([[0., 0., 0.], [.5, .1, 0.], [1., 0., 0.]])
    frames = []
    for xi in (-1., 0., 1.):
        axis = np.array([.5, -.2*xi, 0.]); axis /= np.linalg.norm(axis)
        frames.append(np.column_stack((axis, np.cross([0., 0., 1.], axis), [0., 0., 1.])))
    reference = Reference(coordinates, np.array(frames))
    force = np.array([.13, -.21, .34])
    q = Rotation.from_rotvec([[.11, -.07, .03], [-.08, .04, .09]]).as_matrix()
    direction = np.zeros(42); direction[18:24] = [.3, -.2, .4, -.1, .2, .3]
    direction /= np.linalg.norm(direction)
    def work(h):
        made = Rotation.from_rotvec(h*direction[18:24].reshape(2, 3)).as_matrix() @ q
        return line_work.evaluate(reference, coordinates, np.zeros((3, 3)), made, force, order=4)
    centre = work(0.)
    assert np.linalg.norm(centre.gradient[18:24]) > 1e-5
    for h in (1e-4, 1e-5, 1e-6):
        plus, minus = work(h), work(-h)
        first = (plus.value-minus.value)/(2*h)
        second = ((plus.gradient-minus.gradient)/(2*h)) @ direction
        assert abs(first-centre.gradient@direction) <= 1e-7*max(1., abs(first))
        assert abs(second-direction@centre.hessian@direction) <= 1e-7*max(1., abs(second))
