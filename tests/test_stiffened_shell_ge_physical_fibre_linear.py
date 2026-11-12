"""Existing native physical fibres in the experimental reference-linear owner."""
from dataclasses import replace
import json

import anyfesolver
import numpy as np
import pytest

from anyfesolver import b3_ge
from anyfesolver.b3_ge import Fibre, FlowCurve, PhysicalFibreSection
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_experimental import (
    FINITE_CHECKPOINT, NodalAction,
)
from anyfesolver._stiffened_shell_ge_checkpoint_binary import (
    CHART_MAGIC, S3_MAGIC, encode,
)
from anyfesolver._stiffened_shell_l1_assembly_experimental import ExperimentalL1Line, SPACE_A
from test_stiffened_shell_ge_experimental import _panel


def _definition(mixed=False, triangles=False):
    base = _panel()
    e, g = 210e9, 210e9/2.6
    arm = np.sqrt(8.333333333333333e-6/.01)
    fibres = tuple(Fibre(str(i), y*arm, z*arm, .0025, e,
                         FlowCurve((0.,), (1e20,), 1e9))
                   for i, (y,z) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))))
    background = np.diag(np.sqrt((0., g*.008, g*.008,
                                  g*1.6666666666666667e-5, 0., 0.)))
    section = PhysicalFibreSection(fibres, background)
    beams = []
    for i, definition in enumerate(base.beams):
        element, inertia = definition.instantiate()
        ref = element.operator.reference
        beams.append(definition if mixed and i == 0 else b3_ge.define_beam(
            'b3-ge', element.element_id, element.node_ids, ref.coordinates,
            ref.nodal_triads, section, inertia, quadrature=element.operator.order))
    shell = base.shell
    if triangles:
        shell = replace(shell, q4_elements=(),
                        s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)))
    return replace(base, beams=tuple(beams), shell=shell)


@pytest.mark.parametrize('mixed', (False, True))
@pytest.mark.parametrize('triangles', (False, True))
def test_physical_fibre_linear_panel_matches_elastic_section_and_restarts(mixed, triangles):
    definition = _definition(mixed, triangles)
    baseline = replace(_panel(), shell=definition.shell)
    analysis = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    oracle = b3_ge.create_stiffened_shell_analysis(formulation='b3-ge', definition=baseline)
    assert analysis.capabilities['status'] == 'EXPERIMENTAL_NOT_QUALIFIED'
    assert analysis.capabilities['analysis'] == (
        'reference_linear', 'experimental_native_fibre_finite_point')
    assert 'PHYSICAL_AXIAL_BIAXIAL_FIBRE' in analysis.capabilities['beam_section_families']
    assert ('RESULTANT_ELLIPSOID' in analysis.capabilities['beam_section_families']) == mixed
    assert analysis.preflight()['retained_dofs'] == 48
    actions = (NodalAction('beam', 105, (10., 20., 100., 3., 4., 5.)),)
    result = analysis.solve_linear(actions)
    expected = oracle.solve_linear(actions)
    assert result.status == 'completed'
    assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
    np.testing.assert_allclose(analysis._K.toarray(), oracle._K.toarray(),
                               rtol=2e-12, atol=2e-8)
    for name in ('shell_displacement','beam_coordinates','multipliers','reactions'):
        np.testing.assert_allclose(getattr(result,name),getattr(expected,name),
                                   rtol=1e-8,atol=1e-10)
    # Independent global wrench balance, including eccentric beam coordinates.
    support = result.reactions[:analysis._shell_count].reshape(-1,6)
    force = support[:,:3].sum(axis=0)
    moment = support[:,3:].sum(axis=0)+np.cross(
        np.asarray(definition.shell.coordinates_m),support[:,:3]).sum(axis=0)
    np.testing.assert_allclose(force, -np.asarray(actions[0].wrench[:3]),atol=1e-6)
    applied_moment = np.asarray(actions[0].wrench[3:])+np.cross((2.,-.5,.1),actions[0].wrench[:3])
    np.testing.assert_allclose(moment,-applied_moment,atol=1e-5)
    saved = analysis.checkpoint()
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    restored = resumed.restore(saved)
    np.testing.assert_array_equal(restored.beam_coordinates,result.beam_coordinates)
    twice = tuple(replace(action,wrench=tuple(2*v for v in action.wrench)) for action in actions)
    continued = resumed.solve_linear(twice)
    np.testing.assert_allclose(continued.beam_coordinates,2*result.beam_coordinates,
                               rtol=1e-8,atol=1e-10)
    assert resumed.preflight()['reference_linear_factorizations'] == 1
    assert analysis.checkpoint() == saved
    for owner in (analysis,oracle,resumed):
        owner.close()


@pytest.mark.parametrize('kind', ('json','binary','chart','s3'))
def test_physical_fibre_legacy_finite_checkpoints_reject_without_state_change(kind):
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition(mixed=True))
    actions = (NodalAction('beam',105,(0.,0.,100.,0.,0.,0.)),)
    owner.solve_linear(actions)
    saved = owner.checkpoint()
    with pytest.raises(RuntimeError,match='states cannot be mixed'):
        owner.solve(actions)
    payload = json.loads(saved)
    payload['schema'] = FINITE_CHECKPOINT
    raw = (json.dumps(payload).encode() if kind == 'json' else
           encode('finite',owner._identity,1,(('q',np.asarray(payload['q'])),))
           if kind == 'binary' else CHART_MAGIC if kind == 'chart' else S3_MAGIC)
    fresh = anyfesolver.stiffened_shell.create_analysis(definition=_definition(mixed=True))
    with pytest.raises(ValueError,match='reference-linear checkpoints only|digest mismatch|checkpoint'):
        fresh.restore(raw)
    assert fresh._accepted is None and fresh._finite_driver is None
    assert owner.checkpoint() == saved
    fresh.close()
    owner.close()


def test_physical_fibre_line_admission_rejects_before_mapping():
    line = ExperimentalL1Line((1,2,3,4,5), ((0,0),(0,1),(1,0),(1,1)), SPACE_A)
    definition = replace(_definition(), lines=(line,))
    with pytest.raises(ValueError,match='point attachments only'):
        anyfesolver.stiffened_shell.create_analysis(definition=definition)


def test_physical_fibre_binary_restart_and_cancel_preserve_prefix(monkeypatch):
    from anyfesolver import _stiffened_shell_ge_experimental as owner_module
    monkeypatch.setattr(owner_module,'MAX_JSON_CHECKPOINT_VALUES',1)
    definition = _definition(mixed=True)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    actions = (NodalAction('beam',105,(0.,0.,100.,0.,0.,0.)),)
    result = owner.solve_linear(actions)
    saved = owner.checkpoint()
    fresh = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    restored = fresh.restore(saved)
    np.testing.assert_array_equal(restored.beam_coordinates,result.beam_coordinates)
    token = CancellationToken()
    token.cancel('physical-fibre prefix check')
    with pytest.raises(SolveCancelled,match='physical-fibre prefix check'):
        fresh.solve_linear(actions,cancellation_token=token)
    assert fresh.checkpoint() == saved
    assert fresh.preflight()['reference_linear_factorizations'] == 0
    owner.close()
    fresh.close()
