"""Local existing-kernel parity and pressure context lifecycle; no material solve."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.boundary import LoadCase
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._stiffened_shell_ge_experimental import ShellPanel, _shell_reference
from anyfesolver._stiffened_shell_ge_pressure_load import (
    GEShellPressureAction as Action, GEShellPressureLoad as Request,
    decode_checkpoint,
)


def owner(triangle=False):
    xyz = ((0., 0., 0.), (2., 0., 0.), (1.8, 1., 0.), (.1, 1., 0.))
    nodes = (1, 2, 3) if triangle else (1, 2, 3, 4)
    panel = ShellPanel(nodes, xyz[:len(nodes)], () if triangle else (nodes,),
                       .1, 2.e9, .3, 3., (nodes,) if triangle else ())
    model, _, captured, _, parts = _shell_reference(panel)
    count = 6*len(nodes)
    return SimpleNamespace(_identity='a'*64, _require_open=lambda: None,
        _definition=SimpleNamespace(lines=(), footprints=(), points=(object(),),
            distributing_footprints=(), finite_distributing_footprints=()),
        _line_samples=(), _shell_model=model, _shell_xyz=np.array(captured),
        _shell_parts=tuple(parts), _shell_count=count,
        _K=sparse.csr_matrix((count+6, count+6)))


def context(model, configuration='reference_dead', pressure=2.5):
    return Request(model._identity, (Action(1, pressure, configuration),)).compile(model)


def changed_state(compiled):
    q = np.zeros(compiled.count)
    slots = list(compiled.patches[0].slots)
    n = len(slots)//6
    changes = np.array([[.03, -.02, .08], [-.04, .03, .12], [.06, .02, -.07], [-.02, -.03, .04]])[:n]
    local = np.zeros((n, 6))
    local[:, :3] = changes
    local[:, 3:] = [.1, -.2, .3]
    q[slots] = local.ravel()
    return SimpleNamespace(q=q)


@pytest.mark.parametrize('triangle', [False, True])
@pytest.mark.parametrize('configuration', ['reference_dead', 'current_surface'])
def test_local_native_vector_tangent_and_physical_resultant_parity(triangle, configuration):
    model = owner(triangle)
    compiled = context(model, configuration)
    state = changed_state(compiled)
    force, tangent, recovered = compiled.evaluate(state)
    _, element, slots, _ = model._shell_parts[0]
    xyz = model._shell_xyz+state.q[slots].reshape(-1, 6)[:, :3]
    load = LoadCase('oracle')
    coords = xyz if configuration == 'current_surface' else None
    expected = load._consistent_pressure_load(element, model._shell_model.mesh, 2.5, coords)
    np.testing.assert_array_equal(force[slots], expected)
    local_tangent = (load._consistent_pressure_tangent(element, model._shell_model.mesh, 2.5, xyz)
                     if configuration == 'current_surface' else np.zeros((len(slots), len(slots))))
    np.testing.assert_array_equal(tangent[slots, :][:, slots].toarray(), local_tangent)
    assert sparse.isspmatrix_csr(tangent)
    assert not np.any(force[model._shell_count:])
    nodal = expected.reshape(-1, 6)
    np.testing.assert_allclose(recovered.applied_force, nodal[:, :3].sum(axis=0), atol=1e-14)
    np.testing.assert_allclose(recovered.applied_moment_about_origin,
                               (np.cross(xyz, nodal[:, :3])+nodal[:, 3:]).sum(axis=0), atol=1e-14)
    assert recovered.element_ids == (1,)
    assert compiled.finite_supported == (not triangle or configuration == 'current_surface')
    if triangle and configuration == 'reference_dead':
        assert np.any(nodal[:, 3:] != 0.)
        assert 'S3' in compiled.finite_error and 'reference' in compiled.finite_error
        assert recovered.moment_scope == 'reference_linear_covectors'
    if configuration == 'current_surface' or triangle:
        assert recovered.potential is None
    else:
        assert recovered.potential == pytest.approx(expected@state.q[slots])
    scaled = recovered.scaled(-.5)
    np.testing.assert_allclose(scaled.applied_force, -.5*np.array(recovered.applied_force))
    np.testing.assert_allclose(scaled.applied_moment_about_origin, -.5*np.array(recovered.applied_moment_about_origin))


def test_s3_native_reference_covectors_are_not_current_surface_at_reference():
    model = owner(True)
    dead = context(model)
    follower = context(model, 'current_surface')
    dead_nodes = dead.reference_gradient[:18].reshape(3, 6)
    current_nodes = follower.reference_gradient[:18].reshape(3, 6)
    assert np.linalg.norm(dead_nodes[:, 3:]) > 0.
    assert not np.any(current_nodes[:, 3:])
    # Independent affine triangle pressure formula: p*(edge1 x edge2)/6 per node.
    xyz = model._shell_xyz
    expected = 2.5*np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])/6
    np.testing.assert_allclose(current_nodes[:, :3], np.tile(expected, (3, 1)), rtol=3e-15, atol=1e-14)


@pytest.mark.parametrize('triangle', [False, True])
def test_actual_translation_update_derivative_nonsymmetry_and_rigid_covariance(triangle):
    model = owner(triangle)
    compiled = context(model, 'current_surface', -2.5)
    state = changed_state(compiled)
    force, tangent, recovered = compiled.evaluate(state)
    assert sparse.linalg.norm(tangent-tangent.T) > .1
    step = 3.e-6
    slots = list(compiled.patches[0].slots)
    for slot in slots:
        plus, minus = state.q.copy(), state.q.copy()
        plus[slot] += step
        minus[slot] -= step
        fp = compiled.evaluate(SimpleNamespace(q=plus))[0]
        fm = compiled.evaluate(SimpleNamespace(q=minus))[0]
        np.testing.assert_allclose((fp-fm)/(2*step), tangent[:, slot].toarray().ravel(), rtol=2e-9, atol=2e-10)
    qrot = rotation([.3, -.2, .15])
    shift = np.array([3., -2., 4.])
    positions = model._shell_xyz+state.q[slots].reshape(-1, 6)[:, :3]
    moved = state.q.copy()
    local = moved[slots].reshape(-1, 6)
    local[:, :3] = positions@qrot.T+shift-model._shell_xyz
    moved[slots] = local.ravel()
    moved_force, _, moved_recovery = compiled.evaluate(SimpleNamespace(q=moved))
    np.testing.assert_allclose(moved_force[slots].reshape(-1, 6)[:, :3],
                               force[slots].reshape(-1, 6)[:, :3]@qrot.T, atol=1e-14)
    np.testing.assert_allclose(moved_recovery.applied_force, qrot@recovered.applied_force, atol=2e-14)
    np.testing.assert_allclose(moved_recovery.applied_moment_about_origin,
                               qrot@recovered.applied_moment_about_origin+np.cross(shift, qrot@recovered.applied_force), atol=5e-14)


@pytest.mark.parametrize('triangle', [False, True])
def test_dead_vector_is_captured_once_and_arrays_are_owned(monkeypatch, triangle):
    model = owner(triangle)
    compiled = context(model)
    expected = compiled.reference_gradient.copy()
    monkeypatch.setattr(LoadCase, '_consistent_pressure_load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('recomputed dead force')))
    model._shell_xyz[:] = 100.
    model._shell_parts[0][2][:] = 0
    actual, _, _ = compiled.evaluate(SimpleNamespace(q=np.zeros(compiled.count)))
    np.testing.assert_array_equal(actual, expected)
    for array in (actual, compiled.reference_gradient, compiled.patches[0].coordinates, compiled.patches[0].dead_force):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize('triangle', [False, True])
def test_current_collapsed_surface_and_changed_native_binding_reject(triangle):
    model = owner(triangle)
    compiled = context(model, 'current_surface')
    q = np.zeros(compiled.count)
    q[:model._shell_count].reshape(-1, 6)[:, :3] = -model._shell_xyz
    with pytest.raises(ValueError, match='degenerate'):
        compiled.evaluate(SimpleNamespace(q=q))
    if not triangle:
        q[:model._shell_count].reshape(-1, 6)[:, :3] = model._shell_xyz*1e-7-model._shell_xyz
        with pytest.raises(ValueError, match='degenerate'):
            compiled.evaluate(SimpleNamespace(q=q))
    model._shell_model.mesh.nodes[2].x += .1
    with pytest.raises(ValueError, match='binding changed'):
        compiled.evaluate(SimpleNamespace(q=np.zeros(compiled.count)))


def test_descriptor_canonical_ownership_strict_admission_and_checkpoint():
    request = Request('A'*64, (Action(2, -1., 'current_surface'), Action(1, 2.)))
    assert request.actions[0].element_id == 1
    assert Request.deserialize(request.serialize()) == request
    assert request.model_identity == 'a'*64
    with pytest.raises(FrozenInstanceError):
        request.actions = ()
    payload = request.to_payload()
    payload['actions'][0]['pressure_pa'] = 9.
    with pytest.raises(ValueError):
        Request.from_payload(payload)
    with pytest.raises(ValueError):
        Request.deserialize(b' '+request.serialize())
    for pressure in (0., True, np.nan, np.inf, 1j):
        with pytest.raises(ValueError):
            Action(1, pressure)
    for eid in (0, -1, True, 1.):
        with pytest.raises(ValueError):
            Action(eid, 1.)
    for configuration in ('follower', '', None):
        with pytest.raises(ValueError):
            Action(1, 1., configuration)
    with pytest.raises(ValueError, match='duplicate'):
        Request('a'*64, (Action(1, 1.), Action(1, 2., 'current_surface')))
    with pytest.raises(ValueError):
        Request('a'*64, [Action(1, 1.)])
    model = owner()
    compiled = context(model)
    raw = compiled.checkpoint(b'opaque-inner-checkpoint')
    assert decode_checkpoint(raw) == (compiled.request, b'opaque-inner-checkpoint')
    for bad in (raw[:-1], raw+b'x', raw[:25]+bytes([raw[25]^1])+raw[26:]):
        with pytest.raises(ValueError):
            decode_checkpoint(bad)
    with pytest.raises(ValueError, match='different model'):
        Request('b'*64, (Action(1, 1.),)).compile(model)
    with pytest.raises(ValueError, match='unknown shell'):
        Request(model._identity, (Action(9, 1.),)).compile(model)
    model._definition.lines = (object(),)
    with pytest.raises(ValueError, match='point-owner'):
        compiled.request.compile(model)


def test_context_rejects_nonfinite_state_and_reference_mapping_mismatch():
    model = owner()
    compiled = context(model, 'current_surface')
    for q in (np.full(compiled.count, np.nan), np.zeros(compiled.count, dtype=complex), np.zeros(1)):
        with pytest.raises(ValueError):
            compiled.evaluate(SimpleNamespace(q=q))
    model._shell_xyz[1, 0] += .1
    with pytest.raises(ValueError, match='mapping mismatch'):
        compiled.request.compile(model)


@pytest.mark.parametrize('triangle', [False, True])
@pytest.mark.parametrize('mutation', ['delete', 'replace'])
def test_native_element_registry_identity_is_required_at_compile_and_evaluation(triangle, mutation):
    model = owner(triangle)
    compiled = context(model, 'current_surface')
    registry = model._shell_model.mesh.elements
    if mutation == 'delete':
        del registry[1]
    else:
        other = owner(triangle)
        registry[1] = other._shell_parts[0][1]
    with pytest.raises(ValueError, match='registry binding changed'):
        compiled.evaluate(SimpleNamespace(q=np.zeros(compiled.count)))
    with pytest.raises(ValueError, match='registry binding changed'):
        compiled.request.compile(model)
