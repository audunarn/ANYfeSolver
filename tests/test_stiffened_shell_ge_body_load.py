"""Small body descriptor, compiled work and envelope lifecycle checks."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver._stiffened_shell_ge_body_load import GEBodyLoad, encode_checkpoint, decode_checkpoint
from test_stiffened_shell_native_gravity import reference, inertia


def owner(triangle=False):
    # Vertical shell explicitly exercises frame-aware Q4 mass admission.
    xyz = np.array([[0., 0., 0.], [0., 2., 0.], [0., 2., 1.], [0., 0., 1.]])
    if triangle:
        xyz = xyz[:3]
    shell_count = 6*len(xyz)
    beam = SimpleNamespace(operator=SimpleNamespace(reference=reference(True)))
    panel = SimpleNamespace(density_kg_m3=3., thickness_m=.2)
    return SimpleNamespace(_identity='a'*64, _require_open=lambda: None,
        _definition=SimpleNamespace(shell=panel, lines=(), footprints=(), distributing_footprints=(), finite_distributing_footprints=(), points=(object(),)),
        _line_samples=(), _K=sparse.csr_matrix((shell_count+42, shell_count+42)),
        _shell_xyz=xyz, _shell_count=shell_count,
        _shell_parts=((1, None, np.arange(shell_count), None),),
        _beam_parts=((beam, np.arange(42), None),),
        _reference_modal_inertias=(inertia(offset=(.1, .2, -.15)),))


def test_descriptor_ownership_canonical_and_strict_values():
    source = [0., -9.81, 0.]
    request = GEBodyLoad('A'*64, source)
    source[1] = 0.
    assert request.acceleration_m_s2 == (0., -9.81, 0.)
    assert GEBodyLoad.deserialize(request.serialize()) == request
    with pytest.raises(FrozenInstanceError):
        request.model_identity = 'b'*64
    for value in ((0., 0., 0.), (True, 0., 0.), (0., np.inf, 0.), (1., 2.), (1j, 0., 0.)):
        with pytest.raises(ValueError):
            GEBodyLoad('a'*64, value)
    payload = request.to_payload()
    payload['acceleration_m_s2'][1] = -10.
    with pytest.raises(ValueError):
        GEBodyLoad.from_payload(payload)
    with pytest.raises(ValueError):
        GEBodyLoad.deserialize(b' '+request.serialize())


@pytest.mark.parametrize('triangle', (False, True))
def test_compiled_reference_work_ownership_and_scaling(triangle):
    model = owner(triangle)
    request = GEBodyLoad(model._identity, (1., -2., 3.))
    context = request.compile(model)
    beam_mass = sum(context.beams[0][0].nodal_masses)
    shell_mass = .6 if triangle else 1.2
    assert context.total_mass == pytest.approx(beam_mass+shell_mass)
    np.testing.assert_allclose(context.reference_recovery.applied_force,
                               context.total_mass*np.array(request.acceleration_m_s2))
    assert context.reference_recovery.work == 0.
    assert context.reference_gradient.shape == (model._K.shape[0],)
    assert np.linalg.norm(context.reference_gradient[model._shell_count+18:model._shell_count+24]) > 0.
    assert not context.shell_xyz.flags.writeable
    with pytest.raises(ValueError):
        context.reference_gradient.setflags(write=True)
    before = context.reference_gradient.copy()
    model._shell_xyz[:] = 100.
    state = SimpleNamespace(q=np.zeros(context.count), cells=np.tile(np.eye(3), (1, 2, 1, 1)))
    force, tangent, recovery = context.evaluate(state)
    np.testing.assert_array_equal(force, before)
    assert sparse.isspmatrix_csr(tangent)
    assert tangent.nnz <= 18
    scaled = recovery.scaled(.5)
    assert scaled.total_mass == recovery.total_mass and scaled.identity == recovery.identity
    np.testing.assert_allclose(scaled.applied_force, .5*np.array(recovery.applied_force))
    state.q[:model._shell_count].reshape(-1, 6)[:, :3] = (.1, .2, -.3)
    _, _, moved = context.evaluate(state)
    assert moved.work == pytest.approx(shell_mass*np.dot(request.acceleration_m_s2, (.1, .2, -.3)))


def test_envelope_roundtrip_and_corruption():
    model = owner()
    request = GEBodyLoad(model._identity, (0., 0., -9.81))
    context = request.compile(model)
    raw = encode_checkpoint(context, b'opaque-existing-checkpoint')
    assert decode_checkpoint(raw) == (request, b'opaque-existing-checkpoint')
    for bad in (raw[:-1], raw+b'x', raw[:25]+bytes([raw[25]^1])+raw[26:], b''):
        with pytest.raises(ValueError):
            decode_checkpoint(bad)


def test_admission_rejects_foreign_unsupported_or_nonphysical_inputs():
    model = owner()
    with pytest.raises(ValueError, match='different model'):
        GEBodyLoad('b'*64, (0., -1., 0.)).compile(model)
    request = GEBodyLoad(model._identity, (0., -1., 0.))
    model._definition.lines = (object(),)
    with pytest.raises(ValueError, match='point owners'):
        request.compile(model)
    model._definition.lines = ()
    model._reference_modal_inertias[0][0, 0] *= 2.
    with pytest.raises(ValueError):
        request.compile(model)
