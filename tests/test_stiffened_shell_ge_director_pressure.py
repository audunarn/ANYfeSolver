"""Explicit experimental director-surface pressure through the mixed owner."""
from dataclasses import replace
import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_ge_pressure_load import (
    DIRECTOR_CONFIGURATION, DIRECTOR_LOAD_POLICY, POLICY, VECTOR_EQUILIBRIUM_POLICY, decode_checkpoint,
)
from test_stiffened_shell_ge_pressure_owner import owner, balance, load


def director_load(a):
    return ss.GEShellPressureLoad(a._identity, tuple(
        ss.GEShellPressureAction(eid, 100., DIRECTOR_CONFIGURATION if len(slots) == 18 else 'reference_dead')
        for eid, _, slots, _ in a._shell_parts))


@pytest.mark.parametrize('kind', ['mixed', 's3'])
def test_distinct_identity_reference_parity_and_linear_restart(request, kind):
    a, b = owner(request, kind), owner(request, kind)
    new, old = director_load(a), load(a, 'reference_dead')
    assert new.policy == DIRECTOR_LOAD_POLICY and old.policy == POLICY
    assert new.identity != old.identity
    np.testing.assert_allclose(new.compile(a).reference_gradient, old.compile(a).reference_gradient,
                               rtol=1e-12, atol=1e-13)
    result = a.solve_linear(new)
    assert result.status == 'completed'
    assert result.pressure_load.moment_scope == 'spatial_director_enhanced_surface'
    restored = b.restore(a.checkpoint())
    np.testing.assert_array_equal(result.beam_coordinates, restored.beam_coordinates)
    assert decode_checkpoint(a.checkpoint())[0] == new


@pytest.mark.parametrize('kind', ['mixed', 's3'])
def test_actual_owner_external_work_tangent_and_anchor_rebase(request, kind):
    a = owner(request, kind)
    driver = FinitePointDriver(a)
    pattern, _ = driver._prepare_load(director_load(a), (.1,))
    q = driver._state.q.copy()
    q[:a._shell_count] = np.linspace(-.015, .02, a._shell_count)
    state = replace(driver._state, q=q)
    force, jac, recovery = driver._external.evaluate(state)
    assert recovery.potential is not None
    # Amplify only this algebraic factor-linearity measurement so subtraction
    # of large internal stiffness does not erase the small external Jacobian.
    factor = 1e6
    r1, k1, *_ = driver._assemble(state, factor, driver._shell_origins, driver._beam_origins, pattern=pattern)
    r0, k0, *_ = driver._assemble(state, 0., driver._shell_origins, driver._beam_origins, pattern=pattern)
    n = len(force)
    np.testing.assert_allclose((r1[:n]-r0[:n])/factor, -force, rtol=1e-7, atol=1e-10)
    first, second = k1[:n, :n].toarray(), k0[:n, :n].toarray()
    np.testing.assert_allclose((first-second)/factor, -jac.toarray(), rtol=1e-6, atol=1e-10)
    direction = np.zeros(len(r1))
    direction[:a._shell_count] = np.linspace(-.3, .4, a._shell_count)
    expected = jac@direction[:n]
    for h in (1e-4, 3e-5, 1e-5):
        plus = driver._external.evaluate(driver._advance(state, direction, h))
        minus = driver._external.evaluate(driver._advance(state, direction, -h))
        assert np.linalg.norm((plus[0]-minus[0])/(2*h)-expected)/np.linalg.norm(expected) < 1e-6


@pytest.mark.parametrize('kind', ['mixed', 's3'])
def test_director_pressure_finite_cancel_restore_continue(request, kind):
    a, b = owner(request, kind), owner(request, kind)
    # Finite S3 pressure has tiny Cartesian components after rotation. This
    # lifecycle exercises the separately selected, checkpoint-bound nodal
    # force/moment-vector policy; the default component contract stays strict.
    action = replace(director_load(a), equilibrium_policy=VECTOR_EQUILIBRIUM_POLICY)
    token = CancellationToken()
    control = ss.AdaptiveIncrementControl(2, 16)
    with pytest.raises(SolveCancelled):
        a.solve(action, (.1, .2), increment_control=control,
                on_accepted=lambda _: token.cancel('complete prefix'), cancellation_token=token)
    first = a.recover()
    assert first.accepted_factors == (.1,)
    balance(a, first)
    b.restore(a.checkpoint())
    direct = a.solve(action, (.1, .2), increment_control=control)
    resumed = b.solve(action, (.1, .2), increment_control=control)
    np.testing.assert_array_equal(direct.beam_positions, resumed.beam_positions)
    assert a.checkpoint() == b.checkpoint()
    assert direct.pressure_load.potential is not None
    balance(a, direct)
    a.rebase_rotations(direct.generation)
    c = owner(request, kind)
    c.restore(a.checkpoint())
    final = a.solve(action, (.3,), max_iterations=15)
    restored = c.solve(action, (.3,), max_iterations=15)
    np.testing.assert_array_equal(final.beam_positions, restored.beam_positions)
    balance(a, final)


def test_q4_rejects_s3_specific_surface(request):
    a = owner(request, 'q4')
    with pytest.raises(ValueError, match='requires native S3'):
        load(a, DIRECTOR_CONFIGURATION).compile(a)


def test_policy_substitution_and_mixed_surface_reporting(request):
    from hashlib import sha256
    from anyfesolver._stiffened_shell_ge_pressure_load import MAGIC, _canonical, _json
    a = owner(request, 's3')
    action = director_load(a)
    payload = action.to_payload()
    payload['policy'] = POLICY
    with pytest.raises(ValueError):
        ss.GEShellPressureLoad.from_payload(payload)
    raw = action.compile(a).checkpoint(b'opaque inner')
    start = len(MAGIC)+4
    size = int.from_bytes(raw[len(MAGIC):start], 'little')
    header = _json(raw[start:start+size])
    header['policy'] = POLICY
    encoded = _canonical(header)
    content = MAGIC+len(encoded).to_bytes(4, 'little')+encoded+raw[start+size:-32]
    with pytest.raises(ValueError, match='policy mismatch'):
        decode_checkpoint(content+sha256(content).digest())
    rows = list(action.actions)
    rows[0] = replace(rows[0], configuration='current_surface')
    combined = ss.GEShellPressureLoad(a._identity, tuple(rows)).compile(a)
    assert combined.reference_recovery.potential is None
    assert combined.reference_recovery.moment_scope == 'spatial_mixed_pressure_surfaces'
    rows[0] = replace(rows[0], configuration='reference_dead')
    assert not ss.GEShellPressureLoad(a._identity, tuple(rows)).compile(a).finite_supported
