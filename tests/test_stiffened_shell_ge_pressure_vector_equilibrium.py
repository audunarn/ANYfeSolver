"""Opt-in pressure equilibrium: vector work, compatibility and durable identity."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver, _pressure_vector_equilibrium, _trial_work_ratio,
)
from anyfesolver._stiffened_shell_ge_pressure_load import (
    VECTOR_EQUILIBRIUM_POLICY, decode_checkpoint,
)
from test_stiffened_shell_ge_director_pressure import director_load
from test_stiffened_shell_ge_pressure_owner import balance, load, owner


def test_vector_policy_preserves_global_work_and_separates_node_force_and_moment():
    residual, applied = np.zeros(18), np.zeros(18)
    free = np.arange(18)
    scales = np.ones(18)
    applied[0], applied[2] = 1e-10, 1.
    residual[0] = 1e-10
    # A small component of the same loaded vector is deliberately masked.
    assert _pressure_vector_equilibrium(residual, applied, free, 12, scales) == pytest.approx(1e-10)
    applied[6], residual[6] = 1e-10, 1e-10
    assert _pressure_vector_equilibrium(residual, applied, free, 12, scales) == pytest.approx(1.)
    residual[6] = 0.
    applied[3], residual[3] = 2., 2e-8
    assert _pressure_vector_equilibrium(residual, applied, free, 12, scales) == pytest.approx(1e-8)
    residual[3] = 0.
    residual[15] = 1e-7  # Retained or beam row: global work cannot be omitted.
    assert _pressure_vector_equilibrium(residual, applied, free, 12, scales) > 1e-8
    residual[15] = 0.
    applied[6] = 0.
    residual[6] = 1e-7  # Unloaded node remains in the global work check.
    assert _pressure_vector_equilibrium(residual, applied, free, 12, scales) > 1e-8


def test_vector_policy_uses_only_free_components_and_refuses_bad_rows():
    applied = np.zeros(12)
    applied[0], applied[2] = 100., 1e-7
    residual = np.zeros(12)
    residual[2] = 1e-7
    free = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert _pressure_vector_equilibrium(residual, applied, free, 12, np.ones(12)) == 0.
    residual[0] = 1e-6
    assert _pressure_vector_equilibrium(residual, applied, free, 12, np.ones(12)) == pytest.approx(1e-8)
    assert _pressure_vector_equilibrium(residual*1e-5, applied*1e-5,
                                        free, 12, np.ones(12)) == pytest.approx(1e-8)
    for bad_free, shell_count in (([0, 0], 12), ([12], 12), (free, 10)):
        with pytest.raises(ValueError):
            _pressure_vector_equilibrium(residual, applied, bad_free, shell_count, np.ones(12))


def test_pressure_policy_identity_preserves_legacy_descriptor_and_checkpoint(request):
    a, b = owner(request, 's3'), owner(request, 's3')
    old = director_load(a)
    assert ss.GE_PRESSURE_VECTOR_EQUILIBRIUM_V1_DEV == VECTOR_EQUILIBRIUM_POLICY
    assert a.capabilities['shell_pressure_equilibrium_policies'] == (
        'component_relative_v1', VECTOR_EQUILIBRIUM_POLICY)
    new = replace(old, equilibrium_policy=VECTOR_EQUILIBRIUM_POLICY)
    assert old.equilibrium_policy is None
    assert 'equilibrium_policy' not in old.to_payload()
    assert new.identity != old.identity
    assert new.compile(a).augmented_model_identity != old.compile(a).augmented_model_identity
    assert ss.GEShellPressureLoad.deserialize(new.serialize()) == new
    assert decode_checkpoint(new.compile(a).checkpoint(b'inner'))[0] == new
    linear = a.solve_linear(new)
    b.restore(a.checkpoint())
    assert b.checkpoint() == a.checkpoint()
    assert linear.free_equilibrium_ratio <= 1e-8
    with pytest.raises(ValueError, match='policy'):
        ss.GEShellPressureLoad(a._identity, old.actions, equilibrium_policy='unknown')
    with pytest.raises(ValueError):
        ss.GEShellPressureLoad.from_payload(dict(new.to_payload(), equilibrium_policy=None))


def test_default_driver_keeps_component_assessment(request):
    a, b = owner(request, 'mixed'), owner(request, 'mixed')
    strict, selected = FinitePointDriver(a), FinitePointDriver(b)
    strict._prepare_load(director_load(a), (.1,))
    selected._prepare_load(replace(director_load(b),
        equilibrium_policy=VECTOR_EQUILIBRIUM_POLICY), (.1,))
    applied = np.zeros(a._K.shape[0])
    residual = np.zeros(a._K.shape[0]+a._C.shape[0])
    first = int(a._free[a._free < a._shell_count][0])
    applied[first], applied[first+2] = 1e-10, 1.
    residual[first] = 1e-10
    assert strict._equilibrium_ratio(residual, applied) == pytest.approx(1.)
    assert selected._equilibrium_ratio(residual, applied) == pytest.approx(1e-10)


def test_selected_frozen_trial_merit_detects_small_loaded_node(request):
    a, b = owner(request, 'mixed'), owner(request, 'mixed')
    strict, selected = FinitePointDriver(a), FinitePointDriver(b)
    strict._prepare_load(director_load(a), (.1,))
    selected._prepare_load(replace(director_load(b),
        equilibrium_policy=VECTOR_EQUILIBRIUM_POLICY), (.1,))
    applied = np.zeros(a._K.shape[0])
    residual = np.zeros(a._K.shape[0]+a._C.shape[0])
    nodes = np.unique(a._free[a._free < a._shell_count]//6)
    assert len(nodes) >= 2
    large, small = (6*int(node)+2 for node in nodes[:2])
    applied[large], applied[small] = 1e6, 1e-4
    residual[small] = 1e-4
    assert strict._frozen_trial_ratio(residual, applied, _trial_work_ratio) < 1e-8
    assert selected._frozen_trial_ratio(residual, applied, _trial_work_ratio) == pytest.approx(1.)


@pytest.mark.parametrize('kind,mode', [('q4', 'current_surface'),
                                       ('mixed', 'director'), ('s3', 'director'),
                                       ('mixed', 'current_surface')])
def test_opt_in_pressure_cancel_restore_rebase(request, kind, mode):
    a, b = owner(request, kind), owner(request, kind)
    base = director_load(a) if mode == 'director' else load(a, mode)
    action = replace(base, equilibrium_policy=VECTOR_EQUILIBRIUM_POLICY)
    token = CancellationToken()
    control = ss.AdaptiveIncrementControl(2, 16)
    with pytest.raises(SolveCancelled):
        a.solve(action, (.1, .2), increment_control=control,
                on_accepted=lambda _: token.cancel('complete prefix'), cancellation_token=token)
    first = a.recover()
    assert first.accepted_factors == (.1,)
    assert first.free_equilibrium_ratio <= 1e-8 and first.constraint_max <= 1e-9
    balance(a, first)
    saved = a.checkpoint()
    b.restore(saved)
    with pytest.raises(ValueError):
        b.solve(director_load(b), (.1, .2), increment_control=control)
    assert b.checkpoint() == saved
    direct = a.solve(action, (.1, .2), increment_control=control)
    resumed = b.solve(action, (.1, .2), increment_control=control)
    np.testing.assert_array_equal(direct.beam_positions, resumed.beam_positions)
    assert a.checkpoint() == b.checkpoint()
    assert direct.free_equilibrium_ratio <= 1e-8 and direct.constraint_max <= 1e-9
    balance(a, direct)
    a.rebase_rotations(direct.generation)
    c = owner(request, kind)
    c.restore(a.checkpoint())
    final = a.solve(action, (.3,), max_iterations=15)
    restarted = c.solve(action, (.3,), max_iterations=15)
    np.testing.assert_array_equal(final.beam_positions, restarted.beam_positions)
    balance(a, final)
