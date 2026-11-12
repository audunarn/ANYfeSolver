"""Direct generalized/coupled cyclic displacement controls; unqualified.

These two layouts use the existing axial cyclic law and immutable load pattern.
The small unload/reload does not establish real-zero-load or signed plastic
reversal acceptance; those require their own bounded physical oracle.
"""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_displacement_checkpoint import (
    decode_snapshot, identities, DISTRIBUTING_KIND, DISTRIBUTING_POLICY,
)
from test_stiffened_shell_distributing_cyclic import definition, actions
from test_stiffened_shell_ge_distributing_material import decoded
from test_stiffened_shell_ge_distributing_displacement import _assert_physical


CONTROL = ss.DisplacementControl("beam", 103, 0)
KW = dict(displacement_control=CONTROL, max_iterations=24, max_backtracks=12)


@pytest.fixture(scope="module", params=(False, True), ids=("generalized", "coupled"))
def cyclic_seed(request):
    model = definition(coupled=request.param)
    owner = ss.create_analysis(definition=model)
    try:
        accepted = owner.solve(actions(), (.25, .5, .75, 1.),
                               max_iterations=24, max_backtracks=16)
        assert np.max(accepted.beam_station_accumulated) > 0.
        if request.param:
            assert np.max(accepted.shell_layer_alpha) > 0.
        raw = owner.checkpoint()
        force_kind, _, arrays = decoded(raw, owner._identity)
        initial = float(owner._finite_driver._state.q[
            owner._shell_count+6*owner._beam_index[103]])
        assert np.isfinite(initial) and initial > 0.
        return model, raw, initial, arrays, force_kind, request.param
    finally:
        owner.close()


def _endpoint(owner, result, *, coupled, previous):
    _assert_physical(owner, result)
    assert result.beam_generalized_material
    assert np.max(result.beam_station_accumulated) > 0.
    assert not result.beam_fibre_material
    if coupled:
        assert result.shell_q4_material and np.max(result.shell_layer_alpha) > 0.
    raw = owner.checkpoint()
    metadata, arrays = decode_snapshot(owner, raw)
    assert identities(owner)[:2] == (DISTRIBUTING_KIND, DISTRIBUTING_POLICY)
    assert metadata["policy"] == DISTRIBUTING_POLICY
    assert owner.capabilities["checkpoints"][-1] == DISTRIBUTING_KIND
    assert arrays["beam_plastic"].shape == previous["beam_plastic"].shape
    pairs = [("beam_plastic", "beam_origin_plastic"),
             ("beam_accumulated", "beam_origin_accumulated")]
    if coupled:
        pairs += [("shell_plastic", "shell_origin_plastic"),
                  ("shell_alpha", "shell_origin_alpha")]
    else:
        assert not any(name.startswith("shell_") and "plastic" in name for name in arrays)
        assert "shell_alpha" not in arrays and "shell_origin_alpha" not in arrays
    for accepted, predecessor in pairs:
        np.testing.assert_array_equal(arrays[predecessor], previous[accepted])
        assert not np.shares_memory(arrays[predecessor], previous[accepted])
    # The last axis is a compensated(high,low) scalar, not two material channels.
    # Reconstruct every station independently in the constitutive80-digit context.
    assert arrays["beam_accumulated"].shape == previous["beam_accumulated"].shape
    assert arrays["beam_accumulated"].shape[-1] == 2
    with localcontext() as context:
        context.prec = 80
        for current, parent in zip(arrays["beam_accumulated"], previous["beam_accumulated"]):
            accumulated = Decimal.from_float(float(current[0]))+Decimal.from_float(float(current[1]))
            predecessor = Decimal.from_float(float(parent[0]))+Decimal.from_float(float(parent[1]))
            assert accumulated >= predecessor
    if coupled:
        assert np.all(arrays["shell_alpha"] >= previous["shell_alpha"])
    np.testing.assert_array_equal(arrays["cyclic_seed_factor"], [1.])
    np.testing.assert_array_equal(arrays["base_load"], owner._cyclic_reference_pattern)
    np.testing.assert_array_equal(arrays["load_direction"], owner._cyclic_reference_pattern)
    np.testing.assert_array_equal(result.applied_dead_load,
        arrays["base_load"]+result.amplitude*arrays["load_direction"])
    for value in arrays.values():
        with pytest.raises(ValueError):
            value.setflags(write=True)
    return raw, arrays


def test_cyclic_generalized_layout_unload_restart_continue(cyclic_seed):
    model, seed_raw, initial, seed_arrays, force_kind, coupled = cyclic_seed
    owner = ss.create_analysis(definition=model)
    restored = ss.create_analysis(definition=model)
    targets = (initial*1.01, initial*.99, initial*1.02)
    try:
        owner.restore(seed_raw)
        assert force_kind != DISTRIBUTING_KIND
        first = owner.solve(actions(), targets[:1], **KW)
        first_raw, first_arrays = _endpoint(owner, first, coupled=coupled, previous=seed_arrays)
        first_saved = {name: value.copy() for name, value in first_arrays.items()}
        unloaded = owner.solve(actions(), targets[1:2], **KW)
        raw, unload_arrays = _endpoint(owner, unloaded, coupled=coupled, previous=first_arrays)
        assert unloaded.target < first.target
        assert unloaded.amplitude < first.amplitude
        # Continuing must not mutate the previously detached accepted arrays.
        for name, value in first_saved.items():
            np.testing.assert_array_equal(first_arrays[name], value)
        assert raw != first_raw
        replay = restored.restore(raw)
        _assert_physical(restored, replay)
        assert restored.checkpoint() == raw
        _, replay_arrays = decode_snapshot(restored, restored.checkpoint())
        assert set(replay_arrays) == set(unload_arrays)
        for name, value in unload_arrays.items():
            np.testing.assert_array_equal(replay_arrays[name], value)
            assert not np.shares_memory(replay_arrays[name], value)
        a = owner.solve(actions(), targets[2:], **KW)
        b = restored.solve(actions(), targets[2:], **KW)
        _, continued = _endpoint(owner, a, coupled=coupled, previous=unload_arrays)
        _, replayed = _endpoint(restored, b, coupled=coupled, previous=replay_arrays)
        np.testing.assert_array_equal(continued["history"][:, 0], targets)
        for name, value in continued.items():
            np.testing.assert_array_equal(replayed[name], value)
        for name in ("shell_displacement", "cell_resultants", "support_reactions",
                     "applied_dead_load", "beam_station_accumulated", "shell_layer_alpha"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        np.testing.assert_array_equal(a.distributing_footprint_wrenches[0].shell_action,
                                      b.distributing_footprint_wrenches[0].shell_action)
    finally:
        owner.close()
        restored.close()
