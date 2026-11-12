"""Actual fixed material cycles and model-bound zero-load restart."""
from dataclasses import replace
from math import fsum

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_increment_control import encode_control, _AdaptiveState
from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode, encode
from test_stiffened_shell_ge_distributing_material import definition as base_definition, actions

HISTORY = (.25, .5, .75, 1., .75, .5, .25, 0., -.25, -.5, -.75, -1., -.5, 0., .5, 1.)


def definition(coupled=True):
    return replace(base_definition(coupled), cyclic_reference_actions=actions(),
        finite_distributing_material_policy=(ss.GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1
            if coupled else ss.GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1))


@pytest.mark.parametrize("coupled", (False, True), ids=("generalized", "coupled"))
def test_actual_load_unload_reverse_and_zero_checkpoint_continue(coupled):
    owner = ss.create_analysis(definition=definition(coupled))
    resumed = ss.create_analysis(definition=definition(coupled))
    continued = ss.create_analysis(definition=definition(coupled))
    histories, zeros = [], []
    try:
        def observe(result):
            beam = np.asarray(result.beam_station_accumulated).sum(axis=-1)
            shell = np.asarray(result.shell_layer_alpha)
            stations = tuple(station for member in result.beam_generalized_material for station in member.stations)
            assert all(fsum(station.dissipation_per_length) >= 0. for station in stations)
            assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
            if histories:
                assert np.all(beam >= histories[-1][1])
                assert np.all(shell >= histories[-1][2])
            histories.append((result.load_factor, beam.copy(), shell.copy(),
                np.mean([station.plastic_strain[0]+station.plastic_strain_low[0] for station in stations]),
                np.mean([station.local_resultants[0]+station.local_resultants_low[0] for station in stations])))
            if result.load_factor == 0.:
                zeros.append(owner.checkpoint())
        final = owner.solve(actions(), HISTORY, max_iterations=24, max_backtracks=16, on_accepted=observe)
        assert final.accepted_factors == HISTORY and len(histories) == len(HISTORY)
        # Isotropic hardening does not reverse plastic strain merely because
        # force reverses within its expanded yield surface.
        assert histories[3][3] > 0.
        assert histories[3][4] > 0. and histories[11][4] < 0.
        assert np.max(final.beam_station_accumulated) > 0.
        if coupled:
            assert np.max(final.shell_layer_alpha) > 0.
        restored = resumed.restore(zeros[0])
        assert restored.load_factor == 0. and restored.accepted_factors == HISTORY[:8]
        assert resumed.checkpoint() == zeros[0]
        def shapes(kind, generation):
            driver = resumed._finite_driver
            raw = driver.checkpoint()
            # Reuse this bounded fixture's validated metadata only to recover
            # its exact arrays for canonical tampering, not production decode.
            import json
            from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC
            start = len(MAGIC)+4
            size = int.from_bytes(raw[len(MAGIC):start], "little")
            header = json.loads(raw[start:start+size])
            return tuple((name, tuple(shape)) for name, shape in header["arrays"])
        kind, generation, arrays = decode(zeros[0], resumed._identity, shapes)
        changed = dict(arrays)
        changed["pattern"] = 2.*changed["pattern"]
        corrupted = encode(kind, resumed._identity, generation, tuple(changed.items()))
        untouched = ss.create_analysis(definition=definition(coupled))
        try:
            with pytest.raises(ValueError, match="model-bound reference"):
                untouched.restore(corrupted)
            assert untouched._generation == 0 and untouched._finite_driver is None
        finally: untouched.close()
        token = CancellationToken()
        def stop(result):
            if result.load_factor == -.5:
                token.cancel("complete negative-load accepted prefix")
        with pytest.raises(SolveCancelled):
            resumed.solve(actions(), HISTORY[8:], max_iterations=24, max_backtracks=16,
                cancellation_token=token, on_accepted=stop)
        cancelled = resumed.checkpoint()
        assert resumed.recover().accepted_factors == HISTORY[:10]
        continued.restore(cancelled)
        assert continued.checkpoint() == cancelled
        completed = continued.solve(actions(), HISTORY[10:], max_iterations=24, max_backtracks=16)
        assert completed.accepted_factors == HISTORY
        assert continued.checkpoint() == owner.checkpoint()
        for field in ("shell_displacement", "cell_resultants", "support_reactions",
                      "beam_station_accumulated", "shell_layer_alpha"):
            np.testing.assert_array_equal(getattr(completed, field), getattr(final, field))
    finally:
        owner.close(); resumed.close(); continued.close()


def test_cyclic_admission_is_separate_owned_and_fixed_only():
    original = np.asarray(actions()[0].wrench)
    mutable = replace(actions()[0], wrench=original)
    model = replace(definition(), cyclic_reference_actions=(mutable,))
    owner = ss.create_analysis(definition=model)
    try:
        identity = owner._identity
        reference = owner._cyclic_reference_pattern.copy()
        original[:] = 0.
        np.testing.assert_array_equal(owner._cyclic_reference_pattern, reference)
        assert owner._identity == identity and owner._definition.cyclic_reference_actions[0].wrench[0] == 16e6
        assert owner.capabilities["increment_control"] == ("fixed",)
        for targets in ((0.,), (.25, .25), (float("nan"),), (True,)):
            with pytest.raises(ValueError):
                owner.solve(actions(), targets)
            assert owner._generation == 0 and owner._finite_driver is None
        with pytest.raises(ValueError, match="fixed increments"):
            owner.solve(actions(), (.25,), increment_control=ss.AdaptiveIncrementControl())
        with pytest.raises(ValueError, match="model-bound reference"):
            owner.solve((replace(actions()[0], wrench=(1., 0., 0., 0., 0., 0.)),), (.25,))
        assert owner._generation == 0 and owner._finite_driver is None
        bad = encode_control(owner._identity, _AdaptiveState(ss.AdaptiveIncrementControl(), (.25,), 12, 8, 0, 0.), b"{}")
        with pytest.raises(ValueError, match="exclude adaptive"):
            owner.restore(bad)
        assert owner._generation == 0 and owner._finite_driver is None
    finally: owner.close()
    with pytest.raises(ValueError, match="reference actions"):
        ss.create_analysis(definition=replace(definition(), cyclic_reference_actions=()))
    with pytest.raises(ValueError, match="selected cyclic"):
        ss.create_analysis(definition=replace(base_definition(), cyclic_reference_actions=actions()))
