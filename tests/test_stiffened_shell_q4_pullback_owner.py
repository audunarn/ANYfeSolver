"""Selected full Q4 force-map identity and actual mixed-owner history."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import stiffened_shell
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_finite_experimental import _actions, _q4_shell_j2_panel
from test_stiffened_shell_ge_owned_rebase import _compare_physical


def _definition(plastic=False, mixed=False):
    base = _q4_shell_j2_panel() if plastic else _panel()
    if mixed:
        base = replace(base, shell=replace(base.shell,
            q4_elements=((1, 2, 5, 4),), s3_elements=((2, 3, 6), (2, 6, 5))))
    return replace(base, q4_pullback_policy=stiffened_shell.Q4_ALGORITHMIC_PULLBACK_V1)


def test_selected_q4_reference_map_and_checkpoint_identity(monkeypatch):
    from anyfesolver import _stiffened_shell_ge_experimental as implementation
    definition = _definition()
    evaluations = []
    original = implementation.q4_algorithmic_response

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        evaluations.append((args[1].element_id, result))
        return result

    monkeypatch.setattr(implementation, "q4_algorithmic_response", record)
    selected = stiffened_shell.create_analysis(definition=definition)
    assert [item[0] for item in evaluations] == [1, 2]
    expected = np.zeros((selected._shell_count, selected._shell_count))
    for (_, response), (_, _, slots, _) in zip(evaluations, selected._shell_parts):
        expected[np.ix_(slots, slots)] += response[1]
    np.testing.assert_allclose(selected._K[:selected._shell_count, :selected._shell_count].toarray(),
                               expected, rtol=1e-12, atol=1e-7)
    historical = stiffened_shell.create_analysis(definition=_panel())
    assert historical._identity != selected._identity
    assert historical.capabilities["q4_force_map"] == "historical_corotational"
    assert selected.capabilities["q4_force_map"] == stiffened_shell.Q4_ALGORITHMIC_PULLBACK_V1
    assert selected.preflight()["production_qualified"] is False
    selected.solve_linear(_actions())
    raw = selected.checkpoint()
    with pytest.raises(ValueError):
        historical.restore(raw)
    restored = stiffened_shell.create_analysis(definition=definition)
    restored.restore(raw)
    np.testing.assert_array_equal(restored.recover().shell_displacement,
                                  selected.recover().shell_displacement)
    historical.solve_linear(_actions())
    incompatible = stiffened_shell.create_analysis(definition=definition)
    with pytest.raises(ValueError):
        incompatible.restore(historical.checkpoint())
    assert incompatible._generation == 0
    assert restored.checkpoint() == raw


def test_selected_q4_rejects_unknown_and_inapplicable_policies():
    for policy in ("unknown", None, True):
        with pytest.raises(ValueError, match="policy"):
            stiffened_shell.create_analysis(definition=replace(_panel(), q4_pullback_policy=policy))
    base = _definition()
    all_s3 = replace(base, shell=replace(base.shell, q4_elements=(),
        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))
    with pytest.raises(ValueError, match="Q4"):
        stiffened_shell.create_analysis(definition=all_s3)


def _balance(owner):
    result = owner.recover()
    assert result.free_equilibrium_ratio <= 1e-8
    assert result.constraint_max <= 1e-9
    supports = result.support_reactions[:owner._shell_count].reshape(-1, 6)
    positions = (np.asarray(owner._definition.shell.coordinates_m)+
                 result.shell_displacement.reshape(-1, 6)[:, :3])
    load = np.array((0., 0., 1000.*result.load_factor))
    moment = np.cross(result.beam_positions[-1], load)
    force_error = supports[:, :3].sum(0)+load
    moment_error = supports[:, 3:].sum(0)+np.cross(positions, supports[:, :3]).sum(0)+moment
    assert np.linalg.norm(force_error)/max(1., np.linalg.norm(load)) <= 1e-8
    assert np.linalg.norm(moment_error)/max(1., np.linalg.norm(moment)) <= 1e-8


def _layers(left, right):
    for name in ("shell_q4_material", "shell_s3_material"):
        for a, b in zip(getattr(left.recover(), name), getattr(right.recover(), name)):
            for field in ("layer_strain", "layer_stress", "plastic_strain", "alpha"):
                np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(left.recover().beam_station_accumulated,
                               right.recover().beam_station_accumulated, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("mixed", [False, True])
def test_q4_selected_yielded_unload_reload_rebase_restart_and_balance(mixed):
    definition = _definition(plastic=True, mixed=mixed)
    owner = stiffened_shell.create_analysis(definition=definition)
    first = owner.solve(_actions(), (.5, 1.), max_iterations=20, max_backtracks=12)
    assert max(max(row) for row in first.shell_layer_alpha) > 0.
    assert max(max(row) for row in first.beam_station_accumulated) > 0.
    _balance(owner)
    raw = owner.checkpoint()
    control = stiffened_shell.create_analysis(definition=definition)
    control.restore(raw)
    owner.rebase_rotations(first.generation)
    resumed = stiffened_shell.create_analysis(definition=definition)
    resumed.restore(owner.checkpoint())
    for other in (control, resumed):
        _compare_physical(owner, other)
        _layers(owner, other)
    for factor in (.8, 1.1):
        for target in (control, owner, resumed):
            prefix = target.checkpoint()
            token = CancellationToken(); token.cancel()
            with pytest.raises(SolveCancelled):
                target.solve(_actions(), (factor,), cancellation_token=token)
            assert target.checkpoint() == prefix
            target.solve(_actions(), (factor,), max_iterations=20, max_backtracks=12)
            _balance(target)
        for other in (control, resumed):
            _compare_physical(owner, other)
            _layers(owner, other)
    assert owner.recover().accepted_factors == (.5, 1., .8, 1.1)
