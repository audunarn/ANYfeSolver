from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from anyfesolver.nonlinear_state import (
    ImmutableStateSidecarError,
    NonlinearStateStore,
    PersistentStateEligibilityError,
    ShellStateBatch,
    ShellStateLayout,
    StaleStateTokenError,
    StateMaterializationPolicy,
    begin_state_evaluation,
    finish_state_evaluation,
)


def _layout(*element_ids: int) -> ShellStateLayout:
    return ShellStateLayout(tuple(element_ids), n_gp=2, num_layers=3)


def _state(layout: ShellStateLayout, value: float) -> dict[str, np.ndarray]:
    points = layout.points_per_element
    return {
        "plastic_strain": np.full((points, 3), value, dtype=float),
        "alpha": np.full(points, 10.0 * value, dtype=float),
        "layer_strain": np.full((points, 3), 100.0 * value, dtype=float),
    }


def _trial_arrays(layout: ShellStateLayout, value: float):
    count = layout.n_elements
    points = layout.points_per_element
    return {
        "plastic_strain": np.full((count, points, 3), value, dtype=float),
        "alpha": np.full((count, points), 10.0 * value, dtype=float),
        "layer_strain": np.full((count, points, 3), 100.0 * value, dtype=float),
    }


def test_legacy_state_evaluation_preserves_fresh_trial_mapping_identity() -> None:
    committed: dict[int, object] = {}
    trial = {17: {"alpha": np.array([0.25])}}

    token = begin_state_evaluation(committed)
    assert token is None
    assert finish_state_evaluation(committed, token, trial) is trial


def test_layout_is_immutable_and_pack_materialize_owns_contiguous_state() -> None:
    layout = _layout(11, 17)
    source = {11: _state(layout, 0.1), 17: _state(layout, 0.2)}
    batch = ShellStateBatch(layout, source)

    with pytest.raises(FrozenInstanceError):
        layout.n_gp = 99  # type: ignore[misc]
    assert layout.element_ids == (11, 17)
    assert layout.points_per_element == 6
    assert batch.committed_buffer.flags.c_contiguous
    arrays = batch.committed_arrays()
    assert arrays.plastic_strain.flags.c_contiguous
    assert arrays.alpha.flags.c_contiguous
    assert arrays.layer_strain.flags.c_contiguous
    assert not arrays.plastic_strain.flags.writeable

    source[11]["alpha"].fill(-1.0)
    materialized = batch.materialize(policy=StateMaterializationPolicy.RESTART)
    np.testing.assert_allclose(materialized[11]["alpha"], 1.0)
    np.testing.assert_allclose(materialized[17]["plastic_strain"], 0.2)
    materialized[11]["plastic_strain"].fill(123.0)
    np.testing.assert_allclose(batch[11]["plastic_strain"], 0.1)

    diagnostics = batch.diagnostics()
    assert diagnostics["state_batch_count"] == 1
    assert diagnostics["state_point_count"] == 12
    assert diagnostics["dictionary_fallback_element_count"] == 0
    assert diagnostics["materialization_reasons"]["restart"] == 1


def test_full_trial_commits_by_swap_and_discard_is_copy_free() -> None:
    layout = _layout(1, 2)
    batch = ShellStateBatch(layout, {1: _state(layout, 0.1), 2: _state(layout, 0.2)})
    original_committed_address = batch.committed_buffer.__array_interface__["data"][0]
    original_trial_address = batch.trial_buffer.__array_interface__["data"][0]

    rejected = batch.begin_trial()
    batch.update_trial(rejected, **_trial_arrays(layout, 0.5))
    np.testing.assert_allclose(
        batch.materialize(trial_token=rejected)[1]["plastic_strain"], 0.5
    )
    batch.discard_trial(rejected)
    np.testing.assert_allclose(batch[1]["plastic_strain"], 0.1)
    assert batch.committed_buffer.__array_interface__["data"][0] == original_committed_address
    assert batch.trial_buffer.__array_interface__["data"][0] == original_trial_address
    with pytest.raises(StaleStateTokenError):
        batch.commit(rejected)

    accepted = batch.begin_trial()
    batch.update_trial(accepted, **_trial_arrays(layout, 0.8))
    assert batch.commit(accepted) == 1
    assert batch.committed_buffer.__array_interface__["data"][0] == original_trial_address
    np.testing.assert_allclose(batch[2]["alpha"], 8.0)
    with pytest.raises(StaleStateTokenError):
        batch.discard_trial(accepted)

    diagnostics = batch.diagnostics()
    assert diagnostics["state_swap_commit_count"] == 1
    assert diagnostics["state_bounded_copy_commit_count"] == 0
    assert diagnostics["state_discard_count"] == 1
    assert diagnostics["stale_token_error_count"] == 2


def test_partial_commit_copies_only_missing_fields_and_deleted_rows_are_frozen() -> None:
    layout = _layout(3, 4)
    batch = ShellStateBatch(layout, {3: _state(layout, 0.1), 4: _state(layout, 0.2)})
    batch.freeze_deleted((4,))
    token = batch.begin_trial()
    batch.update_trial(token, **_trial_arrays(layout, 0.7))
    batch.commit(token)

    states = batch.materialize()
    np.testing.assert_allclose(states[3]["plastic_strain"], 0.7)
    np.testing.assert_allclose(states[3]["alpha"], 7.0)
    np.testing.assert_allclose(states[4]["plastic_strain"], 0.2)
    np.testing.assert_allclose(states[4]["alpha"], 2.0)
    diagnostics = batch.diagnostics()
    assert diagnostics["deleted_element_count"] == 1
    assert diagnostics["state_bounded_copy_commit_count"] == 1
    assert diagnostics["state_commit_copied_element_fields"] == 3


def test_initial_fields_and_provenance_are_immutable_owned_sidecars() -> None:
    layout = _layout(8)
    initial_stress = np.array([80.0e6, -4.0e6, 2.0e6])
    provenance = {
        "kind": "shell",
        "source": "weld-map-v3",
        "components": ["membrane_stress"],
    }
    initial = _state(layout, 0.0)
    initial["initial_membrane_stress"] = initial_stress
    initial["initial_field_provenance"] = provenance
    batch = ShellStateBatch(layout, {8: initial})

    initial_stress.fill(-999.0)
    provenance["components"].append("mutated")
    packed = batch.materialize()[8]
    np.testing.assert_allclose(
        packed["initial_membrane_stress"], [80.0e6, -4.0e6, 2.0e6]
    )
    assert packed["initial_field_provenance"]["components"] == ["membrane_stress"]
    packed["initial_membrane_stress"].fill(0.0)
    packed["initial_field_provenance"]["components"].append("local")
    again = batch.materialize()[8]
    np.testing.assert_allclose(
        again["initial_membrane_stress"], [80.0e6, -4.0e6, 2.0e6]
    )
    assert again["initial_field_provenance"]["components"] == ["membrane_stress"]

    token = batch.begin_trial()
    changed = _state(layout, 0.3)
    changed["initial_membrane_stress"] = np.zeros(3)
    changed["initial_field_provenance"] = {
        "kind": "shell",
        "source": "different",
        "components": ["membrane_stress"],
    }
    with pytest.raises(ImmutableStateSidecarError):
        batch.set_trial_state(token, 8, changed)
    batch.discard_trial(token)
    eligible, reason = batch.persistent_eligibility()
    assert eligible is False
    assert reason == "immutable_initial_field_requires_exact_override"


def test_scalar_unknown_state_uses_owned_dictionary_fallback() -> None:
    layout = _layout(21, 22)
    fallback = _state(layout, 0.4)
    fallback["layer_stress"] = np.full((layout.points_per_element, 3), 12.5)
    batch = ShellStateBatch(layout, {21: _state(layout, 0.1), 22: fallback})

    eligible, reason = batch.persistent_eligibility(layout)
    assert eligible is False
    assert reason == "legacy_dictionary_fallback"
    with pytest.raises(PersistentStateEligibilityError):
        token = batch.begin_trial()
        try:
            batch.update_trial(token, **_trial_arrays(layout, 0.7))
        finally:
            batch.discard_trial(token)

    fallback["layer_stress"].fill(-1.0)
    np.testing.assert_allclose(batch[22]["layer_stress"], 12.5)
    token = batch.begin_trial()
    next_fallback = batch[22]
    next_fallback["layer_stress"].fill(99.0)
    batch.set_trial_state(token, 22, next_fallback)
    batch.commit(token)
    np.testing.assert_allclose(batch[22]["layer_stress"], 99.0)

    diagnostics = batch.diagnostics()
    assert diagnostics["dictionary_fallback_element_count"] == 1
    reasons = diagnostics["dictionary_fallback_reasons"]
    assert reasons["unsupported_state_keys:layer_stress"] == [22]


def test_packed_state_can_demote_to_scalar_fallback_without_losing_data() -> None:
    layout = _layout(30)
    batch = ShellStateBatch(layout, {30: _state(layout, 0.2)})
    token = batch.begin_trial()
    scalar_state = _state(layout, 0.9)
    scalar_state["physical_layer_stress"] = np.full(
        (layout.points_per_element, 3), 123.0
    )
    batch.set_trial_state(token, 30, scalar_state)
    batch.commit(token)

    assert batch.all_packed is False
    state = batch[30]
    np.testing.assert_allclose(state["plastic_strain"], 0.9)
    np.testing.assert_allclose(state["physical_layer_stress"], 123.0)
    assert batch.diagnostics()["dictionary_fallback_element_count"] == 1


def test_store_coordinates_batches_fallback_and_owned_materialization() -> None:
    first = _layout(1, 2)
    second = ShellStateLayout((7,), n_gp=1, num_layers=2)
    initial = {
        1: _state(first, 0.1),
        2: _state(first, 0.2),
        7: _state(second, 0.7),
        99: {"custom_history": np.array([1.0, 2.0])},
    }
    store = NonlinearStateStore.from_shell_layouts((first, second), initial)
    token = store.begin_trial()
    store.update_shell_trial(token, 0, **_trial_arrays(first, 0.6))
    second_trial = _trial_arrays(second, 0.8)
    store.update_shell_trial(token, 1, **second_trial)
    store.set_trial_state(token, 99, {"custom_history": np.array([3.0, 4.0])})
    trial = store.materialize(trial_token=token, policy="saved_state")
    np.testing.assert_allclose(trial[1]["alpha"], 6.0)
    np.testing.assert_allclose(trial[7]["alpha"], 8.0)
    np.testing.assert_allclose(trial[99]["custom_history"], [3.0, 4.0])
    store.commit(token)

    public = store.materialize(policy="final_result")
    public[1]["alpha"].fill(-1.0)
    public[99]["custom_history"].fill(-1.0)
    np.testing.assert_allclose(store[1]["alpha"], 6.0)
    np.testing.assert_allclose(store[99]["custom_history"], [3.0, 4.0])
    diagnostics = store.diagnostics()
    assert diagnostics["state_batch_count"] == 2
    assert diagnostics["state_point_count"] == 14
    assert diagnostics["dictionary_fallback_element_count"] == 1
    assert diagnostics["dictionary_fallback_reasons"] == {
        "unbatched_element_state": [99]
    }


def test_store_deleted_fallback_retains_committed_state() -> None:
    store = NonlinearStateStore.from_shell_layouts(
        (), {5: {"alpha": np.array([0.25]), "custom": "beam"}}
    )
    store.freeze_deleted((5,))
    token = store.begin_trial()
    store.set_trial_state(token, 5, {"alpha": np.array([9.0]), "custom": "beam"})
    store.commit(token)
    np.testing.assert_allclose(store[5]["alpha"], 0.25)
    assert store.diagnostics()["deleted_element_count"] == 1


def test_tokens_and_state_are_independent_between_solver_stores() -> None:
    layout = _layout(41)
    first = NonlinearStateStore.from_shell_layouts((layout,), {41: _state(layout, 0.1)})
    second = NonlinearStateStore.from_shell_layouts((layout,), {41: _state(layout, 0.2)})
    first_token = first.begin_trial()
    second_token = second.begin_trial()

    with pytest.raises(StaleStateTokenError):
        second.commit(first_token)
    first.update_shell_trial(first_token, 0, **_trial_arrays(layout, 0.9))
    first.commit(first_token)
    second.discard_trial(second_token)
    np.testing.assert_allclose(first[41]["plastic_strain"], 0.9)
    np.testing.assert_allclose(second[41]["plastic_strain"], 0.2)


def test_shell_batch_persistent_entry_point_is_exact_mapping_parity() -> None:
    from anyfesolver.material_curves import dnv_c208_steel_curve
    from anyfesolver.mesh_gen import generate_simple_panel_mesh
    from anyfesolver.nonlinear_performance import NonlinearAssemblyPlan

    model = generate_simple_panel_mesh(
        1.2,
        0.8,
        0.01,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    first_element = next(iter(model.mesh.elements.values()))
    material = model.get_material(first_element.material_name)
    material.hardening_curve = dnv_c208_steel_curve("S355", 0.01)
    model.bump_revision("material")
    plan = NonlinearAssemblyPlan.build(model, num_layers=5)
    assert len(plan.shell_batches) == 1
    shell_plan = plan.shell_batches[0]
    initial = {
        int(element_id): element.init_nonlinear_state(5)
        for element_id, element in zip(shell_plan.element_ids, shell_plan.elements)
    }
    state_batch = ShellStateBatch(shell_plan.state_layout, initial)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = 0.004 * node.x
        displacement[node.dofs[1]] = -0.001 * node.y
        displacement[node.dofs[4]] = 0.003 * node.x

    token = state_batch.begin_trial()
    force_array, tangent_array, _seconds = shell_plan.evaluate_persistent(
        displacement,
        state_batch,
        token,
        tangent=True,
    )
    force_mapping, tangent_mapping, states_mapping, _seconds = shell_plan.evaluate(
        displacement,
        initial,
        tangent=True,
    )
    np.testing.assert_allclose(force_array, force_mapping, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tangent_array, tangent_mapping, rtol=0.0, atol=0.0)
    states_array = state_batch.materialize(trial_token=token)
    for element_id in states_mapping:
        np.testing.assert_allclose(
            states_array[element_id]["plastic_strain"],
            states_mapping[element_id]["plastic_strain"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            states_array[element_id]["alpha"],
            states_mapping[element_id]["alpha"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            states_array[element_id]["layer_strain"],
            states_mapping[element_id]["layer_strain"],
            rtol=0.0,
            atol=0.0,
        )
    state_batch.discard_trial(token)
