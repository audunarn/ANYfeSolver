"""Focused qualification tests for element activity and deletion state."""

from __future__ import annotations

import json

import numpy as np
import pytest

from anyfesolver.activity import (
    ContributionPolicy,
    CouplingPolicy,
    CouplingResolutionError,
    ElementActivity,
    ElementActivityError,
    ElementActivityPolicy,
    RestartStateError,
)


def test_stable_ids_softening_and_hard_deletion_are_irreversible() -> None:
    manager = ElementActivity([40, 10, 30], activity=[1.0, 0.8, 1.0])

    np.testing.assert_array_equal(manager.element_ids, [40, 10, 30])
    np.testing.assert_array_equal(manager.softened_mask, [False, True, False])
    np.testing.assert_array_equal(manager.hard_deleted_mask, [False, False, False])
    assert manager.activity.dtype == np.dtype(float)
    assert manager.element_ids.flags.writeable is False
    assert manager.activity.flags.writeable is False

    change = manager.set_activity(
        [30, 10],
        [0.0, 0.95],
        step=4,
        time=0.25,
        reason="failure_screen",
    )
    np.testing.assert_array_equal(change.element_ids, [30])
    np.testing.assert_allclose(manager.activity, [1.0, 0.8, 0.0])
    np.testing.assert_array_equal(manager.hard_deleted_mask, [False, False, True])
    np.testing.assert_array_equal(change.newly_hard_deleted_ids, [30])

    manager.set_activity([30], [1.0], allow_healing=True)
    np.testing.assert_allclose(manager.activity, [1.0, 0.8, 0.0])
    manager.hard_delete([40], step=5)
    np.testing.assert_array_equal(manager.active_mask, [False, True, False])
    assert [entry.sequence for entry in manager.history] == [1, 2]


def test_explicit_policies_and_vectorized_scaling_preserve_storage_shape() -> None:
    policy = ElementActivityPolicy(
        stiffness=ContributionPolicy.ACTIVITY,
        mass=ContributionPolicy.DELETE_ONLY,
        damping=ContributionPolicy.RETAIN,
        load=ContributionPolicy.ACTIVITY,
        contact=ContributionPolicy.DELETE_ONLY,
    )
    manager = ElementActivity([1, 2, 3], [1.0, 0.25, 0.0], policy=policy)

    np.testing.assert_allclose(manager.stiffness_scales(), [1.0, 0.25, 0.0])
    np.testing.assert_allclose(manager.mass_scales(), [1.0, 1.0, 0.0])
    np.testing.assert_allclose(manager.damping_scales(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(manager.load_scales(), [1.0, 0.25, 0.0])
    np.testing.assert_allclose(manager.contact_scales(), [1.0, 1.0, 0.0])

    element_blocks = np.arange(12.0).reshape(3, 2, 2)
    output = np.empty_like(element_blocks)
    returned = manager.scale_stiffness(element_blocks, out=output)
    assert returned is output
    assert output.shape == element_blocks.shape
    np.testing.assert_allclose(output[0], element_blocks[0])
    np.testing.assert_allclose(output[1], 0.25 * element_blocks[1])
    np.testing.assert_allclose(output[2], 0.0)

    csr_contributions = np.array([2.0, 3.0, 5.0, 7.0])
    contribution_output = np.empty_like(csr_contributions)
    manager.scale_contributions(
        csr_contributions,
        [1, 2, 2, 3],
        "stiffness",
        out=contribution_output,
    )
    np.testing.assert_allclose(contribution_output, [2.0, 0.75, 1.25, 0.0])


def test_active_load_and_contact_filters_are_reusable_and_scaled() -> None:
    policy = ElementActivityPolicy(
        load="activity",
        contact="delete_only",
    )
    manager = ElementActivity([10, 20, 30], [1.0, 0.5, 0.0], policy=policy)
    owners = np.array([30, 10, 20, 20])

    load_filter = manager.load_filter(owners)
    np.testing.assert_array_equal(load_filter.mask, [False, True, True, True])
    np.testing.assert_array_equal(load_filter.indices, [1, 2, 3])
    loads = np.array([[30.0, 3.0], [10.0, 1.0], [20.0, 2.0], [8.0, 4.0]])
    np.testing.assert_allclose(
        manager.filter_loads(owners, loads),
        [[10.0, 1.0], [10.0, 1.0], [4.0, 2.0]],
    )

    contact_filter = manager.contact_filter(owners)
    np.testing.assert_array_equal(contact_filter.mask, [False, True, True, True])
    np.testing.assert_allclose(
        manager.filter_contacts(owners, np.array([3.0, 1.0, 2.0, 4.0])),
        [1.0, 2.0, 4.0],
    )


def test_orphan_dofs_are_detected_in_bounded_connectivity_batches() -> None:
    manager = ElementActivity([10, 20, 30])
    connectivity = np.array(
        [
            [0, 1, 2, -1],
            [2, 3, 4, -1],
            [4, 5, 6, -1],
        ],
        dtype=np.int64,
    )
    manager.hard_delete([20, 30])

    report = manager.detect_orphan_dofs(
        connectivity,
        total_dofs=8,
        batch_size=1,
    )
    np.testing.assert_array_equal(report.orphan_dofs, [3, 4, 5, 6])
    np.testing.assert_array_equal(report.active_support_counts, [1, 1, 1, 0, 0, 0, 0, 0])
    assert report.original_support_counts[2] == 2
    assert report.orphan_count == 4
    assert report.orphan_mask[7] == np.bool_(False)

    with_unconnected = manager.detect_orphan_dofs(
        connectivity,
        total_dofs=8,
        batch_size=2,
        include_never_connected=True,
    )
    np.testing.assert_array_equal(with_unconnected.orphan_dofs, [3, 4, 5, 6, 7])


def test_coupling_policy_deactivates_or_reassigns_deterministically() -> None:
    manager = ElementActivity(
        [10, 20, 30],
        [1.0, 0.0, 0.5],
        policy=ElementActivityPolicy(coupling=CouplingPolicy.REASSIGN),
    )
    owners = [10, 20, 20, 20]
    candidates = np.array(
        [
            [-1, -1],
            [30, 10],
            [30, -1],
            [999, -1],
        ]
    )

    resolution = manager.resolve_couplings(owners, candidates)
    np.testing.assert_array_equal(resolution.assigned_element_ids, [10, 10, 30, -1])
    np.testing.assert_array_equal(resolution.active_mask, [True, True, True, False])
    np.testing.assert_array_equal(resolution.reassigned_mask, [False, True, True, False])
    assert resolution.reassigned_count == 2
    assert resolution.deactivated_count == 1

    deactivated = manager.resolve_couplings(owners, policy="deactivate")
    np.testing.assert_array_equal(deactivated.active_mask, [True, False, False, False])
    with pytest.raises(CouplingResolutionError):
        manager.resolve_couplings(owners, policy="error")


def test_damage_history_round_trips_through_json_restart_as_owned_state() -> None:
    policy = ElementActivityPolicy(hard_delete_threshold=0.05)
    manager = ElementActivity([7, 3, 11], policy=policy)
    manager.soften([3, 7], [0.5, 0.8], step=1, time=0.1)
    manager.apply_damage([11], [0.98], step=2, time=0.2)

    payload = json.loads(json.dumps(manager.to_restart()))
    restarted = ElementActivity.from_restart(payload)
    np.testing.assert_array_equal(restarted.element_ids, manager.element_ids)
    np.testing.assert_allclose(restarted.activity, manager.activity)
    np.testing.assert_allclose(restarted.minimum_activity, manager.minimum_activity)
    np.testing.assert_array_equal(restarted.hard_deleted_mask, manager.hard_deleted_mask)
    assert restarted.policy == manager.policy
    assert [entry.to_dict() for entry in restarted.history] == [
        entry.to_dict() for entry in manager.history
    ]

    payload["activity"][0] = 0.0
    assert restarted.activity[0] == pytest.approx(0.8)

    incompatible = json.loads(json.dumps(manager.to_restart()))
    incompatible["hard_deleted"][0] = True
    with pytest.raises(RestartStateError):
        ElementActivity.from_restart(incompatible)


def test_conditioning_and_removed_mass_energy_diagnostics_are_policy_aware() -> None:
    manager = ElementActivity(
        [1, 2, 3],
        [1.0, 0.25, 0.0],
        policy=ElementActivityPolicy(
            mass="delete_only",
            conditioning_warning_ratio=300.0,
        ),
    )
    diagnostics = manager.diagnostics(
        element_mass=[2.0, 3.0, 5.0],
        element_energy=[10.0, 20.0, 30.0],
        reference_condition_number=100.0,
    )

    assert diagnostics["fixed_matrix_sparsity"] is True
    assert diagnostics["softened_element_count"] == 1
    assert diagnostics["hard_deleted_element_count"] == 1
    assert diagnostics["stiffness_scale_ratio"] == pytest.approx(4.0)
    assert diagnostics["estimated_condition_number"] == pytest.approx(400.0)
    assert diagnostics["conditioning_warning"] is True
    assert diagnostics["input_mass"] == pytest.approx(10.0)
    assert diagnostics["retained_mass"] == pytest.approx(5.0)
    assert diagnostics["removed_mass"] == pytest.approx(5.0)
    assert diagnostics["removed_energy"] == pytest.approx(45.0)
    assert diagnostics["removed_energy_fraction"] == pytest.approx(0.75)


def test_contract_validation_rejects_ambiguous_or_stale_inputs() -> None:
    with pytest.raises(ElementActivityError):
        ElementActivity([1, 1])
    manager = ElementActivity([1, 2])
    with pytest.raises(KeyError):
        manager.set_activity([9], [0.5])
    with pytest.raises(ElementActivityError):
        manager.set_activity([1, 1], [0.5, 0.4])
    with pytest.raises(ElementActivityError):
        manager.scale_stiffness(np.ones((3, 2)))
    with pytest.raises(ElementActivityError):
        manager.detect_orphan_dofs(np.array([[0.0, 1.0], [1.0, 2.0]]))
