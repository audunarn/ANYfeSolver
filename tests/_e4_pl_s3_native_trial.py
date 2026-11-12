"""Private test support for solver-owned S3 multiplicative trial views."""

from __future__ import annotations

from typing import Any

import numpy as np

from anyfesolver._native_rotation_state import (
    NativeElementRotationView,
    NativeRotationStateStore,
    create_native_rotation_state_store,
)


def native_trial_for_increment(
    current_nodes: Any,
    director_triads: Any,
    increment20: Any,
    *,
    committed_rotation_matrices: Any | None = None,
    committed_rotation_coordinates: Any | None = None,
) -> tuple[
    NativeElementRotationView,
    np.ndarray,
    NativeRotationStateStore,
]:
    """Build an actual node-shared store trial and its exact element increment."""

    nodes = np.asarray(current_nodes, dtype=np.float64).reshape(3, 3)
    triads = np.asarray(director_triads, dtype=np.float64).reshape(4, 3, 3)
    increment = np.asarray(increment20, dtype=np.float64).reshape(20).copy()
    rotations = (
        np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0)
        if committed_rotation_matrices is None
        else np.asarray(committed_rotation_matrices, dtype=np.float64).reshape(3, 3, 3)
    )
    committed_theta = (
        np.zeros((3, 3), dtype=np.float64)
        if committed_rotation_coordinates is None
        else np.asarray(committed_rotation_coordinates, dtype=np.float64).reshape(3, 3)
    )
    reference_directors = np.einsum(
        "nji,nj->ni",
        rotations,
        triads[:3, :, 2],
    )
    committed_full = np.zeros(18, dtype=np.float64)
    for node in range(3):
        committed_full[6 * node + 3 : 6 * node + 6] = committed_theta[node]
    trial_full = committed_full.copy()
    trial_full += increment[:18]
    trial_coordinates = nodes + increment[:18].reshape(3, 6)[:, :3]
    node_ids = (11, 12, 13)
    store = create_native_rotation_state_store(
        node_ids,
        rotational_dofs={
            node_id: (6 * row + 3, 6 * row + 4, 6 * row + 5)
            for row, node_id in enumerate(node_ids)
        },
        coordinate_rows={node_id: row for row, node_id in enumerate(node_ids)},
        coordinate_node_ids=node_ids,
        committed_full_displacement=committed_full,
        committed_full_coordinates=nodes,
        committed_rotation_matrices={
            node_id: rotations[row] for row, node_id in enumerate(node_ids)
        },
    )
    token = store.begin_trial(trial_full, trial_coordinates)
    view = store.element_view(
        "S3_TEST",
        node_ids,
        reference_directors,
        trial_token=token,
    )
    exact_increment = increment.copy()
    by_node = exact_increment[:18].reshape(3, 6)
    by_node[:, :3] = view.coordinate_increment
    by_node[:, 3:6] = view.rotation_coordinate_increment
    return view, exact_increment, store


__all__ = ["native_trial_for_increment"]
