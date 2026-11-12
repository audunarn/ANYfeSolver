"""Full-kernel D3 covariance checks for the private native S3 TL response.

The maps in this module are constructed directly from physical node
permutations, barycentric coordinates, and second-order tensor transport.
They do not use the production element's frame, DOF, or recovery maps.  The
tests exercise the complete seven-station layered response after its two
hierarchical bubble coordinates have been equilibrated.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import numpy as np

import anyfesolver.e4_pl_s3_element as s3
from anyfesolver import OrthotropicMaterial
from _e4_pl_s3_native_trial import native_trial_for_increment


_THICKNESS = 0.083
_LAYER_COUNT = 3


def _unit(vector: np.ndarray) -> np.ndarray:
    made = np.asarray(vector, dtype=float)
    return made / float(np.linalg.norm(made))


def _independent_frame(nodes: np.ndarray, owner_normal: np.ndarray) -> np.ndarray:
    """Build the numbered right-handed frame without formulation helpers."""

    made = np.asarray(nodes, dtype=float)
    normal = _unit(owner_normal)
    first = _unit(made[1] - made[0])
    first = _unit(first - float(first @ normal) * normal)
    second = _unit(np.cross(normal, first))
    first = _unit(np.cross(second, normal))
    return np.column_stack((first, second, normal))


def _eq11_triad(normal: np.ndarray) -> np.ndarray:
    """Independent global-e_y chart with the published e_z fallback."""

    unit_normal = _unit(normal)
    first = np.cross(np.asarray((0.0, 1.0, 0.0)), unit_normal)
    if float(np.linalg.norm(first)) <= 1.0e-12:
        first = np.cross(np.asarray((0.0, 0.0, 1.0)), unit_normal)
    first = _unit(first)
    second = _unit(np.cross(unit_normal, first))
    return np.column_stack((first, second, unit_normal))


def _external_permutation(permutation: tuple[int, int, int]) -> np.ndarray:
    """Return q_numbered = P q_baseline for global nodal coordinates."""

    result = np.zeros((18, 18), dtype=float)
    for new_node, baseline_node in enumerate(permutation):
        new_slice = slice(6 * new_node, 6 * new_node + 6)
        old_slice = slice(6 * baseline_node, 6 * baseline_node + 6)
        result[new_slice, old_slice] = np.eye(6)
    return result


def _engineering_tensor(values: np.ndarray) -> np.ndarray:
    xx, yy, xy = np.asarray(values, dtype=float)
    return np.asarray(((xx, 0.5 * xy), (0.5 * xy, yy)))


def _pack_engineering(tensor: np.ndarray) -> np.ndarray:
    made = np.asarray(tensor, dtype=float)
    return np.asarray((made[0, 0], made[1, 1], 2.0 * made[0, 1]))


def _resultant_tensor(values: np.ndarray) -> np.ndarray:
    xx, yy, xy = np.asarray(values, dtype=float)
    return np.asarray(((xx, xy), (xy, yy)))


def _pack_resultant(tensor: np.ndarray) -> np.ndarray:
    made = np.asarray(tensor, dtype=float)
    return np.asarray((made[0, 0], made[1, 1], made[0, 1]))


def _engineering_transform(rotation: np.ndarray) -> np.ndarray:
    basis = np.eye(3)
    return np.column_stack(
        tuple(
            _pack_engineering(
                rotation @ _engineering_tensor(basis[column]) @ rotation.T
            )
            for column in range(3)
        )
    )


def _resultant_transform(rotation: np.ndarray) -> np.ndarray:
    basis = np.eye(3)
    return np.column_stack(
        tuple(
            _pack_resultant(
                rotation @ _resultant_tensor(basis[column]) @ rotation.T
            )
            for column in range(3)
        )
    )


def _station_map(permutation: tuple[int, int, int]) -> np.ndarray:
    """Map each numbered integration station to its physical baseline row."""

    barycentric = np.asarray(
        [(1.0 - r - s_value, r, s_value) for r, s_value, _ in s3.TRIANGLE_QUADRATURE]
    )
    result = np.empty(len(barycentric), dtype=int)
    for new_index, new_coordinates in enumerate(barycentric):
        baseline_coordinates = np.zeros(3, dtype=float)
        baseline_coordinates[np.asarray(permutation)] = new_coordinates
        distances = np.max(
            np.abs(barycentric - baseline_coordinates[None, :]), axis=1
        )
        result[new_index] = int(np.argmin(distances))
        assert distances[result[new_index]] <= 3.0e-15
    assert sorted(result.tolist()) == list(range(len(barycentric)))
    return result


def _transform_generalized(
    values: np.ndarray,
    engineering: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    made = np.asarray(values, dtype=float)
    result = np.empty_like(made)
    result[..., :3] = made[..., :3] @ engineering.T
    result[..., 3:6] = made[..., 3:6] @ engineering.T
    result[..., 6:] = made[..., 6:] @ rotation.T
    return result


def _transform_generalized_resultant(
    values: np.ndarray,
    resultant: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    made = np.asarray(values, dtype=float)
    result = np.empty_like(made)
    result[..., :3] = made[..., :3] @ resultant.T
    result[..., 3:6] = made[..., 3:6] @ resultant.T
    result[..., 6:] = made[..., 6:] @ rotation.T
    return result


def _numbered_state(
    baseline: Mapping[str, np.ndarray],
    station_map: np.ndarray,
    engineering: np.ndarray,
    resultant: np.ndarray,
    rotation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Transport the committed state into a numbered reference frame."""

    layer_map = np.concatenate(
        [
            np.arange(
                station * _LAYER_COUNT,
                (station + 1) * _LAYER_COUNT,
                dtype=int,
            )
            for station in station_map
        ]
    )
    np.testing.assert_allclose(
        engineering.T @ resultant,
        np.eye(3),
        rtol=0.0,
        atol=2.0e-15,
    )
    return {
        "kinematic_layer_strain": (
            np.asarray(baseline["kinematic_layer_strain"])[layer_map]
            @ engineering.T
        ),
        "station_generalized_strain": _transform_generalized(
            np.asarray(baseline["station_generalized_strain"])[station_map],
            engineering,
            rotation,
        ),
        # These histories live in the fixed physical material frame.
        "plastic_strain": np.asarray(baseline["plastic_strain"])[layer_map].copy(),
        "alpha": np.asarray(baseline["alpha"])[layer_map].copy(),
        "initial_membrane_stress": (
            np.asarray(baseline["initial_membrane_stress"])[station_map]
            @ resultant.T
        ),
        "initial_bending_stress": (
            np.asarray(baseline["initial_bending_stress"])[station_map]
            @ resultant.T
        ),
        "initial_membrane_prestrain": (
            np.asarray(baseline["initial_membrane_prestrain"])[station_map]
            @ engineering.T
        ),
        "initial_curvature_prestrain": (
            np.asarray(baseline["initial_curvature_prestrain"])[station_map]
            @ engineering.T
        ),
    }


def _base_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    OrthotropicMaterial,
    dict[str, np.ndarray],
    np.ndarray,
]:
    reference_nodes = np.asarray(
        (
            (0.17, -0.22, 0.31),
            (1.43, 0.04, 0.58),
            (0.28, 1.06, 0.47),
        ),
        dtype=float,
    )
    owner = _unit(
        np.cross(
            reference_nodes[1] - reference_nodes[0],
            reference_nodes[2] - reference_nodes[0],
        )
    )
    frame = _independent_frame(reference_nodes, owner)
    current_nodes = reference_nodes + np.asarray(
        (
            (0.012, -0.006, 0.004),
            (0.019, 0.008, -0.003),
            (-0.007, 0.014, 0.009),
        )
    )
    director_normals = np.asarray(
        (
            owner + 0.023 * frame[:, 0] - 0.011 * frame[:, 1],
            owner - 0.017 * frame[:, 0] + 0.019 * frame[:, 1],
            owner + 0.009 * frame[:, 0] + 0.014 * frame[:, 1],
            owner - 0.006 * frame[:, 0] - 0.012 * frame[:, 1],
        )
    )
    triads = np.asarray([_eq11_triad(normal) for normal in director_normals])
    material_direction = _unit(
        math.cos(0.37) * frame[:, 0] + math.sin(0.37) * frame[:, 1]
    )
    material = OrthotropicMaterial(
        name="full-d3-ortho",
        elastic_modulus_1=142.0e9,
        elastic_modulus_2=13.0e9,
        elastic_modulus_3=8.5e9,
        poisson_ratio_12=0.23,
        poisson_ratio_13=0.18,
        poisson_ratio_23=0.27,
        shear_modulus_12=5.6e9,
        shear_modulus_13=4.1e9,
        shear_modulus_23=3.2e9,
        density=1610.0,
    )
    stations = len(s3.TRIANGLE_QUADRATURE)
    z = np.asarray((-0.5 * _THICKNESS, 0.0, 0.5 * _THICKNESS))
    membrane = np.empty((stations, 3), dtype=float)
    curvature = np.empty((stations, 3), dtype=float)
    shear = np.empty((stations, 2), dtype=float)
    for index, (r, s_value, _weight) in enumerate(s3.TRIANGLE_QUADRATURE):
        membrane[index] = 1.0e-5 * np.asarray(
            (0.8 + 0.2 * r, -0.4 + 0.3 * s_value, 0.16 - 0.1 * r)
        )
        curvature[index] = 1.0e-4 * np.asarray(
            (0.31 - 0.2 * s_value, -0.23 + 0.15 * r, 0.11 + 0.07 * s_value)
        )
        shear[index] = 1.0e-5 * np.asarray(
            (0.22 + 0.1 * r, -0.18 + 0.12 * s_value)
        )
    generalized = np.concatenate((membrane, curvature, shear), axis=1)
    kinematic = (membrane[:, None, :] + z[None, :, None] * curvature[:, None, :])
    point_count = stations * _LAYER_COUNT
    state = {
        "kinematic_layer_strain": kinematic.reshape(point_count, 3),
        "station_generalized_strain": generalized,
        "plastic_strain": np.linspace(-2.0e-7, 3.0e-7, point_count * 3).reshape(
            point_count, 3
        ),
        "alpha": np.linspace(0.0, 1.0e-6, point_count),
        "initial_membrane_stress": np.column_stack(
            (
                np.linspace(1.1e5, 1.7e5, stations),
                np.linspace(-0.4e5, 0.2e5, stations),
                np.linspace(0.07e5, -0.03e5, stations),
            )
        ),
        "initial_bending_stress": np.column_stack(
            (
                np.linspace(0.8e4, 1.0e4, stations),
                np.linspace(-0.2e4, 0.1e4, stations),
                np.linspace(0.04e4, -0.02e4, stations),
            )
        ),
        "initial_membrane_prestrain": 0.09 * membrane,
        "initial_curvature_prestrain": -0.07 * curvature,
    }
    external = np.asarray(
        (
            1.1e-4,
            -0.7e-4,
            0.5e-4,
            0.8e-4,
            -0.4e-4,
            0.3e-4,
            -0.6e-4,
            1.3e-4,
            -0.8e-4,
            -0.5e-4,
            0.9e-4,
            -0.2e-4,
            0.4e-4,
            -1.0e-4,
            0.7e-4,
            0.6e-4,
            0.2e-4,
            -0.7e-4,
        )
    )
    return (
        reference_nodes,
        current_nodes,
        owner,
        triads,
        material_direction,
        material,
        state,
        external,
    )


def _response(
    reference_nodes: np.ndarray,
    current_nodes: np.ndarray,
    frame: np.ndarray,
    triads: np.ndarray,
    material_direction: np.ndarray,
    material: OrthotropicMaterial,
    state: Mapping[str, np.ndarray],
    external: np.ndarray,
):
    direction_components = frame[:, :2].T @ material_direction
    material_angle = float(
        math.atan2(direction_components[1], direction_components[0])
    )

    def builder(increment: np.ndarray):
        native_trial, exact, _store = native_trial_for_increment(
            current_nodes, triads, increment
        )
        return s3._native_layered_uncondensed_response(
            current_nodes,
            triads,
            exact,
            reference_nodes,
            frame,
            material,
            material_angle,
            _THICKNESS,
            state,
            _LAYER_COUNT,
            native_rotation_trial=native_trial,
        )

    return s3._solve_native_bubble_equilibrium(
        external,
        np.zeros(2),
        builder,
    )


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    made = np.asarray(actual, dtype=float)
    reference = np.asarray(expected, dtype=float)
    scale = max(
        float(np.linalg.norm(made, ord=np.inf)),
        float(np.linalg.norm(reference, ord=np.inf)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(made - reference, ord=np.inf)) / scale


def _assert_trial_state_transport(
    actual: Mapping[str, object],
    baseline: Mapping[str, object],
    station_map: np.ndarray,
    engineering: np.ndarray,
    resultant: np.ndarray,
    rotation: np.ndarray,
) -> None:
    layer_map = np.concatenate(
        [
            np.arange(
                station * _LAYER_COUNT,
                (station + 1) * _LAYER_COUNT,
                dtype=int,
            )
            for station in station_map
        ]
    )
    expected_fields = {
        "plastic_strain": np.asarray(baseline["plastic_strain"])[layer_map],
        "alpha": np.asarray(baseline["alpha"])[layer_map],
        "layer_strain": (
            np.asarray(baseline["layer_strain"])[layer_map] @ engineering.T
        ),
        "layer_strain_material": np.asarray(
            baseline["layer_strain_material"]
        )[layer_map],
        "kinematic_layer_strain": (
            np.asarray(baseline["kinematic_layer_strain"])[layer_map]
            @ engineering.T
        ),
        "layer_stress": (
            np.asarray(baseline["layer_stress"])[layer_map] @ resultant.T
        ),
        "layer_stress_material": np.asarray(
            baseline["layer_stress_material"]
        )[layer_map],
        "station_generalized_strain": _transform_generalized(
            np.asarray(baseline["station_generalized_strain"])[station_map],
            engineering,
            rotation,
        ),
        "station_generalized_resultant": _transform_generalized_resultant(
            np.asarray(baseline["station_generalized_resultant"])[station_map],
            resultant,
            rotation,
        ),
        "membrane_strain": (
            np.asarray(baseline["membrane_strain"])[station_map]
            @ engineering.T
        ),
        "curvature": (
            np.asarray(baseline["curvature"])[station_map] @ engineering.T
        ),
        "transverse_shear_strain": (
            np.asarray(baseline["transverse_shear_strain"])[station_map]
            @ rotation.T
        ),
        "membrane_resultants": (
            np.asarray(baseline["membrane_resultants"])[station_map]
            @ resultant.T
        ),
        "bending_resultants": (
            np.asarray(baseline["bending_resultants"])[station_map]
            @ resultant.T
        ),
        "transverse_shear_resultants": (
            np.asarray(baseline["transverse_shear_resultants"])[station_map]
            @ rotation.T
        ),
        "initial_membrane_stress": (
            np.asarray(baseline["initial_membrane_stress"])[station_map]
            @ resultant.T
        ),
        "initial_bending_stress": (
            np.asarray(baseline["initial_bending_stress"])[station_map]
            @ resultant.T
        ),
        "initial_membrane_prestrain": (
            np.asarray(baseline["initial_membrane_prestrain"])[station_map]
            @ engineering.T
        ),
        "initial_curvature_prestrain": (
            np.asarray(baseline["initial_curvature_prestrain"])[station_map]
            @ engineering.T
        ),
    }
    for key, expected in expected_fields.items():
        assert _relative_error(np.asarray(actual[key]), expected) <= 3.0e-11, key
    actual_station_work = np.einsum(
        "ij,ij->i",
        np.asarray(actual["station_generalized_strain"]),
        np.asarray(actual["station_generalized_resultant"]),
    )
    baseline_station_work = np.einsum(
        "ij,ij->i",
        np.asarray(baseline["station_generalized_strain"])[station_map],
        np.asarray(baseline["station_generalized_resultant"])[station_map],
    )
    np.testing.assert_allclose(
        actual_station_work,
        baseline_station_work,
        rtol=3.0e-11,
        atol=1.0e-12,
    )
    assert actual["membrane_resultant_order"] == baseline["membrane_resultant_order"]
    assert (
        actual["transverse_shear_resultant_order"]
        == baseline["transverse_shear_resultant_order"]
    )
    assert actual["equivalent_stress_measure"] == baseline["equivalent_stress_measure"]


def test_all_six_d3_numberings_transport_full_layered_bubble_response() -> None:
    (
        reference_nodes,
        current_nodes,
        owner,
        triads,
        material_direction,
        material,
        baseline_state,
        external,
    ) = _base_fixture()
    baseline_frame = _independent_frame(reference_nodes, owner)
    baseline_force, baseline_tangent, baseline_trial, baseline_meta = _response(
        reference_nodes,
        current_nodes,
        baseline_frame,
        triads,
        material_direction,
        material,
        baseline_state,
        external,
    )
    rng = np.random.default_rng(671231)
    baseline_virtual = rng.standard_normal(18)
    baseline_work = float(baseline_virtual @ baseline_force)
    baseline_quadratic_work = float(external @ baseline_tangent @ external)
    odd_permutations = 0

    for permutation in itertools.permutations(range(3)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        odd_permutations += inversions % 2
        permutation_matrix = _external_permutation(permutation)
        numbered_reference = reference_nodes[list(permutation)]
        numbered_current = current_nodes[list(permutation)]
        numbered_frame = _independent_frame(numbered_reference, owner)
        numbered_triads = np.concatenate(
            (triads[:3][list(permutation)], triads[3:]), axis=0
        )
        rotation = numbered_frame[:, :2].T @ baseline_frame[:, :2]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(2), atol=8.0e-16)
        assert np.linalg.det(rotation) > 1.0 - 8.0e-16
        engineering = _engineering_transform(rotation)
        resultant = _resultant_transform(rotation)
        np.testing.assert_allclose(
            engineering.T @ resultant, np.eye(3), rtol=0.0, atol=2.0e-15
        )
        station_map = _station_map(permutation)
        numbered_state = _numbered_state(
            baseline_state,
            station_map,
            engineering,
            resultant,
            rotation,
        )
        numbered_external = permutation_matrix @ external
        force, tangent, trial, metadata = _response(
            numbered_reference,
            numbered_current,
            numbered_frame,
            numbered_triads,
            material_direction,
            material,
            numbered_state,
            numbered_external,
        )

        expected_force = permutation_matrix @ baseline_force
        expected_tangent = (
            permutation_matrix @ baseline_tangent @ permutation_matrix.T
        )
        assert _relative_error(force, expected_force) <= 2.0e-11, permutation
        assert _relative_error(tangent, expected_tangent) <= 3.0e-11, permutation
        np.testing.assert_allclose(
            metadata["bubble_increment"],
            baseline_meta["bubble_increment"],
            rtol=2.0e-10,
            atol=2.0e-13,
            err_msg=f"bubble increment for {permutation}",
        )
        assert metadata["bubble_force_condensation_id"] == (
            baseline_meta["bubble_force_condensation_id"]
        )

        numbered_virtual = permutation_matrix @ baseline_virtual
        assert math.isclose(
            float(numbered_virtual @ force),
            baseline_work,
            rel_tol=3.0e-11,
            abs_tol=1.0e-8,
        )
        assert math.isclose(
            float(numbered_external @ tangent @ numbered_external),
            baseline_quadratic_work,
            rel_tol=3.0e-11,
            abs_tol=1.0e-8,
        )
        _assert_trial_state_transport(
            trial,
            baseline_trial,
            station_map,
            engineering,
            resultant,
            rotation,
        )

        full_increment = np.concatenate(
            (numbered_external, np.asarray(metadata["bubble_increment"]))
        )
        baseline_full_increment = np.concatenate(
            (external, np.asarray(baseline_meta["bubble_increment"]))
        )
        numbered_rotation_trial, full_increment, _numbered_store = (
            native_trial_for_increment(
                numbered_current,
                numbered_triads,
                full_increment,
            )
        )
        baseline_rotation_trial, baseline_full_increment, _baseline_store = (
            native_trial_for_increment(
                current_nodes,
                triads,
                baseline_full_increment,
            )
        )
        updated = s3._update_native_director_triads(
            numbered_triads,
            full_increment,
            native_rotation_trial=numbered_rotation_trial,
        )
        baseline_updated = s3._update_native_director_triads(
            triads,
            baseline_full_increment,
            native_rotation_trial=baseline_rotation_trial,
        )
        expected_triads = np.concatenate(
            (baseline_updated[:3][list(permutation)], baseline_updated[3:]),
            axis=0,
        )
        np.testing.assert_allclose(
            updated,
            expected_triads,
            rtol=0.0,
            atol=4.0e-13,
            err_msg=f"equilibrated director state for {permutation}",
        )

    assert odd_permutations == 3
