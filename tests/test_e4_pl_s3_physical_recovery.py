from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import (
    FEModel,
    Hill48Yield,
    OrthotropicMaterial,
    QualifiedE4PLS3ShellElement,
    RecoveryConfig,
    ResourceConfig,
    describe_result_quantities,
    recover_stress_result,
)
from anyfesolver.e4_pl_s3_element import (
    CAPABILITY_GAPS,
    PHYSICAL_EXTERNAL_INDICES,
    TRIANGLE_QUADRATURE,
    triangle_frame,
)
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.plasticity import hill48_plane_stress_equivalent_stress
from anyfesolver.recovery import recover_element_stresses_with_report
from anyfesolver.shell_sections import GeneralizedShellSection


OWNER_NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=float)


def _mesh(coordinates: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for node_id, coordinate in enumerate(np.asarray(coordinates), start=1):
        mesh.add_node(node_id, *coordinate)
    return mesh


def _isotropic() -> Material:
    return Material("steel", 210.0e9, 0.3, density=7850.0)


def _orthotropic(*, hill: bool = False) -> OrthotropicMaterial:
    hill_yield = None
    if hill:
        hill_yield = Hill48Yield(
            120.0e6,
            175.0e6,
            145.0e6,
            71.0e6,
            67.0e6,
            59.0e6,
        )
    return OrthotropicMaterial(
        name="ortho",
        elastic_modulus_1=145.0e9,
        elastic_modulus_2=12.0e9,
        elastic_modulus_3=9.0e9,
        poisson_ratio_12=0.24,
        poisson_ratio_13=0.19,
        poisson_ratio_23=0.28,
        shear_modulus_12=5.4e9,
        shear_modulus_13=4.2e9,
        shear_modulus_23=3.1e9,
        density=1650.0,
        hill_yield=hill_yield,
    )


def _element(
    *,
    material_direction: np.ndarray | None = None,
    material_angle_deg: float = 0.0,
    shell_section: GeneralizedShellSection | None = None,
    reference_normal: np.ndarray = OWNER_NORMAL,
) -> QualifiedE4PLS3ShellElement:
    return QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel" if shell_section is None else "section",
        thickness=0.12,
        material_direction=material_direction,
        material_angle_deg=material_angle_deg,
        shell_section=shell_section,
        reference_normal=reference_normal,
    )


def _surface_tensor(
    recovered: dict[str, object],
    index: int,
    surface: str,
    *,
    basis: str,
) -> np.ndarray:
    return np.asarray(
        (
            (
                recovered[f"{basis}_xx_{surface}"][index],
                recovered[f"{basis}_xy_{surface}"][index],
                recovered[f"{basis}_xz_{surface}"][index],
            ),
            (
                recovered[f"{basis}_xy_{surface}"][index],
                recovered[f"{basis}_yy_{surface}"][index],
                recovered[f"{basis}_yz_{surface}"][index],
            ),
            (
                recovered[f"{basis}_xz_{surface}"][index],
                recovered[f"{basis}_yz_{surface}"][index],
                recovered[f"{basis}_zz_{surface}"][index],
            ),
        ),
        dtype=float,
    )


def _physical_displacements(coordinates: np.ndarray) -> np.ndarray:
    centre = np.mean(coordinates, axis=0)
    translation_gradient = np.asarray(
        (
            (1.2e-4, -0.3e-4, 0.5e-4),
            (0.4e-4, -0.8e-4, 0.2e-4),
            (-0.6e-4, 0.7e-4, 1.1e-4),
        )
    )
    rotation_gradient = np.asarray(
        (
            (0.8e-4, -0.4e-4, 0.2e-4),
            (-0.3e-4, 1.0e-4, -0.5e-4),
            (0.1e-4, 0.2e-4, 0.4e-4),
        )
    )
    values = np.zeros(18, dtype=float)
    for index, coordinate in enumerate(coordinates):
        relative = coordinate - centre
        values[6 * index : 6 * index + 3] = (
            translation_gradient @ relative
            + np.asarray((0.2e-4, -0.1e-4, 0.3e-4))
        )
        values[6 * index + 3 : 6 * index + 6] = (
            rotation_gradient @ relative
            + np.asarray((-0.4e-4, 0.6e-4, 0.1e-4))
        )
    return values


def _qualified_s3_batch_model(count: int = 101) -> FEModel:
    model = FEModel("qualified-s3-recovery-batch")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    height = np.sqrt(3.0) / 2.0
    for offset in range(count):
        first_node = 3 * offset + 1
        x_offset = 2.0 * offset
        for node_id, coordinate in zip(
            (first_node, first_node + 1, first_node + 2),
            (
                (x_offset, 0.0, 0.0),
                (x_offset + 1.0, 0.0, 0.0),
                (x_offset + 0.5, height, 0.0),
            ),
        ):
            model.add_node(node_id, *coordinate)
        element_id = offset + 1
        model.add_element(
            element_id,
            QualifiedE4PLS3ShellElement(
                element_id,
                (first_node, first_node + 1, first_node + 2),
                "steel",
                thickness=0.1,
                reference_normal=OWNER_NORMAL,
            ),
        )
    return model


def _assert_recovery_payload_bytes_equal(
    actual: dict[int, dict[str, object]],
    expected: dict[int, dict[str, object]],
) -> None:
    assert list(actual) == list(expected)
    for element_id in expected:
        assert list(actual[element_id]) == list(expected[element_id])
        for key, expected_value in expected[element_id].items():
            actual_value = actual[element_id][key]
            if isinstance(expected_value, np.ndarray):
                assert isinstance(actual_value, np.ndarray)
                assert actual_value.shape == expected_value.shape
                assert actual_value.dtype == expected_value.dtype
                assert np.all(np.isfinite(actual_value))
                assert actual_value.tobytes(order="C") == expected_value.tobytes(
                    order="C"
                )
            else:
                assert actual_value == expected_value


def _manual_orthotropic_plane_stress(
    material: OrthotropicMaterial,
    angle: float,
    engineering_strain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axes = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    strain = np.asarray(engineering_strain, dtype=float)
    strain_tensor = np.asarray(
        ((strain[0], 0.5 * strain[2]), (0.5 * strain[2], strain[1]))
    )
    material_tensor = axes.T @ strain_tensor @ axes
    material_strain = np.asarray(
        (
            material_tensor[0, 0],
            material_tensor[1, 1],
            2.0 * material_tensor[0, 1],
        )
    )
    compliance = np.asarray(
        (
            (
                1.0 / material.elastic_modulus_1,
                -material.poisson_ratio_12 / material.elastic_modulus_1,
                0.0,
            ),
            (
                -material.poisson_ratio_12 / material.elastic_modulus_1,
                1.0 / material.elastic_modulus_2,
                0.0,
            ),
            (0.0, 0.0, 1.0 / material.shear_modulus_12),
        )
    )
    material_stress = np.linalg.solve(compliance, material_strain)
    material_stress_tensor = np.asarray(
        (
            (material_stress[0], material_stress[2]),
            (material_stress[2], material_stress[1]),
        )
    )
    local_tensor = axes @ material_stress_tensor @ axes.T
    return (
        np.asarray((local_tensor[0, 0], local_tensor[1, 1], local_tensor[0, 1])),
        material_stress,
    )


def test_recovery_capabilities_are_explicitly_replaced() -> None:
    element = _element()
    assert "global_recovery" not in CAPABILITY_GAPS
    assert "orthotropic_physical_recovery" not in CAPABILITY_GAPS
    assert element.capability_matrix()["global_recovery"] == "PARITY_REPLACED"
    assert element.capability_matrix()["orthotropic_physical_recovery"] == (
        "PARITY_REPLACED"
    )
    assert element.capability_matrix()["patch_recovery"] == "PARITY_REPLACED"
    assert element.capability_matrix()["committed_state_recovery"] == (
        "PARITY_REPLACED"
    )
    assert element.capability_matrix()["restart_history"] == (
        "STATIC_AND_ARC_LENGTH_CHECKPOINTS_ONLY"
    )
    assert element.recovery_errors_fail_closed is True


def test_recovery_schema_is_frozen_and_nonfinite_values_fail_closed() -> None:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, np.sqrt(3.0) / 2.0, 0.0))
    )
    mesh = _mesh(coordinates)
    displacement = _physical_displacements(coordinates)
    core = {
        "recovery_scope",
        "physical_stress_available",
        "membrane_resultant_order",
        "transverse_shear_resultant_order",
        "membrane_strain",
        "curvature",
        "transverse_shear_strain",
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
        "bubble_rotations",
        "numerical_fields_excluded",
        "reference_surface_offset",
        "reference_surface_offset_policy_id",
        "section_origin_offset_from_reference",
        "physical_bottom_offset_from_reference",
        "physical_top_offset_from_reference",
    }
    physical_provenance = {
        "bubble_linearization_policy",
        "through_thickness_stress_profile",
    }
    physical = core | physical_provenance | {
        "membrane_xx",
        "membrane_yy",
        "membrane_xy",
        "bending_xx",
        "bending_yy",
        "bending_xy",
        "shear_xz",
        "shear_yz",
        "von_mises",
        "hill_utilization",
        "equivalent_stress",
        "equivalent_stress_measure",
    }
    global_resultants = {
        "global_membrane_resultant_tensors",
        "global_bending_resultant_tensors",
        "global_transverse_shear_resultants",
    }
    surface_fields = {
        f"{basis}_{component}_{surface}"
        for basis in ("local", "global")
        for component in ("xx", "yy", "zz", "xy", "yz", "xz")
        for surface in ("top", "bot")
    }

    element = _element()
    local = element.compute_stresses(mesh, displacement, _isotropic())
    global_values = element.compute_stresses(
        mesh,
        displacement,
        _isotropic(),
        return_global=True,
    )
    assert set(local) == physical
    assert set(global_values) == physical | global_resultants | surface_fields
    assert local["bubble_linearization_policy"] == (
        "REFERENCE_ELASTIC_BUBBLE_SCHUR_DERIVATIVE_V1"
    )
    assert local["through_thickness_stress_profile"] == (
        "HOMOGENEOUS_ELASTIC_LINEAR_THROUGH_THICKNESS_V1"
    )

    for values in (local, global_values):
        for value in values.values():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value))
    for key in (
        "membrane_strain",
        "curvature",
        "membrane_resultants",
        "bending_resultants",
    ):
        assert global_values[key].shape == (7, 3)
    for key in ("transverse_shear_strain", "transverse_shear_resultants"):
        assert global_values[key].shape == (7, 2)
    assert global_values["bubble_rotations"].shape == (2,)
    for key in surface_fields | {
        "membrane_xx",
        "membrane_yy",
        "membrane_xy",
        "bending_xx",
        "bending_yy",
        "bending_xy",
        "shear_xz",
        "shear_yz",
        "von_mises",
        "hill_utilization",
        "equivalent_stress",
    }:
        assert global_values[key].shape == (7,)

    section = GeneralizedShellSection(
        A=np.diag((120.0, 95.0, 42.0)),
        B=np.zeros((3, 3)),
        D=np.diag((15.0, 11.0, 5.0)),
        As=np.diag((25.0, 20.0)),
    )
    generalized = _element(
        material_direction=np.asarray((1.0, 0.0, 0.0)),
        shell_section=section,
    ).compute_stresses(
        mesh,
        displacement,
        _isotropic(),
        return_global=True,
    )
    assert set(generalized) == core | global_resultants | {
        "generalized_stress_scope"
    }
    assert not (physical_provenance & set(generalized))

    nonfinite = displacement.copy()
    nonfinite[4] = np.nan
    with pytest.raises(ValueError, match="recovery requires finite displacements"):
        element.compute_stresses(mesh, nonfinite, _isotropic(), return_global=True)


def test_global_surface_and_resultant_tensors_are_exact_passive_rotations() -> None:
    coordinates = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)),
        dtype=float,
    )
    owner = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    owner /= np.linalg.norm(owner)
    mesh = _mesh(coordinates)
    element = _element(reference_normal=owner)
    displacement = _physical_displacements(coordinates)
    material = _isotropic()

    local = element.compute_stresses(mesh, displacement, material)
    recovered = element.compute_stresses(
        mesh,
        displacement,
        material,
        return_global=True,
    )
    frame, local_nodes, _quality = triangle_frame(coordinates, owner)
    components = element.compute_stiffness_components(mesh, material)
    local_external = element._local_dof_transform(frame) @ displacement
    expected_bubble = components["bubble_map"] @ local_external[
        PHYSICAL_EXTERNAL_INDICES
    ]

    assert recovered["recovery_scope"] == "qualified_s3_local_and_global_physical"
    np.testing.assert_allclose(
        recovered["bubble_rotations"],
        expected_bubble,
        rtol=2.0e-15,
        atol=2.0e-19,
    )
    assert np.linalg.norm(expected_bubble) > 0.0
    for key in (
        "membrane_strain",
        "curvature",
        "transverse_shear_strain",
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
        "equivalent_stress",
    ):
        np.testing.assert_array_equal(recovered[key], local[key])

    for index in range(len(TRIANGLE_QUADRATURE)):
        for surface in ("top", "bot"):
            local_tensor = _surface_tensor(recovered, index, surface, basis="local")
            global_tensor = _surface_tensor(recovered, index, surface, basis="global")
            np.testing.assert_allclose(
                global_tensor,
                frame @ local_tensor @ frame.T,
                rtol=2.0e-15,
                atol=2.0e-8,
            )
            np.testing.assert_allclose(global_tensor, global_tensor.T, rtol=0.0, atol=0.0)

        membrane = np.asarray(recovered["membrane_resultants"])[index]
        membrane_tensor = np.asarray(
            ((membrane[0], membrane[2], 0.0), (membrane[2], membrane[1], 0.0), (0.0, 0.0, 0.0))
        )
        bending = np.asarray(recovered["bending_resultants"])[index]
        bending_tensor = np.asarray(
            ((bending[0], bending[2], 0.0), (bending[2], bending[1], 0.0), (0.0, 0.0, 0.0))
        )
        shear = np.asarray(recovered["transverse_shear_resultants"])[index]
        np.testing.assert_allclose(
            recovered["global_membrane_resultant_tensors"][index],
            frame @ membrane_tensor @ frame.T,
            rtol=2.0e-15,
            atol=2.0e-9,
        )
        np.testing.assert_allclose(
            recovered["global_bending_resultant_tensors"][index],
            frame @ bending_tensor @ frame.T,
            rtol=2.0e-15,
            atol=2.0e-9,
        )
        np.testing.assert_allclose(
            recovered["global_transverse_shear_resultants"][index],
            shear[0] * frame[:, 0] + shear[1] * frame[:, 1],
            rtol=2.0e-15,
            atol=2.0e-9,
        )

    _jacobian = np.asarray(
        (
            (local_nodes[1, 0] - local_nodes[0, 0], local_nodes[1, 1] - local_nodes[0, 1]),
            (local_nodes[2, 0] - local_nodes[0, 0], local_nodes[2, 1] - local_nodes[0, 1]),
        )
    )
    determinant = abs(float(np.linalg.det(_jacobian)))
    work = 0.0
    for index, (_r, _s, weight) in enumerate(TRIANGLE_QUADRATURE):
        work += determinant * weight * (
            float(recovered["membrane_strain"][index] @ recovered["membrane_resultants"][index])
            + float(recovered["curvature"][index] @ recovered["bending_resultants"][index])
            + float(
                recovered["transverse_shear_strain"][index]
                @ recovered["transverse_shear_resultants"][index]
            )
        )
    physical = components["physical"]
    np.testing.assert_allclose(
        work,
        displacement @ physical @ displacement,
        rtol=3.0e-13,
        atol=1.0e-12,
    )
    virtual_displacement = 0.7e-4 * np.sin(np.arange(18, dtype=float) + 0.2)
    virtual = element.compute_stresses(mesh, virtual_displacement, material)
    virtual_work = 0.0
    for index, (_r, _s, weight) in enumerate(TRIANGLE_QUADRATURE):
        virtual_work += determinant * weight * (
            float(virtual["membrane_strain"][index] @ recovered["membrane_resultants"][index])
            + float(virtual["curvature"][index] @ recovered["bending_resultants"][index])
            + float(
                virtual["transverse_shear_strain"][index]
                @ recovered["transverse_shear_resultants"][index]
            )
        )
    np.testing.assert_allclose(
        virtual_work,
        virtual_displacement @ physical @ displacement,
        rtol=4.0e-13,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        components["total"],
        physical + components["pl"],
        rtol=8.0e-15,
        atol=4.0e-6,
    )


def test_orthotropic_material_axes_and_hill_measure_use_recovered_bubble_field() -> None:
    coordinates = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    material = _orthotropic(hill=True)
    direction_angle = np.deg2rad(31.0)
    added_angle = np.deg2rad(13.0)
    direction = np.asarray((np.cos(direction_angle), np.sin(direction_angle), 0.0))
    element = _element(
        material_direction=direction,
        material_angle_deg=np.degrees(added_angle),
    )
    strain = np.asarray((1.3e-4, -0.4e-4, 0.65e-4))
    displacement = np.zeros(18, dtype=float)
    for index, (x, y, _z) in enumerate(coordinates):
        displacement[6 * index] = strain[0] * x + 0.5 * strain[2] * y
        displacement[6 * index + 1] = strain[1] * y + 0.5 * strain[2] * x

    recovered = element.compute_stresses(
        _mesh(coordinates),
        displacement,
        material,
        return_global=True,
    )

    angle = direction_angle + added_angle
    expected_local, material_stress = _manual_orthotropic_plane_stress(
        material,
        angle,
        strain,
    )
    for component, expected in zip(("membrane_xx", "membrane_yy", "membrane_xy"), expected_local):
        np.testing.assert_allclose(recovered[component], expected, rtol=4.0e-13, atol=2.0e-6)

    expected_hill = hill48_plane_stress_equivalent_stress(
        material_stress.reshape(1, 3),
        material.hill_yield,
    )[0]
    np.testing.assert_allclose(recovered["equivalent_stress"], expected_hill, rtol=4.0e-13)
    np.testing.assert_allclose(
        recovered["hill_utilization"],
        expected_hill / material.hill_yield.X,
        rtol=4.0e-13,
    )
    assert recovered["equivalent_stress_measure"] == "hill48"


def test_arbitrary_orthotropic_membrane_bending_and_shear_match_independent_section_law() -> None:
    coordinates = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    material = _orthotropic(hill=True)
    direction_angle = np.deg2rad(28.0)
    added_angle = np.deg2rad(-9.0)
    direction = np.asarray((np.cos(direction_angle), np.sin(direction_angle), 0.0))
    element = _element(
        material_direction=direction,
        material_angle_deg=np.degrees(added_angle),
    )
    recovered = element.compute_stresses(
        _mesh(coordinates),
        _physical_displacements(coordinates),
        material,
        return_global=True,
    )
    angle = direction_angle + added_angle
    axes = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    shear_local = axes @ np.diag(
        (material.shear_modulus_13, material.shear_modulus_23)
    ) @ axes.T

    for index in range(len(TRIANGLE_QUADRATURE)):
        membrane, membrane_material = _manual_orthotropic_plane_stress(
            material,
            angle,
            recovered["membrane_strain"][index],
        )
        curvature_stress, curvature_material = _manual_orthotropic_plane_stress(
            material,
            angle,
            recovered["curvature"][index],
        )
        bending = 0.5 * element.thickness * curvature_stress
        transverse = (
            (5.0 / 6.0)
            * shear_local
            @ recovered["transverse_shear_strain"][index]
        )
        np.testing.assert_allclose(
            [
                recovered["membrane_xx"][index],
                recovered["membrane_yy"][index],
                recovered["membrane_xy"][index],
            ],
            membrane,
            rtol=5.0e-13,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            [
                recovered["bending_xx"][index],
                recovered["bending_yy"][index],
                recovered["bending_xy"][index],
            ],
            bending,
            rtol=5.0e-13,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            [recovered["shear_xz"][index], recovered["shear_yz"][index]],
            transverse,
            rtol=5.0e-13,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            recovered["membrane_resultants"][index],
            element.thickness * membrane,
            rtol=5.0e-13,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            recovered["bending_resultants"][index],
            (element.thickness**2 / 6.0) * bending,
            rtol=5.0e-13,
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            recovered["transverse_shear_resultants"][index],
            element.thickness * transverse,
            rtol=5.0e-13,
            atol=2.0e-7,
        )
        expected_hill = max(
            hill48_plane_stress_equivalent_stress(
                (
                    membrane_material
                    + 0.5 * element.thickness * curvature_material
                ).reshape(1, 3),
                material.hill_yield,
            )[0],
            hill48_plane_stress_equivalent_stress(
                (
                    membrane_material
                    - 0.5 * element.thickness * curvature_material
                ).reshape(1, 3),
                material.hill_yield,
            )[0],
        )
        assert recovered["equivalent_stress"][index] == pytest.approx(
            expected_hill,
            rel=5.0e-13,
        )


def _engineering_tensor(values: np.ndarray) -> np.ndarray:
    made = np.asarray(values, dtype=float)
    return np.asarray(((made[0], 0.5 * made[2]), (0.5 * made[2], made[1])))


def _pack_engineering(tensor: np.ndarray) -> np.ndarray:
    made = np.asarray(tensor, dtype=float)
    return np.asarray((made[0, 0], made[1, 1], 2.0 * made[0, 1]))


def _resultant_tensor(values: np.ndarray) -> np.ndarray:
    made = np.asarray(values, dtype=float)
    return np.asarray(((made[0], made[2]), (made[2], made[1])))


def _pack_resultant(tensor: np.ndarray) -> np.ndarray:
    made = np.asarray(tensor, dtype=float)
    return np.asarray((made[0, 0], made[1, 1], made[0, 1]))


def _assert_normalized_covariance(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    *,
    limit: float = 1.0e-12,
) -> None:
    made = np.asarray(actual, dtype=float)
    reference = np.asarray(expected, dtype=float)
    scale = max(
        float(np.max(np.abs(made))),
        float(np.max(np.abs(reference))),
        np.finfo(float).tiny,
    )
    assert float(np.max(np.abs(made - reference))) / scale <= limit


def test_all_six_d3_numberings_preserve_global_orthotropic_recovery() -> None:
    coordinates = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)),
        dtype=float,
    )
    owner = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    owner /= np.linalg.norm(owner)
    physical_direction = coordinates[1] - coordinates[0]
    physical_direction /= np.linalg.norm(physical_direction)
    material = _orthotropic(hill=True)

    barycentric_stations = np.asarray(
        [(1.0 - r - s, r, s) for r, s, _weight in TRIANGLE_QUADRATURE]
    )
    baseline_recovered = None
    baseline_frame = None
    for permutation in itertools.permutations(range(3)):
        numbered = coordinates[list(permutation)]
        element = _element(
            material_direction=physical_direction,
            material_angle_deg=17.0,
            reference_normal=owner,
        )
        recovered = element._compute_stresses(
            _mesh(numbered),
            _physical_displacements(numbered),
            material,
            return_global=True,
            enforce_positive_winding=False,
        )
        frame, _local, _quality = triangle_frame(numbered, owner)
        if baseline_recovered is None:
            assert permutation == (0, 1, 2)
            baseline_recovered = recovered
            baseline_frame = frame
            continue
        assert baseline_frame is not None
        in_plane = frame[:, :2].T @ baseline_frame[:, :2]
        full = frame.T @ baseline_frame
        np.testing.assert_allclose(in_plane.T @ in_plane, np.eye(2), atol=5.0e-15)
        assert np.linalg.det(in_plane) == pytest.approx(1.0, abs=5.0e-15)
        np.testing.assert_allclose(full[:2, :2], in_plane, atol=5.0e-15)
        np.testing.assert_allclose(full[:2, 2], 0.0, atol=5.0e-15)
        np.testing.assert_allclose(full[2, :2], 0.0, atol=5.0e-15)
        assert full[2, 2] == pytest.approx(1.0, abs=5.0e-15)

        strain_transform = np.column_stack(
            [
                _pack_engineering(
                    in_plane @ _engineering_tensor(np.eye(3)[column]) @ in_plane.T
                )
                for column in range(3)
            ]
        )
        resultant_transform = np.column_stack(
            [
                _pack_resultant(
                    in_plane @ _resultant_tensor(np.eye(3)[column]) @ in_plane.T
                )
                for column in range(3)
            ]
        )
        np.testing.assert_allclose(
            strain_transform.T @ resultant_transform,
            np.eye(3),
            rtol=0.0,
            atol=2.0e-14,
        )
        _assert_normalized_covariance(
            recovered["bubble_rotations"],
            in_plane @ baseline_recovered["bubble_rotations"],
        )

        for new_index, new_barycentric in enumerate(barycentric_stations):
            base_barycentric = np.zeros(3, dtype=float)
            base_barycentric[np.asarray(permutation)] = new_barycentric
            distances = np.max(
                np.abs(barycentric_stations - base_barycentric[None, :]),
                axis=1,
            )
            base_index = int(np.argmin(distances))
            assert distances[base_index] <= 5.0e-15

            for strain_key in ("membrane_strain", "curvature"):
                expected = _pack_engineering(
                    in_plane
                    @ _engineering_tensor(
                        baseline_recovered[strain_key][base_index]
                    )
                    @ in_plane.T
                )
                _assert_normalized_covariance(
                    recovered[strain_key][new_index],
                    expected,
                )
            for resultant_key in ("membrane_resultants", "bending_resultants"):
                expected = _pack_resultant(
                    in_plane
                    @ _resultant_tensor(
                        baseline_recovered[resultant_key][base_index]
                    )
                    @ in_plane.T
                )
                _assert_normalized_covariance(
                    recovered[resultant_key][new_index],
                    expected,
                )
            for strain_key, resultant_key in (
                ("membrane_strain", "membrane_resultants"),
                ("curvature", "bending_resultants"),
            ):
                _assert_normalized_covariance(
                    recovered[strain_key][new_index]
                    @ recovered[resultant_key][new_index],
                    baseline_recovered[strain_key][base_index]
                    @ baseline_recovered[resultant_key][base_index],
                )
            for vector_key in (
                "transverse_shear_strain",
                "transverse_shear_resultants",
            ):
                _assert_normalized_covariance(
                    recovered[vector_key][new_index],
                    in_plane @ baseline_recovered[vector_key][base_index],
                )
            _assert_normalized_covariance(
                recovered["transverse_shear_strain"][new_index]
                @ recovered["transverse_shear_resultants"][new_index],
                baseline_recovered["transverse_shear_strain"][base_index]
                @ baseline_recovered["transverse_shear_resultants"][base_index],
            )

            for surface in ("top", "bot"):
                _assert_normalized_covariance(
                    _surface_tensor(recovered, new_index, surface, basis="local"),
                    full
                    @ _surface_tensor(
                        baseline_recovered,
                        base_index,
                        surface,
                        basis="local",
                    )
                    @ full.T,
                )
                _assert_normalized_covariance(
                    _surface_tensor(recovered, new_index, surface, basis="global"),
                    _surface_tensor(
                        baseline_recovered,
                        base_index,
                        surface,
                        basis="global",
                    ),
                )
            for global_key in (
                "global_membrane_resultant_tensors",
                "global_bending_resultant_tensors",
                "global_transverse_shear_resultants",
                "equivalent_stress",
            ):
                _assert_normalized_covariance(
                    recovered[global_key][new_index],
                    baseline_recovered[global_key][base_index],
                )


def test_proper_global_rotation_and_translation_transport_recovery() -> None:
    coordinates = np.asarray(
        ((0.1, 0.2, 0.3), (1.2, 0.1, 0.5), (0.3, 1.0, 0.4)),
        dtype=float,
    )
    owner = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    owner /= np.linalg.norm(owner)
    direction = coordinates[1] - coordinates[0]
    direction /= np.linalg.norm(direction)
    material = _orthotropic(hill=True)
    displacement = _physical_displacements(coordinates)
    baseline = _element(
        material_direction=direction,
        material_angle_deg=19.0,
        reference_normal=owner,
    ).compute_stresses(
        _mesh(coordinates),
        displacement,
        material,
        return_global=True,
    )

    axis = np.asarray((0.3, -0.7, 0.6), dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    translated = coordinates @ rotation.T + np.asarray((2.4, -1.1, 0.8))
    rotated_displacement = np.zeros_like(displacement)
    for node in range(3):
        rotated_displacement[6 * node : 6 * node + 3] = (
            rotation @ displacement[6 * node : 6 * node + 3]
        )
        rotated_displacement[6 * node + 3 : 6 * node + 6] = (
            rotation @ displacement[6 * node + 3 : 6 * node + 6]
        )
    actual = _element(
        material_direction=rotation @ direction,
        material_angle_deg=19.0,
        reference_normal=rotation @ owner,
    ).compute_stresses(
        _mesh(translated),
        rotated_displacement,
        material,
        return_global=True,
    )

    for key in (
        "membrane_strain",
        "curvature",
        "transverse_shear_strain",
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
        "equivalent_stress",
        "von_mises",
        "hill_utilization",
    ):
        np.testing.assert_allclose(actual[key], baseline[key], rtol=8.0e-13, atol=2.0e-7)
    for index in range(len(TRIANGLE_QUADRATURE)):
        for surface in ("top", "bot"):
            np.testing.assert_allclose(
                _surface_tensor(actual, index, surface, basis="global"),
                rotation
                @ _surface_tensor(baseline, index, surface, basis="global")
                @ rotation.T,
                rtol=8.0e-13,
                atol=2.0e-5,
            )
        for key in (
            "global_membrane_resultant_tensors",
            "global_bending_resultant_tensors",
        ):
            np.testing.assert_allclose(
                actual[key][index],
                rotation @ baseline[key][index] @ rotation.T,
                rtol=8.0e-13,
                atol=2.0e-6,
            )
        np.testing.assert_allclose(
            actual["global_transverse_shear_resultants"][index],
            rotation @ baseline["global_transverse_shear_resultants"][index],
            rtol=8.0e-13,
            atol=2.0e-6,
        )


def test_drill_only_motion_changes_pl_but_no_physical_recovery_field() -> None:
    coordinates = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    mesh = _mesh(coordinates)
    element = _element()
    displacement = np.zeros(18, dtype=float)
    displacement[5::6] = np.asarray((0.2, -0.1, 0.05))
    recovered = element.compute_stresses(
        mesh,
        displacement,
        _isotropic(),
        return_global=True,
    )
    for key in (
        "membrane_strain",
        "curvature",
        "transverse_shear_strain",
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
        "equivalent_stress",
        "von_mises",
        "global_membrane_resultant_tensors",
        "global_bending_resultant_tensors",
        "global_transverse_shear_resultants",
    ):
        np.testing.assert_array_equal(recovered[key], np.zeros_like(recovered[key]))
    numerical = element.numerical_internal_force(displacement)
    assert np.linalg.norm(numerical["pl"]) > 0.0
    np.testing.assert_array_equal(numerical["numerical"], numerical["pl"])


def test_generalized_section_global_resultants_remain_nonphysical() -> None:
    coordinates = np.asarray(((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (0.2, 0.95, 0.0)))
    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, 0.0), (0.1, -0.8, 0.1), (0.0, 0.1, 0.3))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.asarray(((25.0, 2.0), (2.0, 20.0))),
    )
    element = _element(
        material_direction=np.asarray((1.0, 0.2, 0.0)),
        shell_section=section,
    )
    mesh = _mesh(coordinates)
    displacement = _physical_displacements(coordinates)
    recovered = element.compute_stresses(
        mesh,
        displacement,
        _isotropic(),
        return_global=True,
    )
    assert recovered["recovery_scope"] == "section_resultants_only"
    assert recovered["generalized_stress_scope"] == "section_resultants_only"
    assert recovered["physical_stress_available"] is False
    assert "membrane_xx" not in recovered
    assert "global_xx_top" not in recovered
    assert recovered["global_membrane_resultant_tensors"].shape == (7, 3, 3)
    assert recovered["global_bending_resultant_tensors"].shape == (7, 3, 3)
    assert recovered["global_transverse_shear_resultants"].shape == (7, 3)

    frame, local_nodes, _quality = triangle_frame(coordinates, OWNER_NORMAL)
    for index in range(len(TRIANGLE_QUADRATURE)):
        membrane = np.asarray(recovered["membrane_resultants"])[index]
        bending = np.asarray(recovered["bending_resultants"])[index]
        shear = np.asarray(recovered["transverse_shear_resultants"])[index]
        membrane_tensor = np.asarray(
            (
                (membrane[0], membrane[2], 0.0),
                (membrane[2], membrane[1], 0.0),
                (0.0, 0.0, 0.0),
            )
        )
        bending_tensor = np.asarray(
            (
                (bending[0], bending[2], 0.0),
                (bending[2], bending[1], 0.0),
                (0.0, 0.0, 0.0),
            )
        )
        np.testing.assert_allclose(
            recovered["global_membrane_resultant_tensors"][index],
            frame @ membrane_tensor @ frame.T,
            rtol=2.0e-15,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            recovered["global_bending_resultant_tensors"][index],
            frame @ bending_tensor @ frame.T,
            rtol=2.0e-15,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            recovered["global_transverse_shear_resultants"][index],
            shear[0] * frame[:, 0] + shear[1] * frame[:, 1],
            rtol=2.0e-15,
            atol=2.0e-13,
        )
    jacobian = np.asarray(
        (
            (local_nodes[1, 0] - local_nodes[0, 0], local_nodes[1, 1] - local_nodes[0, 1]),
            (local_nodes[2, 0] - local_nodes[0, 0], local_nodes[2, 1] - local_nodes[0, 1]),
        )
    )
    determinant = abs(float(np.linalg.det(jacobian)))
    work = sum(
        determinant
        * weight
        * (
            float(recovered["membrane_strain"][index] @ recovered["membrane_resultants"][index])
            + float(recovered["curvature"][index] @ recovered["bending_resultants"][index])
            + float(
                recovered["transverse_shear_strain"][index]
                @ recovered["transverse_shear_resultants"][index]
            )
        )
        for index, (_r, _s, weight) in enumerate(TRIANGLE_QUADRATURE)
    )
    physical = element.compute_stiffness_components(mesh, _isotropic())["physical"]
    np.testing.assert_allclose(
        work,
        displacement @ physical @ displacement,
        rtol=3.0e-13,
        atol=1.0e-13,
    )

    model = FEModel("qualified-s3-generalized-recovery")
    model.add_material("section", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(coordinates, start=1):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            (1, 2, 3),
            "section",
            thickness=0.12,
            material_direction=np.asarray((1.0, 0.2, 0.0)),
            shell_section=section,
            reference_normal=OWNER_NORMAL,
        ),
    )
    public_result = recover_stress_result(
        model,
        displacement,
        RecoveryConfig(element_ids=[1]),
        return_global=True,
    )
    public_recovered = public_result.element_stresses[1]
    for key in (
        "membrane_strain",
        "curvature",
        "transverse_shear_strain",
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
        "global_membrane_resultant_tensors",
        "global_bending_resultant_tensors",
        "global_transverse_shear_resultants",
    ):
        np.testing.assert_array_equal(public_recovered[key], recovered[key])
    physical_recovered = _element().compute_stresses(
        _mesh(coordinates),
        displacement,
        _isotropic(),
        return_global=True,
    )
    quantities = describe_result_quantities(
        SimpleNamespace(
            element_stresses={10: physical_recovered, 20: public_recovered}
        )
    )
    stress = next(item for item in quantities if item.quantity_id == "stress")
    assert stress.metadata["excluded_nonphysical_element_ids"] == (20,)
    assert "global_xx_top" in stress.components


def test_global_recovery_flows_through_public_result_and_quantity_contracts() -> None:
    coordinates = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    model = FEModel("qualified-s3-global-recovery")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(coordinates, start=1):
        model.add_node(node_id, *coordinate)
    model.add_element(1, _element())
    displacement = _physical_displacements(coordinates)

    result = recover_stress_result(
        model,
        displacement,
        RecoveryConfig(element_ids=[1]),
        return_global=True,
    )
    recovered = result.element_stresses[1]
    assert recovered["physical_stress_available"] is True
    assert recovered["global_xx_top"].shape == (7,)
    quantities = describe_result_quantities(
        SimpleNamespace(element_stresses=result.element_stresses)
    )
    stress = next(item for item in quantities if item.quantity_id == "stress")
    assert "global_xx_top" in stress.components
    assert stress.metadata["component_basis"]["global_xx_top"] == "global"
    assert stress.metadata["excluded_nonphysical_element_ids"] == ()


def test_large_s3_recovery_is_byte_deterministic_across_scalar_schedulers() -> None:
    serial_model = _qualified_s3_batch_model()
    threaded_model = _qualified_s3_batch_model()
    threaded_repeat_model = _qualified_s3_batch_model()
    total_dofs = serial_model.mesh.dof_manager.total_dofs
    assert threaded_model.mesh.dof_manager.total_dofs == total_dofs
    assert threaded_repeat_model.mesh.dof_manager.total_dofs == total_dofs
    displacement = 2.0e-5 * np.sin(np.arange(total_dofs, dtype=float) + 0.17)

    serial, serial_report = recover_element_stresses_with_report(
        serial_model,
        displacement,
        RecoveryConfig(),
        return_global=True,
        resource_config=ResourceConfig(recovery_threads=1),
    )
    threaded, threaded_report = recover_element_stresses_with_report(
        threaded_model,
        displacement,
        RecoveryConfig(),
        return_global=True,
        resource_config=ResourceConfig(recovery_threads=4),
    )
    threaded_repeat, threaded_repeat_report = recover_element_stresses_with_report(
        threaded_repeat_model,
        displacement,
        RecoveryConfig(),
        return_global=True,
        resource_config=ResourceConfig(recovery_threads=4),
    )

    assert list(serial) == list(range(1, 102))
    _assert_recovery_payload_bytes_equal(threaded, serial)
    _assert_recovery_payload_bytes_equal(threaded_repeat, threaded)
    assert serial[1]["global_xx_top"].shape == (7,)

    retained_bytes = 101 * 18 * np.dtype(np.intp).itemsize
    for report, workers, backend, recovery_backend, chunks in (
        (serial_report, 1, "serial", "scalar_serial", 1),
        (
            threaded_report,
            4,
            "thread_pool",
            "scalar_chunk_thread_pool",
            12,
        ),
        (
            threaded_repeat_report,
            4,
            "thread_pool",
            "scalar_chunk_thread_pool",
            12,
        ),
    ):
        assert report.item_count == 101
        assert report.requested_workers == workers
        assert report.used_workers == workers
        assert report.backend == backend
        assert report.reason == (
            "serial: recovery_threads not requested"
            if workers == 1
            else "thread_pool"
        )
        assert report.elapsed_seconds >= 0.0
        metadata = report.metadata
        assert metadata["recovery_backend"] == recovery_backend
        assert metadata["batch_counts"] == {"shell_t3_isotropic": 101}
        assert metadata["compiled_batch_count"] == 0
        assert metadata["eligible_element_count"] == 0
        assert metadata["fallback_element_count"] == 101
        assert metadata["fallback_reasons"] == {
            "unsupported_formulation": list(range(1, 102))
        }
        assert metadata["chunk_count"] == chunks
        assert metadata["plan_reused"] is False
        assert metadata["plan_retained_bytes"] == retained_bytes
        assert metadata["plan_setup_seconds"] >= 0.0
        assert metadata["plan_lookup_seconds"] >= 0.0

    for report in (threaded_report, threaded_repeat_report):
        native_policy = report.metadata["native_thread_policy"]
        assert native_policy["phase"] == "stress_recovery_thread_pool"
        assert native_policy["requested_threads"] == 1
        assert native_policy["restored"] is True

    invalid_model = _qualified_s3_batch_model()
    invalid_displacement = displacement.copy()
    invalid_displacement[0] = np.nan
    with pytest.raises(ValueError, match="recovery requires finite displacements"):
        recover_element_stresses_with_report(
            invalid_model,
            invalid_displacement,
            RecoveryConfig(),
            return_global=True,
            resource_config=ResourceConfig(recovery_threads=4),
        )
    with pytest.raises(ValueError, match="complete in-range DOF mapping"):
        recover_element_stresses_with_report(
            _qualified_s3_batch_model(),
            np.zeros(1),
            RecoveryConfig(),
            return_global=True,
            resource_config=ResourceConfig(recovery_threads=4),
        )
