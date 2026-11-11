from __future__ import annotations

import math

import numpy as np
import pytest

from anyfesolver.e4_pl_element import (
    DIRECTOR_POLARITY_POLICY_ID,
    DIRECTOR_REVERSAL_TRANSFORM_ID,
    IMPLEMENTATION_ID,
    Q4_ACTIVITY_DISPOSITION_SCHEMA_ID,
    Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID,
    Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
    Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
    Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
    Q4_DELETED_FROZEN_POLICY_ID,
    Q4_FAILED_STATE_POLICY_ID,
    Q4_QUADRATURE_AUTHORITY_ID,
    RECOVERY_POLICY_ID,
    STATIONARY_SOLVE_POLICY_ID,
    QualifiedE4PLShellElement,
    QualifiedQ4MigrationWarning,
    equation7_frame,
)
from anyfesolver.elements import LegacyShellElement, shell_element_from_dict
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.materials import OrthotropicMaterial
from anyfesolver.shell_sections import GeneralizedShellSection


GAUSS = tuple(
    (r, s)
    for r, s in (
        (-1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
        (1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
        (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
        (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
    )
)
D4 = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (1, 0, 3, 2),
    (3, 2, 1, 0),
    (0, 3, 2, 1),
    (2, 1, 0, 3),
)
PATCH_STRAIN = np.asarray(
    (2.0, 4.0 / 3.0, -1.0 / 15.0, 2.0 / 5.0, -1.0 / 3.0, 3.0 / 7.0, 2.0 / 3.0, -1.0 / 4.0),
    dtype=float,
)


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for identifier, coordinate in enumerate(np.asarray(nodes, dtype=float), start=1):
        mesh.add_node(identifier, *coordinate)
    return mesh


def _material() -> Material:
    return Material("recovery", 15.0, 0.25, density=7.85)


def _orthotropic() -> OrthotropicMaterial:
    return OrthotropicMaterial(
        name="oriented",
        elastic_modulus_1=150.0,
        elastic_modulus_2=12.0,
        elastic_modulus_3=10.0,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0,
        shear_modulus_13=4.0,
        shear_modulus_23=3.8,
        density=1.0,
    )


def _combined_patch(nodes: np.ndarray) -> np.ndarray:
    result = np.zeros(24, dtype=float)
    for node, (x, y, _z) in enumerate(np.asarray(nodes, dtype=float)):
        base = 6 * node
        result[base] = 2.0 * x + y / 3.0
        result[base + 1] = -2.0 * x / 5.0 + 4.0 * y / 3.0
        result[base + 2] = -x * x / 5.0 + y * y / 6.0 - 3.0 * x * y / 14.0
        result[base + 3] = y / 3.0 - 3.0 * x / 14.0 + 1.0 / 4.0
        result[base + 4] = 2.0 * x / 5.0 + 3.0 * y / 14.0 + 2.0 / 3.0
        result[base + 5] = -11.0 / 30.0
    return result


def _rotate_dofs(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(4, 6).copy()
    result[:, :3] = result[:, :3] @ rotation.T
    result[:, 3:] = result[:, 3:] @ rotation.T
    return result.reshape(24)


def test_stationary_equilibrium_and_recovered_resultants_close_virtual_work() -> None:
    nodes = np.asarray(
        ((-0.2, -0.1, 0.0), (1.3, 0.0, 0.0), (1.1, 0.9, 0.0), (-0.1, 0.8, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "recovery", thickness=0.6)
    displacement = np.linspace(-0.017, 0.023, 24)
    virtual = np.cos(np.linspace(0.2, 2.1, 24)) * 0.013
    actual = element._recover_planar_mixed_fields(mesh, displacement, material, GAUSS)
    variation = element._recover_planar_mixed_fields(mesh, virtual, material, GAUSS)

    residual_scale = max(
        float(
            np.linalg.norm(actual["stationary_matrix"], ord=np.inf)
            * np.linalg.norm(actual["stationary_parameters"], ord=np.inf)
        ),
        float(
            np.linalg.norm(actual["stationary_coupling"], ord=np.inf)
            * np.linalg.norm(actual["local_displacement"], ord=np.inf)
        ),
        1.0,
    )
    assert np.linalg.norm(actual["stationarity_residual"], ord=np.inf) <= 2.0e-12 * residual_scale

    recovered_work = float(
        np.sum(
            actual["jacobian_determinants"]
            * np.einsum("ij,ij->i", variation["compatible"], actual["resultants"])
        )
    )
    components = element.compute_stiffness_components(mesh, material)
    condensed_work = float(virtual @ np.asarray(components["physical"]) @ displacement)
    assert recovered_work == pytest.approx(
        condensed_work,
        rel=2.0e-12,
        abs=2.0e-12,
    )


def test_combined_patch_recovery_is_exact_at_gauss_and_arbitrary_common_points() -> None:
    nodes = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "recovery", thickness=2.0 / 3.0)
    displacement = _combined_patch(nodes)
    arbitrary = ((-0.85, -0.65), (0.0, 0.0), (0.75, -0.2), (-0.4, 0.9))
    mixed = element._recover_planar_mixed_fields(mesh, displacement, material, arbitrary)
    expected_resultant = np.asarray(mixed["constitutive"]) @ PATCH_STRAIN
    np.testing.assert_allclose(
        mixed["compatible"],
        np.broadcast_to(PATCH_STRAIN, mixed["compatible"].shape),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        mixed["independent"],
        np.broadcast_to(PATCH_STRAIN, mixed["independent"].shape),
        rtol=0.0,
        atol=8.0e-13,
    )
    np.testing.assert_allclose(
        mixed["resultants"],
        np.broadcast_to(expected_resultant, mixed["resultants"].shape),
        rtol=0.0,
        atol=8.0e-12,
    )

    recovered = element.compute_stresses(mesh, displacement, material, return_global=True)
    public_resultants = np.column_stack(
        (
            recovered["membrane_resultants"],
            recovered["bending_resultants"],
            recovered["transverse_shear_resultants"],
        )
    )
    np.testing.assert_allclose(
        public_resultants,
        np.broadcast_to(expected_resultant, public_resultants.shape),
        rtol=0.0,
        atol=8.0e-12,
    )
    assert recovered["numerical_fields_excluded"] is True
    assert recovered["recovery_scope"] == "qualified_q4_local_and_global_physical"


def test_generalized_section_keeps_resultants_only_schema_with_mixed_fields() -> None:
    nodes = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=float,
    )
    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, 0.0), (0.1, -0.8, 0.1), (0.0, 0.1, 0.3))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.diag((25.0, 20.0)),
        mass_per_area=8.0,
        rotary_inertia_per_area=0.02,
    )
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "recovery",
        shell_section=section,
        reference_normal=equation7_frame(nodes)[0][:, 2],
    )
    recovered = element.compute_stresses(
        _mesh(nodes),
        _combined_patch(nodes),
        _material(),
        return_global=True,
    )
    expected = np.block([[section.A, section.B], [section.B.T, section.D]])
    expected_resultant = np.concatenate(
        (expected @ PATCH_STRAIN[:6], section.As @ PATCH_STRAIN[6:])
    )
    resultants = np.column_stack(
        (
            recovered["membrane_resultants"],
            recovered["bending_resultants"],
            recovered["transverse_shear_resultants"],
        )
    )
    np.testing.assert_allclose(
        resultants,
        np.broadcast_to(expected_resultant, resultants.shape),
        rtol=0.0,
        atol=8.0e-11,
    )
    assert recovered["recovery_scope"] == "section_resultants_only"
    assert recovered["generalized_stress_scope"] == "section_resultants_only"
    assert recovered["physical_stress_available"] is False
    assert "von_mises" not in recovered
    assert recovered["global_membrane_resultant_tensors"].shape == (4, 3, 3)


def test_b_coupled_section_requires_persistent_director_and_restart_identity() -> None:
    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, 0.0), (0.1, -0.8, 0.1), (0.0, 0.1, 0.3))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.diag((25.0, 20.0)),
    )
    with pytest.raises(ValueError, match="authoritative reference_normal"):
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "recovery",
            shell_section=section,
        )

    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "recovery",
        shell_section=section,
        reference_normal=np.asarray((0.0, 0.0, 2.0)),
        director_polarity=-1,
    )
    payload = element.to_dict()
    assert IMPLEMENTATION_ID == "E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V7"
    assert RECOVERY_POLICY_ID == (
        "Q4_HYBRID_PLANAR_STATIONARY_WARPED_VARYING_FRAME_"
        "PHYSICAL_DIRECTOR_RECOVERY_V3"
    )
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["recovery_policy_id"] == RECOVERY_POLICY_ID
    assert payload["stationary_solve_policy_id"] == STATIONARY_SOLVE_POLICY_ID
    assert payload["director_polarity_policy_id"] == DIRECTOR_POLARITY_POLICY_ID
    assert payload["director_reversal_transform_id"] == DIRECTOR_REVERSAL_TRANSFORM_ID
    assert payload["current_state_binding_schema_id"] == (
        Q4_CURRENT_STATE_BINDING_SCHEMA_ID
    )
    assert payload["current_state_algorithmic_origin_schema_id"] == (
        Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
    )
    assert payload["current_state_tangent_decomposition_policy_id"] == (
        Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
    )
    assert payload["current_state_projection_policy_id"] == (
        Q4_CURRENT_STATE_PROJECTION_POLICY_ID
    )
    assert payload["activity_disposition_schema_id"] == (
        Q4_ACTIVITY_DISPOSITION_SCHEMA_ID
    )
    assert payload["deleted_frozen_policy_id"] == Q4_DELETED_FROZEN_POLICY_ID
    assert payload["failed_state_policy_id"] == Q4_FAILED_STATE_POLICY_ID
    assert payload["quadrature_authority_id"] == Q4_QUADRATURE_AUTHORITY_ID
    assert payload["reference_normal"] == [0.0, 0.0, 1.0]
    assert payload["director_polarity"] == -1
    rebuilt = QualifiedE4PLShellElement.from_dict(payload)
    assert rebuilt.to_dict() == payload
    public_rebuilt = shell_element_from_dict(payload)
    assert type(public_rebuilt) is QualifiedE4PLShellElement
    assert public_rebuilt.to_dict() == payload

    mutated = dict(payload, recovery_policy_id="WRONG")
    with pytest.raises(ValueError, match="recovery_policy_id"):
        QualifiedE4PLShellElement.from_dict(mutated)
    for policy_name in (
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
    ):
        incompatible = dict(payload, **{policy_name: "WRONG"})
        with pytest.raises(ValueError, match=policy_name):
            QualifiedE4PLShellElement.from_dict(incompatible)

    pre_activity = dict(payload)
    for policy_name in (
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
    ):
        pre_activity.pop(policy_name)
    with pytest.warns(QualifiedQ4MigrationWarning, match="activity-disposition"):
        migrated_v7 = QualifiedE4PLShellElement.from_dict(pre_activity)
    assert migrated_v7.to_dict() == payload
    partial_activity = dict(payload)
    partial_activity.pop("failed_state_policy_id")
    with pytest.raises(ValueError, match="identity is incomplete"):
        QualifiedE4PLShellElement.from_dict(partial_activity)

    pre_quadrature = dict(payload)
    pre_quadrature.pop("quadrature_authority_id")
    with pytest.warns(
        QualifiedQ4MigrationWarning,
        match="immutable exact quadrature authority",
    ):
        migrated_quadrature = QualifiedE4PLShellElement.from_dict(pre_quadrature)
    assert migrated_quadrature.to_dict() == payload
    incompatible_quadrature = dict(
        payload,
        quadrature_authority_id="MUTATED_Q4_QUADRATURE",
    )
    with pytest.raises(ValueError, match="quadrature_authority_id"):
        QualifiedE4PLShellElement.from_dict(incompatible_quadrature)

    for interim_name, interim_value in (
        (
            "implementation_id",
            "E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V4",
        ),
        (
            "recovery_policy_id",
            "Q4_PLANAR_STATIONARY_PHYSICAL_DIRECTOR_RECOVERY_V2",
        ),
    ):
        interim = dict(payload)
        interim[interim_name] = interim_value
        with pytest.raises(ValueError, match=interim_name):
            QualifiedE4PLShellElement.from_dict(interim)

    for missing_director_state in ("director_polarity", "reference_normal"):
        incomplete = dict(payload)
        incomplete.pop(missing_director_state)
        with pytest.raises(ValueError, match="current director state is incomplete"):
            QualifiedE4PLShellElement.from_dict(incomplete)

    previous = dict(payload)
    previous["implementation_id"] = (
        "E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V5"
    )
    for key in (
        "current_state_binding_schema_id",
        "current_state_algorithmic_origin_schema_id",
        "current_state_tangent_decomposition_policy_id",
        "current_state_projection_policy_id",
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
    ):
        previous.pop(key)
    with pytest.warns(QualifiedQ4MigrationWarning, match="exact qualified Q4 V5"):
        migrated_v5 = QualifiedE4PLShellElement.from_dict(previous)
    assert migrated_v5.to_dict() == payload

    previous_v6 = dict(payload)
    previous_v6["implementation_id"] = (
        "E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V6"
    )
    previous_v6.pop("current_state_algorithmic_origin_schema_id")
    previous_v6["current_state_binding_schema_id"] = (
        "E4_PL_Q4_COMMITTED_STATE_DISPLACEMENT_BINDING_V1"
    )
    previous_v6["current_state_tangent_decomposition_policy_id"] = (
        "Q4_VON_KARMAN_ALGORITHMIC_MATERIAL_PLUS_MEMBRANE_STRESS_HESSIAN_V1"
    )
    for key in (
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
    ):
        previous_v6.pop(key)
    with pytest.warns(QualifiedQ4MigrationWarning, match="exact qualified Q4 V6"):
        migrated_v6 = QualifiedE4PLShellElement.from_dict(previous_v6)
    assert migrated_v6.to_dict() == payload

    legacy_coupled = dict(payload)
    for key in (
        "implementation_id",
        "recovery_policy_id",
        "stationary_solve_policy_id",
        "director_polarity_policy_id",
        "director_reversal_transform_id",
        "current_state_binding_schema_id",
        "current_state_algorithmic_origin_schema_id",
        "current_state_tangent_decomposition_policy_id",
        "current_state_projection_policy_id",
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
        "reference_normal",
        "director_polarity",
    ):
        legacy_coupled.pop(key)
    with pytest.raises(ValueError, match="cannot be migrated"):
        QualifiedE4PLShellElement.from_dict(legacy_coupled)

    legacy_uncoupled = QualifiedE4PLShellElement(
        2,
        [1, 2, 3, 4],
        "recovery",
        thickness=0.2,
    ).to_dict()
    for key in (
        "implementation_id",
        "recovery_policy_id",
        "stationary_solve_policy_id",
        "director_polarity_policy_id",
        "director_reversal_transform_id",
        "current_state_binding_schema_id",
        "current_state_algorithmic_origin_schema_id",
        "current_state_tangent_decomposition_policy_id",
        "current_state_projection_policy_id",
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
        "reference_normal",
        "director_polarity",
    ):
        legacy_uncoupled.pop(key)
    with pytest.warns(QualifiedQ4MigrationWarning, match="pre-policy uncoupled"):
        migrated = QualifiedE4PLShellElement.from_dict(legacy_uncoupled)
    assert migrated.reference_normal is None
    assert migrated.to_dict()["implementation_id"] == IMPLEMENTATION_ID


@pytest.mark.parametrize(
    "retained_marker",
    (
        "director_polarity",
        "director_polarity_policy_id",
        "director_reversal_transform_id",
        "implementation_id",
        "recovery_policy_id",
        "reference_normal",
        "stationary_solve_policy_id",
        "current_state_binding_schema_id",
        "current_state_algorithmic_origin_schema_id",
        "current_state_tangent_decomposition_policy_id",
        "current_state_projection_policy_id",
        "activity_disposition_schema_id",
        "deleted_frozen_policy_id",
        "failed_state_policy_id",
        "quadrature_authority_id",
    ),
)
def test_qualified_q4_fingerprint_cannot_downgrade_to_legacy_without_formulation_id(
    retained_marker: str,
) -> None:
    payload = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "recovery",
        thickness=0.2,
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    ).to_dict()
    stripped = {
        "type": "ShellElement",
        "element_id": payload["element_id"],
        "node_ids": payload["node_ids"],
        "material_name": payload["material_name"],
        "thickness": payload["thickness"],
        retained_marker: payload[retained_marker],
    }

    with pytest.raises(
        ValueError,
        match=(
            "retains qualified Q4 fingerprint.*missing formulation_id.*"
            + retained_marker
        ),
    ):
        shell_element_from_dict(stripped)


def test_marker_free_historical_q4_record_still_loads_as_legacy() -> None:
    historical = {
        "type": "ShellElement",
        "element_id": 2,
        "node_ids": [1, 2, 3, 4],
        "material_name": "recovery",
        "thickness": 0.2,
    }

    rebuilt = shell_element_from_dict(historical)

    assert type(rebuilt) is LegacyShellElement


def test_b_coupled_stiffness_and_physical_recovery_are_all_d4_covariant() -> None:
    nodes = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=float,
    )
    base_frame = equation7_frame(nodes)[0]
    section = GeneralizedShellSection(
        A=np.asarray(((120.0, 12.0, 4.0), (12.0, 95.0, -3.0), (4.0, -3.0, 42.0))),
        B=np.asarray(((1.2, 0.1, -0.2), (0.05, -0.8, 0.1), (0.3, -0.15, 0.4))),
        D=np.asarray(((15.0, 1.0, 0.2), (1.0, 11.0, -0.1), (0.2, -0.1, 5.0))),
        As=np.asarray(((25.0, 1.5), (1.5, 20.0))),
    )
    material = _material()
    displacement = _combined_patch(nodes)

    def make(numbered_nodes: np.ndarray) -> QualifiedE4PLShellElement:
        return QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "recovery",
            shell_section=section,
            material_direction=base_frame[:, 0],
            reference_normal=base_frame[:, 2],
        )

    baseline_element = make(nodes)
    baseline_stiffness = baseline_element.compute_stiffness_matrix(
        _mesh(nodes), material
    )
    baseline = baseline_element.compute_stresses(
        _mesh(nodes), displacement, material, return_global=True
    )
    assert baseline["physical_director_authoritative"] is True
    np.testing.assert_allclose(
        baseline["physical_director"], base_frame[:, 2], rtol=0.0, atol=2.0e-15
    )

    for permutation in D4:
        slots = np.asarray(permutation, dtype=int)
        numbered_nodes = nodes[slots]
        numbered_displacement = displacement.reshape(4, 6)[slots].reshape(24)
        element = make(numbered_nodes)
        stiffness = element.compute_stiffness_matrix(_mesh(numbered_nodes), material)
        dofs = np.asarray(
            [6 * int(slot) + dof for slot in slots for dof in range(6)],
            dtype=int,
        )
        np.testing.assert_allclose(
            stiffness,
            baseline_stiffness[np.ix_(dofs, dofs)],
            rtol=0.0,
            atol=2.0e-9,
        )
        actual = element.compute_stresses(
            _mesh(numbered_nodes),
            numbered_displacement,
            material,
            return_global=True,
        )
        np.testing.assert_allclose(
            actual["physical_director"], base_frame[:, 2], rtol=0.0, atol=2.0e-15
        )
        for key in (
            "global_membrane_resultant_tensors",
            "global_bending_resultant_tensors",
            "global_transverse_shear_resultants",
        ):
            np.testing.assert_allclose(
                actual[key],
                np.broadcast_to(baseline[key][0], actual[key].shape),
                rtol=0.0,
                atol=5.0e-9,
            )


def test_reflected_abd_congruence_uses_exact_engineering_field_maps() -> None:
    section = GeneralizedShellSection(
        A=np.asarray(((9.0, 1.0, 0.4), (1.0, 8.0, -0.3), (0.4, -0.3, 3.0))),
        B=np.asarray(((0.7, 0.2, -0.1), (0.05, -0.4, 0.15), (0.3, -0.2, 0.25))),
        D=np.asarray(((4.0, 0.3, 0.2), (0.3, 3.5, -0.1), (0.2, -0.1, 1.5))),
        As=np.asarray(((2.0, 0.25), (0.25, 1.6))),
    )
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "recovery",
        shell_section=section,
        material_direction=np.asarray((1.0, 0.0, 0.0)),
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )
    reflected_frame = np.diag((1.0, -1.0, -1.0))
    numbered = element._generalized_section_in_frame(reflected_frame)
    assert numbered is not None

    membrane = np.diag((1.0, 1.0, -1.0))
    curvature = np.diag((-1.0, -1.0, 1.0))
    shear = np.diag((-1.0, 1.0))
    generalized = np.block(
        [
            [membrane, np.zeros((3, 3))],
            [np.zeros((3, 3)), curvature],
        ]
    )
    expected_abd = generalized.T @ section.ABD @ generalized
    expected_shear = shear.T @ section.As @ shear
    np.testing.assert_array_equal(numbered.ABD, expected_abd)
    np.testing.assert_array_equal(numbered.As, expected_shear)
    # In particular B receives both the director sign and the engineering
    # reflection: B_n = E^T B K.  Omitting either factor is the historical gap.
    np.testing.assert_array_equal(numbered.B, membrane.T @ section.B @ curvature)


def test_authoritative_director_keeps_isotropic_top_and_bottom_physical() -> None:
    nodes = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=float,
    )
    director = equation7_frame(nodes)[0][:, 2]
    displacement = _combined_patch(nodes)
    material = _material()
    baseline = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "recovery",
        thickness=0.4,
        reference_normal=director,
    ).compute_stresses(_mesh(nodes), displacement, material, return_global=True)
    reflection = np.asarray(D4[4], dtype=int)
    reflected = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "recovery",
        thickness=0.4,
        reference_normal=director,
    ).compute_stresses(
        _mesh(nodes[reflection]),
        displacement.reshape(4, 6)[reflection].reshape(24),
        material,
        return_global=True,
    )
    for surface in ("top", "bot"):
        for component in ("xx", "yy", "zz", "xy", "yz", "xz"):
            key = f"global_{component}_{surface}"
            np.testing.assert_allclose(
                reflected[key],
                np.broadcast_to(baseline[key][0], reflected[key].shape),
                rtol=0.0,
                atol=5.0e-10,
            )


def test_recovery_is_d4_and_proper_global_covariant_with_physical_material_direction() -> None:
    nodes = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=float,
    )
    material = _orthotropic()
    material_direction = np.asarray((1.0, 0.37, 0.0), dtype=float)
    base_displacement = _combined_patch(nodes)
    base_element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "oriented",
        thickness=0.4,
        material_direction=material_direction,
    )
    baseline = base_element.compute_stresses(
        _mesh(nodes), base_displacement, material, return_global=True
    )
    base_normal = equation7_frame(nodes)[0][:, 2]
    for permutation in D4:
        numbered_nodes = nodes[list(permutation)]
        numbered_displacement = base_displacement.reshape(4, 6)[list(permutation)].reshape(24)
        element = QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "oriented",
            thickness=0.4,
            material_direction=material_direction,
        )
        actual = element.compute_stresses(
            _mesh(numbered_nodes),
            numbered_displacement,
            material,
            return_global=True,
        )
        normal_sign = float(np.dot(equation7_frame(numbered_nodes)[0][:, 2], base_normal))
        for key in ("global_membrane_resultant_tensors",):
            np.testing.assert_allclose(
                actual[key],
                np.broadcast_to(baseline[key][0], actual[key].shape),
                rtol=0.0,
                atol=3.0e-10,
            )
        for key in (
            "global_bending_resultant_tensors",
            "global_transverse_shear_resultants",
        ):
            np.testing.assert_allclose(
                actual[key],
                np.broadcast_to(normal_sign * baseline[key][0], actual[key].shape),
                rtol=0.0,
                atol=3.0e-10,
            )

    angle = 0.63
    axis = np.asarray((0.3, -0.4, 0.7), dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    rotated_nodes = nodes @ rotation.T + np.asarray((2.0, -1.0, 0.8))
    rotated_element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "oriented",
        thickness=0.4,
        material_direction=rotation @ material_direction,
    )
    rotated = rotated_element.compute_stresses(
        _mesh(rotated_nodes),
        _rotate_dofs(base_displacement, rotation),
        material,
        return_global=True,
    )
    for key in (
        "global_membrane_resultant_tensors",
        "global_bending_resultant_tensors",
    ):
        expected = np.asarray(
            [rotation @ tensor @ rotation.T for tensor in baseline[key]],
            dtype=float,
        )
        np.testing.assert_allclose(rotated[key], expected, rtol=0.0, atol=3.0e-10)
    np.testing.assert_allclose(
        rotated["global_transverse_shear_resultants"],
        baseline["global_transverse_shear_resultants"] @ rotation.T,
        rtol=0.0,
        atol=3.0e-10,
    )


def test_numerical_fields_and_nonfinite_values_cannot_enter_planar_recovery() -> None:
    nodes = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=float,
    )
    mesh = _mesh(nodes)
    material = _material()
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], "recovery", thickness=0.4)
    drill_only = np.zeros(24, dtype=float)
    drill_only[5::6] = np.asarray((1.0, -2.0, 3.0, -4.0))
    recovered = element.compute_stresses(mesh, drill_only, material)
    for key in (
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
    ):
        np.testing.assert_array_equal(recovered[key], np.zeros_like(recovered[key]))
    element.compute_stiffness_matrix(mesh, material)
    assert np.linalg.norm(element.numerical_internal_force(drill_only)["numerical"]) > 0.0

    displacement = np.linspace(-0.02, 0.03, 24)
    changed_numerics = QualifiedE4PLShellElement(
        2,
        [1, 2, 3, 4],
        "recovery",
        thickness=0.4,
        pl_stabilization=7.0,
        hourglass_stabilization=0.25,
    ).compute_stresses(mesh, displacement, material)
    baseline = element.compute_stresses(mesh, displacement, material)
    for key in (
        "membrane_resultants",
        "bending_resultants",
        "transverse_shear_resultants",
    ):
        np.testing.assert_array_equal(changed_numerics[key], baseline[key])

    nonfinite = displacement.copy()
    nonfinite[3] = np.nan
    with pytest.raises(ValueError, match="finite displacements"):
        element.compute_stresses(mesh, nonfinite, material)
    with pytest.raises(ValueError, match="finite Nx2"):
        element._recover_planar_mixed_fields(
            mesh,
            displacement,
            material,
            ((0.0, math.nan),),
        )
