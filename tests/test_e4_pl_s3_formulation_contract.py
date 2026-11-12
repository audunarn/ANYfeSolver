from __future__ import annotations

import json
from decimal import Decimal
import math
from pathlib import Path

import pytest

from anyfesolver.algebraic_dynamics import (
    DESCRIPTOR_COORDINATE_SHEAR_LIMIT,
    DESCRIPTOR_DENSE_CONDENSATION_LIMIT,
    DESCRIPTOR_MODAL_POLICY_ID,
    DESCRIPTOR_SHIFT_RATIO,
    DESCRIPTOR_TRANSIENT_FIRST_ORDER_POLICY_ID,
    DESCRIPTOR_TRANSIENT_CONSTRAINED_POLICY_ID,
    DESCRIPTOR_TRANSIENT_POLICY_ID,
    DESCRIPTOR_TRANSIENT_STATIC_POLICY_ID,
)
from anyfesolver.buckling import (
    CURRENT_STATE_BUCKLING_POLICY_ID,
    QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID,
)
from anyfesolver.current_state_tangent import (
    COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID,
    COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID,
    COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID,
    CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID,
)
from anyfesolver.dynamics import QUALIFIED_REFERENCE_TRANSIENT_AUTHORITY_POLICY_ID
from anyfesolver.e4_pl_element import (
    Q4_ACTIVITY_DISPOSITION_SCHEMA_ID,
    Q4_DELETED_FROZEN_POLICY_ID,
    Q4_FAILED_STATE_POLICY_ID,
    Q4_QUADRATURE_AUTHORITY_ID,
)
from anyfesolver.e4_pl_s3_element import (
    ALGEBRAIC_COORDINATE_POLICY_ID,
    BUBBLE_OFFSET_D,
    CURRENT_STATE_BUBBLE_PROJECTION_POLICY_ID,
    CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
    EXPLICIT_BUCKLING_PROFILE_DISPOSITION,
    FORMULATION_ID,
    GEOMETRIC_STIFFNESS_POLICY_ID,
    HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID,
    MASS_MOMENT_ID,
    MITC3_PLUS_NONLINEAR_EQUATION_MAP,
    MITC3_PLUS_NONLINEAR_SOURCE_BYTES,
    MITC3_PLUS_NONLINEAR_SOURCE_SHA256,
    MITC3_PLUS_SOURCE_BYTES,
    MITC3_PLUS_SOURCE_SHA256,
    QUADRATURE_ID,
    RECOVERY_POLICY_ID,
    REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    RESTART_HISTORY_SCOPE,
    RESULTANT_SUMMARY_POLICY_ID,
    S3_ACTIVITY_DISPOSITION_SCHEMA_ID,
    S3_DELETED_FROZEN_POLICY_ID,
    S3_FAILED_STATE_POLICY_ID,
    S3_QUADRATURE_AUTHORITY_ID,
    TRIANGLE_QUADRATURE,
    TYING_POINTS,
    QualifiedE4PLS3ShellElement,
)
from anyfesolver.e4_pl_s3_state import (
    BUBBLE_CONVENTION as STATE_BUBBLE_CONVENTION,
    BUBBLE_CONDITION_LIMIT,
    BUBBLE_FORCE_CONDENSATION_ID,
    BUBBLE_LINE_SEARCH_MIN_FACTOR,
    BUBBLE_LINE_SEARCH_REDUCTION,
    BUBBLE_MAX_ITERATIONS,
    BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
    BUBBLE_RELATIVE_TOLERANCE,
    BUBBLE_STEP_TOLERANCE,
    CANONICALIZATION_ID,
    DIRECTOR_POLARITY_POLICY_ID,
    DIRECTOR_REVERSAL_TRANSFORM_ID,
    DIRECTOR_INCREMENT_MAP_ID,
    DIRECTOR_GAUGE_FALLBACK_AXIS,
    DIRECTOR_GAUGE_ID,
    DIRECTOR_GAUGE_PRIMARY_AXIS,
    DIRECTOR_GAUGE_SWITCH_TOLERANCE,
    EXTERNAL_COORDINATE_LAYOUT_ID,
    EXTERNAL_ROTATION_MAP_ID,
    ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA,
    GENERALIZED_ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA,
    GENERALIZED_INITIAL_FIELD_POLICY_ID,
    GENERALIZED_NONLINEAR_STATE_LAYOUT_ID,
    GENERALIZED_NONLINEAR_STATE_SCHEMA,
    GENERALIZED_NONLINEAR_STATE_VERSION,
    GENERALIZED_RESULTANT_COMPONENT_ORDER,
    GENERALIZED_SECTION_DESCRIPTOR_SCHEMA,
    GENERALIZED_SECTION_HISTORY_POLICY_ID,
    GENERALIZED_SECTION_INTEGRATION_ID,
    GENERALIZED_STATE_MODE,
    GENERALIZED_STRAIN_COMPONENT_ORDER,
    ISOTROPIC_CONSTITUTIVE_INTEGRATION_ID,
    LAYER_STRAIN_COMPONENT_ORDER,
    LAYER_STRESS_COMPONENT_ORDER,
    MATERIAL_DESCRIPTOR_VALIDATION_ID,
    NODAL_ROTATION_UPDATE_POLICY_ID,
    NONLINEAR_KINEMATICS_ID,
    NONLINEAR_PL_ENERGY_POLICY_ID,
    NONLINEAR_POLICY_ID,
    NONLINEAR_STATE_LAYOUT_ID,
    NONLINEAR_STATE_SCHEMA,
    NONLINEAR_STATE_VERSION,
    ORTHOTROPIC_CONSTITUTIVE_INTEGRATION_ID,
    PL_MINIMUM_TWIST_DENOMINATOR,
    PL_MINIMUM_TWIST_DENOMINATOR_ID,
    PL_PHASE_MARGIN,
    PL_PHASE_MARGIN_ID,
    PL_PHASE_POLICY_ID,
    PL_TWIST_POLICY_ID,
    REFERENCE_SURFACE_MASS_SHIFT_ID,
    REFERENCE_SURFACE_OFFSET_POLICY_ID,
    REFERENCE_SURFACE_STRAIN_TRANSFORM_ID,
    REFERENCE_GEOMETRY_VALIDATION_ID,
    STATE_ARRAY_LAYOUT_ID,
    STATE_FIELD_MANIFEST,
    STATE_INTEGRITY_ID,
    STATE_MODE,
    STATE_REDUNDANCY_VALIDATION_ID,
    SUPPORTED_LOBATTO_LAYER_COUNTS,
    SURFACE_ROTATION_POLICY_ID,
    THICKNESS_COORDINATE_SIGN_ID,
    THICKNESS_QUADRATURE_ID,
    formulation_fingerprint,
    formulation_mechanics_contract_payload,
    formulation_mechanics_fingerprint,
    generalized_formulation_fingerprint,
    generalized_state_field_manifest_fingerprint,
    lobatto_table_fingerprint,
    state_field_manifest_fingerprint,
    stiffness_station_table_fingerprint,
)
from anyfesolver.modal import (
    CURRENT_STATE_MODAL_POLICY_ID,
    QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID,
)
from anyfesolver.nonlinear_restart import (
    NONLINEAR_CHECKPOINT_INTEGRITY_ID,
    NONLINEAR_CHECKPOINT_SCHEMA,
    NONLINEAR_CHECKPOINT_VERSION,
    QUALIFIED_CHECKPOINT_LIFECYCLE_POLICY_ID,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/reference_cases/e4_pl_s3_formulation_contract.json"


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value}")


def _load_contract_bytes(raw: bytes) -> dict:
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("contract must be UTF-8 with exactly LF line endings")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=reject_duplicate,
        parse_constant=_reject_constant,
    )

    def require_finite(member: object) -> None:
        if isinstance(member, float) and not math.isfinite(member):
            raise ValueError("contract contains a non-finite number")
        if isinstance(member, dict):
            for nested in member.values():
                require_finite(nested)
        elif isinstance(member, list):
            for nested in member:
                require_finite(nested)

    require_finite(value)
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ValueError("contract JSON is not in sorted canonical form")
    assert isinstance(value, dict)
    return value


def _contract() -> dict:
    return _load_contract_bytes(CONTRACT_PATH.read_bytes())


def test_runtime_capability_contract_has_exact_bounded_successor_scope() -> None:
    """Bind the runtime successor without rewriting historical evidence."""

    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        reference_normal=(0.0, 0.0, 1.0),
    )
    matrix = element.capability_matrix()
    assert matrix["restart_history"] == RESTART_HISTORY_SCOPE == (
        "STATIC_AND_ARC_LENGTH_CHECKPOINTS_ONLY"
    )
    assert element.capability_restrictions["restart_history"] == (
        RESTART_HISTORY_SCOPE
    )
    assert matrix["reference_elastic_buckling"] == "PARITY_REPLACED"
    assert matrix["current_state_buckling_s3"] == "PARITY_REPLACED"
    assert matrix["mixed_current_state_buckling"] == "PARITY_REPLACED"
    assert matrix["mixed_current_state_modal"] == "PARITY_REPLACED"
    assert matrix["buckling"] == EXPLICIT_BUCKLING_PROFILE_DISPOSITION
    assert matrix["linearized_limit_point"] == (
        "UNSUPPORTED_OUTSIDE_ADMITTED_PROFILE"
    )


def test_contract_binds_the_exact_public_source_and_candidate() -> None:
    contract = _contract()
    primary = contract["source_authority"]["primary"]

    assert contract["schema"] == "anysolver.e4-pl-s3-formulation-contract-v1"
    assert contract["candidate_id"] == FORMULATION_ID
    assert primary["bytes"] == MITC3_PLUS_SOURCE_BYTES == 1_146_142
    assert primary["sha256"] == MITC3_PLUS_SOURCE_SHA256
    assert contract["formulation"]["bubble_offset_d"] == "1/10000"
    assert BUBBLE_OFFSET_D == pytest.approx(1.0e-4, rel=0.0, abs=0.0)


def test_contract_binds_the_native_nonlinear_equation_authority() -> None:
    nonlinear = _contract()["source_authority"]["nonlinear"]

    assert nonlinear["bytes"] == MITC3_PLUS_NONLINEAR_SOURCE_BYTES == 2_312_312
    assert nonlinear["sha256"] == MITC3_PLUS_NONLINEAR_SOURCE_SHA256
    assert nonlinear["equation_map"] == {
        "assumed_nonlinear_covariant_shear": "PDF_PAGE_9_EQUATIONS_29_TO_31",
        "current_geometry_and_increment": "PDF_PAGES_4_TO_5_EQUATIONS_7_TO_9",
        "director_update": "PDF_PAGES_5_TO_6_EQUATIONS_10_TO_15",
        "incremental_green_lagrange": "PDF_PAGES_8_TO_9_EQUATIONS_22_TO_28",
        "quadratic_director_increment": "PDF_PAGE_8_EQUATIONS_16_TO_21",
    }
    assert MITC3_PLUS_NONLINEAR_EQUATION_MAP == {
        "current_geometry": "equations_7_to_9",
        "director_update": "equations_10_to_15",
        "quadratic_director_increment": "equations_16_to_21",
        "incremental_green_lagrange": "equations_22_to_28",
        "assumed_nonlinear_covariant_shear": "equations_29_to_31",
    }


def test_contract_binds_native_nonlinear_gauge_rotation_and_state() -> None:
    native = _contract()["native_nonlinear_kinematics"]

    assert native == {
        "bubble_equilibrium_policy": {
            "condition_limit": BUBBLE_CONDITION_LIMIT,
            "force_condensation_id": BUBBLE_FORCE_CONDENSATION_ID,
            "line_search_min_factor": BUBBLE_LINE_SEARCH_MIN_FACTOR,
            "line_search_reduction": BUBBLE_LINE_SEARCH_REDUCTION,
            "max_iterations": BUBBLE_MAX_ITERATIONS,
            "relative_tolerance": BUBBLE_RELATIVE_TOLERANCE,
            "step_tolerance": BUBBLE_STEP_TOLERANCE,
        },
        "committed_current_tangent": {
            "bubble_projection": (
                "G_VERTICAL_STACK_I_AND_T_PROJECTS_BOTH_COMPONENTS_WITH_THE_SAME_T"
            ),
            "bubble_projection_policy_id": (
                CURRENT_STATE_BUBBLE_PROJECTION_POLICY_ID
            ),
            "bubble_sensitivity": (
                "T_EQUALS_MINUS_KAA_TOTAL_INVERSE_KAQ_TOTAL_AFTER_CONVERGED_"
                "BUBBLE_SOLVE"
            ),
            "geometric_component": (
                "RESULTANT_OR_STRESS_HESSIAN_TENSION_POSITIVE"
            ),
            "material_component": (
                "CONSTITUTIVE_ALGORITHMIC_PLUS_OBJECTIVE_PL_MATERIAL_NUMERICAL"
            ),
            "persistence": (
                "READ_ONLY_TRANSIENT_API_NO_MATRIX_SENSITIVITY_OR_"
                "FACTORIZATION_STATE"
            ),
            "policy_id": CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
            "state_immutability": (
                "CANONICAL_INPUT_BYTES_AND_INTEGRITY_DIGEST_VERIFIED_UNCHANGED"
            ),
            "total_closure": (
                "KMATERIAL_PLUS_KGEOMETRIC_EQUALS_EXISTING_CONSISTENT_TOTAL_"
                "TANGENT"
            ),
            "uncondensed_split": (
                "LAYERED_AND_STATELESS_GENERALIZED_CONSTITUTIVE_VERSUS_"
                "RESULTANT_HESSIAN"
            ),
        },
        "constitutive_integration_ids": {
            "generalized_section": GENERALIZED_SECTION_INTEGRATION_ID,
            "isotropic": ISOTROPIC_CONSTITUTIVE_INTEGRATION_ID,
            "orthotropic": ORTHOTROPIC_CONSTITUTIVE_INTEGRATION_ID,
        },
        "corner_rotation_authority": {
            "commit_binding": (
                "ACCEPTED_Q_AND_EXACT_ACCEPTED_FULL_GLOBAL_DISPLACEMENT"
            ),
            "element_mesh_node_mutation": "FORBIDDEN",
            "id": NODAL_ROTATION_UPDATE_POLICY_ID,
            "ownership": "SOLVER_OWNED_NODE_SHARED_PER_ANALYSIS",
            "trial_base": "COMMITTED_Q_ONLY_NEVER_PREVIOUS_TRIAL",
            "trial_update": (
                "Q_TRIAL=EXP(THETA_TRIAL_MINUS_THETA_COMMITTED)@Q_COMMITTED"
            ),
        },
        "director_gauge": {
            "fallback_axis_global": ["0", "0", "1"],
            "fallback_rule": (
                "USE_FALLBACK_WHEN_NORM_PRIMARY_CROSS_VN_LE_1E_MINUS_12"
            ),
            "id": DIRECTOR_GAUGE_ID,
            "primary_axis_global": ["0", "1", "0"],
            "reconstruction": (
                "V1=NORMALIZE(AXIS_CROSS_VN)_AND_V2=VN_CROSS_V1_AFTER_EVERY_COMMIT"
            ),
            "validation": (
                "STORED_TANGENTS_MUST_EQUAL_RECONSTRUCTION_WITH_ABSOLUTE_"
                "TOLERANCE_1E_MINUS_12"
            ),
        },
        "director_increment_map": {
            "bubble_coordinates": (
                "TWO_COMPONENT_HIERARCHICAL_ROTATION_RELATIVE_TO_CORNER_AVERAGE"
            ),
            "bubble_director": (
                "PUBLISHED_EQ14_WITH_HIERARCHICAL_SOURCE_NODE_ONLY"
            ),
            "corner_directors": "VN_I=Q_I@VN_I_REFERENCE",
            "id": DIRECTOR_INCREMENT_MAP_ID,
            "scope": "SOLVER_SHARED_SO3_FOR_CORNERS_EQ14_ONLY_FOR_BUBBLE",
        },
        "external_rotation_map": {
            "component_frame": "GLOBAL_XYZ",
            "id": EXTERNAL_ROTATION_MAP_ID,
            "shared_node_consistency": "EXACT_CROSS_ELEMENT_Q_AGREEMENT_REQUIRED",
            "stored_coordinates": "GLOBAL_ROTATION_VECTOR_COMPONENTS",
        },
        "generalized_state": {
            "field_manifest_sha256": (
                generalized_state_field_manifest_fingerprint()
            ),
            "forbidden_fabricated_fields": [
                "alpha",
                "kinematic_layer_strain",
                "layer_strain",
                "layer_strain_material",
                "layer_stress",
                "layer_stress_material",
                "plastic_strain",
            ],
            "formulation_fingerprint_sha256": (
                generalized_formulation_fingerprint()
            ),
            "history_policy_id": GENERALIZED_SECTION_HISTORY_POLICY_ID,
            "initial_field_contract": {
                "bending_resultant": (
                    "M0_EQUALS_H_SQUARED_OVER_6_TIMES_SIGMA_BENDING_SURFACE"
                ),
                "constitutive_relation": (
                    "RESULTANT_EQUALS_C_TIMES_KINEMATIC_MINUS_PRESTRAIN_"
                    "PLUS_INITIAL_RESULTANT"
                ),
                "fingerprint_binding": (
                    "RAW_STATION_FIELDS_PLUS_CANONICAL_PROVENANCE"
                ),
                "membrane_resultant": "N0_EQUALS_H_TIMES_SIGMA_MEMBRANE",
                "physical_layer_reconstruction": "FORBIDDEN",
                "policy_id": GENERALIZED_INITIAL_FIELD_POLICY_ID,
                "source_convention": (
                    "SHELL_INITIAL_FIELD_LOCAL_SURFACE_STRESS_V1"
                ),
                "station_count": 7,
            },
            "integration_id": GENERALIZED_SECTION_INTEGRATION_ID,
            "layout_id": GENERALIZED_NONLINEAR_STATE_LAYOUT_ID,
            "physical_layer_recovery_available": False,
            "recovery_scope": "section_resultants_only",
            "rejected_history_markers": [
                "committed_state",
                "history_schema",
                "init_nonlinear_state",
                "initialize_state",
                "requires_history",
                "state_layout_id",
                "state_schema",
                "update_nonlinear_state",
                "update_state",
            ],
            "restart_binding": (
                "STATE_CODEC_BINDS_EXACT_SECTION_GEOMETRY_OFFSET_Q_U_BUBBLE_"
                "PL_STRAIN_RESULTANT_INITIAL_FIELDS_PROVENANCE_AND_COMPLETE_"
                "STATE_DIGEST"
            ),
            "schema": GENERALIZED_NONLINEAR_STATE_SCHEMA,
            "section_descriptor_schema": GENERALIZED_SECTION_DESCRIPTOR_SCHEMA,
            "solver_restart_policy": {
                "analysis_kinds": ["static", "arc_length"],
                "checkpoint_integrity_id": NONLINEAR_CHECKPOINT_INTEGRITY_ID,
                "checkpoint_schema": NONLINEAR_CHECKPOINT_SCHEMA,
                "checkpoint_version": NONLINEAR_CHECKPOINT_VERSION,
                "generic_restart_history": (
                    "PARITY_GAP_OUTSIDE_STATIC_AND_ARC_LENGTH_CONTINUATION"
                ),
                "ownership": (
                    "SOLVER_OWNED_MODEL_LOAD_GLOBAL_DISPLACEMENT_ELEMENT_STATE_"
                    "ACTIVITY_DELETION_AND_PATH_BINDING"
                ),
                "status": "STATIC_AND_ARC_LENGTH_PARITY_REPLACED",
            },
            "state_mode": GENERALIZED_STATE_MODE,
            "state_version": GENERALIZED_NONLINEAR_STATE_VERSION,
        },
        "objective_pl": {
            "energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
            "force_tangent": (
                "EXACT_GRADIENT_AND_HESSIAN_OF_BARYCENTRIC_PL_ENERGY"
            ),
            "minimum_twist_denominator_binary64": (
                PL_MINIMUM_TWIST_DENOMINATOR
            ),
            "minimum_twist_denominator_id": PL_MINIMUM_TWIST_DENOMINATOR_ID,
            "phase_interval": "(-PI,PI]",
            "phase_margin_binary64": PL_PHASE_MARGIN,
            "phase_margin_id": PL_PHASE_MARGIN_ID,
            "phase_policy_id": PL_PHASE_POLICY_ID,
            "rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
            "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
            "twist_policy_id": PL_TWIST_POLICY_ID,
        },
        "policy_id": NONLINEAR_POLICY_ID,
        "reference_component_basis": (
            "EXPLICIT_FROZEN_REFERENCE_NODES_AND_REFERENCE_FRAME_REQUIRED_NO_"
            "CURRENT_BASIS_FALLBACK"
        ),
        "state": {
            "bubble_convention": STATE_BUBBLE_CONVENTION,
            "bubble_predictor_commit_policy_id": (
                BUBBLE_PREDICTOR_COMMIT_POLICY_ID
            ),
            "canonicalization_id": CANONICALIZATION_ID,
            "director_increment_map_id": DIRECTOR_INCREMENT_MAP_ID,
            "external_coordinate_layout_id": EXTERNAL_COORDINATE_LAYOUT_ID,
            "external_rotation_map_id": EXTERNAL_ROTATION_MAP_ID,
            "field_manifest_sha256": state_field_manifest_fingerprint(),
            "formulation_fingerprint_sha256": formulation_fingerprint(),
            "formulation_mechanics_sha256": formulation_mechanics_fingerprint(),
            "generalized_resultant_component_order": list(
                GENERALIZED_RESULTANT_COMPONENT_ORDER
            ),
            "generalized_strain_component_order": list(
                GENERALIZED_STRAIN_COMPONENT_ORDER
            ),
            "layer_strain_component_order": list(LAYER_STRAIN_COMPONENT_ORDER),
            "layer_stress_component_order": list(LAYER_STRESS_COMPONENT_ORDER),
            "layout_id": NONLINEAR_STATE_LAYOUT_ID,
            "lobatto_table_sha256": lobatto_table_fingerprint(),
            "material_descriptor_validation_id": (
                MATERIAL_DESCRIPTOR_VALIDATION_ID
            ),
            "nodal_rotation_update_policy_id": (
                NODAL_ROTATION_UPDATE_POLICY_ID
            ),
            "nonlinear_kinematics_id": NONLINEAR_KINEMATICS_ID,
            "nonlinear_pl_energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
            "pl_minimum_twist_denominator_binary64": (
                PL_MINIMUM_TWIST_DENOMINATOR
            ),
            "pl_minimum_twist_denominator_id": (
                PL_MINIMUM_TWIST_DENOMINATOR_ID
            ),
            "pl_phase_margin_binary64": PL_PHASE_MARGIN,
            "pl_phase_margin_id": PL_PHASE_MARGIN_ID,
            "pl_phase_policy_id": PL_PHASE_POLICY_ID,
            "pl_twist_policy_id": PL_TWIST_POLICY_ID,
            "quadrature_id": QUADRATURE_ID,
            "reference_corner_directors_fingerprint_layout_id": (
                "ORDERED_S3_ELEMENT_OWNED_REFERENCE_DIRECTORS_GLOBAL_V1"
            ),
            "reference_geometry_validation_id": (
                REFERENCE_GEOMETRY_VALIDATION_ID
            ),
            "recovery_policy_id": RECOVERY_POLICY_ID,
            "schema": NONLINEAR_STATE_SCHEMA,
            "state_array_layout_id": STATE_ARRAY_LAYOUT_ID,
            "state_integrity_id": STATE_INTEGRITY_ID,
            "state_mode": STATE_MODE,
            "state_redundancy_validation_id": (
                STATE_REDUNDANCY_VALIDATION_ID
            ),
            "state_version": NONLINEAR_STATE_VERSION,
            "stiffness_station_table_sha256": (
                stiffness_station_table_fingerprint()
            ),
            "supported_lobatto_layer_counts": list(
                SUPPORTED_LOBATTO_LAYER_COUNTS
            ),
            "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
            "thickness_coordinate_sign_id": THICKNESS_COORDINATE_SIGN_ID,
            "thickness_quadrature_id": THICKNESS_QUADRATURE_ID,
        },
    }
    assert tuple(
        float(value)
        for value in native["director_gauge"]["primary_axis_global"]
    ) == DIRECTOR_GAUGE_PRIMARY_AXIS
    assert tuple(
        float(value)
        for value in native["director_gauge"]["fallback_axis_global"]
    ) == DIRECTOR_GAUGE_FALLBACK_AXIS
    assert DIRECTOR_GAUGE_SWITCH_TOLERANCE == 1.0e-12
    assert STATE_FIELD_MANIFEST["station_generalized_resultant"][
        "component_frame"
    ] == "numbered_reference_tensor_resultant"
    assert STATE_FIELD_MANIFEST["layer_strain_material"][
        "point_order"
    ] == "station_major_layer_minor_physical_director_bottom_to_top"


@pytest.mark.parametrize(
    "raw",
    (
        b'{\n  "a": 1,\n  "a": 2\n}\n',
        b'{\r\n  "a": 1\r\n}\r\n',
        b'{\n  "a": 1e999\n}\n',
        b'{\n  "a": 1\n}',
        b'\xef\xbb\xbf{\n  "a": 1\n}\n',
        b'{"a":1}\n',
    ),
)
def test_contract_loader_rejects_duplicate_nonfinite_and_noncanonical_json(
    raw: bytes,
) -> None:
    with pytest.raises((AssertionError, ValueError, UnicodeError)):
        _load_contract_bytes(raw)


def test_tying_points_and_quadrature_are_transcribed_without_aliases() -> None:
    contract = _contract()
    mechanics = formulation_mechanics_contract_payload()
    assert set(TYING_POINTS) == {"A", "B", "C", "D", "E", "F"}
    expected_points = {
        "A": (1.0 / 6.0, 2.0 / 3.0),
        "B": (2.0 / 3.0, 1.0 / 6.0),
        "C": (1.0 / 6.0, 1.0 / 6.0),
        "D": (1.0 / 3.0 + 1.0e-4, 1.0 / 3.0 - 2.0e-4),
        "E": (1.0 / 3.0 - 2.0e-4, 1.0 / 3.0 + 1.0e-4),
        "F": (1.0 / 3.0 + 1.0e-4, 1.0 / 3.0 + 1.0e-4),
    }
    for label, expected in expected_points.items():
        assert TYING_POINTS[label] == pytest.approx(expected)
        assert tuple(mechanics["bubble"]["tying_point_binary64"][label]) == (
            TYING_POINTS[label]
        )
        assert list(mechanics["bubble"]["tying_point_definitions"][label]) == (
            contract["formulation"]["tying_points"][label]
        )
    assert mechanics["bubble"]["offset_d_exact"] == contract["formulation"][
        "bubble_offset_d"
    ]
    assert mechanics["pl_completion"]["basis_id"] == "BARYCENTRIC_L1_L2_L3_V1"
    assert mechanics["drilling_scale"]["projector"] == (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
    )
    assert contract["quadrature"]["authority_id"] == (
        S3_QUADRATURE_AUTHORITY_ID
    )
    assert contract["quadrature"]["stiffness"]["id"].startswith("DUNAVANT_DEGREE_5")
    assert QUADRATURE_ID == "dunavant_degree5_7point"
    assert len(TRIANGLE_QUADRATURE) == 7
    assert sum(weight for _r, _s, weight in TRIANGLE_QUADRATURE) == pytest.approx(0.5)


def test_pl_and_rank_contract_are_exactly_frozen() -> None:
    contract = _contract()
    pl = contract["pl_completion"]
    ranks = contract["rank_contract"]

    assert pl["one_point_centroid_integration"] == "FORBIDDEN"
    assert pl["saddle_tau_tau"] == "-M/kD"
    assert ranks["uncondensed_physical_17"] == {
        "inertia_negative": 0,
        "inertia_positive": 11,
        "inertia_zero": 6,
        "nullity": 6,
        "rank": 11,
    }
    assert ranks["condensed_external_18"] == {"nullity": 6, "rank": 12}
    assert ranks["full_saddle_23"] == {
        "inertia_negative": 3,
        "inertia_positive": 14,
        "inertia_zero": 6,
        "nullity": 6,
        "rank": 17,
    }
    assert contract["serialization_fingerprint"] == {
        "algebraic_coordinate_policy": ALGEBRAIC_COORDINATE_POLICY_ID,
        "bubble_convention": "hierarchical_rotation_relative_to_corner_average",
        "director_polarity": "REQUIRED_INTEGER_MINUS_OR_PLUS_ONE",
        "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
        "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
        "dynamic_reduction_policy": "GUYAN_STATIC_BUBBLE_FULL_CONSISTENT_MASS_V1",
        "formulation_id": FORMULATION_ID,
        "formulation_schema": "anysolver.e4_pl_s3.linear.v1",
        "geometric_stiffness_policy": GEOMETRIC_STIFFNESS_POLICY_ID,
        "mass_moment_id": MASS_MOMENT_ID,
        "quadrature_authority_id": S3_QUADRATURE_AUTHORITY_ID,
        "quadrature_id": "dunavant_degree5_7point",
        "reference_surface_mass_shift_id": REFERENCE_SURFACE_MASS_SHIFT_ID,
        "reference_surface_offset": "FINITE_SIGNED_LENGTH_DEFAULT_ZERO",
        "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
        "reference_surface_strain_transform_id": (
            REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
        ),
        "recovery_policy_id": RECOVERY_POLICY_ID,
        "state_layout_id": "S3_EXTERNAL18_BUBBLE2_PL3_LINEAR_V1",
    }
    assert contract["recovery_policy"] == {
        "bubble_state": "ELASTIC_SCHUR_BACK_SUBSTITUTION_USED_BY_THE_TANGENT",
        "committed_state_recovery": (
            "STRICT_MODEL_BOUND_LAYERED_V2_AND_STATELESS_GENERALIZED_V4_WITH_"
            "COMPLETE_STATE_DIGEST"
        ),
        "director_reversal": (
            "IMPLEMENTED_FOR_NATIVE_LINEAR_LAYERED_AND_STATELESS_GENERALIZED_"
            "WITH_TOP_BOTTOM_SWAP"
        ),
        "engineering_conventions": (
            "MEMBRANE_AND_CURVATURE_ENGINEERING_SHEAR_WITH_TENSOR_N_AND_M"
        ),
        "failure_policy": (
            "QUALIFIED_RECOVERY_ERRORS_PROPAGATE_WITHOUT_SILENT_ELEMENT_OMISSION"
        ),
        "generalized_sections": (
            "SEVEN_STATION_RESULTANTS_ONLY_WITHOUT_FABRICATED_PHYSICAL_STRESS"
        ),
        "geometric_prestress_provenance": (
            "HOMOGENEOUS_PHYSICAL_RECOVERY_EMITS_REFERENCE_BUBBLE_AND_"
            "THROUGH_THICKNESS_PROFILE_IDS_GENERALIZED_RECOVERY_DOES_NOT"
        ),
        "global_transport": (
            "OWNER_FRAME_CONGRUENCE_WITH_POLARITY_CONVERSION_FOR_PHYSICAL_M_AND_Q"
        ),
        "hill48_measure": (
            "MAX_TOP_BOTTOM_MATERIAL_AXIS_PLANE_STRESS_TRANSVERSE_SHEAR_"
            "EXCLUDED_UTILIZATION_OVER_X"
        ),
        "homogeneous_surface_stress": "SIGMA_M=N_OVER_H_SIGMA_B=6M_OVER_H_SQUARED",
        "kinematics": "NATIVE_SEVEN_STATION_MITC3_PLUS_WITH_RECOVERED_BUBBLE",
        "material_orientation": (
            "PHYSICAL_SURFACE_DIRECTION_PLUS_SIGNED_IN_PLANE_ANGLE_INVARIANT_"
            "UNDER_DIRECTOR_REEXPRESSION"
        ),
        "nonlinear_history_patch": (
            "LAYERED_V2_COMMITTED_STATION_AND_LAYER_HISTORY_WITH_NATIVE_SEVEN_"
            "TO_THREE_PATCH_GENERALIZED_V4_RESULTANTS_ONLY_NO_PHYSICAL_PATCH"
        ),
        "numerical_fields": "PL_AND_DRILL_EXCLUDED_FROM_PHYSICAL_RECOVERY",
        "policy_id": RECOVERY_POLICY_ID,
        "resultant_sign": "TENSION_POSITIVE_N_M_Q",
        "summary_policy": RESULTANT_SUMMARY_POLICY_ID,
        "surface_sign": (
            "FROM_NODAL_REFERENCE_TOP_PLUS_H_OVER_2_MINUS_OFFSET_ALONG_"
            "PHYSICAL_DIRECTOR_BOTTOM_MINUS_H_OVER_2_MINUS_OFFSET"
        ),
        "transverse_shear": (
            "Q_OVER_H_EQUALS_FIVE_SIXTHS_G_GAMMA_REPEATED_AT_BOTH_SURFACES"
        ),
        "von_mises_measure": (
            "MAX_TOP_BOTTOM_3D_EQUIVALENT_WITH_AVERAGE_TRANSVERSE_SHEAR"
        ),
    }


def test_contract_binds_native_contact_state_parity() -> None:
    assert _contract()["contact_policy"] == {
        "admitted_scope": (
            "NATIVE_LAYERED_AND_STATELESS_GENERALIZED_SPHERE_SHELL_CONTACT"
        ),
        "configuration": (
            "EXACT_NATIVE_ROTATION_TRIAL_AT_CURRENT_CONTACT_CONFIGURATION"
        ),
        "cutback_state": "TRIAL_STATE_DISCARDED_AND_COMMITTED_STATE_BYTE_STABLE",
        "director_and_numbering": (
            "PHYSICAL_DIRECTOR_POLARITY_AND_ALL_SIX_D3_NUMBERINGS"
        ),
        "drill_policy": (
            "ZERO_INERTIA_DESCRIPTOR_STATIC_ALGEBRAIC_TRANSIENT_REDUCTION_"
            "WITHOUT_ARTIFICIAL_DRILL_MASS"
        ),
        "mixed_topology": "Q4_AND_S3_TARGETED_BY_THEIR_OWN_CURRENT_SURFACES",
        "offset_surfaces": (
            "MIDSURFACE_TOP_BOTTOM_WITH_EXACT_FORCE_MOMENT_VIRTUAL_WORK_CONJUGACY"
        ),
        "status": "PARITY_REPLACED",
    }


def test_contract_binds_physical_director_reversal_and_narrow_eigen_workflows() -> None:
    contract = _contract()
    reversal = contract["physical_director_reversal"]
    assert reversal == {
        "constitutive_transform": (
            "S=DIAG_I3_POLARITY_I5_AND_C_POLARITY=S_TRANSPOSE_C_OWNER_S"
        ),
        "director_polarity": "REQUIRED_INTEGER_MINUS_OR_PLUS_ONE",
        "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
        "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
        "element_configuration_descriptor_schemas": {
            "generalized": GENERALIZED_ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA,
            "layered": ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA,
        },
        "initial_fields": (
            "MEMBRANE_UNCHANGED_BENDING_STRESS_AND_CURVATURE_PRESTRAIN_"
            "MULTIPLIED_BY_POLARITY"
        ),
        "layer_order": (
            "STATION_MAJOR_LAYER_MINOR_PHYSICAL_DIRECTOR_BOTTOM_TO_TOP"
        ),
        "material_orientation": (
            "PHYSICAL_SURFACE_DIRECTION_AND_SIGNED_IN_PLANE_ANGLE_REMAIN_"
            "INVARIANT_UNDER_DIRECTOR_REEXPRESSION"
        ),
        "offset_policy": (
            "SIGNED_OFFSET_REVERSES_WITH_DIRECTOR_POLARITY_SO_THE_PHYSICAL_"
            "SECTION_ORIGIN_AND_REFERENCE_SURFACE_REMAIN_FIXED"
        ),
        "owner_normal": (
            "AUTHORITATIVE_SHEET_AREA_ORIENTATION_INDEPENDENT_OF_ELEMENT_"
            "DIRECTOR_POLARITY"
        ),
        "pressure_and_follower_tangent": (
            "OWNER_NORMAL_ORIENTATION_FOR_ALL_SIX_D3_ACTIONS_INDEPENDENT_OF_"
            "POLARITY"
        ),
        "recovery": (
            "LOCAL_FIRST_MOMENTS_AND_SHEAR_CHANGE_SIGN_GLOBAL_PHYSICAL_"
            "RESULTANTS_ARE_INVARIANT_AND_TOP_BOTTOM_SWAP"
        ),
        "state_schema_compatibility": (
            "OLD_PAYLOADS_MISSING_DIRECTOR_POLARITY_REFERENCE_SURFACE_OFFSET_"
            "OR_GENERALIZED_INITIAL_FIELDS_IDENTITY_FAIL_CLOSED"
        ),
        "state_schemas": {
            "generalized": GENERALIZED_NONLINEAR_STATE_SCHEMA,
            "layered": NONLINEAR_STATE_SCHEMA,
        },
        "thickness_coordinate": (
            "BOTTOM_IS_MINUS_ALONG_PHYSICAL_DIRECTOR_TOP_IS_PLUS_ALONG_"
            "PHYSICAL_DIRECTOR"
        ),
        "virtual_work": (
            "GLOBAL_NODAL_FORCE_AND_TANGENT_WORK_INVARIANT_UNDER_PURE_"
            "DIRECTOR_REEXPRESSION"
        ),
    }
    assert contract["runtime_authority"] == {
        "current_state_eigen_activity_policy": (
            "ACTIVE_ONLY_REJECT_SOFTENED_FAILED_NONAUTHORITATIVE_AND_DELETED_"
            "FROZEN_NONCURRENT_BEFORE_MECHANICS"
        ),
        "current_state_eigen_activity_policy_id": (
            CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID
        ),
        "current_state_input_ownership_policy_id": (
            COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID
        ),
        "prestress_operator_authority_policy_id": (
            QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID
        ),
        "q4_activity_disposition_schema_id": Q4_ACTIVITY_DISPOSITION_SCHEMA_ID,
        "q4_deleted_frozen_policy_id": Q4_DELETED_FROZEN_POLICY_ID,
        "q4_failed_state_policy_id": Q4_FAILED_STATE_POLICY_ID,
        "q4_quadrature_authority_id": Q4_QUADRATURE_AUTHORITY_ID,
        "qualified_checkpoint_lifecycle_policy_id": (
            QUALIFIED_CHECKPOINT_LIFECYCLE_POLICY_ID
        ),
        "reference_transient_activity_policy": (
            "ACTIVE_REFERENCE_ONLY_REJECT_SOFTENED_OR_HARD_DELETED_MODEL_"
            "ACTIVITY_BEFORE_MECHANICS"
        ),
        "reference_transient_policy_id": (
            QUALIFIED_REFERENCE_TRANSIENT_AUTHORITY_POLICY_ID
        ),
        "s3_activity_disposition_schema_id": S3_ACTIVITY_DISPOSITION_SCHEMA_ID,
        "s3_deleted_frozen_policy_id": S3_DELETED_FROZEN_POLICY_ID,
        "s3_failed_state_policy_id": S3_FAILED_STATE_POLICY_ID,
        "s3_quadrature_authority_id": S3_QUADRATURE_AUTHORITY_ID,
    }
    assert contract["eigen_workflows"] == {
        "broad_buckling": (
            "PARITY_GAP_UNTAGGED_OR_NONCOMMITTED_PRESTRESS_SCOPE_NOT_QUALIFIED"
        ),
        "current_state_buckling": {
            "activity_policy": (
                "ACTIVE_COMMITTED_LIFECYCLE_ONLY_FAILED_AND_DELETED_"
                "NONCURRENT_STATES_REJECTED_BEFORE_MECHANICS"
            ),
            "activity_policy_id": CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID,
            "admitted_scope": (
                "QUALIFIED_Q4_ONLY_OR_QUALIFIED_S3_ONLY_OR_EXACT_MIXED_"
                "QUALIFIED_Q4_S3_COMMITTED_STATES"
            ),
            "destabilizing_operator": (
                "NEGATIVE_INTERNAL_TENSION_POSITIVE_STRESS_HESSIAN_"
                "COMPRESSION_POSITIVE"
            ),
            "kinematic_scope": (
                "Q4_ADDITIVE_ROTATION_VON_KARMAN_AND_S3_NATIVE_MULTIPLICATIVE_"
                "TOTAL_LAGRANGIAN_WITHOUT_CROSS_FAMILY_OBJECTIVITY_CLAIM"
            ),
            "input_ownership_policy_id": (
                COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID
            ),
            "load_scaling": (
                "DESTABILIZING_OPERATOR_SCALES_LINEARLY_AND_FACTORS_SCALE_"
                "INVERSELY"
            ),
            "policy_id": CURRENT_STATE_BUCKLING_POLICY_ID,
            "q4_or_mixed_policy_id": (
                QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID
            ),
            "reference_surface_offset_scope": (
                "Q4_ZERO_OFFSET_ONLY_AND_S3_NATIVE_SIGNED_OFFSET_WITH_NO_Q4_"
                "OFFSET_PARITY_CLAIM"
            ),
            "route_policy_id": COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID,
            "session_policy": (
                "TRANSIENT_MATRICES_AND_FACTORS_NEVER_PERSISTED_OR_SESSION_"
                "CACHED"
            ),
            "stabilizing_operator": (
                "PER_FORMULATION_COMMITTED_ALGORITHMIC_MATERIAL_AND_NUMERICAL_"
                "TANGENT"
            ),
            "state_authority": (
                "ALL_STATES_PREVALIDATED_BEFORE_COMPONENT_MECHANICS_Q4_"
                "CONFIGURATION_AND_ACCEPTED_ALGORITHMIC_ORIGIN_SEAL_AND_S3_"
                "MODEL_BOUND_ROTATION_INTEGRITY"
            ),
            "status": "PARITY_REPLACED",
        },
        "current_state_modal": {
            "activity_policy": (
                "ACTIVE_COMMITTED_LIFECYCLE_ONLY_FAILED_AND_DELETED_"
                "NONCURRENT_STATES_REJECTED_BEFORE_MECHANICS"
            ),
            "activity_policy_id": CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID,
            "admitted_scope": (
                "QUALIFIED_Q4_ONLY_OR_QUALIFIED_S3_ONLY_OR_EXACT_MIXED_"
                "QUALIFIED_Q4_S3_COMMITTED_STATES"
            ),
            "component_assembly_policy_id": (
                COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID
            ),
            "input_ownership_policy_id": (
                COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID
            ),
            "mass_policy": (
                "FORMULATION_CONSISTENT_REFERENCE_MASS_WITH_DECLARED_ZERO_"
                "INERTIA_ALGEBRAIC_COORDINATES"
            ),
            "policy_id": CURRENT_STATE_MODAL_POLICY_ID,
            "route_policy_id": COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID,
            "session_policy": (
                "TRANSIENT_MATRICES_AND_FACTORS_NEVER_PERSISTED_OR_SESSION_"
                "CACHED"
            ),
            "state_authority": (
                "ALL_STATES_PREVALIDATED_BEFORE_COMPONENT_MECHANICS_Q4_"
                "CONFIGURATION_AND_ACCEPTED_ALGORITHMIC_ORIGIN_SEAL_AND_S3_"
                "MODEL_BOUND_ROTATION_INTEGRITY"
            ),
            "status": "PARITY_REPLACED",
            "tangent_source": (
                "EXACT_COMMITTED_CURRENT_TANGENT_COMPONENT_ASSEMBLY_TOTAL"
            ),
        },
        "reference_elastic_buckling": {
            "policy_id": (
                "REFERENCE_ELASTIC_BUBBLE_CONDENSED_INITIAL_STRESS_V1"
            ),
            "status": "PARITY_REPLACED",
        },
        "reference_elastic_prestressed_modal": {
            "policy_id": (
                "MATERIAL_TANGENT_MINUS_COMPRESSION_POSITIVE_GEOMETRIC_V1"
            ),
            "status": "PARITY_REPLACED",
        },
    }
    assert contract["dynamic_policy"]["descriptor_modal_policy"] == DESCRIPTOR_MODAL_POLICY_ID
    assert float(contract["dynamic_policy"]["descriptor_shift_ratio"]) == (
        DESCRIPTOR_SHIFT_RATIO
    )
    assert contract["dynamic_policy"]["descriptor_bounded_coordinate_solver"] == (
        "STATIC_CONDENSATION_THROUGH_REDUCED_DIMENSION_3072"
    )
    assert int(contract["dynamic_policy"]["descriptor_coordinate_shear_limit"]) == (
        int(DESCRIPTOR_COORDINATE_SHEAR_LIMIT)
    )
    assert DESCRIPTOR_DENSE_CONDENSATION_LIMIT == 3072
    assert contract["dynamic_policy"]["descriptor_large_system_policy"] == (
        "SWAPPED_PENCIL_WITH_FAIL_CLOSED_COORDINATE_SHEAR"
    )
    assert contract["dynamic_policy"]["descriptor_transient_policy"] == (
        DESCRIPTOR_TRANSIENT_POLICY_ID
    )
    assert contract["dynamic_policy"]["descriptor_transient_static_policy"] == (
        DESCRIPTOR_TRANSIENT_STATIC_POLICY_ID
    )
    assert contract["dynamic_policy"][
        "descriptor_transient_first_order_policy"
    ] == DESCRIPTOR_TRANSIENT_FIRST_ORDER_POLICY_ID
    assert contract["dynamic_policy"][
        "descriptor_transient_constrained_policy"
    ] == DESCRIPTOR_TRANSIENT_CONSTRAINED_POLICY_ID
    assert int(
        contract["dynamic_policy"][
            "descriptor_transient_coordinate_shear_limit"
        ]
    ) == int(DESCRIPTOR_COORDINATE_SHEAR_LIMIT)
    assert contract["dynamic_policy"][
        "descriptor_transient_effective_factorization"
    ] == "PRECERTIFIED_SPD_MATRIX_CLASS_NO_CALLER_GENERAL_FALLBACK"
    assert contract["geometric_stiffness_policy"] == {
        "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
        "bubble_reduction": "Gb_TRANSPOSE_KG_FULL_Gb_SCHUR_DERIVATIVE",
        "component_frame": "NUMBERED_LOCAL_SYMMETRIC_TENSORS_AT_SEVEN_ORDERED_STIFFNESS_STATIONS",
        "director_field": "[u+z*theta_2,v-z*theta_1,w]",
        "excluded_terms": "TRANSVERSE_SHEAR_NORMAL_STRESS_AND_FINITE_PRESTRESS_RECONDENSATION",
        "generalized_section_second_moment": "EXPLICIT_H_REQUIRED_FOR_NONZERO_N_OR_M",
        "homogeneous_elastic_second_moment_default": "H=N*h^2/12_ONLY_WITH_FROZEN_PROFILE",
        "homogeneous_elastic_stress_profile": HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID,
        "integration": "DUNAVANT_DEGREE_5_SEVEN_POINT",
        "numerical_fields": "PL_AND_DRILL_EXCLUDED",
        "nonzero_state_authority": "EXPLICIT_REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_REQUIRED",
        "policy_id": GEOMETRIC_STIFFNESS_POLICY_ID,
        "resultant_inputs": "COMPRESSION_POSITIVE_N_M_H",
        "resultant_second_moment": "EXPLICIT_H_OR_FROZEN_HOMOGENEOUS_PROFILE_REQUIRED_FOR_EVERY_NONZERO_N_OR_M",
        "schur_linearization": "DERIVATIVE_AT_REFERENCE_MATERIAL_TANGENT",
    }


def test_contract_binds_signed_reference_surface_offset_parity() -> None:
    assert _contract()["reference_surface_offset"] == {
        "contact_surfaces": (
            "MATERIAL_MIDSURFACE_AND_SIGNED_TOP_BOTTOM_COORDINATES_ARE_Z_MINUS_"
            "OFFSET_FROM_THE_NODAL_REFERENCE_SURFACE"
        ),
        "default": 0.0,
        "direct_couples": (
            "GLOBAL_NODAL_AND_ELEMENT_COUPLES_REMAIN_DIRECTLY_CONJUGATE_TO_"
            "GLOBAL_ROTATION_COORDINATES"
        ),
        "generalized_section_origin": (
            "A_B_D_AND_GENERALIZED_RESULTANTS_ARE_DEFINED_AT_THE_PHYSICAL_"
            "SECTION_ORIGIN"
        ),
        "geometric_resultant_shift": (
            "M_REFERENCE=M_SECTION-OFFSET*N_AND_H_REFERENCE=H_SECTION-2*OFFSET*"
            "M_SECTION+OFFSET_SQUARED*N"
        ),
        "internal_virtual_work": (
            "SECTION_ORIGIN_STRAIN_EQUALS_REFERENCE_SURFACE_STRAIN_MINUS_OFFSET_"
            "TIMES_CURVATURE_WITH_CONSISTENT_FORCE_AND_SCHUR_TANGENT_PULLBACK"
        ),
        "layer_coordinate": (
            "Z_REFERENCE=Z_SECTION-OFFSET_WITH_Z_SECTION_POSITIVE_ALONG_THE_"
            "PHYSICAL_DIRECTOR"
        ),
        "mass_shift_id": REFERENCE_SURFACE_MASS_SHIFT_ID,
        "mass_translation_rotation_coupling": (
            "FULL_NODAL_PLUS_BUBBLE_CONSISTENT_MASS_U_RY_PLUS_M1_AND_V_RX_"
            "MINUS_M1_BEFORE_GUYAN_REDUCTION"
        ),
        "polarity_transform": (
            "DIRECTOR_POLARITY_AND_SIGNED_OFFSET_REVERSE_TOGETHER_WHILE_THE_"
            "PHYSICAL_OFFSET_VECTOR_REMAINS_FIXED"
        ),
        "pressure_surface": (
            "DEAD_AND_FOLLOWER_PRESSURE_ARE_NORMAL_TRACTIONS_ON_THE_NODAL_"
            "REFERENCE_SURFACE_WITH_NO_ARTIFICIAL_OFFSET_COUPLE"
        ),
        "public_api": "ADDITIVE_FINITE_SIGNED_LENGTH_DEFAULT_ZERO",
        "recovery_coordinates": (
            "MATERIAL_SECTION_COORDINATES_AND_LAYER_ORDER_ARE_PHYSICAL_WHILE_"
            "REFERENCE_SURFACE_COORDINATES_REMAIN_SEPARATELY_IDENTIFIED"
        ),
        "restart_policy": (
            "OFFSET_VALUE_AND_ALL_THREE_POLICY_IDENTITIES_ARE_STRICT_ELEMENT_"
            "CONFIGURATION_AND_STATE_FINGERPRINT_INPUTS"
        ),
        "signed_convention": (
            "X_REFERENCE=X_SECTION_ORIGIN+OFFSET_TIMES_PHYSICAL_DIRECTOR"
        ),
        "strain_transform_id": REFERENCE_SURFACE_STRAIN_TRANSFORM_ID,
        "zero_offset": (
            "PRESERVES_THE_PREOFFSET_NUMERICAL_PATH_AND_ARRAY_VALUES_WHERE_NO_"
            "SCHEMA_OR_FINGERPRINT_IS_OBSERVED"
        ),
    }


def test_contract_binds_committed_current_tangent_decomposition() -> None:
    tangent = _contract()["native_nonlinear_kinematics"][
        "committed_current_tangent"
    ]
    assert tangent == {
        "bubble_projection": (
            "G_VERTICAL_STACK_I_AND_T_PROJECTS_BOTH_COMPONENTS_WITH_THE_SAME_T"
        ),
        "bubble_projection_policy_id": (
            CURRENT_STATE_BUBBLE_PROJECTION_POLICY_ID
        ),
        "bubble_sensitivity": (
            "T_EQUALS_MINUS_KAA_TOTAL_INVERSE_KAQ_TOTAL_AFTER_CONVERGED_"
            "BUBBLE_SOLVE"
        ),
        "geometric_component": (
            "RESULTANT_OR_STRESS_HESSIAN_TENSION_POSITIVE"
        ),
        "material_component": (
            "CONSTITUTIVE_ALGORITHMIC_PLUS_OBJECTIVE_PL_MATERIAL_NUMERICAL"
        ),
        "persistence": (
            "READ_ONLY_TRANSIENT_API_NO_MATRIX_SENSITIVITY_OR_FACTORIZATION_"
            "STATE"
        ),
        "policy_id": CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
        "state_immutability": (
            "CANONICAL_INPUT_BYTES_AND_INTEGRITY_DIGEST_VERIFIED_UNCHANGED"
        ),
        "total_closure": (
            "KMATERIAL_PLUS_KGEOMETRIC_EQUALS_EXISTING_CONSISTENT_TOTAL_TANGENT"
        ),
        "uncondensed_split": (
            "LAYERED_AND_STATELESS_GENERALIZED_CONSTITUTIVE_VERSUS_RESULTANT_"
            "HESSIAN"
        ),
    }


def test_quadrature_decimal_authority_integrates_reference_area() -> None:
    contract = _contract()
    rows = contract["quadrature"]["stiffness"]["points_r_s_weight"]
    total = sum(Decimal(row[2]) for row in rows)
    assert abs(total - Decimal("0.5")) <= Decimal("2e-15")
    for r_text, s_text, weight_text in rows:
        r = Decimal(r_text)
        s = Decimal(s_text)
        weight = Decimal(weight_text)
        assert r >= 0 and s >= 0 and r + s <= Decimal(1) + Decimal("1e-15")
        assert weight > 0
