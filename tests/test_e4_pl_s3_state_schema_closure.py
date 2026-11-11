from __future__ import annotations

import copy
from dataclasses import dataclass, make_dataclass
import math
from typing import Any

import numpy as np
import pytest

from anymaterial import Hill48Yield, OrthotropicMaterial
import anyfesolver.e4_pl_s3_state as state_module
from anyfesolver.e4_pl_s3_state import (
    BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
    GENERALIZED_RESULTANT_COMPONENT_ORDER,
    GENERALIZED_STRAIN_COMPONENT_ORDER,
    LAYER_STRAIN_COMPONENT_ORDER,
    LAYER_STRESS_COMPONENT_ORDER,
    LOBATTO_NORMALIZED_TABLES,
    STATE_ARRAY_LAYOUT_ID,
    STATE_FIELD_MANIFEST,
    STIFFNESS_STATION_TABLE,
    S3CommittedStateError,
    build_element_configuration_descriptor,
    build_state_identity,
    canonical_json_bytes,
    formulation_fingerprint_payload,
    initialize_zero_committed_s3_state,
    lobatto_table_fingerprint,
    material_fingerprint,
    qualified_s3_lobatto_layers,
    qualified_s3_triangle_frame,
    reconstruct_director_triad,
    require_qualified_s3_quality,
    resolved_material_descriptor,
    seal_committed_s3_state,
    strict_canonical_json_loads,
    stiffness_station_table_fingerprint,
    validate_committed_s3_state,
)
from anyfesolver.fe_core import Material


ELEMENT_ID = 41
NODE_IDS = (11, 7, 23)
THICKNESS = 0.012
REFERENCE_COORDINATES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.25, 1.5, 0.0),
    ),
    dtype=np.float64,
)
OWNER_NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
REFERENCE_FRAME = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
MATERIAL = Material("steel", 210.0e9, 0.3, density=7850.0)
MATERIAL_DESCRIPTOR = resolved_material_descriptor(MATERIAL)
ELEMENT_DESCRIPTOR = build_element_configuration_descriptor(
    thickness=THICKNESS,
    reference_normal=OWNER_NORMAL,
    director_polarity=1,
    material_direction=(1.0, 0.0, 0.0),
    material_angle_deg=0.0,
    shell_section=None,
)


EXPECTED_STIFFNESS_STATIONS = (
    (1.0 / 3.0, 1.0 / 3.0, 0.1125),
    (0.470142064105115, 0.470142064105115, 0.066197076394253),
    (0.059715871789770, 0.470142064105115, 0.066197076394253),
    (0.470142064105115, 0.059715871789770, 0.066197076394253),
    (0.101286507323456, 0.101286507323456, 0.062969590272414),
    (0.797426985353087, 0.101286507323456, 0.062969590272414),
    (0.101286507323456, 0.797426985353087, 0.062969590272414),
)

EXPECTED_LOBATTO_NORMALIZED = {
    3: (
        (-1.0, 0.0, 1.0),
        (1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0),
    ),
    5: (
        (-1.0, -math.sqrt(3.0 / 7.0), 0.0, math.sqrt(3.0 / 7.0), 1.0),
        (1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0),
    ),
    7: (
        (
            -1.0,
            -0.830223896278567,
            -0.468848793470714,
            0.0,
            0.468848793470714,
            0.830223896278567,
            1.0,
        ),
        (
            0.047619047619048,
            0.276826047361566,
            0.431745381209863,
            0.487619047619048,
            0.431745381209863,
            0.276826047361566,
            0.047619047619048,
        ),
    ),
    9: (
        (
            -1.0,
            -0.899757995411460,
            -0.677186279510738,
            -0.363117463826178,
            0.0,
            0.363117463826178,
            0.677186279510738,
            0.899757995411460,
            1.0,
        ),
        (
            0.027777777777778,
            0.165495361560806,
            0.274538712500162,
            0.346428510973046,
            0.371519274376417,
            0.346428510973046,
            0.274538712500162,
            0.165495361560806,
            0.027777777777778,
        ),
    ),
    11: (
        (
            -1.0,
            -0.934001430408059,
            -0.784483473663144,
            -0.565235326996205,
            -0.295758135586939,
            0.0,
            0.295758135586939,
            0.565235326996205,
            0.784483473663144,
            0.934001430408059,
            1.0,
        ),
        (
            0.018181818181818,
            0.109612273266995,
            0.187169881780305,
            0.248048104264028,
            0.286879124779008,
            0.300217595455691,
            0.286879124779008,
            0.248048104264028,
            0.187169881780305,
            0.109612273266995,
            0.018181818181818,
        ),
    ),
}


def _identity(
    *,
    material_descriptor: dict[str, Any] = MATERIAL_DESCRIPTOR,
    coordinates: np.ndarray = REFERENCE_COORDINATES,
    reference_frame: np.ndarray = REFERENCE_FRAME,
    num_layers: int = 3,
    material_symmetry: str = "isotropic",
    equivalent_stress_measure: str = "von_mises",
) -> dict[str, Any]:
    return build_state_identity(
        element_id=ELEMENT_ID,
        node_ids=NODE_IDS,
        reference_coordinates=coordinates,
        reference_frame=reference_frame,
        element_descriptor=ELEMENT_DESCRIPTOR,
        material_descriptor=material_descriptor,
        num_layers=num_layers,
        material_symmetry=material_symmetry,
        equivalent_stress_measure=equivalent_stress_measure,
        initial_fields=None,
        initial_field_provenance=None,
    )


def _state(
    *,
    material_descriptor: dict[str, Any] = MATERIAL_DESCRIPTOR,
    num_layers: int = 3,
    material_symmetry: str = "isotropic",
    equivalent_stress_measure: str = "von_mises",
) -> dict[str, Any]:
    return initialize_zero_committed_s3_state(
        element_id=ELEMENT_ID,
        node_ids=NODE_IDS,
        reference_coordinates=REFERENCE_COORDINATES,
        reference_frame=REFERENCE_FRAME,
        element_descriptor=ELEMENT_DESCRIPTOR,
        material_descriptor=material_descriptor,
        num_layers=num_layers,
        material_symmetry=material_symmetry,
        equivalent_stress_measure=equivalent_stress_measure,
    )


def _orthotropic_descriptor(*, with_hill48: bool) -> dict[str, Any]:
    hill = None
    if with_hill48:
        hill = Hill48Yield(
            X=900.0e6,
            Y=700.0e6,
            Z=700.0e6,
            S12=400.0e6,
            S13=400.0e6,
            S23=350.0e6,
        )
    material = OrthotropicMaterial(
        name="orthotropic",
        elastic_modulus_1=140.0e9,
        elastic_modulus_2=12.0e9,
        elastic_modulus_3=10.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=4.5e9,
        shear_modulus_23=4.0e9,
        density=1600.0,
        hill_yield=hill,
    )
    return resolved_material_descriptor(material)


def _as_plain_lobatto_tables() -> dict[int, tuple[tuple[float, ...], tuple[float, ...]]]:
    return {
        int(count): (
            tuple(float(value) for value in points),
            tuple(float(value) for value in weights),
        )
        for count, points, weights in LOBATTO_NORMALIZED_TABLES
    }


def test_component_schemas_and_frames_are_explicit_and_not_conflated() -> None:
    assert GENERALIZED_STRAIN_COMPONENT_ORDER == (
        "membrane_xx",
        "membrane_yy",
        "membrane_xy_engineering",
        "curvature_xx",
        "curvature_yy",
        "curvature_xy_engineering",
        "transverse_shear_xz_engineering",
        "transverse_shear_yz_engineering",
    )
    assert GENERALIZED_RESULTANT_COMPONENT_ORDER == (
        "membrane_force_xx",
        "membrane_force_yy",
        "membrane_force_xy_tensor",
        "bending_moment_xx",
        "bending_moment_yy",
        "bending_moment_xy_tensor",
        "transverse_shear_force_xz_tensor",
        "transverse_shear_force_yz_tensor",
    )
    assert LAYER_STRAIN_COMPONENT_ORDER == (
        "inplane_xx",
        "inplane_yy",
        "inplane_xy_engineering",
    )
    assert LAYER_STRESS_COMPONENT_ORDER == (
        "inplane_xx",
        "inplane_yy",
        "inplane_xy_tensor",
    )
    assert GENERALIZED_STRAIN_COMPONENT_ORDER != GENERALIZED_RESULTANT_COMPONENT_ORDER
    assert LAYER_STRAIN_COMPONENT_ORDER != LAYER_STRESS_COMPONENT_ORDER

    expected = {
        "station_generalized_strain": (
            GENERALIZED_STRAIN_COMPONENT_ORDER,
            "numbered_reference_engineering_strain",
        ),
        "station_generalized_resultant": (
            GENERALIZED_RESULTANT_COMPONENT_ORDER,
            "numbered_reference_tensor_resultant",
        ),
        "layer_strain": (
            LAYER_STRAIN_COMPONENT_ORDER,
            "numbered_reference_engineering_strain",
        ),
        "layer_stress": (
            LAYER_STRESS_COMPONENT_ORDER,
            "numbered_reference_tensor_stress",
        ),
        "layer_strain_material": (
            LAYER_STRAIN_COMPONENT_ORDER,
            "physical_material_engineering_strain",
        ),
        "kinematic_layer_strain": (
            LAYER_STRAIN_COMPONENT_ORDER,
            "numbered_reference_engineering_strain",
        ),
        "plastic_strain": (
            LAYER_STRAIN_COMPONENT_ORDER,
            "physical_material_engineering_strain",
        ),
        "layer_stress_material": (
            LAYER_STRESS_COMPONENT_ORDER,
            "physical_material_tensor_stress",
        ),
    }
    for field, (component_order, component_frame) in expected.items():
        assert tuple(STATE_FIELD_MANIFEST[field]["component_order"]) == component_order
        assert STATE_FIELD_MANIFEST[field]["component_frame"] == component_frame

    assert STATE_ARRAY_LAYOUT_ID == "S3_STATION_MAJOR_LAYER_BOTTOM_TO_TOP_MINOR_V1"
    for field in (
        "plastic_strain",
        "alpha",
        "layer_strain",
        "layer_strain_material",
        "kinematic_layer_strain",
        "layer_stress",
        "layer_stress_material",
    ):
        assert STATE_FIELD_MANIFEST[field]["point_order"] == (
            "station_major_layer_minor_physical_director_bottom_to_top"
        )


def test_published_surface_station_table_is_exact_and_fingerprint_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(tuple(row) for row in STIFFNESS_STATION_TABLE) == (
        EXPECTED_STIFFNESS_STATIONS
    )
    assert sum(row[2] for row in STIFFNESS_STATION_TABLE) == pytest.approx(
        0.5, rel=0.0, abs=2.0e-15
    )
    baseline = stiffness_station_table_fingerprint()
    assert len(baseline) == 64 and baseline == baseline.upper()
    assert formulation_fingerprint_payload()["stiffness_station_table_sha256"] == baseline

    changed = list(STIFFNESS_STATION_TABLE)
    r, s, weight = changed[4]
    changed[4] = (r, s, weight + 1.0e-15)
    monkeypatch.setattr(state_module, "STIFFNESS_STATION_TABLE", tuple(changed))
    assert stiffness_station_table_fingerprint() != baseline


def test_lobatto_tables_are_exact_bottom_to_top_and_fingerprint_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _as_plain_lobatto_tables() == EXPECTED_LOBATTO_NORMALIZED
    baseline = lobatto_table_fingerprint()
    assert len(baseline) == 64 and baseline == baseline.upper()
    assert formulation_fingerprint_payload()["lobatto_table_sha256"] == baseline

    changed = list(LOBATTO_NORMALIZED_TABLES)
    count, points, weights = changed[2]
    changed[2] = (count, points, (*weights[:-1], weights[-1] + 1.0e-15))
    monkeypatch.setattr(state_module, "LOBATTO_NORMALIZED_TABLES", tuple(changed))
    assert lobatto_table_fingerprint() != baseline


@pytest.mark.parametrize("num_layers", (3, 5, 7, 9, 11))
def test_qualified_lobatto_scaling_and_flattening_are_exact(
    num_layers: int,
) -> None:
    normalized_points, normalized_weights = EXPECTED_LOBATTO_NORMALIZED[num_layers]
    z, weights = qualified_s3_lobatto_layers(num_layers, THICKNESS)
    np.testing.assert_array_equal(
        z,
        0.5 * THICKNESS * np.asarray(normalized_points, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        weights,
        0.5 * THICKNESS * np.asarray(normalized_weights, dtype=np.float64),
    )
    assert np.all(np.diff(z) > 0.0)
    assert z[0] == -0.5 * THICKNESS
    assert z[-1] == 0.5 * THICKNESS
    assert np.sum(weights) == pytest.approx(THICKNESS, rel=0.0, abs=2.0e-15)

    state = _state(num_layers=num_layers)
    points = 7 * num_layers
    marker = np.arange(points, dtype=np.float64)
    draft = copy.deepcopy(state)
    draft["layer_strain"][:, 0] = marker
    resealed = seal_committed_s3_state(draft)
    validated = validate_committed_s3_state(
        resealed,
        expected_identity=_identity(num_layers=num_layers),
    )
    for station in range(7):
        for layer in range(num_layers):
            flat = station * num_layers + layer
            assert validated["layer_strain"][flat, 0] == float(flat)


@dataclass
class _FakeIsotropicMaterial:
    name: str
    elastic_modulus: float
    poisson_ratio: float
    density: float
    yield_stress: float
    hardening_curve: object | None

    @property
    def elastic_symmetry(self) -> str:
        return "isotropic"


@dataclass
class _EvilNestedState:
    payload: float


def _spoofed_isotropic_material() -> Any:
    spoof = make_dataclass(
        "SpoofedIsotropicMaterial",
        (
            ("name", str),
            ("elastic_modulus", float),
            ("poisson_ratio", float),
            ("density", float),
            ("yield_stress", float),
            ("hardening_curve", object),
        ),
    )
    spoof.__module__ = type(MATERIAL).__module__
    spoof.__qualname__ = type(MATERIAL).__qualname__
    spoof.elastic_symmetry = property(lambda _self: "isotropic")
    return spoof("evil", 210.0e9, 0.3, 7850.0, 0.0, None)


def test_material_descriptors_accept_only_real_supported_material_types() -> None:
    fake = _FakeIsotropicMaterial("fake", 210.0e9, 0.3, 7850.0, 0.0, None)
    with pytest.raises(
        S3CommittedStateError,
        match="registered.*StructuralMaterial|supported material type",
    ):
        resolved_material_descriptor(fake)

    with pytest.raises(
        S3CommittedStateError,
        match="registered.*StructuralMaterial|supported material type",
    ):
        resolved_material_descriptor(_spoofed_isotropic_material())

    evil_nested = copy.deepcopy(MATERIAL)
    evil_nested.hardening_curve = _EvilNestedState(1.0)
    with pytest.raises(S3CommittedStateError, match="supported.*hardening|state-bearing"):
        resolved_material_descriptor(evil_nested)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_material_descriptor_rejects_missing_or_extra_nested_fields(
    mutation: str,
) -> None:
    descriptor = copy.deepcopy(MATERIAL_DESCRIPTOR)
    fields = descriptor["resolved_material"]["fields"]
    if mutation == "missing":
        del fields["poisson_ratio"]
    else:
        fields["legacy_drill_modulus"] = 1.0
    with pytest.raises(S3CommittedStateError, match="material.*fields|keys mismatch"):
        material_fingerprint(descriptor)

    hill_descriptor = _orthotropic_descriptor(with_hill48=True)
    hill_fields = hill_descriptor["resolved_material"]["fields"]["hill_yield"][
        "fields"
    ]
    if mutation == "missing":
        del hill_fields["X"]
    else:
        hill_fields["legacy_label"] = "hill"
    with pytest.raises(S3CommittedStateError, match="Hill48|fields|keys mismatch"):
        material_fingerprint(hill_descriptor)


def test_material_descriptor_recomputes_hill48_label_and_compliance() -> None:
    hill = _orthotropic_descriptor(with_hill48=True)
    hill["equivalent_stress_measure"] = "von_mises"
    with pytest.raises(S3CommittedStateError, match="hill48|equivalent"):
        material_fingerprint(hill)

    no_hill = _orthotropic_descriptor(with_hill48=False)
    no_hill["equivalent_stress_measure"] = "hill48"
    with pytest.raises(S3CommittedStateError, match="hill48|equivalent"):
        material_fingerprint(no_hill)

    invalid_compliance = _orthotropic_descriptor(with_hill48=False)
    invalid_compliance["elastic_compliance_engineering_6x6"][0][0] = -1.0
    with pytest.raises(
        S3CommittedStateError,
        match="compliance|positive definite|orthotropic",
    ):
        material_fingerprint(invalid_compliance)

    stale_compliance = _orthotropic_descriptor(with_hill48=False)
    stale_compliance["resolved_material"]["fields"]["poisson_ratio_12"] = 5.0
    with pytest.raises(
        S3CommittedStateError,
        match="compliance|descriptor|orthotropic",
    ):
        material_fingerprint(stale_compliance)


@pytest.mark.parametrize(
    ("field", "index", "value"),
    (
        ("committed_total_u", 0, 1.0e-12),
        ("committed_internal_force", 4, -3.0),
        ("plastic_strain", (4, 2), 1.0e-6),
        ("alpha", 4, 1.0e-4),
        ("layer_strain", (7, 0), -2.0e-5),
        ("layer_strain_material", (8, 1), 3.0e-5),
        ("layer_stress_material", (11, 1), -6.0),
    ),
)
def test_whole_state_integrity_detects_otherwise_valid_finite_cache_mutation(
    field: str,
    index: object,
    value: float,
) -> None:
    state = _state()
    assert state["thickness"] == THICKNESS
    assert len(state["state_integrity_sha256"]) == 64
    mutated = copy.deepcopy(state)
    mutated[field][index] = value
    with pytest.raises(S3CommittedStateError, match="state integrity"):
        validate_committed_s3_state(mutated, expected_identity=_identity())


def test_whole_state_integrity_covers_semantically_consistent_redundant_caches() -> None:
    strain = _state()
    strain["station_generalized_strain"][2, 0] = 0.125
    start = 2 * 3
    strain["kinematic_layer_strain"][start : start + 3, 0] = 0.125
    with pytest.raises(S3CommittedStateError, match="state integrity"):
        validate_committed_s3_state(strain, expected_identity=_identity())

    stress = _state()
    station = 6
    start = station * 3
    z, weights = qualified_s3_lobatto_layers(3, THICKNESS)
    stress["layer_stress"][start : start + 3, 0] = 5.0
    stress["station_generalized_resultant"][station, 0] = float(
        np.sum(weights * 5.0)
    )
    stress["station_generalized_resultant"][station, 3] = float(
        np.sum(weights * z * 5.0)
    )
    with pytest.raises(S3CommittedStateError, match="state integrity"):
        validate_committed_s3_state(stress, expected_identity=_identity())


def test_state_seal_is_deterministic_canonical_and_covers_every_manifest_field() -> None:
    state = _state()
    original_hash = state["state_integrity_sha256"]
    draft = copy.deepcopy(state)
    draft["committed_internal_force"][3] = 0.125
    draft["layer_stress_material"][5, 1] = 17.0
    first = seal_committed_s3_state(draft)
    second = seal_committed_s3_state(draft)
    assert first["state_integrity_sha256"] == second["state_integrity_sha256"]
    assert first["state_integrity_sha256"] != original_hash
    assert canonical_json_bytes(first) == canonical_json_bytes(second)

    decoded = strict_canonical_json_loads(canonical_json_bytes(first))
    validated = validate_committed_s3_state(decoded, expected_identity=_identity())
    assert canonical_json_bytes(validated) == canonical_json_bytes(first)
    assert set(STATE_FIELD_MANIFEST) <= set(validated)


def test_triangle_frame_is_derived_from_admitted_geometry_and_owner_normal() -> None:
    frame, local, quality = qualified_s3_triangle_frame(
        REFERENCE_COORDINATES, OWNER_NORMAL
    )
    np.testing.assert_array_equal(frame, REFERENCE_FRAME)
    np.testing.assert_array_equal(local[0], (0.0, 0.0))
    assert quality["connectivity_sign"] == 1.0
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=2.0e-15)
    assert np.linalg.det(frame) == pytest.approx(1.0, rel=0.0, abs=2.0e-15)

    identity = _identity(reference_frame=frame)
    assert identity["reference_frame_fingerprint"]

    angle = 0.2
    alternate_chart = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    with pytest.raises(
        S3CommittedStateError,
        match="derived reference frame|admitted numbered geometry",
    ):
        _identity(reference_frame=alternate_chart)


@pytest.mark.parametrize(
    "coordinates",
    (
        np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        np.asarray(((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.01, 0.02, 0.0))),
        REFERENCE_COORDINATES[[0, 2, 1]],
    ),
)
def test_triangle_frame_rejects_degenerate_unadmitted_or_reversed_geometry(
    coordinates: np.ndarray,
) -> None:
    with pytest.raises(
        S3CommittedStateError,
        match="geometry|quality|winding|triangle|distinct|area",
    ):
        qualified_s3_triangle_frame(coordinates, OWNER_NORMAL)


def test_bubble_predictor_is_zero_at_every_commit_restart_and_chart_switch() -> None:
    assert BUBBLE_PREDICTOR_COMMIT_POLICY_ID == (
        "RESET_TO_ZERO_AFTER_EVERY_ACCEPTED_STEP_V1"
    )
    state = _state()
    np.testing.assert_array_equal(state["bubble_rotation_last_increment"], (0.0, 0.0))

    for value in (np.nextafter(0.0, 1.0), -1.0e-300, 1.0e-12):
        draft = copy.deepcopy(state)
        draft["bubble_rotation_last_increment"][0] = value
        with pytest.raises(S3CommittedStateError, match="bubble predictor.*zero"):
            seal_committed_s3_state(draft)

    switched = copy.deepcopy(state)
    switched_triad = reconstruct_director_triad((0.0, 1.0, 0.0))
    switched["committed_director_triads"] = np.repeat(
        switched_triad[np.newaxis, :, :], 4, axis=0
    )
    rotation_z_to_y = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    switched["committed_nodal_rotation_matrices"] = np.repeat(
        rotation_z_to_y[np.newaxis, :, :], 3, axis=0
    )
    switched["bubble_rotation_last_increment"][:] = 0.0
    sealed = seal_committed_s3_state(switched)
    validate_committed_s3_state(sealed, expected_identity=_identity())

    raw = canonical_json_bytes(sealed)
    restarted = strict_canonical_json_loads(raw)
    restarted["bubble_rotation_last_increment"][1] = 1.0e-15
    with pytest.raises(S3CommittedStateError, match="bubble predictor.*zero"):
        validate_committed_s3_state(restarted, expected_identity=_identity())


def test_formulation_fingerprint_binds_every_tying_point_and_mechanics_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = state_module.formulation_fingerprint()
    scalar_mutations = (
        ("BUBBLE_OFFSET_D", np.nextafter(state_module.BUBBLE_OFFSET_D, 1.0)),
        ("BUBBLE_OFFSET_EXACT", "1/9999"),
        ("BUBBLE_POLYNOMIAL_SCALE", 26.0),
        ("PL_BASIS_ID", state_module.PL_BASIS_ID + "_MUTATED"),
        ("PL_CONSTRAINT_ID", state_module.PL_CONSTRAINT_ID + "_MUTATED"),
        ("PL_GRAM_SCALE_ID", state_module.PL_GRAM_SCALE_ID + "_MUTATED"),
        ("PL_BLOCK_SIGN_ID", state_module.PL_BLOCK_SIGN_ID + "_MUTATED"),
        ("PL_CONDENSATION_ID", state_module.PL_CONDENSATION_ID + "_MUTATED"),
        (
            "EXTERNAL_ROTATION_MAP_ID",
            state_module.EXTERNAL_ROTATION_MAP_ID + "_MUTATED",
        ),
        (
            "DIRECTOR_INCREMENT_MAP_ID",
            state_module.DIRECTOR_INCREMENT_MAP_ID + "_MUTATED",
        ),
        (
            "SURFACE_ROTATION_POLICY_ID",
            state_module.SURFACE_ROTATION_POLICY_ID + "_MUTATED",
        ),
        ("PL_TWIST_POLICY_ID", state_module.PL_TWIST_POLICY_ID + "_MUTATED"),
        (
            "NODAL_ROTATION_UPDATE_POLICY_ID",
            state_module.NODAL_ROTATION_UPDATE_POLICY_ID + "_MUTATED",
        ),
        ("PL_PHASE_POLICY_ID", state_module.PL_PHASE_POLICY_ID + "_MUTATED"),
        ("PL_PHASE_MARGIN_ID", state_module.PL_PHASE_MARGIN_ID + "_MUTATED"),
        (
            "PL_MINIMUM_TWIST_DENOMINATOR_ID",
            state_module.PL_MINIMUM_TWIST_DENOMINATOR_ID + "_MUTATED",
        ),
        (
            "NONLINEAR_PL_ENERGY_POLICY_ID",
            state_module.NONLINEAR_PL_ENERGY_POLICY_ID + "_MUTATED",
        ),
        (
            "PL_PHASE_MARGIN",
            np.nextafter(state_module.PL_PHASE_MARGIN, math.inf),
        ),
        (
            "PL_MINIMUM_TWIST_DENOMINATOR",
            np.nextafter(
                state_module.PL_MINIMUM_TWIST_DENOMINATOR, math.inf
            ),
        ),
        (
            "DRILL_SCALE_POLICY_ID",
            state_module.DRILL_SCALE_POLICY_ID + "_MUTATED",
        ),
        ("MITC3_PLUS_SOURCE_SHA256", "0" * 64),
        ("MITC3_PLUS_NONLINEAR_SOURCE_SHA256", "1" * 64),
    )
    for name, replacement in scalar_mutations:
        with monkeypatch.context() as isolated:
            isolated.setattr(state_module, name, replacement)
            assert state_module.formulation_fingerprint() != baseline

    structural_mutations = (
        ("PL_GRAM_NUMERATOR", ((3.0, 1.0, 1.0),) * 3),
        ("DRILL_SCALE_PROJECTOR", ((2.0, 0.0), (-1.0, 0.0), (0.0, 1.0))),
        ("DRILL_SCALE_METRIC", ((1.0, 0.0), (0.0, 0.5))),
        (
            "DRILL_SCALE_INVERSE_METRIC_SQRT",
            ((1.0, 0.0), (0.0, math.sqrt(2.0))),
        ),
        (
            "MITC3_PLUS_EQUATION_MAP",
            {**state_module.MITC3_PLUS_EQUATION_MAP, "bubble_interpolation": "wrong"},
        ),
        (
            "MITC3_PLUS_NONLINEAR_EQUATION_MAP",
            {
                **state_module.MITC3_PLUS_NONLINEAR_EQUATION_MAP,
                "director_update": "wrong",
            },
        ),
    )
    for name, replacement in structural_mutations:
        with monkeypatch.context() as isolated:
            isolated.setattr(state_module, name, replacement)
            assert state_module.formulation_fingerprint() != baseline

    for label in "ABCDEF":
        changed = dict(state_module.TYING_POINTS)
        changed[label] = (
            np.nextafter(changed[label][0], 1.0),
            changed[label][1],
        )
        with monkeypatch.context() as isolated:
            isolated.setattr(state_module, "TYING_POINTS", changed)
            assert state_module.formulation_fingerprint() != baseline


def test_state_field_manifest_binds_all_initial_field_semantics() -> None:
    expected = {
        "initial_membrane_stress": (
            "authoritative_initial_field",
            "numbered_reference_tensor_stress",
            LAYER_STRESS_COMPONENT_ORDER,
        ),
        "initial_bending_stress": (
            "authoritative_initial_field",
            "numbered_reference_tensor_stress",
            LAYER_STRESS_COMPONENT_ORDER,
        ),
        "initial_membrane_prestrain": (
            "authoritative_initial_field",
            "numbered_reference_engineering_strain",
            LAYER_STRAIN_COMPONENT_ORDER,
        ),
        "initial_curvature_prestrain": (
            "authoritative_initial_field",
            "numbered_reference_engineering_curvature",
            LAYER_STRAIN_COMPONENT_ORDER,
        ),
    }
    for name, (role, frame, order) in expected.items():
        assert STATE_FIELD_MANIFEST[name] == {
            "role": role,
            "component_frame": frame,
            "component_order": order,
            "point_order": "ordered_surface_stations",
        }
    assert STATE_FIELD_MANIFEST["initial_field_provenance"] == {
        "role": "authoritative_initial_field_metadata",
        "component_frame": "canonical_metadata",
        "component_order": ("sorted_string_keys",),
        "point_order": "not_applicable",
    }


def test_state_field_manifest_binds_node_shared_so3_and_objective_pl_history() -> None:
    assert STATE_FIELD_MANIFEST["reference_corner_directors"] == {
        "role": "immutable_element_owned_reference_with_strict_polarity",
        "component_frame": "global",
        "component_order": ("x", "y", "z"),
        "point_order": "ordered_connectivity_nodes",
    }
    assert STATE_FIELD_MANIFEST["committed_nodal_rotation_matrices"] == {
        "role": "authoritative_node_shared_redundant_copy",
        "component_frame": "global_operator_rows_columns",
        "component_order": ("global_x", "global_y", "global_z"),
        "point_order": "ordered_connectivity_nodes",
    }
    assert STATE_FIELD_MANIFEST["committed_pl_twist"] == {
        "role": "authoritative_unwrapped_phase",
        "component_frame": "relative_surface_nodal_twist",
        "component_order": ("scalar_angle_radians",),
        "point_order": "ordered_connectivity_nodes",
    }
    assert STATE_FIELD_MANIFEST["committed_pl_turn_count"] == {
        "role": "authoritative_unwrapped_phase_integer",
        "component_frame": "relative_surface_nodal_twist",
        "component_order": ("integer_turn_count",),
        "point_order": "ordered_connectivity_nodes",
    }
    assert STATE_FIELD_MANIFEST["committed_pl_multiplier"] == {
        "role": "numerical_diagnostic",
        "component_frame": "barycentric_pl_constraint",
        "component_order": ("tau_equals_kd_times_twist",),
        "point_order": "ordered_connectivity_nodes",
    }
    assert STATE_FIELD_MANIFEST["committed_pl_internal_force"] == {
        "role": "numerical_diagnostic",
        "component_frame": "global",
        "component_order": (
            "force_global_x",
            "force_global_y",
            "force_global_z",
            "moment_global_x",
            "moment_global_y",
            "moment_global_z",
        ),
        "point_order": "ordered_connectivity_nodes",
    }
    assert STATE_FIELD_MANIFEST["committed_pl_energy"] == {
        "role": "numerical_diagnostic",
        "component_frame": "scalar",
        "component_order": ("energy",),
        "point_order": "single_element",
    }


def test_exported_quality_guard_is_closed_finite_and_checks_every_gate() -> None:
    _frame, _local, quality = qualified_s3_triangle_frame(
        REFERENCE_COORDINATES, OWNER_NORMAL
    )
    require_qualified_s3_quality(quality)

    for key in quality:
        changed = dict(quality)
        changed[key] = math.nan
        with pytest.raises(S3CommittedStateError, match="finite"):
            require_qualified_s3_quality(changed)

    for changed in (
        {key: value for key, value in quality.items() if key != "area"},
        {**quality, "unknown_metric": 1.0},
    ):
        with pytest.raises(S3CommittedStateError, match="keys mismatch"):
            require_qualified_s3_quality(changed)

    rejected_values = {
        "area": 0.0,
        "normalized_twice_area": state_module.MINIMUM_NORMALIZED_TWICE_AREA,
        "reference_normal_alignment": state_module.MINIMUM_OWNER_NORMAL_ALIGNMENT,
        "minimum_angle_deg": state_module.MINIMUM_ANGLE_DEG - 1.0e-6,
        "maximum_angle_deg": state_module.MAXIMUM_ANGLE_DEG + 1.0e-6,
        "edge_ratio": state_module.MAXIMUM_EDGE_RATIO + 1.0e-6,
        "minimum_scaled_jacobian": (
            state_module.MINIMUM_CORNER_SCALED_JACOBIAN - 1.0e-6
        ),
        "normalized_area": state_module.MINIMUM_NORMALIZED_AREA - 1.0e-6,
        "connectivity_sign": 0.0,
    }
    for key, value in rejected_values.items():
        changed = dict(quality)
        changed[key] = value
        with pytest.raises(S3CommittedStateError, match="admission failed"):
            require_qualified_s3_quality(changed)
