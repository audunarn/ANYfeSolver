from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from anymaterial import Hill48Yield, OrthotropicMaterial
from anyfesolver.fe_core import Material
from anyfesolver.e4_pl_s3_state import (
    BUBBLE_STATE_ROLE,
    CANONICALIZATION_ID,
    DIRECTOR_INCREMENT_MAP_ID,
    DIRECTOR_GAUGE_ID,
    EXTERNAL_COORDINATE_LAYOUT_ID,
    EXTERNAL_ROTATION_MAP_ID,
    FORMULATION_ID,
    NONLINEAR_KINEMATICS_ID,
    NONLINEAR_PL_ENERGY_POLICY_ID,
    NONLINEAR_POLICY_ID,
    NONLINEAR_STATE_LAYOUT_ID,
    NONLINEAR_STATE_SCHEMA,
    NONLINEAR_STATE_VERSION,
    NODAL_ROTATION_UPDATE_POLICY_ID,
    PL_MINIMUM_TWIST_DENOMINATOR_ID,
    PL_PHASE_MARGIN_ID,
    PL_PHASE_POLICY_ID,
    PL_TWIST_POLICY_ID,
    QUADRATURE_ID,
    RECOVERY_POLICY_ID,
    SUPPORTED_LOBATTO_LAYER_COUNTS,
    S3CommittedStateError,
    SURFACE_ROTATION_POLICY_ID,
    build_element_configuration_descriptor,
    build_state_identity,
    canonical_json_bytes,
    canonical_sha256,
    element_configuration_fingerprint,
    formulation_fingerprint,
    initialize_zero_committed_s3_state,
    material_fingerprint,
    node_order_fingerprint,
    reference_corner_directors_fingerprint,
    reference_frame_fingerprint,
    reference_geometry_fingerprint,
    reconstruct_director_triad,
    resolved_material_descriptor,
    seal_committed_s3_state,
    strict_canonical_json_loads,
    validate_committed_s3_state as _validate_committed_s3_state,
)


NODE_IDS = (11, 7, 23)
ELEMENT_ID = 41
REFERENCE_COORDINATES = np.asarray(
    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.25, 1.5, 0.0]],
    dtype=float,
)
REFERENCE_FRAME = np.eye(3, dtype=float)
MATERIAL = Material("steel", 210.0e9, 0.3, density=7850.0)
MATERIAL_DESCRIPTOR = resolved_material_descriptor(MATERIAL)
ELEMENT_DESCRIPTOR = build_element_configuration_descriptor(
    thickness=0.012,
    reference_normal=[0.0, 0.0, 1.0],
    director_polarity=1,
    material_direction=[1.0, 0.0, 0.0],
    material_angle_deg=0.0,
    shell_section=None,
)


def _identity(
    *,
    node_ids: tuple[int, int, int] = NODE_IDS,
    coordinates: np.ndarray = REFERENCE_COORDINATES,
    reference_frame: np.ndarray = REFERENCE_FRAME,
    material: dict[str, object] = MATERIAL_DESCRIPTOR,
    element_id: int = ELEMENT_ID,
    element: dict[str, object] = ELEMENT_DESCRIPTOR,
    initial_fields: dict[str, object] | None = None,
    initial_field_provenance: dict[str, object] | None = None,
    num_layers: int = 3,
    material_symmetry: str = "isotropic",
    equivalent_stress_measure: str = "von_mises",
) -> dict[str, object]:
    return build_state_identity(
        element_id=element_id,
        node_ids=node_ids,
        reference_coordinates=coordinates,
        reference_frame=reference_frame,
        element_descriptor=element,
        material_descriptor=material,
        num_layers=num_layers,
        material_symmetry=material_symmetry,
        equivalent_stress_measure=equivalent_stress_measure,
        initial_fields=initial_fields,
        initial_field_provenance=initial_field_provenance,
    )


def _state(**kwargs: object) -> dict[str, object]:
    options: dict[str, object] = {
        "element_id": ELEMENT_ID,
        "node_ids": NODE_IDS,
        "reference_coordinates": REFERENCE_COORDINATES,
        "reference_frame": REFERENCE_FRAME,
        "element_descriptor": ELEMENT_DESCRIPTOR,
        "material_descriptor": MATERIAL_DESCRIPTOR,
        "num_layers": 3,
    }
    options.update(kwargs)
    return initialize_zero_committed_s3_state(**options)


def validate_committed_s3_state(
    state: dict[str, object],
    *,
    expected_identity: dict[str, object] | None = None,
    **kwargs: object,
) -> dict[str, object]:
    if expected_identity is None:
        expected_identity = _identity(
            initial_fields={
                name: state[name]
                for name in (
                    "initial_membrane_stress",
                    "initial_bending_stress",
                    "initial_membrane_prestrain",
                    "initial_curvature_prestrain",
                )
                if name in state
            },
            initial_field_provenance=state.get("initial_field_provenance", {}),
            num_layers=state.get("num_layers", 3),
            material_symmetry=state.get("material_symmetry", "isotropic"),
            equivalent_stress_measure=state.get(
                "equivalent_stress_measure", "von_mises"
            ),
        )
    return _validate_committed_s3_state(
        state,
        expected_identity=expected_identity,
        **kwargs,
    )


def test_exact_state_ids_and_canonical_fingerprint_are_frozen() -> None:
    assert FORMULATION_ID == "E4_PL_QUALIFIED_S3_COMPANION_V1"
    assert EXTERNAL_COORDINATE_LAYOUT_ID == "S3_EXTERNAL18_BUBBLE2_PL3_V1"
    assert NONLINEAR_STATE_SCHEMA == "anysolver.e4_pl_s3.committed_state.v2"
    assert NONLINEAR_STATE_VERSION == 2
    assert NONLINEAR_STATE_LAYOUT_ID == (
        "S3_TL_Q18_NODE_SO3_PL3_TRIADS4_BUBBLE2_STATION7_LAYERED_V2"
    )
    assert NONLINEAR_KINEMATICS_ID == (
        "MITC3_PLUS_TOTAL_LAGRANGIAN_SHARED_SO3_CORNERS_EQ7_31_V2"
    )
    assert DIRECTOR_GAUGE_ID == (
        "MITC3_PLUS_EQ11_GLOBAL_EY_WITH_EZ_PARALLEL_FALLBACK_V1"
    )
    assert BUBBLE_STATE_ROLE == "RESERVED_ZERO_NEW_INCREMENT_PREDICTOR_ONLY"

    first = formulation_fingerprint()
    second = formulation_fingerprint()
    assert first == second
    assert first == "4ADF654B0D3461BE5D863FD550CBFD888853E8C2D4FA9DE45BF6AFDACC035D4D"

    value = {"z": -0.0, "a": np.asarray([1.0, 2.0])}
    assert canonical_json_bytes(value) == b'{"a":[1.0,2.0],"z":0.0}\n'
    assert canonical_sha256(value) == canonical_sha256(
        {"a": [1.0, 2.0], "z": 0.0}
    )
    with pytest.raises(S3CommittedStateError, match="nonfinite"):
        canonical_json_bytes({"bad": float("nan")})


def test_strict_canonical_json_loader_rejects_alternate_and_malformed_bytes() -> None:
    raw = canonical_json_bytes({"a": [1.0, 2.0], "z": 0.0})
    assert strict_canonical_json_loads(raw) == {"a": [1.0, 2.0], "z": 0.0}
    for malformed in (
        b'{"a":1,"a":2}\n',
        b'{"z":0.0, "a":[1.0,2.0]}\n',
        b'{"a":1}\r\n',
        b'{"a":1}',
        b'\xef\xbb\xbf{"a":1}\n',
        b'{"a":NaN}\n',
        b'{"a":1e999}\n',
        b'\xff',
    ):
        with pytest.raises(S3CommittedStateError):
            strict_canonical_json_loads(malformed)


def test_zero_state_has_exact_owned_shapes_and_separates_bubble_from_alpha() -> None:
    state = _state()
    points = 7 * 3

    assert state["node_ids"] == NODE_IDS
    assert state["element_id"] == ELEMENT_ID
    assert np.asarray(state["reference_corner_directors"]).shape == (3, 3)
    assert np.asarray(state["committed_total_u"]).shape == (18,)
    assert np.asarray(state["committed_nodal_rotation_matrices"]).shape == (3, 3, 3)
    assert np.asarray(state["committed_pl_twist"]).shape == (3,)
    assert np.asarray(state["committed_pl_turn_count"]).shape == (3,)
    assert np.asarray(state["committed_pl_multiplier"]).shape == (3,)
    assert np.asarray(state["committed_pl_internal_force"]).shape == (18,)
    assert state["committed_pl_energy"] == 0.0
    assert np.asarray(state["committed_pl_turn_count"]).dtype == np.dtype(np.int64)
    assert np.asarray(state["committed_director_triads"]).shape == (4, 3, 3)
    assert np.all(np.asarray(state["committed_director_triads"]) == np.eye(3))
    assert np.asarray(state["bubble_rotation_last_increment"]).shape == (2,)
    assert np.asarray(state["alpha"]).shape == (points,)
    assert state["bubble_state_role"] == BUBBLE_STATE_ROLE
    assert state["bubble_rotation_last_increment"] is not state["alpha"]

    for key in (
        "plastic_strain",
        "layer_strain",
        "layer_strain_material",
        "kinematic_layer_strain",
        "layer_stress",
        "layer_stress_material",
    ):
        assert np.asarray(state[key]).shape == (points, 3)
    for key in (
        "initial_membrane_stress",
        "initial_bending_stress",
        "initial_membrane_prestrain",
        "initial_curvature_prestrain",
    ):
        assert np.asarray(state[key]).shape == (7, 3)

    normalized = validate_committed_s3_state(
        state,
        expected_identity=_identity(),
        expected_num_layers=3,
        expected_committed_total_u=np.zeros(18),
    )
    np.asarray(normalized["committed_total_u"])[0] = 4.0
    assert np.asarray(state["committed_total_u"])[0] == 0.0


def test_initial_fields_expand_to_seven_stations_and_bind_provenance() -> None:
    membrane = np.asarray([2.0, -1.0, 0.5])
    bending = np.asarray([[0.3, 0.4, 0.5]])
    curvature = np.arange(21, dtype=float).reshape(7, 3)
    state = _state(
        initial_fields={
            "initial_membrane_stress": membrane,
            "initial_bending_stress": bending,
            "initial_curvature_prestrain": curvature,
        },
        initial_field_provenance={"source": "fixture", "revision": 2},
    )

    assert np.all(np.asarray(state["initial_membrane_stress"]) == membrane)
    assert np.all(np.asarray(state["initial_bending_stress"]) == bending[0])
    assert np.array_equal(state["initial_curvature_prestrain"], curvature)
    assert state["initial_field_provenance"] == {"source": "fixture", "revision": 2}

    changed = copy.deepcopy(state)
    changed["initial_membrane_stress"][0, 0] += 1.0
    with pytest.raises(S3CommittedStateError, match="initial-fields fingerprint"):
        validate_committed_s3_state(changed)

    changed = copy.deepcopy(state)
    changed["initial_field_provenance"]["revision"] = 3
    with pytest.raises(S3CommittedStateError, match="initial-fields fingerprint"):
        validate_committed_s3_state(changed)


def test_committed_state_canonical_round_trip_is_byte_identical_and_owned() -> None:
    fields = {"initial_membrane_stress": np.asarray([3.0, 2.0, 1.0])}
    provenance = {"source": "roundtrip", "revision": 1}
    state = _state(
        initial_fields=fields,
        initial_field_provenance=provenance,
    )
    raw = canonical_json_bytes(state)
    decoded = strict_canonical_json_loads(raw)
    normalized = validate_committed_s3_state(
        decoded,
        expected_identity=_identity(
            initial_fields=fields,
            initial_field_provenance=provenance,
        ),
    )
    assert canonical_json_bytes(normalized) == raw

    decoded["committed_total_u"][0] = 9.0
    assert np.asarray(normalized["committed_total_u"])[0] == 0.0
    np.asarray(normalized["committed_total_u"])[1] = 8.0
    assert np.asarray(state["committed_total_u"])[1] == 0.0

def test_identity_hashes_bind_order_geometry_and_material() -> None:
    base = _identity()
    reordered = _identity(node_ids=(7, 11, 23))
    moved_coordinates = REFERENCE_COORDINATES.copy()
    moved_coordinates[2, 1] += 0.01
    moved = _identity(coordinates=moved_coordinates)
    changed_material = resolved_material_descriptor(
        Material("steel", 205.0e9, 0.3, density=7850.0)
    )
    material = _identity(material=changed_material)
    changed_element = dict(ELEMENT_DESCRIPTOR)
    changed_element["thickness"] = 0.013
    element = _identity(element=changed_element)

    assert base["node_order_fingerprint"] != reordered["node_order_fingerprint"]
    assert base["reference_geometry_fingerprint"] != reordered[
        "reference_geometry_fingerprint"
    ]
    assert base["reference_geometry_fingerprint"] != moved[
        "reference_geometry_fingerprint"
    ]
    assert base["material_fingerprint"] != material["material_fingerprint"]
    assert base["element_configuration_fingerprint"] != element[
        "element_configuration_fingerprint"
    ]
    assert base["formulation_fingerprint"] == reordered["formulation_fingerprint"]

    assert node_order_fingerprint(NODE_IDS) == base["node_order_fingerprint"]
    assert reference_geometry_fingerprint(
        NODE_IDS, REFERENCE_COORDINATES
    ) == base["reference_geometry_fingerprint"]
    assert reference_frame_fingerprint(REFERENCE_FRAME) == base[
        "reference_frame_fingerprint"
    ]
    assert reference_corner_directors_fingerprint(
        NODE_IDS,
        np.repeat(REFERENCE_FRAME[:, 2][None, :], 3, axis=0),
    ) == base["reference_corner_directors_fingerprint"]
    assert material_fingerprint(MATERIAL_DESCRIPTOR) == base["material_fingerprint"]
    assert element_configuration_fingerprint(
        ELEMENT_ID, NODE_IDS, ELEMENT_DESCRIPTOR
    ) == base["element_configuration_fingerprint"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("state_schema", "wrong", "state_schema"),
        ("state_version", True, "state_version"),
        ("commit_status", "trial", "commit_status"),
        ("formulation_id", "legacy", "formulation_id"),
        ("nonlinear_state_layout_id", "old", "nonlinear_state_layout_id"),
        ("nonlinear_kinematics_id", "generic_corotational", "nonlinear_kinematics_id"),
        ("director_gauge_id", "reconstructed_chart", "director_gauge_id"),
        ("external_rotation_map_id", "first_order_projection", "external_rotation_map_id"),
        ("director_increment_map_id", "additive_corner_map", "director_increment_map_id"),
        ("surface_rotation_policy_id", "normalized_edge", "surface_rotation_policy_id"),
        ("pl_twist_policy_id", "additive_drill", "pl_twist_policy_id"),
        ("nodal_rotation_update_policy_id", "right_update", "nodal_rotation_update_policy_id"),
        ("pl_phase_policy_id", "principal_only", "pl_phase_policy_id"),
        ("pl_phase_margin_id", "no_margin", "pl_phase_margin_id"),
        (
            "pl_minimum_twist_denominator_id",
            "no_denominator_guard",
            "pl_minimum_twist_denominator_id",
        ),
        ("nonlinear_pl_energy_policy_id", "secant", "nonlinear_pl_energy_policy_id"),
        ("bubble_convention", "absolute", "bubble_convention"),
        ("bubble_state_role", "absolute_rotation", "bubble_state_role"),
        ("quadrature_id", "centroid", "quadrature_id"),
        ("nonlinear_policy_id", "legacy", "nonlinear_policy_id"),
        ("recovery_policy_id", "legacy", "recovery_policy_id"),
        ("canonicalization_id", "platform_json", "canonicalization_id"),
        ("num_layers", 0, "num_layers"),
        ("material_symmetry", "generalized", "material_symmetry"),
        ("equivalent_stress_measure", "tresca", "equivalent_stress_measure"),
        ("element_id", True, "element_id"),
    ],
)
def test_strict_enums_and_ids_fail_closed(key: str, value: object, message: str) -> None:
    state = _state()
    state[key] = value
    with pytest.raises(S3CommittedStateError, match=message):
        validate_committed_s3_state(state)


def test_checkpoint_self_describes_every_restart_fingerprint_component() -> None:
    state = _state()
    identity = _identity()

    assert state["formulation_id"] == FORMULATION_ID
    assert state["external_coordinate_layout_id"] == EXTERNAL_COORDINATE_LAYOUT_ID
    assert state["nonlinear_state_layout_id"] == NONLINEAR_STATE_LAYOUT_ID
    assert state["nonlinear_kinematics_id"] == NONLINEAR_KINEMATICS_ID
    assert state["director_gauge_id"] == DIRECTOR_GAUGE_ID
    assert state["external_rotation_map_id"] == EXTERNAL_ROTATION_MAP_ID
    assert state["reference_corner_directors_fingerprint"] == identity[
        "reference_corner_directors_fingerprint"
    ]
    assert state["director_increment_map_id"] == DIRECTOR_INCREMENT_MAP_ID
    assert state["surface_rotation_policy_id"] == SURFACE_ROTATION_POLICY_ID
    assert state["pl_twist_policy_id"] == PL_TWIST_POLICY_ID
    assert state["nodal_rotation_update_policy_id"] == (
        NODAL_ROTATION_UPDATE_POLICY_ID
    )
    assert state["pl_phase_policy_id"] == PL_PHASE_POLICY_ID
    assert state["pl_phase_margin_id"] == PL_PHASE_MARGIN_ID
    assert state["pl_minimum_twist_denominator_id"] == (
        PL_MINIMUM_TWIST_DENOMINATOR_ID
    )
    assert state["nonlinear_pl_energy_policy_id"] == (
        NONLINEAR_PL_ENERGY_POLICY_ID
    )
    assert state["bubble_convention"] == "hierarchical_rotation_relative_to_corner_average"
    assert state["quadrature_id"] == QUADRATURE_ID
    assert state["nonlinear_policy_id"] == NONLINEAR_POLICY_ID
    assert state["recovery_policy_id"] == RECOVERY_POLICY_ID
    assert state["canonicalization_id"] == CANONICALIZATION_ID


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("reference_corner_directors", np.zeros((2, 3))),
        ("committed_total_u", np.zeros(17)),
        ("committed_nodal_rotation_matrices", np.zeros((2, 3, 3))),
        ("committed_pl_twist", np.zeros(2)),
        ("committed_pl_turn_count", np.zeros(2, dtype=np.int64)),
        ("committed_pl_multiplier", np.zeros(2)),
        ("committed_pl_internal_force", np.zeros(17)),
        ("committed_director_triads", np.zeros((3, 3, 3))),
        ("bubble_rotation_last_increment", np.zeros(3)),
        ("committed_internal_force", np.zeros(17)),
        ("station_generalized_strain", np.zeros((6, 8))),
        ("station_generalized_resultant", np.zeros((7, 7))),
        ("plastic_strain", np.zeros((20, 3))),
        ("alpha", np.zeros(20)),
        ("initial_membrane_stress", np.zeros((1, 3))),
    ],
)
def test_shape_mutations_fail_closed(key: str, bad_value: np.ndarray) -> None:
    state = _state()
    state[key] = bad_value
    with pytest.raises(S3CommittedStateError, match=key):
        validate_committed_s3_state(state)


@pytest.mark.parametrize(
    ("key", "index", "value"),
    [
        ("reference_corner_directors", (0, 0), float("nan")),
        ("committed_total_u", 0, float("nan")),
        ("committed_nodal_rotation_matrices", (0, 0, 0), float("inf")),
        ("committed_pl_twist", 0, float("nan")),
        ("committed_pl_multiplier", 1, float("-inf")),
        ("committed_pl_internal_force", 2, float("nan")),
        ("bubble_rotation_last_increment", 1, float("inf")),
        ("plastic_strain", (0, 0), float("-inf")),
        ("alpha", 0, -1.0e-12),
    ],
)
def test_nonfinite_and_negative_history_fail_closed(
    key: str, index: object, value: float
) -> None:
    state = _state()
    state[key][index] = value
    expected = "nonnegative" if key == "alpha" else "nonfinite"
    with pytest.raises(S3CommittedStateError, match=expected):
        validate_committed_s3_state(state)


@pytest.mark.parametrize("bad", [1, True, np.float64(1.0), float("inf")])
def test_committed_pl_energy_requires_finite_canonical_binary64(bad: object) -> None:
    state = _state()
    state["committed_pl_energy"] = bad
    with pytest.raises(S3CommittedStateError, match="committed_pl_energy.*binary64"):
        validate_committed_s3_state(state)


@pytest.mark.parametrize(
    "bad",
    (
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.float64),
        [0.0, 0.0, 0.0],
        [False, 0, 0],
    ),
)
def test_pl_turn_count_requires_canonical_signed_integer_values(bad: object) -> None:
    state = _state()
    state["committed_pl_turn_count"] = bad
    with pytest.raises(S3CommittedStateError, match="committed_pl_turn_count"):
        validate_committed_s3_state(state)


@pytest.mark.parametrize(
    "key",
    (
        "reference_corner_directors",
        "committed_nodal_rotation_matrices",
        "committed_pl_twist",
        "committed_pl_multiplier",
        "committed_pl_internal_force",
    ),
)
def test_v2_float_arrays_require_canonical_binary64(key: str) -> None:
    state = _state()
    state[key] = np.asarray(state[key], dtype=np.float32)
    with pytest.raises(S3CommittedStateError, match=f"{key}.*binary64"):
        validate_committed_s3_state(state)


def test_node_shared_so3_copy_and_director_relation_fail_closed() -> None:
    nonunit_reference = _state()
    nonunit_reference["reference_corner_directors"][0, 2] = 2.0
    with pytest.raises(S3CommittedStateError, match="unit physical director"):
        validate_committed_s3_state(nonunit_reference)

    improper = _state()
    improper["committed_nodal_rotation_matrices"][0, 0, 0] = -1.0
    with pytest.raises(S3CommittedStateError, match="proper SO\\(3\\)"):
        validate_committed_s3_state(improper)

    contradicted = _state()
    angle = 0.25
    contradicted["committed_nodal_rotation_matrices"][0] = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(angle), -math.sin(angle)),
            (0.0, math.sin(angle), math.cos(angle)),
        ),
        dtype=np.float64,
    )
    with pytest.raises(S3CommittedStateError, match="node-shared rotation operator"):
        validate_committed_s3_state(contradicted)


def test_canonical_pl_turn_count_uses_open_minus_pi_closed_plus_pi() -> None:
    state = _state()
    state["committed_pl_twist"] = np.asarray(
        (math.pi, -math.pi, 3.0 * math.pi), dtype=np.float64
    )
    state["committed_pl_turn_count"] = np.asarray((0, -1, 1), dtype=np.int64)
    normalized = validate_committed_s3_state(
        seal_committed_s3_state(state), expected_identity=_identity()
    )
    principal = normalized["committed_pl_twist"] - (
        2.0 * math.pi * normalized["committed_pl_turn_count"]
    )
    np.testing.assert_array_equal(principal, np.full(3, math.pi))

    noncanonical = copy.deepcopy(normalized)
    noncanonical["committed_pl_turn_count"][0] = -1
    with pytest.raises(S3CommittedStateError, match="turn count"):
        validate_committed_s3_state(
            seal_committed_s3_state(noncanonical), expected_identity=_identity()
        )


def test_integrity_digest_binds_every_v2_rotation_and_pl_field() -> None:
    base = _state()
    baseline_digest = base["state_integrity_sha256"]
    digests: dict[str, str] = {}

    for key in (
        "reference_corner_directors",
        "committed_nodal_rotation_matrices",
        "committed_pl_twist",
        "committed_pl_turn_count",
        "committed_pl_multiplier",
        "committed_pl_internal_force",
        "committed_pl_energy",
    ):
        draft = copy.deepcopy(base)
        if key == "reference_corner_directors":
            draft[key][0] = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
            draft["committed_nodal_rotation_matrices"][0] = np.asarray(
                ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            draft["committed_director_triads"][0] = reconstruct_director_triad(
                (1.0, 0.0, 0.0)
            )
        elif key == "committed_nodal_rotation_matrices":
            draft[key][0] = np.asarray(
                ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
        elif key == "committed_pl_twist":
            draft[key][0] = 0.125
        elif key == "committed_pl_turn_count":
            draft["committed_pl_twist"][0] = 2.0 * math.pi
            draft[key][0] = 1
        elif key == "committed_pl_multiplier":
            draft[key][0] = 0.5
        elif key == "committed_pl_internal_force":
            draft[key][0] = 0.75
        elif key == "committed_pl_energy":
            draft[key] = 1.25

        sealed = seal_committed_s3_state(draft)
        digest = sealed["state_integrity_sha256"]
        assert digest != baseline_digest
        digests[key] = digest
        if key == "reference_corner_directors":
            with pytest.raises(
                S3CommittedStateError,
                match="reference corner directors.*model-bound fingerprint",
            ):
                validate_committed_s3_state(
                    sealed, expected_identity=_identity()
                )
        else:
            validate_committed_s3_state(
                sealed, expected_identity=_identity()
            )

    assert len(set(digests.values())) == len(digests)


def test_director_triads_reject_shear_and_left_handed_orientation() -> None:
    sheared = _state()
    sheared["committed_director_triads"][1, 0, 1] = 1.0e-8
    with pytest.raises(S3CommittedStateError, match="orthonormal"):
        validate_committed_s3_state(sheared)

    reflected = _state()
    reflected["committed_director_triads"][3, :, 2] *= -1.0
    with pytest.raises(S3CommittedStateError, match="right-handed"):
        validate_committed_s3_state(reflected)

    transported = _state()
    angle = 0.2
    spin = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transported["committed_director_triads"][0] = spin
    with pytest.raises(S3CommittedStateError, match="Eq. \\(11\\) gauge"):
        validate_committed_s3_state(transported)


def test_eq11_gauge_fallback_and_reference_chart_identity_are_bound() -> None:
    fallback = reconstruct_director_triad([0.0, 1.0, 0.0])
    np.testing.assert_allclose(fallback.T @ fallback, np.eye(3), atol=2.0e-15)
    assert np.linalg.det(fallback) == pytest.approx(1.0, abs=2.0e-15)
    np.testing.assert_array_equal(fallback[:, 2], [0.0, 1.0, 0.0])

    spun = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    base = _state()
    with pytest.raises(S3CommittedStateError, match="admitted numbered geometry"):
        _state(reference_frame=spun)
    assert reference_frame_fingerprint(spun) != base[
        "reference_frame_fingerprint"
    ]


def test_unknown_missing_and_fingerprint_mutations_fail_closed() -> None:
    state = _state()
    state["unexpected"] = 1
    with pytest.raises(S3CommittedStateError, match="unknown=.*unexpected"):
        validate_committed_s3_state(state)

    state = _state()
    state[1] = "non-string key"
    with pytest.raises(S3CommittedStateError, match="keys must be strings"):
        validate_committed_s3_state(state)

    state = _state()
    del state["alpha"]
    with pytest.raises(S3CommittedStateError, match="missing=.*alpha"):
        validate_committed_s3_state(state)

    for key in (
        "formulation_fingerprint",
        "element_configuration_fingerprint",
        "node_order_fingerprint",
        "reference_geometry_fingerprint",
        "reference_corner_directors_fingerprint",
        "material_fingerprint",
        "initial_fields_fingerprint",
    ):
        state = _state()
        state[key] = "0" * 64
        with pytest.raises(S3CommittedStateError, match="fingerprint"):
            validate_committed_s3_state(state, expected_identity=_identity())


def test_current_model_identity_and_committed_displacement_must_match_exactly() -> None:
    state = _state()
    changed_material = resolved_material_descriptor(
        Material("steel", 210.0e9, 0.29, density=7850.0)
    )
    with pytest.raises(S3CommittedStateError, match="material_fingerprint"):
        validate_committed_s3_state(
            state,
            expected_identity=_identity(material=changed_material),
        )

    expected_u = np.zeros(18)
    expected_u[4] = 1.0e-15
    with pytest.raises(S3CommittedStateError, match="global displacement slice"):
        validate_committed_s3_state(
            state,
            expected_identity=_identity(),
            expected_committed_total_u=expected_u,
        )

    changed_fields = {
        "initial_membrane_stress": np.asarray([1.0, 0.0, 0.0]),
    }
    with pytest.raises(S3CommittedStateError, match="initial_fields_fingerprint"):
        validate_committed_s3_state(
            state,
            expected_identity=_identity(initial_fields=changed_fields),
        )


def test_initializer_rejects_invalid_identity_frame_fields_and_material_values() -> None:
    with pytest.raises(S3CommittedStateError, match="distinct node"):
        _state(node_ids=(11, 11, 23))

    coordinates = REFERENCE_COORDINATES.copy()
    coordinates[0, 0] = float("nan")
    with pytest.raises(S3CommittedStateError, match="nonfinite"):
        _state(reference_coordinates=coordinates)

    reflected = REFERENCE_FRAME.copy()
    reflected[:, 2] *= -1.0
    with pytest.raises(S3CommittedStateError, match="right-handed"):
        _state(reference_frame=reflected)

    with pytest.raises(S3CommittedStateError, match="unknown qualified S3 initial fields"):
        _state(initial_fields={"legacy_drill_stress": np.zeros(3)})

    bad_material = copy.deepcopy(MATERIAL_DESCRIPTOR)
    bad_material["resolved_material"]["fields"]["elastic_modulus"] = float(
        "inf"
    )
    with pytest.raises(S3CommittedStateError, match="nonfinite"):
        _state(material_descriptor=bad_material)


@pytest.mark.parametrize(
    "bad",
    [
        [True] + [0.0] * 17,
        ["0.0"] * 18,
        np.zeros(18, dtype=complex),
        np.zeros(18, dtype=object),
    ],
)
def test_numeric_state_arrays_reject_bool_string_complex_and_object(bad: object) -> None:
    state = _state()
    state["committed_total_u"] = bad
    with pytest.raises(S3CommittedStateError, match="binary64"):
        validate_committed_s3_state(state)


def test_persisted_state_rejects_lossy_integer_and_non_binary64_arrays() -> None:
    state = _state()
    values = [0.0] * 18
    values[0] = 2**53 + 1
    state["committed_total_u"] = values
    with pytest.raises(S3CommittedStateError, match="binary64"):
        validate_committed_s3_state(state)

    state = _state()
    state["committed_total_u"] = np.zeros(18, dtype=np.float32)
    with pytest.raises(S3CommittedStateError, match="binary64"):
        validate_committed_s3_state(state)


@pytest.mark.parametrize("count", [1, 2, 4, 6, 12, True])
def test_initializer_accepts_only_the_frozen_lobatto_layer_counts(count: object) -> None:
    with pytest.raises(S3CommittedStateError, match="num_layers"):
        _state(num_layers=count)
    assert SUPPORTED_LOBATTO_LAYER_COUNTS == (3, 5, 7, 9, 11)


def test_state_validation_requires_current_model_identity() -> None:
    with pytest.raises(TypeError, match="expected_identity"):
        _validate_committed_s3_state(_state())


def test_hostile_scalar_types_are_normalized_to_state_error() -> None:
    state = _state()
    state["material_symmetry"] = []
    with pytest.raises(S3CommittedStateError, match="material_symmetry"):
        validate_committed_s3_state(state)

    state = _state()
    state["state_schema"] = np.asarray([NONLINEAR_STATE_SCHEMA])
    with pytest.raises(S3CommittedStateError, match="state_schema"):
        validate_committed_s3_state(state)

    with pytest.raises(S3CommittedStateError, match="expected_num_layers"):
        validate_committed_s3_state(_state(num_layers=3), expected_num_layers=True)


def test_orthotropic_state_accepts_only_bound_equivalent_measure() -> None:
    orthotropic = OrthotropicMaterial(
        name="ortho",
        elastic_modulus_1=140.0e9,
        elastic_modulus_2=12.0e9,
        elastic_modulus_3=12.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.25,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=5.0e9,
        shear_modulus_23=4.0e9,
        density=1600.0,
        hill_yield=Hill48Yield(
            X=900.0e6,
            Y=700.0e6,
            Z=700.0e6,
            S12=400.0e6,
            S13=400.0e6,
            S23=350.0e6,
        ),
    )
    descriptor = resolved_material_descriptor(orthotropic)
    state = _state(
        material_descriptor=descriptor,
        material_symmetry="orthotropic",
        equivalent_stress_measure="hill48",
    )
    assert state["material_symmetry"] == "orthotropic"
    assert state["equivalent_stress_measure"] == "hill48"
    validate_committed_s3_state(
        state,
        expected_identity=_identity(
            material=descriptor,
            material_symmetry="orthotropic",
            equivalent_stress_measure="hill48",
        ),
    )

    with pytest.raises(S3CommittedStateError, match="isotropic state"):
        _state(material_symmetry="isotropic", equivalent_stress_measure="hill48")
