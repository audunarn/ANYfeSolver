"""P2 committed-state, identity, and restart checks for mixed GE-Beam3."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.fe_core import FEModel
from anyfesolver.ge_beam3_mixed_element import GeometricallyExactBeam3D3NElement
from anyfesolver.ge_beam3_mixed_state import (
    GE_BEAM3_MIXED_CANDIDATE_ID,
    SECTION_SCHEMA,
    STATE_INTEGRITY_ID,
    STATE_LAYOUT_ID,
    STATE_SCHEMA,
    GeBeam3MixedCommittedStateError,
    canonical_json_bytes,
    canonical_sha256,
    deserialize_ge_beam3_mixed_state,
    element_identity_payload,
    element_identity_sha256,
    encode_typed_array,
    initialize_ge_beam3_mixed_state,
    model_bound_identity,
    seal_committed_ge_beam3_mixed_state,
    section_descriptor_payload,
    section_descriptor_sha256,
    serialize_ge_beam3_mixed_state,
    strict_canonical_json_loads,
    validate_committed_ge_beam3_mixed_state,
)


def _identity_inputs() -> dict[str, object]:
    geometry = np.array(
        ((1.0, -2.0, 0.5), (1.5, -2.0, 0.5), (2.0, -2.0, 0.5)),
        dtype="<f8",
    )
    triad = np.eye(3, dtype="<f8")
    stiffness = np.diag((1000.0, 500.0, 600.0, 100.0, 120.0, 140.0)).astype("<f8")
    stiffness[0, 3] = stiffness[3, 0] = 5.0
    mass = np.diag((10.0, 10.0, 10.0, 0.5, 0.6, 0.7)).astype("<f8")
    mass[0, 5] = mass[5, 0] = -0.2
    return {
        "element_id": 17,
        "node_ids": np.array((11, 12, 13), dtype="<i8"),
        "reference_geometry": geometry,
        "reference_triad": triad,
        "reference_orientation": np.array((0.0, 2.0, 0.0), dtype="<f8"),
        "reference_axis_direction": np.array((4.0, 0.0, 0.0), dtype="<f8"),
        "section_name": "COUPLED_PHYSICAL",
        "section_stiffness": stiffness,
        "section_mass": mass,
    }


def _state(*, nonzero: bool = False) -> dict[str, object]:
    inputs = _identity_inputs()
    if not nonzero:
        return initialize_ge_beam3_mixed_state(**inputs)
    strain = np.array(
        (
            (0.010, 0.020, -0.010, 0.004, 0.005, -0.003),
            (0.010, 0.020, -0.010, 0.006, 0.007, -0.002),
            (0.015, 0.010, 0.005, 0.006, 0.007, -0.002),
            (0.015, 0.010, 0.005, 0.008, 0.009, 0.001),
        ),
        dtype="<f8",
    )
    stiffness = np.asarray(inputs["section_stiffness"], dtype="<f8")
    resultants = strain @ stiffness.T
    total = np.linspace(-0.02, 0.04, 18, dtype="<f8")
    nodal = np.repeat(np.eye(3, dtype="<f8")[None, :, :], 3, axis=0)
    local = np.repeat(np.eye(3, dtype="<f8")[None, :, :], 2, axis=0)
    return initialize_ge_beam3_mixed_state(
        **inputs,
        committed_total_u=total,
        committed_nodal_rotation_matrices=nodal,
        committed_local_rotation_matrices=local,
        committed_local_moments=resultants[:, 3:6].reshape(2, 2, 3),
        station_generalized_strain=strain,
        station_generalized_resultant=resultants,
    )


def _validate(state: dict[str, object], **overrides: object) -> dict[str, object]:
    inputs = _identity_inputs()
    inputs.update(overrides)
    return validate_committed_ge_beam3_mixed_state(state, **inputs)


def test_typed_arrays_are_little_endian_bit_exact_and_canonical() -> None:
    array = np.array((0.0, -0.0, 1.0, -2.5), dtype="<f8")
    encoded = encode_typed_array(array)
    assert encoded == {
        "data_hex": array.tobytes(order="C").hex().upper(),
        "dtype": "<f8",
        "shape": [4],
    }
    raw = canonical_json_bytes({"array": array, "snowman": "\u2603"})
    assert raw.endswith(b"\n")
    assert b'"dtype":"<f8"' in raw
    assert b"\\u2603" in raw
    assert canonical_sha256({"array": array}) == hashlib.sha256(
        canonical_json_bytes({"array": array})
    ).hexdigest().upper()
    assert canonical_sha256({"array": array}) != canonical_sha256(
        {"array": np.abs(array)}
    )


def test_section_and_element_identity_preimages_are_exact() -> None:
    inputs = _identity_inputs()
    descriptor = section_descriptor_payload(
        section_name=inputs["section_name"],
        section_stiffness=inputs["section_stiffness"],
        section_mass=inputs["section_mass"],
    )
    assert set(descriptor) == {
        "mass_matrix_per_reference_length",
        "name",
        "schema",
        "stiffness_matrix",
    }
    assert descriptor["schema"] == SECTION_SCHEMA
    descriptor_hash = section_descriptor_sha256(
        section_name=inputs["section_name"],
        section_stiffness=inputs["section_stiffness"],
        section_mass=inputs["section_mass"],
    )
    identity_args = {
        "node_ids": inputs["node_ids"],
        "reference_geometry": inputs["reference_geometry"],
        "reference_triad": inputs["reference_triad"],
        "reference_orientation": inputs["reference_orientation"],
        "reference_axis_direction": inputs["reference_axis_direction"],
        "section_descriptor_sha256_value": descriptor_hash,
    }
    identity = element_identity_payload(**identity_args)
    assert set(identity) == {
        "candidate_id",
        "condensation_id",
        "node_ids",
        "quadrature_id",
        "reference_axis_direction",
        "reference_geometry",
        "reference_id",
        "reference_orientation",
        "reference_triad",
        "rotation_id",
        "section_descriptor_sha256",
        "state_layout_id",
    }
    assert identity["candidate_id"] == GE_BEAM3_MIXED_CANDIDATE_ID
    assert identity["state_layout_id"] == STATE_LAYOUT_ID
    assert element_identity_sha256(**identity_args) == canonical_sha256(identity)


def test_initializer_returns_complete_owned_sealed_state() -> None:
    inputs = _identity_inputs()
    state = initialize_ge_beam3_mixed_state(**inputs)
    assert len(state) == 28
    assert state["state_schema"] == STATE_SCHEMA
    assert state["state_integrity_id"] == STATE_INTEGRITY_ID
    assert state["section_states"] == [None, None, None, None]
    assert state["committed_nodal_rotation_matrices"].shape == (3, 3, 3)
    assert state["committed_local_rotation_matrices"].shape == (2, 3, 3)
    for name in (
        "committed_total_u",
        "committed_nodal_rotation_matrices",
        "reference_geometry",
        "reference_triad",
    ):
        assert state[name].flags.owndata
        assert not state[name].flags.writeable
    geometry = inputs["reference_geometry"]
    geometry[0, 0] = 99.0
    assert state["reference_geometry"][0, 0] == 1.0
    validated = _validate(state)
    assert validated is not state
    assert validated["state_integrity_sha256"] == state["state_integrity_sha256"]


def test_round_trip_is_byte_identical_and_arrays_remain_typed() -> None:
    state = _state(nonzero=True)
    raw = serialize_ge_beam3_mixed_state(state)
    decoded_json = strict_canonical_json_loads(raw)
    assert set(decoded_json["committed_total_u"]) == {"data_hex", "dtype", "shape"}
    assert decoded_json["committed_total_u"]["dtype"] == "<f8"
    loaded = deserialize_ge_beam3_mixed_state(raw)
    assert serialize_ge_beam3_mixed_state(loaded) == raw
    for name in (
        "committed_total_u",
        "committed_nodal_rotation_matrices",
        "committed_local_rotation_matrices",
        "station_generalized_strain",
        "station_generalized_resultant",
    ):
        assert loaded[name].dtype.str == "<f8"
        assert loaded[name].tobytes() == state[name].tobytes()
        assert not loaded[name].flags.writeable
    _validate(loaded, expected_committed_total_u=state["committed_total_u"])


@pytest.mark.parametrize(
    "raw",
    (
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{ "a":1}\n',
        b'\xef\xbb\xbf{"a":1}\n',
        b'{"a":1}',
    ),
)
def test_strict_json_rejects_duplicates_nonfinite_and_alternates(raw: bytes) -> None:
    with pytest.raises(GeBeam3MixedCommittedStateError):
        strict_canonical_json_loads(raw)


def test_integrity_and_closed_key_set_reject_mutation() -> None:
    state = _state(nonzero=True)
    changed = dict(state)
    total = np.array(state["committed_total_u"], copy=True)
    total[0] += 1.0
    changed["committed_total_u"] = total
    with pytest.raises(GeBeam3MixedCommittedStateError, match="integrity"):
        _validate(changed)
    changed = dict(state)
    changed["foreign"] = 1
    with pytest.raises(GeBeam3MixedCommittedStateError, match="keys mismatch"):
        _validate(changed)
    changed = dict(state)
    del changed["station_generalized_strain"]
    with pytest.raises(GeBeam3MixedCommittedStateError, match="keys mismatch"):
        seal_committed_ge_beam3_mixed_state(changed)


def test_exact_scalar_type_shape_dtype_and_typed_hex_guards() -> None:
    state = _state()
    changed = dict(state)
    changed["element_id"] = True
    with pytest.raises(GeBeam3MixedCommittedStateError, match="integer"):
        seal_committed_ge_beam3_mixed_state(changed)
    changed = dict(state)
    changed["state_version"] = True
    with pytest.raises(GeBeam3MixedCommittedStateError, match="integer"):
        seal_committed_ge_beam3_mixed_state(changed)
    changed = dict(state)
    changed["committed_total_u"] = np.zeros(17, dtype="<f8")
    with pytest.raises(GeBeam3MixedCommittedStateError, match="shape"):
        seal_committed_ge_beam3_mixed_state(changed)
    changed = dict(state)
    changed["committed_total_u"] = np.zeros(18, dtype="<f4")
    with pytest.raises(GeBeam3MixedCommittedStateError, match="dtype"):
        seal_committed_ge_beam3_mixed_state(changed)

    payload = json.loads(serialize_ge_beam3_mixed_state(state))
    payload["committed_total_u"]["data_hex"] = (
        payload["committed_total_u"]["data_hex"][:-2] + "aa"
    )
    raw = canonical_json_bytes(payload)
    with pytest.raises(GeBeam3MixedCommittedStateError, match="uppercase hex"):
        deserialize_ge_beam3_mixed_state(raw)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("candidate_id", "FOREIGN", "enum"),
        ("formulation_id", "GE_BEAM3_SR_P1_STRAIGHT_V1", "enum"),
        ("section_states", [None, None, None], "four nulls"),
        ("section_station_ids", ["wrong"] * 4, "station IDs"),
    ),
)
def test_foreign_formulation_and_section_state_layout_fail_closed(
    field: str, replacement: object, message: str
) -> None:
    changed = dict(_state())
    changed[field] = replacement
    with pytest.raises(GeBeam3MixedCommittedStateError, match=message):
        seal_committed_ge_beam3_mixed_state(changed)


def test_model_identity_rejects_connectivity_geometry_triad_section_and_displacement() -> None:
    state = _state(nonzero=True)
    with pytest.raises(GeBeam3MixedCommittedStateError, match="connectivity|identity"):
        _validate(state, node_ids=np.array((11, 13, 12), dtype="<i8"))
    geometry = np.array(_identity_inputs()["reference_geometry"], copy=True)
    geometry += np.array((0.0, 0.0, 1.0))
    with pytest.raises(GeBeam3MixedCommittedStateError, match="geometry|identity"):
        _validate(state, reference_geometry=geometry)
    triad = np.diag((1.0, -1.0, -1.0)).astype("<f8")
    with pytest.raises(GeBeam3MixedCommittedStateError, match="triad|orientation"):
        _validate(state, reference_triad=triad)
    stiffness = np.array(_identity_inputs()["section_stiffness"], copy=True)
    stiffness[0, 0] *= 1.01
    with pytest.raises(GeBeam3MixedCommittedStateError, match="section|identity"):
        _validate(state, section_stiffness=stiffness)
    expected = np.array(state["committed_total_u"], copy=True)
    expected[0] += 1.0
    with pytest.raises(GeBeam3MixedCommittedStateError, match="displacement"):
        _validate(state, expected_committed_total_u=expected)


def test_reference_and_so3_invariants_are_enforced_even_after_resealing() -> None:
    inputs = _identity_inputs()
    bad_geometry = np.array(inputs["reference_geometry"], copy=True)
    bad_geometry[1, 1] += 1.0e-3
    with pytest.raises(GeBeam3MixedCommittedStateError, match="midpoint"):
        initialize_ge_beam3_mixed_state(**{**inputs, "reference_geometry": bad_geometry})
    bad_triad = np.array(inputs["reference_triad"], copy=True)
    bad_triad[1, 1] = 1.01
    with pytest.raises(GeBeam3MixedCommittedStateError, match="proper rotation"):
        initialize_ge_beam3_mixed_state(**{**inputs, "reference_triad": bad_triad})

    changed = dict(_state())
    rotations = np.array(changed["committed_nodal_rotation_matrices"], copy=True)
    rotations[0, 0, 0] = 2.0
    changed["committed_nodal_rotation_matrices"] = rotations
    changed = seal_committed_ge_beam3_mixed_state(changed)
    with pytest.raises(GeBeam3MixedCommittedStateError, match="proper rotation"):
        _validate(changed)

    changed = dict(_state())
    rotations = np.array(changed["committed_nodal_rotation_matrices"], copy=True)
    angle = 0.91 * np.pi
    rotations[1] = np.array(
        ((1.0, 0.0, 0.0), (0.0, np.cos(angle), -np.sin(angle)), (0.0, np.sin(angle), np.cos(angle))),
        dtype="<f8",
    )
    changed["committed_nodal_rotation_matrices"] = rotations
    changed = seal_committed_ge_beam3_mixed_state(changed)
    with pytest.raises(GeBeam3MixedCommittedStateError, match="rotation domain"):
        _validate(changed)


def test_constitutive_moment_and_recomputed_local_state_checks() -> None:
    state = _state(nonzero=True)
    changed = dict(state)
    resultants = np.array(state["station_generalized_resultant"], copy=True)
    resultants[0, 0] += 1.0
    changed["station_generalized_resultant"] = resultants
    changed = seal_committed_ge_beam3_mixed_state(changed)
    with pytest.raises(GeBeam3MixedCommittedStateError, match="stiffness times strain"):
        _validate(changed)

    changed = dict(state)
    moments = np.array(state["committed_local_moments"], copy=True)
    moments[0, 0, 0] += 1.0
    changed["committed_local_moments"] = moments
    changed = seal_committed_ge_beam3_mixed_state(changed)
    with pytest.raises(GeBeam3MixedCommittedStateError, match="local moments"):
        _validate(changed)

    expected = {
        "committed_local_rotation_matrices": state["committed_local_rotation_matrices"],
        "committed_local_moments": state["committed_local_moments"],
        "station_generalized_strain": state["station_generalized_strain"],
        "station_generalized_resultant": state["station_generalized_resultant"],
    }
    _validate(state, expected_local_state=expected)
    expected = copy.copy(expected)
    mismatch = np.array(expected["station_generalized_strain"], copy=True)
    mismatch[0, 0] += 1.0e-4
    expected["station_generalized_strain"] = mismatch
    with pytest.raises(GeBeam3MixedCommittedStateError, match="recomputed local state"):
        _validate(state, expected_local_state=expected)


def test_serialization_rejects_unsealed_and_cross_formulation_payloads() -> None:
    state = _state()
    unsealed = dict(state)
    unsealed["state_integrity_sha256"] = "0" * 64
    with pytest.raises(GeBeam3MixedCommittedStateError, match="integrity"):
        serialize_ge_beam3_mixed_state(unsealed)
    payload = json.loads(serialize_ge_beam3_mixed_state(state))
    payload["formulation_id"] = "LEGACY_B3"
    raw = canonical_json_bytes(payload)
    with pytest.raises(GeBeam3MixedCommittedStateError, match="enum"):
        deserialize_ge_beam3_mixed_state(raw)


def _element_model(
    *, reversed_connectivity: bool, coupled: bool, with_axis: bool
) -> tuple[FEModel, GeometricallyExactBeam3D3NElement]:
    model = FEModel("ge-beam3-state-identity")
    model.add_material("mat", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.5, 0.0, 0.0)
    model.add_node(3, 1.0, 0.0, 0.0)
    if coupled:
        stiffness = np.asarray(_identity_inputs()["section_stiffness"], dtype=float)
        mass = np.asarray(_identity_inputs()["section_mass"], dtype=float)
        # Couple components with opposite reversal parities so the test proves
        # that the effective, polarity-transformed operators enter the hash.
        stiffness[0, 4] = stiffness[4, 0] = 5.0
        mass[0, 4] = mass[4, 0] = 0.1
    else:
        stiffness = np.diag((1000.0, 500.0, 600.0, 100.0, 120.0, 140.0))
        mass = np.diag((10.0, 10.0, 10.0, 0.5, 0.6, 0.7))
    section = GeneralizedBeamSection(
        stiffness=stiffness,
        mass_matrix=mass,
        name="IDENTITY_PARITY",
    )
    element = GeometricallyExactBeam3D3NElement(
        31,
        (3, 2, 1) if reversed_connectivity else (1, 2, 3),
        "mat",
        section=section,
        reference_orientation=(0.0, 1.0, 0.0),
        reference_axis_direction=(1.0, 0.0, 0.0) if with_axis else None,
    )
    return model, element


def test_model_bound_identity_matches_reversed_coupled_element_authority() -> None:
    model, element = _element_model(
        reversed_connectivity=True,
        coupled=True,
        with_axis=True,
    )
    authority = element._state_authority(model.mesh)
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("mat"),
        1,
    )
    identity = model_bound_identity(element, model.mesh)

    np.testing.assert_array_equal(
        identity["section_stiffness"], authority["section_stiffness"]
    )
    np.testing.assert_array_equal(identity["section_mass"], authority["section_mass"])
    assert identity["section_descriptor_sha256"] == state["section_descriptor_sha256"]
    assert identity["element_identity_sha256"] == state["element_identity_sha256"]

    raw_stiffness = element.generalized_section.generalized_stiffness_matrix()
    raw_mass = element.generalized_section.generalized_mass_matrix_per_length()
    assert not np.array_equal(identity["section_stiffness"], raw_stiffness)
    assert not np.array_equal(identity["section_mass"], raw_mass)


def test_model_bound_identity_uses_chord_axis_when_symmetric_element_omits_axis() -> None:
    model, element = _element_model(
        reversed_connectivity=False,
        coupled=False,
        with_axis=False,
    )
    assert element.reference_axis_direction is None
    authority = element._state_authority(model.mesh)
    identity = model_bound_identity(element, model.mesh)
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material("mat"),
        1,
    )

    np.testing.assert_array_equal(
        identity["element_identity_payload"]["reference_axis_direction"],
        authority["reference_axis_direction"],
    )
    np.testing.assert_array_equal(
        identity["element_identity_payload"]["reference_axis_direction"],
        identity["reference_triad"][:, 0],
    )
    assert identity["section_descriptor_sha256"] == state["section_descriptor_sha256"]
    assert identity["element_identity_sha256"] == state["element_identity_sha256"]
