from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver._ge_beam3_mixed_ad import rotation_exponential
from anyfesolver.fe_core import FEModel
from anyfesolver.ge_beam3_mixed_element import (
    GE_BEAM3_MIXED_CANDIDATE_ID,
    GeBeam3MixedGeometryError,
    GeBeam3MixedStateError,
    GeometricallyExactBeam3D3NElement,
)


_CASES_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "reference_cases"
    / "ge_beam3_mixed_p2_cases.json"
)
_CASES = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
_CORRECTION = json.loads(
    _CASES_PATH.with_name(
        "ge_beam3_mixed_p2_recovery_correction.json"
    ).read_text(encoding="utf-8")
)
_DATA = _CASES["cases"]
_GEOMETRY = {item["geometry_id"]: item for item in _DATA["geometry"]}
_SECTIONS = {item["section_id"]: item for item in _DATA["sections"]}


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _rational(value: dict[str, int]) -> float:
    return float(value["numerator"] / value["denominator"])


def _recovery_case(case_id: str) -> dict[str, Any]:
    if case_id == _CORRECTION["replacement"]["case_id"]:
        return _CORRECTION["replacement"]
    return next(
        item for item in _DATA["recovery_states"] if item["case_id"] == case_id
    )


def _section(section_id: str, *, with_mass: bool = True) -> GeneralizedBeamSection:
    authority = _SECTIONS[section_id]
    return GeneralizedBeamSection(
        _array(authority["stiffness_matrix"]),
        mass_matrix=(
            _array(authority["mass_matrix_per_reference_length"])
            if with_mass
            else None
        ),
        name=section_id,
    )


def _element(
    geometry_id: str,
    section_id: str = "STEEL_DIAGONAL",
    *,
    with_mass: bool = True,
) -> tuple[FEModel, GeometricallyExactBeam3D3NElement]:
    authority = _GEOMETRY[geometry_id]
    model = FEModel(f"ge-beam3-p2-{geometry_id.lower()}")
    model.add_material("mat", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(authority["nodes"], start=1):
        model.add_node(node_id, *map(float, coordinates))
    element = GeometricallyExactBeam3D3NElement(
        1,
        authority["connectivity"],
        "mat",
        section=_section(section_id, with_mass=with_mass),
        reference_orientation=tuple(map(float, authority["reference_orientation"])),
        reference_axis_direction=tuple(
            map(float, authority["reference_axis_direction"])
        ),
    )
    return model, element


def _descriptor(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "classification",
            "force_per_reference_length_at_nodes",
            "couple_per_reference_length_at_nodes",
        )
        if key in case
    }


def _reference_mass(
    element: GeometricallyExactBeam3D3NElement,
    mesh: Any,
    section_mass: np.ndarray,
) -> np.ndarray:
    frame = element.reference_triad(mesh)
    rotation = np.zeros((6, 6))
    rotation[:3, :3] = frame
    rotation[3:, 3:] = frame
    spatial = rotation @ section_mass @ rotation.T
    coordinates = element.get_node_coordinates(mesh)
    length = float(np.linalg.norm(coordinates[2] - coordinates[0]))
    cell_length = 0.5 * length
    coefficients = np.array(((1.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 1.0 / 3.0)))
    made = np.zeros((18, 18))
    for left, right in ((0, 1), (1, 2)):
        nodes = (left, right)
        for local_i, node_i in enumerate(nodes):
            for local_j, node_j in enumerate(nodes):
                rows = slice(6 * node_i, 6 * node_i + 6)
                columns = slice(6 * node_j, 6 * node_j + 6)
                made[rows, columns] += (
                    cell_length * coefficients[local_i, local_j] * spatial
                )
    return made


@pytest.mark.parametrize(
    "case_id",
    ["UNIFORM_LINE_SPATIAL_DEAD", "LINEAR_LINE_MATERIAL_DEAD_SKEW"],
)
def test_reference_line_load_matches_frozen_work_and_resultants(case_id: str) -> None:
    case = next(item for item in _DATA["loads"] if item["case_id"] == case_id)
    model, element = _element(case["geometry_id"])
    made = np.asarray(
        element.compute_reference_line_load(model.mesh, _descriptor(case)), dtype=float
    )
    expected = _array(case["expected_element_vector"])
    np.testing.assert_allclose(made, expected, rtol=2.0e-14, atol=2.0e-14)

    nodal = made.reshape(3, 6)
    coordinates = element.get_node_coordinates(model.mesh)
    force = np.sum(nodal[:, :3], axis=0)
    moment = np.sum(nodal[:, 3:], axis=0)
    moment += np.sum(np.cross(coordinates, nodal[:, :3]), axis=0)
    work = float(made @ _array(case["virtual_nodal_coordinates"]).reshape(18))
    np.testing.assert_allclose(force, _array(case["expected_resultant_force"]), rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(moment, _array(case["expected_resultant_moment_about_global_origin"]), rtol=2.0e-14, atol=2.0e-13)
    assert work == pytest.approx(float(case["expected_virtual_work"]), rel=2.0e-14, abs=2.0e-14)


@pytest.mark.parametrize(
    "case_id",
    ["CONSERVATIVE_FOLLOWER_REJECT", "NONCONSERVATIVE_FOLLOWER_REJECT"],
)
def test_follower_line_loads_fail_closed_before_candidate_evaluation(
    case_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = next(item for item in _DATA["loads"] if item["case_id"] == case_id)
    model, element = _element(case["geometry_id"])

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("element mechanics was entered for an unsupported load")

    monkeypatch.setattr(element, "evaluate_candidate", forbidden)
    with pytest.raises(
        GeBeam3MixedStateError,
        match="follower|unsupported|spatial-dead|material-dead",
    ):
        element.compute_reference_line_load(
            model.mesh,
            {
                "classification": case["classification"],
                "force_per_reference_length_at_nodes": (0.0, 1.0, 0.0),
                "couple_per_reference_length_at_nodes": (0.0, 0.0, 0.0),
            },
        )


@pytest.mark.parametrize(
    "case_id",
    ["REFERENCE_MASS_DIAGONAL_UNIT", "REFERENCE_MASS_COUPLED_SKEW"],
)
def test_reference_mass_is_full_coupled_consistent_and_has_frozen_energy(
    case_id: str,
) -> None:
    case = next(item for item in _DATA["mass"] if item["case_id"] == case_id)
    model, element = _element(case["geometry_id"], case["section_id"])
    made = np.asarray(element.compute_mass_matrix(model.mesh, None), dtype=float)
    section_mass = _array(
        _SECTIONS[case["section_id"]]["mass_matrix_per_reference_length"]
    )
    expected = _reference_mass(element, model.mesh, section_mass)
    np.testing.assert_allclose(made, expected, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(made, made.T, rtol=0.0, atol=2.0e-14)
    assert np.linalg.eigvalsh(made)[0] > 0.0

    frame = element.reference_triad(model.mesh)
    rotation = np.zeros((6, 6))
    rotation[:3, :3] = frame
    rotation[3:, 3:] = frame
    velocity = rotation @ _array(case["uniform_local_generalized_velocity"])
    nodal_velocity = np.tile(velocity, 3)
    kinetic = 0.5 * float(nodal_velocity @ made @ nodal_velocity)
    assert kinetic == pytest.approx(float(case["expected_kinetic_energy"]), rel=2.0e-14)

    length = float(np.linalg.norm(element.get_node_coordinates(model.mesh)[2] - element.get_node_coordinates(model.mesh)[0]))
    expected_mass = float(case["expected_total_translational_mass"])
    assert length * section_mass[0, 0] == pytest.approx(expected_mass, rel=2.0e-14)


def test_missing_generalized_mass_authority_fails_closed() -> None:
    model, element = _element("UNIT_X", with_mass=False)
    with pytest.raises((GeBeam3MixedStateError, ValueError), match="mass|inertia"):
        element.compute_mass_matrix(model.mesh, None)


def _diagonal_stiffness_coupled_mass_section() -> GeneralizedBeamSection:
    return GeneralizedBeamSection(
        np.diag((1000.0, 500.0, 600.0, 100.0, 120.0, 140.0)),
        mass_matrix=_array(
            _SECTIONS["COUPLED_PHYSICAL"]["mass_matrix_per_reference_length"]
        ),
        name="DIAGONAL_STIFFNESS_COUPLED_MASS",
    )


def test_coupled_mass_requires_physical_axis_even_with_diagonal_stiffness() -> None:
    with pytest.raises(
        GeBeam3MixedGeometryError,
        match="reversal-sensitive.*mass.*reference_axis_direction",
    ):
        GeometricallyExactBeam3D3NElement(
            1,
            (1, 2, 3),
            "mat",
            section=_diagonal_stiffness_coupled_mass_section(),
            reference_orientation=(0.0, 1.0, 0.0),
        )


def test_p2_constructor_rejects_external_section_protocols_before_callbacks() -> None:
    class _ExternalSectionTrap:
        name = "external-trap"

        def __init__(self, *, requires_history: bool | None) -> None:
            if requires_history is not None:
                self.requires_history = requires_history
            self.calls: list[str] = []

        def generalized_stiffness_matrix(self) -> np.ndarray:
            self.calls.append("stiffness")
            raise AssertionError("external stiffness callback must not run in P2")

        def generalized_mass_matrix_per_length(self) -> np.ndarray:
            self.calls.append("mass")
            raise AssertionError("external mass callback must not run in P2")

    history_bearing = _ExternalSectionTrap(requires_history=True)
    with pytest.raises(GeBeam3MixedStateError, match="history|external|P2"):
        GeometricallyExactBeam3D3NElement(
            1,
            (1, 2, 3),
            "mat",
            section=history_bearing,
            reference_orientation=(0.0, 1.0, 0.0),
            reference_axis_direction=(1.0, 0.0, 0.0),
        )
    assert history_bearing.calls == []

    unmarked_external = _ExternalSectionTrap(requires_history=None)
    with pytest.raises(GeBeam3MixedStateError, match="external|P2|stateless"):
        GeometricallyExactBeam3D3NElement(
            2,
            (1, 2, 3),
            "mat",
            section=unmarked_external,
            reference_orientation=(0.0, 1.0, 0.0),
            reference_axis_direction=(1.0, 0.0, 0.0),
        )
    assert unmarked_external.calls == []

    inline_external = _ExternalSectionTrap(requires_history=None)
    with pytest.raises(GeBeam3MixedStateError, match="external|P2|stateless"):
        GeometricallyExactBeam3D3NElement(
            3,
            (1, 2, 3),
            "mat",
            cross_section={"generalized_stiffness": inline_external},
            reference_orientation=(0.0, 1.0, 0.0),
            reference_axis_direction=(1.0, 0.0, 0.0),
        )
    assert inline_external.calls == []

    stiffness = tuple(
        tuple(float(value) for value in row)
        for row in _SECTIONS["STEEL_DIAGONAL"]["stiffness_matrix"]
    )
    mass = tuple(
        tuple(float(value) for value in row)
        for row in _SECTIONS["STEEL_DIAGONAL"]["mass_matrix_per_reference_length"]
    )
    accepted = (
        GeneralizedBeamSection(stiffness, mass_matrix=mass, name="exact-native"),
        MappingProxyType(
            {"mass_matrix": mass, "name": "exact-mapping", "stiffness": stiffness}
        ),
    )
    for element_id, section in enumerate(accepted, start=4):
        made = GeometricallyExactBeam3D3NElement(
            element_id,
            (1, 2, 3),
            "mat",
            section=section,
            reference_orientation=(0.0, 1.0, 0.0),
            reference_axis_direction=(1.0, 0.0, 0.0),
        )
        assert isinstance(made.generalized_section, GeneralizedBeamSection)


def test_coupled_consistent_mass_is_covariant_under_connectivity_reversal() -> None:
    model = FEModel("ge-beam3-p2-coupled-mass-reversal")
    model.add_material("mat", 1.0, 0.25, density=1.0)
    for node_id, x in enumerate((0.0, 0.5, 1.0), start=1):
        model.add_node(node_id, x, 0.0, 0.0)
    section = _diagonal_stiffness_coupled_mass_section()
    forward = GeometricallyExactBeam3D3NElement(
        1,
        (1, 2, 3),
        "mat",
        section=section,
        reference_orientation=(0.0, 1.0, 0.0),
        reference_axis_direction=(1.0, 0.0, 0.0),
    )
    reverse = GeometricallyExactBeam3D3NElement(
        2,
        (3, 2, 1),
        "mat",
        section=section,
        reference_orientation=(0.0, 1.0, 0.0),
        reference_axis_direction=(1.0, 0.0, 0.0),
    )

    local_mass = _array(
        _SECTIONS["COUPLED_PHYSICAL"]["mass_matrix_per_reference_length"]
    )
    reversal = GeometricallyExactBeam3D3NElement.REVERSAL_STRAIN_MAP
    np.testing.assert_allclose(
        forward._section_mass(model.mesh),
        local_mass,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        reverse._section_mass(model.mesh),
        reversal @ local_mass @ reversal,
        rtol=0.0,
        atol=2.0e-14,
    )

    forward_mass = forward.compute_mass_matrix(model.mesh, None)
    reverse_mass = reverse.compute_mass_matrix(model.mesh, None)
    permutation = np.zeros((18, 18))
    for reverse_node, forward_node in enumerate((2, 1, 0)):
        permutation[
            6 * reverse_node : 6 * reverse_node + 6,
            6 * forward_node : 6 * forward_node + 6,
        ] = np.eye(6)
    np.testing.assert_allclose(
        permutation.T @ reverse_mass @ permutation,
        forward_mass,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    velocity = np.linspace(-0.7, 0.8, 18)
    assert float(velocity @ forward_mass @ velocity) == pytest.approx(
        float((permutation @ velocity) @ reverse_mass @ (permutation @ velocity)),
        rel=2.0e-14,
        abs=2.0e-14,
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "RECOVERY_ZERO_NATIVE_STATE",
        "RECOVERY_UNIFORM_AXIAL_NATIVE_STATE",
        "RECOVERY_MANUFACTURED_UNIFORM_SHEAR_Y_NATIVE_STATE",
        "RECOVERY_TORSION_SIDED_NATIVE_STATE",
    ],
)
def test_native_recovery_uses_absolute_rotations_and_keeps_sided_midpoint(
    case_id: str,
) -> None:
    case = _recovery_case(case_id)
    section_id = case.get("section_id", case.get("section", {}).get("section_id"))
    model, element = _element(case["geometry_id"], section_id)
    displacement = np.zeros((3, 6))
    displacement[:, :3] = _array(case["nodal_translations"])
    if "vertex_rotation" in case:
        rotation = case["vertex_rotation"]
        rotation_vector = _array(rotation["axis"]) * _rational(
            rotation["radians"]
        )
        rotations = np.repeat(
            rotation_exponential(rotation_vector)[None, :, :], 3, axis=0
        )
    else:
        rotations = _array(case["absolute_rotation_matrices"])
    fields = element.recover_native_fields(
        model.mesh,
        displacement.reshape(18),
        rotation_matrices=rotations,
    )
    assert tuple(fields["station_order"]) == tuple(case["station_order"])
    strain = np.asarray(fields["local_generalized_strain"], dtype=float)
    resultant = np.asarray(fields["local_generalized_resultant"], dtype=float)
    assert strain.shape == resultant.shape == (4, 6)
    assert np.all(np.isfinite(strain)) and np.all(np.isfinite(resultant))
    assert fields["candidate_id"] == GE_BEAM3_MIXED_CANDIDATE_ID
    assert not {"fibre_stress", "von_mises", "equivalent_stress"}.intersection(fields)
    assert fields["fibre_stress_available"] is False

    if "expected" in case and "station_signs" in case:
        expected = case["expected"]
        expected_strain = np.zeros((4, 6))
        expected_resultant = np.zeros((4, 6))
        expected_strain[:, 1] = _rational(expected["force_strain_gamma_y"])
        expected_strain[:, 5] = _rational(
            expected["moment_curvature_z_abs"]
        ) * _array(case["station_signs"])
        expected_resultant[:, 1] = _rational(
            expected["shear_resultant_y"]
        )
        expected_resultant[:, 5] = _rational(
            expected["bending_moment_z_abs"]
        ) * _array(case["station_signs"])
        strain_error = np.linalg.norm(strain - expected_strain, ord=np.inf) / max(
            1.0,
            np.linalg.norm(strain, ord=np.inf),
            np.linalg.norm(expected_strain, ord=np.inf),
        )
        resultant_error = np.linalg.norm(
            resultant - expected_resultant, ord=np.inf
        ) / max(
            1.0,
            np.linalg.norm(resultant, ord=np.inf),
            np.linalg.norm(expected_resultant, ord=np.inf),
        )
        assert strain_error <= float(_CORRECTION["tolerance"]["value"])
        assert resultant_error <= float(_CORRECTION["tolerance"]["value"])
        np.testing.assert_allclose(
            fields["station_frames"],
            np.repeat(np.eye(3)[None, :, :], 4, axis=0),
            rtol=0.0,
            atol=2.0e-15,
        )
    elif "expected_station_strain" in case:
        np.testing.assert_allclose(strain, _array(case["expected_station_strain"]), rtol=1.0e-11, atol=1.0e-13)
        np.testing.assert_allclose(resultant, _array(case["expected_station_resultant"]), rtol=1.0e-11, atol=1.0e-10)
    else:
        expected = float(case["expected_torsional_curvature"])
        np.testing.assert_allclose(strain[:, 3], expected, rtol=1.0e-11, atol=1.0e-13)
        assert tuple(fields["station_order"][1:3]) == (
            "XI_ZERO_LEFT",
            "XI_ZERO_RIGHT",
        )


def test_rotation_coordinates_are_not_accepted_as_recovery_state() -> None:
    model, element = _element("UNIT_X")
    displacement = np.zeros(18)
    displacement[3] = 1.0e-6
    with pytest.raises(GeBeam3MixedStateError, match="increments, not accumulated"):
        element.recover_native_fields(
            model.mesh,
            displacement,
            rotation_matrices=np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        )
