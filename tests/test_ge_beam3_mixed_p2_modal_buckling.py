from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.linalg import eig, eigh

from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.fe_core import FEModel
from anyfesolver.ge_beam3_mixed_element import (
    GeBeam3MixedStateError,
    GeometricallyExactBeam3D3NElement,
)


_CASES = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "reference_cases"
        / "ge_beam3_mixed_p2_cases.json"
    ).read_text(encoding="utf-8")
)
_DATA = _CASES["cases"]
_SECTIONS = {item["section_id"]: item for item in _DATA["sections"]}
_DOF = {name: index for index, name in enumerate(("ux", "uy", "uz", "rx", "ry", "rz"))}


def _section() -> GeneralizedBeamSection:
    authority = _SECTIONS["STEEL_DIAGONAL"]
    return GeneralizedBeamSection(
        np.asarray(authority["stiffness_matrix"], dtype=float),
        mass_matrix=np.asarray(authority["mass_matrix_per_reference_length"], dtype=float),
        name="STEEL_DIAGONAL",
    )


def _chain(macro_count: int) -> tuple[FEModel, list[GeometricallyExactBeam3D3NElement]]:
    length = 3.0
    model = FEModel(f"ge-beam3-p2-chain-{macro_count}")
    model.add_material("mat", 210.0e9, 0.3, density=7850.0)
    for index in range(2 * macro_count + 1):
        model.add_node(index + 1, index * length / (2 * macro_count), 0.0, 0.0)
    elements = []
    for index in range(macro_count):
        element = GeometricallyExactBeam3D3NElement(
            index + 1,
            (2 * index + 1, 2 * index + 2, 2 * index + 3),
            "mat",
            section=_section(),
            reference_orientation=(0.0, 1.0, 0.0),
            reference_axis_direction=(1.0, 0.0, 0.0),
        )
        elements.append(element)
    return model, elements


def _scatter(
    model: FEModel,
    elements: list[GeometricallyExactBeam3D3NElement],
    matrix_name: str,
    **kwargs: Any,
) -> np.ndarray:
    size = len(model.mesh.nodes) * 6
    made = np.zeros((size, size))
    for element in elements:
        if matrix_name == "stiffness":
            local = element.compute_candidate_stiffness_matrix(model.mesh)
        elif matrix_name == "mass":
            local = element.compute_mass_matrix(model.mesh, None)
        elif matrix_name == "geometric":
            local = element.compute_reference_geometric_stiffness(model.mesh, **kwargs)
        else:  # pragma: no cover - test helper contract
            raise AssertionError(matrix_name)
        indices = np.asarray(
            [6 * (node - 1) + dof for node in element.node_ids for dof in range(6)]
        )
        made[np.ix_(indices, indices)] += local
    return made


def _free_indices(node_count: int, fixed: set[int]) -> np.ndarray:
    return np.asarray([index for index in range(6 * node_count) if index not in fixed])


def _family_mass(
    vector: np.ndarray, mass: np.ndarray, node_count: int
) -> dict[str, float]:
    blocks = {
        "AXIAL": ("ux",),
        "BENDING_STRONG_Y": ("uz", "ry"),
        "BENDING_WEAK_Z": ("uy", "rz"),
        "TORSION": ("rx",),
    }
    scores: dict[str, float] = {}
    for family, names in blocks.items():
        indices = [6 * node + _DOF[name] for node in range(node_count) for name in names]
        part = np.zeros_like(vector)
        part[indices] = vector[indices]
        scores[family] = float(part @ mass @ part)
    return scores


def _mass_normalize(vector: np.ndarray, mass: np.ndarray) -> np.ndarray:
    denominator = float(np.sqrt(vector @ mass @ vector))
    assert denominator > 0.0 and np.isfinite(denominator)
    made = np.asarray(vector / denominator)
    anchor = next(value for value in made if abs(value) > 1.0e-12)
    return -made if anchor < 0.0 else made


def _euclidean_normalize(vector: np.ndarray) -> np.ndarray:
    made = np.asarray(vector / np.linalg.norm(vector))
    anchor = next(value for value in made if abs(value) > 1.0e-12)
    return -made if anchor < 0.0 else made


def test_reference_modal_chain_has_six_free_modes_and_frozen_family_limits() -> None:
    reference = next(item for item in _DATA["modal"] if item["case_id"] == "CANTILEVER_REFERENCE_LINEAR")
    finest: dict[str, float] = {}
    for macro_count in reference["macro_element_counts"]:
        model, elements = _chain(macro_count)
        stiffness = _scatter(model, elements, "stiffness")
        mass = _scatter(model, elements, "mass")
        node_count = 2 * macro_count + 1
        free_values = eigh(stiffness, mass, eigvals_only=True, check_finite=True)
        spectral_scale = max(float(np.max(np.abs(free_values))), 1.0)
        assert np.count_nonzero(np.abs(free_values) <= 1.0e-9 * spectral_scale) == 6

        fixed = set(range(6))
        free = _free_indices(node_count, fixed)
        values, vectors = eigh(stiffness[np.ix_(free, free)], mass[np.ix_(free, free)], check_finite=True)
        candidates: dict[str, tuple[float, float]] = {}
        for value, reduced in zip(values, vectors.T):
            if value <= 1.0e-12 * max(float(np.max(np.abs(values))), 1.0):
                continue
            full = np.zeros(node_count * 6)
            full[free] = reduced
            full = _mass_normalize(full, mass)
            scores = _family_mass(full, mass, node_count)
            family = max(sorted(scores), key=lambda name: scores[name])
            frequency = float(np.sqrt(value) / (2.0 * np.pi))
            previous = candidates.get(family)
            if previous is None or frequency < previous[0]:
                candidates[family] = (frequency, scores[family])
        assert set(candidates) == set(reference["expected_frequencies_hz"])
        assert all(score > 0.99 for _frequency, score in candidates.values())
        if macro_count == reference["macro_element_counts"][-1]:
            finest = {name: value[0] for name, value in candidates.items()}

    for family, expected in reference["expected_frequencies_hz"].items():
        assert finest[family] == pytest.approx(float(expected), rel=float(_CASES["tolerances"]["modal_reference_relative"]))


def test_reference_geometric_stiffness_sign_scaling_and_argument_guards() -> None:
    model, elements = _chain(2)
    element = elements[0]
    positive = element.compute_reference_geometric_stiffness(model.mesh, axial_compression=2.5)
    force_form = element.compute_reference_geometric_stiffness(model.mesh, axial_force=-2.5)
    tension = element.compute_reference_geometric_stiffness(model.mesh, axial_force=2.5)
    np.testing.assert_allclose(positive, force_form, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(tension, -positive, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(positive, positive.T, rtol=0.0, atol=1.0e-14)

    coordinates = element.get_node_coordinates(model.mesh)
    cell_length = 0.5 * float(np.linalg.norm(coordinates[2] - coordinates[0]))
    expected = np.zeros((18, 18))
    for left, right in ((0, 1), (1, 2)):
        for local_dof in (_DOF["uy"], _DOF["uz"]):
            indices = [6 * left + local_dof, 6 * right + local_dof]
            expected[np.ix_(indices, indices)] += 2.5 / cell_length * np.array(((1.0, -1.0), (-1.0, 1.0)))
    np.testing.assert_allclose(positive, expected, rtol=0.0, atol=1.0e-13)

    invalid = (
        {},
        {"axial_compression": 1.0, "axial_force": -1.0},
        {"axial_compression": float("nan")},
        {"axial_force": float("inf")},
    )
    for arguments in invalid:
        with pytest.raises((GeBeam3MixedStateError, ValueError), match="axial|finite|exactly one"):
            element.compute_reference_geometric_stiffness(model.mesh, **arguments)


@pytest.mark.parametrize("case_id", ["EULER_PINNED_WEAK_Z", "EULER_PINNED_STRONG_Y"])
def test_euler_buckling_chain_uses_compression_positive_generalized_problem(case_id: str) -> None:
    case = next(item for item in _DATA["buckling"] if item["case_id"] == case_id)
    finest = None
    for macro_count in case["macro_element_counts"]:
        model, elements = _chain(macro_count)
        stiffness = _scatter(model, elements, "stiffness")
        geometric = _scatter(model, elements, "geometric", axial_compression=1.0)
        node_count = 2 * macro_count + 1
        fixed: set[int] = set()
        for node in range(node_count):
            fixed.update(6 * node + _DOF[name] for name in case["support"]["fixed_all_nodes"])
        fixed.update(_DOF[name] for name in case["support"]["end_1"])
        fixed.update(6 * (node_count - 1) + _DOF[name] for name in case["support"]["end_2"])
        free = _free_indices(node_count, fixed)
        values, vectors = eig(
            stiffness[np.ix_(free, free)],
            geometric[np.ix_(free, free)],
            check_finite=True,
        )
        admitted = [
            (float(value.real), vector.real)
            for value, vector in zip(values, vectors.T)
            if np.isfinite(value.real)
            and abs(value.imag) <= 1.0e-9 * max(abs(value.real), 1.0)
            and value.real > 0.0
        ]
        critical, vector = min(admitted, key=lambda item: item[0])
        normalized = _euclidean_normalize(vector)
        assert np.linalg.norm(normalized) == pytest.approx(1.0, rel=0.0, abs=2.0e-14)
        assert next(value for value in normalized if abs(value) > 1.0e-12) > 0.0
        if macro_count == case["macro_element_counts"][-1]:
            finest = critical
    assert finest == pytest.approx(
        float(case["expected_critical_load"]),
        rel=float(_CASES["tolerances"]["buckling_reference_relative"]),
    )


def test_matching_and_normalization_rules_are_deterministic_for_repeated_cluster() -> None:
    mass = np.diag((2.0, 3.0, 5.0, 7.0))
    reference = np.eye(4)
    numerical = np.array(
        (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.6, 0.8, 0.0),
            (0.0, -0.8, 0.6, 0.0),
            (0.0, 0.0, 0.0, -1.0),
        )
    ).T
    normalized = np.column_stack([_mass_normalize(numerical[:, i], mass) for i in range(4)])
    assert normalized[0, 0] > 0.0 and normalized[3, 3] > 0.0

    ref_cluster = reference[:, 1:3]
    num_cluster = normalized[:, 1:3]
    ref_cluster = ref_cluster @ np.linalg.inv(np.linalg.cholesky(ref_cluster.T @ mass @ ref_cluster)).T
    num_cluster = num_cluster @ np.linalg.inv(np.linalg.cholesky(num_cluster.T @ mass @ num_cluster)).T
    overlap = ref_cluster.T @ mass @ num_cluster
    singular = np.linalg.svd(overlap, compute_uv=False)
    np.testing.assert_allclose(singular, np.ones(2), rtol=0.0, atol=2.0e-14)


def test_private_analysis_route_guard_accepts_only_preregistered_p2_routes() -> None:
    _model, elements = _chain(1)
    element = elements[0]
    for route in (
        "REFERENCE_STATIC",
        "NATIVE_NONLINEAR_STATIC",
        "REFERENCE_MODAL",
        "REFERENCE_BUCKLING",
        "NATIVE_RECOVERY",
    ):
        assert element.require_private_analysis_route(route) is None

    for route in (
        "CURRENT_STATE_MODAL",
        "CURRENT_STATE_BUCKLING",
        "LINEAR_TRANSIENT",
        "FINITE_ROTATION_TRANSIENT",
        "GYROSCOPIC_TERMS",
        "CONSERVATIVE_FOLLOWER",
        "NONCONSERVATIVE_FOLLOWER",
        "UNKNOWN_ROUTE",
        "",
    ):
        with pytest.raises(GeBeam3MixedStateError, match="outside|authority"):
            element.require_private_analysis_route(route)
