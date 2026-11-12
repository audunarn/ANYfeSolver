from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from anyfesolver.e4_pl_element import (
    STATIONARY_SOLVE_POLICY_ID,
    QualifiedE4PLShellElement,
    _coefficients,
    _solve_stationary_system,
    _stationary_blocks,
    _symmetric_ruiz_congruence,
    equation7_frame,
)
from anyfesolver.fe_core import FEMesh, Material


GEOMETRIES = (
    ("Q0_SQUARE", ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))),
    ("Q1_AFFINE_SKEW", ((0, 0, 0), (3, 1, 0), (5, 4, 0), (2, 3, 0))),
    ("Q2_TRAPEZOID", ((0, 0, 0), (4, 0, 0), (3, 2, 0), (1, 2, 0))),
    ("Q3_TAPERED_SKEW", ((0, 0, 0), (5, 1, 0), (4, 4, 0), (1, 3, 0))),
    (
        "Q4_HOSTILE_ASYMMETRIC_1",
        ((0, 0, 0), (2, 0, 0), (Fraction(5, 2), 1, 0), (Fraction(1, 2), 1, 0)),
    ),
    (
        "Q5_HOSTILE_ASYMMETRIC_2",
        ((0, 0, 0), (2, 0, 0), (Fraction(3, 2), 1, 0), (0, 1, 0)),
    ),
)


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for node_id, point in enumerate(nodes, start=1):
        mesh.add_node(node_id, *point)
    return mesh


def _stationary_case(nodes: np.ndarray, thickness: float):
    material = Material("steel", 210.0e9, 0.3, density=7850.0)
    element = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=thickness,
    )
    frame, local, _warpage = equation7_frame(nodes)
    constitutive = element._constitutive_and_drill_stiffness(material, frame)[0]
    stationary, coupling, _gram = _stationary_blocks(
        local,
        _coefficients(local),
        constitutive,
    )
    return element, material, stationary, coupling


@pytest.mark.parametrize("coordinate_scale", (1.0e-6, 1.0, 1.0e6))
@pytest.mark.parametrize("_geometry_id,nodes", GEOMETRIES)
def test_registered_ultrathin_stationary_systems_are_ruiz_balanced(
    _geometry_id: str,
    nodes,
    coordinate_scale: float,
) -> None:
    coordinates = coordinate_scale * np.asarray(nodes, dtype=float)
    length = max(
        float(np.linalg.norm(coordinates[j] - coordinates[i]))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    element, material, stationary, coupling = _stationary_case(
        coordinates,
        length * 1.0e-6,
    )

    equilibrated, _scaling, equilibration = _symmetric_ruiz_congruence(
        stationary
    )
    solution, diagnostics = _solve_stationary_system(stationary, coupling)
    condition = float(np.linalg.cond(equilibrated))
    residual = stationary @ solution - coupling.T
    denominator = (
        float(np.linalg.norm(stationary, ord=np.inf))
        * float(np.linalg.norm(solution, ord=np.inf))
        + float(np.linalg.norm(coupling.T, ord=np.inf))
    )

    assert condition < 256.0
    assert equilibration["iterations"] == 8
    assert equilibration["row_norm_ratio"] <= 2.0
    assert equilibration["row_norm_ratio_limit"] == 2.0
    assert diagnostics["id"] == STATIONARY_SOLVE_POLICY_ID
    assert diagnostics["relative_backward_error"] == pytest.approx(
        float(np.linalg.norm(residual, ord=np.inf)) / denominator,
        rel=0.0,
        abs=0.0,
    )
    assert diagnostics["relative_backward_error"] <= 2.0e-14

    mesh = _mesh(coordinates)
    components = element.compute_stiffness_components(mesh, material)
    assert np.all(np.isfinite(components["total"]))
    assert components["stationary_solve_diagnostics"]["id"] == (
        STATIONARY_SOLVE_POLICY_ID
    )
    recovered = element.compute_stresses(
        mesh,
        np.zeros(24, dtype=float),
        material,
    )
    np.testing.assert_array_equal(
        recovered["membrane_resultants"],
        np.zeros((4, 3), dtype=float),
    )


def test_stationary_ruiz_solve_is_unit_robust_and_ordinary_case_equivalent() -> None:
    base = np.asarray(GEOMETRIES[1][1], dtype=float)
    _element, _material, ordinary, ordinary_coupling = _stationary_case(base, 0.2)
    ordinary_raw = np.linalg.solve(ordinary, ordinary_coupling.T)
    ordinary_scaled, _diagnostics = _solve_stationary_system(
        ordinary,
        ordinary_coupling,
    )
    np.testing.assert_allclose(
        ordinary_scaled,
        ordinary_raw,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    scaled_nodes = 1.0e6 * base
    element, material, stationary, coupling = _stationary_case(scaled_nodes, 1.0)
    equilibrated, _scaling, diagnostics = _symmetric_ruiz_congruence(stationary)
    assert float(np.linalg.cond(stationary)) > 1.0e30
    assert float(np.linalg.cond(equilibrated)) < 256.0
    assert diagnostics["row_norm_ratio"] <= 2.0
    solution, solve_diagnostics = _solve_stationary_system(stationary, coupling)
    assert np.all(np.isfinite(solution))
    assert solve_diagnostics["relative_backward_error"] <= 2.0e-14

    components = element.compute_stiffness_components(_mesh(scaled_nodes), material)
    assert np.all(np.isfinite(components["total"]))
    assert components["stationary_solve_diagnostics"]["id"] == (
        STATIONARY_SOLVE_POLICY_ID
    )


def test_stationary_ruiz_solve_rejects_singular_or_nonfinite_inputs() -> None:
    coupling = np.zeros((24, 35), dtype=float)
    with pytest.raises(ValueError, match="singular"):
        _solve_stationary_system(np.zeros((35, 35), dtype=float), coupling)
    invalid = np.eye(35, dtype=float)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _solve_stationary_system(invalid, coupling)

    tiny_asymmetric = 1.0e-30 * np.eye(35, dtype=float)
    tiny_asymmetric[0, 1] = 1.0e-32
    with pytest.raises(ValueError, match="not symmetric"):
        _symmetric_ruiz_congruence(tiny_asymmetric)
