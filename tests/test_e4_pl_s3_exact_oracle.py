from __future__ import annotations

import ast
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3_element
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.shell_sections import GeneralizedShellSection


ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "docs/reference_cases/e4_pl_s3_exact_oracle.py"


def _load_oracle():
    spec = importlib.util.spec_from_file_location("e4_pl_s3_exact_fraction_oracle", ORACLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = _load_oracle()
F = Fraction


def _isotropic_block(diagonal: int, coupling: int) -> list[list[Fraction]]:
    return [
        [F(diagonal), F(coupling), F(0)],
        [F(coupling), F(diagonal), F(0)],
        [F(0), F(0), F(diagonal - coupling, 2)],
    ]


def _exact_section(polarity: int) -> list[list[Fraction]]:
    membrane = _isotropic_block(120, 30)
    coupling = _isotropic_block(6, 2)
    bending = _isotropic_block(20, 4)
    result = [[F(0) for _ in range(8)] for _ in range(8)]
    for row in range(3):
        for column in range(3):
            result[row][column] = membrane[row][column]
            result[row][3 + column] = F(polarity) * coupling[row][column]
            result[3 + row][column] = F(polarity) * coupling[column][row]
            result[3 + row][3 + column] = bending[row][column]
    result[6][6] = F(50)
    result[7][7] = F(50)
    return result


def _ultrathin_exact_section() -> list[list[Fraction]]:
    """A positive isotropic section with an exact ``D/A = 1e-12/12``."""

    membrane = _isotropic_block(120, 30)
    bending = [
        [value * F(1, 12 * 10**12) for value in row]
        for row in membrane
    ]
    result = [[F(0) for _ in range(8)] for _ in range(8)]
    for row in range(3):
        for column in range(3):
            result[row][column] = membrane[row][column]
            result[3 + row][3 + column] = bending[row][column]
    result[6][6] = F(50)
    result[7][7] = F(50)
    return result


def _isotropic_ultrathin_exact_section() -> list[list[Fraction]]:
    elastic_modulus = F(120)
    poisson = F(1, 4)
    thickness = F(1, 10**6)
    membrane_scale = elastic_modulus * thickness / (1 - poisson * poisson)
    membrane = [
        [membrane_scale, poisson * membrane_scale, F(0)],
        [poisson * membrane_scale, membrane_scale, F(0)],
        [F(0), F(0), (1 - poisson) * membrane_scale / 2],
    ]
    bending = [
        [value * thickness * thickness / 12 for value in row]
        for row in membrane
    ]
    shear = F(5, 6) * elastic_modulus * thickness / (2 * (1 + poisson))
    result = [[F(0) for _ in range(8)] for _ in range(8)]
    for row in range(3):
        for column in range(3):
            result[row][column] = membrane[row][column]
            result[3 + row][3 + column] = bending[row][column]
    result[6][6] = shear
    result[7][7] = shear
    return result


def _ultrathin_b_coupled_exact_section(polarity: int) -> list[list[Fraction]]:
    result = _ultrathin_exact_section()
    membrane = _isotropic_block(120, 30)
    for row in range(3):
        for column in range(3):
            coupling = F(polarity) * membrane[row][column] / F(10**7)
            result[row][3 + column] = coupling
            result[3 + column][row] = coupling
    return result


def _production_case(polarity: int):
    local = np.asarray(((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.2, 0.9, 0.0)))
    mesh = FEMesh()
    for node_id, point in enumerate(local, start=1):
        mesh.add_node(node_id, *point)
    section = GeneralizedShellSection(
        name="exact-oracle-isotropic-b-coupled",
        A=np.asarray(_isotropic_block(120, 30), dtype=float),
        B=np.asarray(_isotropic_block(6, 2), dtype=float),
        D=np.asarray(_isotropic_block(20, 4), dtype=float),
        As=50.0 * np.eye(2),
        mass_per_area=1.0,
        rotary_inertia_per_area=0.1,
    )
    element = QualifiedE4PLS3ShellElement(
        1,
        (1, 2, 3),
        "carrier",
        thickness=1.0,
        shell_section=section,
        reference_normal=(0.0, 0.0, 1.0),
        director_polarity=polarity,
    )
    components = element.compute_stiffness_components(
        mesh,
        Material("carrier", 1.0, 0.0, density=1.0),
    )
    return element, components


def _float(matrix) -> np.ndarray:
    return np.asarray(oracle.to_float_matrix(matrix), dtype=np.float64)


@pytest.mark.parametrize("polarity", (-1, 1))
def test_exact_fraction_oracle_matches_production_block_hierarchy(polarity: int) -> None:
    element, production = _production_case(polarity)
    exact = oracle.reconstruct_exact_blocks(
        ((F(0), F(0)), (F(6, 5), F(0)), (F(1, 5), F(9, 10))),
        _exact_section(polarity),
        director_polarity=polarity,
    )
    for production_name, exact_value in (
        ("uncondensed_physical", exact.uncondensed_physical),
        ("bubble_block", exact.bubble_block),
        ("bubble_map", exact.bubble_map),
        ("condensed_physical_15", exact.condensed_physical_15),
        ("pl_constraint", exact.pl_constraint),
        ("pl_multiplier_gram", exact.pl_multiplier_gram),
        ("full_saddle", exact.full_saddle_23),
    ):
        expected = _float(exact_value)
        scale = max(float(np.linalg.norm(expected, ord=np.inf)), 1.0)
        np.testing.assert_allclose(
            production[production_name],
            expected,
            rtol=2.0e-12,
            atol=2.0e-12 * scale,
        )
    transform = element._local_dof_transform(production["frame"])
    np.testing.assert_allclose(
        production["total"],
        transform.T @ _float(exact.total_local_18) @ transform,
        rtol=2.0e-12,
        atol=2.0e-10,
    )
    assert exact.k_d == F(45)
    assert production["k_d"] == pytest.approx(float(exact.k_d), rel=3.0e-16)
    assert exact.ranks == {
        "bubble": 2,
        "condensed_physical_15": 9,
        "embedded_physical_18": 9,
        "full_saddle_23": 17,
        "pl": 3,
        "total_18": 12,
        "uncondensed_physical_17": 11,
    }


def test_exact_fraction_oracle_schur_and_pl_cancellation_is_identically_zero() -> None:
    exact = oracle.reconstruct_exact_blocks(
        ((F(0), F(0)), (F(6, 5), F(0)), (F(1, 5), F(9, 10))),
        _exact_section(1),
    )
    coupling_transpose = [row[15:] for row in exact.uncondensed_physical[:15]]
    coupling_transpose = [list(column) for column in zip(*coupling_transpose)]
    residual = oracle._add(
        oracle._multiply(exact.bubble_block, exact.bubble_map),
        coupling_transpose,
    )
    assert all(value == 0 for row in residual for value in row)
    direct_pl = oracle._scale(
        exact.k_d,
        oracle._multiply(
            oracle._transpose(exact.pl_constraint),
            oracle._multiply(exact.pl_multiplier_gram, exact.pl_constraint),
        ),
    )
    assert direct_pl == exact.pl_local_18


@pytest.mark.parametrize("polarity", (-1, 1))
def test_normalized_condensation_preserves_ordinary_thickness_schur_result(
    polarity: int,
) -> None:
    _element, production = _production_case(polarity)
    uncondensed = np.asarray(production["uncondensed_physical"])
    direct_schur = (
        uncondensed[:15, :15]
        + uncondensed[:15, 15:] @ np.asarray(production["bubble_map"])
    )
    normalized = np.asarray(production["condensed_physical_15"])
    relative = float(np.linalg.norm(normalized - direct_schur, ord=np.inf)) / float(
        np.linalg.norm(normalized, ord=np.inf)
    )
    assert relative < 5.0e-16
    np.testing.assert_allclose(normalized, direct_schur, rtol=2.0e-15, atol=5.0e-15)


def test_ultrathin_normalized_bubble_condensation_matches_exact_fraction_oracle() -> None:
    local = ((F(0), F(0)), (F(6, 5), F(0)), (F(1, 5), F(9, 10)))
    exact_section = _ultrathin_exact_section()
    exact = oracle.reconstruct_exact_blocks(local, exact_section)
    local_float = np.asarray(((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.2, 0.9, 0.0)))
    mesh = FEMesh()
    for node_id, point in enumerate(local_float, start=1):
        mesh.add_node(node_id, *point)
    section = GeneralizedShellSection(
        name="exact-oracle-ultrathin",
        A=np.asarray(_isotropic_block(120, 30), dtype=float),
        B=np.zeros((3, 3)),
        D=np.asarray(
            [[float(value) for value in row] for row in exact_section[3:6]],
            dtype=float,
        )[:, 3:6],
        As=50.0 * np.eye(2),
        mass_per_area=1.0,
        rotary_inertia_per_area=0.1,
    )
    element = QualifiedE4PLS3ShellElement(
        1,
        (1, 2, 3),
        "carrier",
        thickness=1.0,
        shell_section=section,
        reference_normal=(0.0, 0.0, 1.0),
    )
    production = element.compute_stiffness_components(
        mesh, Material("carrier", 1.0, 0.0, density=1.0)
    )
    expected = _float(exact.condensed_physical_15)
    actual = np.asarray(production["condensed_physical_15"])
    uncondensed = np.asarray(production["uncondensed_physical"])
    direct = (
        uncondensed[:15, :15]
        + uncondensed[:15, 15:] @ np.asarray(production["bubble_map"])
    )
    nonzero = np.abs(expected) > 0.0
    relative = np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0e-300)
    direct_relative = np.abs(direct - expected) / np.maximum(
        np.abs(expected), 1.0e-300
    )
    normalized_error = float(np.max(relative[nonzero]))
    direct_error = float(np.max(direct_relative[nonzero]))
    assert normalized_error < 5.0e-7
    assert direct_error > 1.0e-4
    assert normalized_error < direct_error / 1000.0
    np.testing.assert_allclose(actual, expected, rtol=5.0e-7, atol=5.0e-25)
    diagnostics = production["floating_matrix_diagnostics"]
    assert diagnostics["bubble_condensation_id"] == (
        s3_element.LINEAR_BUBBLE_CONDENSATION_ID
    )
    assert diagnostics["bubble_backward_error"] < 1.0e-15
    assert diagnostics["bubble_forward_error_bound"] < 1.0e-12


def test_ultrathin_isotropic_material_path_matches_exact_fraction_oracle() -> None:
    local = ((F(0), F(0)), (F(6, 5), F(0)), (F(1, 5), F(9, 10)))
    exact = oracle.reconstruct_exact_blocks(
        local, _isotropic_ultrathin_exact_section()
    )
    mesh = FEMesh()
    for node_id, point in enumerate(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        mesh.add_node(node_id, *point)
    element = QualifiedE4PLS3ShellElement(
        1,
        (1, 2, 3),
        "isotropic",
        thickness=1.0e-6,
        reference_normal=(0.0, 0.0, 1.0),
    )
    production = element.compute_stiffness_components(
        mesh, Material("isotropic", 120.0, 0.25, density=1.0)
    )
    expected = _float(exact.condensed_physical_15)
    actual = np.asarray(production["condensed_physical_15"])
    nonzero = np.abs(expected) > 0.0
    relative = np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0e-300)
    assert float(np.max(relative[nonzero])) < 5.0e-7
    np.testing.assert_allclose(actual, expected, rtol=5.0e-7, atol=5.0e-31)
    diagnostics = production["floating_matrix_diagnostics"]
    assert diagnostics["constitutive_normalization_scale"] == pytest.approx(
        128.0e-6, rel=2.0e-15
    )
    assert diagnostics["bubble_forward_error_bound"] < 1.0e-12


@pytest.mark.parametrize("polarity", (-1, 1))
def test_ultrathin_generalized_B_coupling_and_director_match_exact_oracle(
    polarity: int,
) -> None:
    local = ((F(0), F(0)), (F(6, 5), F(0)), (F(1, 5), F(9, 10)))
    exact_section = _ultrathin_b_coupled_exact_section(polarity)
    exact = oracle.reconstruct_exact_blocks(
        local,
        exact_section,
        director_polarity=polarity,
    )
    mesh = FEMesh()
    for node_id, point in enumerate(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        mesh.add_node(node_id, *point)
    membrane = np.asarray(_isotropic_block(120, 30), dtype=float)
    section = GeneralizedShellSection(
        name="exact-oracle-ultrathin-B-coupled",
        A=membrane,
        B=membrane / 10**7,
        D=membrane / (12.0 * 10**12),
        As=50.0 * np.eye(2),
        mass_per_area=1.0,
        rotary_inertia_per_area=0.1,
    )
    element = QualifiedE4PLS3ShellElement(
        1,
        (1, 2, 3),
        "carrier",
        thickness=1.0,
        shell_section=section,
        reference_normal=(0.0, 0.0, 1.0),
        director_polarity=polarity,
    )
    production = element.compute_stiffness_components(
        mesh, Material("carrier", 1.0, 0.0, density=1.0)
    )
    expected = _float(exact.condensed_physical_15)
    actual = np.asarray(production["condensed_physical_15"])
    matrix_scale = float(np.linalg.norm(expected, ord=np.inf))
    significant = np.abs(expected) > 1.0e-15 * matrix_scale
    relative = np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0e-300)
    normwise_error = float(np.linalg.norm(actual - expected, ord=np.inf)) / matrix_scale
    assert normwise_error < 5.0e-15
    assert float(np.max(relative[significant])) < 5.0e-7
    assert production["floating_matrix_diagnostics"][
        "bubble_forward_error_bound"
    ] < 1.0e-12


def test_normalized_bubble_solve_fails_closed_without_forward_accuracy() -> None:
    operator = np.zeros((8, 17), dtype=float)
    operator[0, 0] = 0.5
    operator[1, 1] = 0.5
    operator[0, 15] = 1.0
    operator[1, 16] = 1.0e-4
    operators = tuple(operator.copy() for _ in s3_element.TRIANGLE_QUADRATURE)
    with pytest.raises(ValueError, match="uncertified forward accuracy"):
        s3_element._normalized_linear_bubble_condensation(
            operators,
            np.eye(8),
            1.0,
        )


def test_exact_oracle_is_independent_and_rejects_nonexact_inputs() -> None:
    tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    assert imports <= {"__future__", "dataclasses", "fractions", "typing"}
    source = ORACLE_PATH.read_text(encoding="utf-8").lower()
    assert "import anyfesolver" not in source
    assert "e4_pl_s3_linear_reference" not in source
    with pytest.raises(TypeError, match="exact oracle inputs"):
        oracle.reconstruct_exact_blocks(
            ((0.0, 0), (1, 0), (0, 1)),
            _exact_section(1),
        )
