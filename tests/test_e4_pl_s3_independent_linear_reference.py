from __future__ import annotations

import ast
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver.shell_sections import GeneralizedShellSection


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "docs/reference_cases/e4_pl_s3_linear_reference.py"


def _load_reference():
    spec = importlib.util.spec_from_file_location(
        "e4_pl_s3_independent_linear_reference",
        REFERENCE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotation resolution requires the module to have a stable
    # import identity while its body executes.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reference = _load_reference()


BASE_NODES = np.asarray(
    (
        (0.17, -0.24, 0.31),
        (1.34, 0.12, 0.69),
        (0.29, 1.03, 0.73),
    ),
    dtype=np.float64,
)
BASE_NORMAL = np.cross(BASE_NODES[1] - BASE_NODES[0], BASE_NODES[2] - BASE_NODES[0])
BASE_NORMAL /= np.linalg.norm(BASE_NORMAL)


def _mesh(nodes: np.ndarray) -> FEMesh:
    mesh = FEMesh()
    for node_id, point in enumerate(nodes, start=1):
        mesh.add_node(node_id, *point)
    return mesh


def _layered_material() -> Material:
    return Material("steel", 210.0e9, 0.29, density=7850.0)


def _generalized_section() -> GeneralizedShellSection:
    return GeneralizedShellSection(
        name="independent-reference-b-coupled",
        A=np.asarray(
            (
                (2.70e8, 0.31e8, 0.10e8),
                (0.31e8, 1.83e8, -0.07e8),
                (0.10e8, -0.07e8, 0.79e8),
            )
        ),
        B=np.asarray(
            (
                (1.9e4, -0.4e4, 0.2e4),
                (0.1e4, 1.3e4, -0.15e4),
                (-0.3e4, 0.2e4, 0.6e4),
            )
        ),
        D=np.asarray(
            (
                (3.6e4, 0.44e4, 0.11e4),
                (0.44e4, 2.3e4, -0.08e4),
                (0.11e4, -0.08e4, 1.2e4),
            )
        ),
        As=np.asarray(((0.91e8, 0.08e8), (0.08e8, 0.63e8))),
        mass_per_area=42.0,
        rotary_inertia_per_area=0.017,
    )


def _assert_close(actual: np.ndarray, expected: np.ndarray, *, factor: float = 3.0e-12) -> None:
    scale = max(float(np.linalg.norm(expected, ord=np.inf)), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=factor, atol=factor * scale)


def _compare_case(
    permutation: tuple[int, int, int],
    polarity: int,
    *,
    generalized: bool,
) -> None:
    nodes = BASE_NODES[np.asarray(permutation)]
    mesh = _mesh(nodes)
    material = _layered_material()
    element = QualifiedE4PLS3ShellElement(
        1,
        (1, 2, 3),
        "steel",
        thickness=0.024,
        material_direction=(0.91, 0.23, 0.34),
        material_angle_deg=11.0,
        shell_section=_generalized_section() if generalized else None,
        reference_normal=BASE_NORMAL,
        director_polarity=polarity,
    )
    # Odd D3 actions are intentionally exercised below the production quality
    # admission layer; their operator covariance is still a formulation gate.
    production = element._compute_stiffness_components(
        mesh,
        material,
        enforce_positive_winding=False,
    )
    independent = reference.reconstruct_linear_blocks(
        production["local_nodes"][:, :2],
        production["constitutive"],
        production["constitutive"][:3, :3],
        director_polarity=polarity,
    )

    _assert_close(production["uncondensed_physical"], independent.uncondensed_physical)
    _assert_close(production["bubble_block"], independent.bubble_block)
    _assert_close(production["bubble_map"], independent.bubble_map, factor=8.0e-12)
    _assert_close(production["condensed_physical_15"], independent.condensed_physical_15)
    _assert_close(production["pl_constraint"], independent.pl_constraint)
    _assert_close(production["pl_multiplier_gram"], independent.pl_multiplier_gram)
    _assert_close(production["full_saddle"], independent.full_saddle_23)
    assert production["k_d"] == pytest.approx(independent.k_d, rel=3.0e-13)

    transform = element._local_dof_transform(production["frame"])
    _assert_close(
        production["physical"],
        transform.T @ independent.physical_local_18 @ transform,
    )
    _assert_close(
        production["pl"],
        transform.T @ independent.pl_local_18 @ transform,
    )
    _assert_close(
        production["total"],
        transform.T @ independent.total_local_18 @ transform,
    )

    rng = np.random.default_rng(1000 + 100 * int(generalized) + 10 * polarity + sum(permutation))
    local_virtual = rng.standard_normal(18)
    external_virtual = transform.T @ local_virtual
    production_work = float(external_virtual @ production["total"] @ external_virtual)
    independent_work = float(local_virtual @ independent.total_local_18 @ local_virtual)
    assert production_work == pytest.approx(independent_work, rel=5.0e-12, abs=1.0e-8)

    assert production["ranks"] == {
        "bubble_rank": 2,
        "condensed_physical_rank": 9,
        "embedded_physical_rank": 9,
        "pl_rank": 3,
        "saddle_inertia": (14, 3, 6),
        "saddle_rank": 17,
        "total_rank": 12,
        "uncondensed_physical_rank": 11,
    }
    assert production["floating_matrix_diagnostics"]["saddle_inertia"] == (14, 3, 6)


@pytest.mark.parametrize("generalized", (False, True), ids=("layered", "generalized"))
@pytest.mark.parametrize("polarity", (-1, 1), ids=("director-minus", "director-plus"))
@pytest.mark.parametrize("permutation", tuple(itertools.permutations((0, 1, 2))))
def test_independent_reference_matches_every_d3_action_and_director_polarity(
    permutation: tuple[int, int, int],
    polarity: int,
    generalized: bool,
) -> None:
    _compare_case(permutation, polarity, generalized=generalized)


def test_reference_is_independent_of_anyfesolver_and_producer_mechanics() -> None:
    tree = ast.parse(REFERENCE_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    assert imports <= {"__future__", "dataclasses", "math", "typing", "numpy"}
    assert "anyfesolver" not in REFERENCE_PATH.read_text(encoding="utf-8").lower().replace(
        "anyfesolver module", ""
    )


def test_reference_rejects_singular_or_nonpositive_inputs() -> None:
    section = np.eye(8)
    with pytest.raises(ValueError, match="singular triangle"):
        reference.reconstruct_linear_blocks(
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            section,
            section[:3, :3],
        )
    bad = section.copy()
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive definite"):
        reference.reconstruct_linear_blocks(
            ((0.0, 0.0), (1.0, 0.0), (0.2, 0.9)),
            bad,
            np.eye(3),
        )
