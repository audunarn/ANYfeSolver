from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = (
    ROOT / "docs" / "reference_cases" / "ge_beam3_mixed_independent_checker.py"
)


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "ge_beam3_mixed_independent_checker_under_test", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_mixed_macroelement_certificate() -> None:
    checker = _load_checker()
    result = checker.reconstruct_certificate()

    assert result["status"] == "PASS_LOCAL_LINEAR_IDENTITY"
    assert result["failed_predicates"] == []
    assert result["counts"] == {
        "condensed_nullity": 6,
        "condensed_rank": 12,
        "external_variables": 18,
        "full_nullity": 6,
        "full_rank": 30,
        "internal_rank": 18,
        "local_variables": 18,
        "quotient_dimension": 12,
        "rigid_modes": 6,
    }
    assert all(result["predicates"].values())
    assert result["hashes"]["condensed_operator_sha256"] == (
        "15D9169BACB17DBAEE066DBB8C10CBD07C564D5146093AB0A59CBC0774102747"
    )
    assert result["hashes"]["section_sha256"] == (
        "BC42A7DCB52A9CD8F9057762E4FB3B47910839400E751ADC3048A4A1BD7C7B16"
    )
    assert result["qualification_claim"] == "LOCAL_REFERENCE_LINEAR_IDENTITY_ONLY"
    assert result["production_boundary"] == {
        "production_authorized": False,
        "selector_authorized": False,
    }


def test_partial_legendre_transform_supports_coupled_spd_section() -> None:
    checker = _load_checker()
    section = checker.default_section()
    blocks = checker.partial_legendre_blocks(section)

    assert any(section[row][column] for row in range(3) for column in range(3, 6))
    assert checker._partial_legendre_reconstruction(section, blocks, None)
    result = checker.reconstruct_certificate(section=section)
    assert result["predicates"]["section_spd"]
    assert result["predicates"]["partial_legendre_reconstructs_coupled_section"]

    scaled = [[2 * value for value in row] for row in section]
    scaled_result = checker.reconstruct_certificate(section=scaled)
    assert scaled_result["status"] == "PASS_LOCAL_LINEAR_IDENTITY"
    assert scaled_result["hashes"]["section_sha256"] != result["hashes"][
        "section_sha256"
    ]


@pytest.mark.parametrize(
    ("mutation", "failed_predicate"),
    [
        ("drop_middle_jump", "full_rank_30_nullity_6"),
        ("force_sign", "six_rigid_modes_exact_condensed_nulls"),
        ("moment_one_point", "two_point_exact_p1_shape_mass"),
        ("reversal_map", "full_reversal_covariance"),
        (
            "wrong_legendre_sign",
            "partial_legendre_reconstructs_coupled_section",
        ),
    ],
)
def test_reviewed_mutations_are_detected(
    mutation: str, failed_predicate: str
) -> None:
    checker = _load_checker()
    result = checker.reconstruct_certificate(mutation=mutation)

    assert result["status"] == "FAIL_LOCAL_LINEAR_IDENTITY"
    assert failed_predicate in result["failed_predicates"]


def test_exact_one_and_two_point_integration_identities() -> None:
    checker = _load_checker()
    checks = checker._integration_checks()

    assert checks == {
        "one_point_exact_for_constant_and_linear_moments": True,
        "one_point_not_exact_for_quadratic_moment_energy": True,
        "two_point_exact_p1_shape_mass": True,
        "two_point_exact_through_cubic": True,
        "two_point_not_claimed_beyond_degree_three": True,
    }


def test_canonical_output_is_deterministic_and_exclusive(tmp_path: Path) -> None:
    checker = _load_checker()
    first = checker.canonical_certificate_bytes()
    second = checker.canonical_certificate_bytes()
    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first)["schema"] == checker.SCHEMA

    output = tmp_path / "checker.json"
    completed = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == first
    duplicate = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert duplicate.returncode != 0
    assert output.read_bytes() == first


def test_checker_has_no_production_or_peer_builder_imports() -> None:
    source = CHECKER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    assert imports.isdisjoint({"anyfesolver", "numpy", "scipy", "sympy"})
    assert "ge_beam3_mixed_reference" not in source
    assert "QuadraticBeamElement" not in source
    assert "BeamElement" not in source
