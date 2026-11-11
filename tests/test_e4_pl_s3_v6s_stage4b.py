from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/reference_cases/e4_pl_s3_v6s_stage4b.py"
CHECKER = ROOT / "docs/reference_cases/e4_pl_s3_v6s_stage4b_checker.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _modal_proof(checker):
    proof = {
        "activation_authorized": False,
        "candidate_formulation_id": checker.CANDIDATE,
        "gate_status": checker.PASS,
        "payload": {
            "fraction_percent": 10,
            "frequency_error_max_hex": float(0.01).hex(),
            "frequency_gate_passed": True,
            "mac_gate_passed": True,
            "minimum_clustered_mac_hex": float(0.99).hex(),
            "rigid_gate_passed": True,
        },
        "predecessor_terminal": "PROVISIONAL_GO_E4_PL_S3_V6R_STAGE4B_PREPARATION",
        "production_restriction": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
        "schema": checker.SCHEMA,
        "worker_id": "MODAL_10",
    }
    import hashlib

    proof["scientific_payload_sha256"] = hashlib.sha256(checker.canonical_bytes(proof)).hexdigest().upper()
    return proof


def test_checker_is_deterministic_and_detects_mutation() -> None:
    checker = _load("_v6s_checker_test", CHECKER)
    proof = _modal_proof(checker)
    first = checker.canonical_bytes(checker.verify(proof))
    second = checker.canonical_bytes(checker.verify(dict(proof)))
    assert first == second
    changed = dict(proof)
    changed["gate_status"] = checker.FAIL
    import pytest

    with pytest.raises(checker.CheckerError):
        checker.verify(changed)


def test_checker_imports_no_production_or_runner() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert not any("v6s_stage4b" in name for name in imports)


def test_bounds_coverage_and_terminal_precedence_are_frozen() -> None:
    runner = _load("_v6s_runner_test", RUNNER)
    assert runner.WORKER_IDS == (
        "MODAL_10", "MODAL_25", "BUCKLING_10", "BUCKLING_25",
        "PERFORMANCE_0", "PERFORMANCE_10", "PERFORMANCE_25",
    )
    assert runner.CHILD_TIMEOUT_SECONDS == 600
    assert runner.WAVE_TIMEOUT_SECONDS == 1800
    assert runner.MEMORY_LIMIT_GIB == 24
    assert runner.WORKER_CONCURRENCY == 3
    assert runner.CHECKER_CONCURRENCY == 4
    assert runner.CYCLES == 2
    assert len({runner.BLOCKED, runner.NO_GO_EIGEN, runner.NO_GO_PERFORMANCE, runner.GO}) == 4


def test_v2d_adapter_is_explicit_and_defaults_are_unchanged() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'formulation="e4-pl-s3-v2d"' in source
    assert 'CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1' in source
    elements = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in elements
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in elements


def test_v2c_adapter_preserves_implicit_positive_director() -> None:
    ecosystem = ROOT.parents[2]
    for repository in ("ANYfileIO", "ANYgeometry", "ANYmaterial"):
        source = ecosystem / repository / "src"
        if source.is_dir():
            sys.path.insert(0, str(source))
    runner = _load("_v6s_adapter_test", RUNNER)
    lane, authorities = runner._lane()
    built = lane._build_case(authorities, 10, auxiliary=False)
    s3 = [
        built.model.mesh.elements[element_id]
        for element_id, kind in built.element_kinds.items()
        if kind == "S3"
    ]
    assert s3
    assert all(type(element).__name__ == "NativeParityE4PLS3V2DShellElement" for element in s3)
    assert all(element.director_polarity == 1 for element in s3)
