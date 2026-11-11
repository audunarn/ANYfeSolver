from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/reference_cases/e4_pl_s3_v5l_stage4b.py"
CHECKER = ROOT / "docs/reference_cases/e4_pl_s3_v5l_checker.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_proof_is_independently_accepted() -> None:
    runner = _load("_v5l_runner_test", RUNNER)
    checker = _load("_v5l_checker_test", CHECKER)
    proof, _diagnostic = runner.produce("MODAL_10")
    first = checker.canonical_bytes(checker.verify(proof))
    second = checker.canonical_bytes(checker.verify(dict(proof)))
    assert first == second
    assert proof["gate_status"] == runner.PASS


def test_checker_imports_no_production_or_runner() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert not any("v5l_stage4b" in name for name in imports)


def test_bounds_coverage_and_terminal_precedence_are_frozen() -> None:
    runner = _load("_v5l_bounds_test", RUNNER)
    assert runner.WORKER_IDS == ("MODAL_10", "MODAL_25", "BUCKLING_10", "BUCKLING_25", "PERFORMANCE_0", "PERFORMANCE_10", "PERFORMANCE_25")
    assert runner.CHILD_TIMEOUT_SECONDS == 600
    assert runner.WAVE_TIMEOUT_SECONDS == 1800
    assert runner.WORKER_CONCURRENCY == 3
    assert runner.CYCLES == 2
    assert runner.BLOCKED != runner.NO_GO_EIGEN != runner.NO_GO_PERFORMANCE != runner.GO
