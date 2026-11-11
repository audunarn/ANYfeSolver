from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/reference_cases/e4_pl_s3_v5m_parity.py"
CHECKER = ROOT / "docs/reference_cases/e4_pl_s3_v5m_checker.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_serialization_restart_proof_is_independently_accepted() -> None:
    runner = _load("_v5m_runner_serialization_test", RUNNER)
    checker = _load("_v5m_checker_serialization_test", CHECKER)
    proof, _diagnostic = runner.produce("SERIALIZATION_RESTART")
    first = checker.canonical_bytes(checker.verify(proof))
    second = checker.canonical_bytes(checker.verify(json.loads(runner.canonical_bytes(proof))))
    assert first == second
    assert proof["gate_status"] == runner.PASS


def test_small_batch_exercises_exact_scalar_cold_and_warm_route() -> None:
    runner = _load("_v5m_runner_batch_test", RUNNER)
    proof, _diagnostic = runner.produce("BATCH_4096", batch_count=8)
    payload = proof["payload"]
    assert payload["element_count"] == 8
    assert payload["cold_scalar_csr_byte_identical"] is True
    assert payload["cold_warm_csr_byte_identical"] is True
    assert payload["hashes_identical"] is True
    assert payload["warm_global_plan_reused"] is True
    assert payload["vectorized_shell_element_count_warm"] == 8


def test_checker_imports_no_production_or_runner() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert not any("v5m_parity" in name for name in imports)


def test_bounds_terminal_precedence_and_exact_worker_set_are_frozen() -> None:
    runner = _load("_v5m_runner_bounds_test", RUNNER)
    assert runner.WORKER_IDS == ("BATCH_4096", "SERIALIZATION_RESTART", "PACKAGE_WHEEL")
    assert runner.BATCH_ELEMENT_COUNT == 4096
    assert runner.CHILD_TIMEOUT_SECONDS == 600
    assert runner.WAVE_TIMEOUT_SECONDS == 1800
    assert runner.MEMORY_LIMIT_GIB == 24
    assert runner.WORKER_CONCURRENCY == 3
    assert runner.CHECKER_CONCURRENCY == 4
    assert runner.CYCLES == 2
    assert runner.BLOCKED != runner.NO_GO != runner.GO


def test_staged_build_artifact_purge_is_scoped(tmp_path: Path) -> None:
    runner = _load("_v5m_runner_purge_test", RUNNER)
    source = tmp_path / "source"
    retained = source / "src" / "anyfesolver" / "module.py"
    retained.parent.mkdir(parents=True)
    retained.write_text("retained\n", encoding="ascii")
    (source / "build" / "lib").mkdir(parents=True)
    (source / "build" / "lib" / "stale.py").write_text("stale\n", encoding="ascii")
    (source / "src" / "ANYfeSolver.egg-info").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("preserved\n", encoding="ascii")
    removed = runner._purge_staged_build_artifacts(source)
    assert removed == ("build", "src/ANYfeSolver.egg-info")
    assert retained.read_text(encoding="ascii") == "retained\n"
    assert outside.read_text(encoding="ascii") == "preserved\n"


def test_installed_source_hash_is_line_ending_invariant() -> None:
    runner = _load("_v5m_runner_line_ending_test", RUNNER)
    assert runner.normalized_source_sha256(b"first\nsecond\n") == runner.normalized_source_sha256(b"first\r\nsecond\r\n")
