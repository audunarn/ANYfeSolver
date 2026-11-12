from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
INPUT = REFERENCE / "e4_pl_s3_v5i_stage4b_input.json"
RUNNER = REFERENCE / "e4_pl_s3_v5i_stage4b.py"
CHECKER = REFERENCE / "e4_pl_s3_v5i_stage4b_protocol_checker.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load("_v5i_stage4b_runner_test", RUNNER)


@pytest.fixture(scope="module")
def checker():
    return _load("_v5i_stage4b_checker_test", CHECKER)


def test_input_is_canonical_hash_bound_and_exact_extent(runner) -> None:
    raw, value = runner.load_input(INPUT)
    assert raw == runner.canonical_bytes(value)
    for row in value["frozen_inputs"]:
        content = (ROOT / row["path"]).read_bytes()
        assert len(content) == row["bytes"]
        assert hashlib.sha256(content).hexdigest().upper() == row["sha256"]
    assert value["authority_commit"]["exact_paths"] == sorted(
        [
            "docs/reference_cases/e4_pl_s3_v5i_stage4b_input.json",
            "docs/reference_cases/e4_pl_s3_v5i_stage4b_plan.json",
            "tests/test_e4_pl_s3_v5i_stage4b.py",
        ]
    )


def test_protocol_checker_is_standard_library_only() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    allowed = {"argparse", "ast", "hashlib", "json", "pathlib", "subprocess", "typing", "__future__"}
    assert set(name.split(".")[0] for name in imported) <= allowed


def test_terminal_precedence_and_two_cycle_agreement(runner) -> None:
    passed = {"terminal": runner.PASS_TERMINAL}
    assert runner.adjudicate([passed, dict(passed)], byte_identical=True) == runner.PASS_TERMINAL
    performance = {"terminal": runner.NO_GO_PERFORMANCE}
    assert runner.adjudicate([performance, dict(performance)], byte_identical=True) == runner.NO_GO_PERFORMANCE
    eigen = {"terminal": runner.NO_GO_EIGEN}
    assert runner.adjudicate([eigen, dict(eigen)], byte_identical=True) == runner.NO_GO_EIGEN
    blocked = {"terminal": runner.BLOCKED}
    assert runner.adjudicate([blocked, dict(blocked)], byte_identical=True) == runner.BLOCKED
    assert runner.adjudicate([passed, dict(passed)], byte_identical=False) == runner.BLOCKED
    assert runner.adjudicate([passed], byte_identical=True, process_complete=False) == runner.BLOCKED


def test_authorization_is_required_and_hash_bound(tmp_path: Path, runner) -> None:
    input_raw, payload = runner.load_input(INPUT)
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        runner.validate_authorization(missing, input_raw, payload)
    authorization = {
        "activation_authorized": False,
        "authority": {"commit": "0" * 40},
        "execution_authorized": True,
        "input_sha256": "0" * 64,
        "protocol_review_path": "missing.json",
        "protocol_review_sha256": "0" * 64,
        "schema": runner.AUTHORIZATION_SCHEMA,
    }
    path = tmp_path / "authorization.json"
    path.write_bytes(runner.canonical_bytes(authorization))
    with pytest.raises(runner.Stage4BError, match="another input"):
        runner.validate_authorization(path, input_raw, payload)


def test_hash_and_bounds_mutations_are_rejected(tmp_path: Path, runner) -> None:
    _raw, value = runner.load_input(INPUT)
    changed = copy.deepcopy(value)
    changed["frozen_inputs"][0]["sha256"] = "0" * 64
    path = tmp_path / "bad-hash.json"
    path.write_bytes(runner.canonical_bytes(changed))
    with pytest.raises(runner.Stage4BError, match="frozen input mismatch"):
        runner.load_input(path)
    changed = copy.deepcopy(value)
    changed["execution"]["child_timeout_seconds"] = 601
    path = tmp_path / "bad-bound.json"
    path.write_bytes(runner.canonical_bytes(changed))
    with pytest.raises(runner.Stage4BError, match="execution bounds changed"):
        runner.load_input(path)


def test_defaults_and_q4_are_unchanged(runner) -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert runner.CHILD_TIMEOUT_SECONDS == 600
    assert runner.WAVE_TIMEOUT_SECONDS == 1800
    assert runner.MEMORY_LIMIT_GIB == 24
    assert runner.WORKER_CONCURRENCY == 3
    assert "BATCH_4096" not in RUNNER.read_text(encoding="utf-8")
