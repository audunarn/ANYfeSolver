from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
AUTHORIZATION = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_authorization.json"
AUTHORITY = REFERENCE / "e4_pl_s3_v6k_stage4a_authority.py"
EXECUTOR = REFERENCE / "e4_pl_s3_v6k_stage4a_serial_executor.py"
REVIEW = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_review.json"
STATUS = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_status.json"
V6J_AUTHORIZATION = REFERENCE / "e4_pl_s3_v6j_stage4a_execution_authorization.json"
FORMAL_ROOT = Path(
    r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease\s3-v2d-stage4a-v6k-2d91bba2"
)
GRAPH = FORMAL_ROOT / "execution-graph.json"
ARCHIVE = FORMAL_ROOT / "candidate-source.tar"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return value


def test_v6k_authorization_is_canonical_reviewed_and_fresh() -> None:
    authorization = _canonical(AUTHORIZATION)
    review = _canonical(REVIEW)
    status = _canonical(STATUS)
    raw = AUTHORIZATION.read_bytes()
    assert len(raw) == status["authorization"]["bytes"] == 45_275
    assert hashlib.sha256(raw).hexdigest().upper() == status["authorization"]["sha256"]
    old = _canonical(V6J_AUTHORIZATION)
    new_ids = {row["request"]["request_id"] for row in authorization["requests"]}
    old_ids = {row["request"]["request_id"] for row in old["requests"]}
    assert len(new_ids) == 27 and not (new_ids & old_ids)
    assert review["findings"] == []
    assert review["verdict"] == "ACCEPT_E4_PL_S3_V6K_RESOURCE_CORRECTION_NO_P0_P1"


def test_v6k_authorization_reproduces_from_frozen_inputs() -> None:
    authority = _module(AUTHORITY, "v6k_closeout_authority")
    made = authority.generate(
        GRAPH, ARCHIVE, FORMAL_ROOT, AUTHORIZATION,
        Path(r"C:\Github\.resource-manager\requests"),
    )
    assert authority.canonical_bytes(made) == AUTHORIZATION.read_bytes()


def test_v6k_executor_validates_without_execution() -> None:
    executor = _module(EXECUTOR, "v6k_closeout_executor")
    authorization, raw, graph, graph_raw = executor.validate_authorization(
        AUTHORIZATION, GRAPH, ARCHIVE
    )
    assert authorization["stage4a_execution_authorized"] is True
    assert authorization["activation_authorized"] is False
    assert graph["runtime_policy"]["maximum_concurrent_workers"] == 2
    assert len(raw) == 45_275 and len(graph_raw) == 155_579


def test_v6k_request_implementation_commit_is_exact() -> None:
    lines = subprocess.run(
        ["git", "show", "-s", "--format=%H%n%T%n%P%n%s", "face73f18355ed51c016120e43a94485afb14264"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    assert lines == [
        "face73f18355ed51c016120e43a94485afb14264",
        "fad1a625b324df48685f331c5d9bcb712edfa644",
        "c8b076e2c0bfa311ce42c1609acbdb1471767737",
        "docs: authorize corrected S3 V2D Stage 4A requests",
    ]


def test_v6k_closeout_does_not_activate_or_change_defaults() -> None:
    status = _canonical(STATUS)
    assert status["activation_authorized"] is False
    assert status["stage4a_execution_authorized"] is True
    tree = ast.parse((ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in targets:
                if isinstance(target, ast.Name) and target.id.startswith("DEFAULT_"):
                    defaults[target.id] = node.value.value
    assert defaults["DEFAULT_Q4_FORMULATION"] == "e4-pl"
    assert defaults["DEFAULT_S3_FORMULATION"] == "legacy-s3"
