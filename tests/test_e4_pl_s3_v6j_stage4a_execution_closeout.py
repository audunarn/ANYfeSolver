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
AUTHORIZATION = REFERENCE / "e4_pl_s3_v6j_stage4a_execution_authorization.json"
AUTHORITY_PROGRAM = REFERENCE / "e4_pl_s3_v6j_stage4a_authority.py"
EXECUTOR = REFERENCE / "e4_pl_s3_v6j_stage4a_serial_executor.py"
REVIEW = REFERENCE / "e4_pl_s3_v6j_stage4a_execution_review.json"
STATUS = REFERENCE / "e4_pl_s3_v6j_stage4a_execution_status.json"
FORMAL_ROOT = Path(
    r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease\s3-v2d-stage4a-v6j-8538d3cd"
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


def test_v6j_authorization_is_canonical_and_reviewed() -> None:
    authorization = _canonical(AUTHORIZATION)
    review = _canonical(REVIEW)
    status = _canonical(STATUS)
    raw = AUTHORIZATION.read_bytes()
    assert len(raw) == status["authorization"]["bytes"] == 45_275
    assert hashlib.sha256(raw).hexdigest().upper() == status["authorization"]["sha256"]
    assert len(authorization["requests"]) == 27
    assert review["findings"] == []
    assert review["verdict"] == "ACCEPT_E4_PL_S3_V6J_EXECUTION_AUTHORITY_NO_P0_P1"


def test_v6j_authorization_reproduces_exactly_from_frozen_external_inputs() -> None:
    authority = _module(AUTHORITY_PROGRAM, "v6j_closeout_authority")
    made = authority.generate(
        GRAPH,
        ARCHIVE,
        FORMAL_ROOT,
        AUTHORIZATION,
        Path(r"C:\Github\.resource-manager\requests"),
    )
    assert authority.canonical_bytes(made) == AUTHORIZATION.read_bytes()


def test_v6j_serial_executor_accepts_the_frozen_authority_without_running() -> None:
    executor = _module(EXECUTOR, "v6j_closeout_executor")
    authorization, raw, graph, graph_raw = executor.validate_authorization(
        AUTHORIZATION, GRAPH, ARCHIVE
    )
    assert authorization["stage4a_execution_authorized"] is True
    assert authorization["activation_authorized"] is False
    assert len(authorization["requests"]) == 27
    assert hashlib.sha256(raw).hexdigest().upper() == "6EF972F03FB9A735F7A81282C2A6429FBABB2371E824180F3E6FC2CDCD27B9C2"
    assert hashlib.sha256(graph_raw).hexdigest().upper() == "8538D3CDCA52E0204FF521DDB09D68896F468401BCC0D3F9B488E0FEFE4CBA45"


def test_v6j_implementation_commit_identity_is_exact() -> None:
    lines = subprocess.run(
        ["git", "show", "-s", "--format=%H%n%T%n%P%n%s", "ce37604deff57836764e43fbfb80495776ff513c"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    assert lines == [
        "ce37604deff57836764e43fbfb80495776ff513c",
        "919f44205bed4b3495ed2570b0671521f306a620",
        "7426549c7a7b9bb7f3a8b120d5f8d0a47c877d92",
        "docs: authorize S3 V2D Stage 4A request execution",
    ]


def test_v6j_execution_authority_does_not_activate_or_change_defaults() -> None:
    status = _canonical(STATUS)
    assert status["activation_authorized"] is False
    assert status["stage4a_execution_authorized"] is True
    tree = ast.parse((ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("DEFAULT_"):
                defaults[target.id] = node.value.value
    assert defaults["DEFAULT_Q4_FORMULATION"] == "e4-pl"
    assert defaults["DEFAULT_S3_FORMULATION"] == "legacy-s3"
