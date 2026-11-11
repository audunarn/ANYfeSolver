from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
AUTHORITY = REFERENCE / "e4_pl_s3_v6i_stage4a_execution_authority.json"
REVIEW = REFERENCE / "e4_pl_s3_v6i_stage4a_execution_review.json"
STATUS = REFERENCE / "e4_pl_s3_v6i_stage4a_execution_status.json"
AUTHORITY_COMMIT = "9b3b5ec640bd0dd59a90ab3a2f59d5e1d4f22ee1"


def _canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return value


def test_v6i_closeout_evidence_is_canonical_and_hash_bound() -> None:
    authority = _canonical(AUTHORITY)
    review = _canonical(REVIEW)
    status = _canonical(STATUS)
    assert len(AUTHORITY.read_bytes()) == status["authority"]["bytes"]
    assert hashlib.sha256(AUTHORITY.read_bytes()).hexdigest().upper() == status["authority"]["sha256"]
    assert len(REVIEW.read_bytes()) == status["review"]["bytes"]
    assert hashlib.sha256(REVIEW.read_bytes()).hexdigest().upper() == status["review"]["sha256"]
    assert review["findings"] == []
    assert review["verdict"] == "ACCEPT_E4_PL_S3_V6I_REQUEST_PREPARATION_NO_P0_P1"
    assert authority["terminal"] == status["terminal"]


def test_v6i_authority_commit_identity_and_extent_are_exact() -> None:
    lines = subprocess.run(
        ["git", "show", "-s", "--format=%H%n%T%n%P%n%s", AUTHORITY_COMMIT],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    assert lines == [
        AUTHORITY_COMMIT,
        "dfbfcf3c77b7513e3062d650fafa8a279d2304a8",
        "0fc839c4e6ca6a854f5126448e30e88c2936cd36",
        "docs: authorize S3 V2D Stage 4A bounded execution graph",
    ]
    paths = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", AUTHORITY_COMMIT],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    assert paths == [
        "docs/reference_cases/e4_pl_s3_v6i_stage4a_execution_contract.json",
        "docs/reference_cases/e4_pl_s3_v6i_stage4a_execution_graph.py",
        "docs/reference_cases/e4_pl_s3_v6i_stage4a_request_publisher.py",
        "tests/test_e4_pl_s3_v6i_stage4a_execution_authority.py",
    ]


def test_v6i_closeout_authorizes_only_request_preparation() -> None:
    authority = _canonical(AUTHORITY)
    status = _canonical(STATUS)
    for value in (authority, status):
        assert value["activation_authorized"] is False
        assert value["stage4a_execution_authorized"] is False
        assert value["stage4a_request_preparation_authorized"] is True
        assert value["stage4a_request_publication_authorized"] is False
    assert status["published_request_count"] == 0
    assert status["next_gate"] == "V6J_STAGE4A_REQUEST_AUTHORIZATION_AND_SERIAL_WAVE_EXECUTION"


def test_v6i_closeout_preserves_q4_and_legacy_s3_defaults() -> None:
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


def test_v6i_authority_and_closeout_have_no_production_delta() -> None:
    paths = subprocess.run(
        ["git", "diff", "--name-only", "0fc839c4e6ca6a854f5126448e30e88c2936cd36"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    assert all(path.startswith("docs/reference_cases/") or path.startswith("tests/") for path in paths)
    assert all(not path.startswith("src/") for path in paths)
