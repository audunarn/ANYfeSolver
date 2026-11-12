"""Strict closeout checks for the private GE Beam3 P4 curved-reference core."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
RESULT = REFERENCE / "ge_beam3_curved_p4_scientific_result.json"
REVIEW = REFERENCE / "ge_beam3_curved_p4_scientific_review.json"
STATUS = REFERENCE / "ge_beam3_curved_p4_status.json"
AUTHORITY = REFERENCE / "ge_beam3_curved_p4_scientific_authority.json"
EXECUTION_REVIEW = REFERENCE / "ge_beam3_curved_p4_scientific_execution_review.json"

STUDY = "study_ge_beam3.curved_mixed_q2_reference_core_v1"
CANDIDATE = "CANDIDATE_GE_BEAM3_DC_MIXED_CURVED_Q2_MACRO_V1"
TERMINAL = "PROVISIONAL_GO_GE_BEAM3_P4_PRIVATE_CURVED_REFERENCE_CORE"
C5 = "d306dcd6789a146f063a512a4a2e2c3b52f01d40"
C5_TREE = "3591ccb7440e47f69ae509dbe2b23ee2fa2f5aae"
C4 = "4d3b398f71813fc1733a172c13e3831f48089a80"
P3 = "7aa359c18d1cf5db3dfb84d364afd9870e2da994"
INCIDENT = "5befff866b25a31151c6d02106cc1b415587c33d"
INCIDENT_TREE = "406c6fccbf31fed233a8d02eb2a38e9147f789f0"
INCIDENT_ARCHIVE = (
    "refs/archive/ge-beam3-curved-p4-c5-preclaim-authority-incident-20260905"
)
REQUEST_ID = "173f5eb7c11f42518db832916705f74f"
ATTEMPT_ID = "16413b7610f04ec4b19c0bb4f886ec5d"

RESULT_IDENTITY = (
    1298,
    "91ABC1D1355E5F4450DD475B03720625A4840A19059521DA0E1622D9B90DFAFF",
)
REVIEW_IDENTITY = (
    7588,
    "CF55C08DC4E91FC5E16CDD0928D58F038B7BF1B20F8C4A64D43CF1D4E5A02026",
)
STATUS_IDENTITY = (
    4384,
    "BD41BF63075CF7B03C9DA944E9D166F3C05539AF3CD55B9009AA060614FAC444",
)
AUTHORITY_IDENTITY = (
    7203,
    "F2E9A0475FFA73BA0A8B4371CC2CA5943A1336D6169153EB119660509F2953D7",
)
EXECUTION_REVIEW_IDENTITY = (
    976,
    "3E23689CCCAF7B61BD89DFCAFC4B1F5D255F0C1BD9C6E783BDDDF68C76AA28DF",
)
CYCLE_IDENTITY = (
    1281,
    "CAC58A406F589DE0ADD09536A3567FC995A09A5CEA2833CB171AA124BA287123",
)
PROOF_IDENTITY = (
    254716,
    "5A24120AFA220D4109F7C960693C99E1D6D67892A362E2BA366FA2FE77311A45",
)
CHECK_IDENTITY = (
    737,
    "90799C511EAAD7E3FCAE2CA0378724BB27F72428F3D1F1B30B543841EF749951",
)
EMPTY_SHA256 = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"

CLOSEOUT_PATHS = [
    "docs/reference_cases/ge_beam3_curved_p4_scientific_result.json",
    "docs/reference_cases/ge_beam3_curved_p4_scientific_review.json",
    "docs/reference_cases/ge_beam3_curved_p4_status.json",
    "tests/test_ge_beam3_curved_p4_closeout.py",
]
TERMINAL_PRECEDENCE = [
    "BLOCKED_GE_BEAM3_P4_BASELINE_OR_AUTHORITY",
    "BLOCKED_GE_BEAM3_P4_REFERENCE_PROCESS_OR_EVIDENCE",
    "NO_GO_GE_BEAM3_P4_REFERENCE_REGULARITY",
    "NO_GO_GE_BEAM3_P4_REFERENCE_FRAME_IDENTITY",
    "NO_GO_GE_BEAM3_P4_REFERENCE_REVERSAL_OR_OBJECTIVITY",
    TERMINAL,
]
PRODUCTION_BOUNDARY = {
    "default_activation_authorized": False,
    "distribution_publication_authorized": False,
    "ecosystem_exposure_authorized": False,
    "existing_defaults_unchanged": True,
    "new_public_export_authorized": False,
    "new_selector_authorized": False,
    "qualified_q4_unchanged": True,
    "qualified_s3_v2d_unchanged": True,
    "straight_ge_beam3_unchanged": True,
}

RELEASE_ROOT = Path(
    os.environ.get(
        "GE_BEAM3_P4_RELEASE_ROOT",
        r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease",
    )
)
RESOURCE_MANAGER = Path(
    os.environ.get("GE_BEAM3_P4_RESOURCE_MANAGER", r"C:\Github\.resource-manager")
)
WORK_ROOT = RELEASE_ROOT / f"ge-beam3-curved-p4-formal-{ATTEMPT_ID}"
EXTERNAL_RESULT = RELEASE_ROOT / f"ge-beam3-curved-p4-formal-{ATTEMPT_ID}.result.json"
EXTERNAL_MANIFEST = Path(f"{EXTERNAL_RESULT}.manifest.json")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _decode_strict(raw: bytes) -> object:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _repository_bytes(path: Path) -> bytes:
    resolved = path.resolve()
    assert resolved.is_relative_to(ROOT.resolve())
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    assert b"\r" not in raw
    return raw


def _canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = _repository_bytes(path)
    value = _decode_strict(raw)
    assert isinstance(value, dict)
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert raw == expected
    return raw, value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _record(path: str, identity: tuple[int, str]) -> dict[str, object]:
    return {"bytes": identity[0], "path": path, "sha256": identity[1]}


def _git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )


def _git_text(*arguments: str) -> str:
    result = _git(*arguments)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout.decode("utf-8").strip()


def _lookup(value: dict[str, object], path: str) -> object:
    current: object = value
    for component in path.split("."):
        if component.isdecimal():
            assert isinstance(current, list)
            current = current[int(component)]
        else:
            assert isinstance(current, dict)
            current = current[component]
    return current


STATUS_IDENTITIES: dict[str, object] = {
    "authority.commit": C5,
    "authority.tree": C5_TREE,
    "authority.parent": C4,
    "authority.execution_authority.sha256": AUTHORITY_IDENTITY[1],
    "authority.execution_review.sha256": EXECUTION_REVIEW_IDENTITY[1],
    "cycles.checker_sha256": CHECK_IDENTITY[1],
    "cycles.cycle_sha256": CYCLE_IDENTITY[1],
    "cycles.proof_sha256": PROOF_IDENTITY[1],
    "execution.attempt_id": ATTEMPT_ID,
    "execution.manifest.sha256": "008FCBE76F4C3A5DC39EC48582EC6F9CAC23EC2ADC46DC27B7F70E9784B3C7A3",
    "execution.receipt.sha256": "58CC8175A8F9001F712B03F36C6663700D5BD7DF1A0C7E415BEC764C3F09A48D",
    "execution.request.sha256": "93F899BAB7E8CAFFA49E507780BFE493570AF95F357A2106E72549C772096FAF",
    "execution.request_id": REQUEST_ID,
    "execution.result.sha256": RESULT_IDENTITY[1],
    "incident_history.cancelled_never_run.0.request_id": "d2683826f36948a288f43f97b07351e4",
    "incident_history.cancelled_never_run.1.request_id": "9db525f7b136474f890d4b3672b92d3a",
    "incident_history.failed_preclaim_authority.commit": INCIDENT,
    "incident_history.failed_preclaim_authority.tree": INCIDENT_TREE,
    "review.sha256": REVIEW_IDENTITY[1],
    "terminal": TERMINAL,
}


def _validate_status(status: dict[str, object]) -> None:
    assert set(status) == {
        "authority",
        "candidate_id",
        "closeout_extent",
        "cycles",
        "execution",
        "incident_history",
        "next_authority",
        "production_boundary",
        "production_restriction",
        "review",
        "schema",
        "study_id",
        "terminal",
    }
    assert status["schema"] == "anysolver.ge-beam3-curved-p4-status-v1"
    assert status["study_id"] == STUDY
    assert status["candidate_id"] == CANDIDATE
    assert status["production_boundary"] == PRODUCTION_BOUNDARY
    assert status["production_restriction"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert status["cycles"]["count"] == 2
    assert status["cycles"]["registered_obligations"] == 26
    assert status["cycles"]["byte_identical"] is True
    assert status["execution"]["consumed_exactly_once"] is True
    assert status["execution"]["request_reuse_authorized"] is False
    assert status["execution"]["retry_authorized"] is False
    assert status["execution"]["ledger"]["lifecycle"] == [
        "APPROVED",
        "EXECUTION_STARTED",
        "COMPLETED_PASS",
    ]
    assert status["incident_history"]["history_preserved"] is True
    assert status["incident_history"]["prior_authorities_reclassified"] is False
    assert status["incident_history"]["failed_preclaim_authority"][
        "scientific_child_created"
    ] is False
    assert status["next_authority"] == {
        "execution_authorized": False,
        "permitted_preparation": (
            "SEPARATELY_PREREGISTERED_HASH_FROZEN_INDEPENDENTLY_REVIEWED_PRIVATE_"
            "CURVED_MIXED_MECHANICS_TRANCHE"
        ),
        "public_exposure_authorized": False,
    }
    for path, expected in STATUS_IDENTITIES.items():
        assert _lookup(status, path) == expected


def _adjudicate(
    *,
    authority_ok: bool = True,
    process_ok: bool = True,
    regularity_ok: bool = True,
    frame_ok: bool = True,
    reversal_and_objectivity_ok: bool = True,
) -> str:
    facts = [
        (not authority_ok, TERMINAL_PRECEDENCE[0]),
        (not process_ok, TERMINAL_PRECEDENCE[1]),
        (not regularity_ok, TERMINAL_PRECEDENCE[2]),
        (not frame_ok, TERMINAL_PRECEDENCE[3]),
        (not reversal_and_objectivity_ok, TERMINAL_PRECEDENCE[4]),
    ]
    for failed, terminal in facts:
        if failed:
            return terminal
    return TERMINAL


def test_closeout_records_are_canonical_hash_bound_and_closed() -> None:
    result_raw, result = _canonical(RESULT)
    review_raw, review = _canonical(REVIEW)
    status_raw, status = _canonical(STATUS)
    authority_raw, _authority = _canonical(AUTHORITY)
    execution_review_raw, _execution_review = _canonical(EXECUTION_REVIEW)

    assert (len(result_raw), _sha(result_raw)) == RESULT_IDENTITY
    assert (len(review_raw), _sha(review_raw)) == REVIEW_IDENTITY
    assert (len(status_raw), _sha(status_raw)) == STATUS_IDENTITY
    assert (len(authority_raw), _sha(authority_raw)) == AUTHORITY_IDENTITY
    assert (len(execution_review_raw), _sha(execution_review_raw)) == (
        EXECUTION_REVIEW_IDENTITY
    )
    assert set(result) == {
        "attempt_id",
        "authority",
        "candidate_id",
        "checks",
        "counts",
        "cycle_diagnostic_terminals",
        "cycle_sha256",
        "post_cycle_protected_blobs",
        "production_restriction",
        "request_id",
        "schema",
        "terminal",
    }
    assert set(review) == {
        "findings",
        "reviewed_inputs",
        "reviewer_independence",
        "schema",
        "verdict",
    }
    assert review["findings"] == []
    assert review["verdict"] == (
        "ACCEPT_PROVISIONAL_GO_GE_BEAM3_P4_PRIVATE_CURVED_REFERENCE_CORE_NO_P0_P1_P2"
    )
    _validate_status(status)
    assert result["terminal"] == status["terminal"] == TERMINAL
    assert result["cycle_sha256"] == [CYCLE_IDENTITY[1], CYCLE_IDENTITY[1]]
    assert status["execution"]["result"] == _record(CLOSEOUT_PATHS[0], RESULT_IDENTITY)
    assert status["review"] == {
        **_record(CLOSEOUT_PATHS[1], REVIEW_IDENTITY),
        "findings": [],
        "verdict": review["verdict"],
    }
    assert review["reviewed_inputs"]["committed_result"] == _record(
        CLOSEOUT_PATHS[0], RESULT_IDENTITY
    )


def test_external_receipt_manifest_cycles_and_checker_chain_when_available() -> None:
    request_path = RESOURCE_MANAGER / "requests" / f"{REQUEST_ID}.json"
    if not request_path.is_file():
        pytest.skip("immutable local P4 resource-manager evidence is not present")

    paths = {
        "request": request_path,
        "claim": RESOURCE_MANAGER / "claims" / f"{REQUEST_ID}.json",
        "attempt": RESOURCE_MANAGER / "attempts" / f"{ATTEMPT_ID}.json",
        "receipt": RESOURCE_MANAGER / "receipts" / f"{REQUEST_ID}.json",
        "result": EXTERNAL_RESULT,
        "manifest": EXTERNAL_MANIFEST,
    }
    assert all(path.is_file() for path in paths.values())
    raw = {name: path.read_bytes() for name, path in paths.items()}
    values = {name: _decode_strict(item) for name, item in raw.items()}
    assert raw["claim"] == raw["attempt"]
    assert (len(raw["request"]), _sha(raw["request"])) == (
        1291,
        STATUS_IDENTITIES["execution.request.sha256"],
    )
    assert (len(raw["receipt"]), _sha(raw["receipt"])) == (
        1362,
        STATUS_IDENTITIES["execution.receipt.sha256"],
    )
    assert (len(raw["manifest"]), _sha(raw["manifest"])) == (
        416,
        STATUS_IDENTITIES["execution.manifest.sha256"],
    )
    assert raw["result"] == _repository_bytes(RESULT)
    assert values["claim"]["attempt_id"] == values["receipt"]["attempt_id"] == ATTEMPT_ID
    assert values["claim"]["authority_commit"] == C5
    assert values["receipt"]["claim_sha256"] == _sha(raw["claim"])
    assert values["receipt"]["pending_manifest"]["sha256"] == _sha(raw["manifest"])
    assert values["manifest"]["result"] == {
        "bytes": RESULT_IDENTITY[0],
        "sha256": RESULT_IDENTITY[1],
    }
    assert values["manifest"]["terminal"] == values["receipt"]["terminal"] == TERMINAL

    cycle_raw: list[bytes] = []
    proof_raw: list[bytes] = []
    for index in (1, 2):
        cycle_root = WORK_ROOT / f"cycle-{index}"
        cycle = (cycle_root / "cycle.json").read_bytes()
        proof = (cycle_root / "proof.json").read_bytes()
        check_1 = (cycle_root / "check-1.json").read_bytes()
        check_2 = (cycle_root / "check-2.json").read_bytes()
        cycle_value = _decode_strict(cycle)
        assert isinstance(cycle_value, dict)
        assert (len(cycle), _sha(cycle)) == CYCLE_IDENTITY
        assert (len(proof), _sha(proof)) == PROOF_IDENTITY
        assert check_1 == check_2
        assert (len(check_1), _sha(check_1)) == CHECK_IDENTITY
        assert cycle_value["terminal"] == TERMINAL_PRECEDENCE[1]
        assert cycle_value["diagnostic_checker_terminal"] == TERMINAL
        assert cycle_value["counts"] == {
            "accepted_cases": 6,
            "covered_obligations": 26,
            "registered_obligations": 26,
            "stations": 144,
        }
        assert cycle_value["process"] == {
            "checkers": ["PASS", "PASS"],
            "fresh_checker_processes": True,
            "producer": "PASS",
        }
        assert all(
            (cycle_root / filename).read_bytes() == b""
            for filename in (
                "checker-1.stderr.log",
                "checker-1.stdout.log",
                "checker-2.stderr.log",
                "checker-2.stdout.log",
                "producer.stderr.log",
                "producer.stdout.log",
            )
        )
        assert set(cycle_value["process_log_sha256"].get("checker_stderr", [])) == {
            EMPTY_SHA256
        }
        cycle_raw.append(cycle)
        proof_raw.append(proof)
    assert cycle_raw[0] == cycle_raw[1]
    assert proof_raw[0] == proof_raw[1]


def test_authority_topology_incident_archive_and_protected_mechanics_are_exact() -> None:
    fields = _git_text("show", "-s", "--format=%H%n%T%n%P%n%s", C5).splitlines()
    assert fields == [
        C5,
        C5_TREE,
        C4,
        "docs: authorize GE Beam3 curved P4 reference execution",
    ]
    assert _git_text("diff-tree", "--no-commit-id", "--name-only", "-r", C5).splitlines() == [
        "docs/reference_cases/ge_beam3_curved_p4_scientific_authority.json",
        "docs/reference_cases/ge_beam3_curved_p4_scientific_execution_review.json",
    ]

    archived = _git("rev-parse", "--verify", INCIDENT_ARCHIVE)
    if archived.returncode != 0:
        is_shallow = _git_text("rev-parse", "--is-shallow-repository") == "true"
        assert os.environ.get("GITHUB_ACTIONS") == "true" and is_shallow
    else:
        assert archived.stdout.decode("ascii").strip() == INCIDENT
        incident_fields = _git_text(
            "show", "-s", "--format=%H%n%T%n%P%n%s", INCIDENT
        ).splitlines()
        assert incident_fields == [
            INCIDENT,
            INCIDENT_TREE,
            C4,
            "docs: authorize GE Beam3 curved P4 reference execution",
        ]

    protected = {
        "src/anyfesolver/ge_beam3_element.py": "70542da7fea26dcb2da5b5f9da5fc0b0b8d484ab",
        "src/anyfesolver/ge_beam3_mixed_element.py": "f49062b55b4bf749a1654100cf3a4ed6ffb61d37",
        "src/anyfesolver/ge_beam3_mixed_state.py": "74caa1814448245f907bd3efdb6c8a1cc1d2269d",
        "src/anyfesolver/ge_beam3_state.py": "ba8f65761197c62cb2b7ce74bf488e75f2c24818",
    }
    for path, blob in protected.items():
        assert _git_text("rev-parse", f"{C5}:{path}") == blob
        assert _git_text("rev-parse", f"{P3}:{path}") == blob
    assert _git_text("diff", "--name-only", P3, C5, "--", "src").splitlines() == [
        "src/anyfesolver/ge_beam3_curved_reference.py"
    ]
    assert _git_text("diff", "--name-only", P3, C5, "--", "pyproject.toml", ".github") == ""


def test_closeout_extent_is_exact_after_the_closeout_commit_exists() -> None:
    _status_raw, status = _canonical(STATUS)
    assert status["closeout_extent"] == {
        "exact_paths": CLOSEOUT_PATHS,
        "expected_parent": C5,
        "expected_subject": "docs: close GE Beam3 curved P4 reference core",
    }
    introducing_commit = _git_text(
        "log", "-1", "--format=%H", "--", STATUS.relative_to(ROOT).as_posix()
    )
    if not introducing_commit:
        assert _git_text("rev-parse", "HEAD") == C5
        return
    fields = _git_text(
        "show", "-s", "--format=%H%n%P%n%s", introducing_commit
    ).splitlines()
    assert fields[1:] == [C5, "docs: close GE Beam3 curved P4 reference core"]
    assert _git_text(
        "diff-tree", "--no-commit-id", "--name-only", "-r", introducing_commit
    ).splitlines() == CLOSEOUT_PATHS
    assert _git_text("diff", "--name-only", C5, introducing_commit, "--", "src") == ""
    assert _git_text(
        "diff", "--name-only", C5, introducing_commit, "--", "pyproject.toml", ".github"
    ) == ""


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"authority_ok": False, "process_ok": False}, TERMINAL_PRECEDENCE[0]),
        ({"process_ok": False, "regularity_ok": False}, TERMINAL_PRECEDENCE[1]),
        ({"regularity_ok": False, "frame_ok": False}, TERMINAL_PRECEDENCE[2]),
        ({"frame_ok": False, "reversal_and_objectivity_ok": False}, TERMINAL_PRECEDENCE[3]),
        ({"reversal_and_objectivity_ok": False}, TERMINAL_PRECEDENCE[4]),
        ({}, TERMINAL),
    ],
)
def test_terminal_precedence_is_recomputed(
    overrides: dict[str, bool], expected: str
) -> None:
    assert _adjudicate(**overrides) == expected


@pytest.mark.parametrize("path", sorted(STATUS_IDENTITIES))
def test_bound_identity_mutations_are_rejected(path: str) -> None:
    _raw, status = _canonical(STATUS)
    mutated = copy.deepcopy(status)
    current: object = mutated
    components = path.split(".")
    for component in components[:-1]:
        if component.isdecimal():
            assert isinstance(current, list)
            current = current[int(component)]
        else:
            assert isinstance(current, dict)
            current = current[component]
    assert isinstance(current, dict)
    current[components[-1]] = "0" * 64
    with pytest.raises(AssertionError):
        _validate_status(mutated)


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _decode_strict(b'{"schema":"a","schema":"b"}\n')
    for constant in (b"NaN", b"Infinity", b"-Infinity"):
        with pytest.raises(ValueError, match="nonfinite JSON value"):
            _decode_strict(b'{"value":' + constant + b"}\n")
