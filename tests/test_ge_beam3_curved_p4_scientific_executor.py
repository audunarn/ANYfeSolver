from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/reference_cases/ge_beam3_curved_p4_scientific_executor.py"
RUNNER = ROOT / "docs/reference_cases/ge_beam3_curved_p4_reference_runner.py"
TEST_RELATIVE = "tests/test_ge_beam3_curved_p4_scientific_executor.py"
EXECUTOR_RELATIVE = "docs/reference_cases/ge_beam3_curved_p4_scientific_executor.py"


def _load_executor(name: str = "p4_scientific_executor"):
    specification = importlib.util.spec_from_file_location(name, PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


executor = _load_executor()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _write(path: Path, value: object) -> bytes:
    raw = executor.canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _ledger_line(fields: list[str]) -> str:
    assert len(fields) == 8
    return "| " + " | ".join(fields) + " |\n"


def _synthetic_commits() -> dict[str, dict[str, str]]:
    c3 = {
        "commit": "3" * 40,
        "parent": executor.C2["commit"],
        "subject": executor.C3_SUBJECT,
        "tree": "4" * 40,
    }
    c4 = {
        "commit": "5" * 40,
        "parent": c3["commit"],
        "subject": executor.C4_SUBJECT,
        "tree": "6" * 40,
    }
    return {"c1": dict(executor.C1), "c2": dict(executor.C2), "c3": c3, "c4": c4}


def _resource_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manager = (tmp_path / "resource-manager").resolve()
    repository = (tmp_path / "repository").resolve()
    manager.mkdir()
    repository.mkdir()
    (manager / "requests").mkdir()
    (manager / "claims").mkdir()
    (manager / "attempts").mkdir()
    (manager / "receipts").mkdir()
    acquire_raw = b"synthetic acquire helper\n"
    release_raw = b"synthetic release helper\n"
    (manager / "acquire-test.ps1").write_bytes(acquire_raw)
    (manager / "release-test.ps1").write_bytes(release_raw)

    request_id = "1" * 32
    attempt_id = "2" * 32
    argv = [sys.executable, "-I", "-B", str(PATH), "--execute", "--output", "OUT"]
    command = executor.expected_request_command(argv)
    request = {
        "command": command,
        "estimate_minutes": 30,
        "repository": str(repository),
        "request_id": request_id,
        "requested_at": "2026-09-05T12:00:00Z",
        "status": "PENDING",
        "task": "GE Beam3 curved P4 bounded scientific execution",
    }
    request_raw = _write(manager / "requests" / f"{request_id}.json", request)
    request_binding = {
        "attempt_id": attempt_id,
        "argv": argv,
        "command": command,
        "record": {"bytes": len(request_raw), "sha256": _sha(request_raw)},
        "request_id": request_id,
    }
    note_authority = {"commits": _synthetic_commits(), "request": request_binding}
    approved = [
        "2026-09-05T12:01:00Z",
        request_id,
        "APPROVED",
        request["task"],
        str(repository),
        "Exact command in immutable request JSON",
        "30 minutes",
        executor.approval_ledger_note(note_authority),
    ]
    started = [
        "2026-09-05T12:02:00Z",
        request_id,
        "EXECUTION_STARTED",
        request["task"],
        str(repository),
        "Exact command in immutable request JSON",
        "30 minutes",
        executor.execution_started_ledger_note(note_authority),
    ]
    header = (
        "# Synthetic resource ledger\n\n"
        "| Timestamp (ISO 8601) | Request ID | Status | Task | Repository | Command / scope | Estimate | Notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    approved_prefix = (header + _ledger_line(approved)).encode("utf-8")
    ledger_raw = approved_prefix + _ledger_line(started).encode("utf-8")
    (manager / "ledger.md").write_bytes(ledger_raw)
    authority = {
        "commits": note_authority["commits"],
        "request": request_binding,
        "resource_manager": {
            "acquire": {
                "bytes": len(acquire_raw),
                "filename": "acquire-test.ps1",
                "sha256": _sha(acquire_raw),
            },
            "approved_ledger_prefix": {
                "bytes": len(approved_prefix),
                "sha256": _sha(approved_prefix),
            },
            "approved_row": approved,
            "execution_started_row": started,
            "ledger": "ledger.md",
            "release": {
                "bytes": len(release_raw),
                "filename": "release-test.ps1",
                "sha256": _sha(release_raw),
            },
            "root": str(manager),
        },
    }
    return manager, repository, authority


def _diagnostic_cycle(terminal: str, *, marker: str = "A") -> dict[str, Any]:
    return {
        "candidate_id": executor.CANDIDATE_ID,
        "check_sha256": [marker * 64, marker * 64],
        "counts": {
            "accepted_cases": 6,
            "covered_obligations": 26,
            "registered_obligations": 26,
            "stations": 144,
        },
        "diagnostic_checker_terminal": terminal,
        "process": {
            "checkers": ["PASS", "PASS"],
            "fresh_checker_processes": True,
            "producer": "PASS",
        },
        "process_log_sha256": {
            "checker_stderr": [marker * 64, marker * 64],
            "checker_stdout": [marker * 64, marker * 64],
            "producer_stderr": marker * 64,
            "producer_stdout": marker * 64,
        },
        "production_restriction": executor.RESTRICTION,
        "proof_sha256": marker * 64,
        "schema": "anysolver.ge-beam3-curved-p4-reference-cycle-v2",
        "status": "COMPLETE",
        "terminal": executor.BLOCKED_PROCESS,
    }


def _validated_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], Any]:
    manager, repository, authority = _resource_fixture(tmp_path)
    authority["locations"] = {
        "work_root": str((tmp_path / "scientific-work").absolute()),
    }
    request_path = manager / "requests" / f"{authority['request']['request_id']}.json"
    request_raw, request = executor.strict_json(request_path)
    authority_raw = executor.canonical_bytes(authority)
    validated = executor.ValidatedAuthority(
        repository=repository,
        authority=authority,
        authority_raw=authority_raw,
        review_raw=b'{"synthetic":true}\n',
        request=request,
        request_raw=request_raw,
        c3={"commit": "3" * 40, "parent": "2" * 40, "subject": executor.C3_SUBJECT, "tree": "4" * 40},
        c4={"commit": "5" * 40, "parent": "3" * 40, "subject": executor.C4_SUBJECT, "tree": "6" * 40},
        c5={"commit": "7" * 40, "parent": "5" * 40, "subject": executor.C5_SUBJECT, "tree": "8" * 40},
    )
    return manager, repository, authority, validated


def _install_active_lock(manager: Path, request: dict[str, Any]) -> bytes:
    owner = {
        "acquired_at": "2026-09-05T12:02:00Z",
        "command": request["command"],
        "process_id": 4242,
        "repository": request["repository"],
        "request_id": request["request_id"],
        "task": request["task"],
    }
    raw = executor.canonical_bytes(owner)
    active = manager / "active-lock"
    active.mkdir()
    (active / "owner.json").write_bytes(raw)
    return raw


def _minimal_result(validated: Any, terminal: str | None = None) -> dict[str, Any]:
    terminal = terminal or executor.PASS
    return {
        "attempt_id": validated.authority["request"]["attempt_id"],
        "authority": {
            "commit": validated.c5["commit"],
            "sha256": _sha(validated.authority_raw),
            "tree": validated.c5["tree"],
        },
        "candidate_id": executor.CANDIDATE_ID,
        "checks": {
            "all_launched_processes_terminal": True,
            "byte_identical_canonical_cycles": True,
            "formal_execution_authorized": True,
            "independent_checker_replicas": True,
            "post_cycle_protected_blobs": True,
            "production_activation_authorized": False,
        },
        "counts": {"cycles_complete": 2, "cycles_launched": 2, "registered_obligations": 26},
        "cycle_diagnostic_terminals": [terminal, terminal],
        "cycle_sha256": ["A" * 64, "A" * 64],
        "post_cycle_protected_blobs": {"count": 7, "sha256": "B" * 64, "status": "PASS"},
        "production_restriction": executor.RESTRICTION,
        "request_id": validated.authority["request"]["request_id"],
        "schema": executor.RESULT_SCHEMA,
        "terminal": terminal,
    }


def _authority_schema_fixture(tmp_path: Path) -> dict[str, Any]:
    manager, repository, resource = _resource_fixture(tmp_path)
    del manager

    def identity(path: str) -> dict[str, Any]:
        target = ROOT / path
        raw = target.read_bytes()
        return {"bytes": len(raw), "path": path, "sha256": _sha(raw)}

    commits = _synthetic_commits()
    c3 = commits["c3"]
    c4 = commits["c4"]
    return {
        "activation_authorized": False,
        "bounds": dict(executor.BOUNDS),
        "candidate_id": executor.CANDIDATE_ID,
        "commits": commits,
        "execution_authorized": True,
        "harness": {
            "authority_manifest_sha256": executor.FROZEN_AUTHORITY_MANIFEST_SHA256,
            "checker": identity("docs/reference_cases/ge_beam3_curved_p4_reference_checker.py"),
            "executor": identity(EXECUTOR_RELATIVE),
            "producer": identity("docs/reference_cases/ge_beam3_curved_p4_reference_producer.py"),
            "runner": identity("docs/reference_cases/ge_beam3_curved_p4_reference_runner.py"),
        },
        "locations": {
            "authority": str(repository / executor.AUTHORITY_RELATIVE),
            "executor": str(repository / EXECUTOR_RELATIVE),
            "output": str((tmp_path / "external" / "result.json").absolute()),
            "repository": str(repository),
            "review": str(repository / executor.REVIEW_RELATIVE),
            "work_root": str((tmp_path / "external" / "work").absolute()),
        },
        "production_boundary": {
            "default_activation_authorized": False,
            "distribution_publication_authorized": False,
            "ecosystem_exposure_authorized": False,
            "existing_defaults_unchanged": True,
            "qualified_q4_unchanged": True,
            "qualified_s3_v2d_unchanged": True,
        },
        "publication_authorized": False,
        "request": resource["request"],
        "resource_manager": resource["resource_manager"],
        "schema": executor.AUTHORITY_SCHEMA,
        "study_id": executor.STUDY_ID,
    }


def test_strict_json_rejects_duplicate_nonfinite_noncanonical_and_nonobject(tmp_path: Path) -> None:
    malformed = {
        "duplicate": b'{"a":1,"a":2}\n',
        "nan": b'{"a":NaN}\n',
        "infinity": b'{"a":Infinity}\n',
        "unsorted": b'{"z":1,"a":2}\n',
        "whitespace": b'{"a": 1}\n',
        "crlf": b'{"a":1}\r\n',
        "array": b'[]\n',
        "trailing": b'{"a":1}\n\n',
    }
    for label, raw in malformed.items():
        path = tmp_path / f"{label}.json"
        path.write_bytes(raw)
        with pytest.raises(executor.ExecutionError):
            executor.strict_json(path)


def test_canonical_serialization_is_deterministic_and_rejects_bad_values() -> None:
    value = {"z": [True, 2], "a": "x"}
    assert executor.canonical_bytes(value) == b'{"a":"x","z":[true,2]}\n'
    assert executor.canonical_bytes(copy.deepcopy(value)) == executor.canonical_bytes(value)
    for bad in (float("nan"), float("inf"), {1: "not-a-string-key"}, (1, 2), object()):
        with pytest.raises(executor.ExecutionError):
            executor.canonical_bytes({"bad": bad})


def test_frozen_chain_paths_subjects_terminals_and_bounds() -> None:
    assert executor.STUDY_ID == "study_ge_beam3.curved_mixed_q2_reference_core_v1"
    assert executor.C1 == {
        "commit": "780991f7adea6fb4dc2da9fcdeca1d37379bf2ba",
        "parent": "8fcf827b6e5364f0e1bc8221a3f74602e3f39261",
        "subject": "test: freeze GE Beam3 curved P4 reference harness",
        "tree": "3b14a1ba42a47cd186b299539196731c4505a408",
    }
    assert executor.C2 == {
        "commit": "20e895e8803de086b8abf19e3183e4d6a3b96fd6",
        "parent": executor.C1["commit"],
        "subject": "docs: review GE Beam3 curved P4 reference harness",
        "tree": "76743dd2167b7accf2f480f5606d0095a5513b7a",
    }
    assert executor.C3_PATHS == (EXECUTOR_RELATIVE, TEST_RELATIVE)
    assert executor.C3_SUBJECT == "test: add ledger-backed GE Beam3 curved P4 executor"
    assert executor.C4_PATHS == (
        "docs/reference_cases/ge_beam3_curved_p4_scientific_executor_review.json",
    )
    assert executor.C4_SUBJECT == "docs: review GE Beam3 curved P4 scientific executor"
    assert executor.C5_PATHS == (
        "docs/reference_cases/ge_beam3_curved_p4_scientific_authority.json",
        "docs/reference_cases/ge_beam3_curved_p4_scientific_execution_review.json",
    )
    assert executor.C5_SUBJECT == "docs: authorize GE Beam3 curved P4 reference execution"
    assert executor.BOUNDS == {
        "child_timeout_seconds": 600,
        "inactivity_seconds": 300,
        "memory_limit_bytes": 24 * (1 << 30),
        "wave_timeout_seconds": 1800,
    }
    assert executor.TERMINALS == (
        executor.BLOCKED_AUTHORITY,
        executor.BLOCKED_PROCESS,
        executor.NO_GO_REGULARITY,
        executor.NO_GO_FRAME,
        executor.NO_GO_COVARIANCE,
        executor.PASS,
    )
    assert set(executor.C3_PATHS).isdisjoint(executor.C4_PATHS)
    assert set(executor.C3_PATHS).isdisjoint(executor.C5_PATHS)
    assert set(executor.C4_PATHS).isdisjoint(executor.C5_PATHS)
    assert executor.FROZEN_AUTHORITY_MANIFEST_SHA256 == (
        "13308601CA45CE5DB7CB8A32138ABB273560F1EF426F0763E45F59618D46F7F7"
    )
    assert executor.FROZEN_C1_PROGRAMS == {
        "checker": {
            "bytes": 93225,
            "path": "docs/reference_cases/ge_beam3_curved_p4_reference_checker.py",
            "sha256": "018C3A84FA89EA9C460C9E2951EE6F694B015BA2E100696DBF77C47BB9CB3083",
        },
        "producer": {
            "bytes": 58885,
            "path": "docs/reference_cases/ge_beam3_curved_p4_reference_producer.py",
            "sha256": "CD61E00D06B48E9905799AE0C83E3EECB9E0692339ECEC3E9D9BAD090078B725",
        },
        "runner": {
            "bytes": 65783,
            "path": "docs/reference_cases/ge_beam3_curved_p4_reference_runner.py",
            "sha256": "7D61E0A10D9291046F2F7B5A8429100AF399711BB13D33FB29BD250CCC68B2CD",
        },
    }


def test_authority_schema_is_noncyclic_closed_and_private(tmp_path: Path) -> None:
    authority = _authority_schema_fixture(tmp_path)
    assert executor._validate_authority_schema(authority) == authority
    assert set(authority["commits"]) == {"c1", "c2", "c3", "c4"}
    assert "c5" not in authority["commits"]
    assert "authority_sha256" not in executor.canonical_bytes(authority).decode("ascii")
    assert authority["execution_authorized"] is True
    assert authority["activation_authorized"] is False
    assert authority["publication_authorized"] is False
    assert authority["production_boundary"] == {
        "default_activation_authorized": False,
        "distribution_publication_authorized": False,
        "ecosystem_exposure_authorized": False,
        "existing_defaults_unchanged": True,
        "qualified_q4_unchanged": True,
        "qualified_s3_v2d_unchanged": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_key", "bounds", "activation", "publication", "c1", "c2",
        "c3_parent", "c4_parent", "production", "relative_location", "harness_path",
        "manifest_hash", "harness_hash", "harness_bytes",
    ),
)
def test_authority_scope_chain_and_hash_bindings_are_closed(
    tmp_path: Path, mutation: str,
) -> None:
    authority = _authority_schema_fixture(tmp_path)
    if mutation == "extra_key":
        authority["unexpected"] = True
    elif mutation == "bounds":
        authority["bounds"]["wave_timeout_seconds"] = 1801
    elif mutation == "activation":
        authority["activation_authorized"] = True
    elif mutation == "publication":
        authority["publication_authorized"] = True
    elif mutation == "c1":
        authority["commits"]["c1"]["tree"] = "f" * 40
    elif mutation == "c2":
        authority["commits"]["c2"]["parent"] = "f" * 40
    elif mutation == "c3_parent":
        authority["commits"]["c3"]["parent"] = "f" * 40
    elif mutation == "c4_parent":
        authority["commits"]["c4"]["parent"] = "f" * 40
    elif mutation == "production":
        authority["production_boundary"]["existing_defaults_unchanged"] = False
    elif mutation == "relative_location":
        authority["locations"]["output"] = "relative/result.json"
    elif mutation == "harness_path":
        authority["harness"]["runner"]["path"] = "src/anyfesolver/elements.py"
    elif mutation == "manifest_hash":
        authority["harness"]["authority_manifest_sha256"] = "F" * 64
    elif mutation == "harness_hash":
        authority["harness"]["checker"]["sha256"] = "F" * 64
    elif mutation == "harness_bytes":
        authority["harness"]["producer"]["bytes"] += 1
    with pytest.raises(executor.ExecutionError):
        executor._validate_authority_schema(authority)


def test_review_binds_authority_and_overlay_without_c5_self_reference(tmp_path: Path) -> None:
    authority = _authority_schema_fixture(tmp_path)
    raw = executor.canonical_bytes(authority)
    c4 = authority["commits"]["c4"]
    review = {
        "findings": [],
        "reviewed_inputs": {
            "authority": {
                "bytes": len(raw),
                "path": executor.AUTHORITY_RELATIVE,
                "sha256": _sha(raw),
            },
            "authorization_overlay": {
                "exact_paths": list(executor.C5_PATHS),
                "expected_parent": c4["commit"],
                "subject": executor.C5_SUBJECT,
            },
        },
        "reviewer_independence": {
            "authority_authorship": False,
            "executor_authorship": False,
            "formal_execution_performed": False,
            "harness_authorship": False,
            "review_artifact_authorship": True,
            "reviewer_role": "INDEPENDENT_GE_BEAM3_CURVED_P4_SCIENTIFIC_EXECUTION_REVIEWER",
            "scientific_execution_performed": False,
        },
        "schema": executor.REVIEW_SCHEMA,
        "verdict": "ACCEPT_GE_BEAM3_CURVED_P4_SCIENTIFIC_EXECUTION_AUTHORITY_NO_P0_P1_P2",
    }
    serialized = executor.canonical_bytes(review).decode("ascii")
    assert "7" * 40 not in serialized and "8" * 40 not in serialized
    executor._validate_review(review, {"bytes": len(raw), "sha256": _sha(raw)}, c4)
    for key, value in (
        ("findings", [{"priority": "P1"}]),
        ("verdict", "REJECT"),
    ):
        mutated = copy.deepcopy(review)
        mutated[key] = value
        with pytest.raises(executor.ExecutionError):
            executor._validate_review(mutated, {"bytes": len(raw), "sha256": _sha(raw)}, c4)

    for key, value in (
        ("expected_parent", "f" * 40),
        ("subject", "docs: wrong authority subject"),
        ("exact_paths", list(reversed(executor.C5_PATHS))),
    ):
        mutated = copy.deepcopy(review)
        mutated["reviewed_inputs"]["authorization_overlay"][key] = value
        with pytest.raises(executor.ExecutionError):
            executor._validate_review(
                mutated, {"bytes": len(raw), "sha256": _sha(raw)}, c4,
            )


def test_c4_executor_review_binds_exact_five_file_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    c3 = _synthetic_commits()["c3"]
    identities = {
        relative: {"bytes": index + 10, "sha256": chr(65 + index) * 64}
        for index, relative in enumerate(executor.C3_PATHS)
    }
    monkeypatch.setattr(
        executor,
        "_blob_identity",
        lambda _repository, _commit, relative: identities[relative],
    )
    review = {
        "findings": [],
        "reviewed_inputs": {
            "candidate": c3,
            "candidate_paths": [
                {**identities[relative], "path": relative}
                for relative in executor.C3_PATHS
            ],
            "test_receipts": [{
                "failed": 0,
                "formal_execution": False,
                "passed": executor.EXPECTED_C4_TEST_COUNT,
                "receipt_id": "INDEPENDENT_P4_SCIENTIFIC_EXECUTOR_FULL_SUITE_20260905",
                "scientific_execution": False,
                "status": "PASS",
                "test_paths": [
                    "tests/test_ge_beam3_curved_p4_preregistration.py",
                    "tests/test_ge_beam3_curved_p4_reference.py",
                    "tests/test_ge_beam3_curved_p4_implementation_review.py",
                    "tests/test_ge_beam3_curved_p4_formal_runner.py",
                    "tests/test_ge_beam3_curved_p4_scientific_executor.py",
                ],
            }],
        },
        "reviewer_independence": {
            "executor_authorship": False,
            "formal_execution_performed": False,
            "review_artifact_authorship": True,
            "reviewer_role": "INDEPENDENT_GE_BEAM3_CURVED_P4_SCIENTIFIC_EXECUTOR_REVIEWER",
            "scientific_execution_performed": False,
            "test_authorship": False,
        },
        "schema": executor.EXECUTOR_REVIEW_SCHEMA,
        "verdict": "ACCEPT_GE_BEAM3_CURVED_P4_SCIENTIFIC_EXECUTOR_NO_P0_P1_P2",
    }
    executor._validate_executor_review(review, c3=c3, repository=tmp_path)
    for mutation in ("count", "path", "finding", "authorship"):
        changed = copy.deepcopy(review)
        if mutation == "count":
            changed["reviewed_inputs"]["test_receipts"][0]["passed"] += 1
        elif mutation == "path":
            changed["reviewed_inputs"]["test_receipts"][0]["test_paths"].pop()
        elif mutation == "finding":
            changed["findings"] = [{"priority": "P1"}]
        else:
            changed["reviewer_independence"]["executor_authorship"] = True
        with pytest.raises(executor.ExecutionError):
            executor._validate_executor_review(changed, c3=c3, repository=tmp_path)


def test_executor_is_stdlib_only_and_cannot_author_request_or_authority() -> None:
    source = PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    allowed = {
        "__future__", "argparse", "dataclasses", "datetime", "hashlib",
        "importlib", "json", "math", "os", "pathlib", "re", "shutil",
        "signal", "stat", "subprocess", "sys", "time", "typing",
    }
    assert imports <= allowed
    assert "numpy" not in imports and "anyfesolver" not in imports
    assert "--create-authority" not in source
    assert "--create-request" not in source
    # The official request helper may be named in a comment, but the executor
    # has no request-authoring subprocess path or CLI mode.
    assert "--create-request" not in source
    assert "run_gate(" not in source
    assert "_adjudicate_terminals(" not in source


def test_all_required_synthetic_lifecycle_apis_exist() -> None:
    for name in (
        "validate_authority",
        "validate_request_and_ledger",
        "validate_active_lock",
        "validate_chain",
        "validate_request",
        "validate_ledger_lifecycle",
        "validate_active_lock_owner",
        "acquire_claim",
        "release_claim",
        "adjudicate_cycles",
        "adjudicate_diagnostic_cycles",
        "build_result",
        "publish_or_recover",
        "execute",
        "main",
    ):
        assert callable(getattr(executor, name, None)), name


def test_request_and_strict_eight_column_ledger_accept_exact_lifecycle(tmp_path: Path) -> None:
    manager, repository, authority = _resource_fixture(tmp_path)
    request, raw, rows = executor.validate_request_and_ledger(
        authority,
        repository=repository,
        resource_manager=manager,
    )
    assert request["request_id"] == "1" * 32
    assert raw == (manager / "requests" / f"{'1' * 32}.json").read_bytes()
    assert [row[2] for row in rows if row[1] == "1" * 32] == [
        "APPROVED",
        "EXECUTION_STARTED",
    ]
    assert all(len(row) == 8 for row in rows)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("request_hash", "request"),
        ("request_id", "request"),
        ("repository", "scope"),
        ("command", "scope"),
        ("argv", "argv|command"),
        ("prefix", "prefix"),
        ("approval_duplicate", "reused|approval|row"),
        ("started_duplicate", "reused|start|row"),
        ("terminal", "reused|terminal"),
        ("cancelled", "reused|cancel"),
        ("row_order", "chronology"),
        ("approval_binding", "approval|row"),
        ("start_binding", "start|row"),
        ("prefix_extra", "prefix|approval"),
        ("timestamp_order", "chronology"),
        ("started_task", "start|scope"),
        ("started_repository", "start|scope"),
        ("started_command_scope", "start|scope"),
        ("started_estimate", "start|scope"),
        ("helper_hash", "acquire"),
    ),
)
def test_request_ledger_and_argv_mutations_are_rejected(
    tmp_path: Path, mutation: str, match: str,
) -> None:
    manager, repository, authority = _resource_fixture(tmp_path)
    request_id = authority["request"]["request_id"]
    if mutation == "request_hash":
        authority["request"]["record"]["sha256"] = "F" * 64
    elif mutation == "request_id":
        authority["request"]["request_id"] = "3" * 32
    elif mutation in {"repository", "command"}:
        request_path = manager / "requests" / f"{request_id}.json"
        request = executor.strict_json(request_path)[1]
        request[mutation] = str(tmp_path / "other") if mutation == "repository" else request[mutation] + " --drift"
        request_path.write_bytes(executor.canonical_bytes(request))
        authority["request"]["record"] = {
            "bytes": request_path.stat().st_size,
            "sha256": _sha(request_path.read_bytes()),
        }
    elif mutation == "argv":
        authority["request"]["argv"][-1] = "DIFFERENT-OUTPUT"
    elif mutation == "prefix":
        authority["resource_manager"]["approved_ledger_prefix"]["sha256"] = "F" * 64
    elif mutation in {"approval_duplicate", "started_duplicate", "terminal", "cancelled", "row_order", "prefix_extra", "timestamp_order", "started_task", "started_repository", "started_command_scope", "started_estimate"}:
        raw = (manager / "ledger.md").read_bytes().decode("utf-8")
        approved = authority["resource_manager"]["approved_row"]
        started = authority["resource_manager"]["execution_started_row"]
        if mutation == "approval_duplicate":
            raw += _ledger_line(approved)
        elif mutation == "started_duplicate":
            raw += _ledger_line(started)
        elif mutation in {"terminal", "cancelled"}:
            row = list(started)
            row[2] = "COMPLETED_PASS" if mutation == "terminal" else "CANCELLED_NOT_RUN"
            raw += _ledger_line(row)
        elif mutation == "row_order":
            prefix = raw[: raw.index(_ledger_line(approved))]
            raw = prefix + _ledger_line(started) + _ledger_line(approved)
            prefix_raw = (prefix + _ledger_line(started)).encode("utf-8")
            authority["resource_manager"]["approved_ledger_prefix"] = {
                "bytes": len(prefix_raw), "sha256": _sha(prefix_raw),
            }
        elif mutation == "prefix_extra":
            unrelated = list(approved)
            unrelated[1] = "9" * 32
            unrelated[7] = "Unrelated approval"
            marker = _ledger_line(approved)
            raw = raw.replace(marker, marker + _ledger_line(unrelated), 1)
            prefix_raw = raw[: raw.index(_ledger_line(started))].encode("utf-8")
            authority["resource_manager"]["approved_ledger_prefix"] = {
                "bytes": len(prefix_raw), "sha256": _sha(prefix_raw),
            }
        elif mutation == "timestamp_order":
            changed = list(started)
            changed[0] = "2026-09-05T12:00:30Z"
            raw = raw.replace(_ledger_line(started), _ledger_line(changed), 1)
            authority["resource_manager"]["execution_started_row"] = changed
        else:
            changed = list(started)
            index = {
                "started_task": 3,
                "started_repository": 4,
                "started_command_scope": 5,
                "started_estimate": 6,
            }[mutation]
            changed[index] += " drift"
            raw = raw.replace(_ledger_line(started), _ledger_line(changed), 1)
            authority["resource_manager"]["execution_started_row"] = changed
        (manager / "ledger.md").write_bytes(raw.encode("utf-8"))
    elif mutation == "approval_binding":
        authority["resource_manager"]["approved_row"][7] += " drift"
    elif mutation == "start_binding":
        authority["resource_manager"]["execution_started_row"][7] += " drift"
    elif mutation == "helper_hash":
        authority["resource_manager"]["acquire"]["sha256"] = "F" * 64
    with pytest.raises(executor.ExecutionError, match=match):
        executor.validate_request_and_ledger(
            authority, repository=repository, resource_manager=manager,
        )


def test_malformed_unrelated_ledger_data_row_is_not_silently_ignored(tmp_path: Path) -> None:
    manager, repository, authority = _resource_fixture(tmp_path)
    ledger = manager / "ledger.md"
    ledger.write_bytes(ledger.read_bytes() + b"| a | b | c | d | e | f | g | h | unexpected |\n")
    with pytest.raises(executor.ExecutionError, match="ledger|column|row"):
        executor.validate_request_and_ledger(
            authority, repository=repository, resource_manager=manager,
        )


@pytest.mark.parametrize(
    "diagnostic,expected",
    (
        (executor.PASS, executor.PASS),
        (executor.NO_GO_REGULARITY, executor.NO_GO_REGULARITY),
        (executor.NO_GO_FRAME, executor.NO_GO_FRAME),
        (executor.NO_GO_COVARIANCE, executor.NO_GO_COVARIANCE),
    ),
)
def test_two_identical_complete_cycles_map_only_diagnostic_terminal(
    diagnostic: str, expected: str,
) -> None:
    cycle = _diagnostic_cycle(diagnostic)
    raw = executor.canonical_bytes(cycle)
    assert executor.adjudicate_cycles([cycle, copy.deepcopy(cycle)], [raw, raw]) == expected


def test_cycle_disagreement_malformed_or_process_failure_is_blocked() -> None:
    cycle = _diagnostic_cycle(executor.PASS)
    raw = executor.canonical_bytes(cycle)
    changed = copy.deepcopy(cycle)
    changed["diagnostic_checker_terminal"] = executor.NO_GO_FRAME
    mutations: list[tuple[list[dict[str, Any]], list[bytes]]] = [
        ([cycle], [raw]),
        ([cycle, changed], [raw, executor.canonical_bytes(changed)]),
        ([cycle, copy.deepcopy(cycle)], [raw, raw + b" "]),
    ]
    failed = copy.deepcopy(cycle)
    failed["status"] = "PROCESS_FAILURE"
    failed["process"]["producer"] = "TIMEOUT"
    mutations.append(([failed, copy.deepcopy(failed)], [executor.canonical_bytes(failed)] * 2))
    malformed = copy.deepcopy(cycle)
    malformed["counts"]["covered_obligations"] = 25
    mutations.append(([malformed, copy.deepcopy(malformed)], [executor.canonical_bytes(malformed)] * 2))
    for cycles, raws in mutations:
        assert executor.adjudicate_cycles(cycles, raws) == executor.BLOCKED_PROCESS


def test_cycle_terminal_precedence_does_not_promote_mixed_outcomes() -> None:
    ordered = [
        executor.BLOCKED_PROCESS,
        executor.NO_GO_REGULARITY,
        executor.NO_GO_FRAME,
        executor.NO_GO_COVARIANCE,
        executor.PASS,
    ]
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            left = _diagnostic_cycle(first)
            right = _diagnostic_cycle(second)
            result = executor.adjudicate_cycles(
                [left, right],
                [executor.canonical_bytes(left), executor.canonical_bytes(right)],
            )
            assert result == executor.BLOCKED_PROCESS


def test_cycle_record_binds_every_proof_checker_and_log_file(tmp_path: Path) -> None:
    cycle_dir = tmp_path / "cycle"
    cycle_dir.mkdir()
    proof_raw = b"synthetic proof\n"
    check_raw = b"synthetic independent checker\n"
    log_raw = b"synthetic log\n"
    (cycle_dir / "proof.json").write_bytes(proof_raw)
    (cycle_dir / "check-1.json").write_bytes(check_raw)
    (cycle_dir / "check-2.json").write_bytes(check_raw)
    for filename in (
        "producer.stderr.log", "producer.stdout.log", "checker-1.stderr.log",
        "checker-1.stdout.log", "checker-2.stderr.log", "checker-2.stdout.log",
    ):
        (cycle_dir / filename).write_bytes(log_raw)
    cycle = _diagnostic_cycle(executor.PASS)
    cycle["proof_sha256"] = _sha(proof_raw)
    cycle["check_sha256"] = [_sha(check_raw), _sha(check_raw)]
    cycle["process_log_sha256"] = {
        "checker_stderr": [_sha(log_raw), _sha(log_raw)],
        "checker_stdout": [_sha(log_raw), _sha(log_raw)],
        "producer_stderr": _sha(log_raw),
        "producer_stdout": _sha(log_raw),
    }
    raw = executor.canonical_bytes(cycle)
    executor._validate_cycle_record(cycle, raw, cycle_dir)

    for mutate in (
        "unknown_key", "identity", "terminal", "coverage", "replica_disagreement",
        "proof_hash", "log_hash",
    ):
        changed = copy.deepcopy(cycle)
        if mutate == "unknown_key":
            changed["unexpected"] = True
        elif mutate == "identity":
            changed["candidate_id"] += "_DRIFT"
        elif mutate == "terminal":
            changed["terminal"] = executor.PASS
        elif mutate == "coverage":
            changed["counts"]["stations"] -= 1
        elif mutate == "replica_disagreement":
            changed["check_sha256"][1] = "F" * 64
        elif mutate == "proof_hash":
            changed["proof_sha256"] = "F" * 64
        elif mutate == "log_hash":
            changed["process_log_sha256"]["producer_stdout"] = "F" * 64
        with pytest.raises(executor.ExecutionError):
            executor._validate_cycle_record(
                changed, executor.canonical_bytes(changed), cycle_dir,
            )


def test_build_result_is_deterministic_and_never_authorizes_activation(tmp_path: Path) -> None:
    _manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    cycle = _diagnostic_cycle(executor.PASS)
    raw = executor.canonical_bytes(cycle)
    protected = {"count": 7, "sha256": "B" * 64, "status": "PASS"}
    first = executor.build_result(
        validated, [cycle, copy.deepcopy(cycle)], [raw, raw], protected,
    )
    second = executor.build_result(
        validated, [copy.deepcopy(cycle), copy.deepcopy(cycle)], [raw, raw], copy.deepcopy(protected),
    )
    assert executor.canonical_bytes(first) == executor.canonical_bytes(second)
    assert first["terminal"] == executor.PASS
    assert first["checks"]["production_activation_authorized"] is False
    assert first["production_restriction"] == executor.RESTRICTION
    assert first["counts"] == {
        "cycles_complete": 2, "cycles_launched": 2, "registered_obligations": 26,
    }

    incomplete = executor.build_result(validated, [cycle], [raw], protected)
    assert incomplete["terminal"] == executor.BLOCKED_PROCESS
    assert incomplete["checks"]["byte_identical_canonical_cycles"] is False


def test_official_active_lock_owner_is_exact_and_does_not_trust_attempt_id(tmp_path: Path) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    owner_raw = _install_active_lock(manager, validated.request)
    active, observed = executor.validate_active_lock(
        validated.authority, validated.request, resource_manager=manager,
    )
    assert active == manager / "active-lock"
    assert observed == owner_raw
    assert executor.validate_active_lock_owner(
        manager,
        validated.authority["request"]["request_id"],
        validated.authority["request"]["attempt_id"],
    ) == owner_raw

    (active / "extra.txt").write_text("unregistered", encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="lock"):
        executor.validate_active_lock(
            validated.authority, validated.request, resource_manager=manager,
        )


@pytest.mark.parametrize("mutation", ("request", "repository", "command", "task", "pid", "duplicate"))
def test_active_lock_owner_mutations_are_rejected(tmp_path: Path, mutation: str) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    raw = _install_active_lock(manager, validated.request)
    owner_path = manager / "active-lock" / "owner.json"
    owner = json.loads(raw)
    if mutation == "request":
        owner["request_id"] = "9" * 32
    elif mutation == "repository":
        owner["repository"] += "-other"
    elif mutation == "command":
        owner["command"] += " --drift"
    elif mutation == "task":
        owner["task"] += " drift"
    elif mutation == "pid":
        owner["process_id"] = 0
    else:
        owner_path.write_bytes(b'{"request_id":"x","request_id":"y"}\n')
    if mutation != "duplicate":
        owner_path.write_bytes(executor.canonical_bytes(owner))
    with pytest.raises(executor.ExecutionError, match="owner|lock"):
        executor.validate_active_lock(
            validated.authority, validated.request, resource_manager=manager,
        )


def test_claim_is_exclusive_and_request_and_attempt_are_never_reused(tmp_path: Path) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    owner_raw = _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    assert claim.request_claim.read_bytes() == claim.claim_raw
    assert claim.attempt_claim.read_bytes() == claim.claim_raw
    claim_value = json.loads(claim.claim_raw)
    assert claim_value["active_lock_owner_sha256"] == _sha(owner_raw)
    assert not claim.receipt.exists()
    executor.release_claim(claim, validated, resource_manager=manager)
    assert claim.request_claim.exists() and claim.attempt_claim.exists()
    with pytest.raises((executor.ExecutionError, FileExistsError), match="claimed|consumed|exist"):
        executor.acquire_claim(validated, resource_manager=manager)


def test_claim_hash_or_lock_owner_change_is_fatal(tmp_path: Path) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    claim.attempt_claim.write_bytes(claim.claim_raw.replace(b'"schema"', b'"schemb"', 1))
    with pytest.raises(executor.ExecutionError, match="claim"):
        executor.release_claim(claim, validated, resource_manager=manager)


def test_publication_writes_receipt_and_terminal_before_atomic_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    result = _minimal_result(validated)
    original_append = executor._append_terminal_ledger
    observed: dict[str, bool] = {}

    def checked_append(path: Path, fields: list[str]) -> list[str]:
        pending, manifest, final_manifest = executor._pending_paths(
            output, validated.authority["request"]["attempt_id"],
        )
        observed.update(
            receipt=claim.receipt.is_file(),
            pending=pending.is_file(),
            manifest=manifest.is_file(),
            output_absent=not output.exists(),
            final_manifest_absent=not final_manifest.exists(),
        )
        return original_append(path, fields)

    monkeypatch.setattr(executor, "_append_terminal_ledger", checked_append)
    assert executor.publish_or_recover(
        validated, claim, result, output, resource_manager=manager,
    ) == result
    assert observed == {
        "receipt": True,
        "pending": True,
        "manifest": True,
        "output_absent": True,
        "final_manifest_absent": True,
    }
    assert output.read_bytes() == executor.canonical_bytes(result)
    assert output.with_name(output.name + ".manifest.json").is_file()
    request_rows = [
        row for row in executor._ledger_rows((manager / "ledger.md").read_bytes())
        if row[1] == claim.request_id
    ]
    assert [row[2] for row in request_rows] == [
        "APPROVED", "EXECUTION_STARTED", "COMPLETED_PASS",
    ]


def test_idempotent_receipt_recovery_promotes_without_rerunning(tmp_path: Path) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    result = _minimal_result(validated, executor.NO_GO_FRAME)
    executor.publish_or_recover(validated, claim, result, output, resource_manager=manager)
    pending, pending_manifest, final_manifest = executor._pending_paths(
        output, validated.authority["request"]["attempt_id"],
    )
    output.replace(pending)
    final_manifest.replace(pending_manifest)
    before_receipt = claim.receipt.read_bytes()
    recovered = executor.publish_or_recover(
        validated, None, None, output, resource_manager=manager,
    )
    assert recovered == result
    assert output.read_bytes() == executor.canonical_bytes(result)
    assert final_manifest.is_file() and not pending.exists() and not pending_manifest.exists()
    assert claim.receipt.read_bytes() == before_receipt
    request_rows = [
        row for row in executor._ledger_rows((manager / "ledger.md").read_bytes())
        if row[1] == claim.request_id
    ]
    assert [row[2] for row in request_rows] == [
        "APPROVED", "EXECUTION_STARTED", "COMPLETED_FAIL",
    ]


def test_pre_receipt_pending_only_is_consumed_and_never_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    original_write = executor._write_exclusive_bytes

    def stop_before_manifest(path: Path, raw: bytes) -> None:
        if path.name.endswith(".pending-manifest.json"):
            raise OSError("synthetic host interruption")
        original_write(path, raw)

    with monkeypatch.context() as context:
        context.setattr(executor, "_write_exclusive_bytes", stop_before_manifest)
        with pytest.raises(OSError):
            executor.publish_or_recover(
                validated, claim, _minimal_result(validated), output,
                resource_manager=manager,
            )
    pending, pending_manifest, _final_manifest = executor._pending_paths(
        output, validated.authority["request"]["attempt_id"],
    )
    before = pending.read_bytes()
    assert pending.is_file() and not pending_manifest.exists()
    with pytest.raises(executor.ExecutionError, match="consumed|unrecoverable"):
        executor.publish_or_recover(
            validated, None, None, output, resource_manager=manager,
        )
    assert pending.read_bytes() == before
    assert not output.exists() and not claim.receipt.exists() and not pending_manifest.exists()


@pytest.mark.parametrize("raw_cycles", ("complete", "rewritten"))
def test_pre_receipt_schema_valid_pass_cannot_promote_even_with_raw_cycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_cycles: str,
) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    original_write = executor._write_exclusive_bytes

    def stop_before_receipt(path: Path, raw: bytes) -> None:
        if path == claim.receipt:
            raise OSError("synthetic host interruption")
        original_write(path, raw)

    with monkeypatch.context() as context:
        context.setattr(executor, "_write_exclusive_bytes", stop_before_receipt)
        with pytest.raises(OSError):
            executor.publish_or_recover(
                validated, claim, _minimal_result(validated, executor.PASS), output,
                resource_manager=manager,
            )
    work_root = Path(validated.authority["locations"]["work_root"])
    work_root.mkdir()
    (work_root / "authority-manifest.json").write_bytes(
        executor.canonical_bytes({"synthetic": True})
    )
    for index in (1, 2):
        cycle_dir = work_root / f"cycle-{index}"
        cycle_dir.mkdir()
        cycle = _diagnostic_cycle(executor.PASS)
        if raw_cycles == "rewritten" and index == 2:
            cycle["diagnostic_checker_terminal"] = executor.NO_GO_FRAME
        (cycle_dir / "cycle.json").write_bytes(executor.canonical_bytes(cycle))
    pending, pending_manifest, final_manifest = executor._pending_paths(
        output, validated.authority["request"]["attempt_id"],
    )
    preserved = {
        path: path.read_bytes()
        for path in (claim.request_claim, claim.attempt_claim, pending, pending_manifest)
    }
    with pytest.raises(executor.ExecutionError, match="consumed|unrecoverable"):
        executor.publish_or_recover(
            validated, None, None, output, resource_manager=manager,
        )
    assert all(path.read_bytes() == raw for path, raw in preserved.items())
    assert not output.exists() and not final_manifest.exists() and not claim.receipt.exists()
    statuses = [
        row[2] for row in executor._ledger_rows((manager / "ledger.md").read_bytes())
        if row[1] == claim.request_id
    ]
    assert statuses == ["APPROVED", "EXECUTION_STARTED"]


def test_staged_artifacts_are_reread_before_receipt_or_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    original_write = executor._write_exclusive_bytes

    def corrupt_pending(path: Path, raw: bytes) -> None:
        original_write(path, raw)
        if path.name.endswith(".pending"):
            path.write_bytes(raw + b" ")

    monkeypatch.setattr(executor, "_write_exclusive_bytes", corrupt_pending)
    with pytest.raises(executor.ExecutionError, match="staged|canonical"):
        executor.publish_or_recover(
            validated, claim, _minimal_result(validated), output,
            resource_manager=manager,
        )
    assert not claim.receipt.exists() and not output.exists()
    statuses = [
        row[2] for row in executor._ledger_rows((manager / "ledger.md").read_bytes())
        if row[1] == claim.request_id
    ]
    assert statuses == ["APPROVED", "EXECUTION_STARTED"]


def test_lock_owner_is_revalidated_after_staging_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    _pending, pending_manifest, _final_manifest = executor._pending_paths(
        output, validated.authority["request"]["attempt_id"],
    )
    original_strict = executor.strict_json

    def replace_owner_after_staged_read(path: Path, label: str | None = None):
        made = original_strict(path, label)
        if path == pending_manifest:
            owner_path = manager / "active-lock" / "owner.json"
            owner = json.loads(owner_path.read_bytes())
            owner["process_id"] += 1
            owner_path.write_bytes(executor.canonical_bytes(owner))
        return made

    monkeypatch.setattr(executor, "strict_json", replace_owner_after_staged_read)
    with pytest.raises(executor.ExecutionError, match="owner|lock|claim"):
        executor.publish_or_recover(
            validated, claim, _minimal_result(validated), output,
            resource_manager=manager,
        )
    assert not claim.receipt.exists() and not output.exists()


def test_recovery_rejects_a_mismatched_existing_terminal_row(tmp_path: Path) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    result = _minimal_result(validated)
    executor.publish_or_recover(validated, claim, result, output, resource_manager=manager)
    ledger = manager / "ledger.md"
    rows = executor._ledger_rows(ledger.read_bytes())
    terminal = next(row for row in rows if row[1] == claim.request_id and row[2].startswith("COMPLETED_"))
    changed = list(terminal)
    changed[7] += " drift"
    raw = ledger.read_text(encoding="utf-8")
    ledger.write_text(raw.replace(_ledger_line(terminal), _ledger_line(changed), 1), encoding="utf-8", newline="")
    with pytest.raises(executor.ExecutionError, match="mismatched terminal"):
        executor.publish_or_recover(
            validated, None, None, output, resource_manager=manager,
        )


def test_claim_without_receipt_blocks_recovery_and_never_creates_canonical_output(tmp_path: Path) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    with pytest.raises(executor.ExecutionError, match="receipt|recover"):
        executor.publish_or_recover(
            validated, None, None, output, resource_manager=manager,
        )
    assert not output.exists()


def test_uncertain_process_result_cannot_be_published(tmp_path: Path) -> None:
    manager, _repository, _authority, validated = _validated_fixture(tmp_path)
    _install_active_lock(manager, validated.request)
    claim = executor.acquire_claim(validated, resource_manager=manager)
    output = (tmp_path / "external" / "canonical-result.json").absolute()
    result = _minimal_result(validated)
    result["checks"]["all_launched_processes_terminal"] = False
    with pytest.raises(executor.ExecutionError, match="process|published|uncertain"):
        executor.publish_or_recover(
            validated, claim, result, output, resource_manager=manager,
        )
    pending, manifest, final_manifest = executor._pending_paths(
        output, validated.authority["request"]["attempt_id"],
    )
    assert not any(path.exists() for path in (output, pending, manifest, final_manifest, claim.receipt))


def test_bounds_are_forwarded_without_relaxation_and_runner_loaded_after_checks() -> None:
    source = PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "execute" in functions
    segment = ast.get_source_segment(source, functions["execute"]) or ""
    assert "_load_runner" in segment
    assert segment.index("validate_authority") < segment.index("_load_runner")
    assert segment.index("validate_request_and_ledger") < segment.index("_load_runner")
    assert segment.index("validate_active_lock") < segment.index("_load_runner")
    assert "_cycle" in segment
    assert "run_gate" not in segment


def test_live_formal_argv_restores_isolated_flags_and_exact_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor.sys, "argv", [str(PATH), "--execute", "--output", "OUT"])
    made = executor.live_formal_argv()
    assert made[:4] == [
        str(Path(sys.executable).resolve()), "-I", "-B", str(PATH.resolve()),
    ]
    assert made[4:] == ["--execute", "--output", "OUT"]


def test_formal_execute_requires_official_manager_and_checks_common_grafts(tmp_path: Path) -> None:
    fake = tmp_path / "resource-manager"
    fake.mkdir()
    with pytest.raises(executor.ExecutionError, match="official resource-manager"):
        executor.require_official_resource_manager(fake)
    source = PATH.read_text(encoding="utf-8")
    assert 'common / "info/grafts"' in source
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    execute_source = ast.get_source_segment(
        source,
        functions["execute"],
    ) or ""
    assert "require_official_resource_manager" in execute_source
    cycle_segment = ast.get_source_segment(source, functions["run_scientific_cycles"]) or ""
    for key in (
        "child_timeout_seconds", "inactivity_seconds", "memory_limit_bytes",
        "wave_timeout_seconds",
    ):
        assert f'BOUNDS["{key}"]' in cycle_segment


def test_production_boundary_is_fail_closed_and_no_formal_mode_runs_on_import() -> None:
    source = PATH.read_text(encoding="utf-8")
    for token in (
        "default_activation_authorized", "distribution_publication_authorized",
        "ecosystem_exposure_authorized", "existing_defaults_unchanged",
        "qualified_q4_unchanged", "qualified_s3_v2d_unchanged",
    ):
        assert token in source
    assert "if __name__ == \"__main__\":" in source
    _load_executor("p4_scientific_executor_import_only")


def test_runner_remains_nonclassifying_and_executor_never_calls_its_adjudicator() -> None:
    runner_source = RUNNER.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    adjudicator = next(
        node for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_adjudicate_terminals"
    )
    segment = ast.get_source_segment(runner_source, adjudicator) or ""
    assert "return BLOCKED" in segment
    executor_source = PATH.read_text(encoding="utf-8")
    assert "._adjudicate_terminals(" not in executor_source
    assert ".run_gate(" not in executor_source
