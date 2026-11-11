from __future__ import annotations

import copy
import io
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_v2_stage4a_coordinator.py"
)


def _load():
    name = "_test_e4_pl_s3_v2_stage4a_coordinator"
    spec = importlib.util.spec_from_file_location(name, PROGRAM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _synthetic_predecessor_incident(module, tmp_path, monkeypatch):
    incident_root = (tmp_path / "cycle-1").resolve()
    incident_root.mkdir(parents=True)
    (incident_root / "candidate-source-tree").mkdir()
    resource_root = (tmp_path / "resource-manager").resolve()
    request_root = resource_root / "requests"
    request_root.mkdir(parents=True)
    repository = (tmp_path / "frozen-repository").resolve()
    repository.mkdir()
    request_id = "1" * 32
    candidate_commit, candidate_tree = "2" * 40, "3" * 40
    authorization_commit, authorization_tree = "4" * 40, "5" * 40
    authorization_parent = "6" * 40
    authorization_subject = "synthetic predecessor authorization"
    authorization_path = "docs/reference_cases/authorization.json"
    contract_path = "docs/reference_cases/contract.json"
    requested_at = "2026-08-31T09:06:41.6684913+02:00"
    task = "synthetic bounded Stage 4A"

    request = {
        "command": "",
        "estimate_minutes": 30,
        "repository": str(repository),
        "request_id": request_id,
        "requested_at": requested_at,
        "status": "PENDING",
        "task": task,
    }

    def historical_request_bytes(value):
        return (
            module.json.dumps(
                value, sort_keys=True, indent=2, ensure_ascii=True
            )
            + "\n"
        ).encode("ascii")

    request["command"] = "x" * (1311 - len(historical_request_bytes(request)))
    request_raw = historical_request_bytes(request)
    assert len(request_raw) == 1311
    assert request_raw != module.canonical_bytes(request)
    request_path = request_root / f"{request_id}.json"
    request_path.write_bytes(request_raw)

    approved_line = f"| t0 | {request_id} | APPROVED | synthetic |"
    started_line = f"| t1 | {request_id} | EXECUTION_STARTED | synthetic |"
    failed_line = f"| t2 | {request_id} | COMPLETED_FAIL | synthetic |"
    row_hashes = {
        status: module.sha256((line + "\n").encode())
        for status, line in (
            ("APPROVED", approved_line),
            ("EXECUTION_STARTED", started_line),
            ("COMPLETED_FAIL", failed_line),
        )
    }
    ledger_snapshot_raw = ("header\n" + approved_line + "\n").encode()
    live_ledger = resource_root / "ledger.md"
    live_ledger.write_text(
        "header\n" + "\n".join((approved_line, started_line, failed_line)) + "\n",
        encoding="utf-8",
    )

    archive_raw = b"synthetic exact archive"
    archive_name = "candidate-source.tar"
    archive_path = incident_root / archive_name
    archive_path.write_bytes(archive_raw)
    candidate_binding = {
        "artifact_path": str(archive_path),
        "artifact_sha256": module.sha256(archive_raw),
        "candidate_id": "CANDIDATE_E4_PL_S3_V2A_FLAT_LINEAR_V1",
        "commit": candidate_commit,
        "formulation_id": "E4_PL_QUALIFIED_S3_COMPANION_V2",
        "schema": "anysolver.e4-pl-s3-v2-flat-candidate-binding-v1",
        "selector": "e4-pl-s3-v2",
        "tree": candidate_tree,
    }
    phase_plan = {
        "advisory_review_triggers": {},
        "formal_thresholds": {},
        "manifest_sha256": "7" * 64,
        "phase": "4A",
        "prerequisites": [],
        "record_count": 81,
        "schema": "anysolver.e4-pl-s3-v2-flat-funnel-plan-v1",
        "scope": "full",
        "selector": "e4-pl-s3-v2",
        "shards": [
            {"assignment_id": assignment_id, "records": [{} for _ in range(27)]}
            for assignment_id in module.EXPECTED_SHARDS
        ],
    }
    phase_raw = module.canonical_bytes(phase_plan)
    wave_root = incident_root / "producer-wave"
    workers = []
    for assignment_id in module.EXPECTED_SHARDS:
        worker_root = wave_root / assignment_id
        workers.append(
            {
                "assignment_id": assignment_id,
                "plan_path": str(incident_root / "phase4a-plan.json"),
                "plan_sha256": module.sha256(phase_raw),
                "progress_path": str(worker_root / "progress.jsonl"),
                "scientific_path": str(worker_root / "scientific.json"),
                "stderr_path": str(worker_root / "stderr.log"),
                "stdout_path": str(worker_root / "stdout.log"),
            }
        )
    manifest = {
        "lane": "flat-proof",
        "output_root": str(wave_root),
        "schema": "anysolver.e4-pl-s3-v2-bounded-wave-manifest-v1",
        "wave_id": "S3_V2_FLAT_FUNNEL_4A_FULL",
        "workers": workers,
    }

    contract_sha = "8" * 64
    authorization = {
        "contract_path": contract_path,
        "contract_sha256": contract_sha,
        "execution_paths": {
            "aggregate_path": str(incident_root / "stage4a-aggregate.json"),
            "approval_snapshot_path": str(incident_root / "approval-snapshot.json"),
            "output_root": str(incident_root),
            "python_executable": sys.executable,
        },
        "formal_execution_authorized": True,
        "implementation_reviews": [],
        "ledger_approval": {},
        "resource_lock_required": True,
        "resource_request": {},
        "schema": module.AUTHORIZATION_SCHEMA,
        "user_approval": {
            "recorded": True,
            "source": f"synthetic approval for {request_id}",
        },
    }
    approval_snapshot = {
        "approved_row": {"line": approved_line, "sha256": row_hashes["APPROVED"]},
        "candidate": {"commit": candidate_commit, "tree": candidate_tree},
        "ledger": {},
        "request": {
            "byte_count": len(request_raw),
            "path": str(request_path),
            "request_id": request_id,
            "sha256": module.sha256(request_raw),
        },
        "schema": "anysolver.e4-pl-s3-v2-stage4a-approval-snapshot-v2",
    }
    transcript_raw = b"REGISTERED_COMMAND_EXIT=2\n"
    aggregate = {
        "advisory_review_required": False,
        "authorization_sha256": "0" * 64,
        "checker_replica_bindings": [],
        "classifying_record_count": 0,
        "contract_sha256": contract_sha,
        "formal_failures": ["FORMAL_PROCESS_FAILED"],
        "producer_wave_result_sha256": None,
        "production_restriction": module.PRODUCTION_RESTRICTION,
        "schema": module.HISTORICAL_AGGREGATE_SCHEMA,
        "sequence_results": [],
        "successor_expansion_authorized": False,
        "terminal": module.BLOCKED,
        "v1_diagnostic_record_count": 0,
    }
    coordinator_raw = (
        b'producer_result_path = (output_root / "producer-wave-result.json").resolve()'
    )
    bounded_raw = (
        b"result_path.relative_to(output_root)\n"
        b'raise BoundedProcessError("canonical wave result path escapes output_root")'
    )
    predecessor_contract = {
        "candidate": {"commit": candidate_commit, "tree": candidate_tree},
        "frozen_files": [
            {
                "git_blob_sha256": module.sha256(coordinator_raw),
                "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_coordinator.py",
                "role": "coordinator",
            },
            {
                "git_blob_sha256": module.sha256(bounded_raw),
                "path": "docs/reference_cases/e4_pl_s3_v2_bounded_process.py",
                "role": "bounded",
            },
        ],
        "schema": "anysolver.e4-pl-s3-v2-stage4a-contract-v2",
    }
    contract_raw = module.canonical_bytes(predecessor_contract)
    contract_sha = module.sha256(contract_raw)
    authorization["contract_sha256"] = contract_sha
    aggregate["contract_sha256"] = contract_sha

    artifacts = {
        "candidate_archive": (archive_name, archive_raw),
        "candidate_binding": (
            "candidate-source-binding.json",
            module.canonical_bytes(candidate_binding),
        ),
        "ledger_snapshot": ("resource-ledger-pre-run.md", ledger_snapshot_raw),
        "manifest": ("producer-wave-manifest.json", module.canonical_bytes(manifest)),
        "phase_plan": ("phase4a-plan.json", phase_raw),
        "transcript": ("formal-transcript.txt", transcript_raw),
    }
    artifact_constants = {
        name: (filename, len(raw), module.sha256(raw))
        for name, (filename, raw) in artifacts.items()
    }
    approval_snapshot["ledger"] = {
        "byte_count": len(ledger_snapshot_raw),
        "path": str(live_ledger),
        "sha256": module.sha256(ledger_snapshot_raw),
        "snapshot_path": str(incident_root / "resource-ledger-pre-run.md"),
    }
    artifacts["approval_snapshot"] = (
        "approval-snapshot.json",
        module.canonical_bytes(approval_snapshot),
    )
    authorization["ledger_approval"] = {
        "approved_row_sha256": row_hashes["APPROVED"],
        "ledger_path": str(live_ledger),
        "snapshot_path": str(incident_root / "approval-snapshot.json"),
        "snapshot_sha256": module.sha256(artifacts["approval_snapshot"][1]),
    }
    authorization["resource_request"] = {
        "command_sha256": module.sha256(request["command"].encode()),
        "repository": str(repository),
        "request_id": request_id,
        "request_path": str(request_path),
        "request_sha256": module.sha256(request_raw),
        "task": task,
    }
    authorization_raw = module.canonical_bytes(authorization)
    aggregate["authorization_sha256"] = module.sha256(authorization_raw)
    artifacts["aggregate"] = (
        "stage4a-aggregate.json",
        module.canonical_bytes(aggregate),
    )
    artifact_constants = {
        name: (filename, len(raw), module.sha256(raw))
        for name, (filename, raw) in artifacts.items()
    }
    for filename, raw in artifacts.values():
        (incident_root / filename).write_bytes(raw)

    incident = {
        name: {
            "byte_count": count,
            "path": str(incident_root / filename),
            "sha256": digest,
        }
        for name, (filename, count, digest) in artifact_constants.items()
    }
    incident.update(
        {
            "authorization": {
                "byte_count": len(authorization_raw),
                "commit": authorization_commit,
                "parent": authorization_parent,
                "path": authorization_path,
                "sha256": module.sha256(authorization_raw),
                "subject": authorization_subject,
                "tree": authorization_tree,
            },
            "output_root": str(incident_root),
            "request": {
                "byte_count": len(request_raw),
                "path": str(request_path),
                "sha256": module.sha256(request_raw),
            },
            "root_cause": "PRODUCER_RESULT_PATH_OUTSIDE_REGISTERED_WAVE_ROOT",
            "scientific_execution": {
                "checker_processes_started": 0,
                "classifying_records": 0,
                "producer_processes_started": 0,
                "producer_result_present": False,
            },
            "terminal_ledger_rows": [
                {
                    "line": started_line,
                    "sha256": row_hashes["EXECUTION_STARTED"],
                    "status": "EXECUTION_STARTED",
                },
                {
                    "line": failed_line,
                    "sha256": row_hashes["COMPLETED_FAIL"],
                    "status": "COMPLETED_FAIL",
                },
            ],
        }
    )

    monkeypatch.setattr(module, "PREDECESSOR_INCIDENT_ROOT", incident_root)
    monkeypatch.setattr(module, "PREDECESSOR_REPOSITORY", str(repository))
    monkeypatch.setattr(module, "PREDECESSOR_REQUEST_ID", request_id)
    monkeypatch.setattr(module, "PREDECESSOR_REQUESTED_AT", requested_at)
    monkeypatch.setattr(module, "PREDECESSOR_TASK", task)
    monkeypatch.setattr(module, "PREDECESSOR_REQUEST_SHA256", module.sha256(request_raw))
    monkeypatch.setattr(
        module, "PREDECESSOR_COMMAND_SHA256", module.sha256(request["command"].encode())
    )
    monkeypatch.setattr(module, "PREDECESSOR_AUTHORIZATION_COMMIT", authorization_commit)
    monkeypatch.setattr(module, "PREDECESSOR_AUTHORIZATION_TREE", authorization_tree)
    monkeypatch.setattr(module, "PREDECESSOR_AUTHORIZATION_PARENT", authorization_parent)
    monkeypatch.setattr(module, "PREDECESSOR_AUTHORIZATION_SUBJECT", authorization_subject)
    monkeypatch.setattr(module, "PREDECESSOR_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(module, "PREDECESSOR_AUTHORIZATION_SHA256", module.sha256(authorization_raw))
    monkeypatch.setattr(module, "PREDECESSOR_AUTHORIZATION_BYTE_COUNT", len(authorization_raw))
    monkeypatch.setattr(module, "PREDECESSOR_CONTRACT_PATH", contract_path)
    monkeypatch.setattr(module, "PREDECESSOR_CONTRACT_SHA256", contract_sha)
    monkeypatch.setattr(module, "PREDECESSOR_CONTRACT_BYTE_COUNT", len(contract_raw))
    monkeypatch.setattr(
        module,
        "PREDECESSOR_CONNECTIVITY_MANIFEST_SHA256",
        phase_plan["manifest_sha256"],
    )
    monkeypatch.setattr(module, "PREDECESSOR_CANDIDATE_COMMIT", candidate_commit)
    monkeypatch.setattr(module, "PREDECESSOR_CANDIDATE_TREE", candidate_tree)
    monkeypatch.setattr(module, "PREDECESSOR_COORDINATOR_SHA256", module.sha256(coordinator_raw))
    monkeypatch.setattr(module, "PREDECESSOR_BOUNDED_SHA256", module.sha256(bounded_raw))
    monkeypatch.setattr(module, "PREDECESSOR_LEDGER_ROW_SHA256", row_hashes)
    monkeypatch.setattr(module, "PREDECESSOR_ARTIFACTS", artifact_constants)
    monkeypatch.setattr(module, "RESOURCE_MANAGER_ROOT", resource_root)
    monkeypatch.setattr(module, "RESOURCE_LEDGER_PATH", live_ledger)
    monkeypatch.setattr(module, "_validate_git_object_authority", lambda: None)

    git_blobs = {
        f"{authorization_commit}:{authorization_path}": authorization_raw,
        f"{authorization_commit}:{contract_path}": contract_raw,
        f"{authorization_commit}:docs/reference_cases/e4_pl_s3_v2_stage4a_coordinator.py": coordinator_raw,
        f"{authorization_commit}:docs/reference_cases/e4_pl_s3_v2_bounded_process.py": bounded_raw,
    }

    def fake_git(*args, binary=False, **_kwargs):
        if args == ("rev-parse", authorization_commit):
            return authorization_commit
        if args == ("rev-parse", f"{authorization_commit}^{{tree}}"):
            return authorization_tree
        if args == ("show", "-s", "--format=%P", authorization_commit):
            return authorization_parent
        if args == ("show", "-s", "--format=%s", authorization_commit):
            return authorization_subject
        if args[0] == "show" and len(args) == 2 and args[1] in git_blobs:
            return git_blobs[args[1]] if binary else git_blobs[args[1]].decode()
        raise AssertionError(f"unexpected synthetic Git query: {args}")

    monkeypatch.setattr(module, "_git", fake_git)

    class SyntheticWorker:
        def __init__(self, assignment_id):
            self.assignment_id = assignment_id

    class SyntheticBounded:
        def validate_manifest(self, value):
            bounded_calls.append(copy.deepcopy(value))
            return (
                value["wave_id"],
                value["lane"],
                Path(value["output_root"]).resolve(),
                tuple(SyntheticWorker(item["assignment_id"]) for item in value["workers"]),
            )

    bounded_calls = []
    bounded = SyntheticBounded()
    real_load_module = module._load_module

    def fake_load_module(name, path):
        if path == module.BOUNDED_PATH:
            return bounded
        return real_load_module(name, path)

    monkeypatch.setattr(module, "_load_module", fake_load_module)

    def replace_artifact(name, value):
        raw = value if isinstance(value, bytes) else module.canonical_bytes(value)
        filename = artifacts[name][0]
        (incident_root / filename).write_bytes(raw)
        constants = dict(module.PREDECESSOR_ARTIFACTS)
        constants[name] = (filename, len(raw), module.sha256(raw))
        monkeypatch.setattr(module, "PREDECESSOR_ARTIFACTS", constants)
        incident[name] = {
            "byte_count": len(raw),
            "path": str(incident_root / filename),
            "sha256": module.sha256(raw),
        }

    def replace_request(value, *, rebind_command):
        value = copy.deepcopy(value)
        value["command"] = str(value["command"])
        raw = historical_request_bytes(value)
        difference = 1311 - len(raw)
        assert difference >= 0
        value["command"] += "x" * difference
        raw = historical_request_bytes(value)
        assert len(raw) == 1311
        request_path.write_bytes(raw)
        monkeypatch.setattr(module, "PREDECESSOR_REQUEST_SHA256", module.sha256(raw))
        if rebind_command:
            monkeypatch.setattr(
                module,
                "PREDECESSOR_COMMAND_SHA256",
                module.sha256(value["command"].encode()),
            )
        incident["request"] = {
            "byte_count": len(raw),
            "path": str(request_path),
            "sha256": module.sha256(raw),
        }

    return {
        "aggregate": aggregate,
        "approval_snapshot": approval_snapshot,
        "bounded_calls": bounded_calls,
        "candidate_binding": candidate_binding,
        "incident": incident,
        "live_ledger": live_ledger,
        "manifest": manifest,
        "phase_plan": phase_plan,
        "replace_artifact": replace_artifact,
        "replace_request": replace_request,
        "request": request,
        "request_path": request_path,
    }


def _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch):
    incident_root = (tmp_path / "resource-deferred-cycle").resolve()
    candidate_tree_root = incident_root / "candidate-source-tree"
    candidate_directory = candidate_tree_root / "docs"
    candidate_directory.mkdir(parents=True)
    (candidate_directory / "one.txt").write_bytes(b"one\n")
    wave_root = incident_root / "producer-wave"
    wave_root.mkdir()

    resource_root = (tmp_path / "resource-manager").resolve()
    request_root = resource_root / "requests"
    attempt_root = resource_root / "attempts"
    request_root.mkdir(parents=True)
    attempt_root.mkdir()
    repository = (tmp_path / "frozen-repository").resolve()
    repository.mkdir()

    request_id = "a" * 32
    candidate_commit, candidate_tree = "b" * 40, "c" * 40
    contract_commit, contract_tree, contract_parent = "d" * 40, "e" * 40, "f" * 40
    authorization_commit = "1" * 40
    authorization_tree, authorization_parent = "2" * 40, "3" * 40
    contract_subject = "synthetic resource-deferred contract"
    authorization_subject = "synthetic resource-deferred authorization"
    requested_at = "2026-08-31T09:58:56.9567735+02:00"
    task = "synthetic bounded Stage 4A"
    connectivity_sha = "4" * 64
    coordinator_raw = b"synthetic corrected coordinator"
    bounded_raw = b"synthetic three-worker bounded runner"

    request = {
        "command": "registered resource command",
        "estimate_minutes": 30,
        "repository": str(repository),
        "request_id": request_id,
        "requested_at": requested_at,
        "status": "PENDING",
        "task": task,
    }
    request_raw = module.canonical_bytes(request)
    request_path = request_root / f"{request_id}.json"
    request_path.write_bytes(request_raw)

    approved_line = f"| t0 | {request_id} | APPROVED | synthetic |"
    started_line = f"| t1 | {request_id} | EXECUTION_STARTED | synthetic |"
    failed_line = f"| t2 | {request_id} | COMPLETED_FAIL | synthetic |"
    row_hashes = {
        status: module.sha256((line + "\n").encode())
        for status, line in (
            ("APPROVED", approved_line),
            ("EXECUTION_STARTED", started_line),
            ("COMPLETED_FAIL", failed_line),
        )
    }
    ledger_snapshot_raw = ("header\n" + approved_line + "\n").encode()
    live_ledger = resource_root / "ledger.md"
    live_ledger.write_text(
        "header\n" + "\n".join((approved_line, started_line, failed_line)) + "\n",
        encoding="utf-8",
    )

    archive_path = incident_root / "candidate-source.tar"
    candidate_payload = b"one\n"
    with tarfile.open(archive_path, "w") as bundle:
        directory = tarfile.TarInfo("docs/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.mtime = 0
        bundle.addfile(directory)
        member = tarfile.TarInfo("docs/one.txt")
        member.mode = 0o644
        member.mtime = 0
        member.size = len(candidate_payload)
        bundle.addfile(member, io.BytesIO(candidate_payload))
    archive_raw = archive_path.read_bytes()
    candidate_binding = {
        "artifact_path": str(archive_path),
        "artifact_sha256": module.sha256(archive_raw),
        "candidate_id": "CANDIDATE_E4_PL_S3_V2A_FLAT_LINEAR_V1",
        "commit": candidate_commit,
        "formulation_id": "E4_PL_QUALIFIED_S3_COMPANION_V2",
        "schema": "anysolver.e4-pl-s3-v2-flat-candidate-binding-v1",
        "selector": "e4-pl-s3-v2",
        "tree": candidate_tree,
    }
    candidate_binding_raw = module.canonical_bytes(candidate_binding)

    plan = {
        "advisory_review_triggers": {},
        "formal_thresholds": {},
        "manifest_sha256": connectivity_sha,
        "phase": "4A",
        "prerequisites": [],
        "record_count": 81,
        "schema": "anysolver.e4-pl-s3-v2-flat-funnel-plan-v1",
        "scope": "full",
        "selector": "e4-pl-s3-v2",
        "shards": [
            {"assignment_id": assignment_id, "records": [{} for _ in range(27)]}
            for assignment_id in module.EXPECTED_SHARDS
        ],
    }
    plan_raw = module.canonical_bytes(plan)

    contract = {
        "candidate": {"commit": candidate_commit, "tree": candidate_tree},
        "execution": {
            "maximum_memory_gib_per_process_tree": 24,
            "maximum_workers": 3,
        },
        "frozen_files": [
            {
                "git_blob_sha256": module.sha256(bounded_raw),
                "path": "docs/reference_cases/e4_pl_s3_v2_bounded_process.py",
                "role": "bounded",
            },
            {
                "git_blob_sha256": module.sha256(coordinator_raw),
                "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_coordinator.py",
                "role": "coordinator",
            },
        ],
        "schema": "anysolver.e4-pl-s3-v2-stage4a-contract-v3",
    }
    contract_raw = module.canonical_bytes(contract)
    contract_sha = module.sha256(contract_raw)

    attempt = {
        "contract_sha256": contract_sha,
        "request_id": request_id,
        "schema": "anysolver.resource-attempt-claim-v1",
    }
    attempt_raw = module.canonical_bytes(attempt)
    attempt_path = attempt_root / f"{request_id}.json"
    attempt_path.write_bytes(attempt_raw)

    approval_snapshot = {
        "approved_row": {"line": approved_line, "sha256": row_hashes["APPROVED"]},
        "candidate": {"commit": candidate_commit, "tree": candidate_tree},
        "ledger": {
            "byte_count": len(ledger_snapshot_raw),
            "path": str(live_ledger),
            "sha256": module.sha256(ledger_snapshot_raw),
            "snapshot_path": str(incident_root / "resource-ledger-pre-run.md"),
        },
        "request": {
            "byte_count": len(request_raw),
            "path": str(request_path),
            "request_id": request_id,
            "sha256": module.sha256(request_raw),
        },
        "schema": "anysolver.e4-pl-s3-v2-stage4a-approval-snapshot-v2",
    }
    approval_raw = module.canonical_bytes(approval_snapshot)
    authorization = {
        "contract_path": module.PREDECESSOR_CONTRACT_PATH,
        "contract_sha256": contract_sha,
        "execution_paths": {
            "aggregate_path": str(incident_root / "stage4a-aggregate.json"),
            "approval_snapshot_path": str(incident_root / "approval-snapshot.json"),
            "output_root": str(incident_root),
            "python_executable": sys.executable,
        },
        "formal_execution_authorized": True,
        "implementation_reviews": [
            {
                "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_process_implementation_review.json",
                "role": "PROCESS_AND_AUTHORITY",
                "sha256": "7B7CF54AD998E31B11B2F4286C3BE638126817D28D7290F90B15BA1AAB0109E3",
                "verdict": "ACCEPT_STAGE4A_PROCESS_IMPLEMENTATION_NO_P0_P1",
            },
            {
                "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_scientific_implementation_review.json",
                "role": "SCIENTIFIC_AND_MECHANICS",
                "sha256": "22EA28DAC7719F8748389204860AA4B90E936EE96AE564F414801D539D84A797",
                "verdict": "ACCEPT_STAGE4A_SCIENTIFIC_IMPLEMENTATION_NO_P0_P1",
            },
        ],
        "ledger_approval": {
            "approved_row_sha256": row_hashes["APPROVED"],
            "ledger_path": str(live_ledger),
            "snapshot_path": str(incident_root / "approval-snapshot.json"),
            "snapshot_sha256": module.sha256(approval_raw),
        },
        "resource_lock_required": True,
        "resource_request": {
            "command_sha256": module.sha256(request["command"].encode()),
            "repository": str(repository),
            "request_id": request_id,
            "request_path": str(request_path),
            "request_sha256": module.sha256(request_raw),
            "task": task,
        },
        "schema": module.AUTHORIZATION_SCHEMA,
        "user_approval": {
            "recorded": True,
            "source": f"synthetic approval for {request_id}",
        },
    }
    authorization_raw = module.canonical_bytes(authorization)

    workers = []
    for assignment_id in module.EXPECTED_SHARDS:
        assignment_root = wave_root / assignment_id
        workers.append(
            {
                "assignment_id": assignment_id,
                "input_hashes": [
                    {"path": str(module.AUTHORITY_PATH), "sha256": "5" * 64},
                    {
                        "path": str(
                            module.REFERENCE_CASES
                            / "e4_pl_s3_v2_stage4a_contract.json"
                        ),
                        "sha256": contract_sha,
                    },
                    {
                        "path": str(
                            module.REFERENCE_CASES
                            / "e4_pl_s3_v2_stage4a_execution_authorization.json"
                        ),
                        "sha256": module.sha256(authorization_raw),
                    },
                    {
                        "path": str(incident_root / "candidate-source-binding.json"),
                        "sha256": module.sha256(candidate_binding_raw),
                    },
                ],
                "plan_path": str(incident_root / "phase4a-plan.json"),
                "plan_sha256": module.sha256(plan_raw),
                "progress_path": str(assignment_root / "progress.jsonl"),
                "scientific_path": str(assignment_root / "scientific.json"),
                "stderr_path": str(assignment_root / "stderr.log"),
                "stdout_path": str(assignment_root / "stdout.log"),
            }
        )
    manifest = {
        "lane": "flat-proof",
        "output_root": str(wave_root),
        "schema": "anysolver.e4-pl-s3-v2-bounded-wave-manifest-v1",
        "wave_id": "S3_V2_FLAT_FUNNEL_4A_FULL",
        "workers": workers,
    }
    manifest_raw = module.canonical_bytes(manifest)
    producer_result = {
        "lane": "flat-proof",
        "manifest_sha256": module.sha256(manifest_raw),
        "schema": module.PRODUCER_RESULT_SCHEMA,
        "terminal": "RESOURCE_DEFERRED",
        "wave_id": "S3_V2_FLAT_FUNNEL_4A_FULL",
        "workers": [],
    }
    producer_raw = module.canonical_bytes(producer_result)
    producer_path = wave_root / "producer-wave-result.json"
    producer_path.write_bytes(producer_raw)
    aggregate = {
        "advisory_review_required": False,
        "authorization_sha256": module.sha256(authorization_raw),
        "checker_replica_bindings": [],
        "classifying_record_count": 0,
        "contract_sha256": contract_sha,
        "formal_failures": ["PRODUCER_WAVE_NOT_COMPLETED"],
        "producer_wave_result_sha256": module.sha256(producer_raw),
        "production_restriction": module.PRODUCTION_RESTRICTION,
        "schema": module.HISTORICAL_AGGREGATE_SCHEMA,
        "sequence_results": [],
        "successor_expansion_authorized": False,
        "terminal": module.BLOCKED,
        "v1_diagnostic_record_count": 0,
    }
    transcript_raw = b"REGISTERED_COMMAND_EXIT=2\n"
    artifacts = {
        "aggregate": ("stage4a-aggregate.json", module.canonical_bytes(aggregate)),
        "approval_snapshot": ("approval-snapshot.json", approval_raw),
        "candidate_archive": ("candidate-source.tar", archive_raw),
        "candidate_binding": (
            "candidate-source-binding.json",
            candidate_binding_raw,
        ),
        "ledger_snapshot": ("resource-ledger-pre-run.md", ledger_snapshot_raw),
        "manifest": ("producer-wave-manifest.json", manifest_raw),
        "phase_plan": ("phase4a-plan.json", plan_raw),
        "transcript": ("formal-transcript.txt", transcript_raw),
    }
    artifact_constants = {
        name: (filename, len(raw), module.sha256(raw))
        for name, (filename, raw) in artifacts.items()
    }
    for filename, raw in artifacts.values():
        (incident_root / filename).write_bytes(raw)

    incident = {
        name: {
            "byte_count": count,
            "path": str(incident_root / filename),
            "sha256": digest,
        }
        for name, (filename, count, digest) in artifact_constants.items()
    }
    incident.update(
        {
            "archive_ref": {
                "commit": authorization_commit,
                "ref": "refs/archive/synthetic-resource-deferred",
            },
            "attempt_claim": {
                "byte_count": len(attempt_raw),
                "path": str(attempt_path),
                "sha256": module.sha256(attempt_raw),
            },
            "authorization": {
                "byte_count": len(authorization_raw),
                "commit": authorization_commit,
                "parent": authorization_parent,
                "path": module.PREDECESSOR_AUTHORIZATION_PATH,
                "sha256": module.sha256(authorization_raw),
                "subject": authorization_subject,
                "tree": authorization_tree,
            },
            "candidate_tree": {
                "directory_count": 1,
                "file_count": 1,
                "path": str(candidate_tree_root),
            },
            "contract": {
                "byte_count": len(contract_raw),
                "commit": contract_commit,
                "parent": contract_parent,
                "path": module.PREDECESSOR_CONTRACT_PATH,
                "sha256": contract_sha,
                "subject": contract_subject,
                "tree": contract_tree,
            },
            "memory_admission": {
                "concurrent_workers_assumed": 3,
                "maximum_memory_gib_per_process_tree": 24,
                "observed_at_event_available_bytes": None,
                "observation_status": "NOT_RECORDED",
                "os_headroom_gib": 16,
                "registered_workers": 3,
                "required_bytes": 88 * (1 << 30),
            },
            "output_root": str(incident_root),
            "producer_result": {
                "byte_count": len(producer_raw),
                "path": str(producer_path),
                "sha256": module.sha256(producer_raw),
            },
            "request": {
                "byte_count": len(request_raw),
                "path": str(request_path),
                "sha256": module.sha256(request_raw),
            },
            "request_reuse_forbidden": True,
            "root_cause": "RESOURCE_ADMISSION_DEFERRED_BEFORE_WORKER_LAUNCH",
            "scientific_execution": {
                "checker_processes_started": 0,
                "classifying_records": 0,
                "producer_processes_started": 0,
                "producer_result_present": True,
            },
            "terminal_ledger_rows": [
                {
                    "line": started_line,
                    "sha256": row_hashes["EXECUTION_STARTED"],
                    "status": "EXECUTION_STARTED",
                },
                {
                    "line": failed_line,
                    "sha256": row_hashes["COMPLETED_FAIL"],
                    "status": "COMPLETED_FAIL",
                },
            ],
        }
    )

    patches = {
        "RESOURCE_DEFERRED_INCIDENT_ROOT": incident_root,
        "RESOURCE_DEFERRED_REPOSITORY": str(repository),
        "RESOURCE_DEFERRED_REQUEST_ID": request_id,
        "RESOURCE_DEFERRED_REQUESTED_AT": requested_at,
        "RESOURCE_DEFERRED_TASK": task,
        "RESOURCE_DEFERRED_REQUEST_BYTE_COUNT": len(request_raw),
        "RESOURCE_DEFERRED_REQUEST_SHA256": module.sha256(request_raw),
        "RESOURCE_DEFERRED_COMMAND_SHA256": module.sha256(request["command"].encode()),
        "RESOURCE_DEFERRED_ATTEMPT_BYTE_COUNT": len(attempt_raw),
        "RESOURCE_DEFERRED_ATTEMPT_SHA256": module.sha256(attempt_raw),
        "RESOURCE_DEFERRED_AUTHORIZATION_COMMIT": authorization_commit,
        "RESOURCE_DEFERRED_AUTHORIZATION_TREE": authorization_tree,
        "RESOURCE_DEFERRED_AUTHORIZATION_PARENT": authorization_parent,
        "RESOURCE_DEFERRED_AUTHORIZATION_SUBJECT": authorization_subject,
        "RESOURCE_DEFERRED_AUTHORIZATION_BYTE_COUNT": len(authorization_raw),
        "RESOURCE_DEFERRED_AUTHORIZATION_SHA256": module.sha256(authorization_raw),
        "RESOURCE_DEFERRED_CONTRACT_COMMIT": contract_commit,
        "RESOURCE_DEFERRED_CONTRACT_TREE": contract_tree,
        "RESOURCE_DEFERRED_CONTRACT_PARENT": contract_parent,
        "RESOURCE_DEFERRED_CONTRACT_SUBJECT": contract_subject,
        "RESOURCE_DEFERRED_CONTRACT_BYTE_COUNT": len(contract_raw),
        "RESOURCE_DEFERRED_CONTRACT_SHA256": contract_sha,
        "RESOURCE_DEFERRED_CANDIDATE_COMMIT": candidate_commit,
        "RESOURCE_DEFERRED_CANDIDATE_TREE": candidate_tree,
        "RESOURCE_DEFERRED_AUTHORITY_SHA256": "5" * 64,
        "RESOURCE_DEFERRED_COORDINATOR_SHA256": module.sha256(coordinator_raw),
        "RESOURCE_DEFERRED_BOUNDED_SHA256": module.sha256(bounded_raw),
        "RESOURCE_DEFERRED_CONNECTIVITY_MANIFEST_SHA256": connectivity_sha,
        "RESOURCE_DEFERRED_ARCHIVE_REF": "refs/archive/synthetic-resource-deferred",
        "RESOURCE_DEFERRED_ARCHIVE_COMMIT": authorization_commit,
        "RESOURCE_DEFERRED_CANDIDATE_FILE_COUNT": 1,
        "RESOURCE_DEFERRED_CANDIDATE_DIRECTORY_COUNT": 1,
        "RESOURCE_DEFERRED_OLD_ADMISSION_REQUIRED_BYTES": 88 * (1 << 30),
        "RESOURCE_DEFERRED_LEDGER_ROW_SHA256": row_hashes,
        "RESOURCE_DEFERRED_ARTIFACTS": artifact_constants,
        "RESOURCE_DEFERRED_PRODUCER_RESULT": (
            "producer-wave/producer-wave-result.json",
            len(producer_raw),
            module.sha256(producer_raw),
        ),
        "RESOURCE_MANAGER_ROOT": resource_root,
        "RESOURCE_LEDGER_PATH": live_ledger,
    }
    for name, made in patches.items():
        monkeypatch.setattr(module, name, made)
    monkeypatch.setattr(module, "_validate_git_object_authority", lambda: None)

    git_blobs = {
        f"{authorization_commit}:{module.PREDECESSOR_AUTHORIZATION_PATH}": authorization_raw,
        f"{authorization_commit}:{module.PREDECESSOR_CONTRACT_PATH}": contract_raw,
        f"{contract_commit}:{module.PREDECESSOR_CONTRACT_PATH}": contract_raw,
        f"{authorization_commit}:docs/reference_cases/e4_pl_s3_v2_stage4a_coordinator.py": coordinator_raw,
        f"{authorization_commit}:docs/reference_cases/e4_pl_s3_v2_bounded_process.py": bounded_raw,
    }
    identities = {
        authorization_commit: (
            authorization_tree,
            authorization_parent,
            authorization_subject,
        ),
        contract_commit: (contract_tree, contract_parent, contract_subject),
    }

    def fake_git(*args, binary=False, **_kwargs):
        if args == ("rev-parse", "refs/archive/synthetic-resource-deferred"):
            return authorization_commit
        for commit, (tree, parent, subject) in identities.items():
            if args == ("rev-parse", commit):
                return commit
            if args == ("rev-parse", f"{commit}^{{tree}}"):
                return tree
            if args == ("show", "-s", "--format=%P", commit):
                return parent
            if args == ("show", "-s", "--format=%s", commit):
                return subject
        if args[0] == "show" and len(args) == 2 and args[1] in git_blobs:
            return git_blobs[args[1]] if binary else git_blobs[args[1]].decode()
        raise AssertionError(f"unexpected synthetic Git query: {args}")

    monkeypatch.setattr(module, "_git", fake_git)

    class SyntheticWorker:
        def __init__(self, assignment_id):
            self.assignment_id = assignment_id

    class SyntheticBounded:
        def validate_manifest(self, value):
            return (
                value["wave_id"],
                value["lane"],
                Path(value["output_root"]).resolve(),
                tuple(SyntheticWorker(item["assignment_id"]) for item in value["workers"]),
            )

    bounded = SyntheticBounded()
    real_load_module = module._load_module

    def fake_load_module(name, path):
        if path == module.BOUNDED_PATH:
            return bounded
        return real_load_module(name, path)

    monkeypatch.setattr(module, "_load_module", fake_load_module)

    def replace_artifact(name, value):
        raw = value if isinstance(value, bytes) else module.canonical_bytes(value)
        filename = artifacts[name][0]
        (incident_root / filename).write_bytes(raw)
        constants = dict(module.RESOURCE_DEFERRED_ARTIFACTS)
        constants[name] = (filename, len(raw), module.sha256(raw))
        monkeypatch.setattr(module, "RESOURCE_DEFERRED_ARTIFACTS", constants)
        incident[name] = {
            "byte_count": len(raw),
            "path": str(incident_root / filename),
            "sha256": module.sha256(raw),
        }

    def replace_producer(value):
        raw = module.canonical_bytes(value)
        producer_path.write_bytes(raw)
        monkeypatch.setattr(
            module,
            "RESOURCE_DEFERRED_PRODUCER_RESULT",
            ("producer-wave/producer-wave-result.json", len(raw), module.sha256(raw)),
        )
        incident["producer_result"] = {
            "byte_count": len(raw),
            "path": str(producer_path),
            "sha256": module.sha256(raw),
        }

    return {
        "aggregate": aggregate,
        "candidate_tree_root": candidate_tree_root,
        "incident": incident,
        "live_ledger": live_ledger,
        "producer_result": producer_result,
        "replace_artifact": replace_artifact,
        "replace_producer": replace_producer,
    }


def _sequence(mask: str, fraction: int, *, advisory: bool = False, failure: str | None = None):
    failures = [] if failure is None else [failure]
    return {
        "advisory_triggered": advisory,
        "all_q4_response_slope": 2.1,
        "energy_norm_slope": 1.2,
        "energy_norm_slope_lower_95_percent": 1.0,
        "energy_norm_values": [0.4, 0.2, 0.1],
        "failed_subgates": failures,
        "finest_error_ratio_to_all_q4": 1.1,
        "fraction_percent": fraction,
        "mask": mask,
        "record_ids": [
            f"{mask}-{fraction}-N20",
            f"{mask}-{fraction}-N40",
            f"{mask}-{fraction}-N80",
        ],
        "response_error_slope": 2.0,
        "response_errors": [0.04, 0.01, 0.0025],
        "slope_deficit_from_all_q4": 0.1,
        "successive_refinement_passed": failure != "SUCCESSIVE_RESPONSE_ERROR",
    }


def _checker_value(
    module,
    assignment_id: str,
    *,
    advisory: bool = False,
    no_go: bool = False,
):
    diagonal = module.EXPECTED_SHARDS[assignment_id]
    sequences = [
        _sequence(
            mask,
            fraction,
            advisory=advisory and mask == "dispersed" and fraction == 1,
            failure=(
                "RESPONSE_SLOPE"
                if no_go and mask == "dispersed" and fraction == 1
                else None
            ),
        )
        for mask in module.MASK_ORDER
        for fraction in module.FRACTION_ORDER
    ]
    failures = ["dispersed:1:RESPONSE_SLOPE"] if no_go else []
    return {
        "advisory_review_required": bool(advisory and not no_go),
        "assignment_id": assignment_id,
        "assignment_sha256": "A" * 64,
        "classifying_record_count": 27,
        "diagonal": diagonal,
        "formal_failures": failures,
        "plan_sha256": "B" * 64,
        "production_restriction": module.PRODUCTION_RESTRICTION,
        "proof_sha256": module.sha256(b""),
        "schema": module.CHECKER_RESULT_SCHEMA,
        "sequence_results": sequences,
        "successor_expansion_authorized": bool(not advisory and not no_go),
        "terminal": module.NO_GO if no_go else module.PASS,
        "v1_diagnostic_record_count": 0,
    }


def _replica(
    module,
    tmp_path: Path,
    replica_index: int,
    values: dict[str, dict],
):
    made = []
    for assignment_id in module.EXPECTED_SHARDS:
        value = values[assignment_id]
        raw = module.canonical_bytes(value)
        output = (tmp_path / f"replica-{replica_index}" / f"{assignment_id}.json").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
        proof = (tmp_path / "proofs" / f"{assignment_id}.json").resolve()
        proof.parent.mkdir(parents=True, exist_ok=True)
        if not proof.exists():
            proof.write_bytes(b"")
        made.append(
            {
                "assignment_id": assignment_id,
                "cpu_100ns": 10,
                "output_path": str(output),
                "output_sha256": module.sha256(raw),
                "peak_tree_memory_bytes": 1024,
                "proof_path": str(proof),
                "proof_sha256": module.sha256(proof.read_bytes()),
                "stderr_sha256": module.sha256(b""),
                "stdout_sha256": module.sha256(b""),
                "termination_proven": True,
                "value": value,
            }
        )
    return made


def _values(module, **overrides):
    made = {
        assignment_id: _checker_value(module, assignment_id)
        for assignment_id in module.EXPECTED_SHARDS
    }
    for assignment_id, options in overrides.items():
        made[assignment_id] = _checker_value(module, assignment_id, **options)
    return made


def _aggregate(module, tmp_path: Path, first_values, second_values=None):
    if second_values is None:
        second_values = copy.deepcopy(first_values)
    replicas = [
        _replica(module, tmp_path, 1, first_values),
        _replica(module, tmp_path, 2, second_values),
    ]
    producer_proofs = {
        assignment_id: {
            "assignment_sha256": "A" * 64,
            "plan_sha256": "B" * 64,
            "proof_path": str((tmp_path / "proofs" / f"{assignment_id}.json").resolve()),
            "proof_sha256": module.sha256(b""),
        }
        for assignment_id in module.EXPECTED_SHARDS
    }
    return module.aggregate_checker_results(
        replicas,
        producer_proofs=producer_proofs,
        producer_result_sha256="D" * 64,
        contract_sha256="E" * 64,
        authorization_sha256="F" * 64,
    )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants():
    module = _load()
    with pytest.raises(module.CoordinatorError, match="duplicate JSON key"):
        module.strict_json_bytes(b'{"a":1,"a":2}', "duplicate")
    with pytest.raises(module.CoordinatorError, match="non-finite JSON constant"):
        module.strict_json_bytes(b'{"a":NaN}', "nonfinite")
    with pytest.raises(module.CoordinatorError, match="non-finite number"):
        module.canonical_bytes({"a": float("inf")})


def test_external_incident_binding_is_exact_and_supports_utf8_bom(tmp_path):
    module = _load()
    path = (tmp_path / "request.json").resolve()
    raw = b'\xef\xbb\xbf{"request_id":"' + b"1" * 32 + b'"}\r\n'
    path.write_bytes(raw)
    binding = {
        "byte_count": len(raw),
        "path": str(path),
        "sha256": module.sha256(raw),
    }
    assert module._validate_external_file_binding(binding, "incident") == (path, raw)
    value, observed = module._strict_external_json(path, "incident request")
    assert value == {"request_id": "1" * 32}
    assert observed == raw

    for field, replacement in (
        ("byte_count", len(raw) + 1),
        ("sha256", "0" * 64),
    ):
        changed = dict(binding)
        changed[field] = replacement
        with pytest.raises(module.CoordinatorError, match="identity differs"):
            module._validate_external_file_binding(changed, "incident")


@pytest.mark.parametrize("alias", [True, float(1)])
def test_external_incident_binding_rejects_boolean_and_float_byte_counts(
    tmp_path, alias
):
    module = _load()
    path = (tmp_path / "one-byte.bin").resolve()
    path.write_bytes(b"x")
    binding = {
        "byte_count": alias,
        "path": str(path),
        "sha256": module.sha256(b"x"),
    }
    with pytest.raises(module.CoordinatorError, match="nonnegative integer"):
        module._validate_external_file_binding(binding, "incident")


@pytest.mark.parametrize("digest", [None, 7, "a" * 64, "G" * 64, "A" * 63])
def test_external_incident_binding_rejects_noncanonical_digest(tmp_path, digest):
    module = _load()
    path = (tmp_path / "artifact.bin").resolve()
    path.write_bytes(b"exact")
    binding = {"byte_count": 5, "path": str(path), "sha256": digest}
    with pytest.raises(module.CoordinatorError, match="SHA-256"):
        module._validate_external_file_binding(binding, "incident")


def test_external_incident_binding_rejects_raw_reparse_alias_before_resolve(
    tmp_path, monkeypatch
):
    module = _load()
    target = (tmp_path / "target.bin").resolve()
    target.write_bytes(b"exact")

    class SyntheticRawReparsePath:
        def is_absolute(self):
            return True

        def lstat(self):
            class SyntheticStat:
                st_mode = target.stat().st_mode
                st_file_attributes = 0x400

            return SyntheticStat()

        def is_symlink(self):
            return False

        def resolve(self):
            return target

    monkeypatch.setattr(module, "Path", lambda _value: SyntheticRawReparsePath())
    binding = {
        "byte_count": 5,
        "path": str(tmp_path / "synthetic-reparse.bin"),
        "sha256": module.sha256(b"exact"),
    }
    with pytest.raises(module.CoordinatorError, match="reparse|identity"):
        module._validate_external_file_binding(binding, "incident")


def test_correction7_stable_external_reader_rejects_reparse(tmp_path, monkeypatch):
    module = _load()
    target = (tmp_path / "ledger-target.md").resolve()
    target.write_bytes(b"exact-ledger\n")

    class SyntheticRawReparsePath:
        def is_absolute(self):
            return True

        def lstat(self):
            class SyntheticStat:
                st_mode = target.stat().st_mode
                st_file_attributes = 0x400

            return SyntheticStat()

        def is_symlink(self):
            return False

    monkeypatch.setattr(module, "Path", lambda _value: SyntheticRawReparsePath())
    with pytest.raises(module.CoordinatorError, match="regular non-reparse"):
        module._read_regular_nonreparse_file(target, "ledger")


def test_correction7_stable_external_reader_rejects_nonregular_file(tmp_path):
    module = _load()
    with pytest.raises(module.CoordinatorError, match="regular non-reparse"):
        module._read_regular_nonreparse_file(tmp_path.resolve(), "ledger")


def test_correction9_formal_request_reader_rejects_noncanonical_and_nonregular(
    tmp_path,
):
    module = _load()
    with pytest.raises(module.CoordinatorError, match="regular non-reparse"):
        module._read_canonical_resource_request(tmp_path.resolve(), "resource request")

    request_path = (tmp_path / "request.json").resolve()
    request_path.write_bytes(b'\xef\xbb\xbf{"request_id":"' + b"1" * 32 + b'"}\r\n')
    with pytest.raises(module.CoordinatorError, match="invalid strict JSON|not canonical"):
        module._read_canonical_resource_request(request_path, "resource request")


def test_correction9_historical_request_accepts_exact_pretty_bytes_but_stays_strict(
    tmp_path,
):
    module = _load()
    request_path = (tmp_path / "historical-request.json").resolve()
    pretty_raw = b'{\n  "request_id": "' + b"1" * 32 + b'"\n}\n'
    request_path.write_bytes(pretty_raw)
    binding = {
        "byte_count": len(pretty_raw),
        "path": str(request_path),
        "sha256": module.sha256(pretty_raw),
    }
    resolved, value, observed = module._validate_resource_request_file_binding(
        binding,
        "historical request",
        expected_path=request_path,
        require_canonical=False,
    )
    assert resolved == request_path
    assert value == {"request_id": "1" * 32}
    assert observed == pretty_raw
    with pytest.raises(module.CoordinatorError, match="not canonical"):
        module._validate_resource_request_file_binding(
            binding, "active request", expected_path=request_path
        )

    for invalid in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n'):
        request_path.write_bytes(invalid)
        rebound = {
            "byte_count": len(invalid),
            "path": str(request_path),
            "sha256": module.sha256(invalid),
        }
        with pytest.raises(
            module.CoordinatorError,
            match="duplicate JSON key|non-finite JSON constant",
        ):
            module._validate_resource_request_file_binding(
                rebound,
                "historical request",
                expected_path=request_path,
                require_canonical=False,
            )


def test_correction9_formal_request_reader_rejects_request_file_reparse(
    tmp_path, monkeypatch,
):
    module = _load()
    request_path = (tmp_path / "request.json").resolve()
    request_path.write_bytes(module.canonical_bytes({"request_id": "1" * 32}))
    path_type = type(request_path)
    real_is_symlink = path_type.is_symlink

    def synthetic_is_symlink(path):
        return path == request_path or real_is_symlink(path)

    monkeypatch.setattr(path_type, "is_symlink", synthetic_is_symlink)
    with pytest.raises(module.CoordinatorError, match="regular non-reparse"):
        module._read_canonical_resource_request(request_path, "resource request")


def test_correction9_formal_request_reader_rejects_reparse_ancestor(
    tmp_path, monkeypatch,
):
    module = _load()
    ancestor = (tmp_path / "request-root").resolve()
    ancestor.mkdir()
    request_path = ancestor / "request.json"
    request_path.write_bytes(module.canonical_bytes({"request_id": "1" * 32}))
    path_type = type(request_path)
    real_is_symlink = path_type.is_symlink

    def synthetic_is_symlink(path):
        return path == ancestor or real_is_symlink(path)

    monkeypatch.setattr(path_type, "is_symlink", synthetic_is_symlink)
    with pytest.raises(module.CoordinatorError, match="reparse-path ancestor"):
        module._read_canonical_resource_request(request_path, "resource request")


def test_correction9_formal_request_reader_rejects_identity_change_during_read(
    tmp_path, monkeypatch,
):
    module = _load()
    request_path = (tmp_path / "request.json").resolve()
    request_path.write_bytes(module.canonical_bytes({"request_id": "1" * 32}))
    real_fstat = module.os.fstat
    calls = 0

    class ChangedStat:
        def __init__(self, original):
            self._original = original
            self.st_size = original.st_size + 1
            self.st_mtime_ns = original.st_mtime_ns

        def __getattr__(self, name):
            return getattr(self._original, name)

    def changing_fstat(file_descriptor):
        nonlocal calls
        calls += 1
        observed = real_fstat(file_descriptor)
        return observed if calls == 1 else ChangedStat(observed)

    monkeypatch.setattr(module.os, "fstat", changing_fstat)
    with pytest.raises(module.CoordinatorError, match="changed while reading"):
        module._read_canonical_resource_request(request_path, "resource request")


@pytest.mark.parametrize("mutation", ["rewrite", "truncate"])
def test_correction9_finalizer_ledger_rejects_rewritten_or_truncated_snapshot(
    monkeypatch, mutation,
):
    module = _load()
    approved = "| t0 | " + "1" * 32 + " | APPROVED | exact |"
    preserved = ("header\n" + approved + "\n").encode()
    live = (
        b"rewritten-header\n" + preserved.split(b"\n", 1)[1]
        if mutation == "rewrite"
        else preserved[:-1]
    )
    monkeypatch.setattr(
        module,
        "_read_resource_ledger",
        lambda *_args, **_kwargs: (live, live.decode().splitlines()),
    )
    with pytest.raises(module.CoordinatorError, match="append-only extension"):
        module._validate_finalizer_live_ledger(preserved, approved)


def test_correction9_finalizer_ledger_allows_append_but_rejects_content_race(
    monkeypatch,
):
    module = _load()
    approved = "| t0 | " + "1" * 32 + " | APPROVED | exact |"
    preserved = ("header\n" + approved + "\n").encode()
    appended = preserved + b"| t1 | unrelated | APPROVED | exact |\n"
    observations = iter((preserved, appended))

    def read_ledger(*_args, **_kwargs):
        raw = next(observations)
        return raw, raw.decode().splitlines()

    monkeypatch.setattr(module, "_read_resource_ledger", read_ledger)
    module._validate_finalizer_live_ledger(preserved, approved)

    rewritten = b"changed\n" + preserved.split(b"\n", 1)[1]
    observations = iter((preserved, rewritten))
    with pytest.raises(module.CoordinatorError, match="changed other than by append"):
        module._validate_finalizer_live_ledger(preserved, approved)


def test_correction9_live_ledger_reader_rejects_reparse(tmp_path, monkeypatch):
    module = _load()
    ledger_path = (tmp_path / "ledger.md").resolve()
    ledger_path.write_bytes(b"header\n")
    monkeypatch.setattr(module, "RESOURCE_LEDGER_PATH", ledger_path)
    path_type = type(ledger_path)
    real_is_symlink = path_type.is_symlink

    def synthetic_is_symlink(path):
        return path == ledger_path or real_is_symlink(path)

    monkeypatch.setattr(path_type, "is_symlink", synthetic_is_symlink)
    with pytest.raises(module.CoordinatorError, match="regular non-reparse"):
        module._read_resource_ledger()


@pytest.mark.parametrize("reparse_target", ["file", "ancestor"])
def test_correction9_finalizer_approval_snapshot_preserves_lexical_reparse_checks(
    tmp_path, monkeypatch, reparse_target,
):
    module = _load()
    approval_root = (tmp_path / "finalizer-approval").resolve()
    approval_root.mkdir()
    request_path = (tmp_path / "request.json").resolve()
    request_raw = module.canonical_bytes({"request_id": "1" * 32})
    request_path.write_bytes(request_raw)
    ledger_path = (tmp_path / "ledger.md").resolve()
    approved_line = "| t0 | " + "1" * 32 + " | APPROVED | exact |"
    ledger_raw = ("header\n" + approved_line + "\n").encode()
    ledger_path.write_bytes(ledger_raw)
    ledger_snapshot_path = approval_root / "resource-ledger-pre-run.md"
    ledger_snapshot_path.write_bytes(ledger_raw)
    contract = {"candidate": {"commit": "a" * 40, "tree": "b" * 40}}
    snapshot = {
        "approved_row": {
            "line": approved_line,
            "sha256": module.sha256((approved_line + "\n").encode()),
        },
        "candidate": contract["candidate"],
        "ledger": {
            "byte_count": len(ledger_raw),
            "path": str(ledger_path),
            "sha256": module.sha256(ledger_raw),
            "snapshot_path": str(ledger_snapshot_path),
        },
        "request": {
            "byte_count": len(request_raw),
            "path": str(request_path),
            "request_id": "1" * 32,
            "sha256": module.sha256(request_raw),
        },
        "schema": "anysolver.e4-pl-s3-v2-stage4a-approval-snapshot-v2",
    }
    snapshot_path = approval_root / "approval-snapshot.json"
    snapshot_path.write_bytes(module.canonical_bytes(snapshot))
    target = snapshot_path if reparse_target == "file" else approval_root
    path_type = type(snapshot_path)
    real_is_symlink = path_type.is_symlink

    def synthetic_is_symlink(path):
        return path == target or real_is_symlink(path)

    monkeypatch.setattr(path_type, "is_symlink", synthetic_is_symlink)
    with pytest.raises(module.CoordinatorError, match="reparse"):
        module._validate_approval_snapshot(
            snapshot_path,
            contract=contract,
            request_id="1" * 32,
            request_path=request_path,
            request_raw=request_raw,
        )


def test_correction9_finalizer_approval_snapshot_rejects_relative_lexical_path():
    module = _load()
    with pytest.raises(module.CoordinatorError, match="must be absolute"):
        module._absolute_lexical_path(
            "approvals/finalizer.json", "resource approval snapshot path"
        )


@pytest.mark.parametrize("route", ["completed-wave", "execution-state"])
def test_correction9_active_ledger_routes_use_stable_reader(
    tmp_path, monkeypatch, route,
):
    module = _load()

    def reject_ledger(*_args, **_kwargs):
        raise module.CoordinatorError("synthetic ledger identity changed while reading")

    monkeypatch.setattr(module, "_read_resource_ledger", reject_ledger)
    if route == "completed-wave":
        with pytest.raises(module.CoordinatorError, match="identity changed while reading"):
            module._validate_completed_leaf_wave_ledger_row(
                "1" * 32,
                {"repository": "synthetic", "task": "synthetic"},
                receipt_raw=b"receipt\n",
            )
        return

    resource_root = (tmp_path / "resource-manager").resolve()
    request_root = resource_root / "requests"
    request_root.mkdir(parents=True)
    request_id = "1" * 32
    request_path = request_root / f"{request_id}.json"
    request = {
        "command": "synthetic-command",
        "estimate_minutes": 30,
        "repository": "synthetic-repository",
        "request_id": request_id,
        "requested_at": "2026-08-31T12:00:00+02:00",
        "status": "PENDING",
        "task": "synthetic-task",
    }
    request_raw = module.canonical_bytes(request)
    request_path.write_bytes(request_raw)
    monkeypatch.setattr(module, "RESOURCE_MANAGER_ROOT", resource_root)
    authorization = {
        "contract_sha256": "A" * 64,
        "resource_request": {
            "command_sha256": module.sha256(request["command"].encode()),
            "repository": request["repository"],
            "request_id": request_id,
            "request_path": str(request_path),
            "request_sha256": module.sha256(request_raw),
            "task": request["task"],
        },
    }
    with pytest.raises(module.CoordinatorError, match="identity changed while reading"):
        module.validate_resource_execution_state(authorization)


def test_predecessor_incident_validates_complete_synthetic_graph(tmp_path, monkeypatch):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ROOT", tmp_path / "unrelated-live-root")
    module._validate_predecessor_process_incident(fixture["incident"])
    assert fixture["bounded_calls"] == [fixture["manifest"]]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("checker_processes_started", False),
        ("classifying_records", 0.0),
        ("producer_processes_started", False),
        ("producer_result_present", 0),
    ],
)
def test_predecessor_incident_rejects_scientific_type_aliases(
    tmp_path, monkeypatch, field, replacement
):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    fixture["incident"]["scientific_execution"][field] = replacement
    with pytest.raises(module.CoordinatorError, match="scientific|nonnegative integer"):
        module._validate_predecessor_process_incident(fixture["incident"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("authorization_sha256", "9" * 64),
        ("producer_wave_result_sha256", "9" * 64),
        ("advisory_review_required", 0),
        ("classifying_record_count", False),
        ("v1_diagnostic_record_count", 0.0),
    ],
)
def test_predecessor_incident_rejects_rebound_aggregate_disposition_mutations(
    tmp_path, monkeypatch, field, replacement
):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["aggregate"])
    changed[field] = replacement
    fixture["replace_artifact"]("aggregate", changed)
    with pytest.raises(module.CoordinatorError, match="aggregate"):
        module._validate_predecessor_process_incident(fixture["incident"])


def test_predecessor_incident_rejects_extra_aggregate_key(tmp_path, monkeypatch):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["aggregate"])
    changed["unexpected"] = False
    fixture["replace_artifact"]("aggregate", changed)
    with pytest.raises(module.CoordinatorError, match="keys differ"):
        module._validate_predecessor_process_incident(fixture["incident"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda request: request.__setitem__("estimate_minutes", False), "nonnegative integer"),
        (lambda request: request.__setitem__("estimate_minutes", 30.0), "nonnegative integer"),
        (lambda request: request.__setitem__("command", "tampered"), "request identity"),
        (lambda request: request.__setitem__("unexpected", True), "keys differ"),
    ],
)
def test_predecessor_incident_rejects_rebound_request_mutations(
    tmp_path, monkeypatch, mutation, message
):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["request"])
    changed["command"] = "registered"
    mutation(changed)
    fixture["replace_request"](
        changed,
        rebind_command=changed.get("command") != "tampered",
    )
    with pytest.raises(module.CoordinatorError, match=message):
        module._validate_predecessor_process_incident(fixture["incident"])


def test_predecessor_incident_rejects_arbitrary_authorization_commit(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    fixture["incident"]["authorization"]["commit"] = "f" * 40
    with pytest.raises(module.CoordinatorError, match="authorization binding"):
        module._validate_predecessor_process_incident(fixture["incident"])


@pytest.mark.parametrize("artifact", ["approval_snapshot", "candidate_binding"])
def test_predecessor_incident_rejects_rebound_cross_join_mutations(
    tmp_path, monkeypatch, artifact
):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture[artifact])
    if artifact == "approval_snapshot":
        changed["candidate"]["commit"] = "f" * 40
    else:
        changed["artifact_sha256"] = "f" * 64
    fixture["replace_artifact"](artifact, changed)
    with pytest.raises(module.CoordinatorError, match="approval snapshot|archive join"):
        module._validate_predecessor_process_incident(fixture["incident"])


def test_predecessor_incident_rejects_rebound_phase_plan_manifest_hash(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["phase_plan"])
    changed["manifest_sha256"] = "f" * 64
    fixture["replace_artifact"]("phase_plan", changed)
    changed_manifest = copy.deepcopy(fixture["manifest"])
    for worker in changed_manifest["workers"]:
        worker["plan_sha256"] = fixture["incident"]["phase_plan"]["sha256"]
    fixture["replace_artifact"]("manifest", changed_manifest)
    with pytest.raises(module.CoordinatorError, match="phase-plan|manifest"):
        module._validate_predecessor_process_incident(fixture["incident"])


def test_predecessor_incident_rejects_manifest_coverage_and_ledger_reordering(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["manifest"])
    changed["workers"].pop()
    fixture["replace_artifact"]("manifest", changed)
    with pytest.raises(module.CoordinatorError, match="manifest"):
        module._validate_predecessor_process_incident(fixture["incident"])

    fixture = _synthetic_predecessor_incident(module, tmp_path / "other", monkeypatch)
    rows = fixture["live_ledger"].read_text(encoding="utf-8").splitlines()
    fixture["live_ledger"].write_text(
        "\n".join([rows[0], rows[1], rows[3], rows[2]]) + "\n", encoding="utf-8"
    )
    with pytest.raises(module.CoordinatorError, match="ledger history"):
        module._validate_predecessor_process_incident(fixture["incident"])


def test_resource_deferred_incident_validates_complete_synthetic_graph(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    module._validate_predecessor_resource_deferred_incident(fixture["incident"])


@pytest.mark.parametrize("route", ["predecessor", "resource-deferred"])
def test_correction9_historical_ledger_routes_use_stable_reader(
    tmp_path, monkeypatch, route,
):
    module = _load()
    if route == "predecessor":
        fixture = _synthetic_predecessor_incident(module, tmp_path, monkeypatch)
        validate = module._validate_predecessor_process_incident
    else:
        fixture = _synthetic_resource_deferred_incident(
            module, tmp_path, monkeypatch
        )
        validate = module._validate_predecessor_resource_deferred_incident

    def reject_ledger(*_args, **_kwargs):
        raise module.CoordinatorError("synthetic ledger identity changed while reading")

    monkeypatch.setattr(module, "_read_resource_ledger", reject_ledger)
    with pytest.raises(module.CoordinatorError, match="identity changed while reading"):
        validate(fixture["incident"])


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("memory_admission", "observed_at_event_available_bytes"), 1, "memory admission"),
        (("memory_admission", "required_bytes"), False, "nonnegative integer"),
        (("scientific_execution", "producer_processes_started"), True, "nonnegative integer"),
        (("scientific_execution", "classifying_records"), 0.0, "nonnegative integer"),
        (("scientific_execution", "producer_result_present"), False, "scientific"),
        (("request_reuse_forbidden",), False, "identity"),
    ],
)
def test_resource_deferred_incident_rejects_disposition_mutations(
    tmp_path, monkeypatch, path, replacement, message
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    target = fixture["incident"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(module.CoordinatorError, match=message):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("terminal", "COMPLETED"),
        ("workers", [{"status": "COMPLETED"}]),
        ("manifest_sha256", "F" * 64),
    ],
)
def test_resource_deferred_incident_rejects_rebound_producer_result(
    tmp_path, monkeypatch, field, replacement
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["producer_result"])
    changed[field] = replacement
    fixture["replace_producer"](changed)
    with pytest.raises(module.CoordinatorError, match="producer result"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])


def test_resource_deferred_incident_rejects_aggregate_result_cross_join(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["aggregate"])
    changed["producer_wave_result_sha256"] = "F" * 64
    fixture["replace_artifact"]("aggregate", changed)
    with pytest.raises(module.CoordinatorError, match="aggregate"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])


def test_resource_deferred_incident_rejects_tree_and_root_extent_mutations(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    (fixture["candidate_tree_root"] / "unexpected.txt").write_text(
        "unexpected\n", encoding="utf-8"
    )
    with pytest.raises(module.CoordinatorError, match="candidate-tree extent"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])

    fixture = _synthetic_resource_deferred_incident(
        module, tmp_path / "other", monkeypatch
    )
    (module.RESOURCE_DEFERRED_INCIDENT_ROOT / "unregistered.bin").write_bytes(b"x")
    with pytest.raises(module.CoordinatorError, match="artifact extent"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])


def test_resource_deferred_incident_rejects_same_count_candidate_tree_rename(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    candidate_directory = fixture["candidate_tree_root"] / "docs"
    (candidate_directory / "one.txt").rename(candidate_directory / "renamed.txt")
    with pytest.raises(module.CoordinatorError, match="extent differs from archive"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])


def test_resource_deferred_incident_rejects_same_count_candidate_tree_content_mutation(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    (fixture["candidate_tree_root"] / "docs" / "one.txt").write_bytes(b"two\n")
    with pytest.raises(module.CoordinatorError, match="content differs from archive"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])


def test_resource_deferred_incident_rejects_archive_ref_and_ledger_mutations(
    tmp_path, monkeypatch
):
    module = _load()
    fixture = _synthetic_resource_deferred_incident(module, tmp_path, monkeypatch)
    fixture["incident"]["archive_ref"]["commit"] = "0" * 40
    with pytest.raises(module.CoordinatorError, match="archive ref"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])

    fixture = _synthetic_resource_deferred_incident(
        module, tmp_path / "other", monkeypatch
    )
    rows = fixture["live_ledger"].read_text(encoding="utf-8").splitlines()
    fixture["live_ledger"].write_text(
        "\n".join([rows[0], rows[1], rows[3], rows[2]]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(module.CoordinatorError, match="ledger history"):
        module._validate_predecessor_resource_deferred_incident(fixture["incident"])


def test_stage4a_contract_v4_requires_both_predecessor_incidents():
    module = _load()
    assert module.CONTRACT_KEYS == {
        "adjudication",
        "authority",
        "candidate",
        "coverage",
        "dependencies",
        "execution",
        "frozen_files",
        "git_authority",
        "predecessor_process_incident",
        "predecessor_resource_deferred_incident",
        "production_boundary",
        "protocol",
        "schema",
        "stage",
    }


def test_aggregate_pass_has_exact_coverage_and_registered_order(tmp_path):
    module = _load()
    aggregate = _aggregate(module, tmp_path, _values(module))
    assert aggregate["terminal"] == module.PASS
    assert aggregate["successor_expansion_authorized"] is True
    assert aggregate["advisory_review_required"] is False
    assert aggregate["classifying_record_count"] == 81
    assert aggregate["v1_diagnostic_record_count"] == 0
    assert aggregate["v1_comparator_disposition"] == module.LEAF_V1_DISPOSITION
    assert len(aggregate["sequence_results"]) == 24
    assert len(aggregate["checker_replica_bindings"]) == 3
    assert all(
        len(binding["checker_output_sha256"]) == 2
        for binding in aggregate["checker_replica_bindings"]
    )
    assert [
        (item["diagonal"], item["mask"], item["fraction_percent"])
        for item in aggregate["sequence_results"]
    ] == [
        (diagonal, mask, fraction)
        for diagonal in module.DIAGONAL_ORDER
        for mask in module.MASK_ORDER
        for fraction in module.FRACTION_ORDER
    ]


def test_aggregate_advisory_pass_requires_review_and_blocks_expansion(tmp_path):
    module = _load()
    assignment_id = "S3_V2_FLAT_4A_SLASH"
    aggregate = _aggregate(
        module,
        tmp_path,
        _values(module, **{assignment_id: {"advisory": True}}),
    )
    assert aggregate["terminal"] == module.PASS
    assert aggregate["advisory_review_required"] is True
    assert aggregate["successor_expansion_authorized"] is False
    assert aggregate["formal_failures"] == []


def test_aggregate_no_go_has_precedence_and_scopes_failure_to_diagonal(tmp_path):
    module = _load()
    assignment_id = "S3_V2_FLAT_4A_BACKSLASH"
    aggregate = _aggregate(
        module,
        tmp_path,
        _values(module, **{assignment_id: {"no_go": True}}),
    )
    assert aggregate["terminal"] == module.NO_GO
    assert aggregate["successor_expansion_authorized"] is False
    assert aggregate["advisory_review_required"] is False
    assert aggregate["formal_failures"] == [
        "backslash:dispersed:1:RESPONSE_SLOPE"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.__setitem__("classifying_record_count", 26),
            "identity or coverage",
        ),
        (
            lambda value: value["sequence_results"].pop(),
            "exactly eight sequences",
        ),
        (
            lambda value: value["sequence_results"].__setitem__(
                7, copy.deepcopy(value["sequence_results"][0])
            ),
            "duplicated",
        ),
    ],
)
def test_aggregate_rejects_incomplete_or_duplicate_coverage(tmp_path, mutation, message):
    module = _load()
    values = _values(module)
    mutation(values["S3_V2_FLAT_4A_SLASH"])
    with pytest.raises(module.CoordinatorError, match=message):
        _aggregate(module, tmp_path, values)


def test_aggregate_rejects_replica_byte_disagreement(tmp_path):
    module = _load()
    first = _values(module)
    second = copy.deepcopy(first)
    second["S3_V2_FLAT_4A_ALTERNATING"]["sequence_results"][0][
        "energy_norm_slope"
    ] = 1.21
    with pytest.raises(module.CoordinatorError, match="checker replicas disagree"):
        _aggregate(module, tmp_path, first, second)


def test_blocked_aggregate_uses_the_same_fixed_schema_as_scientific_aggregate(tmp_path):
    module = _load()
    passed = _aggregate(module, tmp_path, _values(module))
    blocked = module.blocked_aggregate(
        authorization_sha256="F" * 64,
        contract_sha256="E" * 64,
        producer_result_sha256=None,
        reason="FORMAL_PROCESS_FAILED",
    )
    assert set(blocked) == set(passed)
    assert blocked["terminal"] == module.BLOCKED
    assert blocked["formal_failures"] == ["FORMAL_PROCESS_FAILED"]
    assert blocked["sequence_results"] == []
    assert blocked["producer_wave_result_sha256"] is None
    assert blocked["successor_expansion_authorized"] is False


def test_coordinator_wall_guard_publishes_canonical_blocked_aggregate(tmp_path):
    module = _load()
    aggregate_path = (tmp_path / "stage4a-aggregate.json").resolve()
    guard = module._CoordinatorWallGuard(
        aggregate_path=aggregate_path,
        started=time.monotonic(),
        exit_function=lambda _code: None,
    )
    guard.bind_evidence(
        authorization_sha256="A" * 64,
        contract_sha256="B" * 64,
    )
    aggregate = guard.publish_fail_closed()
    assert aggregate["terminal"] == module.BLOCKED
    assert aggregate["formal_failures"] == ["COORDINATOR_WALL_EXCEEDED"]
    value, raw = module.strict_json_load(aggregate_path)
    assert value == aggregate
    assert raw == module.canonical_bytes(value)


def test_watchdog_never_publishes_while_a_process_tree_may_be_live(
    tmp_path, monkeypatch
):
    module = _load()
    observed_waits = []
    observed_exits = []

    class NeverStopped:
        @staticmethod
        def wait(seconds):
            observed_waits.append(seconds)
            return False

    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    aggregate_path = (tmp_path / "stage4a-aggregate.json").resolve()
    guard = module._CoordinatorWallGuard(
        aggregate_path=aggregate_path,
        started=100.0,
        exit_function=observed_exits.append,
    )
    guard.bind_evidence(
        authorization_sha256="A" * 64,
        contract_sha256="B" * 64,
    )
    guard.mark_process_phase_active()
    guard._stop = NeverStopped()
    guard._publisher_main()
    assert guard._expired.is_set()
    assert guard.publish_fail_closed() is None
    assert not aggregate_path.exists()
    guard._hard_exit_main()
    assert observed_waits == [
        float(
            module.COORDINATOR_WALL_SECONDS
            - module.COORDINATOR_FAIL_CLOSED_PUBLICATION_RESERVE_SECONDS
        ),
        float(module.COORDINATOR_WALL_SECONDS),
    ]
    assert observed_exits == [module.COORDINATOR_HARD_EXIT_CODE]


def test_expired_main_checkpoint_publishes_only_after_tree_terminal_proof(
    tmp_path
):
    module = _load()
    aggregate_path = (tmp_path / "stage4a-aggregate.json").resolve()
    guard = module._CoordinatorWallGuard(
        aggregate_path=aggregate_path,
        started=time.monotonic(),
        exit_function=lambda _code: None,
    )
    guard.bind_evidence(
        authorization_sha256="A" * 64,
        contract_sha256="B" * 64,
    )
    guard.mark_process_phase_active()
    guard._expired.set()
    with pytest.raises(module._CoordinatorWallExceeded):
        guard.checkpoint()
    assert not aggregate_path.exists()
    guard.mark_process_phase_terminal(proven=True)
    with pytest.raises(module._CoordinatorWallExceeded):
        guard.checkpoint()
    value, raw = module.strict_json_load(aggregate_path)
    assert raw == module.canonical_bytes(value)
    assert value["formal_failures"] == ["COORDINATOR_WALL_EXCEEDED"]


def test_coordinator_hard_wall_uses_absolute_deadline_and_exit_code(monkeypatch):
    module = _load()
    observed_waits = []
    observed_exits = []

    class NeverStopped:
        @staticmethod
        def wait(seconds):
            observed_waits.append(seconds)
            return False

    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    guard = module._CoordinatorWallGuard(
        aggregate_path=Path("aggregate.json"),
        started=100.0,
        exit_function=observed_exits.append,
    )
    guard._stop = NeverStopped()
    guard._hard_exit_main()
    assert observed_waits == [float(module.COORDINATOR_WALL_SECONDS)]
    assert observed_exits == [module.COORDINATOR_HARD_EXIT_CODE]


def test_normal_publication_fails_before_writing_in_reserved_window(
    tmp_path, monkeypatch
):
    module = _load()
    path = (tmp_path / "canonical.json").resolve()

    class ExpiredGuard:
        @staticmethod
        def require_canonical_publication_is_safe(_path):
            return None

        @staticmethod
        def checkpoint():
            raise module._CoordinatorWallExceeded("synthetic wall")

    monkeypatch.setattr(module, "_ACTIVE_COORDINATOR_GUARD", ExpiredGuard())
    with pytest.raises(module._CoordinatorWallExceeded, match="synthetic wall"):
        module._write_exclusive(path, b"{}\n")
    assert not path.exists()


def test_canonical_aggregate_is_rejected_when_tree_terminal_is_unproven(
    tmp_path, monkeypatch
):
    module = _load()
    aggregate_path = (tmp_path / "stage4a-aggregate.json").resolve()
    diagnostic_path = (tmp_path / "stage4a-process-incident.json").resolve()
    guard = module._CoordinatorWallGuard(
        aggregate_path=aggregate_path,
        started=time.monotonic(),
        exit_function=lambda _code: None,
    )
    guard.mark_process_phase_active()
    monkeypatch.setattr(module, "_ACTIVE_COORDINATOR_GUARD", guard)
    with pytest.raises(module.CoordinatorError, match="proven-empty"):
        module._write_exclusive(aggregate_path, b"{}\n")
    module._write_exclusive(diagnostic_path, b"{}\n")
    assert not aggregate_path.exists()
    assert diagnostic_path.read_bytes() == b"{}\n"


def test_correction6_rejects_legacy_stage4a_before_any_guarded_work(
    tmp_path, monkeypatch
):
    module = _load()
    monkeypatch.setattr(
        module,
        "validate_contract",
        lambda _path: pytest.fail("legacy execution loaded the contract"),
    )
    arguments = (
        tmp_path / "contract.json",
        tmp_path / "authorization.json",
        tmp_path,
        tmp_path / "aggregate.json",
    )
    for runner in (module.run_stage4a, module._historical_run_stage4a):
        with pytest.raises(module.CoordinatorError, match="correction 6"):
            runner(*arguments)
    for runner in (
        module._run_stage4a_guarded,
        module._historical_run_stage4a_guarded,
    ):
        with pytest.raises(module.CoordinatorError, match="correction 6"):
            runner(*arguments, object())
    assert module._ACTIVE_COORDINATOR_GUARD is None
    assert not (tmp_path / "aggregate.json").exists()


def test_registered_resource_command_isolated_and_supplies_only_bound_dependencies(tmp_path):
    module = _load()
    assert module.DEPENDENCY_REPOSITORIES == (
        (
            "ANYmaterial",
            Path(
                r"C:\Github\ANYfeSolver\.perf2-worktrees"
                r"\s3-v2-stage4a-anymaterial-dependency"
            ),
        ),
        (
            "ANYgeometry",
            Path(
                r"C:\Github\ANYfeSolver\.perf2-worktrees"
                r"\s3-v2-stage4a-anygeometry-dependency"
            ),
        ),
        (
            "ANYmesh",
            Path(
                r"C:\Github\ANYfeSolver\.perf2-worktrees"
                r"\s3-v2-stage4a-anymesh-dependency"
            ),
        ),
        (
            "ANYfileIO",
            Path(
                r"C:\Github\ANYfeSolver\.perf2-worktrees"
                r"\s3-v2-stage4a-anyfileio-dependency"
            ),
        ),
    )
    plan_path = (tmp_path / "phase4a-plan.json").resolve()
    union_path = (tmp_path / "leaf-union.json").resolve()
    command = module.expected_resource_command(
        python_executable=Path(sys.executable),
        contract_path=ROOT / "contract.json",
        authorization_path=ROOT / "authorization.json",
        output_root=tmp_path,
        aggregate_path=tmp_path / "stage4a-aggregate.json",
        execution_mode="leaf-finalizer",
        plan_path=plan_path,
        leaf_union_path=union_path,
        plan_sha256="A" * 64,
        leaf_union_sha256="B" * 64,
    )
    assert "$env:PYTHONNOUSERSITE='1';" in command
    assert "$env:PYTHONDONTWRITEBYTECODE='1';" in command
    assert " -I -B " in command
    for _name, repository in module.DEPENDENCY_REPOSITORIES:
        assert str((repository / "src").resolve()) in command
    assert str(module.ROOT / "src") not in command
    assert command.count("--finalize-leaf-union") == 1
    assert "--run-stage4a" not in command


def _make_synthetic_dependency_repository(module, tmp_path):
    repository = (tmp_path / "dependency").resolve()
    source = repository / "src"
    package = source / "synthetic_dependency"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repository / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    module._git_run("init", repository=repository)
    module._git_run("add", "--all", repository=repository)
    module._git_run(
        "-c",
        "user.name=Stage4A Test",
        "-c",
        "user.email=stage4a@example.invalid",
        "commit",
        "-m",
        "freeze synthetic dependency",
        repository=repository,
    )
    commit = module._git("rev-parse", "HEAD", repository=repository)
    tree = module._git("rev-parse", "HEAD^{tree}", repository=repository)
    return repository, source, commit, tree


def test_correction7_dependency_authority_rejects_old_live_root():
    module = _load()
    authority, _raw = module.strict_json_load(module.AUTHORITY_PATH)
    dependencies = copy.deepcopy(authority["dependency_authority"]["entries"])
    dependencies[0]["path"] = r"C:\Github\ANYmaterial"
    dependencies[0]["source_path"] = r"C:\Github\ANYmaterial\src"
    with pytest.raises(module.CoordinatorError, match="differs from authority"):
        module._validate_stage4a_dependency_graph(
            dependencies, authority["dependency_authority"]
        )


@pytest.mark.parametrize("extra_kind", ["ignored", "untracked"])
def test_correction7_dependency_source_snapshot_rejects_extra_files(
    tmp_path, extra_kind
):
    module = _load()
    repository, source, commit, tree = _make_synthetic_dependency_repository(
        module, tmp_path
    )
    if extra_kind == "ignored":
        extra = source / "synthetic_dependency" / "__pycache__" / "module.pyc"
    else:
        extra = source / "synthetic_dependency" / "untracked.py"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"not importable authority\n")
    with pytest.raises(
        module.CoordinatorError, match="dirty, untracked, or ignored source files"
    ):
        module._dependency_source_bindings(
            repository,
            source,
            "synthetic",
            expected_commit=commit,
            expected_tree=tree,
        )


def test_correction7_dependency_source_snapshot_rejects_mid_hash_mutation(
    tmp_path, monkeypatch
):
    module = _load()
    repository, source, commit, tree = _make_synthetic_dependency_repository(
        module, tmp_path
    )
    original = module._dependency_source_file_binding
    mutated = False

    def mutate_after_binding(repository_path, source_path, relative_path, location):
        nonlocal mutated
        binding = original(
            repository_path, source_path, relative_path, location
        )
        if not mutated:
            mutated = True
            target = repository_path.joinpath(*Path(relative_path).parts)
            target.write_bytes(target.read_bytes() + b"# post-hash mutation\n")
        return binding

    monkeypatch.setattr(
        module, "_dependency_source_file_binding", mutate_after_binding
    )
    with pytest.raises(module.CoordinatorError, match="snapshot changed while hashing"):
        module._dependency_source_bindings(
            repository,
            source,
            "synthetic",
            expected_commit=commit,
            expected_tree=tree,
        )


def test_correction7_dependency_snapshot_joins_expected_git_identity(tmp_path):
    module = _load()
    repository, source, _commit, tree = _make_synthetic_dependency_repository(
        module, tmp_path
    )
    with pytest.raises(module.CoordinatorError, match="identity changed before hashing"):
        module._dependency_source_bindings(
            repository,
            source,
            "synthetic",
            expected_commit="0" * 40,
            expected_tree=tree,
        )


def test_correction7_dependency_source_rejects_reparse_root(tmp_path, monkeypatch):
    module = _load()
    repository, source, _commit, _tree = _make_synthetic_dependency_repository(
        module, tmp_path
    )
    path_type = type(repository)
    original_lstat = path_type.lstat

    class ReparseInformation:
        def __init__(self, information):
            self._information = information
            self.st_file_attributes = getattr(
                information, "st_file_attributes", 0
            ) | getattr(module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

        def __getattr__(self, name):
            return getattr(self._information, name)

    def marked_root(path):
        information = original_lstat(path)
        if path == repository:
            return ReparseInformation(information)
        return information

    monkeypatch.setattr(path_type, "lstat", marked_root)
    with pytest.raises(module.CoordinatorError, match="reparse or symlink source root"):
        module._dependency_source_file_binding(
            repository,
            source,
            "src/synthetic_dependency/module.py",
            "synthetic",
        )


def test_correction6_resource_command_requires_current_mode_and_rejects_legacy(
    tmp_path,
):
    module = _load()
    arguments = {
        "python_executable": Path(sys.executable),
        "contract_path": ROOT / "contract.json",
        "authorization_path": ROOT / "authorization.json",
        "output_root": tmp_path,
        "aggregate_path": tmp_path / "stage4a-aggregate.json",
    }
    with pytest.raises(TypeError, match="execution_mode"):
        module.expected_resource_command(**arguments)
    with pytest.raises(module.CoordinatorError, match="not authorized by correction 6"):
        module.expected_resource_command(**arguments, execution_mode="legacy")


def test_correction6_rejects_legacy_preparation_before_contract_or_output(
    tmp_path, monkeypatch,
):
    module = _load()
    output_root = (tmp_path / "legacy-output").resolve()
    monkeypatch.setattr(
        module,
        "validate_contract",
        lambda _path: pytest.fail("legacy preparation loaded the contract"),
    )
    for prepare in (
        module.prepare_wave,
        module.run_prepare_stage4a,
        module._historical_prepare_wave,
        module._historical_run_prepare_stage4a,
    ):
        with pytest.raises(module.CoordinatorError, match="correction 6"):
            prepare(tmp_path / "contract.json", output_root)
    with pytest.raises(
        module.CoordinatorError, match="preparation is not authorized by correction 6"
    ):
        module.main(
            [
                "--prepare-only",
                "--contract",
                str(tmp_path / "contract.json"),
                "--output-root",
                str(output_root),
            ]
        )
    assert not output_root.exists()


def _checker_proofs(module, tmp_path):
    return {
        assignment_id: (tmp_path / assignment_id / "proof.json").resolve()
        for assignment_id in module.EXPECTED_SHARDS
    }


class _SyntheticCheckerResources:
    def __init__(self, available_values):
        self._available_values = iter(available_values)
        self.calls = 0

    def available_physical_memory_bytes(self):
        self.calls += 1
        return next(self._available_values)


def test_checker_policy_freezes_three_registered_but_only_two_concurrent_workers():
    module = _load()
    assert module.AUTHORITY_SCHEMA == "anysolver.e4-pl-s3-v2-stage4a-authority-v11"
    assert module.CONTRACT_SCHEMA == "anysolver.e4-pl-s3-v2-stage4a-contract-v8"
    assert module.CHECKER_REGISTERED_SHARDS == 3
    assert module.MAXIMUM_CONCURRENT_WORKERS == 2
    assert module._checker_replica_required_memory_bytes() == 64 * (1 << 30)
    assert module._execution_policy() == {
        "approval_and_ledger_authority_files_regular_nonreparse_required": True,
        "canonical_aggregate_requires_proven_empty_process_trees": True,
        "checker_tree_drain_required_before_queue_advance": True,
        "checker_phase_finalization_reserve_seconds": 60,
        "checker_phase_required_seconds": 960,
        "checker_phase_schedule": "REPLICA_PAIRS_BY_FROZEN_SHARD_ORDER",
        "checker_registered_shards": 3,
        "checker_replica_wall_seconds": 300,
        "checker_replicas_per_shard": 2,
        "coordinator_wall_seconds": 1800,
        "coordinator_fail_closed_publication_reserve_seconds": 15,
        "coordinator_hard_exit_code": 124,
        "coordinator_work_deadline_action": "MARK_EXPIRED_ONLY",
        "git_subprocess_wall_seconds": 60,
        "hard_coordinator_wall_enforced": True,
        "inactivity_seconds": 300,
        "frozen_pre_run_ledger_snapshot_count": 41,
        "frozen_pre_run_ledger_snapshots_byte_identical_required": True,
        "leaf_catalog_count": 81,
        "leaf_finalizer_wall_seconds": 1740,
        "leaf_formal_v1_diagnostic_count": 0,
        "leaf_historical_v1_disposition": (
            "HISTORICAL_V1_COMPARATOR_EXCLUDED_FROM_FORMAL_RUNTIME_NO_FALLBACK"
        ),
        "leaf_logical_record_count": 81,
        "leaf_pair_wave_count": 40,
        "leaf_pairing": "CONSECUTIVE_V2_LEAVES_IN_FROZEN_CATALOG_ORDER",
        "leaf_singleton_wave_count": 1,
        "leaf_v2_classifying_count": 81,
        "leaf_wave_count": 41,
        "leaf_wave_receipt_count": 41,
        "leaf_wave_wall_seconds": 1740,
        "leaf_worker_wall_seconds": 1500,
        "live_ledger_append_only_extension_revalidated_at_every_v5_validation": True,
        "maximum_concurrent_workers": 2,
        "maximum_memory_gib_per_process_tree": 24,
        "memory_admission_headroom_gib": 16,
        "memory_admission_required_bytes": 68_719_476_736,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "resource_request_authority_files_regular_nonreparse_stable_read_required": True,
        "timeout_aggregate_requires_proven_empty_process_trees": True,
        "unproven_tree_hard_deadline_action": (
            "EXIT_WITHOUT_CANONICAL_AGGREGATE"
        ),
        "wave_wall_seconds": 1800,
    }
    assert (
        "tests/test_e4_pl_s3_v2_bounded_process.py"
        in module.REQUIRED_FROZEN_PATHS
    )


def test_checker_phase_runs_replica_pairs_in_frozen_order_without_lost_coverage(
    tmp_path, monkeypatch
):
    module = _load()
    required = module._checker_replica_required_memory_bytes()
    resources = _SyntheticCheckerResources([required])
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    events = []
    calls = {}

    def fake_checker(**kwargs):
        nonlocal active, maximum_active
        assignment_id = kwargs["assignment_id"]
        replica = int(kwargs["output"].parent.parent.name.rsplit("-", 1)[1])
        task_id = (replica, assignment_id)
        with lock:
            calls[task_id] = calls.get(task_id, 0) + 1
            active += 1
            maximum_active = max(maximum_active, active)
            events.append(("START", task_id))
        time.sleep(0.15)
        with lock:
            events.append(("END", task_id))
            active -= 1
        return {"assignment_id": assignment_id, "replica": replica}

    monkeypatch.setattr(module, "_run_checker_process", fake_checker)
    results = module._run_checker_phase(
        bounded=resources,
        proofs=_checker_proofs(module, tmp_path),
        plan=(tmp_path / "plan.json").resolve(),
        output_root=tmp_path.resolve(),
        deadline=time.monotonic() + module.CHECKER_PHASE_REQUIRED_SECONDS + 1,
    )

    frozen_order = list(module.EXPECTED_SHARDS)
    submission_order = [
        (replica, assignment_id)
        for assignment_id in frozen_order
        for replica in (1, 2)
    ]
    assert [result["assignment_id"] for result in results[0]] == frozen_order
    assert [result["assignment_id"] for result in results[1]] == frozen_order
    assert calls == {task_id: 1 for task_id in submission_order}
    assert resources.calls == 1
    assert maximum_active == 2
    assert [task_id for event, task_id in events[:2] if event == "START"] == (
        submission_order[:2]
    )
    for pair_index in range(1, 3):
        pair_start = events.index(("START", submission_order[pair_index * 2]))
        assert sum(event == "END" for event, _task_id in events[:pair_start]) >= 2


def test_checker_phase_resource_deferral_occurs_before_any_launch(
    tmp_path, monkeypatch
):
    module = _load()
    required = module._checker_replica_required_memory_bytes()
    resources = _SyntheticCheckerResources([required - 1])
    calls = []

    def fake_checker(**kwargs):
        calls.append((kwargs["assignment_id"], kwargs["output"]))
        return {"assignment_id": kwargs["assignment_id"]}

    monkeypatch.setattr(module, "_run_checker_process", fake_checker)
    with pytest.raises(module.CoordinatorError, match="resources are deferred"):
        module._run_checker_phase(
            bounded=resources,
            proofs=_checker_proofs(module, tmp_path),
            plan=(tmp_path / "plan.json").resolve(),
            output_root=tmp_path.resolve(),
            deadline=time.monotonic() + module.CHECKER_PHASE_REQUIRED_SECONDS + 1,
        )
    assert resources.calls == 1
    assert calls == []
    assert not list(tmp_path.glob("checker-replica-*"))


def test_checker_phase_time_deferral_occurs_before_any_launch(tmp_path, monkeypatch):
    module = _load()
    resources = _SyntheticCheckerResources(
        [module._checker_replica_required_memory_bytes()]
    )
    calls = []
    monkeypatch.setattr(
        module, "_run_checker_process", lambda **kwargs: calls.append(kwargs)
    )
    with pytest.raises(module.CoordinatorError, match="insufficient coordinator-wall"):
        module._run_checker_phase(
            bounded=resources,
            proofs=_checker_proofs(module, tmp_path),
            plan=(tmp_path / "plan.json").resolve(),
            output_root=tmp_path.resolve(),
            deadline=time.monotonic() + module.CHECKER_PHASE_REQUIRED_SECONDS - 1,
        )
    assert calls == []
    assert not list(tmp_path.glob("checker-replica-*"))


def test_checker_failure_is_not_retried_and_does_not_drop_queued_work(
    tmp_path, monkeypatch
):
    module = _load()
    resources = _SyntheticCheckerResources(
        [module._checker_replica_required_memory_bytes()]
    )
    frozen_order = list(module.EXPECTED_SHARDS)
    calls = {(replica, assignment_id): 0 for assignment_id in frozen_order for replica in (1, 2)}
    lock = threading.Lock()

    def fake_checker(**kwargs):
        assignment_id = kwargs["assignment_id"]
        replica = int(kwargs["output"].parent.parent.name.rsplit("-", 1)[1])
        task_id = (replica, assignment_id)
        with lock:
            calls[task_id] += 1
        if task_id == (1, frozen_order[0]):
            raise module.CoordinatorError("synthetic checker failure")
        time.sleep(0.05)
        return {"assignment_id": assignment_id, "replica": replica}

    monkeypatch.setattr(module, "_run_checker_process", fake_checker)
    with pytest.raises(module.CoordinatorError, match="all six registered tasks"):
        module._run_checker_phase(
            bounded=resources,
            proofs=_checker_proofs(module, tmp_path),
            plan=(tmp_path / "plan.json").resolve(),
            output_root=tmp_path.resolve(),
            deadline=time.monotonic() + module.CHECKER_PHASE_REQUIRED_SECONDS + 1,
        )
    assert all(count == 1 for count in calls.values())


def test_checker_tree_drain_failure_blocks_later_pairs_without_cancelling_peer(
    tmp_path, monkeypatch
):
    module = _load()
    resources = _SyntheticCheckerResources(
        [module._checker_replica_required_memory_bytes()]
    )
    guard = module._CoordinatorWallGuard(
        aggregate_path=(tmp_path / "stage4a-aggregate.json").resolve(),
        started=time.monotonic(),
        exit_function=lambda _code: None,
    )
    guard.bind_evidence(
        authorization_sha256="A" * 64,
        contract_sha256="B" * 64,
    )
    frozen_order = list(module.EXPECTED_SHARDS)
    calls = []
    completed = []
    lock = threading.Lock()

    def fake_checker(**kwargs):
        assignment_id = kwargs["assignment_id"]
        replica = int(kwargs["output"].parent.parent.name.rsplit("-", 1)[1])
        task_id = (replica, assignment_id)
        with lock:
            calls.append(task_id)
        if task_id == (1, frozen_order[0]):
            raise module._CheckerTreeNotDrained("synthetic uncontained tree")
        time.sleep(0.05)
        with lock:
            completed.append(task_id)
        return {"assignment_id": assignment_id, "replica": replica}

    monkeypatch.setattr(module, "_run_checker_process", fake_checker)
    with pytest.raises(module.CoordinatorError, match="queue blocked"):
        module._run_checker_phase(
            bounded=resources,
            proofs=_checker_proofs(module, tmp_path),
            plan=(tmp_path / "plan.json").resolve(),
            output_root=tmp_path.resolve(),
            deadline=time.monotonic() + module.CHECKER_PHASE_REQUIRED_SECONDS + 1,
            wall_guard=guard,
        )
    assert sorted(calls) == sorted(
        [(1, frozen_order[0]), (2, frozen_order[0])]
    )
    assert completed == [(2, frozen_order[0])]
    guard._expired.set()
    assert guard.publish_fail_closed() is None
    assert not guard.aggregate_path.exists()


def test_checker_root_exit_with_live_tree_requires_proven_termination(
    tmp_path, monkeypatch
):
    module = _load()

    class Process:
        returncode = 0

        @staticmethod
        def poll():
            return 0

    class Job:
        instance = None

        def __init__(self, _memory_limit):
            type(self).instance = self
            self.terminate_calls = 0
            self.closed = False

        @staticmethod
        def launch(*_args, **_kwargs):
            return Process()

        @staticmethod
        def accounting():
            return 1, 1, 1

        def terminate(self):
            self.terminate_calls += 1
            return False

        def close(self):
            self.closed = True

    class Bounded:
        _ProcessJob = Job

        @staticmethod
        def _environment():
            return {}

    monkeypatch.setattr(module, "_load_module", lambda *_args: Bounded)
    root = (tmp_path / "checker").resolve()
    with pytest.raises(module._CheckerTreeNotDrained, match="could not prove"):
        module._run_checker_process(
            assignment_id=next(iter(module.EXPECTED_SHARDS)),
            proof=(tmp_path / "proof.json").resolve(),
            plan=(tmp_path / "plan.json").resolve(),
            output=root / "checker.json",
            stdout_path=root / "stdout.log",
            stderr_path=root / "stderr.log",
            deadline=time.monotonic() + 10,
        )
    assert Job.instance.terminate_calls == 1
    assert Job.instance.closed is True


def test_producer_phase_requires_every_launched_tree_terminal_proof():
    module = _load()
    assert module._producer_process_trees_proven_terminal(
        {"terminal": "RESOURCE_DEFERRED", "workers": []}
    )
    assert module._producer_process_trees_proven_terminal(
        {
            "terminal": "COMPLETED",
            "workers": [
                {"termination_proven": True},
                {"termination_proven": True},
                {"termination_proven": True},
            ],
        }
    )
    assert not module._producer_process_trees_proven_terminal(
        {
            "terminal": "BLOCKED_E4_PL_S3_V2_PROCESS_OR_EVIDENCE",
            "workers": [
                {"termination_proven": True},
                {"termination_proven": False},
            ],
        }
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("assignment_sha256", "7" * 64),
        ("plan_sha256", "8" * 64),
        ("proof_sha256", "9" * 64),
    ],
)
def test_checker_must_join_exact_producer_assignment_plan_and_proof(
    tmp_path, field, replacement
):
    module = _load()
    values = _values(module)
    values["S3_V2_FLAT_4A_SLASH"][field] = replacement
    with pytest.raises(module.CoordinatorError, match="not joined"):
        _aggregate(module, tmp_path, values)


def test_validate_producer_proofs_recomputes_each_scientific_hash(tmp_path):
    module = _load()
    workers = []
    completed = []
    for assignment_id in module.EXPECTED_SHARDS:
        proof = (tmp_path / assignment_id / "scientific.json").resolve()
        proof.parent.mkdir(parents=True)
        proof.write_bytes(assignment_id.encode("ascii"))
        workers.append(
            {
                "assignment_id": assignment_id,
                "assignment_sha256": "A" * 64,
                "plan_sha256": "B" * 64,
                "scientific_path": str(proof),
            }
        )
        completed.append(
            {
                "assignment_id": assignment_id,
                "assignment_sha256": "A" * 64,
                "cpu_100ns": 1,
                "input_hashes": [],
                "last_progress_sequence": 31,
                "peak_tree_memory_bytes": 1,
                "plan_sha256": "B" * 64,
                "program_sha256": "C" * 64,
                "returncode": 0,
                "scientific_byte_count": proof.stat().st_size,
                "scientific_payload_sha256": "D" * 64,
                "scientific_record_count": 27,
                "scientific_record_ids_sha256": "E" * 64,
                "scientific_schema": "proof-v1",
                "scientific_sha256": module.sha256(proof.read_bytes()),
                "scientific_terminal": "ACCEPTED_FOR_AGGREGATION",
                "status": "COMPLETED",
                "stderr_sha256": module.sha256(b""),
                "stdout_sha256": module.sha256(b""),
                "termination_proven": True,
            }
        )
    manifest = {"workers": workers}
    manifest_raw = module.canonical_bytes(manifest)
    result = {
        "lane": "flat-proof",
        "manifest_sha256": module.sha256(manifest_raw),
        "schema": module.PRODUCER_RESULT_SCHEMA,
        "terminal": "COMPLETED",
        "wave_id": "S3_V2_FLAT_FUNNEL_4A_FULL",
        "workers": completed,
    }
    result_raw = module.canonical_bytes(result)
    bindings = module.validate_producer_proofs(
        manifest, manifest_raw, result, result_raw
    )
    assert set(bindings) == set(module.EXPECTED_SHARDS)
    completed[0]["scientific_sha256"] = "F" * 64
    mutated_raw = module.canonical_bytes(result)
    with pytest.raises(module.CoordinatorError, match="proof binding differs"):
        module.validate_producer_proofs(manifest, manifest_raw, result, mutated_raw)


def test_approval_snapshot_recomputes_preserved_ledger_bytes(tmp_path, monkeypatch):
    module = _load()
    request_id = "1" * 32
    request_path = (tmp_path / f"{request_id}.json").resolve()
    request_raw = module.canonical_bytes({"request_id": request_id})
    request_path.write_bytes(request_raw)
    live_ledger = (tmp_path / "live-ledger.md").resolve()
    approved_line = f"| 2026-08-31 | {request_id} | APPROVED | exact |"
    ledger_raw = (approved_line + "\n").encode("utf-8")
    live_ledger.write_bytes(ledger_raw)
    monkeypatch.setattr(module, "RESOURCE_LEDGER_PATH", live_ledger)
    preserved = (tmp_path / "resource-ledger-pre-run.md").resolve()
    preserved.write_bytes(ledger_raw)
    contract = {"candidate": {"commit": "a" * 40, "tree": "b" * 40}}
    snapshot = {
        "approved_row": {
            "line": approved_line,
            "sha256": module.sha256((approved_line + "\n").encode("utf-8")),
        },
        "candidate": contract["candidate"],
        "ledger": {
            "byte_count": len(ledger_raw),
            "path": str(live_ledger),
            "sha256": module.sha256(ledger_raw),
            "snapshot_path": str(preserved),
        },
        "request": {
            "byte_count": len(request_raw),
            "path": str(request_path),
            "request_id": request_id,
            "sha256": module.sha256(request_raw),
        },
        "schema": "anysolver.e4-pl-s3-v2-stage4a-approval-snapshot-v2",
    }
    snapshot_path = (tmp_path / "approval-snapshot.json").resolve()
    snapshot_path.write_bytes(module.canonical_bytes(snapshot))
    value, raw, preserved_raw = module._validate_approval_snapshot(
        snapshot_path,
        contract=contract,
        request_id=request_id,
        request_path=request_path,
        request_raw=request_raw,
    )
    assert value == snapshot
    assert raw == module.canonical_bytes(snapshot)
    assert preserved_raw == ledger_raw
    preserved.write_bytes(ledger_raw + b"mutation\n")
    with pytest.raises(module.CoordinatorError, match="identity differs"):
        module._validate_approval_snapshot(
            snapshot_path,
            contract=contract,
            request_id=request_id,
            request_path=request_path,
            request_raw=request_raw,
        )


def test_atomic_publication_never_exposes_partial_canonical_path(tmp_path, monkeypatch):
    module = _load()
    output = tmp_path / "canonical.json"
    real_link = module.os.link

    def fail_link(_source, _target):
        raise OSError("synthetic promotion failure")

    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic"):
        module._write_exclusive(output, b"complete\n")
    assert not output.exists()
    assert not list(tmp_path.glob(".canonical.json.pending-*"))
    monkeypatch.setattr(module.os, "link", real_link)
    module._write_exclusive(output, b"complete\n")
    assert output.read_bytes() == b"complete\n"
    with pytest.raises(module.CoordinatorError, match="overwrite"):
        module._write_exclusive(output, b"other\n")


def test_candidate_archive_extraction_rejects_escape_and_is_exclusive(
    tmp_path, monkeypatch
):
    module = _load()
    monkeypatch.setattr(
        module.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("candidate extraction must inherit the output-root ACL")
        ),
    )
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    safe_archive = safe_root / "candidate.tar"
    with tarfile.open(safe_archive, "w") as bundle:
        directory = tarfile.TarInfo("src/anyfesolver/")
        directory.type = tarfile.DIRTYPE
        bundle.addfile(directory)
        payload = b"# exact candidate\n"
        member = tarfile.TarInfo("src/anyfesolver/__init__.py")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    extracted = module._extract_candidate_archive(safe_archive, safe_root)
    assert (extracted / "src" / "anyfesolver" / "__init__.py").read_bytes() == payload
    fresh_child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())",
            str(extracted / "src" / "anyfesolver" / "__init__.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert fresh_child.stdout.strip() == "# exact candidate"
    with pytest.raises(module.CoordinatorError, match="already exists"):
        module._extract_candidate_archive(safe_archive, safe_root)

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    bad_archive = bad_root / "candidate.tar"
    with tarfile.open(bad_archive, "w") as bundle:
        payload = b"escape"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with pytest.raises(module.CoordinatorError, match="unsafe path"):
        module._extract_candidate_archive(bad_archive, bad_root)
    assert not (tmp_path / "escape.txt").exists()


def test_producer_result_is_contained_by_registered_wave_root(tmp_path):
    module = _load()
    wave_root = (tmp_path / "producer-wave").resolve()
    manifest = {
        "lane": "flat-proof",
        "output_root": str(wave_root),
        "schema": "anysolver.e4-pl-s3-v2-bounded-wave-manifest-v1",
        "wave_id": "S3_V2_FLAT_FUNNEL_4A_FULL",
        "workers": [],
    }
    manifest_path = (tmp_path / "producer-wave-manifest.json").resolve()
    manifest_path.write_bytes(module.canonical_bytes(manifest))
    result_path = module._producer_result_path(manifest_path)
    assert result_path == wave_root / "producer-wave-result.json"
    assert result_path.is_relative_to(wave_root)
    assert result_path != tmp_path / "producer-wave-result.json"


def test_process_incident_records_exact_failure_stage(tmp_path):
    module = _load()
    path = (tmp_path / "stage4a-process-incident.json").resolve()
    module._write_process_incident(
        path,
        authorization_sha256="A" * 64,
        contract_sha256="B" * 64,
        error=module.CoordinatorError("synthetic containment failure"),
        phase="PRODUCER_WAVE",
        producer_result_path=None,
    )
    value, raw = module.strict_json_load(path)
    assert raw == module.canonical_bytes(value)
    assert value == {
        "authorization_sha256": "A" * 64,
        "contract_sha256": "B" * 64,
        "errno": None,
        "exception_message": "synthetic containment failure",
        "exception_type": (
            f"{module.CoordinatorError.__module__}.CoordinatorError"
        ),
        "phase": "PRODUCER_WAVE",
        "producer_result_sha256": None,
        "schema": "anysolver.e4-pl-s3-v2-stage4a-process-incident-v1",
        "winerror": None,
    }


def test_correction6_cli_rejects_legacy_before_runner_access(tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setattr(
        module,
        "run_stage4a",
        lambda *_args, **_kwargs: pytest.fail("legacy runner was reached"),
    )
    with pytest.raises(module.CoordinatorError, match="not authorized by correction 6"):
        module.main(
            [
                "--run-stage4a",
                "--contract",
                str(tmp_path / "contract.json"),
                "--authorization",
                str(tmp_path / "authorization.json"),
                "--output-root",
                str(tmp_path),
                "--aggregate",
                str(tmp_path / "aggregate.json"),
            ]
        )


def test_git_launcher_and_engine_identity_is_complete():
    module = _load()
    runtime = module._discover_git_runtime()
    assert Path(runtime["launcher_path"]).is_file()
    assert Path(runtime["engine_path"]).is_file()
    assert Path(runtime["exec_path"]).is_dir()
    assert runtime["launcher_sha256"] == module.sha256(
        Path(runtime["launcher_path"]).read_bytes()
    )
    assert runtime["engine_sha256"] == module.sha256(
        Path(runtime["engine_path"]).read_bytes()
    )
    assert runtime["version"].startswith("git version ")


def test_every_git_subprocess_receives_a_finite_wall_timeout(monkeypatch):
    module = _load()
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(module, "_git_command", lambda *_args, **_kwargs: ["git"])
    monkeypatch.setattr(module, "_git_environment", lambda: {})
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._git_run("status")
    assert len(observed) == 1
    assert observed[0][1]["timeout"] == float(module.GIT_SUBPROCESS_WALL_SECONDS)


def test_git_timeout_is_capped_by_coordinator_reserved_deadline(monkeypatch):
    module = _load()
    now = time.monotonic()
    observed = []

    class ActiveGuard:
        work_deadline = now + 5.0

        @staticmethod
        def checkpoint():
            return None

    def fake_run(command, **kwargs):
        observed.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(module, "_ACTIVE_COORDINATOR_GUARD", ActiveGuard())
    monkeypatch.setattr(module, "_git_command", lambda *_args, **_kwargs: ["git"])
    monkeypatch.setattr(module, "_git_environment", lambda: {})
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._git_run("status")
    assert len(observed) == 1
    assert 0 < observed[0] <= 5.0


def test_git_timeout_failure_is_fail_closed(monkeypatch):
    module = _load()

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["git"], 1)

    monkeypatch.setattr(module, "_git_command", lambda *_args, **_kwargs: ["git"])
    monkeypatch.setattr(module, "_git_environment", lambda: {})
    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(module.CoordinatorError, match="exceeded its bound"):
        module._git_run("status")


def _correction4_plan(module):
    funnel = module._load_module(
        "_test_stage4a_correction4_funnel", module.FUNNEL_PATH
    )
    manifest, manifest_raw = funnel.strict_json_load(module.MANIFEST_PATH)
    records = funnel.validate_manifest(manifest, manifest_raw)
    plan = funnel.build_phase_plan(records, "4A")
    return plan, funnel.canonical_bytes(plan)


def _correction4_candidate(module, tmp_path=None):
    archive_raw = b"synthetic correction4 candidate archive"
    authority = {
        "candidate_archive_sha256": module.sha256(archive_raw),
        "candidate_commit": "a" * 40,
        "candidate_tree": "b" * 40,
        "producer_program_sha256": module.sha256(module.PRODUCER_PATH.read_bytes()),
    }
    archive_path = None
    if tmp_path is not None:
        archive_path = (tmp_path / "candidate-source.tar").resolve()
        archive_path.write_bytes(archive_raw)
    return authority, archive_path


def _correction4_record(module, member, *, diagnostic=False):
    frozen = member["record"]
    made = {
        "classification": (
            module.LEAF_V1_CLASSIFICATION
            if diagnostic
            else module.LEAF_CLASSIFICATION
        ),
        "connectivity_sha256": frozen["connectivity_sha256"],
        "diagonal": frozen["diagonal"],
        "element_counts": {},
        "energy_norm": {},
        "formulation_counts": {},
        "level": frozen["level"],
        "manifest_index": member["manifest_index"],
        "mask": frozen["mask"],
        "node_count": frozen["node_count"],
        "participation": {},
        "quadratic_forms": {},
        "record_id": member["record_id"],
        "reference": {},
        "response": {},
        "s3_area_fraction_percent": frozen["s3_area_fraction_percent"],
        "solution_energies": {},
        "solver": {},
        "support_counts": {},
    }
    if diagnostic:
        made["formulation_id"] = module.LEAF_V1_FORMULATION_ID
    return made


def _correction4_leaf_proof(module, entry, member, *, protocol_suffix=""):
    assignment = entry["assignment"]
    diagnostic = assignment["computation_role"] == module.LEAF_V1_ROLE
    payload = {
        "computation_role": assignment["computation_role"],
        "leaf_assignment": assignment,
        "phase": "4A",
        "protocol": {
            "classification": module.LEAF_CLASSIFICATION,
            "energy_norm_id": "ENERGY" + protocol_suffix,
            "load_id": "LOAD",
            "reference_id": "REFERENCE",
            "support_id": "SUPPORT",
        },
        "record": _correction4_record(module, member, diagnostic=diagnostic),
        "schema": module.LEAF_PAYLOAD_SCHEMA,
        "v1_comparator_disposition": module.LEAF_V1_DISPOSITION,
    }
    record_ids = [member["record_id"]]
    return {
        "assignment_sha256": entry["leaf_assignment_sha256"],
        "plan_sha256": assignment["plan_sha256"],
        "record_count": 1,
        "record_ids": record_ids,
        "record_ids_sha256": module.sha256(module.canonical_bytes(record_ids)),
        "schema": module.LEAF_SCIENTIFIC_SCHEMA,
        "scientific_payload": payload,
        "scientific_payload_sha256": module.sha256(
            module.canonical_bytes(payload)
        ),
        "selector": module.LEAF_SELECTOR,
        "terminal": module.LEAF_PROOF_TERMINAL,
    }


def _correction4_union(
    module, tmp_path, monkeypatch, *, protocol_mutation_index=None
):
    plan, plan_raw = _correction4_plan(module)
    candidate_authority, archive_path = _correction4_candidate(module, tmp_path)
    catalog = module.build_stage4a_leaf_catalog(
        plan, plan_raw, **candidate_authority
    )
    members = {
        member["record_id"]: member
        for shard in plan["shards"]
        for member in shard["records"]
    }
    paths = {}
    for index, entry in enumerate(catalog["leaves"]):
        member = members[entry["assignment"]["record_id"]]
        path = (tmp_path / "leaf-proofs" / entry["leaf_id"] / "scientific.json").resolve()
        path.parent.mkdir(parents=True)
        proof = _correction4_leaf_proof(
            module,
            entry,
            member,
            protocol_suffix=(
                "_MUTATED" if protocol_mutation_index == index else ""
            ),
        )
        path.write_bytes(module.canonical_bytes(proof))
        paths[entry["leaf_id"]] = path
    wave_catalog = module.build_stage4a_leaf_wave_catalog(catalog)
    receipts = {}
    validated_receipts = {}
    for wave_index, wave in enumerate(wave_catalog["waves"], start=1):
        receipt_path = (tmp_path / f"leaf-wave-receipt-{wave_index:02d}.json").resolve()
        request_id = f"{wave_index:032x}"
        receipt = {
            "attempt": {"sha256": "A" * 64},
            "authorization": {"sha256": "B" * 64},
            "request_command_sha256": "C" * 64,
            "request_id": request_id,
            "result": {"sha256": "D" * 64},
        }
        receipt_raw = module.canonical_bytes(receipt)
        receipt_path.write_bytes(receipt_raw)
        workers = []
        for leaf_id in wave["leaf_ids"]:
            entry = next(item for item in catalog["leaves"] if item["leaf_id"] == leaf_id)
            proof_path = paths[leaf_id]
            workers.append(
                {
                    "assignment_sha256": entry["leaf_assignment_sha256"],
                    "leaf_id": leaf_id,
                    "proof": module._external_file_binding(proof_path, "proof"),
                    "status": "COMPLETED",
                    "termination_proven": True,
                }
            )
        receipts[wave_index] = receipt_path
        validated_receipts[receipt_path] = {
            "path": receipt_path,
            "raw": receipt_raw,
            "receipt": receipt,
            "terminal_ledger_row": {
                "line": f"| t2 | {request_id} | COMPLETED_PASS | synthetic |",
                "sha256": module.sha256(
                    (
                        f"| t2 | {request_id} | COMPLETED_PASS | synthetic |\n"
                    ).encode()
                ),
                "status": "COMPLETED_PASS",
            },
            "workers": workers,
        }

    def fake_cycle(**kwargs):
        wave_index = kwargs["wave_index"]
        return {
            "candidate_authority": candidate_authority,
            "catalog": catalog,
            "plan": plan,
            "plan_raw": plan_raw,
            "wave": wave_catalog["waves"][wave_index - 1],
        }

    monkeypatch.setattr(module, "_validate_stage4a_leaf_cycle", fake_cycle)
    monkeypatch.setattr(
        module,
        "validate_stage4a_leaf_wave_receipt",
        lambda receipt_path, **_kwargs: validated_receipts[Path(receipt_path)],
    )
    contract = {"candidate": {
        "commit": candidate_authority["candidate_commit"],
        "tree": candidate_authority["candidate_tree"],
    }}
    contract_raw = module.canonical_bytes(contract)
    contract_path = (tmp_path / "contract.json").resolve()
    contract_path.write_bytes(contract_raw)
    authorization_path = (tmp_path / "leaf-wave-authorization.json").resolve()
    authorization_raw = module.canonical_bytes({"synthetic": True})
    authorization_path.write_bytes(authorization_raw)
    union = module.build_stage4a_leaf_union(
        catalog,
        receipts,
        candidate_archive_path=archive_path,
        contract=contract,
        contract_path=contract_path,
        contract_raw=contract_raw,
        authorization_path=authorization_path,
        authorization_raw=authorization_raw,
        output_root=tmp_path,
    )
    union_path = (tmp_path / "leaf-union.json").resolve()
    union_path.write_bytes(module.canonical_bytes(union))
    return (
        plan,
        plan_raw,
        catalog,
        union,
        union_path,
        candidate_authority,
        contract,
        contract_path,
        contract_raw,
    )


def _correction4_success_receipt(module, tmp_path, monkeypatch):
    repository = (tmp_path / "repository").resolve()
    reference_cases = repository / "docs" / "reference_cases"
    reference_cases.mkdir(parents=True)
    process_review_path = reference_cases / "process-review.json"
    scientific_review_path = reference_cases / "scientific-review.json"
    monkeypatch.setattr(module, "ROOT", repository)
    monkeypatch.setattr(module, "PROCESS_REVIEW_PATH", process_review_path)
    monkeypatch.setattr(module, "SCIENTIFIC_REVIEW_PATH", scientific_review_path)

    resource_root = (tmp_path / "resource-manager").resolve()
    (resource_root / "requests").mkdir(parents=True)
    (resource_root / "attempts").mkdir()
    live_ledger = resource_root / "ledger.md"
    monkeypatch.setattr(module, "RESOURCE_MANAGER_ROOT", resource_root)
    monkeypatch.setattr(module, "RESOURCE_LEDGER_PATH", live_ledger)

    output_root = (tmp_path / "cycle").resolve()
    output_root.mkdir()
    plan, plan_raw = _correction4_plan(module)
    plan_path = output_root / "phase4a-plan.json"
    plan_path.write_bytes(plan_raw)
    candidate_authority, _archive_path = _correction4_candidate(module, tmp_path)
    catalog = module.build_stage4a_leaf_catalog(
        plan, plan_raw, **candidate_authority
    )
    wave_catalog = module.build_stage4a_leaf_wave_catalog(catalog)
    catalog_sha = module.sha256(module.canonical_bytes(catalog))

    contract = {
        "candidate": {
            "changed_paths": ["docs/reference_cases/synthetic.py"],
            "commit": candidate_authority["candidate_commit"],
            "tree": candidate_authority["candidate_tree"],
        },
        "dependencies": [],
        "frozen_files": [],
    }
    contract_path = reference_cases / "contract.json"
    contract_raw = module.canonical_bytes(contract)
    contract_path.write_bytes(contract_raw)
    expected_review_inputs = module._review_inputs(contract, contract_raw)
    review_bindings = []
    for reviewer_index, (role, verdict, path) in enumerate(
        (
            (
                "PROCESS_AND_AUTHORITY",
                module.EXPECTED_REVIEW_VERDICTS["PROCESS_AND_AUTHORITY"],
                process_review_path,
            ),
            (
                "SCIENTIFIC_AND_MECHANICS",
                module.EXPECTED_REVIEW_VERDICTS["SCIENTIFIC_AND_MECHANICS"],
                scientific_review_path,
            ),
        ),
        start=1,
    ):
        review = {
            "findings": {"P0": [], "P1": []},
            "reviewed_inputs": expected_review_inputs,
            "reviewer_independence": {
                "authored_candidate": False,
                "independent_of_other_reviewer": True,
                "reviewer_id": f"reviewer-{reviewer_index}",
                "reviewer_role": role,
            },
            "schema": module.REVIEW_SCHEMA,
            "verdict": verdict,
        }
        review_raw = module.canonical_bytes(review)
        path.write_bytes(review_raw)
        review_bindings.append(
            {
                "path": path.relative_to(repository).as_posix(),
                "role": role,
                "sha256": module.sha256(review_raw),
                "verdict": verdict,
            }
        )

    authorization_path = (
        reference_cases / "e4_pl_s3_v2_stage4a_leaf_wave_authorization.json"
    )
    monkeypatch.setattr(module, "LEAF_WAVE_AUTHORIZATION_PATH", authorization_path)
    approved_rows = []
    wave_inputs = []
    for wave_index, wave in enumerate(wave_catalog["waves"], start=1):
        request_id = f"{wave_index:032x}"
        wave_root = output_root / "leaf-waves" / f"wave-{wave_index:02d}"
        wave_root.mkdir(parents=True)
        manifest_path = wave_root / "manifest.json"
        result_path = wave_root / "bounded-result.json"
        receipt_path = wave_root / "receipt.json"
        approval_root = output_root / "approvals" / f"wave-{wave_index:02d}"
        approval_root.mkdir(parents=True)
        approval_path = approval_root / "approval-snapshot.json"
        manifest_sha = (
            "D" * 64 if wave_index != 1 else None
        )
        execution = {
            "aggregate_path": str(receipt_path),
            "approval_snapshot_path": str(approval_path),
            "output_root": str(output_root),
            "python_executable": str(Path(sys.executable).resolve()),
        }
        command = module.expected_resource_command(
            python_executable=Path(sys.executable),
            contract_path=contract_path,
            authorization_path=authorization_path,
            output_root=output_root,
            aggregate_path=receipt_path,
            execution_mode="leaf-wave",
            plan_sha256=module.sha256(plan_raw),
            leaf_wave_index=wave_index,
            leaf_catalog_sha256=catalog_sha,
            leaf_wave_manifest_sha256=manifest_sha or "0" * 64,
            leaf_wave_result_path=result_path,
        )
        request = {
            "command": command,
            "estimate_minutes": 30,
            "repository": str(repository),
            "request_id": request_id,
            "requested_at": f"2026-08-31T12:{wave_index:02d}:00+02:00",
            "status": "PENDING",
            "task": f"ANYfeSolver S3 V2A Stage 4A bounded leaf wave {wave_index:02d}",
        }
        request_raw = module.canonical_bytes(request)
        request_path = resource_root / "requests" / f"{request_id}.json"
        request_path.write_bytes(request_raw)
        approved_line = (
            f"| t0-{wave_index:02d} | {request_id} | APPROVED | "
            f"{request['task']} | {repository} | synthetic |"
        )
        approved_rows.append(approved_line)
        wave_inputs.append(
            {
                "approval_path": approval_path,
                "approved_line": approved_line,
                "execution": execution,
                "manifest_path": manifest_path,
                "manifest_sha": manifest_sha,
                "receipt_path": receipt_path,
                "request": request,
                "request_path": request_path,
                "request_raw": request_raw,
                "result_path": result_path,
                "wave": wave,
            }
        )
    ledger_snapshot_raw = (
        "header\n" + "\n".join(approved_rows) + "\n"
    ).encode()
    live_ledger.write_bytes(ledger_snapshot_raw)

    # Complete wave 1's real leaf manifest/proofs before the immutable
    # authorization is serialized, then rebuild its exact request command.
    selected_input = wave_inputs[0]
    manifest_workers = []
    members = {
        member["record_id"]: member
        for shard in plan["shards"]
        for member in shard["records"]
    }
    for leaf_id in selected_input["wave"]["leaf_ids"]:
        leaf = next(item for item in catalog["leaves"] if item["leaf_id"] == leaf_id)
        proof_path = selected_input["manifest_path"].parent / leaf_id / "scientific.json"
        proof_path.parent.mkdir()
        proof = _correction4_leaf_proof(
            module, leaf, members[leaf["assignment"]["record_id"]]
        )
        proof_raw = module.canonical_bytes(proof)
        proof_path.write_bytes(proof_raw)
        manifest_workers.append(
            {
                "assignment_id": leaf_id,
                "assignment_sha256": leaf["leaf_assignment_sha256"],
                "input_hashes": [],
                "plan_sha256": module.sha256(plan_raw),
                "program_sha256": candidate_authority["producer_program_sha256"],
                "scientific_path": str(proof_path),
            }
        )
    manifest = {
        "lane": "flat-leaf",
        "output_root": str(selected_input["manifest_path"].parent),
        "schema": "anysolver.e4-pl-s3-v2-bounded-wave-manifest-v1",
        "wave_id": selected_input["wave"]["wave_id"],
        "workers": manifest_workers,
    }
    manifest_raw = module.canonical_bytes(manifest)
    selected_input["manifest_path"].write_bytes(manifest_raw)
    selected_input["manifest_sha"] = module.sha256(manifest_raw)
    selected_request = selected_input["request"]
    selected_request["command"] = module.expected_resource_command(
        python_executable=Path(sys.executable),
        contract_path=contract_path,
        authorization_path=authorization_path,
        output_root=output_root,
        aggregate_path=selected_input["receipt_path"],
        execution_mode="leaf-wave",
        plan_sha256=module.sha256(plan_raw),
        leaf_wave_index=1,
        leaf_catalog_sha256=catalog_sha,
        leaf_wave_manifest_sha256=selected_input["manifest_sha"],
        leaf_wave_result_path=selected_input["result_path"],
    )
    selected_input["request_raw"] = module.canonical_bytes(selected_request)
    selected_input["request_path"].write_bytes(selected_input["request_raw"])

    authorization_waves = []
    for wave_index, item in enumerate(wave_inputs, start=1):
        request = item["request"]
        request_raw = item["request_raw"]
        approval_root = item["approval_path"].parent
        ledger_snapshot_path = approval_root / "resource-ledger-pre-run.md"
        ledger_snapshot_path.write_bytes(ledger_snapshot_raw)
        approval_snapshot = {
            "approved_row": {
                "line": item["approved_line"],
                "sha256": module.sha256((item["approved_line"] + "\n").encode()),
            },
            "candidate": {
                "commit": contract["candidate"]["commit"],
                "tree": contract["candidate"]["tree"],
            },
            "ledger": {
                "byte_count": len(ledger_snapshot_raw),
                "path": str(live_ledger),
                "sha256": module.sha256(ledger_snapshot_raw),
                "snapshot_path": str(ledger_snapshot_path),
            },
            "request": {
                "byte_count": len(request_raw),
                "path": str(item["request_path"]),
                "request_id": request["request_id"],
                "sha256": module.sha256(request_raw),
            },
            "schema": "anysolver.e4-pl-s3-v2-stage4a-approval-snapshot-v2",
        }
        approval_raw = module.canonical_bytes(approval_snapshot)
        item["approval_path"].write_bytes(approval_raw)
        authorization_waves.append(
            {
                "execution_paths": item["execution"],
                "leaf_catalog_sha256": catalog_sha,
                "leaf_wave_manifest_sha256": item["manifest_sha"] or "0" * 64,
                "leaf_wave_result_path": str(item["result_path"]),
                "ledger_approval": {
                    "approved_row_sha256": approval_snapshot["approved_row"]["sha256"],
                    "ledger_path": str(live_ledger),
                    "snapshot_path": str(item["approval_path"]),
                    "snapshot_sha256": module.sha256(approval_raw),
                },
                "plan_sha256": module.sha256(plan_raw),
                "resource_request": {
                    "command_sha256": module.sha256(request["command"].encode()),
                    "request_id": request["request_id"],
                    "request_path": str(item["request_path"]),
                    "request_sha256": module.sha256(request_raw),
                    "repository": str(repository),
                    "task": request["task"],
                },
                "wave_index": wave_index,
            }
        )
    authorization = {
        "contract_path": contract_path.relative_to(repository).as_posix(),
        "contract_sha256": module.sha256(contract_raw),
        "formal_execution_authorized": True,
        "implementation_reviews": review_bindings,
        "leaf_waves": authorization_waves,
        "resource_lock_required": True,
        "schema": module.LEAF_WAVE_AUTHORIZATION_SCHEMA,
        "user_approval": {"recorded": True, "source": "synthetic user approval"},
    }
    authorization_raw = module.canonical_bytes(authorization)
    authorization_path.write_bytes(authorization_raw)

    cycle = {
        "candidate_authority": candidate_authority,
        "catalog": catalog,
        "manifest": manifest,
        "manifest_path": selected_input["manifest_path"],
        "plan": plan,
        "plan_raw": plan_raw,
        "wave": selected_input["wave"],
    }
    result_workers = []
    for spec in manifest_workers:
        proof_path = Path(spec["scientific_path"])
        proof = module.strict_json_bytes(proof_path.read_bytes(), str(proof_path))
        result_workers.append(
            {
                "assignment_id": spec["assignment_id"],
                "assignment_sha256": spec["assignment_sha256"],
                "cpu_100ns": 1,
                "input_hashes": [],
                "last_progress_sequence": 1,
                "peak_tree_memory_bytes": 1,
                "plan_sha256": spec["plan_sha256"],
                "program_sha256": spec["program_sha256"],
                "returncode": 0,
                "scientific_byte_count": proof_path.stat().st_size,
                "scientific_payload_sha256": proof["scientific_payload_sha256"],
                "scientific_record_count": 1,
                "scientific_record_ids_sha256": proof["record_ids_sha256"],
                "scientific_schema": module.LEAF_SCIENTIFIC_SCHEMA,
                "scientific_sha256": module.sha256(proof_path.read_bytes()),
                "scientific_terminal": module.LEAF_PROOF_TERMINAL,
                "status": "COMPLETED",
                "stderr_sha256": module.sha256(b""),
                "stdout_sha256": module.sha256(b""),
                "termination_proven": True,
            }
        )
    result = {
        "lane": "flat-leaf",
        "manifest_sha256": module.sha256(manifest_raw),
        "schema": module.PRODUCER_RESULT_SCHEMA,
        "terminal": "COMPLETED",
        "wave_id": selected_input["wave"]["wave_id"],
        "workers": result_workers,
    }
    result_raw = module.canonical_bytes(result)
    selected_input["result_path"].write_bytes(result_raw)
    attempt = {
        "contract_sha256": module.sha256(contract_raw),
        "request_id": selected_request["request_id"],
        "schema": "anysolver.resource-attempt-claim-v1",
    }
    (resource_root / "attempts" / f"{selected_request['request_id']}.json").write_bytes(
        module.canonical_bytes(attempt)
    )
    selected_authorization, _ = module._validate_leaf_wave_authorization_v5(
        path=authorization_path,
        value=authorization,
        raw=authorization_raw,
        contract_path=contract_path,
        contract_raw=contract_raw,
        selected_wave_index=1,
        selected_plan_sha256=module.sha256(plan_raw),
        selected_leaf_catalog_sha256=catalog_sha,
        selected_manifest_sha256=module.sha256(manifest_raw),
        selected_result_path=selected_input["result_path"],
    )
    _result, _result_raw, accepted = module._validate_stage4a_leaf_wave_result(
        selected_input["result_path"], cycle
    )
    receipt = module._stage4a_leaf_wave_receipt(
        authorization=selected_authorization,
        authorization_path=authorization_path,
        contract_path=contract_path,
        cycle=cycle,
        result_path=selected_input["result_path"],
        result_raw=result_raw,
        workers=accepted,
        wave_index=1,
    )
    receipt_raw = module.canonical_bytes(receipt)
    selected_input["receipt_path"].write_bytes(receipt_raw)
    started_line = (
        f"| t1 | {selected_request['request_id']} | EXECUTION_STARTED | "
        f"{selected_request['task']} | {repository} | synthetic |"
    )
    completed_line = (
        f"| t2 | {selected_request['request_id']} | COMPLETED_PASS | "
        f"{selected_request['task']} | {repository} | receipt bytes "
        f"{len(receipt_raw)} SHA-256 {module.sha256(receipt_raw)} |"
    )
    live_ledger.write_bytes(
        ledger_snapshot_raw
        + (started_line + "\n" + completed_line + "\n").encode("utf-8")
    )
    return {
        "authorization_path": authorization_path,
        "authorization_raw": authorization_raw,
        "completed_line": completed_line,
        "contract": contract,
        "contract_path": contract_path,
        "contract_raw": contract_raw,
        "cycle": cycle,
        "live_ledger": live_ledger,
        "output_root": output_root,
        "receipt": receipt,
        "receipt_path": selected_input["receipt_path"],
        "receipt_raw": receipt_raw,
    }


def _validate_correction4_leaf_authorization(module, fixture, wave_index=1):
    authorization, authorization_raw = module.strict_json_load(
        fixture["authorization_path"]
    )
    wave = authorization["leaf_waves"][wave_index - 1]
    return module._validate_leaf_wave_authorization_v5(
        path=fixture["authorization_path"],
        value=authorization,
        raw=authorization_raw,
        contract_path=fixture["contract_path"],
        contract_raw=fixture["contract_raw"],
        selected_wave_index=wave_index,
        selected_plan_sha256=wave["plan_sha256"],
        selected_leaf_catalog_sha256=wave["leaf_catalog_sha256"],
        selected_manifest_sha256=wave["leaf_wave_manifest_sha256"],
        selected_result_path=Path(wave["leaf_wave_result_path"]),
    )


@pytest.mark.parametrize("route", ["leaf", "finalizer", "execution-state"])
def test_correction9_all_formal_request_routes_use_stable_nonreparse_reader(
    tmp_path, monkeypatch, route,
):
    module = _load()
    fixture = _correction4_success_receipt(module, tmp_path, monkeypatch)
    leaf_authorization, leaf_authorization_raw = module.strict_json_load(
        fixture["authorization_path"]
    )
    selected_wave = leaf_authorization["leaf_waves"][0]
    request_path = Path(
        selected_wave["resource_request"]["request_path"]
    ).resolve()

    finalizer_path = (
        fixture["contract_path"].parent / "finalizer-authorization.json"
    ).resolve()
    finalizer_authorization = {
        "contract_path": leaf_authorization["contract_path"],
        "contract_sha256": leaf_authorization["contract_sha256"],
        "execution_paths": selected_wave["execution_paths"],
        "formal_execution_authorized": True,
        "implementation_reviews": leaf_authorization["implementation_reviews"],
        "ledger_approval": selected_wave["ledger_approval"],
        "resource_lock_required": True,
        "resource_request": selected_wave["resource_request"],
        "schema": module.AUTHORIZATION_SCHEMA,
        "user_approval": leaf_authorization["user_approval"],
    }
    finalizer_path.write_bytes(module.canonical_bytes(finalizer_authorization))
    monkeypatch.setattr(module, "EXECUTION_AUTHORIZATION_PATH", finalizer_path)

    if route == "execution-state":
        selected_authorization, _selected_raw = (
            module._validate_leaf_wave_authorization_v5(
                path=fixture["authorization_path"],
                value=leaf_authorization,
                raw=leaf_authorization_raw,
                contract_path=fixture["contract_path"],
                contract_raw=fixture["contract_raw"],
                selected_wave_index=1,
                selected_plan_sha256=selected_wave["plan_sha256"],
                selected_leaf_catalog_sha256=selected_wave["leaf_catalog_sha256"],
                selected_manifest_sha256=selected_wave["leaf_wave_manifest_sha256"],
                selected_result_path=Path(selected_wave["leaf_wave_result_path"]),
            )
        )

    real_reader = module._read_regular_nonreparse_file

    def reject_request(path, location):
        if Path(path).resolve() == request_path:
            raise module.CoordinatorError(
                "synthetic resource request identity changed while reading"
            )
        return real_reader(path, location)

    monkeypatch.setattr(module, "_read_regular_nonreparse_file", reject_request)
    with pytest.raises(module.CoordinatorError, match="identity changed while reading"):
        if route == "leaf":
            _validate_correction4_leaf_authorization(module, fixture)
        elif route == "finalizer":
            module.validate_authorization(
                finalizer_path,
                contract_path=fixture["contract_path"],
                contract_raw=fixture["contract_raw"],
                execution_mode="leaf-finalizer",
            )
        else:
            module.validate_resource_execution_state(selected_authorization)


def test_correction9_execution_state_rejects_request_content_changed_after_authority(
    tmp_path, monkeypatch,
):
    module = _load()
    fixture = _correction4_success_receipt(module, tmp_path, monkeypatch)
    selected, _raw = _validate_correction4_leaf_authorization(module, fixture)
    request_path = Path(selected["resource_request"]["request_path"])
    request, _request_raw = module.strict_json_load(request_path)
    request["requested_at"] = "2026-08-31T23:59:59+02:00"
    request_path.write_bytes(module.canonical_bytes(request))
    with pytest.raises(module.CoordinatorError, match="changed after authorization"):
        module.validate_resource_execution_state(selected)


def test_correction4_catalog_matches_producer_and_has_exact_wave_partition(
    monkeypatch,
):
    module = _load()
    assert module.AUTHORITY_SCHEMA == "anysolver.e4-pl-s3-v2-stage4a-authority-v11"
    assert "tests/test_e4_pl_s3_v2_component_cache.py" in module.REQUIRED_FROZEN_PATHS
    plan, plan_raw = _correction4_plan(module)
    candidate_authority, _archive = _correction4_candidate(module)
    catalog = module.build_stage4a_leaf_catalog(
        plan, plan_raw, **candidate_authority
    )
    monkeypatch.syspath_prepend(str(module.REFERENCE_CASES))
    producer = module._load_module(
        "_test_stage4a_correction4_producer", module.PRODUCER_PATH
    )
    assert catalog["leaves"] == producer.build_leaf_catalog(
        plan, module.sha256(plan_raw), **candidate_authority
    )
    assert catalog["leaf_count"] == 81
    assert catalog["logical_record_count"] == 81
    assert catalog["v2_classifying_count"] == 81
    assert catalog["v1_diagnostic_count"] == 0
    assert catalog["v1_comparator_disposition"] == module.LEAF_V1_DISPOSITION
    assert [leaf["assignment"]["catalog_index"] for leaf in catalog["leaves"]] == list(
        range(81)
    )
    assert sum(
        leaf["assignment"]["computation_role"] == module.LEAF_V2_ROLE
        for leaf in catalog["leaves"]
    ) == 81
    assert sum(
        leaf["assignment"]["computation_role"] == module.LEAF_V1_ROLE
        for leaf in catalog["leaves"]
    ) == 0

    waves = module.build_stage4a_leaf_wave_catalog(catalog)
    assert waves["wave_count"] == 41
    assert waves["pair_wave_count"] == 40
    assert waves["singleton_wave_count"] == 1
    assert sum(wave["worker_count"] == 2 for wave in waves["waves"]) == 40
    assert sum(wave["worker_count"] == 1 for wave in waves["waves"]) == 1
    for wave in waves["waves"]:
        roles = [
            next(
                leaf["assignment"]["computation_role"]
                for leaf in catalog["leaves"]
                if leaf["leaf_id"] == leaf_id
            )
            for leaf_id in wave["leaf_ids"]
        ]
        assert roles == [module.LEAF_V2_ROLE] * wave["worker_count"]
    assert [leaf_id for wave in waves["waves"] for leaf_id in wave["leaf_ids"]] == [
        leaf["leaf_id"] for leaf in catalog["leaves"]
    ]
    assert len(
        {
            digest
            for wave in waves["waves"]
            for digest in wave["leaf_assignment_sha256"]
        }
    ) == 81
    assert [
        index
        for wave in waves["waves"]
        for index in wave["logical_record_indices"]
    ] == list(range(81))


def test_correction4_leaf_proof_keeps_generic_ten_key_envelope_and_binds_candidate():
    module = _load()
    plan, plan_raw = _correction4_plan(module)
    candidate_authority, _archive = _correction4_candidate(module)
    catalog = module.build_stage4a_leaf_catalog(
        plan, plan_raw, **candidate_authority
    )
    entry = catalog["leaves"][0]
    member = plan["shards"][0]["records"][0]
    proof = _correction4_leaf_proof(module, entry, member)
    assert set(proof) == {
        "assignment_sha256",
        "plan_sha256",
        "record_count",
        "record_ids",
        "record_ids_sha256",
        "schema",
        "scientific_payload",
        "scientific_payload_sha256",
        "selector",
        "terminal",
    }
    assert proof["scientific_payload"]["leaf_assignment"] == entry["assignment"]
    for key, value in candidate_authority.items():
        assert proof["scientific_payload"]["leaf_assignment"][key] == value


def test_correction6_formal_catalog_and_proofs_reject_v1_and_correction5_artifacts():
    module = _load()
    plan, plan_raw = _correction4_plan(module)
    candidate_authority, _archive = _correction4_candidate(module)
    catalog = module.build_stage4a_leaf_catalog(
        plan, plan_raw, **candidate_authority
    )
    assert all(
        leaf["assignment"]["computation_role"] == module.LEAF_V2_ROLE
        for leaf in catalog["leaves"]
    )
    entry = catalog["leaves"][0]
    member = plan["shards"][0]["records"][0]
    proof = _correction4_leaf_proof(module, entry, member)
    module.validate_stage4a_leaf_proof(
        proof,
        module.canonical_bytes(proof),
        entry=entry,
        member=member,
    )

    forged = copy.deepcopy(proof)
    forged["scientific_payload"]["computation_role"] = module.LEAF_V1_ROLE
    forged["scientific_payload_sha256"] = module.sha256(
        module.canonical_bytes(forged["scientific_payload"])
    )
    with pytest.raises(module.CoordinatorError, match="identity differs"):
        module.validate_stage4a_leaf_proof(
            forged,
            module.canonical_bytes(forged),
            entry=entry,
            member=member,
        )

    old_catalog = copy.deepcopy(catalog)
    old_catalog["schema"] = "anysolver.e4-pl-s3-v2-stage4a-leaf-catalog-v2"
    with pytest.raises(module.CoordinatorError, match="identity differs"):
        module.build_stage4a_leaf_wave_catalog(old_catalog)
    old_proof = copy.deepcopy(proof)
    old_proof["schema"] = "anysolver.e4-pl-s3-v2-stage4a-leaf-scientific-v2"
    with pytest.raises(module.CoordinatorError, match="identity differs"):
        module.validate_stage4a_leaf_proof(
            old_proof,
            module.canonical_bytes(old_proof),
            entry=entry,
            member=member,
        )


def test_correction4_leaf_manifests_are_bounded_executable_pairs_and_singleton(
    tmp_path, monkeypatch,
):
    module = _load()
    plan, plan_raw = _correction4_plan(module)
    candidate_authority, archive_path = _correction4_candidate(module, tmp_path)
    catalog = module.build_stage4a_leaf_catalog(
        plan, plan_raw, **candidate_authority
    )
    waves = module.build_stage4a_leaf_wave_catalog(catalog)["waves"]
    plan_path = (tmp_path / "phase4a-plan.json").resolve()
    plan_path.write_bytes(plan_raw)
    binding_path = (tmp_path / "candidate-binding.json").resolve()
    binding_path.write_bytes(module.canonical_bytes({"synthetic": True}))
    contract_path = (tmp_path / "contract.json").resolve()
    contract_path.write_bytes(module.canonical_bytes({"synthetic": True}))
    authority, authority_raw = module.strict_json_load(module.AUTHORITY_PATH)
    assert authority_raw == module.canonical_bytes(authority)
    dependency_input_hashes = module._validate_stage4a_dependency_graph(
        authority["dependency_authority"]["entries"],
        authority["dependency_authority"],
    )
    assert len(dependency_input_hashes) == 126
    assert len(module.canonical_bytes(dependency_input_hashes)) == 24730
    assert module.sha256(module.canonical_bytes(dependency_input_hashes)) == (
        "FDB2585F82394FBC4E631E7F4960FFEE800DDF8485FE7B3574DA9BB90ABA77D5"
    )
    source_root = (tmp_path / "candidate-source").resolve()
    source_root.mkdir()
    monkeypatch.syspath_prepend(str(module.REFERENCE_CASES))
    bounded = module._load_module(
        "_test_stage4a_correction4_bounded", module.BOUNDED_PATH
    )
    selected_waves = [
        next(wave for wave in waves if wave["worker_count"] == worker_count)
        for worker_count in (2, 1)
    ]
    for wave in selected_waves:
        wave_index = waves.index(wave) + 1
        manifest = module._build_stage4a_leaf_wave_manifest(
            wave=wave,
            catalog=catalog,
            plan_path=plan_path,
            candidate_source_root=source_root,
            candidate_archive_path=archive_path,
            candidate_binding_path=binding_path,
            contract_path=contract_path,
            dependency_input_hashes=dependency_input_hashes,
            wave_root=(tmp_path / f"wave-{wave_index:02d}").resolve(),
        )
        _wave_id, lane, _root, workers = bounded.validate_manifest(manifest)
        assert lane == "flat-leaf"
        assert len(workers) == wave["worker_count"]
        assert all(worker.wall_seconds == 1500 for worker in workers)
        assert len({worker.scientific_path.parent for worker in workers}) == len(workers)
        for worker in manifest["workers"]:
            assert len(worker["input_hashes"]) == 134
            assert all(item in worker["input_hashes"] for item in dependency_input_hashes)
            command = worker["command"]
            assert command[command.index("--candidate-commit") + 1] == candidate_authority[
                "candidate_commit"
            ]
            assert command[command.index("--candidate-tree") + 1] == candidate_authority[
                "candidate_tree"
            ]
            assert command[
                command.index("--producer-program-sha256") + 1
            ] == candidate_authority["producer_program_sha256"]

    pair = next(wave for wave in waves if wave["worker_count"] == 2)
    replacement = next(
        leaf
        for leaf in catalog["leaves"]
        if leaf["assignment"]["logical_record_index"] == 3
    )
    cross_record = copy.deepcopy(pair)
    cross_record["leaf_ids"][1] = replacement["leaf_id"]
    cross_record["leaf_assignment_sha256"][1] = replacement[
        "leaf_assignment_sha256"
    ]
    with pytest.raises(module.CoordinatorError, match="consecutive V2 partition"):
        module._build_stage4a_leaf_wave_manifest(
            wave=cross_record,
            catalog=catalog,
            plan_path=plan_path,
            candidate_source_root=source_root,
            candidate_archive_path=archive_path,
            candidate_binding_path=binding_path,
            contract_path=contract_path,
            dependency_input_hashes=dependency_input_hashes,
            wave_root=(tmp_path / "cross-record-wave").resolve(),
        )


def test_correction4_success_receipt_requires_full_authority_and_exact_terminal_hash(
    tmp_path, monkeypatch,
):
    module = _load()
    fixture = _correction4_success_receipt(module, tmp_path, monkeypatch)
    validated = module.validate_stage4a_leaf_wave_receipt(
        fixture["receipt_path"],
        contract=fixture["contract"],
        contract_path=fixture["contract_path"],
        contract_raw=fixture["contract_raw"],
        cycle=fixture["cycle"],
        wave_index=1,
        allowed_root=fixture["output_root"],
        expected_authorization_path=fixture["authorization_path"],
        expected_authorization_raw=fixture["authorization_raw"],
    )
    assert validated["terminal_ledger_row"] == {
        "line": fixture["completed_line"],
        "sha256": module.sha256((fixture["completed_line"] + "\n").encode()),
        "status": "COMPLETED_PASS",
    }
    correction5_receipt = copy.deepcopy(fixture["receipt"])
    correction5_receipt["schema"] = (
        "anysolver.e4-pl-s3-v2-stage4a-leaf-wave-receipt-v2"
    )
    fixture["receipt_path"].write_bytes(
        module.canonical_bytes(correction5_receipt)
    )
    with pytest.raises(module.CoordinatorError, match="identity differs"):
        module.validate_stage4a_leaf_wave_receipt(
            fixture["receipt_path"],
            contract=fixture["contract"],
            contract_path=fixture["contract_path"],
            contract_raw=fixture["contract_raw"],
            cycle=fixture["cycle"],
            wave_index=1,
            allowed_root=fixture["output_root"],
            expected_authorization_path=fixture["authorization_path"],
            expected_authorization_raw=fixture["authorization_raw"],
        )
    fixture["receipt_path"].write_bytes(fixture["receipt_raw"])
    forged = fixture["completed_line"].replace(
        module.sha256(fixture["receipt_raw"]), "F" * 64
    )
    ledger_raw = fixture["live_ledger"].read_bytes()
    fixture["live_ledger"].write_bytes(
        ledger_raw.replace(
            fixture["completed_line"].encode("utf-8"), forged.encode("utf-8")
        )
    )
    with pytest.raises(module.CoordinatorError, match="exact receipt bytes"):
        module.validate_stage4a_leaf_wave_receipt(
            fixture["receipt_path"],
            contract=fixture["contract"],
            contract_path=fixture["contract_path"],
            contract_raw=fixture["contract_raw"],
            cycle=fixture["cycle"],
            wave_index=1,
            allowed_root=fixture["output_root"],
            expected_authorization_path=fixture["authorization_path"],
            expected_authorization_raw=fixture["authorization_raw"],
        )


@pytest.mark.parametrize("mutation", ["rewrite", "truncate"])
def test_correction7_leaf_authorization_rejects_nonappend_ledger_mutation(
    tmp_path, monkeypatch, mutation,
):
    module = _load()
    fixture = _correction4_success_receipt(module, tmp_path, monkeypatch)
    _validate_correction4_leaf_authorization(module, fixture)
    authorization, _authorization_raw = module.strict_json_load(
        fixture["authorization_path"]
    )
    approval_path = Path(
        authorization["leaf_waves"][0]["execution_paths"]["approval_snapshot_path"]
    )
    approval, _approval_raw = module.strict_json_load(approval_path)
    preserved_raw = Path(approval["ledger"]["snapshot_path"]).read_bytes()
    live_raw = fixture["live_ledger"].read_bytes()
    if mutation == "rewrite":
        _header, preserved_tail = preserved_raw.split(b"\n", 1)
        mutated = b"rewritten-header\n" + preserved_tail + live_raw[len(preserved_raw):]
    else:
        mutated = preserved_raw[:-1]
    fixture["live_ledger"].write_bytes(mutated)
    with pytest.raises(module.CoordinatorError, match="append-only extension"):
        _validate_correction4_leaf_authorization(module, fixture)


@pytest.mark.parametrize("mutation", ["rewrite", "truncate", "append"])
def test_correction7_leaf_authorization_requires_one_common_ledger_snapshot(
    tmp_path, monkeypatch, mutation,
):
    module = _load()
    fixture = _correction4_success_receipt(module, tmp_path, monkeypatch)
    authorization, _authorization_raw = module.strict_json_load(
        fixture["authorization_path"]
    )
    second_wave = authorization["leaf_waves"][1]
    approval_path = Path(
        second_wave["execution_paths"]["approval_snapshot_path"]
    )
    approval, _approval_raw = module.strict_json_load(approval_path)
    preserved_path = Path(approval["ledger"]["snapshot_path"])
    common_raw = preserved_path.read_bytes()
    if mutation == "rewrite":
        different_raw = common_raw.replace(b"header\n", b"Header\n", 1)
    elif mutation == "truncate":
        assert common_raw.endswith(b"\n")
        different_raw = common_raw[:-1]
    else:
        different_raw = common_raw + b"different-wave-snapshot\n"
    preserved_path.write_bytes(different_raw)
    approval["ledger"]["byte_count"] = len(different_raw)
    approval["ledger"]["sha256"] = module.sha256(different_raw)
    approval_raw = module.canonical_bytes(approval)
    approval_path.write_bytes(approval_raw)
    second_wave["ledger_approval"]["snapshot_sha256"] = module.sha256(approval_raw)
    fixture["authorization_path"].write_bytes(module.canonical_bytes(authorization))
    with pytest.raises(module.CoordinatorError, match="share one preserved"):
        _validate_correction4_leaf_authorization(module, fixture)


def test_correction6_uses_distinct_tracked_wave_and_finalizer_authority_paths(
    tmp_path, monkeypatch,
):
    module = _load()
    assert module.LEAF_WAVE_AUTHORIZATION_PATH.name == (
        "e4_pl_s3_v2_stage4a_leaf_wave_authorization.json"
    )
    assert module.EXECUTION_AUTHORIZATION_PATH.name == (
        "e4_pl_s3_v2_stage4a_execution_authorization.json"
    )
    assert module.LEAF_WAVE_AUTHORIZATION_PATH != module.EXECUTION_AUTHORIZATION_PATH
    wrong_path = (tmp_path / "untracked-authorization.json").resolve()
    raw = module.canonical_bytes({"schema": "synthetic"})
    wrong_path.write_bytes(raw)
    with pytest.raises(module.CoordinatorError, match="finalizer.*path differs"):
        module.validate_authorization(
            wrong_path,
            contract_path=(tmp_path / "contract.json").resolve(),
            contract_raw=module.canonical_bytes({"contract": True}),
            execution_mode="leaf-finalizer",
        )
    with pytest.raises(module.CoordinatorError, match="wave authorization path differs"):
        module._validate_leaf_wave_authorization_v5(
            path=wrong_path,
            value={"schema": "synthetic"},
            raw=raw,
            contract_path=(tmp_path / "contract.json").resolve(),
            contract_raw=module.canonical_bytes({"contract": True}),
            selected_wave_index=1,
            selected_plan_sha256="A" * 64,
            selected_leaf_catalog_sha256="B" * 64,
            selected_manifest_sha256="C" * 64,
            selected_result_path=(tmp_path / "result.json").resolve(),
        )
    tracked_wave_path = (tmp_path / "tracked-wave-authorization.json").resolve()
    tracked_wave_path.write_bytes(
        module.canonical_bytes({"schema": module.AUTHORIZATION_SCHEMA})
    )
    monkeypatch.setattr(module, "LEAF_WAVE_AUTHORIZATION_PATH", tracked_wave_path)
    with pytest.raises(module.CoordinatorError, match="wave authorization schema differs"):
        module.validate_authorization(
            tracked_wave_path,
            contract_path=(tmp_path / "contract.json").resolve(),
            contract_raw=module.canonical_bytes({"contract": True}),
            execution_mode="leaf-wave",
            leaf_wave_index=1,
            plan_sha256="A" * 64,
            leaf_catalog_sha256="B" * 64,
            leaf_wave_manifest_sha256="C" * 64,
            leaf_wave_result_path=(tmp_path / "result.json").resolve(),
        )


def test_correction6_authorization_rejects_legacy_before_file_access(
    tmp_path, monkeypatch,
):
    module = _load()
    monkeypatch.setattr(
        module,
        "strict_json_load",
        lambda _path: pytest.fail("authorization file was accessed"),
    )
    with pytest.raises(module.CoordinatorError, match="not authorized by correction 6"):
        module.validate_authorization(
            tmp_path / "missing-authorization.json",
            contract_path=(tmp_path / "contract.json").resolve(),
            contract_raw=module.canonical_bytes({"contract": True}),
            execution_mode="legacy",
        )


def test_correction4_catalog_rejects_duplicate_plan_and_catalog_mutations():
    module = _load()
    plan, plan_raw = _correction4_plan(module)
    candidate_authority, _archive = _correction4_candidate(module)
    duplicate = copy.deepcopy(plan)
    duplicate["shards"][0]["records"][1] = copy.deepcopy(
        duplicate["shards"][0]["records"][0]
    )
    core = dict(duplicate["shards"][0])
    core.pop("assignment_sha256")
    duplicate["shards"][0]["assignment_sha256"] = module.sha256(
        module.canonical_bytes(core)
    )
    with pytest.raises(module.CoordinatorError, match="record coverage"):
        module.build_stage4a_leaf_catalog(
            duplicate, module.canonical_bytes(duplicate), **candidate_authority
        )

    catalog = module.build_stage4a_leaf_catalog(
        plan, plan_raw, **candidate_authority
    )
    reordered = copy.deepcopy(catalog)
    reordered["leaves"][0], reordered["leaves"][1] = (
        reordered["leaves"][1],
        reordered["leaves"][0],
    )
    with pytest.raises(module.CoordinatorError, match="member identity"):
        module.build_stage4a_leaf_wave_catalog(reordered)
    rebound = copy.deepcopy(catalog)
    rebound["leaves"][0]["leaf_assignment_sha256"] = "A" * 64
    with pytest.raises(module.CoordinatorError, match="member identity"):
        module.build_stage4a_leaf_wave_catalog(rebound)


def test_correction4_union_validates_all_leaves_and_reconstructs_legacy_proofs(
    tmp_path, monkeypatch,
):
    module = _load()
    (
        plan, plan_raw, catalog, union, union_path, candidate_authority,
        contract, contract_path, contract_raw,
    ) = _correction4_union(
        module, tmp_path, monkeypatch
    )
    validated = module.validate_stage4a_leaf_union(
        union_path,
        catalog=catalog,
        plan=plan,
        plan_raw=plan_raw,
        candidate_authority=candidate_authority,
        contract=contract,
        contract_path=contract_path,
        contract_raw=contract_raw,
        allowed_root=tmp_path,
    )
    assert len(validated["proofs"]) == 81
    assert module.sha256(validated["union_raw"]) == module.sha256(
        module.canonical_bytes(union)
    )
    documents = module.reconstruct_stage4a_diagonal_documents(
        plan, plan_raw, validated
    )
    assert list(documents) == list(module.EXPECTED_SHARDS)
    for shard, document in zip(plan["shards"], documents.values()):
        assert document["record_count"] == 27
        assert document["record_ids"] == [
            member["record_id"] for member in shard["records"]
        ]
        assert document["schema"] == module.DIAGONAL_SCIENTIFIC_SCHEMA
        assert len(document["scientific_payload"]["classifying_records"]) == 27
        assert document["scientific_payload"]["v1_comparator_diagnostics"] == []
        assert (
            document["scientific_payload"]["v1_comparator_disposition"]
            == module.LEAF_V1_DISPOSITION
        )
        assert document["scientific_payload_sha256"] == module.sha256(
            module.canonical_bytes(document["scientific_payload"])
        )
    bindings = module.publish_stage4a_diagonal_documents(documents, tmp_path)
    assert list(bindings) == list(module.EXPECTED_SHARDS)
    for binding in bindings.values():
        proof_path = Path(binding["proof_path"])
        assert proof_path.is_file()
        assert binding["proof_sha256"] == module.sha256(proof_path.read_bytes())


def test_correction4_union_rejects_missing_alias_hash_and_noncanonical_mutations(
    tmp_path, monkeypatch,
):
    module = _load()
    (
        plan, plan_raw, catalog, union, union_path, candidate_authority,
        contract, contract_path, contract_raw,
    ) = _correction4_union(
        module, tmp_path, monkeypatch
    )
    correction5_union = copy.deepcopy(union)
    correction5_union["schema"] = "anysolver.e4-pl-s3-v2-stage4a-leaf-union-v2"
    union_path.write_bytes(module.canonical_bytes(correction5_union))
    with pytest.raises(module.CoordinatorError, match="identity differs"):
        module.validate_stage4a_leaf_union(
            union_path,
            catalog=catalog,
            plan=plan,
            plan_raw=plan_raw,
            candidate_authority=candidate_authority,
            contract=contract,
            contract_path=contract_path,
            contract_raw=contract_raw,
            allowed_root=tmp_path,
        )
    mutated = copy.deepcopy(union)
    mutated["proofs"][0]["sha256"] = "F" * 64
    union_path.unlink()
    union_path.write_bytes(module.canonical_bytes(mutated))
    with pytest.raises(module.CoordinatorError, match="bounded wave receipts"):
        module.validate_stage4a_leaf_union(
            union_path,
            catalog=catalog,
            plan=plan,
            plan_raw=plan_raw,
            candidate_authority=candidate_authority,
            contract=contract,
            contract_path=contract_path,
            contract_raw=contract_raw,
            allowed_root=tmp_path,
        )
    union_path.write_bytes(module.canonical_bytes(union) + b" ")
    with pytest.raises(module.CoordinatorError, match="not canonical"):
        module.validate_stage4a_leaf_union(
            union_path,
            catalog=catalog,
            plan=plan,
            plan_raw=plan_raw,
            candidate_authority=candidate_authority,
            contract=contract,
            contract_path=contract_path,
            contract_raw=contract_raw,
            allowed_root=tmp_path,
        )

    missing_receipt = copy.deepcopy(union)
    missing_receipt["wave_receipts"].pop()
    union_path.write_bytes(module.canonical_bytes(missing_receipt))
    with pytest.raises(module.CoordinatorError, match="receipt coverage"):
        module.validate_stage4a_leaf_union(
            union_path,
            catalog=catalog,
            plan=plan,
            plan_raw=plan_raw,
            candidate_authority=candidate_authority,
            contract=contract,
            contract_path=contract_path,
            contract_raw=contract_raw,
            allowed_root=tmp_path,
        )


def test_correction4_union_classifies_from_captured_bytes_not_reread_path(
    tmp_path, monkeypatch,
):
    module = _load()
    (
        plan, plan_raw, catalog, union, union_path, candidate_authority,
        contract, contract_path, contract_raw,
    ) = _correction4_union(module, tmp_path, monkeypatch)
    captured = module.canonical_bytes(union)
    union_path.write_bytes(module.canonical_bytes({"substituted": True}))
    validated = module.validate_stage4a_leaf_union(
        union_path,
        catalog=catalog,
        plan=plan,
        plan_raw=plan_raw,
        candidate_authority=candidate_authority,
        contract=contract,
        contract_path=contract_path,
        contract_raw=contract_raw,
        allowed_root=tmp_path,
        frozen_union_raw=captured,
    )
    assert validated["union_raw"] == captured
    with pytest.raises(module.CoordinatorError):
        module.validate_stage4a_leaf_union(
            union_path,
            catalog=catalog,
            plan=plan,
            plan_raw=plan_raw,
            candidate_authority=candidate_authority,
            contract=contract,
            contract_path=contract_path,
            contract_raw=contract_raw,
            allowed_root=tmp_path,
        )


def test_correction4_leaf_outputs_cannot_escape_cycle_root(tmp_path):
    module = _load()
    root = (tmp_path / "cycle").resolve()
    root.mkdir()
    contained = module._contained_leaf_output(
        (root / "nested" / "proof.json").resolve(), root, "synthetic output"
    )
    assert contained == (root / "nested" / "proof.json").resolve()
    with pytest.raises(module.CoordinatorError, match="escapes"):
        module._contained_leaf_output(
            (tmp_path / "outside.json").resolve(), root, "synthetic output"
        )


@pytest.mark.parametrize("final_dependency_valid", [True, False])
def test_correction5_guarded_wave_revalidates_before_receipt(
    tmp_path, monkeypatch, final_dependency_valid
):
    module = _load()
    output_root = (tmp_path / "cycle").resolve()
    wave_root = output_root / "leaf-waves" / "wave-01"
    wave_root.mkdir(parents=True)
    receipt_path = wave_root / "receipt.json"
    result_path = wave_root / "bounded-result.json"
    contract_path = (tmp_path / "contract.json").resolve()
    authorization_path = (tmp_path / "authorization.json").resolve()
    contract_raw = module.canonical_bytes({"contract": True})
    authorization_raw = module.canonical_bytes({"authorization": True})
    contract_path.write_bytes(contract_raw)
    authorization_path.write_bytes(authorization_raw)
    manifest_path = wave_root / "manifest.json"
    manifest_raw = module.canonical_bytes({"manifest": True})
    manifest_path.write_bytes(manifest_raw)
    plan_raw = module.canonical_bytes({"plan": True})
    catalog = {"catalog": True}
    candidate_authority = {
        "candidate_archive_sha256": "A" * 64,
        "candidate_commit": "a" * 40,
        "candidate_tree": "b" * 40,
        "producer_program_sha256": "B" * 64,
    }
    cycle = {
        "candidate_authority": candidate_authority,
        "catalog": catalog,
        "manifest_path": manifest_path,
        "plan_raw": plan_raw,
        "wave": {"wave_id": "S3_V2_FLAT_4A_WAVE_01"},
        "wave_root": wave_root,
    }
    authorization = {
        "execution_paths": {
            "aggregate_path": str(receipt_path),
            "output_root": str(output_root),
            "python_executable": str(Path(sys.executable).resolve()),
        }
    }

    monkeypatch.setattr(
        module.sys,
        "flags",
        type("Flags", (), {"isolated": 1, "dont_write_bytecode": 1})(),
    )
    monkeypatch.setattr(module.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(
        module, "validate_contract", lambda _path: ({"contract": True}, contract_raw)
    )
    monkeypatch.setattr(
        module,
        "validate_authorization",
        lambda *_args, **_kwargs: (authorization, authorization_raw),
    )
    monkeypatch.setattr(
        module, "validate_resource_execution_state", lambda *_args, **_kwargs: None
    )
    cycle_validation_calls = 0

    def validate_cycle(**_kwargs):
        nonlocal cycle_validation_calls
        cycle_validation_calls += 1
        if cycle_validation_calls == 2 and not final_dependency_valid:
            raise module.CoordinatorError(
                "Stage 4A dependency source graph differs after execution"
            )
        return cycle

    monkeypatch.setattr(module, "_validate_stage4a_leaf_cycle", validate_cycle)

    class Bounded:
        @staticmethod
        def run_wave(_manifest_path, output_path):
            result = {"terminal": "COMPLETED"}
            output_path.write_bytes(module.canonical_bytes(result))
            return result

    monkeypatch.setattr(module, "_load_module", lambda *_args, **_kwargs: Bounded)
    monkeypatch.setattr(
        module, "_producer_process_trees_proven_terminal", lambda _result: True
    )
    monkeypatch.setattr(
        module,
        "_validate_stage4a_leaf_wave_result",
        lambda path, _cycle: (
            {"terminal": "COMPLETED"},
            path.read_bytes(),
            [],
        ),
    )
    expected_receipt = {
        "nested_path_accepted": True,
        "terminal": "COMPLETED",
    }
    monkeypatch.setattr(
        module, "_stage4a_leaf_wave_receipt", lambda **_kwargs: expected_receipt
    )

    class Guard:
        def bind_evidence(self, **_kwargs):
            return None

        def bind_producer_result(self, _path):
            return None

        def mark_process_phase_active(self):
            return None

        def mark_process_phase_terminal(self, *, proven):
            assert proven is True

    made = module._run_stage4a_leaf_wave_guarded(
        contract_path,
        authorization_path,
        output_root,
        receipt_path,
        result_path,
        wave_index=1,
        plan_sha256=module.sha256(plan_raw),
        leaf_catalog_sha256=module.sha256(module.canonical_bytes(catalog)),
        leaf_wave_manifest_sha256=module.sha256(manifest_raw),
        wall_guard=Guard(),
    )
    assert cycle_validation_calls == 2
    assert receipt_path.parent == wave_root
    if final_dependency_valid:
        assert made == expected_receipt
        assert module.strict_json_load(receipt_path)[0] == expected_receipt
    else:
        assert made["terminal"] == module.BLOCKED
        assert made["formal_failures"] == ["PRODUCER_WAVE_NOT_COMPLETED"]
        assert module.strict_json_load(receipt_path)[0] == made


def test_correction4_reconstruction_rejects_protocol_disagreement(
    tmp_path, monkeypatch
):
    module = _load()
    (
        plan, plan_raw, catalog, _union, union_path, candidate_authority,
        contract, contract_path, contract_raw,
    ) = _correction4_union(
        module, tmp_path, monkeypatch, protocol_mutation_index=80
    )
    validated = module.validate_stage4a_leaf_union(
        union_path,
        catalog=catalog,
        plan=plan,
        plan_raw=plan_raw,
        candidate_authority=candidate_authority,
        contract=contract,
        contract_path=contract_path,
        contract_raw=contract_raw,
        allowed_root=tmp_path,
    )
    with pytest.raises(module.CoordinatorError, match="protocols disagree"):
        module.reconstruct_stage4a_diagonal_documents(plan, plan_raw, validated)


def test_correction4_finalizer_cli_is_separate_and_strictly_sub_30_minutes(
    tmp_path, monkeypatch
):
    module = _load()
    assert 0 < module.LEAF_FINALIZER_WALL_SECONDS < 30 * 60
    with pytest.raises(SystemExit):
        module._parser().parse_args(
            [
                "--run-stage4a",
                "--finalize-leaf-union",
                "--contract",
                "contract.json",
                "--output-root",
                str(tmp_path),
            ]
        )


def test_correction4_prepare_wave_receipt_and_union_cli_paths_are_wired(
    tmp_path, monkeypatch,
):
    module = _load()
    observed = {}
    monkeypatch.setattr(
        module,
        "run_prepare_stage4a_leaf_cycle",
        lambda contract, root: observed.setdefault("prepare", (contract, root)),
    )
    assert module.main(
        [
            "--prepare-leaf-cycle",
            "--contract",
            str(tmp_path / "contract.json"),
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    monkeypatch.setattr(
        module,
        "run_stage4a_leaf_wave",
        lambda *args, **kwargs: observed.setdefault(
            "wave", (args, kwargs)
        ) and {"terminal": "COMPLETED"},
    )
    assert module.main(
        [
            "--run-leaf-wave",
            "--contract",
            str(tmp_path / "contract.json"),
            "--authorization",
            str(tmp_path / "authorization.json"),
            "--output-root",
            str(tmp_path),
            "--aggregate",
            str(tmp_path / "receipt.json"),
            "--leaf-wave-index",
            "7",
            "--plan-sha256",
            "A" * 64,
            "--leaf-catalog-sha256",
            "B" * 64,
            "--leaf-wave-manifest-sha256",
            "C" * 64,
            "--leaf-wave-result",
            str(tmp_path / "result.json"),
        ]
    ) == 0
    assert observed["wave"][1]["wave_index"] == 7
    assert observed["wave"][1]["leaf_wave_manifest_sha256"] == "C" * 64
    monkeypatch.setattr(
        module,
        "run_assemble_stage4a_leaf_union",
        lambda *args: observed.setdefault("union", args) or {},
    )
    assert module.main(
        [
            "--assemble-leaf-union",
            "--contract",
            str(tmp_path / "contract.json"),
            "--authorization",
            str(tmp_path / "authorization.json"),
            "--output-root",
            str(tmp_path),
            "--aggregate",
            str(tmp_path / "leaf-union.json"),
        ]
    ) == 0
    assert observed["union"][-1] == (tmp_path / "leaf-union.json").resolve()
    observed = {}

    def fake_finalizer(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"terminal": module.PASS}

    monkeypatch.setattr(module, "run_stage4a_leaf_finalizer", fake_finalizer)
    code = module.main(
        [
            "--finalize-leaf-union",
            "--contract",
            str(tmp_path / "contract.json"),
            "--authorization",
            str(tmp_path / "authorization.json"),
            "--output-root",
            str(tmp_path),
            "--aggregate",
            str(tmp_path / "aggregate.json"),
            "--plan",
            str(tmp_path / "plan.json"),
            "--leaf-union",
            str(tmp_path / "union.json"),
            "--plan-sha256",
            "A" * 64,
            "--leaf-union-sha256",
            "B" * 64,
        ]
    )
    assert code == 0
    assert observed["args"][2] == (tmp_path / "plan.json").resolve()
    with pytest.raises(module.CoordinatorError, match="not authorized by correction 6"):
        module.main(
            [
                "--run-stage4a",
                "--contract",
                str(tmp_path / "contract.json"),
                "--authorization",
                str(tmp_path / "authorization.json"),
                "--output-root",
                str(tmp_path),
                "--aggregate",
                str(tmp_path / "aggregate.json"),
                "--plan",
                str(tmp_path / "plan.json"),
            ]
        )


def test_correction4_guarded_finalizer_feeds_reconstructed_union_to_legacy_checker(
    tmp_path, monkeypatch
):
    module = _load()
    (
        plan, plan_raw, catalog, union, union_path, candidate_authority,
        contract, contract_path, contract_raw,
    ) = _correction4_union(
        module, tmp_path, monkeypatch
    )
    plan_path = (tmp_path / "phase4a-plan.json").resolve()
    plan_path.write_bytes(plan_raw)
    authorization_path = (tmp_path / "authorization.json").resolve()
    authorization_path.write_bytes(b"authorization")
    aggregate_path = (tmp_path / "aggregate.json").resolve()
    authorization_raw = b"authorization-raw"
    authorization = {
        "execution_paths": {
            "aggregate_path": str(aggregate_path),
            "output_root": str(tmp_path.resolve()),
            "python_executable": str(Path(sys.executable).resolve()),
        }
    }
    authorization_calls = []

    def fake_validate_authorization(path, **kwargs):
        authorization_calls.append((path, kwargs))
        return authorization, authorization_raw

    monkeypatch.setattr(module.sys, "flags", type("Flags", (), {
        "isolated": 1,
        "dont_write_bytecode": 1,
    })())
    monkeypatch.setattr(module.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(module, "validate_contract", lambda _path: (contract, contract_raw))
    monkeypatch.setattr(module, "validate_authorization", fake_validate_authorization)
    monkeypatch.setattr(module, "validate_resource_execution_state", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_validate_stage4a_plan_raw",
        lambda _raw, **_kwargs: (plan, plan_raw),
    )
    validated = module.validate_stage4a_leaf_union(
        union_path,
        catalog=catalog,
        plan=plan,
        plan_raw=plan_raw,
        candidate_authority=candidate_authority,
        contract=contract,
        contract_path=contract_path,
        contract_raw=contract_raw,
        allowed_root=tmp_path,
    )
    monkeypatch.setattr(module, "validate_stage4a_leaf_union", lambda *_a, **_k: validated)
    monkeypatch.setattr(module, "_load_module", lambda *_a, **_k: object())
    checker_calls = []

    def fake_checker_phase(**kwargs):
        checker_calls.append(kwargs)
        return [[{"replica": 1}], [{"replica": 2}]]

    monkeypatch.setattr(module, "_run_checker_phase", fake_checker_phase)
    aggregate_calls = []

    def fake_aggregate(replicas, **kwargs):
        aggregate_calls.append((replicas, kwargs))
        return {"terminal": module.PASS}

    monkeypatch.setattr(module, "aggregate_checker_results", fake_aggregate)

    class Guard:
        work_deadline = time.monotonic() + 60

        def bind_evidence(self, **_kwargs):
            return None

        def bind_producer_result(self, _path):
            return None

    aggregate = module._run_stage4a_leaf_finalizer_guarded(
        contract_path,
        authorization_path,
        plan_path,
        union_path,
        tmp_path.resolve(),
        aggregate_path,
        Guard(),
        expected_plan_sha256=module.sha256(plan_raw),
        expected_leaf_union_sha256=module.sha256(module.canonical_bytes(union)),
    )
    assert aggregate == {"terminal": module.PASS}
    assert len(authorization_calls) == 2
    assert all(
        call[1]["execution_mode"] == "leaf-finalizer"
        and call[1]["plan_path"] == plan_path
        and call[1]["leaf_union_path"] == union_path
        for call in authorization_calls
    )
    assert len(checker_calls) == 1
    assert list(checker_calls[0]["proofs"]) == list(module.EXPECTED_SHARDS)
    assert len(aggregate_calls) == 1
    assert aggregate_calls[0][1]["producer_result_sha256"] == module.sha256(
        module.canonical_bytes(union)
    )
    assert aggregate_path.read_bytes() == module.canonical_bytes(aggregate)
