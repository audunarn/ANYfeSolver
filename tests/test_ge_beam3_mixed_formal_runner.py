from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "docs/reference_cases/ge_beam3_mixed_formal_runner.py"


def _load_runner(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


template = _load_runner(RUNNER_PATH, "ge_beam3_mixed_formal_runner_template")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("GIT_")
            },
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    return completed.stdout.strip()


def _fake_producer_source() -> str:
    return """from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'src'))
import anyfesolver
import numpy
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
started=time.monotonic_ns();time.sleep(0.12);ended=time.monotonic_ns()
(a.output.parent/'producer-timing.json').write_text(json.dumps({'start':started,'end':ended}),encoding='utf-8')
payload={'anyfesolver_value':anyfesolver.VALUE,'schema':'fake-proof-v1','threads':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
with a.output.open('xb') as stream:stream.write((json.dumps(payload,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True)+'\\n').encode('ascii'))
"""


def _fake_checker_source() -> str:
    return f"""from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy
p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.proof.read_bytes();started=time.monotonic_ns();time.sleep(0.12);ended=time.monotonic_ns()
(a.output.parent/'checker-timing.json').write_text(json.dumps({{'start':started,'end':ended}}),encoding='utf-8')
case_ids={list(template.CASE_ORDER)!r}
metric_keys={sorted(template.CASE_METRIC_KEYS)!r}
predicate_keys={sorted(template.CASE_EVIDENCE_PREDICATES | template.CASE_SCIENTIFIC_PREDICATES)!r}
coverage_keys={sorted(template.COVERAGE_EVIDENCE_PREDICATES | template.COVERAGE_SCIENTIFIC_PREDICATES)!r}
payload={{'authority_bindings':{{key:True for key in {sorted(template.CHECK_BINDING_KEYS)!r}}},'candidate_id':{template.CANDIDATE_ID!r},'cases':[{{'case_id':case_id,'metrics':{{key:'0' for key in metric_keys}},'predicates':{{key:True for key in predicate_keys}},'registered_input_sha256':'A'*64}} for case_id in case_ids],'checker_import_boundary':{{'anyfesolver':False,'legacy_beam':False,'production_ad':False,'v1_beam':False}},'coverage':{{key:True for key in coverage_keys}},'reversal':{{'energy':True,'residual':True,'tangent':True}},'schema':{template.CHECK_SCHEMA!r},'terminal':'NONCLASSIFYING_FINITE_GATE_PASS'}}
with a.output.open('xb') as stream:stream.write((json.dumps(payload,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True)+'\\n').encode('ascii'))
"""


@dataclass
class Harness:
    authority_check_1: Path
    authority_check_2: Path
    external: Path
    harness_commit: str
    harness_subject: str
    harness_tree: str
    mechanics_commit: str
    mechanics_tree: str
    repository: Path
    runner: object


def _make_harness(tmp_path: Path) -> Harness:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Formal Runner Test")
    _git(repository, "config", "user.email", "formal-runner@example.invalid")
    _git(repository, "config", "core.autocrlf", "false")
    _write(repository / "src/anyfesolver/__init__.py", 'VALUE="materialized-candidate"\n')
    _write(repository / ".gitattributes", "explicit-crlf.txt text eol=crlf\n")
    _write(repository / "explicit-crlf.txt", "first\nsecond\n")
    _git(
        repository,
        "add",
        "src/anyfesolver/__init__.py",
        ".gitattributes",
        "explicit-crlf.txt",
    )
    _git(repository, "commit", "-m", "test: synthetic base")
    base_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "commit", "--allow-empty", "-m", "test: mechanics origin")
    mechanics_commit = _git(repository, "rev-parse", "HEAD")
    mechanics_tree = _git(repository, "show", "-s", "--format=%T", "HEAD")

    for relative in template.CANDIDATE_PATHS:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "docs/reference_cases/ge_beam3_mixed_formal_runner.py":
            path.write_bytes(RUNNER_PATH.read_bytes())
        elif relative == "tests/test_ge_beam3_mixed_formal_runner.py":
            path.write_bytes(Path(__file__).read_bytes())
        elif relative == "docs/reference_cases/ge_beam3_mixed_finite_producer.py":
            _write(path, _fake_producer_source())
        elif relative == "docs/reference_cases/ge_beam3_mixed_finite_checker.py":
            _write(path, _fake_checker_source())
        elif path.suffix == ".json":
            _write(path, "{}\n")
        else:
            _write(path, "# synthetic committed candidate input\n")
    _git(repository, "add", "--", *template.CANDIDATE_PATHS)
    harness_subject = "test: freeze exact twenty-path harness"
    _git(repository, "commit", "-m", harness_subject)
    harness_commit = _git(repository, "rev-parse", "HEAD")
    harness_tree = _git(repository, "show", "-s", "--format=%T", "HEAD")
    assert tuple(
        sorted(
            entry
            for entry in _git(
                repository, "diff", "--name-only", f"{base_commit}...{harness_commit}"
            ).splitlines()
            if entry
        )
    ) == template.CANDIDATE_PATHS

    runner_path = repository / "docs/reference_cases/ge_beam3_mixed_formal_runner.py"
    runner = _load_runner(runner_path, f"synthetic_formal_runner_{uuid.uuid4().hex}")
    runner.BASE_COMMIT = base_commit
    runner.MECHANICS_COMMIT = mechanics_commit
    runner.MECHANICS_TREE = mechanics_tree
    external = tmp_path / "external"
    external.mkdir()
    manifest = runner.candidate_manifest(repository, harness_commit)
    by_path = {row["path"]: row for row in manifest}
    booleans = {
        group: {key: True for key in sorted(keys)}
        for group, keys in runner.AUTHORITY_BOOLEAN_GROUP_FIELDS.items()
    }
    authority_check = {
        **booleans,
        "candidate_extent": list(runner.CANDIDATE_PATHS),
        "candidate_extent_components": {
            "committed_since_base": list(runner.CANDIDATE_PATHS),
            "index": [],
            "untracked": [],
            "working_tree": [],
        },
        "candidate_id": runner.CANDIDATE_ID,
        "input_hashes": {
            Path(path).name: by_path[path]["sha256"]
            for path in runner.SOURCE_INPUT_PATHS
        },
        "input_git_blob_oids": {
            Path(path).name: by_path[path]["git_blob_oid"]
            for path in runner.SOURCE_INPUT_PATHS
        },
        "input_validation_checks": {
            name: {key: True for key in sorted(runner.AUTHORITY_INPUT_VALIDATION_FIELDS)}
            for name in runner.AUTHORITY_INPUT_NAMES
        },
        "input_working_hashes": {
            Path(path).name: by_path[path]["sha256"]
            for path in runner.SOURCE_INPUT_PATHS
        },
        "pytest_diagnostic_exclusion_policy": runner.PYTEST_DIAGNOSTIC_EXCLUSION_POLICY,
        "schema": runner.AUTHORITY_CHECK_SCHEMA,
        "source_verification_mode": "EXTERNAL_FILES_HASHED",
        "study_id": runner.STUDY_ID,
        "terminal": "AUTHORITY_CHECK_ONLY_PASS",
    }
    check_payload = runner._canonical_bytes(authority_check)
    check_1 = external / "authority-check-1.json"
    check_2 = external / "authority-check-2.json"
    check_1.write_bytes(check_payload)
    check_2.write_bytes(check_payload)
    return Harness(
        authority_check_1=check_1,
        authority_check_2=check_2,
        external=external,
        harness_commit=harness_commit,
        harness_subject=harness_subject,
        harness_tree=harness_tree,
        mechanics_commit=mechanics_commit,
        mechanics_tree=mechanics_tree,
        repository=repository,
        runner=runner,
    )


def _bound_file(runner, path: Path, terminal: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "path": str(path.resolve()),
        "sha256": runner._sha256_bytes(payload),
        "terminal": terminal,
    }


def _common_authority(
    harness: Harness,
    *,
    mode: str,
    request_id: str,
    attempt_id: str,
    registry: Path,
) -> dict[str, object]:
    runner = harness.runner
    manifest = runner.candidate_manifest(harness.repository, harness.harness_commit)
    return {
        "candidate_id": runner.CANDIDATE_ID,
        "cycle_count": 2,
        "execution_bounds": runner.FROZEN_BOUNDS.authority_value(),
        "harness": {
            "candidate_manifest": manifest,
            "commit": harness.harness_commit,
            "parent": harness.mechanics_commit,
            "subject": harness.harness_subject,
            "tree": harness.harness_tree,
        },
        "mechanics_origin": {
            "commit": harness.mechanics_commit,
            "parent": runner.BASE_COMMIT,
            "tree": harness.mechanics_tree,
        },
        "mode": mode,
        "program_bindings": [row for row in manifest if row["role"] == "PROGRAM"],
        "python_runtime": runner._runtime_binding(),
        "request": {
            "attempt_id": attempt_id,
            "no_retry": True,
            "one_time": True,
            "registry_root": str(registry.resolve()),
            "request_id": request_id,
        },
        "scientific_execution_authorized": mode == runner.FORMAL_MODE,
        "source_input_bindings": [
            row for row in manifest if row["role"] == "SOURCE_INPUT"
        ],
        "study_id": runner.STUDY_ID,
    }


def _rehearsal_authority(
    harness: Harness,
    *,
    request_id: str = "1" * 32,
    attempt_id: str = "2" * 32,
) -> Path:
    runner = harness.runner
    registry = harness.external / "registry"
    registry.mkdir(exist_ok=True)
    record = {
        **_common_authority(
            harness,
            mode=runner.REHEARSAL_MODE,
            request_id=request_id,
            attempt_id=attempt_id,
            registry=registry,
        ),
        "authority_check": _bound_file(
            runner, harness.authority_check_1, "AUTHORITY_CHECK_ONLY_PASS"
        ),
        "schema": runner.REHEARSAL_AUTHORITY_SCHEMA,
        "terminal": runner.REHEARSAL_AUTHORITY_TERMINAL,
    }
    path = harness.external / f"rehearsal-authority-{request_id}.json"
    path.write_bytes(runner._canonical_bytes(record))
    return path


def _run_rehearsal(harness: Harness) -> tuple[Path, dict[str, object], Path]:
    runner = harness.runner
    authority = _rehearsal_authority(harness)
    aggregate_path = harness.external / "rehearsal-aggregate.json"
    work_root = harness.external / "rehearsal-work"
    aggregate = runner.run_bounded(
        authority_path=authority,
        output_path=aggregate_path,
        work_root=work_root,
        mode=runner.REHEARSAL_MODE,
        repository=harness.repository,
    )
    return aggregate_path, aggregate, work_root


def _overlap(first: dict[str, int], second: dict[str, int]) -> bool:
    return max(first["start"], second["start"]) < min(first["end"], second["end"])


def _make_formal_overlay(
    harness: Harness,
    rehearsal_aggregate: Path,
    *,
    mutate_review: bool = False,
) -> Path:
    runner = harness.runner
    raw_preauthorization = {
        "authority_checks": [
            _bound_file(runner, harness.authority_check_1, "AUTHORITY_CHECK_ONLY_PASS"),
            _bound_file(runner, harness.authority_check_2, "AUTHORITY_CHECK_ONLY_PASS"),
        ],
        "rehearsal_aggregate": _bound_file(
            runner,
            rehearsal_aggregate,
            "NONCLASSIFYING_GE_BEAM3_MIXED_REHEARSAL_PASS",
        ),
    }
    normalized = runner._validate_preauthorization_records(
        raw_preauthorization,
        harness.external / "future-authority.json",
        runner.candidate_manifest(harness.repository, harness.harness_commit),
        harness_commit=harness.harness_commit,
        harness_tree=harness.harness_tree,
    )
    reviewed_inputs = {
        "candidate_manifest": runner.candidate_manifest(
            harness.repository, harness.harness_commit
        ),
        "preauthorization_records": normalized,
    }
    if mutate_review:
        reviewed_inputs["candidate_manifest"] = []
    review = {
        "findings": [],
        "reviewed_inputs": reviewed_inputs,
        "reviewer_independence": runner.REVIEWER_INDEPENDENCE,
        "schema": runner.REVIEW_SCHEMA,
        "verdict": runner.REVIEW_VERDICT,
    }
    review_path = harness.repository / runner.REVIEW_RELATIVE_PATH
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_payload = runner._canonical_bytes(review)
    review_path.write_bytes(review_payload)
    review_oid = _git(harness.repository, "hash-object", str(review_path))
    registry = harness.external / "registry"
    authority = {
        **_common_authority(
            harness,
            mode=runner.FORMAL_MODE,
            request_id="3" * 32,
            attempt_id="4" * 32,
            registry=registry,
        ),
        "authorization_commit": {
            "exact_paths": list(runner.AUTHORIZATION_PATHS),
            "expected_parent": harness.harness_commit,
            "expected_subject": runner.AUTHORIZATION_SUBJECT,
        },
        "preauthorization_records": raw_preauthorization,
        "review": {
            "git_blob_oid": review_oid,
            "path": runner.REVIEW_RELATIVE_PATH,
            "sha256": runner._sha256_bytes(review_payload),
            "verdict": runner.REVIEW_VERDICT,
        },
        "schema": runner.AUTHORITY_SCHEMA,
        "terminal": runner.FORMAL_AUTHORITY_TERMINAL,
    }
    authority_path = harness.repository / runner.AUTHORITY_RELATIVE_PATH
    authority_path.write_bytes(runner._canonical_bytes(authority))
    _git(
        harness.repository,
        "add",
        "--",
        runner.AUTHORITY_RELATIVE_PATH,
        runner.REVIEW_RELATIVE_PATH,
    )
    _git(harness.repository, "commit", "-m", runner.AUTHORIZATION_SUBJECT)
    return authority_path


def _check_record(runner, *, scientific_pass: bool = True) -> dict[str, object]:
    predicates = {
        key: True
        for key in runner.CASE_EVIDENCE_PREDICATES
        | runner.CASE_SCIENTIFIC_PREDICATES
    }
    if not scientific_pass:
        predicates["energy"] = False
    return {
        "authority_bindings": {key: True for key in runner.CHECK_BINDING_KEYS},
        "candidate_id": runner.CANDIDATE_ID,
        "cases": [
            {
                "case_id": case_id,
                "metrics": {key: "0" for key in runner.CASE_METRIC_KEYS},
                "predicates": dict(predicates),
                "registered_input_sha256": "A" * 64,
            }
            for case_id in runner.CASE_ORDER
        ],
        "checker_import_boundary": {
            "anyfesolver": False,
            "legacy_beam": False,
            "production_ad": False,
            "v1_beam": False,
        },
        "coverage": {
            key: True
            for key in runner.COVERAGE_EVIDENCE_PREDICATES
            | runner.COVERAGE_SCIENTIFIC_PREDICATES
        },
        "reversal": {"energy": True, "residual": True, "tangent": True},
        "schema": runner.CHECK_SCHEMA,
        "terminal": (
            "NONCLASSIFYING_FINITE_GATE_PASS"
            if scientific_pass
            else "NONCLASSIFYING_FINITE_GATE_FINDING"
        ),
    }


def test_authenticated_rehearsal_is_isolated_overlapping_and_one_time(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    aggregate_path, aggregate, work = _run_rehearsal(harness)
    assert aggregate_path.read_bytes() == runner._canonical_bytes(aggregate)
    assert aggregate["terminal"] == "NONCLASSIFYING_GE_BEAM3_MIXED_REHEARSAL_PASS"
    assert aggregate["adjudication"] == {
        "checker_dispositions": ["PASS", "PASS"],
        "finding_groups": [],
        "scientific_finding": False,
    }
    assert all(
        process["tree_drained"] is True
        for role in ("producers", "checkers")
        for process in aggregate["processes"][role]
    )
    assert aggregate["determinism"] == {
        "check_bytes_identical": True,
        "checker_origin_bytes_identical": True,
        "producer_origin_bytes_identical": True,
        "proof_bytes_identical": True,
    }
    assert (work / "materialized-harness/explicit-crlf.txt").read_bytes() == (
        b"first\nsecond\n"
    )
    proof = json.loads((work / "cycle-1/proof.json").read_text(encoding="utf-8"))
    assert set(proof["threads"].values()) == {"1"}
    producer_times = [
        json.loads((work / f"cycle-{cycle}/producer-timing.json").read_text())
        for cycle in (1, 2)
    ]
    checker_times = [
        json.loads((work / f"cycle-{cycle}/checker/checker-timing.json").read_text())
        for cycle in (1, 2)
    ]
    assert _overlap(*producer_times)
    assert _overlap(*checker_times)
    producer_origin = json.loads(
        (work / "cycle-1/producer-origin.json").read_text(encoding="utf-8")
    )
    checker_origin = json.loads(
        (work / "cycle-1/checker/checker-origin.json").read_text(encoding="utf-8")
    )
    assert producer_origin["anyfesolver_origins"]
    assert all(
        "materialized-harness" in path
        for path in producer_origin["anyfesolver_origins"]
    )
    assert checker_origin["anyfesolver_origins"] == []
    assert producer_origin["isolated"] is checker_origin["isolated"] is True

    # Publication recovery is idempotent and does not execute or consume the
    # request again.  A simulated crash-owned global slot is reconciled only
    # after its kernel owner is proven dead, and the authoritative receipt plus
    # raw evidence reconstruct the exact bytes.
    original_aggregate = aggregate_path.read_bytes()
    aggregate_path.unlink()
    validated = runner.validate_execution_authority(
        harness.external / f"rehearsal-authority-{'1' * 32}.json",
        mode=runner.REHEARSAL_MODE,
        repository=harness.repository,
    )
    stale_lock = harness.external / "registry/global-slot.lock"
    released_slot = (
        harness.external
        / "registry/released-slots"
        / f"{'1' * 32}.{'2' * 32}.released-slot"
    )
    released_slot.rename(stale_lock)
    stale_owner_path = stale_lock / "owner.json"
    stale_owner = json.loads(stale_owner_path.read_text(encoding="ascii"))
    stale_owner["owner_pid"] = 2_147_483_647
    stale_owner["owner_process_start"] = "PROVEN_DEAD_SYNTHETIC_PROCESS"
    stale_owner_path.write_bytes(runner._canonical_bytes(stale_owner))
    recovered = runner.run_bounded(
        authority_path=harness.external / f"rehearsal-authority-{'1' * 32}.json",
        output_path=aggregate_path,
        work_root=work,
        mode=runner.REHEARSAL_MODE,
        repository=harness.repository,
    )
    assert runner._canonical_bytes(recovered) == original_aggregate
    assert aggregate_path.read_bytes() == original_aggregate
    assert not stale_lock.exists()
    with pytest.raises(runner.AuthorityError, match="already consumed"):
        runner._acquire_one_time_claim(
            validated,
            output_path=aggregate_path,
            work_root=work,
        )

    with pytest.raises(runner.AuthorityError, match="receipt execution identity"):
        runner.run_bounded(
            authority_path=harness.external
            / f"rehearsal-authority-{'1' * 32}.json",
            output_path=harness.external / "reused-aggregate.json",
            work_root=harness.external / "reused-work",
            mode=runner.REHEARSAL_MODE,
            repository=harness.repository,
        )
    assert not (harness.external / "reused-aggregate.json").exists()
    assert len(list((harness.external / "registry/claims").glob("*.claim.json"))) == 1


def test_full_authority_schema_manifest_runtime_and_clean_tree_fail_closed(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    authority = _rehearsal_authority(harness)
    original = authority.read_bytes()
    record = json.loads(original)
    check_record = json.loads(harness.authority_check_1.read_text(encoding="utf-8"))
    check_record["base_checks"].pop("base_tree")
    harness.authority_check_1.write_bytes(runner._canonical_bytes(check_record))
    record["authority_check"] = _bound_file(
        runner, harness.authority_check_1, "AUTHORITY_CHECK_ONLY_PASS"
    )
    authority.write_bytes(runner._canonical_bytes(record))
    with pytest.raises(runner.AuthorityError, match="base_checks"):
        runner.validate_execution_authority(
            authority, mode=runner.REHEARSAL_MODE, repository=harness.repository
        )

    harness.authority_check_1.write_bytes(harness.authority_check_2.read_bytes())
    authority.write_bytes(original)
    record = json.loads(original)
    record["harness"]["candidate_manifest"][0]["git_blob_oid"] = "0" * 40
    authority.write_bytes(runner._canonical_bytes(record))
    with pytest.raises(runner.AuthorityError, match="manifest"):
        runner.validate_execution_authority(
            authority, mode=runner.REHEARSAL_MODE, repository=harness.repository
        )

    authority.write_bytes(original)
    dirty = harness.repository / "unregistered.txt"
    dirty.write_text("dirty", encoding="utf-8")
    with pytest.raises(runner.AuthorityError, match="fully clean"):
        runner.validate_execution_authority(
            authority, mode=runner.REHEARSAL_MODE, repository=harness.repository
        )
    dirty.unlink()
    record = json.loads(original)
    record["python_runtime"]["sha256"] = "0" * 64
    authority.write_bytes(runner._canonical_bytes(record))
    with pytest.raises(runner.AuthorityError, match="Python executable identity"):
        runner.validate_execution_authority(
            authority, mode=runner.REHEARSAL_MODE, repository=harness.repository
        )


def test_formal_overlay_requires_clean_preauthorization_and_exact_five_key_review(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    rehearsal, _aggregate, _work = _run_rehearsal(harness)
    authority = _make_formal_overlay(harness, rehearsal)
    validated = runner.validate_execution_authority(
        authority, mode=runner.FORMAL_MODE, repository=harness.repository
    )
    assert validated.binding["authorization"]["commit"] == _git(
        harness.repository, "rev-parse", "HEAD"
    )
    assert validated.binding["preauthorization_records"]["rehearsal_aggregate"][
        "terminal"
    ] == "NONCLASSIFYING_GE_BEAM3_MIXED_REHEARSAL_PASS"
    result = runner.run_bounded(
        authority_path=authority,
        output_path=harness.external / "formal-aggregate.json",
        work_root=harness.external / "formal-work",
        mode=runner.FORMAL_MODE,
        repository=harness.repository,
    )
    assert result["terminal"] == "PROVISIONAL_GO_GE_BEAM3_MIXED_STRAIGHT_STATIC_CORE"
    assert result["authority"]["review"]["verdict"] == runner.REVIEW_VERDICT


def test_reviewed_inputs_mismatch_blocks_formal_overlay(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    rehearsal, _aggregate, _work = _run_rehearsal(harness)
    authority = _make_formal_overlay(harness, rehearsal, mutate_review=True)
    with pytest.raises(runner.AuthorityError, match="review contents"):
        runner.validate_execution_authority(
            authority, mode=runner.FORMAL_MODE, repository=harness.repository
        )


def test_checker_schema_distinguishes_evidence_failure_from_scientific_finding(
    tmp_path: Path,
) -> None:
    runner = template
    path = tmp_path / "check.json"
    passing = _check_record(runner)
    path.write_bytes(runner._canonical_bytes(passing))
    process = {"reason": None, "returncode": 0}
    assert runner._validate_check_record(path, process)[0] == "PASS"

    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["authority_bindings"]["programs"] = False
    path.write_bytes(runner._canonical_bytes(malformed))
    assert runner._validate_check_record(path, process)[0] == "MALFORMED_EVIDENCE"

    finding = _check_record(runner, scientific_pass=False)
    path.write_bytes(runner._canonical_bytes(finding))
    assert (
        runner._validate_check_record(path, {"reason": None, "returncode": 2})[0]
        == "SCIENTIFIC_FINDING"
    )

    finding["terminal"] = "NONCLASSIFYING_FINITE_GATE_PASS"
    path.write_bytes(runner._canonical_bytes(finding))
    assert (
        runner._validate_check_record(path, {"reason": None, "returncode": 2})[0]
        == "MALFORMED_EVIDENCE"
    )

    variational = _check_record(runner, scientific_pass=False)
    for case in variational["cases"]:
        case["predicates"]["energy"] = True
    variational["reversal"]["tangent"] = False
    path.write_bytes(runner._canonical_bytes(variational))
    disposition, details = runner._validate_check_record(
        path, {"reason": None, "returncode": 2}
    )
    assert disposition == "SCIENTIFIC_FINDING"
    assert details["finding_groups"] == ["VARIATIONAL_IDENTITY"]
    assert (
        runner._finding_terminal(runner.FORMAL_MODE, details["finding_groups"])
        == "NO_GO_GE_BEAM3_MIXED_VARIATIONAL_IDENTITY"
    )
    assert (
        runner._finding_terminal(
            runner.FORMAL_MODE, ["REFERENCE_LINEAR_ALGEBRA", "FINITE_STATIC"]
        )
        == "NO_GO_GE_BEAM3_MIXED_REFERENCE_LINEAR_ALGEBRA"
    )


def test_global_slot_is_exclusive_before_any_child_launch(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    authority = _rehearsal_authority(
        harness, request_id="5" * 32, attempt_id="6" * 32
    )
    validated = runner.validate_execution_authority(
        authority, mode=runner.REHEARSAL_MODE, repository=harness.repository
    )
    lock = harness.external / "registry/global-slot.lock"
    lock.mkdir()
    owner = runner._owner_record(
        validated,
        output_path=harness.external / "slot-output.json",
        work_root=harness.external / "slot-work",
    )
    (lock / "owner.json").write_bytes(runner._canonical_bytes(owner))
    with pytest.raises(runner.AuthorityError, match="live owner"):
        runner._acquire_one_time_claim(
            validated,
            output_path=harness.external / "slot-output.json",
            work_root=harness.external / "slot-work",
        )
    assert not list((harness.external / "registry/claims").glob("*.claim.json"))


def test_proven_dead_pre_receipt_owner_is_terminalized_without_request_reuse(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    authority = _rehearsal_authority(
        harness, request_id="9" * 32, attempt_id="a" * 32
    )
    validated = runner.validate_execution_authority(
        authority, mode=runner.REHEARSAL_MODE, repository=harness.repository
    )
    output = harness.external / "orphan-output.json"
    work = harness.external / "orphan-work"
    first_claim = runner._acquire_one_time_claim(
        validated,
        output_path=output,
        work_root=work,
    )
    owner = json.loads(first_claim.owner_path.read_text(encoding="ascii"))
    owner["owner_pid"] = 2_147_483_647
    owner["owner_process_start"] = "PROVEN_DEAD_SYNTHETIC_PROCESS"
    first_claim.owner_path.write_bytes(runner._canonical_bytes(owner))

    with pytest.raises(runner.AuthorityError, match="already consumed"):
        runner._acquire_one_time_claim(
            validated,
            output_path=output,
            work_root=work,
        )
    orphan = (
        harness.external
        / "registry/orphaned"
        / f"{'9' * 32}.{'a' * 32}.orphaned.json"
    )
    record = json.loads(orphan.read_text(encoding="ascii"))
    assert record["terminal"] == "NONCLASSIFYING_GE_BEAM3_MIXED_REHEARSAL_BLOCKED"
    assert record["reason"] == "PROVEN_DEAD_OWNER_BEFORE_TERMINAL_RECEIPT"
    assert first_claim.claim_path.exists()
    assert not first_claim.receipt_path.exists()
    assert not first_claim.lock_directory.exists()


def test_publication_recovery_rejects_forged_go_core_against_raw_evidence(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    output, aggregate, work = _run_rehearsal(harness)
    output.unlink()
    receipt = Path(aggregate["consumption"]["receipt"]["path"])
    record = json.loads(receipt.read_text(encoding="ascii"))
    record["aggregate_core"]["cycles"][0]["proof"]["sha256"] = "0" * 64
    receipt.write_bytes(runner._canonical_bytes(record))

    with pytest.raises(runner.AuthorityError, match="raw proof binding"):
        runner.run_bounded(
            authority_path=harness.external
            / f"rehearsal-authority-{'1' * 32}.json",
            output_path=output,
            work_root=work,
            mode=runner.REHEARSAL_MODE,
            repository=harness.repository,
        )
    assert not output.exists()


def test_post_claim_materialization_failure_is_receipted_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _make_harness(tmp_path)
    runner = harness.runner
    authority = _rehearsal_authority(
        harness, request_id="7" * 32, attempt_id="8" * 32
    )

    def fail_materialization(*_args, **_kwargs):
        raise OSError("synthetic materialization failure")

    monkeypatch.setattr(runner, "_materialize_harness", fail_materialization)
    output = harness.external / "blocked-aggregate.json"
    result = runner.run_bounded(
        authority_path=authority,
        output_path=output,
        work_root=harness.external / "blocked-work",
        mode=runner.REHEARSAL_MODE,
        repository=harness.repository,
    )
    assert result["terminal"] == "NONCLASSIFYING_GE_BEAM3_MIXED_REHEARSAL_BLOCKED"
    assert result["failure_class"] == "MATERIALIZATION_OSERROR"
    assert result["processes"] == {"checkers": [], "producers": []}
    assert result["consumption"]["receipt"]["terminal"] == result["terminal"]
    assert output.read_bytes() == runner._canonical_bytes(result)
    assert not (harness.external / "registry/global-slot.lock").exists()


def test_wall_limit_terminates_and_drains_descendant_process_tree(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived.txt"
    script = tmp_path / "tree.py"
    _write(
        script,
        """import subprocess,sys,time
subprocess.Popen([sys.executable,'-c',"import sys,time;from pathlib import Path;time.sleep(1.2);Path(sys.argv[1]).write_text('bad')",sys.argv[1]])
time.sleep(20)
""",
    )
    wave_root = tmp_path / "wave"
    wave_root.mkdir()
    spec = template.ChildSpec(
        role="tree-test",
        cycle=1,
        command=(sys.executable, str(script), str(marker)),
        directory=wave_root / "child",
        watched_output=wave_root / "child/never.json",
    )
    bounds = template.ExecutionBounds(
        child_wall_seconds=0.25,
        complete_wave_wall_seconds=2.0,
        inactivity_seconds=1.0,
        memory_limit_bytes=256 * 1024**2,
        numerical_library_threads=1,
    )
    records = template._run_wave((spec,), bounds=bounds, poll_seconds=0.02)
    assert records[0]["reason"] == "CHILD_WALL_LIMIT"
    assert records[0]["tree_drained"] is True
    time.sleep(1.3)
    assert not marker.exists()


def test_finish_child_closes_job_and_logs_when_job_drain_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class Process:
        returncode = 0

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == 10
            return 0

    class Stream:
        closed = False

        def __init__(self, name: str):
            self.name = name

        def close(self) -> None:
            events.append(f"close-{self.name}")
            self.closed = True

    class Job:
        handle: int | None = 123

        def drain(self) -> None:
            events.append("drain")
            raise RuntimeError("synthetic drain failure")

        def close(self) -> None:
            events.append("close-job")
            self.handle = None

    stdout = Stream("stdout")
    stderr = Stream("stderr")
    job = Job()
    spec = template.ChildSpec(
        role="drain-test",
        cycle=1,
        command=("unused",),
        directory=tmp_path,
        watched_output=tmp_path / "unused.json",
    )
    state = template.ChildState(
        spec=spec,
        process=Process(),
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stdout_stream=stdout,
        stderr_stream=stderr,
        started=0.0,
        last_activity=0.0,
        last_signature=(0, 0, 0),
        job=job,
        known_pids=set(),
    )

    monkeypatch.setattr(template.os, "name", "nt")
    with pytest.raises(RuntimeError, match="synthetic drain failure"):
        template._finish_child(state)
    assert events == ["drain", "close-stdout", "close-stderr", "close-job"]
    assert stdout.closed is True
    assert stderr.closed is True
    assert job.handle is None
    assert state.tree_drained is False


def test_cleanup_attempts_every_child_after_termination_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class Process:
        returncode = None

        def __init__(self, cycle: int):
            self.cycle = cycle

        @staticmethod
        def poll() -> None:
            return None

    def state(cycle: int) -> object:
        spec = template.ChildSpec(
            role="cleanup-test",
            cycle=cycle,
            command=("unused",),
            directory=tmp_path / str(cycle),
            watched_output=tmp_path / str(cycle) / "unused.json",
        )
        return template.ChildState(
            spec=spec,
            process=Process(cycle),
            stdout_path=tmp_path / str(cycle) / "stdout.log",
            stderr_path=tmp_path / str(cycle) / "stderr.log",
            stdout_stream=None,
            stderr_stream=None,
            started=0.0,
            last_activity=0.0,
            last_signature=(0, 0, 0),
            job=None,
            known_pids=set(),
        )

    states = [state(1), state(2)]

    def terminate(item) -> None:
        events.append(f"terminate-{item.spec.cycle}")
        if item.spec.cycle == 1:
            raise RuntimeError("synthetic terminate failure")

    def finish(item) -> None:
        events.append(f"finish-{item.spec.cycle}")

    monkeypatch.setattr(template, "_terminate_tree", terminate)
    monkeypatch.setattr(template, "_finish_child", finish)
    with pytest.raises(RuntimeError, match="synthetic terminate failure"):
        template._cleanup_wave_states(states)
    assert events == ["terminate-1", "terminate-2", "finish-1", "finish-2"]
    assert all(item.forced_reason == "COORDINATOR_EXCEPTION" for item in states)


def test_runner_import_boundary_is_standard_library_only() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__",
        "argparse",
        "ctypes",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "platform",
        "resource",
        "signal",
        "stat",
        "subprocess",
        "sys",
        "time",
        "typing",
    }
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "import numpy" not in text
    assert "import anyfesolver" not in text
    assert "from anyfesolver" not in text
    assert "ge_beam3_mixed_finite_producer import" not in text
    assert "ge_beam3_mixed_finite_checker import" not in text


def test_controlled_git_environment_accepts_only_canonical_line_endings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "line-endings"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Line Ending Test")
    _git(repository, "config", "user.email", "line-ending@example.invalid")
    _git(repository, "config", "core.autocrlf", "false")
    source = repository / "source.py"
    source.write_bytes(b"first\nsecond\n")
    _git(repository, "add", "--", "source.py")
    _git(repository, "commit", "-m", "base")
    source.write_bytes(b"first\r\nsecond\r\n")

    monkeypatch.setenv("GIT_CONFIG_COUNT", "7")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.safecrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    environment = template._git_environment()
    assert "GIT_CONFIG_COUNT" not in environment
    assert "GIT_CONFIG_KEY_0" not in environment
    assert "GIT_CONFIG_VALUE_0" not in environment
    assert template._checkout_has_changes(repository) is False
    assert template._git_bytes(repository, "cat-file", "blob", "HEAD:source.py") == (
        b"first\nsecond\n"
    )

    source.write_bytes(b"first\r\nchanged\r\n")
    assert template._checkout_has_changes(repository) is True
