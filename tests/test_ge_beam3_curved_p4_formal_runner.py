from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
PRODUCER = REFERENCE / "ge_beam3_curved_p4_reference_producer.py"
CHECKER = REFERENCE / "ge_beam3_curved_p4_reference_checker.py"
RUNNER = REFERENCE / "ge_beam3_curved_p4_reference_runner.py"
REFERENCE_MODULE = ROOT / "src" / "anyfesolver" / "ge_beam3_curved_reference.py"
CASES = REFERENCE / "ge_beam3_curved_p4_cases.json"
CONTRACT = REFERENCE / "ge_beam3_curved_p4_contract.json"
PASS = "PROVISIONAL_GO_GE_BEAM3_P4_PRIVATE_CURVED_REFERENCE_CORE"
BLOCKED = "BLOCKED_GE_BEAM3_P4_REFERENCE_PROCESS_OR_EVIDENCE"
NO_GO_REGULARITY = "NO_GO_GE_BEAM3_P4_REFERENCE_REGULARITY"
NO_GO_FRAME = "NO_GO_GE_BEAM3_P4_REFERENCE_FRAME_IDENTITY"
NO_GO_COVARIANCE = "NO_GO_GE_BEAM3_P4_REFERENCE_REVERSAL_OR_OBJECTIVITY"


def _load_script(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _changed_sha256(value: str) -> str:
    return ("0" if value[0] != "0" else "1") + value[1:]


def _file_sha256(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    assert b"\r" not in raw
    return _sha256(raw)


def _authority(runner) -> tuple[dict[str, object], str]:
    manifest = runner._build_authority_manifest(
        ROOT,
        require_clean_inputs=False,
    )
    return manifest, _sha256(_canonical_bytes(manifest))


def _produce(tmp_path: Path):
    runner = _load_script(RUNNER, f"_p4_runner_for_producer_{tmp_path.name}")
    producer = _load_script(PRODUCER, f"_p4_producer_{tmp_path.name}")
    manifest, manifest_sha256 = _authority(runner)
    authority_path = tmp_path / "authority-manifest.json"
    authority_path.write_bytes(_canonical_bytes(manifest))
    proof_path = tmp_path / "proof.json"
    proof = producer.produce(
        REFERENCE_MODULE,
        CASES,
        CONTRACT,
        authority_path,
        proof_path,
        expected_reference_sha256=_file_sha256(REFERENCE_MODULE),
        expected_producer_sha256=_file_sha256(PRODUCER),
        expected_authority_manifest_sha256=manifest_sha256,
    )
    return runner, proof_path, proof, manifest_sha256


def _run_checker(proof: Path, output: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-I", "-B", str(CHECKER), "--proof", str(proof), "--output", str(output)],
        cwd=output.parent,
        capture_output=True,
        check=False,
        timeout=30.0,
    )


def _rewrite_payload(path: Path, value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = _sha256(_canonical_bytes(payload))
    path.write_bytes(_canonical_bytes(payload))


def _complete_cycle(terminal: str) -> dict[str, object]:
    return {
        "counts": {"covered_obligations": 26},
        "status": "COMPLETE",
        "terminal": terminal,
    }


@pytest.fixture
def external_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(prefix="anyfesolver-p4-disposable-") as directory:
        path = Path(directory).resolve()
        with pytest.raises(ValueError):
            path.relative_to(ROOT)
        yield path


def test_independent_checker_imports_no_anyfesolver_or_mechanics_modules() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imported)
    assert not any(token in name.lower() for name in imported for token in ("element", "mechanics", "corotational", "recovery"))


def test_authority_manifest_binds_all_programs_commits_and_protected_blobs() -> None:
    runner = _load_script(RUNNER, "_p4_runner_authority_test")
    manifest, digest = _authority(runner)
    assert len(digest) == 64
    assert [row["role"] for row in manifest["commits"]] == [
        "PREREGISTRATION",
        "REFERENCE_CORE_INITIAL",
        "REFERENCE_CORE_CORRECTION",
        "IMPLEMENTATION_REVIEW",
    ]
    assert manifest["commits"][-1]["commit"] == "8fcf827b6e5364f0e1bc8221a3f74602e3f39261"
    assert {row["path"] for row in manifest["inputs"]} == set(runner.AUTHORITY_INPUT_PATHS)
    assert all(row["status"] == "PASS" for row in manifest["protected_straight_blobs"])
    assert {
        row["path"]: row["expected_git_blob"]
        for row in manifest["protected_straight_blobs"]
    } == runner.PROTECTED_STRAIGHT_BLOBS
    assert len(manifest["protected_straight_blobs"]) == 7
    assert manifest["production_boundary"]["status"] == "PASS"
    assert manifest["production_boundary"]["independent_checker_authority"] is True


@pytest.mark.parametrize(
    "path",
    ("pyproject.toml", "src/anyfesolver/__init__.py", "src/anyfesolver/elements.py"),
)
def test_added_baseline_blob_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    runner = _load_script(RUNNER, f"_p4_runner_blob_drift_{path.replace('/', '_')}")
    monkeypatch.setitem(runner.PROTECTED_STRAIGHT_BLOBS, path, "0" * 40)
    with pytest.raises(runner.RunnerError, match="accepted straight GE-B3 blobs differ"):
        runner._build_authority_manifest(ROOT, require_clean_inputs=False)


def test_producer_executes_every_registered_obligation_and_all_fixture_families(tmp_path: Path) -> None:
    runner, proof_path, proof, authority_sha256 = _produce(tmp_path)
    assert proof_path.read_bytes() == _canonical_bytes(proof)
    payload = dict(proof)
    claimed = payload.pop("payload_sha256")
    assert claimed == _sha256(_canonical_bytes(payload))
    assert proof["authority_manifest_sha256"] == authority_sha256
    assert proof["obligation_order"] == list(runner.REGISTERED_CASE_IDS)
    assert proof["coverage_count"] == 26
    assert [row["case_id"] for row in proof["obligation_records"]] == list(runner.REGISTERED_CASE_IDS)
    assert {row["status"] for row in proof["obligation_records"]} == {"PASS"}
    assert proof["terminal"] == PASS
    assert proof["reason"] == "ALL_26_PREREGISTERED_OBLIGATIONS_EXECUTED"
    assert [row["case_id"] for row in proof["cases"]] == list(runner.POSITIVE_FIXTURE_IDS)
    assert [row["fixture_id"] for row in proof["negative_fixture_manifest"]] == list(runner.NEGATIVE_FIXTURE_IDS)
    assert {row["status"] for row in proof["auxiliary_evidence"]["negative_admission"]} == {"REJECTED_AS_REGISTERED"}
    assert proof["auxiliary_evidence"]["spatial_chain"]["passed"] is True
    assert proof["auxiliary_evidence"]["curved_stiffener"]["passed"] is True
    assert proof["auxiliary_evidence"]["transformed_scaled"]["passed"] is True
    assert proof["auxiliary_evidence"]["canonical_integrity"]["mutation_detected"] is True
    assert proof["auxiliary_evidence"]["canonical_integrity"]["repeat_identical"] is True
    static_scope = proof["auxiliary_evidence"]["static_scope"]
    assert {
        key: static_scope[key]
        for key in ("production_boundary", "ring_seam_disposition", "straight_blob_freeze")
    } == {
        "production_boundary": True,
        "ring_seam_disposition": "TYPED_REJECTION_REQUIRED_NO_P4_SEAM_AUTHORITY",
        "straight_blob_freeze": True,
    }
    assert static_scope["production_boundary_record"]["status"] == "PASS"
    assert all(
        row["status"] == "PASS"
        for row in static_scope["protected_straight_blob_rows"]
    )


def test_all_60_canonical_mutations_have_exact_order_and_hash_semantics(
    tmp_path: Path,
) -> None:
    runner, _proof_path, proof, _authority_sha = _produce(tmp_path)
    integrity = proof["auxiliary_evidence"]["canonical_integrity"]
    matrix = integrity["mutation_matrix"]
    specs = runner._registered_mutation_specs()
    rows = matrix["mutations"]
    assert len(specs) == len(rows) == runner.REGISTERED_MUTATION_COUNT == 60
    assert matrix["coverage"] == runner.REGISTERED_MUTATION_COVERAGE
    assert matrix["mutation_ids"] == [spec["mutation_id"] for spec in specs]
    assert matrix["mutation_order_sha256"] == runner.REGISTERED_MUTATION_ORDER_SHA256
    runner._validate_mutation_matrix(integrity, manifest=proof["authority_manifest"])
    for row, spec in zip(rows, specs):
        assert row["category"] == spec["category"]
        assert row["mutation_id"] == spec["mutation_id"]
        assert row["operation"] == spec["operation"]
        assert row["target_path"] == spec["target_path"]


def test_runner_rejects_every_mutation_row_identity_and_payload_hash_change(
    tmp_path: Path,
) -> None:
    runner, _proof_path, proof, _authority_sha = _produce(tmp_path)
    integrity = proof["auxiliary_evidence"]["canonical_integrity"]
    manifest = proof["authority_manifest"]
    rows = integrity["mutation_matrix"]["mutations"]
    assert len(rows) == 60
    for index in range(len(rows)):
        target_drift = copy.deepcopy(integrity)
        target_drift["mutation_matrix"]["mutations"][index]["target_path"] += ".DRIFT"
        with pytest.raises(runner.RunnerError, match="row target_path differs"):
            runner._validate_mutation_matrix(target_drift, manifest=manifest)

        hash_drift = copy.deepcopy(integrity)
        row = hash_drift["mutation_matrix"]["mutations"][index]
        row["mutated_payload_sha256"] = _changed_sha256(
            row["mutated_payload_sha256"]
        )
        with pytest.raises(runner.RunnerError, match="payload hash semantics differ"):
            runner._validate_mutation_matrix(hash_drift, manifest=manifest)

    partial = copy.deepcopy(integrity)
    matrix = partial["mutation_matrix"]
    removed = matrix["mutations"].pop()
    matrix["mutation_ids"].pop()
    matrix["mutation_count"] = 59
    matrix["detected_count"] = 59
    matrix["coverage"][removed["category"]] -= 1
    matrix["mutation_order_sha256"] = _sha256(
        _canonical_bytes(matrix["mutation_ids"])
    )
    matrix["passed"] = True
    with pytest.raises(runner.RunnerError, match="mutation inventory differs"):
        runner._validate_mutation_matrix(partial, manifest=manifest)


def test_two_independent_checker_replicas_are_byte_identical(tmp_path: Path) -> None:
    _runner, proof_path, _proof, _authority_sha = _produce(tmp_path)
    outputs = [tmp_path / "check-1.json", tmp_path / "check-2.json"]
    for output in outputs:
        result = _run_checker(proof_path, output)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    value = json.loads(outputs[0].read_text(encoding="ascii"))
    assert value["coverage_count"] == 26
    assert value["terminal"] == PASS
    assert all(value["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "expected_terminal"),
    (
        ("regularity", NO_GO_REGULARITY),
        ("frame", NO_GO_FRAME),
        ("covariance", NO_GO_COVARIANCE),
    ),
)
def test_scientific_mutations_map_to_registered_no_go_terminals(
    tmp_path: Path,
    mutation: str,
    expected_terminal: str,
) -> None:
    runner, proof_path, proof, _authority_sha = _produce(tmp_path)
    producer = _load_script(PRODUCER, f"_p4_producer_no_go_{mutation}_{tmp_path.name}")
    if mutation == "regularity":
        proof["cases"][1]["base"]["regularity"]["minimum_jacobian_squared"] = 0.0
    elif mutation == "frame":
        row = next(
            item
            for item in proof["auxiliary_evidence"]["negative_admission"]
            if item["case_id"] == "HALF_FRAME_BRANCH_CUTOFF_REJECTION"
        )
        row["status"] = "UNEXPECTED_ACCEPTANCE"
    else:
        proof["cases"][1]["objectivity_input"]["translation"][0] += 0.125
    proof["obligation_records"] = producer._obligation_records(
        proof["auxiliary_evidence"], proof["cases"]
    )
    proof["coverage_count"] = sum(
        row["status"] == "PASS" for row in proof["obligation_records"]
    )
    proof["terminal"] = expected_terminal
    proof["reason"] = "PRODUCER_OBSERVED_SCIENTIFIC_CONTRADICTION"
    _rewrite_payload(proof_path, proof)
    output = tmp_path / f"{mutation}.check.json"
    result = _run_checker(proof_path, output)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    value = json.loads(output.read_text(encoding="ascii"))
    assert value["terminal"] == expected_terminal
    assert value["checks"]["scientific_identities"] is False
    assert value["checks"]["complete_coverage"] is (
        value["coverage_count"] == len(runner.REGISTERED_CASE_IDS)
    )
    assert all(
        flag
        for key, flag in value["checks"].items()
        if key not in {"complete_coverage", "scientific_identities"}
    )
    _raw, validated = runner._validate_check_record(
        output,
        proof_sha256=_sha256(proof_path.read_bytes()),
        independent_checker_authority=True,
    )
    assert validated["terminal"] == expected_terminal


def test_malformed_hash_mutation_is_blocking_and_creates_no_checker_output(tmp_path: Path) -> None:
    _runner, proof_path, proof, _authority_sha = _produce(tmp_path)
    proof["coverage_count"] = 25
    proof_path.write_bytes(_canonical_bytes(proof))
    output = tmp_path / "malformed.check.json"
    result = _run_checker(proof_path, output)
    assert result.returncode != 0
    assert not output.exists()


def test_disposable_runner_is_deterministic_complete_and_cannot_classify(
    external_tmp_path: Path,
) -> None:
    runner = _load_script(RUNNER, "_p4_runner_complete_test")
    _manifest, expected = _authority(runner)
    output = external_tmp_path / "aggregate.json"
    work = external_tmp_path / "work"
    aggregate = runner.run_gate(
        output,
        work,
        expected_authority_manifest_sha256=expected,
        require_clean_inputs=False,
        child_timeout_seconds=30.0,
        wave_timeout_seconds=120.0,
        inactivity_seconds=15.0,
        memory_limit_bytes=1 << 30,
    )
    assert aggregate["terminal"] == BLOCKED
    assert aggregate["execution_mode"] == runner.DISPOSABLE_MODE
    assert not any("formal" in key or "request" in key for key in aggregate)
    assert aggregate["checks"] == {
        "all_launched_processes_terminal": True,
        "canonical_cycles_byte_identical": True,
        "complete_preregistered_coverage": True,
        "exclusive_fresh_directories": True,
        "independent_checker_authority": True,
        "one_numerical_thread_per_child": True,
        "post_cycle_live_protected_blobs": True,
        "scientific_classification_authorized": False,
        "successor_executor_required": True,
    }
    assert aggregate["post_cycle_protected_blobs_count"] == 7
    assert aggregate["post_cycle_protected_blobs_sha256"] == _sha256(
        _canonical_bytes(runner._live_protected_blob_rows(ROOT))
    )
    assert aggregate["cycle_sha256"][0] == aggregate["cycle_sha256"][1]
    assert aggregate["cycle_process_log_sha256"][0] == aggregate["cycle_process_log_sha256"][1]
    assert output.read_bytes() == _canonical_bytes(aggregate)
    for index in (1, 2):
        cycle = work / f"cycle-{index}"
        assert (cycle / "cycle.json").is_file()
        assert (cycle / "check-1.json").read_bytes() == (cycle / "check-2.json").read_bytes()
        assert (cycle / "checker-1-runtime").is_dir()
        assert (cycle / "checker-2-runtime").is_dir()
        cycle_record = json.loads((cycle / "cycle.json").read_text(encoding="ascii"))
        assert cycle_record["terminal"] == BLOCKED
        assert cycle_record["diagnostic_checker_terminal"] == PASS
        assert cycle_record["process"]["fresh_checker_processes"] is True
        for name in ("producer.stdout.log", "producer.stderr.log"):
            key = name.replace(".", "_").replace("_log", "")
            assert cycle_record["process_log_sha256"][key] == _sha256((cycle / name).read_bytes())
        assert cycle_record["process_log_sha256"]["checker_stdout"] == [
            _sha256((cycle / f"checker-{replica}.stdout.log").read_bytes())
            for replica in (1, 2)
        ]
        assert cycle_record["process_log_sha256"]["checker_stderr"] == [
            _sha256((cycle / f"checker-{replica}.stderr.log").read_bytes())
            for replica in (1, 2)
        ]
    with pytest.raises(FileExistsError):
        runner.run_gate(
            output,
            external_tmp_path / "unused",
            expected_authority_manifest_sha256=expected,
        )


def test_disposable_adjudication_is_unconditionally_nonclassifying() -> None:
    runner = _load_script(RUNNER, "_p4_runner_precedence_test")
    assert runner._adjudicate_terminals(
        [_complete_cycle(PASS)] * 2,
        canonical_cycles_byte_identical=True,
    ) == BLOCKED
    for terminal in (NO_GO_COVARIANCE, NO_GO_FRAME, NO_GO_REGULARITY, BLOCKED):
        cycles = [_complete_cycle(PASS), _complete_cycle(terminal)]
        assert runner._adjudicate_terminals(
            cycles,
            canonical_cycles_byte_identical=True,
        ) == BLOCKED
    cycles = [_complete_cycle(NO_GO_REGULARITY), _complete_cycle(BLOCKED)]
    assert runner._adjudicate_terminals(
        cycles,
        canonical_cycles_byte_identical=True,
    ) == BLOCKED


def test_authority_identity_drift_is_rejected_before_output_or_work_creation(
    external_tmp_path: Path,
) -> None:
    runner = _load_script(RUNNER, "_p4_runner_identity_drift_test")
    output = external_tmp_path / "aggregate.json"
    work = external_tmp_path / "work"
    with pytest.raises(runner.RunnerError, match="manifest differs"):
        runner.run_gate(
            output,
            work,
            expected_authority_manifest_sha256="0" * 64,
            require_clean_inputs=False,
        )
    assert not output.exists()
    assert not work.exists()


def test_no_formal_execution_interface_exists(tmp_path: Path) -> None:
    runner = _load_script(RUNNER, "_p4_runner_no_formal_interface_test")
    parameters = set(inspect.signature(runner.run_gate).parameters)
    assert not parameters & {
        "mode",
        "formal_authority",
        "formal_request",
        "expected_formal_authority_sha256",
        "expected_formal_request_sha256",
    }
    assert not hasattr(runner, "FORMAL_MODE")
    assert not hasattr(runner, "_validate_formal_authority_and_request")
    parser_options = {
        option
        for action in runner._parser()._actions
        for option in action.option_strings
    }
    assert not parser_options & {
        "--mode",
        "--formal-authority",
        "--formal-request",
        "--expected-formal-authority-sha256",
        "--expected-formal-request-sha256",
    }
    output = tmp_path / "aggregate.json"
    work = tmp_path / "work"
    with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
        runner.run_gate(
            output,
            work,
            expected_authority_manifest_sha256="0" * 64,
            mode="FORMAL_AUTHORITY",
        )
    assert not output.exists()
    assert not work.exists()


def test_disposable_runner_rejects_repository_internal_artifacts() -> None:
    runner = _load_script(RUNNER, "_p4_runner_external_artifact_guard_test")
    output = ROOT / ".forbidden-p4-disposable-aggregate.json"
    work = ROOT / ".forbidden-p4-disposable-work"
    assert not output.exists()
    assert not work.exists()
    with pytest.raises(runner.RunnerError, match="must be external"):
        runner.run_gate(
            output,
            work,
            expected_authority_manifest_sha256="0" * 64,
            require_clean_inputs=False,
        )
    assert not output.exists()
    assert not work.exists()


def test_checker_replica_disagreement_is_detected() -> None:
    runner = _load_script(RUNNER, "_p4_runner_disagreement_test")
    assert runner._adjudicate_terminals(
        [_complete_cycle(PASS), _complete_cycle(PASS)],
        canonical_cycles_byte_identical=False,
    ) == BLOCKED


def test_child_interpreter_and_environment_are_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script(RUNNER, "_p4_runner_isolation_test")
    cycle_source = inspect.getsource(runner._cycle)
    assert cycle_source.count('sys.executable,\n            "-I",\n            "-B"') == 1
    assert cycle_source.count('sys.executable,\n                        "-I",\n                        "-B"') == 1
    monkeypatch.setenv("PYTHONPATH", "FORBIDDEN")
    monkeypatch.setenv("PYTHONSTARTUP", "FORBIDDEN")
    private_home = tmp_path / "private-home"
    environment = runner._one_thread_environment(private_home)
    assert "PYTHONPATH" not in environment
    assert "PYTHONSTARTUP" not in environment
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert all(environment[name] == "1" for name in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ))
    for name in ("APPDATA", "HOME", "LOCALAPPDATA", "TEMP", "TMP", "USERPROFILE"):
        assert Path(environment[name]) == private_home.resolve()


@pytest.mark.parametrize(
    ("command", "timeout", "inactivity", "memory", "expected"),
    (
        ([sys.executable, "-c", "while True: pass"], 0.2, 2.0, 1 << 30, "WALL_TIMEOUT"),
        ([sys.executable, "-c", "import time; time.sleep(30)"], 2.0, 0.1, 1 << 30, "INACTIVITY_TIMEOUT"),
        ([sys.executable, "-c", "import time; x=bytearray(64*1024*1024); time.sleep(30)"], 2.0, 1.0, 16 << 20, "MEMORY_LIMIT"),
    ),
)
def test_child_process_tree_bounds_fail_closed(
    tmp_path: Path,
    command: list[str],
    timeout: float,
    inactivity: float,
    memory: int,
    expected: str,
) -> None:
    runner = _load_script(RUNNER, f"_p4_runner_bound_{expected}")
    result = runner._run_child(
        command,
        cwd=tmp_path,
        stdout_path=tmp_path / f"{expected}.stdout.log",
        stderr_path=tmp_path / f"{expected}.stderr.log",
        timeout_seconds=timeout,
        inactivity_seconds=inactivity,
        memory_limit_bytes=memory,
    )
    assert result.status == expected
    assert result.returncode != 0
    assert not (tmp_path / "aggregate.json").exists()


def test_wall_bound_terminates_spawned_descendant_process(tmp_path: Path) -> None:
    runner = _load_script(RUNNER, "_p4_runner_descendant_bound_test")
    child_code = (
        "import subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        "print(p.pid,flush=True);time.sleep(30)"
    )
    stdout = tmp_path / "tree.stdout.log"
    result = runner._run_child(
        [sys.executable, "-I", "-B", "-c", child_code],
        cwd=tmp_path,
        stdout_path=stdout,
        stderr_path=tmp_path / "tree.stderr.log",
        timeout_seconds=0.4,
        inactivity_seconds=2.0,
        memory_limit_bytes=1 << 30,
    )
    assert result.status == "WALL_TIMEOUT"
    descendant_pid = int(stdout.read_text(encoding="ascii").strip())
    assert descendant_pid not in {pid for pid, _parent in runner._process_parent_pairs()}


def test_process_failure_emits_only_complete_blocked_aggregate_not_partial_cycle(
    external_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script(RUNNER, "_p4_runner_no_partial_test")
    _manifest, expected = _authority(runner)

    def failed_cycle(*_args, **_kwargs):
        return {
            "candidate_id": runner.CANDIDATE_ID,
            "check_sha256": ["ABSENT", "ABSENT"],
            "counts": {"accepted_cases": 0, "covered_obligations": 0, "registered_obligations": 26, "stations": 0},
            "process": {"checkers": ["NOT_LAUNCHED", "NOT_LAUNCHED"], "producer": "WALL_TIMEOUT"},
            "production_restriction": runner.RESTRICTION,
            "proof_sha256": "ABSENT",
            "schema": "anysolver.ge-beam3-curved-p4-reference-cycle-v2",
            "status": "PROCESS_FAILURE",
            "terminal": BLOCKED,
        }

    monkeypatch.setattr(runner, "_cycle", failed_cycle)
    output = external_tmp_path / "aggregate.json"
    work = external_tmp_path / "work"
    aggregate = runner.run_gate(
        output,
        work,
        expected_authority_manifest_sha256=expected,
        require_clean_inputs=False,
    )
    assert aggregate["terminal"] == BLOCKED
    assert aggregate["counts"]["cycles_launched"] == 1
    assert not (work / "cycle-1" / "cycle.json").exists()
    assert output.is_file()


def test_post_cycle_live_blob_drift_is_bound_and_blocked(
    external_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script(RUNNER, "_p4_runner_post_cycle_blob_drift_test")
    _manifest, expected = _authority(runner)
    complete = {
        "counts": {
            "accepted_cases": 6,
            "covered_obligations": 26,
            "registered_obligations": 26,
            "stations": 144,
        },
        "status": "COMPLETE",
        "terminal": BLOCKED,
    }
    monkeypatch.setattr(runner, "_cycle", lambda *_args, **_kwargs: copy.deepcopy(complete))
    rows = runner._live_protected_blob_rows(ROOT)
    rows[0]["working_git_blob"] = "0" * 40
    rows[0]["status"] = "FAIL"
    monkeypatch.setattr(
        runner,
        "_live_protected_blob_rows",
        lambda _repository: copy.deepcopy(rows),
    )
    output = external_tmp_path / "aggregate.json"
    aggregate = runner.run_gate(
        output,
        external_tmp_path / "work",
        expected_authority_manifest_sha256=expected,
        require_clean_inputs=False,
    )
    assert aggregate["terminal"] == BLOCKED
    assert aggregate["checks"]["post_cycle_live_protected_blobs"] is False
    assert aggregate["post_cycle_protected_blobs_count"] == 7
    assert aggregate["post_cycle_protected_blobs_sha256"] == _sha256(
        _canonical_bytes(rows)
    )


def test_runner_rejects_bounds_above_frozen_limits_before_launch(tmp_path: Path) -> None:
    runner = _load_script(RUNNER, "_p4_runner_bounds_test")
    with pytest.raises(runner.RunnerError, match="wave timeout"):
        runner.run_gate(
            tmp_path / "aggregate.json",
            tmp_path / "work",
            expected_authority_manifest_sha256="0" * 64,
            wave_timeout_seconds=1800.001,
        )
