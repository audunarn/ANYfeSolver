"""Inert tests for the proof-compressed G3c completion inventory."""
from copy import deepcopy
from hashlib import sha256
import ast
import importlib.util
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "ge_beam3_g3c_proof_compressed.py"
SPEC = importlib.util.spec_from_file_location("g3c_proof_compressed", PATH)
PC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PC)
CHECKER_PATH = ROOT / "docs" / "reference_cases" / "ge_beam3_g3c_proof_compressed_checker.py"
CHECKER_SPEC = importlib.util.spec_from_file_location("g3c_proof_compressed_checker", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(CHECKER)


def _digest(label):
    return sha256(label.encode("ascii")).hexdigest()


def _execution(cycle=1):
    manifest = PC.manifest()
    histories = []
    finals = {}
    by_id = {row["case_id"]: row for row in PC.cases()}
    for case_id in PC.executed_case_ids():
        final = _digest("final:" + case_id); finals[case_id] = final
        histories.append(dict(case_id=case_id,
            events=by_id[case_id]["accepted_stages"],
            history_sha256=_digest("history:" + case_id), final_sha256=final,
            transport_sha256=_digest("transport:" + case_id), passed=True))
    restarts = []
    for row in PC.restart_plan():
        for prefix in row["prefixes"]:
            restarts.append(dict(case_id=row["case_id"], prefix=prefix,
                input_sha256=_digest(f'input:{row["case_id"]}:{prefix}'),
                final_sha256=finals[row["case_id"]], passed=True))
    return dict(schema="GE_BEAM3_G3C_PROOF_COMPRESSED_EXECUTION_V2",
        cycle=cycle, candidate={"commit":"1"*40,"tree":"2"*40},
        manifest_sha256=sha256(PC.canonical(manifest)).hexdigest(),
        histories=histories, restarts=restarts,
        prerequisite_receipts=[dict(name=name,sha256=digest)
                               for name,digest in PC.PREREQUISITES])


def test_proof_compressed_inventory_and_order():
    value = PC.validate(PC.manifest())
    body = value["body"]
    assert body["counts"] == {
        "histories": 375, "executed_histories": 25,
        "accepted_events": 3075, "restart_prefixes": 3450,
        "executed_restart_continuations": 120, "assignments": 3825,
    }
    assert len(body["executed_case_ids"]) == 25
    assert len(set(body["executed_case_ids"])) == 25
    assert body["cases"][0]["case_id"] == "J_B2_PAIR::BASE::S0::NONE"
    assert body["cases"][-1]["case_id"] == (
        "J_MULTIFAMILY_LOOP::PROPER_GLOBAL_TRANSFORM::S2::CM3")
    assert body["assignments"][0]["assignment_index"] == 0
    assert body["assignments"][-1]["assignment_index"] == 3824


def test_proof_compressed_truthful_provenance_and_roots():
    body = PC.manifest()["body"]
    executed = set(body["executed_case_ids"])
    assert sum(row["provenance"] == "EXECUTED" for row in body["derivations"]) == 25
    for row in body["derivations"]:
        assert row["root_case_id"] in executed
        if row["provenance"] == "EXECUTED":
            assert row["case_id"] == row["root_case_id"]
            assert row["transforms"] == []
        else:
            assert row["transforms"]


def test_proof_compressed_restarts_are_genuine_bounded_basis():
    plan = PC.restart_plan()
    assert len(plan) == 25
    assert sum(len(row["prefixes"]) for row in plan) == 120
    assert all(row["prefixes"] == list(range(6)) for row in plan[:15])
    assert all(len(row["prefixes"]) == 3 for row in plan[15:])


@pytest.mark.parametrize("mutation", (
    "case", "assignment", "root", "holdout", "provenance", "hash", "count", "type"))
def test_proof_compressed_mutations_rejected(mutation):
    value = deepcopy(PC.manifest())
    if mutation == "case":
        value["body"]["cases"][0]["graph"] = "FOREIGN"
    elif mutation == "assignment":
        value["body"]["assignments"][-1]["assignment_index"] = 0
    elif mutation == "root":
        value["body"]["derivations"][1]["root_case_id"] = value["body"]["executed_case_ids"][-1]
    elif mutation == "holdout":
        value["body"]["derivations"][1]["holdout_case_id"] = "FOREIGN"
    elif mutation == "provenance":
        value["body"]["derivations"][1]["provenance"] = "EXECUTED_BY_ASSERTION"
    elif mutation == "hash":
        value["body"]["prerequisites"][0]["sha256"] = "0" * 64
    elif mutation == "count":
        value["body"]["counts"]["histories"] = 374
    else:
        value["body"]["counts"]["histories"] = 375.0
    value["self_sha256"] = sha256(PC.canonical(value["body"])).hexdigest()
    with pytest.raises(ValueError):
        PC.validate(value)


def test_checker_is_standard_library_only_and_mechanics_independent():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    assert imports <= {"hashlib", "json"}
    assert imports.isdisjoint({"anyfesolver", "numpy", "scipy", "sympy"})


def test_manifest_is_deterministic():
    first = PC.canonical(PC.manifest())
    second = PC.canonical(PC.manifest())
    assert first == second
    assert sha256(first).hexdigest() == sha256(second).hexdigest()


def test_assignment_neutral_aggregate_and_independent_checker():
    aggregate = PC.aggregate(_execution())
    verification = CHECKER.verify(aggregate)
    assert verification == {
        "independently_verified": True, "histories": 375,
        "assignments": 3825, "executed_histories": 25,
        "executed_restart_continuations": 120,
        "full_g3c_qualified": False, "production_qualified": False,
    }
    assert sum(row["provenance"] == "EXECUTED"
               for row in aggregate["histories"]) == 25
    assert sum(row["provenance"] == "EXECUTED"
               for row in aggregate["assignments"]) == 145
    assert PC.canonical(PC.aggregate(_execution(1))) == PC.canonical(PC.aggregate(_execution(2)))


@pytest.mark.parametrize("field", ("history", "restart", "receipt", "cycle", "candidate"))
def test_execution_evidence_mutations_rejected(field):
    execution = _execution()
    if field == "history":
        execution["histories"][0]["passed"] = False
    elif field == "restart":
        execution["restarts"][0]["final_sha256"] = "0" * 64
    elif field == "receipt":
        execution["prerequisite_receipts"][0]["sha256"] = "0" * 64
    elif field == "cycle":
        execution["cycle"] = 0
    else:
        execution["candidate"]["commit"] = "not-a-commit"
    with pytest.raises(ValueError):
        PC.aggregate(execution)


def test_independent_checker_rejects_derived_claim_relabeling():
    aggregate = PC.aggregate(_execution())
    row = next(item for item in aggregate["histories"]
               if item["provenance"] == "DERIVED_BY_VERIFIED_TRANSPORT")
    row["provenance"] = "EXECUTED"
    body = dict(row); body.pop("record_sha256")
    row["record_sha256"] = sha256(PC.canonical(body)).hexdigest()
    aggregate_body = dict(aggregate); aggregate_body.pop("self_sha256")
    aggregate["self_sha256"] = sha256(PC.canonical(aggregate_body)).hexdigest()
    with pytest.raises(ValueError):
        CHECKER.verify(aggregate)


def test_independent_checker_imports_no_producer_or_mechanics():
    tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    assert imports <= {"hashlib", "json"}


def test_fresh_checker_replicas_are_byte_identical(tmp_path):
    aggregate=PC.canonical(PC.aggregate(_execution()))
    (tmp_path / "aggregate.pending.json").write_bytes(aggregate)
    digest=sha256(aggregate).hexdigest()
    command=[sys.executable,"-I","-S","-B","-u",
        str(ROOT / "scripts" / "run_ge_beam3_qualification.py"),
        "--physical-proof-checker",str(tmp_path),digest]
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
    first=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,check=True).stdout
    second=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,check=True).stdout
    assert first == second
    assert CHECKER.canonical(CHECKER.verify(PC.aggregate(_execution()))) == first


def test_proof_compressed_projection_matches_frozen_source_inventory():
    """Join the independent literals to the original inert authority."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import ge_beam3_g3c_physical_history_owner as source

    original = source.history_matrix()
    assert sha256(PC.canonical(original)).hexdigest() == PC.SOURCE_HISTORY_INVENTORY_SHA256
    projected = [{key: row[key] for key in (
        "case_id", "graph", "variant", "force_scale", "common_motion",
        "accepted_stages")} for row in original]
    independent = [{key: row[key] for key in (
        "case_id", "graph", "variant", "force_scale", "common_motion",
        "accepted_stages")} for row in PC.cases()]
    assert projected == independent

    assignments = source.work_inventory("formal")
    assert sha256(PC.canonical(assignments)).hexdigest() == PC.SOURCE_ASSIGNMENT_INVENTORY_SHA256
    projected_assignments = []
    for index, row in enumerate(assignments):
        projected_assignments.append(dict(
            assignment_index=index, kind=row["kind"],
            case_id=row["case"]["case_id"], prefix=row.get("prefix")))
    assert projected_assignments == PC.assignment_plan()


def test_shared_runner_partition_covers_only_the_frozen_basis():
    runner_path = ROOT / "scripts" / "run_ge_beam3_qualification.py"
    runner_spec = importlib.util.spec_from_file_location("g3c_runner_compressed", runner_path)
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    partition = runner.physical_proof_compressed_partition()
    assert partition["self_sha256"] == runner.PHYSICAL_PROOF_COMPRESSED_PARTITION_SHA
    assert partition["body"]["mode"] == "compressed"
    assert partition["body"]["proof_manifest_sha256"] == sha256(
        PC.canonical(PC.manifest())).hexdigest()
    shards = partition["body"]["shards"]
    assert len(shards) == 100
    assert sum(row["assignment"]["kind"] == "history-producer" for row in shards) == 25
    assert sum(len(row["assignment"].get("assignment_indexes", [])) for row in shards) == 120
    assert max(len(row["assignment"].get("assignment_indexes", [])) for row in shards) == 2
    assert runner.PHYSICAL_PROOF_COMPRESSED_TOOL_SHA == sha256(
        (ROOT / runner.PHYSICAL_PROOF_COMPRESSED_TOOL).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert runner.PHYSICAL_PROOF_COMPRESSED_CHECKER_SHA == sha256(
        CHECKER_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_shared_runner_renews_only_a_drained_wave():
    runner_path = ROOT / "scripts" / "run_ge_beam3_qualification.py"
    runner_spec = importlib.util.spec_from_file_location("g3c_runner_watchdog", runner_path)
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)

    class Timer:
        def __init__(self, delay, callback):
            self.delay=delay; self.callback=callback; self.daemon=False
            self.started=False; self.cancelled=False
        def start(self): self.started=True
        def cancel(self): self.cancelled=True

    watchdog=runner.WaveWatchdog(timer=Timer,exit_process=lambda code:None)
    first=(watchdog.soft,watchdog.hard)
    watchdog.renew_wave()
    assert all(timer.cancelled for timer in first)
    assert watchdog.soft.started and watchdog.hard.started
    watchdog.jobs.append(object())
    with pytest.raises(ValueError): watchdog.renew_wave()
    watchdog.jobs.clear();watchdog.close()


def test_physical_owner_backtracks_only_the_typed_chart_cutback():
    """Freeze the successor's narrow line-search exception boundary."""
    owner_path = ROOT / "src" / "anyfesolver" / "_ge_beam3_g3c_physical_owner.py"
    tree = ast.parse(owner_path.read_text(encoding="utf-8"))
    run = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "_run")
    handlers = [node for node in ast.walk(run) if isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name) and node.type.id == "ValueError"]
    assert len(handlers) == 1
    handler = handlers[0]
    source = ast.get_source_segment(owner_path.read_text(encoding="utf-8"), handler)
    assert "str(exc)!='trial chart requires cutback'" in source
    assert any(isinstance(node, ast.Raise) and node.exc is None
               for node in ast.walk(handler))
    assert any(isinstance(node, ast.Continue) for node in ast.walk(handler))
    backtrack = next(node for node in ast.walk(run) if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name) and node.target.id == "cut")
    assert (isinstance(backtrack.iter, ast.Call)
            and isinstance(backtrack.iter.func, ast.Name)
            and backtrack.iter.func.id == "range"
            and ast.literal_eval(backtrack.iter.args[0]) == 16)
