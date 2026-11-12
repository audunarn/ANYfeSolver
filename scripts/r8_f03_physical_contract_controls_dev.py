"""Bounded stdlib rational controls for the adopted decision-support slice."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/agent_plans/development/r8-f03-physical-contract-20261010/evidence"
F06 = ROOT / "docs/agent_plans/development/r8-f06-cell-moment-20261010/evidence/ledger.json"
ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")}
ALLOWANCE, RESERVE, CAP, RSS_CAP = 90., 60., 10., 2*1024**3


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def sources():
    return [Path(__file__), ROOT / "scripts/r8_f03_physical_contract_dev.py",
            ROOT / "tests/test_r8_f03_physical_contract_20261010.py",
            ROOT / "src/anyfesolver/_ge_beam3_retained_generalized.py",
            ROOT / "src/anyfesolver/_stiffened_shell_l1_local_experimental.py",
            ROOT / "src/anyfesolver/_stiffened_shell_bl5_local_experimental.py",
            ROOT / "src/anyfesolver/e4_pl_element.py",
            ROOT / "scripts/r8_bl5_actual_primal_work_dev.py"]


def child(folder):
    import psutil
    psutil.Process().cpu_affinity([0])
    import unittest
    spec = importlib.util.spec_from_file_location("native_physical_contract_controls", ROOT / "tests/test_r8_f03_physical_contract_20261010.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == 13, "exact twelve controls plus conditional L1/BL5 proof required")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    numerical = [name for name in sys.modules if name == "numpy" or name.startswith("numpy.") or name == "scipy" or name.startswith("scipy.")]
    runtime = {"affinity": psutil.Process().cpu_affinity(), "environment": {key: os.environ.get(key) for key in ENV},
               "numerical_modules_loaded": numerical, "numerical_pool_claim": "none; no numerical libraries imported"}
    require(runtime["affinity"] == [0] and runtime["environment"] == ENV and not numerical,
            "exact-control effective runtime contract violated")
    from scripts.r8_f03_physical_contract_dev import write_record
    write_record(folder / "exact-structural-record.json", controls=True)
    save(folder / "result.json", {"status": "exact_structural_controls_passed" if result.wasSuccessful() else "failed",
         "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
         "runtime": runtime, "qualification": False,
         "scope": "primitive exact decision controls, not production stability/reference or F03 tests"})
    return 0 if result.wasSuccessful() else 1


def supervise(attempt):
    ledger_path = OUT / "ledger.json"
    if ledger_path.exists():
        require(json.loads(ledger_path.read_text())["status"] != "CLOSED",
                "exact-control ledger CLOSED; unused authority was discarded and cannot be reused")
    import psutil
    require(json.loads(F06.read_text())["status"] == "CLOSED", "F06 must release the sole job slot")
    require(attempt.isascii() and attempt.replace("-", "").isalnum(), "simple unique attempt required")
    OUT.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(path.read_text()) for path in OUT.glob("attempt-*/receipt.json")]
    used = sum(row["process_wall_seconds"] for row in prior)
    cap = min(CAP, ALLOWANCE-RESERVE-used)
    require(cap > 0., "active ordinary allowance exhausted; preserve reserve")
    folder = OUT / ("attempt-"+attempt)
    require(not folder.exists(), "preserve every unique attempt")
    folder.mkdir()
    hashes = {}
    for i, path in enumerate(sources()):
        hashes[str(path.relative_to(ROOT))] = sha(path)
        (folder / (f"source-{i}-"+path.name)).write_bytes(path.read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(folder)]
    env = os.environ.copy()
    env.update(ENV, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    ledger = {"status": "OPEN", "authority": "root fresh90s ordinary exact-controls, 60s reserve, no physical execution",
              "budget_seconds": ALLOWANCE, "reserve_seconds": RESERVE, "prior_used_seconds": used,
              "active_process_cap_seconds": cap, "F06_closed_ledger_sha256": sha(F06)}
    save(OUT / "ledger.json", ledger)
    receipt = {"command": command, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_sha256": hashes, "environment": ENV, "budget_seconds": ALLOWANCE,
               "reserve_seconds": RESERVE, "process_cap_seconds": cap, "rss_cap_bytes": RSS_CAP,
               "peak_tree_rss_bytes": 0, "status": "launching"}
    start = time.perf_counter()
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        stopped = None
        while process.poll() is None:
            try:
                tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree if p.is_running())
                receipt["peak_tree_rss_bytes"] = max(rss, receipt["peak_tree_rss_bytes"])
            except psutil.Error:
                tree, rss = [], 0
            if time.perf_counter()-start >= cap or rss > RSS_CAP:
                stopped = "timeout" if rss <= RSS_CAP else "rss_limit"
                for p in reversed(tree):
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                if process.poll() is None:
                    process.kill()
                break
            time.sleep(.01)
        process.wait()
    wall = time.perf_counter()-start
    receipt.update(status=stopped or ("completed" if process.returncode == 0 else "failed"), exit_code=process.returncode,
                   process_wall_seconds=wall, cumulative_used_seconds=used+wall,
                   remaining_seconds=ALLOWANCE-used-wall, stdout_sha256=sha(folder / "stdout.txt"), stderr_sha256=sha(folder / "stderr.txt"))
    if (folder / "result.json").exists():
        receipt["result_sha256"] = sha(folder / "result.json")
    if (folder / "exact-structural-record.json").exists():
        receipt["exact_record_sha256"] = sha(folder / "exact-structural-record.json")
    save(folder / "receipt.json", receipt)
    ledger.update(status="CLOSED", process_wall_seconds=wall, cumulative_used_seconds=used+wall,
                  remaining_seconds=ALLOWANCE-used-wall, attempts=len(prior)+1,
                  disposition=receipt["status"], unused_authority="discarded; closed exact-control slice")
    save(OUT / "ledger.json", ledger)
    print(json.dumps({key: receipt[key] for key in ("status", "exit_code", "process_wall_seconds", "cumulative_used_seconds", "remaining_seconds")}))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt")
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    require(bool(args.attempt) != bool(args.child), "choose supervisor attempt or its one child")
    sys.exit(child(args.child) if args.child else supervise(args.attempt))
