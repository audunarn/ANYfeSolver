"""Prepare bindings without numerical imports; prospective bounded saved checks.

No default execution authority. Parent must write a matching authority record.
One child at a time, unique attempts, CLOSED ledger never reused.
"""
from __future__ import annotations
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
OUT = ROOT / "docs/agent_plans/development/r8-f03-common-density-20261010"
ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")}
RSS_CAP = 2*1024**3


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def prepared_binding():
    from scripts.r8_f03_common_density_dev import source_binding
    sources = [Path(__file__).resolve(), ROOT / "scripts/r8_f03_common_density_dev.py",
               ROOT / "tests/test_r8_f03_common_density_20261010.py",
               ROOT / "src/anyfesolver/_ge_beam3_retained_generalized.py"]
    return {"sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "inputs": source_binding(), "controls": 12, "qualification": False}


def child(folder):
    import psutil
    psutil.Process().cpu_affinity([0])
    import unittest
    import numpy as np
    from threadpoolctl import threadpool_info
    runtime = {"affinity": psutil.Process().cpu_affinity(),
               "environment": {key: os.environ.get(key) for key in ENV},
               "pools": threadpool_info(), "python": sys.version,
               "numpy": np.__version__}
    require(runtime["affinity"] == [0] and runtime["environment"] == ENV and
            runtime["pools"] and all(row["num_threads"] == 1 for row in runtime["pools"]),
            "effective CPU0/one-thread/JIT-off contract required")
    spec = importlib.util.spec_from_file_location("common_density_controls", ROOT / "tests/test_r8_f03_common_density_20261010.py")
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    suite = unittest.defaultTestLoader.loadTestsFromModule(controls)
    require(suite.countTestCases() == 12, "twelve prospective focused controls required")
    tested = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    result = {"runtime": runtime, "qualification": False, "tests_run": tested.testsRun,
              "failures": len(tested.failures), "errors": len(tested.errors),
              "status": "controls_failed"}
    save(folder / "result.json", result)
    if not tested.wasSuccessful():
        return 1
    from scripts import r8_f03_common_density_dev as contract
    saved = contract.load_saved_contract()
    lift = saved["stationarity"]
    layout = json.loads((contract.NATIVE / "layout.json").read_text())
    inputs = json.loads((contract.NATIVE / "inputs.json").read_text())
    zn, ze = np.zeros(1092), np.zeros(708)
    for row, xyz in zip(layout["typed_physical_nodes"], inputs["xyz_m"]):
        if row["domain"] == "beam":
            i = row["q_offset"]
            zn[i+1] = ze[i+1] = xyz[0]
            zn[i+5] = ze[i+5] = 1
    for row in layout["retained_members"]:
        zn[row["cell_rotation_slots"][2]] = zn[row["cell_rotation_slots"][5]] = 1
    analytic = np.zeros(40)
    for j in range(8):
        analytic[5*j+1] = -float((2*j+1)/128)
    rn = zn @ saved["actions"].native
    re = ze @ saved["actions"].external
    x = zn[list(lift.x)]
    stationary = lift.lift(x)
    # Structural/source controls decide admission. Roundoff measurements have
    # no new acceptance threshold; root adjudicates their recorded scope.
    result.update(status="saved_maps_and_local_stationarity_prepared_development_only",
                  source_binding=saved["binding"],
                  maps={"native_shape": list(saved["actions"].native.shape),
                        "external_shape": list(saved["actions"].external.shape),
                        "native_resultant_action_nonzeros": int(np.count_nonzero(saved["actions"].native[list(lift.p)])),
                        "native_cell_rotation_action_nonzeros": int(np.count_nonzero(saved["actions"].native[[i for i in lift.x if i >= 708]]))},
                  roles={"x": len(lift.x), "p": len(lift.p), "local_D_blocks": len(lift.local_d)},
                  local_D_feasibility_certificates=lift.local_certificates,
                  measured_remainders={
                      "native_yaw_action_column_max_abs_Nm_per_density_unit": float(np.max(abs(rn-analytic))),
                      "external_yaw_action_column_max_abs_Nm_per_density_unit": float(np.max(abs(re-analytic))),
                      "native_signed_yaw_action_max_abs_dual": float(np.max(abs(saved["native_K_signed"] @ zn))),
                      "external_yaw_action_max_abs_dual": float(np.max(abs(saved["external_K"] @ ze))),
                      "stationary_yaw_resultant_row_max_abs": float(np.max(abs((saved["native_K_signed"] @ stationary)[list(lift.p)]))),
                      "lifted_yaw_minus_original_max_abs_coordinate": float(np.max(abs(stationary-zn))),
                      "external_raw_asymmetry_max_abs": float(np.max(abs(saved["external_augmented_raw"]-saved["external_augmented_raw"].T)))},
                  scope="all40 dual action columns, local18-resultant lifts, no equilibrium solve or F03/stability metric")
    save(folder / "result.json", result)
    return 0


def supervise(authority_path, attempt):
    ledger_path = OUT / "evidence/ledger.json"
    if ledger_path.exists():
        require(json.loads(ledger_path.read_text())["status"] != "CLOSED",
                "CLOSED ledger cannot be reused; unused authority discarded")
    authority = json.loads(authority_path.read_text())
    require(authority["status"] == "AUTHORIZED" and authority["sole_slot_released"] is True,
            "explicit prospective authority and released sole numerical slot required")
    binding = prepared_binding()
    require(authority["binding"] == binding, "source/input/control binding changed")
    require(authority["process_cap_seconds"] <= 30 and authority["reserve_seconds"] >= 60,
            "process cap <=30s and reserve >=60s required")
    require(attempt.isascii() and attempt.replace("-", "").isalnum(), "simple unique attempt required")
    import psutil
    prior = [json.loads(p.read_text()) for p in (OUT / "evidence").glob("attempt-*/receipt.json")]
    used = sum(r["process_wall_seconds"] for r in prior)
    cap = min(authority["process_cap_seconds"], authority["budget_seconds"]-authority["reserve_seconds"]-used)
    require(cap > 0, "preserve integration reserve")
    folder = OUT / "evidence" / ("attempt-"+attempt)
    require(not folder.exists(), "unique attempts preserve all failures")
    folder.mkdir(parents=True)
    for i, name in enumerate(binding["sources"]):
        (folder / (f"source-{i}-"+Path(name).name)).write_bytes((ROOT / name).read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(folder)]
    env = os.environ.copy()
    env.update(ENV, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    receipt = {"command": command, "binding": binding, "authority_sha256": sha(authority_path),
               "environment": ENV, "process_cap_seconds": cap, "rss_cap_bytes": RSS_CAP,
               "peak_tree_rss_bytes": 0, "status": "launching"}
    ledger = {"status": "OPEN", "authority_sha256": sha(authority_path), "prior_used_seconds": used,
              "budget_seconds": authority["budget_seconds"], "reserve_seconds": authority["reserve_seconds"]}
    save(ledger_path, ledger)
    start = time.perf_counter()
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        stopped = None
        while process.poll() is None:
            try:
                tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree if p.is_running())
                receipt["peak_tree_rss_bytes"] = max(receipt["peak_tree_rss_bytes"], rss)
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
    receipt.update(status=stopped or ("completed" if process.returncode == 0 else "failed"),
                   exit_code=process.returncode, process_wall_seconds=wall,
                   cumulative_used_seconds=used+wall, remaining_seconds=authority["budget_seconds"]-used-wall,
                   stdout_sha256=sha(folder / "stdout.txt"), stderr_sha256=sha(folder / "stderr.txt"))
    if (folder / "result.json").exists():
        receipt["result_sha256"] = sha(folder / "result.json")
    save(folder / "receipt.json", receipt)
    ledger.update(status="CLOSED", cumulative_used_seconds=used+wall,
                  process_wall_seconds=wall, remaining_seconds=receipt["remaining_seconds"],
                  disposition=receipt["status"], unused_authority="discarded; terminal slice")
    save(ledger_path, ledger)
    print(json.dumps({k: receipt[k] for k in ("status", "exit_code", "process_wall_seconds")}))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--attempt")
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    require(sum(bool(v) for v in (args.prepare, args.authority, args.child)) == 1, "choose prepare, authorized supervisor or its child")
    if args.prepare:
        save(args.prepare, prepared_binding())
    elif args.child:
        sys.exit(child(args.child))
    else:
        require(args.attempt, "unique attempt required")
        sys.exit(supervise(args.authority, args.attempt))
