"""Source preparation and prospectively authorized small B2 route controls.

Prepare uses stdlib IO/AST only. No default execution authority, model export,
large-case replay, precision/backend change or qualification claim.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/agent_plans/development/r8-b2-original-operator-20261010"
ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")}
RSS_CAP = 2*1024**3
TEST_FILE = ROOT / "tests/test_r8_b2_original_operator_20261010.py"
PRODUCT = ROOT / "src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py"
OLD = ROOT / "docs/agent_plans/development/r8-b2-compensated-refinement-20261009"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, record):
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")


def binding():
    sources = [Path(__file__).resolve(), PRODUCT, TEST_FILE,
               ROOT / "src/anyfesolver/_b2_physical_linear_batch_dev.py",
               ROOT / "src/anyfesolver/_b2_loaded_point_trace_dev.py",
               ROOT / "examples/stiffened_shell_b2_physical_rigid_footprint.py",
               ROOT / "tests/test_stiffened_shell_b2_q4_panel_experimental.py",
               ROOT / "tests/test_b2_loaded_point_owner.py"]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    old = json.loads((OLD / "registration.json").read_text())
    require(old["residual_methods"]["decimal80_correction"].startswith("Decimal.from_float of stored binary64 A,z,b"),
            "closed correction target/source premise changed")
    return {"sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "controls": 12, "qualification": False,
            "scope": "small original/stored/restoration/affine/caller/refusal controls only",
            "closed_evidence": {"registration_sha256": sha(OLD / "registration.json"),
                                "executed_decimal_source_sha256": sha(OLD / "compensated_case1.py"),
                                "saved_original_state_sha256_historical": old["saved_state_sha256"],
                                "saved_original_matrix_sha256_historical": old["matrix_sha256"],
                                "payload_use": "source/metadata reference only; no large array decoding or replay"}}


def child(folder, authority_path):
    authority = json.loads(authority_path.read_text())
    receipt = json.loads((folder / "receipt.json").read_text())
    ledger = json.loads((OUT / "evidence/ledger.json").read_text())
    require(authority["status"] == "AUTHORIZED" and authority["sole_slot_released"] is True and
            receipt["authority_sha256"] == sha(authority_path) and
            ledger["status"] == "OPEN" and ledger["current_attempt"] == folder.name and
            receipt["binding"] == binding(), "matching ordinary receipt/authority required before numerical imports")
    import psutil
    psutil.Process().cpu_affinity([0])
    import numpy as np
    import scipy
    import numba
    import anyfesolver
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as panel
    from threadpoolctl import threadpool_info
    import pytest
    expected_root = Path(receipt["installed_site"]).resolve() if receipt["installed_site"] else ROOT / "src"
    actual = Path(panel.__file__).resolve()
    require(actual.is_relative_to(expected_root) and sha(actual) == receipt["binding"]["sources"][str(PRODUCT.relative_to(ROOT))],
            "actual imported panel must match source binding within requested source/installed site")
    runtime = {"affinity": psutil.Process().cpu_affinity(), "environment": {k: os.environ.get(k) for k in ENV},
               "pools": threadpool_info(), "python": sys.version, "python_executable": sys.executable,
               "numpy": np.__version__, "scipy": scipy.__version__, "python_optimize": sys.flags.optimize,
               "numba": {"version": numba.__version__, "disable_jit": int(numba.config.DISABLE_JIT),
                         "threads": numba.get_num_threads()},
               "installed_site": receipt["installed_site"],
               "imported": {m.__name__: {"path": str(Path(m.__file__).resolve()), "sha256": sha(m.__file__)}
                            for m in (anyfesolver, panel, np, scipy, numba)}}
    require(runtime["affinity"] == [0] and runtime["environment"] == ENV and
            runtime["pools"] and all(p["num_threads"] == 1 for p in runtime["pools"]) and
            runtime["numba"]["disable_jit"] == 1 and runtime["numba"]["threads"] == 1 and
            sys.flags.optimize == 0, "effective CPU0/one-thread/JIT-off/optimize0 required")
    save(folder / "runtime.json", runtime)

    class Counts:
        collected = 0
        passed = 0
        failed = 0
        skipped = 0

        def pytest_collection_finish(self, session):
            self.collected = len(session.items)
            require(self.collected == 12, "twelve prospective focused controls required")

        def pytest_runtest_logreport(self, report):
            if report.failed:
                self.failed += 1
            elif report.when == "call" and report.passed:
                self.passed += 1
            elif report.skipped:
                self.skipped += 1

    counts = Counts()
    code = pytest.main([str(TEST_FILE), "-q", "-p", "no:cacheprovider", "-o", "pythonpath=",
                       "--junitxml", str(folder / "junit.xml")], plugins=[counts])
    runtime["post_test_pools"] = threadpool_info()
    runtime["post_test_numba"] = {"disable_jit": int(numba.config.DISABLE_JIT), "threads": numba.get_num_threads()}
    require(all(p["num_threads"] == 1 for p in runtime["post_test_pools"]) and
            runtime["post_test_numba"] == {"disable_jit": 1, "threads": 1},
            "one thread and JIT off required also after tests")
    save(folder / "runtime.json", runtime)
    if code == 0 and (counts.passed != 12 or counts.failed or counts.skipped):
        code = 1
    save(folder / "result.json", {"status": "small_original_operator_controls_passed" if code == 0 else "failed",
                                  "exit_code": int(code), "collected": counts.collected,
                                  "passed": counts.passed, "failed_reports": counts.failed,
                                  "skipped": counts.skipped, "runtime": runtime,
                                  "qualification": False, "large_case_jobs": 0})
    return int(code)


def supervise(authority_path, attempt, installed_site=None):
    ledger_path = OUT / "evidence/ledger.json"
    ledger = None
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        require(ledger["status"] != "CLOSED", "CLOSED authority cannot be reused")
    authority = json.loads(authority_path.read_text())
    prepared = binding()
    require(authority["status"] == "AUTHORIZED" and authority["sole_slot_released"] is True,
            "explicit root authority and released sole numerical slot required")
    if ledger is None:
        require(authority["binding"] == prepared, "initial source/control/input binding changed")
    else:
        require(ledger["authority_sha256"] == sha(authority_path) and ledger.get("current_attempt") is None,
                "idle same ordinary cumulative authority required")
    require(authority["process_cap_seconds"] <= 30 and authority["reserve_seconds"] >= 60,
            "process<=30s and integration reserve>=60s required")
    require(attempt.isascii() and attempt.replace("-", "").isalnum(), "simple unique attempt name required")
    import psutil
    prior = [json.loads(p.read_text()) for p in (OUT / "evidence").glob("attempt-*/receipt.json")]
    used = sum(p["process_wall_seconds"] for p in prior)
    cap = min(authority["process_cap_seconds"], authority["budget_seconds"]-authority["reserve_seconds"]-used)
    require(cap > 0, "preserve remaining integration reserve")
    folder = OUT / "evidence" / ("attempt-"+attempt)
    require(not folder.exists(), "preserve every unique failed/successful attempt")
    folder.mkdir(parents=True)
    for i, name in enumerate(prepared["sources"]):
        (folder / (f"source-{i}-"+Path(name).name)).write_bytes((ROOT / name).read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(folder), "--authority", str(authority_path.resolve())]
    env = os.environ.copy()
    import_root = installed_site.resolve() if installed_site else ROOT / "src"
    require(import_root.is_dir(), "existing source/installed site required; no build/install job here")
    env.update(ENV, PYTHONPATH=os.pathsep.join((str(import_root), str(ROOT), str(ROOT / "tests"))),
               PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    receipt = {"status": "launching", "command": command, "binding": prepared,
               "installed_site": str(installed_site.resolve()) if installed_site else None,
               "authority_sha256": sha(authority_path), "environment": {k: env[k] for k in (*ENV, "PYTHONPATH", "PYTHONOPTIMIZE")},
               "process_cap_seconds": cap, "rss_cap_bytes": RSS_CAP, "peak_tree_rss_bytes": 0}
    ledger = {"status": "OPEN", "authority_sha256": sha(authority_path), "budget_seconds": authority["budget_seconds"],
              "reserve_seconds": authority["reserve_seconds"], "prior_used_seconds": used,
              "current_attempt": folder.name}
    save(folder / "receipt.json", receipt)
    save(ledger_path, ledger)
    start = time.perf_counter()
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        except OSError as exc:
            receipt.update(status="PRELAUNCH_NO_CHILD", failure=str(exc), exit_code=None)
            process = None
        if process is not None:
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
            receipt.update(status=stopped or ("completed" if process.returncode == 0 else "failed"), exit_code=process.returncode)
    wall = time.perf_counter()-start
    receipt.update(process_wall_seconds=wall, cumulative_used_seconds=used+wall,
                   remaining_seconds=authority["budget_seconds"]-used-wall,
                   stdout_sha256=sha(folder / "stdout.txt"), stderr_sha256=sha(folder / "stderr.txt"))
    for name in ("result.json", "runtime.json", "junit.xml"):
        if (folder / name).exists():
            receipt[name+"_sha256"] = sha(folder / name)
    save(folder / "receipt.json", receipt)
    ledger.update(status="OPEN", cumulative_used_seconds=used+wall, process_wall_seconds=wall,
                  disposition=receipt["status"], current_attempt=None,
                  remaining_seconds=authority["budget_seconds"]-used-wall)
    save(ledger_path, ledger)
    print(json.dumps({k: receipt[k] for k in ("status", "exit_code", "process_wall_seconds")}))
    return 0 if receipt["status"] == "completed" else 1


def close_ledger(authority_path):
    ledger_path = OUT / "evidence/ledger.json"
    ledger = json.loads(ledger_path.read_text())
    require(ledger["status"] == "OPEN" and ledger.get("current_attempt") is None and
            ledger["authority_sha256"] == sha(authority_path), "idle matching OPEN ordinary ledger required")
    ledger.update(status="CLOSED", unused_authority="discarded; explicit terminal handoff")
    save(ledger_path, ledger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--attempt")
    parser.add_argument("--child", type=Path)
    parser.add_argument("--installed-site", type=Path)
    parser.add_argument("--close", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        require(not args.authority and not args.child and not args.attempt and not args.close and not args.installed_site, "prepare only")
        save(args.prepare, binding())
    elif args.close:
        require(args.authority and not args.child and not args.attempt and not args.installed_site, "close only")
        close_ledger(args.authority)
    elif args.child:
        require(args.authority and not args.attempt and not args.installed_site, "authority-bound child only")
        sys.exit(child(args.child, args.authority))
    else:
        require(args.authority and args.attempt, "root authority and unique supervisor attempt required")
        sys.exit(supervise(args.authority, args.attempt, args.installed_site))
