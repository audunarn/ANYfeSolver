"""Bounded ordinary attachment elimination source/build/installed controls.

Prepare uses stdlib IO/AST only. No default execution authority, model export,
large-case replay, precision/backend change or qualification claim.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/agent_plans/development/r8-b2-original-operator-20261010/evidence/attachment-elimination-v1"
ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")}
RSS_CAP = 2*1024**3
TEST_FILES = [ROOT / "tests/test_r8_b2_attachment_elimination_dev.py", ROOT / "tests/test_r8_b2_original_operator_20261010.py"]
TEST_COUNT = 32
HELPER = ROOT / "src/anyfesolver/_b2_attachment_elimination_dev.py"
PACKAGE = OUT / "package"
PRODUCT = ROOT / "src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, record):
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")


def package_identity(base):
    return {str(p.relative_to(base)).replace("\\", "/"): sha(p) for p in sorted(base.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def binding():
    sources = [Path(__file__).resolve(), PRODUCT, HELPER, *TEST_FILES,
               ROOT / "src/anyfesolver/_b2_physical_linear_batch_dev.py",
               ROOT / "src/anyfesolver/_b2_loaded_point_trace_dev.py",
               ROOT / "tests/test_stiffened_shell_b2_q4_panel_experimental.py",
               ROOT / "tests/test_b2_loaded_point_owner.py", ROOT / "tests/conftest.py"]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    return {"sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
            "package_files": package_identity(ROOT / "src/anyfesolver"),
            "build_inputs": {name: sha(ROOT/name) for name in
                ("pyproject.toml", "README.md", "LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md")},
            "controls": TEST_COUNT, "qualification": False,
            "scope": "NEW bounded attachment fixtures plus affected old12; no historical arrays or campaign"}


def package_job(folder, receipt):
    identity = receipt["binding"]["package_files"]
    operation = receipt["operation"]
    if operation == "snapshot-build":
        passed = [json.loads(p.read_text()) for p in OUT.glob("attempt-*/result.json")]
        require(any(p.get("passed") == TEST_COUNT and p.get("exit_code") == 0 for p in passed),
                "focused source controls must pass before ONE frozen build")
        snapshot = PACKAGE / "snapshot"
        destination = snapshot / "src/anyfesolver"
        if PACKAGE.exists():
            original = json.loads((OUT / "attempt-build1/receipt.json").read_text())
            require(original["status"] == "failed" and original["binding"]["package_files"] == identity and
                    original["binding"]["build_inputs"] == receipt["binding"]["build_inputs"] and
                    not (PACKAGE / "manifest.json").exists() and not list((PACKAGE / "wheel").glob("*.whl")),
                    "only artifact-free failed build1 snapshot may be reused; no replacement wheel")
            require(package_identity(destination) == identity and all(
                    sha(snapshot/name) == digest for name, digest in receipt["binding"]["build_inputs"].items()),
                    "original frozen package/build inputs must remain byteidentical; never recopy")
        else:
            destination.mkdir(parents=True)
            for name, digest in identity.items():
                source, target = ROOT / "src/anyfesolver" / name, destination / name
                require(sha(source) == digest, "source changed during snapshot")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for name, digest in receipt["binding"]["build_inputs"].items():
                require(sha(ROOT/name) == digest, "build input changed during snapshot")
                shutil.copyfile(ROOT/name, snapshot/name)
        require(package_identity(destination) == identity, "frozen package identity required")
        wheel_dir = PACKAGE / "wheel"
        wheel_dir.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir", "--no-index",
                   "--wheel-dir", str(wheel_dir), str(snapshot)]
        save(folder / "subcommand.json", {"command": command, "cwd": str(snapshot)})
        code = subprocess.call(command, cwd=snapshot)
        require(code == 0, "frozen wheel build failed; preserve evidence")
        wheels = list(wheel_dir.glob("*.whl"))
        require(len(wheels) == 1, "exactly one wheel required")
        with zipfile.ZipFile(wheels[0]) as archive:
            wheel_identity = {name[len("anyfesolver/"):]: hashlib.sha256(archive.read(name)).hexdigest()
                              for name in archive.namelist() if name.startswith("anyfesolver/") and not name.endswith("/")}
        require(wheel_identity == identity, "source/snapshot/wheel package byte identity required")
        save(PACKAGE / "manifest.json", {"package_files": identity, "file_count": len(identity),
             "wheel": str(wheels[0]), "wheel_sha256": sha(wheels[0]), "binding": receipt["binding"]})
    else:
        manifest = json.loads((PACKAGE / "manifest.json").read_text())
        wheel, site = Path(manifest["wheel"]), PACKAGE / "site"
        require(manifest["package_files"] == identity and sha(wheel) == manifest["wheel_sha256"] and not site.exists(),
                "same immutable wheel and unique cold install required")
        command = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--no-cache-dir", "--no-index",
                   "--target", str(site), str(wheel)]
        save(folder / "subcommand.json", {"command": command, "cwd": str(ROOT)})
        code = subprocess.call(command, cwd=ROOT)
        require(code == 0, "cold wheel install failed; preserve evidence")
        require(package_identity(site / "anyfesolver") == identity, "wheel/site byte identity required")
        manifest.update(installed_site=str(site), installed_files=len(identity))
        save(PACKAGE / "manifest.json", manifest)
    save(folder / "result.json", {"status": operation+"_passed", "exit_code": 0,
         "file_count": len(identity), "qualification": False, "large_case_jobs": 0})
    return 0


def child(folder, authority_path):
    authority = json.loads(authority_path.read_text())
    receipt = json.loads((folder / "receipt.json").read_text())
    ledger = json.loads((OUT / "ledger.json").read_text())
    require(authority["status"] == "AUTHORIZED" and authority["sole_slot_released"] is True and
            receipt["authority_sha256"] == sha(authority_path) and
            ledger["status"] == "OPEN" and ledger["current_attempt"] == folder.name and
            receipt["binding"] == binding(), "matching ordinary receipt/authority required before numerical imports")
    import psutil
    psutil.Process().cpu_affinity([0])
    basic = {"affinity": psutil.Process().cpu_affinity(), "environment": {k: os.environ.get(k) for k in ENV},
             "python": sys.version, "python_executable": sys.executable, "python_optimize": sys.flags.optimize,
             "temporary_environment": {k: os.environ.get(k) for k in ("TEMP", "TMP", "PIP_NO_CACHE_DIR", "PIP_NO_INDEX")}}
    require(basic["affinity"] == [0] and basic["environment"] == ENV and sys.flags.optimize == 0,
            "effective CPU0/one-thread-environment/optimize0 required")
    require(os.environ.get("TEMP") == str(folder / "temp") and os.environ.get("TMP") == str(folder / "temp") and
            os.environ.get("PIP_NO_CACHE_DIR") == "1" and os.environ.get("PIP_NO_INDEX") == "1",
            "owned per-attempt temporary directory/no-cache/no-index required")
    save(folder / "runtime.json", basic)
    if receipt["operation"] != "controls":
        return package_job(folder, receipt)
    import numpy as np
    import scipy
    import numba
    import anyfesolver
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as panel
    import anyfesolver._b2_attachment_elimination_dev as helper
    import anyfesolver._b2_physical_linear_batch_dev as batch
    import anyfesolver._b2_loaded_point_trace_dev as loaded
    from threadpoolctl import threadpool_info
    import pytest
    expected_root = Path(receipt["installed_site"]).resolve() if receipt["installed_site"] else ROOT / "src"
    actual = Path(panel.__file__).resolve()
    require(actual.is_relative_to(expected_root) and sha(actual) == receipt["binding"]["sources"][PRODUCT.relative_to(ROOT).as_posix()],
            "actual imported panel must match source binding within requested source/installed site")
    require(package_identity(expected_root / "anyfesolver") == receipt["binding"]["package_files"],
            "all imported package files must match source/immutable wheel binding")
    for module in (panel, helper, batch, loaded):
        actual_path = Path(module.__file__).resolve()
        name = "src/anyfesolver/"+actual_path.name
        require(actual_path.is_relative_to(expected_root) and sha(actual_path) == receipt["binding"]["sources"][name],
                "actual imported affected module must match requested source/site binding")
    runtime = {"affinity": psutil.Process().cpu_affinity(), "environment": {k: os.environ.get(k) for k in ENV},
               "pools": threadpool_info(), "python": sys.version, "python_executable": sys.executable,
               "numpy": np.__version__, "scipy": scipy.__version__, "python_optimize": sys.flags.optimize,
               "numba": {"version": numba.__version__, "disable_jit": int(numba.config.DISABLE_JIT),
                         "threads": numba.get_num_threads()},
               "installed_site": receipt["installed_site"],
               "imported": {m.__name__: {"path": str(Path(m.__file__).resolve()), "sha256": sha(m.__file__)}
                            for m in (anyfesolver, panel, helper, batch, loaded, np, scipy, numba)}}
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
            require(self.collected == TEST_COUNT, "exact prospective focused control count required")

        def pytest_runtest_logreport(self, report):
            if report.failed:
                self.failed += 1
            elif report.when == "call" and report.passed:
                self.passed += 1
            elif report.skipped:
                self.skipped += 1

    counts = Counts()
    code = pytest.main([*(str(p) for p in TEST_FILES), "-q", "-p", "no:cacheprovider", "-o", "pythonpath=",
                       "--junitxml", str(folder / "junit.xml")], plugins=[counts])
    runtime["post_test_pools"] = threadpool_info()
    runtime["post_test_numba"] = {"disable_jit": int(numba.config.DISABLE_JIT), "threads": numba.get_num_threads()}
    require(all(p["num_threads"] == 1 for p in runtime["post_test_pools"]) and
            runtime["post_test_numba"] == {"disable_jit": 1, "threads": 1},
            "one thread and JIT off required also after tests")
    save(folder / "runtime.json", runtime)
    if code == 0 and (counts.passed != TEST_COUNT or counts.failed or counts.skipped):
        code = 1
    save(folder / "result.json", {"status": "attachment_and_affected_controls_passed" if code == 0 else "failed",
                                  "exit_code": int(code), "collected": counts.collected,
                                  "passed": counts.passed, "failed_reports": counts.failed,
                                  "skipped": counts.skipped, "runtime": runtime,
                                  "qualification": False, "large_case_jobs": 0})
    return int(code)


def supervise(authority_path, attempt, installed_site=None, operation="controls"):
    ledger_path = OUT / "ledger.json"
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
    require(authority["budget_seconds"] <= 300 and authority["process_cap_seconds"] <= 40 and authority["reserve_seconds"] >= 60,
            "process<=40s and integration reserve>=60s required")
    require(operation == "controls" or installed_site is None, "installed-site is for cold controls only")
    require(attempt.isascii() and attempt.replace("-", "").isalnum(), "simple unique attempt name required")
    import psutil
    prior = [json.loads(p.read_text()) for p in OUT.glob("attempt-*/receipt.json")]
    used = sum(p["process_wall_seconds"] for p in prior)
    cap = min(authority["process_cap_seconds"], authority["budget_seconds"]-authority["reserve_seconds"]-used)
    require(cap > 0, "preserve remaining integration reserve")
    folder = OUT / ("attempt-"+attempt)
    require(not folder.exists(), "preserve every unique failed/successful attempt")
    folder.mkdir(parents=True)
    (folder / "temp").mkdir()
    for i, name in enumerate(prepared["sources"]):
        (folder / (f"source-{i}-"+Path(name).name)).write_bytes((ROOT / name).read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(folder), "--authority", str(authority_path.resolve())]
    env = os.environ.copy()
    import_root = installed_site.resolve() if installed_site else ROOT / "src"
    require(import_root.is_dir(), "existing source/installed site required; no build/install job here")
    env.update(ENV, PYTHONPATH=os.pathsep.join((str(import_root), str(ROOT), str(ROOT / "tests"))),
               PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               TEMP=str(folder / "temp"), TMP=str(folder / "temp"), PIP_NO_CACHE_DIR="1", PIP_NO_INDEX="1",
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    receipt = {"status": "launching", "command": command, "binding": prepared,
               "operation": operation, "installed_site": str(installed_site.resolve()) if installed_site else None,
               "authority_sha256": sha(authority_path), "environment": {k: env[k] for k in (*ENV, "PYTHONPATH", "PYTHONOPTIMIZE", "TEMP", "TMP", "PIP_NO_CACHE_DIR", "PIP_NO_INDEX", "PIP_DISABLE_PIP_VERSION_CHECK")},
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
    ledger_path = OUT / "ledger.json"
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
    parser.add_argument("--operation", choices=("controls", "snapshot-build", "install"), default="controls")
    parser.add_argument("--close", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        require(not args.authority and not args.child and not args.attempt and not args.close and not args.installed_site, "prepare only")
        args.prepare.parent.mkdir(parents=True, exist_ok=True)
        save(args.prepare, binding())
    elif args.close:
        require(args.authority and not args.child and not args.attempt and not args.installed_site, "close only")
        close_ledger(args.authority)
    elif args.child:
        require(args.authority and not args.attempt and not args.installed_site, "authority-bound child only")
        sys.exit(child(args.child, args.authority))
    else:
        require(args.authority and args.attempt, "root authority and unique supervisor attempt required")
        sys.exit(supervise(args.authority, args.attempt, args.installed_site, args.operation))
