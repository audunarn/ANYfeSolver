"""Small cumulative wall-time/resource harness for the adopted saved-data slice."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/agent_plans/development/r8-f03-supported-work-20261010"
MAX_WALL, RESERVE, CHILD_MAX, MAX_RSS = 180., 60., 30., 2*1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controls", "evaluate"), required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    import psutil  # resource observer only; no numerical runtime
    require = lambda test, msg: None if test else (_ for _ in ()).throw(ValueError(msg))
    require(args.attempt.isascii() and args.attempt.replace("-", "").isalnum(), "simple unique attempt name required")
    prior = [json.loads(p.read_text()) for p in OUT.glob("attempt-*/receipt.json")]
    used = sum(row["process_wall_seconds"] for row in prior)
    cap = min(CHILD_MAX, MAX_WALL-RESERVE-used)
    require(cap > 0., "ordinary allowance exhausted; preserve verification reserve")
    folder = OUT / ("attempt-"+args.attempt)
    require(not folder.exists(), "preserve unique attempt evidence")
    folder.mkdir(parents=True)
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    env.update(NUMBA_DISABLE_JIT="1", PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    if args.mode == "controls":
        child_args = ["-m", "pytest", "-q", "tests/test_r8_f03_supported_work_20261010.py", "-p", "no:cacheprovider"]
    else:
        require(args.plan is not None, "root plan required")
        child_args = [str(ROOT / "scripts/r8_f03_supported_work_dev.py"), "--evaluate",
                      "--output", str(folder / "output"), "--plan", str(args.plan)]
    # Set child affinity before importing any numerical library; runpy keeps the
    # recorded single child command reproducible for module and script modes.
    boot = "import psutil,runpy,sys;psutil.Process().cpu_affinity([0]);a=sys.argv[1:];"
    if args.mode == "controls":
        boot += "sys.argv=a[1:];runpy.run_module(a[1],run_name='__main__',alter_sys=True)"
    else:
        boot += "sys.argv=a;runpy.run_path(a[0],run_name='__main__')"
    command = [sys.executable, "-c", boot, *child_args]
    sources = [Path(__file__), ROOT / "scripts/r8_f03_supported_work_dev.py",
               ROOT / "tests/test_r8_f03_supported_work_20261010.py",
               ROOT / "src/anyfesolver/_stiffened_shell_l1_supported_work_experimental.py",
               ROOT / "src/anyfesolver/_stiffened_shell_l1_assembly_experimental.py"]
    source_hashes = {}
    for i, path in enumerate(sources):
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
        (folder / (f"source-{i}-"+path.name)).write_bytes(path.read_bytes())
    receipt = {"command": command, "mode": args.mode, "source_sha256": source_hashes,
               "environment": {key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT")},
               "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "prior_used_seconds": used, "process_cap_seconds": cap,
               "allowance_seconds": MAX_WALL, "reserve_seconds": RESERVE, "max_rss_bytes": MAX_RSS,
               "status": "launching", "peak_tree_rss_bytes": 0}
    if args.plan:
        receipt["plan_sha256"] = sha(args.plan)
        (folder / "plan.json").write_bytes(args.plan.read_bytes())
    started = time.perf_counter()
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        reason = None
        while process.poll() is None:
            elapsed = time.perf_counter()-started
            try:
                tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree if p.is_running())
                receipt["peak_tree_rss_bytes"] = max(rss, receipt["peak_tree_rss_bytes"])
            except psutil.Error:
                tree, rss = [], 0
            if elapsed >= cap or rss > MAX_RSS:
                reason = "timeout" if elapsed >= cap else "rss_limit"
                for p in reversed(tree):
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                if process.poll() is None:
                    process.kill()
                break
            time.sleep(.02)
        process.wait()
    wall = time.perf_counter()-started
    receipt.update(status=reason or ("completed" if process.returncode == 0 else "failed"),
                   exit_code=process.returncode, process_wall_seconds=wall,
                   cumulative_used_seconds=used+wall, remaining_seconds=MAX_WALL-used-wall,
                   stdout_sha256=sha(folder / "stdout.txt"), stderr_sha256=sha(folder / "stderr.txt"))
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: receipt[key] for key in ("status", "exit_code", "process_wall_seconds", "cumulative_used_seconds", "remaining_seconds")}))
    sys.exit(0 if receipt["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
