"""Bounded G1 development tests. Never grants formal qualification."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", default=["tests/test_ge_beam3_g1_elastic.py", "tests/test_ge_beam3_g1_integration.py"])
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("this runner requires Windows process-tree memory enforcement")
    for value in args.tests:
        path = value.split("::")[0]
        if path not in {"tests/test_ge_beam3_g1_elastic.py", "tests/test_ge_beam3_g1_integration.py"}:
            raise ValueError("only G1 registered development tests")
    spec = importlib.util.spec_from_file_location("g1_bound_process", ROOT / "docs/reference_cases/e4_pl_s3_v2_bounded_process.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = Path(tempfile.mkdtemp(prefix="anyfesolver-g1-development-"))
    print("DIAGNOSTICS " + str(output), flush=True)
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")+os.pathsep+str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["G1_DIAGNOSTIC_PAYLOAD"] = str(output / "scientific-development.json")
    job = module._ProcessJob(24*1024**3)
    started = time.monotonic(); last = started; previous = None; status = "FAILED"
    try:
        with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
            process = job.launch([sys.executable, "-u", "-B", "-m", "pytest", "-vv", "-s", "-p", "no:cacheprovider",
                                  "--basetemp", str(output / "pytest"), *args.tests], cwd=ROOT, env=env, stdout=out, stderr=err)
            while True:
                cpu, active, peak = job.accounting()
                tick = time.monotonic()
                progress = (cpu, (output / "stdout.log").stat().st_size, (output / "stderr.log").stat().st_size)
                if progress != previous:
                    previous = progress; last = tick
                if process.poll() is not None and active == 0:
                    status = "PASSED" if process.returncode == 0 else "FAILED"
                    break
                if tick-started >= 600 or tick-last >= 120 or peak > 24*1024**3:
                    status = "RESOURCE_BLOCKED"; job.terminate(); break
                time.sleep(.1)
            process.wait(timeout=15)
        record = dict(kind="DEVELOPMENT_ONLY_NOT_QUALIFICATION", status=status,
                      elapsed_seconds=time.monotonic()-started, returncode=process.returncode,
                      peak_tree_bytes=peak, active_processes=job.accounting()[1])
        with (output / "development-process.json").open("x") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
        print((output / "stdout.log").read_text(errors="replace"))
        print((output / "stderr.log").read_text(errors="replace"))
        print(json.dumps(record, sort_keys=True), flush=True)
        return 0 if status == "PASSED" else 1
    finally:
        if job.accounting()[1]: job.terminate()
        job.close()


if __name__ == "__main__":
    raise SystemExit(main())
