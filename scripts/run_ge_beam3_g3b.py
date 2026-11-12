"""Bounded G3b beam development lanes; do not grant formal acceptance."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tests", nargs="*", default=["tests/test_ge_beam3_g3b_mb2.py"])
    a = p.parse_args()
    allowed = {"tests/test_ge_beam3_g3b_mb2.py", "tests/test_ge_beam3_g2_constraints.py", "tests/test_ge_beam3_g2_corrections.py", "tests/test_ge_beam3_g1_elastic.py", "tests/test_ge_beam3_g1_integration.py"}
    allowed.add("tests/test_ge_beam3_g3b_mb3.py")
    allowed.add("tests/test_ge_beam3_g3b_mq4.py")
    allowed.add("tests/test_ge_beam3_g3b_ms3.py")
    allowed.add("tests/test_ge_beam3_g3b_weighted.py")
    allowed.add("tests/test_ge_beam3_g3b_transport.py")
    allowed.add("tests/test_ge_beam3_g3b_owner.py")
    allowed.add("tests/test_ge_beam3_g3b_owner_corrections.py")
    allowed.add("tests/test_ge_beam3_g3b_confirmation_authority.py")
    allowed.add("tests/test_ge_beam3_g3b_confirmation_runner.py")
    if os.name != "nt" or any(t.split("::")[0] not in allowed for t in a.tests): raise ValueError("registered Windows development lane required")
    spec = importlib.util.spec_from_file_location("g3b_bounds", ROOT/"docs/reference_cases/e4_pl_s3_v2_bounded_process.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    out = Path(tempfile.mkdtemp(prefix="anyfesolver-g3b-development-")); print("DIAGNOSTICS "+str(out), flush=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS", "TBB_NUM_THREADS"): env[key] = "1"
    env.update(PYTHONPATH=str(ROOT/"src")+os.pathsep+str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
               G3_PROGRESS="1", G3B_DIAGNOSTIC_PAYLOAD=str(out/"packet.json"),
               G3B_MB3_DIAGNOSTIC_PAYLOAD=str(out/"mb3-packet.json"),
               G3B_MQ4_DIAGNOSTIC_PAYLOAD=str(out/"mq4-packet.json"),
               G3B_MS3_DIAGNOSTIC_PAYLOAD=str(out/"ms3-packet.json"),
               G3B_TRANSPORT_DIAGNOSTIC_PAYLOAD=str(out/"transport-packet.json"),
               G3B_OWNER_DIAGNOSTIC_PAYLOAD=str(out/"owner-packet.json"),
               G3B_WEIGHTED_DIAGNOSTIC_PAYLOAD=str(out/"weighted-packet.json"), G1_DIAGNOSTIC_PAYLOAD=str(out/"g1-packet.json"))
    job = m._ProcessJob(24*1024**3); start = time.monotonic(); last = start; previous = None; status = "FAILED"
    try:
        with (out/"stdout.log").open("xb") as stdout, (out/"stderr.log").open("xb") as stderr:
            process = job.launch([sys.executable, "-u", "-B", "-m", "pytest", "-vv", "-s", "-p", "no:cacheprovider", "--basetemp", str(out/"pytest"), *a.tests], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while True:
                cpu, active, peak = job.accounting(); now = time.monotonic()
                progress = (cpu, (out/"stdout.log").stat().st_size, (out/"stderr.log").stat().st_size)
                if progress != previous: previous = progress; last = now
                if now-start >= 600 or now-last >= 120 or peak > 24*1024**3:
                    status = "RESOURCE_BLOCKED"; job.terminate(); break
                if process.poll() is not None and active == 0:
                    status = "PASSED" if process.returncode == 0 else "FAILED"; break
                time.sleep(.1)
            process.wait(timeout=15)
        record = dict(kind="DEVELOPMENT_ONLY_NOT_QUALIFICATION", status=status, elapsed_seconds=time.monotonic()-start,
                      returncode=process.returncode, peak_tree_bytes=peak, active_processes=job.accounting()[1])
        with (out/"process.json").open("x") as f: json.dump(record, f, sort_keys=True, indent=2)
        print((out/"stdout.log").read_text(errors="replace")); print((out/"stderr.log").read_text(errors="replace")); print(json.dumps(record), flush=True)
        return 0 if status == "PASSED" else 1
    finally:
        if job.accounting()[1]: job.terminate()
        job.close()


if __name__ == "__main__": raise SystemExit(main())
