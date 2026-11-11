"""Fresh, bounded, hash-bound G2 S09-S14 confirmation; no general parity claim."""
import argparse
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/reference_cases/ge_beam3_g2_confirmation_inventory.json"
TESTS = ["tests/test_ge_beam3_g2_constraints.py", "tests/test_ge_beam3_g2_corrections.py"]
ACCEPTED = "ACCEPTED_G2_IMPLEMENTATION_REVIEW"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS", "TBB_NUM_THREADS")


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=True)+"\n").encode()


def strict(data, *, compact=False):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError("duplicate key")
            result[key] = value
        return result
    def bad(value): raise ValueError("nonfinite value")
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=bad)
    expected = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode() if compact else canonical(value)
    if expected != data: raise ValueError("noncanonical JSON")
    return value


def write(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())


def publish(path, value):
    # Failure can leave a diagnostic .pending, never a partial acceptance path.
    pending = path.with_name(path.name+".pending")
    write(pending, value)
    if strict(pending.read_bytes()) != value: raise ValueError("staged evidence mismatch")
    os.link(pending, path)


def digest(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=sha256(data).hexdigest())


def git(*args):
    return subprocess.check_output(["git", "-c", "safe.directory="+ROOT.as_posix(), *args], cwd=ROOT).decode().strip()


def identity(candidate, *, clean=True):
    if git("rev-parse", "HEAD") != candidate:
        raise ValueError("candidate HEAD mismatch")
    if clean and git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("dirty candidate")
    paths = sorted(p for p in git("ls-files").splitlines()
                   if p.startswith("src/") or p in TESTS or p in (
                       "scripts/confirm_ge_beam3_g2.py", "docs/reference_cases/ge_beam3_g2_confirmation_inventory.json",
                       "docs/GE_BEAM3_G2_CORRECTION_CONFIRMATION.md",
                       "docs/GE_BEAM3_G2_CONSTRAINT_CONTRACT.md",
                       "docs/reference_cases/ge_beam3_g2_fixtures_v1.json",
                       "tests/test_ge_beam3_g1_elastic.py", "tests/test_ge_beam3_g1_integration.py",
                       "tests/conftest.py", "conftest.py", "pyproject.toml",
                       "tests/test_ge_beam3_g2_confirmation.py",
                       "docs/reference_cases/ge_beam3_g2_correction_records.json",
                       "docs/GE_BEAM3_GENERAL_STATIC_INTEGRATION_CONTRACT.md",
                       "docs/GE_BEAM3_G1A_EXACT_ELASTIC_CONTRACT.md",
                       "docs/reference_cases/e4_pl_s3_v2_bounded_process.py"))
    files = {p: sha256((ROOT/p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}
    distributions = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    return dict(commit=candidate, tree=git("rev-parse", "HEAD^{tree}"), files=files,
                python_version=sys.version, python=digest(Path(sys.executable)), distributions=distributions)


def review_input(path, expected_hash, candidate, tree):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_hash: raise ValueError("review hash mismatch")
    body = strict(raw)
    if set(body) != {"decision", "findings", "reviewer", "scope", "subject_commit"}:
        raise ValueError("review schema")
    if (body["subject_commit"] != candidate or body["scope"]["subject_tree"] != tree or
            body["decision"] != ACCEPTED or body["findings"] != [] or
            body["reviewer"]["independent"] is not True):
        raise ValueError("independent review not accepted for this candidate")
    return raw


class Recorder:
    def __init__(self): self.collected = []; self.reports = []
    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]
    def pytest_runtest_logreport(self, report):
        self.reports.append(dict(node=report.nodeid, phase=report.when, outcome=report.outcome))


def validate_results(body, expected):
    if set(body) != {"collected", "reports", "exitcode"} or body["exitcode"] != 0:
        raise ValueError("test process failure")
    if body["collected"] != expected or len(set(expected)) != len(expected):
        raise ValueError("test inventory mismatch")
    reports = body["reports"]
    want = [dict(node=n, phase=p, outcome="passed") for n in expected for p in ("setup", "call", "teardown")]
    if reports != want: raise ValueError("missing, duplicate, skipped or failed test phase")


def worker(output):
    import pytest
    from contextlib import ExitStack
    from unittest.mock import patch
    from anyfesolver._ge_beam3_g2_analysis import ConstrainedAnalysis
    from anyfesolver import _ge_beam3_g1_element as element
    phases = set()
    def mark(phase):
        phases.add(phase)
        print("G2_CHECKPOINT "+phase, flush=True)
    def instrument(function, phase):
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            if phase not in phases: mark(phase)
            return result
        return call
    recorder = Recorder()
    mark("capture")
    # Observe completed public/private seam operations; never change their result.
    with ExitStack() as stack:
        for target, name, phase in ((ConstrainedAnalysis, "_evaluate", "trial_assembly"),
                                    (ConstrainedAnalysis, "solve", "acceptance"),
                                    (element, "local_solve", "local_solve")):
            stack.enter_context(patch.object(target, name, instrument(getattr(target, name), phase)))
        stack.enter_context(patch.object(ConstrainedAnalysis, "resume",
                                         classmethod(instrument(ConstrainedAnalysis.resume.__func__, "restart"))))
        result = pytest.main(["-vv", "-s", "-p", "no:cacheprovider", "--basetemp", str(output/"pytest"), *TESTS], plugins=[recorder])
    write(output/"tests.json", dict(collected=recorder.collected, reports=recorder.reports, exitcode=int(result)))
    mark("output")
    write(output/"checkpoints.json", sorted(phases))
    return int(result)


def run_child(output):
    spec = importlib.util.spec_from_file_location("g2_process", ROOT/"docs/reference_cases/e4_pl_s3_v2_bounded_process.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    env = dict(os.environ)
    for name in THREADS: env[name] = "1"
    env.update(PYTHONPATH=str(ROOT/"src")+os.pathsep+str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
               G2_DIAGNOSTIC_PAYLOAD=str(output/"packet.json"), G2_CORRECTION_RECORDS=str(output/"corrections"))
    (output/"corrections").mkdir(exist_ok=False)
    job = module._ProcessJob(24*1024**3)
    start = time.monotonic(); last = start; previous = None; status = "FAILED"
    try:
        with (output/"stdout.log").open("xb") as out, (output/"stderr.log").open("xb") as err:
            process = job.launch([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker", str(output)],
                                 cwd=ROOT, env=env, stdout=out, stderr=err)
            while True:
                cpu, active, peak = job.accounting(); now = time.monotonic()
                progress = (cpu, (output/"stdout.log").stat().st_size, (output/"stderr.log").stat().st_size)
                if progress != previous: last = now; previous = progress
                if now-start >= 600 or now-last >= 120 or peak > 24*1024**3:
                    status = "RESOURCE_BLOCKED"; job.terminate(); break
                if process.poll() is not None and active == 0:
                    status = "PASSED" if process.returncode == 0 else "FAILED"; break
                time.sleep(.1)
            process.wait(timeout=15)
        record = dict(status=status, elapsed_seconds=time.monotonic()-start, peak_tree_bytes=peak,
                      active_processes=job.accounting()[1], returncode=process.returncode,
                      stdout=digest(output/"stdout.log"), stderr=digest(output/"stderr.log"))
        write(output/"process.json", record)
        if status != "PASSED" or record["active_processes"] != 0: raise ValueError("bounded child failure")
    finally:
        if job.accounting()[1]: job.terminate()
        job.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--rehearse", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker: return worker(args.worker)
    if os.name != "nt" or not args.candidate or args.output is None:
        parser.error("Windows, candidate and fresh external output required")
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists(): raise ValueError("fresh external directory required")
    frozen = identity(args.candidate, clean=not args.rehearse)
    expected = strict(INVENTORY.read_bytes().replace(b"\r\n", b"\n"))
    required_records = strict((ROOT/"docs/reference_cases/ge_beam3_g2_correction_records.json").read_bytes().replace(b"\r\n", b"\n"))
    review = None
    if not args.rehearse:
        if args.review is None or not args.review_sha256: raise ValueError("accepted review required")
        review = review_input(args.review, args.review_sha256, args.candidate, frozen["tree"])
    output.mkdir(parents=True, exist_ok=False)
    write(output/"frozen-inputs.json", frozen)
    if review is not None:
        with (output/"independent-review.json").open("xb") as stream: stream.write(review)
    started = time.monotonic(); packets = []; results = []; corrections = []
    try:
        for cycle in range(1, 2 if args.rehearse else 3):
            if time.monotonic()-started >= 1200: raise TimeoutError("insufficient wave budget")
            if identity(args.candidate, clean=not args.rehearse) != frozen: raise ValueError("changed frozen inputs")
            directory = output/f"cycle-{cycle}"; directory.mkdir(exist_ok=False)
            print(f"START cycle {cycle}: {directory}", flush=True)
            run_child(directory)
            body = strict((directory/"tests.json").read_bytes()); validate_results(body, expected)
            if strict((directory/"checkpoints.json").read_bytes()) != sorted([
                    "capture", "trial_assembly", "local_solve", "acceptance", "restart", "output"]):
                raise ValueError("missing execution checkpoints")
            packet = (directory/"packet.json").read_bytes()
            # Preserve the existing deterministic packet, without relabeling it.
            p = strict(packet, compact=True)
            if (set(p) != {"schema", "qualification", "first", "second", "recovery", "checkpoint_sha256"} or
                    p["schema"] != "GE_BEAM3_G2_DEVELOPMENT_PACKET_V1" or p["qualification"] is not False):
                raise ValueError("unexpected diagnostic packet")
            if identity(args.candidate, clean=not args.rehearse) != frozen: raise ValueError("changed frozen inputs")
            entries = {}
            files = sorted((directory/"corrections").iterdir())
            if [p.name for p in files] != required_records or any(not p.is_file() or p.is_symlink() for p in files):
                raise ValueError("correction scientific inventory mismatch")
            for path in files:
                raw = path.read_bytes()
                # Test scientific records use the native compact canonical serializer.
                strict(raw, compact=True)
                entries[path.name] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
            corrections.append(entries)
            packets.append(packet); results.append(body)
            print(f"PASS cycle {cycle}: {len(expected)} tests", flush=True)
        if (any(p != packets[0] for p in packets) or any(r != results[0] for r in results) or
                any(c != corrections[0] for c in corrections)):
            raise ValueError("nondeterministic confirmation")
        for name in required_records:
            if any((output/f"cycle-{i}"/"corrections"/name).read_bytes() !=
                   (output/"cycle-1"/"corrections"/name).read_bytes() for i in range(2, len(results)+1)):
                raise ValueError("nondeterministic confirmation")
        if time.monotonic()-started >= 1800: raise TimeoutError("wave deadline")
        if identity(args.candidate, clean=not args.rehearse) != frozen: raise ValueError("changed frozen inputs")
        aggregate = dict(schema="GE_BEAM3_G2_CONFIRMATION_V1", candidate=args.candidate, tree=frozen["tree"],
                         inputs_sha256=sha256(canonical(frozen)).hexdigest(), review_sha256=args.review_sha256,
                         terminal="REHEARSAL_ONLY" if args.rehearse else "PROVISIONAL_GO_GE_BEAM3_G2_CONSTRAINED_ELASTIC_STATIC_ONLY",
                         scope="G2_S09_S14_PRIVATE_ELASTIC_ONE_OR_TWO_ELEMENTS_ONLY", cycles=len(results),
                         tests=expected, packet=dict(bytes=len(packets[0]), sha256=sha256(packets[0]).hexdigest()),
                         correction_records=corrections[0], defaults_changed=False, general_static_parity=False)
        publish(output/("rehearsal.json" if args.rehearse else "aggregate.json"), aggregate)
        print(json.dumps(aggregate, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write(output/"failure.json", dict(status="BLOCKED_G2_CONFIRMATION", exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__": raise SystemExit(main())
