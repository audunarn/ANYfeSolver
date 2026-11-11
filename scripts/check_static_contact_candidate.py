"""Ordinary installed-candidate check adapter for the slice-2 static-contact candidate.

Subcommands (plain pip/pytest adapters, no new harness framework):

  install   fresh venv outside the checkout with the cold resolver and the six
            private wheels (CI), or pip --no-index --target install reusing an
            existing numerical site (local); ``pip check`` in venv mode.
  origins   private package and example origin checks before any solver run.
  run       exact base-batch selectors, staged friction/orientation cases and
            the installed normal/friction/archive example commands, each one
            bounded subprocess; JSON/JUnit/stdout collected even on failure;
            explicit admitted/executed/skipped counts; a staged case whose
            target node skipped is a failure (no skip-based green).
  compare   cross-host compact primary quantities under the adopted bound
            maxabs(delta) <= 1e-9*maxabs(reference) + 1e-12*scale with declared
            positive scales, and declared non-numeric fields exact; no
            blanket float tolerance.

Worker mode (--worker JOBFILE) is internal: inserts recorded sys.path entries
and runs one pytest/module/origins job in a clean interpreter. pytest cases
import test/example sources from the checkout like the 20261019 installed
batches; the solver always resolves from the installed wheels, and the
module-form example commands run from the bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Packaging-only 0.5.0 successor; the frozen predecessor
# scripts/contact_candidate_slice2.json is retained unchanged.
CONFIG_PATH = ROOT / "scripts" / "contact_candidate_050.json"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    "NUMBA_DISABLE_JIT": "1", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "ANYMESHER_DISABLE_NATIVE": "1", "ANYMESHER_REQUIRE_NATIVE": "0",
}
PRIVATE_PACKAGES = ("ANYfeSolver", "anyfesolver-results", "ANYmaterial",
                    "ANYgeometry", "ANYmesher", "ANYfileio")
PRIVATE_MODULES = ("anyfesolver", "anyfesolver_results", "anymaterial",
                   "anygeometry", "anymesher", "anyfileio")
EXAMPLE_MODULES = ("general_contact_s3_physical_orientation",
                   "general_contact_s3_friction_physical_path",
                   "general_contact_static_results_archive")
try:
    import psutil
except ImportError:
    psutil = None


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def private_requirements(config: dict) -> list[str]:
    return [f"ANYgeometry[planar]=={config['private_versions']['ANYgeometry']}"
            if p == "ANYgeometry" else f"{p}=={config['private_versions'][p]}"
            for p in PRIVATE_PACKAGES]


def clean_env(extra: dict | None = None) -> dict:
    """Worker/pip environment without inherited path contamination."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(THREAD_ENV)
    if extra:
        env.update(extra)
    return env


def runtime(args, out: Path, bundle: Path) -> tuple[list[str], list[str], Path]:
    """Return (worker command prefix, sys.path inserts, install root).

    venv mode uses exactly one executable: the venv interpreter itself, run
    isolated (``-I -B``) so inherited PYTHONPATH and user sites cannot
    contaminate the cold environment. Target mode uses the adapter's own
    interpreter isolated the same way, with the install target, the existing
    numerical site and the bundle inserted explicitly.
    """
    if args.venv:
        venv = args.venv.resolve()
        python = str(venv / ("Scripts" if os.name == "nt" else "bin") / "python")
        return [python, "-I", "-B"], [str(bundle)], venv
    if not args.numerical_site:
        raise SystemExit("target mode needs --numerical-site (run install first)")
    root = out / "installed"
    if not root.is_dir():
        raise SystemExit(f"missing installed target: {root}")
    return ([sys.executable, "-I", "-S", "-B"],
            [str(root), str(args.numerical_site.resolve()), str(bundle)], root)


def verify_worker_origins(job: dict) -> None:
    """Preload and verify installed solver and bundle examples in a worker.

    Must run before pytest loads ``tests/conftest.py``: the conftest inserts
    the checkout root at ``sys.path[0]``, which would otherwise redirect
    ``anyfesolver`` and the delivered ``examples`` imports to the checkout.
    Preloading caches the installed/bundle modules in ``sys.modules`` and the
    explicit origin checks refuse any redirected or non-candidate origin.
    """
    root = Path(job["root"])
    checkout = Path(job["checkout"]).resolve()
    bundle_examples = (Path(job["bundle"]) / "examples").resolve()
    solver = Path(__import__("anyfesolver").__file__).resolve()
    if not solver.is_relative_to(root):
        raise SystemExit(f"anyfesolver origin outside install root: {solver}")
    if checkout in solver.parents:
        raise SystemExit(f"anyfesolver origin inside checkout: {solver}")
    import examples
    bundle_on_path = any(Path(p).resolve() == bundle_examples
                         for p in getattr(examples, "__path__", []))
    if not bundle_on_path:
        raise SystemExit(f"examples package not loaded from bundle: "
                         f"{list(getattr(examples, '__path__', []))}")
    for name in load_config()["bundle_examples"]:
        module = __import__(f"examples.{name}", fromlist=["__name__"])
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(bundle_examples):
            raise SystemExit(f"example origin outside bundle: {name}: {origin}")


def run_worker(job: dict) -> int:
    """Internal: execute one job with explicit path inserts (no PYTHONPATH)."""
    sys.path[0:0] = job["inserts"]
    kind = job["kind"]
    if kind == "pytest":
        verify_worker_origins(job)
        import pytest
        raise SystemExit(pytest.main(job["argv"] + ["-o", "pythonpath=",
                                                    "-p", "no:cacheprovider"]))
    if kind == "module":
        verify_worker_origins(job)
        import runpy
        sys.argv = job["argv"]
        runpy.run_module(job["argv"][0], run_name="__main__")
        return 0
    if kind == "extract":
        if len(job["argv"]) != 2:
            raise SystemExit(f"extract needs source and target, got {job['argv']!r}")
        source, target = job["argv"]
        document = json.loads(Path(source).read_text(encoding="utf-8"))
        if "checkpoint" not in document:
            raise SystemExit(f"no checkpoint field in {source}")
        Path(target).write_text(json.dumps(document["checkpoint"]), encoding="utf-8")
        return 0
    if kind == "origins":
        import importlib.metadata
        config = load_config()
        root, checkout = Path(job["root"]), Path(job["checkout"]).resolve()
        record = dict(modules={}, versions={}, examples={}, failures=[])
        for name in PRIVATE_MODULES:
            record["modules"][name] = str(Path(__import__(name).__file__).resolve())
        for name in PRIVATE_PACKAGES:
            record["versions"][name] = importlib.metadata.version(name)
        numerical = ("numpy", "scipy", "threadpoolctl", "numba", "shapely")
        record["numerical_versions"] = {}
        for name in numerical:
            try:
                record["numerical_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                # shapely is only required by the ANYgeometry[planar] extra and
                # was absent (cold install unmeasured) in the measured env.
                record["numerical_versions"][name] = None
        for pin in config["numerical_pins"]:
            if "==" in pin:
                name, version = pin.split("==")
                if record["numerical_versions"].get(name) != version:
                    record["failures"].append(
                        f"numerical pin {name}=={version} violated: "
                        f"{record['numerical_versions'].get(name)}")
        import numba
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        record["runtime"] = dict(
            python=sys.version, executable=sys.executable,
            isolated=bool(sys.flags.isolated),
            hash_randomization=bool(sys.flags.hash_randomization),
            hash_seed_environment=os.environ.get("PYTHONHASHSEED"),
            hash_seed_note="Isolated Python ignores PYTHONHASHSEED; no fixed-hash-seed claim",
            numba_jit_disabled=bool(numba.config.DISABLE_JIT),
            numba_threads=numba.get_num_threads(), pools=pools)
        if not numba.config.DISABLE_JIT or numba.get_num_threads() != 1:
            record["failures"].append("candidate needs JIT off and one numba thread")
        if any(pool["num_threads"] != 1 for pool in pools):
            record["failures"].append("candidate native pools must use one thread")
        for name in EXAMPLE_MODULES:
            module = __import__(f"examples.{name}", fromlist=["__name__"])
            record["examples"][name] = str(Path(module.__file__).resolve())
        for name in PRIVATE_PACKAGES:
            if record["versions"][name] != config["private_versions"][name]:
                record["failures"].append(f"version {name}={record['versions'][name]}")
        for name, path in record["modules"].items():
            path = Path(path)
            if not path.is_relative_to(Path(job["root"])):
                record["failures"].append(f"origin {name} outside install root: {path}")
            if checkout in path.parents:
                record["failures"].append(f"origin {name} inside checkout: {path}")
        for name, path in record["examples"].items():
            if not Path(path).is_relative_to(Path(job["bundle"]) / "examples"):
                record["failures"].append(f"example origin {name}: {path}")
        record["status"] = "PASS" if not record["failures"] else "FAIL"
        Path(job["out"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(dict(status=record["status"], failures=record["failures"])))
        return 0 if record["status"] == "PASS" else 1
    raise SystemExit(f"unknown worker kind: {kind}")


def spawn(cmd: list[str], job: dict, timeout: float, env: dict,
          stdout_path: Path, stderr_path: Path, rss_cap: int) -> dict:
    """Run one bounded worker subprocess with sampled aggregate RSS.

    The aggregate always includes this adapter's own RSS plus the whole worker
    process tree. psutil is required: without it no cap can be claimed, so the
    launch is refused instead of silently continuing. Terminal cleanup is
    verified and any surviving process is recorded; outputs are preserved
    (possibly partial) on every failure mode.
    """
    if psutil is None:
        return dict(status="supervision_unavailable", wall_seconds=0.0,
                    exit_code=None, sampled_peak_aggregate_rss_bytes=0,
                    survivors=[], failure="psutil unavailable; refusing unsupervised launch")
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(
            cmd + [str(Path(__file__).resolve()), "--worker", job["jobfile"]],
            stdout=out, stderr=err, env=env)
        owned = {process.pid: psutil.Process(process.pid)}
        status, peak = "completed", 0
        while True:
            live = []
            try:
                for proc in psutil.Process(process.pid).children(recursive=True):
                    owned.setdefault(proc.pid, proc)
            except psutil.NoSuchProcess:
                pass
            rss = psutil.Process(os.getpid()).memory_info().rss
            for proc in owned.values():
                try:
                    if proc.is_running():
                        live.append(proc)
                        rss += proc.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            if rss_cap and rss > rss_cap:
                status = "rss_limit"
                break
            if process.poll() is not None:
                break
            if time.monotonic() - started >= timeout:
                status = "timeout"
                break
            time.sleep(0.05)
        termination_reason = status
        cleanup_errors = []
        if status != "completed" or process.poll() is None:
            cleanup_deadline = time.monotonic() + 5.0
            for proc in reversed(list(owned.values())):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as exc:
                    cleanup_errors.append(f"pid {proc.pid}: {exc}")
            try:
                process.wait(timeout=max(0.0, cleanup_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                status = "cleanup_unknown"
            # Killing is asynchronous, especially on Windows. Reap/wait for
            # every observed member within the same five-second cleanup bound.
            psutil.wait_procs(list(owned.values()),
                              timeout=max(0.0, cleanup_deadline-time.monotonic()))
        else:
            process.wait()
        survivors = []
        for proc in owned.values():
            try:
                if proc.is_running():
                    survivors.append(proc.pid)
            except psutil.NoSuchProcess:
                pass
        if survivors:
            status = "cleanup_unknown"
    return dict(status=status, wall_seconds=time.monotonic() - started,
                exit_code=process.returncode if status == "completed" else None,
                sampled_peak_aggregate_rss_bytes=peak,
                survivors=survivors, termination_reason=termination_reason,
                cleanup_errors=cleanup_errors)


def junit_counts(path: Path) -> dict:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    return dict(tests=len(cases),
                failures=sum(c.find("failure") is not None for c in cases),
                errors=sum(c.find("error") is not None for c in cases),
                skips=sum(c.find("skipped") is not None for c in cases))


def build_cases(config: dict, tests_root: Path, evidence: Path, results: Path,
                out: Path, stages: list[str], with_examples: bool) -> list[dict]:
    if not with_examples:
        # Transferred checkpoints are a separate prospective candidate check.
        # Keep the original same-host exact-history pytest assertions intact.
        cases = []
        for stage in stages:
            if stage not in ("active-resume", "resume"):
                raise SystemExit("transfer mode accepts active-resume,resume only")
            prefix = "active-prefix" if stage == "active-resume" else "prefix"
            checkpoint = out / f"{stage}-checkpoint.json"
            cases.extend([
                dict(name=f"transfer-{stage}-extract", kind="extract", env={},
                     argv=[str(evidence / "friction" / f"{prefix}.json"),
                           str(checkpoint)]),
                dict(name=f"transfer-{stage}", kind="module", env={},
                     argv=["examples.general_contact_s3_friction_physical_path",
                           "--through-step", "7", "--restart", str(checkpoint),
                           "--output", str(results / "friction" / f"{stage}.json")]),
            ])
        return cases
    cases = [dict(name="base-batch", kind="pytest", junit=out / "base-junit.xml",
                  argv=["-q", "--tb=short", f"--junitxml={out / 'base-junit.xml'}"]
                  + [str(tests_root / s) for s in config["base_selectors"]],
                  env={"CONTACT_APP_RESULT_RECORD": str(out / "archive-report.json")})]
    # Named additional selector batches (e.g. the s7 curved-skins regressions)
    # run as their own fixed pytest cases right after the unchanged original
    # base batch and before the lifecycle cases: the original Windows base
    # batch already fills most of the unchanged per-case seconds cap, so the
    # new selectors must not join it.  Same kind, JUnit, origins, per-case
    # timeout, RSS cap and budget accounting as every other case; no new
    # runner or harness.
    for batch_name, selectors in config.get("additional_selector_batches",
                                           {}).items():
        junit = out / f"{batch_name}-junit.xml"
        cases.append(dict(name=f"{batch_name}-batch", kind="pytest",
                          junit=junit,
                          argv=["-q", "--tb=short", f"--junitxml={junit}"]
                          + [str(tests_root / s) for s in selectors],
                          env={}))
    for stage in [s for s in config["friction_stages"] if s in stages]:
        node = (config["friction_rollback_node"] if stage == "rollback"
                else config["friction_lifecycle_node"])
        cases.append(dict(name=f"friction-{stage}", kind="pytest",
                          junit=out / f"friction-{stage}-junit.xml",
                          argv=["-q", "--tb=short",
                                f"--junitxml={out / f'friction-{stage}-junit.xml'}",
                                str(tests_root / node)],
                          env={"FRICTION_STAGE": stage,
                               "AUTOMATIC_WORKFLOW_EVIDENCE": str(evidence / "friction"),
                               "AUTOMATIC_DOMAIN_RESULT": str(results / "friction" / f"{stage}.json")}))
    for stage in [s for s in config["orientation_stages"] if s in stages]:
        cases.append(dict(name=f"orientation-{stage}", kind="pytest",
                          junit=out / f"orientation-{stage}-junit.xml",
                          argv=["-q", "--tb=short",
                                f"--junitxml={out / f'orientation-{stage}-junit.xml'}",
                                str(tests_root / config["orientation_node"])],
                          env={"ANY_RUN_S3_PHYSICAL_ORIENTATION": "1",
                               "S3_PHYSICAL_ORIENTATION_STAGE": stage,
                               "AUTOMATIC_WORKFLOW_EVIDENCE": str(evidence / "orientation"),
                               "AUTOMATIC_DOMAIN_RESULT": str(results / "orientation" / f"{stage}.json")}))
    if with_examples:
        examples = evidence / "examples"

        def rebase(value: str) -> str:
            return str(examples / value[len("examples/"):]) \
                if value.startswith("examples/") else value

        for name, argv in config["example_commands"].items():
            argv = [rebase(a) for a in argv]
            kind = "extract" if argv[0] == "extract-checkpoint" else "module"
            cases.append(dict(name=f"example-{name}", kind=kind,
                              argv=argv if kind == "module" else argv[1:],
                              env={}, junit=None))
    return cases


def get_path(obj, path: str):
    """Navigate 'a.b[*].c[0]'; '[*]' maps the remainder over a list."""
    for index, segment in enumerate(path.split(".")):
        rest = ".".join(path.split(".")[index + 1:])
        if segment.endswith("[*]"):
            items = obj[segment[:-3]] if segment[:-3] else obj
            return [get_path(item, rest) for item in items] if rest else list(items)
        head, *brackets = segment.split("[")
        if head:
            obj = obj[head]
        for bracket in brackets:
            obj = obj[int(bracket.rstrip("]"))]
    return obj


def flatten(value) -> list:
    return [item for sub in value for item in flatten(sub)] if \
        isinstance(value, list) else [value]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def compare(config: dict, reference: Path, candidate: Path, report: Path) -> int:
    def load(root: Path, file_key: str) -> dict:
        return json.loads((root / config["cross_host_files"][file_key])
                         .read_text(encoding="utf-8"))

    def resolve(root: Path, file_key: str, path: str):
        try:
            return get_path(load(root, file_key), path), None
        except Exception as exc:  # missing configured quantity is a failure
            return None, repr(exc)

    entries = []
    for file_key, path, family, scale in config["cross_host_quantities"]:
        if not finite(scale) or scale <= 0:
            entries.append(dict(file=file_key, path=path, quantity_family=family,
                                declared_positive_scale=scale,
                                passed=False,
                                failure=f"invalid declared scale: {scale!r}"))
            continue
        left, left_error = resolve(reference, file_key, path)
        right, right_error = resolve(candidate, file_key, path)
        if left_error or right_error:
            entries.append(dict(file=file_key, path=path, quantity_family=family,
                                declared_positive_scale=scale,
                                passed=False,
                                failure=f"missing configured quantity: "
                                         f"{left_error or right_error}"))
            continue
        left, right = flatten(left), flatten(right)
        nonfinite = [v for v in left + right if not finite(v)]
        if nonfinite:
            entries.append(dict(file=file_key, path=path, quantity_family=family,
                                declared_positive_scale=scale,
                                passed=False,
                                failure=f"nonfinite comparison input: {nonfinite[:3]}"))
            continue
        if not left or len(left) != len(right):
            entries.append(dict(file=file_key, path=path, quantity_family=family,
                                declared_positive_scale=scale,
                                passed=False,
                                failure=f"incomplete quantity inventory: "
                                         f"{len(left)} reference vs {len(right)} candidate"))
            continue
        delta = max(abs(a - b) for a, b in zip(left, right))
        norm = max(abs(a) for a in left)
        bound = 1e-9 * norm + 1e-12 * scale
        entries.append(dict(file=file_key, path=path, quantity_family=family,
                            declared_positive_scale=scale,
                            norm_inf_difference=delta, norm_inf_reference=norm,
                            bound=bound, passed=delta <= bound))
    for file_key, path in config["cross_host_exact"]:
        left, left_error = resolve(reference, file_key, path)
        right, right_error = resolve(candidate, file_key, path)
        passed = (not left_error and not right_error and
                  json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True))
        entries.append(dict(file=file_key, path=path, exact=True, passed=passed,
                            failure=left_error or right_error or
                            (None if passed else "exact field mismatch")))
    passed = all(entry["passed"] for entry in entries)
    report.write_text(json.dumps(dict(
        status="PASS" if passed else "FAIL", bound=config["cross_host_bound"],
        entries=entries), indent=2), encoding="utf-8")
    print(json.dumps(dict(status="PASS" if passed else "FAIL",
                          quantities=len(entries),
                          failures=sum(not e["passed"] for e in entries))))
    return 0 if passed else 1


def resume_parity(reference: Path, candidate: Path, config: dict) -> dict:
    """Check supported friction replay under the adopted dimensional bound."""
    left = json.loads(reference.read_text(encoding="utf-8"))
    right = json.loads(candidate.read_text(encoding="utf-8"))

    def normalized(document):
        return document if "info" in document else dict(
            displacements=document["checkpoint"]["displacements"],
            info=dict(force_displacement_history=document["phases"]))

    ref, actual = normalized(left), normalized(right)
    checks = []
    for file_key, path, family, scale in config["cross_host_quantities"]:
        if file_key != "friction_full" or path.startswith("poses"):
            continue
        a, b = flatten(get_path(ref, path)), flatten(get_path(actual, path))
        valid = len(a) == len(b) and bool(a) and finite(scale) and scale > 0
        valid = valid and all(finite(v) for v in a + b)
        delta = max(abs(x-y) for x,y in zip(a,b)) if valid else None
        bound = 1e-9 * max(abs(v) for v in a) + 1e-12 * scale if valid else None
        checks.append(dict(path=path, quantity_family=family,
                           norm_inf_difference=delta, bound=bound,
                           passed=valid and delta <= bound))
    for key in ("schema", "version", "integrity_id", "analysis_kind",
                "model_fingerprint", "analysis_fingerprint", "total_dofs"):
        checks.append(dict(path=f"checkpoint.{key}", exact=True,
                           passed=left["checkpoint"][key] == right["checkpoint"][key]))
    for path in ("step_index", "friction.branch", "friction.target_elements"):
        a = get_path(ref, f"info.force_displacement_history[*].{path}")
        b = get_path(actual, f"info.force_displacement_history[*].{path}")
        checks.append(dict(path=path, exact=True, passed=a == b))
    checks.append(dict(path="status", exact=True,
                       passed=left["status"] == right["status"] == "completed"))
    return dict(status="PASS" if all(row["passed"] for row in checks) else "FAIL",
                reference=str(reference), candidate=str(candidate), checks=checks)


def verify_manifest(bundle: Path, config: dict) -> dict:
    """Require the six wheel files byte-identical to the build manifest."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    wheels = manifest.get("wheels", [])
    if len(wheels) != 6 or {w["package"] for w in wheels} != set(config["private_versions"]):
        raise SystemExit(f"manifest does not describe the six private wheels: {bundle}")
    for wheel in wheels:
        path = bundle / "wheelhouse" / wheel["file"]
        if not path.is_file():
            raise SystemExit(f"manifest wheel missing from bundle: {wheel['file']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != wheel["sha256"]:
            raise SystemExit(f"wheel digest mismatch (corrupted bundle?) {wheel['file']}: "
                             f"manifest {wheel['sha256']} != actual {digest}")
    expected_config = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    if manifest.get("config_sha256") != expected_config:
        raise SystemExit("bundle was built from a different cases config; rebuild it")
    return manifest


def pip_run(cmd: list[str], out: Path, name: str, timeout: float,
            network: bool = False) -> None:
    """One pip subprocess with a cleaned environment and a finite timeout."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    if not network:
        env["PIP_NO_INDEX"] = "1"
    else:
        env.pop("PIP_NO_INDEX", None)
    with (out / f"{name}-stdout.txt").open("wb") as stdout, \
            (out / f"{name}-stderr.txt").open("wb") as stderr:
        completed = subprocess.run(cmd, stdout=stdout, stderr=stderr, env=env,
                                   timeout=timeout)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with code {completed.returncode}; "
                         f"see {out / f'{name}-stdout.txt'} and "
                         f"{out / f'{name}-stderr.txt'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "origins", "run", "compare"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests-root", type=Path, default=ROOT)
    parser.add_argument("--venv", type=Path, help="fresh venv directory (venv mode)")
    parser.add_argument("--python", help="pip-capable interpreter (target-mode install)")
    parser.add_argument("--numerical-site", type=Path,
                        help="existing numerical site-packages (target mode)")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--case-seconds", type=float)
    parser.add_argument("--host-seconds", type=float)
    parser.add_argument("--budget-file", type=Path,
                        help="shared host budget ledger (seconds_used); "
                             "chains run calls into one host budget")
    args = parser.parse_args(argv)
    config = load_config()
    bundle, out = args.bundle.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "install":
        verify_manifest(bundle, config)
        if args.venv:
            venv = args.venv.resolve()
            if venv.exists():
                raise SystemExit("venv must not exist; fresh install only")
            pip_run([sys.executable, "-I", "-m", "venv", str(venv)], out,
                    "create-venv", 90)
            executable = "python.exe" if os.name == "nt" else "python"
            vpython = str(venv / ("Scripts" if os.name == "nt" else "bin") / executable)
            pip_run([vpython, "-I", "-m", "pip", "install", "--no-cache-dir"]
                    + config["numerical_pins"], out, "install-runtime", 240,
                    network=True)
            pip_run([vpython, "-I", "-m", "pip", "install", "--no-cache-dir",
                     "--no-index", f"--find-links={bundle / 'wheelhouse'}"]
                    + private_requirements(config), out, "install-private", 120)
            pip_run([vpython, "-I", "-m", "pip", "check"], out, "pip-check", 30)
            record = dict(mode="venv", venv=str(venv), pip_check="PASS")
        else:
            if not args.python or not args.numerical_site:
                raise SystemExit("target-mode install needs --python and --numerical-site")
            target = out / "installed"
            wheels = sorted((bundle / "wheelhouse").glob("*.whl"))
            if len(wheels) != 6:
                raise SystemExit(f"expected six wheels, found {len(wheels)}")
            subprocess.check_call([args.python, "-m", "pip", "install", "--no-index",
                                  "--no-deps", "--no-compile", f"--target={target}"]
                                 + [str(w) for w in wheels])
            record = dict(mode="target", installed=str(target),
                          numerical_site=str(args.numerical_site.resolve()),
                          pip_check="NOT_APPLICABLE_TARGET_MODE")
        (out / "install.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record))
        return 0
    if args.command == "compare":
        if not args.reference or not args.candidate:
            raise SystemExit("compare needs --reference and --candidate")
        return compare(config, args.reference.resolve(), args.candidate.resolve(),
                       out / "compare-report.json")
    cmd, inserts, install_root = runtime(args, out, bundle)
    if args.command == "origins":
        verify_manifest(bundle, config)
        jobs = out / "jobs"
        jobs.mkdir(exist_ok=True)
        job = dict(kind="origins", inserts=inserts, root=str(install_root),
                   bundle=str(bundle), checkout=str(ROOT),
                   out=str(out / "origins.json"),
                   jobfile=str(jobs / "origins.json"))
        Path(job["jobfile"]).write_text(json.dumps(job), encoding="utf-8")
        result = spawn(cmd, job, 300, clean_env(),
                       out / "origins-stdout.txt", out / "origins-stderr.txt",
                       config["timeouts"]["rss_limit_mib"] * 1024 * 1024)
        return 0 if result["status"] == "completed" and result["exit_code"] == 0 else 1
    # run
    evidence = (args.evidence_dir or out / "evidence").resolve()
    results = (args.results_dir or evidence).resolve()
    for sub in ("friction", "orientation", "examples"):
        (results / sub).mkdir(parents=True, exist_ok=True)
    with_examples = args.stages == "all"
    stages = (config["friction_stages"] + config["orientation_stages"]
              if with_examples else args.stages.split(","))
    cases = build_cases(config, args.tests_root.resolve(), evidence, results, out,
                        stages, with_examples)
    case_seconds = args.case_seconds or config["timeouts"]["case_seconds"]
    host_seconds = args.host_seconds or config["timeouts"]["host_numerical_seconds"]
    rss_cap = config["timeouts"]["rss_limit_mib"] * 1024 * 1024
    ledger_path = args.budget_file.resolve() if args.budget_file else None
    budget_used = 0.0
    if ledger_path:
        if ledger_path.is_file():
            budget_used = float(json.loads(
                ledger_path.read_text(encoding="utf-8"))["seconds_used"])
        else:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(dict(seconds_used=budget_used)),
                                   encoding="utf-8")
    jobs = out / "jobs"
    jobs.mkdir(exist_ok=True)
    records, peak_rss, failures = [], 0, []
    stop_reason = None
    for index, case in enumerate(cases):
        remaining = host_seconds - budget_used
        if remaining <= 0:
            records.append(dict(name=case["name"], status="budget_exhausted",
                                executed=False, passed=False))
            failures.append(f"{case['name']}: host budget exhausted")
            continue
        job = dict(kind=case["kind"], inserts=inserts, argv=case["argv"],
                   root=str(install_root), bundle=str(bundle), checkout=str(ROOT),
                   jobfile=str(jobs / f"{case['name']}.json"))
        Path(job["jobfile"]).write_text(json.dumps(job), encoding="utf-8")
        result = spawn(cmd, job, min(case_seconds, remaining),
                       clean_env(case["env"]),
                       out / f"{case['name']}-stdout.txt",
                       out / f"{case['name']}-stderr.txt", rss_cap)
        budget_used += result["wall_seconds"]
        if ledger_path:
            ledger_path.write_text(json.dumps(dict(seconds_used=budget_used)),
                                   encoding="utf-8")
        peak_rss = max(peak_rss, result["sampled_peak_aggregate_rss_bytes"])
        record = dict(name=case["name"], kind=case["kind"], **result)
        if case["kind"] == "pytest":
            record.update(junit_counts(case["junit"])
                          if result["status"] == "completed" and case["junit"].is_file()
                          else dict(tests=0, failures=0, errors=0, skips=0))
            record["executed"] = record["tests"] > 0
            record["passed"] = (result["status"] == "completed" and result["exit_code"] == 0
                                and record["tests"] > 0 and record["skips"] == 0
                                and record["failures"] == 0 and record["errors"] == 0)
            if not record["passed"]:
                failures.append(f"{case['name']}: {result['status']} code={result['exit_code']} "
                                f"tests={record['tests']} failures={record['failures']} "
                                f"errors={record['errors']} skips={record['skips']}")
        else:
            record["executed"] = result["status"] == "completed"
            record["passed"] = record["executed"] and result["exit_code"] == 0
            if not record["passed"]:
                failures.append(f"{case['name']}: {result['status']} code={result['exit_code']}")
        records.append(record)
        if result["status"] in ("rss_limit", "cleanup_unknown",
                                "supervision_unavailable"):
            stop_reason = result["status"]
            break
    if stop_reason:
        for case in cases[len(records):]:
            records.append(dict(name=case["name"],
                                status=f"not_run_after_{stop_reason}",
                                executed=False, passed=False))
            failures.append(f"{case['name']}: not run after {stop_reason}")
    replay = []
    if not failures:
        pairs = [(evidence / "friction/full.json", results / "friction" / f"{s}.json")
                 for s in ("active-resume", "resume") if s in stages]
        if with_examples:
            pairs.append((evidence / "examples/friction.json",
                          evidence / "examples/restart.json"))
        for reference, candidate in pairs:
            try:
                row = resume_parity(reference, candidate, config)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                row = dict(status="FAIL", candidate=str(candidate), error=str(exc))
            replay.append(row)
            if row["status"] != "PASS":
                failures.append(f"candidate replay parity: {candidate}")
    summary = dict(status="PASS" if not failures else "FAIL",
                   admitted=len(cases),
                   executed=sum(r.get("executed", False) for r in records),
                   skipped=sum(r.get("skips", 0) for r in records),
                   passed=sum(r.get("passed", False) for r in records),
                   failed=len(failures), failures=failures,
                   restart_parity=replay,
                   budget_seconds_used=budget_used,
                   budget_seconds_limit=host_seconds,
                   budget_file=str(ledger_path) if ledger_path else None,
                   rss_limit_bytes=rss_cap,
                   sampled_peak_aggregate_rss_bytes=peak_rss,
                   rss_measurement="sampled aggregate adapter and owned worker tree; psutil required; not kernel lifetime peak",
                   cases=records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("status", "admitted", "executed", "skipped", "passed", "failed")}))
    return 0 if not failures else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(run_worker(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))))
    raise SystemExit(main())
