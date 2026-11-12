"""Bounded B2 physical-footprint scale probe.

Run `python scripts/b2_large_scale_development.py --size small --cases 1 --out PATH`.
The parent monitors a single child, preserves stdout/stderr, and records a budget
receipt. The optional installed profile is a development probe, not G5/B16 qualification.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, is_dataclass
import hashlib
import io
import json
import math
import os
from numbers import Real
from pathlib import Path
import subprocess
import sys
import time
from types import MappingProxyType
from typing import NamedTuple
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MAX_RSS = 2 * 1024**3
MAX_CHILD_SECONDS = 90.0
MAX_TOTAL_SECONDS = 240.0
RESERVE_SECONDS = 60.0
DEFAULT_BUDGET = ROOT / "docs/agent_plans/development/r8-b2-large-scale-20261009/budget.json"
INSTALLED_LARGE_WHEEL = (ROOT / "docs/agent_plans/development/"
    "r8-b2-footprint-installed-20261009/dist-v2/anysolver-0.4.7-py3-none-any.whl")
INSTALLED_LARGE_WHEEL_SHA256 = "edd8d0ff42b33b298baceb4f5f87327f05917b0f544083b61a1862e7a34fbf04"
INSTALLED_REFINEMENT_WHEEL = (ROOT / "docs/agent_plans/development/"
    "r8-b2-refinement-installed-20261009/dist-v1/anysolver-0.4.7-py3-none-any.whl")
INSTALLED_REFINEMENT_WHEEL_SHA256 = "3b6eb09e9e60221bee9fa8e9366c83ad158d82ba4aad36abb18487382a124571"
INSTALLED_REFINEMENT_SITE = ROOT / "docs/agent_plans/development/r8-b2-refinement-installed-20261009/site-v2"
INSTALLED_REFINEMENT_BUDGET = ROOT / "docs/agent_plans/development/r8-b2-refinement-large-20261009/budget.json"
INSTALLED_REFINEMENT_FOCUSED_BUDGET = ROOT / "docs/agent_plans/development/r8-b2-refinement-installed-20261009/budget.json"
INSTALLED_LARGE_BUDGET = ROOT / "docs/agent_plans/development/r8-b2-installed-large-20261009/budget.json"
INSTALLED_BEAM_RICH_BUDGET = ROOT / "docs/agent_plans/development/r8-b2-beam-rich-20261009/budget.json"
INSTALLED_BEAM_RICH_OUT = ROOT / "docs/agent_plans/development/r8-b2-beam-rich-20261009/attempt-v1"
INSTALLED_BEAM_RICH_REGISTRATION = ROOT / "docs/agent_plans/development/r8-b2-beam-rich-20261009/registration.json"
INSTALLED_PROFILES = frozenset(("installed_large_dev", "installed_large_refinement_dev",
                                "installed_beam_rich_dev"))
INSTALLED_CLEANUP_MARGIN_SECONDS = 10.0
INSTALLED_PRODUCT_MODULES = ("__init__.py", "stiffened_shell.py",
    "_stiffened_shell_b2_q4_panel_experimental.py",
    "_stiffened_shell_b2_physical_rigid_footprint.py")
INSTALLED_EXPECTED_COUNTS = MappingProxyType(dict(
    q4_shells=10_000, physical_b2_members=1_000, attached_patches=100))
INSTALLED_COMPILED_COUNTS = MappingProxyType(dict(
    q4_shells=10_000, physical_b2_members=1_000, attached_patches=100,
    footprint_incidences=400, footprint_rows=2_000, constraint_rows=2_000))
BEAM_RICH_EXPECTED_COUNTS = MappingProxyType(dict(
    q4_shells=400, physical_b2_members=1_000, attached_patches=100))
BEAM_RICH_COMPILED_COUNTS = MappingProxyType(dict(
    q4_shells=400, physical_b2_members=1_000, attached_patches=100,
    footprint_incidences=400, footprint_rows=2_000, constraint_rows=2_000))


class RunProfile(NamedTuple):
    max_rss_bytes: int
    child_seconds: float
    total_seconds: float
    reserve_seconds: float


RUN_PROFILES = MappingProxyType({
    "source_dev": RunProfile(MAX_RSS, MAX_CHILD_SECONDS, MAX_TOTAL_SECONDS, RESERVE_SECONDS),
    "installed_large_dev": RunProfile(4 * 1024**3, 240.0, 300.0, 60.0),
    "installed_large_refinement_dev": RunProfile(4 * 1024**3, 240.0, 300.0, 60.0),
    "installed_beam_rich_dev": RunProfile(4 * 1024**3, 240.0, 300.0, 60.0),
})
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "NUMBA_NUM_THREADS")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def require_separate_budget(run_profile, budget):
    reserved = ((INSTALLED_LARGE_BUDGET, "installed_large_dev"),
        (INSTALLED_REFINEMENT_BUDGET, "installed_large_refinement_dev"),
        (INSTALLED_BEAM_RICH_BUDGET, "installed_beam_rich_dev"),
        (INSTALLED_REFINEMENT_FOCUSED_BUDGET, None))
    for path, owner in reserved:
        if budget == path.resolve(strict=False) and run_profile != owner:
            raise ValueError("budget belongs to a different registered activity")


def resolve_run(args):
    """Refuse ambiguous installed runs before creating output or spawning a child."""
    name = args.run_profile
    profile = RUN_PROFILES[name]
    installed_site = getattr(args, "installed_site", None)
    budget_arg = getattr(args, "budget", None)
    if name in INSTALLED_PROFILES:
        required_size = "beam_rich" if name == "installed_beam_rich_dev" else "large"
        if args.size != required_size:
            raise ValueError(f"{name} requires --size {required_size}")
        if args.cases != 100:
            raise ValueError(f"{name} requires exactly 100 cases")
        if not installed_site:
            raise ValueError(f"{name} requires --installed-site")
        if not budget_arg:
            raise ValueError(f"{name} requires a separate --budget")
        # Windows sandboxes can deny strict realpath on an accessible directory.
        site = Path(installed_site).resolve(strict=False)
        if not site.is_dir() or not (site / "anyfesolver/__init__.py").is_file():
            raise ValueError("installed site must contain an anyfesolver package")
        budget = Path(budget_arg).resolve()
        if budget == DEFAULT_BUDGET.resolve():
            raise ValueError(f"{name} cannot use the source development budget")
        require_separate_budget(name, budget)
        if name in ("installed_large_refinement_dev", "installed_beam_rich_dev"):
            if site != INSTALLED_REFINEMENT_SITE.resolve(strict=False):
                raise ValueError(f"{name} requires the bound installed site")
            required_budget = (INSTALLED_BEAM_RICH_BUDGET if name == "installed_beam_rich_dev"
                               else INSTALLED_REFINEMENT_BUDGET)
            if budget != required_budget.resolve(strict=False):
                raise ValueError(f"{name} requires its separate budget")
        if name == "installed_beam_rich_dev" and Path(args.out).resolve(strict=False) != (
                INSTALLED_BEAM_RICH_OUT.resolve(strict=False)):
            raise ValueError("installed_beam_rich_dev requires its unique registered output")
        return name, profile, budget, site
    if installed_site:
        raise ValueError("--installed-site requires an installed run profile")
    if args.size == "beam_rich":
        raise ValueError("--size beam_rich requires installed_beam_rich_dev")
    budget = Path(budget_arg).resolve() if budget_arg else DEFAULT_BUDGET.resolve()
    require_separate_budget(name, budget)
    return name, profile, budget, None


def verify_installed_import_origin(module_file, site):
    """Check the actual child import after Python resolves its module search path."""
    site = Path(site).resolve(strict=False)
    expected = site / "anyfesolver"
    origin = Path(module_file).resolve(strict=False)
    if not (expected / "__init__.py").is_file() or not origin.is_file() or origin.parent != expected:
        raise RuntimeError(f"anyfesolver imported outside installed site: {origin}")
    return origin


def installed_site_identity(site, *, run_profile="installed_large_dev"):
    """Bind the site to the frozen wheel, including every module used by this route."""
    site = Path(site)
    if run_profile == "installed_large_dev":
        wheel_path, expected_sha = INSTALLED_LARGE_WHEEL, INSTALLED_LARGE_WHEEL_SHA256
    elif run_profile in ("installed_large_refinement_dev", "installed_beam_rich_dev"):
        wheel_path, expected_sha = INSTALLED_REFINEMENT_WHEEL, INSTALLED_REFINEMENT_WHEEL_SHA256
    else:
        raise ValueError("installed wheel identity requires a named installed profile")
    wheel = wheel_path.resolve(strict=False)
    wheel_sha = digest(wheel)
    if wheel_sha != expected_sha:
        raise ValueError(f"{run_profile} wheel differs from frozen SHA256")
    records = sorted(site.glob("anyfesolver-*.dist-info/RECORD"))
    if len(records) != 1:
        raise ValueError("installed site must contain exactly one anyfesolver wheel RECORD")
    record_name = f"{records[0].parent.name}/RECORD"
    module_hashes = {}
    product_file_count = 0
    with ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("anyfesolver/") or member.endswith("/"):
                continue
            wheel_bytes = archive.read(member)
            try:
                site_bytes = (site / member).read_bytes()
            except OSError as exc:
                raise ValueError(f"installed site lacks frozen wheel product file: {member}") from exc
            if site_bytes != wheel_bytes:
                raise ValueError(f"installed site differs from frozen wheel: {member}")
            product_file_count += 1
            name = member.removeprefix("anyfesolver/")
            if name in INSTALLED_PRODUCT_MODULES:
                module_hashes[name] = hashlib.sha256(site_bytes).hexdigest()
        wheel_record = archive.read(record_name)
    if set(module_hashes) != set(INSTALLED_PRODUCT_MODULES):
        raise ValueError("frozen wheel omits a required installed product module")
    site_record = records[0].read_bytes()
    def record_entries(blob):
        rows = list(csv.reader(io.StringIO(blob.decode("utf-8"))))
        if any(len(row) != 3 for row in rows) or len({row[0] for row in rows}) != len(rows):
            raise ValueError("invalid installed wheel RECORD")
        return {row[0]: (row[1], row[2]) for row in rows}
    wheel_entries = record_entries(wheel_record)
    site_entries = record_entries(site_record)
    if any(site_entries.get(name) != row for name, row in wheel_entries.items()):
        raise ValueError("installed site RECORD differs from frozen wheel entries")
    allowed_installer_records = {f"{records[0].parent.name}/{name}"
        for name in ("direct_url.json", "REQUESTED", "INSTALLER")}
    extras = site_entries.keys() - wheel_entries.keys()
    if any(name not in allowed_installer_records and not (
            "/__pycache__/" in name and name.endswith(".pyc")) for name in extras):
        raise ValueError("installed site RECORD contains unexpected entries")
    return dict(installed_site=str(site), installed_module_sha256=module_hashes,
                installed_product_file_count=product_file_count,
                installed_wheel=str(wheel), installed_wheel_sha256=wheel_sha,
                expected_wheel_sha256=expected_sha,
                installed_wheel_record_sha256=hashlib.sha256(wheel_record).hexdigest(),
                installed_site_record_sha256=hashlib.sha256(site_record).hexdigest(),
                installed_site_record_extra_entries=len(extras),
                installed_wheel_record=str(records[0]))


def verify_beam_rich_registration(config, identity, limits, budget, outdir):
    """Require the frozen development request before spending its separate ledger."""
    registration = json.loads(INSTALLED_BEAM_RICH_REGISTRATION.read_text(encoding="utf-8"))
    candidate = registration["candidate"]
    run = registration["run"]
    resources = registration["resources"]
    expected = registration["expected_inventory"]
    if (registration["registration_id"] != "B16_B2_BEAM_RICH_INSTALLED_DEV_V1" or
            registration["status"] != "FROZEN_PRE_RUN" or
            candidate["wheel"] != str(INSTALLED_REFINEMENT_WHEEL.relative_to(ROOT)).replace("\\", "/") or
            candidate["wheel_sha256"] != identity["installed_wheel_sha256"] or
            candidate["site"] != str(INSTALLED_REFINEMENT_SITE.relative_to(ROOT)).replace("\\", "/") or
            candidate["runner_sha256"] != digest(__file__) or
            run["profile"] != "installed_beam_rich_dev" or
            run["size"] != "beam_rich" or run["cases"] != 100 or
            run["config"] != json.loads(json.dumps(config)) or
            run["budget"] != str(budget.relative_to(ROOT)).replace("\\", "/") or
            run["output"] != str(outdir.relative_to(ROOT)).replace("\\", "/") or
            expected != dict(BEAM_RICH_COMPILED_COUNTS) or
            resources != dict(cumulative_budget_s=300, reserve_s=60,
                              configured_child_limit_s=240, cleanup_margin_s=10,
                              sampled_process_tree_memory_limit_bytes=4 * 1024**3,
                              cpu_affinity=[0], numerical_threads=1,
                              jit_disabled=True, python_optimize=0,
                              one_heavy_numerical_job=True) or
            limits != RUN_PROFILES["installed_beam_rich_dev"]):
        raise ValueError("beam-rich frozen registration differs from runner request")
    return dict(registration_file=str(INSTALLED_BEAM_RICH_REGISTRATION),
                registration_sha256=digest(INSTALLED_BEAM_RICH_REGISTRATION))


def verify_installed_inventory(config, inventory, *, definition=None, owner=None,
                               run_profile="installed_large_dev"):
    beam_rich = run_profile == "installed_beam_rich_dev"
    expected_config = settings("beam_rich" if beam_rich else "large", 100)
    if config != expected_config:
        if beam_rich:
            raise RuntimeError("installed_beam_rich_dev input is not the exact 100-case model")
        raise RuntimeError("installed_large_dev input is not the exact full-size 100-case model")
    expected = dict(BEAM_RICH_EXPECTED_COUNTS if beam_rich else INSTALLED_EXPECTED_COUNTS)
    if any(inventory.get(key) != value for key, value in expected.items()):
        raise RuntimeError("installed_large_dev generated inventory differs from target")
    if definition is not None:
        actual = dict(q4_shells=len(definition.shells),
            physical_b2_members=len(definition.members),
            attached_patches=len(definition.physical_footprints))
        if actual != expected:
            raise RuntimeError("installed_large_dev definition inventory differs from target")
    if owner is not None:
        actual = dict(q4_shells=len(owner._shell_formulations),
            physical_b2_members=len(owner._members),
            attached_patches=len(owner.footprints),
            footprint_incidences=len(owner._footprint_incidence),
            footprint_rows=sum(item.row_count for item in owner._footprint_incidence),
            constraint_rows=owner._C.shape[0])
        compiled_expected = dict(BEAM_RICH_COMPILED_COUNTS if beam_rich
                                 else INSTALLED_COMPILED_COUNTS)
        if actual != compiled_expected:
            raise RuntimeError(f"{run_profile} compiled inventory differs from target")
        return actual
    return expected


def _finite_physical_response(value):
    import numpy as np
    if value is None or isinstance(value, (str, bytes, bool)):
        return True
    if is_dataclass(value):
        return all(_finite_physical_response(getattr(value, field.name))
                   for field in fields(value))
    if isinstance(value, (tuple, list)):
        return all(_finite_physical_response(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, Real):
        return math.isfinite(float(value))
    raise RuntimeError(f"unsupported physical response type: {type(value).__name__}")


def verify_refinement_work(capture, result, config, *, solves_before):
    """Account for every same-factorization solve and every complete response."""
    expected_ids = tuple(f"case-{i+1}" for i in range(config["cases"]))
    rows = capture.completed_case_solve_counts
    if (result.status != "completed" or result.failure_reason is not None or
            result.factorization_count != 1 or result.maximum_rhs_chunk != 1 or
            solves_before != 0 or len(result.cases) != 100 or len(rows) != 100 or
            tuple(case.case_id for case in result.cases) != expected_ids or
            tuple(row["case_id"] for row in rows) != expected_ids):
        raise RuntimeError("refinement batch is incomplete or changed its work contract")
    corrections = tuple(row["corrective_solve_attempts"] for row in rows)
    if (capture.primary_solve_attempts != 100 or
            any(row["primary_solve_attempts"] != 1 for row in rows) or
            any(count not in (0, 1) for count in corrections) or
            capture.corrective_solve_attempts != sum(corrections) or
            result.solve_count != 100 + sum(corrections) or
            not 100 <= result.solve_count <= 200):
        raise RuntimeError("refinement solve work is missing, excessive or unaccounted")
    responses = (tuple(result.cases) + tuple(result.displacement_envelopes) +
                 tuple(result.affine_reaction_envelopes) +
                 tuple(result.member_resultant_envelopes))
    if not _finite_physical_response(responses):
        raise RuntimeError("refinement physical response is nonfinite")
    return dict(primary_solves=100, corrective_solves=sum(corrections),
                total_solves=result.solve_count,
                corrections_by_case=tuple(zip(expected_ids, corrections)))


def verify_beam_rich_recovery(result, *, selected_member, selected_pad):
    """Count complete selected physical recovery without retaining large arrays."""
    stations = components = wrenches = displacements = 0
    for case in result.cases:
        if (len(case.selected_displacements) != 2 or
                len(case.beam_resultants_local) != 1 or
                case.beam_resultants_local[0][0] != selected_member or
                len(case.footprint_wrenches) != 1 or
                case.footprint_wrenches[0].footprint_id != selected_pad):
            raise RuntimeError("beam-rich selected physical recovery is incomplete")
        station_values = case.beam_resultants_local[0][1]
        if len(station_values) != 3 or any(len(values) != 6 for values in station_values):
            raise RuntimeError("beam-rich member station recovery is incomplete")
        stations += len(station_values)
        components += sum(len(values) for values in station_values)
        wrenches += len(case.footprint_wrenches)
        displacements += len(case.selected_displacements)
    recovery = dict(cases=len(result.cases), selected_member_stations=stations,
        selected_member_components=components, selected_footprint_wrenches=wrenches,
        selected_displacement_values=displacements,
        member_resultant_envelopes=len(result.member_resultant_envelopes))
    if recovery != dict(cases=100, selected_member_stations=300,
                        selected_member_components=1_800,
                        selected_footprint_wrenches=100,
                        selected_displacement_values=200,
                        member_resultant_envelopes=18):
        raise RuntimeError("beam-rich selected recovery count differs from target")
    return recovery


def verify_installed_child_record(outdir, config, site, identity, *,
                                  run_profile="installed_large_dev"):
    """Require a completed child record before a zero exit can pass the profile."""
    try:
        record = json.loads((Path(outdir) / "child.json").read_text(encoding="utf-8"))
        if record["status"] != "passed_development_probe":
            raise ValueError("installed child did not report a completed probe")
        if record["config"] != json.loads(json.dumps(config)):
            raise ValueError("installed child input differs from parent input")
        beam_rich = run_profile == "installed_beam_rich_dev"
        expected = dict(BEAM_RICH_EXPECTED_COUNTS if beam_rich else INSTALLED_EXPECTED_COUNTS)
        if any(record["inventory"][key] != value for key, value in expected.items()):
            raise ValueError("installed child generated inventory differs from target")
        compiled_expected = dict(BEAM_RICH_COMPILED_COUNTS if beam_rich
                                 else INSTALLED_COMPILED_COUNTS)
        if record["compiled_inventory"] != compiled_expected:
            raise ValueError("installed child compiled inventory differs from target")
        runtime = record["runtime"]
        verify_python_optimization(runtime["python_optimize"])
        verify_installed_runtime(runtime)
        verify_installed_import_origin(runtime["anyfesolver_file"], site)
        if runtime["anyfesolver_sha256"] != identity["installed_module_sha256"]["__init__.py"]:
            raise ValueError("installed child imported a different anyfesolver package")
        if run_profile in ("installed_large_refinement_dev", "installed_beam_rich_dev"):
            if record["run_profile"] != run_profile:
                raise ValueError("refinement child used a different run profile")
            work, batch = record["refinement_work"], record["batch"]
            expected_ids = tuple(f"case-{i+1}" for i in range(100))
            by_case = work["corrections_by_case"]
            if (len(by_case) != 100 or
                    tuple(row[0] for row in by_case) != expected_ids or
                    any(row[1] not in (0, 1) for row in by_case)):
                raise ValueError("refinement child correction inventory is invalid")
            corrections = sum(row[1] for row in by_case)
            if (work["primary_solves"] != 100 or work["corrective_solves"] != corrections or
                    work["total_solves"] != 100 + corrections or
                    batch["status"] != "completed" or batch["case_count"] != 100 or
                    batch["failure_reason"] is not None or
                    batch["factorization_count"] != 1 or
                    batch["maximum_rhs_chunk"] != 1 or
                    batch["solve_count"] != work["total_solves"] or
                    batch["captured_primary_solve_attempts"] != 100 or
                    batch["captured_corrective_solve_attempts"] != corrections or
                    len(batch["accepted_case_solve_counts"]) != 100 or
                    any(row["primary_solve_attempts"] != 1
                        for row in batch["accepted_case_solve_counts"]) or
                    tuple((row["case_id"], row["corrective_solve_attempts"])
                          for row in batch["accepted_case_solve_counts"]) !=
                          tuple((row[0], row[1]) for row in by_case) or
                    not _finite_physical_response(batch["selected_displacements"]) or
                    not _finite_physical_response(batch["maximum_constraint_gap"])):
                raise ValueError("refinement child work or physical response is invalid")
            if beam_rich:
                recovery = record["beam_rich_recovery"]
                if recovery != dict(cases=100, selected_member_stations=300,
                                    selected_member_components=1_800,
                                    selected_footprint_wrenches=100,
                                    selected_displacement_values=200,
                                    member_resultant_envelopes=18):
                    raise ValueError("beam-rich selected recovery inventory is invalid")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError(f"invalid installed child record: {exc}") from exc
    return record


def verify_installed_runtime(runtime):
    if runtime["affinity"] != [0]:
        raise RuntimeError("installed child CPU0 affinity is not effective")
    if runtime["NUMBA_DISABLE_JIT"] != "1" or any(
            runtime["thread_env"][name] != "1" for name in THREAD_ENV):
        raise RuntimeError("installed child numerical thread or JIT settings are not effective")
    if any(pool.get("num_threads") != 1 for pool in runtime["threadpools"]):
        raise RuntimeError("installed child has a numerical thread pool above one thread")


def verify_python_optimization(level):
    if level != 0:
        raise RuntimeError("Python optimization must be disabled for assert-based verification")


def record_child_optimize(receipt, outdir):
    child_file = Path(outdir) / "child.json"
    receipt["effective_child_python_optimize"] = None
    if child_file.exists():
        try:
            child_record = json.loads(child_file.read_text(encoding="utf-8"))
            receipt["effective_child_python_optimize"] = child_record.get("runtime", {}).get("python_optimize")
        except (OSError, ValueError, AttributeError, TypeError):
            receipt["effective_child_python_optimize"] = None


class SpawnedAffinityRestoreError(RuntimeError):
    def __init__(self, process, cause):
        super().__init__(f"supervisor affinity restore failed after child creation: {cause}")
        self.process = process


def spawn_with_cpu0(process_factory, supervisor, command, **kwargs):
    """Make the child inherit CPU0 affinity from the parent at creation."""
    original = supervisor.cpu_affinity()
    process = None
    try:
        supervisor.cpu_affinity([0])
        if supervisor.cpu_affinity() != [0]:
            raise RuntimeError("supervisor CPU0 affinity did not take effect")
        process = process_factory(command, **kwargs)
        return process
    finally:
        try:
            supervisor.cpu_affinity(original)
        except BaseException as exc:
            if process is not None:
                raise SpawnedAffinityRestoreError(process, exc) from exc
            raise


def remaining_wait(deadline):
    return 5.0 if deadline is None else max(0.0, min(5.0, deadline - time.perf_counter()))


def child_time_limit(profile, used_s, *, installed=False):
    available = profile.total_seconds - used_s - profile.reserve_seconds
    margin = INSTALLED_CLEANUP_MARGIN_SECONDS if installed else 0.0
    limit = min(profile.child_seconds, available - margin)
    if limit <= 0:
        raise RuntimeError("no development budget remains after reserve and cleanup margin")
    return limit, available, margin


def terminate_process_tree(process, psutil_module, *, deadline=None):
    """Stop descendants, then reap the child within the remaining wall allowance."""
    try:
        root = psutil_module.Process(process.pid)
    except psutil_module.Error:
        root = None
    if root is not None:
        try:
            descendants = root.children(recursive=True)
        except psutil_module.Error:
            descendants = ()
        for descendant in descendants:
            try: descendant.kill()
            except psutil_module.Error: pass
        try: root.kill()
        except psutil_module.Error: pass
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        process.kill()
    try: process.wait(timeout=remaining_wait(deadline))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=remaining_wait(deadline))


def supervise_child(process, psutil_module, started, limit, max_rss_bytes):
    """Propagate monitoring errors so the parent can fail, stop and charge."""
    watched = psutil_module.Process(process.pid)
    if watched.cpu_affinity() != [0]:
        return "child_did_not_inherit_cpu0", 0
    peak = 0
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        if elapsed >= limit:
            return "timeout", peak
        family = [watched] + watched.children(recursive=True)
        rss = sum(p.memory_info().rss for p in family if p.is_running())
        peak = max(peak, rss)
        if peak >= max_rss_bytes:
            return "memory_limit", peak
        time.sleep(min(.1, max(0., limit - elapsed)))
    return None, peak


def record_spawn_failure(receipt, prior, budget_file, outdir, elapsed, exc, *,
                         child_created, exit_code=None):
    """Preserve the pre/post-spawn distinction and charge created children."""
    status = "post_spawn_affinity_restore_failed" if child_created else "spawn_refused"
    receipt.update(status=status, exit_code=exit_code,
        process_wall_s=elapsed if child_created else 0.,
        attempt_wall_s=elapsed, sampled_peak_process_tree_rss_bytes=None,
        child_record_exists=(Path(outdir) / "child.json").exists(),
        failure_reason=f"{type(exc).__name__}: {exc}")
    record_child_optimize(receipt, outdir)
    write_json(Path(outdir) / "receipt.json", receipt)
    if child_created:
        prior["used_s"] += elapsed
        prior["attempts"].append(dict(out=str(outdir), wall_s=elapsed,
                                      status=status, exit_code=exit_code))
        write_json(budget_file, prior)
    return status


def settings(size, cases):
    if size == "small":
        return dict(nx=4, ny=6, line_rows=(0, 4), patch_x=(1,), cases=cases)
    if size == "medium":
        return dict(nx=20, ny=20, line_rows=(2, 12), patch_x=(5, 15), cases=cases)
    if size == "scale":
        return dict(nx=50, ny=50, line_rows=tuple(range(4, 50, 10)),
                    patch_x=tuple(range(5, 50, 10)), cases=cases)
    if size == "beam_rich":
        return dict(nx=20, ny=20, line_rows=tuple(range(0, 20, 2)),
                    patch_x=tuple(range(1, 20, 2)),
                    beam_segments_per_shell=5, cases=cases)
    return dict(nx=100, ny=100, line_rows=tuple(range(4, 100, 10)),
                patch_x=tuple(range(5, 100, 10)), cases=cases)


def model(config):
    import numpy as np
    from scipy import sparse
    from anyfesolver import stiffened_shell as api

    nx, ny = config["nx"], config["ny"]
    shell_id = lambda x, y: 1 + y * (nx + 1) + x
    nodes = {shell_id(x, y): (float(x), float(y), 0.)
             for y in range(ny + 1) for x in range(nx + 1)}
    shells = tuple(api.Q4PanelShell(
        f"q4-{y}-{x}", (shell_id(x, y), shell_id(x + 1, y),
                       shell_id(x + 1, y + 1), shell_id(x, y + 1)),
        1.0e10, .3, .15)
        for y in range(ny) for x in range(nx))
    section = api.LinearSection.isotropic(E=2.0e10, nu=.25, area=.04,
        Iy=.04*.2**2/12., Iz=.04*.2**2/12., J=2*.04*.2**2/12., ky=.8, kz=.8)
    beam_base = (nx + 1) * (ny + 1) + 1
    subdivisions = config.get("beam_segments_per_shell", 1)
    beam_segments = nx * subdivisions
    beam_id = lambda line, x: beam_base + line * (beam_segments + 1) + x
    members = tuple(api.PhysicalB2Member(
        f"b2-{line}-{x}", f"square-{line}-{x}",
        (beam_id(line, x), beam_id(line, x + 1)),
        ((float(x / subdivisions), row + .5, .2),
         (float((x + 1) / subdivisions), row + .5, .2)),
        (0., 1., 0.), "square", section)
        for line, row in enumerate(config["line_rows"]) for x in range(beam_segments))
    footprints = tuple(api.B2PhysicalRigidFootprint(
        f"pad-{line}-{x}", (shell_id(x, row), shell_id(x + 1, row),
                            shell_id(x + 1, row + 1), shell_id(x, row + 1)),
        beam_id(line, x * subdivisions), (0., 0., 1.),
        ((1., 0.), (0., 1.), (0., 0.)),
        ((shell_id(x, row), shell_id(x + 1, row),
          shell_id(x + 1, row + 1), shell_id(x, row + 1)),))
        for line, row in enumerate(config["line_rows"]) for x in config["patch_x"])
    n_shell = len(nodes)
    n_beam = len(config["line_rows"]) * (beam_segments + 1)
    n_dof = 6 * (n_shell + n_beam)
    # Fully clamp the x=0 shell edge. Beam lines attach through explicit pads.
    fixed = {6 * y * (nx + 1) + component
             for y in range(ny + 1) for component in range(6)}
    free = np.fromiter((i for i in range(n_dof) if i not in fixed), int)
    transformation = sparse.eye(n_dof, format="csc")[:, free].tocsr()
    definition = api.B2PhysicalRigidFootprintDefinition(
        api.B2_E4_PL_FAST_LINEAR_V1, nodes, shells, members, footprints,
        transformation, np.zeros(n_dof))
    inventory = dict(shell_nodes=n_shell, q4_shells=len(shells),
        beam_nodes=n_beam, physical_b2_members=len(members),
        attached_patches=len(footprints), footprint_rows=20 * len(footprints),
        dofs=n_dof, free_dofs=len(free), fixed_dofs=len(fixed),
        connected_incidence="shared Q4 edges; each B2 line sequential; one physical pad per selected line node",
        beam_eccentricity_m=.2)
    if subdivisions != 1:
        inventory["beam_segments_per_shell"] = subdivisions
    return definition, inventory, shell_id


class BatchSolveCapture:
    """Keep the current primary solve and a bounded correction history."""

    MAX_CORRECTIVE_SNAPSHOTS = 8

    def __init__(self, cases):
        self.cases = cases
        self.next_index = 0
        self.latest = None
        self.primary_solve_attempts = 0
        self.corrective_solve_attempts = 0
        self.completed_case_solve_counts = []

    def solve(self, original_solve, rhs):
        import numpy as np
        snapshot = None
        if self.next_index < len(self.cases):
            case_id, applied = self.cases[self.next_index]
            if self.latest is None or self.latest["case_id"] != case_id:
                self.latest = dict(case_id=case_id,
                    applied=np.array(applied, copy=True),
                    scaled_rhs=np.array(rhs, copy=True),
                    primary_status="started", corrective_solve_count=0,
                    corrective_solves=[])
                self.primary_solve_attempts += 1
                snapshot = self.latest
            else:
                self.corrective_solve_attempts += 1
                self.latest["corrective_solve_count"] += 1
                snapshot = dict(ordinal=self.latest["corrective_solve_count"],
                    scaled_rhs=np.array(rhs, copy=True), status="started")
                stored = self.latest["corrective_solves"]
                if len(stored) < self.MAX_CORRECTIVE_SNAPSHOTS:
                    stored.append(snapshot)
                else:
                    stored[-1] = snapshot  # preserve the most recent failed attempt
        try:
            answer = original_solve(rhs)
        except BaseException as exc:
            if snapshot is not None:
                snapshot["primary_status" if snapshot is self.latest else "status"] = "raised"
                snapshot["error"] = f"{type(exc).__name__}: {exc}"
            raise
        if snapshot is not None:
            snapshot["raw_answer"] = np.array(answer, copy=True)
            snapshot["primary_status" if snapshot is self.latest else "status"] = "returned"
        return answer

    def on_case(self, case_id):
        if (self.next_index >= len(self.cases) or
                case_id != self.cases[self.next_index][0]):
            raise RuntimeError("batch completion order changed")
        self.completed_case_solve_counts.append(dict(case_id=case_id,
            primary_solve_attempts=int(self.latest is not None and self.latest["case_id"] == case_id),
            corrective_solve_attempts=0 if self.latest is None or self.latest["case_id"] != case_id
                else self.latest["corrective_solve_count"]))
        self.next_index += 1


def failed_solve_diagnostic(owner, expected_case_id, captured, *, accurate_residual=None):
    """Read-only reconstruction of the existing public admission check."""
    import numpy as np
    if captured is None or captured["case_id"] != expected_case_id:
        return dict(available=False, expected_case_id=expected_case_id,
                    captured_case_id=None if captured is None else captured["case_id"],
                    reason="no solve snapshot for the failed case")
    corrections = captured.get("corrective_solves", ())
    corrective_count = captured.get("corrective_solve_count", len(corrections))
    last_correction = corrections[-1] if corrections else None
    capture_summary = dict(primary_status=captured.get("primary_status",
            "returned" if "raw_answer" in captured else "missing"),
        corrective_solve_attempts=corrective_count,
        corrective_snapshots_stored=len(corrections),
        corrective_snapshots_omitted=corrective_count-len(corrections),
        last_corrective_ordinal=None if last_correction is None else last_correction["ordinal"],
        last_corrective_status=None if last_correction is None else last_correction["status"])
    if capture_summary["primary_status"] != "returned":
        return dict(available=False, expected_case_id=expected_case_id,
            captured_case_id=captured["case_id"], solve_capture=capture_summary,
            reason="primary solve did not return an answer")
    applied = captured["applied"]
    scaled_rhs, raw_answer = captured["scaled_rhs"], captured["raw_answer"]
    rhs = scaled_rhs / owner._factor_scale
    answer = owner._factor_scale * raw_answer
    q, lam = answer[:-len(owner._cu0)], answer[-len(owner._cu0):]
    u = np.asarray(owner._T @ q).reshape(-1) + owner._u0
    gap = np.asarray(owner._C @ u).reshape(-1)
    force = np.asarray(owner._K @ u).reshape(-1)
    constraint_force = np.asarray(owner._C.T @ lam).reshape(-1)
    projected = owner._project(force + constraint_force - applied)
    gap_scale = np.asarray(owner._abs_C @ np.abs(u)).reshape(-1)
    physical_force_scale = np.asarray(owner._abs_K @ np.abs(u) +
        owner._abs_Ct @ np.abs(lam) + np.abs(applied)).reshape(-1)
    equilibrium_scale = np.asarray(owner._abs_Tt @ physical_force_scale).reshape(-1)
    by_node = physical_force_scale.reshape(-1, 6)
    force_roundoff = float(np.max(by_node[:, :3]))
    moment_roundoff = float(np.max(by_node[:, 3:]))
    physical_roundoff = np.tile((force_roundoff,)*3 + (moment_roundoff,)*3,
                                len(by_node))
    projected_roundoff = np.asarray(owner._abs_Tt @ physical_roundoff).reshape(-1)
    roundoff_fraction = len(rhs) * np.finfo(float).eps
    equilibrium_tolerance = 1e-9 * equilibrium_scale + roundoff_fraction * projected_roundoff
    gap_tolerance = 1e-9 * gap_scale
    explicit_rhs = np.abs(owner._project(applied))
    resolution_floor = roundoff_fraction * projected_roundoff
    active = explicit_rhs > 0.
    needs_certification = bool(np.any(resolution_floor[active] > 1e-9*explicit_rhs[active]))
    certified = owner._explicit_load_resolved(answer, rhs, explicit_rhs) if needs_certification else False
    relaxed_gap = False
    if np.any(np.abs(gap) > gap_tolerance):
        if certified or owner._explicit_load_resolved(answer, rhs, explicit_rhs):
            gap_tolerance = owner._footprint_gap_tolerance(u, gap_scale, roundoff_fraction)
            relaxed_gap = True
    for item in owner._footprint_incidence:
        width = getattr(item, "row_count", 6)
        rows = slice(item.row_start, item.row_start + width)
        cap = 1e-9*np.array((owner._compatibility_length,)*3 + (1.,)*(width-3))
        gap_tolerance[rows] = np.minimum(gap_tolerance[rows], cap)
    gap_violation = np.abs(gap) > gap_tolerance
    equilibrium_violation = np.abs(projected) > equilibrium_tolerance
    gap_index = int(np.argmax(np.abs(gap) / np.maximum(gap_tolerance, np.finfo(float).tiny)))
    eq_index = int(np.argmax(np.abs(projected) / np.maximum(equilibrium_tolerance, np.finfo(float).tiny)))
    accurate_row = dict(available=False, row=eq_index,
        raw_projected=float(projected[eq_index]),
        unchanged_tolerance=float(equilibrium_tolerance[eq_index]))
    try:
        if accurate_residual is None:
            from anyfesolver._stiffened_shell_b2_q4_panel_experimental import (
                _bounded_accurate_residual as accurate_residual)
        # Recompute the original unscaled production RHS rather than undoing
        # binary64 factor scaling on the captured solver input.
        unscaled_rhs = np.r_[owner._project(applied-owner._ku0), -owner._cu0]
        if owner._resolution_kkt is None:
            owner._explicit_load_resolved(answer, unscaled_rhs, explicit_rhs)
        kkt = owner._resolution_kkt
        if kkt is None or kkt.shape[0] != len(unscaled_rhs):
            raise RuntimeError("unscaled resolution KKT is unavailable")
        defect, upper = accurate_residual(
            kkt[eq_index:eq_index+1], answer[:, None],
            unscaled_rhs[eq_index:eq_index+1, None])
        accurate_row.update(available=True,
            unscaled_rhs=float(unscaled_rhs[eq_index]),
            signed_defect=float(defect[0, 0]),
            outward_abs_upper=float(upper[0, 0]),
            within_unchanged_tolerance=bool(upper[0, 0] <= equilibrium_tolerance[eq_index]))
    except Exception as exc:
        accurate_row["reason"] = f"{type(exc).__name__}: {exc}"
    return dict(available=True, case_id=expected_case_id,
        source="private read-only reconstruction after public failure",
        diagnostic_solution="primary_solve_only", solve_capture=capture_summary,
        finite=bool(np.isfinite(gap).all() and np.isfinite(projected).all()),
        needs_explicit_certification=needs_certification, certified=certified,
        relaxed_gap=relaxed_gap,
        gap_violation_count=int(np.count_nonzero(gap_violation)),
        equilibrium_violation_count=int(np.count_nonzero(equilibrium_violation)),
        worst_gap_row=gap_index, worst_gap_abs=float(abs(gap[gap_index])),
        worst_gap_tolerance=float(gap_tolerance[gap_index]),
        worst_equilibrium_dof=eq_index, worst_equilibrium_abs=float(abs(projected[eq_index])),
        worst_equilibrium_tolerance=float(equilibrium_tolerance[eq_index]),
        accurate_kkt_worst_equilibrium_row=accurate_row,
        max_abs_gap=float(np.max(np.abs(gap))),
        max_abs_projected=float(np.max(np.abs(projected))),
        roundoff_fraction=float(roundoff_fraction))


def child(config, out, *, profile=False, run_profile="source_dev", installed_site=None):
    try:
        verify_python_optimization(sys.flags.optimize)
    except RuntimeError as exc:
        write_json(out, dict(config=config, status="python_optimization_refused",
                             runtime=dict(python_optimize=sys.flags.optimize),
                             failure_reason=str(exc)))
        return 1
    import numpy as np
    import psutil
    import scipy
    import threadpoolctl
    import anyfesolver
    if run_profile in INSTALLED_PROFILES:
        try:
            origin = verify_installed_import_origin(anyfesolver.__file__, installed_site)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            write_json(out, dict(config=config, status="installed_import_origin_refused",
                                 anyfesolver_file=str(anyfesolver.__file__),
                                 runtime=dict(python_optimize=sys.flags.optimize),
                                 failure_reason=f"{type(exc).__name__}: {exc}"))
            return 1
    from anyfesolver import stiffened_shell as api
    from anyfesolver import _stiffened_shell_b2_q4_panel_experimental as panel_module
    started = time.perf_counter()
    record = dict(config=config, status="started", phases={},
                  runtime=dict(python=sys.version, python_optimize=sys.flags.optimize,
                    numpy=np.__version__,
                    scipy=scipy.__version__, anyfesolver_file=str(Path(anyfesolver.__file__).resolve()),
                    anyfesolver_sha256=digest(anyfesolver.__file__),
                    thread_env={k: os.environ.get(k) for k in THREAD_ENV},
                    NUMBA_DISABLE_JIT=os.environ.get("NUMBA_DISABLE_JIT"),
                    affinity=psutil.Process().cpu_affinity(),
                    threadpools=threadpoolctl.threadpool_info()))
    if run_profile in INSTALLED_PROFILES:
        record["run_profile"] = run_profile
        record["installed_site"] = str(Path(installed_site).resolve())
        record["runtime"]["anyfesolver_file"] = str(origin)
        try:
            verify_installed_runtime(record["runtime"])
        except RuntimeError as exc:
            record.update(status="runtime_control_refused", failure_reason=str(exc))
            write_json(out, record)
            return 1
    owner = None
    try:
        t = time.perf_counter()
        definition, inventory, shell_id = model(config)
        record["inventory"] = inventory
        if run_profile in INSTALLED_PROFILES:
            verify_installed_inventory(config, inventory, definition=definition,
                                       run_profile=run_profile)
        record["phases"]["generation_s"] = time.perf_counter() - t
        write_json(out, record)
        t = time.perf_counter()
        factorization_times = []
        original_factorize = panel_module.factorize_cached
        def timed_factorize(*args, **kwargs):
            factor_started = time.perf_counter()
            try:
                return original_factorize(*args, **kwargs)
            finally:
                factorization_times.append(time.perf_counter() - factor_started)
        panel_module.factorize_cached = timed_factorize
        profiler = None
        try:
            if profile:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            owner = api.create_analysis(definition=definition)
            if run_profile in INSTALLED_PROFILES:
                record["compiled_inventory"] = verify_installed_inventory(
                    config, inventory, owner=owner, run_profile=run_profile)
            construction_s = time.perf_counter() - t
        finally:
            panel_module.factorize_cached = original_factorize
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(str(out.parent / "compile.profile"))
        record["phases"]["compile_factor_s"] = construction_s
        record["phases"]["factorization_s"] = sum(factorization_times)
        record["phases"]["nonfactor_construction_s"] = construction_s - sum(factorization_times)
        record["factorization_calls"] = len(factorization_times)
        assert len(factorization_times) == 1
        record["owner_counts_before_batch"] = dict(
            factorization=owner._handle.factorization_count,
            solves=owner._handle.solve_count)
        write_json(out, record)
        n_dof = inventory["dofs"]
        load = np.zeros(n_dof)
        # Apply downward shell load at the far free edge, away from supports.
        mid = config["ny"] // 2
        chosen_node = shell_id(config["nx"], mid)
        load[6 * (chosen_node - 1) + 2] = -1.
        selected_dofs = (6 * (chosen_node - 1) + 2,
                         6 * (chosen_node - 1) + 0)
        selected_pad = definition.physical_footprints[0].footprint_id
        selected_member = definition.members[0].member_id
        t = time.perf_counter()
        cases = tuple((f"case-{i+1}", (i + 1) * load)
                      for i in range(config["cases"]))
        capture = BatchSolveCapture(cases)
        original_solve = owner._handle.solve
        owner._handle.solve = lambda rhs: capture.solve(original_solve, rhs)
        try:
            result = owner.solve_linear_many(cases, selected_dofs=selected_dofs,
                selected_footprints=api.B2PhysicalRigidFootprintSelection((selected_pad,)),
                recover_members={selected_member: (0., .5, 1.)},
                on_case=capture.on_case)
        finally:
            owner._handle.solve = original_solve
        record["phases"]["batch_s"] = time.perf_counter() - t
        failed_case_id = (cases[len(result.cases)][0]
            if result.status == "failed" and len(result.cases) < len(cases) else None)
        record["batch"] = dict(status=result.status,
            failure_reason=result.failure_reason, case_count=len(result.cases),
            accepted_prefix_case_ids=tuple(case.case_id for case in result.cases),
            failed_case_id=failed_case_id,
            factorization_count=result.factorization_count,
            solve_count=result.solve_count,
            captured_primary_solve_attempts=capture.primary_solve_attempts,
            captured_corrective_solve_attempts=capture.corrective_solve_attempts,
            accepted_case_solve_counts=capture.completed_case_solve_counts,
            maximum_rhs_chunk=result.maximum_rhs_chunk,
            selected_dofs=selected_dofs, selected_pad=selected_pad,
            selected_member=selected_member,
            selected_resultant_envelopes=len(result.member_resultant_envelopes),
            selected_displacements=[case.selected_displacements for case in result.cases],
            maximum_constraint_gap=max((abs(v) for case in result.cases
                for v in case.constraint_gaps), default=0.))
        if failed_case_id is not None:
            record["failed_solve_diagnostic"] = failed_solve_diagnostic(
                owner, failed_case_id, capture.latest)
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == 1
        if run_profile in ("installed_large_refinement_dev", "installed_beam_rich_dev"):
            record["refinement_work"] = verify_refinement_work(
                capture, result, config,
                solves_before=record["owner_counts_before_batch"]["solves"])
            if run_profile == "installed_beam_rich_dev":
                record["beam_rich_recovery"] = verify_beam_rich_recovery(
                    result, selected_member=selected_member, selected_pad=selected_pad)
        else:
            assert result.solve_count == config["cases"]
        assert len(result.cases) == config["cases"]
        assert len(result.member_resultant_envelopes) == 18
        record["status"] = "passed_development_probe"
    except BaseException as exc:
        import traceback
        record["status"] = "failed_development_probe"
        record["error"] = dict(type=type(exc).__name__, message=str(exc),
                               traceback=traceback.format_exc())
    finally:
        if owner is not None:
            owner.close()
        record["wall_s"] = time.perf_counter() - started
        record["peak_wset_bytes"] = psutil.Process().memory_info().peak_wset
        write_json(out, record)
    return 0 if record["status"] == "passed_development_probe" else 1


def run_parent(args):
    verify_python_optimization(sys.flags.optimize)
    import psutil
    run_profile, limits, budget_file, installed_site = resolve_run(args)
    if args.cases not in (1, 10, 100):
        raise ValueError("case count must be 1, 10 or 100")
    config = settings(args.size, args.cases)
    identity = installed_site_identity(installed_site, run_profile=run_profile) if installed_site else None
    outdir = Path(args.out).resolve()
    registration_identity = (verify_beam_rich_registration(config, identity, limits,
        budget_file, outdir) if run_profile == "installed_beam_rich_dev" else None)
    outdir.mkdir(parents=True, exist_ok=False)
    prior = json.loads(budget_file.read_text()) if budget_file.exists() else {"used_s": 0., "attempts": []}
    limit, available, cleanup_margin = child_time_limit(
        limits, prior["used_s"], installed=installed_site is not None)
    env = os.environ.copy()
    for name in THREAD_ENV:
        env[name] = "1"
    env["NUMBA_DISABLE_JIT"] = "1"
    env["PYTHONOPTIMIZE"] = "0"
    env["PYTHONPATH"] = str(installed_site) if installed_site else str(ROOT / "src") + os.pathsep + str(ROOT)
    source_paths = [Path(__file__), ROOT / "src/anyfesolver/stiffened_shell.py",
                    ROOT / "src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py",
                    ROOT / "src/anyfesolver/_stiffened_shell_b2_physical_rigid_footprint.py"]
    command = [sys.executable, str(Path(__file__).resolve()), "--child",
               "--size", args.size, "--cases", str(args.cases),
               "--out", str(outdir / "child.json")]
    if installed_site:
        command.extend(("--run-profile", run_profile, "--installed-site", str(installed_site)))
    if args.profile:
        command.append("--profile")
    receipt = dict(command=command, cwd=str(ROOT), config=config,
        effective_parent_python_optimize=sys.flags.optimize,
        requested_child_python_optimize=0,
        input_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        limits=dict(cpu=0, numerical_threads=1, jit_disabled=True,
                    max_rss_bytes=limits.max_rss_bytes, child_seconds=limit,
                    configured_child_seconds=limits.child_seconds,
                    cleanup_margin_seconds=cleanup_margin,
                    total_seconds=limits.total_seconds, reserve_seconds=limits.reserve_seconds,
                    used_before_seconds=prior["used_s"]), status="started")
    if installed_site:
        receipt.update(run_profile=run_profile, budget_file=str(budget_file),
                       pythonpath=env["PYTHONPATH"], **identity)
    if registration_identity:
        receipt.update(registration_identity)
    write_json(outdir / "receipt.json", receipt)
    started = time.perf_counter()
    deadline = started + available
    peak = 0
    reason = None
    with (outdir / "stdout.txt").open("wb") as stdout, (outdir / "stderr.txt").open("wb") as stderr:
        try:
            process = spawn_with_cpu0(subprocess.Popen, psutil.Process(), command,
                                      cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        except SpawnedAffinityRestoreError as exc:
            try:
                terminate_process_tree(exc.process, psutil, deadline=deadline)
            except (psutil.Error, OSError, subprocess.TimeoutExpired) as stop_exc:
                receipt["termination_error"] = f"{type(stop_exc).__name__}: {stop_exc}"
            elapsed = time.perf_counter() - started
            status = record_spawn_failure(receipt, prior, budget_file, outdir,
                elapsed, exc, child_created=True,
                exit_code=exc.process.returncode)
            print(json.dumps(dict(status=status, wall_s=elapsed,
                out=str(outdir), budget_used_s=prior["used_s"])))
            return 1
        except (psutil.Error, OSError, RuntimeError) as exc:
            elapsed = time.perf_counter() - started
            status = record_spawn_failure(receipt, prior, budget_file, outdir,
                elapsed, exc, child_created=False)
            print(json.dumps(dict(status=status, out=str(outdir))))
            return 1
        try:
            reason, peak = supervise_child(process, psutil, started, limit,
                                           limits.max_rss_bytes)
        except (psutil.Error, OSError, RuntimeError) as exc:
            reason = "supervision_failed"
            receipt["supervision_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if reason is None:
                process.wait(timeout=remaining_wait(deadline))
            else:
                terminate_process_tree(process, psutil, deadline=deadline)
        except (psutil.Error, OSError, subprocess.TimeoutExpired) as exc:
            receipt["termination_error"] = f"{type(exc).__name__}: {exc}"
            if reason is None:
                reason = "reap_failed"
            try:
                terminate_process_tree(process, psutil, deadline=deadline)
            except (psutil.Error, OSError, subprocess.TimeoutExpired) as retry_exc:
                receipt["termination_error"] += (
                    f"; fallback {type(retry_exc).__name__}: {retry_exc}")
    elapsed = time.perf_counter() - started
    if elapsed > available:
        receipt["budget_overrun_seconds"] = elapsed - available
        reason = "budget_overrun"
    if installed_site and reason is None and process.returncode == 0:
        try:
            verify_installed_child_record(outdir, config, installed_site, identity,
                                          run_profile=run_profile)
        except RuntimeError as exc:
            reason = "child_receipt_invalid"
            receipt["child_receipt_error"] = str(exc)
    receipt.update(status=reason or ("exited" if process.returncode == 0 else "child_failed"),
                   exit_code=process.returncode, process_wall_s=elapsed,
                   sampled_peak_process_tree_rss_bytes=peak,
                   child_record_exists=(outdir / "child.json").exists())
    record_child_optimize(receipt, outdir)
    write_json(outdir / "receipt.json", receipt)
    prior["used_s"] += elapsed
    prior["attempts"].append(dict(out=str(outdir), wall_s=elapsed,
                                  status=receipt["status"], exit_code=process.returncode))
    write_json(budget_file, prior)
    print(json.dumps(dict(status=receipt["status"], wall_s=elapsed,
                          peak_rss_bytes=peak, out=str(outdir),
                          budget_used_s=prior["used_s"])))
    return 0 if receipt["status"] == "exited" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--run-profile", choices=tuple(RUN_PROFILES), default="source_dev")
    parser.add_argument("--installed-site")
    parser.add_argument("--size", choices=("small", "medium", "scale", "large", "beam_rich"), required=True)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget")
    args = parser.parse_args()
    if args.child:
        return child(settings(args.size, args.cases), Path(args.out), profile=args.profile,
                     run_profile=args.run_profile, installed_site=args.installed_site)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
