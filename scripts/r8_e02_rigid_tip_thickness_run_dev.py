"""Bounded, one-use development execution of the prepared E02 nz=4/8 decks.

The archived nz=2 result is read, never executed. This runner has no E02/F04
qualification authority. Every attempted process retains its raw files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r8_e02_rigid_tip_reference import build_deck, parse_result, CCX_SHA256

CCX = Path("C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/calculix-223/"
           "CalculiX-2.23.0-win-x64/bin/ccx.exe")
ARCHIVE = ROOT / "docs/agent_plans/development/r8-e02-rigid-tip-reference-20261001/raw-v1.zip"
ARCHIVE_MEMBER = "r8-e02-rigid-tip-dev-v3/external-ccx-20x8-s1/Mx/"
EVIDENCE = ROOT / ".development/e02-rigid-thickness-run-20261009-v4"
PRIOR_WALL_S = 2.063392  # v2/v3 runner wall plus focused pytest process wall
EXPECTED = {2: "9dea22c76fd75236c7f53359bf463357b4b359d25cc57e8144386deac15945db",
            4: "d319d0a26bc0762cb3d31dbd01de838f7250104b0b520755229d9b9af9637b76",
            8: "407411bb76bb7838c59488527a03c73023e2ac5fa705ed2322df22d28568832f"}
MAX_TOTAL_S = 180.0
FINAL_RESERVE_S = 60.0
MAX_CCX_S = 45.0
MAX_SOURCE_S = 30.0
MAX_TREE_RSS = 2 * 1024**3


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_inputs() -> tuple[dict, dict, dict]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        old = json.loads(archive.read(ARCHIVE_MEMBER + "result.json"))
        identity = json.loads(archive.read(ARCHIVE_MEMBER + "identity.json"))
        execution = json.loads(archive.read(ARCHIVE_MEMBER + "execution.json"))
    if old["status"] != "PARSED_DEVELOPMENT_ONLY" or old["deck_sha256"] != EXPECTED[2]:
        raise ValueError("archived nz=2 result identity mismatch")
    if identity["deck_sha256"] != EXPECTED[2] or identity["executable_sha256"] != CCX_SHA256:
        raise ValueError("archived nz=2 executable/deck identity mismatch")
    old_deck, old_meta = build_deck(20, 8, 1, "Mx")
    if sha(old_deck.encode("ascii")) != EXPECTED[2]:
        raise ValueError("regenerated nz=2 deck identity mismatch")
    for key in ("geometry_m", "material", "element_type", "applied_force_N",
                "applied_moment_Nm", "expected_root_force_N", "expected_root_moment_Nm"):
        if identity["metadata"][key] != json.loads(json.dumps(old_meta[key])):
            raise ValueError(f"archived nz=2 {key} mismatch")
    return old, old_meta, execution


def deck_inputs(old_meta: dict, nz: int) -> tuple[bytes, dict]:
    deck, meta = build_deck(20, 8, 1, "Mx", nz=nz)
    data = deck.encode("ascii")
    if sha(data) != EXPECTED[nz] or meta["deck_sha256"] != EXPECTED[nz]:
        raise ValueError(f"nz={nz} prepared deck identity mismatch")
    fixed = ("nx", "ny", "sign", "load_name", "geometry_m", "material", "element_type",
             "applied_force_N", "applied_moment_Nm", "expected_root_force_N",
             "expected_root_moment_Nm", "expected_output_time")
    if any(meta[key] != old_meta[key] for key in fixed):
        raise ValueError(f"nz={nz} fixed physical definition changed")
    if any(meta["control_nodes"][meta["controls"][role]] !=
           old_meta["control_nodes"][old_meta["controls"][role]]
           for role in ("root_ref", "root_rot", "patch_ref", "patch_rot")):
        raise ValueError(f"nz={nz} port/control coordinates changed")
    for name, predicate in (("patch", lambda p: p[0] >= .8),
                            ("root", lambda p: p[0] == 0.)):
        pts = [meta["physical_nodes"][node] for node in meta[name + "_nodes"]]
        if not pts or not all(predicate(p) for p in pts):
            raise ValueError(f"nz={nz} {name} physical boundary changed")
    # Deck tail carries identical rigid-body, material, boundary, load and output cards,
    # modulo node IDs. Their exact semantics are checked by the regenerated manifest.
    return data, meta


def physical_checks(parsed: dict) -> dict:
    if parsed["status"] != "PARSED_DEVELOPMENT_ONLY":
        raise ValueError("missing complete parsed output")
    if not math.isfinite(parsed["conjugate_external_work_J"]) or parsed["conjugate_external_work_J"] <= 0:
        raise ValueError("nonpositive/nonfinite external work")
    if any(item["max_gap_beyond_printed_rounding_m"] != 0 for item in parsed["rigid_mapping"].values()):
        raise ValueError("rigid patch/root mapping exceeds output rounding")
    if any(value != 0 for value in parsed["root_ref_displacement_m"] + parsed["root_rot_theta_rad"]):
        raise ValueError("root controls moved")
    # A development sanity screen only; DAT print precision cannot certify 1e-8 balance.
    residual = max(*(abs(v) for v in parsed["root_force_residual_N"]),
                   *(abs(v) for v in parsed["root_moment_residual_Nm"]))
    if residual > 1e-6:
        raise ValueError(f"root wrench development sanity screen failed: {residual:g}")
    return {"complete_sections": True, "positive_work": True,
            "rigid_mapping_within_printed_rounding": True, "fixed_root": True,
            "max_abs_root_wrench_residual_SI": residual,
            "root_wrench_sanity_limit_SI_development_only": 1e-6,
            "relative_balance_1e_minus_8_certified": False}


def process_tree(child: psutil.Process) -> tuple[list[psutil.Process], int]:
    try:
        members = [child, *child.children(recursive=True)]
    except psutil.NoSuchProcess:
        return [], 0
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return members, total


def run_one(nz: int, data: bytes, meta: dict, output: Path, cap: float,
            source_s: float, used_s: float) -> tuple[dict, dict | None]:
    case = output / f"nz{nz}"
    case.mkdir()  # one-use evidence; never overwrite
    (case / "Mx.inp").write_bytes(data)
    save(case / "metadata.json", meta)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env["NUMBA_DISABLE_JIT"] = "1"
    env["CCX_NPROC_RESULTS"] = "1"
    env["CCX_NPROC_EQUATION_SOLVER"] = "1"
    command = [str(CCX), "-i", str(case / "Mx")]
    receipt = {"case": f"20x8/nz={nz}/+Mx=1 N m", "command": command,
               "cwd": str(case), "binary_sha256": CCX_SHA256,
               "binary_version_expected": "2.23", "deck_sha256": sha(data),
               "generator_sha256": sha((ROOT / "scripts/r8_e02_rigid_tip_reference.py").read_bytes()),
               "runner_sha256": sha(Path(__file__).read_bytes()),
               "source_check_wall_s": source_s, "prior_cumulative_wall_s": used_s,
               "cap_s": cap, "total_budget_s": MAX_TOTAL_S, "final_reserve_s": FINAL_RESERVE_S,
               "memory_limit_bytes": MAX_TREE_RSS,
               "environment": {key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                "NUMBA_DISABLE_JIT", "CCX_NPROC_RESULTS",
                                "CCX_NPROC_EQUATION_SOLVER")}}
    parent = psutil.Process()
    parent.cpu_affinity([0])
    receipt["parent_cpu_affinity"] = parent.cpu_affinity()
    start = time.perf_counter_ns()
    exit_code = None
    child_affinity = None
    failure = None
    peak = 0
    with (case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr:
        try:
            child = subprocess.Popen(command, cwd=case, env=env, stdout=stdout, stderr=stderr)
            proc = psutil.Process(child.pid)
            proc.cpu_affinity([0])
            child_affinity = proc.cpu_affinity()
            if child_affinity != [0]:
                failure = "CPU_AFFINITY_MISMATCH"
            while child.poll() is None:
                members, rss = process_tree(proc)
                peak = max(peak, rss)
                wall = (time.perf_counter_ns() - start) / 1e9
                if rss > MAX_TREE_RSS:
                    failure = "TREE_RSS_LIMIT"
                elif wall >= cap:
                    failure = "PROCESS_WALL_TIMEOUT"
                if failure:
                    for member in reversed(members):
                        try:
                            member.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    break
                time.sleep(.01)
            exit_code = child.wait()
        except (OSError, psutil.Error) as exc:
            failure = f"LAUNCH_OR_SUPERVISION_{type(exc).__name__}: {exc}"
            if 'child' in locals() and child.poll() is None:
                child.kill()
                exit_code = child.wait()
    wall = (time.perf_counter_ns() - start) / 1e9
    stdout_text = (case / "stdout.log").read_text(errors="replace")
    stderr_text = (case / "stderr.log").read_text(errors="replace")
    sta = case / "Mx.sta"
    accepted = [line.strip() for line in sta.read_text(errors="replace").splitlines()
                if re.match(r"^\s*\d+\s+\d+\b", line)] if sta.exists() else []
    receipt.update({"wall_s": wall, "exit_code": exit_code,
                    "child_cpu_affinity": child_affinity, "peak_sampled_tree_rss_bytes": peak,
                    "accepted_step_rows": accepted, "stdout_sha256": sha((case / "stdout.log").read_bytes()),
                    "stderr_sha256": sha((case / "stderr.log").read_bytes())})
    parsed = None
    if failure is None and exit_code != 0:
        failure = f"CCX_EXIT_{exit_code}"
    if failure is None and (stderr_text.strip() or len(accepted) != 1):
        failure = "STDERR_OR_ACCEPTED_STEP_INVENTORY"
    if failure is None:
        try:
            parsed = parse_result((case / "Mx.dat").read_text(errors="replace"),
                                  stdout_text, meta)
            save(case / "parsed.json", parsed)
            receipt["physical_checks"] = physical_checks(parsed)
            receipt["binary_version_observed"] = "2.23"  # enforced by parser
        except (OSError, ValueError) as exc:
            failure = f"PARSE_OR_PHYSICAL_{type(exc).__name__}: {exc}"
    receipt["failure"] = failure
    save(case / "receipt.json", receipt)
    return receipt, parsed if failure is None else None


def main() -> int:
    if EVIDENCE.exists():
        raise FileExistsError(f"one-use evidence already exists: {EVIDENCE}")
    EVIDENCE.mkdir(parents=True)
    begin = time.perf_counter_ns()
    summary = {"status": "INCOMPLETE", "classification": "development only; E02/F04 open",
               "evidence": str(EVIDENCE), "prior_failed_attempt_evidence":
               [str(ROOT / ".development/e02-rigid-thickness-run-20261009-v2"),
                str(ROOT / ".development/e02-rigid-thickness-run-20261009-v3")],
               "prior_consumed_wall_s": PRIOR_WALL_S, "attempts": []}
    try:
        old, old_meta, old_execution = source_inputs()
        decks = {nz: deck_inputs(old_meta, nz) for nz in (4, 8)}
        if sha(CCX.read_bytes()) != CCX_SHA256:
            raise ValueError("pinned CalculiX executable changed")
        source_s = (time.perf_counter_ns() - begin) / 1e9
        if source_s > MAX_SOURCE_S:
            raise TimeoutError(f"source checks exceeded {MAX_SOURCE_S}s")
        summary.update({"source_check_wall_s": source_s,
                        "baseline_archive_sha256": sha(ARCHIVE.read_bytes()),
                        "baseline_deck_sha256": old["deck_sha256"],
                        "baseline_work_J": old["conjugate_external_work_J"],
                        "baseline_execution": {key: old_execution.get(key)
                                               for key in ("exit_code", "elapsed_seconds")}})
        used = PRIOR_WALL_S + source_s
        parsed_by_nz = {2: old}
        for nz in (4, 8):
            if MAX_TOTAL_S - used - FINAL_RESERVE_S < MAX_CCX_S:
                raise TimeoutError("insufficient remaining authority for capped numerical job")
            receipt, parsed = run_one(nz, *decks[nz], EVIDENCE, MAX_CCX_S, source_s, used)
            summary["attempts"].append({"nz": nz, "wall_s": receipt["wall_s"],
                                        "failure": receipt["failure"]})
            used += receipt["wall_s"]
            summary["cumulative_process_wall_s"] = used
            summary["remaining_budget_s"] = MAX_TOTAL_S - used
            save(EVIDENCE / "summary.json", summary)
            if parsed is None:
                summary["status"] = f"STOPPED_AFTER_NZ{nz}_INVALID"
                return 2
            parsed_by_nz[nz] = parsed
        works = {nz: parsed_by_nz[nz]["conjugate_external_work_J"] for nz in (2, 4, 8)}
        summary["works_J"] = works
        summary["relative_change_4_vs_2"] = works[4] / works[2] - 1
        summary["relative_change_8_vs_4"] = works[8] / works[4] - 1
        summary["relative_change_8_vs_2"] = works[8] / works[2] - 1
        summary["status"] = "BOTH_PARSED_DEVELOPMENT_ONLY"
        return 0
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return 2
    finally:
        summary["overall_wall_s"] = (time.perf_counter_ns() - begin) / 1e9
        save(EVIDENCE / "summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(main())
