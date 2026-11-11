"""One bounded signed torque screen of the existing knot rigid footprint.

Development evidence only: this is neither the native weak-line connection
nor a section-matched G0 external comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from _b3_e4_ccx_3d_canary_parser import parse_completed_response
from _b3_e4_ccx_3d_tip_work_parser import recover_tip_section_motion
from generate_b3_e4_ccx_3d_knot_wrench import build_loaded_tip_torque_x


WORKTREE = Path(__file__).resolve().parents[1]
EXE = Path("C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/"
           "calculix-223/CalculiX-2.23.0-win-x64/bin/ccx.exe")
EXE_SHA = "27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f"
TIMEOUT_SECONDS = 120


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if digest(EXE) != EXE_SHA:
        raise ValueError("CalculiX executable identity changed")
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=WORKTREE,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(("git", "status", "--porcelain"), cwd=WORKTREE,
                           check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("signed external screen requires a clean source")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = (Path(__file__),
               WORKTREE / "scripts/generate_b3_e4_ccx_3d_knot_wrench.py",
               WORKTREE / "scripts/generate_b3_e4_ccx_3d_knot_patch.py",
               WORKTREE / "scripts/generate_b3_e4_ccx_3d_reciprocity_influences.py",
               WORKTREE / "scripts/generate_b3_e4_ccx_3d_footprint_feasibility.py",
               WORKTREE / "scripts/_b3_e4_ccx_3d_tip_work_parser.py",
               WORKTREE / "scripts/_b3_e4_ccx_3d_canary_parser.py")
    manifest = {
        "status": "EXPLORATORY_NOT_G0_ACCEPTANCE",
        "purpose": "signed B31/S4 knot-footprint tip-Mx response and mean-rotation check",
        "source_head": head,
        "source_sha256": {str(path.relative_to(WORKTREE)): digest(path)
                          for path in sources},
        "executable": str(EXE), "executable_sha256": EXE_SHA,
        "command": [str(EXE), "-i", "<case-name>"],
        "omp_threads": 1, "timeout_seconds_per_case": TIMEOUT_SECONDS,
        "limitations": ["rigid four-point footprint, not a native weak line",
                        "B31 square-section torsion not matched to native generalized J",
                        "fitted rotations apply only to this symmetric linear MEANROTBS section"],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    records = []
    for sign, name in ((-1, "mx-minus"), (1, "mx-plus")):
        case = root / name
        case.mkdir()
        deck = build_loaded_tip_torque_x(sign)
        (case / f"{name}.inp").write_text(deck, encoding="ascii")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        start = time.perf_counter()
        failure = None
        try:
            process = subprocess.run((str(EXE), "-i", name), cwd=case,
                                     env=environment, capture_output=True,
                                     timeout=TIMEOUT_SECONDS, check=False)
            code, stdout, stderr = (process.returncode, process.stdout,
                                    process.stderr)
        except subprocess.TimeoutExpired as exc:
            code, stdout, stderr = None, exc.stdout or b"", exc.stderr or b""
            failure = "TIMEOUT_120_SECONDS"
        (case / "stdout.log").write_bytes(stdout)
        (case / "stderr.log").write_bytes(stderr)
        parsed = None
        if failure is None and code != 0:
            failure = f"CALCULIX_EXIT_{code}"
        if failure is None:
            try:
                out = stdout.decode(errors="replace")
                sta = (case / f"{name}.sta").read_text(errors="replace")
                dat = (case / f"{name}.dat").read_text(errors="replace")
                frd = (case / f"{name}.frd").read_text(errors="replace")
                trace = (case / f"{name}.12d").read_text(errors="replace")
                if (re.findall(r"a KNOT was generated in node\s+(\d+)",
                               trace) != ["47"] or
                        re.search(r"(?i)\b(warning|error|fatal)\b",
                                  out + sta + dat + frd + trace +
                                  stderr.decode(errors="replace"))):
                    raise ValueError("unexpected knot or solver diagnostic")
                times = [line for line in frd.splitlines()
                         if line.startswith("  100CL")]
                if len(times) != 1 or abs(float(times[0][12:24])-1.) > 1e-8:
                    raise ValueError("one full-step FRD result required")
                response = parse_completed_response(
                    out, sta, dat, require_nonzero_tip=False)
                motion = recover_tip_section_motion(frd)
                if abs(motion["rotation_rad"][0]) < 1e-10:
                    raise ValueError("unresolved tip twist")
                parsed = {"response": response, "tip_motion": motion,
                          "applied_mx_nm": 100*sign,
                          "tip_rotation_work_j": 100*sign*motion["rotation_rad"][0]}
            except (OSError, ValueError) as exc:
                failure = f"PARSE_{type(exc).__name__}: {exc}"
        record = {
            "name": name, "sign": sign,
            "status": "EXPLORATORY_COMPLETE" if failure is None else "EXPLORATORY_FAILURE",
            "failure": failure, "exit_code": code,
            "elapsed_seconds": time.perf_counter() - start,
            "files_sha256": {p.name: digest(p) for p in sorted(case.iterdir()) if p.is_file()},
            "parsed": parsed,
        }
        (case / "result.json").write_text(
            json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        records.append(record)
        if failure:
            break
    relative_oddness = None
    if len(records) == 2 and all(row["failure"] is None for row in records):
        minus, plus = (row["parsed"]["tip_motion"]["rotation_rad"][0]
                       for row in records)
        relative_oddness = abs(plus+minus)/max(abs(plus),abs(minus))
    summary = {"cases": records, "tip_mx_rotation_oddness_relative": relative_oddness,
               "status": "EXPLORATORY_NOT_G0_ACCEPTANCE"}
    (root / "result.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"cases": [{"case": row["name"], "status": row["status"],
                       "failure": row["failure"], "tip_motion":
                       row["parsed"]["tip_motion"] if row["parsed"] else None}
                      for row in records], "oddness":relative_oddness}, sort_keys=True))
    return 0 if len(records) == 2 and all(row["failure"] is None for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
