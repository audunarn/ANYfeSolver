"""Adjudicate the registered installed 24-Q4 cycle without rerunning it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(registration_path: Path, result_path: Path, verdict_path: Path) -> None:
    if verdict_path.exists():
        raise FileExistsError(verdict_path)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reference_path = Path(registration["reference_report"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    criteria = registration["criteria"]
    wheel_path = Path(registration["wheel_path"])
    install_target = Path(registration["install_target"]).resolve()
    histories = reference["history"]
    steps = result["steps"]
    comparisons = []
    for index, (step, row) in enumerate(zip(steps, histories)):
        comparisons.append({
            "step_index": index+1,
            "normal_force_difference_n": abs(
                float(step["normal_force_n"])
                - float(row["contact"]["integration_force"][2])),
            "minimum_gap_difference_m": abs(
                float(step["minimum_gap_m"])
                - float(row["contact"]["minimum_gap"])),
            "pressure_change_difference_pa": abs(
                float(step["pressure_change_pa"])
                - float(row["normal_multiplier_change"])),
            "active_pressure_mode_match": (
                int(step["active_pressure_modes"])
                == int(row["contact"]["active_pressure_dofs"])),
        })
    maxima = {field: max((float(row[field]) for row in comparisons),
                         default=float("inf")) for field in (
        "normal_force_difference_n", "minimum_gap_difference_m",
        "pressure_change_difference_pa")}
    module_path = Path(result["anyfesolver_module"]).resolve()
    checks = {
        "wheel_sha256": _sha256(wheel_path)
        == registration["wheel_sha256"],
        "reference_sha256": _sha256(reference_path)
        == registration["reference_sha256"],
        "status": result["status"] == criteria["status"],
        "columns": result["columns"] == criteria["columns"],
        "step_count": len(steps) == len(histories) == criteria["steps"],
        "checkpoint": bool(result["has_restart_checkpoint"])
        == criteria["checkpoint_present"],
        "module_inside_install_target": module_path.is_relative_to(install_target),
        "force_history": maxima["normal_force_difference_n"]
        <= criteria["maximum_absolute_normal_force_difference_n"],
        "gap_history": maxima["minimum_gap_difference_m"]
        <= criteria["maximum_absolute_minimum_gap_difference_m"],
        "pressure_change_history": maxima["pressure_change_difference_pa"]
        <= criteria["maximum_absolute_pressure_change_difference_pa"],
        "active_counts": all(
            row["active_pressure_mode_match"] for row in comparisons),
        "independent_gaps": all(
            float(step["minimum_gap_m"]) >= -5e-5 for step in steps),
        "step_order": all(
            int(step["step"]) == int(row["step_index"]) == index+1
            for index, (step, row) in enumerate(zip(steps, histories))),
        "profile": result["capability_profile"] == reference["profile"],
    }
    verdict = {
        "registration": str(registration_path),
        "registration_sha256": _sha256(registration_path),
        "raw_result": str(result_path),
        "raw_result_sha256": _sha256(result_path),
        "installed_module": str(module_path),
        "checks": checks,
        "maxima": maxima,
        "step_comparisons": comparisons,
        "status": "pass" if all(checks.values()) else "fail",
    }
    verdict_path.write_text(json.dumps(verdict, indent=2)+"\n",
                            encoding="utf-8")
    print(json.dumps({key: verdict[key] for key in
                      ("status", "maxima", "checks")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: installed_24q4_adjudicate.py REG RAW VERDICT")
    main(*(Path(argument) for argument in sys.argv[1:]))
