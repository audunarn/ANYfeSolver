"""Check the registered fixed-pose pressure benchmark without contact kernels."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy

import anymaterial
import anyfesolver


ROOT = Path(__file__).resolve().parents[1]


def _git_blob(path: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _source_clean() -> bool:
    paths = ["src/anyfesolver", "scripts/general_contact_d1_finite_area_load.py",
             "scripts/general_contact_d1_pressure_space_probe.py"]
    return not subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _norm(values) -> float:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("contact wrench must contain three finite components")
    return float(np.linalg.norm(vector))


def adjudicate(contract: dict, result: dict) -> dict:
    checks: dict[str, bool] = {}
    checks["registered_case"] = (
        contract.get("case_id") == "general_contact_d1_affine_partial_pressure_v1"
        and contract.get("status") == "registered_pending_execution"
    )
    checks["source_blobs"] = all(
        _git_blob(path) == expected
        for path, expected in contract["source_blobs"].items()
    ) and _source_clean()
    registered_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", contract["source_commit"]],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    checks["registered_source_tree"] = registered_tree == contract["source_tree"]
    environment = contract["environment"]
    checks["environment"] = (
        sys.version.split()[0] == environment["python"]
        and np.__version__ == environment["numpy"]
        and scipy.__version__ == environment["scipy"]
        and Path(anyfesolver.__file__).resolve().parents[1]
            == Path(environment["anyfesolver_source_root"]).resolve()
        and Path(anymaterial.__file__).resolve().parents[1]
            == Path(environment["anymaterial_source_root"]).resolve()
    )
    model = contract["model"]
    runner = contract["runner"]
    reference = contract["independent_reference"]
    criteria = contract["criteria"]
    force_scale = float(reference["force_scale_n"])
    moment_scale = float(reference["moment_scale_nm"])
    force_reference = float(reference["force_z_n"])
    checks["input_profile"] = (
        result.get("profile") == "fixed_affine_partial_pressure_central_q4_s3_dev"
        and result.get("target_x_range_m") == [0.24, 0.77]
        and result.get("target_y_range_m") == [0.21, 0.73]
        and math.isclose(result.get("left_penetration_m", math.nan),
                         25.0e-6, rel_tol=0.0, abs_tol=1.0e-16)
        and math.isclose(result.get("normal_stiffness_pa_per_m", math.nan),
                         model["normal_stiffness_pa_per_m"],
                         rel_tol=0.0, abs_tol=1.0e-12)
    )
    rows = result.get("rows", [])
    checks["row_inventory"] = (
        len(rows) == 2
        and [row.get("divisions") for row in rows] == runner["x_divisions"]
        and all(row.get("rows") == runner["y_divisions"] for row in rows)
    )
    checks["wall_budget"] = (
        0.0 <= float(result.get("elapsed_seconds_total", math.inf))
        <= runner["maximum_wall_seconds"]
    )
    metrics = []
    if checks["row_inventory"]:
        for row in rows:
            force = float(row["reaction_z_n"])
            error = abs(force - force_reference) / force_reference
            source_force = np.asarray(row["source_force_n"], dtype=float)
            target_force = np.asarray(row["target_force_n"], dtype=float)
            source_moment = np.asarray(
                row["source_moment_about_origin_nm"], dtype=float,
            )
            target_moment = np.asarray(
                row["target_moment_about_origin_nm"], dtype=float,
            )
            force_balance = _norm(source_force + target_force) / force_scale
            moment_balance = _norm(source_moment + target_moment) / moment_scale
            integration_fraction = (
                float(row["integration_force_error_indicator_n"])
                / (criteria["x64_relative_force_error_max"] * force_scale)
            )
            metrics.append({
                "x_divisions": row["divisions"],
                "reaction_z_n": force,
                "relative_reference_error": error,
                "normalized_force_balance": force_balance,
                "normalized_moment_balance": moment_balance,
                "integration_fraction_of_response_budget": integration_fraction,
                "minimum_gap_m": row["minimum_gap_m"],
                "supported_modes": row["supported_pressure_modes"],
                "scaled_rank": row["scaled_pressure_rank"],
            })
            prefix = f"x{row['divisions']}"
            checks[prefix + "_reference_identity"] = math.isclose(
                row["continuous_local_plane_reference_z_n"],
                force_reference, rel_tol=0.0, abs_tol=1.0e-12,
            ) and math.isclose(
                row["relative_reference_error"], error,
                rel_tol=0.0, abs_tol=1.0e-12,
            )
            checks[prefix + "_wrench"] = (
                force_balance <= criteria["normalized_force_balance_max"]
                and moment_balance <= criteria["normalized_moment_balance_max"]
            )
            checks[prefix + "_gap"] = (
                row["gap_tolerance_met"] is criteria["gap_tolerance_met"]
                and abs(row["minimum_gap_m"] - reference["minimum_gap_m"])
                    <= criteria["minimum_gap_absolute_error_m_max"]
                and row["sampled_overlap_gap_m"] is not None
                and abs(row["sampled_overlap_gap_m"]
                        - reference["minimum_gap_m"])
                    <= criteria["sampled_overlap_gap_absolute_error_m_max"]
            )
            checks[prefix + "_rank_and_activity"] = (
                row["scaled_pressure_rank"] == row["supported_pressure_modes"]
                and 0 < row["active_pressure_dofs"]
                < row["supported_pressure_modes"]
            )
            checks[prefix + "_integration_indicator"] = (
                0.0 <= integration_fraction
                <= criteria["integration_error_indicator_fraction_of_response_budget_max"]
            )
        checks["x64_analytic_force"] = (
            metrics[1]["relative_reference_error"]
            <= criteria["x64_relative_force_error_max"]
        )
        checks["refinement_trend"] = (
            metrics[1]["relative_reference_error"]
            < metrics[0]["relative_reference_error"]
        )
    return {
        "case_id": contract["case_id"],
        "status": "passed_numerical_gate" if checks and all(checks.values())
                  else "failed_numerical_gate",
        "checks": checks,
        "metrics": metrics,
        "source_commit": contract["source_commit"],
        "claim_boundary": contract["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("verdict", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    verdict = adjudicate(contract, result)
    verdict["contract_sha256"] = hashlib.sha256(
        args.contract.read_bytes()).hexdigest()
    verdict["result_sha256"] = hashlib.sha256(
        args.result.read_bytes()).hexdigest()
    args.verdict.write_text(json.dumps(verdict, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    if verdict["status"] != "passed_numerical_gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
