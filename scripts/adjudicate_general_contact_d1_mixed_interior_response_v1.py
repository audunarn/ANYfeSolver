"""Adjudicate the registered 32-Q4 interior press response subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
import anyfesolver
import anymaterial


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = (ROOT / "docs/reference_cases" /
                "general_contact_d1_mixed_interior_response_v1.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(verdict_path: Path) -> None:
    if verdict_path.exists():
        raise FileExistsError(verdict_path)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    criteria = registration["criteria"]
    runner = registration["runner"]
    evidence_path = ROOT / runner["output"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    predicates: dict[str, bool] = {}

    def check(name: str, condition) -> None:
        predicates[name] = bool(condition)

    check("registration_pending", registration["status"]
          == "registered_pending_execution")
    check("source_commit_exists", subprocess.check_output(
        ["git", "cat-file", "-t", registration["source_commit"]],
        cwd=ROOT, text=True,
    ).strip() == "commit")
    check("source_tree", subprocess.check_output(
        ["git", "rev-parse", registration["source_commit"]+"^{tree}"],
        cwd=ROOT, text=True,
    ).strip() == registration["source_tree"])
    check("source_ancestor", subprocess.run(
        ["git", "merge-base", "--is-ancestor", registration["source_commit"],
         "HEAD"], cwd=ROOT, check=False,
    ).returncode == 0)
    for path, expected in registration["source_blobs"].items():
        actual = subprocess.check_output(
            ["git", "hash-object", path], cwd=ROOT, text=True,
        ).strip()
        check("source_blob:"+path, actual == expected)
    environment = registration["environment"]
    check("python_version", sys.version.split()[0] == environment["python"])
    check("numpy_version", np.__version__ == environment["numpy"])
    check("scipy_version", scipy.__version__ == environment["scipy"])
    check("solver_import", Path(anyfesolver.__file__).resolve()
          == Path(environment["anyfesolver_source_root"]) / "anyfesolver/__init__.py")
    check("material_import", Path(anymaterial.__file__).resolve()
          == Path(environment["anymaterial_source_root"]) / "anymaterial/__init__.py")

    prior_forces = {}
    for item in registration["prior_development_inputs"]:
        path = ROOT / item["path"]
        prior = json.loads(path.read_text(encoding="utf-8"))
        columns = item["columns"]
        check(f"prior_hash_{columns}", digest(path) == item["sha256"])
        check(f"prior_complete_{columns}", prior["status"] == "completed"
              and prior["accepted_steps"] == 12
              and prior["columns"] == columns
              and prior["target_right_x_m"] == 1.3)
        force = float(prior["history"][2]["contact"]["integration_force"][2])
        check(f"prior_force_{columns}", force == item["loaded_force_n"])
        prior_forces[columns] = force

    check("formal_status", evidence["status"] == criteria["status"])
    check("accepted_steps", evidence["accepted_steps"]
          == criteria["accepted_steps"] and len(evidence["history"]) == 2)
    check("runner_inputs", evidence["columns"] == 32
          and evidence["candidate"]
          == "mixed_bending_refinement_jet2_schur_press_dev"
          and evidence["profile"]
          == "Q4_S3_PATCH_REFINED_PROJECTED_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V0"
          and evidence["target_right_x_m"] == 1.3
          and evidence["four_phase"] is False
          and evidence["steps_per_segment"] == 2
          and evidence["load_profile"] == "tip"
          and evidence["total_downward_force_n"] == 0.1
          and evidence["normal_regularization_pa_per_m"] == 10000
          and evidence["schur_maximum_pressure_pa"] == 250
          and evidence["schur_maximum_step_pa"] == 200
          and evidence["schur_maximum_relinearizations"] == 0
          and evidence["schur_maximum_attempts"] == 2
          and evidence["maximum_augmentations"] == 7
          and evidence["maximum_candidate_pairs"] == 64
          and evidence["pressure_tolerance_pa"] == 1e-5
          and evidence["gap_tolerance_m"] == 5e-5
          and evidence["supported_pressure_mode_cap"] == 66
          and evidence["has_checkpoint"] is True)
    check("runtime_solver_import", Path(
        evidence["runtime_identity"]["anyfesolver_module"]
    ).resolve() == Path(anyfesolver.__file__).resolve())

    if len(evidence["history"]) < 2:
        verdict = {
            "case_id": registration["case_id"],
            "status": "fail",
            "claim_boundary": registration["claim_boundary"],
            "registration_sha256": digest(REGISTRATION),
            "evidence_sha256": digest(evidence_path),
            "source_commit": registration["source_commit"],
            "failure_reason": evidence.get("failure_reason"),
            "accepted_steps": evidence.get("accepted_steps"),
            "predicates": predicates,
            "failed_predicates": [key for key, passed in predicates.items()
                                  if not passed],
            "independent_review": "pending",
        }
        verdict_path.write_text(json.dumps(verdict, indent=2)+"\n",
                                encoding="utf-8")
        print(json.dumps({key: verdict[key] for key in (
            "status", "failure_reason", "accepted_steps", "failed_predicates",
        )}, indent=2))
        return

    loaded = evidence["history"][1]
    contact = loaded["contact"]
    force = float(contact["integration_force"][2])
    applied_load = np.asarray(loaded["nodal_forces_n"], dtype=float)
    check("accepted_full_load", applied_load.shape == (2,)
          and np.allclose(applied_load, [-0.05, -0.05], rtol=0.0,
                          atol=1e-14))
    check("finite_contact_wrench", all(np.all(np.isfinite(
        np.asarray(contact[name], dtype=float))) for name in (
            "integration_force", "target_force", "integration_moment",
            "target_moment")))
    prediction = registration["frozen_prediction"]["force_32_n"]
    prior24 = prior_forces[24]
    fit_x = np.array([16.0, 24.0, 32.0])
    fit_y = np.array([prior_forces[16], prior24, force])
    fit = np.linalg.lstsq(
        np.column_stack((np.ones(3), 1.0/fit_x)), fit_y, rcond=None,
    )[0]
    limit = float(fit[0])
    prediction_error = abs(force-prediction)/abs(prediction)
    change_24_32 = abs(force-prior24)/abs(force)
    limit_error_24 = abs(limit-prior24)/abs(limit)
    check("positive_force", force > 0.0)
    check("prediction_error", prediction_error
          <= criteria["loaded_force_relative_prediction_error_max"])
    check("change_24_to_32", change_24_32
          <= criteria["loaded_force_relative_change_24_to_32_max"])
    check("extrapolated_limit_24", limit > 0.0 and limit_error_24
          <= criteria["force_24_relative_extrapolated_limit_error_max"])

    check("full_scaled_pressure_rank",
          contact["pressure_gram_scaled_rank"]
          == contact["pressure_gram_supported_modes"] == 66)
    check("active_pressure", contact["active_pressure_dofs"] > 0)
    force_scale = criteria["force_scale_n"]
    moment_scale = criteria["moment_scale_nm"]
    force_balance = float(np.linalg.norm(
        np.asarray(contact["integration_force"])
        + np.asarray(contact["target_force"])
    ))/force_scale
    moment_balance = float(np.linalg.norm(
        np.asarray(contact["integration_moment"])
        + np.asarray(contact["target_moment"])
    ))/moment_scale
    check("contact_wrench", max(force_balance, moment_balance)
          <= criteria["normalized_force_and_moment_wrench_error_max"])
    support_balance = abs(evidence["support_vertical_reactions_n"][1]-0.1)
    check("support_balance", support_balance/force_scale
          <= criteria["normalized_force_and_moment_wrench_error_max"])
    check("equilibrium", loaded["residual_norm"]/force_scale
          <= criteria["normalized_equilibrium_error_max"])
    check("pressure_change", loaded["normal_multiplier_change"]
          <= criteria["pressure_change_pa_max"])
    check("physical_gap", contact["gap_tolerance_met"]
          and contact["minimum_gap"]
          >= criteria["independent_minimum_gap_m_min"])
    pressure_max = max(contact["normal_multiplier_estimate_pa"])
    integration_indicator = (
        pressure_max*contact["area_integration_error_m2"]
        + registration["model"]["normal_regularization_pa_per_m"]
        * contact["weighted_gap_integration_error_m3"]
    )
    integration_fraction = integration_indicator/(0.02*force)
    check("integration_indicator", integration_fraction
          <= criteria["contact_integration_error_fraction_of_2pct_response_budget_max"])

    verdict = {
        "case_id": registration["case_id"],
        "status": "pass" if all(predicates.values()) else "fail",
        "claim_boundary": registration["claim_boundary"],
        "registration_sha256": digest(REGISTRATION),
        "evidence_sha256": digest(evidence_path),
        "source_commit": registration["source_commit"],
        "formal_force_n": force,
        "frozen_prediction_n": prediction,
        "prediction_relative_error": prediction_error,
        "force_24_to_32_relative_change": change_24_32,
        "extrapolated_force_limit_n": limit,
        "force_24_relative_limit_error": limit_error_24,
        "normalized_force_balance": force_balance,
        "normalized_moment_balance": moment_balance,
        "normalized_support_balance": support_balance/force_scale,
        "integration_indicator_fraction_of_response_budget": integration_fraction,
        "predicates": predicates,
        "failed_predicates": [key for key, passed in predicates.items()
                              if not passed],
        "independent_review": "pending",
    }
    verdict_path.write_text(json.dumps(verdict, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: verdict[key] for key in (
        "status", "formal_force_n", "prediction_relative_error",
        "force_24_to_32_relative_change", "force_24_relative_limit_error",
        "failed_predicates", "independent_review",
    )}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python adjudicate.py VERDICT_JSON")
    main(Path(sys.argv[1]))
