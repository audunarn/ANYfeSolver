"""Adjudicate the registered 24-Q4 distributed Q4/S3 full-cycle subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import anymaterial
import anyfesolver
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = (ROOT / "docs/reference_cases" /
                "general_contact_d1_distributed_cycle_response_v1.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(verdict_path: Path) -> None:
    if verdict_path.exists():
        raise FileExistsError(verdict_path)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    criteria = registration["criteria"]
    evidence_path = ROOT / registration["runner"]["output"]
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
        n = item["columns"]
        check(f"prior_hash_{n}", digest(path) == item["sha256"])
        check(f"prior_complete_{n}", prior["status"] == "completed"
              and prior["accepted_steps"] == 12
              and prior["columns"] == n
              and prior["target_right_x_m"] == 1.3
              and prior["load_profile"] == "distributed"
              and prior["total_downward_force_n"] == 0.3)
        loaded_force = float(prior["history"][2]["contact"]
                             ["integration_force"][2])
        unload_force = float(prior["history"][6]["contact"]
                             ["integration_force"][2])
        check(f"prior_loaded_force_{n}", loaded_force
              == item["loaded_force_n"])
        check(f"prior_unload_force_{n}", unload_force
              == item["first_unload_force_n"])
        prior_forces[n] = (loaded_force, unload_force)

    check("formal_status", evidence["status"] == criteria["status"])
    history = evidence.get("history") or []
    check("accepted_steps", evidence["accepted_steps"]
          == criteria["accepted_steps"] and len(history) == 12)
    check("runner_inputs", evidence["columns"] == 24
          and evidence["candidate"]
          == "mixed_bending_refinement_jet2_schur_four_phase_dev"
          and evidence["profile"]
          == "Q4_S3_PATCH_REFINED_PROJECTED_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V0"
          and evidence["target_right_x_m"] == 1.3
          and evidence["four_phase"] is True
          and evidence["steps_per_segment"] == 3
          and evidence["load_profile"] == "distributed"
          and evidence["total_downward_force_n"] == 0.3
          and evidence["normal_regularization_pa_per_m"] == 10000
          and evidence["schur_maximum_pressure_pa"] == 250
          and evidence["schur_maximum_step_pa"] == 200
          and evidence["schur_maximum_relinearizations"] == 0
          and evidence["schur_maximum_attempts"] == 2
          and evidence["maximum_augmentations"] == 7
          and evidence["maximum_candidate_pairs"] == 48
          and evidence["pressure_tolerance_pa"] == 1e-5
          and evidence["gap_tolerance_m"] == 5e-5
          and evidence["supported_pressure_mode_cap"] == 50
          and evidence["has_checkpoint"] is True)
    check("runtime_solver_import", Path(
        evidence["runtime_identity"]["anyfesolver_module"]
    ).resolve() == Path(anyfesolver.__file__).resolve())

    verdict = {
        "case_id": registration["case_id"],
        "claim_boundary": registration["claim_boundary"],
        "registration_sha256": digest(REGISTRATION),
        "evidence_sha256": digest(evidence_path),
        "source_commit": registration["source_commit"],
        "predicates": predicates,
        "independent_review": "pending",
    }
    if len(history) < 12:
        verdict.update(status="fail",
                       failure_reason=evidence.get("failure_reason"),
                       accepted_steps=evidence.get("accepted_steps"),
                       failed_predicates=[key for key, passed in predicates.items()
                                          if not passed])
        verdict_path.write_text(json.dumps(verdict, indent=2)+"\n",
                                encoding="utf-8")
        print(json.dumps({key: verdict[key] for key in (
            "status", "failure_reason", "accepted_steps", "failed_predicates",
        )}, indent=2))
        return

    force_scale = criteria["force_scale_n"]
    moment_scale = criteria["moment_scale_nm"]
    force_balances = []
    moment_balances = []
    support_balances = []
    integration_fractions = []
    for index, step in enumerate(history):
        contact = step["contact"]
        nodal_forces = np.asarray(step["nodal_forces_n"], dtype=float)
        factor = -criteria["expected_load_totals_n"][index]/0.3
        weights = np.asarray([
            (0.5 if column in (0, 24) else 1.0)/(2*24)
            for _row in range(2) for column in range(25)
        ])
        expected_forces = -0.3*factor*weights
        check(f"accepted_nodal_load_{index+1}",
              nodal_forces.shape == (50,)
              and np.allclose(nodal_forces, expected_forces,
                              rtol=0.0, atol=1e-14))
        check(f"pressure_rank_{index+1}",
              contact["pressure_gram_scaled_rank"]
              == contact["pressure_gram_supported_modes"] == 50)
        check(f"equilibrium_{index+1}",
              step["residual_norm"]/force_scale
              <= criteria["normalized_equilibrium_error_max"])
        check(f"pressure_change_{index+1}",
              step["normal_multiplier_change"]
              <= criteria["pressure_change_pa_max"])
        check(f"physical_gap_{index+1}",
              contact["gap_tolerance_met"]
              and contact["minimum_gap"]
              >= criteria["independent_minimum_gap_m_min"])
        force = np.asarray(contact["integration_force"], dtype=float)
        target_force = np.asarray(contact["target_force"], dtype=float)
        moment = np.asarray(contact["integration_moment"], dtype=float)
        target_moment = np.asarray(contact["target_moment"], dtype=float)
        check(f"finite_wrench_{index+1}", all(np.all(np.isfinite(value))
              for value in (force, target_force, moment, target_moment)))
        force_balances.append(float(np.linalg.norm(force+target_force))
                              / force_scale)
        moment_balances.append(float(np.linalg.norm(moment+target_moment))
                               / moment_scale)
        support_balances.append(abs(
            evidence["support_vertical_reactions_n"][index]
            + float(np.sum(nodal_forces))
        )/force_scale)
        if contact["active_pressure_dofs"] > 0:
            pressure_max = max(contact["normal_multiplier_estimate_pa"])
            indicator = (
                pressure_max*contact["area_integration_error_m2"]
                + registration["model"]["normal_regularization_pa_per_m"]
                * contact["weighted_gap_integration_error_m3"]
            )
            contact_force = abs(float(force[2]))
            integration_fractions.append(
                indicator/(0.02*contact_force) if contact_force > 0.0
                else float("inf")
            )

    loaded = [float(history[index]["contact"]["integration_force"][2])
              for index in (2, 5, 11)]
    first_unload = float(history[6]["contact"]["integration_force"][2])
    recontact = float(history[10]["contact"]["integration_force"][2])
    prior16_loaded, prior16_unload = prior_forces[16]
    prediction = registration["frozen_prediction"]
    loaded_prediction_error = abs(loaded[0]-prediction["loaded_force_24_n"])/abs(
        prediction["loaded_force_24_n"])
    unload_prediction_error = abs(
        first_unload-prediction["first_unload_force_24_n"]
    )/abs(prediction["first_unload_force_24_n"])
    loaded_change = (abs(loaded[0]-prior16_loaded)
                     / max(abs(loaded[0]), np.finfo(float).tiny))
    unload_change = (abs(first_unload-prior16_unload)
                     / max(abs(first_unload), np.finfo(float).tiny))
    loaded_spread = ((max(loaded)-min(loaded))
                     / max(abs(max(loaded)), np.finfo(float).tiny))
    check("loaded_prediction", loaded_prediction_error
          <= criteria["loaded_force_relative_prediction_error_max"])
    check("first_unload_prediction", unload_prediction_error
          <= criteria["first_unload_relative_prediction_error_max"])
    check("loaded_change_16_to_24", loaded_change
          <= criteria["loaded_force_relative_change_16_to_24_max"])
    check("first_unload_change_16_to_24", unload_change
          <= criteria["first_unload_relative_change_16_to_24_max"])
    check("loaded_press_slide_recontact", min(loaded) > 0.0
          and loaded_spread
          <= criteria["loaded_press_slide_recontact_relative_spread_max"]
          and all(history[index]["contact"]["active_pressure_dofs"]
                  >= criteria["loaded_active_pressure_modes_min"]
                  for index in (2, 5, 11)))
    check("first_unload_active", first_unload > 0.0
          and history[6]["contact"]["active_pressure_dofs"]
          >= criteria["first_unload_active_pressure_modes_min"])
    check("open_steps", all(
        history[index-1]["contact"]["active_pressure_dofs"] == 0
        and abs(float(history[index-1]["contact"]["integration_force"][2]))
        <= criteria["open_contact_force_abs_max_n"]
        for index in criteria["open_steps"]
    ))
    check("recontact", recontact > 0.0
          and history[criteria["recontact_step"]-1]["contact"]
          ["active_pressure_dofs"] > 0)
    check("contact_wrench", max(force_balances+moment_balances)
          <= criteria["normalized_force_and_moment_wrench_error_max"])
    check("support_balance", max(support_balances)
          <= criteria["normalized_force_and_moment_wrench_error_max"])
    worst_integration = max(integration_fractions, default=float("inf"))
    check("integration_indicator", np.isfinite(worst_integration)
          and worst_integration
          <= criteria["contact_integration_error_fraction_of_2pct_response_budget_max"])

    verdict.update({
        "status": "pass" if all(predicates.values()) else "fail",
        "loaded_force_n": loaded[0],
        "first_unload_force_n": first_unload,
        "loaded_prediction_relative_error": loaded_prediction_error,
        "first_unload_prediction_relative_error": unload_prediction_error,
        "loaded_force_change_16_to_24": loaded_change,
        "first_unload_force_change_16_to_24": unload_change,
        "loaded_force_spread": loaded_spread,
        "maximum_normalized_force_balance": max(force_balances),
        "maximum_normalized_moment_balance": max(moment_balances),
        "maximum_normalized_support_balance": max(support_balances),
        "maximum_integration_indicator_fraction": worst_integration,
        "failed_predicates": [key for key, passed in predicates.items()
                              if not passed],
    })
    verdict_path.write_text(json.dumps(verdict, indent=2)+"\n",
                            encoding="utf-8")
    print(json.dumps({key: verdict[key] for key in (
        "status", "loaded_force_n", "first_unload_force_n",
        "loaded_force_change_16_to_24", "first_unload_force_change_16_to_24",
        "failed_predicates", "independent_review",
    )}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python adjudicate.py VERDICT_JSON")
    main(Path(sys.argv[1]))
