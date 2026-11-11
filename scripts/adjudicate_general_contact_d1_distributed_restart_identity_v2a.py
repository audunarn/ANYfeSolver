"""Correct only the checkpoint payload schema in the frozen v2 adjudication."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True,
    ).strip()


def _bounded(actual: object, limit: float) -> bool:
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and 0 <= actual <= limit)


def main(registration_path: Path, report_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    root = registration_path.resolve().parents[2]
    reference_path = root / registration["reference_report"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    criteria = dict(registration["criteria"])
    if criteria["checkpoint_schema"] != "ANYSOLVER_STATIC_RESTART_CONTRACT_V2":
        raise ValueError("unexpected frozen v2 checkpoint-schema field")
    criteria["checkpoint_schema"] = "ANYSOLVER_NONLINEAR_CHECKPOINT_V2"
    runtime = report.get("runtime_identity", {})
    environment = registration["environment"]
    checks: dict[str, bool] = {
        "reference_hash": _sha(reference_path) == registration["reference_sha256"],
        "report_reference_hash": (
            report.get("reference_sha256") == registration["reference_sha256"]),
        "source_tree": (_git(root, "show", "-s", "--format=%T",
                             registration["source_commit"])
                        == registration["source_tree"]),
        "source_blobs": all(
            _git(root, "hash-object", path) == blob
            for path, blob in registration["source_blobs"].items()
        ),
        "reference_identity": (
            Path(report.get("reference", ""))
            == Path(registration["reference_report"])
            and report.get("columns") == 24
            and report.get("target_right_x_m") == 1.3),
        "runtime_versions": all(
            runtime.get(key) == environment[key]
            for key in ("python", "numpy", "scipy")
        ),
        "runtime_source_roots": (
            Path(runtime.get("anyfesolver_module", "")).resolve().parents[1]
            == Path(environment["anyfesolver_source_root"]).resolve()
            and Path(runtime.get("anymaterial_module", "")).resolve().parents[1]
            == Path(environment["anymaterial_source_root"]).resolve()),
        "wall_time": _bounded(
            report.get("elapsed_seconds"),
            registration["runner"]["maximum_wall_seconds"]),
        "checkpoint_contract": (
            report.get("prefix_checkpoint_schema")
            == criteria["checkpoint_schema"]
            and report.get("prefix_checkpoint_version")
            == criteria["checkpoint_version"]),
    }
    differences = report.get("differences", ())
    checks["step_records"] = (
        len(differences) == len(reference["history"]) == 12
        and [row.get("step") for row in differences] == list(range(1, 13))
    )
    identity_checks = []
    for row, saved in zip(differences, reference["history"]):
        contact = saved["contact"]
        source_ids = row["reference_pressure_node_ids"]
        resumed_ids = row["resumed_pressure_node_ids"]
        source_pressures = row["reference_pressure_vector_pa"]
        resumed_pressures = row["resumed_pressure_vector_pa"]
        source_active = [node for node, pressure in zip(source_ids, source_pressures)
                         if pressure > 0]
        resumed_active = [node for node, pressure in zip(resumed_ids, resumed_pressures)
                          if pressure > 0]
        local = (
            source_ids == contact["source_pressure_node_ids"]
            and source_pressures == contact["normal_multiplier_estimate_pa"]
            and len(source_ids) == len(source_pressures)
            and len(resumed_ids) == len(resumed_pressures)
            and len(set(source_ids)) == len(source_ids)
            and len(set(resumed_ids)) == len(resumed_ids)
            and all(math.isfinite(value) for value in
                    (*source_pressures, *resumed_pressures))
            and source_ids == resumed_ids
            and source_active == resumed_active
            and source_active == row["reference_active_pressure_node_ids"]
            and resumed_active == row["resumed_active_pressure_node_ids"]
            and len(source_active) == contact["active_pressure_dofs"]
            and row["active_counts_match"]
            and row["pressure_node_ids_match"]
            and row["active_pressure_node_ids_match"]
            and row["active_membership_consistent"]
            and _bounded(
                row["pressure_vector_difference_pa"],
                criteria["maximum_pressure_vector_difference_pa"])
            and max(abs(a-b) for a, b in
                    zip(source_pressures, resumed_pressures))
            == row["pressure_vector_difference_pa"]
        )
        identity_checks.append(local)
    checks["independently_recomputed_pressure_identity"] = (
        len(identity_checks) == 12 and all(identity_checks)
    )
    for key, limit in criteria.items():
        if key in ("checkpoint_schema", "checkpoint_version",
                   "force_wrench_imbalance_n", "moment_wrench_imbalance_nm"):
            continue
        actual = report.get(key)
        if isinstance(limit, bool):
            checks[key] = actual is limit
        elif key.endswith("_steps"):
            checks[key] = actual == limit
        else:
            checks[key] = _bounded(actual, limit)
    checks["all_step_wrenches"] = all(
        _bounded(row.get("force_wrench_imbalance_n"),
                 criteria["force_wrench_imbalance_n"])
        and _bounded(row.get("moment_wrench_imbalance_nm"),
                     criteria["moment_wrench_imbalance_nm"])
        for row in differences
    )
    verdict = {
        "case_id": registration["case_id"] + "_schema_correction_v2a",
        "status": "pass" if all(checks.values()) else "fail",
        "registration_sha256": _sha(registration_path),
        "reference_sha256": _sha(reference_path),
        "report_sha256": _sha(report_path),
        "checks": checks,
        "corrected_payload_schema": criteria["checkpoint_schema"],
        "independent_review": "pending",
    }
    output_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": verdict["status"],
                      "failed": [key for key, passed in checks.items()
                                 if not passed]}))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: adjudicator REGISTRATION REPORT OUTPUT")
    main(*(Path(argument) for argument in sys.argv[1:]))
