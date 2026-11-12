"""Run one registered clipped-overlap Q4/S3 coupled contact gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

import anymaterial
import anyfesolver
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]


def _git(*arguments: str, cwd: Path = ROOT) -> str:
    command = ["git"]
    if cwd != ROOT:
        command.extend(("-c", f"safe.directory={cwd.as_posix()}"))
    command.extend(("-C", str(cwd), *arguments))
    return subprocess.check_output(command, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    output_path = arguments.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"formal output already exists: {output_path}")
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if contract["case_id"] != "general_contact_d1_clipped_coupled_v2":
        raise ValueError("wrong case identity")
    registered_contract = (ROOT / "docs" / "reference_cases" /
                           "general_contact_d1_clipped_coupled_v2.json").resolve()
    if contract_path != registered_contract:
        raise ValueError("wrong registered contract path")
    relative_contract = contract_path.relative_to(ROOT).as_posix()
    _git("ls-files", "--error-unmatch", "--", relative_contract)
    if _git("status", "--porcelain", "--", relative_contract):
        raise ValueError("registered contract has uncommitted changes")
    if output_path != (ROOT / contract["runner"]["output"]).resolve():
        raise ValueError("output does not match registered path")
    if output_path.with_suffix(output_path.suffix + ".partial").exists():
        raise FileExistsError("formal partial output already exists")

    source_commit = contract["source_commit"]
    identity_checks = {}
    for path in contract["source_paths"]:
        identity_checks[path] = (
            _git("rev-parse", f"HEAD:{path}")
            == _git("rev-parse", f"{source_commit}:{path}")
            and not _git("status", "--porcelain", "--", path)
        )
    material_root = Path(contract["environment"]["anymaterial_source_root"]).parent
    material_identity = _git("rev-parse", "HEAD:src/anymaterial",
                             cwd=material_root)
    identity_checks["anymaterial_tree"] = (
        material_identity == contract["environment"]["anymaterial_tree"]
        and not _git("status", "--porcelain", "--", "src/anymaterial",
                     cwd=material_root)
    )
    identity_checks["python"] = (
        sys.version.split()[0] == contract["environment"]["python"]
    )
    identity_checks["numpy"] = (
        np.__version__ == contract["environment"]["numpy"]
    )
    identity_checks["scipy"] = (
        scipy.__version__ == contract["environment"]["scipy"]
    )
    identity_checks["anyfesolver_module"] = (
        Path(anyfesolver.__file__).resolve()
        == (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    )
    identity_checks["anymaterial_module"] = (
        Path(anymaterial.__file__).resolve()
        == (material_root / "src" / "anymaterial" / "__init__.py").resolve()
    )
    if not all(identity_checks.values()):
        raise ValueError(f"formal source/environment identity failed: {identity_checks}")

    start = time.perf_counter()
    run = runpy.run_path(str(ROOT / "tests" /
                              "test_general_contact_linear_patch_oracle.py"))
    metrics = run["_loaded_augmented_patch_oracle"](target_left_x=0.2)
    metrics.pop("contact_augmentation_attempts")
    elapsed = time.perf_counter() - start
    criteria = contract["criteria"]
    reference = contract["independent_reference"]
    expected_pressure = np.asarray(metrics["expected_pressure_pa"])
    actual_pressure = np.asarray(metrics["actual_pressure_pa"])
    expected_support = np.asarray(metrics["independent_source_support_m2"])
    actual_support = np.asarray(metrics["actual_source_support_m2"])
    expected_moment = np.asarray(metrics["expected_source_moment_nm"])
    actual_moment = np.asarray(metrics["actual_source_moment_nm"])
    checks = {
        "reference_identity": (
            np.allclose(metrics["expected_source_uz_m"],
                        reference["source_uz_m"], rtol=0.0, atol=1.0e-15)
            and np.allclose(expected_pressure, reference["pressure_pa"],
                            rtol=0.0, atol=1.0e-15)
            and np.allclose(expected_support,
                            reference["source_support_m2"],
                            rtol=0.0, atol=1.0e-14)
            and abs(metrics["expected_integrated_normal_force_n"]
                    - reference["integrated_normal_force_n"]) <= 1.0e-15
            and abs(metrics["expected_minimum_gap_m"]
                    - reference["minimum_physical_gap_m"]) <= 1.0e-15
            and np.allclose(expected_moment,
                            reference["source_moment_about_origin_nm"],
                            rtol=0.0, atol=1.0e-14)
        ),
        "status": metrics["status"] == "completed",
        "accepted_steps": metrics["accepted_steps"]
        == criteria["required_accepted_steps"],
        "displacement": metrics["maximum_absolute_uz_error_m"]
        <= criteria["maximum_absolute_uz_error_m"],
        "integrated_force": metrics["relative_integrated_force_error"]
        <= criteria["relative_integrated_force_error"],
        "pressure": float(np.max(np.abs(actual_pressure - expected_pressure)))
        <= criteria["maximum_absolute_pressure_error_pa"],
        "support": float(np.max(np.abs(actual_support - expected_support)))
        <= criteria["maximum_absolute_support_error_m2"],
        "pressure_node_ids": metrics["source_pressure_node_ids"]
        == reference["source_pressure_node_ids"],
        "active_node_ids": (np.flatnonzero(actual_pressure > 0.0) + 1).tolist()
        == reference["active_pressure_node_ids"],
        "pressure_rank": (
            metrics["pressure_gram_supported_modes"]
            == criteria["required_supported_pressure_modes"]
            and metrics["pressure_gram_scaled_rank"]
            == criteria["required_scaled_pressure_rank"]
        ),
        "gap": abs(metrics["actual_minimum_gap_m"]
                   - metrics["expected_minimum_gap_m"])
        <= criteria["minimum_gap_absolute_error_m"],
        "gap_tolerance": metrics["gap_tolerance_met"],
        "active_modes": metrics["active_pressure_dofs"]
        == criteria["required_active_pressure_dofs"],
        "equilibrium": metrics["equilibrium_residual_norm_n"]
        <= criteria["normalized_equilibrium_max"]
        * contract["scales"]["force_n"],
        "multiplier_change": metrics["normal_multiplier_change_pa"]
        <= criteria["normal_multiplier_change_pa"],
        "force_balance": metrics["force_wrench_balance_n"]
        <= criteria["normalized_force_balance_max"]
        * contract["scales"]["force_n"],
        "moment_balance": metrics["moment_wrench_balance_nm"]
        <= criteria["normalized_moment_balance_max"]
        * contract["scales"]["moment_nm"],
        "independent_free_residual": metrics[
            "independent_linear_free_residual_norm_n"
        ] <= criteria["independent_linear_free_residual_fraction_max"]
        * contract["scales"]["force_n"],
        "source_moment": float(np.linalg.norm(actual_moment-expected_moment))
        <= criteria["source_moment_fraction_max"]
        * contract["scales"]["moment_nm"],
        "checkpoint": metrics["has_restart_checkpoint"]
        == criteria["required_checkpoint"],
        "wall_budget": elapsed <= contract["runner"]["maximum_wall_seconds"],
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    record = {
        "case_id": contract["case_id"],
        "status": "passed_numerical_gate" if all(checks.values())
        else "failed_numerical_gate",
        "checks": checks,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_commit": source_commit,
        "execution_head": _git("rev-parse", "HEAD"),
        "identity_checks": identity_checks,
        "runtime_identity": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
            "anymaterial_module": str(Path(anymaterial.__file__).resolve()),
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({key: record[key] for key in (
        "case_id", "status", "checks", "elapsed_seconds",
    )}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
