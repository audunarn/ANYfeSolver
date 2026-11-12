"""Independent small-motion check of ordinary displacement with a free S3 target."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_free_target_ordinary_displacement_matches_independent_q1_kkt():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(root / "src"), str(root), str(root / "packages" / "results" / "src"),
    ))
    completed = subprocess.run(
        [sys.executable, str(root / "tests" / "fixtures" / "d1_two_body_displacement_compare.py")],
        cwd=root, env=environment, capture_output=True, text=True,
        check=True, timeout=120,
    )
    result = json.loads(completed.stdout)
    assert Path(result["anyfesolver_module"]).resolve() == (root / "src/anyfesolver/__init__.py").resolve()
    for key, path in (("oracle_source_sha256", "tests/fixtures/d1_two_body_displacement_oracle.py"),
                      ("reference_oracle_source_sha256", "scripts/d1_reference_gap_work.py")):
        assert result[key] == hashlib.sha256((root / path).read_bytes()).hexdigest()
    assert result["production_status"] == "completed"
    assert result["accepted_steps"] == 4
    assert result["has_checkpoint"]
    assert result["oracle_active_modes"] == result["production_active_modes"] == 1
    assert result["oracle_integrated_force_n"] > 0
    assert result["production_integrated_force_n"] > 0
    assert result["gap_tolerance_met"]
    assert result["minimum_independent_gap_m"] >= -50e-6
    assert abs(result["source_target_vertical_force_balance_n"]) <= 1e-9
    assert abs(result["control_uz_m"]+80e-6) <= 1e-12
    for name in (
        "integrated_force_relative_error",
        "pressure_relative_l2_error",
        "support_relative_l2_error",
        "control_reaction_relative_error",
        "endpoint_pairing_relative_error",
    ):
        assert result[name] <= 1e-6, (name, result[name])
    assert result["maximum_gap_difference_at_production_pose_m"] <= 1e-9
    assert result["target_translation_max_difference_m"] <= 1e-9
    assert result["target_rotation_max_difference_rad"] <= 1e-9
    assert result["source_inplane_max_m"] <= 1e-8
