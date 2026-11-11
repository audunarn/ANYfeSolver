"""S4 validity hardening checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from anyfesolver.s4_validity import (
    bending_patch_metric,
    free_element_mode_metric,
    generate_s4_validity_report,
    membrane_patch_metric,
    reference_s4_geometries,
    s4_s8_comparison,
    shear_patch_metric,
    thin_plate_locking_sweep,
)


def test_s4_free_element_has_six_rigid_body_modes_for_reference_geometries() -> None:
    for name, coords in reference_s4_geometries().items():
        metric = free_element_mode_metric(coords)
        assert metric["zero_mode_count"] == 6, name
        assert abs(metric["relative_negative_eigenvalue"]) < 1.0e-12, name
        assert metric["max_eigenvalue"] > 0.0


def test_s4_membrane_bending_and_shear_patch_exact_for_affine_quads() -> None:
    for name in ("square", "parallelogram"):
        coords = reference_s4_geometries()[name]
        membrane = membrane_patch_metric(coords)
        assert max(membrane["relative_errors"].values()) < 1.0e-10, name
        assert max(abs(value) for value in membrane["relative_spreads"].values()) < 1.0e-10, name

        bending = bending_patch_metric(coords)
        assert bending["relative_error"] < 1.0e-10, name
        assert bending["relative_spread"] < 1.0e-10, name

        shear = shear_patch_metric(coords)
        assert shear["relative_error"] < 1.0e-10, name
        assert shear["relative_spread"] < 1.0e-10, name


def test_s4_skew_and_mild_warp_distortion_metrics_are_bounded_and_finite() -> None:
    skew = reference_s4_geometries()["skew"]
    # Membrane and bending patch metrics compare global-frame stresses, so the
    # constant-strain patch state is reproduced exactly even on skew geometry.
    skew_membrane = membrane_patch_metric(skew)
    assert max(skew_membrane["relative_errors"].values()) < 1.0e-10
    assert bending_patch_metric(skew)["relative_error"] < 1.0e-10
    # MITC4 assumed transverse shear keeps a small interpolation residual on
    # distorted geometry; this is expected element behavior, not a frame issue.
    assert shear_patch_metric(skew)["relative_error"] < 0.01

    report = generate_s4_validity_report()
    warped = report["geometry_metrics"]["mild_warp"]["warped_quad"]
    assert warped["stiffness_finite"] is True
    assert warped["relative_symmetry_error"] < 1.0e-12


def test_s4_thin_plate_locking_sweep_tracks_beam_reference() -> None:
    rows = thin_plate_locking_sweep()
    assert [row["solver_status"] for row in rows] == ["converged", "converged", "converged"]
    assert min(row["span_to_thickness"] for row in rows) >= 100.0
    for row in rows:
        assert row["relative_error"] < 0.05
        assert 0.9 < row["ratio_to_reference"] < 1.05


def test_s4_s8_displacement_comparison_improves_with_refinement() -> None:
    rows = s4_s8_comparison()
    assert len(rows) >= 2
    assert all(row["s4_status"] == "converged" and row["s8_status"] == "converged" for row in rows)
    coarse, fine = rows[0], rows[-1]
    assert abs(fine["displacement_ratio_s4_to_s8"] - 1.0) < abs(coarse["displacement_ratio_s4_to_s8"] - 1.0)
    assert 0.85 < fine["displacement_ratio_s4_to_s8"] < 1.10
    assert fine["stress_ratio_s4_to_s8"] > coarse["stress_ratio_s4_to_s8"]


def test_s4_validity_cli_writes_json_and_markdown() -> None:
    output_dir = Path(".pytest_tmp_s4_validity") / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "s4_validity.json"
        md_path = output_dir / "s4_validity.md"
        completed = subprocess.run(
            [sys.executable, "scripts/run_s4_validity.py", "--output", str(json_path), "--markdown", str(md_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["element"] == "S4"
        assert "thin_plate_locking_sweep" in report
        assert "s4_s8_comparison" in report
        assert md_path.exists()
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
