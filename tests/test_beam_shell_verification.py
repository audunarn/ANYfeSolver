"""Manifest-driven beam/shell verification report tests."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from anyfesolver import run_beam_shell_verification, verification_manifest_cases, write_beam_shell_verification_report
import anyfesolver.beam_shell_verification as beam_shell_verification


def test_manifest_contains_stable_case_ids_from_spec() -> None:
    cases = verification_manifest_cases()
    ids = {case.case_id for case in cases}

    assert len(cases) == 130
    assert "META-001" in ids
    assert "ALG-001" in ids
    assert "BEAM-010" in ids
    assert "BEAM-011" in ids
    assert "SHELL-008" in ids
    assert "SHELL-011" in ids
    assert "SHELL-007" in ids
    assert "COUP-005" in ids
    assert "COUP-011" in ids
    assert "COUP-014" in ids
    assert "COUP-016" in ids
    assert "MAT-007" in ids
    assert "PERF-002" in ids
    assert "EIG-005" in ids
    assert "COUP-019" in ids
    assert "MLBC-001" in ids
    assert "MLBC-009" in ids
    assert "CONTACT-001" in ids
    assert "CONTACT-006" in ids
    assert "CONTACT-012" in ids
    assert "FRACT-001" in ids
    assert "FRACT-012" in ids
    assert "CYL-003" in ids
    assert "NLG-008" in ids
    assert "DYN-001" in ids
    assert "EXT-002" in ids
    assert "VVR-001" in ids


def test_nlg_008_follower_pressure_case_passes() -> None:
    report = run_beam_shell_verification(selected_ids={"NLG-008"})

    assert report["status"] == "passed"
    assert report["required_failures"] == []
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["case_id"] == "NLG-008"
    assert result["status"] == "PASS"
    ring_diagnostics = result["checks"]["ring_buckling"]
    assert ring_diagnostics["solver"] == "dense_scipy_eigh_rigid_quotient"
    assert ring_diagnostics["rigid_body_handling"] == "projected"
    assert ring_diagnostics["rigid_projection"]["applied"] is True
    assert ring_diagnostics["rigid_projection"]["metric_version"] == "dimensionless_full_dof_bbox_v1"


def test_composite_buckling_rows_are_cached_only_within_one_report(
    monkeypatch,
) -> None:
    """Duplicate verification views reuse one exact buckling computation."""

    calls: list[str] = []

    def rows(metric: str):
        calls.append(metric)
        assert metric == "buckling"
        return [
            {
                "relative_error": 0.0,
                "critical_load_factor": 1.0,
                "nested": {"value": 0.0},
            },
            {
                "relative_error": 0.0,
                "critical_load_factor": 1.0,
                "nested": {"value": 0.0},
            },
        ]

    monkeypatch.setattr(beam_shell_verification, "composite_strip_metric_rows", rows)
    first = run_beam_shell_verification(selected_ids={"BUC-005", "COUP-016"})
    second = run_beam_shell_verification(selected_ids={"BUC-005", "COUP-016"})

    assert calls == ["buckling", "buckling"]
    assert [item["case_id"] for item in first["results"]] == [
        "COUP-016",
        "BUC-005",
    ]
    assert [item["case_id"] for item in second["results"]] == [
        "COUP-016",
        "BUC-005",
    ]
    first["results"][0]["checks"]["rows"][0]["nested"]["value"] = 99.0
    assert second["results"][0]["checks"]["rows"][0]["nested"]["value"] == 0.0


def test_plate_buckling_is_cached_only_within_one_report(
    monkeypatch,
) -> None:
    """The BUC-004 compatibility view reuses SHELL-007's exact solve."""

    calls = 0

    def solve(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            solver_status="ok",
            critical_load_factor=beam_shell_verification._plate_uniaxial_buckling_resultant(),
            diagnostics={"nested": {"residual": 0.0}},
        )

    monkeypatch.setattr(beam_shell_verification, "solve_eigenvalue_buckling", solve)
    first = run_beam_shell_verification(selected_ids={"SHELL-007", "BUC-004"})
    second = run_beam_shell_verification(selected_ids={"SHELL-007", "BUC-004"})

    assert calls == 2
    assert [item["case_id"] for item in first["results"]] == [
        "SHELL-007",
        "BUC-004",
    ]
    first["results"][0]["checks"]["nested"]["residual"] = 99.0
    assert second["results"][0]["checks"]["nested"]["residual"] == 0.0


def test_external_reference_decks_are_cached_only_within_one_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The two deck-inspection views share one immutable handoff bundle."""

    calls = 0
    original = beam_shell_verification.generate_external_reference_report

    def generate(deck_dir: Path):
        nonlocal calls
        calls += 1
        return original(deck_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(beam_shell_verification, "generate_external_reference_report", generate)
    first = run_beam_shell_verification(selected_ids={"EXT-001", "EXT-002"})
    second = run_beam_shell_verification(selected_ids={"EXT-001", "EXT-002"})

    assert calls == 2
    assert [item["case_id"] for item in first["results"]] == ["EXT-001", "EXT-002"]
    assert [item["case_id"] for item in second["results"]] == ["EXT-001", "EXT-002"]
    first["results"][0]["checks"]["report"]["cases"][0]["name"] = "mutated"
    assert second["results"][0]["checks"]["report"]["cases"][0]["name"] != "mutated"


def test_nonlinear_framework_ingredients_are_cached_only_within_one_report(
    monkeypatch,
) -> None:
    """NLG-003 shares its three mechanics checks with standalone manifest rows."""

    calls: list[str] = []

    def evaluator(case):
        calls.append(case.case_id)
        return beam_shell_verification._pass(case, checks={"nested": {"value": 0.0}})

    monkeypatch.setattr(beam_shell_verification, "_evaluate_nlg_002", evaluator)
    monkeypatch.setattr(beam_shell_verification, "_evaluate_nlg_007", evaluator)
    monkeypatch.setattr(beam_shell_verification, "_evaluate_nlg_008", evaluator)
    selected = {"NLG-002", "NLG-003", "NLG-007", "NLG-008"}
    first = run_beam_shell_verification(selected_ids=selected)
    second = run_beam_shell_verification(selected_ids=selected)

    assert calls == ["NLG-002", "NLG-003", "NLG-003"] * 2
    assert [item["case_id"] for item in first["results"]] == [
        "NLG-002",
        "NLG-003",
        "NLG-007",
        "NLG-008",
    ]
    assert [item["case_id"] for item in second["results"]] == [
        "NLG-002",
        "NLG-003",
        "NLG-007",
        "NLG-008",
    ]
    first["results"][0]["checks"]["nested"]["value"] = 99.0
    assert second["results"][0]["checks"]["nested"]["value"] == 0.0


def test_beam_shell_verification_report_separates_pass_and_xfail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_beam_shell_verification()
    (tmp_path / "beam_shell_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    failed = [row for row in report["results"] if row["status"] == "FAIL"]

    assert report["status"] == "passed", failed
    assert report["test_execution_status"] == "passed"
    assert report["verification_completion_status"] == "complete"
    assert report["release_gate_status"] == "passed"
    assert "verification_programme" in report
    assert "V1" in report["verification_programme"]["batches"]
    assert "fully_documented_verified_release" in report["release_gates"]
    assert report["counts"]["PASS"] >= 35
    assert report["counts"]["XFAIL"] >= 1
    assert report["required_failures"] == []
    statuses = {item["case_id"]: item["status"] for item in report["results"]}
    semantics = {item["case_id"]: item for item in report["results"]}
    assert statuses["META-001"] == "PASS"
    assert semantics["META-001"]["batch"] == "V1"
    assert semantics["META-001"]["evidence_type"] == "invariant"
    assert "solver_commit" in semantics["META-001"]
    assert statuses["ALG-001"] == "PASS"
    assert statuses["BEAM-001"] == "PASS"
    assert statuses["SHELL-005"] == "PASS"
    assert statuses["COUP-007"] == "PASS"
    assert statuses["SHELL-008"] == "PASS"
    assert statuses["BUC-005"] == "PASS"
    assert statuses["BENCH-002"] == "PASS"
    assert statuses["COUP-014"] == "PASS"
    assert statuses["COUP-015"] == "PASS"
    assert statuses["COUP-016"] == "PASS"
    assert statuses["COUP-017"] == "PASS"
    assert statuses["EIG-005"] == "PASS"
    assert statuses["PERF-001"] == "PASS"
    assert statuses["PERF-002"] == "PASS"
    assert statuses["MLBC-001"] == "PASS"
    assert statuses["MLBC-009"] == "PASS"
    assert statuses["CONTACT-001"] == "PASS"
    assert statuses["CONTACT-006"] == "PASS"
    assert statuses["CONTACT-012"] == "PASS"
    assert statuses["FRACT-001"] == "PASS"
    assert statuses["FRACT-012"] == "PASS"
    assert semantics["COUP-014"]["test_execution_status"] == "passed"
    assert semantics["COUP-014"]["verification_completion_status"] == "complete"
    assert semantics["COUP-014"]["checks"]["rows"]
    assert report["release_gates"]["thin_stiffened_shell"]["status"] == "passed"
    assert report["release_gates"]["thin_stiffened_shell"]["alias_for"] == "flat_thin_stiffened_shell"
    assert report["release_gates"]["flat_thin_stiffened_shell"]["status"] == "passed"
    assert report["release_gates"]["flat_thin_shell"]["status"] == "passed"
    assert report["release_gates"]["mesh_load_bc"]["status"] == "passed"
    assert report["release_gates"]["contact"]["status"] == "passed"
    assert report["release_gates"]["simplified_fracture"]["status"] == "passed"
    assert report["release_gates"]["thin_stiffened_shell"]["release_gate_status"] == "passed"
    assert report["release_gates"]["thin_stiffened_shell"]["verification_completion_status"] == "complete"
    blocker_ids = {item["case_id"] for item in report["release_gates"]["thin_stiffened_shell"]["blockers"]}
    assert blocker_ids == set()
    curved_blocker_ids = {item["case_id"] for item in report["release_gates"]["curved_thin_stiffened_shell"]["blockers"]}
    assert curved_blocker_ids == set()
    assert report["release_gates"]["curved_thin_stiffened_shell"]["status"] == "passed"


def test_external_deck_case_is_a_handoff_pass_without_numerical_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    external_report_path = tmp_path / "verification" / "external_reference_report.json"

    report = run_beam_shell_verification(
        selected_ids=["EXT-001", "VVR-001"],
        external_reference_report=external_report_path,
    )
    results = {item["case_id"]: item for item in report["results"]}
    external = results["EXT-001"]
    package = results["VVR-001"]

    assert external["status"] == "PASS"
    assert external["evidence_type"] == "handoff_artifact"
    assert external["analysis_type"] == "external_solver_handoff_artifact"
    assert external["reference"]["numerical_validation_claim"] is False
    assert external["result"]["external_report_status"] == "not_executed"
    assert external["checks"]["evidence"]["numerical_validation_performed"] is False

    manifest = package["checks"]
    assert package["status"] == "PASS"
    assert manifest["external_evidence_kind"] == "handoff_artifact"
    assert manifest["external_handoff_artifact_status"] == "passed"
    assert manifest["external_numerical_validation_performed"] is False
    assert manifest["external_numerical_validation_status"] == "not_performed"
    assert manifest["external_reference_report_status"] == "not_executed"
    assert external_report_path.exists()


def test_vvr_preserves_an_executed_external_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "preserved_evidence" / "external_reference_report.json"
    markdown_path = report_path.with_suffix(".md")
    report_path.parent.mkdir(parents=True)
    executed_report = {
        "schema_version": 2,
        "status": "passed",
        "execution_mode": "calculix",
        "validation_performed": True,
        "solver": {"name": "CalculiX CrunchiX", "version": "2.22"},
        "cases": [
            {
                "name": "preserved_case",
                "validation": {
                    "status": "passed",
                    "executed": True,
                    "comparisons": [{"status": "passed"}],
                },
            }
        ],
    }
    original_json = json.dumps(executed_report, indent=2) + "\n"
    original_markdown = "# Preserved executed CalculiX report\n"
    report_path.write_text(original_json, encoding="utf-8")
    markdown_path.write_text(original_markdown, encoding="utf-8")

    report = run_beam_shell_verification(
        selected_ids=["VVR-001"],
        external_reference_report=report_path,
    )
    package = report["results"][0]
    manifest = package["checks"]

    assert package["status"] == "PASS"
    assert manifest["external_report_disposition"] == "preserved_existing"
    assert manifest["external_evidence_kind"] == "executed_numerical_validation"
    assert manifest["external_numerical_validation_performed"] is True
    assert manifest["external_numerical_validation_status"] == "passed"
    assert report_path.read_text(encoding="utf-8") == original_json
    assert markdown_path.read_text(encoding="utf-8") == original_markdown


def test_vvr_replaces_a_stale_nonexecuted_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "legacy" / "external_reference_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"schema_version": 1, "status": "passed", "cases": []})
        + "\n",
        encoding="utf-8",
    )

    report = run_beam_shell_verification(
        selected_ids=["VVR-001"],
        external_reference_report=report_path,
    )
    package = report["results"][0]
    manifest = package["checks"]
    replacement = json.loads(report_path.read_text(encoding="utf-8"))

    assert package["status"] == "PASS"
    assert manifest["external_report_disposition"] == "replaced_invalid_nonexecuted"
    assert replacement["execution_mode"] == "deck_only"
    assert replacement["validation_performed"] is False
    assert replacement["status"] == "not_executed"


def test_vvr_preserves_and_rejects_invalid_executed_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "invalid" / "external_reference_report.json"
    report_path.parent.mkdir(parents=True)
    invalid_executed = {
        "schema_version": 2,
        "status": "failed",
        "execution_mode": "calculix",
        "validation_performed": True,
        "cases": [
            {
                "name": "failed_case",
                "validation": {
                    "status": "failed",
                    "executed": True,
                    "comparisons": [{"status": "failed"}],
                },
            }
        ],
    }
    original = json.dumps(invalid_executed, indent=2) + "\n"
    report_path.write_text(original, encoding="utf-8")

    report = run_beam_shell_verification(
        selected_ids=["VVR-001"],
        external_reference_report=report_path,
    )
    manifest = json.loads(
        (
            tmp_path
            / "reports"
            / "verification_package"
            / "release_evidence_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert report["results"][0]["status"] == "FAIL"
    assert manifest["status"] == "incomplete"
    assert manifest["external_report_disposition"] == "preserved_invalid_executed"
    assert report_path.read_text(encoding="utf-8") == original


def test_beam_shell_verification_report_writer_and_cli(tmp_path: Path) -> None:
    output_dir = tmp_path
    output = output_dir / "beam_shell_verification.json"
    markdown = output_dir / "beam_shell_verification.md"

    report = write_beam_shell_verification_report(output, markdown=markdown, selected_ids=["ALG-001", "BENCH-001"])

    assert report["status"] == "passed"
    assert output.exists()
    assert markdown.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [item["case_id"] for item in payload["results"]] == ["ALG-001", "BENCH-001"]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_beam_shell_verification.py",
            "--output",
            str(output_dir / "cli.json"),
            "--markdown",
            str(output_dir / "cli.md"),
            "--case-id",
            "ALG-001",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
