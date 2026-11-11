"""Focused mesh/load/boundary-condition verification tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from anyfesolver import (
    mesh_load_bc_manifest_cases,
    run_mesh_load_bc_verification,
    write_mesh_load_bc_verification_report,
)


def test_mesh_load_bc_manifest_contains_stable_case_ids() -> None:
    cases = mesh_load_bc_manifest_cases()
    ids = [case.case_id for case in cases]

    assert ids == [f"MLBC-{index:03d}" for index in range(1, 10)]
    assert all(case.required for case in cases)


def test_mesh_load_bc_verification_report_passes() -> None:
    report = run_mesh_load_bc_verification()
    statuses = {item["case_id"]: item["status"] for item in report["results"]}
    by_case = {item["case_id"]: item for item in report["results"]}

    assert report["status"] == "passed"
    assert report["counts"] == {"PASS": 9}
    assert report["required_failures"] == []
    assert statuses == {f"MLBC-{index:03d}": "PASS" for index in range(1, 10)}
    assert by_case["MLBC-004"]["measured"]["flat_positive_force"] == [0.0, 0.0, 10.0]
    assert by_case["MLBC-005"]["measured"]["selected_element_ids"] == [1]
    assert abs(by_case["MLBC-009"]["measured"]["free_extension"] - 0.1) < 1.0e-10


def test_mesh_load_bc_report_writer_and_cli() -> None:
    output_dir = Path(".pytest_tmp_mesh_load_bc_verification")
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "mesh_load_bc_verification.json"
    markdown = output_dir / "mesh_load_bc_verification.md"

    report = write_mesh_load_bc_verification_report(output, markdown=markdown, selected_ids=["MLBC-004", "MLBC-009"])

    assert report["status"] == "passed"
    assert output.exists()
    assert markdown.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [item["case_id"] for item in payload["results"]] == ["MLBC-004", "MLBC-009"]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_mesh_load_bc_verification.py",
            "--output",
            str(output_dir / "cli.json"),
            "--markdown",
            str(output_dir / "cli.md"),
            "--case-id",
            "MLBC-004",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    cli_payload = json.loads((output_dir / "cli.json").read_text(encoding="utf-8"))
    assert [item["case_id"] for item in cli_payload["results"]] == ["MLBC-004"]
