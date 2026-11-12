"""Production-readiness artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.production_readiness import (
    build_capability_matrix,
    build_verification_scope_statement,
    scope_statement_markdown,
    write_production_readiness_artifacts,
)
from anyfesolver.validation import validate_production_model


def _fake_report() -> dict:
    return {
        "release_gates": {
            "flat_thin_shell": {
                "status": "blocked",
                "blockers": [{"case_id": "SHELL-009"}],
            },
            "flat_thin_stiffened_shell": {
                "status": "blocked",
                "blockers": [{"case_id": "COUP-012"}, {"case_id": "BUC-005"}],
            },
            "curved_thin_stiffened_shell": {
                "status": "not_evaluated",
                "blockers": [],
            },
            "nonlinear_capacity": {
                "status": "blocked",
                "blockers": [{"case_id": "NLG-007"}],
            },
            "fully_documented_verified_release": {
                "status": "blocked",
                "blockers": [{"case_id": "EXT-001"}],
            },
        }
    }


def test_capability_matrix_reflects_blocked_gates() -> None:
    matrix = build_capability_matrix(_fake_report())
    by_feature = {entry.feature: entry for entry in matrix}

    assert by_feature["flat_thin_shell_linear_static_modal_buckling"].status == "not_qualified"
    assert by_feature["flat_thin_shell_linear_static_modal_buckling"].gate_blockers == ["SHELL-009"]
    assert by_feature["flat_thin_shell_linear_static_modal_buckling"].limits["shell_formulations"] == ["Q4", "Q8"]
    assert any(
        "Q8R is experimental" in limitation
        for limitation in by_feature["flat_thin_shell_linear_static_modal_buckling"].limitations
    )
    assert by_feature["curved_thin_stiffened_shell"].status == "not_evaluated"
    s3 = by_feature["qualified_s3_companion_shell"]
    assert s3.status == "qualified"
    assert s3.release_gate == "qualified_s3_companion_activation"
    assert s3.limits == {
        "default_formulation": "e4-pl-s3-v2d",
        "explicit_selectors": [
            "e4-pl-s3",
            "qualified-s3",
            "e4-pl-s3-v2d",
            "qualified-s3-v2d",
        ],
        "formulation_id": "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
        "shell_topology": "S3",
    }
    assert s3.verification_cases == ["S3-V6W"]
    assert s3.gate_blockers == []
    assert by_feature["unsupported_general_purpose_fe"].status == "unsupported"


def test_production_validation_marks_q8r_as_experimental() -> None:
    model = FEModel("q8r_validation")
    model.add_material("steel", 210.0e9, 0.3)
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 1.0, 0.0),
        (0.0, 0.5, 0.0),
    )
    for node_id, coordinate in enumerate(coordinates, start=1):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        ShellElement(1, list(range(1, 9)), "steel", thickness=0.01, reduced_integration=True),
    )

    report = validate_production_model(model, allow_free_mechanisms=True)
    issue = next(item for item in report.issues if item.code == "SHELL002")
    assert issue.severity == "warning"
    assert "experimental" in issue.message


def test_scope_statement_and_artifact_writer(tmp_path: Path) -> None:
    report = _fake_report()
    scope = build_verification_scope_statement(report)
    markdown = scope_statement_markdown(scope)
    output_dir = tmp_path / "production-readiness"

    assert scope["production_release_status"] == "not_qualified"
    assert "unsupported_general_purpose_fe" in scope["unsupported"]
    assert "qualified_s3_companion_shell" in scope["verified"]
    assert any(
        "Follower pressure is limited to nonlinear static and arc-length equilibrium"
        in limitation
        for limitation in scope["explicit_limitations"]
    )
    assert "Production release status: not_qualified" in markdown

    result = write_production_readiness_artifacts(output_dir, report=report)
    assert result["production_release_status"] == "not_qualified"
    matrix = json.loads((output_dir / "capability_matrix.json").read_text(encoding="utf-8"))
    scope_payload = json.loads((output_dir / "verification_scope.json").read_text(encoding="utf-8"))
    assert len(matrix) >= 5
    assert scope_payload["production_release_status"] == "not_qualified"
    assert (output_dir / "verification_scope.md").exists()
