"""Bounded ANYsolver 0.4.7 performance-release guards."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from anyfesolver import __version__, b3_ge
from anyfesolver.elements import QuadraticBeamElement, create_element
from scripts import verify_release_047 as gate


ROOT = Path(__file__).resolve().parents[1]


def test_release_identity_and_g7_boundary_are_exact() -> None:
    assert gate.VERSION == "0.4.7"
    assert __version__ == "0.5.0"
    records = gate._validate_g7()
    assert set(records) == {"confirmation", "contract", "review", "status"}
    evidence = gate._validate_performance_evidence()
    assert set(evidence) == {"static", "spectral", "nonlinear", "nonlinear_review"}
    assert b3_ge.SELECTOR == "b3-ge"
    assert b3_ge.NAME == "B3-GE"
    assert type(create_element("quadratic_beam", 1, [1, 2, 3])) is QuadraticBeamElement
    with pytest.raises(ValueError, match="Unknown element type"):
        create_element("b3-ge", 2, [1, 2, 3])


def test_evidence_hash_uses_committed_text_across_checkout_line_endings() -> None:
    assert gate._normalized_evidence_bytes(b"one\r\ntwo\r\n") == b"one\ntwo\n"
    assert gate._normalized_evidence_bytes(b"one\ntwo\n") == b"one\ntwo\n"
    with pytest.raises(gate.GateError, match="unsupported evidence line ending"):
        gate._normalized_evidence_bytes(b"one\rtwo\n")


def test_release_archive_contains_only_current_user_documents() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    included_docs = {
        line.removeprefix("include ")
        for line in manifest
        if line.startswith("include docs/")
    }
    assert included_docs == gate.RELEASE_DOCS | {
        "docs/NATIVE_SOURCE_AUTHORITY.md", "docs/releases/ANYfeSolver_0.5.0.md"
    }
    assert {
        "prune docs",
        "prune reports",
        "prune scripts",
        "prune tests",
    } <= set(manifest)


@pytest.mark.parametrize(
    "payload, message",
    (
        ('{"a":1,"a":2}\n', "duplicate JSON key"),
        ('{"a":NaN}\n', "nonfinite JSON value"),
        ('{"b":1,"a":2}\n', "not canonical"),
    ),
)
def test_strict_json_rejects_malformed_evidence(tmp_path: Path, payload: str, message: str) -> None:
    path = tmp_path / "record.json"
    path.write_text(payload, encoding="ascii")
    with pytest.raises(gate.GateError, match=message):
        gate._strict_json(path)


def test_publish_workflow_uses_bounded_gate_before_upload() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    command = "python scripts/verify_release_050.py"
    upload = "name: release-050-distributions"
    assert workflow.count(command) == 1
    assert workflow.index(command) < workflow.index(upload)
    assert "--solver-wheel dist/solver/anyfesolver-${SOLVER_VERSION}-py3-none-any.whl" in workflow
    assert "--solver-sdist dist/solver/anyfesolver-${SOLVER_VERSION}.tar.gz" in workflow
    assert '--expected-commit "$EXPECTED_SHA"' in workflow


def test_full_ci_can_be_dispatched_for_a_release_branch() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "  pull_request:\n  workflow_dispatch:\n" in workflow


def test_installed_probe_installs_declared_runtime_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "-c" in command:
            payload = {
                "dependencies": {
                    "ANYfileio": "0.3.2",
                    "ANYgeometry": "0.4.3",
                    "ANYmaterial": "0.2.0",
                    "ANYmesher": "0.5.0",
                },
                "installed_origin": True,
                "legacy_b3_default": True,
                "name": "B3-GE",
                "selector": "b3-ge",
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert gate._installed_probe(tmp_path / "anysolver-0.4.7-py3-none-any.whl")[
        "legacy_b3_default"
    ]
    assert calls[0][1:5] == ["-m", "pip", "install", "--disable-pip-version-check"]
    assert "--no-deps" not in calls[0]
    assert "--target" in calls[0]


def test_release_gate_result_serializes_without_nonfinite_values() -> None:
    sample = {
        "schema": "ANYSOLVER_0_4_7_BOUNDED_PERFORMANCE_RELEASE_GATE_V1",
        "terminal": gate.TERMINAL,
    }
    assert json.dumps(sample, sort_keys=True, separators=(",", ":"), allow_nan=False)
