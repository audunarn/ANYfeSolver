from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs" / "reference_cases" / "ge_beam3_mixed_exact_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ge_beam3_mixed_exact_reference", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_reference_certificate_closes_linear_algebra() -> None:
    module = _load_module()
    record = module.build_certificate()
    assert record["counts"] == {
        "cells": 2,
        "external_dofs": 18,
        "internal_dofs": 18,
        "internal_moment_dofs": 12,
        "internal_rotation_dofs": 6,
    }
    assert record["hashes"]["condensed_operator_sha256"] == (
        "15D9169BACB17DBAEE066DBB8C10CBD07C564D5146093AB0A59CBC0774102747"
    )
    assert record["integration"] == {
        "complementary_moment_polynomial_degree": 2,
        "force_strain_polynomial_degree": 0,
        "force_term_gauss_points": 1,
        "moment_term_gauss_points": 2,
    }
    assert all(record["predicates"].values())
    assert record["terminal"] == "NONCLASSIFYING_EXACT_LOCAL_GATE_PASS"


def test_mutations_are_detected() -> None:
    module = _load_module()
    reversal = module.build_certificate(mutate="reversal")
    entry = module.build_certificate(mutate="condensed_entry")
    section = module.build_certificate(mutate="section")
    assert reversal["predicates"]["reversal_covariance_exact"] is False
    assert entry["predicates"]["six_rigid_modes_exact"] is False
    assert section["terminal"] == "NONCLASSIFYING_EXACT_LOCAL_GATE_FINDING"
    assert section["hashes"]["condensed_operator_sha256"] != (
        "15D9169BACB17DBAEE066DBB8C10CBD07C564D5146093AB0A59CBC0774102747"
    )


def test_cli_is_canonical_deterministic_and_exclusive(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        subprocess.run(
            [sys.executable, str(MODULE_PATH), "--output", str(output)],
            cwd=ROOT,
            check=True,
            timeout=30,
        )
    assert first.read_bytes() == second.read_bytes()
    parsed = json.loads(first.read_text(encoding="utf-8"))
    assert first.read_bytes() == (
        json.dumps(parsed, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    duplicate = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--output", str(first)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert duplicate.returncode != 0


def test_reference_builder_has_no_production_or_symbolic_import() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"anyfesolver", "numpy", "scipy", "sympy", "mpmath"})


def test_preregistration_records_are_strict_json_and_preserve_v1() -> None:
    directory = ROOT / "docs" / "reference_cases"
    names = (
        "ge_beam3_mixed_baseline.json",
        "ge_beam3_mixed_equation_map_a.json",
        "ge_beam3_mixed_equation_map_b.json",
        "ge_beam3_mixed_local_contract.json",
        "ge_beam3_mixed_source_ledger.json",
        "ge_beam3_mixed_status.json",
    )
    records = {}
    for name in names:
        text = (directory / name).read_text(encoding="utf-8")
        records[name] = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    baseline = records["ge_beam3_mixed_baseline.json"]
    assert baseline["v1_closeout"]["terminal"] == "NO_GO_GE_BEAM3_DISCRETE_VARIATIONAL_IDENTITY"
    assert baseline["archive_ref"] == "refs/archive/ge-beam3-sr-p1-straight-v1-nogo-20260904"
    contract = records["ge_beam3_mixed_local_contract.json"]
    assert contract["scientific_execution_authorized"] is False
    assert contract["production_boundary"]["selector_available"] is False
    assert "CLAIM_DYNAMICS_OR_HISTORY_SECTION_QUALIFICATION" in contract["fatal_local_gate"]["forbidden"]
