from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
PROGRAM = REFERENCE / "e4_pl_s3_v5a_preregistration.py"
SELECTION = REFERENCE / "e4_pl_s3_v5a_source_selection.json"
CONTRACT = REFERENCE / "e4_pl_s3_v5a_preregistration_contract.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("s3_v5a_preregistration", PROGRAM)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_uses_only_standard_library() -> None:
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "typing",
    }
    tree = ast.parse(PROGRAM.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= allowed


def test_source_selection_is_complete_only_for_unrelaxed_screen() -> None:
    module = _load_module()
    selection = module.load_canonical(SELECTION)
    assert selection["candidate"]["implementation_status"] == "NOT_IMPLEMENTED"
    assert selection["candidate"]["relaxation_policy"] == "PHI_SQUARED_EXACTLY_ONE_FOR_SCREEN_ONLY"
    assert selection["equation_authority"]["complete_for_unrelaxed_local_interface_screen"] is True
    assert selection["equation_authority"]["relaxed_or_thin_regime_screen_authorized"] is False
    assert selection["screen_scope"]["nonpatch_q4_dirichlet_to_neumann"] == "DIAGNOSTIC_ONLY"


def test_preregistration_authorizes_only_bounded_unrelaxed_screen() -> None:
    module = _load_module()
    result = module.validate()
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V5A_UNRELAXED_LOCAL_INTERFACE_SCREEN"
    assert result["next_gate_authorized"] is True
    assert result["empirical_relaxation_authorized"] is False
    assert result["relaxed_or_thin_regime_screen_authorized"] is False
    assert result["stage4a_rerun_authorized"] is result["activation_authorized"] is False


def test_two_exclusive_runs_are_byte_identical(tmp_path: Path) -> None:
    outputs = [tmp_path / "cycle1" / "result.json", tmp_path / "cycle2" / "result.json"]
    for output in outputs:
        subprocess.run(
            [sys.executable, str(PROGRAM), "--output", str(output)],
            cwd=ROOT,
            check=True,
            timeout=30,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert hashlib.sha256(outputs[0].read_bytes()).hexdigest().upper() == hashlib.sha256(outputs[1].read_bytes()).hexdigest().upper()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [sys.executable, str(PROGRAM), "--output", str(outputs[0])],
            cwd=ROOT,
            check=True,
            timeout=30,
        )


def test_duplicate_nonfinite_and_authority_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        module.load_canonical(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        module.load_canonical(nonfinite)

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    selection["equation_authority"]["relaxed_or_thin_regime_screen_authorized"] = True
    mutated = tmp_path / "selection.json"
    mutated.write_bytes(module.canonical_bytes(selection))
    monkeypatch.setattr(module, "SELECTION", mutated)
    with pytest.raises(ValueError, match="equation-authority"):
        module.validate()


def test_contract_freezes_bounds_terminal_precedence_and_production_boundary() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["execution"] == {
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_cycle_count": 2,
    }
    assert contract["terminal_precedence"] == [
        "BLOCKED_E4_PL_S3_V5A_PROCESS_OR_EVIDENCE",
        "NO_GO_E4_PL_S3_V5A_SOURCE_OR_LOCAL_OPERATOR",
        "NO_GO_E4_PL_S3_V5A_MIXED_INTERFACE",
        "UNCLASSIFIED_E4_PL_S3_V5A_RELAXATION_AUTHORITY_REQUIRED",
        "PROVISIONAL_GO_E4_PL_S3_V5A_RELAXED_SUCCESSOR_SCREEN",
    ]
    assert contract["production_boundary"] == {
        "anymesh_untouched": True,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
        "s3_activation_authorized": False,
    }


def test_frozen_inputs_and_external_receipts_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    assert len(selection["external_sources"]) == 3
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in selection["external_sources"])


def test_production_boundary_sources_remain_unchanged() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v5a*"))
