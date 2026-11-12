from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
PROGRAM = REFERENCE / "e4_pl_s3_v5b_relaxation_authority.py"
SELECTION = REFERENCE / "e4_pl_s3_v5b_relaxation_source_selection.json"
CONTRACT = REFERENCE / "e4_pl_s3_v5b_relaxation_authority_contract.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("s3_v5b_relaxation_authority", PROGRAM)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_uses_only_standard_library() -> None:
    allowed = {
        "__future__",
        "argparse",
        "fractions",
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


def test_exact_mystran_to_uhm_parameter_map() -> None:
    module = _load_module()
    for alpha in (
        Fraction(1, 1000),
        Fraction(1, 7),
        Fraction(1, 1),
        Fraction(19, 3),
        Fraction(1000000, 1),
    ):
        assert module.relaxation_from_mystran(alpha) == module.relaxation_from_uhm(alpha)
    assert module.relaxation_from_uhm(Fraction(3, 2)) == Fraction(4, 7)
    with pytest.raises(ValueError, match="positive"):
        module.relaxation_from_uhm(Fraction(0))


def test_authority_selects_source_constant_without_fitting() -> None:
    module = _load_module()
    selection = module.load_canonical(SELECTION)
    assert selection["equation_authority"] == {
        "coefficient_fitting_forbidden": True,
        "complete_for_bounded_relaxed_repair_funnel": True,
        "mystran_cbmin3": "2",
        "uhm_c_s": "1/2",
    }
    assert selection["exact_mapping"]["proof_steps"][-1] == "C_S=1/CBMIN3=1/2"


def test_authority_allows_only_bounded_v5b_screen() -> None:
    result = _load_module().validate()
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V5B_RELAXED_REPAIR_FUNNEL"
    assert result["next_gate_authorized"] is True
    assert result["empirical_coefficient_fitting_authorized"] is False
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


def test_duplicate_nonfinite_and_equation_mutations_fail_closed(
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
    selection["equation_authority"]["uhm_c_s"] = "2"
    mutated = tmp_path / "selection.json"
    mutated.write_bytes(module.canonical_bytes(selection))
    monkeypatch.setattr(module, "SELECTION", mutated)
    with pytest.raises(ValueError, match="equation authority"):
        module.validate()


def test_official_repository_files_and_manual_are_hash_bound() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    sources = selection["external_sources"]
    source_repo = next(row for row in sources if row["authority"] == "OFFICIAL_MIN3_RELAXATION_IMPLEMENTATION")
    docs_repo = next(row for row in sources if row["authority"] == "OFFICIAL_USER_MANUAL_PARAMETER_AND_REFERENCE_IDENTITY")
    assert len(source_repo["files"]) == 5
    assert len(docs_repo["files"]) == 2
    assert all(item["bytes"] > 0 and len(item["sha256"]) == 64 and len(item["blob"]) == 40 for row in (source_repo, docs_repo) for item in row["files"])
    assert docs_repo["reference_three"]["doi"] == "10.1016/0045-7825(85)90114-8"


def test_contract_freezes_bounds_terminal_precedence_and_inputs() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["terminal_precedence"] == [
        "BLOCKED_E4_PL_S3_V5B_PROCESS_OR_EVIDENCE",
        "NO_GO_E4_PL_S3_V5B_RELAXATION_SOURCE_OR_LOCAL_OPERATOR",
        "NO_GO_E4_PL_S3_V5B_MIXED_INTERFACE",
        "NO_GO_E4_PL_S3_V5B_THIN_REGIME",
        "PROVISIONAL_GO_E4_PL_S3_V5B_STAGE4A_RERUN",
    ]
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]


def test_production_boundary_remains_unchanged() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v5b*"))
