from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs" / "reference_cases"
CONTRACT = CASES / "e4_pl_q1h_domain_contract.json"
EVIDENCE = CASES / "e4_pl_q1h_domain_evidence.json"
CONTRACT_SHA256 = "11B9D7B616A004312D0D7F23610546BE2F4073B8C5AC5B553870B3DF4F274A8E"
EXTERNAL_CYCLE_SHA256 = "150975B3F362691CC2670C5CC36DE7F63430DBCA7E46576F2866C7D7B56765CF"
PARTITION_SHA256 = "1365DAB7F792C1920C4F8AB10DF2E8F65395D613EFFAE5CD51E323AA4092AA00"


def _reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> dict[str, object]:
    value = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite: {token}")),
    )
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if canonical != raw:
        raise ValueError("not canonical")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _external_environment() -> Path:
    value = os.environ.get("Q1T_EXACT_ENV_ROOT")
    if value:
        return Path(value)
    return Path(r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease\q1t-exact-environment-20260820-h")


def test_q1h_contract_evidence_hash_dag_and_strict_json() -> None:
    contract_raw = CONTRACT.read_bytes()
    contract = _strict_json(contract_raw)
    assert _sha256(contract_raw) == CONTRACT_SHA256
    assert contract["alpha_star"] == [1, 1_000_000]
    assert contract["h_kernel_certificate"]["rigid_range_authority"] == "Q1G_ESTABLISHED_NOT_REVISITED"
    assert contract["coercivity_certificate"]["local_to_global"] == "INHERITED_Q1F_ELEMENT_SUM_WITH_NO_MESH_CONSTANT"
    for row in contract["inputs"]:
        raw = (ROOT / row["path"]).read_bytes()
        assert len(raw) == row["bytes"]
        assert _sha256(raw) == row["sha256"]

    evidence = _strict_json(EVIDENCE.read_bytes())
    assert evidence["contract"]["sha256"] == CONTRACT_SHA256
    assert evidence["classification"] == "PROVISIONAL_GO_E4_PL_Q1H_DOMAIN_COERCIVITY"
    assert evidence["cycles"] == [
        {"bytes": 1_984_603, "record_id": "cycle_1", "sha256": EXTERNAL_CYCLE_SHA256},
        {"bytes": 1_984_603, "record_id": "cycle_2", "sha256": EXTERNAL_CYCLE_SHA256},
    ]
    coverage = evidence["domain_coverage"]
    assert coverage["coverage_complete"] is True
    assert coverage["partition_sha256"] == PARTITION_SHA256
    assert coverage["processed_count"] == 9_645
    assert coverage["positive_leaf_count"] == 3_955
    assert coverage["excluded_leaf_count"] == 868
    assert coverage["unresolved_leaf_count"] == coverage["pending_count"] == 0
    assert evidence["q1g_disposition"] == {
        "domain_gap_closed": True,
        "rigid_range_recomputed": False,
        "rigid_range_result_inherited": True,
    }
    for row in evidence["implementation_inputs"]:
        raw = (ROOT / row["path"]).read_bytes()
        assert len(raw) == row["bytes"]
        assert _sha256(raw) == row["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        _strict_json(b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="non-finite"):
        _strict_json(b'{"a":NaN}\n')


def test_q1h_independent_symbolic_control_and_h_kernel_certificates() -> None:
    environment = _external_environment()
    if not (environment / "sympy").is_dir():
        pytest.skip("frozen Q1T SymPy environment is not available")
    command_environment = os.environ.copy()
    command_environment["PYTHONPATH"] = os.pathsep.join((str(CASES), str(environment)))
    command_environment.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"
    )
    program = CASES / "e4_pl_q1h_symbolic_kernel.py"
    control = subprocess.run(
        [sys.executable, str(program), "--determinant-certificate"],
        cwd=ROOT, env=command_environment, check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        timeout=90,
    )
    control_record = json.loads(control.stdout)
    assert control_record["status"] == "PASS"
    assert control_record["determinant"] == "32*q**12/729"
    assert control_record["residual"] == "0"
    kernel = subprocess.run(
        [sys.executable, str(program), "--h-kernel-certificate"],
        cwd=ROOT, env=command_environment, check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        timeout=90,
    )
    kernel_record = json.loads(kernel.stdout)
    assert kernel_record["status"] == "PASS"
    assert kernel_record["bending_residual"] == kernel_record["membrane_residual"] == "0"
    assert kernel_record["rigid_factor_residual_nonzero_count"] == 0
    assert kernel_record["rigid_range_authority"] == "Q1G_ESTABLISHED_NOT_REVISITED"


def test_q1h_outward_leaf_certificate_and_exact_exclusion() -> None:
    environment = _external_environment()
    if environment.is_dir():
        sys.path.insert(0, str(environment))
    sys.path.insert(0, str(CASES))
    try:
        import e4_pl_q1h_coverage as coverage
        import e4_pl_q1h_interval as intervals
    except ImportError as exc:
        pytest.skip(f"research environment unavailable: {exc}")
    positive_bounds = {
        "a": (Fraction(-9, 32), Fraction(-3, 16)),
        "b": (Fraction(-3, 8), Fraction(-9, 32)),
        "p": (Fraction(-5, 4), Fraction(-1, 1)),
        "q": (Fraction(31, 64), Fraction(77, 128)),
    }
    certificate = intervals.certify_variation_box_control(positive_bounds)
    assert certificate["coercivity_certified"] is True
    assert certificate["control_congruence_margin"] > 0.39
    assert certificate["control_inverse_exact_dag"] is True
    excluded = coverage.Box(
        (
            (Fraction(-4), Fraction(-2)),
            (Fraction(1, 4), Fraction(17, 8)),
            (Fraction(-3, 8), Fraction(3, 8)),
            (Fraction(-3, 8), Fraction(3, 8)),
        ),
        depth=3,
        path="000",
    )
    assert coverage.exclusion_reason(excluded) == "CENTRE_SINGULAR_RATIO_IMPOSSIBLE"


def test_q1h_external_cycles_are_canonical_identical_complete_partitions() -> None:
    value = os.environ.get("Q1H_EVIDENCE_ROOT")
    if not value:
        pytest.skip("Q1H_EVIDENCE_ROOT is required for full external partition verification")
    directory = Path(value)
    left = (directory / "cycle1.json").read_bytes()
    right = (directory / "cycle2.json").read_bytes()
    assert left == right
    assert len(left) == 1_984_603
    assert _sha256(left) == EXTERNAL_CYCLE_SHA256
    record = _strict_json(left)
    assert record["classification"] == "PROVISIONAL_GO_E4_PL_Q1H_DOMAIN_COERCIVITY"
    assert record["coverage_complete"] is True
    assert record["pending_count"] == record["unresolved_leaf_count"] == 0
    leaves = record["leaf_records"]
    assert len(leaves) == record["positive_leaf_count"] + record["excluded_leaf_count"] == 4_823
    leaf_bytes = (json.dumps(leaves, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    assert _sha256(leaf_bytes) == record["partition_sha256"] == PARTITION_SHA256
    paths = sorted(row["path"] for row in leaves)
    assert len(paths) == len(set(paths))
    assert all(not right_path.startswith(left_path) for left_path, right_path in zip(paths, paths[1:]))
    assert math.fsum(2.0 ** -len(path) for path in paths) == 1.0
    sys.path.insert(0, str(CASES))
    import e4_pl_q1h_coverage as coverage
    coverage.verify_leaf_partition(leaves)
    mutated = json.loads(json.dumps(record))
    mutated["leaf_records"][0]["path"] = mutated["leaf_records"][1]["path"]
    with pytest.raises(ValueError, match="duplicate leaf path"):
        coverage.verify_leaf_partition(mutated["leaf_records"])


def test_q1h_research_only_extent_and_production_boundary() -> None:
    allowed = {
        "docs/agent_plans/S4_E4_PL_Q1H_DOMAIN_COERCIVITY_COMPLETION_PLAN.md",
        "docs/reference_cases/e4_pl_q1h_coverage.py",
        "docs/reference_cases/e4_pl_q1h_domain_contract.json",
        "docs/reference_cases/e4_pl_q1h_domain_evidence.json",
        "docs/reference_cases/e4_pl_q1h_interval.py",
        "docs/reference_cases/e4_pl_q1h_point_mechanics.py",
        "docs/reference_cases/e4_pl_q1h_scientific_review.json",
        "docs/reference_cases/e4_pl_q1h_status.json",
        "docs/reference_cases/e4_pl_q1h_symbolic_kernel.py",
        "tests/test_e4_pl_q1h_domain_coercivity.py",
    }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    ).stdout.splitlines()
    changed = {
        row[3:].replace("\\", "/")
        for row in status
        if not row[3:].replace("\\", "/").startswith((".pytest", "docs/reference_cases/__pycache__", "tests/__pycache__"))
    }
    assert changed <= allowed
    assert not any(path == ".gitattributes" or path == "pyproject.toml" or path.startswith(("src/", ".github/")) for path in changed)
    for source in ("e4_pl_q1h_coverage.py", "e4_pl_q1h_interval.py", "e4_pl_q1h_symbolic_kernel.py"):
        tree = ast.parse((CASES / source).read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any(name.startswith("src.anyfesolver") or name == "anyfesolver" for name in imports)
    plan = (ROOT / "docs" / "agent_plans" / "S4_E4_PL_Q1H_DOMAIN_COERCIVITY_COMPLETION_PLAN.md").read_text(encoding="utf-8")
    assert _strict_json(CONTRACT.read_bytes())["production"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert "replacement of legacy `ShellElement`" in plan
