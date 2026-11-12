from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
BASE = "3abdb32ed5fd9d3ddf8c03b780ae1c006bb96e01"


def _git_show(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def test_contract_is_canonical_and_binds_frozen_diagnosis() -> None:
    path = REFERENCE / "e4_pl_s3_v6t_global_cache_contract.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    for row in value["frozen_inputs"]:
        frozen = _git_show(row["path"])
        assert len(frozen) == row["bytes"]
        assert hashlib.sha256(frozen).hexdigest().upper() == row["sha256"]


def test_diagnosis_is_specific_to_global_reassembly() -> None:
    v2c = _git_show("src/anyfesolver/s3_v2c_fast_assembly.py").decode("utf-8")
    v2d = _git_show("src/anyfesolver/s3_v2d_fast_assembly.py").decode("utf-8")
    assembly = _git_show("src/anyfesolver/matrix_assembly.py").decode("utf-8")
    assert "lookup_v2c_global_stiffness_plan" in v2c
    assert "bind_v2c_global_stiffness_plan" in v2c
    assert "get_v2d_stiffness_plan" in v2d
    assert "lookup_v2d_global_stiffness_plan" not in v2d
    assert "lookup_v2c_global_stiffness_plan" in assembly
    assert "lookup_v2d_global_stiffness_plan" not in assembly


def test_authority_preserves_defaults_and_limits_extent() -> None:
    contract = json.loads((REFERENCE / "e4_pl_s3_v6t_global_cache_contract.json").read_text(encoding="ascii"))
    assert contract["implementation_extent"] == [
        "src/anyfesolver/matrix_assembly.py",
        "src/anyfesolver/s3_v2d_fast_assembly.py",
        "tests/test_e4_pl_s3_v6t_global_cache.py",
    ]
    elements = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in elements
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in elements
