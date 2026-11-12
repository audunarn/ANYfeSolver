from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
AUDIT_PATH = REFERENCE / "e4_pl_s3_v6f_final_parity_audit.py"
CONTRACT_PATH = REFERENCE / "e4_pl_s3_v6f_final_parity_audit_contract.json"


def _module():
    spec = importlib.util.spec_from_file_location("v6f_audit", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v6f_audit_is_standard_library_only() -> None:
    tree = ast.parse(AUDIT_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "ast",
        "hashlib",
        "json",
        "pathlib",
        "typing",
    }
    source = AUDIT_PATH.read_text(encoding="utf-8")
    assert "import anyfesolver" not in source
    assert "from anyfesolver" not in source


def test_v6f_frozen_audit_preserves_the_three_reviewed_open_routes() -> None:
    result = json.loads(
        (REFERENCE / "e4_pl_s3_v6f_final_parity_audit_result.json").read_bytes()
    )
    assert result["audit"]["open_routes"] == [
        "COMMITTED_LAYERED_PUBLIC_RECOVERY",
        "CURRENT_STATE_MODAL_AND_BUCKLING",
        "GLOBAL_TENSOR_AND_PATCH_RECOVERY",
    ]
    assert result["audit"]["open_route_count"] == 3
    assert all(result["audit"]["closed_routes"].values())
    assert result["terminal"] == (
        "UNCLASSIFIED_E4_PL_S3_V6F_REMAINING_PRODUCTION_PARITY"
    )
    assert result["stage4a_scientific_rerun_authorized"] is False


def test_v6f_two_process_outputs_are_byte_identical(tmp_path: Path) -> None:
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                str(AUDIT_PATH),
                "--root",
                str(ROOT),
                "--output",
                str(output),
            ],
            check=True,
            timeout=60,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_v6f_contract_is_canonical_and_fail_closed() -> None:
    raw = CONTRACT_PATH.read_bytes()
    contract = json.loads(raw)
    assert raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert contract["runtime_policy"] == {
        "automatic_retry": False,
        "maximum_seconds": 60,
        "runs": 2,
        "standard_library_only": True,
    }
    assert contract["production_boundary"]["activation_authorized"] is False
    assert contract["production_boundary"]["stage4a_scientific_rerun_authorized"] is False
    assert contract["next_gate_on_open_routes"] == (
        "V6G_V2D_RECOVERY_AND_CURRENT_STATE_EIGEN_PARITY"
    )


def test_v6f_canonical_evidence_binds_the_deterministic_audit() -> None:
    values = {}
    raws = {}
    for name in ("result", "review", "status"):
        raw = (
            REFERENCE / f"e4_pl_s3_v6f_final_parity_audit_{name}.json"
        ).read_bytes()
        value = json.loads(raw)
        assert raw == (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        raws[name] = raw
        values[name] = value
    assert values["result"]["audit"]["open_route_count"] == 3
    assert values["review"]["verdict"] == (
        "ACCEPT_E4_PL_S3_V6F_REMAINING_PARITY_SCOPE"
    )
    assert [item["route"] for item in values["review"]["findings"]] == (
        values["result"]["audit"]["open_routes"]
    )
    status = values["status"]
    for name in ("result", "review"):
        assert status[name] == {
            "bytes": len(raws[name]),
            "sha256": hashlib.sha256(raws[name]).hexdigest().upper(),
        }
    contract_raw = CONTRACT_PATH.read_bytes()
    assert status["contract"] == {
        "bytes": len(contract_raw),
        "sha256": hashlib.sha256(contract_raw).hexdigest().upper(),
    }
    assert status["terminal"] == (
        "UNCLASSIFIED_E4_PL_S3_V6F_REMAINING_PRODUCTION_PARITY"
    )
    assert status["activation_authorized"] is False
    assert status["stage4a_scientific_rerun_authorized"] is False
