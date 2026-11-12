from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from anyfesolver.elements import DEFAULT_Q4_FORMULATION, DEFAULT_S3_FORMULATION


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v6h_stage4a_authority_contract.json"
ADAPTER = REFERENCE / "e4_pl_s3_v6h_stage4a_adapter.py"
AUTHORITY = REFERENCE / "e4_pl_s3_v6h_stage4a_authority.py"
MANIFEST = REFERENCE / "e4_pl_s3_mixed_mesh_connectivity_manifest.json"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v6h_contract_is_canonical_and_nonexecuting() -> None:
    raw = CONTRACT.read_bytes()
    contract = json.loads(raw)
    assert raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert contract["production_boundary"]["stage4a_execution_authorized"] is False
    assert contract["production_boundary"]["activation_authorized"] is False
    assert contract["scientific_schema_compatibility"]["thresholds_changed"] is False
    assert contract["scientific_schema_compatibility"][
        "topology_or_case_coverage_changed"
    ] is False
    assert contract["runtime_policy"]["child_wall_seconds"] == 600
    assert contract["runtime_policy"]["complete_wave_wall_seconds"] == 1800
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"


def test_v6h_authority_generator_is_standard_library_only() -> None:
    tree = ast.parse(AUTHORITY.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "subprocess",
        "typing",
    }


def _one_percent_record() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_bytes())
    for record in manifest["records"]:
        if (
            record["level"] == 20
            and record["s3_area_fraction_percent"] == 1
            and record["mask"] == "dispersed"
            and record["diagonal"] == "slash"
        ):
            return record
    raise AssertionError("missing frozen N20 1% dispersed slash record")


def test_v6h_adapter_builds_exact_v2d_and_restores_public_factory() -> None:
    import anyfesolver.elements as element_factory

    adapter = _module(ADAPTER, "v6h_adapter_build")
    original = element_factory.create_shell_element
    model, kinds, counts, combined = adapter.build_model_for_validation(
        _one_percent_record()
    )
    assert element_factory.create_shell_element is original
    assert counts == {"Q4": 396, "S3": 8}
    assert combined["v2a_s3"] == 8
    s3_elements = [
        element
        for element_id, element in model.mesh.elements.items()
        if kinds[element_id] == "S3"
    ]
    assert len(s3_elements) == 8
    assert {
        (type(element).__name__, element.formulation_id, element.implementation_id)
        for element in s3_elements
    } == {
        (
            "NativeParityE4PLS3V2DShellElement",
            "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
            "E4_PL_S3_V2D_RECOVERY_CURRENT_EIGEN_GATE_V1",
        )
    }


def test_v6h_adapter_restores_factory_when_model_build_fails() -> None:
    import anyfesolver.elements as element_factory

    adapter = _module(ADAPTER, "v6h_adapter_failure")
    original = element_factory.create_shell_element
    changed = dict(_one_percent_record())
    changed["connectivity_sha256"] = "0" * 64
    with pytest.raises(Exception, match="connectivity"):
        adapter.build_model_for_validation(changed)
    assert element_factory.create_shell_element is original


def test_v6h_adapter_emits_byte_identical_bounded_n20_science(tmp_path: Path) -> None:
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                str(ADAPTER),
                "--emit-disposable-validation",
                str(output),
            ],
            check=True,
            timeout=600,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    document = json.loads(outputs[0].read_bytes())
    assert document["candidate_formulation_id"] == (
        "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
    )
    record = document["record"]
    assert record["classification"] == "CLASSIFYING_Q4_V2A_PRODUCTION_MECHANICS"
    assert record["formulation_counts"] == {
        "qualified_q4": 396,
        "v1_s3": 0,
        "v2a_s3": 8,
    }
    assert record["solver"]["status"] == "CONVERGED_DIRECT_SPARSE"
    assert math.isfinite(record["response"]["relative_error"])
    assert math.isfinite(record["energy_norm"]["relative"])


def test_v6h_frozen_authority_is_byte_identical_in_two_processes(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_bytes())
    frozen_authority_path = (
        REFERENCE / "e4_pl_s3_v6h_stage4a_preparation_authority.json"
    )
    if frozen_authority_path.exists():
        head = json.loads(frozen_authority_path.read_bytes())["authority_commit"][
            "commit"
        ]
    else:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if head == contract["authority_commit"]["expected_parent"]:
        pytest.skip("authority commit has not yet been frozen")
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                str(AUTHORITY),
                "--root",
                str(ROOT),
                "--contract",
                str(CONTRACT),
                "--authority-commit",
                head,
                "--output",
                str(output),
            ],
            check=True,
            timeout=60,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    value = json.loads(outputs[0].read_bytes())
    assert value["authority_commit"]["commit"] == head
    assert value["stage4a_preparation_authorized"] is True
    assert value["stage4a_execution_authorized"] is False
    assert value["activation_authorized"] is False
    assert hashlib.sha256(outputs[0].read_bytes()).hexdigest()
