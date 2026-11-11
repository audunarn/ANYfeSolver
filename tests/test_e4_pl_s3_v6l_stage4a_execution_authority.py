from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
INCIDENT = REFERENCE / "e4_pl_s3_v6k_dependency_closure_incident.json"
CONTRACT = REFERENCE / "e4_pl_s3_v6l_stage4a_execution_contract.json"
PREDECESSOR = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_graph.py"
SUCCESSOR = REFERENCE / "e4_pl_s3_v6l_stage4a_execution_graph.py"
V6K_ROOT = Path(
    r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease"
    r"\s3-v2d-stage4a-v6k-2d91bba2"
)
CANDIDATE_ARCHIVE = V6K_ROOT / "candidate-source.tar"
V6L_ROOT = Path(
    r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease"
    r"\s3-v2d-stage4a-v6l-dependency-closure"
)
SUPPORT_ARCHIVE = V6L_ROOT / "anyfileio-source.tar"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return value


def test_v6k_incident_is_exact_and_precedes_scientific_evaluation() -> None:
    incident = _canonical(INCIDENT)
    assert incident["terminal"] == "BLOCKED_E4_PL_S3_V6K_PROCESS_OR_EVIDENCE"
    assert incident["scientific_disposition"] == {
        "candidate_evaluated": False,
        "case_coverage_changed": False,
        "mechanics_changed": False,
        "protocol_changed": False,
        "scientific_records_created": 0,
        "thresholds_changed": False,
    }
    assert incident["cancellation"] == {
        "cancelled_not_run_request_count": 26,
        "consumed_request_count": 1,
        "request_reuse_forbidden": True,
        "unstarted_request_count": 26,
    }
    for binding in incident["external_evidence"]:
        path = V6K_ROOT / binding["path"]
        raw = path.read_bytes()
        assert len(raw) == binding["bytes"]
        assert hashlib.sha256(raw).hexdigest().upper() == binding["sha256"]


def test_v6l_contract_and_support_archive_are_exact() -> None:
    contract = _canonical(CONTRACT)
    raw = SUPPORT_ARCHIVE.read_bytes()
    assert len(raw) == contract["dependency_archive"]["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == contract["dependency_archive"]["sha256"]
    assert contract["scientific_invariance"] == {
        "case_count": 81,
        "case_order_unchanged": True,
        "mechanics_unchanged": True,
        "plan_hash_unchanged": True,
        "protocol_unchanged": True,
        "thresholds_unchanged": True,
        "wave_count": 27,
    }
    assert contract["production_boundary"]["activation_authorized"] is False


def _normalized_catalog(graph: dict[str, object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for leaf in graph["leaf_catalog"]:
        assignment = copy.deepcopy(leaf["assignment"])
        assignment.pop("producer_program_sha256")
        normalized.append(assignment)
    return normalized


def test_v6l_graph_changes_only_program_and_dependency_closure() -> None:
    predecessor = _module(PREDECESSOR, "v6l_predecessor")
    successor = _module(SUCCESSOR, "v6l_successor")
    old = predecessor.build_graph(CANDIDATE_ARCHIVE)
    new = successor.build_graph(CANDIDATE_ARCHIVE)
    assert old["plan"] == new["plan"]
    assert _normalized_catalog(old) == _normalized_catalog(new)
    assert [wave["sequence_key"] for wave in old["waves"]] == [
        wave["sequence_key"] for wave in new["waves"]
    ]
    assert [
        [worker["record_id"] for worker in wave["workers"]] for wave in old["waves"]
    ] == [
        [worker["record_id"] for worker in wave["workers"]] for wave in new["waves"]
    ]
    assert all(len(wave["workers"]) == 3 for wave in new["waves"])
    assert new["runtime_policy"]["maximum_concurrent_workers"] == 2
    successor.validate_graph(new)
    assert successor.canonical_bytes(new) == successor.canonical_bytes(
        successor.build_graph(CANDIDATE_ARCHIVE)
    )


def test_v6l_disposable_leaf_closes_the_isolated_import(tmp_path: Path) -> None:
    successor = _module(SUCCESSOR, "v6l_disposable_leaf")
    graph = successor.build_graph(CANDIDATE_ARCHIVE)
    plan_path = tmp_path / "stage4a-plan.json"
    plan_path.write_bytes(successor.canonical_bytes(graph["plan"]))
    leaf = graph["leaf_catalog"][0]
    output = tmp_path / "scientific.json"
    progress = tmp_path / "progress.jsonl"
    candidate_source = tmp_path / "candidate-source"
    result = successor.run_flat_leaf(
        [
            "--run-flat-leaf",
            str(plan_path),
            "--leaf-assignment-sha256",
            leaf["leaf_assignment_sha256"],
            "--selector",
            "e4-pl-s3-v2",
            "--candidate-source-root",
            str(candidate_source),
            "--candidate-archive",
            str(CANDIDATE_ARCHIVE),
            "--candidate-archive-sha256",
            successor.CANDIDATE_ARCHIVE_SHA256,
            "--candidate-commit",
            successor.CANDIDATE_COMMIT,
            "--candidate-tree",
            successor.CANDIDATE_TREE,
            "--producer-program-sha256",
            successor._program_sha256(),
            "--output",
            str(output),
            "--progress",
            str(progress),
        ]
    )
    assert result == 0
    scientific = json.loads(output.read_bytes())
    assert scientific["schema"] == "anysolver.e4-pl-s3-v2-stage4a-leaf-scientific-v3"
    assert scientific["record_count"] == 1
    assert scientific["scientific_payload"]["record"]["record_id"] == scientific["record_ids"][0]
    assert (tmp_path / "support-source/src/anyfileio/calculix/__init__.py").is_file()


def test_v6l_rejects_support_mutation_and_preserves_defaults(tmp_path: Path) -> None:
    successor = _module(SUCCESSOR, "v6l_mutation")
    original = successor.SUPPORT_ARCHIVE
    changed = tmp_path / "changed.tar"
    changed.write_bytes(original.read_bytes() + b"x")
    successor.SUPPORT_ARCHIVE = changed
    with pytest.raises(successor.V6LError, match="identity"):
        successor.verify_support_archive()
    tree = ast.parse(SUCCESSOR.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__", "argparse", "copy", "hashlib", "importlib", "json",
        "os", "pathlib", "shutil", "stat", "sys", "tarfile", "types", "typing",
    }
    elements = ast.parse((ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    for node in elements.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in targets:
                if isinstance(target, ast.Name) and target.id.startswith("DEFAULT_"):
                    defaults[target.id] = node.value.value
    assert defaults["DEFAULT_Q4_FORMULATION"] == "e4-pl"
    assert defaults["DEFAULT_S3_FORMULATION"] == "legacy-s3"
