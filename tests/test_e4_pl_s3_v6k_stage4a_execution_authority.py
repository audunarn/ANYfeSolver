from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
INCIDENT = REFERENCE / "e4_pl_s3_v6j_resource_deferred_incident.json"
CONTRACT = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_contract.json"
PREDECESSOR = REFERENCE / "e4_pl_s3_v6i_stage4a_execution_graph.py"
SUCCESSOR = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_graph.py"
CANDIDATE_COMMIT = "c6e596c64321225e36aaff02b98ddb8fa81b6620"
FORMAL_V6J_ROOT = Path(
    r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease\s3-v2d-stage4a-v6j-8538d3cd"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def candidate_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("v6k-candidate") / "candidate-source.tar"
    subprocess.run(
        ["git", "-c", "core.hooksPath=NUL", "-c", "core.attributesFile=NUL",
         "archive", "--format=tar", f"--output={path}", CANDIDATE_COMMIT],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=60,
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == (
        "AC1EA6D71A273355439B25F915EE7BB383DC60769F22EEA8CC84095CDDAF426F"
    )
    return path


def test_v6j_resource_deferred_incident_is_exact_and_non_scientific() -> None:
    raw = INCIDENT.read_bytes()
    incident = json.loads(raw)
    assert raw == (
        json.dumps(incident, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert incident["terminal"] == "BLOCKED_E4_PL_S3_V6J_PROCESS_OR_EVIDENCE"
    assert incident["incident_classification"] == "RESOURCE_ADMISSION_ONLY_NO_SCIENTIFIC_WORKER_LAUNCHED"
    assert incident["scientific_disposition"]["candidate_evaluated"] is False
    assert incident["cancellation"] == {
        "cancelled_not_run_request_count": 26,
        "consumed_request_count": 1,
        "request_reuse_forbidden": True,
        "unstarted_request_count": 26,
    }
    bounded = json.loads((FORMAL_V6J_ROOT / "wave-01/bounded-result.json").read_bytes())
    assert bounded["terminal"] == "RESOURCE_DEFERRED"
    assert bounded["workers"] == []
    for binding in incident["external_evidence"]:
        path = FORMAL_V6J_ROOT / binding["path"]
        payload = path.read_bytes()
        assert len(payload) == binding["bytes"]
        assert hashlib.sha256(payload).hexdigest().upper() == binding["sha256"]


def test_v6k_contract_is_canonical_and_reduces_only_concurrency() -> None:
    raw = CONTRACT.read_bytes()
    contract = json.loads(raw)
    assert raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert contract["runtime_policy"] == {
        "automatic_retry": False,
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "maximum_concurrent_workers": 2,
        "memory_limit_gib_per_process_tree": 24,
        "os_headroom_gib": 16,
        "registered_workers_per_wave": 3,
        "required_admission_gib": 64,
        "threads_per_worker": 1,
        "wave_count": 27,
    }
    assert contract["correction"]["mechanics_changed"] is False
    assert contract["correction"]["case_coverage_changed"] is False
    assert contract["correction"]["v6j_consumed_request_reused"] is False


def _normalized_catalog(graph: dict[str, object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for leaf in graph["leaf_catalog"]:
        assignment = copy.deepcopy(leaf["assignment"])
        assignment.pop("producer_program_sha256")
        normalized.append(assignment)
    return normalized


def test_v6k_graph_preserves_plan_leaves_and_three_worker_waves(
    candidate_archive: Path,
) -> None:
    predecessor = _module(PREDECESSOR, "v6k_predecessor")
    successor = _module(SUCCESSOR, "v6k_successor")
    old = predecessor.build_graph(candidate_archive)
    new = successor.build_graph(candidate_archive)
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
    assert new["runtime_policy"]["memory_limit_gib_per_process_tree"] == 24
    successor.validate_graph(new)


def test_v6k_graph_is_deterministic_and_rejects_policy_mutation(
    candidate_archive: Path,
) -> None:
    successor = _module(SUCCESSOR, "v6k_determinism")
    first = successor.canonical_bytes(successor.build_graph(candidate_archive))
    second = successor.canonical_bytes(successor.build_graph(candidate_archive))
    assert first == second
    changed = json.loads(first)
    changed["runtime_policy"]["maximum_concurrent_workers"] = 3
    with pytest.raises(successor.V6KError, match="concurrency"):
        successor.validate_graph(changed)


def test_v6k_program_is_standard_library_only_and_nonactivating() -> None:
    tree = ast.parse(SUCCESSOR.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__", "argparse", "copy", "hashlib", "importlib", "pathlib",
        "stat", "sys", "types", "typing",
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
