from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v6k_stage4a_request_contract.json"
AUTHORITY = REFERENCE / "e4_pl_s3_v6k_stage4a_authority.py"
EXECUTOR = REFERENCE / "e4_pl_s3_v6k_stage4a_serial_executor.py"
V6J_AUTHORIZATION = REFERENCE / "e4_pl_s3_v6j_stage4a_execution_authorization.json"
FORMAL_ROOT = Path(
    r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease\s3-v2d-stage4a-v6k-2d91bba2"
)
GRAPH = FORMAL_ROOT / "execution-graph.json"
ARCHIVE = FORMAL_ROOT / "candidate-source.tar"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v6k_request_contract_is_canonical_and_bounded() -> None:
    raw = CONTRACT.read_bytes()
    contract = json.loads(raw)
    assert raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert contract["execution_graph"]["registered_workers_per_wave"] == 3
    assert contract["execution_graph"]["maximum_concurrent_workers"] == 2
    assert contract["runtime_policy"]["automatic_retry"] is False
    assert contract["request_authority"]["predecessor_ids_reused"] is False
    assert contract["production_boundary"]["activation_authorized"] is False


@pytest.mark.parametrize("path", [AUTHORITY, EXECUTOR])
def test_v6k_request_programs_are_standard_library_only(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__", "argparse", "hashlib", "importlib", "json", "os",
        "pathlib", "sys", "types", "typing",
    }


def test_v6k_fresh_authorization_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    authority = _module(AUTHORITY, "v6k_request_authority")
    authorization_path = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_authorization.json"
    request_root = Path(r"C:\Github\.resource-manager\requests")
    first = authority.canonical_bytes(
        authority.generate(GRAPH, ARCHIVE, FORMAL_ROOT, authorization_path, request_root)
    )
    second = authority.canonical_bytes(
        authority.generate(GRAPH, ARCHIVE, FORMAL_ROOT, authorization_path, request_root)
    )
    assert first == second
    value = json.loads(first)
    old = json.loads(V6J_AUTHORIZATION.read_bytes())
    new_ids = {row["request"]["request_id"] for row in value["requests"]}
    old_ids = {row["request"]["request_id"] for row in old["requests"]}
    assert len(new_ids) == 27
    assert not (new_ids & old_ids)
    assert value["graph_sha256"] == "2D91BBA2D88B6D1D16A308EFF67AC73705B0E3C988521155C4E2B67BB68228B6"
    assert all("e4_pl_s3_v6k_stage4a_execution_graph.py" in row["request"]["command"] for row in value["requests"])
    assert all(row["request_sha256"] == hashlib.sha256(authority.canonical_bytes(row["request"])).hexdigest().upper() for row in value["requests"])


def test_v6k_executor_accepts_disposable_successor_authority(tmp_path: Path) -> None:
    authority = _module(AUTHORITY, "v6k_executor_authority")
    executor = _module(EXECUTOR, "v6k_executor_validation")
    authorization_path = REFERENCE / "e4_pl_s3_v6k_stage4a_execution_authorization.json"
    made = authority.generate(
        GRAPH, ARCHIVE, FORMAL_ROOT, authorization_path,
        Path(r"C:\Github\.resource-manager\requests"),
    )
    disposable = tmp_path / "authorization.json"
    # Commands bind the tracked authorization path, while validation binds only
    # the canonical authorization bytes and frozen graph/archive identities.
    disposable.write_bytes(authority.canonical_bytes(made))
    value, raw, graph, graph_raw = executor.validate_authorization(
        disposable, GRAPH, ARCHIVE
    )
    assert len(value["requests"]) == 27
    assert value["activation_authorized"] is False
    assert graph["runtime_policy"]["maximum_concurrent_workers"] == 2
    assert len(raw) > 40_000 and len(graph_raw) == 155_579


def test_v6k_extent_and_defaults_remain_nonproduction() -> None:
    contract = json.loads(CONTRACT.read_bytes())
    assert all(
        path.startswith("docs/reference_cases/") or path.startswith("tests/")
        for path in contract["authority_commit"]["expected_paths"]
    )
    tree = ast.parse((ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in targets:
                if isinstance(target, ast.Name) and target.id.startswith("DEFAULT_"):
                    defaults[target.id] = node.value.value
    assert defaults["DEFAULT_Q4_FORMULATION"] == "e4-pl"
    assert defaults["DEFAULT_S3_FORMULATION"] == "legacy-s3"
