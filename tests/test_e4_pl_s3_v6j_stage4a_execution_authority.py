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
CONTRACT = REFERENCE / "e4_pl_s3_v6j_stage4a_execution_contract.json"
AUTHORITY = REFERENCE / "e4_pl_s3_v6j_stage4a_authority.py"
EXECUTOR = REFERENCE / "e4_pl_s3_v6j_stage4a_serial_executor.py"
GRAPH = REFERENCE / "e4_pl_s3_v6i_stage4a_execution_graph.py"
CANDIDATE_COMMIT = "c6e596c64321225e36aaff02b98ddb8fa81b6620"
ARCHIVE_SHA256 = "AC1EA6D71A273355439B25F915EE7BB383DC60769F22EEA8CC84095CDDAF426F"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def candidate_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("v6j-candidate") / "candidate-source.tar"
    subprocess.run(
        ["git", "-c", "core.hooksPath=NUL", "-c", "core.attributesFile=NUL",
         "archive", "--format=tar", f"--output={path}", CANDIDATE_COMMIT],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=60,
    )
    raw = path.read_bytes()
    assert len(raw) == 29_767_680
    assert hashlib.sha256(raw).hexdigest().upper() == ARCHIVE_SHA256
    return path


def _graph(candidate_archive: Path, path: Path):
    graph = _module(GRAPH, f"v6j_graph_{path.stem}")
    path.write_bytes(graph.canonical_bytes(graph.build_graph(candidate_archive)))
    return graph


def test_v6j_contract_is_canonical_and_bounded() -> None:
    raw = CONTRACT.read_bytes()
    contract = json.loads(raw)
    assert raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert contract["runtime_policy"] == {
        "automatic_retry": False,
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "outer_executor_wall_seconds": 1830,
        "serial_request_execution": True,
        "threads_per_worker": 1,
    }
    assert contract["production_boundary"]["activation_authorized"] is False
    assert contract["production_boundary"]["anymesh_untouched"] is True


@pytest.mark.parametrize("path", [AUTHORITY, EXECUTOR])
def test_v6j_authority_programs_are_standard_library_only(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__", "argparse", "datetime", "hashlib", "importlib", "json",
        "math", "os", "pathlib", "re", "stat", "subprocess", "sys", "types",
        "typing",
    }


def test_v6j_authorization_is_deterministic_and_binds_27_exact_commands(
    candidate_archive: Path, tmp_path: Path
) -> None:
    graph_path = tmp_path / "graph.json"
    graph = _graph(candidate_archive, graph_path)
    authority = _module(AUTHORITY, "v6j_authority_determinism")
    authorization_path = REFERENCE / "e4_pl_s3_v6j_stage4a_execution_authorization.json"
    qualification_root = Path(
        r"C:\Users\AudunArnesenNyhus\AppData\Local\ANYrelease\s3-v2d-stage4a-v6j-8538d3cd"
    )
    request_root = Path(r"C:\Github\.resource-manager\requests")
    first = authority.canonical_bytes(
        authority.generate(
            graph_path, candidate_archive, qualification_root,
            authorization_path, request_root,
        )
    )
    second = authority.canonical_bytes(
        authority.generate(
            graph_path, candidate_archive, qualification_root,
            authorization_path, request_root,
        )
    )
    assert first == second
    value = json.loads(first)
    assert value["stage4a_execution_authorized"] is True
    assert value["activation_authorized"] is False
    assert len(value["requests"]) == 27
    ids = [row["request"]["request_id"] for row in value["requests"]]
    assert len(ids) == len(set(ids)) == 27
    for index, row in enumerate(value["requests"]):
        request = row["request"]
        assert row["wave_index"] == index
        assert request["request_id"] == authority.request_id(index)
        assert request["estimate_minutes"] == 30
        assert "--run-registered-wave" in request["command"]
        assert f"'--wave-index' '{index}'" in request["command"]
        assert row["request_sha256"] == hashlib.sha256(
            authority.canonical_bytes(request)
        ).hexdigest().upper()
    assert graph_path.read_bytes() == graph.canonical_bytes(json.loads(graph_path.read_bytes()))


def test_v6j_authority_rejects_graph_and_archive_mutations(
    candidate_archive: Path, tmp_path: Path
) -> None:
    graph_path = tmp_path / "graph.json"
    _graph(candidate_archive, graph_path)
    authority = _module(AUTHORITY, "v6j_authority_mutation")
    changed_graph = tmp_path / "changed-graph.json"
    value = json.loads(graph_path.read_bytes())
    value["waves"][0]["workers"][0]["leaf_assignment_sha256"] = "0" * 64
    changed_graph.write_bytes(authority.canonical_bytes(value))
    with pytest.raises(Exception):
        authority.generate(
            changed_graph, candidate_archive, tmp_path / "qualification",
            tmp_path / "authorization.json", Path(r"C:\Github\.resource-manager\requests"),
        )
    changed_archive = tmp_path / "changed.tar"
    changed_archive.write_bytes(candidate_archive.read_bytes() + b"X")
    with pytest.raises(Exception):
        authority.generate(
            graph_path, changed_archive, tmp_path / "qualification",
            tmp_path / "authorization.json", Path(r"C:\Github\.resource-manager\requests"),
        )


def test_v6j_executor_has_no_retry_and_validates_before_resource_mutation() -> None:
    tree = ast.parse(EXECUTOR.read_text(encoding="utf-8"))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("run_wave") == 1
    text = EXECUTOR.read_text(encoding="utf-8")
    assert "OUTER_WALL_SECONDS = 1_830" in text
    assert "taskkill.exe" in text
    assert "request was already consumed" in text
    assert "no automatic retry" in text


def test_v6j_preserves_production_defaults_and_extent() -> None:
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
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("DEFAULT_"):
                defaults[target.id] = node.value.value
    assert defaults["DEFAULT_Q4_FORMULATION"] == "e4-pl"
    assert defaults["DEFAULT_S3_FORMULATION"] == "legacy-s3"
