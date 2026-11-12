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
GRAPH = REFERENCE / "e4_pl_s3_v6i_stage4a_execution_graph.py"
PUBLISHER = REFERENCE / "e4_pl_s3_v6i_stage4a_request_publisher.py"
CONTRACT = REFERENCE / "e4_pl_s3_v6i_stage4a_execution_contract.json"
CANDIDATE_COMMIT = "c6e596c64321225e36aaff02b98ddb8fa81b6620"
CANDIDATE_ARCHIVE_SHA256 = (
    "AC1EA6D71A273355439B25F915EE7BB383DC60769F22EEA8CC84095CDDAF426F"
)


def _default_formulations() -> dict[str, str]:
    tree = ast.parse((ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8"))
    made: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {
                "DEFAULT_Q4_FORMULATION", "DEFAULT_S3_FORMULATION"
            }:
                made[target.id] = value.value
    return made


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def candidate_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("v6i-candidate") / "candidate-source.tar"
    subprocess.run(
        [
            "git", "-c", "core.hooksPath=NUL", "-c", "core.attributesFile=NUL",
            "archive", "--format=tar", f"--output={path}", CANDIDATE_COMMIT,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=60,
    )
    raw = path.read_bytes()
    assert len(raw) == 29_767_680
    assert hashlib.sha256(raw).hexdigest().upper() == CANDIDATE_ARCHIVE_SHA256
    return path


def test_v6i_contract_is_canonical_and_nonexecuting() -> None:
    raw = CONTRACT.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert value["runtime_policy"] == {
        "automatic_retry": False,
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "leaf_count": 81,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "numerical_library_threads_per_worker": 1,
        "wave_count": 27,
        "workers_per_wave": 3,
    }
    assert value["request_policy"]["ledger_mutation_authorized"] is False
    assert value["request_policy"]["request_execution_authorized"] is False
    assert value["production_boundary"]["stage4a_execution_authorized"] is False
    assert value["production_boundary"]["activation_authorized"] is False
    for binding in value["implementation"].values():
        raw_program = (ROOT / binding["path"]).read_bytes()
        assert len(raw_program) == binding["bytes"]
        assert hashlib.sha256(raw_program).hexdigest().upper() == binding["sha256"]
    assert _default_formulations() == {
        "DEFAULT_Q4_FORMULATION": "e4-pl",
        "DEFAULT_S3_FORMULATION": "legacy-s3",
    }


@pytest.mark.parametrize("path", [GRAPH, PUBLISHER])
def test_v6i_process_programs_are_standard_library_only(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__", "argparse", "hashlib", "importlib", "json", "math", "os",
        "pathlib", "shutil", "stat", "subprocess", "sys", "tarfile", "types",
        "typing",
    }


def test_v6i_graph_is_byte_identical_and_exactly_covers_81_leaves(
    candidate_archive: Path,
) -> None:
    graph = _module(GRAPH, "v6i_graph_determinism")
    first = graph.canonical_bytes(graph.build_graph(candidate_archive))
    second = graph.canonical_bytes(graph.build_graph(candidate_archive))
    assert first == second
    value = json.loads(first)
    assert value["leaf_count"] == 81
    assert value["wave_count"] == 27
    assert len(value["leaf_catalog"]) == 81
    assert len(value["waves"]) == 27
    hashes: list[str] = []
    for index, wave in enumerate(value["waves"]):
        assert wave["wave_index"] == index
        assert [worker["diagonal"] for worker in wave["workers"]] == [
            "slash", "backslash", "alternating"
        ]
        hashes.extend(worker["leaf_assignment_sha256"] for worker in wave["workers"])
    assert len(hashes) == len(set(hashes)) == 81
    assert value["runtime_policy"]["child_wall_seconds"] == 600
    assert value["runtime_policy"]["complete_wave_wall_seconds"] == 1800
    assert value["stage4a_execution_authorized"] is False


def test_v6i_graph_rejects_leaf_policy_and_archive_mutations(
    candidate_archive: Path, tmp_path: Path
) -> None:
    graph = _module(GRAPH, "v6i_graph_mutations")
    value = graph.build_graph(candidate_archive)
    changed = copy.deepcopy(value)
    changed["waves"][0]["workers"][0]["leaf_assignment_sha256"] = "0" * 64
    with pytest.raises(graph.V6IError, match="coverage"):
        graph.validate_graph(changed)
    changed = copy.deepcopy(value)
    changed["runtime_policy"]["child_wall_seconds"] = 601
    with pytest.raises(graph.V6IError, match="policy"):
        graph.validate_graph(changed)
    altered_archive = tmp_path / "altered.tar"
    altered_archive.write_bytes(candidate_archive.read_bytes() + b"X")
    with pytest.raises(graph.V6IError, match="archive identity"):
        graph.verify_archive(altered_archive)


def test_v6i_request_preview_is_deterministic_but_cannot_publish(
    candidate_archive: Path, tmp_path: Path
) -> None:
    graph = _module(GRAPH, "v6i_graph_preview")
    publisher = _module(PUBLISHER, "v6i_publisher_preview")
    graph_path = tmp_path / "graph.json"
    graph_path.write_bytes(graph.canonical_bytes(graph.build_graph(candidate_archive)))
    missing_authority = tmp_path / "missing-successor-authorization.json"
    request_root = tmp_path / "requests"
    qualification_root = tmp_path / "qualification"
    first = publisher.canonical_bytes(
        publisher.build_preview(
            graph_path, candidate_archive, qualification_root,
            missing_authority, request_root,
        )
    )
    second = publisher.canonical_bytes(
        publisher.build_preview(
            graph_path, candidate_archive, qualification_root,
            missing_authority, request_root,
        )
    )
    assert first == second
    value = json.loads(first)
    assert value["request_count"] == 27
    assert value["publication_authorized"] is False
    assert value["stage4a_execution_authorized"] is False
    assert all("--run-registered-wave" in row["command"] for row in value["requests"])
    with pytest.raises(FileNotFoundError):
        publisher.publish_authorized(graph_path, missing_authority, request_root)
    assert not request_root.exists()
    assert not qualification_root.exists()

    request_root.mkdir()
    authority_path = tmp_path / "disposable-v6j-authorization.json"
    rows = []
    for wave_index in range(graph.WAVE_COUNT):
        request_id = f"{wave_index + 1:032x}"
        request_path = request_root / f"{request_id}.json"
        wave_root = qualification_root / f"wave-{wave_index + 1:02d}"
        result_path = wave_root / "wave-wrapper-result.json"
        request = {
            "command": graph.registered_command(
                graph_path=graph_path,
                wave_index=wave_index,
                candidate_archive=candidate_archive,
                output_root=wave_root,
                authorization_path=authority_path,
                request_path=request_path,
                result_path=result_path,
            ),
            "estimate_minutes": 30,
            "repository": str(ROOT),
            "request_id": request_id,
            "requested_at": "2099-01-01T00:00:00+00:00",
            "status": "PENDING",
            "task": f"ANYfeSolver S3 V2D Stage 4A bounded wave {wave_index + 1:02d}",
        }
        raw_request = publisher.canonical_bytes(request)
        rows.append(
            {
                "request": request,
                "request_sha256": hashlib.sha256(raw_request).hexdigest().upper(),
                "wave_index": wave_index,
            }
        )
    authority_path.write_bytes(
        publisher.canonical_bytes(
            {
                "activation_authorized": False,
                "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest().upper(),
                "requests": rows,
                "schema": graph.AUTHORIZATION_SCHEMA,
                "stage4a_execution_authorized": True,
            }
        )
    )
    receipt = publisher.publish_authorized(graph_path, authority_path, request_root)
    assert receipt["request_count"] == 27
    assert len(list(request_root.glob("*.json"))) == 27
    with pytest.raises(FileExistsError):
        publisher.publish_authorized(graph_path, authority_path, request_root)


def test_v6i_registered_execution_fails_before_outputs_without_v6j_authority(
    candidate_archive: Path, tmp_path: Path
) -> None:
    graph = _module(GRAPH, "v6i_graph_fail_closed")
    graph_path = tmp_path / "graph.json"
    graph_path.write_bytes(graph.canonical_bytes(graph.build_graph(candidate_archive)))
    request_id = "1" * 32
    request_path = tmp_path / f"{request_id}.json"
    request_path.write_bytes(
        graph.canonical_bytes(
            {
                "command": "NOT_AUTHORIZED",
                "estimate_minutes": 30,
                "repository": str(ROOT),
                "request_id": request_id,
                "requested_at": "FROZEN_BY_SUCCESSOR_AUTHORITY",
                "status": "PENDING",
                "task": "ANYfeSolver S3 V2D Stage 4A bounded wave 01",
            }
        )
    )
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(FileNotFoundError):
        graph.run_registered_wave(
            graph_path, 0, candidate_archive, output_root,
            tmp_path / "missing-v6j-authority.json", request_path,
            output_root / "result.json",
        )
    assert not output_root.exists()


def test_v6i_authority_extent_has_no_production_paths() -> None:
    contract = json.loads(CONTRACT.read_bytes())
    assert contract["authority_commit"]["expected_paths"] == sorted(
        contract["authority_commit"]["expected_paths"]
    )
    assert all(
        path.startswith("docs/reference_cases/") or path.startswith("tests/")
        for path in contract["authority_commit"]["expected_paths"]
    )
    assert all(not path.startswith("src/") for path in contract["authority_commit"]["expected_paths"])
