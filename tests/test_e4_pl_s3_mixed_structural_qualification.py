from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "docs" / "reference_cases"
GITHUB_ROOT = Path(os.environ["ANY_ECOSYSTEM_ROOT"]) if "ANY_ECOSYSTEM_ROOT" in os.environ else next(
    (parent for parent in ROOT.parents if (parent / "ANYgeometry" / "src").is_dir()),
    ROOT.parent,
)
INPUT = REFERENCE_CASES / "e4_pl_s3_mixed_structural_input.json"
SCHEMA = REFERENCE_CASES / "e4_pl_s3_mixed_structural_input_schema.json"
PRODUCER = REFERENCE_CASES / "e4_pl_s3_mixed_structural_producer.py"
COORDINATOR = REFERENCE_CASES / "e4_pl_s3_mixed_structural_coordinator.py"


def _source_paths() -> list[Path]:
    candidates = [
        ROOT / "src",
        GITHUB_ROOT / "ANYgeometry" / "src",
        GITHUB_ROOT / "ANYmaterial" / "src",
        GITHUB_ROOT / "ANYmesh" / "src",
        GITHUB_ROOT / "ANYfileIO" / "src",
        REFERENCE_CASES,
    ]
    return [path for path in candidates if path.is_dir()]


for _path in reversed(_source_paths()):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


common = importlib.import_module("e4_pl_s3_mixed_structural_common")
producer = importlib.import_module("e4_pl_s3_mixed_structural_producer")
coordinator = importlib.import_module("e4_pl_s3_mixed_structural_coordinator")
bounded = importlib.import_module("e4_pl_q1w_bounded_runner")


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    pieces = [str(path) for path in _source_paths()]
    inherited = environment.get("PYTHONPATH", "")
    if inherited:
        pieces.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(pieces)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


@pytest.fixture(scope="module")
def authorities() -> Any:
    return common.load_authorities(INPUT)


@pytest.fixture(scope="module")
def deterministic_quick_cycles(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, dict[str, Any], Path]:
    root = tmp_path_factory.mktemp("mixed_structural_cycles")
    command = [
        sys.executable,
        str(COORDINATOR),
        "--run-bounded-structural",
        "--repository-root",
        str(ROOT),
        "--input",
        str(INPUT),
        "--producer",
        str(PRODUCER),
        "--output-directory",
        "diagnostics",
        "--aggregate",
        "aggregate.json",
        "--quick-smoke",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    raw = (root / "aggregate.json").read_bytes()
    return raw, json.loads(raw), root


def test_input_schema_candidate_and_frozen_authorities_are_exact(authorities: Any) -> None:
    assert authorities.input_raw == common.pretty_canonical_bytes(authorities.input)
    assert authorities.input["candidate"] == {
        "commit": "4e5b5976d4286ffd0cda5b8424d154132f3f8da0",
        "parent": "7d85ecef35daa6ebe11f11536a7ede4d288e0aa3",
        "qualified_q4_formulation_id": "E4_PL_QUALIFIED_Q4_HYBRID_V2",
        "qualified_s3_formulation_id": "E4_PL_QUALIFIED_S3_COMPANION_V1",
        "subject": "perf: amortize immutable S3 matrix validation",
        "tree": "96f60dcdd61a78111091ce4f93d7170cf7d0878a",
    }
    schema_raw, schema = common.read_canonical(SCHEMA, pretty=True, label="schema")
    assert schema["$id"] == common.INPUT_SCHEMA
    assert schema["properties"]["execution"]["const"] == authorities.input["execution"]
    assert schema["properties"]["candidate"]["properties"]["commit"]["const"] == (
        authorities.input["candidate"]["commit"]
    )
    assert schema["properties"]["authority"]["properties"]["programs"][
        "additionalProperties"
    ] is False
    coverage_schema = schema["properties"]["coverage"]["properties"]
    assert coverage_schema["convergence_reference"]["additionalProperties"] is False
    assert coverage_schema["locking_fixture"]["additionalProperties"] is False
    assert coverage_schema["convergence_sequences"]["minItems"] == 5
    assert coverage_schema["convergence_sequences"]["maxItems"] == 5
    assert hashlib.sha256(schema_raw).hexdigest().upper() == (
        authorities.input["authority"]["input_schema"]["sha256"]
    )
    assert authorities.input["authority"]["programs"] == {
        name: {
            "bytes": len(authorities.program_raw[name]),
            "path": relative,
            "sha256": hashlib.sha256(authorities.program_raw[name]).hexdigest().upper(),
        }
        for name, relative in common.PROGRAM_PATHS.items()
    }
    extent = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--no-renames",
            f"{authorities.input['candidate']['commit']}..HEAD",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert sorted(line.split("\t", 1)[1] for line in extent) == sorted(
        common.ALLOWED_EXECUTION_EXTENT
    )
    assert all(line.startswith("A\t") for line in extent)
    q4 = ROOT / "src" / "anyfesolver" / "e4_pl_element.py"
    assert hashlib.sha256(q4.read_bytes()).hexdigest().upper() == (
        "EE49BAE1C9439C41EC2D61798A8A8B88CBA9081DCAD2DFDC857FE313C6C0D4D1"
    )


def test_all_252_registered_topologies_regenerate_byte_identically(authorities: Any) -> None:
    audit = producer.audit_manifest(authorities)
    assert audit == {
        "coverage_exact": True,
        "gated_record_count": 252,
        "manifest_regeneration_byte_identical": True,
        "observed": {
            "diagonals": ["alternating", "backslash", "slash"],
            "fractions_percent": [0, 1, 5, 10, 25],
            "levels": [20, 40, 80, 160],
            "masks": ["boundary_band", "chain", "compact_cluster", "dispersed", "hole_band"],
        },
        "research_control_record_count": 12,
    }


def test_producer_classifies_membrane_and_bending_patch_mutations(
    authorities: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        producer,
        "audit_manifest",
        lambda _authorities: {
            "coverage_exact": True,
            "manifest_regeneration_byte_identical": True,
        },
    )
    base = {
        "connectivity_sha256": "0" * 64,
        "covariance_residual": 0.0,
        "force_loaded_in_plane": {
            "action_reaction_residual": 0.0,
            "edge_work_residual": 0.0,
            "force_residual": 0.0,
            "moment_residual": 0.0,
            "patch_residual": 0.0,
        },
        "patch_residuals": {"bending": 0.0, "membrane": 0.0, "shear": 0.0},
        "pl_participation": {"Q4_PL": 0.0, "S3_PL": 0.0},
        "q4_residual_hourglass_participation": 0.0,
        "record_id": "MUTATED",
        "symmetry_residual": 0.0,
        "transverse_shear_classification": (
            "NONCLASSIFYING_NOT_THE_PUBLISHED_FORCE_LOADED_PATCH"
        ),
    }
    for component in ("membrane", "bending"):
        row = copy.deepcopy(base)
        row["patch_residuals"][component] = 1.0
        monkeypatch.setattr(
            producer,
            "_patch_basis",
            lambda _authorities, *, quick, row=row: [row],
        )
        payload, status = producer.produce_patch(authorities, quick=False)
        assert status["patch_and_equilibrium"] == common.FAIL
        assert payload["contradictions"] == [f"MUTATED:{component.upper()}_PATCH"]
        assert payload["contradictions_classifying"] is True


def test_independent_mindlin_reference_is_positive_and_converges_to_kirchhoff(
    authorities: Any,
) -> None:
    producer.activate_numerics(authorities)
    thin = producer._mindlin_plate_reference(
        length=1.0,
        width=1.0,
        thickness=1.0e-4,
        pressure=1000.0,
        elastic_modulus=210.0e9,
        poisson_ratio=0.3,
        terms=31,
    )
    classical = 0.0
    rigidity = 210.0e9 * 1.0e-12 / (12.0 * (1.0 - 0.3**2))
    for m in range(1, 32, 2):
        for n in range(1, 32, 2):
            classical += (
                16.0
                * 1000.0
                * math.sin(m * math.pi / 2.0)
                * math.sin(n * math.pi / 2.0)
                / (
                    math.pi**6
                    * rigidity
                    * m
                    * n
                    * (m * m + n * n) ** 2
                )
            )
    assert thin["center_displacement"] > 0.0
    assert thin["strain_energy"] > 0.0
    assert thin["center_displacement"] == pytest.approx(abs(classical), rel=1.0e-7)


def test_two_quick_cycles_overlap_are_byte_identical_and_fail_closed(
    deterministic_quick_cycles: tuple[bytes, dict[str, Any], Path],
) -> None:
    raw, aggregate, root = deterministic_quick_cycles
    assert raw == common.canonical_bytes(aggregate)
    assert aggregate["schema"] == common.AGGREGATE_SCHEMA
    assert aggregate["execution_tier"] == "QUICK_NONCLASSIFYING"
    assert aggregate["authority"]["candidate_commit"] == (
        "4e5b5976d4286ffd0cda5b8424d154132f3f8da0"
    )
    assert aggregate["authority"]["candidate_parent"] == (
        "7d85ecef35daa6ebe11f11536a7ede4d288e0aa3"
    )
    assert aggregate["authority"]["candidate_subject"] == (
        "perf: amortize immutable S3 matrix validation"
    )
    assert aggregate["authority"]["candidate_tree"] == (
        "96f60dcdd61a78111091ce4f93d7170cf7d0878a"
    )
    assert aggregate["authority"]["allowed_changed_paths"] == list(
        common.ALLOWED_EXECUTION_EXTENT
    )
    assert set(aggregate["authority"]["program_sha256"]) == set(common.PROGRAM_PATHS)
    assert aggregate["authority"]["execution_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert aggregate["authority"]["execution_tree"] == subprocess.run(
        ["git", "show", "-s", "--format=%T", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert aggregate["terminal"] == common.TERMINALS[2]
    assert aggregate["gate_status"] == {gate: common.PARTIAL for gate in common.GATE_IDS}
    assert aggregate["cycle_agreement"] == {
        "byte_identical": True,
        "canonical_cycle_count": 2,
        "cycle_payload_sha256": [
            aggregate["cycle_agreement"]["cycle_payload_sha256"][0],
            aggregate["cycle_agreement"]["cycle_payload_sha256"][0],
        ],
        "fresh_distinct_directories": True,
        "schema": "anysolver.e4-pl-s3-mixed-structural-cycle-agreement-v1",
    }
    assert aggregate["worker_overlap_verified"] is True
    assert [row["shard_id"] for row in aggregate["shards"]] == list(common.SHARD_IDS)
    assert all(row["process_status"] == "COMPLETE" for row in aggregate["shards"])
    assert all(row["shard_sha256"] for row in aggregate["shards"])
    for cycle in (1, 2):
        diagnostics = root / "diagnostics" / f"cycle-{cycle}"
        shard_directories = sorted(path for path in diagnostics.iterdir() if path.is_dir())
        assert len(shard_directories) == 3
        assert len({path.resolve() for path in shard_directories}) == 3
        assert all((path / "shard.json").is_file() for path in shard_directories)
        assert all((path / "progress.jsonl").is_file() for path in shard_directories)
        assert all((path / "process.json").is_file() for path in shard_directories)
        process_rows = [json.loads((path / "process.json").read_bytes()) for path in shard_directories]
        assert max(row["started_monotonic_ns"] for row in process_rows) < min(
            row["finished_monotonic_ns"] for row in process_rows
        )
        assert all(row["peak_rss_bytes"] > 0 for row in process_rows)
        assert all(not list(path.glob("*/*/shard.json")) for path in shard_directories)


def test_quick_diagnostics_are_real_but_never_formal_qualification(
    deterministic_quick_cycles: tuple[bytes, dict[str, Any], Path],
) -> None:
    _raw, _aggregate, root = deterministic_quick_cycles
    authorities = common.load_authorities(INPUT)
    shards = {
        value["shard_id"]: common.validate_shard(value, authorities=authorities)
        for path in (root / "diagnostics" / "cycle-1").glob("*/shard.json")
        for value in (json.loads(path.read_bytes()),)
    }
    patch = shards[common.SHARD_IDS[0]]["diagnostic_payload"]
    assert patch["manifest_audit"]["gated_record_count"] == 252
    assert patch["scope"]["transverse_force_loaded_shear_patch"].startswith("UNEXECUTED_")
    assert not patch["contradictions"]
    for row in patch["basis"]:
        assert row["force_loaded_in_plane"]["patch_residual"] < 1.0e-10
        assert row["patch_residuals"]["membrane"] < 1.0e-10
        assert row["patch_residuals"]["bending"] < 1.0e-10
        assert row["force_loaded_in_plane"]["action_reaction_residual"] < 1.0e-10
        assert row["force_loaded_in_plane"]["edge_work_residual"] < 1.0e-10
        assert row["symmetry_residual"] < 1.0e-12
        assert row["covariance_residual"] < 1.0e-12

    convergence = shards[common.SHARD_IDS[1]]["diagnostic_payload"]
    assert convergence["selected_sequence_count"] < convergence["complete_registered_sequence_count"]
    assert convergence["scope_complete"] is False
    assert convergence["energy_scope"].endswith("NOT_A_PROVEN_ENERGY_NORM_ERROR")
    assert convergence["contradictions_classifying"] is False
    assert set(convergence["rows"][0]["records"][0]["pl_participation"]) == {
        "Q4_PL",
        "S3_PL",
    }
    assert "q4_residual_hourglass_participation" in convergence["rows"][0]["records"][0]

    locking = shards[common.SHARD_IDS[2]]["diagnostic_payload"]
    assert locking["analytical_reference"].startswith("EULER_BERNOULLI_")
    assert locking["scope"].endswith("NOT_THE_COMPLETE_REGISTERED_SQUARE_MASK_CAMPAIGN")
    assert locking["contradictions_classifying"] is False
    assert set(locking["special_fixtures"].values()) == {
        "UNEXECUTED_NO_DEDICATED_FIXTURE_CONSTRUCTED"
    }
    assert all(
        row["relative_error"] < 0.02
        for group in locking["rows"]
        for row in group["rows"]
    )


def test_terminal_precedence_and_malformed_shards_fail_closed(
    deterministic_quick_cycles: tuple[bytes, dict[str, Any], Path],
    authorities: Any,
) -> None:
    assert common.choose_terminal([common.COMPLETE] * 7, blocked=False) == common.TERMINALS[3]
    assert common.choose_terminal([common.COMPLETE, common.PARTIAL], blocked=False) == common.TERMINALS[2]
    assert common.choose_terminal([common.COMPLETE, common.FAIL], blocked=False) == common.TERMINALS[1]
    assert common.choose_terminal([common.COMPLETE] * 7, blocked=True) == common.TERMINALS[0]
    _raw, _aggregate, root = deterministic_quick_cycles
    source = next(
        (root / "diagnostics" / "cycle-1" / common.SHARD_IDS[0].lower()).glob("shard.json")
    )
    original = json.loads(source.read_bytes())

    bad_hash = copy.deepcopy(original)
    bad_hash["diagnostic_payload"]["basis"][0]["force_loaded_in_plane"][
        "patch_residual"
    ] = 1.0
    with pytest.raises(common.StructuralEvidenceError, match="diagnostic payload hash mismatch"):
        common.validate_shard(bad_hash, authorities=authorities)

    for shard_id in common.SHARD_IDS:
        shard_path = root / "diagnostics" / "cycle-1" / shard_id.lower() / "shard.json"
        extra = json.loads(shard_path.read_bytes())
        extra["diagnostic_payload"]["undeclared"] = True
        extra["diagnostic_payload_sha256"] = common.sha256(
            common.canonical_bytes(extra["diagnostic_payload"])
        )
        with pytest.raises(common.StructuralEvidenceError, match="payload keys differ"):
            common.validate_shard(extra, authorities=authorities)

    stale_predicate = copy.deepcopy(original)
    stale_predicate["diagnostic_payload"]["basis"][0]["force_loaded_in_plane"][
        "patch_residual"
    ] = 1.0
    stale_predicate["diagnostic_payload_sha256"] = common.sha256(
        common.canonical_bytes(stale_predicate["diagnostic_payload"])
    )
    with pytest.raises(common.StructuralEvidenceError, match="contradictions were not recomputed"):
        common.validate_shard(stale_predicate, authorities=authorities)

    for component in ("membrane", "bending"):
        stale_component = copy.deepcopy(original)
        stale_component["diagnostic_payload"]["basis"][0]["patch_residuals"][
            component
        ] = 1.0
        stale_component["diagnostic_payload_sha256"] = common.sha256(
            common.canonical_bytes(stale_component["diagnostic_payload"])
        )
        with pytest.raises(
            common.StructuralEvidenceError,
            match="contradictions were not recomputed",
        ):
            common.validate_shard(stale_component, authorities=authorities)

    nonclassifying_shear = copy.deepcopy(original)
    nonclassifying_shear["diagnostic_payload"]["basis"][0]["patch_residuals"][
        "shear"
    ] = 1.0
    nonclassifying_shear["diagnostic_payload_sha256"] = common.sha256(
        common.canonical_bytes(nonclassifying_shear["diagnostic_payload"])
    )
    validated_shear = common.validate_shard(
        nonclassifying_shear,
        authorities=authorities,
    )
    assert validated_shear["gate_status"]["patch_and_equilibrium"] == common.PARTIAL

    convergence_path = (
        root
        / "diagnostics"
        / "cycle-1"
        / common.SHARD_IDS[1].lower()
        / "shard.json"
    )
    q4_residual_only = json.loads(convergence_path.read_bytes())
    q4_residual_only["diagnostic_payload"]["rows"][0]["records"][0][
        "q4_residual_hourglass_participation"
    ] = 100.0
    q4_residual_only["diagnostic_payload_sha256"] = common.sha256(
        common.canonical_bytes(q4_residual_only["diagnostic_payload"])
    )
    validated = common.validate_shard(q4_residual_only, authorities=authorities)
    assert validated["gate_status"]["pl_participation"] == common.PARTIAL

    stale_pl = json.loads(convergence_path.read_bytes())
    stale_pl["diagnostic_payload"]["rows"][0]["records"][-1]["pl_participation"][
        "S3_PL"
    ] = 1.0
    stale_pl["diagnostic_payload_sha256"] = common.sha256(
        common.canonical_bytes(stale_pl["diagnostic_payload"])
    )
    with pytest.raises(common.StructuralEvidenceError, match="contradictions were not recomputed"):
        common.validate_shard(stale_pl, authorities=authorities)

    for key, value, message in (
        ("coverage", {**original["coverage"], "executed_gate_count": 0}, "coverage counts"),
        (
            "gate_status",
            {**original["gate_status"], "patch_and_equilibrium": common.COMPLETE},
            "gate predicates",
        ),
        ("terminal_status", "CONTRADICTION", "terminal was not recomputed"),
    ):
        mutated = copy.deepcopy(original)
        mutated[key] = value
        with pytest.raises(common.StructuralEvidenceError, match=message):
            common.validate_shard(mutated, authorities=authorities)


def test_duplicate_noncanonical_and_candidate_mutations_fail_before_mechanics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = INPUT.read_bytes()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(original.replace(b"{\n", b'{\n  "schema": "duplicate",\n', 1))
    with pytest.raises(common.StructuralEvidenceError, match="duplicate key 'schema'"):
        common.load_authorities(duplicate)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(original + b"\n")
    with pytest.raises(common.StructuralEvidenceError, match="not canonical pretty JSON"):
        common.load_authorities(noncanonical)

    mutated = copy.deepcopy(json.loads(original))
    mutated["candidate"]["tree"] = "0" * 40
    path = tmp_path / "mutated.json"
    path.write_bytes(common.pretty_canonical_bytes(mutated))
    with pytest.raises(common.StructuralEvidenceError, match="candidate identity changed"):
        common.load_authorities(path)

    exact_extent = [("A", relative) for relative in common.ALLOWED_EXECUTION_EXTENT]
    common._validate_execution_extent(exact_extent)
    with pytest.raises(
        common.StructuralEvidenceError,
        match="nine-file research boundary",
    ):
        common._validate_execution_extent(
            exact_extent + [("A", "docs/reference_cases/undeclared.py")]
        )
    with pytest.raises(
        common.StructuralEvidenceError,
        match="nine-file research boundary",
    ):
        common._validate_execution_extent(exact_extent[:-1])
    changed_status = list(exact_extent)
    changed_status[0] = ("M", changed_status[0][1])
    with pytest.raises(
        common.StructuralEvidenceError,
        match="non-addition",
    ):
        common._validate_execution_extent(
            changed_status
        )

    program_mutation = copy.deepcopy(json.loads(original))
    program_mutation["authority"]["programs"]["producer"]["sha256"] = "0" * 64
    program_path = tmp_path / "program-mutation.json"
    program_path.write_bytes(common.pretty_canonical_bytes(program_mutation))
    mechanics_loaded = False

    def forbidden_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal mechanics_loaded
        mechanics_loaded = True
        raise AssertionError("mechanics loaded before program authority failure")

    monkeypatch.setattr(common, "_load_source", forbidden_load)
    with pytest.raises(common.StructuralEvidenceError, match="producer program hash mismatch"):
        common.load_authorities(program_path)
    assert mechanics_loaded is False


def test_three_bounded_processes_overlap_and_timeout_memory_remove_output(tmp_path: Path) -> None:
    environment = _environment()

    def sleeper(index: int) -> bounded.ProcessResult:
        return bounded.run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            cwd=ROOT,
            environment=environment,
            stdout_path=tmp_path / f"parallel-{index}.out",
            stderr_path=tmp_path / f"parallel-{index}.err",
            timeout_seconds=2,
            memory_limit_bytes=1024**3,
            rss_reader=lambda _pid: 1,
        )

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(sleeper, range(3)))
    assert time.monotonic() - started < 0.6
    assert all(row.status == "COMPLETE" for row in rows)

    child_code = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid),encoding='ascii');"
        "pathlib.Path(sys.argv[2]).write_bytes(b'partial');"
        "time.sleep(30)"
    )
    timeout_pid = tmp_path / "timeout-child.pid"
    timeout_partial = tmp_path / "timeout-partial.json"
    timeout = bounded.run_bounded_process(
        [sys.executable, "-c", child_code, str(timeout_pid), str(timeout_partial)],
        cwd=ROOT,
        environment=environment,
        stdout_path=tmp_path / "timeout.out",
        stderr_path=tmp_path / "timeout.err",
        timeout_seconds=0.25,
        memory_limit_bytes=1024**3,
        rss_reader=lambda _pid: 1,
    )
    coordinator._discard_incomplete(timeout_partial, timeout)
    assert timeout.status == "TIMEOUT"
    assert timeout_pid.is_file()
    assert not timeout_partial.exists()

    memory_pid = tmp_path / "memory-child.pid"
    memory_partial = tmp_path / "memory-partial.json"
    rss_calls = 0

    def breach_after_start(_pid: int) -> int:
        nonlocal rss_calls
        rss_calls += 1
        return 1 if rss_calls < 5 else 11

    memory = bounded.run_bounded_process(
        [sys.executable, "-c", child_code, str(memory_pid), str(memory_partial)],
        cwd=ROOT,
        environment=environment,
        stdout_path=tmp_path / "memory.out",
        stderr_path=tmp_path / "memory.err",
        timeout_seconds=2,
        memory_limit_bytes=10,
        rss_reader=breach_after_start,
    )
    coordinator._discard_incomplete(memory_partial, memory)
    assert memory.status == "MEMORY_LIMIT"
    assert memory_pid.is_file()
    assert not memory_partial.exists()

    if os.name == "nt":
        for pid_path in (timeout_pid, memory_pid):
            pid = pid_path.read_text(encoding="ascii")
            listing = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout
            assert f'"{pid}"' not in listing

    rss_missing = bounded.ProcessResult(
        status="COMPLETE",
        returncode=0,
        elapsed_ms=1,
        peak_rss_bytes=None,
        stdout_path="stdout.log",
        stderr_path="stderr.log",
    )
    normalized = coordinator._require_observed_rss(rss_missing)
    rss_partial = tmp_path / "rss-partial.json"
    rss_partial.write_bytes(b"partial")
    coordinator._discard_incomplete(rss_partial, normalized)
    assert normalized.status == "RSS_UNAVAILABLE"
    assert not rss_partial.exists()
