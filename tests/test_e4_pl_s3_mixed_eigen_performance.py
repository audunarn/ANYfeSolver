"""Focused tests for the bounded mixed eigen/performance lane."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_mixed_eigen_performance.py"
)
INPUT_PATH = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_mixed_eigen_performance_input.json"
)


def _load() -> Any:
    name = "_e4_pl_s3_mixed_eigen_performance_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lane() -> Any:
    return _load()


@pytest.fixture(scope="module")
def authorities(lane: Any) -> Any:
    return lane.load_authorities(INPUT_PATH)


def test_input_is_canonical_hash_bound_and_uses_the_frozen_candidate(
    lane: Any, authorities: Any
) -> None:
    raw = INPUT_PATH.read_bytes()
    assert raw == lane.pretty_canonical_bytes(json.loads(raw))
    assert authorities.input_raw == raw
    assert authorities.input["authority"]["candidate"] == {
        "allowed_successor_paths": [
            "docs/reference_cases/e4_pl_s3_mixed_eigen_performance.py",
            "docs/reference_cases/e4_pl_s3_mixed_eigen_performance_input.json",
            "docs/reference_cases/e4_pl_s3_mixed_structural_common.py",
            "docs/reference_cases/e4_pl_s3_mixed_structural_coordinator.py",
            "docs/reference_cases/e4_pl_s3_mixed_structural_input.json",
            "docs/reference_cases/e4_pl_s3_mixed_structural_input_schema.json",
            "docs/reference_cases/e4_pl_s3_mixed_structural_producer.py",
            "tests/test_e4_pl_s3_mixed_eigen_performance.py",
            "tests/test_e4_pl_s3_mixed_structural_qualification.py",
        ],
        "changed_paths": [
            "src/anyfesolver/matrix_assembly.py",
            "src/anyfesolver/s3_reference_batch.py",
            "tests/test_e4_pl_s3_reference_batch.py",
        ],
        "commit": "4e5b5976d4286ffd0cda5b8424d154132f3f8da0",
        "parent": "7d85ecef35daa6ebe11f11536a7ede4d288e0aa3",
        "subject": "perf: amortize immutable S3 matrix validation",
        "tree": "96f60dcdd61a78111091ce4f93d7170cf7d0878a",
    }
    assert authorities.input["execution"] == {
        "automatic_retry": False,
        "canonical_cycles": 2,
        "memory_limit_gib_per_process": 24,
        "numerical_library_threads_per_process": 1,
        "performance_workers_are_serial": True,
        "timeout_seconds_per_process": 600,
        "worker_concurrency": 3,
        "worker_ids": list(lane.WORKER_IDS),
    }
    assert authorities.contract["candidate"]["qualified_q4_mechanics_sha256"] == (
        "EE49BAE1C9439C41EC2D61798A8A8B88CBA9081DCAD2DFDC857FE313C6C0D4D1"
    )


def test_duplicate_nonfinite_hash_and_topology_mutations_fail_before_mechanics(
    lane: Any, tmp_path: Path
) -> None:
    raw = INPUT_PATH.read_bytes()
    duplicate = raw.replace(
        b'  "schema": "anysolver.e4-pl-s3-mixed-eigen-performance-input-v1"',
        b'  "schema": "bad",\n  "schema": "anysolver.e4-pl-s3-mixed-eigen-performance-input-v1"',
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_bytes(duplicate)
    with pytest.raises(lane.EigenPerformanceError, match="duplicate key"):
        lane.load_authorities(duplicate_path)

    payload = json.loads(raw)
    payload["coverage"]["modal"]["shift"] = float("nan")
    with pytest.raises(ValueError):
        lane.pretty_canonical_bytes(payload)

    payload = json.loads(raw)
    payload["authority"]["production_boundary"]["q4_mechanics"][
        "sha256"
    ] = "0" * 64
    mutation = tmp_path / "hash.json"
    mutation.write_bytes(lane.pretty_canonical_bytes(payload))
    with pytest.raises(lane.EigenPerformanceError, match="hash mismatch"):
        lane.load_authorities(mutation)

    payload = json.loads(raw)
    payload["authority"]["candidate"]["changed_paths"] = sorted(
        [
            *payload["authority"]["candidate"]["changed_paths"][:-1],
            "src/anyfesolver/e4_pl_element.py",
        ]
    )
    extent = tmp_path / "extent.json"
    extent.write_bytes(lane.pretty_canonical_bytes(payload))
    with pytest.raises(lane.EigenPerformanceError, match="changed extent mismatch"):
        lane.load_authorities(extent)

    payload = json.loads(raw)
    payload["authority"]["candidate"]["allowed_successor_paths"] = (
        payload["authority"]["candidate"]["allowed_successor_paths"][:-1]
    )
    successor_extent = tmp_path / "successor-extent.json"
    successor_extent.write_bytes(lane.pretty_canonical_bytes(payload))
    with pytest.raises(
        lane.EigenPerformanceError,
        match="candidate-to-current evidence extent mismatch",
    ):
        lane.load_authorities(successor_extent)

    payload = json.loads(raw)
    payload["authority"]["prior_hot_path_baseline"]["files"][
        "cycle_1_batch"
    ]["sha256"] = "0" * 64
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(lane.pretty_canonical_bytes(payload))
    with pytest.raises(lane.EigenPerformanceError, match="hash mismatch"):
        lane.load_authorities(baseline)

    payload = json.loads(raw)
    payload["coverage"]["candidate_hot_path_baseline"][
        "maximum_regression"
    ] = "0.06"
    policy = tmp_path / "policy.json"
    policy.write_bytes(lane.pretty_canonical_bytes(payload))
    with pytest.raises(lane.EigenPerformanceError, match="gate policy changed"):
        lane.load_authorities(policy)

    payload = json.loads(raw)
    payload["coverage"]["matched_topologies"]["mixed_10_percent"][
        "connectivity_sha256"
    ] = "A" * 64
    topology = tmp_path / "topology.json"
    topology.write_bytes(lane.pretty_canonical_bytes(payload))
    with pytest.raises(lane.EigenPerformanceError, match="topology 10% mismatch"):
        lane.load_authorities(topology)


def test_registered_free_assemblies_have_exactly_six_scaled_null_roots(
    lane: Any, authorities: Any
) -> None:
    for fraction in (0, 10):
        built = lane._build_case(authorities, fraction, auxiliary=False)
        certificate = lane._free_rigid_certificate(built.model, 10)
        assert certificate["support_state"] == "NO_BOUNDARY_CONDITIONS_APPLIED"
        assert certificate["analytic_basis_columns"] == 6
        assert certificate["analytic_basis_rank"] == 6
        assert certificate["numerical_nullity"] == 6
        assert certificate["first_flexible_root"] > 1.0e-8
        assert certificate["analytic_null_action_residual"] <= 1.0e-12
        assert certificate["passed"] is True


def test_clustered_mac_is_basis_invariant_and_detects_a_changed_subspace(
    lane: Any
) -> None:
    reference = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
    )
    angle = 0.37
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        )
    )
    candidate = reference.copy()
    candidate[:, :2] = reference[:, :2] @ rotation
    values = (1.0, 1.0 + 1.0e-6, 3.0)
    result = lane._clustered_mac(
        reference,
        candidate,
        sparse.eye(4, format="csr"),
        values,
        1.0e-4,
    )
    assert result["minimum_clustered_mac"] == pytest.approx(1.0, abs=2.0e-15)

    candidate[:, 2] = np.asarray((0.0, 0.0, 0.0, 1.0))
    changed = lane._clustered_mac(
        reference,
        candidate,
        sparse.eye(4, format="csr"),
        values,
        1.0e-4,
    )
    assert changed["minimum_clustered_mac"] == pytest.approx(0.0)


def test_physical_global_buckling_prestress_is_transformed_per_element_frame(
    lane: Any, authorities: Any
) -> None:
    built = lane._build_case(authorities, 10, auxiliary=False)
    spec = authorities.input["coverage"]["buckling"]
    states = lane._reference_elastic_states(built.model, spec)
    global_tensor = np.asarray(
        spec["physical_global_membrane_compression_tensor"], dtype=float
    )
    observed_local: set[tuple[float, float, float]] = set()
    for element_id, element in built.model.mesh.elements.items():
        material = built.model.get_material(element.material_name)
        frame = np.asarray(
            element.compute_stiffness_components(built.model.mesh, material)["frame"],
            dtype=float,
        )
        expected = frame[:, :2].T @ global_tensor @ frame[:, :2]
        state = states[int(element_id)]
        actual = np.asarray(state["membrane_compression"], dtype=float)
        np.testing.assert_allclose(
            actual,
            (expected[0, 0], expected[1, 1], expected[0, 1]),
            rtol=0.0,
            atol=4.0e-16,
        )
        np.testing.assert_allclose(
            state["stress_second_moment"],
            actual * float(element.thickness) ** 2 / 12.0,
            rtol=0.0,
            atol=4.0e-20,
        )
        observed_local.add(tuple(round(float(value), 12) for value in actual))
    assert {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, -0.5)} <= observed_local


def test_production_buckling_worker_passes_the_physical_global_n20_gate(
    lane: Any, authorities: Any
) -> None:
    record = lane.run_worker(authorities, "BUCKLING_MIXED_10")
    lane.validate_worker(record, "BUCKLING_MIXED_10")
    statuses = record["common"]["gate_status"]
    assert statuses["buckling_factors"] == lane.PASS
    assert statuses["buckling_mac"] == lane.PASS
    assert statuses["buckling_positive"] == lane.PASS
    diagnostics = record["diagnostic_payload"]
    assert diagnostics["fraction_percent"] == 10
    assert len(diagnostics["reference_factors"]) == 5
    assert len(diagnostics["candidate_factors"]) == 5
    assert diagnostics["maximum_factor_relative_error"] < 0.003
    assert diagnostics["clustered_mac"]["minimum_clustered_mac"] >= 0.95
    assert "all boundary nodes constrain ux, uy, uz" in diagnostics[
        "boundary_construction"
    ]


def _fake_worker(
    lane: Any,
    authorities: Any,
    worker_id: str,
    statuses: dict[str, str],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "gate_status": statuses,
        "production_restriction": lane.PRODUCTION_RESTRICTION,
        "worker_id": worker_id,
    }
    return {
        "authority_sha256": lane.sha256(authorities.input_raw),
        "common": common,
        "diagnostic_payload": diagnostics,
        "diagnostic_payload_sha256": lane.sha256(lane.canonical_bytes(diagnostics)),
        "schema": lane.WORKER_SCHEMA,
        "worker_id": worker_id,
    }


def test_aggregate_is_deterministic_excludes_timings_and_closes_hot_path_gate(
    lane: Any, authorities: Any, tmp_path: Path
) -> None:
    process_results = []
    for index, worker_id in enumerate(lane.WORKER_IDS):
        directory = tmp_path / worker_id.lower()
        directory.mkdir()
        if worker_id.startswith("MODAL_"):
            statuses = {
                "modal_frequency": lane.PASS,
                "modal_mac": lane.PASS,
                "rigid_modes": lane.PASS,
            }
            diagnostics = {"worker_elapsed_seconds": 0.01 + index}
        elif worker_id.startswith("BUCKLING_"):
            statuses = {
                "buckling_factors": lane.PASS,
                "buckling_mac": lane.PASS,
                "buckling_positive": lane.PASS,
            }
            diagnostics = {"worker_elapsed_seconds": 0.01 + index}
        elif worker_id.startswith("PERFORMANCE_"):
            fraction = 0 if worker_id.endswith("ALL_Q4") else int(
                worker_id.rsplit("_", 1)[1]
            )
            statuses = {"performance_measurement": lane.PASS}
            diagnostics = {
                "assembly": {"median_seconds": 1.0 + 0.01 * (fraction > 0)},
                "fraction_percent": fraction,
                "peak_rss_bytes": 1000 + (10 if fraction else 0),
                "production_end_to_end_solve": {
                    "median_seconds": 2.0 + 0.02 * (fraction > 0)
                },
                "worker_elapsed_seconds": 10.0 + index,
            }
        else:
            statuses = {
                "batch_equality": lane.PASS,
                "batch_scalar_fallback": lane.PASS,
                "batch_throughput": lane.PASS,
                "warm_s3_vs_q4": lane.PASS,
            }
            diagnostics = {
                "benchmark": {
                    "qualified_q4_comparator": {
                        "batch": {"median_seconds": 1.0}
                    },
                    "recovery": {
                        "batch": {"median_seconds": 0.1},
                        "scalar": {"median_seconds": 1.0},
                    },
                    "stiffness": {
                        "batch": {"median_seconds": 0.04},
                        "scalar": {"median_seconds": 0.2},
                    },
                },
                "worker_elapsed_seconds": 99.0,
            }
        lane.write_exclusive(
            directory / "record.json",
            _fake_worker(lane, authorities, worker_id, statuses, diagnostics),
        )
        (directory / "stdout.log").write_bytes(b"")
        (directory / "stderr.log").write_bytes(b"")
        process_results.append(
            lane.ProcessResult(
                worker_id=worker_id,
                status="COMPLETE",
                returncode=0,
                elapsed_seconds=100.0 + index,
                peak_rss_bytes=5000 + index,
                directory=directory,
            )
        )
    first, first_diagnostics = lane._aggregate(authorities, 1, process_results)
    second, second_diagnostics = lane._aggregate(authorities, 2, process_results)
    assert lane.canonical_bytes(first) == lane.canonical_bytes(second)
    assert first["gate_status"]["candidate_hot_path_regression"] == lane.PASS
    assert first["terminal"] == lane.TERMINALS[3]
    comparison = first_diagnostics["candidate_hot_path_comparison"]
    assert comparison["primary_metric"]["metric"] == lane.HOT_PATH_PRIMARY_METRIC
    assert comparison["primary_metric"]["gate_status"] == lane.PASS
    assert tuple(comparison["normalized_diagnostics"]) == (
        lane.HOT_PATH_DIAGNOSTIC_METRICS
    )
    assert "seconds" not in lane.canonical_bytes(first).decode("ascii")
    assert first_diagnostics != second_diagnostics


def test_candidate_hot_path_gate_rejects_a_regressed_normalized_metric(
    lane: Any, authorities: Any
) -> None:
    accepted = copy.deepcopy(authorities.hot_path_baseline_cycles[0])
    status, diagnostics = lane._candidate_hot_path_gate(authorities, accepted)
    assert status == lane.PASS
    assert diagnostics["primary_metric"]["gate_status"] == lane.PASS
    assert diagnostics["primary_metric"]["baseline_cycle_values"] == pytest.approx(
        [0.061557499997434206, 0.06595939999533584],
        rel=0.0,
        abs=0.0,
    )
    assert diagnostics["primary_metric"]["gate_limit"] == pytest.approx(
        0.06595939999533584 * 1.05,
        rel=0.0,
        abs=0.0,
    )

    diagnostic_only = copy.deepcopy(accepted)
    diagnostic_only["PERFORMANCE_MIXED_10"]["diagnostic_payload"]["assembly"][
        "median_seconds"
    ] *= 2.0
    status, changed_diagnostics = lane._candidate_hot_path_gate(
        authorities, diagnostic_only
    )
    assert status == lane.PASS
    assert changed_diagnostics["normalized_diagnostics"][
        "mixed_10_assembly_ratio_to_all_q4"
    ]["candidate_value"] > diagnostics["normalized_diagnostics"][
        "mixed_10_assembly_ratio_to_all_q4"
    ]["candidate_value"]

    regressed = copy.deepcopy(accepted)
    regressed["BATCH_4096"]["diagnostic_payload"]["benchmark"]["stiffness"][
        "batch"
    ][
        "median_seconds"
    ] *= 2.0
    status, diagnostics = lane._candidate_hot_path_gate(authorities, regressed)
    assert status == lane.FAIL
    assert diagnostics["primary_metric"]["gate_status"] == lane.FAIL


def test_timeout_terminates_worker_and_removes_partial_canonical_output(
    lane: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "timeout-worker"
    code = (
        "from pathlib import Path; import sys,time; "
        "Path(sys.argv[1]).write_text('{}'); time.sleep(5)"
    )
    result = lane.run_bounded_process(
        "TIMEOUT_TEST",
        [sys.executable, "-c", code, str(directory / "record.json")],
        directory=directory,
        environment={**os.environ, **lane.THREAD_ENVIRONMENT},
        timeout_seconds=1,
        memory_limit_bytes=1 << 30,
        rss_reader=lambda _pid: 1,
    )
    assert result.status == "TIMEOUT"
    assert not (directory / "record.json").exists()


def test_default_s3_remains_legacy_and_q4_mechanics_is_unchanged(
    authorities: Any,
) -> None:
    from anyfesolver.elements import DEFAULT_S3_FORMULATION

    assert DEFAULT_S3_FORMULATION == "legacy-s3"
    q4 = ROOT / "src" / "anyfesolver" / "e4_pl_element.py"
    assert authorities.input["authority"]["production_boundary"]["q4_mechanics"][
        "sha256"
    ] == (
        __import__("hashlib").sha256(q4.read_bytes()).hexdigest().upper()
    )
    changed = set(authorities.input["authority"]["candidate"]["changed_paths"])
    boundary = authorities.input["authority"]["production_boundary"]
    assert changed.isdisjoint(row["path"] for row in boundary.values())
    for row in boundary.values():
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert __import__("hashlib").sha256(path.read_bytes()).hexdigest().upper() == (
            row["sha256"]
        )


def test_cli_requires_exclusive_mode_and_output(lane: Any) -> None:
    with pytest.raises(SystemExit):
        lane.main([])
    with pytest.raises(SystemExit):
        lane.main(["--worker", "BATCH_4096"])
