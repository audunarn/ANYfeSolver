"""Focused gates for the quarantined S4 nullspace-semantics proof oracle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
ORACLE_PATH = REPOSITORY / "docs" / "reference_cases" / "s4_nullspace_semantics_oracle.py"
CASES_PATH = ORACLE_PATH.with_name("s4_nullspace_semantics_cases.json")

ORACLE = None
NP = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(result: dict, identifier: str) -> dict:
    for group in ("local_cases", "topology_cases"):
        for item in result[group]:
            if item["id"] == identifier:
                return item
    raise AssertionError(f"missing case {identifier}")


_CACHE: dict[tuple[str, ...], dict] = {}


def _run(*identifiers: str) -> dict:
    key = tuple(sorted(identifiers))
    if key not in _CACHE:
        _CACHE[key] = ORACLE.run_proof(set(identifiers) if identifiers else None)
    return _CACHE[key]


def _check_registered_plan_source_and_case_domains() -> None:
    data = ORACLE.load_cases()
    assert ORACLE.PLAN_SHA256 == "855d76f0ca40549cbaead8360152f973b3162671295c53b16f58a5341cc382ca"
    assert ORACLE.EDITOR_PLAN_SHA256 == "136cd18281f61d2705fd3c1145c95c63498be8b255c1d0f8118701d3e33ff3a6"
    assert ORACLE.AUDITOR_PLAN_SHA256 == "6ab98ddb6f50139544610c2058d8ffb0c833749f25f126d404c8f18b76811530"
    assert data["plan_hashes"] == {
        "proof_plan": ORACLE.PLAN_SHA256,
        "vidar_editor_plan": ORACLE.EDITOR_PLAN_SHA256,
        "heimdall_auditor_plan": ORACLE.AUDITOR_PLAN_SHA256,
    }
    assert data["source_hashes"] == ORACLE.SOURCE_HASHES
    assert len(ORACLE.case_ids(data)) == 24
    assert _sha256(CASES_PATH) == data["cases_sha256"]


def _check_exact_two_dimensional_quotient_counterexample() -> None:
    result = _run("algebraic_quotient_counterexample")
    evidence = result["algebraic_cases"][0]
    assert evidence["classification"] == "exact_algebraic_counterexample"
    for multiplier in ("0.25", "1", "4"):
        item = evidence["sensitivity"][multiplier]
        assert item["dimensions"] == {
            "R_N_intersect_P": 0,
            "R_N_intersect_G": 0,
            "RQ": 1,
            "Z": 0,
        }
        assert item["intersection_commutativity"] == 0.0
        assert item["Y_R_parent_scale"] == 1.0


def _check_inherited_parent_scale_family_and_fixed_residual_tolerance() -> None:
    result = _run("algebraic_inherited_scale_family")
    family = result["algebraic_cases"][0]["results"]
    assert [item["expected_rank"] for item in family] == [0, 0, 1, 1]
    assert [item["relative_only_rank"] for item in family] == [0, 1, 1, 1]
    for expected, item in zip((0, 0, 1, 1), family):
        assert [item["sensitivity"][key]["rank"] for key in ("0.25", "1", "4")] == [expected] * 3
        assert {item["sensitivity"][key]["parent_scale"] for key in ("0.25", "1", "4")} == {1.0}
    tolerance = ORACLE.residual_tolerance(2, 2)
    assert tolerance == ORACLE.RESIDUAL_FACTOR * 2 * ORACLE.EPS64


def _check_unit_square_quotient_partition_and_drill_semantics() -> None:
    square = _case(_run("local_square_uniform"), "local_square_uniform")
    assert square["free"]["rank"] == 16
    assert square["free"]["dimensions"] == {
        "N": 8,
        "G": 1,
        "P": 7,
        "R": 6,
        "R_N": 6,
        "R_G": 0,
        "RQ": 6,
        "Z": 1,
    }
    assert square["free"]["sensitivity_stable"]
    assert square["drill"]["constant_candidates"][0]["classification"] == "gauge_evidence"
    checkerboard = square["drill"]["checkerboard_candidates"][0]
    assert checkerboard["classification"] == "positive_mass_strain_null"
    assert checkerboard["mass_null"] is False


def _check_every_square_sensitivity_subspace_has_full_gates() -> None:
    square = _case(_run("local_square_uniform"), "local_square_uniform")
    expected_names = {"N", "G", "P", "R", "R_N", "R_G", "RQ", "Z"}
    for multiplier, variant in square["free"]["sensitivity"].items():
        assert multiplier in {"0.25", "1", "4"}
        assert set(variant["projectors"]) == expected_names
        assert set(variant["gates"]) == expected_names
        assert variant["dimensions"] == square["free"]["dimensions"]
        for gate in variant["gates"].values():
            assert set(gate["projector"]) == {"symmetry", "idempotence", "trace"}
            assert all(value <= ORACLE.residual_tolerance(24, 24) for value in gate["projector"].values())
        assert all(value <= ORACLE.residual_tolerance(24, 24) for value in variant["containment"].values())
        assert variant["derived"]["Y_R_rank"] == variant["derived"]["Y_R_expected_rank"] == 6


def _check_constrained_lc_semantics_affine_rhs_and_couplings() -> None:
    model = _case(
        _run("topology_two_shared_edge_constraints"),
        "topology_two_shared_edge_constraints",
    )
    constraints = {item["id"]: item for item in model["constraints"]}
    assert set(constraints) == {
        "fixed_drill",
        "tied_drill",
        "weighted_affine",
        "dependent_rows",
        "redundant_zero_row",
        "inconsistent_affine",
        "abstract_shell_shell",
        "abstract_beam_shell",
    }
    assert constraints["fixed_drill"]["dimensions"] == {
        "N": 7, "G": 0, "P": 7, "S_RG": 7, "L_C": 6,
        "L_G_C": 0, "RQ": 6, "Z": 1,
    }
    assert constraints["tied_drill"]["dimensions"] == {
        "N": 7, "G": 1, "P": 6, "S_RG": 7, "L_C": 7,
        "L_G_C": 1, "RQ": 6, "Z": 0,
    }
    assert constraints["dependent_rows"]["constraint_rank"] == 1
    assert constraints["redundant_zero_row"]["omitted_zero_rows"] == 1
    assert constraints["weighted_affine"]["feasibility"]["feasible"] is True
    assert constraints["inconsistent_affine"]["feasibility"]["feasible"] is False
    for identifier in ("abstract_shell_shell", "abstract_beam_shell"):
        assert constraints[identifier]["intended_work_conjugacy"]
        assert constraints[identifier]["dimensions"]["RQ"] == 6


def _check_zero_empty_signed_zero_and_canonical_basis_conventions() -> None:
    zero = ORACLE.rank_kernel(NP.zeros((3, 4), dtype=NP.float64))
    assert zero["rank"] == 0 and zero["kernel_dimension"] == 4 and zero["tau"] == 0.0
    empty = ORACLE.rank_kernel(NP.empty((0, 3), dtype=NP.float64))
    assert empty["rank"] == 0 and empty["kernel_dimension"] == 3
    no_columns = ORACLE.rank_kernel(NP.empty((5, 0), dtype=NP.float64))
    assert no_columns["projector"].shape == (0, 0)
    basis = ORACLE.canonical_basis(NP.eye(3, dtype=NP.float64), 3)
    assert NP.array_equal(basis, NP.eye(3, dtype=NP.float64))
    environment = "0" * 64
    positive = NP.array([[0.0]], dtype=NP.float64)
    negative = NP.array([[-0.0]], dtype=NP.float64)
    assert ORACLE.snapshot_digest(positive, "projector", environment) == ORACLE.snapshot_digest(negative, "projector", environment)
    assert ORACLE.canonical_json_bytes({"zero": -0.0}) == b'{"zero":0}\n'


def _check_environment_manifest_and_same_environment_snapshots() -> None:
    manifest, digest = ORACLE.environment_manifest()
    repeated_manifest, repeated_digest = ORACLE.environment_manifest()
    assert repeated_manifest == manifest
    assert repeated_digest == digest
    snapshot_runtime_supported = (
        platform.python_implementation() == "CPython"
        and platform.system() == "Windows"
        and sys.byteorder == "little"
    )
    if not snapshot_runtime_supported:
        assert manifest is None
        assert digest is None
        return

    assert manifest is not None and manifest["schema"] == ORACLE.ENVIRONMENT_SCHEMA
    assert digest is not None and len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert hashlib.sha256(ORACLE._manifest_json_bytes(manifest)).hexdigest() == digest
    assert manifest["thread_controls"] == {name: "1" for name in ORACLE._THREAD_CONTROLS}
    assert set(manifest["numpy_cpu"]) == {"features", "baseline", "dispatch"}
    assert set(manifest["blas_runtime"]) == {"library", "distribution"}
    assert set(manifest["blas_runtime"]["library"]["binary"]) == {"role", "name", "sha256"}
    assert all(set(item) == {"role", "name", "size", "sha256"} for item in manifest["numeric_binary_artifacts"])
    sample = NP.eye(2, dtype=NP.float64)
    first = ORACLE.snapshot_digest(sample, "projector", digest)
    second = ORACLE.snapshot_digest(sample, "projector", digest)
    assert first == second
    assert first != ORACLE.snapshot_digest(sample, "projector", "1" * 64)


def _check_synthetic_loader_binds_exact_six_and_hashes_sources() -> None:
    modules = ORACLE.load_numeric_modules()
    expected = {"anyfesolver", "anyfesolver.shell_formulations"}
    expected.update(f"anyfesolver.shell_formulations.{Path(name).stem}" for name in ORACLE.SOURCE_ORDER)
    assert set(modules) == expected
    for filename, digest in ORACLE.SOURCE_HASHES.items():
        path = REPOSITORY / "src" / "anyfesolver" / "shell_formulations" / filename
        assert ORACLE._canonical_source_hash(path) == digest


def _check_hostile_constraints_and_model_indices_fail_early() -> None:
    scale = NP.ones(24, dtype=NP.float64)
    invalid_specs = [
        {"id": "bool_index", "rows": [{"terms": [[True, 1.0]], "rhs": 0.0}]},
        {"id": "out_of_range", "rows": [{"terms": [[24, 1.0]], "rhs": 0.0}]},
        {"id": "zero_nonzero", "rows": [{"terms": [], "rhs": 1.0}]},
        {"id": "nonfinite", "rows": [{"terms": [[0, float("inf")]], "rhs": 0.0}]},
    ]
    for specification in invalid_specs:
        try:
            ORACLE.normalized_constraints(specification, scale)
        except ORACLE.ProofInputError:
            pass
        else:
            raise AssertionError(f"hostile constraint unexpectedly accepted: {specification['id']}")


def _check_preloaded_anyfesolver_module_is_rejected_in_isolated_process() -> None:
    script = (
        "import importlib.util,pathlib,sys,types;"
        f"p=pathlib.Path({str(ORACLE_PATH)!r});"
        "s=importlib.util.spec_from_file_location('hostile_oracle',p);"
        "o=importlib.util.module_from_spec(s);s.loader.exec_module(o);"
        "sys.modules['AnYsOlVeR']=types.ModuleType('AnYsOlVeR');"
        "\ntry:o.load_numeric_modules()\n"
        "except RuntimeError as e:\n assert 'pre-existing' in str(e)\n"
        "else:raise AssertionError('loader accepted hostile preload')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=REPOSITORY, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _check_numbering_topology_deletion_and_threshold_sensitive_evidence() -> None:
    result = _run(
        "local_square_uniform::cyclic",
        "local_square_uniform::anchored_reversal",
        "topology_odd_cycle_prism",
        "topology_deletion_split",
        "topology_deletion_orphans",
        "topology_curved_warped_patch",
    )
    assert len(result["numbering_invariance"]) == 2
    assert all(item["invariant"] for item in result["numbering_invariance"])
    odd = _case(result, "topology_odd_cycle_prism")
    assert odd["bipartite"] is False and odd["drill"]["checkerboard_candidates"] == []
    split = _case(result, "topology_deletion_split")
    assert len(split["components"]) == 2 and split["orphans"]["dimension"] == 0
    orphan = _case(result, "topology_deletion_orphans")
    assert orphan["orphans"]["nodes"] == [4, 5, 6, 7]
    assert orphan["orphans"]["dimension"] == 24
    curved = _case(result, "topology_curved_warped_patch")
    assert curved["free"]["sensitivity_stable"] is False


def _check_full_registered_proof_catalog_and_byte_repeatability() -> None:
    first = _run()
    assert len(first["algebraic_cases"]) == 2
    assert len(first["local_cases"]) == 12
    assert len(first["topology_cases"]) == 10
    assert all(
        candidate["classification"] != "gauge_evidence"
        for case in [*first["local_cases"], *first["topology_cases"]]
        for candidate in case["drill"]["checkerboard_candidates"]
    )
    normalized_first = ORACLE.canonical_json_bytes(ORACLE.proof_summary(first))
    repeated = ORACLE.run_proof()
    normalized_second = ORACLE.canonical_json_bytes(ORACLE.proof_summary(repeated))
    assert normalized_first == normalized_second
    assert json.loads(normalized_first)["environment_manifest_sha256"] == first["environment_manifest_sha256"]


_WORKER_CHECKS = (
    _check_registered_plan_source_and_case_domains,
    _check_exact_two_dimensional_quotient_counterexample,
    _check_inherited_parent_scale_family_and_fixed_residual_tolerance,
    _check_unit_square_quotient_partition_and_drill_semantics,
    _check_every_square_sensitivity_subspace_has_full_gates,
    _check_constrained_lc_semantics_affine_rhs_and_couplings,
    _check_zero_empty_signed_zero_and_canonical_basis_conventions,
    _check_environment_manifest_and_same_environment_snapshots,
    _check_synthetic_loader_binds_exact_six_and_hashes_sources,
    _check_hostile_constraints_and_model_indices_fail_early,
    _check_preloaded_anyfesolver_module_is_rejected_in_isolated_process,
    _check_numbering_topology_deletion_and_threshold_sensitive_evidence,
    _check_full_registered_proof_catalog_and_byte_repeatability,
)


def _worker_main() -> int:
    """Run all gates in a clean process before NumPy or ANYfeSolver imports."""

    global ORACLE, NP
    assert "numpy" not in sys.modules
    assert not any(
        name.casefold() == "anyfesolver" or name.casefold().startswith("anyfesolver.")
        for name in sys.modules
    )
    specification = importlib.util.spec_from_file_location(
        "s4_nullspace_semantics_test_oracle", ORACLE_PATH
    )
    assert specification is not None and specification.loader is not None
    ORACLE = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(ORACLE)
    NP = ORACLE.np
    completed = []
    for check in _WORKER_CHECKS:
        check()
        completed.append(check.__name__)
    sys.stdout.write(json.dumps({"passed": len(completed), "checks": completed}, sort_keys=True) + "\n")
    return 0


def test_clean_process_proof_worker() -> None:
    """Pytest-facing wrapper immune to plugins that preload NumPy."""

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--clean-worker"],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] == len(_WORKER_CHECKS) == 13


def test_unsupported_snapshot_runtime_requires_no_manifest(monkeypatch) -> None:
    """Cross-runtime proof uses numerical invariants, not a foreign snapshot digest."""

    class UnsupportedOracle:
        calls = 0

        @classmethod
        def environment_manifest(cls):
            cls.calls += 1
            return None, None

    monkeypatch.setattr(sys.modules[__name__], "ORACLE", UnsupportedOracle)
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    _check_environment_manifest_and_same_environment_snapshots()

    assert UnsupportedOracle.calls == 2


if __name__ == "__main__":
    if sys.argv[1:] != ["--clean-worker"]:
        raise SystemExit("expected exactly --clean-worker")
    raise SystemExit(_worker_main())
