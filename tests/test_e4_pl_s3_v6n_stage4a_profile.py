from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "docs" / "reference_cases" / "e4_pl_s3_v6n_stage4a_profile.py"


def _load():
    spec = importlib.util.spec_from_file_location("_s3_v6n_profile", PROGRAM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_is_narrow_nonclassifying_and_bounded() -> None:
    module = _load()
    assert module.ALLOWED_RECORD_IDS == (
        "N80:10PCT:dispersed:slash",
        "N80:10PCT:dispersed:backslash",
        "N80:10PCT:dispersed:alternating",
    )
    assert module.MAX_WALL_SECONDS == 600
    assert module.MEMORY_LIMIT_BYTES == 24 * 1024**3
    assert module.CANDIDATE_ARCHIVE_SHA256 == (
        "AC1EA6D71A273355439B25F915EE7BB383DC60769F22EEA8CC84095CDDAF426F"
    )
    assert module.DEPENDENCY_ARCHIVE_SHA256 == (
        "ABDFD6F6B6E04185FD277E4EE80400FA05B702BD43BA50851C4F9E85A5970C90"
    )
    assert module.OPTIMIZED_CANDIDATE_ARCHIVE_SHA256 == (
        "DBEFBF12554832962C375F0CD827BE5310E0507145A5B6C84CFD68EB9BC2ABA1"
    )
    source = PROGRAM.read_text(encoding="utf-8")
    assert "NONCLASSIFYING_RUNTIME_DIAGNOSTIC_ONLY" in source
    assert "PROVISIONAL_GO" not in source
    assert "NO_GO_E4_PL" not in source
    assert "DEFAULT_S3_FORMULATION" not in source


def test_profile_retains_original_solve_and_component_operations() -> None:
    tree = ast.parse(PROGRAM.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "spsolve" in calls
    source = PROGRAM.read_text(encoding="utf-8")
    assert "assemble_stiffness_matrix(model)" in source
    assert "assemble_load_vector(model, model.load_cases[0])" in source
    assert "element.compute_stiffness_components(model.mesh, material)" in source
    assert "component energies do not reconstruct total energy" in source


def test_profile_bindings_and_registered_timeout_members() -> None:
    module = _load()
    module._require_bindings()
    for record_id in module.ALLOWED_RECORD_IDS:
        member = module._member(record_id)
        assert member["record_id"] == record_id
        assert member["record"]["level"] == 80
        assert member["record"]["s3_area_fraction_percent"] == 10
        assert member["record"]["mask"] == "dispersed"


def test_pressure_surface_traversal_uses_captured_constant_time_guard() -> None:
    # Original intent (a source-text assertion that broke when e593afe02 made the
    # caller's guard argument conditional): the pressure-surface traversal must
    # observe each record through a captured constant-time guard, never through
    # one complete model-wide lifecycle scan per S3 pressure record, which made
    # the diagnostic quadratic in mesh size.  Asserted behaviourally: the number
    # of complete scans over the whole model does not grow with the number of
    # S3 V2D pressure elements on the owned-kernel route.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from test_load_kernel_v2d_pressure_acceleration import (
            _count_scans,
            _mixed_model,
            _pressure_case,
        )
    finally:
        sys.path.pop(0)
    counts = []
    records = []
    for pairs in (3, 12):
        model, q4, v2d = _mixed_model(pairs)
        scans, (_vector, info) = _count_scans(model, _pressure_case(q4, v2d))
        counts.append(scans)
        records.append(len(info["qualified_s3_pressure_surfaces"]))
    assert records == [6, 24]
    assert 0 < counts[0] and counts[1] <= counts[0] + 2, counts


def test_equivalence_pair_is_preserved_and_nonclassifying() -> None:
    module = _load()
    assert tuple(module.COMPARISON_RECORDS) == (
        "N20:1PCT:dispersed:slash",
        "N40:10PCT:dispersed:slash",
    )
    for record_id, (path, digest) in module.COMPARISON_RECORDS.items():
        assert module._sha256_path(path) == digest
        assert module._member(record_id)["record_id"] == record_id
    source = PROGRAM.read_text(encoding="utf-8")
    assert "NONCLASSIFYING_BYTE_EQUIVALENCE_DIAGNOSTIC_ONLY" in source
    assert '"optimized-profile"' in source
