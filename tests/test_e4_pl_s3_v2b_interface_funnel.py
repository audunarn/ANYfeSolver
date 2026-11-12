from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from anyfesolver.elements import DEFAULT_Q4_FORMULATION, DEFAULT_S3_FORMULATION


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from e4_pl_s3_v2b_interface_checker import verify
from e4_pl_s3_v2b_interface_funnel import (
    FORMULATION_ID,
    adjudicate,
    boundary_map,
    macrocell_components,
    produce_proof,
)


def test_v2a_nogo_and_production_boundary_are_frozen() -> None:
    status = json.loads((REFERENCE / "e4_pl_s3_v2_stage4a_nogo_status.json").read_text())
    assert status["aggregate"]["sha256"] == "47CD9DEF9AC306635C16B662ECBF3628324350CCC80803D1AC586BC0A22D60F1"
    assert status["terminal"] == "NO_GO_E4_PL_S3_V2A_MIXED_FLEXURAL_CONVERGENCE"
    assert status["activation_authorized"] is False
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"


def test_contract_binds_frozen_programs_and_process_limits() -> None:
    contract = json.loads((REFERENCE / "e4_pl_s3_v2b_interface_contract.json").read_text())
    for entry in contract["frozen_inputs"].values():
        if "path" not in entry:
            continue
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == entry["sha256"]
    assert contract["execution"] == {
        "checker_replica_count": 2,
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_cycle_count": 2,
    }


@pytest.mark.parametrize("diagonal", ["slash", "backslash", "alternating"])
def test_macrocell_components_are_symmetric_and_nonidentical(diagonal: str) -> None:
    components = macrocell_components(diagonal)
    for name, matrix in components.items():
        np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=2.0e-5, err_msg=name)
    assert not np.array_equal(components["q4_total"], components["s3_total"])
    assert np.linalg.matrix_rank(components["s3_total"], tol=np.linalg.norm(components["s3_total"], 2) * 1.0e-9) == 18


def test_boundary_maps_cover_1_2_4_and_remain_symmetric() -> None:
    for size in (1, 2, 4):
        q4 = boundary_map(size, "q4", "alternating")
        s3 = boundary_map(size, "s3", "alternating")
        expected = 6 * ((size + 1) ** 2 - max(size - 1, 0) ** 2)
        assert q4.shape == s3.shape == (expected, expected)
        np.testing.assert_allclose(q4, q4.T, rtol=0.0, atol=2.0e-5)
        np.testing.assert_allclose(s3, s3.T, rtol=0.0, atol=2.0e-5)


def test_independent_source_reconstruction_agrees_and_mutation_is_rejected() -> None:
    proof = produce_proof(include_development=False)
    report = verify(proof)
    assert report["source_equation_agreement"] is True
    assert report["authority_complete"] is True
    assert report["macrocell_operator_mismatch_nonzero"] is True
    assert report["v2a_mixed_no_go_bound"] is True
    assert report["development_successive_response_failed"] is False
    mutated = copy.deepcopy(proof)
    mutated["matrices"]["slash"]["s3_total"]["values_hex"][0] = float(1.0).hex()
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(mutated)


def test_terminal_precedence_and_v2a_replacement_disposition() -> None:
    base = {
        "authority_complete": True,
        "development_successive_response_failed": True,
        "macrocell_operator_mismatch_nonzero": True,
        "source_equation_agreement": True,
        "v2a_mixed_no_go_bound": True,
    }
    assert adjudicate(checker_identical=False, report=base, formulation_id=FORMULATION_ID) == "BLOCKED_E4_PL_S3_V2B_PROCESS_OR_EVIDENCE"
    local = base | {"source_equation_agreement": False}
    assert adjudicate(checker_identical=True, report=local, formulation_id=FORMULATION_ID) == "NO_GO_E4_PL_S3_V2B_LOCAL_OPERATOR"
    assert adjudicate(checker_identical=True, report=base, formulation_id=FORMULATION_ID) == "UNCLASSIFIED_E4_PL_S3_V2B_FORMULATION_REPLACEMENT_REQUIRED"
    assert adjudicate(checker_identical=True, report=base, formulation_id="CANDIDATE_E4_PL_S3_V2B_FLAT_LINEAR_V1") == "NO_GO_E4_PL_S3_V2B_MIXED_INTERFACE"
    passed = base | {"development_successive_response_failed": False, "macrocell_operator_mismatch_nonzero": False, "v2a_mixed_no_go_bound": False}
    assert adjudicate(checker_identical=True, report=passed, formulation_id="CANDIDATE_E4_PL_S3_V2B_FLAT_LINEAR_V1") == "PROVISIONAL_GO_E4_PL_S3_V2B_STAGE4A_RERUN"
