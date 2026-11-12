from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
AUTHORITY_PATH = REFERENCE / "e4_pl_s3_v2_successor_authority.json"
SOURCE_PATH = REFERENCE / "e4_pl_s3_v2_source_equation_contract.json"
IDENTITY_PATH = REFERENCE / "e4_pl_s3_v2_identity_recovery_contract.json"
BOUNDS_PATH = REFERENCE / "e4_pl_s3_v2_bounded_execution_contract.json"


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite JSON value: {value}")


def _decode(raw: bytes) -> dict[str, object]:
    data = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_nonfinite,
    )
    assert isinstance(data, dict)
    canonical = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert raw == canonical
    return data


def _load(path: Path) -> dict[str, object]:
    return _decode(path.read_bytes())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_contracts_are_strict_canonical_json_with_exact_schemas() -> None:
    authority = _load(AUTHORITY_PATH)
    source = _load(SOURCE_PATH)
    identity = _load(IDENTITY_PATH)
    bounds = _load(BOUNDS_PATH)

    assert set(authority) == {
        "attachment",
        "base",
        "current_stage_disposition",
        "default_activation_authorized",
        "defaults",
        "formulation_ids",
        "frozen_inputs",
        "historical_evidence_mutation_authorized",
        "production_restriction",
        "q4_mechanics_change_authorized",
        "schema",
        "source_or_derivation_gate_closed",
        "stage",
        "terminal_precedence",
    }
    assert set(source) == {
        "classification",
        "equation_requirements",
        "gate",
        "indispensable_equation_policy",
        "metadata_is_equation_authority",
        "printed_primary_source_requirements",
        "schema",
        "sources",
    }
    assert set(identity) == {
        "candidate_identities",
        "default_policy",
        "formulation_identity_changes",
        "historical_replay",
        "mechanics_and_recovery_separation",
        "production_restriction",
        "recovery",
        "recovery_identity_changes",
        "schema",
        "selectors",
        "supersession",
    }
    assert set(bounds) == {
        "automatic_retry_allowed",
        "command_hard_walls_seconds",
        "evidence_binding",
        "exception_policy",
        "finalization_reserve_seconds",
        "inactivity",
        "job_object",
        "max_concurrent_workers",
        "memory_limit_bytes_per_process_tree",
        "memory_limit_gib_per_process_tree",
        "no_multiple_waves_per_command",
        "numerical_library_threads_per_worker",
        "process_failure_terminal",
        "process_limits_are_scientific_acceptance_gates",
        "production_restriction",
        "progress",
        "request_reuse_allowed",
        "schema",
        "scientific_and_advisory_ratios",
        "termination_reserve_seconds",
        "timeout_or_memory_breach_is_scientific_no_go",
        "wave_hard_wall_seconds",
        "worst_case_commissioning",
    }
    assert authority["schema"] == "anysolver.e4-pl-s3-v2-successor-authority-v1"
    assert source["schema"] == "anysolver.e4-pl-s3-v2-source-equation-contract-v1"
    assert identity["schema"] == "anysolver.e4-pl-s3-v2-identity-recovery-contract-v1"
    assert bounds["schema"] == "anysolver.e4-pl-s3-v2-bounded-execution-contract-v1"


def test_canonical_parser_rejects_duplicates_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _decode(b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="nonfinite JSON value"):
        _decode(b'{"a":NaN}\n')


def test_successor_binds_exact_base_predecessors_and_defaults() -> None:
    authority = _load(AUTHORITY_PATH)
    assert authority["base"] == {
        "commit": "4dc9ddf48bbd0248ab140693723f2db09b2c960a",
        "subject": "Merge pull request #31 from audunarn/codex/s3-e4-pl-v1-nogo-closeout",
        "tree": "0680e65aaf1ce9a2721e807202e6c55ea574b5ac",
    }
    assert authority["attachment"] == {
        "bytes": 51223,
        "role": "BACKGROUND_PLAN_NOT_EXECUTION_OR_EQUATION_AUTHORITY",
        "sha256": "814F639696D88E4B02487B6BA28B503ECE3790D62FC8331280CE07690F30A871",
    }
    assert authority["defaults"] == {
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
    }
    assert authority["formulation_ids"] == {
        "q4": "E4_PL_QUALIFIED_Q4_HYBRID_V2",
        "s3_v1": "E4_PL_QUALIFIED_S3_COMPANION_V1",
    }
    assert authority["default_activation_authorized"] is False
    assert authority["q4_mechanics_change_authorized"] is False
    assert authority["historical_evidence_mutation_authorized"] is False

    elements = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    q4 = re.search(r'^DEFAULT_Q4_FORMULATION = "([^"]+)"$', elements, re.MULTILINE)
    s3 = re.search(r'^DEFAULT_S3_FORMULATION = "([^"]+)"$', elements, re.MULTILINE)
    assert q4 is not None and q4.group(1) == "e4-pl"
    assert s3 is not None and s3.group(1) == "legacy-s3"
    assert 'FORMULATION_ID = "E4_PL_QUALIFIED_Q4_HYBRID_V2"' in (
        ROOT / "src" / "anyfesolver" / "e4_pl_element.py"
    ).read_text(encoding="utf-8")
    assert 'FORMULATION_ID = "E4_PL_QUALIFIED_S3_COMPANION_V1"' in (
        ROOT / "src" / "anyfesolver" / "e4_pl_s3_state.py"
    ).read_text(encoding="utf-8")


def test_frozen_historical_inputs_retain_exact_bytes_and_hashes() -> None:
    authority = _load(AUTHORITY_PATH)
    expected = {
        "docs/reference_cases/e4_pl_s3_qv9_v1_nogo_result.json": (
            2314,
            "8A497BF1ABA4E048E23ACEAD6974D263D19A02F602AAC53DD07D12257EF34804",
        ),
        "docs/reference_cases/e4_pl_s3_qv9_all_q4_reference_audit.json": (
            1655,
            "7D0AF3B3C5ABF84DE820E62ECD62589B65B10A04371538D219A6E17C040FA7EA",
        ),
        "docs/reference_cases/e4_pl_s3_qv9_v1_nogo_status.json": (
            1162,
            "78D2AA38532FA0928227110263385F47686BC867C879972A7EF2D4BFC8610ED5",
        ),
        "docs/agent_plans/S3_E4_PL_V2_FORMULATION_PLAN.md": (
            4672,
            "B3D93A25AA294BFD20AFE5AF8F4692AD2281A2623B051C761A0A6949AEFF1285",
        ),
        "docs/reference_cases/e4_pl_s3_mixed_mesh_connectivity_manifest.json": (
            110005,
            "3EA7ABD0B332831D62B30B3CD52E0DB85EC951B125340FFAF40A891DC37BD589",
        ),
    }
    registered = {record["path"]: (record["bytes"], record["sha256"]) for record in authority["frozen_inputs"]}
    assert registered == expected
    for relative, (size, digest) in expected.items():
        path = ROOT / relative
        assert path.stat().st_size == size
        assert _sha256(path) == digest
    manifest = next(record for record in authority["frozen_inputs"] if "manifest" in record["path"])
    assert manifest["records"] == 252


def test_metadata_is_route_background_and_cannot_close_equation_gate() -> None:
    source = _load(SOURCE_PATH)
    assert source["metadata_is_equation_authority"] is False
    assert {record["classification"] for record in source["sources"]} == {
        "B",
        "D",
        "P",
    }
    sources = {record["id"]: record for record in source["sources"]}
    assert sources["katili_2019_dkmt_review"]["classification"] == "P"
    assert sources["katili_2019_dkmt_review"]["equation_bytes_bound"] is True
    for identifier in ("s3_v2_dkmt_equation_map", "s3_v2_dkmt_source_ledger"):
        assert sources[identifier]["classification"] == "D"
        assert sources[identifier]["equation_bytes_bound"] is True
    assert all(
        record["equation_bytes_bound"] is False
        for record in source["sources"]
        if record["classification"] == "B"
    )
    assert all(item["required_authority"] == "P_OR_D" for item in source["equation_requirements"])
    requirements = {item["id"]: item for item in source["equation_requirements"]}
    flat = [item for item in requirements.values() if item["phase"] == "FLAT_LINEAR"]
    assert flat and all(item["current_authority"] in {"P", "D"} for item in flat)
    for identifier in (
        "curved_reference_director_and_pseudocurvature_mapping",
        "consistent_mass_and_zero_drill_inertia",
        "total_lagrangian_residual_and_consistent_tangent",
    ):
        assert requirements[identifier]["current_authority"] == "NONE"
    midside = requirements["dkmt_midside_rotation_enhancements"]
    assert midside["status"] == "INCOMPLETE_QUADRATIC_DELTA_BETA_REQUIRED"
    assert "AFFINE_BETA_SUBSTITUTE" in midside["excluded_scope"]
    phi = requirements["dkmt_phi_k_a_delta_inverse_a_u_curvature_shear_coupling"]
    assert phi["authority_scope"].startswith("HOMOGENEOUS_ISOTROPIC")
    assert requirements["physical_quadrature_and_generalized_section_work"]["status"] == (
        "FLAT_ISOTROPIC_ONLY_GENERALIZED_COUPLED_BLOCKED"
    )
    assert source["gate"] == {
        "current_terminal": "BLOCKED_E4_PL_S3_V2_PUBLIC_SOURCE_OR_DERIVATION",
        "flat_linear_gate": "PASS_E4_PL_S3_V2A_STRICT_FLAT_ISOTROPIC_EQUATION_AUTHORITY",
        "flat_linear_production_allowed": True,
        "overall_qualification": False,
        "passes": False,
        "reason": "FLAT_LINEAR_STRICT_ISOTROPIC_SUBSET_IS_AUTHORIZED_BUT_CURVED_DYNAMIC_NONLINEAR_AND_GENERALIZED_SECTION_AUTHORITY_REMAINS_OPEN",
        "unclosed_phases": [
            "CURVED_LINEAR",
            "DYNAMIC",
            "NONLINEAR",
            "ARBITRARY_GENERALIZED_SECTION",
        ],
    }


def test_formulation_and_recovery_identities_are_separate_and_fail_closed() -> None:
    identity = _load(IDENTITY_PATH)
    assert identity["candidate_identities"] == {
        "curved_linear": "CANDIDATE_E4_PL_S3_V2A_CURVED_LINEAR_V1",
        "flat_linear": "CANDIDATE_E4_PL_S3_V2A_FLAT_LINEAR_V1",
        "qualified_final_reserved": "E4_PL_QUALIFIED_S3_COMPANION_V2",
        "qualified_final_usable": False,
    }
    assert identity["recovery"]["qualified_identity_reserved"] == "SHELL_RESULTANT_RECOVERY_V1"
    assert identity["recovery"]["qualified_identity_usable"] is False
    assert identity["selectors"]["explicit_candidate_selectors_available_before_complete_mechanics_freeze"] is False
    assert identity["selectors"]["unversioned_s3_v2_activation_authorized"] is False
    assert identity["historical_replay"]["v1_role"] == "EXACT_EXPLICIT_HISTORICAL_REPLAY_ONLY"
    assert identity["historical_replay"]["silent_fallback_allowed"] is False
    assert identity["supersession"]["new_rule"] == (
        "RECOVERY_ONLY_CHANGES_CREATE_A_NEW_RECOVERY_ID_BUT_NOT_A_NEW_FORMULATION_ID"
    )
    assert set(identity["formulation_identity_changes"]).isdisjoint(identity["recovery_identity_changes"])


def test_execution_contract_has_unconditional_short_hard_walls_and_tree_limits() -> None:
    bounds = _load(BOUNDS_PATH)
    assert bounds["command_hard_walls_seconds"] == {
        "authority_and_static": 300,
        "final_aggregation": 300,
        "flat_exact_or_local_proof": 900,
        "mixed_curved_or_recovery": 1200,
        "nonlinear_performance_or_qv10": 1500,
        "package_isolation": 600,
    }
    assert bounds["wave_hard_wall_seconds"] == 1800
    assert max(bounds["command_hard_walls_seconds"].values()) < bounds["wave_hard_wall_seconds"]
    assert bounds["inactivity"]["hard_seconds"] == 300
    assert bounds["inactivity"]["log_volume_alone_is_progress"] is False
    assert bounds["memory_limit_gib_per_process_tree"] == 24
    assert bounds["memory_limit_bytes_per_process_tree"] == 24 * 1024**3
    assert bounds["max_concurrent_workers"] == 3
    assert bounds["numerical_library_threads_per_worker"] == 1
    assert bounds["automatic_retry_allowed"] is False
    assert bounds["request_reuse_allowed"] is False
    assert bounds["no_multiple_waves_per_command"] is True
    assert bounds["job_object"] == {
        "aggregate_memory_accounting": True,
        "assign_process_while_suspended": True,
        "fallback": "FAIL_CLOSED_BEFORE_RESUME_IF_WINDOWS_JOB_OBJECT_CONTAINMENT_IS_UNAVAILABLE",
        "kill_complete_tree_on_close": True,
        "required": True,
        "verify_zero_surviving_descendants": True,
    }
    assert bounds["evidence_binding"] == {
        "canonical_manifest_required": True,
        "canonical_result_policy": "ABSOLUTE_EXCLUSIVE_UNDER_OUTPUT_ROOT_NONALIASING",
        "formal_process_environment": "WINDOWS_JOB_OBJECT_REQUIRED",
        "prelaunch_and_finalization_bindings": [
            "ASSIGNMENT_SHA256",
            "PROGRAM_PATH_AND_SHA256",
            "PLAN_PATH_AND_SHA256",
            "STRICTLY_SORTED_UNIQUE_INPUT_PATHS_AND_SHA256",
        ],
        "scientific_envelope_exact_keys": [
            "assignment_sha256",
            "plan_sha256",
            "record_count",
            "record_ids",
            "record_ids_sha256",
            "schema",
            "scientific_payload",
            "scientific_payload_sha256",
            "selector",
            "terminal",
        ],
        "scientific_nested_hashes_recomputed": True,
        "successful_scientific_terminal": "ACCEPTED_FOR_AGGREGATION",
    }
    assert bounds["exception_policy"] == {
        "authorization": "EXPLICIT_USER_APPROVAL",
        "evidence": "PROOF_THAT_THE_WORK_CANNOT_BE_PARTITIONED",
        "finite_hard_wall_required": True,
        "new_hash_bound_authority_required": True,
        "reuse_original_request_allowed": False,
        "review": "INDEPENDENT_ACCEPTED_REVIEW",
    }


def test_advisory_margins_cannot_create_scientific_no_go() -> None:
    ratios = _load(BOUNDS_PATH)["scientific_and_advisory_ratios"]
    assert ratios == {
        "advisory_finest_error_ratio_at_25_percent": 1.35,
        "advisory_finest_error_ratio_through_10_percent": 1.15,
        "advisory_miss_disposition": "RUN_PREREGISTERED_N160_SENTINEL_AND_REQUIRE_REVIEW",
        "advisory_miss_is_scientific_no_go": False,
        "formal_finest_error_ratio_at_25_percent": 1.5,
        "formal_finest_error_ratio_through_10_percent": 1.25,
    }


def test_terminal_precedence_keeps_process_blocks_distinct_from_scientific_no_go() -> None:
    authority = _load(AUTHORITY_PATH)
    bounds = _load(BOUNDS_PATH)
    terminals = authority["terminal_precedence"]
    assert terminals[:4] == [
        "BLOCKED_E4_PL_S3_V2_BASE_ADVANCED_REBIND_REQUIRED",
        "BLOCKED_E4_PL_S3_V2_BASE_OR_PREDECESSOR_AUTHORITY",
        "BLOCKED_E4_PL_S3_V2_PUBLIC_SOURCE_OR_DERIVATION",
        "BLOCKED_E4_PL_S3_V2_PROCESS_OR_EVIDENCE",
    ]
    assert all(item.startswith("NO_GO_") for item in terminals[4:-1])
    assert terminals[-1] == "PROVISIONAL_GO_E4_PL_S3_V2_DEFAULT_ACTIVATION"
    assert authority["current_stage_disposition"] == terminals[2]
    assert bounds["process_failure_terminal"] == terminals[3]
    assert bounds["timeout_or_memory_breach_is_scientific_no_go"] is False
    assert bounds["process_limits_are_scientific_acceptance_gates"] is False
    assert authority["production_restriction"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert bounds["production_restriction"] == authority["production_restriction"]
