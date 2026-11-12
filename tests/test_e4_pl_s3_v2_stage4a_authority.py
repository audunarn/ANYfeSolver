from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json"
V10_LEDGER_REQUIREMENTS = {
    "approval_and_ledger_authority_files_regular_nonreparse_required": True,
    "frozen_pre_run_ledger_snapshot_count": 41,
    "frozen_pre_run_ledger_snapshots_byte_identical_required": True,
    "live_ledger_append_only_extension_revalidated_at_every_v5_validation": True,
}
V11_RESOURCE_REQUEST_REQUIREMENTS = {
    "resource_request_authority_files_regular_nonreparse_stable_read_required": True,
}


def _pre_v10_review_requirements(authority: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in authority["review_correction_requirements"].items()
        if key not in V10_LEDGER_REQUIREMENTS
        and key not in V11_RESOURCE_REQUEST_REQUIREMENTS
    }


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    made: dict[str, object] = {}
    for key, value in pairs:
        if key in made:
            raise ValueError(f"duplicate key: {key}")
        made[key] = value
    return made


def _canonical(value: object) -> bytes:
    def visit(item: object) -> None:
        if isinstance(item, float):
            assert math.isfinite(item)
        elif isinstance(item, list):
            for member in item:
                visit(member)
        elif isinstance(item, dict):
            for key, member in item.items():
                assert isinstance(key, str)
                visit(member)

    visit(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.fixture(scope="module")
def authority() -> dict[str, object]:
    raw = AUTHORITY.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical(value)
    return value


def test_stage4a_authority_binds_exact_parent_and_protocol(authority) -> None:
    assert set(authority) == {
        "advisory_policy",
        "allowed_extent",
        "candidate",
        "component_cache_policy",
        "correction",
        "dependency_authority",
        "execution",
        "formal_phase",
        "formal_protocol",
        "frozen_inputs",
        "parent",
        "production_boundary",
        "review_correction_requirements",
        "schema",
        "scope_base",
        "terminals",
    }
    assert authority["schema"] == "anysolver.e4-pl-s3-v2-stage4a-authority-v11"
    assert authority["parent"] == {
        "commit": "171df65eef875508effe16018875ffccf6b0f4f6",
        "subject": "docs: freeze S3 V2A Stage 4A execution",
        "tree": "e6fb4c15b70124c47615497e746ce9f5a6ed36f1",
    }
    assert authority["scope_base"] == {
        "commit": "d1f6d3d264882cc70a34b6a764476f5ec6baeb3b",
        "subject": "test: make S3 V2 evidence hashes checkout portable",
        "tree": "5fb2bfa183018e7365cb9fa5f864912fab003b97",
    }
    phase = authority["formal_phase"]
    assert phase["classifying_record_count"] == 81
    assert phase["v1_diagnostic_record_count"] == 0
    assert authority["formal_protocol"]["support"] == {
        "all_edge_translations": ["ux", "uy", "uz"],
        "constant_x_edge_rotation": "theta_x",
        "constant_y_edge_rotation": "theta_y",
        "identity": "HARD_NAVIER_TRANSLATIONS_PLUS_EDGE_ZERO_SHELL_ROTATIONS_V3",
    }
    assert authority["formal_protocol"]["reference_identity"] == (
        "INDEPENDENT_NAVIER_REISSNER_MINDLIN_UNIFORM_PRESSURE_SHELL_EMBEDDED_V3"
    )


def test_stage4a_advisory_margins_cannot_become_scientific_no_go(authority) -> None:
    advisory = authority["advisory_policy"]
    assert advisory["classification"] == "NONCLASSIFYING_INDEPENDENT_REVIEW_TRIGGER"
    assert advisory["successor_expansion_authorized_on_trigger"] is False
    assert authority["terminals"]["formal_no_go"] == (
        "NO_GO_E4_PL_S3_V2A_MIXED_FLEXURAL_CONVERGENCE"
    )


def test_stage4a_authority_preserves_defaults_and_mechanics_boundaries(authority) -> None:
    assert authority["production_boundary"] == {
        "default_s3_formulation": "legacy-s3",
        "default_s3_unchanged": True,
        "q4_default": "e4-pl",
        "q4_mechanics_unchanged": True,
        "v1_fallback_forbidden": True,
    }
    extent = authority["allowed_extent"]
    assert extent["q4_or_v1_mechanics_change_authorized"] is False
    assert extent["v2_matrix_or_equation_change_authorized"] is False
    assert "src/anyfesolver/e4_pl_element.py" not in extent["implementation_paths"]
    assert "src/anyfesolver/e4_pl_s3_element.py" not in extent["implementation_paths"]
    assert "src/anyfesolver/boundary.py" not in extent["implementation_paths"]
    assert "tests/test_e4_pl_s3_v2_component_cache.py" in extent["implementation_paths"]


def test_stage4a_authority_correction_preserves_original_and_closes_paths(authority) -> None:
    correction = authority["correction"]
    assert correction["classification"] == (
        "INDEPENDENT_REVIEW_CORRECTION_AND_HISTORICAL_TEST_SCOPE"
    )
    assert correction["mechanics_or_protocol_changed"] is True
    assert correction["production_mechanics_changed"] is False
    assert correction["scientific_protocol_corrected"] is True
    assert correction["history"][:2] == [
        {
            "bytes": 3726,
            "commit": "0f0979db7548cf0e715451d052fa837da61cbdf6",
            "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
            "reason": "INCOMPLETE_DOWNSTREAM_PATH_ENUMERATION",
            "sha256": "061BD82022D7FB6D6403057DA81389C8E59282FCD48E57CC3B2B824693AFE6B3",
            "subject": "docs: authorize S3 V2A flat mixed funnel",
            "tree": "873962fcfafbd71ff1be1240f47afa04e2acad41",
        },
        {
            "bytes": 4912,
            "commit": "9f332946ea91cc1ef45ae6776c72b7aa569f92df",
            "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
            "reason": "PREDECESSOR_BINDING_TESTS_USED_MUTABLE_WORKING_BYTES",
            "sha256": "77688BCEB41A878109360CAF299DD415C45B102361E24A138A360C15F4187C60",
            "subject": "docs: correct S3 V2A Stage 4A authority extent",
            "tree": "732e87fdf2319405659f0685ae71d333a4645269",
        },
    ]
    extent = authority["allowed_extent"]
    assert {
        "docs/reference_cases/e4_pl_s3_v2_bounded_process.py",
        "tests/test_e4_pl_s3_v2_candidate_binding.py",
        "tests/test_e4_pl_s3_v2_bounded_process.py",
        "tests/test_e4_pl_s3_v2_flat_candidate_review.py",
    } <= set(extent["implementation_paths"])
    assert correction["predecessor_v5_current_change"] == {
        "cases_changed": False,
        "classification": "PROCESS_WALL_AND_CHECKER_TREE_DRAIN_ONLY",
        "consumed_request": {
            "checker_processes_started": 0,
            "classifying_records": 0,
            "producer_processes_started": 0,
            "request_id": "3725cb19803543bfa789903b2a11f59a",
            "request_reuse_forbidden": True,
            "terminal": "BLOCKED_E4_PL_S3_V2_PROCESS_OR_EVIDENCE",
            "wave_terminal": "RESOURCE_DEFERRED",
        },
        "defaults_changed": False,
        "mechanics_changed": False,
        "process_control_changed": True,
        "protocol_changed": False,
        "tolerances_changed": False,
    }
    assert correction["predecessor_v6_current_change"] == {
        "cases_changed": False,
        "classification": (
            "EXACT_COMPONENT_CACHE_AND_CONTENT_ADDRESSED_LEAF_PROCESS_ONLY"
        ),
        "defaults_changed": False,
        "mechanics_changed": False,
        "process_control_changed": True,
        "protocol_changed": False,
        "tolerances_changed": False,
        "v2_equations_changed": False,
        "v2_matrices_changed": False,
    }
    assert correction["current_change"] == {
        "cases_changed": False,
        "classification": (
            "RESOURCE_REQUEST_AUTHORITY_AND_REMAINING_FORMAL_LEDGER_STABLE_"
            "READ_FINALIZER_PREFIX_REVALIDATION_CORRECTION_ONLY"
        ),
        "classifying_scientific_protocol_changed": False,
        "defaults_changed": False,
        "dependency_paths_changed": False,
        "finalizer_preserved_ledger_prefix_revalidation_added": True,
        "mechanics_changed": False,
        "process_or_evidence_protocol_changed": True,
        "protocol_changed": False,
        "remaining_formal_ledger_stable_reads_completed": True,
        "resource_request_authority_files_regular_nonreparse_stable_read_added": True,
        "tolerances_changed": False,
        "v1_mechanics_changed": False,
        "v2_equations_changed": False,
        "v2_matrices_changed": False,
    }
    assert correction["history"][2] == {
        "bytes": 6458,
        "commit": "3bbf5d3e85b8ba218846004a4f84a2bbaf0818a0",
        "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
        "reason": "RESOURCE_DEFERRED_WITH_ZERO_WORKER_LAUNCHES",
        "sha256": "17B7B768EDE593A600F4B37C120E0653BE3C522B2E5C62687EFCD8C02272C322",
        "subject": "docs: authorize S3 V2A Stage 4A review corrections",
        "tree": "95bdde42a9260b10cebfa59e7315d0530610c4cf",
    }
    assert correction["history"][3] == {
        "bytes": 7809,
        "commit": "3ba32964a75dfe077440692535876b3c3a6b076e",
        "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
        "reason": "INCOMPLETE_COORDINATOR_WALL_AND_CHECKER_TREE_DRAIN_GUARDS",
        "sha256": "6F5D2C676D7728EEBC36E12A27678CDE95151E2747E721F10B5DCE80FC2E4C6D",
        "subject": "fix: serialize Stage 4A resource admission",
        "tree": "5666a6d729db322b49a7ce1b242dec7c0f98b960",
    }
    assert correction["history"][4] == {
        "bytes": 8668,
        "commit": "d085c097c434db396024d06c3f113b540189b75e",
        "parent": "3ba32964a75dfe077440692535876b3c3a6b076e",
        "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
        "reason": "MONOLITHIC_SHARDS_EXCEEDED_BOUNDED_WORKER_WALL",
        "sha256": (
            "D145B76F394EC4733FB2EE2A5DD0844C49F9ED711918F53F539D81394A2F64AA"
        ),
        "subject": "fix: enforce Stage 4A hard process bounds",
        "tree": "d9208ec9d98fd9c7ac9f4156ee9dda97a1de86cb",
    }
    assert correction["history"][5] == {
        "bytes": 12266,
        "commit": "0de98307c02a5b98cb924941fee9c21f94b60d66",
        "parent": "d085c097c434db396024d06c3f113b540189b75e",
        "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
        "reason": "COMBINED_ROLE_N80_LEAVES_EXCEEDED_900_SECOND_WORKER_WALL",
        "sha256": (
            "69B75020D165EFBA3642D4575F624D99A1556B2352739E396C34754D0BC3987E"
        ),
        "subject": "fix: partition Stage 4A bounded evidence",
        "tree": "55d836332cecca7f9a38d8f5b3484adab5138fcc",
    }
    assert correction["history"][6] == {
        "bytes": 17991,
        "commit": "0470804a56b00cd15fc1aca92a5bee9151783716",
        "parent": "0de98307c02a5b98cb924941fee9c21f94b60d66",
        "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
        "reason": (
            "NONCLASSIFYING_V1_DIAGNOSTIC_EXCEEDED_1500_SECOND_WORKER_WALL_"
            "AFTER_V2_COMPLETED"
        ),
        "sha256": (
            "0B4E8ADCE4BAE49DB1F8429296C9CC246B7FB56870A3087CC6EEB08EC498B4EB"
        ),
        "subject": "fix: split Stage 4A formulation roles",
        "tree": "9833c99edb92c88d6442ac960354030bf080191c",
    }
    assert correction["review_incident"] == {
        "candidate_commit": "0f93779feded35846ce7a093d27a8a98f5c0fc81",
        "findings": [
            "BLOCKED_PROCESS_RETURNED_ZERO",
            "CANDIDATE_ARCHIVE_NOT_EXECUTED",
            "CANONICAL_PUBLICATION_NOT_ATOMIC",
            "CHECKER_PROOF_DIGESTS_NOT_JOINED_TO_PRODUCER",
            "GIT_ENGINE_AND_EXTERNAL_ATTRIBUTES_NOT_BOUND",
            "IMMUTABLE_LEDGER_SNAPSHOT_NOT_RECOMPUTED",
            "MINDLIN_SHELL_ROTATION_EMBEDDING_AND_SUPPORT_MISMATCH",
            "MIXED_Q4_NORMAL_AUTHORITY_MISSING",
            "PREDECESSOR_BINDING_TESTS_USED_MUTABLE_WORKING_BYTES",
        ],
        "p0_count": 0,
        "p1_count": 9,
        "process_review": "REJECT_PENDING_CORRECTION",
        "scientific_review": "REJECT_PENDING_CORRECTION",
    }
    assert authority["review_correction_requirements"] == {
        **V10_LEDGER_REQUIREMENTS,
        **V11_RESOURCE_REQUEST_REQUIREMENTS,
        "atomic_canonical_publication": True,
        "blocked_process_exit_nonzero": True,
        "checker_producer_digest_join": True,
        "exact_candidate_archive_execution": True,
        "git_launcher_and_engine_binding": True,
        "immutable_pre_run_ledger_snapshot": True,
        "mixed_q4_reference_normal": ["0", "0", "1"],
        "shell_embedding": {"beta_x": "theta_y", "beta_y": "-theta_x"},
    }
    assert set(extent["freeze_authorization_paths"]) == {
        "docs/reference_cases/e4_pl_s3_v2_stage4a_contract.json",
        "docs/reference_cases/e4_pl_s3_v2_stage4a_execution_authorization.json",
        "docs/reference_cases/e4_pl_s3_v2_stage4a_leaf_wave_authorization.json",
        "docs/reference_cases/e4_pl_s3_v2_stage4a_process_implementation_review.json",
        "docs/reference_cases/e4_pl_s3_v2_stage4a_scientific_implementation_review.json",
    }
    assert set(extent["outcome_paths"]) == {
        "docs/reference_cases/e4_pl_s3_v2_stage4a_evidence.json",
        "docs/reference_cases/e4_pl_s3_v2_stage4a_result.json",
        "docs/reference_cases/e4_pl_s3_v2_stage4a_scientific_review.json",
        "docs/reference_cases/e4_pl_s3_v2_stage4a_status.json",
        "tests/test_e4_pl_s3_v2_stage4a_closeout.py",
    }


def test_stage4a_correction4_exact_component_cache_policy(authority) -> None:
    assert authority["component_cache_policy"] == {
        "capacity": 512,
        "closure_owned": True,
        "deterministic_eviction": (
            "RETAIN_LEXICOGRAPHICALLY_SMALLEST_EXACT_CONTENT_KEYS"
        ),
        "fresh_writable_copies_on_every_return": True,
        "guard_before_lookup": True,
        "key": "FULL_VALIDATED_NUMERIC_CONTENT_AND_FROZEN_FORMULATION_IDENTITY",
        "key_validation": ["DTYPE", "FINITE", "SHAPE", "VALUE"],
        "lock": "THREADING_RLOCK",
        "pl_and_numerical_outputs_distinct": True,
        "v2_equations_or_matrices_changed": False,
    }


def test_stage4a_correction3_consumed_timeout_incident_is_exact(authority) -> None:
    assert authority["correction"]["correction3_incident"] == {
        "canonical_aggregate": {
            "byte_count": 681,
            "sha256": (
                "79E58C76EDE5DEEBC84C98014333FBBAEE7DB6A0A4BC978374138EAB4BEB42DB"
            ),
        },
        "classifying_record_count": 0,
        "completed_partial_records": {
            "alternating": {"completed": 17, "registered": 27},
            "backslash": {"completed": 16, "registered": 27},
            "slash": {"completed": 16, "registered": 27},
        },
        "partial_records_classification": "NONCLASSIFYING",
        "producer_result": {
            "byte_count": 7476,
            "sha256": (
                "23E56EF27DBF81C80EF79F1A888E89FCFF0F64787AF9892E2497ABA6A93BAB8A"
            ),
        },
        "producer_workers": {"count": 3, "terminal_counts": {"TIMEOUT": 3}},
        "request": {
            "request_id": "da3f0b058f124d8488dcd8bae341ac22",
            "request_reuse_forbidden": True,
        },
        "scientific_checker_count": 0,
        "scientific_proof_count": 0,
        "terminal": "BLOCKED_E4_PL_S3_V2_PROCESS_OR_EVIDENCE",
        "transcript": {
            "byte_count": 4526,
            "sha256": (
                "9C3214F6554D362CB881C1A5BBFC62156A8EE5B2E4933B34EE05BC71D46F617E"
            ),
        },
    }


def test_stage4a_correction4_nonclassifying_calibration_is_exact(authority) -> None:
    calibration = authority["correction"]["correction4_calibration"]
    assert calibration["disposition"] == "ROLE_SPLIT_REQUIRED_BEFORE_FORMAL_EXECUTION"
    assert calibration["earlier_requests"] == [
        {
            "disposition": "MALFORMED_COMMAND_NEVER_APPROVED_OR_EXECUTED",
            "request": {
                "byte_count": 1187,
                "request_id": "d6fe49a58b3444a092ae64a734a7a31a",
                "request_reuse_forbidden": True,
                "sha256": (
                    "3B75A3FDC7A1DF40E9B6FA7EA124F52E8BF3CF253321245B63FE74712DB7D8B3"
                ),
            },
            "terminal": "CANCELLED_UNAPPROVED",
        },
        {
            "disposition": "INVALID_PILOT_GIT_ENVIRONMENT_BEFORE_OUTPUT",
            "lock_released": True,
            "output_root_present": False,
            "request": {
                "byte_count": 1264,
                "command_sha256": (
                    "F01956582A71F19A9BCB0A73367BBF3675BF552251C4406C4F3FD63542BB657F"
                ),
                "request_id": "36380f3acd8c43e48a17eb0ac3eaf4ea",
                "request_reuse_forbidden": True,
                "sha256": (
                    "64D68EEEAA1C95877E2A1BBC528CE66C6F2575E06D40E5D5745FA1FC5290D26D"
                ),
            },
            "terminal": "COMPLETED_FAIL",
            "transcript": {
                "byte_count": 954,
                "sha256": (
                    "D3233B391C0D020C9E642F7CD07E3C84F03DBAE228A26A342336544098E68A80"
                ),
            },
        },
    ]
    assert calibration["n20_success"] == {
        "bounded_result": {
            "byte_count": 3120,
            "sha256": (
                "4F595234B5E8C27D388B5BF36DBE7616DD2F58037A8CE76EC1741CF2CC1AB4C2"
            ),
        },
        "peak_tree_memory_bytes": 332_537_856,
        "request": {
            "byte_count": 1277,
            "command_sha256": (
                "028FBA2C16E80C46A691C73E866F82CAB1D05475AA6DC14E66D0ADB27BC851CA"
            ),
            "request_id": "f5a6be8595484898ba566af5d91aa483",
            "request_reuse_forbidden": True,
            "sha256": (
                "57793FB7E6D2F352F8795288A6A18B32398F3CB3235B554A852834776C20A66C"
            ),
        },
        "scientific_classification": "NONCLASSIFYING_RUNTIME_CALIBRATION_ONLY",
        "summary": {
            "byte_count": 3716,
            "sha256": (
                "D19C6160A424518A909851D97ED42250FB5C949A52AA7ABDCABE6401930E5DF0"
            ),
        },
        "terminal": "COMPLETED_PASS",
        "termination_proven": True,
        "transcript": {
            "byte_count": 3717,
            "sha256": (
                "CF18AB52FE057FF712A642003D5E530C12F27E111A2700852ED36D352DFC850B"
            ),
        },
        "wall_seconds": "29.683154",
    }
    assert calibration["n80_timeout"] == {
        "bounded_result": {
            "byte_count": 5539,
            "sha256": (
                "5ECD12B629D320E076A2DA9252A6DCA9EA4D37FA40EECA1E271365D5FBAF02CE"
            ),
        },
        "last_progress_sequence": [1, 1],
        "peak_tree_memory_bytes": [1_937_715_200, 1_935_978_496],
        "request": {
            "byte_count": 1254,
            "command_sha256": (
                "2B7D302AD348F3FC3BFC8A31A8DE1FB3C33F05EF165F2B6455CE36D140BBFAF1"
            ),
            "request_id": "dc08835483b747d3afd3397fb35ef46d",
            "request_reuse_forbidden": True,
            "sha256": (
                "C2F2C206486E1CF043B67D8340F674E3C5650FA1DB9956EE6164F13EC53315E2"
            ),
        },
        "scientific_classification": "NONCLASSIFYING_RUNTIME_CALIBRATION_ONLY",
        "scientific_proof_count": 0,
        "summary": {
            "byte_count": 6301,
            "sha256": (
                "1A9D1BE927D71B38E3CE3F510CB9A1158B4930F18ACFA0188A2364ADFC23B781"
            ),
        },
        "terminal": "COMPLETED_FAIL",
        "termination_proven": [True, True],
        "transcript": {
            "byte_count": 6302,
            "sha256": (
                "1F01661205F5FD3A9EED0CB71ECCB57FD591EBAD94B7D4C2124665E87F3B51AB"
            ),
        },
        "wall_seconds": "900.872125",
        "worker_terminal_counts": {"TIMEOUT": 2},
    }
    assert calibration["canceled_never_run"] == [
        {
            "byte_count": 1266,
            "command_sha256": (
                "A0DC4A08E8F60DD351CDC09A3768125886C1F9A92C1ED78A3EE6242DEE1038BB"
            ),
            "request_id": "92297500432c46c9ad46523232817d9d",
            "request_reuse_forbidden": True,
            "sha256": (
                "0C370444B534C58D2C818A53FB3306522C95A8E3F0F206834D6FCDEE59062F57"
            ),
            "terminal": "CANCELLED_NOT_RUN",
        },
        {
            "byte_count": 1272,
            "command_sha256": (
                "89ACD2D229474CFA3A68F9CEC93C40BAAD442642499DEC2DC1B11B91BC35901A"
            ),
            "request_id": "96bf7d1d6d514e14b9b0a6da3307ec0e",
            "request_reuse_forbidden": True,
            "sha256": (
                "00A0DC143E30FCC11077935E463DA224D236A1E339C073339E7A91D3B5E23D2A"
            ),
            "terminal": "CANCELLED_NOT_RUN",
        },
    ]


def test_stage4a_correction5_nonclassifying_calibration_is_exact(authority) -> None:
    calibration = authority["correction"]["correction5_calibration"]
    assert calibration["disposition"] == (
        "HISTORICAL_V1_COMPARATOR_EXCLUDED_FROM_FORMAL_RUNTIME_NO_FALLBACK"
    )
    assert calibration["n20_role_equivalence"] == {
        "bounded_result": {
            "byte_count": 5962,
            "sha256": (
                "92BFCFECFF0DA5D728FFA71AD35D7EE84C9643052E74064F087891FEDE3CCB90"
            ),
        },
        "equivalence": {
            "byte_count": 2502,
            "disposition": "NONCLASSIFYING_RUNTIME_EQUIVALENCE_ONLY",
            "sha256": (
                "83352E5121E4B925321B3B6AA8575EB24BF6BF496545B174F77FC04DAED9B1F1"
            ),
            "terminal": "PASS_EXACT_INNER_RECORD_EQUIVALENCE",
        },
        "pilot_runner": {
            "byte_count": 10495,
            "sha256": (
                "441DC8E39871A2836513B7A7CF10A3FB4AA33B7A983DCA3217180B372C254539"
            ),
        },
        "proofs": {
            "v1_diagnostic": {
                "byte_count": 3652,
                "inner_record": {
                    "byte_count": 1570,
                    "correction4_sha256": (
                        "4DF1908BBD3C7BCE71FB050975A0D514AF88802CD2B1729E16021B5AD3CC4DA6"
                    ),
                    "sha256": (
                        "4DF1908BBD3C7BCE71FB050975A0D514AF88802CD2B1729E16021B5AD3CC4DA6"
                    ),
                },
                "sha256": (
                    "A2201E843CD389CCD0A7906DD9AB7DCD92DB303CA18A725EBCD1604F3B3C2A45"
                ),
                "status": "COMPLETED",
            },
            "v2_classifying": {
                "byte_count": 3615,
                "inner_record": {
                    "byte_count": 1528,
                    "correction4_sha256": (
                        "682353D9B6849CF05AE3C61BB6F545D753C5C82B07128115FB3A959123A65FD5"
                    ),
                    "sha256": (
                        "682353D9B6849CF05AE3C61BB6F545D753C5C82B07128115FB3A959123A65FD5"
                    ),
                },
                "sha256": (
                    "D1C6D1B44B10249A9DB423E0BA5ED64942FAD68EBA1C531400A065237240D5F9"
                ),
                "status": "COMPLETED",
            },
        },
        "request": {
            "byte_count": 1292,
            "command_sha256": (
                "4ACF18BA8E9A3E946641264D818BC96F536D2A937BBCB1F340F084E08DF9506C"
            ),
            "request_id": "91d5550d3b18459d80d1c6ce157de056",
            "request_reuse_forbidden": True,
            "sha256": (
                "A16711471E99C1B524D9C2E213BFC0E47BA6736F40C67985EC2A0C58E2739267"
            ),
        },
        "scientific_classification": "NONCLASSIFYING_RUNTIME_CALIBRATION_ONLY",
        "summary": {
            "byte_count": 2730,
            "sha256": (
                "9FADA46C663CF2A7D753DE36B7AC080DC16B618092152277ACFD8B109628AB83"
            ),
        },
        "terminal": "COMPLETED_PASS",
        "termination_proven": [True, True],
        "transcript": {
            "byte_count": 2731,
            "sha256": (
                "05CD6295399281AD032659856809781FB5397023767C8444395D6EE22645AB8A"
            ),
        },
        "wall_seconds": "26.272784",
    }
    assert calibration["n80_v1_timeout"] == {
        "bounded_result": {
            "byte_count": 5803,
            "sha256": (
                "274933EEE33FF8B50E5C88835C7A24506E6255913F2B892792EFB8A0C1C80826"
            ),
            "terminal": "BLOCKED_E4_PL_S3_V2_PROCESS_OR_EVIDENCE",
        },
        "manifest": {
            "byte_count": 10736,
            "sha256": (
                "2111BB7F2740D218F272FBC89AE942AA1616202223F14D7B7FC977EF1287A8A0"
            ),
        },
        "pilot_runner": {
            "byte_count": 10495,
            "sha256": (
                "441DC8E39871A2836513B7A7CF10A3FB4AA33B7A983DCA3217180B372C254539"
            ),
        },
        "record_id": "N80:25PCT:dispersed:slash",
        "request": {
            "byte_count": 1291,
            "command_sha256": (
                "61895014712DBCD31CDD69C21444B0857E4E3311A69E7FE4BB08817D3723C196"
            ),
            "request_id": "f016d9a4d0a9429bb09517a29998046a",
            "request_reuse_forbidden": True,
            "sha256": (
                "618E9051C9E1830A81551C2D5E9A11305E7888B3A8B9791238A3C014128ECE69"
            ),
        },
        "scientific_classification": "NONCLASSIFYING_RUNTIME_CALIBRATION_ONLY",
        "summary": {
            "byte_count": 2709,
            "sha256": (
                "509C1CF2D135A024D50BC2BB92FCA2A42030ED9C426B76DC4BBB9453746E2D1F"
            ),
        },
        "terminal": "COMPLETED_FAIL",
        "transcript": {
            "byte_count": 2710,
            "sha256": (
                "2966E53C7FC0963BC39813B2571E578CFC2E2F7F88EF2097ECE91AC5BAE26205"
            ),
        },
        "wall_seconds": "1500.962516",
        "workers": {
            "v1_diagnostic": {
                "cpu_100ns": 14_869_687_500,
                "last_progress_sequence": 1,
                "peak_tree_memory_bytes": 964_890_624,
                "returncode": 124,
                "scientific_proof_count": 0,
                "status": "TIMEOUT",
                "termination_proven": True,
            },
            "v2_classifying": {
                "cpu_100ns": 597_500_000,
                "last_progress_sequence": 5,
                "peak_tree_memory_bytes": 1_954_713_600,
                "proof": {
                    "byte_count": 3625,
                    "sha256": (
                        "9A765D7B44D55736A75017660D7BA9C790A6B451A1FC8BAEF84F7C7F8E89EBC9"
                    ),
                },
                "returncode": 0,
                "status": "COMPLETED",
                "termination_proven": True,
            },
        },
    }


def test_stage4a_correction6_calibration_and_monitor_incident_are_nonclassifying(
    authority,
) -> None:
    calibration = authority["correction"]["correction6_calibration"]
    assert calibration["disposition"] == "USABLE_FOR_FORMAL_AUTHORITY_PREPARATION_ONLY"
    assert calibration["formal_evidence_eligible"] is False
    assert calibration["no_scientific_adjudication"] is True
    assert calibration["summary"] == {
        "byte_count": 5977,
        "sha256": "5871DE245CDF2C91C0E27A1F2688F8E2E0B5043A030E7E904B46EA4072B7C993",
        "terminal": "COMPLETED_NONCLASSIFYING_CALIBRATION",
    }
    assert calibration["bounded_result"] == {
        "byte_count": 57435,
        "sha256": "5CCCFCBEC1A7C6343C8F50360EEE6F9FE1E1622FDFD2CDC4DE02ECC8AAE34DFF",
    }
    assert calibration["incident"] == {
        "byte_count": 4553,
        "ledger_chain": {
            "final_ledger": {
                "byte_count": 117260,
                "line_count": 214,
                "sha256": (
                    "386CA9114C7F21A6FF940FCCE87E7CA5D82832624A01DA055A369ADBB63EA676"
                ),
            },
            "ordered_rows": [
                {
                    "byte_count": 992,
                    "preceding_ledger": {
                        "byte_count": 113842,
                        "line_count": 210,
                        "sha256": (
                            "C65CC8D5BD15EBA13D03D9897174EAB58102714EC854B2739B4829B72AE9E84A"
                        ),
                    },
                    "recorded_incident_byte_count": 4339,
                    "sha256": (
                        "EAC114403ACC354C87A857610BC6595C97949D2490422A0797DE91003F146090"
                    ),
                    "status": "COMPLETED_WITH_MONITOR_BOOKKEEPING_INCIDENT",
                },
                {
                    "actual_preceding_ledger": {
                        "byte_count": 114834,
                        "line_count": 211,
                        "sha256": (
                            "4848DC1CAFEDAED7EA381FF5DDC031D4CA29E03E50023E06B8BBBA70E838783D"
                        ),
                    },
                    "byte_count": 648,
                    "corrected_incident_byte_count": 4553,
                    "recorded_preceding_ledger_byte_count": 114959,
                    "sha256": (
                        "1E67813A47EB9FDB319E3E129C437D43F2AAE29F21BE9BE1B389B3B15B64EE64"
                    ),
                    "status": "LEDGER_BOOKKEEPING_CORRECTION",
                },
                {
                    "actual_row_212_sha256": (
                        "1E67813A47EB9FDB319E3E129C437D43F2AAE29F21BE9BE1B389B3B15B64EE64"
                    ),
                    "byte_count": 890,
                    "preceding_ledger": {
                        "byte_count": 115482,
                        "line_count": 212,
                        "sha256": (
                            "55B7F5CEBB48D66605278BD66BA4EC3EEF5D231F76CF2E6F7464C4F50ED763D9"
                        ),
                    },
                    "recorded_row_212_sha256": (
                        "1E67813A2B1759016C69055F4EC09E0ED04C4E8010A956701714CC8D3AA5BAA4"
                    ),
                    "sha256": (
                        "AC20F999AB9174BD0D7B728ABC7A2EC5635DBC2C9EE35F8B40AA05729AC58C74"
                    ),
                    "status": "LEDGER_BOOKKEEPING_CORRECTION",
                },
                {
                    "byte_count": 888,
                    "corrected_row_212_sha256": (
                        "1E67813A47EB9FDB319E3E129C437D43F2AAE29F21BE9BE1B389B3B15B64EE64"
                    ),
                    "preceding_ledger": {
                        "byte_count": 116372,
                        "line_count": 213,
                        "sha256": (
                            "66633828C6BF1B6EA31D17E001C646819718E17A0226A5AB58BD7E43546D913D"
                        ),
                    },
                    "sha256": (
                        "EA92545767989F1183A2649435CCA8C5E5D54D56E8CE6F5FBBFEE7AC9A2C1B40"
                    ),
                    "status": "LEDGER_BOOKKEEPING_CORRECTION",
                },
            ],
            "request_id": "4e8ea97778d543babba5586ba74bc4fb",
        },
        "ledger_terminal": "COMPLETED_WITH_MONITOR_BOOKKEEPING_INCIDENT",
        "path": (
            "C:\\Users\\AudunArnesenNyhus\\AppData\\Local\\ANYrelease\\"
            "s3-v2-stage4a-c6-calibration-incidents\\"
            "4e8ea97778d543babba5586ba74bc4fb-monitor-incident.json"
        ),
        "resource_execution_clean_pass": False,
        "sha256": "47C5A1217BEDFC0B650DA933D4FF3294793AA39C3E855BCCC9325CAF24FECBC6",
    }
    assert calibration["request"]["request_id"] == (
        "4e8ea97778d543babba5586ba74bc4fb"
    )
    assert calibration["request"]["request_reuse_forbidden"] is True
    assert calibration["v1_live_worker_count"] == 0
    assert calibration["wall_seconds"] == "84.376511"
    assert calibration["worker_termination_proven"] == [True, True]
    chain = calibration["incident"]["ledger_chain"]
    rows = chain["ordered_rows"]
    prefixes = [
        rows[0]["preceding_ledger"],
        rows[1]["actual_preceding_ledger"],
        rows[2]["preceding_ledger"],
        rows[3]["preceding_ledger"],
    ]
    for index in range(1, len(prefixes)):
        assert prefixes[index]["byte_count"] == (
            prefixes[index - 1]["byte_count"] + rows[index - 1]["byte_count"]
        )
        assert prefixes[index]["line_count"] == (
            prefixes[index - 1]["line_count"] + 1
        )
    assert chain["final_ledger"]["byte_count"] == (
        prefixes[-1]["byte_count"] + rows[-1]["byte_count"]
    )
    assert chain["final_ledger"]["line_count"] == prefixes[-1]["line_count"] + 1
    assert rows[1]["corrected_incident_byte_count"] == calibration["incident"][
        "byte_count"
    ]
    assert rows[1]["recorded_preceding_ledger_byte_count"] != rows[1][
        "actual_preceding_ledger"
    ]["byte_count"]
    assert rows[2]["recorded_row_212_sha256"] != rows[2][
        "actual_row_212_sha256"
    ]


def test_stage4a_correction7_dependency_authority_is_exact(authority) -> None:
    assert authority["dependency_authority"] == {
        "combined_graph": {
            "byte_count": 24730,
            "file_count": 126,
            "sha256": (
                "FDB2585F82394FBC4E631E7F4960FFEE800DDF8485FE7B3574DA9BB90ABA77D5"
            ),
        },
        "entries": [
            {
                "commit": "de2be5819dd07b9ad9c02d87335d61482808bcec",
                "name": "ANYmaterial",
                "path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anymaterial-dependency"
                ),
                "source_file_count": 15,
                "source_graph_sha256": (
                    "AA316B42C0D4053F4A468C2D453EA6BFAC8EA7A17ABC1F7F0BFDD4F4F343C109"
                ),
                "source_path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anymaterial-dependency\\src"
                ),
                "tree": "e1e75423f623e9aad787bf584b3f0c9a6fe2360a",
            },
            {
                "commit": "b2c4df0892025963623201b27e45a1e9af615d8e",
                "name": "ANYgeometry",
                "path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anygeometry-dependency"
                ),
                "source_file_count": 37,
                "source_graph_sha256": (
                    "60AEC025D7659B5714A9588150B1EFABF74DA3398FAA9297420BB2E1D0A29149"
                ),
                "source_path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anygeometry-dependency\\src"
                ),
                "tree": "8ee3a63ade1127f45245bf9f0b1c442717ec34d5",
            },
            {
                "commit": "ffeb538f6f1422cd7c8f3f2b86610c29dc26e078",
                "name": "ANYmesh",
                "path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anymesh-dependency"
                ),
                "source_file_count": 50,
                "source_graph_sha256": (
                    "7226109CD4986C4F18828F72B0DAA80EEDD37DA86AC9FEBB908CA4779F21F324"
                ),
                "source_path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anymesh-dependency\\src"
                ),
                "tree": "c6092a1fce57a7182d647c34d393838ef700a846",
            },
            {
                "commit": "9b1e5adea77a20155bbc23866af8c9aad853ddfd",
                "name": "ANYfileIO",
                "path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anyfileio-dependency"
                ),
                "source_file_count": 24,
                "source_graph_sha256": (
                    "FCDECDEFAF7E1DC370287215D6C2EE705332C721360C0D95B477E6F003F9512D"
                ),
                "source_path": (
                    "C:\\Github\\ANYfeSolver\\.perf2-worktrees\\"
                    "s3-v2-stage4a-anyfileio-dependency\\src"
                ),
                "tree": "70b406be2574adceab4a7b688c0e489e0937df5d",
            },
        ],
        "ignored_source_files_forbidden": True,
        "reparse_or_symlink_source_files_forbidden": True,
        "schema": "anysolver.e4-pl-s3-v2-stage4a-dependency-authority-v1",
    }
    entries = authority["dependency_authority"]["entries"]
    assert [
        {
            key: entry[key]
            for key in ("commit", "name", "path", "tree")
        }
        for entry in entries
    ] == authority["correction"]["correction6_calibration"]["dependency_graph"]
    assert sum(entry["source_file_count"] for entry in entries) == 126


def test_stage4a_correction8_live_ledger_authority_is_exact(authority) -> None:
    correction = authority["correction"]
    assert correction["history"][8] == {
        "bytes": 34312,
        "commit": "cdb8a04991943e6f0e4ff3b8c3afc389cb1bb776",
        "parent": "3760372cf88dfe37c8a74c893d7cf09c7b74b72b",
        "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
        "reason": "LIVE_LEDGER_APPEND_ONLY_PREFIX_NOT_REVALIDATED",
        "sha256": (
            "39E28E73E4EC7AF59340574B18CEEBDAFF1915A140155456A03BDB2B96798D0D"
        ),
        "subject": "fix: bind calibrated Stage 4A dependency graph",
        "tree": "1cc0a77641a6fc6260a9533343eb8e14454bf489",
    }
    assert len(correction["history"]) == 10
    assert correction["predecessor_v9_current_change"] == {
        "cases_changed": False,
        "classification": "CALIBRATED_CLEAN_DEPENDENCY_AUTHORITY_ONLY",
        "classifying_scientific_protocol_changed": False,
        "defaults_changed": False,
        "dependency_paths_changed": True,
        "mechanics_changed": False,
        "process_or_evidence_protocol_changed": True,
        "protocol_changed": False,
        "tolerances_changed": False,
        "v1_mechanics_changed": False,
        "v2_equations_changed": False,
        "v2_matrices_changed": False,
    }
    assert correction["predecessor_v9_execution"] == {
        key: value
        for key, value in authority["execution"].items()
        if key not in V10_LEDGER_REQUIREMENTS
        and key not in V11_RESOURCE_REQUEST_REQUIREMENTS
    }
    for section in (
        authority["execution"],
        authority["review_correction_requirements"],
    ):
        assert {key: section[key] for key in V10_LEDGER_REQUIREMENTS} == (
            V10_LEDGER_REQUIREMENTS
        )
    assert correction["current_change"]["cases_changed"] is False
    assert correction["current_change"]["mechanics_changed"] is False
    assert correction["current_change"]["tolerances_changed"] is False
    assert correction["current_change"]["defaults_changed"] is False
    assert correction["current_change"]["protocol_changed"] is False
    assert correction["current_change"]["process_or_evidence_protocol_changed"] is True


def test_stage4a_correction9_resource_request_authority_is_exact(authority) -> None:
    correction = authority["correction"]
    assert correction["history"][9] == {
        "bytes": 37790,
        "commit": "f0e8dd207ca3ec860a1becf291be25e073e06813",
        "parent": "cdb8a04991943e6f0e4ff3b8c3afc389cb1bb776",
        "path": "docs/reference_cases/e4_pl_s3_v2_stage4a_authority.json",
        "reason": "RESOURCE_REQUEST_FILE_NOT_STABLY_READ",
        "sha256": (
            "BD53B31FD6AD028DBFA4D05DEBF89D98A3E5C4C567597D5265E8C5FC1CC04E04"
        ),
        "subject": "fix: bind append-only Stage 4A ledger authority",
        "tree": "fe9d62aaa275b58ec3dcbb4130330671df049ba0",
    }
    assert len(correction["history"]) == 10
    assert correction["predecessor_v10_current_change"] == {
        "cases_changed": False,
        "classification": (
            "LIVE_LEDGER_APPEND_ONLY_PREFIX_AND_REGULAR_NONREPARSE_"
            "BINDING_CORRECTION_ONLY"
        ),
        "classifying_scientific_protocol_changed": False,
        "defaults_changed": False,
        "dependency_paths_changed": False,
        "live_ledger_append_only_prefix_revalidation_added": True,
        "mechanics_changed": False,
        "process_or_evidence_protocol_changed": True,
        "protocol_changed": False,
        "regular_nonreparse_external_binding_added": True,
        "tolerances_changed": False,
        "v1_mechanics_changed": False,
        "v2_equations_changed": False,
        "v2_matrices_changed": False,
    }
    assert correction["predecessor_v10_execution"] == {
        key: value
        for key, value in authority["execution"].items()
        if key not in V11_RESOURCE_REQUEST_REQUIREMENTS
    }
    assert correction["predecessor_v10_review_correction_requirements"] == {
        key: value
        for key, value in authority["review_correction_requirements"].items()
        if key not in V11_RESOURCE_REQUEST_REQUIREMENTS
    }
    assert correction["current_change"] == {
        "cases_changed": False,
        "classification": (
            "RESOURCE_REQUEST_AUTHORITY_AND_REMAINING_FORMAL_LEDGER_STABLE_"
            "READ_FINALIZER_PREFIX_REVALIDATION_CORRECTION_ONLY"
        ),
        "classifying_scientific_protocol_changed": False,
        "defaults_changed": False,
        "dependency_paths_changed": False,
        "finalizer_preserved_ledger_prefix_revalidation_added": True,
        "mechanics_changed": False,
        "process_or_evidence_protocol_changed": True,
        "protocol_changed": False,
        "remaining_formal_ledger_stable_reads_completed": True,
        "resource_request_authority_files_regular_nonreparse_stable_read_added": True,
        "tolerances_changed": False,
        "v1_mechanics_changed": False,
        "v2_equations_changed": False,
        "v2_matrices_changed": False,
    }
    for section in (
        authority["execution"],
        authority["review_correction_requirements"],
    ):
        assert {
            key: section[key] for key in V11_RESOURCE_REQUEST_REQUIREMENTS
        } == V11_RESOURCE_REQUEST_REQUIREMENTS


def test_stage4a_correction6_v2_only_leaf_bounds(authority) -> None:
    assert authority["execution"] == {
        **V10_LEDGER_REQUIREMENTS,
        **V11_RESOURCE_REQUEST_REQUIREMENTS,
        "all_launched_process_terminals_bound": True,
        "candidate_authority_bound_in_every_leaf": True,
        "canonical_aggregate_requires_complete_leaf_union": True,
        "canonical_aggregate_requires_proven_empty_process_trees": True,
        "checker_replica_wall_seconds": 300,
        "checker_tree_drain_required_before_queue_advance": True,
        "classifying_record_count": 81,
        "common_leaf_wave_authorization_required": True,
        "complete_receipt_coverage_required": True,
        "computation_leaf_count": 81,
        "consumed_request_reuse_forbidden": True,
        "content_addressed_assignments": True,
        "diagnostic_record_count": 0,
        "finalizer_accepts_complete_receipt_bound_union_only": True,
        "finalizer_input_hash_revalidated_before_publication": True,
        "finalizer_wall_seconds": 1740,
        "incomplete_or_partial_leaf_classification": "NONCLASSIFYING",
        "leaf_roles": ["V2_CLASSIFYING"],
        "leaf_worker_wall_seconds": 1500,
        "live_v1_diagnostic_execution_authorized": False,
        "maximum_concurrent_workers": 2,
        "maximum_memory_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "pair_wave_count": 40,
        "pairing_policy": "CONSECUTIVE_V2_CLASSIFYING_LEAVES_IN_FROZEN_ORDER",
        "process_tree_termination_proven_required": True,
        "receipt_binding": [
            "ATTEMPT",
            "AUTHORIZATION",
            "CANDIDATE_ARCHIVE_SHA256",
            "CANDIDATE_COMMIT",
            "CANDIDATE_TREE",
            "CONTRACT",
            "LEAF_CATALOG_SHA256",
            "LEAF_MANIFEST",
            "LEAF_SCIENTIFIC_HASHES",
            "PLAN_SHA256",
            "PRODUCER_PROGRAM_SHA256",
            "REQUEST_COMMAND_SHA256",
            "REQUEST_ID",
            "REQUEST_SHA256",
            "RESULT_SHA256",
            "TERMINATION_PROVEN",
            "WORKER_TERMINALS",
        ],
        "receipt_required_per_wave": True,
        "registered_wave_count": 41,
        "required_receipt_count": 41,
        "role_split_required": False,
        "schedule": (
            "40_CONSECUTIVE_TWO_V2_WORKER_PAIRS_THEN_1_V2_SINGLETON_IN_FROZEN_ORDER"
        ),
        "singleton_wave_count": 1,
        "v1_comparator_disposition": (
            "HISTORICAL_V1_COMPARATOR_EXCLUDED_FROM_FORMAL_RUNTIME_NO_FALLBACK"
        ),
        "v1_diagnostic_leaf_count": 0,
        "v1_diagnostic_never_classifies_or_falls_back": True,
        "v2_classifying_leaf_count": 81,
        "wave_wall_seconds": 1740,
    }
    assert authority["correction"]["predecessor_v5_execution"][
        "wave_wall_seconds"
    ] == 1800
    assert authority["execution"]["leaf_worker_wall_seconds"] < 30 * 60
    assert authority["execution"]["wave_wall_seconds"] < 30 * 60
    assert authority["execution"]["finalizer_wall_seconds"] < 30 * 60
    assert authority["execution"]["computation_leaf_count"] == (
        authority["execution"]["v2_classifying_leaf_count"]
        + authority["execution"]["v1_diagnostic_leaf_count"]
    )
    assert authority["execution"]["registered_wave_count"] == (
        authority["execution"]["pair_wave_count"]
        + authority["execution"]["singleton_wave_count"]
    )


def test_stage4a_prior_v3_authority_binding_is_exact(authority) -> None:
    prior = authority["correction"]["history"][2]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree) == (prior["subject"], prior["tree"])


def test_stage4a_prior_v4_authority_binding_is_exact(authority) -> None:
    prior = authority["correction"]["history"][3]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree) == (prior["subject"], prior["tree"])


def test_stage4a_prior_v5_authority_binding_and_history_are_exact(authority) -> None:
    prior = authority["correction"]["history"][4]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predecessor = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical(predecessor)
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree, parent) == (
        prior["subject"],
        prior["tree"],
        prior["parent"],
    )
    assert predecessor["schema"] == "anysolver.e4-pl-s3-v2-stage4a-authority-v5"
    assert predecessor["correction"]["history"] == authority["correction"][
        "history"
    ][:4]
    assert predecessor["correction"]["current_change"] == authority["correction"][
        "predecessor_v5_current_change"
    ]
    assert predecessor["execution"] == authority["correction"][
        "predecessor_v5_execution"
    ]
    for unchanged in (
        "advisory_policy",
        "formal_protocol",
        "frozen_inputs",
        "parent",
        "production_boundary",
        "scope_base",
        "terminals",
    ):
        assert predecessor[unchanged] == authority[unchanged]
    assert predecessor["review_correction_requirements"] == (
        _pre_v10_review_requirements(authority)
    )
    current_candidate = dict(authority["candidate"])
    assert current_candidate.pop("v1_formal_runtime_disposition") == (
        "HISTORICAL_V1_COMPARATOR_EXCLUDED_FROM_FORMAL_RUNTIME_NO_FALLBACK"
    )
    assert predecessor["candidate"] == current_candidate
    assert predecessor["formal_phase"] == authority["correction"][
        "predecessor_v7_formal_phase"
    ]


def test_stage4a_prior_v6_authority_binding_and_history_are_exact(authority) -> None:
    prior = authority["correction"]["history"][5]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predecessor = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical(predecessor)
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree, parent) == (
        prior["subject"],
        prior["tree"],
        prior["parent"],
    )
    assert predecessor["schema"] == "anysolver.e4-pl-s3-v2-stage4a-authority-v6"
    assert predecessor["correction"]["history"] == authority["correction"][
        "history"
    ][:5]
    assert predecessor["correction"]["current_change"] == authority["correction"][
        "predecessor_v6_current_change"
    ]
    assert predecessor["execution"] == authority["correction"][
        "predecessor_v6_execution"
    ]
    for unchanged in (
        "advisory_policy",
        "component_cache_policy",
        "formal_protocol",
        "frozen_inputs",
        "parent",
        "production_boundary",
        "scope_base",
        "terminals",
    ):
        assert predecessor[unchanged] == authority[unchanged]
    assert predecessor["review_correction_requirements"] == (
        _pre_v10_review_requirements(authority)
    )
    current_candidate = dict(authority["candidate"])
    current_candidate.pop("v1_formal_runtime_disposition")
    assert predecessor["candidate"] == current_candidate
    assert predecessor["formal_phase"] == authority["correction"][
        "predecessor_v7_formal_phase"
    ]
    predecessor_paths = set(predecessor["allowed_extent"]["freeze_authorization_paths"])
    current_paths = set(authority["allowed_extent"]["freeze_authorization_paths"])
    assert current_paths - predecessor_paths == {
        "docs/reference_cases/e4_pl_s3_v2_stage4a_leaf_wave_authorization.json"
    }


def test_stage4a_prior_v7_authority_binding_and_history_are_exact(authority) -> None:
    prior = authority["correction"]["history"][6]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predecessor = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical(predecessor)
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree, parent) == (
        prior["subject"],
        prior["tree"],
        prior["parent"],
    )
    assert predecessor["schema"] == "anysolver.e4-pl-s3-v2-stage4a-authority-v7"
    assert predecessor["correction"]["history"] == authority["correction"][
        "history"
    ][:6]
    assert predecessor["correction"]["current_change"] == authority["correction"][
        "predecessor_v7_current_change"
    ]
    assert predecessor["execution"] == authority["correction"][
        "predecessor_v7_execution"
    ]
    assert predecessor["formal_phase"] == authority["correction"][
        "predecessor_v7_formal_phase"
    ]
    current_candidate = dict(authority["candidate"])
    current_candidate.pop("v1_formal_runtime_disposition")
    assert predecessor["candidate"] == current_candidate
    predecessor_paths = set(predecessor["allowed_extent"]["freeze_authorization_paths"])
    current_paths = set(authority["allowed_extent"]["freeze_authorization_paths"])
    assert current_paths - predecessor_paths == {
        "docs/reference_cases/e4_pl_s3_v2_stage4a_leaf_wave_authorization.json"
    }
    for unchanged in (
        "advisory_policy",
        "component_cache_policy",
        "formal_protocol",
        "frozen_inputs",
        "parent",
        "production_boundary",
        "scope_base",
        "terminals",
    ):
        assert predecessor[unchanged] == authority[unchanged]
    assert predecessor["review_correction_requirements"] == (
        _pre_v10_review_requirements(authority)
    )


def test_stage4a_prior_v8_authority_binding_and_history_are_exact(authority) -> None:
    prior = authority["correction"]["history"][7]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predecessor = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical(predecessor)
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree, parent) == (
        prior["subject"],
        prior["tree"],
        prior["parent"],
    )
    assert predecessor["schema"] == "anysolver.e4-pl-s3-v2-stage4a-authority-v8"
    assert predecessor["correction"]["history"] == authority["correction"][
        "history"
    ][:7]
    assert predecessor["correction"]["current_change"] == authority["correction"][
        "predecessor_v8_current_change"
    ]
    assert predecessor["execution"] == authority["correction"][
        "predecessor_v8_execution"
    ]
    assert predecessor["formal_phase"] == authority["formal_phase"]
    assert predecessor["candidate"] == authority["candidate"]
    for unchanged in (
        "advisory_policy",
        "allowed_extent",
        "component_cache_policy",
        "formal_protocol",
        "frozen_inputs",
        "parent",
        "production_boundary",
        "scope_base",
        "terminals",
    ):
        assert predecessor[unchanged] == authority[unchanged]
    assert predecessor["review_correction_requirements"] == (
        _pre_v10_review_requirements(authority)
    )


def test_stage4a_prior_v9_authority_binding_and_history_are_exact(authority) -> None:
    prior = authority["correction"]["history"][8]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predecessor = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical(predecessor)
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree, parent) == (
        prior["subject"],
        prior["tree"],
        prior["parent"],
    )
    assert predecessor["schema"] == "anysolver.e4-pl-s3-v2-stage4a-authority-v9"
    assert predecessor["correction"]["history"] == authority["correction"][
        "history"
    ][:8]
    assert predecessor["correction"]["current_change"] == authority["correction"][
        "predecessor_v9_current_change"
    ]
    assert predecessor["execution"] == authority["correction"][
        "predecessor_v9_execution"
    ]
    assert predecessor["formal_phase"] == authority["formal_phase"]
    assert predecessor["candidate"] == authority["candidate"]
    assert predecessor["dependency_authority"] == authority["dependency_authority"]
    for unchanged in (
        "advisory_policy",
        "allowed_extent",
        "component_cache_policy",
        "formal_protocol",
        "frozen_inputs",
        "parent",
        "production_boundary",
        "scope_base",
        "terminals",
    ):
        assert predecessor[unchanged] == authority[unchanged]
    assert predecessor["review_correction_requirements"] == (
        _pre_v10_review_requirements(authority)
    )


def test_stage4a_prior_v10_authority_binding_and_history_are_exact(authority) -> None:
    prior = authority["correction"]["history"][9]
    raw = subprocess.run(
        ["git", "show", f"{prior['commit']}:{prior['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", prior["commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{prior['commit']}^"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predecessor = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical(predecessor)
    assert len(raw) == prior["bytes"]
    assert hashlib.sha256(raw).hexdigest().upper() == prior["sha256"]
    assert (subject, tree, parent) == (
        prior["subject"],
        prior["tree"],
        prior["parent"],
    )
    assert predecessor["schema"] == "anysolver.e4-pl-s3-v2-stage4a-authority-v10"
    assert predecessor["correction"]["history"] == authority["correction"][
        "history"
    ][:9]
    assert predecessor["correction"]["current_change"] == authority["correction"][
        "predecessor_v10_current_change"
    ]
    assert predecessor["execution"] == authority["correction"][
        "predecessor_v10_execution"
    ]
    assert predecessor["review_correction_requirements"] == authority["correction"][
        "predecessor_v10_review_correction_requirements"
    ]
    assert predecessor["formal_phase"] == authority["formal_phase"]
    assert predecessor["candidate"] == authority["candidate"]
    assert predecessor["dependency_authority"] == authority["dependency_authority"]
    for unchanged in (
        "advisory_policy",
        "allowed_extent",
        "component_cache_policy",
        "formal_protocol",
        "frozen_inputs",
        "parent",
        "production_boundary",
        "scope_base",
        "terminals",
    ):
        assert predecessor[unchanged] == authority[unchanged]


def test_registered_parent_objects_exist_and_match(authority) -> None:
    for identity in (authority["parent"], authority["scope_base"]):
        commit = subprocess.run(
            ["git", "rev-parse", identity["commit"]],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", f"{identity['commit']}^{{tree}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subject = subprocess.run(
            ["git", "show", "-s", "--format=%s", identity["commit"]],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert (commit, tree, subject) == (
            identity["commit"],
            identity["tree"],
            identity["subject"],
        )


def test_stage4a_authority_has_stable_identity() -> None:
    raw = AUTHORITY.read_bytes()
    assert len(raw) > 2_000
    assert len(hashlib.sha256(raw).hexdigest()) == 64
