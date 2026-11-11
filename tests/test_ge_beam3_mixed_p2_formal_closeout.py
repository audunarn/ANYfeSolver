"""Scope-tight closeout checks for the private GE Beam3 P2 parity gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
EVIDENCE = REFERENCE / "ge_beam3_mixed_p2_formal_evidence.json"
REVIEW = REFERENCE / "ge_beam3_mixed_p2_formal_closeout_review.json"
STATUS = REFERENCE / "ge_beam3_mixed_p2_formal_closeout_status.json"
AUTHORITY = REFERENCE / "ge_beam3_mixed_p2_execution_authority.json"
AUTHORITY_REVIEW = REFERENCE / "ge_beam3_mixed_p2_execution_review.json"
CONTRACT = REFERENCE / "ge_beam3_mixed_p2_contract.json"

TERMINAL = "PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY"
VERDICT = "ACCEPT_PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY_NO_P0_P1"
SCOPE = "PRIVATE_STRAIGHT_SOLVER_STATE_LOAD_MASS_RECOVERY_MODAL_BUCKLING_PARITY_ONLY"
REQUEST_ID = "071b0deaf3efa9cbef7b3286670eeba3"
ATTEMPT_ID = "9eda46ce597513f303a4d73cc3931667"
CYCLE_SHA256 = "979379D8AB906557B7C4FEF10ADCFF0A7C7E9FB13E81B0396DF79B41411CFC96"
ACCEPTED_CLOSEOUT_COMMIT = "e31c9e292a2fc9f6b57472bb8c5b90919a535492"


def _sanitized_git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )


def _is_explicit_github_shallow_boundary() -> bool:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return False
    shallow_repository = _sanitized_git("rev-parse", "--is-shallow-repository")
    shallow_name = _sanitized_git("rev-parse", "--git-path", "shallow")
    head = _sanitized_git("rev-parse", "HEAD")
    if (
        shallow_repository.returncode
        or shallow_repository.stdout.strip() != b"true"
        or shallow_name.returncode
        or head.returncode
    ):
        return False
    shallow = Path(os.fsdecode(shallow_name.stdout.strip()))
    if not shallow.is_absolute():
        shallow = (ROOT / shallow).resolve()
    return shallow.is_file() and head.stdout.decode("ascii").strip() in shallow.read_text(
        encoding="ascii"
    ).splitlines()


def _accepted_route_text(path: str) -> str:
    object_name = f"{ACCEPTED_CLOSEOUT_COMMIT}:{path}"
    shown = _sanitized_git("show", "--no-ext-diff", "--no-textconv", object_name)
    if shown.returncode:
        assert _is_explicit_github_shallow_boundary(), (
            f"accepted route is missing outside an explicit GitHub shallow boundary: "
            f"{object_name}"
        )
        pytest.skip("accepted historical route is beyond the explicit GitHub shallow boundary")
    return shown.stdout.decode(
        "utf-8",
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_strict(raw: bytes) -> object:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    value = _decode_strict(raw)
    assert isinstance(value, dict)
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert raw == expected
    return raw, value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _must_reject(raw: bytes) -> None:
    try:
        _decode_strict(raw)
    except ValueError:
        return
    raise AssertionError("strict JSON decoder accepted invalid input")


def test_closeout_records_are_exact_canonical_and_hash_chained() -> None:
    evidence_raw, evidence = _canonical(EVIDENCE)
    review_raw, review = _canonical(REVIEW)
    status_raw, status = _canonical(STATUS)

    assert (len(evidence_raw), _sha(evidence_raw)) == (
        6283,
        "BA6247C4A434132CDF02A3186E7E9C3855E16E8B3D654E083CAB890CF242B2E3",
    )
    assert (len(review_raw), _sha(review_raw)) == (
        3763,
        "5D0ABE862AC511FA61371BA482F40162AEA97864FE3AE501A0D2325B874AD59C",
    )
    assert (len(status_raw), _sha(status_raw)) == (
        2357,
        "FCB538D18DEF333E845D7B1CA84D88DB2CA8656C03518EDDE394663086B9758A",
    )
    assert set(review) == {
        "findings",
        "reviewed_inputs",
        "reviewer_independence",
        "schema",
        "verdict",
    }
    assert review["findings"] == []
    assert review["verdict"] == VERDICT
    assert status["accepted_closeout"] == {
        "evidence": {
            "bytes": len(evidence_raw),
            "path": "docs/reference_cases/ge_beam3_mixed_p2_formal_evidence.json",
            "sha256": _sha(evidence_raw),
        },
        "review": {
            "bytes": len(review_raw),
            "path": (
                "docs/reference_cases/"
                "ge_beam3_mixed_p2_formal_closeout_review.json"
            ),
            "sha256": _sha(review_raw),
            "verdict": VERDICT,
        },
    }


def test_terminal_is_recomputed_from_two_complete_identical_cycles() -> None:
    _evidence_raw, evidence = _canonical(EVIDENCE)
    _status_raw, status = _canonical(STATUS)
    _contract_raw, contract = _canonical(CONTRACT)

    assert evidence["aggregate"] == {
        "bytes": 22463,
        "path": (
            "external/formal/"
            "formal-aggregate-071b0deaf3efa9cbef7b3286670eeba3.json"
        ),
        "sha256": "E1863B0DE713B0CC7DBC2E22E547A2EA7A1B87FE3EDCCC20F46424085D2A0FBA",
        "terminal": TERMINAL,
    }
    assert evidence["adjudication"] == {
        "checker_replica_count_per_cycle": 2,
        "collected_test_node_count": 65,
        "finding_groups": [],
        "formal_cycle_count": 2,
        "guard_test_file_count": 7,
        "scientific_finding": False,
        "terminal_recomputed": True,
    }
    cycles = evidence["cycles"]
    assert isinstance(cycles, list)
    assert [cycle["cycle"] for cycle in cycles] == [1, 2]
    for cycle in cycles:
        assert cycle["cycle_summary"] == {
            "bytes": 3407,
            "sha256": CYCLE_SHA256,
        }
        assert cycle["proof"] == {
            "bytes": 125190,
            "content_sha256": (
                "8E5A1488E7EE340E10A4CC73FA6BA1FDB1D95B2D11EDD0E013F868B6B9E515FC"
            ),
            "raw_record_count": 21,
            "sha256": (
                "A648E4095E2D0FD5F7600EB672A930175D6EC479283E8D1257EBC1B8E1812CF0"
            ),
        }
        assert cycle["checker_replicas_byte_identical"] is True
        assert cycle["scientific_finding_groups"] == []
        assert cycle["test_file_count"] == 7
        assert cycle["test_node_count"] == 65
        assert cycle["terminal"] == TERMINAL
    assert evidence["determinism"] == {
        "checker_outputs_byte_identical_across_replicas_and_cycles": True,
        "cycle_summaries_byte_identical": True,
        "cycle_summary_sha256": CYCLE_SHA256,
        "proof_outputs_byte_identical": True,
        "raw_timing_equality_claimed": False,
    }
    assert contract["terminal_precedence"][-1] == TERMINAL
    assert evidence["terminal"] == TERMINAL
    assert status["terminal"] == TERMINAL
    assert evidence["scientific_evidence_accepted"] is False
    assert status["scientific_evidence_accepted"] is True


def test_authority_consumption_and_frozen_inputs_are_exact() -> None:
    authority_raw, authority = _canonical(AUTHORITY)
    authority_review_raw, authority_review = _canonical(AUTHORITY_REVIEW)
    _evidence_raw, evidence = _canonical(EVIDENCE)
    _status_raw, status = _canonical(STATUS)

    assert (len(authority_raw), _sha(authority_raw)) == (
        20905,
        "3F49D806872DD094A45540D45A127B6A9A407BAD000CDA259C1B5BB36E153D1E",
    )
    assert (len(authority_review_raw), _sha(authority_review_raw)) == (
        19351,
        "325D312DA91B97BAB0F01AC04B4C39184DBA7C5A0193EA85C86081EC9017EDC3",
    )
    assert authority["terminal"] == "AUTHORIZED_GE_BEAM3_P2_FORMAL_EXECUTION"
    assert authority_review["findings"] == []
    assert authority_review["verdict"] == (
        "ACCEPT_GE_BEAM3_MIXED_P2_FORMAL_EXECUTION_NO_P0_P1"
    )
    assert evidence["authority"]["authorization_commit"] == (
        "31b6ca4385e2f2f00dd8e9d874c2e89d0bac05dc"
    )
    assert evidence["authority"]["authorization_tree"] == (
        "42a32e3a046ccfaee7461e40cdcece09c4cdb237"
    )
    assert status["authority"] == {
        "commit": evidence["authority"]["authorization_commit"],
        "tree": evidence["authority"]["authorization_tree"],
    }
    assert evidence["execution"] == {
        "attempt_id": ATTEMPT_ID,
        "claim": {
            "bytes": 638,
            "path": (
                "external/formal-registry/claims/"
                f"{REQUEST_ID}.{ATTEMPT_ID}.claim.json"
            ),
            "sha256": (
                "60A26E8643C0B0CAF39305C57AE056FE19EC29269A0F218478D61D29C55D0DCF"
            ),
        },
        "failure_class": None,
        "no_retry": True,
        "publication_recovery_child_execution_authorized": False,
        "receipt": {
            "bytes": 22700,
            "path": (
                "external/formal-registry/receipts/"
                f"{REQUEST_ID}.{ATTEMPT_ID}.receipt.json"
            ),
            "sha256": (
                "6014344532130B895116183F95D11FD883D5D8FF9BE674C8C157688B081A067C"
            ),
            "terminal": TERMINAL,
        },
        "request_id": REQUEST_ID,
        "request_reuse_authorized": False,
        "state": "CONSUMED_TERMINAL",
        "terminal_receipt_preceded_publication": True,
    }
    assert evidence["frozen_inputs"]["materialization"] == {
        "bytes": 357167,
        "sha256": "5C4BB691FBC17220F0EB7AA139CBE2FB215F61991A699A085EDD8BC04F007391",
    }
    assert evidence["frozen_inputs"]["solver_tree"] == {
        "file_count": 1550,
        "manifest_sha256": (
            "5F24D57EAE084CCFA338BDFCB5100279796DEFD6091AE5523E30BCF871B37199"
        ),
        "total_bytes": 30457792,
    }


def test_independent_review_binds_every_required_input_once() -> None:
    evidence_raw, _evidence = _canonical(EVIDENCE)
    _review_raw, review = _canonical(REVIEW)
    inputs = review["reviewed_inputs"]
    assert isinstance(inputs, list)
    assert len(inputs) == 18
    paths = [row["path"] for row in inputs]
    assert len(paths) == len(set(paths))
    assert inputs[0] == {
        "bytes": len(evidence_raw),
        "path": "docs/reference_cases/ge_beam3_mixed_p2_formal_evidence.json",
        "sha256": _sha(evidence_raw),
    }
    assert all(
        isinstance(row["bytes"], int)
        and row["bytes"] > 0
        and len(row["sha256"]) == 64
        and row["sha256"] == row["sha256"].upper()
        for row in inputs
    )
    assert review["reviewer_independence"] == {
        "closeout_evidence_record_authorship": True,
        "evidence_recovery_method": (
            "READ_ONLY_CANONICAL_HASH_AUTHORITY_AND_TERMINAL_RECOMPUTATION"
        ),
        "formal_evidence_source_authorship": False,
        "formal_process_executed": False,
        "harness_or_mechanics_authorship": False,
        "role": "INDEPENDENT_GE_BEAM3_MIXED_P2_FORMAL_CLOSEOUT_REVIEWER",
        "scope": (
            "AUTHORITY_CONSUMPTION_EVIDENCE_DETERMINISM_TERMINAL_"
            "AND_PRODUCTION_BOUNDARY_ONLY"
        ),
    }


def test_scope_and_production_boundary_remain_private_and_fail_closed() -> None:
    _evidence_raw, evidence = _canonical(EVIDENCE)
    _status_raw, status = _canonical(STATUS)

    expected_boundary = {
        "default_activation_authorized": False,
        "existing_b2_unchanged": True,
        "existing_b3_unchanged": True,
        "existing_beam_aliases_unchanged": True,
        "existing_beam_routes_unchanged": True,
        "public_selector_available": False,
        "qualified_formulation_id_issued": False,
        "qualified_q4_unchanged": True,
        "qualified_s3_v2d_unchanged": True,
        "standalone_opt_in_authorized": False,
    }
    assert evidence["scope"] == SCOPE
    assert status["scope"] == SCOPE
    assert evidence["production_boundary"] == expected_boundary
    assert status["production_boundary"] == expected_boundary
    assert status["execution"] == {
        "attempt_id": ATTEMPT_ID,
        "further_execution_authorized": False,
        "request_id": REQUEST_ID,
        "request_reuse_authorized": False,
        "state": "CONSUMED_TERMINAL",
    }
    assert status["next_gate"] == (
        "SEPARATELY_PREREGISTERED_P3_OPT_IN_SELECTOR_SERIALIZATION_"
        "PACKAGE_AND_PERFORMANCE"
    )
    assert status["production_restriction"] == (
        "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    )
    assert status["upstream_static_core_terminal"] == (
        "PROVISIONAL_GO_GE_BEAM3_MIXED_STRAIGHT_STATIC_CORE"
    )
    for route in (
        "src/anyfesolver/__init__.py",
        "src/anyfesolver/elements.py",
    ):
        text = _accepted_route_text(route)
        assert "GeometricallyExactBeam3D3NElement" not in text
        assert "GE_BEAM3_DC_MIXED_K1_MACRO_V2" not in text
        assert '"ge-beam3"' not in text


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    _must_reject(b'{"a":1,"a":2}\n')
    _must_reject(b'{"a":NaN}\n')
    _must_reject(b'{"a":Infinity}\n')
