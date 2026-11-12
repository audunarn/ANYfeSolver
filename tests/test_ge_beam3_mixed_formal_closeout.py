"""Scope-tight closeout checks for the GE Beam3 mixed straight static core."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
EVIDENCE = REFERENCE / "ge_beam3_mixed_formal_evidence.json"
REVIEW = REFERENCE / "ge_beam3_mixed_formal_closeout_review.json"
STATUS = REFERENCE / "ge_beam3_mixed_formal_closeout_status.json"
AUTHORITY = REFERENCE / "ge_beam3_mixed_execution_authority.json"
AUTHORITY_REVIEW = REFERENCE / "ge_beam3_mixed_execution_review.json"

TERMINAL = "PROVISIONAL_GO_GE_BEAM3_MIXED_STRAIGHT_STATIC_CORE"
SCOPE = (
    "GLOBALLY_STRAIGHT_COLLINEAR_TWO_EQUAL_CELL_ZERO_REFERENCE_JUMP_"
    "STATIC_ELASTIC_CORE_ONLY"
)
ACCEPTED_CLOSEOUT_COMMIT = "9d2bde784356bc7a5a8d314cd91be60c607acac2"


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


def _strict(path: Path) -> tuple[bytes, dict[str, object]]:
    """Read a frozen input without rewriting its checkout newline policy."""
    raw = path.read_bytes()
    value = _decode_strict(raw)
    assert isinstance(value, dict)
    return raw, value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _must_reject(raw: bytes) -> None:
    try:
        _decode_strict(raw)
    except ValueError:
        return
    raise AssertionError("strict JSON decoder accepted invalid input")


def test_committed_evidence_and_review_are_exact_and_canonical() -> None:
    evidence_raw, evidence = _canonical(EVIDENCE)
    review_raw, review = _canonical(REVIEW)
    status_raw, status = _canonical(STATUS)

    assert len(evidence_raw) == 3779
    assert _sha(evidence_raw) == (
        "AB5ACEA336EA721B8159106ACD18A6BE76D856CF07E71C956CB6015AE9003F9E"
    )
    assert len(review_raw) == 3247
    assert _sha(review_raw) == (
        "F5CCAB9EEE80F9DC89E8CB694C87F12F4F8F2D767BCD34F01544BB1407659A12"
    )
    assert len(status_raw) == 2542
    assert _sha(status_raw) == (
        "845297D87E9E6413F7F510BD0B9445C10C4396EE169DC52904EFD86302FB3B0C"
    )
    assert set(review) == {
        "findings",
        "reviewed_inputs",
        "reviewer_independence",
        "schema",
        "verdict",
    }
    assert review["findings"] == []
    assert len(review["reviewed_inputs"]) == 15
    assert review["verdict"] == (
        "ACCEPT_PROVISIONAL_GO_GE_BEAM3_MIXED_STRAIGHT_STATIC_CORE_NO_P0_P1"
    )
    assert status["accepted_closeout"] == {
        "evidence": {
            "bytes": len(evidence_raw),
            "path": "docs/reference_cases/ge_beam3_mixed_formal_evidence.json",
            "sha256": _sha(evidence_raw),
        },
        "review": {
            "bytes": len(review_raw),
            "path": (
                "docs/reference_cases/"
                "ge_beam3_mixed_formal_closeout_review.json"
            ),
            "sha256": _sha(review_raw),
            "verdict": review["verdict"],
        },
    }


def test_terminal_is_recomputed_from_bound_formal_facts() -> None:
    _evidence_raw, evidence = _canonical(EVIDENCE)
    _status_raw, status = _canonical(STATUS)
    _contract_raw, contract = _strict(
        REFERENCE / "ge_beam3_mixed_local_contract.json"
    )

    assert evidence["aggregate"] == {
        "bytes": 6201,
        "sha256": (
            "E0C4153DCB6C69F9B575EF80D74DA787EB0F1BE90B00C44285311F4667DDB403"
        ),
        "terminal": TERMINAL,
    }
    assert evidence["adjudication"] == {
        "checker_dispositions": ["PASS", "PASS"],
        "finding_groups": [],
        "scientific_finding": False,
    }
    assert evidence["determinism"] == {
        "check_bytes_identical": True,
        "checker_origin_bytes_identical": True,
        "producer_origin_bytes_identical": True,
        "proof_bytes_identical": True,
    }
    assert [cycle["cycle"] for cycle in evidence["cycles"]] == [1, 2]
    assert evidence["processes"] == {
        "checker_count": 2,
        "checker_returncodes": [0, 0],
        "checker_trees_drained": [True, True],
        "producer_count": 2,
        "producer_returncodes": [0, 0],
        "producer_trees_drained": [True, True],
    }
    assert evidence["execution"]["failure_class"] is None
    assert contract["terminal_precedence"][-1] == TERMINAL
    assert evidence["terminal"] == TERMINAL
    assert status["terminal"] == TERMINAL
    assert evidence["closeout_review_state"] == (
        "COMPLETE_PENDING_INDEPENDENT_REVIEW"
    )
    assert evidence["scientific_evidence_accepted"] is False
    assert status["scientific_evidence_accepted"] is True


def test_authority_execution_and_materialization_bindings_are_exact() -> None:
    authority_raw, _authority = _canonical(AUTHORITY)
    authority_review_raw, _authority_review = _canonical(AUTHORITY_REVIEW)
    _evidence_raw, evidence = _canonical(EVIDENCE)
    _status_raw, status = _canonical(STATUS)

    assert len(authority_raw) == 10959
    assert _sha(authority_raw) == (
        "1D9BEB6A1FD6E78514625883CEA04C2514E1B97FAD64D69C1064F3E248188180"
    )
    assert len(authority_review_raw) == 6073
    assert _sha(authority_review_raw) == (
        "7D884027248847B889AF1A1E87CED5046C49B65C30349C829BA60E857638AC86"
    )
    assert evidence["authority"]["commit"] == (
        "80c45adc06a96d34bc04d86b43e5ed78df973853"
    )
    assert evidence["authority"]["tree"] == (
        "b5bcec5b816016bbbc4291baf7010295edc60002"
    )
    assert status["authority"] == {
        "commit": evidence["authority"]["commit"],
        "tree": evidence["authority"]["tree"],
    }
    assert evidence["execution"] == {
        "attempt_id": "edd19d2c13b749cdafd4a4e56b295d84",
        "claim": {
            "bytes": 298,
            "sha256": (
                "956E7270418DAA39D4964634DED8DFEA953D70BC10EB71053835E0ECAEA66948"
            ),
        },
        "failure_class": None,
        "receipt": {
            "bytes": 6107,
            "sha256": (
                "096FD9C8B176EE8C2208EAD16450685895F5344A4E5D430B5D668162C28ED3A7"
            ),
        },
        "request_id": "95c5345cf0b44c9ea3b3ed32aff8c44b",
    }
    assert evidence["materialization"] == {
        "file_count": 1518,
        "manifest_bytes": 318880,
        "manifest_sha256": (
            "6003A50E676610C0EB9C1620084B25BF25D34C242267B105BB7DA846728482E1"
        ),
        "tree": "00cb1de420a560e1c96c4ab259f496f2590a3643",
    }


def test_scope_and_production_boundary_remain_fail_closed() -> None:
    _evidence_raw, evidence = _canonical(EVIDENCE)
    _status_raw, status = _canonical(STATUS)

    expected_boundary = {
        "default_activation_authorized": False,
        "existing_b2_unchanged": True,
        "existing_b3_unchanged": True,
        "existing_beam_aliases_unchanged": True,
        "existing_beam_routes_unchanged": True,
        "public_selector_available": False,
        "qualified_q4_unchanged": True,
        "qualified_s3_v2d_unchanged": True,
        "standalone_opt_in_authorized": False,
    }
    assert evidence["scope"] == SCOPE
    assert status["scope"] == SCOPE
    assert evidence["production_boundary"] == expected_boundary
    assert status["production_boundary"] == {
        **expected_boundary,
        "qualified_formulation_id_issued": False,
        "standalone_opt_in_authorized": False,
    }
    assert status["deferred"] == [
        "PUBLIC_SELECTOR_AND_PACKAGE_INTEGRATION",
        "DISTRIBUTED_AND_FOLLOWER_LOADS",
        "HISTORY_BEARING_SECTION_STATE",
        "RESTART_AND_SERIALIZATION",
        "MASS_MODAL_BUCKLING_AND_TRANSIENT",
        "PIECEWISE_STRAIGHT_AND_CURVED_REFERENCE_GEOMETRY",
        "BEAM_SHELL_OBJECTIVE_JOINT",
        "DEFAULT_ACTIVATION_VERSIONING_TAGGING_AND_PUBLICATION",
    ]
    assert status["v1_terminal_preserved"] == (
        "NO_GO_GE_BEAM3_DISCRETE_VARIATIONAL_IDENTITY"
    )
    assert status["execution"] == {
        "attempt_id": "edd19d2c13b749cdafd4a4e56b295d84",
        "further_execution_authorized": False,
        "request_id": "95c5345cf0b44c9ea3b3ed32aff8c44b",
        "request_reuse_authorized": False,
        "state": "CONSUMED_TERMINAL",
    }
    assert status["archives"] == evidence["archives"]
    assert status["production_restriction"] == (
        "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
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
