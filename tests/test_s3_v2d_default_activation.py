"""Public and migration boundaries for qualified S3 V2D activation."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from anyfesolver import (
    AnyStructureFEMConfig,
    LegacyS3MigrationWarning,
    LegacyShellElement,
    NativeParityE4PLS3V2DShellElement,
    S3_V2D_FORMULATION_ID,
    build_fe_model_from_generated_geometry,
    create_element,
    create_shell_element,
    shell_element_from_dict,
    shell_formulation_diagnostics,
)


NORMAL = (0.0, 0.0, 1.0)
ROOT = Path(__file__).resolve().parents[1]
ACTIVATION = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_v2d_default_activation_candidate.json"
)
INTEGRATION_REPAIR = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_v2d_default_integration_repair.json"
)
Q4_REPLAY_HOTFIX = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_q4_0_4_1_replay_hotfix.json"
)
Q4_REPLAY_CANONICALIZATION = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_q4_0_4_2_replay_canonicalization.json"
)
QUALIFIED_SHELL_RUNTIME_GUARD_SUCCESSOR = (
    ROOT
    / "docs"
    / "reference_cases"
    / "qualified_shell_runtime_guard_performance_successor.json"
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    made: dict[str, object] = {}
    for key, value in pairs:
        if key in made:
            raise ValueError(f"duplicate key: {key}")
        made[key] = value
    return made


def _q4_replay_hotfix() -> dict[str, object]:
    raw = Q4_REPLAY_HOTFIX.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert raw == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    for record in payload["historical_authority"]:
        source = ROOT / record["path"]
        data = source.read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest().upper() == record["sha256"]
    return payload


def _q4_replay_canonicalization() -> dict[str, object]:
    raw = Q4_REPLAY_CANONICALIZATION.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert raw == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    prior = payload["prior_evidence"]
    assert prior["path"] == Q4_REPLAY_HOTFIX.relative_to(ROOT).as_posix()
    assert hashlib.sha256(Q4_REPLAY_HOTFIX.read_bytes()).hexdigest().upper() == prior[
        "sha256"
    ]
    return payload


def _qualified_shell_runtime_guard_successor() -> dict[str, object]:
    raw = QUALIFIED_SHELL_RUNTIME_GUARD_SUCCESSOR.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert raw == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    prior = payload["preserved_evidence"]
    assert prior["path"] == Q4_REPLAY_CANONICALIZATION.relative_to(ROOT).as_posix()
    canonicalization = Q4_REPLAY_CANONICALIZATION.read_bytes()
    assert len(canonicalization) == prior["bytes"]
    assert hashlib.sha256(canonicalization).hexdigest().upper() == prior["sha256"]
    _assert_runtime_guard_successor_bindings(payload)
    report = payload["validation"]["static_performance_report"]
    report_data = (ROOT / report["path"]).read_bytes()
    assert len(report_data) == report["bytes"]
    assert hashlib.sha256(report_data).hexdigest().upper() == report["sha256"]
    return payload


def _assert_runtime_guard_successor_bindings(payload: dict[str, object]) -> None:
    assert payload["base"] == {
        "commit": "d04199ac851c0d0f61430c2bc40136582aa8d659",
        "tree": "605fb200b9b7bfffbeeafeae549c6c2ea34d8930",
    }
    change = payload["change"]
    assert change["commit"] == "c31a223fe63f886a1a9b0b47c926bbe748ad4554"
    assert change["tree"] == "533729f4529ea02316ade00adcf20f64e4a3a7eb"
    assert change["e4_pl_element"] == {
        "current_git_blob": "bd6c0dc2cb3661eb73942fc0fea56e527aa8d0b4",
        "prior_git_blob": "b0a53a47ca68cf9ff9ab6925fbdfb4f31301344a",
    }
    assert change["e4_pl_s3_element"] == {
        "current_git_blob": "adc6562cbd3fb768d7f0d0ab69011f21ff7c9392",
        "prior_git_blob": "823b5b0cdc450f7c7f2f2861aefcf4dd2062b99e",
    }
    assert change["scope"] == [
        "context-local trusted operation lease",
        "full authority validation at capture and finalization",
        "generation and exact-instance checks inside trusted loops",
        "guard-cache invalidation on rejected finalization",
    ]
    assert payload["qualification_boundary"] == {
        "default_policy_changed": False,
        "formulation_ids_changed": False,
        "mass_policy_changed": False,
        "mechanics_coefficients_or_operators_changed": False,
        "numerical_tolerances_changed": False,
        "prior_evidence_modified": False,
        "qualified_recovery_changed": False,
    }
    assert payload["review"] == {
        "finding_count": 0,
        "outcome": "ACCEPT",
        "scope": "runtime guard performance successor and state-safety regressions",
    }
    assert payload["schema"] == (
        "anysolver.qualified-shell-runtime-guard-performance-successor-v1"
    )
    assert payload["terminal"] == (
        "PASS_QUALIFIED_SHELL_RUNTIME_GUARD_PERFORMANCE_SUCCESSOR"
    )
    assert payload["validation"] == {
        "authority_and_termination_regressions_passed": True,
        "full_scientific_tests_required": True,
        "static_performance_report": {
            "bytes": 4561,
            "path": "docs/STATIC_RUNTIME_PERFORMANCE.md",
            "sha256": "CBB92A59FDDDA847AA5953EDAA7A2ED6BE1EA64ED1995D415A59B72831BB31D1",
        },
    }


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("base", "commit"),
        ("base", "tree"),
        ("change", "commit"),
        ("change", "tree"),
        ("qualification_boundary", "mechanics_coefficients_or_operators_changed"),
        ("review", "outcome"),
        ("validation", "full_scientific_tests_required"),
    ),
)
def test_runtime_guard_successor_rejects_identity_mutations(
    section: str, field: str
) -> None:
    payload = _qualified_shell_runtime_guard_successor()
    mutated = copy.deepcopy(payload)
    mutated[section][field] = "MUTATED"
    with pytest.raises(AssertionError):
        _assert_runtime_guard_successor_bindings(mutated)


def _git_blob(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


Q4_SOURCE_BINDING_SUCCESSOR = (
    ROOT / "docs" / "reference_cases" / "e4_pl_source_binding_20261111.json"
)
MANIFEST_TEST = ROOT / "tests" / "test_e4_pl_s3_mixed_mesh_manifest.py"


def _adopted_successor_pin() -> str | None:
    """Single source of truth: the pin assigned in the mixed-mesh manifest test."""

    for node in ast.parse(MANIFEST_TEST.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "Q4_SOURCE_BINDING_SUCCESSOR_SHA256"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("successor pin assignment not found")


def _q4_source_binding_successor() -> dict[str, object]:
    pin = _adopted_successor_pin()
    if pin is None:
        pytest.fail(
            "source-binding successor is not adopted: run "
            "scripts/build_q4_source_binding_20261111.py --write, obtain independent "
            "review outside the generating invocation, then pin it in "
            "tests/test_e4_pl_s3_mixed_mesh_manifest.py",
            pytrace=False,
        )
    raw = Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()
    assert hashlib.sha256(raw).hexdigest().upper() == pin
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert payload["qualification_boundary"] == (
        "SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION"
    )
    assert all(value is False for value in payload["boundary_flags"].values())
    return payload


def _assert_live_sources_bound_through_successor(
    q4_blob: str, s3_blob: str, runtime_successor: dict[str, object]
) -> None:
    """The V6 terminal blobs are historical; live bytes bind via the successor.

    This is a mechanics-CHANGED link (October 23 assumed-strain work lies between
    the runtime-guard terminal and the live source).  It does not extend the
    terminal's "mechanics unchanged" assertion and grants no qualification.
    """

    record = _q4_source_binding_successor()
    change = runtime_successor["change"]
    for name, live_blob, key in (
        ("e4_pl_element.py", q4_blob, "e4_pl_element"),
        ("e4_pl_s3_element.py", s3_blob, "e4_pl_s3_element"),
    ):
        row = record["carriers"][name]
        assert row["live_path"] == f"src/anyfesolver/{name}"
        assert row["live_lf"]["git_blob"] == live_blob
        assert row["v6_terminal_git_blob"] == change[key]["current_git_blob"]
        assert row["live_lf_equals_v6_terminal"] is (
            live_blob == change[key]["current_git_blob"]
        )
    link = record["links"]["L1_OCT19_AND_V6_TERMINAL_TO_QUALIFIED_Q4_R4"]
    assert link["mechanics_changed"] is True
    assert link["v6_terminal_mechanics_unchanged_claim_extended"] is False


def test_successor_link_rejects_wrong_live_blob_and_unadopted_pin() -> None:
    runtime_successor = _qualified_shell_runtime_guard_successor()
    q4 = _git_blob(ROOT / "src" / "anyfesolver" / "e4_pl_element.py")
    s3 = _git_blob(ROOT / "src" / "anyfesolver" / "e4_pl_s3_element.py")
    _assert_live_sources_bound_through_successor(q4, s3, runtime_successor)
    with pytest.raises(AssertionError):
        _assert_live_sources_bound_through_successor("0" * 40, s3, runtime_successor)
    with pytest.raises(AssertionError):
        _assert_live_sources_bound_through_successor(q4, "0" * 40, runtime_successor)


def test_activation_candidate_binds_accepted_v6w_and_historical_q4_terminal() -> None:
    raw = ACTIVATION.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert raw == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

    for record in payload["predecessor_qualification"]:
        source = ROOT / record["path"]
        data = source.read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest().upper() == record["sha256"]

    q4_blob = _git_blob(ROOT / "src" / "anyfesolver" / "e4_pl_element.py")
    s3_blob = _git_blob(ROOT / "src" / "anyfesolver" / "e4_pl_s3_element.py")
    hotfix = _q4_replay_hotfix()
    successor = _q4_replay_canonicalization()
    runtime_successor = _qualified_shell_runtime_guard_successor()
    assert payload["production_boundary"]["q4_mechanics_blob"] == hotfix[
        "source_correction"
    ]["prior_git_blob"]
    assert hotfix["source_correction"]["current_git_blob"] == successor["correction"][
        "prior_e4_pl_element_blob"
    ]
    assert successor["correction"]["e4_pl_element_blob"] == runtime_successor[
        "change"
    ]["e4_pl_element"]["prior_git_blob"]
    _assert_live_sources_bound_through_successor(q4_blob, s3_blob, runtime_successor)
    assert runtime_successor["qualification_boundary"][
        "mechanics_coefficients_or_operators_changed"
    ] is False
    assert runtime_successor["qualification_boundary"]["prior_evidence_modified"] is False
    assert hotfix["qualification_boundary"]["q4_coefficients_or_operators_changed"] is False
    assert payload["production_boundary"]["q4_mechanics_unchanged"] is True
    assert payload["production_boundary"]["publication_authorized"] is False
    assert payload["default_policy"]["current_model_q4"] == "e4-pl"
    assert payload["default_policy"]["current_model_s3"] == "e4-pl-s3-v2d"
    assert payload["terminal"] == (
        "PROVISIONAL_GO_E4_PL_S3_V2D_DEFAULT_ACTIVATION_CANDIDATE"
    )


def test_integration_repair_preserves_qualification_and_q4_boundary() -> None:
    raw = INTEGRATION_REPAIR.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_strict_object)
    assert raw == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

    for record in payload["preserved_qualification"]:
        source = ROOT / record["path"]
        data = source.read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest().upper() == record["sha256"]
    activation = payload["activation_candidate"]
    activation_data = (ROOT / activation["path"]).read_bytes()
    assert len(activation_data) == activation["bytes"]
    assert hashlib.sha256(activation_data).hexdigest().upper() == activation["sha256"]

    q4_blob = _git_blob(ROOT / "src" / "anyfesolver" / "e4_pl_element.py")
    s3_blob = _git_blob(ROOT / "src" / "anyfesolver" / "e4_pl_s3_element.py")
    hotfix = _q4_replay_hotfix()
    successor = _q4_replay_canonicalization()
    runtime_successor = _qualified_shell_runtime_guard_successor()
    assert payload["production_boundary"]["q4_mechanics_blob"] == hotfix[
        "source_correction"
    ]["prior_git_blob"]
    assert hotfix["source_correction"]["current_git_blob"] == successor["correction"][
        "prior_e4_pl_element_blob"
    ]
    assert successor["correction"]["e4_pl_element_blob"] == runtime_successor[
        "change"
    ]["e4_pl_element"]["prior_git_blob"]
    _assert_live_sources_bound_through_successor(q4_blob, s3_blob, runtime_successor)
    assert runtime_successor["qualification_boundary"][
        "mechanics_coefficients_or_operators_changed"
    ] is False
    assert hotfix["qualification_boundary"]["s3_evidence_changed"] is False
    assert payload["production_boundary"]["v6w_evidence_changed"] is False
    assert payload["production_boundary"]["s3_linear_stiffness_operator_changed"] is False
    assert payload["default_policy"]["current_model_s3"] == "e4-pl-s3-v2d"
    assert payload["terminal"] == (
        "PROVISIONAL_GO_E4_PL_S3_V2D_DEFAULT_INTEGRATION_REPAIRED"
    )


@pytest.mark.parametrize(
    "alias", ("shell", "s3", "tri3", "tria3", "t3", "shell3", "qualified-s3")
)
def test_current_three_node_aliases_select_exact_v2d(alias: str) -> None:
    element = create_element(
        alias,
        1,
        [1, 2, 3],
        "steel",
        thickness=0.01,
        reference_normal=NORMAL,
    )

    assert type(element) is NativeParityE4PLS3V2DShellElement
    assert element.formulation_id == S3_V2D_FORMULATION_ID


def test_default_fails_closed_without_physical_normal() -> None:
    with pytest.raises(ValueError, match="authoritative reference_normal"):
        create_shell_element(1, [1, 2, 3], "steel")


def test_direct_and_explicit_legacy_routes_remain_available() -> None:
    direct = LegacyShellElement(1, [1, 2, 3], "steel")
    explicit = create_shell_element(
        2, [1, 2, 3], "steel", formulation="legacy-s3"
    )

    assert type(direct) is LegacyShellElement
    assert type(explicit) is LegacyShellElement


def test_missing_identity_historical_record_never_inherits_new_default() -> None:
    payload = {
        "element_id": 7,
        "node_ids": [1, 2, 3],
        "material_name": "steel",
        "thickness": 0.01,
    }
    with pytest.warns(LegacyS3MigrationWarning, match="legacy-s3"):
        restored = shell_element_from_dict(payload)

    assert type(restored) is LegacyShellElement


def test_v2d_non_axis_owner_normal_round_trip_is_binary64_exact() -> None:
    element = create_shell_element(
        8,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2d",
        reference_normal=(0.8819212643483547, -0.4713967368259981, -4.8139e-16),
    )
    payload = element.to_dict()

    first = shell_element_from_dict(payload)
    second = shell_element_from_dict(first.to_dict())

    assert first.to_dict() == payload
    assert second.to_dict() == payload

    malformed = dict(payload)
    malformed["reference_normal"] = [2.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="must be unit length"):
        shell_element_from_dict(malformed)


def test_diagnostics_report_both_topology_defaults() -> None:
    s3 = shell_formulation_diagnostics(node_count=3)
    q4 = shell_formulation_diagnostics(node_count=4)

    assert s3["production_default"] is True
    assert s3["selected_formulation"] == "e4-pl-s3-v2d"
    assert s3["topology_policy"] == "NATIVE_PARITY_E4_PL_S3_V2D_DEFAULT"
    assert q4["production_default"] is True
    assert q4["selected_formulation"] == "e4-pl"


def _authorized_generated_s3() -> dict[str, object]:
    return {
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [1.0, 0.0, 0.0]},
            {"id": 3, "coords": [0.5, 0.8660254037844386, 0.0]},
        ],
        "shells": [
            {
                "id": 1,
                "node_ids": [1, 2, 3],
                "thickness": 0.01,
                "formulation": "e4-pl-s3-v2d",
                "formulation_id": S3_V2D_FORMULATION_ID,
                "reference_normal": list(NORMAL),
                "owner_normal_authority": (
                    "PHYSICAL_SURFACE_OWNER_NORMAL_V2D_V1"
                ),
            }
        ],
    }


def test_generated_geometry_requires_and_constructs_exact_v2d_authority() -> None:
    model = build_fe_model_from_generated_geometry(
        _authorized_generated_s3(),
        AnyStructureFEMConfig(),
    )

    assert type(model.mesh.get_element(1)) is NativeParityE4PLS3V2DShellElement


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("formulation_id", "requires-formulation-id"),
        ("reference_normal", "requires-owner-normal"),
        ("owner_normal_authority", "requires-physical-owner-normal-authority"),
    ),
)
def test_generated_geometry_rejects_incomplete_v2d_authority(
    field: str,
    message: str,
) -> None:
    payload = _authorized_generated_s3()
    payload["shells"][0].pop(field)  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match=message):
        build_fe_model_from_generated_geometry(payload, AnyStructureFEMConfig())
