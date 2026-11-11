from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from anyfesolver.e4_pl_s3_v2_element import (
    FORMULATION_ID,
    IMPLEMENTATION_ID,
    SELECTOR,
)
from anyfesolver.elements import DEFAULT_Q4_FORMULATION, DEFAULT_S3_FORMULATION


ROOT = Path(__file__).resolve().parents[1]
REGISTERED_COMMIT = "d1f6d3d264882cc70a34b6a764476f5ec6baeb3b"
BINDING_PATH = "docs/reference_cases/e4_pl_s3_v2_candidate_binding.json"


def _sanitized_git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
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
    if shallow_repository.returncode or shallow_repository.stdout.strip() != b"true":
        return False
    shallow_name = _sanitized_git("rev-parse", "--git-path", "shallow")
    head = _sanitized_git("rev-parse", "HEAD")
    if shallow_name.returncode or head.returncode:
        return False
    shallow = Path(os.fsdecode(shallow_name.stdout.strip()))
    if not shallow.is_absolute():
        shallow = (ROOT / shallow).resolve()
    if not shallow.is_file():
        return False
    return head.stdout.decode("ascii").strip() in shallow.read_text(
        encoding="ascii"
    ).splitlines()


def _registered_bytes(path: str) -> bytes:
    replacement_refs = _sanitized_git("replace", "-l")
    assert replacement_refs.returncode == 0, replacement_refs.stderr.decode(
        "utf-8", errors="replace"
    )
    assert replacement_refs.stdout == b"", "replacement refs are forbidden"
    graft_name = _sanitized_git("rev-parse", "--git-path", "info/grafts")
    assert graft_name.returncode == 0, graft_name.stderr.decode(
        "utf-8", errors="replace"
    )
    graft = Path(os.fsdecode(graft_name.stdout.strip()))
    if not graft.is_absolute():
        graft = (ROOT / graft).resolve()
    assert not graft.exists(), "Git grafts are forbidden"

    object_name = f"{REGISTERED_COMMIT}:{path}"
    probe = _sanitized_git("cat-file", "-e", object_name)
    if probe.returncode:
        assert _is_explicit_github_shallow_boundary(), (
            f"registered object is unavailable outside an explicit GitHub shallow "
            f"boundary: {object_name}: "
            f"{probe.stderr.decode('utf-8', errors='replace').strip()}"
        )
        pytest.skip(
            "immutable registered bytes are unavailable at the explicit GitHub "
            "shallow boundary; working-tree bytes are intentionally forbidden"
        )
    shown = _sanitized_git("show", "--no-ext-diff", "--no-textconv", object_name)
    assert shown.returncode == 0, shown.stderr.decode("utf-8", errors="replace")
    return shown.stdout


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    made: dict[str, object] = {}
    for key, value in pairs:
        if key in made:
            raise ValueError(f"duplicate JSON key: {key}")
        made[key] = value
    return made


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite JSON value: {value}")


def _decode(raw: bytes) -> dict[str, object]:
    made = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_nonfinite,
    )
    assert isinstance(made, dict)
    canonical = (json.dumps(made, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert raw == canonical
    return made


def _canonical_repository_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(_canonical_repository_bytes(raw)).hexdigest().upper()


def test_candidate_binding_is_canonical_and_binds_every_registered_file() -> None:
    binding = _decode(_registered_bytes(BINDING_PATH))
    assert set(binding) == {
        "authority_inputs",
        "candidate",
        "defaults",
        "formal_execution",
        "independent_evidence",
        "parent",
        "production_paths",
        "production_restriction",
        "schema",
        "scope",
        "test_lanes",
    }
    assert binding["schema"] == "anysolver.e4-pl-s3-v2-flat-candidate-binding-v1"

    registered = binding["authority_inputs"] + binding["production_paths"]
    paths = [record["path"] for record in registered]
    assert len(paths) == len(set(paths))
    for record in registered:
        raw = _registered_bytes(record["path"])
        assert len(_canonical_repository_bytes(raw)) == record["bytes"]
        assert _sha256(raw) == record["sha256"]


def test_candidate_identity_scope_defaults_and_formal_stop_are_exact() -> None:
    binding = _decode(_registered_bytes(BINDING_PATH))
    assert binding["candidate"] == {
        "formulation_id": FORMULATION_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "production_class": "StrictFlatLinearE4PLS3V2ShellElement",
        "selectors": [SELECTOR, "qualified-s3-v2"],
        "stage_gate": "PASS_E4_PL_S3_V2A_LOCAL_FORMULATION",
    }
    assert binding["defaults"] == {
        "default_q4_formulation": DEFAULT_Q4_FORMULATION,
        "default_s3_formulation": DEFAULT_S3_FORMULATION,
        "q4_mechanics_unchanged": True,
        "s3_v1_mechanics_unchanged": True,
    }
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"
    assert binding["scope"] == {
        "blocked": [
            "curved_shell",
            "dynamic",
            "generalized_section",
            "material_nonlinearity",
            "mixed_element_mesh",
            "nonlinear_geometry",
            "qualified_recovery",
            "restart",
            "serialization",
        ],
        "model_boundary": "PURE_EXACT_V2_GLOBAL_FLAT_PLANE_COMMON_PHYSICAL_DIRECTOR",
        "supported": [
            "dead_uniform_or_affine_transverse_pressure",
            "linear_internal_force",
            "linear_stiffness",
            "raw_variational_resultants",
        ],
    }
    assert binding["formal_execution"] == {
        "candidate_wheel_frozen": False,
        "default_activation_authorized": False,
        "mixed_funnel_authorized": False,
        "next_required_authority": "STAGE_4_FLAT_MIXED_FUNNEL_SUCCESSOR_CONTRACT",
        "qualification_claimed": False,
    }
    assert binding["production_restriction"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"


def test_candidate_binding_parser_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _decode(b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="nonfinite JSON value"):
        _decode(b'{"a":NaN}\n')
