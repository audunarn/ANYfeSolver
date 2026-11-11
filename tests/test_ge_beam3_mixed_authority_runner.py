from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs" / "reference_cases" / "ge_beam3_mixed_authority_runner.py"
REFERENCE_DIRECTORY = RUNNER.parent


def _load_runner():
    sys.path.insert(0, str(REFERENCE_DIRECTORY))
    try:
        spec = importlib.util.spec_from_file_location(
            "ge_beam3_mixed_authority_runner_under_test", RUNNER
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(REFERENCE_DIRECTORY))


def test_authority_record_is_explicitly_nonauthoritative_without_sources() -> None:
    runner = _load_runner()
    record = runner.build_record(verify_external_sources=False)

    assert record["terminal"] == "NONAUTHORITATIVE_AUTHORITY_CHECK_SKIPPED_SOURCES"
    assert record["source_verification_mode"] == "SKIPPED_TEST_ONLY"
    assert all(record["base_checks"].values())
    assert all(record["exact_agreement"].values())
    assert all(record["policy_checks"].values())
    assert all(record["record_identity_checks"].values())
    assert not any(record["source_checks"].values())
    assert record["exact_agreement"]["condensed_hash"]


def test_current_candidate_extent_and_protected_boundary_are_exact() -> None:
    runner = _load_runner()
    components, diagnostics = runner._candidate_extent_snapshot()
    extent = frozenset().union(*components.values())

    assert extent == runner.EXPECTED_CANDIDATE_EXTENT
    assert all(runner._is_pytest_generated_diagnostic(path) for path in diagnostics)
    assert all(
        runner._working_tree_blob(path) == blob
        for path, blob in runner.EXPECTED_CURRENT_PROTECTED_BLOBS.items()
    )


def test_repository_input_hashes_are_git_blob_and_checkout_eol_independent(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    name = "ge_beam3_mixed_baseline.json"
    relative = f"docs/reference_cases/{name}"
    original = (REFERENCE_DIRECTORY / name).read_bytes()
    assert b"\r" not in original
    crlf_copy = tmp_path / name
    crlf_copy.write_bytes(original.replace(b"\n", b"\r\n"))

    binding = runner._repository_text_binding(relative, working_path=crlf_copy)

    assert binding["git_blob_is_canonical_lf_text"] is True
    assert binding["working_tree_matches_git_blob"] is True
    assert binding["sha256"] == runner.EXPECTED_INPUTS[name][1]
    assert binding["working_sha256"] == binding["sha256"]


def test_repository_text_clean_rejects_binary_and_lone_carriage_return() -> None:
    runner = _load_runner()
    with pytest.raises(ValueError, match="NUL"):
        runner._canonical_lf_text_bytes(b"a\0b")
    with pytest.raises(ValueError, match="lone carriage return"):
        runner._canonical_lf_text_bytes(b"a\rb")


def test_extra_changed_path_blocks_candidate_boundary(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_candidate_extent_snapshot",
        lambda: (
            {
                "committed_since_base": frozenset(),
                "index": frozenset(),
                "working_tree": frozenset(),
                "untracked": runner.EXPECTED_CANDIDATE_EXTENT | {"README.md"},
            },
            frozenset(),
        ),
    )

    record = runner.build_record(verify_external_sources=False)

    assert record["terminal"] == "BLOCKED_GE_BEAM3_MIXED_BASELINE_OR_AUTHORITY"
    assert record["candidate_boundary_checks"]["complete_extent_exact"] is False
    assert "README.md" in record["candidate_extent"]


def test_modified_current_protected_file_blocks_boundary(monkeypatch) -> None:
    runner = _load_runner()
    real_blob = runner._working_tree_blob
    monkeypatch.setattr(
        runner,
        "_working_tree_blob",
        lambda path: "0" * 40 if path == "src/anyfesolver/elements.py" else real_blob(path),
    )
    monkeypatch.setattr(
        runner,
        "_candidate_extent_snapshot",
        lambda: (
            {
                "committed_since_base": frozenset(),
                "index": frozenset(),
                "working_tree": frozenset(),
                "untracked": runner.EXPECTED_CANDIDATE_EXTENT,
            },
            frozenset(),
        ),
    )

    record = runner.build_record(verify_external_sources=False)

    assert record["terminal"] == "BLOCKED_GE_BEAM3_MIXED_BASELINE_OR_AUTHORITY"
    assert record["current_boundary_checks"]["all_protected_current_blobs_equal_base"] is False
    assert record["current_protected_blob_checks"]["src/anyfesolver/elements.py"] is False


def _mutated_record(
    tmp_path: Path,
    *,
    filename: str,
    mutate,
):
    runner = _load_runner()
    for name in runner.JSON_INPUTS:
        (tmp_path / name).write_bytes((REFERENCE_DIRECTORY / name).read_bytes())
    path = tmp_path / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runner.DIRECTORY = tmp_path
    return runner.build_record(verify_external_sources=False)


def _change_source_route(payload: dict) -> None:
    humer = next(
        row
        for row in payload["sources"]
        if row["id"] == "HUMER_STEINBRECHER_PECHSTEIN_2026"
    )
    humer["equation_routes"][4]["equations"] = "43-46"


def _change_source_hash(payload: dict) -> None:
    humer = next(
        row
        for row in payload["sources"]
        if row["id"] == "HUMER_STEINBRECHER_PECHSTEIN_2026"
    )
    humer["artifact"]["sha256"] = "0" * 64


def _change_derived_statement(payload: dict) -> None:
    payload["derived_authority"][0]["statement"] += " ALTERED"


def _change_execution_bound(payload: dict) -> None:
    payload["execution_bounds"]["child_wall_seconds"] = 601


def _change_terminal_precedence(payload: dict) -> None:
    payload["terminal_precedence"][0], payload["terminal_precedence"][1] = (
        payload["terminal_precedence"][1],
        payload["terminal_precedence"][0],
    )


@pytest.mark.parametrize(
    ("filename", "mutate", "check_group", "check_name"),
    (
        (
            "ge_beam3_mixed_source_ledger.json",
            _change_source_route,
            "source_semantic_checks",
            "source_equation_routes_exact",
        ),
        (
            "ge_beam3_mixed_source_ledger.json",
            _change_source_hash,
            "source_semantic_checks",
            "source_artifacts_exact",
        ),
        (
            "ge_beam3_mixed_source_ledger.json",
            _change_derived_statement,
            "source_semantic_checks",
            "derived_authority_exact",
        ),
        (
            "ge_beam3_mixed_local_contract.json",
            _change_execution_bound,
            "policy_checks",
            "execution_bounds_exact",
        ),
        (
            "ge_beam3_mixed_local_contract.json",
            _change_terminal_precedence,
            "policy_checks",
            "terminal_precedence_exact",
        ),
    ),
)
def test_authority_sensitive_mutations_are_blocked_even_in_skip_mode(
    tmp_path: Path,
    filename: str,
    mutate,
    check_group: str,
    check_name: str,
) -> None:
    record = _mutated_record(tmp_path, filename=filename, mutate=mutate)

    assert record["terminal"] == "BLOCKED_GE_BEAM3_MIXED_BASELINE_OR_AUTHORITY"
    assert record["source_verification_mode"] == "SKIPPED_TEST_ONLY"
    assert record[check_group][check_name] is False
    assert (
        record["input_validation_checks"][filename]["complete_field_set_and_values"]
        is False
    )


def test_schema_or_field_set_drift_is_blocked(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        payload["schema"] = "anysolver.ge-beam3-mixed-status-v999"
        payload["unregistered_field"] = True

    record = _mutated_record(
        tmp_path,
        filename="ge_beam3_mixed_status.json",
        mutate=mutate,
    )
    checks = record["input_validation_checks"]["ge_beam3_mixed_status.json"]
    assert record["terminal"] == "BLOCKED_GE_BEAM3_MIXED_BASELINE_OR_AUTHORITY"
    assert checks["exact_schema"] is False
    assert checks["exact_top_level_fields"] is False
    assert checks["complete_field_set_and_values"] is False


def test_cli_output_is_canonical_deterministic_and_exclusive(tmp_path: Path) -> None:
    first = tmp_path / "authority-a.json"
    second = tmp_path / "authority-b.json"
    command = [sys.executable, str(RUNNER), "--skip-external-source-files"]
    for output in (first, second):
        completed = subprocess.run(
            [*command, "--output", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 2, completed.stderr

    assert first.read_bytes() == second.read_bytes()
    parsed = json.loads(first.read_text(encoding="utf-8"))
    assert first.read_bytes() == (
        json.dumps(
            parsed,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    duplicate = subprocess.run(
        [*command, "--output", str(first)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert duplicate.returncode != 0
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    "payload",
    (
        '{"a":1,"a":2}\n',
        '{"a":NaN}\n',
        '{"a":Infinity}\n',
        '{"a":-Infinity}\n',
    ),
)
def test_strict_json_loader_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path, payload: str
) -> None:
    runner = _load_runner()
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        runner._load(path)


def test_runner_import_boundary_is_authority_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
            imported_modules.add(node.module)

    assert imports.isdisjoint({"anyfesolver", "numpy", "scipy", "sympy", "mpmath"})
    assert "ge_beam3_mixed_element" not in imported_modules
    assert "ge_beam3_mixed_finite_producer" not in imported_modules
    assert "ge_beam3_mixed_finite_checker" not in imported_modules
    assert not any(module.startswith("ge_beam3_sr") for module in imported_modules)
