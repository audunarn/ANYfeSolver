"""Focused static tests for the ANYfeSolver 0.5.0 / results 2.2.0 packaging.

These tests read configuration only.  They do not build, install or publish;
the full artifact gate is scripts/verify_release_050.py.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load("_verify_release_050", "scripts/verify_release_050.py")


def _project(relative: str) -> dict:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def test_identities_and_dependency_bands(gate) -> None:
    solver = _project("pyproject.toml")["project"]
    results = _project("packages/results/pyproject.toml")["project"]
    assert (solver["name"], solver["version"]) == ("ANYfeSolver", "0.5.0")
    assert (results["name"], results["version"]) == ("ANYfeSolver-results", "2.2.0")
    bands = dict(gate._split_requirement(item) for item in solver["dependencies"])
    assert bands == gate.EXPECTED_SOLVER_REQUIRES
    assert bands["anyfesolver-results"] == frozenset({">=2.2", "<3"})
    assert results["dependencies"] == ["numpy>=2.0"]
    assert "ANYsolver-results" not in json.dumps(solver)


def test_source_layout_has_no_old_import_trees() -> None:
    assert (ROOT / "src" / "anyfesolver" / "__init__.py").is_file()
    assert (ROOT / "packages" / "results" / "src" / "anyfesolver_results" / "__init__.py").is_file()
    assert not (ROOT / "src" / "anysolver").exists()
    assert not (ROOT / "src" / "anysolver_results").exists()
    assert not (ROOT / "packages" / "results" / "src" / "anysolver_results").exists()


def test_setuptools_and_pytest_configuration() -> None:
    solver = _project("pyproject.toml")
    tool = solver["tool"]
    assert tool["setuptools"]["package-dir"] == {"": "src"}
    assert tool["setuptools"]["packages"]["find"]["include"] == ["anyfesolver", "anyfesolver.*"]
    assert tool["setuptools"]["package-data"] == {"anyfesolver": ["_native_guard.cpp"]}
    assert tool["pytest"]["ini_options"]["pythonpath"] == ["src", "packages/results/src"]
    results = _project("packages/results/pyproject.toml")["tool"]["setuptools"]
    assert results["package-dir"] == {"": "src"}
    assert results["packages"]["find"]["where"] == ["src"]
    assert results["packages"]["find"]["include"] == ["anyfesolver_results", "anyfesolver_results.*"]
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "src/anyfesolver/_native_guard.cpp" in setup
    assert "anyfesolver._native_guard" in setup
    assert "anysolver._native_guard" not in setup


def _setup_namespace(monkeypatch) -> dict:
    """Execute setup.py without invoking setuptools.setup()."""
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert source.rstrip().endswith("setup(ext_modules=extensions)")
    source = source.rstrip()[: -len("setup(ext_modules=extensions)")]
    monkeypatch.chdir(ROOT)
    namespace: dict = {"__name__": "setup_under_test"}
    exec(compile(source, "setup.py", "exec"), namespace)
    return namespace


def test_native_guard_macro_and_environment_names(monkeypatch) -> None:
    cpp = (ROOT / "src" / "anyfesolver" / "_native_guard.cpp").read_text(encoding="utf-8")
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    # The unchanged C++ reads the old macro name; setup.py must define exactly that one.
    assert "#ifndef ANYSOLVER_GUARD_SOURCE_SHA" in cpp
    assert "'ANYSOLVER_GUARD_SOURCE_SHA'" in setup
    assert "ANYFESOLVER_GUARD_SOURCE_SHA" not in setup
    enabled = _setup_namespace(monkeypatch)["native_guard_enabled"]
    assert enabled({}) is True
    assert enabled({"ANYFESOLVER_BUILD_NATIVE_GUARD": "0"}) is False
    assert enabled({"ANYFESOLVER_BUILD_NATIVE_GUARD": "1"}) is True
    assert enabled({"ANYSOLVER_BUILD_NATIVE_GUARD": "0"}) is False  # old name honoured when new is absent
    assert enabled({"ANYSOLVER_BUILD_NATIVE_GUARD": "1"}) is True
    # The new name wins when both are present.
    assert enabled({"ANYFESOLVER_BUILD_NATIVE_GUARD": "1", "ANYSOLVER_BUILD_NATIVE_GUARD": "0"}) is True
    assert enabled({"ANYFESOLVER_BUILD_NATIVE_GUARD": "0", "ANYSOLVER_BUILD_NATIVE_GUARD": "1"}) is False


def test_project_urls_use_the_renamed_repository() -> None:
    for relative in ("pyproject.toml", "packages/results/pyproject.toml"):
        urls = _project(relative)["project"]["urls"]
        assert urls
        for value in urls.values():
            assert value.startswith("https://github.com/audunarn/ANYfeSolver"), value
            assert "ANYsolver" not in value


def test_index_plan_fails_closed_on_conflicts(gate, tmp_path: Path) -> None:
    from hashlib import sha256

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "a.whl").write_bytes(b"A")
    (dist / "a.tar.gz").write_bytes(b"B")
    digest = {n: sha256((dist / n).read_bytes()).hexdigest() for n in ("a.whl", "a.tar.gz")}
    missing = tmp_path / "missing"
    plan = gate.plan_upload({"a.whl": digest["a.whl"]}, dist, missing)
    assert plan["missing"] == ["a.tar.gz"]
    assert [p.name for p in missing.iterdir()] == ["a.tar.gz"]
    assert gate.plan_upload(digest, dist, None)["missing"] == []
    with pytest.raises(gate.GateError):
        gate.plan_upload({**digest, "a.whl": "0" * 64}, dist, None)
    with pytest.raises(gate.GateError):
        gate.plan_upload({**digest, "other.whl": "1" * 64}, dist, None)


def test_license_files_and_inventory_agree() -> None:
    for relative in ("pyproject.toml", "packages/results/pyproject.toml"):
        project = _project(relative)["project"]
        assert project["license"] == "MPL-2.0"
        assert set(project["license-files"]) == {"LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md"}
        base = Path(relative).parent
        for notice in project["license-files"]:
            assert (ROOT / base / notice).is_file(), (base, notice)
    solver = _project("pyproject.toml")["project"]
    inventory = json.loads((ROOT / "dependency-licenses.json").read_text(encoding="utf-8"))
    assert (inventory["project"], inventory["release"]) == ("ANYfeSolver", "0.5.0")
    recorded = {item["requirement"] for item in inventory["dependencies"]}
    declared = set(solver["dependencies"])
    for group in solver["optional-dependencies"].values():
        declared.update(group)
    assert declared == recorded
    solver_notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert all(f"`{requirement}`" in solver_notice for requirement in declared)
    results = _project("packages/results/pyproject.toml")["project"]
    results_notice = (ROOT / "packages/results/THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert all(f"`{requirement}`" in results_notice for requirement in results["dependencies"])


def test_manifest_ships_native_source_and_release_note() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "include src/anyfesolver/_native_guard.cpp" in manifest
    assert "include docs/releases/ANYfeSolver_0.5.0.md" in manifest
    assert "prune packages" in manifest
    assert "src/anysolver/" not in "\n".join(manifest)
    assert (ROOT / "docs" / "releases" / "ANYfeSolver_0.5.0.md").is_file()


def test_historical_verifier_is_loadable_and_still_0_4_7() -> None:
    historical = _load("_historical_047", "scripts/verify_release_047.py")
    assert historical.VERSION == "0.4.7"
    gate = _load("_gate_for_terminal", "scripts/verify_release_050.py")
    assert gate.TERMINAL != historical.TERMINAL
    assert "0_4_7" not in gate.TERMINAL


def test_runtime_comparison_detects_drift(gate) -> None:
    source = {"anyfesolver/a.py": b"x = 1\n"}
    gate._same_runtime(dict(source), source, "ok")
    with pytest.raises(gate.GateError):
        gate._same_runtime({"anyfesolver/a.py": b"x = 2\n"}, source, "changed")
    with pytest.raises(gate.GateError):
        gate._same_runtime({**source, "anyfesolver/b.py": b""}, source, "extra")
    # A host-specific extension may be present without being in the source tree.
    gate._same_runtime({**source, "anyfesolver/_native_guard.pyd": b"\0"}, source, "binary")


def test_sdist_extraction_refuses_traversal(gate, tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("pkg-1/../../escape.txt")
        payload = b"no"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    with pytest.raises(gate.GateError):
        gate._extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_publish_workflow_policy() -> None:
    raw = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    # Judge executable YAML only: comment lines may mention forbidden words.
    text = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("#"))
    assert re.search(r"^on:\s*$", text, re.M)
    assert not re.search(r"^\s*pull_request(_target)?\s*:", text, re.M)
    for action in re.findall(r"uses:\s*([^\s]+)", text):
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", action), action
    assert "verify_release_050.py" in text and "verify_release_047" not in text
    assert "ci.yml/runs?head_sha=" in text
    assert text.index("Publish missing ANYfeSolver-results files to TestPyPI") < text.index(
        "Publish missing ANYfeSolver files to TestPyPI")
    assert text.index("Publish missing ANYfeSolver-results files to PyPI") < text.index(
        "Publish missing ANYfeSolver files to PyPI")
    def job(name: str, following: str | None) -> str:
        start = text.index(f"\n  {name}:")
        return text[start:text.index(f"\n  {following}:", start)] if following else text[start:]

    build, testpypi = job("build", "testpypi"), job("testpypi", "stage")
    stage, pypi = job("stage", "pypi"), job("pypi", None)
    # Staging: dispatch only. Promotion: release only, never a rebuild.
    assert "if: github.event_name == 'workflow_dispatch'" in build
    assert "if: github.event_name == 'workflow_dispatch'" in testpypi
    assert "if: github.event_name == 'release'" in stage and "if: github.event_name == 'release'" in pypi
    assert text.count("if: github.event_name == 'release'") == 2
    assert "needs: [validate, license-gate, dependency-gate, stage]" in pypi
    for promotion in (stage, pypi):
        assert "python -m build" not in promotion and "setup-python" not in promotion
        assert "build" not in re.search(r"needs:\s*\[[^\]]*\]", promotion).group(0)
    # No second-build equality gate: sdists are not byte-reproducible.
    assert "SOURCE_DATE_EPOCH" not in text and "repro/" not in text and "reproducible" not in text
    # Promotion retrieves exactly one trusted dispatch artifact by run id, in this repository.
    assert text.count("run-id:") == 1 and text.count("github-token:") == 1
    assert "run-id:" in stage and "actions: read" in stage
    for requirement in ('workflows/publish.yml"', 'run["event"] == "workflow_dispatch"',
                        'run["conclusion"] == "success"', 'run["head_sha"] == sha',
                        'run["head_branch"] == tag', "pull_requests", '"fork"', "repo_id",
                        'artifact["expired"]', "exactly one release-050-distributions",
                        "manifest commit mismatch", "stage first", "stage again"):
        assert requirement in stage, requirement
    assert "release-050-promoted" in stage and "release-050-promoted" in pypi
    assert "release-050-distributions" not in pypi
    assert "retention-days: 30" in build
    assert "sha256sum -c" in testpypi and "sha256sum -c" in stage and "sha256sum -c" in pypi
    assert text.count("sha256sum -c") == 3
    # PyPI promotion still needs TestPyPI to serve identical digests (no sleep loop beyond 5 tries).
    assert "--index https://test.pypi.org" in pypi and "--require-complete" in pypi
    loops = re.findall(r"for attempt in ([\d ]+);", text)
    assert loops and all(loop == "1 2 3 4 5" for loop in loops)
    # A published file is never overwritten or skipped by name, and no other repository is read.
    for forbidden in ("skip-existing", "skip_existing", "repository:"):
        assert forbidden not in text, forbidden
    # Uploads are always the planned missing files, never the full dist directories.
    assert "packages-dir: stage/" not in text
    assert text.count("packages-dir: plan/") == 4
    assert text.count("index-plan") >= 8
    # Separate protected environments; dispatch has no production input.
    assert "name: testpypi" in testpypi and "name: pypi" in pypi
    assert "environment:" not in stage and "id-token" not in stage
    assert "inputs:" not in text
