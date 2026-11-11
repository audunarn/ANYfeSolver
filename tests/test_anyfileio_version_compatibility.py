"""Pinned source and installed-wheel compatibility checks for ANYfileio."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import tomllib
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


EXPECTED_REQUIREMENT = "ANYfileio>=0.3.2,<0.4"
EXPECTED_SPECIFIER = SpecifierSet(">=0.3.2,<0.4")
RELEVANT_PACKAGES = {
    "anymaterial",
    "anygeometry",
    "anymesher",
    "anyfileio",
}


def _project(root: Path) -> dict[str, Any]:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def _assert_expected_fileio_requirement(requirement: str) -> None:
    parsed = Requirement(requirement)
    assert canonicalize_name(parsed.name) == "anyfileio"
    assert not parsed.extras
    assert parsed.marker is None
    assert parsed.url is None
    assert parsed.specifier == EXPECTED_SPECIFIER


def _source_fileio_requirements() -> list[str]:
    dependencies = _project(Path(__file__).resolve().parents[1])["dependencies"]
    return [
        item
        for item in dependencies
        if canonicalize_name(Requirement(item).name) == "anyfileio"
    ]


def _installed_fileio_requirements() -> list[str]:
    requirements = metadata.requires("ANYfeSolver") or []
    return [
        item
        for item in requirements
        if re.match(r"(?i)^anyfileio(?:\s|[<>=!~])", item)
    ]


def _is_beneath(path: Path, root: Path) -> bool:
    path_text = os.path.normcase(str(path.resolve()))
    root_text = os.path.normcase(str(root.resolve()))
    try:
        return os.path.commonpath((path_text, root_text)) == root_text
    except ValueError:
        return False


def test_is_beneath_treats_commonpath_value_error_as_not_beneath(
    monkeypatch,
) -> None:
    def reject_cross_drive(_paths: tuple[str, str]) -> str:
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", reject_cross_drive)
    assert _is_beneath(Path("installed-origin"), Path("workspace")) is False


def test_is_beneath_propagates_non_value_error(monkeypatch) -> None:
    import pytest

    def fail_unexpectedly(_paths: tuple[str, str]) -> str:
        raise RuntimeError("unexpected commonpath failure")

    monkeypatch.setattr(os.path, "commonpath", fail_unexpectedly)
    with pytest.raises(RuntimeError, match="unexpected commonpath failure"):
        _is_beneath(Path("installed-origin"), Path("workspace"))


def _assert_module_identity(
    name: str,
    module: ModuleType,
    versions: dict[str, str],
    roots: dict[str, str],
) -> None:
    assert getattr(module, "__version__") == versions[name]
    origin = Path(module.__file__).resolve()
    root = Path(roots[name]).resolve()
    assert _is_beneath(origin, root), (name, origin, root)
    assert "site-packages" not in os.path.normcase(str(origin))
    normal_checkout = Path(f"C:/Github/{name}")
    assert not _is_beneath(origin, normal_checkout)


def _source_ledgers() -> tuple[dict[str, str], dict[str, str]] | None:
    raw_versions = os.environ.get("ANYSOLVER_EXPECTED_SOURCE_VERSIONS")
    raw_roots = os.environ.get("ANYSOLVER_EXPECTED_SOURCE_ROOTS")
    if raw_versions is None and raw_roots is None:
        return None
    assert raw_versions is not None and raw_roots is not None
    versions = json.loads(raw_versions)
    roots = json.loads(raw_roots)
    expected = {
        "anyfesolver",
        "anymaterial",
        "anygeometry",
        "anymesher",
        "anyfileio",
    }
    assert set(versions) == expected
    assert set(roots) == expected
    return versions, roots


def _requirements_for(root: Path, package_name: str) -> list[Requirement]:
    project = _project(root)
    raw = list(project.get("dependencies", ()))
    if package_name == "anyfileio":
        raw.extend(project.get("optional-dependencies", {}).get("semantics", ()))
    requirements: list[Requirement] = []
    seen: set[str] = set()
    for item in raw:
        requirement = Requirement(item)
        name = canonicalize_name(requirement.name)
        if name not in RELEVANT_PACKAGES:
            continue
        assert name not in seen, (package_name, name)
        assert requirement.marker is None
        assert requirement.url is None
        seen.add(name)
        requirements.append(requirement)
    return requirements


def _assert_graph(
    roots: dict[str, Path], versions: dict[str, str]
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for source, root in roots.items():
        project = _project(root)
        assert canonicalize_name(project["name"]) == source
        assert project["version"] == versions[source]
        for requirement in _requirements_for(root, source):
            target = canonicalize_name(requirement.name)
            assert target in versions
            assert Version(versions[target]) in requirement.specifier, (
                source,
                str(requirement),
                versions[target],
            )
            edges.add((source, target))
    required_edges = {
        ("anyfesolver", "anymaterial"),
        ("anyfesolver", "anymesher"),
        ("anyfesolver", "anyfileio"),
        ("anymesher", "anygeometry"),
    }
    assert required_edges <= edges
    return edges


def _assert_complete_source_graphs(roots: dict[str, str]) -> None:
    archives = Path(roots["anymaterial"]).resolve().parents[1]
    solver_root = Path(roots["anyfesolver"]).resolve().parent
    cells = {
        "coordinated": (
            {
                "anyfesolver": solver_root,
                "anymaterial": archives / "material-current",
                "anygeometry": archives / "geometry-current",
                "anymesher": archives / "mesh-current",
                "anyfileio": archives / "fileio-current",
            },
            {
                "anyfesolver": "0.5.0",
                "anymaterial": "0.2.0",
                "anygeometry": "0.4.3",
                "anymesher": "0.5.0",
                "anyfileio": "0.3.2",
            },
        ),
    }
    for roots_by_name, versions in cells.values():
        _assert_graph(roots_by_name, versions)


def _metadata_requirement(metadata_mode: str) -> str:
    if metadata_mode == "source":
        requirements = _source_fileio_requirements()
    else:
        requirements = _installed_fileio_requirements()
    assert len(requirements) == 1
    _assert_expected_fileio_requirement(requirements[0])
    return requirements[0]


def probe_anyfileio_public_contract(
    *, metadata_mode: str, temporary_root: Path | None = None
) -> dict[str, Any]:
    import anyfileio
    import anygeometry
    import anymaterial
    import anymesher
    import anyfesolver
    from anyfesolver import external_references

    modules = {
        "ANYfeSolver": anyfesolver,
        "ANYmaterial": anymaterial,
        "ANYgeometry": anygeometry,
        "ANYmesher": anymesher,
        "ANYfileio": anyfileio,
    }
    expected_variables = {
        "ANYfeSolver": "EXPECTED_ANYSOLVER_VERSION",
        "ANYmaterial": "EXPECTED_ANYMATERIAL_VERSION",
        "ANYgeometry": "EXPECTED_ANYGEOMETRY_VERSION",
        "ANYmesher": "EXPECTED_ANYMESHER_VERSION",
        "ANYfileio": "EXPECTED_ANYFILEIO_VERSION",
    }
    workspace_raw = os.environ.get("GITHUB_WORKSPACE")
    if metadata_mode == "installed":
        assert workspace_raw is not None
    for distribution_name, module in modules.items():
        expected = os.environ.get(expected_variables[distribution_name])
        if metadata_mode == "installed":
            assert expected is not None, expected_variables[distribution_name]
        if expected is not None:
            assert module.__version__ == expected
        if metadata_mode == "installed":
            distribution = metadata.distribution(distribution_name)
            assert distribution.version == expected
            distribution_root = Path(distribution.locate_file("")).resolve()
            origin = Path(module.__file__).resolve()
            assert _is_beneath(origin, distribution_root)
            assert not _is_beneath(origin, Path(workspace_raw))
            assert not _is_beneath(distribution_root, Path(workspace_raw))

    symbols = (
        "SesamFemDocument",
        "FemRawRecord",
        "CalculixParsedResults",
        "read_sesam_fem_document",
        "write_sesam_fem_document",
        "parse_frd",
        "parse_dat",
        "merge_results",
    )
    assert all(hasattr(anyfileio, name) for name in symbols)
    assert anyfesolver.SesamFemDocument is anyfileio.SesamFemDocument
    assert anyfesolver.FemRawRecord is anyfileio.FemRawRecord
    assert anyfesolver.CalculixParsedResults is anyfileio.CalculixParsedResults
    assert external_references.parse_calculix_frd is anyfileio.parse_frd
    assert external_references.parse_calculix_dat is anyfileio.parse_dat
    assert external_references.merge_calculix_results is anyfileio.merge_results

    def round_trip(root: Path) -> None:
        source = root / "minimal.fem"
        target = root / "roundtrip.fem"
        source.write_text("IDENT          1\nIEND\n", encoding="ascii")
        document = anyfileio.read_sesam_fem_document(source)
        assert isinstance(document, anyfileio.SesamFemDocument)
        anyfileio.write_sesam_fem_document(document, target)
        reread = anyfileio.read_sesam_fem_document(target)
        assert isinstance(reread, anyfileio.SesamFemDocument)

    if temporary_root is None:
        with tempfile.TemporaryDirectory() as directory:
            round_trip(Path(directory))
    else:
        round_trip(temporary_root)

    requirement = _metadata_requirement(metadata_mode)
    origin = Path(anyfileio.__file__).resolve()
    return {
        "anyfileio_version": anyfileio.__version__,
        "anyfileio_origin": str(origin),
        "anyfesolver_version": anyfesolver.__version__,
        "anyfesolver_origin": str(Path(anyfesolver.__file__).resolve()),
        "anymaterial_version": anymaterial.__version__,
        "anymaterial_origin": str(Path(anymaterial.__file__).resolve()),
        "anygeometry_version": anygeometry.__version__,
        "anygeometry_origin": str(Path(anygeometry.__file__).resolve()),
        "anymesher_version": anymesher.__version__,
        "anymesher_origin": str(Path(anymesher.__file__).resolve()),
        "requirement": requirement,
        "metadata_mode": metadata_mode,
    }


def test_source_declares_exact_anyfileio_compatibility_range() -> None:
    assert _source_fileio_requirements() == [EXPECTED_REQUIREMENT]


def test_fileio_requirement_accepts_canonicalized_specifier_order() -> None:
    _assert_expected_fileio_requirement("ANYfileio<0.4,>=0.3.2")


def test_current_source_origins_and_complete_graphs() -> None:
    import pytest

    ledgers = _source_ledgers()
    if ledgers is None:
        pytest.skip("frozen source ledgers are supplied by the focused gate")
    versions, roots = ledgers

    import anyfileio
    import anygeometry
    import anymaterial
    import anymesher
    import anyfesolver

    modules = {
        "anyfesolver": anyfesolver,
        "anymaterial": anymaterial,
        "anygeometry": anygeometry,
        "anymesher": anymesher,
        "anyfileio": anyfileio,
    }
    for name, module in modules.items():
        _assert_module_identity(name, module, versions, roots)
    _assert_complete_source_graphs(roots)


def test_anyfileio_public_contract_used_by_anyfesolver(tmp_path: Path) -> None:
    result = probe_anyfileio_public_contract(
        metadata_mode="source", temporary_root=tmp_path
    )
    assert result["metadata_mode"] == "source"


def test_workflows_pin_compatibility_graph_and_actions() -> None:
    root = Path(__file__).resolve().parents[1]
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    publish = (root / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    action_refs = {
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
    }
    combined = ci + publish
    assert set(re.findall(r"(?m)^\s*(?:- )?uses: (\S+)\s*$", combined)) == action_refs

    ci_header, ci_jobs = ci.split("jobs:\n", maxsplit=1)
    assert ci_header == (
        "name: Tests\n\non:\n  push:\n    branches: [main]\n"
        "  pull_request:\n  workflow_dispatch:\n\n"
    )
    assert re.findall(r"(?m)^  ([a-z0-9-]+):\n", ci_jobs) == [
        "license",
        "resolver",
        "pytest",
        "anymesher-compatibility",
        "anyfileio-compatibility",
        "wheel",
        "numba",
        "pardiso",
    ]
    # The 0.5.0 publish workflow is a two-project (ANYfeSolver-results then
    # ANYfeSolver) pipeline: validate -> gates -> one build -> TestPyPI -> PyPI.
    publish_header, publish_jobs = publish.split("\njobs:\n", maxsplit=1)
    assert publish_header.startswith("name: Publish\n")
    assert (
        "\non:\n  workflow_dispatch:\n  release:\n    types: [published]\n"
        in publish_header
    )
    assert "\npermissions:\n  contents: read\n" in publish_header
    assert "cancel-in-progress: false" in publish_header
    for line in (
        "  RELEASE_TAG: v0.5.0\n",
        '  SOLVER_VERSION: "0.5.0"\n',
        '  RESULTS_VERSION: "2.2.0"\n',
    ):
        assert line in publish_header, line
    assert re.findall(r"(?m)^  ([a-z0-9-]+):\n", publish_jobs) == [
        "validate",
        "license-gate",
        "dependency-gate",
        "build",
        "testpypi",
        "stage",
        "pypi",
    ]

    ci_hashes = set(re.findall(r"\b[0-9a-f]{40}\b", ci))
    assert ci_hashes == {
        "11d5960a326750d5838078e36cf38b85af677262",
        "a26af69be951a213d495a4c3e4e4022e16d87065",
        "d8a233ef4c5e38d25dbba0eb20e6cfa8d44ec5a2",
        "dd954f088a4cb95e267280cc4777b09e16232bd9",
        "27e428188a891705288fef82bab0b166e330aff2",
        "91846898b03fa02b029abde82508eddb981efdc0",
        "2ccef378c3efb4ba3a9957b9ab896ac8fc454b9d",
        "db73950018e1c87dab0ca618c25c965030ba08ce",
        "ebbe0e3b500a3f0cfcc62059c3dac4b0ac178b6c",
        "b6f7cf466865b1c4c34b8e5511ace97d00290f6c",
        "24072503398f5774e0b42e25c74443d1ed3413df",
        "06f3276d9207376316502aca651c3455db984104",
    }
    assert set(re.findall(r"\b[0-9a-f]{40}\b", publish)) == {
        "11d5960a326750d5838078e36cf38b85af677262",
        "a26af69be951a213d495a4c3e4e4022e16d87065",
        "ea165f8d65b6e75b540449e92b4886f43607fa02",
        "d3f86a106a0bac45b974a628896c90dbdf5c8093",
        "dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
    }

    def job_block(workflow: str, name: str) -> str:
        match = re.search(
            rf"(?ms)^  {re.escape(name)}:\n.*?(?=^  [a-z0-9-]+:\n|\Z)",
            workflow,
        )
        assert match is not None, name
        return match.group(0)

    checkout_ref = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
    setup_ref = "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
    upload_ref = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    download_ref = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
    publish_ref = (
        "pypa/gh-action-pypi-publish@"
        "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"
    )

    def action_sequence(block: str) -> list[str]:
        return re.findall(r"(?m)^(?:      - |        )uses: (\S+)\s*$", block)

    assert action_sequence(job_block(ci, "license")) == [checkout_ref, setup_ref]
    assert action_sequence(job_block(ci, "resolver")) == [checkout_ref, setup_ref]
    pytest_block = job_block(ci, "pytest")
    assert action_sequence(pytest_block) == [checkout_ref] * 5 + [setup_ref, checkout_ref]
    solid_condition = "        if: matrix.shard-index == 0 && matrix.subshard-index == 0\n"
    assert pytest_block.count(solid_condition) == 2
    solid_checkout = (
        f"      - uses: {checkout_ref}\n" + solid_condition +
        "        with:\n"
        "          repository: audunarn/ANYsolid\n"
        "          ref: 06f3276d9207376316502aca651c3455db984104\n"
        "          path: .ecosystem/ANYsolid\n"
        "          persist-credentials: false\n"
    )
    assert pytest_block.count(solid_checkout) == 1
    for name in (
        "anymesher-compatibility",
        "anyfileio-compatibility",
        "wheel",
        "numba",
        "pardiso",
    ):
        assert action_sequence(job_block(ci, name)) == [checkout_ref] * 5 + [
            setup_ref
        ]
    assert action_sequence(job_block(publish, "validate")) == [
        checkout_ref,
        setup_ref,
    ]
    assert action_sequence(job_block(publish, "license-gate")) == [
        checkout_ref,
        setup_ref,
    ]
    assert action_sequence(job_block(publish, "dependency-gate")) == [setup_ref]
    assert action_sequence(job_block(publish, "build")) == [
        checkout_ref,
        setup_ref,
        upload_ref,
    ]
    build_block = job_block(publish, "build")
    assert build_block.index("python scripts/verify_release_050.py") < build_block.index(
        upload_ref
    )
    for argument in (
        "--solver-wheel dist/solver/anyfesolver-${SOLVER_VERSION}-py3-none-any.whl",
        "--solver-sdist dist/solver/anyfesolver-${SOLVER_VERSION}.tar.gz",
        "--results-wheel dist/results/anyfesolver_results-${RESULTS_VERSION}-py3-none-any.whl",
        "--results-sdist dist/results/anyfesolver_results-${RESULTS_VERSION}.tar.gz",
        "--expected-commit",
        "--output release-050-gate.json",
        "python -m build --outdir dist/results packages/results",
        "python -m build --outdir dist/solver .",
        "sha256sum results/* solver/*",
    ):
        assert argument in build_block, argument
    assert "name: release-050-distributions" in build_block
    for stage in ("testpypi", "pypi"):
        assert action_sequence(job_block(publish, stage)) == [
            checkout_ref,
            download_ref,
            publish_ref,
            publish_ref,
        ], stage

    checkout_pattern = re.compile(
        rf"(?m)^(?:      - |        )uses: {re.escape(checkout_ref)}\n"
        r"        with:\n"
        r"          repository: (\S+)\n"
        r"          ref: (.+)\n"
        r"          path: (\S+)\n"
        r"          persist-credentials: false$"
    )
    current_checkouts = [
        (
            "audunarn/ANYmaterial",
            "d8a233ef4c5e38d25dbba0eb20e6cfa8d44ec5a2",
            ".ecosystem/ANYmaterial",
        ),
        (
            "audunarn/ANYgeometry",
            "91846898b03fa02b029abde82508eddb981efdc0",
            ".ecosystem/ANYgeometry",
        ),
        (
            "audunarn/ANYmesh",
            "2ccef378c3efb4ba3a9957b9ab896ac8fc454b9d",
            ".ecosystem/ANYmesh",
        ),
        (
            "audunarn/ANYfileIO",
            "db73950018e1c87dab0ca618c25c965030ba08ce",
            ".ecosystem/ANYfileIO",
        ),
    ]
    compatibility_checkouts = {
        "anymesher-compatibility": [
            current_checkouts[0],
            (
                "audunarn/ANYfileIO",
                "${{ matrix.anyfileio-ref }}",
                ".ecosystem/ANYfileIO",
            ),
            (
                "audunarn/ANYgeometry",
                "${{ matrix.anygeometry-ref }}",
                ".ecosystem/ANYgeometry",
            ),
            (
                "audunarn/ANYmesh",
                "${{ matrix.anymesher-ref }}",
                ".ecosystem/ANYmesh",
            ),
        ],
        "anyfileio-compatibility": [
            current_checkouts[0],
            (
                "audunarn/ANYgeometry",
                "${{ matrix.anygeometry-ref }}",
                ".ecosystem/ANYgeometry",
            ),
            (
                "audunarn/ANYmesh",
                "${{ matrix.anymesher-ref }}",
                ".ecosystem/ANYmesh",
            ),
            (
                "audunarn/ANYfileIO",
                "${{ matrix.anyfileio-ref }}",
                ".ecosystem/ANYfileIO",
            ),
        ],
    }
    development_checkouts = [
        current_checkouts[0],
        ("audunarn/ANYgeometry", "ebbe0e3b500a3f0cfcc62059c3dac4b0ac178b6c", ".ecosystem/ANYgeometry"),
        ("audunarn/ANYmesh", "b6f7cf466865b1c4c34b8e5511ace97d00290f6c", ".ecosystem/ANYmesh"),
        ("audunarn/ANYfileIO", "24072503398f5774e0b42e25c74443d1ed3413df", ".ecosystem/ANYfileIO"),
    ]
    assert checkout_pattern.findall(job_block(ci, "pytest")) == development_checkouts
    for name in ("wheel", "numba", "pardiso"):
        block = job_block(ci, name)
        assert checkout_pattern.findall(block) == current_checkouts
        assert block.count(checkout_ref) == 5
    numba_block = job_block(ci, "numba")
    numba_test_install = 'python -m pip install -e ".[dev,numba]"'
    assert pytest_block.count(numba_test_install) == 1
    assert numba_block.count(numba_test_install) == 1
    for name, expected in compatibility_checkouts.items():
        block = job_block(ci, name)
        actual = checkout_pattern.findall(block)
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected, strict=True):
            assert actual_row[0] == expected_row[0]
            assert actual_row[2] == expected_row[2]
            assert actual_row[1] == expected_row[1]
        assert block.count(checkout_ref) == 5
        probe = block.split(
            "      - name: Probe the installed endpoint outside the checkout\n",
            maxsplit=1,
        )[1]
        assert "python -m pip install $wheel[0].FullName" in probe
        assert "pip install --no-deps" not in probe
        assert probe.index("python -m pip install $wheel[0].FullName") < probe.index(
            "python -m pip check"
        )
        assert "--metadata-mode installed" in probe

    mesh_job = job_block(ci, "anymesher-compatibility")
    fileio_job = job_block(ci, "anyfileio-compatibility")
    wheel_job = job_block(ci, "wheel")

    wheel_target_marker = (
        "      - name: Install wheel and pinned siblings into a clean target\n"
    )
    wheel_import_marker = (
        "      - name: Import the installed wheel without site packages\n"
    )
    assert wheel_job.count(wheel_target_marker) == 1
    assert wheel_job.count(wheel_import_marker) == 1
    wheel_target_tail = wheel_job.split(wheel_target_marker, maxsplit=1)[1]
    wheel_target_step, separator, wheel_import_step = wheel_target_tail.partition(
        wheel_import_marker
    )
    assert separator
    assert wheel_target_step == (
        "        run: >-\n"
        '          python -c "import glob, subprocess, sys;\n'
        "          subprocess.check_call([sys.executable, '-m', 'pip', 'install',\n"
        "          '--target', '.wheel-smoke', './packages/results', '.ecosystem/ANYmaterial',\n"
        "          '.ecosystem/ANYgeometry', '.ecosystem/ANYmesh', '.ecosystem/ANYfileIO',\n"
        "          *glob.glob('dist/*.whl')])\"\n"
    )
    assert "--no-deps" not in wheel_target_step
    assert wheel_import_step.count("python -S -c") == 1
    assert wheel_import_step.count("str(Path('.wheel-smoke').resolve())") == 1

    def matrix_rows(block: str) -> str:
        marker = "      matrix:\n        include:\n"
        assert block.count(marker) == 1
        tail = block.split(marker, maxsplit=1)[1]
        rows, separator, _ = tail.partition("\n\n    steps:\n")
        assert separator
        return rows

    assert matrix_rows(mesh_job) == "\n".join(
        (
            '          - anymesher-version: "0.4.0"',
            "            anymesher-ref: 27e428188a891705288fef82bab0b166e330aff2",
            "            anymesher-install: .ecosystem/ANYmesh",
            '            anygeometry-version: "0.4.2"',
            "            anygeometry-ref: dd954f088a4cb95e267280cc4777b09e16232bd9",
            "            anygeometry-install: .ecosystem/ANYgeometry",
            '            anyfileio-version: "0.3.2"',
            "            anyfileio-ref: db73950018e1c87dab0ca618c25c965030ba08ce",
            "            anyfileio-install: ANYfileio==0.3.2",
            '          - anymesher-version: "0.5.0"',
            "            anymesher-ref: 2ccef378c3efb4ba3a9957b9ab896ac8fc454b9d",
            "            anymesher-install: ANYmesher==0.5.0",
            '            anygeometry-version: "0.4.3"',
            "            anygeometry-ref: 91846898b03fa02b029abde82508eddb981efdc0",
            "            anygeometry-install: ANYgeometry==0.4.3",
            '            anyfileio-version: "0.3.2"',
            "            anyfileio-ref: db73950018e1c87dab0ca618c25c965030ba08ce",
            "            anyfileio-install: ANYfileio==0.3.2",
        )
    )
    assert matrix_rows(fileio_job) == "\n".join(
        (
            '          - anyfileio-version: "0.3.2"',
            "            anyfileio-ref: db73950018e1c87dab0ca618c25c965030ba08ce",
            "            anyfileio-install: ANYfileio==0.3.2",
            '            anymesher-version: "0.5.0"',
            "            anymesher-ref: 2ccef378c3efb4ba3a9957b9ab896ac8fc454b9d",
            "            anymesher-install: ANYmesher==0.5.0",
            '            anygeometry-version: "0.4.3"',
            "            anygeometry-ref: 91846898b03fa02b029abde82508eddb981efdc0",
            "            anygeometry-install: ANYgeometry==0.4.3",
        )
    )

    def probe_environment(block: str) -> str:
        probe = block.split(
            "      - name: Probe the installed endpoint outside the checkout\n",
            maxsplit=1,
        )[1]
        marker = "        env:\n"
        assert probe.count(marker) == 1
        environment, separator, _ = probe.split(marker, maxsplit=1)[1].partition(
            "        run: |\n"
        )
        assert separator
        return environment

    assert probe_environment(mesh_job) == "\n".join(
        (
            '          EXPECTED_ANYSOLVER_VERSION: "0.5.0"',
            '          EXPECTED_ANYMATERIAL_VERSION: "0.2.0"',
            "          EXPECTED_ANYMESHER_VERSION: ${{ matrix.anymesher-version }}",
            "          EXPECTED_ANYGEOMETRY_VERSION: ${{ matrix.anygeometry-version }}",
            "          EXPECTED_ANYFILEIO_VERSION: ${{ matrix.anyfileio-version }}",
            "",
        )
    )
    assert probe_environment(fileio_job) == "\n".join(
        (
            '          EXPECTED_ANYSOLVER_VERSION: "0.5.0"',
            '          EXPECTED_ANYMATERIAL_VERSION: "0.2.0"',
            "          EXPECTED_ANYFILEIO_VERSION: ${{ matrix.anyfileio-version }}",
            "          EXPECTED_ANYMESHER_VERSION: ${{ matrix.anymesher-version }}",
            "          EXPECTED_ANYGEOMETRY_VERSION: ${{ matrix.anygeometry-version }}",
            "",
        )
    )
    for value in (
        "2ccef378c3efb4ba3a9957b9ab896ac8fc454b9d",
        "91846898b03fa02b029abde82508eddb981efdc0",
    ):
        assert mesh_job.count(value) == 1
        assert fileio_job.count(value) == 1
    assert mesh_job.count("db73950018e1c87dab0ca618c25c965030ba08ce") == 2
    assert fileio_job.count("db73950018e1c87dab0ca618c25c965030ba08ce") == 1
    for value in (
        "27e428188a891705288fef82bab0b166e330aff2",
        "dd954f088a4cb95e267280cc4777b09e16232bd9",
    ):
        assert mesh_job.count(value) == 1
        assert fileio_job.count(value) == 0
    assert mesh_job.count("          - anymesher-version:") == 2
    assert fileio_job.count("          - anyfileio-version:") == 1
    expected_install = (
        'python -m pip install .ecosystem/ANYmaterial "${{ matrix.anygeometry-install }}" '
        '"${{ matrix.anymesher-install }}" "${{ matrix.anyfileio-install }}"'
    )
    assert mesh_job.count(expected_install) == 1
    assert fileio_job.count(expected_install) == 1

    def job_preamble(block: str) -> str:
        preamble, separator, _ = block.partition("    steps:\n")
        assert separator
        return preamble

    def preamble_keys(block: str) -> dict[str, str]:
        """Top-level job keys (indent 4) with their inline value."""
        return {
            key: value.strip()
            for key, value in re.findall(
                r"(?m)^    ([a-z][a-z-]*):(.*)$", job_preamble(block)
            )
        }

    def nested_lines(block: str, key: str) -> list[str]:
        match = re.search(
            rf"(?m)^    {key}:\n((?:^      [^\n]*\n)+)", job_preamble(block)
        )
        assert match is not None, key
        return match.group(1).splitlines()

    expected_pre = {
        "validate": {"runs-on": "ubuntu-latest"},
        "license-gate": {"needs": "validate", "runs-on": "ubuntu-latest"},
        "dependency-gate": {"needs": "validate", "runs-on": "ubuntu-latest"},
        "build": {
            "needs": "[validate, dependency-gate, license-gate]",
            "runs-on": "ubuntu-latest",
        },
    }
    for name, expected in expected_pre.items():
        actual = preamble_keys(job_block(publish, name))
        for key, value in expected.items():
            assert actual.get(key) == value, (name, key, actual)
        if name == "build":
            assert actual["if"] == "github.event_name == 'workflow_dispatch'"
        else:
            assert "if" not in actual, name
        assert "environment" not in actual, name
    assert preamble_keys(job_block(publish, "dependency-gate")).get("name") == (
        "Verify production sibling releases"
    )

    # Workflow_dispatch only stages to TestPyPI; production promotion happens
    # only on a published release, behind the protected environment, after
    # TestPyPI served the identical digests.
    testpypi_pre = preamble_keys(job_block(publish, "testpypi"))
    assert testpypi_pre["needs"] == "[validate, build]"
    assert testpypi_pre["runs-on"] == "ubuntu-latest"
    assert testpypi_pre["if"] == "github.event_name == 'workflow_dispatch'"
    assert nested_lines(job_block(publish, "testpypi"), "environment") == [
        "      name: testpypi",
        "      url: https://test.pypi.org/p/ANYfeSolver",
    ]
    pypi_pre = preamble_keys(job_block(publish, "pypi"))
    assert pypi_pre["if"] == "github.event_name == 'release'"
    assert pypi_pre["needs"] == "[validate, license-gate, dependency-gate, stage]"
    assert pypi_pre["runs-on"] == "ubuntu-latest"
    assert nested_lines(job_block(publish, "pypi"), "environment") == [
        "      name: pypi",
        "      url: https://pypi.org/p/ANYfeSolver",
    ]

    dependency_job = job_block(publish, "dependency-gate")
    assert "TARGET_INDEX_URL" not in dependency_job
    assert "--index-url https://pypi.org/simple" in dependency_job
    testpypi_job = job_block(publish, "testpypi")
    pypi_job = job_block(publish, "pypi")
    # One legacy-endpoint override per project upload step, TestPyPI only.
    assert testpypi_job.count(
        "repository-url: https://test.pypi.org/legacy/"
    ) == 2
    assert "repository-url:" not in pypi_job
    assert publish.count("repository-url:") == 2

    # Results first, then the solver, per index; nothing is overwritten.
    for job, index in ((testpypi_job, "https://test.pypi.org"), (pypi_job, "https://pypi.org")):
        results_plan = job.index(
            f'--index {index} --name ANYfeSolver-results --version "$RESULTS_VERSION"'
        )
        solver_plan = job.index(
            f'--index {index} --name ANYfeSolver --version "$SOLVER_VERSION"'
        )
        assert results_plan < solver_plan
        assert "packages-dir: plan/results/" in job
        assert "packages-dir: plan/solver/" in job
        assert job.index("packages-dir: plan/results/") < job.index(
            "packages-dir: plan/solver/"
        )
        assert "--require-complete" in job
        assert "sha256sum -c ../SHA256SUMS" in job
        assert "skip-existing" not in job
        assert "--missing-dir plan/results" in job
        assert "--missing-dir plan/solver" in job
    assert (
        "--index https://test.pypi.org --name ANYfeSolver-results"
        in pypi_job
    ), "PyPI promotion must require TestPyPI to serve the identical digests"

    # Least privilege: workflow read-only; only validate adds actions: read and
    # only the two index jobs add id-token: write.
    def job_permissions(name: str) -> list[str] | None:
        match = re.search(
            r"(?m)^    permissions:\n((?:^      [^\n]*\n)+)",
            job_block(publish, name),
        )
        return None if match is None else match.group(1).splitlines()

    assert job_permissions("validate") == [
        "      contents: read",
        "      actions: read",
    ]
    assert job_permissions("stage") == job_permissions("validate")
    assert job_permissions("testpypi") == ["      contents: read", "      id-token: write"]
    assert job_permissions("pypi") == ["      contents: read", "      id-token: write"]
    for name in ("license-gate", "dependency-gate", "build"):
        assert job_permissions(name) is None, name
        assert "permissions:" not in job_block(publish, name)
    assert "id-token" not in job_block(publish, "validate")

    # The validate job binds the tag, versions, main ancestry and trusted CI.
    validate_block = job_block(publish, "validate")
    for fragment in (
        'ANYfeSolver", os.environ.get("SOLVER_VERSION", "0.5.0")',
        'ANYfeSolver-results", os.environ.get("RESULTS_VERSION", "2.2.0")',
        "git merge-base --is-ancestor",
        "actions/workflows/ci.yml/runs?head_sha=",
        'run["head_branch"] == "main"',
    ):
        assert fragment in validate_block, fragment

    assert "SIBLINGS" not in ci
    assert "git+https://" not in ci
    assert ci.count("python -m pip check") == 8
    # The CI resolver builds the same-checkout results wheel with the solver
    # wheel and resolves the exact sibling graph.
    resolver_block = job_block(ci, "resolver")
    assert "./packages/results" in resolver_block
    for pin in (
        "ANYmaterial==0.2.0",
        "ANYgeometry==0.4.3",
        "ANYmesher==0.5.0",
        "ANYfileio==0.3.2",
    ):
        assert pin in resolver_block, pin
    assert '"ANYmaterial==0.2.0"' in publish
    assert '"ANYgeometry==0.4.3"' in publish
    assert '"ANYmesher==0.5.0"' in publish
    assert '"ANYfileio==0.3.2"' in publish
    assert "pip install --dry-run --only-binary=:all:" in publish
    for pin in (
        "ANYmaterial==0.2.0",
        "ANYgeometry==0.4.3",
        "ANYmesher==0.5.0",
        "ANYfileio==0.3.2",
    ):
        assert pin in build_block, pin


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--metadata-mode", choices=("source", "installed"), required=True
    )
    arguments = parser.parse_args()
    if not arguments.probe:
        parser.error("the standalone entry point requires --probe")
    print(
        json.dumps(
            probe_anyfileio_public_contract(
                metadata_mode=arguments.metadata_mode
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
