"""Packaging gate for ANYfeSolver 0.5.0 and ANYfeSolver-results 2.2.0.

This gate verifies, for one candidate commit, that both wheel/sdist pairs were
built from the same clean source, that each sdist independently rebuilds an
installable wheel with identical shipped content, and that a clean install of
the two wheels imports under the new names only.  It also re-checks the frozen
0.4.7 evidence files by their recorded hashes; that evidence stays bound to
0.4.7 and is NOT scientific requalification of 0.5.0.

A passing run is a packaging record.  It is not a publication decision and it
does not approve release; it does not rerun any numerical campaign.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOLVER_NAME = "anyfesolver"
SOLVER_VERSION = "0.5.0"
RESULTS_NAME = "anyfesolver_results"
RESULTS_VERSION = "2.2.0"
TIMEOUT_SECONDS = 900
PROBE_TIMEOUT_SECONDS = 180
LICENSE_FILES = {"LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md"}
SOLVER_SDIST_EXTRA = {"LICENSING.md", "dependency-licenses.json", "README.md", "setup.py"}
OLD_IMPORT_NAMES = ("anysolver", "anysolver_results")
TERMINAL = "ANYFESOLVER_0_5_0_PACKAGING_GATE_PASSED_NOT_A_PUBLICATION_DECISION"
SCHEMA = "ANYFESOLVER_0_5_0_PACKAGING_GATE_V1"
# Public solver dependency bands that must stay as released in 0.4.7.
EXPECTED_SOLVER_REQUIRES = {
    "numpy": frozenset({">=2.0"}),
    "scipy": frozenset({">=1.13"}),
    "threadpoolctl": frozenset({">=3.5"}),
    "anymaterial": frozenset({">=0.2", "<0.3"}),
    "anymesher": frozenset({">=0.4", "<0.6"}),
    "anyfileio": frozenset({">=0.3.2", "<0.4"}),
    "anyfesolver-results": frozenset({">=2.2", "<3"}),
}
EXPECTED_RESULTS_REQUIRES = {"numpy": frozenset({">=2.0"})}
SOLVER_DOCS = {
    "docs/ARCHITECTURE.md",
    "docs/ARC_LENGTH.md",
    "docs/B3_GE_PRODUCTION_OPT_IN.md",
    "docs/E4_PL_MIGRATION.md",
    "docs/GE_BEAM3_NATIVE_WORKFLOWS.md",
    "docs/NATIVE_SOURCE_AUTHORITY.md",
    "docs/NONLINEAR_PERFORMANCE.md",
    "docs/PERFORMANCE_ARCHITECTURE.md",
    "docs/QUALITY_CONTROL.md",
    "docs/THEORY.md",
    "docs/releases/ANYfeSolver_0.5.0.md",
    "docs/releases/ANYsolver_0.4.6.md",
    "docs/releases/ANYsolver_0.4.7.md",
}
DEV_ONLY_PREFIXES = (
    "artifacts/",
    "docs/agent_plans/",
    "docs/reference_cases/",
    "packages/",
    "reports/",
    "scripts/",
    "tests/",
)
BINARY_SUFFIXES = (".pyd", ".so", ".dylib")


class GateError(ValueError):
    """The packaging gate was not satisfied."""


def _record(raw: bytes) -> dict[str, Any]:
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def _skip(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or path.suffix == ".pyc"
        or path.suffix in BINARY_SUFFIXES
        or any(part.endswith(".egg-info") for part in path.parts)
    )


def _tree(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        raise GateError(f"source tree is missing: {directory}")
    files = {
        path.relative_to(directory.parent).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not _skip(path)
    }
    if not files:
        raise GateError(f"source tree is empty: {directory}")
    return files


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_requirement(text: str) -> tuple[str, frozenset[str]]:
    body = text.split(";", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*\(?([^)]*)\)?$", body)
    if match is None:
        raise GateError(f"unparseable requirement: {text!r}")
    specs = frozenset(s.strip().replace(" ", "") for s in match.group(2).split(",") if s.strip())
    return _normalize(match.group(1)), specs


def _requires(headers: Any) -> dict[str, frozenset[str]]:
    made: dict[str, frozenset[str]] = {}
    for item in headers.get_all("Requires-Dist") or []:
        if "extra ==" in item:
            continue
        name, specs = _split_requirement(item)
        made[name] = specs
    return made


def _git_identity(expected_commit: str) -> tuple[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise GateError("expected commit is not a full lowercase Git object ID")
    command = ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT)]

    def run(*args: str) -> str:
        return subprocess.check_output(
            command + list(args), text=True, timeout=PROBE_TIMEOUT_SECONDS
        ).strip()

    commit = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    dirty = run("status", "--porcelain=v1", "--untracked-files=no")
    if commit != expected_commit:
        raise GateError("candidate commit mismatch")
    if dirty:
        raise GateError("candidate has tracked modifications")
    return commit, tree


def _pyproject(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)["project"]


def _check_project_files() -> dict[str, Any]:
    solver = _pyproject(ROOT / "pyproject.toml")
    results = _pyproject(ROOT / "packages" / "results" / "pyproject.toml")
    if (solver["name"], solver["version"]) != ("ANYfeSolver", SOLVER_VERSION):
        raise GateError("solver pyproject identity mismatch")
    if (results["name"], results["version"]) != ("ANYfeSolver-results", RESULTS_VERSION):
        raise GateError("results pyproject identity mismatch")
    if dict(_split_requirement(r) for r in solver["dependencies"]) != EXPECTED_SOLVER_REQUIRES:
        raise GateError("solver public dependency bands changed")
    if dict(_split_requirement(r) for r in results["dependencies"]) != EXPECTED_RESULTS_REQUIRES:
        raise GateError("results dependency bands changed")
    return {"solver": f"{solver['name']}=={solver['version']}",
            "results": f"{results['name']}=={results['version']}"}


def _members(path: Path) -> dict[str, bytes]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise GateError(f"artifact is missing or empty: {path.name}")
    made: dict[str, bytes] = {}
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise GateError(f"{path.name} has duplicate entries")
            for name in names:
                if not name.endswith("/"):
                    made[name] = archive.read(name)
    else:
        with tarfile.open(path, "r:gz") as archive:
            seen: set[str] = set()
            for member in archive.getmembers():
                if member.name in seen:
                    raise GateError(f"{path.name} has duplicate entries")
                seen.add(member.name)
                if member.isfile():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    made[member.name] = stream.read()
                elif not member.isdir():
                    raise GateError(f"{path.name} has a non-regular member: {member.name}")
    for name in made:
        parts = name.split("/")
        if name.startswith("/") or ".." in parts:
            raise GateError(f"{path.name} has an unsafe member path: {name}")
        if parts[0] in OLD_IMPORT_NAMES or "src/anysolver" in name or "src/anysolver_results" in name:
            raise GateError(f"{path.name} ships an old import tree: {name}")
    return made


def _metadata(members: dict[str, bytes], label: str) -> Any:
    names = [n for n in members if n.endswith(".dist-info/METADATA")]
    if len(names) != 1:
        raise GateError(f"{label} must contain exactly one METADATA file")
    return BytesParser().parsebytes(members[names[0]])


def _check_metadata(headers: Any, name: str, version: str, expected: dict, label: str) -> dict:
    if _normalize(headers["Name"]) != _normalize(name) or headers["Version"] != version:
        raise GateError(f"{label} metadata identity mismatch")
    if headers.get("License-Expression") != "MPL-2.0":
        raise GateError(f"{label} does not declare License-Expression MPL-2.0")
    declared = set(headers.get_all("License-File") or [])
    declared = {Path(item).name for item in declared}
    if not LICENSE_FILES <= declared:
        raise GateError(f"{label} metadata omits license files")
    if _requires(headers) != expected:
        raise GateError(f"{label} Requires-Dist differs from the declared bands")
    return {"name": headers["Name"], "version": headers["Version"],
            "requires_python": headers["Requires-Python"],
            "license_expression": headers["License-Expression"],
            "license_files": sorted(declared)}


def _runtime(members: dict[str, bytes], package: str) -> dict[str, bytes]:
    return {n: b for n, b in members.items() if n.startswith(package + "/")}


def _same_runtime(runtime: dict[str, bytes], source: dict[str, bytes], label: str) -> None:
    extra = {n for n in runtime if n not in source and not n.endswith(BINARY_SUFFIXES)}
    missing = set(source) - set(runtime)
    if extra or missing:
        raise GateError(
            f"{label} path set differs from source: extra={sorted(extra)[:5]} "
            f"missing={sorted(missing)[:5]}"
        )
    for name, raw in source.items():
        if runtime[name] != raw:
            raise GateError(f"{label} differs from source: {name}")


def _inventory(members: dict[str, bytes]) -> list[dict[str, Any]]:
    return [{"path": n, **_record(b)} for n, b in sorted(members.items())]


def _check_sdist(path: Path, name: str, version: str, package: str, source: dict[str, bytes],
                 expected: dict, solver: bool) -> dict[str, Any]:
    if path.name != f"{name}-{version}.tar.gz":
        raise GateError(f"{path.name}: sdist filename mismatch")
    members = _members(path)
    prefix = f"{name}-{version}/"
    if any(not n.startswith(prefix) for n in members):
        raise GateError(f"{path.name} has an unexpected root")
    rel = {n[len(prefix):]: b for n, b in members.items()}
    if solver:
        if any(n.startswith(DEV_ONLY_PREFIXES) for n in rel):
            raise GateError("solver sdist contains development-only material")
        docs = {n for n in rel if n.startswith("docs/")}
        if docs != SOLVER_DOCS:
            raise GateError(f"solver sdist documentation set mismatch: {sorted(docs ^ SOLVER_DOCS)}")
        required = LICENSE_FILES | SOLVER_SDIST_EXTRA
    else:
        required = LICENSE_FILES | {"README.md"}
    missing = sorted(f for f in required | {"pyproject.toml"} if f not in rel)
    if missing:
        raise GateError(f"{path.name} lacks required files: {missing}")
    shipped = {n: b for n, b in rel.items() if n.startswith(("src/" + package + "/"))}
    prefixed = {"src/" + n: b for n, b in source.items()}
    _same_runtime(shipped, prefixed, f"{path.name} source")
    others = [n for n in rel if n.startswith("src/") and not n.startswith("src/" + package + "/")
              and ".egg-info/" not in n]
    if others:
        raise GateError(f"{path.name} ships unexpected source: {others[:5]}")
    headers = BytesParser().parsebytes(rel["PKG-INFO"])
    if _normalize(headers["Name"]) != _normalize(name) or headers["Version"] != version:
        raise GateError(f"{path.name} PKG-INFO identity mismatch")
    if _requires(headers) != expected:
        raise GateError(f"{path.name} PKG-INFO Requires-Dist differs from the declared bands")
    return {"archive": {"name": path.name, **_record(path.read_bytes())}, "files": _inventory(rel)}


def _extract(path: Path, destination: Path) -> Path:
    """Extract regular files only, refusing any path that leaves destination."""
    destination.mkdir(parents=True)
    root = destination.resolve()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise GateError(f"unsafe sdist member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                assert stream is not None
                target.write_bytes(stream.read())
            else:
                raise GateError(f"non-regular sdist member: {member.name}")
    roots = [p for p in destination.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise GateError("sdist must extract to exactly one root")
    return roots[0]


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(extra)
    return env


def _run(args: list[str], cwd: Path, env: dict[str, str], timeout: int) -> str:
    done = subprocess.run(args, cwd=cwd, env=env, check=False, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if done.returncode != 0:
        raise GateError(f"command failed ({' '.join(args[:4])}...): "
                        f"{(done.stderr.strip() or done.stdout.strip())[-1500:]}")
    return done.stdout


def _rebuild(sdist: Path, work: Path, tag: str, isolated: bool) -> Path:
    root = _extract(sdist, work / f"{tag}-src")
    out = work / f"{tag}-wheel"
    out.mkdir()
    command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(root)]
    if not isolated:
        command.insert(4, "--no-isolation")
    # Pure rebuild: a native guard binary is host-specific and compared by presence only.
    _run(command, work, _clean_env(ANYFESOLVER_BUILD_NATIVE_GUARD="0"), TIMEOUT_SECONDS)
    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != 1:
        raise GateError(f"{tag} sdist did not rebuild exactly one wheel")
    return wheels[0]


def _compare_wheels(original: Path, rebuilt: Path, label: str) -> dict[str, Any]:
    a, b = _members(original), _members(rebuilt)

    def comparable(members: dict[str, bytes]) -> dict[str, bytes]:
        return {n: v for n, v in members.items()
                if not n.endswith((".dist-info/RECORD", ".dist-info/WHEEL")) and not n.endswith(BINARY_SUFFIXES)}

    ca, cb = comparable(a), comparable(b)
    if set(ca) != set(cb):
        raise GateError(f"{label}: sdist rebuild changed the file set: {sorted(set(ca) ^ set(cb))[:5]}")
    for name in ca:
        if ca[name] != cb[name]:
            raise GateError(f"{label}: sdist rebuild differs from the published wheel: {name}")
    return {"compared_files": len(ca), "identical": True, "rebuilt_wheel": rebuilt.name,
            "binary_members_in_original": sorted(n for n in a if n.endswith(BINARY_SUFFIXES))}


_IMPORT_PROBE = r'''import json, pathlib, sys
from importlib import metadata
site = pathlib.Path(sys.argv[1]).resolve()
mode = sys.argv[2]
sys.path.insert(0, str(site))
import anyfesolver, anyfesolver_results
for module in (anyfesolver, anyfesolver_results):
    if site not in pathlib.Path(module.__file__).resolve().parents:
        raise RuntimeError("installed origin mismatch: " + module.__name__)
if anyfesolver.__version__ != "0.5.0" or metadata.version("ANYfeSolver") != "0.5.0":
    raise RuntimeError("solver version mismatch")
if metadata.version("ANYfeSolver-results") != "2.2.0":
    raise RuntimeError("results version mismatch")
for old in ("anysolver", "anysolver_results"):
    try:
        __import__(old)
    except ModuleNotFoundError:
        pass
    else:
        raise RuntimeError("old import name resolves: " + old)
out = {"old_imports_absent": True, "installed_origin": True}
if mode == "full":
    from anyfesolver import b3_ge
    from anyfesolver.elements import ELEMENT_TYPES, QuadraticBeamElement, create_element
    from anyfesolver.nonlinear_static import NonlinearConvergenceSettings
    if b3_ge.SELECTOR != "b3-ge" or b3_ge.NAME != "B3-GE": raise RuntimeError("B3-GE identity mismatch")
    if "b3-ge" in ELEMENT_TYPES: raise RuntimeError("B3-GE entered generic registry")
    if type(create_element("quadratic_beam", 1, [1, 2, 3])) is not QuadraticBeamElement: raise RuntimeError("legacy B3 default changed")
    if NonlinearConvergenceSettings().line_search == "armijo": raise RuntimeError("Armijo became a default")
    if NonlinearConvergenceSettings(line_search="armijo").line_search != "armijo": raise RuntimeError("Armijo opt-in unavailable")
    try: create_element("b3-ge", 2, [1, 2, 3])
    except ValueError: pass
    else: raise RuntimeError("B3-GE became a generic/default route")
    out.update(legacy_b3_default=True, b3_ge_opt_in_only=True, armijo_opt_in_only=True)
    bands = {"ANYfileio": ((0, 3, 2), (0, 4)), "ANYmaterial": ((0, 2), (0, 3)), "ANYmesher": ((0, 4), (0, 6))}
    found = {}
    for name, (low, high) in bands.items():
        v = metadata.version(name); found[name] = v
        t = tuple(int(p) for p in v.split(".")[:3] if p.isdigit())
        if not (t >= low and t[:len(high)] < high): raise RuntimeError("dependency outside band: " + name + " " + v)
    out["dependencies"] = found
print(json.dumps(out, sort_keys=True, separators=(",", ":")))
'''


def _installed_probe(wheels: list[Path], work: Path, tag: str, full: bool) -> dict[str, Any]:
    site = work / f"{tag}-site"
    install = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "--target", str(site)]
    # Dependencies resolve from the configured index: the solver imports its
    # sibling distributions at import time, so a --no-deps install cannot import.
    _run(install + [str(w.resolve()) for w in wheels], work, _clean_env(), TIMEOUT_SECONDS)
    # Isolated mode keeps the working directory and user site off sys.path.
    output = _run([sys.executable, "-I", "-c", _IMPORT_PROBE, str(site), "full" if full else "import"],
                  work, _clean_env(), PROBE_TIMEOUT_SECONDS)
    old = [p.name for p in site.iterdir() if p.name.lower() in OLD_IMPORT_NAMES]
    if old:
        raise GateError(f"installed tree contains old import directories: {old}")
    return json.loads(output)


def _load_historical() -> Any:
    path = ROOT / "scripts" / "verify_release_047.py"
    spec = importlib.util.spec_from_file_location("_historical_verify_release_047", path)
    if spec is None or spec.loader is None:
        raise GateError("historical verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _historical_evidence() -> dict[str, Any]:
    """Re-hash the frozen 0.4.7 evidence with the untouched historical verifier.

    The result is cited only as evidence bound to the 0.4.7 identities in that
    verifier; it says nothing about the 0.5.0 artifacts.
    """
    historical = _load_historical()
    if historical.VERSION != "0.4.7":
        raise GateError("historical verifier identity changed")
    return {
        "bound_release": "0.4.7",
        "bound_terminal_not_reused": historical.TERMINAL,
        "g7": historical._validate_g7(),
        "performance": historical._validate_performance_evidence(),
        "scope": "frozen 0.4.7 evidence hashes only; no 0.5.0 scientific requalification was performed",
    }


def run(solver_wheel: Path, solver_sdist: Path, results_wheel: Path, results_sdist: Path,
        expected_commit: str, isolated: bool = True, work_dir: Path | None = None) -> dict[str, Any]:
    commit, tree = _git_identity(expected_commit)
    project = _check_project_files()
    solver_src = _tree(ROOT / "src" / SOLVER_NAME)
    results_src = _tree(ROOT / "packages" / "results" / "src" / RESULTS_NAME)

    if not re.fullmatch(rf"{SOLVER_NAME}-{re.escape(SOLVER_VERSION)}-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl",
                        solver_wheel.name):
        raise GateError("solver wheel filename mismatch")
    if results_wheel.name != f"{RESULTS_NAME}-{RESULTS_VERSION}-py3-none-any.whl":
        raise GateError("results wheel filename mismatch")

    members = {"solver": _members(solver_wheel), "results": _members(results_wheel)}
    _same_runtime(_runtime(members["solver"], SOLVER_NAME), solver_src, "solver wheel")
    _same_runtime(_runtime(members["results"], RESULTS_NAME), results_src, "results wheel")
    metadata = {
        "solver_wheel": _check_metadata(_metadata(members["solver"], "solver wheel"), "ANYfeSolver",
                                        SOLVER_VERSION, EXPECTED_SOLVER_REQUIRES, "solver wheel"),
        "results_wheel": _check_metadata(_metadata(members["results"], "results wheel"), "ANYfeSolver-results",
                                         RESULTS_VERSION, EXPECTED_RESULTS_REQUIRES, "results wheel"),
    }
    sdists = {
        "solver": _check_sdist(solver_sdist, SOLVER_NAME, SOLVER_VERSION, SOLVER_NAME, solver_src,
                               EXPECTED_SOLVER_REQUIRES, True),
        "results": _check_sdist(results_sdist, RESULTS_NAME, RESULTS_VERSION, RESULTS_NAME, results_src,
                                EXPECTED_RESULTS_REQUIRES, False),
    }
    historical = _historical_evidence()

    with tempfile.TemporaryDirectory(prefix="anyfesolver-050-gate-", dir=work_dir) as raw:
        work = Path(raw)
        rebuilt_solver = _rebuild(solver_sdist, work, "solver", isolated)
        rebuilt_results = _rebuild(results_sdist, work, "results", isolated)
        rebuild = {
            "solver": _compare_wheels(solver_wheel, rebuilt_solver, "solver"),
            "results": _compare_wheels(results_wheel, rebuilt_results, "results"),
        }
        # The sdist-rebuilt pair must itself install and import (dependencies resolve from the index).
        rebuilt_install = _installed_probe([rebuilt_solver, rebuilt_results], work, "rebuilt", full=False)
        installed = _installed_probe([solver_wheel, results_wheel], work, "published", full=True)

    return {
        "candidate": {"commit": commit, "tree": tree},
        "checks": {
            "both_pairs_built_from_candidate_source": True,
            "dependency_bands_unchanged": True,
            "historical_verifier_untouched_and_not_reused_as_terminal": True,
            "installed_smoke": True,
            "no_old_import_trees": True,
            "sdist_rebuilds_identical_installable_wheel": True,
        },
        "historical_evidence": historical,
        "installed": installed,
        "metadata": metadata,
        "project": project,
        "rebuild": rebuild,
        "rebuilt_install": rebuilt_install,
        "schema": SCHEMA,
        "sdists": sdists,
        "shipped_source": {
            "solver_files": len(solver_src),
            "results_files": len(results_src),
            "solver_inventory": _inventory(solver_src),
            "results_inventory": _inventory(results_src),
        },
        "terminal": TERMINAL,
        "versions": {"ANYfeSolver": SOLVER_VERSION, "ANYfeSolver-results": RESULTS_VERSION},
        "wheels": {
            "solver": {"name": solver_wheel.name, **_record(solver_wheel.read_bytes()),
                       "files": _inventory(members["solver"])},
            "results": {"name": results_wheel.name, **_record(results_wheel.read_bytes()),
                        "files": _inventory(members["results"])},
        },
    }


ALLOWED_INDEXES = ("https://test.pypi.org", "https://pypi.org")


def _fetch_index_files(index: str, name: str, version: str) -> dict[str, str]:
    """Return {filename: sha256} the index serves for one release (empty if none)."""
    import urllib.error
    import urllib.request

    if index not in ALLOWED_INDEXES:
        raise GateError(f"index is not an allowed host: {index}")
    url = f"{index}/pypi/{name}/{version}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "anyfesolver-release-gate"})
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise GateError(f"index query failed with HTTP {error.code}: {url}") from error
    files = {}
    for item in payload.get("urls", []):
        files[item["filename"]] = item["digests"]["sha256"]
    return files


def plan_upload(remote: dict[str, str], dist_dir: Path, missing_dir: Path | None) -> dict[str, Any]:
    """Compare local files with the index by per-file digest; fail closed on any conflict.

    Only candidate files that are absent from the index are scheduled for upload.
    A file present with a different digest, or any remote file that is not a
    candidate file, is an error: nothing is overwritten and nothing is skipped
    on filename alone.
    """
    local = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(dist_dir.iterdir()) if p.is_file()}
    if not local:
        raise GateError(f"no candidate files in {dist_dir}")
    conflicts = sorted(n for n in local if n in remote and remote[n] != local[n])
    if conflicts:
        raise GateError(f"index holds different bytes for: {conflicts}")
    unexpected = sorted(set(remote) - set(local))
    if unexpected:
        raise GateError(f"index holds files that are not candidate files: {unexpected}")
    missing = sorted(set(local) - set(remote))
    if missing_dir is not None:
        if missing_dir.exists() and any(missing_dir.iterdir()):
            raise GateError("missing-file directory must start empty")
        missing_dir.mkdir(parents=True, exist_ok=True)
        for name in missing:
            (missing_dir / name).write_bytes((dist_dir / name).read_bytes())
    return {"present_matching": sorted(set(local) - set(missing)), "missing": missing}


def _index_plan_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="verify_release_050.py index-plan")
    parser.add_argument("--index", required=True, choices=ALLOWED_INDEXES)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--missing-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true",
                        help="fail unless the index already serves every candidate file")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = plan_upload(_fetch_index_files(args.index, args.name, args.version),
                           args.dist_dir, args.missing_dir)
    except (GateError, OSError, KeyError, ValueError) as error:
        print(f"index plan failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(plan, sort_keys=True))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"missing={len(plan['missing'])}\n")
    if args.require_complete and plan["missing"]:
        print("index does not yet serve every candidate file", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["index-plan"]:
        return _index_plan_main(arguments[1:])
    argv = arguments
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("solver-wheel", "solver-sdist", "results-wheel", "results-sdist"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-build-isolation", action="store_true",
                        help="rebuild sdists with the current environment instead of isolated build deps")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = run(args.solver_wheel.resolve(), args.solver_sdist.resolve(),
                     args.results_wheel.resolve(), args.results_sdist.resolve(),
                     args.expected_commit, not args.no_build_isolation,
                     args.work_dir.resolve() if args.work_dir else None)
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        with args.output.open("xb") as stream:
            stream.write(raw)
    except (GateError, OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        print(f"packaging gate failed: {error}", file=sys.stderr)
        return 1
    print(sha256(raw).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
