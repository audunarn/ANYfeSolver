"""Validate ANYsolver licence metadata and release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LICENSE_SHA256 = (
    "1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5"
)
EXPECTED_PROJECT_LICENSE = "MPL-2.0"
EXPECTED_RELEASE = "0.5.0a1"
EXPECTED_PROJECT_NAME = "ANYfeSolver"
EXPECTED_RESULTS_NAME = "ANYfeSolver-results"
EXPECTED_RESULTS_RELEASE = "2.2.0a1"
EXPECTED_RESULTS_REQUIREMENT = "ANYfeSolver-results==2.2.0a1"
REQUIRED_NOTICE_FILES = {"LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md"}
REQUIRED_SDIST_FILES = REQUIRED_NOTICE_FILES | {
    "LICENSING.md",
    "dependency-licenses.json",
}


class ComplianceError(ValueError):
    """A licence or release invariant is not satisfied."""


def _normalized_license_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return ("\n".join(line.rstrip() for line in text.split("\n")).rstrip() + "\n").encode()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    made: dict[str, Any] = {}
    for key, value in pairs:
        if key in made:
            raise ComplianceError(f"duplicate JSON key: {key}")
        made[key] = value
    return made


def _load_inventory(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ComplianceError(f"nonfinite JSON value: {value}")
        ),
    )
    if not isinstance(data, dict):
        raise ComplianceError("dependency inventory must be an object")
    canonical = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.read_text(encoding="utf-8").replace("\r\n", "\n") != canonical:
        raise ComplianceError("dependency inventory is not canonical JSON")
    return data


def _dependency_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    if match is None:
        raise ComplianceError(f"cannot parse dependency requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def validate_repository(root: Path = ROOT) -> None:
    errors: list[str] = []
    license_path = root / "LICENSE"
    actual_hash = hashlib.sha256(_normalized_license_bytes(license_path)).hexdigest()
    if actual_hash != EXPECTED_LICENSE_SHA256:
        errors.append(f"LICENSE hash mismatch: {actual_hash}")

    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    if project.get("name") != EXPECTED_PROJECT_NAME:
        errors.append(f"project name is not {EXPECTED_PROJECT_NAME}")
    if project.get("version") != EXPECTED_RELEASE:
        errors.append(f"project version is not {EXPECTED_RELEASE}")
    if EXPECTED_RESULTS_REQUIREMENT not in project.get("dependencies", ()):
        errors.append(f"project lacks dependency {EXPECTED_RESULTS_REQUIREMENT}")

    results_toml = root / "packages" / "results" / "pyproject.toml"
    with results_toml.open("rb") as stream:
        results = tomllib.load(stream)["project"]
    if results.get("name") != EXPECTED_RESULTS_NAME:
        errors.append(f"results name is not {EXPECTED_RESULTS_NAME}")
    if results.get("version") != EXPECTED_RESULTS_RELEASE:
        errors.append(f"results version is not {EXPECTED_RESULTS_RELEASE}")
    if results.get("license") != EXPECTED_PROJECT_LICENSE:
        errors.append("results license is not MPL-2.0")
    if set(results.get("license-files", ())) != REQUIRED_NOTICE_FILES:
        errors.append("results license-files does not bind every required notice")
    for notice in sorted(REQUIRED_NOTICE_FILES):
        if not (root / "packages" / "results" / notice).is_file():
            errors.append(f"results notice file is missing: {notice}")
    # The results package carries its own LICENSE copy; it must be the same text.
    results_license = root / "packages" / "results" / "LICENSE"
    if results_license.is_file():
        results_hash = hashlib.sha256(_normalized_license_bytes(results_license)).hexdigest()
        if results_hash != EXPECTED_LICENSE_SHA256:
            errors.append(f"results LICENSE hash mismatch: {results_hash}")
    if project.get("license") != EXPECTED_PROJECT_LICENSE:
        errors.append(f"project license is not {EXPECTED_PROJECT_LICENSE}")
    if set(project.get("license-files", ())) != REQUIRED_NOTICE_FILES:
        errors.append("project license-files does not bind every required notice")
    if any(str(item).startswith("License ::") for item in project.get("classifiers", ())):
        errors.append("legacy license classifier conflicts with the SPDX expression")

    inventory = _load_inventory(root / "dependency-licenses.json")
    if set(inventory) != {
        "dependencies",
        "project",
        "project_licence",
        "release",
        "schema",
    }:
        errors.append("dependency inventory top-level schema is incorrect")
    if inventory.get("schema") != "anyfesolver.dependency-licences-v1":
        errors.append("dependency inventory schema ID is incorrect")
    if inventory.get("project") != EXPECTED_PROJECT_NAME:
        errors.append("dependency inventory project is incorrect")
    if inventory.get("project_licence") != EXPECTED_PROJECT_LICENSE:
        errors.append("dependency inventory project licence is incorrect")
    if inventory.get("release") != EXPECTED_RELEASE:
        errors.append("dependency inventory release is incorrect")

    records = inventory.get("dependencies")
    if not isinstance(records, list):
        errors.append("dependency inventory records must be a list")
        records = []
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "distribution",
            "licence",
            "requirement",
            "scope",
            "status",
        }:
            errors.append("dependency record schema is incorrect")
            continue
        name = _dependency_name(str(record["distribution"]))
        if name in by_name:
            errors.append(f"duplicate dependency inventory record: {name}")
        by_name[name] = record
        licence = str(record["licence"])
        status = str(record["status"])
        if any(token in licence for token in ("GPL", "AGPL", "LGPL")) and not status.startswith(
            "reviewed-transitional-"
        ):
            errors.append(f"copyleft dependency lacks explicit review status: {name}")

    requirements: list[tuple[str, str]] = [
        (str(item), "runtime") for item in project.get("dependencies", ())
    ]
    for group, entries in project.get("optional-dependencies", {}).items():
        scope = "development" if group == "dev" else "optional-runtime"
        requirements.extend((str(item), scope) for item in entries)
    for requirement, scope in requirements:
        name = _dependency_name(requirement)
        record = by_name.get(name)
        if record is None:
            errors.append(f"dependency is absent from inventory: {requirement}")
            continue
        if record["requirement"] != requirement:
            errors.append(f"dependency requirement mismatch for {name}")
        if record["scope"] != scope:
            errors.append(f"dependency scope mismatch for {name}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if "licensed under the Mozilla Public\nLicense 2.0" not in readme:
        errors.append("README MPL-2.0 statement is missing")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if "## 0.5.0 - " not in changelog:
        errors.append("0.5.0 changelog section is missing")
    if "## 0.4.7 - 2026-09-21" not in changelog:
        errors.append("0.4.7 changelog section is missing")
    if "## 0.4.6 - 2026-09-16" not in changelog:
        errors.append("0.4.6 changelog section is missing")
    if "## 0.4.3 - 2026-09-09" not in changelog:
        errors.append("0.4.3 changelog history is missing")
    if "coordinated ecosystem dependency bands" not in changelog:
        errors.append("0.4.1 dependency correction note is missing")
    if "## 0.4.0 - 2026-09-03" not in changelog:
        errors.append("0.4.0 licence-transition history is missing")
    if "Relicense ANYsolver source releases" not in changelog:
        errors.append("0.4.0 relicensing note is missing")
    if "GPL-3.0-or-later" in (root / "pyproject.toml").read_text(encoding="utf-8"):
        errors.append("obsolete GPL project metadata remains")
    for filename in (*REQUIRED_SDIST_FILES, "README.md", "docs/releases/ANYfeSolver_0.5.0.md"):
        if not (root / filename).is_file():
            errors.append(f"required release file is missing: {filename}")

    if errors:
        raise ComplianceError("; ".join(errors))


def _artifact_members(path: Path) -> tuple[set[str], str | None]:
    metadata: str | None = None
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise ComplianceError("wheel must contain exactly one METADATA file")
            metadata = archive.read(metadata_names[0]).decode("utf-8")
            return names, metadata
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return {member.name for member in archive.getmembers() if member.isfile()}, None
    raise ComplianceError(f"unsupported release artifact: {path.name}")


def validate_artifact(
    path: Path,
    *,
    name: str = EXPECTED_PROJECT_NAME,
    release: str = EXPECTED_RELEASE,
    results: bool = False,
) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ComplianceError(f"release artifact is missing or empty: {path}")
    names, metadata = _artifact_members(path)
    basenames = {Path(member).name for member in names}
    if path.suffix == ".whl" or results:
        required = REQUIRED_NOTICE_FILES
    else:
        required = REQUIRED_SDIST_FILES
    missing = sorted(required - basenames)
    if missing:
        raise ComplianceError(f"artifact {path.name} is missing: {', '.join(missing)}")
    if metadata is not None:
        normalized = metadata.replace("\r\n", "\n")
        if f"Version: {release}\n" not in normalized:
            raise ComplianceError(f"wheel metadata version is not {release}")
        if f"Name: {name}\n" not in normalized:
            raise ComplianceError(f"wheel metadata name is not {name}")
        if not any(
            marker in normalized
            for marker in ("License-Expression: MPL-2.0\n", "License: MPL-2.0\n")
        ):
            raise ComplianceError("wheel metadata does not declare MPL-2.0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", default=[], type=Path)
    parser.add_argument("--results-artifact", action="append", default=[], type=Path)
    args = parser.parse_args(argv)
    try:
        validate_repository()
        for artifact in args.artifact:
            validate_artifact(artifact.resolve())
        for artifact in args.results_artifact:
            validate_artifact(
                artifact.resolve(),
                name=EXPECTED_RESULTS_NAME,
                release=EXPECTED_RESULTS_RELEASE,
                results=True,
            )
    except (ComplianceError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"license compliance failed: {error}", file=sys.stderr)
        return 1
    print(
        f"license compliance passed for {EXPECTED_PROJECT_NAME} {EXPECTED_RELEASE} "
        f"and {EXPECTED_RESULTS_NAME} {EXPECTED_RESULTS_RELEASE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
