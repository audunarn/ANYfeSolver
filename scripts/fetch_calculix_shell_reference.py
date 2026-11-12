"""Fetch the upstream CalculiX shell convergence reference files.

This script downloads only the lightweight source files needed to generate the
reference case locally.  It does not run CGX/CCX and does not commit generated
results.  After running the upstream workflow and producing .inp/.frd pairs, the
ordinary reference-case discovery tests will find them automatically.

Usage from repository root:

    python scripts/fetch_calculix_shell_reference.py

Optional output directory:

    python scripts/fetch_calculix_shell_reference.py --output tests/reference_cases/calculix_shell_convergence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from anyfesolver.reference_cases import upstream_calculix_reference_manifest


DEFAULT_OUTPUT = Path("tests/reference_cases/calculix_shell_convergence")


def _download_text(url: str, timeout: float = 30.0) -> str:
    with urlopen(url, timeout=timeout) as response:  # nosec B310 - explicit developer tool for public reference download
        return response.read().decode("utf-8")


def fetch_calculix_shell_reference(output_dir: Path = DEFAULT_OUTPUT, overwrite: bool = False) -> Path:
    manifest = upstream_calculix_reference_manifest()
    case = next(entry for entry in manifest if entry["name"] == "calculix_examples_shell_convergence")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_base_url = case["raw_base_url"].rstrip("/")

    for filename in case["source_files"]:
        target = output_dir / filename
        if target.exists() and not overwrite:
            continue
        target.write_text(_download_text(f"{raw_base_url}/{filename}"), encoding="utf-8")

    metadata = {
        "name": case["name"],
        "kind": case["kind"],
        "source_repository": case["repository"],
        "source_ref": case["ref"],
        "source_directory": case["directory"],
        "requires_generated_includes": case["requires_generated_includes"],
        "expected_outputs": case["expected_outputs"],
        "reference_values": case.get("reference_values", {}),
        "notes": case.get("notes", []),
        "workflow": [
            "Run the downloaded upstream test.py/shell.fbd workflow with CGX/CCX installed.",
            "Copy or keep generated .inp/.frd pairs in this directory.",
            "Run python -m pytest tests/test_fe_solver_reference_cases.py -q to verify discovery.",
        ],
    }
    (output_dir / "reference_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to write upstream reference files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloaded files")
    args = parser.parse_args()

    output = fetch_calculix_shell_reference(args.output, overwrite=args.overwrite)
    print(f"Fetched upstream CalculiX shell reference sources to: {output}")
    print("Install CGX/CCX and run the upstream workflow there to generate concrete .inp/.frd result pairs.")


if __name__ == "__main__":
    main()
