"""Fresh-process G7 installed-artifact and consumer probe.

This intentionally checks the admission/interface boundary only.  G1--G6
scientific mechanics are hash-bound inputs and are not recomputed here.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import subprocess
import sys


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(root: Path) -> tuple[str, str]:
    root = root.resolve(strict=True)
    command = ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse"]
    commit = subprocess.check_output(command + ["HEAD"], text=True).strip()
    tree = subprocess.check_output(command + ["HEAD^{tree}"], text=True).strip()
    if len(commit) != 40 or len(tree) != 40:
        raise RuntimeError("invalid candidate Git identity")
    return commit, tree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--solver-tree", required=True)
    parser.add_argument("--anyfem-root", required=True, type=Path)
    parser.add_argument("--anystructure-root", required=True, type=Path)
    parser.add_argument("--g6-confirmation", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    site = args.site.resolve(strict=True)
    sys.path.insert(0, str(site))
    anyfem_root = args.anyfem_root.resolve(strict=True)
    anystructure_root = args.anystructure_root.resolve(strict=True)
    sys.path.append(str(anyfem_root / "src"))
    sys.path.append(str(anystructure_root))

    anyfem_commit, anyfem_tree = git_identity(anyfem_root)
    anystructure_commit, anystructure_tree = git_identity(anystructure_root)
    if len(args.solver_commit) != 40 or len(args.solver_tree) != 40:
        raise RuntimeError("invalid frozen solver identity")

    import numpy as np
    import anyfesolver
    from anyfesolver import b3_ge
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import ELEMENT_TYPES, QuadraticBeamElement, create_element

    origin = Path(anyfesolver.__file__).resolve(strict=True)
    if site not in origin.parents:
        raise RuntimeError("ANYSolver was not imported from the installed target")
    if b3_ge.SELECTOR != "b3-ge" or b3_ge.NAME != "B3-GE":
        raise RuntimeError("B3-GE public identity mismatch")
    if "b3-ge" in ELEMENT_TYPES:
        raise RuntimeError("B3-GE entered the generic element registry")
    if type(create_element("quadratic_beam", 90, [1, 2, 3])) is not QuadraticBeamElement:
        raise RuntimeError("legacy B3 default changed")

    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((10.0, 11.0, 12.0, 13.0, 14.0, 15.0)), np.eye(6), 1.0e6, 1.0
    )
    definition = b3_ge.define_beam(
        b3_ge.SELECTOR,
        7,
        (1, 2, 3),
        np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))),
        np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        section,
        np.diag((2.0, 2.0, 2.0, 0.07, 0.09, 0.11)),
    )
    boundaries = (
        BoundaryCondition(
            "fixed",
            [1],
            {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        ),
    )
    owner = b3_ge.create_analysis(b3_ge.SELECTOR, (definition,), boundaries)
    provenance = json.loads(b3_ge.workflow_provenance(owner))

    fem = importlib.import_module("anyfem.solve.ge_beam3")
    fem_policy = fem.B3GENativeOptIn((definition,))
    fem_roundtrip = fem.B3GENativeOptIn.from_dict(fem_policy.to_dict())
    fem_owner = fem_roundtrip.create_analysis(boundaries)
    if fem_owner.identity != owner.identity:
        raise RuntimeError("ANYfem changed the native definition graph")

    structure = importlib.import_module("anystruct.ge_beam3_optin")
    structure_policy = structure.B3GERuntimeDefinition((definition,))
    structure_roundtrip = structure.B3GERuntimeDefinition.from_dict(
        structure_policy.to_dict()
    )
    structure_owner = structure_roundtrip.create_analysis(boundaries)
    status = structure.b3_ge_runtime_status(structure_roundtrip)
    if structure_owner.identity != owner.identity:
        raise RuntimeError("ANYstructure changed the native definition graph")
    if status["beam_element"] != "B3-GE — geometrically exact Simo–Reissner":
        raise RuntimeError("ANYstructure B3-GE status mismatch")

    old_fem = fem.GeBeam3OptIn(
        np.diag((10.0, 11.0, 12.0, 13.0, 14.0, 15.0)),
        np.diag((2.0, 2.0, 2.0, 0.1, 0.2, 0.3)),
        (0.0, 1.0, 0.0),
    )
    old_fem = fem.GeBeam3OptIn.from_dict(old_fem.to_dict())
    old_element = old_fem.build(8, (4, 5, 6), "steel")
    if old_element.selector != "ge-beam3":
        raise RuntimeError("historical schema-v1 record was reinterpreted")

    result = {
        "admission_sha256": b3_ge.ADMISSION_SHA256,
        "checks": {
            "anyfem_native_identity": True,
            "anystructure_native_identity": True,
            "historical_ge_beam3_preserved": True,
            "installed_origin": True,
            "legacy_b3_default": True,
            "native_provenance": True,
            "no_generic_element_alias": True,
            "status_name": True,
        },
        "definition_sha256": definition.sha256,
        "g6_confirmation_file_sha256": file_sha(args.g6_confirmation.resolve(strict=True)),
        "g7_contract_sha256": file_sha(args.contract.resolve(strict=True)),
        "harness_sha256": file_sha(Path(__file__).resolve(strict=True)),
        "native_owner_identity": owner.identity,
        "native_profile_id": provenance["native"]["profile_id"],
        "repository_candidates": {
            "anyfem": {
                "adapter_sha256": file_sha(anyfem_root / "src/anyfem/solve/ge_beam3.py"),
                "commit": anyfem_commit,
                "tree": anyfem_tree,
            },
            "anyfesolver": {"commit": args.solver_commit, "tree": args.solver_tree},
            "anystructure": {
                "adapter_sha256": file_sha(anystructure_root / "anystruct/ge_beam3_optin.py"),
                "commit": anystructure_commit,
                "tree": anystructure_tree,
            },
        },
        "schema": "B3_GE_G7_INSTALLED_CONFIRMATION_V1",
        "selector": "b3-ge",
        "terminal": "PROVISIONAL_GO_B3_GE_PRODUCTION_OPT_IN",
        "wheel_sha256": file_sha(args.wheel.resolve(strict=True)),
    }
    raw = canonical(result)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(sha256(raw).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
