"""G0A installed 0.4.7 baseline smoke; not a new B3-GE qualification."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--site", required=True, type=Path)
    args = parser.parse_args()

    wheel = args.wheel.resolve(strict=True)
    site = args.site.resolve(strict=True)
    actual_sha256 = sha256(wheel.read_bytes()).hexdigest()
    if actual_sha256 != args.expected_sha256:
        raise RuntimeError("installed baseline wheel SHA-256 mismatch")

    import numpy as np
    import anyfesolver
    from anyfesolver import b3_ge
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import ELEMENT_TYPES, QuadraticBeamElement, create_element

    origin = Path(anyfesolver.__file__).resolve(strict=True)
    if not origin.is_relative_to(site) or version("ANYfeSolver") != "0.4.7":
        raise RuntimeError("baseline import must come from the isolated 0.4.7 wheel")
    if b3_ge.SELECTOR != "b3-ge" or "b3-ge" in ELEMENT_TYPES:
        raise RuntimeError("native B3-GE selector changed the legacy element registry")
    if type(create_element("quadratic_beam", 90, [1, 2, 3])) is not QuadraticBeamElement:
        raise RuntimeError("legacy B3 factory changed")

    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((10.0, 11.0, 12.0, 13.0, 14.0, 15.0)),
        np.eye(6), 1.0e6, 1.0,
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
            {dof: 0.0 for dof in ("ux", "uy", "uz", "rx", "ry", "rz")},
        ),
    )
    owner = b3_ge.create_analysis(b3_ge.SELECTOR, (definition,), boundaries)
    provenance = json.loads(b3_ge.workflow_provenance(owner))
    if provenance["selector"] != "b3-ge" or provenance["default_changed"]:
        raise RuntimeError("public B3-GE provenance changed")

    print(json.dumps({
        "status": "passed_installed_baseline_smoke",
        "import_origin": str(origin),
        "wheel_sha256": actual_sha256,
        "native_owner_identity": owner.identity,
        "admission_sha256": b3_ge.ADMISSION_SHA256,
        "versions": {name: version(name) for name in (
            "ANYfeSolver", "ANYmaterial", "ANYgeometry", "ANYmesher",
            "ANYfileio", "numpy", "scipy",
        )},
        "checks": ["installed_origin", "legacy_factory", "native_owner", "public_provenance"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
