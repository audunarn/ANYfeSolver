"""Experimental supported mixed/all-S3 reference modes with declared inertia.

S3 facets use native lumped translational mass and omit all rotary inertia.
Shared Q4 nodes retain Q4 director inertia. This illustrates the discrete
descriptor workflow, not continuum frequencies or complete qualification.
"""
import argparse
from dataclasses import asdict, replace
import json

import numpy as np
from anyfesolver import stiffened_shell as api
from stiffened_shell_b2_mixed_panel import definition as static_definition


def definition(mesh="mixed"):
    static = static_definition(mesh)
    area, iy, iz, density = .04, .00015, .0003, 7.5
    mass = api.MassMetadata("declared-centroid-density-7.5", (0., 0., 0.),
        np.diag((density*area,)*3+(density*(iy+iz), density*iy, density*iz)))
    values = dict(static.__dict__)
    values["shells"] = tuple(replace(shell, density_kg_m3=2.) for shell in static.shells)
    values["members"] = tuple(replace(member, mass=mass) for member in static.members)
    return api.B2MixedReferenceModalDefinition(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=("mixed", "s3"), default="mixed")
    args = parser.parse_args()
    owner = api.create_analysis(definition=definition(args.mesh))
    try:
        before = owner.solve_linear(np.zeros(54))
        result = owner.modes(api.B2ReferenceModesRequest(count=3))
        after = owner.solve_linear(np.zeros(54))
        if result.status != "ok" or before.status != "completed" or after.status != "completed":
            raise RuntimeError(result.failure_reason or "reference workflow failed")
        np.testing.assert_array_equal(before.displacements, after.displacements)
        assert before.factorization_count == after.factorization_count == 1
        print(json.dumps({"mesh": args.mesh, "experimental": True,
            "production_qualified": False, "capabilities": asdict(owner.capabilities),
            "frequencies_hz": result.frequencies_hz,
            "equation_residuals": result.equation_residuals,
            "constraint_residuals": result.constraint_residuals,
            "modal_factorizations": result.factorization_count,
            "static_factorizations": after.factorization_count}, indent=2))
    finally:
        owner.close()


if __name__ == "__main__":
    main()
