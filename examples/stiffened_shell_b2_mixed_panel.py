"""Experimental physical B2 with mixed Q4/S3-V2D or all-S3 static shells.

Finite-vertex point ports transfer work through shell translations. They are
not BL5 bonded lines or qualified welds. S3 mass, modes and nonlinear analysis
are outside this reference-linear example.
"""
import argparse
from dataclasses import asdict
import json

import numpy as np
from scipy import sparse
from anyfesolver import stiffened_shell as api


def definition(mesh="mixed"):
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
             4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.)}
    shells = [api.S3V2DPanelShell("left-low", (1, 2, 5), 1000., .3, .15, (0., 0., 1.)),
              api.S3V2DPanelShell("left-high", (1, 5, 4), 1000., .3, .15, (0., 0., 1.))]
    if mesh == "mixed":
        shells.append(api.Q4PanelShell("right", (2, 3, 6, 5), 1000., .3, .15))
        right = "right"
    elif mesh == "s3":
        shells.extend((api.S3V2DPanelShell("right-low", (2, 3, 6), 1000., .3, .15, (0., 0., 1.)),
                       api.S3V2DPanelShell("right-high", (2, 6, 5), 1000., .3, .15, (0., 0., 1.))))
        right = "right-low"
    else:
        raise ValueError("mesh must be mixed or s3")
    section = api.LinearSection.isotropic(E=2000., nu=.25, area=.04,
        Iy=.00015, Iz=.0003, J=.0004, ky=.8, kz=.7)
    members = tuple(api.PhysicalB2Member(f"beam-{i}", f"source-{i}",
        (10+i, 11+i), ((.2+i, .5, .2), (1.2+i, .5, .2)),
        (0., 1., 0.), "section", section) for i in range(2))
    points = (api.Q4B2Point("left-point", "left-low", "beam-0", .5, (0., .08, -.03)),
              api.Q4B2Point("right-point", right, "beam-1", .5, (0., .08, -.03)))
    # Sorted shell nodes1..6, then beam nodes10..12; no coordinate merging.
    fixed = set(range(6)) | set(range(18, 24)) | set(range(36, 42))
    fixed.update(6*i+5 for i in (1, 2, 4, 5))
    free = [i for i in range(54) if i not in fixed]
    return api.B2MixedPanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
        nodes, tuple(shells), members, points,
        sparse.eye(54, format="csr")[:, free], np.zeros(54))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=("mixed", "s3"), default="mixed")
    args = parser.parse_args()
    owner = api.create_analysis(definition=definition(args.mesh))
    try:
        load = np.zeros(54)
        load[49], load[50], load[53] = .01, -.005, .001
        inventory = []
        for count in (1, 10, 100):
            cases = tuple((f"case-{i:03}", (1.+i/count)*load) for i in range(count))
            result = owner.solve_linear_many(cases, selected_dofs=(49, 50, 53),
                                             recover_members={})
            if result.status != "completed" or len(result.cases) != count:
                raise RuntimeError(result.failure_reason)
            if result.factorization_count != 1:
                raise RuntimeError("unchanged static operator must use one factorization")
            inventory.append({"cases": count, "status": result.status,
                "factorization_count": result.factorization_count,
                "first": asdict(result.cases[0]), "last": asdict(result.cases[-1])})
        print(json.dumps({"mesh": args.mesh, "experimental": True,
            "production_qualified": False, "capabilities": asdict(owner.capabilities),
            "batches": inventory}, indent=2))
    finally:
        owner.close()


if __name__ == "__main__":
    main()
