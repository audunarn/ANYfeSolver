"""Headless experimental reference-linear selected-node footprint example.

This elastic Q4/physical-B2 example locks normal spin. It is not a qualified
weld, distributing coupling, finite analysis or physical drilling assessment.
"""
from dataclasses import asdict
import json

import numpy as np
from scipy import sparse
from anyfesolver import stiffened_shell as api


def main():
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
             4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.)}
    shells = (api.Q4PanelShell("left", (1, 2, 5, 4), 1000., .3, .15),
              api.Q4PanelShell("right", (2, 3, 6, 5), 1000., .3, .15))
    section = api.LinearSection.isotropic(E=2000., nu=.25, area=.04,
        Iy=.00015, Iz=.0003, J=.0004, ky=.8, kz=.7)
    members = tuple(api.PhysicalB2Member(f"beam-{i}", f"source-{i}",
        (10+i, 11+i), ((float(i), .5, .2), (float(i+1), .5, .2)),
        (0., 1., 0.), "section", section) for i in range(2))
    footprint = api.B2NodalRigidFootprint("right-patch", (2, 3, 6, 5), 11,
                                          ((2, 3, 6, 5),))
    # Typed domain ordering: sorted six shell nodes, then beam nodes10,11,12.
    fixed = set(range(6)) | set(range(18, 24)) | set(range(36, 42))
    free = [i for i in range(54) if i not in fixed]
    definition = api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, (), sparse.eye(54, format="csr")[:, free],
        np.zeros(54), (footprint,))
    owner = api.create_analysis(definition=definition)
    try:
        load = np.zeros(54); load[49] = 2.; load[50] = -.5
        load[51] = .1; load[53] = .15
        result = owner.solve_linear_many((("elastic", load),),
            selected_dofs=(49, 50, 53), recover_members={},
            selected_footprints=api.B2NodalRigidFootprintSelection(("right-patch",)))
        if result.status != "completed":
            raise RuntimeError(result.failure_reason)
        print(json.dumps(dict(status=result.status, experimental=True,
            production_qualified=False, law=result.footprint_law_id,
            factorization_count=result.factorization_count,
            case=asdict(result.cases[0])), indent=2))
    finally:
        owner.close()


if __name__ == "__main__":
    main()
