"""Experimental supported reference modes of an eccentric rigid-footprint panel.

SI units and illustrative low stiffness; this is not an engineering frequency
reference. Beam mass uses explicit density, area and geometric bending inertias
at the centroid, with polar mass inertia Iy+Iz (not the torsion constant J).
The selected-node rigid law locks normal spin. No prestress, free-free, weld,
drilling independence or complete spectral qualification is claimed.
"""
import json

import numpy as np
from scipy import sparse
from anyfesolver import stiffened_shell as api


def main():
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
             4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.)}
    shells = (api.Q4PanelShell('left', (1, 2, 5, 4), 1000., .3, .15, density_kg_m3=2.),
              api.Q4PanelShell('right', (2, 3, 6, 5), 1000., .3, .15, density_kg_m3=2.))
    area, iy, iz, density = .04, .00015, .0003, 7.5
    section = api.LinearSection.isotropic(E=2000., nu=.25, area=area,
        Iy=iy, Iz=iz, J=.0004, ky=.8, kz=.7)
    mass = api.MassMetadata('explicit-centroid-density-7.5', (0., 0., 0.),
        np.diag((density*area,)*3+(density*(iy+iz), density*iy, density*iz)))
    members = tuple(api.PhysicalB2Member(f'beam-{i}', f'source-{i}',
        (10+i, 11+i), ((float(i), .5, .2), (float(i+1), .5, .2)),
        (0., 1., 0.), 'section', section, mass=mass) for i in range(2))
    footprint = api.B2NodalRigidFootprint('right-patch', (2, 3, 6, 5), 11,
                                         ((2, 3, 6, 5),))
    # Sorted shell nodes, then beam nodes 10/11/12; clamp the physical left ends.
    fixed = set(range(6)) | set(range(18, 24)) | set(range(36, 42))
    free = [i for i in range(54) if i not in fixed]
    definition = api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, (), sparse.eye(54, format='csr')[:, free],
        np.zeros(54), (footprint,))
    analysis = api.create_analysis(definition=definition)
    try:
        before = analysis.solve_linear(np.zeros(54))
        result = analysis.modes(api.B2ReferenceModesRequest(count=3))
        after = analysis.solve_linear(np.zeros(54))
        if before.status != 'completed' or after.status != 'completed' or result.status != 'ok':
            raise RuntimeError(result.failure_reason or 'reference workflow failed')
        np.testing.assert_array_equal(before.displacements, after.displacements)
        assert before.factorization_count == after.factorization_count == 1
        print(json.dumps(dict(status=result.status, experimental=True,
            production_qualified=False, footprint_law=result.footprint_law_id,
            footprint_ids=result.footprint_ids, frequencies_hz=result.frequencies_hz,
            equation_residuals=result.equation_residuals,
            constraint_residuals=result.constraint_residuals,
            constraint_rows=len(result.constraint_rows), normal_spin=result.normal_spin,
            modal_factorizations=result.factorization_count,
            static_factorizations=after.factorization_count), indent=2))
    finally:
        analysis.close()


if __name__ == '__main__':
    main()
