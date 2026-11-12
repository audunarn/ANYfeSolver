"""Experimental reference-linear weight loads on a two-member stiffened panel.

SI units, centroidal solid rectangular beams, separate shell plate mass. Point
attachments are reference-linear ports, not qualified bonded lines or welds.
Typed loads retain provenance and return separate equilibrated force/strain
fields with body loading and interior point reactions. These fields are not a
loaded interior pose satisfying the original nodal-trace attachment constraints.
Finite-configuration gravity, GE loading and mechanical qualification remain open.
"""
import argparse
import json

import numpy as np
from scipy import sparse
from anyfesolver import stiffened_shell as api


def definition(mesh="mixed"):
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
             4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.)}
    rho, young, poisson, thickness = 7850., 210e9, .3, .02
    shells = []
    for name, ids in (("left", (1, 2, 5, 4)), ("right", (2, 3, 6, 5))):
        if mesh == "s3" or (mesh == "mixed" and name == "right"):
            shells.extend((api.S3V2DPanelShell(name, ids[:3], young, poisson, thickness,
                                              (0., 0., 1.), rho),
                           api.S3V2DPanelShell(name+"-upper", (ids[0], ids[2], ids[3]),
                                              young, poisson, thickness, (0., 0., 1.), rho)))
        else:
            shells.append(api.Q4PanelShell(name, ids, young, poisson, thickness, density_kg_m3=rho))
    width, height = .04, .06
    area, iy, iz = width*height, width*height**3/12., height*width**3/12.
    # Saint-Venant rectangular torsion series (long side height, short width).
    odd = np.arange(1., 1002., 2.)
    torsion = float(height*width**3/3.*(1.-192.*width/(np.pi**5*height)*
                                     np.sum(np.tanh(odd*np.pi*height/(2.*width))/odd**5)))
    section = api.LinearSection.isotropic(E=young, nu=poisson, area=area,
                                          Iy=iy, Iz=iz, J=torsion, ky=5./6., kz=5./6.)
    mass = api.MassMetadata("steel-centroidal-rectangle", (0., 0., 0.),
        np.diag((rho*area,)*3+(rho*(iy+iz), rho*iy, rho*iz)))
    # The plate spans z=-.01..+.01; the beam rectangle spans z=.01..+.07.
    # No effective shell flange or repeated parallel-axis contribution is added.
    members = tuple(api.PhysicalB2Member(f"beam-{i}", f"source-{i}", (10+i, 11+i),
        ((float(i), .5, .04), (float(i+1), .5, .04)), (0., 1., 0.),
        "solid-rectangle", section, mass=mass) for i in range(2))
    points = (api.Q4B2Point("left-port", "left", "beam-0", .5),
              api.Q4B2Point("right-port", "right", "beam-1", .5))
    # Explicit shell1..6 then beam10..12 numbering. Clamp both physical left
    # ends and restrain normal spin at the other shell nodes in this example.
    fixed = set(range(6)) | set(range(18, 24)) | set(range(36, 42))
    fixed.update(6*i+5 for i in (1, 2, 4, 5))
    free = [i for i in range(54) if i not in fixed]
    return api.B2MixedPanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1, nodes,
        tuple(shells), members, points, sparse.eye(54, format="csr")[:, free], np.zeros(54))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=("q4", "mixed", "s3"), default="mixed")
    args = parser.parse_args()
    analysis = api.create_analysis(definition=definition(args.mesh))
    try:
        body = analysis.body_acceleration_load((0., 0., -9.81))
        inventory = []
        for count in (1, 10, 100):
            cases = tuple((f"weight-{i:03}", analysis.body_acceleration_load(
                (0., 0., -9.81*(1.+i/count)))) for i in range(count))
            result = analysis.solve_linear_many(cases, selected_dofs=(49, 50),
                                                recover_members={"beam-0": (0., .5, 1.)})
            if result.status != "completed" or len(result.cases) != count or result.factorization_count != 1:
                raise RuntimeError(result.failure_reason or "body-load batch failed")
            fields = result.cases[0].equilibrated_sections[0]
            inventory.append(dict(cases=count, status=result.status, factorization_count=result.factorization_count,
                first_displacements=result.cases[0].selected_displacements,
                first_section_policy=fields.policy, first_section_stations=fields.stations,
                first_resultants_left=fields.resultants_left, first_resultants_right=fields.resultants_right))
        static = analysis.solve_linear(body, recover_members={})
        if static.status != "completed":
            raise RuntimeError(static.failure_reason)
        force = body.load.reshape(-1, 6)[:, :3].sum(axis=0)
        reaction = np.asarray(static.affine_reactions).reshape(-1, 6)[:, :3].sum(axis=0)
        np.testing.assert_allclose(force+reaction, 0., atol=1e-8, rtol=0.)
        print(json.dumps(dict(mesh=args.mesh, experimental=True, production_qualified=False,
            body_load_policy=body.policy, member_recovery=body.member_recovery,
            total_weight_force_N=force.tolist(), total_support_force_N=reaction.tolist(),
            batches=inventory), indent=2))
    finally:
        analysis.close()


if __name__ == "__main__":
    main()
