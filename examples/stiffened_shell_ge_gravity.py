"""Experimental native GE gravity on a two-shell/two-element panel.

Physical section mass is supplied independently of stiffness. This headless
example demonstrates explicit objective Q4 selection, physical gravity recovery,
checkpoint restoration and continued stepping, not G4 qualification.
"""
from contextlib import ExitStack
import json
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss


def definition():
    e, nu, rho = 210e9, .3, 7850.
    g=e/(2*(1+nu));area=.01;iy=iz=8.333333333333333e-6
    section=b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*area,g*.008,g*.008,g*(iy+iz),e*iy,e*iz)),np.eye(6),1e20,1e9)
    mass=rho*area;offset=np.array([0.,.015,.02])
    skew=np.array([[0.,-offset[2],offset[1]],[offset[2],0.,-offset[0]],[-offset[1],offset[0],0.]])
    inertia=np.block([[mass*np.eye(3),-mass*skew],
                      [mass*skew,rho*np.diag((iy+iz,iy,iz))-mass*skew@skew]])
    coordinates=np.array([(x,-.5,.1) for x in (0.,.5,1.,1.5,2.)])
    beams=tuple(b3_ge.define_beam('b3-ge',i+1,tuple(101+2*i+j for j in range(3)),
        coordinates[2*i:2*i+3],np.tile(np.eye(3),(3,1,1)),section,inertia) for i in range(2))
    shell=ss.ShellPanel((1,2,3,4,5,6),
        ((0.,-.5,0.),(1.,-.5,0.),(2.,-.5,0.),(0.,.5,0.),(1.,.5,0.),(2.,.5,0.)),
        ((1,2,5,4),(2,3,6,5)),.07,e,nu,rho)
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,shell,beams,
        tuple(ss.PointConnection(i+1,101+2*i) for i in range(3)),
        tuple(ss.FixedCoordinate('shell',node,axis) for node in (1,4) for axis in range(6)),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)


def workflow():
    with ExitStack() as stack:
        original=ss.create_analysis(definition=definition())
        restarted=ss.create_analysis(definition=definition())
        stack.callback(original.close);stack.callback(restarted.close)
        gravity=ss.GEBodyLoad(original.preflight()['identity'],(0.,0.,-9.81))
        prefix=original.solve(gravity,(.25,))
        assert prefix.status == 'completed' and prefix.accepted_factors == (.25,)
        saved=original.checkpoint();restarted.restore(saved)
        assert restarted.checkpoint() == saved
        final=original.solve(gravity,(.5,1.))
        continued=restarted.solve(gravity,(.5,1.))
        assert final.status == continued.status == 'completed'
        assert final.accepted_factors == continued.accepted_factors == (.25,.5,1.)
        assert original.checkpoint() == restarted.checkpoint()
        assert final.free_equilibrium_ratio <= 1e-8 and final.constraint_max <= 1e-9
        return dict(status=final.status,experimental=True,production_qualified=False,
            mass_kg=final.body_load.total_mass,applied_force_N=final.body_load.applied_force,
            applied_moment_Nm=final.body_load.applied_moment_about_origin,
            accepted_factors=final.accepted_factors,equilibrium=final.free_equilibrium_ratio,
            compatibility=final.constraint_max,checkpoint_bytes=len(original.checkpoint()))


if __name__ == '__main__':
    print(json.dumps(workflow(),indent=2))
