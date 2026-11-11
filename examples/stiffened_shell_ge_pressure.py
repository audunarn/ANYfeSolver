"""Experimental mixed-owner current-area pressure with physical recovery/restart.

Uses the public two-shell/two-beam panel. Pressure acts on the interpolation
surface with its declared sheet orientation, not a beam axis or shell top face.
No aggregate follower-pressure qualification is claimed by this example.
"""
from contextlib import ExitStack
import json
from anyfesolver import stiffened_shell as ss
from stiffened_shell_ge_gravity import definition


def workflow():
    with ExitStack() as stack:
        a=ss.create_analysis(definition=definition());stack.callback(a.close)
        b=ss.create_analysis(definition=definition());stack.callback(b.close)
        request=ss.GEShellPressureLoad(a.preflight()['identity'],tuple(
            ss.GEShellPressureAction(i,100.,'current_surface') for i in (1,2)))
        first=a.solve(request,(.25,));assert first.status=='completed'
        b.restore(a.checkpoint())
        final=a.solve(request,(.5,1.));resumed=b.solve(request,(.5,1.))
        assert final.status==resumed.status=='completed'
        assert a.checkpoint()==b.checkpoint()
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        assert final.pressure_load.potential is None
        return dict(status=final.status,experimental=True,production_qualified=False,
            accepted_factors=final.accepted_factors,force_N=final.pressure_load.applied_force,
            moment_Nm=final.pressure_load.applied_moment_about_origin,potential=None,
            equilibrium=final.free_equilibrium_ratio,compatibility=final.constraint_max)


if __name__=='__main__':print(json.dumps(workflow(),indent=2))
