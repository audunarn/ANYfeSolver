"""Experimental reference-arclength force and fixed-spatial couple on GE beams.

Uses the eccentric two-shell/two-beam panel from the gravity example. Densities
act along the native reference axis; this is not a follower load or a welded-line
qualification. Fixed spatial couples have no endpoint potential.
"""
from contextlib import ExitStack
import json
from anyfesolver import stiffened_shell as ss
from stiffened_shell_ge_gravity import definition


def workflow():
    with ExitStack() as stack:
        a=ss.create_analysis(definition=definition())
        b=ss.create_analysis(definition=definition())
        stack.callback(a.close);stack.callback(b.close)
        load=ss.GEBeamDistributedLoad(a.preflight()['identity'],tuple(
            ss.GEBeamLineAction(i,(0.,0.,-100.),(3.,5.,2.)) for i in (1,2)))
        first=a.solve(load,(.25,))
        assert first.status=='completed'
        b.restore(a.checkpoint())
        final=a.solve(load,(.5,1.));resumed=b.solve(load,(.5,1.))
        assert final.status==resumed.status=='completed'
        assert a.checkpoint()==b.checkpoint()
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        assert final.distributed_load.potential is None
        return dict(status=final.status,experimental=True,production_qualified=False,
            accepted_factors=final.accepted_factors,applied_force_N=final.distributed_load.applied_force,
            applied_moment_Nm=final.distributed_load.applied_moment_about_origin,
            force_work_J=final.distributed_load.force_potential,total_potential=None,
            equilibrium=final.free_equilibrium_ratio,compatibility=final.constraint_max)


if __name__=='__main__':print(json.dumps(workflow(),indent=2))
