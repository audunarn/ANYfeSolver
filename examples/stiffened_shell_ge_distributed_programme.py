"""Signed distributed loading on the experimental eccentric point-connected panel.

Absolute densities are staged at positive programme positions. The two stages
are +100 and -100 N/m in global Z. No zero-load equilibrium point is requested:
convergence at zero applied load remains unresolved under the current assessment.
This is an elastic lifecycle example, not bonded-line or capacity qualification.
"""
from contextlib import ExitStack
import json

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from stiffened_shell_ge_gravity import definition


def workflow():
    with ExitStack() as stack:
        direct=ss.create_analysis(definition=definition());stack.callback(direct.close)
        restored=ss.create_analysis(definition=definition());stack.callback(restored.close)
        programme=ss.GEBeamDistributedProgramme(direct.preflight()['identity'],tuple(
            tuple(ss.GEBeamLineAction(i,(0.,0.,value)) for i in (1,2))
            for value in (100.,-100.)))
        token=CancellationToken();control=ss.AdaptiveIncrementControl(2,16)
        try:
            direct.solve(programme,(1.,2.),increment_control=control,
                cancellation_token=token,on_accepted=lambda r:token.cancel('accepted stage'))
        except SolveCancelled:
            pass
        else:
            raise AssertionError('expected accepted-stage cancellation')
        prefix=direct.recover()
        assert prefix.load_position==1. and prefix.distributed_load.applied_force[2]>0.
        restored.restore(direct.checkpoint())
        final=direct.solve(programme,(1.,2.),increment_control=control)
        resumed=restored.solve(programme,(1.,2.),increment_control=control)
        assert final.status==resumed.status=='completed' and final.load_programme_complete
        assert direct.checkpoint()==restored.checkpoint()
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        assert final.distributed_load.applied_force[2]<0.
        return dict(status=final.status,experimental=True,production_qualified=False,
            programme_identity=programme.identity,accepted_positions=final.accepted_factors,
            final_force_N=final.distributed_load.applied_force,restart_identical=True,
            equilibrium=final.free_equilibrium_ratio,compatibility=final.constraint_max)


if __name__=='__main__':print(json.dumps(workflow(),indent=2))
