"""SOURCE_ONLY_UNRUN: explicit flat native BL5 director-metric prefix/restart.

This example must not be executed without separate numerical authority. Equal
positive weights are an explicit illustrative choice, not a refinement rule.
No capacity, branch uniqueness, scientific acceptance or qualification claim.
"""
import json
import numpy as np
from anyfesolver import stiffened_shell as ss
from examples.stiffened_shell_native_fibre_bl5_mixed_j2_displacement import definition
from examples.stiffened_shell_native_fibre_bl5_displacement import ACTIONS,OPTIONS


def arc_control(owner):
    order=owner.preflight()['arc_metric_node_order']
    shell_nodes=sum(domain=='shell' for domain,node in order)
    metric=ss.NativeDirectorChordMetric(np.full(len(order),1./len(order)),1.,shell_nodes)
    return ss.ArcLengthControl('beam',105,0,metric,1e-6)


def workflow():
    model=definition('mixed','biaxial','mixed','continuous_p1_coarse')
    owner=ss.create_analysis(definition=model);fresh=ss.create_analysis(definition=model)
    try:
        owner.solve(ACTIONS,(.001,),**OPTIONS)
        control=arc_control(owner)
        prefix=owner.continue_arc_length(ACTIONS,control)
        raw=owner.checkpoint();restored=fresh.restore(raw)
        assert restored.status=='accepted_prefix' and fresh.checkpoint()==raw
        final=owner.continue_arc_length(ACTIONS,control)
        repeated=fresh.continue_arc_length(ACTIONS,control)
        assert owner.checkpoint()==fresh.checkpoint()
        return dict(status=final.status,metric='flat_native_bl5_position_director_v1',
            prefix_generation=prefix.generation,generation=final.generation,
            continued_generation=repeated.generation,history=final.history,
            free_equilibrium_ratio=final.free_equilibrium_ratio,constraint_max=final.constraint_max,
            fresh_restore_bytes_equal=True,continued_bytes_equal=True,production_qualified=False)
    finally:owner.close();fresh.close()


if __name__=='__main__':print(json.dumps(workflow(),indent=2))
