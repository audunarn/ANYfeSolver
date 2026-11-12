"""Unverified DEV native BL5 tangent-plane prefix and fresh-owner restart.

The explicit metric measures full frames, including shell normal spin. This is
not a capacity, branch uniqueness or director-quotient step-size demonstration.
"""
import json
import numpy as np
from anyfesolver import stiffened_shell as ss
from examples.stiffened_shell_native_fibre_bl5_mixed_j2_displacement import definition
from examples.stiffened_shell_native_fibre_bl5_displacement import ACTIONS,CONTROL,OPTIONS,TARGETS


def arc_control(owner,orientation=1):
    nodes=owner.preflight()['arc_metric_node_order']
    # This example chooses equal positive weights explicitly, not as a default
    # mass/quadrature rule or a mesh-independent scientific prescription.
    metric=ss.PhysicalChordMetric(np.full(len(nodes),1./len(nodes)),1.)
    return ss.ArcLengthControl('beam',105,0,metric,1e-6,orientation=orientation)


def workflow(shell='mixed',family='biaxial',beams='mixed',basis='p0',seed='translation'):
    if seed not in ('force','translation'):raise ValueError('ordinary force or fixed translation seed required')
    model=definition(shell,family,beams,basis)
    owner=ss.create_analysis(definition=model);fresh=ss.create_analysis(definition=model)
    try:
        owner.solve(ACTIONS,(.001,),**OPTIONS)
        if seed=='translation':
            owner.solve(ACTIONS,TARGETS[:2],displacement_control=CONTROL,**OPTIONS)
        control=arc_control(owner)
        prefix=owner.continue_arc_length(ACTIONS,control)
        raw=owner.checkpoint();restored=fresh.restore(raw)
        assert restored.status=='accepted_prefix' and fresh.checkpoint()==raw
        final=owner.continue_arc_length(ACTIONS,control)
        continued=fresh.continue_arc_length(ACTIONS,control)
        assert owner.checkpoint()==fresh.checkpoint()
        return dict(status=final.status,restored_status=restored.status,
            generation=final.generation,continued_generation=continued.generation,
            policy=model.bl5_material_policy,history=final.history,
            shell_s3_ids=[v.element_id for v in final.shell_s3_material],
            physical_fibre_ids=sorted({v.element_id for v in final.beam_fibre_material}),
            generalized_ids=[v.element_id for v in final.beam_generalized_material],
            seed=seed,prefix_generation=prefix.generation,
            free_equilibrium_ratio=final.free_equilibrium_ratio,constraint_max=final.constraint_max,
            fresh_restore_bytes_equal=True,continued_bytes_equal=True,
            production_qualified=False)
    finally:owner.close();fresh.close()


if __name__=='__main__':print(json.dumps(workflow(),indent=2))
