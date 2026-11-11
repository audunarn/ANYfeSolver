"""Experimental native S3/fibre tangent-plane restart through the public API.

This small response of a plastic-capable model does not demonstrate yielding,
a limit point, a descending branch or qualified engineering capacity.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json
import numpy as np
from anyfesolver import stiffened_shell as ss
if __package__:
    from .stiffened_shell_coupled_s3_adaptive_history import definition
else:
    from stiffened_shell_coupled_s3_adaptive_history import definition


def workflow(output):
    output=Path(output);output.mkdir(exist_ok=False)
    model=definition()
    analysis=ss.create_analysis(definition=model)
    resumed=ss.create_analysis(definition=model)
    try:
        zero=tuple(ss.NodalAction('beam',node,(0.,0.,0.,0.,0.,0.)) for node in (105,205))
        analysis.solve(zero,(1.,),max_iterations=20,max_backtracks=12)
        order=analysis.preflight()['arc_metric_node_order']
        metric=ss.PhysicalChordMetric(np.full(len(order),1./len(order)),1.)
        control=ss.ArcLengthControl('beam',105,2,metric,1e-6)
        actions=tuple(ss.NodalAction('beam',node,(0.,0.,100.,0.,0.,0.)) for node in (105,205))
        first=analysis.continue_arc_length(actions,control)
        saved=analysis.checkpoint();(output/'prefix.checkpoint').write_bytes(saved)
        resumed.restore(saved);assert resumed.checkpoint()==saved
        final=analysis.continue_arc_length(actions,control)
        repeated=resumed.continue_arc_length(actions,control)
        assert analysis.checkpoint()==resumed.checkpoint()
        assert final.history==repeated.history and final.status=='completed'
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        (output/'completed.checkpoint').write_bytes(analysis.checkpoint())
        result=dict(status=final.status,production_qualified=False,
            capability=analysis.capabilities['arc_length'],
            model_identity=analysis.preflight()['identity'],
            physical_metric_node_order=order,first_amplitude=first.amplitude,
            history=final.history,exact_restart=True,
            equilibrium=final.free_equilibrium_ratio,gap=final.constraint_max,
            checkpoint_sha256=sha256(analysis.checkpoint()).hexdigest(),
            limitation='Small continuation/restart only; no limit-point or descending-branch evidence.')
        (output/'result.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('ascii'))
        return result
    finally:analysis.close();resumed.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    print(json.dumps(workflow(parser.parse_args().output)))
