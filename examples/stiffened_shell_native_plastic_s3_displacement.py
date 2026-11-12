"""Experimental yielded all-S3 point panel: fixed targets and exact restart.

This low-yield shell is a lifecycle control, not a qualified capacity model.
This example covers all-S3; simultaneous fibre flow and physical qualification
remain open. The failed mixed fixed histories remain separate evidence.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import numpy as np
from anymaterial import DNVC208MaterialCurve
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from stiffened_shell_ge_point_zero_unload import definition as reference_definition


def definition():
    base=reference_definition()
    pressure=20_000.
    curve=DNVC208MaterialCurve(pressure,1.05*pressure,1.1*pressure,.005,.01,4.*pressure,.2)
    shell=replace(base.shell,q4_elements=(),s3_elements=(
        (1,2,5),(1,5,4),(2,3,6),(2,6,5)),j2_curve=curve)
    return ss.LinearDefinition(base.profile,shell,base.beams,base.points,base.fixed)


def workflow(output,*,adaptive=False):
    output=Path(output);output.mkdir(exist_ok=False)
    model=definition();analysis=ss.create_analysis(definition=model)
    resumed=ss.create_analysis(definition=model)
    try:
        options=dict(max_iterations=20,max_backtracks=12)
        actions=(ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
        direction=(ss.NodalAction('beam',105,(0.,0.,100.,0.,0.,0.)),)
        seed=analysis.solve(actions,(.05,),**options)
        initial=float(seed.beam_positions[-1,2]-model.beams[-1].instantiate()[0].operator.reference.coordinates[-1,2])
        options['displacement_control']=ss.DisplacementControl('beam',105,2)
        targets=(initial*1.01,initial*1.02)
        if adaptive:
            options['increment_control']=ss.AdaptiveIncrementControl(2,8)
            token=CancellationToken()
            try:
                analysis.solve(direction,targets,**options,cancellation_token=token,
                    on_accepted=lambda result:token.cancel('accepted adaptive prefix'))
            except SolveCancelled:prefix=analysis.recover()
            else:raise AssertionError('adaptive cancellation did not preserve prefix')
        else:prefix=analysis.solve(direction,targets[:1],**options)
        raw=analysis.checkpoint();(output/'prefix.checkpoint').write_bytes(raw)
        restored=resumed.restore(raw)
        assert resumed.checkpoint()==raw and restored.shell_layer_alpha==prefix.shell_layer_alpha
        remaining=targets if adaptive else targets[1:]
        final=analysis.solve(direction,remaining,**options)
        continued=resumed.solve(direction,remaining,**options)
        assert analysis.checkpoint()==resumed.checkpoint()
        assert final.shell_layer_alpha==continued.shell_layer_alpha
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        alpha=np.concatenate([row.alpha for row in final.shell_s3_material])
        previous=np.concatenate([row.alpha for row in prefix.shell_s3_material])
        assert np.any(alpha>previous)
        result=dict(status=final.status,production_qualified=False,mesh='all-S3',
            adaptive=adaptive,exact_restart=True,shell_last_increment_flow=True,max_shell_alpha=float(alpha.max()),
            equilibrium=final.free_equilibrium_ratio,gap=final.constraint_max,history=final.history)
        (output/'completed.checkpoint').write_bytes(analysis.checkpoint())
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
    finally:analysis.close();resumed.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--adaptive',action='store_true',help='use bounded native cutback and durable controller restart')
    arguments=parser.parse_args();print(json.dumps(workflow(arguments.output,adaptive=arguments.adaptive)))
