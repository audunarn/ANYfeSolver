"""Actual compensated updater/evaluator, original immutable saved F06 inputs."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'.development/r8-f06-quiet-caller-repair-20261009/source2/s3'

def definition():
    from scripts.r8_f06_quiet_caller_repair_dev import definition as original
    from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
    from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import POLICY
    base=original();beams=[]
    for item in base.beams:
        e,inertia=item.instantiate();op=e.operator
        made=NativeGeneralizedStaticElement(e.element_id,e.node_ids,op.reference,op.section,
            order=op.order,arithmetic_policy=POLICY)
        beams.append(NativeBeamDefinition.capture(made,inertia))
    return replace(base,beams=tuple(beams))

def owner():
    from anyfesolver import stiffened_shell as ss
    made=ss.create_analysis(definition=definition())
    made.restore((SAVED/'accepted-before.bin').read_bytes(),legacy_compensated_base=True)
    return made

def actual(out):
    from anyfesolver._stiffened_shell_ge_finite_experimental import _State
    out=Path(out);o=owner();d=o._finite_driver;results=[]
    try:
        origins=(d._shell_origins,d._beam_origins)
        for side in ('before','after'):
            if side=='after':o.rebase_rotations(o._generation)
            data=dict(np.load(SAVED/(side+'-base.npz')))
            fd=dict(np.load(SAVED/(side+'-fd-0.002.npz')))
            state=d._zero_compensation(_State(data['q'].copy(),data['frames'].copy(),data['cells'].copy(),
                data['resultants'].copy(),data['multipliers'].copy(),d._state.shell_anchors))
            direction=data['direction'];h=.002;rows=np.arange(66,114)
            baseline=d._assemble(state,.06,*origins)
            plus=d._advance(state,direction,h);minus=d._advance(state,direction,-h)
            pe=d._assemble(plus,.06,*origins);me=d._assemble(minus,.06,*origins)
            predicted=baseline[1]@direction;observed=(pe[0]-me[0])/(2*h)
            scale=data['rowscale'][rows];den=max(float(np.linalg.norm(data['predicted'][rows]*scale)),1e-10)
            metric=dict(side=side,h=h,rows=rows.tolist(),denominator=den,
                raw_actual_error=float(np.linalg.norm((observed-predicted)[rows]*scale)/den),
                original_raw_error=float(np.linalg.norm((fd['observed']-data['predicted'])[rows]*scale)/den),
                tangent_vs_saved=float(np.linalg.norm((predicted-data['predicted'])[rows]*scale)/den),
                pair_nonzero=dict(q=int(np.count_nonzero(plus.q_low)),Q=int(np.count_nonzero(plus.frames_low)),
                    U=int(np.count_nonzero(plus.cells_low)),p=int(np.count_nonzero(plus.resultants_low))))
            np.savez(out/(side+'-actual-full48.npz'),predicted=predicted,observed=observed,rows=rows,
                q_low=plus.q_low,frames_low=plus.frames_low,cells_low=plus.cells_low,resultants_low=plus.resultants_low)
            (out/(side+'-metrics.json')).write_text(json.dumps(metric,indent=2)+'\n');results.append(metric)
            assert all(v>0 for v in metric['pair_nonzero'].values()),metric
            assert metric['raw_actual_error']<=1e-6,metric
            assert metric['tangent_vs_saved']<=1e-6,metric
        return results
    finally:o.close()
