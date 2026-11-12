"""Shared bounded retained-coordinate Newton reduction and correction measure."""
import numpy as np


def retained_newton_step(context,state,residual,tangent,*,family='generalized'):
    step=np.zeros(context.count)
    step[context.free]=np.linalg.solve(tangent[np.ix_(context.free,context.free)],-residual[context.free])
    if not np.isfinite(step).all():raise ValueError(f'nonfinite retained {family} Newton step')
    nodal=step[:context.nodal_count].reshape(-1,6)
    correction=max(float(np.linalg.norm(nodal[:,:3]))/context.length,float(np.linalg.norm(nodal[:,3:])))
    for i in range(len(context.elements)):
        first=context.nodal_count+24*i
        correction=max(correction,float(np.linalg.norm(step[first:first+6])),
            float(np.linalg.norm(step[first+6:first+24]))/max(1.,float(np.linalg.norm(state.resultants[i]))))
    return step,correction
