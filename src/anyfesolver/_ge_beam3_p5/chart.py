"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp
from .algebra import skew
from .arrays import _array, _readonly

def _axial(matrix):
    return np.array([matrix[2,1]-matrix[1,2],matrix[0,2]-matrix[2,0],matrix[1,0]-matrix[0,1]])/2


def exp_chart_terms(vector):
    """Spatial left Exp Jacobian and its derivative, from analytic Jet2."""
    value=_array(vector,(3,),'native rotation increment')
    if np.linalg.norm(value)>=.9*np.pi: raise ValueError('native increment requires explicit cutback')
    jets=so3_exp([Jet2.variable(float(v),i,3) for i,v in enumerate(value)])
    r=np.array([[item.value for item in row] for row in jets])
    dr=np.array([[item.gradient for item in row] for row in jets])
    ddr=np.array([[item.hessian for item in row] for row in jets])
    a=np.column_stack([_axial(dr[:,:,j]@r.T) for j in range(3)])
    da=np.empty((3,3,3))
    for j in range(3):
        for k in range(3): da[:,j,k]=_axial(ddr[:,:,j,k]@r.T+dr[:,:,j]@dr[:,:,k].T)
    return a,da


def pullback(force,hessian,increments):
    force=_array(force,(18,),'spatial force');hessian=_array(hessian,(18,18),'spatial energy Hessian')
    increments=_array(increments,(3,3),'native increments')
    if np.linalg.norm(hessian-hessian.T)>1e-11*max(1.,np.linalg.norm(hessian)):
        raise ValueError('symmetric spatial energy Hessian required')
    chart=np.eye(18);spatial=hessian.copy();extra=np.zeros((18,18))
    for node,vector in enumerate(increments):
        start=6*node+3;a,da=exp_chart_terms(vector);moment=force[start:start+3]
        chart[start:start+3,start:start+3]=a
        spatial[start:start+3,start:start+3]-=.5*skew(moment)
        for k in range(3): extra[start:start+3,start+k]=da[:,:,k].T@moment
    transformed=chart.T@spatial@chart+extra
    if np.linalg.norm(transformed-transformed.T)>1e-11*max(1.,np.linalg.norm(transformed)):
        raise ValueError('native chart Hessian symmetry failed')
    return _readonly(chart.T@force),_readonly(transformed),_readonly(chart)
