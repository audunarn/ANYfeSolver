import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2
from anyfesolver._ge_beam3_g3c_so3_numerics import so3_exp
from anyfesolver._ge_beam3_p5.arrays import _array


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
