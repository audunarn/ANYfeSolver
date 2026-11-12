"""Private load-only variations extracted from the bound load-work probe.

No material-potential subtraction; no source/research imports at runtime.
"""
from dataclasses import dataclass
import numpy as np
from anyfesolver._ge_beam3_p5.arrays import _array, _readonly, _frames
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair
from anyfesolver._ge_beam3_p5.compensated import sum_jets
from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp, matmul, constant_matrix

@dataclass(frozen=True)
class AugmentedWork:
    work: float
    force: np.ndarray
    hessian: np.ndarray


def load_work(reference, positions, position_low, cells, force, *, order=8):
    """Load-only analytic AD: never subtract two material potentials.

    Use the frozen centered reference evaluator and its load expression,
    independently of the rational-Q2/closed-form derivative oracle.
    """
    x = _array(positions,(3,3),'positions'); low = _array(position_low,(3,3),'position low')
    cells = _frames(cells,2,'cell rotations'); force = _array(force,(3,),'line force')
    if type(order) is not int or order not in (4,8,24): raise ValueError('registered load quadrature required')
    for a,b in zip(x.flat,low.flat): validate_pair(float(a),float(b))
    delta = [Jet2.variable(0.,i,36) for i in range(36)]
    points, weights = np.polynomial.legendre.leggauss(order); terms = []
    for cell in (0,1):
        made_u = matmul(so3_exp(delta[18+3*cell:21+3*cell]),constant_matrix(cells[cell],36))
        for point,weight in zip(points,weights):
            t = float((point+1)/2); xi = cell-1+t
            offset = reference.half_cell_lift(cell,t); measure = weight*reference.jacobian(xi)/2
            for i in range(3):
                position_terms = [(made_u[i][k]-(1 if i == k else 0))*offset[k] for k in range(3)]
                for node,shape in ((cell,1-t),(cell+1,t)):
                    high,tail = split_sum((float(x[node,i]),float(low[node,i]),-float(reference.coordinates[node,i])))
                    position_terms.extend((Jet2.constant(shape*high,36),Jet2.constant(shape*tail,36),shape*delta[6*node+i]))
                terms.append(measure*force[i]*sum_jets(position_terms))
    made = sum_jets(terms)
    if not np.isfinite(made.value) or not np.isfinite(made.gradient).all() or not np.isfinite(made.hessian).all():
        raise ValueError('nonfinite load-only variations')
    return AugmentedWork(float(made.value),_readonly(made.gradient),_readonly(made.hessian))
