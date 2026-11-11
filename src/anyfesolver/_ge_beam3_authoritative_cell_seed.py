"""Initial cell rotations from authoritative nodal rotation operators.

Avoid reconstructing Q from (Q R0) R0.T: its roundoff stretch cannot be
removed by subsequent multiplicative SO(3) corrections. No operator,
quadrature, local convergence criterion or section law is changed here.
"""

import numpy as np
from ._ge_beam3_p5.arrays import _frames
from ._ge_beam3_p5.algebra import HALVES, rotation, log_rotation
from ._native_reference_modal import _owned


POLICY = 'AUTHORITATIVE_NODAL_OPERATOR_CELL_MIDPOINT_SEED_V1'


def cell_initial_rotations(operators):
    q = _frames(operators, 3, 'authoritative nodal rotation operators')
    result = []
    for left, right in HALVES:
        if np.array_equal(q[left], q[right]):
            result.append(q[left].copy())
        else:
            result.append(q[left]@rotation(log_rotation(q[left].T@q[right])/2))
    return _owned(result)
