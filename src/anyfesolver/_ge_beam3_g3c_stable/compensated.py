import math
import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2
from anyfesolver._ge_beam3_mixed_ad import transpose
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum


def sum_jets(terms):
    """Analytic value/first/second variations of a sum, stable scalar value."""
    terms = tuple(terms)
    if not terms: raise ValueError('nonempty jet sum required')
    return Jet2(math.fsum(t.value for t in terms),
                sum((t.gradient for t in terms), np.zeros_like(terms[0].gradient)),
                sum((t.hessian for t in terms), np.zeros_like(terms[0].hessian)))


def compensated_strain(made_u, reference, position_high, position_low, variables, left, right):
    """All derivative terms use the two-part chord before scalar collapse.

    (U^T-I)d0 + U^T(d-d0), retaining both parts of d0 and d-d0. Coordinate
    differences are linear maps: their high jet carries the exact unit nodal
    derivatives, while their low jet is a constant at the evaluation point.
    Products with U retain the low part in rotational first/second variations.
    """
    size = variables[0].gradient.size
    base_parts, change_parts = [], []
    for i in range(3):
        baseline = (float(reference[right, i]), -float(reference[left, i]))
        base_parts.append(split_sum(baseline))
        terms = (float(position_high[right, i]), float(position_low[right, i]),
                 -float(position_high[left, i]), -float(position_low[left, i]),
                 variables[6*right+i].value, -variables[6*left+i].value,
                 -baseline[0], -baseline[1])
        high, low = split_sum(terms)
        derivative = variables[6*right+i]-variables[6*left+i]
        change_parts.append((Jet2(high, derivative.gradient, derivative.hessian), Jet2.constant(low, size)))
    u_t = transpose(made_u)
    result = []
    for i in range(3):
        terms = []
        for k in range(3):
            difference = u_t[i][k]-(1 if i == k else 0)
            terms.extend(difference*v for v in base_parts[k])
            terms.extend(u_t[i][k]*v for v in change_parts[k])
        result.append(sum_jets(terms))
    return result
