"""Private frame-chord geometry differentiated in CURRENT spatial increments.

Unlike the older accepted-origin Exp-chart row, delta A = hat(delta theta) A.
No physical operator, constitutive coefficient or historical policy is changed.
"""
from fractions import Fraction
from math import fsum, isfinite
import numpy as np
from ._ge_beam3_p5.algebra import skew, log_rotation
from ._ge_beam3_p5.arrays import _array, _frames
from ._native_reference_modal import _owned

POLICY = 'GE_BEAM3_RETAINED_CURRENT_SPATIAL_FRAME_CHORD_V1'


def frame_terms(current, origin, predictor):
    q = _frames(np.asarray([current, origin]), 2, 'arc frames')
    w = _array(predictor, (3,), 'spatial arc predictor')
    a = q[0] @ q[1].T
    log_rotation(a)  # Domain/proper-frame validation, not numerical differentiation.
    axial = np.array((a[2, 1]-a[1, 2], a[0, 2]-a[2, 0], a[1, 0]-a[0, 1]))*.5
    value = fsum(float(x)*float(y) for x, y in zip(w, axial))
    b = skew(w) @ a.T
    gradient = np.array((b[2, 1]-b[1, 2], b[0, 2]-b[2, 0], b[1, 0]-b[0, 1]))*.5
    if not isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError('nonfinite current-spatial arc geometry')
    return value, _owned(gradient)


def constraint(current, origin, predictor, metric, parameter, old_parameter, step):
    """Physical nodal chord; internal retained variables have exactly zero weight."""
    n = len(current.positions)
    d = _array(predictor, np.asarray(metric).shape, 'complete arc predictor')
    weights = _array(metric, d.shape, 'complete arc metric')
    if d.ndim != 1 or len(d) < 6*n+1 or np.any(weights < 0.) or weights[-1] <= 0.:
        raise ValueError('complete nonnegative physical metric required')
    if np.any(weights[6*n:-1] != 0.):
        raise ValueError('internal coordinates are not physical arclength DOFs')
    blocks = weights[:6*n].reshape(-1, 3)
    if np.any(blocks <= 0.) or any(not np.all(b == b[0]) for b in blocks):
        raise ValueError('positive isotropic physical arc metric required')
    for scalar in (parameter, old_parameter, step):
        if type(scalar) is not float or not isfinite(scalar):
            raise ValueError('finite explicit arc scalars required')
    if not 0. <= step <= .25:
        raise ValueError('bounded arc step required')
    row = weights*d
    # One rounding after exact dyadic products, including both low parts.
    translation = Fraction(0)
    for node in range(n):
        for axis in range(3):
            delta = (Fraction(float(current.positions[node, axis]))
                     + Fraction(float(current.position_low[node, axis]))
                     - Fraction(float(origin.positions[node, axis]))
                     - Fraction(float(origin.position_low[node, axis])))
            translation += Fraction(float(row[6*node+axis]))*delta
    terms = [float(translation), row[-1]*(parameter-old_parameter), -step]
    for node in range(n):
        slots = slice(6*node+3, 6*node+6)
        value, gradient = frame_terms(current.nodal_frames[node], origin.nodal_frames[node], d[slots])
        terms.append(weights[6*node+3]*value)
        row[slots] = weights[6*node+3]*gradient
    gap = fsum(terms)
    if not isfinite(gap) or not np.isfinite(row).all():
        raise ValueError('nonfinite physical arc constraint')
    return gap, _owned(row)
