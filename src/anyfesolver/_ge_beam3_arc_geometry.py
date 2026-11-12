"""Objective frame-chord hyperplane in native spatial Exp coordinates.

Private continuation geometry only; no mechanics, state commit or public route.
The caller supplies actual step increments, not differences of rounded world
positions or accumulated rotation vectors interpreted as physical orientation.
"""

import math

import numpy as np

from anyfesolver._ge_beam3_p5.arrays import _array, _readonly


POLICY = 'GE_BEAM3_SPATIAL_EXP_FRAME_CHORD_HYPERPLANE_V1'


def frame_chord_terms(increment,direction):
    """Value and exact first-variation formula for .5 <Q-Q0, hat(w) Q0>.

    Q=Exp(hat(v))Q0 gives w dot (sinc(|v|) v), independently of Q0.
    Its native-coordinate gradient is sinc(t) w + beta(t)(w dot v)v,
    beta=(t cos(t)-sin(t))/t**3. Evaluating from v avoids subtracting
    nearly equal dense frames. This is not w dot v at finite rotation.
    """
    v = _array(increment,(3,),'spatial frame increment')
    w = _array(direction,(3,),'spatial predictor direction')
    theta = math.hypot(*v)
    if theta >= .9*math.pi: raise ValueError('frame increment requires explicit cutback')
    square = theta*theta
    if theta < 1e-3:
        # At this switch the omitted alpha term is <= 1e-24/9!, and
        # the omitted beta term is <= 1e-24/498960. No equality tolerance
        # or constitutive coefficient is changed by these stable evaluations.
        alpha = 1.+square*(-1./6.+square*(1./120.-square/5040.))
        beta = -1./3.+square*(1./30.+square*(-1./840.+square/45360.))
    else:
        alpha = math.sin(theta)/theta
        beta = (theta*math.cos(theta)-math.sin(theta))/(theta**3)
    dot = math.fsum(float(a)*float(b) for a,b in zip(w,v))
    value = alpha*dot
    gradient = alpha*w+beta*dot*v
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError('nonfinite frame-chord value/gradient')
    return value,_readonly(gradient)


def constraint(increments,direction,metric,parameter_increment,parameter_direction,parameter_weight,step):
    """Frame-chord pseudo-arclength hyperplane and native-coordinate row.

    Arrays have shape (nodes,6) and contain full spatial translation/rotation
    blocks. Fixed coordinates have zero increments/predictors; their isotropic
    weights may remain positive. Each translation and rotation metric block
    must be isotropic, including optional all-zero fixed blocks. An anisotropic
    diagonal metric cannot remain unchanged under a common rigid frame change.
    The returned row includes the final load-parameter entry.
    """
    raw = np.asarray(increments)
    if raw.ndim != 2 or raw.shape[1] != 6 or not 1 <= raw.shape[0] <= 42:
        raise ValueError('bounded node-by-six increment array required')
    v = _array(increments,raw.shape,'native continuation increments')
    d = _array(direction,raw.shape,'spatial continuation direction')
    weights = _array(metric,raw.shape,'continuation metric')
    if np.any(weights < 0.) or np.any(d[weights == 0.] != 0.):
        raise ValueError('nonnegative metric and zero constrained direction required')
    for value in (parameter_increment,parameter_direction,parameter_weight,step):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError('finite explicit continuation scalars required')
    if parameter_weight <= 0. or not 0. < step <= .25:
        raise ValueError('positive parameter weight and bounded positive step required')
    if any(not np.all(block == block[0]) for block in weights.reshape(-1,3)):
        raise ValueError('isotropic translation and rotation metric blocks required')
    norm_squared = float(np.sum(weights*d*d)+parameter_weight*parameter_direction**2)
    if not math.isfinite(norm_squared) or norm_squared <= 0.:
        raise ValueError('finite nonzero metric predictor required')
    gradient = np.zeros_like(v)
    gradient[:,:3] = weights[:,:3]*d[:,:3]
    terms = [float(a)*float(b) for a,b in zip(gradient[:,:3].ravel(),v[:,:3].ravel())]
    for node in range(len(v)):
        value,row = frame_chord_terms(v[node,3:],d[node,3:])
        terms.append(float(weights[node,3])*value)
        gradient[node,3:] = weights[node,3]*row
    parameter_row = parameter_weight*parameter_direction
    terms.extend((parameter_row*parameter_increment,-step))
    value = math.fsum(terms)
    row = np.r_[gradient.ravel(),parameter_row]
    if not math.isfinite(value) or not np.isfinite(row).all():
        raise ValueError('nonfinite continuation hyperplane')
    return value,_readonly(row)
