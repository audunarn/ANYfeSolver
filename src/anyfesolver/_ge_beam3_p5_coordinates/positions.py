"""Bound total-displacement coordinate pairs and cold-path physical recovery.

Nodal addition reuses the frozen two-component summation implementation. The
recovery sum uses exact products of supplied binary64 data before rounding to
two components; this does not make the reference geometry arbitrary precision.
"""

from fractions import Fraction
import math

import numpy as np

from anyfesolver._ge_beam3_p5.arrays import _array, _frames, _readonly
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair
from anyfesolver._ge_beam3_p5.inertia import LiftedInertia


POLICY = 'GE_BEAM3_P5_REFERENCE_PLUS_TOTAL_TWO_COMPONENT_V1'


def from_total(reference, total, high):
    """Own the exact high pose and retain the residual of binary64 X + u.

    High coordinates must agree exactly with the native view. A displacement
    cannot be recovered by subtracting two previously rounded configurations.
    """
    reference = _array(reference, (3, 3), 'reference coordinates')
    total = _array(total, (18,), 'authoritative total displacement')
    high = _array(high, (3, 3), 'native high pose')
    displacement = total.reshape(3, 6)[:, :3]
    low = np.empty((3, 3))
    for index in np.ndindex((3, 3)):
        expected, low[index] = split_sum((float(reference[index]), float(displacement[index])))
        if float(high[index]) != expected:
            raise ValueError('native high pose disagrees with reference plus total displacement')
    return _readonly(high), _readonly(low)


def validate_total_pair(reference, total, high, low, policy):
    if type(policy) is not str or policy != POLICY:
        raise ValueError('coordinate policy mismatch')
    high, expected = from_total(reference, total, high)
    low = _array(low, (3, 3), 'committed coordinate low parts')
    if not np.array_equal(low, expected):
        raise ValueError('coordinate low parts disagree with authoritative displacement')
    return high, _readonly(low)


def _rounded_pair(value):
    try:
        high = float(value)
        low = float(value-Fraction(high))
    except (OverflowError, ValueError) as error:
        raise ValueError('physical position expansion overflow') from error
    if not math.isfinite(high) or not math.isfinite(low):
        raise ValueError('nonfinite physical position expansion')
    return split_sum((high, low))


def station_position(high, low, cell, weight, rotation, offset):
    """Two rounded parts of I(high+low)+U*offset; no dropped nodal lows.

    All inputs are existing discrete-field values. Cold-path exact rational
    accumulation avoids a rounded product hiding the small displacement before
    the final two-component representation. The last component may itself round.
    """
    high = _array(high, (3, 3), 'current coordinates')
    low = _array(low, (3, 3), 'coordinate low parts')
    offset = _array(offset, (3,), 'reference lift')
    rotation = _frames([rotation], 1, 'physical lift rotation')[0]
    if type(cell) is not int or cell not in (0, 1) or type(weight) is not float or not 0. <= weight <= 1.:
        raise ValueError('exact cell and finite station weight required')
    for a, b in zip(high.flat, low.flat):
        validate_pair(float(a), float(b))
    first = Fraction(float(1.-weight)); second = Fraction(weight)
    result_high = np.empty(3); result_low = np.empty(3)
    for component in range(3):
        value = first*(Fraction(float(high[cell, component]))+Fraction(float(low[cell, component])))
        value += second*(Fraction(float(high[cell+1, component]))+Fraction(float(low[cell+1, component])))
        value += sum((Fraction(float(rotation[component, k]))*Fraction(float(offset[k])) for k in range(3)), Fraction())
        result_high[component], result_low[component] = _rounded_pair(value)
    return _readonly(result_high), _readonly(result_low)


def rest_mass(reference, section_inertia, order, high, low, rotations):
    """Retained-cell mass at rest, not finite-velocity momentum or dynamics.

    In the frozen kinetic field B and Q depend on the reference offset and cell
    rotations, not on absolute nodal positions. Positions occur only in the
    angular-momentum diagnostic; with v=a=0 that diagnostic is identically zero.
    Both coordinate parts are nevertheless validated and bound by the caller's
    accepted state. No low part is discarded from a position-dependent mass term.
    """
    high = _array(high, (3, 3), 'rest mass high coordinates')
    low = _array(low, (3, 3), 'rest mass low coordinates')
    for a, b in zip(high.flat, low.flat):
        validate_pair(float(a), float(b))
    response = LiftedInertia(reference, section_inertia, order=order).evaluate(
        high, rotations, np.zeros(24), np.zeros(24))
    if response.kinetic_energy != 0. or np.any(response.generalized_momentum) or np.any(response.angular_momentum):
        raise ValueError('rest mass path produced nonzero kinetic state')
    return response.mass
