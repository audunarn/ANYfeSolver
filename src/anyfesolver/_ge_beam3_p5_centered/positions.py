"""V2 nodal authority with affine split recovery and centered rest mass."""

from fractions import Fraction as F
import numpy as np

from anyfesolver._ge_beam3_p5_coordinates.positions import (
    POLICY, from_total, validate_total_pair, _rounded_pair,
)
from anyfesolver._ge_beam3_p5.arrays import _array, _frames, _readonly
from anyfesolver._ge_beam3_p5.compensated_coordinates import validate_pair
from .mass import current_rest_mass as rest_mass


def station_position(high, low, cell, weight, rotation, offset):
    """Affine interpolation retains partition of unity before final rounding.

    Do not treat separately rounded (1-t) and t as exact rational weights:
    their rational sum may differ from one, magnifying a common world offset.
    The physical identity is left+t*(right-left)+U*reference_lift.
    """
    high = _array(high,(3,3),'current coordinates'); low = _array(low,(3,3),'coordinate low parts')
    offset = _array(offset,(3,),'reference lift'); rotation = _frames([rotation],1,'lift rotation')[0]
    if type(cell) is not int or cell not in (0,1) or type(weight) is not float or not 0. <= weight <= 1.:
        raise ValueError('exact cell and finite station weight required')
    for a,b in zip(high.flat,low.flat): validate_pair(float(a),float(b))
    result = []
    for component in range(3):
        left = F(float(high[cell,component]))+F(float(low[cell,component]))
        right = F(float(high[cell+1,component]))+F(float(low[cell+1,component]))
        value = left+F(weight)*(right-left)
        value += sum((F(float(rotation[component,k]))*F(float(offset[k])) for k in range(3)),F())
        result.append(_rounded_pair(value))
    high_result, low_result = np.array(result).T
    return _readonly(high_result),_readonly(low_result)
