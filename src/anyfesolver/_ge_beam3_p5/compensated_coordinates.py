"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

import math


SCHEMA = 'GE_BEAM3_P5_COMPENSATED_COORDINATES_V1'


def split_sum(values):
    """Two rounded components of a short finite sum, not arbitrary precision.

    fsum computes the rounded high part and then the rounded residual of the
    original terms. The second component may itself round; no universal exact
    summation claim is made. Overflow and nonfinite inputs fail closed.
    """
    values = tuple(values)
    if not values or any(type(v) is not float or not math.isfinite(v) for v in values):
        raise ValueError('nonempty finite binary64 coordinate terms required')
    try:
        high = math.fsum(values)
        low = math.fsum((*values, -high))
    except OverflowError as error:
        raise ValueError('coordinate expansion overflow') from error
    if not math.isfinite(high) or not math.isfinite(low):
        raise ValueError('nonfinite coordinate expansion')
    return (0. if high == 0 else high, 0. if low == 0 else low)


def validate_pair(high, low):
    if (type(high) is not float or type(low) is not float or
            not math.isfinite(high) or not math.isfinite(low)):
        raise ValueError('finite binary64 coordinate pair required')
    if split_sum((high, low)) != (high, low):
        raise ValueError('normalized coordinate pair required')
    return high, low


def advance(high, low, increment):
    validate_pair(high, low)
    return split_sum((high, low, increment))
