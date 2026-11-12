"""Shared effective wall-clock budget for the GE beam3 cooperative deadlines.

The cooperative deadlines are finite-work controls: they bound the wall-clock
budget of a construction, solve or validation route and keep cancellation and
rollback explicit. They are not scientific checks and never enter a
descriptor, checkpoint identity or recorded byte sequence.

This service separates the test/CI execution budget from physics/settings
identity: the effective threshold is `base * scale`, where the scale is read
from the environment at every check so a caller can declare a bounded
execution budget (for example on a slow shared CI runner) without touching any
recorded contract. The default scale is exactly 1.0, which reproduces the
historical thresholds bit-for-bit. Out-of-range or unparseable values fail
closed instead of silently reverting.
"""
from math import isfinite
from os import environ

POLICY = 'GE_BEAM3_EFFECTIVE_DEADLINE_V1'
SCALE_VARIABLE = 'ANYSOLVER_GE_BEAM3_DEADLINE_SCALE'
MIN_SCALE = 1.0
MAX_SCALE = 8.0


def deadline_scale():
    """Return the explicitly declared, bounded wall-clock budget scale."""
    raw = environ.get(SCALE_VARIABLE)
    if raw is None:
        return 1.0
    try:
        scale = float(raw)
    except ValueError:
        raise ValueError(SCALE_VARIABLE + ' must be a real number, received ' + repr(raw)) from None
    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise ValueError(SCALE_VARIABLE + ' must be within [%g, %g], received %r'
            % (MIN_SCALE, MAX_SCALE, scale))
    return scale


def effective_deadline_seconds(base):
    """Return the effective cooperative wall-clock threshold for `base`.

    The guard is a finite-work control: a nonfinite base or an overflowed
    effective threshold is refused instead of silently disabling the guard.
    """
    if type(base) not in (int, float) or not base > 0.:
        raise ValueError('cooperative deadline base must be a positive number')
    scale = deadline_scale()
    try:
        effective = base * scale
        finite = isfinite(base) and isfinite(effective)
    except OverflowError:
        effective = None
        finite = False
    if not finite:
        raise ValueError('cooperative deadline base and effective deadline must be finite, received %r'
            % (base,))
    return effective
