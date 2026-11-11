"""Precise scalar angular coordinates for the existing material normal work.

The gap is an affine axial field plus a circular sinusoid. Decimal arithmetic
keeps its value and angular partition consistent when the axial slope is tiny.
Jet2 still supplies all work derivatives; this is no new differentiation engine.
Angles are confined to the surface rule's [0, 2*pi] range. All loops are bounded.
"""
from decimal import (Decimal, Inexact, Rounded, getcontext, localcontext,
                     ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP)
from math import acos, atan2, pi


PRECISION = 50


# Prepared Taylor step divisors: the same exact integers, converted once.
_SINE_STEP_DIVISORS = tuple(Decimal(2*k*(2*k+1)) for k in range(1, 81))
_COSINE_STEP_DIVISORS = tuple(Decimal((2*k-1)*2*k) for k in range(1, 81))

# Certified early completion of the pure Taylor loop. The loop may stop after
# a pair only when the remaining pairs provably cannot change either rounded
# sum or raise any context flag the completed pairs have not already raised:
# a half rounding mode (a directed mode can move its sums by any nonzero tail
# term), both rounding flags already raised, tail step factors below a tenth
# of the next pair's divisors (both term sequences then shrink strictly per
# remaining pair; a low-precision nearest rounding can still round a step
# factor or product to a small fraction of the previous term, so no larger
# shrink is claimed), terms already negligible against their sums, and every
# remaining division and product provably normal in the caller's exponent
# range. Any unmet condition keeps the full 80 pairs.
_HALF_ROUNDINGS = frozenset((ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_HALF_DOWN))
# Exact by construction (integer digits with exponent -1): importing this
# module raises no flag and fires no trap under any Decimal context, however
# small its precision or strict its traps.
_TAIL_SHRINK_BOUNDS = tuple(
    Decimal(f"{(2*k+1)*(2*k+2)}e-1") for k in range(1, 81))


def _certification_plan(context, magnitude):
    """Per-call constants of the certification test; None never certifies."""
    if (not magnitude.is_finite() or not magnitude
            or context.rounding not in _HALF_ROUNDINGS
            or magnitude >= _TAIL_SHRINK_BOUNDS[-1]):
        return None
    # Integer shrink bound, proof-safe at every positive Decimal precision:
    # with a = magnitude.adjusted() the magnitude is at least 10^a, and each
    # remaining pair's rounded division and multiplication are normal nearest
    # roundings that each lose at most a factor 2/3, so the pair keeps at
    # least (4/9)*10^a/25760 > 10^(a-5) of its term (25760 is the largest
    # remaining divisor). No float, logarithm or ceiling can overstate this
    # at small precisions or near the binary64 subnormal range, where a
    # float conversion of the magnitude can err by nearly a factor of two.
    return (Decimal((0, (1,), context.Emin+7)),
            Decimal((0, (1,), -context.prec-2)),
            5 - magnitude.adjusted())


def _tail_certified(context, k, magnitude, plan, sine_term, cosine_term,
                    sine, cosine):
    """True only when skipping pairs k+1..80 provably changes nothing."""
    magnitude_floor, threshold, floor_step = plan
    if magnitude >= _TAIL_SHRINK_BOUNDS[k-1] or magnitude < magnitude_floor:
        return False
    flags = context.flags
    if not (flags[Inexact] and flags[Rounded]):
        return False
    if not (sine_term.is_finite() and cosine_term.is_finite()
            and sine.is_finite() and cosine.is_finite()):
        return False
    # Negligibility is necessary before the exact comparisons below.
    if (sine_term.adjusted() > sine.adjusted() - context.prec - 2
            or cosine_term.adjusted() > cosine.adjusted() - context.prec - 2):
        return False
    # Remaining terms stay normal by direct induction on the integer bound
    # floor_step = 5 - a: a term at or above 10^(Emin+3+(80-k)*floor_step)
    # bounds every later term at 10^(Emin+3) or more, above the caller's
    # smallest normal, so the remaining divisions and products raise no
    # flag beyond the already raised Inexact/Rounded and fire no trap the
    # completed pairs have not. The floor comparisons precede the threshold
    # products: a negligible term at or above the floor pins each sum at
    # 10^(Emin+prec+5) or more (the negligibility margin is a factor 100
    # beyond the precision), so every threshold product stays normal.
    # Magnitudes use copy_abs: unlike unary abs it applies no context, so
    # even a subnormal operand can signal no flag or fire no trap.
    floor = Decimal((0, (1,), context.Emin + 3 + (80-k)*floor_step))
    if sine_term.copy_abs() < floor or cosine_term.copy_abs() < floor:
        return False
    return (sine_term.copy_abs() <= sine.copy_abs()*threshold
            and cosine_term.copy_abs() <= cosine.copy_abs()*threshold)


# Bounded exact reuse of the pure angular kernel. The Taylor values depend
# only on the angle representation and the effective Decimal context, never
# on geometry or material history. The key carries every effective context
# setting, so any context change misses instead of serving another setting.
# Flags stay out of the key (they never change results), but each entry
# also carries the complete flag set its own loop raises, measured with
# the flags cleared so preexisting raised flags cannot mask any of it;
# a hit re-raises exactly that set on top of the caller's current flags,
# so reuse leaves behind exactly the flags the original loop would
# leave, whatever flags were raised before the call. A full cache is
# cleared whole, keeping reuse bounded and deterministic.
_SINCOS_CACHE_LIMIT = 8192
_sincos_cache = {}


def _context_signature():
    context = getcontext()
    return (context.prec, context.rounding, context.Emin, context.Emax,
            context.clamp, context.capitals,
            frozenset(context.traps.items()))


def _sincos(angle):
    """Taylor evaluation on the admitted bounded angle range, at 50 digits.

    At |angle| <= 2*pi, the omitted term after 80 pairs is below 1e-150.
    Decimal roundoff dominates that remainder; finite precision is still
    checked by the root and interval admission, rather than claiming exactness.
    Identical angle representations under an identical effective context are
    reused exactly: a hit returns the very tuple a fresh loop would produce,
    and leaves behind the very context flags such a loop would raise. The
    loop itself completes early only when the remaining pairs provably
    cannot change either rounded sum or raise any context flag the completed
    pairs have not already raised; every other evaluation runs all 80 pairs.
    """
    x = Decimal(angle)
    context = getcontext()
    key = (x.as_tuple(), _context_signature())
    cached = _sincos_cache.get(key)
    if cached is not None:
        result, raised = cached
        for flag in raised:
            context.flags[flag] = True
        return result
    initial = frozenset(flag for flag, value in context.flags.items() if value)
    context.clear_flags()
    try:
        sine = sine_term = x
        cosine = cosine_term = Decimal(1)
        square = -x*x
        asq = abs(square)
        plan = _certification_plan(context, asq)
        bounds = _TAIL_SHRINK_BOUNDS
        for k in range(1, 81):
            sine_term *= square/_SINE_STEP_DIVISORS[k-1]
            cosine_term *= square/_COSINE_STEP_DIVISORS[k-1]
            sine += sine_term
            cosine += cosine_term
            if (plan is not None and k >= 8 and not k & 3
                    and asq < bounds[k-1] and _tail_certified(
                        context, k, asq, plan, sine_term, cosine_term,
                        sine, cosine)):
                break
        raised = frozenset(flag for flag, value in context.flags.items()
                           if value)
    finally:
        for flag in initial:
            context.flags[flag] = True
    if len(_sincos_cache) >= _SINCOS_CACHE_LIMIT:
        _sincos_cache.clear()
    result = (sine, cosine)
    _sincos_cache[key] = (result, raised)
    return result


class PreciseNormalAngles:
    def __init__(self, offset, slope, width, cosine, sine):
        self.offset, self.cosine, self.sine = map(Decimal, (offset, cosine, sine))
        with localcontext() as ctx:
            ctx.prec = PRECISION
            self.axial_change = Decimal(width)*Decimal(slope)

    def value(self, angle):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            sine, cosine = _sincos(angle)
            return self.offset+self.cosine*cosine+self.sine*sine

    def endpoint_values(self, angle):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            first = self.value(angle)
            return float(first), float(first+self.axial_change)

    def breakpoints(self, lower, upper):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            lo, hi = Decimal(lower), Decimal(upper)
            points = {lo, hi}
            amplitude = (self.cosine*self.cosine+self.sine*self.sine).sqrt()
            if not amplitude:
                return sorted(points)
            phase = atan2(float(self.sine), float(self.cosine))
            for k in range(-2, 5):
                angle = Decimal(phase+k*pi)
                if lo < angle < hi:
                    points.add(angle)
            roots = {}
            offsets = ((self.offset, self.offset+self.axial_change) if self.axial_change
                       else (self.offset,))
            for offset in offsets:
                ratio = -offset/amplitude
                if not -1 <= ratio <= 1:
                    continue
                root = acos(float(ratio))
                for sign in (-1., 1.):
                    for k in range(-2, 4):
                        angle = Decimal(phase+sign*root+2*k*pi)
                        if not lo < angle < hi:
                            continue
                        if self.axial_change and self.offset+self.axial_change == +self.offset:
                            raise ValueError('material plane angular axial separation unresolved')
                        for _ in range(12):
                            sine, cosine = _sincos(angle)
                            derivative = -self.cosine*sine+self.sine*cosine
                            if not derivative:
                                raise ValueError('material plane angular root is unresolved; grazing excluded')
                            change = (offset+self.cosine*cosine+self.sine*sine)/derivative
                            angle -= change
                            if abs(change) <= Decimal('1e-40'):
                                break
                        else:
                            raise ValueError('material plane angular root precision exhausted')
                        if not lo < angle < hi:
                            raise ValueError('material plane angular root ownership unresolved')
                        # Distinct endpoint equations may describe a very thin
                        # partial-contact arc. Never silently merge that arc.
                        if angle in roots and roots[angle] != offset:
                            raise ValueError('material plane angular root separation unresolved')
                        roots[angle] = offset
                        points.add(angle)
            return sorted(points)
