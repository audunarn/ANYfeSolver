"""Exact represented-coefficient Bernstein certificate for enriched target-plane gap.

This is the original GLOBAL XY footprint/target-plane metric, not a sampled
source-normal gap. Rational box bounds cover the complete bicubic B4 skin.
"""
from fractions import Fraction as F
from math import comb, sqrt
import numpy as np
from anyfesolver.e4_pl_enriched import EnrichedProfileError
ReferenceRefused=EnrichedProfileError
def require(ok,message):
    if not ok: raise EnrichedProfileError(message)


def rat(x):
    return F.from_float(float(x))


def add(a, b):
    c = dict(a)
    for key, value in b.items():
        c[key] = c.get(key, F(0))+value
    return {k: v for k, v in c.items() if v}


def scale(a, value):
    return {k: v*value for k, v in a.items() if v*value}


def multiply(a, b):
    c = {}
    for (i, j), x in a.items():
        for (k, l), y in b.items():
            key = i+k, j+l
            c[key] = c.get(key, F(0))+x*y
    return {k: v for k, v in c.items() if v}


def coefficients(corners, beta, normal):
    """Exact coefficients of represented Q1 skin + B{1,r,s,rs}*n_mid."""
    hats = [
        {(0, 0): F(1, 4), (1, 0): F(-1, 4), (0, 1): F(-1, 4), (1, 1): F(1, 4)},
        {(0, 0): F(1, 4), (1, 0): F(1, 4), (0, 1): F(-1, 4), (1, 1): F(-1, 4)},
        {(0, 0): F(1, 4), (1, 0): F(1, 4), (0, 1): F(1, 4), (1, 1): F(1, 4)},
        {(0, 0): F(1, 4), (1, 0): F(-1, 4), (0, 1): F(1, 4), (1, 1): F(-1, 4)}]
    base = {(0, 0): F(1), (2, 0): F(-1), (0, 2): F(-1), (2, 2): F(1)}
    amplitude = {(0, 0): rat(beta[0]), (1, 0): rat(beta[1]),
                 (0, 1): rat(beta[2]), (1, 1): rat(beta[3])}
    bubble = multiply(base, amplitude)
    result = []
    for axis in range(3):
        polynomial = scale(bubble, rat(normal[axis]))
        for i in range(4):
            polynomial = add(polynomial, scale(hats[i], rat(corners[i, axis])))
        result.append(polynomial)
    return result


def linear_form(polynomials, vector, offset):
    result = {(0, 0): -sum(x*y for x, y in zip(vector, offset))}
    for p, value in zip(polynomials, vector):
        result = add(result, scale(p, value))
    return result


def bernstein_range(poly, box):
    """Exact power-to-Bernstein conversion on an exact dyadic parameter box."""
    r0, r1, s0, s1 = box
    power = {}
    for (i, j), value in poly.items():
        for k in range(i+1):
            for l in range(j+1):
                key = k, l
                power[key] = power.get(key, F(0))+value*comb(i, k)*r0**(i-k)*(r1-r0)**k*comb(j, l)*s0**(j-l)*(s1-s0)**l
    values = []
    for k in range(4):
        for l in range(4):
            values.append(sum(value*F(comb(k, i), comb(3, i))*F(comb(l, j), comb(3, j))
                              for (i, j), value in power.items() if i <= k and j <= l))
    return min(values), max(values)


def exact_cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def sqrt_bracket(value):
    center = sqrt(float(value))
    require(np.isfinite(center) and center > 0, 'finite target plane norm')
    lo, hi = center, center
    while rat(lo)**2 > value:
        lo = float(np.nextafter(lo, -np.inf))
    while rat(hi)**2 < value:
        hi = float(np.nextafter(hi, np.inf))
    return rat(lo), rat(hi)


def global_gap_certificate(oracle, pose, *, floor=-50e-6, max_boxes=100000, max_depth=14):
    """No unknown/partial region can pass. Every relevant box has a lower bound."""
    oracle.verify()
    sources, targets = oracle.vertices(pose)
    retained = visited = 0
    global_lower = None
    witnesses = []
    for sid, (corners, beta, body) in sources.items():
        poly = coefficients(corners, beta, body)
        for tid, target in targets.items():
            t = [[rat(x) for x in p] for p in target]
            a = [t[1][i]-t[0][i] for i in range(3)]
            b = [t[2][i]-t[0][i] for i in range(3)]
            normal = exact_cross(a, b)
            norm_lo, norm_hi = sqrt_bracket(sum(x*x for x in normal))
            require(normal[2]/norm_hi > F(1, 2), 'original target-normal direction/XY projection domain')
            determinant = a[0]*b[1]-a[1]*b[0]
            require(determinant != 0, 'nondegenerate original global XY footprint')
            e1 = linear_form(poly, (b[1]/determinant, -b[0]/determinant, F(0)), t[0])
            e2 = linear_form(poly, (-a[1]/determinant, a[0]/determinant, F(0)), t[0])
            e0 = add({(0, 0): F(1)}, scale(add(e1, e2), F(-1)))
            numerator = linear_form(poly, normal, t[0])
            boxes = [(F(-1), F(1), F(-1), F(1), 0)]
            while boxes:
                r0, r1, s0, s1, depth = boxes.pop()
                visited += 1
                require(visited <= max_boxes, 'global physical gap box budget exhausted')
                if visited % 64 == 0:
                    oracle.verify()
                box = r0, r1, s0, s1
                bary = [bernstein_range(e, box) for e in (e0, e1, e2)]
                if any(hi < 0 for lo, hi in bary):
                    continue
                lower_num, upper_num = bernstein_range(numerator, box)
                lower = lower_num/(norm_lo if lower_num < 0 else norm_hi)
                upper = upper_num/(norm_hi if upper_num < 0 else norm_lo)
                if lower >= rat(floor):
                    retained += 1
                    if global_lower is None or lower < global_lower:
                        global_lower = lower
                        witnesses = [sid, tid, [str(x) for x in box]]
                    continue
                if upper < rat(floor) and all(lo >= 0 for lo, hi in bary):
                    raise ReferenceRefused('independent global target-plane physical gap violates original floor')
                require(depth < max_depth, 'global physical gap unresolved; no sample substitution')
                rm, sm = (r0+r1)/2, (s0+s1)/2
                boxes.extend((a0, a1, b0, b1, depth+1)
                             for a0, a1 in ((r0, rm), (rm, r1))
                             for b0, b1 in ((s0, sm), (sm, s1)))
    require(retained > 0 and global_lower is not None, 'global gap has no certified overlap')
    lower = float(global_lower)
    if rat(lower) > global_lower:
        lower = float(np.nextafter(lower, -np.inf))
    oracle.verify()
    return dict(status='CERTIFIED_REPRESENTED_BICUBIC_GLOBAL_XY_TARGET_PLANE',
                lower_m=lower, original_floor_m=floor, visited=visited, retained=retained,
                witness=witnesses, scope='whole reference square intersected with each ORIGINAL global XY S3 footprint; exact rational Bernstein coefficients, outward final bound',
                source_normal_metric=False, sampled_surrogate=False,
                limitation='represented binary64 physical corner/director coefficients; no PDE/model-error certificate')
