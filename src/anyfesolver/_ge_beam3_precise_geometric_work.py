"""Private expression-preserving scalar work; not an activated beam operator.

Only the scalar contraction p.k(q)-Psi*(p) is evaluated here, with 80-digit
working arithmetic and one final binary64 rounding. No material evaluation,
derivative substitution, state ownership, history update or public routing.
The native analytic variations must be independently checked before adoption.
"""
from decimal import Decimal as D, Context, ROUND_HALF_EVEN, localcontext
from math import isfinite
from numbers import Real

POLICY = 'GE_BEAM3_RETAINED_GEOMETRIC_WORK_DECIMAL80_V1'
_PI = D('3.141592653589793238462643383279502884197169399375105820974944592307816406286208999')


def _number(x):
    if isinstance(x, bool) or not isinstance(x, Real) or not isfinite(float(x)):
        raise ValueError('finite real binary64 geometric input required')
    return D.from_float(float(x))


def _array(value, shape):
    if not shape:
        return _number(value)
    if isinstance(value, (str, bytes)) or len(value) != shape[0]:
        raise ValueError('exact geometric input dimensions required')
    return [_array(v, shape[1:]) for v in value]


def _tr(a):
    return list(map(list, zip(*a)))


def _mul(a, b):
    return [[sum((x*y for x, y in zip(row, col)), D(0)) for col in zip(*b)] for row in a]


def _exp(v):
    square = sum((x*x for x in v), D(0))
    if square >= (D('.9')*_PI)**2:
        raise ValueError('rotation increment requires refinement or cutback')
    # Entire Rodrigues coefficient series; no subtraction of nearly equal cosines.
    a = ta = D(1); b = tb = D('.5')
    for n in range(1, 161):
        ta *= -square/D(2*n*(2*n+1)); tb *= -square/D((2*n+1)*(2*n+2))
        a += ta; b += tb
        if max(abs(ta), abs(tb)) < D('1e-78'):
            break
    else:
        raise ArithmeticError('bounded exponential series did not converge')
    x, y, z = v; cross = [[D(0), -z, y], [z, D(0), -x], [-y, x, D(0)]]
    square_cross = _mul(cross, cross)
    return [[D(i == j)+a*cross[i][j]+b*square_cross[i][j] for j in range(3)] for i in range(3)]


def _atan_positive(x):
    if x < 0:
        raise ValueError('nonnegative half-angle argument')
    factor = 1
    for _ in range(16):
        if x <= D('.125'):
            break
        x /= 1+(1+x*x).sqrt(); factor *= 2
    else:
        raise ArithmeticError('bounded atan range reduction')
    power = x; value = x
    for n in range(1, 161):
        power *= -x*x; term = power/D(2*n+1); value += term
        if abs(term) < D('1e-78'):
            return factor*value
    raise ArithmeticError('bounded atan series did not converge')


def _log(a):
    cosine = (sum(a[i][i] for i in range(3))-1)/2
    delta = 1-cosine
    # This convergent form also handles tiny positive trace drift of an input
    # binary64 rotation. It agrees with the native near-identity continuation.
    if abs(delta) < D('.125'):
        value = term = D(1)
        for n in range(160):
            term *= delta*D(n+1)/D(2*n+3); value += term
            if abs(term) < D('1e-78'):
                break
        else:
            raise ArithmeticError('bounded logarithm series did not converge')
    else:
        if not -1 < cosine < 1:
            raise ValueError('proper relative rotation logarithm required')
        angle = 2*_atan_positive((delta/(1+cosine)).sqrt())
        if angle >= D('.9')*_PI:
            raise ValueError('relative rotation requires refinement or cutback')
        value = angle/(1-cosine*cosine).sqrt()
    return [value*(a[2][1]-a[1][2])/2, value*(a[0][2]-a[2][0])/2,
            value*(a[1][0]-a[0][1])/2]


def _proper(a):
    product = _mul(_tr(a), a)
    error = sum((product[i][j]-D(i == j))**2 for i in range(3) for j in range(3))
    determinant = sum(a[0][i]*(a[1][(i+1)%3]*a[2][(i+2)%3]-a[1][(i+2)%3]*a[2][(i+1)%3]) for i in range(3))
    if error > D('1e-22') or abs(determinant-1) > D('1e-11'):
        raise ValueError('proper input triad required')


def potential(reference_coordinates, reference_triads, positions, position_low,
              nodal_frames, cell_rotations, increment, resultants,
              material_potential, *, check=lambda: None,
              effective_nodal_deltas=None, effective_cell_deltas=None):
    """Resultants are the already updated native p; do not add their increment.

    material_potential is its actual high/low conjugate value pair. Inputs and
    caller Decimal context are unchanged. All arrays are copied before use.
    """
    check()
    with localcontext(Context(prec=80, rounding=ROUND_HALF_EVEN)):
        r0 = _array(reference_coordinates, (3, 3)); f0 = _array(reference_triads, (3, 3, 3))
        x = _array(positions, (3, 3)); low = _array(position_low, (3, 3))
        q = _array(nodal_frames, (3, 3, 3)); u = _array(cell_rotations, (2, 3, 3))
        d = _array(increment, (42,)); p = _array(resultants, (18,)); material = _array(material_potential, (2,))
        for frame in f0+q+u:
            _proper(frame)
        if effective_nodal_deltas is not None or effective_cell_deltas is not None:
            if effective_nodal_deltas is None or effective_cell_deltas is None:
                raise ValueError('complete explicit effective SO3 base required')
            dq = _array(effective_nodal_deltas, (3, 3, 3)); du = _array(effective_cell_deltas, (2, 3, 3))
            # Preserve tiny D BEFORE adding I in Decimal80, using the exact same
            # captured effective base as the complete Jet2 successor work.
            q = [[[D(i == j)+dq[n][i][j] for j in range(3)] for i in range(3)] for n in range(3)]
            u = [[[D(i == j)+du[c][i][j] for j in range(3)] for i in range(3)] for c in range(2)]
            for frame in q+u:
                _proper(frame)
        made_q = [_mul(_exp(d[6*n+3:6*n+6]), q[n]) for n in range(3)]
        z = []; ell = []
        for c, (left, right) in enumerate(((0, 1), (1, 2))):
            check(); made_u = _mul(_exp(d[18+3*c:21+3*c]), u[c])
            chord = [[x[right][j]+low[right][j]+d[6*right+j]-x[left][j]-low[left][j]-d[6*left+j]] for j in range(3)]
            moved = _mul(_tr(made_u), chord)
            z.extend(moved[j][0]-r0[right][j]+r0[left][j] for j in range(3))
            for n, sign in ((left, -1), (right, 1)):
                frame = _mul(made_u, f0[n])
                ell.extend(D(sign)*v for v in _log(_mul(_tr(frame), made_q[n])))
        answer = float(sum((a*b for a, b in zip(p, z+ell)), D(0))-sum(material))
    check()
    if not isfinite(answer):
        raise ValueError('finite scalar geometric work required')
    return answer
