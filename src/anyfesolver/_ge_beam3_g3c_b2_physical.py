"""Private physical-flexibility B2 core. Not publicly routed or qualified.

Modal coefficients and all intermediate arithmetic use exact ratios of the
supplied binary64 values; only final outputs are rounded. Force/recovery retain
the modal split. Dense output arrays cannot retain entries smaller than the
binary64 subnormal range, and do not promise arbitrary condition numbers.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np

OPERATOR_ID = 'GE_BEAM3_G3C_B2_PHYSICAL_FLEXIBILITY_V1'


class RepresentabilityError(ValueError):
    """A required positive coefficient or computed output is not representable."""


def owned(value):
    a = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise RepresentabilityError('nonfinite physical B2 output')
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('positive finite nonboolean scalar required')
    return float(value)


def rounded_positive(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise RepresentabilityError('positive coefficient overflow') from exc
    if not math.isfinite(result) or result <= 0:
        raise RepresentabilityError('positive coefficient overflow or underflow')
    return result


def rounded(value, require_nonzero=False):
    """Round only final outputs; no intermediate float products or squares."""
    try:
        result = float(value)
    except OverflowError as exc:
        raise RepresentabilityError('physical B2 output overflow') from exc
    if not math.isfinite(result):
        raise RepresentabilityError('nonfinite physical B2 output')
    if require_nonzero and value != 0 and result == 0:
        raise RepresentabilityError('nonzero physical energy underflow')
    return result


def physical_rigidities(E, nu, area, Iy, Iz, J, ky, kz):
    E, area, Iy, Iz, J, ky, kz = map(positive, (E, area, Iy, Iz, J, ky, kz))
    if type(nu) not in (int, float) or not math.isfinite(nu) or not -1 < nu < .5:
        raise ValueError('physical isotropic Poisson ratio required')
    e, a, iy, iz, j, y, z = map(Fraction, (E, area, Iy, Iz, J, ky, kz))
    g = e / (2 * (1 + Fraction(nu)))
    return owned([rounded_positive(v) for v in (e*a, g*a*y, g*a*z, g*j, e*iy, e*iz)])


@dataclass(frozen=True)
class CoreTrial:
    energy: float
    force: np.ndarray
    tangent: np.ndarray
    strains: np.ndarray
    resultants: np.ndarray
    strain_differential: np.ndarray
    stations: np.ndarray
    weights: np.ndarray
    operator_id: str = OPERATOR_ID
    production_qualified: bool = False


def evaluate(length, rigidities, deformation):
    """Evaluate in the authoritative physical reference section frame only.

    No matrices/poses, family dispatch, accepted state or restart are owned here.
    The successor adapter must supply their separately reviewed contracts.
    """
    L = positive(length)
    if (type(rigidities) is not np.ndarray or rigidities.dtype != np.dtype('float64')
            or rigidities.shape != (6,) or not np.isfinite(rigidities).all()
            or np.any(rigidities <= 0)):
        raise ValueError('six physical positive binary64 rigidities required')
    if (type(deformation) is not np.ndarray or deformation.dtype != np.dtype('float64')
            or deformation.shape != (12,) or not np.isfinite(deformation).all()):
        raise ValueError('twelve finite binary64 local coordinates required')
    S, d = owned(rigidities), owned(deformation)
    EA, Sy, Sz, GJ, EIy, EIz = map(Fraction, map(float, S))
    d = np.array([Fraction(float(v)) for v in d], dtype=object)
    lv = Fraction(L)
    B = np.full((6, 12), Fraction(0), dtype=object)
    B[0, 0], B[0, 6] = -1, 1
    B[1, 3], B[1, 9] = -1, 1
    for row, rot in ((2, 4), (3, 10)):
        B[row, 2], B[row, 8], B[row, rot] = -1/lv, 1/lv, 1
    for row, rot in ((4, 5), (5, 11)):
        B[row, 1], B[row, 7], B[row, rot] = 1/lv, -1/lv, 1
    torsion = GJ/lv
    force = torsion * (B[1] @ d) * B[1]
    tangent = torsion * np.outer(B[1], B[1])
    energy = Fraction(1, 2) * torsion * (B[1] @ d)**2
    mode_data = []
    for first, EI, shear in ((2, EIy, Sz), (4, EIz, Sy)):
        # Never infer the antisymmetric mode from rounded diagonal differences.
        p = lv/(6*Fraction(EI)) + 2/(Fraction(shear)*lv)
        kp = 1/p
        km = 2*EI/lv
        plus, minus = B[first]+B[first+1], B[first]-B[first+1]
        bp, bm = plus @ d, minus @ d
        qp, qm = kp*bp/2, km*bm/2
        force = force + qp*plus + qm*minus
        tangent = tangent + kp*np.outer(plus,plus)/2 + km*np.outer(minus,minus)/2
        energy += kp*bp**2/4 + km*bm**2/4
        mode_data.append((qp,qm,kp*plus/2,km*minus/2))
    du, dv, dw = (d[6:9]-d[:3])/lv
    strain = du + (dv*dv+dw*dw)/2
    de = np.full(12, Fraction(0), dtype=object)
    de[:3] = -np.array([Fraction(1),dv,dw], dtype=object)/lv; de[6:9] = -de[:3]
    he = np.full((12,12), Fraction(0), dtype=object)
    for i in (1,2):
        a = np.full(12, Fraction(0), dtype=object); a[i],a[i+6] = -1/lv,1/lv; he += np.outer(a,a)
    N = EA*strain
    force += lv*N*de
    tangent += EA*lv*(np.outer(de,de)+strain*he)
    energy += EA*lv*strain**2/2
    points = np.array([-math.sqrt(3/5),0.,math.sqrt(3/5)])
    weights = [rounded(w*lv/2) for w in (Fraction(5,9),Fraction(8,9),Fraction(5,9))]
    sf = np.array([EA,Sy,Sz,GJ,EIy,EIz], dtype=object)
    rows, differentials = [], []
    for xi in points:
        xi = Fraction(float(xi))
        row = np.full(6, Fraction(0), dtype=object); dr = np.full((6,12), Fraction(0), dtype=object)
        row[0],dr[0] = N,EA*de
        row[3],dr[3] = torsion*(B[1]@d),torsion*B[1]
        for (qp,qm,dp,dm), shear_index, moment_index, sign in (
                (mode_data[0],2,4,1), (mode_data[1],1,5,-1)):
            row[shear_index],dr[shear_index] = sign*2*qp/lv,sign*2*dp/lv
            row[moment_index],dr[moment_index] = xi*qp-qm,xi*dp-dm
        rows.append(row); differentials.append(dr/sf[:,None])
    def final(values):
        values = np.asarray(values, dtype=object)
        return owned(np.array([rounded(v) for v in values.flat]).reshape(values.shape))
    resultants = final(rows)
    if energy < 0:
        raise RepresentabilityError('nonfinite or negative physical energy')
    return CoreTrial(rounded(energy, require_nonzero=True),final(force),final(tangent),final(np.array(rows)/sf),
                     resultants,final(differentials),owned(points),owned(weights))
