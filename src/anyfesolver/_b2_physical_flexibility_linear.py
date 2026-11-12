"""Experimental straight-reference B2 elastic force-flexibility operator.

This local operator makes no qualification claim. Its reference-linear consumers
admit operations separately. Coordinates are local [u,v,w,rx,ry,rz] at each node;
section resultants/strains are [N,Vy,Vz,T,My,Mz].  Exact rational arithmetic
retains weak shear modes until the requested binary64 output is rounded.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


OPERATOR_ID = "B2_PHYSICAL_FLEXIBILITY_LINEAR_V1"
OUTPUT_REPRESENTATION_ID = "B2_BOUNDED_OUTPUT_REPRESENTATION_V1"
_ENERGY_LIMIT = Fraction(1, 10**8)
_RIGID_LIMIT = Fraction(1, 10**9)


class RepresentabilityError(ValueError):
    """A nonzero physical result cannot be represented in binary64."""


class ConditioningAdmissionError(ValueError):
    """Local numerical output cannot meet its declared representation limits."""


@dataclass(frozen=True)
class OutputRepresentationCertificate:
    """Basic energies and separate bounded rigid/cross work, not graph stability.

    Errors are exact rational ratios. This certifies rounding of the implemented
    law, whose physical correctness still needs independent mechanics oracles.
    """

    basic_energy_preserved: bool
    rigid_work_preserved: bool
    cross_work_preserved: bool
    rigid_wrench_error: Fraction
    rigid_work_error: Fraction
    cross_work_error: Fraction
    identity: str = OUTPUT_REPRESENTATION_ID

    @property
    def passed(self):
        return (self.basic_energy_preserved and self.rigid_work_preserved and
                self.cross_work_preserved)


class OutputRepresentationError(ConditioningAdmissionError):
    """The emitted binary64 tangent failed its bounded local work certificate."""

    def __init__(self, certificate):
        self.certificate = certificate
        failed = [name for name, passed in (
            ("basic energy", certificate.basic_energy_preserved),
            ("rigid work", certificate.rigid_work_preserved),
            ("cross work", certificate.cross_work_preserved)) if not passed]
        super().__init__("physical B2 tangent output representation failed: " +
                         ", ".join(failed))


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite nonboolean physical input required")
    return Fraction(float(value))


def _float(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise RepresentabilityError("physical B2 output overflow") from exc
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise RepresentabilityError("physical B2 output overflow or underflow")
    # A nonzero subnormal can still be a poor representation of an exact
    # physical result.  Compare as rationals so the check itself does not
    # underflow or round away the error.  Normal binary64 rounding satisfies
    # this bound. This checks the declared numerical output units (public
    # physical definitions use SI); it is not a physical or unit-covariant
    # constitutive tolerance near binary64 underflow.
    if value != 0:
        exact = Fraction(value)
        if abs(Fraction(result) - exact) > abs(exact) / (1 << 52):
            raise RepresentabilityError("physical B2 output loses binary64 relative precision")
    return result


def _array(value):
    array = np.asarray(value, dtype=object)
    return np.array([_float(item) for item in array.flat], dtype=float).reshape(array.shape)


def _inverse_spd(matrix):
    """Exact LDL and inverse of the supplied law; no floating pivot heuristic.

    SPD is an input-law check. Emitted numerical tangents receive a separate
    output certificate, and assembled solves retain their equilibrium checks.
    """
    n = len(matrix)
    diagonal = [matrix[i][i] for i in range(n)]
    if any(d <= 0 for d in diagonal):
        raise ValueError("positive-definite physical section/flexibility required")
    lower = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    pivots = []
    for j in range(n):
        pivot = matrix[j][j] - sum(lower[j][k]**2 * pivots[k] for k in range(j))
        if pivot <= 0:
            raise ValueError("positive-definite physical section/flexibility required")
        pivots.append(pivot)
        for i in range(j + 1, n):
            lower[i][j] = (matrix[i][j] - sum(
                lower[i][k] * pivots[k] * lower[j][k] for k in range(j)
            )) / pivot
    columns = []
    for column in range(n):
        y = []
        for i in range(n):
            y.append(Fraction(int(i == column)) - sum(lower[i][k] * y[k] for k in range(i)))
        z = [y[i] / pivots[i] for i in range(n)]
        x = [Fraction(0)] * n
        for i in range(n - 1, -1, -1):
            x[i] = z[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))
        columns.append(x)
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def _basic(length):
    B = [[Fraction(0) for _ in range(12)] for _ in range(6)]
    B[0][0], B[0][6] = -1, 1
    B[1][3], B[1][9] = -1, 1
    for row, rotation in ((2, 4), (3, 10)):
        B[row][2], B[row][8], B[row][rotation] = -1 / length, 1 / length, 1
    for row, rotation in ((4, 5), (5, 11)):
        B[row][1], B[row][7], B[row][rotation] = 1 / length, -1 / length, 1
    return B


def _interpolation(length, station):
    t = station / length
    H = [[Fraction(0) for _ in range(6)] for _ in range(6)]
    H[0][0] = H[3][1] = 1
    H[2][2] = H[2][3] = 1 / length
    H[4][2], H[4][3] = t - 1, t
    H[1][4] = H[1][5] = -1 / length
    H[5][4], H[5][5] = t - 1, t
    return H


def _multiply(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _transpose(matrix):
    return list(map(list, zip(*matrix)))


def _matmul(a, b):
    return [[sum((x*y for x, y in zip(row, column)), Fraction(0))
             for column in zip(*b)] for row in a]


def _positive_semidefinite(matrix):
    """Exact symmetric Schur test; zero pivots require a zero residual column."""
    work = [list(row) for row in matrix]
    if work != _transpose(work):
        raise ValueError("symmetric physical work certificate required")
    for j in range(len(work)):
        pivot = work[j][j]
        if pivot < 0:
            return False
        if not pivot:
            if any(work[i][j] for i in range(j+1, len(work))):
                return False
            continue
        for i in range(j+1, len(work)):
            for k in range(j+1, len(work)):
                work[i][k] -= work[i][j]*work[j][k]/pivot
    return True


def _nodal_tangent(operator):
    # One exact action per nodal column; all returned coefficients retain the
    # previous exact formulas before binary64 rounding.
    columns = [operator._basic_action([row[column] for row in operator._B])
               for column in range(12)]
    return tuple(tuple(_float(sum(operator._B[j][i]*columns[column][j]
                                  for j in range(6)))
                       for column in range(12)) for i in range(12))


def _certify_output_tangent(operator, tangent):
    """Certify the emitted tangent with the reviewed bounded physical norms.

    The basic-energy inequalities hold for all basic directions. Separate rigid
    and cross limits apply to normalized amplitude cubes, not an all-amplitude
    relative energy bound or one combined total-energy estimate. No matrix is
    symmetrized. Assembled graph/affine solve checks remain mandatory.
    """
    emitted = np.asarray(tangent, dtype=float)
    if (emitted.shape != (12, 12) or not np.isfinite(emitted).all() or
            not np.array_equal(emitted, emitted.T)):
        raise RepresentabilityError("finite symmetric physical B2 tangent required")
    length = operator.length
    B = operator._B
    A = _transpose([operator._basic_action(
        [Fraction(i == j) for i in range(6)]) for j in range(6)])
    # Canonical right inverse: first node fixed, B[:, 6:] inverted exactly.
    R = [[Fraction(0)]*6 for _ in range(12)]
    R[6][0] = R[9][1] = 1
    R[8][2], R[10][2], R[10][3] = length, -1, 1
    R[7][4], R[11][4], R[11][5] = -length, -1, 1
    if _matmul(B, R) != [[Fraction(i == j) for j in range(6)] for i in range(6)]:
        raise RuntimeError("physical B2 basic/right-inverse contract mismatch")
    Z = [[Fraction(0)]*6 for _ in range(12)]
    for end, x in ((0, -length/2), (6, length/2)):
        for axis in range(3):
            Z[end+axis][axis] = length
            Z[end+3+axis][3+axis] = 1
        Z[end+2][4], Z[end+1][5] = -x, x
    if any(v for row in _matmul(B, Z) for v in row):
        raise RuntimeError("physical B2 basic/rigid-kernel contract mismatch")
    K = [[Fraction(float(v)) for v in row] for row in emitted]
    pulled = _matmul(_matmul(_transpose(R), K), R)
    delta = [[pulled[i][j]-A[i][j] for j in range(6)] for i in range(6)]
    energy_ok = all(_positive_semidefinite([
        [_ENERGY_LIMIT*A[i][j]+sign*delta[i][j] for j in range(6)]
        for i in range(6)]) for sign in (-1, 1))
    S = operator._stiffness
    force = max([S[i][i] for i in range(3)] +
                [S[i][i]/length**2 for i in range(3, 6)])
    energy = force*length
    wrench_scales = [force]*3+[force*length]*3
    KZ = _matmul(K, Z)
    rigid_wrench = max(sum(abs(v) for v in row)/wrench_scales[i % 6]
                       for i, row in enumerate(KZ))
    rigid_work = sum(abs(v) for row in _matmul(_transpose(Z), KZ)
                     for v in row)/energy
    R_scaled = [[v*(length if j == 0 else 1) for j, v in enumerate(row)]
                for row in R]
    cross_work = sum(abs(v) for row in _matmul(
        _matmul(_transpose(Z), K), R_scaled) for v in row)/energy
    certificate = OutputRepresentationCertificate(
        energy_ok, rigid_wrench <= _RIGID_LIMIT and rigid_work <= _ENERGY_LIMIT,
        cross_work <= _ENERGY_LIMIT, rigid_wrench, rigid_work, cross_work)
    if not certificate.passed:
        raise OutputRepresentationError(certificate)
    return certificate


def _flexibility(length, compliance):
    # H(t) is affine: the exact integral uses endpoint values and their cross terms.
    h0, h1 = _interpolation(length, Fraction(0)), _interpolation(length, length)
    result = [[Fraction(0) for _ in range(6)] for _ in range(6)]
    for i in range(6):
        for j in range(6):
            result[i][j] = length * sum(
                compliance[a][b] * (
                    h0[a][i]*h0[b][j]*Fraction(1, 3)
                    + h1[a][i]*h1[b][j]*Fraction(1, 3)
                    + (h0[a][i]*h1[b][j] + h1[a][i]*h0[b][j])*Fraction(1, 6)
                ) for a in range(6) for b in range(6)
            )
    return result


def _separated_basic_inverse(length, flexibility):
    """Invert in a fixed midpoint shear/moment basis, return original q rows.

    The exact congruence changes only the inversion coordinates.  Section
    resultants, basic displacements and station traces retain their original
    physical conventions.
    """
    change = [[Fraction(0) for _ in range(6)] for _ in range(6)]
    change[0][0], change[1][1] = 1, length
    change[2][2] = change[3][2] = length/2
    change[2][3], change[3][3] = -length, length
    change[4][4] = change[5][4] = -length/2
    change[4][5], change[5][5] = -length, length
    separated = [[sum(change[a][i]*flexibility[a][b]*change[b][j]
                      for a in range(6) for b in range(6))
                  for j in range(6)] for i in range(6)]
    inverse = _inverse_spd(separated)
    return [[sum(change[i][a]*inverse[a][b]*change[j][b]
                 for a in range(6) for b in range(6))
             for j in range(6)] for i in range(6)]


@dataclass(frozen=True)
class LinearSection:
    stiffness: tuple[tuple[float, ...], ...]
    kind: str
    operator_id: str = OPERATOR_ID
    # E, nu, area, Iy, Iz, J, ky, kz; absent for a generalized 6x6 law.
    scalar_parameters: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.operator_id != OPERATOR_ID or self.kind not in ("isotropic_scalar", "elastic_generalized"):
            raise ValueError("explicit physical linear B2 section identity required")
        try:
            data = np.asarray(self.stiffness, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("finite 6x6 physical section required") from exc
        if data.shape != (6, 6) or not np.isfinite(data).all() or not np.array_equal(data, data.T):
            raise ValueError("finite symmetric 6x6 physical section required")
        if self.kind == "isotropic_scalar" and np.count_nonzero(data - np.diag(np.diag(data))):
            raise ValueError("isotropic scalar section cannot contain coupling")
        if self.kind == "isotropic_scalar":
            values = self.scalar_parameters
            if values is None or len(values) != 8:
                raise ValueError("isotropic scalar source parameters required")
            E, nu, area, Iy, Iz, J, ky, kz = values
            e, a, iy, iz, j, y, z = map(_positive, (E, area, Iy, Iz, J, ky, kz))
            if type(nu) not in (int, float) or not math.isfinite(nu) or not -1 < nu < .5:
                raise ValueError("physical isotropic Poisson ratio required")
            g = e / (2 * (1 + Fraction(float(nu))))
            expected = [_float(value) for value in
                        (e*a, g*a*y, g*a*z, g*j, e*iy, e*iz)]
            if not np.array_equal(np.diag(data), expected):
                raise ValueError("scalar stiffness does not match source parameters")
            object.__setattr__(self, "scalar_parameters", tuple(float(v) for v in values))
        elif self.scalar_parameters is not None:
            raise ValueError("generalized stiffness has no scalar source parameters")
        _inverse_spd([[Fraction(v) for v in row] for row in data])
        # Own the descriptor even when a caller supplied mutable nested lists.
        object.__setattr__(self, "stiffness", tuple(tuple(float(v) for v in row) for row in data))

    @classmethod
    def isotropic(cls, *, E, nu, area, Iy, Iz, J, ky, kz):
        e, a, iy, iz, j, y, z = map(_positive, (E, area, Iy, Iz, J, ky, kz))
        if type(nu) not in (int, float) or not math.isfinite(nu) or not -1 < nu < .5:
            raise ValueError("physical isotropic Poisson ratio required")
        g = e / (2 * (1 + Fraction(float(nu))))
        values = [e*a, g*a*y, g*a*z, g*j, e*iy, e*iz]
        # Do not hide unrepresentable physical section data behind a floor.
        diagonal = [_float(value) for value in values]
        return cls(tuple(tuple(diagonal[i] if i == j else 0.0 for j in range(6))
                         for i in range(6)), "isotropic_scalar",
                   scalar_parameters=(E, nu, area, Iy, Iz, J, ky, kz))

    @classmethod
    def generalized(cls, stiffness):
        try:
            data = np.asarray(stiffness, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("finite 6x6 generalized section required") from exc
        if data.shape != (6, 6) or not np.isfinite(data).all() or not np.array_equal(data, data.T):
            raise ValueError("finite symmetric 6x6 generalized section required")
        return cls(tuple(tuple(float(v) for v in row) for row in data), "elastic_generalized")


@dataclass(frozen=True)
class LocalResponse:
    energy: float
    force: np.ndarray
    tangent: np.ndarray
    basic_force: np.ndarray
    resultants: np.ndarray
    strains: np.ndarray
    stations: np.ndarray
    operator_id: str = OPERATOR_ID
    production_qualified: bool = False


@dataclass(frozen=True)
class LocalSectionRecovery:
    """Selected station quantities without a force/tangent evaluation."""

    resultants: np.ndarray
    strains: np.ndarray
    stations: np.ndarray
    operator_id: str = OPERATOR_ID
    production_qualified: bool = False


@dataclass(frozen=True)
class EquilibratedSectionRecovery:
    """Static admissible fields, distinct from the nodal interpolation strain.

    Interior concentrated actions have explicit left/right traces. At member
    endpoints both columns are the inward trace. No loaded interior pose or
    attachment compatibility is implied by these reconstructed fields.
    """

    stations: np.ndarray
    resultants_left: np.ndarray
    resultants_right: np.ndarray
    strains_left: np.ndarray
    strains_right: np.ndarray
    endpoint_tractions: np.ndarray
    operator_id: str = OPERATOR_ID
    production_qualified: bool = False

    def __post_init__(self):
        n = len(self.stations)
        for name, shape in (("stations", (n,)), ("resultants_left", (n, 6)),
                            ("resultants_right", (n, 6)), ("strains_left", (n, 6)),
                            ("strains_right", (n, 6)), ("endpoint_tractions", (12,))):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise RepresentabilityError("equilibrated section field is not representable")
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=float).reshape(shape))


@dataclass(frozen=True)
class LinearTrace:
    """Owned local reference points, offset poses, and their nodal derivatives."""

    stations: np.ndarray
    reference_points: np.ndarray
    pose: np.ndarray
    jacobian: np.ndarray
    operator_id: str = OPERATOR_ID
    production_qualified: bool = False


def _pose_polynomials(length, e0, e1, nodal):
    """Exact coefficients in normalized station t, ordered [u,v,w,rx,ry,rz]."""
    zero = Fraction(0)
    result = [[zero]*4 for _ in range(6)]
    result[0] = [nodal[0], length*e0[0], length*e1[0]/2, zero]
    for row, strain in ((3, 3), (4, 4), (5, 5)):
        result[row] = [nodal[row], length*e0[strain], length*e1[strain]/2, zero]
    result[1] = [nodal[1], length*(e0[1]+nodal[5]),
                 length*(e1[1]+length*e0[5])/2, length**2*e1[5]/6]
    result[2] = [nodal[2], length*(e0[2]-nodal[4]),
                 length*(e1[2]-length*e0[4])/2, -length**2*e1[4]/6]
    return result


class LinearOperator:
    """Physical local tangent and station response; independent of public B2."""

    __slots__ = ("length", "section", "_stiffness", "_compliance", "_B",
                 "_modal", "_basic_inverse", "_trace_polynomials", "_sealed",
                 "_tangent", "_output_certificate", "_uniform_load_map")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("physical B2 operator is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise AttributeError("physical B2 operator is immutable")
        object.__delattr__(self, name)

    def __init__(self, length, section: LinearSection):
        self.length = _positive(length)
        if not isinstance(section, LinearSection) or section.operator_id != OPERATOR_ID:
            raise ValueError("explicit physical linear B2 section required")
        self.section = section
        self._stiffness = tuple(tuple(Fraction(value) for value in row) for row in section.stiffness)
        self._compliance = tuple(tuple(row) for row in _inverse_spd(self._stiffness))
        self._B = tuple(tuple(row) for row in _basic(self.length))
        if section.kind == "isotropic_scalar":
            s = [self._stiffness[i][i] for i in range(6)]
            kp_y = 1 / (self.length/(6*s[4]) + 2/(s[2]*self.length))
            kp_z = 1 / (self.length/(6*s[5]) + 2/(s[1]*self.length))
            self._modal = (s[0]/self.length, s[3]/self.length,
                           kp_y, 2*s[4]/self.length, kp_z, 2*s[5]/self.length)
            for value in self._modal:
                _float(value)
        else:
            self._basic_inverse = tuple(tuple(row) for row in
                _separated_basic_inverse(self.length,
                    _flexibility(self.length, self._compliance)))
        h0 = _interpolation(self.length, Fraction(0))
        h1 = _interpolation(self.length, self.length)
        coefficients = [[[Fraction(0) for _ in range(4)] for _ in range(12)]
                        for _ in range(6)]
        for column in range(12):
            nodal = [Fraction(int(i == column)) for i in range(12)]
            q = self._basic_action(_multiply(self._B, nodal))
            e0 = _multiply(self._compliance, _multiply(h0, q))
            e1 = _multiply(self._compliance, _multiply(
                [[h1[i][j]-h0[i][j] for j in range(6)] for i in range(6)], q))
            pose = _pose_polynomials(self.length, e0, e1, nodal)
            for row in range(6):
                coefficients[row][column] = pose[row]
        self._trace_polynomials = tuple(tuple(tuple(poly) for poly in row)
                                        for row in coefficients)
        self._uniform_load_map = tuple(tuple(self.length*sum(
            coefficient/Fraction(degree+1) for degree, coefficient in
            enumerate(self._trace_polynomials[row][column])) for row in range(6))
            for column in range(12))
        self._tangent = _nodal_tangent(self)
        self._output_certificate = _certify_output_tangent(self, self._tangent)
        self._sealed = True

    @property
    def output_certificate(self):
        """Immutable bounded local certificate; not production qualification."""
        return self._output_certificate

    @property
    def basic_matrix(self):
        return _array(self._B)

    @property
    def modal_stiffness(self):
        return None if self.section.kind != "isotropic_scalar" else _array(self._modal)

    def _basic_action(self, basic):
        if self.section.kind != "isotropic_scalar":
            return _multiply(self._basic_inverse, basic)
        ka, kt, kp_y, km_y, kp_z, km_z = self._modal
        q = [ka*basic[0], kt*basic[1]]
        for i, kp, km in ((2, kp_y, km_y), (4, kp_z, km_z)):
            plus = kp*(basic[i]+basic[i+1])/2
            minus = km*(basic[i]-basic[i+1])/2
            q.extend((plus+minus, plus-minus))
        return q

    def trace(self, displacement, *, stations=(0.0, 0.5, 1.0),
              offset_local_m=(0.0, 0.0, 0.0)):
        """Private small-rotation physical pose and exact offset-point pullback."""
        try:
            u = np.asarray(displacement, dtype=float)
            x = np.asarray(stations, dtype=float)
            offset = np.asarray(offset_local_m, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("finite local coordinates, stations and offset required") from exc
        if u.shape != (12,) or not np.isfinite(u).all():
            raise ValueError("twelve finite local coordinates required")
        if x.ndim != 1 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("stations must be finite normalized coordinates in [0,1]")
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("three finite local offset coordinates required")
        d = [Fraction(float(value)) for value in offset]
        nodal = [Fraction(float(value)) for value in u]
        points, poses, maps = [], [], []
        for station in x:
            t = Fraction(float(station))
            center = [[sum(coefficient*t**degree for degree, coefficient in enumerate(poly))
                       for poly in row] for row in self._trace_polynomials]
            # The point displacement is r + theta cross d. Its force
            # pullback therefore includes the offset's d cross F moment.
            for row, rotational, sign in ((0, (4, 5), (d[2], -d[1])),
                                          (1, (5, 3), (d[0], -d[2])),
                                          (2, (3, 4), (d[1], -d[0]))):
                center[row] = [value + sign[0]*center[rotational[0]][col]
                               + sign[1]*center[rotational[1]][col]
                               for col, value in enumerate(center[row])]
            maps.append(center)
            poses.append(_multiply(center, nodal))
            points.append([self.length*t+d[0], d[1], d[2]])
        n = len(x)
        return LinearTrace(np.array(x, dtype=float, copy=True),
                           _array(np.asarray(points, dtype=object).reshape(n, 3)),
                           _array(np.asarray(poses, dtype=object).reshape(n, 6)),
                           _array(np.asarray(maps, dtype=object).reshape(n, 6, 12)))

    def _section_response(self, basic_force, stations):
        resultants, strains = [], []
        for t in stations:
            resultant = _multiply(
                _interpolation(self.length, self.length*Fraction(float(t))),
                basic_force)
            resultants.append(resultant)
            strains.append(_multiply(self._compliance, resultant))
        return _array(resultants), _array(strains)

    def recover_sections(self, displacement, *, stations=(0.0, 0.5, 1.0)):
        """Recover only requested local resultants and strains from a state."""
        try:
            u = np.asarray(displacement, dtype=float)
            x = np.asarray(stations, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("finite local coordinates and stations required") from exc
        if u.shape != (12,) or not np.isfinite(u).all():
            raise ValueError("twelve finite local coordinates required")
        if x.ndim != 1 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("stations must be finite normalized coordinates in [0,1]")
        basic = _multiply(self._B, [Fraction(float(v)) for v in u])
        q = self._basic_action(basic)
        resultants, strains = self._section_response(q, x)
        return LocalSectionRecovery(
            resultants, strains, np.array(x, dtype=float, copy=True))

    def recover_equilibrated_sections(self, displacement, *, uniform_wrench=(0.,)*6,
                                      point_actions=(), stations=(0., .5, 1.)):
        """Reconstruct equilibrium under force/couple per reference length.

        Point actions are (normalized interior station, external local wrench
        about the reference axis). Endpoint actions remain nodal tractions.
        Existing trace/element equations and nodal recovery are unchanged.
        """
        try:
            u = np.asarray(displacement, dtype=float)
            w = np.asarray(uniform_wrench, dtype=float)
            x = np.asarray(stations, dtype=float)
            submitted = tuple(point_actions)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("finite local state and physical section actions required") from exc
        if (u.shape != (12,) or w.shape != (6,) or x.ndim != 1 or
                not np.isfinite(u).all() or not np.isfinite(w).all() or
                not np.isfinite(x).all() or np.any((x < 0.) | (x > 1.))):
            raise ValueError("finite local state, wrench and normalized stations required")
        actions = []
        for action in submitted:
            try:
                station, wrench = action
                values = np.asarray(wrench, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("interior physical point action required") from exc
            if (isinstance(station, (bool, np.bool_)) or not isinstance(station, (int, float)) or
                    not math.isfinite(station) or not 0. < station < 1. or
                    values.shape != (6,) or not np.isfinite(values).all()):
                raise ValueError("finite strictly interior physical point action required")
            actions.append((Fraction(float(station)), [Fraction(float(v)) for v in values]))
        nodal = [Fraction(float(v)) for v in u]
        distributed = [Fraction(float(v)) for v in w]
        basic = self._basic_action(_multiply(self._B, nodal))
        internal = _multiply(_transpose(self._B), basic)
        # Integrate the existing polynomial trace exactly. No new stiffness,
        # internal coordinate, numerical quadrature or dense global operation.
        equivalent = _multiply(self._uniform_load_map, distributed)
        for station, wrench in actions:
            for column in range(12):
                equivalent[column] += sum(wrench[row]*sum(
                    coefficient*station**degree for degree, coefficient in
                    enumerate(self._trace_polynomials[row][column])) for row in range(6))
        tractions = [a-b for a,b in zip(internal, equivalent)]
        left = [-v for v in tractions[:6]]

        def cross_axis(vector):
            return (Fraction(0), -vector[2], vector[1])

        def section(station, right):
            distance = self.length*station
            n = [left[i]-distance*distributed[i] for i in range(3)]
            hn, wp = cross_axis(left), cross_axis(distributed)
            m = [left[i+3]-distance*hn[i]-distance*distributed[i+3]+distance**2*wp[i]/2
                 for i in range(3)]
            for at, wrench in actions:
                if station > at or (right and station == at):
                    lever = self.length*(station-at)
                    turn = cross_axis(wrench)
                    n = [n[i]-wrench[i] for i in range(3)]
                    m = [m[i]-wrench[i+3]+lever*turn[i] for i in range(3)]
            return n+m

        # Exact stored-input work maps must give force and moment closure.
        if section(Fraction(1), True) != tractions[6:]:
            raise RepresentabilityError("equilibrated endpoint traction closure failed")
        low, high, strain_low, strain_high = [], [], [], []
        for value in x:
            station = Fraction(float(value))
            a, b = section(station, False), section(station, True)
            low.append(a); high.append(b)
            strain_low.append(_multiply(self._compliance, a))
            strain_high.append(_multiply(self._compliance, b))
        shape = (len(x), 6)
        return EquilibratedSectionRecovery(np.array(x, copy=True),
            _array(low).reshape(shape), _array(high).reshape(shape),
            _array(strain_low).reshape(shape), _array(strain_high).reshape(shape), _array(tractions))

    def candidate_constant_axial_geometric_stiffness(self, axial_force):
        """Private centerline-stretch Hessian for constant conservative N.

        Positive ``axial_force`` denotes tension. This is a member-only
        candidate, without attachment, load or support curvature.
        """
        if (type(axial_force) not in (int, float)
                or not math.isfinite(axial_force)):
            raise ValueError("finite constant physical axial force required")
        force = Fraction(float(axial_force))
        matrix = [[Fraction(0) for _ in range(12)] for _ in range(12)]
        for position_row in (1, 2):
            derivatives = [
                tuple(degree*poly[degree]/self.length for degree in range(1, 4))
                for poly in self._trace_polynomials[position_row]
            ]
            for i in range(12):
                for j in range(i, 12):
                    integral = sum(
                        a*b/Fraction(p+q+1)
                        for p, a in enumerate(derivatives[i])
                        for q, b in enumerate(derivatives[j]))
                    matrix[i][j] += force*self.length*integral
        for i in range(12):
            for j in range(i):
                matrix[i][j] = matrix[j][i]
        return _array(matrix)

    def evaluate(self, displacement, *, stations=(0.0, 0.5, 1.0)):
        try:
            u = np.asarray(displacement, dtype=float)
            x = np.asarray(stations, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("finite local coordinates and stations required") from exc
        if u.shape != (12,) or not np.isfinite(u).all():
            raise ValueError("twelve finite local coordinates required")
        if x.ndim != 1 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("stations must be finite normalized coordinates in [0,1]")
        basic = _multiply(self._B, [Fraction(float(v)) for v in u])
        q = self._basic_action(basic)
        force = [sum(self._B[j][i]*q[j] for j in range(6)) for i in range(12)]
        resultants, strains = self._section_response(q, x)
        return LocalResponse(
            _float(sum(b*v for b, v in zip(basic, q))/2), _array(force),
            np.array(self._tangent, dtype=float, copy=True),
            _array(q), resultants, strains, np.array(x, dtype=float, copy=True),
        )
