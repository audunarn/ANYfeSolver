"""Private physical-fibre incremental potential; not a beam material route.

Physical fibre strain is eps_x + z*kappa_y - y*kappa_z. Histories and
responses are immutable values; the section never commits a trial implicitly.
Decimal evaluation is bounded numerical arithmetic, not certification.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal as D, localcontext
import hashlib
import json
from math import isfinite
from numbers import Real
from time import monotonic

import numpy as np

from ._ge_beam3_station_resultant_cell import cholesky, history_row, pair
from ._ge_beam3_effective_deadline import effective_deadline_seconds


POLICY = 'GE_BEAM3_PHYSICAL_FIBRE_INCREMENTAL_POTENTIAL_V1'
MEASURE = 'NATIVE_AXIAL_FIBRE_STRAIN_REFERENCE_AREA_WORK_CONJUGATE_STRESS'
PARAMETER_CAPTURE = 'EXPLICIT_REINTERPRETATION_AS_NATIVE_CONSTITUTIVE_PARAMETERS'


def canonical(value):
    def encode(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if hasattr(item, '__dataclass_fields__'): return asdict(item)
        raise TypeError('unsupported fibre evidence value')
    return (json.dumps(value, default=encode, sort_keys=True, separators=(',', ':'),
                       allow_nan=False)+'\n').encode('ascii')


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('finite real fibre input required')
    result = float(value)
    if not isfinite(result): raise ValueError('finite real fibre input required')
    return result


def _owned(value, shape):
    original = np.asarray(value, dtype=object)
    if original.shape != shape: raise ValueError('fibre array shape mismatch')
    data = np.array([_number(x) for x in original.ravel()], dtype=float).reshape(shape)
    return np.frombuffer(data.tobytes(), dtype=float).reshape(shape)


def _scalar_pair(value):
    high, low = pair([value])
    return (high[0], low[0])


def _array_pair(values, shape):
    high, low = pair(values)
    return _owned(np.array(high).reshape(shape), shape), _owned(np.array(low).reshape(shape), shape)


@dataclass(frozen=True)
class FlowCurve:
    """Positive continuous nondecreasing affine segments and explicit tail."""
    plastic_strain: tuple
    flow_stress: tuple
    tail_slope: float

    def __post_init__(self):
        x = tuple(_number(v) for v in self.plastic_strain)
        y = tuple(_number(v) for v in self.flow_stress)
        tail = _number(self.tail_slope)
        if (not 1 <= len(x) <= 1024 or len(x) != len(y) or x[0] != 0.
                or any(b <= a for a, b in zip(x, x[1:])) or any(v <= 0 for v in y)
                or any(b < a for a, b in zip(y, y[1:])) or tail < 0):
            raise ValueError('positive monotone flow curve and nonnegative tail required')
        object.__setattr__(self, 'plastic_strain', x)
        object.__setattr__(self, 'flow_stress', y)
        object.__setattr__(self, 'tail_slope', tail)

    @classmethod
    def linear(cls, yield_stress, hardening):
        return cls((0.,), (yield_stress,), hardening)

    @classmethod
    def from_anymaterial(cls, curve, *, parameter_interpretation):
        # ANYmaterial documents true stress/true plastic strain. Copying its
        # numbers is NOT a true-to-nominal finite-strain material conversion.
        # This private entry requires explicit re-authoring as native law data.
        if parameter_interpretation != PARAMETER_CAPTURE:
            raise ValueError('explicit native constitutive parameter reinterpretation required')
        from anymaterial.curves import LinearHardeningCurve, PiecewiseLinearCurve
        if type(curve) is LinearHardeningCurve:
            return cls.linear(curve.sigma_yield, curve.hardening_modulus_value)
        if type(curve) is PiecewiseLinearCurve:
            return cls(tuple(curve.plastic_strain), tuple(curve.flow_stress_values), 0.)
        raise ValueError('no declared energy-integral adapter for this material curve')

    def _segments(self):
        x = tuple(map(D.from_float, self.plastic_strain))
        y = tuple(map(D.from_float, self.flow_stress))
        slopes = tuple((b-a)/(v-u) for a, b, u, v in zip(y, y[1:], x, x[1:]))
        return x, y, slopes+(D.from_float(self.tail_slope),)

    def _at(self, p):
        if p < 0: raise ValueError('negative accumulated plastic strain')
        x, y, slopes = self._segments()
        index = sum(k <= p for k in x)-1
        return y[index]+slopes[index]*(p-x[index]), slopes[index], p in x[1:]

    def _integral(self, left, right):
        if left < 0 or right < left: raise ValueError('ordered plastic integral limits required')
        x, y, slopes = self._segments()
        total = D(0)
        for i in range(len(x)):
            a = max(left, x[i]); b = min(right, x[i+1]) if i+1 < len(x) else right
            if b > a:
                total += (b-a)*(y[i]+slopes[i]*((a-x[i])+(b-x[i]))/2)
        return total

    def _return(self, elastic, strain, z0, p0):
        x, y, slopes = self._segments()
        first = sum(k <= p0 for k in x)-1
        drive = elastic*(strain-z0); radius = y[first]+slopes[first]*(p0-x[first])
        if abs(drive) <= radius:
            return D(0), elastic, ('YIELD_BOUNDARY' if abs(drive) == radius else 'ELASTIC')
        # One linear traversal; do not rebuild the full curve at each interval.
        for index in range(first, len(x)):
            start = max(p0, x[index]); slope = slopes[index]
            force = y[index]+slope*(start-x[index])
            u0 = start-p0
            u = u0+(abs(drive)-elastic*u0-force)/(elastic+slope)
            if index+1 == len(x) or p0+u < x[index+1]:
                knot = p0+u in x[1:]
                return (u if drive > 0 else -u), elastic*slope/(elastic+slope), (
                    'HARDENING_KNOT' if knot else ('PLASTIC_POSITIVE' if drive > 0 else 'PLASTIC_NEGATIVE'))
        raise ValueError('bounded fibre interval traversal failed')


@dataclass(frozen=True)
class Fibre:
    fibre_id: str
    y: float
    z: float
    area: float
    young: float
    curve: FlowCurve

    def __post_init__(self):
        if (type(self.fibre_id) is not str or not self.fibre_id.isascii()
                or not 1 <= len(self.fibre_id) <= 128 or type(self.curve) is not FlowCurve):
            raise ValueError('explicit fibre ID and energy curve required')
        for name in ('y', 'z', 'area', 'young'):
            object.__setattr__(self, name, _number(getattr(self, name)))
        if self.area <= 0 or self.young <= 0: raise ValueError('positive fibre area and Young modulus required')


@dataclass(frozen=True)
class FibreHistory:
    section_identity: str
    # One normalized [plastic high, plastic low, accumulated high, accumulated low] row per fibre.
    rows: tuple


@dataclass(frozen=True)
class FibreRecovery:
    fibre_id: str
    total_strain: tuple
    elastic_strain: tuple
    stress: tuple
    algorithmic_modulus: tuple
    plastic_increment: tuple
    branch: str


@dataclass(frozen=True)
class FibreResponse:
    origin: FibreHistory
    history: FibreHistory
    strain: np.ndarray
    strain_low: np.ndarray
    resultants: np.ndarray
    resultants_low: np.ndarray
    tangent: np.ndarray
    tangent_low: np.ndarray
    incremental_potential: tuple
    stored_energy: tuple
    dissipation: tuple
    fibres: tuple
    derivative_kind: str
    section_identity: str
    production_qualified: bool = False


class PhysicalFibreSection:
    """Native axial/biaxial fibre plasticity plus declared elastic background.

    The background factor may contain generalized coupling. It is additive;
    this class never guesses or subtracts physical fibre stiffness from it.
    This is not yet an assembled retained-resultant beam adapter.
    """
    __slots__ = ('fibres', 'background_factor', 'identity', '_background', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('physical fibre section is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, fibres, background_factor):
        self.fibres = tuple(fibres)
        if (not 1 <= len(self.fibres) <= 1024 or any(type(f) is not Fibre for f in self.fibres)
                or len({f.fibre_id for f in self.fibres}) != len(self.fibres)):
            raise ValueError('bounded distinct explicit physical fibres required')
        shape = np.shape(background_factor)
        if len(shape) != 2 or shape[1] != 6 or not 1 <= shape[0] <= 64:
            raise ValueError('bounded six-column elastic background factor required')
        self.background_factor = _owned(background_factor, shape)
        with localcontext() as context:
            context.prec = 80
            factor = [list(map(D.from_float, row)) for row in self.background_factor]
            cb = [[sum((row[i]*row[j] for row in factor), D(0)) for j in range(6)] for i in range(6)]
            elastic = [row[:] for row in cb]
            for fibre in self.fibres:
                b = self._map(fibre); weight = D.from_float(fibre.area)*D.from_float(fibre.young)
                for i in range(6):
                    for j in range(6): elastic[i][j] += weight*b[i]*b[j]
            cholesky(elastic)  # No null-mode clipping or stiffness floor.
            self._background = tuple(map(tuple, cb))
        self.identity = hashlib.sha256(canonical(dict(policy=POLICY, measure=MEASURE, fibres=self.fibres,
            background_factor=self.background_factor))).hexdigest()
        self._sealed = True

    @staticmethod
    def _map(fibre):
        return (D(1), D(0), D(0), D(0), D.from_float(fibre.z), -D.from_float(fibre.y))

    def virgin(self):
        return FibreHistory(self.identity, tuple((0., 0., 0., 0.) for _ in self.fibres))

    def _origins(self, origin):
        if (type(origin) is not FibreHistory or origin.section_identity != self.identity
                or type(origin.rows) is not tuple or len(origin.rows) != len(self.fibres)
                or any(type(row) is not tuple for row in origin.rows)):
            raise ValueError('immutable history belonging to this section required')
        return [history_row(row) for row in origin.rows]

    def response(self, strain, origin=None, *, strain_low=None, check=None):
        started = monotonic()
        def guard():
            if check is not None: check()
            if monotonic()-started > effective_deadline_seconds(30.): raise TimeoutError('physical fibre response bound')
        guard()
        strain = _owned(strain, (6,))
        strain_low = _owned(np.zeros(6) if strain_low is None else strain_low, (6,))
        origin = self.virgin() if origin is None else origin
        with localcontext() as context:
            context.prec = 80
            origins = self._origins(origin)
            e = [D.from_float(h)+D.from_float(l) for h, l in zip(strain, strain_low)]
            stress = [sum((c*x for c, x in zip(row, e)), D(0)) for row in self._background]
            tangent = [list(row) for row in self._background]
            stored = sum((x*s for x, s in zip(e, stress)), D(0))/2
            previous_hardening = D(0); dissipation = D(0); rows = []; recovered = []
            for fibre, (z0, p0) in zip(self.fibres, origins):
                guard(); b = self._map(fibre); area = D.from_float(fibre.area); young = D.from_float(fibre.young)
                ef = sum((a*x for a, x in zip(b, e)), D(0))
                delta, et, branch = fibre.curve._return(young, ef, z0, p0)
                z = z0+delta; p = p0+abs(delta); ee = ef-z; sigma = young*ee
                y0 = D.from_float(fibre.curve.flow_stress[0])
                stored += area*(young*ee*ee/2+fibre.curve._integral(D(0), p)-y0*p)
                previous_hardening += area*(fibre.curve._integral(D(0), p0)-y0*p0)
                dissipation += area*y0*abs(delta)
                for i in range(6):
                    stress[i] += area*b[i]*sigma
                    for j in range(6): tangent[i][j] += area*et*b[i]*b[j]
                zh, zl = _scalar_pair(z); ph, pl = _scalar_pair(p)
                rows.append((zh, zl, ph, pl))
                recovered.append(FibreRecovery(fibre.fibre_id, _scalar_pair(ef), _scalar_pair(ee),
                    _scalar_pair(sigma), _scalar_pair(et), _scalar_pair(delta), branch))
            resultants, resultants_low = _array_pair(stress, (6,))
            stiffness, stiffness_low = _array_pair([v for row in tangent for v in row], (6, 6))
            proposed = FibreHistory(self.identity, tuple(rows)); self._origins(proposed)
            kind = ('SEMISMOOTH_BRANCH_SELECTION' if any(f.branch in ('YIELD_BOUNDARY', 'HARDENING_KNOT')
                for f in recovered) else 'CLASSICAL_SMOOTH_BRANCH')
            result = FibreResponse(origin, proposed, strain, strain_low, resultants, resultants_low,
                stiffness, stiffness_low, _scalar_pair(stored-previous_hardening+dissipation),
                _scalar_pair(stored), _scalar_pair(dissipation), tuple(recovered), kind, self.identity)
        guard()
        return result
