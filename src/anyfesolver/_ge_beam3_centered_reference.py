"""Private stable Q2 reference evaluator; not yet a selectable beam formulation.

The curve and shortest-transport/linear-roll frame policy are unchanged. This
successor evaluates r = X_mid + xi*a + xi**2*b/2, r' = a + xi*b and the half-cell
lift = t*(t-1)*b/2. It avoids multiplying large common translations by shape
derivatives or subtracting large interpolated positions to recover a small lift.
"""

from fractions import Fraction
from dataclasses import dataclass
import math

import numpy as np

from anyfesolver import ge_beam3_curved_reference as source
from anyfesolver._ge_beam3_deterministic_reduction import deterministic_dot, deterministic_norm


EVALUATION_ID = 'CENTERED_Q2_TWO_COMPONENT_COEFFICIENT_ANALYTIC_LIFT_V1'
SCHEMA = 'GE_BEAM3_CENTERED_Q2_REFERENCE_GEOMETRY_SCHEMA_V2'


@dataclass(frozen=True)
class CenteredReferenceStation:
    xi: float
    position: np.ndarray
    position_low: np.ndarray
    derivative: np.ndarray
    jacobian: float
    frame: np.ndarray
    evaluation_id: str = EVALUATION_ID

    def __post_init__(self):
        for name in ('position','position_low','derivative','frame'):
            object.__setattr__(self,name,source._readonly(getattr(self,name)))


def _split_exact(value):
    """Two rounded components of an exact rational, not unlimited precision."""
    try:
        high = float(value)
        low = float(value-Fraction(high))
    except (OverflowError, ValueError) as error:
        raise source.GeBeam3CurvedGeometryError('reference coefficient expansion overflow') from error
    if not math.isfinite(high) or not math.isfinite(low):
        raise source.GeBeam3CurvedGeometryError('nonfinite reference coefficient expansion')
    return (0. if high == 0. else high, 0. if low == 0. else low)


def _sum_pair(terms):
    values = tuple(float(value) for value in terms)
    try:
        high = math.fsum(values)
        low = math.fsum((*values, -high))
    except (OverflowError, ValueError) as error:
        raise source.GeBeam3CurvedGeometryError('reference evaluation overflow') from error
    if not math.isfinite(high) or not math.isfinite(low):
        raise source.GeBeam3CurvedGeometryError('nonfinite reference evaluation')
    return (0. if high == 0. else high, 0. if low == 0. else low)


class CenteredCurvedBeam3ReferenceGeometry(source.CurvedBeam3ReferenceGeometry):
    """Stable evaluation of supplied binary64 geometry, not recovery of lost input.

    Exact rational arithmetic is confined to coefficient construction. Runtime
    evaluations use short compensated sums. Frame interpolation and its analytic
    derivatives reuse the preserved source policy through the overridden tangent
    methods. Consumers must explicitly adopt this type and half_cell_lift; an
    older consumer that clones the base class does not inherit this correction.
    """

    __slots__ = ('_coefficient_high', '_coefficient_low')

    def __init__(self, coordinates, nodal_triads, *,
                 regularity_relative_tolerance=source.DEFAULT_REGULARITY_RELATIVE_TOLERANCE,
                 rotation_tolerance=source.DEFAULT_ROTATION_TOLERANCE,
                 frame_tolerance=source.DEFAULT_FRAME_TOLERANCE):
        regularity = source._positive_tolerance(regularity_relative_tolerance, name='regularity_relative_tolerance')
        rotation = source._positive_tolerance(rotation_tolerance, name='rotation_tolerance')
        frame = source._positive_tolerance(frame_tolerance, name='frame_tolerance')
        if regularity < source.DEFAULT_REGULARITY_RELATIVE_TOLERANCE:
            raise source.GeBeam3CurvedReferenceError('regularity threshold cannot be weakened')
        if rotation > source.DEFAULT_ROTATION_TOLERANCE or frame > source.DEFAULT_FRAME_TOLERANCE:
            raise source.GeBeam3CurvedReferenceError('frame/rotation tolerances cannot be weakened')
        self._coordinates = source._readonly(source._finite_array(coordinates,(3,3),name='coordinates'))
        self._nodal_triads = source._readonly(source._finite_array(nodal_triads,(3,3,3),name='nodal_triads'))
        high = np.empty((2,3)); low = np.empty((2,3))
        for i in range(3):
            left, mid, right = (Fraction(float(value)) for value in self._coordinates[:,i])
            high[0,i], low[0,i] = _split_exact((right-left)/2)
            high[1,i], low[1,i] = _split_exact(left-2*mid+right)
        self._coefficient_high = source._readonly(high); self._coefficient_low = source._readonly(low)
        self._regularity_relative_tolerance = regularity
        self._rotation_tolerance = rotation; self._frame_tolerance = frame
        self._regularity = self._make_regularity()
        self._validate_frames()
        self._half_frame_data = tuple(source._half_frame_data(self._coordinates, self._nodal_triads,
            cell, rotation_tolerance=rotation) for cell in (0,1))

    @property
    def evaluation_id(self):
        return EVALUATION_ID

    def _coefficient(self, index):
        return np.array([_sum_pair((a,b))[0] for a,b in zip(
            self._coefficient_high[index],self._coefficient_low[index])])

    def derivative(self, xi):
        value = source._parameter(xi)
        a, b = self._coefficient_high; al, bl = self._coefficient_low
        return np.array([_sum_pair((a[i],al[i],value*b[i],value*bl[i]))[0] for i in range(3)])

    def tangent(self, xi):
        derivative = self.derivative(xi)
        return derivative/deterministic_norm(derivative)

    def tangent_derivative(self, xi):
        value = source._parameter(xi); derivative = self.derivative(value)
        jacobian = float(deterministic_norm(derivative)); tangent = derivative/jacobian
        curvature = self._coefficient(1)
        return (curvature-tangent*deterministic_dot(tangent,curvature))/jacobian

    def position_pair(self, xi):
        value = source._parameter(xi)
        if value in (-1.,0.,1.):
            return source._readonly(self._coordinates[int(value)+1]), source._readonly(np.zeros(3))
        a, b = self._coefficient_high; al, bl = self._coefficient_low
        square = .5*value*value
        pairs = [_sum_pair((self._coordinates[1,i],value*a[i],value*al[i],square*b[i],square*bl[i]))
                 for i in range(3)]
        high, low = np.array(pairs).T
        return source._readonly(high), source._readonly(low)

    def position(self, xi):
        """Rounded display/high component; position_pair retains the low part."""
        return self.position_pair(xi)[0].copy()

    def station(self, xi):
        value = source._parameter(xi); high, low = self.position_pair(value)
        derivative = self.derivative(value)
        return CenteredReferenceStation(value,high,low,derivative,float(np.linalg.norm(derivative)),self.frame(value))

    def half_cell_lift(self, cell, weight):
        if type(cell) is not int or cell not in (0,1):
            raise source.GeBeam3CurvedReferenceError('half-cell must be exact integer zero or one')
        t = source._finite_scalar(weight,name='half-cell weight')
        if not 0. <= t <= 1.:
            raise source.GeBeam3CurvedReferenceError('half-cell weight must lie in [0,1]')
        factor = .5*t*(t-1.)
        return np.array([_sum_pair((factor*a,factor*b))[0] for a,b in zip(
            self._coefficient_high[1],self._coefficient_low[1])])

    def _make_regularity(self):
        a = self._coefficient(0); b = self._coefficient(1)
        square = float(b@b)
        if not np.isfinite(square):
            raise source.GeBeam3CurvedGeometryError('reference tangent coefficient norm overflow')
        minimizer = 0. if square == 0. else float(np.clip(-float(a@b)/square,-1.,1.))
        if not math.isfinite(minimizer):
            raise source.GeBeam3CurvedGeometryError('nonfinite reference minimizer')
        minimum = self.derivative(minimizer); minimum_squared = float(minimum@minimum)
        lengths = []
        for left,right in ((0,1),(1,2),(0,2)):
            chord = [_sum_pair((self._coordinates[right,i],-self._coordinates[left,i]))[0] for i in range(3)]
            lengths.append(math.hypot(*chord))
        scale = max(lengths); threshold = self._regularity_relative_tolerance*scale
        if (not math.isfinite(scale) or scale <= 0. or not math.isfinite(minimum_squared)
                or minimum_squared <= threshold*threshold):
            raise source.GeBeam3CurvedGeometryError('quadratic tangent is below the frozen relative regularity threshold')
        return source.CurvedBeam3RegularityCertificate(tuple(a),tuple(b),0. if minimizer == 0. else minimizer,
            minimum_squared,scale,threshold)

    def _validate_frames(self):
        for index, xi in enumerate((-1.,0.,1.)):
            triad = self._nodal_triads[index]
            if (np.linalg.norm(triad.T@triad-np.eye(3),np.inf) > self._rotation_tolerance
                    or abs(np.linalg.det(triad)-1.) > self._rotation_tolerance):
                raise source.GeBeam3CurvedFrameError('nodal triads must be proper SO(3) frames')
            if np.linalg.norm(triad[:,0]-self.tangent(xi)) > self._rotation_tolerance:
                raise source.GeBeam3CurvedFrameError('nodal frame axis must match the oriented reference tangent')

    def canonical_data(self):
        value = super().canonical_data()
        value.update(schema=SCHEMA, reference_evaluation=EVALUATION_ID,
            affine_coefficients_high_binary64=source._binary64_array(self._coefficient_high),
            affine_coefficients_low_binary64=source._binary64_array(self._coefficient_low))
        return value
