"""Bounded normal-potential integration on a straight retained ROUND patch.

Axial integration is analytic on every angular active branch. Angular Gauss
8/16 differences control energy, force and tangent estimates independently.
Those estimates are not rigorous error bounds or engineering acceptance.
Reference area is fixed; no friction, history transfer or swept admission is
introduced here.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import acos, atan2, cos, fsum, hypot, isfinite, pi, sin

import numpy as np
from numpy.polynomial.legendre import leggauss

from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_mixed_ad import (
    Jet2, constant_matrix, dot, matmul, so3_exp, sqrt,
)
from .native_normal_angles import PRECISION, PreciseNormalAngles
from .native_retained_station import _readonly
from .native_surface_extrema import straight_patch_s3_gap
from .native_surface_quadrature import NativeRoundSurfaceRule
from .native_surface_s3_friction import NativeS3MaterialConfiguration
from .s3_friction_point import S3SkinMaterialPoint


@dataclass(frozen=True)
class StraightNormalIntegrationSettings:
    energy_absolute_tolerance: float
    force_absolute_tolerance: float
    tangent_absolute_tolerance: float
    max_intervals: int
    characteristic_length_m: float

    def __post_init__(self):
        for name in ('energy_absolute_tolerance', 'force_absolute_tolerance',
                     'tangent_absolute_tolerance','characteristic_length_m'):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.:
                raise ValueError('normal integration tolerances must be finite and positive')
            object.__setattr__(self, name, value)
        if type(self.max_intervals) is not int or self.max_intervals < 1:
            raise ValueError('normal integration requires a finite positive interval cap')


@dataclass(frozen=True)
class StraightNormalIntegrationDiagnostics:
    intervals: int
    accepted_intervals: int
    integrand_evaluations: int
    initial_intervals: int
    estimated_energy_error: float
    estimated_force_error: float
    estimated_tangent_error: float
    estimator: str = 'GAUSS_8_16_ABSOLUTE_DIFFERENCE'


@dataclass(frozen=True)
class StraightNormalIntegrationResult:
    residual: np.ndarray
    tangent: np.ndarray
    normal_storage_j: float
    diagnostics: StraightNormalIntegrationDiagnostics

    def spatial_tangent(self):
        return self.tangent


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]]


def _breakpoints(left_offset, right_offset, a, b, lower, upper):
    """Include every endpoint-gap zero and sinusoidal extremum in the arc."""
    points = {float(lower), float(upper)}
    amplitude = hypot(a, b)
    if amplitude == 0.:
        return sorted(points)
    phase = atan2(b, a)
    for k in range(-2, 5):
        angle = phase+k*pi
        if lower < angle < upper:
            points.add(angle)
    for offset in (left_offset, right_offset):
        ratio = -offset/amplitude
        if -1. <= ratio <= 1.:
            root = acos(ratio)
            for sign in (-1., 1.):
                for k in range(-2, 4):
                    angle = phase+sign*root+2*k*pi
                    if lower < angle < upper:
                        points.add(angle)
    return sorted(points)


def _cell_data(rule, target, configuration, corners, cell, left, right, stiffness,
               *, compensated_projection=False):
    """Eighteen local chart coordinates; no full-system AD arrays."""
    size = 18
    q = [Jet2.variable(0., i, size) for i in range(size)]
    edges = [[Jet2.constant(corners[a, i]-corners[0, i], size)
              +q[9+3*a+i]-q[9+i] for i in range(3)] for a in (1, 2)]
    raw_normal = _cross(*edges)
    length = sqrt(dot(raw_normal, raw_normal))
    normal = [v/length for v in raw_normal]
    count = 42+target.adapter.work_map.shape[1]
    mapping = np.zeros((18, count))
    mapping[:3, 6*cell:6*cell+3] = np.eye(3)
    mapping[3:6, 6*(cell+1):6*(cell+1)+3] = np.eye(3)
    mapping[6:9, 18+3*cell:21+3*cell] = np.eye(3)
    mapping[9:, 42:] = target.adapter.work_map.toarray()
    return _material_cell_normal_data(rule, configuration, cell, left, right,
        stiffness, q[:9], normal, corners[0], q[9:12], mapping,
        compensated_projection=compensated_projection)


def _material_cell_normal_data(rule, configuration, cell, left, right,
                               stiffness, q, normal, origin, origin_variations, mapping,
                               *, compensated_projection=False):
    """Shared physical section map; callers supply the target's work columns."""
    size = q[0].gradient.size
    reference = rule.samples[0].station.reference
    frame = configuration.cell_rotations[cell] @ reference.nodal_triads[0]
    rotated = matmul(so3_exp(q[6:9]), constant_matrix(frame, size))
    radius = rule.definition['radius']
    cosine = radius*dot(normal, [row[1] for row in rotated])
    sine = radius*dot(normal, [row[2] for row in rotated])
    t = left+1. if cell == 0 else left
    base = []
    chord = []
    for i in range(3):
        offset = fsum(((1-t)*float(configuration.positions[cell, i]),
                       (1-t)*float(configuration.position_low[cell, i]),
                       t*float(configuration.positions[cell+1, i]),
                       t*float(configuration.position_low[cell+1, i]),
                       -float(origin[i])))
        difference = fsum((float(configuration.positions[cell+1, i]),
                           float(configuration.position_low[cell+1, i]),
                           -float(configuration.positions[cell, i]),
                           -float(configuration.position_low[cell, i])))
        base.append(offset+(1-t)*q[i]+t*q[3+i]-origin_variations[i])
        chord.append(difference+q[3+i]-q[i])
    offset = dot(normal, base)
    slope = dot(normal, chord)
    width = right-left
    reference_measure = radius*float(np.linalg.norm(
        reference.coordinates[cell+1]-reference.coordinates[cell]))
    if not isfinite(reference_measure) or reference_measure <= 0.:
        raise ValueError('normal integration requires positive finite reference area measure')
    alo, ahi = rule.definition['angle_range']
    precise = (PreciseNormalAngles(offset.value, slope.value, width, cosine.value, sine.value)
               if compensated_projection else None)
    if compensated_projection:
        # This development profile excludes unresolved angular grazing. A
        # rounded acos ratio can omit a tiny active pocket which both Gauss
        # rules miss. Reject ambiguous endpoints/extrema before quadrature;
        # never interpret a zero estimator as proof of an inactive interval.
        phase=atan2(sine.value,cosine.value)
        critical=[alo,ahi]+[phase+k*pi for k in range(-2,5) if alo<phase+k*pi<ahi]
        resolution=32.*np.finfo(float).eps*(abs(offset.value)+abs(width*slope.value)
                                             +abs(cosine.value)+abs(sine.value))
        for angle in critical:
            for gap in precise.endpoint_values(angle):
                if abs(float(gap))<=resolution:
                    raise ValueError('material plane angular endpoint/extremum gap unresolved; grazing excluded')

    def integrand(angle):
        first = offset+cos(angle)*cosine+sin(angle)*sine
        last = first+width*slope
        if compensated_projection:
            # Scalar values and partition roots use the same circular field.
            # Jet2 differentiates that field on the current smooth branch;
            # legacy uncompensated paths keep their numerical convention.
            first_value, last_value = precise.endpoint_values(angle)
            first=Jet2(first_value,first.gradient,first.hessian)
            last=Jet2(last_value,last.gradient,last.hessian)
        if first.value >= 0. and last.value >= 0.:
            energy = Jet2.constant(0., size)
        elif first.value <= 0. and last.value <= 0.:
            energy = (width*stiffness/6.)*(first*first+first*last+last*last)
        else:
            negative = -first if first.value < 0. else -last
            magnitude = slope if slope.value > 0. else -slope
            energy = stiffness*(negative*negative*negative)/(6.*magnitude)
        energy = reference_measure*energy
        if (not isfinite(energy.value) or not np.all(np.isfinite(energy.gradient))
                or not np.all(np.isfinite(energy.hessian))):
            raise ValueError('nonfinite straight normal integrand')
        return energy

    breaks = (precise.breakpoints(alo, ahi) if compensated_projection else
              _breakpoints(offset.value, offset.value+width*slope.value,
                           cosine.value, sine.value, alo, ahi))
    return integrand, mapping, breaks


def integrate_straight_normal(rule, target, configuration,
                              normal_stiffness_Pa_per_m, settings,
                              *, compensated_projection=False):
    """Integrate reference-area penalty work and its complete spatial tangent.

    Each accepted angular interval consumes its width-proportional share of
    all three absolute error tolerances (J, N, N/m). Rotational residual/tangent
    entries are scaled using the explicitly declared characteristic length.
    Partition locations are not differentiated: potential and first variations
    agree across internal active boundaries, cancelling their boundary terms.
    """
    if (type(rule) is not NativeRoundSurfaceRule
            or type(target) is not S3SkinMaterialPoint
            or type(configuration) is not NativeS3MaterialConfiguration
            or type(settings) is not StraightNormalIntegrationSettings):
        raise TypeError('exact straight normal rule, target, configuration and settings required')
    stiffness = float(normal_stiffness_Pa_per_m)
    if not isfinite(stiffness) or stiffness <= 0.:
        raise ValueError('normal penalty stiffness must be finite and positive')
    shape=target.adapter.work_map.shape
    extended_columns=(compensated_projection and shape[0]==9 and shape[1]>=18 and shape[1]%6==0)
    if target.adapter.side != 'top' or (shape != (9,18) and not extended_columns):
        raise ValueError('straight normal integration requires one 18-DOF top S3 skin')
    # Shared exact-reference, current orientation and continuous target ownership
    # admission. The returned gap is not replaced by angular sampling.
    straight_patch_s3_gap(rule, target, configuration)
    corners = target.adapter.current_positions(configuration.shell_displacement)
    lo, hi = rule.definition['xi_range']
    cells = [(c, l, r) for c, l, r in
             ((0, max(lo, -1.), min(hi, 0.)), (1, max(lo, 0.), min(hi, 1.))) if r > l]
    data = [_cell_data(rule, target, configuration, corners, *cell, stiffness,
                       compensated_projection=compensated_projection) for cell in cells]
    return _integrate_material_normal_data(data,42+target.adapter.work_map.shape[1],settings)


def _integrate_material_normal_data(data, count, settings):
    if any(isinstance(item[2][0], Decimal) for item in data):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            return _integrate_material_normal_data_impl(data, count, settings)
    return _integrate_material_normal_data_impl(data, count, settings)


def _integrate_material_normal_data_impl(data, count, settings):
    """Bounded shared angular integration and one spatial-chart conversion."""
    initial_intervals = sum(len(item[2])-1 for item in data)
    def exhausted(visited=0,accepted=0,evaluations=0,last_error=None,last_interval=None):
        error=MemoryError('straight normal integration interval budget exhausted')
        error.integration_diagnostics=dict(initial_intervals=initial_intervals,
            maximum_intervals=settings.max_intervals,visited_intervals=visited,
            accepted_intervals=accepted,integrand_evaluations=evaluations,
            last_estimated_error=last_error,last_interval=last_interval)
        return error
    if initial_intervals > settings.max_intervals:raise exhausted()
    gauss = (leggauss(8), leggauss(16))
    total_width = sum(item[2][-1]-item[2][0] for item in data)
    tolerances = np.array((settings.energy_absolute_tolerance,
                           settings.force_absolute_tolerance,
                           settings.tangent_absolute_tolerance))
    residual = np.zeros(count)
    hessian = np.zeros((count, count))
    energies = []
    errors = []
    visited = accepted = evaluations = 0
    last_error=last_interval=None
    scale=np.ones(count)
    for first in (3,9,15,18,21,*range(45,count,6)):
        if first+3 <= count:
            scale[first:first+3]=1./settings.characteristic_length_m
    for integrand, mapping, breaks in data:
        size = mapping.shape[0]
        local = Jet2.constant(0., size)
        pending = list(zip(breaks[:-1], breaks[1:]))
        while pending:
            if visited >= settings.max_intervals:
                raise exhausted(visited,accepted,evaluations,last_error,last_interval)
            left, right = pending.pop()
            half = (right-left)/2
            middle = left+half
            if middle == left or middle == right:
                raise ValueError('straight normal integration angular resolution exhausted')
            if float(half) == 0.:
                raise ValueError('straight normal integration angular measure unresolved')
            estimates = []
            for points, weights in gauss:
                estimate = Jet2.constant(0., size)
                for point, weight in zip(points, weights):
                    factor = float(half)*float(weight)
                    if factor == 0.:
                        raise ValueError('straight normal integration quadrature measure unresolved')
                    angle = (middle+half*Decimal(float(point)) if isinstance(half, Decimal)
                             else middle+half*float(point))
                    if not left < angle < right:
                        raise ValueError('straight normal integration angular resolution exhausted')
                    estimate = estimate+factor*integrand(angle)
                    evaluations += 1
                estimates.append(estimate)
            visited += 1
            difference = estimates[1]-estimates[0]
            mapped_force=mapping.T@difference.gradient
            mapped_tangent=mapping.T@difference.hessian@mapping
            mapped_tangent[:42,:42]=spatial_jacobian(mapped_force[:42],mapped_tangent[:42,:42])
            error = np.array((abs(difference.value),float(np.max(np.abs(scale*mapped_force))),
                              float(np.max(np.abs(scale[:,None]*mapped_tangent*scale[None,:])))))
            if not np.all(np.isfinite(error)):
                raise ValueError('nonfinite straight normal integration error estimate')
            last_error=error.tolist();last_interval=(float(left),float(right))
            if np.all(error <= tolerances*float((right-left)/total_width)):
                local = local+estimates[1]
                errors.append(error)
                accepted += 1
            else:
                pending.extend(((left, middle), (middle, right)))
        energies.append(local.value)
        residual += mapping.T @ local.gradient
        hessian += mapping.T @ local.hessian @ mapping
    # Exactly one chart-to-spatial conversion after complete native assembly.
    hessian[:42, :42] = spatial_jacobian(residual[:42], hessian[:42, :42])
    energy = fsum(energies)
    error_sum = tuple(fsum(float(e[i]) for e in errors) for i in range(3))
    if (not isfinite(energy) or not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(hessian))):
        raise ValueError('nonfinite assembled straight normal integration')
    diagnostics = StraightNormalIntegrationDiagnostics(
        visited, accepted, evaluations, initial_intervals, *error_sum)
    return StraightNormalIntegrationResult(_readonly(residual), _readonly(hessian), energy, diagnostics)
