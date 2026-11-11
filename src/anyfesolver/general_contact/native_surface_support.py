"""Conservative separation test for an entire straight material panel.

Spatial minima are analytic, not sampled. A uniform temporal Lipschitz bound
encloses every material point between inspected poses. False means contact or
unresolved separation; it does not prove that contact occurred. This is not a
replacement for source-orientation, target-ownership or history admission.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import atan2, cos, fsum, pi, sin

import numpy as np

from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_surface_history import NativeSurfaceHistoryField
from .native_surface_path import material_path_pose


@dataclass
class SupportIntervalBudget:
    """Shared temporal-interval cap across an integration's support queries."""
    maximum_intervals: int
    intervals: int = 0

    def __post_init__(self):
        if type(self.maximum_intervals) is not int or self.maximum_intervals < 1:
            raise ValueError('positive shared support interval cap required')
        if type(self.intervals) is not int or self.intervals != 0:
            raise ValueError('new shared support budget must start at zero')

    def consume(self):
        if self.intervals >= self.maximum_intervals:
            return False
        self.intervals += 1
        return True


def _base(pose, cell, xi, origin):
    t = xi+1. if cell == 0 else xi
    return np.array([fsum(((1-t)*float(pose.positions[cell, i]),
                           (1-t)*float(pose.position_low[cell, i]),
                           t*float(pose.positions[cell+1, i]),
                           t*float(pose.position_low[cell+1, i]),
                           -float(origin[i]))) for i in range(3)])


def _convex_support_origin(pose,cell,xis,corners):
    """Choose fixed bounded plane coordinates, never extrapolated weights.

    Selection accuracy affects tightness only. Any exact-simplex pair defines
    an admissible material origin; edge-relative evaluation supplies its motion.
    """
    edges=np.column_stack((corners[1]-corners[0],corners[2]-corners[0]))
    weights=np.linalg.lstsq(edges,_base(pose,cell,sum(xis)/2,corners[0]),rcond=None)[0]
    if not np.all(np.isfinite(weights)):return (0.,0.)
    weights=np.maximum(weights,0.)
    if float(np.sum(weights))>1.:
        weights=np.nextafter(weights/float(np.sum(weights)),0.)
    first,second=map(float,weights)
    if Fraction(first)+Fraction(second)>1:return (0.,0.)
    return first,second


def _convex_relative(pose,cell,xi,corners,weights):
    alpha=xi+1 if cell==0 else xi
    value=[];scale=1.
    for i in range(3):
        terms=((1-alpha)*float(pose.positions[cell,i]),
               (1-alpha)*float(pose.position_low[cell,i]),
               alpha*float(pose.positions[cell+1,i]),alpha*float(pose.position_low[cell+1,i]),
               -float(corners[0,i]),-weights[0]*float(corners[1,i]-corners[0,i]),
               -weights[1]*float(corners[2,i]-corners[0,i]))
        value.append(fsum(terms));scale=max(scale,sum(abs(v) for v in terms))
    return np.asarray(value),scale


def _convex_velocity(segment,cell,xi,motion,weights):
    alpha=xi+1 if cell==0 else xi
    value=[];scale=1.
    for i in range(3):
        terms=((1-alpha)*float(segment.translations[cell,i]),alpha*float(segment.translations[cell+1,i]),
               -float(motion[0,i]),-weights[0]*float(motion[1,i]-motion[0,i]),
               -weights[1]*float(motion[2,i]-motion[0,i]))
        value.append(fsum(terms));scale=max(scale,sum(abs(v) for v in terms))
    return np.asarray(value),scale


def _minimum(pose, target, reference_frame, radius, cell, xis, angles, *, gap_sign=1.):
    corners = target.adapter.current_positions(pose.shell_displacement)
    normal = np.asarray(target.physical_pose(pose.shell_displacement).material_frame)[:, 2]
    frame = pose.cell_rotations[cell] @ reference_frame
    a = radius*float(normal @ frame[:, 1])
    b = radius*float(normal @ frame[:, 2])
    phase = atan2(b, a)
    candidates = [*angles, *(phase+k*pi for k in range(-2, 5)
                              if angles[0] <= phase+k*pi <= angles[1])]
    values = [fsum((float(normal @ _base(pose, cell, xi, corners[0])),
                    a*cos(angle), b*sin(angle)))
              for xi in xis for angle in candidates]
    scale = max(1., radius, float(np.max(np.abs(pose.positions))),
                float(np.max(np.abs(pose.position_low))), float(np.max(np.abs(corners))))
    return min(gap_sign*v for v in values), 512*np.finfo(float).eps*scale


def _support_path(field,segment_index):
    if segment_index is None:return field.initial,field.segments
    if type(segment_index) is not int or not 0<=segment_index<len(field.segments):
        raise ValueError('support segment index outside captured history')
    pose=field.initial
    for segment in field.segments[:segment_index]:
        pose=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
    return pose,(field.segments[segment_index],)


def panel_open_over_history(field,cell,xi_bounds,angle_bounds,*,maximum_intervals,budget=None,segment_index=None):
    return _panel_signed_gap_over_history(field,cell,xi_bounds,angle_bounds,
        maximum_intervals=maximum_intervals,budget=budget,segment_index=segment_index,gap_sign=1.)


def panel_closed_over_history(field,cell,xi_bounds,angle_bounds,*,maximum_intervals,budget=None,segment_index=None):
    return _panel_signed_gap_over_history(field,cell,xi_bounds,angle_bounds,
        maximum_intervals=maximum_intervals,budget=budget,segment_index=segment_index,gap_sign=-1.)


def _panel_signed_gap_over_history(field,cell,xi_bounds,angle_bounds,*,maximum_intervals,budget,segment_index,gap_sign):
    """True only for whole-panel strictly signed gap over the initial pose and history.

    The interval cap is shared by all segments. Each segment also retains the
    history field's captured per-path cap. An optional budget additionally shares
    that work limit across multiple calls. Target cross-product Bernstein bounds
    retain its orientation/area guard. Unresolved area or temporal bounds return
    False; malformed inputs and encountered invalid target geometry raise.
    """
    if type(field) is not NativeSurfaceHistoryField:
        raise TypeError('exact native surface history field required')
    if type(cell) is not int or cell not in (0, 1):
        raise ValueError('one native half-cell required')
    if type(maximum_intervals) is not int or maximum_intervals < 1:
        raise ValueError('positive aggregate support interval cap required')
    if budget is not None and type(budget) is not SupportIntervalBudget:
        raise TypeError('exact shared support interval budget required')
    if sha(field._definition()) != field.identity:
        raise ValueError('native material history field definition changed')
    xis = np.asarray(xi_bounds, dtype=float)
    angles = np.asarray(angle_bounds, dtype=float)
    rule = field.patch.rule
    for values, patch_bounds, domain in ((xis, rule.definition['xi_range'], (-1.+cell, float(cell))),
                                         (angles, rule.definition['angle_range'], (0., 2*pi))):
        if (values.shape != (2,) or not np.all(np.isfinite(values))
                or not max(domain[0], patch_bounds[0]) <= values[0] < values[1] <= min(domain[1], patch_bounds[1])):
            raise ValueError('support rectangle must lie in one declared material half-cell')
    reference = field.patch.source.reference
    coordinates, frames = reference.coordinates, reference.nodal_triads
    if (any(Fraction(float(coordinates[0, i]))+Fraction(float(coordinates[2, i]))
            != 2*Fraction(float(coordinates[1, i])) for i in range(3))
            or not np.array_equal(frames, np.tile(frames[0], (3, 1, 1)))):
        raise ValueError('material support requires exact straight untwisted reference')
    target = field.patch.target
    radius = float(rule.definition['radius'])
    pose,segments = _support_path(field,segment_index)
    minimum, padding = _minimum(pose, target, frames[0], radius, cell, xis, angles,gap_sign=gap_sign)
    if minimum <= padding:
        return False
    reference_corners = target.adapter.reference_positions
    reference_cross = np.cross(reference_corners[1]-reference_corners[0],
                               reference_corners[2]-reference_corners[0])
    reference_area = float(np.linalg.norm(reference_cross))
    reference_normal = reference_cross/reference_area
    visited = 0
    for segment in segments:
        end = material_path_pose(pose, segment.translations, segment.angular, segment.shell_increment, 1.)
        first = target.adapter.current_positions(pose.shell_displacement)
        last = target.adapter.current_positions(end.shell_displacement)
        # Also validates endpoint target degeneracy and the existing polarity.
        endpoint_minimum, endpoint_padding = _minimum(end, target, frames[0], radius, cell, xis, angles,gap_sign=gap_sign)
        if endpoint_minimum <= endpoint_padding:
            return False
        a0, b0 = first[1]-first[0], first[2]-first[0]
        a1, b1 = last[1]-last[0], last[2]-last[0]
        cross_controls = (np.cross(a0, b0),
                          .5*(np.cross(a0, b1)+np.cross(a1, b0)), np.cross(a1, b1))
        area_lower = min(float(reference_normal @ c) for c in cross_controls)
        area_scale = max(reference_area, max(np.linalg.norm(a0), np.linalg.norm(a1))
                         *max(np.linalg.norm(b0), np.linalg.norm(b1)))
        if area_lower <= 1e-12*area_scale:
            return False
        target_increments = np.asarray(target.adapter.work_map @ segment.shell_increment).reshape(3, 3)
        if (not np.any(segment.angular[cell])
                and np.array_equal(target_increments, np.tile(target_increments[0], (3, 1)))):
            # Every fixed material point has an affine gap. Positive exact
            # whole-panel endpoint minima cover the entire segment, regardless
            # of tangential travel. Charge one interval to both work budgets.
            if visited >= maximum_intervals:
                return False
            if budget is not None and not budget.consume():
                return False
            visited += 1
            pose = end
            padding = endpoint_padding
            continue
        normal_speed = (np.linalg.norm(a1-a0)*max(np.linalg.norm(b0), np.linalg.norm(b1))
                        +np.linalg.norm(b1-b0)*max(np.linalg.norm(a0), np.linalg.norm(a1)))/area_lower
        # Relative affine base is bilinear in (xi,time), hence lies in the
        # convex hull of these four corners. Adding the section radius bounds
        # all material surface points, including angular extrema not sampled.
        relative_radius = max(np.linalg.norm(_base(p, cell, xi, corner))
                              for p, corner in ((pose, first[0]), (end, last[0]))
                              for xi in xis)+radius
        target_motion = target_increments[0]
        velocities = []
        for xi in xis:
            t = xi+1. if cell == 0 else xi
            velocities.append((1-t)*segment.translations[cell]+t*segment.translations[cell+1]-target_motion)
        speed = max(np.linalg.norm(v) for v in velocities)+np.linalg.norm(segment.angular[cell])*radius
        rate = float(speed+normal_speed*relative_radius)
        # A second-derivative bound encloses gap below the chord joining its
        # endpoint values: g(t) >= (1-t)*g(0)+t*g(1)-curvature*t*(1-t)/2.
        # Exact spatial endpoint minima therefore give a whole-panel proof.
        da,db=a1-a0,b1-b0
        normal_acceleration=2*np.linalg.norm(np.cross(da,db))/area_lower+3*normal_speed**2
        curvature=(normal_acceleration*relative_radius+2*normal_speed*speed
                   +np.linalg.norm(segment.angular[cell])**2*radius)
        start_minimum,start_padding=_minimum(pose,target,frames[0],radius,cell,xis,angles,gap_sign=gap_sign)
        if (np.isfinite(curvature) and min(start_minimum,endpoint_minimum)-curvature/8
                >max(start_padding,endpoint_padding)):
            if visited>=maximum_intervals or (budget is not None and not budget.consume()):
                return False
            visited+=1
            pose=end
            padding=endpoint_padding
            continue
        if not np.isfinite(rate):
            raise ValueError('nonfinite whole-panel support rate')
        pending = [(0., 1.)]
        segment_visits = 0
        while pending:
            if visited >= maximum_intervals or segment_visits >= field.maximum_intervals:
                return False
            left, right = pending.pop()
            midpoint = left+(right-left)/2.
            if midpoint == left or midpoint == right:
                return False
            if budget is not None and not budget.consume():
                return False
            current = material_path_pose(pose, segment.translations, segment.angular,
                                         segment.shell_increment, midpoint)
            value, roundoff = _minimum(current, target, frames[0], radius, cell, xis, angles,gap_sign=gap_sign)
            visited += 1
            segment_visits += 1
            if value <= roundoff:
                return False
            if value-rate*(right-left)/2. > max(roundoff, padding, endpoint_padding):
                continue
            pending.extend(((left, midpoint), (midpoint, right)))
        pose = end
        padding = endpoint_padding
    return True


def _sinusoidal_extrema(offsets, a, b, angles):
    """Exact extrema of affine-xi + a*cos(angle)+b*sin(angle)."""
    phase = atan2(b, a)
    candidates = [*angles, *(phase+k*pi for k in range(-2, 5)
                              if angles[0] <= phase+k*pi <= angles[1])]
    values = [fsum((offset, a*cos(angle), b*sin(angle)))
              for offset in offsets for angle in candidates]
    return min(values), max(values)


def history_endpoint_support_certified(field, *, xi_bounds=None, angle_bounds=None, segment_index=None):
    """Sufficient whole-patch proof that contact cannot hide between poses.

    Every point's gap must be affine in time, or have a derivative of one
    rigorously bounded sign on each segment. Different segments may have
    different signs. This does not certify spatial quadrature accuracy or
    resolve stick/slip fronts inside the endpoint-supported contact region.
    """
    if type(field) is not NativeSurfaceHistoryField:
        raise TypeError('exact native surface history field required')
    if sha(field._definition()) != field.identity:
        raise ValueError('native material history field definition changed')
    rule = field.patch.rule
    reference = field.patch.source.reference
    coordinates, frames = reference.coordinates, reference.nodal_triads
    if (any(Fraction(float(coordinates[0, i]))+Fraction(float(coordinates[2, i]))
            != 2*Fraction(float(coordinates[1, i])) for i in range(3))
            or not np.array_equal(frames, np.tile(frames[0], (3, 1, 1)))):
        raise ValueError('material support requires exact straight untwisted reference')
    target = field.patch.target
    radius = float(rule.definition['radius'])
    lo, hi = rule.definition['xi_range']
    angles = rule.definition['angle_range']
    for bounds,domain in ((xi_bounds,(lo,hi)),(angle_bounds,angles)):
        if bounds is not None:
            values=np.asarray(bounds,dtype=float)
            if (values.shape!=(2,) or not np.all(np.isfinite(values))
                    or not domain[0]<=values[0]<values[1]<=domain[1]):
                raise ValueError('endpoint support rectangle outside declared patch')
    if xi_bounds is not None:lo,hi=map(float,xi_bounds)
    if angle_bounds is not None:angles=tuple(map(float,angle_bounds))
    cells = [(c, (l, r)) for c, l, r in
             ((0, max(lo, -1.), min(hi, 0.)), (1, max(lo, 0.), min(hi, 1.))) if r > l]
    reference_corners = target.adapter.reference_positions
    reference_cross = np.cross(reference_corners[1]-reference_corners[0],
                               reference_corners[2]-reference_corners[0])
    reference_area = float(np.linalg.norm(reference_cross))
    reference_normal = reference_cross/reference_area
    pose,segments = _support_path(field,segment_index)
    target.physical_pose(pose.shell_displacement)
    for segment in segments:
        end = material_path_pose(pose, segment.translations, segment.angular, segment.shell_increment, 1.)
        target.physical_pose(end.shell_displacement)
        first = target.adapter.current_positions(pose.shell_displacement)
        last = target.adapter.current_positions(end.shell_displacement)
        a0, b0 = first[1]-first[0], first[2]-first[0]
        a1, b1 = last[1]-last[0], last[2]-last[0]
        c0, c2 = np.cross(a0, b0), np.cross(a1, b1)
        c1 = .5*(np.cross(a0, b1)+np.cross(a1, b0))
        area_lower = min(float(reference_normal @ c) for c in (c0, c1, c2))
        area_scale = max(reference_area, max(np.linalg.norm(a0), np.linalg.norm(a1))
                         *max(np.linalg.norm(b0), np.linalg.norm(b1)))
        if area_lower <= 1e-12*area_scale:
            return False
        target_motion = np.asarray(target.adapter.work_map @ segment.shell_increment).reshape(3, 3)
        if (not np.any(segment.angular)
                and np.array_equal(target_motion, np.tile(target_motion[0], (3, 1)))):
            # For each fixed material coordinate the gap is affine. Its slope
            # may change sign across the patch, which is harmless to this proof.
            pose = end
            continue
        da, db = a1-a0, b1-b0
        cp0 = np.cross(da, b0)+np.cross(a0, db)
        cp1 = np.cross(da, b1)+np.cross(a1, db)
        cross_speed = max(np.linalg.norm(cp0), np.linalg.norm(cp1))
        normal_speed = cross_speed/area_lower
        normal_acceleration = (2*np.linalg.norm(np.cross(da, db))/area_lower
                               +3*(cross_speed/area_lower)**2)
        normal0 = c0/np.linalg.norm(c0)
        normal_derivative = (cp0-normal0*float(normal0 @ cp0))/np.linalg.norm(c0)
        for cell, xis in cells:
            omega = segment.angular[cell]
            frame = pose.cell_rotations[cell] @ frames[0]
            velocities = []
            offsets = []
            for xi in xis:
                t = xi+1. if cell == 0 else xi
                velocity = ((1-t)*segment.translations[cell]+t*segment.translations[cell+1]
                            -target_motion[0])
                velocities.append(velocity)
                offsets.append(float(normal_derivative @ _base(pose, cell, xi, first[0])
                                     +normal0 @ velocity))
            a = radius*float(normal_derivative @ frame[:, 1]
                             +normal0 @ np.cross(omega, frame[:, 1]))
            b = radius*float(normal_derivative @ frame[:, 2]
                             +normal0 @ np.cross(omega, frame[:, 2]))
            derivative_min, derivative_max = _sinusoidal_extrema(offsets, a, b, angles)
            relative_radius = max(np.linalg.norm(_base(p, cell, xi, corner))
                                  for p, corner in ((pose, first[0]), (end, last[0]))
                                  for xi in xis)+radius
            speed = max(np.linalg.norm(v) for v in velocities)+np.linalg.norm(omega)*radius
            curvature = (normal_acceleration*relative_radius+2*normal_speed*speed
                         +np.linalg.norm(omega)**2*radius)
            padding = 512*np.finfo(float).eps*max(1., relative_radius, speed, curvature,
                                                  abs(derivative_min), abs(derivative_max))
            if not np.all(np.isfinite((derivative_min, derivative_max, curvature, padding))):
                raise ValueError('nonfinite endpoint support derivative bound')
            if not (derivative_min-curvature > padding or derivative_max+curvature < -padding):
                return False
        pose = end
    return True


@dataclass(frozen=True)
class HistoryPanelSupport:
    """A geometric admission result, not a quadrature error estimate."""
    status: str
    region: tuple
    reason: str
    event_covered: bool=False


def segment_gap_slab_enclosure(field,cell,xi_bounds,angle_bounds,segment_index):
    """Whole-rectangle gap/derivative bounds on the original captured time.

    Endpoint spatial extrema and a uniform |g''| bound enclose the gap below
    and above its chord. Midpoint derivative extrema expanded by |g''|*dt/2
    enclose the derivative. No physical history is modified by this factory.
    """
    # Reuse the exact field/reference/selection contract. Its Boolean outcome
    # concerns whole-segment monotonicity and is not this certificate's result.
    history_endpoint_support_certified(field,xi_bounds=xi_bounds,
        angle_bounds=angle_bounds,segment_index=segment_index)
    if type(cell) is not int or cell not in (0,1) or not -1+cell<=xi_bounds[0]<xi_bounds[1]<=cell:
        raise ValueError('support rectangle must lie in one declared material half-cell')
    start,segments=_support_path(field,segment_index);segment=segments[0]
    return _material_gap_slab_enclosure(start,segment,field.patch.target,
        field.patch.source.reference.nodal_triads[0],float(field.patch.rule.definition['radius']),
        cell,xi_bounds,angle_bounds)


def point_gap_slab_enclosure(pair,start,translations,angular,shell_increment):
    """Point specialization under the same straight/untwisted geometry contract."""
    from .native_surface_history import NativeSurfaceMotionSegment
    reference=pair.source.reference;coordinates=reference.coordinates;frames=reference.nodal_triads
    if (any(Fraction(float(coordinates[0,i]))+Fraction(float(coordinates[2,i]))
            !=2*Fraction(float(coordinates[1,i])) for i in range(3))
            or not np.array_equal(frames,np.tile(frames[0],(3,1,1)))
            or np.any(reference.half_cell_lift(pair.source.cell,pair.source.t))):
        raise ValueError('event sequence requires straight untwisted reference')
    segment=NativeSurfaceMotionSegment(translations,angular,shell_increment)
    return _material_gap_slab_enclosure(start,segment,pair.target,frames[0],pair.source.radius,
        pair.source.cell,(pair.source.xi,pair.source.xi),(pair.source.angle,pair.source.angle))


def _material_gap_slab_enclosure(start,segment,target,frame0,radius,cell,xi_bounds,angle_bounds):
    from .native_surface_crossing import GapSlabEnclosure
    omega=segment.angular[cell]
    target_motion=np.asarray(target.adapter.work_map@segment.shell_increment).reshape(3,3)
    da=target_motion[1]-target_motion[0];db=target_motion[2]-target_motion[0]
    ref=target.adapter.reference_positions
    refcross=np.cross(ref[1]-ref[0],ref[2]-ref[0]);refarea=np.linalg.norm(refcross)
    refnormal=refcross/refarea
    origin_weights=_convex_support_origin(start,cell,xi_bounds,target.adapter.current_positions(start.shell_displacement))
    velocity_rows=[_convex_velocity(segment,cell,xi,target_motion,origin_weights) for xi in xi_bounds]
    velocities=[v for v,_ in velocity_rows]
    velocity_error=512*np.finfo(float).eps*max(s for _,s in velocity_rows)
    speed=max(np.linalg.norm(v) for v in velocities)+np.linalg.norm(omega)*radius+3*velocity_error
    def at(time):
        return material_path_pose(start,segment.translations,segment.angular,segment.shell_increment,time)
    def enclose(left,right):
        if not 0<=left<right<=1:
            raise ValueError('ordered original segment time slab required')
        first,last=at(left),at(right);mid=at((left+right)/2)
        corners=[target.adapter.current_positions(p.shell_displacement) for p in (first,last,mid)]
        edges=[(c[1]-c[0],c[2]-c[0]) for c in corners]
        a0,b0=edges[0];a1,b1=edges[1];am,bm=edges[2]
        controls=(np.cross(a0,b0),.5*(np.cross(a0,b1)+np.cross(a1,b0)),np.cross(a1,b1))
        area_scale=max(refarea,max(np.linalg.norm(a0),np.linalg.norm(a1))*max(np.linalg.norm(b0),np.linalg.norm(b1)))
        area_lower=min(float(refnormal@c) for c in controls)-512*np.finfo(float).eps*area_scale
        if area_lower<=1e-12*area_scale:
            raise ValueError('unresolved target area/orientation in support slab')
        cp=[np.cross(da,b)+np.cross(a,db) for a,b in edges[:2]]
        normal_speed=max(np.linalg.norm(c) for c in cp)/area_lower
        normal_acceleration=2*np.linalg.norm(np.cross(da,db))/area_lower+3*normal_speed**2
        relative_rows=[_convex_relative(p,cell,xi,c,origin_weights)
            for p,c in zip((first,last),corners[:2]) for xi in xi_bounds]
        position_error=512*np.finfo(float).eps*max(s for _,s in relative_rows)
        relative_radius=max(np.linalg.norm(v) for v,_ in relative_rows)+radius+3*position_error
        curvature=normal_acceleration*relative_radius+2*normal_speed*speed+np.linalg.norm(omega)**2*radius
        cross=np.cross(am,bm);area=np.linalg.norm(cross);normal=cross/area
        cp_mid=np.cross(da,bm)+np.cross(am,db)
        normal_rate=(cp_mid-normal*float(normal@cp_mid))/area
        frame=mid.cell_rotations[cell]@frame0
        offsets=[float(normal_rate@_convex_relative(mid,cell,xi,corners[2],origin_weights)[0]+normal@v)
                 for xi,v in zip(xi_bounds,velocities)]
        ca=radius*float(normal_rate@frame[:,1]+normal@np.cross(omega,frame[:,1]))
        cb=radius*float(normal_rate@frame[:,2]+normal@np.cross(omega,frame[:,2]))
        dl,du=_sinusoidal_extrema(offsets,ca,cb,angle_bounds)
        minima=[_minimum(p,target,frame0,radius,cell,xi_bounds,angle_bounds) for p in (first,last)]
        negative_maxima=[_minimum(p,target,frame0,radius,cell,xi_bounds,angle_bounds,gap_sign=-1.) for p in (first,last)]
        scale=max(1.,relative_radius,speed,curvature,abs(dl),abs(du),
            *(float(np.max(np.abs(c))) for c in corners),
            *(float(np.max(np.abs(p.positions))) for p in (first,last,mid)))
        padding=max(512*np.finfo(float).eps*scale,*(p for _,p in minima+negative_maxima))
        # Cover cancellation before the shifted radius became small, including
        # derivative products and target normal/rate conditioning. This follows
        # the existing padded binary64 bound convention, not interval arithmetic.
        edge_scale=max(np.linalg.norm(a) for edge in edges for a in edge)
        normal_error=512*np.finfo(float).eps*max(1.,edge_scale**2/area_lower)+6*position_error*edge_scale/area_lower
        normal_rate_error=(512*np.finfo(float).eps*max(1.,normal_speed)
                           +3*normal_error*normal_speed)
        derivative_error=(3*position_error*np.linalg.norm(normal_rate)+3*velocity_error
                          +normal_error*speed+normal_rate_error*relative_radius)
        padding=max(padding,derivative_error)
        curvature+=padding
        width=right-left
        # Preserve the correlation between axial gap and gap rate. k is only
        # a fixed enclosure coordinate: at g=0, q=g_t-k*g equals g_t exactly.
        relative_mid=[_convex_relative(mid,cell,xi,corners[2],origin_weights)[0] for xi in xi_bounds]
        axial_gap=float(normal@(relative_mid[1]-relative_mid[0]))
        k=0. if axial_gap==0. else (offsets[1]-offsets[0])/axial_gap
        if not np.isfinite(k):raise ValueError('nonfinite zero-set rate coordinate')
        qnormal=normal_rate-k*normal
        qoffsets=[float(qnormal@r+normal@v) for r,v in zip(relative_mid,velocities)]
        qa=radius*float(qnormal@frame[:,1]+normal@np.cross(omega,frame[:,1]))
        qb=radius*float(qnormal@frame[:,2]+normal@np.cross(omega,frame[:,2]))
        qlo,qhi=_sinusoidal_extrema(qoffsets,qa,qb,angle_bounds)
        qpadding=padding*(1+abs(k))+512*np.finfo(float).eps*max(1.,np.linalg.norm(qnormal)*relative_radius+speed)
        rate_bound=max(abs(dl),abs(du))+curvature*width/2+padding
        qcurvature=curvature+abs(k)*rate_bound
        return GapSlabEnclosure(min(v for v,_ in minima)-curvature*width**2/8-padding,
            -min(v for v,_ in negative_maxima)+curvature*width**2/8+padding,
            dl-curvature*width/2-padding,du+curvature*width/2+padding,
            spatial_derivative_width=qhi-qlo,temporal_derivative_width=qcurvature*width+2*qpadding,
            zero_derivative_lower=qlo-qcurvature*width/2-qpadding,
            zero_derivative_upper=qhi+qcurvature*width/2+qpadding)
    return enclose


def panel_transverse_event_cover(field,cell,xi_bounds,angle_bounds,segment_index,*,budget):
    """Complete time cover with transverse roots, allowing either orientation.

    Strict signed slabs contain no root; strict zero-rate slabs contain only
    transverse roots. A complete finite time cover may contain opposite crossing
    directions regardless of endpoint signs. Point replay must still resolve the
    ordered events and physical history; this support certificate cannot replace
    that admission. Tangencies remain unresolved and are never admitted: a
    possible-touch enclosure proves only that a touch cannot be excluded, never
    a complete force/tangent/history certificate, so a stalled cover yields
    unresolved for spatial subdivision or refusal exactly as before.
    This proves support, never zero history or quadrature accuracy.
    """
    enclose=segment_gap_slab_enclosure(field,cell,xi_bounds,angle_bounds,segment_index)
    pending=[(0.,1.)];visited=0;roots=False;closed=False
    while pending:
        if visited>=field.maximum_intervals:return None
        if not budget.consume():raise MemoryError('native friction aggregate support interval cap exhausted')
        left,right=pending.pop();visited+=1;bounds=enclose(left,right)
        if bounds.gap_lower>0. or bounds.gap_upper<0.:
            closed=closed or bounds.gap_upper<0.
            continue
        # q = g_t - k*g equals g_t at a root. Its strict enclosure
        # therefore proves transverse orientation even where the ordinary
        # rate enclosure loses the correlation between gap and rate.
        direction=(1 if bounds.zero_derivative_lower>0. else
                   -1 if bounds.zero_derivative_upper<0. else 0)
        if direction:
            roots=True
            continue
        # Temporal refinement cannot reduce this spatial contribution. Yield
        # unresolved to the existing spatial refiner; this is never admission.
        if (bounds.spatial_derivative_width>0.
                and bounds.spatial_derivative_width>=bounds.temporal_derivative_width):
            return None
        middle=(left+right)/2
        if not left<middle<right:return None
        pending.extend(((middle,right),(left,middle)))
    if roots:return 'events'
    return 'closed' if closed else 'open'


def classify_history_panel(field,cell,xi_bounds,angle_bounds,*,budget):
    """Resolve one full integration rectangle without sampling its support.

    Unresolved regions must be subdivided or rejected even if quadrature gives
    nonzero, mutually agreeing estimates. No constitutive history is altered.
    """
    if type(field) is not NativeSurfaceHistoryField:
        raise TypeError('exact native surface history field required')
    if type(cell) is not int or cell not in (0,1):
        raise ValueError('one native half-cell required')
    for bounds,patch_bounds,domain in (
            (xi_bounds,field.patch.rule.definition['xi_range'],(-1.+cell,float(cell))),
            (angle_bounds,field.patch.rule.definition['angle_range'],(0.,2*pi))):
        values=np.asarray(bounds,dtype=float)
        if (values.shape!=(2,) or not np.all(np.isfinite(values))
                or not max(domain[0],patch_bounds[0])<=values[0]<values[1]<=min(domain[1],patch_bounds[1])):
            raise ValueError('support rectangle must lie in one declared material half-cell')
    if type(budget) is not SupportIntervalBudget:
        raise TypeError('exact shared support interval budget required')
    if not budget.consume():
        raise MemoryError('native friction aggregate support interval cap exhausted')
    region=(cell,*map(float,xi_bounds),*map(float,angle_bounds))
    pose=field.initial;all_open=True;event_covered=False
    for index,segment in enumerate(field.segments):
        end=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
        monotone=history_endpoint_support_certified(field,xi_bounds=xi_bounds,angle_bounds=angle_bounds,
            segment_index=index)
        if monotone:
            for endpoint in (pose,end):
                value,padding=_minimum(endpoint,field.patch.target,field.patch.source.reference.nodal_triads[0],
                    field.patch.rule.definition['radius'],cell,xi_bounds,angle_bounds)
                all_open=all_open and value>padding
        # A coarse interval proof yields to spatial refinement when unresolved;
        # it must not consume the full shared budget inside one local panel.
        elif panel_open_over_history(field,cell,xi_bounds,angle_bounds,maximum_intervals=1,
                budget=budget,segment_index=index):
            pass
        elif panel_closed_over_history(field,cell,xi_bounds,angle_bounds,maximum_intervals=1,
                budget=budget,segment_index=index):
            all_open=False
        else:
            cover=None
            if field.patch.pairs[0].activation_policy in ('transverse_events',
                    'transverse_event_sequence'):
                cover=panel_transverse_event_cover(field,cell,xi_bounds,angle_bounds,index,budget=budget)
            if cover is None:
                return HistoryPanelSupport('unresolved',region,f'unresolved history segment {index}')
            if cover=='events':
                all_open=False;event_covered=True
            elif cover=='closed':
                all_open=False
        pose=end
    if not field.segments:
        # Validate the geometry contract even for an empty physical history.
        history_endpoint_support_certified(field,xi_bounds=xi_bounds,angle_bounds=angle_bounds)
        value,padding=_minimum(pose,field.patch.target,field.patch.source.reference.nodal_triads[0],
            field.patch.rule.definition['radius'],cell,xi_bounds,angle_bounds)
        all_open=value>padding
    return HistoryPanelSupport('open' if all_open else 'resolved',region,
        'complete segmentwise support',event_covered)


def history_endpoint_support_cover_certified(field, *, budget):
    """Cover every material point by monotone history or proven separation.

    This preserves the endpoint-support implication needed by integration;
    uniform temporal derivative sign across the entire patch is unnecessary.
    An unresolved region or exhausted shared budget never establishes admission.
    """
    if type(budget) is not SupportIntervalBudget:
        raise TypeError('exact shared support interval budget required')
    if history_endpoint_support_certified(field):return True
    rule=field.patch.rule;lo,hi=rule.definition['xi_range'];alo,ahi=rule.definition['angle_range']
    pending=[(c,l,r,alo,ahi) for c,l,r in
        ((0,max(-1.,lo),min(0.,hi)),(1,max(0.,lo),min(1.,hi))) if l<r]
    while pending:
        if not budget.consume():
            raise MemoryError('native friction aggregate support interval cap exhausted')
        cell,left,right,lower,upper=pending.pop()
        if history_endpoint_support_certified(field,xi_bounds=(left,right),angle_bounds=(lower,upper)):
            continue
        if panel_open_over_history(field,cell,(left,right),(lower,upper),
                maximum_intervals=field.maximum_intervals,budget=budget):
            continue
        # A sampled hidden event can REJECT a cover, never certify it. This
        # catches genuine open-contact-open pulses without blind subdivision.
        xi=(left+right)/2;angle=(lower+upper)/2;pose=field.initial
        for segment in field.segments:
            middle=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,.5)
            end=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
            values=[_minimum(p,field.patch.target,field.patch.source.reference.nodal_triads[0],
                rule.definition['radius'],cell,(xi,xi),(angle,angle)) for p in (pose,middle,end)]
            if values[0][0]>values[0][1] and values[2][0]>values[2][1] and values[1][0]<-values[1][1]:
                return False
            pose=end
        if (right-left)/(hi-lo)>=(upper-lower)/(ahi-alo):
            midpoint=(left+right)/2
            if midpoint in (left,right):return False
            pending.extend(((cell,left,midpoint,lower,upper),(cell,midpoint,right,lower,upper)))
        else:
            midpoint=(lower+upper)/2
            if midpoint in (lower,upper):return False
            pending.extend(((cell,left,right,lower,midpoint),(cell,left,right,midpoint,upper)))
    return True


def history_target_ownership_certified(field, *, budget):
    """Prove the whole straight patch stays inside its S3 target over history.

    Three inward edge-plane minima are exact in both material coordinates.
    Temporal interval bounds cover every point, not only integration nodes.
    Source orientation remains a separate required admission check.
    """
    if type(field) is not NativeSurfaceHistoryField:
        raise TypeError('exact native surface history field required')
    if type(budget) is not SupportIntervalBudget:
        raise TypeError('exact shared support interval budget required')
    if sha(field._definition()) != field.identity:
        raise ValueError('native material history field definition changed')
    rule = field.patch.rule
    reference = field.patch.source.reference
    coordinates, frames = reference.coordinates, reference.nodal_triads
    if (any(Fraction(float(coordinates[0, i]))+Fraction(float(coordinates[2, i]))
            != 2*Fraction(float(coordinates[1, i])) for i in range(3))
            or not np.array_equal(frames, np.tile(frames[0], (3, 1, 1)))):
        raise ValueError('material support requires exact straight untwisted reference')
    target = field.patch.target
    radius = float(rule.definition['radius'])
    lo, hi = rule.definition['xi_range']
    angles = rule.definition['angle_range']
    cells = [(c, (l, r)) for c, l, r in
             ((0, max(lo, -1.), min(hi, 0.)), (1, max(lo, 0.), min(hi, 1.))) if r > l]
    reference_corners = target.adapter.reference_positions
    reference_cross = np.cross(reference_corners[1]-reference_corners[0],
                               reference_corners[2]-reference_corners[0])
    reference_area = float(np.linalg.norm(reference_cross))
    reference_normal = reference_cross/reference_area

    def margins(pose, clearance):
        corners = target.adapter.current_positions(pose.shell_displacement)
        normal = np.asarray(target.physical_pose(pose.shell_displacement).material_frame)[:, 2]
        values = []
        for cell, xis in cells:
            frame = pose.cell_rotations[cell] @ frames[0]
            for i in range(3):
                edge = corners[(i+1)%3]-corners[i]
                inward = np.cross(normal, edge/np.linalg.norm(edge))
                offsets = [float(inward @ _base(pose, cell, xi, corners[i])) for xi in xis]
                a, b = radius*float(inward @ frame[:, 1]), radius*float(inward @ frame[:, 2])
                minimum, _ = _sinusoidal_extrema(offsets, a, b, angles)
                values.append(minimum-clearance)
        scale = max(1., radius, float(np.max(np.abs(pose.positions))),
                    float(np.max(np.abs(pose.position_low))), float(np.max(np.abs(corners))))
        return np.asarray(values), 512*np.finfo(float).eps*scale

    pose = field.initial
    first = target.adapter.current_positions(pose.shell_displacement)
    initial_clearance = 1e-12*max(np.linalg.norm(first[(i+1)%3]-first[i]) for i in range(3))
    initial_values, initial_padding = margins(pose, initial_clearance)
    if np.min(initial_values) <= initial_padding:
        return False
    for segment in field.segments:
        end = material_path_pose(pose, segment.translations, segment.angular, segment.shell_increment, 1.)
        last = target.adapter.current_positions(end.shell_displacement)
        clearance = 1e-12*max(np.linalg.norm(corners[(i+1)%3]-corners[i])
                              for corners in (first, last) for i in range(3))
        first_values, first_padding = margins(pose, clearance)
        last_values, last_padding = margins(end, clearance)
        if min(np.min(first_values)-first_padding, np.min(last_values)-last_padding) <= 0.:
            return False
        a0, b0 = first[1]-first[0], first[2]-first[0]
        a1, b1 = last[1]-last[0], last[2]-last[0]
        controls = (np.cross(a0, b0), .5*(np.cross(a0, b1)+np.cross(a1, b0)), np.cross(a1, b1))
        area_lower = min(float(reference_normal @ c) for c in controls)
        area_scale = max(reference_area, max(np.linalg.norm(a0), np.linalg.norm(a1))
                         *max(np.linalg.norm(b0), np.linalg.norm(b1)))
        if area_lower <= 1e-12*area_scale:
            return False
        target_motion = np.asarray(target.adapter.work_map @ segment.shell_increment).reshape(3, 3)
        if (not np.any(segment.angular)
                and np.array_equal(target_motion, np.tile(target_motion[0], (3, 1)))):
            # All inward planes translate without changing direction; every
            # material point's edge distance is affine along the segment.
            if not budget.consume():
                return False
            pose, first = end, last
            continue
        normal_speed = (np.linalg.norm(a1-a0)*max(np.linalg.norm(b0), np.linalg.norm(b1))
                        +np.linalg.norm(b1-b0)*max(np.linalg.norm(a0), np.linalg.norm(a1)))/area_lower
        inward_rates = []
        for i in range(3):
            edge = first[(i+1)%3]-first[i]
            change = (last[(i+1)%3]-last[i])-edge
            denominator = float(change @ change)
            alpha = 0. if denominator == 0. else float(np.clip(-float(edge @ change)/denominator, 0., 1.))
            edge_lower = float(np.linalg.norm(edge+alpha*change))
            if edge_lower <= 0.:
                return False
            inward_rates.append(normal_speed+np.linalg.norm(change)/edge_lower)
        rates = []
        for cell, xis in cells:
            for i in range(3):
                relative_radius = max(np.linalg.norm(_base(p, cell, xi, corner))
                                      for p, corner in ((pose, first[i]), (end, last[i]))
                                      for xi in xis)+radius
                speeds = []
                for xi in xis:
                    t = xi+1. if cell == 0 else xi
                    speeds.append(np.linalg.norm((1-t)*segment.translations[cell]
                                  +t*segment.translations[cell+1]-target_motion[i]))
                relative_speed = max(speeds)+np.linalg.norm(segment.angular[cell])*radius
                rates.append(relative_speed+inward_rates[i]*relative_radius)
        rates = np.asarray(rates)
        if not np.all(np.isfinite(rates)):
            raise ValueError('nonfinite swept target ownership rate')
        pending = [(0., 1.)]
        segment_visits = 0
        while pending:
            if segment_visits >= field.maximum_intervals:
                return False
            left, right = pending.pop()
            midpoint = left+(right-left)/2.
            if midpoint == left or midpoint == right or not budget.consume():
                return False
            current = material_path_pose(pose, segment.translations, segment.angular,
                                         segment.shell_increment, midpoint)
            values, padding = margins(current, clearance)
            segment_visits += 1
            if np.min(values) <= padding:
                return False
            if np.all(values-rates*(right-left)/2. > max(padding, first_padding, last_padding)):
                continue
            pending.extend(((left, midpoint), (midpoint, right)))
        pose, first = end, last
    return True
