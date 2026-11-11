"""Reference-area quadrature for the actual retained round material surface.

Development geometry only. Cell interfaces are integration breaks; this rule
does not create end caps, resolve target overlap, or certify between-point gap.
"""
from dataclasses import dataclass
from math import cos, sin, tau

import numpy as np
from numpy.polynomial.legendre import leggauss

from anyfesolver._ge_beam3_deterministic_reduction import deterministic_dot, deterministic_norm
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_retained_station import NativeRetainedRoundStation, _readonly


@dataclass(frozen=True)
class NativeRoundSurfaceSample:
    station: NativeRetainedRoundStation
    reference_weight_m2: float
    parametric_weight: float


@dataclass(frozen=True)
class NativeRoundSurfaceGeometry:
    station_state: object
    xi_tangent: np.ndarray
    angle_tangent: np.ndarray
    outward_normal: np.ndarray
    area_jacobian: float


def _oriented_measure(xi, angle, radial):
    cross=np.cross(angle,xi)
    jacobian=float(deterministic_norm(cross))
    if not np.isfinite(jacobian) or jacobian<=0 or deterministic_dot(cross,radial)<=0:
        raise ValueError('native round surface singular or reversed at sample')
    return cross,jacobian


@dataclass(frozen=True)
class _ReferenceRoundArea:
    """Candidate-local preparation for an already validated reference station.

    Uses the same lift, parameter derivatives and outward check as
    surface_geometry. No current geometry or contact law is cached here.
    """
    frame: np.ndarray
    derivative: np.ndarray
    chord: np.ndarray
    lift_derivative: np.ndarray
    radius: float

    def area_jacobian(self, angle):
        radial=self.radius*np.array((0.,cos(angle),sin(angle)))
        angular=self.radius*np.array((0.,-sin(angle),cos(angle)))
        # Retain the reference lift/chord arithmetic of surface_geometry.
        xi=self.chord+(self.lift_derivative+self.derivative@radial)
        _,jacobian=_oriented_measure(xi,self.frame@angular,self.frame@radial)
        return jacobian


def _prepare_reference_area(reference, xi, radius):
    """Prepare only after the caller validates its immutable rule identity."""
    cell=0 if xi<0 else 1
    chord=reference.coordinates[cell+1]-reference.coordinates[cell]
    return _ReferenceRoundArea(_readonly(reference.frame(xi)),
        _readonly(reference.frame_derivative(xi)),_readonly(chord),
        _readonly(reference.derivative(xi)-chord),float(radius))


def surface_geometry(station, positions, position_low, cell_rotations):
    """Exact within-cell parameter derivatives, including reference frame turn."""
    if type(station) is not NativeRetainedRoundStation:
        raise TypeError('exact retained round station required')
    bound=NativeRetainedRoundStation(station.element_id,station.reference,
                                     station.xi,station.radius,station.angle)
    if bound.identity!=station.identity or (bound.cell,bound.t)!=(station.cell,station.t):
        raise ValueError('native surface station definition changed')
    state=station.evaluate(positions,position_low,cell_rotations)
    reference=station.reference; cell=station.cell
    frame=reference.frame(station.xi)
    derivative=reference.frame_derivative(station.xi)
    radial=station.radius*np.array((0.,cos(station.angle),sin(station.angle)))
    angular=station.radius*np.array((0.,-sin(station.angle),cos(station.angle)))
    # r0 minus its half-cell chord is the reference lift. Differentiate that
    # same lift; changing this to a centreline tangent loses curved arm work.
    chord=reference.coordinates[cell+1]-reference.coordinates[cell]
    lift_derivative=reference.derivative(station.xi)-chord
    rotation=np.asarray(cell_rotations)[cell]
    current_chord=(np.asarray(positions)[cell+1]-np.asarray(positions)[cell]
                   +np.asarray(position_low)[cell+1]-np.asarray(position_low)[cell])
    xi=current_chord+rotation@(lift_derivative+derivative@radial)
    angle=rotation@frame@angular
    # Preserve the declared outward side. Taking only an absolute area would
    # conceal inversion on a thick curved surface or a deformed cell.
    cross,jacobian=_oriented_measure(xi,angle,rotation@frame@radial)
    return NativeRoundSurfaceGeometry(state,_readonly(xi),_readonly(angle),
                                      _readonly(cross/jacobian),jacobian)


class NativeRoundSurfaceRule:
    """Fixed material quadrature, split at native half-cell interfaces.

    Reference weights never follow current area. Refinement creates a new rule
    identity and cannot silently change the physical history discretization.
    """
    def __init__(self, element_id, reference, radius, *, xi_range=(-1.,1.),
                 angle_range=(0.,tau), axial_order=4, angular_order=8,
                 maximum_points=256):
        for name,value in (('axial_order',axial_order),('angular_order',angular_order),
                           ('maximum_points',maximum_points)):
            if type(value) is not int or value<1:
                raise ValueError(name+' must be a positive integer')
        limits=[]
        for bounds,lower,upper in ((xi_range,-1.,1.),(angle_range,0.,tau)):
            values=np.asarray(bounds,dtype=float)
            if values.shape!=(2,) or not np.all(np.isfinite(values)) or not lower<=values[0]<values[1]<=upper:
                raise ValueError('invalid material surface parameter interval')
            limits.append(tuple(map(float,values)))
        lo,hi=limits[0]; alo,ahi=limits[1]
        breaks=[lo]+([0.] if lo<0.<hi else [])+[hi]
        count=(len(breaks)-1)*axial_order*angular_order
        if count>maximum_points:
            raise MemoryError('native surface quadrature point cap exceeded')
        # Validate the exact structural geometry before any quadrature allocation.
        NativeRetainedRoundStation(element_id,reference,float((breaks[0]+breaks[1])/2),radius,float((alo+ahi)/2))
        x,w=leggauss(axial_order); a,v=leggauss(angular_order)
        samples=[]; rotations=np.tile(np.eye(3),(2,1,1))
        for left,right in zip(breaks[:-1],breaks[1:]):
            for q,weight_x in zip(x,w):
                xi=float((left+right)/2+(right-left)*q/2)
                for angle,weight_a in zip(a,v):
                    alpha=float((alo+ahi)/2+(ahi-alo)*angle/2)
                    station=NativeRetainedRoundStation(element_id,reference,xi,radius,alpha)
                    geometry=surface_geometry(station,reference.coordinates,
                                               np.zeros((3,3)),rotations)
                    weight=float(weight_x*weight_a*(right-left)*(ahi-alo)/4)
                    samples.append(NativeRoundSurfaceSample(station,weight*geometry.area_jacobian,weight))
        self.samples=tuple(samples)
        self.definition=dict(profile='NATIVE_ROUND_REFERENCE_SURFACE_RULE_DEV_V1',
            element=element_id,reference=reference.canonical_data(),radius=float(radius),
            xi_range=limits[0],angle_range=limits[1],axial_order=axial_order,
            angular_order=angular_order,maximum_points=maximum_points,
            samples=[(s.station.identity,s.reference_weight_m2,s.parametric_weight) for s in samples])
        self.identity=sha(self.definition)

    def evaluate(self,positions,position_low,cell_rotations):
        if sha(self.definition)!=self.identity:
            raise ValueError('native surface quadrature definition changed')
        if tuple((s.station.identity,s.reference_weight_m2,s.parametric_weight) for s in self.samples)!=tuple(self.definition['samples']):
            raise ValueError('native surface quadrature samples changed')
        return tuple(surface_geometry(s.station,positions,position_low,cell_rotations) for s in self.samples)

    def current_bounds(self, positions, position_low, cell_rotations):
        """Whole-patch current search box; not a gap or swept certificate."""
        from .native_surface_bounds import native_round_patch_bounds
        return native_round_patch_bounds(self, positions, position_low, cell_rotations)

    def swept_bounds(self, positions, position_low, cell_rotations, translations, angular):
        """Whole affine/Exp material path search box; no event/gap certificate."""
        from .native_surface_bounds import native_round_patch_swept_bounds
        return native_round_patch_swept_bounds(self, positions, position_low,
                                              cell_rotations, translations, angular)

    def certify_orientation_path(self,start,translations,angular,shell_increment,*,maximum_intervals=256):
        """Positive signed surface area throughout the actual exponential path.

        At a fixed material sample, signed area divided by radius squared is
        c(alpha).dot(R(alpha)e1) + (lift' + R0' rho).dot(e1).
        The second term is constant. A chord/rotation Lipschitz bound encloses
        the first term on every interval; sampled positivity alone is not used.
        """
        return sum(certify_station_orientation_path(sample.station,start,translations,angular,
            shell_increment,maximum_intervals=maximum_intervals) for sample in self.samples)


def certify_station_orientation_path(station,start,translations,angular,shell_increment,*,maximum_intervals=256):
    """Same conservative orientation bound for existing or newly queried points."""
    from .native_surface_path import material_path_pose
    if type(station) is not NativeRetainedRoundStation:
        raise TypeError('exact native material station required')
    if type(maximum_intervals) is not int or maximum_intervals<1:
        raise ValueError('positive surface orientation interval cap required')
    end=material_path_pose(start,translations,angular,shell_increment,1.)
    visited=0
    cell=station.cell; reference=station.reference
    frame=reference.frame(station.xi); director=frame[:,0]
    radial=station.radius*np.array((0.,cos(station.angle),sin(station.angle)))
    lift=reference.derivative(station.xi)-(reference.coordinates[cell+1]-reference.coordinates[cell])
    constant=float((lift+reference.frame_derivative(station.xi)@radial)@director)
    def chord(pose):
        return pose.positions[cell+1]-pose.positions[cell]+pose.position_low[cell+1]-pose.position_low[cell]
    c0,c1=chord(start),chord(end)
    rate=np.linalg.norm(c1-c0)+max(np.linalg.norm(c0),np.linalg.norm(c1))*np.linalg.norm(np.asarray(angular)[cell])
    padding=512*np.finfo(float).eps*max(1.,np.linalg.norm(c0),np.linalg.norm(c1),abs(constant))
    pending=[(0.,1.)]; point_visits=0
    while pending:
        if point_visits>=maximum_intervals:
            raise ValueError('native/S3 material history path requires subdivision: source orientation budget')
        left,right=pending.pop(); middle=(left+right)/2
        pose=material_path_pose(start,translations,angular,shell_increment,middle)
        value=float(chord(pose)@(pose.cell_rotations[cell]@director))+constant
        visited+=1; point_visits+=1
        if value<=padding:
            raise ValueError('native/S3 material history path requires subdivision: source surface orientation')
        if value-rate*(right-left)/2>padding:
            continue
        if middle==left or middle==right:
            raise ValueError('native/S3 material history path requires subdivision: source parameter resolution')
        pending.extend(((left,middle),(middle,right)))
    return visited
