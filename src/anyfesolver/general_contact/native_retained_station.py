"""Physical round station on a retained, two-cell native B3-GE beam.

This is a local C3 geometry/work map. It owns no nonlinear analysis and does
not condense the cell rotations or admit the native beam to generic FEModel.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from struct import pack
from collections import OrderedDict
from math import cos, fsum, isfinite, sin, tau

import numpy as np

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from anyfesolver._ge_beam3_p5.compensated_coordinates import validate_pair
from anyfesolver._ge_beam3_p5_seeded.core import sha


def _cross_matrix(value: np.ndarray) -> np.ndarray:
    x, y, z = map(float, value)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


# Exp(delta)=I+hat(delta)+hat(delta)^2/2+O(||delta||^3).
# Thus its exact zero-increment derivatives are Gi and (Gi Gj+Gj Gi)/2.
# The physical spatial chart and all work maps remain unchanged.
_SPATIAL_ROTATION_GENERATORS=_readonly(np.stack([_cross_matrix(axis) for axis in np.eye(3)]))
_SPATIAL_ROTATION_SECOND=_readonly(np.stack([
    np.stack([.5*(a@b+b@a) for b in _SPATIAL_ROTATION_GENERATORS])
    for a in _SPATIAL_ROTATION_GENERATORS]))


@dataclass(frozen=True)
class NativeRetainedRoundStationState:
    """Position and virtual-work maps before structural/contact assembly."""

    centerline_position: np.ndarray
    surface_position: np.ndarray
    centerline_geometric_jacobian: np.ndarray
    surface_geometric_jacobian: np.ndarray

    def generalized_surface_jacobian(self) -> np.ndarray:
        """The final 18 constitutive/resultant coordinates carry no position."""
        result = np.zeros((3, 42))
        result[:, :24] = self.surface_geometric_jacobian
        return result

    def generalized_surface_work(self, physical_force: np.ndarray) -> np.ndarray:
        force = _array(physical_force, (3,), "native station physical force")
        return self.generalized_surface_jacobian().T @ force


@dataclass(frozen=True)
class NativeRetainedRoundSurfaceVariations:
    """Local spatial rotation-chart derivatives of one material section point.

    Second derivatives use Exp(delta_theta) Uc at delta_theta=0. Arrays stay
    local to the three physical cell rotations. The section frame is not a
    swept-surface normal frame: shear can make the radial direction non-normal.
    """

    station: NativeRetainedRoundStationState
    angular_columns: tuple[int, int, int]
    surface_arm: np.ndarray
    section_frame: np.ndarray
    section_frame_gradient: np.ndarray
    section_frame_hessian: np.ndarray
    surface_rotation_hessian: np.ndarray

    def spatial_surface_force_tangent(self, physical_force, position_tangent=None):
        """Derivative of J(x)^T f(x) under a new spatial multiplicative update.

        position_tangent is df/dx with any other bodies held fixed. Other-body
        and frame/history variations belong to the coupled law, not this map.
        A chart energy Hessian alone is not this generally nonsymmetric tangent.
        """
        force=_array(physical_force,(3,),"native surface physical force")
        derivative=(np.zeros((3,3)) if position_tangent is None else
                    _array(position_tangent,(3,3),"native surface force derivative"))
        J=self.station.generalized_surface_jacobian()
        tangent=J.T@derivative@J
        angular=np.asarray(self.angular_columns)
        # d(v cross f)/d(theta) at fixed spatial force is [f]x [v]x.
        tangent[np.ix_(angular,angular)]+=_cross_matrix(force)@_cross_matrix(self.surface_arm)
        return _readonly(tangent)


def _reference_guard_value(value):
    if isinstance(value,np.ndarray):return (value.shape,value.dtype.str,value.tobytes())
    if isinstance(value,(float,np.floating)):return pack('>d',float(value))
    if isinstance(value,tuple):return tuple(_reference_guard_value(v) for v in value)
    if is_dataclass(value):return tuple((f.name,_reference_guard_value(getattr(value,f.name))) for f in fields(value))
    return value


def _reference_guard_snapshot(reference):
    # Exact bytes cover the full instance data underlying canonical_data,
    # including derived coefficients/frame branches and regularity controls.
    return (type(reference),tuple((name,_reference_guard_value(value))
        for name in sorted({slot for cls in type(reference).__mro__ for slot in getattr(cls,'__slots__',())})
        for value in (getattr(reference,name),)))


class NativeRetainedRoundStation:
    """One owned interior material station and declared circular section point."""

    def __init__(self, element_id: int,
                 reference: CenteredCurvedBeam3ReferenceGeometry,
                 xi: float, radius: float, angle: float):
        if type(element_id) is not int or element_id <= 0:
            raise ValueError("native contact station needs positive element identity")
        if type(reference) is not CenteredCurvedBeam3ReferenceGeometry:
            raise TypeError("native contact station needs exact centered B3-GE reference")
        parameters = (xi, radius, angle)
        if any(type(item) not in (int, float) or not isfinite(item)
               for item in parameters):
            raise ValueError("native contact station needs finite binary64 geometry")
        xi, radius, angle = map(float, parameters)
        if not (-1.0 < xi < 0.0 or 0.0 < xi < 1.0):
            raise ValueError("native contact station must be interior to one half-cell")
        if radius <= 0.0 or not 0.0 <= angle < tau:
            raise ValueError("native round station needs positive radius and angle in [0, 2pi)")
        self.element_id = element_id
        self.reference = reference
        self.xi = xi
        self.radius = radius
        self.angle = angle
        self.cell = 0 if xi < 0.0 else 1
        self.t = xi + 1.0 if self.cell == 0 else xi
        self._reference_digest = sha(reference.canonical_data())
        self.identity = sha(dict(
            policy="NATIVE_RETAINED_ROUND_STATION_GEOMETRY_DEV_V0",
            element_id=element_id, reference=self._reference_digest,
            xi=xi, radius=radius, angle=angle, cell=self.cell,
        ))
        self._geometry_signature=(element_id,xi,radius,angle,self.cell,self.t)
        self._reference_guard=_reference_guard_snapshot(reference)
        self._reference_frame=_readonly(reference.frame(xi))
        self._reference_lift=_readonly(reference.half_cell_lift(self.cell,self.t))
        self._radial_reference=_readonly(radius*(cos(angle)*self._reference_frame[:,1]
            +sin(angle)*self._reference_frame[:,2]))
        # Exact pose memo, local to one material station. At most eight small
        # readonly geometry/work-map records; no material state is stored.
        self._pose_cache=OrderedDict()

    def evaluate(self, positions: np.ndarray, position_low: np.ndarray,
                 cell_rotations: np.ndarray) -> NativeRetainedRoundStationState:
        if (self.element_id,self.xi,self.radius,self.angle,self.cell,self.t)!=self._geometry_signature:
            raise ValueError("native contact station geometry changed")
        if _reference_guard_snapshot(self.reference) != self._reference_guard:
            raise ValueError("native contact station reference identity changed")
        high=np.asarray(positions,dtype=float);low=np.asarray(position_low,dtype=float)
        rotations=np.asarray(cell_rotations,dtype=float)
        key=(high.shape,low.shape,rotations.shape,high.tobytes(),low.tobytes(),rotations.tobytes())
        if key in self._pose_cache:
            # Byte-identical values have already passed all finite, rotation,
            # shape and compensated-pair checks. Changed values take the full
            # validation path; geometry/reference binding is checked above.
            self._pose_cache.move_to_end(key)
            return self._pose_cache[key]
        high = _array(high, (3, 3), "native contact nodal positions")
        low = _array(low, (3, 3), "native contact nodal low parts")
        rotations = _frames(rotations, 2, "native contact cell rotations")
        for first, second in zip(high.flat, low.flat):
            validate_pair(float(first), float(second))
        lift = self._reference_lift
        frame = self._reference_frame
        radial_reference = self._radial_reference
        rotation = rotations[self.cell]
        curved_lift = rotation @ lift
        surface_lift = rotation @ (lift+radial_reference)
        left, right = self.cell, self.cell+1
        weights = (1.0-self.t, self.t)
        base = np.array([
            fsum((weights[0]*float(high[left, axis]),
                  weights[0]*float(low[left, axis]),
                  weights[1]*float(high[right, axis]),
                  weights[1]*float(low[right, axis])))
            for axis in range(3)
        ])
        centerline = base+curved_lift
        surface = base+surface_lift
        centerline_jacobian = np.zeros((3, 24))
        surface_jacobian = np.zeros((3, 24))
        for node, weight in ((left, weights[0]), (right, weights[1])):
            centerline_jacobian[:, 6*node:6*node+3] = weight*np.eye(3)
            surface_jacobian[:, 6*node:6*node+3] = weight*np.eye(3)
        angular = slice(18+3*self.cell, 21+3*self.cell)
        centerline_jacobian[:, angular] = -_cross_matrix(curved_lift)
        surface_jacobian[:, angular] = -_cross_matrix(surface_lift)
        result=NativeRetainedRoundStationState(
            _readonly(centerline), _readonly(surface),
            _readonly(centerline_jacobian), _readonly(surface_jacobian),
        )
        self._pose_cache[key]=result
        if len(self._pose_cache)>8:self._pose_cache.popitem(last=False)
        return result

    def surface_variations(self, positions, position_low, cell_rotations):
        """Element-owned R=Uc R0 and exact zero-increment exponential variations."""
        state=self.evaluate(positions,position_low,cell_rotations)
        rotation=np.asarray(cell_rotations)[self.cell]
        reference_frame=self._reference_frame
        arm=rotation@(self._reference_lift+self._radial_reference)
        frame=rotation@reference_frame
        frame_gradient=np.einsum('iab,bc->aci',_SPATIAL_ROTATION_GENERATORS,frame)
        frame_hessian=np.einsum('ijab,bc->acij',_SPATIAL_ROTATION_SECOND,frame)
        arm_hessian=np.einsum('ijab,b->aij',_SPATIAL_ROTATION_SECOND,arm)
        return NativeRetainedRoundSurfaceVariations(state,
            tuple(range(18+3*self.cell,21+3*self.cell)),_readonly(arm),_readonly(frame),
            _readonly(frame_gradient),_readonly(frame_hessian),_readonly(arm_hessian))
