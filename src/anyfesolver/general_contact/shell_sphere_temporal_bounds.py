"""Complete Q4 physical-skin/sphere gap bounds on the represented Hermite path.

The two independent Q4 coordinates and time form an exact rational tensor
Bernstein field. This encloses the declared linear skin work map and cubic
endpoint interpolation; it does not certify an intervening continuum motion.
"""

from __future__ import annotations

from fractions import Fraction
import heapq
import itertools
import math
from numbers import Integral
from typing import Callable

import numpy as np

from .beam_temporal_bounds import (
    BeamHermiteGapBounds, BeamHermiteGapBoundsError,
    _finite_scalar, _finite_vector, _fraction, _split_curve,
)
from .sphere_temporal_bounds import _gap_interval


_ScalarTensor = tuple[tuple[tuple[Fraction, ...], ...], ...]
_VectorTensor = tuple[_ScalarTensor, ...]


def _values(tensor):
    return (value for slab in tensor for row in slab for value in row)


def _shape(tensor):
    return len(tensor), len(tensor[0]), len(tensor[0][0])


def _freeze(tensor):
    return tuple(tuple(tuple(row) for row in slab) for slab in tensor)


def _split_tensor(tensor: _ScalarTensor, axis: int):
    """Exact midpoint restriction along one of the three tensor coordinates."""
    shape = _shape(tensor)
    outputs = [[[[Fraction(0) for _ in range(shape[2])]
                 for _ in range(shape[1])] for _ in range(shape[0])]
               for _ in range(2)]
    other_axes = tuple(index for index in range(3) if index != axis)
    for fixed in itertools.product(*(range(shape[index]) for index in other_axes)):
        indices = [0, 0, 0]
        for index, value in zip(other_axes, fixed):
            indices[index] = value
        curve = []
        for value in range(shape[axis]):
            indices[axis] = value
            curve.append(tensor[indices[0]][indices[1]][indices[2]])
        for side, restricted in enumerate(_split_curve(tuple(curve))):
            for value, control in enumerate(restricted):
                indices[axis] = value
                outputs[side][indices[0]][indices[1]][indices[2]] = control
    return tuple(_freeze(output) for output in outputs)


def _squared_controls(controls: _VectorTensor) -> _ScalarTensor:
    """Exact sum of squares, preserving both spatial coordinates and time."""
    degrees = tuple(size - 1 for size in _shape(controls[0]))
    result = [[[Fraction(0) for _ in range(2*degrees[2]+1)]
               for _ in range(2*degrees[1]+1)] for _ in range(2*degrees[0]+1)]
    weights = tuple(tuple(tuple(Fraction(math.comb(degree, i)*math.comb(degree, j),
                                        math.comb(2*degree, i+j))
                                for j in range(degree+1)) for i in range(degree+1))
                    for degree in degrees)
    indices = tuple(itertools.product(*(range(degree+1) for degree in degrees)))
    for lane in controls:
        for left_index, (i, j, k) in enumerate(indices):
            left = lane[i][j][k]
            for right_index in range(left_index, len(indices)):
                a, b, c = indices[right_index]
                multiplier = 1 if left_index == right_index else 2
                result[i+a][j+b][k+c] += (multiplier*left*lane[a][b][c]
                    *weights[0][i][a]*weights[1][j][b]*weights[2][k][c])
    return _freeze(result)


def _samples(tensor):
    """Actual values on the corners, edge/face centres and patch centre."""
    def sample_curve(curve):
        return curve[0], _split_curve(tuple(curve))[0][-1], curve[-1]
    time = tuple(tuple(sample_curve(row) for row in slab) for slab in tensor)
    space = tuple(tuple(sample_curve(tuple(row[k] for row in slab))
                        for k in range(3)) for slab in time)
    return tuple(value for k in range(3) for j in range(3)
                 for value in sample_curve(tuple(slab[k][j] for slab in space)))


def _squared_bounds(controls, squared):
    lower = Fraction(0)
    candidates = [Fraction(0) for _ in range(27)]
    for lane in controls:
        lo, hi = min(_values(lane)), max(_values(lane))
        nearest = lo if lo > 0 else -hi if hi < 0 else Fraction(0)
        lower += nearest*nearest
        for index, value in enumerate(_samples(lane)):
            candidates[index] += value*value
    return max(lower, min(_values(squared)), Fraction(0)), min(candidates)


def _variation(controls, axis):
    shape = _shape(controls[0])
    maximum = Fraction(0)
    for lane in controls:
        for indices in itertools.product(*(range(size) for size in shape)):
            if indices[axis]+1 == shape[axis]:
                continue
            other = list(indices)
            other[axis] += 1
            i, j, k = indices
            a, b, c = other
            maximum = max(maximum, abs(lane[i][j][k]-lane[a][b][c]))
    return maximum


def _children(controls, squared, cancelled):
    partitions = []
    for axis in range(3):
        cancelled()
        lanes = tuple(_split_tensor(lane, axis) for lane in controls)
        squared_pair = _split_tensor(squared, axis)
        children = []
        for side in (0, 1):
            child = tuple(pair[side] for pair in lanes)
            child_squared = squared_pair[side]
            lower, upper = _squared_bounds(child, child_squared)
            children.append((child, child_squared, lower, upper))
        partitions.append(tuple(children))
    # Compare the entire interval, including feasible values: optimizing only
    # the lower bound can refine time forever while an unsampled interior
    # spatial closest point keeps the upper bound distant. Exact squared
    # widths compare both improvements; local variation breaks equal widths.
    widths = tuple(min(child[3] for child in partition)-min(child[2] for child in partition)
                   for partition in partitions)
    variations = tuple(_variation(controls, axis) for axis in range(3))
    axis = min(range(3), key=lambda index: (widths[index], -variations[index]))
    lower, upper = _squared_bounds(controls, squared)
    if 16*(upper-lower-widths[axis]) < upper-lower:
        # An off-centre minimizer can require two spatial splits before any
        # sample improves. Tiny temporal gains must not starve that search.
        # Refining the largest remaining coordinate variation also ensures
        # shrinking patches when all one-step interval gains are negligible.
        axis = max(range(3), key=lambda index: variations[index])
    return partitions[axis]


def _face_controls(face, states, step, sphere):
    reference = np.asarray(face.reference, dtype=float)
    dofs = np.asarray(face.dofs)
    director = _finite_vector(face.director, "shell reference director")
    offset = _fraction(_finite_scalar(face.offset, "shell skin offset"))
    if (len(face.node_ids) != 4 or reference.shape != (4, 3)
            or not np.isfinite(reference).all()):
        raise ValueError("shell sphere enclosure requires finite Q4 reference coordinates")
    if (dofs.shape != (4, 6) or not np.issubdtype(dofs.dtype, np.integer)
            or np.any(dofs < 0) or np.any(dofs >= states[0].size)):
        raise ValueError("Q4 skin DOFs must be valid full-vector six-component indices")
    if director.shape != (3,) or not np.any(director):
        raise ValueError("shell reference director must have three nonzero components as a vector")
    points = tuple(tuple(map(_fraction, row)) for row in reference)
    # Scale exact differences before the floating reference validity check,
    # avoiding overflow for otherwise finite binary64 coordinate data.
    edges = tuple(tuple(value-origin for value, origin in zip(row, points[0]))
                  for row in points[1:])
    scale = max(abs(value) for edge in edges for value in edge)
    if scale == 0:
        raise ValueError("Q4 reference is degenerate")
    scaled = np.asarray([[float(value/scale) for value in edge] for edge in edges])
    normal = np.cross(scaled[0], scaled[2])
    measure = np.linalg.norm(normal)
    if measure == 0 or abs(float(normal @ scaled[1])) > 128*np.finfo(float).eps*measure:
        raise ValueError("shell sphere enclosure requires nondegenerate planar Q4 reference")
    # A folded/concave or collapsed boundary is outside the physical Q4 map.
    polygon = np.vstack((np.zeros(3), scaled))
    signs = [float(np.cross(polygon[(i+1)%4]-polygon[i],
                           polygon[(i+2)%4]-polygon[(i+1)%4]) @ normal)
             for i in range(4)]
    if any(value <= 0 for value in signs):
        raise ValueError("shell sphere enclosure requires a convex Q4 reference")
    d = tuple(map(_fraction, director))
    # theta cross director is the fixed small-rotation physical skin map.
    rotation = ((Fraction(0), d[2], -d[1]),
                (-d[2], Fraction(0), d[0]),
                (d[1], -d[0], Fraction(0)))
    nodal = []
    for point, indices in zip(points, dofs):
        components = []
        for component in range(3):
            times = []
            for temporal in range(4):
                base, velocity = (0, 2) if temporal < 2 else (1, 3)
                def dof_value(index):
                    value = _fraction(states[base][index])
                    if temporal in (1, 2):
                        value += (1 if temporal == 1 else -1)*step*_fraction(states[velocity][index])/3
                    return value
                value = (point[component]+offset*d[component]
                         +dof_value(int(indices[component]))-sphere[component][temporal])
                value += offset*sum(rotation[component][axis]*dof_value(int(indices[3+axis]))
                                    for axis in range(3))
                times.append(value)
            components.append(tuple(times))
        nodal.append(tuple(components))
    # Q4 node order is (-1,-1),(1,-1),(1,1),(-1,1); tensor
    # first/second axes are their independent affine [0,1] coordinates.
    return tuple(((nodal[0][axis], nodal[3][axis]),
                  (nodal[1][axis], nodal[2][axis])) for axis in range(3))


def q4_sphere_hermite_gap_bounds(
    manager, start, end, start_velocity, end_velocity, dt,
    *, sphere_start_point, sphere_end_point,
    sphere_start_velocity, sphere_end_velocity,
    tolerance_m: float, maximum_subdivisions: int,
    check_cancel: Callable[[], object] | None = None,
) -> BeamHermiteGapBounds:
    """Enclose min(norm(Q4 physical skin - sphere centre) - sphere radius).

    Every face, both full Q4 coordinates and the full cubic Hermite time
    interval are retained. Inputs are finite binary64 data interpreted exactly;
    ``dt`` scales both bodies' endpoint velocities. The supplied sphere radius
    is subtracted once, with no beam radius or additional thickness deduction.
    A single budget counts binary subdivisions across all faces. Unsupported
    geometry, failed resolution, unrepresentable reporting and cancellation
    fail explicitly. Callback exceptions propagate unchanged.

    This geometric enclosure supports the fixed linear Q4 skin map. A dynamic
    profile must separately admit its mechanics and kinematic restrictions.
    """
    states = tuple(_finite_vector(value, name) for value, name in (
        (start, "start"), (end, "end"),
        (start_velocity, "start_velocity"), (end_velocity, "end_velocity")))
    if not states[0].size or any(value.shape != states[0].shape for value in states[1:]):
        raise ValueError("full displacement and velocity vectors must have equal nonempty shape")
    if hasattr(manager, "n_dof") and states[0].size != manager.n_dof:
        raise ValueError("Q4 full vectors must match the manager DOF count")
    step = _fraction(_finite_scalar(dt, "dt", positive=True))
    tolerance = _fraction(_finite_scalar(tolerance_m, "tolerance_m", positive=True))
    if (isinstance(maximum_subdivisions, bool)
            or not isinstance(maximum_subdivisions, Integral) or maximum_subdivisions < 0):
        raise ValueError("maximum_subdivisions must be a nonnegative integer")
    if check_cancel is not None and not callable(check_cancel):
        raise ValueError("check_cancel must be callable")
    if (getattr(manager, "kind", None) != "sphere"
            or not np.array_equal(getattr(manager, "metric", None), np.eye(3))):
        raise ValueError("Q4 sphere enclosure requires an isotropic sphere manager")
    sphere_states = tuple(_finite_vector(value, name) for value, name in (
        (sphere_start_point, "sphere_start_point"), (sphere_end_point, "sphere_end_point"),
        (sphere_start_velocity, "sphere_start_velocity"),
        (sphere_end_velocity, "sphere_end_velocity")))
    if any(value.shape != (3,) for value in sphere_states):
        raise ValueError("sphere points and velocities must have three components")
    radius_value = _finite_scalar(manager.radius, "sphere radius")
    if radius_value < 0:
        raise ValueError("sphere radius must be nonnegative")
    radius = _fraction(radius_value)
    sphere = tuple((
        _fraction(sphere_states[0][axis]),
        _fraction(sphere_states[0][axis])+step*_fraction(sphere_states[2][axis])/3,
        _fraction(sphere_states[1][axis])-step*_fraction(sphere_states[3][axis])/3,
        _fraction(sphere_states[1][axis]),
    ) for axis in range(3))

    def cancelled():
        if check_cancel is not None and check_cancel():
            raise BeamHermiteGapBoundsError("Q4 sphere Hermite gap enclosure cancelled")

    heap, upper, serial = [], None, 0
    for face in manager.faces:
        cancelled()
        controls = _face_controls(face, states, step, sphere)
        squared = _squared_controls(controls)
        lower, candidate = _squared_bounds(controls, squared)
        upper = candidate if upper is None else min(upper, candidate)
        heapq.heappush(heap, (lower, serial, controls, squared))
        serial += 1
    if upper is None:
        raise ValueError("Q4 sphere gap enclosure requires at least one face")
    subdivisions = 0
    while True:
        cancelled()
        lower = min(heap[0][0], upper) if heap else upper
        lo, hi = _gap_interval(lower, upper, radius, tolerance)
        if _fraction(hi)-_fraction(lo) <= tolerance:
            return BeamHermiteGapBounds(lo, hi, subdivisions)
        if lower == upper:
            raise BeamHermiteGapBoundsError("requested Q4 sphere gap tolerance is below float roundoff enclosure")
        if subdivisions >= maximum_subdivisions:
            raise BeamHermiteGapBoundsError(
                f"Q4 sphere Hermite gap enclosure unresolved after {subdivisions} subdivisions; "
                f"gap interval [{lo!r}, {hi!r}]")
        _, _, controls, squared = heapq.heappop(heap)
        for child, child_squared, child_lower, candidate in _children(controls, squared, cancelled):
            upper = min(upper, candidate)
            if child_lower < upper:
                heapq.heappush(heap, (child_lower, serial, child, child_squared))
                serial += 1
        subdivisions += 1
