"""Conservative beam/plane gap enclosure on a complete Hermite time path.

The displacement and velocity arguments use the manager's full DOF indexing.
Each B2/B3 centreline has tensor Bernstein degree (1/2, 3) in spatial
coordinate eta and time fraction. Exact rational arithmetic treats the supplied
binary floats as real input data; only the returned enclosure is converted to
float, with outward one-ULP padding. No sampled position supplies a lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
from numbers import Integral
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BeamHermiteGapBounds:
    lower_gap: float
    upper_gap: float
    subdivisions: int


class BeamHermiteGapBoundsError(RuntimeError):
    """The finite refinement budget cannot provide the requested enclosure."""


_Controls = tuple[tuple[Fraction, ...], ...]


def _finite_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return vector


def _finite_scalar(value, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _split_curve(values: tuple[Fraction, ...]):
    """Exact midpoint de Casteljau restriction, retaining both boundary values."""
    level = values
    left, right = [level[0]], [level[-1]]
    while len(level) > 1:
        level = tuple((a + b) / 2 for a, b in zip(level, level[1:]))
        left.append(level[0])
        right.append(level[-1])
    return tuple(left), tuple(reversed(right))


def _split(controls: _Controls, axis: int) -> tuple[_Controls, _Controls]:
    if axis == 1:
        pairs = tuple(_split_curve(row) for row in controls)
        return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    columns = tuple(zip(*controls))
    pairs = tuple(_split_curve(column) for column in columns)
    return tuple(zip(*(pair[0] for pair in pairs))), tuple(zip(*(pair[1] for pair in pairs)))


def _split_axis(controls: _Controls) -> int:
    # These local control differences measure variation on the current patch;
    # a constant coordinate is never refined at the expense of the other one.
    space = max(abs(b - a) for arow, brow in zip(controls, controls[1:])
                for a, b in zip(arow, brow))
    time = max(abs(b - a) for row in controls for a, b in zip(row, row[1:]))
    return 0 if space > time else 1


def _patch_bounds(controls: _Controls) -> tuple[Fraction, Fraction]:
    lower = min(value for row in controls for value in row)
    # Corners and centre are actual curve/time values: they provide upper
    # bounds on the minimum. The lower bound comes solely from the convex hull.
    centre_row = tuple(_split_curve(tuple(column))[0][-1]
                       for column in zip(*controls))
    centre = _split_curve(centre_row)[0][-1]
    upper = min(controls[0][0], controls[0][-1],
                controls[-1][0], controls[-1][-1], centre)
    return lower, upper


def _outward(value: Fraction, direction: float) -> float:
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise BeamHermiteGapBoundsError("gap bounds exceed finite float range") from exc
    # Padding even an exactly representable bound makes the conversion's
    # roundoff allowance explicit; subdivision itself has no rounding error.
    padded = math.nextafter(rounded, direction)
    if not math.isfinite(padded):
        raise BeamHermiteGapBoundsError("gap bounds exceed finite float range")
    return padded


def _cell_controls(cell, states, dt: Fraction, point, normal, radius, end_point=None) -> _Controls:
    reference = np.asarray(cell.reference, dtype=float)
    dofs = np.asarray(cell.dofs)
    if (reference.ndim != 2 or reference.shape not in ((2, 3), (3, 3))
            or not np.all(np.isfinite(reference))):
        raise ValueError("beam reference must be finite B2/B3 node coordinates")
    if (dofs.shape != reference.shape or not np.issubdtype(dofs.dtype, np.integer)
            or np.any(dofs < 0) or np.any(dofs >= len(states[0]))):
        raise ValueError("beam DOFs must be valid full-vector B2/B3 translation indices")
    if hasattr(cell, "node_ids") and len(cell.node_ids) != len(reference):
        raise ValueError("beam node count and reference coordinates disagree")
    nodal = []
    for coordinates, indices in zip(reference, dofs):
        time_controls = []
        for temporal in range(4):
            gap = -radius
            for axis in range(3):
                index = int(indices[axis])
                displacement = _fraction(states[0 if temporal < 2 else 1][index])
                if temporal == 1:
                    displacement += dt * _fraction(states[2][index]) / 3
                elif temporal == 2:
                    displacement -= dt * _fraction(states[3][index]) / 3
                plane = (point[axis] if end_point is None
                         else point[axis]+Fraction(temporal, 3)*(end_point[axis]-point[axis]))
                gap += (_fraction(coordinates[axis]) + displacement - plane) * normal[axis]
            time_controls.append(gap)
        nodal.append(tuple(time_controls))
    if len(nodal) == 2:
        return tuple(nodal)
    # B3 Lagrange nodes eta = 0, 1/2, 1. Its middle Bernstein
    # control is not the midpoint node: negative shape weights are retained.
    middle = tuple(2 * hm - (h0 + h1) / 2
                   for h0, hm, h1 in zip(*nodal))
    return nodal[0], middle, nodal[2]


def beam_hermite_gap_bounds(
    manager, start, end, start_velocity, end_velocity, dt,
    *, tolerance_m: float, maximum_subdivisions: int,
    check_cancel: Callable[[], object] | None = None,
    plane_start_point=None, plane_end_point=None,
) -> BeamHermiteGapBounds:
    """Enclose the minimum round-centreline gap over every beam and all time.

    ``start``/``end`` are full DOF displacements and the velocities are their
    time derivatives. ``dt`` is positive. The fixed plane uses the manager's
    point, normal and radius without changing its geometric convention.
    ``maximum_subdivisions`` counts binary patch splits across all cells.
    Insufficient budget or unrepresentable float bounds raise explicitly.
    A cancellation callback may raise its own exception or return truthy.
    Optional paired plane endpoints define affine translation with fixed normal.
    """
    states = tuple(_finite_vector(value, name) for value, name in (
        (start, "start"), (end, "end"),
        (start_velocity, "start_velocity"), (end_velocity, "end_velocity")))
    if not states[0].size or any(value.shape != states[0].shape for value in states[1:]):
        raise ValueError("full displacement and velocity vectors must have equal nonempty shape")
    step = _fraction(_finite_scalar(dt, "dt", positive=True))
    tolerance = _fraction(_finite_scalar(tolerance_m, "tolerance_m", positive=True))
    if (isinstance(maximum_subdivisions, bool)
            or not isinstance(maximum_subdivisions, Integral) or maximum_subdivisions < 0):
        raise ValueError("maximum_subdivisions must be a nonnegative integer")
    if check_cancel is not None and not callable(check_cancel):
        raise ValueError("check_cancel must be callable")
    if (plane_start_point is None) != (plane_end_point is None):
        raise ValueError("translated plane needs both start and end points")
    point_array = _finite_vector(manager.point if plane_start_point is None else plane_start_point, "plane point")
    end_array = None if plane_end_point is None else _finite_vector(plane_end_point, "plane end point")
    normal_array = _finite_vector(manager.normal, "plane normal")
    if (point_array.shape != (3,) or (end_array is not None and end_array.shape != (3,))
            or normal_array.shape != (3,) or not np.any(normal_array)):
        raise ValueError("plane point and nonzero normal must have three components")
    radius_value = _finite_scalar(manager.radius, "beam radius")
    if radius_value < 0:
        raise ValueError("beam radius must be nonnegative")
    point, normal = tuple(map(_fraction, point_array)), tuple(map(_fraction, normal_array))
    radius = _fraction(radius_value)
    end_point = None if end_array is None else tuple(map(_fraction, end_array))

    def cancelled():
        if check_cancel is not None and check_cancel():
            raise BeamHermiteGapBoundsError("beam Hermite gap enclosure cancelled")

    heap = []
    upper = None
    serial = 0
    for cell in manager.cells:
        cancelled()
        controls = _cell_controls(cell, states, step, point, normal, radius, end_point)
        lower, candidate = _patch_bounds(controls)
        upper = candidate if upper is None else min(upper, candidate)
        heapq.heappush(heap, (lower, serial, controls))
        serial += 1
    if upper is None:
        raise ValueError("beam gap enclosure requires at least one B2/B3 cell")
    subdivisions = 0
    while True:
        cancelled()
        lower = min(heap[0][0], upper) if heap else upper
        lower_float = _outward(lower, -math.inf)
        upper_float = _outward(upper, math.inf)
        if _fraction(upper_float) - _fraction(lower_float) <= tolerance:
            return BeamHermiteGapBounds(lower_float, upper_float, subdivisions)
        if lower == upper:
            raise BeamHermiteGapBoundsError("requested gap tolerance is below float roundoff enclosure")
        if subdivisions >= maximum_subdivisions:
            raise BeamHermiteGapBoundsError(
                f"beam Hermite gap enclosure unresolved after {subdivisions} subdivisions; "
                f"gap interval [{lower_float!r}, {upper_float!r}]")
        _, _, controls = heapq.heappop(heap)
        for child in _split(controls, _split_axis(controls)):
            child_lower, candidate = _patch_bounds(child)
            upper = min(upper, candidate)
            # A patch whose certified lower bound exceeds a feasible value
            # cannot contain the global minimum; no contact cell is omitted.
            if child_lower < upper:
                heapq.heappush(heap, (child_lower, serial, child))
                serial += 1
        subdivisions += 1
