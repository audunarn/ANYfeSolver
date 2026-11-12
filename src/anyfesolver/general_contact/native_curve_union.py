"""Conservative local search over the two physical cells of a native curve."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from math import isfinite

import numpy as np

from anyfesolver._ge_beam3_deterministic_reduction import deterministic_norm
from .beam_curve_pair import (
    _bezier_controls,
    _curve,
    _positions,
    _source_interval_target_gap_lower,
    closest_on_b3,
)


@dataclass(frozen=True)
class NativePhysicalGapCertificate:
    lower_gap_m: float
    upper_gap_m: float
    visited_cells: int


def resolved_physical_gap_lower(searches, accuracy):
    """Bound the global physical gap independently of ownership scan leaves.

    Every heap cell covers a complete target and one source interval. Its
    guarded Q2-to-chord lower is valid on that whole rectangle. Feasible
    source/target station pairs provide upper bounds, never gap certificates.
    Best-first subdivision resolves the global lower/upper bracket to the
    requested dimensional accuracy; exhausted existing budgets fail closed.
    """
    if not searches or not np.isfinite(accuracy) or accuracy <= 0.0:
        raise ValueError("native physical gap resolution must be positive finite")
    budget = min(search.maximum_cells for search in searches)
    if accuracy <= 4.0*max(search.roundoff for search in searches):
        raise MemoryError("native physical gap tolerance is below spatial evaluation floor")
    heap, serial, visited = [], 0, 0
    upper = float('inf')

    def add(search, target, left, right, depth, inherited=-float('inf')):
        nonlocal upper, serial, visited
        visited += 1
        if visited > budget:
            raise MemoryError("native physical gap search cell budget exhausted")
        lower = max(inherited, _source_interval_target_gap_lower(
            search.source, left, right, target, search.radius,
        )-search.roundoff)
        for station in (left, .5*(left+right), right):
            projection = closest_on_b3(_curve(search.source, station)[0], target)
            # A feasible station pair, with existing spatial evaluation guard.
            upper = min(upper, projection.distance-search.radius+search.roundoff)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise MemoryError("native physical gap bound is nonfinite")
        serial += 1
        heappush(heap, (lower, serial, search, target, left, right, depth))

    for search in searches:
        for target in search.targets:
            add(search, target, 0., 1., 0)
    while heap:
        lower = heap[0][0]
        if lower > upper:
            raise MemoryError("native physical gap bound ordering unresolved")
        if upper-lower <= accuracy:
            return NativePhysicalGapCertificate(lower, upper, visited)
        lower, _, search, target, left, right, depth = heappop(heap)
        if lower >= upper:
            continue
        if depth >= search.maximum_depth:
            raise MemoryError("native physical gap search depth budget exhausted")
        middle = .5*(left+right)
        if middle == left or middle == right:
            raise MemoryError("native physical gap parameter resolution exhausted")
        add(search, target, left, middle, depth+1, lower)
        add(search, target, middle, right, depth+1, lower)
        # Rectangles removed here are proved no lower than a feasible global
        # upper; they cannot contain an unaccounted smaller physical gap.
        while heap and heap[0][0] >= upper:
            heappop(heap)
    return NativePhysicalGapCertificate(upper, upper, visited)


def _owned(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeUnionNearest:
    gap_m: float
    target_cell: int
    target_station: float
    target_point_m: np.ndarray
    distances_m: tuple[float, float]
    shared_junction_tie: bool


@dataclass(frozen=True)
class NativeUnionInterval:
    left: float
    right: float
    status: str  # open, active, or unresolved
    target_cell: int | None
    lower_gap_m: float
    upper_gap_m: float


@dataclass(frozen=True)
class NativeUnionScan:
    intervals: tuple[NativeUnionInterval, ...]
    minimum_sampled_gap_m: float
    conservative_lower_gap_m: float
    visited_cells: int

    @property
    def unresolved_intervals(self) -> int:
        return sum(interval.status == "unresolved" for interval in self.intervals)


class NativeTwoCellUnionSearch:
    """Bound contact support and nearest material-cell ownership, never drop it."""

    def __init__(self, source: np.ndarray, target_cells: tuple[np.ndarray, np.ndarray],
                 target_node_ids: tuple[tuple[int, int], tuple[int, int]],
                 *, combined_radius_m: float, distance_tolerance_m: float,
                 feature_tolerance_m: float, parameter_tolerance: float,
                 maximum_depth: int, maximum_cells: int,
                 certify_open_intervals: bool = False):
        if type(certify_open_intervals) is not bool:
            raise ValueError('native union open certificate selection must be bool')
        self.certify_open_intervals = certify_open_intervals
        self.source = _owned(_positions(source, "native union source"))
        if type(target_cells) is not tuple or len(target_cells) != 2:
            raise ValueError("native union needs two ordered target half-cells")
        if (type(target_node_ids) is not tuple or len(target_node_ids) != 2
                or any(type(row) is not tuple or len(row) != 2
                       or any(type(value) is not int or value <= 0 for value in row)
                       for row in target_node_ids)
                or target_node_ids[0][1] != target_node_ids[1][0]
                or len(set((*target_node_ids[0], *target_node_ids[1]))) != 3):
            raise ValueError("native union needs one shared material junction identity")
        self.target_node_ids = target_node_ids
        self.targets = tuple(_owned(_positions(value, "native union target"))
                             for value in target_cells)
        controls = (combined_radius_m, distance_tolerance_m,
                    feature_tolerance_m, parameter_tolerance)
        if any(type(value) not in (int, float) or not isfinite(value) or value <= 0.0
               for value in controls):
            raise ValueError("native union needs positive dimensional controls")
        if parameter_tolerance >= 1.0:
            raise ValueError("native union parameter tolerance must be below one")
        if (type(maximum_depth) is not int or maximum_depth < 0
                or type(maximum_cells) is not int or maximum_cells < 1):
            raise ValueError("native union needs finite subdivision budgets")
        if (float(np.linalg.norm(self.targets[0][2]-self.targets[1][0]))
                > feature_tolerance_m):
            raise ValueError("native union target cells lack a shared physical junction")
        self.radius = float(combined_radius_m)
        self.distance_tolerance = float(distance_tolerance_m)
        self.feature_tolerance = float(feature_tolerance_m)
        self.parameter_tolerance = float(parameter_tolerance)
        self.maximum_depth = maximum_depth
        self.maximum_cells = maximum_cells
        scale = max(1.0, self.radius,
                    *(float(np.max(np.abs(value)))
                      for value in (self.source, *self.targets)))
        self.roundoff = 256.0*np.finfo(float).eps*scale

    def nearest(self, source_station: float) -> NativeUnionNearest:
        if (type(source_station) not in (int, float)
                or not isfinite(source_station)
                or not 0.0 <= source_station <= 1.0):
            raise ValueError("native union source station must lie in [0,1]")
        point = _curve(self.source, float(source_station))[0]
        projections = tuple(closest_on_b3(point, target) for target in self.targets)
        if any(not projection.unique for projection in projections):
            raise ValueError("native union has competing target minima in one cell")
        distances = tuple(float(projection.distance) for projection in projections)
        stations = tuple(projection.minimizers[0] for projection in projections)
        points = tuple(_curve(target, station)[0]
                       for target, station in zip(self.targets, stations))
        tie = abs(distances[0]-distances[1]) <= self.distance_tolerance
        if tie:
            junction = self.targets[0][2]
            shared = (float(np.linalg.norm(points[0]-junction))
                      <= self.feature_tolerance
                      and float(np.linalg.norm(points[1]-junction))
                      <= self.feature_tolerance)
            if not shared:
                raise ValueError("native union has competing remote target minima")
            owner = 0  # stable material feature ownership at the shared junction
        else:
            owner = 0 if distances[0] < distances[1] else 1
        return NativeUnionNearest(
            distances[owner]-self.radius, owner, float(stations[owner]),
            _owned(points[owner]), distances, tie,
        )

    def scan(self) -> NativeUnionScan:
        pending = [(0.0, 1.0, 0)]
        leaves: list[NativeUnionInterval] = []
        visited = 0
        sampled = float("inf")
        # A complete-source lower bound applies to every source subinterval.
        # Computing it once avoids subdividing clear, long near-parallel cells
        # solely because their tangential reach exceeds the normal clearance.
        global_lower = (min(
            _source_interval_target_gap_lower(
                self.source, 0.0, 1.0, target, self.radius,
            ) for target in self.targets
        ) if self.certify_open_intervals else -float('inf'))
        while pending:
            left, right, depth = pending.pop()
            visited += 1
            if visited > self.maximum_cells:
                raise MemoryError("native union search cell budget exhausted")
            middle = 0.5*(left+right)
            nearest = self.nearest(middle)
            sampled = min(sampled, nearest.gap_m)
            point = _curve(self.source, middle)[0]
            controls = _bezier_controls(self.source, left, right)
            reach = max(float(deterministic_norm(control-point))
                        for control in controls)
            lower = max(nearest.gap_m-reach-self.roundoff, global_lower)
            upper = nearest.gap_m+reach+self.roundoff
            separation = abs(nearest.distances_m[0]-nearest.distances_m[1])
            owner_certified = (separation > 2.0*reach
                               + self.distance_tolerance+2.0*self.roundoff)
            if (self.certify_open_intervals and lower < 0.0 and upper >= 0.0
                    and right-left <= self.parameter_tolerance):
                # Ownership is unnecessary for a certified open interval.
                # Bound against both complete target curves, never samples.
                lower = max(lower, min(
                    _source_interval_target_gap_lower(
                        self.source, left, right, target, self.radius,
                    ) for target in self.targets
                ))
            if lower >= 0.0:
                leaves.append(NativeUnionInterval(
                    left, right, "open", None, lower, upper,
                ))
            elif upper < 0.0 and owner_certified:
                leaves.append(NativeUnionInterval(
                    left, right, "active", nearest.target_cell,
                    lower, upper,
                ))
            elif right-left <= self.parameter_tolerance:
                leaves.append(NativeUnionInterval(
                    left, right, "unresolved",
                    nearest.target_cell if owner_certified else None,
                    lower, upper,
                ))
            else:
                if depth >= self.maximum_depth:
                    raise MemoryError("native union search depth budget exhausted")
                pending.extend(((left, middle, depth+1),
                                (middle, right, depth+1)))
        leaves.sort(key=lambda interval: interval.left)
        return NativeUnionScan(
            tuple(leaves), sampled,
            min(interval.lower_gap_m for interval in leaves), visited,
        )
