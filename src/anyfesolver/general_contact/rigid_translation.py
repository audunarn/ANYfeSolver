"""Immutable prescribed translation and represented-path work for fixed-normal planes."""
from dataclasses import dataclass
from bisect import bisect_right
import heapq

import numpy as np


@dataclass(frozen=True)
class ContactRigidTranslationHistory:
    """Named body offsets: ``(time_s, dx_m, dy_m, dz_m)``; affine between knots."""

    body_name: str
    points: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self):
        try:
            points = tuple(tuple(point) for point in self.points)
        except TypeError as exc:
            raise ValueError("rigid translation needs time/offset quadruples") from exc
        if (type(self.body_name) is not str or not self.body_name
                or len(points) < 2 or any(len(point) != 4 for point in points)
                or any(any(isinstance(value, (bool, np.bool_))
                           or not isinstance(value, (int, float, np.integer, np.floating))
                           or not np.isfinite(value) for value in point) for point in points)
                or points[0] != (0., 0., 0., 0.)
                or any(b[0] <= a[0] for a, b in zip(points, points[1:]))):
            raise ValueError("rigid translation needs finite increasing time/offset quadruples from zero offset at t=0")
        object.__setattr__(self, "points", tuple(tuple(float(value) for value in point) for point in points))

    def offset(self, time_s):
        if not np.isfinite(time_s) or not 0. <= time_s <= self.points[-1][0]:
            raise ValueError("rigid translation does not cover requested time")
        if time_s == self.points[-1][0]:
            return np.asarray(self.points[-1][1:])
        index = bisect_right(tuple(point[0] for point in self.points), time_s)-1
        first, last = self.points[index:index+2]
        fraction = (time_s-first[0])/(last[0]-first[0])
        return (1.-fraction)*np.asarray(first[1:])+fraction*np.asarray(last[1:])


class _PlaneTranslations:
    """Disposable manager poses; immutable reference bodies retain ownership."""

    def __init__(self, manager, histories):
        from .multiple_beam_rigid import MultipleBeamRigidManager
        self.pairs = manager.pairs if type(manager) is MultipleBeamRigidManager else ((None, manager),)
        self.histories = {history.body_name: history for history in histories}
        self.names = tuple(sorted(self.histories))
        self.references = {id(child): child.point.copy() for _, child in self.pairs}

    def point(self, child, time_s):
        name = child.definition.rigid_bodies[0].name
        history = self.histories.get(name)
        return self.references[id(child)]+(history.offset(time_s) if history is not None else 0.)

    def set_pose(self, time_s):
        for _, child in self.pairs:
            child.point = self.point(child, time_s)

    def work(self, start, end, start_velocity, end_velocity, time_s, dt, controls, check_cancel):
        """Input work ``-F_rigid dot dpoint`` on the represented Hermite path.

        Embedded 3/5-point Gauss differences are measured estimates, not certified
        continuum error bounds. One finite split budget covers all moving bodies.
        Every temporary manager pose is restored on success, failure or cancellation.
        """
        index = {name: i for i, name in enumerate(self.names)}
        moving = []
        saved = [(child, child.point.copy()) for _, child in self.pairs]
        for _, child in self.pairs:
            name = child.definition.rigid_bodies[0].name
            if name in index:
                first, last = self.point(child, time_s), self.point(child, time_s+dt)
                moving.append((child, index[name], first, last, last-first))
        rules = tuple(np.polynomial.legendre.leggauss(order) for order in (3, 5))
        remaining_support_splits = controls.maximum_beam_swept_subdivisions

        # Contact can appear/disappear at real ends or an interior spatial
        # minimum. Partition time at roots of the endpoint cubic gaps and
        # the B3 spatial discriminant. This isolates short support pulses
        # before sampling, including pulses with small nonzero Gauss values.
        cuts = {0., 1.}
        def roots(coefficients):
            scale = float(np.max(np.abs(coefficients)))
            if not np.isfinite(scale):
                raise RuntimeError("nonfinite tool-work support polynomial")
            if scale:
                values = np.polynomial.polynomial.polyroots(coefficients/scale)
                for value in values:
                    if abs(value.imag) <= 1e-10 and 0. < value.real < 1.:
                        cuts.add(float(value.real))
        for child, _, first, last, delta in moving:
            check_cancel()
            if float(child.normal @ delta) == 0.:
                continue
            for cell in child.cells:
                check_cancel()
                q0, q1 = start[cell.dofs], end[cell.dofs]
                v0, v1 = start_velocity[cell.dofs], end_velocity[cell.dofs]
                heights = np.column_stack((
                    (cell.reference+q0-first) @ child.normal-child.radius,
                    (dt*v0-delta) @ child.normal,
                    (-3*q0-2*dt*v0+3*q1-dt*v1) @ child.normal,
                    (2*q0+dt*v0-2*q1+dt*v1) @ child.normal,
                ))
                roots(heights[0])
                roots(heights[-1])
                if len(heights) == 3:
                    a = 2*(heights[0]+heights[2]-2*heights[1])
                    b = 4*heights[1]-3*heights[0]-heights[2]
                    discriminant = np.polynomial.polynomial.polysub(
                        np.polynomial.polynomial.polymul(b, b),
                        np.polynomial.polynomial.polymul(4*a, heights[0]))
                    roots(discriminant)
        cuts = sorted(cuts)
        initial_splits = len(cuts)-2
        if initial_splits > controls.maximum_tool_work_subdivisions:
            raise RuntimeError("prescribed tool work quadrature subdivision limit exhausted")

        def state(s):
            pose = ((2*s**3-3*s*s+1)*start+(s**3-2*s*s+s)*dt*start_velocity
                    +(-2*s**3+3*s*s)*end+(s**3-s*s)*dt*end_velocity)
            velocity = ((6*s*s-6*s)*start+(3*s*s-4*s+1)*dt*start_velocity
                        +(-6*s*s+6*s)*end+(3*s*s-2*s)*dt*end_velocity)/dt
            return pose, velocity

        def integrand(s):
            check_cancel()
            pose = state(s)[0]
            value = np.zeros(len(self.names))
            for child, i, first, last, delta in moving:
                if np.any(delta):
                    check_cancel()
                    child.point = (1.-s)*first+s*last
                    reaction = child.evaluate(pose, tangent=False)[2].rigid_force
                    value[i] -= float(np.asarray(reaction) @ delta)
            if not np.all(np.isfinite(value)):
                raise RuntimeError("nonfinite prescribed tool work")
            return value

        def interval(a, b):
            nonlocal remaining_support_splits
            mid, half = .5*(a+b), .5*(b-a)
            estimates = [sum((half*w*integrand(mid+half*x) for x, w in zip(nodes, weights)),
                             np.zeros(len(self.names))) for nodes, weights in rules]
            error = np.abs(estimates[1]-estimates[0])
            # Two zero rules are not evidence of no contact. A narrow support
            # pulse may lie between all quadrature points. Bound its possible
            # work, forcing subdivision unless that bound fits the work budget.
            from .beam_temporal_bounds import beam_hermite_gap_bounds
            for child, i, first, last, delta in moving:
                if estimates[0][i] == 0. and estimates[1][i] == 0. and abs(float(child.normal @ delta)) > 0.:
                    qa, va = state(a)
                    qb, vb = state(b)
                    bounds = beam_hermite_gap_bounds(child, qa, qb, va, vb, dt*(b-a),
                        tolerance_m=controls.beam_swept_gap_tolerance_m,
                        maximum_subdivisions=remaining_support_splits, check_cancel=check_cancel,
                        plane_start_point=(1.-a)*first+a*last,
                        plane_end_point=(1.-b)*first+b*last)
                    remaining_support_splits -= bounds.subdivisions
                    error[i] += (child.stiffness*sum(cell.length for cell in child.cells)
                                 *max(0., -bounds.lower_gap)*abs(float(child.normal @ delta))*(b-a))
            return estimates[1], error

        try:
            total, errors = np.zeros(len(self.names)), np.zeros(len(self.names))
            heap, serial, splits = [], 0, initial_splits
            for a, b in zip(cuts, cuts[1:]):
                value, error = interval(a, b)
                total += value
                errors += error
                heapq.heappush(heap, (-float(np.sum(error)), serial, a, b, value, error))
                serial += 1
            while float(np.sum(errors)) > (controls.tool_work_absolute_tolerance_j
                                           +controls.tool_work_relative_tolerance*float(np.sum(np.abs(total)))):
                check_cancel()
                if splits >= controls.maximum_tool_work_subdivisions:
                    raise RuntimeError("prescribed tool work quadrature subdivision limit exhausted")
                _, _, a, b, old_value, old_error = heapq.heappop(heap)
                mid = .5*(a+b)
                if mid == a or mid == b:
                    raise RuntimeError("prescribed tool work quadrature reached time resolution")
                total -= old_value
                errors -= old_error
                for lo, hi in ((a, mid), (mid, b)):
                    value, error = interval(lo, hi)
                    total += value
                    errors += error
                    heapq.heappush(heap, (-float(np.sum(error)), serial, lo, hi, value, error))
                    serial += 1
                errors = np.maximum(errors, 0.)
                splits += 1
            if not np.all(np.isfinite(total)) or not np.all(np.isfinite(errors)):
                raise RuntimeError("nonfinite prescribed tool work accumulation")
            return dict(zip(self.names, map(float, total))), dict(zip(self.names, map(float, errors)))
        finally:
            for child, point in saved:
                child.point = point
