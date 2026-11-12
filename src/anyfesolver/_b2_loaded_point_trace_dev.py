"""Member-local compilation for the explicit fixed-end loaded-point policy.

B maps local PORT pose to constraint channels, without an offset shift. External
port wrench is -B.T*lambda. FixedEndLoadBubbleMaps owns the offset work maps;
section actions shift port moment to the axis once. Green blocks are B_i G_ij
B_j.T, including every same-member pair. No global dense operator is created.
Energy uses compiled, segmented polynomial pose samples, not recovered strains.
"""
from fractions import Fraction

import numpy as np

from ._b2_physical_flexibility_linear import (
    RepresentabilityError, _array, _float, _matmul, _multiply, _transpose,
)
from ._b2_physical_load_bubbles import (
    FixedEndLoadBubbleMaps, _axis_bubble, _axis_wrench, _poly_value,
)


def _rational(matrix):
    return tuple(tuple(Fraction(float(v)) for v in row) for row in matrix)


def _owned(array):
    value = np.asarray(array, dtype=float)
    if not np.isfinite(value).all():
        raise RepresentabilityError("loaded-point map is not representable")
    return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


def checked_sparse_action(matrix, values):
    """Retain actual stored contribution sums without an absolute unit floor."""
    rational = tuple(Fraction(float(v)) for v in values)
    result, scale = [], []
    for row in range(matrix.shape[0]):
        low, high = matrix.indptr[row:row+2]
        products = tuple(Fraction(float(coefficient))*rational[index]
                         for coefficient, index in zip(matrix.data[low:high], matrix.indices[low:high]))
        result.append(sum(products)); scale.append(sum(abs(v) for v in products))
    return _array(result), _array(scale)


class LoadedPointMember:
    """Compile once for one member; cache only requested observation maps."""

    def __init__(self, member, entries, *, body_enabled):
        self.member = member
        self.entries = tuple(entries)
        self.point_indices = tuple(entry[0] for entry in entries)
        self.stations = tuple(float(entry[1]) for entry in entries)
        self.offsets = tuple(tuple(float(v) for v in entry[2]) for entry in entries)
        self.port_constraints = tuple(_owned(entry[3]) for entry in entries)
        self._constraints = tuple(_rational(value) for value in self.port_constraints)
        self._recovery_maps = {}
        knots = (0.,) + tuple(sorted({s for s in self.stations if 0. < s < 1.})) + (1.,)
        nodes, weights = np.polynomial.legendre.leggauss(4)
        quadrature, measure = [], []
        for left, right in zip(knots, knots[1:]):
            quadrature.extend(left+(right-left)*(nodes+1.)/2.)
            measure.extend(float(member.operator.length)*(right-left)*weights/2.)
        self.quadrature_stations = tuple(map(float, quadrature))
        self.quadrature_weights = _owned(measure)
        self.maps = FixedEndLoadBubbleMaps(member.operator,
            stations=self.stations+self.quadrature_stations,
            point_stations=self.stations,
            observation_offsets_local_m=self.offsets+((0., 0., 0.),)*len(quadrature),
            point_offsets_local_m=self.offsets)
        count = len(entries)
        compliance = np.zeros((6*count, 6*count))
        for i in range(count):
            for j in range(i, count):
                green = _rational(self.maps.point_compliance[6*i:6*i+6, 6*j:6*j+6])
                block = _array(_matmul(_matmul(self._constraints[i], green),
                                       _transpose(self._constraints[j])))
                compliance[6*i:6*i+6, 6*j:6*j+6] = block
                compliance[6*j:6*j+6, 6*i:6*i+6] = block.T
        self.constraint_compliance = _owned(compliance)
        body = np.zeros((6, 3))
        if body_enabled:
            with np.errstate(over="ignore", invalid="ignore"):
                body = (member.definition.mass.matrix_at_reference_axis()[:, :3]
                        @ member.definition.local_frame_rows())
        self.body_wrench_map = _owned(body)
        self._body_wrench = _rational(body)
        constraint_body = []
        for i in range(count):
            pose = _rational(self.maps.uniform_pose_map[i])
            constraint_body.extend(_matmul(_matmul(self._constraints[i], pose), self._body_wrench))
        self._body_constraint = tuple(tuple(row) for row in constraint_body)
        self.body_constraint_map = _owned(_array(constraint_body).reshape(6*count, 3))

    def body_uniform(self, acceleration):
        if acceleration is None:
            return np.zeros(6)
        return _array(_multiply(self._body_wrench, tuple(Fraction(float(v)) for v in acceleration)))

    def body_constraint(self, acceleration):
        if acceleration is None:
            return np.zeros(6*len(self.entries)), np.zeros(6*len(self.entries))
        values = tuple(Fraction(float(v)) for v in acceleration)
        result = _array(_multiply(self._body_constraint, values))
        scale = _array(tuple(sum(abs(a*b) for a, b in zip(row, values))
                             for row in self._body_constraint))
        return result, scale

    def port_wrenches(self, multipliers):
        return np.asarray(tuple(-_array(_multiply(_transpose(matrix),
            tuple(Fraction(float(v)) for v in multipliers[6*index:6*index+6])))
            for matrix, index in zip(self._constraints, self.point_indices))).reshape(-1, 6)

    def section_actions(self, point_wrenches):
        result = []
        for station, offset, wrench in zip(self.stations, self.offsets, point_wrenches):
            if 0. < station < 1.:
                shifted = _axis_wrench(tuple(Fraction(float(v)) for v in wrench),
                                       tuple(Fraction(v) for v in offset))
                result.append((station, _array(shifted)))
        return tuple(result)

    def bubble_state(self, acceleration, multipliers):
        uniform = self.body_uniform(acceleration)
        points = self.port_wrenches(multipliers)
        response = self.maps.evaluate(uniform, point_wrenches=points)
        work = Fraction(0)
        count = len(self.entries)
        for wrench, pose in zip(points, response.pose[:count]):
            work += sum(Fraction(float(a))*Fraction(float(b)) for a, b in zip(wrench, pose))
        for weight, pose in zip(self.quadrature_weights, response.pose[count:]):
            work += Fraction(float(weight))*sum(Fraction(float(a))*Fraction(float(b))
                                               for a, b in zip(uniform, pose))
        energy = _float(work/2)
        if energy < 0.:
            raise RepresentabilityError("loaded-point bubble energy is negative")
        return uniform, points, energy

    def recover_pose(self, local, stations, uniform, point_wrenches):
        stations = tuple(float(v) for v in stations)
        operator = self.member.operator
        if stations not in self._recovery_maps:
            # Use the canonical exact bubble kernel; do not compile another Green
            # matrix or integrate the final recovered section strains each case.
            basis = tuple(tuple(Fraction(int(i == j)) for i in range(6)) for j in range(6))
            zero = (Fraction(0),)*6
            uniform_maps, point_maps = [], []
            for value in stations:
                at = Fraction(value)
                columns = tuple(_axis_bubble(operator.length, operator._compliance,
                    _multiply(operator._uniform_load_map, unit), unit, at) for unit in basis)
                uniform_maps.append(tuple(tuple(columns[c][r] for c in range(6)) for r in range(6)))
                sources = []
                for j, source in enumerate(self.stations):
                    columns = tuple(_axis_bubble(operator.length, operator._compliance,
                        tuple(self.maps._point_equivalent_exact[r][j][c] for r in range(12)),
                        zero, at, Fraction(source),
                        _axis_wrench(unit, tuple(Fraction(v) for v in self.offsets[j])))
                        for c, unit in enumerate(basis))
                    sources.append(tuple(tuple(columns[c][r] for c in range(6)) for r in range(6)))
                point_maps.append(tuple(sources))
            self._recovery_maps[stations] = (tuple(uniform_maps), tuple(point_maps))
        uniform_maps, point_maps = self._recovery_maps[stations]
        w = tuple(Fraction(float(v)) for v in uniform)
        points = tuple(tuple(Fraction(float(v)) for v in row) for row in point_wrenches)
        nodal = tuple(Fraction(float(v)) for v in local)
        poses = []
        for at, body, ports in zip(stations, uniform_maps, point_maps):
            homogeneous = tuple(sum(_poly_value(poly, Fraction(at))*value for poly, value in zip(row, nodal))
                                for row in operator._trace_polynomials)
            loaded = _multiply(body, w)
            for matrix, wrench in zip(ports, points):
                loaded = [a+b for a, b in zip(loaded, _multiply(matrix, wrench))]
            poses.append(tuple(a+b for a, b in zip(homogeneous, loaded)))
        return _array(poses).reshape(len(stations), 6)
