"""Local complete work of a classical round line on a deformable S3 skin.

Physical coordinates only. No structural admission is established by this
kernel. The line measure is the original straight reference-cell length.
Three Gauss points integrate every fixed active polynomial branch exactly
up to arithmetic: gap and all its positional variations are degree <=2.
"""
from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, sqrt
from .beam_plane import _active_intervals, _gap_coefficients, _minimum_polynomial


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def _controls(points):
    return points if len(points) == 2 else np.array(
        [points[0], 2*points[1]-.5*(points[0]+points[2]), points[2]])


def _shape(xi, count):
    return np.array(((1-xi)/2, (1+xi)/2)) if count == 2 else np.array(
        (.5*xi*(xi-1), 1-xi*xi, .5*xi*(xi+1)))


@dataclass(frozen=True)
class BeamS3LineWork:
    potential: float
    gradient: np.ndarray
    hessian: np.ndarray
    minimum_gap: float
    minimum_rim_clearance: float
    active_line_length: float
    active_intervals: tuple
    integration_points: int


class DeformableBeamS3Line:
    def __init__(self, reference_beam, reference_skin, *, radius, line_stiffness,
                 side_normal, rim_clearance):
        beam, skin = np.asarray(reference_beam, float), np.asarray(reference_skin, float)
        normal = np.asarray(side_normal, float)
        if (beam.shape not in {(2, 3), (3, 3)} or skin.shape != (3, 3)
                or normal.shape != (3,) or not np.isfinite(np.r_[beam.ravel(), skin.ravel(), normal]).all()):
            raise ValueError("line-S3 requires finite B2/B3 coordinates and three physical skin vertices")
        values = np.array((radius, line_stiffness, rim_clearance), float)
        if not np.isfinite(values).all() or np.min(values) <= 0 or rim_clearance < radius:
            raise ValueError("line-S3 requires dimensioned positive line controls and a radius-covering rim budget")
        length = np.linalg.norm(beam[-1]-beam[0])
        if not np.isfinite(length) or length <= 0 or (len(beam) == 3 and np.linalg.norm(beam[1]-.5*(beam[0]+beam[2])) > 1.e-10*length):
            raise ValueError("classical reference curve must be a straight cell with its actual midpoint")
        C = np.cross(skin[1]-skin[0], skin[2]-skin[0])
        size = np.linalg.norm(C)
        if not np.isfinite(size) or size <= 0 or not np.isfinite(np.linalg.norm(normal)) or np.linalg.norm(normal) <= 0:
            raise ValueError("line-S3 reference face or physical side is degenerate")
        self.reference_normal = C/size
        alignment = float(self.reference_normal@normal/np.linalg.norm(normal))
        if abs(alignment) < 1-1.e-10:
            raise ValueError("physical side must align with the reference skin orientation")
        self.orientation = 1. if alignment > 0 else -1.
        self.count, self.length = len(beam), float(length)
        self.radius, self.stiffness, self.rim_clearance = map(float, values)

    def geometry(self, beam, skin):
        beam, skin = np.asarray(beam, float), np.asarray(skin, float)
        if (beam.shape != (self.count, 3) or skin.shape != (3, 3)
                or not np.isfinite(np.r_[beam.ravel(), skin.ravel()]).all()):
            raise ValueError("line-S3 candidate coordinates invalid")
        C = np.cross(skin[1]-skin[0], skin[2]-skin[0])
        area = np.linalg.norm(C)
        scale = max(np.max(np.abs(beam)), np.max(np.abs(skin)), self.radius)
        if not np.isfinite(area) or area <= 128*np.finfo(float).eps*scale*scale or C@self.reference_normal <= 0:
            raise ValueError("line-S3 face inverted, collapsed or numerically unresolved")
        geometric = C/area
        normal = self.orientation*geometric
        heights = (beam-skin[0])@normal-self.radius
        coefficients = _gap_coefficients(heights)
        rim = np.inf
        for i in range(3):
            edge = skin[(i+1)%3]-skin[i]
            size = np.linalg.norm(edge)
            if size == 0:
                raise ValueError("line-S3 physical edge collapsed")
            inward = np.cross(geometric, edge/size)
            rim = min(rim, float(np.min((_controls(beam)-skin[i])@inward)))
        return normal, coefficients, float(rim)

    def evaluate(self, beam, skin):
        beam, skin = np.asarray(beam, float), np.asarray(skin, float)
        normal, coefficients, rim = self.geometry(beam, skin)
        if rim < self.rim_clearance:
            raise ValueError("line-S3 curve approaches an unadmitted physical rim")
        intervals = _active_intervals(coefficients)
        size = 3*(self.count+3)
        coordinates = [Jet2.variable(value, i, size) for i, value in enumerate(
            np.r_[beam.ravel(), skin.ravel()])]
        source = [coordinates[3*i:3*i+3] for i in range(self.count)]
        target = [coordinates[3*(self.count+i):3*(self.count+i)+3] for i in range(3)]
        e1 = [target[1][j]-target[0][j] for j in range(3)]
        e2 = [target[2][j]-target[0][j] for j in range(3)]
        C = _cross(e1, e2)
        magnitude = sqrt(_dot(C, C))
        n = [self.orientation*value/magnitude for value in C]
        energy = Jet2.constant(0., size)
        abscissae, weights = np.polynomial.legendre.leggauss(3)
        for left, right in intervals:
            middle, half = .5*(left+right), .5*(right-left)
            for a, w in zip(abscissae, weights):
                shape = _shape(middle+half*a, self.count)
                point = [sum(N*row[j] for N, row in zip(shape, source)) for j in range(3)]
                gap = _dot(n, [point[j]-target[0][j] for j in range(3)])-self.radius
                measure = .5*self.length*half*w
                energy += .5*self.stiffness*measure*gap*gap
        if not np.isfinite(energy.value) or not np.isfinite(energy.gradient).all() or not np.isfinite(energy.hessian).all():
            raise ValueError("line-S3 work exceeds numerical admission")
        return BeamS3LineWork(energy.value, energy.gradient, energy.hessian,
                             _minimum_polynomial(coefficients), rim,
                             .5*self.length*sum(b-a for a, b in intervals),
                             intervals, 3*len(intervals))

    def swept_admissible(self, beam0, skin0, beam1, skin1, *, gap_tolerance,
                         maximum_intervals):
        """Conservative Lipschitz controls on the affine physical-coordinate path.

        Cross products have quadratic Bernstein bounds. Their positive
        projection bounds area and normal variation; affine physical edges
        similarly bound normalized-edge variation. Both gap and true rim
        distances are checked over the entire path, with a finite work budget.
        Float error allowances tighten these tests; they never enlarge a gap.
        """
        if (not np.isfinite(gap_tolerance) or gap_tolerance <= 0
                or type(maximum_intervals) is not int or maximum_intervals < 1):
            raise ValueError("line-S3 swept controls must be positive and bounded")
        beam0, skin0, beam1, skin1 = map(lambda x: np.asarray(x, float), (beam0, skin0, beam1, skin1))
        self.geometry(beam0, skin0)
        self.geometry(beam1, skin1)
        beam_rate, skin_rate = beam1-beam0, skin1-skin0
        controls0, controls1 = _controls(beam0), _controls(beam1)
        scale = max(np.max(np.abs(x)) for x in (beam0, skin0, beam1, skin1))
        error_length = np.nextafter(512*np.finfo(float).eps*max(scale, self.radius), np.inf)
        error_area = np.nextafter(4096*np.finfo(float).eps*max(scale, self.radius)**2, np.inf)
        if error_length >= gap_tolerance:
            raise ValueError("line-S3 swept gap is below coordinate resolution")
        intervals, visited = [(0., 1.)], 0
        while intervals:
            left, right = intervals.pop()
            visited += 1
            if visited > maximum_intervals:
                return False
            middle, half = .5*(left+right), .5*(right-left)
            if middle == left or middle == right:
                return False
            poses = [(beam0+t*beam_rate, skin0+t*skin_rate) for t in (left, middle, right)]
            crosses = [np.cross(s[1]-s[0], s[2]-s[0]) for _, s in poses]
            normal_length = np.linalg.norm(crosses[1])
            if normal_length <= error_area:
                return False
            n_mid = crosses[1]/normal_length
            bernstein = np.array((crosses[0], 2*crosses[1]-.5*(crosses[0]+crosses[2]), crosses[2]))
            area_lower = np.nextafter(float(np.min(bernstein@n_mid))-error_area, -np.inf)
            orientation_lower = np.nextafter(float(np.min(bernstein@self.reference_normal))-error_area, -np.inf)
            if area_lower <= 0 or orientation_lower <= 0:
                intervals.extend(((left, middle), (middle, right)))
                continue
            derivative = 2*np.diff(bernstein, axis=0)/(right-left)
            normal_rate = np.nextafter((max(np.linalg.norm(x) for x in derivative)
                                       +4*error_area/(right-left))/area_lower, np.inf)
            normal_error = np.nextafter(4*error_area/area_lower+64*np.finfo(float).eps, np.inf)
            beam, skin = poses[1]
            _, coefficients, rim = self.geometry(beam, skin)
            gap = _minimum_polynomial(coefficients)
            rel0, rel1 = controls0-skin0[0], controls1-skin1[0]
            extent = max(np.linalg.norm(x) for x in np.r_[rel0, rel1])+4*error_length
            relative_rate = max(np.linalg.norm(x) for x in (controls1-controls0-skin_rate[0]))+4*error_length
            gap_rate = np.nextafter(relative_rate+extent*normal_rate, np.inf)
            admitted = gap-half*gap_rate-8*error_length-extent*normal_error >= -gap_tolerance
            for edge_index in range(3):
                j = (edge_index+1)%3
                edges = [s[j]-s[edge_index] for _, s in poses]
                edge_size = np.linalg.norm(edges[1])
                if edge_size <= error_length:
                    admitted = False
                    break
                unit_edge = edges[1]/edge_size
                edge_lower = min(edges[0]@unit_edge, edges[2]@unit_edge)-4*error_length
                if edge_lower <= 0:
                    admitted = False
                    break
                edge_rate = (np.linalg.norm(skin_rate[j]-skin_rate[edge_index])+4*error_length)/edge_lower
                edge_error = 8*error_length/edge_lower+64*np.finfo(float).eps
                inward = np.cross(n_mid, unit_edge)
                values = (_controls(beam)-skin[edge_index])@inward
                rel0, rel1 = controls0-skin0[edge_index], controls1-skin1[edge_index]
                extent = max(np.linalg.norm(x) for x in np.r_[rel0, rel1])+4*error_length
                relative_rate = max(np.linalg.norm(x) for x in
                                    controls1-controls0-skin_rate[edge_index])+4*error_length
                bound = relative_rate+extent*(normal_rate+edge_rate)
                admitted &= np.min(values)-half*bound-8*error_length-extent*(normal_error+edge_error) >= self.rim_clearance
            if not admitted:
                if gap < -gap_tolerance or rim < self.rim_clearance:
                    return False
                intervals.extend(((left, middle), (middle, right)))
        return True
