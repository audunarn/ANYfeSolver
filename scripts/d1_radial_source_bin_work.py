"""Saved-pose source-bin work for the flat-reference guided radial fixture.

Independent geometry integration uses projected polygon clipping, a bilinear
inverse map and tensor Gauss/Duffy quadrature. It imports neither contact
clipping nor AD for that calculation. Optional production comparison sums
source pair gradients before nodal assembly, retaining moving normalization.
Quadrature differences are estimates, not certified interval error bounds.
No target-local-bin, moment, continuum accuracy or acceptance claim follows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math

import numpy as np

from docs.d1_review_support.d1_radial_reference_observables import (
    PRINCIPAL_BIN_EDGES, q1_reference_pressure_observables,
)


@dataclass(frozen=True)
class PhysicalSkinFace:
    node_ids: tuple[int, ...]
    reference_positions: np.ndarray
    current_positions: np.ndarray

    def __post_init__(self):
        if any(not isinstance(node, (int, np.integer)) or isinstance(node, bool) or node <= 0
               for node in self.node_ids):
            raise ValueError("physical skin node IDs must be positive integers")
        object.__setattr__(self, "node_ids", tuple(map(int, self.node_ids)))
        object.__setattr__(self, "reference_positions", np.asarray(self.reference_positions, dtype=float))
        object.__setattr__(self, "current_positions", np.asarray(self.current_positions, dtype=float))


def _cross2(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _clip(subject, triangle):
    """Intersect CCW polygons without contact-library clipping dependencies."""
    polygon = list(np.asarray(subject, dtype=float))
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        result = []
        for start, end in zip(polygon, polygon[1:]+polygon[:1]):
            first, last = _cross2(b-a, start-a), _cross2(b-a, end-a)
            if (first >= 0.) != (last >= 0.):
                result.append(start+(end-start)*(first/(first-last)))
            if last >= 0.:
                result.append(end)
        polygon = result
        if len(polygon) < 3:
            return np.empty((0, 2))
    return np.asarray(polygon)


def vertical_gap_factor(source_slope, projection_slope, target_slope):
    """Exact derivative of the implemented projected gap under source +z."""
    ss, s0, st = (np.asarray(value, dtype=float) for value in
                  (source_slope, projection_slope, target_slope))
    return (1.+np.sum(ss*s0, axis=-1))/(np.sqrt(1.+np.sum(ss*ss, axis=-1))
                                      *(1.+np.sum(st*s0, axis=-1)))


def conversion_bound(source_slope_max, target_slope_max):
    """Conservative two-sided bound, including physical/projection mismatch."""
    s, t = float(source_slope_max), float(target_slope_max)
    if not (math.isfinite(s) and math.isfinite(t) and 0. <= s < 1. and t >= 0.):
        raise ValueError("source-bin slope bound requires s<1 and s*t<1")
    sf, tf = Fraction.from_float(s), Fraction.from_float(t)
    if sf*tf >= 1:
        raise ValueError("source-bin slope bound requires s<1 and s*t<1")
    # Exact arithmetic encloses the supplied binary-float slope bounds. The
    # geometric quadrature estimates below are intentionally not certified.
    cf = (1-sf*sf)/(1+sf*sf)
    cosine = _round_lower(cf)
    lower = _round_lower(cf/(1+sf*tf))
    upper = _sqrt_upper((1+sf*sf)/(1-sf*tf)**2)
    alpha = np.nextafter(max(1.-lower, upper-1.), np.inf)
    return dict(source_slope_max=s, target_slope_max=t,
                physical_projection_cosine_lower=float(cosine),
                kappa_lower=float(lower), kappa_upper=float(upper),
                relative_conversion_bound=float(alpha))


def _round_lower(value):
    result = float(value)
    while Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _sqrt_upper(value):
    result = math.sqrt(float(value))
    while Fraction.from_float(result)**2 < value:
        result = math.nextafter(result, math.inf)
    return result


def _exact_source_slope_bound(points):
    q = [[Fraction.from_float(float(value)) for value in row] for row in points]
    hx, hy = q[1][0]-q[0][0], q[3][1]-q[0][1]
    gx = ((q[1][2]-q[0][2])/hx, (q[2][2]-q[3][2])/hx)
    gy = ((q[3][2]-q[0][2])/hy, (q[2][2]-q[1][2])/hy)
    # The physical gradient is affine in material coordinates, so its norm
    # is bounded by the four corner norms. The corner projection is one of
    # those normals and requires no additional unmeasured assumption.
    return _sqrt_upper(max(x*x+y*y for x in gx for y in gy))


def _exact_target_slope_bound(points):
    q = [[Fraction.from_float(float(value)) for value in row] for row in points]
    a = [q[1][i]-q[0][i] for i in range(3)]
    b = [q[2][i]-q[0][i] for i in range(3)]
    determinant = a[0]*b[1]-a[1]*b[0]
    if determinant <= 0:
        raise ValueError("target must be an upward, nondegenerate P1 graph")
    gx, gy = (a[2]*b[1]-a[1]*b[2])/determinant, (a[0]*b[2]-a[2]*b[0])/determinant
    return _sqrt_upper(gx*gx+gy*gy)


def _source_frame(points):
    first, second = points[1]-points[0], points[3]-points[0]
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal)
    tangent = first/np.linalg.norm(first)
    return normal, tangent, np.cross(normal, tangent)


def _validate_faces(source_faces, target_faces, bin_edges):
    if tuple(bin_edges) != PRINCIPAL_BIN_EDGES:
        raise ValueError("source-bin scope requires principal edges (.1,.6,1.1)")
    if not source_faces or not target_faces:
        raise ValueError("source-bin scope needs both physical skins")
    all_nodes, cells, bins, areas = {}, set(), [], []
    eps = 256*np.finfo(float).eps
    for face_index, face in enumerate((*source_faces, *target_faces)):
        reference, current = (np.asarray(value, dtype=float) for value in
                              (face.reference_positions, face.current_positions))
        count = 4 if face_index < len(source_faces) else 3
        if (reference.shape != (count, 3) or current.shape != reference.shape
                or len(face.node_ids) != count or len(set(face.node_ids)) != count
                or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(current))):
            raise ValueError("invalid physical source/target face")
        if not np.array_equal(reference[:, :2], current[:, :2]):
            raise ValueError("guided source-bin scope requires fixed physical XY")
        for node, ref, point in zip(face.node_ids, reference, current):
            if node in all_nodes and not (np.array_equal(all_nodes[node][0], ref)
                                         and np.array_equal(all_nodes[node][1], point)):
                raise ValueError("shared physical skin node is discontinuous")
            all_nodes[node] = (ref, point)
    source_nodes = {node for face in source_faces for node in face.node_ids}
    target_nodes = {node for face in target_faces for node in face.node_ids}
    if source_nodes & target_nodes:
        raise ValueError("source and target skin nodes must be disjoint")
    xs = np.unique([all_nodes[node][0][0] for node in source_nodes])
    ys = np.unique([all_nodes[node][0][1] for node in source_nodes])
    if not (np.array_equal(xs[[0, -1]], (.1, 1.1))
            and np.array_equal(ys[[0, -1]], (.1, 1.1))):
        raise ValueError("source skin must span the radial reference rectangle")
    for face in source_faces:
        q = np.asarray(face.reference_positions)
        xa, xb, ya, yb = q[0, 0], q[1, 0], q[0, 1], q[3, 1]
        expected = np.array(((xa, ya), (xb, ya), (xb, yb), (xa, yb)))
        if (xb <= xa or yb <= ya or not np.array_equal(q[:, :2], expected)
                or np.any(q[:, 2] != q[0, 2])):
            raise ValueError("source-bin scope needs flat-reference rectangular Q4s")
        i, j = int(np.argmin(abs(xs-xa))), int(np.argmin(abs(ys-ya)))
        if (i+1 >= len(xs) or j+1 >= len(ys) or xs[i+1] != xb
                or ys[j+1] != yb or (i, j) in cells):
            raise ValueError("source Q4s do not form one complete tensor grid")
        cells.add((i, j))
        bix, biy = np.searchsorted(bin_edges, .5*(xa+xb))-1, np.searchsorted(bin_edges, .5*(ya+yb))-1
        if xa < bin_edges[bix] or xb > bin_edges[bix+1] or (
                ya < bin_edges[biy] or yb > bin_edges[biy+1]):
            raise ValueError("principal bins must contain whole source faces")
        bins.append((int(biy), int(bix)))
        areas.append(float((xb-xa)*(yb-ya)))
    if len(cells) != (len(xs)-1)*(len(ys)-1) or len(source_nodes) != len(xs)*len(ys):
        raise ValueError("source tensor grid has missing faces or duplicate nodes")
    # A target overlap cannot be traded against missing coverage. Check its
    # XY graph for positive-area overlaps independently of the area sum.
    for i, face in enumerate(target_faces):
        tri = np.asarray(face.current_positions)
        if _cross2(tri[1, :2]-tri[0, :2], tri[2, :2]-tri[0, :2]) <= 0.:
            raise ValueError("target must be an upward, nondegenerate P1 graph")
        for other in target_faces[:i]:
            xy = np.asarray(other.current_positions)[:, :2]
            if np.any(np.max(xy, axis=0) <= np.min(tri[:, :2], axis=0)) or np.any(
                    np.max(tri[:, :2], axis=0) <= np.min(xy, axis=0)):
                continue
            overlap = _clip(xy, tri[:, :2])
            if len(overlap) >= 3 and abs(np.sum(_cross2(overlap, np.roll(overlap, -1, axis=0))))/2 > eps:
                raise ValueError("target P1 faces overlap")
    return all_nodes, source_nodes, xs, ys, bins, areas


def _inverse_shape(xy, quad):
    a, b, c = quad[1]-quad[0], quad[3]-quad[0], quad[2]-quad[1]-quad[3]+quad[0]
    uv = (xy-quad[0]) @ np.linalg.inv(np.column_stack((a, b))).T
    for _ in range(20):
        u, v = uv.T
        defect = quad[0]+u[:, None]*a+v[:, None]*b+(u*v)[:, None]*c-xy
        ju, jv = a+v[:, None]*c, b+u[:, None]*c
        det = _cross2(ju, jv)
        if np.any(det <= 0.):
            raise ValueError("projected Q4 inverse has nonpositive Jacobian")
        change = np.column_stack((_cross2(defect, jv)/det, _cross2(ju, defect)/det))
        uv -= change
        if np.max(abs(change)) < 64*np.finfo(float).eps:
            break
    else:
        raise ValueError("independent projected Q4 inverse did not converge")
    if np.any(uv < -1e-10) or np.any(uv > 1.+1e-10):
        raise ValueError("independent overlap point is outside source Q4")
    u, v = uv.T
    shape = np.column_stack(((1-u)*(1-v), u*(1-v), u*v, (1-u)*v))
    return shape, u, v


def _coverage_scope(quad, triangles, target_faces):
    """Reject boundary cuts and holes rather than rounding their area away."""
    for point in quad:
        if not any(np.all(_cross2(np.roll(tri, -1, axis=0)-tri, point-tri) >= 0.)
                   for tri in triangles):
            raise ValueError("source projected corner is outside target union")
    edges = {}
    for face_index, face in enumerate(target_faces):
        for i in range(3):
            edge = tuple(sorted((face.node_ids[i], face.node_ids[(i+1) % 3])))
            edges.setdefault(edge, []).append((face_index, i))
    for uses in edges.values():
        if len(uses) > 2:
            raise ValueError("target union has a nonmanifold edge")
        if len(uses) != 1:
            continue
        face_index, i = uses[0]
        a, b = triangles[face_index][i], triangles[face_index][(i+1) % 3]
        lo, hi = 0., 1.
        for q, r in zip(quad, np.roll(quad, -1, axis=0)):
            first, last = _cross2(r-q, a-q), _cross2(r-q, b-q)
            if first <= 0. and last <= 0.:
                lo = hi
                break
            if first < 0.:
                lo = max(lo, first/(first-last))
            if last < 0.:
                hi = min(hi, first/(first-last))
        if lo < hi:
            middle = a+.5*(lo+hi)*(b-a)
            if np.all(_cross2(np.roll(quad, -1, axis=0)-quad, middle-quad) > 0.):
                raise ValueError("target boundary cuts a source face")


def _geometry_integrals(source_faces, target_faces, pressure, areas, bins,
                        force_tolerance, coverage_tolerance, maximum_triangles):
    roots = []
    smax = max(_exact_source_slope_bound(face.current_positions) for face in source_faces)
    target_normals = [np.cross(face.current_positions[1]-face.current_positions[0],
                               face.current_positions[2]-face.current_positions[0])
                      for face in target_faces]
    tmax = max(_exact_target_slope_bound(face.current_positions) for face in target_faces)
    bound = conversion_bound(smax, tmax)
    for source_index, face in enumerate(source_faces):
        q = np.asarray(face.current_positions)
        normal, tangent, binormal = _source_frame(q)
        frame = np.column_stack((tangent, binormal))
        quad = (q-q[0]) @ frame
        triangles = [(np.asarray(target.current_positions)-q[0]) @ frame
                     for target in target_faces]
        _coverage_scope(quad, triangles, target_faces)
        for tri, nt in zip(triangles, target_normals):
            if np.any(np.max(tri, axis=0) < np.min(quad, axis=0)) or np.any(
                    np.max(quad, axis=0) < np.min(tri, axis=0)):
                continue
            if nt @ normal <= 0.:
                raise ValueError("target faces away from source projection")
            polygon = _clip(quad, tri)
            for k in range(1, len(polygon)-1):
                vertices = polygon[[0, k, k+1]]
                if _cross2(vertices[1]-vertices[0], vertices[2]-vertices[0]) > 0.:
                    roots.append((source_index, vertices, q, quad, normal, nt))
    if not roots:
        raise ValueError("source-bin scope has no projected overlap")
    rules = {}
    for order in (4, 8):
        points, weights = np.polynomial.legendre.leggauss(order)
        points, weights = .5*(points+1), .5*weights
        u, v = np.meshgrid(points, points, indexing="ij")
        wu, wv = np.meshgrid(weights, weights, indexing="ij")
        rules[order] = (u.ravel(), v.ravel(), (wu*wv*(1-u)).ravel())
    values, errors = np.zeros((len(source_faces), 2)), np.zeros((len(source_faces), 2))
    visited = 0
    def integrate(root, vertices, tolerance):
        nonlocal visited
        visited += 1
        if visited > maximum_triangles:
            raise MemoryError("independent source-bin triangle budget exhausted")
        index, _, q, quad, normal, nt = root
        a, b, c = vertices
        determinant = _cross2(b-a, c-a)
        results = []
        for order in (4, 8):
            u, v, weights = rules[order]
            xy = a+u[:, None]*(b-a)+((1-u)*v)[:, None]*(c-a)
            shape, r, s = _inverse_shape(xy, quad)
            cross_term = q[2]-q[1]-q[3]+q[0]
            jr, js = q[1]-q[0]+s[:, None]*cross_term, q[3]-q[0]+r[:, None]*cross_term
            cross = np.cross(jr, js)
            jacobian = cross @ normal
            physical_normal = cross/np.linalg.norm(cross, axis=1)[:, None]
            kappa = (physical_normal @ normal)*nt[2]/(nt @ normal)
            measure = determinant*weights*areas[index]/jacobian
            p = shape @ np.array([pressure[node] for node in source_faces[index].node_ids])
            results.append(np.array((math.fsum(measure), math.fsum(measure*p*kappa))))
        error = (abs(results[1]-results[0])
                 +512*np.finfo(float).eps*np.maximum(abs(results[0]), abs(results[1])))
        if np.all(error <= tolerance):
            return results[1], error
        ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
        result = np.zeros(2); error_sum = np.zeros(2)
        for child in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)):
            part, estimate = integrate(root, np.asarray(child), tolerance/4)
            result += part; error_sum += estimate
        return result, error_sum
    tolerance = np.array((coverage_tolerance, force_tolerance))/len(roots)
    for root in roots:
        result, estimate = integrate(root, root[1], tolerance)
        values[root[0]] += result
        errors[root[0]] += estimate
    if np.any(abs(values[:, 0]-areas)+errors[:, 0] > coverage_tolerance):
        raise ValueError("target union does not wholly cover each source face")
    binned, binned_error = np.zeros((2, 2)), np.zeros((2, 2))
    for index, cell in enumerate(bins):
        binned[cell] += values[index, 1]
        binned_error[cell] += errors[index, 1]
    return binned, binned_error, bound, visited


def _production_pair_bins(geometry, source_faces, target_faces, force, bins, support,
                          coverage_tolerance):
    """Use existing Jet2 pair gradients without changing production behavior."""
    from anyfesolver.general_contact.q4_s3_overlap_derivatives import q4_s3_overlap_derivatives
    from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
    if geometry.differentiation != "jet2_q4_s3":
        raise ValueError("production comparison requires Q4/S3 Jet2 geometry")
    for declared, actual in ((geometry.source_faces, source_faces),
                             (geometry.target_faces, target_faces)):
        if len(declared) != len(actual) or any(
            a.node_ids != b.node_ids or not np.array_equal(a.reference_positions, b.reference_positions)
            for a, b in zip(declared, actual)
        ):
            raise ValueError("production geometry does not match saved physical skins")
    current = {node: point for face in (*source_faces, *target_faces)
               for node, point in zip(face.node_ids, face.current_positions)}
    source = np.array([current[node] for node in geometry.source_node_ids])
    target = np.array([current[node] for node in geometry.target_node_ids])
    state = geometry.evaluate(source, target)
    analytic_support = np.array([support[node] for node in geometry.source_node_ids])
    if np.max(abs(state.reference_support-analytic_support)) > coverage_tolerance:
        raise ValueError("production shared support differs from complete-face support")
    p = np.array([force[node] for node in geometry.source_node_ids])/state.reference_support
    g = state.reference_weighted_gap/state.reference_support
    result, normalization = np.zeros((2, 2)), np.zeros((2, 2))
    evaluations = sum(local is not None for local in state.pair_states)
    base_evaluations = evaluations
    derivative_evaluations = 0
    columns = (2, 5, 8, 11)
    for pair, local in enumerate(state.pair_states):
        if local is None or local.reference_overlap_area == 0.:
            continue
        i, j = divmod(pair, len(target_faces))
        evaluations += 1
        derivative_evaluations += 1
        geometry._check_energy_evaluation_budget(evaluations)
        tolerances = geometry.q4_source_derivative_tolerances
        derivative = q4_s3_overlap_derivatives(
            source_faces[i].reference_positions, source_faces[i].current_positions,
            target_faces[j].current_positions, allow_warped_source=True,
            area_tolerance=geometry.q4_source_area_tolerance,
            weighted_gap_tolerance=geometry.q4_source_weighted_gap_tolerance,
            maximum_triangles=geometry.maximum_q4_source_integration_triangles,
            support_gradient_tolerance=tolerances[0], support_hessian_tolerance=tolerances[1],
            weighted_gap_gradient_tolerance=tolerances[2], weighted_gap_hessian_tolerance=tolerances[3])
        nodes = geometry.source_face_indices[i]
        dw = np.sum(derivative.weighted_gap_gradient[:, columns], axis=1)
        ds = np.sum(derivative.support_gradient[:, columns], axis=1)
        term = float((p[nodes]*g[nodes]) @ ds)
        normalization[bins[i]] += term
        result[bins[i]] += float(p[nodes] @ dw)-term
    # Validate the ordinary straight Jet2 work branch as well as accessing
    # its pair gradients. Reuse the already evaluated same-pose float state;
    # charge only the additional AD passes. Unsupported seam/tiny recovery is
    # deliberately reported as a missing check, never assembled into bins.
    geometry._check_energy_evaluation_budget(evaluations+derivative_evaluations)
    normal = MovingS3PatchProjectedNormal(geometry)
    work = normal._jet2_s3_work(source, target, replace(state, pressure=p, energy=0.),
                                tangent=False, nodal_force_mode=True)
    evaluations += work.energy_evaluations-base_evaluations
    geometry._check_energy_evaluation_budget(evaluations)
    total = -float(np.sum(work.residual[:3*len(geometry.source_node_ids)].reshape(-1, 3)[:, 2]))
    return dict(status="measured", source_pair_gradient_bins_n=result.tolist(),
                moving_normalization_term_bins_n=normalization.tolist(),
                pair_energy_evaluations=evaluations, straight_jet2_work_branch_validated=True,
                validated_total_source_vertical_work_n=total,
                sum_of_bins_minus_validated_total_n=float(np.sum(result)-total))


def source_bin_geometry_work(source_faces, target_faces, nodal_contact_forces_n, *,
                             bin_edges=PRINCIPAL_BIN_EDGES, quadrature_tolerance_n,
                             maximum_triangles, coverage_tolerance_m2=1e-10,
                             production_geometry=None):
    """Check one saved physical-skin pose; returned differences are evidence."""
    source_faces, target_faces = tuple(source_faces), tuple(target_faces)
    if (not np.isfinite(quadrature_tolerance_n) or quadrature_tolerance_n <= 0.
            or not np.isfinite(coverage_tolerance_m2) or coverage_tolerance_m2 <= 0.
            or type(maximum_triangles) is not int or maximum_triangles < 1):
        raise ValueError("source-bin quadrature controls must be positive")
    nodes, source_ids, xs, ys, bins, areas = _validate_faces(source_faces, target_faces, bin_edges)
    force = dict(nodal_contact_forces_n)
    if set(force) != source_ids or any(not np.isfinite(value) or value < 0. for value in force.values()):
        raise ValueError("nonnegative nodal forces must name every source node exactly")
    support = dict.fromkeys(source_ids, 0.)
    for face, area in zip(source_faces, areas):
        for node in face.node_ids:
            support[node] += area/4
    pressure = {node: float(force[node]/support[node]) for node in source_ids}
    grid = np.full((len(ys), len(xs)), np.nan)
    for node in source_ids:
        x, y = nodes[node][0][:2]
        cell = (int(np.argmin(abs(ys-y))), int(np.argmin(abs(xs-x))))
        if np.isfinite(grid[cell]):
            raise ValueError("multiple source nodes occupy one grid point")
        grid[cell] = pressure[node]
    observable = q1_reference_pressure_observables(xs, ys, grid)
    reference_bins = np.array([[item.force_n for item in row] for row in observable.bins])
    integral, error, bound, visited = _geometry_integrals(
        source_faces, target_faces, pressure, areas, bins, quadrature_tolerance_n,
        coverage_tolerance_m2, maximum_triangles)
    result = dict(scope="guided flat-reference source whole-face vertical work bins; no target-bin or moment claim",
                  bin_edges=list(bin_edges), slope_conversion_bound=bound,
                  reference_pressure_bins_n=reference_bins.tolist(),
                  reference_pressure_estimated_error_bins_n=[
                      [item.estimated_absolute_error[0] for item in row]
                      for row in observable.bins],
                  independent_geometry_work_bins_n=integral.tolist(),
                  estimated_quadrature_error_bins_n=error.tolist(),
                  conversion_allowance_bins_n=(bound["relative_conversion_bound"]*reference_bins).tolist(),
                  visited_triangles=visited,
                  production_comparison=dict(status="not_checked", reason="production geometry not supplied"))
    if production_geometry is not None:
        try:
            comparison = _production_pair_bins(production_geometry, source_faces, target_faces,
                                               force, bins, support, coverage_tolerance_m2)
        except (ValueError, MemoryError) as exc:
            comparison = dict(status="not_checked", reason=f"{type(exc).__name__}: {exc}")
        if comparison["status"] == "measured":
            comparison["gradient_minus_independent_bins_n"] = (
                np.asarray(comparison["source_pair_gradient_bins_n"])-integral).tolist()
        result["production_comparison"] = comparison
    return result


def check_guided_source_bins(model, definition, displacement, nodal_contact_forces_n, **options):
    """Reconstruct physical skins through the installed shell work-map contract."""
    from anyfesolver.general_contact.s3_patch_static import _adapters, _selected
    source, target, _ = _selected(definition)
    adapters = _adapters(model, (source, target))
    u = np.asarray(displacement, dtype=float).reshape(-1)
    if u.size != model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
        raise ValueError("saved displacement has invalid size or values")
    for side in adapters:
        for face in side:
            for node in face.node_ids:
                dofs = model.mesh.dof_manager.get_node_dofs(node)
                if np.any(u[np.asarray(dofs)[(0, 1, 3, 4, 5),]] != 0.):
                    raise ValueError("guided source-bin scope requires fixed XY and directors")
    faces = tuple(tuple(PhysicalSkinFace(face.node_ids, face.reference_positions,
                                        face.current_positions(u)) for face in side)
                  for side in adapters)
    return source_bin_geometry_work(*faces, nodal_contact_forces_n, **options)
