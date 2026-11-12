"""Opt-in geometric and work-map admission checks for the D1 interior slice.

These checks limit the scope of a development candidate. They do not establish
pressure-space stability or replace the independent physical-gap certificate.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import comb

import numpy as np

from .overlap import PlanarFace, planar_projected_overlap
from .q4_source_overlap import projected_q4_source_face


def _interval_add(a, b):
    return (np.nextafter(a[0] + b[0], -np.inf),
            np.nextafter(a[1] + b[1], np.inf))


def _interval_sub(a, b):
    return (np.nextafter(a[0] - b[1], -np.inf),
            np.nextafter(a[1] - b[0], np.inf))


def _interval_multiply(a, b):
    products = np.stack((a[0]*b[0], a[0]*b[1],
                         a[1]*b[0], a[1]*b[1]))
    return (np.nextafter(np.min(products, axis=0), -np.inf),
            np.nextafter(np.max(products, axis=0), np.inf))


def _bernstein_product(a, b):
    """Outward interval coefficients of a Bernstein polynomial product.

    Axis zero stores coefficients; other axes broadcast. All input physical
    coordinates are exact binary floats enclosed as singleton intervals.
    IEEE binary64 rounding (including coefficient weights) is enclosed after
    each elementary operation. Overflow/NaN is rejected by the caller.
    """
    degree_a, degree_b = len(a[0])-1, len(b[0])-1
    lower, upper = [], []
    for k in range(degree_a + degree_b + 1):
        total = None
        for i in range(max(0, k-degree_b), min(degree_a, k)+1):
            j = k-i
            weight = float(Fraction(comb(degree_a, i)*comb(degree_b, j),
                                    comb(degree_a+degree_b, k)))
            weight_interval = (np.nextafter(weight, -np.inf),
                               np.nextafter(weight, np.inf))
            term = _interval_multiply(
                _interval_multiply((a[0][i], a[1][i]),
                                   (b[0][j], b[1][j])), weight_interval)
            total = term if total is None else _interval_add(total, term)
        lower.append(total[0])
        upper.append(total[1])
    return np.asarray(lower), np.asarray(upper)


def _bernstein_cross(a, b):
    components = []
    for i, j in ((1, 2), (2, 0), (0, 1)):
        components.append(_interval_sub(
            _bernstein_product((a[0][..., i], a[1][..., i]),
                               (b[0][..., j], b[1][..., j])),
            _bernstein_product((a[0][..., j], a[1][..., j]),
                               (b[0][..., i], b[1][..., i]))))
    return tuple(np.stack([item[side] for item in components], axis=-1)
                 for side in (0, 1))


def _bernstein_dot(a, b):
    terms = [_bernstein_product((a[0][..., i], a[1][..., i]),
                                (b[0][..., i], b[1][..., i]))
             for i in range(3)]
    return _interval_add(_interval_add(terms[0], terms[1]), terms[2])


def _oriented_disk_boundary(indices, node_count):
    """Extract physical boundary by connectivity, never coordinate welding."""
    faces = np.asarray(indices)
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or not np.issubdtype(faces.dtype, np.integer)
            or np.min(faces) < 0 or np.max(faces) >= node_count):
        return None
    edges = {}
    links = {}
    adjacency = [set() for _ in faces]
    for face_id, face in enumerate(faces):
        if len(set(map(int, face))) != 3:
            return None
        for a, b, c in ((face[0], face[1], face[2]),
                        (face[1], face[2], face[0]),
                        (face[2], face[0], face[1])):
            a, b, c = int(a), int(b), int(c)
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b, face_id))
            links.setdefault(a, []).append((b, c))
    boundary = []
    for edge in edges.values():
        if len(edge) == 1:
            boundary.append(edge[0][:2])
        elif (len(edge) == 2 and edge[0][:2] == edge[1][1::-1]):
            adjacency[edge[0][2]].add(edge[1][2])
            adjacency[edge[1][2]].add(edge[0][2])
        else:
            return None
    if not boundary or len(links)-len(edges)+len(faces) != 1:
        return None
    reached, pending = set(), [0]
    while pending:
        face = pending.pop()
        if face not in reached:
            reached.add(face)
            pending.extend(adjacency[face]-reached)
    if len(reached) != len(faces):
        return None
    successors = dict(boundary)
    if (len(successors) != len(boundary)
            or len(set(successors.values())) != len(boundary)):
        return None
    ordered = []
    first = boundary[0][0]
    node = first
    for _ in boundary:
        if node not in successors:
            return None
        next_node = successors[node]
        ordered.append((node, next_node))
        node = next_node
        if node == first:
            break
    if node != first or len(ordered) != len(boundary):
        return None
    # Vertex links must be a connected path (rim) or cycle (interior).
    # Euler characteristic plus a single rim alone would permit pinches.
    for vertex, pairs in links.items():
        graph = {}
        for a, b in pairs:
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)
        degrees = sorted(map(len, graph.values()))
        expected = ([1, 1] + [2]*(len(graph)-2)
                    if vertex in successors else [2]*len(graph))
        if degrees != expected:
            return None
        visited, pending = set(), [next(iter(graph))]
        while pending:
            item = pending.pop()
            if item not in visited:
                visited.add(item)
                pending.extend(set(graph[item])-visited)
        if len(visited) != len(graph):
            return None
    return np.asarray(ordered, dtype=int)


def _exact_projected_winding(source, target, boundary, sign):
    """Initial boundary degree about source corner 0, with exact rationals."""
    def vector(point):
        return tuple(Fraction(float(value)) for value in point)

    def subtract(a, b):
        return tuple(x-y for x, y in zip(a, b))

    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0])

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    corners = [vector(point) for point in source]
    origin = corners[0]
    normal = cross(subtract(corners[1], origin),
                   subtract(corners[3], origin))
    axis = [Fraction(0)]*3
    axis[min(range(3), key=lambda i: abs(normal[i]))] = Fraction(1)
    vertical = cross(normal, cross(normal, axis))
    points = {int(node): vector(target[node]) for node in boundary.flat}
    winding = 0
    for a, b in boundary:
        if sign < 0:
            a, b = b, a
        pa, pb = points[int(a)], points[int(b)]
        ya = dot(subtract(pa, origin), vertical)
        yb = dot(subtract(pb, origin), vertical)
        orientation = dot(cross(subtract(pb, pa), subtract(origin, pa)), normal)
        if ya <= 0 < yb and orientation > 0:
            winding += 1
        elif yb <= 0 < ya and orientation < 0:
            winding -= 1
    return winding


def swept_projected_coverage_admissible(
        source_start, source_end, target_start, target_end,
        source_indices, target_indices) -> bool:
    """Sufficient full-union/zero-overlap certificate for a physical nodal sweep.

    The path is exactly affine in the supplied physical skin coordinates.
    For each Q4, its unnormalized projection normal is (x1-x0)x(x3-x0),
    a quadratic Bernstein polynomial. Projected source convexity, every S3
    orientation, and boundary-edge/corner containment are degree-four scalar
    polynomials. Outward interval Bernstein coefficients bound them on the
    *whole* time interval; no temporal/spatial sampling admits a path.

    A consistently oriented, connected S3 topological disk is required. For a
    covered Q4, every corner stays strictly in the inward halfplane of every
    physical target boundary edge. Q1 positions are convex combinations of
    corners, so their projections cannot cross that boundary. Exact initial
    projected boundary winding 1 remains 1 for every source point throughout
    the sweep. Positive projected S3 Jacobians and boundary degree imply one
    covering triangle almost everywhere (internal seams have zero area).
    Thus the *union* covers the entire source without duplicate positive-area
    coverage. The target need not be physically planar or globally convex;
    containment in its projected boundary kernel is sufficient.

    A zero-overlap Q4 instead requires a separating source/triangle edge for
    every target triangle throughout the sweep. Corner halfplanes suffice for
    the convex projected Q4 and P1 triangle. Different source faces may use
    either proof; transition through a positive boundary cut cannot pass.

    Ambiguous connectivity, nonfinite/overflow geometry, degeneracy, near
    grazing projection, boundary touching and inconclusive coefficient bounds
    return False. The existing caller may subdivide time; this helper does not
    narrow settings, drop pressure modes, tolerate positive cuts, or replace
    the independent physical-gap certificate. Exact rational winding anchors
    only the initial degree; coverage over time comes from polynomial bounds.
    Conditioning margins are rejection thresholds, never negative allowances.
    """
    try:
        a0, a1, b0, b1 = (np.asarray(item, dtype=float) for item in
                          (source_start, source_end, target_start, target_end))
        si, ti = np.asarray(source_indices), np.asarray(target_indices)
    except (TypeError, ValueError, OverflowError):
        return False
    if (a0.ndim != 2 or b0.ndim != 2 or a0.shape[1:] != (3,)
            or b0.shape[1:] != (3,) or a1.shape != a0.shape or b1.shape != b0.shape
            or not all(np.all(np.isfinite(item)) for item in (a0, a1, b0, b1))
            or si.ndim != 2 or si.shape[1] != 4 or len(si) == 0
            or not np.issubdtype(si.dtype, np.integer)
            or np.min(si) < 0 or np.max(si) >= len(a0)):
        return False
    boundary = _oriented_disk_boundary(ti, len(b0))
    if boundary is None:
        return False
    target = np.stack((b0, b1))
    target = (target, target)
    triangles = tuple(item[:, ti] for item in target)
    threshold = 4096.0*np.finfo(float).eps
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        target_normals = _bernstein_cross(
            _interval_sub(tuple(item[:, :, 1] for item in triangles),
                          tuple(item[:, :, 0] for item in triangles)),
            _interval_sub(tuple(item[:, :, 2] for item in triangles),
                          tuple(item[:, :, 0] for item in triangles)))
        for indices in si:
            if len(set(map(int, indices))) != 4:
                return False
            source = np.stack((a0[indices], a1[indices]))
            source = (source, source)
            corner_normals = _bernstein_cross(
                _interval_sub(tuple(item[:, (1, 2, 3, 0)] for item in source), source),
                _interval_sub(tuple(item[:, (3, 0, 1, 2)] for item in source), source))
            normal = tuple(item[:, 0] for item in corner_normals)
            normal_size = np.max(np.abs(np.stack(normal)))
            source_scale = np.max(np.abs(source[0]-source[0][:, :1]))
            normal_squared = _bernstein_dot(normal, normal)
            convex = _bernstein_dot(corner_normals,
                                    tuple(item[:, None] for item in normal))
            if (not np.all(np.isfinite(normal_squared))
                    or not np.all(np.isfinite(convex))
                    or np.min(normal_squared[0]) <= (threshold*source_scale**2)**2
                    or np.min(convex[0]) <= threshold*source_scale**4):
                return False
            orientation = _bernstein_dot(
                target_normals, tuple(item[:, None] for item in normal))
            sign = 1 if orientation[0][0, 0] > 0 else -1
            if sign < 0:
                orientation = (-orientation[1], -orientation[0])
            target_size = np.max(np.abs(np.stack(target_normals)), axis=(0, 1, 3))
            # Upper component norms give a conservative cosine conditioning
            # screen; strict positive orientation itself carries the proof.
            projection_floor = 1.0e-8*3.0*normal_size*target_size
            if (not np.all(np.isfinite(orientation))
                    or np.any(np.min(orientation[0], axis=0) <= projection_floor)):
                return False
            edge_a = tuple(item[:, boundary[:, 0], None] for item in target)
            edge_b = tuple(item[:, boundary[:, 1], None] for item in target)
            inside = _bernstein_dot(
                _bernstein_cross(_interval_sub(edge_b, edge_a),
                    _interval_sub(tuple(item[:, None] for item in source), edge_a)),
                tuple(item[:, None, None] for item in normal))
            if sign < 0:
                inside = (-inside[1], -inside[0])
            if (np.all(np.isfinite(inside))
                    and np.min(inside[0]) > threshold*normal_size**2
                    and _exact_projected_winding(a0[indices], b0, boundary, sign) == 1):
                continue
            # Sufficient zero-overlap proof. A separating edge need not be the
            # same for every triangle, but remains fixed over this time slice.
            source_a = tuple(item[:, :, None, None] for item in source)
            source_b = tuple(item[:, (1, 2, 3, 0), None, None] for item in source)
            outside_source = _bernstein_dot(
                _bernstein_cross(_interval_sub(source_b, source_a),
                    _interval_sub(tuple(item[:, None] for item in triangles), source_a)),
                tuple(item[:, None, None, None] for item in normal))
            triangle_a = tuple(item[:, :, :, None] for item in triangles)
            triangle_b = tuple(item[:, :, (1, 2, 0), None] for item in triangles)
            outside_target = _bernstein_dot(
                _bernstein_cross(_interval_sub(triangle_b, triangle_a),
                    _interval_sub(tuple(item[:, None, None] for item in source), triangle_a)),
                tuple(item[:, None, None, None] for item in normal))
            if sign < 0:
                outside_target = (-outside_target[1], -outside_target[0])
            if (not np.all(np.isfinite(outside_source))
                    or not np.all(np.isfinite(outside_target))):
                return False
            separation_floor = threshold*normal_size**2
            separated_source = np.any(np.max(outside_source[1], axis=(0, 3))
                                      < -separation_floor, axis=0)
            separated_target = np.any(np.max(outside_target[1], axis=(0, 3))
                                      < -separation_floor, axis=1)
            if not np.all(separated_source | separated_target):
                return False
    return True


def physical_q4_rim_node_ids(model, selected_faces) -> frozenset[int]:
    """Find selected nodes on a physical Q4 edge, including adjoining Q4s."""
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement

    edge_counts: Counter[tuple[int, int]] = Counter()
    for element in model.mesh.elements.values():
        if type(element) is not QualifiedE4PLShellElement:
            continue
        nodes = element.node_ids
        edge_counts.update(tuple(sorted((a, b))) for a, b in
                           zip(nodes, nodes[1:] + nodes[:1]))
    rim = set()
    for face in selected_faces:
        nodes = face.node_ids
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            if edge_counts[tuple(sorted((a, b)))] == 1:
                rim.update((a, b))
    return frozenset(rim)


def projected_coverage_issue(source: np.ndarray, target: np.ndarray,
                             source_indices, target_indices,
                             area_tolerance_m2: float) -> str | None:
    """Require every positively overlapping Q4 to be covered by S3 union."""
    for source_index, indices in enumerate(source_indices):
        face, normal = projected_q4_source_face(source[indices])
        corners = face.corners
        source_area = 0.5 * sum(float(np.cross(
            corners[j] - corners[0], corners[j + 1] - corners[0]) @ normal)
            for j in range(1, len(corners) - 1))
        if source_area <= 0.0:
            raise ValueError("D1 interior source projection has nonpositive area")
        overlap_area = 0.0
        for target_index, target_nodes in enumerate(target_indices):
            points = target[target_nodes]
            projected = points - ((points - corners[0]) @ normal)[:, None] * normal
            target_face = PlanarFace((2, target_index + 1), projected)
            overlap = planar_projected_overlap(face, target_face)
            if overlap is not None:
                overlap_area += overlap.area
        if overlap_area > source_area + area_tolerance_m2:
            return (f"source face {source_index + 1} has overlapping target "
                    f"coverage by {overlap_area - source_area:.6g} m2")
        if overlap_area > 0.0 and source_area - overlap_area > area_tolerance_m2:
            return (f"source face {source_index + 1} has partial target "
                    f"coverage; missing {source_area - overlap_area:.6g} m2")
    return None


def active_boundary_issue(certificate, source: np.ndarray,
                          source_indices, source_node_ids,
                          physical_rim_node_ids, work_map, transform,
                          force_tolerance_n: float, *, control_rows=None) -> str | None:
    """Reject active rim pressure and zero physical normal work directions."""
    pressure_ids = tuple(certificate.source_pressure_node_ids)
    pressure = np.asarray(certificate.numerical_pressure_pa, dtype=float)
    support = np.asarray(certificate.reference_support_m2, dtype=float)
    if (pressure.shape != support.shape
            or pressure.shape != (len(pressure_ids),)):
        raise ValueError("D1 interior pressure report has inconsistent modes")
    control_basis = None
    if control_rows is not None:
        rows = np.asarray(control_rows, dtype=float)
        if (rows.ndim != 2 or rows.shape[1] != transform.shape[1]
                or not np.isfinite(rows).all()):
            raise ValueError("D1 interior control rows have invalid size or values")
        if rows.shape[0] and rows.shape[1]:
            _, values, vectors = np.linalg.svd(rows, full_matrices=False)
            cutoff = max(rows.shape)*np.finfo(float).eps*values[0]
            control_basis = vectors[values > cutoff].T
    source_index = {node: index for index, node in enumerate(source_node_ids)}
    normal_sums = np.zeros_like(source)
    for indices in source_indices:
        _face, normal = projected_q4_source_face(source[indices])
        for index in indices:
            normal_sums[index] += normal
    for node, nodal_force in zip(pressure_ids, pressure * support):
        if nodal_force <= force_tolerance_n:
            continue
        if node in physical_rim_node_ids:
            return f"active pressure at physical source rim node {node} ({nodal_force:.6g} N)"
        index = source_index[node]
        normal = normal_sums[index]
        normal /= np.linalg.norm(normal)
        rows = work_map[3 * index:3 * index + 3, :]
        full_variation = np.asarray(rows.T @ normal).reshape(-1)
        free_variation = np.asarray((rows @ transform).T @ normal).reshape(-1)
        if control_basis is not None:
            # Restrict the physical normal covector to the path's actual
            # free span. A coupled control need not prescribe either node.
            free_variation -= control_basis @ (control_basis.T @ free_variation)
        numerical_zero = (256.0 * np.finfo(float).eps
                          * max(1.0, float(np.max(np.abs(full_variation)))))
        if np.max(np.abs(free_variation), initial=0.0) <= numerical_zero:
            return f"active pressure at prescribed-normal source node {node} ({nodal_force:.6g} N)"
    return None
