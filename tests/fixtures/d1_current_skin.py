"""Independent current-skin quadrature for the active D1 regression.

The clipping and quadrature definitions follow the September 2026 reference
oracle and current-skin comparison preserved in docs/d1_review_support.
"""
import numpy as np


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]*b[1] - a[1]*b[0])


def _clip(polygon: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Intersect a convex XY polygon with a counterclockwise triangle."""
    output = [np.asarray(point, dtype=float) for point in polygon]
    if _cross(triangle[1]-triangle[0], triangle[2]-triangle[0]) <= 0:
        raise ValueError("oracle triangle must be counterclockwise")
    for index in range(3):
        start, end = triangle[index], triangle[(index+1) % 3]
        edge = end-start
        source = output
        output = []
        if not source:
            break
        for first, second in zip(source, source[1:]+source[:1]):
            first_side = _cross(edge, first-start)
            second_side = _cross(edge, second-start)
            first_inside = first_side >= -1e-14
            second_inside = second_side >= -1e-14
            if first_inside != second_inside:
                fraction = first_side/(first_side-second_side)
                output.append(first + fraction*(second-first))
            if second_inside:
                output.append(second)
    return np.asarray(output).reshape(-1, 2)


def _triangle_quadrature(vertices: np.ndarray):
    """Integrate polynomials of sufficient degree via a tensor Gauss map."""
    roots, weights = np.polynomial.legendre.leggauss(4)
    roots = 0.5*(roots+1)
    weights = 0.5*weights
    origin, first, second = vertices
    jacobian = abs(_cross(first-origin, second-origin))
    for u, wu in zip(roots, weights):
        for v, wv in zip(roots, weights):
            point = origin+u*(first-origin)+(1-u)*v*(second-origin)
            yield point, float(wu*wv*(1-u)*jacobian)


def _target_shape(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    denominator = _cross(b-a, c-a)
    beta = _cross(point-a, c-a)/denominator
    gamma = _cross(b-a, point-a)/denominator
    return np.array((1-beta-gamma, beta, gamma))


def physical_vertices(model, element, displacement, side):
    nodes = [model.mesh.nodes[node] for node in element.node_ids]
    reference = np.asarray([(node.x, node.y, node.z) for node in nodes])
    if side == "bottom":
        assert len(nodes) == 4
        director = np.cross(reference[1]-reference[0],
                            reference[3]-reference[0])
        director /= np.linalg.norm(director)
        offset = -0.5*element.thickness
    else:
        assert side == "top" and len(nodes) == 3
        director = (np.asarray(element.reference_normal)
                    *element.director_polarity)
        offset = 0.5*element.thickness-element.reference_surface_offset
    current = []
    for node, point in zip(element.node_ids, reference, strict=True):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        current.append(point+displacement[dofs[:3]]
                       +offset*(director+np.cross(displacement[dofs[3:6]],
                                                       director)))
    return reference+offset*director, np.asarray(current)


def q4_shape(xi, eta):
    return np.array(((1-xi)*(1-eta), xi*(1-eta), xi*eta,
                     (1-xi)*eta))


def q4_derivatives(vertices, xi, eta):
    along_x = (1-eta)*(vertices[1]-vertices[0]) \
              +eta*(vertices[2]-vertices[3])
    along_y = (1-xi)*(vertices[3]-vertices[0]) \
              +xi*(vertices[2]-vertices[1])
    return along_x, along_y


def source_shape_current_xy(point, vertices):
    """Invert a convex bilinear Q4 independently in the projected plane."""
    xi = eta = 0.5
    for _ in range(12):
        shape = q4_shape(xi, eta)
        difference = shape @ vertices-point
        dx, dy = q4_derivatives(vertices, xi, eta)
        correction = np.linalg.solve(np.column_stack((dx, dy)), difference)
        xi -= float(correction[0])
        eta -= float(correction[1])
        if np.linalg.norm(correction, ord=np.inf) < 2e-14:
            break
    if (not -1e-10 <= xi <= 1+1e-10
            or not -1e-10 <= eta <= 1+1e-10
            or np.linalg.norm(q4_shape(xi, eta) @ vertices-point,
                              ord=np.inf) > 1e-11):
        raise ValueError("independent current Q4 inverse failed")
    return q4_shape(xi, eta)


def overlap_incidence(source_xy, target_xy, polygon):
    """Identify the source and target edges incident to clipped vertices.

    This is an oracle branch signature, not a physical contact-feature label.
    Cyclic normalization removes the arbitrary starting corner of the clipper
    while retaining vertex order around the overlap.
    """
    length = max(float(np.linalg.norm(np.roll(vertices, -1, axis=0)
                                      -vertices, axis=1).max())
                 for vertices in (source_xy, target_xy))
    tolerance = 1e-10*length

    def edges_at(point, vertices):
        matches = []
        for edge_id, start in enumerate(vertices):
            edge = vertices[(edge_id+1) % len(vertices)]-start
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 0:
                raise ValueError("degenerate oracle polygon edge")
            along = float(np.dot(point-start, edge))/edge_length
            distance = abs(float(np.linalg.det(np.stack((edge, point-start))))) / edge_length
            if (distance <= tolerance
                    and -tolerance <= along <= edge_length+tolerance):
                matches.append(edge_id)
        return tuple(matches)

    incidences = []
    for point in polygon:
        source_edges = edges_at(point, source_xy)
        target_edges = edges_at(point, target_xy)
        if not source_edges and not target_edges:
            raise AssertionError("clipped vertex has no parent edge")
        incidences.append((source_edges, target_edges))
    if not incidences:
        return (0, ())
    rotations = (tuple(incidences[index:]+incidences[:index])
                 for index in range(len(incidences)))
    return (len(polygon), min(rotations))


def integrate_current_skin(model, definition, pose, *, pressure_gram=False,
                           support_by_source_face=False):
    pressure_nodes = sorted({
        node for element_id in definition.surfaces[0].element_ids
        for node in model.mesh.elements[element_id].node_ids
    })
    pressure_index = {node: index for index, node in enumerate(pressure_nodes)}
    support = np.zeros(len(pressure_nodes))
    weighted_gap = np.zeros(len(pressure_nodes))
    gram = (np.zeros((len(pressure_nodes), len(pressure_nodes)))
            if pressure_gram else None)
    face_support = (np.zeros((len(definition.surfaces[0].element_ids),
                              len(pressure_nodes)))
                    if support_by_source_face else None)
    current_xy_area = 0.0
    quadrature_points = 0
    topology = []
    for source_index, source_id in enumerate(definition.surfaces[0].element_ids):
        source_element = model.mesh.elements[source_id]
        reference_source, current_source = physical_vertices(
            model, source_element, pose, "bottom")
        source_xy = current_source[:, :2]
        for target_id in definition.surfaces[1].element_ids:
            target_element = model.mesh.elements[target_id]
            _reference_target, current_target = physical_vertices(
                model, target_element, pose, "top")
            polygon = _clip(source_xy, current_target[:, :2])
            topology.append((int(source_id), int(target_id), overlap_incidence(
                source_xy, current_target[:, :2], polygon)))
            for fan in range(1, len(polygon)-1):
                triangle = polygon[[0, fan, fan+1]]
                for point, weight in _triangle_quadrature(triangle):
                    source_shape = source_shape_current_xy(point, source_xy)
                    target_shape = _target_shape(point, current_target[:, :2])
                    xi = source_shape[1]+source_shape[2]
                    eta = source_shape[2]+source_shape[3]
                    assert np.max(abs(source_shape-q4_shape(xi, eta))) < 1e-12
                    dx, dy = q4_derivatives(current_source, xi, eta)
                    rdx, rdy = q4_derivatives(reference_source, xi, eta)
                    physical_normal = np.cross(dx, dy)
                    projected_jacobian = float(physical_normal[2])
                    assert projected_jacobian > 0
                    reference_jacobian = float(np.linalg.norm(
                        np.cross(rdx, rdy)))
                    physical_normal /= np.linalg.norm(physical_normal)
                    source_point = source_shape @ current_source
                    target_point = target_shape @ current_target
                    gap = float(physical_normal @ (source_point-target_point))
                    reference_weight = (weight*reference_jacobian
                                        /projected_jacobian)
                    for local, node in enumerate(source_element.node_ids):
                        index = pressure_index[node]
                        value = reference_weight*source_shape[local]
                        support[index] += value
                        if face_support is not None:
                            face_support[source_index, index] += value
                        weighted_gap[index] += value*gap
                    if gram is not None:
                        indices = [pressure_index[node]
                                   for node in source_element.node_ids]
                        gram[np.ix_(indices, indices)] += (
                            reference_weight*np.outer(source_shape,
                                                      source_shape))
                    current_xy_area += weight
                    quadrature_points += 1
    assert np.all(support > 0)
    result = (support, weighted_gap, current_xy_area, quadrature_points,
              tuple(topology))
    if gram is not None:
        result = (*result, gram)
    if face_support is not None:
        result = (*result, face_support)
    return result
