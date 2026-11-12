"""Frozen-footprint linear Q1/S3 gap quadrature for the saved-state assessment."""

from __future__ import annotations

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


def _source_shape(point: np.ndarray, corners: np.ndarray) -> np.ndarray:
    x0, y0 = corners[0]
    y1 = corners[2, 1]
    eta = (point[1]-y0)/(y1-y0)
    left = (1-eta)*x0+eta*corners[3, 0]
    right = (1-eta)*corners[1, 0]+eta*corners[2, 0]
    xi = (point[0]-left)/(right-left)
    return np.array(((1-xi)*(1-eta), xi*(1-eta), xi*eta,
                     (1-xi)*eta))


def _target_shape(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    denominator = _cross(b-a, c-a)
    beta = _cross(point-a, c-a)/denominator
    gamma = _cross(b-a, point-a)/denominator
    return np.array((1-beta-gamma, beta, gamma))


def independent_gap_work(model, definition, source_shift_x: float,
                         target_skin_xy: dict[int, np.ndarray] | None = None,
                         source_skin_xy: dict[int, np.ndarray] | None = None):
    """Return reference support and the full-coordinate weighted gap map."""
    source_ids = sorted({node for element_id in definition.surfaces[0].element_ids
                         for node in model.mesh.elements[element_id].node_ids})
    target_ids = sorted({node for element_id in definition.surfaces[1].element_ids
                         for node in model.mesh.elements[element_id].node_ids})
    index = {node: row for row, node in enumerate(source_ids)}
    ndof = model.mesh.dof_manager.total_dofs
    support = np.zeros(len(source_ids))
    work = np.zeros((len(source_ids), ndof))
    integrated_area = 0.0
    for source_element_id in definition.surfaces[0].element_ids:
        source_element = model.mesh.elements[source_element_id]
        source_nodes = tuple(source_element.node_ids)
        source_xy = np.array([
            ((model.mesh.nodes[node].x+source_shift_x,
              model.mesh.nodes[node].y)
             if source_skin_xy is None else source_skin_xy[node])
            for node in source_nodes])
        if not (abs(source_xy[0, 1]-source_xy[1, 1]) < 1e-13
                and abs(source_xy[2, 1]-source_xy[3, 1]) < 1e-13):
            raise ValueError("oracle admits horizontal Q4 rows only")
        for target_element_id in definition.surfaces[1].element_ids:
            target_element = model.mesh.elements[target_element_id]
            target_nodes = tuple(target_element.node_ids)
            target_xy = np.array([
                ((model.mesh.nodes[node].x, model.mesh.nodes[node].y)
                 if target_skin_xy is None else target_skin_xy[node])
                for node in target_nodes])
            polygon = _clip(source_xy, target_xy)
            if len(polygon) < 3:
                continue
            for fan in range(1, len(polygon)-1):
                triangle = np.asarray((polygon[0], polygon[fan],
                                       polygon[fan+1]))
                if abs(_cross(triangle[1]-triangle[0],
                              triangle[2]-triangle[0])) < 1e-16:
                    continue
                for point, weight in _triangle_quadrature(triangle):
                    source_shape = _source_shape(point, source_xy)
                    target_shape = _target_shape(point, target_xy)
                    integrated_area += weight
                    for local_pressure, source_node in enumerate(source_nodes):
                        row = index[source_node]
                        mass = weight*source_shape[local_pressure]
                        support[row] += mass
                        for local_displacement, node in enumerate(source_nodes):
                            dof = model.mesh.dof_manager.get_node_dofs(node)[2]
                            work[row, dof] += mass*source_shape[local_displacement]
                        for local_displacement, node in enumerate(target_nodes):
                            dof = model.mesh.dof_manager.get_node_dofs(node)[2]
                            work[row, dof] -= mass*target_shape[local_displacement]
    if np.min(support) <= 0.0:
        raise ValueError("oracle needs positive support on every pressure node")
    if abs(np.sum(support)-integrated_area) > 1e-12:
        raise AssertionError("oracle partition of unity failed")
    return tuple(source_ids), support, work, integrated_area
