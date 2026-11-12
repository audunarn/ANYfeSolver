"""Reassemble a saved radial linear QP state; never optimize or solve contact.

The contact map is the independent reference-XY polygon/quadrature oracle.
Results describe observed saved-state discrepancies, not a uniform nonlinear
model-error bound or a physical current-skin force/moment certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import product
import json
import math
import warnings

import numpy as np
from scipy import linalg

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from scripts.d1_reference_gap_work import (
    independent_gap_work, _source_shape, _target_shape,
)
from docs.d1_review_support.d1_radial_reference_observables import (
    q1_reference_pressure_observables,
)


SCOPE = "saved parallel-skin linear fixture; observed final-state discrepancy only"
POINTS = ((.6, .6), (.6, .85), (.6, 1.0))
EXPECTED = {
    "source_guided_shear_n_m": 1602.5641025641025,
    "target_guided_shear_n_m": 3163.400547104653,
    "target_phi_squared_range": (.39479238827866037, .3947923882786611),
    "target_cells": 12, "target_right_x_m": 1.297, "target_y0_m": -.113,
    "gap_m": 7.741263872132846e-6, "exact_force_n": .020435219974701022,
}
MAXIMUM_CORRECTION_DIMENSION = 512


def _up(value):
    return float(np.nextafter(value, np.inf))


def _gamma(count):
    value = count*np.finfo(float).eps
    if value >= .01:
        raise ValueError("saved correction arithmetic budget exceeded")
    return _up(value/(1.-value))


def _fraction_upper(value):
    result = float(value)
    return _up(result) if Fraction(result) < value else result


def _rational_clip(polygon, triangle):
    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]
    output = list(polygon)
    for start, end in zip(triangle, triangle[1:]+triangle[:1]):
        source, output = output, []
        if not source:
            break
        edge = (end[0]-start[0], end[1]-start[1])
        for first, second in zip(source, source[1:]+source[:1]):
            first_side = cross(edge, (first[0]-start[0], first[1]-start[1]))
            second_side = cross(edge, (second[0]-start[0], second[1]-start[1]))
            if (first_side >= 0) != (second_side >= 0):
                fraction = first_side/(first_side-second_side)
                output.append(tuple(first[j]+fraction*(second[j]-first[j]) for j in (0, 1)))
            if second_side >= 0:
                output.append(second)
    return output


def _triangle_moments(vertices):
    x, y = tuple(zip(*vertices))
    area = abs((x[1]-x[0])*(y[2]-y[0])-(y[1]-y[0])*(x[2]-x[0]))/2
    sx, sy = sum(x), sum(y)
    moments = {(0, 0): area, (1, 0): area*sx/3, (0, 1): area*sy/3,
               (1, 1): area*(sx*sy+sum(a*b for a, b in zip(x, y)))/12,
               (2, 0): area*(sum(a*a for a in x)+sum(x[i]*x[j] for i in range(3) for j in range(i)))/6,
               (0, 2): area*(sum(a*a for a in y)+sum(y[i]*y[j] for i in range(3) for j in range(i)))/6}
    # Integral of a degree-three barycentric product: 2A*prod(c_i!)/5!.
    for powers, values in (((2, 1), (x, x, y)), ((1, 2), (x, y, y))):
        total = Fraction(0)
        for indices in product(range(3), repeat=3):
            weight = math.prod(math.factorial(indices.count(i)) for i in range(3))
            total += weight*math.prod(values[j][indices[j]] for j in range(3))
        moments[powers] = area*total/60
    return moments


def _target_polynomials(points):
    a, b, c = points
    bx, by, cx, cy = b[0]-a[0], b[1]-a[1], c[0]-a[0], c[1]-a[1]
    determinant = bx*cy-by*cx
    if determinant <= 0:
        raise ValueError("rational audit requires counterclockwise target faces")
    beta = { (0, 0): (-a[0]*cy+a[1]*cx)/determinant,
             (1, 0): cy/determinant, (0, 1): -cx/determinant }
    gamma = { (0, 0): (-bx*a[1]+by*a[0])/determinant,
              (1, 0): -by/determinant, (0, 1): bx/determinant }
    first = {key: (int(key == (0, 0))-beta[key]-gamma[key]) for key in beta}
    return first, beta, gamma


def _audit_gap_coefficients(model, definition, prepared):
    """Exact binary-coordinate Q1/P1 coefficient audit of this frozen footprint.

    Original oracle Q1*P1 degree is three; Duffy Jacobian adds one degree.
    Four-point tensor Gauss therefore integrates it exactly mathematically.
    This rational audit encloses the *floating* clipping/integration/support
    differences, including the original oracle's tiny-polygon cutoff.
    """
    source_row = {node: i for i, node in enumerate(prepared.source_ids)}
    exact_work = {}
    exact_support = [Fraction(0) for _ in prepared.source_ids]
    targets = []
    for element_id in definition.surfaces[1].element_ids:
        nodes = tuple(model.mesh.elements[element_id].node_ids)
        points = [tuple(Fraction(float(v)) for v in (model.mesh.nodes[n].x, model.mesh.nodes[n].y)) for n in nodes]
        targets.append((nodes, points, _target_polynomials(points)))
    overlap_count = 0
    for element_id in definition.surfaces[0].element_ids:
        nodes = tuple(model.mesh.elements[element_id].node_ids)
        points = [tuple(Fraction(float(v)) for v in (model.mesh.nodes[n].x, model.mesh.nodes[n].y)) for n in nodes]
        (x0, y0), (x1, _), (_, y1), _ = points
        if points != [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] or x1 <= x0 or y1 <= y0:
            raise ValueError("rational audit requires axis-aligned Q1 rectangles")
        hx, hy, area = x1-x0, y1-y0, (x1-x0)*(y1-y0)
        bits = ((0, 0), (1, 0), (1, 1), (0, 1))
        polynomials = []
        for ix, iy in bits:
            ax, bx = ((-x0/hx, 1/hx) if ix else (x1/hx, -1/hx))
            ay, by = ((-y0/hy, 1/hy) if iy else (y1/hy, -1/hy))
            polynomials.append({(0, 0): ax*ay, (1, 0): bx*ay,
                                (0, 1): ax*by, (1, 1): bx*by})
        covered = [Fraction(0)]*4
        for i, node in enumerate(nodes):
            row = source_row[node]
            exact_support[row] += area/4
            for j, other in enumerate(nodes):
                key = row, model.mesh.dof_manager.get_node_dofs(other)[2]
                mass = area/(3 if bits[i][0] == bits[j][0] else 6)/(3 if bits[i][1] == bits[j][1] else 6)
                exact_work[key] = exact_work.get(key, Fraction(0))+mass
        for target_nodes, triangle, target_poly in targets:
            if (max(p[0] for p in triangle) < x0 or min(p[0] for p in triangle) > x1
                    or max(p[1] for p in triangle) < y0 or min(p[1] for p in triangle) > y1):
                continue
            polygon = _rational_clip(points, triangle)
            if len(polygon) < 3:
                continue
            overlap_count += 1
            for fan in range(1, len(polygon)-1):
                moments = _triangle_moments((polygon[0], polygon[fan], polygon[fan+1]))
                for i, poly in enumerate(polynomials):
                    covered[i] += sum(value*moments[key] for key, value in poly.items())
                    row = source_row[nodes[i]]
                    for j, target in enumerate(target_nodes):
                        mass = sum(a*b*moments[(pa[0]+pb[0], pa[1]+pb[1])]
                                   for pa, a in poly.items() for pb, b in target_poly[j].items())
                        key = row, model.mesh.dof_manager.get_node_dofs(target)[2]
                        exact_work[key] = exact_work.get(key, Fraction(0))-mass
        if any(value != area/4 for value in covered):
            raise ValueError("rational target-union coverage does not equal the full source face")
    support_error = np.asarray([_fraction_upper(abs(value-Fraction(float(stored))))
                               for value, stored in zip(exact_support, prepared.support, strict=True)])
    gap_error = np.zeros_like(prepared.gap_map)
    for row in range(len(exact_support)):
        for column in np.flatnonzero(prepared.gap_map[row]):
            gap_error[row, column] = abs(prepared.gap_map[row, column])
    for (row, column), value in exact_work.items():
        gap_error[row, column] = _fraction_upper(abs(value/exact_support[row]
                                               -Fraction(float(prepared.gap_map[row, column]))))
    return gap_error, support_error, dict(
        kind="exact rational coefficient differences for supplied binary XY coordinates",
        source_faces=len(definition.surfaces[0].element_ids), target_faces=len(targets),
        clipped_pairs=overlap_count, maximum_normalized_gap_coefficient_error=float(np.max(gap_error)),
        maximum_reference_support_error_m2=float(np.max(support_error)),
        quadrature_scope="Q1 times P1 degree3; Duffy integrand degree at most4; Gauss4 is exact mathematically")


def _same(name, actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if (actual.shape != expected.shape or not np.all(np.isfinite(actual))
            or not np.allclose(actual, expected, rtol=1e-12, atol=0.0)):
        raise ValueError(f"saved radial fixture identity mismatch: {name}")


def _identity(reference, metadata, saved):
    for name, expected in EXPECTED.items():
        if name not in metadata or name not in saved:
            raise ValueError(f"missing saved radial fixture identity: {name}")
        _same(name, metadata[name], expected)
        _same(name, saved[name], metadata[name])
    for name, actual in (("source_guided_shear_n_m", reference.ds),
                        ("target_guided_shear_n_m", reference.dt),
                        ("gap_m", reference.gap_m),
                        ("exact_force_n", reference.force_n)):
        _same(name, actual, metadata[name])
    _same("radial source load", reference.load_pa, .08)
    _same("radial radii", (reference.radius_m, reference.contact_radius_m), (1., .35))


@dataclass(frozen=True)
class _Prepared:
    stiffness: object
    transform: np.ndarray
    free_map: np.ndarray
    controls: np.ndarray
    source_ids: tuple
    support: np.ndarray
    gap_map: np.ndarray
    area: float


def _prepare(model, definition, program, reference):
    if (len(definition.surfaces) != 2 or definition.surfaces[0].side != "bottom"
            or definition.surfaces[1].side != "top"):
        raise ValueError("saved radial assessment requires bottom/top parallel skins")
    ids = [sorted({n for e in side.element_ids
                   for n in model.mesh.elements[e].node_ids}) for side in definition.surfaces]
    if (len(ids[0]) != 169 or len(ids[1]) != 169
            or len(definition.surfaces[0].element_ids) != 144
            or len(definition.surfaces[1].element_ids) != 288):
        raise ValueError("saved radial assessment requires the same 12/12 mesh")
    target_xy = np.asarray([(model.mesh.nodes[n].x, model.mesh.nodes[n].y) for n in ids[1]])
    _same("target geometry extrema", (np.min(target_xy[:, 1]), np.max(target_xy[:, 0])),
          (-.113, 1.297))
    heights = []
    for side_index, side in enumerate(definition.surfaces):
        local_heights = []
        for element_id in side.element_ids:
            element = model.mesh.elements[element_id]
            normal = np.array(element.reference_normal, dtype=float, copy=True)
            normal /= np.linalg.norm(normal)
            if not np.allclose(normal, (0., 0., 1.), rtol=0., atol=1e-12):
                raise ValueError("linear assessment requires flat upward reference skins")
            offset = (-.5*element.thickness if side_index == 0 else
                      .5*element.thickness-element.reference_surface_offset)
            local_heights.extend(model.mesh.nodes[n].z+offset for n in element.node_ids)
            material = model.materials[element.material_name]
            raw_shear = (5./6.)*element.thickness*material.shear_modulus
            if side_index == 0:
                _same("actual source guided shear", raw_shear, reference.ds)
            if side_index == 1:
                component = element.compute_stiffness_components(
                    model.mesh, material)
                _same("actual target phi", component["phi_squared"],
                      EXPECTED["target_phi_squared_range"][0])
                _same("actual target guided shear", component["phi_squared"]*raw_shear, reference.dt)
        if np.ptp(local_heights) > 1e-12:
            raise ValueError("linear assessment requires constant reference skin heights")
        heights.append(local_heights[0])
    if abs(heights[0]-heights[1]-reference.gap_m) > 1e-12:
        raise ValueError("actual parallel-skin reference gap differs from saved fixture")
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    size = model.mesh.dof_manager.total_dofs
    _, _, transform, offset, _, _ = build_constraint_transformation(stiffness, np.zeros(size), model)
    if np.any(offset):
        raise ValueError("saved linear assessment requires homogeneous structural constraints")
    transform = transform.toarray()
    normal_dofs = [model.mesh.dof_manager.get_node_dofs(n)[2] for side in ids for n in side]
    others = np.ones(size, dtype=bool)
    others[normal_dofs] = False
    if np.any(transform[others]):
        raise ValueError("linear assessment requires fixed XY and directors on both bodies")
    controls = np.zeros((len(program.controls), size))
    if len(program.controls) != 2:
        raise ValueError("linear assessment requires both body trace controls")
    for row, control in enumerate(program.controls):
        for node, name, coefficient in control.terms:
            if name != "uz":
                raise ValueError("linear assessment requires normal controls")
            controls[row, model.mesh.dof_manager.get_node_dofs(node)[2]] += coefficient
    reduced_controls = controls @ transform
    if np.linalg.matrix_rank(reduced_controls) != 2:
        raise ValueError("linear assessment requires independent body trace controls")
    null = linalg.null_space(reduced_controls)
    free_map = transform @ null
    # This checks the original full joint normal space, not source-only work.
    if (free_map.shape[1] != 242 or transform.shape[1] != 244
            or not all(np.any(free_map[np.asarray([
                model.mesh.dof_manager.get_node_dofs(n)[2] for n in side])]) for side in ids)):
        raise ValueError("saved radial joint normal free/control span changed")
    source_ids, support, work, area = independent_gap_work(model, definition, 0.)
    if abs(area-1.) > 1e-12:
        raise ValueError("saved radial independent contact footprint changed")
    return _Prepared(stiffness, transform, free_map, controls,
                     source_ids, support, work/support[:, None], area)


def _samples(model, definition, displacement):
    result = []
    for point in POINTS:
        values = []
        for side_index, side in enumerate(definition.surfaces):
            matches = []
            for element_id in side.element_ids:
                nodes = tuple(model.mesh.elements[element_id].node_ids)
                xy = np.asarray([(model.mesh.nodes[n].x, model.mesh.nodes[n].y) for n in nodes])
                shape = (_source_shape(np.asarray(point), xy) if side_index == 0 else
                         _target_shape(np.asarray(point), xy))
                if np.min(shape) >= -1e-12 and np.max(shape) <= 1.+1e-12:
                    uz = np.asarray([displacement[model.mesh.dof_manager.get_node_dofs(n)[2]]
                                     for n in nodes])
                    matches.append(float(shape @ uz))
            if not matches or np.ptp(matches) > 1e-12:
                raise ValueError("saved radial sample is uncovered or not continuous")
            values.append(matches[0])
        result.append({"xy_m": list(point), "source_uz_m": values[0], "target_uz_m": values[1]})
    return result


def _observables(model, prepared, z):
    positions = [(model.mesh.nodes[n].x, model.mesh.nodes[n].y) for n in prepared.source_ids]
    xs, ys = sorted({p[0] for p in positions}), sorted({p[1] for p in positions})
    if len(xs)*len(ys) != len(positions):
        raise ValueError("reference pressure requires a complete Q1 tensor grid")
    pressure = np.zeros((len(ys), len(xs)))
    for (x, y), value in zip(positions, z/prepared.support, strict=True):
        pressure[ys.index(y), xs.index(x)] = value
    return {"source_x_nodes_m": xs, "source_y_nodes_m": ys,
            "nodal_reference_pressure_pa": pressure.tolist(),
            **asdict(q1_reference_pressure_observables(xs, ys, pressure))}


def _checked_inverse(matrix):
    count = len(matrix)
    if count > MAXIMUM_CORRECTION_DIMENSION or not np.all(np.isfinite(matrix)):
        raise ValueError("active correction exceeds the finite dimension budget")
    with warnings.catch_warnings():
        warnings.simplefilter("error", linalg.LinAlgWarning)
        factor = linalg.lu_factor(matrix)
        inverse = linalg.lu_solve(factor, np.eye(count))
    residual = np.abs(np.eye(count)-matrix @ inverse)
    residual += _gamma(2*count+2)*(np.abs(matrix) @ np.abs(inverse)+np.eye(count))
    rho = _up(float(np.max(np.sum(residual, axis=1)))*(1.+_gamma(count)))
    if not np.isfinite(rho) or rho >= .01:
        raise ValueError("active KKT inverse residual is unresolved")
    row_norm = np.sum(np.abs(inverse), axis=1)*(1.+_gamma(count))
    inverse_norm = _up(float(np.max(row_norm))/(1.-rho))
    return factor, inverse, rho, inverse_norm


def _coefficient_perturbation_envelope(base, eta):
    # e <= b + |A^-1 E| e only bounds ||e|| by max(b)/(1-eta).
    # Each component must allow transfer from the largest component of b.
    return base+eta*float(np.max(base, initial=0.))/(1.-eta)


def _fixed_active_correction(prepared, u, z, external, targets, reference, gap_error):
    active = z > 0.
    reduced_controls = prepared.controls @ prepared.transform
    coordinates = linalg.lstsq(prepared.transform, u)[0]
    coordinates += reduced_controls.T @ np.linalg.solve(
        reduced_controls @ reduced_controls.T, targets-reduced_controls @ coordinates)
    base = prepared.transform @ coordinates
    free = prepared.free_map
    stiffness = prepared.stiffness
    k = np.asarray(free.T @ stiffness @ free)
    g = prepared.gap_map @ free
    count, modes = k.shape[0], int(np.count_nonzero(active))
    matrix = np.block([[k*reference.gap_m/reference.load_pa, -g[active].T],
                       [g[active], np.zeros((modes, modes))]])
    factor, inverse, rho, inverse_norm = _checked_inverse(matrix)
    stationarity = free.T @ (stiffness @ base-external-prepared.gap_map.T @ z)
    active_gap = (reference.gap_m+prepared.gap_map @ base)[active]
    rhs = np.r_[stationarity/reference.load_pa, active_gap/reference.gap_m]
    delta = linalg.lu_solve(factor, -rhs)
    corrected_u = base+free @ (reference.gap_m*delta[:count])
    corrected_z = z.copy()
    corrected_z[active] += reference.load_pa*delta[count:]
    corrected_stationarity = free.T @ (stiffness @ corrected_u-external-prepared.gap_map.T @ corrected_z)
    corrected_gap = reference.gap_m+prepared.gap_map @ corrected_u
    # Bounds below treat admitted structural K and basis floats as the frozen
    # operator. Exact G/support audit supplies contact coefficient allowances.
    size = len(u)
    absolute_free = np.abs(free)
    k_product = np.asarray(absolute_free.T @ abs(stiffness) @ absolute_free)
    g_product = np.abs(prepared.gap_map) @ absolute_free
    k_error = _gamma(4*size)*k_product
    g_error = gap_error @ absolute_free+_gamma(2*size)*g_product
    coefficient_error = np.block([[k_error*reference.gap_m/reference.load_pa, g_error[active].T],
                                   [g_error[active], np.zeros((modes, modes))]])
    coefficient_norm = _up(float(np.max(np.sum(coefficient_error, axis=1)))*(1.+_gamma(count+modes)))
    perturbation = _up(inverse_norm*coefficient_norm)
    if not np.isfinite(perturbation) or perturbation >= .1:
        raise ValueError("active KKT coefficient uncertainty is unresolved")
    magnitude = (abs(stiffness) @ np.abs(corrected_u)+np.abs(external)
                 +np.abs(prepared.gap_map).T @ np.abs(corrected_z))
    force_error = (_gamma(4*size)*magnitude+gap_error.T @ np.abs(corrected_z))
    stationarity_error = (absolute_free.T @ force_error
                          +_gamma(2*size)*(absolute_free.T @ magnitude))
    gap_evaluation_error = (gap_error @ np.abs(corrected_u)
        +_gamma(2*size+2)*(abs(reference.gap_m)+np.abs(prepared.gap_map) @ np.abs(corrected_u)))
    residual_upper = np.r_[
        (np.abs(corrected_stationarity)+stationarity_error)/reference.load_pa,
        (np.abs(corrected_gap[active])+gap_evaluation_error[active])/reference.gap_m]
    # A^-1=B(I-R)^-1, ||R||<rho. This componentwise first term and row-norm
    # remainder bound propagate through the same checked inverse; coefficient
    # perturbation permits transfer between components via its infinity norm.
    row_norm = np.sum(np.abs(inverse), axis=1)*(1.+_gamma(count+modes))
    base_error = (np.abs(inverse) @ residual_upper
                  +row_norm*rho/(1.-rho)*float(np.max(residual_upper, initial=0.)))
    error = _coefficient_perturbation_envelope(base_error, perturbation)
    error = np.nextafter(error*(1.+_gamma(2*(count+modes)+8)), np.inf)
    u_error = absolute_free @ (reference.gap_m*error[:count])
    u_error += _gamma(4*size)*(np.abs(base)+absolute_free @ np.abs(reference.gap_m*delta[:count]))
    z_error = np.zeros_like(z)
    z_error[active] = reference.load_pa*error[count:]
    gap_sign_error = gap_evaluation_error+(np.abs(prepared.gap_map)+gap_error) @ u_error
    if (np.any(corrected_z[active]-z_error[active] <= 0.)
            or np.any(corrected_gap[~active]-gap_sign_error[~active] <= 0.)
            or np.any(z[~active] != 0.) or not np.all(np.isfinite(error))):
        raise ValueError("corrected active branch has unresolved pressure or inactive-gap signs")
    scaled_residual = np.r_[corrected_stationarity/reference.load_pa, corrected_gap[active]/reference.gap_m]
    if np.max(np.abs(scaled_residual), initial=0.) > 1e-8:
        raise ValueError("active correction remaining residual exceeds its verification budget")
    return corrected_u, corrected_z, u_error, z_error, dict(
        active_modes=modes, scaled_kkt_dimension=count+modes,
        maximum_dimension=MAXIMUM_CORRECTION_DIMENSION,
        inverse_residual_infinity_bound=rho, inverse_norm_infinity_bound=inverse_norm,
        scaled_condition_infinity_estimate=float(np.linalg.norm(matrix, np.inf)*np.linalg.norm(inverse, np.inf)),
        coefficient_perturbation_bound=perturbation,
        scaled_remaining_residual_infinity=float(np.max(abs(scaled_residual), initial=0.)),
        maximum_active_gap_m=float(np.max(abs(corrected_gap[active]), initial=0.)),
        minimum_inactive_gap_margin_m=float(np.min(corrected_gap[~active]-gap_sign_error[~active])) if np.any(~active) else None,
        minimum_active_force_margin_n=float(np.min(corrected_z[active]-z_error[active])) if modes else None,
        correction_displacement_infinity_m=float(np.max(abs(corrected_u-u), initial=0.)),
        correction_force_l1_n=float(np.sum(abs(corrected_z-z))),
        remaining_displacement_infinity_m=float(np.max(u_error, initial=0.)),
        remaining_nodal_force_l1_n=float(np.sum(z_error)),
        branch="original strictly positive z modes only; no active-set changes",
        coefficient_perturbation_propagation="b_i + eta*max(b)/(1-eta); includes component coupling",
        inactive_gap_propagation="coefficient envelope (abs(G)+gap_error) times displacement envelope",
        arithmetic_scope="directed coefficient audit plus gamma floating arithmetic allowances; fixed admitted structural K")


def _solution_uncertainty(model, definition, program, reference, prepared, u, z, external, observable):
    gap_error, support_error, audit = _audit_gap_coefficients(model, definition, prepared)
    corrected_u, corrected_z, u_error, z_error, diagnostics = _fixed_active_correction(
        prepared, u, z, external, np.asarray(program.target_at(2)), reference, gap_error)
    floor = prepared.support-support_error
    if np.any(floor <= 0.):
        raise ValueError("reference pressure support uncertainty is unresolved")
    # Convert z/support error to the original Q1 pressure field, preserving
    # original observed values. Signed nodal changes are bounded in magnitude.
    equivalent_z_error = (abs(corrected_z-z)+z_error*prepared.support/floor
                          +abs(corrected_z)*support_error/floor+_gamma(4)*abs(z))
    equivalent_z_error = np.nextafter(equivalent_z_error*(1.+_gamma(8)), np.inf)
    envelope = _observables(model, prepared, equivalent_z_error)
    total = envelope["total"]
    force = _up(total["force_n"]+total["estimated_absolute_error"][0]
                +observable["total"]["estimated_absolute_error"][0])
    bins = [[_up(cell["force_n"]+cell["estimated_absolute_error"][0]
                 +observable["bins"][j][i]["estimated_absolute_error"][0])
             for i, cell in enumerate(row)] for j, row in enumerate(envelope["bins"])]
    moments = [_up(sum(abs(cell["moment_nm"][axis])+cell["estimated_moment_error_nm"][axis]
                       for row in envelope["bins"] for cell in row)
                   +observable["total"]["estimated_moment_error_nm"][axis]) for axis in (0, 1)] + [0.]
    old_samples = _samples(model, definition, u)
    sample_errors = _sample_uncertainties(model, definition, old_samples, corrected_u, u_error)
    resultant_error = None
    old = observable["total"]
    if old["resultant_xy_m"] is not None:
        denominator = old["force_n"]-force
        if denominator <= 0.:
            raise ValueError("moment-equivalent resultant denominator is unresolved")
        first_error = [total["first_x_nm"]+total["estimated_absolute_error"][1]+old["estimated_absolute_error"][1],
                       total["first_y_nm"]+total["estimated_absolute_error"][2]+old["estimated_absolute_error"][2]]
        resultant_error = _up(float(np.linalg.norm([
            (e+abs(r)*force)/denominator for e, r in zip(first_error, old["resultant_xy_m"], strict=True)])))
    return dict(assessed=True, diagnostics=diagnostics, coefficient_audit=audit,
        force_solution_uncertainty_n=force, bin_solution_uncertainty_n=bins,
        moment_solution_uncertainty_nm=moments, resultant_solution_uncertainty_m=resultant_error,
        sample_uncertainty_m=sample_errors)


def _sample_uncertainties(model, definition, old_samples, corrected_u, u_error):
    """Exact frozen-cell basis audit includes sample interpolation rounding."""
    result = []
    for side_index, side in enumerate(definition.surfaces):
        errors = []
        for sample in old_samples:
            point = tuple(Fraction(float(v)) for v in sample["xy_m"])
            matches = []
            for element_id in side.element_ids:
                nodes = tuple(model.mesh.elements[element_id].node_ids)
                xy = [tuple(Fraction(float(v)) for v in (model.mesh.nodes[n].x, model.mesh.nodes[n].y)) for n in nodes]
                if side_index == 0:
                    xi = (point[0]-xy[0][0])/(xy[1][0]-xy[0][0])
                    eta = (point[1]-xy[0][1])/(xy[3][1]-xy[0][1])
                    shape = ((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta)
                else:
                    shape = tuple(sum(c*point[0]**p[0]*point[1]**p[1] for p, c in poly.items())
                                  for poly in _target_polynomials(xy))
                if min(shape) < 0 or max(shape) > 1:
                    continue
                dofs = [model.mesh.dof_manager.get_node_dofs(n)[2] for n in nodes]
                corrected = sum(w*Fraction(float(corrected_u[d])) for w, d in zip(shape, dofs, strict=True))
                envelope = sum(abs(w)*Fraction(float(u_error[d])) for w, d in zip(shape, dofs, strict=True))
                old = Fraction(float(sample["source_uz_m" if side_index == 0 else "target_uz_m"]))
                matches.append(abs(old-corrected)+envelope)
            if not matches:
                raise ValueError("rational sample basis is uncovered")
            errors.append(_fraction_upper(max(matches)))
        result.append(errors)
    return result


def assess_saved_radial_linear(model, definition, program, reference, fixture_metadata, saved_record):
    """Evaluate saved steps using the original linear KKT gates, without solving.

    Legacy QP records store z in independent sorted source-node order; an
    explicit source_node_ids field, when present, must match that order.
    Missing/nonfinite states or changed fixture identity are rejected.
    """
    _identity(reference, fixture_metadata, saved_record)
    rows = saved_record.get("steps")
    if not isinstance(rows, list) or [r.get("step") for r in rows] != [1, 2]:
        raise ValueError("saved assessment requires complete steps 1 and 2")
    prepared = _prepare(model, definition, program, reference)
    if ("source_node_ids" in saved_record
            and tuple(saved_record["source_node_ids"]) != prepared.source_ids):
        raise ValueError("saved nodal-force order differs from independent source IDs")
    size = model.mesh.dof_manager.total_dofs
    records = []
    for row in rows:
        if "displacements" not in row or "nodal_force_n" not in row:
            raise ValueError("missing saved displacement or nodal-force state")
        u, z = np.asarray(row["displacements"], dtype=float), np.asarray(row["nodal_force_n"], dtype=float)
        if (u.shape != (size,) or z.shape != prepared.support.shape
                or not np.all(np.isfinite(u)) or not np.all(np.isfinite(z))):
            raise ValueError("saved state dimensions or finite values are invalid")
        external = np.zeros(size)
        for force, value in zip(program.nodal_forces, program.force_values_at(row["step"]), strict=True):
            if force.dof != "uz" or not np.isfinite(value):
                raise ValueError("saved radial assessment requires finite normal nodal loads")
            external[model.mesh.dof_manager.get_node_dofs(force.node_id)[2]] += value
        residual = prepared.stiffness @ u-external-prepared.gap_map.T @ z
        coordinates = linalg.lstsq(prepared.transform, u)[0]
        constraint = u-prepared.transform @ coordinates
        control_error = prepared.controls @ u-np.asarray(program.target_at(row["step"]))
        reduced_controls = prepared.controls @ prepared.transform
        reduced_residual = prepared.transform.T @ residual
        reactions = np.linalg.solve(reduced_controls @ reduced_controls.T,
                                    reduced_controls @ reduced_residual)
        free_residual = prepared.free_map.T @ residual
        span_residual = reduced_residual-reduced_controls.T @ reactions
        gap = reference.gap_m+prepared.gap_map @ u
        ncp = z-np.maximum(0., z-definition.interactions[0].normal_stiffness*prepared.support*gap)
        eq = float(np.linalg.norm(free_residual)/reference.load_pa)
        ncp_relative = float(np.sum(abs(ncp))/reference.load_pa)
        structural_error = float(np.max(abs(constraint), initial=0.)/reference.gap_m)
        prescribed_error = float(np.max(abs(control_error), initial=0.)/reference.gap_m)
        minimum_gap = float(np.min(gap))
        nonnegative = bool(np.all(z >= 0.))
        passed = (eq <= 1e-6 and ncp_relative <= 1e-8 and structural_error <= 1e-6
                  and prescribed_error <= 1e-6 and nonnegative
                  and minimum_gap/reference.gap_m >= -1e-6)
        samples = _samples(model, definition, u)
        for sample in samples:
            sample["parallel_skin_gap_m"] = reference.gap_m+sample["source_uz_m"]-sample["target_uz_m"]
        records.append(dict(step=row["step"], kkt_passed=passed,
            equilibrium_relative_force=eq, equilibrium_residual_n=float(np.linalg.norm(free_residual)),
            control_span_stationarity_relative_force=float(np.linalg.norm(span_residual)/reference.load_pa),
            structural_constraint_relative_gap=structural_error, control_constraint_relative_gap=prescribed_error,
            complementarity_relative_force=ncp_relative, minimum_normalized_gap_m=minimum_gap,
            minimum_gap_relative_reference=minimum_gap/reference.gap_m,
            minimum_nodal_force_n=float(np.min(z)), nonnegative_nodal_force=nonnegative,
            nodal_force_n=z.tolist(), normalized_gap_m=gap.tolist(),
            contact_force_n=float(np.sum(z)), control_reaction_n=reactions.tolist(),
            samples=samples))
    observable = _observables(model, prepared, np.asarray(records[-1]["nodal_force_n"]))
    samples = records[-1]["samples"]
    final = dict(force_n=observable["total"]["force_n"],
        bins_n=[[cell["force_n"] for cell in row] for row in observable["bins"]],
        bin_estimated_error_n=[[cell["estimated_absolute_error"][0] for cell in row]
                               for row in observable["bins"]],
        moment_about_origin_nm=observable["total"]["moment_nm"],
        moment_estimated_error_nm=observable["total"]["estimated_moment_error_nm"],
        moment_equivalent_resultant_xy_m=observable["total"]["resultant_xy_m"],
        source_samples=[dict(x_m=sample["xy_m"][0], y_m=sample["xy_m"][1],
                             uz_m=sample["source_uz_m"]) for sample in samples],
        target_samples=[dict(x_m=sample["xy_m"][0], y_m=sample["xy_m"][1],
                             uz_m=sample["target_uz_m"]) for sample in samples])
    solution = dict(assessed=False, reason="original saved KKT checks failed")
    if all(row["kkt_passed"] for row in records):
        try:
            solution = _solution_uncertainty(model, definition, program, reference,
                                            prepared, u, z, external, observable)
        except (ValueError, linalg.LinAlgError, linalg.LinAlgWarning) as exc:
            solution = dict(assessed=False, reason=str(exc))
    final.update({name: solution.get(name) for name in (
        "force_solution_uncertainty_n", "bin_solution_uncertainty_n",
        "moment_solution_uncertainty_nm", "resultant_solution_uncertainty_m")})
    for side, index in (("source", 0), ("target", 1)):
        for sample_index, sample in enumerate(final[side+"_samples"]):
            sample["solution_uncertainty_m"] = (solution["sample_uncertainty_m"][index][sample_index]
                                                if solution["assessed"] else None)
    output = dict(**fixture_metadata, scope=SCOPE, fixture_metadata=dict(fixture_metadata),
        kkt_passed=all(row["kkt_passed"] for row in records), steps=records,
        rows=records, development_checks_met=all(row["kkt_passed"] for row in records) and solution["assessed"],
        solution_uncertainty_assessed=solution["assessed"], solution_uncertainty=solution,
        final_observables=final,
        source_node_ids=list(prepared.source_ids),
        independent_reference_support_m2=prepared.support.tolist(),
        nodal_force_order="independent sorted source node IDs",
        saved_order_identity="explicit" if "source_node_ids" in saved_record else "legacy QP writer contract",
        independent_overlap_area_m2=prepared.area,
        joint_normal_free_dimensions=prepared.free_map.shape[1],
        joint_normal_control_dimensions=prepared.controls.shape[0],
        gates=dict(equilibrium_relative_force=1e-6, complementarity_relative_force=1e-8,
                   constraint_relative_reference_gap=1e-6, gap_relative_reference=1e-6,
                   nodal_force_minimum_n=0.),
        final_reference_pressure_observables=observable)
    # Reject overflow and nonfinite scalar diagnostics rather than encoding NaN.
    json.dumps(output, allow_nan=False)
    return output
