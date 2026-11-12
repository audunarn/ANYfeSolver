"""Bounded independent external affine BL5 references, development only.

The default 16x4 ASDShellQ4/32 ElasticTimoshenkoBeam model and one bounded
32x8/64-interval Mx discriminator retain the SAME eight physical supports.
An explicit sixteen-support option on that finer grid admits Fy/Mx/Mz. Both
[1, xi] modes apply in every channel. No native C or product module is imported.
Physical rows and dependent-row normalization are retained separately. This is
not a mesh campaign, locking or F03 assessment.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback


HELPER_SHA256 = {
    "b3_e4_bl5_external_p0_refinement_dev.py": "6f3d19d0c95a851d7e1d7dd694f1c1dce051716a811fc4fb7c733e314c4c733a",
    "b3_e4_bl5_external_p0_enforcement_dev.py": "11e593497dab40ae052a491740a6ed50bfe51a3580f275fa87d950dc1004a004",
    "b3_e4_bl5_matched_mz_reference_dev.py": "5b6eb1dac1bb4f58548c7555e10cc798ef4727406ca25efb735808b776444702",
    "b3_e4_bl5_matched_reference_actions.py": "a42d4bdeb96c7106f296e92129104eb896fbc7cb2904d9867bc61c7a9ca169c6",
    "b3_e4_f03_bl5_opensees_line_reference.py": "fe654b7a7c384620dae65f0b0981bb18e512513c090abdfe0dbf6c62c43d15cc",
}
BALANCE_LIMIT, GAP_LIMIT, DERIVATIVE_LIMIT = 1e-8, 1e-9, 1e-6
DERIVATIVE_STEPS = (1e-4, 3e-5, 1e-5)
BASIS = "discontinuous_p1"
PROJECTED_BASIS = "discontinuous_p1_cell_average"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def services():
    """Bind the existing ordinary-reference helpers before executing any of them."""
    directory = Path(__file__).resolve().parent
    paths = {name: directory/name for name in HELPER_SHA256}
    actual = {name: sha(path) for name, path in paths.items()}
    if actual != HELPER_SHA256:
        raise ValueError("external affine helper source identity changed: "+str(actual))
    path = paths["b3_e4_bl5_external_p0_refinement_dev.py"]
    spec = importlib.util.spec_from_file_location("_bl5_affine_reference_services", path)
    refinement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refinement)
    p0, frozen, service, reference, loaded = refinement.services()
    if {p.name for p in loaded} != set(paths)-{path.name}:
        raise ValueError("declared external helper graph required")
    return refinement, p0, frozen, service, reference


def shape_weights(x, intervals, first_node):
    x = F(x)
    if not 0 <= x <= 1 or type(intervals) is not int or intervals <= 0:
        raise ValueError("finite unit trace and positive interval count required")
    if x == 1:
        return {first_node+intervals: F(1)}
    coordinate = x*intervals
    left = coordinate.numerator//coordinate.denominator
    fraction = coordinate-left
    return {first_node+left: 1-fraction, first_node+left+1: fraction}


def integrated_affine_weights(left, right, intervals, first_node, union_knots):
    """Exact rational integrals of N and xi*N on every union-knot piece.

    Signed degree-one weights, including zeros, are intentional. Coordinates
    and supplied binary64 geometry are interpreted as their exact stored values.
    """
    left, right = F(left), F(right)
    if not 0 <= left < right <= 1:
        raise ValueError("positive physical support inside the unit trace required")
    knots = sorted({F(x) for x in union_knots} | {left, right})
    required = {F(i, intervals) for i in range(intervals+1)}
    if not required.issubset(knots):
        raise ValueError("all trace shape boundaries must be integration knots")
    terms, pieces = ({}, {}), []
    for a, b in zip(knots, knots[1:]):
        if a < left or b > right:
            continue
        na, nb = shape_weights(a, intervals, first_node), shape_weights(b, intervals, first_node)
        xa, xb = 2*(a-left)/(right-left)-1, 2*(b-left)/(right-left)-1
        for node in set(na) | set(nb):
            aa, bb = na.get(node, F(0)), nb.get(node, F(0))
            terms[0][node] = terms[0].get(node, F(0))+(b-a)*(aa+bb)/2
            terms[1][node] = terms[1].get(node, F(0))+(b-a)*(2*aa*xa+aa*xb+bb*xa+2*bb*xb)/6
        pieces.append({"interval_m": [float(a), float(b)],
                       "shape_at_left": {n: float(v) for n, v in na.items()},
                       "shape_at_right": {n: float(v) for n, v in nb.items()},
                       "xi_at_endpoints": [float(xa), float(xb)]})
    if sum(terms[0].values()) != right-left or sum(terms[1].values()) != 0:
        raise ValueError("complete affine partition identities required")
    for degree, weights in enumerate(terms):
        moment = sum(F(n-first_node, intervals)*w for n, w in weights.items())
        expected = (right*right-left*left)/2 if degree == 0 else (right-left)**2/6
        if moment != expected:
            raise ValueError("affine shape first-moment identity failed")
    return terms, pieces


def integrated_projected_weights(left, right, intervals, halfcells, first_node=0):
    """Exact hats times fixed native reference half-cell means of xi.

    This changes the whole density action, including support endpoints. It is
    not a nodal interpolation of the native retained rotation or condensation.
    """
    left, right = F(left), F(right)
    if (type(intervals) is not int or intervals < 1 or type(halfcells) is not int or
            halfcells < 2 or type(first_node) is not int or not 0 <= left < right <= 1 or
            (left*halfcells).denominator != 1 or (right*halfcells).denominator != 1):
        raise ValueError("positive unit trace and whole reference half-cell supports required")
    if (right-left)*halfcells < 2:
        raise ValueError("projected affine support requires at least two half-cells")
    knots = sorted({left, right} | {F(i, intervals) for i in range(intervals+1)} |
                   {F(i, halfcells) for i in range(halfcells+1)})
    terms, pieces = ({}, {}), []
    for a, b in zip(knots, knots[1:]):
        if a < left or b > right:
            continue
        edge = int((a+b)*halfcells/2)
        cell_left, cell_right = F(edge, halfcells), F(edge+1, halfcells)
        if not cell_left <= a < b <= cell_right:
            raise ValueError("integration must split at every native half-cell")
        xi = (cell_left+cell_right-2*left)/(right-left)-1
        na, nb = shape_weights(a, intervals, first_node), shape_weights(b, intervals, first_node)
        for node in set(na) | set(nb):
            integral = (b-a)*(na.get(node, F(0))+nb.get(node, F(0)))/2
            for degree, coefficient in enumerate((F(1), xi)):
                terms[degree][node] = terms[degree].get(node, F(0))+integral*coefficient
        pieces.append({"interval_m": [float(a), float(b)],
            "shape_at_left": {n: float(v) for n, v in na.items()},
            "shape_at_right": {n: float(v) for n, v in nb.items()},
            "xi_halfcell_mean": float(xi), "xi_halfcell_mean_rational": str(xi)})
    if sum(terms[0].values()) != right-left or sum(terms[1].values()) != 0:
        raise ValueError("complete projected partition identities required")
    expected = ((right*right-left*left)/2, ((right-left)**2-F(1, halfcells)**2)/6)
    for degree in (0, 1):
        if sum(F(node-first_node, intervals)*v for node, v in terms[degree].items()) != expected[degree]:
            raise ValueError("projected density shape first-moment identity failed")
    return terms, pieces


def inverse_fraction(matrix):
    """Exact fixed-size 10x10 dependent-column elimination, not mechanics."""
    if len(matrix) != 10 or any(len(row) != 10 for row in matrix):
        raise ValueError("one complete 10x10 support block required")
    a = [[F(v) for v in row]+[F(i == j) for j in range(10)] for i, row in enumerate(matrix)]
    for column in range(10):
        pivot = next((i for i in range(column, 10) if a[i][column]), None)
        if pivot is None:
            raise ValueError("dependent support columns are not independent")
        a[column], a[pivot] = a[pivot], a[column]
        value = a[column][column]
        a[column] = [v/value for v in a[column]]
        for i in range(10):
            if i != column:
                factor = a[i][column]
                a[i] = [v-factor*w for v, w in zip(a[i], a[column])]
    return [row[10:] for row in a]


def flat_row(coefficients, first=None):
    keys = sorted(k for k, value in coefficients.items() if value)
    if first is not None:
        if first not in keys:
            raise ValueError("nonzero unique dependent coefficient required")
        keys.remove(first); keys.insert(0, first)
    return [v for key in keys for v in (*key, float(coefficients[key]))]


def coefficient_maps(rows):
    return [{(n, d): c for n, d, c in zip(row[::3], row[1::3], row[2::3])} for row in rows]


def aggregate_rows(model, reference, *, basis=BASIS):
    if basis not in (BASIS, PROJECTED_BASIS):
        raise ValueError("explicit affine or fixed half-cell projected density required")
    """Independent physical rows and row-equivalent unique-dependent rows."""
    nx, nb, ns = model["nx"], model["beam_intervals"], model["support_count"]
    if (type(nx) is not int or type(nb) is not int or type(ns) is not int or
            (nx, model["ny"], nb, ns) not in ((16, 4, 32, 8), (32, 8, 64, 8), (32, 8, 64, 16))):
        raise ValueError("only the declared eight-support grids or finer32 sixteen-support model is admitted")
    first, rb, rs = model["beam_root"], F(reference.BEAM_BOTTOM_ARM_M), F(reference.SHELL_THICKNESS_M)/2
    if (tuple(model["beam_node_ids"]) != tuple(range(first, first+nb+1)) or
            any(tuple(model["nodes"][first+i]) != (float(F(i, nb)), 0., reference.BEAM_AXIS_Z_M)
                for i in range(nb+1)) or
            any(tuple(model["nodes"][1+i]) != (float(F(i, nx)), 0., 0.) for i in range(nx+1))):
        raise ValueError("declared uniform matching physical traces required")
    knots = sorted({F(i, nb) for i in range(nb+1)} | {F(i, nx) for i in range(nx+1)})
    stride, hb = nb//ns, F(1, nb)
    physical, emitted, supports, all_dependent = [], [], [], []
    exact_blocks = []
    for index in range(ns):
        left, right = F(index, ns), F(index+1, ns)
        if basis == PROJECTED_BASIS:
            bw, bp = integrated_projected_weights(left, right, nb, nb, first)
            sw, sp = integrated_projected_weights(left, right, nx, nb, 1)
        else:
            bw, bp = integrated_affine_weights(left, right, nb, first, knots)
            sw, sp = integrated_affine_weights(left, right, nx, 1, knots)
        block = [{} for _ in range(10)]  # support, channel, degree
        for weights, sign, arm in ((bw, F(1), rb), (sw, F(-1), rs)):
            for degree, nodes in enumerate(weights):
                for n, w in nodes.items():
                    for d in range(5):
                        block[2*d+degree][n, d+1] = sign*w
                    block[degree][n, 5] = sign*arm*w
                    block[2+degree][n, 4] = -sign*arm*w
        quarter_nodes = (first+stride*index+stride//4, first+stride*index+3*stride//4)
        if any((bw[0][n], bw[1][n]) != (hb, sign*hb/2)
               for n, sign in zip(quarter_nodes, (-1, 1))):
            raise ValueError("exact uniform quarter-station shape integrals required")
        dependent = [(n, d) for n in quarter_nodes for d in range(1, 6)]
        if any(model["fixes"].get(n, (0,)*6)[d-1] for n, d in dependent):
            raise ValueError("all selected interior dependent coordinates must be free")
        dmatrix = [[row.get(key, F(0)) for key in dependent] for row in block]
        e = inverse_fraction(dmatrix)
        columns = sorted({key for row in block for key in row})
        transformed = [{key: sum(e[i][j]*block[j].get(key, F(0)) for j in range(10))
                        for key in columns} for i in range(10)]
        if any(transformed[i].get(key, F(0)) != (i == j)
               for i in range(10) for j, key in enumerate(dependent)):
            raise ValueError("exact dependent-column identity failed")
        physical.extend(flat_row(row) for row in block)
        emitted.extend(flat_row(row, key) for row, key in zip(transformed, dependent))
        supports.append({"support_id": index, "basis": basis, "interval_m": [float(left), float(right)],
            "width_m": float(right-left), "row_indices": list(range(10*index, 10*index+10)),
            "beam_spacing_m": float(hb),
            "physical_row_order": "channel0..4, degree0/1 per channel",
            "beam_shape_integrals_m": [{n: float(v) for n, v in w.items()} for w in bw],
            "shell_shape_integrals_m": [{n: float(v) for n, v in w.items()} for w in sw],
            "beam_pieces": bp, "shell_pieces": sp, "recovery_beam_nodes": list(quarter_nodes),
            "dependent_node_dofs": dependent, "dependent_coefficient_block_D": [[float(v) for v in row] for row in dmatrix],
            "emitted_from_physical_E": [[float(v) for v in row] for row in e],
            "exact_E_rationals": [[str(v) for v in row] for row in e]})
        all_dependent.extend(dependent); exact_blocks.append(block)
    if len(set(all_dependent)) != 10*ns:
        raise ValueError("all dependent physical coordinates must be distinct")
    for index, block in enumerate(exact_blocks):
        other = all_dependent[:10*index]+all_dependent[10*index+10:]
        if any(row.get(key, 0) for row in block for key in other):
            raise ValueError("other-support selected columns must be absent")
    return {"basis": basis, "physical_rows": physical, "emitted_rows": emitted, "supports": supports}


def row_action(rows, field):
    return [math.fsum(c*field[n][d-1] for n, d, c in zip(r[::3], r[1::3], r[2::3])) for r in rows]


def transpose_action(rows, multipliers, nodes):
    terms = {n: [[] for _ in range(6)] for n in nodes}
    for r, multiplier in zip(rows, multipliers):
        for n, d, c in zip(r[::3], r[1::3], r[2::3]):
            terms[n][d-1].append(c*multiplier)
    return {n: [math.fsum(t) for t in axis] for n, axis in terms.items()}


def recover_densities(pairing, beam_resisting, rb):
    """Independent closed two-station recovery of D.T*lambda=L-G.

    Quarter/three-quarter integrals are w0=hb, w1=-/+hb/2, hb=1/nb. Undo the
    physical arm map at each node before recovering the constant/slope pair.
    The center's degree-one shape integral is zero and cannot recover a slope.
    """
    densities, records = [], []
    maps = coefficient_maps(pairing["physical_rows"])
    for support in pairing["supports"]:
        nodes, h, hb = support["recovery_beam_nodes"], support["width_m"], support["beam_spacing_m"]
        rhs = [float(beam_resisting[n][d]) for n in nodes for d in range(5)]
        pulled = [[*rhs[5*i:5*i+3], math.fsum((rhs[5*i+3], rb*rhs[5*i+1])),
                   math.fsum((rhs[5*i+4], -rb*rhs[5*i]))] for i in range(2)]
        if hb == h/4:
            # Preserve the completed coarse route's operation order exactly.
            lam = [v for d in range(5) for v in
                   (2*math.fsum((pulled[0][d], pulled[1][d]))/h,
                    4*math.fsum((pulled[1][d], -pulled[0][d]))/h)]
        else:
            lam = [v for d in range(5) for v in
                   (math.fsum((pulled[0][d], pulled[1][d]))/(2*hb),
                    math.fsum((pulled[1][d], -pulled[0][d]))/hb)]
        indices = support["row_indices"]
        actual = [[maps[r].get((n, d), 0.) for r in indices]
                  for n in nodes for d in range(1, 6)]
        residual = [math.fsum([a*v for a, v in zip(row, lam)]+[-value]) for row, value in zip(actual, rhs)]
        if any(not math.isfinite(v) for v in lam+residual):
            raise ValueError("finite affine density recovery required")
        records.append({"support_id": support["support_id"], "recovery_beam_nodes": nodes,
            "actual_D_transpose": actual, "rhs_L_minus_G": rhs,
            "lambda_physical_channel_degree": lam, "residual": residual,
            "formula": "undo arm map; a=(left+right)/(2*hb); b=(right-left)/hb; hb=beam spacing"})
        densities.extend(lam)
    return densities, records


def polynomial_integral(coefficients, left, right):
    left, right = F(left), F(right)
    return float(sum(F(c)*(right**(i+1)-left**(i+1))/F(i+1) for i, c in enumerate(coefficients)))


def multiply_polynomials(a, b):
    result = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += F(x)*F(y)
    return result


def density_segments(support, lam):
    """Physical coefficients on every density interval; never reinterpret modes."""
    if support.get("basis", BASIS) == PROJECTED_BASIS:
        for piece in support["beam_pieces"]:
            left, right = map(F, piece["interval_m"])
            xi = F(piece["xi_halfcell_mean_rational"])
            yield left, right, [[F(lam[2*d])+F(lam[2*d+1])*xi, F(0)] for d in range(5)]
    else:
        left, right = map(F, support["interval_m"])
        yield left, right, [[F(lam[2*d])-F(lam[2*d+1])*(left+right)/(right-left),
                            2*F(lam[2*d+1])/(right-left)] for d in range(5)]


def density_measures(supports, densities, reference):
    """Exact polynomial physical beam-exerted wrench measures, no nodal atoms."""
    rb = F(reference.BEAM_BOTTOM_ARM_M)
    moments = {str(k): [[] for _ in range(6)] for k in range(4)}
    windows = {name: [[] for _ in range(6)] for name in ("open", "closed", "smooth_unit_peak")}
    regions = [(0., .125), (.125, .25), (.25, .5), (.5, .75), (.75, .875), (.875, 1.)]
    regional = [[[] for _ in range(6)] for _ in regions]
    for support in supports:
        index = support["support_id"]
        lam = densities[10*index:10*index+10]
        for left, right, coefficients in density_segments(support, lam):
            force = [[-v for v in c] for c in coefficients[:3]]
            moment = [[rb*coefficients[1][k]-coefficients[3][k] for k in range(2)],
                      [-rb*coefficients[0][k]-coefficients[4][k] for k in range(2)]+[F(0)],
                      [F(0), *force[1]]]
            for k in range(2):
                moment[1][k+1] -= force[2][k]
            wrench = force+moment
            for d, c in enumerate(wrench):
                for k in range(4):
                    moments[str(k)][d].append(polynomial_integral([F(0)]*k+c, left, right))
                a, b = max(left, F(1, 4)), min(right, F(3, 4))
                if b > a:
                    value = polynomial_integral(c, a, b)
                    for name in ("open", "closed"):
                        windows[name][d].append(value)
                    windows["smooth_unit_peak"][d].append(polynomial_integral(
                        multiply_polynomials(c, (9, -96, 352, -512, 256)), a, b))
                for r, (a, b) in enumerate(regions):
                    a, b = max(left, F(a)), min(right, F(b))
                    if b > a:
                        regional[r][d].append(polynomial_integral(c, a, b))
    return {"polynomial_x_moments": {k: [math.fsum(t) for t in axes] for k, axes in moments.items()},
            "fixed_region_025_075": {name: [math.fsum(t) for t in axes] for name, axes in windows.items()} |
                {"endpoint_atoms": [0.]*6},
            "fixed_distance_regions": [{"interval_m": list(interval), "wrench": [math.fsum(t) for t in axes]}
                                       for interval, axes in zip(regions, regional)],
            "measure_provenance": "Declared "+supports[0].get("basis", BASIS)+" density, all five channels, physical arm and x cross F; "
                                  "about beam root. Nodal allocation is a separate transpose work map."}


def integrated_objective(field, model, supports, refinement, p0, frozen, service, reference):
    values = []
    for support in supports:
        a, b = support["interval_m"]
        terms = [[] for _ in range(10)]
        for piece in support["beam_pieces"]:
            left, right = piece["interval_m"]
            for xi, w in zip((-math.sqrt(3/5), 0., math.sqrt(3/5)), (5/9, 8/9, 5/9)):
                x = (left+right)/2+(right-left)*xi/2
                gap = refinement.sampled_gap(x, field, model, reference, p0, frozen, service)
                basis = (1., piece["xi_halfcell_mean"] if support.get("basis", BASIS) == PROJECTED_BASIS
                         else 2*(x-a)/(b-a)-1.)
                for channel in range(5):
                    for degree in range(2):
                        terms[2*channel+degree].append((right-left)*w*basis[degree]*gap[channel]/2)
        values.extend(math.fsum(t) for t in terms)
    return values


def preparation_controls(model, pairing, refinement, p0, frozen, service, reference):
    rows, supports = pairing["physical_rows"], pairing["supports"]
    nodes, width, beam = model["nodes"], 1./model["support_count"], set(model["beam_node_ids"])
    rigid = []
    for channel in range(6):
        a = tuple(.01 if channel == i else 0. for i in range(3))
        omega = tuple(.01 if channel == i+3 else 0. for i in range(3))
        field = {n: tuple(a[i]+service._cross(omega, xyz)[i] for i in range(3))+omega for n, xyz in nodes.items()}
        rigid.append(max(abs(v)/width for v in row_action(rows, field)))
    direction = {n: ((.017*x+.003, .013*x-.002, .011*x+.004, .029*x+.007, .021*x-.003, .031*x+.005)
                    if n in beam else (.011*x+.003*y, .014*x-.002*y, .006*x*y+.005*x,
                                       .023*x+.004*y, .017*x-.002*y, .019*x+.008*y))
                 for n, (x, y, z) in nodes.items()}
    exact = row_action(rows, direction)
    denominator = max(max(map(abs, exact)), 1e-12*width)
    derivative = []
    for step in DERIVATIVE_STEPS:
        plus = {n: tuple(step*v for v in q) for n, q in direction.items()}
        minus = {n: tuple(-step*v for v in q) for n, q in direction.items()}
        gp = integrated_objective(plus, model, supports, refinement, p0, frozen, service, reference)
        gm = integrated_objective(minus, model, supports, refinement, p0, frozen, service, reference)
        derivative.append(max(abs((a-b)/(2*step)-v) for a, b, v in zip(gp, gm, exact))/denominator)
    angle, shift = (.21, -.17, .13), (.03, -.02, .01)
    common = {n: tuple(frozen.rotated(angle, xyz, service._cross)[d]+shift[d]-xyz[d] for d in range(3))+angle
              for n, xyz in nodes.items()}
    fields = {"reference": {n: (0.,)*6 for n in nodes}, "common_finite_pose": common,
              "released_spin": {n: (0., 0., 0., 0., 0., .13) if n in beam else (0.,)*6 for n in nodes}}
    objective = {name: max(abs(v)/width for v in integrated_objective(
        field, model, supports, refinement, p0, frozen, service, reference)) for name, field in fields.items()}
    pm, em = coefficient_maps(rows), coefficient_maps(pairing["emitted_rows"])
    transformation_error = 0.
    for support in supports:
        ids, e = support["row_indices"], support["emitted_from_physical_E"]
        columns = {key for r in ids for key in pm[r]} | {key for r in ids for key in em[r]}
        for i, r in enumerate(ids):
            for key in columns:
                predicted = math.fsum(e[i][j]*pm[s].get(key, 0.) for j, s in enumerate(ids))
                transformation_error = max(transformation_error, abs(predicted-em[r].get(key, 0.)))
    return {"six_common_rigid_field_average_gap_scaled": rigid, "objective_average_gap_scaled": objective,
            "actual_left_exp_derivative_steps": DERIVATIVE_STEPS, "actual_left_exp_derivative_relative": derivative,
            "derivative_integrated_gap_denominator": denominator, "row_transform_max_abs": transformation_error,
            "normal_spin_columns_zero": all(d != 6 for r in rows for d in r[1::3]),
            "scope": "Exact virgin affine-weighted P1/Q1 derivative; finite pose quadrature is a derivative/objectivity control."}


def recover(raw, model, reference, pairing, case, refinement, service):
    nodes, poses = model["nodes"], raw["poses"]
    shell, beam = set(model["shell_node_ids"]), set(model["beam_node_ids"])
    sg = service._sum_nodal(raw["elements"], raw["element_global_covectors"], shell, "ASDShellQ4")
    bg = service._sum_nodal(raw["elements"], raw["element_global_covectors"], beam, "ElasticTimoshenkoBeam")
    g, rb, rs = sg | bg, reference.BEAM_BOTTOM_ARM_M, reference.SHELL_THICKNESS_M/2
    applied = {n: raw["loads"].get(n, [0.]*6) for n in nodes}
    requested = {n: [math.fsum((applied[n][d], -bg[n][d])) for d in range(6)] for n in beam}
    density, recovery = recover_densities(pairing, requested, rb)
    resisting = transpose_action(pairing["physical_rows"], density, nodes)
    exerted = {n: [-v for v in row] for n, row in resisting.items()}
    independent_terms = {n: [[] for _ in range(6)] for n in nodes}
    origin = model["moment_origin_m"]
    shifted = {n: tuple(xyz[d]-origin[d] for d in range(3)) for n, xyz in nodes.items()}
    load = refinement.LOADS[case]
    scale = max(max(map(abs, load[:3])), max(map(abs, load[3:])))  # Lref=1m.
    local_errors, support_records = [], []
    for support in pairing["supports"]:
        i, local = support["support_id"], {}
        coefficients = density[10*i:10*i+10]
        for weights, arm, sign in ((support["beam_shape_integrals_m"], rb, -1),
                                  (support["shell_shape_integrals_m"], rs, 1)):
            for n in set(weights[0]) | set(weights[1]):
                lam = [math.fsum(weights[k].get(n, 0.)*coefficients[2*d+k] for k in range(2)) for d in range(5)]
                local[n] = [sign*v for v in (lam[0], lam[1], lam[2], -arm*lam[1]+lam[3], arm*lam[0]+lam[4], 0.)]
        for n, row in local.items():
            for d, v in enumerate(row):
                independent_terms[n][d].append(v)
        wrench = service._wrench(local, shifted)
        local_errors.append(max(map(abs, wrench))/scale)
        support_records.append({"support_id": i, "interval_m": support["interval_m"],
            "density_channel_degree_N_per_m_Nm_per_m": coefficients,
            "independent_exerted_loads": local, "common_origin_interface_wrench": wrench})
    independent = {n: [math.fsum(t) for t in axes] for n, axes in independent_terms.items()}
    residual = {n: [math.fsum((g[n][d], resisting[n][d], -applied[n][d])) for d in range(6)] for n in nodes}
    ground = {n: [residual[n][d] if model["fixes"].get(n, (0,)*6)[d] else 0. for d in range(6)] for n in nodes}
    free = {n: [0. if model["fixes"].get(n, (0,)*6)[d] else residual[n][d] for d in range(6)] for n in nodes}
    gaps = row_action(pairing["physical_rows"], poses)
    emitted_gaps = row_action(pairing["emitted_rows"], poses)
    work = {"internal": math.fsum(service._dot(g[n], poses[n]) for n in nodes),
            "applied": math.fsum(service._dot(applied[n], poses[n]) for n in nodes),
            "interface_exerted": math.fsum(service._dot(exerted[n], poses[n]) for n in nodes),
            "ground": math.fsum(service._dot(ground[n], poses[n]) for n in nodes),
            "constraint_resisting": math.fsum(lam*gap for lam, gap in zip(density, gaps))}
    work_scale = max(abs(work["internal"]), abs(work["applied"]), 1e-12)
    virtual = {n: (.017*x+.003*y, .013*x-.002*z, .011*x+.004*y,
                   .029*x+.007*y, .021*x-.003*z, .031*x+.005*y) for n, (x, y, z) in nodes.items()}
    virtual_gaps = row_action(pairing["physical_rows"], virtual)
    vw = math.fsum(service._dot(exerted[n], virtual[n]) for n in nodes)
    vr = math.fsum(lam*gap for lam, gap in zip(density, virtual_gaps))
    virtual_scale = max(math.fsum(abs(exerted[n][d]*virtual[n][d]) for n in nodes for d in range(6)),
                        math.fsum(abs(lam*gap) for lam, gap in zip(density, virtual_gaps)), 1e-12)
    aw, gw, iw = (service._wrench(field, shifted) for field in (applied, ground, exerted))
    cuts = {label: {"applied_exerted_ground_wrench": service._wrench(
        {n: [math.fsum((applied[n][d], exerted[n][d], ground[n][d])) for d in range(6)] for n in domain}, shifted),
        "internal_G_wrench": service._wrench({n: g[n] for n in domain}, shifted)} for label, domain in (("shell", shell), ("beam", beam))}
    measures = density_measures(pairing["supports"], density, reference)
    checks = {"all_free_coordinate_equilibrium_scaled": max(abs(v)/scale for q in free.values() for v in q),
        "average_five_channel_gap_scaled": max(map(abs, gaps))/(1./model["support_count"]),
        "emitted_coordinate_gap_scaled": max(map(abs, emitted_gaps)),
        "local_recovery_residual_scaled": max(abs(v)/scale for r in recovery for v in r["residual"]),
        "independent_arm_pullback_scaled": max(abs(exerted[n][d]-independent[n][d])/scale for n in nodes for d in range(6)),
        "each_support_action_reaction_scaled": max(local_errors), "interface_wrench_scaled": max(map(abs, iw))/scale,
        "ground_plus_applied_wrench_scaled": max(abs(a+b)/scale for a, b in zip(aw, gw)),
        "complete_domain_cut_wrench_scaled": max(abs(v)/scale for cut in cuts.values() for v in cut["applied_exerted_ground_wrench"]),
        "physical_work_relative": abs(work["internal"]-work["applied"]-work["interface_exerted"]-work["ground"])/work_scale,
        "internal_applied_work_relative": abs(work["internal"]-work["applied"])/work_scale,
        "compatible_interface_work_relative": abs(work["interface_exerted"])/work_scale,
        "solved_dual_work_relative": abs(work["interface_exerted"]+work["constraint_resisting"])/work_scale,
        "virtual_dual_work_relative": abs(vw+vr)/virtual_scale,
        "continuous_density_nodal_total_wrench_scaled": max(abs(a-b)/scale for a, b in zip(
            measures["polynomial_x_moments"]["0"], service._wrench({n: exerted[n] for n in beam}, shifted)))}
    root = set(model["fixes"])
    return {"basis": pairing["basis"], "sign_provenance": "+Cphysical.T lambda resisting; -Cphysical.T lambda exerted; "
        "ground=(G+Cphysical.T lambda-L) on prescribed DOFs. Raw nodeReaction unused.",
        "row_density_multipliers": density, "support_records": support_records, "local_recovery": recovery,
        "full_q_node_poses": poses, "applied_nodal_L": applied, "shell_domain_nodal_G": sg, "beam_domain_nodal_G": bg,
        "connection_resisting_C_transpose_lambda": resisting, "connection_exerted_minus_C_transpose_lambda": exerted,
        "independent_density_pullback": independent, "actual_ground_exerted_loads": ground,
        "all_free_equilibrium_residuals": free, "integrated_equation_gaps_Cq": gaps, "emitted_equation_gaps": emitted_gaps,
        "virtual_q": virtual, "virtual_Cq": virtual_gaps, "work_J": work | {"virtual_exerted": vw, "virtual_resisting": vr},
        "applied_wrench_about_beam_root": aw, "ground_wrench_about_beam_root": gw, "interface_wrench_about_beam_root": iw,
        "complete_domain_cuts": cuts, "root_G_wrench": service._wrench({n: g[n] for n in root}, shifted),
        "root_interface_wrench": service._wrench({n: exerted[n] for n in root}, shifted),
        "beam_ground_wrench": service._wrench({model["beam_root"]: ground[model["beam_root"]]}, shifted),
        "shell_ground_wrench": service._wrench({n: ground[n] for n in shell}, shifted),
        "root_provenance": "Shell x0 clamped; beam root first5 directly grounded, normal spin free. "
                           "Both first-support density modes allocate root actions; retain G/interface/actual ground separately.",
        "continuous_density_measures": measures, "units": "Physical integrated position rows m^2/director rows m; "
            "density coefficients N/m,Nm/m for BOTH dimensionless modes. Emitted multipliers are not physical coefficients.",
        "scales": {"length_m": 1., "force_N": scale, "moment_Nm": scale, "work_J": work_scale,
                   "virtual_absolute_work_J": virtual_scale, "work_floor_J": 1e-12}, "checks": checks}


def write_summary(output, summary):
    """Failure retention must remain available before any helper is loaded."""
    temporary = output/("summary."+str(os.getpid())+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output/"summary.json")


def finalize(recorder, service, output, summary, *, earlier_error):
    errors = []
    if recorder is not None:
        try:
            service._atomic(output/"raw-inventory.json", recorder.raw)
        except BaseException as error:
            errors.append(error)
            summary["raw_retention_error"] = traceback.format_exc()
        try:
            recorder.ops.wipe()
        except BaseException as error:
            errors.append(error)
            summary["cleanup_error"] = traceback.format_exc()
    if errors:
        summary["status"] = "failed"
    write_summary(output, summary)
    if errors and not earlier_error:
        raise errors[0]


def admit_case(nx, case, support_count=8):
    if (type(nx) is not int or type(support_count) is not int or
            (nx, support_count) not in ((16, 8), (32, 8), (32, 16)) or
            case not in ("fy", "mx", "mz") or (nx == 32 and support_count == 8 and case != "mx")):
        raise ValueError("coarse16/eight-support Fy/Mx/Mz, finer32/eight-support Mx, "
                         "or finer32/sixteen-support Fy/Mx/Mz required")


def main(site, output, case, *, nx=16, support_count=8, basis=BASIS):
    admit_case(nx, case, support_count)
    if basis not in (BASIS, PROJECTED_BASIS) or (basis == PROJECTED_BASIS and (nx, support_count) != (16, 8)):
        raise ValueError("projected development reference admits one matched coarse16/eight-support fixture only")
    output.mkdir(parents=True, exist_ok=False)
    recorder, service = None, None
    summary = {"status": "failed", "nx": nx, "case": case, "basis": basis, "supports": support_count,
        "classification": "Fixed-grid independent affine development only; no locking/accuracy/F03 acceptance",
        "source_sha256": sha(__file__), "helper_sources": HELPER_SHA256,
        "python": sys.version, "requested_threads": {name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "thread_scope": "Environment requests only; effective external OpenSees threads unresolved; no resource-qualified claim."}
    try:
        refinement, p0, frozen, service, reference = services()
        if any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in sys.modules):
            raise ValueError("independent external runner must not import product modules")
        model = refinement.geometry(nx, reference)
        model["support_count"] = support_count  # Explicit endpoints i/ns; eight remains the default.
        pairing = aggregate_rows(model, reference, basis=basis)
        commands = refinement.command_recipe(model, reference, pairing["emitted_rows"], case)
        expected = refinement.planned_inventory(commands)
        service._atomic(output/"inputs.json", {"model": model, **pairing, "load": refinement.LOADS[case],
            "command_recipe": commands, "sections": {name: getattr(reference, name) for name in
                ("E_PA", "NU", "G_PA", "SHELL_THICKNESS_M", "BEAM_AREA_M2", "BEAM_TORSION_M4",
                 "BEAM_BENDING_M4", "BEAM_SHEAR_AREA_M2")}})
        controls = preparation_controls(model, pairing, refinement, p0, frozen, service, reference)
        service._atomic(output/"preparation-controls.json", controls)
        values = controls["six_common_rigid_field_average_gap_scaled"]+list(controls["objective_average_gap_scaled"].values())
        if (not controls["normal_spin_columns_zero"] or any(not math.isfinite(v) or v > GAP_LIMIT for v in values)
                or any(not math.isfinite(v) or v > DERIVATIVE_LIMIT for v in controls["actual_left_exp_derivative_relative"])
                or not math.isfinite(controls["row_transform_max_abs"]) or controls["row_transform_max_abs"] > BALANCE_LIMIT):
            raise ValueError("unchanged work/derivative preparation controls failed")
        if sys.version_info[:2] != (3, 12):
            raise ValueError("verified external OpenSees binary requires Python3.12")
        sys.path.insert(0, str(site.resolve(strict=True)))
        import openseespy.opensees as ops
        from openseespywin import opensees as core
        summary.update(opensees=ops.version(), binary=str(Path(core.__file__).resolve()), binary_sha256=sha(core.__file__))
        if not summary["opensees"].startswith("3.8") or summary["binary_sha256"] != service.BINARY_SHA256:
            raise ValueError("verified OpenSees3.8 binary identity required")
        recorder = service.RecordingOps(ops)
        refinement.solve(recorder, commands, model, reference)
        raw = recorder.raw
        service._atomic(output/"raw-inventory.json", raw)  # Primary evidence precedes physical predicates.
        inventory = {name: raw[name] == value for name, value in expected.items()}
        inventory["all_primary_poses"] = set(raw["poses"]) == set(model["nodes"])
        inventory["all_element_covectors"] = set(raw["element_global_covectors"]) == set(expected["elements"])
        service._atomic(output/"inventory-predicates.json", inventory)
        if not all(inventory.values()):
            raise ValueError("complete fixed physical inventory required: "+str(inventory))
        derived = recover(raw, model, reference, pairing, case, refinement, service)
        service._atomic(output/"recovered-actions.json", derived)
        service._atomic(output/"interface-recovery-blocks.json", derived["local_recovery"])
        for name, value in derived["checks"].items():
            limit = GAP_LIMIT if name in ("average_five_channel_gap_scaled", "emitted_coordinate_gap_scaled") else BALANCE_LIMIT
            if not math.isfinite(value) or value > limit:
                raise ValueError(f"{name}={value} exceeds unchanged physical budget {limit}")
        summary.update(status="completed", physical_nodes=len(model["nodes"]), shell_elements=len(model["shell_cells"]),
            beam_elements=len(model["beam_cells"]), rows=10*support_count, supports=support_count,
            applied_work_J=derived["work_J"]["applied"],
            checks=derived["checks"])
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        finalize(recorder, service, output, summary, earlier_error="error" in summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("fy", "mx", "mz"))
    parser.add_argument("--nx", type=int, choices=(16, 32), default=16)
    parser.add_argument("--support-count", type=int, choices=(8, 16), default=8)
    parser.add_argument("--basis", choices=(BASIS, PROJECTED_BASIS), default=BASIS)
    args = parser.parse_args()
    main(args.site, args.output_dir, args.case, nx=args.nx, support_count=args.support_count, basis=args.basis)
