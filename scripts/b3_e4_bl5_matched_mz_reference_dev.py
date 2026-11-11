"""One declared 16x4, five-channel linear OpenSees reference: Fy or new Mz.

Independent ASDShellQ4/ElasticTimoshenkoBeam discretization of the flat BL5
target, not an equal native discrete operator or a converged accuracy oracle.
Shell edge interpolation is work conjugate. Beam root DOFs 1..5 are grounded;
normal spin is free. One case per fresh Python 3.12 process; no ANYfeSolver/NumPy.
Existing recorder, atomic output and small recovery services are reused.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import traceback


NX, NY, BEAM_INTERVALS = 16, 4, 32
LOADS = {"fy": (0., 100., 0., 0., 0., 0.),
         "mz": (0., 0., 0., 0., 0., 10.)}
BALANCE_LIMIT, GAP_LIMIT, DERIVATIVE_LIMIT = 1e-8, 1e-9, 1e-6
DERIVATIVE_STEPS = (1e-4, 3e-5, 1e-5)


def services():
    path = Path(__file__).with_name("b3_e4_bl5_matched_reference_actions.py")
    spec = importlib.util.spec_from_file_location("_bl5_retention_services", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference, reference_path = module._reference()
    return module, path, reference, reference_path


def geometry(reference):
    nodes = {1+j*(NX+1)+i: (i/NX, reference.WIDTH_M*j/NY, 0.)
             for j in range(NY+1) for i in range(NX+1)}
    shell = tuple((1+j*(NX+1)+i, 2+j*(NX+1)+i,
                   2+(j+1)*(NX+1)+i, 1+(j+1)*(NX+1)+i)
                  for j in range(NY) for i in range(NX))
    for i in range(BEAM_INTERVALS+1):
        nodes[1001+i] = (i/BEAM_INTERVALS, 0., reference.BEAM_AXIS_Z_M)
    beam = tuple((1001+i, 1002+i) for i in range(BEAM_INTERVALS))
    stations = []
    for i in range(1, BEAM_INTERVALS+1):
        left, odd = divmod(i, 2)
        weights = ((left+1, .5), (left+2, .5)) if odd else ((left+1, 1.),)
        stations.append({"beam": 1001+i, "x_m": i/BEAM_INTERVALS,
                         "shell_weights": weights})
    shell_roots = tuple(1+j*(NX+1) for j in range(NY+1))
    fixes = {node: (1, 1, 1, 1, 1, 1) for node in shell_roots}
    fixes[1001] = (1, 1, 1, 1, 1, 0)
    return {"nodes": nodes, "shell_cells": shell, "beam_cells": beam,
            "stations": stations, "root_shell_nodes": shell_roots,
            "beam_root": 1001, "beam_tip": 1001+BEAM_INTERVALS,
            "fixes": fixes, "physical_node_order": sorted(nodes),
            "moment_origin_m": nodes[1001]}


def equations(model, reference):
    """c=p_beam-p_shell; (theta_bx-theta_sx, theta_by-theta_sy)."""
    rb, rs = reference.BEAM_BOTTOM_ARM_M, reference.SHELL_THICKNESS_M/2
    rows = []
    for station in model["stations"]:
        beam, weights = station["beam"], station["shell_weights"]
        block = [[beam, d+1, 1.] for d in range(5)]
        block[0].extend((beam, 5, rb))
        block[1].extend((beam, 4, -rb))
        for shell, weight in weights:
            block[0].extend((shell, 1, -weight, shell, 5, -rs*weight))
            block[1].extend((shell, 2, -weight, shell, 4, rs*weight))
            block[2].extend((shell, 3, -weight))
            block[3].extend((shell, 4, -weight))
            block[4].extend((shell, 5, -weight))
        rows.extend(block)
    return rows


def row_action(rows, field):
    return [math.fsum(coefficient*field[node][dof-1]
                     for node, dof, coefficient in zip(row[::3], row[1::3], row[2::3]))
            for row in rows]


def rotated(omega, vector, cross):
    angle = math.sqrt(math.fsum(v*v for v in omega))
    if angle == 0.:
        return tuple(vector)
    first, second = cross(omega, vector), cross(omega, cross(omega, vector))
    a, b = math.sin(angle)/angle, (1.-math.cos(angle))/(angle*angle)
    return tuple(vector[i]+a*first[i]+b*second[i] for i in range(3))


def objective_gap(model, reference, station, field, service):
    """Actual left-Exp director update; only the virgin derivative is claimed.

    These row signs/order are (-c_t, c_d[1], -c_d[0]) for nu=e_z,
    b1=e_x,b2=e_y. This is not the finite native trace or a nonlinear solve.
    """
    beam, weights = station["beam"], station["shell_weights"]
    director = [math.fsum(w*rotated(field[node][3:], (0., 0., 1.), service._cross)[i]
                         for node, w in weights) for i in range(3)]
    magnitude = math.sqrt(math.fsum(v*v for v in director))
    if magnitude == 0.:
        raise ValueError("degenerate reference director")
    director = [v/magnitude for v in director]
    rs = reference.SHELL_THICKNESS_M/2
    ps = [math.fsum(w*(model["nodes"][node][i]+field[node][i]) for node, w in weights)
          +rs*director[i] for i in range(3)]
    pb = [model["nodes"][beam][i]+field[beam][i] for i in range(3)]
    inverse = tuple(-v for v in field[beam][3:])
    relative = rotated(inverse, tuple(ps[i]-pb[i] for i in range(3)), service._cross)
    local_director = rotated(inverse, director, service._cross)
    arm = (0., 0., reference.BEAM_BOTTOM_ARM_M)
    return tuple(arm[i]-relative[i] for i in range(3))+(local_director[1], -local_director[0])


def controls(model, reference, service):
    rows, nodes = equations(model, reference), model["nodes"]
    zero = {node: (0.,)*6 for node in nodes}
    rigid_errors = []
    for channel in range(6):
        translation = tuple(.01 if channel == i else 0. for i in range(3))
        omega = tuple(.01 if channel == i+3 else 0. for i in range(3))
        field = {node: tuple(translation[i]+service._cross(omega, xyz)[i] for i in range(3))+omega
                 for node, xyz in nodes.items()}
        rigid_errors.append(max(map(abs, row_action(rows, field))))
    direction = {}
    for node, (x, y, _) in nodes.items():
        direction[node] = ((.013+.009*x, .014*x*x+.007, .003*x+.011,
                            .037-.013*x, -.029+.017*x, .043*x+.015) if node >= 1001 else
                           (.011*x+.003*y, .014*x-.002*y, .006*x*y+.005*x,
                            .023*x+.004*y, .017*x-.002*y, .019*x+.008*y))
    expected = row_action(rows, direction)
    denominator = max(max(map(abs, expected)), 1e-12)
    plateau = []
    for step in DERIVATIVE_STEPS:
        plus = {node: tuple(step*v for v in row) for node, row in direction.items()}
        minus = {node: tuple(-step*v for v in row) for node, row in direction.items()}
        finite = []
        for station in model["stations"]:
            gp, gm = objective_gap(model, reference, station, plus, service), objective_gap(
                model, reference, station, minus, service)
            finite.extend((a-b)/(2*step) for a, b in zip(gp, gm))
        plateau.append(max(abs(a-b) for a, b in zip(finite, expected))/denominator)
    reference_error = max(abs(v) for station in model["stations"]
                          for v in objective_gap(model, reference, station, zero, service))
    spin = {node: ((0., 0., 0., 0., 0., .13) if node >= 1001 else (0.,)*6) for node in nodes}
    spin_error = max(abs(v) for station in model["stations"]
                     for v in objective_gap(model, reference, station, spin, service))
    rotation, translation = (.21, -.17, .13), (.03, -.02, .01)
    common = {node: tuple(rotated(rotation, xyz, service._cross)[i]+translation[i]-xyz[i]
                         for i in range(3))+rotation for node, xyz in nodes.items()}
    common_error = max(abs(v) for station in model["stations"]
                       for v in objective_gap(model, reference, station, common, service))
    old = reference.geometry(16)
    # Independently transcribe the documented historical direct rows, mapped at
    # common stations. Interior odd beam stations and extra width rows are new.
    rb, rs = reference.BEAM_BOTTOM_ARM_M, reference.SHELL_THICKNESS_M/2
    old_equal = True
    for i in range(1, 17):
        beam, shell = 1001+2*i, 1+i
        historic = [[beam, 1, 1., beam, 5, rb, shell, 1, -1., shell, 5, -rs],
                    [beam, 2, 1., beam, 4, -rb, shell, 2, -1., shell, 4, rs],
                    [beam, 3, 1., shell, 3, -1.], [beam, 4, 1., shell, 4, -1.],
                    [beam, 5, 1., shell, 5, -1.]]
        old_equal &= rows[5*(2*i-1):5*(2*i)] == historic
        old_equal &= nodes[beam] == old["nodes"][1001+i] and nodes[shell] == old["nodes"][shell]
        old_equal &= nodes[1+NY*(NX+1)+i] == old["nodes"][18+i]
    root_matrix = [[1., 0., 0., 0., rb], [0., 1., 0., -rb, 0.],
                   [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]]
    root_equivalence = service._solve5(root_matrix, [0.]*5) == [0.]*5
    interpolation_error = max(abs(math.fsum(w*nodes[n][d] for n, w in station["shell_weights"])
                                  -nodes[station["beam"]][d])
                              for station in model["stations"] for d in (0, 1))
    return {"rigid_errors": rigid_errors, "reference_gap": reference_error,
            "released_spin_gap": spin_error, "common_finite_pose_gap": common_error,
            "actual_left_exp_derivative_steps": DERIVATIVE_STEPS,
            "actual_left_exp_derivative_relative": plateau, "derivative_denominator": denominator,
            "historical_16x1_common_station_rows_geometry_equal": old_equal,
            "historical_grounded_shell_root_rows_equivalent_to_beam_first5_ground": root_equivalence,
            "normal_spin_columns_exactly_zero": all(dof != 6 for row in rows for dof in row[1::3]),
            "edge_interpolation_reference_xy_error_m": interpolation_error,
            "edge_interpolation_partition_of_unity": all(
                math.fsum(w for _, w in station["shell_weights"]) == 1.
                and all(w > 0. for _, w in station["shell_weights"])
                for station in model["stations"]),
            "limits": "Virgin five-channel derivative and common-station equality only. Added transverse "
                      "cells/odd stations/direct root grounds are declared inputs; whole operators differ."}


def solve(recorder, model, reference, case):
    """Documented command sequence from the direct comparator; new inputs explicit."""
    recorder.wipe()
    recorder.model("basic", "-ndm", 3, "-ndf", 6)
    for node, xyz in model["nodes"].items():
        recorder.node(node, *xyz)
    recorder.section("ElasticMembranePlateSection", 1, reference.E_PA, reference.NU,
                     reference.SHELL_THICKNESS_M, 7850.)
    for tag, cell in enumerate(model["shell_cells"], 1):
        recorder.element("ASDShellQ4", tag, *cell, 1)
    recorder.geomTransf("Linear", 1, 0., 0., 1.)
    for tag, cell in enumerate(model["beam_cells"], 101):
        recorder.element("ElasticTimoshenkoBeam", tag, *cell, reference.E_PA, reference.G_PA,
                         reference.BEAM_AREA_M2, reference.BEAM_TORSION_M4,
                         reference.BEAM_BENDING_M4, reference.BEAM_BENDING_M4,
                         reference.BEAM_SHEAR_AREA_M2, reference.BEAM_SHEAR_AREA_M2, 1)
    for node, flags in model["fixes"].items():
        recorder.fix(node, *flags)
    for row in equations(model, reference):
        recorder.equationConstraint(*row)
    recorder.timeSeries("Linear", 1)
    recorder.pattern("Plain", 1, 1)
    recorder.load(model["beam_tip"], *LOADS[case])
    recorder.constraints("Lagrange", 1e9, 1e9)
    recorder.numberer("RCM")
    recorder.system("BandGeneral")
    recorder.test("NormUnbalance", 1e-8, 20)
    recorder.algorithm("Linear")
    recorder.integrator("LoadControl", 1.)
    recorder.analysis("Static")
    code = recorder.analyze(1)
    if code != 0:
        raise RuntimeError(f"OpenSees analyze returned {code}")
    for node in model["nodes"]:
        values = recorder.nodeDisp(node)
        if len(values) != 6 or not all(map(math.isfinite, values)):
            raise ValueError("finite six-coordinate pose required")
    for tag, cell in enumerate(model["shell_cells"], 1):
        reference._element_forces(recorder, tag, cell)
    for tag, cell in enumerate(model["beam_cells"], 101):
        reference._element_forces(recorder, tag, cell)


def recover(raw, model, reference, case, service):
    """Same recorded transpose recovery, adapted to interpolation/partial fixes."""
    nodes, poses, rows = raw["nodes"], raw["poses"], raw["equations"]
    if (nodes != {node: list(xyz) for node, xyz in model["nodes"].items()} or set(poses) != set(nodes)
        or raw["fixes"] != {node: list(flags) for node, flags in model["fixes"].items()}
        or rows != equations(model, reference) or raw["loads"] != {model["beam_tip"]: list(LOADS[case])}):
        raise ValueError("exact declared nodes/poses/supports/rows/load inventory required")
    expected = {**{i: ("ASDShellQ4", cell) for i, cell in enumerate(model["shell_cells"], 1)},
                **{i: ("ElasticTimoshenkoBeam", cell) for i, cell in enumerate(model["beam_cells"], 101)}}
    if set(raw["elements"]) != set(expected) or set(raw["element_global_covectors"]) != set(expected):
        raise ValueError("complete 64-shell/32-beam global covector inventory required")
    for tag, (kind, cell) in expected.items():
        if raw["elements"][tag]["kind"] != kind or tuple(raw["elements"][tag]["nodes"]) != cell:
            raise ValueError("declared connectivity required")
    shell_nodes = {node for cell in model["shell_cells"] for node in cell}
    beam_nodes = {node for cell in model["beam_cells"] for node in cell}
    sg = service._sum_nodal(raw["elements"], raw["element_global_covectors"], shell_nodes, "ASDShellQ4")
    bg = service._sum_nodal(raw["elements"], raw["element_global_covectors"], beam_nodes, "ElasticTimoshenkoBeam")
    g, zero = sg | bg, [0.]*6
    applied = {node: raw["loads"].get(node, zero) for node in nodes}
    terms = {node: [[] for _ in range(6)] for node in nodes}
    station_records, gaps, dual_terms = [], [], []
    force_scale = max(max(map(abs, LOADS[case][:3])), max(map(abs, LOADS[case][3:])))
    scales = [force_scale]*6  # Lref=1m; first3 N, last3 Nm.
    origin = model["moment_origin_m"]
    shifted = {node: tuple(xyz[i]-origin[i] for i in range(3)) for node, xyz in nodes.items()}
    station_errors, map_errors = [], []
    for index, station in enumerate(model["stations"]):
        beam, block = station["beam"], rows[5*index:5*index+5]
        matrix = [[0.]*5 for _ in range(5)]
        for column, row in enumerate(block):
            for node, dof, coefficient in zip(row[::3], row[1::3], row[2::3]):
                if node == beam:
                    matrix[dof-1][column] += coefficient
        lam = service._solve5(matrix, [applied[beam][d]-bg[beam][d] for d in range(5)])
        local = {node: [0.]*6 for node in (beam, *(node for node, _ in station["shell_weights"]))}
        for row, value in zip(block, lam):
            gap = row_action((row,), poses)[0]
            gaps.append(gap)
            dual_terms.append(value*gap)
            for node, dof, coefficient in zip(row[::3], row[1::3], row[2::3]):
                terms[node][dof-1].append(coefficient*value)
                local[node][dof-1] -= coefficient*value
        rb, rs = reference.BEAM_BOTTOM_ARM_M, reference.SHELL_THICKNESS_M/2
        independent = {beam: [-lam[0], -lam[1], -lam[2], rb*lam[1]-lam[3], -rb*lam[0]-lam[4], 0.]}
        for shell, weight in station["shell_weights"]:
            independent[shell] = [weight*v for v in (lam[0], lam[1], lam[2],
                                                      -rs*lam[1]+lam[3], rs*lam[0]+lam[4], 0.)]
        map_errors.append(max(abs(local[node][d]-independent[node][d])/scales[d]
                              for node in local for d in range(6)))
        station_errors.append(max(abs(v)/scales[d] for d, v in enumerate(service._wrench(local, shifted))))
        beam_wrench = service._wrench({beam: local[beam]}, shifted)
        station_records.append({**station, "row_indices": list(range(5*index, 5*index+5)),
                                "row_multipliers_N_Nm": lam, "connection_exerted_loads": local,
                                "beam_wrench_about_beam_root": beam_wrench})
    resisting = {node: [math.fsum(values) for values in channel] for node, channel in terms.items()}
    exerted = {node: [-v for v in row] for node, row in resisting.items()}
    residual = {node: [g[node][d]+resisting[node][d]-applied[node][d] for d in range(6)] for node in nodes}
    ground = {node: [residual[node][d] if model["fixes"].get(node, (0,)*6)[d] else 0.
                     for d in range(6)] for node in nodes}
    free = {node: [residual[node][d] if not model["fixes"].get(node, (0,)*6)[d] else 0.
                   for d in range(6)] for node in nodes}
    free_error = max(abs(value)/scales[d] for node in free for d, value in enumerate(free[node]))
    applied_wrench, ground_wrench = service._wrench(applied, shifted), service._wrench(ground, shifted)
    interface_wrench = service._wrench(exerted, shifted)
    domain_cuts = {}
    for label, domain_nodes in (("shell", shell_nodes), ("beam", beam_nodes)):
        cut = {node: [applied[node][d]+exerted[node][d]+ground[node][d] for d in range(6)]
               for node in domain_nodes}
        domain_cuts[label] = {"exerted_load_wrench_about_beam_root": service._wrench(cut, shifted),
            "internal_G_wrench_about_beam_root": service._wrench({n: g[n] for n in domain_nodes}, shifted)}
    internal_work = math.fsum(service._dot(g[node], poses[node]) for node in nodes)
    applied_work = math.fsum(service._dot(applied[node], poses[node]) for node in nodes)
    interface_work = math.fsum(service._dot(exerted[node], poses[node]) for node in nodes)
    ground_work = math.fsum(service._dot(ground[node], poses[node]) for node in nodes)
    work_scale = max(abs(internal_work), abs(applied_work), 1e-12)
    virtual = {node: [(.01 if d < 3 else .01)*math.sin((node+1)*(d+1)) for d in range(6)] for node in nodes}
    virtual_work = math.fsum(service._dot(exerted[node], virtual[node]) for node in nodes)
    virtual_resisting = math.fsum(value*row_action((row,), virtual)[0]
        for record in station_records for row, value in zip(rows[record["row_indices"][0]:record["row_indices"][-1]+1],
                                                          record["row_multipliers_N_Nm"]))
    virtual_scale = max(math.fsum(abs(exerted[node][d]*virtual[node][d]) for node in nodes for d in range(6)), 1e-12)
    moments = {str(power): [math.fsum(record["x_m"]**power*record["beam_wrench_about_beam_root"][d]
                                    for record in station_records) for d in range(6)] for power in (0, 1, 2)}
    windows = {}
    for label, weight in (("open", lambda x: float(.25 < x < .75)),
                          ("closed", lambda x: float(.25 <= x <= .75)),
                          ("endpoint_atoms", lambda x: float(x in (.25, .75))),
                          ("smooth_unit_peak", lambda x: 256*(x-.25)**2*(.75-x)**2 if .25 < x < .75 else 0.)):
        windows[label] = [math.fsum(weight(record["x_m"])*record["beam_wrench_about_beam_root"][d]
                                   for record in station_records) for d in range(6)]
    root_nodes = set(model["fixes"])
    root_g = {node: g[node] for node in root_nodes}
    root_interface = {node: exerted[node] for node in root_nodes}
    return {"sign_provenance": "G=assembled global element resisting covector; L=applied load; "
                "C^T mu=L-G on each free beam station; connection resisting=C^T mu, exerted=-C^T mu; "
                "ground=(G+C^T mu-L) restricted to prescribed components. Raw nodeReaction unused.",
            "row_units": "Three position gaps m/two rotation gaps rad; strong nodal mu in N/Nm, "
                         "not native weak multiplier density N/m or Nm/m.",
            "shell_domain_nodal_G": sg, "beam_domain_nodal_G": bg,
            "connection_resisting_covectors": resisting, "connection_exerted_loads": exerted,
            "ground_exerted_loads": ground, "free_equilibrium_residuals": free,
            "station_records": station_records, "equation_gaps": gaps,
            "ground_wrench_about_beam_root": ground_wrench, "applied_wrench_about_beam_root": applied_wrench,
            "beam_ground_wrench_about_beam_root": service._wrench({model["beam_root"]: ground[model["beam_root"]]}, shifted),
            "shell_ground_wrench_about_beam_root": service._wrench({n: ground[n] for n in shell_nodes}, shifted),
            "root_nodal_G_wrench_about_beam_root": service._wrench(root_g, shifted),
            "root_nodal_interface_wrench_about_beam_root": service._wrench(root_interface, shifted),
            "connection_total_wrench_about_beam_root": interface_wrench,
            "complete_domain_cuts": domain_cuts,
            "beam_connection_polynomial_wrench_moments": moments,
            "beam_connection_fixed_region_025_075": windows,
            "fixed_region_provenance": "Station x tags refer to physical beam axes; open/closed/endpoint "
                "atomic conventions all retained. Smooth window has unit peak and zero value/slope at boundaries. "
                "Native continuous density and external nodal atoms are not pointwise interchangeable.",
            "root_provenance": "No root station equation. Beam first5 directly grounded; sixth component "
                "remains free equilibrium. Positive-x interpolation can allocate shell interface loads to fixed "
                "root nodes. Root G alone is not ground: the recorded root interface term must be added. "
                "Do not identify this partition with the old root connection into a fixed shell.",
            "work_j": {"internal": internal_work, "applied": applied_work, "connection_exerted": interface_work,
                       "ground": ground_work, "constraint_resisting": math.fsum(dual_terms),
                       "noncompatible_virtual_exerted": virtual_work, "noncompatible_virtual_resisting": virtual_resisting},
            "scales": {"length_m": 1., "force_n": force_scale, "moment_nm": force_scale,
                       "physical_work_j": work_scale, "work_floor_j": 1e-12,
                       "virtual_absolute_work_j": virtual_scale},
            "checks": {"free_equilibrium_scaled": free_error,
                "linear_five_channel_gap_scaled": max(map(abs, gaps)),
                "ground_applied_wrench_scaled": max(abs(a+b)/scales[d] for d, (a, b) in enumerate(zip(ground_wrench, applied_wrench))),
                "interface_action_reaction_scaled": max(abs(v)/scales[d] for d, v in enumerate(interface_wrench)),
                "complete_domain_cut_balance_scaled": max(abs(v)/scales[d]
                    for record in domain_cuts.values()
                    for d, v in enumerate(record["exerted_load_wrench_about_beam_root"])),
                "station_action_reaction_scaled": max(station_errors), "independent_force_arm_map_scaled": max(map_errors),
                "physical_work_relative": abs(internal_work-applied_work-interface_work-ground_work)/work_scale,
                "internal_applied_work_relative": abs(internal_work-applied_work)/work_scale,
                "compatible_interface_work_relative": abs(interface_work)/work_scale,
                "solved_dual_work_relative": abs(interface_work+math.fsum(dual_terms))/work_scale,
                "noncompatible_virtual_dual_work_relative": abs(virtual_work+virtual_resisting)/virtual_scale}}


def main(site, output, case):
    output.mkdir(parents=True, exist_ok=False)
    service, service_path, reference, reference_path = services()
    model, recorder, original_error = geometry(reference), None, None
    summary = {"classification": "new fixed-mesh five-channel physical-reference development case",
               "status": "failed", "case": case, "python": sys.version,
               "source_sha256": service._sha(__file__), "service_sha256": service._sha(service_path),
               "reference_source_sha256": service._sha(reference_path),
               "requested_threads": {name: __import__("os").environ.get(name) for name in
                                     ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
               "thread_scope": "Environment requests only; no effective OpenSees thread claim",
               "limits": "One newly declared fixture/case, not historical16x1 equality, finite native "
                         "BL5, converged1% accuracy, drilling independence, locking or F03 acceptance."}
    try:
        service._atomic(output/"inputs.json", {"case": case, "load": LOADS[case], "model": model,
            "constraint_rows": equations(model, reference), "sections": {key: getattr(reference, key) for key in
                ("E_PA", "NU", "G_PA", "SHELL_THICKNESS_M", "BEAM_AREA_M2", "BEAM_SHEAR_AREA_M2",
                 "BEAM_TORSION_M4", "BEAM_BENDING_M4", "BEAM_AXIS_Z_M", "BEAM_BOTTOM_ARM_M")},
            "frames": "Beam local y=global y via Linear transform vector (0,0,1); shell director +z",
            "native_correspondence": "Saved16x4/32halfcells/c2span4: native16B3+40P0 rows "
                "versus external32Timoshenko+160strong rows; same physical target, different operators",
            "native_nodal_pose_id_correspondence": {"shell": {n: n for n in model["nodes"] if n < 1001},
                "beam_external_to_native": {n: n-900 for n in model["nodes"] if n >= 1001},
                "limits": "Only common physical nodal translations/rotations. Native retained cell rotations "
                          "and resultants have no external nodal slot equivalence."},
            "changes_from_old_reference": "Four transverse shell cells,32externalbeamintervals, "
                "work-conjugate edge interpolation, direct beamrootfirst5 ground; Mz is a new load case"})
        control = controls(model, reference, service)
        service._atomic(output/"controls.json", control)
        if (not all(math.isfinite(v) and v <= GAP_LIMIT for v in control["rigid_errors"])
            or any(not math.isfinite(control[key]) or control[key] > GAP_LIMIT for key in
                   ("reference_gap", "released_spin_gap", "common_finite_pose_gap",
                    "edge_interpolation_reference_xy_error_m"))
            or len(control["actual_left_exp_derivative_relative"]) != 3
            or any(not math.isfinite(v) or v > DERIVATIVE_LIMIT for v in control["actual_left_exp_derivative_relative"])
            or not control["historical_16x1_common_station_rows_geometry_equal"]
            or not control["historical_grounded_shell_root_rows_equivalent_to_beam_first5_ground"]
            or not control["normal_spin_columns_exactly_zero"]
            or not control["edge_interpolation_partition_of_unity"]):
            raise ValueError("reference five-channel/geometry/derivative control failed")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("verified OpenSees binary requires Python3.12")
        sys.path.insert(0, str(site.resolve(strict=True)))
        import openseespy.opensees as ops
        from openseespywin import opensees as core
        summary.update(opensees=ops.version(), binary=str(Path(core.__file__).resolve()),
                       binary_sha256=service._sha(core.__file__))
        if not summary["opensees"].startswith("3.8") or summary["binary_sha256"] != service.BINARY_SHA256:
            raise RuntimeError("verified OpenSees3.8 binary required")
        recorder = service.RecordingOps(ops)
        solve(recorder, model, reference, case)
        service._atomic(output/"raw-inventory.json", recorder.raw)
        derived = recover(recorder.raw, model, reference, case, service)
        service._atomic(output/"recovered-actions.json", derived)
        for name, value in derived["checks"].items():
            limit = GAP_LIMIT if name == "linear_five_channel_gap_scaled" else BALANCE_LIMIT
            if not math.isfinite(value) or value > limit:
                raise ValueError(f"reference predicate failed: {name}={value} > {limit}")
        summary["status"] = "completed"
    except BaseException as error:
        original_error = error
        summary["failure"] = {"type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc()}
        raise
    finally:
        cleanup_error = None
        if recorder is not None:
            for name, action in (("retain_raw", lambda: service._atomic(output/"raw-inventory.json", recorder.raw)),
                                 ("wipe", recorder.wipe)):
                try:
                    action()
                except BaseException as error:
                    cleanup_error = error
                    summary["status"] = "failed"
                    summary.setdefault("retention_cleanup_failures", []).append(
                        {"operation": name, "type": type(error).__name__, "message": str(error)})
        try:
            service._atomic(output/"summary.json", summary)
        except BaseException:
            if original_error is None:
                raise
            traceback.print_exc()  # Preserve the original analysis failure on rethrow.
        if cleanup_error is not None and original_error is None:
            raise cleanup_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("fy", "mz"), required=True)
    args = parser.parse_args()
    main(args.site, args.output_dir, args.case)
