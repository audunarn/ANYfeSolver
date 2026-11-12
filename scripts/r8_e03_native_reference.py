"""Existing public GE/B2 adapters for the unexecuted E03 source candidate.

Imports are deferred until a caller explicitly constructs or executes a native
definition. This module launches no processes and supplies no numerical authority.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from scripts.r8_e03_curved_reference import B2, GE, BALANCE_LIMIT, balance_measure, validate_case


def b2_definition(case):
    validate_case(case)
    from anyfesolver import stiffened_shell as ss

    grid, nx = case["grids"]["b2"], case["nx"]
    section = ss.LinearSection.isotropic(
        E=case["young_pa"], nu=case["poisson"], area=case["section"]["area_m2"],
        Iy=case["section"]["Iy_m4"], Iz=case["section"]["Iz_m4"],
        J=case["section"]["J_m4"], ky=case["section"]["shear_y"], kz=case["section"]["shear_z"])
    shells = []
    for row, ids in enumerate(grid["quads"]):
        xyz = np.array([grid["nodes"][n-1] for n in ids])
        normal = np.cross(xyz[1]-xyz[0], xyz[3]-xyz[0])
        normal /= np.linalg.norm(normal)
        shells.append(ss.Q4PanelShell("shell"+str(row), tuple(ids), case["young_pa"],
            case["poisson"], case["thickness_m"], tuple(float(v) for v in normal)))
    members, points, beam_positions = [], [], {}
    for rib_index, rib in enumerate(case["ribs"]):
        for i, node in enumerate(rib["b2_ids"]):
            p = list(rib["root_center"])
            p[0] += case["length_m"]*i/nx
            beam_positions[node] = p
        for i in range(nx):
            member_id = f"rib{rib_index}-member{i}"
            ids = tuple(rib["b2_ids"][i:i+2])
            members.append(ss.PhysicalB2Member(member_id, "e03-"+case["case_id"], ids,
                tuple(tuple(beam_positions[n]) for n in ids), tuple(rib["tangent"]),
                "rectangular-elastic", section))
            points.append(ss.Q4B2Point(f"rib{rib_index}-point{i}",
                "shell"+str(i*case["ny"]+rib["strip"]), member_id, .5))
    shell_ids = list(range(1, len(grid["nodes"])+1))
    beam_ids = sorted(beam_positions)
    numbering = shell_ids+beam_ids
    fixed_nodes = set(grid["root_nodes"]+[rib["b2_ids"][0] for rib in case["ribs"]])
    free = [6*i+axis for i, node in enumerate(numbering) if node not in fixed_nodes for axis in range(6)]
    dimension = 6*len(numbering)
    definition = ss.B2Q4PanelDefinition(ss.B2_E4_PL_FAST_LINEAR_V1,
        dict(zip(shell_ids, (tuple(p) for p in grid["nodes"]))), tuple(shells),
        tuple(members), tuple(points), sparse.eye(dimension, format="csr")[:, free], np.zeros(dimension))
    load = np.zeros(dimension)
    for rib in case["ribs"]:
        row = numbering.index(rib["b2_ids"][-1])
        load[6*row:6*row+3] = case["tip_force_each_n"]
    return definition, load, numbering, grid["nodes"]+[beam_positions[n] for n in beam_ids]


def ge_definition(case, *, density_kg_m3):
    validate_case(case)
    if type(density_kg_m3) not in (int, float) or not math.isfinite(density_kg_m3) or density_kg_m3 <= 0:
        raise ValueError("explicit positive physical density required; no spectral claim")
    from anyfesolver import b3_ge, stiffened_shell as ss

    p, e, nu = case["section"], case["young_pa"], case["poisson"]
    shear = e/(2*(1+nu))
    stiffness = np.diag((e*p["area_m2"], shear*p["area_m2"]*p["shear_y"],
        shear*p["area_m2"]*p["shear_z"], shear*p["J_m4"], e*p["Iy_m4"], e*p["Iz_m4"]))
    section = b3_ge.EllipsoidalGeneralizedSection(stiffness, np.eye(6), 1e20, 1e9)
    inertia = np.diag(density_kg_m3*np.array((p["area_m2"], p["area_m2"], p["area_m2"],
                                            p["Iy_m4"]+p["Iz_m4"], p["Iy_m4"], p["Iz_m4"])))
    grid, nx, ny = case["grids"]["ge"], case["nx"], case["ny"]
    shell_ids = tuple(range(1, len(grid["nodes"])+1))
    shell = ss.ShellPanel(shell_ids, tuple(tuple(p) for p in grid["nodes"]),
        tuple(tuple(ids) for ids in grid["quads"]), case["thickness_m"], e, nu, density_kg_m3)
    beams, points, fixed, loads = [], [], [], []
    for node in grid["root_nodes"]:
        fixed.extend(ss.FixedCoordinate("shell", node, axis) for axis in range(6))
    for rib_index, rib in enumerate(case["ribs"]):
        coordinates = np.array([rib["root_center"] for _ in range(2*nx+1)])
        coordinates[:, 0] += case["length_m"]*np.arange(2*nx+1)/(2*nx)
        frames = np.tile(np.array(rib["frame"]), (2*nx+1, 1, 1))
        for i in range(nx):
            start = 2*i
            beams.append(b3_ge.define_beam("b3-ge", rib_index*nx+i+1,
                tuple(rib["ge_ids"][start:start+3]), coordinates[start:start+3],
                frames[start:start+3], section, inertia))
            shell_node = (2*i+1)*(2*ny+1)+2*rib["strip"]+2
            points.append(ss.PointConnection(shell_node, rib["ge_ids"][start+1]))
        fixed.extend(ss.FixedCoordinate("beam", rib["ge_ids"][0], axis) for axis in range(6))
        loads.append(ss.NodalAction("beam", rib["ge_ids"][-1], tuple(case["tip_force_each_n"])+(0., 0., 0.)))
    definition = ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1, shell,
        tuple(beams), tuple(points), tuple(fixed), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        shell_reference_policy=ss.CURVED_Q4_POINT_REFERENCE_V1)
    return definition, tuple(loads)


def execute_b2(case):
    """Run once only under the caller's separately granted numerical authority."""
    from anyfesolver import stiffened_shell as ss

    definition, load, numbering, positions = b2_definition(case)
    owner = ss.create_analysis(definition=definition)
    try:
        result = owner.solve_linear(load)
        if result.status != "completed":
            raise RuntimeError("E03 B2 native failure: "+str(result.failure_reason))
        displacement = np.asarray(result.displacements, dtype=float)
        reactions = np.asarray(result.affine_reactions, dtype=float).reshape(-1, 6)
        if displacement.shape != load.shape or reactions.shape != (len(numbering), 6) or not np.isfinite(displacement).all() or not np.isfinite(reactions).all():
            raise RuntimeError("complete finite native physical response required")
        force = reactions[:, :3].sum(axis=0)+load.reshape(-1, 6)[:, :3].sum(axis=0)
        moment = np.zeros(3)
        root_moment = np.zeros(3)
        for p, r, f in zip(positions, reactions, load.reshape(-1, 6)):
            moment += np.cross(p, r[:3]+f[:3])+r[3:]+f[3:]
            root_moment += np.cross(p, r[:3])+r[3:]
        scaled_balance = balance_measure(case, force, moment)
        if scaled_balance > BALANCE_LIMIT:
            raise RuntimeError("native common-origin force/moment balance failed")
        work = float(load @ displacement)
        if not math.isfinite(work) or work <= 0.:
            raise RuntimeError("positive finite native applied-force work required")
        tips = []
        for rib_index, rib in enumerate(case["ribs"]):
            row = numbering.index(rib["b2_ids"][-1])
            tips.append(dict(rib_index=rib_index, world_pose=displacement[6*row:6*row+6].tolist()))
        return dict(status="COMPLETED_UNQUALIFIED_B2_CANDIDATE", profile=B2,
            case_id=case["case_id"], level=case["level"], offset_sign=case["offset_sign"],
            external_work_j=work, tip_poses=tips, force_balance_n=force.tolist(),
            moment_balance_nm=moment.tolist(),
            scaled_balance=scaled_balance,
            root_reaction_wrench_world_origin=np.r_[reactions[:, :3].sum(axis=0), root_moment].tolist(),
            native_capability_law=owner.capabilities.point_law_id,
            point_multiplier_coordinates="native ReferencePointLaw rows; not world spatial wrenches",
            point_gap_coordinates="native ReferencePointLaw rows; not world pose components",
            point_multipliers_native_rows=[[str(name), [float(v) for v in values]]
                                           for name, values in result.point_multipliers],
            point_gaps_native_rows=[[str(name), [float(v) for v in values]]
                                    for name, values in result.point_gaps],
            production_qualified=False)
    finally:
        owner.close()


def refuse_ge_reference_scoring():
    raise ValueError("GE nodal independent shell rotation/drill has no adopted external work map")
