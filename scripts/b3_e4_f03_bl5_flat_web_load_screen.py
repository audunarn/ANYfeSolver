"""Bounded same-geometry BL5/L1 flat-web load and work screen.

The L1 and BL5 laws are physically distinct. This checks each law's own
assembled equilibrium, torque/shear work and span sensitivity; it does not
rank their fidelity against a welded continuum web.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from scripts.b3_e4_f03_l1_matched_flat_web import build, SPACE_A, WEB_H_M


def _resultant(reactions, shell, beam_ids, beam_nodes):
    shell_positions = np.asarray(shell.coordinates_m)
    shell_wrenches = reactions[:6*len(shell.node_ids)].reshape(-1, 6)
    offset = 6*len(shell.node_ids)
    beam_positions = np.array([beam_nodes[node][0] for node in beam_ids])
    beam_wrenches = reactions[offset:offset+6*len(beam_ids)].reshape(-1, 6)
    positions = np.vstack((shell_positions, beam_positions))
    wrenches = np.vstack((shell_wrenches, beam_wrenches))
    return (wrenches[:, :3].sum(axis=0),
            (wrenches[:, 3:]+np.cross(positions, wrenches[:, :3])).sum(axis=0))


def _case(owner, model, name, action, applied_force, applied_moment,
          plate_tip, beam_tip):
    result = owner.solve_linear((action,))
    if (result.status != "completed" or
            result.free_equilibrium_ratio > 1e-8 or
            result.constraint_max > 1e-9):
        raise AssertionError("flat-web mixed solve did not reach equilibrium")
    shell_slot = model.shell.node_ids.index(plate_tip)
    beam_slot = owner._beam_index[beam_tip]
    shell_tip = result.shell_displacement[6*shell_slot:6*shell_slot+6]
    beam_tip_q = result.beam_coordinates[6*beam_slot:6*beam_slot+6]
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    load = owner._accepted[1]
    external_work = float(load@q)
    strain_work = float(q@(owner._K@q))
    if abs(strain_work-external_work) > 1e-8*max(abs(external_work), 1e-12):
        raise AssertionError("flat-web strain and applied work disagree")
    force, moment = _resultant(result.reactions, model.shell,
                               owner._beam_ids, owner._beam_nodes)
    force_error = float(np.max(np.abs(force+applied_force)))
    moment_error = float(np.max(np.abs(moment+applied_moment)))
    if force_error > 1e-6 or moment_error > 1e-5:
        raise AssertionError("flat-web physical support wrench did not balance")
    return {
        "name": name,
        "shell_tip": shell_tip.tolist(),
        "beam_tip": beam_tip_q.tolist(),
        "external_work_j_twice_energy": external_work,
        "internal_work_j_twice_energy": strain_work,
        "force_balance_max_n": force_error,
        "moment_balance_max_nm": moment_error,
        "free_equilibrium_ratio": result.free_equilibrium_ratio,
        "constraint_max": result.constraint_max,
        "factorizations": owner.preflight()["reference_linear_factorizations"],
    }


def _run(model, plate_tip, beam_tip, law):
    owner = stiffened_shell.create_analysis(definition=model)
    try:
        plate_action = stiffened_shell.NodalAction(
            "shell", plate_tip,
            (0., 100., 0., -100.*model.shell.thickness_m/2, 0., 0.))
        beam_action = stiffened_shell.NodalAction(
            "beam", beam_tip, (0., 0., 0., 10., 0., 0.))
        cases = {
            "plate_fy": _case(
                owner, model, "plate_fy", plate_action,
                np.array((0., 100., 0.)),
                np.array((-100.*model.shell.thickness_m/2, 0., 100.)),
                plate_tip, beam_tip),
            "beam_mx": _case(
                owner, model, "beam_mx", beam_action,
                np.zeros(3), np.array((10., 0., 0.)),
                plate_tip, beam_tip),
        }
        cross_plate = (100.*cases["beam_mx"]["shell_tip"][1]-
                       100.*model.shell.thickness_m/2*
                       cases["beam_mx"]["shell_tip"][3])
        cross_beam = 10.*cases["plate_fy"]["beam_tip"][3]
        reciprocity = abs(cross_plate-cross_beam)/max(
            abs(cross_plate), abs(cross_beam), 1e-30)
        return {"law": law, "model_identity": owner.preflight()["identity"],
                "preflight": owner.preflight(), "cases": cases,
                "cross_work_reciprocity_relative": float(reciprocity),
                "reciprocity_within_1e-8": bool(reciprocity <= 1e-8),
                "cross_work_j": {"plate_under_beam": float(cross_plate),
                                 "beam_under_plate": float(cross_beam)}}
    finally:
        owner.close()


def analyze():
    base, plate_tip, beam_tip, web = build(SPACE_A)
    # The legacy flat-web deck clamps both coincident trace and beam roots.
    # Exact weak rows then overlap the beam root support. Keep the plate-root
    # support for this same-support candidate comparison and audit every row.
    base = replace(base, fixed=tuple(v for v in base.fixed
                                     if v.domain == "shell"))
    results = {"L1_A_different_law_control":
               _run(base, plate_tip, beam_tip, "L1_A")}
    for span in (1, 2):
        old = base.lines[0]
        line = stiffened_shell.ExperimentalBL5Line(
            old.shell_nodes, old.beam_halfcells,
            multiplier_span=span,
            expected_beam_arm=(0., 0., -WEB_H_M/2))
        model = replace(base, lines=(line,))
        results[f"BL5_P0_span_{span}"] = _run(
            model, plate_tip, beam_tip, line.multiplier_space)
    return {
        "status": "BL5_FLAT_WEB_SAME_GEOMETRY_DEVELOPMENT_ONLY",
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
        "geometry": {"plate_length_m": 1., "plate_width_m": .2,
                     "plate_thickness_m": base.shell.thickness_m,
                     "beam_axis_offset_m": base.shell.thickness_m/2+WEB_H_M/2,
                     "web_section": web},
        "results": results,
        "effective_threadpools": threadpool_info(),
        "limit": ("One flat Q4/B3 model and two loads per law. Different "
                  "connection laws; no external-law parity, mesh convergence, "
                  "multiplier selection, or formal F03 acceptance."),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        report = analyze()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"],
                      "laws": tuple(report["results"]),
                      "output": str(args.output)}))
