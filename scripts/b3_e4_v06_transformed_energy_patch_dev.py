"""Check the signed transformed-section cross energy in the linear GE owner.

This is a compatible imposed-field patch, not a loaded V06 qualification case.
Native internal B3 coordinates are made stationary for each imposed field.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsqr, spsolve

from scripts.b3_e4_e01_native_point_refinement_screen import E, NU, definition
from anyfesolver.stiffened_shell import create_analysis
from anyfesolver._stiffened_shell_ge_experimental import NodalAction


LENGTH_M = 2.0
BEAM_AREA_M2 = 0.01
BEAM_IY_M4 = 8.333333333333333e-6
SHELL_THICKNESS_M = 0.07
SHELL_WIDTH_M = 1.0
OFFSET_M = 0.1
AXIAL_STRAIN = 1e-4
CURVATURE_PER_M = 1e-4


def _field(owner, *, axial: float, curvature: float, offset: float):
    count = owner._K.shape[0]
    physical_end = owner._shell_count+6*len(owner._beam_ids)
    q = np.zeros(count)
    for i, position in enumerate(owner._shell_xyz):
        x = float(position[0])
        q[6*i] = axial*x
        q[6*i+2] = -0.5*curvature*x*x
        q[6*i+4] = curvature*x
    for i, node in enumerate(owner._beam_ids):
        x = float(owner._beam_nodes[node][0][0])
        base = owner._shell_count+6*i
        q[base] = (axial+offset*curvature)*x
        q[base+2] = -0.5*curvature*x*x
        q[base+4] = curvature*x
    gap = float(np.max(np.abs(owner._C@q), initial=0.))
    retained = np.arange(physical_end, count)
    if retained.size:
        krr = owner._K[retained, :][:, retained].tocsc()
        krp = owner._K[retained, :][:, :physical_end]
        q[retained] = spsolve(krr, -krp@q[:physical_end])
        if not np.isfinite(q[retained]).all():
            raise ValueError("finite stationary retained field required")
        stationarity = float(np.max(np.abs((owner._K@q)[retained]),
                                    initial=0.))
    else:
        stationarity = 0.
    return q, gap, stationarity


def _analytic_actions(intervals: int, sign: int):
    """Consistent boundary tractions/couples for the declared section field."""
    transverse = intervals//2
    dx = LENGTH_M/intervals
    dy = SHELL_WIDTH_M/transverse
    membrane = E*SHELL_THICKNESS_M/(1.-NU*NU)
    bending = E*SHELL_THICKNESS_M**3/(12.*(1.-NU*NU))
    nx = membrane*AXIAL_STRAIN
    ny = NU*nx
    my = bending*CURVATURE_PER_M
    mx = NU*my
    actions = []
    for row in range(transverse+1):
        wy = dy*(0.5 if row in (0, transverse) else 1.)
        for col in range(1, intervals+1):
            wx = dx*(0.5 if col == intervals else 1.)
            edge_sign = -1. if row == 0 else (1. if row == transverse else 0.)
            wrench = (nx*wy if col == intervals else 0.,
                      edge_sign*ny*wx, 0.,
                      -edge_sign*mx*wx, my*wy if col == intervals else 0.,
                      0.)
            if any(wrench):
                node = 1+row*(intervals+1)+col
                actions.append(NodalAction("shell", node, wrench))
    tip = 101+2*intervals
    beam_axial = E*BEAM_AREA_M2*(AXIAL_STRAIN+
                                  sign*OFFSET_M*CURVATURE_PER_M)
    beam_my = E*BEAM_IY_M4*CURVATURE_PER_M
    actions.append(NodalAction("beam", tip,
                               (beam_axial, 0., 0., 0., beam_my, 0.)))
    return tuple(actions)


def _loaded_patch(owner, intervals: int, sign: int, imposed: np.ndarray):
    actions = _analytic_actions(intervals, sign)
    result = owner.solve_linear(actions)
    if result.status != "completed":
        raise RuntimeError(f"loaded V06 patch: {result.status}")
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    physical_end = owner._shell_count+6*len(owner._beam_ids)
    translation_slots = np.r_[np.arange(0, owner._shell_count, 6),
                              np.arange(owner._shell_count, physical_end, 6)]
    translations = np.concatenate([q[translation_slots+d]-
                                   imposed[translation_slots+d]
                                   for d in range(3)])
    rotations = np.concatenate([q[translation_slots+d]-
                                imposed[translation_slots+d]
                                for d in range(3, 6)])
    load = np.zeros(owner._K.shape[0])
    for action in actions:
        if action.domain == "shell":
            base = 6*owner._shell_index[action.node]
        else:
            base = owner._shell_count+6*owner._beam_index[action.node]
        load[base:base+6] += action.wrench
    exact_residual = np.asarray(owner._K@imposed-load)
    free_exact_residual = float(np.max(np.abs(exact_residual[owner._free]),
                                       initial=0.))
    load_scale = max(float(np.max(np.abs(load), initial=0.)), 1.)
    transfer = owner._C[:, owner._free].T.tocsr()
    fitted = lsqr(transfer, -exact_residual[owner._free],
                  atol=1e-13, btol=1e-13, iter_lim=1000)
    mixed_residual = exact_residual[owner._free]+transfer@fitted[0]
    mixed_residual_max = float(np.max(np.abs(mixed_residual), initial=0.))
    applied_force = np.zeros(3)
    applied_moment = np.zeros(3)
    for action in actions:
        if action.domain == "shell":
            position = owner._shell_xyz[owner._shell_index[action.node]]
        else:
            position = owner._beam_nodes[action.node][0]
        applied_force += action.wrench[:3]
        applied_moment += (np.asarray(action.wrench[3:])+
                           np.cross(position, action.wrench[:3]))
    support_force = np.zeros(3)
    support_moment = np.zeros(3)
    for i, position in enumerate(owner._shell_xyz):
        reaction = result.reactions[6*i:6*i+6]
        support_force += reaction[:3]
        support_moment += reaction[3:]+np.cross(position, reaction[:3])
    force_error = float(np.max(np.abs(applied_force+support_force)))
    moment_error = float(np.max(np.abs(applied_moment+support_moment)))
    work = float(q@load)
    self_energy = 0.5*float(imposed@(owner._K@imposed))
    return {
        "status": result.status,
        "action_count": len(actions),
        "max_translation_error_m": float(np.max(np.abs(translations))),
        "max_rotation_error_rad": float(np.max(np.abs(rotations))),
        "max_retained_error": float(np.max(np.abs(q[physical_end:]-
                                                imposed[physical_end:]),
                                           initial=0.)),
        "max_retained_scale_error": float(np.max(
            np.abs(q[physical_end:]-imposed[physical_end:])/
            owner._coordinate_scale[physical_end:], initial=0.)),
        "exact_field_free_residual_before_point_transfer":
            free_exact_residual,
        "exact_field_free_ratio_before_point_transfer":
            free_exact_residual/load_scale,
        "exact_field_free_residual_after_point_transfer":
            mixed_residual_max,
        "exact_field_free_ratio_after_point_transfer":
            mixed_residual_max/load_scale,
        "point_transfer_fit_stop_code": int(fitted[1]),
        "point_transfer_fit_iterations": int(fitted[2]),
        "support_force_error_n": force_error,
        "support_moment_error_nm": moment_error,
        "constraint_max": result.constraint_max,
        "free_equilibrium_ratio": result.free_equilibrium_ratio,
        "half_applied_work_j": 0.5*work,
        "imposed_energy_j": self_energy,
        "relative_energy_error": abs(0.5*work-self_energy)/self_energy,
    }


def evaluate(intervals: int, sign: int):
    if intervals not in (2, 4, 8) or sign not in (-1, +1):
        raise ValueError("registered development grids/signs required")
    definition_, _tip = definition(intervals, sign, intervals//2)
    owner = create_analysis(definition=definition_)
    a, gap_a, station_a = _field(owner, axial=AXIAL_STRAIN,
                                  curvature=0., offset=sign*OFFSET_M)
    b, gap_b, station_b = _field(owner, axial=0.,
                                  curvature=CURVATURE_PER_M,
                                  offset=sign*OFFSET_M)
    k = owner._K
    cross = float(a@(k@b))
    reverse = float(b@(k@a))
    axial_energy = 0.5*float(a@(k@a))
    curvature_energy = 0.5*float(b@(k@b))
    shell_cross = float(a[:owner._shell_count] @
                        (k[:owner._shell_count, :owner._shell_count] @
                         b[:owner._shell_count]))
    expected = (E*BEAM_AREA_M2*sign*OFFSET_M*AXIAL_STRAIN*
                CURVATURE_PER_M*LENGTH_M)
    loaded = _loaded_patch(owner, intervals, sign, a+b)
    shell_axial = E*SHELL_THICKNESS_M*SHELL_WIDTH_M/(1.-NU*NU)
    shell_bending = (E*SHELL_THICKNESS_M**3*SHELL_WIDTH_M/
                     (12.*(1.-NU*NU)))
    beam_axial = E*BEAM_AREA_M2
    beam_bending = E*BEAM_IY_M4
    axial_expected = 0.5*(shell_axial+beam_axial)*AXIAL_STRAIN**2*LENGTH_M
    curvature_expected = 0.5*(shell_bending+beam_bending+
                              beam_axial*OFFSET_M**2)*CURVATURE_PER_M**2*LENGTH_M
    relative = abs(cross-expected)/abs(expected)
    symmetry = abs(cross-reverse)/abs(expected)
    return {
        "shell_intervals": intervals,
        "transverse_intervals": intervals//2,
        "offset_sign": sign,
        "expected_beam_cross_energy_j": expected,
        "assembled_cross_energy_j": cross,
        "reverse_cross_energy_j": reverse,
        "axial_energy_j": axial_energy,
        "axial_expected_j": axial_expected,
        "axial_relative_error": abs(axial_energy-axial_expected)/axial_expected,
        "curvature_energy_j": curvature_energy,
        "curvature_expected_j": curvature_expected,
        "curvature_relative_excess": (curvature_energy-curvature_expected)/
                                     curvature_expected,
        "shell_cross_energy_j": shell_cross,
        "relative_cross_error": relative,
        "normalized_skew": symmetry,
        "max_point_gap": max(gap_a, gap_b),
        "max_retained_stationarity": max(station_a, station_b),
        "physical_dofs": owner._shell_count+6*len(owner._beam_ids),
        "retained_dofs": k.shape[0]-owner._shell_count-
                         6*len(owner._beam_ids),
        "nnz": int(k.nnz),
        "loaded_patch": loaded,
    }


def main(output: Path):
    if output.exists():
        raise FileExistsError("preserve previous V06 development evidence")
    cases = [evaluate(intervals, sign)
             for intervals in (2, 4, 8) for sign in (-1, +1)]
    record = {
        "classification": "V06 imposed transformed-section energy patch; development only",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_inputs": {"length_m": LENGTH_M,
                            "beam_area_m2": BEAM_AREA_M2,
                            "beam_iy_m4": BEAM_IY_M4,
                            "shell_thickness_m": SHELL_THICKNESS_M,
                            "shell_width_m": SHELL_WIDTH_M,
                            "poisson": NU,
                            "offset_magnitude_m": OFFSET_M,
                            "axial_strain": AXIAL_STRAIN,
                            "curvature_per_m": CURVATURE_PER_M},
        "cases": cases,
        "limit": ("Imposed compatible field and stationary beam retained variables; "
                  "analytical boundary-action solve remains a finite patch, "
                  "not a complete V06 acceptance result."),
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = main(args.output)
    print(json.dumps({"max_relative_cross_error": max(
        case["relative_cross_error"] for case in record["cases"]),
        "max_axial_relative_error": max(case["axial_relative_error"]
                                         for case in record["cases"]),
        "curvature_relative_excess_by_level": [
            case["curvature_relative_excess"] for case in record["cases"]
            if case["offset_sign"] == 1],
        "max_point_gap": max(case["max_point_gap"] for case in record["cases"]),
        "max_loaded_translation_error_m": max(
            case["loaded_patch"]["max_translation_error_m"]
            for case in record["cases"]),
        "max_normalized_skew": max(case["normalized_skew"]
                                   for case in record["cases"])}))
