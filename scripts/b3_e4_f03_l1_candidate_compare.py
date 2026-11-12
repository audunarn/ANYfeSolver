"""Bounded virgin L1 A/B KKT development screen; never an F03 assessment.

The 0.5--1 m eccentric material line, 1 x 0.2 m Q4 panel, native B3 section,
root supports, and physical loads are identical between candidates at each
refinement. Only the two locally non-null mixed six-channel dual spaces from
the macroelement design are solved. The actual E4 PL tangent and its
separately exposed physical component are compared explicitly; the component
omits PL completion, including terms outside the drill-only block.
An additional zero-drill-coordinate solve retains the actual PL non-drill
operator and is only a sensitivity experiment, not a physical gauge claim.
There is no converged same-law reference or production line-law selection.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import linalg

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_l1_native_connection_probe import (
    SegmentedL1WeakWork, shell_trace,
)
from scripts.b3_e4_f03_l1_space_design import (
    Macroelement, action_matrix, reference_gap_jacobian, sampled_dual_spaces,
)
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent


SPACES = (
    "P1_translation_twist_P0_pitch_normal",
    "P0_translation_twist_pair_P0_pitch_normal",
)
LOADS = ("shell_fy", "beam_tip_mx")


def _assembled_macro(segments: int) -> Macroelement:
    if type(segments) is not int or segments not in (2, 4, 8, 16):
        raise ValueError("bounded 2-, 4-, 8-, or 16-edge L1 comparison required")
    fixture = build(2*segments, z_offset=.0675, assemble=True)
    cuts = fixture.abscissae[segments:].copy()
    reference = np.column_stack((cuts, np.zeros(segments+1),
                                 np.zeros(segments+1)))
    total = np.zeros((segments+1, 6))
    state = fixture.context.initial.mechanical

    def port_at(station):
        element, cell, weight, _ = fixture._locate(station)
        return NativeHalfCellSpatialPort(fixture.context, element, cell, weight)

    blocks = tuple(SegmentedL1WeakWork.at_reference(
        reference[i:i+3], total[i:i+3], .035, cuts[i:i+3], port_at, state)
        for i in range(0, segments, 2))
    stations = np.array([s for block in blocks for s in block.stations])
    weights = np.array([w for block in blocks for w in block.weights])
    return Macroelement(fixture, blocks, cuts, stations, weights)


def _supported_problem(macro: Macroelement, *, globally_refined: bool = False,
                       transverse_segments: int | None = None):
    fixture = macro.fixture
    cuts = macro.cuts
    if transverse_segments is not None and (
            macro.segments != 16 or transverse_segments != 8 or
            not globally_refined):
        raise ValueError("only the bounded 16-edge longitudinal extension is admitted")
    if globally_refined:
        xs = np.linspace(0., 1., 2*macro.segments+1)
        ys = np.linspace(-.1, .1,
                         (transverse_segments or macro.segments)+1)
    else:
        xs = np.r_[(0., .25), cuts]
        ys = np.array((-.1, 0., .1))
    if 6*len(xs)*len(ys) > 1800:
        raise ValueError("bounded E4 reference fixture excludes this shell grid")
    shell, physical_shell = raw_e4_tangent(xs, ys, thickness=.035)
    ns, nb = shell.shape[0], fixture.context.count
    nondrill = np.array([j for j in range(ns) if j % 6 != 5], dtype=int)
    completion = shell-physical_shell
    nondrill_change = float(np.linalg.norm(
        completion[np.ix_(nondrill, nondrill)], ord=np.inf)/max(
            1., np.linalg.norm(shell, ord=np.inf)))
    if not np.array_equal(fixture.abscissae[macro.segments:], cuts):
        raise AssertionError("fixed shell and native material cuts differ")
    line_x = np.searchsorted(xs, cuts)
    line_y = int(np.argmin(np.abs(ys)))
    line_shell = np.concatenate([
        6*(line_y*len(xs)+int(j))+np.arange(6) for j in line_x])
    if (not np.array_equal(xs[line_x], cuts) or
            abs(ys[line_y]) > 1e-15):
        raise AssertionError("Q4 centerline node identity differs")
    shell_free = np.array([j for j in range(ns) if (j//6) % len(xs) != 0])
    drill = np.array([j for j in shell_free if j % 6 == 5], dtype=int)
    p_native = np.concatenate([
        fixture.context.nodal_count+24*element+np.arange(6, 24)
        for element in range(len(fixture.context.elements))])
    p = ns+p_native
    free = np.r_[shell_free, ns+np.asarray(fixture.context.free, dtype=int)]
    p_set = set(p)
    x = np.array([j for j in free if j not in p_set], dtype=int)
    if len(x)+len(p) != len(free):
        raise AssertionError("supported physical/resultant partition differs")

    gap = reference_gap_jacobian(macro)
    sample_spaces = sampled_dual_spaces(macro)
    action = {}
    for name in SPACES:
        local = action_matrix(macro, sample_spaces[name], gap)
        global_action = np.zeros((local.shape[0], ns+nb))
        global_action[:, line_shell] = local[:, :6*(macro.segments+1)]
        global_action[:, ns:] = local[:, 6*(macro.segments+1):]
        if np.max(np.abs(global_action[:, p])) != 0. or np.max(
                np.abs(global_action[:, drill])) > 1e-13:
            raise AssertionError("L1 action touches resultants or shell drill")
        action[name] = global_action

    # Concentrated but physical work-conjugate actions: shell centerline Fy
    # at x=1 and native B3 tip nodal Mx. These are identical at both levels.
    loads = np.zeros((ns+nb, 2))
    shell_tip_y = 6*(line_y*len(xs)+len(xs)-1)+1
    native_tip_mx = ns+6*(2*macro.segments)+3
    loads[shell_tip_y, 0] = 1000.
    loads[native_tip_mx, 1] = 100.
    if shell_tip_y not in x or native_tip_mx not in x:
        raise AssertionError("physical actions must lie in supported coordinates")
    return {
        "shell": shell, "physical_shell": physical_shell,
        "native": fixture.tangent.toarray(), "ns": ns, "nb": nb,
        "x": x, "p": p, "drill": drill, "action": action,
        "samples": sample_spaces, "loads": loads,
        "shell_tip_y": shell_tip_y, "native_tip_mx": native_tip_mx,
        "completion_non_drill_relative": nondrill_change,
        "xs": xs, "ys": ys,
    }


def _interface_balance(macro, samples, multipliers):
    """Integrated virgin spatial pose-wrench balance about one origin.

    ``RigidPoseJoint.evaluate`` differentiates work with respect to spatial
    port translations and left rotations, so its first 12 residual entries
    are physical port forces and moments at the supplied reference poses.
    """
    state = macro.fixture.context.initial.mechanical
    dual_at = (samples@multipliers).reshape(-1, 6)
    forces = np.zeros((2, 3))
    moments = np.zeros((2, 3))
    for sample, connection in enumerate(macro.connections):
        edge = sample//2
        reference = connection.reference
        shell, shell_frame, _, _ = shell_trace(
            reference, np.zeros((2, 6)), .035,
            station=connection.station)
        high, low, beam_frame = connection.port.pose_pair(state)
        beam = high+low+beam_frame@np.asarray(connection.beam_offset)
        pose = connection.joint.evaluate(
            np.array((shell, beam)), np.array((shell_frame, beam_frame)),
            dual_at[sample])
        f_shell, m_shell, f_beam, m_beam = pose.residual[:12].reshape(4, 3)
        weight = macro.weights[sample]
        forces[0] += weight*f_shell
        forces[1] += weight*f_beam
        moments[0] += weight*(m_shell+np.cross(shell, f_shell))
        moments[1] += weight*(m_beam+np.cross(beam, f_beam))
    return (float(np.linalg.norm(forces.sum(axis=0))),
            float(np.linalg.norm(moments.sum(axis=0))),
            float(np.linalg.norm(forces[0])),
            float(np.linalg.norm(moments[0])))


def _solve_variant(macro, problem, name: str, *,
                   variant: str):
    ns, nb = problem["ns"], problem["nb"]
    if variant not in ("actual_PL", "actual_PL_drill_frozen",
                       "E4_physical_component"):
        raise ValueError("declared virgin E4 comparison variant required")
    physical_component = variant == "E4_physical_component"
    freeze_drill = variant != "actual_PL"
    drill = set(problem["drill"])
    x = (np.array([j for j in problem["x"] if j not in drill], dtype=int)
         if freeze_drill else problem["x"])
    p = problem["p"]
    shell = (problem["physical_shell"] if physical_component
             else problem["shell"])
    if physical_component:
        # This variant quotients an exact gauge of the separately exposed
        # physical component. A changed element decomposition must fail here.
        free_drill = problem["drill"]
        if (np.count_nonzero(shell[free_drill, :]) or
                np.count_nonzero(shell[:, free_drill])):
            raise AssertionError("E4 physical-component drill is not an exact gauge")
    h = linalg.block_diag(shell, problem["native"])
    symmetry = np.linalg.norm(h-h.T, ord=np.inf)/max(1., np.linalg.norm(h, ord=np.inf))
    if symmetry > 1e-10:
        raise AssertionError("virgin supported Hessian is asymmetric")
    hxx = h[np.ix_(x, x)]
    hxp = h[np.ix_(x, p)]
    hpp = h[np.ix_(p, p)]
    negative_pp = linalg.cho_factor(-hpp, lower=True)
    stationary = linalg.cho_solve(negative_pp, -hxp.T)
    kphys = hxx-hxp@stationary
    if np.linalg.norm(kphys-kphys.T, ord=np.inf) > 1e-10*max(
            1., np.linalg.norm(kphys, ord=np.inf)):
        raise AssertionError("stationary physical tangent is asymmetric")
    action = problem["action"][name]
    bx = action[:, x]
    rows = bx.shape[0]
    if len(x)+rows > 100_000 or 8*(len(x)+rows)**2 > 8*1024**3:
        raise ValueError("declared KKT row or memory ceiling exceeded")
    # Exact KKT block solve through a positive virgin supported block and
    # its multiplier Schur complement. Scale the latter by its own diagonal;
    # raw mixed force/moment units make the unscaled KKT matrix misleadingly
    # ill-conditioned. No penalty or approximate constraint is introduced.
    physical_factor = linalg.cho_factor(kphys, lower=True, check_finite=True)
    free_response = linalg.cho_solve(physical_factor, problem["loads"][x])
    influence = linalg.cho_solve(physical_factor, bx.T)
    dual_schur = bx@influence
    diagonal = np.diag(dual_schur)
    if np.any(diagonal <= 0.):
        raise AssertionError("nonpositive six-channel multiplier work")
    dual_scale = 1./np.sqrt(diagonal)
    scaled_schur = dual_scale[:, None]*dual_schur*dual_scale[None, :]
    multiplier_factor = linalg.cho_factor(scaled_schur, lower=True,
                                         check_finite=True)
    multipliers = dual_scale[:, None]*linalg.cho_solve(
        multiplier_factor,
        dual_scale[:, None]*(bx@free_response))
    qx = free_response-influence@multipliers
    qp = -stationary@qx
    q = np.zeros((ns+nb, 2))
    q[x] = qx
    q[p] = qp
    force = problem["loads"]
    residual = h@q+action.T@multipliers-force
    active = np.r_[x, p]
    scale = max(1., np.linalg.norm(force[active], ord=np.inf),
                np.linalg.norm((h@q)[active], ord=np.inf),
                np.linalg.norm((action.T@multipliers)[active], ord=np.inf))
    equilibrium = float(np.linalg.norm(residual[active], ord=np.inf)/scale)
    compatibility = float(np.linalg.norm(action@q, ord=np.inf)/max(
        1., np.linalg.norm(bx, ord=np.inf)*np.linalg.norm(qx, ord=np.inf)))
    if equilibrium > 2e-8 or compatibility > 2e-8:
        raise AssertionError("exact L1 KKT equilibrium or compatibility failed")

    cases = {}
    for load_index, label in enumerate(LOADS):
        lam = multipliers[:, load_index]
        displacement = q[:, load_index]
        shell_work = float(lam@action[:, :ns]@displacement[:ns])
        native_work = float(lam@action[:, ns:]@displacement[ns:])
        force_error, moment_error, force_size, moment_size = _interface_balance(
            macro, problem["samples"][name], lam)
        if (abs(shell_work+native_work) > 2e-8*max(
                1., abs(shell_work), abs(native_work)) or
                force_error > 2e-8*max(1., force_size) or
                moment_error > 2e-8*max(1., moment_size)):
            raise AssertionError("integrated L1 interface work or wrench imbalance")
        shell_q = displacement[:ns]
        native_q = displacement[ns:]
        shell_operator_work = float(shell_q@shell@shell_q)
        shell_physical_work = float(
            shell_q@problem["physical_shell"]@shell_q)
        shell_completion_work = shell_operator_work-shell_physical_work
        native_operator_work = float(native_q@problem["native"]@native_q)
        external_work = float(force[:, load_index]@displacement)
        if abs(shell_operator_work+native_operator_work-external_work) > (
                1e-8*max(abs(external_work), abs(shell_operator_work),
                         abs(native_operator_work), 1e-12)):
            raise AssertionError("solved quadratic work does not balance external work")
        cases[label] = {
            "shell_tip_y_m": float(displacement[problem["shell_tip_y"]]),
            "native_tip_twist_rad": float(displacement[problem["native_tip_mx"]]),
            # f^T q is twice stored energy for this linear elastic solve.
            "external_work_Nm": external_work,
            "shell_operator_work_Nm": shell_operator_work,
            "shell_physical_component_work_Nm": shell_physical_work,
            "shell_completion_work_Nm": shell_completion_work,
            "native_retained_operator_work_Nm": native_operator_work,
            "shell_interface_work_Nm": shell_work,
            "native_interface_work_Nm": native_work,
            "interface_force_N": force_size,
            "interface_origin_moment_Nm": moment_size,
            "force_balance_N": force_error,
            "origin_moment_balance_Nm": moment_error,
        }
    return {"dual_rows": rows, "physical_dofs": len(x),
            "resultant_dofs": len(p), "equilibrium_relative": equilibrium,
            "compatibility_relative": compatibility,
            "scaled_constraint_schur_min_eigen": float(
                np.linalg.eigvalsh(scaled_schur)[0]),
            "cases": cases}


def evaluate_level(segments: int, *, globally_refined: bool = False,
                   transverse_segments: int | None = None,
                   actual_only: bool = False) -> dict:
    macro = _assembled_macro(segments)
    problem = _supported_problem(macro, globally_refined=globally_refined,
                                 transverse_segments=transverse_segments)
    candidates = {}
    for name in SPACES:
        actual = _solve_variant(macro, problem, name, variant="actual_PL")
        if actual_only:
            candidates[name] = {"actual_PL": actual}
            continue
        frozen = _solve_variant(macro, problem, name,
                                variant="actual_PL_drill_frozen")
        physical = _solve_variant(macro, problem, name,
                                  variant="E4_physical_component")
        completion_change, drill_change = {}, {}
        for load in LOADS:
            a = actual["cases"][load]["external_work_Nm"]
            f = frozen["cases"][load]["external_work_Nm"]
            b = physical["cases"][load]["external_work_Nm"]
            completion_change[load] = float(abs(b-a)/max(abs(a), abs(b), 1e-30))
            drill_change[load] = float(abs(f-a)/max(abs(a), abs(f), 1e-30))
        candidates[name] = {"actual_PL": actual,
                            "actual_PL_drill_frozen": frozen,
                            "E4_physical_component": physical,
                            "PL_completion_removal_work_change_relative":
                            completion_change,
                            "drill_coordinate_freeze_work_change_relative":
                            drill_change}
    return {"segments": segments, "line_interval_m": macro.cuts[[0, -1]].tolist(),
            "grid_policy": ("longitudinal_shell_and_beam" if
                            transverse_segments is not None else
                            "whole_shell_and_beam" if globally_refined else
                            "bond_shell_and_whole_beam"),
            "shell_grid_x_m": problem["xs"].tolist(),
            "shell_grid_y_m": problem["ys"].tolist(),
            "native_half_cells": 2*segments,
            "shell_thickness_m": .035, "beam_axis_z_m": .0675,
            "line_section_offset_z_m": -.05,
            "completion_non_drill_relative": problem["completion_non_drill_relative"],
            "loads": {"shell_fy_N": 1000., "beam_tip_mx_Nm": 100.},
            "candidates": candidates,
            "scope": "virgin supported local KKT development screen only"}


def run_provenance() -> dict:
    """Bind a development screen to its actual source and numerical runtime."""
    import hashlib
    import os
    import platform
    from pathlib import Path
    import subprocess
    import sys

    import anyfesolver
    import scipy
    from threadpoolctl import threadpool_info

    root = Path(__file__).resolve().parents[1]
    relevant = (
        "scripts/b3_e4_f03_l1_candidate_compare.py",
        "scripts/b3_e4_f03_l1_space_design.py",
        "scripts/b3_e4_f03_l1_native_connection_probe.py",
        "scripts/_b3_e4_f03_native_fixture.py",
        "scripts/b3_e4_f03_plate_patch_work.py",
        "src/anyfesolver/e4_pl_element.py",
        "src/anyfesolver/_ge_beam3_pose_joint.py",
    )
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in relevant}
    return {
        "git_head": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=root, text=True).strip(),
        "source_sha256": hashes,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "anyfesolver_import": str(Path(anyfesolver.__file__).resolve()),
        "thread_environment": {key: os.getenv(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMBA_DISABLE_JIT")},
        "effective_threadpools": threadpool_info(),
    }


if __name__ == "__main__":
    print(json.dumps({"provenance": run_provenance(),
                      "results": [evaluate_level(n) for n in (2, 4)]},
                     indent=2))
