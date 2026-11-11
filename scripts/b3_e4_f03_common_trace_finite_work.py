"""Finite-chart work check for P0/P1/signed-dual common top-face trace."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_info

from anyfesolver._ge_beam3_mixed_ad import rotation_exponential
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_weak_line_candidate import LineMultiplierSpace
from scripts.b3_e4_f03_distributed_surface_finite import FiniteSurfaceMomentPatch
from scripts.b3_e4_f03_plate_patch_work import local_grid
from scripts.b3_e4_f03_square_section_control import square_torsion_constant
from scripts.b3_e4_f03_top_face_common_trace import _raw_rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _CommonRawWork(FiniteSurfaceMomentPatch):
    """Private common co-moving frame with a signed longitudinal test basis."""

    def __init__(self, xs, ys, beam, *, kind: str, index: int):
        super().__init__(xs, ys, beam, thickness=.035,
                         quadrature_order=3)
        space = LineMultiplierSpace(kind, tuple(np.linspace(0., 1., 5)))
        if not 0 <= index < space.count:
            raise ValueError("bounded multiplier basis index required")
        self.points = [replace(point, weight=.1*point.weight*basis)
                       for point in self.points
                       if (basis := space.values((point.x-.9)/.1)[index])]
        self.inertia = np.eye(3)
        if not self.points:
            raise ArithmeticError("basis has no reference quadrature")


def _rigid_state(patch, coordinates, frames, beam_state):
    omega = np.array((.07, -.05, .04))
    translation = np.array((.003, -.002, .004))
    rigid = rotation_exponential(omega)
    shell_new = coordinates@rigid.T+translation
    frame_new = rigid@frames
    step = np.zeros(patch.beam.context.count)
    for node in range(1, patch.beam.context.nodal_count//6):
        start = 6*node
        position = beam_state.positions[node]+beam_state.position_low[node]
        step[start:start+3] = rigid@position+translation-position
        step[start+3:start+6] = omega
    for element in range(len(patch.beam.context.elements)):
        begin = patch.beam.context.nodal_count+24*element
        step[begin:begin+6] = np.tile(omega, 2)
    return shell_new, frame_new, patch.beam.context.advance(beam_state, step)


def _case(patch, seed: int) -> dict:
    beam = patch.beam
    count = patch.shell_count
    shell0 = patch.reference_nodes.copy()
    frame0 = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    beam0 = beam.context.initial.mechanical
    _, _, initial_frame = patch.center.pose_pair(beam0)
    if np.max(np.abs(initial_frame-np.eye(3))) > 1e-13:
        raise ValueError("bounded raw-arm test assumes identity initial frame")
    reference = patch.evaluate(shell0, frame0, beam0)[:, 0]
    rng = np.random.default_rng(seed)
    shell_base = rng.normal(size=(count, 6))*np.array(
        (.001, .001, .001, .025, .025, .025))
    beam_base = np.zeros(beam.context.count)
    beam_base[6:beam.context.nodal_count] = rng.normal(
        size=beam.context.nodal_count-6)*.001
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        beam_base[begin:begin+6] = rng.normal(size=6)*.02
    coordinates, frames, state = patch.advance(
        shell0, frame0, beam0, shell_base, beam_base)
    shell_u = rng.normal(size=(count, 6))*np.array(
        (.002, .002, .002, .04, .04, .04))
    shell_v = rng.normal(size=(count, 6))*np.array(
        (.002, .002, .002, .04, .04, .04))
    beam_u = rng.normal(size=beam.context.count)*.001
    beam_v = rng.normal(size=beam.context.count)*.001
    beam_u[:6] = 0.
    beam_v[:6] = 0.
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        beam_u[begin:begin+6] *= 30
        beam_v[begin:begin+6] *= 30
    derivatives = patch.evaluate(coordinates, frames, state,
                                 (shell_u, beam_u), (shell_v, beam_v))
    dual = np.array((18., -23., 31., 7., -100., 13.))

    def work(a, b):
        c, f, s = patch.advance(
            coordinates, frames, state, a*shell_u+b*shell_v,
            a*beam_u+b*beam_v)
        return float(dual@patch.evaluate(c, f, s)[:, 0])

    h1, h2 = 1e-5, 2e-3
    first_analytic = np.array((dual@derivatives[:, 1],
                               dual@derivatives[:, 2]))
    first_fd = np.array(((work(h1, 0)-work(-h1, 0))/(2*h1),
                         (work(0, h1)-work(0, -h1))/(2*h1)))
    mixed_analytic = float(dual@derivatives[:, 3])
    mixed_fd = ((work(h2, h2)-work(h2, -h2)-work(-h2, h2)+work(-h2, -h2))
                /(4*h2*h2))
    ref_direction = np.r_[shell_u.ravel(), beam_u]
    raw_trace = _raw_rows(patch, initial_frame)
    ref_first = patch.evaluate(
        shell0, frame0, beam0,
        (shell_u, beam_u), (shell_v, beam_v))[:, 1]
    reference_jacobian_error = float(np.max(np.abs(
        ref_first-raw_trace@ref_direction)))
    rigid_coordinates, rigid_frames, rigid_state = _rigid_state(
        patch, coordinates, frames, state)
    rigid_gap_error = float(np.max(np.abs(
        patch.evaluate(rigid_coordinates, rigid_frames, rigid_state)[:, 0]
        -derivatives[:, 0])))
    return {
        "reference_gap_max": float(np.max(np.abs(reference))),
        "reference_jacobian_error": reference_jacobian_error,
        "nonzero_gap_norm": float(np.linalg.norm(derivatives[:, 0])),
        "signed_weight_min": float(min(p.weight for p in patch.points)),
        "first_analytic_work": first_analytic.tolist(),
        "first_fd_work": first_fd.tolist(),
        "first_max_error_J": float(np.max(np.abs(first_analytic-first_fd))),
        "mixed_analytic_work_J": mixed_analytic,
        "mixed_fd_work_J": float(mixed_fd),
        "mixed_error_J": float(abs(mixed_analytic-mixed_fd)),
        "finite_rigid_gap_error": rigid_gap_error,
        "quadrature_points": len(patch.points),
    }


def evaluate() -> dict:
    started = monotonic()
    xs, ys = local_grid(4, transverse_cells=4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    cases = {}
    for kind, index in (("piecewise_constant", 1),
                        ("continuous_linear", 2),
                        ("dual_linear", 0)):
        patch = _CommonRawWork(xs, ys, beam, kind=kind, index=index)
        case = _case(patch, 1729)
        if (case["reference_gap_max"] > 1e-11 or
                case["reference_jacobian_error"] > 1e-10 or
                case["first_max_error_J"] > 1e-7 or
                case["mixed_error_J"] > 1e-6 or
                case["finite_rigid_gap_error"] > 1e-10 or
                case["nonzero_gap_norm"] <= 1e-12):
            raise ArithmeticError(f"finite common trace work failed for {kind}")
        if kind == "dual_linear" and case["signed_weight_min"] >= 0.:
            raise ArithmeticError("signed dual path did not exercise negative weight")
        cases[kind] = case
    return {
        "classification": "bounded finite-chart common-work discriminator, not F03 acceptance",
        "question": "Does one co-moving material-frame position law retain objective P0/P1/signed-dual multiplier work and actual E4/B3 chart derivatives?",
        "model": "Q4 x4/y4 35mm, C1 B3-12, common x=.95 frame, selected basis rows",
        "derivative_steps": {"first": 1e-5, "mixed": 2e-3},
        "development_limits": {"reference_gap": 1e-11,
                               "reference_jacobian": 1e-10,
                               "first_work_J": 1e-7,
                               "mixed_work_J": 1e-6,
                               "finite_rigid_gap": 1e-10},
        "cases": cases,
        "elapsed_s_not_benchmark": monotonic()-started,
        "limitations": ["one bounded chart and selected basis rows, not all multipliers",
                        "registered stability, physical transfer and signed dual units remain open"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate()
    if result["elapsed_s_not_benchmark"] > 120.:
        raise TimeoutError("bounded finite work check exceeded 120 s")
    result["source_sha256"] = {str(path): _sha(path) for path in (
        Path(__file__), Path("scripts/b3_e4_f03_distributed_surface_finite.py"),
        Path("scripts/b3_e4_f03_top_face_common_trace.py"))}
    result["python"] = platform.python_version()
    result["numpy"] = np.__version__
    result["threadpool_info"] = threadpool_info()
    result["thread_env"] = {key: os.environ.get(key) for key in
                            ("NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS",
                             "MKL_NUM_THREADS", "OMP_NUM_THREADS")}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"cases": {key: {
        "first_max_error_J": value["first_max_error_J"],
        "mixed_error_J": value["mixed_error_J"],
        "finite_rigid_gap_error": value["finite_rigid_gap_error"]}
        for key, value in result["cases"].items()},
        "elapsed_s": result["elapsed_s_not_benchmark"]}, sort_keys=True))


if __name__ == "__main__":
    main()
