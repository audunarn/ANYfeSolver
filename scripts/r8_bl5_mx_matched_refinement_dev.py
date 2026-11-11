"""Matched native/independent-solid Mx BL5 P0 refinement development screen.

Both bodies and the five-channel multiplier space grow together on one fixed
physical panel. Source checks do not replace external execution or F03 review.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from threadpoolctl import threadpool_info

from anyfesolver import b3_ge, stiffened_shell as ss
from scripts import r8_bl5_matched_native_reference as saved_native
from scripts import r8_bl5_mx_coupled_body_refinement_dev as coupled
from scripts import r8_bl5_mx_coupled_shell_refinement_dev as shell_ref
from scripts import r8_bl5_mx_torsion_refinement_dev as beam_ref
from scripts import r8_bl5_released_director_reference as solid
from scripts import r8_bl5_weak_continuum_reference as continuum


ROOT = Path(__file__).resolve().parents[1]
SHELL_LEVELS = ((4, 2, 1), (8, 4, 2), (16, 8, 4))
TORQUE_NM = 10.


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _native_definition(nx: int, ny: int) -> ss.LinearDefinition:
    """Same physical-fibre section, shell domain, root, arm and P0 rule."""
    data = saved_native.rectangular_section()
    young = 210e9
    fibres = tuple(b3_ge.Fibre(str(i), sy*.04/math.sqrt(12), sz*.06/math.sqrt(12),
                                data["area"]/4, young,
                                b3_ge.FlowCurve((0.,), (1e20,), 1e9))
                   for i, (sy, sz) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))))
    background = data["stiffness"].copy()
    background[[0, 4, 5]] = 0.
    section = b3_ge.PhysicalFibreSection(fibres, np.diag(np.sqrt(background)))
    beam_positions = np.array([(x/nx, 0., .04) for x in range(nx+1)])
    beams = tuple(b3_ge.define_beam("b3-ge", i+1,
                                    (101+2*i, 102+2*i, 103+2*i),
                                    beam_positions[2*i:2*i+3],
                                    np.tile(np.eye(3), (3, 1, 1)), section, data["mass"])
                  for i in range(nx//2))
    row = nx+1
    coordinates = tuple((i/nx, -.1+.2*j/ny, 0.)
                        for j in range(ny+1) for i in range(row))
    cells = tuple((1+j*row+i, 2+j*row+i,
                   2+(j+1)*row+i, 1+(j+1)*row+i)
                  for j in range(ny) for i in range(nx))
    panel = ss.ShellPanel(tuple(range(1, len(coordinates)+1)), coordinates,
                          cells, .02, young, .3, 7850.)
    middle = ny//2
    line_nodes = tuple(1+middle*row+i for i in range(row))
    halfcells = tuple((member, cell) for member in range(nx//2) for cell in (0, 1))
    line = ss.ExperimentalBL5Line(line_nodes, halfcells, multiplier_span=1,
                                  expected_beam_arm=(0., 0., -.03), basis="p0")
    fixed = tuple(ss.FixedCoordinate("shell", 1+j*row, axis)
                  for j in range(ny+1) for axis in range(6))
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,
                               panel, beams, (), fixed, (line,),
                               q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
                               bl5_material_policy=ss.BL5_ELASTIC_NATIVE_FIBRE_V1)


def _coarse_native_identity(candidate: ss.LinearDefinition) -> None:
    earlier = saved_native.definition()
    if (candidate.profile != earlier.profile or
            candidate.shell != earlier.shell or candidate.fixed != earlier.fixed or
            tuple(beam.sha256 for beam in candidate.beams) !=
            tuple(beam.sha256 for beam in earlier.beams) or
            tuple(line.descriptor() for line in candidate.lines) !=
            tuple(line.descriptor() for line in earlier.lines) or
            candidate.q4_pullback_policy != earlier.q4_pullback_policy or
            candidate.bl5_material_policy != earlier.bl5_material_policy):
        raise ValueError("coarse native definition differs from saved matched model")


def _native_solve(definition: ss.LinearDefinition, nx: int) -> dict:
    owner = ss.create_analysis(definition=definition)
    try:
        response = owner.solve_linear((ss.NodalAction("beam", 101+nx,
                                                      (0., 0., 0., TORQUE_NM, 0., 0.)),))
        if response.status != "completed" or owner._C.shape[0] != 5*nx:
            raise ValueError("complete native five-channel P0 solve required")
        q = np.r_[response.shell_displacement, response.beam_coordinates]
        work = float(owner._accepted[1] @ q)
        internal = float(q @ (owner._K @ q))
        relative_work = abs(internal-work)/max(abs(work), 1e-12)
        if (not np.isfinite(q).all() or response.free_equilibrium_ratio > 1e-8 or
                response.constraint_max > 1e-9 or relative_work > 1e-8):
            raise ValueError("native physical equilibrium, gap, or work failed")
        return {"applied_work_J": work,
                "tip_spin_x_rad": float(response.beam_coordinates.reshape(-1)[6*nx+3]),
                "free_equilibrium_ratio": float(response.free_equilibrium_ratio),
                "constraint_max": float(response.constraint_max),
                "relative_internal_work_error": relative_work,
                "physical_dofs": len(q), "multiplier_rows": owner._C.shape[0],
                "matrix_nnz": owner._K.nnz}
    finally:
        owner.close()


def _face_index(model: dict) -> dict:
    result = {}
    for domain in ("shell", "beam"):
        elements = []
        for eid in model[domain+"_elements"]:
            ids, low, high, center, span = solid._brick(model["elements"][eid],
                                                         model["physical_nodes"])
            elements.append((ids, np.array(low), np.array(high), np.array(center),
                             np.array(span)))
        result[domain] = elements
    return result


def _face_row(model: dict, index: dict, faces: dict, point: tuple,
              domain: str) -> np.ndarray:
    position = np.asarray(point)
    result = None
    for ids, low, high, center, span in faces[domain]:
        if np.any(position < low-1e-14) or np.any(position > high+1e-14):
            continue
        reference = 2*(position-center)/span
        for axis in range(3):
            if abs(position[axis]-low[axis]) <= 1e-14:
                reference[axis] = -1.
            elif abs(position[axis]-high[axis]) <= 1e-14:
                reference[axis] = 1.
        shape, _ = solid._shape(reference)
        row = np.zeros((3, 3*len(index)))
        for node, weight in zip(ids, shape):
            for axis in range(3):
                row[axis, index[node]+axis] += weight
        if result is not None and np.max(np.abs(result-row)) > 1e-12:
            raise ValueError("inconsistent shared solid material trace")
        result = row
    if result is None:
        raise ValueError("material trace point lacks declared host")
    return result


def _five_rows(model: dict, index: dict, faces: dict, x: float) -> np.ndarray:
    sb = _face_row(model, index, faces, (x, 0., -.01), "shell")
    st = _face_row(model, index, faces, (x, 0., .01), "shell")
    bb = _face_row(model, index, faces, (x, 0., .01), "beam")
    bt = _face_row(model, index, faces, (x, 0., .07), "beam")
    return np.vstack((bb-st, ((bt-bb)/.06-(st-sb)/.02)[:2]))


def _weak_rows(model: dict, intervals: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    order, index, _, _, _ = continuum.layout(model)
    faces = _face_index(model)
    c = np.zeros((5*intervals, 3*len(order)))
    alpha_values = (.5-.5/math.sqrt(3), .5+.5/math.sqrt(3))
    measure = 1./intervals
    for edge in range(intervals):
        for alpha in alpha_values:
            x = (edge+alpha)*measure
            rows = _five_rows(model, index, faces, x)
            for channel in range(5):
                c[channel*intervals+edge] += .5*measure*rows[channel]
    if intervals == 4:
        older, _ = continuum.constraints(model, "weak")
        if np.max(np.abs(c-older.toarray())) > 1e-12:
            raise ValueError("indexed weak rows fail independent coarse oracle")
    return sparse.csr_matrix(c), np.full(5*intervals, measure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-local", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("preserve prior matched refinement output")
    if _sha(args.saved_local) != coupled.PRIOR_SHA256:
        raise ValueError("exact saved native/external coarse comparison required")
    saved = json.loads(args.saved_local.read_text(encoding="utf-8"))
    pools = threadpool_info()
    if (saved["status"] != "DEVELOPMENT_LOCAL_TRANSFER_NOT_QUALIFIED" or
            psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(p["num_threads"] != 1 for p in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT/"src"/"anyfesolver"):
        raise ValueError("saved input or effective source/runtime changed")
    args.output_dir.mkdir(parents=True)
    for level in SHELL_LEVELS:
        started = time.perf_counter()
        nx, ny, nz = level
        definition = _native_definition(nx, ny)
        if level == SHELL_LEVELS[0]:
            _coarse_native_identity(definition)
        native_result = _native_solve(definition, nx)
        shell_nodes, shell_elements = shell_ref._shell_mesh(nx, ny, nz)
        beam_nodes, beam_elements = beam_ref._mesh(nx, ny, nz)
        model = shell_ref._combined(shell_nodes, shell_elements, beam_nodes, beam_elements)
        shell = shell_ref._shell_body(shell_nodes, shell_elements, level)
        beam = beam_ref._body(beam_nodes, beam_elements, level)
        body = sparse.block_diag((shell, beam), format="csr")
        c, scale = _weak_rows(model, nx)
        tip, meta = beam_ref._port(beam_nodes, beam_elements, 1., level)
        load = np.zeros(body.shape[0])
        load[-beam.shape[0]:] = tip.T @ np.array((0., 0., 0., TORQUE_NM, 0., 0.))
        resolved = coupled._solve(model, body, c, scale, load, tip)
        if level == SHELL_LEVELS[0]:
            old_native = saved["cases"]["mx"]["native"]["summary"]["applied_work_J"]
            if abs(native_result["applied_work_J"]-old_native)/old_native > 1e-10:
                raise ValueError("coarse native physical response changed")
        record = {"status": "DEVELOPMENT_MATCHED_REFINEMENT_NOT_QUALIFIED",
                  "identity": "R8_BL5_MX_MATCHED_NATIVE_C3D20_P0_DEV_V1",
                  "source_sha256": _sha(Path(__file__)),
                  "saved_local_sha256": _sha(args.saved_local),
                  "level": level, "native_q4_elements": nx*ny,
                  "native_b3_members": nx//2, "resolved_shell_elements": len(shell_elements),
                  "resolved_beam_elements": len(beam_elements),
                  "multiplier_rows": 5*nx,
                  "native": native_result, "resolved": resolved,
                  "native_over_resolved": native_result["applied_work_J"]/
                                          resolved["applied_work_J"],
                  "tip_port": meta,
                  "effective_runtime": {"cpu_affinity": psutil.Process().cpu_affinity(),
                                        "numba_disable_jit": int(numba.config.DISABLE_JIT),
                                        "numba_threads": int(numba.get_num_threads()),
                                        "threadpools": pools,
                                        "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())},
                  "level_wall_seconds": time.perf_counter()-started}
        name = "x".join(map(str, level))+".json"
        (args.output_dir/name).write_text(json.dumps(record, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
        print(json.dumps({"level": level,
                          "native_over_resolved": record["native_over_resolved"],
                          "wall_seconds": record["level_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
