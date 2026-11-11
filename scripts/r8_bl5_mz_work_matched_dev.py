"""Work-matched native/weak-continuum BL5 in-plane couple screen.

Development only. Reuses the declared five-channel material trace and the
existing body assemblers, without modifying their frozen Fz evidence. A
consistent end-face axial traction has resultant +Mz and zero force. Different
primal body spaces mean response differences are not a connection-law score.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import numba
import psutil
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_experimental import (
    _linear_equilibrium, _scaled_constraint_gap,
)
from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_weak_continuum_reference as continuum


IDENTITY = "R8_BL5_MZ_WORK_MATCHED_DEV_V2"
MOMENT_NM = 1.0


def end_traction(model):
    """Integrate t_x=-Mz*y/Iz on each x=1 C3D20 bar end face."""
    order, index, _, _, _ = continuum.layout(model)
    load = np.zeros(3 * len(order))
    points, weights = np.polynomial.legendre.leggauss(3)
    iz = float(model["section"]["Iz_m4"])
    selected = []
    for eid in model["beam_elements"]:
        ids, low, high, center, span = continuum.original._brick(
            model["elements"][eid], model["physical_nodes"])
        if not np.isclose(high[0], 1., rtol=0., atol=1e-14):
            continue
        selected.append(eid)
        for iy, izeta in itertools.product(range(3), repeat=2):
            eta, zeta = points[iy], points[izeta]
            shape, _ = continuum.original._shape((1., eta, zeta))
            y = center[1] + .5 * span[1] * eta
            d_area = .25 * span[1] * span[2] * weights[iy] * weights[izeta]
            for node, value in zip(ids, shape):
                load[index[node]] += value * (-MOMENT_NM * y / iz) * d_area
    if len(selected) != 2:
        raise ValueError("expected two complete bar end faces")
    # The exact traction and geometry are odd in y. Enforce that analytical
    # parity on the assembled nodal load, including exact zero at y=0. Tiny
    # quadrature roundoff on a theoretically unloaded row would otherwise
    # trigger the solver's active-row relative-equilibrium check.
    tip_nodes = [n for n in model["beam_nodes"]
                 if np.isclose(model["physical_nodes"][n][0], 1., rtol=0., atol=1e-14)]
    corrected = load.copy()
    for node in model["beam_nodes"]:
        if node not in tip_nodes:
            load[index[node]] = 0.
    for node in tip_nodes:
        x, y, z = model["physical_nodes"][node]
        if y == 0.:
            load[index[node]] = 0.
            continue
        counterpart = [other for other in tip_nodes if
                       np.allclose(model["physical_nodes"][other],
                                   (x, -y, z), rtol=0., atol=1e-14)]
        if len(counterpart) != 1:
            raise ValueError("symmetric bar end-face topology required")
        other = counterpart[0]
        if node < other:
            odd = .5 * (load[index[node]] - load[index[other]])
            load[index[node]], load[index[other]] = odd, -odd
    parity_roundoff = float(np.max(np.abs(corrected - load)))
    rigid = continuum.rigid_fields(model, "beam")
    work = rigid.T @ load
    expected = np.array((0., 0., 0., 0., 0., MOMENT_NM))
    if not np.allclose(work, expected, rtol=0., atol=2e-13):
        raise ValueError("end traction does not supply the declared six-mode work")
    if not np.isfinite(load).all() or np.count_nonzero(load) > 27:
        raise ValueError("invalid or nonlocal end traction")
    return load, selected, work, parity_roundoff


def solve_continuum(model, k, c, scale, load):
    """The existing weak reference KKT with the new owned load vector."""
    _, _, fixed, free, _ = continuum.layout(model)
    kf = k[free, :][:, free].tocsc()
    cf = c[:, free].tocsc()
    diagonal = kf.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError("positive finite displacement diagonal required")
    dq = 1 / np.sqrt(diagonal)
    norm = np.sqrt(np.asarray((cf @ sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("nonzero finite constraint rows required")
    dl = 1 / norm
    dp, dm = sparse.diags(dq), sparse.diags(dl)
    matrix = sparse.bmat([[dp @ kf @ dp, dp @ cf.T @ dm],
                          [dm @ cf @ dp, None]], format="csc")
    rhs = np.r_[dq * load[free], np.zeros(c.shape[0])]
    lu = splu(matrix)
    z = lu.solve(rhs)
    for _ in range(3):
        z += lu.solve(rhs - matrix @ z)
    q = np.zeros(len(load))
    q[free] = dq * z[:len(free)]
    multipliers = dl * z[len(free):]
    internal = k @ q
    interface = -c.T @ multipliers
    residual = internal - interface - load
    support = np.zeros_like(q)
    support[fixed] = residual[fixed]
    equilibrium = _linear_equilibrium(residual[free], load[free],
                                      np.ones(len(free)))
    compatibility = _scaled_constraint_gap(c @ q, scale)
    work = float(load @ q)
    denominator = max(abs(work), 1e-12)
    work_error = abs(float(q @ internal) - work) / denominator
    interface_work_error = abs(float(q @ interface)) / denominator
    if not (np.isfinite(q).all() and np.isfinite(multipliers).all()):
        raise ValueError("nonfinite continuum state")
    if (equilibrium > 1e-8 or compatibility > 1e-9 or
            work_error > 1e-8 or interface_work_error > 1e-8):
        raise ValueError("continuum equilibrium, gap or work gate failed: " +
                         repr(dict(equilibrium=equilibrium,
                                   compatibility=compatibility,
                                   work_error=work_error,
                                   interface_work_error=interface_work_error,
                                   applied_pairing=work)))
    return q, multipliers, interface, support, dict(
        free_equilibrium_ratio=float(equilibrium),
        compatibility=float(compatibility),
        internal_work_relative_error=float(work_error),
        interface_work_relative_error=float(interface_work_error),
        applied_pairing_J=work,
        matrix_dimension=int(matrix.shape[0]),
        matrix_nnz=int(matrix.nnz),
    )


def wrench(model, vector, body=None):
    field = continuum.field_dict(model, vector)
    if body is not None:
        field = {n: value for n, value in field.items()
                 if n in model[body + "_nodes"]}
    return native.common_origin_wrench(model["physical_nodes"], field)


def body_work(k, q, applied, interface, shell_slots, beam_slots):
    """Separate actual body energy from external-plus-interface work."""
    if k[shell_slots, :][:, beam_slots].nnz:
        raise ValueError("unexpected direct shell/beam body coupling")
    result = {}
    denominator = max(abs(float(applied @ q)), 1e-12)
    for name, slots in (("shell", shell_slots), ("beam", beam_slots)):
        local = q[slots]
        internal = float(local @ (k[slots, :][:, slots] @ local))
        work = float(local @ (applied[slots] + interface[slots]))
        error = abs(internal - work) / denominator
        if error > 1e-8:
            raise ValueError(name + " body virtual-work balance failed")
        result[name] = dict(internal_pairing_J=internal,
                            applied_plus_interface_pairing_J=work,
                            relative_work_error=error)
    return result


def evaluate():
    affinity = psutil.Process().cpu_affinity()
    pools = threadpool_info()
    if (affinity != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(pool["num_threads"] != 1 for pool in pools)):
        raise ValueError("effective CPU/thread/JIT settings differ")
    model = continuum.specimen()
    load, faces, rigid_work, parity_roundoff = end_traction(model)
    k, _ = continuum.assemble_body(model)
    c, scale = continuum.constraints(model, "weak")
    q, lam, interface, support, external_checks = solve_continuum(
        model, k, c, scale, load)
    order, index, _, _, _ = continuum.layout(model)
    shell_slots = np.array([index[n] + axis for n in model["shell_nodes"]
                            for axis in range(3)])
    beam_slots = np.array([index[n] + axis for n in model["beam_nodes"]
                           for axis in range(3)])
    continuum_work = body_work(k, q, load, interface, shell_slots, beam_slots)
    external_wrench = wrench(model, load + interface + support)
    if not np.allclose(external_wrench, 0., rtol=0., atol=1e-8):
        raise ValueError("continuum physical wrench did not balance")

    owner = ss.create_analysis(definition=native.definition())
    try:
        tip = 105
        response = owner.solve_linear((ss.NodalAction(
            "beam", tip, (0., 0., 0., 0., 0., MOMENT_NM)),))
        if response.status != "completed":
            raise ValueError("native reference-linear Mz solve incomplete")
        if lam.shape != (20,) or response.multipliers.shape != (20,):
            raise ValueError("unchanged five-channel by four-edge P0 layout required")
        reference_density = lam.reshape(5, 4).copy()
        native_equivalent_density = -response.multipliers.reshape(5, 4).copy()
        edge_x = np.asarray([station["x"] for station in model["stations"]])
        edge_lengths = np.diff(edge_x)
        edge_centers = .5 * (edge_x[:-1] + edge_x[1:])
        if not np.array_equal(edge_lengths, np.full(4, .25)):
            raise ValueError("four unchanged quarter-length supports required")
        def density_action(values):
            return dict(force_per_edge_N=(values[:3] * edge_lengths).T.tolist(),
                        fy_force_per_edge_N=(values[1] * edge_lengths).tolist(),
                        mz_force_arm_per_edge_Nm=(values[1] * edge_lengths * edge_centers).tolist(),
                        resultant_force_N=(values[:3] @ edge_lengths).tolist(),
                        resultant_mz_from_fy_Nm=float(np.sum(values[1] * edge_lengths * edge_centers)))
        density_comparison = dict(
            convention="C_ref=beam-shell; C_native=shell-beam; comparable physical density is -native multiplier",
            reference_density_channel_major=reference_density.tolist(),
            native_equivalent_density_channel_major=native_equivalent_density.tolist(),
            difference_channel_major=(native_equivalent_density-reference_density).tolist(),
            reference_action=density_action(reference_density),
            native_action=density_action(native_equivalent_density),
            reference_translation_density_norm=float(np.linalg.norm(reference_density[:3])),
            difference_translation_density_norm=float(np.linalg.norm(
                native_equivalent_density[:3]-reference_density[:3])),
            reference_director_density_norm=float(np.linalg.norm(reference_density[3:])),
            difference_director_density_norm=float(np.linalg.norm(
                native_equivalent_density[3:]-reference_density[3:])),
            meaning="diagnostic P0 action distribution; mixed force/director units are not collapsed into one quotient")
        nq = np.r_[response.shell_displacement, response.beam_coordinates]
        native_load = owner._accepted[1]
        node = owner._beam_ids.index(tip)
        tip_spin = float(response.beam_coordinates[6 * node + 5])
        native_work = float(native_load @ nq)
        if abs(native_work - MOMENT_NM * tip_spin) > 1e-12:
            raise ValueError("native nodal Mz work mismatch")
        ns = response.reactions[:owner._shell_count].reshape(-1, 6)
        xyz = np.asarray(owner._definition.shell.coordinates_m)
        support_wrench = np.r_[ns[:, :3].sum(axis=0),
                               (ns[:, 3:] + np.cross(xyz, ns[:, :3])).sum(axis=0)]
        # Native Kq+C.T lambda=f: the exerted action is -C.T lambda.
        ni = -np.asarray(owner._C.T @ response.multipliers).reshape(-1)
        native_work_by_body = body_work(
            owner._K, nq, native_load, ni.reshape(-1),
            np.arange(owner._shell_count),
            np.arange(owner._shell_count, len(nq)))
        ni = ni[:owner._shell_count].reshape(-1, 6)
        interface_wrench = np.r_[ni[:, :3].sum(axis=0),
                                 (ni[:, 3:] + np.cross(xyz, ni[:, :3])).sum(axis=0)]
        if not np.allclose(support_wrench,
                           (0., 0., 0., 0., 0., -MOMENT_NM),
                           rtol=0., atol=1e-8):
            raise ValueError("native physical support wrench did not balance")
        native_checks = dict(
            status=response.status,
            free_equilibrium_ratio=float(response.free_equilibrium_ratio),
            compatibility=float(response.constraint_max),
            tip_rotation_z_rad=tip_spin,
            applied_pairing_J=native_work,
            body_work=native_work_by_body,
            support_wrench=support_wrench.tolist(),
            interface_wrench_shell=interface_wrench.tolist(),
            physical_dofs=int(len(nq)),
            multiplier_rows=int(owner._C.shape[0]),
        )
    finally:
        owner.close()

    continuum_tip_rotation = float(load @ q / MOMENT_NM)
    return dict(identity=IDENTITY, status="DEVELOPMENT_ONLY_NOT_QUALIFIED",
                model_sha256=model["model_sha256"],
                connection="five-channel material-director weak P0; normal spin free",
                load="consistent axial traction -Mz*y/Iz on bar x=1 face",
                section_end_faces=faces,
                end_traction_parity_roundoff_N=parity_roundoff,
                end_traction_rigid_work_N_Nm=rigid_work.tolist(),
                continuum=dict(external_checks,
                               body_work=continuum_work,
                               work_conjugate_tip_rotation_z_rad=continuum_tip_rotation,
                               applied_wrench=wrench(model, load).tolist(),
                               interface_wrench_shell=wrench(model, interface, "shell").tolist(),
                               support_wrench=wrench(model, support).tolist(),
                               total_wrench=external_wrench.tolist()),
                native=native_checks,
                interface_density=density_comparison,
                comparison=dict(
                    tip_rotation_difference_rad=tip_spin-continuum_tip_rotation,
                    tip_rotation_relative_difference=(
                        tip_spin-continuum_tip_rotation) /
                    max(abs(continuum_tip_rotation), 1e-12),
                    interpretation="same common five-channel work; different beam/shell body spaces and independent native half-cell ports; no physical accuracy score"),
                runtime=dict(cpu_affinity=affinity,
                             numba_disable_jit=int(numba.config.DISABLE_JIT),
                             numba_threads=int(numba.get_num_threads()),
                             loaded_threadpools=pools),
                source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                               for name, module in (("native", native), ("continuum", continuum))})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("preserve existing evidence; choose a fresh output")
    args.output.mkdir(parents=True)
    result = evaluate()
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "continuum_rotation": result["continuum"]["work_conjugate_tip_rotation_z_rad"],
                      "native_rotation": result["native"]["tip_rotation_z_rad"],
                      "continuum_equilibrium": result["continuum"]["free_equilibrium_ratio"],
                      "native_equilibrium": result["native"]["free_equilibrium_ratio"]}))


if __name__ == "__main__":
    main()
