"""Compare native and resolved beam work under one BL5 Mz interface density.

Ordinary source development. This is a new Mz beam-only action, not a retry of
the historical failed Fz shell/body job. It does not score the bonded line,
select a multiplier space, run an external solver, or consume formal F03.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numba
import numpy as np
import psutil
from scipy import sparse
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium
from scripts import r8_bl5_body_port_probe as port
from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_mz_work_matched_dev as mz
from scripts import r8_bl5_weak_continuum_reference as continuum


IDENTITY = "R8_BL5_MZ_COMMON_BEAM_ACTION_DEV_V2"


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def symmetry_controls(model, k, c, load, q, lam):
    """Prove the actual discrete anti-y-reflection sector before projection."""
    order, index, fixed, free, _ = continuum.layout(model)
    coordinates = model["physical_nodes"]
    shell_nodes = set(model["shell_nodes"])
    beam_nodes = set(model["beam_nodes"])
    rows, columns, values = [], [], []
    signs = (-1., 1., -1.)  # -diag(1,-1,1): Mz load's odd sector
    for node in order:
        x, y, z = coordinates[node]
        candidates = shell_nodes if node in shell_nodes else beam_nodes
        mirror = [other for other in candidates if
                  np.allclose(coordinates[other], (x, -y, z),
                              rtol=0., atol=1e-14)]
        if len(mirror) != 1:
            raise ValueError("unique reflected physical node required")
        for axis, sign in enumerate(signs):
            rows.append(index[node] + axis)
            columns.append(index[mirror[0]] + axis)
            values.append(sign)
    transform = sparse.csr_matrix((values, (rows, columns)), shape=k.shape)
    parity = np.repeat(np.array((-1., 1., -1., -1., 1.)), 4)
    dual = sparse.diags(parity)
    def maximum(matrix):
        return float(np.max(np.abs(matrix.data), initial=0.))
    k_error = maximum(transform.T @ k @ transform - k) / max(maximum(k), 1.)
    c_error = maximum(c @ transform - dual @ c) / max(maximum(c), 1.)
    load_error = float(np.max(np.abs(transform @ load-load))) / max(
        float(np.max(np.abs(load))), 1.)
    state_error = float(np.max(np.abs(transform @ q-q))) / max(
        float(np.max(np.abs(q))), 1e-12)
    density_error = float(np.max(np.abs((parity-1.) * lam))) / max(
        float(np.max(np.abs(lam))), 1e-12)
    fixed_set = set(int(slot) for slot in fixed)
    fixed_image = all(transform.indices[transform.indptr[slot]] in fixed_set
                      for slot in fixed)
    projected = lam * .5 * (1. + parity)
    residual = k @ q + c.T @ projected - load
    projected_ratio = _linear_equilibrium(residual[free], load[free],
                                          np.ones(len(free)))
    report = dict(status="CALCULATED_NOT_VALIDATED",
                  physical_reflection="(x,y,z)->(x,-y,z); displacement signs (-,+,-)",
                  five_channel_parity=(-1., 1., -1., -1., 1.),
                  k_relative_equivariance_error=k_error,
                  c_relative_equivariance_error=c_error,
                  load_relative_parity_error=load_error,
                  coupled_state_relative_parity_error=state_error,
                  density_relative_parity_error=density_error,
                  fixed_supports_reflection_invariant=fixed_image,
                  projected_density_channel_major=projected.reshape(5, 4).tolist(),
                  removed_density_channel_major=(lam-projected).reshape(5, 4).tolist(),
                  projected_original_equilibrium_ratio=float(projected_ratio))
    return projected, report, transform


def evaluate(output):
    affinity = psutil.Process().cpu_affinity()
    pools = threadpool_info()
    if (affinity != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(pool["num_threads"] != 1 for pool in pools)):
        raise ValueError("effective CPU/thread/JIT settings differ")

    model = continuum.specimen()
    external, faces, rigid_work, parity_roundoff = mz.end_traction(model)
    k, _ = continuum.assemble_body(model)
    c, scale = continuum.constraints(model, "weak")
    q, lam, interface, _, checks = mz.solve_continuum(
        model, k, c, scale, external)
    projected_lam, symmetry, transform = symmetry_controls(
        model, k, c, external, q, lam)
    save(output / "symmetry-calculated.json", symmetry)
    if (not symmetry["fixed_supports_reflection_invariant"] or
            max(symmetry[key] for key in (
                "k_relative_equivariance_error", "c_relative_equivariance_error",
                "load_relative_parity_error")) > 1e-11 or
            max(symmetry[key] for key in (
                "coupled_state_relative_parity_error",
                "density_relative_parity_error")) > 1e-8 or
            symmetry["projected_original_equilibrium_ratio"] > 1e-8):
        raise ValueError("discrete reflection or original equilibrium did not justify sector projection")
    order, index, _, _, _ = continuum.layout(model)
    beam_slots = np.array([index[n] + axis for n in model["beam_nodes"]
                           for axis in range(3)])
    reference_body_load = np.asarray(
        (external - c.T @ projected_lam)[beam_slots]).reshape(-1)
    body_reflection = transform[beam_slots, :][:, beam_slots]
    projected_body_load = .5 * (reference_body_load +
                                 body_reflection @ reference_body_load)
    load_roundoff = float(np.max(np.abs(projected_body_load-reference_body_load)))
    if load_roundoff / max(float(np.max(np.abs(reference_body_load))), 1.) > 1e-11:
        raise ValueError("reference beam load does not lie in proven reflection sector")
    reference_body_load = np.asarray(projected_body_load).reshape(-1)
    reference_k = k[beam_slots, :][:, beam_slots]
    reference_rigid = continuum.rigid_fields(model, "beam")[beam_slots, :]
    reference_wrench = reference_rigid.T @ reference_body_load
    if np.max(np.abs(reference_wrench)) > 1e-8:
        raise ValueError("resolved beam body load is unbalanced")

    owner = ss.create_analysis(definition=native.definition())
    try:
        ns = owner._shell_count
        if (owner._C.shape != (20, 168) or ns != 90 or
                owner._beam_count != 78 or lam.shape != (20,)):
            raise ValueError("unchanged native five-channel specimen required")
        native_external = np.zeros(owner._K.shape[0])
        native_external[ns + 6 * owner._beam_ids.index(105) + 5] = 1.
        # C_native=-C_ref on the proven common trace subset.  With reference
        # multiplier lam, the physical native exerted interface is C_native.T lam.
        native_interface = np.asarray(owner._C.T @ projected_lam).reshape(-1)
        native_body_load = native_external[ns:] + native_interface[ns:]
        rigid = port.native_rigid_fields(owner)
        native_wrench = port.require_balanced(native_body_load, rigid)
        if np.max(np.abs(native_wrench-reference_wrench)) > 1e-8:
            raise ValueError("common physical body wrench mapping failed")
        preflight = dict(
            identity=IDENTITY, status="PREFLIGHT_ONLY",
            model_sha256=model["model_sha256"],
            reference_end_faces=faces,
            end_traction_rigid_work_N_Nm=rigid_work.tolist(),
            parity_roundoff_N=parity_roundoff,
            reference_body_load_parity_roundoff_N=load_roundoff,
            continuum_coupled_checks=checks,
            original_reference_density_channel_major=lam.reshape(5, 4).tolist(),
            projected_common_density_channel_major=projected_lam.reshape(5, 4).tolist(),
            symmetry=symmetry,
            reference_beam_wrench_N_Nm=reference_wrench.tolist(),
            native_beam_wrench_N_Nm=native_wrench.tolist(),
            continuum_coupled_beam_work_J=float(q[beam_slots] @
                (external + interface)[beam_slots]),
            native_beam_load_covector=native_body_load.tolist(),
            source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                           for name, module in (("native", native),
                                                ("continuum", continuum),
                                                ("port", port), ("mz", mz))},
            runtime=dict(cpu_affinity=affinity,
                         numba_disable_jit=int(numba.config.DISABLE_JIT),
                         numba_threads=int(numba.get_num_threads()),
                         loaded_threadpools=pools))
        save(output / "preflight.json", preflight)

        reference_q, reference_beam = port.solve_body(
            reference_k, reference_body_load,
            np.ones(len(reference_body_load)), rigid=reference_rigid,
            diagnostics=output / "reference-beam-calculated.json")
        save(output / "reference-beam-solved.json", reference_beam)
        reference_work = float(reference_beam["paired_work_J"])
        beam_q, beam = port.solve_body(
            owner._K[ns:, ns:], native_body_load,
            owner._coordinate_scale[ns:], rigid=rigid,
            diagnostics=output / "beam-calculated.json")
        save(output / "beam-solved.json", beam)
        material = port.beam_material(owner, beam_q)
        save(output / "material.json", material)
        native_work = float(beam["paired_work_J"])
        denominator = max(abs(native_work), 1e-12)
        material_error = abs(2 * material["stored_material_energy_J"] -
                             native_work) / denominator
        if material_error > 1e-8:
            raise ValueError("native physical beam material work failed")
        report = dict(
            identity=IDENTITY, status="DEVELOPMENT_BODY_SOLVED_NOT_ACCEPTED",
            compute_inventory=dict(resolved_coupled_RHS=1, resolved_beam_RHS=1,
                                   native_beam_RHS=1,
                                   native_coupled_RHS=0, native_shell_RHS=0,
                                   external_runs=0, formal_F03_runs=0),
            common_density=projected_lam.reshape(5, 4).tolist(),
            symmetry=symmetry,
            continuum_beam_body_work_J=reference_work,
            native_beam_body_work_J=native_work,
            work_difference_J=native_work-reference_work,
            work_ratio=native_work/reference_work,
            native_equation_ratio=float(beam["equation_ratio"]),
            native_material_energy_relative_error=material_error,
            reference_beam_wrench_N_Nm=reference_wrench.tolist(),
            native_beam_wrench_N_Nm=native_wrench.tolist(),
            interpretation="same P0 physical density on proven common trace fields; different primal beam spaces and native retained half-cell ports remain",
            limitations="not a coupled native response or standalone constitutive compliance; no line-law, physical-accuracy, F03 or gate acceptance")
        save(output / "result.json", report)
        return report
    finally:
        owner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("preserve existing attempt evidence")
    args.output.mkdir(parents=True)
    report = evaluate(args.output)
    print(json.dumps({key: report[key] for key in (
        "status", "continuum_beam_body_work_J", "native_beam_body_work_J",
        "work_ratio", "native_equation_ratio")}))


if __name__ == "__main__":
    main()
