"""One fixed BL5 force-couple discriminator; not F03 or fidelity acceptance."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import argparse
import json

import numpy as np
from threadpoolctl import threadpool_info
import numba
import anyfesolver
from anyfesolver import stiffened_shell
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from test_stiffened_shell_l1_assembly_experimental import _bl5_joint_mesh_definition


LOADS = {"mz": (0., 0., 0., 0., 0., 10.),
         "mx": (0., 0., 0., 10., 0., 0.),
         "fy": (0., 100., 0., 0., 0., 0.)}


def main(output, case="mz"):
    if output.exists():
        raise FileExistsError("preserve earlier development output")
    pools = threadpool_info()
    if not pools or any(p["num_threads"] != 1 for p in pools) or not numba.config.DISABLE_JIT:
        raise RuntimeError("effective one-thread/JIT-off settings required")
    base = _bl5_joint_mesh_definition(16, 16, 1)
    definition = replace(base, fixed=base.fixed+tuple(
        stiffened_shell.FixedCoordinate("beam", 101, i) for i in range(5)))
    owner = stiffened_shell.create_analysis(definition=definition)
    action = stiffened_shell.NodalAction("beam", 117, LOADS[case])
    result = owner.solve_linear((action,))
    if result.status != "completed" or result.free_equilibrium_ratio > 1e-8 or result.constraint_max > 1e-9:
        raise AssertionError("mixed equilibrium/weak constraint failed")
    shell_count = owner._shell_count
    positions = np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])
    frames = np.array([owner._beam_nodes[n][1] for n in owner._beam_ids])
    state = Mechanical(positions, np.zeros_like(positions), frames,
        np.tile(np.eye(3), (len(owner._beam_parts), 2, 1, 1)),
        np.zeros((len(owner._beam_parts), 18)))
    root = positions[0]
    recovered = np.zeros(owner._K.shape[0])
    beam_force = np.zeros(3); shell_force = np.zeros(3)
    beam_moment = np.zeros(3); shell_moment = np.zeros(3)
    force_arm_mz = 0.; normal_couple_max = 0.; pullback_error = 0.
    records = []
    for sample, (rows, basis) in zip(owner._line_samples, owner._line_terms):
        local, slots, measure, _, _ = sample
        density = basis@result.multipliers[rows]
        high, low, rb = local.port.pose_pair(state)
        x_b = high+low
        ns = np.array((1.-local.station, local.station))
        x_s = ns@local.reference
        director = rb@local.reference_director
        arm_s = .5*local.thickness*director
        arm_b = rb@local.reference_arm
        # Direct variations of the declared ct/cd work, before any native
        # interpolation/pullback. MINUS gradients are loads on the domains.
        fb = rb@density[:3]
        fs = -fb
        director_couple = np.cross(director, rb@local.director_basis.T@density[3:])
        mb = np.cross(arm_b, fb)+director_couple
        ms = np.cross(arm_s, fs)-director_couple
        np.testing.assert_allclose(x_s+arm_s, x_b+arm_b, rtol=0., atol=1e-13)
        np.testing.assert_allclose(np.cross(x_s-root,fs)+ms+np.cross(x_b-root,fb)+mb,
                                   0., atol=1e-10)
        # Independent virtual-work pullback: native port translation is the
        # endpoint interpolation PLUS its retained-cell reference lift.
        expected = np.r_[ns[0]*fs, ns[0]*ms, ns[1]*fs, ns[1]*ms,
                         (1.-local.port.weight)*fb, local.port.weight*fb,
                         mb+np.cross(state.cell_rotations[local.port.element_index,
                                                        local.port.cell]@local.port.lift, fb)]
        evaluated = local.evaluate(np.zeros((2, 6)), state, density)
        error = np.max(np.abs(expected+evaluated.gradient))
        pullback_error = max(pullback_error, float(error))
        np.testing.assert_allclose(expected, -evaluated.gradient, rtol=1e-10, atol=1e-10)
        np.add.at(recovered, slots, measure*expected)
        shell_force += measure*fs; beam_force += measure*fb
        shell_moment += measure*(np.cross(x_s-root,fs)+ms)
        beam_moment += measure*(np.cross(x_b-root,fb)+mb)
        force_arm_mz += measure*np.cross(x_s-root,fs)[2]
        normal_couple_max = max(normal_couple_max, float(abs(director@ms)), float(abs(director@mb)))
        records.append(dict(x_shell=x_s.tolist(), x_beam=x_b.tolist(), measure_m=float(measure),
            multiplier_density=density.tolist(), shell_force_density=fs.tolist(),
            beam_force_density=fb.tolist(), shell_moment_density=ms.tolist(),
            beam_moment_density=mb.tolist()))
    assembled = -np.asarray(owner._C.T@result.multipliers)
    parity = float(np.max(np.abs(assembled-recovered)))
    np.testing.assert_allclose(recovered, assembled, rtol=1e-10, atol=1e-9)
    drill = np.arange(5, shell_count, 6)
    drilling_entries = owner._C[:, drill].data
    drilling_max = float(np.max(np.abs(drilling_entries))) if drilling_entries.size else 0.
    if drilling_max > 1e-12 or normal_couple_max > 1e-10:
        raise AssertionError("unexpected direct normal-spin/drilling transfer")
    applied = np.asarray(LOADS[case])
    applied_moment = applied[3:]+np.cross(positions[-1]-root,applied[:3])
    np.testing.assert_allclose(beam_moment[2], -applied_moment[2], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(shell_moment[2], applied_moment[2], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(force_arm_mz, applied_moment[2], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(beam_force+shell_force, 0., atol=1e-9)
    np.testing.assert_allclose(beam_moment+shell_moment, 0., atol=1e-9)
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    rng = np.random.default_rng(790)
    virtual = rng.standard_normal(len(q))*owner._coordinate_scale
    work_error = float(abs(recovered@virtual-assembled@virtual))
    assert work_error <= 1e-8*max(abs(assembled@virtual), 1e-12)
    load = owner._accepted[1]
    np.testing.assert_allclose(q@(owner._K@q), q@load, rtol=1e-8, atol=1e-12)
    ground_positions = np.vstack((owner._shell_xyz, positions))
    nodal = result.reactions[:6*len(ground_positions)].reshape(-1, 6)
    ground_moment = (nodal[:, 3:]+np.cross(ground_positions-root,nodal[:, :3])).sum(axis=0)
    np.testing.assert_allclose(ground_moment, -applied_moment, rtol=1e-8, atol=1e-7)
    beam_ground = result.reactions[shell_count:shell_count+6*len(positions)].reshape(-1,6)
    beam_ground_force = beam_ground[:,:3].sum(axis=0)
    beam_ground_moment = (beam_ground[:,3:]+np.cross(positions-root,beam_ground[:,:3])).sum(axis=0)
    np.testing.assert_allclose(beam_moment+beam_ground_moment+applied_moment,0.,atol=1e-7)
    np.testing.assert_allclose(beam_force+beam_ground_force+applied[:3],0.,atol=1e-7)
    root_normal_reaction = float(result.reactions[shell_count+5])
    assert root_normal_reaction == 0.
    output.write_text(json.dumps(dict(status="PASSED_DEVELOPMENT_ONLY", source_sha256=sha256(
        Path(__file__).read_bytes()).hexdigest(), imported_anyfesolver=anyfesolver.__file__, threads=pools,
        jit_disabled=bool(numba.config.DISABLE_JIT), case=case, load=LOADS[case],
        shell_edges=16, beam_halfcells=16,
        multiplier_rows=owner._C.shape[0], model_identity=owner.preflight()["identity"],
        free_equilibrium_ratio=result.free_equilibrium_ratio, constraint_max=result.constraint_max,
        applied_work_j=float(q@load), beam_line_moment_about_root_nm=beam_moment.tolist(),
        shell_line_moment_about_root_nm=shell_moment.tolist(), shell_force_arm_mz_nm=float(force_arm_mz),
        direct_normal_couple_density_max=normal_couple_max, drilling_column_max=drilling_max,
        pullback_error_max=pullback_error, assembly_parity_max=parity, virtual_work_error=work_error,
        beam_root_normal_ground_reaction_nm=root_normal_reaction, station_records=records,
        beam_ground_force_n=beam_ground_force.tolist(), beam_ground_moment_about_root_nm=beam_ground_moment.tolist(),
        nodal_shell_poses={str(n):result.shell_displacement[6*i:6*i+6].tolist() for i,n in enumerate(owner._shell_ids)},
        nodal_beam_poses={str(n):result.beam_coordinates[6*i:6*i+6].tolist() for i,n in enumerate(owner._beam_ids)},
        beam_retained_coordinates=result.beam_coordinates[6*len(owner._beam_ids):].tolist(),
        shell_internal_covectors=(owner._K[:shell_count,:shell_count]@q[:shell_count]).tolist(),
        beam_internal_covectors=(owner._K[shell_count:,shell_count:]@q[shell_count:]).tolist(),
        shell_connection_actions=recovered[:shell_count].tolist(), beam_connection_pullback=recovered[shell_count:].tolist(),
        support_reactions=result.reactions.tolist(),
        limitation="Distributed force-couple path only; no response-drilling independence, axial torsion accuracy, F03 or M2 acceptance"
    ), indent=2, allow_nan=False)+"\n")
    owner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(LOADS), default="mz")
    args = parser.parse_args()
    main(args.output,args.case)
