"""Rigid-carrier normal-spin torque discriminator; ordinary development only.

The continuous BL5 limit fixes beam-axis transverse position but releases spin
about the carrier normal. Its Timoshenko solution is an independent analytical
reference, not an exact oracle for a finite P0 constraint space. No F03 authority
or production qualification is conferred by running this helper.
"""
from dataclasses import replace
from pathlib import Path
import argparse, json, math, sys, time, traceback

import numpy as np


def carrier_reference(length, bending, shear, moment, x):
    """Solution of EI*phi''-kGA*phi=0 with free root spin and tip moment.

    Order: phi, phi', Vy, Mz. Positive Vy=kGA*(v'-phi); v=0.
    Exponential evaluation avoids overflowing cosh/sinh for slender limits.
    """
    values = (length, bending, shear, moment, x)
    if (not all(math.isfinite(value) for value in values) or
            min(length, bending, shear) <= 0. or not 0. <= x <= length):
        raise ValueError("finite positive physical section and in-member coordinate required")
    kappa = math.sqrt(shear)/math.sqrt(bending)
    z = kappa*length
    if not math.isfinite(z) or z <= 0. or not math.isfinite(kappa) or kappa <= 0.:
        raise ValueError("analytical length/section combination is not representable")
    denominator = -math.expm1(-2.*z)
    forward, reverse = math.exp(kappa*(x-length)), math.exp(-kappa*(x+length))
    phi = moment*(forward+reverse)/(bending*kappa*denominator)
    slope = moment*(forward-reverse)/(bending*denominator)
    return np.array((phi, slope, -shear*phi, bending*slope))


def rigid_carrier_definition(nx=16, ny=4, halfcells=32):
    from anyfesolver import stiffened_shell as api
    from b3_e4_bl5_coherent_aggregation_dev import model
    definition, policy = model(nx, ny, halfcells, 2)
    shell_fixed = tuple(api.FixedCoordinate("shell", node, axis)
                        for node in definition.shell.node_ids for axis in range(6))
    # Fixing numerical carrier spin does not constrain beam normal spin: the
    # BL5 physical port has only position/director work. Check its actual work
    # below rather than inferring this from the names of rotation coordinates.
    beam_fixed = tuple(api.FixedCoordinate("beam", 101, axis) for axis in range(5))
    beam_fixed += (api.FixedCoordinate("beam", 101+halfcells, 1),)
    return replace(definition, fixed=shell_fixed+beam_fixed), policy


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run(output, nx=16, ny=4, halfcells=32):
    import numba
    from threadpoolctl import threadpool_info
    import anyfesolver
    from anyfesolver import stiffened_shell as api
    from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
    from b3_e4_bl5_coherent_aggregation_dev import physical_transfer
    output.mkdir(exist_ok=False)
    pools = threadpool_info()
    if (not pools or any(p["num_threads"] != 1 for p in pools) or
            numba.get_num_threads() != 1 or not numba.config.DISABLE_JIT):
        raise RuntimeError("one effective numerical thread and JIT off required")
    definition, policy = rigid_carrier_definition(nx, ny, halfcells)
    section = definition.beams[0].instantiate()[0].section.elastic
    if not np.array_equal(section, np.diag(np.diag(section))):
        raise ValueError("analytical in-plane comparator requires uncoupled section")
    length, moment = 1., 10.
    bending, shear = float(section[5, 5]), float(section[1, 1])
    reference = carrier_reference(length, bending, shear, moment, length)
    summary = dict(status="IN_PROGRESS", policy=policy,
        source=anyfesolver.__file__, settings=dict(pools=pools,
            numba_threads=numba.get_num_threads(), jit_disabled=bool(numba.config.DISABLE_JIT)),
        physical=dict(length_m=length, EI_z_Nm2=bending, kGA_y_N=shear,
            tip_moment_Nm=moment, beam_axis_z_m=.0675, shell_top_z_m=.0175,
            connection_arm_z_m=-.05, carrier="all nodal positions/directors fixed",
            beam_root="coordinates0-4 fixed, normal spin free", beam_tip="uy fixed, Mz applied"),
        analytic=dict(phi_tip_rad=float(reference[0]), tip_work_J=float(moment*reference[0]),
            root_natural_Mz_Nm=0., direct_normal_couple_density=0.),
        interpretation="continuous rigid-carrier limit versus finite native P0 enforcement; not elastic coupled accuracy",
        norms=dict(reference_length_m=length, force_N=abs(moment)/length,
            moment_Nm=abs(moment),work_floor_J=1e-12,
            equilibrium_work_balance_cap=1e-8,weak_gap_cap=1e-9),
        checks={}, comparison={})
    write(output/"summary.json", summary)
    owner = None
    try:
        started = time.perf_counter()
        owner = api.create_analysis(definition=definition)
        summary.update(setup_seconds=time.perf_counter()-started,
            preflight=owner.preflight(), matrix_order=int(owner._A.shape[0]),
            matrix_nnz=int(owner._A.nnz))
        # Preserve the true sparse indefinite operator; no dense condensation.
        from scipy.sparse import save_npz
        for name, matrix in (("K",owner._K),("C",owner._C),("A",owner._A)):
            save_npz(output/(name+".npz"), matrix)
        tip = owner._shell_count+6*owner._beam_index[101+halfcells]
        load = np.zeros(owner._K.shape[0]); load[tip+5] = moment
        solved = owner.solve_linear((api.NodalAction("beam",101+halfcells,(0.,0.,0.,0.,0.,moment)),))
        q = np.r_[solved.shell_displacement, solved.beam_coordinates]
        ct, g = owner._C.T@solved.multipliers, owner._K@q
        xyz = np.vstack((owner._shell_xyz,
            np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])))
        root = xyz[len(owner._shell_ids)]
        reactions = solved.reactions[:6*len(xyz)].reshape(-1,6)
        ground = np.r_[reactions[:,:3].sum(axis=0),
            (reactions[:,3:]+np.cross(xyz-root,reactions[:,:3])).sum(axis=0)]
        np.savez_compressed(output/"state.npz", q=q, load=load,
            multipliers=solved.multipliers, reactions=solved.reactions,
            internal=g, connection_exerted=-ct, weak_gaps=owner._C@q,
            free=owner._free, fixed=owner._fixed, coordinate_scales=owner._coordinate_scale,
            xyz=xyz)
        transfer = physical_transfer(owner, solved)
        write(output/"transfer.json", transfer)
        # Physical port moment at the axis, distinct from force-arm work at
        # nodal/retained coordinates or integrated moments about another origin.
        positions = np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])
        frames = np.array([owner._beam_nodes[n][1] for n in owner._beam_ids])
        state = Mechanical(positions,np.zeros_like(positions),frames,
            np.tile(np.eye(3),(len(owner._beam_parts),2,1,1)),np.zeros((len(owner._beam_parts),18)))
        normal_couple, normal_axis_moment = 0., 0.
        for (local, slots, weight, edge, alpha),(rows,basis) in zip(owner._line_samples,owner._line_terms):
            density = basis@solved.multipliers[rows]
            rb = local.port.pose_pair(state)[2]
            director = rb@local.reference_director
            force = rb@density[:3]
            couple = np.cross(director,rb@local.director_basis.T@density[3:])
            axis_moment = np.cross(rb@local.reference_arm,force)+couple
            normal_couple = max(normal_couple,abs(float(couple@director)))
            normal_axis_moment = max(normal_axis_moment,abs(float(axis_moment@director)))
        work = float(q@load)
        checks = dict(equilibrium=float(solved.free_equilibrium_ratio),weak_gap=float(solved.constraint_max),
            energy_work=abs(float(q@g)-work)/max(abs(work),1e-12),
            connection_work=abs(float(q@ct))/max(abs(work),1e-12),
            support_work=abs(float(q@solved.reactions))/max(abs(work),1e-12),
            total_wrench_balance=float(np.max(abs(ground+np.array((0.,0.,0.,0.,0.,moment)))/
                np.array((abs(moment)/length,)*3+(abs(moment),)*3))),
            direct_normal_couple=normal_couple*length/abs(moment),
            axis_normal_couple=normal_axis_moment*length/abs(moment),
            fixed_state=float(np.max(abs(q[owner._fixed]))))
        summary.update(checks=checks, native=dict(phi_tip_rad=float(q[tip+5]),tip_work_J=work,
            beam_root_phi_rad=float(q[owner._shell_count+5]),support_wrench_about_beam_root=ground.tolist()),
            comparison=dict(tip_work_relative=work/(moment*reference[0])-1.,
                tip_spin_relative=float(q[tip+5])/float(reference[0])-1.),
            solve_status=solved.status, preflight_after=owner.preflight())
        failures={name:value for name,value in checks.items()
                  if not math.isfinite(value) or value > (1e-9 if name == "weak_gap" else 1e-8)}
        if solved.status != "completed" or work <= 0.: failures["completion_positive_work"] = solved.status
        summary.update(status="PHYSICAL_CONTROL_FAILED" if failures else "COMPLETED_DEVELOPMENT_ONLY",
            failures=failures,
            decision="release/work failure: repair; correct work with response discrepancy: diagnose beam/weak enforcement; agreement only supports this limiting property")
        write(output/"summary.json", summary)
        return summary
    except BaseException as error:
        summary.update(status="ERROR_PROPAGATED",failure=dict(type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc()))
        write(output/"summary.json", summary)
        raise
    finally:
        if owner is not None: owner.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--nx",type=int,default=16)
    parser.add_argument("--ny",type=int,default=4)
    parser.add_argument("--halfcells",type=int,default=32)
    args=parser.parse_args()
    result=run(args.output,args.nx,args.ny,args.halfcells)
    print(json.dumps({name:result[name] for name in ("status","native","comparison","checks")},allow_nan=False))
    sys.exit(0 if result["status"] == "COMPLETED_DEVELOPMENT_ONLY" else 1)
