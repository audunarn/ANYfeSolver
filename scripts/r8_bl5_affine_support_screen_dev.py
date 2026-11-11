"""One affine-density BL5 decision screen; no new carrier mesh or F03 run.

Compare independent continuous P0/P1 references on the SAME 8/16/32 supports,
then one native 16x4/32-halfcell rigid-carrier fixture against its actual affine
reference. Physical geometry/supports/loads are inherited unchanged. This is
normal-spin bending, not Saint-Venant torsion, external agreement or qualification.
"""
from dataclasses import replace
from pathlib import Path
from fractions import Fraction
import argparse, json

import numpy as np


def affine_force_moment(left, right, a, b, order):
    """Exact stored-coefficient integral of x**order*(a+b*xi)."""
    left, right, a, b = map(lambda v: Fraction(float(v)), (left, right, a, b))
    slope = 2*b/(right-left)
    constant = a-b-slope*left
    return float(constant*(right**(order+1)-left**(order+1))/(order+1)+
                 slope*(right**(order+2)-left**(order+2))/(order+2))


def run(output):
    from scipy.integrate import quad
    from scipy.sparse import save_npz
    import numba, anyfesolver
    from threadpoolctl import threadpool_info
    from anyfesolver import stiffened_shell as api
    from b3_e4_bl5_coherent_aggregation_dev import model, physical_transfer
    from r8_bl5_affine_carrier_reference_dev import solve_affine_carrier
    from r8_bl5_p0_carrier_reference_dev import solve_p0_carrier
    from r8_bl5_rigid_carrier_dev import carrier_reference

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pools = threadpool_info()
    if (not pools or any(p["num_threads"] != 1 for p in pools) or
            numba.get_num_threads() != 1 or not numba.config.DISABLE_JIT):
        raise RuntimeError("one effective numerical thread/JIT off required")
    definition, policy = model(16, 4, 32, 2, basis="discontinuous_p1")
    fixed = tuple(api.FixedCoordinate("shell", node, axis)
        for node in definition.shell.node_ids for axis in range(6))
    fixed += tuple(api.FixedCoordinate("beam", 101, axis) for axis in range(5))
    fixed += (api.FixedCoordinate("beam", 133, 1),)
    definition = replace(definition, fixed=fixed)
    section = definition.beams[0].instantiate()[0].section.elastic
    if not np.array_equal(section, np.diag(np.diag(section))):
        raise ValueError("uncoupled physical reference section required")
    bending, shear, moment = float(section[5, 5]), float(section[1, 1]), 10.
    strong = np.array([quad(lambda x: x**k*shear*carrier_reference(
        1., bending, shear, moment, x)[1], 0., 1.,
        epsabs=1e-11, epsrel=1e-12)[0] for k in range(4)])
    records = []
    for count in (8, 16, 32):
        endpoints = np.linspace(0., 1., count+1)
        p0 = solve_p0_carrier(1., bending, shear, moment, endpoints)
        p1 = solve_affine_carrier(1., bending, shear, moment, endpoints)
        records.append(dict(support_count=count,
            width_over_transfer_length=(1./count)/np.sqrt(bending/shear),
            p0_controls=p0.controls(), affine_controls=p1.controls(),
            p0_force_moments=p0.transfer_moments().tolist(),
            affine_force_moments=p1.transfer_moments().tolist(),
            p0_vs_strong=(p0.transfer_moments()/strong-1.).tolist(),
            affine_vs_strong=(p1.transfer_moments()/strong-1.).tolist()))
    summary = dict(status="IN_PROGRESS", qualification="EXPERIMENTAL_NOT_QUALIFIED",
        imports=dict(anyfesolver=anyfesolver.__file__), settings=dict(pools=pools,
            numba_threads=numba.get_num_threads(), jit_disabled=True),
        physical=dict(length_m=1., EI_z_Nm2=bending, kGA_y_N=shear, Mz_tip_Nm=moment,
            shell_width_m=.2, shell_thickness_m=.035, beam_axis_z_m=.0675,
            carrier="all nodal shell positions/directors fixed",
            beam_root="coordinates0-4 fixed; normal spin free", beam_tip="uy fixed; Mz=10Nm"),
        policy=policy, strong_tie_force_moments=strong.tolist(),
        independent_same_supports=records,
        limits="No finite flexible-carrier accuracy, seven-ratio stability, locking or external qualification.")
    def save():
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    save()
    owner = api.create_analysis(definition=definition)
    try:
        if owner._A.shape[0] > 100000:
            raise ValueError("development row ceiling exceeded")
        for name, matrix in (("K", owner._K), ("C", owner._C), ("A", owner._A)):
            save_npz(output/(name+".npz"), matrix)
        summary.update(matrix_order=int(owner._A.shape[0]), nnz=int(owner._A.nnz),
                       preflight=owner.preflight())
        solved = owner.solve_linear((api.NodalAction("beam", 133, (0., 0., 0., 0., 0., moment)),))
        transfer = physical_transfer(owner, solved)
        (output/"transfer.json").write_text(json.dumps(transfer, indent=2, allow_nan=False)+"\n")
        q = np.r_[solved.shell_displacement, solved.beam_coordinates]
        load = owner._accepted[1]
        np.savez_compressed(output/"state.npz", q=q, load=load,
            multipliers=solved.multipliers, reactions=solved.reactions,
            weak_gaps=owner._C@q, free=owner._free, fixed=owner._fixed)
        endpoints = np.array(policy["support_endpoints_m"])
        supports = len(endpoints)-1
        coefficients = solved.multipliers[2*supports:4*supports].reshape(-1, 2)
        # At this admitted reference state rb=I; establish the physical Fy
        # coefficient mapping against recovered station actions before scoring.
        for record in transfer["station_records"]:
            x = record["x_m"]
            j = min(int(np.searchsorted(endpoints, x, side="right"))-1, supports-1)
            xi = 2.*(x-endpoints[j])/(endpoints[j+1]-endpoints[j])-1.
            np.testing.assert_allclose(record["beam_wrench_about_root_density"][1],
                coefficients[j, 0]+xi*coefficients[j, 1], rtol=1e-12, atol=1e-8)
        native = np.array([sum(affine_force_moment(left, right, a, b, order)
            for left, right, (a, b) in zip(endpoints[:-1], endpoints[1:], coefficients))
            for order in range(4)])
        np.testing.assert_allclose(native[:3],
            np.array(transfer["polynomial_transfer_moments"])[:, 1], rtol=1e-11, atol=1e-8)
        reference = solve_affine_carrier(1., bending, shear, moment,
                                         policy["support_endpoints_m"])
        work = float(q@load)
        checks = dict(equilibrium=float(solved.free_equilibrium_ratio),
            weak_gap=float(solved.constraint_max),
            energy_work=abs(float(q@owner._K@q)-work)/abs(work),
            interface_work=abs(float(q@owner._C.T@solved.multipliers))/abs(work),
            support_work=abs(float(q@solved.reactions))/abs(work),
            released_normal_couple=max(abs(r["beam_wrench_about_root_density"][5]-
                r["x_m"]*r["beam_wrench_about_root_density"][1])/moment
                for r in transfer["station_records"]))
        summary.update(status="COMPLETED_DEVELOPMENT_ONLY", checks=checks,
            native_force_moments=native.tolist(),
            native_vs_same_affine=(native/reference.transfer_moments()-1.).tolist(),
            native_vs_strong=(native/strong-1.).tolist(),
            pointwise_translation_gap_max_m=transfer["pointwise_translation_gap_max_m"])
        save()  # Numerical evidence survives a failed scientific/control predicate.
        if (solved.status != "completed" or work <= 0. or
                checks["weak_gap"] > 1e-9 or
                any(not np.isfinite(value) or value > 1e-8 for key, value in checks.items()
                    if key != "weak_gap")):
            summary["status"] = "FAILED_PHYSICAL_CONTROL"; save()
            raise ValueError("affine screen physical control failed")
        print(json.dumps(dict(status=summary["status"],
            native_vs_same_affine=summary["native_vs_same_affine"],
            native_vs_strong=summary["native_vs_strong"], checks=checks)))
    finally:
        owner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
