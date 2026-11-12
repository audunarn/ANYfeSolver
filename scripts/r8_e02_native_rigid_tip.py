"""Restricted E02 development comparison; no scientific acceptance adjudication.

Use an installed ANYfeSolver. A free unloaded beam stub supplies the public
footprint pose port, not a second load path. The rigid patch occupies x>=.8
on a flat isotropic panel. Fz/Mx/My test the out-of-plane physical sector;
neither six-channel continuum equivalence nor normal-spin physics is claimed.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time


LEVELS = ((10, 4), (20, 8), (40, 16))
LOADS = {"Fz": (0., 0., 10., 0., 0., 0.),
         "Mx": (0., 0., 0., 1., 0., 0.),
         "My": (0., 0., 0., 0., 1., 0.)}


def geometry(nx, ny):
    if (nx, ny) not in LEVELS:
        raise ValueError("registered three-level geometry required")
    nodes = {1+j*(nx+1)+i: (i/nx, -.1+.2*j/ny, 0.)
             for j in range(ny+1) for i in range(nx+1)}
    cells = []
    patch = []
    for j in range(ny):
        for i in range(nx):
            a = 1+j*(nx+1)+i
            cell = (a, a+1, a+nx+2, a+nx+1)
            cells.append(cell)
            if 5*i >= 4*nx:
                patch.append(cell)
    patch_nodes = tuple(sorted({n for cell in patch for n in cell}))
    root = tuple(1+j*(nx+1) for j in range(ny+1))
    return nodes, tuple(cells), tuple(patch), patch_nodes, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--profile", choices=("b2", "ge"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.site.resolve()))
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_info
    from anyfesolver import b3_ge, stiffened_shell as api
    import anyfesolver
    assert Path(anyfesolver.__file__).resolve().is_relative_to(args.site.resolve())
    pools = threadpool_info()
    if any(p.get("num_threads") != 1 for p in pools):
        raise RuntimeError("effective numerical pools must use one thread")
    if os.environ.get("NUMBA_DISABLE_JIT") != "1":
        raise RuntimeError("declared JIT-off fixture required")
    started = time.monotonic()
    nodes, cells, patch, patch_nodes, root = geometry(args.nx, args.ny)
    e, nu, t = 210e9, .3, .02
    g = e/(2*(1+nu))
    origin = (.9, 0., args.sign*.05)
    area, iy, iz, jt = .0004, 1.3333333333333335e-8, 1.3333333333333335e-8, 2.249e-8
    if args.profile == "b2":
        section = api.LinearSection.isotropic(E=e, nu=nu, area=area,
            Iy=iy, Iz=iz, J=jt, ky=5/6, kz=5/6)
        members = tuple(api.PhysicalB2Member(f"port-stub-{i}", f"E02-free-port-{i}",
            (100001+i, 100002+i), ((.9+.05*i, 0., origin[2]),
            (.9+.05*(i+1), 0., origin[2])), (0., 1., 0.), "port-only", section)
            for i in range(2))
        shells = tuple(api.Q4PanelShell(f"cell-{i}", cell, e, nu, t)
                       for i, cell in enumerate(cells))
        group = api.B2NodalRigidFootprint("rigid-tip", patch_nodes, 100001, patch)
        count = 6*(len(nodes)+3)
        fixed = {6*(n-1)+d for n in root for d in range(6)}
        free = [i for i in range(count) if i not in fixed]
        definition = api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, (), sparse.eye(count, format="csr")[:, free],
            np.zeros(count), (group,))
        selection = api.B2NodalRigidFootprintSelection(("rigid-tip",))
    else:
        section = b3_ge.EllipsoidalGeneralizedSection(
            np.diag((e*area, g*area*5/6, g*area*5/6, g*jt, e*iy, e*iz)),
            np.eye(6), 1e20, 1e9)
        xyz = np.array((origin, (.95, 0., origin[2]), (1., 0., origin[2])))
        member = b3_ge.define_beam("b3-ge", 1, (100001, 100002, 100003),
            xyz, np.tile(np.eye(3), (3, 1, 1)), section, np.eye(6))
        panel = api.ShellPanel(tuple(nodes), tuple(nodes.values()), cells, t, e, nu, 7850.)
        group = api.NodalRigidFootprint("rigid-tip", patch_nodes, 100001, patch)
        definition = api.LinearDefinition(api.B3_GE_E4_PL_STIFFENED_SHELL_V1,
            panel, (member,), (), tuple(api.FixedCoordinate("shell", n, d)
            for n in root for d in range(6)),
            q4_pullback_policy=api.Q4_ALGORITHMIC_PULLBACK_V1, footprints=(group,))
        selection = api.NodalRigidFootprintSelection(("rigid-tip",))
    owner = api.create_analysis(definition=definition)
    results = {}
    try:
        # Actual admitted reference matrix, sparse inspection only. Its cross
        # block is an implementation-sector diagnostic, not a positive norm.
        shell_count = 6*len(nodes)
        kshell = owner._K[:shell_count, :shell_count]
        membrane = np.array([6*i+d for i in range(len(nodes)) for d in (0, 1, 5)])
        bending = np.array([6*i+d for i in range(len(nodes)) for d in (2, 3, 4)])
        cross = kshell[membrane][:, bending]
        scale = float(np.max(np.abs(kshell.data)))
        cross_ratio = float(np.max(np.abs(cross.data), initial=0.))/scale
        # No hidden model-wide inverse/dense derivative tensor is constructed.
        for name, wrench in LOADS.items():
            if args.profile == "b2":
                load = np.zeros(count)
                load[shell_count:shell_count+6] = wrench
                result = owner.solve_linear(load, selected_footprints=selection,
                    recover_members={m.member_id: (0., .5, 1.) for m in members})
                if result.status != "completed":
                    raise RuntimeError(result.failure_reason)
                q = np.array(result.displacements)
                port = q[shell_count:shell_count+6]
                beam = q[shell_count:]
                shell_action, = result.footprint_wrenches
                transferred = np.r_[shell_action.shell_force_N, shell_action.shell_moment_Nm]
                support = np.array(result.affine_reactions)
                actions = [list(a.shell_direct_couple_Nm) for a in shell_action.node_actions]
                records = asdict(shell_action)
                factors = result.factorization_count
                beam_resultants = result.beam_resultants_local
                energy = result.internal_strain_energy
                # Public physical B2 element oracle, independent of mixed KKT.
                responses = [m.local_operator().evaluate(
                    m.local_dof_map()@beam[6*i:6*i+12]) for i,m in enumerate(members)]
                stub_force = np.concatenate([r.force for r in responses])
                stub_energy = sum(r.energy for r in responses)
            else:
                result = owner.solve_linear((api.NodalAction("beam", 100001, wrench),))
                if result.status != "completed":
                    raise RuntimeError("native reference-linear solve did not complete")
                q = np.r_[result.shell_displacement, result.beam_coordinates]
                port = result.beam_coordinates[:6]
                beam = result.beam_coordinates
                shell_action, = owner.recover(selection=selection)
                transferred = np.array(shell_action.shell_action)
                support = result.reactions
                records = asdict(shell_action)
                actions = [list(a.shell_action[3:]) for a in shell_action.point_wrenches]
                factors = owner.preflight()["reference_linear_factorizations"]
                kb = owner._K[shell_count:, shell_count:]
                stub_force = kb@beam
                stub_energy = .5*float(beam@stub_force)
                energy = .5*float(q@(owner._K@q))
                beam_resultants = None
            support_force = np.zeros(3)
            support_moment = np.zeros(3)
            for node in root:
                f, m = support[6*(node-1):6*(node-1)+3], support[6*(node-1)+3:6*node]
                support_force += f
                support_moment += m+np.cross(nodes[node], f)
            expected_force = -np.array(wrench[:3])
            expected_moment = -np.array(wrench[3:])-np.cross(origin, wrench[:3])
            shell = q[:shell_count].reshape((-1, 6))
            patch_gap = max(float(np.max(np.abs(shell[n-1, :3]-port[:3]-np.cross(port[3:], np.array(nodes[n])-origin)))) for n in patch_nodes)
            direct_mz = max(abs(a[2]) for a in actions)
            pair = float(np.array(wrench)@port)
            checks = dict(actual_shell_cross_block_ratio=cross_ratio,
                shell_membrane_drilling_coordinate_max=float(np.max(np.abs(shell[:, (0, 1, 5)]))),
                shell_normal_spin_max=float(np.max(np.abs(shell[:, 5]))),
                footprint_direct_Mz_max_Nm=direct_mz,
                stub_reference_force_max=float(np.max(np.abs(stub_force))),
                stub_quadratic_work_J=stub_energy,
                physical_patch_translation_gap_m=patch_gap,
                footprint_wrench_error=(transferred-np.array(wrench)).tolist(),
                root_wrench_error=np.r_[support_force-expected_force, support_moment-expected_moment].tolist(),
                port_offset_identity_m=[float(port[0]-origin[2]*port[4]), float(port[1]+origin[2]*port[3])])
            if cross_ratio > 1e-12 or checks['shell_normal_spin_max'] > 1e-12 or direct_mz > 1e-8:
                raise RuntimeError("out-of-plane decoupling not established")
            if (abs(stub_energy) > 1e-10*abs(pair) or
                float(np.max(np.abs(stub_force))) > 1e-7 or
                float(np.max(np.abs(transferred-np.array(wrench)))) > 1e-8):
                raise RuntimeError("free-stub/physical transfer control failed")
            if (patch_gap > 1e-9 or
                float(np.max(np.abs(support_force-expected_force))) > 1e-7 or
                float(np.max(np.abs(support_moment-expected_moment))) > 1e-7 or
                max(abs(v) for v in checks['port_offset_identity_m']) > 1e-9 or
                checks['shell_membrane_drilling_coordinate_max'] > 1e-12 or
                abs(2*energy-pair) > 1e-8*abs(pair)):
                raise RuntimeError("physical equilibrium/mapping/work control failed")
            results[name] = dict(status=result.status, port_translation_m=list(port[:3]),
                port_rotation_rad=list(port[3:]), external_final_pairing_J=pair,
                native_internal_strain_energy_J=energy, factors=factors,
                expected_root_force_N=expected_force.tolist(), expected_root_moment_Nm=expected_moment.tolist(),
                root_force_N=support_force.tolist(), root_moment_Nm=support_moment.tolist(),
                controls=checks, footprint=records, beam_resultants=beam_resultants)
        manifest = dict(scope="restricted reference-linear out-of-plane development; no E02/F03 acceptance",
            profile=args.profile, nx=args.nx, ny=args.ny, sign=args.sign,
            geometry=dict(length_m=1., width_m=.2, thickness_m=t, rigid_patch_x=(.8, 1.),
                          port_origin_m=origin, young_pa=e, poisson=nu),
            nodes=nodes, cells=cells, patch_cells=patch, patch_nodes=patch_nodes, root_nodes=root,
            loads=LOADS, source_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            installed_anyfesolver=str(anyfesolver.__file__), numerical_pools=pools,
            runtime_seconds=time.monotonic()-started, results=results)
        with (args.output/'result.json').open('x') as f:
            json.dump(manifest, f, indent=2); f.write('\n')
        print(json.dumps(dict(status="COMPLETED_DEVELOPMENT_ONLY", profile=args.profile,
            nx=args.nx, ny=args.ny, sign=args.sign, patch_nodes=len(patch_nodes),
            factors=factors, cases={k:dict(pairing_J=v['external_final_pairing_J'],
            rotation_rad=v['port_rotation_rad']) for k,v in results.items()})))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
