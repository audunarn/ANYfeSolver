"""Bounded installed affine BL5 comparison on two explicit mechanical grids.

External rows are independently assembled, never imported into the native owner.
Common physical fields are compared separately from native retained coordinates.
Raw numerical evidence precedes all physical/comparison predicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import zipfile

WHEEL_SHA = "ef96a35e7d24912500d42422be59171b0f4e9f5830157e0088d226960ce5a06b"
LOADS = {"fy": (0., 100., 0., 0., 0., 0.), "mx": (0., 0., 0., 10., 0., 0.),
         "mz": (0., 0., 0., 0., 0., 10.)}
EXTERNAL_CHECKS = {
    'all_free_coordinate_equilibrium_scaled', 'average_five_channel_gap_scaled', 'emitted_coordinate_gap_scaled',
    'local_recovery_residual_scaled', 'independent_arm_pullback_scaled', 'each_support_action_reaction_scaled',
    'interface_wrench_scaled', 'ground_plus_applied_wrench_scaled', 'complete_domain_cut_wrench_scaled',
    'physical_work_relative', 'internal_applied_work_relative', 'compatible_interface_work_relative',
    'solved_dual_work_relative', 'virtual_dual_work_relative', 'continuous_density_nodal_total_wrench_scaled'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_transfer_moments(records, bounds=(0., 1.)):
    """Recover affine force/intrinsic couple on each existing two-Gauss segment.

    About-root torque is quadratic because it includes x e_x cross force.
    x^2-weighted torque is quartic; two Gauss points must not integrate that.
    Fixed straight fixture only, with root at x=0 on the beam axis.
    """
    from math import sqrt
    require(len(records) % 2 == 0, 'complete two-Gauss segment records required')
    terms = [[[] for _ in range(6)] for _ in range(3)]
    for i in range(0, len(records), 2):
        one, two = sorted(records[i:i+2], key=lambda r: r['x_m'])
        x1, x2 = one['x_m'], two['x_m']
        width = (x2-x1)*sqrt(3.)
        left, right = (x1+x2-width)/2, (x1+x2+width)/2
        require(width > 0 and math.isclose(one['measure_m']+two['measure_m'], width,
                                          rel_tol=1e-12, abs_tol=1e-14), 'physical segment measure required')
        left, right = max(left, bounds[0]), min(right, bounds[1])
        if right <= left:
            continue
        w1, w2 = one['beam_wrench_about_root_density'], two['beam_wrench_about_root_density']
        f1, f2 = w1[:3], w2[:3]
        c1 = [w1[3], w1[4]+x1*f1[2], w1[5]-x1*f1[1]]
        c2 = [w2[3], w2[4]+x2*f2[2], w2[5]-x2*f2[1]]
        fs = [(b-a)/(x2-x1) for a, b in zip(f1, f2)]
        fc = [a-x1*b for a, b in zip(f1, fs)]
        cs = [(b-a)/(x2-x1) for a, b in zip(c1, c2)]
        cc = [a-x1*b for a, b in zip(c1, cs)]
        p0 = fc+cc
        p1 = fs+[cs[0], cs[1]-fc[2], cs[2]+fc[1]]
        p2 = [0., 0., 0., 0., -fs[2], fs[1]]
        for order in range(3):
            for component in range(6):
                terms[order][component].append(math.fsum(coeff[component]*(right**(order+power+1)
                    -left**(order+power+1))/(order+power+1) for power, coeff in enumerate((p0, p1, p2))))
    return [[math.fsum(component) for component in row] for row in terms]


def support_model(coherent, nx, support_count=8, *, basis='discontinuous_p1'):
    """Construct either fixed-eight or one complete growing-space candidate.

    The explicit growing space remains experimental, not a selected policy.
    Never infer support locations from an automatic resolution policy here.
    """
    from dataclasses import replace
    require(type(nx) is int and type(support_count) is int and (nx, support_count) in ((16, 8), (32, 8), (32, 16)),
            'only declared fixed-eight or fine growing-sixteen grid admitted')
    require(basis in ('discontinuous_p1', 'discontinuous_p1_cell_average'), 'explicit tested density basis required')
    require(basis != 'discontinuous_p1_cell_average' or (nx,support_count)==(16,8),
            'projected helper is bounded to the checked coarse fixture')
    # Reuse the frozen geometry constructor; substitute only this explicitly
    # identified density law before constructing any owner.
    definition, policy = coherent.model(nx, nx//4, 2*nx, 2, basis='discontinuous_p1')
    span = 2*nx//support_count
    definition = replace(definition, lines=(replace(definition.lines[0], multiplier_span=span,basis=basis),))
    policy = dict(policy, span_halfcells=span,
        support_membership=[list(range(a, a+span)) for a in range(0, 2*nx, span)],
        support_endpoints_m=[i/support_count for i in range(support_count+1)],
        fixed_support_policy=('SAME_EIGHT_AFFINE_MECHANICAL_DISCRIMINATOR_V1' if support_count == 8
                              else 'GROWING_COMPLETE_AFFINE_SPAN4_CANDIDATE_V1'),
        c_lambda_actual=nx/support_count,
        scope='explicit development space, not qualified production/stability policy')
    if basis == 'discontinuous_p1_cell_average':
        policy.update(basis=basis,density_basis=('1','fixed reference half-cell arclength mean of xi'),
            fixed_support_policy='SAME_EIGHT_PROJECTED_DENSITY_MECHANICAL_DISCRIMINATOR_V1')
    return definition, policy


def fixed_support_model(coherent, nx):
    return support_model(coherent, nx, 8)


def run(site, wheel, source, fixtures, external, output, *, nx=16, cases=('fy', 'mx', 'mz'), support_count=8,
        basis='discontinuous_p1', expected_wheel_sha=WHEEL_SHA):
    require(nx in (16, 32) and cases and all(c in LOADS for c in cases), 'bounded grid/load request required')
    require((nx, support_count) in ((16, 8), (32, 8), (32, 16)), 'declared support/grid pairing required')
    require(nx == 16 or support_count == 16 or tuple(cases) == ('mx',), 'the fixed-eight finer discriminator is Mx only')
    require(basis in ('discontinuous_p1', 'discontinuous_p1_cell_average'), 'explicit tested density basis required')
    require(basis != 'discontinuous_p1_cell_average' or (nx,support_count)==(16,8), 'projected comparison is bounded to the checked coarse fixture')
    require(not any(n.split('.')[0] == 'anyfesolver' for n in sys.modules), 'fresh installed process required')
    site, fixtures = site.resolve(strict=True), fixtures.resolve(strict=True)
    sys.path[:0] = [str(site), str(fixtures)]
    import numpy as np
    import numba
    from threadpoolctl import threadpool_info
    import anyfesolver
    from anyfesolver import stiffened_shell
    import b3_e4_bl5_coherent_aggregation_dev as coherent
    import b3_e4_bl5_matched_mz_native_dev as matched
    import b3_e4_bl5_matched_p0_native_refinement_dev as inventory
    import r8_bl5_external_affine_reference_dev as independent
    inventory.np = np
    inventory.matched = matched
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'FAILED', 'scope': 'bounded flexible physical development; no locking/F03/accuracy acceptance',
               'source_sha256': sha(__file__), 'basis': basis, 'nx': nx, 'support_count': support_count, 'cases': {}}
    try:
        _, _, service, _, reference, _ = matched.reference_services()
    except BaseException as error:
        summary['failure'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        raise
    write = lambda name, value: service._atomic(output/(name+'.json'), matched.plain(value))
    owner = None
    try:
        require(isinstance(expected_wheel_sha,str) and len(expected_wheel_sha)==64
                and sha(wheel) == expected_wheel_sha, 'declared installed wheel required')
        modules = {}
        source_files = {'anyfesolver/'+p.relative_to(source/'src/anyfesolver').as_posix(): p
                        for p in (source/'src/anyfesolver').rglob('*.py')}
        with zipfile.ZipFile(wheel) as archive:
            names = {n for n in archive.namelist() if n.startswith('anyfesolver/') and n.endswith('.py')}
            require(len(names) == 417 and names == set(source_files), 'complete417module graph required')
            for name in sorted(names):
                data = archive.read(name)
                require(data == (site/name).read_bytes() == source_files[name].read_bytes(), 'module binding: '+name)
                modules[name] = hashlib.sha256(data).hexdigest()
        pools = threadpool_info()
        require(pools and all(p['num_threads'] == 1 for p in pools) and numba.get_num_threads() == 1
                and numba.config.DISABLE_JIT, 'effective one-thread/JIT-off settings required')
        write('identity', {'wheel_sha256': expected_wheel_sha, 'wheel': str(wheel), 'module_sha256': modules,
              'anyfesolver_import': anyfesolver.__file__, 'pools': pools, 'numba_threads': numba.get_num_threads(),
              'jit_disabled': bool(numba.config.DISABLE_JIT),
              'helpers': {Path(m.__file__).name: sha(m.__file__) for m in (coherent, matched, inventory)}})
        definition, policy = support_model(coherent, nx, support_count, basis=basis)
        owner = stiffened_shell.create_analysis(definition=definition)
        xyz, layout = inventory.layout(owner, definition, policy, nx)
        trace = inventory.station_inventory(owner)
        ns, nb = len(owner._shell_ids), len(owner._beam_ids)
        require(owner._C.shape == (10*support_count, 6*(ns+nb)+24*nx) and ns == (nx+1)*(nx//4+1) and nb == 2*nx+1,
                'fixed native physical/retained layout required')
        require(policy['span_halfcells'] == 2*nx//support_count and policy['support_endpoints_m'] == [i/support_count for i in range(support_count+1)],
                'same declared complete physical supports required')
        require(np.allclose(xyz[:ns], [(i/nx, .2*j/(nx//4), 0.) for j in range(nx//4+1) for i in range(nx+1)],
                            rtol=0, atol=1e-13), 'fixed shell geometry required')
        require(np.allclose(xyz[ns:], [(i/(2*nx), 0., reference.BEAM_AXIS_Z_M) for i in range(2*nx+1)],
                            rtol=0, atol=1e-13), 'fixed beam geometry required')
        elastic = np.diag((reference.E_PA*reference.BEAM_AREA_M2,
            reference.G_PA*reference.BEAM_SHEAR_AREA_M2, reference.G_PA*reference.BEAM_SHEAR_AREA_M2,
            reference.G_PA*reference.BEAM_TORSION_M4, reference.E_PA*reference.BEAM_BENDING_M4,
            reference.E_PA*reference.BEAM_BENDING_M4))
        require(all(np.array_equal(part[0].section.elastic, elastic) for part in owner._beam_parts)
                and definition.shell.thickness_m == reference.SHELL_THICKNESS_M
                and definition.shell.young_pa == reference.E_PA and definition.shell.poisson == reference.NU,
                'unchanged physical sections required')
        for station in trace:
            require(np.allclose(station['beam_port']['frame'], np.eye(3), rtol=0, atol=1e-13)
                and np.allclose(station['reference_arm_m'], (0, 0, reference.BEAM_BOTTOM_ARM_M), rtol=0, atol=1e-13)
                and np.allclose(station['reference_director'], (0, 0, 1), rtol=0, atol=1e-13)
                and np.linalg.norm(station['beam_port']['lift_m']) <= 1e-13, 'physical reference ports required')
        maps = matched.rigid_maps(owner, xyz)
        matched.save_npz(output/'operators.npz', inventory.sparse_arrays(owner, maps))
        write('inputs', {'policy': policy, 'layout': layout, 'xyz_m': xyz, 'loads': LOADS,
            'shell': {'thickness_m': definition.shell.thickness_m, 'young_pa': definition.shell.young_pa,
                      'poisson': definition.shell.poisson, 'q4_elements': definition.shell.q4_elements},
            'beams': [json.loads(b.raw) for b in definition.beams], 'line': definition.lines[0].descriptor(),
            'fixed': layout['fixed_coordinates'], 'station_traces': trace})
        ids = [r['external_node'] for r in layout['typed_physical_nodes']]
        shell_index = {n: i for i, n in enumerate(owner._shell_ids)}
        cells = [tuple(shell_index[n] for n in cell) for cell in definition.shell.q4_elements]
        for case in cases:
            wrench = LOADS[case]
            folder = output/case; folder.mkdir()
            cw = lambda name, value: service._atomic(folder/(name+'.json'), matched.plain(value))
            result = owner.solve_linear((stiffened_shell.NodalAction('beam', 101+2*nx, wrench),))
            q = np.r_[result.shell_displacement, result.beam_coordinates]
            load = np.array(owner._accepted[1], copy=True)
            g, ct = owner._K@q, owner._C.T@result.multipliers
            matched.save_npz(folder/'raw-state.npz', {'q': q, 'load': load, 'multipliers': result.multipliers,
                'Kq': g, 'connection_resisting': ct, 'reactions': result.reactions, 'xyz_m': xyz,
                'external_ids': ids, 'weak_gaps': owner._C@q, 'full_equilibrium': g+ct-load,
                'shell_poses': q[:6*ns].reshape(ns, 6), 'beam_nodal_poses': q[6*ns:6*(ns+nb)].reshape(nb, 6),
                'beam_retained': q[6*(ns+nb):].reshape(nx, 24)})
            transfer = coherent.physical_transfer(owner, result)
            exact = exact_transfer_moments(transfer['station_records'])
            np.testing.assert_allclose(exact[:2], transfer['polynomial_transfer_moments'][:2], rtol=1e-10, atol=1e-9)
            transfer['two_gauss_polynomial_transfer_moments'] = transfer['polynomial_transfer_moments']
            transfer['polynomial_transfer_moments'] = exact
            transfer['fixed_region_025_075_wrench'] = exact_transfer_moments(transfer['station_records'], (.25, .75))[0]
            transfer['moment_recovery'] = 'Exact affine force/couple polynomials; includes quadratic lever-arm wrench, no product quadrature change'
            cw('physical-transfer', transfer)
            internal, exerted, ground, applied = ({d: matrix@vector for d, matrix in maps.items()}
                for vector in (g, -ct, result.reactions, load))
            work = float(q@load); scale = max(map(abs, wrench)); ws = max(abs(work), 1e-12)
            checks = {'equilibrium_ratio': result.free_equilibrium_ratio, 'weak_gap': result.constraint_max,
                'energy_work_relative': abs(float(q@g)-work)/ws,
                'compatible_connection_work_relative': abs(float(q@ct))/ws,
                'ground_work_relative': abs(float(q@result.reactions))/ws,
                'complete_domain_cuts_scaled': max(float(np.max(np.abs(applied[d]+exerted[d]+ground[d])))/scale for d in maps),
                'global_ground_applied_scaled': float(np.max(np.abs(sum(ground.values())+sum(applied.values()))))/scale}
            cw('physical-actions', {'domain_wrenches': {'internal': internal, 'exerted': exerted, 'ground': ground,
                'applied': applied}, 'work_J': {'applied': work, 'internal': float(q@g)}, 'checks': checks,
                'root_internal': g[owner._fixed], 'root_connection_exerted': (-ct)[owner._fixed],
                'actual_ground': result.reactions[owner._fixed], 'fixed_slots': owner._fixed})
            require(result.status == 'completed', 'native solve must complete')
            for key, value in checks.items():
                require(math.isfinite(value) and value <= (1e-9 if key == 'weak_gap' else 1e-8), 'native physical check '+key)
            es = json.loads((external/case/'summary.json').read_text())
            ei = json.loads((external/case/'inputs.json').read_text())
            er = json.loads((external/case/'raw-inventory.json').read_text())
            ea = json.loads((external/case/'recovered-actions.json').read_text())
            require(es['status'] == 'completed' and es['case'] == case and es['nx'] == nx
                    and es['basis'] == basis and ea['basis'] == basis
                    and es['source_sha256'] == sha(independent.__file__),
                    'same independent external affine candidate required')
            ext_refinement, _, _, _, ext_reference = independent.services()
            ext_model = ext_refinement.geometry(nx, ext_reference)
            ext_model['support_count'] = support_count
            ext_pairing = independent.aggregate_rows(ext_model, ext_reference, basis=basis)
            expected = ext_refinement.planned_inventory(ext_refinement.command_recipe(
                ext_model, ext_reference, ext_pairing['emitted_rows'], case))
            require(all(er[k] == json.loads(json.dumps(v)) for k, v in expected.items()),
                    'complete matching external section/frame/connectivity/row/analysis command inventory required')
            require(ei['physical_rows'] == ext_pairing['physical_rows'] and ei['supports'] == json.loads(json.dumps(ext_pairing['supports']))
                    and ei['basis'] == ext_pairing['basis'], 'independent complete physical rows and same supports required')
            require(set(ea['checks']) == EXTERNAL_CHECKS and ea['checks'] == es['checks'],
                    'complete consistent external physical-control inventory required')
            for key, value in ea['checks'].items():
                require(math.isfinite(value) and value <= (1e-9 if key in
                    ('average_five_channel_gap_scaled', 'emitted_coordinate_gap_scaled') else 1e-8),
                    'external physical check '+key)
            require(set(map(int, er['poses'])) == set(ids) and np.allclose([er['nodes'][str(n)] for n in ids],
                    xyz, rtol=0, atol=1e-13) and ei['load'] == list(wrench), 'matched geometry/loads required')
            fixed = {(f.domain, f.node if f.domain == 'shell' else 100001+f.node-101, f.component) for f in definition.fixed}
            efixed = {('shell' if int(n) in set(owner._shell_ids) else 'beam', int(n), d)
                for n, flags in er['fixes'].items() for d, flag in enumerate(flags) if flag}
            require(fixed == efixed, 'same direct physical grounds required')
            other = np.array([er['poses'][str(n)] for n in ids]); native = q[:6*(ns+nb)].reshape(ns+nb, 6)
            require(np.isfinite(other).all(), 'finite external fields required')
            ework = ea['work_J']['applied']
            em = np.array([ea['continuous_density_measures']['polynomial_x_moments'][str(p)] for p in range(3)])
            cw('comparison', {'fields': {'shell': matched.field_errors(native[:ns], other[:ns], cells, xyz[:ns]),
                'beam': matched.field_errors(native[ns:], other[ns:], [(i, i+1) for i in range(nb-1)], xyz[ns:], True)},
                'native_applied_work_J': work, 'external_applied_work_J': ework,
                'work_relative_signed': (work-ework)/max(abs(ework), 1e-12),
                'native_transfer_moments': transfer['polynomial_transfer_moments'], 'external_transfer_moments': em,
                'transfer_difference': np.array(transfer['polynomial_transfer_moments'])-em,
                'fixed_region_difference': np.array(transfer['fixed_region_025_075_wrench'])
                    -np.array(ea['continuous_density_measures']['fixed_region_025_075']['open']),
                'beam_ground_difference': ground['beam']-np.array(ea['beam_ground_wrench']),
                'shell_ground_difference': ground['shell']-np.array(ea['shell_ground_wrench']),
                'native_tip_pose': native[-1], 'external_tip_pose': other[-1],
                'external_files': {n: sha(external/case/n) for n in ('summary.json', 'inputs.json', 'raw-inventory.json', 'recovered-actions.json')},
                'scope': 'same declared '+basis+' support action, different element and retained-trace discretizations; no locking/converged1% claim'})
            summary['cases'][case] = {'checks': checks, 'work_J': work, 'raw_state_sha256': sha(folder/'raw-state.npz')}
        require(owner.preflight()['reference_linear_factorizations'] == 1, 'one eligible factor for selected compatible loads')
        imports = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.copy().items()
                   if n.split('.')[0] == 'anyfesolver' and getattr(m, '__file__', None)}
        require(all(Path(p).is_relative_to(site) for p in imports.values()), 'canonical installed imports required')
        summary.update(status='COMPLETED_DEVELOPMENT_ONLY', imports=imports, preflight=owner.preflight())
    except BaseException as error:
        summary['failure'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        write('summary', summary)
        if owner is not None:
            owner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('site', 'wheel', 'source', 'fixtures', 'external', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--nx', type=int, choices=(16, 32), default=16)
    parser.add_argument('--case', choices=tuple(LOADS))
    parser.add_argument('--support-count', type=int, choices=(8, 16), default=8)
    parser.add_argument('--basis', choices=('discontinuous_p1','discontinuous_p1_cell_average'), default='discontinuous_p1')
    parser.add_argument('--wheel-sha', default=WHEEL_SHA)
    args = parser.parse_args()
    run(args.site, args.wheel, args.source, args.fixtures, args.external, args.output,
        nx=args.nx, cases=(args.case,) if args.case else tuple(LOADS), support_count=args.support_count,
        basis=args.basis, expected_wheel_sha=args.wheel_sha)
