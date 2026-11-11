"""One saved reference-linear BL5 Mz completion derivative, development only.

K(alpha)=K+(alpha-1)Kn; all supports, loads and C stay fixed. No production
coefficient, law, mesh or qualification decision is changed by this experiment.
Use the saved coupled state, never an uncoupled common-action control.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_refined_solve(path, rhs, shape):
    """Load an owned same-RHS candidate; acceptance still uses the actual KKT."""
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        require(np.array_equal(data['rhs'], rhs), 'unchanged refined derivative RHS required')
        candidate = data['solution'].copy()
    require(candidate.shape == shape and candidate.dtype == np.float64 and
            np.isfinite(candidate).all(), 'finite binary64 refined derivative layout required')
    return candidate


def main(site, saved, output, reuse=None, refined=None):
    require(refined is None or reuse is not None, 'refined candidate requires bound retained solve')
    site, saved = site.resolve(strict=True), saved.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'classification': 'saved-system local derivative; no M2/F03 acceptance',
              'source_sha256': sha(__file__), 'parameter': 'common PL+hourglass scale at alpha=1',
              'limits': 'No finite-removal, drilling independence, causation of external discrepancy, or coefficient tuning'}
    started = time.perf_counter()
    try:
        require(not any(n.split('.')[0] == 'anyfesolver' for n in sys.modules), 'fresh product process required')
        identity = json.loads((saved/'identity.json').read_text())
        wheel = Path(identity['wheel_path'])
        require(sha(wheel) == identity['wheel_sha256'], 'saved wheel byte identity required')
        with zipfile.ZipFile(wheel) as z:
            for name, digest in identity['module_sha256'].items():
                require(hashlib.sha256(z.read(name)).hexdigest() == digest == sha(site/name),
                        'saved wheel/site module mismatch: '+name)
        bindings = {name: sha(saved/name) for name in
                    ('operators.npz', 'inputs.json', 'layout.json', 'station-traces.json',
                     'mz/raw-state.npz', 'mz/raw-state.json', 'mz/physical-transfer.json')}
        inputs = json.loads((saved/'inputs.json').read_text())
        layout = json.loads((saved/'layout.json').read_text())
        metadata = json.loads((saved/'mz/raw-state.json').read_text())
        require(bindings['operators.npz'] == inputs['operators_sha256'] and
                bindings['mz/raw-state.npz'] == metadata['arrays_sha256'] and metadata['status'] == 'completed',
                'completed saved coupled operator/state required')
        sys.path.insert(0, str(site))
        import numpy as np
        import numba
        from scipy import sparse
        from scipy.sparse.linalg import splu
        from threadpoolctl import threadpool_info
        from anyfesolver import stiffened_shell as ss, corotational as cr
        from anyfesolver._stiffened_shell_ge_experimental import (
            _shell_reference, _linear_equilibrium, _scaled_constraint_gap)
        from anyfesolver._b2_physical_linear_batch_dev import _accurate_sparse_residual
        pools = threadpool_info()
        require(pools and all(p['num_threads'] == 1 for p in pools) and
                numba.get_num_threads() == 1 and numba.config.DISABLE_JIT, 'effective thread1/JIT-off required')
        report.update(wheel_sha256=identity['wheel_sha256'], matching_modules=len(identity['module_sha256']),
                      saved_input_sha256=bindings, runtime={'pools': pools, 'numba_threads': 1, 'jit_disabled': True})
        with np.load(saved/'operators.npz', allow_pickle=False) as v:
            matrices = {name: sparse.csr_matrix((v[name+'_data'], v[name+'_indices'], v[name+'_indptr']),
                                               shape=tuple(v[name+'_shape'])) for name in ('K', 'C', 'A')}
            free, fixed = v['free'].copy(), v['fixed'].copy()
            scales, row_scales = v['coordinate_scale'].copy(), v['constraint_scale'].copy()
            maps = {name: v[name+'_rigid_work_map'].copy() for name in ('shell', 'beam')}
        with np.load(saved/'mz/raw-state.npz', allow_pickle=False) as v:
            q, load, lam, xyz = (v[name].copy() for name in ('q', 'load', 'multipliers', 'xyz_m'))
            require(np.array_equal(free, v['free']) and np.array_equal(fixed, v['fixed']) and
                    np.array_equal(scales, v['coordinate_scale']) and np.array_equal(row_scales, v['constraint_scale']),
                    'identical supported coordinate ordering and scales required')
        K, C, A = (matrices[n] for n in ('K', 'C', 'A'))
        require(layout['nx'] == 64 and layout['counts']['shell_nodes'] == 1105 and
                layout['counts']['beam_nodes'] == 129 and layout['counts']['members'] == 64 and
                layout['shell_node_ids'] == list(range(1, 1106)) and
                layout['beam_node_ids'] == list(range(101, 230)), 'declared finest typed node layout required')
        expected_load = np.zeros(8940)
        tip_start = 6*(layout['counts']['shell_nodes']+layout['counts']['beam_nodes']-1)
        require(inputs['loads']['mz'] == [0., 0., 0., 0., 0., 10.], 'declared pure tip Mz action required')
        expected_load[tip_start:tip_start+6] = inputs['loads']['mz']
        require(np.array_equal(load, expected_load), 'saved load must be the exact declared tip Mz action')
        require(K.shape == (8940, 8940) and C.shape == (160, 8940) and np.all(q[fixed] == 0.),
                'registered finest homogeneous supported coupled fixture required')
        expected_A = sparse.bmat([[K[free][:, free], C[:, free].T], [C[:, free], None]], format='csr')
        require((A-expected_A).nnz == 0, 'actual saved KKT layout required')
        baseline = _linear_equilibrium((K@q+C.T@lam-load)[free], load[free], scales[free])
        baseline_gap = _scaled_constraint_gap(C@q, row_scales)
        require(baseline <= 1e-8 and baseline_gap <= 1e-9, 'saved baseline physical gates required')
        s = inputs['shell']
        panel = ss.ShellPanel(node_ids=tuple(s['node_ids']), coordinates_m=tuple(map(tuple, s['coordinates_m'])),
            q4_elements=tuple(map(tuple, s['q4_elements'])), thickness_m=s['thickness_m'], young_pa=s['young_pa'],
            poisson=s['poisson'], density_kg_m3=s['density_kg_m3'])
        model, ids, coords, index, pieces = _shell_reference(panel)
        shell_count = 6*len(ids)
        require(shell_count == 6630 and len(pieces) == 1024 and np.array_equal(coords, xyz[:len(ids)]),
                'unchanged flat shell geometry/counts required')
        if reuse is None:
            refs = cr._corotational_cache(model)
            rows, cols, data = [], [], {n: [] for n in ('core', 'pl', 'hourglass')}
            material = model.get_material('shell')
            for eid, element, slots, origin in pieces:
                components = element.compute_stiffness_components(model.mesh, material)
                require(not components['legacy_fallback'] and not components['warped_direct'],
                        'qualified planar component decomposition required')
                _, rigid, deformed = cr._corotational_deformation_state(refs[eid], element, np.zeros(24))
                E = cr._dense_block_rotation(rigid, 4)
                rows.extend(np.repeat(slots, 24)); cols.extend(np.tile(slots, 24))
                for name in data:
                    tangent, geometric = cr._consistent_corotational_tangent_components(
                        refs[eid], element, deformed, np.zeros((4, 3)), rigid, E, np.zeros(24), components[name])
                    require(np.all(geometric == 0.), 'stress-free mapping required')
                    data[name].extend(tangent.ravel())
            parts = {name: sparse.coo_matrix((values, (rows, cols)), shape=K.shape).tocsr()
                     for name, values in data.items()}
            Kn = parts['pl']+parts['hourglass']
            reconstructed = parts['core']+Kn
            shell_saved = K[:shell_count, :shell_count]
            err = reconstructed[:shell_count, :shell_count]-shell_saved
            scale_shell = scales[:shell_count]
            weighted = sparse.diags(scale_shell)@err@sparse.diags(scale_shell)
            reference = sparse.diags(scale_shell)@shell_saved@sparse.diags(scale_shell)
            parity = float(np.linalg.norm(weighted.data)/max(np.linalg.norm(reference.data), 1e-30))
            require(parity <= 1e-10 and Kn[shell_count:, :].nnz == 0 and Kn[:, shell_count:].nnz == 0,
                    'mapped component reconstruction or exact beam/retained zeros failed')
        else:
            reuse = reuse.resolve(strict=True)
            previous = json.loads((reuse/'result.json').read_text())
            require(previous['status'] == 'failed' and previous['source_sha256'] ==
                    'be7eb0f008974aad72d9970b906038c9bfa5b2caceed4113dccd267482d682ca' and
                    previous['saved_input_sha256'] == bindings and
                    previous['wheel_sha256'] == identity['wheel_sha256'], 'exact failed-v1 dependency binding required')
            old_runner = reuse.parent/'b3_e4_bl5_completion_sensitivity_dev.py'
            require(sha(old_runner) == previous['source_sha256'], 'original consumed runner binding required')
            for name, digest in {
                'numerical-completion.npz': 'b710b71f7581a8761168b39a319bb9d14cfac394207a616d1b78649634606287',
                'solve-raw.npz': '1ec75347e297a1856f7a2c5e91bc5eba5bba57fe0ea0a62748ae9b9be9bc5800',
                'derivative.npz': '2842095c689b675a672a36da4933e35a8debe48cff3df21deb410e10e3ccadaf'} .items():
                require(sha(reuse/name) == digest, 'exact failed-v1 raw evidence required: '+name)
            # The bound v1 writes these files only after its reconstruction gate.
            # Reuse that successful predicate, without inventing an unrecorded value.
            Kn = sparse.load_npz(reuse/'numerical-completion.npz')
            parity = None
            report['reused_component_gate'] = {'source_sha256': previous['source_sha256'], 'limit': 1e-10,
                'result': 'passed before retained derivative predicate failure', 'numeric_value': 'not recorded'}
        numerical_action = Kn@q
        rhs = np.r_[-numerical_action[free], np.zeros(C.shape[0])]
        if reuse is None:
            factor = splu(A.tocsc())
            derivative = factor.solve(rhs)
            derivative -= factor.solve(_accurate_sparse_residual(A, derivative[:, None], rhs[:, None])[:, 0])
        else:
            with np.load(reuse/'solve-raw.npz', allow_pickle=False) as v:
                require(np.array_equal(rhs, v['rhs']), 'unchanged derivative RHS required')
                derivative = v['derivative'].copy()
        if refined is not None:
            refined = refined.resolve(strict=True)
            derivative = load_refined_solve(refined, rhs, derivative.shape)
            report['refined_solve'] = {'path': str(refined), 'sha256': sha(refined),
                'note': 'Retained repair candidate; no factorization in this assessment. '
                        'Repair work and all earlier failures remain separate evidence.'}
        dq = np.zeros_like(q); dq[free] = derivative[:len(free)]
        dl = derivative[len(free):]
        np.savez(output/'solve-raw.npz', derivative=derivative, dq=dq, multipliers=dl, rhs=rhs)
        sparse.save_npz(output/'numerical-completion.npz', Kn)
        d_internal, d_constraint = K@dq+numerical_action, C.T@dl
        residual = d_internal+d_constraint
        accurate = _accurate_sparse_residual(A, derivative[:, None], rhs[:, None])[:, 0]
        denominator = abs(A)@abs(derivative)+abs(rhs)
        require(np.isfinite(denominator).all(), 'finite backward-error denominator required')
        require(np.all(accurate[denominator == 0.] == 0.), 'nonzero defect in a zero equation row')
        ratios = np.zeros_like(accurate)
        np.divide(abs(accurate), denominator, out=ratios, where=denominator != 0.)
        backward = float(np.max(ratios))
        weighted_defect = accurate[:len(free)]*scales[free]
        weighted_rhs = rhs[:len(free)]*scales[free]
        eq = float(np.linalg.norm(weighted_defect)/np.linalg.norm(weighted_rhs)) if np.any(weighted_rhs) else (
            0. if np.all(weighted_defect == 0.) else float('inf'))
        original_load_control = _linear_equilibrium(accurate[:len(free)], load[free], scales[free])
        gap = _scaled_constraint_gap(C@dq, row_scales)
        dr = np.zeros_like(q); dr[fixed] = residual[fixed]
        np.savez(output/'derivative.npz', dq=dq, multipliers=dl, internal=d_internal,
                 constraint_resisting=d_constraint, reactions=dr, numerical_action=numerical_action)
        report['derivative_diagnostic_controls'] = {'componentwise_backward_error': backward,
            'upper_physical_work_norm_ratio': eq, 'original_physical_load_reference_ratio': original_load_control,
            'unsupported_manufactured_active_row_ratio': _linear_equilibrium(residual[free], -numerical_action[free], scales[free])}
        require(backward <= 1e-8 and eq <= 1e-8 and original_load_control <= 1e-8 and gap <= 1e-9
                and np.all(dq[fixed] == 0.) and np.isfinite(derivative).all(),
                'derivative equilibrium, compatibility or finite-state failure')
        wrenches = {domain: {name: (work_map@vector).tolist() for name, vector in
                    (('internal', d_internal), ('connection_exerted', -d_constraint), ('ground', dr))}
                    for domain, work_map in maps.items()}
        require(max(np.max(np.abs(maps[d]@(d_internal+d_constraint-dr))) for d in maps) <= 1e-7,
                'derivative domain wrench balance failed')
        traces = json.loads((saved/'station-traces.json').read_text())
        def transfer(multiplier):
            moments = np.zeros((3, 6)); recovered = np.zeros_like(q); records = []
            for t in traces:
                density = np.array(t['density_basis'])@multiplier[t['multiplier_rows']]
                R = np.array(t['beam_port']['frame']); direction = R@t['reference_director']
                fb = R@density[:3]; couple = np.cross(direction, R@np.array(t['director_basis']).T@density[3:])
                mb = np.cross(R@t['reference_arm_m'], fb)+couple
                ms = np.cross(.5*t['thickness_m']*direction, -fb)-couple
                a, b = t['shell_alpha'], t['beam_port']['weight']
                pullback = np.r_[(1-a)*(-fb), (1-a)*ms, a*(-fb), a*ms,
                                (1-b)*fb, b*fb, mb+np.cross(R@t['beam_port']['lift_m'], fb)]
                np.add.at(recovered, t['full_slots'], t['measure_m']*pullback)
                i, j = t['beam_port']['node_indices']
                beam_xyz = xyz[len(ids):]; xb = (1-b)*beam_xyz[i]+b*beam_xyz[j]
                xs = (1-a)*np.array(t['shell_reference_m'][0])+a*np.array(t['shell_reference_m'][1])
                wrench = np.r_[fb, mb+np.cross(xb-xyz[len(ids)], fb)]
                for power in range(3): moments[power] += t['measure_m']*xs[0]**power*wrench
                records.append({'x_m': float(xs[0]), 'measure_m': t['measure_m'], 'wrench_density': wrench.tolist()})
            relative = np.max(np.abs((recovered+C.T@multiplier)*scales))/max(
                np.max(np.abs((C.T@multiplier)*scales)), 1e-12)
            require(relative <= 1e-8, 'physical transpose work map failed')
            return moments, records, float(relative)
        base_transfer, _, base_map = transfer(lam)
        derivative_transfer, records, derivative_map = transfer(dl)
        saved_transfer = json.loads((saved/'mz/physical-transfer.json').read_text())
        np.testing.assert_allclose(base_transfer, saved_transfer['polynomial_transfer_moments'], rtol=1e-8, atol=1e-8)
        report.update(counts={'q': len(q), 'constraints': C.shape[0],
            'kkt_rows': A.shape[0], 'shell_elements': len(pieces)}, factorizations=1 if reuse is None else 0,
            controls={'baseline_equilibrium': baseline, 'baseline_gap': baseline_gap, 'mapped_component_relative': parity,
                      'derivative_equilibrium': eq, 'derivative_gap': gap, 'base_work_map': base_map,
                      'derivative_work_map': derivative_map},
            applied_work_j=float(load@q), applied_work_derivative_j=float(load@dq),
            applied_work_relative_derivative=float((load@dq)/(load@q)),
            signed_completion_pairing_j=float(q@numerical_action),
            work_note='Direct f^T dq; no assumed symmetry/SPD or automatic equality to -q^T Kn q.',
            beam_tip_derivative=dq[tip_start:tip_start+6].tolist(),
            shell_component_derivative_max=np.max(np.abs(dq[:shell_count].reshape(-1, 6)), axis=0).tolist(),
            multiplier_derivative_max=float(np.max(np.abs(dl))), domain_wrench_derivatives=wrenches,
            transfer_derivative_moments=derivative_transfer.tolist(), station_derivative_records=records)
        report['status'] = 'COMPLETED_DEVELOPMENT_ONLY'
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        report['imports'] = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.copy().items()
                             if n.split('.')[0] == 'anyfesolver' and getattr(m, '__file__', None)}
        require(all(Path(p).is_relative_to(site) for p in report['imports'].values()), 'all product imports must be onsite')
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('status', 'elapsed_seconds')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('site', 'saved', 'output'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--reuse-solve', type=Path)
    p.add_argument('--refined-solve', type=Path,
                   help='Assess an owned, same-RHS retained repair without another factorization')
    a = p.parse_args()
    main(a.site, a.saved, a.output, a.reuse_solve, a.refined_solve)
