"""One saved-envelope auxiliary native beta; never mechanics/F03 acceptance.

The positive kinematic norm retains native half-cell directors. Reference shell
Grams are reused from the frozen scalar registration; only the coupled beam
reference is assembled. No legacy module globals or product equations change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import zipfile

import numpy as np
from scipy import linalg, sparse
from threadpoolctl import threadpool_info

from scripts import r8_bl5_projected_envelope_norm_dev as envelope
from scripts import r8_bl5_projected_envelope_owner_dev as owner_helpers
from scripts.r8_bl5_projected_ratio_owner_dev import plain
from scripts import r8_bl5_sparse_envelope_kernel_dev as saved
from scripts.r8_bl5_projected_auxiliary_norm_dev import integrated_p1, lower_gram
from scripts.b3_e4_bl5_riesz_controls_dev import shell_scalar, beam_norm
from scripts.b3_e4_bl5_reference_accuracy_dev import beam_bracket

IDENTITY = 'BL5_SAVED_ENVELOPE_NATIVE_AUXILIARY_BETA_DEV_V1'
SOURCE = Path(__file__).resolve().parents[1]
MAX_ROWS = 100_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fresh_output(output, root):
    output = Path(output).resolve()
    require(output.suffix == '.json' and output.is_relative_to(SOURCE),
            'owned repository .json output required')
    require(not output.is_relative_to(Path(root).resolve()), 'output cannot alter saved owner')
    require(not output.exists() and not output.with_suffix('.npz').exists(), 'fresh JSON/NPZ output required')
    require(not any(p in ('.git', '.agents', '.codex', '.aws') for p in output.parts), 'protected output path')
    return output


def validate_counts(case):
    envelope._case(case)
    counts = case.counts
    require(all(type(v) is int and 0 < v <= MAX_ROWS for v in counts.values()) and
            0 < case.supported_kkt_count <= MAX_ROWS, 'finite admitted counts/100k row ceiling required')
    return counts


def validate_reference_rows(case, subdivision):
    require(subdivision in (1, 2, 4, 8, 16, 32), 'declared reference preparation required')
    rows = 3*case.halfcells*(subdivision+1)
    require(rows <= MAX_ROWS, 'auxiliary reference exceeds 100k row ceiling')
    return rows


def finite(value, name):
    value = np.asarray(value)
    require(value.dtype.kind in 'fiu' and np.isfinite(value).all(), name+' must be finite real')
    return value


def json_record(report):
    return envelope._failure_json(plain(report))


def validate_policy(case, data):
    policy = data['policy']
    require(policy['basis'] == 'discontinuous_p1_cell_average' and
            policy['span_halfcells'] == case.span and
            np.allclose(policy['support_endpoints_m'], np.linspace(0., 1., case.supports+1),
                        rtol=0., atol=1e-13), 'complete declared density policy differs')


def frozen_owner(case, root):
    """Immutable evidence chain, independent of today's product source tree."""
    manifest = SOURCE/'docs/agent_plans/development/r8-bl5-projected-assessment-20261003/raw-manifest-v8.json'
    require(envelope.sha(manifest) == saved.MANIFEST_SHA, 'immutable campaign manifest differs')
    records = {}
    for version in (8, 7, 6, 5):
        record = json.loads(manifest.read_text())
        for row in record['files']:
            previous = records.setdefault(row['path'], row['sha256'])
            require(previous == row['sha256'], 'conflicting immutable file bindings')
        if version > 5:
            manifest = manifest.with_name(f'raw-manifest-v{version-1}.json')
            require(envelope.sha(manifest) == saved.previous_hash(record, version), 'immutable manifest hash chain differs')
    aggregate = saved.RAW/'envelope/aggregate-v1.json'
    require(envelope.sha(aggregate) == records[aggregate.relative_to(SOURCE).as_posix()], 'aggregate binding differs')
    row = json.loads(aggregate.read_text())['cases'][case.fixture_case]
    summary = json.loads((root/'summary.json').read_text())
    require(envelope.sha(root/'summary.json') == row['owner_summary_sha256'] and
            summary['status'] == 'COMPLETED_INSTALLED_DEVELOPMENT_ONLY' and summary['fixture'] == case.fixture and
            summary['wheel_sha256'] == owner_helpers.WHEEL_SHA and summary['matching_modules'] == 417 and
            summary['factorization_count'] == 1 and summary['accurate_refinements'] == 6 and
            summary['predicates'] and all(v is True for v in summary['predicates'].values()), 'frozen completed owner required')
    require(set(summary['native_input_sha256']) == {'inputs.json', 'operators.npz'}, 'complete native input inventory required')
    for name, digest in summary['native_input_sha256'].items():
        require(envelope.sha(root/'native'/name) == digest, 'saved actual native input differs')
    version = 2 if case.fixture_case in ('ns64_b4', 'ns64_b8') else 1
    registration = saved.RAW/'envelope/registrations'/case.fixture_case/f'registration-v{version}.json'
    require(envelope.sha(registration) == row['scalar_sha256'], 'independent scalar record differs')
    _, scalar_binding = owner_helpers.registration_identity(case, registration, SOURCE)
    reused = {'ns16_b1_2': saved.RAW/'ratios/whole_half/owner-v1',
              'ns16_b2': saved.RAW/'ratios/whole_double/owner-v1', 'ns32_b1': saved.RAW/'joint32/installed-v2/owner'}
    canonical = reused.get(case.fixture_case, saved.RAW/'envelope/owners'/case.fixture_case/'owner-v1')
    states = {}
    for load in ('fy', 'mx', 'mz'):
        folder = root/'native'/load
        state = json.loads((folder/'raw-state.json').read_text())
        for name in ('raw-state.json', 'raw-state.npz', 'physical-actions.json'):
            saved.frozen_file(folder/name, canonical/'native'/load/name, records)
        require(state['status'] == 'completed' and envelope.sha(folder/'raw-state.npz') == state['arrays_sha256'] and
                json.loads((folder/'physical-actions.json').read_text()) == summary['cases'][load], 'saved complete state/action differs')
        states[load] = {name: envelope.sha(folder/name) for name in ('raw-state.json', 'raw-state.npz', 'physical-actions.json')}
    return summary, dict(owner=str(root), summary_sha256=envelope.sha(root/'summary.json'), inputs_sha256=summary['native_input_sha256'],
                         states=states, scalar=scalar_binding, manifest_sha256=saved.MANIFEST_SHA,
                         wheel_sha256=summary['wheel_sha256'], historical_owner_source_sha256=summary['source_sha256'])


def installed_graph(site, wheel, summary):
    """Bind all original wheel/site Python modules; current product may differ."""
    require(envelope.sha(wheel) == summary['wheel_sha256'], 'frozen installed wheel differs')
    modules = summary['module_sha256']
    require(len(modules) == 417, 'complete 417-module historical graph required')
    with zipfile.ZipFile(wheel) as archive:
        names = {n for n in archive.namelist() if n.startswith('anyfesolver/') and n.endswith('.py')}
        installed_names = {'anyfesolver/'+p.relative_to(site/'anyfesolver').as_posix() for p in (site/'anyfesolver').rglob('*.py')}
        require(names == set(modules) == installed_names, 'complete historical wheel/site module names differ')
        for name in sorted(names):
            require(hashlib.sha256(archive.read(name)).hexdigest() == modules[name] == envelope.sha(site/name),
                    'historical wheel/site module mismatch: '+name)
    return modules


def scalar_floor(contract, raw, case):
    p = envelope.projection(case)
    require(np.array_equal(raw['projection'], p), 'registered projected density differs')
    d = np.diag(1/np.sqrt(envelope.density_mass_diagonal(case)))
    require(np.array_equal(raw['density_L2_scale'], d) and
            np.array_equal(contract['physical_L2_basis_scale'], np.diag(d)), 'physical L2 basis differs')
    g, m = raw['scalar_lower'], raw['reference_upper']
    n = 2*case.supports
    require(g.shape == m.shape == (n, n), 'complete scalar Gram dimensions required')
    finite(g, 'scalar Gram'); finite(m, 'reference Gram')
    cs2 = float(linalg.eigvalsh(d@m@d, d@g@d)[-1])
    rho = np.hypot(.05, .1/2)/float(envelope.ELL)
    product = max(2*cs2, max(2., 1+2*rho*rho)*max(2*cs2, 1+rho*rho))
    floor = 1/np.sqrt(product)/np.sqrt(1.001)-envelope.CRITERIA['arithmetic_beta_margin']
    require(np.isfinite(floor) and floor > 0 and
            abs(floor-contract['safe_case_floor']) <= 1e-12 and
            abs(cs2-contract['cs_squared_upper']) <= 1e-10*max(1., cs2),
            'changed independent scalar floor')
    return floor


def sparse_input(raw, name, shape):
    require(tuple(raw[name+'_shape']) == shape, name+' dimensions differ')
    data = finite(raw[name+'_data'], name)
    indices, ptr = raw[name+'_indices'], raw[name+'_indptr']
    require(indices.dtype.kind in 'iu' and ptr.dtype.kind in 'iu' and
            len(ptr) == shape[0]+1 and ptr[0] == 0 and ptr[-1] == len(data) == len(indices) and
            np.all(np.diff(ptr) >= 0) and np.all((indices >= 0)&(indices < shape[1])),
            name+' invalid CSR structure')
    matrix = sparse.csr_matrix((data, indices, ptr), shape=shape)
    require(matrix.has_canonical_format, name+' canonical CSR required')
    return matrix


def compile_constraints(case, data, c):
    from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
    from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
    from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator
    cells, ns = case.halfcells, case.counts['shell_nodes']
    xyz = finite(data['xyz_m'], 'coordinates')
    state = Mechanical(xyz[ns:], np.zeros_like(xyz[ns:]), np.tile(np.eye(3), (cells+1, 1, 1)),
                       np.tile(np.eye(3), (cells//2, 2, 1, 1)), np.zeros((cells//2, 18)))
    rr, cc, vv, records = [], [], [], []
    for record in data['station_traces']:
        b = record['beam_port']
        port = NativeHalfCellPort(tuple(b['node_indices']), b['member_index'], b['native_cell'], b['weight'],
                                 b['nodal_count'], b['retained_count'], np.array(b['lift_m']), np.array(b['frame']))
        local = BL5LocalOperator.at_reference(record['shell_reference_m'], np.zeros((2, 6)),
                                             record['thickness_m'], port, state, station=record['shell_alpha'])
        jet = local.evaluate(np.zeros((2, 6)), state, np.zeros(5))
        require(np.max(np.abs(jet.gap)) <= 1e-13, 'nonzero installed reference gap')
        edge = 2*b['member_index']+b['native_cell']
        mean = 2*((edge % case.span)+.5)/case.span-1
        phi = np.zeros((5, 10))
        for channel in range(5):
            phi[channel, 2*channel:2*channel+2] = (1., mean)
        require(np.array_equal(phi, record['density_basis']), 'saved density mean differs')
        rows, slots = np.array(record['multiplier_rows']), np.array(record['full_slots'])
        require(rows.dtype.kind in 'iu' and slots.dtype.kind in 'iu' and
                np.all((rows >= 0)&(rows < c.shape[0])) and np.all((slots >= 0)&(slots < c.shape[1])),
                'invalid physical constraint slots')
        weight = record['measure_m']
        require(np.isfinite(weight) and weight > 0, 'positive station measure required')
        rr.extend(np.repeat(rows, len(slots))); cc.extend(np.tile(slots, len(rows)))
        vv.extend((weight*phi.T@jet.gap_jacobian).ravel())
        require(np.allclose(local.reference_arm, record['reference_arm_m'], rtol=0., atol=1e-13) and
                np.allclose(local.director_basis, record['director_basis'], rtol=0., atol=1e-13),
                'saved installed frame/arm/director basis differs')
        records.append(local)
    require(bool(records), 'complete nonempty station traces required')
    actual = sparse.coo_matrix((vv, (rr, cc)), shape=c.shape).tocsr()
    error = float(sparse.linalg.norm(actual-c))
    require(np.isfinite(error) and error <= 1e-12, 'actual installed projected action differs')
    return records, actual, error


def physical_work(case, c, fixed, frame, director_basis, ab, ass):
    """Independent density virtual work including beam retained director slots."""
    cells, count, cols = case.halfcells, case.counts['shell_nodes'], 10*case.supports
    length, width, ell = map(float, (envelope.LENGTH, envelope.WIDTH, envelope.ELL))
    scale = np.repeat((1., 1., 1., ell, ell), 2*case.supports)
    p = envelope.projection(case)
    densities = np.zeros((cells, 5, cols))
    for channel in range(5):
        indices = slice(2*case.supports*channel, 2*case.supports*(channel+1))
        densities[:, channel, indices] = p*scale[2*case.supports*channel]
    fb = np.einsum('ij,cjk->cik', frame, densities[:, :3])
    director = np.einsum('ij,cjk->cik', frame@director_basis.T, densities[:, 3:])
    n = frame@np.array((0., 0., 1.))
    couple = np.cross(n, director.transpose(0, 2, 1)).transpose(0, 2, 1)
    mb = np.cross(ab, fb.transpose(0, 2, 1)).transpose(0, 2, 1)+couple
    fs = -fb
    ms = np.cross(ass, fs.transpose(0, 2, 1)).transpose(0, 2, 1)-couple
    knots, hb, ba, _ = beam_norm(length, ell, cells, 1, ab, .1)
    xs, _, hs, sa = shell_scalar(length, width, ell, case.nx, case.ny)
    shell = np.zeros((5, count, cols))
    for channel in range(5):
        shell[channel, :case.nx+1] = integrated_p1(xs, fs[:, channel] if channel < 3 else ms[:, channel-3], knots)
    beam_u = np.stack([integrated_p1(knots, fb[:, a], knots) for a in range(3)], axis=1)
    beam = np.concatenate((beam_u.reshape(-1, cols), (mb/cells).reshape(-1, cols)))
    full_work = np.concatenate((shell.reshape(-1, cols), beam))
    physical = case.counts['physical_dofs']
    all_fields = np.r_[np.concatenate([6*np.arange(count)+a for a in range(5)]),
        np.concatenate([6*count+6*i+np.arange(3) for i in range(cells+1)]),
        np.concatenate([physical+24*member+3*half+np.arange(3) for member in range(cells//2) for half in (0, 1)])]
    active = np.r_[np.concatenate([a*count+sa for a in range(5)]), 5*count+ba]
    slots = all_fields[active]
    require(np.array_equal(np.setdiff1d(all_fields, slots), np.intersect1d(fixed, all_fields)) and
            not np.intersect1d(slots, fixed).size, 'supported physical trace mapping differs')
    omitted = c[:, np.setdiff1d(np.arange(c.shape[1]), all_fields)].data
    require(not len(omitted) or np.max(np.abs(omitted)) <= 1e-13, 'nonzero work outside auxiliary physical slots')
    error = float(np.max(np.abs(full_work-(-c[:, all_fields].T@sparse.diags(scale)).toarray())))
    require(error <= 1e-12, 'independent all-column physical work mismatch')
    h = sparse.block_diag((hs, hs, hs, ell**2*hs, ell**2*hs, hb), format='csc')
    return dict(densities=densities, full_work=full_work, action=full_work[active], slots=slots,
                all_fields=all_fields, active=active, scale=scale, fb=fb, mb=mb, fs=fs, ms=ms), h, error


def spectrum(g, m, basis_scale):
    d = np.diag(basis_scale)
    gn, mn = d@g@d, d@m@d
    finite(gn, 'native Gram'); finite(mn, 'reference Gram')
    values, vectors = linalg.eigh(gn, mn)
    residual = float(np.linalg.norm(gn@vectors-(mn@vectors)*values)/np.linalg.norm(gn@vectors))
    return values, vectors, residual, dict(native=float(np.linalg.cond(gn)), reference=float(np.linalg.cond(mn)))


def assess(case, root, registration, wheel, site, output):
    counts = validate_counts(case)
    root, registration, wheel, site = (Path(p).resolve(strict=True) for p in (root, registration, wheel, site))
    output = fresh_output(output, root)
    require(saved.ROOT == SOURCE, 'saved binding helper repository differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Own both outputs exclusively before numerical allocations; retain all failures.
    with output.open('x', encoding='utf-8') as stream:
        stream.write('{}\n')
    with output.with_suffix('.npz').open('xb') as stream:
        np.savez(stream)
    raw, report = {}, dict(status='STARTED', identity=IDENTITY, fixture_case=case.fixture_case,
        fixture=case.fixture, source_sha256=envelope.sha(__file__), formal_f03_authority_consumed=False,
        limitation='Auxiliary native beta only; no mechanical SPD/kernel, uniform stability, physical accuracy or F03 acceptance.')

    def save(stage):
        report['stage'] = stage
        np.savez_compressed(output.with_suffix('.npz'), **raw)
        report['arrays_sha256'] = envelope.sha(output.with_suffix('.npz'))
        output.write_text(json.dumps(json_record(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')

    try:
        require(not any(name.split('.')[0] == 'anyfesolver' for name in sys.modules), 'fresh process without product preimports required')
        pools = threadpool_info()
        require(pools and all(pool['num_threads'] == 1 for pool in pools), 'effective numerical pools must each be one')
        summary, binding = frozen_owner(case, root)
        contract, scalar_binding = owner_helpers.registration_identity(case, registration, SOURCE)
        require(all(scalar_binding[key] == binding['scalar'][key] for key in
                    ('json_sha256', 'arrays_sha256', 'source_sha256', 'helper_sha256')),
                'requested registration differs from frozen owner registration')
        modules = installed_graph(site, wheel, summary)
        report.update(binding=binding, scalar_binding=scalar_binding, runtime=dict(numerical_pools=pools),
                      fixed_criteria=dict(envelope.CRITERIA), helper_sha256={
                          Path(function.__code__.co_filename).name: envelope.sha(function.__code__.co_filename)
                          for function in (beam_norm, beam_bracket, shell_scalar, lower_gram, integrated_p1,
                                           saved.previous_hash, saved.frozen_file, owner_helpers.registration_identity)})
        with np.load(registration.with_suffix('.npz'), allow_pickle=False) as z:
            scalar_raw = {name: z[name].copy() for name in ('projection', 'density_L2_scale', 'scalar_lower', 'reference_lower', 'reference_upper')}
        floor = scalar_floor(contract, scalar_raw, case)
        raw.update({'scalar_'+name: value for name, value in scalar_raw.items()})
        report['safe_case_floor'] = floor
        data = json.loads((root/'native/inputs.json').read_text())
        require(data['layout']['counts'] == {k: counts[k] for k in data['layout']['counts']} and
                (data['layout']['nx'], data['layout']['ny']) == (case.nx, case.ny), 'saved native counts differ')
        validate_policy(case, data)
        xyz = finite(data['xyz_m'], 'coordinates')
        expected_shell = np.array([(i/case.nx, j*float(envelope.WIDTH)/case.ny, 0.) for j in range(case.ny+1) for i in range(case.nx+1)])
        expected_beam = np.c_[np.linspace(0., 1., case.halfcells+1), np.zeros(case.halfcells+1), np.full(case.halfcells+1, .0675)]
        require(xyz.shape == (counts['shell_nodes']+counts['beam_nodes'], 3) and
                np.allclose(xyz, np.r_[expected_shell, expected_beam], rtol=0., atol=1e-13), 'changed reference geometry')
        with np.load(root/'native/operators.npz', allow_pickle=False) as z:
            c = sparse_input(z, 'C', (counts['multipliers'], counts['full_q']))
            k = sparse_input(z, 'K', (counts['full_q'], counts['full_q']))
            fixed = z['fixed'].copy(); yaw = finite(z['beam_rigid_work_map'], 'rigid work map').T[:, 5].copy()
        require(fixed.dtype.kind in 'iu' and len(fixed) == case.fixed_count and len(np.unique(fixed)) == len(fixed) and
                np.all((fixed >= 0)&(fixed < counts['full_q'])) and yaw.shape == (counts['full_q'],), 'fixed/rigid map dimensions differ')
        for name, matrix in (('C', c), ('K', k)):
            raw.update({name+'_data': matrix.data, name+'_indices': matrix.indices, name+'_indptr': matrix.indptr, name+'_shape': np.array(matrix.shape)})
        raw.update(fixed=fixed, bare_yaw=yaw); save('BOUND_INPUTS')
        sys.path.insert(0, str(site))
        records, actual, constraint_error = compile_constraints(case, data, c)
        local = records[0]; frame = local.port.frame; n = frame@local.reference_director
        ab, ass = frame@local.reference_arm, local.thickness*n/2
        require(np.allclose(frame, np.eye(3), rtol=0., atol=1e-13) and
                np.allclose(n, (0, 0, 1), rtol=0., atol=1e-13) and
                np.allclose(ab, (0, 0, -.05), rtol=0., atol=1e-13) and
                np.allclose(ass, (0, 0, .0175), rtol=0., atol=1e-13), 'reference physical frame/arms differ')
        require(all(np.linalg.norm(op.port.lift) <= 1e-13 and
                    np.allclose(op.port.frame, frame, rtol=0., atol=1e-13) and
                    np.allclose(frame@op.reference_arm, ab, rtol=0., atol=1e-13) and
                    np.allclose(frame@op.reference_director, n, rtol=0., atol=1e-13) and op.thickness == local.thickness
                    for op in records), 'nonconstant physical trace/frame/arms')
        for name, module in sys.modules.copy().items():
            if name.split('.')[0] == 'anyfesolver' and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(site) and envelope.sha(path) == modules.get(path.relative_to(site).as_posix()), 'changed installed import: '+name)
                report.setdefault('imported_modules', {})[name] = dict(path=str(path), sha256=envelope.sha(path))
        import numba
        require(bool(numba.config.DISABLE_JIT) and numba.get_num_threads() == 1,
                'effective native JIT disabled and one Numba thread required')
        pools = threadpool_info()
        require(pools and all(pool['num_threads'] == 1 for pool in pools), 'effective installed numerical pools must each be one')
        report['runtime'].update(numerical_pools=pools, numba_threads=numba.get_num_threads(), jit_disabled=bool(numba.config.DISABLE_JIT))
        work, h, work_error = physical_work(case, c, fixed, frame, local.director_basis, ab, ass)
        report['dimensions'] = dict(H_shape=list(h.shape), H_nnz=h.nnz, action_shape=list(work['action'].shape),
                                    constraint_shape=list(c.shape), constraint_nnz=c.nnz, full_work_shape=list(work['full_work'].shape))
        raw.update(work); raw.update(H_data=h.data, H_indices=h.indices, H_indptr=h.indptr, H_shape=np.array(h.shape))
        raw.update(actual_C_data=actual.data, actual_C_indices=actual.indices, actual_C_indptr=actual.indptr)
        save('COMPLETE_PHYSICAL_WORK')
        g, inverse_residual, skew = lower_gram(h, work['action']); raw['native_lower'] = g; save('NATIVE_GRAM')
        ell = float(envelope.ELL); cols = counts['multipliers']
        # Saved projected scalar references share the same scalar p basis.
        shell_lo, shell_hi = np.zeros((cols, cols)), np.zeros((cols, cols))
        p = scalar_raw['projection']
        for channel in range(5):
            field = work['fs'][:, channel] if channel < 3 else work['ms'][:, channel-3]/ell
            coefficients = linalg.lstsq(p, field)[0]
            require(np.max(np.abs(p@coefficients-field)) <= 1e-12, 'shell physical density outside registered projection')
            shell_lo += coefficients.T@scalar_raw['reference_lower']@coefficients
            shell_hi += coefficients.T@scalar_raw['reference_upper']@coefficients
        for subdivision in (1, 2, 4, 8, 16, 32):
            validate_reference_rows(case, subdivision)
            bl, bu, info = beam_bracket(work['fb'], work['mb']/case.halfcells, length=1., ell=ell, arm=ab, width=.1, subdivisions=subdivision)
            lower, upper = shell_lo+bl, shell_hi+bu
            error = float(linalg.eigvalsh(upper-lower, lower)[-1])
            raw.update(reference_lower=lower, reference_upper=upper)
            report.setdefault('reference_preparation', []).append(dict(beam=info, total_relative_reference_error=error))
            save('COUPLED_REFERENCE')
            if error <= envelope.CRITERIA['reference_error']:
                break
        else:
            raise ArithmeticError('preset complete physical reference budget exhausted')
        basis_scale = np.tile(contract['physical_L2_basis_scale'], 5)
        values, vectors, eigen_residual, conditions = spectrum(g, upper, basis_scale)
        coefficients = basis_scale*vectors[:, 0]
        raw.update(eigenvalues=values, eigenvectors=vectors, physical_L2_basis_scale=basis_scale,
                   smallest_coefficients=coefficients, smallest_density=np.einsum('cik,k->ci', work['densities'], coefficients),
                   smallest_full_work=work['full_work']@coefficients, smallest_action=work['action']@coefficients)
        raw['smallest_riesz_lift'] = sparse.linalg.splu(h).solve(raw['smallest_action'])
        raw['smallest_riesz_residual'] = h@raw['smallest_riesz_lift']-raw['smallest_action']
        yaw_norm = float(yaw[work['slots']]@(h@yaw[work['slots']]))
        beta = float(np.sqrt(values[0])) if values[0] > 0 else None
        criteria = envelope.CRITERIA
        predicates = dict(all_column_work=work_error <= 1e-12, installed_constraint_reconstruction=constraint_error <= 1e-12,
            reference_error=error <= criteria['reference_error'],
            inverse_eigen_symmetry=max(inverse_residual, eigen_residual, skew) <= criteria['inverse_eigen_symmetry'],
            physical_L2_condition=max(conditions.values()) <= criteria['condition_budget'],
            full_spectrum_positive=bool(np.all(values > 0)), independent_scalar_floor=beta is not None and beta >= floor,
            variational_ceiling=bool(values[-1] <= criteria['variational_ceiling']),
            bare_yaw_included=bool(yaw_norm > 0 and np.linalg.norm(c@yaw) > 0))
        report.update(beta=beta, conditions=conditions, predicates=predicates,
            numerical=dict(work_error=work_error, constraint_error=constraint_error, inverse_residual=inverse_residual,
                           eigen_residual=eigen_residual, gram_skew=skew, reference_error=error, bare_yaw_norm=yaw_norm),
            status='PASSED_NATIVE_AUXILIARY_BETA_DEVELOPMENT_ONLY' if all(predicates.values()) else 'UNRESOLVED_NATIVE_AUXILIARY_BETA')
        save('COMPLETE_NATIVE_SPECTRUM')
        return report
    except BaseException as error:
        report.update(status='FAILED_NATIVE_AUXILIARY_BETA', failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
        try:
            save('FAILED_PARTIAL_DIAGNOSTICS')
        except BaseException as recording_error:
            error.add_note('Partial diagnostic recording failed: '+repr(recording_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=[c.fixture_case for c in envelope.all_cases()])
    parser.add_argument('--root', type=Path, required=True, help='Saved installed owner directory')
    for name in ('registration', 'wheel', 'site', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = assess(envelope.case_from_id(args.case), args.root, args.registration, args.wheel, args.site, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'fixture_case', 'beta', 'safe_case_floor', 'conditions')}))
