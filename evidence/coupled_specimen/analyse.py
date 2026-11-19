"""R0-R2 and the timing table from raw/*.json (spec section 7, rev2 section 5.2). Prints JSON to stdout."""
import json, math
from pathlib import Path
raw = Path(__file__).parent / 'raw'
J = lambda n: json.loads((raw / f'{n}.json').read_text())  # noqa: E731
P, L, T, E, NU = 1000.0, 1.0, 0.02, 210e9, 0.3
D = E * T ** 3 / (12 * (1 - NU ** 2))
out = {'scales': {'pL4/D': P * L ** 4 / D, 'pL2': P * L ** 2}}
def ill(ref, scale):
    return abs(ref) < 1e-12 * scale
# R0
c10, c20 = J('r0_coupled_n10'), J('timing_coupled_t1')
r0 = []
for c, s in ((c10, J('shell_n10')), (c20, J('shell_n20'))):
    for k, v in c['observables'].items():
        ref = s['observables'][k]['w']
        rel = abs(v['w'] - ref) / abs(ref)
        r0.append({'h': c['h'], 'obs': k, 'coupled': v['w'], 'shell': ref, 'rel': rel, 'gate': 0.005,
                   'pass': rel <= 0.005 and not ill(ref, out['scales']['pL4/D'])})
    ref = s['reaction_z']
    rel = abs(c['reaction_z'] - ref) / abs(ref)
    r0.append({'h': c['h'], 'obs': 'reaction_z', 'coupled': c['reaction_z'], 'shell': ref, 'rel': rel, 'gate': 1e-9,
               'pass': rel <= 1e-9})
out['R0'] = r0
# R1
r1 = []
fine = [J(f'shell_n{n}') for n in (40, 80, 160)]
for k in ('centre', 'quarter_x0.25'):
    R2_, R4, R8 = (f['observables'][k]['w'] for f in fine)
    d1, d2 = R4 - R2_, R8 - R4
    row = {'obs': k, 'R_h/2': R2_, 'R_h/4': R4, 'R_h/8': R8, 'd1': d1, 'd2': d2, 'x': c20['observables'][k]['w']}
    if d1 == 0 or d2 == 0 or (d1 > 0) != (d2 > 0):
        row.update(status='FAIL: no asymptotic range (d1, d2 zero or of opposite sign)', **{'pass': False})
    else:
        p = math.log2(d1 / d2)
        row['p'] = p
        if not 1.5 <= p <= 2.5:
            row.update(status='FAIL: order outside [1.5, 2.5]', **{'pass': False})
        else:
            ext = R8 + d2 / (2 ** p - 1)
            U = abs(R8 - ext)
            dev = abs(row['x'] - ext)
            row.update(R_ext=ext, U=U, dev=dev, limit=0.01 * abs(ext) + U, rel=dev / abs(ext))
            row['pass'] = dev <= row['limit']
            row['status'] = 'PASS' if row['pass'] else 'FAIL: gate'
    r1.append(row)
out['R1'] = r1
# R2
runs = {(n, l): J(f'solid_n{n}_l{l}') for n in (20, 40) for l in (2, 4)}
def r2(key):
    kap = {k: v[key]['kappa_xx'] for k, v in runs.items()}
    row = {'sample': key, 'runs': {f'h={1/k[0]:g},layers={k[1]}': v for k, v in kap.items()},
           'coupled': c20[key]['kappa_xx'],
           'counts': {f'h={1/k[0]:g},layers={k[1]}': v[key].get('count_end') for k, v in runs.items()}}
    if any(v is None for v in kap.values()) or row['coupled'] is None:
        row.update(status='FAIL (protocol): fewer than 2 skin triangles in a sample set', **{'pass': False})
        return row
    ref = kap[(40, 4)]
    U = max(abs(ref - kap[(20, 4)]), abs(ref - kap[(40, 2)]))
    dev = abs(row['coupled'] - ref)
    row.update(kappa_ref=ref, U=U, dev=dev, limit=0.02 * abs(ref) + U, rel=dev / abs(ref))
    row['pass'] = dev <= row['limit']
    row['status'] = 'PASS' if row['pass'] else 'FAIL: gate'
    return row
out['R2'] = [r2('kappa_protocol'), r2('kappa_diagnostic')]
out['R2_solid_checks'] = {f'h={1/k[0]:g},layers={k[1]}': {'ndof': v['ndof'], 'reaction_z': v['reaction_z'],
                          'relative_residual': v['relative_residual'], 'w_centre': v['observables'].get('centre', {}).get('w'),
                          'wall': v['wall']} for k, v in runs.items()}
# timing
ow, svd, dflt, pk = J('timing_oneway_t1'), J('timing_svd_t1'), J('timing_coupled_default'), J('peak_coupled_t1')
out['timing'] = {'coupled_median_t1': c20['timings_median'], 'coupled_runs_wall_t1': [t['wall'] for t in c20['timings_all']],
                 'coupled_default_threads': dflt['timings_all'][0], 'oneway_median_t1': ow['timings_median'],
                 'ratio_coupled_over_native_plus_oneway': c20['timings_median']['total'] / ow['timings_median']['total'],
                 'svd5000_t1_times': svd['times'], 'svd5000_t1_median': svd['median'],
                 'gate_svd_45s': svd['median'] <= 45.0, 'gate_specimen_60s': c20['timings_median']['total'] <= 60.0,
                 'peak_commit_minus_post_import_mib': pk['peak_commit_minus_post_import_mib'],
                 'environment': c20['environment']}
out['sizes'] = {'h=0.1': c10['sizes'], 'h=0.05': c20['sizes']}
out['certificates'] = {'h=0.1': [c10['certificate'], c10['verification'], c10['backward_max_ratio']],
                       'h=0.05': [c20['certificate'], c20['verification'], c20['backward_max_ratio']]}
print(json.dumps(out, indent=1, default=float))
