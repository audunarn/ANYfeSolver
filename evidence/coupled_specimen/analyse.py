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
                   'gated': c['h'] == 0.05, 'pass': rel <= 0.005 and not ill(ref, out['scales']['pL4/D'])})
    ref = s['reaction_z']
    rel = abs(c['reaction_z'] - ref) / abs(ref)
    r0.append({'h': c['h'], 'obs': 'reaction_z', 'coupled': c['reaction_z'], 'shell': ref, 'rel': rel, 'gate': 1e-9,
               'gated': c['h'] == 0.05, 'pass': rel <= 1e-9})
out['R0'] = r0      # OD10: gated at h = 0.05; h = 0.1 is reported only
out['R0_gate_pass'] = all(r['pass'] for r in r0 if r['gated'])
# R1 (OD10, 2026-11-20): coupled against all-shell at the same h. Errors against the finest all-shell
# run R(h/8) (n = 160); observed order p = log2(e(0.1) / e(0.05)) from the centre deflection (the only
# observable that is a node at both h); gates |p_coupled - p_shell| <= 0.2 and e_coupled <= 1.2 e_shell
# at every compared h and observable.
s10, s20, s160 = J('shell_n10'), J('shell_n20'), J('shell_n160')
fine3 = [J(f'shell_n{n}')['observables']['centre']['w'] for n in (40, 80, 160)]
p_fine = math.log2((fine3[1] - fine3[0]) / (fine3[2] - fine3[1]))
richardson = fine3[2] + (fine3[2] - fine3[1]) / (2 ** p_fine - 1)
od10 = {}
for label, refc in (('R(h/8)', s160['observables']['centre']['w']), ('Richardson(p_fine) [sensitivity]', richardson)):
    e_c = {0.1: abs(c10['observables']['centre']['w'] - refc), 0.05: abs(c20['observables']['centre']['w'] - refc)}
    e_s = {0.1: abs(s10['observables']['centre']['w'] - refc), 0.05: abs(s20['observables']['centre']['w'] - refc)}
    p_c, p_s = math.log2(e_c[0.1] / e_c[0.05]), math.log2(e_s[0.1] / e_s[0.05])
    ratios = [{'h': h, 'obs': 'centre', 'e_coupled': e_c[h], 'e_shell': e_s[h], 'ratio': e_c[h] / e_s[h],
               'pass': e_c[h] <= 1.2 * e_s[h]} for h in (0.1, 0.05)]
    if label == 'R(h/8)':
        for c, s, h in ((c10, s10, 0.1), (c20, s20, 0.05)):
            for k in c['observables']:
                if k != 'centre' and k in s160['observables']:
                    ec, es = abs(c['observables'][k]['w'] - s160['observables'][k]['w']), abs(s['observables'][k]['w'] - s160['observables'][k]['w'])
                    ratios.append({'h': h, 'obs': k, 'e_coupled': ec, 'e_shell': es, 'ratio': ec / es, 'pass': ec <= 1.2 * es})
    od10[label] = {'reference_centre': refc, 'p_coupled': p_c, 'p_shell': p_s, 'order_difference': abs(p_c - p_s),
                   'order_pass': abs(p_c - p_s) <= 0.2, 'ratios': ratios,
                   'pass': abs(p_c - p_s) <= 0.2 and all(r['pass'] for r in ratios)}
out['R1_od10'] = {'gate': 'R(h/8)', 'p_fine_40_80_160': p_fine, **od10}
# R1 as first specified (rev2 section 5.2), kept for the record; superseded by OD10.
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
    # OD10: the 0.25a sample set is the gate, limit 2.7 % of |kappa_ref| (U is reported).
    row.update(kappa_ref=ref, U=U, dev=dev, limit=0.027 * abs(ref), rel=dev / abs(ref))
    row['pass'] = dev <= row['limit']
    row['status'] = 'PASS' if row['pass'] else 'FAIL: gate'
    return row
out['R2'] = [r2('kappa_protocol'), r2('kappa_diagnostic')]
out['R2_gate'] = 'kappa_diagnostic (radius 0.25a, OD10); kappa_protocol (0.05a) reported only'
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
