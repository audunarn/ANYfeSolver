"""R0-R2 and the timing table from raw/*.json (spec section 7, rev2 section 5.2). Prints JSON to stdout.

Review r3 / OD10 R1 (a), 2026-11-20: the specimen footprint is [0.3, 0.7]^2 with one P2 layer. A coupled run
may be a recorded refusal ({'refused': ...}); every gate that needs it then fails. The diag_* runs (candidate
refined 5 times in the measurement process) are evaluated with the same code and reported as diagnostics only."""
import json, math
from pathlib import Path
raw = Path(__file__).parent / 'raw'
J = lambda n: json.loads((raw / f'{n}.json').read_text())  # noqa: E731
P, L, T, E, NU = 1000.0, 1.0, 0.02, 210e9, 0.3
D = E * T ** 3 / (12 * (1 - NU ** 2))
out = {'scales': {'pL4/D': P * L ** 4 / D, 'pL2': P * L ** 2}}
def ill(ref, scale):
    return abs(ref) < 1e-12 * scale
def refused(c):
    return c.get('refused')
s10, s20, s160 = J('shell_n10'), J('shell_n20'), J('shell_n160')
fine3 = [J(f'shell_n{n}')['observables']['centre']['w'] for n in (40, 80, 160)]
p_fine = math.log2((fine3[1] - fine3[0]) / (fine3[2] - fine3[1]))
richardson = fine3[2] + (fine3[2] - fine3[1]) / (2 ** p_fine - 1)
runs = {(n, l): J(f'solid_n{n}_l{l}') for n in (20, 40) for l in (2, 4)}


def r0(c10, c20):
    rows = []
    for c, s in ((c10, s10), (c20, s20)):
        gated = c['h'] == 0.05
        if refused(c):
            rows.append({'h': c['h'], 'obs': 'all', 'refused': c['refused'], 'gated': gated, 'pass': False})
            continue
        for k, v in c['observables'].items():
            ref = s['observables'][k]['w']
            rel = abs(v['w'] - ref) / abs(ref)
            rows.append({'h': c['h'], 'obs': k, 'coupled': v['w'], 'shell': ref, 'rel': rel, 'gate': 0.005,
                         'gated': gated, 'pass': rel <= 0.005 and not ill(ref, out['scales']['pL4/D'])})
        ref = s['reaction_z']
        rel = abs(c['reaction_z'] - ref) / abs(ref)
        rows.append({'h': c['h'], 'obs': 'reaction_z', 'coupled': c['reaction_z'], 'shell': ref, 'rel': rel,
                     'gate': 1e-9, 'gated': gated, 'pass': rel <= 1e-9})
    return rows


def r1_od10(c10, c20):
    """OD10: coupled against all-shell at the same h. Errors against the finest all-shell run R(h/8)
    (n = 160); observed order p = log2(e(0.1) / e(0.05)) from the centre deflection; gates
    |p_coupled - p_shell| <= 0.2 and e_coupled <= 1.2 e_shell at every compared h and observable."""
    if refused(c10) or refused(c20):
        return {'gate': 'R(h/8)', 'pass': False,
                'status': 'FAIL: the coupled route refused at ' + ', '.join(
                    f"h = {c['h']:g} ({c['refused']['code']} {c['refused']['reason']})" for c in (c10, c20) if refused(c))}
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
                        ec = abs(c['observables'][k]['w'] - s160['observables'][k]['w'])
                        es = abs(s['observables'][k]['w'] - s160['observables'][k]['w'])
                        ratios.append({'h': h, 'obs': k, 'e_coupled': ec, 'e_shell': es, 'ratio': ec / es,
                                       'pass': ec <= 1.2 * es})
        od10[label] = {'reference_centre': refc, 'p_coupled': p_c, 'p_shell': p_s, 'order_difference': abs(p_c - p_s),
                       'order_pass': abs(p_c - p_s) <= 0.2, 'ratios': ratios,
                       'pass': abs(p_c - p_s) <= 0.2 and all(r['pass'] for r in ratios)}
    return {'gate': 'R(h/8)', 'p_fine_40_80_160': p_fine, **od10, 'pass': od10['R(h/8)']['pass']}


def r2(c20, key):
    kap = {k: v[key]['kappa_xx'] for k, v in runs.items()}
    row = {'sample': key, 'runs': {f'h={1/k[0]:g},layers={k[1]}': v for k, v in kap.items()},
           'counts': {f'h={1/k[0]:g},layers={k[1]}': v[key].get('count_end') for k, v in runs.items()}}
    if refused(c20):
        row.update(status=f"FAIL: the coupled route refused at h = 0.05 ({c20['refused']['reason']})", **{'pass': False})
        return row
    row['coupled'] = c20[key]['kappa_xx']
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


def gates(c10, c20):
    rows = r0(c10, c20)
    return {'R0': rows, 'R0_gate_pass': all(r['pass'] for r in rows if r['gated']),
            'R1_od10': r1_od10(c10, c20), 'R2': [r2(c20, 'kappa_protocol'), r2(c20, 'kappa_diagnostic')],
            'R2_gate': 'kappa_diagnostic (radius 0.25a, OD10); kappa_protocol (0.05a) reported only',
            'sizes': {'h=0.1': c10.get('sizes'), 'h=0.05': c20.get('sizes')},
            'certificates': {f"h={c['h']:g}": ({'refused': c['refused'], 'forward_error': c.get('forward_error'),
                                                 'certificate': c.get('certificate')} if refused(c) else
                                                [c['certificate'], c['verification'], c['backward_max_ratio']])
                             for c in (c10, c20)}}


c10, c20 = J('r0_coupled_n10'), J('timing_coupled_t1')
out.update(gates(c10, c20))
d10, d20 = J('diag_coupled_n10_ref5'), J('diag_coupled_n20_ref5')
out['diagnostic_refinement_5'] = {'note': 'NOT A GATE: candidate refined 5 times in the measurement process (finding R1-3)',
                                  **gates(d10, d20)}
out['R2_solid_checks'] = {f'h={1/k[0]:g},layers={k[1]}': {'ndof': v['ndof'], 'reaction_z': v['reaction_z'],
                          'relative_residual': v['relative_residual'], 'w_centre': v['observables'].get('centre', {}).get('w'),
                          'wall': v['wall']} for k, v in runs.items()}
# timing
ow, svd, dflt, pk = J('timing_oneway_t1'), J('timing_svd_t1'), J('timing_coupled_default'), J('peak_coupled_t1')
specimen_total = c20['wall_to_refusal'] if refused(c20) else c20['timings_median']['total']
out['timing'] = {'coupled_median_t1': c20.get('timings_median'), 'coupled_runs_wall_t1': [t['wall'] for t in c20.get('timings_all', [])],
                 'coupled_t1_refused': c20.get('refused'), 'coupled_t1_wall_to_refusal': c20.get('wall_to_refusal'),
                 'diagnostic_ref5_median': d20.get('timings_median'),
                 'coupled_default_threads': dflt.get('timings_all', [None])[0] if not refused(dflt) else dflt,
                 'oneway_median_t1': ow['timings_median'],
                 'svd5000_t1_times': svd['times'], 'svd5000_t1_median': svd['median'],
                 'gate_svd_45s': svd['median'] <= 45.0,
                 'gate_specimen_60s': (not refused(c20)) and specimen_total <= 60.0,
                 'peak_commit_minus_post_import_mib': pk['peak_commit_minus_post_import_mib'], 'peak_run_refused': pk.get('refused'),
                 'environment': ow['environment']}
print(json.dumps(out, indent=1, default=float))
