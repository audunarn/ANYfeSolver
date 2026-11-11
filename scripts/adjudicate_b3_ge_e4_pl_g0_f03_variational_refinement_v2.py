"""Read-only, source-bound adjudication of the registered F03 v2 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-variational-refinement-v2')
FROZEN = '1f123848af8f8c6527d186d4f334040ef33cc2e0'
U = 1e6 / (210e9 * .07)
PARENT = Path(__file__).resolve().parent
SOURCES = ('run_b3_ge_e4_pl_g0_f03_variational_refinement_v2.py',
           'run_b3_ge_e4_pl_g0_f03_variational_line_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_variational_line_v1.py',
           'probe_b3_ge_e4_pl_g0_f03_resolved_continuum.py',
           'probe_b3_ge_e4_pl_g0_f03_resolved_sparse_v5.py',
           'probe_b3_ge_e4_pl_g0_f03_independent_membrane.py')
BASE = ((16,8,8,16),(16,8,16,2),(16,8,16,4),(16,8,16,8),
        (16,8,16,16),(16,8,32,4),(32,16,16,8),(64,32,16,8))
SHELL = ((16,16,16,8),(32,8,16,8),(32,32,16,8),(64,16,16,8))
CASES = tuple((topology,ns,ny,nm,subcells,load,law)
              for grids in (BASE,SHELL)
              for topology in ('Q4','S3-V2D')
              for load in ('smooth','sine')
              for law in ('weak-p0','weak-p1')
              for ns,ny,nm,subcells in grids)
V1_HASHES = {
    'manifest.json':'3895a90ec6e94dc92b9c3738a8fc880fca20f56c51dd182b58664446660e0483',
    'preflight.json':'8a6dc936d517a2fb060208d086d51a36c71d3efb93fc6989e2075491e8904d78',
    'records.jsonl':'e4925757567209706512f0aeec5ce234871eafdd1b669822b4b049bdf6511204',
}
V1_ROOT = Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-variational-line-v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change(a, b):
    return abs(a - b) / max(abs(a), abs(b), .01 * U)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    preflight = json.loads((ROOT / 'preflight.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (ROOT / 'records.jsonl').read_text(
        encoding='utf-8').splitlines()]
    identity = (manifest['revision'] == FROZEN and len(rows) == len(CASES) == 96
                and sha(ROOT / 'preflight.json') == V1_HASHES['preflight.json']
                and manifest['cases'] == [list(case) for case in CASES]
                and all(manifest['source_sha256'][name] == sha(PARENT / name)
                        for name in SOURCES)
                and all(manifest['predecessor_v1_sha256'][name] == digest
                        and sha(V1_ROOT / name) == digest
                        for name, digest in V1_HASHES.items()))
    print('frozen_identity', identity)
    print('raw_sha256', {name: sha(ROOT / name) for name in
                         ('manifest.json', 'preflight.json', 'records.jsonl')})
    preflight_max = max(
        max(p['shell_quadrature_relative'], p['beam_quadrature_relative'])
        for p in preflight)
    print('preflight_max_quadrature_relative', preflight_max)
    by_key = {}
    for i, (row, case) in enumerate(zip(rows, CASES)):
        c = row['case']
        actual = tuple(c[k] for k in ('topology', 'n_shell', 'n_y', 'n_mortar',
                                     'subcells', 'load', 'law'))
        identity &= (actual == case and row['status'] == 'EXPLORATORY_COMPLETE'
                     and row['exit_code'] == 0
                     and json.loads((ROOT / f'{i:02d}.stdout.json').read_text(
                         encoding='utf-8')) == row['response']
                     and not (ROOT / f'{i:02d}.stderr.txt').read_text(encoding='utf-8'))
        if row['status'] == 'EXPLORATORY_COMPLETE':
            by_key[actual] = row['response']
    print('complete_ordered_stdout_identity', identity, 'unique', len(by_key))
    if not identity or len(by_key) != 96:
        raise ValueError('raw evidence identity/completion failed')

    fields = ('free_equilibrium_relative', 'force_balance_relative',
              'force_x_balance_relative', 'moment_balance_relative',
              'work_balance_relative', 'normalized_weak_gap_max',
              'load_resultant_force_error_n', 'load_resultant_moment_error_nm',
              'interface_work_discrepancy_j')
    maxima = {key: max(abs(r[key]) for r in by_key.values()) for key in fields}
    gram_min = min(r['row_normalized_gram_ratio'] for r in by_key.values())
    numerical = (len(preflight) == 4 and preflight_max <= 1e-11
                 and gram_min > 1e-12
                 and all(maxima[k] <= 1e-8 for k in fields[:5])
                 and maxima['normalized_weak_gap_max'] <= 1e-9
                 and all(maxima[k] <= 1e-6 for k in fields[6:8])
                 and maxima['interface_work_discrepancy_j'] <= 1e-8)
    print('numerical_pass', numerical, 'gram_min', gram_min, 'maxima', maxima)
    wall_max = max(r['wall_seconds'] for r in rows)
    rss_max = max(r['sampled_peak_child_rss_bytes'] for r in rows)
    wall_sum = sum(r['wall_seconds'] for r in rows)
    resource = (wall_max <= manifest['case_wall_seconds']
                and wall_sum <= manifest['total_wall_seconds']
                and rss_max <= manifest['memory_limit_bytes'])
    print('resource_max_wall_seconds', wall_max,
          'sum_wall_seconds', wall_sum,
          'max_sampled_rss_bytes', rss_max,
          'resource_pass', resource)

    def tip(topology, load, law, ns, ny, nm, subcells):
        return by_key[(topology, ns, ny, nm, subcells, load, law)][
            'station_probes'][-1]['beam'][1]

    checks = []
    for topology in ('Q4', 'S3-V2D'):
        for load in ('smooth', 'sine'):
            row = {'topology': topology, 'load': load, 'laws': {}}
            p0 = tip(topology, load, 'weak-p0', 16, 8, 32, 4)
            p1 = tip(topology, load, 'weak-p1', 16, 8, 32, 4)
            row['p0_p1_tip_um'] = [p0 * 1e6, p1 * 1e6]
            row['p0_p1_finest_mortar_relative'] = change(p0, p1)
            checks.append(row['p0_p1_finest_mortar_relative'])
            for law in ('weak-p0', 'weak-p1'):
                t = lambda ns, ny, nm, sub: tip(topology, load, law, ns, ny, nm, sub)
                edges = []
                for ns, ny in ((16, 8), (32, 16)):
                    for a, b in (((ns, ny), (2*ns, ny)),
                                 ((ns, ny), (ns, 2*ny)),
                                 ((2*ns, ny), (2*ns, 2*ny)),
                                 ((ns, 2*ny), (2*ns, 2*ny))):
                        edges.append(change(t(*a, 16, 8), t(*b, 16, 8)))
                measures = {
                    'mortar_16_32': change(t(16, 8, 16, 8), t(16, 8, 32, 4)),
                    'beam_128_256': change(t(16, 8, 16, 8), t(16, 8, 16, 16)),
                    'shell_32_16_to_64_32': change(t(32, 16, 16, 8),
                                                   t(64, 32, 16, 8)),
                    'crossed_shell_edges': edges,
                    'max_sampled_scaled_gap_over_u': max(
                        by_key[key]['sampled_scaled_gap_lower_bound_over_u']
                        for key in by_key if key[0] == topology and key[5:] == (load, law)),
                }
                row['laws'][law] = measures
                checks.extend((measures['mortar_16_32'], measures['beam_128_256'],
                               measures['shell_32_16_to_64_32'], *edges))
                checks.append(measures['max_sampled_scaled_gap_over_u'] / 5)
            print('physical', json.dumps(row, sort_keys=True))
    print('provisional_classical_warning_screen_pass', all(c <= .01 for c in checks))
    if not numerical:
        raise ValueError('registered numerical check failed')
    if not resource:
        raise ValueError('registered resource check failed')


if __name__ == '__main__':
    main()
