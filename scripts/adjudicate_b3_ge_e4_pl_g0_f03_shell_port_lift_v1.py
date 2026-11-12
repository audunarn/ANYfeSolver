"""Read-only independent-file adjudication of the F03 shell-port pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-shell-port-lift-v1')
PREVIOUS = Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/f03-local-trace-norm-v1')
FROZEN = '424e9148527e440930dc17662455445f78ccd91f'
PREVIOUS_HASHES = {
    'manifest.json':'b7a93390ecb2fb3e45a29f8f4ebc5a0f495d58297d09f42591a42252d5c8e71b',
    'records.jsonl':'bf69b46e865162882c61ea17f398f30661c4d756baf9e17fc88da32d0c683790',
}
CASES = tuple((topology,n,n) for topology in ('Q4','S3-V2D')
              for n in (4,8,16)) + tuple((topology,8,n)
                                    for topology in ('Q4','S3-V2D') for n in (1,64))
SCRIPTS = Path(__file__).resolve().parent
ADDED_SOURCES = {'probe_b3_ge_e4_pl_g0_f03_shell_port_lift_v1.py',
                 'run_b3_ge_e4_pl_g0_f03_shell_port_lift_v1.py'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (ROOT/'records.jsonl').read_text(
        encoding='utf-8').splitlines()]
    prior = json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    identity = (manifest['revision'] == FROZEN
                and manifest['cases'] == [list(x) for x in CASES]
                and len(rows) == len(CASES) == 10
                and manifest['predecessor_sha256'] == PREVIOUS_HASHES
                and all(sha(PREVIOUS/name) == digest
                        for name,digest in PREVIOUS_HASHES.items())
                and prior['revision'] == '86da5763fbeb849cb060f809c2f1010d6b0a408d'
                and set(manifest['source_sha256']) == set(prior['source_sha256']) | ADDED_SOURCES
                and all(manifest['source_sha256'][name] == sha(SCRIPTS/name)
                        for name in manifest['source_sha256'])
                and all(manifest['source_sha256'].get(name) == digest
                        for name,digest in prior['source_sha256'].items())
                and Path(manifest['anyfesolver_import']).resolve().is_relative_to(
                    (SCRIPTS.parent/'src').resolve()))
    values = {}
    for i,(record,case) in enumerate(zip(rows,CASES)):
        actual = tuple(record['case'][key] for key in ('topology','n_beam','n_shell'))
        identity &= (actual == case and record['status'] == 'EXPLORATORY_COMPLETE'
                     and record['exit_code'] == 0
                     and json.loads((ROOT/f'{i:02d}.stdout.json').read_text(
                         encoding='utf-8')) == record['response']
                     and not (ROOT/f'{i:02d}.stderr.txt').read_text(encoding='utf-8'))
        values[actual] = record['response']
    print('identity_and_completion',identity,'unique_cases',len(values))
    print('raw_sha256',{name:sha(ROOT/name) for name in ('manifest.json','records.jsonl')})
    if not identity or len(values) != 10:
        raise ValueError('raw identity or completion failure')
    wall = max(record['wall_seconds'] for record in rows)
    total = sum(record['wall_seconds'] for record in rows)
    rss = max(record['sampled_peak_child_rss_bytes'] for record in rows)
    resources = (wall <= manifest['case_wall_seconds']
                 and total <= manifest['total_wall_seconds']
                 and rss <= manifest['memory_limit_bytes'])
    print('resource_pass',resources,'max_case_s',wall,'sum_case_s',total,'sampled_rss_b',rss)

    bounds = {
        'root_port_error':max(x['root_port_error'] for x in values.values()),
        'graph_pullback_parity':max(x['graph_pullback_parity'] for x in values.values()),
        'constraint_parity':max(z['constraint_parity'] for x in values.values()
                                for z in x['spaces'].values()),
        'moment_parity':max(z['moment_parity'] for x in values.values()
                            for z in x['spaces'].values()),
        'root_parity':max(z['root_parity'] for x in values.values()
                          for z in x['spaces'].values()),
        'idempotence_error':max(z['idempotence_error'] for x in values.values()
                                for z in x['spaces'].values()),
        'numerical_null_action':max(z['cutoffs']['1e-12']['dropped_subspace_action']
                                    for x in values.values() for z in x['spaces'].values()),
    }
    numerical = (bounds['root_port_error'] <= 1e-12
                 and bounds['graph_pullback_parity'] <= 1e-10
                 and bounds['constraint_parity'] <= 1e-10
                 and max(bounds['moment_parity'],bounds['root_parity'],
                         bounds['idempotence_error']) <= 1e-9
                 and bounds['numerical_null_action'] <= 1e-8
                 and all(x['native_yaw_jumps'] == x['n_beam']-1
                         and set(x['spaces']) == {'k4','local-dual'}
                         for x in values.values()))
    print('numerical_pass',numerical,'maxima',bounds)
    adverse = False
    max_dual_cutoff_change = 0.
    max_amplification_cutoff_change = 0.
    ranks_same = True
    for topology in ('Q4','S3-V2D'):
        for law in ('k4','local-dual'):
            joint = []
            for n in (4,8,16):
                cutoffs = values[(topology,n,n)]['spaces'][law]['cutoffs']
                selected = cutoffs['1e-10']
                joint.append((n,selected['max_reachable_over_port_dual'],
                              selected['max_lift_energy_amplification']))
                for k in ('1e-12','1e-08'):
                    ranks_same &= cutoffs[k]['image_rank'] == selected['image_rank']
                    max_dual_cutoff_change = max(max_dual_cutoff_change,
                        abs(cutoffs[k]['max_reachable_over_port_dual']-
                            selected['max_reachable_over_port_dual']))
                    max_amplification_cutoff_change = max(max_amplification_cutoff_change,
                        abs(cutoffs[k]['max_lift_energy_amplification']-
                            selected['max_lift_energy_amplification']))
            growth = [(joint[i+1][j]/joint[i][j]) for i in (0,1) for j in (1,2)]
            adverse |= any(x > 2 for x in growth)
            fixed = [(n,values[(topology,8,n)]['spaces'][law]['cutoffs']['1e-10'])
                     for n in (1,8,64)]
            print('series',topology,law,'joint',joint,'adjacent_growth',growth,
                  'fixed_shell',[(n,x['max_reachable_over_port_dual'],
                                   x['max_lift_energy_amplification']) for n,x in fixed])
    print('adverse_joint_growth_screen',adverse)
    print('cutoff_rank_same',ranks_same,'max_dual_change',max_dual_cutoff_change,
          'max_amplification_change',max_amplification_cutoff_change)
    print('beam_only_nodal_ramp',[(x['topology'],x['n_beam'],x['n_shell'],
                                    x['beam_only_root_ramp_over_port_energy'])
                                   for x in values.values()])
    if not resources or not numerical or not ranks_same:
        raise ValueError('registered numerical/resource check failed')


if __name__ == '__main__':
    main()
