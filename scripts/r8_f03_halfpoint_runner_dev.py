"""Single owned fresh120s development ledger, bounded serial process-tree checks.

Explicit run only; final60s reserve. Existing attempts/failures are immutable.
No formal F03 assessment, old authority reuse or growing-family campaign.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'docs/agent_plans/development/r8-f03-common-density-20261010/halfpoint-v1'
ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
       'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMBA_DISABLE_JIT')}
ENV['PYTHONOPTIMIZE'] = '0'
SOURCES = ('scripts/r8_f03_halfpoint_dev.py', 'scripts/r8_f03_halfpoint_runner_dev.py',
    'scripts/r8_f03_physical_interval_dev.py',
    'tests/test_r8_f03_halfpoint_dev.py', 'scripts/r8_f03_finite_local_dev.py',
    'scripts/r8_f03_finite_inventory_dev.py', 'scripts/r8_f03_finite_sparse_dev.py',
    'src/anyfesolver/e4_pl_element.py', 'src/anyfesolver/_stiffened_shell_ge_experimental.py',
    'src/anyfesolver/_stiffened_shell_q4_reference.py', 'src/anyfesolver/_ge_beam3_retained_generalized.py',
    'src/anyfesolver/_ge_beam3_generalized_cell.py', 'src/anyfesolver/_ge_beam3_generalized_ellipsoid_section.py',
    'src/anyfesolver/_ge_beam3_centered_reference.py', 'src/anyfesolver/_ge_beam3_p5/compensated.py',
    'src/anyfesolver/_ge_beam3_generalized_static_boundary.py', 'src/anyfesolver/_stiffened_shell_l1_assembly_experimental.py',
    'src/anyfesolver/_stiffened_shell_bl5_local_experimental.py', 'src/anyfesolver/_stiffened_shell_l1_local_experimental.py',
    'src/anyfesolver/linalg.py', 'scripts/_b3_e4_f03_local_scientific_supervisor.py',
    'docs/agent_plans/development/r8-f03-common-density-20261010/REFERENCE_MECHANICS_CONTRACT.md',
    'docs/agent_plans/development/r8-f03-common-density-20261010/physical-interval-v1/ROOT_DISPOSITION.md',
    'docs/agent_plans/development/r8-f03-common-density-20261010/physical-interval-v1/INDEPENDENT_REVIEW.md')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_assessment(folder, assessment):
    if assessment is None:
        save(folder/'assessment.json', {'status': 'PIPELINE_SETUP_FAILED', 'qualification': False})
        return
    r=assessment; ref=r['reference']
    def entries(mapping): return [[i,j,str(value)] for (i,j),value in sorted(mapping.items())]
    summary={k:r[k] for k in ('status','family','certificate','factor_diagnostics','target_nonsingularity',
        'symmetric_admissible_inertia','symmetric_admissible_C_rank','literal_numerical_reserve_N_per_m',
        'raw_stored_coercivity','raw_stored_inertia','scientific_floor_comparison','physical_acceptance','qualification','exact_operations',
        'exact_operations_components')}
    summary.update(dimension=ref['dim'],kappa=str(ref['kappa']),Cgeom=str(ref['inventory']['Cgeom']),
        congruence=r['congruence'],source={k:v for k,v in r['stored'].items() if k!='entries'},
        maximum_exact_box_difference=str(r['box']['maximum_exact_difference']),target_provenance=r['targets']['provenance'])
    save(folder/'assessment.json',summary)
    save(folder/'sparse-reference.json',{'shape':[ref['dim']]*2,'ideal':entries(ref['ideal']),'Gz':entries(ref['G']),
        'full_free':ref['full_free'],'physical_free':ref['physical_free']})
    save(folder/'sparse-stored.json',{'shape':[ref['dim']]*2,'entries':entries(r['stored']['entries'])})
    save(folder/'halfpoint-box.json',{'dimension':ref['dim'],'center':entries(r['box']['center_entries']),
        'outward_radius':entries(r['box']['radius_entries']),'target_provenance':r['targets']['provenance']})
    save(folder/'exact-targets.json',{'dimension':ref['dim'],'entries':{name:entries(v) for name,v in r['targets']['entries'].items()},
        'provenance':r['targets']['provenance']})
    factor=r['_factor']
    def factor_entries(matrix):
        matrix=matrix.tocoo()
        return [[int(i),int(j),str(float(v))] for i,j,v in zip(matrix.row,matrix.col,matrix.data)]
    save(folder/'sparse-factors.json',{'shape':[ref['dim']]*2,'L':factor_entries(factor.L),'U':factor_entries(factor.U),
        'perm_r':list(map(int,factor.perm_r)),'perm_c':list(map(int,factor.perm_c))})


def child(folder):
    import psutil
    psutil.Process().cpu_affinity([0])
    import numpy as np
    import numba
    from threadpoolctl import threadpool_info
    import unittest
    import xml.etree.ElementTree as xml
    runtime = {'affinity': psutil.Process().cpu_affinity(), 'environment': {key: os.getenv(key) for key in ENV},
               'optimization': sys.flags.optimize, 'pools': threadpool_info(), 'python': sys.version,
               'numpy': np.__version__, 'numba_disable_jit': bool(numba.config.DISABLE_JIT),
               'numba_threads': numba.get_num_threads()}
    save(folder/'effective-runtime.json', runtime)
    if (runtime['affinity'] != [0] or runtime['environment'] != ENV or runtime['optimization'] != 0 or
            not runtime['numba_disable_jit'] or runtime['numba_threads'] != 1 or
            not runtime['pools'] or any(p['num_threads'] != 1 for p in runtime['pools'])):
        raise RuntimeError('effective CPU0/thread1/JIT-off/opt0 contract required')
    path = ROOT/'tests/test_r8_f03_halfpoint_dev.py'
    spec = importlib.util.spec_from_file_location('halfpoint_controls', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
    class Result(unittest.TextTestResult):
        def startTest(self, test):
            self.started = time.perf_counter(); super().startTest(test)
        def stopTest(self, test):
            records.append({'name': test.id(), 'seconds': time.perf_counter()-self.started})
            super().stopTest(test)
    records = []
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=Result).run(suite)
    runtime['final_pools'] = threadpool_info()
    runtime['final_numba_disable_jit'] = bool(numba.config.DISABLE_JIT)
    runtime['final_numba_threads'] = numba.get_num_threads()
    runtime['final_affinity'] = psutil.Process().cpu_affinity()
    save(folder/'effective-runtime.json', runtime)
    pools_ok = (bool(runtime['final_pools']) and all(p['num_threads'] == 1 for p in runtime['final_pools'])
                and runtime['final_numba_disable_jit'] and runtime['final_numba_threads'] == 1 and runtime['final_affinity']==[0])
    errors = {test.id(): detail for test, detail in (*result.failures, *result.errors)}
    tree = xml.Element('testsuite', tests=str(result.testsRun), failures=str(len(result.failures)), errors=str(len(result.errors)))
    for record in records:
        node = xml.SubElement(tree, 'testcase', name=record['name'], time=str(record['seconds']))
        if record['name'] in errors:
            xml.SubElement(node, 'failure').text = errors[record['name']]
    xml.ElementTree(tree).write(folder/'junit.xml', encoding='utf-8', xml_declaration=True)
    save(folder/'result.json', {'status': 'DEVELOPMENT_CONTROLS_PASSED' if result.wasSuccessful() and pools_ok else 'DEVELOPMENT_CONTROLS_FAILED',
          'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
          'final_runtime_thread_contract_passed': pools_ok,
          'measurements': mod.MEASUREMENTS, 'qualification': False})
    export_assessment(folder, mod.ASSESSMENT)
    return 0 if result.wasSuccessful() and pools_ok else 1


def run(final):
    from scripts._b3_e4_f03_local_scientific_supervisor import supervise, SupervisorLimits
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT/'ledger.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger['status'] != 'ACTIVE': raise RuntimeError('closed fresh ledger cannot be reused')
        if any(a['status'] == 'RUNNING' for a in ledger['attempts']): raise RuntimeError('owned check already running')
    else:
        ledger = {'authority': 'root adopted half-point uncertainty-box EDIT brief; fresh DEVELOPMENT allowance; numerical slot owned now',
                  'allowance_seconds': 120., 'final_reserve_seconds': 60., 'used_seconds': 0.,
                  'status': 'ACTIVE', 'attempts': []}
    remaining = 120.-ledger['used_seconds']
    seconds = min(30., remaining if final else remaining-60.)
    if seconds <= 0: raise RuntimeError('no permitted process budget remains')
    attempt = OUT/f'attempt-{len(ledger["attempts"])+1:02d}'
    attempt.mkdir(exist_ok=False)
    environment = dict(os.environ); environment.update(ENV)
    environment['PYTHONPATH'] = os.pathsep.join((str(ROOT/'src'), str(ROOT), str(ROOT/'tests')))
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--child', str(attempt)]
    from scripts.r8_f03_halfpoint_dev import CAPS
    caps = dict(CAPS)
    caps.update(process_seconds=seconds, process_tree_rss_bytes=2*1024**3,
                numerical_concurrency=1, final_reserve_seconds=60.)
    binding = {'command': command, 'cwd': str(ROOT), 'environment': {k: environment[k] for k in (*ENV, 'PYTHONPATH')},
               'sources': {p: sha(ROOT/p) for p in SOURCES}, 'caps': caps,
               'inputs': {'shell_E_Pa': 210e9, 'nu': '3/10', 'thickness_m': '7/200',
                          'a_m': '1/4', 'b_m': '1/10', 'ell_m': '2/5',
                          'actual_arm_z_m': '-1/20', 'width_m': '1/5', 'shell_top_m': '7/400',
                          'native_section': 'exact rational of original coefficient binary64 literals',
                          'force_equivalent_T': ['1', '1', '1', '2/5', '2/5'],
                          'point_shift': 'exact kappa/2',
                          'targets': ['J(Kideal-exact_half_shift)J', 'J(F(raw_stored_physical)-exact_half_shift)J',
                                      'F(rounded_stored_half)', 'F(rounded_stored_unshifted_congruence)-J(exact_half_shift)J'],
                          'scope': 'actual unchanged compiler sparse66 half-point; explicit4-target enclosure; symmetric signed family and raw nonsym nonsingularity only'},
               'limits': {'process_seconds': seconds, 'process_tree_rss_bytes': 2*1024**3},
               'remaining_before_seconds': remaining, 'qualification': False}
    binding['candidate_input_identity_sha256'] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    save(attempt/'binding.json', binding)
    for relative, expected in binding['sources'].items():
        source = ROOT/relative
        if sha(source) != expected: raise RuntimeError('source changed before launch')
        target = attempt/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    item = {'path': str(attempt), 'status': 'RUNNING'}; ledger['attempts'].append(item); save(ledger_path, ledger)
    with (attempt/'stdout.txt').open('w') as stdout, (attempt/'stderr.txt').open('w') as stderr:
        receipt = supervise(command, cwd=ROOT, environment=environment, stdout=stdout, stderr=stderr,
                            limits=SupervisorLimits(seconds, 2*1024**3, .05))
    record = asdict(receipt); save(attempt/'process-tree-receipt.json', record)
    ledger['used_seconds'] += receipt.elapsed_seconds
    item.update(status=receipt.reason, exit_code=receipt.exit_code, process_wall_seconds=receipt.elapsed_seconds)
    ledger['remaining_seconds'] = 120.-ledger['used_seconds']
    if final:
        ledger['status'] = 'CLOSED'
        ledger['discarded_unused_seconds'] = ledger['remaining_seconds']
    save(ledger_path, ledger)
    print(json.dumps({'attempt': str(attempt), 'reason': receipt.reason, 'exit_code': receipt.exit_code,
                      'wall_seconds': receipt.elapsed_seconds, 'used_seconds': ledger['used_seconds'],
                      'remaining_seconds': ledger['remaining_seconds']}))
    return 0 if receipt.reason == 'completed' and receipt.exit_code == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--child', type=Path)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    if args.child: sys.exit(child(args.child))
    if args.run: sys.exit(run(args.final))
    parser.error('explicit --run development check or owned --child required')
