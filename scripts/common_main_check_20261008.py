"""Bounded, serial integration checks; not a scientific qualification runner."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / '.development/common-main-20261008'
SITE = RAW / 'site-v5'
DEPS = Path('C:/Github/ANYfeSolver/.perf2-worktrees/stiffened-main-reconcile-20261001/.qualification/b3-ge-e4-pl/r8-native-s3-restart-binding-dev-20261005-v1/ci-site-network-v2')
BASE = Path('C:/Python/Python314/Lib/site-packages')
SIBLINGS = [Path('C:/Github') / n / 'src' for n in ('ANYmaterial', 'ANYfileIO', 'ANYmesh', 'ANYgeometry')]
ENV = dict(NUMBA_DISABLE_JIT='1', NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PIP_NO_INDEX='1')

def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def setup(mode):
    os.environ.update(ENV)
    sys.path[:0] = [str(DEPS), str(BASE), *map(str, SIBLINGS)]
    import psutil
    psutil.Process().cpu_affinity([0])
    package_root = REPO / 'src' if mode in ('source', 'policy') else SITE
    sys.path.insert(0, str(package_root))
    sys.path.append(str(REPO))  # scripts/examples, never src in installed mode
    os.environ['PYTHONPATH'] = os.pathsep.join(map(str, [package_root, DEPS, BASE, *SIBLINGS, REPO]))
    os.environ['ANYSOLVER_TEST_PACKAGE_ROOT'] = str(package_root)

def child(args):
    setup(args.mode)
    out = RAW / args.phase
    if args.mode == 'policy':
        from scripts.check_license_compliance import validate_repository
        from scripts.run_portable_ci import merge_test_modules, _ge_beam3_inventory
        validate_repository(REPO)
        ge = _ge_beam3_inventory()
        modules = merge_test_modules()
        write(out / 'policy.json', dict(license='passed', portable_module_count=len(modules), ge_portable=len(ge['portable']), ge_extended=len(ge['extended'])))
        return 0
    if args.mode == 'build':
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(map(str, [DEPS, BASE, *SIBLINGS]))
        subprocess.run([sys.executable, '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(out / 'wheel'), str(REPO)], env=env, check=True)
        wheels = list((out / 'wheel').glob('*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Expected exactly one wheel')
        if SITE.exists():
            raise RuntimeError('Fresh artifact requires a new install directory')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(SITE), str(wheels[0])], env=env, check=True)
        write(out / 'artifact.json', {'wheel': str(wheels[0]), 'sha256': sha(wheels[0]), 'site': str(SITE)})
        return 0
    import anyfesolver, psutil, numpy, scipy, numba
    from threadpoolctl import threadpool_info
    imported = Path(anyfesolver.__file__).resolve()
    expected = (REPO / 'src' if args.mode == 'source' else SITE).resolve()
    if not imported.is_relative_to(expected):
        raise RuntimeError(f'Wrong ANYfeSolver import: {imported}')
    if numba.get_num_threads() != 1 or not numba.config.DISABLE_JIT:
        raise RuntimeError('Effective Numba settings mismatch')
    runtime = dict(anyfesolver=str(imported), numpy=numpy.__version__, scipy=scipy.__version__, numba=numba.__version__, affinity=psutil.Process().cpu_affinity(), jit_disabled=bool(numba.config.DISABLE_JIT), numba_threads=numba.get_num_threads(), pools=threadpool_info(), environment=ENV, mode=args.mode)
    write(out / 'runtime-before.json', runtime)
    if runtime['affinity'] != [0] or any(p['num_threads'] != 1 for p in runtime['pools']):
        raise RuntimeError('Effective CPU/thread settings mismatch')
    if args.mode == 'example':
        import runpy
        sys.path.insert(0, str((REPO / args.targets[0]).parent))
        runpy.run_path(str(REPO / args.targets[0]), run_name='__main__')
        pools = threadpool_info()
        if any(p['num_threads'] != 1 for p in pools):
            raise RuntimeError('Example changed thread settings')
        write(out / 'runtime-after.json', dict(imports={n: str(m.__file__) for n, m in list(sys.modules.items()) if n.startswith('anyfesolver') and getattr(m, '__file__', None)}, pools=pools))
        return 0
    import pytest
    # Existing runtime plugin also validates imports after callbacks and tests.
    os.environ['ANYSOLVER_DEVELOPMENT_RECORD'] = str(out / 'runtime-after.json')
    os.environ['ANYSOLVER_DEVELOPMENT_SOURCE'] = str(expected / 'anyfesolver')
    return int(pytest.main(['-q', '-x', '-o', 'pythonpath=', '-p', 'no:cacheprovider', '-p', 'scripts.development_test_runtime', '--junitxml=' + str(out / 'junit.xml'), '--basetemp=' + str(out / 'temp'), *args.targets]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase')
    parser.add_argument('--mode', choices=('source', 'installed', 'build', 'example', 'policy'), default='source')
    parser.add_argument('--cap', type=float, default=120)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('targets', nargs='*')
    args = parser.parse_intermixed_args()
    if args.child:
        return child(args)
    import psutil
    RAW.mkdir(parents=True, exist_ok=True)
    ledger_path = RAW / 'ledger-v2.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(scope='Fresh human-approved 600-second common-main CI-repair allowance 2026-10-08; old ledger preserved, no authority transfer', budget_seconds=600.0, reserve_seconds=60.0, used_seconds=0.0, attempts=[], in_flight=None)
    if ledger['in_flight'] is not None:
        raise RuntimeError('Prior execution unresolved')
    cap = min(args.cap, 600.0, ledger['budget_seconds'] - ledger['used_seconds'] - ledger['reserve_seconds'])
    if cap <= 0:
        raise RuntimeError('Verification reserve reached')
    out = RAW / args.phase
    out.mkdir(exist_ok=False)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    paths = sorted((REPO / 'src/anyfesolver').rglob('*.py')) + [REPO / 'pyproject.toml', Path(__file__).resolve()]
    paths += [REPO / p for p in ('dependency-licenses.json', 'THIRD_PARTY_NOTICES.md', 'scripts/ge_beam3_release_test_inventory.json', 'scripts/run_portable_ci.py', 'scripts/check_license_compliance.py', 'scripts/development_test_runtime.py')]
    paths += [REPO / t.partition('::')[0] for t in args.targets]
    identities = {str(p.relative_to(REPO)): sha(p) for p in paths}
    command = [sys.executable, '-I', str(Path(__file__).resolve()), args.phase, '--child', '--mode', args.mode, *args.targets]
    write(out / 'inputs.json', dict(candidate=head, file_sha256=identities, command=command, cap_seconds=cap, memory_limit_bytes=2 * 1024**3, cpu_affinity=[0], environment=ENV))
    ledger['in_flight'] = args.phase
    write(ledger_path, ledger)
    peak = 0
    reason = None
    started = time.perf_counter_ns()
    with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
        proc = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr, env={**os.environ, **ENV})
        root = psutil.Process(proc.pid)
        root.cpu_affinity([0])
        while proc.poll() is None:
            try:
                tree = [root, *root.children(recursive=True)]
                peak = max(peak, sum(p.memory_info().rss for p in tree if p.is_running()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if (time.perf_counter_ns() - started) / 1e9 >= cap:
                reason = 'timeout'
            if peak > 2 * 1024**3:
                reason = 'memory_limit'
            if reason:
                for p in reversed([root, *root.children(recursive=True)]):
                    try: p.kill()
                    except psutil.NoSuchProcess: pass
                break
            time.sleep(.05)
        proc.wait()
    elapsed = (time.perf_counter_ns() - started) / 1e9
    receipt = dict(candidate=head, phase=args.phase, mode=args.mode, status=reason or ('passed' if proc.returncode == 0 else 'failed'), exit_code=proc.returncode, process_wall_seconds=elapsed, peak_process_tree_rss_bytes=peak, inputs_changed=[p for p, digest in identities.items() if sha(REPO / p) != digest])
    if (out / 'junit.xml').exists():
        import xml.etree.ElementTree as ET
        suites = ET.parse(out / 'junit.xml').getroot().findall('testsuite')
        receipt['counts'] = {k: sum(int(s.attrib.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if receipt['inputs_changed']:
        receipt['status'] = 'inputs_changed'
    ledger['used_seconds'] += elapsed
    ledger['remaining_seconds'] = ledger['budget_seconds'] - ledger['used_seconds']
    ledger['in_flight'] = None
    ledger['attempts'].append(receipt)
    write(out / 'execution.json', receipt)
    write(ledger_path, ledger)
    print(json.dumps(receipt))
    print('Remaining process-wall seconds:', ledger['remaining_seconds'])
    return 0 if receipt['status'] == 'passed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
