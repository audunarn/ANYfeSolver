"""Reviewed private routing runner; authority precedes scientific imports."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_ge_beam3_g3c_so3_numerics as infrastructure
import ge_beam3_g3c_stable_map as source_map

environment = infrastructure.environment
canonical = infrastructure.canonical
write = infrastructure.write
inputs = infrastructure.inputs
git = infrastructure.git
clean_identity = infrastructure.clean_identity
BASE = '2b6a488fe5807250c4005efbf296e5ebd43a614a'
BASE_TREE = 'f05764d6609cdc454bf2e1d42e3e257fd6060f5c'
MAP_SHA = '30848dea0530f5a49b44cea184f4ff38d73f7e3449131086d9f2c9f312a51ca8'
DESIGN_REVIEW = 'docs/reference_cases/ge_beam3_g3c_stable_source_map_review_v1.json'
DESIGN_SHA = 'd700a1284eb8324442e523d04b21da1c8c24f1c92d19da97d9e37922528a787f'
GUARD = 'tests/test_ge_beam3_g3c_stable_guards.py'
BRIDGE = 'tests/test_ge_beam3_g3c_stable_bridge.py'
INVENTORIES = {
    'guard': [GUARD],
    'diagnostic': ['tests/test_ge_beam3_g3c_stable_diagnostic.py::test_component_directional_diagnostic'],
    'bridge': [BRIDGE],
}
EXPECTED_NODES = {'guard': 10, 'diagnostic': 1, 'bridge': 18}
EXTRA = [
    'src/anyfesolver/_ge_beam3_g3c_stable/__init__.py',
    'src/anyfesolver/_ge_beam3_g3c_stable/authority.py',
    'scripts/run_ge_beam3_g3c_stable.py',
    'tests/test_ge_beam3_g3c_stable_implementation.py', GUARD,
    'docs/GE_BEAM3_G3C_STABLE_IMPLEMENTATION.md',
]


def verify_review(raw, expected_hash, candidate, frozen_inputs):
    if sha256(raw).hexdigest() != expected_hash:
        raise ValueError('implementation review hash mismatch')
    review = environment.strict(raw)
    if (set(review) != {'decision', 'findings', 'reviewer', 'scope', 'subject_commit'} or
            review['decision'] != 'ACCEPTED_G3C_STABLE_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT' or
            review['findings'] or review['reviewer'].get('independent') is not True or
            review['subject_commit'] != candidate['commit'] or
            review['scope'].get('subject_tree') != candidate['tree'] or
            review['scope'].get('source_map_sha256') != MAP_SHA or
            review['scope'].get('inputs_sha256') != sha256(canonical(frozen_inputs)).hexdigest()):
        raise ValueError('independent frozen implementation acceptance required')
    return review


def authority(review_path, review_hash, *, verify_environment=True):
    # Only standard-library modules have been imported at this point.
    candidate = clean_identity()
    if git('rev-parse', BASE+'^{tree}') != BASE_TREE:
        raise ValueError('source-map closeout tree')
    git('merge-base', '--is-ancestor', BASE, 'HEAD')
    raw = environment.regular(ROOT/DESIGN_REVIEW).read_bytes().replace(b'\r\n', b'\n')
    if sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('changed independent source-map acceptance')
    review = environment.strict(raw)
    if review['decision'] != 'ACCEPTED_G3C_PRIVATE_SOURCE_MAP_DESIGN_ONLY' or review['findings']:
        raise ValueError('unaccepted source-map design')
    raw = environment.regular(ROOT/source_map.MAP).read_bytes().replace(b'\r\n', b'\n')
    if sha256(raw).hexdigest() != MAP_SHA:
        raise ValueError('source-map changed')
    mapping = source_map.strict(raw)
    source_map.audit(mapping, implemented=True)
    allowed = {r['destination'] for r in mapping['copies']} | set(EXTRA)
    changed = []
    for line in git('diff', '--name-status', '--no-renames', BASE, 'HEAD').splitlines():
        kind, path = line.split('\t')
        if kind != 'A' or path not in allowed:
            raise ValueError('inherited mechanics or extent changed: '+path)
        changed.append(path)
    if set(changed) != allowed:
        raise ValueError('incomplete private implementation extent')
    frozen = inputs()
    raw_review = environment.regular(review_path).read_bytes()
    verify_review(raw_review, review_hash, candidate, frozen)
    if verify_environment:
        environment.verify(infrastructure.CAPSULE, infrastructure.CAPSULE_SHA)
    return candidate, frozen, raw_review


class Inventory:
    def __init__(self, expected):
        self.expected = expected
        self.count = None
        self.passed = 0
        self.nonpassing = False

    def pytest_collection_modifyitems(self, session, config, items):
        self.count = len(items)
        if self.count != self.expected:
            raise ValueError('registered lane count mismatch')

    def pytest_runtest_logreport(self, report):
        if report.failed or report.skipped or hasattr(report, 'wasxfail'):
            self.nonpassing = True
        if report.when == 'call' and report.passed:
            self.passed += 1


def worker(out, expected):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('isolated worker required')
    raw = environment.regular(out/'lease.json').read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError('lease identity')
    lease = environment.strict(raw)
    print('G3C CHECKPOINT authority', flush=True)
    candidate, frozen, review = authority(out/'review.json', lease['review_sha256'])
    if candidate != lease['candidate'] or frozen != lease['inputs']:
        raise ValueError('leased inputs changed')
    lane = lease['lane']
    if INVENTORIES.get(lane) != lease['inventory']:
        raise ValueError('unregistered lane')
    # Full candidate and environment were verified before any numerical import.
    sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(infrastructure.CAPSULE.parent/'site')]
    from anyfesolver._ge_beam3_g3c_stable import authority as runtime
    runtime.capture(raw)
    import pytest
    inventory = Inventory(EXPECTED_NODES[lane])
    code = pytest.main(['-vv', '-s', '-p', 'no:cacheprovider', '--basetemp',
                       str(out/'pytest'), *INVENTORIES[lane]], plugins=[inventory])
    # Finalization rechecks everything, including the external environment.
    final_candidate, final_inputs, final_review = authority(out/'review.json', lease['review_sha256'])
    if (final_candidate != candidate or final_inputs != frozen or final_review != review or
            inventory.count != EXPECTED_NODES[lane]):
        raise ValueError('final input/inventory mismatch')
    print('G3C CHECKPOINT complete', flush=True)
    return int(code or inventory.nonpassing or inventory.passed != EXPECTED_NODES[lane])


def close_tree(job, process, cleanup_failure):
    """Prove terminal zero before returning; always close the owned handle.

    A drain failure is external process-failure evidence, never a successful
    terminal record. Closing the kill-on-close job remains mandatory even if
    waiting or accounting raises. This does not retry the scientific worker.
    """
    try:
        if job.accounting()[1] and not job.terminate():
            raise RuntimeError('complete process-tree drain was not proven')
        if process is not None:
            process.wait(timeout=15)
        accounting = job.accounting()
        if accounting[1] != 0:
            raise RuntimeError('process tree still active after drain')
        return accounting
    except BaseException as error:
        cleanup_failure(dict(kind='G3C_STABLE_CLEANUP_FAILURE',
                             exception_type=type(error).__name__, message=str(error),
                             terminal_zero_proven=False))
        raise
    finally:
        job.close()


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        return worker(Path(sys.argv[2]), sys.argv[3])
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', choices=INVENTORIES, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    if os.name != 'nt' or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('Windows -I -S -B runner required')
    candidate, frozen, review = authority(args.review, args.review_sha256)
    out = Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-stable-'))
    print('DIAGNOSTICS '+str(out), flush=True)
    with (out/'review.json').open('xb') as stream:
        stream.write(review)
    write(out/'lease.json', dict(kind='G3C_STABLE_PRIVATE_DEVELOPMENT', lane=args.lane,
        inventory=INVENTORIES[args.lane], candidate=candidate, inputs=frozen,
        source_map_sha256=MAP_SHA, review_sha256=args.review_sha256,
        implementation_review=environment.strict(review)))
    lease_hash = sha256((out/'lease.json').read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('g3c_process', ROOT/'docs/reference_cases/e4_pl_s3_v2_bounded_process.py')
    process_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = process_module
    spec.loader.exec_module(process_module)
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'BLIS_NUM_THREADS','NUMBA_NUM_THREADS','TBB_NUM_THREADS'):
        env[key] = '1'
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='', PYTHONDONTWRITEBYTECODE='1')
    job = process_module._ProcessJob(24*1024**3)
    start = time.monotonic(); last = start; previous = None; status = 'FAILED'
    process = None
    try:
        with (out/'stdout.log').open('xb') as stdout, (out/'stderr.log').open('xb') as stderr:
            process = job.launch([sys.executable, '-I', '-S', '-B', '-u', str(Path(__file__).resolve()),
                                 '--worker', str(out), lease_hash], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while True:
                cpu, active, peak = job.accounting(); now = time.monotonic()
                progress = (cpu, (out/'stdout.log').stat().st_size, (out/'stderr.log').stat().st_size)
                if progress != previous:
                    previous = progress; last = now
                if now-start >= 600 or now-last >= 120 or peak > 24*1024**3:
                    status = 'RESOURCE_BLOCKED'; break
                if process.poll() is not None and active == 0:
                    status = 'PASSED' if process.returncode == 0 else 'FAILED'; break
                time.sleep(.1)
    finally:
        final_accounting = close_tree(job, process, lambda row: write(out/'cleanup-failure.json', row))
    record = dict(kind='G3C_STABLE_PRIVATE_DEVELOPMENT', status=status, lane=args.lane,
        elapsed_seconds=time.monotonic()-start, returncode=process.returncode,
        active_processes=final_accounting[1], peak_tree_bytes=final_accounting[2], lease_sha256=lease_hash)
    write(out/'process.json', record)
    print((out/'stdout.log').read_text(errors='replace'))
    print((out/'stderr.log').read_text(errors='replace'))
    print(canonical(record).decode(), flush=True)
    return int(status != 'PASSED')


if __name__ == '__main__':
    raise SystemExit(main())
