"""Bounded current-owner c2/P0 physical-load bridge; development evidence only.

The refinement-1 comparison uses the same discrete auxiliary H on both sides.
It independently checks physical density load integration and scattering, not
a continuous M_ref bound. No signed-KKT, beta, transfer, or F03 acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numba
import numpy as np
import psutil
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info

# The shared virtual environment points at the main checkout. Pin product code
# to this assigned worktree before importing anyfesolver or the frozen builder.
WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / 'src'))
sys.path.insert(0, str(WORKTREE / 'tests'))
import anyfesolver
from anyfesolver import stiffened_shell
from scripts.b3_e4_bl5_coherent_aggregation_dev import model
from scripts.r8_bl5_actual_mref_dev import flat_p0_reference_dual
from scripts.r8_bl5_actual_primal_work_dev import actual_flat_primal_work


ROOT = Path('.development/f03-c2-current-owner-bridge-20261009-v1')
LIMIT = 180.0
RESERVE = 60.0
MEMORY = 2 * (1 << 30)
GRIDS = {32: (32, 8, 64), 64: (64, 16, 128)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case(level):
    if level not in GRIDS:
        raise ValueError('only the declared 32x8 or 64x16 c2 grids are admitted')
    nx, ny, cells = GRIDS[level]
    definition, policy = model(nx, ny, cells, 2)
    if (policy['span_halfcells'] != 4 or
            policy['shell_trace_segments'] != nx or
            policy['beam_halfcells'] != cells):
        raise ValueError('frozen c2 mapping changed')
    return definition, policy


def effective_runtime():
    pools = threadpool_info()
    runtime = dict(cpu_affinity=psutil.Process().cpu_affinity(),
                   numba_threads=numba.get_num_threads(),
                   jit_disabled=bool(numba.config.DISABLE_JIT), pools=pools)
    if (runtime['cpu_affinity'] != [0] or runtime['numba_threads'] != 1 or
            not runtime['jit_disabled'] or not pools or
            any(pool['num_threads'] != 1 for pool in pools)):
        raise RuntimeError('effective CPU0, one-thread, JIT-off runtime required')
    return runtime


def product_provenance(owner=None):
    modules = dict(anyfesolver=anyfesolver.__file__,
                   stiffened_shell=stiffened_shell.__file__)
    if owner is not None:
        modules['owner'] = sys.modules[type(owner).__module__].__file__
    src = (WORKTREE / 'src').resolve()
    for name, path in modules.items():
        if not Path(path).resolve().is_relative_to(src):
            raise RuntimeError(f'{name} import escaped assigned worktree: {path}')
    scripts = dict(model=model.__module__,
                   actual_primal=actual_flat_primal_work.__module__,
                   discrete_reference=flat_p0_reference_dual.__module__,
                   shell_norm='scripts.r8_bl5_mixed_shell_h1_dev',
                   beam_and_scalar_norm='scripts.b3_e4_bl5_riesz_controls_dev',
                   physical_loads='scripts.b3_e4_bl5_reference_actions_dev',
                   shell_modal_reference='scripts.b3_e4_bl5_modal_reference_dev',
                   beam_reference='scripts.b3_e4_bl5_reference_accuracy_dev')
    expected_scripts = (WORKTREE / 'scripts').resolve()
    for name, module_name in scripts.items():
        path = sys.modules[module_name].__file__
        if not Path(path).resolve().is_relative_to(expected_scripts):
            raise RuntimeError(f'{name} import escaped assigned worktree: {path}')
        modules[name] = path
    fixture = sys.modules['test_stiffened_shell_l1_assembly_experimental'].__file__
    if not Path(fixture).resolve().is_relative_to((WORKTREE / 'tests').resolve()):
        raise RuntimeError(f'frozen model fixture escaped assigned worktree: {fixture}')
    modules['frozen_model_fixture'] = fixture
    return {name: dict(path=str(Path(path).resolve()), sha256=sha(path))
            for name, path in modules.items()}


def evaluate(level, output):
    runtime = effective_runtime()
    provenance = product_provenance()
    definition, policy = case(level)
    owner = stiffened_shell.create_analysis(definition=definition)
    try:
        provenance = product_provenance(owner)
        h, embedding, work = actual_flat_primal_work(owner, .4, .1)
        rows = 5 * (GRIDS[level][2] // 4)
        if (owner._C.shape[0] != rows or work.shape != (rows, h.shape[0]) or
                embedding.shape != (owner._K.shape[0], h.shape[0]) or
                embedding.nnz != h.shape[0] or embedding[owner._fixed].nnz):
            raise ArithmeticError('actual supported owner embedding inventory changed')
        difference = (owner._C @ embedding - work).tocsr()
        difference.eliminate_zeros()
        row_error = float(np.max(np.abs(difference.data), initial=0.))
        symmetry = float(sparse.linalg.norm(h-h.T) / max(sparse.linalg.norm(h), 1e-30))
        if row_error > 2e-13 or symmetry > 1e-13:
            raise ArithmeticError('supported owner work or norm symmetry failed')
        qscale = np.repeat((1., 1., 1., .4, .4), rows // 5)
        action = (-work.T @ sparse.diags(qscale)).toarray()
        factor = splu(h)
        representers = factor.solve(action)
        inverse_residual = float(np.linalg.norm(h @ representers - action) /
                                 max(np.linalg.norm(action), 1e-30))
        gram = action.T @ representers
        gram = (gram + gram.T) / 2
        pairing_error, records = flat_p0_reference_dual(
            owner, .4, .1, refinements=(1,))
        reference = records[0]['matrix']
        relative_action = float(np.linalg.norm(gram-reference) /
                                max(np.linalg.norm(reference), 1e-30))
        max_action = float(np.max(np.abs(gram-reference)))
        eigen = linalg.eigvalsh(gram)
        if (inverse_residual > 1e-10 or relative_action > 1e-10 or
                pairing_error > 2e-12 or eigen[0] <= 0.):
            raise ArithmeticError('actual and independent physical actions differ')
        np.savez_compressed(output / 'raw_actions.npz',
                            actual=gram, physical_rhs_reference_same_h=reference,
                            qscale=qscale, physical_rhs=action)
        return dict(status='PASS_DEVELOPMENT_ONLY',
                    scope='current c2/P0 owner at one grid: supported E and H, and discrete Gram parity from independently integrated physical density RHS/scatter using the same H; no continuous M_ref bound, beta, signed-KKT, physical transfer, full F03, or acceptance',
                    grid=dict(shell_edges=GRIDS[level][0], transverse_edges=GRIDS[level][1],
                              native_halfcells=GRIDS[level][2], multiplier_rows=rows),
                    policy=policy, runtime=runtime, product_provenance=provenance,
                    dimensions=dict(owner_q=owner._K.shape[0], supported_h=h.shape[0],
                                    owner_c=rows, embedding_nnz=embedding.nnz,
                                    work_nnz=work.nnz, h_nnz=h.nnz),
                    construction_identity=dict(c_e_equals_returned_work_max_abs=row_error),
                    checks=dict(h_relative_skew=symmetry,
                                inverse_relative_residual=inverse_residual,
                                physical_pairing_max_abs=pairing_error,
                                same_h_physical_rhs_gram_relative_frobenius=relative_action,
                                same_h_physical_rhs_gram_max_abs=max_action,
                                actual_action_minimum_eigenvalue=float(eigen[0]),
                                reference_discrete_inverse_relative_residual=records[0]['inverse_residual'],
                                reference_discrete_action_minimum_eigenvalue=records[0]['minimum_eigenvalue']),
                    raw_actions='raw_actions.npz')
    finally:
        owner.close()


def prior_wall(root):
    total = 0.0
    for receipt in root.glob('*/receipt.json'):
        elapsed = json.loads(receipt.read_text())['process_wall_seconds']
        if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid prior process-wall receipt')
        total += elapsed
    return total


def source_fingerprint():
    """Hash every source and imported helper used by this bounded check."""
    names = (
        'scripts/r8_f03_c2_current_owner_bridge_dev.py',
        'scripts/r8_bl5_actual_primal_work_dev.py',
        'scripts/r8_bl5_actual_mref_dev.py',
        'scripts/b3_e4_bl5_coherent_aggregation_dev.py',
        'tests/test_r8_f03_c2_current_owner_bridge_dev.py',
    )
    hashes = {name: sha(WORKTREE / name) for name in names}
    hashes['product_provenance'] = product_provenance()
    return hashes


def guarded_run(kind, root=ROOT):
    if kind not in ('source', '32', '64'):
        raise ValueError('unregistered check')
    root = Path(root)
    if kind == 'source':
        index = 1
        while (root / f'source-check-{index}').exists():
            index += 1
        path = root / f'source-check-{index}'
    else:
        stem = {'32': 'grid-32x8', '64': 'grid-64x16'}[kind]
        path = root / stem
        index = 2
        while path.exists():
            path = root / f'{stem}-{index}'
            index += 1
    if path.exists():
        raise FileExistsError(path)
    timeout = 30.0 if kind == 'source' else 45.0
    used = prior_wall(root)
    if used + timeout + RESERVE > LIMIT:
        raise RuntimeError('insufficient prospective process-wall allowance')
    hashes = source_fingerprint()
    if kind != 'source':
        source_receipts = [json.loads(p.read_text()) for p in
                           root.glob('source-check*/receipt.json')]
        if not any(r.get('exit_code') == 0 and r.get('stop_reason') == 'EXITED' and
                   r.get('source_sha256') == hashes
                   for r in source_receipts):
            raise RuntimeError('passing matching full source check required before numerical child')
    if kind == '64':
        candidates = sorted(root.glob('grid-32x8*/receipt.json'))
        valid = [json.loads(p.read_text()) for p in candidates
                 if json.loads(p.read_text()).get('source_sha256') == hashes]
        if not valid:
            raise RuntimeError('32x8 memory evidence required before 64x16')
        previous = valid[-1]
        if (previous['exit_code'] != 0 or previous['stop_reason'] != 'EXITED' or
                4 * previous['peak_process_tree_rss_bytes'] > MEMORY):
            raise RuntimeError('conservative 64x16 memory estimate exceeds 2 GiB')
    path.mkdir(parents=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_DISABLE_JIT='1',
               PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = os.pathsep.join((str(WORKTREE / 'src'), str(WORKTREE / 'tests'),
                                        env.get('PYTHONPATH', '')))
    command = ([sys.executable, '-m', 'pytest', '-q',
                'tests/test_r8_f03_c2_current_owner_bridge_dev.py',
                '--basetemp', str(path / 'pytest-temp')] if kind == 'source'
               else [sys.executable, '-m',
                     'scripts.r8_f03_c2_current_owner_bridge_dev', '--evaluate', kind,
                     '--output', str(path)])
    started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    started = time.monotonic()
    peak = 0
    reason = None
    members = []
    with (path / 'stdout.log').open('wb') as stdout, (path / 'stderr.log').open('wb') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        tree = psutil.Process(process.pid)
        tree.cpu_affinity([0])
        while process.poll() is None:
            try:
                members = [tree] + tree.children(recursive=True)
                peak = max(peak, sum(p.memory_info().rss for p in members if p.is_running()))
            except psutil.Error:
                pass
            if peak > MEMORY:
                reason = 'MEMORY_CAP'
                break
            if time.monotonic() - started >= timeout:
                reason = 'WALL_TIMEOUT'
                break
            time.sleep(.05)
        if reason:
            for member in reversed(members):
                try:
                    member.kill()
                except psutil.Error:
                    pass
        process.wait()
    elapsed = time.monotonic() - started
    receipt = dict(command=command, kind=kind, source_sha256=hashes,
                   started_utc=started_utc,
                   cpu='logical CPU 0', threads=1, jit_disabled=True,
                   memory_cap_bytes=MEMORY, timeout_seconds=timeout,
                   exit_code=process.returncode, stop_reason=reason or 'EXITED',
                   process_wall_seconds=elapsed,
                   peak_process_tree_rss_bytes=peak,
                   prior_wall_seconds=used, cumulative_wall_seconds=used+elapsed,
                   remaining_wall_seconds=LIMIT-used-elapsed,
                   final_reserve_seconds=RESERVE)
    (path / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', choices=('32', '64'))
    parser.add_argument('--check', choices=('source', '32', '64'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.evaluate:
        if args.output is None:
            parser.error('--output required with --evaluate')
        try:
            result = evaluate(int(args.evaluate), args.output)
        except Exception as error:
            result = dict(status='HOLD_DEVELOPMENT_ONLY', error=repr(error))
            (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            raise
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'status': result['status'], 'grid': result['grid'],
                          'checks': result['checks']}))
    elif args.check:
        receipt = guarded_run(args.check)
        print(json.dumps(receipt))
        raise SystemExit(0 if receipt['exit_code'] == 0 and receipt['stop_reason'] == 'EXITED' else 1)
    else:
        parser.error('--evaluate or --check required')
