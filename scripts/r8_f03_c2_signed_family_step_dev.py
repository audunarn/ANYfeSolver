"""One current-owner 32x8 c2 P0 signed-family development discriminator.

This is a reference-state, single-load check, not F03 qualification. In
particular the full mixed K is signed: only checked local resultant blocks are
eliminated, and all nodal and retained rotational coordinates remain physical.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKTREE/'src'))
sys.path.insert(0,str(WORKTREE/'tests'))

import numba
import numpy as np
import psutil
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info

import anyfesolver
from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium, _scaled_constraint_gap
from anyfesolver._b2_physical_linear_batch_dev import _accurate_sparse_residual
from scripts.b3_e4_bl5_coherent_aggregation_dev import model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def product_provenance(owner=None):
    modules = dict(candidate=__file__,anyfesolver=anyfesolver.__file__,
        stiffened_shell=stiffened_shell.__file__,
        equilibrium=sys.modules[_linear_equilibrium.__module__].__file__,
        compensated_residual=sys.modules[_accurate_sparse_residual.__module__].__file__,
        model=sys.modules[model.__module__].__file__,
        frozen_model_fixture=sys.modules['test_stiffened_shell_l1_assembly_experimental'].__file__)
    if owner is not None:
        modules['owner']=sys.modules[type(owner).__module__].__file__
    for module_name,module in tuple(sys.modules.items()):
        if module_name.startswith('anyfesolver.') and getattr(module,'__file__',None):
            modules['product_module:'+module_name]=module.__file__
    roots = dict(candidate=WORKTREE/'scripts',model=WORKTREE/'scripts',
        frozen_model_fixture=WORKTREE/'tests')
    for name,path in modules.items():
        root = roots.get(name,WORKTREE/'src')
        require(Path(path).resolve().is_relative_to(root.resolve()),
                name+' import escaped assigned worktree: '+str(path))
    return {name:dict(path=str(Path(path).resolve()),sha256=sha(path))
            for name,path in modules.items()}


def relative(a, b, floor=1e-14):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), floor))


def spectral_norm(a):
    return float(linalg.norm(a, 2))


def null_basis(cs):
    """Sparse fundamental basis after rank-revealing choice of pivot columns."""
    m, n = cs.shape
    dense = cs.toarray()
    _, r, permutation = linalg.qr(dense, pivoting=True, mode='economic')
    rank_budget = 64*np.finfo(float).eps*max(m,n)*abs(r[0,0])
    require(abs(r[m-1,m-1]) > rank_budget, 'constraint rank unresolved')
    pivot, rest = permutation[:m], permutation[m:]
    coefficients = -linalg.solve(dense[:,pivot], dense[:,rest], assume_a='gen')
    rows, cols = np.nonzero(coefficients)
    z = sparse.coo_matrix((np.r_[np.ones(len(rest)),coefficients[rows,cols]],
        (np.r_[rest,pivot[rows]],np.r_[np.arange(len(rest)),cols])),
        shape=(n,len(rest))).tocsr()
    leakage = float(np.linalg.norm(dense@z.toarray(),ord='fro'))
    pivot_inverse_norm = float(spectral_norm(linalg.inv(dense[:,pivot])))
    return z, dict(rank_budget=float(rank_budget),minimum_qr_pivot=float(abs(r[m-1,m-1])),
        pivot_count=m,basis_nnz=int(z.nnz),constraint_leakage_fro=leakage,
        pivot_inverse_norm_2=pivot_inverse_norm,
        pivot_condition_2=float(np.linalg.cond(dense[:,pivot])))


def cholesky_certificate(matrix):
    """Binary64 Cholesky positivity estimate; not an interval certificate."""
    a = matrix.toarray()
    require(relative(a,a.T) <= 1e-10, 'projected kernel skew')
    a = .5*(a+a.T)
    factor = linalg.cholesky(a, lower=True, check_finite=True)
    residual = float(np.linalg.norm(a-factor@factor.T,ord='fro'))
    inverse = linalg.solve_triangular(factor,np.eye(len(a)),lower=True)
    inverse_fro_sq = float(np.sum(inverse**2))
    lower = 1./inverse_fro_sq-residual
    return dict(lower_bound=lower,inverse_fro_sq=inverse_fro_sq,
        factor_residual_fro=residual,order=len(a),dense_bytes=int(a.nbytes),
        factor_bytes=int(factor.nbytes),inverse_bytes=int(inverse.nbytes))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def evaluate(screen_path=None):
    started = time.monotonic()
    provenance = product_provenance()
    pools = threadpool_info()
    require(numba.config.DISABLE_JIT and numba.get_num_threads() == 1 and
            pools and all(p['num_threads'] == 1 for p in pools), 'effective one-thread/JIT-off settings required')
    definition, policy = model(32, 8, 64, 2)
    require(policy['span_halfcells'] == 4 and policy['beam_halfcells'] == 64 and
            len(policy['support_membership']) == 16,
            'exact c2 P0 case required')
    owner = stiffened_shell.create_analysis(definition=definition)
    try:
        provenance = product_provenance(owner)
        k, c = owner._K, owner._C
        free, fixed = owner._free, owner._fixed
        coordinate, constraint = owner._coordinate_scale, owner._constraint_scale
        require(c.shape[0] == 80 and len(owner._beam_parts) == 32 and
                len(free) + c.shape[0] <= 5000, 'actual owner dimensions outside bounded case')
        assembled = sparse.bmat([[k[free][:,free], c[:,free].T],
                [c[:,free], None]], format='csc')
        require(sparse.linalg.norm(owner._A-assembled)==0.,
                'owner supported KKT differs from reference K/C/supports')
        require(sparse.linalg.norm(k-k.T)/sparse.linalg.norm(k) <= 1e-9,
                'full signed K symmetry unresolved')
        base = owner._shell_count + 6*len(owner._beam_ids)
        blocks = [base + 24*i + np.arange(6, 24) for i in range(32)]
        pp = np.concatenate(blocks)
        x = np.setdiff1d(free, pp)
        require(len(x) <= 3500 and np.isin(pp, free).all() and c[:,pp].nnz == 0,
                'local resultant inventory or constraint separation unsupported')
        off = k[pp][:,pp] - sparse.block_diag([k[p][:,p] for p in blocks])
        require(np.max(np.abs(off.data), initial=0.) == 0., 'resultants are not independent local blocks')
        free_map = {int(v):i for i,v in enumerate(free)}
        rows = [free_map[int(v)] for v in x]
        cols = list(range(len(x)))
        vals = [1.]*len(x)
        s = k[x][:,x].tocsr()
        maps = []
        local = []
        for p in blocks:
            d = k[p][:,p].toarray()
            require(relative(d,d.T) <= 1e-9, 'local block skew')
            factor = linalg.cho_factor(-d, lower=True)
            chol = np.tril(factor[0])
            ferr = d+chol@chol.T
            eig = linalg.eigvalsh(d, lower=True)
            budget = spectral_norm(ferr)+64*np.finfo(float).eps*len(p)*spectral_norm(d)
            require(-eig[-1] > budget, 'local negative resultant unresolved')
            coupling = k[p][:,x]
            neighbours = np.unique(coupling.nonzero()[1])
            require(len(neighbours) <= 24, 'unexpected broad local coupling')
            px = coupling[:,neighbours].toarray()
            lift = linalg.cho_solve(factor, px)
            residual = d@lift+px
            backward = float(np.linalg.norm(residual)/max(np.linalg.norm(d)*np.linalg.norm(lift)+np.linalg.norm(px),1e-30))
            require(backward <= 1e-10, 'stationary lift residual')
            update = k[x[neighbours]][:,p].toarray()@lift
            s += sparse.coo_matrix((update.ravel(),(np.repeat(neighbours,len(neighbours)),
                    np.tile(neighbours,len(neighbours)))),shape=s.shape).tocsr()
            rows.extend(np.repeat([free_map[int(v)] for v in p],len(neighbours)))
            cols.extend(np.tile(neighbours,len(p)))
            vals.extend(lift.ravel())
            maps.append((neighbours,lift,factor,residual,ferr))
            local.append(dict(maximum_eigenvalue=float(eig[-1]),negative_budget=float(budget),
                              stationary_backward=float(backward)))
        lifted = sparse.coo_matrix((vals,(rows,cols)),shape=(len(free),len(x))).tocsr()
        pulled = lifted.T@k[free][:,free]@lifted
        work_pullback = float(sparse.linalg.norm(pulled-s)/max(sparse.linalg.norm(s),1e-30))
        cx = c[:,x].toarray()
        constraint_pullback = relative((c[:,free]@lifted).toarray(), cx)
        require(max(work_pullback,constraint_pullback) <= 1e-10, 'stationary work/C pullback mismatch')
        scale = coordinate[x].copy()
        diag = np.abs(s.diagonal()*scale**2)
        require(np.max(diag)>0, 'zero reduced diagonal scale')
        scale /= np.sqrt(np.where(diag>0,diag,np.max(diag)))
        sd = sparse.diags(scale)
        scaled = (sd@s@sd).tocsr()
        require(sparse.linalg.norm(scaled-scaled.T)/sparse.linalg.norm(scaled)<=1e-9,
                'reduced matrix skew')
        cs = cx*scale
        row_norm = np.linalg.norm(cs,axis=1)
        require(np.all(row_norm>0), 'zero constraint row')
        cs /= row_norm[:,None]
        singular = linalg.svdvals(cs)
        eps = np.finfo(float).eps
        rank_budget = 64*eps*max(cs.shape)*singular[0]
        require(singular[-1]>rank_budget,'constraint rank unresolved')
        z, basis = null_basis(sparse.csr_matrix(cs))
        projected = (z.T@scaled@z).tocsr()
        dense_bytes = projected.shape[0]**2*8
        screen = dict(full_q=int(k.shape[0]),free=int(len(free)),physical=int(len(x)),
            local_resultants=int(len(pp)),constraints=int(c.shape[0]),
            projected_order=int(projected.shape[0]),projected_nnz=int(projected.nnz),
            projected_dense_bytes=int(dense_bytes),estimated_three_dense_arrays_bytes=int(3*dense_bytes),
            basis=basis,elapsed_seconds=time.monotonic()-started)
        if screen_path is not None:
            Path(screen_path).write_text(json.dumps(screen,indent=2)+'\n')
        require(projected.shape[0] <= 3000 and dense_bytes*5 <= 2*(1<<30),
                'projected Cholesky memory screen failed')
        certificate = cholesky_certificate(projected)
        snorm = float(sparse.linalg.norm(scaled,ord='fro'))
        leakage_correction = basis['pivot_inverse_norm_2']*basis['constraint_leakage_fro']
        basis_norm = float(sparse.linalg.norm(z,ord='fro'))
        basis_norm_2_sq_bound = float(sparse.linalg.norm(z,ord=1)*sparse.linalg.norm(z,ord=np.inf))
        leakage_error = snorm*(2*basis_norm*leakage_correction+leakage_correction**2)
        eliminated = float(sparse.linalg.norm(sd@(pulled-s)@sd,ord='fro'))
        for neighbours,lift,factor,residual,ferr in maps:
            r = residual*scale[neighbours]
            eliminated += float(np.linalg.norm(r.T@linalg.cho_solve(factor,r),ord='fro'))
            ls = lift*scale[neighbours]
            eliminated += float(np.linalg.norm(ls.T@ferr@ls,ord='fro'))
        terms = dict(rounding=64*eps*len(x)*snorm*basis_norm_2_sq_bound,
            elimination=eliminated*basis_norm_2_sq_bound,
            kernel_leakage=leakage_error,
            physical_triangle_completion=float(sparse.linalg.norm(scaled-scaled.T,ord='fro'))*basis_norm_2_sq_bound)
        uncertainty = float(sum(terms.values()))
        kernel_positive = bool(certificate['lower_bound']>uncertainty)
        inertia = (dict(positive=len(x),negative=c.shape[0],unresolved=0)
                   if kernel_positive else None)
        expected = dict(positive=len(x),negative=c.shape[0],unresolved=0)
        action = stiffened_shell.NodalAction('beam',165,(0.,100.,0.,0.,0.,0.))
        owner_result = owner.solve_linear((action,))
        load = owner._accepted[1].copy()
        require(np.max(np.abs(load[pp]))==0.,'nonzero resultant RHS unsupported')
        rhs = np.r_[load[free],np.zeros(c.shape[0])]
        full_factor = splu(owner._A)
        full = full_factor.solve(rhs)
        for _ in range(3):
            defect = _accurate_sparse_residual(owner._A,full[:,None],rhs[:,None])[:,0]
            full += full_factor.solve(-defect)
        saddle = sparse.bmat([[scaled,sparse.csr_matrix(cs.T)],
            [sparse.csr_matrix(cs),None]],format='csc')
        red_rhs = np.r_[load[x]*scale,np.zeros(c.shape[0])]
        red_factor = splu(saddle)
        red = red_factor.solve(red_rhs)
        for _ in range(3):
            defect = _accurate_sparse_residual(saddle,red[:,None],red_rhs[:,None])[:,0]
            red += red_factor.solve(-defect)
        triples = {}
        for name in ('full','reduced','owner'):
            q = np.zeros(k.shape[0])
            if name=='full':
                q[free] = full[:len(free)]; lm = full[len(free):]
            elif name=='reduced':
                q[free] = lifted@(red[:len(x)]*scale); lm = red[len(x):]/row_norm
            else:
                q = np.r_[owner_result.shell_displacement,owner_result.beam_coordinates]
                lm = owner_result.multipliers
            residual = k@q+c.T@lm-load
            reaction = np.zeros_like(q); reaction[fixed] = residual[fixed]
            triples[name] = (q,lm,reaction,residual)
        shell_end = owner._shell_count
        roles = dict(shell=x[x<shell_end],beam_nodes=x[(x>=shell_end)&(x<base)],
                     retained_rotations=x[x>=base])
        require(all(len(v)>0 for v in roles.values()),'physical role inventory incomplete')
        def pair_parity(a,b):
            values = {name:relative(a[0][slots]/coordinate[slots],b[0][slots]/coordinate[slots])
                      for name,slots in roles.items()}
            values.update(resultants=relative(a[0][pp]/coordinate[pp],b[0][pp]/coordinate[pp]),
                          multipliers=relative(a[1]*constraint,b[1]*constraint),
                          reactions=relative(a[2]*coordinate,b[2]*coordinate),
                          applied_work=relative(np.array(a[0]@load),np.array(b[0]@load)))
            return values
        parity = pair_parity(triples['full'],triples['reduced'])
        owner_parity = pair_parity(triples['owner'],triples['reduced'])
        metrics = {}
        for name,(q,lm,reaction,residual) in triples.items():
            work = float(q@load)
            metrics[name] = dict(equilibrium=float(_linear_equilibrium(residual[free],load[free],coordinate[free])),
                weak_gap=float(_scaled_constraint_gap(c@q,constraint)),
                energy_work_relative=relative(np.array(q@(k@q)),np.array(work)),
                compatible_work_relative=abs(float(q@(c.T@lm)))/max(abs(work),1e-12),
                ground_work_relative=abs(float(q@reaction))/max(abs(work),1e-12),
                work_j=work)
        predicates = dict(kernel_positive=kernel_positive,
            reduced_inertia=inertia==expected,
            solution_parity=all(v<=1e-10 for row in (parity,owner_parity) for v in row.values()),
            equilibrium=all(v['equilibrium']<=1e-8 for v in metrics.values()),
            weak_gap=all(v['weak_gap']<=1e-9 for v in metrics.values()),
            work=all(max(v['energy_work_relative'],v['compatible_work_relative'],
                         v['ground_work_relative'])<=1e-8 for v in metrics.values()))
        return dict(status='PASS_DEVELOPMENT_ONLY' if all(predicates.values()) else 'HOLD_DEVELOPMENT_ONLY',
            scope='single matched 32x8/64 native halfcell c2 P0 reference case; no F03 or family acceptance',
            policy=policy,source_sha256=sha(__file__),model_source_sha256=sha(sys.modules[model.__module__].__file__),
            owner_source_sha256=sha(sys.modules[type(owner).__module__].__file__),
            product_provenance=provenance,
            anyfesolver_import=anyfesolver.__file__,effective=dict(cpu_affinity=psutil.Process().cpu_affinity(),
                numba_threads=numba.get_num_threads(),jit_disabled=bool(numba.config.DISABLE_JIT),pools=pools),
            dimensions=dict(full_q=k.shape[0],free=len(free),physical=len(x),resultants=len(pp),
                            constraints=c.shape[0],fixed=len(fixed)),
            reference_multiplier_curvature='zero at zero reference multiplier; owner._K plus actual C is owner._A',
            local_blocks=local,work_pullback=work_pullback,constraint_pullback=constraint_pullback,
            rank_min_singular=float(singular[-1]),rank_budget=float(rank_budget),
            kernel_certificate=certificate,screen=screen,
            basis_norm_2_sq_bound=basis_norm_2_sq_bound,spectral_uncertainty=uncertainty,
            spectral_uncertainty_terms=terms,reduced_inertia=inertia,expected_reduced_inertia=expected,
            implied_full_inertia=dict(positive=len(x),negative=len(pp)+c.shape[0],unresolved=0)
                if predicates['kernel_positive'] and predicates['reduced_inertia'] else None,
            full_compensated_refinements=3,reduced_compensated_refinements=3,
            parity=parity,owner_reduced_parity=owner_parity,metrics=metrics,predicates=predicates,
            wall_seconds=time.monotonic()-started)
    finally:
        owner.close()


def prior_process_wall(root):
    """Account for every attempt, refusing missing or malformed receipts."""
    root = Path(root)
    totals = {'numerical': 0., 'source_control': 0.}
    for kind, pattern in (('numerical', 'attempt*'),
                          ('source_control', 'source*')):
        for directory in root.glob(pattern):
            if not directory.is_dir():
                continue
            path = directory/'receipt.json'
            require(path.is_file(), 'unmetered prior attempt: '+str(directory))
            item = json.loads(path.read_text())
            value = item.get('process_wall_seconds',item.get('process_wall_seconds_upper_bound'))
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0.,
                    'invalid process-wall receipt: '+str(path))
            totals[kind] += value
    return totals


def guarded_run(output, kind='numerical', timeout=None, memory=2*(1<<30)):
    """One bounded child under the shared 180 s prospective development allowance."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    require(kind in ('numerical','source_control'), 'unsupported child kind')
    cap = 45. if kind == 'numerical' else 30.
    timeout = cap if timeout is None else float(timeout)
    require(0. < timeout <= cap and memory <= 2*(1<<30), 'child cap violated')
    prior = prior_process_wall(output.parent)
    used = sum(prior.values())
    require(used+timeout+60. <= 180.,
            'insufficient cumulative allowance for child cap and 60s final reserve')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1',NUMBA_DISABLE_JIT='1',PYTHONUNBUFFERED='1')
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    env['PYTHONPATH']=os.pathsep.join((str(WORKTREE/'src'),str(WORKTREE/'tests'),
                                      env.get('PYTHONPATH','')))
    command = ([sys.executable,'-m','scripts.r8_f03_c2_signed_family_step_dev',
                '--evaluate',str(output/'result.json')] if kind == 'numerical' else
               [sys.executable,'-m','pytest','-q',
                'tests/test_r8_f03_c2_signed_family_step_dev.py',
                '--basetemp',str(output/'pytest_tmp')])
    products = product_provenance()
    output.mkdir(parents=True)
    receipt = dict(kind=kind,command=command,source_product_sha256=products,
        cpu='logical CPU 0',threads=1,jit_disabled=True,memory_cap_bytes=memory,
        timeout_seconds=timeout,input=dict(shell_edges=32,transverse_edges=8,halfcells=64,c_lambda=2,basis='p0'),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    started = time.monotonic()
    process = None
    reason = None
    peak = 0
    supervisor_error = None
    try:
        with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
            tree = psutil.Process(process.pid)
            tree.cpu_affinity([0])
            while process.poll() is None:
                try:
                    members = [tree]+tree.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in members if p.is_running())
                    peak = max(peak,rss)
                    if rss>memory:
                        reason='MEMORY_CAP'; break
                except psutil.Error:
                    pass
                if time.monotonic()-started>=timeout:
                    reason='WALL_TIMEOUT'; break
                time.sleep(.05)
            if reason:
                for member in reversed(members):
                    try:member.kill()
                    except psutil.Error:pass
            process.wait()
    except Exception as error:
        reason = 'SUPERVISOR_ERROR'
        supervisor_error = repr(error)
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            process.wait()
    elapsed = time.monotonic()-started
    receipt.update(ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   exit_code=None if process is None else process.returncode,
                   stop_reason=reason or 'EXITED',supervisor_error=supervisor_error,
                   process_wall_seconds=elapsed,peak_process_tree_rss_bytes=peak,
                   budget_seconds=180.,prior_numerical_wall_seconds=prior['numerical'],
                   prior_source_control_wall_seconds=prior['source_control'],
                   cumulative_wall_seconds=used+elapsed,
                   remaining_seconds=max(0.,180.-used-elapsed))
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source-output',type=Path)
    args=parser.parse_args()
    if args.evaluate:
        try:
            result=evaluate(args.evaluate.parent/'screen.json')
        except Exception as error:
            result=dict(status='HOLD_DEVELOPMENT_ONLY',error=repr(error),source_sha256=sha(__file__))
            args.evaluate.write_text(json.dumps(result,indent=2)+'\n')
            raise
        args.evaluate.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('status','kernel_certificate','spectral_uncertainty','reduced_inertia')}))
        raise SystemExit(0 if result['status']=='PASS_DEVELOPMENT_ONLY' else 1)
    if args.output:
        receipt=guarded_run(args.output)
        print(json.dumps({k:receipt[k] for k in ('kind','exit_code','stop_reason',
            'process_wall_seconds','peak_process_tree_rss_bytes','cumulative_wall_seconds',
            'remaining_seconds')}))
        raise SystemExit(0 if receipt['exit_code']==0 else 1)
    if args.source_output:
        receipt=guarded_run(args.source_output,kind='source_control')
        print(json.dumps({k:receipt[k] for k in ('kind','exit_code','stop_reason',
            'process_wall_seconds','peak_process_tree_rss_bytes','cumulative_wall_seconds',
            'remaining_seconds')}))
        raise SystemExit(0 if receipt['exit_code']==0 else 1)
    parser.error('--evaluate, --output, or --source-output required')
