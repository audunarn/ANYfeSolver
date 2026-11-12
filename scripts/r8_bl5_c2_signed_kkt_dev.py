"""One current-source BL5 c2 P0 signed-KKT development discriminator.

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


def relative(a, b, floor=1e-14):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), floor))


def spectral_norm(a):
    return float(linalg.norm(a, 2))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def evaluate():
    started = time.monotonic()
    pools = threadpool_info()
    require(numba.config.DISABLE_JIT and numba.get_num_threads() == 1 and
            pools and all(p['num_threads'] == 1 for p in pools), 'effective one-thread/JIT-off settings required')
    definition, policy = model(16, 4, 32, 2)
    require(policy['span_halfcells'] == 4 and policy['beam_halfcells'] == 32,
            'exact c2 P0 case required')
    owner = stiffened_shell.create_analysis(definition=definition)
    try:
        k, c = owner._K, owner._C
        free, fixed = owner._free, owner._fixed
        coordinate, constraint = owner._coordinate_scale, owner._constraint_scale
        require(c.shape[0] == 40 and len(owner._beam_parts) == 16 and
                len(free) + c.shape[0] <= 1500, 'actual owner dimensions outside bounded case')
        require(relative((owner._A - sparse.bmat([[k[free][:,free], c[:,free].T],
                [c[:,free], None]], format='csc')).toarray(), np.zeros(owner._A.shape)) == 0.,
                'owner supported KKT differs from reference K/C/supports')
        require(sparse.linalg.norm(k-k.T)/sparse.linalg.norm(k) <= 1e-9,
                'full signed K symmetry unresolved')
        base = owner._shell_count + 6*len(owner._beam_ids)
        blocks = [base + 24*i + np.arange(6, 24) for i in range(16)]
        pp = np.concatenate(blocks)
        x = np.setdiff1d(free, pp)
        require(len(x) <= 1000 and np.isin(pp, free).all() and c[:,pp].nnz == 0,
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
        scaled = (sd@s@sd).toarray()
        require(relative(scaled,scaled.T)<=1e-9, 'reduced matrix skew')
        cs = cx*scale
        row_norm = np.linalg.norm(cs,axis=1)
        require(np.all(row_norm>0), 'zero constraint row')
        cs /= row_norm[:,None]
        singular = linalg.svdvals(cs)
        eps = np.finfo(float).eps
        rank_budget = 64*eps*max(cs.shape)*singular[0]
        require(singular[-1]>rank_budget,'constraint rank unresolved')
        qbasis,_ = linalg.qr(cs.T,mode='full')
        z = qbasis[:,c.shape[0]:]
        restricted = z.T@scaled@z
        eig,vectors = linalg.eigh(restricted,lower=True)
        snorm = float(np.max(np.abs(linalg.eigvalsh(scaled,lower=True))))
        orth = spectral_norm(z.T@z-np.eye(z.shape[1]))
        leak = spectral_norm(cs@z)/singular[-1]
        eliminated = spectral_norm((sd@(pulled-s)@sd).toarray())
        for neighbours,lift,factor,residual,ferr in maps:
            r = residual*scale[neighbours]
            eliminated += spectral_norm(r.T@linalg.cho_solve(factor,r))
            ls = lift*scale[neighbours]
            eliminated += spectral_norm(ls.T@ferr@ls)
        triangle = lambda a: spectral_norm(a-(np.tril(a)+np.tril(a,-1).T))
        terms = dict(rounding=64*eps*len(x)*snorm,elimination=eliminated,
            kernel_leakage=(2*leak+leak**2)*snorm,
            kernel_orthogonality=(2*orth+orth**2)*snorm,
            physical_triangle_completion=triangle(scaled),
            eigen_residual=spectral_norm(restricted@vectors-vectors*eig))
        uncertainty = float(sum(terms.values()))
        saddle = np.block([[scaled,cs.T],[cs,np.zeros((c.shape[0],c.shape[0]))]])
        saddle_eig = linalg.eigvalsh(saddle,lower=True)
        saddle_budget = 64*eps*len(saddle)*float(np.max(np.abs(saddle_eig)))+uncertainty+triangle(saddle)
        inertia = dict(positive=int(np.sum(saddle_eig>saddle_budget)),
                       negative=int(np.sum(saddle_eig< -saddle_budget)),
                       unresolved=int(np.sum(np.abs(saddle_eig)<=saddle_budget)))
        expected = dict(positive=len(x),negative=c.shape[0],unresolved=0)
        action = stiffened_shell.NodalAction('beam',133,(0.,100.,0.,0.,0.,0.))
        owner_result = owner.solve_linear((action,))
        load = owner._accepted[1].copy()
        require(np.max(np.abs(load[pp]))==0.,'nonzero resultant RHS unsupported')
        rhs = np.r_[load[free],np.zeros(c.shape[0])]
        full_factor = splu(owner._A)
        full = full_factor.solve(rhs)
        for _ in range(3):
            defect = _accurate_sparse_residual(owner._A,full[:,None],rhs[:,None])[:,0]
            full += full_factor.solve(-defect)
        red = linalg.solve(saddle,np.r_[load[x]*scale,np.zeros(c.shape[0])],assume_a='sym',lower=True)
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
        predicates = dict(kernel_positive=bool(eig[0]>uncertainty),
            reduced_inertia=inertia==expected,
            solution_parity=all(v<=1e-10 for row in (parity,owner_parity) for v in row.values()),
            equilibrium=all(v['equilibrium']<=1e-8 for v in metrics.values()),
            weak_gap=all(v['weak_gap']<=1e-9 for v in metrics.values()),
            work=all(max(v['energy_work_relative'],v['compatible_work_relative'],
                         v['ground_work_relative'])<=1e-8 for v in metrics.values()))
        return dict(status='PASS_DEVELOPMENT_ONLY' if all(predicates.values()) else 'HOLD_DEVELOPMENT_ONLY',
            scope='single matched 16x4/32 native halfcell c2 P0 reference case; no F03 or family acceptance',
            policy=policy,source_sha256=sha(__file__),model_source_sha256=sha(sys.modules[model.__module__].__file__),
            owner_source_sha256=sha(sys.modules[type(owner).__module__].__file__),
            anyfesolver_import=anyfesolver.__file__,effective=dict(cpu_affinity=psutil.Process().cpu_affinity(),
                numba_threads=numba.get_num_threads(),jit_disabled=bool(numba.config.DISABLE_JIT),pools=pools),
            dimensions=dict(full_q=k.shape[0],free=len(free),physical=len(x),resultants=len(pp),
                            constraints=c.shape[0],fixed=len(fixed)),
            reference_multiplier_curvature='zero at zero reference multiplier; owner._K plus actual C is owner._A',
            local_blocks=local,work_pullback=work_pullback,constraint_pullback=constraint_pullback,
            rank_min_singular=float(singular[-1]),rank_budget=float(rank_budget),
            minimum_kernel_eigenvalue=float(eig[0]),spectral_uncertainty=uncertainty,
            spectral_uncertainty_terms=terms,reduced_inertia=inertia,expected_reduced_inertia=expected,
            implied_full_inertia=dict(positive=len(x),negative=len(pp)+c.shape[0],unresolved=0)
                if predicates['kernel_positive'] and predicates['reduced_inertia'] else None,
            saddle_budget=float(saddle_budget),full_compensated_refinements=3,
            parity=parity,owner_reduced_parity=owner_parity,metrics=metrics,predicates=predicates,
            wall_seconds=time.monotonic()-started)
    finally:
        owner.close()


def prior_process_wall(root):
    """Account for every recorded numerical attempt and source control."""
    root = Path(root)
    totals = {'numerical': 0., 'source_control': 0.}
    for kind, paths in (('numerical', root.glob('attempt*/receipt.json')),
                        ('source_control', root.glob('source*-receipt.json'))):
        for path in paths:
            value = json.loads(path.read_text())['process_wall_seconds']
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0.,
                    'invalid process-wall receipt: '+str(path))
            totals[kind] += value
    return totals


def guarded_run(output, timeout=60., memory=2*(1<<30)):
    """One child process when the shared 150s allowance can fit cap and reserve."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    require(timeout == 60. and memory <= 2*(1<<30), 'fixed 60s/2GiB execution caps required')
    prior = prior_process_wall(output.parent)
    used = sum(prior.values())
    require(used+timeout+60. <= 150.,
            'insufficient cumulative allowance for 60s attempt and 60s final reserve')
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1',NUMBA_DISABLE_JIT='1',PYTHONUNBUFFERED='1')
    command = [sys.executable,'-m','scripts.r8_bl5_c2_signed_kkt_dev','--evaluate',str(output/'result.json')]
    receipt = dict(command=command,source_sha256=sha(__file__),
        model_source_sha256=sha(sys.modules[model.__module__].__file__),
        cpu='logical CPU 0',threads=1,jit_disabled=True,memory_cap_bytes=memory,
        timeout_seconds=timeout,input=dict(shell_edges=16,transverse_edges=4,halfcells=32,c_lambda=2,basis='p0'),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    started = time.monotonic()
    with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
        process = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
        tree = psutil.Process(process.pid)
        tree.cpu_affinity([0])
        reason = None
        peak = 0
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
            for p in reversed(members):
                try:p.kill()
                except psutil.Error:pass
        process.wait()
    elapsed = time.monotonic()-started
    receipt.update(exit_code=process.returncode,stop_reason=reason or 'EXITED',
                   process_wall_seconds=elapsed,peak_process_tree_rss_bytes=peak,
                   budget_seconds=150.,prior_numerical_wall_seconds=prior['numerical'],
                   prior_source_control_wall_seconds=prior['source_control'],
                   cumulative_wall_seconds=used+elapsed,
                   remaining_seconds=max(0.,150.-used-elapsed))
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.evaluate:
        try:
            result=evaluate()
        except Exception as error:
            result=dict(status='HOLD_DEVELOPMENT_ONLY',error=repr(error),source_sha256=sha(__file__))
            args.evaluate.write_text(json.dumps(result,indent=2)+'\n')
            raise
        args.evaluate.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('status','minimum_kernel_eigenvalue','spectral_uncertainty','reduced_inertia')}))
        raise SystemExit(0 if result['status']=='PASS_DEVELOPMENT_ONLY' else 1)
    if args.output:
        receipt=guarded_run(args.output)
        print(json.dumps(receipt))
        raise SystemExit(0 if receipt['exit_code']==0 else 1)
    parser.error('--evaluate or --output required')
