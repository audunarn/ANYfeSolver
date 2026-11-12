"""One adopted missing original-target correction direction; no primary solve.

The exact-work evaluator observes stored operators and candidates. It does not
change the residual, backend, factor ordering, physical law or work scorer.
"""
import argparse
import ast
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/agent_plans/development/r8-b2-physical-work-unblock-20261010'
ARCHIVE = ROOT/'docs/agent_plans/development/r8-b2-large-work-allocation-20261009/attempt-v1/full-state.npz'
OLD = ROOT/'docs/agent_plans/development/r8-b2-compensated-refinement-20261009'
ENV = {k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT')}
ENV['PYTHONOPTIMIZE'] = '0'
CAPS = dict(seconds=60., rss_bytes=2*1024**3, operations=50000000,
    bits=4096, fill=4000000, factor_fill=50000000, row_terms=8192)
SOURCES = ('scripts/r8_b2_physical_work_unblock_dev.py',
    'src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py',
    'src/anyfesolver/_b2_physical_linear_batch_dev.py',
    'src/anyfesolver/_stiffened_shell_b2_affine_resolution.py',
    'src/anyfesolver/linalg.py', 'src/anyfesolver/threading_policy.py',
    'scripts/_b3_e4_f03_local_scientific_supervisor.py',
    'scripts/_b3_e4_f03_local_scientific_cases.py')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
        allow_nan=False, default=str)+'\n', encoding='utf-8')


def static_binding():
    reg = json.loads((OLD/'registration.json').read_text())
    require(sha(ARCHIVE) == reg['saved_state_sha256'], 'original archive drift')
    require(sha(ROOT/'src/anyfesolver/linalg.py') == reg['old_linalg_source_sha256'],
        'same factor implementation required')
    frozen = ROOT/'docs/agent_plans/development/r8-b2-refinement-installed-20261009/site-v2/anyfesolver'
    require(sha(ROOT/'src/anyfesolver/threading_policy.py') == sha(frozen/'threading_policy.py'),
        'factor thread policy drift')
    segments = []
    for p in (ROOT/SOURCES[1], frozen/'_stiffened_shell_b2_q4_panel_experimental.py'):
        text = p.read_text()
        node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)
            and n.name == '_scaled_point_kkt')
        segments.append(hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest())
    require(segments == ['efca15a5d63373b7ea5d3e04951737940d9b3fedafc25bebf7c9943fe3dde917']*2,
        'scaling source drift')
    return dict(archive_sha256=sha(ARCHIVE), matrix_sha256=reg['matrix_sha256'],
        source_sha256={p:sha(ROOT/p) for p in SOURCES}, scaling_source_sha256=segments[0],
        caps=CAPS, primary_rhs_solve=False, maximum_factors=1, maximum_direction_solves=1,
        adopted_scope='one missing original-target direction; not saved-case acceptance')


class ExactWork:
    """Streaming dyadic products in a fixed 2^-3222 unit, no dense exact tensor.

    A binary64 operand denominator divides 2^1074, so any triple product is
    represented exactly. Cancellation occurs before conversion or absolute
    value. Counters include two multiplies, shift and accumulator addition.
    """
    def __init__(self): self.operations = 0

    def tick(self, count, value):
        self.operations += count
        require(self.operations <= CAPS['operations'], 'exact operation cap')
        require(abs(value).bit_length() <= CAPS['bits'], 'exact accumulator bit cap')

    @staticmethod
    def vector(values):
        result=[]
        for x in values:
            x=float(x); require(math.isfinite(x), 'nonfinite exact-work operand')
            n,d=x.as_integer_ratio(); result.append((n,d.bit_length()-1))
        return result

    @staticmethod
    def difference(left, right):
        result=[]
        for (a,e),(b,f) in zip(left,right):
            g=max(e,f); result.append(((a<<(g-e))-(b<<(g-f)),g))
        return result

    def dot(self, left, right):
        value=0
        for (a,e),(b,f) in zip(left,right):
            value += (a*b) << (3222-e-f)
            self.tick(3,value)
        return Fraction(value,1<<3222)

    def action(self, left, matrix, right):
        require(matrix.format == 'csr' and matrix.nnz <= CAPS['fill'], 'exact CSR fill cap')
        value=0
        for row in range(matrix.shape[0]):
            first,last=map(int,matrix.indptr[row:row+2])
            require(last-first <= CAPS['row_terms'], 'exact row traversal cap')
            a,e=left[row]
            if not a: continue
            for index in range(first,last):
                coefficient=float(matrix.data[index])
                require(math.isfinite(coefficient), 'nonfinite exact coefficient')
                c,d=coefficient.as_integer_ratio(); f=d.bit_length()-1
                b,g=right[int(matrix.indices[index])]
                value += (a*c*b) << (3222-e-f-g)
                self.tick(4,value)
        return Fraction(value,1<<3222)


def rational_record(value):
    return {'numerator':str(value.numerator),'denominator':str(value.denominator),
        'float':float(value)}


def child(attempt):
    receipt=json.loads((attempt/'binding.json').read_text())
    require(receipt['input'] == static_binding(), 'child binding drift')
    import psutil
    psutil.Process().cpu_affinity([0])
    import numpy as np
    from scipy import sparse
    import scipy
    import numba
    from threadpoolctl import threadpool_info
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as panel
    from anyfesolver.linalg import MatrixClass, factorize_cached

    def runtime():
        value=dict(affinity=psutil.Process().cpu_affinity(),environment={k:os.environ.get(k) for k in ENV},
            pools=threadpool_info(), optimize=sys.flags.optimize, python=sys.version,
            executable=sys.executable, numpy=np.__version__,scipy=scipy.__version__,
            numba=dict(version=numba.__version__,disable_jit=int(numba.config.DISABLE_JIT),
                threads=numba.get_num_threads()), panel_path=panel.__file__)
        require(value['affinity']==[0] and value['environment']==ENV and value['optimize']==0
            and value['numba']['disable_jit']==1 and value['numba']['threads']==1
            and value['pools'] and all(p['num_threads']==1 for p in value['pools']),
            'effective runtime refusal')
        require(sha(panel.__file__)==receipt['input']['source_sha256'][SOURCES[1]],'imported panel drift')
        return value

    save(attempt/'runtime-before.json',runtime())
    result=dict(status='UNRESOLVED',stage='load',qualification=False,quantitative_entry=False)
    handle=None
    try:
        matrices={}
        with np.load(ARCHIVE,allow_pickle=False) as stored:
            for name in ('K','C','T'):
                M=sparse.csr_matrix((stored[name+'_data'],stored[name+'_indices'],stored[name+'_indptr']),
                    shape=tuple(stored[name+'_shape']))
                digest=hashlib.sha256()
                for a,dtype in ((M.shape,'<i8'),(M.indptr,'<i8'),(M.indices,'<i8'),(M.data,'<f8')):
                    digest.update(np.asarray(a,dtype=dtype).tobytes())
                require(digest.hexdigest()==receipt['input']['matrix_sha256'][name],name+' identity drift')
                require(M.nnz<=CAPS['fill'] and np.isfinite(M.data).all(),'source fill/finite refusal')
                matrices[name]=M
            x_saved=np.r_[stored['q'],stored['lambda_']]
            load=stored['load'].copy(); offset=stored['u0'].copy(); saved_u=stored['u'].copy()
        K,C,T=(matrices[k] for k in ('K','C','T'))
        require(K.shape==(67266,67266) and C.shape==(2000,67266) and T.shape==(67266,66660),
            'literal dimensions required')
        require(np.all(offset==0.) and np.all(np.diff(T.indptr)<=1) and np.all(T.data==1.)
            and len(np.unique(T.indices))==T.shape[1], 'literal zero-offset selector required')
        reduced_k=(T.T@K@T).tocsc(); reduced_c=(C@T).tocsc()
        A,scale=panel._scaled_point_kkt(reduced_k,reduced_c,None)
        require(A.nnz==1296304 and A.shape==(68660,68660),'same represented matrix layout required')
        del reduced_k,reduced_c
        result.update(stage='factor_prepared',represented_nnz=A.nnz,
            scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(),
            represented_sha256=hashlib.sha256(A.indptr.tobytes()+A.indices.tobytes()+A.data.tobytes()).hexdigest())
        owner=panel.ExperimentalB2Q4PanelAnalysis.__new__(panel.ExperimentalB2Q4PanelAnalysis)
        owner._K,owner._C,owner._T=K,C,T; owner._u0=offset; owner._Tt=T.T.tocsr()
        owner._projection_multi=np.flatnonzero(np.diff(owner._Tt.indptr)>1)
        owner._projection_single=np.flatnonzero(np.diff(owner._Tt.indptr)==1)
        owner._cu0=np.asarray(C@offset).reshape(-1);owner._factor_scale=scale
        owner._abs_C=abs(C);owner._physical_footprint_law=True
        owner.point_ids=();owner._point_trace_policy=None;owner._H=None
        owner._physical_singletons=owner._structural_physical_singletons(C,T,offset)
        x0=owner._restore_physical_singletons(x_saved)
        target=owner._physical_original_residual(x0,load)
        correction_rhs=-scale*target
        u0=np.asarray(T@x0[:-2000]).reshape(-1)+offset
        result.update(singleton_count=len(owner._physical_singletons),initial_restoration_count=int(np.count_nonzero(x0!=x_saved)),
            initial_reconstruction_diff_inf=float(np.max(np.abs(u0-saved_u))))
        save(attempt/'result.json',result)
        handle=factorize_cached(A,MatrixClass.SYMMETRIC_INDEFINITE,options={'backend':'scipy_superlu'})
        result['factor']=handle.diagnostics();save(attempt/'result.json',result)
        require(handle.status=='ok' and handle.backend_name=='scipy_superlu' and handle.ordering=='COLAMD'
            and handle.factorization_count==1 and handle.solve_count==0,'same factor refusal')
        require(result['factor']['factor_nnz']<=CAPS['factor_fill'],'factor fill refusal')
        dz=np.asarray(handle.solve(correction_rhs),dtype=float)
        require(np.isfinite(dz).all(),'direction nonfinite')
        delta=scale*dz; raw=x0+delta; x1=owner._restore_physical_singletons(raw)
        u1=np.asarray(T@x1[:-2000]).reshape(-1)+offset
        defect=A@dz-correction_rhs
        target1=owner._physical_original_residual(x1,load)
        np.savez(attempt/'direction.npz',x_saved=x_saved,x0=x0,delta_scaled=dz,delta=delta,
            raw_candidate=raw,candidate=x1,u0=u0,u1=u1,original_target=target,
            candidate_original_target=target1,scaled_rhs=correction_rhs,
            scaled_solve_defect=defect,original_solve_defect=defect/scale,
            gap0=C@u0,gap1=C@u1,scale=scale)
        result.update(stage='direction_captured',factor=handle.diagnostics(),
            solve_defect_inf=float(np.max(np.abs(defect))),
            final_restoration_count=int(np.count_nonzero(x1!=raw)))
        save(attempt/'result.json',result)
        exact=ExactWork(); f=exact.vector(load); a=exact.vector(u0);b=exact.vector(u1)
        v=exact.difference(b,a)
        W0=exact.action(a,K,a)-exact.dot(a,f)
        W1=exact.action(b,K,b)-exact.dot(b,f)
        quadratic=exact.action(v,K,v);linear=W1-W0-quadratic
        external0=exact.dot(a,f);external1=exact.dot(b,f)
        lam0=exact.vector(x0[-2000:]);lam1=exact.vector(x1[-2000:])
        gapwork0=exact.action(lam0,C,a);gapwork1=exact.action(lam1,C,b)
        projected0=exact.dot(exact.vector(x0[:-2000]),exact.vector(target[:-2000]))
        projected1=exact.dot(exact.vector(x1[:-2000]),exact.vector(target1[:-2000]))
        gapstored0=exact.dot(lam0,exact.vector(np.asarray(C@u0).reshape(-1)))
        gapstored1=exact.dot(lam1,exact.vector(np.asarray(C@u1).reshape(-1)))
        internal=float(u1@(K@u1)); external=float(u1@load);scored=internal-external
        require(all(map(math.isfinite,(internal,external,scored))),'scorer nonfinite')
        scorererror=Fraction.from_float(scored)-W1
        producttol=1e-12+1e-8*abs(external)
        relativeexact=Fraction.from_float(1e-8)*abs(external1)
        observed_product=abs(scored)<=producttol
        # Exact source work plus separately measured scorer deviation is a
        # sufficient bound for this actual stored candidate, not general FE
        # assembly/continuum error or an independently qualified tolerance.
        sufficient_original=abs(W1)+abs(scorererror)<=relativeexact
        result.update(stage='work_evaluated',status=('MISSING_DIRECTION_ENTRY_EVIDENCE' if sufficient_original else 'UNRESOLVED_ENTRY'),exact_operations=exact.operations,
            exact={k:rational_record(v) for k,v in dict(W0=W0,W1=W1,linear=linear,quadratic=quadratic,
                external0=external0,external1=external1,source_gap_work0=gapwork0,source_gap_work1=gapwork1,
                ordinary_projected_work0=projected0,ordinary_projected_work1=projected1,
                residual_contraction_error0=projected0-gapstored0-W0,
                residual_contraction_error1=projected1-gapstored1-W1,
                gap_evaluation_error0=gapstored0-gapwork0,gap_evaluation_error1=gapstored1-gapwork1,
                scorer_error=scorererror,original_relative_tolerance=relativeexact).items()},
            internal_work=internal,external_work=external,scored_work_defect=scored,
            product_tolerance=producttol,observed_product_guard_pass=observed_product,
            original_relative_sufficient_bound_pass=sufficient_original,
            quantitative_entry=sufficient_original,acceptance=False,reconstruction_exact=True,
            final_candidate_direction_includes_update_and_restoration=True,
            physical_assembly_uncertainty='not certified: frozen stored operator diagnostic only')
        if np.any(x1!=raw):
            uraw=exact.vector(np.asarray(T@raw[:-2000]).reshape(-1)+offset)
            Wraw=exact.action(uraw,K,uraw)-exact.dot(uraw,f)
            result['exact']['restoration_work_change']=rational_record(W1-Wraw)
        else: result['exact']['restoration_work_change']=rational_record(Fraction(0))
        result['exact_operations']=exact.operations
    except Exception as error:
        import traceback
        result.update(status='UNRESOLVED',quantitative_entry=False,error_type=type(error).__name__,error=str(error))
        (attempt/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        if handle is not None:
            result['factor']=handle.diagnostics()
            require(handle.factorization_count==1 and handle.solve_count<=1,'unexpected extra factor/solve')
            # This unchanged FactorizationHandle has no close method. The
            # supervised single-use process owns and releases its SuperLU data
            # at process exit; no shared cache or session survives this child.
        save(attempt/'result.json',result)
        save(attempt/'runtime-after.json',runtime())
    print(json.dumps({'status':result['status'],'stage':result['stage'],
        'quantitative_entry':result['quantitative_entry']}))
    return 0


def run():
    sys.path.insert(0,str(ROOT))
    from scripts._b3_e4_f03_local_scientific_supervisor import supervise,SupervisorLimits
    require(not (OUT/'direction-ledger.json').exists(),'one attempt only; ledger already exists')
    binding=static_binding()
    attempt=OUT/'direction-attempt-01';attempt.mkdir(exist_ok=False)
    env=dict(os.environ);env.update(ENV);env['PYTHONPATH']=os.pathsep.join((str(ROOT/'src'),str(ROOT)))
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',str(attempt)]
    record=dict(command=command,cwd=str(ROOT),input=binding,environment={k:env[k] for k in (*ENV,'PYTHONPATH')},
        authority='root explicit transfer 2026-10-10: ONE<=60s original-target missing-response diagnostic',
        qualification=False)
    save(attempt/'binding.json',record)
    for relative,expected in binding['source_sha256'].items():
        p=ROOT/relative;require(sha(p)==expected,'prelaunch source drift')
        dest=attempt/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    ledger=dict(status='OPEN',allowance_seconds=180.,reserve_seconds=60.,used_seconds=0.,attempts=1)
    save(OUT/'direction-ledger.json',ledger)
    with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr:
        receipt=supervise(command,cwd=ROOT,environment=env,stdout=stdout,stderr=stderr,
            limits=SupervisorLimits(60.,2*1024**3,.05))
    save(attempt/'process-tree-receipt.json',asdict(receipt))
    ledger.update(status='CLOSED',used_seconds=receipt.elapsed_seconds,
        discarded_unused_seconds=180.-receipt.elapsed_seconds,reason=receipt.reason,exit_code=receipt.exit_code)
    save(OUT/'direction-ledger.json',ledger)
    print(json.dumps(ledger))
    return 0 if receipt.reason=='completed' and receipt.exit_code==0 else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--child',type=Path)
    args=parser.parse_args()
    if args.child: sys.exit(child(args.child))
    if args.run:sys.exit(run())
    parser.error('explicit adopted --run or supervised --child required')
