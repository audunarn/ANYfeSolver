"""Private complete-pencil development oracle. Numerical dispatch requires root GO."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'.development/r8-f07-public-descriptor-20261009'
DOC = ROOT/'docs/agent_plans/development/r8-f07-public-descriptor-20261009'
EPS_FACTOR = 64.
GAP_FACTOR = 1000.


class Unresolved(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def need(condition, reason):
    if not condition:
        raise Unresolved(reason)


def homogeneous_valid(roots):
    import numpy as np
    need(np.isfinite(roots).all(), 'nonfinite homogeneous pair')
    need(np.all((abs(roots[0])+abs(roots[1])) > 0), 'indeterminate homogeneous 0/0 pair')


def rank_screen(matrix, out, name):
    import numpy as np
    from scipy.linalg import svd
    u, s, vh = svd(matrix, full_matrices=True)
    tau = EPS_FACTOR*np.finfo(float).eps*max(matrix.shape)*float(s[0] if len(s) else 0.)
    np.savez(out/(name+'.npz'), singular_values=s, tau=tau, right_vectors=vh, left_vectors=u)
    rank = int(np.count_nonzero(s > tau))
    need(rank == 0 or s[rank-1] >= GAP_FACTOR*tau, name+': unresolved numerical rank gap')
    return rank, vh[rank:].conj().T, u[:,rank:], tau


def zero_cluster(As, Bs, out):
    import numpy as np
    from scipy.linalg import svdvals
    rank, right, left, tau = rank_screen(As, out, 'zero-nullspace')
    cross = left.conj().T@Bs@right
    singular = svdvals(cross)
    np.savez(out/'zero-cross-overlap.npz', left=left, right=right, cross=cross, singular_values=singular)
    need(len(singular)>0, 'no zero cluster')
    overlap_tau=EPS_FACTOR*np.finfo(float).eps*len(As)*float(np.linalg.norm(Bs,2))
    need(singular[-1]>=GAP_FACTOR*overlap_tau, 'defective or unresolved zero cluster cross-overlap')
    delta=EPS_FACTOR*np.finfo(float).eps*len(As)*float(np.linalg.norm(As,2))/float(singular[-1])
    return rank,right,left,delta


def balance(A, B, out):
    import numpy as np
    need(np.isfinite(A).all() and np.isfinite(B).all(), 'nonfinite pencil')
    need(np.max(abs(A)) > 0 and np.max(abs(B)) > 0, 'empty pencil operator')
    frequency = float(np.exp2(np.rint(np.log2(np.max(abs(A))/np.max(abs(B))))))
    As = A.copy(); Bs = frequency*B; D = np.ones(len(A))
    for _ in range(12):
        row = np.max(abs(As)+abs(Bs), axis=1)
        need(np.all(row > 0), 'singular pencil has an empty row')
        d = np.exp2(np.rint(-.5*np.log2(row)))
        As = d[:, None]*As*d[None, :]; Bs = d[:, None]*Bs*d[None, :]; D *= d
        need(np.isfinite(As).all() and np.isfinite(Bs).all() and np.isfinite(D).all(), 'nonrepresentable balancing')
        need(np.array_equal(As != 0, A != 0) and np.array_equal(Bs != 0, B != 0), 'balancing lost an entry')
    np.savez(out/'pencil.npz', A=A, B=B, As=As, Bs=Bs, D=D, frequency=frequency)
    return As, Bs, D, frequency


def classify(A, B, rigid, out):
    """All roots plus independent rank chains; no production normalization."""
    import numpy as np
    from scipy.linalg import eig, cholesky, svdvals
    As, Bs, D, frequency = balance(A, B, out)
    N = len(A)
    need(N <= 256, 'bounded oracle dimension exceeded')
    roots, left, right = eig(As, Bs, left=True, right=True, homogeneous_eigvals=True)
    alpha, beta = roots
    np.savez(out/'homogeneous-qz.npz', alpha=alpha, beta=beta, left=left, right=right)
    homogeneous_valid(roots)
    n_prev = 0; d_prev = N+1; infinite = None; chains = []
    for k in range(1, 5):
        H = np.kron(np.eye(k), Bs)
        if k > 1:
            H += np.kron(np.eye(k, k, -1), -As)
        rank, _, _, tau = rank_screen(H, out, 'infinite-chain-'+str(k))
        nullity = k*N-rank; increment = nullity-n_prev
        chains.append(dict(k=k, nullity=nullity, increment=increment, tau=tau))
        write(out/'infinite-chains.json', chains)
        need(0 <= increment <= d_prev, 'inconsistent reversed-pencil chain ranks')
        if increment == 0:
            infinite = nullity; break
        n_prev = nullity; d_prev = increment
    need(infinite is not None, 'infinite chains not resolved within length four')
    projective = abs(beta)/np.hypot(abs(alpha), abs(beta))
    ordered = np.argsort(projective); ii = ordered[:infinite]; finite = ordered[infinite:]
    screen = EPS_FACTOR*np.finfo(float).eps*N
    write(out/'root-inventory-preliminary.json', dict(infinite_count=infinite, finite_count=len(finite), exact_beta_zero=int(np.count_nonzero(beta == 0)), projective=projective.tolist(), infinite_indices=ii.tolist(), finite_indices=finite.tolist(), screen=screen))
    need(not len(ii) or np.max(projective[ii]) <= screen, 'infinite QZ cluster unresolved')
    need(len(finite) > 6 and np.min(projective[finite]) >= GAP_FACTOR*screen, 'finite/infinite separation unresolved')
    rank, null, null_left, cluster_delta = zero_cluster(As, Bs, out)
    need(N-rank == 6, 'full-pencil zero nullity is not exactly six')
    null_gram = null.conj().T@Bs@null
    np.savez(out/'nullspace-kinetic.npz', gram=null_gram, vectors=null)
    try:
        lower = cholesky(null_gram, lower=True)
    except np.linalg.LinAlgError as exc:
        raise Unresolved('nonpositive rigid nullspace kinetic Gram') from exc
    Z = null@np.linalg.solve(lower.conj().T, np.eye(6))
    R = rigid/D[:, None]
    rg = R.conj().T@Bs@R
    lower = cholesky(rg, lower=True)
    W = R@np.linalg.solve(lower.conj().T, np.eye(6))
    overlap = svdvals(W.conj().T@Bs@Z)
    need(np.max(abs(overlap-1)) <= 1e-7, 'zero nullspace is not the physical rigid space')
    theta = alpha[finite]/beta[finite]
    normA = float(np.linalg.norm(As, 2)); normB = float(np.linalg.norm(Bs, 2))
    near = abs(theta) <= cluster_delta
    nonzero = abs(theta) >= GAP_FACTOR*cluster_delta
    deltas = []
    for j, t in zip(finite, theta):
        if abs(t)<=cluster_delta:
            deltas.append(cluster_delta);continue
        pairing = abs(left[:, j].conj()@Bs@right[:, j])
        need(pairing > 0, 'unresolved eigenpair sensitivity')
        deltas.append(screen*(normA+abs(t)*normB)*np.linalg.norm(left[:, j])*np.linalg.norm(right[:, j])/pairing)
    delta = np.asarray(deltas)
    nonzero &= abs(theta)>=GAP_FACTOR*delta
    write(out/'finite-inventory.json', dict(indices=finite.tolist(), theta_real=theta.real.tolist(), theta_imag=theta.imag.tolist(), cluster_delta=cluster_delta, sensitivity=delta.tolist(), nearzero=near.tolist(), resolved_nonzero=nonzero.tolist(), physical_lambda_real=(frequency*theta.real).tolist(), physical_lambda_imag=(frequency*theta.imag).tolist(), rigid_overlap=overlap.tolist()))
    need(np.isfinite(delta).all() and np.all(near | nonzero), 'finite nearzero separation unresolved')
    need(np.count_nonzero(near) == 6, 'finite QZ nearzero count is not exactly six')
    need(np.all(abs(theta.imag) <= np.where(near, delta, 1e-8*abs(theta.real))), 'nonreal finite root')
    need(np.all(theta.real[nonzero] > 0), 'negative resolved finite root')
    zero_vectors = right[:, finite[near]]
    zg = zero_vectors.conj().T@Bs@zero_vectors
    lower = cholesky(zg, lower=True)
    Zq = zero_vectors@np.linalg.solve(lower.conj().T, np.eye(6))
    qz_overlap = svdvals(W.conj().T@Bs@Zq)
    need(np.max(abs(qz_overlap-1)) <= 1e-7, 'QZ nearzero subspace is not physical rigid space')
    positive = finite[nonzero][np.argsort(theta.real[nonzero])]
    return dict(As=As, Bs=Bs, D=D, frequency=frequency, alpha=alpha, beta=beta, right=right, finite=finite, infinite=ii, near=near, first=positive[:2], rigid_overlap=overlap, qz_overlap=qz_overlap)


def canonical(owner):
    import numpy as np
    R = np.zeros((len(owner._free)+owner._C.shape[0], 6)); origin=owner._shell_xyz.mean(axis=0)
    need(np.array_equal(owner._free, np.arange(owner._K.shape[0])), 'fixture is not wholly free')
    for offset, xyz in ((0, owner._shell_xyz), (owner._shell_count, np.array([owner._beam_nodes[i][0] for i in owner._beam_ids]))):
        for i, position in enumerate(xyz):
            row=offset+6*i; R[row:row+3, :3]=np.eye(3)
            for j in range(3):
                R[row:row+3, 3+j]=np.cross(np.eye(3)[j], position-origin)
            R[row+3:row+6, 3:]=np.eye(3)
    for _, slots, _ in owner._beam_parts:
        for start in (18,21):
            R[owner._shell_count+slots[start:start+3], 3:]=np.eye(3)
    return R


def run_fixture(out):
    import numpy as np
    from scipy import sparse
    from anyfesolver import stiffened_shell as ss
    from examples.stiffened_shell_ge_mixed_reference_modes import definition
    owner=ss.create_analysis(definition=definition('mixed','biaxial','mixed','point',free=True))
    try:
        A=owner._A.toarray(); K=owner._K.copy(); C=owner._C.copy(); identity=owner._identity
        result=owner.modes(ss.GEMixedReferenceModesRequest(count=2))
        summary=dict(status=result.status, failure=result.failure_reason, identity=identity, policy=result.policy)
        write(out/'public-status.json', summary)
        need(result.status=='completed', 'public sparse caller failed: '+str(result.failure_reason))
        public=result.flexible; mass,zero,massid=owner._mixed_reference_modal_mass
        B=sparse.block_diag((mass[owner._free,:][:,owner._free], sparse.csr_matrix((C.shape[0],C.shape[0])))).toarray()
        R=canonical(owner)
        np.savez(out/'original-physical.npz', K=K.toarray(), C=C.toarray(), M=mass.toarray(), canonical=R, public_q=np.column_stack(public.full_coordinates), public_multipliers=np.column_stack(public.multiplier_modes), public_rigid=np.column_stack(result.rigid_coordinates), owner_coordinate_scale=owner._coordinate_scale, owner_constraint_scale=owner._constraint_scale)
        cells=[]; retained=[]
        for _, slots, _ in owner._beam_parts:
            cells.extend((owner._shell_count+slots[18:24]).tolist()); retained.extend((owner._shell_count+slots[24:]).tolist())
        nodal=[owner._shell_count+6*i+j for i in range(len(owner._beam_ids)) for j in (3,4,5)]
        spin=[6*i+5 for i in range(len(owner._shell_ids))]
        declared=sorted(set(nodal+spin+retained))
        block=dict(cells=cells,native_nodal_rotations=nodal,resultants=retained,shell_normal_spin=spin,multipliers=list(range(len(owner._free),len(A))),declared_zero=list(zero),mass_identity=massid,dimensions=dict(primal=mass.shape[0],constraints=C.shape[0],pencil=len(A)),mass_nnz=mass.nnz)
        write(out/'blocks.json',block)
        need(declared==list(zero), 'massless coordinate inventory mismatch')
        need(mass[:,zero].nnz==mass[zero,:].nnz==0 and np.count_nonzero(B[len(owner._free):,:])==0, 'invented algebraic mass')
        need(np.min(np.linalg.eigvalsh(mass[cells,:][:,cells].toarray()))>0,'native cell inertia missing')
        answer=classify(A,B,R,out)
        D=answer['D']; pairs=[]; physical=len(owner._free)
        for j in answer['first']:
            value=float((answer['frequency']*answer['alpha'][j]/answer['beta'][j]).real)
            x=D*answer['right'][:,j]
            need(np.linalg.norm(x.imag)<=1e-8*np.linalg.norm(x.real),'complex flexible vector')
            x=x.real; q=x[:physical]; lam=x[physical:]; kinetic=float(q@mass@q)
            need(kinetic>0,'nonpositive flexible modal mass'); x/=np.sqrt(kinetic);q=x[:physical];lam=x[physical:]
            residual=A@x-value*(B@x);bound=abs(A)@abs(x)+abs(value)*(abs(B)@abs(x))
            gap=C@q;gapbound=abs(C)@abs(q)
            equation=float(np.max(abs(residual)*D)/np.max(bound*D))
            constraint=float(np.max(abs(gap)*D[physical:])/np.max(gapbound*D[physical:]))
            np.savez(out/('pair-'+str(j)+'.npz'),x=x,q=q,multiplier=lam,residual=residual,bound=bound,gap=gap,gap_bound=gapbound,eigenvalue=value)
            pairs.append(dict(index=int(j),eigenvalue=value,equation=equation,constraint=constraint))
            write(out/'pairs.json',pairs)
            need(equation<=1e-8 and constraint<=1e-9,'oracle physical residual failed')
        public_values=np.array(public.eigenvalues_rad2_s2)
        errors=abs(np.array([p['eigenvalue'] for p in pairs])-public_values)/abs(public_values)
        summary.update(finite_count=len(answer['finite']),infinite_count=len(answer['infinite']),nearzero_count=int(np.count_nonzero(answer['near'])),pairs=pairs,public_eigenvalues=public_values.tolist(),relative_pair_error=errors.tolist(),public_equation=list(public.equation_residuals),public_physical_equation=list(public.physical_equation_residuals),public_constraint=list(public.constraint_residuals),rigid_equation=list(result.rigid_equation_residuals),rigid_constraint=list(result.rigid_constraint_residuals),rigid_flexible_overlap=result.rigid_flexible_mass_overlap)
        write(out/'comparison.json',summary)
        need(np.max(errors)<=1e-8,'public first flexible pair mismatch')
        need(max(public.equation_residuals+public.physical_equation_residuals)<=1e-8 and max(public.constraint_residuals)<=1e-9,'public residual guard failed')
        need(max(result.rigid_equation_residuals)<=1e-8 and max(result.rigid_constraint_residuals)<=1e-9 and result.rigid_flexible_mass_overlap<=1e-7,'public rigid guard failed')
        need(owner._identity==identity and owner._generation==0 and owner._accepted is owner._finite_driver is None and not owner._writing and (owner._K!=K).nnz==(owner._C!=C).nnz==0,'virgin owner mutated')
        summary['status']='PASS';write(out/'result.json',summary);return summary
    finally:
        owner.close()


def run_control(name,out):
    import numpy as np
    from scipy.linalg import block_diag
    if name=='defective-zero':
        A=np.array([[0.,1.],[0.,0.]]);B=np.eye(2)
        np.savez(out/'control-input.npz',A=A,B=B)
        try:
            zero_cluster(A,B,out)
        except Unresolved as exc:
            need('zero cluster cross-overlap' in str(exc), 'defective control refused at unintended gate: '+str(exc))
            write(out/'result.json',dict(status='PASS',expected_refusal=str(exc)));return
        raise AssertionError('defective zero cluster admitted')
    A=block_diag(np.diag([0.]*6+[1.,2.]),np.array([[0.,1.],[1.,0.]]));B=block_diag(np.eye(8),np.diag([1.,0.]));R=np.eye(10)[:,:6]
    if name=='seventh': A[6,6]=0
    elif name=='ambiguous': A[6,6]=1e-13
    elif name=='indeterminate': A[-1,:]=0;A[:,-1]=0;B[-1,:]=0;B[:,-1]=0
    elif name=='negative-rigid-mass': B[0,0]=-1
    np.savez(out/'control-input.npz',A=A,B=B,R=R)
    expected={'seventh':'zero nullity is not exactly six','ambiguous':'zero nullity is not exactly six',
        'indeterminate':'indeterminate homogeneous 0/0 pair','negative-rigid-mass':'nonpositive rigid nullspace kinetic Gram'}
    try:
        if name=='indeterminate':
            from scipy.linalg import eig
            roots=eig(A,B,homogeneous_eigvals=True,left=False,right=False)
            np.savez(out/'homogeneous-qz.npz',alpha=roots[0],beta=roots[1])
            homogeneous_valid(roots)
        else:answer=classify(A,B,R,out)
    except Unresolved as exc:
        need(name!='baseline','baseline control refused: '+str(exc))
        need(expected[name] in str(exc),'control refused at unintended gate: '+str(exc))
        write(out/'result.json',dict(status='PASS',expected_refusal=str(exc)));return
    need(name=='baseline','negative control unexpectedly admitted')
    need(len(answer['infinite'])==2 and len(answer['finite'])==8,'infinite-chain control count wrong')
    write(out/'result.json',dict(status='PASS',finite=8,infinite=2,nearzero=6))


def runtime(out, name, selected):
    import psutil, numba, threadpoolctl, importlib.metadata
    settings=dict(affinity=psutil.Process().cpu_affinity(), optimize=sys.flags.optimize,
        python=dict(executable=sys.executable,version=sys.version,base_executable=getattr(sys,'_base_executable',None)),
        dependencies={n:importlib.metadata.version(n) for n in ('numpy','scipy','numba','pytest','psutil','threadpoolctl')},
        numba=dict(disable_jit=bool(numba.config.DISABLE_JIT),num_threads=numba.get_num_threads()),
        pools=threadpoolctl.threadpool_info())
    imports={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
        if (n=='anyfesolver' or n.startswith('anyfesolver.')) and getattr(m,'__file__',None)}
    settings.update(imports=imports,import_sha256={n:sha(p) for n,p in imports.items()})
    write(out/(name+'.json'),settings)
    need(settings['affinity']==[0] and settings['optimize']==0 and settings['numba']==dict(disable_jit=True,num_threads=1) and all(p['num_threads']==1 for p in settings['pools']), 'effective runtime mismatch')
    for n,p in imports.items():
        need(Path(p).is_relative_to(selected/'anyfesolver'), 'unexpected anyfesolver import path: '+n)
        need(sha(p)==sha(ROOT/'src'/Path(p).relative_to(selected)), 'actual anyfesolver import hash mismatch: '+n)
    return settings


def prepare():
    """IO only: save literal caller/helper bytes and prospective bindings."""
    start=time.perf_counter_ns();HERE.mkdir(exist_ok=True,parents=True);DOC.mkdir(exist_ok=True,parents=True)
    output=DOC/'registration.json';need(not output.exists(),'registration already exists')
    bindings={}
    for rel in ('scripts/r8_f07_public_descriptor_dev.py','tests/test_r8_f07_public_descriptor_dev.py',
            'examples/stiffened_shell_ge_mixed_reference_modes.py','examples/stiffened_shell_native_fibre_bl5.py',
            'tests/test_stiffened_shell_ge_mixed_reference_modal.py'):
        path=ROOT/rel;bindings[rel]=sha(path);(DOC/('frozen-'+path.name)).write_bytes(path.read_bytes())
    source={p.relative_to(ROOT/'src').as_posix():sha(p) for p in (ROOT/'src/anyfesolver').rglob('*.py')}
    artifact=ROOT/'.development/r8-f13-dependency-repair-20261009/artifact-v2'
    saved=json.loads((artifact/'artifact-binding.json').read_text())
    need(all(sha(artifact/'site'/rel)==digest for rel,digest in source.items()),'source/site artifact mismatch')
    wheel=artifact/'dist/anysolver-0.4.7-py3-none-any.whl'
    need(sha(wheel)==saved['wheel_sha256'],'wheel changed')
    (DOC/'bound-artifact-manifest.json').write_bytes((artifact/'artifact-binding.json').read_bytes())
    env=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMBA_DISABLE_JIT='1',PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',NUMEXPR_NUM_THREADS='1')
    reg=dict(status='CANDIDATE_HOLD',root_go=False,python=str(Path(sys.executable).resolve()),python_sha256=sha(sys.executable),
        candidate_bindings=bindings,contract_sha256=sha(DOC/'TASK.md'),source_bindings=source,
        artifact_manifest_sha256=sha(artifact/'artifact-binding.json'),wheel=str(wheel),wheel_sha256=sha(wheel),
        source_site=str(ROOT/'src'),installed_site=str(artifact/'site'),environment=env,
        budget_s=240.,reserve_s=60.,stage_cap_s=60.,memory_cap_bytes=2147483648,
        cases=dict(source=['baseline','seventh','ambiguous','indeterminate','negative-rigid-mass','defective-zero','current-public-mixed-point'],installed=['current-public-mixed-point']),
        classification=dict(eps_factor=EPS_FACTOR,gap_factor=GAP_FACTOR,max_chain_length=4,max_pencil_dimension=256),
        commands={stage:[str(Path(sys.executable).resolve()),str(Path(__file__).resolve()),'--child',stage,'--site',str(ROOT/'src' if stage=='source' else artifact/'site'),'--out',str(HERE/(stage+'1'))] for stage in ('source','installed')})
    (DOC/'frozen-TASK.md').write_bytes((DOC/'TASK.md').read_bytes())
    write(output,reg);write(DOC/'prepare-io.json',dict(wall_s=(time.perf_counter_ns()-start)/1e9,scope='IO hashes/copies only; no solver or oracle execution'))


def child(stage,site,out):
    import traceback
    selected=Path(site).resolve();out=Path(out).resolve()
    reg=json.loads((DOC/'registration.json').read_text())
    need(reg['root_go'] is True and reg['status']=='FROZEN_BY_ROOT','child has no frozen root GO')
    need(os.environ.get('F07_SUPERVISED')==sha(DOC/'registration.json'),'child requires supervisor-only launch marker')
    need(stage in ('source','installed') and selected==Path(reg[stage+'_site']).resolve(),'child site/stage mismatch')
    need(str(out)==reg['commands'][stage][-1],'child output differs from registered command')
    need(sha(sys.executable)==reg['python_sha256'] and sha(DOC/'TASK.md')==reg['contract_sha256'],'child Python/contract mismatch')
    for rel,digest in reg['candidate_bindings'].items():need(sha(ROOT/rel)==digest,'child frozen candidate changed: '+rel)
    for rel,digest in reg['source_bindings'].items():
        need(sha(ROOT/'src'/rel)==sha(selected/rel)==digest,'child source/site changed: '+rel)
    ledger=json.loads((HERE/'ledger.json').read_text());launch=json.loads((out/'launch.json').read_text())
    need(ledger['status']=='OPEN' and launch['status']=='launching' and launch['stage']==stage
        and launch['registration_sha256']==sha(DOC/'registration.json'),'child requires active supervised launch')
    sys.path[:0]=[str(selected),str(ROOT)]
    os.environ['F07_OUTPUT']=str(out)
    import psutil
    psutil.Process().cpu_affinity([0])
    runtime(out,'runtime-before',selected)
    import pytest
    test=str(ROOT/'tests/test_r8_f07_public_descriptor_dev.py')
    if stage=='installed':test+='::test_current_public_complete_pencil'
    code=None
    try:
        code=int(pytest.main([test,'-o','pythonpath=','-q','-x','-p','no:cacheprovider','--basetemp',str(out/'pytest-tmp')]))
        return code
    except BaseException as exc:
        write(out/'child-failure.json',dict(exception=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        runtime(out,'runtime-after',selected)
        write(out/'child-status.json',dict(exit_code=code,stage=stage,explicit_output=os.environ['F07_OUTPUT']))


def supervise(stage):
    import psutil,subprocess
    reg=json.loads((DOC/'registration.json').read_text())
    need(reg['root_go'] is True and reg['status']=='FROZEN_BY_ROOT','numerical execution not adopted/frozen')
    need(stage in ('source','installed'),'unknown stage')
    need(sha(sys.executable)==reg['python_sha256'] and str(Path(sys.executable).resolve())==reg['python'],'Python identity changed')
    need(sha(DOC/'TASK.md')==reg['contract_sha256'],'contract changed')
    for rel,digest in reg['candidate_bindings'].items():need(sha(ROOT/rel)==digest,'frozen candidate changed: '+rel)
    for rel,digest in reg['source_bindings'].items():need(sha(ROOT/'src'/rel)==digest,'source product changed: '+rel)
    lp=HERE/'ledger.json';ledger=json.loads(lp.read_text()) if lp.exists() else dict(status='OPEN',budget_s=reg['budget_s'],reserve_s=reg['reserve_s'],used_s=0.,attempts=[])
    need(ledger['status']=='OPEN','closed ledger')
    if stage=='installed':need(len(ledger['attempts'])==1 and ledger['attempts'][0]['stage']=='source' and ledger['attempts'][0]['status']=='PASS','source gate not passed once')
    else:need(not ledger['attempts'],'source already attempted')
    cap=min(reg['stage_cap_s'],reg['budget_s']-reg['reserve_s']-ledger['used_s']);need(cap>0,'budget/reserve exhausted')
    selected=Path(reg[stage+'_site']);need(all(sha(selected/rel)==digest for rel,digest in reg['source_bindings'].items()),'selected site changed')
    if stage=='installed':need(sha(reg['wheel'])==reg['wheel_sha256'],'wheel changed')
    out=HERE/(stage+'1');out.mkdir()
    snapshots=out/'source-snapshots';snapshots.mkdir()
    for rel in reg['candidate_bindings']:(snapshots/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    (snapshots/'registration.json').write_bytes((DOC/'registration.json').read_bytes())
    (snapshots/'TASK.md').write_bytes((DOC/'TASK.md').read_bytes())
    command=reg['commands'][stage];env=os.environ.copy();env.update(reg['environment'])
    env['F07_SUPERVISED']=sha(DOC/'registration.json')
    receipt=dict(status='launching',stage=stage,command=command,registration_sha256=sha(DOC/'registration.json'),environment=reg['environment'],selected_site=str(selected),cap_s=cap,prior_used_s=ledger['used_s'],memory_cap_bytes=reg['memory_cap_bytes'],peak_tree_rss_bytes=0)
    write(out/'launch.json',receipt);write(lp,ledger)
    start=time.perf_counter_ns();proc=None;code=None;failure=None;error=None
    try:
        with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
            psutil.Process().cpu_affinity([0])
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr);process=psutil.Process(proc.pid);process.cpu_affinity([0])
            while proc.poll() is None:
                try:members=[process]+process.children(recursive=True)
                except psutil.NoSuchProcess:members=[]
                rss=0
                for member in members:
                    try:rss+=member.memory_info().rss
                    except psutil.NoSuchProcess:pass
                receipt['peak_tree_rss_bytes']=max(receipt['peak_tree_rss_bytes'],rss)
                if rss>reg['memory_cap_bytes']:failure='MEMORY_LIMIT'
                elif (time.perf_counter_ns()-start)/1e9>=cap:failure='TIMEOUT'
                if failure:
                    for member in reversed(members):
                        try:member.kill()
                        except psutil.NoSuchProcess:pass
                    break
                time.sleep(.01)
            code=proc.wait()
    except BaseException as exc:
        error=exc;failure='LAUNCH_OR_SUPERVISION_FAILURE';receipt['exception']=repr(exc)
        if proc is not None and proc.poll() is None:
            try:members=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:members=[]
            for member in reversed(members):
                try:member.kill()
                except psutil.NoSuchProcess:pass
            code=proc.wait()
    finally:
        wall=(time.perf_counter_ns()-start)/1e9;ledger['used_s']+=wall
        receipt.update(status='PASS' if code==0 and failure is None else 'FAIL',exit_code=code,failure=failure,wall_s=wall,cumulative_wall_s=ledger['used_s'],remaining_s=reg['budget_s']-ledger['used_s'])
        ledger['attempts'].append(dict(stage=stage,status=receipt['status'],wall_s=wall))
        if stage=='installed' or receipt['status']=='FAIL':ledger.update(status='CLOSED',unused_discarded_s=reg['budget_s']-ledger['used_s'])
        write(out/'receipt.json',receipt);write(lp,ledger)
    print(json.dumps({k:receipt[k] for k in ('status','wall_s','cumulative_wall_s','remaining_s')}))
    if error:raise error
    return int(receipt['status']!='PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--child',choices=('source','installed'));parser.add_argument('--run',choices=('source','installed'));parser.add_argument('--site');parser.add_argument('--out');args=parser.parse_args()
    if args.prepare:prepare()
    else:raise SystemExit(child(args.child,args.site,args.out) if args.child else supervise(args.run))
