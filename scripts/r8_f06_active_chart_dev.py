"""Bounded material-active same-physical-variation shell chart seam."""
from __future__ import annotations
import argparse,dataclasses,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'.development/r8-f06-active-chart-20261009'
DOC=ROOT/'docs/agent_plans/development/r8-f06-active-chart-20261009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def physical_to_chart(state, physical, shell_count):
    from anyfesolver._ge_beam3_pose_joint import _exp_terms
    result=physical.copy()
    for start in range(3,shell_count,6):
        result[start:start+3]=__import__('numpy').linalg.solve(_exp_terms(state.q[start:start+3])[1],physical[start:start+3])
    return result

def branch(shell,beams,shell_origins,beam_origins):
    import numpy as np
    from math import fsum
    increments=[];mask=[]
    for candidate,origin in zip(shell,shell_origins,strict=True):
        delta=np.asarray(candidate['alpha'])-np.asarray(origin['alpha'])
        allowance=64*np.finfo(float).eps*np.maximum(np.abs(candidate['alpha']),np.abs(origin['alpha']))
        increments.extend(delta.tolist());mask.extend((delta>allowance).tolist())
    native=[]
    for candidate,origin in zip(beams,beam_origins,strict=True):
        for current,old in zip(candidate.stations,origin.stations,strict=True):
            new=fsum(current.accumulated);previous=fsum(old.accumulated)
            delta=fsum((*current.accumulated,*(-v for v in old.accumulated)))
            native.append(delta);mask.append(bool(delta>64*np.finfo(float).eps*max(abs(new),abs(previous))))
    return dict(mask=mask,shell_increments=increments,native_increments=native,shell_active=sum(v>0 for v in mask[:len(increments)]),native_active=sum(v>0 for v in mask[len(increments):]))

def run_case(family,out):
    import numpy as np,psutil,numba,threadpoolctl
    from scipy.sparse.linalg import spsolve
    from scipy.spatial.transform import Rotation
    from anymaterial import DNVC208MaterialCurve
    from anyfesolver import stiffened_shell as ss
    from anyfesolver._ge_beam3_pose_joint import _exp_terms
    from anyfesolver._stiffened_shell_rotation_charts import physical_frames
    from anyfesolver.control import CancellationToken,SolveCancelled
    from test_stiffened_shell_q4_pullback_owner import _definition
    from test_stiffened_shell_ge_finite_experimental import _actions,_q4_shell_j2_panel
    runtime_before=dict(affinity=psutil.Process().cpu_affinity(),optimize=sys.flags.optimize,numba=dict(disable_jit=bool(numba.config.DISABLE_JIT),num_threads=numba.get_num_threads()),pools=threadpoolctl.threadpool_info())
    write(out/'runtime-before.json',runtime_before)
    assert runtime_before['affinity']==[0] and runtime_before['optimize']==0 and runtime_before['numba']==dict(disable_jit=True,num_threads=1) and all(p['num_threads']==1 for p in runtime_before['pools'])
    definition=_definition(plastic=True)
    if family=='s3':
        definition=_q4_shell_j2_panel()
        pressure=20000.;curve=DNVC208MaterialCurve(pressure,1.05*pressure,1.1*pressure,.005,.01,4*pressure,.2)
        definition=dataclasses.replace(definition,shell=dataclasses.replace(definition.shell,q4_elements=(),s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)),j2_curve=curve))
    owner=ss.create_analysis(definition=definition)
    factors,next_factor=((.5,1.),1.1) if family=='q4' else ((.05,),.06)
    cache=Path(os.environ.get('F06_PREFIX_CACHE','__absent__'))/family/'accepted-before.bin'
    if cache.exists():
        result=owner.restore(cache.read_bytes())
        assert tuple(result.accepted_factors)==factors
        write(out/'prefix-reuse.json',dict(path=str(cache.resolve()),sha256=sha(cache),identity_verified_by_restore=True))
    else:result=owner.solve(_actions(),factors,max_iterations=20,max_backtracks=12)
    assert result.status=='completed' and result.constraint_max<=1e-9 and result.free_equilibrium_ratio<=1e-8 and max(max(row) for row in result.shell_layer_alpha)>0
    driver=owner._finite_driver;accepted=driver._state;origins=(driver._shell_origins,driver._beam_origins)
    raw=owner.checkpoint();(out/'accepted-before.bin').write_bytes(raw)
    args=(next_factor,*origins)
    residual,tangent,*_=driver._assemble(accepted,*args)
    count=owner._K.shape[0];free=np.r_[owner._free,np.arange(count,tangent.shape[0])]
    # Governing owner coordinate scales make every mechanical row a work
    # coordinate; compatibility rows retain separate length/radian scaling.
    rowscale=np.r_[owner._coordinate_scale/2000.,1./owner._constraint_scale]
    nodal_end=owner._shell_count+driver._beam_nodal
    groups=dict(shell=free[free<owner._shell_count],native_nodal=free[(free>=owner._shell_count)&(free<nodal_end)],native_retained=free[(free>=nodal_end)&(free<count)],interface=np.arange(count,tangent.shape[0]))
    chart_direction=np.zeros(tangent.shape[0]);chart_direction[free]=spsolve(tangent[free,:][:,free],-residual[free])
    physical=chart_direction.copy()
    for start in range(3,owner._shell_count,6):physical[start:start+3]=_exp_terms(accepted.q[start:start+3])[1]@chart_direction[start:start+3]
    coverage={name:dict(nonzero_direction_count=int(np.count_nonzero(physical[indices])),coordinate_count=len(indices)) for name,indices in groups.items()}
    write(out/'coverage.json',coverage)
    assert coverage['native_retained']['nonzero_direction_count']>0 and coverage['interface']['nonzero_direction_count']>0
    before=driver._advance(accepted,chart_direction,.37)
    assert owner.checkpoint()==raw
    rebased=owner.rebase_rotations(result.generation)
    assert rebased.constraint_max<=1e-9 and rebased.free_equilibrium_ratio<=1e-8
    assert driver._shell_origins is origins[0] and driver._beam_origins is origins[1]
    anchor_state=driver._state;raw_after=owner.checkpoint();(out/'accepted-after.bin').write_bytes(raw_after)
    after=driver._advance(anchor_state,chart_direction,.37)
    q=after.q.copy();pose=physical_frames(before.q[:owner._shell_count],before.shell_anchors)
    for node in range(len(owner._shell_ids)):
        start=6*node+3;q[start:start+3]=Rotation.from_matrix(pose[node]@anchor_state.shell_anchors[node].T).as_rotvec()
    after=dataclasses.replace(after,q=q)
    np.testing.assert_allclose(physical_frames(after.q[:owner._shell_count],after.shell_anchors),pose,rtol=0,atol=1e-12)
    for field in ('frames','cells','resultants','multipliers'):np.testing.assert_array_equal(getattr(before,field),getattr(after,field))
    if family=='s3' and os.environ.get('F06_PROBE')=='resolved-resultant-multiplier':
        pattern=.1*np.array((1.,-.7,.5,.6,-.45,.2))*np.array((1000.,1000.,1000.,2000.,2000.,2000.))
        for base in range(nodal_end,count,24):physical[base+6:base+24]+=np.tile(pattern,3)
        physical[count:]+=np.tile(pattern,(len(physical)-count)//6)
        write(out/'resolved-probe.json',dict(pattern=pattern.tolist(),retained_columns=list(range(nodal_end,count)),multiplier_columns=list(range(count,len(physical))),trial_unchanged=True))
    observations={};residuals={};derivatives={}
    for side,state in (('before',before),('after',after)):
        r,k,relative,gap,shell,beam=driver._assemble(state,*args)
        activity=branch(shell,beam,*origins)
        write(out/(side+'-branch.json'),activity)
        assert activity['shell_active']>0,'no active shell branch at trial'
        direction=physical_to_chart(state,physical,owner._shell_count);predicted=k@direction
        np.savez(out/(side+'-operators.npz'),physical_direction=physical,chart_direction=direction,residual=r,predicted=predicted,q=state.q,rowscale=rowscale,scaled_gap=r[count:]/owner._constraint_scale)
        samples=[]
        for h in (.002,.001):
            plus=driver._assemble(driver._advance(state,direction,h),*args)
            minus=driver._assemble(driver._advance(state,direction,-h),*args)
            plus_branch=branch(plus[4],plus[5],*origins);minus_branch=branch(minus[4],minus[5],*origins)
            if plus_branch['mask']==activity['mask']==minus_branch['mask']:
                observed=(plus[0]-minus[0])/(2*h);scheme='central smooth-active'
            elif plus_branch['mask']==activity['mask']:
                observed=(plus[0]-r)/h;scheme='forward active transition'
            elif minus_branch['mask']==activity['mask']:
                observed=(r-minus[0])/h;scheme='backward active transition'
            else:raise AssertionError('no resolved one-sided branch at registered step')
            np.savez(out/(side+'-fd-'+str(h)+'.npz'),base=r,plus=plus[0],minus=minus[0],observed=observed,predicted=predicted,error=observed-predicted,scaled_error=(observed-predicted)*rowscale,rowscale=rowscale)
            errors={name:float(np.linalg.norm(((observed-predicted)*rowscale)[indices])/max(np.linalg.norm((predicted*rowscale)[indices]),1e-10)) for name,indices in groups.items() if name!='interface'}
            error=max(errors.values())
            interface=float(np.max(np.abs((observed-predicted)[count:])/owner._constraint_scale))
            sample=dict(h=h,scheme=scheme,relative_error=error,block_errors=errors,interface_error=interface)
            samples.append(sample);write(out/(side+'-samples.json'),samples)
            assert error<=1e-5,(family,side,sample)
            if activity['native_active']==0:assert max(errors['native_nodal'],errors['native_retained'])<=1e-6,(family,side,sample)
            assert interface<=1e-9,(family,side,sample)
            assert owner.checkpoint()==raw_after
        residuals[side]=r;derivatives[side]=predicted
        observations[side]=dict(branch=activity,derivative_samples=samples,gap=gap,physical_virtual_work=float(r@physical),trial_relative_equilibrium=relative)
    left,right=(observations[s]['physical_virtual_work'] for s in ('before','after'))
    work_error=abs(left-right)/max(abs(left),abs(right),1e-12)
    tangent_errors={name:float(np.linalg.norm(((derivatives['before']-derivatives['after'])*rowscale)[indices])/max(np.linalg.norm((derivatives['before']*rowscale)[indices]),np.linalg.norm((derivatives['after']*rowscale)[indices]),1e-10)) for name,indices in groups.items()}
    tangent_error=max(tangent_errors.values())
    gap_invariance=float(np.max(np.abs(residuals['before'][count:]-residuals['after'][count:])/owner._constraint_scale))
    write(out/'paired-metrics.json',dict(work_error=work_error,tangent_block_errors=tangent_errors,gap_invariance=gap_invariance))
    assert work_error<=1e-8 and tangent_error<=1e-5 and gap_invariance<=1e-9,(work_error,tangent_error,gap_invariance)
    assert owner.checkpoint()==raw_after
    token=CancellationToken();token.cancel('registered active-chart trial refusal')
    try:owner.solve(_actions(),(next_factor,),max_iterations=20,max_backtracks=12,cancellation_token=token)
    except SolveCancelled:pass
    else:raise AssertionError('pre-cancel rebase unexpectedly accepted')
    assert owner.checkpoint()==raw_after
    imports={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n.startswith('anyfesolver') and getattr(m,'__file__',None)}
    site=Path(os.environ['F06_SELECTED_SITE']).resolve()
    assert all(Path(path).is_relative_to(site) for path in imports.values())
    for path in imports.values():
        rel=Path(path).relative_to(site);assert sha(path)==sha(ROOT/'src'/rel)
    runtime=dict(affinity=psutil.Process().cpu_affinity(),optimize=sys.flags.optimize,numba=dict(disable_jit=bool(numba.config.DISABLE_JIT),num_threads=numba.get_num_threads()),pools=threadpoolctl.threadpool_info(),imports=imports,import_sha256={n:sha(p) for n,p in imports.items()})
    write(out/'runtime.json',runtime)
    assert runtime['affinity']==[0] and runtime['optimize']==0 and runtime['numba']==dict(disable_jit=True,num_threads=1) and all(p['num_threads']==1 for p in runtime['pools'])
    return dict(status='PASS',family=family,policy=definition.q4_pullback_policy,prefix=list(factors),target=next_factor,accepted_predecessor_preserved=True,work_error=work_error,gap_invariance=gap_invariance,same_physical_tangent_error=tangent_error,tangent_block_errors=tangent_errors,coverage=coverage,observations=observations)

def child(family,site,out):
    out=Path(out);sys.path[:0]=[str(Path(site).resolve()),str(ROOT/'tests'),str(ROOT)]
    os.environ['F06_SELECTED_SITE']=str(Path(site).resolve())
    if family=='all':
        os.environ['F06_OUTPUT']=str(out)
        import pytest
        return int(pytest.main([str(ROOT/'tests/test_r8_f06_active_chart_dev.py'),'-o','pythonpath=','-q','-x','-p','no:cacheprovider','--basetemp',str(out/'pytest-tmp')]))
    caseout=out/family;caseout.mkdir()
    try:result=run_case(family,caseout);write(caseout/'result.json',result);print(json.dumps(dict(status=result['status'],family=family,work_error=result['work_error'],tangent_error=result['same_physical_tangent_error'])));return 0
    except BaseException as exc:
        import traceback
        write(out/'failure.json',dict(exception=repr(exc),traceback=traceback.format_exc()));raise

def supervise(family,site,name,cache=None,probe=None):
    import psutil,subprocess
    HERE.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    lp=HERE/'ledger.json';ledger=json.loads(lp.read_text()) if lp.exists() else dict(status='OPEN',budget_s=240.,reserve_s=60.,used_s=0.,attempts=[])
    if ledger['status']!='OPEN':raise RuntimeError('closed ledger')
    cap=min(60.,240.-60.-ledger['used_s'])
    if cap<=0:raise RuntimeError('budget/reserve exhausted')
    out=HERE/name;out.mkdir();command=[sys.executable,str(Path(__file__).resolve()),'--child',family,'--site',str(Path(site).resolve()),'--out',str(out)]
    env=os.environ.copy();settings=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMBA_DISABLE_JIT='1',PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',NUMEXPR_NUM_THREADS='1');env.update(settings)
    if cache:env['F06_PREFIX_CACHE']=str(Path(cache).resolve())
    if probe:env['F06_PROBE']=probe
    source_hashes={p.relative_to(ROOT/'src').as_posix():sha(p) for p in (ROOT/'src/anyfesolver').rglob('*.py')}
    snapshots=out/'source-snapshots';snapshots.mkdir();(snapshots/'runner.py').write_bytes(Path(__file__).read_bytes())
    fixtures={}
    for rel in ('tests/test_r8_f06_active_chart_dev.py','tests/test_stiffened_shell_q4_pullback_owner.py','tests/test_stiffened_shell_ge_finite_experimental.py','tests/test_stiffened_shell_ge_experimental.py'):
        fixtures[rel]=sha(ROOT/rel);(snapshots/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    selected=Path(site).resolve()
    assert all(sha(selected/rel)==digest for rel,digest in source_hashes.items())
    receipt=dict(status='launching',command=command,family=family,probe=probe,site=str(selected),script_sha256=sha(__file__),source_hashes=source_hashes,fixture_hashes=fixtures,cache=str(Path(cache).resolve()) if cache else None,cache_sha256={p.relative_to(Path(cache)).as_posix():sha(p) for p in Path(cache).rglob('accepted-before.bin')} if cache else {},environment=settings,cap_s=cap,prior_used_s=ledger['used_s'],memory_cap_bytes=2147483648,peak_tree_rss_bytes=0)
    write(out/'registration.json',receipt);write(DOC/(name+'-registration.json'),receipt)
    if selected!= (ROOT/'src').resolve():receipt['wheel_sha256']=sha(selected.parent/'dist/anysolver-0.4.7-py3-none-any.whl')
    psutil.Process().cpu_affinity([0]);start=time.perf_counter_ns();failure=None;proc=None;code=None;error=None
    try:
        with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr);process=psutil.Process(proc.pid);process.cpu_affinity([0])
            while proc.poll() is None:
                try:members=[process]+process.children(recursive=True)
                except psutil.NoSuchProcess:members=[]
                rss=0
                for member in members:
                    try:rss+=member.memory_info().rss
                    except psutil.NoSuchProcess:pass
                receipt['peak_tree_rss_bytes']=max(receipt['peak_tree_rss_bytes'],rss)
                if rss>2147483648:failure='MEMORY_LIMIT'
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
            for member in reversed([psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)):
                try:member.kill()
                except psutil.NoSuchProcess:pass
            code=proc.wait()
    finally:
        wall=(time.perf_counter_ns()-start)/1e9;ledger['used_s']+=wall
        receipt.update(status='PASS' if code==0 and failure is None else 'FAIL',exit_code=code,failure=failure,wall_s=wall,cumulative_wall_s=ledger['used_s'],remaining_s=240.-ledger['used_s'])
        ledger['attempts'].append(dict(name=name,status=receipt['status'],wall_s=wall))
        if selected!=(ROOT/'src').resolve():ledger.update(status='CLOSED',unused_discarded_s=240.-ledger['used_s'])
        write(lp,ledger);write(out/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','wall_s','cumulative_wall_s','remaining_s')}))
    if error:raise error
    return int(receipt['status']!='PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--run');p.add_argument('--site',required=True);p.add_argument('--out');p.add_argument('--name');p.add_argument('--prefix-cache');p.add_argument('--probe',choices=('resolved-resultant-multiplier',));a=p.parse_args()
    raise SystemExit(child(a.child,a.site,a.out) if a.child else supervise(a.run,a.site,a.name,a.prefix_cache,a.probe))
