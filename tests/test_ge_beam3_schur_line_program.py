"""New bounded path/transaction comparisons; historical paths remain untouched."""
import json
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver import _ge_beam3_schur_line_program as native
from anyfesolver import _ge_beam3_fibre_line_program as dense
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken
from test_ge_beam3_fibre_line_program import model,line


def inputs(case):
    if case=='straight-elastic': return model(),ForceProgram((.2,.6,1.,.35,0.),(),max_iterations=24),line((.12,-.04,.02))
    if case=='curved-elastic': return model(curved=True),ForceProgram((.2,.6,1.,.35,0.),(),max_iterations=24),line((.11,-.035,.025))
    if case=='plastic-cycle': return model(plastic=True),ForceProgram((.2,.55,1.,.4,0.,-.4,0.),(),max_iterations=24),line((.82,.025,-.015))
    raise ValueError('registered development case')


def save(path,value):
    raw=value if type(value) is bytes else canonical(value)
    with path.open('xb') as stream: stream.write(raw)


def numerical_difference(a,b):
    """Compare physical fields recursively; strings/IDs/branches must agree."""
    a=json.loads(canonical(a)); b=json.loads(canonical(b)); maximum=0.
    def visit(x,y):
        nonlocal maximum
        if type(x) in (int,float) and type(y) in (int,float):
            maximum=max(maximum,abs(x-y)/max(1.,abs(x),abs(y))); return
        assert type(x) is type(y)
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x: visit(x[key],y[key])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for left,right in zip(x,y): visit(left,right)
        else: assert x==y
    visit(a,b)
    assert maximum<=1e-11
    return maximum


@pytest.fixture(scope='module',params=['straight-elastic','curved-elastic','plastic-cycle'])
def path_pair(request,tmp_path_factory):
    root=tmp_path_factory.mktemp(request.param); outputs=[]; events=[]
    for module,label in ((dense,'dense'),(native,'schur')):
        made,p,f=inputs(request.param)
        result=module.solve_force_program(made,p,line_forces=f,progress=events.append if label=='schur' else None)
        save(root/(label+'.json'),result.checkpoint)
        save(root/(label+'-status.json'),dict(status=result.status,failure=result.failure,completed_targets=result.completed_targets))
        assert result.status=='completed',result.failure
        outputs.append(result)
    save(root/'progress.json',events)
    return request.param,root,outputs,events


def test_complete_path_recovery_origins_and_sparse_factor_use(path_pair):
    case,root,(old,new),events=path_pair
    a=json.loads(old.checkpoint); b=json.loads(new.checkpoint)
    assert a['model_sha256']!=b['model_sha256']
    assert a['program']['schema']!=b['program']['schema']
    assert len(a['records'])==len(b['records'])==new.completed_targets
    errors=[]
    for x,y in zip(a['records'],b['records']):
        assert x['target']==y['target'] and x['parameter']==y['parameter']
        row={key:numerical_difference(x[key],y[key]) for key in ('mechanical','origins','histories','residual')}
        assert max(x['metrics']+y['metrics'])<=1e-11
        errors.append(row)
    c,p,f=inputs(case); nc=native.Context(c,p,line_forces=f); state,records=nc.restore(new.checkpoint)
    c,p,f=inputs(case); dc=dense.Context(c,p,line_forces=f); state0,_=dc.restore(old.checkpoint)
    recovery_error=numerical_difference(nc.recover(state),dc.recover(state0))
    assert nc.checkpoint(records)==new.checkpoint
    factors=[e['linear_solver'] for e in events if e['stage']=='fibre-schur-line.factorized']
    assert factors and all(e['backend']=='scipy_superlu' and e['factorizations'] in (1,2) for e in factors)
    assert all(e['eliminated_force_coordinates']==18 and e['retained_physical_cell_rotations']==6 for e in factors)
    if case=='plastic-cycle': assert any(r[2]>0 for c in state.histories for s in c.stations for r in s.rows)
    save(root/'comparison.json',dict(case=case,records=errors,recovery_error=recovery_error,
        dense_checkpoint_sha256=sha256(old.checkpoint).hexdigest(),schur_checkpoint_sha256=sha256(new.checkpoint).hexdigest(),
        accepted_targets=new.completed_targets,production_qualified=False))


def test_stop_resume_exact_checkpoint_and_cross_backend_rejection(path_pair):
    case,root,(old,whole),_=path_pair
    made,p,f=inputs(case)
    paused=native.solve_force_program(made,p,line_forces=f,stop_after=2)
    assert paused.status=='paused'
    made,p,f=inputs(case)
    resumed=native.solve_force_program(made,p,line_forces=f,checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    save(root/'paused.json',paused.checkpoint); save(root/'resumed.json',resumed.checkpoint)
    assert resumed.status=='completed' and resumed.checkpoint==whole.checkpoint
    made,p,f=inputs(case)
    old_paused=dense.solve_force_program(made,p,line_forces=f,stop_after=2)
    made,p,f=inputs(case)
    old_resumed=dense.solve_force_program(made,p,line_forces=f,checkpoint=old_paused.checkpoint,
        expected_checkpoint_sha256=sha256(old_paused.checkpoint).hexdigest())
    save(root/'dense-paused.json',old_paused.checkpoint); save(root/'dense-resumed.json',old_resumed.checkpoint)
    assert old_paused.status=='paused' and old_resumed.status=='completed' and old_resumed.checkpoint==old.checkpoint
    made,p,f=inputs(case)
    with pytest.raises(ValueError,match='binding'): native.Context(made,p,line_forces=f).restore(old.checkpoint)
    made,p,f=inputs(case)
    with pytest.raises(ValueError,match='binding'): dense.Context(made,p,line_forces=f).restore(whole.checkpoint)


@pytest.fixture(scope='module')
def prefix(tmp_path_factory):
    made,p,f=inputs('plastic-cycle')
    paused=native.solve_force_program(made,p,line_forces=f,stop_after=1)
    assert paused.status=='paused'
    save(tmp_path_factory.mktemp('prefix')/'checkpoint.json',paused.checkpoint)
    return paused


@pytest.mark.parametrize('stage',['before_trial','before_commit','factorized'])
def test_cancellation_does_not_publish_trial_history(prefix,stage,tmp_path):
    token=CancellationToken()
    def stop(event):
        if event['stage']=='fibre-schur-line.'+stage: token.cancel('transaction test')
    made,p,f=inputs('plastic-cycle')
    result=native.solve_force_program(made,p,line_forces=f,checkpoint=prefix.checkpoint,cancellation_token=token,progress=stop)
    save(tmp_path/'cancelled.json',dict(status=result.status,failure=result.failure,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))
    assert result.status=='cancelled' and result.checkpoint==prefix.checkpoint
    numerical_difference(result.state.histories,prefix.state.histories)


def test_linear_solve_failure_preserves_accepted_checkpoint(prefix,monkeypatch,tmp_path):
    def broken(*args,**kw): raise ValueError('injected factor failure')
    monkeypatch.setattr(native,'ResultantSchurSolver',broken)
    made,p,f=inputs('plastic-cycle')
    result=native.solve_force_program(made,p,line_forces=f,checkpoint=prefix.checkpoint)
    save(tmp_path/'failure.json',dict(status=result.status,failure=result.failure,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))
    assert result.status=='failed' and 'injected factor failure' in result.failure
    assert result.checkpoint==prefix.checkpoint


def test_frozen_program_mutation_and_failed_iteration_limit(tmp_path):
    made,p,f=inputs('curved-elastic'); c=native.Context(made,p,line_forces=f)
    c.program_data['linear_solver']='dense'
    with pytest.raises(ValueError,match='frozen'): c.guard()
    made,p,f=inputs('curved-elastic'); p=ForceProgram(p.targets,(),max_iterations=0)
    c=native.Context(made,p,line_forces=f)
    result=native.solve_force_program(made,p,line_forces=f)
    save(tmp_path/'failure.json',dict(status=result.status,failure=result.failure,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))
    assert result.status=='failed' and result.completed_targets==0
    assert result.checkpoint==c.checkpoint(())


@pytest.mark.parametrize('case',['straight-elastic','curved-elastic','plastic-cycle'])
def test_exactly_unloaded_rows_need_full_system_refinement(case,tmp_path):
    from anyfesolver._ge_beam3_fibre_schur_solver import ResultantSchurSolver as Old
    made,p,f=inputs(case); c=native.Context(made,p,line_forces=f)
    r,h,_,_=c.assemble(c.initial.mechanical,p.targets[0],c.initial.histories)
    a=c.tangent(r,h,p.targets[0]); unrefined=Old(c.layout,a)
    with pytest.raises(ValueError,match='recovered full'): unrefined.solve(-r)
    refined=native.ResultantSchurSolver(c.layout,a); answer=refined.solve(-r)
    full=np.zeros_like(r); full[c.layout.free]=np.linalg.solve(a[np.ix_(c.layout.free,c.layout.free)],-r[c.layout.free])
    assert 1<=answer.refinements<=3 and answer.backward_error<=1e-11
    assert np.linalg.norm(answer.increment-full)<=1e-11*np.linalg.norm(full)
    save(tmp_path/'refined.json',dict(case=case,answer=answer,diagnostics=refined.diagnostics()))


def test_refinement_exhaustion_uses_checked_full_mixed_fallback(monkeypatch):
    made,p,f=inputs('curved-elastic'); c=native.Context(made,p,line_forces=f)
    r,h,_,_=c.assemble(c.initial.mechanical,p.targets[0],c.initial.histories)
    factor=native.ResultantSchurSolver(c.layout,c.tangent(r,h,p.targets[0]))
    calls=[]
    def bad_apply(self,rhs): calls.append(1); return np.zeros_like(rhs)
    monkeypatch.setattr(native.ResultantSchurSolver,'_apply',bad_apply)
    answer=factor.solve(-r)
    assert len(calls)==4 and answer.full_mixed_fallback and answer.backward_error<=1e-11
    assert factor.diagnostics()['factorizations']==2
    count=factor.diagnostics()['solves']
    repeated=factor.solve(-r)
    assert factor.diagnostics()['factorizations']==2 and factor.diagnostics()['solves']==count+1
    np.testing.assert_array_equal(repeated.increment,answer.increment)
