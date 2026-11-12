"""Successor path transactions; compare preserved dense evidence without reruns."""
import json
from pathlib import Path
from hashlib import sha256
import pytest
from anyfesolver import _ge_beam3_equilibrium_line_program as native
from anyfesolver import _ge_beam3_fibre_line_program as dense
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver.control import CancellationToken
from test_ge_beam3_schur_line_program import inputs,save,numerical_difference

ARCHIVE=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-schur-controller-2756a7b-20260907')
MANIFEST=Path(__file__).resolve().parents[1]/'docs/reference_cases/ge_beam3_schur_controller_archive.json'


def historical(case):
    path='cycle-a/pytest/'+case+'0/dense.json'
    item=next(r for r in json.loads(MANIFEST.read_bytes())['files'] if r['path']==path)
    raw=(ARCHIVE/path).read_bytes()
    assert len(raw)==item['bytes'] and sha256(raw).hexdigest()==item['sha256']
    return raw


@pytest.fixture(autouse=True)
def no_dense_path_rerun(monkeypatch):
    def reject(*args,**kw): raise AssertionError('preserved dense path may not be rerun')
    monkeypatch.setattr(dense,'solve_force_program',reject)


@pytest.fixture(scope='module',params=['straight-elastic','curved-elastic','plastic-cycle'])
def path_result(request,tmp_path_factory):
    root=tmp_path_factory.mktemp(request.param); events=[]
    made,p,f=inputs(request.param)
    result=native.solve_force_program(made,p,line_forces=f,progress=events.append)
    save(root/'checkpoint.json',result.checkpoint)
    save(root/'status.json',dict(status=result.status,failure=result.failure,completed_targets=result.completed_targets))
    save(root/'progress.json',events)
    assert result.status=='completed',result.failure
    return request.param,root,result,events


def test_path_physical_recovery_and_no_mixed_fallback(path_result):
    case,root,result,events=path_result; old=historical(case)
    a=json.loads(old); b=json.loads(result.checkpoint)
    assert a['model_sha256']!=b['model_sha256'] and len(a['records'])==len(b['records'])
    differences=[]
    for x,y in zip(a['records'],b['records']):
        assert x['target']==y['target'] and x['parameter']==y['parameter']
        differences.append({k:numerical_difference(x[k],y[k]) for k in ('mechanical','origins','histories','residual')})
        assert max(x['metrics']+y['metrics'])<=1e-11
    made,p,f=inputs(case); c=native.Context(made,p,line_forces=f)
    state,records=c.restore(result.checkpoint)
    made,p,f=inputs(case); d=dense.Context(made,p,line_forces=f)
    old_state,_=d.restore(old)
    recovery=numerical_difference(c.recover(state),d.recover(old_state))
    assert c.checkpoint(records)==result.checkpoint
    factors=[e['linear_solver'] for e in events if e['stage']=='fibre-equilibrium-line.factorized']
    assert factors and all(not e['full_mixed_fallback'] and e['factorizations']==2 for e in factors)
    assert all(e['eliminated_force_coordinates']==18 and e['retained_physical_cell_rotations']==6 for e in factors)
    if case=='plastic-cycle': assert any(r[2]>0 for cell in state.histories for station in cell.stations for r in station.rows)
    save(root/'comparison.json',dict(case=case,records=differences,recovery_error=recovery,
        historical_dense_sha256=sha256(old).hexdigest(),checkpoint_sha256=sha256(result.checkpoint).hexdigest(),
        newton_factors=len(factors),full_mixed_fallbacks=0,production_qualified=False))


def test_resume_and_cross_backend_rejection(path_result):
    case,root,whole,_=path_result
    made,p,f=inputs(case); paused=native.solve_force_program(made,p,line_forces=f,stop_after=2)
    save(root/'paused.json',paused.checkpoint); assert paused.status=='paused',paused.failure
    made,p,f=inputs(case)
    resumed=native.solve_force_program(made,p,line_forces=f,checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    save(root/'resumed.json',resumed.checkpoint)
    assert resumed.status=='completed' and resumed.checkpoint==whole.checkpoint,resumed.failure
    made,p,f=inputs(case)
    with pytest.raises(ValueError,match='binding'): native.Context(made,p,line_forces=f).restore(historical(case))
    made,p,f=inputs(case)
    with pytest.raises(ValueError,match='binding'): dense.Context(made,p,line_forces=f).restore(whole.checkpoint)


@pytest.fixture(scope='module')
def prefix(tmp_path_factory):
    made,p,f=inputs('plastic-cycle'); result=native.solve_force_program(made,p,line_forces=f,stop_after=1)
    save(tmp_path_factory.mktemp('prefix')/'checkpoint.json',result.checkpoint)
    assert result.status=='paused',result.failure
    return result


@pytest.mark.parametrize('stage',['before_trial','before_commit','factorized'])
def test_cancellation_preserves_accepted_history(prefix,stage,tmp_path):
    token=CancellationToken()
    def stop(event):
        if event['stage']=='fibre-equilibrium-line.'+stage: token.cancel('transaction test')
    made,p,f=inputs('plastic-cycle')
    result=native.solve_force_program(made,p,line_forces=f,checkpoint=prefix.checkpoint,cancellation_token=token,progress=stop)
    save(tmp_path/'cancelled.json',dict(status=result.status,failure=result.failure,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))
    assert result.status=='cancelled' and result.checkpoint==prefix.checkpoint
    numerical_difference(result.state.histories,prefix.state.histories)


def test_factor_failure_preserves_accepted_checkpoint(prefix,monkeypatch,tmp_path):
    def reject(*args,**kw): raise ValueError('injected force recovery failure')
    monkeypatch.setattr(native,'ResultantSchurSolver',reject)
    made,p,f=inputs('plastic-cycle'); result=native.solve_force_program(made,p,line_forces=f,checkpoint=prefix.checkpoint)
    save(tmp_path/'failure.json',dict(status=result.status,failure=result.failure,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))
    assert result.status=='failed' and 'injected force recovery failure' in result.failure
    assert result.checkpoint==prefix.checkpoint


def test_program_mutation_and_iteration_limit(tmp_path):
    made,p,f=inputs('curved-elastic'); c=native.Context(made,p,line_forces=f)
    c.program_data['linear_solver']='dense'
    with pytest.raises(ValueError,match='frozen'): c.guard()
    made,p,f=inputs('curved-elastic'); p=ForceProgram(p.targets,(),max_iterations=0)
    c=native.Context(made,p,line_forces=f); result=native.solve_force_program(made,p,line_forces=f)
    save(tmp_path/'failure.json',dict(status=result.status,failure=result.failure,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))
    assert result.status=='failed' and result.completed_targets==0 and result.checkpoint==c.checkpoint(())
