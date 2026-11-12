"""Private retained generalized transactions, port equality and restart."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_retained_generalized_state import Context,Program
from anyfesolver._ge_beam3_retained_generalized_program import solve
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_generalized_slenderness_diagnostic import make
from test_ge_beam3_native_generalized_restart import make as plastic_model,pattern as plastic_load
from test_ge_beam3_schur_line_program import save

ARCHIVE=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-retained-force-a2a157b-20260908')
LOAD=DistributedPattern(LinePattern(((1,.005,-.003,.002),)),((1,.002,-.001,.003),))

def model(rho=100.,curved=True):return make(rho,curved,np.eye(3))[0]
def resume(m,p,result,**kwargs):return solve(m,p,checkpoint=result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest(),**kwargs)

def save_result(root,result):
    save(root/'produced-checkpoint.json',result.checkpoint)
    save(root/'result.json',dict(status=result.status,completed_targets=result.completed_targets,state=result.state,
        failure=result.failure,production_qualified=result.production_qualified,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))

@pytest.mark.parametrize('rho,curved',((100.,False),(100.,True),(10000.,False),(10000.,True),(1000000.,False),(1000000.,True)))
def test_native_port(rho,curved,tmp_path):
    raw=(ARCHIVE/'archive-manifest.json').read_bytes()
    assert len(raw)==4913 and sha256(raw).hexdigest().upper()=='6ACBA171BF00C5F04360E84CE0DE843AF971881A7426EE4647B7B2E0B2CD9821'
    lane='smoke' if rho==100. else str(rho)
    rel=lane+'/pytest/test_retained_generalized_forc'+str(int(curved))+'/retained.json'
    b=(ARCHIVE/rel).read_bytes();assert [len(b),sha256(b).hexdigest().upper()]==json.loads(raw)[rel]
    expected=json.loads(b);p=Program((.5,1.),LOAD);m=model(rho,curved)
    result=solve(m,p);save_result(tmp_path,result)
    assert result.status=='completed',result.failure
    s=result.state.mechanical
    for key,a in s.descriptor().items():
        old=np.array(expected['state'][key]);actual=a[0] if key in ('cell_rotations','resultants') else a
        np.testing.assert_array_equal(actual,old)
        with pytest.raises(ValueError):a.setflags(write=True)
    paused=solve(model(rho,curved),p,stop_after=1);assert paused.status=='paused'
    restarted=resume(model(rho,curved),p,paused);assert restarted.status=='completed',restarted.failure
    assert restarted.checkpoint==result.checkpoint
    replay=resume(model(rho,curved),p,restarted);assert replay.checkpoint==result.checkpoint
    save(tmp_path/'checkpoint.json',result.checkpoint)

@pytest.mark.parametrize('case',('curved-plastic','connected-plastic'))
def test_connected_plastic_history(case,tmp_path):
    m=plastic_model(case);p=Program((.25,.5,1.,.5,0.,-.5,0.),plastic_load(m))
    result=solve(m,p);save_result(tmp_path,result)
    assert result.status=='completed',result.failure
    assert any(sum(h.accumulated)>0 for cell in result.state.histories for h in cell.stations)
    paused=solve(plastic_model(case),p,stop_after=3);assert paused.status=='paused',paused.failure
    restarted=resume(plastic_model(case),p,paused);assert restarted.status=='completed',restarted.failure
    assert restarted.checkpoint==result.checkpoint
    c=Context(plastic_model(case),p);state,records=c.restore(result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    before=canonical(state);recovery=c.recover(state)
    assert canonical(state)==before and c.checkpoint(records)==result.checkpoint
    save(tmp_path/'recovery.json',recovery);save(tmp_path/'checkpoint.json',result.checkpoint)

@pytest.fixture(scope='module')
def accepted():
    p=Program((.5,1.),LOAD);r=solve(model(),p,stop_after=1)
    assert r.status=='paused',r.failure
    return p,r

@pytest.mark.parametrize('mutation',('position','rotation','resultant','origin','history','work','reaction','target','schema','extra','bool','hash'))
def test_rehashed_corruption_rejected(accepted,mutation):
    p,r=accepted;v=json.loads(r.checkpoint);row=v['records'][0]
    if mutation=='position':row['mechanical']['positions'][-1][0]+=.01
    elif mutation=='rotation':row['mechanical']['nodal_frames'][-1][0][0]+=.01
    elif mutation=='resultant':row['mechanical']['resultants'][0][0]+=.01
    elif mutation=='origin':row['origins'][0]['stations'][0]['plastic'][0][0]+=.01
    elif mutation=='history':row['histories'][0]['stations'][0]['plastic'][0][0]+=.01
    elif mutation=='work':row['work'][0]+=.01
    elif mutation=='reaction':row['residual'][0]+=.01
    elif mutation=='target':row['parameter']=.6
    elif mutation=='schema':v['schema']='HISTORICAL_OTHER_FORMULATION'
    elif mutation=='extra':row['extra']=False
    elif mutation=='bool':row['mechanical']['positions'][0][0]=True
    else:v['model_sha256']='0'*64
    row['record_sha256']=sha({k:x for k,x in row.items() if k!='record_sha256'})
    v['checkpoint_sha256']=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'})
    raw=canonical(v)
    with pytest.raises((ValueError,KeyError,TypeError)):
        solve(model(),p,checkpoint=raw,expected_sha256=sha256(raw).hexdigest())

@pytest.mark.parametrize('stage',('before_assembly','before_factorization','before_trial','before_commit','committed'))
def test_cancellation_publication(accepted,stage,tmp_path):
    p,old=accepted;token=CancellationToken()
    def progress(row):
        if row['stage']=='retained-generalized.'+stage:token.cancel()
    r=resume(model(),p,old,cancellation_token=token,progress=progress)
    assert r.status=='cancelled'
    if stage!='committed':assert r.checkpoint==old.checkpoint and r.completed_targets==1
    else:assert r.completed_targets==2 and r.checkpoint!=old.checkpoint
    replay=resume(model(),p,r);assert replay.status=='completed',replay.failure
    save(tmp_path/'accepted.json',r.checkpoint)

def test_rejected_iteration_and_foreign_inputs(accepted):
    p,old=accepted
    failed=solve(model(),Program((.5,1.),LOAD,max_iterations=1))
    assert failed.status=='failed' and failed.completed_targets==0
    with pytest.raises(ValueError):resume(model(10000.),p,old)
    with pytest.raises(ValueError):solve(model(),p,checkpoint=old.checkpoint)
    with pytest.raises(ValueError):resume(model(),p,old,stop_after=0)
    for targets in ((True,),(.5,float('nan')),()):
        with pytest.raises(ValueError):Program(targets,LOAD)

@pytest.mark.parametrize('raw', (b'{"schema":1,"schema":2}\n',b'{"x":NaN}\n',b'{}'))
def test_strict_json(accepted,raw):
    p,_=accepted
    with pytest.raises(ValueError):solve(model(),p,checkpoint=raw,expected_sha256=sha256(raw).hexdigest())

def test_exception_before_commit_preserves_capsule(accepted):
    p,old=accepted
    def progress(row):
        if row['stage']=='retained-generalized.before_commit':raise RuntimeError('injected')
    result=resume(model(),p,old,progress=progress)
    assert result.status=='failed' and result.checkpoint==old.checkpoint
