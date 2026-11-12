"""Accepted-prefix cancellation/failure and preflight programme guards."""
from dataclasses import replace
from hashlib import sha256
import pytest
from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_native_translation import solve_translation,_PATH
from anyfesolver._ge_beam3_native_translation_restart import decode_checkpoint
from anyfesolver._ge_beam3_native_generalized_combined_couples import _RUN
from anyfesolver._ge_beam3_native_generalized_loading import _ACTIVE
from test_ge_beam3_native_translation import bar,packet
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('kind',['cancel','factorization'])
def test_accepted_prefix_preserved(kind,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_translation as module
    m,p=bar();prefix=solve_translation(m,p,stop_after=1);assert prefix.status=='paused'
    m,p=bar();token=CancellationToken();committed=False;original=module.factorize
    def observer(event):
        nonlocal committed
        if event['stage']=='committed' and event['target']==1:committed=True
        if kind=='cancel' and event['stage']=='before_commit' and event['target']==2:token.cancel('prescribed accepted-prefix cancellation')
    def factor(*a,**kw):
        if committed:raise RuntimeError('prescribed second-target factorization failure')
        return original(*a,**kw)
    if kind=='factorization':monkeypatch.setattr(module,'factorize',factor)
    result=solve_translation(m,p,progress=observer,cancellation_token=token)
    assert result.status==('cancelled' if kind=='cancel' else 'failed')
    assert result.completed_targets==1 and result.checkpoint==prefix.checkpoint
    fresh,_=bar();chain,records=decode_checkpoint(fresh,p,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert len(chain)==2 and len(records)==1 and _PATH.get() is None and _RUN.get() is None and _ACTIVE.get() is None
    save(tmp_path/'preserved.json',dict(kind=kind,result=packet(result),accepted_prefix_preserved=True))


@pytest.mark.parametrize('kind',['targets','target-type','control','node','iterations','backtracks','pattern','stop','observer','hash'])
def test_preflight_before_assembly(kind,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_translation as module
    m,p=bar();kwargs={}
    if kind=='targets':p=replace(p,targets=())
    elif kind=='target-type':p=replace(p,targets=(True,))
    elif kind=='control':p=replace(p,control_component='rx')
    elif kind=='node':p=replace(p,control_node=1)
    elif kind=='iterations':p=replace(p,max_iterations=True)
    elif kind=='backtracks':p=replace(p,max_backtracks=9)
    elif kind=='pattern':p=replace(p,distributed=object())
    elif kind=='stop':kwargs['stop_after']=True
    elif kind=='observer':kwargs['progress']=3
    else:kwargs['expected_sha256']='0'*64
    def forbidden(*a,**kw):raise AssertionError('assembly must not be entered')
    monkeypatch.setattr(module,'assemble',forbidden)
    with pytest.raises(ValueError):solve_translation(m,p,**kwargs)
    assert _PATH.get() is None
    save(tmp_path/'preflight.json',dict(kind=kind,rejected_before_assembly=True))
