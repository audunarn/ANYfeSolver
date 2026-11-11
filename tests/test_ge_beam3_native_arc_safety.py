"""Native arc accepted-prefix safety and fail-closed admission."""
from dataclasses import replace
from hashlib import sha256
import pytest
from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_native_arc import solve_arc,_ARC
from anyfesolver._ge_beam3_native_arc_restart import decode_checkpoint
from anyfesolver._ge_beam3_native_translation import _PATH
from anyfesolver._ge_beam3_native_generalized_program import _PROGRAM
from test_ge_beam3_native_arc import bar,packet
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('kind',['cancel','factorization','mutation'])
def test_accepted_prefix_preserved(kind,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_arc as module
    m,p=bar();prefix=solve_arc(m,p,stop_after=1);assert prefix.status=='paused'
    m,p=bar();token=CancellationToken();committed=False;original=module.factorize
    def observer(event):
        nonlocal committed
        if event['stage']=='committed' and event['step']==1:committed=True
        if event['stage']=='before_commit' and event['step']==2:
            if kind=='cancel':token.cancel('prescribed arc cancellation')
            elif kind=='mutation':object.__setattr__(p,'length_scale',2.)
    def factor(*a,**kw):
        if committed:raise RuntimeError('prescribed second-step arc factorization failure')
        return original(*a,**kw)
    if kind=='factorization':monkeypatch.setattr(module,'factorize',factor)
    result=solve_arc(m,p,progress=observer,cancellation_token=token)
    assert result.status==('cancelled' if kind=='cancel' else 'failed') and result.completed_steps==1
    assert result.checkpoint==prefix.checkpoint and _ARC.get() is None
    # Replay has its own real factorization; the worker fault has ended.
    monkeypatch.setattr(module,'factorize',original)
    fresh,original_program=bar();chain,records=decode_checkpoint(fresh,original_program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert len(chain)==2 and len(records)==1
    save(tmp_path/'preserved.json',dict(kind=kind,result=packet(result),accepted_prefix_preserved=True))


@pytest.mark.parametrize('kind',['steps','step-zero','step-range','step-type','length','tiny-length','huge-length','parameter-scale','sign','iterations','backtracks','stop','hash','translation-scope','force-scope','arc-scope'])
def test_preflight(kind,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_arc as module
    m,p=bar();kwargs={};scope=None;token=None
    if kind=='steps':p=replace(p,steps=())
    elif kind=='step-zero':p=replace(p,steps=(0.,))
    elif kind=='step-range':p=replace(p,steps=(.3,))
    elif kind=='step-type':p=replace(p,steps=(True,))
    elif kind=='length':p=replace(p,length_scale=-1.)
    elif kind=='tiny-length':p=replace(p,length_scale=1e-200)
    elif kind=='huge-length':p=replace(p,length_scale=1e200)
    elif kind=='parameter-scale':p=replace(p,parameter_scale=float('nan'))
    elif kind=='sign':p=replace(p,initial_sign=0.)
    elif kind=='iterations':p=replace(p,max_iterations=True)
    elif kind=='backtracks':p=replace(p,max_backtracks=9)
    elif kind=='stop':kwargs['stop_after']=True
    elif kind=='hash':kwargs['expected_sha256']='0'*64
    else:
        scope={'translation-scope':_PATH,'force-scope':_PROGRAM,'arc-scope':_ARC}[kind];token=scope.set(object())
    def forbidden(*a,**kw):raise AssertionError('arc assembly must not be entered')
    monkeypatch.setattr(module,'assemble',forbidden)
    try:
        with pytest.raises(ValueError):solve_arc(m,p,**kwargs)
    finally:
        if scope is not None:scope.reset(token)
    assert _ARC.get() is None
    save(tmp_path/'preflight.json',dict(kind=kind,rejected_before_assembly=True))
