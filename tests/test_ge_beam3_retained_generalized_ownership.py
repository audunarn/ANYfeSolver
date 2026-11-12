"""Direct-object recovery and staging must require this context's issuance."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver._ge_beam3_retained_generalized_state import Context,Program
from anyfesolver._ge_beam3_retained_generalized_program import solve
from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_retained_generalized_state import model,LOAD,resume

@pytest.mark.parametrize('kind',('copy','foreign','changed','mutated'))
@pytest.mark.parametrize('operation',('recover','stage'))
def test_reject_unissued_before_mechanics(kind,operation,monkeypatch):
    p=Program((.5,1.),LOAD);c=Context(model(),p);state=c.initial
    if kind=='copy':state=replace(state)
    elif kind=='foreign':state=Context(model(),p).initial
    elif kind=='changed':
        a=state.mechanical.resultants.copy();a[0,0]=.01
        state=replace(state,mechanical=replace(state.mechanical,resultants=a))
    else:object.__setattr__(state,'completed_targets',1)
    def forbidden(*args,**kwargs):raise AssertionError('unissued state entered mechanics')
    monkeypatch.setattr(RetainedGeneralizedOperator,'recover',forbidden)
    monkeypatch.setattr(RetainedGeneralizedOperator,'evaluate',forbidden)
    with pytest.raises(ValueError,match='issued|changed'):
        if operation=='recover':c.recover(state)
        else:c.stage(c.initial.mechanical,state,(),0)

def test_checkpoint_rejects_unissued_record():
    c=Context(model(),Program((.5,),LOAD))
    with pytest.raises(ValueError,match='issued|chain'):c.checkpoint((c.genesis,))

def test_issued_recovery_and_authenticated_clone():
    from hashlib import sha256
    p=Program((.5,1.),LOAD);m=model();c=Context(m,p)
    before=canonical(c.initial);c.recover(c.initial);assert canonical(c.initial)==before
    r=solve(m,p);assert r.status=='completed',r.failure
    c=Context(model(),p);state,records=c.restore(r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    before=canonical(state);c.recover(state)
    assert canonical(state)==before and c.checkpoint(records)==r.checkpoint
    assert resume(model(),p,r).checkpoint==r.checkpoint
