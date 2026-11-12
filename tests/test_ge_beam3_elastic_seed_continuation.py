"""Authentic seed/chain ownership and actual analytical elastic continuation."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_elastic_seed_continuation as c
from anyfesolver import _ge_beam3_retained_translation_control as old
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_generalized_modal import make

def model():return make(False,1,clamped=True,coupled=False)[0]
def programme(targets=(.0002,.0003,0.,-.0001),**kw):
    return old.Program(targets,3,(1.,0.,0.),old.NodalDeadForces(((3,1.,0.,0.),)),**kw)

@pytest.fixture(scope='module')
def seed():
    m=model();p=programme((.0001,));result=old.solve(m,p)
    assert result.status=='completed',result.failure
    context=old.Context(m,p);physical=context.physical
    return canonical(dict(schema=c.SEED_SCHEMA,model_sha256=physical.model_identity,
        operators=[e.operator.identity for _,e in physical.elements],control_node=3,direction=p.direction,
        nodal_forces=p.nodal_forces,mechanical=result.state.mechanical.descriptor(),parameter=result.state.parameter,
        displacement=.0001,source_sha256=sha256(result.checkpoint).hexdigest()))

def context(seed,p=None):return c.Context(model(),p or programme(),seed,expected_seed_sha256=sha256(seed).hexdigest())
def solve(seed,**kw):return c.solve(model(),programme(),seed,expected_seed_sha256=sha256(seed).hexdigest(),**kw)

def test_actual_load_unload_reverse_original_prefix_resume(seed,tmp_path):
    full=solve(seed);assert full.status=='completed',full.failure
    prefix=solve(seed,stop_after=2);assert prefix.status=='paused',prefix.failure
    resumed=solve(seed,checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert resumed.status=='completed',resumed.failure
    assert full.checkpoint==resumed.checkpoint
    owner=context(seed);state,records=owner.restore(full.checkpoint,expected_sha256=sha256(full.checkpoint).hexdigest())
    assert owner.checkpoint(records)==full.checkpoint
    for raw in records:
        row=json.loads(raw);target=row['displacement_target']
        assert abs(row['parameter']-500.*target)<1e-11
        assert max(*row['metrics'],row['correction'])<=1e-11
        assert abs(row['work'][-1]-row['parameter']*target)<1e-11
        assert abs(row['residual'][0]+row['parameter'])<1e-11
    before=canonical(state);owner.recover(state);assert canonical(state)==before
    assert not full.production_qualified and not full.physical_loading_path_from_rest
    (tmp_path/'full.json').write_bytes(full.checkpoint);(tmp_path/'prefix.json').write_bytes(prefix.checkpoint)

@pytest.mark.parametrize('kind',('parameter','displacement','mechanical','operator','model','load','source','history','duplicate','nonfinite','whitespace','external_hash'))
def test_seed_mutations(seed,kind):
    value=json.loads(seed);raw=seed;expected=None
    if kind=='parameter':value['parameter']+=.1
    elif kind=='displacement':value['displacement']+=.001
    elif kind=='mechanical':value['mechanical']['positions'][2][0]+=.01
    elif kind=='operator':value['operators'][0]='0'*64
    elif kind=='model':value['model_sha256']='0'*64
    elif kind=='load':value['nodal_forces']['rows'][0][1]=2.
    elif kind=='source':value['source_sha256']='unknown'
    elif kind=='history':value['histories']=[]
    elif kind=='duplicate':raw=raw.replace(b'{',b'{"schema":"duplicate",',1)
    elif kind=='nonfinite':raw=raw.replace(b'"parameter":',b'"parameter":NaN,"extra":',1)
    elif kind=='whitespace':raw+=b' '
    elif kind=='external_hash':expected='0'*64
    if kind not in ('duplicate','nonfinite','whitespace','external_hash'):raw=canonical(value)
    with pytest.raises((ValueError,TypeError)):
        c.Context(model(),programme(),raw,expected_seed_sha256=expected or sha256(raw).hexdigest())

@pytest.mark.parametrize('kind',('clone','foreign','state','map','program','identity','seed','target'))
def test_owner_mutations(seed,kind):
    owner=context(seed)
    if kind=='clone':state=replace(owner.initial)
    elif kind=='foreign':state=context(seed).initial
    else:
        state=owner.initial
        if kind=='state':object.__setattr__(state,'parameter',1.)
        elif kind=='map':owner.row=-owner.row
        elif kind=='program':object.__setattr__(owner.program,'targets',(.9,))
        elif kind=='identity':owner.identity='0'*64
        elif kind=='seed':owner.seed_bytes+=b' '
        else:owner.seed_target=.5
    with pytest.raises(ValueError):owner.recover(state)

@pytest.mark.parametrize('kind',('parameter','work','recovery','origins','cursor_bool','schema','seed_hash','path_claim','genesis'))
def test_rehashed_checkpoint_mutations(seed,kind):
    owner=context(seed,programme((.0001,)))
    state,record=owner.stage(owner.initial.mechanical,owner.initial.parameter,owner.initial,(),0)
    raw=owner.checkpoint((record,));value=json.loads(raw);row=value['records'][0]
    if kind=='parameter':row['parameter']+=.1
    elif kind=='work':row['work'][-1]+=.1
    elif kind=='recovery':row['recovery_sha256']='0'*64
    elif kind=='origins':row['origins']=[]
    elif kind=='cursor_bool':value['completed_targets']=True
    elif kind=='schema':value['schema']=old.SCHEMA
    elif kind=='seed_hash':value['seed_sha256']='0'*64
    elif kind=='path_claim':value['physical_loading_path_from_rest']=True
    else:value['initial']['parameter']+=.1
    row['record_sha256']=sha({k:v for k,v in row.items() if k!='record_sha256'})
    value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'})
    changed=canonical(value)
    with pytest.raises(ValueError):owner.restore(changed,expected_sha256=sha256(changed).hexdigest())

@pytest.mark.parametrize('when',('before_commit','committed'))
def test_cancel_retains_original_accepted_prefix(seed,when):
    token=CancellationToken()
    def progress(row):
        if row['stage']=='elastic-continuation.'+when:token.cancel()
    result=solve(seed,cancellation_token=token,progress=progress)
    assert result.status=='cancelled'
    expected=0 if when=='before_commit' else 1
    assert result.completed_targets==expected
    baseline=solve(seed,stop_after=expected)
    assert result.checkpoint==baseline.checkpoint
    owner=context(seed);state,_=owner.restore(result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert state.completed_targets==expected

def test_ordinary_capsules_and_initializers_are_not_seed_authority(seed):
    owner=context(seed);ordinary=old.Context(model(),programme())
    with pytest.raises(ValueError):ordinary.restore(owner.checkpoint(()),expected_sha256=sha256(owner.checkpoint(())).hexdigest())
    with pytest.raises(ValueError):owner.restore(ordinary.checkpoint(()),expected_sha256=sha256(ordinary.checkpoint(())).hexdigest())
    with pytest.raises(ValueError):owner.recover(ordinary.initial)

def test_elastic_origin_and_trial_history_rejection(seed,monkeypatch):
    owner=context(seed);origins=list(owner.initial.histories);origins=[]
    with pytest.raises(ValueError,match='material history'):
        owner.assemble(owner.initial.mechanical,owner.initial.parameter,origins,owner.seed_target)
    original=owner.physical.assemble
    def changed(*args):
        r,j,metrics,responses,work=original(*args)
        from types import SimpleNamespace
        responses=[SimpleNamespace(history={'not':'virgin'}) for _ in responses]
        return r,j,metrics,responses,work
    monkeypatch.setattr(owner.physical,'assemble',changed)
    with pytest.raises(ValueError,match='plastic trial'):
        owner.assemble(owner.initial.mechanical,owner.initial.parameter,owner.initial.histories,owner.seed_target)
