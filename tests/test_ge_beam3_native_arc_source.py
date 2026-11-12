"""Native translation-to-arc history, orientation, immutable prefixes and restart."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_translation import solve_translation
from anyfesolver._ge_beam3_native_translation_restart import decode_checkpoint as decode_translation,encode_checkpoint as encode_translation
from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
from anyfesolver._ge_beam3_native_arc import ArcProgram,solve_arc
from anyfesolver._ge_beam3_native_arc_restart import encode_checkpoint,decode_checkpoint,SOURCE_SCHEMA
from anyfesolver._ge_beam3_native_history_profile import HISTORY8M
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_ge_beam3_native_translation import bar
from test_ge_beam3_native_arc import packet
from test_ge_beam3_schur_line_program import save


@pytest.fixture(scope='module',params=(None,HISTORY8M))
def source(request,tmp_path_factory):
    root=tmp_path_factory.mktemp('source-'+('default' if request.param is None else '8m'))
    model,p=bar();p=replace(p,history_profile=request.param)
    result=solve_translation(model,p,stop_after=1);assert result.status=='paused'
    save(root/'translation.json',result.checkpoint)
    seed=TranslationArcSource(p,result.checkpoint,sha256(result.checkpoint).hexdigest(),1.)
    arc=ArcProgram((.1,.1),1.,p.distributed,history_profile=request.param,source=seed)
    fresh,_=bar();made=solve_arc(fresh,arc);save(root/'result.json',packet(made));save(root/'checkpoint.json',made.checkpoint)
    assert made.status=='completed',made.failure
    return root,seed,arc,made


def test_source_history_and_analytic_solution(source):
    root,seed,arc,result=source;model,_=bar()
    chain,records=decode_checkpoint(model,arc,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    prefix,old=decode_translation(model,seed.program,seed.checkpoint,expected_sha256=seed.expected_sha256)
    assert canonical(chain[:len(prefix)])==canonical(prefix) and len(chain)==4 and len(records)==2
    assert records[0]['parameter']>old[-1]['parameter'] and records[1]['parameter']>records[0]['parameter']
    errors=[]
    for row,snapshot in zip(records,chain[len(prefix):],strict=True):
        errors.append(abs(snapshot['displacements'][12]-.125*row['parameter']))
        errors.append(abs(snapshot['displacements'][6]-.09375*row['parameter']))
    assert max(errors)<=1e-11 and encode_checkpoint(model,arc,chain,records)==result.checkpoint
    value=json.loads(result.checkpoint);assert canonical(value['source_checkpoint'])==seed.checkpoint
    save(root/'analytic.json',dict(errors=errors,source_prefix_exact=True,complete_snapshots=len(chain),arc_records=len(records)))


def test_source_prefix_resume(source):
    root,seed,arc,whole=source;model,_=bar();part=solve_arc(model,arc,stop_after=1)
    assert part.status=='paused'
    fresh,_=bar();resumed=solve_arc(fresh,arc,checkpoint=part.checkpoint,expected_sha256=sha256(part.checkpoint).hexdigest())
    assert resumed.status=='completed' and resumed.checkpoint==whole.checkpoint
    save(root/'resume.json',dict(byte_identical=True))


def test_source_reverse_orientation(source,tmp_path):
    _,seed,arc,_=source;model,_=bar()
    reverse=replace(arc,source=replace(seed,forward_sign=-1.))
    result=solve_arc(model,reverse);assert result.status=='completed',result.failure
    assert result.parameter<.08
    save(tmp_path/'reverse.json',packet(result))


@pytest.mark.parametrize('kind',('source-hash','source-bytes','source-type','sign','profile','load','initial-sign','history-count','genesis'))
def test_source_rejection(source,kind,tmp_path):
    _,seed,arc,_=source;model,_=bar()
    if kind=='source-hash':changed=replace(arc,source=replace(seed,expected_sha256='0'*64))
    elif kind=='source-bytes':changed=replace(arc,source=replace(seed,checkpoint=seed.checkpoint+b' '))
    elif kind=='source-type':changed=replace(arc,source=object())
    elif kind=='sign':changed=replace(arc,source=replace(seed,forward_sign=True))
    elif kind=='profile':changed=replace(arc,history_profile=HISTORY8M if arc.history_profile is None else None)
    elif kind=='load':
        from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
        from anyfesolver._ge_beam3_native_line_loading import LinePattern
        changed=replace(arc,distributed=DistributedPattern(LinePattern(((1,2.,0.,0.),)),()))
    elif kind=='initial-sign':changed=replace(arc,initial_sign=-1.)
    elif kind=='history-count':changed=replace(arc,steps=(.1,)*64)
    else:
        empty=solve_translation(model,seed.program,stop_after=0).checkpoint
        changed=replace(arc,source=replace(seed,checkpoint=empty,expected_sha256=sha256(empty).hexdigest()))
    with pytest.raises(ValueError):solve_arc(model,changed)
    save(tmp_path/'rejected.json',dict(kind=kind,rejected=True))


@pytest.mark.parametrize('kind',('embedded','source-descriptor','prefix','direction','schema','external'))
def test_checkpoint_mutation(source,kind,tmp_path):
    _,seed,arc,result=source;model,_=bar();v=json.loads(result.checkpoint)
    if kind=='embedded':v['source_checkpoint']['records'][0]['parameter']+=.01
    elif kind=='source-descriptor':v['program']['source']['forward_sign']=-1.;v['program_sha256']=sha(v['program'])
    elif kind=='prefix':v['accepted_chain']['snapshots'][1]['displacements'][12]+=.01
    elif kind=='direction':v['records'][0]['direction'][-1]*=-1.
    elif kind=='schema':v['schema']='wrong'
    v['checkpoint_sha256']=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'});raw=canonical(v)
    with pytest.raises(ValueError):decode_checkpoint(model,arc,raw,expected_sha256='0'*64 if kind=='external' else sha256(raw).hexdigest())
    save(tmp_path/'mutation.json',dict(kind=kind,rejected=True))


def test_live_source_mutation_preserves_arc_prefix(source,tmp_path):
    _,seed,arc,result=source;model,_=bar()
    changed=replace(arc,source=replace(seed))
    def observer(row):
        if row['stage']=='before_assembly' and row['step']==2:object.__setattr__(changed.source,'forward_sign',-1.)
    failed=solve_arc(model,changed,progress=observer)
    assert failed.status=='failed'
    fresh,_=bar();chain,records=decode_checkpoint(fresh,arc,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    expected=encode_checkpoint(fresh,arc,chain[:3],records[:1])
    assert failed.checkpoint==expected
    save(tmp_path/'preserved.json',dict(accepted_prefix_exact=True,result=packet(failed)))


def test_unseeded_rejects_seeded_checkpoint(source,tmp_path):
    _,_,arc,result=source;model,_=bar()
    with pytest.raises(ValueError):decode_checkpoint(model,replace(arc,source=None),result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    save(tmp_path/'rejected.json',dict(unseeded_rejects_seeded=True))


def test_external_arc_guard_precedes_source_replay(source,monkeypatch,tmp_path):
    from anyfesolver import _ge_beam3_native_arc_source as module
    _,_,arc,result=source;model,_=bar()
    def tripwire(*args):raise RuntimeError('source replay must not run')
    monkeypatch.setattr(module,'load_source',tripwire)
    with pytest.raises(ValueError,match='external checkpoint SHA-256'):
        solve_arc(model,arc,checkpoint=result.checkpoint,expected_sha256='0'*64)
    save(tmp_path/'early-guard.json',dict(rejected_before_source_replay=True))


def test_secant_objectivity_and_zero(tmp_path):
    from anyfesolver._ge_beam3_native_arc_source import secant_orientation
    from anyfesolver._native_rotation_state import rotation_exponential
    s=rotation_exponential(np.array([.7,-.3,.4]))
    old=np.arange(12,dtype=float)*.002
    new=old+np.array([.02,-.01,.03,.04,.03,-.01,-.02,.03,.01,-.03,.01,.02])
    chain=({'displacements':old},{'displacements':new})
    metric=np.r_[np.tile([2.,2.,2.,3.,3.,3.],2),4.]
    expected=secant_orientation(chain,.3,.2,metric,1.)
    moved=tuple({'displacements':np.array([s@v for v in row['displacements'].reshape(-1,3)]).ravel()} for row in chain)
    actual=secant_orientation(moved,.3,.2,metric,1.)
    rotated=np.r_[np.array([s@v for v in expected[:-1].reshape(-1,3)]).ravel(),expected[-1]]
    error=float(np.max(np.abs(actual-rotated)));assert error<=1e-11
    assert np.array_equal(secant_orientation(chain,.3,.2,metric,-1.),-expected)
    with pytest.raises(ValueError,match='nonzero'):secant_orientation((chain[0],chain[0]),.2,.2,metric,1.)
    save(tmp_path/'secant.json',dict(covariance_error=error,reverse_exact=True,zero_rejected=True))


def test_curved_plastic_source(tmp_path):
    from test_ge_beam3_native_generalized_restart import make,pattern
    from anyfesolver._ge_beam3_native_translation import TranslationProgram
    from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
    from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
    model=make('curved-plastic')
    p=TranslationProgram((.03,),3,'ux',pattern(model),SpatialNodalMoments(((3,.015,-.01,.008),)))
    original=solve_translation(model,p);save(tmp_path/'source-result.json',dict(status=original.status,failure=original.failure))
    save(tmp_path/'translation.json',original.checkpoint);assert original.status=='completed',original.failure
    seed=TranslationArcSource(p,original.checkpoint,sha256(original.checkpoint).hexdigest(),1.)
    arc=ArcProgram((.04,.04),1.,p.distributed,p.nodal_moments,source=seed)
    fresh=make('curved-plastic');result=solve_arc(fresh,arc)
    save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
    assert result.status=='completed',result.failure
    check=make('curved-plastic');chain,records=decode_checkpoint(check,arc,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    prefix,_=decode_translation(check,p,seed.checkpoint,expected_sha256=seed.expected_sha256)
    assert canonical(prefix)==canonical(chain[:2])
    positive=sum(sum(s.accumulated)>0 for state in chain[-1]['states'].values() for s in state['response'].history.stations)
    assert positive>0
    recovery={eid:recover_native_fields(e,check.mesh,chain[-1]['states'][eid],expected_committed_total_u=chain[-1]['displacements'][list(e.get_dof_mapping(check.mesh))]) for eid,e in check.mesh.elements.items()}
    partial=encode_checkpoint(check,arc,chain[:3],records[:1]);again=make('curved-plastic')
    resumed=solve_arc(again,arc,checkpoint=partial,expected_sha256=sha256(partial).hexdigest())
    assert resumed.status=='completed' and resumed.checkpoint==result.checkpoint,resumed.failure
    save(tmp_path/'recovery.json',dict(fields=recovery,positive_plastic_stations=positive,source_prefix_exact=True,resume_byte_identical=True,parameters=[r['parameter'] for r in records]))
