"""Actual bordered native equilibrium, signed paths and complete restart."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from scipy import sparse
from anyfesolver._ge_beam3_native_translation import TranslationProgram,solve_translation,bordered,effective,load_point,_PATH
from anyfesolver._ge_beam3_native_translation_restart import encode_checkpoint,decode_checkpoint
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern,_ACTIVE
from anyfesolver._ge_beam3_native_generalized_combined_couples import _RUN
from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_ge_beam3_native_generalized_restart import make,pattern
from test_ge_beam3_native_generalized_combined_couples import convert,simple
from test_ge_beam3_schur_line_program import save


def bar():
    m=convert(simple()[0]);p=TranslationProgram((.01,-.005,0.),3,'ux',DistributedPattern(LinePattern(((1,1.,0.,0.),)),()))
    return m,p


def packet(result):
    return dict(status=result.status,cursor=result.completed_targets,parameter=result.parameter,u=result.displacements,
        reaction=result.physical_imbalance,checkpoint_sha256=sha256(result.checkpoint).hexdigest(),failure=result.failure,events=result.events)


@pytest.fixture(scope='module')
def saved_bar(tmp_path_factory):
    root=tmp_path_factory.mktemp('signed-bar');m,p=bar();r=solve_translation(m,p)
    save(root/'result.json',packet(r));save(root/'checkpoint.json',r.checkpoint)
    assert r.status=='completed',r.failure
    return root,p,r


def test_signed_analytical_bar(saved_bar):
    root,p,r=saved_bar;m,_=bar();chain,records=decode_checkpoint(m,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    errors=[]
    for row,state in zip(records,chain[1:]):
        parameter=8*row['target'];errors.append(abs(row['parameter']-parameter))
        expected=np.zeros(18);expected[6]=parameter*.5*(1-.25)/4;expected[12]=row['target']
        errors.append(float(np.max(np.abs(state['displacements']-expected))))
    assert max(errors)<=1e-11 and records[1]['parameter']<0.
    assert encode_checkpoint(m,p,chain,records)==r.checkpoint
    save(root/'analytical.json',dict(errors=errors,parameters=[v['parameter'] for v in records],roundtrip=True))


def test_bar_prefix_resume(saved_bar):
    root,p,whole=saved_bar;m,_=bar();prefix=solve_translation(m,p,stop_after=1)
    assert prefix.status=='paused'
    fresh,_=bar();resumed=solve_translation(fresh,p,checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert resumed.status=='completed',resumed.failure
    assert resumed.checkpoint==whole.checkpoint
    save(root/'resume.json',dict(result=packet(resumed),prefix_sha256=sha256(prefix.checkpoint).hexdigest(),byte_identical=True))


@pytest.mark.parametrize('kind',['hash','duplicate','nonfinite','schema','target','parameter','iteration','program','history','missing'])
def test_checkpoint_mutation(saved_bar,kind):
    root,p,r=saved_bar;m,_=bar();v=json.loads(r.checkpoint);raw=r.checkpoint
    if kind=='hash':expected='0'*64
    elif kind=='duplicate':raw=b'{"schema":"bad",'+raw[1:]
    elif kind=='nonfinite':raw=b'{"bad":NaN}'
    else:
        if kind=='schema':v['schema']='GE_BEAM3_SUPPORTED_NATIVE_GENERALIZED_COMBINED_COUPLE_RESTART_V1'
        elif kind=='target':v['records'][0]['target']+=.001
        elif kind=='parameter':v['records'][0]['parameter']+=.1
        elif kind=='iteration':v['records'][0]['iterations']=True
        elif kind=='program':v['program']['control_component']='uy';v['program_sha256']=sha(v['program'])
        elif kind=='history':v['accepted_chain']['snapshots'][1]['states'][0]['state']['epoch']+=1
        elif kind=='missing':v['records'].pop()
        v['checkpoint_sha256']=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'});raw=canonical(v)
    expected='0'*64 if kind=='hash' else sha256(raw).hexdigest()
    with pytest.raises(ValueError):decode_checkpoint(m,p,raw,expected_sha256=expected)
    save(root/(kind+'-rejected.json'),dict(kind=kind,rejected=True))


def test_bordered_at_simple_limit(tmp_path):
    # A singular K is not inverted. The independently specified augmented
    # equations remain nonsingular with this control row and load column.
    a=bordered(sparse.diags([0.,2.],format='csr'),np.array([-1.,-3.]),np.array([0,1]),0).toarray()
    expected=np.array([.2,.4,-.1]);rhs=np.array([.1,1.1,.2])
    np.testing.assert_allclose(a@expected,rhs,atol=1e-15,rtol=0.)
    np.testing.assert_allclose(np.linalg.solve(a,rhs),expected,atol=1e-15,rtol=0.)
    save(tmp_path/'bordered.json',dict(matrix=a,rhs=rhs,solution=expected,beam_postbuckling_claim=False))


def test_signed_zero_mapping(tmp_path):
    m,p=bar();values=[]
    for factor in (-.04,-0.,0.,.08):
        pattern,_=effective(p,factor);point=load_point(p,factor)
        assert canonical(pattern)==canonical(point.distributed.effective(m))
        values.append(dict(factor=factor,pattern=pattern))
    save(tmp_path/'signed-zero.json',values)


@pytest.mark.parametrize('kind',['failure','mutation','nested'])
def test_transaction_failure(kind,tmp_path,monkeypatch):
    import anyfesolver._ge_beam3_native_translation as module
    m,p=bar();genesis=solve_translation(m,p,stop_after=0)
    fresh,p=bar();original=module.factorize
    if kind=='failure':
        def fail(*a,**kw):raise RuntimeError('prescribed factorization failure')
        monkeypatch.setattr(module,'factorize',fail)
    def observer(event):
        if event['stage']=='before_commit':
            if kind=='mutation':object.__setattr__(p,'targets',(.02,-.005,0.))
            elif kind=='nested':solve_translation(fresh,p)
    result=solve_translation(fresh,p,progress=observer)
    assert result.status=='failed' and result.completed_targets==0
    assert result.checkpoint==genesis.checkpoint and not np.any(result.displacements)
    assert _PATH.get() is None and _RUN.get() is None and _ACTIVE.get() is None
    save(tmp_path/'rollback.json',dict(kind=kind,result=packet(result),genesis_preserved=True))


@pytest.mark.parametrize('case',['curved-plastic','connected-plastic'])
def test_geometry_path(case,tmp_path):
    m=make(case);targets=(.03,.06) if case=='curved-plastic' else (.06,.12)
    p=TranslationProgram(targets,3,'ux',pattern(m),SpatialNodalMoments(((3,.015,-.01,.008),)))
    result=solve_translation(m,p);save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
    assert result.status=='completed',result.failure
    fresh=make(case);chain,records=decode_checkpoint(fresh,p,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert encode_checkpoint(fresh,p,chain,records)==result.checkpoint
    last=chain[-1];positive=sum(sum(s.accumulated)>0 for state in last['states'].values() for s in state['response'].history.stations)
    assert positive>0
    recovery={i:recover_native_fields(e,fresh.mesh,last['states'][i],expected_committed_total_u=last['displacements'][list(e.get_dof_mapping(fresh.mesh))]) for i,e in fresh.mesh.elements.items()}
    prefix=encode_checkpoint(fresh,p,chain[:2],records[:1]);again=make(case)
    resumed=solve_translation(again,p,checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest())
    assert resumed.status=='completed' and resumed.checkpoint==result.checkpoint,resumed.failure
    save(tmp_path/'recovery.json',dict(fields=recovery,positive_plastic_stations=positive,byte_identical_resume=True,parameters=[row['parameter'] for row in records]))
