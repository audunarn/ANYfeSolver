"""First actual mixed-owner development gate. Not full G3c confirmation."""
from copy import deepcopy
import json
import threading
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver._ge_beam3_g3c_definition import expand, command, canonical
from anyfesolver._ge_beam3_g3c_owner import MixedGraphOwner

def load(stage,scale=.01):
    return dict(kind='LOAD_STAGE',load_factor=[0.,.5,1.,.25,0.][stage],force_scale=scale,root_stage=stage)

def state(owner): return json.loads(owner.snapshot_bytes())['state']

def invariant(a,b,tol=1e-11):
    a,b=np.asarray(a),np.asarray(b)
    error=np.linalg.norm(a-b)/max(1.,np.linalg.norm(a),np.linalg.norm(b))
    assert np.isfinite(error) and error<=tol

def test_exact_25_definition_expansions_and_anchor_transport():
    names=['J_B2_PAIR','J_B3_PAIR','J_Q4_PAIR','J_S3_PAIR','J_MULTIFAMILY_LOOP']
    variants=['BASE','SHUFFLED_INSERTION','RENUMBERED','CONNECTIVITY_REVERSED','PROPER_GLOBAL_TRANSFORM']
    for name in names:
        _,base,_=expand(name)
        for variant in variants:
            d,e,_=expand(name,variant)
            assert d['fixture_id']==name and d['variant']==variant
            assert len(e['graph']['nodes'])==len(base['graph']['nodes'])
            if variant=='SHUFFLED_INSERTION': assert canonical(e)==canonical(base)
            if variant=='CONNECTIVITY_REVERSED':
                assert e['anchors']==base['anchors']
                assert e['graph']['joints']==base['graph']['joints']
            if variant=='RENUMBERED':
                assert e['anchors']==[dict(element_id=20000+5*a['element_id'],anchor_node=10000+7*a['anchor_node']) for a in base['anchors']]
    for args in [('J_B2_PAIR','UNREGISTERED'),('bad','BASE'),('J_B2_PAIR','BASE','bad')]:
        with pytest.raises(ValueError): expand(*args)

def test_command_copy_order_types_and_preparation_prefix():
    c=load(0); captured=command(c,[],'NONE'); c['root_stage']=4
    assert captured==load(0)
    for bad in [load(1),dict(load(0),root_stage=False),dict(load(0),load_factor=0),dict(load(0),extra=1)]:
        with pytest.raises(ValueError): command(bad,[],'NONE')
    history=[]
    for step in range(1,5):
        c=dict(kind='PREPARE_COMMON_MOTION',step=step)
        assert command(c,history,'CM3')==c; history.append(dict(command=c))
    assert command(load(0),history,'CM3')==load(0)
    with pytest.raises(ValueError): command(load(0),[],'CM3')

def test_b2_pair_zero_equilibrium_commits_actual_native_states():
    owner=MixedGraphOwner(); initial=state(owner)
    assert owner.size==48 and initial['epoch']==0
    assert [r['payload']['epoch'] for r in initial['native_rows']]==[0,0]
    result=owner.solve(load(0)); accepted=state(owner)
    assert result['epoch']==1 and result['production_qualified'] is False
    invariant(result['residual'],np.zeros(72))
    assert [r['payload']['epoch'] for r in accepted['native_rows']]==[1,1]
    for old,new in zip(initial['native_rows'],accepted['native_rows']):
        assert new['payload']['previous_state_sha256']==old['payload']['state_sha256']
    assert accepted['adapter_rows'][0]['source_material_committed'] is False
    assert accepted['adapter_rows'][0]['physical_recovery_complete'] is False

def test_b2_pair_finite_root_and_fresh_native_origin():
    owner=MixedGraphOwner(); owner.solve(load(0)); previous=state(owner)
    result=owner.solve(load(1)); accepted=state(owner)
    assert result['epoch']==2 and np.linalg.norm(result['total_u'])>1e-3
    invariant(result['residual'],np.zeros(72))
    invariant(result['rotations'][0],Rotation.from_rotvec([.12,0.,0.]).as_matrix())
    for old,new in zip(previous['native_rows'],accepted['native_rows']):
        assert new['payload']['epoch']==2
        assert new['payload']['previous_state_sha256']==old['payload']['state_sha256']
        response=new['payload']['response']
        assert np.shape(response['full']['jacobian'])==(42,42)
        assert np.shape(response['lift'])==(24,18)
    # Actual next-origin validation occurs again, without publication.
    saved=owner.snapshot_bytes()
    trial=owner.trial(np.array(accepted['total_u']),np.array(accepted['multipliers']),load(2))
    assert trial['state_committed'] is False and owner.snapshot_bytes()==saved
    assert [r['payload']['epoch'] for r in json.loads(trial['candidate'])['native_rows']]==[3,3]

def test_full_kkt_directional_and_independent_nonidentity_joint_values():
    owner=MixedGraphOwner(); snapshot=owner.snapshot_bytes()
    u=np.linspace(-.002,.003,48); mu=np.tile([.3,-.2,.4,-.1,.25,-.15],4)
    trial=owner.trial(u,mu,load(0)); z=np.r_[u,mu]
    direction=np.sin(np.arange(72)+1.); direction/=np.linalg.norm(direction)
    invariant(trial['tangent'],trial['tangent'].T)
    for h in (1e-4,1e-5,1e-6):
        plus=z+h*direction; minus=z-h*direction
        rp=owner.trial(plus[:48],plus[48:],load(0))['residual']
        rm=owner.trial(minus[:48],minus[48:],load(0))['residual']
        invariant((rp-rm)/(2*h),trial['tangent']@direction,1e-7)
    _,expanded,_=expand('J_B2_PAIR'); g=expanded['graph']; ids=[n for n,_ in g['nodes']]
    X=np.array([x for _,x in g['nodes']],dtype=float); x=X+u.reshape(-1,6)[:,:3]
    q=Rotation.from_rotvec(u.reshape(-1,6)[:,3:]).as_matrix()
    for j,row in enumerate(g['joints']):
        i,k=ids.index(row['master']),ids.index(row['slave'])
        dm0,ds0=np.array(row['master_frame']),np.array(row['slave_frame'])
        dm,ds=q[i]@dm0,q[k]@ds0
        expected=np.r_[dm.T@(x[k]-x[i])-dm0.T@(X[k]-X[i]),
            Rotation.from_matrix((dm0.T@ds0).T@dm.T@ds).as_rotvec()]
        invariant(trial['residual'][60+6*j:66+6*j],expected)
    assert owner.snapshot_bytes()==snapshot

@pytest.mark.parametrize('stage',['native:2','family:11','native_committed','before_publish'])
def test_last_family_and_late_commit_failure_preserve_generation(stage):
    owner=MixedGraphOwner(); saved=owner.snapshot_bytes()
    def fail(where):
        if where==stage: raise RuntimeError('injected '+stage)
    with pytest.raises(RuntimeError,match='injected'): owner.solve(load(0),hook=fail)
    assert owner.snapshot_bytes()==saved
    assert state(owner)['epoch']==0

def test_owned_inputs_and_reentry_at_callback():
    owner=MixedGraphOwner(); u=np.linspace(-.001,.002,48); mu=np.zeros(24); cmd=load(0)
    expected=owner.trial(u,mu,cmd); saved=owner.snapshot_bytes(); seen=[]
    def mutate(stage):
        if stage=='pose':
            u[:]=17.; mu[:]=19.; cmd['root_stage']=4
            with pytest.raises(RuntimeError): owner.solve(load(0))
            seen.append(stage)
    actual=owner.trial(u,mu,cmd,hook=mutate)
    assert seen==['pose']
    invariant(actual['residual'],expected['residual']); invariant(actual['tangent'],expected['tangent'])
    assert actual['candidate']==expected['candidate'] and owner.snapshot_bytes()==saved

def test_changed_trial_capability_never_publishes():
    owner=MixedGraphOwner(); saved=owner.snapshot_bytes()
    def foreign(stage):
        if stage=='native_committed': object.__setattr__(owner,'_active',object())
    with pytest.raises(ValueError,match='capability'): owner.solve(load(0),hook=foreign)
    assert owner.snapshot_bytes()==saved

def test_closed_runtime_dispatch_and_no_restart_admission(monkeypatch):
    owner=MixedGraphOwner()
    with pytest.raises(AttributeError): owner._published=None
    assert not hasattr(owner,'resume') and not hasattr(owner,'checkpoint')
    from anyfesolver import _ge_beam3_g3c_local_beam as beam
    original=beam.LocalBeam.evaluate
    def changed(*args,**kwargs): return original(*args,**kwargs)
    monkeypatch.setattr(beam.LocalBeam,'evaluate',changed)
    with pytest.raises(ValueError,match='dispatch'): owner.solve(load(0))
    monkeypatch.setattr(beam.LocalBeam,'evaluate',original)
    with pytest.raises(ValueError,match='captured'): owner.snapshot_bytes()

def test_replaced_lock_is_rejected_before_publication_and_original_released():
    owner=MixedGraphOwner(); published=owner._published; original=owner._lock
    def replace(stage):
        if stage=='before_publish': object.__setattr__(owner,'_lock',threading.Lock())
    with pytest.raises(ValueError,match='captured'): owner.solve(load(0),hook=replace)
    assert owner._published is published and not original.locked()
    object.__setattr__(owner,'_lock',original)
    with pytest.raises(ValueError,match='captured'): owner.snapshot_bytes()

@pytest.mark.parametrize('module_name',['owner','definition'])
def test_replaced_common_transform_is_rejected_and_poisoned(monkeypatch,module_name):
    from anyfesolver import _ge_beam3_g3c_owner as implementation
    from anyfesolver import _ge_beam3_g3c_definition as definition
    target=implementation if module_name=='owner' else definition
    owner=MixedGraphOwner(common_motion='CM0'); published=owner._published; original=target.SHIFT
    def replace(stage):
        if stage=='pose': monkeypatch.setattr(target,'SHIFT',(99,-3,1))
    with pytest.raises(ValueError): owner.solve(dict(kind='PREPARE_COMMON_MOTION',step=1),hook=replace)
    assert owner._published is published and not owner._owned_lock.locked()
    monkeypatch.setattr(target,'SHIFT',original)
    with pytest.raises(ValueError,match='captured'): owner.snapshot_bytes()

@pytest.mark.parametrize('family',['section','reference'])
def test_imported_class_dispatch_cannot_change(monkeypatch,family):
    from anyfesolver._ge_beam3_g1_elastic import ElasticSection
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    owner=MixedGraphOwner(); published=owner._published
    cls,name=(ElasticSection,'response') if family=='section' else (Reference,'frame')
    original=getattr(cls,name)
    def changed(*a,**kw): return original(*a,**kw)
    monkeypatch.setattr(cls,name,changed)
    with pytest.raises(ValueError,match='dispatch'): owner.solve(load(0))
    assert owner._published is published and not owner._owned_lock.locked()

def test_authority_read_exception_permanently_poisoned(monkeypatch):
    from anyfesolver import _ge_beam3_g3c_definition as definition
    owner=MixedGraphOwner(); published=owner._published; original=definition.bound
    def unreadable(*args): raise ValueError('injected frozen authority read failure')
    monkeypatch.setattr(definition,'bound',unreadable)
    with pytest.raises(ValueError,match='authority read'): owner.snapshot_bytes()
    assert owner._published is published and not owner._owned_lock.locked()
    monkeypatch.setattr(definition,'bound',original)
    with pytest.raises(ValueError,match='captured'): owner.snapshot_bytes()
