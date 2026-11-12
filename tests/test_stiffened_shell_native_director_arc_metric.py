"""Source-only DEV candidate regression specifications; NOT RUN.

Metric/chart cases use constructed poses, not loaded mechanics fixtures. Public
checkpoint cases reuse the existing native BL5 fixture only when later execution
authority is granted. They do not establish capacity or scientific acceptance.
"""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._stiffened_shell_arc_metric import (
    NativeDirectorChordMetric, PhysicalChordMetric, ChordDomainError,
    DIRECTOR_METRIC_ID, require_metric_owner, require_director_chart)
from anyfesolver._stiffened_shell_arc_trial import path_row
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver, _State
from anyfesolver._stiffened_shell_rotation_charts import physical_frames


def poses():
    reference=np.array([rotation([.3,-.2,.1]),rotation([-.1,.2,.3])])
    frames=np.array([rotation([.1,.04,-.02])@reference[0],
                     rotation([-.03,.07,.05])@reference[1]])
    u=np.array([[.04,-.03,.02],[-.01,.02,.03]])
    translations=np.array([[.2,-.1,.3],[-.2,.4,.1]])
    spins=np.array([[.3,.2,-.4],[.1,-.3,.2]])
    return u,frames,reference,translations,spins


def test_owned_weights_exact_types_and_compiled_admission(monkeypatch):
    weights=np.array([.4,.6]);metric=NativeDirectorChordMetric(weights,2.,1)
    weights[:]=9.
    np.testing.assert_array_equal(metric.weights,[.4,.6])
    with pytest.raises(ValueError):metric.weights.setflags(write=True)
    with pytest.raises(FrozenInstanceError):metric.shell_nodes=2
    class ChangedMetric(NativeDirectorChordMetric):pass
    with pytest.raises(ValueError,match='explicit physical arc metric'):
        ss.ArcLengthControl('beam',105,0,ChangedMetric([.4,.6],2.,1),1e-6)
    for count in (True,0,3,1.,np.int64(1)):
        with pytest.raises(ValueError):NativeDirectorChordMetric([.4,.6],2.,count)
    for weights,length in (([],1.),([0.],1.),([np.nan],1.),([1.],0.),
                           ([1.],1e-200),([1.],1e200)):
        with pytest.raises(ValueError):NativeDirectorChordMetric(weights,length,1)
    import anyfesolver._stiffened_shell_affine_endpoint as endpoint
    definition=SimpleNamespace(shell_reference_policy='flat',points=(),footprints=(),
        distributing_footprints=(),finite_distributing_footprints=())
    owner=SimpleNamespace(_shell_ids=(1,),_beam_ids=(2,),_definition=definition)
    monkeypatch.setattr(endpoint,'is_native_material_bl5_arc',lambda owner:True)
    require_metric_owner(owner,metric)
    with pytest.raises(ValueError,match='shell ownership'):
        require_metric_owner(owner,NativeDirectorChordMetric([.4,.6],2.,2))
    with pytest.raises(ValueError,match='weights'):
        require_metric_owner(owner,NativeDirectorChordMetric([1.],2.,1))
    for name,value in (('shell_reference_policy','curved'),('points',(object(),)),
                       ('footprints',(object(),)),('distributing_footprints',(object(),)),
                       ('finite_distributing_footprints',(object(),))):
        original=getattr(definition,name);setattr(definition,name,value)
        with pytest.raises(ValueError,match='flat native material BL5'):require_metric_owner(owner,metric)
        setattr(definition,name,original)
    monkeypatch.setattr(endpoint,'is_native_material_bl5_arc',lambda owner:False)
    with pytest.raises(ValueError,match='flat native material BL5'):require_metric_owner(owner,metric)


def test_shell_current_normal_drill_is_null_and_beam_same_spin_is_measured():
    metric=NativeDirectorChordMetric([.4,.6],2.,1)
    reference=np.array([rotation([.5,-.3,.2]),np.eye(3)])
    d0=reference[0,:,2]
    frames=reference.copy();frames[0]=rotation(2.3*d0)@reference[0]
    value,translation,spin=metric.evaluate(np.zeros((2,3)),frames,reference)
    assert value==pytest.approx(0.,abs=1e-28)
    assert np.linalg.norm(spin)==pytest.approx(0.,abs=1e-14)
    np.testing.assert_array_equal(translation,np.zeros((2,3)))
    # The full-frame metric retains its pi/2 guard even for pure shell drill.
    with pytest.raises(ChordDomainError):PhysicalChordMetric([.4,.6],2.).evaluate(np.zeros((2,3)),frames,reference)
    spins=np.array([2.*d0,np.zeros(3)])
    assert metric.tangent_norm_squared(reference,np.zeros((2,3)),spins)==pytest.approx(0.,abs=1e-28)
    spins[1]=2.*d0
    assert metric.tangent_norm_squared(reference,np.zeros((2,3)),spins)==pytest.approx(2.4,rel=1e-14)
    # A global z spin tilts this rotated shell and is not its drill null.
    spins[:]=0.;spins[0]=[0.,0.,1.]
    assert metric.tangent_norm_squared(reference,np.zeros((2,3)),spins)>0.
    frames=reference.copy();frames[0]=rotation([.1,0.,0.])@frames[0]
    assert metric.evaluate(np.zeros((2,3)),frames,reference)[0]>0.


@pytest.mark.parametrize('plane',(False,True))
def test_common_rigid_rotation_and_actual_spatial_derivative(plane):
    metric=NativeDirectorChordMetric([.4,.6],2.,1)
    u,frames,reference,translations,spins=poses()
    normal=(translations,spins)
    function=lambda u,r,r0:metric.plane(u,r,r0,*normal) if plane else metric.evaluate(u,r,r0)
    value,gt,gr=function(u,frames,reference)
    du=np.array([[.3,.1,-.2],[-.2,.4,.3]])
    omega=np.array([[.2,-.3,.4],[.1,.2,-.1]])
    h=1e-6
    plus=np.array([rotation(h*v)@r for v,r in zip(omega,frames)])
    minus=np.array([rotation(-h*v)@r for v,r in zip(omega,frames)])
    derivative=(function(u+h*du,plus,reference)[0]-function(u-h*du,minus,reference)[0])/(2*h)
    assert derivative==pytest.approx(np.sum(gt*du)+np.sum(gr*omega),rel=2e-8,abs=1e-10)
    common=rotation([.7,-.3,.5])
    rotated=(metric.plane(u@common.T,common@frames,common@reference,
                         translations@common.T,spins@common.T) if plane else
             metric.evaluate(u@common.T,common@frames,common@reference))
    assert rotated[0]==pytest.approx(value,rel=1e-13,abs=1e-14)
    np.testing.assert_allclose(rotated[1],gt@common.T,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(rotated[2],gr@common.T,rtol=1e-13,atol=1e-14)


def test_predictor_norm_equals_origin_plane_derivative_without_projecting_normal():
    metric=NativeDirectorChordMetric([.4,.6],2.,1)
    _,_,reference,translations,spins=poses()
    # Retain a large true predictor drill component in the stored normal.
    spins[0]+=13.*reference[0,:,2]
    full_spin=spins.copy()
    norm=np.sqrt(metric.tangent_norm_squared(reference,translations,spins))
    value,gt,gr=metric.plane(np.zeros((2,3)),reference,reference,translations/norm,spins/norm)
    assert value==0.
    assert np.sum(gt*translations/norm)+np.sum(gr*spins/norm)==pytest.approx(1.,rel=1e-13)
    np.testing.assert_array_equal(spins,full_spin)
    assert np.sum(metric.weights*np.sum(full_spin*full_spin,axis=1))>norm**2
    tilted=reference.copy();tilted[0]=rotation([.04,.03,-.02])@reference[0]
    normal=(translations/norm,spins/norm)
    shifted=normal[1].copy();shifted[0]+=21.*reference[0,:,2]
    assert metric.plane(np.zeros((2,3)),tilted,reference,*normal)[0]==pytest.approx(
        metric.plane(np.zeros((2,3)),tilted,reference,normal[0],shifted)[0],abs=1e-14)


@pytest.mark.parametrize('plane',(False,True))
def test_actual_advance_chart_rows_leave_retained_and_multipliers_unmeasured(plane):
    owner=SimpleNamespace(_shell_count=6,_beam_ids=(1,),_beam_parts=(None,),
        _coordinate_support=None)
    driver=object.__new__(FinitePointDriver);driver.owner=owner;driver._beam_nodal=6
    driver._compensated=False
    q=np.zeros(36);q[3:6]=[.3,-.2,.1]
    anchors=np.array([rotation([-.2,.1,.4])])
    parent=_State(q,np.array([rotation([.2,.3,-.1])]),np.tile(np.eye(3),(1,2,1,1)),
                  np.zeros((1,18)),np.zeros(2),anchors)
    state=driver._advance(parent,np.linspace(-.05,.06,38),1.)
    metric=NativeDirectorChordMetric([.4,.6],2.,1)
    normal=poses()[3:] if plane else None
    _,row=path_row(driver,state,parent,metric,normal)
    direction=np.linspace(.3,-.2,38);h=1e-6
    plus=path_row(driver,driver._advance(state,direction,h),parent,metric,normal)[0]
    minus=path_row(driver,driver._advance(state,direction,-h),parent,metric,normal)[0]
    assert (plus-minus)/(2*h)==pytest.approx(row@direction,rel=2e-8,abs=1e-10)
    np.testing.assert_array_equal(row[12:],np.zeros(26))
    # Mechanics still advances retained coordinates and multipliers.
    assert np.any(state.resultants) and np.any(state.multipliers)
    before_q=state.q.copy();before_frames=state.frames.copy()
    path_row(driver,state,parent,metric,normal)
    np.testing.assert_array_equal(state.q,before_q);np.testing.assert_array_equal(state.frames,before_frames)
    # Pullback uses the actual exp-chart J, rather than treating chart entries as spatial spin.
    from anyfesolver._ge_beam3_pose_joint import _exp_terms
    normal_spin=physical_frames(state.q[:6],anchors)[0,:,2]
    drill=np.zeros(38);drill[3:6]=np.linalg.solve(_exp_terms(state.q[3:6])[1],normal_spin)
    assert row@drill==pytest.approx(0.,abs=1e-14)


def test_domains_malformed_rotations_and_refinement_weight_semantics():
    metric=NativeDirectorChordMetric([.4,.6],2.,1)
    u,frames,reference,translations,spins=poses()
    for bad in (np.full_like(u,np.nan),np.zeros((3,3))):
        with pytest.raises(ValueError):metric.evaluate(bad,frames,reference)
    for bad in (np.full_like(frames,np.inf),-frames,frames*1.1,np.zeros((2,3,2))):
        with pytest.raises(ValueError,match='poses|rotation'):metric.evaluate(u,bad,reference)
    for bad in (np.full_like(spins,np.nan),np.zeros((1,3))):
        with pytest.raises(ValueError,match='directions'):metric.plane(u,frames,reference,translations,bad)
    identity=np.tile(np.eye(3),(2,1,1))
    for node in (0,1):
        outside=identity.copy();outside[node]=rotation([0.,2.,0.])
        with pytest.raises(ChordDomainError):metric.evaluate(u,outside,identity)
    antipodal=identity.copy();antipodal[0]=rotation([np.pi,0.,0.])
    with pytest.raises(ChordDomainError):metric.evaluate(u,antipodal,identity)
    for angle in (2*np.pi,2*np.pi+.1):
        q=np.zeros(6);q[5]=angle
        with pytest.raises(ChordDomainError,match='chart'):require_director_chart(q)
    q=np.zeros(6);q[5]=np.pi+.1;require_director_chart(q)
    one=NativeDirectorChordMetric([1.],2.,1)
    refined=NativeDirectorChordMetric([.25,.75],2.,2)
    assert one.evaluate(u[:1],frames[:1],reference[:1])[0]==pytest.approx(
        refined.evaluate(np.repeat(u[:1],2,axis=0),np.repeat(frames[:1],2,axis=0),
                         np.repeat(reference[:1],2,axis=0))[0],rel=1e-14)
    converted=NativeDirectorChordMetric(metric.weights,2000.,1)
    assert converted.evaluate(1000.*u,frames,reference)[0]==pytest.approx(
        metric.evaluate(u,frames,reference)[0],rel=1e-14)
    assert converted.plane(1000.*u,frames,reference,1000.*translations,spins)[0]==pytest.approx(
        metric.plane(u,frames,reference,translations,spins)[0],rel=1e-14)


def director_control(owner,control):
    return replace(control,metric=NativeDirectorChordMetric(control.metric.weights,control.metric.length,
                                                            len(owner._shell_ids)))


def synthetic_corrector(monkeypatch):
    """Actual _advance with a controlled equilibrium/predictor oracle."""
    from threading import RLock
    from scipy import sparse
    from anyfesolver import _stiffened_shell_arc_trial as arc
    from anyfesolver import _stiffened_shell_arc_checkpoint as checkpoint
    from anyfesolver import _stiffened_shell_affine_endpoint as endpoint
    definition=SimpleNamespace(shell_reference_policy='flat',points=(),footprints=(),
        distributing_footprints=(),finite_distributing_footprints=())
    owner=SimpleNamespace(_shell_ids=(1,),_shell_count=6,_beam_ids=(2,),_beam_parts=(),
        _definition=definition,_K=sparse.eye(12),_free=np.arange(12),_generation=3,
        _commit_lock=RLock(),_bl5_shell_material=False,_coordinate_scale=np.ones(12),
        _constraint_scale=np.zeros(0),_coordinate_support=None)
    driver=object.__new__(FinitePointDriver);driver.owner=owner;driver._beam_nodal=6
    driver._compensated=False
    parent=_State(np.zeros(12),np.array([np.eye(3)]),np.zeros((0,2,3,3)),
                  np.zeros((0,18)),np.zeros(0),None)
    driver._state=parent;driver._accepted=object();driver._shell_origins=();driver._beam_origins=()
    driver._external=None;driver._programme=None;driver._load_factor=0.;driver._pattern=np.zeros(12)
    driver._increment_origin=lambda parent:parent
    assembled=[]
    def assemble(state,*args,**kwargs):
        assembled.append(state)
        return np.ones(12),sparse.eye(12),1.,0.,(),()
    driver._assemble=assemble;owner._finite_driver=driver
    monkeypatch.setattr(endpoint,'is_native_material_bl5_arc',lambda owner:True)
    monkeypatch.setattr(checkpoint,'require_native_seed',lambda *args:None)
    metric=NativeDirectorChordMetric([1.,1.],1.,1)
    kwargs=dict(load_direction=np.r_[1.,np.zeros(11)],metric=metric,predictor_dof=0,
                orientation=1,max_iterations=1,max_backtracks=0)
    return arc,driver,parent,assembled,kwargs


@pytest.mark.parametrize('failure',('chart','hemisphere','beam_increment'))
def test_invalid_initial_predictor_cutbacks_preserve_origin_and_work(monkeypatch,failure):
    arc,driver,parent,assembled,kwargs=synthetic_corrector(monkeypatch)
    step=np.zeros(12);step[0]=1.
    if failure=='chart':step[5]=30.
    elif failure=='hemisphere':step[3]=4.
    else:step[9]=3.
    monkeypatch.setattr(arc,'bordered_correction',lambda *args:(step.copy(),0.))
    # Shell drill is not projected out. The beam case uses a larger radius to
    # reach the unchanged _advance guard after full beam norm normalization.
    radius=4. if failure=='beam_increment' else 2. if failure=='hemisphere' else 1.
    with pytest.raises(arc.ArcTrialFailure) as caught:arc.correct(driver,radius=radius,**kwargs)
    trial=caught.value.trial
    assert trial.state is parent and trial.evaluation[2]==1.
    assert trial.evaluations==1 and trial.linear_solves==1 and trial.rejected_trials==1
    assert trial.path_error==-1. and trial.amplitude==0.
    assert assembled==[parent] and driver._state is parent
    if failure=='chart':
        assembled.clear()
        with pytest.raises(arc.ArcSelectionFailure) as selected:
            arc.select_trial(driver,radius=radius,max_attempts=2,max_cutbacks=1,**kwargs)
        assert len(selected.value.attempts)==2
        assert sum(row.evaluations for row in selected.value.attempts)==2
        assert sum(row.rejected_trials for row in selected.value.attempts)==2
        assert len(assembled)==2 and all(state is parent for state in assembled)


def test_invalid_accepted_chart_refused_before_assembly_and_newton_candidate_before_assembly(monkeypatch):
    arc,driver,parent,assembled,kwargs=synthetic_corrector(monkeypatch)
    invalid=parent.q.copy();invalid[5]=2*np.pi
    driver._state=replace(parent,q=invalid)
    with pytest.raises(ChordDomainError):arc.correct(driver,radius=.1,**kwargs)
    assert assembled==[]
    driver._state=parent
    step=np.zeros(12);step[0]=1.
    correction=np.zeros(12);correction[5]=7.
    monkeypatch.setattr(arc,'bordered_correction',lambda *args:(step.copy(),0.))
    monkeypatch.setattr(arc,'bordered_path_correction',lambda *args:(correction.copy(),0.))
    with pytest.raises(arc.ArcTrialFailure,match='backtrack') as caught:
        arc.correct(driver,radius=.1,**kwargs)
    # Origin and valid predictor were evaluated; rejected Newton chart was not.
    assert len(assembled)==2 and caught.value.trial.evaluations==2
    assert caught.value.trial.linear_solves==2 and caught.value.trial.rejected_trials==1
    assert driver._state is parent


def test_public_director_history_restart_and_atomic_refusals(request,tmp_path,monkeypatch):
    import os
    from copy import deepcopy
    from test_stiffened_shell_native_material_bl5_arc import analysis,exact,_exact_value
    from test_native_material_bl5_arc_same_origin_jacobian import _loaded_seed
    from examples.stiffened_shell_native_fibre_bl5_mixed_j2_displacement import definition
    from examples.stiffened_shell_native_fibre_bl5_displacement import ACTIONS
    from examples.stiffened_shell_native_material_bl5_arc import arc_control
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver._stiffened_shell_arc_trial import ArcSelectionFailure
    from anyfesolver import _stiffened_shell_arc_control as controller
    from anyfesolver import _stiffened_shell_arc_publication as publication
    from anyfesolver._stiffened_shell_arc_checkpoint import (
        decode_snapshot,encode_snapshot,director_policy,metric_from_snapshot,requires_coordinate_progress)
    def capture(owner):
        driver=owner._finite_driver
        return SimpleNamespace(raw=owner.checkpoint(),recovery=deepcopy(owner.recover()),
            driver=driver,state=driver._state,shell_history=driver._shell_origins,
            beam_history=driver._beam_origins,accepted=driver._accepted,
            arc_snapshot=getattr(owner,'_arc_snapshot',None))

    def unchanged(owner,before):
        assert owner.checkpoint()==before.raw and owner._finite_driver is before.driver
        assert before.driver._state is before.state and before.driver._accepted is before.accepted
        assert before.driver._shell_origins is before.shell_history
        assert before.driver._beam_origins is before.beam_history
        assert getattr(owner,'_arc_snapshot',None) is before.arc_snapshot
        # Compare every public field, including status, amplitude and scalars.
        # The copied witness also detects mutation through shared record arrays.
        _exact_value(owner.recover(),before.recovery)
    # This test deliberately requires the existing authentic fixture. It never
    # uses the helper's optional fresh-load generation branch.
    if os.environ.get('ANY_NATIVE_BL5_ARC_JACOBIAN_FIXTURE') is None:
        pytest.skip('local loaded BL5 checkpoint evidence is unavailable; '
            'set ANY_NATIVE_BL5_ARC_JACOBIAN_FIXTURE to exercise recovery')
    model=definition('mixed','biaxial','mixed','continuous_p1_coarse')
    owner=analysis(request,model);_loaded_seed(owner,tmp_path)
    seed_raw=owner.checkpoint();old=arc_control(owner)
    control=director_control(owner,old)
    before=capture(owner)
    token=CancellationToken();token.cancel('before director arc')
    with pytest.raises(SolveCancelled):owner.continue_arc_length(ACTIONS,control,cancellation_token=token)
    unchanged(owner,before)
    def fail(*args,**kwargs):raise ArcSelectionFailure('injected selection failure',())
    with monkeypatch.context() as patch:
        patch.setattr(controller,'select_trial',fail)
        with pytest.raises(ArcSelectionFailure):owner.continue_arc_length(ACTIONS,control)
    unchanged(owner,before)
    token=CancellationToken();safe=publication.cancellation_safe_point
    def stop(token,stage):
        if stage=='stiffened-shell.arc.commit':token.cancel('before publication')
        return safe(token,stage)
    with monkeypatch.context() as patch:
        patch.setattr(publication,'cancellation_safe_point',stop)
        with pytest.raises(SolveCancelled):owner.continue_arc_length(ACTIONS,control,cancellation_token=token)
    unchanged(owner,before)
    result=owner.continue_arc_length(ACTIONS,control)
    assert result.status=='completed'
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    assert m['policy']==director_policy(owner) and m['metric_kind']==DIRECTOR_METRIC_ID
    assert m['metric_shell_nodes']==len(owner._shell_ids) and requires_coordinate_progress(owner)
    assert type(metric_from_snapshot(owner,m,a)) is NativeDirectorChordMetric
    assert metric_from_snapshot(owner,m,a).tangent_norm_squared(a['origin_frames'],
        a['normal_translations'],a['normal_spins'])==pytest.approx(1.,abs=1e-10)
    fresh=analysis(request,model);restored=fresh.restore(raw)
    assert restored.status=='accepted_prefix'
    exact(restored,result)
    assert fresh.checkpoint()==raw
    exact(owner.continue_arc_length(ACTIONS,control),fresh.continue_arc_length(ACTIONS,control))
    assert owner.checkpoint()==fresh.checkpoint()
    before=capture(owner)
    with pytest.raises(ValueError,match='metric identity'):owner.continue_arc_length(ACTIONS,old)
    unchanged(owner,before)
    for metadata in (dict(m,metric_kind='unknown'),dict(m,metric_shell_nodes=m['metric_shell_nodes']+1),
                     {k:v for k,v in m.items() if k!='metric_kind'},
                     dict(m,policy=m['policy'].replace('POSITION_DIRECTOR_',''))):
        with pytest.raises(ValueError):encode_snapshot(owner,metadata,a)
        unchanged(owner,before)
    forged=dict(a);forged['normal_translations']=a['normal_translations']*2.
    forged['normal_spins']=a['normal_spins']*2.
    with pytest.raises(ValueError,match='normal'):encode_snapshot(owner,m,forged)
    unchanged(owner,before)
    damaged=bytearray(raw);damaged[-1]^=1
    with pytest.raises(ValueError,match='digest'):decode_snapshot(owner,bytes(damaged))
    unchanged(owner,before)
    # A full-frame prefix remains full-frame and cannot adopt this opt-in.
    legacy=analysis(request,model);legacy.restore(seed_raw);legacy_control=arc_control(legacy)
    legacy.continue_arc_length(ACTIONS,legacy_control)
    legacy_before=capture(legacy);legacy_raw=legacy_before.raw
    lm,la=decode_snapshot(legacy,legacy_raw)
    assert 'metric_kind' not in lm and type(metric_from_snapshot(legacy,lm,la)) is PhysicalChordMetric
    with pytest.raises(ValueError,match='metric identity'):
        legacy.continue_arc_length(ACTIONS,director_control(legacy,legacy_control))
    unchanged(legacy,legacy_before)
    # Even a correctly bound predecessor cannot publish a cross-metric header.
    from hashlib import sha256
    live_metadata=dict(m,generation=owner._generation+1,
        predecessor_sha256=sha256(owner.checkpoint()).hexdigest(),metric_kind='crossed')
    before=capture(owner);live=before.driver
    with pytest.raises(ValueError,match='metric history identities'):
        publication._bind_predecessor(owner,live_metadata,a,live,owner._generation)
    unchanged(owner,before)
    # Cancellation after the first accepted step keeps that accepted prefix.
    token=CancellationToken();prefix=[]
    def accepted(result):
        prefix.append(capture(owner));token.cancel('after prefix')
    with pytest.raises(SolveCancelled):
        owner.continue_arc_length(ACTIONS,control,steps=2,cancellation_token=token,on_accepted=accepted)
    assert len(prefix)==1 and prefix[0].recovery.status=='accepted_prefix'
    unchanged(owner,prefix[0])
