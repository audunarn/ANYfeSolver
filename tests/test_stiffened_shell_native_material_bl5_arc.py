"""Planned native BL5 arc regression coverage; source candidate, NOT RUN."""
from dataclasses import replace,fields,is_dataclass
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_arc_checkpoint import (
    identities,decode_snapshot,encode_snapshot,NATIVE_BL5_CHART_POLICY)
from anyfesolver._stiffened_shell_beam_history import arrays as beam_arrays
from examples.stiffened_shell_native_fibre_bl5_mixed_j2_displacement import definition
from examples.stiffened_shell_native_fibre_bl5_displacement import ACTIONS,CONTROL,OPTIONS,TARGETS
from examples.stiffened_shell_native_material_bl5_arc import arc_control


def analysis(request,model):
    owner=ss.create_analysis(definition=model)
    request.addfinalizer(owner.close)
    return owner


def seeded(request,shell='mixed',seed='translation'):
    basis='continuous_p1_coarse' if shell=='mixed' and seed=='translation' else 'p0'
    model=definition(shell,'biaxial','mixed',basis);owner=analysis(request,model)
    owner.solve(ACTIONS,(.001,),**OPTIONS)
    if seed=='translation':owner.solve(ACTIONS,TARGETS[:2],displacement_control=CONTROL,**OPTIONS)
    return model,owner,arc_control(owner)


def exact(left,right):
    assert left.history==right.history and left.amplitude==right.amplitude
    assert left.shell_layer_alpha==right.shell_layer_alpha
    assert left.beam_station_accumulated==right.beam_station_accumulated
    assert left.beam_fibre_energy==right.beam_fibre_energy
    _exact_value(left.beam_generalized_material,right.beam_generalized_material)
    for name in ('shell_displacement','beam_positions','beam_frames','cell_frames',
                 'cell_resultants','multipliers','support_reactions','applied_dead_load',
                 'shell_chart_coordinates','shell_rotation_anchors'):
        np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    for name in ('shell_q4_material','shell_s3_material','beam_fibre_material',
                 'beam_fibre_energy','beam_shell_line_work','point_wrenches'):
        _exact_value(getattr(left,name),getattr(right,name))


def _exact_value(left,right):
    if isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):_exact_value(getattr(left,field.name),getattr(right,field.name))
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True):_exact_value(a,b)
    else:assert left==right


@pytest.mark.parametrize('shell',('s3','mixed'))
@pytest.mark.parametrize('seed',('force','translation'))
def test_native_material_bl5_arc_restart_histories_and_retained_state(request,monkeypatch,shell,seed):
    model,owner,control=seeded(request,shell,seed)
    predecessor=dict(beam_arrays(owner._beam_parts,owner._finite_driver._beam_origins,'beam_origin'))
    if seed=='translation':
        assert np.any(predecessor['beam_origin_fibres'][:,2:]>0.)
    if seed=='translation' and shell=='mixed':
        # Matched mixed/biaxial/P1 fixture: actual gen3 loaded endpoint yields S3.
        # Other force or P0 cases have no simultaneous-yield premise.
        assert any(np.any(row.alpha>0.) for row in owner.recover().shell_s3_material)
    import anyfesolver._stiffened_shell_branch_event_trial as events
    def forbidden(*args,**kwargs):raise AssertionError('native arc reached generalized event-cell selection')
    monkeypatch.setattr(events,'trial_fraction',forbidden)
    seen=[];result=owner.continue_arc_length(ACTIONS,control,on_accepted=seen.append)
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    for name,value in predecessor.items():np.testing.assert_array_equal(a[name],value)
    assert identities(owner)==('finite_arc_native_fibre_bl5_mixed_shell',
        'B3_GE_NATIVE_FIBRE_BL5_MIXED_SHELL_J2_TANGENT_PLANE_V1_DEV')
    assert m['native_chart_policy']==NATIVE_BL5_CHART_POLICY
    assert seen[0].status=='accepted_prefix' and result.status=='completed'
    assert result.shell_s3_material and result.beam_fibre_material and result.beam_generalized_material
    assert bool(result.shell_q4_material)==(shell=='mixed')
    assert len(a['s3_states'])==2*len(model.shell.s3_elements)
    assert result.cell_frames.size and result.cell_resultants.size and result.multipliers.size
    fresh=analysis(request,model);restored=fresh.restore(raw)
    exact(restored,result);assert fresh.checkpoint()==raw
    final=owner.continue_arc_length(ACTIONS,control)
    repeated=fresh.continue_arc_length(ACTIONS,control)
    exact(final,repeated);assert owner.checkpoint()==fresh.checkpoint()
    for name,value in beam_arrays(owner._beam_parts,owner._finite_driver._beam_origins,'beam'):
        np.testing.assert_array_equal(value,dict(beam_arrays(fresh._beam_parts,
            fresh._finite_driver._beam_origins,'beam'))[name])
    assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
    assert not final.production_qualified


def test_atomic_cancel_failure_and_prefix(request,monkeypatch):
    model,owner,control=seeded(request)
    before=owner.checkpoint();driver=owner._finite_driver
    token=CancellationToken();token.cancel('before arc')
    with pytest.raises(SolveCancelled):owner.continue_arc_length(ACTIONS,control,cancellation_token=token)
    assert owner.checkpoint()==before and owner._finite_driver is driver
    import anyfesolver._stiffened_shell_arc_control as controller
    from anyfesolver._stiffened_shell_arc_trial import ArcSelectionFailure
    def fail(*args,**kwargs):raise ArcSelectionFailure('injected bounded selection failure',())
    with monkeypatch.context() as patch:
        patch.setattr(controller,'select_trial',fail)
        with pytest.raises(ArcSelectionFailure):owner.continue_arc_length(ACTIONS,control)
    assert owner.checkpoint()==before and owner._finite_driver is driver
    token=CancellationToken();prefix=[]
    def stop(result):prefix.append(owner.checkpoint());token.cancel('after prefix')
    with pytest.raises(SolveCancelled):
        owner.continue_arc_length(ACTIONS,control,steps=2,cancellation_token=token,on_accepted=stop)
    assert len(prefix)==1 and owner.checkpoint()==prefix[0]
    import anyfesolver._stiffened_shell_arc_publication as publication
    before=owner.checkpoint();driver=owner._finite_driver;token=CancellationToken()
    safe=publication.cancellation_safe_point
    def cancel_at_commit(value,stage):
        if stage=='stiffened-shell.arc.commit':token.cancel('atomic commit')
        return safe(value,stage)
    with monkeypatch.context() as patch:
        patch.setattr(publication,'cancellation_safe_point',cancel_at_commit)
        with pytest.raises(SolveCancelled):owner.continue_arc_length(ACTIONS,control,cancellation_token=token)
    assert owner.checkpoint()==before and owner._finite_driver is driver


def test_checkpoint_refusals_and_mutation(request):
    model,owner,control=seeded(request);owner.continue_arc_length(ACTIONS,control)
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    for metadata in (dict(m,policy='B3_GE_NATIVE_POINT_COUPLED_J2_TANGENT_PLANE_V2_DEV'),
                     dict(m,native_chart_policy='unbound'),dict(m,model='wrong')):
        with pytest.raises(ValueError):encode_snapshot(owner,metadata,a)
    for name in ('beam_origin_fibres','shell_origin_q'):
        forged=dict(a);forged[name]=a[name].copy();forged[name].flat[-1]+=.001
        corrupt=encode_snapshot(owner,m,forged);fresh=analysis(request,model)
        with pytest.raises(ValueError):fresh.restore(corrupt)
        assert fresh._finite_driver is None and fresh._generation==0
    with pytest.raises(ValueError):owner.continue_arc_length(ACTIONS,replace(control,radius=2*control.radius))
    with pytest.raises(ValueError):owner.rebase_rotations(owner._generation)
    with pytest.raises(ValueError):owner.solve(ACTIONS,(.1,),**OPTIONS)
    assert owner.checkpoint()==raw
    for flag in ('_programme','_external','_adaptive_state'):
        _,seed_owner,seed_control=seeded(request,seed='force')
        saved=seed_owner.checkpoint();setattr(seed_owner._finite_driver,flag,object())
        with pytest.raises(ValueError):seed_owner.continue_arc_length(ACTIONS,seed_control)
        setattr(seed_owner._finite_driver,flag,None)
        assert seed_owner.checkpoint()==saved


def test_older_q4_native_material_identity_and_history_layout(request):
    from examples.stiffened_shell_native_fibre_bl5_material import definition as q4_definition
    for shell_material in (False,True):
        model=q4_definition(family='biaxial',shell_material=shell_material)
        owner=analysis(request,model);owner.solve(ACTIONS,(.001,),**OPTIONS)
        control=arc_control(owner);result=owner.continue_arc_length(ACTIONS,control)
        assert identities(owner)==('finite_arc_native_fibre_bl5',
            'B3_GE_NATIVE_FIBRE_BL5_TANGENT_PLANE_V1_DEV')
        assert result.beam_fibre_material and not result.shell_s3_material
        assert bool(result.shell_q4_material)==shell_material
        fresh=analysis(request,model);fresh.restore(owner.checkpoint())
        assert fresh.checkpoint()==owner.checkpoint()


def test_unadmitted_native_history_flag_does_not_admit_arc(request):
    model=definition();owner=analysis(request,model)
    from anyfesolver._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
    from anyfesolver._stiffened_shell_arc_checkpoint import identities as arc_identities
    assert is_native_material_bl5_arc(owner)
    owner._finite_bl5=False
    assert not is_native_material_bl5_arc(owner)
    with pytest.raises(ValueError):arc_identities(owner)


def test_full_frame_metric_measures_normal_spin_and_is_objective():
    theta=.2;weight=2.;metric=ss.PhysicalChordMetric(np.array([weight]),1.)
    reference=np.eye(3)[None,:,:];frames=Rotation.from_rotvec((0.,0.,theta)).as_matrix()[None,:,:]
    delta=np.zeros((1,3))
    value,_,_=metric.evaluate(delta,frames,reference)
    assert value==pytest.approx(2*weight*(1-np.cos(theta)))
    np.testing.assert_array_equal(frames[0,:,2],reference[0,:,2])
    assert value>0. # Same position/director, positive full-frame distance.
    rotation=Rotation.from_rotvec((.3,-.1,.2)).as_matrix()
    rotated,_,_=metric.evaluate(delta@rotation.T,rotation@frames,rotation@reference)
    assert rotated==pytest.approx(value)


def test_actual_update_path_row_and_full_retained_border(request):
    from anyfesolver._stiffened_shell_arc_trial import path_row
    from anyfesolver._stiffened_shell_displacement_corrector import bordered_path_correction
    _,owner,control=seeded(request,seed='force');driver=owner._finite_driver
    parent=driver._state;n=len(parent.q);size=n+len(parent.multipliers)
    free=np.r_[owner._free,np.arange(n,size)]
    step=np.zeros(size);step[free]=.01*np.sin(np.arange(len(free))+1.)
    state=driver._advance(parent,step,.05)
    nodes=len(control.metric.weights)
    normal=(np.tile((.3,.2,-.1),(nodes,1)),np.tile((-.2,.4,.1),(nodes,1)))
    value,row=path_row(driver,state,parent,control.metric,normal)
    h=1e-5
    plus=path_row(driver,driver._advance(state,step,h),parent,control.metric,normal)[0]
    minus=path_row(driver,driver._advance(state,step,-h),parent,control.metric,normal)[0]
    assert (plus-minus)/(2*h)==pytest.approx(row@step,rel=1e-8,abs=1e-10)
    nodal=owner._shell_count+6*len(owner._beam_ids)
    np.testing.assert_array_equal(row[nodal:],np.zeros(size-nodal))
    assert np.any(state.cells!=parent.cells) and np.any(state.resultants!=parent.resultants)
    assert np.any(state.multipliers!=parent.multipliers)
    applied=driver._load_factor*driver._pattern
    evaluation=driver._assemble(parent,0.,driver._shell_origins,driver._beam_origins,applied_load=applied)
    matrix=evaluation[1];direction=np.r_[driver._load(ACTIONS),np.zeros(size-n)]
    # Manufactured full-system RHS exercises the actual assembled retained block,
    # not a condensed physical-only tangent or a second constitutive formulation.
    expected=.001*step;amplitude=.003
    residual=-(matrix@expected-direction*amplitude);gap=-float(row@expected)
    actual,load=bordered_path_correction(matrix,residual,direction,row,gap,free)
    scale=max(float(np.max(abs(residual),initial=0.)),np.finfo(float).tiny)
    assert np.max(abs((matrix@actual-load*direction+residual)[free]),initial=0.)/scale<=1e-8
    assert row@actual==pytest.approx(-gap,rel=1e-8,abs=1e-10)
