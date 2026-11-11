"""Tilted material references and experimental curved point histories."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as ss
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import create_shell_element
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import response,local_coordinates
from test_stiffened_shell_q4_algorithmic_pullback import yielded,_local,_rigid
from test_stiffened_shell_ge_finite_experimental import _q4_shell_j2_panel,_actions
from test_stiffened_shell_curved_reference_seam import curved_panel
from anyfesolver.control import CancellationToken,SolveCancelled


def assert_generalized_records_equal(left,right):
    from dataclasses import fields
    assert len(left)==len(right)
    for a,b in zip(left,right):
        assert a.element_id==b.element_id and len(a.stations)==len(b.stations)
        for x,y in zip(a.stations,b.stations):
            for field in fields(x):
                u,v=getattr(x,field.name),getattr(y,field.name)
                if isinstance(u,np.ndarray):np.testing.assert_array_equal(u,v)
                else:assert u==v


def curved_material_panel():
    geometry=curved_panel();material=_q4_shell_j2_panel();beams=[]
    for original,curved in zip(material.beams,geometry.beams):
        element,inertia=original.instantiate();shape,_=curved.instantiate()
        rebuilt=type(element)(element.element_id,element.node_ids,
            shape.operator.reference,element.section,order=element.operator.order,
            arithmetic_policy=element.operator.arithmetic_policy)
        beams.append(type(original).capture(rebuilt,inertia))
    return replace(geometry,shell=replace(geometry.shell,j2_curve=material.shell.j2_curve),
        beams=tuple(beams),shell_reference_policy=ss.CURVED_Q4_POINT_MATERIAL_REFERENCE_V1)


def make_owner(request,definition):
    value=ss.create_analysis(definition=definition);request.addfinalizer(value.close);return value


def test_curved_shell_beam_yield_unload_restart_and_rejected_commit(request,monkeypatch,tmp_path):
    definition=curved_material_panel();owner=make_owner(request,definition)
    result=owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    assert max(max(v) for v in result.shell_layer_alpha)>0.
    assert max(max(v) for v in result.beam_station_accumulated)>0.
    assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
    saved=owner.checkpoint();fresh=make_owner(request,definition);restored=fresh.restore(saved)
    assert fresh.checkpoint()==saved
    np.testing.assert_array_equal(restored.shell_layer_alpha,result.shell_layer_alpha)
    np.testing.assert_array_equal(restored.beam_station_accumulated,result.beam_station_accumulated)
    assert result.beam_generalized_material
    assert_generalized_records_equal(restored.beam_generalized_material,result.beam_generalized_material)
    for target in (.9,1.1):
        backtracks=16 if target==1.1 else 12
        try:
            actual=owner.solve(_actions(),(target,),max_iterations=20,max_backtracks=backtracks)
        except Exception as error:
            import json
            from dataclasses import asdict
            if hasattr(error,'attempts'):
                (tmp_path/'failed-attempts.json').write_text(json.dumps([asdict(a) for a in error.attempts],indent=2))
            raise
        repeated=fresh.solve(_actions(),(target,),max_iterations=20,max_backtracks=backtracks)
        assert owner.checkpoint()==fresh.checkpoint()
        for left,right in zip(actual.shell_q4_material,repeated.shell_q4_material):
            for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
                np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
        np.testing.assert_array_equal(actual.cell_resultants,repeated.cell_resultants)
        assert_generalized_records_equal(actual.beam_generalized_material,repeated.beam_generalized_material)
    prefix=owner.checkpoint();generation=owner._generation;token=CancellationToken()
    driver=owner._finite_driver;publish=driver._publish_equilibrium
    def cancelled(*args,**kwargs):
        token.cancel('after trial material advancement, before commit')
        return publish(*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(driver,'_publish_equilibrium',cancelled)
        with pytest.raises(SolveCancelled):
            owner.solve(_actions(),(1.2,),max_iterations=20,max_backtracks=16,cancellation_token=token)
    assert owner._generation==generation and owner.checkpoint()==prefix


def test_old_curved_material_guards_and_new_policy_limits(request):
    d=curved_material_panel()
    with pytest.raises(ValueError,match='elastic Q4'):
        ss.create_analysis(definition=replace(d,shell_reference_policy=ss.CURVED_Q4_POINT_REFERENCE_V1))
    owner=make_owner(request,d);cap=owner.capabilities
    assert 'J2' in cap['shell'] and cap['arc_length']=='experimental_curved_material_point_tangent_plane_v1'
    assert 'MATERIAL_REFERENCE_TRANSPORT' in cap['stress_free_reference_construction']
    assert not owner.reference_modal_capabilities['eligible']
    with pytest.raises(ValueError,match='reference definition'):
        ss.create_stress_free_point_reference(replace(d,q4_pullback_policy='historical_corotational'),())
    with pytest.raises(ValueError,match='dead translations'):
        owner.solve((ss.NodalAction('beam',105,(0.,0.,1.,0.,1.,0.)),),(.1,))
    assert owner._generation==0 and owner._finite_driver is None
    control=ss.ArcLengthControl('beam',105,2,ss.PhysicalChordMetric(np.ones(11),1.),1e-4,1)
    with pytest.raises(ValueError,match='accepted coupled'):
        owner.continue_arc_length(_actions(),control)
    with pytest.raises(ValueError,match='curved material displacement control'):
        owner.solve(_actions(),(.1,),displacement_control=object())
    from anyfesolver._stiffened_shell_arc_checkpoint import MAGIC as ARC_MAGIC
    from anyfesolver._stiffened_shell_displacement_checkpoint import MAGIC as DISPLACEMENT_MAGIC
    from anyfesolver._stiffened_shell_displacement_adaptive import MAGIC as ADAPTIVE_MAGIC,EVENT_MAGIC
    with pytest.raises(ValueError,match='arc snapshot'):
        owner.restore(ARC_MAGIC)
    assert cap['displacement_control']=='experimental_curved_material_point_translation_v1'
    for magic in (DISPLACEMENT_MAGIC,ADAPTIVE_MAGIC,EVENT_MAGIC):
        with pytest.raises(ValueError):
            owner.restore(magic)
    assert owner._generation==0 and owner._finite_driver is None
    assert cap['fixed_step_max_backtracks']==16 and cap['adaptive_max_backtracks']==12
    with pytest.raises(ValueError,match='bounded'):
        owner.solve(_actions(),(.1,),max_backtracks=17)
    with pytest.raises(ValueError,match='adaptive targets and Newton controls'):
        owner.solve(_actions(),(.1,),max_backtracks=16,increment_control=ss.AdaptiveIncrementControl())


def test_curved_material_rebase_preserves_and_audits_histories(request,monkeypatch):
    owner=make_owner(request,curved_material_panel())
    accepted=owner.solve(_actions(),(.5,1.,.9),max_iterations=20,max_backtracks=12)
    driver=owner._finite_driver;assemble=driver._assemble;prefix=owner.checkpoint()
    def changed_history(*args,**kwargs):
        values=list(assemble(*args,**kwargs))
        shell=list(values[4]);bad=dict(shell[0]);bad['alpha']=np.array(bad['alpha'])+1e-3
        shell[0]=bad;values[4]=tuple(shell);return tuple(values)
    with monkeypatch.context() as patch:
        patch.setattr(driver,'_assemble',changed_history)
        with pytest.raises(ValueError,match='changed shell material history'):
            owner.rebase_rotations(owner._generation)
    assert owner.checkpoint()==prefix
    actual=owner.rebase_rotations(owner._generation)
    np.testing.assert_array_equal(actual.shell_layer_alpha,accepted.shell_layer_alpha)
    np.testing.assert_array_equal(actual.beam_station_accumulated,accepted.beam_station_accumulated)
    assert_generalized_records_equal(actual.beam_generalized_material,accepted.beam_generalized_material)
    for before,after in zip(accepted.shell_q4_material,actual.shell_q4_material):
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_allclose(getattr(before,name),getattr(after,name),rtol=1e-10,atol=1e-8)
    saved=owner.checkpoint();fresh=make_owner(request,owner._definition);fresh.restore(saved)
    assert fresh.checkpoint()==saved
    assert_generalized_records_equal(fresh.recover().beam_generalized_material,actual.beam_generalized_material)
    live=owner.solve(_actions(),(.95,),max_iterations=20,max_backtracks=16)
    replay=fresh.solve(_actions(),(.95,),max_iterations=20,max_backtracks=16)
    assert owner.checkpoint()==fresh.checkpoint()
    np.testing.assert_array_equal(live.cell_resultants,replay.cell_resultants)


@pytest.fixture(scope='module')
def tilted_material():
    flat=yielded.__wrapped__()
    model,material,element,reference,origin,trial=flat
    rotation=Rotation.from_rotvec((.3,-.4,.2));matrix=rotation.as_matrix()
    target=FEModel('tilted-q4-material-reference')
    target_material=target.add_material('shell',material.elastic_modulus,material.poisson_ratio,density=7850.,
        hardening_curve=material.hardening_curve)
    xyz=reference@matrix.T+(.1,-.2,.3)
    for n,p in enumerate(xyz,1):target.add_node(n,*p)
    q4=create_shell_element(1,(1,2,3,4),'shell',formulation='e4-pl',
        thickness=.02,reference_normal=tuple(matrix[:,2]))
    target.add_element(1,q4)
    accepted=np.zeros((4,6));accepted[1,0]=accepted[2,0]=.004
    accepted[1,2]=.0004;accepted[2,2]=.0007
    accepted=accepted.reshape(-1,3)@matrix.T
    local=local_coordinates(target,q4,accepted.ravel())
    _,_,candidate=q4.compute_nonlinear_response(target.mesh,target_material,
        local,q4.init_nonlinear_state(3),3,True)
    predecessor=q4.seal_committed_current_tangent_state(target.mesh,target_material,
        local,candidate,3)
    moved=(trial.reshape(-1,3)@matrix.T).ravel()
    return flat,(target,target_material,q4,xyz,predecessor,moved),rotation


def test_tilted_material_reference_covariance_and_fixed_predecessor(tilted_material):
    flat,tilted,rotation=tilted_material
    fm,_,fe,_,fp,fu=flat;tm,_,te,_,tp,tu=tilted
    before=canonical_json_bytes(tp)
    force,_,state=response(fm,fe,fu,fp,3)
    actual,_,candidate=response(tm,te,tu,tp,3)
    np.testing.assert_allclose(actual.reshape(-1,3),
        force.reshape(-1,3)@rotation.as_matrix().T,rtol=1e-9,atol=2e-6)
    for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
        np.testing.assert_allclose(candidate[name],state[name],rtol=1e-9,atol=2e-6)
    assert canonical_json_bytes(tp)==before


@pytest.mark.parametrize('anchored',(False,True))
def test_tilted_yielded_actual_update_derivative_work_and_rebase(tilted_material,anchored):
    _,(model,material,element,reference,origin,trial),_=tilted_material
    _,trial=_rigid(reference,trial);before=canonical_json_bytes(origin)
    anchors=None
    if anchored:
        anchors=Rotation.from_rotvec(trial.reshape(4,6)[:,3:].copy()).as_matrix()
        trial=trial.copy();trial.reshape(4,6)[:,3:]=(.001,-.002,.003)
    force,tangent,candidate=response(model,element,trial,origin,3,rotation_anchors=anchors)
    active=candidate['alpha']>origin['alpha'];assert active.any()
    local=_local(reference,trial,anchors)
    local_force=element.compute_nonlinear_response(model.mesh,material,local,origin,3,True)[0]
    for direction in (np.sin(np.arange(24)+.3),np.cos(.4+.73*np.arange(24))):
        direction/=np.linalg.norm(direction);predicted=tangent@direction
        for h in (2e-7,1e-7):
            plus,_,a=response(model,element,trial+h*direction,origin,3,rotation_anchors=anchors)
            minus,_,b=response(model,element,trial-h*direction,origin,3,rotation_anchors=anchors)
            np.testing.assert_array_equal(a['alpha']>origin['alpha'],active)
            np.testing.assert_array_equal(b['alpha']>origin['alpha'],active)
            assert np.linalg.norm((plus-minus)/(2*h)-predicted)/np.linalg.norm(predicted)<=1e-5
        h=1e-7
        velocity=(_local(reference,trial+h*direction,anchors)-
                  _local(reference,trial-h*direction,anchors))/(2*h)
        rows=trial.reshape(4,6);spatial=direction.reshape(4,6).copy()
        plus=Rotation.from_rotvec(rows[:,3:]+h*spatial[:,3:])
        minus=Rotation.from_rotvec(rows[:,3:]-h*spatial[:,3:])
        spatial[:,3:]=(plus*minus.inv()).as_rotvec()/(2*h)
        expected=float(local_force@velocity);actual=float(force@spatial.ravel())
        assert abs(actual-expected)/max(1.,abs(actual),abs(expected))<=1e-8
    physical=Rotation.from_rotvec(trial.reshape(4,6)[:,3:].copy()).as_matrix()
    if anchors is not None:physical=physical@anchors
    reset=trial.copy();reset.reshape(4,6)[:,3:]=0.
    repeated,_,state=response(model,element,reset,origin,3,rotation_anchors=physical)
    np.testing.assert_allclose(repeated,force,rtol=1e-9,atol=2e-6)
    for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
        np.testing.assert_allclose(state[name],candidate[name],rtol=1e-9,atol=2e-6)
        assert not np.shares_memory(state[name],origin[name])
    assert canonical_json_bytes(origin)==before


@pytest.mark.parametrize('extension,active_expected',((.0038,False),(-.002,True)))
def test_tilted_material_unload_reverse_derivative(tilted_material,extension,active_expected):
    _,(model,material,element,reference,origin,_),rotation=tilted_material
    field=np.zeros((4,6));field[1,0]=field[2,0]=extension
    field[1,2]=.0004;field[2,2]=.0007
    trial=(field.reshape(-1,3)@rotation.as_matrix().T).ravel()
    before=canonical_json_bytes(origin)
    force,tangent,state=response(model,element,trial,origin,3)
    active=state['alpha']>origin['alpha']
    assert bool(active.any())==active_expected
    direction=np.sin(np.arange(24)+.3);direction/=np.linalg.norm(direction)
    for h in (2e-7,1e-7):
        plus,_,a=response(model,element,trial+h*direction,origin,3)
        minus,_,b=response(model,element,trial-h*direction,origin,3)
        np.testing.assert_array_equal(a['alpha']>origin['alpha'],active)
        np.testing.assert_array_equal(b['alpha']>origin['alpha'],active)
        predicted=tangent@direction
        assert np.linalg.norm((plus-minus)/(2*h)-predicted)/np.linalg.norm(predicted)<=1e-5
    assert np.isfinite(force).all() and canonical_json_bytes(origin)==before
