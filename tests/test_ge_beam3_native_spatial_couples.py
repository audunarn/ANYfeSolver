"""Actual Newton spatial-couple work/tangents; no conservative spectrum claim."""
from copy import deepcopy
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_native_line_static_element import NativeLineFibreStaticElement
from anyfesolver._ge_beam3_native_line_loading import LinePattern,assemble_line_trial
from anyfesolver._ge_beam3_native_spatial_couples import solve_spatial_static,external_at,_Run,_RUN
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver._ge_beam3_p5.algebra import rotation,skew,log_rotation
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_spatial_nodal_moments import model as source_model
from test_ge_beam3_native_line_loading import problem
from test_ge_beam3_native_fibre_static_element import store_for,sample
from test_ge_beam3_schur_line_program import save


def make(common=None,plastic=False,coupled=False):
    source=source_model(common,plastic=plastic,coupled=coupled);model=FEModel('actual-native-spatial-couples')
    for i,node in source.mesh.nodes.items():model.add_node(i,*node.coords())
    old=source.mesh.elements[1];e=NativeLineFibreStaticElement(1,(1,2,3),old.operator.reference,old.section,order=4)
    model.add_element(1,e);model.materials[e.material_name]=e.section
    for bc in source.boundary_conditions:model.add_boundary_condition(bc)
    return model,e


def packet(result):return dict(status=result.status,displacements=result.displacements,states=result.element_states,
    steps=[s.to_dict() for s in result.steps],general_matrix=result.info.get('native_spatial_couple_general_matrix',False))


def left_jacobian(v):
    """Independent closed Rodrigues expression, no production chart/Jet2."""
    angle=float(np.linalg.norm(v));cross=skew(v);s=angle*angle
    a=.5-s/24+s*s/720 if angle<1e-4 else (1-np.cos(angle))/s
    b=1/6-s/120+s*s/5040 if angle<1e-4 else (angle-np.sin(angle))/(angle*s)
    return np.eye(3)+a*cross+b*cross@cross


@pytest.mark.parametrize('plastic',[False,True])
def test_external_work_and_combined_directional_tangent(plastic,tmp_path):
    model,e=problem(True,plastic);store=store_for(model,e);u=sample();before=canonical(store.materialize())
    moments=SpatialNodalMoments(((3,.2,-.15,.12),));run=_Run(model,moments);token=_RUN.set(run)
    line=LinePattern(((1,.09,-.03,.02),));direction=np.sin(np.arange(18)+.3);direction[:6]=0.
    try:
        internal,k,states,load=assemble_line_trial(model,u,store,line)
        extra,kt=external_at(model,store,u,.7,tangent=True)
        moment=.7*np.array([.2,-.15,.12]);delta=u.reshape(3,6)[2,3:]
        expected=left_jacobian(delta).T@moment
        np.testing.assert_allclose(extra[15:18],expected,atol=1e-11,rtol=0.)
        virtual=left_jacobian(delta)@direction[15:18]
        work_error=float(abs(extra@direction-moment@virtual));assert work_error<=1e-11
        analytical=(k-kt)@direction;errors=[]
        for h in (2e-5,1e-5):
            plus,_,_,lp=assemble_line_trial(model,u+h*direction,store,line,tangent=False)
            ep,_=external_at(model,store,u+h*direction,.7,tangent=False)
            minus,_,_,lm=assemble_line_trial(model,u-h*direction,store,line,tangent=False)
            em,_=external_at(model,store,u-h*direction,.7,tangent=False)
            observed=((plus-lp-ep)-(minus-lm-em))/(2*h)
            errors.append(float(np.linalg.norm(observed-analytical)/max(1.,np.linalg.norm(analytical))))
        assert max(errors)<=1e-7 and np.linalg.norm((kt-kt.T).toarray())>1e-3
        assert canonical(store.materialize())==before
        save(tmp_path/'directional.json',dict(plastic=plastic,work_error=work_error,errors=errors,external_force=extra,
            external_tangent=kt.toarray(),committed_unchanged=True,conservative_potential=False))
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        _RUN.reset(token)


def test_actual_large_torsion_and_general_solver(monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    model,e=make();calls=[];original=solver.factorize
    def record(matrix,kind,*args,**kwargs):
        if str(kwargs.get('signature','')).startswith('nonlinear.static_newton'):
            calls.append(dict(kind=kind.value,asymmetry=float(np.linalg.norm((matrix-matrix.T).toarray()))))
        return original(matrix,kind,*args,**kwargs)
    monkeypatch.setattr(solver,'factorize',record)
    result,evidence=solve_spatial_static(model,SpatialNodalMoments(((3,2.4,0.,0.),)),steps=3)
    save(tmp_path/'torsion.json',dict(result=packet(result),evidence=evidence,factorizations=calls))
    assert result.status=='completed',result.info
    for snapshot,step in zip(result.snapshots,result.steps):
        angle=1.2*snapshot.load_factor;state=snapshot.element_states[1]
        np.testing.assert_allclose(state['committed_nodal_rotation_matrices'][-1],rotation([angle,0.,0.]),atol=1e-11,rtol=0.)
        np.testing.assert_allclose(state['positions'],e.operator.reference.coordinates,atol=1e-11,rtol=0.)
        np.testing.assert_allclose(step.support_reactions['root'][3:],[-2.4*snapshot.load_factor,0.,0.],atol=1e-11,rtol=0.)
    from anyfesolver.linalg import MatrixClass
    assert calls and all(c['kind']==MatrixClass.GENERAL.value for c in calls)
    assert max(c['asymmetry'] for c in calls)>1e-3
    assert not evidence['program']['conservative_potential'] and _RUN.get() is None


def test_biaxial_bending_and_rigid_frame_covariance(tmp_path):
    moment=np.array([0.,.6,.8]);model,e=make()
    a,ea=solve_spatial_static(model,SpatialNodalMoments(((3,*map(float,moment)),)))
    save(tmp_path/'base.json',dict(result=packet(a),evidence=ea));assert a.status=='completed',a.info
    common=rotation([2.5,-.7,.8]);rotated,_=make(common)
    b,eb=solve_spatial_static(rotated,SpatialNodalMoments(((3,*map(float,common@moment)),)))
    save(tmp_path/'rotated.json',dict(result=packet(b),evidence=eb));assert b.status=='completed',b.info
    sa=a.element_states[1];sb=b.element_states[1]
    np.testing.assert_allclose(sa['committed_nodal_rotation_matrices'][-1],rotation(moment/4),atol=1e-11,rtol=0.)
    position_error=float(np.linalg.norm(sb['positions']-(common@sa['positions'].T).T))
    rotation_error=float(np.linalg.norm(sb['committed_nodal_rotation_matrices']-common@sa['committed_nodal_rotation_matrices']@common.T))
    assert max(position_error,rotation_error)<=1e-11
    save(tmp_path/'covariance.json',dict(position_error=position_error,rotation_error=rotation_error,production_qualified=False))


def test_coupled_curved_plastic_line_forces_and_moments(tmp_path):
    model,e=problem(True,True);moment=np.array([.12,.10,-.08]);line=LinePattern(((1,.34,-.014,.007),))
    result,evidence=solve_spatial_static(model,SpatialNodalMoments(((3,*map(float,moment)),)),line_pattern=line)
    save(tmp_path/'mixed.json',dict(result=packet(result),evidence=evidence));assert result.status=='completed',result.info
    errors=[]
    for snapshot in result.snapshots:
        state=snapshot.element_states[1];net=state['response'].residual.copy();net[15:18]-=snapshot.load_factor*moment
        error=float(np.linalg.norm(net[6:]));assert error<=1e-11;errors.append(error)
    assert any(row[2]>0 for station in result.element_states[1]['response'].history.stations for row in station.rows)
    assert canonical(result.snapshots[1].element_states[1]['origins'])==canonical(result.snapshots[0].element_states[1]['response'].history)
    save(tmp_path/'mixed-audit.json',dict(free_residual_errors=errors,plastic_history_present=True,production_qualified=False))


def test_direct_support_moment_is_external_reaction(tmp_path):
    model,_=make();moment=np.array([.2,-.1,.3])
    result,evidence=solve_spatial_static(model,SpatialNodalMoments(((1,*map(float,moment)),)),steps=1)
    save(tmp_path/'root-couple.json',dict(result=packet(result),evidence=evidence));assert result.status=='completed',result.info
    np.testing.assert_array_equal(result.displacements,np.zeros(18))
    np.testing.assert_allclose(result.steps[0].support_reactions['root'][3:],-moment,atol=1e-11,rtol=0.)


@pytest.mark.parametrize('kind',['absent-node','wrong-type','mutated-pattern'])
def test_authority_preflight(kind,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_spatial_couples as module
    model,_=make();moment=SpatialNodalMoments(((3,.1,.2,.3),))
    if kind=='absent-node':moment=SpatialNodalMoments(((9,.1,.2,.3),))
    elif kind=='wrong-type':moment=moment.rows
    else:object.__setattr__(moment,'rows',((3,float('nan'),.2,.3),))
    def forbidden(*a,**k):raise AssertionError('solver entered')
    monkeypatch.setattr(module,'solve_line_static',forbidden)
    with pytest.raises(ValueError):solve_spatial_static(model,moment)
    assert _RUN.get() is None
    save(tmp_path/'rejected.json',dict(kind=kind,rejected=True))


def test_closed_orientation_loop_has_nonzero_work():
    moment=SpatialNodalMoments(((3,0.,0.,1.),));q=np.eye(3);steps=np.array([[.2,0.,0.],[0.,.3,0.],[-.2,0.,0.],[0.,-.3,0.]])
    for step in steps:q=rotation(step)@q
    closing=-log_rotation(q);np.testing.assert_allclose(rotation(closing)@q,np.eye(3),atol=1e-14,rtol=0.)
    assert abs(np.dot([0.,0.,1.],steps.sum(axis=0)+closing))>.01
    with pytest.raises(ValueError):moment.potential(q)


def test_connected_junction_couple_is_counted_once(tmp_path):
    from test_ge_beam3_native_line_restart import make as connected
    model=connected('curved-connected-plastic');moment=np.array([.03,.02,-.01])
    line=LinePattern(tuple((i,.17,-.007,.0035) for i in sorted(model.mesh.elements)))
    result,evidence=solve_spatial_static(model,SpatialNodalMoments(((3,*map(float,moment)),)),line_pattern=line)
    save(tmp_path/'connected.json',dict(result=packet(result),evidence=evidence));assert result.status=='completed',result.info
    net=np.zeros(30)
    for eid,e in model.mesh.elements.items():net[list(e.get_dof_mapping(model.mesh))]+=result.element_states[eid]['response'].residual
    np.testing.assert_array_equal(result.element_states[1]['committed_nodal_rotation_matrices'][2],result.element_states[2]['committed_nodal_rotation_matrices'][0])
    net[15:18]-=moment;error=float(np.linalg.norm(net[6:]));assert error<=1e-11
    save(tmp_path/'connected-audit.json',dict(free_residual_error=error,junction_moment_counted_once=True,shared_rotation_exact=True))


@pytest.mark.parametrize('mutation',['pattern','foreign-store','unissued-pose'])
def test_live_external_authority_and_cleanup(mutation,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_spatial_couples as module
    model,e=problem(True,True);store=store_for(model,e);before=canonical(store.materialize());u=sample()
    run=_Run(model,SpatialNodalMoments(((3,.1,.2,.3),)));token=_RUN.set(run)
    try:
        assemble_line_trial(model,u,store,LinePattern(()))
        external_at(model,store,u,.5,tangent=True)
        if mutation=='pattern':
            original=module.exp_chart_terms
            def change(v):
                value=original(v);run.proportional=SpatialNodalMoments(((3,.2,.2,.3),));return value
            monkeypatch.setattr(module,'exp_chart_terms',change)
            with pytest.raises(ValueError,match='authority'):external_at(model,store,u,.5,tangent=True)
            assert not store.has_active_trial
        elif mutation=='foreign-store':
            other,e2=problem(True,True);foreign=store_for(other,e2)
            with pytest.raises(ValueError,match='foreign'):external_at(model,foreign,u,.5,tangent=True)
            assert not foreign.has_active_trial
        else:
            store.discard_trial(store.active_trial_token())
            with pytest.raises(ValueError,match='unissued'):external_at(model,store,u,.5,tangent=True)
        assert canonical(store.materialize())==before
        save(tmp_path/'authority.json',dict(mutation=mutation,committed_unchanged=True,rejected=True))
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        _RUN.reset(token)


def test_failed_actual_increment_preserves_committed_state(tmp_path):
    model,e=make();initial=e.init_model_bound_nonlinear_state(model.mesh,e.section,1)
    result,evidence=solve_spatial_static(model,SpatialNodalMoments(((3,.2,.3,.4),)),steps=1,max_iterations=1)
    save(tmp_path/'failed.json',dict(result=packet(result),evidence=evidence))
    assert result.status!='completed' and canonical(result.element_states[1])==canonical(initial)
    assert _RUN.get() is None


@pytest.mark.parametrize('kind',['effective-sum','external-norm','combined-load'])
def test_numeric_range_fails_closed(kind,tmp_path):
    model,e=make();save(tmp_path/'range-input.json',dict(kind=kind,finite_components=True))
    if kind=='effective-sum':
        large=SpatialNodalMoments(((3,1e308,0.,0.),));run=_Run(model,large,large)
        with pytest.raises(ValueError,match='range'):run.effective(1.)
    elif kind=='external-norm':
        store=store_for(model,e);run=_Run(model,SpatialNodalMoments(((3,1e200,0.,0.),)));token=_RUN.set(run)
        try:
            with pytest.raises(ValueError,match='range'):external_at(model,store,np.zeros(18),1.,tangent=True)
        finally:_RUN.reset(token)
    else:
        with pytest.raises(ValueError,match='range'):
            solve_spatial_static(model,SpatialNodalMoments(((3,.1,0.,0.),)),line_pattern=LinePattern(((1,1e200,0.,0.),)),steps=1)
    assert _RUN.get() is None
    save(tmp_path/'range-rejected.json',dict(kind=kind,rejected=True,production_qualified=False))
