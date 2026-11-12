"""Native generalized section internal equilibrium and actual state-store dispatch."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement,seal
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern,assemble_distributed_trial,_ACTIVE
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_generalized_static_boundary import solve_distributed_static,cell_couple_load
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from docs.reference_cases import ge_beam3_distributed_couple_oracle as oracle
from test_ge_beam3_native_line_loading import problem as geometry_problem,independent_work
from test_ge_beam3_native_fibre_static_element import store_for,sample
from test_ge_beam3_generalized_ellipsoid_section import section
from test_ge_beam3_schur_line_program import save

CASES=((False,False),(True,False),(True,True))


def problem(curved=True,plastic=True):
    source,old=geometry_problem(curved,plastic);m=FEModel('native-generalized-static')
    for i,node in source.mesh.nodes.items():m.add_node(i,*node.coords())
    c=section();law=EllipsoidalGeneralizedSection(c.elastic,c.metric,.025 if plastic else 1e6,c.hardening)
    e=NativeGeneralizedStaticElement(1,(1,2,3),old.operator.reference,law,order=4)
    m.add_element(1,e);m.materials[e.material_name]=e.section
    for bc in source.boundary_conditions:m.add_boundary_condition(bc)
    return m,e


def pattern():
    return DistributedPattern(LinePattern(((1,.09,-.03,.02),)),((1,.07,-.04,.05),))


@pytest.mark.parametrize('curved,plastic',CASES)
def test_actual_state_work_commit_discard(curved,plastic,tmp_path):
    m,e=problem(curved,plastic);store=store_for(m,e);before=canonical(store.materialize());load=pattern()
    force,k,states,external=assemble_distributed_trial(m,sample(),store,load);state=deepcopy(states[1]);r=state['response']
    assert canonical(store.materialize())==before;e._validate(m.mesh,state,sample())
    material=e.operator.evaluate(state['positions'],state['position_low'],state['committed_nodal_rotation_matrices']@e.operator.reference.nodal_triads,
        r.rotations,r.resultants,origin=state['origins'])
    _,g,h=independent_work(e,state,load.force(1));couple=oracle.load(e.operator.reference.coordinates,load.density(1),e.operator.order)
    rc=material.residual-g;hc=material.hessian+material.hessian_low-h;j=hc.copy()
    for start in (3,9,15,18,21):
        for col,unit in enumerate(np.eye(3)):j[start:start+3,start+col]-=.5*np.cross(rc[start:start+3],unit)
    reduced=j[:18,:18]-j[:18,18:]@np.linalg.solve(j[18:,18:],j[18:,:18])
    expected_f,expected_k=oracle.pullback(rc[:18]+g[:18],reduced,sample().reshape(3,6)[:,3:])
    errors=dict(residual=float(np.max(np.abs(rc-couple-r.full_residual))),jacobian=float(np.max(np.abs(j-r.full_spatial_jacobian))),
        external=float(np.max(np.abs(external-g[:18]))),force=float(np.max(np.abs(force-expected_f))),tangent=float(np.max(np.abs(k.toarray()-expected_k))))
    save(tmp_path/'trial.json',dict(state=state,errors=errors,tangent=k.toarray()))
    assert max(errors.values())<=1e-11 and np.linalg.norm((k-k.T).toarray())>1e-3
    if plastic:
        assert all(any(sum(history.plastic[i])!=0 for history in r.history.stations) for i in range(6))
    token=store.active_trial_token();store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=state['positions'])
    assert store.generation==1 and canonical(store[1])==canonical(state)
    accepted=canonical(store.materialize());_,_,later,_=assemble_distributed_trial(m,.4*sample(),store,load)
    assert canonical(later[1]['origins'])==canonical(store[1]['response'].history)
    store.discard_trial(store.active_trial_token());assert canonical(store.materialize())==accepted
    e._validate(m.mesh,store[1],sample())
    save(tmp_path/'commit.json',dict(committed_state=store[1],generation=store.generation,discard_preserved=True,accepted_replay=True))


@pytest.mark.parametrize('curved,plastic',CASES)
def test_actual_directional_tangent(curved,plastic,tmp_path):
    m,e=problem(curved,plastic);store=store_for(m,e);load=pattern();base=sample();before=canonical(store.materialize())
    direction=np.sin(np.arange(18)+.3);direction[:6]=0.
    _,k,_,_=assemble_distributed_trial(m,base,store,load);expected=k@direction;errors=[];observations=[]
    steps=(2e-5,1e-5,5e-6,2.5e-6)
    for step in steps:
        plus,_,_,fp=assemble_distributed_trial(m,base+step*direction,store,load,tangent=False)
        minus,_,_,fm=assemble_distributed_trial(m,base-step*direction,store,load,tangent=False)
        observed=((plus-fp)-(minus-fm))/(2*step)
        observations.append(observed)
        errors.append(float(np.linalg.norm(observed-expected)/max(1.,np.linalg.norm(expected))))
    extrapolated=[(4*b-a)/3 for a,b in zip(observations,observations[1:])]
    extrapolated_errors=[float(np.linalg.norm(v-expected)/max(1.,np.linalg.norm(expected))) for v in extrapolated]
    save(tmp_path/'directional.json',dict(steps=steps,errors=errors,richardson_errors=extrapolated_errors,observed=observations,expected=expected,fixed_origin=True))
    assert max(errors[-2:])<=1e-7 and max(extrapolated_errors)<=1e-7 and canonical(store.materialize())==before
    if max(errors[:2])>1e-7:
        assert all(errors[i+1]<.3*errors[i] for i in range(3))
    store.discard_trial(store.active_trial_token())


@pytest.mark.parametrize('mutation',['density','line','signature','input','jacobian','applied','history','qualification','schema'])
def test_resealed_state_corruption(mutation,tmp_path):
    m,e=problem();store=store_for(m,e);_,_,states,_=assemble_distributed_trial(m,sample(),store,pattern())
    broken=deepcopy(states[1]);r=broken['response']
    if mutation=='density':broken['load_pattern']=DistributedPattern(broken['load_pattern'].line,((1,.08,-.04,.05),))
    elif mutation=='line':broken['load_pattern']=DistributedPattern(LinePattern(((1,.1,-.03,.02),)),broken['load_pattern'].couples)
    elif mutation=='signature':object.__setattr__(broken['load_pattern'],'signature','0'*64)
    elif mutation=='input':broken['response']=replace(r,input_sha256='0'*64)
    elif mutation=='jacobian':a=r.spatial_jacobian.copy();a[0,0]+=.01;broken['response']=replace(r,spatial_jacobian=a)
    elif mutation=='applied':a=r.applied_couple.copy();a[18]+=.01;broken['response']=replace(r,applied_couple=a)
    elif mutation=='history':broken['origins']=r.history
    elif mutation=='schema':broken['schema']='GE_BEAM3_NATIVE_PHYSICAL_FIBRE_DISTRIBUTED_STATIC_STATE_V1'
    else:broken['response']=replace(r,production_qualified=True)
    with pytest.raises(ValueError):e._validate(m.mesh,seal(broken))
    store.discard_trial(store.active_trial_token());save(tmp_path/'rejected.json',dict(mutation=mutation,rejected=True))


def test_wrong_load_and_foreign_store_cannot_commit(tmp_path):
    m,e=problem();store=store_for(m,e);before=canonical(store.materialize())
    with pytest.raises(ValueError,match='load scope'):_assemble_nonlinear_system(m,sample(),store,1)
    _,_,a,_=assemble_distributed_trial(m,sample(),store,pattern());a=deepcopy(a[1])
    changed=DistributedPattern(LinePattern(()),((1,.08,-.04,.05),))
    assemble_distributed_trial(m,sample(),store,changed);token=store.active_trial_token()
    with pytest.raises(ValueError,match='not issued'):store.set_trial_state(token,1,a)
    store._fallback_trial[1]=a
    with pytest.raises(ValueError,match='not issued'):store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=a['positions'])
    store.discard_trial(token);assert canonical(store.materialize())==before
    _,_,a,_=assemble_distributed_trial(m,sample(),store,pattern());a=deepcopy(a[1]);token=store.active_trial_token()
    other=store_for(m,e);_,_,b,_=assemble_distributed_trial(m,sample(),other,pattern())
    assert canonical(a)==canonical(b[1])
    with pytest.raises(ValueError,match='foreign'):store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=a['positions'])
    store.discard_trial(token);other.discard_trial(other.active_trial_token())
    save(tmp_path/'ownership.json',dict(wrong_trial_rejected=True,foreign_store_rejected=True))


def test_post_assembly_failure_discards_trial(monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    m,e=problem();store=store_for(m,e);before=canonical(store.materialize());original=solver._assemble_nonlinear_system
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);object.__setattr__(_ACTIVE.get().pattern,'signature','0'*64);return result
    monkeypatch.setattr(solver,'_assemble_nonlinear_system',changed)
    with pytest.raises(ValueError,match='authority'):assemble_distributed_trial(m,sample(),store,pattern())
    assert not store.has_active_trial and canonical(store.materialize())==before and _ACTIVE.get() is None
    save(tmp_path/'failure.json',dict(discarded=True,committed_preserved=True))


def test_internal_limits_and_unsupported_routes(monkeypatch,tmp_path):
    m,e=problem();op=e.operator;origin=op.cell.virgin();before=canonical(origin)
    kwargs=dict(origin=origin,initial_rotations=np.array([np.eye(3)]*2),initial_resultants=np.zeros(18),
        spatial_line_force=pattern().force(1),spatial_couple_density=pattern().density(1))
    with pytest.raises(ValueError,match='iteration limit'):
        solve_distributed_static(op,op.reference.coordinates,np.zeros((3,3)),op.reference.nodal_triads,max_iterations=0,**kwargs)
    for name in ('compute_mass_matrix','compute_geometric_stiffness_matrix','compute_stresses','compute_native_current_pressure_load'):
        with pytest.raises(ValueError,match='not qualified'):getattr(e,name)(m.mesh,e.section)
    with pytest.raises(ValueError,match='range'):cell_couple_load(op,np.array([1e200,0.,0.]))
    import anyfesolver._ge_beam3_generalized_static_boundary as module
    ticks=[]
    def clock():ticks.append(1);return 61.*len(ticks)
    monkeypatch.setattr(module,'monotonic',clock)
    with pytest.raises(RuntimeError,match='deadline'):solve_distributed_static(op,op.reference.coordinates,np.zeros((3,3)),op.reference.nodal_triads,**kwargs)
    assert canonical(origin)==before
    save(tmp_path/'limits.json',dict(iteration_limit=True,deadline=True,unsupported=True,range_rejected=True))


@pytest.mark.parametrize('bad',['old-pattern','absent-element','mutated-pattern'])
def test_load_preflight_rejects_before_mechanics(bad,monkeypatch,tmp_path):
    m,e=problem();store=store_for(m,e);load=pattern()
    if bad=='old-pattern':
        from anyfesolver._ge_beam3_native_distributed_loading import DistributedPattern as Old
        load=Old(load.line,load.couples)
    elif bad=='absent-element':load=DistributedPattern(LinePattern(()),((99,.1,0.,0.),))
    else:object.__setattr__(load,'signature','0'*64)
    def forbidden(*a,**k):raise AssertionError('invalid authority entered mechanics')
    monkeypatch.setattr(e,'compute_nonlinear_response',forbidden)
    with pytest.raises(ValueError):assemble_distributed_trial(m,sample(),store,load)
    assert not store.has_active_trial
    save(tmp_path/'preflight.json',dict(case=bad,rejected=True))



def test_connected_native_pose_and_material_transactions(tmp_path):
    from anyfesolver.nonlinear_state import NonlinearStateStore,create_model_native_rotation_store
    from test_ge_beam3_curved_contrast_probe import make_curved
    source=make_curved(100.)[0];m=FEModel('connected-native-generalized')
    for i,node in source.mesh.nodes.items():m.add_node(i,*node.coords())
    base=section();law=EllipsoidalGeneralizedSection(base.elastic,base.metric,.025,base.hardening)
    for i,old in source.mesh.elements.items():
        e=NativeGeneralizedStaticElement(i,old.node_ids,old.core.reference,law,order=4)
        m.add_element(i,e);m.materials[e.material_name]=e.section
    states={i:e.init_model_bound_nonlinear_state(m.mesh,e.section,1) for i,e in m.mesh.elements.items()}
    ndof=m.mesh.dof_manager.total_dofs
    store=NonlinearStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(m,states,np.zeros(ndof)))
    displacement=np.array([[.01*i,.001*i,-.0005*i,.004*i,-.003*i,.002*i] for i in range(5)]).ravel()
    load=DistributedPattern(LinePattern(tuple((i,.09,-.03,.02) for i in sorted(m.mesh.elements))),tuple((i,.07,-.04,.05) for i in sorted(m.mesh.elements)))
    before=canonical(store.materialize());force,k,proposed,external=assemble_distributed_trial(m,displacement,store,load)
    proposed=deepcopy(dict(proposed));assert canonical(store.materialize())==before
    np.testing.assert_array_equal(proposed[1]['committed_nodal_rotation_matrices'][2],proposed[2]['committed_nodal_rotation_matrices'][0])
    coords=np.array([node.coords() for _,node in sorted(m.mesh.nodes.items())])+displacement.reshape(5,6)[:,:3]
    store.commit(store.active_trial_token(),accepted_full_displacement=displacement,accepted_full_coordinates=coords)
    assert store.generation==1 and canonical(store.materialize())==canonical(proposed)
    for i,e in m.mesh.elements.items():
        e._validate(m.mesh,store[i],displacement[list(e.get_dof_mapping(m.mesh))])
        assert any(sum(h.accumulated)>0 for h in store[i]['response'].history.stations)
    saved=canonical(store.materialize());assemble_distributed_trial(m,.8*displacement,store,load)
    store.discard_trial(store.active_trial_token());assert canonical(store.materialize())==saved
    save(tmp_path/'connected.json',dict(states=store.materialize(),force=force,external=external,tangent=k.toarray(),shared_pose_exact=True,discard_preserved=True))


def test_zero_couple_conservative_chart_symmetry(tmp_path):
    m,e=problem();store=store_for(m,e);load=DistributedPattern(pattern().line,())
    force,k,states,external=assemble_distributed_trial(m,sample(),store,load)
    error=float(np.linalg.norm((k-k.T).toarray())/max(1.,np.linalg.norm(k.toarray())))
    assert error<=1e-11 and states[1]['response'].internal_error<=1e-11
    save(tmp_path/'zero-couple.json',dict(symmetry_error=error,tangent=k.toarray(),state=states[1],spectral_authority=False))
    store.discard_trial(store.active_trial_token())
