"""Actual native distributed-couple state/solver integration, private only."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import LoadCase
from anyfesolver._ge_beam3_native_distributed_element import NativeDistributedFibreStaticElement,seal
from anyfesolver._ge_beam3_native_distributed_loading import DistributedPattern,assemble_distributed_trial,_ACTIVE
from anyfesolver._ge_beam3_native_distributed_program import solve_distributed_model,_PROGRAM
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from docs.reference_cases import ge_beam3_distributed_couple_oracle as oracle
from test_ge_beam3_native_line_loading import problem as line_problem,independent_work
from test_ge_beam3_native_line_restart import make as line_connected
from test_ge_beam3_native_fibre_static_element import store_for,sample
from test_ge_beam3_schur_line_program import save

CASES=((False,False),(True,False),(True,True))


def convert(source):
    model=FEModel('native-distributed-static')
    for i,node in source.mesh.nodes.items():model.add_node(i,*node.coords())
    for i,e in source.mesh.elements.items():
        made=NativeDistributedFibreStaticElement(i,e.node_ids,e.operator.reference,e.section,order=4)
        model.add_element(i,made);model.materials[made.material_name]=made.section
    for bc in source.boundary_conditions:model.add_boundary_condition(bc)
    return model


def problem(curved=True,plastic=True):
    m=convert(line_problem(curved,plastic)[0]);return m,m.mesh.elements[1]


def pattern(model,*,line=True,couple=True):
    return DistributedPattern(LinePattern(tuple((i,.09,-.03,.02) for i in sorted(model.mesh.elements)) if line else ()),
        tuple((i,.07,-.04,.05) for i in sorted(model.mesh.elements)) if couple else ())


def packet(result):return dict(status=result.status,displacements=result.displacements,states=result.element_states)


@pytest.mark.parametrize('curved,plastic',CASES)
def test_actual_trial_work_replay_commit_discard(curved,plastic,tmp_path):
    m,e=problem(curved,plastic);store=store_for(m,e);before=canonical(store.materialize());load=pattern(m)
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
    errors=dict(full_residual=float(np.max(np.abs(rc-couple-r.full_residual))),
        full_jacobian=float(np.max(np.abs(j-r.full_spatial_jacobian))),external=float(np.max(np.abs(external-g[:18]))),
        force=float(np.max(np.abs(force-expected_f))),tangent=float(np.max(np.abs(k.toarray()-expected_k))))
    assert max(errors.values())<=1e-11 and np.linalg.norm((k-k.T).toarray())>1e-3
    token=store.active_trial_token();store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=state['positions'])
    assert store.generation==1 and canonical(store[1])==canonical(state)
    accepted=canonical(store.materialize());_,_,later,_=assemble_distributed_trial(m,.4*sample(),store,load)
    assert canonical(later[1]['origins'])==canonical(store[1]['response'].history)
    store.discard_trial(store.active_trial_token());assert canonical(store.materialize())==accepted
    if plastic:assert any(row[2]>0 for station in state['response'].history.stations for row in station.rows)
    save(tmp_path/'trial.json',dict(state=state,errors=errors,tangent=k.toarray(),committed_then_discarded=True,production_qualified=False))


@pytest.mark.parametrize('curved,plastic',CASES)
def test_actual_fixed_origin_directional(curved,plastic,tmp_path):
    m,e=problem(curved,plastic);store=store_for(m,e);load=pattern(m);base=sample();before=canonical(store.materialize())
    direction=np.sin(np.arange(18)+.3);direction[:6]=0.
    _,k,_,_=assemble_distributed_trial(m,base,store,load);expected=k@direction;errors=[]
    for step in (2e-5,1e-5):
        plus,_,_,fp=assemble_distributed_trial(m,base+step*direction,store,load,tangent=False)
        minus,_,_,fm=assemble_distributed_trial(m,base-step*direction,store,load,tangent=False)
        observed=((plus-fp)-(minus-fm))/(2*step)
        errors.append(float(np.linalg.norm(observed-expected)/max(1.,np.linalg.norm(expected))))
    assert max(errors)<=1e-7 and canonical(store.materialize())==before
    store.discard_trial(store.active_trial_token());save(tmp_path/'directional.json',dict(errors=errors,fixed_origin=True))


@pytest.mark.parametrize('case',['straight','curved-elastic','curved-plastic','connected-plastic'])
def test_actual_global_equilibrium_and_general_factorization(case,monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    from anyfesolver.linalg import MatrixClass
    m=convert(line_connected('curved-connected-plastic')) if case=='connected-plastic' else problem(case!='straight',case=='curved-plastic')[0]
    load=pattern(m);calls=[];original=solver.factorize
    def record(matrix,kind,*args,**kwargs):
        if str(kwargs.get('signature','')).startswith('nonlinear.static_newton'):
            calls.append(dict(kind=kind.value,asymmetry=float(np.linalg.norm((matrix-matrix.T).toarray()))))
        return original(matrix,kind,*args,**kwargs)
    monkeypatch.setattr(solver,'factorize',record)
    result,evidence=solve_distributed_model(m,load);save(tmp_path/'result.json',dict(result=packet(result),evidence=evidence,calls=calls))
    assert result.status=='completed',result.info
    assert result.info['native_distributed_couple_general_matrix'] and calls and all(c['kind']==MatrixClass.GENERAL.value for c in calls)
    assert max(c['asymmetry'] for c in calls)>1e-3
    errors=[]
    for snapshot in result.snapshots:
        net=np.zeros_like(snapshot.displacements);shared={}
        for eid,e in m.mesh.elements.items():
            state=snapshot.element_states[eid];e._validate(m.mesh,state,snapshot.displacements[list(e.get_dof_mapping(m.mesh))])
            net[list(e.get_dof_mapping(m.mesh))]+=state['response'].residual
            for local,node in enumerate(e.node_ids):
                r=state['committed_nodal_rotation_matrices'][local]
                if node in shared:np.testing.assert_array_equal(shared[node],r)
                shared[node]=r
        errors.append(float(np.linalg.norm(net[6:])));assert errors[-1]<=1e-11
    assert _PROGRAM.get() is None and _ACTIVE.get() is None
    save(tmp_path/'audit.json',dict(case=case,free_residual_errors=errors,shared_rotations_exact=True,general_factorization=True,production_qualified=False))


def test_pure_distributed_torsion_and_zero_nodal_load(tmp_path):
    from test_ge_beam3_native_spatial_couples import make
    from anyfesolver._ge_beam3_p5.algebra import rotation
    m=convert(make()[0]);load=DistributedPattern(LinePattern(()),((1,.12,0.,0.),))
    result,evidence=solve_distributed_model(m,load)
    save(tmp_path/'torsion.json',dict(result=packet(result),evidence=evidence))
    assert result.status=='completed',result.info
    for snapshot,step in zip(result.snapshots,result.steps):
        state=snapshot.element_states[1]
        np.testing.assert_allclose(state['positions'],m.mesh.elements[1].operator.reference.coordinates,atol=1e-11,rtol=0.)
        # GJ=2, L=1: theta(s)=m*(L*s-s*s/2)/GJ.
        for i,s in enumerate((0.,.5,1.)):
            expected=rotation([snapshot.load_factor*.12*(s-s*s/2)/2,0.,0.])
            np.testing.assert_allclose(state['committed_nodal_rotation_matrices'][i],expected,atol=1e-11,rtol=0.)
        np.testing.assert_allclose(step.support_reactions['root'][3:],[-snapshot.load_factor*.12,0.,0.],atol=1e-11,rtol=0.)


@pytest.mark.parametrize('mutation',['density','line','signature','input','jacobian','applied','history','qualification'])
def test_resealed_state_corruption(mutation,tmp_path):
    m,e=problem();store=store_for(m,e);_,_,states,_=assemble_distributed_trial(m,sample(),store,pattern(m));broken=deepcopy(states[1]);r=broken['response']
    if mutation=='density':broken['load_pattern']=DistributedPattern(broken['load_pattern'].line,((1,.08,-.04,.05),))
    elif mutation=='line':broken['load_pattern']=DistributedPattern(LinePattern(((1,.1,-.03,.02),)),broken['load_pattern'].couples)
    elif mutation=='signature':object.__setattr__(broken['load_pattern'],'signature','0'*64)
    elif mutation=='input':broken['response']=replace(r,input_sha256='0'*64)
    elif mutation=='jacobian':a=r.spatial_jacobian.copy();a[0,0]+=.01;broken['response']=replace(r,spatial_jacobian=a)
    elif mutation=='applied':a=r.applied_couple.copy();a[18]+=.01;broken['response']=replace(r,applied_couple=a)
    elif mutation=='history':broken['origins']=r.history
    else:broken['response']=replace(r,production_qualified=True)
    broken=seal(broken)
    with pytest.raises(ValueError):e._validate(m.mesh,broken)
    store.discard_trial(store.active_trial_token());save(tmp_path/'rejected.json',dict(mutation=mutation,rejected=True))


def test_wrong_load_trial_and_foreign_registration(tmp_path):
    m,e=problem();store=store_for(m,e);before=canonical(store.materialize())
    with pytest.raises(ValueError,match='load scope'):_assemble_nonlinear_system(m,sample(),store,1)
    _,_,a,_=assemble_distributed_trial(m,sample(),store,pattern(m));a=deepcopy(a[1])
    _,_,b,_=assemble_distributed_trial(m,sample(),store,DistributedPattern(LinePattern(()),((1,.08,-.04,.05),)));b=deepcopy(b[1])
    e._validate(m.mesh,a);e._validate(m.mesh,b);token=store.active_trial_token()
    with pytest.raises(ValueError,match='not issued'):store.set_trial_state(token,1,a)
    store._fallback_trial[1]=a
    with pytest.raises(ValueError,match='not issued'):store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=a['positions'])
    store.discard_trial(token);assert canonical(store.materialize())==before
    _,_,a,_=assemble_distributed_trial(m,sample(),store,pattern(m));a=deepcopy(a[1]);old_token=store.active_trial_token()
    other=store_for(m,e);_,_,b,_=assemble_distributed_trial(m,sample(),other,pattern(m));assert canonical(a)==canonical(b[1])
    with pytest.raises(ValueError,match='foreign'):store.commit(old_token,accepted_full_displacement=sample(),accepted_full_coordinates=a['positions'])
    store.discard_trial(old_token);other.discard_trial(other.active_trial_token())
    save(tmp_path/'ownership.json',dict(wrong_load_trial_rejected=True,foreign_store_commit_rejected=True,committed_unchanged=True))


def test_mid_assembly_failure_discards_owned_trial(monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    m,e=problem();store=store_for(m,e);before=canonical(store.materialize());original=solver._assemble_nonlinear_system
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);object.__setattr__(_ACTIVE.get().pattern,'signature','0'*64);return result
    monkeypatch.setattr(solver,'_assemble_nonlinear_system',changed)
    with pytest.raises(ValueError,match='authority'):assemble_distributed_trial(m,sample(),store,pattern(m))
    assert not store.has_active_trial and canonical(store.materialize())==before and _ACTIVE.get() is None
    save(tmp_path/'discard.json',dict(post_failure_discard=True,committed_unchanged=True))


def test_failed_global_step_and_restart_fail_closed(tmp_path):
    import anyfesolver.nonlinear_static as solver
    m,e=problem();initial=e.init_model_bound_nonlinear_state(m.mesh,e.section,1)
    result,evidence=solve_distributed_model(m,pattern(m),steps=1,max_iterations=1)
    save(tmp_path/'failed.json',dict(result=packet(result),evidence=evidence))
    assert result.status!='completed' and canonical(result.element_states[1])==canonical(initial)
    with pytest.raises(ValueError,match='authenticated chain'):
        solver.solve_static_nonlinear(m,LoadCase('raw-state'),initial_element_states={1:initial})
    with pytest.raises(ValueError,match='live native distributed'):
        solver.solve_static_nonlinear(m,LoadCase('unissued-driver'))
    assert _PROGRAM.get() is None and _ACTIVE.get() is None


@pytest.mark.parametrize('mutation',['pattern','absent-element','unsupported-pressure','norm-range'])
def test_preflight_and_unsupported_routes(mutation,monkeypatch):
    m,e=problem();load=pattern(m)
    if mutation=='pattern':object.__setattr__(load,'signature','0'*64)
    elif mutation=='absent-element':load=DistributedPattern(LinePattern(()),((99,.1,0.,0.),))
    elif mutation=='unsupported-pressure':
        force=LoadCase('pressure');force.add_pressure_load(1,1.)
        with pytest.raises(ValueError,match='not qualified'):force.get_load_vector(m.mesh,m.mesh.dof_manager,m.get_material,displacements=np.zeros(18))
        return
    else:
        with pytest.raises(ValueError,match='range'):solve_distributed_model(m,DistributedPattern(LinePattern(()),((1,1e200,0.,0.),)))
        return
    def forbidden(*a,**k):raise AssertionError('mutated authority entered mechanics')
    monkeypatch.setattr(e,'compute_stiffness_matrix',forbidden)
    with pytest.raises(ValueError):solve_distributed_model(m,load)
