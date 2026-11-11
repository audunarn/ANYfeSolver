"""Native line-load scope and independent analytical load-only reconstruction."""
from copy import deepcopy
from math import fsum
import numpy as np
import pytest
from scipy import linalg
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_native_line_static_element import NativeLineFibreStaticElement,seal
from anyfesolver._ge_beam3_native_line_loading import LinePattern,assemble_line_trial
from anyfesolver._ge_beam3_p5.chart import pullback
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from test_ge_beam3_native_fibre_static_element import problem as old_problem,store_for,sample
from test_ge_beam3_schur_line_program import save

CASES=((False,False),(True,False),(True,True))
FORCE=(.09,-.03,.02)


def problem(curved=True,plastic=True):
    source,old=old_problem(curved,plastic);m=FEModel('native-load-aware-static-development')
    for i,node in source.mesh.nodes.items():m.add_node(i,*node.coords())
    e=NativeLineFibreStaticElement(1,(1,2,3),old.operator.reference,old.section,order=4)
    m.add_element(1,e);m.materials[e.material_name]=e.section
    for bc in source.boundary_conditions:m.add_boundary_condition(bc)
    return m,e


def independent_work(e,state,force):
    """Q2 nodal polynomial and explicit spatial cross-product derivatives.

    Does not call the producer load potential, Jet2, its lift or load quadrature
    helpers. This checks the load-only equations, not the full beam formulation.
    """
    xyz=e.operator.reference.coordinates;x=state['positions']+state['position_low'];u=state['response'].rotations
    a=(xyz[2]-xyz[0])/2;b=xyz[0]-2*xyz[1]+xyz[2]
    points,weights=np.polynomial.legendre.leggauss(e.operator.order)
    terms=[];g=np.zeros(42);h=np.zeros((42,42));f=np.array(force)
    for c in (0,1):
        for point,weight in zip(points,weights):
            t=(point+1)/2;xi=c-1+t;measure=weight*np.linalg.norm(a+xi*b)/2
            lift=.5*t*(t-1)*b;current=u[c]@lift
            displacement=(1-t)*(x[c]-xyz[c])+t*(x[c+1]-xyz[c+1])+current-lift
            terms.append(float(measure*np.dot(f,displacement)))
            g[6*c:6*c+3]+=measure*(1-t)*f;g[6*(c+1):6*(c+1)+3]+=measure*t*f
            g[18+3*c:21+3*c]+=measure*np.cross(current,f)
            h[18+3*c:21+3*c,18+3*c:21+3*c]+=measure*(.5*(np.outer(current,f)+np.outer(f,current))-np.dot(current,f)*np.eye(3))
    return fsum(terms),g,h


@pytest.mark.parametrize('curved,plastic',CASES)
def test_actual_dispatch_and_independent_line_work(curved,plastic,tmp_path):
    m,e=problem(curved,plastic);store=store_for(m,e);before=canonical(store.materialize());pattern=LinePattern(((1,*FORCE),))
    internal,k,states,external=assemble_line_trial(m,sample(),store,pattern);state=states[1]
    assert canonical(store.materialize())==before;e._validate(m.mesh,state,sample())
    value,g,h=independent_work(e,state,FORCE);r=state['response']
    material=e.operator.evaluate(state['positions'],state['position_low'],state['committed_nodal_rotation_matrices']@e.operator.reference.nodal_triads,r.rotations,r.resultants,origin=state['origins'])
    expected_r=material.residual-g;expected_h=material.hessian+material.hessian_low-h
    schur=expected_h[:18,:18]-expected_h[:18,18:]@linalg.solve(expected_h[18:,18:],expected_h[18:,:18])
    expected_f,expected_k,_=pullback(expected_r[:18],schur,sample().reshape(3,6)[:,3:])
    errors=dict(full_residual=float(np.max(np.abs(expected_r-r.full_residual))),full_hessian=float(np.max(np.abs(expected_h-r.full_hessian))),
        potential=abs(material.potential-value-r.potential),nodal_load=float(np.max(np.abs(external-g[:18]))),
        condensed_force=float(np.max(np.abs(internal-external-expected_f))),condensed_tangent=float(np.max(np.abs(k.toarray()-expected_k))))
    assert max(errors.values())<=1e-11
    if curved:assert np.linalg.norm(g[18:24])>1e-7
    else:assert np.array_equal(g[18:24],np.zeros(6))
    token=store.active_trial_token();store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=state['positions'])
    assert store.generation==1 and canonical(store[1])==canonical(state)
    save(tmp_path/'trial.json',dict(curved=curved,plastic=plastic,state=state,internal=internal,external=external,tangent=k.toarray(),load_errors=errors,production_qualified=False))


@pytest.mark.parametrize('curved,plastic',CASES)
def test_fixed_origin_directional_tangent(curved,plastic,tmp_path):
    m,e=problem(curved,plastic);store=store_for(m,e);pattern=LinePattern(((1,*FORCE),));base=sample();direction=np.sin(np.arange(18)+.3);direction[:6]=0.
    internal,k,states,external=assemble_line_trial(m,base,store,pattern);expected=k@direction;before=canonical(store.materialize());errors=[]
    for step in (2e-5,1e-5):
        plus,_,_,fp=assemble_line_trial(m,base+step*direction,store,pattern,tangent=False)
        minus,_,_,fm=assemble_line_trial(m,base-step*direction,store,pattern,tangent=False)
        derivative=((plus-fp)-(minus-fm))/(2*step)
        errors.append(float(np.linalg.norm(derivative-expected)/max(1.,np.linalg.norm(expected))))
    assert max(errors)<=1e-7 and canonical(store.materialize())==before
    store.discard_trial(store.active_trial_token());save(tmp_path/'directional.json',dict(curved=curved,plastic=plastic,errors=errors,origins_unchanged=True))


def test_scope_missing_and_stale_state_authority(monkeypatch,tmp_path):
    m,e=problem();store=store_for(m,e);before=canonical(store.materialize())
    with pytest.raises(ValueError,match='load scope'):_assemble_nonlinear_system(m,sample(),store,1)
    assert not store.has_active_trial and canonical(store.materialize())==before
    _,_,a,_=assemble_line_trial(m,sample(),store,LinePattern(((1,*FORCE),)))
    a=deepcopy(a[1])
    _,_,b,_=assemble_line_trial(m,sample(),store,LinePattern(((1,.07,-.02,.01),)))
    b=deepcopy(b[1])
    # Both responses are locally valid, but only B belongs to this live trial.
    e._validate(m.mesh,a);e._validate(m.mesh,b);token=store.active_trial_token()
    with pytest.raises(ValueError,match='not issued'):store.set_trial_state(token,1,a)
    store._fallback_trial[1]=a
    with pytest.raises(ValueError,match='not issued'):store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=b['positions'])
    assert canonical(store.materialize())==before;store.discard_trial(token)
    save(tmp_path/'rejected.json',dict(missing_scope=True,locally_valid_wrong_load_trial_rejected=True,committed_unchanged=True))


@pytest.mark.parametrize('mutation',['force','signature','input','potential'])
def test_resealed_line_state_corruption(mutation,tmp_path):
    m,e=problem();store=store_for(m,e);_,_,states,_=assemble_line_trial(m,sample(),store,LinePattern(((1,*FORCE),)))
    from dataclasses import replace
    broken=deepcopy(states[1])
    if mutation=='force':broken['load_pattern']=LinePattern(((1,.1,-.03,.02),))
    elif mutation=='signature':object.__setattr__(broken['load_pattern'],'signature','0'*64)
    elif mutation=='input':broken['response']=replace(broken['response'],input_sha256='0'*64)
    else:broken['response']=replace(broken['response'],potential=broken['response'].potential+.1)
    broken=seal(broken)
    with pytest.raises(ValueError):e._validate(m.mesh,broken)
    store.discard_trial(store.active_trial_token());save(tmp_path/'rejected.json',dict(mutation=mutation,rejected=True))


def test_zero_line_matches_preserved_adapter(tmp_path):
    m,e=problem();store=store_for(m,e);internal,k,states,external=assemble_line_trial(m,sample(),store,LinePattern(()))
    old,oe=old_problem();previous=store_for(old,oe);f0,k0,p0=_assemble_nonlinear_system(old,sample(),previous,1)
    np.testing.assert_array_equal(external,np.zeros(18));np.testing.assert_array_equal(internal,f0);np.testing.assert_array_equal(k.toarray(),k0.toarray())
    assert canonical(states[1]['response'])==canonical(p0[1]['response'])
    save(tmp_path/'zero.json',dict(response=states[1]['response'],zero_line_byte_identical=True))
    store.discard_trial(store.active_trial_token());previous.discard_trial(previous.active_trial_token())


def test_changed_pattern_rejected_before_mechanics(monkeypatch):
    m,e=problem();store=store_for(m,e);pattern=LinePattern(((1,*FORCE),));object.__setattr__(pattern,'rows',((1,.1,-.03,.02),))
    def forbidden(*a,**k):raise AssertionError('changed load entered mechanics')
    monkeypatch.setattr(e,'compute_nonlinear_response',forbidden)
    with pytest.raises(ValueError,match='authority changed'):assemble_line_trial(m,sample(),store,pattern)


def test_accepted_load_change_discard_and_origin_link(tmp_path):
    m,e=problem();store=store_for(m,e);_,_,states,_=assemble_line_trial(m,sample(),store,LinePattern(((1,*FORCE),)))
    store.commit(store.active_trial_token(),accepted_full_displacement=sample(),accepted_full_coordinates=states[1]['positions'])
    before=canonical(store.materialize());_,_,trial,_=assemble_line_trial(m,.4*sample(),store,LinePattern(((1,-.04,.01,-.01),)))
    assert canonical(trial[1]['origins'])==canonical(store[1]['response'].history)
    store.discard_trial(store.active_trial_token());assert canonical(store.materialize())==before
    save(tmp_path/'discard.json',dict(accepted=store[1],history_preserved=True))


def test_distinct_store_registration_cannot_reuse_load_authority(tmp_path):
    m,e=problem();a=store_for(m,e);pattern=LinePattern(((1,*FORCE),))
    _,_,sa,_=assemble_line_trial(m,sample(),a,pattern);old_token=a.active_trial_token()
    b=store_for(m,e);new_token=b.begin_trial(full_displacement=sample(),full_coordinates=sa[1]['positions'])
    with pytest.raises(ValueError,match='not issued'):b.set_trial_state(new_token,1,sa[1])
    b.discard_trial(new_token)
    _,_,sb,_=assemble_line_trial(m,sample(),b,pattern)
    assert canonical(sa[1])==canonical(sb[1])  # Same physical state, different live authority.
    with pytest.raises(ValueError,match='foreign native line validator'):
        a.commit(old_token,accepted_full_displacement=sample(),accepted_full_coordinates=sa[1]['positions'])
    a.discard_trial(old_token);b.discard_trial(b.active_trial_token())
    save(tmp_path/'foreign-store.json',dict(identical_physical_state=True,foreign_store_commit_rejected=True))


@pytest.mark.parametrize('current',[False,True])
def test_pressure_fallback_stays_closed_for_successor(current):
    from anyfesolver.boundary import LoadCase
    m,_=problem();load=LoadCase('unsupported-shell-pressure',follower_pressure=current);load.add_pressure_load(1,1.)
    with pytest.raises(ValueError,match='not qualified'):
        load.get_load_vector(m.mesh,m.mesh.dof_manager,m.get_material,displacements=np.zeros(18))


def test_post_assembly_authority_failure_discards_trial(monkeypatch,tmp_path):
    from anyfesolver import nonlinear_static
    from anyfesolver import _ge_beam3_native_line_loading as loads
    m,e=problem();store=store_for(m,e);before=canonical(store.materialize());original=nonlinear_static._assemble_nonlinear_system
    def corrupt_after(*a,**k):
        result=original(*a,**k)
        object.__setattr__(loads._ACTIVE.get().pattern,'signature','0'*64)
        return result
    monkeypatch.setattr(nonlinear_static,'_assemble_nonlinear_system',corrupt_after)
    with pytest.raises(ValueError,match='authority changed'):assemble_line_trial(m,sample(),store,LinePattern(((1,*FORCE),)))
    assert not store.has_active_trial and not store.native_rotation_store.has_active_trial
    assert canonical(store.materialize())==before and loads._ACTIVE.get() is None
    save(tmp_path/'post-failure.json',dict(discarded=True,committed_unchanged=True,scope_reset=True))
