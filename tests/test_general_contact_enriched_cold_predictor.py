"""New cold direction controls; original physical cold test remains unchanged."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver import FEModel,BoundaryCondition,create_shell_element
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact import enriched_cold_predictor as predictor
from anyfesolver.general_contact import enriched_nonlinear as nonlinear
from anyfesolver.general_contact.enriched_static import EnrichedStaticSystem,MixedCriteria,IterationControl
from anyfesolver.general_contact.enriched_nonlinear import NonlinearEnrichedSystem,PairDefinition
from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy


def certify(h,g,gap,r,z,d,p):
    np.testing.assert_allclose(h @ d-g.T @ (p-z),-r,rtol=0,atol=1e-12)
    w=gap+g @ d
    assert np.all(p>=0)and np.min(w)>=-1e-12
    np.testing.assert_allclose(p*w,0.,rtol=0,atol=1e-12)


def model_call(h,g,gap,r,z=None,**kw):
    h,g,gap,r=map(np.asarray,(h,g,gap,r))
    z=np.zeros(gap.size)if z is None else np.asarray(z)
    counts=dict(evaluations=1,linear_solves=0,rejected_trials=0)
    before=tuple(x.tobytes()for x in(h,g,gap,r,z))
    d,p,report=predictor.cold_unilateral_direction(h,g,gap,r,z,counts=counts,**kw)
    assert before==tuple(x.tobytes()for x in(h,g,gap,r,z))
    assert counts['evaluations']==1 and counts['rejected_trials']==0 and counts['linear_solves']>=1
    certify(h,g,gap,r,z,d,p)
    return d,p,report


def test_scalar_open_closed_and_absolute_force_model_matches_closed_form(record_property):
    h=np.array([[2.]]);g=np.array([[1.]]);gap=np.array([.25])
    d,p,_=model_call(h,g,gap,[.1])
    np.testing.assert_allclose(d,[-.05],rtol=0,atol=1e-14);assert p[0]==0
    d,p,report=model_call(h,g,gap,[1.])
    np.testing.assert_allclose(d,[-.25],rtol=0,atol=1e-14)
    np.testing.assert_allclose(p,[.5],rtol=0,atol=1e-14)
    # Current z is not an increment: r+G.T*z is the primal linear term.
    d,p,_=model_call(h,g,gap,[.7],z=[.3])
    np.testing.assert_allclose(d,[-.25],rtol=0,atol=1e-14)
    np.testing.assert_allclose(p,[.5],rtol=0,atol=1e-14)
    record_property('scalar_actual_model_certificate',json.dumps(report))


def test_coupled_bound_removal_and_small_nonsymmetric_actual_equations(record_property):
    # Known independent KKT solution. The third bound enters first, then leaves.
    qmat=np.array([[1.,0.,.7],[0.,1.,.7],[.7,.7,1.]])
    g=np.linalg.cholesky(qmat);h=np.eye(3);gap=np.full(3,.1)
    expected_p=np.array([1.,1.,0.]);expected_w=np.array([0.,0.,.1])
    q=-qmat @ expected_p+expected_w
    r=np.linalg.solve(g,gap-q)
    d,p,report=model_call(h,g,gap,r)
    np.testing.assert_allclose(p,expected_p,rtol=0,atol=1e-12)
    np.testing.assert_allclose(gap+g @ d,expected_w,rtol=0,atol=1e-12)
    assert report['active_solves']>2
    # Small finite skew is NOT replaced by its symmetric part in any solve.
    h=np.array([[2.,1e-14],[-1e-14,3.]])
    g=np.eye(2);gap=np.array([.1,.2]);expected_p=np.array([.4,.7]);expected_d=-gap
    r=g.T @ expected_p-h @ expected_d
    d,p,skew_report=model_call(h,g,gap,r)
    np.testing.assert_allclose(d,expected_d,rtol=0,atol=1e-12)
    np.testing.assert_allclose(p,expected_p,rtol=0,atol=1e-12)
    # Large nonsymmetry is outside the admitted work-conjugate numerical domain,
    # despite positive symmetric part. No symmetric-QP convergence claim.
    with pytest.raises(predictor.ColdPredictorError,match='nonconjugate'):
        model_call([[2.,.1],[-.1,2.]],np.eye(2),[.1,.2],[1.,1.])
    record_property('coupled_and_actual_skew_certificates',json.dumps([report,skew_report]))


def test_model_rank_nonfinite_cap_cancellation_and_input_atomicity(monkeypatch):
    for h,g,gap,r in (([[-1.]],[[1.]],[.1],[1.]),
                      ([[1.]],[[0.]],[.1],[1.]),
                      ([[1.]],[[1.],[1.]],[.1,.1],[1.]),
                      ([[np.nan]],[[1.]],[.1],[1.])):
        with pytest.raises(EnrichedProfileError):model_call(h,g,gap,r)
    # Refusal after the actual primal solve, not an earlier authority stub.
    counts=dict(linear_solves=0);seen=[]
    with monkeypatch.context()as patch:
        patch.setattr(predictor,'_active_solve_limit',lambda m:seen.append(m)or 0)
        with pytest.raises(predictor.ColdPredictorError,match='bounded'):
            predictor.cold_unilateral_direction([[2.]],[[1.]],[.25],[1.],[0.],counts=counts)
    assert seen==[1]and counts['linear_solves']==1
    h=np.array([[2.]]);g=np.array([[1.]]);gap=np.array([.25]);r=np.array([1.]);z=np.zeros(1)
    before=tuple(x.tobytes()for x in(h,g,gap,r,z));counts=dict(linear_solves=0)
    with pytest.raises(SolveCancelled):
        predictor.cold_unilateral_direction(h,g,gap,r,z,counts=counts,
            cancel=lambda:counts['linear_solves']>=1)
    assert counts['linear_solves']==1
    assert before==tuple(x.tobytes()for x in(h,g,gap,r,z))


def test_default_and_warm_direction_retain_original_newton(monkeypatch):
    j=np.array([[2.,-1.],[0.,1.]])
    r=np.array([1.]);c=np.array([.2]);counts=dict(linear_solves=0)
    expected=np.linalg.solve(j,-np.concatenate((r,c)))
    d=EnrichedStaticSystem._mixed_direction(None,np.zeros(1),np.zeros(1),None,r,c,j,
        iteration=0,counts=counts,cancel=lambda:False)
    np.testing.assert_array_equal(d,expected);assert counts['linear_solves']==1
    owner=object.__new__(NonlinearEnrichedSystem)
    object.__setattr__(owner,'_accepted',object())
    raw=SimpleNamespace(support=np.ones(1),normalized_gap=np.ones(1))
    def forbidden(*args,**kwargs):raise AssertionError('warm predictor must not run')
    with monkeypatch.context()as patch:
        patch.setattr(predictor,'cold_unilateral_direction',forbidden)
        for iteration in(0,1):
            counts=dict(linear_solves=0)
            d=owner._mixed_direction(np.zeros(1),np.zeros(1),raw,r,c,j,
                iteration=iteration,counts=counts,cancel=lambda:False)
            np.testing.assert_array_equal(d,expected);assert counts['linear_solves']==1
        # Noneligible warm route must not inspect a contact-state object.
        counts=dict(linear_solves=0)
        d=owner._mixed_direction(np.zeros(1),np.zeros(1),None,r,c,j,
            iteration=0,counts=counts,cancel=lambda:False)
        np.testing.assert_array_equal(d,expected);assert counts['linear_solves']==1
        object.__setattr__(owner,'_accepted',None)
        for reduced,z,iteration in((np.ones(1),np.zeros(1),0),
                                  (np.zeros(1),np.ones(1),0),(np.zeros(1),np.zeros(1),1)):
            counts=dict(linear_solves=0)
            d=owner._mixed_direction(reduced,z,None,r,c,j,
                iteration=iteration,counts=counts,cancel=lambda:False)
            np.testing.assert_array_equal(d,expected);assert counts['linear_solves']==1
        class OtherProfile(NonlinearEnrichedSystem):pass
        other=object.__new__(OtherProfile);object.__setattr__(other,'_accepted',None)
        counts=dict(linear_solves=0)
        d=other._mixed_direction(np.zeros(1),np.zeros(1),None,r,c,j,
            iteration=0,counts=counts,cancel=lambda:False)
        np.testing.assert_array_equal(d,expected);assert counts['linear_solves']==1
    # Synthetic exact hook-layout control: unsupported force is retained at0;
    # no physical-model evidence is inferred from this unit control.
    verified=[]
    object.__setattr__(owner,'_nodal',SimpleNamespace(verify=lambda:verified.append(True)))
    object.__setattr__(owner,'transform',np.ones((1,1)))
    raw=SimpleNamespace(support=np.array([1.,0.]),normalized_gap=np.array([.25,0.]),
                        gap_gradient=np.array([[1.],[0.]]))
    j=np.diag([2.,1.,1.]);j[0,1]=-1.
    d=owner._mixed_direction(np.zeros(1),np.zeros(2),raw,np.array([1.]),np.zeros(2),j,
        iteration=0,counts=dict(linear_solves=0),cancel=lambda:False)
    np.testing.assert_allclose(d,[-.25,.5,0.],rtol=0,atol=1e-14)
    assert len(verified)==2 and d[-1]==0.
    # Genuine algebraic-domain decline on an invertible indefinite model:
    # product fallback is exactly the old mixed Newton result, no regularization.
    j[0,0]=-2.
    counts=dict(linear_solves=0)
    delta,report=owner._cold_direction(np.zeros(1),np.zeros(2),raw,np.array([1.]),j,
        counts=counts,cancel=lambda:False)
    assert delta is None and report['status']=='DECLINED'and counts['linear_solves']==0
    d=owner._mixed_direction(np.zeros(1),np.zeros(2),raw,np.array([1.]),np.zeros(2),j,
        iteration=0,counts=counts,cancel=lambda:False)
    np.testing.assert_array_equal(d,np.linalg.solve(j,[-1.,0.,0.]))
    assert counts['linear_solves']==1


def small_owner():
    model=FEModel('cold_direction_actual_owner')
    model.add_material('elastic',1e5,.3,density=1.)
    vertices=((-0.5,-0.5,0.),(.5,-.5,0.),(.5,.5,0.),(-.5,.5,0.),
              (-3.,-3.,-.2),(0.,4.,-.2),(4.,-3.,-.2))
    for i,xyz in enumerate(vertices,1):model.add_node(i,*xyz)
    model.add_element(1,create_shell_element(1,(1,2,3,4),'elastic',thickness=.05,
        reference_normal=(0.,0.,1.)))
    model.add_element(2,create_shell_element(2,(5,6,7),'elastic',thickness=.05,
        formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,-1.),reference_surface_offset=.01))
    target_offset=model.mesh.elements[2].material_surface_offset_from_reference(1.)
    model.add_boundary_condition(BoundaryCondition('fixed_nodal',tuple(model.mesh.nodes),
        {name:0. for name in('ux','uy','uz','rx','ry','rz')}))
    owner=NonlinearEnrichedSystem.from_model(model,[PairDefinition(1,2,-.025,target_offset)],
        normal_stiffness=1e4,quadrature=QuadraturePolicy(1e-10,1e-8,1e-5,14144))
    return model,owner


def test_actual_cold_direction_and_warm_checkpoint_failures_are_atomic(monkeypatch,record_property):
    original=predictor.cold_unilateral_direction;observed=[]
    for mode in('refusal','cancellation','mutation','decline'):
        model,owner=small_owner();load=np.zeros(owner.offset.size);load[owner.physical_dofs]=.01
        assert owner._accepted is None
        with pytest.raises(EnrichedProfileError,match='no accepted'):owner.checkpoint()
        coordinate=model.mesh.nodes[1].z
        def inspected(h,g,gap,r,z,*,counts,cancel):
            assert owner._busy and owner._accepted is None and np.all(z==0)
            assert h.shape==(4,4)and g.shape==(4,4)and np.all(gap>0)
            result=original(h,g,gap,r,z,counts=counts,cancel=cancel)
            observed.append(dict(mode=mode,certificate=result[2]))
            if mode=='cancellation':raise SolveCancelled('post-model caller control')
            if mode=='decline':raise predictor.ColdPredictorDeclined('expected algebraic decline')
            if mode=='mutation':model.mesh.nodes[1].z=coordinate+1e-4
            else:raise predictor.ColdPredictorError('post-model caller refusal')
            return result
        with monkeypatch.context()as patch:
            patch.setattr(predictor,'cold_unilateral_direction',inspected)
            fallback=[]
            if mode=='decline':
                default=EnrichedStaticSystem._mixed_direction
                def fallback_then_cancel(self,*args,**kwargs):
                    value=default(self,*args,**kwargs);fallback.append(True)
                    raise SolveCancelled('after actual algebraic-decline fallback')
                patch.setattr(EnrichedStaticSystem,'_mixed_direction',fallback_then_cancel)
            try:
                with pytest.raises(SolveCancelled if mode in('cancellation','decline')else EnrichedProfileError):
                    owner.solve(load,criteria=MixedCriteria(.04,1e-6,1e-8),control=IterationControl(18,8))
            finally:model.mesh.nodes[1].z=coordinate
        if mode=='decline':assert fallback==[True]
        assert owner._accepted is None and owner._busy is False
        with pytest.raises(EnrichedProfileError,match='no accepted'):owner.checkpoint()
    # A genuine accepted owner exists only on the WARM path. After an actual
    # default solve-factorization cancellation its exact accepted bytes survive.
    _,owner=small_owner();criteria=MixedCriteria(.04,1e-6,1e-8)
    owner.solve(np.zeros(owner.offset.size),criteria=criteria,control=IterationControl(18,8))
    accepted=owner._accepted;checkpoint=owner.checkpoint();reached=[]
    default=EnrichedStaticSystem._mixed_direction
    def cancelled(self,*args,**kwargs):
        result=default(self,*args,**kwargs);reached.append(True)
        raise SolveCancelled('after actual warm default direction')
    load=np.zeros(owner.offset.size);load[owner.physical_dofs]=.01
    with monkeypatch.context()as patch:
        patch.setattr(EnrichedStaticSystem,'_mixed_direction',cancelled)
        with pytest.raises(SolveCancelled):
            owner.solve(load,criteria=criteria,control=IterationControl(18,8))
    assert reached==[True]and owner._accepted is accepted and owner._busy is False
    assert owner.checkpoint()==checkpoint
    assert [r['mode']for r in observed]==['refusal','cancellation','mutation','decline']
    record_property('actual_cold_direction_calls_and_certificates',json.dumps(observed))
