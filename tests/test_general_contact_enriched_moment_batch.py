"""Focused actual-old-operator oracles for the nonlinear-only point batch."""
import json
import time
import numpy as np
import pytest
from anyfesolver import FEModel,BoundaryCondition,create_shell_element
from anyfesolver.control import SolveCancelled
from anyfesolver._ge_beam3_mixed_ad import Jet2,sqrt as old_sqrt
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact.enriched_overlap import (
    moments as old_moments,QuadraturePolicy,EnrichedGeometryError)
from anyfesolver.general_contact.enriched_union import selected_pair_moments as old_selected
from anyfesolver.general_contact import enriched_moment_batch as kernel
from anyfesolver.general_contact.moment_batch_ad import PointJet,sqrt as batch_sqrt,choose
from anyfesolver.general_contact.enriched_nonlinear import (
    NonlinearEnrichedSystem,PairDefinition,MixedCriteria,IterationControl)

POLICY=QuadraturePolicy(1e-10,1e-8,1e-5,14144)

def small(*,paired=False,fixed=False):
    model=FEModel('point_batch_physical_oracle')
    model.add_material('elastic',1e5,.3,density=1.)
    source=((-0.5,-0.5,0.),(.5,-.5,0.),(.5,.5,0.),(-.5,.5,0.))
    target=((-2.,-2.,-.2),(2.,-2.,-.2),(2.,2.,-.2),(-2.,2.,-.2))if paired else ((-3.,-3.,-.2),(0.,4.,-.2),(4.,-3.,-.2))
    faces=((5,6,8),(6,7,8))if paired else((5,6,7),)
    for node,xyz in enumerate((*source,*target),1):model.add_node(node,*xyz)
    model.add_element(1,create_shell_element(1,(1,2,3,4),'elastic',thickness=.05,reference_normal=(0.,0.,1.)))
    pairs=[]
    for element,nodes in enumerate(faces,2):
        model.add_element(element,create_shell_element(element,nodes,'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.)if paired else(0.,0.,-1.),
            reference_surface_offset=.01))
        target_offset=model.mesh.elements[element].material_surface_offset_from_reference(1.)
        pairs.append(PairDefinition(1,element,-.025,target_offset))
    if fixed:model.add_boundary_condition(BoundaryCondition('fixed_nodal',tuple(model.mesh.nodes),
        {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    return model,NonlinearEnrichedSystem.from_model(model,pairs,normal_stiffness=1e4,quadrature=POLICY)

def arguments(system,u,index=0,*,selected=False):
    pair=system.pairs[index];x=pair.map(u)[0]
    args=(pair.reference_source,x[:12].reshape(4,3),x[12:21].reshape(3,3),x[21:25],pair.reference_directors[0],POLICY)
    kw=dict(current_midsurface=x[25:37].reshape(4,3))
    if selected:
        certificate=system.union_certificates(u)[str(pair.definition.source_element)]
        kw.update(internal_edge_nodes=certificate['shared_nodes'],target_node_ids=pair.target_node_ids)
    return args,kw

def array_errors(observed,expected):
    errors={}
    for name in ('support','weighted_gap','pressure_gram'):
        errors[name]=float(np.max(np.abs(getattr(observed,name)-getattr(expected,name))))
        assert errors[name]<=POLICY.value_absolute
    for name in ('support_gradient','gap_integral_gradient'):
        errors[name]=float(np.max(np.abs(getattr(observed,name)-getattr(expected,name))))
        assert errors[name]<=POLICY.gradient_absolute
    for name in ('support_hessian','gap_integral_hessian'):
        errors[name]=float(np.max(np.abs(getattr(observed,name)-getattr(expected,name))))
        assert errors[name]<=POLICY.hessian_absolute
    assert observed.branch==expected.branch
    assert 0<observed.triangles<=POLICY.maximum_triangles
    assert 0<expected.triangles<=POLICY.maximum_triangles
    assert np.all(np.asarray(observed.quadrature_difference)<=np.array((POLICY.value_absolute,POLICY.gradient_absolute,POLICY.hessian_absolute)))
    errors['triangles']=dict(old=expected.triangles,new=observed.triangles)
    errors['quadrature_difference']=dict(old=expected.quadrature_difference,new=observed.quadrature_difference)
    return errors

def _dense_constant_oracle(jet,other,divide=False):
    """Original dense PointJet product, with unchanged reciprocal if requested."""
    y=jet.coerce(other)
    if divide:y=y.reciprocal()
    return PointJet(jet.value*y.value,
        jet.gradient*y.value[:,None]+y.gradient*jet.value[:,None],
        jet.hessian*y.value[:,None,None]+y.hessian*jet.value[:,None,None]
        +jet.gradient[:,:,None]*y.gradient[:,None,:]
        +y.gradient[:,:,None]*jet.gradient[:,None,:])


def _assert_constant_result(actual,expected):
    for name in ('value','gradient','hessian'):
        a,b=getattr(actual,name),getattr(expected,name)
        np.testing.assert_array_equal(a,b)
        # assert_array_equal treats signed zeros as equal; their sign is observable.
        zero=(a==0.)&(b==0.)
        np.testing.assert_array_equal(np.signbit(a[zero]),np.signbit(b[zero]))
        assert not np.shares_memory(a,b)


def test_point_lanes_preserve_complete_original_jet_derivatives(record_property):
    original=[[Jet2.variable(.4+.01*j+.002*k,k,37)for k in range(37)]for j in range(25)]
    batched=[PointJet.stack([row[k]for row in original])for k in range(37)]
    def expression(x,sqrt):
        return sqrt(1+sum(a*a for a in x)/37)*(x[0]*x[17]+x[8]/(1+x[3]*x[3]))/(2+x[23]*x[23])
    expected=[expression(row,old_sqrt)for row in original];actual=expression(batched,batch_sqrt)
    errors=(float(np.max(np.abs(actual.value-np.array([x.value for x in expected])))),
        float(np.max(np.abs(actual.gradient-np.stack([x.gradient for x in expected])))),
        float(np.max(np.abs(actual.hessian-np.stack([x.hessian for x in expected])))))
    assert errors[0]<=1e-10 and errors[1]<=1e-8 and errors[2]<=1e-5
    assert actual.gradient.shape==(25,37)and actual.hessian.shape==(25,37,37)
    # Explicit heterogeneous lane mask: all values AND37gradient/Hessian
    # entries must select the matching original scalar jet, not one mask
    # for the wholebatch. This covers the inverse-map freeze primitive.
    mask=np.arange(25)%2==0
    selected=choose(mask,actual,-actual)
    scalar_selected=[value if flag else -value for flag,value in zip(mask,expected)]
    np.testing.assert_allclose(selected.value,[x.value for x in scalar_selected],rtol=0,atol=1e-10)
    np.testing.assert_allclose(selected.gradient,np.stack([x.gradient for x in scalar_selected]),rtol=0,atol=1e-8)
    np.testing.assert_allclose(selected.hessian,np.stack([x.hessian for x in scalar_selected]),rtol=0,atol=1e-5)
    with pytest.raises(ZeroDivisionError):PointJet.constant(0.,25,37).reciprocal()
    with pytest.raises(ValueError,match='positive'):batch_sqrt(PointJet.constant(-1.,25,37))
    record_property('original37_jet_errors',json.dumps(errors))
    # Finite scalar and per-lane constants: original scalar Jet2 remains the
    # independent complete37 derivative oracle, including reversed multiplication.
    constant_checks=0
    for constants in (0.,-0.,2.5,-3.,np.linspace(-2.,2.,25),np.linspace(.5,2.5,25)):
        values=np.broadcast_to(np.asarray(constants,dtype=float),(25,))
        for division in (False,True):
            if division and np.any(values==0.):continue
            observed=batched[0]/constants if division else batched[0]*constants
            scalar=[row[0]/float(c) if division else row[0]*float(c)
                    for row,c in zip(original,values)]
            np.testing.assert_allclose(observed.value,[x.value for x in scalar],rtol=0,atol=1e-10)
            np.testing.assert_allclose(observed.gradient,np.stack([x.gradient for x in scalar]),rtol=0,atol=1e-8)
            np.testing.assert_allclose(observed.hessian,np.stack([x.hessian for x in scalar]),rtol=0,atol=1e-5)
            _assert_constant_result(observed,_dense_constant_oracle(batched[0],constants,division))
            constant_checks+=1
        _assert_constant_result(constants*batched[0],_dense_constant_oracle(batched[0],constants))
    # Nonfinite derivatives MUST retain zero products, not drop 0*inf/NaN.
    # Original dense PointJet arithmetic is the IEEE/vector fallback oracle.
    exceptional=PointJet(batched[0].value.copy(),batched[0].gradient.copy(),batched[0].hessian.copy())
    exceptional.gradient[0,0]=np.inf;exceptional.hessian[1,0,1]=np.nan
    input_before=tuple(x.tobytes()for x in (exceptional.value,exceptional.gradient,exceptional.hessian))
    with np.errstate(all='ignore'):
        for candidate in (exceptional,PointJet(np.where(mask,np.inf,np.nan),
                          batched[0].gradient,batched[0].hessian)):
            for constants in (0.,-0.,-2.,np.inf,np.nan):
                _assert_constant_result(candidate*constants,_dense_constant_oracle(candidate,constants))
                if constants!=0.:
                    _assert_constant_result(candidate/constants,_dense_constant_oracle(candidate,constants,True))
            # Reciprocal coefficient overflow follows the unchanged unary fallback.
            for constants in (1e-200,1e200):
                _assert_constant_result(candidate/constants,_dense_constant_oracle(candidate,constants,True))
    assert input_before==tuple(x.tobytes()for x in (exceptional.value,exceptional.gradient,exceptional.hessian))
    class Converted:
        def __init__(self):self.calls=0
        def __array__(self,dtype=None,copy=None):
            self.calls+=1
            return np.asarray(2.,dtype=dtype)
    for division in (False,True):
        custom=Converted()
        observed=batched[0]/custom if division else batched[0]*custom
        assert custom.calls==1
        _assert_constant_result(observed,_dense_constant_oracle(batched[0],2.,division))
    for bad in (0.,np.zeros(25)):
        with pytest.raises(ZeroDivisionError):batched[0]/bad
        with pytest.raises(ZeroDivisionError):_dense_constant_oracle(batched[0],bad,True)
    for bad in (np.ones(2),'not-numeric'):
        for division in (False,True):
            with pytest.raises(ValueError):
                _dense_constant_oracle(batched[0],bad,division)
            with pytest.raises(ValueError):
                _=batched[0]/bad if division else batched[0]*bad
    class Derived(PointJet):pass
    for fallback in (Derived(batched[0].value,batched[0].gradient,batched[0].hessian),
                     PointJet(np.ones(25),np.empty((25,0)),np.empty((25,0,0)))):
        _assert_constant_result(fallback*-2.,_dense_constant_oracle(fallback,-2.))
        _assert_constant_result(fallback/2.,_dense_constant_oracle(fallback,2.,True))
    # Under raise mode the first nonfinite zero-product error must still escape.
    for operation in (lambda:exceptional*0.,lambda:_dense_constant_oracle(exceptional,0.)):
        with np.errstate(invalid='raise'):
            with pytest.raises(FloatingPointError):operation()
    record_property('constant_contraction_checks',constant_checks)
    record_property('constant_exception_contract','scalar/vector/coercion/zero/nonfinite/layout/signed-zero')


def test_smooth_moments_match_old_operator_and_moving_derivatives(record_property):
    _,system=small();u=np.linspace(-1e-4,2e-4,system.offset.size);u[system.physical_dofs:]=(2e-5,-1e-5,1.5e-5,5e-6)
    args,kw=arguments(system,u)
    start=time.perf_counter();old=old_moments(*args,**kw);old_seconds=time.perf_counter()-start
    start=time.perf_counter();new=kernel.batch_moments(*args,**kw);new_seconds=time.perf_counter()-start
    errors=array_errors(new,old)
    x=np.concatenate((args[1].ravel(),args[2].ravel(),args[3],kw['current_midsurface'].ravel()))
    d=np.linspace(-.2,.3,37)
    expected_g=np.vstack((new.support_gradient,new.gap_integral_gradient))@d
    expected_h=np.einsum('ijk,k->ij',np.concatenate((new.support_hessian,new.gap_integral_hessian)),d)
    assert np.linalg.norm(expected_g)>0 and np.linalg.norm(expected_h)>0
    fd_errors=[]
    def evaluate(v):
        return kernel.batch_moments(args[0],v[:12].reshape(4,3),v[12:21].reshape(3,3),
            v[21:25],args[4],POLICY,current_midsurface=v[25:].reshape(4,3))
    for h in (1e-5,3e-6):
        plus,minus=evaluate(x+h*d),evaluate(x-h*d)
        assert plus.branch==minus.branch==new.branch
        vg=(np.concatenate((plus.support,plus.weighted_gap))-np.concatenate((minus.support,minus.weighted_gap)))/(2*h)
        hh=(np.vstack((plus.support_gradient,plus.gap_integral_gradient))-np.vstack((minus.support_gradient,minus.gap_integral_gradient)))/(2*h)
        fd_errors.append((float(np.linalg.norm(vg-expected_g)/np.linalg.norm(expected_g)),
                          float(np.linalg.norm(hh-expected_h)/np.linalg.norm(expected_h))))
    assert max(max(v)for v in fd_errors)<=1e-5
    record_property('old_operator_absolute_errors',json.dumps(errors))
    record_property('same_child_single_pair_seconds',json.dumps(dict(old=old_seconds,new=new_seconds,not_authoritative_whole_process_charge=True)))
    record_property('all37_moving_direction_errors',json.dumps(fd_errors))

def test_internal_union_moments_match_old_operator_at_nonzero_state(record_property):
    model,system=small(paired=True);u=np.zeros(system.offset.size)
    for node in (5,6,7,8):
        dofs=model.mesh.dof_manager.get_node_dofs(node);u[dofs[2]]=.02*model.mesh.nodes[node].x
    u[system.physical_dofs:]=(2e-5,-1e-5,1.5e-5,5e-6)
    args,kw=arguments(system,u,selected=True)
    old=old_selected(*args,**kw);new=kernel.selected_batch_moments(*args,**kw)
    errors=array_errors(new,old)
    assert old.branch[-1]==1 and np.any(args[3]!=0)
    record_property('selected_internal_tie_old_operator_errors',json.dumps(errors))

def test_actual_nonlinear_caller_preserves_old_residual_tangent_and_identity(monkeypatch,record_property):
    _,system=small(paired=True);reduced=np.linspace(-1e-5,2e-5,system.transform.shape[1]);z=np.full(4,.01);load=np.zeros(system.offset.size)
    identity=system.identity;before=(reduced.tobytes(),z.tobytes(),load.tobytes());seen=[]
    actual_smooth,actual_selected=kernel.batch_moments,kernel.selected_batch_moments
    def observe_smooth(*args,**kw):seen.append('batch');return actual_smooth(*args,**kw)
    def observe_selected(*args,**kw):seen.append('selected_batch');return actual_selected(*args,**kw)
    with monkeypatch.context()as patch:
        patch.setattr(kernel,'batch_moments',observe_smooth);patch.setattr(kernel,'selected_batch_moments',observe_selected)
        actual=system.equations(reduced,z,load)
    assert seen and set(seen)=={'selected_batch'}
    with monkeypatch.context()as patch:
        patch.setattr(kernel,'batch_moments',old_moments);patch.setattr(kernel,'selected_batch_moments',old_selected)
        expected=system.equations(reduced,z,load)
    assert kernel.batch_moments is actual_smooth and kernel.selected_batch_moments is actual_selected
    np.testing.assert_array_equal(actual[0],expected[0])
    assert actual[1].branches==expected[1].branches
    eq=float(np.linalg.norm(actual[2]-expected[2])/.04);ncp=float(np.linalg.norm(actual[3]-expected[3],1)/.04)
    assert eq<=1e-6 and ncp<=1e-8 and np.linalg.norm(expected[4])>0
    tangent=float(np.linalg.norm(actual[4]-expected[4])/np.linalg.norm(expected[4]));assert tangent<=1e-5
    assert system.identity==identity and system._accepted is None and system.pressure_ids==(1,2,3,4)
    assert before==(reduced.tobytes(),z.tobytes(),load.tobytes())
    record_property('actual_mixed_caller_old_operator_errors',json.dumps(dict(equilibrium=eq,ncp=ncp,tangent=tangent)))

def test_kernel_domain_refusals_and_input_bytes_are_preserved():
    _,system=small();args,kw=arguments(system,np.zeros(system.offset.size))
    snapshot=[(a.tobytes(),a.flags.writeable)for a in args if isinstance(a,np.ndarray)]
    kernel.batch_moments(*args,**kw)
    assert snapshot==[(a.tobytes(),a.flags.writeable)for a in args if isinstance(a,np.ndarray)]
    for evaluator in (old_moments,kernel.batch_moments):
        invalid=list(args);invalid[2]=args[2].copy();invalid[2][0,0]=np.nan
        with pytest.raises(EnrichedProfileError,match='finite'):evaluator(*invalid,**kw)
        tied=list(args);tied[2]=args[1][[0,2,1]].copy()
        with pytest.raises(EnrichedGeometryError,match='clipping boundary'):evaluator(*tied,**kw)
        with pytest.raises(SolveCancelled,match='integration'):evaluator(*args,**kw,cancel=lambda:True)

def test_actual_caller_kernel_cancellation_and_definition_mutation_are_atomic(monkeypatch):
    model,system=small(fixed=True);load=np.zeros(system.offset.size);criteria=MixedCriteria(.04,1e-6,1e-8)
    system.solve(load,criteria=criteria,control=IterationControl(2,2))
    accepted=system._accepted;checkpoint=system.checkpoint();entered=[];actual=kernel.batch_moments
    def at_kernel(*args,**kw):
        entered.append('batch_moments')
        return actual(*args,**kw)
    with monkeypatch.context()as patch:
        patch.setattr(kernel,'batch_moments',at_kernel)
        with pytest.raises(SolveCancelled,match='integration'):
            system.solve(load,criteria=criteria,control=IterationControl(2,2),cancel=lambda:bool(entered))
    assert entered==['batch_moments']and system._accepted is accepted and system.checkpoint()==checkpoint and not system._busy
    original_z=model.mesh.nodes[1].z;model.mesh.nodes[1].z=original_z+.001;reached=[]
    def must_not_evaluate(*args,**kw):reached.append(True);return actual(*args,**kw)
    try:
        with monkeypatch.context()as patch:
            patch.setattr(kernel,'batch_moments',must_not_evaluate)
            with pytest.raises(EnrichedProfileError,match='definition changed'):
                system.linearize(np.zeros(system.offset.size),np.zeros(4))
        with pytest.raises(EnrichedProfileError,match='definition changed'):system.checkpoint()
        assert not reached and system._accepted is accepted and not system._busy
    finally:model.mesh.nodes[1].z=original_z
    assert system._accepted is accepted and system.checkpoint()==checkpoint and not system._busy
