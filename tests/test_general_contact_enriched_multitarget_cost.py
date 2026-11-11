"""V18 complete-caller cost equivalence and inter-iteration refusal controls."""
import json
import numpy as np
import pytest

from anyfesolver.control import SolveCancelled
from anyfesolver.matrix_assembly import AssemblyError
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy, EnrichedGeometryError
from anyfesolver.general_contact.enriched_multitarget import MultiTargetSystem
from anyfesolver.general_contact._enriched_multitarget_trial import AcceptedTrial
from anyfesolver.general_contact.enriched_static import MixedCriteria, IterationControl


def fixture():
    # Real protected nonlinear assembly, all explicit B4 modes, real public S3.
    from anyfesolver import FEModel,BoundaryCondition,create_shell_element
    from anyfesolver.general_contact.enriched_nonlinear import NonlinearEnrichedSystem,PairDefinition
    model=FEModel('multitarget_reuse_real_elastic')
    model.add_material('elastic',1e5,.3,density=1.)
    source=((-0.5,-0.5,0.),(.5,-.5,0.),(.5,.5,0.),(-.5,.5,0.))
    target=((-2.,-2.,-.2),(2.,-2.,-.2),(2.,2.,-.2),(-2.,2.,-.2))
    for i,xyz in enumerate(source+target,1):model.add_node(i,*xyz)
    model.add_element(1,create_shell_element(1,(1,2,3,4),'elastic',thickness=.05,reference_normal=(0.,0.,1.)))
    definitions=[]
    for i,face in enumerate(((5,6,8),(6,7,8)),2):
        model.add_element(i,create_shell_element(i,face,'elastic',thickness=.05,
            reference_normal=(0.,0.,1.),reference_surface_offset=.01))
        definitions.append(PairDefinition(1,i,-.025,.015))
    model.add_boundary_condition(BoundaryCondition('fixed_nodal',tuple(model.mesh.nodes),
        {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    base=NonlinearEnrichedSystem.from_model(model,definitions,normal_stiffness=1e4,
        quadrature=QuadraturePolicy(1e-10,1e-8,1e-5,14144))
    system = object.__new__(MultiTargetSystem)
    for name, value in vars(base).items():
        object.__setattr__(system, name, value)
    object.__setattr__(system, '_start', None)
    load = np.zeros(system.offset.size)
    load[system.physical_dofs:] = (1e-4, -2e-5, 3e-5, 1e-5)
    return model, system, load


CRITERIA = MixedCriteria(.04, 1e-6, 1e-8)
CONTROL = IterationControl(18, 8)


def test_bounded_value_graph_matches_scalar_jet_and_exception_fallback():
    from anyfesolver._ge_beam3_mixed_ad import Jet2 as Full, sqrt as full_sqrt
    from anyfesolver.general_contact._enriched_multitarget_empty import Jet2 as Bound, sqrt
    for value in (.125, 1., 3.5):
        gradient = np.linspace(-.3, .4, 37)
        hessian = np.outer(gradient, gradient)
        a = Full(value, gradient, hessian)
        b = Bound(value, float(np.max(np.abs(gradient))), float(np.max(np.abs(hessian))))
        for old, new in ((a/a, b/b), (full_sqrt(a), sqrt(b)), (a*.3+a, b*.3+b)):
            assert old.value.hex() == new.value.hex()
            assert np.max(np.abs(old.gradient)) <= new.g
            assert np.max(np.abs(old.hessian)) <= new.h
    with pytest.raises(ZeroDivisionError): Bound(0.).reciprocal()
    with pytest.raises(ValueError): sqrt(Bound(-1.))


def test_empty_full37_values_signatures_guards_and_nonempty_fallback(record_property):
    from anyfesolver.general_contact._enriched_multitarget_empty import empty_preflight
    from anyfesolver.general_contact.enriched_multitarget_moments import (
        selected_batch_moments, _selected_batch_moments_full)
    ref = np.array(((-.5,-.5,0.),(.5,-.5,0.),(.5,.5,0.),(-.5,.5,0.)))
    target = np.array(((3.,3.,-.2),(3.,4.,-.2),(4.,3.,-.2)))
    args = (ref,ref.copy(),target,np.array((1e-4,2e-5,-3e-5,4e-6)),np.array((0.,0.,1.)),
            QuadraturePolicy(1e-10,1e-8,1e-5,14144))
    kw = dict(current_midsurface=ref.copy(),internal_edges=(),target_node_ids=(5,6,7))
    old = _selected_batch_moments_full(*args,**kw)
    new = selected_batch_moments(*args,**kw)
    assert empty_preflight(*args,**kw) is not None and old.triangles == new.triangles == 0
    assert old.branch == new.branch and old.quadrature_difference == new.quadrature_difference
    for name in ('support','weighted_gap','pressure_gram','support_gradient',
                 'gap_integral_gradient','support_hessian','gap_integral_hessian'):
        assert np.array_equal(getattr(old,name),getattr(new,name))
    for bad in (target[::-1].copy(),np.zeros((3,3))):
        with pytest.raises(EnrichedGeometryError): selected_batch_moments(*(args[:2]+(bad,)+args[3:]),**kw)
        with pytest.raises(EnrichedGeometryError): _selected_batch_moments_full(*(args[:2]+(bad,)+args[3:]),**kw)
    with np.errstate(all='raise'): assert empty_preflight(*args,**kw) is None
    # Iterator-bearing requests must reach the original function only once.
    assert empty_preflight(*args,**dict(kw,internal_edges=iter(()))) is None
    crossing=np.array(((-3.,-3.,-.2),(0.,4.,-.2),(4.,-3.,-.2)))
    assert empty_preflight(*(args[:2]+(crossing,)+args[3:]),**kw) is None
    record_property('empty_oracle',json.dumps(dict(branch=old.branch,triangles=0,full37=True)))


def test_actual_trial_reuse_matches_recomputed_equations(monkeypatch,record_property):
    _, fast, load = fixture()
    virgin=fast._nodal.states
    assert type(virgin) is dict and not virgin
    result = fast.solve(load,criteria=CRITERIA,control=CONTROL)
    assert result['counts']['accepted_trial_reuses'] > 0
    assert fast._nodal.states is virgin and not virgin
    _, original, original_load = fixture()
    monkeypatch.setattr(AcceptedTrial,'capture',classmethod(lambda cls,*a,**k:None))
    baseline = original.solve(original_load,criteria=CRITERIA,control=CONTROL)
    for name in ('coordinates','reduced','nodal_forces','load'):
        assert np.array_equal(getattr(fast._accepted,name),getattr(original._accepted,name))
    assert result['equilibrium'] == baseline['equilibrium']
    assert result['complementarity'] == baseline['complementarity']
    assert result['counts']['evaluations'] < baseline['counts']['evaluations']
    record_property('actual_equations_reuse',json.dumps(dict(fast=result['counts'],baseline=baseline['counts'])))


def test_actual_interiteration_cancel_mutation_nested_and_cache_refusals(monkeypatch,record_property):
    from anyfesolver.general_contact.runtime import _ACTIVE_CONTACT
    original = AcceptedTrial.consume
    reached = {}
    for mode in ('cancel','model','nested'):
        model, system, load = fixture()
        system.solve(np.zeros_like(load),criteria=CRITERIA,control=CONTROL)
        accepted, checkpoint = system._accepted, system.checkpoint()
        flag=[False];token=[None]
        node=model.mesh.nodes[1];old_z=node.z
        old_cache=getattr(model.mesh,'_sparsity_cache',None)
        def observe(self,*args,**kwargs):
            reached[mode]=reached.get(mode,0)+1
            if mode=='cancel':flag[0]=True
            elif mode=='model':node.z=old_z+.001
            elif mode=='nested':token[0]=_ACTIVE_CONTACT.set(object())
            return original(self,*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(AcceptedTrial,'consume',observe)
            try:
                with pytest.raises((SolveCancelled,EnrichedProfileError,ValueError)):
                    system.solve(load,criteria=CRITERIA,control=CONTROL,cancel=lambda:flag[0])
            finally:
                node.z=old_z
                if token[0] is not None:_ACTIVE_CONTACT.reset(token[0])
        assert reached[mode]==1 and system._accepted is accepted and not system._busy
        assert system.checkpoint()==checkpoint
    record_property('actual_consumption_refusals',json.dumps(reached))


def test_trial_scope_operator_change_and_final_callback_fallback(monkeypatch):
    _, system, load = fixture()
    q=np.zeros(system.transform.shape[1]);z=np.zeros(len(system.pressure_ids))
    deps=AcceptedTrial.dependencies(system,q,z,load)
    result=system.equations(q,z,load)
    token=AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,deps)
    assert token is not None
    _, other, _ = fixture()
    old=system.equations
    object.__setattr__(system,'equations',other.equations)
    try: assert token.consume(system,q,z,load,CRITERIA,CONTROL,lambda:False) is None
    finally:object.__setattr__(system,'equations',old)
    token=AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,AcceptedTrial.dependencies(system,q,z,load))
    calls=[0];scales=system.scales
    def mutate_last():
        calls[0]+=1
        if calls[0]==1:object.__setattr__(system,'scales',scales.copy()*2)
        return False
    try: assert token.consume(system,q,z,load,CRITERIA,CONTROL,mutate_last) is None
    finally:object.__setattr__(system,'scales',scales)
    token=AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,AcceptedTrial.dependencies(system,q,z,load))
    with pytest.raises(EnrichedProfileError):token.consume(other,q,z,load,CRITERIA,CONTROL,lambda:False)
    with pytest.raises(EnrichedProfileError):token.consume(system,q,z,load+1,CRITERIA,CONTROL,lambda:False)
    assert token.consume(system,q,z,load,CRITERIA,CONTROL,lambda:False) is result
    with pytest.raises(EnrichedProfileError):token.consume(system,q,z,load,CRITERIA,CONTROL,lambda:False)
    # Benign changed cache eligibility falls through to the original evaluator.
    token=AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,AcceptedTrial.dependencies(system,q,z,load))
    _,baseline,_=fixture()
    system._nodal.model.mesh._sparsity_cache={}
    baseline._nodal.model.mesh._sparsity_cache={}
    assert token.consume(system,q,z,load,CRITERIA,CONTROL,lambda:False) is None
    def original_outcome(owner):
        try:return owner.equations(q,z,load),None
        except (AssemblyError,ElementCapabilityError) as exc:return None,type(exc)
    actual,error=original_outcome(system);expected,old_error=original_outcome(baseline)
    assert error is old_error
    if error is None:
        for i in (0,2,3,4):assert np.array_equal(actual[i],expected[i])
    # The actual profile owns an empty legacy dict, not a manufactured Store.
    states=system._nodal.states
    assert type(states) is dict and dict.__len__(states)==0
    for changed in ({}, {999:None}):
        deps=AcceptedTrial.dependencies(system,q,z,load)
        token=AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,deps)
        assert token is not None
        system._nodal.states=changed
        try:assert token.consume(system,q,z,load,CRITERIA,CONTROL,lambda:False) is None
        finally:system._nodal.states=states
    deps=AcceptedTrial.dependencies(system,q,z,load)
    token=AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,deps)
    states[999]=None
    try:
        assert token.consume(system,q,z,load,CRITERIA,CONTROL,lambda:False) is None
        assert AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,
            AcceptedTrial.dependencies(system,q,z,load)) is None
    finally:states.clear()
    # Unknown stable providers are not admitted merely because their identity is stable.
    object.__setattr__(system,'equations',lambda *a,**k:result)
    try:
        deps=AcceptedTrial.dependencies(system,q,z,load)
        assert AcceptedTrial.capture(system,q,z,load,CRITERIA,CONTROL,result,lambda:False,deps) is None
    finally:object.__setattr__(system,'equations',old)
