"""Finite BL5 mechanics/state checks; no selection or physical qualification."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_rotation_charts import owned_anchors, physical_frames
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_l1_assembly_experimental import _bl5_nonmatching_definition


def _model(edges=3, basis="p0"):
    definition = _bl5_nonmatching_definition(edges, 1)
    if basis != "p0":
        definition = replace(definition, lines=(replace(definition.lines[0],basis=basis),))
    return definition


def _action(force=100.):
    return (stiffened_shell.NodalAction("beam",105,(0.,0.,force,0.,0.,0.)),)


def _evaluate(driver,state,pattern):
    return driver._assemble(state,0.,driver._shell_origins,driver._beam_origins,pattern=pattern)


@pytest.mark.parametrize("basis",["p0","continuous_p1_coarse"])
@pytest.mark.parametrize("objective",[False,True])
def test_finite_line_reference_operator_matches_existing_sparse_linear(basis,objective):
    definition = _model(basis=basis)
    if objective:
        definition = replace(definition,q4_pullback_policy=stiffened_shell.Q4_ALGORITHMIC_PULLBACK_V1)
    owner=stiffened_shell.create_analysis(definition=definition)
    driver=FinitePointDriver(owner)
    r,k,_,gap,_,_=_evaluate(driver,driver._state,np.zeros(owner._K.shape[0]))
    assert np.max(np.abs(r))<1e-10 and gap<1e-13
    free=np.r_[owner._free,np.arange(owner._K.shape[0],k.shape[0])]
    delta=(k[free,:][:,free]-owner._A).tocoo()
    assert np.max(np.abs(delta.data),initial=0.) < 2e-5
    assert len(owner._line_terms)==len(owner._line_samples)


@pytest.mark.parametrize("objective", [False, True])
def test_nonzero_multiplier_derivative_through_actual_owned_chart_updates(objective):
    model = _model()
    if objective:
        model = replace(model, q4_pullback_policy=stiffened_shell.Q4_ALGORITHMIC_PULLBACK_V1)
    owner=stiffened_shell.create_analysis(definition=model)
    driver=FinitePointDriver(owner); rng=np.random.default_rng(817)
    count=owner._K.shape[0]
    delta=rng.normal(size=count+owner._C.shape[0])*.002
    # Keep constraint forces above cancellation from subtracting two large
    # element residuals; this arbitrary virtual multiplier is not an equilibrium.
    delta[count:]=rng.normal(size=owner._C.shape[0])*50000.
    state=driver._advance(driver._state,delta,1.)
    anchors=owned_anchors(np.tile(rotation(np.array([.04,-.03,.02])),
                                 (len(owner._shell_ids),1,1)),len(owner._shell_ids))
    state=replace(state,shell_anchors=anchors)
    zero=replace(state,multipliers=np.zeros_like(state.multipliers))
    pattern=np.zeros(count)
    r,k,*_=_evaluate(driver,state,pattern)
    r0,k0,*_=_evaluate(driver,zero,pattern)
    direction=rng.normal(size=len(r)); direction[count:]=0.
    expected=(k-k0)@direction
    errors=[]
    for h in (1e-5,3e-6,1e-6):
        pairs=[]
        for sign in (-1.,1.):
            moved=driver._advance(state,direction,sign*h)
            moved0=replace(moved,multipliers=np.zeros_like(moved.multipliers))
            rr,*_=_evaluate(driver,moved,pattern)
            rz,*_=_evaluate(driver,moved0,pattern)
            pairs.append(rr-rz)
        derivative=(pairs[1]-pairs[0])/(2*h)
        errors.append(np.linalg.norm(derivative-expected)/max(np.linalg.norm(expected),1.))
    assert min(errors)<1e-6,errors
    assert np.linalg.norm(expected)>1.


@pytest.mark.parametrize("edges,basis",[(3,"p0"),(2,"continuous_p1_coarse")])
def test_nonmatching_finite_history_physical_recovery_rebase_restart(edges,basis):
    definition=_model(edges,basis); owner=stiffened_shell.create_analysis(definition=definition)
    result=owner.solve(_action(),targets=(.5,1.))
    assert result.status=="completed" and result.accepted_factors==(.5,1.)
    assert result.free_equilibrium_ratio<1e-8 and result.constraint_max<1e-9
    assert len(result.beam_shell_line_work)==len(owner._line_samples)
    driver=owner._finite_driver
    positions=driver._beam_positions(driver._state)
    shell=np.array(definition.shell.coordinates_m)+result.shell_displacement.reshape(-1,6)[:,:3]
    for sample,record in zip(owner._line_samples,result.beam_shell_line_work):
        op,*_=sample; slots=record.physical_slots; g=record.integrated_spatial_gradient
        assert slots.dtype.kind in "iu" and record.constraint_rows.dtype.kind in "iu"
        np.testing.assert_array_equal(driver._state.q[slots],driver._state.q[sample[1]])
        beam_nodes=op.port.nodes
        endpoints=[shell[slots[0]//6],shell[slots[6]//6],positions[beam_nodes[0]],positions[beam_nodes[1]]]
        forces=[g[:3],g[6:9],g[12:15],g[15:18]]
        np.testing.assert_allclose(sum(forces),0.,atol=1e-9)
        moment=sum(np.cross(x,f) for x,f in zip(endpoints,forces))+g[3:6]+g[9:12]+g[18:21]
        np.testing.assert_allclose(moment,0.,atol=1e-8)
    saved=owner.checkpoint(); restored=stiffened_shell.create_analysis(definition=definition)
    repeated=restored.restore(saved)
    assert restored.checkpoint()==saved
    np.testing.assert_array_equal(repeated.beam_positions,result.beam_positions)
    rebased=owner.rebase_rotations(expected_generation=result.generation)
    charted=owner.checkpoint(); anchored=stiffened_shell.create_analysis(definition=definition)
    anchored.restore(charted); assert anchored.checkpoint()==charted
    a=owner.solve(_action(),targets=(.75,)); b=anchored.solve(_action(),targets=(.75,))
    np.testing.assert_array_equal(a.beam_positions,b.beam_positions)
    assert owner.checkpoint()==anchored.checkpoint()
    unloaded=stiffened_shell.create_analysis(definition=definition)
    unloaded.restore(owner.checkpoint()); assert unloaded.checkpoint()==owner.checkpoint()
    direct=owner.solve(_action(),targets=(.9,)); restarted=unloaded.solve(_action(),targets=(.9,))
    np.testing.assert_array_equal(direct.beam_positions,restarted.beam_positions)
    assert unloaded.checkpoint()==owner.checkpoint()
    assert owner.capabilities["status"]=="EXPERIMENTAL_NOT_QUALIFIED"


def test_line_rejection_preserves_accepted_prefix_and_new_schema_is_required(monkeypatch):
    definition=_model(); owner=stiffened_shell.create_analysis(definition=definition)
    owner.solve(_action(),targets=(.5,)); saved=owner.checkpoint(); generation=owner._generation
    with pytest.raises(RuntimeError,match="iteration limit"):
        owner.solve(_action(),targets=(1.,),max_iterations=1)
    assert owner._generation==generation and owner.checkpoint()==saved
    import json
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode,encode,SCHEMAS
    kind,gen,arrays=decode(saved,owner._identity,lambda _,g:tuple(
        (k,v.shape) for k,v in {"q":owner._finite_driver._state.q,
         "frames":owner._finite_driver._state.frames,"cells":owner._finite_driver._state.cells,
         "resultants":owner._finite_driver._state.resultants,"multipliers":owner._finite_driver._state.multipliers,
         "pattern":owner._finite_driver._pattern,"accepted_factors":np.zeros(g)}.items()))
    assert kind=="finite_bl5"
    legacy=encode("finite",owner._identity,gen,tuple(arrays.items()))
    fresh=stiffened_shell.create_analysis(definition=definition)
    with pytest.raises(ValueError):fresh.restore(legacy)
    assert fresh._generation==0 and fresh._finite_driver is None
    for schema in ("finite_bl5", "finite_chart_bl5"):
        with pytest.raises(ValueError,match="finite BL5 checkpoint requires binary framing"):
            fresh.restore(json.dumps({"schema":SCHEMAS[schema]}).encode())
        assert fresh._generation==0 and fresh._finite_driver is None


def test_finite_line_adaptive_cancel_restore_and_resume(monkeypatch):
    import anyfesolver._stiffened_shell_ge_finite_experimental as finite
    definition=_model(); owner=stiffened_shell.create_analysis(definition=definition)
    original=finite.cancellation_safe_point; token=CancellationToken()
    control=stiffened_shell.AdaptiveIncrementControl(2,8)
    def cancel(value,stage):
        if owner._generation==1 and stage=="stiffened-shell.finite.increment":
            token.cancel("line accepted prefix")
        return original(value,stage)
    with monkeypatch.context() as context:
        context.setattr(finite,"cancellation_safe_point",cancel)
        with pytest.raises(SolveCancelled,match="line accepted prefix"):
            owner.solve(_action(),targets=(.5,1.),increment_control=control,cancellation_token=token)
    prefix=owner.recover(); assert prefix.status=="accepted_prefix"
    saved=owner.checkpoint(); fresh=stiffened_shell.create_analysis(definition=definition)
    fresh.restore(saved); assert fresh.checkpoint()==saved
    for analysis in (owner,fresh):
        result=analysis.solve(_action(),targets=(.5,1.),increment_control=control)
        assert result.status=="completed" and result.accepted_factors==(.5,1.)
    assert owner.checkpoint()==fresh.checkpoint()


def test_typed_line_domain_failure_backtracks_but_programming_error_propagates(monkeypatch):
    from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator,BL5TrialDomainError
    original=BL5LocalOperator.evaluate_spatial; rejected=[]
    def candidate_failure(self,total,frames,state,multiplier):
        if not rejected and np.any(multiplier):
            rejected.append(True)
            raise BL5TrialDomainError("injected chart boundary")
        return original(self,total,frames,state,multiplier)
    owner=stiffened_shell.create_analysis(definition=_model())
    with monkeypatch.context() as context:
        context.setattr(BL5LocalOperator,"evaluate_spatial",candidate_failure)
        result=owner.solve(_action(),targets=(1.,))
    assert result.status=="completed" and len(rejected)==1
    saved=owner.checkpoint(); failure=ValueError("programming guard"); calls=[]
    def invalid(*args,**kwargs):
        calls.append(True); raise failure
    with monkeypatch.context() as context:
        context.setattr(BL5LocalOperator,"evaluate_spatial",invalid)
        with pytest.raises(ValueError) as caught:owner.solve(_action(),targets=(.75,))
    assert caught.value is failure and len(calls)==1 and owner.checkpoint()==saved


def test_cancel_line_before_publish_keeps_fresh_owner(monkeypatch):
    import anyfesolver._stiffened_shell_ge_finite_experimental as finite
    owner=stiffened_shell.create_analysis(definition=_model()); token=CancellationToken()
    original=finite.cancellation_safe_point
    def cancel(value,stage):
        if stage=="stiffened-shell.finite.publish":token.cancel("line before publish")
        return original(value,stage)
    with monkeypatch.context() as context:
        context.setattr(finite,"cancellation_safe_point",cancel)
        with pytest.raises(SolveCancelled):owner.solve(_action(),cancellation_token=token)
    assert owner._generation==0 and owner._finite_driver is None
