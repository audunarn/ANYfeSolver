"""Source controls: passive seam mechanics, ownership and restoration."""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp, so3_log
from examples import stiffened_shell_native_fold_local_diagnostic as local


@pytest.fixture
def seam(monkeypatch):
    elements = [object() for _ in range(16)]
    model = object(); origin = object(); seen = []; returns = []
    local_coordinates = np.arange(24,dtype=float)
    differential = np.eye(24); second = np.zeros((24,24,24))
    force = np.arange(24,dtype=float)+1.; stiffness = np.eye(24)*2.
    candidate = object(); payload = object()
    def deformation(reference,u,**options):
        return local_coordinates,differential,second
    def spatial(u,f,k,d,s,**options):
        return f,k
    def response(actual_model,element,u,actual_origin,layers,**options):
        seen.append((actual_model,element,u,actual_origin,layers,options))
        coords,d,s = local.q4.deformation(np.zeros((4,3)),u,stable_translation=True,
            rotation_anchors=None,stable_rotation=False)
        residual,tangent = local.q4.spatial_response(u,force,stiffness,d,s,stable_rotation=False)
        returned = (residual,tangent,candidate,payload)
        returns.append(returned)
        return returned
    monkeypatch.setattr(local.q4,'response',response)
    monkeypatch.setattr(local.finite,'q4_algorithmic_response',response)
    monkeypatch.setattr(local.q4,'deformation',deformation)
    monkeypatch.setattr(local.q4,'spatial_response',spatial)
    owner = SimpleNamespace(_shell_model=model,
        _shell_parts=[(i,e,np.arange(24),None) for i,e in enumerate(elements)])
    def assemble(state,factor,shell_origins,beam_origins,*,applied_load):
        owner.scratch = state  # Ordinary evaluator scratch lifecycle must remain.
        for e in elements:
            local.finite.q4_algorithmic_response(model,e,np.zeros(24),origin,3,
                prepared_correction=payload,capture_material_trial=True)
        return (np.zeros(552),None,0.,0.,(),())
    owner._finite_driver = SimpleNamespace(_assemble=assemble,_shell_origins=(origin,)*16)
    return SimpleNamespace(owner=owner,elements=elements,model=model,origin=origin,
        response=response,deformation=deformation,spatial=spatial,seen=seen,returns=returns,
        force=force,stiffness=stiffness,coords=local_coordinates)


def test_original_calls_return_identity_and_owned_captures(seam):
    with local.Q4LocalCapture(seam.owner) as capture:
        returned = local.finite.q4_algorithmic_response(seam.model,seam.elements[0],
            np.zeros(24),seam.origin,3,capture_material_trial=True)
        assert returned is seam.returns[0]
        assert capture.selected_deformation is seam.deformation
        assert capture.records[0]['numerical_candidate'] == local.BASELINE_NUMERICS
        assert capture.records[0]['candidate_log_calls'] == 0
        assert returned[0] is seam.force and returned[1] is seam.stiffness
        assert seam.seen[0][3] is seam.origin
        saved = deepcopy(capture.records)
        seam.force[:] = -90.; seam.stiffness[:] = -80.; seam.coords[:] = -70.
        np.testing.assert_array_equal(capture.records[0]['local_force'],np.arange(24)+1.)
        np.testing.assert_array_equal(capture.records[0]['local_coordinates'],np.arange(24))
        capture.records[0]['derived_work_covector'][:] = -60.
        np.testing.assert_array_equal(saved[0]['derived_work_covector'],np.arange(24)+1.)
    assert local.finite.q4_algorithmic_response is seam.response
    assert local.q4.deformation is seam.deformation
    assert local.q4.spatial_response is seam.spatial


def test_all_actual_elements_captured_without_restoring_evaluator_scratch(seam):
    state = object()
    with local.Q4LocalCapture(seam.owner) as capture:
        _,records = capture.assemble(state,(),(),np.zeros(462))
        assert len(records) == 16 and len(seam.seen) == 16
        capture.records[0]['returned_residual'][:] = -1.
        assert records[0]['returned_residual'][0] == 1.
    assert seam.owner.scratch is state


def test_programming_failure_propagates_and_restores_aliases(seam,monkeypatch):
    def broken(*args,**kwargs): raise ValueError('original evaluator defect')
    monkeypatch.setattr(local.q4,'response',broken)
    monkeypatch.setattr(local.finite,'q4_algorithmic_response',broken)
    with pytest.raises(ValueError,match='original evaluator defect'):
        with local.Q4LocalCapture(seam.owner):
            local.finite.q4_algorithmic_response(seam.model,seam.elements[0],np.zeros(24),seam.origin,3)
    assert local.finite.q4_algorithmic_response is broken
    assert local.q4.deformation is seam.deformation
    assert local.q4.spatial_response is seam.spatial


def test_precancellation_performs_no_evaluation_and_restores(seam):
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled):
        with local.Q4LocalCapture(seam.owner) as capture:
            capture.assemble(object(),(),(),np.zeros(462),token)
    assert seam.seen == []
    assert not hasattr(seam.owner,'scratch')
    assert local.finite.q4_algorithmic_response is seam.response


def test_postevaluation_cancellation_preserves_original_scratch_lifecycle(seam):
    class AfterAssembly:
        def raise_if_cancelled(self,stage):
            if stage.startswith('after'): raise SolveCancelled('after original evaluator')
    state = object()
    with pytest.raises(SolveCancelled):
        with local.Q4LocalCapture(seam.owner) as capture:
            capture.assemble(state,(),(),np.zeros(462),AfterAssembly())
    assert len(seam.seen) == 16 and seam.owner.scratch is state
    assert local.finite.q4_algorithmic_response is seam.response


def test_nested_foreign_and_missing_seam_refusals_restore(seam,monkeypatch):
    with local.Q4LocalCapture(seam.owner):
        with pytest.raises(RuntimeError,match='nested or concurrent'):
            with local.Q4LocalCapture(seam.owner): pass
        with pytest.raises(RuntimeError,match='foreign or reentrant'):
            local.finite.q4_algorithmic_response(object(),seam.elements[0],np.zeros(24),seam.origin,3)
    assert local.finite.q4_algorithmic_response is seam.response
    monkeypatch.setattr(local.finite,'q4_algorithmic_response',lambda *args:None)
    with pytest.raises(RuntimeError,match='alias already replaced'):
        with local.Q4LocalCapture(seam.owner): pass
    assert local.q4.deformation is seam.deformation


def test_owned_alias_tampering_is_detected_after_exact_restoration(seam):
    with pytest.raises(RuntimeError,match='alias changed'):
        with local.Q4LocalCapture(seam.owner):
            local.q4.deformation = lambda *args:None
    assert local.finite.q4_algorithmic_response is seam.response
    assert local.q4.deformation is seam.deformation
    assert local.q4.spatial_response is seam.spatial


def test_wrong_case_refuses_before_alias_changes(seam):
    seam.owner._shell_parts.pop()
    with pytest.raises(local.fold.AssessmentRefusal,match='sixteen_Q4'):
        local.Q4LocalCapture(seam.owner)
    assert local.finite.q4_algorithmic_response is seam.response


def test_Q4_summation_bound_is_separate_from_local_operator_difference():
    h = local.fold.FD_STEPS[0]
    def record(eid,force,k=0.):
        return dict(element_id=eid,slots=np.array([0]),returned_residual=np.array([force]),
            returned_tangent=np.array([[k]]),local_coordinates=np.array([force]),
            deformation_differential=np.array([[1.]]),local_force=np.array([force]))
    # Exact analytic input: changing returned element forces while supplying
    # zero baseline derivative must remain a discrepancy, not a fold proof.
    baseline = [record(1,1e8),record(2,-1e8)]
    minus = [record(1,1e8-h),record(2,-1e8)]
    plus = [record(1,1e8+h),record(2,-1e8)]
    witness = local.q4_difference_witness(baseline,minus,plus,np.ones(1),h,
        np.ones(1),np.array([0]),1,np.ones(1))
    assert witness['Q4_only_original_expected_derivative'][0] == 0.
    assert witness['Q4_only_original_observed_derivative'][0] > .99
    assert witness['Q4_only_conditional_assembly_difference_bound'][0] > 0.
    assert witness['unproved_element_arithmetic_bound'] is True
    assert witness['operator_defect_proved'] is False
    assert witness['status'] == 'INCONCLUSIVE_LOCAL_Q4_DIAGNOSTIC'
    with pytest.raises(ValueError,match='fixed registered step'):
        local.q4_difference_witness(baseline,minus,plus,np.ones(1),1e-4,
            np.ones(1),np.array([0]),1,np.ones(1))
    # A pure-Q4 counterexample: combined-tangent and elementwise-product
    # arithmetic need not distribute identically in binary64. The remainder
    # cannot therefore be attributed to non-Q4 mechanics.
    pure = [record(1,0.,1e16),record(2,0.,1.),record(3,0.,-1e16)]
    reconstructed = local.q4_difference_witness(pure,pure,pure,np.array([.1]),h,
        np.ones(1),np.array([0]),1,np.ones(1))
    assembled_expected = ((1e16+1.)-1e16)*.1
    assembled_difference = -assembled_expected
    reconstruction_remainder = assembled_difference-reconstructed['Q4_only_original_difference'][0]
    assert reconstructed['Q4_only_original_expected_derivative'][0] == .125
    assert reconstruction_remainder == .125


@pytest.mark.parametrize('operation,a,b',(('add',.1,.2),('multiply',.1,.2),
    ('divide',.1,.3),('add',0.,.2),('multiply',0.,.2),('divide',0.,.3)))
def test_outward_positive_operations_cover_exact_Fraction_results(operation,a,b):
    fa,fb = Fraction.from_float(a),Fraction.from_float(b)
    exact = fa+fb if operation == 'add' else fa*fb if operation == 'multiply' else fa/fb
    bounded = local._positive_up(operation,a,b)
    assert Fraction.from_float(bounded) >= exact
    if exact == 0: assert bounded == 0.


def test_cancellation_and_central_envelope_cover_exact_returned_float_sums():
    h = local.fold.FD_STEPS[0]
    def record(eid,force):
        return dict(element_id=eid,slots=np.array([0]),returned_residual=np.array([force]),
            returned_tangent=np.zeros((1,1)),local_coordinates=np.zeros(1),
            deformation_differential=np.ones((1,1)),local_force=np.array([force]))
    # Large cancellation plus an unrepresentable final sum: exact Fraction
    # witnesses concern GIVEN returned floats, not their physical accuracy.
    baseline = [record(1,1e16),record(2,-1e16),record(3,.1)]
    minus = [record(1,1e16),record(2,-1e16),record(3,.1-h)]
    plus = [record(1,1e16),record(2,-1e16),record(3,.1+h)]
    witnessed = local.q4_difference_witness(baseline,minus,plus,np.ones(1),h,
        np.ones(1),np.array([0]),1,np.ones(1))
    exact_plus = sum(Fraction.from_float(float(row['returned_residual'][0])) for row in plus)
    exact_minus = sum(Fraction.from_float(float(row['returned_residual'][0])) for row in minus)
    exact_derivative = (exact_plus-exact_minus)/Fraction.from_float(2*h)
    computed = Fraction.from_float(float(witnessed['Q4_only_original_observed_derivative'][0]))
    bound = Fraction.from_float(float(witnessed['Q4_only_conditional_assembly_difference_bound'][0]))
    assert abs(computed-exact_derivative) <= bound
    exact_magnitude = sum(abs(Fraction.from_float(float(row['returned_residual'][0]))) for row in plus)
    assert Fraction.from_float(float(witnessed['Q4_plus_absolute_sum_upper'][0])) >= exact_magnitude
    assert witnessed['Q4_only_conditional_assembly_difference_bound'][0] > 0.


def test_subnormal_underflow_and_exact_zero_bound_domains():
    subnormal = np.nextafter(0.,1.)
    with pytest.raises(local.fold.AssessmentRefusal,match='subnormal_Q4_bound_domain'):
        local._sum_rows([dict(slots=np.array([0]),returned_residual=np.array([subnormal]))],1)
    with pytest.raises(local.fold.AssessmentRefusal,match='underflow_Q4_bound_domain'):
        local._positive_up('multiply',np.finfo(float).tiny,np.finfo(float).tiny)
    assert local._positive_up('multiply',0.,1.) == 0.
    assert local._positive_up('add',0.,0.) == 0.


def test_entry_refusal_preserves_complete_public_state_bytes_and_closes(monkeypatch):
    raw = b'bound fixture'; before = dict(generation=13); closed = []
    packet = dict(observation=dict(state=deepcopy(before)))
    owner = SimpleNamespace(restore=lambda value:SimpleNamespace(status='accepted_prefix'),
        recover=lambda:deepcopy(before),checkpoint=lambda:raw,close=lambda:closed.append(True),
        preflight=lambda:dict(identity=local.case.VERIFIED_FORCE_PREFIX_MODEL),
        _finite_driver=SimpleNamespace(_external=object(),_programme=None,_adaptive_state=None))
    monkeypatch.setattr(local.fold,'load_saved',lambda folder,generation:(raw,deepcopy(packet)))
    monkeypatch.setattr(local.ss,'create_analysis',lambda **kwargs:owner)
    reported = local.diagnose_saved_g13('fixture')
    assert reported['status'] == 'INCONCLUSIVE_LOCAL_Q4_DIAGNOSTIC'
    assert reported['reason'] == 'unsupported_local_g13_diagnostic_domain'
    assert reported['complete_public_recovery_unchanged'] is True
    assert reported['checkpoint_bytes_unchanged'] is True
    assert closed == [True]
    assert owner.recover() == before and owner.checkpoint() == raw


def test_entry_precancellation_never_constructs_model(monkeypatch):
    token = CancellationToken(); token.cancel()
    monkeypatch.setattr(local.fold,'load_saved',lambda folder,generation:(b'fixture',{}))
    def forbidden(**kwargs): raise AssertionError('model construction before cancellation')
    monkeypatch.setattr(local.ss,'create_analysis',forbidden)
    with pytest.raises(SolveCancelled):
        local.diagnose_saved_g13('fixture',cancellation_token=token)


def test_callback_mutation_is_detached_from_retained_capture_and_input():
    retained = dict(evaluations=[])
    evaluation = dict(label='baseline',status='COMPLETED_EVALUATION',diagnostic_completion='PARTIAL',
        Q4=[dict(local_force=np.array([2.,3.]))])
    def mutate(packet): packet['Q4'][0]['local_force'][0] = -99.
    local._publish_evaluation(retained,evaluation,mutate)
    assert retained['evaluations'][0]['Q4'][0]['local_force'] == [2.,3.]
    np.testing.assert_array_equal(evaluation['Q4'][0]['local_force'],(2.,3.))
    def broken(packet): raise OSError('durable writer failed')
    with pytest.raises(OSError,match='durable writer failed'):
        local._publish_evaluation(retained,evaluation,broken)
    assert len(retained['evaluations']) == 2


@pytest.mark.parametrize('exception_type',(OSError,local.fold.AssessmentRefusal))
@pytest.mark.parametrize('numerical_candidate',(local.BASELINE_NUMERICS,local.LOG_ONLY_CANDIDATE))
def test_callback_failure_after_actual_baseline_restores_seam_and_closes(log_seam,monkeypatch,exception_type,numerical_candidate):
    seam=log_seam.seam
    @dataclass
    class Recovery:
        generation: int
        amplitude: float
        applied_dead_load: np.ndarray
    recovery = Recovery(13,.1,np.zeros(462)); raw = b'fixture'
    packet = dict(observation=dict(state=local.case._plain(recovery),proportional_factor=.6))
    owner = seam.owner; driver = owner._finite_driver; closed = []; streamed = []
    owner.restore = lambda value:SimpleNamespace(status='accepted_prefix')
    owner.recover = lambda:deepcopy(recovery); owner.checkpoint = lambda:raw
    owner.close = lambda:closed.append(True)
    owner.preflight = lambda:dict(identity=local.case.VERIFIED_FORCE_PREFIX_MODEL,arc_metric_node_order=())
    owner._definition = SimpleNamespace(shell=SimpleNamespace(j2_curve=None),
        shell_reference_policy='flat',q4_pullback_policy=local.q4.POLICY)
    owner._footprint_increment_chart=False; owner._shell_count=162; owner._beam_ids=tuple(range(18))
    owner._K = local.fold.sparse.eye(462,format='csc')
    owner._coordinate_scale=np.ones(462); owner._constraint_scale=np.ones(90); owner._free=np.arange(462)
    driver._external=None; driver._programme=None; driver._adaptive_state=None
    driver._state=SimpleNamespace(q=np.zeros(462),frames=np.tile(np.eye(3),(18,1,1)),
        cells=np.tile(np.eye(3),(8,2,1,1)),resultants=np.zeros((8,18)),
        multipliers=np.zeros(90),shell_anchors=None)
    driver._increment_origin=lambda state:state; driver._beam_origins=()
    driver._load=lambda actions:np.zeros(462)
    monkeypatch.setattr(local.fold,'load_saved',lambda folder,generation:(raw,deepcopy(packet)))
    monkeypatch.setattr(local.ss,'create_analysis',lambda **kwargs:owner)
    monkeypatch.setattr(local.case,'material_activity',lambda value:dict(elastic_fold_interpretation_eligible=True))
    monkeypatch.setattr(local.case,'metric_for',lambda *args:object())
    monkeypatch.setattr(local.fold,'require_metric_owner',lambda *args:None)
    def failing_writer(observation):
        streamed.append(deepcopy(observation)); raise exception_type('writer failed after baseline')
    with pytest.raises(exception_type,match='writer failed after baseline'):
        local.diagnose_saved_g13('fixture',on_evaluation=failing_writer,numerical_candidate=numerical_candidate)
    assert len(seam.seen) == 16 and closed == [True]
    assert streamed[0]['label'] == 'baseline'
    assert streamed[0]['status'] == 'COMPLETED_EVALUATION'
    assert streamed[0]['diagnostic_completion'] == 'PARTIAL'
    assert streamed[0]['numerical_candidate'] == numerical_candidate
    assert local.finite.q4_algorithmic_response is seam.response
    assert local.q4.deformation is log_seam.kernel and local.q4.spatial_response is seam.spatial
    assert local.case._plain(owner.recover()) == packet['observation']['state']
    assert owner.checkpoint() == raw


def test_changed_predecessor_and_foreign_thread_refuse_without_evaluation(seam,monkeypatch):
    with local.Q4LocalCapture(seam.owner) as capture:
        with pytest.raises(RuntimeError,match='predecessor identity changed'):
            local.finite.q4_algorithmic_response(seam.model,seam.elements[0],np.zeros(24),object(),3)
        monkeypatch.setattr(local,'get_ident',lambda:capture.thread+1)
        with pytest.raises(RuntimeError,match='foreign or asynchronous'):
            local.finite.q4_algorithmic_response(seam.model,seam.elements[0],np.zeros(24),seam.origin,3)
    assert seam.seen == []
    assert local.finite.q4_algorithmic_response is seam.response


def test_candidate_reuses_exact_real_code_context_without_global_changes(seam,monkeypatch):
    original = local.shell_map.deformation
    monkeypatch.setattr(local.q4,'deformation',original)
    bindings = dict(original.__globals__)
    with local.Q4LocalCapture(seam.owner,numerical_candidate=local.LOG_ONLY_CANDIDATE) as capture:
        selected = capture.selected_deformation
        assert selected.__code__ is original.__code__
        assert selected.__closure__ is original.__closure__
        assert selected.__defaults__ is original.__defaults__
        assert selected.__kwdefaults__ == original.__kwdefaults__
        assert set(selected.__globals__) == set(bindings)
        changed = [key for key in bindings if selected.__globals__[key] is not bindings[key]]
        assert changed == ['so3_log']
        assert selected.__globals__['so3_exp'] is bindings['so3_exp']
        assert selected.__globals__['_quaternion_jets'] is bindings['_quaternion_jets']
        assert all(original.__globals__[key] is value for key,value in bindings.items())
        with pytest.raises(RuntimeError,match='outside actual owned Q4 deformation'):
            capture._candidate_log(None)
    assert local.q4.deformation is original
    assert local.finite.q4_algorithmic_response is seam.response
    assert local.q4.spatial_response is seam.spatial
    assert all(original.__globals__[key] is value for key,value in bindings.items())


@pytest.fixture
def log_seam(seam,monkeypatch):
    calls = dict(exp=0,original_log=0,candidate_log=0)
    original_exp = so3_exp; original_log = so3_log; candidate_log = local.stable_log.so3_log
    def exp_spy(vector):
        calls['exp'] += 1; return original_exp(vector)
    def original_log_spy(matrix):
        calls['original_log'] += 1; return original_log(matrix)
    def candidate_log_spy(matrix):
        calls['candidate_log'] += 1; return candidate_log(matrix)
    def kernel(reference,u,*,stable_translation=False,rotation_anchors=None,stable_rotation=False):
        values=np.zeros(24); differential=np.zeros((24,24)); second=np.zeros((24,24,24))
        for node in range(4):
            vector=[Jet2.variable(.011 if axis == 0 else 0.,6*node+3+axis,24) for axis in range(3)]
            # These global bindings are the selection discriminator. Only Log
            # may change in the owned shallow function context.
            logarithm=so3_log(so3_exp(vector))
            for axis,jet in enumerate(logarithm):
                slot=6*node+3+axis
                values[slot]=jet.value; differential[slot]=jet.gradient; second[slot]=jet.hessian
        return values,differential,second
    monkeypatch.setitem(kernel.__globals__,'so3_exp',exp_spy)
    monkeypatch.setitem(kernel.__globals__,'so3_log',original_log_spy)
    monkeypatch.setattr(local.shell_map,'so3_exp',exp_spy)
    monkeypatch.setattr(local.shell_map,'so3_log',original_log_spy)
    monkeypatch.setattr(local.shell_map,'deformation',kernel)
    monkeypatch.setattr(local.q4,'deformation',kernel)
    monkeypatch.setattr(local.stable_log,'so3_log',candidate_log_spy)
    return SimpleNamespace(seam=seam,calls=calls,kernel=kernel)


def test_only_actual_candidate_Q4_calls_select_stable_Log_and_keep_Exp_chart(log_seam):
    seam=log_seam.seam
    with local.Q4LocalCapture(seam.owner) as baseline:
        returned=local.finite.q4_algorithmic_response(seam.model,seam.elements[0],np.zeros(24),seam.origin,3)
        assert returned is seam.returns[-1]
        assert baseline.selected_deformation is log_seam.kernel
        assert log_seam.calls == dict(exp=4,original_log=4,candidate_log=0)
    bindings=dict(log_seam.kernel.__globals__)
    with local.Q4LocalCapture(seam.owner,numerical_candidate=local.LOG_ONLY_CANDIDATE) as capture:
        returned=local.finite.q4_algorithmic_response(seam.model,seam.elements[0],np.zeros(24),seam.origin,3)
        assert returned is seam.returns[-1]
        assert log_seam.calls == dict(exp=8,original_log=4,candidate_log=4)
        assert capture.records[0]['candidate_log_calls'] == 4
        assert capture.records[0]['stable_rotation'] is False
        assert capture.records[0]['original_exp_and_spatial_chart'] is True
        np.testing.assert_allclose(capture.records[0]['local_coordinates'][3::6],.011,rtol=0.,atol=1e-12)
        saved=deepcopy(capture.records)
        capture.records[0]['local_coordinates'][:]=-1.
        assert saved[0]['local_coordinates'][3] > 0.
        assert all(log_seam.kernel.__globals__[key] is value for key,value in bindings.items())
    assert local.q4.deformation is log_seam.kernel and local.q4.spatial_response is seam.spatial


@pytest.mark.parametrize('kind',('refusal','cancel','programming'))
def test_candidate_failures_propagate_and_restore_all_aliases(log_seam,monkeypatch,kind):
    seam=log_seam.seam
    failure = local.fold.AssessmentRefusal('candidate local domain') if kind == 'refusal' else (
        SolveCancelled('candidate cancelled') if kind == 'cancel' else ValueError('candidate evaluator defect'))
    def broken(matrix): raise failure
    monkeypatch.setattr(local.stable_log,'so3_log',broken)
    with pytest.raises(type(failure)) as caught:
        with local.Q4LocalCapture(seam.owner,numerical_candidate=local.LOG_ONLY_CANDIDATE):
            local.finite.q4_algorithmic_response(seam.model,seam.elements[0],np.zeros(24),seam.origin,3)
    assert caught.value is failure
    assert local.q4.deformation is log_seam.kernel
    assert local.finite.q4_algorithmic_response is seam.response
    assert local.q4.spatial_response is seam.spatial


def test_candidate_foreign_nested_and_stable_route_refuse(log_seam):
    seam=log_seam.seam
    with local.Q4LocalCapture(seam.owner,numerical_candidate=local.LOG_ONLY_CANDIDATE) as capture:
        with pytest.raises(RuntimeError,match='nested or concurrent'):
            with local.Q4LocalCapture(seam.owner,numerical_candidate=local.LOG_ONLY_CANDIDATE): pass
        with pytest.raises(RuntimeError,match='foreign or reentrant'):
            local.finite.q4_algorithmic_response(object(),seam.elements[0],np.zeros(24),seam.origin,3)
        capture.current=dict(candidate_log_calls=0)
        with pytest.raises(RuntimeError,match='original Exp and chart'):
            capture._deformation(np.zeros((4,3)),np.zeros(24),stable_rotation=True)
        with pytest.raises(RuntimeError,match='original spatial chart'):
            capture._spatial(np.zeros(24),np.zeros(24),np.eye(24),np.eye(24),np.zeros((24,24,24)),stable_rotation=True)
        capture.current=None
    assert local.q4.deformation is log_seam.kernel
    assert log_seam.calls == dict(exp=0,original_log=0,candidate_log=0)


def test_unknown_candidate_identity_refuses_before_model_or_alias_change(seam,monkeypatch):
    with pytest.raises(ValueError,match='known private Q4 numerical identity'):
        local.Q4LocalCapture(seam.owner,numerical_candidate='implicit fallback')
    def forbidden(*args,**kwargs): raise AssertionError('fixture/model must not be opened')
    monkeypatch.setattr(local.fold,'load_saved',forbidden)
    monkeypatch.setattr(local.ss,'create_analysis',forbidden)
    with pytest.raises(ValueError,match='known private Q4 numerical identity'):
        local.diagnose_saved_g13('fixture',numerical_candidate='implicit fallback')
    assert local.finite.q4_algorithmic_response is seam.response


@pytest.mark.parametrize('pair',('arbitrary-step-sweep',None,(3e-6,1e-6)))
def test_unknown_difference_pair_refuses_before_saved_input_or_owner(monkeypatch,pair):
    def forbidden(*args,**kwargs): raise AssertionError('fixture/model must not be opened')
    monkeypatch.setattr(local.fold,'load_saved',forbidden)
    monkeypatch.setattr(local.ss,'create_analysis',forbidden)
    with pytest.raises(ValueError,match='known private difference-pair'):
        local.diagnose_saved_g13('fixture',numerical_candidate=local.LOG_ONLY_CANDIDATE,
            difference_pair=pair)


def test_intermediate_pair_does_not_change_original_selection_or_allow_baseline(monkeypatch):
    assert local._fd_steps(local.REGISTERED_PAIR) == local.fold.FD_STEPS == (1e-5,1e-6)
    assert local._fd_steps(local.INTERMEDIATE_PAIR) == (3e-6,1e-6)
    def forbidden(*args,**kwargs): raise AssertionError('fixture/model must not be opened')
    monkeypatch.setattr(local.fold,'load_saved',forbidden)
    with pytest.raises(ValueError,match='requires explicit Log-only'):
        local.diagnose_saved_g13('fixture',difference_pair=local.INTERMEDIATE_PAIR)


def test_intermediate_witness_requires_explicit_pair_and_keeps_criterion():
    h=3e-6
    def record(force):
        return dict(element_id=1,slots=np.array([0]),returned_residual=np.array([force]),
            returned_tangent=np.array([[1.]]),local_coordinates=np.array([force]),
            deformation_differential=np.array([[1.]]),local_force=np.array([force]))
    args=([record(1.)],[record(1.-h)],[record(1.+h)],np.ones(1),h,
          np.ones(1),np.array([0]),1,np.ones(1))
    with pytest.raises(ValueError,match='fixed registered step'):
        local.q4_difference_witness(*args)
    witnessed=local.q4_difference_witness(*args,difference_pair=local.INTERMEDIATE_PAIR)
    assert witnessed['Q4_only_original_observed_derivative'][0] == pytest.approx(1.,abs=2e-11)
    assert witnessed['criterion'] == local.fold.ELASTIC_FD_LIMIT == 1e-6
    assert witnessed['operator_defect_excluded'] is False
    with pytest.raises(ValueError,match='fixed registered step'):
        local.q4_difference_witness(*args[:4],1e-5,*args[5:],difference_pair=local.INTERMEDIATE_PAIR)


@pytest.mark.parametrize('generation',(None,True,12,16,'14'))
def test_saved_endpoint_generation_refuses_before_loading_or_owner(monkeypatch,generation):
    def forbidden(*args,**kwargs): raise AssertionError('fixture/model must not be opened')
    monkeypatch.setattr(local.fold,'load_saved',forbidden)
    monkeypatch.setattr(local.ss,'create_analysis',forbidden)
    with pytest.raises(ValueError,match='exact frozen saved generation'):
        local.diagnose_saved_endpoint('fixture',generation=generation,
            numerical_candidate=local.LOG_ONLY_CANDIDATE,difference_pair=local.INTERMEDIATE_PAIR)


def test_historical_g13_facade_preserves_options_and_owned_result(monkeypatch):
    marker=object();token=object();observer=object();calls=[]
    def saved(folder,**options):
        calls.append((folder,options));return marker
    monkeypatch.setattr(local,'diagnose_saved_endpoint',saved)
    assert local.diagnose_saved_g13('fixture') is marker
    assert calls.pop() == ('fixture',dict(generation=13,cancellation_token=None,on_evaluation=None,
        numerical_candidate=local.BASELINE_NUMERICS,difference_pair=local.REGISTERED_PAIR))
    assert local.diagnose_saved_g13('fixture',cancellation_token=token,on_evaluation=observer,
        numerical_candidate=local.LOG_ONLY_CANDIDATE,difference_pair=local.INTERMEDIATE_PAIR) is marker
    assert calls.pop() == ('fixture',dict(generation=13,cancellation_token=token,on_evaluation=observer,
        numerical_candidate=local.LOG_ONLY_CANDIDATE,difference_pair=local.INTERMEDIATE_PAIR))


def _sampling_inputs():
    size = local.MIXED_ROWS
    tangent = np.zeros(size); tangent[0] = .3; tangent[3] = .7
    minus = np.zeros(local.SHELL_ROWS); plus = minus.copy()
    h = 1e-6
    minus[0] = -.25*h; plus[0] = .25*h
    minus[3] = -9*h; plus[3] = 9*h  # rotation sampling intentionally excluded
    difference = np.zeros(size); difference[0] = -.08; difference[3] = .04
    load = np.ones(size)
    return [minus,plus,tangent,h,np.eye(size)*2,np.ones(size),
            np.array([0,3,500]),462,load,difference]


def test_sampling_witness_identity_rows_and_preserved_failure():
    args = _sampling_inputs()
    result = local.shell_translation_sampling_witness(*args)
    delta = result['translation_direction_sampling_difference']
    assert delta[0] == pytest.approx(-.05)
    assert np.count_nonzero(delta) == 1 and delta[3] == 0
    np.testing.assert_allclose(result['first_order_baseline_J_contribution'],args[4]@delta)
    np.testing.assert_allclose(result['first_order_represented_translation_remainder'],
                               args[-1]-args[4]@delta)
    original = local.fold.directional_error_witness(args[-1],args[5],np.array([0,3]),462,args[8])
    assert result['original_witness'] == original
    assert result['diagnostic_only']
    for name in ('is_exact_error_bound','operator_defect_excluded',
                 'changes_registered_acceptance','original_failures_superseded'):
        assert result[name] is False


def test_sampling_witness_arrays_owned_and_mutation_isolated():
    args = _sampling_inputs()
    saved = [value.copy() if isinstance(value,np.ndarray) else value for value in args]
    result = local.shell_translation_sampling_witness(*args)
    for value in result.values():
        if isinstance(value,np.ndarray):
            assert value.flags.owndata
            for source in args:
                if isinstance(source,np.ndarray): assert not np.shares_memory(value,source)
    for original,copy in zip(args,saved):
        if isinstance(original,np.ndarray): np.testing.assert_array_equal(original,copy)
    remembered = result['first_order_baseline_J_contribution'].copy()
    for source in args:
        if isinstance(source,np.ndarray): source.fill(99)
    np.testing.assert_array_equal(result['first_order_baseline_J_contribution'],remembered)


def test_sampling_zero_and_intermediate_pair_sparse_operator():
    from scipy.sparse import eye
    args = _sampling_inputs()
    args[0].fill(0); args[1].fill(0); args[2].fill(0)
    args[3] = 3e-6; args[4] = eye(local.MIXED_ROWS,format='csr')
    result = local.shell_translation_sampling_witness(*args,difference_pair=local.INTERMEDIATE_PAIR)
    assert not result['translation_direction_sampling_difference'].any()
    assert result['first_order_remainder_witness'] == result['original_witness']


@pytest.mark.parametrize('position,bad',[
    (0,np.zeros(local.SHELL_ROWS-1)),(1,np.full(local.SHELL_ROWS,np.nan)),
    (2,np.zeros(local.MIXED_ROWS-1)),(3,2e-6),(5,np.full(local.MIXED_ROWS,np.inf)),
    (6,np.array([-1])),(6,np.array([0.,3.])),(6,np.array([0,0])),
    (6,np.array([local.MIXED_ROWS])),(7,True),
    (8,np.zeros(local.MIXED_ROWS-1)),(9,np.full(local.MIXED_ROWS,np.nan))])
def test_sampling_refuses_bad_inputs_before_operator_copy(position,bad):
    class MustNotCopy:
        shape = (local.MIXED_ROWS,local.MIXED_ROWS)
        def copy(self): raise AssertionError('invalid sampling inputs reached operator')
    args = _sampling_inputs(); args[4] = MustNotCopy(); args[position] = bad
    with pytest.raises(ValueError): local.shell_translation_sampling_witness(*args)


@pytest.mark.parametrize('operator',[np.eye(2),np.full((local.MIXED_ROWS,local.MIXED_ROWS),np.inf)])
def test_sampling_refuses_wrong_or_nonfinite_baseline(operator):
    args = _sampling_inputs(); args[4] = operator
    with pytest.raises(ValueError): local.shell_translation_sampling_witness(*args)


def test_sampling_zero_load_reference_refused():
    args = _sampling_inputs(); args[8].fill(0)
    with pytest.raises(local.fold.AssessmentRefusal):
        local.shell_translation_sampling_witness(*args)