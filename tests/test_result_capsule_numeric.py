"""Complete numeric-publication comparison against original detailed records."""
from collections.abc import Mapping
import struct
import numpy as np
import pytest

from anyfesolver import (PatternCase, RecoveryConfig, FixedSupport, BoundaryCondition,
    prepare_static_response_evaluator, export_recovery_capsule, export_recovery_capsule_to_sink)
from anyfesolver.prepared_static_numeric import NumericCase, float_bits_equal
from test_prepared_static_evaluator import _bar, _case, _shell_case
from test_result_capsule_export import _rotated_shell_model
from test_generalized_beam_sections import _model, _stiffness
from anyfesolver.beam_sections import GeneralizedBeamSection
from anyfesolver.elements import QuadraticBeamElement


@pytest.mark.parametrize('indices',[[],[1,3],[3,1],[1,2,3]])
def test_source_vector_publication_calls_original_complete_oracle_once_per_element(indices):
    from anyfesolver.prepared_static_numeric import source_vectors
    from anyfesolver.prepared_static_response import PreparedStaticResponseEvaluator,_PatternPacket
    random=np.random.default_rng(916)
    matrices=[np.asfortranarray(random.uniform(-1e100,1e100,(7,4))) for _ in range(3)]
    packet=_PatternPacket(np.asarray(indices,dtype=np.intp),np.array([.125,-3.,2.][:len(indices)]),None)
    calls=[]
    def original(matrix,source_packet):
        calls.append((matrix,source_packet))
        return PreparedStaticResponseEvaluator._combine(matrix,source_packet)
    values,fallback,oracle_seconds,trial_seconds=source_vectors([(None,None,None,None,matrix) for matrix in matrices],packet,original)
    expected=np.stack([PreparedStaticResponseEvaluator._combine(matrix,packet) for matrix in matrices])
    assert len(calls)==3 and all(left[0] is right for left,right in zip(calls,matrices))
    assert values.tobytes()==expected.tobytes()
    assert fallback==0 and trial_seconds==0. and oracle_seconds>=0.


def assert_identical(actual, expected):
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, Mapping):
        assert set(actual) == set(expected)
        for key in expected: assert_identical(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert type(actual) == type(expected) and len(actual) == len(expected)
        for left,right in zip(actual,expected): assert_identical(left,right)
    elif type(expected) is float:
        assert struct.pack('d',actual) == struct.pack('d',expected)
    else:
        assert actual == expected


@pytest.mark.parametrize('fixture',['beam2','beam3','rotated-shell','principal-shell'])
def test_numeric_complete_capsule_bytewise_matches_records(fixture):
    if fixture in ('rotated-shell','principal-shell'):
        model = _rotated_shell_model()
        patterns = {f'p{i}':_shell_case(i+1,pressure=(i+1)*.7).load_case for i in range(4)}
        baseline = _shell_case(0,pressure=.5).load_case
    elif fixture == 'beam3':
        model,_ = _model(QuadraticBeamElement,section=GeneralizedBeamSection(_stiffness()))
        model.add_boundary_condition(FixedSupport('base',[1]))
        model.add_boundary_condition(BoundaryCondition('axial',[2,3],{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
        patterns = {f'p{i}':_case(i+1,(i+1)*.7).load_case for i in range(4)}
        baseline = _case(0,.5).load_case
    else:
        model = _bar()
        patterns = {f'p{i}':_case(i+1,(i+1)*.7).load_case for i in range(4)}
        baseline = _case(0,.5).load_case
    components = ['principal_stresses_top','principal_stresses_bottom'] if fixture == 'principal-shell' else None
    config = RecoveryConfig(include_stresses=True,include_reactions=True,components=components)
    cases = [PatternCase('dense',{'p0':1.7,'p1':-2.3,'p2':.31,'p3':5.1}),PatternCase('reversed',{'p3':2.2,'p0':-.9}),PatternCase('zero',{}),
             PatternCase('large',{'p3':1.e70,'p0':-3.e69}),PatternCase('small',{'p2':-1.e-70})]
    with prepare_static_response_evaluator(model,patterns,baseline_load_case=baseline,recovery_config=config,stress_execution='certified_library') as evaluator:
        expected = export_recovery_capsule(evaluator,cases,model_identity='model',mesh_identity='mesh',batch_size=2,recovery_execution='records')
        actual = export_recovery_capsule(evaluator,cases,model_identity='model',mesh_identity='mesh',batch_size=2,recovery_execution='auto')
        assert_identical(actual,expected)
        detailed = next(evaluator.iter_batches(cases,batch_size=1))
        assert isinstance(detailed.case_results[0],dict) and detailed.case_results[0]['element_stresses']
        numeric = next(evaluator._iter_capsule_batches(cases,batch_size=1))
        assert isinstance(numeric.case_results[0],NumericCase)
        assert 'element_stresses' not in numeric.case_results[0]
        assert numeric.diagnostics['shared_recovery']['numeric_element_cases'] == len(model.mesh.elements)


def test_numeric_canonical_source_disagreement_keeps_actual_fallback():
    cases=[PatternCase('fallback',{'p':1.},canonical_load_case=_case(2,7.).load_case),PatternCase('portable',{'p':2.})]
    with prepare_static_response_evaluator(_bar(),{'p':_case(1,1.).load_case},recovery_config=RecoveryConfig(include_stresses=True,include_reactions=True),stress_execution='certified_library') as evaluator:
        expected=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x',batch_size=1,recovery_execution='records')
        actual=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x',batch_size=1,recovery_execution='auto')
        assert_identical(actual,expected)
        assert actual['cases'][0]['route'].endswith('canonical_fallback')
        assert not actual['cases'][0]['portable']


def test_numeric_float_admission_includes_signed_zero():
    assert float_bits_equal(np.array([0.]),np.array([0.]))
    assert not float_bits_equal(np.array([0.]),np.array([-0.]))


def test_numeric_source_authority_and_interrupted_sink(monkeypatch):
    from anyfesolver import PreparedStaticCapabilityError
    from anyfesolver import prepared_static_numeric as numeric
    cases=[PatternCase('a',{'p':1.})]
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution='certified_library') as evaluator:
        with monkeypatch.context() as patch:
            patch.setattr(numeric,'source_fields',lambda *args:{} )
            with pytest.raises(PreparedStaticCapabilityError,match='authority'):
                export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x')
        class Sink:
            failures=[]
            def capsule_begin(self,header): pass
            def capsule_case(self,record,coefficients):
                assert not coefficients.flags.writeable
                raise RuntimeError('owned sink interruption')
            def capsule_stored(self,*args): pytest.fail('interrupted output')
            def capsule_end(self,*args): pytest.fail('interrupted output')
            def capsule_failed(self,payload): self.failures.append(payload)
        sink=Sink()
        with pytest.raises(RuntimeError,match='owned sink interruption'):
            export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity='m',mesh_identity='x')
        assert len(sink.failures)==1


def test_numeric_cancellation_and_live_model_mutation_keep_owned_state():
    from anyfesolver import CancellationToken, SolveCancelled
    model = _bar()
    cases = [PatternCase('a',{'p':1.}),PatternCase('b',{'p':2.})]
    with prepare_static_response_evaluator(model,{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution='certified_library') as evaluator:
        expected=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x',recovery_execution='records')
        model.mesh.nodes[2].x = 99.
        actual=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x')
        assert_identical(actual,expected)
        token = CancellationToken()
        iterator = evaluator._iter_capsule_batches(cases,batch_size=1,cancellation_token=token)
        first = next(iterator)
        assert isinstance(first.case_results[0],NumericCase)
        token.cancel()
        with pytest.raises(SolveCancelled): next(iterator)


def test_numeric_extra_workspace_refuses_under_source_resource_policy():
    from anyfesolver import ResourceConfig
    from anyfesolver.recovery import ResourcePolicyError
    cases=[PatternCase('a',{'p':1.})]
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),stress_execution='certified_library') as evaluator:
        estimate=next(evaluator.iter_batches(cases,batch_size=1)).diagnostics['estimated_block_bytes']
        prepared_required=evaluator.diagnostics()['source_response']['preparation_memory_peak_required_bytes']
    # Preparation now accounts for its bounded pending recipe tile too. Preserve
    # that earlier refusal, then admit the actual prepared/batch requirement;
    # extra numeric publication must still refuse under that measured limit.
    with pytest.raises(ResourcePolicyError) as preparation_refusal:
        prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},
            recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),
            resource_config=ResourceConfig(memory_limit_bytes=1),
            stress_execution='certified_library')
    assert preparation_refusal.value.memory_estimate.total_bytes_estimate>1
    limit=max(int(estimate),int(prepared_required))
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True,include_reactions=False),resource_config=ResourceConfig(memory_limit_bytes=limit),stress_execution='certified_library') as evaluator:
        assert evaluator.diagnostics()['source_response']['preparation_memory_peak_required_bytes']<=limit
        export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x',batch_size=1,recovery_execution='records',replay_version=1)
        with pytest.raises(ResourcePolicyError) as numeric_refusal:
            export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x',batch_size=1)
        assert numeric_refusal.value.memory_estimate.total_bytes_estimate>limit
        assert numeric_refusal.value.context=='prepared_response'
        assert evaluator._numeric_export_plan is None


@pytest.mark.parametrize('coefficient',[float('nan'),float('inf')])
def test_numeric_nonfinite_input_refuses_without_success(coefficient):
    from anyfesolver import PreparedStaticError
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution='certified_library') as evaluator:
        with pytest.raises(PreparedStaticError,match='finite'):
            export_recovery_capsule(evaluator,[PatternCase('bad',{'p':coefficient})],model_identity='m',mesh_identity='x')


def test_numeric_tile_caller_observation_outside_owned_scope(monkeypatch):
    from anyfesolver.prepared_static_recovery import _OWNED_RECOVERY_SCOPE
    from anyfesolver import PreparedStaticCapabilityError, prepared_static_numeric as numeric
    class Token:
        observed = 0
        def raise_if_cancelled(self,stage):
            assert _OWNED_RECOVERY_SCOPE.get() is None
            if stage == 'prepared_response.numeric_tile':
                self.observed += 1
                monkeypatch.setattr(numeric,'source_fields',lambda *args:{})
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution='certified_library') as evaluator:
        token=Token()
        with pytest.raises(PreparedStaticCapabilityError,match='authority'):
            next(evaluator._iter_capsule_batches([PatternCase('a',{'p':1.})],cancellation_token=token))
        assert token.observed == 1


def test_numeric_export_with_envelope_preserves_record_route():
    config=RecoveryConfig(include_stresses=True,include_reactions=True)
    cases=[PatternCase('a',{'p':1.}),PatternCase('b',{'p':-2.})]
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},recovery_config=config,envelope_recovery_config=config,stress_execution='certified_library') as evaluator:
        expected=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x',recovery_execution='records')
        actual=export_recovery_capsule(evaluator,cases,model_identity='m',mesh_identity='x')
        assert_identical(actual,expected)
        assert evaluator._numeric_export_plan is None
