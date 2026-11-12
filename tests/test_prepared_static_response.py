import numpy as np
import pytest
from anyfesolver import (PatternCase, prepare_static_response_evaluator, prepare_static_evaluator,
    RecoveryConfig, ResourceConfig, StaticCase, LoadCase, PreparedStaticError, CancellationToken, SolveCancelled)
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case


def test_mixed_baseline_and_derived_response_owned():
    model = _bar()
    baseline = _case(0,2).load_case
    pattern = _case(1,3).load_case
    recovery = RecoveryConfig(include_stresses=True,include_reactions=True)
    with prepare_static_response_evaluator(model,{'p':pattern},baseline_load_case=baseline,recovery_config=recovery) as response:
        pattern.nodal_loads.clear()
        baseline.nodal_loads.clear()
        cases = [PatternCase('a',{'p':-2.}),_case(2,4),PatternCase('b',{})]
        actual = next(response.iter_batches(cases))
        with prepare_static_evaluator(model,recovery_config=recovery) as direct:
            expected = next(direct.iter_batches([_case(0,-4),_case(2,4),_case(3,2)]))
        np.testing.assert_allclose(actual.displacements,expected.displacements,rtol=1e-12,atol=1e-14)
        for a,b in zip(actual.case_results,expected.case_results):
            for eid,record in a['element_stresses'].items():
                for name,value in record.items():
                    if isinstance(value,(np.ndarray,int,float)):
                        np.testing.assert_allclose(value,b['element_stresses'][eid][name],rtol=1e-11,atol=1e-10)
            for node,value in a['reactions'].items():
                np.testing.assert_allclose(value,b['reactions'][node],rtol=1e-12,atol=1e-12)
        assert response.diagnostics()['source_response']['reconstructed_element_records'] == 0
        assert response.diagnostics()['source_response']['canonical_stress_cases'] == 2


def test_freefree_unbalanced_columns_balanced_combination():
    model = _bar(False)
    a = LoadCase('a'); a.add_nodal_load(1,[-3.,0,0,0,0,0])
    b = LoadCase('b'); b.add_nodal_load(2,[3.,0,0,0,0,0])
    with prepare_static_response_evaluator(model,{'a':a,'b':b}) as evaluator:
        batch = next(evaluator.iter_batches([PatternCase('balanced',{'a':1.,'b':1.})]))
        assert batch.case_results[0]['solver_info']['relative_backward_error'] <= 1e-10
        with pytest.raises(PreparedStaticError,match='rigid-body'):
            next(evaluator.iter_batches([PatternCase('bad',{'a':1.})]))


def test_original_packet_forces_certification_fallback_and_envelope_identity():
    model = _bar()
    recovery = RecoveryConfig(include_stresses=True,include_reactions=True)
    p = _case(1,1).load_case
    cases = [PatternCase('same',{'p':1.},2.,canonical_load_case=_case(3,7).load_case)]
    with prepare_static_response_evaluator(model,{'p':p},recovery_config=recovery) as evaluator:
        result = next(evaluator.iter_batches(cases))
        assert evaluator.diagnostics()['source_response']['certification_fallback_cases'] == 1
    with prepare_static_response_evaluator(model,{'p':p},recovery_config=recovery,envelope_recovery_config=recovery) as evaluator:
        envelope = next(evaluator.iter_batches(cases))
        assert evaluator.envelope_result() is not None
        assert evaluator.diagnostics()['source_response']['canonical_envelope_cases'] == 1
        np.testing.assert_array_equal(result.displacements,envelope.displacements)


def test_shell_components_and_nonfinite_refusal():
    model = _shell_model()
    p = _shell_case(1,pressure=3).load_case
    recovery = RecoveryConfig(include_stresses=True,include_reactions=True)
    with prepare_static_response_evaluator(model,{'p':p},recovery_config=recovery) as evaluator:
        result = next(evaluator.iter_batches([PatternCase('p',{'p':1.})]))
        assert result.case_results[0]['element_stresses']
        with pytest.raises(PreparedStaticError):
            next(evaluator.iter_batches([PatternCase('bad',{'p':float('nan')})]))
        with pytest.raises(PreparedStaticError):
            next(evaluator.iter_batches([PatternCase('bad',{'missing':1.})]))


def test_memory_batches_and_callback_cancel():
    model = _bar()
    with prepare_static_response_evaluator(model,{'p':_case(1).load_case},
            resource_config=ResourceConfig(memory_limit_bytes=10_000_000),batch_size=3) as evaluator:
        batches = list(evaluator.iter_batches([PatternCase(str(i),{'p':float(i)}) for i in range(5)]))
        assert [len(b.case_ids) for b in batches] == [3,2]
        token = CancellationToken(); token.cancel()
        with pytest.raises(SolveCancelled):
            next(evaluator.iter_batches([PatternCase('x',{})],cancellation_token=token))

def test_dense_coefficients_and_execution_authority(monkeypatch):
    from anyfesolver.prepared_static_response import PreparedStaticResponseEvaluator
    model = _bar()
    with prepare_static_response_evaluator(model,{'p':_case(1).load_case}) as evaluator:
        batches = list(evaluator.iter_batches([PatternCase('a',[2.]),PatternCase('b',np.array([2.]))]))
        np.testing.assert_array_equal(batches[0].displacements[:,0],batches[0].displacements[:,1])
        original = PreparedStaticResponseEvaluator._combine
        monkeypatch.setattr(PreparedStaticResponseEvaluator,'_combine',staticmethod(lambda *a: original(*a)))
        with pytest.raises(PreparedStaticError,match='authority'):
            next(evaluator.iter_batches([PatternCase('bad',[1.])]))


def test_callback_failure_does_not_publish_envelope():
    recovery = RecoveryConfig(include_stresses=True,include_reactions=True)
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},envelope_recovery_config=recovery) as evaluator:
        before = evaluator.diagnostics()['blocks_yielded']
        def fail(*args,**kwargs):
            raise RuntimeError('callback failure')
        with pytest.raises(RuntimeError,match='callback failure'):
            next(evaluator.iter_batches([PatternCase('a',{'p':1.})],progress_callback=fail))
        assert evaluator.diagnostics()['blocks_yielded'] == before


def test_nonzero_prescribed_offset_applied_once():
    from anyfesolver import BoundaryCondition
    model = _bar()
    # Use the native boundary API for a nonzero support offset.
    model = _bar(False)
    model.add_boundary_condition(BoundaryCondition('left',[1],{'ux':0.001,'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
    model.add_boundary_condition(BoundaryCondition('right',[2],{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
    with prepare_static_response_evaluator(model,{'p':_case(1,4).load_case},baseline_load_case=_case(2,2).load_case) as response:
        got = next(response.iter_batches([PatternCase('x',{'p':2.})]))
    with prepare_static_evaluator(model) as direct:
        expected = next(direct.iter_batches([_case(3,10)]))
    np.testing.assert_allclose(got.displacements,expected.displacements,rtol=1e-12,atol=1e-14)

def test_large_original_packet_reduces_memory_block_before_copy():
    model = _bar()
    large = _case(2,2)
    large.load_case.name = 'n'*50000
    # A packet can fit alone while the requested 32-case reserve cannot.
    with prepare_static_response_evaluator(model,{'p':_case(1).load_case},batch_size=32,
            resource_config=ResourceConfig(memory_limit_bytes=350000)) as evaluator:
        blocks = list(evaluator.iter_batches([large,PatternCase('p',{'p':1.})]))
        assert sum(len(b.case_ids) for b in blocks) == 2
        assert all(b.diagnostics['estimated_block_bytes'] <= 350000 for b in blocks)


def test_tiny_actual_source_discrepancy_still_replays():
    with prepare_static_response_evaluator(_bar(),{'p':_case(1,1).load_case}) as evaluator:
        next(evaluator.iter_batches([PatternCase('x',{'p':1.},canonical_load_case=_case(2,1+1e-14).load_case)]))
        assert evaluator.diagnostics()['source_response']['reconstructed_cases'] == 0
        assert evaluator.diagnostics()['source_response']['certification_fallback_cases'] == 1

def test_library_metadata_is_retained_and_budgeted():
    from anyfesolver.prepared_static_response import _library_storage_bytes
    with prepare_static_response_evaluator(_bar(),{'long_pattern_name'*100:_case(1).load_case}) as evaluator:
        info = evaluator.diagnostics()['source_response']
        numeric = evaluator._force_columns.nbytes + evaluator._solution_columns.nbytes
        actual = _library_storage_bytes((evaluator._pattern_names,evaluator._pattern_lookup,
            evaluator._force_columns,evaluator._solution_columns))
        assert info['library_numeric_bytes'] == numeric
        assert info['library_bytes'] == actual
        assert info['library_metadata_bytes'] == actual-numeric > 0
        # Arrays share a bytes-backed base chain; aliasing must not duplicate it.
        assert _library_storage_bytes((evaluator._force_columns,evaluator._force_columns)) < 2*_library_storage_bytes(evaluator._force_columns)
    assert evaluator.diagnostics()['source_response']['library_bytes'] == 0


@pytest.mark.parametrize('stage', ['before_library','ready'])
def test_preparation_token_never_runs_in_owned_scope_and_rechecks_authority(monkeypatch, stage):
    import anyfesolver.prepared_static_response as response_module
    import anyfesolver.prepared_static_recovery as recovery_module
    observed = []
    class HostileToken:
        def raise_if_cancelled(self, where):
            if where == 'prepared_response.'+stage:
                observed.append(recovery_module._OWNED_RECOVERY_SCOPE.get())
                monkeypatch.setattr(response_module,'evaluate_load_vector',lambda *args: pytest.fail('mutated provider executed'))
    with pytest.raises(PreparedStaticError,match='authority'):
        prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},
            recovery_config=RecoveryConfig(include_stresses=True),cancellation_token=HostileToken())
    assert observed == [None]

def test_pattern_label_memory_is_checked_before_ingestion():
    from anyfesolver.recovery import ResourcePolicyError
    with prepare_static_response_evaluator(_bar(),{'p':_case(1).load_case},
            resource_config=ResourceConfig(memory_limit_bytes=200000)) as owner:
        with pytest.raises(ResourcePolicyError,match='before copy'):
            next(owner.iter_batches([PatternCase('x'*1000000,{'p':1.})]))
        assert owner.diagnostics()['cases_certified'] == 0
        class ForeignLabel:
            def __sizeof__(self):
                pytest.fail('foreign label callback executed')
        with pytest.raises(PreparedStaticError,match='plain string'):
            next(owner.iter_batches([PatternCase(ForeignLabel(),{'p':1.})]))

@pytest.mark.parametrize('kind',['base','response'])
@pytest.mark.parametrize('hook',['_load_vector','_require_open'])
def test_evaluator_subclasses_refused_before_trusted_preparation(kind,hook):
    from anyfesolver import PreparedStaticEvaluator, PreparedStaticResponseEvaluator
    from anyfesolver.prepared_static_response import _require_response_authority
    parent = PreparedStaticEvaluator if kind == 'base' else PreparedStaticResponseEvaluator
    def foreign(*args,**kwargs):
        pytest.fail('foreign evaluator hook executed')
    Foreign = type('ForeignEvaluator',(parent,),{hook:foreign})
    with pytest.raises(PreparedStaticError,match='subclasses'):
        Foreign(None,**({'patterns':{}} if kind == 'response' else {}))
    with pytest.raises(PreparedStaticError,match='subclasses'):
        _require_response_authority(object.__new__(Foreign))

@pytest.mark.parametrize('where',['class','instance','global'])
@pytest.mark.parametrize('when',['before_iteration','during_input'])
def test_native_base_hook_authority_before_owned_load_span(monkeypatch,where,when):
    from anyfesolver import PreparedStaticEvaluator
    import anyfesolver.prepared_static as base_module
    owner = prepare_static_evaluator(_bar())
    called = []
    def foreign(*args,**kwargs):
        called.append(True)
        pytest.fail('foreign base hook executed')
    def mutate():
        if where == 'class':
            monkeypatch.setattr(PreparedStaticEvaluator,'_load_vector',foreign)
        elif where == 'instance':
            monkeypatch.setattr(owner,'_load_vector',foreign)
            # Attempt to bypass the overridable open check too.
            monkeypatch.setattr(owner,'_require_open',lambda: owner._snapshot)
        else:
            monkeypatch.setattr(base_module,'evaluate_load_vector',foreign)
    def cases():
        if when == 'during_input':
            mutate()
        yield _case(1)
    try:
        if when == 'before_iteration':
            mutate()
        with pytest.raises(PreparedStaticError,match='authority'):
            next(owner.iter_batches(cases()))
        assert not called
    finally:
        owner.close()
