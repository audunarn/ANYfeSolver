"""M17 vectorized certified route: behaviour, envelope structure and development fixtures."""
import json
import numpy as np
import pytest
from anyfesolver import (PatternCase, prepare_static_response_evaluator, RecoveryConfig,
    PreparedStaticError, PreparedStaticCapabilityError)
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case
import prepared_static_response_m16_assessment as assessment

VECTOR = dict(stress_execution='certified_vectorized')
STRESS = RecoveryConfig(include_stresses=True, include_reactions=True)
ENVELOPE = RecoveryConfig(include_displacements=True, include_stresses=True, include_reactions=True)


def _shell_run(execution, cases, **kwargs):
    with prepare_static_response_evaluator(_shell_model(), {'p': _shell_case(0, pressure=2.).load_case,
            'g': _shell_case(1, gravity=True).load_case}, recovery_config=STRESS,
            envelope_recovery_config=ENVELOPE, stress_execution=execution) as owner:
        blocks = list(owner.iter_batches(cases, **kwargs))
        return blocks, owner.envelope_result(), owner.diagnostics()['source_response']


def test_vectorized_route_counts_and_labels():
    cases = [PatternCase('a', {'p': .3}), PatternCase('b', {'p': -1., 'g': 2.}), _shell_case(5, pressure=1.)]
    blocks, summary, info = _shell_run('certified_vectorized', cases, batch_size=3)
    batch = blocks[0]
    assert info['contract'] == 'certified_vectorized_route_m17'
    assert info['vectorized_stress_cases'] == 3 and info['vectorized_elements'] == 2
    assert info['vectorized_worst_verification_residual'] <= 1e-9
    assert batch.diagnostics['shared_recovery']['physical_element_evaluations'] == 0
    states = [r['solver_info']['source_response']['state'] for r in batch.case_results]
    assert states == ['reconstructed', 'reconstructed', 'canonical_solve']
    assert summary.complete and summary.diagnostics['recovery'] == 'certified_vectorized_compiled_affine'
    for result in batch.case_results:
        assert set(result['element_stresses']) == {1, 2} and result['reactions']


def test_envelope_structure_matches_canonical_route():
    cases = [PatternCase(str(i), {'p': c, 'g': d}) for i, (c, d) in enumerate([(1., 0.), (-2., .5), (.5, -1.), (3., 1.)])]
    _, vector, _ = _shell_run('certified_vectorized', cases, batch_size=2)
    _, canonical, _ = _shell_run('canonical', cases, batch_size=2)
    assert len(vector.extrema) == len(canonical.extrema) and vector.cases_processed == canonical.cases_processed == 4
    for got, wanted in zip(vector.extrema, canonical.extrema):
        assert (got.quantity, got.entity_id, got.component, got.original_shape) == \
               (wanted.quantity, wanted.entity_id, wanted.component, wanted.original_shape)
        for mode in ('minimum', 'maximum', 'absolute'):
            g, w = getattr(got, mode), getattr(wanted, mode)
            assert g.values.shape == w.values.shape and g.input_indices.shape == w.input_indices.shape
            assert set(g.associated_fields) == set(w.associated_fields)
            for key in w.associated_fields:
                assert g.associated_fields[key].shape == w.associated_fields[key].shape
            np.testing.assert_allclose(g.values, w.values, rtol=1e-9, atol=1e-6)


def test_exact_ties_keep_earliest_input_across_blocks():
    cases = [PatternCase('first', {'p': 1.}), PatternCase('second', {'p': 1.}), PatternCase('third', {'p': 1.})]
    _, summary, _ = _shell_run('certified_vectorized', cases, batch_size=2)
    for entry in summary.extrema:
        for mode in ('minimum', 'maximum', 'absolute'):
            assert np.all(getattr(entry, mode).input_indices == 0)
            assert np.all(getattr(entry, mode).case_ids == 'first')


def test_callback_failure_does_not_publish_vectorized_envelope():
    with prepare_static_response_evaluator(_bar(), {'p': _case(1).load_case}, envelope_recovery_config=STRESS, **VECTOR) as owner:
        def fail(*args, **kwargs):
            raise RuntimeError('callback failure')
        stream = owner.iter_batches([PatternCase('a', {'p': 1.}), PatternCase('b', {'p': 9.})], batch_size=1)
        next(stream)
        stream.close()
        before = owner.envelope_result()
        with pytest.raises(RuntimeError, match='callback failure'):
            next(owner.iter_batches([PatternCase('c', {'p': 50.})], progress_callback=fail))
        after = owner.envelope_result()
        assert before.cases_processed == 1 and not before.complete
        assert after.cases_processed == 0 and not after.complete


def test_vectorized_token_never_runs_in_owned_scope(monkeypatch):
    import anyfesolver.prepared_static_response as response_module
    import anyfesolver.prepared_static_recovery as recovery_module
    observed = []
    class HostileToken:
        def raise_if_cancelled(self, where):
            if where == 'prepared_response.vector_recovery':
                observed.append(recovery_module._OWNED_RECOVERY_SCOPE.get())
                monkeypatch.setattr(response_module, 'evaluate_load_vector', lambda *args: pytest.fail('mutated provider executed'))
    with pytest.raises(PreparedStaticError, match='authority'):
        prepare_static_response_evaluator(_bar(), {'p': _case(1).load_case}, recovery_config=STRESS,
                                          cancellation_token=HostileToken(), **VECTOR)
    assert observed == [None]


def test_nonaffine_vectorized_probe_refused(monkeypatch):
    import anyfesolver.prepared_static_vectorized as vector_module
    monkeypatch.setattr(vector_module, '_recover_owned_local',
                        lambda recipe, local: {'axial_stress': np.full(2, float(np.sum(local)) ** 2), 'label': 'beam'})
    with pytest.raises(PreparedStaticCapabilityError, match='not affine'):
        vector_module._probe_element(3, None, 12, set())


def test_development_fixtures_under_frozen_criteria_vectorized():
    for summary in assessment.development_fixtures(stress_execution='certified_vectorized'):
        print('M17 development summary:', json.dumps({k: v for k, v in summary.items() if k != 'failures'}, default=float))
        assert summary['failure_count'] == 0, summary
