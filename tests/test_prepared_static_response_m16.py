"""M16 certified response route: behaviour (criterion C6) and development fixtures."""
import numpy as np
import pytest
from anyfesolver import (PatternCase, prepare_static_response_evaluator, RecoveryConfig,
    PreparedStaticError, PreparedStaticCapabilityError, LoadCase)
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case
import prepared_static_response_m16_assessment as assessment

LIBRARY = dict(stress_execution='certified_library')
STRESS = RecoveryConfig(include_stresses=True, include_reactions=True)


def test_default_route_is_unchanged_canonical():
    with prepare_static_response_evaluator(_bar(), {'p': _case(1, 3).load_case}, recovery_config=STRESS) as owner:
        batch = next(owner.iter_batches([PatternCase('a', {'p': 2.})]))
        info = owner.diagnostics()['source_response']
        assert info['stress_execution'] == 'canonical'
        assert info['canonical_stress_cases'] == 1 and info['library_stress_cases'] == 0
        assert 'contract' not in info
        assert batch.diagnostics['shared_recovery']['physical_element_evaluations'] == 1


@pytest.mark.parametrize('value', ['library', None, 1])
def test_invalid_stress_execution_refused(value):
    with pytest.raises(PreparedStaticCapabilityError, match='stress_execution'):
        prepare_static_response_evaluator(_bar(), {'p': _case(1).load_case}, stress_execution=value)


def test_library_route_reconstructs_stress_and_counts():
    with prepare_static_response_evaluator(_shell_model(), {'p': _shell_case(0, pressure=2.).load_case},
                                           recovery_config=STRESS, **LIBRARY) as owner:
        batch = next(owner.iter_batches([PatternCase('a', {'p': .3}), PatternCase('b', {'p': -1.})]))
        info = owner.diagnostics()['source_response']
        assert info['reconstructed_cases'] == 2 and info['library_stress_cases'] == 2
        assert info['contract'] == 'certified_response_route_m16'
        assert info['element_library_elements'] == 2
        assert info['element_library_worst_verification_residual'] <= 1e-9
        assert batch.diagnostics['shared_recovery']['physical_element_evaluations'] == 0
        assert batch.diagnostics['shared_recovery']['library_element_records'] == 4
        for result in batch.case_results:
            assert result['solver_info']['source_response']['stress_execution'] == 'certified_library'
            assert result['solver_info']['relative_backward_error'] <= 1e-10
            assert set(result['element_stresses']) == {1, 2}


def test_static_case_in_library_route_stays_canonical():
    with prepare_static_response_evaluator(_bar(), {'p': _case(1, 3).load_case}, recovery_config=STRESS, **LIBRARY) as owner:
        batch = next(owner.iter_batches([_case(2, 4), PatternCase('a', {'p': 1.})]))
        flags = [r['solver_info']['source_response']['stress_execution'] for r in batch.case_results]
        assert flags == ['canonical_fallback', 'certified_library']
        assert batch.diagnostics['shared_recovery']['physical_element_evaluations'] == 1


def test_source_agreement_is_term_scaled_not_bitwise():
    with prepare_static_response_evaluator(_bar(), {'p': _case(1, 1).load_case}, recovery_config=STRESS, **LIBRARY) as owner:
        next(owner.iter_batches([PatternCase('tiny', {'p': 1.}, canonical_load_case=_case(2, 1 + 1e-14).load_case)]))
        info = owner.diagnostics()['source_response']
        assert info['reconstructed_cases'] == 1 and info['source_agreement_fallback_cases'] == 0
        assert 0. < info['max_source_discrepancy_ratio'] <= 1e-12
        batch = next(owner.iter_batches([PatternCase('large', {'p': 1.}, canonical_load_case=_case(3, 1 + 1e-9).load_case)]))
        info = owner.diagnostics()['source_response']
        assert info['source_agreement_fallback_cases'] == 1 and info['reconstructed_cases'] == 1
        assert batch.case_results[0]['solver_info']['source_response']['stress_execution'] == 'canonical_fallback'
        assert batch.case_results[0]['solver_info']['load']['source_agreement']['agrees'] is False


def test_failed_certificate_falls_back(monkeypatch):
    with prepare_static_response_evaluator(_bar(), {'p': _case(1, 3).load_case}, recovery_config=STRESS, **LIBRARY) as owner:
        real = owner._backward_error
        calls = []
        def strict(*args):
            calls.append(1)
            return 1.0 if len(calls) == 1 else real(*args)
        owner._backward_error = strict
        batch = next(owner.iter_batches([PatternCase('a', {'p': 2.})]))
        info = owner.diagnostics()['source_response']
        assert info['certification_fallback_cases'] == 1 and info['library_stress_cases'] == 0
        assert batch.case_results[0]['solver_info']['source_response']['stress_execution'] == 'canonical_fallback'


def test_nonaffine_recovery_refused(monkeypatch):
    import anyfesolver.prepared_static_response as response_module
    zero = {'axial_stress': np.zeros(2), 'label': 'beam'}
    schema = response_module._library_schema(7, zero)
    monkeypatch.setattr(response_module, '_recover_owned_local',
                        lambda recipe, local: {'axial_stress': np.full(2, float(np.sum(local)) ** 2), 'label': 'beam'})
    with pytest.raises(PreparedStaticCapabilityError, match='not affine'):
        response_module._verify_library_element(7, None, 12, zero, schema)


def test_unreproducible_derived_measure_refused(monkeypatch):
    import anyfesolver.prepared_static_response as response_module
    zero = {'axial_stress': np.zeros(2), 'von_mises': np.zeros(2)}
    schema = response_module._library_schema(8, zero)
    monkeypatch.setattr(response_module, '_recover_owned_local',
                        lambda recipe, local: {'axial_stress': np.full(2, float(np.sum(local))), 'von_mises': np.zeros(2)})
    with pytest.raises(PreparedStaticCapabilityError, match='derived measure'):
        response_module._verify_library_element(8, None, 12, zero, schema)


@pytest.mark.parametrize('stage', ['element_schema', 'element_library'])
def test_library_token_never_runs_in_owned_scope(monkeypatch, stage):
    import anyfesolver.prepared_static_response as response_module
    import anyfesolver.prepared_static_recovery as recovery_module
    observed = []
    class HostileToken:
        def raise_if_cancelled(self, where):
            if where == 'prepared_response.' + stage:
                observed.append(recovery_module._OWNED_RECOVERY_SCOPE.get())
                monkeypatch.setattr(response_module, 'evaluate_load_vector', lambda *args: pytest.fail('mutated provider executed'))
    with pytest.raises(PreparedStaticError, match='authority'):
        prepare_static_response_evaluator(_bar(), {'p': _case(1).load_case}, recovery_config=STRESS,
                                          cancellation_token=HostileToken(), **LIBRARY)
    assert observed == [None]


def test_envelope_route_labels_and_close_releases_library():
    envelope = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)
    owner = prepare_static_response_evaluator(_bar(), {'p': _case(1, 3).load_case}, envelope_recovery_config=envelope, **LIBRARY)
    with owner:
        list(owner.iter_batches([PatternCase(str(i), {'p': c}) for i, c in enumerate((1., -2., .5))]))
        summary = owner.envelope_result()
        assert summary.complete and summary.diagnostics['governing_identity_basis'] == 'certified_response_values'
        assert summary.diagnostics['recovery'] == 'certified_response_library_with_canonical_fallback'
        assert owner.diagnostics()['source_response']['compiled_envelope_responses'] is True
        assert owner.diagnostics()['source_response']['library_bytes'] > 0
    assert owner._element_library == {}
    assert owner.diagnostics()['source_response']['library_bytes'] == 0


def test_offsets_and_constrained_loads_once_in_library_route():
    from anyfesolver import BoundaryCondition, prepare_static_evaluator
    model = _bar(False)
    model.add_boundary_condition(BoundaryCondition('left', [1], {'ux': 0.001, 'uy': 0., 'uz': 0., 'rx': 0., 'ry': 0., 'rz': 0.}))
    model.add_boundary_condition(BoundaryCondition('right', [2], {'uy': 0., 'uz': 0., 'rx': 0., 'ry': 0., 'rz': 0.}))
    with prepare_static_response_evaluator(model, {'p': _case(1, 4).load_case}, baseline_load_case=_case(2, 2).load_case,
                                           recovery_config=STRESS, **LIBRARY) as response:
        got = next(response.iter_batches([PatternCase('x', {'p': 2.})]))
    with prepare_static_evaluator(model, recovery_config=STRESS) as direct:
        expected = next(direct.iter_batches([_case(3, 10)]))
    np.testing.assert_allclose(got.displacements, expected.displacements, rtol=1e-12, atol=1e-14)
    for name, value in expected.case_results[0]['element_stresses'][1].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(got.case_results[0]['element_stresses'][1][name], value, rtol=1e-12, atol=1e-9)


def test_freefree_imbalance_policy_unchanged_in_library_route():
    model = _bar(False)
    a = LoadCase('a'); a.add_nodal_load(1, [-3., 0, 0, 0, 0, 0])
    b = LoadCase('b'); b.add_nodal_load(2, [3., 0, 0, 0, 0, 0])
    with prepare_static_response_evaluator(model, {'a': a, 'b': b}, recovery_config=STRESS, **LIBRARY) as evaluator:
        batch = next(evaluator.iter_batches([PatternCase('balanced', {'a': 1., 'b': 1.})]))
        assert batch.case_results[0]['solver_info']['source_response']['stress_execution'] == 'certified_library'
        with pytest.raises(PreparedStaticError, match='rigid-body'):
            next(evaluator.iter_batches([PatternCase('bad', {'a': 1.})]))


def test_development_fixtures_under_frozen_criteria():
    """Disclosed development cases (M8/M9); informs development, not blind validation."""
    import json
    for summary in assessment.development_fixtures():
        print('M16 development summary:', json.dumps(summary, default=float))
        assert summary['failure_count'] == 0, summary
