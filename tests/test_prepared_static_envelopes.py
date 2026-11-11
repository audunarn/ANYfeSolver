import numpy as np
import pytest
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case
from anyfesolver import prepare_static_evaluator, RecoveryConfig, StaticCase
from anyfesolver.prepared_static_loads import ingest_static_case, evaluate_load_vector
from anyfesolver.matrix_assembly import assemble_load_vector


def test_canonical_load_staging_is_bitwise_for_pressure_and_added_mass():
    model = _shell_model()
    model.add_point_mass(1, 1.23456789)
    with prepare_static_evaluator(model) as owner:
        for pressure in (1.23456789, -3.45678901, 1e-9):
            case = _shell_case(1, pressure=pressure, gravity=True, added=.33333333)
            expected = assemble_load_vector(model, case.load_case)[0]
            actual = evaluate_load_vector(owner._snapshot.load_maps, ingest_static_case(case)[2])[0]
            np.testing.assert_array_equal(actual, expected)


def test_envelope_canonical_min_max_absolute_ties_and_detachment():
    config = RecoveryConfig(components=['axial_stress'], include_displacements=False, include_reactions=True)
    cases = [_case(0), _case(1), _case(0)]
    with prepare_static_evaluator(_bar(), recovery_config=config, envelope_recovery_config=config) as owner:
        blocks = list(owner.iter_batches(cases, batch_size=2))
        summary = owner.envelope_result()
        assert summary.complete and summary.cases_processed == 3
        entry = next(e for e in summary.extrema if e.quantity == 'stress')
        samples = [row['element_stresses'][entry.entity_id]['axial_stress'] for block in blocks for row in block.case_results]
        np.testing.assert_equal(entry.minimum.values, min(samples))
        np.testing.assert_equal(entry.maximum.values, max(samples))
        assert entry.minimum.input_indices == np.argmin(samples)
        assert entry.maximum.input_indices == np.argmax(samples)
        entry.maximum.values[...] = 12345
        assert owner.envelope_result().extrema[0].maximum.values != 12345
        list(owner.iter_batches([]))
        assert owner.envelope_result().complete and owner.envelope_result().cases_processed == 0


def test_envelope_transaction_callback_failure_and_single_active_run():
    config = RecoveryConfig(include_displacements=True, include_stresses=False, include_reactions=False)
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as owner:
        stream = owner.iter_batches([_case(0), _case(1)], batch_size=1)
        next(stream)
        other = owner.iter_batches([_case(0)])
        with pytest.raises(RuntimeError, match='one prepared envelope'):
            next(other)
        stream.close()
        assert not owner.envelope_result().complete
        assert owner.envelope_result().cases_processed == 1
        def progress(event):
            raise ValueError('no publication')
        with pytest.raises(ValueError, match='no publication'):
            list(owner.iter_batches([_case(0)], progress_callback=progress))
        assert owner.envelope_result().cases_processed == 0
        assert not owner.envelope_result().complete


def test_envelope_principal_associated_fields_are_same_sample():
    config = RecoveryConfig(components=['principal_stresses_top', 'membrane_xx'], include_displacements=False, include_reactions=False)
    cases = [_shell_case(i, pressure=p) for i, p in enumerate((2., -1., 3.))]
    with prepare_static_evaluator(_shell_model(), recovery_config=config, envelope_recovery_config=config) as owner:
        blocks = list(owner.iter_batches(cases, batch_size=2))
        records = [row for block in blocks for row in block.case_results]
        for entry in owner.envelope_result().extrema:
            for mode in (entry.minimum, entry.maximum, entry.absolute):
                for index in np.ndindex(entry.original_shape):
                    winner = int(mode.input_indices[index])
                    actual = mode.associated_fields['membrane_xx'][index]
                    expected = records[winner]['element_stresses'][entry.entity_id]['membrane_xx']
                    np.testing.assert_array_equal(actual, expected)


def test_owned_scope_refuses_replaced_numpy_proof(monkeypatch):
    import anyfesolver.boundary as boundary
    from anyfesolver.prepared_static_recovery import owned_recovery_scope
    from anyfesolver import PreparedStaticCapabilityError
    monkeypatch.setattr(boundary, '_numpy_error_routes_are_builtin', lambda: True)
    with pytest.raises(PreparedStaticCapabilityError, match='guard binding authority'):
        with owned_recovery_scope(()):
            pytest.fail('untrusted scope entered')


def test_envelope_callback_reads_prior_summary_and_close_keeps_it():
    config = RecoveryConfig(include_displacements=True, include_stresses=False, include_reactions=False)
    owner = prepare_static_evaluator(_bar(), envelope_recovery_config=config)
    observed = []
    def progress(event):
        observed.append(owner.envelope_result().cases_processed)
        if len(observed) == 2:
            owner.close()
    with pytest.raises(RuntimeError, match='closed'):
        list(owner.iter_batches([_case(0), _case(1)], batch_size=1, progress_callback=progress))
    assert observed == [0, 1]
    assert owner.envelope_result().cases_processed == 1
    assert not owner.envelope_result().complete


def test_envelope_input_cleanup_failure_does_not_mark_complete():
    config = RecoveryConfig(include_displacements=True, include_stresses=False, include_reactions=False)
    class Source:
        def __iter__(self):
            return self
        def __next__(self):
            raise StopIteration
        def close(self):
            raise ValueError('cleanup failed')
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as owner:
        with pytest.raises(ValueError, match='cleanup failed'):
            list(owner.iter_batches(Source()))
        assert not owner.envelope_result().complete


def test_original_free_counterexamples_have_exact_load_displacement_and_envelopes():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from benchmark_static_batches import mixed_free_model, free_cases, numeric_leaves
    from anyfesolver import iter_linear_batches, EnvelopeReducer
    from dataclasses import replace
    model, groups, balancing = mixed_free_model()
    cases = [StaticCase(name, load, instant) for name, instant, load in free_cases(model, groups, balancing, [0, 48, 576])]
    config = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)
    baseline = list(iter_linear_batches(model, cases, recovery_config=config, batch_size=2))
    reducer = EnvelopeReducer()
    for block in baseline:
        reducer.update_batch(replace(block, output_dofs=np.zeros(0, dtype=int), displacements=np.zeros((0, len(block.case_ids)))))
    with prepare_static_evaluator(model, recovery_config=config, envelope_recovery_config=config) as owner:
        prepared = list(owner.iter_batches(cases, batch_size=2))
        for expected, actual in zip(baseline, prepared):
            np.testing.assert_array_equal(expected.displacements, actual.displacements)
            for left, right in zip(expected.case_results, actual.case_results):
                assert dict(numeric_leaves(left['element_stresses'])) == dict(numeric_leaves(right['element_stresses']))
        keys = set()
        for entry in owner.envelope_result().extrema:
            prefix = (entry.quantity, entry.entity_id, entry.component) if entry.quantity == 'stress' else (entry.quantity, entry.entity_id)
            for coordinate in np.ndindex(entry.original_shape):
                key = prefix + coordinate
                keys.add(key)
                expected = reducer.extrema[key]
                for ours, theirs in ((entry.minimum, expected.minimum), (entry.maximum, expected.maximum), (entry.absolute, expected.absolute_maximum)):
                    assert float(ours.values[coordinate]) == theirs.value
                    assert int(ours.input_indices[coordinate]) == theirs.input_index
                    assert ours.case_ids[coordinate] == theirs.case_id
        assert keys == set(reducer.extrema)


@pytest.mark.parametrize('module_name,factory_name', [('e4_pl_element', '_q4_trusted_operation_scope'), ('e4_pl_s3_element', '_s3_trusted_operation_scope')])
@pytest.mark.parametrize('mutation', ['binding', 'code'])
def test_owned_scope_pins_factories_before_lending_private_elements(monkeypatch, module_name, factory_name, mutation):
    import importlib
    from anyfesolver.prepared_static_recovery import owned_recovery_scope
    from anyfesolver import PreparedStaticCapabilityError
    module = importlib.import_module('anyfesolver.' + module_name)
    factory = getattr(module, factory_name)
    observed = []
    if mutation == 'binding':
        monkeypatch.setattr(module, factory_name, lambda elements: observed.append(elements))
    else:
        monkeypatch.setattr(factory, '__code__', factory.__code__.replace(co_filename='changed-scope-factory'))
    with pytest.raises(PreparedStaticCapabilityError, match='scope factory authority'):
        with owned_recovery_scope(()):
            pytest.fail('untrusted scope entered')
    assert observed == []


@pytest.mark.parametrize('phase', ['exhaustion', 'cleanup'])
@pytest.mark.parametrize('action', ['close', 'cancel'])
def test_envelope_exhaustion_and_cleanup_observations_cannot_certify_closed_or_cancelled(phase, action):
    from anyfesolver import CancellationToken, SolveCancelled
    config = RecoveryConfig(include_displacements=True, include_stresses=False, include_reactions=False)
    token = CancellationToken()
    owner = prepare_static_evaluator(_bar(), envelope_recovery_config=config)
    class Source:
        def __iter__(self):
            return self
        def change(self):
            owner.close() if action == 'close' else token.cancel()
        def __next__(self):
            if phase == 'exhaustion':
                self.change()
            raise StopIteration
        def close(self):
            if phase == 'cleanup':
                self.change()
    with pytest.raises(RuntimeError if action == 'close' else SolveCancelled):
        list(owner.iter_batches(Source(), cancellation_token=token))
    assert not owner.envelope_result().complete
    owner.close()


def test_envelope_one_case_policy_counts_transaction_and_tile_before_factor():
    # Replacing a module function after import is outside the supported contract
    # (the response authority check refuses it); the policy refusal is checked directly.
    from anyfesolver import ResourceConfig
    from anyfesolver.recovery import ResourcePolicyError
    config = RecoveryConfig(components=['axial_stress'], include_displacements=True, include_reactions=True)
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as probe:
        snapshot = probe._snapshot
        matrix = probe._operator_matrix
        matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        without_envelope_workspace = snapshot.nbytes() + matrix_bytes * 5 + snapshot.total_dofs * 80
        assert probe._envelope.nbytes() > 0
        assert snapshot.envelope_plan.tile_bytes > 0
    with pytest.raises(ResourcePolicyError, match='one case'):
        prepare_static_evaluator(_bar(), envelope_recovery_config=config,
            resource_config=ResourceConfig(memory_limit_bytes=int(without_envelope_workspace)))


def test_envelope_accumulator_storage_is_case_count_independent_and_resets():
    config = RecoveryConfig(components=['axial_stress'], include_displacements=False, include_reactions=False)
    case = _case(0)
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as owner:
        stream = owner.iter_batches([case] * 9, batch_size=1)
        next(stream)
        first_bytes = owner.envelope_result().diagnostics['retained_accumulator_bytes']
        list(stream)
        assert owner.envelope_result().diagnostics['retained_accumulator_bytes'] == first_bytes
        assert owner.envelope_result().cases_processed == 9
        list(owner.iter_batches([case]))
        assert owner.envelope_result().cases_processed == 1
        assert owner.envelope_result().diagnostics['retained_accumulator_bytes'] == first_bytes
