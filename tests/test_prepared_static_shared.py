"""M4 shared truth, bounded winner tiles and copy-on-write publication."""
import numpy as np
import pytest
from anyfesolver import prepare_static_evaluator, RecoveryConfig, PreparedStaticCapabilityError
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case


def test_detailed_derived_field_does_not_expand_native_envelope_schema():
    selected = RecoveryConfig(components=['principal_stresses_top'], include_displacements=False, include_reactions=False)
    whole = RecoveryConfig(include_displacements=False, include_reactions=False)
    case = _shell_case(0, pressure=2.)
    with prepare_static_evaluator(_shell_model(), recovery_config=selected, envelope_recovery_config=whole) as owner:
        block = next(owner.iter_batches([case]))
        assert 'principal_stresses_top' in block.case_results[0]['element_stresses'][1]
        assert all('principal_stresses' not in entry.component for entry in owner.envelope_result().extrema)
        assert block.diagnostics['shared_recovery']['physical_element_evaluations'] == 2


@pytest.mark.parametrize('batch_size', [1, 8, 32])
def test_shared_overlap_block_winners_and_independent_reaction_selection(batch_size):
    config = RecoveryConfig(components=['axial_stress'], node_ids=[2], include_reactions=True)
    envelope = RecoveryConfig(components=['axial_stress'], include_reactions=True)
    forces = [1., -2., 2., 0., 2., -2., 1., 0., 1.] * 2 + [2.]
    cases = [_case(i % 3, force=value) for i, value in enumerate(forces)]
    with prepare_static_evaluator(_bar(), recovery_config=config, envelope_recovery_config=envelope) as owner:
        blocks = list(owner.iter_batches(cases, batch_size=batch_size))
        records = [row for block in blocks for row in block.case_results]
        for block in blocks:
            stats = block.diagnostics['shared_recovery']
            assert stats['physical_element_evaluations'] == len(block.case_results)
            assert stats['reaction_diagnostic_evaluations'] == len(block.case_results)
            assert stats['overlapping_element_cases'] == len(block.case_results)
            assert stats['max_case_tile'] <= 8
        assert all(1 not in row['constraint_forces']['support_reactions'] for row in records)
        for entry in owner.envelope_result().extrema:
            if entry.quantity == 'stress':
                samples = np.asarray([row['element_stresses'][entry.entity_id][entry.component] for row in records])
                for mode, winners in ((entry.minimum, np.argmin(samples, axis=0)),
                                      (entry.maximum, np.argmax(samples, axis=0)),
                                      (entry.absolute, np.argmax(np.abs(samples), axis=0))):
                    np.testing.assert_array_equal(mode.input_indices, winners)
                    np.testing.assert_array_equal(mode.values, samples[winners])
            if entry.quantity == 'support_reactions' and entry.entity_id == 1:
                # The canonical solve/reaction arithmetic need not return the
                # exact decimal input; use the independently retained full truth.
                expected = [-row['element_stresses'][1]['axial_stress'] for row in records]
                assert entry.minimum.values[0] == min(expected)
                assert entry.maximum.values[0] == max(expected)


def test_disjoint_selected_and_envelope_elements_recover_union_once():
    selected = RecoveryConfig(element_ids=[1], components=['membrane_xx'], include_displacements=False, include_reactions=False)
    envelope = RecoveryConfig(element_ids=[2], components=['membrane_xx'], include_displacements=False, include_reactions=False)
    with prepare_static_evaluator(_shell_model(), recovery_config=selected, envelope_recovery_config=envelope) as owner:
        block = next(owner.iter_batches([_shell_case(0, pressure=1.)]))
        assert set(block.case_results[0]['element_stresses']) == {1}
        assert {entry.entity_id for entry in owner.envelope_result().extrema} == {2}
        assert block.diagnostics['shared_recovery']['physical_element_evaluations'] == 2
        assert block.diagnostics['shared_recovery']['overlapping_element_cases'] == 0


def test_transaction_copies_only_changed_modes_and_preserves_committed_arrays():
    config = RecoveryConfig(components=['axial_stress'], include_displacements=False, include_reactions=False)
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as owner:
        list(owner.iter_batches([_case(0, force=2.)]))
        committed = owner._envelope
        key = next(key for key in committed.entries if key[0] == 'stress')
        previous = committed.entries[key][1]
        original = previous[0]['values'][0]
        transaction = committed.transaction()
        transaction.fold_tile('stress', 1, {'axial_stress': np.asarray([original])}, [('later', 9., 1)])
        assert transaction.copied_mode_count == 0
        assert transaction.entries[key][1][0] is previous[0]
        transaction.fold_tile('stress', 1, {'axial_stress': np.asarray([-3.])}, [('new', 10., 2)])
        assert transaction.copied_mode_count == 2
        assert transaction.entries[key][1][1] is previous[1]
        assert previous[0]['values'][0] == original
        assert previous[2]['values'][0] == original


def test_shared_dispatch_rebinding_refused_before_receiving_owned_buffers(monkeypatch):
    import anyfesolver.prepared_static_shared as shared
    with prepare_static_evaluator(_bar()) as owner:
        seen = []
        monkeypatch.setattr(shared, 'recover_shared_block', lambda *args: seen.append(args))
        with pytest.raises(PreparedStaticCapabilityError, match='shared recovery'):
            list(owner.iter_batches([_case(0)]))
        assert seen == []


@pytest.mark.parametrize('observation', ['result', 'restart', 'resume'])
def test_reducer_storage_helper_refused_before_owned_data_observation(monkeypatch, observation):
    import anyfesolver.prepared_static_envelopes as envelopes
    config = RecoveryConfig(include_stresses=False)
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as owner:
        iterator = owner.iter_batches([_case(0), _case(1)], batch_size=1)
        next(iterator)
        if observation != 'resume':
            iterator.close()
        seen = []
        monkeypatch.setattr(envelopes, '_storage', lambda value, *args: seen.append(value) or 0)
        with pytest.raises(PreparedStaticCapabilityError, match='shared recovery'):
            if observation == 'result':
                owner.envelope_result()
            elif observation == 'restart':
                next(owner.iter_batches([_case(0)]))
            else:
                next(iterator)
        assert seen == []


def test_driver_constructor_alias_refused_before_private_plan(monkeypatch):
    import anyfesolver.prepared_static as driver
    config = RecoveryConfig(include_stresses=False)
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as owner:
        seen = []
        monkeypatch.setattr(driver, 'EnvelopeAccumulator', lambda plan: seen.append(plan))
        # The response authority check (M-later) may refuse the replaced binding first;
        # either refusal comes before the private plan reaches the alias.
        with pytest.raises(PreparedStaticCapabilityError,
                           match='constructor authority|executable authority'):
            next(owner.iter_batches([_case(0)]))
        assert seen == []


def test_tile_failure_has_case_element_and_stage_without_publication():
    from anyfesolver import PreparedStaticError
    from anyfesolver.prepared_static_shared import _fold_labeled
    config = RecoveryConfig(components=['axial_stress'], include_displacements=False, include_reactions=False)
    with prepare_static_evaluator(_bar(), envelope_recovery_config=config) as owner:
        scratch = owner._envelope.empty_subset(('stress',), 1)
        with pytest.raises(PreparedStaticError) as caught:
            _fold_labeled(scratch, 'stress', 1, {'axial_stress': np.asarray([np.nan])}, [('bad', 3., 4)])
        assert caught.value.diagnostics['case_id'] == 'bad'
        assert caught.value.diagnostics['input_index'] == 4
        assert caught.value.diagnostics['element_id'] == 1
        assert caught.value.diagnostics['stage'] == 'envelope reduction'
        assert caught.value.__cause__ is not None
        assert owner.envelope_result().cases_processed == 0
