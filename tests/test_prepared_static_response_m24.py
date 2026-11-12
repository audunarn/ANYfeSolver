"""M24: opt-in lean selected reactions (certified_vectorized only)."""
import json
from pathlib import Path

import numpy as np
import pytest
from anyfesolver import PatternCase, prepare_static_response_evaluator, RecoveryConfig, PreparedStaticCapabilityError
from test_prepared_static_evaluator import _bar, _case
import prepared_static_response_m16_assessment as assessment

EVIDENCE = Path(__file__).resolve().parents[1] / '.prepared-static-evidence' / 'm24-assessment-01.json'
BUCKETS = ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces')


def _run(diagnostics):
    patterns = assessment._shell_patterns()
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True, components=assessment.SHELL_COMPONENTS)
    envelope = RecoveryConfig(include_displacements=True, include_stresses=True, include_reactions=True,
                              components=assessment.SHELL_COMPONENTS)
    sets = assessment._random_sets(24, 5, tuple(patterns), -3.0, 3.0)
    cases = [PatternCase(str(i), c, canonical_load_case=assessment._combined_load_case(patterns, c, str(i)))
             for i, c in enumerate(sets)]
    with prepare_static_response_evaluator(assessment._mpc_shell_model(), patterns, recovery_config=recovery,
            envelope_recovery_config=envelope, stress_execution='certified_vectorized',
            selected_reaction_diagnostics=diagnostics) as owner:
        blocks = list(owner.iter_batches(cases, batch_size=2))
        return blocks, owner.envelope_result(), owner.diagnostics()['source_response']


def _bits(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def test_lean_matches_full_bitwise_and_drops_model_wide_diagnostics():
    lean_blocks, lean_env, lean_info = _run('lean')
    full_blocks, full_env, full_info = _run('full')
    assert lean_info['selected_reaction_diagnostics'] == 'lean' and full_info['selected_reaction_diagnostics'] == 'full'
    for lean_block, full_block in zip(lean_blocks, full_blocks):
        assert np.array_equal(lean_block.displacements, full_block.displacements)
        for lean, full in zip(lean_block.case_results, full_block.case_results):
            assert list(lean['reactions']) == list(full['reactions'])
            assert all(_bits(lean['reactions'][n], full['reactions'][n]) for n in full['reactions'])
            forces = lean['constraint_forces']
            assert forces['diagnostics'] == 'lean_selected_reaction_maps'
            assert not {'residual', 'mpc_constraint_forces', 'constraint_info', 'nullspace_info', 'residual_norm'} & set(forces)
            for bucket in BUCKETS:
                assert list(forces[bucket]) == list(full['constraint_forces'][bucket])
                assert all(_bits(forces[bucket][n], full['constraint_forces'][bucket][n]) for n in forces[bucket])
            assert full['constraint_forces']['mpc_master_equivalent_forces'], 'fixture must have master forces'
            for eid, record in full['element_stresses'].items():
                for name, value in record.items():
                    if isinstance(value, np.ndarray):
                        assert _bits(lean['element_stresses'][eid][name], value)
    for got, wanted in zip(lean_env.extrema, full_env.extrema):
        for mode in ('minimum', 'maximum', 'absolute'):
            assert _bits(getattr(got, mode).values, getattr(wanted, mode).values)
            assert np.array_equal(getattr(got, mode).input_indices, getattr(wanted, mode).input_indices)


@pytest.mark.parametrize('value, execution', [('compact', 'certified_vectorized'), ('lean', 'canonical'),
                                              ('lean', 'certified_library')])
def test_lean_selected_reactions_refusals(value, execution):
    with pytest.raises(PreparedStaticCapabilityError, match='selected_reaction_diagnostics|lean selected'):
        prepare_static_response_evaluator(_bar(), {'p': _case(1).load_case}, stress_execution=execution,
                                          selected_reaction_diagnostics=value)


def test_lean_holdouts_meet_frozen_criteria():
    lean = {'selected_reaction_diagnostics': 'lean'}
    mode = 'certified_vectorized'
    summaries = [assessment.holdout_a1(mode, native_loads=True, route_options=lean),
                 assessment.holdout_a3(mode, native_loads=True, route_options=lean),
                 assessment.holdout_a5(mode, native_loads=True, route_options=lean),
                 assessment.holdout_a4(mode, route_options=lean)]
    # M24 evidence (EVIDENCE, m24-assessment-01.json) was written once by the M24
    # version of this test; since the M25 review the test is repeatable and writes
    # nothing (milestone evidence lives in *_assessment.py one-shot tests).
    for summary in summaries:
        assert summary['failure_count'] == 0, summary
    assert summaries[2]['master_entries_checked'] > 0
