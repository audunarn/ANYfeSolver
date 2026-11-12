"""M20: lean reaction maps, metadata copies, per-block authority checks and compiled pressure loads."""
import copy
import json
import sys

import numpy as np
import pytest
from anyfesolver import (PatternCase, prepare_static_response_evaluator, prepare_static_evaluator, RecoveryConfig,
    LoadCase, BoundaryCondition, PreparedStaticCapabilityError)
from anyfesolver.prepared_static_loads import ingest_static_case
from anyfesolver.prepared_static_recovery import constraint_force_diagnostics, constraint_force_maps
from anyfesolver.prepared_static_vectorized import make_metadata_copier, _structurally_identical
from anyfesolver.static_batch import StaticCase
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case
from test_prepared_static_independent import model as independent_model
import prepared_static_response_m16_assessment as assessment

VECTOR = dict(stress_execution='certified_vectorized')
STRESS = RecoveryConfig(include_stresses=True, include_reactions=True)


def _mpc_snapshot_owner():
    m = independent_model()
    m.boundary_conditions[0].dof_constraints['ux'] = .02
    m.add_node(3, 2., 0., 0.)
    m.add_boundary_condition(BoundaryCondition('slave_other', [3], dict(uy=0., uz=0., rx=0., ry=0., rz=0.)))
    m.add_constraint_equation(terms=((12, 1.), (6, -1.5)), rhs=.01, source_id='weighted affine', dependent_dof=12)
    return prepare_static_evaluator(m, recovery_config=RecoveryConfig(include_stresses=False))


def _payload_equal(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def test_force_maps_bitwise_equal_full_diagnostics_and_metadata_copies():
    with _mpc_snapshot_owner() as owner:
        snap = owner._snapshot
        assert snap.reaction_maps.mpc_constraints, 'fixture must exercise MPC buckets'
        rng = np.random.default_rng(20)
        for _ in range(3):
            u = rng.normal(size=snap.total_dofs)
            F = rng.normal(size=snap.total_dofs)
            args = (snap.reaction_maps, snap.stiffness, snap.transformation, snap.rigid_modes, u, F)
            full = constraint_force_diagnostics(*args, constraint_info=snap.constraint_info, nullspace_info=snap.nullspace_info)
            lean = constraint_force_maps(snap.reaction_maps, snap.stiffness, u, F)
            for bucket in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
                assert list(lean[bucket]) == list(full[bucket])
                assert all(_payload_equal(lean[bucket][k], full[bucket][k]) for k in full[bucket])
            copies = make_metadata_copier(snap.constraint_info, snap.nullspace_info)
            assert copies is not None
            fast = constraint_force_diagnostics(*args, constraint_info=snap.constraint_info,
                                                nullspace_info=snap.nullspace_info, metadata_copies=copies)
            assert list(fast) == list(full)
            for key in full:
                if key in ('constraint_info', 'nullspace_info'):
                    assert _structurally_identical(fast[key], full[key])
                elif isinstance(full[key], np.ndarray):
                    assert _payload_equal(fast[key], full[key])
                else:
                    assert json.dumps(fast[key], sort_keys=True, default=repr) == json.dumps(full[key], sort_keys=True, default=repr)
        first, second = copies(), copies()
        assert first[0] is not second[0] and first[1] is not second[1]
        first[0]['__mutated__'] = True
        assert '__mutated__' not in second[0] and '__mutated__' not in snap.constraint_info


def test_per_case_authority_check_skipped_only_inside_block_scope():
    calls = []
    def tracer(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'require_native_source_authority' and \
                frame.f_back is not None and frame.f_back.f_code.co_name == '_load_vector':
            calls.append(1)
    patterns = {'p': _shell_case(0, pressure=2.).load_case}
    cases = [PatternCase(str(i), {'p': c}, canonical_load_case=assessment._combined_load_case(patterns, {'p': c}, str(i)))
             for i, c in enumerate((1., -2., .5, 3.))]
    with prepare_static_response_evaluator(_shell_model(), patterns, recovery_config=STRESS, **VECTOR) as owner:
        sys.setprofile(tracer)
        try:
            list(owner.iter_batches(cases, batch_size=4))
        finally:
            sys.setprofile(None)
        assert calls == []
        # Outside a live owned scope the per-case check still runs.
        _, _, packet = owner._ingest_case(cases[0])
        sys.setprofile(tracer)
        try:
            owner._load_vector(owner._snapshot, packet)
        finally:
            sys.setprofile(None)
        assert calls == [1]


def _shell_patterns_and_cases():
    second = LoadCase('second')
    second.add_pressure_load(1, -3.)
    second.add_pressure_load(2, .5)
    patterns = {'p': _shell_case(0, pressure=2.).load_case, 'g': _shell_case(1, gravity=True).load_case, 's': second}
    coefficients = [{'p': 1.3, 'g': -.4, 's': 2.}, {'p': -2.}, {'s': .7, 'g': 1.}]
    cases = [PatternCase(str(i), c, canonical_load_case=assessment._combined_load_case(patterns, c, str(i)))
             for i, c in enumerate(coefficients)]
    return patterns, cases


def test_compiled_pressure_loads_used_and_meet_c7():
    patterns, cases = _shell_patterns_and_cases()
    with prepare_static_response_evaluator(_shell_model(), patterns, recovery_config=STRESS, **VECTOR) as owner:
        batch = next(owner.iter_batches(cases, batch_size=3))
        info = owner.diagnostics()['source_response']
        assert info['load_execution'] == 'compiled_pressure_bases'
        assert info['compiled_pressure_worst_c7_ratio'] <= 1.0
        assert info['compiled_load_cases'] == 3 and info['native_load_cases'] == 0
        assert info['source_agreement_fallback_cases'] == 0 and info['reconstructed_cases'] == 3
        assert info['reaction_metadata_copies'] == 'pickle_of_plain_snapshot'
        for result in batch.case_results:
            assert result['solver_info']['load']['load_execution'] == 'compiled_pressure_bases'
            assert result['solver_info']['relative_backward_error'] <= 1e-10
        assert owner._compiled_pressure.verified_kinds == frozenset({'scalar'})


def test_follower_and_direct_cases_keep_native_loads():
    patterns = {'p': _shell_case(0, pressure=2.).load_case}
    follower = assessment._combined_load_case(patterns, {'p': 1.}, 'f')
    follower.follower_pressure = True
    with prepare_static_response_evaluator(_shell_model(), patterns, recovery_config=STRESS, **VECTOR) as owner:
        next(owner.iter_batches([PatternCase('f', {'p': 1.}, canonical_load_case=follower), _shell_case(3, pressure=1.)]))
        info = owner.diagnostics()['source_response']
        assert info['native_load_cases'] == 1 and info['compiled_load_cases'] == 0


def test_c7_failure_at_preparation_disables_compiled_loads():
    patterns, _cases = _shell_patterns_and_cases()
    with prepare_static_response_evaluator(_shell_model(), patterns, recovery_config=STRESS, **VECTOR) as owner:
        snap = owner._snapshot
        packets = [ingest_static_case(StaticCase('baseline', None))[2]] + [
            ingest_static_case(StaticCase(name, value))[2] for name, value in patterns.items()]
        wrong = np.array(owner._force_columns, dtype=float) * (1.0 + 1e-9)
        owner._prepare_compiled_pressure(snap, packets, wrong)
        assert owner._compiled_pressure is None
        assert 'C7 failed' in owner.diagnostics()['source_response']['compiled_pressure_note']


def test_development_fixtures_still_meet_frozen_criteria():
    for summary in assessment.development_fixtures(stress_execution='certified_vectorized'):
        assert summary['failure_count'] == 0, summary
