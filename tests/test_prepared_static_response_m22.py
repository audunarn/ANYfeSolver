"""M22: review repairs (F1-F4, F8) and output-identical speed-ups (S1, S2a, S5, S8)."""
import numpy as np
import pytest
from anyfesolver import (PatternCase, prepare_static_response_evaluator, prepare_static_evaluator, RecoveryConfig,
    LoadCase, BoundaryCondition, PreparedStaticError, PreparedStaticCapabilityError)
from anyfesolver.prepared_static_recovery import (constraint_force_diagnostics, ReactionPlan,
    planned_constraint_force_diagnostics)
from anyfesolver.static_batch import StaticCase
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case
from test_prepared_static_independent import model as independent_model
import prepared_static_response_m16_assessment as assessment
from benchmark_static_batches import mixed_free_model

VECTOR = dict(stress_execution='certified_vectorized')
STRESS = RecoveryConfig(include_stresses=True, include_reactions=True)


def _identical(a, b, path='root'):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), path
    elif isinstance(a, dict):
        assert list(a) == list(b), (path, list(a), list(b))
        for key in a:
            _identical(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            _identical(x, y, f'{path}[{index}]')
    elif isinstance(a, float):
        assert np.array([a]).view(np.uint64)[0] == np.array([b]).view(np.uint64)[0], (path, a, b)
    else:
        assert a == b, (path, a, b)


def _weighted_mpc_model():
    m = independent_model()
    m.boundary_conditions[0].dof_constraints['ux'] = .02
    m.add_node(3, 2., 0., 0.)
    m.add_boundary_condition(BoundaryCondition('slave_other', [3], dict(uy=0., uz=0., rx=0., ry=0., rz=0.)))
    m.add_constraint_equation(terms=((12, 1.), (6, -1.5)), rhs=.01, source_id='weighted affine', dependent_dof=12)
    m.add_constraint_equation(terms=((13, 1.), (7, .5), (6, .25)), rhs=0., source_id='two masters', dependent_dof=13)
    return m


@pytest.mark.parametrize('builder', ['weighted_mpc', 'mixed_free'])
def test_planned_reaction_diagnostics_identical_to_canonical(builder):
    model = _weighted_mpc_model() if builder == 'weighted_mpc' else mixed_free_model()[0]
    with prepare_static_evaluator(model, recovery_config=RecoveryConfig(include_stresses=False)) as owner:
        snap = owner._snapshot
        plan = ReactionPlan(snap.reaction_maps)
        rng = np.random.default_rng(22)
        for _ in range(4):
            u, F = rng.normal(size=snap.total_dofs), rng.normal(size=snap.total_dofs)
            args = (snap.stiffness, snap.transformation, snap.rigid_modes, u, F)
            canonical = constraint_force_diagnostics(snap.reaction_maps, *args,
                constraint_info=snap.constraint_info, nullspace_info=snap.nullspace_info)
            planned = planned_constraint_force_diagnostics(plan, *args,
                constraint_info=snap.constraint_info, nullspace_info=snap.nullspace_info)
            _identical(planned, canonical)
            lean = planned_constraint_force_diagnostics(plan, *args, constraint_info=snap.constraint_info,
                nullspace_info=snap.nullspace_info, maps_only=True)
            for bucket in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
                _identical(lean[bucket], canonical[bucket])
        if builder == 'weighted_mpc':
            assert len(snap.reaction_maps.mpc_constraints) >= 2


def test_node_only_envelope_runs_in_vectorized_route():
    nodes = RecoveryConfig(include_displacements=True, include_stresses=False, include_reactions=True)
    with prepare_static_response_evaluator(_bar(), {'p': _case(1, 3).load_case}, recovery_config=nodes,
                                           envelope_recovery_config=nodes, **VECTOR) as owner:
        batch = next(owner.iter_batches([PatternCase('a', {'p': 2.}), PatternCase('b', {'p': -1.})]))
        summary = owner.envelope_result()
        assert summary.cases_processed == 2 and len(summary.extrema) > 0
        assert batch.case_results[0]['reactions']


def test_compiled_disagreement_falls_back_with_native_load():
    pressure_pattern = _shell_case(0, pressure=2.).load_case
    canonical = LoadCase('one-element')
    canonical.pressure_loads[1] = 1.0
    with prepare_static_response_evaluator(_shell_model(), {'p': pressure_pattern}, recovery_config=STRESS, **VECTOR) as owner:
        batch = next(owner.iter_batches([PatternCase('x', {'p': .5}, canonical_load_case=canonical)]))
        info = owner.diagnostics()['source_response']
        assert info['compiled_disagreement_cases'] == 1 and info['native_load_cases'] == 1 and info['compiled_load_cases'] == 0
        solver_info = batch.case_results[0]['solver_info']
        assert solver_info['load']['load_execution'] == 'native_after_compiled_disagreement'
        assert solver_info['source_response']['state'] == 'canonical_solve'
    with prepare_static_evaluator(_shell_model(), recovery_config=STRESS) as direct:
        expected = next(direct.iter_batches([StaticCase('x', canonical)]))
    assert np.array_equal(batch.displacements, expected.displacements)


def test_new_bindings_are_pinned_by_response_guard(monkeypatch):
    import anyfesolver.prepared_static_vectorized as vector_module
    patterns = {'p': _shell_case(0, pressure=2.).load_case}
    with prepare_static_response_evaluator(_shell_model(), patterns, recovery_config=STRESS, **VECTOR) as owner:
        monkeypatch.setattr(vector_module, 'evaluate_load_vector', lambda *args: pytest.fail('replaced helper executed'))
        with pytest.raises(PreparedStaticCapabilityError, match='authority'):
            next(owner.iter_batches([PatternCase('a', {'p': 1.})]))


def test_vector_memory_checked_before_allocation():
    from anyfesolver import prepared_static_vectorized as vector
    with prepare_static_response_evaluator(_shell_model(), {'p': _shell_case(0, pressure=2.).load_case},
                                           recovery_config=STRESS) as owner:
        union = owner._library_elements(owner._snapshot)
        seen = []
        class Refused(Exception):
            pass
        def check(extra):
            seen.append(extra)
            raise Refused()
        with pytest.raises(Refused):
            vector.build_vector_recovery(union, owner._snapshot.envelope_plan, None, lambda stage: None, memory_check=check)
        assert len(seen) == 1 and seen[0] > 0


def test_load_scope_selection_counts():
    patterns = {'p': _shell_case(0, pressure=2.).load_case}
    compiled_cases = [PatternCase(str(i), {'p': c}, canonical_load_case=assessment._combined_load_case(patterns, {'p': c}, str(i)))
                      for i, c in enumerate((1., -.5))]
    with prepare_static_response_evaluator(_shell_model(), patterns, recovery_config=STRESS, **VECTOR) as owner:
        list(owner.iter_batches(compiled_cases, batch_size=2))
        list(owner.iter_batches([compiled_cases[0], _shell_case(4, pressure=1.)], batch_size=2))
        info = owner.diagnostics()['source_response']
        assert info['compiled_load_scope_blocks'] == 1 and info['native_load_scope_blocks'] == 1
