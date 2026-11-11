"""Root-authored counterexamples, independent of the implementation owner."""
import gc
import weakref

import numpy as np
import pytest

from anyfesolver import (FEModel, LoadCase, FixedSupport, BoundaryCondition,
                       AnalysisSession, RecoveryConfig, StaticCase, solve_linear,
                       compute_constraint_force_diagnostics, CancellationToken,
                       SolveCancelled)
from anyfesolver.elements import BeamElement
from anyfesolver.prepared_static import prepare_static_evaluator


def model():
    m = FEModel('independent prepared check')
    m.add_material('steel', 100., .3, density=2.)
    m.add_node(1, 0., 0., 0.)
    m.add_node(2, 1., 0., 0.)
    m.add_element(1, BeamElement(1, [1, 2], 'steel',
        dict(area=1., Iy=.001, Iz=.001, J=.001)))
    m.add_boundary_condition(FixedSupport('fixed', [1]))
    m.add_boundary_condition(BoundaryCondition('slide', [2],
        dict(uy=0., uz=0., rx=0., ry=0., rz=0.)))
    return m


def load(value=1., support=0.):
    case = LoadCase('load')
    case.add_nodal_load(2, [value, 0., 0., 0., 0., 0.])
    case.add_nodal_load(1, [support, 0., 0., 0., 0., 0.])
    return case


def config():
    return RecoveryConfig(node_ids=[1, 2], element_ids=[1], include_stresses=False)


def test_source_session_container_and_header_rebinding_cannot_change_snapshot():
    m = model()
    session = AnalysisSession(m)
    with prepare_static_evaluator(m, output_dofs=[6], recovery_config=config(), session=session) as prepared:
        a = next(prepared.iter_batches([StaticCase('a', load())]))
        matrix = session.stiffness_plan().matrix
        matrix.data = np.zeros_like(matrix.data)
        matrix.indices = matrix.indices.copy()
        constraint = session.constraint_plan()
        constraint.T.data = np.zeros_like(constraint.T.data)
        constraint.u0.shape = (constraint.u0.size, 1)
        m.mesh.nodes[2].x = 17.
        session.close()
        b = next(prepared.iter_batches([StaticCase('b', load())]))
        np.testing.assert_array_equal(a.displacements, b.displacements)
        np.testing.assert_allclose(b.displacements, [[.01]], rtol=1e-12, atol=1e-14)


def test_source_model_can_be_collected_and_outputs_cannot_mutate_next_result():
    m = model()
    ref = weakref.ref(m)
    prepared = prepare_static_evaluator(m, output_dofs=[6], recovery_config=config())
    del m
    gc.collect()
    assert ref() is None, 'evaluator retains its live source model'
    first = next(prepared.iter_batches([StaticCase('a', load())]))
    try:
        first.displacements[...] = 999.
    except ValueError:
        pass
    second = next(prepared.iter_batches([StaticCase('b', load())]))
    np.testing.assert_allclose(second.displacements, [[.01]], rtol=1e-12, atol=1e-14)
    prepared.close()
    prepared.close()
    with pytest.raises(Exception):
        list(prepared.iter_batches([StaticCase('closed', load())]))


def test_identical_reduced_loads_preserve_different_support_forces_and_all_diagnostics():
    m = model()
    cases = [load(support=v) for v in (0., 17., -3.)]
    references = []
    for case in cases:
        u, _ = solve_linear(m, case)
        references.append(compute_constraint_force_diagnostics(m, u, case))
    with prepare_static_evaluator(m, output_dofs=[6], recovery_config=config()) as prepared:
        block = next(prepared.iter_batches([StaticCase(str(i), c) for i, c in enumerate(cases)]))
    np.testing.assert_allclose(block.displacements, [[.01, .01, .01]], rtol=1e-12, atol=1e-14)
    for actual_case, expected in zip(block.case_results, references):
        actual = actual_case['constraint_forces']
        for key in ('residual_norm', 'reduced_residual_norm', 'nullspace_generalized_force_norm',
                    'support_reaction_norm', 'mpc_slave_force_norm', 'mpc_master_equivalent_force_norm'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-14)
        for key in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
            assert set(actual[key]) == set(expected[key])
            for node, force in expected[key].items():
                np.testing.assert_allclose(actual[key][node], force, rtol=1e-12, atol=1e-14)


def test_ingress_captures_reused_load_before_observing_next_case():
    m = model()
    shared = load()
    def records():
        for i, value in enumerate((1., -4., 3.)):
            shared.nodal_loads[2][:] = [value, 0., 0., 0., 0., 0.]
            yield StaticCase(str(i), shared, float(i*i))
    with prepare_static_evaluator(m, output_dofs=[6], recovery_config=config()) as prepared:
        blocks = list(prepared.iter_batches(records(), batch_size=2))
    np.testing.assert_allclose(np.hstack([b.displacements for b in blocks]),
                               [[.01, -.04, .03]], rtol=1e-12, atol=1e-14)
    assert tuple(t for b in blocks for t in b.times) == (0., 1., 4.)


def test_per_case_added_mass_acceleration_is_not_frozen_or_double_counted():
    m = model()
    cases = []
    for mass, acceleration in ((0., 2.), (3., -1.), (1., 0.)):
        c = LoadCase('inertia')
        c.gravity = np.array([acceleration, 0., 0.])
        c.added_node_masses[2] = mass
        cases.append(c)
    expected = [solve_linear(m, c)[0][6] for c in cases]
    with prepare_static_evaluator(m, output_dofs=[6], recovery_config=config()) as prepared:
        block = next(prepared.iter_batches([StaticCase(str(i), c) for i,c in enumerate(cases)]))
    np.testing.assert_allclose(block.displacements, [expected], rtol=1e-12, atol=1e-14)


def test_cancel_after_certified_block_and_close_iterator_releases_active_run():
    m = model()
    token = CancellationToken()
    with prepare_static_evaluator(m, output_dofs=[6], recovery_config=config()) as prepared:
        stream = prepared.iter_batches([StaticCase(str(i), load(i+1)) for i in range(4)],
                                       batch_size=1, cancellation_token=token)
        first = next(stream)
        token.cancel()
        with pytest.raises(SolveCancelled):
            next(stream)
        stream.close()
        np.testing.assert_allclose(first.displacements, [[.01]], rtol=1e-12, atol=1e-14)
        again = next(prepared.iter_batches([StaticCase('again', load(2.))]))
        np.testing.assert_allclose(again.displacements, [[.02]], rtol=1e-12, atol=1e-14)

def test_preparation_rejects_raw_structural_change_without_revision_bump(monkeypatch):
    import anyfesolver.prepared_static_snapshot as module
    m = model()
    original = module.compile_canonical_load_maps
    def changing(*args, **kwargs):
        made = original(*args, **kwargs)
        m.mesh.point_masses[2] = 11.
        return made
    monkeypatch.setattr(module, 'compile_canonical_load_maps', changing)
    with pytest.raises(Exception, match='mixed snapshot|snapshot capture'):
        prepare_static_evaluator(m)


def test_weighted_mpc_offsets_and_full_diagnostic_buckets():
    m = model()
    m.boundary_conditions[0].dof_constraints['ux'] = .02
    m.add_node(3, 2., 0., 0.)
    m.add_boundary_condition(BoundaryCondition('slave_other', [3],
        dict(uy=0., uz=0., rx=0., ry=0., rz=0.)))
    m.add_constraint_equation(terms=((12, 1.), (6, -1.5)), rhs=.01,
        source_id='weighted affine', dependent_dof=12)
    case = load(2., 17.)
    case.add_nodal_load(3, [3., 0., 0., 0., 0., 0.])
    u, _ = solve_linear(m, case)
    expected = compute_constraint_force_diagnostics(m, u, case)
    with prepare_static_evaluator(m, recovery_config=RecoveryConfig(include_stresses=False)) as prepared:
        result = next(prepared.iter_batches([StaticCase('offset', case)]))
    np.testing.assert_allclose(result.displacements[:, 0], u, rtol=1e-12, atol=1e-14)
    forces = result.case_results[0]['constraint_forces']
    for key in ('residual_norm', 'reduced_residual_norm', 'nullspace_generalized_force_norm'):
        np.testing.assert_allclose(forces[key], expected[key], rtol=1e-12, atol=1e-14)
    for bucket in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
        assert forces[bucket].keys() == expected[bucket].keys()
        for node in expected[bucket]:
            np.testing.assert_allclose(forces[bucket][node], expected[bucket][node], rtol=1e-12, atol=1e-14)


def test_disconnected_free_bodies_require_individual_balance():
    m = model()
    m.clear_boundary_conditions()
    m.add_node(3, 3., 0., 0.)
    m.add_node(4, 4., 0., 0.)
    m.add_element(2, BeamElement(2, [3, 4], 'steel',
        dict(area=1., Iy=.001, Iz=.001, J=.001)))
    balanced = LoadCase('each body balanced')
    for node, force in ((1, -1.), (2, 1.), (3, 2.), (4, -2.)):
        balanced.add_nodal_load(node, [force, 0., 0., 0., 0., 0.])
    # Zero total force/moment, but opposite nonzero resultants on disconnected bodies.
    unbalanced = LoadCase('global cancellation is not sufficient')
    unbalanced.add_nodal_load(1, [1., 0., 0., 0., 0., 0.])
    unbalanced.add_nodal_load(3, [-1., 0., 0., 0., 0., 0.])
    reference, _ = solve_linear(m, balanced)
    with prepare_static_evaluator(m) as prepared:
        actual = next(prepared.iter_batches([StaticCase('physical', balanced)]))
        np.testing.assert_allclose(actual.displacements[:, 0], reference, rtol=1e-11, atol=1e-13)
        assert actual.case_results[0]['solver_info']['nullspace_rank'] == 12
        with pytest.raises(Exception, match='rigid-body'):
            list(prepared.iter_batches([StaticCase('unphysical', unbalanced)]))
