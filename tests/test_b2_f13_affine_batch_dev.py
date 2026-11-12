"""Private B2 1/10/100 RHS and complete-case cancellation feasibility."""

from dataclasses import replace
from fractions import Fraction
import math

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

import anyfesolver._b2_physical_linear_batch_dev as batch_module
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._b2_physical_linear_batch_dev import CompiledB2AffineBatch
from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import PhysicalB2OperatorCache
from anyfesolver.control import CancellationToken
from anyfesolver.fe_core import FEModel
from anyfesolver.linalg import FactorizationCache
from test_b2_physical_linear_oriented_graph import (
    _exact_linear_solve, _force_method_energy, _members, _solve,
)


def _plan():
    transform = sparse.eye(18, format="csr")[:, 6:]
    offset = np.zeros(18)
    return transform, offset


def _cases(count):
    cases = []
    for i in range(count):
        load = np.zeros(18)
        load[12:18] = (.4+.1*np.sin(i), -.9+.1*np.cos(i),
                       .7+.05*np.sin(i+.3), -.2+.03*np.cos(i+.5),
                       .3+.02*np.sin(i+.7), .1+.01*np.cos(i+.9))
        cases.append((f"case-{i:03}", load))
    return cases


def test_fast_residual_gate_rejects_many_lost_terms_and_boundary_margin():
    small_count = 2000
    entries = np.r_[1e10, np.full(small_count, 1e-8), -1e10]
    row = sparse.csr_matrix((entries,
                             (np.zeros(small_count+2, dtype=int),
                              np.arange(small_count+2))),
                            shape=(1, small_count+2))
    solution = np.ones((small_count+2, 1))
    rhs = np.zeros((1, 1))
    fast = row@solution-rhs
    scale = abs(row)@abs(solution)+abs(rhs)
    applied_scale = np.array([[1e4]])
    assert fast[0, 0] == 0.
    assert np.finfo(float).eps*scale[0, 0] < 1e-9*applied_scale[0, 0]
    accurate = batch_module._accurate_sparse_residual(row, solution, rhs)
    assert accurate[0, 0] == pytest.approx(2e-5, rel=1e-14)
    assert accurate[0, 0] > 1e-9*applied_scale[0, 0]
    assert not batch_module._fast_residual_within_margin(
        fast, scale, applied_scale, np.array([row.nnz], dtype=float))

    ordinary = sparse.csr_matrix([[2., -1.]])
    answer = np.ones((2, 1))
    target = np.ones((1, 1))
    ordinary_scale = abs(ordinary)@abs(answer)+abs(target)
    assert batch_module._fast_residual_within_margin(
        ordinary@answer-target, ordinary_scale, target,
        np.array([ordinary.nnz], dtype=float))
    near_limit = np.array([[1e-9-1e-18]])
    assert not batch_module._fast_residual_within_margin(
        near_limit, np.ones((1, 1)), np.ones((1, 1)),
        np.array([ordinary.nnz], dtype=float))


def test_backward_supplement_is_invariant_to_force_moment_unit_change():
    backward = np.array(((2., 3.), (.01, 5.), (.7, 2.)))
    equilibrium = np.array(((1., 4.), (3., 2.), (0., 1.)))
    units = np.array((1e3, 1e6, 1e6))[:, None]
    reference = batch_module._normalized_backward_limit(
        backward, equilibrium)
    np.testing.assert_allclose(batch_module._normalized_backward_limit(
        units*backward, units*equilibrium), units*reference,
                               rtol=2e-15, atol=0.)
    assert reference[2, 0] == 0.


def test_ordinary_batch_keeps_bounded_fast_residual_path(monkeypatch):
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18), max_chunk=4)
    accurate = batch_module._accurate_sparse_residual
    calls = []

    def counted(*args):
        calls.append(1)
        return accurate(*args)

    monkeypatch.setattr(batch_module, "_accurate_sparse_residual", counted)
    result = batch.solve_many(_cases(10))
    assert result.status == "completed" and len(result.cases) == 10
    assert result.factorization_count == 1
    assert calls == []
    batch.close()


@pytest.mark.parametrize("tip_component", (1, 2, 4, 5))
def test_pure_transverse_or_bending_load_has_physical_equilibrium_scale(
        tip_component):
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18))
    load = np.zeros(18)
    load[12+tip_component] = .73
    result = batch.solve_many(((f"component-{tip_component}", load),))
    assert result.status == "completed" and len(result.cases) == 1
    expected = spsolve(batch.full_stiffness[6:, 6:], load[6:])
    np.testing.assert_allclose(result.cases[0].values, expected[6:],
                               rtol=1e-10, atol=1e-10)
    batch.close()


@pytest.mark.parametrize("without_fma", (False, True))
def test_accurate_sparse_residual_preserves_cancelling_row(without_fma,
                                                           monkeypatch):
    if without_fma:
        monkeypatch.delattr(batch_module.math, "fma", raising=False)
    matrix = sparse.csr_matrix(np.array(((1e16, 1., -1e16),)))
    residual = batch_module._accurate_sparse_residual(
        matrix, np.ones((3, 1)), np.zeros((1, 1)))
    assert residual.shape == (1, 1) and residual[0, 0] == 1.
    with pytest.raises(RuntimeError, match="residual is not representable"):
        batch_module._accurate_sparse_residual(
            sparse.csr_matrix(np.array(((1e308, -1e308),))),
            np.full((2, 1), 2.), np.zeros((1, 1)))
    with pytest.raises(RuntimeError, match="residual is not representable"):
        batch_module._accurate_sparse_residual(
            sparse.csr_matrix(np.full((1, 3), 1e308)),
            np.ones((3, 1)), np.zeros((1, 1)))


@pytest.mark.parametrize("count", (1, 10, 100))
def test_ordered_selected_batch_matches_independent_sparse_solves(count):
    members, _ = _members()
    compiled = tuple(member.compile_local() for member in members)
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(compiled, transform, offset,
                                  np.arange(12, 18), max_chunk=8)
    stiffness = batch.full_stiffness
    transform.data[:] = 0.
    offset[:] = 1.
    cases = _cases(count)
    result = batch.solve_many(cases)
    assert result.status == "completed"
    assert len(result.cases) == count
    assert result.factorization_count == 1
    assert result.maximum_rhs_chunk <= 8
    assert [case.case_id for case in result.cases] == [identity for identity, _ in cases]
    independent_k = stiffness[6:, 6:]
    for (identity, load), selected in zip(cases, result.cases):
        oracle = spsolve(independent_k, load[6:])
        np.testing.assert_allclose(selected.values, oracle[6:12],
                                   rtol=3e-12, atol=3e-12)
        assert selected.case_id == identity
    assert batch.full_stiffness is not stiffness
    batch.close()
    with pytest.raises(RuntimeError, match="closed"):
        batch.solve_many(cases[:1])


def test_shared_operator_cache_preserves_batch_and_selected_recovery():
    members, _ = _members()
    members = (members[0], replace(
        members[1], reference_points_m=((2., 0., 0.), (2., 2., 0.))))
    assert members[0].section is members[1].section
    assert members[0].length_m == members[1].length_m == 2.
    plain = tuple(member.compile_local() for member in members)
    cache = PhysicalB2OperatorCache()
    reused = tuple(member.compile_local(operator_cache=cache)
                   for member in members)
    assert (cache.hits, cache.misses) == (1, 1)
    assert reused[0].operator is reused[1].operator
    transform, offset = _plan()
    plain_batch = CompiledB2AffineBatch(
        plain, transform, offset, np.arange(12, 18), max_chunk=4)
    reused_batch = CompiledB2AffineBatch(
        reused, transform, offset, np.arange(12, 18), max_chunk=4)
    cases = _cases(10)
    selection = {"member-0": (.25, .75)}
    plain_result = plain_batch.solve_many(cases, recover_members=selection)
    reused_result = reused_batch.solve_many(cases, recover_members=selection)
    assert plain_result.status == reused_result.status == "completed"
    assert plain_result.factorization_count == reused_result.factorization_count == 1
    np.testing.assert_array_equal(plain_batch.full_stiffness.toarray(),
                                  reused_batch.full_stiffness.toarray())
    for original, candidate in zip(plain_result.cases, reused_result.cases):
        assert original.case_id == candidate.case_id
        np.testing.assert_array_equal(original.values, candidate.values)
        np.testing.assert_array_equal(original.members[0].resultants_local,
                                      candidate.members[0].resultants_local)
        np.testing.assert_array_equal(original.members[0].strains_local,
                                      candidate.members[0].strains_local)
    plain_batch.close()
    reused_batch.close()


def test_callback_cannot_change_submitted_case_inventory():
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18), max_chunk=4)
    submitted = _cases(10)
    expected = tuple(submitted)

    def mutate_future_cases(identity):
        if identity == "case-000":
            submitted[5] = ("replacement", np.full(18, 42.))
            submitted.pop()

    result = batch.solve_many(submitted, on_case=mutate_future_cases)
    assert result.status == "completed"
    assert [case.case_id for case in result.cases] == [
        identity for identity, _ in expected]
    assert result.maximum_rhs_chunk == 4
    stiffness = batch.full_stiffness[6:, 6:]
    for (identity, load), selected in zip(expected, result.cases):
        oracle = spsolve(stiffness, load[6:])
        np.testing.assert_allclose(selected.values, oracle[6:12],
                                   rtol=3e-12, atol=3e-12)
        assert selected.case_id == identity
    batch.close()


def test_callback_mutation_of_future_load_is_rejected_before_next_chunk():
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18), max_chunk=4)
    submitted = _cases(10)
    completed = []

    def mutate_future_load(identity):
        completed.append(identity)
        if identity == "case-000":
            submitted[5][1][12] += 1.

    with pytest.raises(ValueError, match="changed after submission"):
        batch.solve_many(submitted, on_case=mutate_future_load)
    assert completed == [f"case-{i:03}" for i in range(4)]
    assert batch.solve_many(_cases(1)).status == "completed"
    batch.close()


def test_active_batch_cannot_be_closed_or_reentered_from_callback():
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18), max_chunk=4)

    def check_active_owner(identity):
        if identity == "case-000":
            with pytest.raises(RuntimeError, match="operation is active"):
                batch.close()
            with pytest.raises(RuntimeError, match="operation is active"):
                batch.solve_many(_cases(1))

    result = batch.solve_many(_cases(10), on_case=check_active_owner)
    assert result.status == "completed" and len(result.cases) == 10
    assert result.factorization_count == 1

    def fail_callback(_identity):
        raise LookupError("caller callback failed")

    with pytest.raises(LookupError, match="caller callback failed"):
        batch.solve_many(_cases(1), on_case=fail_callback)
    assert batch.solve_many(_cases(1)).status == "completed"
    batch.close()
    with pytest.raises(RuntimeError, match="closed"):
        batch.solve_many(_cases(1))


def test_cancellation_publishes_only_ordered_complete_cases():
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18), max_chunk=8)
    token = CancellationToken()
    stations = [.25]
    def stop_after_five(identity):
        stations[0] = .75
        if identity == "case-004":
            token.cancel("bounded prefix")
    result = batch.solve_many(_cases(100), cancellation_token=token,
                              on_case=stop_after_five,
                              recover_members={members[0].member_id: stations})
    assert result.status == "cancelled"
    assert [case.case_id for case in result.cases] == [
        f"case-{i:03}" for i in range(5)]
    assert result.factorization_count == 1
    assert result.maximum_rhs_chunk == 8
    assert all(case.members[0].stations == (.25,) for case in result.cases)
    batch.close()


def test_failed_factorization_does_not_return_offset_only_case():
    members, _ = _members()
    transform, offset = _plan()
    compiled = tuple(member.compile_local() for member in members)
    with pytest.raises(ValueError, match="selected DOFs"):
        CompiledB2AffineBatch(compiled, transform, offset,
                              np.array([12.5]))
    singular = transform.tolil()
    singular[:, 1] = 0.
    offset[12:18] = 42.
    with pytest.raises(RuntimeError, match="factorization failed"):
        CompiledB2AffineBatch(
            compiled,
            singular.tocsr(), offset, np.arange(12, 18), max_chunk=8)


def test_inaccurate_backend_solution_is_not_published(monkeypatch):
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18), max_chunk=8)
    monkeypatch.setattr(batch._handle, "solve_many",
                        lambda rhs: np.zeros_like(rhs))
    observed = []
    with pytest.raises(RuntimeError, match="solve residual failed"):
        batch.solve_many(_cases(1), on_case=observed.append)
    assert observed == []
    batch.close()


def test_selected_member_recovery_matches_independent_physical_states():
    members, _ = _members()
    compiled = tuple(member.compile_local() for member in members)
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        compiled, transform, offset, np.arange(12, 18), max_chunk=8)
    requested = {member.definition.member_id: (.2, .8)
                 for member in compiled}
    cases = _cases(100)
    result = batch.solve_many(cases, recover_members=requested)
    assert result.status == "completed" and len(result.cases) == 100
    assert result.maximum_rhs_chunk == 8
    stiffness = batch.full_stiffness
    for (identity, load), selected in zip(cases, result.cases):
        state = np.zeros(18)
        state[6:] = spsolve(stiffness[6:, 6:], load[6:])
        assert selected.case_id == identity
        assert len(selected.members) == 2
        for recovery, member in zip(selected.members, compiled):
            dofs = [6*batch.node_ids.index(node)+j
                    for node in member.definition.node_ids for j in range(6)]
            oracle = member.operator.evaluate(
                member.local_dof_map()@state[dofs], stations=(.2, .8))
            assert recovery.member_id == member.definition.member_id
            np.testing.assert_allclose(recovery.resultants_local,
                                       oracle.resultants, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(recovery.strains_local,
                                       oracle.strains, rtol=1e-10, atol=1e-12)
    assert all(len(case.members) == 2 for case in result.cases)
    batch.close()


def test_selected_recovery_rejects_invalid_requests_before_case_publication():
    members, _ = _members()
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in members),
        transform, offset, np.arange(12, 18))
    observed = []
    with pytest.raises(ValueError, match="not compiled"):
        batch.solve_many(_cases(1), recover_members={"unknown": (.5,)},
                         on_case=observed.append)
    with pytest.raises(ValueError, match="recovery mapping"):
        batch.solve_many(_cases(1), recover_members=(.5,),
                         on_case=observed.append)
    with pytest.raises(ValueError, match="recovery stations"):
        batch.solve_many(_cases(1),
                         recover_members={members[0].member_id: (1.1,)},
                         on_case=observed.append)
    assert observed == []
    batch.close()


def test_cancellation_during_selected_recovery_discards_incomplete_case(monkeypatch):
    members, _ = _members()
    compiled = tuple(member.compile_local() for member in members)
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        compiled, transform, offset, np.arange(12, 18))
    token = CancellationToken()
    original = type(compiled[0].operator).recover_sections

    def cancel_during_recovery(operator, displacement, *, stations):
        response = original(operator, displacement, stations=stations)
        token.cancel("during selected recovery")
        return response

    monkeypatch.setattr(type(compiled[0].operator), "recover_sections",
                        cancel_during_recovery)
    observed = []
    result = batch.solve_many(
        _cases(10), cancellation_token=token,
        recover_members={compiled[0].definition.member_id: (.5,)},
        on_case=observed.append)
    assert result.status == "cancelled"
    assert result.cases == () and observed == []
    batch.close()


@pytest.mark.parametrize("radius_m", (.005, .02, .08))
def test_scaled_batch_preserves_coherent_circular_elbow_work(radius_m):
    area = math.pi*radius_m**2
    inertia = math.pi*radius_m**4/4
    section = LinearSection.isotropic(
        E=210e9, nu=.3, area=area, Iy=inertia, Iz=inertia,
        J=2*inertia, ky=.9, kz=.9)
    members, _ = _members()
    physical = tuple(replace(member, section_id=f"circle-{radius_m}",
                             section=section) for member in members)
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in physical),
        transform, offset, np.arange(12, 18), max_chunk=8)
    tip = np.array((.1, -.08, .06, .01, -.012, .008))
    load = np.zeros(18)
    load[12:] = tip
    result = batch.solve_many((("mixed", load),))
    assert result.status == "completed" and result.factorization_count == 1
    selected = np.asarray(result.cases[0].values)
    reduced = batch.full_stiffness[6:, 6:].toarray()
    exact = np.asarray(_exact_linear_solve(reduced, load[6:]),
                       dtype=float)
    np.testing.assert_allclose(selected, exact[6:12], rtol=1e-9,
                               atol=1e-13)
    physical_work = float(_force_method_energy(physical, tip))
    assert abs(.5*tip@selected-physical_work)/physical_work <= 1e-9
    batch.close()


def test_moderately_weak_generalized_elbow_preserves_physical_balance():
    exponent = 20
    section_matrix = np.diag([12., 2.**-exponent,
                              2.**(1-exponent), 4., 3., 5.])
    section_matrix[0, 4] = section_matrix[4, 0] = .5
    section_matrix[1, 2] = section_matrix[2, 1] = 2.**(-exponent-2)
    section_matrix[3, 5] = section_matrix[5, 3] = .25
    section = LinearSection.generalized(section_matrix)
    members, _ = _members()
    physical = tuple(replace(member, section_id="moderately-weak",
                             section=section) for member in members)
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in physical),
        transform, offset, np.arange(6, 18), max_chunk=1)
    wrench = np.array((.7, -.8, .5, .2, -.3, .4))
    load = np.zeros(18)
    load[12:] = wrench
    result = batch.solve_many((("moderate-mixed", load),))
    assert result.status == "completed" and result.factorization_count == 1
    solution = np.zeros(18)
    solution[6:] = result.cases[0].values
    matrix = batch.full_stiffness.tocsr()
    exact_defect = []
    for row in range(6, 18):
        start, stop = matrix.indptr[row:row+2]
        exact = sum((Fraction(float(a))*Fraction(float(solution[col]))
                     for a, col in zip(matrix.data[start:stop],
                                       matrix.indices[start:stop])),
                    Fraction(0))-Fraction(float(load[row]))
        exact_defect.append(float(exact))
    force_scale = np.max(np.abs(wrench[:3]))
    moment_scale = max(np.max(np.abs(wrench[3:])),
                       max(member.length_m for member in physical)*force_scale)
    assert max(abs(exact_defect[row]) for row in (0, 1, 2, 6, 7, 8))/force_scale <= 1e-9
    assert max(abs(exact_defect[row]) for row in (3, 4, 5, 9, 10, 11))/moment_scale <= 1e-9
    batch.close()


def test_weak_generalized_elbow_rejects_unbalanced_binary64_solution(monkeypatch):
    section_matrix = np.diag([12., 2.**-40, 2.**-39, 4., 3., 5.])
    section_matrix[0, 4] = section_matrix[4, 0] = .5
    section_matrix[1, 2] = section_matrix[2, 1] = 2.**-42
    section_matrix[3, 5] = section_matrix[5, 3] = .25
    section = LinearSection.generalized(section_matrix)
    members, _ = _members()
    physical = tuple(replace(member, section_id="weak-generalized",
                             section=section) for member in members)
    transform, offset = _plan()
    batch = CompiledB2AffineBatch(
        tuple(member.compile_local() for member in physical),
        transform, offset, np.arange(12, 18), max_chunk=2)
    wrench = np.array((.7, -.8, .5, .2, -.3, .4))
    load = np.zeros(18)
    load[12:] = wrench
    solve_columns = []
    original_solve = batch._handle.solve_many

    def capture_solve(rhs):
        solved = np.asarray(original_solve(rhs), dtype=float)
        solve_columns.append(solved.copy())
        return solved

    monkeypatch.setattr(batch._handle, "solve_many", capture_solve)
    observed = []
    with pytest.raises(RuntimeError, match="batch solve residual failed"):
        batch.solve_many((("weak-mixed", load),), on_case=observed.append)
    assert observed == []
    assert batch._handle.factorization_count == 1
    assert 2 <= len(solve_columns) <= 3
    reduced_solution = batch._scaling[:, None]*solve_columns[0]
    for correction in solve_columns[1:]:
        reduced_solution -= batch._scaling[:, None]*correction
    full_solution = np.asarray(batch._T@reduced_solution).reshape(-1)+batch._u0
    full_stiffness = batch.full_stiffness.tocsr()
    def exact_row_defects(solution):
        defects = []
        for row in range(18):
            start, stop = full_stiffness.indptr[row:row+2]
            exact = sum((Fraction(float(a))*Fraction(float(solution[col]))
                         for a, col in zip(full_stiffness.data[start:stop],
                                           full_stiffness.indices[start:stop])),
                        Fraction(0))-Fraction(float(load[row]))
            defects.append(float(exact))
        return defects

    exact_defect = exact_row_defects(full_solution)
    force_scale = np.max(np.abs(wrench[:3]))
    moment_scale = max(np.max(np.abs(wrench[3:])),
                       max(member.length_m for member in physical)*force_scale)
    physical_imbalance = max(
        max(abs(exact_defect[row]) for row in (6, 7, 8, 12, 13, 14))/force_scale,
        max(abs(exact_defect[row]) for row in (9, 10, 11, 15, 16, 17))/moment_scale)
    assert physical_imbalance > 1e-9
    reduced = batch.full_stiffness[6:, 6:].toarray()
    exact = np.asarray(_exact_linear_solve(reduced, load[6:]), dtype=float)
    np.testing.assert_allclose(full_solution[12:18], exact[6:12],
                               rtol=1e-8, atol=1e-8)
    # Even exact algebra on the rounded matrix loses physical balance once
    # its very large solution is stored as ordinary binary64 coordinates.
    exact_as_float = np.zeros(18)
    exact_as_float[6:] = exact
    exact_float_defect = exact_row_defects(exact_as_float)
    assert max(abs(exact_float_defect[row]) for row in
               (6, 7, 8, 12, 13, 14))/force_scale > 1e-9
    expected_work = float(_force_method_energy(physical, wrench))
    assert abs(.5*wrench@exact[6:12]-expected_work)/expected_work <= 1e-8
    batch.close()


def test_existing_auditor_builds_b2_affine_map_and_tracks_its_dependencies():
    members, points = _members()
    _, _, stiffness = _solve(members, np.zeros(18))
    model = FEModel("bounded-B2-audited-affine")
    for node_id, point in enumerate(points):
        model.add_node(node_id, *point)
    model.add_boundary_condition(BoundaryCondition(
        "root", [0], {name: 0. for name in
                       ("ux", "uy", "uz", "rx", "ry", "rz")}))

    def equation(coefficient, value):
        model.constraint_equations.clear()
        model.add_constraint_equation(
            terms=((12, 1.), (7, -coefficient)), rhs=value,
            dependent_dof=12, source_id="tip-x-to-mid-y")
        return build_constraint_transformation(
            stiffness, np.zeros(18), model)

    _, _, first_t, first_u0, independent, _ = equation(.25, .001)
    assert first_t.shape == (18, 11) and independent.size == 11
    load = _cases(1)[0][1]
    first_k = (first_t.T@stiffness@first_t).tocsc()
    reduced_load = first_t.T@(load-stiffness@first_u0)
    state = first_t@spsolve(first_k, reduced_load)+first_u0
    assert abs(state[12]-.25*state[7]-.001) <= 1e-13
    np.testing.assert_allclose(first_t.T@(stiffness@state-load), 0.,
                               atol=1e-11)
    compiled = tuple(member.compile_local() for member in members)
    cache = FactorizationCache(name="bounded-b2-affine", max_entries=2)
    batch = CompiledB2AffineBatch(
        compiled, first_t, first_u0, np.arange(12, 18),
        max_chunk=8, factorization_cache=cache)
    assert (cache.hits, cache.misses) == (0, 1)
    selected = batch.solve_many((("audited", load),)).cases[0].values
    np.testing.assert_allclose(selected, state[12:18],
                               rtol=1e-11, atol=1e-11)
    first_handle = batch._handle
    batch.close()

    _, _, value_t, value_u0, value_independent, _ = equation(.25, .002)
    np.testing.assert_array_equal(value_t.toarray(), first_t.toarray())
    np.testing.assert_array_equal(value_independent, independent)
    assert value_u0[12] == .002 and first_u0[12] == .001
    value_batch = CompiledB2AffineBatch(
        compiled, value_t, value_u0, np.arange(12, 18),
        factorization_cache=cache)
    assert (cache.hits, cache.misses) == (1, 1)
    assert value_batch._handle is first_handle
    value_state = value_t@spsolve(
        (value_t.T@stiffness@value_t).tocsc(),
        value_t.T@(load-stiffness@value_u0))+value_u0
    value_u0[12] = 9.
    value_selected = value_batch.solve_many((("new-value", load),)).cases[0].values
    np.testing.assert_allclose(value_selected, value_state[12:18],
                               rtol=1e-11, atol=1e-11)
    assert not np.array_equal(value_selected, selected)
    value_batch.close()

    _, _, structural_t, structural_u0, _, _ = equation(.5, .002)
    assert not np.array_equal(structural_t.toarray(), first_t.toarray())
    structural_batch = CompiledB2AffineBatch(
        compiled, structural_t, structural_u0, np.arange(12, 18),
        factorization_cache=cache)
    assert (cache.hits, cache.misses) == (1, 2)
    assert structural_batch._handle is not first_handle
    structural_state = structural_t@spsolve(
        (structural_t.T@stiffness@structural_t).tocsc(),
        structural_t.T@(load-stiffness@structural_u0))+structural_u0
    np.testing.assert_allclose(
        structural_batch.solve_many((("new-structure", load),)).cases[0].values,
        structural_state[12:18], rtol=1e-11, atol=1e-11)
    structural_batch.close()

    model.add_boundary_condition(BoundaryCondition("conflict", [0],
                                                   {"ux": 1.}))
    with pytest.raises(ValueError, match="Invalid constraint system"):
        build_constraint_transformation(stiffness, np.zeros(18), model)
