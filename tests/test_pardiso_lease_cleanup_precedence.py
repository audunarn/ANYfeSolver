"""Cleanup precedence and holder audit for PARDISO leases (Codex review round 2).

One rule decides which exception propagates when a holder ends a lease or
closes a lease-owning object on exit (``linalg._cleanup_outcome``, the rule
of ``FactorizationLease.__exit__``): a cleanup interrupt never replaces a
propagating interrupt (it is noted on it) but replaces a propagating
ordinary exception (chained); a cleanup ``Exception`` never replaces a
propagating exception; ``GeneratorExit`` counts as nothing propagating.
Every holder uses ``linalg.CleanupScope`` / ``close_propagating`` so the
primary is the exception actually propagating, never ``None`` from a
``finally``.  The general-contact holders that solve one factorization for
several right-hand sides are leased.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.linalg as linalg
from anyfesolver.linalg import FactorizationCache, MatrixClass, PyPardisoSolverBackend

from test_pardiso_leases import _disturb_default, _forced_one_slot_default
from test_pardiso_perturbed_pivots import _beam_chain, _install_fake, requires_pypardiso


class _PardisoDefault:
    """Every default factorization on one one-slot PARDISO backend (no SuperLU policy)."""

    def __init__(self, monkeypatch) -> None:
        self.pardiso_backend = PyPardisoSolverBackend(max_pattern_slots=1)
        monkeypatch.setattr(linalg, "DEFAULT_BACKEND", self.pardiso_backend)

    def release_pattern_slots(self) -> None:
        self.pardiso_backend.release_pattern_slots()


def _notes(exc: BaseException) -> str:
    return " ".join(getattr(exc, "__notes__", []))


def _interrupting_lease_end(monkeypatch) -> list:
    """Every pinned lease end releases its slot normally, then reports an interrupt."""

    raised: list = []
    original = linalg._end_slot_lease

    def ending(slot):
        original(slot)
        interrupt = KeyboardInterrupt("lease end")
        raised.append(interrupt)
        return interrupt

    monkeypatch.setattr(linalg, "_end_slot_lease", ending)
    return raised


def _interrupting_lease_close(monkeypatch) -> list:
    """Every explicit ``FactorizationLease.close`` ends the lease, then raises an interrupt."""

    raised: list = []
    original = linalg.FactorizationLease.close

    def close(lease):
        original(lease)
        interrupt = KeyboardInterrupt("lease close")
        raised.append(interrupt)
        raise interrupt

    monkeypatch.setattr(linalg.FactorizationLease, "close", close)
    return raised


# --------------------------------------------------------------------------
# The helper: one precedence rule, the primary is what actually propagates.
# --------------------------------------------------------------------------


def _scope_outcome(primary, cleanup_error):
    def failing():
        raise cleanup_error

    try:
        with linalg.CleanupScope() as scope:
            scope.callback(failing, "test cleanup")
            if primary is not None:
                raise primary
    except BaseException as error:  # noqa: BLE001 - the outcome is the subject
        return error
    return None


def _lease_outcome(monkeypatch, primary, interrupt):
    """``FactorizationLease.__exit__`` on a pinned (fake) PARDISO lease."""

    _install_fake(monkeypatch, lambda matrix: 0)
    backend = PyPardisoSolverBackend()
    lease = linalg.leased_factorization(sparse.csr_matrix(np.diag([1.0, 2.0])), MatrixClass.SPD,
                                        backend=backend)
    assert lease.pinned
    original = linalg._end_slot_lease
    lease._finalizer.detach()
    lease._finalizer = linalg.weakref.finalize(lease, lambda slot: (original(slot), interrupt)[1],
                                               lease.handle._solver._slot)
    try:
        with lease:
            if primary is not None:
                raise primary
    except BaseException as error:  # noqa: BLE001
        return error
    return None


@pytest.mark.parametrize("primary_type", [None, ValueError, SystemExit, KeyboardInterrupt, GeneratorExit])
def test_cleanup_scope_precedence_is_exactly_the_lease_exit_rule(monkeypatch, primary_type) -> None:
    for build in ("scope", "lease"):
        primary = None if primary_type is None else primary_type("primary")
        interrupt = KeyboardInterrupt("cleanup")
        if build == "scope":
            outcome = _scope_outcome(primary, interrupt)
        else:
            outcome = _lease_outcome(monkeypatch, primary, interrupt)
        if primary_type in (None, ValueError, GeneratorExit):
            assert outcome is interrupt, (build, primary_type)
            if primary is not None:
                assert outcome.__context__ is primary
        else:
            assert outcome is primary, (build, primary_type)
            assert "KeyboardInterrupt" in _notes(primary)


def test_cleanup_scope_exception_never_replaces_a_propagating_exception() -> None:
    primary = SystemExit(2)
    assert _scope_outcome(primary, RuntimeError("cleanup failed")) is primary
    assert "cleanup failed" in _notes(primary)
    error = RuntimeError("only failure")
    assert _scope_outcome(None, error) is error


def test_cleanup_scope_runs_every_cleanup_with_the_current_primary() -> None:
    seen = []
    first_interrupt = KeyboardInterrupt("first")

    class Resource:
        def __exit__(self, exc_type, exc, tb):
            seen.append(exc)

    def raising():
        raise first_interrupt

    primary = ValueError("body")
    with pytest.raises(KeyboardInterrupt) as raised:
        with linalg.CleanupScope() as scope:
            scope.push(Resource())
            scope.callback(raising)  # runs first (last registered)
            raise primary
    assert raised.value is first_interrupt and raised.value.__context__ is primary
    assert seen == [first_interrupt]  # the earlier resource saw the replacing interrupt


def test_close_propagating_inside_an_unrelated_except_block_uses_its_explicit_primary() -> None:
    def interrupting():
        raise KeyboardInterrupt("cleanup")

    with pytest.raises(KeyboardInterrupt):
        try:
            raise ValueError("handled")
        except ValueError:
            linalg.close_propagating(interrupting, None)  # nothing propagates
    primary = SystemExit(1)
    linalg.close_propagating(interrupting, primary)
    assert "KeyboardInterrupt" in _notes(primary)


# --------------------------------------------------------------------------
# M1: explicit-close holders keep a propagating interrupt (SystemExit).
# --------------------------------------------------------------------------


class _Caught:
    """Catch any exit (an interrupt must never abort the test session itself)."""

    value = None

    def __enter__(self) -> "_Caught":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.value = exc
        return True


def _exits_with_noted_interrupt(raised: _Caught, primary: BaseException, interrupts: list) -> None:
    assert raised.value is primary, repr(raised.value)
    assert interrupts, "the lease end was not exercised"
    assert "KeyboardInterrupt" in _notes(primary)


@requires_pypardiso
def test_newmark_lease_end_interrupt_never_replaces_a_propagating_system_exit(monkeypatch) -> None:
    """On daf1877bd the cleanup KeyboardInterrupt replaced the callback's SystemExit."""

    from anyfesolver import LoadCase
    from anyfesolver.dynamics import TransientConfig, solve_transient_newmark

    backend = _forced_one_slot_default(monkeypatch)
    interrupts = _interrupting_lease_end(monkeypatch)
    primary = SystemExit(7)
    calls = {"n": 0}

    def progress(*_args, **_kwargs) -> None:
        calls["n"] += 1
        if backend.pardiso_backend.leased_pattern_slots:
            raise primary

    load = LoadCase("tip")
    load.add_nodal_load(3, [1.0e3, -2.0e3, 5.0e2, 0.0, 10.0, -20.0])
    try:
        with _Caught() as raised:
            solve_transient_newmark(_beam_chain(2), TransientConfig(dt=0.001, t_end=0.002),
                                    base_load_case=load, progress_callback=progress)
        _exits_with_noted_interrupt(raised, primary, interrupts)
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_linear_impact_lease_end_interrupt_never_replaces_a_propagating_system_exit(monkeypatch) -> None:
    from anyfesolver.contact import (
        RigidSphereImpact, SphereContactConfig, _verification_contact_panel,
        solve_transient_sphere_impact,
    )
    from anyfesolver.dynamics import TransientConfig

    backend = _PardisoDefault(monkeypatch)
    interrupts = _interrupting_lease_end(monkeypatch)
    primary = SystemExit(8)
    calls = {"n": 0}

    def progress(*_args, **_kwargs) -> None:
        calls["n"] += 1
        if backend.pardiso_backend.leased_pattern_slots:
            raise primary

    try:
        with _Caught() as raised:
            solve_transient_sphere_impact(
                _verification_contact_panel(),
                TransientConfig(dt=0.0025, t_end=0.02, output_nodes=[1]),
                RigidSphereImpact("lease_hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25),
                                  travel_direction=(0.0, 0.0, -1.0), speed=2.0),
                SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
                progress_callback=progress,
            )
        _exits_with_noted_interrupt(raised, primary, interrupts)
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_nonlinear_impact_lease_end_interrupt_never_replaces_a_propagating_system_exit(monkeypatch) -> None:
    import anyfesolver as fs
    from anyfesolver.contact import _verification_contact_panel

    backend = linalg.AutoSparseSolverBackend(  # Auto: the nonlinear route installs one
        pardiso_backend=PyPardisoSolverBackend(max_pattern_slots=1), pypardiso_min_dimension=1,
        pypardiso_min_nnz=1, pypardiso_warm_min_dimension=1, pypardiso_warm_min_nnz=1,
    )
    monkeypatch.setattr(linalg, "DEFAULT_BACKEND", backend)
    interrupts = _interrupting_lease_end(monkeypatch)
    primary = SystemExit(9)
    calls = {"n": 0}

    def status(_message: str) -> None:
        calls["n"] += 1
        if backend.pardiso_backend.leased_pattern_slots:
            raise primary

    try:
        with _Caught() as raised:
            fs.solve_transient_sphere_impact(
                _verification_contact_panel(),
                fs.TransientConfig(dt=0.0025, t_end=0.04),
                fs.RigidSphereImpact("lease_reuse", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.12),
                                     travel_direction=(0.0, 0.0, -1.0), speed=2.0),
                fs.SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
                nonlinear_config=fs.NonlinearTransientConfig(
                    enabled=True, max_iterations=15, max_cutbacks=4, tangent_reuse_iterations=2,
                ),
                status_callback=status,
            )
        _exits_with_noted_interrupt(raised, primary, interrupts)
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_descriptor_spectrum_lease_end_interrupt_never_replaces_a_propagating_system_exit(monkeypatch) -> None:
    from scipy.sparse import linalg as sparse_linalg

    from anyfesolver.algebraic_dynamics import solve_descriptor_spectrum
    from test_e4_pl_s3_descriptor_spectrum import _repeated_descriptor

    backend = _PardisoDefault(monkeypatch)
    interrupts = _interrupting_lease_end(monkeypatch)
    primary = SystemExit(10)

    original = sparse_linalg.eigsh

    def eigsh(*args, **kwargs):
        if backend.pardiso_backend.leased_pattern_slots:  # inside the leased target eigensolve
            raise primary
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse_linalg, "eigsh", eigsh)
    stiffness, mass = _repeated_descriptor()
    try:
        with _Caught() as raised:
            solve_descriptor_spectrum(stiffness, mass, num_modes=4, dense_size_limit=1, algebraic_nullity=2,
                                      static_condensation_limit=1, target_shift=2.5)
        _exits_with_noted_interrupt(raised, primary, interrupts)
    finally:
        backend.release_pattern_slots()


def test_analysis_session_with_exit_keeps_a_propagating_system_exit(monkeypatch) -> None:
    from anyfesolver.analysis_session import AnalysisSession

    from test_static_batch import _bar

    interrupts = _interrupting_lease_close(monkeypatch)
    primary = SystemExit(11)
    model = _bar()
    model.apply_boundary_conditions()
    with _Caught() as raised:
        with AnalysisSession(model) as session:
            assert session.static_operator_plan().handle is not None
            raise primary
    _exits_with_noted_interrupt(raised, primary, interrupts)
    assert session._closed  # the interrupt at lease end leaves the session closed


def test_prepared_static_evaluator_with_exit_keeps_a_propagating_system_exit(monkeypatch) -> None:
    from anyfesolver.prepared_static import prepare_static_evaluator

    from test_static_batch import _bar

    interrupts: list = []

    def release(_backend):
        interrupts.append(KeyboardInterrupt("release"))
        raise interrupts[-1]

    primary = SystemExit(12)
    with _Caught() as raised:
        with prepare_static_evaluator(_bar()) as evaluator:
            monkeypatch.setattr(linalg.AutoSparseSolverBackend, "release_pattern_slots", release)
            raise primary
    _exits_with_noted_interrupt(raised, primary, interrupts)
    assert evaluator.closed


# --------------------------------------------------------------------------
# M2: an ordinary case failure never swallows a cleanup interrupt.
# --------------------------------------------------------------------------


def test_static_stream_lease_cleanup_interrupt_replaces_an_ordinary_case_failure(monkeypatch) -> None:
    """On daf1877bd the interrupt became a note and the case error was re-raised."""

    from anyfesolver.static_batch import StaticBatchError, StaticCase, iter_linear_batches

    from test_static_batch import _bar, _case

    interrupts = _interrupting_lease_close(monkeypatch)
    with pytest.raises(KeyboardInterrupt) as raised:
        list(iter_linear_batches(_bar(), [_case(0), StaticCase("invalid", None, float("inf"))], batch_size=1))
    assert interrupts and raised.value is interrupts[0]
    assert isinstance(raised.value.__context__, StaticBatchError)


def test_solve_linear_many_session_cleanup_interrupt_replaces_an_ordinary_failure(monkeypatch) -> None:
    import anyfesolver.static_batch as static_batch
    from anyfesolver.analysis_session import AnalysisSession
    from anyfesolver.assembly import solve_linear_many

    from test_static_batch import _bar

    interrupts: list = []
    original = AnalysisSession.close

    def close(session):
        original(session)
        interrupts.append(KeyboardInterrupt("session close"))
        raise interrupts[-1]

    def failing(*_args, **_kwargs):
        raise ValueError("case failure")

    monkeypatch.setattr(AnalysisSession, "close", close)
    monkeypatch.setattr(static_batch, "iter_linear_batches", failing)
    with pytest.raises(KeyboardInterrupt) as raised:
        solve_linear_many(_bar(), [None], constraint_mode="nullspace")
    assert interrupts and raised.value is interrupts[0]
    assert isinstance(raised.value.__context__, ValueError)


# --------------------------------------------------------------------------
# P2: general-contact holders solving one factorization several times.
# --------------------------------------------------------------------------


def _disturb_after_first_solve(monkeypatch, signature_part: str) -> dict:
    """After the first solve of a matching handle, evict it from the 1-slot LRU."""

    state = {"solves": 0, "disturbed": False}
    original = linalg.FactorizationHandle.solve

    def solve(handle, rhs):
        answer = original(handle, rhs)
        if signature_part in str(handle.signature):
            state["solves"] += 1
            if state["solves"] == 1:
                _disturb_default()
                state["disturbed"] = True
        return answer

    monkeypatch.setattr(linalg.FactorizationHandle, "solve", solve)
    return state


def _predictor():
    from anyfesolver.general_contact.static_path import _elastic_displacement_force_predictor

    stiffness = sparse.csr_matrix(np.array([[4.0, -1.0, 0.0], [-1.0, 5.0, -2.0], [0.0, -2.0, 6.0]]))
    return _elastic_displacement_force_predictor(
        stiffness, sparse.eye(3, format="csr"), np.array([1.0, 0.0, 0.0]),
        np.array([[1.0e-3], [2.0e-3], [-5.0e-4]]),
    )


@requires_pypardiso
def test_static_path_force_predictor_is_leased_across_waypoints(monkeypatch) -> None:
    backend = _forced_one_slot_default(monkeypatch)
    try:
        expected = _predictor()
        backend.release_pattern_slots()
        state = _disturb_after_first_solve(monkeypatch, "elastic_displacement_force_predictor")
        observed = _predictor()
        assert state["disturbed"] and state["solves"] == 3
        assert observed == expected
        assert backend.pardiso_backend.leased_pattern_slots == 0
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_initial_contact_seed_constrained_operator_is_leased_across_solves(monkeypatch) -> None:
    from test_general_contact_initial_seed_oracle import _blocks, _propose

    backend = _forced_one_slot_default(monkeypatch)
    try:
        expected = _propose(_blocks(2.0, 3.0, 0.01))
        assert not expected.refused
        backend.release_pattern_slots()
        state = _disturb_after_first_solve(monkeypatch, "oracle:constrained")
        observed = _propose(_blocks(2.0, 3.0, 0.01))
        assert state["disturbed"] and state["solves"] == 2
        assert not observed.refused, getattr(observed, "reason", observed)
        for name in ("q", "reactions", "nodal_forces"):
            assert np.asarray(getattr(observed, name)).tobytes() == np.asarray(getattr(expected, name)).tobytes(), name
        assert backend.pardiso_backend.leased_pattern_slots == 0
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_linearized_pivot_pressure_response_is_leased_across_columns(monkeypatch) -> None:
    from dataclasses import replace

    from anyfesolver.general_contact.linearized_pivot import _constrained_gap_response
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3

    def response():
        model, definition, settings = _two_q4_two_s3()
        for node in (7, 8, 9, 10):
            model.mesh.get_node(node).x -= 0.5
        settings = replace(
            settings, general_s3_differentiation="jet2_q4_s3_dev",
            q4_source_support_gradient_tolerance=1e-5, q4_source_support_hessian_tolerance=1e-5,
            q4_source_weighted_gap_gradient_tolerance=1e-5, q4_source_weighted_gap_hessian_tolerance=1e-5,
            maximum_q4_source_integration_triangles=128,
        )
        manager = GeneralS3PatchManager(model, definition, settings)
        base_u = np.zeros(manager.n_dof)
        for node in range(1, 7):
            base_u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
        manager.set_normal_multipliers(np.full(6, 10.0))
        free = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in (1, 5)]
        transform = sparse.eye(manager.n_dof, format="csr")[:, free]
        manager.restrict_tangent_to_reduced_variations(transform)
        _force, contact_tangent, state = manager.evaluate(base_u)
        supported = np.flatnonzero(np.asarray(state.reference_support_m2) > 0.0)
        active = np.flatnonzero(np.asarray(state.numerical_pressure_pa) > 0.0)
        spring_full = transform @ sparse.csr_matrix(np.diag([2.0e4, 3.0e4])) @ transform.T
        diagnostics: dict = {}
        made = _constrained_gap_response(
            manager, base_u, contact_tangent + spring_full, transform,
            np.zeros((0, len(free))), supported, active, difference_step=2e-6,
            diagnostics=diagnostics,
        )
        return made, len(active), diagnostics

    backend = _forced_one_slot_default(monkeypatch)
    try:
        expected, columns, _ = response()
        assert expected is not None and columns >= 2
        backend.release_pattern_slots()
        state = _disturb_after_first_solve(monkeypatch, "contact:pressure_response")
        observed, _, diagnostics = response()
        assert state["disturbed"] and state["solves"] == columns
        assert observed is not None, diagnostics.get("schur_response_failure")
        assert observed.tobytes() == expected.tobytes()
        assert backend.pardiso_backend.leased_pattern_slots == 0
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_linearized_pivot_stale_pin_on_both_attempts_is_a_failure_not_an_abort(monkeypatch) -> None:
    """D2 round 3: lease_from() raising PardisoStaleHandleError (both pin attempts stale
    under concurrent invalidation) takes the existing None/schur_response_failure path."""
    from dataclasses import replace

    from anyfesolver.general_contact.linearized_pivot import _constrained_gap_response
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3

    model, definition, settings = _two_q4_two_s3()
    for node in (7, 8, 9, 10):
        model.mesh.get_node(node).x -= 0.5
    settings = replace(
        settings, general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5, q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5, q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    base_u = np.zeros(manager.n_dof)
    for node in range(1, 7):
        base_u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    manager.set_normal_multipliers(np.full(6, 10.0))
    free = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in (1, 5)]
    transform = sparse.eye(manager.n_dof, format="csr")[:, free]
    manager.restrict_tangent_to_reduced_variations(transform)
    _force, contact_tangent, state = manager.evaluate(base_u)
    supported = np.flatnonzero(np.asarray(state.reference_support_m2) > 0.0)
    active = np.flatnonzero(np.asarray(state.numerical_pressure_pa) > 0.0)
    spring_full = transform @ sparse.csr_matrix(np.diag([2.0e4, 3.0e4])) @ transform.T

    pins = {"count": 0}

    def always_stale(_handle):
        pins["count"] += 1
        raise linalg.PardisoStaleHandleError()

    monkeypatch.setattr(linalg, "lease_handle", always_stale)
    diagnostics: dict = {}
    made = _constrained_gap_response(
        manager, base_u, contact_tangent + spring_full, transform,
        np.zeros((0, len(free))), supported, active, difference_step=2e-6,
        diagnostics=diagnostics,
    )
    assert pins["count"] == 2
    assert made is None
    assert diagnostics.get("schur_response_failure")


# --------------------------------------------------------------------------
# Low: the cache-miss path reads one state snapshot.
# --------------------------------------------------------------------------


def test_cache_miss_reads_one_state_snapshot_across_a_concurrent_invalidation(monkeypatch) -> None:
    """A fresh handle caught between the two writes of a stale mark is stale, not failed.

    The hook stops a concurrent ``_mark_stale`` after its first write (status)
    and before its second (stale kind); on daf1877bd the cache read status
    and kind separately and counted a factorization failure.
    """

    _install_fake(monkeypatch, lambda matrix: 0)
    pardiso = PyPardisoSolverBackend()

    class HalfInvalidating:
        name = "half_invalidating"

        def factorize(self, matrix, matrix_class, *, signature=None, options=None):
            handle = pardiso.factorize(matrix, matrix_class, signature=signature, options=options)
            slot = handle._solver._slot
            slot.generation += 1  # invalidated by "another thread" ...
            handle.status = "failed"  # ... whose _mark_stale has written status only
            return handle

    cache = FactorizationCache(name="snapshot", backend=HalfInvalidating())
    handle = cache.factorize(sparse.csr_matrix(np.diag([1.0, 2.0])), MatrixClass.SPD)
    assert cache.factorization_failures == 0
    assert cache.diagnostics()["entries"] == 0
    assert handle.metadata.get("pardiso_failure_kind") == linalg.PARDISO_STALE_HANDLE
    pardiso.release_pattern_slots()
