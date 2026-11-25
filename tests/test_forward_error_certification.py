"""Static linear-solve certification rules A and B.

Rule A (factorization known-solution check, ``linalg``): every new numerical
factorization solves ``A z`` for a fixed z and is refused when
``max|x - z| / max|z|`` exceeds 1e-3 (mode "refuse", the default) or only
records it (mode "report", used by Newton/continuation tangents).

Rule B (per-RHS refinement correction): every certified linear solution is
refused when ``max|solve(A y - b)| / max|y|`` exceeds 1e-3.

The normwise backward error alone certified huge wrong solutions of
(rounding-)singular systems (it is tiny exactly when |x| is huge).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.assembly as assembly
import anyfesolver.linalg as linalg
from anyfesolver import (AnalysisSession, BoundaryCondition, LoadCase, StaticBatchError,
    StaticCase, iter_linear_batches, prepare_static_evaluator, solve_linear)
from anyfesolver.assembly import solve_linear_many
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.linalg import (AutoSparseSolverBackend, FactorizationCache, MatrixClass,
    PyPardisoSolverBackend, SparseSolverBackend)
from anyfesolver.prepared_static_loads import PreparedStaticError

requires_pypardiso = pytest.mark.skipif(not linalg._HAS_PYPARDISO, reason="pypardiso not installed")

_SECTION = {"area": 0.02, "Iy": 1.6667e-5, "Iz": 6.6667e-5, "J": 2.29e-5}


def _beam(n: int, fixed: dict | None) -> FEModel:
    model = FEModel("beam")
    model.add_material("mat", 2.1e11, 0.3, density=7850.0)
    for i in range(n + 1):
        model.add_node(i + 1, 10.0 * i / n, 0.0, 0.0)
    for i in range(n):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(_SECTION)))
    if fixed is not None:
        model.add_boundary_condition(BoundaryCondition("support", [1], fixed))
    return model


def _hinge_model(n: int = 20):
    """Cantilever whose support leaves rz free: a rigid rotation mechanism."""

    model = _beam(n, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0})
    load = LoadCase("tip")
    load.add_nodal_load(n + 1, [0.0, 1.0e3, -2.0e4, 50.0, 0.0, 0.0])
    return model, load


def _balanced_free_beam(n: int = 400):
    model = _beam(n, None)
    load = LoadCase("balanced")
    load.add_nodal_load(1, [0.0, 0.0, 1.0e3, 0, 0, 0])
    load.add_nodal_load(n + 1, [0.0, 0.0, 1.0e3, 0, 0, 0])
    load.add_nodal_load(n // 2 + 1, [0.0, 0.0, -2.0e3, 0, 0, 0])
    return model, load


def _band(n: int, half_bandwidth: int, weights: bool = False) -> sparse.csr_matrix:
    """Band graph Laplacian: singular (constant nullspace)."""

    rng = np.random.default_rng(7)
    offsets = [k for k in range(-half_bandwidth, half_bandwidth + 1) if k]
    upper = {k: (rng.uniform(0.5, 2.0, n - k) if weights else np.ones(n - k))
             for k in offsets if k > 0}
    adjacency = sparse.diags([upper[abs(k)] for k in offsets], offsets, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree) - adjacency).tocsr()


def _loads(n: int):
    return {
        "inconsistent": np.ones(n),
        "consistent": np.r_[1.0, np.zeros(n - 2), -1.0],
    }


_ADVERSARIAL = [(60, 1, False), (60, 3, True), (400, 3, False), (400, 13, True), (1500, 13, False)]


# --------------------------------------------------------------------------
# Rule A: end-to-end refusals
# --------------------------------------------------------------------------


def test_hinge_mechanism_is_refused_through_solve_linear() -> None:
    """Formerly |x| ~ 2e15 returned converged/CERTIFIED (rz-free support)."""

    model, load = _hinge_model()
    u, info = solve_linear(model, load)
    convergence = info["convergence_info"]
    assert convergence["status"] == "failed"
    assert str(convergence["error"]).startswith(linalg.FORWARD_ERROR_PROBE_EXCEEDED)
    assert convergence["backend"]["forward_error_probe_disposition"] == "EXCEEDED"
    assert convergence["backend"]["failure_kind"] == linalg.FORWARD_ERROR_PROBE_EXCEEDED
    assert convergence.get("relative_backward_error_disposition") != "CERTIFIED"
    assert np.all(np.isfinite(u)) and np.max(np.abs(u)) < 1.0


@pytest.mark.parametrize(("n", "half_bandwidth", "weights"), _ADVERSARIAL)
@pytest.mark.parametrize("load", ["inconsistent", "consistent"])
def test_adversarial_singular_bands_are_refused_by_the_static_solve(n, half_bandwidth, weights, load) -> None:
    matrix = _band(n, half_bandwidth, weights)
    q, info = assembly._solve_reduced_system(matrix, _loads(n)[load], "direct")
    assert info["status"] == "failed"
    assert info.get("relative_backward_error_disposition") != "CERTIFIED"
    assert np.all(np.isfinite(q))


@pytest.mark.parametrize(("n", "half_bandwidth", "weights"), _ADVERSARIAL)
def test_adversarial_singular_bands_are_refused_by_superlu(n, half_bandwidth, weights) -> None:
    handle = SparseSolverBackend().factorize(_band(n, half_bandwidth, weights), MatrixClass.SYMMETRIC_INDEFINITE)
    assert handle.status == "failed" and handle._solver is None
    reason = str(handle.failure_reason)
    assert reason.startswith(linalg.FORWARD_ERROR_PROBE_EXCEEDED) or "singular" in reason.lower()


@requires_pypardiso
@pytest.mark.parametrize(("n", "half_bandwidth", "weights"), _ADVERSARIAL)
@pytest.mark.parametrize(
    ("matrix_class", "mtype"),
    [(MatrixClass.SPD, 2), (MatrixClass.SYMMETRIC_INDEFINITE, -2), (MatrixClass.GENERAL, 11)],
)
def test_adversarial_singular_bands_are_refused_by_pardiso(n, half_bandwidth, weights, matrix_class, mtype) -> None:
    backend = PyPardisoSolverBackend()
    handle = backend.factorize(_band(n, half_bandwidth, weights), matrix_class)
    assert handle.status == "failed" and handle._solver is None
    assert handle.metadata["pardiso_failure_kind"] in (
        linalg.PARDISO_PERTURBED_PIVOTS, linalg.FORWARD_ERROR_PROBE_EXCEEDED)
    if handle.metadata["pardiso_failure_kind"] == linalg.FORWARD_ERROR_PROBE_EXCEEDED:
        assert handle.metadata["pardiso_mtype"] == mtype
        assert handle.metadata["forward_error_probe_disposition"] in ("EXCEEDED", "NONFINITE")
    assert backend.retained_pattern_slots == 0


@requires_pypardiso
def test_spd_cholesky_refusals_are_rule_a_and_end_to_end_falls_back() -> None:
    """mtype 2 reports no documented pivot count: rule A is what refuses it."""

    refused = [PyPardisoSolverBackend().factorize(_band(n, hb, w), MatrixClass.SPD)
               for n, hb, w in _ADVERSARIAL]
    assert any(h.metadata.get("pardiso_failure_kind") == linalg.FORWARD_ERROR_PROBE_EXCEEDED
               for h in refused)
    # The auto backend falls back to SuperLU, which runs its own check.
    auto = AutoSparseSolverBackend(pypardiso_min_dimension=1, pypardiso_min_nnz=0)
    handle = auto.factorize(_band(400, 13), MatrixClass.SPD)
    assert handle.backend_name == "scipy_superlu" and handle.status == "failed"
    assert handle.metadata["fallback_from_backend"] == "pypardiso"


@requires_pypardiso
def test_forced_pardiso_probe_refusal_fails_closed_without_fallback() -> None:
    # mtype 2 Cholesky factorizes this singular band through a rounding-level
    # pivot without a documented perturbation report: rule A refuses it.
    handle = AutoSparseSolverBackend().factorize(
        _band(400, 3), MatrixClass.SPD, options={"backend": "pypardiso"})
    assert handle.backend_name == "pypardiso" and handle.status == "failed"
    assert handle.metadata["pardiso_failure_kind"] == linalg.FORWARD_ERROR_PROBE_EXCEEDED


# --------------------------------------------------------------------------
# Rule A: ordinary systems pass and record the value
# --------------------------------------------------------------------------


def test_ordinary_factorization_records_the_probe_and_solves_unchanged() -> None:
    matrix = (_band(400, 3) + sparse.eye(400)).tocsr()
    handle = linalg.factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE)
    assert handle.status == "ok"
    assert handle.metadata["forward_error_probe_mode"] == "refuse"
    assert handle.metadata["forward_error_probe_limit"] == 1.0e-3
    assert handle.metadata["forward_error_probe_disposition"] == "PASSED"
    assert 0.0 <= handle.metadata["forward_error_probe"] < 1.0e-10
    # The probe is not counted as a user solve and does not change the answer.
    assert handle.solve_count == 0 and handle.solve_time == 0.0
    rhs = np.linspace(1.0, 2.0, 400)
    expected = sparse.linalg.splu(sparse.csc_matrix(matrix)).solve(rhs)
    assert np.array_equal(handle.solve(rhs), expected)
    assert handle.solve_count == 1


def test_free_free_balanced_beam_probes_the_displacement_block() -> None:
    model, load = _balanced_free_beam(400)
    u, info = solve_linear(model, load)
    convergence = info["convergence_info"]
    assert convergence["status"] == "converged"
    assert convergence["method"] == "rigid_body_nullspace_augmented"
    backend = convergence["backend"]
    n_displacement = backend["shape"][0] - convergence["nullspace_rank"]
    assert backend["forward_error_probe_measured_entries"] == n_displacement
    assert backend["forward_error_probe_disposition"] == "PASSED"
    assert backend["forward_error_probe"] <= 1.0e-6
    assert 0.0 <= convergence["relative_forward_error_estimate"] <= 1.0e-3
    with AnalysisSession(model) as session:
        u_session, info_session = solve_linear(model, load, session=session)
        again, _ = solve_linear(model, load, session=session)
        session_convergence = info_session["convergence_info"]
        assert session_convergence["status"] == "converged"
        assert session_convergence["backend"]["forward_error_probe_measured_entries"] == n_displacement
        # The masked factorization is reused (the hit path passes the same options).
        assert session.diagnostics()["counters"]["static_operator_builds"] == 1
        np.testing.assert_array_equal(again, u_session)
    np.testing.assert_allclose(u_session, u, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------
# Rule A: report mode, options and caching
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["superlu", pytest.param("pypardiso", marks=requires_pypardiso)])
def test_report_mode_records_but_never_refuses(backend) -> None:
    matrix = _band(400, 3)
    if backend == "superlu":
        handle = SparseSolverBackend().factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE,
                                                 options={"forward_error_probe": "report"})
    else:
        handle = PyPardisoSolverBackend().factorize(matrix, MatrixClass.SPD,
                                                    options={"forward_error_probe": "report"})
    assert handle.status == "ok"
    assert handle.metadata["forward_error_probe_mode"] == "report"
    assert handle.metadata["forward_error_probe_disposition"] in ("EXCEEDED_REPORTED", "NONFINITE_REPORTED")
    rhs = np.ones(400)
    x = handle.solve(rhs)
    assert x.shape == (400,)
    # Rule B (defence in depth) flags the inconsistent load on this handle.
    assert linalg.relative_forward_error_estimate(matrix, x, rhs, handle=handle) > 1.0e-3


def test_nonlinear_tangent_call_sites_use_report_mode(monkeypatch) -> None:
    import anyfesolver.nonlinear_static as nonlinear_static
    from anyfesolver import solve_static_nonlinear

    seen = []
    original = nonlinear_static.factorize

    def spy(matrix, matrix_class, **kwargs):
        seen.append(dict(kwargs.get("options") or {}))
        return original(matrix, matrix_class, **kwargs)

    monkeypatch.setattr(nonlinear_static, "factorize", spy)
    model = _beam(4, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})
    load = LoadCase("tip")
    load.add_nodal_load(5, [0.0, 1.0e3, 0.0, 0.0, 0.0, 0.0])
    solve_static_nonlinear(model, load, num_steps=2)
    assert seen and all(options.get("forward_error_probe") == "report" for options in seen)


@pytest.mark.parametrize("backend_factory", [
    SparseSolverBackend,
    AutoSparseSolverBackend,
    pytest.param(PyPardisoSolverBackend, marks=requires_pypardiso),
])
@pytest.mark.parametrize("options", [
    {"forward_error_probe": "off"},
    {"forward_error_probe": True},
    {"forward_error_probe_mask": np.ones(3, dtype=bool)},
    {"forward_error_probe_mask": np.zeros(5, dtype=bool)},
    {"forward_error_probe_mask": np.array([0, 7])},
    {"forward_error_probe_mask": np.array([0.0, 1.0])},
    {"forward_error_probe_mask": np.ones((5, 1), dtype=bool)},
])
def test_invalid_probe_options_raise_value_error(backend_factory, options) -> None:
    matrix = sparse.eye(5, format="csr")
    with pytest.raises(ValueError, match="forward_error_probe"):
        backend_factory().factorize(matrix, MatrixClass.SPD, options=options)


def test_probe_mask_selects_entries_and_keys_the_cache_by_full_content() -> None:
    matrix = (_band(2000, 3) + sparse.eye(2000)).tocsr()
    index = np.arange(1000)
    boolean = np.zeros(2000, dtype=bool)
    boolean[:1000] = True
    for mask in (index, boolean, list(index)):
        handle = linalg.factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE,
                                  options={"forward_error_probe_mask": mask})
        assert handle.metadata["forward_error_probe_measured_entries"] == 1000
    # Two masks whose abbreviated str() is identical still key differently.
    a = np.arange(2000)
    b = a.copy()
    b[1000] = 1001
    assert str(a) == str(b)
    signature_a = linalg._options_signature({"forward_error_probe_mask": a})
    assert signature_a != linalg._options_signature({"forward_error_probe_mask": b})
    assert signature_a == linalg._options_signature({"forward_error_probe_mask": a.copy()})
    assert linalg._options_signature({"forward_error_probe_mask": None}) == linalg._options_signature(
        {"forward_error_probe_mask": None})


def test_empty_matrix_skips_the_probe() -> None:
    metadata: dict = {}
    refused = linalg._run_forward_error_probe(metadata, {}, 0, None, None)
    assert refused is False
    assert metadata["forward_error_probe"] is None
    assert metadata["forward_error_probe_disposition"] == "SKIPPED_EMPTY"


def test_refused_factorizations_are_never_cached() -> None:
    cache = FactorizationCache(name="probe", max_entries=4)
    matrix = _band(400, 3)
    for _ in range(2):
        handle = cache.factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE)
        assert handle.status == "failed"
        assert str(handle.failure_reason).startswith(linalg.FORWARD_ERROR_PROBE_EXCEEDED)
    diagnostics = cache.diagnostics()
    assert diagnostics["entries"] == 0 and diagnostics["hits"] == 0
    assert diagnostics["misses"] == 2 and diagnostics["factorization_failures"] == 2
    # A report-mode handle is cached under its own key, never served to a refuse request.
    reported = cache.factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE,
                               options={"forward_error_probe": "report"})
    assert reported.status == "ok"
    refused = cache.factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE)
    assert refused is not reported and refused.status == "failed"


def test_inverse_operator_of_a_refused_factorization_raises() -> None:
    with pytest.raises(RuntimeError, match=linalg.FORWARD_ERROR_PROBE_EXCEEDED):
        linalg.cached_inverse_operator(_band(200, 3), MatrixClass.SYMMETRIC_INDEFINITE)


# --------------------------------------------------------------------------
# Rule B: refuses on every certified route, also with rule A out of the way
# --------------------------------------------------------------------------


@pytest.fixture
def rule_a_disabled(monkeypatch):
    """Raise the rule A limit so that rule B alone decides (probe still recorded)."""

    monkeypatch.setattr(linalg, "_FORWARD_ERROR_PROBE_LIMIT", float("inf"))


def test_rule_b_refuses_the_reduced_direct_solve(rule_a_disabled) -> None:
    n = 2000
    q, info = assembly._solve_reduced_system(_band(n, 13), np.ones(n), "direct")
    assert info["status"] == "failed"
    assert str(info["error"]).startswith(linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED)
    assert info["relative_backward_error_disposition"] == linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED
    assert info["relative_backward_error"] <= assembly._LINEAR_STATIC_BACKWARD_ERROR_LIMIT
    assert (info["relative_forward_error_estimate"] is None
            or info["relative_forward_error_estimate"] > 1.0e-3)
    assert info["relative_forward_error_estimate_limit"] == 1.0e-3


def test_rule_b_refuses_the_hinge_mechanism_through_solve_linear(rule_a_disabled) -> None:
    model, load = _hinge_model()
    _u, info = solve_linear(model, load)
    convergence = info["convergence_info"]
    assert convergence["status"] == "failed"
    assert convergence["relative_backward_error_disposition"] == linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED


def test_rule_b_refuses_solve_linear_many(rule_a_disabled) -> None:
    model, load = _hinge_model()
    _u, info = solve_linear_many(model, [load, load])
    assert info["status"] == "failed"
    assert info["relative_backward_error_disposition"] == linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED
    assert str(info["error"]).startswith(linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED)


def test_rule_b_refuses_the_static_batch_route(rule_a_disabled) -> None:
    model, load = _hinge_model()
    with AnalysisSession(model) as session:
        with pytest.raises(StaticBatchError) as raised:
            list(iter_linear_batches(model, [StaticCase("tip", load)], session=session))
    assert raised.value.diagnostics["relative_backward_error_disposition"] == linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED


def test_rule_b_refuses_the_prepared_static_route(rule_a_disabled) -> None:
    model, load = _hinge_model()
    with prepare_static_evaluator(model) as evaluator:
        with pytest.raises(PreparedStaticError) as raised:
            list(evaluator.iter_batches([StaticCase("tip", load)]))
    assert raised.value.diagnostics["relative_backward_error_disposition"] == linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED


def _two_component_free_system(n: int = 300):
    """Two disconnected bands, but only the global constant declared as rigid mode."""

    K = sparse.block_diag((_band(n, 3), _band(n, 3)), format="csr")
    Q = np.ones((2 * n, 1)) / np.sqrt(2 * n)
    F = np.r_[np.ones(n), -np.ones(n)]  # globally balanced, inconsistent per component
    return K, F, Q


def test_nullspace_route_rule_a_refuses_an_undeclared_mechanism() -> None:
    K, F, Q = _two_component_free_system()
    q, info = assembly._solve_nullspace_augmented_system(K, F, Q)
    assert info["status"] == "failed"
    assert str(info["error"]).startswith(linalg.FORWARD_ERROR_PROBE_EXCEEDED)
    assert np.all(q == 0.0)


def test_nullspace_route_rule_b_refuses_on_the_displacement_block(rule_a_disabled) -> None:
    K, F, Q = _two_component_free_system()
    _q, info = assembly._solve_nullspace_augmented_system(K, F, Q)
    assert info["status"] == "failed"
    assert info["relative_forward_error_estimate_disposition"] == linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED
    assert str(info["error"]).startswith(linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED)


def test_rule_b_estimate_handles_columns_rows_scaling_and_nonfinite() -> None:
    matrix = sparse.diags([2.0, 4.0, 8.0], format="csr")
    handle = linalg.factorize(matrix, MatrixClass.SPD)
    rhs = np.array([[2.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    exact = handle.solve(rhs)
    values = linalg.relative_forward_error_estimates(matrix, exact, rhs, handle=handle)
    assert values.shape == (2,) and np.all(values == 0.0)
    wrong = exact.copy()
    wrong[2, 0] += 1.0
    assert linalg.relative_forward_error_estimates(matrix, wrong, rhs, handle=handle)[0] == pytest.approx(0.5)
    # The wrong entry's correction is outside the measured rows.
    assert linalg.relative_forward_error_estimates(
        matrix, wrong, rhs, handle=handle, rows=slice(0, 2))[0] == 0.0
    wrong[0, 1] = np.nan
    assert np.isinf(linalg.relative_forward_error_estimates(matrix, wrong, rhs, handle=handle)).all()
    # Scaling: matrix = diag(s) K diag(s), solution/rhs in K's variables.
    s = np.array([1.0, 0.5, 0.25])
    K = sparse.diags([2.0, 16.0, 128.0], format="csr")
    x = np.array([1.0, 0.25, 0.0625])
    f = K @ x
    assert linalg.relative_forward_error_estimate(matrix, x, f, handle=handle, scaling=s) == 0.0
    assert handle.solve_count == 2  # certification solves are not counted
