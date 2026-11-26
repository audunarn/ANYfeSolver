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
        session_convergence = info_session["convergence_info"]
        assert session_convergence["status"] == "converged"
        assert session_convergence["backend"]["forward_error_probe_measured_entries"] == n_displacement
        first_handle = session._static_operator.handle
        cache = session.factorization_cache
        hits, misses = cache.hits, cache.misses
        again, info_again = solve_linear(model, load, session=session)
        # The masked factorization is reused (the hit path passes the same
        # options): a factorization-cache hit, no miss, the identical handle.
        assert (cache.hits - hits, cache.misses - misses) == (1, 0)
        assert session._static_operator.handle is first_handle
        assert first_handle.metadata["cache_hit"] is True
        assert first_handle.metadata["forward_error_probe_measured_entries"] == n_displacement
        assert (info_again["convergence_info"]["backend"]["forward_error_probe_measured_entries"]
                == n_displacement)
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
    # A None mask is no mask: it keys apart from a real mask, and the options
    # without it are unchanged (no digest is substituted).
    none_signature = linalg._options_signature({"forward_error_probe_mask": None})
    assert none_signature == '{"forward_error_probe_mask":null}'
    assert none_signature != signature_a
    handle = linalg.factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE,
                              options={"forward_error_probe_mask": None})
    assert handle.metadata["forward_error_probe_measured_entries"] == 2000


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
    """R1 #8: two different loads, one consistent with the rz mechanism (passes), one not."""

    model, failing = _hinge_model()
    passing = LoadCase("out_of_plane")
    passing.add_nodal_load(21, [0.0, 0.0, -2.0e4, 50.0, 0.0, 0.0])
    u, info = solve_linear_many(model, [passing, failing])
    assert info["status"] == "failed"
    assert info["relative_backward_error_disposition"] == linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED
    assert str(info["error"]).startswith(linalg.FORWARD_ERROR_ESTIMATE_EXCEEDED)
    # Per input column evidence (additive); the failure shape is unchanged.
    estimates = info["relative_forward_error_estimates"]
    assert len(estimates) == 2
    assert estimates[0] is not None and estimates[0] <= 1.0e-3
    assert estimates[1] is None or estimates[1] > 1.0e-3
    assert info["relative_forward_error_failed_columns"] == [1]
    assert info["relative_forward_error_estimate"] == (None if estimates[1] is None else max(estimates))
    # As on the LIMIT_EXCEEDED path: every column is the constrained baseline u0.
    assert u.shape[1] == 2 and np.array_equal(u[:, 0], u[:, 1])
    assert np.all(u == 0.0)


def test_solve_linear_many_reports_per_column_estimates_when_certified() -> None:
    model = _beam(4, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})
    loads = []
    for k, component in enumerate((1, 2)):
        load = LoadCase(f"tip{k}")
        vector = [0.0] * 6
        vector[component] = 1.0e3
        load.add_nodal_load(5, vector)
        loads.append(load)
    _u, info = solve_linear_many(model, loads)
    assert info["status"] == "converged"
    assert len(info["relative_forward_error_estimates"]) == 2
    assert info["relative_forward_error_failed_columns"] == []
    assert info["relative_forward_error_estimate"] == max(info["relative_forward_error_estimates"])


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
    # A non-finite column fails alone; the finite column is still measured.
    values = linalg.relative_forward_error_estimates(matrix, wrong, rhs, handle=handle)
    assert np.isinf(values[1]) and values[0] == pytest.approx(0.5)
    # Scaling: matrix = diag(s) K diag(s), solution/rhs in K's variables.
    s = np.array([1.0, 0.5, 0.25])
    K = sparse.diags([2.0, 16.0, 128.0], format="csr")
    x = np.array([1.0, 0.25, 0.0625])
    f = K @ x
    assert linalg.relative_forward_error_estimate(matrix, x, f, handle=handle, scaling=s) == 0.0
    assert handle.solve_count == 2  # certification solves are not counted


def _shifted_laplacian_at_eigenvalue(n: int = 50, mode: int = 7):
    stiffness = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tocsc()
    shift = 2.0 - 2.0 * np.cos(np.pi * mode / (n + 1))
    return (stiffness - shift * sparse.identity(n)).tocsc()


def test_shift_at_eigenvalue_refused_by_default_but_reported_in_report_mode():
    matrix = _shifted_laplacian_at_eigenvalue()
    with pytest.raises(RuntimeError):
        linalg.cached_inverse_operator(
            matrix, MatrixClass.SYMMETRIC_INDEFINITE,
            cache=FactorizationCache(name="refuse", max_entries=2),
        )
    _, handle = linalg.cached_inverse_operator(
        matrix, MatrixClass.SYMMETRIC_INDEFINITE,
        cache=FactorizationCache(name="report", max_entries=2),
        options={"forward_error_probe": "report"},
    )
    assert handle.status == "ok"
    assert handle.metadata["forward_error_probe_mode"] == "report"
    assert handle.metadata["forward_error_probe"] > 1.0e-3


def test_buckling_shift_invert_at_eigenvalue_uses_report_mode():
    from anyfesolver import solve_eigenvalue_buckling
    from test_fe_solver_buckling import _beam_column_model

    model, _, _ = _beam_column_model(num_elements=12)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    first = solve_eigenvalue_buckling(model, states, num_modes=1, dense_size_limit=2)
    cache = FactorizationCache(name="buckling_report", max_entries=2)
    shifted = solve_eigenvalue_buckling(
        model, states, num_modes=1, dense_size_limit=2,
        shift_load_factor=first.critical_load_factor, factorization_cache=cache,
    )
    assert shifted.solver_status == "ok"
    assert shifted.critical_load_factor == pytest.approx(first.critical_load_factor, rel=1e-6)
    handles = list(cache._handles.values())
    assert handles
    for handle in handles:
        assert handle.metadata.get("forward_error_probe_mode") == "report"


# --------------------------------------------------------------------------
# Review round 1 (2026-11-26): probe execution, cleanup, stale, operator,
# dtype and cache-key regressions
# --------------------------------------------------------------------------


def _ordinary_spd(n: int = 400) -> sparse.csr_matrix:
    return (_band(n, 3) + sparse.eye(n)).tocsr()


def _failing_wrapper_solve(monkeypatch, error_type):
    """The next PARDISO wrapper solve (the probe's raw solve) raises ``error_type``."""

    original = linalg._PardisoFactorization.solve
    calls = {"armed": True}

    def solve(self, rhs):
        if calls["armed"]:
            calls["armed"] = False
            raise error_type("injected phase-33 failure")
        return original(self, rhs)

    monkeypatch.setattr(linalg._PardisoFactorization, "solve", solve)
    return calls


@requires_pypardiso
@pytest.mark.parametrize("mode", ["report", "refuse"])
@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
@pytest.mark.parametrize("reused", [False, True])
def test_pardiso_probe_execution_failure_never_publishes_an_ok_handle(
        monkeypatch, mode, error_type, reused) -> None:
    """R1 #1: a probe that could not execute is not a bad value: typed failed handle, slot discarded."""

    backend = PyPardisoSolverBackend()
    matrix = _ordinary_spd()
    earlier = None
    if reused:
        earlier = backend.factorize(matrix, MatrixClass.SPD)
        assert earlier.status == "ok" and backend.retained_pattern_slots == 1
        matrix = (matrix * 2.0).tocsr()  # same pattern: the slot is reused
    _failing_wrapper_solve(monkeypatch, error_type)
    cache = FactorizationCache(name="probe_failure", max_entries=4, backend=backend)
    handle = cache.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": mode})
    assert handle.status == "failed" and handle._solver is None
    assert handle.metadata["failure_kind"] == linalg.FORWARD_ERROR_PROBE_FAILED
    assert handle.metadata["pardiso_failure_kind"] == linalg.FORWARD_ERROR_PROBE_FAILED
    assert handle.metadata["forward_error_probe_disposition"] == "EXECUTION_FAILED"
    assert str(handle.failure_reason).startswith(linalg.FORWARD_ERROR_PROBE_FAILED)
    assert backend.retained_pattern_slots == 0
    assert cache.diagnostics()["entries"] == 0
    lease = linalg.lease_handle(handle)
    assert not lease.pinned
    if earlier is not None:
        with pytest.raises(linalg.PardisoStaleHandleError):
            earlier.solve(np.ones(matrix.shape[0]))


@requires_pypardiso
@pytest.mark.parametrize("mode", ["report", "refuse"])
@pytest.mark.parametrize("reused", [False, True])
def test_pardiso_probe_interrupt_propagates_and_discards_the_slot(monkeypatch, mode, reused) -> None:
    backend = PyPardisoSolverBackend()
    matrix = _ordinary_spd()
    if reused:
        assert backend.factorize(matrix, MatrixClass.SPD).status == "ok"
        matrix = (matrix * 2.0).tocsr()
    _failing_wrapper_solve(monkeypatch, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt, match="injected"):
        backend.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": mode})
    assert backend.retained_pattern_slots == 0


@pytest.mark.parametrize("mode", ["report", "refuse"])
@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
def test_superlu_probe_execution_failure_is_a_typed_failed_handle(monkeypatch, mode, error_type) -> None:
    real_splu = linalg.splu

    class Failing:
        def __init__(self, factor):
            self._factor = factor

        def solve(self, rhs):
            raise error_type("injected solve failure")

    monkeypatch.setattr(linalg, "splu", lambda *a, **k: Failing(real_splu(*a, **k)))
    handle = SparseSolverBackend().factorize(_ordinary_spd(), MatrixClass.SPD,
                                             options={"forward_error_probe": mode})
    assert handle.status == "failed" and handle._solver is None
    assert handle.metadata["failure_kind"] == linalg.FORWARD_ERROR_PROBE_FAILED
    assert handle.metadata["forward_error_probe_disposition"] == "EXECUTION_FAILED"


def test_superlu_probe_interrupt_propagates(monkeypatch) -> None:
    real_splu = linalg.splu

    class Interrupted:
        def __init__(self, factor):
            self._factor = factor

        def solve(self, rhs):
            raise KeyboardInterrupt("injected")

    monkeypatch.setattr(linalg, "splu", lambda *a, **k: Interrupted(real_splu(*a, **k)))
    with pytest.raises(KeyboardInterrupt):
        SparseSolverBackend().factorize(_ordinary_spd(), MatrixClass.SPD,
                                        options={"forward_error_probe": "report"})


@requires_pypardiso
def test_probe_refusal_cleanup_interrupt_reaches_the_caller_without_fallback(monkeypatch) -> None:
    """R1 #2: phase -1 raising KeyboardInterrupt while discarding a refused factorization."""

    real_free = linalg._free_mkl_solver

    def free(solver):
        error = real_free(solver)
        return error if error is not None else KeyboardInterrupt("injected phase -1 interrupt")

    monkeypatch.setattr(linalg, "_free_mkl_solver", free)
    auto = AutoSparseSolverBackend(pypardiso_min_dimension=1, pypardiso_min_nnz=0)
    fallbacks = []
    original = auto.scipy_backend.factorize
    monkeypatch.setattr(auto.scipy_backend, "factorize",
                        lambda *a, **k: fallbacks.append(1) or original(*a, **k))
    with pytest.raises(KeyboardInterrupt, match="phase -1"):
        auto.factorize(_band(400, 3), MatrixClass.SPD)
    assert fallbacks == []
    assert auto.pardiso_backend.retained_pattern_slots == 0


@requires_pypardiso
def test_probe_failure_cleanup_interrupt_replaces_the_ordinary_probe_failure(monkeypatch) -> None:
    real_free = linalg._free_mkl_solver
    monkeypatch.setattr(linalg, "_free_mkl_solver",
                        lambda solver: real_free(solver) or SystemExit("injected phase -1 exit"))
    _failing_wrapper_solve(monkeypatch, MemoryError)
    auto = AutoSparseSolverBackend(pypardiso_min_dimension=1, pypardiso_min_nnz=0)
    with pytest.raises(SystemExit, match="phase -1"):
        auto.factorize(_ordinary_spd(), MatrixClass.SPD, options={"forward_error_probe": "report"})


@requires_pypardiso
def test_probe_interrupt_keeps_primary_over_a_cleanup_interrupt(monkeypatch) -> None:
    real_free = linalg._free_mkl_solver
    monkeypatch.setattr(linalg, "_free_mkl_solver",
                        lambda solver: real_free(solver) or SystemExit("injected phase -1 exit"))
    _failing_wrapper_solve(monkeypatch, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt, match="injected phase-33") as raised:
        PyPardisoSolverBackend().factorize(_ordinary_spd(), MatrixClass.SPD)
    assert any("SystemExit" in note for note in getattr(raised.value, "__notes__", []))


@requires_pypardiso
def test_rule_b_keeps_the_typed_stale_handle_error() -> None:
    """R1 #3: a handle gone stale before the correction solve is not an inf estimate."""

    backend = PyPardisoSolverBackend()
    matrix = _ordinary_spd()
    handle = backend.factorize(matrix, MatrixClass.SPD)
    rhs = np.ones(matrix.shape[0])
    x = handle.solve(rhs)
    assert backend.factorize((matrix * 2.0).tocsr(), MatrixClass.SPD).status == "ok"
    with pytest.raises(linalg.PardisoStaleHandleError):
        linalg.relative_forward_error_estimates(matrix, x, rhs, handle=handle)
    with pytest.raises(linalg.PardisoStaleHandleError):
        linalg.relative_forward_error_estimate(matrix, x, rhs, handle=handle)


@requires_pypardiso
def test_uncounted_solve_race_after_require_solvable_is_the_stale_type(monkeypatch) -> None:
    """An invalidation between the state check and the solve keeps the stale type."""

    backend = PyPardisoSolverBackend()
    matrix = _ordinary_spd()
    handle = backend.factorize(matrix, MatrixClass.SPD)
    assert backend.factorize((matrix * 2.0).tocsr(), MatrixClass.SPD).status == "ok"
    monkeypatch.setattr(linalg, "_require_solvable", lambda _handle: None)
    with pytest.raises(linalg.PardisoStaleHandleError):
        linalg._uncounted_solve(handle, np.ones(matrix.shape[0]))


@requires_pypardiso
@pytest.mark.parametrize(("matrix_class", "mtype"), [(MatrixClass.SPD, 2),
                                                     (MatrixClass.SYMMETRIC_INDEFINITE, -2)])
@pytest.mark.parametrize("dense", [True, False])
def test_symmetric_pardiso_probe_uses_the_factorized_operator_for_upper_storage(
        matrix_class, mtype, dense) -> None:
    """R1 #4: upper-triangle storage of an SPD matrix is factorized as the symmetric operator."""

    full = sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]])) if dense else _ordinary_spd()
    upper = sparse.triu(full, format="csr")
    handle = PyPardisoSolverBackend().factorize(upper, matrix_class)
    assert handle.status == "ok"
    assert handle.metadata["pardiso_mtype"] == mtype
    assert handle.metadata["forward_error_probe_disposition"] == "PASSED"
    assert handle.metadata["forward_error_probe"] < 1.0e-10
    # The solve answers the symmetric operator (MKL's), as the probe now measures.
    rhs = np.arange(1.0, full.shape[0] + 1.0)
    np.testing.assert_allclose(full @ handle.solve(rhs), rhs, rtol=1e-10)


@requires_pypardiso
@pytest.mark.parametrize("matrix_class", [MatrixClass.SPD, MatrixClass.SYMMETRIC_INDEFINITE,
                                          MatrixClass.GENERAL])
def test_full_storage_probe_measures_the_factorized_operator(matrix_class) -> None:
    """R2 #3: full storage probes the prepared upper operator (mtypes 2/-2), csr @ z for mtype 11."""

    matrix = _ordinary_spd()
    handle = PyPardisoSolverBackend().factorize(matrix, matrix_class)
    assert handle.status == "ok"
    z = np.random.default_rng(linalg._FORWARD_ERROR_PROBE_SEED).uniform(1.0, 2.0, matrix.shape[0])
    if handle.metadata["pardiso_mtype"] == 11:
        rhs = matrix @ z
    else:
        upper = sparse.triu(matrix, format="csr")
        rhs = upper @ z + sparse.triu(matrix, k=1, format="csr").T @ z
    x = handle._solver.solve(rhs)
    assert handle.metadata["forward_error_probe"] == float(np.max(np.abs(x - z)) / np.max(np.abs(z)))


def test_float32_superlu_identity_factorization_passes_the_probe() -> None:
    """R1 #7: the probe's raw-solve input is in the factor's dtype."""

    handle = SparseSolverBackend().factorize(sparse.eye(5, dtype=np.float32, format="csc"), MatrixClass.SPD)
    assert handle.status == "ok"
    assert handle.metadata["forward_error_probe_disposition"] == "PASSED"
    assert handle.metadata["forward_error_probe"] < 1.0e-6


@requires_pypardiso
def test_float32_pardiso_is_unchanged_by_the_probe() -> None:
    """PyPardiso itself rejects float32 (before any probe), exactly as before rule A."""

    handle = PyPardisoSolverBackend().factorize(sparse.eye(5, dtype=np.float32, format="csr"), MatrixClass.SPD)
    assert handle.status == "failed"
    assert "float64" in str(handle.failure_reason)
    assert "forward_error_probe" not in handle.metadata


def test_cache_hit_validates_the_probe_mode() -> None:
    """R1 #9: a non-string mode raises ValueError on a hit as on a miss."""

    cache = FactorizationCache(name="probe_key", max_entries=4)
    matrix = _ordinary_spd(50)
    assert cache.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": "report"}).status == "ok"
    with pytest.raises(ValueError, match="forward_error_probe"):
        cache.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": np.array("report")})
    with pytest.raises(ValueError, match="forward_error_probe_mask"):
        cache.factorize(matrix, MatrixClass.SPD,
                        options={"forward_error_probe": "report", "forward_error_probe_mask": np.array([0, 99])})
    assert cache.hits == 0
    # A numpy string scalar is a str and keys exactly as the plain string (a hit).
    again = cache.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": np.str_("report")})
    assert again.status == "ok" and cache.hits == 1


def test_native_beam_arc_predictors_use_report_mode(monkeypatch) -> None:
    """R1 #5 (N1): the native GE beam arc tangents (bordered predictors) use report mode."""

    import anyfesolver._ge_beam3_arc_program as arc_program
    import anyfesolver._ge_beam3_native_arc as native_arc

    matrix = sparse.diags([1.0, 2.0], format="csr")
    column = np.array([-1.0, 0.0])
    for module, direction in ((arc_program, arc_program._direction), (native_arc, native_arc.direction)):
        seen = []
        original = module.factorize

        def spy(matrix, matrix_class, _original=original, _seen=seen, **kwargs):
            _seen.append(dict(kwargs.get("options") or {}))
            return _original(matrix, matrix_class, **kwargs)

        monkeypatch.setattr(module, "factorize", spy)
        direction(matrix, column, np.array([0, 1]), np.array([1.0, 0.0, 0.0]), np.ones(3))
        assert seen and all(options.get("forward_error_probe") == "report" for options in seen)


def test_shell_b2_reference_buckling_shift_invert_uses_report_mode(monkeypatch) -> None:
    """R1 #5 (decision 2a): a stiffened-shell shift-invert factorization uses report mode."""

    import importlib.util
    from pathlib import Path

    from anyfesolver import stiffened_shell as api

    spec = importlib.util.spec_from_file_location(
        "b2_buckling_example_r1",
        Path(__file__).parents[1] / "examples/stiffened_shell_b2_reference_buckling.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    seen = []
    original = linalg.factorize_cached

    def spy(matrix, matrix_class, **kwargs):
        seen.append((linalg._coerce_matrix_class(matrix_class), dict(kwargs.get("options") or {})))
        return original(matrix, matrix_class, **kwargs)

    owner = api.create_analysis(definition=example.definition())
    try:
        load = np.zeros(54)
        load[[12, 30]] = -1.0
        monkeypatch.setattr(linalg, "factorize_cached", spy)
        result = owner.buckling(api.B2ReferenceBucklingRequest(load, example.traction_sources()))
        assert result.status == "ok", result.failure_reason
    finally:
        owner.close()
    shifted = [options for matrix_class, options in seen if matrix_class == MatrixClass.GENERAL]
    assert shifted and all(options.get("forward_error_probe") == "report" for options in shifted)


# --------------------------------------------------------------------------
# Review round 2 (2026-11-26): probe operator, complete probe wrapping,
# float32 Rule B and the literal probe mode
# --------------------------------------------------------------------------


@requires_pypardiso
@pytest.mark.parametrize("stored", ["one_lower_entry", "differing_lower_value"])
def test_symmetric_probe_uses_the_prepared_upper_operator_for_any_storage(stored) -> None:
    """R2 #3: MKL factorizes triu(A) mirrored; a stored lower entry must not switch the probe."""

    if stored == "one_lower_entry":
        given = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 0.0, 2.0]])
    else:
        given = np.array([[4.0, 1.0, 0.0], [3.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
    upper = np.triu(given)
    factorized = upper + np.triu(given, 1).T  # the operator MKL factorizes
    handle = PyPardisoSolverBackend().factorize(sparse.csr_matrix(given), MatrixClass.SPD)
    assert handle.status == "ok", handle.failure_reason
    assert handle.metadata["pardiso_mtype"] == 2
    assert handle.metadata["forward_error_probe_disposition"] == "PASSED"
    assert handle.metadata["forward_error_probe"] < 1.0e-12
    rhs = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(factorized @ handle.solve(rhs), rhs, rtol=1e-12)


@requires_pypardiso
def test_symmetric_probe_counts_the_diagonal_once_without_overflow() -> None:
    """R2 #3: diag(max_float/2) has a finite probe RHS; U z + U^T z - d z overflowed."""

    huge = np.finfo(float).max / 2.0
    handle = PyPardisoSolverBackend().factorize(sparse.diags([huge] * 3, format="csr"), MatrixClass.SPD)
    assert handle.status == "ok", handle.failure_reason
    assert handle.metadata["forward_error_probe_disposition"] == "PASSED"
    assert handle.metadata["forward_error_probe"] < 1.0e-12


def _failing_probe_rng(monkeypatch, error_type=MemoryError):
    """Allocating the probe's z (the probe seed's generator) raises ``error_type``."""

    original = np.random.default_rng

    def rng(seed=None, *args, **kwargs):
        if seed == linalg._FORWARD_ERROR_PROBE_SEED:
            raise error_type("injected z allocation failure")
        return original(seed, *args, **kwargs)

    monkeypatch.setattr(linalg.np.random, "default_rng", rng)


@requires_pypardiso
@pytest.mark.parametrize("mode", ["report", "refuse"])
def test_pardiso_probe_z_allocation_failure_is_an_execution_failure(monkeypatch, mode) -> None:
    """R2 #4: a MemoryError allocating z is a typed failed handle, not a raw exception."""

    backend = PyPardisoSolverBackend()
    _failing_probe_rng(monkeypatch)
    handle = backend.factorize(_ordinary_spd(), MatrixClass.SPD, options={"forward_error_probe": mode})
    assert handle.status == "failed" and handle._solver is None
    assert handle.metadata["failure_kind"] == linalg.FORWARD_ERROR_PROBE_FAILED
    assert handle.metadata["forward_error_probe_disposition"] == "EXECUTION_FAILED"
    assert backend.retained_pattern_slots == 0


@pytest.mark.parametrize("mode", ["report", "refuse"])
def test_superlu_probe_z_allocation_failure_is_an_execution_failure(monkeypatch, mode) -> None:
    _failing_probe_rng(monkeypatch)
    handle = SparseSolverBackend().factorize(_ordinary_spd(), MatrixClass.SPD,
                                             options={"forward_error_probe": mode})
    assert handle.status == "failed" and handle._solver is None
    assert handle.metadata["failure_kind"] == linalg.FORWARD_ERROR_PROBE_FAILED
    assert handle.metadata["forward_error_probe_disposition"] == "EXECUTION_FAILED"


def test_superlu_probe_z_allocation_interrupt_propagates(monkeypatch) -> None:
    _failing_probe_rng(monkeypatch, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt, match="injected"):
        SparseSolverBackend().factorize(_ordinary_spd(), MatrixClass.SPD)


class _MalformedSubtraction(np.ndarray):
    """A raw-solve result whose ``x - z`` temporary cannot be allocated."""

    def __sub__(self, other):
        raise MemoryError("injected temporary allocation failure")


_MALFORMED_RESULTS = {
    "short": lambda x: x[:-1],
    "two_dimensional": lambda x: x[:, None],
    "list": lambda x: list(x),
    "complex": lambda x: x.astype(complex),
    "object": lambda x: x.astype(object),
    "temporary_memory_error": lambda x: x.view(_MalformedSubtraction),
}


@pytest.mark.parametrize("mode", ["report", "refuse"])
@pytest.mark.parametrize("kind", sorted(_MALFORMED_RESULTS))
def test_superlu_malformed_probe_solve_is_an_execution_failure(monkeypatch, mode, kind) -> None:
    """R2 #4: a malformed raw-solve result never publishes ok (report mode included)."""

    real_splu = linalg.splu
    corrupt = _MALFORMED_RESULTS[kind]

    class Malformed:
        def __init__(self, factor):
            self._factor = factor

        def solve(self, rhs):
            return corrupt(self._factor.solve(rhs))

    monkeypatch.setattr(linalg, "splu", lambda *a, **k: Malformed(real_splu(*a, **k)))
    handle = SparseSolverBackend().factorize(_ordinary_spd(50), MatrixClass.SPD,
                                             options={"forward_error_probe": mode})
    assert handle.status == "failed" and handle._solver is None
    assert handle.metadata["failure_kind"] == linalg.FORWARD_ERROR_PROBE_FAILED
    assert handle.metadata["forward_error_probe_disposition"] == "EXECUTION_FAILED"


@requires_pypardiso
@pytest.mark.parametrize("mode", ["report", "refuse"])
def test_pardiso_wrong_length_probe_solve_is_an_execution_failure(monkeypatch, mode) -> None:
    """R2 #4: the PARDISO raw solve returning a wrong-length result: typed failure, slot discarded."""

    backend = PyPardisoSolverBackend()
    original = linalg._PardisoFactorization.solve
    monkeypatch.setattr(linalg._PardisoFactorization, "solve",
                        lambda self, rhs: original(self, rhs)[:-1])
    handle = backend.factorize(_ordinary_spd(), MatrixClass.SPD, options={"forward_error_probe": mode})
    assert handle.status == "failed" and handle._solver is None
    assert handle.metadata["failure_kind"] == linalg.FORWARD_ERROR_PROBE_FAILED
    assert handle.metadata["forward_error_probe_disposition"] == "EXECUTION_FAILED"
    assert backend.retained_pattern_slots == 0


def test_rule_b_on_a_float32_superlu_identity_fails_closed() -> None:
    """R2 #5 (pre-existing handle.solve limit, follow-up): Rule B is never ok on such a handle."""

    matrix = sparse.eye(5, dtype=np.float32, format="csc")
    handle = SparseSolverBackend().factorize(matrix, MatrixClass.SPD)
    assert handle.status == "ok"  # as on main (round-1 #7)
    y = np.ones(5)
    values = linalg.relative_forward_error_estimates(matrix, y, y, handle=handle)
    assert values.shape == (1,) and np.isinf(values[0])
    assert not linalg.forward_error_estimate_accepted(
        linalg.relative_forward_error_estimate(matrix, y, y, handle=handle))


class _DisplayedMode(str):
    """A str subclass whose rendering differs from its value."""

    def __str__(self) -> str:
        return "ProbeMode.DEFAULT"

    __repr__ = __str__


class _AgreeingMode(str):
    """A str subclass whose ``__eq__`` claims equality with everything."""

    def __eq__(self, other):
        return True

    __hash__ = str.__hash__


def test_probe_mode_subclasses_normalize_to_the_literal_for_key_and_execution() -> None:
    """R2 #7: report and refuse never alias in the cache; execution uses the same literal."""

    report, refuse = _DisplayedMode("report"), _DisplayedMode("refuse")
    assert str(report) == str(refuse)
    cache = FactorizationCache(name="probe_literal", max_entries=4)
    matrix = _ordinary_spd(50)
    assert (cache.key(matrix, MatrixClass.SPD, options={"forward_error_probe": report})
            != cache.key(matrix, MatrixClass.SPD, options={"forward_error_probe": refuse}))
    assert (cache.key(matrix, MatrixClass.SPD, options={"forward_error_probe": report})
            == cache.key(matrix, MatrixClass.SPD, options={"forward_error_probe": "report"}))
    first = cache.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": report})
    assert first.status == "ok"
    assert type(first.metadata["forward_error_probe_mode"]) is str
    assert first.metadata["forward_error_probe_mode"] == "report"
    second = cache.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": refuse})
    assert second is not first and cache.hits == 0
    assert second.metadata["forward_error_probe_mode"] == "refuse"
    third = cache.factorize(matrix, MatrixClass.SPD, options={"forward_error_probe": "report"})
    assert third is first and cache.hits == 1


@pytest.mark.parametrize("raw", [_AgreeingMode("bogus"), _DisplayedMode("Report"), b"report"])
def test_probe_mode_rejects_values_that_are_not_a_literal(raw) -> None:
    with pytest.raises(ValueError, match="forward_error_probe"):
        linalg._forward_error_probe_settings({"forward_error_probe": raw}, 3)
    with pytest.raises(ValueError, match="forward_error_probe"):
        SparseSolverBackend().factorize(sparse.eye(3, format="csc"), MatrixClass.SPD,
                                        options={"forward_error_probe": raw})


def test_descriptor_target_shift_invert_uses_report_mode(monkeypatch) -> None:
    """R2 #6 (decision 2a): the descriptor spectrum's target shift-invert uses report
    mode; its unshifted metric factor keeps the default."""

    import anyfesolver.algebraic_dynamics as algebraic
    from test_e4_pl_s3_descriptor_spectrum import _repeated_descriptor

    stiffness, mass = _repeated_descriptor()
    seen = []
    original = algebraic.factorize_cached

    def spy(matrix, matrix_class, **kwargs):
        seen.append((linalg._coerce_matrix_class(matrix_class), dict(kwargs.get("options") or {})))
        return original(matrix, matrix_class, **kwargs)

    monkeypatch.setattr(algebraic, "factorize_cached", spy)
    kwargs = dict(num_modes=4, dense_size_limit=1, algebraic_nullity=2, static_condensation_limit=1)
    spectrum = algebraic.solve_descriptor_spectrum(stiffness, mass, target_shift=2.5, **kwargs)
    assert spectrum.eigenvalues.size == 4
    assert seen == [(MatrixClass.SYMMETRIC_INDEFINITE, {"forward_error_probe": "report"})]
    seen.clear()
    algebraic.solve_descriptor_spectrum(stiffness, mass, **kwargs)
    assert seen == [(MatrixClass.SPD, {})]
