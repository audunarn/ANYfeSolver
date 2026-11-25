"""Iterative reduced solves must verify the true residual (option I1).

SciPy info == 0 alone is not trusted: MINRES reports success on ordinary
models with large true residuals.  RESIDUAL_VERIFIED certifies the residual
only, not the forward error.
"""
import warnings

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.assembly as asm
from anyfesolver import FEModel, LoadCase, solve_linear
from anyfesolver.assembly import _solve_reduced_system
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.elements import BeamElement, create_shell_element

SEC = {"area": 0.02, "Iy": 1.6667e-5, "Iz": 6.6667e-5, "J": 2.29e-5}


def _fix(model, specs):
    bc = {}
    for nodes, dofs in specs:
        for n in np.atleast_1d(nodes):
            d = bc.setdefault(int(n), {})
            for q in dofs:
                d[q] = 0.0
    for n, d in bc.items():
        model.add_boundary_condition(BoundaryCondition(f"bc{n}", [n], d))


def _beam_cantilever(n=50):
    m = FEModel("beam")
    m.add_material("mat", 2.1e11, 0.3, density=7850.0)
    for i in range(n + 1):
        m.add_node(i + 1, 10.0 * i / n, 0.0, 0.0)
    for i in range(n):
        m.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(SEC)))
    _fix(m, [([1], ["ux", "uy", "uz", "rx", "ry", "rz"])])
    lc = LoadCase("tip")
    lc.add_nodal_load(n + 1, [0, 1e3, -2e4, 0, 0, 0])
    return m, lc


def _grid_shell(name, pts, t, E, nu, rho=None):
    m = FEModel(name)
    if rho is None:
        m.add_material("mat", E, nu)
    else:
        m.add_material("mat", E, nu, density=rho)
    ni, nj = pts.shape[:2]
    ids = np.zeros((ni, nj), int)
    k = 0
    for i in range(ni):
        for j in range(nj):
            k += 1
            m.add_node(k, *map(float, pts[i, j]))
            ids[i, j] = k
    els = []
    e = 0
    for i in range(ni - 1):
        for j in range(nj - 1):
            e += 1
            quad = [int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1])]
            m.add_element(e, create_shell_element(e, quad, "mat", thickness=t))
            els.append(e)
    return m, ids, els


def _q4_plate(n=8):
    E, nu, t, a = 1.7472e7, 0.3, 1e-4, 2.0
    xs = np.linspace(0, a / 2, n + 1)
    P = np.array([[(x, y, 0.0) for y in xs] for x in xs])
    m, N, El = _grid_shell("pl", P, t, E, nu)
    edge = list(N[-1, :]) + list(N[:, -1])
    _fix(m, [(N.ravel(), ["ux", "uy", "rz"]), (edge, ["uz"]), (N[0, :], ["ry"]), (N[:, 0], ["rx"])])
    lc = LoadCase("uniform")
    for e in El:
        lc.add_pressure_load(int(e), 1e-4)
    return m, lc


def _scordelis(n=8):
    R, L, t, E = 25.0, 50.0, 0.25, 4.32e8
    th = np.radians(40.0)
    P = np.array([[(L / 2 * i / n, R * np.sin(th * j / n), R * np.cos(th * j / n))
                   for j in range(n + 1)] for i in range(n + 1)])
    m, N, _ = _grid_shell("sl", P, t, E, 0.0, rho=90.0 / t)
    _fix(m, [(N[0, :], ["uy", "uz"]), (N[-1, :], ["ux", "ry", "rz"]), (N[:, 0], ["uy", "rx", "rz"])])
    lc = LoadCase("g")
    lc.set_gravity(0.0, 0.0, -1.0)
    return m, lc


def _grillage(n=14, d=1.0):
    m = FEModel("gr")
    m.add_material("mat", 2.1e11, 0.3, density=7850.0)
    lc = LoadCase("lc")

    def nid(i, j):
        return 1 + i + n * j

    for j in range(n):
        for i in range(n):
            m.add_node(nid(i, j), d * i, d * j, 0.0)
    eid = 0
    for j in range(n):
        for i in range(n):
            pairs = []
            if i + 1 < n:
                pairs.append((nid(i, j), nid(i + 1, j)))
            if j + 1 < n:
                pairs.append((nid(i, j), nid(i, j + 1)))
            for a, b in pairs:
                eid += 1
                m.add_element(eid, BeamElement(eid, [a, b], "mat", dict(SEC, orientation=[0.0, 0.0, 1.0])))
    edge = [nid(i, j) for j in range(n) for i in range(n) if i in (0, n - 1) or j in (0, n - 1)]
    for f in edge:
        m.add_boundary_condition(BoundaryCondition(f"bz{f}", [f], {"uz": 0.0}))
    m.add_boundary_condition(BoundaryCondition("bin", [nid(0, 0)], {"ux": 0.0, "uy": 0.0, "rz": 0.0}))
    m.add_boundary_condition(BoundaryCondition("bin2", [nid(n - 1, 0)], {"uy": 0.0}))
    for j in range(n):
        for i in range(n):
            if (i, j) not in ((0, 0), (n - 1, 0)):
                m.add_boundary_condition(
                    BoundaryCondition(f"bi{nid(i, j)}", [nid(i, j)], {"ux": 0.0, "uy": 0.0, "rz": 0.0}))
    for j in range(1, n - 1, 3):
        for i in range(1, n - 1, 3):
            lc.add_nodal_load(nid(i, j), [0, 0, -1e3, 0, 0, 0])
    return m, lc


def _run_with_spy(builder, solver):
    model, lc = builder()
    seen = {}
    orig = asm._solve_reduced_system

    def spy(K, F, st, **kw):
        q, info = orig(K, F, st, **kw)
        seen["res"] = float(np.linalg.norm(K @ q - F) / np.linalg.norm(F))
        seen["info"] = info
        return q, info

    asm._solve_reduced_system = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, solver_info = solve_linear(model, lc, solver_type=solver)
        seen["solver_info"] = solver_info
    finally:
        asm._solve_reduced_system = orig
    return seen


def test_minres_inconsistent_2x2_is_not_converged():
    A = sparse.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    b = np.array([1.0, 1.0])  # b not in range(A): no solution exists
    q, info = _solve_reduced_system(A, b, "minres")
    assert info["status"] != "converged"
    assert info["relative_residual_disposition"] != "RESIDUAL_VERIFIED"
    assert info["relative_residual"] > 1.0e-8


@pytest.mark.parametrize("builder", [_beam_cantilever, _q4_plate, _scordelis, _grillage],
                         ids=["beam50", "q4plate8", "scordelis8", "grillage"])
@pytest.mark.parametrize("solver", ["minres", "gmres", "bicgstab"])
def test_converged_status_carries_a_verified_residual(builder, solver):
    seen = _run_with_spy(builder, solver)
    info = seen["info"]
    assert seen["solver_info"]["convergence_info"]["status"] == info["status"]
    assert (seen["solver_info"]["convergence_info"]["relative_residual_disposition"]
            == info["relative_residual_disposition"])
    if info["status"] == "converged":
        assert seen["res"] <= 1.0e-8 * (1 + 1e-6)
        assert info["relative_residual_disposition"] == "RESIDUAL_VERIFIED"
    else:
        assert info["status"] == "failed"
        assert info["relative_residual_disposition"] != "RESIDUAL_VERIFIED"
    assert info["relative_residual"] == pytest.approx(seen["res"], rel=1e-6, abs=1e-15)


@pytest.mark.parametrize("builder", [_beam_cantilever, _q4_plate, _scordelis],
                         ids=["beam50", "q4plate8", "scordelis8"])
def test_minres_ordinary_models_not_falsely_converged(builder):
    """Old code: MINRES said converged with true residual 1e-2..3e-1 here."""
    seen = _run_with_spy(builder, "minres")
    info = seen["info"]
    assert info["status"] != "converged" or seen["res"] <= 1.0e-8 * (1 + 1e-6)


@pytest.mark.parametrize("solver", ["minres", "bicgstab", "gmres"])
def test_well_posed_spd_stays_converged_and_verified(solver):
    n = 30
    A = sparse.diags([-1.0, 2.5, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    b = np.ones(n)
    q, info = _solve_reduced_system(A, b, solver)
    assert info["status"] == "converged"
    assert info["relative_residual_disposition"] == "RESIDUAL_VERIFIED"
    assert info["relative_residual"] <= info["relative_residual_limit"]
    assert "iterations" in info
    assert np.linalg.norm(A @ q - b) <= 1e-8 * np.linalg.norm(b) * (1 + 1e-6)


def test_zero_rhs_is_verified_trivially():
    A = sparse.identity(5, format="csr")
    q, info = _solve_reduced_system(A, np.zeros(5), "minres")
    assert info["status"] == "converged"
    assert info["relative_residual_disposition"] == "RESIDUAL_VERIFIED"


@pytest.mark.parametrize("scale", [1e-200, 1e-170])
@pytest.mark.parametrize("solver", ["minres", "gmres", "bicgstab"])
def test_tiny_rhs_underflow_is_not_verified(scale, solver, monkeypatch):
    """Old code: norms underflow to 0, so 0 <= max(0, atol) passed with relative residual 1.

    Tests the option-(a) part-based decision alone (range guard disabled); with the
    guard these inputs are refused as UNSUPPORTED_SCALE (see the range-guard tests)."""
    monkeypatch.setattr(asm, "_iterative_inputs_in_safe_range", lambda *a: True)
    A = sparse.identity(1, format="csr")
    b = np.array([scale])
    q, info = asm._verify_iterative_residual(A, b, np.zeros(1), 0, 1e-8, 0.0 if solver == "minres" else 1e-12)
    assert info["status"] == "failed"
    assert info["relative_residual_disposition"] == "RESIDUAL_LIMIT_EXCEEDED"
    assert info["relative_residual"] == pytest.approx(1.0)


def test_minres_tiny_rhs_end_to_end():
    A = sparse.identity(1, format="csr")
    q, info = _solve_reduced_system(A, np.array([1e-200]), "minres")
    assert info["relative_residual_disposition"] != "RESIDUAL_VERIFIED" or q[0] == pytest.approx(1e-200, rel=1e-6)
    assert info["status"] == "failed" or info["relative_residual"] <= 1e-8


def test_huge_rhs_does_not_overflow_and_is_judged_relatively(monkeypatch):
    # Option-(a) decision alone (range guard disabled; with it 1e200 is UNSUPPORTED_SCALE).
    monkeypatch.setattr(asm, "_iterative_inputs_in_safe_range", lambda *a: True)
    A = sparse.identity(3, format="csr")
    b = np.full(3, 1e200)
    _, info = asm._verify_iterative_residual(A, b, b.copy(), 0, 1e-8, 1e-12)
    assert info["status"] == "converged"
    assert info["relative_residual"] == 0.0
    _, info = asm._verify_iterative_residual(A, b, np.zeros(3), 0, 1e-8, 1e-12)
    assert info["status"] == "failed"
    assert info["relative_residual"] == pytest.approx(1.0)


def test_zero_rhs_nonzero_iterate_fails_and_zero_passes():
    A = sparse.identity(3, format="csr")
    _, info = asm._verify_iterative_residual(A, np.zeros(3), np.zeros(3), 0, 1e-8, 1e-12)
    assert info["status"] == "converged"
    _, info = asm._verify_iterative_residual(A, np.zeros(3), np.full(3, 1.0), 0, 1e-8, 1e-12)
    assert info["status"] == "failed"


def test_reported_limit_is_the_deciding_one():
    A = sparse.identity(1, format="csr")
    b = np.array([1.0])
    # relative residual 1e-10 -> accepted by rtol
    _, info = asm._verify_iterative_residual(A, b, np.array([1.0 - 1e-10]), 0, 1e-8, 1e-12)
    assert info["relative_residual_criterion"] == "rtol"
    assert info["relative_residual_limit"] == 1e-8
    assert info["residual_limit_abs"] == pytest.approx(1e-8)
    assert info["residual_norm"] == pytest.approx(1e-10, rel=1e-3)
    # rtol too tight, atol (1e-3 < |F|) accepts residual 1e-5
    _, info = asm._verify_iterative_residual(A, b, np.array([1.0 - 1e-5]), 0, 1e-8, 1e-3)
    assert info["status"] == "converged"
    assert info["relative_residual_criterion"] == "atol"
    assert info["residual_limit_abs"] == 1e-3
    assert info["relative_residual"] <= info["relative_residual_limit"]


def test_scipy_info_key_reported():
    A = sparse.diags([-1.0, 2.5, -1.0], [-1, 0, 1], shape=(10, 10), format="csr")
    _, info = _solve_reduced_system(A, np.ones(10), "gmres")
    assert info["scipy_info"] == 0 and info["iterations"] == info["scipy_info"]


# Review round 2 (Codex): compare norms in one common scale; report the applicable limit.
def _verify(K, F, q, info=0, rtol=1.0e-8, atol=0.0, range_guard=False):
    """Call the verifier directly.  ``range_guard=False`` disables the option-(b)
    safe-range refusal so the option-(a) part-based decision is tested on its own
    over the whole float range; ``True`` tests the shipped behaviour."""
    from unittest import mock

    import anyfesolver.assembly as assembly_module

    call = lambda: assembly_module._verify_iterative_residual(  # noqa: E731
        sparse.csr_matrix(K), np.asarray(F), np.asarray(q), info, rtol, atol
    )[1]
    if range_guard:
        return call()
    with mock.patch.object(assembly_module, "_iterative_inputs_in_safe_range", lambda *a: True):
        return call()


def test_subnormal_residual_is_not_rounded_into_acceptance():
    u = np.nextafter(0.0, 1.0)  # smallest subnormal
    F = np.array([1.2e8 * u, 0.0])
    q = np.array([1.2e8 * u - u, -u])  # K = I: residual (-u, -u), true r/|F| = sqrt(2)/1.2e8 = 1.18e-8 > rtol
    out = _verify(np.eye(2), F, q)
    assert out["relative_residual_disposition"] == "RESIDUAL_LIMIT_EXCEEDED"
    assert out["relative_residual"] > 1.0e-8


def test_near_overflow_rhs_with_exact_iterate_is_verified():
    F = np.array([1.4e308, 1.4e308])
    out = _verify(np.eye(2), F, F.copy())
    assert out["relative_residual_disposition"] == "RESIDUAL_VERIFIED"
    assert out["relative_residual"] == 0.0


def test_failed_check_reports_the_applicable_atol_threshold():
    F = np.array([2.0e-12])
    out = _verify(np.eye(1), F, np.array([0.0]), atol=1.0e-12)  # r = 2e-12 > max(rtol|F|, atol)
    assert out["relative_residual_disposition"] == "RESIDUAL_LIMIT_EXCEEDED"
    assert out["residual_limit_abs"] == 1.0e-12
    assert out["relative_residual_limit"] == pytest.approx(0.5)


def test_rejected_zero_rhs_reports_atol_as_limit():
    out = _verify(np.eye(1), np.array([0.0]), np.array([1.0]), atol=1.0e-12)
    assert out["relative_residual_disposition"] == "RESIDUAL_LIMIT_EXCEEDED"
    assert out["residual_limit_abs"] == 1.0e-12


def test_complex_input_is_refused_not_cast():
    out = _verify(np.eye(1), np.array([1.0 + 1.0j]), np.array([1.0 + 0.0j]))
    assert out["relative_residual_disposition"] == "UNSUPPORTED_DTYPE"
    assert out["status"] == "failed"


# Review round 3 (Codex) and owner option (a): separate-scale ratio, exact-oracle property test.
def test_nonzero_rhs_is_never_taken_for_zero():
    # r3 P2: K=[[5e187]], F=[1e-200], q=F: common-scale |F|/s underflowed to 0 -> "F == 0" -> certified.
    out = _verify(np.array([[5.0e187]]), np.array([1.0e-200]), np.array([1.0e-200]), atol=1.0e-12)
    assert out["relative_residual_disposition"] == "RESIDUAL_LIMIT_EXCEEDED"
    assert out["relative_residual"] > 1.0e180


def test_reported_abs_limit_is_finite_when_representable():
    # r3 P3: rtol*|F| ~ 2e300 must not be reported as inf.
    F = np.array([1.4e308, 1.4e308])
    out = _verify(np.eye(2), F, F.copy())
    assert np.isfinite(out["residual_limit_abs"])
    assert out["residual_limit_abs"] == pytest.approx(1.0e-8 * 1.4e308 * np.sqrt(2.0), rel=1e-12)


def _exact_decision(K, F, q, rtol, atol, r_float):
    """Exact rational verdict, or None when float rounding of r = Kq - F could decide either way."""
    from fractions import Fraction as Fr

    Kf = [[Fr(float(v)) for v in row] for row in K]
    Ff = [Fr(float(v)) for v in F]
    qf = [Fr(float(v)) for v in q]
    r = [sum(Kf[i][j] * qf[j] for j in range(len(qf))) - Ff[i] for i in range(len(Ff))]
    r2 = sum(v * v for v in r)
    f2 = sum(v * v for v in Ff)
    # Rounding bound of the float residual the verifier sees (|r_float - r| per entry).
    eps = Fr(2.0 ** -52)
    tiny = Fr(5.0e-324)
    bound = [
        4 * eps * (sum(abs(Kf[i][j] * qf[j]) for j in range(len(qf))) + abs(Ff[i])) + 4 * tiny
        for i in range(len(Ff))
    ]
    lo2 = sum(max(abs(v) - b, Fr(0)) ** 2 for v, b in zip(r, bound))
    hi2 = sum((abs(v) + b) ** 2 for v, b in zip(r, bound))
    rt, at = Fr(rtol), Fr(atol)

    def verdict(rr2):
        by_rtol = f2 > 0 and rr2 <= rt * rt * f2
        atol_allowed = f2 == 0 or at * at < f2
        return bool(by_rtol or (atol_allowed and rr2 <= at * at))

    lo, hi = verdict(lo2), verdict(hi2)
    return lo if lo == hi else None


@pytest.mark.parametrize("atol", [0.0, 1.0e-12])
def test_exact_oracle_property_sweep_over_exponents(atol):
    exps = list(range(-300, 301, 30)) + [-307, -310, -320, 305, 307]
    checked = mismatches = 0
    for ek in exps:
        for ef in exps:
            k = 10.0 ** ek if -308 <= ek <= 308 else 10.0 ** -300 * 10.0 ** (ek + 300)
            f = 10.0 ** ef if -308 <= ef <= 308 else 10.0 ** -300 * 10.0 ** (ef + 300)
            K = np.array([[k, 0.0], [0.0, 2.0 * k]])
            F = np.array([f, 3.0 * f])
            if not (np.all(np.isfinite(K)) and np.all(np.isfinite(F))):
                continue
            with np.errstate(all="ignore"):
                q_exact = F / np.diag(K)
            candidates = [q_exact, q_exact * (1.0 + 1.0e-6), np.zeros(2), F.copy(), q_exact * (1.0 + 3.0e-9)]
            for q in candidates:
                if not np.all(np.isfinite(q)):
                    continue
                with np.errstate(all="ignore"):
                    r_float = K @ q - F
                if not np.all(np.isfinite(r_float)):
                    continue
                expected = _exact_decision(K, F, q, 1.0e-8, atol, r_float)
                if expected is None:
                    continue
                out = _verify(K, F, q, atol=atol)
                got = out["relative_residual_disposition"] == "RESIDUAL_VERIFIED"
                checked += 1
                if got != expected:
                    mismatches += 1
                    raise AssertionError(f"k={k:g} f={f:g} q={q} expected {expected} got {out}")
                assert out["residual_limit_abs"] is None or not np.isnan(out["residual_limit_abs"])
    assert checked > 2000 and mismatches == 0


# Owner option (b) on top of (a): typed refusal outside the safe dynamic range.
def test_subnormal_residual_formation_is_refused_by_the_range_guard():
    # Codex round 4: fl(1.25*u - u) = 0 for the smallest subnormal u although |r|/|F| = 0.25.
    u = np.nextafter(0.0, 1.0)
    out = _verify(np.array([[1.25]]), np.array([u]), np.array([u]), range_guard=True)
    assert out["relative_residual_disposition"] == "UNSUPPORTED_SCALE"
    assert out["status"] == "failed"


@pytest.mark.parametrize("atol", [0.0, 1.0e-12])
def test_range_guard_property_sweep(atol):
    lo, hi = 1.0e-150, 1.0e150
    for ek in range(-300, 301, 30):
        for ef in range(-300, 301, 30):
            K = np.array([[10.0 ** ek, 0.0], [0.0, 2.0 * 10.0 ** ek]])
            F = np.array([10.0 ** ef, 3.0 * 10.0 ** ef])
            with np.errstate(all="ignore"):
                q = F / np.diag(K)
            if not np.all(np.isfinite(q)) or np.all(q == 0.0):
                continue
            out = _verify(K, F, q, atol=atol, range_guard=True)
            in_range = all(lo <= float(np.max(np.abs(v))) <= hi for v in (K, F, q))
            if in_range:
                assert out["relative_residual_disposition"] == "RESIDUAL_VERIFIED", (ek, ef, out)
            else:
                assert out["relative_residual_disposition"] == "UNSUPPORTED_SCALE", (ek, ef, out)


def test_si_unit_models_sit_inside_the_safe_range():
    from anyfesolver.assembly import _ITERATIVE_SAFE_RANGE, _iterative_inputs_in_safe_range

    lo, hi = _ITERATIVE_SAFE_RANGE
    seen = []
    for builder in (_beam_cantilever, _q4_plate, _scordelis, _grillage):
        model, load = builder()
        captured = {}

        def spy(K, F, solver_type, **kwargs):
            captured["K"], captured["F"] = K, F
            return original(K, F, solver_type, **kwargs)

        import anyfesolver.assembly as assembly_module

        original = assembly_module._solve_reduced_system
        assembly_module._solve_reduced_system = spy
        try:
            solve_linear(model, load, solver_type="direct")
        finally:
            assembly_module._solve_reduced_system = original
        K, F = captured["K"], captured["F"]
        q = __import__("scipy").sparse.linalg.spsolve(K.tocsc(), F)
        assert _iterative_inputs_in_safe_range(K, F, q)
        seen.append(max(float(abs(K).max()), float(np.abs(F).max()), float(np.abs(q).max())))
    assert lo < min(seen) and max(seen) < hi


def test_nonfinite_input_is_labelled_nonfinite_regardless_of_order():
    out = _verify(np.array([[1.0e151]]), np.array([np.nan]), np.array([1.0]), range_guard=True)
    assert out["relative_residual_disposition"] == "NONFINITE"
