"""Coupled route S2: the r11 finding classes (design rev2 section 4) on the ANYfem-only entry, the
offset (u0 on a cut DOF) against a monolithic reference, the import rule
and coded refusals only. Every test fails on 642c43d64 (solid_coupling.py absent).

Outcome rules (spec S2): a refusal is required only where the test exhibits an exact S v = 0;
acceptance is required only where an independent kappa (the test's own equilibration and SVD) is at
most 1e7 and an independent candidate's exact backward residual has a 100x margin (review r1 #11: the
oracle never reads the route's diagnostics); otherwise an accepted result must lie within its
returned bounds of the reference."""

from __future__ import annotations

import math
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

import _coupled_fixtures as fx  # noqa: E402
from anyfesolver import BoundaryCondition  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from anyfesolver._coupled_certificate import CoupledSolveError  # noqa: E402
from anyfesolver.boundary import LoadCase  # noqa: E402

SIDES = ('side_0', 'side_1', 'side_2', 'side_3')
T = 0.02
HOLE = (1, 2, 1, 2)


def small(*, supports='simple', nu=fx.NU, modulus=fx.E, pressure=1000.0, hook=None, skin='end', case=None,
          prepared=None):
    model, grid, removed, footprint = fx.plate(3, thickness=T, hole=HOLE, supports=supports, nu=nu, modulus=modulus)
    if hook:
        hook(model, grid)
    if prepared is None:
        prepared = fx.prepare(fx.box_mesh(footprint, T, 2, 2, 1), nu=nu, modulus=modulus)
    if case is None:
        case = fx.pressure_case(removed, model, pressure) if skin else LoadCase(name='none')
    return model, grid, removed, prepared, case, sc.BoxFaces((2, 2, 1), skin, SIDES)


def run(*args, **kwargs):
    model, grid, removed, prepared, case, box = small(*args, **kwargs)
    return sc.solve_shell_solid_coupled(model, prepared, removed_element_ids=removed, box=box, load_case=case)


def coded(call, *args, **kwargs):
    with pytest.raises(CoupledSolveError) as info:
        call(*args, **kwargs)
    assert info.value.code.startswith('COUPLED_')
    return info.value


# --- independent oracle (review r1 #11, class 9): nothing below reads the route's diagnostics ----------

def exact_residual(S, x, b):
    """r = b - S x and the OD7 row reference sum_j |S_ij| ||x||_inf + |b_i|, in Fractions (test-local)."""
    S = S.tocsr()
    xq = [Fraction(float(v)) for v in x]
    x_inf = max((abs(v) for v in xq), default=Fraction(0))
    residual, reference = [], []
    for i in range(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        acc = Fraction(float(b[i]))
        row = Fraction(0)
        for k in range(lo, hi):
            s = Fraction(float(S.data[k]))
            acc -= s * xq[S.indices[k]]
            row += abs(s)
        residual.append(acc)
        reference.append(row * x_inf + abs(Fraction(float(b[i]))))
    return residual, reference


def reference_solution(S, b, steps=4, dps=60):
    """The S2 mpmath reference (review r2): x is held in mpmath at ``dps`` digits; each step adds a
    binary64 dense correction for the residual b - S x evaluated in mpmath at that precision. With
    kappa <= 1e7 every step gains about 9 digits, so 4 steps reach far below binary64; the reference
    is returned rounded to nearest binary64."""
    import mpmath
    with mpmath.workdps(dps):
        S = S.tocsr()
        dense = S.toarray()
        x = [mpmath.mpf(float(v)) for v in np.linalg.solve(dense, b)]
        entries = [[(int(S.indices[k]), mpmath.mpf(float(S.data[k]))) for k in range(S.indptr[i], S.indptr[i + 1])]
                   for i in range(S.shape[0])]
        bq = [mpmath.mpf(float(v)) for v in b]
        for _ in range(steps):
            r = [bq[i] - mpmath.fsum(s * x[j] for j, s in row) for i, row in enumerate(entries)]
            correction = np.linalg.solve(dense, np.array([float(v) for v in r]))
            x = [xi + mpmath.mpf(float(c)) for xi, c in zip(x, correction)]
        return np.array([float(v) for v in x])


def independent_condition(S, dps=50):
    """kappa_2 of the specified power-of-two Ruiz scaling A* = D S D, in high precision (review r3).

    The scaling (review r2: the spec's scaling, implemented here independently in pure Python per
    row): t_i -= floor(e_i / 2) with e_i the frexp exponent of the row maximum of 2^(t_i + t_j) |S_ij|,
    until no shift changes, applied to the original entries (exact: powers of two).

    The extreme singular values are computed as the r2 reference solution is: a binary64 SVD gives
    the starting triplets (u, sigma, v), and Newton steps on A v = sigma u, A^T u = sigma v,
    (v.v - 1)/2 = 0 use binary64 corrections of residuals evaluated in mpmath at ``dps`` digits. The
    returned kappa = sigma_1 / sigma_N is an mpmath number accurate to far below 1e-20 relative; the
    oracle raises (never decides) when a triplet does not converge or the extreme singular values
    are not separated from their neighbours, so the refined triplet is the extreme one."""
    import math
    import mpmath
    S = S.tocsr()
    rows = [[(int(S.indices[k]), float(S.data[k])) for k in range(S.indptr[i], S.indptr[i + 1])]
            for i in range(S.shape[0])]
    t = [0] * len(rows)
    for _ in range(65):
        shifts = []
        for i, row in enumerate(rows):
            largest = max(abs(math.ldexp(v, t[i] + t[j])) for j, v in row)
            shifts.append(-(math.frexp(largest)[1] // 2))
        if not any(shifts):
            break
        t = [a + s for a, s in zip(t, shifts)]
    n = len(rows)
    scaled = [[(j, math.ldexp(v, t[i] + t[j])) for j, v in row if v != 0.0] for i, row in enumerate(rows)]
    A = np.zeros(S.shape)
    for i, row in enumerate(scaled):
        for j, v in row:
            A[i, j] = v
    U, s, Vt = np.linalg.svd(A)
    separation = 1e-10 * s[0]
    if n > 1 and not (s[0] - s[1] > separation and s[-2] - s[-1] > separation):
        raise AssertionError('oracle: the extreme singular values are not separated; kappa undecided')
    with mpmath.workdps(dps):
        entries = [[(j, mpmath.mpf(v)) for j, v in row] for row in scaled]
        transposed = [[] for _ in range(n)]
        for i, row in enumerate(entries):
            for j, v in row:
                transposed[j].append((i, v))
        tolerance = mpmath.mpf(10) ** (8 - dps) * mpmath.mpf(float(s[0]))

        def triplet(k):
            u = [mpmath.mpf(float(v)) for v in U[:, k]]
            v = [mpmath.mpf(float(w)) for w in Vt[k]]
            sigma = mpmath.mpf(float(s[k]))
            for _ in range(8):
                r1 = [mpmath.fsum(a * v[j] for j, a in row) - sigma * u[i] for i, row in enumerate(entries)]
                r2 = [mpmath.fsum(a * u[i] for i, a in col) - sigma * v[j] for j, col in enumerate(transposed)]
                r3 = (mpmath.fsum(w * w for w in v) - 1) / 2
                if max(max(abs(x) for x in r1 + r2), abs(r3)) <= tolerance:
                    return sigma
                sf, uf, vf = float(sigma), np.array([float(x) for x in u]), np.array([float(x) for x in v])
                J = np.zeros((2 * n + 1, 2 * n + 1))
                J[:n, :n] = -sf * np.eye(n)
                J[:n, n:2 * n] = A
                J[:n, 2 * n] = -uf
                J[n:2 * n, :n] = A.T
                J[n:2 * n, n:2 * n] = -sf * np.eye(n)
                J[n:2 * n, 2 * n] = -vf
                J[2 * n, n:2 * n] = vf
                rhs = -np.array([float(x) for x in r1 + r2 + [r3]])
                step = np.linalg.solve(J, rhs)
                u = [a + mpmath.mpf(float(d)) for a, d in zip(u, step[:n])]
                v = [a + mpmath.mpf(float(d)) for a, d in zip(v, step[n:2 * n])]
                sigma = sigma + mpmath.mpf(float(step[2 * n]))
            raise AssertionError(f'oracle: singular triplet {k} did not converge; kappa undecided')
        return triplet(0) / triplet(n - 1)


# Owner decision OD11 (2026-11-20, review r3): the specified cutoff, exactly; no margin.
KAPPA_LIMIT = 10 ** 7


def acceptance_warranted(S, b):
    """S2: acceptance is required where kappa(A*) of the specified scaling is at most 1e7 (the exact
    cutoff, decided on the high-precision kappa) and the mpmath reference has a 100x backward margin
    (|r_i| <= 1e-12 ref_i, exact)."""
    x = reference_solution(S, b)
    residual, reference = exact_residual(S, x, b)
    margin = all(abs(r) <= Fraction(1, 10 ** 12) * ref for r, ref in zip(residual, reference))
    return bool(independent_condition(S) <= KAPPA_LIMIT) and margin, x


def all_finite(result):
    arrays = [result.shell_displacements, result.solid_coefficients, result.multipliers,
              result.shell_interface_forces, result.solid_interface_forces]
    arrays += [s.force for s in result.side_resultants] + [s.moment for s in result.side_resultants]
    ok = all(np.isfinite(b.values).all() and np.isfinite(b.bounds).all() for b in arrays)
    numbers = [v for v in list(result.certificate.values()) + list(result.verification.values())
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return ok and np.isfinite(result.reactions).all() and all(math.isfinite(v) for v in numbers)


# --- r11 #1, #5, #8, #9, #10: no caller arrays or protocol objects reach the route -------------------

def test_r11_1_and_8_prepared_must_be_exact_type():
    from anysolid.backend import PreparedSolid

    class Sub(PreparedSolid):
        pass
    model, grid, removed, prepared, case, box = small()
    fake = Sub.__new__(Sub)
    error = coded(sc.solve_shell_solid_coupled, model, fake, removed_element_ids=removed, box=box, load_case=case)
    assert error.reason == 'prepared_type'


def test_r11_1_integer_or_coo_blocks_never_reach_formation():
    one = sparse.csr_matrix(np.eye(1))
    kwargs = dict(K=one, F=np.ones(1), K_s=one, f_s=np.ones(1), T=one, u0=np.zeros(1), C_shell=one, C_solid=one,
                  K_v=one, f_v=np.zeros(1))
    for bad in (sparse.csr_matrix(np.array([[1]], dtype=np.int8)), sparse.coo_matrix(np.eye(1))):
        assert coded(sc._form_system, **{**kwargs, 'K_v': bad}).code == 'COUPLED_FORMATION_RANGE'


@pytest.mark.parametrize('pressure', [np.ma.masked_array([1000.0]), np.longdouble(1000.0), np.array(1000.0), 1000])
def test_r11_5_9_masked_wide_or_array_pressure_refuses(pressure):
    model, grid, removed, prepared, case, box = small()
    case.pressure_loads[removed[0]] = pressure
    error = coded(sc.solve_shell_solid_coupled, model, prepared, removed_element_ids=removed, box=box, load_case=case)
    assert error.code == 'COUPLED_LOAD_UNSUPPORTED'


@pytest.mark.parametrize('load_case', ['default', ('default',), None])
def test_r11_10_load_case_must_be_one_load_case(load_case):
    model, grid, removed, prepared, case, box = small()
    error = coded(sc.solve_shell_solid_coupled, model, prepared, removed_element_ids=removed, box=box,
                  load_case=load_case)
    assert error.reason == 'load_case_type'


# --- r11 #2: the affine offset is carried (u0 on a cut DOF) against a monolithic reference --------------

def test_r11_2_offset_on_cut_dof_matches_monolithic_reference(monkeypatch):
    captured = {}
    original = sc._form_system

    def capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(sc, '_form_system', capture)
    delta = 1e-4

    def hook(model, grid):
        model.add_boundary_condition(BoundaryCondition(name='lift', node_ids=[grid[1, 1]], dof_constraints={'uz': delta}))
    result = run(hook=hook)
    K, F, C_shell, C_solid, K_v, f_v, u0 = (captured[k] for k in ('K', 'F', 'C_shell', 'C_solid', 'K_v', 'f_v', 'u0'))
    T = captured['T']
    dependent = np.flatnonzero(np.diff(T.tocsr().indptr) == 0)
    n, n_v, m = K.shape[0], K_v.shape[0], C_shell.shape[0]
    G = sparse.csr_matrix((np.ones(len(dependent)), (np.arange(len(dependent)), dependent)), shape=(len(dependent), n))
    A = sparse.bmat([[K, None, C_shell.T, G.T], [None, K_v, -C_solid.T, None], [C_shell, -C_solid, None, None],
                     [G, None, None, None]], format='csr')
    A.sum_duplicates()
    b = np.concatenate([F, f_v, np.zeros(m), u0[dependent]])
    x = reference_solution(A, b)                 # test-local exact residual (r1 #11: no route code)
    assert u0[captured['C_shell'].indices].any(), 'the prescribed DOF must be a connector column'
    order = np.concatenate([np.arange(6 * i, 6 * i + 6) for i in range(len(result.shell_node_ids))])
    u = result.shell_displacements
    scale = np.abs(x[:n]).max()
    assert np.all(np.abs(u.values - x[:n][order]) <= u.bounds * (1 + 1e-6) + 1e-14 * scale)
    lam = result.multipliers
    assert np.all(np.abs(lam.values - x[n + n_v:n + n_v + m]) <= lam.bounds * (1 + 1e-6)
                  + 1e-14 * np.abs(x[n + n_v:n + n_v + m]).max())
    assert np.all(np.abs(result.solid_coefficients.values - x[n:n + n_v]) <= result.solid_coefficients.bounds
                  * (1 + 1e-6) + 1e-14 * np.abs(x[n:n + n_v]).max())


# --- r11 #3, #4, #11: reactions carry no claim; accepted results are finite --------------------------

def test_r11_3_reactions_carry_no_claim_and_4_11_accepted_result_is_finite(monkeypatch):
    captured = {}
    original = sc.certified_solve

    def spy(system, rhs, **kwargs):
        captured.update(S=system.copy(), b=rhs.copy())
        return original(system, rhs, **kwargs)
    monkeypatch.setattr(sc, 'certified_solve', spy)
    model, grid, removed, prepared, case, box = small()
    try:
        result = sc.solve_shell_solid_coupled(model, prepared, removed_element_ids=removed, box=box, load_case=case)
    except CoupledSolveError as exc:
        result = exc
    warranted, x = acceptance_warranted(captured['S'], captured['b'])
    assert warranted, 'the fixture must be independently well conditioned (S2 acceptance case)'
    assert not isinstance(result, CoupledSolveError), (result.code, result.reason)
    assert type(result.reactions) is np.ndarray            # no bounds: not a claim (O12)
    assert np.abs(result.reactions).max() > 0
    assert all_finite(result)
    # The claims against the independent reference solution of the stored system.
    n_s, n_v, m = result.sizes['n_s'], result.sizes['n_v'], result.sizes['m']
    for bounded, exact in ((result.solid_coefficients, x[n_s:n_s + n_v]), (result.multipliers, x[n_s + n_v:])):
        assert np.all(np.abs(bounded.values - exact) <= bounded.bounds * (1 + 1e-6) + 1e-15 * np.abs(exact).max())


def test_r1_11_claims_cover_a_perturbed_candidate(monkeypatch):
    """Review r1 #11: the bounds are judged against an independent reference, for a candidate that
    is wrong by +-100 ulp per entry (still backward stable), so a bound that is too small fails."""
    from anyfesolver import _coupled_certificate as cc
    captured = {}
    original = sc.certified_solve

    def spy(system, rhs, **kwargs):
        captured.update(S=system.copy(), b=rhs.copy())
        return original(system, rhs, **kwargs)

    def perturbed(system, rhs):
        x = reference_solution(system, rhs)
        signs = np.where(np.arange(len(x)) % 2, 1.0, -1.0)
        return x + 100 * signs * np.spacing(np.abs(x))
    monkeypatch.setattr(sc, 'certified_solve', spy)
    monkeypatch.setattr(cc, '_candidate', perturbed)
    result = run()
    x = reference_solution(captured['S'], captured['b'])
    n_s, n_v = result.sizes['n_s'], result.sizes['n_v']
    for bounded, exact in ((result.solid_coefficients, x[n_s:n_s + n_v]), (result.multipliers, x[n_s + n_v:])):
        error = np.abs(bounded.values - exact)
        assert error.max() > 0
        assert np.all(error <= bounded.bounds + 2 * np.spacing(np.abs(exact)))


def test_power_of_two_scaling_keeps_the_decision_and_the_bounds():
    base = run()
    scaled = run(modulus=fx.E * 2.0 ** -20, pressure=1000.0 * 2.0 ** -20)
    a, b = base.shell_displacements, scaled.shell_displacements
    assert np.all(np.abs(a.values - b.values) <= a.bounds + b.bounds)


# --- r11 #6, #7: silently ignored targets and uncoded native refusals ------------------------------

def test_r11_6_added_mass_on_unknown_node_refuses():
    model, grid, removed, prepared, case, box = small()
    case.added_node_masses[777] = 1.0
    error = coded(sc.solve_shell_solid_coupled, model, prepared, removed_element_ids=removed, box=box, load_case=case)
    assert error.code == 'COUPLED_TARGET_REMOVED'


def test_r11_7_contradictory_supports_are_coded():
    def hook(model, grid):
        model.add_boundary_condition(BoundaryCondition(name='a', node_ids=[grid[0, 1]], dof_constraints={'ux': 0.0}))
        model.add_boundary_condition(BoundaryCondition(name='b', node_ids=[grid[0, 1]], dof_constraints={'ux': 1.0}))
    assert coded(run, hook=hook).code == 'COUPLED_NATIVE_INVALID'


@pytest.mark.parametrize('pressure', [1e300, math.inf, math.nan])
def test_r11_7_extreme_pressure_is_coded(pressure):
    coded(run, pressure=pressure)


# --- r1-r10: singular or mechanism systems refuse (exact S v = 0 exhibited) -------------------------

@pytest.mark.parametrize('supports', [None, 'uz_only'])
def test_r1_r10_mechanism_refuses(supports):
    def hook(model, grid):
        if supports == 'uz_only':
            edge = sorted({grid[i, j] for (i, j) in grid if i in (0, 3) or j in (0, 3)})
            model.add_boundary_condition(BoundaryCondition(name='ss', node_ids=edge, dof_constraints={'uz': 0.0}))
    # the rigid in-plane motions (all of them without supports) are exact null vectors of S
    error = coded(run, supports='none', hook=hook)
    assert error.code in ('COUPLED_UNCERTIFIED', 'COUPLED_MECHANISM')


# --- OD8 (2026-11-19): V4 interface balance against the per-side normwise reference ------------------

def _rows(*rows):
    return [[Fraction(v) for v in row] for row in rows]


def test_od8_v4_noise_component_is_judged_against_the_side_norm():
    # Row 0 carries a large x force; row 1 a noise-sized y force whose two sides differ by 2^-10 of
    # its own terms. The per-component reference of 862d099d0 refuses (ratio ~4.9e-4 > 1e-10); the
    # OD8 per-side reference sum_i max_r(|CsRs|+|CvRv|)_ir |lam_i| accepts (ratio ~3.5e-18).
    CRs = _rows([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])
    CRv = _rows([1, 0, 0, 0, 0, 0], [0, 1 + Fraction(1, 2 ** 10), 0, 0, 0, 0])
    lam = [Fraction(2 ** 17), Fraction(1, 2 ** 30)]
    worst = sc._v4_side_balance('side_0', CRs, CRv, lam)
    reference = 2 * Fraction(2 ** 17) + (2 + Fraction(1, 2 ** 10)) * Fraction(1, 2 ** 30)
    assert worst == Fraction(1, 2 ** 40) / reference


def test_od8_v4_threshold_is_decided_exactly_and_a_real_imbalance_refuses():
    tol = sc.V4_BALANCE_TOLERANCE
    delta = 2 * tol / (1 - tol)                 # imbalance == tol * (2 + delta) exactly
    lam = [Fraction(1)]
    CRs = _rows([1, 0, 0, 0, 0, 0])
    assert sc._v4_side_balance('s', CRs, [[1 + delta] + [Fraction(0)] * 5], lam) == tol
    bump = delta + Fraction(1, 10 ** 30)
    with pytest.raises(CoupledSolveError) as info:
        sc._v4_side_balance('s', CRs, [[1 + bump] + [Fraction(0)] * 5], lam)
    assert (info.value.code, info.value.reason) == ('COUPLED_UNCERTIFIED', 'v4_interface_balance')
    # a moment imbalance is judged against the same side reference (no per-component exemption)
    CRm = _rows([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1])
    with pytest.raises(CoupledSolveError):
        sc._v4_side_balance('s', CRm, _rows([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, Fraction(11, 10)]),
                            [Fraction(1), Fraction(1)])


def test_od8_v4_zero_reference_is_exempt():
    CRs = _rows([1, 2, 3, 4, 5, 6])
    assert sc._v4_side_balance('s', CRs, _rows([1, 2, 3, 4, 5, 7]), [Fraction(0)]) == 0


def test_nu_zero_patch_test():
    """Uniform stretch u_x = eps x imposed on the outer edge, nu = 0, no load: the exact solution is
    u_x = eps x, every other shell DOF zero, the solid coefficients eps x on x and zero otherwise. It
    refused on 862d099d0 at V4 (OD8); it must be accepted with a patch error <= 1e-13 (1e-9 eps)."""
    eps = 1e-4

    def hook(model, grid):
        for (i, j), nid in grid.items():
            if i in (0, 3) or j in (0, 3):
                x = model.mesh.nodes[nid].x
                model.add_boundary_condition(BoundaryCondition(
                    name=f'p{nid}', node_ids=[nid],
                    dof_constraints={'ux': eps * x, 'uy': 0.0, 'uz': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0}))
    result = run(supports='none', nu=0.0, skin=None, hook=hook)
    assert result.certificate['accepted']
    assert result.verification['v4_balance_max'] <= float(sc.V4_BALANCE_TOLERANCE)
    model, grid, *_ = small(supports='none', nu=0.0, skin=None)
    x = np.array([model.mesh.nodes[nid].x for nid in result.shell_node_ids])
    u = result.shell_displacements.values.reshape(-1, 6)
    assert np.abs(u[:, 0] - eps * x).max() <= 1e-13
    assert np.abs(u[:, 1:]).max() <= 1e-13
    prepared = small(nu=0.0)[3]
    coords = np.asarray(prepared.coefficient_coordinates)
    comps = np.asarray(prepared.coefficient_components)
    expected = np.where(comps == 0, eps * coords[:, 0], 0.0)
    assert np.abs(result.solid_coefficients.values - expected).max() <= 1e-13


# --- further S2 tests ---------------------------------------------------------------------------------

def test_module_is_not_imported_or_exported():
    code = ("import sys, anyfesolver; "
            "assert 'anyfesolver.solid_coupling' not in sys.modules; "
            "assert 'anyfesolver._coupled_certificate' not in sys.modules; "
            "assert not hasattr(anyfesolver, 'solve_shell_solid_coupled')")
    subprocess.run([sys.executable, '-c', code], check=True, timeout=600)


@pytest.mark.parametrize('mutation', ['model', 'removed_float', 'removed_unknown', 'box_tuple', 'box_skin',
                                      'divisions', 'cancel_bad', 'cancelled', 'removed_empty'])
def test_no_uncoded_exception_escapes(mutation):
    model, grid, removed, prepared, case, box = small()
    kwargs = dict(removed_element_ids=removed, box=box, load_case=case)
    target = model
    if mutation == 'model':
        target = 'not a model'
    elif mutation == 'removed_float':
        kwargs['removed_element_ids'] = [float(removed[0])]
    elif mutation == 'removed_unknown':
        kwargs['removed_element_ids'] = [10 ** 6]
    elif mutation == 'removed_empty':
        kwargs['removed_element_ids'] = []
    elif mutation == 'box_tuple':
        kwargs['box'] = ((2, 2, 1), 'end', SIDES)
    elif mutation == 'box_skin':
        kwargs['box'] = sc.BoxFaces((2, 2, 1), 'top', SIDES)
    elif mutation == 'divisions':
        kwargs['box'] = sc.BoxFaces((2, 2.0, 1), 'end', SIDES)
    elif mutation == 'cancel_bad':
        kwargs['cancel'] = 5
    elif mutation == 'cancelled':
        kwargs['cancel'] = lambda: True
    error = coded(sc.solve_shell_solid_coupled, target, prepared, **kwargs)
    if mutation == 'cancelled':
        assert error.code == 'COUPLED_CANCELLED'


def test_caller_model_is_not_mutated():
    model, grid, removed, prepared, case, box = small()
    before = (sorted(model.mesh.elements), sorted(model.mesh.nodes), model.mesh.dof_manager.total_dofs,
              dict(case.pressure_loads))
    sc.solve_shell_solid_coupled(model, prepared, removed_element_ids=removed, box=box, load_case=case)
    assert before == (sorted(model.mesh.elements), sorted(model.mesh.nodes), model.mesh.dof_manager.total_dofs,
                      dict(case.pressure_loads))
