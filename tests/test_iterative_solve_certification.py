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
            try:
                solve_linear(model, lc, solver_type=solver)
            except Exception:
                pass
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
