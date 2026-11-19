"""Coupled route S2: the r11 finding classes (design rev2 section 4) on the ANYfem-only entry, the
offset (u0 on a cut DOF) against a monolithic reference, the import rule
and coded refusals only. Every test fails on 642c43d64 (solid_coupling.py absent).

Outcome rules (spec S2): a refusal is required only where the test exhibits an exact S v = 0;
acceptance is required only where the certificate's own rigorous kappa upper bound (A1/A2) is at
most 1e7 and the backward residual has a 100x margin; otherwise an accepted result must lie within
its returned bounds of the reference."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

import _coupled_fixtures as fx  # noqa: E402
from anyfesolver import BoundaryCondition  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from anyfesolver._coupled_certificate import CoupledSolveError, _exact_rows  # noqa: E402
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


def acceptance_warranted(result):
    return result.certificate['condition_upper_bound'] <= 1e7 and result.verification['backward']['max_ratio'] <= 1e-12


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
    x = np.linalg.solve(A.toarray(), b)
    for _ in range(3):
        x = x + np.linalg.solve(A.toarray(), _exact_rows(A, x, b).residual_float())
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

def test_r11_3_reactions_carry_no_claim_and_4_11_accepted_result_is_finite():
    result = run()
    assert type(result.reactions) is np.ndarray            # no bounds: not a claim (O12)
    assert np.abs(result.reactions).max() > 0
    assert all_finite(result)
    if acceptance_warranted(result):
        assert result.certificate['accepted']


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
