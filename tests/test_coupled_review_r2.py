"""Coupled route: regression tests for Codex code review round 2 (4 P1 + 7 P2) under the narrowed
contract of owner decision OD9 (docs/SOLID_COUPLING.md, "Trusted-construction boundary"). Each test
names its finding; the fails-on-cb33c1da0 evidence is in the commit message."""

from __future__ import annotations

import contextlib
import ctypes
import inspect
import struct
import sys
import threading
import types
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

import _coupled_fixtures as fx  # noqa: E402
from anyfesolver import BoundaryCondition  # noqa: E402
from anyfesolver import _coupled_certificate as cc  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from anyfesolver.boundary import LoadCase  # noqa: E402
from anyfesolver.elements import create_shell_element  # noqa: E402
from test_coupled_review_r1 import SIDES, T, HOLE, TINY, flushing, refused, small, solve  # noqa: E402


def nodal_case(grid, values=(0.0, 0.0, 1234.0, 0.0, 0.0, 0.0)):
    case = LoadCase(name='point')
    case.nodal_loads[grid[1, 1]] = np.array(values, dtype=float)        # a free cut corner
    return case


@contextlib.contextmanager
def rounding(mode):
    """Directed rounding on this thread through the MSVC runtime (_RC_DOWN 0x100, _RC_UP 0x200,
    _RC_CHOP 0x300 under _MCW_RC 0x300), restored afterwards."""
    if sys.platform != 'win32':
        pytest.skip('the rounding mode is switched through the MSVC runtime')
    crt = ctypes.CDLL('ucrtbase')
    crt._controlfp.restype = ctypes.c_uint
    crt._controlfp.argtypes = [ctypes.c_uint, ctypes.c_uint]
    old = crt._controlfp(0, 0)
    crt._controlfp(mode, 0x300)
    try:
        yield
    finally:
        crt._controlfp(old, 0x300)
    assert cc._nearest_rounding()


# --- P1 (a): the removed set is derived from the footprint geometry ------------------------------------

def _coincident_pair(model, grid):
    """Two coincident, equally oriented Q4s on their own nodes outside the plate: every extra edge
    occurs twice in the removed set, so the cut loop is unchanged (the reviewer's construction)."""
    for k, (x, y) in enumerate(((1.5, 0.0), (1.8, 0.0), (1.8, 0.3), (1.5, 0.3))):
        model.add_node(900 + k, x, y, 0.0)
    for eid in (1000, 1001):
        model.add_element(eid, create_shell_element(eid, [900, 901, 902, 903], 'steel', thickness=T))


def test_r2_p1a_removed_set_with_an_unrelated_patch_refuses():
    model, grid, removed, prepared, _, box = small(skin=None, hook=_coincident_pair)
    refused('COUPLED_CUT_MISMATCH', solve, model, prepared, removed + [1000, 1001], box, nodal_case(grid),
            reason='removed_set_geometry')
    model, grid, removed, prepared, _, box = small(skin=None)
    solve(model, prepared, removed, box, nodal_case(grid))  # without the patch the same case is accepted


# --- P1 (b): the coefficient layout of the P2 topology ---------------------------------------------------

def test_r2_p1b_relabelled_p1_basis_refuses():
    model, grid, removed, footprint = fx.plate(3, thickness=T, hole=HOLE)
    prepared = fx.prepare(fx.box_mesh(footprint, T, 2, 2, 2), order=1)
    object.__setattr__(prepared, 'order', 2)                       # the label only; basis and arrays P1
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, sc.BoxFaces((2, 2, 2), 'end', SIDES),
            fx.pressure_case(removed, model), reason='prepared_layout')


# --- P1 (c): nonzero tests on value bits, not stored structure --------------------------------------------

def _fixed_corner(model, grid):
    model.add_boundary_condition(BoundaryCondition(name='cut-corner', node_ids=[grid[1, 1]],
                                                   dof_constraints={'ux': 0.0}))


@pytest.mark.parametrize('stored_zero', [False, True])
def test_r2_p1c_explicit_csr_zero_does_not_activate_a_fixed_connector_column(monkeypatch, stored_zero):
    from anyfesolver import assembly
    original = assembly.build_constraint_transformation
    model, grid, removed, prepared, case, box = small(hook=_fixed_corner)

    def with_stored_zero(K, F, holed):
        K_s, f_s, Tm, u0, independent, info = original(K, F, holed)
        if stored_zero:
            Tm = Tm.tocsr()
            empty = int(holed.mesh.nodes[grid[1, 1]].dofs[0])             # the fixed cut-corner ux row
            assert np.diff(Tm.indptr)[empty] == 0
            coo = Tm.tocoo()
            rows = np.concatenate([coo.row, [empty]])
            cols = np.concatenate([coo.col, [0]])
            data = np.concatenate([coo.data, [0.0]])
            order = np.lexsort((cols, rows))
            indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=Tm.shape[0]))])
            Tm = sparse.csr_matrix((data[order], cols[order].astype(np.int32), indptr.astype(np.int32)), shape=Tm.shape)
            assert Tm.nnz == coo.nnz + 1 and Tm.has_canonical_format      # the explicit 0.0 is stored
        return K_s, f_s, Tm, u0, independent, info
    monkeypatch.setattr(assembly, 'build_constraint_transformation', with_stored_zero)
    refused('COUPLED_TARGET_REMOVED', solve, model, prepared, removed, box, case, reason='connector_column')


# --- P1 (d): raw load scalar types before native assembly; the zero test on the original values ----------

@pytest.mark.parametrize('value', [[Fraction(1, 2 ** 1100)], [True, 0.0, 0.0], ['1.0'], np.array([1.0], dtype=np.float32)],
                         ids=['fraction', 'bool', 'str', 'float32'])
def test_r2_p1d_unsupported_nodal_scalar_refuses_before_assembly(monkeypatch, value):
    from anyfesolver import assembly
    called = []
    original = assembly.assemble_system
    monkeypatch.setattr(assembly, 'assemble_system', lambda *a, **k: called.append(1) or original(*a, **k))
    model, grid, removed, prepared, _, box = small(skin=None)
    case = LoadCase(name='raw')
    case.nodal_loads[grid[0, 1]] = value
    refused('COUPLED_LOAD_UNSUPPORTED', solve, model, prepared, removed, box, case, reason='raw_scalar_type')
    assert not called


def test_r2_p1d_integer_nodal_load_is_admitted():
    model, grid, removed, prepared, _, box = small(skin=None)
    case = LoadCase(name='int')
    case.nodal_loads[grid[1, 1]] = [0, 0, 1234]
    result = solve(model, prepared, removed, box, case)
    reference = solve(model, prepared, removed, box, nodal_case(grid))
    assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


def test_r2_p1d_unsupported_support_scalar_refuses():
    # (MPC right-hand sides and coefficients are stored as float by ConstraintEquation itself.)
    def hook(model, grid):
        model.add_boundary_condition(BoundaryCondition(name='frac', node_ids=[grid[0, 1]],
                                                       dof_constraints={'ux': Fraction(1, 2 ** 1100)}))
    model, grid, removed, prepared, case, box = small(skin=None, hook=hook)
    refused('COUPLED_LOAD_UNSUPPORTED', solve, model, prepared, removed, box, case, reason='raw_scalar_type')


# --- P2: the FP environment of the certificate --------------------------------------------------------------

@pytest.mark.parametrize('mode', [0x100, 0x200, 0x300], ids=['down', 'up', 'chop'])
def test_r2_fp_directed_rounding_refuses(mode):
    model, grid, removed, prepared, case, box = small()
    with rounding(mode):
        assert not cc._nearest_rounding()
        refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case, reason='rounding_mode')
    solve(model, prepared, removed, box, case)


def test_r2_fp_flush_enabled_only_during_certification_refuses():
    """A clock callback switches FTZ/DAZ on for the certificate and off before the next poll."""
    stack = contextlib.ExitStack()
    state = {'calls': 0}

    def clock():
        caller = inspect.currentframe().f_back
        while caller.f_code.co_name == 'tick':          # the route's checked-clock wrapper (review r3)
            caller = caller.f_back
        if caller.f_code.co_name == 'certified_solve':
            state['calls'] += 1
            if state['calls'] == 1:
                stack.enter_context(flushing())
            elif state['calls'] == 2:
                stack.close()
        return 0.0
    model, grid, removed, prepared, case, box = small()
    try:
        refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case, clock=clock,
                reason='subnormal_flush')
    finally:
        stack.close()


# --- P2: overlap without positive tolerance, with a geometric prefilter -------------------------------------

def test_r2_overlap_a_loaded_kept_q4_past_the_rim_refuses():
    def hook(model, grid):
        x0, w = 1 / 3, 1 / 3
        reach = x0 + 5e-10 * w                                       # 5e-10 of the footprint width inside
        corners = ((x0 - 0.1, x0 + 0.2 * w), (reach, x0 + 0.2 * w), (reach, x0 + 0.4 * w), (x0 - 0.1, x0 + 0.4 * w))
        for k, (x, y) in enumerate(corners):
            model.add_node(800 + k, x, y, 0.0)
        model.add_element(1100, create_shell_element(1100, [800, 801, 802, 803], 'steel', thickness=T))
        model.add_boundary_condition(BoundaryCondition(name='hold', node_ids=[800, 801, 802, 803], dof_constraints={
            k: 0.0 for k in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    refused('COUPLED_CUT_MISMATCH', lambda: solve(*_small_with(hook)), reason='kept_inside_footprint')


def _small_with(hook):
    model, grid, removed, prepared, case, box = small(hook=hook)
    return model, prepared, removed, box, case


def test_r2_overlap_rule_is_hull_exclusion_a_warped_kept_q4_refuses():
    """Documented rule (docs/SOLID_COUPLING.md): the convex hull decides. This Q4 has corner heights
    0, 0.1, 0, 0.1 above the footprint; its bilinear surface clears the solid, its hull does not."""
    def hook(model, grid):
        for k, (x, y, z) in enumerate(((-1.0, -1.0, 0.0), (2.0, -1.0, 0.1), (2.0, 2.0, 0.0), (-1.0, 2.0, 0.1))):
            model.add_node(700 + k, x, y, z)
        model.add_element(1200, create_shell_element(1200, [700, 701, 702, 703], 'steel', thickness=T))
        model.add_boundary_condition(BoundaryCondition(name='hold', node_ids=[700, 701, 702, 703], dof_constraints={
            k: 0.0 for k in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    refused('COUPLED_CUT_MISMATCH', lambda: solve(*_small_with(hook)), reason='kept_inside_footprint')


def test_r2_overlap_prefilter_keeps_an_exact_interior_point():
    """The reviewer's helper case: a = (1,2,3), b = (2,2,-2), p = (a+b)/8 lies exactly at h = 0; a float
    solve gave h ~ 4e-17 and skipped it for thickness 1e-19."""
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, -2.0])
    n = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    node = types.SimpleNamespace(coords=lambda: tuple((a + b) / 8))
    model = types.SimpleNamespace(mesh=types.SimpleNamespace(nodes={1: node},
                                                             elements={7: types.SimpleNamespace(node_ids=[1])}))
    corners = (np.zeros(3), a, a + b, b)
    refused('COUPLED_CUT_MISMATCH', sc._check_footprint_partition, model, (), (), corners, n, 1e-19,
            reason='kept_inside_footprint')


# --- P2: the zero-load rule counts only consumed prescriptions --------------------------------------------

def test_r2_zero_load_support_without_nodes_is_not_a_load():
    def hook(model, grid):
        model.add_boundary_condition(BoundaryCondition(name='empty', node_ids=[], dof_constraints={'uz': 0.25}))
    model, grid, removed, prepared, case, box = small(skin=None, hook=hook)
    result = solve(model, prepared, removed, box, case)
    assert not np.any(result.shell_displacements.values) and not np.any(result.multipliers.values)


# --- P2: cancellation during result construction ------------------------------------------------------------

def test_r2_cancellation_during_result_construction_refuses(monkeypatch):
    event = threading.Event()
    original = sc.CoupledSolution

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        event.set()
        return result
    monkeypatch.setattr(sc, 'CoupledSolution', build)
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_CANCELLED', solve, model, prepared, removed, box, case, cancel=event)


# --- P2: exception translation calls no user-defined formatting -------------------------------------------

class _Interrupting(Exception):
    def __str__(self):
        raise KeyboardInterrupt


class _HostileArgument:
    def __str__(self):
        raise KeyboardInterrupt

    __repr__ = __str__


def _no_interrupt(call, *args, **kwargs):
    """A KeyboardInterrupt raised by formatting must not escape (and must not stop the test run)."""
    try:
        return call(*args, **kwargs)
    except KeyboardInterrupt:
        pytest.fail('a KeyboardInterrupt from user formatting code escaped the translation')


def test_r2_translation_never_calls_a_user_str():
    def cancel():
        raise _Interrupting('boom')
    model, grid, removed, prepared, case, box = small()
    error = _no_interrupt(refused, 'COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, cancel=cancel)
    assert error.reason == 'unexpected:_Interrupting'


def test_r2_translation_never_formats_a_hostile_argument(monkeypatch):
    def broken(*args):
        raise ValueError(_HostileArgument())
    monkeypatch.setattr(sc, '_holed_model', broken)
    model, grid, removed, prepared, case, box = small()
    error = _no_interrupt(refused, 'COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case,
                          reason='ValueError')
    assert 'not shown' in error.args[0]


def test_r2_keyboard_interrupt_propagates():
    def cancel():
        raise KeyboardInterrupt
    model, grid, removed, prepared, case, box = small()
    with pytest.raises(KeyboardInterrupt):
        solve(model, prepared, removed, box, case, cancel=cancel)


# --- P2: C1 structure before use -------------------------------------------------------------------------------

def test_r2_c1_csr_column_index_out_of_range_refuses():
    bad = sparse.csr_matrix(np.ones((1, 1)))
    bad.indices[0] = -1
    refused('COUPLED_FORMATION_RANGE', sc._require_csr, bad, 'X', reason='csr_structure')
    bad = sparse.csr_matrix(np.ones((2, 2)))
    bad.indptr[2] = 3                                                # nnz mismatch
    refused('COUPLED_FORMATION_RANGE', sc._require_csr, bad, 'X', reason='csr_structure')


def test_r2_c1_connector_affine_of_wrong_length_refuses(monkeypatch):
    import anysolid.coupling as coupling
    original = coupling.build_weak_connector

    def empty_affine(*args, **kwargs):
        operator = original(*args, **kwargs)
        return types.SimpleNamespace(C_shell=operator.C_shell, C_solid=operator.C_solid, affine=np.zeros(0))
    monkeypatch.setattr(coupling, 'build_weak_connector', empty_affine)
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_CONNECTOR_INVALID', solve, model, prepared, removed, box, case, reason='affine')


def test_r2_c1_native_K_is_checked_before_reduction(monkeypatch):
    from anyfesolver import assembly
    original, reduced = assembly.assemble_system, []

    def nonfinite(model, case):
        K, F, info = original(model, case)
        K = K.tocsr().copy()
        K.data[0] = np.inf
        return K, F, info
    monkeypatch.setattr(assembly, 'assemble_system', nonfinite)
    reduce_original = assembly.build_constraint_transformation
    monkeypatch.setattr(assembly, 'build_constraint_transformation',
                        lambda *a: reduced.append(1) or reduce_original(*a))
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_FORMATION_RANGE', solve, model, prepared, removed, box, case, reason='nonfinite')
    assert not reduced


# --- P2: the ledger's component inventory against the raw records -----------------------------------------

def test_r2_ledger_inventory_sees_a_component_dropped_from_both_traversals(monkeypatch):
    original = LoadCase._load_contributions

    def dropping(self, *args, **kwargs):
        for item in original(self, *args, **kwargs):
            if item[0][0] == 'nodal' and item[0][2] == 1:
                continue
            yield item
    monkeypatch.setattr(LoadCase, '_load_contributions', dropping)
    model, grid, removed, prepared, _, box = small(skin=None)
    refused('COUPLED_LOAD_UNCOVERED', solve, model, prepared, removed, box, nodal_case(grid, (0.0, 0.0, 1234.0)),
            reason='ledger_inventory')


# --- P2: the forward-export fixture forces downward nearest rounding --------------------------------------

def test_r2_exported_forward_error_rounds_up_where_nearest_rounds_down(monkeypatch):
    import dataclasses
    captured = {}
    original = sc.certified_solve

    def spy(*args, **kwargs):
        solved = original(*args, **kwargs)
        up = Fraction(cc._out_up(solved.forward_error))
        forced = up + Fraction(1, 2 ** 200) * up                      # nearest rounding gives `up`, below it
        assert Fraction(cc._to_float(forced)) < forced
        captured['E'] = forced
        return dataclasses.replace(solved, forward_error=forced)
    monkeypatch.setattr(sc, 'certified_solve', spy)
    model, grid, removed, prepared, case, box = small()
    result = solve(model, prepared, removed, box, case)
    assert Fraction(result.verification['forward_error']) >= captured['E']


# --- P2: the test oracle uses the specified scaling (r11 classes module) ----------------------------------

def test_r2_oracle_scaling_is_the_specified_power_of_two_ruiz():
    import test_coupled_r11_classes as r11
    s = 1.4 * (1 - 2.1e-7)
    S = sparse.csr_matrix(np.array([[1.0, s], [s, 1.96]]))
    production = cc._certify_solve_matrix(S)
    assert not production['shifts'].any()                           # power-of-two Ruiz leaves it unchanged
    kappa = r11.independent_condition(S)
    assert kappa > 1e7 and abs(kappa / production['condition_number'] - 1) < 1e-6
