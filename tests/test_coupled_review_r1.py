"""Coupled route: regression tests for Codex code review round 1 (4 P1 + 7 P2) and the other
instances of each finding's class (docs/SOLID_COUPLING.md). Each test names its finding; the
fails-on-454540a6c evidence is in the commit message."""

from __future__ import annotations

import contextlib
import ctypes
import math
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
from anyfesolver._coupled_certificate import CoupledSolveError  # noqa: E402
from anyfesolver.boundary import LoadCase  # noqa: E402
from anyfesolver.constraint_audit import ConstraintEquation  # noqa: E402
from anyfesolver.elements import create_shell_element  # noqa: E402

SIDES = ('side_0', 'side_1', 'side_2', 'side_3')
T = 0.02
HOLE = (1, 2, 1, 2)
TINY = struct.unpack('<d', struct.pack('<Q', 1))[0]          # 2^-1074, built without arithmetic


def small(*, skin='end', pressure=1000.0, case=None, hook=None, order=2):
    model, grid, removed, footprint = fx.plate(3, thickness=T, hole=HOLE)
    if hook:
        hook(model, grid)
    prepared = fx.prepare(fx.box_mesh(footprint, T, 2, 2, 1), order=order)
    if case is None:
        case = fx.pressure_case(removed, model, pressure) if skin else LoadCase(name='none')
    return model, grid, removed, prepared, case, sc.BoxFaces((2, 2, 1), skin, SIDES)


def solve(model, prepared, removed, box, case, **kwargs):
    return sc.solve_shell_solid_coupled(model, prepared, removed_element_ids=removed, box=box, load_case=case,
                                        **kwargs)


def refused(code, call, *args, reason=None, **kwargs):
    with pytest.raises(CoupledSolveError) as info:
        call(*args, **kwargs)
    assert info.value.code == code, (info.value.code, info.value.reason, str(info.value))
    if reason is not None:
        assert info.value.reason == reason, info.value.reason
    return info.value


def bits(value):
    return struct.unpack('<q', struct.pack('<d', value))[0]


def _preserved():
    return bits(TINY * 3.0) == 3 and bool((np.full(64, TINY) * 3.0 != 0).all())


@contextlib.contextmanager
def flushing():
    """Denormals-are-zero and flush-to-zero on this thread (MSVC runtime), restored afterwards."""
    if sys.platform != 'win32':
        pytest.skip('denormal flushing is switched through the MSVC runtime')
    crt = ctypes.CDLL('ucrtbase')
    crt._controlfp.restype = ctypes.c_uint
    crt._controlfp.argtypes = [ctypes.c_uint, ctypes.c_uint]
    mask = 0x03000000                                          # _MCW_DN
    old = crt._controlfp(0, 0)
    crt._controlfp(0x01000000, mask)                           # _DN_FLUSH
    try:
        if _preserved():
            pytest.skip('this runtime does not flush denormals')
        yield
    finally:
        crt._controlfp(old, mask)
    assert _preserved()


# --- #1 (class 1): float decode/encode independent of the FP environment -----------------------------

def test_r1_1_exact_residual_keeps_a_subnormal_candidate_under_flush():
    S = sparse.csr_matrix(np.eye(2))
    x = np.array([1.0, TINY])
    b = np.array([1.0, 0.0])
    with flushing():
        rows = cc._exact_rows(S, x, b)
        mantissa, exponent = cc._decompose(x)
        residual = rows.residual_fraction(1)
    assert (int(mantissa[1]), int(exponent[1])) == (1, -1074)
    assert residual == -Fraction(1, 2 ** 1074)


def test_r1_1_certified_solve_never_accepts_a_wrong_coefficient_with_zero_error(monkeypatch):
    S = sparse.csr_matrix(np.eye(2))
    b = np.array([1.0, 0.0])
    monkeypatch.setattr(cc, '_candidate', lambda system, rhs: np.array([1.0, TINY]))
    with flushing():
        refused('COUPLED_UNCERTIFIED', cc.certified_solve, S, b, raw_load_zero=False, reason='subnormal_flush')
    solved = cc.certified_solve(S, b, raw_load_zero=False)    # IEEE gradual underflow: a true claim
    assert solved.forward_error > 0 and solved.bound(1) >= Fraction(1, 2 ** 1074)


def test_r1_1_out_up_and_nearest_are_exact_under_flush():
    value = Fraction(1, 2 ** 1050)                             # subnormal: bit pattern 2^24
    with flushing():
        up, nearest = sc._out_up(value), cc._to_float(value)
        third_up, third = sc._out_up(Fraction(1, 3 * 2 ** 1060)), cc._to_float(Fraction(1, 3 * 2 ** 1060))
    assert bits(up) == bits(nearest) == 1 << 24
    assert bits(third) == bits(float(Fraction(1, 3 * 2 ** 1060))) and bits(third_up) == bits(third) + 1


def test_r1_1_conversions_agree_with_ieee_rounding():
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 2 ** 63, size=4000, dtype=np.uint64)
    raw[:40] = np.arange(40)                                   # subnormals and zero
    values = raw.view(np.float64)
    values = values[np.isfinite(values)]
    for v in values[:1500]:
        v = float(v)
        assert cc._exact(v) == Fraction(v) and cc._exact(-v) == -Fraction(v)
        assert bits(cc._to_float(Fraction(v))) == bits(v)
    m, e = cc._decompose(values)
    assert all(Fraction(int(a)) * Fraction(2) ** int(k) == Fraction(float(v)) for a, k, v in zip(m[:500], e[:500], values))
    for _ in range(1500):
        q = Fraction(int(rng.integers(1, 2 ** 62)), int(rng.integers(1, 2 ** 62))) * Fraction(2) ** int(rng.integers(-1100, 1000))
        assert bits(cc._to_float(q)) == bits(float(q))
        up = cc._out_up(q)
        assert Fraction(up) >= q and (up == 0 or Fraction(math.nextafter(up, 0.0)) < q)
    assert cc._to_float(Fraction(2) ** 1024) == math.inf and cc._out_up(Fraction(2) ** 1024) == math.inf
    assert cc._to_float(Fraction(1, 2 ** 1076)) == 0.0 and cc._out_up(Fraction(1, 2 ** 1076)) == TINY


def test_r1_1_raw_window_sees_a_subnormal_under_flush():
    with flushing():
        refused('COUPLED_FORMATION_RANGE', sc._check_window, np.array([0.0, TINY]), 'x', reason='raw_window')


def test_r1_1_zero_test_and_abs_max_use_bits_under_flush():
    with flushing():
        assert cc._any_nonzero(np.array([0.0, TINY])) and not cc._any_nonzero(np.array([0.0, -0.0]))
        assert cc._abs_max_exact(np.array([0.0, -TINY])) == Fraction(1, 2 ** 1074)


def test_r1_1_route_refuses_when_the_thread_flushes_subnormals():
    model, grid, removed, prepared, case, box = small()
    with flushing():
        refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case, reason='subnormal_flush')
    solve(model, prepared, removed, box, case)                # unchanged outside the flush mode


def test_r1_1_a_callback_that_switches_on_flushing_is_refused():
    model, grid, removed, prepared, case, box = small()
    stack = contextlib.ExitStack()

    def cancel():
        if not stack._exit_callbacks:
            stack.enter_context(flushing())
        return False
    try:
        refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case, cancel=cancel,
                reason='subnormal_flush')
    finally:
        stack.close()


# --- #2 (class 2): contract properties of PreparedSolid and the FEModel ------------------------------

@pytest.mark.parametrize('nz', [1, 2])
def test_r1_2_p1_solid_refuses(nz):
    # nz = 2 is the reviewer's case (P1 accepted on 454540a6c); nz = 1 is singular for P1 there
    model, grid, removed, footprint = fx.plate(3, thickness=T, hole=HOLE)
    prepared = fx.prepare(fx.box_mesh(footprint, T, 2, 2, nz), order=1)
    box = sc.BoxFaces((2, 2, nz), 'end', SIDES)
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, fx.pressure_case(removed, model),
            reason='prepared_order')


@pytest.mark.parametrize('attribute, value, reason', [
    ('order', 2.0, 'prepared_order'),
    ('ndof', 225.0, 'prepared_ndof'),
    ('coefficient_components', 'float', 'prepared_coefficients'),
    ('coefficient_coordinates', 'float32', 'prepared_coefficients'),
])
def test_r1_2_prepared_properties_are_checked(attribute, value, reason):
    model, grid, removed, prepared, case, box = small()
    current = getattr(prepared, attribute)
    if value == 'float':
        value = np.asarray(current, dtype=float)
    elif value == 'float32':
        value = np.asarray(current, dtype=np.float32)
    object.__setattr__(prepared, attribute, value)
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, reason=reason)


def test_r1_2_model_dof_map_must_match_the_dof_manager():
    def hook(model, grid):
        a, b = model.mesh.nodes[grid[0, 0]], model.mesh.nodes[grid[3, 3]]
        a.dofs, b.dofs = list(b.dofs), list(a.dofs)
    model, grid, removed, prepared, case, box = small(hook=hook)
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, reason='dof_map')


# --- #3 (class 3): geometric deactivation ------------------------------------------------------------

def _coupled(model_hook):
    model, grid, removed, footprint = fx.plate(6, thickness=T, hole=(2, 4, 2, 4))
    model_hook(model, grid)
    prepared = fx.prepare(fx.box_mesh(footprint, T, 4, 4, 2))
    return solve(model, prepared, removed, sc.BoxFaces((4, 4, 2), 'end', SIDES), fx.pressure_case(removed, model))


@pytest.mark.parametrize('corners', [((2, 2), (4, 2), (4, 4), (2, 4)), ((3, 2), (4, 3), (3, 4), (2, 3))])
def test_r1_3_kept_element_spanning_the_hole_refuses(corners):
    def hook(model, grid):
        ids = [grid[c] for c in corners]
        model.add_element(1000, create_shell_element(1000, ids, 'steel', thickness=T))
    refused('COUPLED_CUT_MISMATCH', _coupled, hook, reason='kept_inside_footprint')


def test_r1_3_free_node_inside_the_footprint_refuses():
    def hook(model, grid):
        model.add_node(5000, 0.5, 0.5, 0.0)
        model.add_boundary_condition(BoundaryCondition(name='free', node_ids=[5000], dof_constraints={
            k: 0.0 for k in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    refused('COUPLED_CUT_MISMATCH', _coupled, hook, reason='kept_inside_footprint')


def test_r1_3_hull_test_is_exact_at_the_boundary():
    lo, hi = (Fraction(0),) * 3, (Fraction(1),) * 3
    P = lambda *v: tuple(Fraction(c) for c in v)  # noqa: E731
    # an adjacent element touching the face s = 0, and a segment touching an edge: no overlap
    assert not sc._hull_meets_open_box([P(-1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(-1, 1, 0)], lo, hi)
    assert not sc._hull_meets_open_box([P(0, 0, Fraction(1, 2)), P(0, 1, Fraction(1, 2))], lo, hi)
    # a diagonal segment through the interior, and a triangle whose nodes all lie on the boundary
    assert sc._hull_meets_open_box([P(0, 0, Fraction(1, 2)), P(1, 1, Fraction(1, 2))], lo, hi)
    assert sc._hull_meets_open_box([P(0, 0, Fraction(1, 2)), P(1, 0, Fraction(1, 2)), P(1, 1, Fraction(1, 2))], lo, hi)
    # a tetrahedron hull that pokes into the box only by its tip
    assert sc._hull_meets_open_box([P(-1, -1, -1), P(-1, 3, -1), P(3, -1, -1), P(Fraction(1, 2), Fraction(1, 2),
                                                                                    Fraction(1, 10 ** 30))], lo, hi)


# --- #4 (class 4): the zero-load shortcut applies the formation-loss refusal ---------------------------

def test_r1_4_subnormal_kept_pressure_refuses_formation_loss():
    model, grid, removed, prepared, case, box = small(skin=None)
    kept = next(eid for eid in model.mesh.elements if eid not in removed)
    case.pressure_loads[kept] = TINY
    refused('COUPLED_FORMATION_LOSS', solve, model, prepared, removed, box, case, reason='load_stored_as_zero')


def test_r1_4_mpc_rhs_lost_by_the_reduction_refuses_formation_loss():
    def hook(model, grid):
        a, b = model.mesh.nodes[grid[0, 1]].dofs[0], model.mesh.nodes[grid[0, 2]].dofs[0]
        model.constraint_equations.append(ConstraintEquation(terms=((a, 4.0), (b, -1.0)), rhs=TINY,
                                                             source_id='tiny', dependent_dof=a))
    model, grid, removed, prepared, case, box = small(skin=None, hook=hook)
    refused('COUPLED_FORMATION_LOSS', solve, model, prepared, removed, box, case, reason='load_stored_as_zero')


# --- #5: V5 includes the MPC master constraint forces ---------------------------------------------------

def test_r1_5_v5_balances_with_an_mpc_tie_across_the_cut():
    def hook(model, grid):
        slave, master = model.mesh.nodes[grid[1, 1]].dofs[2], model.mesh.nodes[grid[2, 2]].dofs[2]
        model.constraint_equations.append(ConstraintEquation(terms=((slave, 1.0), (master, -1.0)), rhs=0.0,
                                                             source_id='tie', dependent_dof=slave))
    model, grid, removed, prepared, _, box = small(skin=None, hook=hook)
    case = LoadCase(name='point')
    case.nodal_loads[grid[1, 1]] = np.array([0.0, 0.0, 1234.0, 0.0, 0.0, 0.0])
    result = solve(model, prepared, removed, box, case)
    assert result.verification['v5_balance'] <= 1e-10     # V5 tolerance 1e-9; r1 showed ~0.5
    # independent: the support reactions carry the whole vertical load
    supported = [i for i, nid in enumerate(result.shell_node_ids) if nid not in (grid[1, 1],)]
    vertical = result.reactions.reshape(-1, 6)[supported, 2].sum()
    assert abs(vertical + 1234.0) <= 1e-6 * 1234.0


# --- #6 (class 5): exported values are the bounded ones -------------------------------------------------

def test_r1_6_exported_centre_is_the_moment_origin(monkeypatch):
    seen = []
    original = sc._rigid_basis

    def spy(coords, names, centre):
        seen.append(tuple(centre))
        return original(coords, names, centre)
    monkeypatch.setattr(sc, '_rigid_basis', spy)
    model, grid, removed, prepared, case, box = small()
    result = solve(model, prepared, removed, box, case)
    exported = tuple(Fraction(float(v)) for v in result.footprint_centre)
    assert seen and all(c == exported for c in seen)


def test_r1_6_side_resultants_match_an_independent_virtual_work(monkeypatch):
    import anysolid.coupling as coupling
    operators = []
    original = coupling.build_weak_connector

    def spy(*args, **kwargs):
        operator = original(*args, **kwargs)
        operators.append(operator.C_shell.tocsr())
        return operator
    monkeypatch.setattr(coupling, 'build_weak_connector', spy)
    model, grid, removed, prepared, case, box = small()
    result = solve(model, prepared, removed, box, case)
    xc = [Fraction(float(v)) for v in result.footprint_centre]
    lam = [Fraction(float(v)) for v in result.multipliers.values]
    start, largest = 0, 0.0
    for side, C in zip(result.side_resultants, operators):
        C = C.tocoo()
        f = {}
        for i, j, v in zip(C.row, C.col, C.data):
            f[int(j)] = f.get(int(j), Fraction(0)) + Fraction(float(v)) * lam[start + int(i)]
        force, moment = [Fraction(0)] * 3, [Fraction(0)] * 3
        for dof, value in f.items():
            nid = result.shell_node_ids[dof // 6]
            r = [Fraction(float(c)) - x for c, x in zip(model.mesh.nodes[nid].coords(), xc)]
            k = dof % 6
            if k < 3:
                force[k] -= value
                e = [Fraction(0)] * 3
                e[k] = value
                cross = (r[1] * e[2] - r[2] * e[1], r[2] * e[0] - r[0] * e[2], r[0] * e[1] - r[1] * e[0])
                moment = [m - c for m, c in zip(moment, cross)]
            else:
                moment[k - 3] -= value
        for bounded, oracle in ((side.force, force), (side.moment, moment)):
            for value, bound, exact in zip(bounded.values, bounded.bounds, oracle):
                assert abs(Fraction(float(value)) - exact) <= Fraction(float(bound))
        largest = max(largest, float(np.abs(side.moment.values).max()))
        start += C.shape[0]
    assert largest > 1.0                                        # nonzero moments: a sign flip would fail


def test_r1_6_exported_forward_error_is_not_below_the_bound(monkeypatch):
    captured = {}
    original = sc.certified_solve

    def spy(*args, **kwargs):
        captured['solved'] = original(*args, **kwargs)
        return captured['solved']
    monkeypatch.setattr(sc, 'certified_solve', spy)
    model, grid, removed, prepared, case, box = small()
    result = solve(model, prepared, removed, box, case)
    assert Fraction(result.verification['forward_error']) >= captured['solved'].forward_error


# --- #7 (class 6): exact types before coercion ----------------------------------------------------------

def test_r1_7_float32_skin_traction_refuses(monkeypatch):
    from anysolid.backend import PreparedSolid
    original = PreparedSolid.traction
    monkeypatch.setattr(PreparedSolid, 'traction',
                        lambda self, *a, **k: np.asarray(original(self, *a, **k), dtype=np.float32))
    model, grid, removed, prepared, case, box = small(pressure=0.0)
    refused('COUPLED_FORMATION_RANGE', solve, model, prepared, removed, box, case, reason='type')


def test_r1_7_float32_connector_block_refuses(monkeypatch):
    import anysolid.coupling as coupling
    original = coupling.build_weak_connector

    calls = []

    def float32(*args, **kwargs):            # one side only: vstack would upcast it to float64
        operator = original(*args, **kwargs)
        calls.append(1)
        if len(calls) > 1:
            return operator
        return types.SimpleNamespace(C_shell=operator.C_shell.astype(np.float32), C_solid=operator.C_solid,
                                     affine=operator.affine)
    monkeypatch.setattr(coupling, 'build_weak_connector', float32)
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_FORMATION_RANGE', solve, model, prepared, removed, box, case, reason='type')


def test_r1_7_rigid_modes_must_be_a_float64_array(monkeypatch):
    from anysolid.backend import PreparedSolid
    original = PreparedSolid.rigid_modes
    monkeypatch.setattr(PreparedSolid, 'rigid_modes', lambda self: original(self).tolist())
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, reason='prepared_rigid_modes')


@pytest.mark.parametrize('cancel', [lambda: 'no', lambda: 0, lambda: np.array([False])])
def test_r1_7_cancel_flag_must_be_a_bool(cancel):
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, cancel=cancel,
            reason='cancel_invalid')


@pytest.mark.parametrize('limit', [1e12, True, -1])
def test_r1_7_memory_limit_must_be_an_integer(limit):
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case,
            resources=types.SimpleNamespace(memory_limit_bytes=limit), reason='resources')


# --- #8, #9 (class 7): cancellation and exception paths ----------------------------------------------

@pytest.mark.parametrize('target', ['recovery', 'verification', 'v4'])
def test_r1_8_cancellation_late_in_the_solve_never_returns_success(monkeypatch, target):
    event = threading.Event()
    # OD14: the verification runs in the worker; 'verification' trips while its verdict is decoded,
    # 'recovery' while the recovered values are wrapped.
    module, name = {'recovery': (sc, 'Bounded'), 'verification': (cc, '_decode'),
                    'v4': (sc, '_v4_side_balance')}[target]
    original = getattr(module, name)

    def trip(*args, **kwargs):
        event.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, trip)
    model, grid, removed, prepared, case, box = small()
    refused('COUPLED_CANCELLED', solve, model, prepared, removed, box, case, cancel=event)


class _Unprintable(Exception):
    def __str__(self):
        raise RuntimeError('no text')


def test_r1_9_an_unprintable_callback_exception_is_coded():
    def cancel():
        raise _Unprintable()
    model, grid, removed, prepared, case, box = small()
    error = refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, cancel=cancel)
    assert error.reason == 'unexpected:_Unprintable'


def test_r1_9_an_unprintable_native_error_is_coded(monkeypatch):
    class Bad(ValueError):
        def __str__(self):
            raise RuntimeError('no text')

    def broken(*args):
        raise Bad()
    monkeypatch.setattr(sc, '_holed_model', broken)
    model, grid, removed, prepared, case, box = small()
    error = refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case)
    assert error.reason == 'Bad'


# --- #10 (class 8): ledger completeness -----------------------------------------------------------------

@pytest.mark.parametrize('empty', [[], np.zeros(0)])
def test_r1_10_ledger_lists_an_empty_load_record(empty):
    model, grid, removed, prepared, case, box = small()
    node = grid[0, 1]
    case.nodal_loads[node] = empty
    result = solve(model, prepared, removed, box, case)
    rows = [row for row in result.ledger if row['record'] == 'nodal' and row['id'] == node]
    assert len(rows) == 1 and rows[0]['contributions'] == [] and rows[0]['fate'] == 'kept'
    kept_pressure = {row['id'] for row in result.ledger if row['record'] == 'pressure' and row['fate'] == 'kept'}
    assert kept_pressure == {eid for eid in case.pressure_loads if eid not in removed}
