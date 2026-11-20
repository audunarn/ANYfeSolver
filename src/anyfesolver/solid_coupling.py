"""Two-way coupled shell-solid linear static solve with a dense certificate (private, ANYfem only).

The shell is the native FEModel minus ``removed_element_ids`` (and minus the nodes only those
elements referenced), reduced by the unchanged native ``assemble_system`` and
``build_constraint_transformation``. The solid is a floating ANYsolid ``PreparedSolid`` on an
explicit structured box lattice. Per box side, the solver builds the shell edge trace and the weak
section-moment connector itself, forms the KKT system exactly (C1), certifies it with the dense
power-of-two equilibrated SVD and verifies the candidate with an exact residual (V1-V3, V6). See
``docs/SOLID_COUPLING.md`` for the contract, the claims and the refusal codes.

Not exported from :mod:`anyfesolver`; the documented sole caller is ANYfem's coupled solid detail.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, field
from fractions import Fraction
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from ._coupled_certificate import (
    _ABS_MASK,
    COUPLED_MAX_EQUATIONS,
    CoupledSolveError,
    _any_nonzero,
    _asymmetric,
    _bits_of,
    _checked_clock,
    _dense_bytes,
    _exact,
    _exception_text,
    _float_bits,
    _type_name,
    _nonzero_mask,
    _out_up,
    _refuse,
    _require_fp_environment,
    _to_float,
    certified_solve,
)

__all__: List[str] = []

# Raw window for every stored nonzero (C1).
W_RAW_LOW = Fraction(1, 2 ** 400)
W_RAW_HIGH = Fraction(2 ** 64)
# Section 3.1 tolerances, decided exactly.
PLANE_TOLERANCE = Fraction(1, 10 ** 12)      # times the footprint diagonal
LATTICE_TOLERANCE = Fraction(1, 10 ** 9)     # times the smallest lattice edge
NORMAL_TOLERANCE = 1e-12                     # parallel, equally oriented normals
TRANSFER_TOLERANCE = Fraction(1, 10 ** 12)   # transfer resultant gates
V4_COMPATIBILITY_TOLERANCE = Fraction(1, 10 ** 12)
V4_BALANCE_TOLERANCE = Fraction(1, 10 ** 10)
V5_BALANCE_TOLERANCE = Fraction(1, 10 ** 9)
_MAX_CONNECTOR_FACETS = 4096
_MAX_CONNECTOR_ROWS = 256
_SECTION_MODES = 5
_SQRT_BITS = 120
_DOF_NAMES = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')
_SKINS = ('start', 'end')


# --- exact rational helpers ----------------------------------------------------------------------

def _q(value: Any) -> Fraction:
    """The exact rational value of a finite float64 (IEEE bit extraction, review r1 class 1)."""
    return _exact(value)


def _sqrt_lo(value: Fraction) -> Fraction:
    """A lower bound of sqrt(value), value >= 0 (exact integer root, 120 fraction bits)."""
    if value <= 0:
        return Fraction(0)
    scaled = (value.numerator << (2 * _SQRT_BITS)) // value.denominator
    return Fraction(math.isqrt(scaled), 1 << _SQRT_BITS)


def _sqrt_hi(value: Fraction) -> Fraction:
    """An upper bound of sqrt(value), value >= 0."""
    if value <= 0:
        return Fraction(0)
    numerator = value.numerator << (2 * _SQRT_BITS)
    scaled = -(-numerator // value.denominator)
    root = math.isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, 1 << _SQRT_BITS)


def _ratio_le(a: Fraction, nn: Fraction, c: Fraction) -> bool:
    """a / sqrt(nn) <= c, decided exactly (nn > 0)."""
    if c >= 0:
        return a <= 0 or a * a <= c * c * nn
    return a < 0 and a * a >= c * c * nn


def _ratio_ge(a: Fraction, nn: Fraction, c: Fraction) -> bool:
    """a / sqrt(nn) >= c, decided exactly (nn > 0)."""
    return _ratio_le(-a, nn, -c)


def _safe_text(exc: BaseException) -> str:
    """'TypeName: message' of an exception; never calls a user-defined __str__/__repr__ or a
    metaclass hook (r1 #9, r2 (d))."""
    return _exception_text(exc)


def _safe_name(exc: BaseException) -> str:
    return _type_name(exc)


def _vec(point: Sequence[float]) -> Tuple[Fraction, Fraction, Fraction]:
    return tuple(_q(v) for v in point)  # type: ignore[return-value]


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b) -> Fraction:
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm2(a) -> Fraction:
    return _dot(a, a)


def _in_window(value: Fraction) -> bool:
    magnitude = abs(value)
    return W_RAW_LOW <= magnitude <= W_RAW_HIGH


# --- public data --------------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxFaces:
    """The structured box lattice: ``divisions = (nx, ny, nz)`` cells along the footprint sides
    c1-c0 and c3-c0 and through the thickness; ``skin`` the loaded face (``'start'`` at -t/2 or
    ``'end'`` at +t/2 along the shell director) or None; ``sides`` the four side group names in
    counter-clockwise order about the director, ``sides[k]`` spanning footprint corners k, k+1."""

    divisions: Tuple[int, int, int]
    skin: Optional[str]
    sides: Tuple[str, str, str, str]


@dataclass(frozen=True)
class Bounded:
    """Exactly evaluated values rounded once to float64, with per-entry upper error bounds
    against the exact solution of the stored system (claims under A1/A2)."""

    values: np.ndarray
    bounds: np.ndarray


@dataclass(frozen=True)
class SideResultant:
    """Force and moment (about the stored footprint centre) the solid exerts on the shell."""

    side: str
    force: Bounded
    moment: Bounded


@dataclass(frozen=True)
class CoupledSolution:
    """Claims (O12a): ``shell_displacements``, ``solid_coefficients``, ``multipliers``,
    ``shell_interface_forces``, ``solid_interface_forces`` and ``side_resultants`` carry bounds;
    ``reactions`` carry none."""

    shell_node_ids: Tuple[int, ...]
    shell_displacements: Bounded           # (n_nodes * 6,), node-major in shell_node_ids order
    solid_coefficients: Bounded
    multipliers: Bounded
    shell_interface_forces: Bounded        # C_shell^T lambda on the shell DOFs (same layout)
    solid_interface_forces: Bounded        # C_solid^T lambda on the solid coefficients
    side_resultants: Tuple[SideResultant, ...]
    reactions: np.ndarray                  # no claim
    footprint_centre: np.ndarray
    echo: Mapping[str, Any]
    ledger: Tuple[Mapping[str, Any], ...]
    certificate: Mapping[str, Any]
    verification: Mapping[str, Any]
    sizes: Mapping[str, int]
    timings: Mapping[str, float] = field(default_factory=dict)


# --- small utilities ----------------------------------------------------------------------------

def _cancel_probe(cancel: Any) -> Optional[Callable[[], None]]:
    if cancel is None:
        return None
    is_set = getattr(cancel, 'is_set', None)
    if callable(is_set):
        flag_value = is_set
    elif callable(cancel):
        flag_value = cancel
    else:
        raise _refuse('COUPLED_NATIVE_INVALID', 'cancel_invalid',
                      'cancel must be None, a callable returning a flag, or an object with is_set()')

    def check() -> None:
        value = flag_value()
        # The flag's exact type is checked before it is used (r1 class 6): no truthiness coercion.
        if type(value) is not bool and type(value) is not np.bool_:
            raise _refuse('COUPLED_NATIVE_INVALID', 'cancel_invalid', 'the cancellation flag must be a bool')
        if value:
            raise _refuse('COUPLED_CANCELLED', None, 'the coupled solve was cancelled')
    return check


def _exact_int(value: Any, what: str, code: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise _refuse(code, 'invalid_identifier', f'{what} must be an exact integer')
    return int(value)


def _layout(element: Any):
    from .solid_trace import ShellEdgeTrace
    return ShellEdgeTrace._layout(element)


def _element_unit_normal(model: Any, element: Any) -> np.ndarray:
    """The shell director as the trace uses it: the physical director, else the geometric normal."""
    from .solid_trace import _element_coordinates, _scaled_unit
    director = element.physical_reference_director
    if director is None:
        coordinates = _element_coordinates(model, element)
        director = np.cross(coordinates[1] - coordinates[0], coordinates[-1] - coordinates[0])
    unit = _scaled_unit(np.asarray(director, dtype=float))
    if unit is None or not np.isfinite(unit).all():
        raise _refuse('COUPLED_CUT_MISMATCH', 'degenerate_normal', 'a shell element normal is degenerate')
    return np.asarray(unit, dtype=float)


# --- section 4: cut, orphans and targets -------------------------------------------------------

@dataclass(frozen=True)
class _Cut:
    removed: Tuple[int, ...]
    orphans: Tuple[int, ...]
    chains: Tuple[Tuple[Tuple[int, int], ...], ...]   # kept (element, edge) per straight chain, loop order
    corners: Tuple[Tuple[int, int], ...]              # (first, last) shell node of each chain
    normal: np.ndarray
    thickness: float


def _analyse_cut(model: Any, removed_ids: Sequence[int]) -> _Cut:
    elements = model.mesh.elements
    # The caller's sequence is iterated and converted once, then the FP environment is re-checked
    # (review r3 P2: iteration and int conversion can run caller code).
    ids = [_exact_int(value, 'removed element id', 'COUPLED_CUT_MISMATCH') for value in removed_ids]
    _require_fp_environment()
    removed = []
    for eid in ids:
        if eid not in elements:
            raise _refuse('COUPLED_CUT_MISMATCH', 'unknown_element', f'removed element {eid} is not in the model')
        removed.append(eid)
    if not removed or len(set(removed)) != len(removed):
        raise _refuse('COUPLED_CUT_MISMATCH', 'removed_set', 'the removed set must be nonempty and unique')
    removed_set = set(removed)
    counts: Counter = Counter()
    normals, thicknesses = [], []
    for eid in removed:
        element = elements[eid]
        try:
            edges, _ = _layout(element)
        except CoupledSolveError as exc:
            raise _refuse('COUPLED_CUT_MISMATCH', 'removed_element_kind',
                          'removed elements must be production E4-PL Q4 or S3 V2D shells') from exc
        ids = tuple(int(n) for n in element.node_ids)
        for a, b in edges:
            counts[frozenset((ids[a], ids[b]))] += 1
        normals.append(_element_unit_normal(model, element))
        thicknesses.append(float(element.thickness))
        if float(getattr(element, 'reference_surface_offset', 0.0) or 0.0) != 0.0:
            raise _refuse('COUPLED_CUT_MISMATCH', 'reference_surface_offset',
                          'v1 requires a zero shell reference-surface offset')
    if any(c > 2 for c in counts.values()):
        raise _refuse('COUPLED_CUT_MISMATCH', 'nonmanifold', 'a removed edge is shared by more than two elements')
    boundary = {key for key, c in counts.items() if c == 1}
    interior = {key for key, c in counts.items() if c == 2}
    kept_nodes = set()
    owners: Dict[frozenset, List[Tuple[int, int, int, int]]] = {key: [] for key in boundary}
    for eid, element in elements.items():
        if eid in removed_set:
            continue
        ids = tuple(int(n) for n in element.node_ids)
        kept_nodes.update(ids)
        try:
            edges, _ = _layout(element)
        except CoupledSolveError:
            edges = tuple(zip(range(len(ids)), list(range(1, len(ids))) + [0]))
        for k, (a, b) in enumerate(edges):
            key = frozenset((ids[a], ids[b]))
            if key in interior:
                raise _refuse('COUPLED_CUT_MISMATCH', 'overlap', f'kept element {eid} lies on the removed interior')
            if key in boundary:
                owners[key].append((eid, k, ids[a], ids[b]))
    for key, found in owners.items():
        if len(found) != 1:
            raise _refuse('COUPLED_CUT_MISMATCH', 'cut_edge_owner',
                          f'each cut edge needs exactly one kept shell element (edge {sorted(key)} has {len(found)})')
    removed_nodes = {int(n) for eid in removed for n in elements[eid].node_ids}
    orphans = tuple(sorted(removed_nodes - kept_nodes))
    # A single cycle through the boundary edges, followed in the kept elements' edge direction.
    directed = {}
    for key, ((eid, k, a, b),) in owners.items():
        if a in directed:
            raise _refuse('COUPLED_CUT_MISMATCH', 'cut_loop', 'the cut edges do not form one consistently oriented loop')
        directed[a] = (b, eid, k)
    start = next(iter(directed))
    loop, node = [], start
    for _ in range(len(directed)):
        if node not in directed:
            raise _refuse('COUPLED_CUT_MISMATCH', 'cut_loop', 'the cut edges do not form one closed loop')
        nxt, eid, k = directed[node]
        loop.append((node, nxt, eid, k))
        node = nxt
    if node != start or len({a for a, *_ in loop}) != len(loop):
        raise _refuse('COUPLED_CUT_MISMATCH', 'cut_loop', 'the cut edges do not form one closed loop')
    coords = {nid: np.asarray(model.mesh.nodes[nid].coords(), dtype=float) for nid in removed_nodes | {a for a, *_ in loop}}
    n0 = normals[0]
    if any(np.max(np.abs(n - n0)) > NORMAL_TOLERANCE for n in normals) or any(t != thicknesses[0] for t in thicknesses):
        raise _refuse('COUPLED_CUT_MISMATCH', 'mixed_faces', 'the removed elements must share one normal and thickness')
    # Corners: direction changes along the loop.
    directions = [coords[b] - coords[a] for a, b, _, _ in loop]
    turns = []
    for i in range(len(loop)):
        d0, d1 = directions[i - 1], directions[i]
        cross = np.cross(d0, d1)
        if np.linalg.norm(cross) > 1e-9 * np.linalg.norm(d0) * np.linalg.norm(d1):
            turns.append(i)
    if len(turns) != 4:
        raise _refuse('COUPLED_CUT_MISMATCH', 'not_rectangular', f'the cut loop has {len(turns)} corners, not 4')
    chains, corners = [], []
    for c in range(4):
        first, last = turns[c], turns[(c + 1) % 4]
        indices = list(range(first, last if last > first else last + len(loop)))
        segment = [loop[i % len(loop)] for i in indices]
        chains.append(tuple((eid, k) for _, _, eid, k in segment))
        corners.append((segment[0][0], segment[-1][1]))
    return _Cut(tuple(removed), orphans, tuple(chains), tuple(corners), n0, thicknesses[0])


def _holed_model(model: Any, cut: _Cut):
    """A deep copy minus the removed elements and the orphan nodes, DOFs renumbered in node order.
    Returns (holed, old_to_new) with old_to_new mapping original DOFs of kept nodes."""
    from .fe_core import DOFManager
    from .constraint_audit import ConstraintEquation
    holed = copy.deepcopy(model)
    for eid in cut.removed:
        del holed.mesh.elements[eid]
    for nid in cut.orphans:
        del holed.mesh.nodes[nid]
        holed.mesh.point_masses.pop(nid, None)
        holed.mesh.concentrated_masses.pop(nid, None)
    manager = DOFManager()
    old_to_new: Dict[int, int] = {}
    for nid, node in holed.mesh.nodes.items():
        old = list(model.mesh.nodes[nid].dofs)
        node.dofs = manager.add_node(nid)
        old_to_new.update(zip(old, node.dofs))
    holed.mesh.dof_manager = manager
    equations = []
    for equation in model.constraint_equations:
        terms = []
        for dof, coefficient in equation.coefficients:
            if int(dof) not in old_to_new:
                raise _refuse('COUPLED_TARGET_REMOVED', 'mpc_term', f'an MPC term names DOF {int(dof)} outside the holed shell')
            terms.append((old_to_new[int(dof)], float(coefficient)))
        dependent = int(equation.dependent_dof)
        if dependent not in old_to_new:
            raise _refuse('COUPLED_TARGET_REMOVED', 'mpc_slave', 'an MPC slave lies outside the holed shell')
        equations.append(ConstraintEquation(terms=tuple(terms), rhs=float(equation.value), source_id=equation.origin,
                                            dependent_dof=old_to_new[dependent], kind=equation.kind))
    holed.constraint_equations = equations
    holed.load_cases = []
    for category in ('topology', 'boundary', 'mpc', 'mass', 'load'):
        holed.mesh.bump_revision(category)
    return holed, old_to_new


def _check_targets(model: Any, cut: _Cut, load_case: Any) -> None:
    """Orphans named by a load, support or MPC refuse COUPLED_FREE_NODE; unknown targets refuse
    COUPLED_TARGET_REMOVED."""
    orphans = set(cut.orphans)
    nodes = model.mesh.nodes
    orphan_dofs = {int(d) for nid in orphans for d in nodes[nid].dofs}
    for bc in model.boundary_conditions:
        for nid in bc.node_ids:
            nid = _exact_int(nid, 'support node id', 'COUPLED_TARGET_REMOVED')
            if nid in orphans:
                raise _refuse('COUPLED_FREE_NODE', 'support', f'a support names node {nid}, freed by the cut')
            if nid not in nodes:
                raise _refuse('COUPLED_TARGET_REMOVED', 'support_node', f'a support names unknown node {nid}')
        for name in bc.dof_constraints:
            if name not in _DOF_NAMES:
                raise _refuse('COUPLED_TARGET_REMOVED', 'support_dof', f'a support names unknown DOF {name!r}')
    for equation in model.constraint_equations:
        for dof, _ in equation.coefficients:
            if int(dof) in orphan_dofs:
                raise _refuse('COUPLED_FREE_NODE', 'mpc', 'an MPC names a node freed by the cut')
    for nid in load_case.nodal_loads:
        if nid in orphans:
            raise _refuse('COUPLED_FREE_NODE', 'nodal_load', f'a nodal load names node {nid}, freed by the cut')
    for nid in getattr(load_case, 'added_node_masses', {}) or {}:
        if nid not in nodes:
            raise _refuse('COUPLED_TARGET_REMOVED', 'added_mass', f'an added mass names unknown node {nid}')


# --- section 3: load route, skin and ledger ----------------------------------------------------

def _check_load_route(model: Any, load_case: Any) -> None:
    if model.mesh.element_activity is not None:
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'element_activity', 'the coupled route requires no element activity')
    if bool(load_case.follower_pressure):
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'follower_pressure', 'follower pressure is not supported')
    if load_case.element_loads:
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'element_loads', 'element load vectors are not supported')
    if load_case.gravity is not None:
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'gravity', 'gravity is not supported')
    for eid, pressure in load_case.pressure_loads.items():
        if type(pressure) not in (float, np.float64) or not math.isfinite(float(pressure)):
            raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'pressure_kind',
                          f'pressure on element {eid} must be a finite scalar float64')
    _check_raw_scalars(model, load_case)


_RAW_SCALAR_TYPES = (float, np.float64, int)


def _raw_scalar(value: Any) -> bool:
    """A raw load, support or MPC scalar as ANYfem emits it: an exact float, np.float64 or int
    (never bool, Fraction, Decimal or another numeric type that native code would round)."""
    return type(value) in _RAW_SCALAR_TYPES


def _check_raw_scalars(model: Any, load_case: Any) -> None:
    """Review r2 P1 (d): the raw scalar types are validated before native assembly, so that the
    zero-load decision below sees the values native code consumes, not a rounded image of them."""
    def bad(what: str) -> CoupledSolveError:
        return _refuse('COUPLED_LOAD_UNSUPPORTED', 'raw_scalar_type', f'{what} must be float or int scalars')
    for nid, load in load_case.nodal_loads.items():
        if type(load) is np.ndarray:
            if load.dtype != np.float64 or load.ndim != 1:
                raise bad(f'the nodal load on node {nid}')
        elif type(load) not in (list, tuple) or not all(_raw_scalar(v) for v in load):
            raise bad(f'the nodal load on node {nid}')
    for bc in model.boundary_conditions:
        if not all(_raw_scalar(v) for v in bc.dof_constraints.values()):
            raise bad(f'the prescribed values of support {bc.name!r}')
    for equation in model.constraint_equations:
        if not _raw_scalar(equation.value) or not all(_raw_scalar(c) for _, c in equation.coefficients):
            raise bad('an MPC right-hand side or coefficient')


def _scalar_nonzero(value: Any) -> bool:
    """value != 0 for a validated raw scalar, on the bit pattern for floats (a subnormal is nonzero)."""
    if type(value) is int:
        return value != 0
    return bool(_bits_of(float(value)) & 0x7FFF_FFFF_FFFF_FFFF)


def _skin_pressure(model: Any, cut: _Cut, load_case: Any, box: BoxFaces) -> Optional[float]:
    loaded = [eid for eid in cut.removed if eid in load_case.pressure_loads]
    if box.skin is None:
        if loaded:
            raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'skin_missing', 'a removed element carries pressure but skin is None')
        return None
    if len(loaded) != len(cut.removed):
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'skin_without_pressure',
                      'with a skin, every removed element must carry the same pressure')
    values = {float(load_case.pressure_loads[eid]) for eid in cut.removed}
    if len(values) != 1:
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'nonuniform_pressure', 'the removed pressure must be uniform')
    return values.pop()


def _ledger(holed: Any, holed_case: Any, active: int, F_native: np.ndarray) -> Tuple[Mapping[str, Any], ...]:
    """Consume the shared native traversal with ``report_skips``; refuse any skipped component or
    inactive DOF, and require the stored F to be exactly the sum of the ledger contributions."""
    # Review r1 #10 (class 8): every load record gets a row, in native traversal order (nodal, then
    # pressure), also when it yields no component (an empty nodal vector contributes nothing).
    rows: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for kind, records in (('nodal', holed_case.nodal_loads), ('pressure', holed_case.pressure_loads)):
        for ident in records:
            rows[(kind, ident)] = {'record': kind, 'id': ident, 'fate': 'kept', 'contributions': []}
    replay = np.zeros(active)
    one = lambda element_id: 1.0  # noqa: E731  (element_activity is None: native scale is 1.0)
    for record, dof, value in holed_case._load_contributions(holed.mesh, None, one, report_skips=True):
        kind, ident, component = record
        if kind.endswith('_skipped'):
            raise _refuse('COUPLED_LOAD_UNCOVERED', kind, f'a load component is not assembled ({kind} {ident}: {component})')
        if isinstance(dof, (bool, np.bool_)) or not isinstance(dof, (int, np.integer)) or not 0 <= int(dof) < active:
            raise _refuse('COUPLED_LOAD_UNCOVERED', 'inactive_dof', f'{kind} {ident} targets an inactive DOF')
        replay[int(dof)] += value
        if (kind, ident) not in rows:
            raise _refuse('COUPLED_LOAD_UNCOVERED', 'ledger_record', f'{kind} {ident} is not a load record of the case')
        rows[(kind, ident)]['contributions'].append((int(dof), component, float(value)))
    if replay.shape != F_native.shape or not np.array_equal(replay.view(np.int64), F_native.view(np.int64)):
        raise _refuse('COUPLED_LOAD_UNCOVERED', 'ledger_binding', 'the stored native load is not the ledger sum')
    # Review r2 (ledger): the component inventory against the raw records, not only the replay. A
    # nodal record of length L yields components 0..min(L, 6)-1, a pressure record one component
    # per DOF of its element's mapping, each exactly once.
    for (kind, ident), row in rows.items():
        if kind == 'nodal':
            expected = list(range(min(len(holed_case.nodal_loads[ident]), len(holed.mesh.nodes[ident].dofs))))
        else:
            expected = list(range(len(holed.mesh.elements[ident].get_dof_mapping(holed.mesh))))
        seen = sorted(component for _, component, _ in row['contributions'])
        if seen != expected:
            raise _refuse('COUPLED_LOAD_UNCOVERED', 'ledger_inventory',
                          f'{kind} {ident}: components {seen[:12]} do not match the raw record ({len(expected)})')
    return tuple(rows.values())


def _transferred(model: Any, cut: _Cut, load_case: Any) -> List[Tuple[int, int, float]]:
    """Native contributions (original DOFs) of the removed elements' pressure records."""
    replay = copy.deepcopy(load_case)
    replay.nodal_loads = {}
    replay.pressure_loads = {eid: load_case.pressure_loads[eid] for eid in cut.removed if eid in load_case.pressure_loads}
    out = []
    one = lambda element_id: 1.0  # noqa: E731
    for record, dof, value in replay._load_contributions(model.mesh, None, one, report_skips=True):
        if record[0].endswith('_skipped'):
            raise _refuse('COUPLED_LOAD_UNCOVERED', record[0], 'a transferred pressure component is not assembled')
        if type(value) not in (float, np.float64) or not math.isfinite(float(value)):
            raise _refuse('COUPLED_LOAD_UNCOVERED', 'transferred_value', 'a transferred component is not a finite float64')
        out.append((int(record[1]), int(dof), float(value)))
    return out


# --- section 5: common rigid basis --------------------------------------------------------------

def _rigid_basis(coords: Sequence[Sequence[float]], dof_names: Sequence[str], centre) -> List[Tuple[Fraction, ...]]:
    """Rows (one per DOF) of the unscaled rigid motions about ``centre`` (exact): translations e,
    rotations e x r on translational DOFs (r = x - centre) and e on rotational DOFs."""
    rows = []
    for x, name in zip(coords, dof_names):
        if name not in _DOF_NAMES:
            raise _refuse('COUPLED_TARGET_REMOVED', 'dof_name', f'DOF name {name!r} has no rigid motion')
        axis = _DOF_NAMES.index(name)
        row = [Fraction(0)] * 6
        if axis < 3:
            row[axis] = Fraction(1)
            r = _sub(_vec(x), centre)
            for e in range(3):
                unit = [Fraction(0)] * 3
                unit[e] = Fraction(1)
                row[3 + e] = _cross(unit, r)[axis]
        else:
            row[axis] = Fraction(1)
        rows.append(tuple(row))
    return rows


def _times_basis(matrix: sparse.csr_matrix, basis: Sequence[Tuple[Fraction, ...]]) -> List[List[Fraction]]:
    """Exact (matrix @ basis), the basis given as rational rows."""
    out = []
    for i in range(matrix.shape[0]):
        acc = [Fraction(0)] * 6
        for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
            c = _q(matrix.data[k])
            row = basis[matrix.indices[k]]
            for j in range(6):
                if row[j]:
                    acc[j] += c * row[j]
        out.append(acc)
    return out


def _abs_times_basis_inf(matrix: sparse.csr_matrix, basis) -> Fraction:
    best = Fraction(0)
    for i in range(matrix.shape[0]):
        total = Fraction(0)
        for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
            c = abs(_q(matrix.data[k]))
            total += c * sum(abs(v) for v in basis[matrix.indices[k]])
        best = max(best, total)
    return best


# --- section 3.1: lattice provenance ------------------------------------------------------------

def _group_facets(mesh: Any, name: str) -> np.ndarray:
    ids = mesh.boundary_groups[name]
    return np.asarray(mesh.boundary_facets[np.isin(mesh.boundary_ids, ids)], dtype=np.int64)


def _provenance(code_reason: str, message: str) -> CoupledSolveError:
    return _refuse('COUPLED_MESH_PROVENANCE', code_reason, message)


def _check_lattice_face(name: str, facets: np.ndarray, coordinates: np.ndarray, expected: Sequence[np.ndarray],
                        na: int, nb: int, outward: np.ndarray) -> Tuple[np.ndarray, Fraction]:
    """Section 3.1(c): the face is the affine (na x nb) lattice spanned by its corner vertices
    nearest ``expected = (p00, p10, p01)``, each cell split by one diagonal into two positively
    oriented triangles. Returns the vertex index grid (na+1, nb+1) and the squared tolerance."""
    if facets.shape != (2 * na * nb, 3):
        raise _provenance('facet_count', f'face {name!r} has {len(facets)} facets, expected {2 * na * nb}')
    vertices = np.unique(facets)
    if len(vertices) != (na + 1) * (nb + 1):
        raise _provenance('vertex_count', f'face {name!r} has {len(vertices)} vertices, expected {(na + 1) * (nb + 1)}')
    points = coordinates[vertices]
    picks = []
    for target in expected:
        picks.append(int(vertices[np.argmin(np.linalg.norm(points - target, axis=1))]))
    p00, p10, p01 = (_vec(coordinates[v]) for v in picks)
    A, B = _sub(p10, p00), _sub(p01, p00)
    h2 = min(_norm2(A) / (na * na), _norm2(B) / (nb * nb))
    if h2 <= 0:
        raise _provenance('degenerate_face', f'face {name!r} is degenerate')
    tol2 = LATTICE_TOLERANCE * LATTICE_TOLERANCE * h2
    a, b = coordinates[picks[1]] - coordinates[picks[0]], coordinates[picks[2]] - coordinates[picks[0]]
    gram = np.array([[a @ a, a @ b], [a @ b, b @ b]])
    rel = points - coordinates[picks[0]]
    try:
        alpha = np.linalg.solve(gram, np.stack([rel @ a, rel @ b]))
    except np.linalg.LinAlgError as exc:
        raise _provenance('degenerate_face', f'face {name!r} is degenerate') from exc
    ij = np.rint(alpha * np.array([[na], [nb]])).astype(np.int64)
    if ij.min() < 0 or (ij[0] > na).any() or (ij[1] > nb).any():
        raise _provenance('lattice', f'face {name!r} is not the expected lattice')
    grid = -np.ones((na + 1, nb + 1), dtype=np.int64)
    for v, i, j in zip(vertices, ij[0], ij[1]):
        if grid[i, j] >= 0:
            raise _provenance('lattice', f'face {name!r} is not the expected lattice')
        grid[i, j] = v
        target = tuple(p00[c] + Fraction(int(i), na) * A[c] + Fraction(int(j), nb) * B[c] for c in range(3))
        if _norm2(_sub(_vec(coordinates[v]), target)) > tol2:
            raise _provenance('lattice', f'face {name!r} vertex ({i},{j}) is off the affine lattice')
    index = {int(v): (int(i), int(j)) for v, i, j in zip(vertices, ij[0], ij[1])}
    cells: Dict[Tuple[int, int], List[frozenset]] = {}
    normal = _vec(outward)
    for facet in facets:
        corners = [index[int(v)] for v in facet]
        i0, j0 = min(c[0] for c in corners), min(c[1] for c in corners)
        if len(set(corners)) != 3 or any(not (i0 <= c[0] <= i0 + 1 and j0 <= c[1] <= j0 + 1) for c in corners) \
                or i0 >= na or j0 >= nb:
            raise _provenance('cell', f'face {name!r} has a facet outside one lattice cell')
        x0, x1, x2 = (_vec(coordinates[int(v)]) for v in facet)
        if not _dot(_cross(_sub(x1, x0), _sub(x2, x0)), normal) > 0:
            raise _provenance('orientation', f'face {name!r} has a facet that is not positively oriented')
        cells.setdefault((i0, j0), []).append(frozenset(corners))
    for i in range(na):
        for j in range(nb):
            pair = cells.get((i, j), [])
            full = {(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)}
            if len(pair) != 2 or pair[0] | pair[1] != full:
                raise _provenance('cell', f'face {name!r} cell ({i},{j}) is not split by one diagonal')
            missing = [tuple(full - p)[0] for p in pair]
            if missing[0][0] == missing[1][0] or missing[0][1] == missing[1][1]:
                raise _provenance('cell', f'face {name!r} cell ({i},{j}) is not split by one diagonal')
    return grid, tol2


@dataclass(frozen=True)
class _Frame:
    corners: Tuple[np.ndarray, ...]       # c0..c3 counter-clockwise about the director
    centre: Tuple[Fraction, Fraction, Fraction]
    normal: np.ndarray
    thickness: float
    diagonal_lo: Fraction
    chains: Tuple[Tuple[Tuple[int, int], ...], ...]   # chains[k]: the trace edges of side k


def _check_box(mesh: Any, box: BoxFaces, frame: _Frame, outwards: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Sections 3.1(b) and (c) against the shell frame. Returns the vertex grid of each face."""
    nx, ny, nz = box.divisions
    coordinates = np.asarray(mesh.coordinates, dtype=float)
    if len(coordinates) != (nx + 1) * (ny + 1) * (nz + 1):
        raise _provenance('vertex_count', 'the solid is not the structured box lattice of the given divisions')
    n = frame.normal
    t = frame.thickness
    c = frame.corners
    nn = _norm2(_vec(n))
    half = _q(t) / 2
    tau = PLANE_TOLERANCE * frame.diagonal_lo
    lower = [ck - (t / 2) * n for ck in c]
    upper = [ck + (t / 2) * n for ck in c]
    faces = {'start': ((lower[0], lower[1], lower[3]), nx, ny, -n, -1),
             'end': ((upper[0], upper[1], upper[3]), nx, ny, n, +1)}
    for k, name in enumerate(box.sides):
        faces[name] = ((lower[k], lower[(k + 1) % 4], upper[k]), nx if k % 2 == 0 else ny, nz, outwards[k], 0)
    grids = {}
    normal_q = _vec(n)
    c0 = _vec(c[0])
    for name, (expected, na, nb, outward, sigma) in faces.items():
        facets = _group_facets(mesh, name)
        vertices = np.unique(facets)
        # (b) plane check, exact.
        if sigma:
            target = sigma * half
            for v in vertices:
                a = _dot(normal_q, _sub(_vec(coordinates[v]), c0))
                if not (_ratio_le(a, nn, target + tau) and _ratio_ge(a, nn, target - tau)):
                    raise _provenance('plane', f'face {name!r} is off its skin plane')
        else:
            k = box.sides.index(name)
            m = _vec(outward)
            mm = _norm2(m)
            ck = _vec(c[k])
            for v in vertices:
                a = _dot(m, _sub(_vec(coordinates[v]), ck))
                if a * a > tau * tau * mm:
                    raise _provenance('plane', f'side {name!r} is off its cut plane')
        # (c) affine lattice, exact.
        grid, tol2 = _check_lattice_face(name, facets, coordinates, expected, na, nb, outward)
        grids[name] = grid
        # Face corners against the box corners of the shell frame (within the lattice tolerance).
        for (i, j), expect in zip(((0, 0), (na, 0), (0, nb)), expected):
            index = int(grid[i, j])
            corner_k = int(np.argmin([np.linalg.norm(expect - (ck + s * (t / 2) * n)) for ck in c for s in (-1, 1)]))
            ck, s = c[corner_k // 2], (-1, 1)[corner_k % 2]
            w = _sub(_vec(coordinates[index]), _vec(ck))
            if not _ratio_ge(s * _q(t) * _dot(w, normal_q), nn, _norm2(w) + half * half - tol2):
                raise _provenance('corner', f'face {name!r} corner is off the shell frame')
    return grids


# --- C1: formation ------------------------------------------------------------------------------

def _require_csr(matrix: Any, what: str) -> sparse.csr_matrix:
    if type(matrix) is not sparse.csr_matrix or matrix.dtype != np.float64:
        raise _refuse('COUPLED_FORMATION_RANGE', 'type', f'{what} must be a float64 scipy.sparse.csr_matrix')
    _check_csr_structure(matrix, what)
    matrix = matrix.copy()
    matrix.sum_duplicates()
    steps = np.diff(matrix.indices.astype(np.int64))
    row_start = np.zeros(len(matrix.indices), dtype=bool)
    row_start[matrix.indptr[:-1][matrix.indptr[:-1] < len(matrix.indices)]] = True
    if np.any((steps <= 0) & ~row_start[1:]):
        raise _refuse('COUPLED_FORMATION_RANGE', 'indices', f'{what} CSR indices do not strictly increase')
    if not np.isfinite(matrix.data).all():
        raise _refuse('COUPLED_FORMATION_RANGE', 'nonfinite', f'{what} is not finite')
    return matrix


def _check_csr_structure(matrix: sparse.csr_matrix, what: str) -> None:
    """CSR index validity before any use of the matrix (review r2, C1 structure): integer 1-D index
    arrays, indptr from 0 to nnz and nondecreasing, data and indices of length nnz, and every column
    index in [0, ncols)."""
    def bad(message: str) -> CoupledSolveError:
        return _refuse('COUPLED_FORMATION_RANGE', 'csr_structure', f'{what}: {message}')
    shape = matrix.shape
    if type(shape) is not tuple or len(shape) != 2 or not all(_is_int(s) and int(s) >= 0 for s in shape):
        raise bad('invalid shape')
    rows, cols = int(shape[0]), int(shape[1])
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    for array in (indptr, indices, data):
        if type(array) is not np.ndarray or array.ndim != 1:
            raise bad('index arrays must be 1-D ndarrays')
    if not (np.issubdtype(indptr.dtype, np.integer) and np.issubdtype(indices.dtype, np.integer)):
        raise bad('index arrays must be integer')
    if len(indptr) != rows + 1 or int(indptr[0]) != 0 or np.any(np.diff(indptr.astype(np.int64)) < 0):
        raise bad('indptr is not a valid row pointer')
    nnz = int(indptr[-1])
    if len(indices) != nnz or len(data) != nnz:
        raise bad('indices/data do not have indptr[-1] entries')
    if nnz and (int(indices.min()) < 0 or int(indices.max()) >= cols):
        raise bad('a column index lies outside the matrix')


def _require_vector(vector: Any, size: int, what: str) -> np.ndarray:
    if type(vector) is not np.ndarray or vector.dtype != np.float64 or vector.shape != (size,):
        raise _refuse('COUPLED_FORMATION_RANGE', 'type', f'{what} must be a float64 ndarray of length {size}')
    if not np.isfinite(vector).all():
        raise _refuse('COUPLED_FORMATION_RANGE', 'nonfinite', f'{what} is not finite')
    return vector


_WINDOW_LOW_BITS = _bits_of(2.0 ** -400)
_WINDOW_HIGH_BITS = _bits_of(2.0 ** 64)


def _check_window(values: np.ndarray, what: str) -> None:
    """Every nonzero lies in [2^-400, 2^64], decided on the bit patterns (magnitudes of binary64
    values order as their sign-cleared bits), so a subnormal is never mistaken for zero."""
    magnitude = _float_bits(values) & _ABS_MASK
    magnitude = magnitude[magnitude != 0]
    if magnitude.size and (int(magnitude.min()) < _WINDOW_LOW_BITS or int(magnitude.max()) > _WINDOW_HIGH_BITS):
        raise _refuse('COUPLED_FORMATION_RANGE', 'raw_window', f'a nonzero of {what} lies outside [2^-400, 2^64]')


def _exact_product(C: sparse.csr_matrix, T: sparse.csr_matrix) -> List[Dict[int, Fraction]]:
    T = T.tocsr()
    rows = []
    for i in range(C.shape[0]):
        acc: Dict[int, Fraction] = {}
        for k in range(C.indptr[i], C.indptr[i + 1]):
            c = _q(C.data[k])
            col = C.indices[k]
            for p in range(T.indptr[col], T.indptr[col + 1]):
                j = int(T.indices[p])
                acc[j] = acc.get(j, Fraction(0)) + c * _q(T.data[p])
        rows.append(acc)
    return rows


def _rounded(value: Fraction, what: str) -> float:
    result = _to_float(value)
    if value != 0 and not (W_RAW_LOW <= abs(_q(result)) <= W_RAW_HIGH if math.isfinite(result) else False):
        raise _refuse('COUPLED_FORMATION_LOSS', 'formation_loss',
                      f'an exactly nonzero entry of {what} rounds outside the raw window')
    return result


@dataclass(frozen=True)
class _Formed:
    system: sparse.csr_matrix
    rhs: np.ndarray
    B_s_exact: List[Dict[int, Fraction]]
    sizes: Mapping[str, int]


def _form_system(*, K, F, K_s, f_s, T, u0, C_shell, C_solid, K_v, f_v) -> _Formed:
    """C1: input checks, the raw window, exact B_s = C_shell T and g = -C_shell u0, symmetric
    stiffness blocks, and the stored KKT system S z = b."""
    K = _require_csr(K, 'K')
    K_s = _require_csr(K_s, 'K_s')
    T = _require_csr(T, 'T')
    C_shell = _require_csr(C_shell, 'C_shell')
    C_solid = _require_csr(C_solid, 'C_solid')
    K_v = _require_csr(K_v, 'K_v')
    n_full, n_s = T.shape
    n_v = K_v.shape[0]
    m = C_shell.shape[0]
    if K.shape != (n_full, n_full) or K_s.shape != (n_s, n_s) or K_v.shape != (n_v, n_v) \
            or C_shell.shape != (m, n_full) or C_solid.shape != (m, n_v):
        raise _refuse('COUPLED_FORMATION_RANGE', 'shape', 'inconsistent coupled block shapes')
    F = _require_vector(F, n_full, 'F')
    f_s = _require_vector(f_s, n_s, 'f_s')
    u0 = _require_vector(u0, n_full, 'u0')
    f_v = _require_vector(f_v, n_v, 'f_v')
    for values, what in ((K.data, 'K'), (F, 'F'), (K_s.data, 'K_s'), (f_s, 'f_s'), (T.data, 'T'), (u0, 'u0'),
                         (C_shell.data, 'C_shell'), (C_solid.data, 'C_solid'), (K_v.data, 'K_v'), (f_v, 'f_v')):
        _check_window(values, what)
    B_exact = _exact_product(C_shell, T)
    rows, cols, data = [], [], []
    for i, acc in enumerate(B_exact):
        for j in sorted(acc):
            if acc[j] != 0:
                rows.append(i)
                cols.append(j)
                data.append(_rounded(acc[j], 'B_s'))
    B_s = sparse.csr_matrix((np.asarray(data, dtype=float), (rows, cols)), shape=(m, n_s))
    g = np.zeros(m)
    u0_nonzero = _nonzero_mask(u0)
    for i in range(m):
        acc = Fraction(0)
        for k in range(C_shell.indptr[i], C_shell.indptr[i + 1]):
            j = C_shell.indices[k]
            if u0_nonzero[j]:
                acc -= _q(C_shell.data[k]) * _q(u0[j])
        g[i] = _rounded(acc, 'g')
    blocks = []
    for name, block in (('K_s', K_s), ('K_v', K_v)):
        if _asymmetric(block):
            raise _refuse('COUPLED_UNCERTIFIED', 'asymmetric_stiffness', f'{name} is not symmetric to 1e-10')
        symmetric = ((block + block.T) * 0.5).tocsr()
        symmetric.sum_duplicates()
        _check_window(symmetric.data, f'{name} (symmetrised)')
        blocks.append(symmetric)
    S = sparse.bmat([[blocks[0], None, B_s.T], [None, blocks[1], -C_solid.T], [B_s, -C_solid, None]],
                    format='csr', dtype=np.float64)
    if S.shape[0] == 0:
        S = sparse.csr_matrix((0, 0))
    S.sum_duplicates()
    b = np.concatenate([f_s, f_v, g]).astype(np.float64)
    return _Formed(S, b, B_exact, {'n_s': n_s, 'n_v': n_v, 'm': m, 'N': n_s + n_v + m, 'n_full': n_full})


def _check_zero_rows(S: sparse.csr_matrix) -> None:
    counts = np.diff(S.indptr)
    nonzero = _nonzero_mask(S.data)
    zero = [i for i in range(S.shape[0]) if counts[i] == 0 or not nonzero[S.indptr[i]:S.indptr[i + 1]].any()]
    if zero:
        raise _refuse('COUPLED_MECHANISM', 'zero_row', f'{len(zero)} rows of the coupled system are zero',
                      rows=zero[:20])


def _check_solid_rigid_rank(C_solid: sparse.csr_matrix, rigid: np.ndarray) -> int:
    """rank(C_solid R_v) must equal the number of rigid modes (6 per connected solid part)."""
    product = np.asarray(C_solid @ rigid, dtype=float)
    rank = int(np.linalg.matrix_rank(product)) if product.size else 0
    if rank < rigid.shape[1] or rigid.shape[1] < 6:
        raise _refuse('COUPLED_MECHANISM', 'solid_rigid_mode_free',
                      f'the connector restrains {rank} of the {rigid.shape[1]} solid rigid modes')
    return rank


# --- section 4: connector prechecks -------------------------------------------------------------

def _knots(breakpoints: np.ndarray) -> np.ndarray:
    """Every second shell edge node, corners included; with an odd count the last segment spans
    one edge."""
    indices = list(range(0, len(breakpoints), 2))
    if indices[-1] != len(breakpoints) - 1:
        indices.append(len(breakpoints) - 1)
    return np.asarray(breakpoints, dtype=float)[indices]


def _connector_prechecks(knots: np.ndarray, stations: np.ndarray, length: float, tolerance: float) -> None:
    if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all() or np.any(np.diff(knots) <= tolerance) \
            or abs(knots[0]) > tolerance or abs(knots[-1] - length) > tolerance:
        raise _refuse('COUPLED_CONNECTOR_PARTITION', 'knots', 'the multiplier partition is invalid')
    for a, b in zip(knots[:-1], knots[1:]):
        columns = int(np.count_nonzero((stations[:-1] < b) & (stations[1:] > a)))
        if columns < 2:
            raise _refuse('COUPLED_CONNECTOR_PARTITION', 'segment_columns',
                          f'a multiplier segment meets {columns} solid lattice column(s), at least 2 are required')
    if _SECTION_MODES * len(knots) > _MAX_CONNECTOR_ROWS:
        raise _refuse('COUPLED_CONNECTOR_RANK', 'row_limit',
                      f'{_SECTION_MODES * len(knots)} multiplier rows exceed the {_MAX_CONNECTOR_ROWS}-row limit')


def _group_precheck(mesh: Any, name: Any) -> None:
    if not isinstance(name, str) or name not in mesh.boundary_groups:
        raise _refuse('COUPLED_CONNECTOR_GROUP', 'unknown_group', f'unknown solid boundary group {name!r}')
    count = len(_group_facets(mesh, name))
    if not 1 <= count <= _MAX_CONNECTOR_FACETS:
        raise _refuse('COUPLED_CONNECTOR_GROUP', 'facet_count',
                      f'group {name!r} has {count} facets (1..{_MAX_CONNECTOR_FACETS} allowed)')


def _build_connector(trace, prepared, name, knots, n_shell, n_solid):
    from anysolid.coupling import build_weak_connector
    try:
        operator = build_weak_connector(trace, prepared, name, multiplier_breakpoints=knots)
    except CoupledSolveError:
        raise
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory in the connector') from exc
    except ValueError as exc:
        raise _refuse('COUPLED_CONNECTOR_INVALID', 'connector', f'connector refused: {_safe_text(exc)}') from exc
    # Exact types before any conversion (r1 class 6): vstack would upcast a float32 block.
    C_shell = _require_csr(operator.C_shell, f'C_shell of {name!r}')
    C_solid = _require_csr(operator.C_solid, f'C_solid of {name!r}')
    rows = C_shell.shape[0]
    # Dimensions before use (review r2, C1 structure): one affine entry per multiplier row.
    if rows < 1 or C_shell.shape != (rows, n_shell) or C_solid.shape != (rows, n_solid):
        raise _refuse('COUPLED_CONNECTOR_INVALID', 'dimensions', f'the connector blocks of {name!r} have wrong shapes')
    affine = operator.affine
    if type(affine) is not np.ndarray or affine.dtype != np.float64 or affine.shape != (rows,) or _any_nonzero(affine):
        raise _refuse('COUPLED_CONNECTOR_INVALID', 'affine', 'the connector affine term is not exactly zero')
    return C_shell, C_solid


# --- recovery bounds ------------------------------------------------------------------------------

def _bounded_rows(matrix_rows: Sequence[Sequence[Tuple[int, Fraction]]], offsets: Optional[Sequence[Fraction]],
                  z: np.ndarray, shifts: np.ndarray, error: Fraction) -> Bounded:
    """values_i = sum_j M_ij z_j (+ offset_i) exactly, rounded once; bound
    |float - exact| + sqrt_hi(sum_j (M_ij d_j)^2) E."""
    values, bounds = [], []
    zq = {}
    for r, row in enumerate(matrix_rows):
        exact = offsets[r] if offsets is not None else Fraction(0)
        square = Fraction(0)
        for j, coefficient in row:
            if j not in zq:
                zq[j] = _q(z[j])
            exact += coefficient * zq[j]
            square += (coefficient * Fraction(2) ** int(shifts[j])) ** 2
        value = _to_float(exact)
        values.append(value)
        rounding = abs(_q(value) - exact) if math.isfinite(value) else Fraction(0)
        bounds.append(_out_up(rounding + _sqrt_hi(square) * error))
    values = np.asarray(values, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if not (np.isfinite(values).all() and np.isfinite(bounds).all()):
        raise _refuse('COUPLED_UNCERTIFIED', 'bound_not_finite', 'a recovered value or bound is not finite')
    return Bounded(values, bounds)


def _csr_rows(matrix: sparse.csr_matrix, col_offset: int = 0) -> List[List[Tuple[int, Fraction]]]:
    matrix = matrix.tocsr()
    return [[(int(matrix.indices[k]) + col_offset, _q(matrix.data[k])) for k in range(matrix.indptr[i], matrix.indptr[i + 1])]
            for i in range(matrix.shape[0])]


# --- contract properties the route assumes (review r1 #2, class 2) --------------------------------

def _is_int(value: Any) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))


def _array(value: Any, dtype: Any, shape: Tuple[Optional[int], ...]) -> bool:
    return (type(value) is np.ndarray and value.dtype == dtype and value.ndim == len(shape)
            and all(s is None or s == d for s, d in zip(shape, value.shape)))


def _check_prepared(prepared: Any) -> int:
    """The PreparedSolid properties the formulation assumes, checked before any of them is used:
    P2 (order 2) on straight tetrahedra in metres, and consistent coefficient arrays. Returns
    the number of solid coefficients."""
    def bad(reason: str, message: str) -> CoupledSolveError:
        return _refuse('COUPLED_NATIVE_INVALID', reason, message)
    order = prepared.order
    if not _is_int(order) or int(order) != 2:
        raise bad('prepared_order', 'the coupled route is formulated for a P2 solid (prepared.order == 2)')
    mesh = prepared.mesh
    geometry_order = mesh.geometry_order
    if not _is_int(geometry_order) or int(geometry_order) != 1 or type(mesh.units) is not str or mesh.units != 'm':
        raise bad('prepared_mesh', "the solid mesh must have geometry_order 1 and units 'm'")
    if not (_array(mesh.coordinates, np.float64, (None, 3)) and np.isfinite(mesh.coordinates).all()
            and _array(mesh.cells, np.int64, (None, 4)) and _array(mesh.boundary_facets, np.int64, (None, 3))
            and _array(mesh.boundary_ids, np.int64, (len(mesh.boundary_facets),))):
        raise bad('prepared_mesh', 'the solid mesh arrays have unexpected types or shapes')
    ndof = prepared.ndof
    if not _is_int(ndof) or int(ndof) < 1:
        raise bad('prepared_ndof', 'prepared.ndof must be a positive integer')
    ndof = int(ndof)
    coordinates, components = prepared.coefficient_coordinates, prepared.coefficient_components
    if not (_array(coordinates, np.float64, (ndof, 3)) and np.isfinite(coordinates).all()
            and _array(components, np.int64, (ndof,)) and np.isin(components, (0, 1, 2)).all()):
        raise bad('prepared_coefficients', 'the solid coefficient coordinates/components are inconsistent')
    # Review r2 P1: the coefficient layout of the P2 topology, not only the order label. Vector P2 on
    # tetrahedra has one coefficient per vertex and per edge for each of the 3 components, and the
    # vertex coefficients sit at the vertices (a relabelled P1 basis has 3 V coefficients).
    cells = mesh.cells
    vertices = np.unique(cells)
    if len(vertices) != len(mesh.coordinates) or (len(vertices) and (vertices[0] != 0 or vertices[-1] != len(vertices) - 1)):
        raise bad('prepared_layout', 'every solid vertex must belong to a cell')
    edges = np.unique(np.sort(cells[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), axis=1), axis=0)
    per_component = len(vertices) + len(edges)
    if ndof != 3 * per_component or any(int(np.count_nonzero(components == c)) != per_component for c in range(3)):
        raise bad('prepared_layout', f'{ndof} solid coefficients do not match the P2 layout 3 x ({len(vertices)} '
                                     f'vertices + {len(edges)} edges)')
    keys = {tuple(_float_bits(row).tolist()) for row in mesh.coordinates}
    at_vertex = sum(1 for row in coordinates if tuple(_float_bits(row).tolist()) in keys)
    if at_vertex < 3 * len(vertices):
        raise bad('prepared_layout', 'the solid coefficients do not include the 3 vertex coefficients per vertex')
    return ndof


def _check_model_dofs(model: Any) -> None:
    """Every node carries six distinct DOFs equal to the model's DOF manager mapping (MPC terms
    and the holed renumbering are read through it)."""
    manager = model.mesh.dof_manager
    seen = set()
    for nid, node in model.mesh.nodes.items():
        dofs = list(node.dofs)
        if len(dofs) != 6 or not all(_is_int(d) for d in dofs) or dofs != list(manager.get_node_dofs(nid)):
            raise _refuse('COUPLED_NATIVE_INVALID', 'dof_map', f'node {nid} has an inconsistent DOF mapping')
        seen.update(int(d) for d in dofs)
    if len(seen) != 6 * len(model.mesh.nodes):
        raise _refuse('COUPLED_NATIVE_INVALID', 'dof_map', 'two nodes share a DOF')


def _resources_limit(resources: Any) -> Optional[int]:
    if resources is None:
        return None
    limit = getattr(resources, 'memory_limit_bytes', None)
    if limit is None:
        return None
    if not _is_int(limit):
        raise _refuse('COUPLED_NATIVE_INVALID', 'resources', 'memory_limit_bytes must be a non-negative integer')
    value = int(limit)                  # converted once; an exact int from here on (review r3 P2)
    if type(value) is not int or value < 0:
        raise _refuse('COUPLED_NATIVE_INVALID', 'resources', 'memory_limit_bytes must be a non-negative integer')
    return value


def _float_nonzero(value: Any) -> bool:
    return bool(_bits_of(float(value)) & 0x7FFF_FFFF_FFFF_FFFF)


def _raw_inputs_zero(model: Any, load_case: Any, holed: Any, pressure: Optional[float],
                     ledger: Sequence[Mapping[str, Any]], F: np.ndarray, u0: np.ndarray) -> bool:
    """True iff every raw input the route consumes is exactly zero, decided on the original values
    (review r2 (d): never on a float image of them) and on bit patterns for floats.

    Consumed: the native F and u0; every pressure of the case (kept and transferred); the first
    min(L, 6) components of every nodal record on a kept node; the prescribed values of every
    support that names at least one node (review r2 (c): a support with no nodes prescribes
    nothing, so native reduction consumes none of its values); every MPC right-hand side."""
    if _any_nonzero(F) or _any_nonzero(u0):
        return False
    if pressure is not None and _scalar_nonzero(pressure):
        return False
    if any(_scalar_nonzero(p) for p in load_case.pressure_loads.values()):
        return False
    for nid, load in load_case.nodal_loads.items():
        if nid in holed.mesh.nodes and any(_scalar_nonzero(v) for v in list(load)[:6]):
            return False
    for row in ledger:
        if any(_float_nonzero(value) for _, _, value in row.get('contributions', ())):
            return False
    for bc in model.boundary_conditions:
        if len(bc.node_ids) and any(_scalar_nonzero(value) for value in bc.dof_constraints.values()):
            return False
    return not any(_scalar_nonzero(equation.value) for equation in model.constraint_equations)


# --- footprint overlap by geometry (review r1 #3, class 3) ------------------------------------------

def _out_down(value: Fraction) -> float:
    """The largest binary64 <= value (independent of the FP environment)."""
    return -_out_up(-value)


def _hull_meets_open_box(points: Sequence[Tuple[Fraction, ...]], lo: Sequence[Fraction],
                         hi: Sequence[Fraction]) -> bool:
    """True iff the convex hull of ``points`` meets the open box (lo, hi), decided exactly by the
    separating-axis theorem: the box faces, the hull faces (all point triples and their in-plane
    edge normals, for a flat hull) and edge x box-axis directions are a complete axis set; an axis
    with touching projections separates the hull from the open box."""
    units = ((Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)),
             (Fraction(0), Fraction(0), Fraction(1)))
    pts = list(points)
    edges = [_sub(q, p) for i, p in enumerate(pts) for q in pts[i + 1:]]
    normals = [_cross(_sub(pts[j], pts[i]), _sub(pts[k], pts[i]))
               for i in range(len(pts)) for j in range(i + 1, len(pts)) for k in range(j + 1, len(pts))]
    axes = list(units) + [_cross(e, u) for e in edges for u in units] + normals \
        + [_cross(nu, e) for nu in normals for e in edges]
    for w in axes:
        if not any(w):
            continue
        projection = [_dot(w, p) for p in pts]
        box_lo = sum((min(w[k] * lo[k], w[k] * hi[k]) for k in range(3)), Fraction(0))
        box_hi = sum((max(w[k] * lo[k], w[k] * hi[k]) for k in range(3)), Fraction(0))
        if max(projection) <= box_lo or box_hi <= min(projection):
            return False
    return True


def _check_footprint_partition(model: Any, removed: Sequence[int], orphans: Sequence[int],
                               corners: Sequence[np.ndarray], normal: np.ndarray, thickness: float) -> None:
    """The removed set is derived from the footprint geometry (review r2 P1 (a)) and no kept shell
    reaches into the footprint (review r1 #3, r2 (b)), both decided exactly against the open prism

        P = {c0 + s a + t b + h n : 0 < s < 1, 0 < t < 1, |h| < thickness / 2}

    of the corner rectangle, with no tolerance. The rule is hull exclusion: an element meets P iff
    the convex hull of its nodes does (exact separating-axis test), so a warped kept element whose
    hull crosses P is refused even when its curved surface would clear it.

    - A removed element must have every node in the closed prism and meet P; the removed set must
      be exactly the elements that do (``removed_set_geometry``).
    - A kept element must not meet P, and a kept node must not lie in P (``kept_inside_footprint``).

    The prefilter is geometric and sound: an element whose node bounding box misses the bounding box
    of P (its 8 vertices exactly, rounded outward to binary64) cannot meet P."""
    c0 = _vec(corners[0])
    a, b, nq = _sub(_vec(corners[1]), c0), _sub(_vec(corners[3]), c0), _vec(normal)
    det = _dot(a, _cross(b, nq))
    if det == 0:
        raise _refuse('COUPLED_CUT_MISMATCH', 'degenerate_footprint', 'the footprint frame is degenerate')
    duals = (_cross(b, nq), _cross(nq, a), _cross(a, b))
    half = _q(thickness) / 2
    lo, hi = (Fraction(0), Fraction(0), -half), (Fraction(1), Fraction(1), half)
    vertices = [tuple(c0[k] + s * a[k] + t * b[k] + h * nq[k] for k in range(3))
                for s in (0, 1) for t in (0, 1) for h in (-half, half)]
    box_lo = np.array([_out_down(min(v[k] for v in vertices)) for k in range(3)])
    box_hi = np.array([_out_up(max(v[k] for v in vertices)) for k in range(3)])
    nodes = model.mesh.nodes
    ids = list(nodes)
    xyz = np.array([nodes[nid].coords() for nid in ids], dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise _refuse('COUPLED_CUT_MISMATCH', 'kept_element_geometry', 'a shell node coordinate is not finite')
    where = {nid: i for i, nid in enumerate(ids)}
    cache: Dict[int, Tuple[Fraction, Fraction, Fraction]] = {}

    def local(nid: int):
        if nid not in cache:
            r = _sub(_vec(xyz[where[nid]]), c0)
            cache[nid] = tuple(_dot(r, d) / det for d in duals)
        return cache[nid]

    def near(members: Sequence[int]) -> bool:
        rows = xyz[[where[n] for n in members]]
        return bool((rows.max(axis=0) >= box_lo).all() and (rows.min(axis=0) <= box_hi).all())

    removed_set = set(removed)
    orphan_set = set(orphans)
    for eid, element in model.mesh.elements.items():
        node_ids = getattr(element, 'node_ids', None)
        if node_ids is None or not len(node_ids) or not all(_is_int(n) and int(n) in where for n in node_ids):
            raise _refuse('COUPLED_CUT_MISMATCH', 'kept_element_geometry', f'element {eid} has no checkable nodes')
        members = tuple(dict.fromkeys(int(n) for n in node_ids))
        meets = near(members) and _hull_meets_open_box([local(n) for n in members], lo, hi)
        if eid in removed_set:
            closed = meets and all(lo[k] <= p[k] <= hi[k] for p in map(local, members) for k in range(3))
            if not closed:
                raise _refuse('COUPLED_CUT_MISMATCH', 'removed_set_geometry',
                              f'removed element {eid} is not an element of the footprint')
        elif meets:
            raise _refuse('COUPLED_CUT_MISMATCH', 'kept_inside_footprint', f'kept element {eid} overlaps the footprint interior')
    for nid in ids:
        if nid in orphan_set or not near((nid,)):
            continue
        p = local(nid)
        if all(lo[k] < p[k] < hi[k] for k in range(3)):
            raise _refuse('COUPLED_CUT_MISMATCH', 'kept_inside_footprint', f'node {nid} lies in the footprint interior')


# --- the entry ------------------------------------------------------------------------------------

_NATIVE_ERRORS = (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError, KeyError, TypeError,
                  IndexError, AttributeError, RuntimeError)


def _native(call, *args):
    try:
        return call(*args)
    except CoupledSolveError:
        raise
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory in native assembly') from exc
    except _NATIVE_ERRORS as exc:
        raise _refuse('COUPLED_NATIVE_INVALID', _safe_name(exc),
                      f'native shell assembly refused: {_safe_text(exc)}') from exc


def solve_shell_solid_coupled(fe_model: Any, prepared: Any, *, removed_element_ids: Sequence[int], box: BoxFaces,
                              load_case: Any, cancel: Any = None, resources: Any = None,
                              clock: Optional[Callable[[], float]] = None) -> CoupledSolution:
    """Solve the shell (minus the removed elements) and the floating solid detail coupled along the
    cut; certify and verify the stored system exactly, or refuse with a ``COUPLED_*`` code.

    Every refusal is a :class:`CoupledSolveError`. Every ``Exception`` is translated into one;
    ``KeyboardInterrupt``, ``SystemExit`` and other ``BaseException`` subclasses that are not
    ``Exception`` propagate unchanged (owner decision OD9, docs/SOLID_COUPLING.md).

    Trusted-construction boundary (OD9): the claims cover untouched objects built by ANYfem's
    pipeline only (``PreparedSolid`` from ``anysolid.prepare``, the FEModel from ANYfem's builder,
    the cut from ``cut_solid_submodel``). Objects modified after construction are out of scope."""
    try:
        return _solve(fe_model, prepared, removed_element_ids, box, load_case, cancel, resources,
                      time.perf_counter if clock is None else clock)   # identity, no __bool__ (r4 P2)
    except CoupledSolveError:
        raise
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory in the coupled solve') from exc
    except Exception as exc:  # no uncoded exception escapes; the translation itself cannot raise (r1 #9)
        raise _refuse('COUPLED_NATIVE_INVALID', f'unexpected:{_safe_name(exc)}',
                      f'the coupled solve failed unexpectedly ({_safe_text(exc)})') from exc


def _solve(fe_model, prepared, removed_element_ids, box, load_case, cancel, resources, clock) -> CoupledSolution:
    from anysolid.backend import PreparedSolid
    from .boundary import LoadCase
    from .fe_core import FEModel
    from .assembly import assemble_system, build_constraint_transformation
    from .solid_trace import ShellEdgeTrace

    # Caller code runs only in the callbacks (review r3 P2): every clock call, the cancellation
    # check (getattr of is_set, then each flag read), the resources attribute and its integer
    # conversion, and the iteration and integer conversion of removed_element_ids and the box
    # divisions. Each is followed by the FP-environment probe before any float step runs.
    clock = _checked_clock(clock)
    timings: Dict[str, float] = {}
    t_start = clock()
    check = _cancel_probe(cancel)

    def poll():
        # Cancellation, then the FP environment (a callback or a lazily imported extension may have
        # switched on FTZ/DAZ; r1 class 1).
        if check is not None:
            check()
        _require_fp_environment()

    _require_fp_environment()
    # Exact types (r11 #5, #8-#10): no caller arrays or protocol objects reach the route.
    if type(fe_model) is not FEModel:
        raise _refuse('COUPLED_NATIVE_INVALID', 'model_type', 'fe_model must be an anyfesolver FEModel')
    if type(prepared) is not PreparedSolid:
        raise _refuse('COUPLED_NATIVE_INVALID', 'prepared_type', 'prepared must be an anysolid PreparedSolid')
    if type(load_case) is not LoadCase:
        raise _refuse('COUPLED_NATIVE_INVALID', 'load_case_type', 'load_case must be one anyfesolver LoadCase')
    if type(box) is not BoxFaces or type(box.sides) is not tuple or len(box.sides) != 4 \
            or not all(type(s) is str for s in box.sides) or len(set(box.sides)) != 4 \
            or not (box.skin is None or (type(box.skin) is str and box.skin in _SKINS)) \
            or type(box.divisions) is not tuple or len(box.divisions) != 3:
        raise _refuse('COUPLED_MESH_PROVENANCE', 'box', 'box must be BoxFaces with 3 divisions, a skin and 4 sides')
    divisions = tuple(_exact_int(v, 'box division', 'COUPLED_MESH_PROVENANCE') for v in box.divisions)
    if min(divisions) < 1:
        raise _refuse('COUPLED_MESH_PROVENANCE', 'box', 'box divisions must be positive')
    if set(box.sides) & set(_SKINS):
        raise _refuse('COUPLED_MESH_PROVENANCE', 'box', 'side names must differ from the skin names')
    memory_limit = _resources_limit(resources)
    _require_fp_environment()          # after the box divisions and the resources callback (review r3 P2)
    # Contract properties the formulation assumes (r1 #2, class 2), before any of them is used.
    n_v = _check_prepared(prepared)
    _check_model_dofs(fe_model)
    poll()

    # Section 3.2 load route, section 4 cut and targets.
    _check_load_route(fe_model, load_case)
    cut = _analyse_cut(fe_model, removed_element_ids)
    _check_targets(fe_model, cut, load_case)
    pressure = _skin_pressure(fe_model, cut, load_case, box)
    holed, _ = _native(_holed_model, fe_model, cut)
    holed_case = copy.deepcopy(load_case)
    for eid in cut.removed:
        holed_case.pressure_loads.pop(eid, None)
    for nid in cut.orphans:
        holed_case.added_node_masses.pop(nid, None)
    poll()

    # Traces per straight chain (on the holed model) and the shell frame.
    traces = []
    for chain in cut.chains:
        traces.append(ShellEdgeTrace(holed, chain, offset=0.0))
    n = np.asarray(traces[0].normal, dtype=float)
    for trace in traces:
        if np.max(np.abs(np.asarray(trace.normal) - n)) > NORMAL_TOLERANCE or trace.thickness != traces[0].thickness:
            raise _refuse('COUPLED_CUT_MISMATCH', 'mixed_faces', 'the cut sides need one director and thickness')
    if np.max(np.abs(cut.normal - n)) > NORMAL_TOLERANCE or cut.thickness != traces[0].thickness:
        raise _refuse('COUPLED_CUT_MISMATCH', 'mixed_faces', 'the removed and kept shells need one director and thickness')
    thickness = float(traces[0].thickness)
    node_xyz = lambda nid: np.asarray(fe_model.mesh.nodes[nid].coords(), dtype=float)  # noqa: E731
    # Loop corners in loop order; the loop runs along the kept edges, clockwise about n for
    # counter-clockwise kept elements. Order the footprint counter-clockwise about n.
    loop_corners = [node_xyz(first) for first, _ in cut.corners]
    area = sum(np.cross(loop_corners[i], loop_corners[(i + 1) % 4]) @ n for i in range(4))
    if area > 0:
        ccw_chains = list(range(4))
        ccw_corner = [cut.corners[k][0] for k in range(4)]
    else:
        ccw_chains = [(-k - 1) % 4 for k in range(4)]          # chain j reversed: c_j = its last node
        ccw_corner = [cut.corners[ccw_chains[k]][1] for k in range(4)]
    mesh = prepared.mesh
    for name in _SKINS:
        if name not in mesh.boundary_groups:
            raise _refuse('COUPLED_MESH_PROVENANCE', 'skin_group', f'the solid has no {name!r} face group')
    for name in box.sides:
        _group_precheck(mesh, name)
    # Match sides[0] to the nearest chain line, then require the counter-clockwise order.
    coordinates = np.asarray(mesh.coordinates, dtype=float)
    centre_side = coordinates[np.unique(_group_facets(mesh, box.sides[0]))].mean(axis=0)
    distances = []
    for j in range(4):
        a, b = node_xyz(ccw_corner[j]), node_xyz(ccw_corner[(j + 1) % 4])
        distances.append(np.linalg.norm(centre_side - (a + b) / 2))
    j0 = int(np.argmin(distances))
    corners = tuple(node_xyz(ccw_corner[(j0 + k) % 4]) for k in range(4))
    side_traces = [traces[ccw_chains[(j0 + k) % 4]] for k in range(4)]
    # The stored footprint centre (r1 #6, class 5): the corner mean rounded once to binary64; the
    # exported centre is exactly the point every moment, V4 and the transfer gate refer to.
    centre_float = np.array([_to_float(sum((_q(c[i]) for c in corners), Fraction(0)) / 4) for i in range(3)])
    centre = _vec(centre_float)
    diagonal_lo = _sqrt_lo(_norm2(_sub(_vec(corners[2]), _vec(corners[0]))))
    frame = _Frame(corners, centre, n, thickness, diagonal_lo, tuple(cut.chains[ccw_chains[(j0 + k) % 4]] for k in range(4)))
    _check_footprint_partition(fe_model, cut.removed, cut.orphans, corners, n, thickness)
    outwards = [np.asarray(tr.solid_outward, dtype=float) for tr in side_traces]
    t0 = clock()
    box_checked = BoxFaces(divisions, box.skin, box.sides)
    grids = _check_box(mesh, box_checked, frame, outwards)
    timings['provenance'] = clock() - t0
    poll()

    # Skin traction, echo and the transfer resultant gates.
    echo: Dict[str, Any] = {'skin': box.skin, 'pressure': pressure, 'offset': None, 'traction': None}
    if pressure is None:
        f_v = np.zeros(n_v)
    else:
        q = pressure * n
        echo.update(offset=(-0.5 if box.skin == 'start' else 0.5) * thickness, traction=q.tolist())
        # The exact type is checked before any use or conversion (r1 #7, class 6).
        f_v = _require_vector(prepared.traction(box.skin, q), n_v, 'the skin traction f_v')
        _transfer_gate(copy.deepcopy(fe_model), cut, load_case, prepared, f_v, centre, diagonal_lo)
    poll()

    # Native shell: unchanged assembly and reduction on the holed model.
    t0 = clock()
    K, F, _info = _native(assemble_system, holed, holed_case)
    active = int(holed.mesh.dof_manager.total_dofs)
    # K and F are validated before native reduction consumes them (review r2, C1 structure): exact
    # types, CSR structure, shape, finiteness and the raw window. The original objects go on.
    K_checked = _require_csr(K, 'K')
    if K_checked.shape != (active, active):
        raise _refuse('COUPLED_FORMATION_RANGE', 'shape', 'K does not match the holed shell DOFs')
    _check_window(K_checked.data, 'K')
    F = _require_vector(F, active, 'the native load F')
    _check_window(F, 'F')
    del K_checked
    K_s, f_s, T, u0, independent, _reduction = _native(build_constraint_transformation, K, F, holed)
    timings['native'] = clock() - t0
    # Native outputs: exact types before any use (r1 class 6; previously T.tocsr() ran first).
    T = _require_csr(T, 'T')
    if T.shape[0] != active:
        raise _refuse('COUPLED_FORMATION_RANGE', 'shape', 'T does not match the holed shell DOFs')
    u0 = _require_vector(u0, active, 'u0')
    if type(independent) is not np.ndarray or not np.issubdtype(independent.dtype, np.integer) \
            or independent.shape != (T.shape[1],):
        raise _refuse('COUPLED_FORMATION_RANGE', 'type', 'the independent DOF list must be an integer array')
    ledger = list(_ledger(holed, holed_case, active, F))
    skin_entries = [(int(i), float(f_v[i])) for i in np.flatnonzero(_nonzero_mask(f_v))]
    for eid in cut.removed:
        if eid in load_case.pressure_loads:
            ledger.append({'record': 'pressure', 'id': eid, 'fate': 'transferred', 'group': box.skin,
                           'f_v_entries': skin_entries})
    for nid in getattr(load_case, 'added_node_masses', {}) or {}:
        ledger.append({'record': 'added_node_mass', 'id': nid, 'fate': 'inert'})
    ledger = tuple(ledger)
    poll()

    # Connectors per side.
    t0 = clock()
    shell_rows, solid_rows, side_rows = [], [], []
    for k, name in enumerate(box.sides):
        trace = side_traces[k]
        knots = _knots(np.asarray(trace.breakpoints, dtype=float))
        grid = grids[name]
        stations = trace.chart(coordinates[grid[:, 0]])[:, 0]
        if stations[0] > stations[-1]:
            stations = stations[::-1]
        _connector_prechecks(knots, stations, float(trace.length), float(trace.tolerance))
        Ck_shell, Ck_solid = _build_connector(trace, prepared, name, knots, active, n_v)
        shell_rows.append(Ck_shell)
        solid_rows.append(Ck_solid)
        side_rows.append(Ck_shell.shape[0])
        poll()
    C_shell = sparse.vstack(shell_rows, format='csr')
    C_solid = sparse.vstack(solid_rows, format='csr')
    timings['connector'] = clock() - t0
    # The C_shell column rule. A T row is active iff it holds a nonzero value, decided on the bits
    # (review r2 P1 (c)): a stored explicit 0.0 must not make an inactive DOF look active.
    t_rows = np.zeros(T.shape[0], dtype=bool)
    t_rows[np.repeat(np.arange(T.shape[0]), np.diff(T.indptr))[_nonzero_mask(T.data)]] = True
    u0_nonzero = _nonzero_mask(u0)
    for j in np.unique(C_shell.indices[_nonzero_mask(C_shell.data)]):
        if not t_rows[j] and not u0_nonzero[j]:
            raise _refuse('COUPLED_TARGET_REMOVED', 'connector_column', f'connector DOF {int(j)} is inactive')
    poll()

    # C1 formation and the coupled system.
    n_total = int(T.shape[1]) + n_v + int(C_shell.shape[0])
    if n_total > COUPLED_MAX_EQUATIONS:
        raise _refuse('COUPLED_TOO_LARGE', 'equation_cap',
                      f'the coupled system has {n_total} equations (cap {COUPLED_MAX_EQUATIONS})', size=n_total)
    if memory_limit is not None and _dense_bytes(n_total) > int(memory_limit):
        raise _refuse('COUPLED_TOO_LARGE', 'memory_limit', 'the dense certificate exceeds the memory limit')
    t0 = clock()
    K_v = prepared.assemble_linear()
    formed = _form_system(K=K, F=F, K_s=K_s, f_s=f_s, T=T, u0=u0, C_shell=C_shell, C_solid=C_solid, K_v=K_v, f_v=f_v)
    _check_zero_rows(formed.system)
    modes = prepared.rigid_modes()
    if not (_array(modes, np.float64, (n_v, None)) and modes.shape[1] % 6 == 0 and np.isfinite(modes).all()):
        raise _refuse('COUPLED_NATIVE_INVALID', 'prepared_rigid_modes', 'the solid rigid modes are not float64 (n_v, 6k)')
    _check_solid_rigid_rank(C_solid, modes)
    timings['formation'] = clock() - t0
    sizes = dict(formed.sizes)
    # Zero-load path only when every consumed raw input is exactly zero (r1 #4, class 4): the native
    # F and u0, every pressure (kept and transferred), every consumed nodal component, every
    # prescribed support value and MPC right-hand side.
    raw_zero = _raw_inputs_zero(fe_model, load_case, holed, pressure, ledger, F, u0)
    poll()
    solved = certified_solve(formed.system, formed.rhs, raw_load_zero=raw_zero, memory_limit_bytes=memory_limit,
                             cancel=check, clock=clock)
    timings.update({f'certified_{k}': v for k, v in solved.timings.items()})

    # Recovery with bounds (section 5).
    t0 = clock()
    z, shifts, E = solved.x, solved.shifts, solved.forward_error
    n_s, m = sizes['n_s'], sizes['m']
    sl_v = slice(n_s, n_s + n_v)
    sl_l = slice(n_s + n_v, n_s + n_v + m)
    identity = lambda count, offset: [[(offset + i, Fraction(1))] for i in range(count)]  # noqa: E731
    solid = _bounded_rows(identity(n_v, n_s), None, z, shifts, E)
    lam = _bounded_rows(identity(m, n_s + n_v), None, z, shifts, E)
    shell = _bounded_rows(_csr_rows(T), [_q(v) for v in u0], z, shifts, E)
    shell_force = _bounded_rows(_csr_rows(C_shell.T.tocsr(), n_s + n_v), None, z, shifts, E)
    solid_force = _bounded_rows(_csr_rows(C_solid.T.tocsr(), n_s + n_v), None, z, shifts, E)
    poll()

    # Common rigid basis; V4 and the per-side resultants.
    manager = holed.mesh.dof_manager
    shell_coords, shell_names = [], []
    for dof in range(active):
        nid, _, name = manager.get_dof_info(dof)
        shell_coords.append(holed.mesh.nodes[nid].coords())
        shell_names.append(name)
    R_s = _rigid_basis(shell_coords, shell_names, centre)
    solid_coords = prepared.coefficient_coordinates      # float64 (n_v, 3) by _check_prepared
    R_v = _rigid_basis(solid_coords, [_DOF_NAMES[int(c)] for c in prepared.coefficient_components], centre)
    # The exported forward error is the bounded E rounded up, never below it (r1 class 5).
    verification: Dict[str, Any] = {'backward': dict(solved.backward), 'forward_error': _out_up(E)}
    resultants = []
    start = 0
    lam_q = [_q(v) for v in z[sl_l]]
    worst_compat = Fraction(0)
    worst_balance = Fraction(0)
    for k, name in enumerate(box.sides):
        rows = slice(start, start + side_rows[k])
        Ck_s, Ck_v = C_shell[rows], C_solid[rows]
        CRs, CRv = _times_basis(Ck_s, R_s), _times_basis(Ck_v, R_v)
        scale = max(_abs_times_basis_inf(Ck_s, R_s), _abs_times_basis_inf(Ck_v, R_v))
        for a_row, b_row in zip(CRs, CRv):
            for a, b in zip(a_row, b_row):
                if abs(a - b) > V4_COMPATIBILITY_TOLERANCE * scale:
                    raise _refuse('COUPLED_UNCERTIFIED', 'v4_rigid_compatibility',
                                  f'side {name!r}: C_shell R_s and C_solid R_v differ')
                if scale:
                    worst_compat = max(worst_compat, abs(a - b) / scale)
        lk = lam_q[start:start + side_rows[k]]
        worst_balance = max(worst_balance, _v4_side_balance(name, CRs, CRv, lk))
        Q_rows = [[(n_s + n_v + start + i, -CRs[i][r]) for i in range(len(lk)) if CRs[i][r]] for r in range(6)]
        bounded = _bounded_rows(Q_rows, None, z, shifts, E)
        resultants.append(SideResultant(name, Bounded(bounded.values[:3], bounded.bounds[:3]),
                                        Bounded(bounded.values[3:], bounded.bounds[3:])))
        start += side_rows[k]
        poll()
    verification['v4_compatibility_max'] = _to_float(worst_compat)
    verification['v4_balance_max'] = _to_float(worst_balance)

    # V5: kept loads, reactions and the skin load balance (diagnostic, only adds refusals).
    u_full = shell.values
    reactions = np.asarray(K @ u_full + C_shell.T @ z[sl_l] - F, dtype=float)
    independent = np.asarray(independent, dtype=np.int64)
    dependent = np.setdiff1d(np.arange(active), independent)
    external = F.copy()
    external[dependent] += reactions[dependent]
    # Review r1 #5: an MPC slave's constraint force r_d acts on its masters as -T_dj r_d (reduced
    # equilibrium T^T r = 0); supports have empty T rows and add nothing here.
    master_forces = np.asarray(T[dependent].T @ reactions[dependent], dtype=float)
    external[independent] -= master_forces
    total = [Fraction(0)] * 6
    reference = Fraction(0)
    for vector, basis in ((external, R_s), (f_v, R_v)):
        for i in np.flatnonzero(_nonzero_mask(vector)):
            value = _q(vector[i])
            for r in range(6):
                term = value * basis[i][r]
                total[r] += term
                reference += abs(term)
    imbalance = max(abs(v) for v in total)
    if reference and imbalance > V5_BALANCE_TOLERANCE * reference:
        raise _refuse('COUPLED_UNCERTIFIED', 'v5_global_balance', 'reactions, loads and the skin load do not balance')
    verification['v5_balance'] = _to_float(imbalance / reference) if reference else 0.0
    timings['recovery'] = clock() - t0
    timings['total'] = clock() - t_start

    node_ids = tuple(int(nid) for nid in holed.mesh.nodes)
    order = np.concatenate([np.asarray(holed.mesh.nodes[nid].dofs, dtype=np.int64) for nid in node_ids]) \
        if node_ids else np.zeros(0, dtype=np.int64)
    pick = lambda b: Bounded(b.values[order], b.bounds[order])  # noqa: E731
    reaction_out = np.zeros(active)
    reaction_out[dependent] = reactions[dependent]
    result = CoupledSolution(
        shell_node_ids=node_ids, shell_displacements=pick(shell), solid_coefficients=solid, multipliers=lam,
        shell_interface_forces=pick(shell_force), solid_interface_forces=solid_force,
        side_resultants=tuple(resultants), reactions=reaction_out[order],
        footprint_centre=centre_float.copy(), echo=echo, ledger=ledger,
        certificate=dict(solved.certificate), verification=verification, sizes=sizes, timings=timings)
    # r1 #8 (class 7), r2 (d): the result is built first, then the last poll, so a cancellation during
    # recovery, verification or result construction never returns success.
    poll()
    return result


def _v4_side_balance(name: str, CRs: Sequence[Sequence[Fraction]], CRv: Sequence[Sequence[Fraction]],
                     lam: Sequence[Fraction]) -> Fraction:
    """V4 interface balance on one side (owner decision OD8, 2026-11-19): for every rigid component r,
    |sum_i (C_v R_v - C_s R_s)_ir lam_i| <= 1e-10 ref, exactly, with the per-side normwise reference
    ref = sum_i max_r (|C_s R_s| + |C_v R_v|)_ir |lam_i| (the OD7 form). Returns the worst ratio; a side
    with ref = 0 has a zero imbalance and is exempt (a nonzero imbalance there cannot occur)."""
    reference = sum((max(abs(a) + abs(b) for a, b in zip(CRs[i], CRv[i])) * abs(lam[i]) for i in range(len(lam))),
                    Fraction(0))
    worst = Fraction(0)
    for r in range(6):
        imbalance = abs(sum(((CRv[i][r] - CRs[i][r]) * lam[i] for i in range(len(lam))), Fraction(0)))
        if reference == 0:
            if imbalance:
                raise _refuse('COUPLED_UNCERTIFIED', 'internal_balance_inconsistency',
                              f'side {name!r}: a nonzero imbalance with a zero reference')
            continue
        if imbalance > V4_BALANCE_TOLERANCE * reference:
            raise _refuse('COUPLED_UNCERTIFIED', 'v4_interface_balance', f'side {name!r}: interface forces do not cancel')
        worst = max(worst, imbalance / reference)
    return worst


def _transfer_gate(model, cut, load_case, prepared, f_v, centre, diagonal_lo) -> None:
    """The native loads of the removed elements against ANYsolid's skin traction assembly: force to
    1e-12 |F_ref|, moment about the centre to 1e-12 (|F_ref| d + |M_ref|); both exactly zero when
    both references are."""
    manager = model.mesh.dof_manager
    F_ref = [Fraction(0)] * 3
    M_ref = [Fraction(0)] * 3
    for _, dof, value in _transferred(model, cut, load_case):
        nid, local, _ = manager.get_dof_info(dof)
        v = _q(value)
        if local < 3:
            F_ref[local] += v
            r = _sub(_vec(model.mesh.nodes[nid].coords()), centre)
            force = [Fraction(0)] * 3
            force[local] = v
            M_ref = [a + b for a, b in zip(M_ref, _cross(r, force))]
        else:
            M_ref[local - 3] += v
    F_sol = [Fraction(0)] * 3
    M_sol = [Fraction(0)] * 3
    coords = prepared.coefficient_coordinates      # float64 / int64 by _check_prepared
    components = prepared.coefficient_components
    for i in np.flatnonzero(_nonzero_mask(f_v)):
        c = int(components[i])
        v = _q(f_v[i])
        F_sol[c] += v
        force = [Fraction(0)] * 3
        force[c] = v
        M_sol = [a + b for a, b in zip(M_sol, _cross(_sub(_vec(coords[i]), centre), force))]
    if not any(F_ref) and not any(M_ref):
        if any(F_sol) or any(M_sol):
            raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'transfer_resultant', 'the skin load is nonzero for a zero reference')
        return
    dF = _norm2(_sub(F_sol, F_ref))
    if dF > TRANSFER_TOLERANCE ** 2 * _norm2(F_ref):
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'transfer_resultant', 'the skin force differs from the removed load')
    limit = TRANSFER_TOLERANCE * (_sqrt_lo(_norm2(F_ref)) * diagonal_lo + _sqrt_lo(_norm2(M_ref)))
    if _norm2(_sub(M_sol, M_ref)) > limit * limit:
        raise _refuse('COUPLED_LOAD_UNSUPPORTED', 'transfer_resultant', 'the skin moment differs from the removed load')
