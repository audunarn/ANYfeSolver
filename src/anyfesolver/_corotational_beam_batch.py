"""Stacked, bit-identical evaluation of rotated corotational 2-node beams.

``corotational_element_response`` evaluates one element at a time.  For a
plain elastic 2-node :class:`~anyfesolver.elements.BeamElement` with the
default ``"rotated"`` tangent, almost all of that time is spent in Python
overhead around 3x3 algebra.  This module evaluates the identical arithmetic
for all admitted beams at once:

* every BLAS product of the scalar path (``@``, ``np.dot``, ``np.linalg.norm``)
  is reproduced by a stacked ``np.matmul`` on operands with the *same
  per-item memory layout* (C order or transposed view), because OpenBLAS
  ``gemv`` results depend on the operand layout;
* every elementwise operation is evaluated in the same order;
* ``np.float64 ** 2`` (a libm ``pow`` call, not an exact square) is
  reproduced by ``np.power`` with an exponent *array*, which avoids NumPy's
  square fast path;
* transcendental ufuncs are trusted only after a once-per-process bitwise
  probe against the scalar functions.

Branches that the stacked code does not reproduce (antiparallel minimal
rotation, near-pi logarithm, any non-finite result) route that element back
to the scalar function.  Each evaluation additionally compares the first and
last batched element with the scalar oracle before anything is returned; a
mismatch discards the whole batch and disables it for the process.

The scalar path remains the oracle and the fallback.  Admission is fail
closed: exact ``BeamElement`` type, no instance overrides of class methods,
no fiber plasticity, generalized section, offsets or corotational local
formulation, existing reference and local caches, exact model/mesh types,
and only the ``rotated`` tangent mode.
"""

# --------------------------------------------------------------------------
# Admission inventory (trust boundary: NumPy, Python builtins, numba and
# ANYfeSolver's own classes/functions are trusted; every value reachable from
# the model, mesh, elements, nodes, materials, committed states, displacements
# and the hook's parameters is caller data).
#
# Rule A (exact type): a read is made only after ``type(x) is C`` proves the
#   exact class; ``is`` and ``type()`` run no caller code.  Type tests never
#   use ``in``/``==`` on type objects (a metaclass ``__eq__`` could run).
# Rule B (exact keys): before any ``in``, ``[]``, ``get`` or lookup on a
#   mapping, ``type(d) is dict`` (or exact ``_QualifiedStateMapping``) is
#   proven and every key is swept with ``all(t is K for t in map(type, keys))``
#   (``map(type)`` and ``is`` call nothing), so hashing/equality are builtin.
# Rule C (instance dicts): ``vars(x)`` is read only for an exact trusted class,
#   must itself be an exact dict with exact-str keys, and no key may shadow a
#   class callable (so every method the scalar loop calls is the library one).
# Rule D (no early raise/warn): admission makes no arithmetic on caller data
#   before bounds are proven and uses getattr-with-default for slots; every
#   doubt returns ``None`` (refusal), which itself runs no caller code.
#
# Reads/calls, with the rule that makes each one inert:
#  hook   type(kinematics) is str and kinematics == "corotational"     A (exact
#         str ==); deleted_set/element_scales are sets/dicts of int()s.
#  params type(tangent|stable_shell_rotation) is bool; type(num_layers) is
#         int; type(tangent_mode) is str then .strip().lower() (exact str);
#         type(excluded) is tuple/set/frozenset then len().               A
#  numpy  np.geterr()["under"] == "ignore" (numpy-owned dict).
#  model  type(model) is FEModel; vars(model) ............................. C
#         ["mesh"], ["materials"] ....................................... B,C
#  mesh   type(mesh) is FEMesh; vars(mesh) .............................. C
#         "element_activity" present and None; ["revisions"], ["elements"],
#         ["nodes"], ["_corotational_cache"] ............................ B,C
#  materials  exact dict, exact-str keys, then "default" in / [] and the
#         spot check's dict.get(materials, name, default) ................. B
#  revisions  exact dict, exact-str keys AND exact-int values swept before
#         the "topology"/"geometry" lookups (revision_signature int()s them) B
#  _corotational_cache  exact tuple (exact tuple of 2 exact ints, exact dict
#         with exact-int keys); signature compared as ints ................ A,B
#  elements/nodes  exact dict, or exact _QualifiedStateMapping whose vars()
#         passes C against that class's callables (so .get/.items are
#         dict's and read the same storage as dict.get/dict.items); keys
#         swept exact int ............................................... A,B,C
#  element  type is BeamElement; vars() ................................... C
#         formulation_native_total_lagrangian absent or `is False`;
#         _linear_nodal_offsets absent; _fiber_plasticity / generalized_section
#         present and None; _geometric_nonlinearity exact str not
#         corotational; material_name exact non-empty str; node_ids exact
#         list of 2 exact ints; _nl_cache exact dict with exact-str keys .. A,B
#  _nl_cache  "T"/"K_noax" exact float64 C-contiguous aligned (12,12)
#         ndarrays; "L"/"EA" exactly float or np.float64 ................. A,B
#  reference  exact _CorotationalReference (slots, no __dict__); category and
#         coordinates read with getattr(..., None) (an unset slot cannot
#         raise); coordinates exact float64 (2,3) ndarray ................ A,D
#  node   type is Node; vars() ................................................ C
#         dofs exact list of exact ints, each 0 <= dof < displacements.size
#         checked before np.intp conversion ............................. A,D
#  committed_states  exact dict with exact-int keys and every beam id absent
#         or None; or exact NonlinearStateStore whose vars() passes C, with
#         exact-dict exact-int-key _element_to_batch (no beam id) and
#         _fallback_committed (beam id absent or None, so Mapping.get ->
#         __getitem__ -> deepcopy(None) or KeyError: no caller code) .... A,B,C
#  displacements  exact float64 1-D ndarray.
#  numerics  bounds (|U|,|X|,|T|,|K_noax|,|EA| < 1e100, 1e-50 < L < 1e50)
#         proven before L**2; stacked kernel and rotation under
#         errstate(all="ignore"); any non-finite final row is routed. With
#         those bounds every floating-point exception of the scalar arithmetic
#         propagates to a non-finite output (no division by an unbounded
#         denominator, overflow only in products feeding the output), so a
#         batched row is one on which the scalar raises no FP flag; the spot
#         check additionally runs the oracle with divide/over/invalid=raise and
#         refuses on any exception.                                      D
#  take() dict.pop on the exact-int-keyed responses dict; trusted counters.
#
# Staleness: with every object above proven exact and free of instance
# overrides, the scalar loop between prepare() and the last take() executes
# only library code (dict-backed .items/.get, BeamElement.get_dof_mapping,
# exact-ndarray indexing, the admitted committed-state lookup, take(), np.add.at)
# none of which writes a batched input, so the precomputed responses cannot
# go stale and no caller code can observe a different call order or count.
# --------------------------------------------------------------------------

from __future__ import annotations

import inspect
import os
import threading
import types
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import corotational as _cr
from . import corotational_performance as _cp
from .elements import BeamElement
from .fe_core import FEMesh, FEModel, Node, _QualifiedStateMapping
from .jit_compiler import JIT_ENABLED, njit
from .nonlinear_state import NonlinearStateStore

_SMALL = 1.0e-12
_BOUND = 1.0e100
_EYE3 = np.eye(3)
_ATOL = 64.0 * np.finfo(float).eps
_COROTATIONAL_LOCAL = frozenset({"corotational", "co_rotational", "corot"})
_ENV = "ANYFESOLVER_COROTATIONAL_BEAM_BATCH"

_LOCK = threading.Lock()
_STATUS: Dict[str, Any] = {
    "enabled": os.environ.get(_ENV, "1").strip().lower() not in {"0", "false", "off", "no"},
    "probe": None,  # None: not run, True: passed, False: failed
    "disabled_reason": None,
    "batched_evaluations": 0,
    "batched_elements": 0,
    "routed_elements": 0,
    "refused_evaluations": 0,
    "spot_check_failures": 0,
}


def _class_callables(cls: type) -> frozenset:
    names = set()
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if isinstance(
                value,
                (types.FunctionType, property, staticmethod, classmethod),
            ) or hasattr(value, "__get__"):
                names.add(name)
    return frozenset(names)


_BEAM_CALLABLES = _class_callables(BeamElement)
_MODEL_CALLABLES = _class_callables(FEModel)
_MESH_CALLABLES = _class_callables(FEMesh)
_NODE_CALLABLES = _class_callables(Node)
_QUALIFIED_MAPPING_CALLABLES = _class_callables(_QualifiedStateMapping)
_STORE_CALLABLES = _class_callables(NonlinearStateStore)

# Library functions whose replacement at run time must disable the batch, so a
# patched (spied, wrapped or altered) scalar route is always honoured.
_PINNED = (
    (_cr, "corotational_element_response"),
    (_cr, "_corotational_deformation_state"),
    (_cr, "_beam_rigid_rotation"),
    (_cr, "_minimal_rotation"),
    (_cr, "rotation_matrix_from_vector"),
    (_cr, "rotation_vector_from_matrix"),
    (_cr, "_stable_shell_rotation_difference"),
    (_cr, "_centered_nodal_vectors"),
    (_cr, "_corotational_cache"),
    (_cr, "_skew"),
    (_cp, "rotate_corotational_force_tangent_blocks"),
    (_cp, "rotate_corotational_force_blocks"),
    (_cp, "_rotate_force_tangent_blocks_jit"),
    (_cp, "_rotate_force_blocks_jit"),
    (_cp, "_validated_inputs"),
    (BeamElement, "compute_nonlinear_response"),
    (BeamElement, "_refuse_offset_route"),
    (BeamElement, "_fiber_plasticity_config"),
    (BeamElement, "num_nodes"),
    (BeamElement, "get_dof_mapping"),
    (BeamElement, "get_node_coordinates"),
    (FEModel, "get_material"),
    (FEMesh, "revision_signature"),
    (FEMesh, "get_node"),
)


def _pinned_objects() -> Tuple[Any, ...]:
    return tuple(
        inspect.getattr_static(owner, name, None)
        if isinstance(owner, type)
        else getattr(owner, name, None)
        for owner, name in _PINNED
    )


_PINNED_ORIGINALS = _pinned_objects()
_PINNED_MODULES_OK = all(
    getattr(getattr(obj, "fget", obj), "__module__", "").startswith("anyfesolver.")
    for obj in _PINNED_ORIGINALS
)


def _library_intact() -> bool:
    if not _PINNED_MODULES_OK:
        return False
    current = _pinned_objects()
    return all(a is b for a, b in zip(current, _PINNED_ORIGINALS))


def set_corotational_beam_batch_enabled(enabled: bool) -> None:
    """Enable or disable the stacked beam route for this process."""

    with _LOCK:
        _STATUS["enabled"] = bool(enabled)
        if enabled:
            _STATUS["disabled_reason"] = None


def reset_corotational_beam_batch_probe() -> None:
    """Forget the probe outcome (testing hook); it reruns on next use."""

    with _LOCK:
        _STATUS["probe"] = None


def corotational_beam_batch_status() -> Dict[str, Any]:
    with _LOCK:
        return dict(_STATUS)


# --------------------------------------------------------------------------
# Stacked kernels.  Each function mirrors one scalar function in
# corotational.py; comments name the scalar expression it reproduces.
# --------------------------------------------------------------------------


def _mv(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Stacked ``matrix @ vector`` (per-item gemv, layout preserved)."""
    return np.matmul(matrix, vector[..., None])[..., 0]


def _vm(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Stacked ``vector @ matrix``."""
    return np.matmul(vector[..., None, :], matrix)[..., 0, :]


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stacked ``np.dot(a, b)`` / ``a @ b`` for 3-vectors."""
    return np.matmul(a[..., None, :], b[..., :, None])[..., 0, 0]


def _norm(a: np.ndarray) -> np.ndarray:
    """Stacked ``np.linalg.norm`` of 1-D vectors: ``sqrt(dot(a, a))``."""
    return np.sqrt(_dot(a, a))


def _skew(v: np.ndarray) -> np.ndarray:
    K = np.zeros(v.shape[:-1] + (3, 3), dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    K[..., 0, 1] = -z
    K[..., 0, 2] = y
    K[..., 1, 0] = z
    K[..., 1, 2] = -x
    K[..., 2, 0] = -y
    K[..., 2, 1] = x
    return K


def _trace(M: np.ndarray) -> np.ndarray:
    return M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]


def _rotation_matrix(v: np.ndarray) -> np.ndarray:
    """``rotation_matrix_from_vector`` for a stack of vectors."""
    a2 = _dot(v, v)
    angle = np.sqrt(a2)
    K = _skew(v)
    small = angle < 1.0e-2
    c1 = 1.0 - a2 * (1.0 / 6.0 - a2 * (1.0 / 120.0 - a2 / 5040.0))
    c2 = 0.5 - a2 * (1.0 / 24.0 - a2 * (1.0 / 720.0 - a2 / 40320.0))
    R_small = _EYE3 + c1[:, None, None] * K + c2[:, None, None] * np.matmul(K, K)
    axis = v / angle[:, None]
    K_axis = _skew(axis)
    half_sin = np.sin(0.5 * angle)
    R_large = (
        _EYE3
        + np.sin(angle)[:, None, None] * K_axis
        + (2.0 * half_sin * half_sin)[:, None, None] * np.matmul(K_axis, K_axis)
    )
    return np.where(small[:, None, None], R_small, R_large)


def _minimal_rotation(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``_minimal_rotation``; returns (R, antiparallel routing mask)."""
    a_norm = _norm(a)
    b_norm = _norm(b)
    a = np.where((a_norm > _SMALL)[:, None], a / a_norm[:, None], a)
    b = np.where((b_norm > _SMALL)[:, None], b / b_norm[:, None], b)
    cross = np.cross(a, b)
    dot = _dot(a, b)
    antiparallel = ~(dot >= -1.0 + 1.0e-12)
    K = _skew(cross)
    R = _EYE3 + K + np.matmul(K, K) / (1.0 + dot)[:, None, None]
    return R, antiparallel


def _beam_rigid_rotation(X: np.ndarray, deformed: np.ndarray, rotations: np.ndarray):
    """``_beam_rigid_rotation`` for two-node beams; returns (R, route mask)."""
    t_ref = X[:, 1] - X[:, 0]
    t_def = deformed[:, 1] - deformed[:, 0]
    t_ref = t_ref / np.maximum(_norm(t_ref), _SMALL)[:, None]
    t_def = t_def / np.maximum(_norm(t_def), _SMALL)[:, None]
    aligned = []
    node_rotations = []
    route = np.zeros(X.shape[0], dtype=bool)
    for node in range(2):
        R_node = _rotation_matrix(rotations[:, node])
        node_rotations.append(R_node)
        R_tilt, anti = _minimal_rotation(_mv(R_node, t_ref), t_def)
        route |= anti
        aligned.append(np.matmul(R_tilt, R_node))
    R0 = aligned[0]
    M = np.matmul(R0.transpose(0, 2, 1), aligned[1])
    sin_psi = 0.5 * (
        (M[:, 2, 1] - M[:, 1, 2]) * t_ref[:, 0]
        + (M[:, 0, 2] - M[:, 2, 0]) * t_ref[:, 1]
        + (M[:, 1, 0] - M[:, 0, 1]) * t_ref[:, 2]
    )
    cos_psi = 0.5 * (_trace(M) - _dot(_vm(t_ref, M), t_ref))
    twist = np.arctan2(sin_psi, cos_psi)
    mean_twist = (0.0 + twist) / 2.0  # np.mean([0.0, twist])
    R = np.matmul(R0, _rotation_matrix(mean_twist[:, None] * t_ref))
    return R, route, node_rotations


def _rotation_vector(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``rotation_vector_from_matrix``; returns (vector, near-pi route mask)."""
    trace = _trace(R)
    v = 0.5 * np.stack(
        (R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]),
        axis=-1,
    )
    sin_angle = _norm(v)
    cos_angle = np.minimum(np.maximum(0.5 * (trace - 1.0), -1.0), 1.0)
    angle = np.arctan2(sin_angle, cos_angle)
    small = angle < 1.0e-4
    near_pi = angle > np.pi - 1.0e-6
    s2 = sin_angle * sin_angle
    v_small = (1.0 + s2 * (1.0 / 6.0 + 0.075 * s2))[:, None] * v
    v_general = (angle / sin_angle)[:, None] * v
    return np.where(small[:, None], v_small, v_general), near_pi & ~small


def _deformation_state(X: np.ndarray, U: np.ndarray):
    """``_corotational_deformation_state`` (beam branch) for 2-node beams."""
    n = X.shape[0]
    u = U.reshape(n, 2, 6)
    translations = u[:, :, :3]
    rotations = u[:, :, 3:]
    half = 0.5 * (X[:, 1] - X[:, 0])
    reference_arm = np.stack((-half, half), axis=1)
    half_t = 0.5 * (translations[:, 1] - translations[:, 0])
    translation_offset = np.stack((-half_t, half_t), axis=1)
    deformed = reference_arm + translation_offset
    # The scalar code recomputes each nodal exponential from identical
    # inputs; the deterministic stacked results are reused.
    R_rig, route, node_rotations = _beam_rigid_rotation(X, deformed, rotations)
    R_rig_T = R_rig.transpose(0, 2, 1)

    # _stable_shell_rotation_difference(R_rig)
    cosine = 0.5 * (_trace(R_rig) - 1.0)
    finite = np.isfinite(R_rig).all(axis=(1, 2))
    orthogonal = (np.abs(np.matmul(R_rig_T, R_rig) - _EYE3) <= _ATOL).all(axis=(1, 2))
    proper = np.linalg.det(R_rig) > 0.0
    fallback = ~finite | ~(cosine > 0.5) | ~orthogonal | ~proper
    # Per-item layouts mirror the scalar arrays (gemv depends on layout):
    # ``R.T - I`` and ``skew`` are C ordered; the stable result is a ``.T`` view.
    difference_fallback = np.ascontiguousarray(R_rig_T - _EYE3)
    skew = np.ascontiguousarray(0.5 * (R_rig - R_rig_T))
    difference_stable_T = np.ascontiguousarray(
        skew + np.matmul(skew, skew) / (1.0 + cosine)[:, None, None]
    )
    difference_stable = difference_stable_T.transpose(0, 2, 1)

    u_d = np.zeros((n, 2, 6), dtype=float)
    for node in range(2):
        pulled = _mv(R_rig_T, translation_offset[:, node])
        arm = reference_arm[:, node]
        u_d[:, node, :3] = pulled + np.where(
            fallback[:, None],
            _mv(difference_fallback, arm),
            _mv(difference_stable, arm),
        )
        R_node = node_rotations[node]
        theta, near_pi = _rotation_vector(np.matmul(R_rig_T, R_node))
        route |= near_pi
        u_d[:, node, 3:] = theta
    return u_d.reshape(n, 12), R_rig, route


def _local_response(u_d, T, K_noax, L, L_squared, EA, tangent: bool):
    """Plain elastic von Karman branch of ``BeamElement.compute_nonlinear_response``."""
    n = u_d.shape[0]
    T_T = T.transpose(0, 2, 1)
    u_loc = _mv(T, u_d)
    du = u_loc[:, 6] - u_loc[:, 0]
    dv = u_loc[:, 7] - u_loc[:, 1]
    dw = u_loc[:, 8] - u_loc[:, 2]
    two = np.full(n, 2.0)
    # ``dv**2`` on NumPy scalars calls libm pow; an exponent array does too.
    eps = du / L + (np.power(dv, two) + np.power(dw, two)) / (2.0 * L_squared)
    N_force = EA * eps
    d_eps = np.zeros((n, 12), dtype=float)
    d_eps[:, 0] = -1.0 / L
    d_eps[:, 6] = 1.0 / L
    d_eps[:, 1] = -dv / L_squared
    d_eps[:, 7] = dv / L_squared
    d_eps[:, 2] = -dw / L_squared
    d_eps[:, 8] = dw / L_squared
    F_loc = _mv(K_noax, u_loc) + (N_force * L)[:, None] * d_eps
    f_ref = _mv(T_T, F_loc)
    if not tangent:
        return f_ref, None
    K_loc = K_noax + (EA * L)[:, None, None] * (d_eps[:, :, None] * d_eps[:, None, :])
    string = N_force / L
    for a, b in ((1, 7), (2, 8)):
        K_loc[:, a, a] += string
        K_loc[:, b, b] += string
        K_loc[:, a, b] -= string
        K_loc[:, b, a] -= string
    k_ref = np.matmul(np.matmul(T_T, K_loc), T)
    return f_ref, k_ref


@njit(cache=True)
def _rotate_tangent_batch_jit(force, tangent, rotation, force_out, tangent_out):
    for item in range(force.shape[0]):
        f_item, k_item = _cp._rotate_force_tangent_blocks_jit(
            force[item], tangent[item], rotation[item]
        )
        force_out[item] = f_item
        tangent_out[item] = k_item


@njit(cache=True)
def _rotate_force_batch_jit(force, rotation, force_out):
    for item in range(force.shape[0]):
        force_out[item] = _cp._rotate_force_blocks_jit(force[item], rotation[item])


def _rotate(f_ref, k_ref, R_rig):
    f_out = np.empty_like(f_ref)
    if k_ref is None:
        if JIT_ENABLED:
            _rotate_force_batch_jit(f_ref, R_rig, f_out)
        else:
            for item in range(f_ref.shape[0]):
                f_out[item] = _cp._rotate_force_blocks_jit(f_ref[item], R_rig[item])
        return f_out, None
    k_out = np.empty_like(k_ref)
    if JIT_ENABLED:
        _rotate_tangent_batch_jit(f_ref, k_ref, R_rig, f_out, k_out)
    else:
        for item in range(f_ref.shape[0]):
            f_out[item], k_out[item] = _cp._rotate_force_tangent_blocks_jit(
                f_ref[item], k_ref[item], R_rig[item]
            )
    return f_out, k_out


def _kernel(X, U, T, K_noax, L, L_squared, EA, tangent: bool):
    """Return stacked global (f, k) and a per-element scalar-routing mask."""
    with np.errstate(all="ignore"):
        u_d, R_rig, route = _deformation_state(X, U)
        f_ref, k_ref = _local_response(u_d, T, K_noax, L, L_squared, EA, tangent)
        finite = (
            np.isfinite(u_d).all(axis=1)
            & np.isfinite(R_rig).all(axis=(1, 2))
            & np.isfinite(f_ref).all(axis=1)
        )
        if k_ref is not None:
            finite &= np.isfinite(k_ref).all(axis=(1, 2))
    route = route | ~finite
    # Routed rows may hold non-finite garbage; never let it reach a kernel.
    if route.any():
        f_ref = np.where(route[:, None], 0.0, f_ref)
        R_rig = np.where(route[:, None, None], _EYE3, R_rig)
        if k_ref is not None:
            k_ref = np.where(route[:, None, None], 0.0, k_ref)
    with np.errstate(all="ignore"):
        f, k = _rotate(f_ref, k_ref, R_rig)
        # A rotation that overflows stays with the scalar call (which then
        # reports it on its own turn).
        finite = np.isfinite(f).all(axis=1)
        if k is not None:
            finite &= np.isfinite(k).all(axis=(1, 2))
    return f, k, route | ~finite


def _scalar_oracle(reference, element, mesh, material, u_element, tangent: bool):
    """The scalar route minus its diagnostics counters."""
    u_d, R_rig, _deformed = _cr._corotational_deformation_state(
        reference, element, u_element
    )
    f_ref, k_ref, _state = element.compute_nonlinear_response(
        mesh, material, u_d, None, 5, tangent
    )
    f_ref = np.asarray(f_ref, dtype=float).reshape(-1)
    if not tangent:
        return _cp._rotate_force_blocks_jit(f_ref, R_rig), None
    return _cp._rotate_force_tangent_blocks_jit(
        f_ref, np.asarray(k_ref, dtype=float), R_rig
    )


def _bitwise_equal(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(
        np.array_equal(
            np.ascontiguousarray(a).view(np.uint64),
            np.ascontiguousarray(b).view(np.uint64),
        )
    )


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------


def _keys_are(mapping, key_type) -> bool:
    """Every key's exact type is ``key_type``; iterates dict storage only."""
    return all(t is key_type for t in map(type, dict.keys(mapping)))


def _instance_dict(obj, callables: frozenset):
    """``vars(obj)`` for an object whose exact class the caller has proven.

    Admitted only as an exact dict with exact-str keys none of which shadows
    a class callable; otherwise ``None``.  Nothing here runs caller code.
    """
    attributes = vars(obj)
    if type(attributes) is not dict or not _keys_are(attributes, str):
        return None
    if not callables.isdisjoint(attributes):
        return None
    return attributes


def _exact_int_mapping(mapping) -> bool:
    """Exact dict, or exact ``_QualifiedStateMapping`` without overrides."""
    kind = type(mapping)
    if kind is _QualifiedStateMapping:
        if _instance_dict(mapping, _QUALIFIED_MAPPING_CALLABLES) is None:
            return False
    elif kind is not dict:
        return False
    return _keys_are(mapping, int)


def _is_float_scalar(value) -> bool:
    kind = type(value)
    return kind is float or kind is np.float64


def _admit(element, elem_id, nodes, references, size):
    """Return (reference, cache, dof_mapping, material_name) or None.

    Only identity/type checks and exact built-in container reads run here; no
    method of the element, mesh, nodes or model is called, so no caller code
    can run before the assembly loop does.
    """
    if type(element) is not BeamElement or type(elem_id) is not int:
        return None
    attributes = _instance_dict(element, _BEAM_CALLABLES)
    if attributes is None:
        return None
    if attributes.get("formulation_native_total_lagrangian", False) is not False:
        return None
    if "_linear_nodal_offsets" in attributes:
        return None
    if attributes.get("_fiber_plasticity", 0) is not None:
        return None
    if attributes.get("generalized_section", 0) is not None:
        return None
    geometry = attributes.get("_geometric_nonlinearity")
    if type(geometry) is not str or geometry in _COROTATIONAL_LOCAL:
        return None
    name = attributes.get("material_name")
    if type(name) is not str or not name:
        return None
    cache = attributes.get("_nl_cache")
    if type(cache) is not dict or not _keys_are(cache, str):
        return None
    for key in ("T", "K_noax"):
        matrix = dict.get(cache, key)
        if (
            type(matrix) is not np.ndarray
            or matrix.dtype != np.float64
            or matrix.shape != (12, 12)
            or not matrix.flags.c_contiguous
            or not matrix.flags.aligned
        ):
            return None
    if not _is_float_scalar(dict.get(cache, "L")):
        return None
    if not _is_float_scalar(dict.get(cache, "EA")):
        return None
    reference = dict.get(references, elem_id)
    if type(reference) is not _cr._CorotationalReference:
        return None
    category = getattr(reference, "category", None)  # slots: unset -> None
    if type(category) is not str or category != "beam":
        return None
    coordinates = getattr(reference, "coordinates", None)
    if (
        type(coordinates) is not np.ndarray
        or coordinates.dtype != np.float64
        or coordinates.shape != (2, 3)
    ):
        return None
    # Element.get_dof_mapping without calling it: node_ids -> nodes -> dofs.
    node_ids = attributes.get("node_ids")
    if type(node_ids) is not list or len(node_ids) != 2:
        return None
    dof_mapping = []
    for node_id in node_ids:
        if type(node_id) is not int:
            return None
        node = dict.get(nodes, node_id)
        if type(node) is not Node:
            return None
        node_attributes = _instance_dict(node, _NODE_CALLABLES)
        if node_attributes is None:
            return None
        node_dofs = node_attributes.get("dofs")
        if type(node_dofs) is not list or len(node_dofs) != 6:
            return None
        for dof in node_dofs:
            # Exact-int bounds before any conversion: an out-of-range DOF
            # must raise on its scalar turn, not here.
            if type(dof) is not int or dof < 0 or dof >= size:
                return None
        dof_mapping.extend(node_dofs)
    return reference, cache, np.array(dof_mapping, dtype=np.intp), name


def _cached_references(mesh_attributes):
    """The scalar route's reference cache, under the scalar reuse rule.

    ``_corotational_cache`` reuses ``cached[1]`` only when ``cached[0] ==
    (topology, geometry)``.  The comparison is evaluated here on exact ints
    only, so no caller ``__eq__`` can run or disagree with the scalar test.
    """
    revisions = mesh_attributes.get("revisions")
    if type(revisions) is not dict or not _keys_are(revisions, str):
        return None
    # revision_signature() int()s every value: exact ints only.
    if not all(t is int for t in map(type, dict.values(revisions))):
        return None
    topology = dict.get(revisions, "topology", 0)
    geometry = dict.get(revisions, "geometry", 0)
    cached = mesh_attributes.get("_corotational_cache")
    if type(cached) is not tuple or len(cached) != 2:
        return None
    signature, references = cached
    if type(signature) is not tuple or len(signature) != 2:
        return None
    if type(signature[0]) is not int or type(signature[1]) is not int:
        return None
    if signature[0] != topology or signature[1] != geometry:
        return None
    if type(references) is not dict or not _keys_are(references, int):
        return None
    return references


def _states_admitted(committed_states, beam_ids) -> bool:
    """The scalar ``committed_states.get(beam_id)`` runs no caller code.

    Admitted: an exact dict with exact-int keys whose beam entries are absent
    or None, or an exact NonlinearStateStore (no instance overrides) whose
    beam ids are in no shell batch and whose fallback entry is absent or
    None, so ``Mapping.get`` -> ``__getitem__`` ends in ``KeyError`` or
    ``deepcopy(None)``.
    """
    kind = type(committed_states)
    if kind is dict:
        if not _keys_are(committed_states, int):
            return False
        return all(dict.get(committed_states, i) is None for i in beam_ids)
    if kind is not NonlinearStateStore:
        return False
    attributes = _instance_dict(committed_states, _STORE_CALLABLES)
    if attributes is None:
        return False
    batched = attributes.get("_element_to_batch")
    fallback = attributes.get("_fallback_committed")
    if type(batched) is not dict or type(fallback) is not dict:
        return False
    if not _keys_are(batched, int) or not _keys_are(fallback, int):
        return False
    for i in beam_ids:
        if i in batched or dict.get(fallback, i) is not None:
            return False
    return True


def _evaluation_admitted(model, displacements):
    """Whole-evaluation admission by identity/type checks only."""
    if type(model) is not FEModel:
        return None
    model_attributes = _instance_dict(model, _MODEL_CALLABLES)
    if model_attributes is None:
        return None
    mesh = model_attributes.get("mesh")
    if type(mesh) is not FEMesh:
        return None
    mesh_attributes = _instance_dict(mesh, _MESH_CALLABLES)
    if mesh_attributes is None:
        return None
    if mesh_attributes.get("element_activity", 0) is not None:
        return None
    materials = model_attributes.get("materials")
    if type(materials) is not dict or not _keys_are(materials, str):
        return None
    if "default" not in materials:
        return None
    elements = mesh_attributes.get("elements")
    nodes = mesh_attributes.get("nodes")
    if not _exact_int_mapping(elements) or not _exact_int_mapping(nodes):
        return None
    if (
        type(displacements) is not np.ndarray
        or displacements.dtype != np.float64
        or displacements.ndim != 1
    ):
        return None
    return mesh, mesh_attributes, materials, elements, nodes


class CorotationalBeamBatch:
    """Precomputed responses consumed element by element by the assembler."""

    __slots__ = ("_responses", "_tangent")

    def __init__(self, responses: Dict[Any, Tuple[np.ndarray, Optional[np.ndarray]]], tangent: bool):
        self._responses = responses
        self._tangent = tangent

    def __len__(self) -> int:
        return len(self._responses)

    def take(self, elem_id: Any):
        """Return (f, k) for a batched element, recording the scalar counters.

        Call it where the scalar route would run, i.e. after the committed
        state lookup, so counters advance in the scalar order.
        """
        response = self._responses.pop(elem_id, None)
        if response is None:
            return None
        # Identical bookkeeping to rotate_corotational_force(_tangent)_blocks.
        if self._tangent:
            with _cp._STATUS_LOCK:
                _cp._STATUS["force_block_rotations"] += 1
                _cp._STATUS["tangent_block_rotations"] += 1
            _cp.record_corotational_analysis_execution(force_blocks=1, tangent_blocks=1)
        else:
            with _cp._STATUS_LOCK:
                _cp._STATUS["force_block_rotations"] += 1
            _cp.record_corotational_analysis_execution(force_blocks=1)
        return response


_DEFAULT_UNDERFLOW = "ignore"


def _ready() -> bool:
    with _LOCK:
        enabled = bool(_STATUS["enabled"])
        probe = _STATUS["probe"]
    if not enabled:
        return False
    if probe is None:
        passed, reason = _run_probe()
        with _LOCK:
            _STATUS["probe"] = passed
            if not passed:
                _STATUS["enabled"] = False
                _STATUS["disabled_reason"] = reason
        return passed
    return bool(probe)


def _refuse() -> None:
    with _LOCK:
        _STATUS["refused_evaluations"] += 1


def prepare_corotational_beam_batch(
    model: Any,
    displacements: Any,
    *,
    committed_states: Any,
    tangent: Any,
    tangent_mode: Any,
    num_layers: Any,
    stable_shell_rotation: Any,
    excluded: Any = (),
) -> Optional[CorotationalBeamBatch]:
    """Evaluate a pure plain-beam evaluation at once, or return ``None``.

    Admission is all-or-nothing and uses identity/type checks only: every
    element must be an admitted beam, and no instance attribute of the model,
    mesh, nodes or elements may shadow a class callable.  Hence no caller code
    runs here and nothing the scalar loop runs can change a batched input.
    Elements whose stacked arithmetic is not exact are left to the scalar
    call.  No state, cache or counter is written here.
    """

    if type(tangent) is not bool or type(stable_shell_rotation) is not bool:
        return None
    if type(num_layers) is not int:
        return None
    if type(tangent_mode) is not str or tangent_mode.strip().lower() != "rotated":
        return None
    kind = type(excluded)
    if kind is not tuple and kind is not set and kind is not frozenset:
        return None
    if len(excluded):
        return None  # deleted or scaled elements: not a pure evaluation
    if np.geterr().get("under") != _DEFAULT_UNDERFLOW:
        return None
    # Toggle and library checks first: with the batch off, no caller object
    # is even read.
    if not _library_intact() or not _ready():
        return None
    admitted_evaluation = _evaluation_admitted(model, displacements)
    if admitted_evaluation is None:
        return None
    mesh, mesh_attributes, materials, elements, nodes = admitted_evaluation
    references = _cached_references(mesh_attributes)
    if references is None:
        return None
    size = displacements.size
    ids = []
    element_list = []
    refs = []
    caches = []
    dofs = []
    names = []
    for elem_id, element in dict.items(elements):
        admitted = _admit(element, elem_id, nodes, references, size)
        if admitted is None:
            return None  # mixed or unadmitted model: whole evaluation scalar
        ids.append(elem_id)
        element_list.append(element)
        refs.append(admitted[0])
        caches.append(admitted[1])
        dofs.append(admitted[2])
        names.append(admitted[3])
    if len(ids) < 2:
        return None
    if not _states_admitted(committed_states, ids):
        return None
    try:
        with np.errstate(all="ignore"):
            U = displacements[np.stack(dofs)]
            X = np.stack([getattr(reference, "coordinates") for reference in refs])
            T = np.stack([dict.__getitem__(cache, "T") for cache in caches])
            K_noax = np.stack([dict.__getitem__(cache, "K_noax") for cache in caches])
            L_values = [dict.__getitem__(cache, "L") for cache in caches]
            L = np.array(L_values, dtype=float)
            EA = np.array([dict.__getitem__(cache, "EA") for cache in caches], dtype=float)
            # Bounded finite data keeps every scalar floating-point exception
            # visible as a non-finite (routed) output; anything else stays
            # scalar.  Bounds come before any arithmetic on the data.
            bounded = all(
                bool(np.all(np.abs(array) < _BOUND)) for array in (U, X, T, K_noax, EA)
            ) and bool(np.all(L > 1.0e-50)) and bool(np.all(L < 1.0e50))
        if not bounded:
            return None
        # 1e-100 < L**2 < 1e100: the scalar ``L**2`` cannot flag either.
        L_squared = np.array([value**2 for value in L_values], dtype=float)
        f, k, route = _kernel(X, U, T, K_noax, L, L_squared, EA, tangent)
    except Exception:  # the batch's own numerics only: refuse, scalar recomputes
        _refuse()
        return None
    batched = np.flatnonzero(~route)
    if batched.size == 0:
        return None
    # Spot check the first and last batched element against the oracle.  The
    # material is resolved as FEModel.get_material would, by exact dict reads.
    # A batched row must be one on which the scalar raises no floating-point
    # flag; the oracle runs with those flags raising and any exception refuses.
    default = dict.__getitem__(materials, "default")
    for position in (int(batched[0]), int(batched[-1])):
        material = dict.get(materials, names[position], default)
        try:
            with np.errstate(divide="raise", over="raise", invalid="raise"):
                f_scalar, k_scalar = _scalar_oracle(
                    refs[position], element_list[position], mesh, material,
                    displacements[dofs[position]], tangent,
                )
        except Exception:
            _refuse()
            return None
        if not (
            _bitwise_equal(f[position], f_scalar)
            and _bitwise_equal(None if k is None else k[position], k_scalar)
        ):
            with _LOCK:
                _STATUS["spot_check_failures"] += 1
                _STATUS["enabled"] = False
                _STATUS["disabled_reason"] = "spot_check_mismatch"
            return None
    responses = {
        ids[position]: (f[position], None if k is None else k[position])
        for position in batched.tolist()
    }
    with _LOCK:
        _STATUS["batched_evaluations"] += 1
        _STATUS["batched_elements"] += len(responses)
        _STATUS["routed_elements"] += int(route.sum())
    return CorotationalBeamBatch(responses, tangent)


# --------------------------------------------------------------------------
# Once-per-process bitwise probe
# --------------------------------------------------------------------------


def _probe_states(rng: np.random.Generator):
    """Element displacement states covering every scalar branch."""
    states = []
    states.append(np.zeros(12))  # zero rotation, zero translation
    for scale in (1.0e-6, 1.0e-3, 5.0e-3, 2.0e-2, 0.3, 1.5, 3.0):
        states.append(rng.standard_normal(12) * scale)
    for angle in (0.5, 1.2, 2.5, 3.1, np.pi - 1.0e-7):  # large rigid rotations
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        theta = angle * axis
        u = np.zeros(12)
        u[3:6] = theta + rng.standard_normal(3) * 1.0e-5
        u[9:12] = theta + rng.standard_normal(3) * 1.0e-5
        states.append(u)
    # Small deformational rotation below / above the 1e-4 log-map switch.
    for spread in (1.0e-6, 3.0e-4):
        u = rng.standard_normal(12) * 1.0e-3
        u[9:12] = u[3:6] + spread
        states.append(u)
    # Antiparallel minimal rotation: nodal rotation of pi about a normal axis.
    u = np.zeros(12)
    u[3:6] = np.array([0.0, np.pi, 0.0])
    states.append(u)
    # Opposite end twists of nearly pi (large mean-twist correction).
    u = np.zeros(12)
    u[9:12] = np.array([0.0, 0.0, np.pi - 1.0e-8])
    states.append(u)
    return states


def _run_probe() -> Tuple[bool, Optional[str]]:
    try:
        rng = np.random.default_rng(20261124)
        model = FEModel("corotational_beam_batch_probe")
        model.add_material("probe", 2.1e11, 0.3)
        section = {"area": 0.02, "Iy": 1.6667e-5, "Iz": 6.6667e-5, "J": 2.29e-5}
        endpoints = [
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), None),
            ((0.3, -0.2, 0.1), (0.4, 1.7, -0.6), None),
            ((1.0, 2.0, 3.0), (1.0, 2.0, 5.5), (1.0, 0.0, 0.0)),
            ((-2.0, 0.5, 0.0), (-0.7, -0.4, 0.9), (0.2, 0.3, 0.9)),
        ]
        node = 0
        elements = []
        for index, (start, end, orientation) in enumerate(endpoints):
            model.add_node(node + 1, *start)
            model.add_node(node + 2, *end)
            cross_section = dict(section)
            if orientation is not None:
                cross_section["orientation"] = orientation
            element = BeamElement(index + 1, [node + 1, node + 2], "probe", cross_section)
            model.add_element(index + 1, element)
            elements.append(element)
            node += 2
        mesh = model.mesh
        material = model.get_material("probe")
        references = _cr._corotational_cache(model)
        for element in elements:  # build the local caches through the scalar code
            element.compute_nonlinear_response(mesh, material, np.zeros(12), None, 5, True)
        states = _probe_states(rng)
        rows = [(e, s) for e in elements for s in states]
        X = np.stack([references[e.element_id].coordinates for e, _ in rows])
        U = np.stack([s for _, s in rows])
        T = np.stack([e._nl_cache["T"] for e, _ in rows])
        K_noax = np.stack([e._nl_cache["K_noax"] for e, _ in rows])
        L = np.array([e._nl_cache["L"] for e, _ in rows])
        L_squared = np.array([e._nl_cache["L"] ** 2 for e, _ in rows])
        EA = np.array([e._nl_cache["EA"] for e, _ in rows], dtype=float)
        for tangent in (True, False):
            f, k, route = _kernel(X, U, T, K_noax, L, L_squared, EA, tangent)
            if route.all() or not route.any():
                return False, "probe_routing_coverage"
            for position, (element, state) in enumerate(rows):
                if route[position]:
                    continue
                f_scalar, k_scalar = _scalar_oracle(
                    references[element.element_id], element, mesh, material, state, tangent
                )
                if not (
                    _bitwise_equal(f[position], f_scalar)
                    and _bitwise_equal(None if k is None else k[position], k_scalar)
                ):
                    return False, "probe_bitwise_mismatch"
        # Logarithm branches, including near-pi rows that must be routed.
        vectors = []
        for angle in (0.0, 1.0e-7, 5.0e-5, 2.0e-4, 0.3, 2.0, 3.0, np.pi - 1.0e-9):
            axis = rng.standard_normal(3)
            vectors.append(angle * axis / np.linalg.norm(axis))
        matrices = np.stack([_cr.rotation_matrix_from_vector(v) for v in vectors])
        with np.errstate(all="ignore"):
            theta, near_pi = _rotation_vector(matrices)
        if not near_pi[-1] or near_pi[:-1].any():
            return False, "probe_log_routing"
        for position in range(len(vectors) - 1):
            if not _bitwise_equal(
                theta[position], _cr.rotation_vector_from_matrix(matrices[position])
            ):
                return False, "probe_log_mismatch"
        return True, None
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"probe_error:{type(exc).__name__}"
