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
from .fe_core import FEMesh, FEModel
from .jit_compiler import JIT_ENABLED, njit

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
    (FEModel, "get_material"),
    (FEMesh, "revision_signature"),
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
    f, k = _rotate(f_ref, k_ref, R_rig)
    return f, k, route


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


def _admit(element, mesh, references, elem_id):
    """Return (reference, cache, dof_mapping) or None for the scalar route."""
    if type(element) is not BeamElement:
        return None
    attributes = vars(element)
    if not _BEAM_CALLABLES.isdisjoint(attributes):
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
    if type(attributes.get("material_name")) is not str:
        return None
    cache = attributes.get("_nl_cache")
    if type(cache) is not dict:
        return None
    for key in ("T", "K_noax"):
        matrix = cache.get(key)
        if (
            type(matrix) is not np.ndarray
            or matrix.dtype != np.float64
            or matrix.shape != (12, 12)
            or not matrix.flags.c_contiguous
            or not matrix.flags.aligned
        ):
            return None
    if type(cache.get("L")) not in (float, np.float64):
        return None
    if type(cache.get("EA")) not in (float, np.float64):
        return None
    reference = references.get(int(elem_id))
    if (
        reference is None
        or type(reference) is not _cr._CorotationalReference
        or reference.category != "beam"
    ):
        return None
    coordinates = reference.coordinates
    if (
        type(coordinates) is not np.ndarray
        or coordinates.dtype != np.float64
        or coordinates.shape != (2, 3)
    ):
        return None
    dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
    if dof_mapping.shape != (12,):
        return None
    return reference, cache, dof_mapping


def _cached_references(model):
    """The scalar route's reference cache, only if it already exists."""
    mesh = model.mesh
    revisions = mesh.revision_signature()
    signature = (
        int(revisions.get("topology", 0)),
        int(revisions.get("geometry", 0)),
    )
    cached = getattr(mesh, "_corotational_cache", None)
    if (
        type(cached) is not tuple
        or len(cached) != 2
        or cached[0] != signature
        or type(cached[1]) is not dict
    ):
        return None
    return cached[1]


class CorotationalBeamBatch:
    """Precomputed responses consumed element by element by the assembler."""

    __slots__ = ("_responses", "_tangent")

    def __init__(self, responses: Dict[Any, Tuple[np.ndarray, Optional[np.ndarray]]], tangent: bool):
        self._responses = responses
        self._tangent = tangent

    def __len__(self) -> int:
        return len(self._responses)

    def take(self, elem_id: Any):
        """Return (f, k) for a batched element, recording the scalar counters."""
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
    tangent: Any,
    tangent_mode: Any,
    num_layers: Any,
    stable_shell_rotation: Any,
    excluded: Any = (),
) -> Optional[CorotationalBeamBatch]:
    """Evaluate all admitted beams at once, or return ``None`` (scalar route).

    The returned object holds only elements whose stacked response equals the
    scalar response; every other element keeps the scalar call.  No state,
    cache or counter is written here.
    """

    try:
        if type(tangent) is not bool or type(stable_shell_rotation) is not bool:
            return None
        if type(num_layers) is not int:
            return None
        if type(tangent_mode) is not str or tangent_mode.strip().lower() != "rotated":
            return None
        if type(model) is not FEModel or type(model.mesh) is not FEMesh:
            return None
        if type(model.materials) is not dict:
            return None
        if model.mesh.element_activity is not None:
            # revision_signature() reads the activity object; keep its
            # per-element call pattern on the scalar route.
            return None
        if np.geterr().get("under") != _DEFAULT_UNDERFLOW:
            return None
        if (
            type(displacements) is not np.ndarray
            or displacements.dtype != np.float64
            or displacements.ndim != 1
        ):
            return None
        if not _library_intact() or not _ready():
            return None
        mesh = model.mesh
        references = _cached_references(model)
        if references is None:
            return None
        excluded = set(excluded)
        ids = []
        elements = []
        refs = []
        caches = []
        dofs = []
        for elem_id, element in mesh.elements.items():
            if elem_id in excluded:
                continue
            admitted = _admit(element, mesh, references, elem_id)
            if admitted is None:
                continue
            # The scalar route resolves the material per element; keep its
            # lookup (and any KeyError) on the same inputs.
            model.get_material(element.material_name)
            ids.append(elem_id)
            elements.append(element)
            refs.append(admitted[0])
            caches.append(admitted[1])
            dofs.append(admitted[2])
        if len(ids) < 2:
            return None
        dof_matrix = np.stack(dofs)
        if int(dof_matrix.min()) < 0 or int(dof_matrix.max()) >= displacements.size:
            return None
        U = displacements[dof_matrix]
        X = np.stack([reference.coordinates for reference in refs])
        T = np.stack([cache["T"] for cache in caches])
        K_noax = np.stack([cache["K_noax"] for cache in caches])
        L_values = [cache["L"] for cache in caches]
        L = np.array(L_values, dtype=float)
        L_squared = np.array([value**2 for value in L_values], dtype=float)
        EA = np.array([cache["EA"] for cache in caches], dtype=float)
        # Bounded finite data keeps every scalar intermediate free of
        # floating-point exceptions; anything else stays scalar.
        for array in (U, X, T, K_noax, EA):
            if not np.all(np.abs(array) < _BOUND):
                return None
        if not (np.all(L > 1.0e-50) and np.all(L < 1.0e50)):
            return None
        f, k, route = _kernel(X, U, T, K_noax, L, L_squared, EA, tangent)
        batched = np.flatnonzero(~route)
        if batched.size == 0:
            return None
        # Spot check the first and last batched element against the oracle.
        for position in (int(batched[0]), int(batched[-1])):
            material = model.get_material(elements[position].material_name)
            f_scalar, k_scalar = _scalar_oracle(
                refs[position], elements[position], mesh, material,
                displacements[dofs[position]], tangent,
            )
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
    except Exception:  # fail closed: the scalar loop reproduces any error
        _refuse()
        return None


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
