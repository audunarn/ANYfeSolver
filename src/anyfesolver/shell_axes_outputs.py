"""Opt-in shell result-axes outputs, derived outside the pinned elements (D2-1, B2/B3).

``recover_shell_axes(element, mesh, displacements, material)`` (route B3,
owner 2026-11-26) runs the recovery itself through the trusted class method
``type(element).compute_stresses(element, mesh, displacements, material,
return_global=True)`` and returns that recovery (the dict, exactly as the
element produced it) together with the axes outputs derived from it, per
recovery station:

* the physical station positions (shape-function interpolation of the nodes at
  exactly the stations the element used, in the element's own order),
* the tensor-basis normal of each stored family (membrane, bending, shear),
* the director polarity metadata where the branch has a director,
* the three global family tensors (membrane N, bending M, transverse shear Q);
  where the element emits none (legacy homogeneous and e4-pl warped
  homogeneous) they are derived from the native local values,
  ``N = h sigma_m``, ``M = sigma_b max(h^2, 1e-12) / 6`` (the inverse of the
  legacy recovery floor ``6 M / max(h^2, 1e-12)``), ``Q = h tau``, each rotated
  once with the frame that family is in; where the element emits them they are
  passed through unchanged (bit-identical read-only copies).

The frames are RECOMPUTED here through the existing read-only helpers (that is
what is validated) and every call cross-checks them against quantities the
solver itself emitted (design D2 section 1, rules (1)-(5), revision 4).  Rule
(6) (B3) certifies every basis normal geometrically, independently of the
tensors (which cannot fix the sign of a normal): legacy and e4-pl normals
against the normalized cross product of the shape-function tangents at the
station (the element centre for planar e4-pl and the warped centre frame),
oriented by the physical reference director where the element has one; S3
and V2D normals against the normalized owner ``reference_normal``; the V2D
frame also against the frame the recovery emitted.  A cross-check failure
never raises: the family is refused with ``AXES_FRAME_RULE_MISMATCH`` and its
derived array is omitted; a frame that a refused family shows wrong feeds no
returned value.

Contract (owner 2026-11-26, option 1).  Trust boundary A'' with one
narrowing for this opt-in module: customising the Python object model INSIDE
solver-owned values (instance overrides in a section, mapping, material or
activity object; custom mapping or key behaviour; metaclass tricks) is out of
contract.  As defence in depth the module still admits, by type identity and
before any other access, only exact element / FEMesh / Node objects with an
exact ``dict`` instance dict, ``str`` keys, no instance attribute naming a
class attribute (declared dataclass fields excepted) and plain values (exact
scalars, containers, numeric ndarrays, or exact solver-owned classes, which
are not looked into); custom subclasses of these objects are refused.  Three
solver-owned caches the solve and the ordinary stress recovery leave behind are admitted
only under their own key on their owner type (2026-11-27): the batch plan
``FEMesh._recovery_batch_plan`` (exact ``RecoveryBatchPlan``, not looked into;
the axes recovery never reads it), the qualified S3 stiffness plan
``FEMesh._qualified_s3_reference_stiffness_plan`` (exact
``PreparedReferenceS3Components``, not looked into, never read here; owner
2026-11-27) and the V2D capture subscriptions
``_qualified_v2d_lease_epochs`` on a V2D element (an exact ``list`` of exact
callback-free ``weakref.ref``, never dereferenced).
``admit_shell_axes(element, mesh)`` runs this gate alone; it is the first step
of ``recover_shell_axes``.  The
scalar attributes both the recovery and this module read must be exact types
(``float``/``int``/``numpy.float64``; polarity ``int`` +-1; reference normal a
float64 ndarray of shape (3,) or None).

Geometry domain: every element must have geometry conditioning ``G = (max|x|
/ shortest tangent, diagonal or edge + 1) |a||b| / |a x b| <= 2.5e5`` (see
``GEOMETRY_CONDITIONING_LIMIT``); outside it the module raises
``AXES_GEOMETRY_CONDITIONING``.  Inside it no reversed basis normal and no
normal tilt above 1e-9 can certify.  Models far from their origin relative to
the element size (e.g. georeferenced coordinates) must be translated first.

Frames per branch (``station`` = per recovery point, ``centre`` = element
centre; families m = membrane, b = bending, q = shear):

* legacy ``ShellElement`` 3/4/6/8 nodes, homogeneous and generalized, and e4-pl
  Q4 warped without ``reference_normal`` (it falls back to the legacy
  recovery): m, b, q all in the numbered station frame
  ``_local_frame_and_derivatives`` (or ``native_recovery_frames`` exactly where
  the legacy homogeneous recovery uses it).  For 3/4 nodes the shear values are
  centre-frame MITC values but the element places them in the station frame,
  so the station frame is the frame of the stored shear tensor.
* e4-pl Q4 planar (warpage <= ``planar_tolerance``): one constant frame,
  ``_physical_director_context(equation7_frame(nodes)[0])``, for m, b, q.
* e4-pl Q4 warped with ``reference_normal``: m, b in the physical station frame
  ``_physical_director_context(_local_frame_and_derivatives(...))``; q in the
  physical CENTRE frame ``_physical_director_context(_center_frame(nodes))``.
  The surface tensors carry that centre shear projected into each station's
  tangent plane, so the surface-resolved shear is compared after the same
  projection (rule (1)); the centre frame is checked through q only (rule (4)).
* qualified S3: one constant owner frame ``triangle_frame(nodes,
  reference_normal)`` (normal = owner ``reference_normal``, not multiplied by
  the polarity); the emitted global M and Q are multiplied by the polarity
  (owner-normal convention), local M and Q are in director convention.
* S3 V2D: one constant frame recomputed here from the facet (x along the first
  internal edge, normal = owner ``reference_normal``); global M, Q are in the
  director convention (not re-multiplied).  Owner decision 2026-11-26 (a): V2D
  basis normals follow the OWNER normal, i.e. the frame the stresses are
  actually in; rule (6) certifies that frame (owner normal and emitted frame)
  within the solver's own admission ``|e . n| <= 1e-10 sqrt(edge_scale)``, so
  a V2D normal may differ from the facet normal by that admitted tilt.

Surface-only branches (legacy homogeneous, e4-pl warped homogeneous) resolve
each family from the solver's global surface tensors rotated into the
recomputed station frame; every other branch is a direct-reference branch whose
own global family tensors are the reference.
"""

from __future__ import annotations

import functools
import gc
import math
import types
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "AXES_FRAME_RULE_MISMATCH",
    "FAMILIES",
    "Refusal",
    "ShellAxesFrameRuleMismatch",
    "ShellAxesOutputs",
    "ShellAxesRecovery",
    "ShellAxesRefusal",
    "admit_shell_axes",
    "recover_shell_axes",
]

AXES_FRAME_RULE_MISMATCH = "AXES_FRAME_RULE_MISMATCH"
FAMILIES: Tuple[str, str, str] = ("membrane", "bending", "shear")
REFUSAL_REASONS = frozenset(
    {"residual", "nonfinite", "no_reference", "unresolvable_family", "out_of_plane", "normal_geometry"}
)
# reasons that are evidence against the frame a family uses
_FRAME_EVIDENCE = ("residual", "out_of_plane", "nonfinite", "normal_geometry")

EPS = 2.0**-52
RELATIVE_BOUND = 1.0e-13  # rule (1) relative allowance and the tangency bound
ROUNDOFF = 64.0 * EPS  # rule (1) allowance of the emitted m +- b surfaces
RESOLUTION = 1.0e-9  # rule (3): a_f <= 1e-9 s_f
LEGACY_RECOVERY_FLOOR = 1.0e-12  # elements._SMALL in 6 M / max(h^2, _SMALL)
ORTHONORMAL_BOUND = 1.0e-13  # |G^T G - I| of every frame used (and det G > 0); V2D: see _v2d_frame
GEOMETRY_BOUND = 1.0e-13  # rule (6): |n - n_geometry| (measured <= 2.3e-16 on every fixture) ...
# ... plus the roundoff of coordinate differences: 64 eps max|x| / (shortest
# tangent, diagonal or edge used), so a large origin is not a spurious refusal
# Measured |n - n_geometry| <= 0.335 eps G (300 random elements, origins to
# 1e7, slivers to 1e-4) and <= 0.008 eps G on ANYfem barge5: 8 eps is 24x that.
GEOMETRY_ROUNDOFF = 8.0 * EPS
# Declared geometry domain (owner 2026-11-26): G <= 2.5e5 keeps the rule (6)
# bound on the EUCLIDEAN normal difference <= 1e-13 + 8 eps 2.5e5 = 4.5e-10,
# half the 1e-9 resolution, so inside the domain no reversed normal and no
# normal tilt above 1e-9 certifies (Codex r7: the first limit 5e5 left no
# margin and the check was per component); outside it the module refuses
# (AXES_GEOMETRY_CONDITIONING).  Measured: barge5 G <= 5.0e3 (median 5.4e2),
# solver test fixtures G <= 5.4.
# The cutoff is evaluated in double precision.  The computed G has relative
# error <= 8 eps G (the same roundoff model as the rule (6) bound; verified in
# exact arithmetic for triangle edges, <= 0.03 eps G on the Codex r9/r10
# cases; for shape-function tangents it is the measured rule (6) roundoff,
# <= 0.335 eps G over 300 random quadrilaterals), i.e. <= 5e-10 up to 2.8e5,
# so near the cutoff an element whose exact G lies within 1e-9 of 2.5e5 may
# fall on either side.  That decides only which side of the stated
# boundary it lands on: the tilt guarantee keeps its 2x margin for every G up
# to 2.8e5 (bound 1e-13 + 8 eps 2.8e5 = 4.97e-10 <= 0.5e-9), so it cannot fail
# open.
GEOMETRY_CONDITIONING_LIMIT = 2.5e5
GEOMETRY_GUARANTEE_REACH = 2.8e5  # largest G at which the 2x margin still holds
_FLOAT_TYPES = (float, int, np.float64)
_INT_TYPES = (int, np.int64)
# Option P (owner 2026-11-26): plain-object admission by type IDENTITY (never
# ``==`` or ``in``, which a metaclass can answer).
_PLAIN_SCALARS = (float, int, bool, str, bytes, type(None), np.float64, np.float32, np.int64, np.int32, np.bool_)
_PLAIN_CONTAINERS = (tuple, list, set, frozenset)
_PLAIN_BUDGET = 200_000  # values visited per object (measured qualified S3 guard: see the API note)

FAMILY_KEYS = {
    "membrane": "global_membrane_resultant_tensors",
    "bending": "global_bending_resultant_tensors",
    "shear": "global_transverse_shear_resultants",
}
NORMAL_KEYS = {
    "membrane": "membrane_basis_normals",
    "bending": "bending_basis_normals",
    "shear": "shear_basis_normals",
}
_SURFACE_COMPONENTS = (
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("xy", 0, 1),
    ("yz", 1, 2),
    ("xz", 0, 2),
)
_SURFACES = ("top", "bot")


class ShellAxesRefusal(ValueError):
    """Typed refusal of an axes request (not a cross-check failure)."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = str(code)
        self.reason = str(reason)
        super().__init__(f"{self.code}: {self.reason}")


class ShellAxesFrameRuleMismatch(ShellAxesRefusal):
    """Raised by :meth:`ShellAxesOutputs.require_all` when a family was refused."""

    def __init__(self, refusals: Mapping[str, "Refusal"]) -> None:
        self.refusals = dict(refusals)
        text = "; ".join(
            f"{family}: {refusal.reason} ({refusal.detail})"
            for family, refusal in sorted(self.refusals.items())
        )
        super().__init__(AXES_FRAME_RULE_MISMATCH, text)


@dataclass(frozen=True)
class Refusal:
    code: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class ShellAxesOutputs:
    """Derived axes outputs of one element and one recovery.

    ``arrays`` holds read-only float arrays (``director_polarity`` and the
    offsets as 0-d arrays).  ``refusals`` maps a refused family to its
    ``Refusal``; ``certified`` maps every family to whether it passed rules
    (1)-(5).  ``diagnostics`` records per family ``residual`` (r_f),
    ``allowance`` (a_f), ``scale`` (s_f) and, for surface-only branches,
    ``surface_scale`` and ``out_of_plane_residual``.  ``conventions`` names the
    sign convention of each family's global tensor: ``numbered_normal`` (legacy,
    no director), ``physical_director`` (positive moment = tension on the
    +physical-director side) or ``owner_normal`` (qualified S3 global M, Q,
    multiplied back by the polarity); membrane is ``polarity_even``.
    """

    branch: str
    arrays: Dict[str, np.ndarray]
    refusals: Dict[str, Refusal]
    certified: Dict[str, bool]
    diagnostics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    conventions: Dict[str, str] = field(default_factory=dict)

    def require_all(self) -> "ShellAxesOutputs":
        if self.refusals:
            raise ShellAxesFrameRuleMismatch(self.refusals)
        return self


@dataclass(frozen=True)
class ShellAxesRecovery:
    """``recovery``: the dict the element's own ``compute_stresses(...,
    return_global=True)`` returned (not copied, not modified); ``axes``: the
    outputs derived from exactly that recovery."""

    recovery: Dict[str, Any]
    axes: ShellAxesOutputs


# ----------------------------------------------------------------- helpers
def _frozen(values: Any, dtype=float) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _in_plane_tensors(rows: np.ndarray) -> np.ndarray:
    """(n,3) rows (11, 22, 12) -> (n,3,3) local in-plane tensors."""

    tensors = np.zeros((len(rows), 3, 3), dtype=float)
    tensors[:, 0, 0] = rows[:, 0]
    tensors[:, 1, 1] = rows[:, 1]
    tensors[:, 0, 1] = rows[:, 2]
    tensors[:, 1, 0] = rows[:, 2]
    return tensors


def _rotate_tensors(frames: np.ndarray, local: np.ndarray) -> np.ndarray:
    """G T G^T per station (one rotation)."""

    return np.asarray([g @ t @ g.T for g, t in zip(frames, local)], dtype=float)


def _rotate_vectors(frames: np.ndarray, local: np.ndarray) -> np.ndarray:
    """(q1 g1 + q2 g2) per station, same expression order as the elements."""

    return np.asarray(
        [q[0] * g[:, 0] + q[1] * g[:, 1] for g, q in zip(frames, local)], dtype=float
    )


def _max_abs(*arrays: np.ndarray) -> float:
    """Largest magnitude; NaN if any entry is NaN (never order dependent)."""

    values = [float(np.max(np.abs(a))) for a in arrays if np.size(a)]
    return float(np.max(values)) if values else 0.0


def _finite(*arrays: Any) -> bool:
    return all(np.all(np.isfinite(np.asarray(a, dtype=float))) for a in arrays)


def _numeric(value: Any, what: str) -> np.ndarray:
    """Float array of a recovery value, or a typed refusal (never a bare error)."""

    try:
        # inspection inside the typed boundary too (ragged input raises here)
        if np.iscomplexobj(value):
            raise ShellAxesRefusal("AXES_RECOVERY_MALFORMED", f"{what} is complex")
        return np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ShellAxesRefusal("AXES_RECOVERY_MALFORMED", f"{what} is not numeric: {exc}") from None


def _column_stack(recovered: Mapping[str, Any], keys: Tuple[str, ...], n: int) -> Optional[np.ndarray]:
    """(n, len(keys)) native values, or None when any key is missing or not (n,)."""

    if not all(key in recovered for key in keys):
        return None
    columns = [_numeric(recovered[key], key) for key in keys]
    if any(column.shape != (n,) for column in columns):
        return None
    return np.column_stack(columns)


def _frame_problem(frames: np.ndarray, bound: float = ORTHONORMAL_BOUND) -> str:
    """'' when every frame is finite, orthonormal (to ``bound``) and right-handed."""

    frames = np.asarray(frames, dtype=float)
    if not _finite(frames):
        return "frame is not finite"
    gram = np.einsum("nki,nkj->nij", frames, frames) - np.eye(3)[None]
    if not float(np.max(np.abs(gram))) <= bound:
        return "frame is not orthonormal"
    if not bool(np.all(np.linalg.det(frames) > 0.0)):
        return "frame is not right-handed (reversed basis normal)"
    return ""


_TYPE_NAME = type.__dict__["__name__"]  # the C slot of ``type``; no metaclass lookup


def _type_name(kind: type) -> str:
    """``kind.__name__`` for a refusal text without running a metaclass
    ``__getattribute__`` or ``__name__`` override (Codex r2 of 2026-11-27).
    A class may carry a ``str`` subclass as its name, whose ``__format__``
    would run in the refusal f-string; only an exact ``str`` is used (r3)."""

    name = _TYPE_NAME.__get__(kind, type)
    return name if type(name) is str else "<type with a non-str name>"


def _id_text(value: Any) -> str:
    """An exact int id for a refusal text; very large ids exceed Python's
    int-to-str digit limit, so they are described by size (Codex r2/r3)."""

    if -(10**18) < value < 10**18:
        return str(int(value))
    return f"<int of {int(value).bit_length()} bits>"


def _is(kind: type, allowed: Tuple[type, ...]) -> bool:
    return any(kind is candidate for candidate in allowed)


@functools.lru_cache(maxsize=None)
def _solver_value_types() -> Tuple[type, ...]:
    """Exact solver-owned classes ordinary elements, meshes and nodes hold as
    attribute values (measured on ANYfem models); trusted under A'' and
    admitted by identity without looking inside."""

    from anymaterial import IsotropicMaterial, OrthotropicMaterial

    from .activity import ElementActivity
    from .fe_core import DOFManager, FEMesh, _QualifiedMutationEpoch, _QualifiedStateMapping
    from .shell_sections import GeneralizedShellSection

    # FEMesh and the materials appear inside the qualified recovery guards
    # (measured on ANYfem barge5 after a solve); admitted by identity only
    return (
        _QualifiedMutationEpoch,
        _QualifiedStateMapping,
        DOFManager,
        GeneralizedShellSection,
        FEMesh,
        IsotropicMaterial,
        OrthotropicMaterial,
        ElementActivity,  # FEModel.set_element_activity attaches it to the mesh
    )


def _admit_recovery_batch_plan(value: Any) -> bool:
    from .recovery_batches import RecoveryBatchPlan

    return type(value) is RecoveryBatchPlan


def _admit_s3_stiffness_plan(value: Any) -> bool:
    from .s3_reference_batch import PreparedReferenceS3Components

    return type(value) is PreparedReferenceS3Components


def _admit_v2d_lease_epochs(value: Any) -> bool:
    """What ``_subscribe_v2d_lease_epoch`` stores: an exact list of exact
    callback-free ``weakref.ref``.  The references are never dereferenced: a
    temporary strong reference to the referent could, on release, run weak
    reference callbacks registered elsewhere (Codex r1).  The referents are
    ignored; nothing in this module reads them."""

    if type(value) is not list:
        return False
    items = list.copy(value)  # one snapshot; the solver appends under its own lock
    if len(items) > _PLAIN_BUDGET:
        return False
    for item in items:
        if type(item) is not weakref.ReferenceType or item.__callback__ is not None:
            return False
    return True


@functools.lru_cache(maxsize=None)
def _solver_cache_admitters() -> Mapping[Tuple[type, str], Callable[[Any], bool]]:
    """Solver-owned caches the solve and the ordinary stress recovery attach
    (measured 2026-11-27 after ``solve_linear`` + ``recover_stress_result`` on
    legacy Q4/Q8, e4-pl Q4, qualified S3 and V2D models; the S3 stiffness plan
    after a repeated assembly on qualified S3 shells), admitted only under
    their own key on their exact owner type.  Everything else keeps the plain
    rule."""

    from .e4_pl_s3_v2d_element import _V2D_LEASE_EPOCHS_KEY, NativeParityE4PLS3V2DShellElement
    from .fe_core import FEMesh

    return types.MappingProxyType(
        {
            (FEMesh, "_recovery_batch_plan"): _admit_recovery_batch_plan,
            (FEMesh, "_qualified_s3_reference_stiffness_plan"): _admit_s3_stiffness_plan,
            (NativeParityE4PLS3V2DShellElement, _V2D_LEASE_EPOCHS_KEY): _admit_v2d_lease_epochs,
        }
    )


def _plain_value(value: Any) -> bool:
    """Plain: exact scalars, exact numeric ndarrays, exact containers of plain
    values (any depth, iterative, cycle-safe, bounded by ``_PLAIN_BUDGET``), or
    an exact admitted solver class (not looked into)."""

    solver_types = _solver_value_types()
    stack = [value]
    seen = set()
    visited = 0
    while stack:
        item = stack.pop()
        visited += 1
        if visited > _PLAIN_BUDGET:
            return False
        kind = type(item)
        if _is(kind, _PLAIN_SCALARS):
            continue
        if kind is np.ndarray:
            if item.dtype.hasobject or item.dtype.kind not in "biuf":
                return False
            continue
        if _is(kind, solver_types):
            continue
        if id(item) in seen:
            continue
        if _is(kind, _PLAIN_CONTAINERS):
            seen.add(id(item))
            stack.extend(item)
        elif kind is dict:
            seen.add(id(item))
            for key, entry in dict.items(item):
                stack.append(key)
                stack.append(entry)
        elif kind is types.MappingProxyType:  # qualified S3 component record (measured)
            # read the backing mapping through the C traversal, never through
            # the proxy: a proxy over a Python mapping would dispatch to its
            # __iter__/__getitem__ (Codex r2 of 2026-11-27); only exact dict
            try:
                backing = gc.get_referents(item)
            except Exception:  # an audit hook may deny the inspection: refuse (Codex r3)
                return False
            if type(backing) is not list or len(backing) != 1 or type(backing[0]) is not dict:
                return False
            seen.add(id(item))
            for key, entry in dict.items(backing[0]):
                stack.append(key)
                stack.append(entry)
        else:
            return False
    return True


@functools.lru_cache(maxsize=None)
def _class_attribute_names(kind: type) -> frozenset:
    """Every name bound in the MRO of an admitted class (data or callable),
    except the declared dataclass fields of ``kind``: those are instance-owned
    by declaration and their class value is only the default (measured:
    FEMesh.element_activity).  Cached per class: A'' trusts the class
    internals not to be rebound."""

    names = set()
    for cls in kind.__mro__:
        names.update(vars(cls))
    fields = vars(kind).get("__dataclass_fields__")
    if type(fields) is dict:
        names.difference_update(fields)
    return frozenset(names)


def _admit_plain_object(obj: Any, kind: type, what: str) -> Dict[str, Any]:
    """Option P for one object already admitted by exact type ``kind``.

    The instance dict must be an exact ``dict`` with exact ``str`` keys, no key
    may name any class attribute (data or method; the attribute lookup would
    otherwise prefer the instance value), and every value must be plain (exact
    scalars, exact containers of plain values, exact numeric ndarrays) or an
    exact solver-owned class.  Returns the instance dict.
    """

    try:
        own = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        raise ShellAxesRefusal("AXES_OBJECT_NOT_PLAIN", f"{what} has no instance dict") from None
    if type(own) is not dict:
        raise ShellAxesRefusal("AXES_OBJECT_NOT_PLAIN", f"{what} instance dict is {_type_name(type(own))}, not dict")
    keys = list(dict.keys(own))
    if not all(type(key) is str for key in keys):
        raise ShellAxesRefusal("AXES_OBJECT_NOT_PLAIN", f"{what} instance dict has a non-str key")
    shadows = _class_attribute_names(kind).intersection(keys)
    if shadows:
        raise ShellAxesRefusal(
            "AXES_INSTANCE_SHADOW",
            f"{what} instance attribute(s) {sorted(shadows)!r} shadow {_type_name(kind)} class attributes",
        )
    admitters = _solver_cache_admitters()
    for key in keys:
        value = dict.__getitem__(own, key)
        admit_cache = admitters.get((kind, key))
        if admit_cache is not None and admit_cache(value):
            continue
        if not _plain_value(value):
            raise ShellAxesRefusal(
                "AXES_OBJECT_NOT_PLAIN",
                f"{what}.{key} is {_type_name(type(dict.__getitem__(own, key)))}, not a plain or admitted solver type",
            )
    return own


def _admit_mesh_and_nodes(element_dict: Mapping[str, Any], mesh: Any) -> None:
    """Option P for the mesh and the element's own nodes (exact FEMesh / Node)."""

    from .fe_core import FEMesh, Node, _QualifiedStateMapping

    if type(mesh) is not FEMesh:
        raise ShellAxesRefusal("AXES_OBJECT_NOT_PLAIN", f"mesh is {_type_name(type(mesh))}, not FEMesh")
    mesh_dict = _admit_plain_object(mesh, FEMesh, "mesh")
    nodes = mesh_dict.get("nodes")
    if not (type(nodes) is _QualifiedStateMapping or type(nodes) is dict):
        raise ShellAxesRefusal("AXES_OBJECT_NOT_PLAIN", "mesh.nodes is not the solver node mapping")
    node_ids = element_dict.get("node_ids")
    if not (_is(type(node_ids), (tuple, list)) and all(_is(type(n), (int, np.int64, np.int32)) for n in node_ids)):
        raise ShellAxesRefusal("AXES_OBJECT_NOT_PLAIN", "element node_ids are not exact ints")
    for node_id in node_ids:
        node = dict.get(nodes, node_id)
        if type(node) is not Node:
            raise ShellAxesRefusal("AXES_OBJECT_NOT_PLAIN", f"node {_id_text(node_id)} is {_type_name(type(node))}, not Node")
        _admit_plain_object(node, Node, f"node {_id_text(node_id)}")


def _check_attribute_types(element: Any, names: Tuple[str, ...]) -> None:
    """Exact built-in scalar types for the attributes the recovery and B2 both read.

    A float subclass (say with its own ``**`` or ``==``) would let the recovery
    and this module give the same attribute different meanings (Codex r4).
    """

    for name in names:
        try:
            value = getattr(element, name)  # exact admitted class: trusted lookup
        except AttributeError:
            raise ShellAxesRefusal("AXES_ATTRIBUTE_TYPE", f"{name} is missing") from None
        if name == "director_polarity":
            ok = _is(type(value), _INT_TYPES) and (int(value) == 1 or int(value) == -1)
        elif name == "reference_normal":
            ok = value is None or (
                type(value) is np.ndarray and value.dtype == np.float64 and value.shape == (3,)
            )
        else:
            ok = _is(type(value), _FLOAT_TYPES)
        if not ok:
            raise ShellAxesRefusal(
                "AXES_ATTRIBUTE_TYPE",
                f"{name} is {_type_name(type(value))}; an exact built-in type is required",
            )


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]),
        dtype=float,
    )


def _safe_norm(vector: np.ndarray) -> float:
    """Euclidean norm without overflow or underflow of the squares."""

    largest = float(np.max(np.abs(vector)))
    if largest == 0.0 or not math.isfinite(largest):
        return largest
    scaled = np.asarray(vector, dtype=float) / largest
    return largest * math.sqrt(float(scaled @ scaled))


def _power_of_two_scaled(coords: np.ndarray) -> np.ndarray:
    """Coordinates times an exact power of two so max|x| is in [0.5, 1): G is
    scale invariant and differences can no longer overflow (Codex r9)."""

    coords = np.asarray(coords, dtype=float)
    largest = float(np.max(np.abs(coords)))
    if not (math.isfinite(largest) and largest > 0.0):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "node coordinates are zero or not finite")
    _mantissa, exponent = math.frexp(largest)
    # a negligible coordinate may vanish here (e.g. z = 2**-1074 next to 1):
    # that changes no difference that matters (Codex r11: refusing it was a
    # regression); used only when the unscaled differences overflow
    return np.ldexp(coords, -exponent)


def _geometry_conditioning(coords: np.ndarray, a: np.ndarray, b: np.ndarray, *lengths: float) -> float:
    """G = (max|x| / shortest vector + 1) * |a||b| / |a x b| for the two vectors
    a, b whose cross product gives a geometric normal (tangents, diagonals or
    edges).  Relative roundoff of the normal is <= 8 eps G: the first factor
    covers coordinate differences far from the origin, the second the
    cancellation of a nearly parallel pair.

    Exponent safe (Codex r8-r11): each vector is normalized by its own largest
    component before the cross product, so the dimensionless ratio neither
    overflows nor underflows; lengths use ``_safe_norm``; an extreme
    reach / shortest overflows to inf, which is outside the domain."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not _finite(a, b, coords) or not all(math.isfinite(float(length)) for length in lengths):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "nonfinite tangent, diagonal or edge")
    largest_a = float(np.max(np.abs(a)))
    largest_b = float(np.max(np.abs(b)))
    if largest_a == 0.0 or largest_b == 0.0 or any(float(length) == 0.0 for length in lengths):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "zero tangent, diagonal or edge")
    a_hat = a / largest_a
    b_hat = b / largest_b
    area_hat = _safe_norm(_cross(a_hat, b_hat))
    if area_hat == 0.0:
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "parallel tangents, diagonals or edges")
    norm_a_hat = _safe_norm(a_hat)
    norm_b_hat = _safe_norm(b_hat)
    ratio = norm_a_hat * norm_b_hat / area_hat
    with np.errstate(over="ignore"):
        shortest = min([largest_a * norm_a_hat, largest_b * norm_b_hat, *(float(length) for length in lengths)])
        reach = float(np.max(np.abs(coords)))
        conditioning = (reach / shortest + 1.0) * ratio
    return conditioning if math.isfinite(conditioning) else math.inf


def _relative_tangents(element: Any, coords: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Shape-function tangents from element-relative coordinates (the
    derivatives sum to zero, so this is the same tangent without the
    cancellation of large absolute coordinates; Codex r7)."""

    relative = coords - coords[0]
    _n, d_xi, d_eta = element.compute_shape_functions(float(xi), float(eta))
    return np.asarray(d_xi, dtype=float) @ relative, np.asarray(d_eta, dtype=float) @ relative


def _check_geometry_domain(element: Any, coords: np.ndarray, requested: Optional[np.ndarray] = None) -> float:
    """The declared domain, the same measure for every branch: the largest G
    over the edges (3-node), or the diagonals (4-node) and the shape-function
    tangents at the element's own stations and at any requested
    ``natural_points`` (4-, 6- and 8-node), plus the centre tangents of 4-node
    quadrilaterals.  Evaluated on power-of-two scaled coordinates (G is scale
    invariant)."""

    coords = np.asarray(coords, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        differences = coords - coords[0]
    if not _finite(differences):
        coords = _power_of_two_scaled(coords)  # G is scale invariant
    if len(coords) == 3:
        values = [_geometry_conditioning(coords, coords[1] - coords[0], coords[2] - coords[0],
                                         _safe_norm(coords[2] - coords[1]))]
    else:
        values = []
        if len(coords) == 4:
            values.append(_geometry_conditioning(coords, coords[2] - coords[0], coords[3] - coords[1]))
        # default stations, the centre of a 4-node quadrilateral (warped Q4
        # builds its centre shear frame there; Codex r8; for triangles (0, 0) is
        # a vertex, Codex r9) and any requested natural_points
        stations = np.asarray(element.gauss_points, dtype=float)
        if len(coords) == 4:
            stations = np.concatenate((stations, np.zeros((1, 2))))
        if requested is not None:
            stations = np.concatenate((stations, np.asarray(requested, dtype=float)))
        for xi, eta in stations:
            a, b = _relative_tangents(element, coords, xi, eta)
            values.append(_geometry_conditioning(coords, a, b))
    conditioning = max(values)
    if not conditioning <= GEOMETRY_CONDITIONING_LIMIT:
        raise ShellAxesRefusal(
            "AXES_GEOMETRY_CONDITIONING",
            f"geometry conditioning G={conditioning:.3e} is outside the declared domain "
            f"G <= {GEOMETRY_CONDITIONING_LIMIT:.6g} (|x| / element size, or nearly parallel edges)",
        )
    return conditioning


def _geometry_bound(coords: np.ndarray, a: np.ndarray, b: np.ndarray, *lengths: float) -> float:
    """Rule (6) bound inside the declared geometry domain; outside it a typed refusal."""

    conditioning = _geometry_conditioning(coords, a, b, *lengths)
    if not conditioning <= GEOMETRY_CONDITIONING_LIMIT:
        raise ShellAxesRefusal(
            "AXES_GEOMETRY_CONDITIONING",
            f"geometry conditioning G={conditioning:.3e} is outside the declared domain "
            f"G <= {GEOMETRY_CONDITIONING_LIMIT:.6g} (|x| / element size, or nearly parallel edges)",
        )
    return GEOMETRY_BOUND + GEOMETRY_ROUNDOFF * conditioning


def _unit_normal(cross: np.ndarray) -> np.ndarray:
    length = math.sqrt(float(cross @ cross))
    if not (math.isfinite(length) and length > 0.0):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "zero or nonfinite surface normal")
    return cross / length


def _normal_of(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit a x b, scaled first so that large finite vectors do not overflow."""

    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if not (math.isfinite(scale) and scale > 0.0):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "zero or nonfinite surface normal")
    return _unit_normal(_cross(np.asarray(a, dtype=float) / scale, np.asarray(b, dtype=float) / scale))


def _diagonal_normal(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """Bilinear Q4 centre normal: X_xi x X_eta at the centre = (x2-x0) x (x3-x1) / 8,
    from coordinate differences (well conditioned far from the origin)."""

    d1 = coords[2] - coords[0]
    d2 = coords[3] - coords[1]
    normal = _normal_of(d1, d2)
    return normal[None], _geometry_bound(coords, d1, d2)


def _facet_normal(coords: np.ndarray, owner: Any) -> Tuple[np.ndarray, float]:
    """Flat-triangle facet normal oriented by the owner normal (qualified S3 convention)."""

    e1 = coords[1] - coords[0]
    e2 = coords[2] - coords[0]
    e3 = coords[2] - coords[1]
    normal = _normal_of(e1, e2)
    alignment = float(normal @ np.asarray(owner, dtype=float))
    if not (math.isfinite(alignment) and alignment != 0.0):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "owner normal is tangential to the facet")
    bound = _geometry_bound(coords, e1, e2, math.sqrt(float(e3 @ e3)))
    return (normal if alignment > 0.0 else -normal)[None], bound


def _tangent_normals(element: Any, coords: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rule (6) geometry: normalized cross product of the shape-function
    tangents, with the per-station bound (see ``_geometry_bound``)."""

    out = np.zeros((len(points), 3), dtype=float)
    bounds = np.zeros(len(points), dtype=float)
    for index, (xi, eta) in enumerate(points):
        a, b = _relative_tangents(element, coords, xi, eta)
        out[index] = _normal_of(a, b)
        bounds[index] = _geometry_bound(coords, a, b)
    return out, bounds


def _oriented(normals: np.ndarray, director: np.ndarray) -> np.ndarray:
    """Orient each geometric normal to the physical reference director."""

    alignment = normals @ np.asarray(director, dtype=float)
    if not (_finite(alignment) and bool(np.all(alignment != 0.0))):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "geometric normal is orthogonal to the director")
    return normals * np.sign(alignment)[:, None]


def _unit_rows(vector: Any, n: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if not (math.isfinite(length) and length > 0.0):
        raise ShellAxesRefusal("AXES_GEOMETRY_DEGENERATE", "reference normal is zero or not finite")
    return np.repeat((vector / length)[None], n, axis=0)


def _requested_points(natural_points: Any) -> Optional[np.ndarray]:
    """The caller's compute_stresses natural_points request, validated like the solver."""

    if natural_points is None:
        return None
    # option P: validate the uncast request (Codex r5: object arrays hid a complex inf)
    kind = type(natural_points)
    if kind is np.ndarray:
        plain = natural_points.dtype.kind in "iuf" and not natural_points.dtype.hasobject
    elif _is(kind, (list, tuple)):
        plain = all(
            (
                _is(type(row), (list, tuple))
                and len(row) == 2
                and all(_is(type(v), (float, int, np.float64, np.float32, np.int64, np.int32)) for v in row)
            )
            or (
                type(row) is np.ndarray
                and row.shape == (2,)
                and row.dtype.kind in "iuf"
                and not row.dtype.hasobject
            )
            for row in natural_points
        )
    else:
        plain = False
    if not plain:
        raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "natural_points must be plain real numbers (list/tuple or float ndarray)")
    points = _numeric(natural_points, "natural_points")
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) == 0
        or not _finite(points)
        or np.any(np.abs(points) > 1.0)
    ):
        raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "natural_points must be nonempty finite Nx2 in [-1,1]")
    return points


def _shape_positions(element: Any, coords: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.asarray(element.compute_shape_functions(float(p[0]), float(p[1]))[0], dtype=float)
            @ coords
            for p in points
        ],
        dtype=float,
    )


# ------------------------------------------------------------ branch model
@dataclass
class _Branch:
    name: str
    surface_only: bool
    positions: np.ndarray
    station_frames: np.ndarray  # (n,3,3): frame of membrane and bending
    shear_frames: np.ndarray  # (n,3,3): frame of the shear vector
    shear_role: str  # "station" or "centre"
    moment_shear_sign: float  # factor the element applied to global M and Q
    conventions: Dict[str, str]
    metadata: Dict[str, Any]
    consistency_problem: str = ""
    expected_station_normals: Optional[np.ndarray] = None  # rule (6), (n,3)
    expected_shear_normals: Optional[np.ndarray] = None  # rule (6), (n,3)
    station_normal_bounds: Optional[np.ndarray] = None  # rule (6), (n,)
    shear_normal_bounds: Optional[np.ndarray] = None  # rule (6), (n,)
    orthonormal_bound: float = ORTHONORMAL_BOUND
    emitted_frame: Optional[np.ndarray] = None  # V2D recovery "frame"


def _apply_override(
    override: Optional[Callable[[str, np.ndarray], np.ndarray]],
    role: str,
    frames: np.ndarray,
) -> np.ndarray:
    if override is None:
        return frames
    replaced = np.array(override(role, np.array(frames, copy=True)), dtype=float)
    if replaced.shape != frames.shape:
        raise ShellAxesRefusal("AXES_INTERNAL", "frame override changed the frame shape")
    return replaced


def _station_count(recovered: Mapping[str, Any], generalized: bool) -> int:
    key = "membrane_resultants" if generalized else "membrane_xx"
    if key not in recovered:
        key = "membrane_resultants" if "membrane_resultants" in recovered else "membrane_xx"
    if key not in recovered:
        raise ShellAxesRefusal("AXES_FORMULATION_NOT_QUALIFIED", "no native family values in the recovery")
    values = _numeric(recovered[key], key)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ShellAxesRefusal("AXES_RECOVERY_MALFORMED", f"{key} has no stations")
    return int(values.shape[0])


def _exact_table(recovered: Mapping[str, Any], key: str, expected: np.ndarray, what: str) -> None:
    """The emitted station table must equal the trusted sampling rule exactly."""

    if key not in recovered:
        raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", f"{what} recovery without {key}")
    table = _numeric(recovered[key], key)
    if table.shape != expected.shape or not np.array_equal(table, expected):
        raise ShellAxesRefusal(
            "AXES_STATIONS_NOT_QUALIFIED",
            f"{what}: emitted {key} is not the sampling rule of the request",
        )


def _legacy_station_frames(element: Any, coords: np.ndarray, points: np.ndarray, material: Any, homogeneous: bool) -> np.ndarray:
    frames = None
    if homogeneous:
        # exactly where the legacy homogeneous recovery uses the batched frames
        from .recovery_shell_frames import native_recovery_frames

        frames = native_recovery_frames(element, coords, points, material)
    out = np.zeros((len(points), 3, 3), dtype=float)
    for index, (xi, eta) in enumerate(points):
        if frames is None:
            _n, d_xi, d_eta = element.compute_shape_functions(float(xi), float(eta))
            out[index] = element._local_frame_and_derivatives(coords, d_xi, d_eta)[0]
        else:
            out[index] = np.asarray(frames[index][1], dtype=float)
    return out


def _numbered_conventions() -> Dict[str, str]:
    return {"membrane": "polarity_even", "bending": "numbered_normal", "shear": "numbered_normal"}


def admit_shell_axes(element: Any, mesh: Any) -> type:
    """The module's pre-recovery admission gate, alone (public, 2026-11-27).

    Exactly the first step of ``recover_shell_axes`` (the same function, not a
    copy): exact element type, option P plain-object admission of the element,
    the mesh and the element's nodes (owner 2026-11-26), then the exact scalar
    attribute types.  Runs no element code and reads nothing else.  Raises
    ``ShellAxesRefusal`` (the code ``recover_shell_axes`` would raise for this
    gate); returns the admitted element class.  Passing it does not promise
    that the later steps (natural points, geometry domain, recovery) succeed.
    """

    from .e4_pl_element import QualifiedE4PLShellElement
    from .e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    from .elements import ShellElement

    # type() never runs element code; no attribute of an unsupported class is read
    kind = type(element)
    names = {
        ShellElement: ("thickness",),
        QualifiedE4PLShellElement: ("thickness", "planar_tolerance", "reference_normal", "director_polarity"),
        QualifiedE4PLS3ShellElement: ("thickness", "reference_normal", "director_polarity", "reference_surface_offset"),
        NativeParityE4PLS3V2DShellElement: ("thickness", "reference_normal", "director_polarity", "reference_surface_offset"),
    }
    if not any(kind is admitted for admitted in names):
        raise ShellAxesRefusal(
            "AXES_FORMULATION_NOT_QUALIFIED",
            f"no axes adapter for element type {_type_name(kind)}",
        )
    element_dict = _admit_plain_object(element, kind, "element")
    _admit_mesh_and_nodes(element_dict, mesh)
    _check_attribute_types(element, names[kind])
    return kind


_admit = admit_shell_axes  # the private name callers used before it was public


def _resolve_branch(element, mesh, material, recovered, override, requested) -> _Branch:
    from .e4_pl_element import _GAUSS, QualifiedE4PLShellElement, equation7_frame
    from .e4_pl_s3_element import TRIANGLE_QUADRATURE, QualifiedE4PLS3ShellElement, triangle_frame
    from .e4_pl_s3_v2d_element import _NORMAL_TOLERANCE, HAMMER_POINTS, NativeParityE4PLS3V2DShellElement
    from .elements import ShellElement

    kind = type(element)  # admitted by the caller (_derive_shell_axes_from_recovery)
    generalized = getattr(element, "shell_section", None) is not None
    try:
        # V2D recovers on ``_coordinates`` (exact FEMesh); the others on get_node_coordinates
        raw = element._coordinates(mesh) if kind is NativeParityE4PLS3V2DShellElement else element.get_node_coordinates(mesh)
    except (TypeError, ValueError) as exc:
        raise ShellAxesRefusal("AXES_RECOVERY_MALFORMED", f"node coordinates: {exc}") from None
    coords = _numeric(raw, "node coordinates")
    _check_geometry_domain(element, coords, requested)
    n = _station_count(recovered, generalized)

    def no_custom_stations(what: str) -> None:
        if requested is not None:
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", f"{what} takes no custom natural_points")

    def legacy(name: str) -> _Branch:
        if generalized:
            no_custom_stations(f"{name} generalized section")
        points = np.asarray(element.gauss_points, dtype=float) if requested is None else requested
        if len(points) != n:
            raise ShellAxesRefusal(
                "AXES_STATIONS_NOT_QUALIFIED",
                f"{name}: recovery has {n} stations, the element rule has {len(points)} (custom natural_points?)",
            )
        stations = _apply_override(
            override, "station", _legacy_station_frames(element, coords, points, material, not generalized)
        )
        geometry, bounds = _tangent_normals(element, coords, points)  # numbered convention
        return _Branch(
            name=name,
            surface_only=not generalized,
            positions=_shape_positions(element, coords, points),
            station_frames=stations,
            shear_frames=stations,
            shear_role="station",
            moment_shear_sign=1.0,
            conventions=_numbered_conventions(),
            metadata={},
            expected_station_normals=geometry,
            expected_shear_normals=geometry,
            station_normal_bounds=bounds,
            shear_normal_bounds=bounds,
        )

    if kind is ShellElement:
        return legacy("legacy")

    if kind is QualifiedE4PLShellElement:
        warpage = float(equation7_frame(coords)[2])
        warped = warpage > float(element.planar_tolerance)
        director: Dict[str, Any] = {}
        conventions = _numbered_conventions()
        if element.reference_normal is not None:
            director = {
                "director_polarity": int(element.director_polarity),
                "physical_reference_director": np.asarray(element.physical_reference_director, dtype=float),
            }
            conventions = {"membrane": "polarity_even", "bending": "physical_director", "shear": "physical_director"}
        if warped:
            no_custom_stations("warped e4-pl Q4")
        if warped and element.reference_normal is None:
            branch = legacy("e4pl-warped-unreferenced")
            if "warped_direct" in recovered or "natural_point_coordinates" in recovered:
                branch.consistency_problem = "recovery markers disagree with the recomputed warped legacy fallback"
            return branch
        if warped:
            points = np.asarray(element.gauss_points, dtype=float)
            if len(points) != n:
                raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "warped Q4 station count mismatch")
            numbered = _legacy_station_frames(element, coords, points, material, False)
            stations = np.asarray([element._physical_director_context(f)[0] for f in numbered], dtype=float)
            centre = np.asarray(element._physical_director_context(element._center_frame(coords))[0], dtype=float)
            stations = _apply_override(override, "station", stations)
            centre = _apply_override(override, "centre", centre)
            physical = np.asarray(element.physical_reference_director, dtype=float)
            centre_normal, centre_bound = _diagonal_normal(coords)
            # the centre frame comes from the centre tangents: their conditioning
            # bounds the centre normal too (Codex r8)
            _a0, _b0 = _relative_tangents(element, coords, 0.0, 0.0)
            centre_bound = max(centre_bound, _geometry_bound(coords, _a0, _b0))
            centre_geometry = _oriented(centre_normal, physical)
            station_geometry, station_bounds = _tangent_normals(element, coords, points)
            branch = _Branch(
                name="e4pl-warped",
                surface_only=not generalized,
                positions=_shape_positions(element, coords, points),
                station_frames=stations,
                shear_frames=np.repeat(centre[None], n, axis=0),
                shear_role="centre",
                moment_shear_sign=1.0,
                conventions=conventions,
                metadata=director,
                expected_station_normals=_oriented(station_geometry, physical),
                expected_shear_normals=np.repeat(centre_geometry, n, axis=0),
                station_normal_bounds=station_bounds,
                shear_normal_bounds=np.full(n, centre_bound),
            )
            if recovered.get("warped_direct") is not True:
                branch.consistency_problem = "recovery is not the warped direct path the module recomputed"
            return branch
        if generalized:
            no_custom_stations("planar e4-pl generalized section")
        # the trusted sampling request (default rule or the caller's natural_points)
        points = np.asarray(_GAUSS, dtype=float) if requested is None else requested
        _exact_table(recovered, "natural_point_coordinates", points, "planar e4-pl")
        if points.shape != (n, 2):
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "planar e4-pl station table mismatch")
        frame = np.asarray(element._physical_director_context(equation7_frame(coords)[0])[0], dtype=float)
        stations = _apply_override(override, "station", np.repeat(frame[None], n, axis=0))
        # bilinear Q4: the centre tangent cross product is the diagonal cross
        # product, exact for any admitted (near-)planar facet
        centre_geometry, centre_bound = _diagonal_normal(coords)
        if element.reference_normal is not None:
            centre_geometry = _oriented(centre_geometry, np.asarray(element.physical_reference_director, dtype=float))
        geometry = np.repeat(centre_geometry, n, axis=0)
        return _Branch(
            name="e4pl-planar",
            expected_station_normals=geometry,
            expected_shear_normals=geometry,
            station_normal_bounds=np.full(n, centre_bound),
            shear_normal_bounds=np.full(n, centre_bound),
            surface_only=False,
            positions=_shape_positions(element, coords, points),
            station_frames=stations,
            shear_frames=stations,
            shear_role="station",
            moment_shear_sign=1.0,
            conventions=conventions,
            metadata=director,
        )

    if kind is QualifiedE4PLS3ShellElement:
        no_custom_stations("qualified S3")
        if element.reference_normal is None:
            raise ShellAxesRefusal("AXES_FORMULATION_NOT_QUALIFIED", "qualified S3 without reference_normal")
        if n != len(TRIANGLE_QUADRATURE):
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "qualified S3 station count mismatch")
        barycentric = np.asarray([(1.0 - r - s, r, s) for r, s, _w in TRIANGLE_QUADRATURE], dtype=float)
        frame = np.asarray(triangle_frame(coords, element.reference_normal)[0], dtype=float)
        stations = _apply_override(override, "station", np.repeat(frame[None], n, axis=0))
        polarity = int(element.director_polarity)
        # facet normal oriented by the owner normal (not times the polarity)
        facet, facet_bound = _facet_normal(coords, element.reference_normal)
        geometry = np.repeat(facet, n, axis=0)
        return _Branch(
            name="s3",
            expected_station_normals=geometry,
            expected_shear_normals=geometry,
            station_normal_bounds=np.full(n, facet_bound),
            shear_normal_bounds=np.full(n, facet_bound),
            surface_only=False,
            positions=barycentric @ coords,
            station_frames=stations,
            shear_frames=stations,
            shear_role="station",
            moment_shear_sign=float(polarity),
            conventions={"membrane": "polarity_even", "bending": "owner_normal", "shear": "owner_normal"},
            metadata={
                "director_polarity": polarity,
                "physical_reference_director": np.asarray(element.physical_reference_director, dtype=float),
                **_offsets(element),
            },
        )

    if kind is NativeParityE4PLS3V2DShellElement:
        no_custom_stations("S3 V2D")
        hammer = np.asarray(HAMMER_POINTS, dtype=float)
        barycentric = np.column_stack((1.0 - hammer[:, 0] - hammer[:, 1], hammer[:, 0], hammer[:, 1]))
        _exact_table(recovered, "external_barycentric_coordinates", barycentric, "S3 V2D")
        if barycentric.shape != (n, 3):
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "V2D station table mismatch")
        positions = barycentric @ coords
        if "physical_station_coordinates" not in recovered:
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "V2D recovery without physical_station_coordinates")
        emitted = _numeric(recovered["physical_station_coordinates"], "physical_station_coordinates")
        # the same expression on the same coordinates: exact (Codex r4: a
        # relative bound weakens far from the origin)
        if emitted.shape != positions.shape or not np.array_equal(emitted, positions):
            raise ShellAxesRefusal(
                "AXES_STATIONS_NOT_QUALIFIED",
                "V2D emitted physical_station_coordinates differ from the recomputed stations",
            )
        if element.reference_normal is None:
            raise ShellAxesRefusal("AXES_FORMULATION_NOT_QUALIFIED", "S3 V2D without reference_normal")
        _facet_normal(coords, element.reference_normal)  # geometry domain (the V2D normal is the owner normal)
        frame, orthonormal_bound = _v2d_frame(coords, np.asarray(element.reference_normal, dtype=float), _NORMAL_TOLERANCE)
        if "frame" not in recovered:
            raise ShellAxesRefusal("AXES_RETURN_GLOBAL_REQUIRED", "V2D recovery without its frame")
        emitted_frame = _numeric(recovered["frame"], "frame")
        if emitted_frame.shape != (3, 3):
            raise ShellAxesRefusal("AXES_RECOVERY_MALFORMED", "V2D emitted frame is not 3x3")
        stations = _apply_override(override, "station", np.repeat(frame[None], n, axis=0))
        geometry = _unit_rows(element.reference_normal, n)
        return _Branch(
            name="v2d",
            expected_station_normals=geometry,
            expected_shear_normals=geometry,
            orthonormal_bound=orthonormal_bound,
            emitted_frame=emitted_frame,
            surface_only=False,
            positions=positions,
            station_frames=stations,
            shear_frames=stations,
            shear_role="station",
            moment_shear_sign=1.0,
            conventions={"membrane": "polarity_even", "bending": "physical_director", "shear": "physical_director"},
            metadata={
                "director_polarity": int(element.director_polarity),
                "physical_reference_director": np.asarray(element.physical_reference_director, dtype=float),
                **_offsets(element),
            },
        )

    raise AssertionError("unreachable: element type was admitted above")


def _offsets(element: Any) -> Dict[str, float]:
    return {
        "reference_surface_offset": float(element.reference_surface_offset),
        "section_origin_offset_from_reference": float(element.material_mid_surface_offset_from_reference),
        "physical_bottom_offset_from_reference": float(element.material_surface_offset_from_reference(-1.0)),
        "physical_top_offset_from_reference": float(element.material_surface_offset_from_reference(1.0)),
    }


def _v2d_frame(external: np.ndarray, normal: np.ndarray, normal_tolerance: float) -> Tuple[np.ndarray, float]:
    """V2D/V2C facet frame, recomputed here (x along the first internal edge).

    Returns the frame and its orthonormality bound.  The frame normal is the
    owner reference normal, which V2D admits when ``|e . n| <=
    _NORMAL_TOLERANCE sqrt(edge_scale)`` for the two edges at node 0
    (``edge_scale`` = largest squared edge), so ``|x . n|`` may reach
    ``_NORMAL_TOLERANCE sqrt(edge_scale) / |x edge|``; the bound is that plus
    the ordinary ``ORTHONORMAL_BOUND`` (y = n x x is exactly orthogonal to n
    and x up to roundoff).
    """

    cross = np.cross(external[1] - external[0], external[2] - external[0])
    order = (0, 1, 2) if float(cross @ normal) > 0.0 else (0, 2, 1)
    ordered = external[np.asarray(order, dtype=np.intp)]
    x_edge = ordered[1] - ordered[0]
    x_length = float(np.linalg.norm(x_edge))
    x_axis = x_edge / x_length
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    edge_scale = max(
        float((external[1] - external[0]) @ (external[1] - external[0])),
        float((external[2] - external[0]) @ (external[2] - external[0])),
        float((external[2] - external[1]) @ (external[2] - external[1])),
    )
    admitted = float(normal_tolerance) * math.sqrt(edge_scale) / x_length
    return np.column_stack((x_axis, y_axis, normal)), ORTHONORMAL_BOUND + admitted


# --------------------------------------------------------------- the check
def _certify(
    derived: np.ndarray,
    reference: np.ndarray,
    allowance_extra: float,
    conditioning: float = 1.0,
) -> Tuple[Optional[str], Dict[str, float], str]:
    """Rules (2)-(3) for one family; returns (reason or None, diagnostics, detail).

    ``conditioning`` (<= 1) is the smallest singular value of the map from the
    certified quantity to the compared one (warped centre shear projected into
    the station tangent planes): an error e in the certified quantity shows as
    at least ``conditioning |e|``, so rule (3) requires ``a_f / conditioning <=
    1e-9 s_f``.
    """

    if not _finite(derived, reference):
        return "nonfinite", {}, "derived or reference values are not finite"
    residual = float(np.max(np.abs(derived - reference))) if derived.size else 0.0
    scale = _max_abs(reference)
    allowance = RELATIVE_BOUND * scale + allowance_extra
    diag = {"residual": residual, "allowance": allowance, "scale": scale}
    if scale == 0.0:
        if np.all(derived == 0.0):
            return None, diag, ""
        return "residual", diag, "reference family is exactly zero, derived is not"
    if not allowance <= RESOLUTION * scale * conditioning:
        return (
            "unresolvable_family",
            diag,
            f"a_f={allowance:.3e} > 1e-9 s_f c={RESOLUTION * scale * conditioning:.3e} (conditioning c={conditioning:.3e})",
        )
    if not residual <= allowance:
        return "residual", diag, f"r_f={residual:.3e} > a_f={allowance:.3e}"
    return None, diag, ""


def _tangency(normals: np.ndarray, tensors: np.ndarray, nonorthogonality: float = 0.0) -> Tuple[bool, float]:
    """|nu . T| <= (1e-13 + 2 |x . nu|) s_T: a tensor built in a frame whose x
    axis is not exactly orthogonal to nu (admitted V2D) has that normal part."""

    scale = _max_abs(tensors)
    value = float(np.max(np.abs(np.einsum("ni,nij->nj", normals, tensors)))) if tensors.size else 0.0
    return value <= (RELATIVE_BOUND + 2.0 * nonorthogonality) * scale, value


def recover_shell_axes(
    element: Any,
    mesh: Any,
    displacements: Any,
    material: Any,
    *,
    natural_points: Any = None,
    _frame_override: Optional[Callable[[str, np.ndarray], np.ndarray]] = None,
) -> ShellAxesRecovery:
    """Recover one shell element and derive its cross-checked axes outputs (B3).

    The recovery is ``type(element).compute_stresses(element, mesh,
    displacements, material, return_global=True[, natural_points=...])``,
    called through the trusted class method after admission; use
    ``result.recovery`` in place of a separate ``compute_stresses`` call (it
    is that dict, bit-identical to a direct call).  Solver errors of the
    recovery propagate unchanged.  ``natural_points`` (legacy and planar
    e4-pl homogeneous only) is passed to the recovery and binds the stations.
    """

    from .e4_pl_element import QualifiedE4PLShellElement
    from .elements import ShellElement

    kind = admit_shell_axes(element, mesh)
    requested = _requested_points(natural_points)
    if requested is not None and (
        not (kind is ShellElement or kind is QualifiedE4PLShellElement) or element.shell_section is not None
    ):
        raise ShellAxesRefusal(
            "AXES_STATIONS_NOT_QUALIFIED",
            "natural_points apply to homogeneous legacy and planar e4-pl recovery only",
        )
    # the declared geometry domain before the recovery, so the typed refusal is
    # not pre-empted by a solver error on extreme geometry (Codex r7)
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    try:
        raw = element._coordinates(mesh) if kind is NativeParityE4PLS3V2DShellElement else element.get_node_coordinates(mesh)
    except (TypeError, ValueError) as exc:
        raise ShellAxesRefusal("AXES_RECOVERY_MALFORMED", f"node coordinates: {exc}") from None
    _check_geometry_domain(element, _numeric(raw, "node coordinates"), requested)
    keywords: Dict[str, Any] = {} if requested is None else {"natural_points": np.array(requested, copy=True)}
    recovery = kind.compute_stresses(element, mesh, displacements, material, return_global=True, **keywords)
    axes = _derive_shell_axes_from_recovery(
        element,
        mesh,
        displacements,
        material,
        recovery,
        natural_points=requested,
        _frame_override=_frame_override,
    )
    return ShellAxesRecovery(recovery=recovery, axes=axes)


def _derive_shell_axes_from_recovery(
    element: Any,
    mesh: Any,
    displacements: Any,
    material: Any,
    recovered: Mapping[str, Any],
    *,
    natural_points: Any = None,
    _frame_override: Optional[Callable[[str, np.ndarray], np.ndarray]] = None,
) -> ShellAxesOutputs:
    """Derive and cross-check the axes outputs of one shell element (private;
    the public route is :func:`recover_shell_axes`, which supplies the
    recovery itself).  The checks below stay defensive against a recovery
    dict that is not the element's own.

    ``recovered`` must be the dict ``element.compute_stresses(mesh,
    displacements, material, return_global=True)`` returned; it is read only.
    ``displacements`` is accepted for the call contract (the module recomputes
    frames on the reference geometry the recovery used) and is not modified.
    ``natural_points`` must be the same request passed to ``compute_stresses``
    (``None`` = the element's default rule).  The legacy recovery emits no
    station table, so its stations are bound to this request; branches that
    emit one (planar e4-pl, V2D) must emit exactly the requested rule, and
    branches that take no custom stations refuse a request.
    Element types are admitted by exact type before any attribute is read, and
    instance attributes that shadow element methods are refused.
    ``_frame_override(role, frames)`` (role ``"station"`` with an (n,3,3)
    array or ``"centre"`` with a (3,3) array) is a private test hook that
    replaces the recomputed frames, so a wrong frame can be shown refused.
    """

    del displacements  # frames depend on the reference geometry only
    admit_shell_axes(element, mesh)  # before any attribute access (Codex r4: no early lookups)
    if not isinstance(recovered, Mapping):
        raise ShellAxesRefusal("AXES_RETURN_GLOBAL_REQUIRED", "recovery output is not a mapping")
    surface_keys = [f"global_{label}_{surface}" for surface in _SURFACES for label, _i, _j in _SURFACE_COMPONENTS]
    has_families = any(key in recovered for key in FAMILY_KEYS.values())
    has_surfaces = any(key in recovered for key in surface_keys)
    if not has_families and not has_surfaces:
        raise ShellAxesRefusal(
            "AXES_RETURN_GLOBAL_REQUIRED",
            "recovery lacks the global outputs; call compute_stresses(..., return_global=True)",
        )

    branch = _resolve_branch(element, mesh, material, recovered, _frame_override, _requested_points(natural_points))
    n = len(branch.positions)
    stations = branch.station_frames
    shear_frames = branch.shear_frames
    refusals: Dict[str, Refusal] = {}
    diagnostics: Dict[str, Dict[str, float]] = {}
    family_arrays: Dict[str, np.ndarray] = {}

    def refuse(family: str, reason: str, detail: str) -> None:
        if family not in refusals:
            refusals[family] = Refusal(AXES_FRAME_RULE_MISMATCH, reason, detail)

    if branch.consistency_problem:
        for family in FAMILIES:
            refuse(family, "residual", branch.consistency_problem)

    # Every frame used must be finite, orthonormal and right-handed: the
    # in-plane tensor comparisons cannot see a reversed basis normal (Codex r3).
    station_families = ("membrane", "bending", "shear") if branch.shear_role == "station" else ("membrane", "bending")
    station_problem = _frame_problem(stations, branch.orthonormal_bound)
    if station_problem:
        for family in station_families:
            refuse(family, "residual", f"station {station_problem}")
    shear_problem = _frame_problem(shear_frames, branch.orthonormal_bound)
    if shear_problem:
        refuse("shear", "residual", f"shear {shear_problem}")

    with np.errstate(invalid="ignore"):
        nonorthogonality = float(np.max(np.abs(np.einsum("ni,ni->n", stations[:, :, 0], stations[:, :, 2]))))
    if not math.isfinite(nonorthogonality):
        nonorthogonality = 0.0  # nonfinite frames are refused above

    station_normals = stations[:, :, 2]
    shear_normals = shear_frames[:, :, 2]

    if branch.surface_only:
        # ---- surface-only branches: legacy homogeneous, e4-pl warped homogeneous
        h = float(element.thickness)
        sigma_m = _column_stack(recovered, ("membrane_xx", "membrane_yy", "membrane_xy"), n)
        sigma_b = _column_stack(recovered, ("bending_xx", "bending_yy", "bending_xy"), n)
        tau = _column_stack(recovered, ("shear_xz", "shear_yz"), n)
        surfaces: Dict[str, np.ndarray] = {}
        missing_surface = False
        for surface in _SURFACES:
            tensor = np.zeros((n, 3, 3), dtype=float)
            for label, i, j in _SURFACE_COMPONENTS:
                key = f"global_{label}_{surface}"
                if key not in recovered:
                    missing_surface = True
                    continue
                values = _numeric(recovered[key], key)
                if values.shape != (n,):
                    missing_surface = True
                    continue
                tensor[:, i, j] = values
                tensor[:, j, i] = values
            surfaces[surface] = tensor
        if sigma_m is None or sigma_b is None or tau is None:
            for family in FAMILIES:
                refuse(family, "no_reference", "native local family values are missing")
        elif missing_surface:
            for family in FAMILIES:
                refuse(family, "no_reference", "solver global surface tensors are missing")
        else:
            # Derived global families: each rotated once with its own frame.
            membrane_global = _rotate_tensors(stations, _in_plane_tensors(h * sigma_m))
            bending_global = _rotate_tensors(
                stations, _in_plane_tensors(sigma_b * (max(h * h, LEGACY_RECOVERY_FLOOR) / 6.0))
            )
            shear_global = _rotate_vectors(shear_frames, h * tau)
            # Shear as it appears in the surface tensors, in station components.
            if branch.shear_role == "centre":
                # warped Q4: centre-frame vector projected into each station's
                # tangent plane, exactly as the solver documents it (F3)
                stress_vector = _rotate_vectors(shear_frames, tau)
                surface_shear = np.zeros((n, 2), dtype=float)
                for index in range(n):
                    frame = stations[index]
                    tangent = stress_vector[index] - float(stress_vector[index] @ frame[:, 2]) * frame[:, 2]
                    surface_shear[index] = (float(tangent @ frame[:, 0]), float(tangent @ frame[:, 1]))
            else:
                surface_shear = tau
            if not _finite(*surfaces.values(), sigma_m, sigma_b, tau, stations, shear_frames):
                for family in FAMILIES:
                    refuse(family, "nonfinite", "surface tensors, native values or frames are not finite")
            else:
                # Rule (1): resolve the families from G^T S G.
                local = {s: np.einsum("nki,nkl,nlj->nij", stations, surfaces[s], stations) for s in _SURFACES}
                mean = 0.5 * (local["top"] + local["bot"])
                half_difference = 0.5 * (local["top"] - local["bot"])
                resolved = {
                    "membrane": np.column_stack((mean[:, 0, 0], mean[:, 1, 1], mean[:, 0, 1])),
                    "bending": np.column_stack(
                        (half_difference[:, 0, 0], half_difference[:, 1, 1], half_difference[:, 0, 1])
                    ),
                    "shear": np.column_stack((mean[:, 0, 2], mean[:, 1, 2])),
                }
                derived_stress = {"membrane": sigma_m, "bending": sigma_b, "shear": surface_shear}
                s_surf = _max_abs(*surfaces.values())
                conditioning = {family: 1.0 for family in FAMILIES}
                if branch.shear_role == "centre":
                    # Codex r5: the projection centre tangent plane -> station
                    # tangent plane can be ill conditioned on extreme warping
                    with np.errstate(invalid="ignore"):
                        maps = np.einsum("nki,nkj->nij", stations[:, :, :2], shear_frames[:, :, :2])
                        sigma = np.linalg.svd(maps, compute_uv=False)[:, -1] if _finite(maps) else np.array([np.nan])
                    conditioning["shear"] = float(np.min(sigma)) if _finite(sigma) else 0.0
                for family in FAMILIES:
                    reason, diag, detail = _certify(
                        derived_stress[family], resolved[family], ROUNDOFF * s_surf, conditioning[family]
                    )
                    diag["surface_scale"] = s_surf
                    if branch.shear_role == "centre" and family == "shear":
                        diag["projection_conditioning"] = conditioning["shear"]
                    diagnostics[family] = diag
                    if reason is not None:
                        refuse(family, reason, detail)
                # Rule (5): rebuild the full surface tensors and rotate back out.
                # NaN-safe: a nonfinite rebuild refuses every family (Codex r3).
                residuals = []
                rebuilt_finite = True
                for surface, sign in (("top", 1.0), ("bot", -1.0)):
                    in_plane = sigma_m + sign * sigma_b
                    rebuilt_local = _in_plane_tensors(in_plane)
                    rebuilt_local[:, 0, 2] = surface_shear[:, 0]
                    rebuilt_local[:, 2, 0] = surface_shear[:, 0]
                    rebuilt_local[:, 1, 2] = surface_shear[:, 1]
                    rebuilt_local[:, 2, 1] = surface_shear[:, 1]
                    with np.errstate(over="ignore", invalid="ignore"):
                        rebuilt = _rotate_tensors(stations, rebuilt_local)
                        difference = np.abs(rebuilt - surfaces[surface])
                    rebuilt_finite = rebuilt_finite and _finite(rebuilt, difference)
                    residuals.append(float(np.max(difference)))
                worst = float(np.max(residuals))
                bound = (RELATIVE_BOUND + ROUNDOFF) * s_surf
                for family in FAMILIES:
                    diagnostics[family]["out_of_plane_residual"] = worst
                    diagnostics[family]["out_of_plane_allowance"] = bound
                if not rebuilt_finite:
                    for family in FAMILIES:
                        refuse(family, "nonfinite", "rule (5) surface rebuild is not finite")
                elif not worst <= bound:
                    for family in FAMILIES:
                        refuse(
                            family,
                            "out_of_plane",
                            f"rule (5) surface residual {worst:.3e} > {bound:.3e}",
                        )
                family_arrays = {
                    "membrane": membrane_global,
                    "bending": bending_global,
                    "shear": shear_global,
                }
    else:
        # ---- direct-reference branches: the element emits each family tensor
        native = {
            "membrane": recovered.get("membrane_resultants"),
            "bending": recovered.get("bending_resultants"),
            "shear": recovered.get("transverse_shear_resultants"),
        }
        sign = branch.moment_shear_sign
        for family in FAMILIES:
            key = FAMILY_KEYS[family]
            if key not in recovered or native[family] is None:
                refuse(family, "no_reference", f"solver did not emit {key} or the native values")
                continue
            reference = _numeric(recovered[key], key)
            values = _numeric(native[family], f"native {family} values")
            expected_shape = (n, 3) if family == "shear" else (n, 3, 3)
            if reference.shape != expected_shape or values.shape != ((n, 2) if family == "shear" else (n, 3)):
                refuse(family, "no_reference", f"{key} has an unexpected shape")
                continue
            # each family against its own frame only (Codex r4)
            if not _finite(reference, values, shear_frames if family == "shear" else stations):
                refuse(family, "nonfinite", f"{key}, native values or frames are not finite")
                continue
            if family == "membrane":
                derived = _rotate_tensors(stations, _in_plane_tensors(values))
            elif family == "bending":
                derived = sign * _rotate_tensors(stations, _in_plane_tensors(values))
            else:
                derived = sign * _rotate_vectors(shear_frames, values)
            reason, diag, detail = _certify(derived, reference, 0.0)
            diagnostics[family] = diag
            if reason is not None:
                refuse(family, reason, detail)
            family_arrays[family] = reference

    # Rule (5) / tangency of the membrane and bending tensors with the normal.
    for family in ("membrane", "bending"):
        tensors = family_arrays.get(family)
        if tensors is None or not _finite(tensors, station_normals):
            continue
        ok, value = _tangency(station_normals, tensors, nonorthogonality)
        diagnostics.setdefault(family, {})["normal_component"] = value
        if not ok:
            refuse(family, "out_of_plane", f"|nu.T|={value:.3e} > 1e-13 s_T")

    # Derived resultants N = h sigma, M, Q can overflow while the stresses are
    # finite (Codex r1): a nonfinite family is never certified or returned.
    for family, tensors in family_arrays.items():
        if not _finite(tensors):
            refuse(family, "nonfinite", "derived family tensor is not finite")

    arrays: Dict[str, np.ndarray] = {}
    if _finite(branch.positions):
        arrays["recovery_station_positions"] = _frozen(branch.positions)
    else:
        refuse("positions", "nonfinite", "station positions are not finite")
    # Rule (6): every basis normal against independent geometry (after rules
    # (1)-(5), so their refusal reasons stay the first recorded ones).
    def normal_mismatch(frames: np.ndarray, expected: Optional[np.ndarray], bounds: Optional[np.ndarray]) -> str:
        if expected is None or expected.shape != frames[:, :, 2].shape:
            return "no geometric normal for this branch"
        limit = np.full(len(expected), GEOMETRY_BOUND) if bounds is None else np.asarray(bounds, dtype=float)
        with np.errstate(invalid="ignore"):
            difference = np.linalg.norm(frames[:, :, 2] - expected, axis=1)  # Euclidean (Codex r7)
        if not bool(np.all(difference <= limit)):
            worst = float(np.max(difference - limit))
            return f"rule (6) basis normal differs from the geometry by {worst:.3e} beyond its bound"
        return ""

    station_geometry = normal_mismatch(stations, branch.expected_station_normals, branch.station_normal_bounds)
    if station_geometry:
        for family in station_families:
            refuse(family, "normal_geometry", station_geometry)
    shear_geometry = normal_mismatch(shear_frames, branch.expected_shear_normals, branch.shear_normal_bounds)
    if shear_geometry:
        refuse("shear", "normal_geometry", shear_geometry)
    if branch.emitted_frame is not None:
        with np.errstate(invalid="ignore"):
            frame_difference = float(np.max(np.abs(stations - branch.emitted_frame[None])))
        if not frame_difference <= GEOMETRY_BOUND:
            for family in FAMILIES:
                refuse(family, "normal_geometry", f"frame differs from the emitted V2D frame by {frame_difference:.3e}")
    for family, normals in (("membrane", station_normals), ("bending", station_normals), ("shear", shear_normals)):
        ok = _finite(normals) and float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0))) <= ROUNDOFF
        if not ok:
            refuse(family, "residual", "tensor-basis normal is not unit length")

    def frame_certified(families: Tuple[str, ...]) -> bool:
        # A frame is certified when a family that uses it certified and no
        # family that uses it shows a mismatch (an unresolvable or missing
        # family is no evidence against the frame).  Codex r3: a refused
        # frame feeds no returned value.
        if not any(family not in refusals for family in families):
            return False
        return not any(
            refusals[family].reason in _FRAME_EVIDENCE
            for family in families
            if family in refusals
        )

    station_ok = frame_certified(station_families)
    shear_ok = frame_certified(station_families if branch.shear_role == "station" else ("shear",))
    if station_ok:
        arrays[NORMAL_KEYS["membrane"]] = _frozen(station_normals)
        arrays[NORMAL_KEYS["bending"]] = _frozen(station_normals)
    if shear_ok:
        arrays[NORMAL_KEYS["shear"]] = _frozen(shear_normals)
    for family in FAMILIES:
        if family in family_arrays and family not in refusals:
            arrays[FAMILY_KEYS[family]] = _frozen(family_arrays[family])
    if "shear" in family_arrays and station_ok and "shear" not in refusals:
        # nu . Q with the station (membrane/bending) normal: recorded, not required
        components = np.einsum("ni,ni->n", station_normals, family_arrays["shear"])
        if _finite(components):
            arrays["shear_normal_components"] = _frozen(components)
        else:
            refuse("shear", "nonfinite", "nu . Q is not finite")
            arrays.pop(FAMILY_KEYS["shear"], None)
    for key, value in branch.metadata.items():
        if not _finite(value):
            refuse("metadata", "nonfinite", f"{key} is not finite")
            continue
        arrays[key] = _frozen(value, dtype=int if key == "director_polarity" else float)

    certified = {family: family not in refusals for family in FAMILIES}
    return ShellAxesOutputs(
        branch=branch.name,
        arrays=arrays,
        refusals=refusals,
        certified=certified,
        diagnostics=diagnostics,
        conventions=dict(branch.conventions),
    )
