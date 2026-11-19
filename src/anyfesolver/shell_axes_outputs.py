"""Opt-in shell result-axes outputs, derived outside the pinned elements (D2-1, B2).

``derive_shell_axes_outputs`` takes the dict that
``element.compute_stresses(mesh, displacements, material, return_global=True)``
already returned and derives, per recovery station,

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
solver itself emitted (design D2 section 1, rules (1)-(5), revision 4).  A
cross-check failure never raises: the family is refused with
``AXES_FRAME_RULE_MISMATCH`` and its derived array is omitted.

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
  director convention (not re-multiplied).

Surface-only branches (legacy homogeneous, e4-pl warped homogeneous) resolve
each family from the solver's global surface tensors rotated into the
recomputed station frame; every other branch is a direct-reference branch whose
own global family tensors are the reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "AXES_FRAME_RULE_MISMATCH",
    "FAMILIES",
    "Refusal",
    "ShellAxesFrameRuleMismatch",
    "ShellAxesOutputs",
    "ShellAxesRefusal",
    "derive_shell_axes_outputs",
]

AXES_FRAME_RULE_MISMATCH = "AXES_FRAME_RULE_MISMATCH"
FAMILIES: Tuple[str, str, str] = ("membrane", "bending", "shear")
REFUSAL_REASONS = frozenset(
    {"residual", "nonfinite", "no_reference", "unresolvable_family", "out_of_plane"}
)

EPS = 2.0**-52
RELATIVE_BOUND = 1.0e-13  # rule (1) relative allowance and the tangency bound
ROUNDOFF = 64.0 * EPS  # rule (1) allowance of the emitted m +- b surfaces
RESOLUTION = 1.0e-9  # rule (3): a_f <= 1e-9 s_f
LEGACY_RECOVERY_FLOOR = 1.0e-12  # elements._SMALL in 6 M / max(h^2, _SMALL)

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
    values = [float(np.max(np.abs(a))) for a in arrays if np.size(a)]
    return max(values) if values else 0.0


def _finite(*arrays: Any) -> bool:
    return all(np.all(np.isfinite(np.asarray(a, dtype=float))) for a in arrays)


def _column_stack(recovered: Mapping[str, Any], keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    if not all(key in recovered for key in keys):
        return None
    return np.column_stack([np.asarray(recovered[key], dtype=float) for key in keys])


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
    return int(np.asarray(recovered[key]).shape[0])


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


def _resolve_branch(element, mesh, material, recovered, override) -> _Branch:
    from .e4_pl_element import QualifiedE4PLShellElement, equation7_frame
    from .e4_pl_s3_element import TRIANGLE_QUADRATURE, QualifiedE4PLS3ShellElement, triangle_frame
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    from .elements import ShellElement

    generalized = getattr(element, "shell_section", None) is not None
    coords = np.asarray(element.get_node_coordinates(mesh), dtype=float)
    n = _station_count(recovered, generalized)
    kind = type(element)

    def legacy(name: str) -> _Branch:
        points = np.asarray(element.gauss_points, dtype=float)
        if len(points) != n:
            raise ShellAxesRefusal(
                "AXES_STATIONS_NOT_QUALIFIED",
                f"{name}: recovery has {n} stations, the element rule has {len(points)} (custom natural_points?)",
            )
        stations = _apply_override(
            override, "station", _legacy_station_frames(element, coords, points, material, not generalized)
        )
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
            )
            if recovered.get("warped_direct") is not True:
                branch.consistency_problem = "recovery is not the warped direct path the module recomputed"
            return branch
        if "natural_point_coordinates" not in recovered:
            raise ShellAxesRefusal(
                "AXES_STATIONS_NOT_QUALIFIED",
                "planar e4-pl recovery without natural_point_coordinates",
            )
        points = np.asarray(recovered["natural_point_coordinates"], dtype=float)
        if points.shape != (n, 2):
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "planar e4-pl station table mismatch")
        frame = np.asarray(element._physical_director_context(equation7_frame(coords)[0])[0], dtype=float)
        stations = _apply_override(override, "station", np.repeat(frame[None], n, axis=0))
        return _Branch(
            name="e4pl-planar",
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
        if n != len(TRIANGLE_QUADRATURE):
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "qualified S3 station count mismatch")
        barycentric = np.asarray([(1.0 - r - s, r, s) for r, s, _w in TRIANGLE_QUADRATURE], dtype=float)
        frame = np.asarray(triangle_frame(coords, element.reference_normal)[0], dtype=float)
        stations = _apply_override(override, "station", np.repeat(frame[None], n, axis=0))
        polarity = int(element.director_polarity)
        return _Branch(
            name="s3",
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
        if "external_barycentric_coordinates" not in recovered:
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "V2D recovery without station coordinates")
        barycentric = np.asarray(recovered["external_barycentric_coordinates"], dtype=float)
        if barycentric.shape != (n, 3):
            raise ShellAxesRefusal("AXES_STATIONS_NOT_QUALIFIED", "V2D station table mismatch")
        frame = _v2d_frame(coords, np.asarray(element.reference_normal, dtype=float))
        stations = _apply_override(override, "station", np.repeat(frame[None], n, axis=0))
        return _Branch(
            name="v2d",
            surface_only=False,
            positions=barycentric @ coords,
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

    raise ShellAxesRefusal(
        "AXES_FORMULATION_NOT_QUALIFIED",
        f"no axes adapter for element type {kind.__name__}",
    )


def _offsets(element: Any) -> Dict[str, float]:
    return {
        "reference_surface_offset": float(element.reference_surface_offset),
        "section_origin_offset_from_reference": float(element.material_mid_surface_offset_from_reference),
        "physical_bottom_offset_from_reference": float(element.material_surface_offset_from_reference(-1.0)),
        "physical_top_offset_from_reference": float(element.material_surface_offset_from_reference(1.0)),
    }


def _v2d_frame(external: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """V2D/V2C facet frame, recomputed here (x along the first internal edge)."""

    cross = np.cross(external[1] - external[0], external[2] - external[0])
    order = (0, 1, 2) if float(cross @ normal) > 0.0 else (0, 2, 1)
    ordered = external[np.asarray(order, dtype=np.intp)]
    x_axis = ordered[1] - ordered[0]
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    return np.column_stack((x_axis, y_axis, normal))


# --------------------------------------------------------------- the check
def _certify(
    derived: np.ndarray,
    reference: np.ndarray,
    allowance_extra: float,
) -> Tuple[Optional[str], Dict[str, float], str]:
    """Rules (2)-(3) for one family; returns (reason or None, diagnostics, detail)."""

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
    if allowance > RESOLUTION * scale:
        return "unresolvable_family", diag, f"a_f={allowance:.3e} > 1e-9 s_f={RESOLUTION * scale:.3e}"
    if residual > allowance:
        return "residual", diag, f"r_f={residual:.3e} > a_f={allowance:.3e}"
    return None, diag, ""


def _tangency(normals: np.ndarray, tensors: np.ndarray) -> Tuple[bool, float]:
    scale = _max_abs(tensors)
    value = float(np.max(np.abs(np.einsum("ni,nij->nj", normals, tensors)))) if tensors.size else 0.0
    return value <= RELATIVE_BOUND * scale, value


def derive_shell_axes_outputs(
    element: Any,
    mesh: Any,
    displacements: Any,
    material: Any,
    recovered: Mapping[str, Any],
    *,
    _frame_override: Optional[Callable[[str, np.ndarray], np.ndarray]] = None,
) -> ShellAxesOutputs:
    """Derive and cross-check the axes outputs of one shell element.

    ``recovered`` must be the dict ``element.compute_stresses(mesh,
    displacements, material, return_global=True)`` returned; it is read only.
    ``displacements`` is accepted for the call contract (the module recomputes
    frames on the reference geometry the recovery used) and is not modified.
    ``_frame_override(role, frames)`` (role ``"station"`` with an (n,3,3)
    array or ``"centre"`` with a (3,3) array) is a private test hook that
    replaces the recomputed frames, so a wrong frame can be shown refused.
    """

    del displacements  # frames depend on the reference geometry only
    if not isinstance(recovered, Mapping):
        raise ShellAxesRefusal("AXES_RETURN_GLOBAL_REQUIRED", "recovery output is not a mapping")
    generalized = getattr(element, "shell_section", None) is not None
    surface_keys = [f"global_{label}_{surface}" for surface in _SURFACES for label, _i, _j in _SURFACE_COMPONENTS]
    has_families = any(key in recovered for key in FAMILY_KEYS.values())
    has_surfaces = any(key in recovered for key in surface_keys)
    if not has_families and not has_surfaces:
        raise ShellAxesRefusal(
            "AXES_RETURN_GLOBAL_REQUIRED",
            "recovery lacks the global outputs; call compute_stresses(..., return_global=True)",
        )

    branch = _resolve_branch(element, mesh, material, recovered, _frame_override)
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

    station_normals = stations[:, :, 2]
    shear_normals = shear_frames[:, :, 2]

    if branch.surface_only:
        # ---- surface-only branches: legacy homogeneous, e4-pl warped homogeneous
        h = float(element.thickness)
        sigma_m = _column_stack(recovered, ("membrane_xx", "membrane_yy", "membrane_xy"))
        sigma_b = _column_stack(recovered, ("bending_xx", "bending_yy", "bending_xy"))
        tau = _column_stack(recovered, ("shear_xz", "shear_yz"))
        surfaces: Dict[str, np.ndarray] = {}
        missing_surface = False
        for surface in _SURFACES:
            tensor = np.zeros((n, 3, 3), dtype=float)
            for label, i, j in _SURFACE_COMPONENTS:
                key = f"global_{label}_{surface}"
                if key not in recovered:
                    missing_surface = True
                    continue
                values = np.asarray(recovered[key], dtype=float)
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
                for family in FAMILIES:
                    reason, diag, detail = _certify(derived_stress[family], resolved[family], ROUNDOFF * s_surf)
                    diag["surface_scale"] = s_surf
                    diagnostics[family] = diag
                    if reason is not None:
                        refuse(family, reason, detail)
                # Rule (5): rebuild the full surface tensors and rotate back out.
                worst = 0.0
                for surface, sign in (("top", 1.0), ("bot", -1.0)):
                    in_plane = sigma_m + sign * sigma_b
                    rebuilt_local = _in_plane_tensors(in_plane)
                    rebuilt_local[:, 0, 2] = surface_shear[:, 0]
                    rebuilt_local[:, 2, 0] = surface_shear[:, 0]
                    rebuilt_local[:, 1, 2] = surface_shear[:, 1]
                    rebuilt_local[:, 2, 1] = surface_shear[:, 1]
                    rebuilt = _rotate_tensors(stations, rebuilt_local)
                    worst = max(worst, float(np.max(np.abs(rebuilt - surfaces[surface]))))
                bound = (RELATIVE_BOUND + ROUNDOFF) * s_surf
                for family in FAMILIES:
                    diagnostics[family]["out_of_plane_residual"] = worst
                    diagnostics[family]["out_of_plane_allowance"] = bound
                if worst > bound:
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
            reference = np.asarray(recovered[key], dtype=float)
            values = np.asarray(native[family], dtype=float)
            expected_shape = (n, 3) if family == "shear" else (n, 3, 3)
            if reference.shape != expected_shape or values.shape != ((n, 2) if family == "shear" else (n, 3)):
                refuse(family, "no_reference", f"{key} has an unexpected shape")
                continue
            if not _finite(reference, values, stations, shear_frames):
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
        ok, value = _tangency(station_normals, tensors)
        diagnostics.setdefault(family, {})["normal_component"] = value
        if not ok:
            refuse(family, "out_of_plane", f"|nu.T|={value:.3e} > 1e-13 s_T")

    arrays: Dict[str, np.ndarray] = {}
    if _finite(branch.positions):
        arrays["recovery_station_positions"] = _frozen(branch.positions)
    normals_ok = {}
    for family, normals in (("membrane", station_normals), ("bending", station_normals), ("shear", shear_normals)):
        ok = _finite(normals) and float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0))) <= ROUNDOFF
        normals_ok[family] = ok
        if ok:
            arrays[NORMAL_KEYS[family]] = _frozen(normals)
        else:
            refuse(family, "residual", "tensor-basis normal is not unit length")
    for family in FAMILIES:
        if family in family_arrays and family not in refusals:
            arrays[FAMILY_KEYS[family]] = _frozen(family_arrays[family])
    if "shear" in family_arrays and normals_ok["membrane"] and "shear" not in refusals:
        # nu . Q with the station (membrane/bending) normal: recorded, not required
        arrays["shear_normal_components"] = _frozen(
            np.einsum("ni,ni->n", station_normals, family_arrays["shear"])
        )
    for key, value in branch.metadata.items():
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
