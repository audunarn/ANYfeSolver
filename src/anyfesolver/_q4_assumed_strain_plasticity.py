"""Assumed-strain layered plasticity for the qualified (E4-PL) Q4 shell.

Background
----------
The qualified Q4 shell is the inherited compatible von-Karman shell plus the
constant elastic delta ``Cr = K_qualified - K_compatible``.  With the
hybrid's assumed generalized strain ``E_hat`` at the 2x2 Gauss points (the
condensed E4 strain interpolation) the membrane/bending part of ``Cr`` is
exactly

    Cr_mb = sum_g w_g (E_hat_g' D E_hat_g - B_g' D B_g) = -sum_g w_g dE_g' D dE_g,

``dE = E_hat - B`` (the hybrid stress/strain orthogonality
``sum_g w_g E_hat_g' D dE_g = 0`` makes it negative semidefinite).  ``Cr``
relieves the parasitic in-plane-bending and bending strains of the compatible
element.  Evaluating plasticity at the *compatible* strain while keeping this
elastic relief (historical route) makes the element tangent indefinite once
the layers yield: the relief stays elastic while the compatible stiffness
collapses, which produced spurious plastic mechanisms and limit points.

Formulation
-----------
For a supported plastic element the layered return map is evaluated at the
assumed strain

    eps_hat = eps_c(u) + dE u     (membrane and curvature at each Gauss point),

``eps_c`` the compatible von-Karman strain.  The element force (without
``Cr u``, which every caller adds as before) is

    F = F_el(u) + dE_m' A eps_nl + B_nl' A dE_m u
        - sum_g w_g (B_eff + dE)' [N_p; M_p](eps_hat),

``F_el`` the inherited *elastic* kernel, ``A = h C``, ``N_p = sum_l w_l C
eps_p``, ``M_p = sum_l w_l z_l C eps_p``.  Together with ``Cr u`` this is the
force of the assumed-strain layered shell ``sum_g w_g (B_eff + dE)' N_hat``
(plus the unchanged shear/drilling/stabilisation parts), so the tangent is
the exact, symmetric

    K = sum_g w_g (B_eff + dE)' D_t (B_eff + dE) + K_geo(N_hat) + K_rest.

Stability: positive semidefiniteness of (qualified elastic Hessian - this
model's elastic Hessian) - here the remainder ``K_rest`` (hybrid transverse
shear plus stabilisation) - is a *sufficient* condition for the elastic relief
not to destabilise yielded layers.  It is not a necessary condition, and the
geometric (compressive) stiffness ``K_geo`` can destabilise the response
independently, as it physically should.

Elastic limits:

* elastic materials never enter this route (bit-identical);
* for a plastic-capable material before yield the force differs from the
  historical route only by the von-Karman cross terms
  ``dE_m' A eps_nl + B_nl' A dE_m u`` (zero for linear kinematics);
* the plastic state keeps its schema (``plastic_strain``, ``alpha``,
  ``layer_strain`` per Gauss point and layer); ``layer_strain`` now holds the
  assumed layer strain at which the return map was evaluated.

Scope: isotropic von Mises layered plasticity on planar facets without active
initial fields.  Every other plastic-capable qualified Q4 configuration
(Hill48/orthotropic plasticity, active initial fields, generalized sections
with a hardening curve, warped facets) is refused with
:class:`UnsupportedQ4PlasticRouteError`; there is no fallback to the
historical compatible-strain plastic route.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import elements as _elements
from .initial_field_state import state_has_active_initial_fields
from .plasticity import lobatto_layers, plane_stress_elastic_matrix, plane_stress_return_map

__all__ = [
    "ROUTE_ID",
    "UnsupportedQ4PlasticRouteError",
    "plastic_capable",
    "supported",
    "require_supported",
    "require_state_supported",
    "offset_operator",
    "kernel_batch",
]

ROUTE_ID = "Q4_ASSUMED_STRAIN_LAYERED_PLASTICITY_V1"

_SQRT3_INVERSE = 0.57735026918962576451
_GP_KERNEL = (
    (-_SQRT3_INVERSE, -_SQRT3_INVERSE),
    (_SQRT3_INVERSE, -_SQRT3_INVERSE),
    (-_SQRT3_INVERSE, _SQRT3_INVERSE),
    (_SQRT3_INVERSE, _SQRT3_INVERSE),
)
_PATCH_TOLERANCE = 1.0e-9
_CACHE_LIMIT = 65536
_offset_cache: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()


class UnsupportedQ4PlasticRouteError(ValueError):
    """A plastic qualified Q4 configuration has no stable plastic route."""


def _qualified_type() -> Any:
    from .e4_pl_element import QualifiedE4PLShellElement

    return QualifiedE4PLShellElement


def plastic_capable(element: Any, material: Any) -> bool:
    """Qualified Q4 whose layered kernel would run a plastic return map.

    Same criterion as the inherited kernel: a hardening curve, or an
    orthotropic material with a Hill48 yield criterion (perfectly plastic
    without a curve).
    """

    if type(element) is not _qualified_type():
        return False
    if getattr(material, "hardening_curve", None) is not None:
        return True
    return bool(
        getattr(material, "hill_yield", None) is not None
        and _elements._elastic_symmetry(material, None) == "orthotropic"
    )


def _unsupported_reason(element: Any, mesh: Any, material: Any) -> Optional[str]:
    if getattr(element, "shell_section", None) is not None:
        return "generalized shell section with a hardening curve"
    if _elements._elastic_symmetry(material, None) != "isotropic":
        return "orthotropic/Hill48 plasticity"
    if bool(getattr(element, "reduced_integration", False)):
        return "reduced integration"
    if int(getattr(element, "num_nodes", 0)) != 4:
        return "not a four-node element"
    from .e4_pl_element import equation7_frame

    _frame, _local, warpage = equation7_frame(element.get_node_coordinates(mesh))
    if warpage > float(element.planar_tolerance):
        return "warped facet"
    return None


def _refusal(element: Any, reason: str) -> UnsupportedQ4PlasticRouteError:
    return UnsupportedQ4PlasticRouteError(
        f"qualified Q4 element {getattr(element, 'element_id', '?')}: plastic analysis with "
        f"{reason} is not supported (the compatible-strain route with the qualified hybrid "
        "correction is unstable once layers yield); use formulation='legacy' for this "
        "element or an elastic material"
    )


def elastic_route(element: Any, mesh: Any, material: Any) -> bool:
    """SINGLE-LAW PROTOTYPE: curve-less isotropic qualified Q4 on the same kernel."""

    if type(element) is not _qualified_type() or plastic_capable(element, material):
        return False
    return _unsupported_reason(element, mesh, material) is None


def route(element: Any, mesh: Any, material: Any) -> bool:
    """Assumed-strain kernel applies (plastic, refusing out of scope, or elastic)."""

    return require_supported(element, mesh, material) or elastic_route(element, mesh, material)


def supported(element: Any, mesh: Any, material: Any) -> bool:
    """Element/material admits the assumed-strain route (state not considered)."""

    return plastic_capable(element, material) and _unsupported_reason(element, mesh, material) is None


def require_supported(element: Any, mesh: Any, material: Any) -> bool:
    """True for the assumed-strain route, False when not plastic-capable.

    A plastic-capable qualified Q4 outside the supported scope raises
    :class:`UnsupportedQ4PlasticRouteError` (fail closed).
    """

    if not plastic_capable(element, material):
        return False
    reason = _unsupported_reason(element, mesh, material)
    if reason is not None:
        raise _refusal(element, reason)
    return True


def require_state_supported(element: Any, state: Any) -> None:
    """Refuse an active initial field on the assumed-strain route.

    The same predicate (:func:`state_has_active_initial_fields`) selects the
    scalar initial-field override in the assembly plans, so scalar and batch
    evaluation agree: inactive (all-zero, provenance-free) fields are ignored
    by both and active fields are refused by both.
    """

    if state_has_active_initial_fields(state):
        raise _refusal(element, "an active initial stress/prestrain field")


def _voigt_rotation(R: np.ndarray) -> np.ndarray:
    """Engineering-Voigt strain map for in-plane axes ``x_k = R x_h``."""

    c = np.zeros((3, 3), dtype=float)
    pairs = ((0, 0), (1, 1), (0, 1))
    for j, (a, b) in enumerate(pairs):
        tensor = np.zeros((2, 2), dtype=float)
        if a == b:
            tensor[a, a] = 1.0
        else:
            tensor[0, 1] = tensor[1, 0] = 0.5
        rotated = R @ tensor @ R.T
        c[0, j] = rotated[0, 0]
        c[1, j] = rotated[1, 1]
        c[2, j] = 2.0 * rotated[0, 1]
    return c


def offset_operator(element: Any, mesh: Any, material: Any) -> np.ndarray:
    """Return ``dE`` (4 x 6 x 24, kernel-local dofs, kernel Gauss order).

    Rows are membrane strain and curvature in the kernel frame.  The operator
    is certified by the constant-strain patch test (``E_hat u = B u`` for
    every linear field) and fails closed otherwise.
    """

    from . import e4_pl_element as E4

    coordinates = np.asarray(element.get_node_coordinates(mesh), dtype=float)
    cache = element._nonlinear_geometry(mesh)
    T0 = np.asarray(cache["T0"], dtype=float)
    B_m = np.asarray(cache["B_m_all"], dtype=float)
    B_b = np.asarray(cache["B_b_all"], dtype=float)
    if B_m.shape != (4, 3, 24):
        raise ValueError("qualified Q4 assumed-strain route requires 2x2 Gauss points")
    frame, local, _warpage = E4.equation7_frame(coordinates)
    constitutive, _drill = element._constitutive_and_drill_stiffness(material, frame)
    key = (
        coordinates.tobytes(),
        np.asarray(constitutive, dtype=float).tobytes(),
        T0.tobytes(),
        B_m.tobytes(),
        B_b.tobytes(),
    )
    hit = _offset_cache.get(key)
    if hit is not None:
        _offset_cache.move_to_end(key)
        return hit
    c = E4._coefficients(local)
    stationary, coupling, _gram = E4._stationary_blocks(local, c, constitutive)
    solution, _diagnostics = E4._solve_stationary_system(stationary, coupling)
    alpha = solution[14:]
    transform = E4._global_transform(frame)
    # Hybrid frame axes are the columns of ``frame`` (u_glob = T u_hybrid);
    # kernel axes are the rows of the T0 node block (u_kernel = T0 u_glob).
    relative = T0[:3, :3] @ np.asarray(frame, dtype=float)
    if not (abs(relative[2, 2] - 1.0) <= 1.0e-10 and np.allclose(relative[:2, 2], 0.0, atol=1.0e-10)
            and np.allclose(relative[2, :2], 0.0, atol=1.0e-10)):
        raise ValueError("qualified Q4 hybrid and kernel frames do not share the normal")
    rotation = _voigt_rotation(relative[:2, :2])
    offsets = np.empty((4, 6, 24), dtype=float)
    for g, (r, s) in enumerate(_GP_KERNEL):
        _n_sigma, n_epsilon = E4._source_fields(c, float(r), float(s))
        e_hat_global = -(n_epsilon[:6] @ alpha) @ transform.T
        e_hat = e_hat_global @ T0.T
        e_hat[:3] = rotation @ e_hat[:3]
        e_hat[3:] = rotation @ e_hat[3:]
        offsets[g, :3] = e_hat[:3] - B_m[g]
        offsets[g, 3:] = e_hat[3:] - B_b[g]
    _certify_patch(offsets, B_m, B_b, T0, coordinates)
    offsets.setflags(write=False)
    _offset_cache[key] = offsets
    while len(_offset_cache) > _CACHE_LIMIT:
        _offset_cache.popitem(last=False)
    return offsets


def _certify_patch(offsets, B_m, B_b, T0, coordinates) -> None:
    """dE must vanish on every linear (constant-strain/curvature) field."""

    rotation = T0[:3, :3]
    centre = coordinates.mean(axis=0)
    local = (coordinates - centre) @ rotation.T
    scale_b = max(float(np.max(np.abs(B_m))), float(np.max(np.abs(B_b))), 1.0e-300)
    for component in (0, 1, 3, 4):
        for gradient in (0, 1):
            u_loc = np.zeros(24, dtype=float)
            for node in range(4):
                u_loc[6 * node + component] = local[node, gradient]
            residual = np.abs(np.einsum("gki,i->gk", offsets, u_loc))
            reference = np.abs(np.einsum("gki,i->gk", np.concatenate((B_m, B_b), axis=1), u_loc))
            bound = _PATCH_TOLERANCE * max(float(np.max(reference)), scale_b * float(np.max(np.abs(local))))
            if float(np.max(residual)) > bound:
                raise ValueError(
                    "qualified Q4 assumed-strain operator fails the constant-strain patch test"
                )


def layer_offsets(state: Any, n_gp: int, num_layers: int, thickness: float, C: np.ndarray) -> Optional[np.ndarray]:
    """Local layer strain offset of prescribed initial stress/prestrain.

    Same definition as the inherited kernel: ``C^-1 sigma_0 - eps_pre`` with
    ``sigma_0 = N0 + 2 z / h M0`` and ``eps_pre = e0 + z k0``.
    """

    names = (
        "initial_membrane_stress",
        "initial_bending_stress",
        "initial_membrane_prestrain",
        "initial_curvature_prestrain",
    )
    if state is None or not any(name in state for name in names):
        return None
    z, _w = lobatto_layers(int(num_layers), float(thickness))
    zero_rows = np.zeros((n_gp, 3), dtype=float)
    rows = [_elements._shell_field_rows(state.get(name, zero_rows), n_gp, name) for name in names]
    initial_stress = rows[0][:, None, :] + (2.0 * z[None, :, None] / float(thickness)) * rows[1][:, None, :]
    eigenstrain = rows[2][:, None, :] + z[None, :, None] * rows[3][:, None, :]
    offset = np.einsum("ij,glj->gli", np.linalg.inv(np.asarray(C, dtype=float)), initial_stress) - eigenstrain
    return offset.reshape(n_gp * int(num_layers), 3)

def _geometric(detw, Gw, N):
    Gx = Gw[:, :, 0]
    Gy = Gw[:, :, 1]
    return (
        np.einsum("egi,egj->eij", (detw * N[:, :, 0])[:, :, None] * Gx, Gx)
        + np.einsum("egi,egj->eij", (detw * N[:, :, 1])[:, :, None] * Gy, Gy)
        + np.einsum("egi,egj->eij", (detw * N[:, :, 2])[:, :, None] * Gx, Gy)
        + np.einsum("egi,egj->eij", (detw * N[:, :, 2])[:, :, None] * Gy, Gx)
    )


def _transpose_last(matrix: np.ndarray) -> np.ndarray:
    return np.swapaxes(matrix, -1, -2)


def kernel_batch(
    u_elem_batch: np.ndarray,
    T0: np.ndarray,
    B_m: np.ndarray,
    B_b: np.ndarray,
    B_d: np.ndarray,
    Gw: np.ndarray,
    detw: np.ndarray,
    B_s: np.ndarray,
    detw_shear: np.ndarray,
    E: float,
    nu: float,
    G_mod: float,
    h: float,
    drilling_stabilization: float,
    tangent: bool,
    curve: Any,
    committed_plastic: np.ndarray,
    committed_alpha: np.ndarray,
    num_layers: int,
    offsets: np.ndarray,
    *,
    return_geometric: bool = False,
    initial_layer_offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Assumed-strain replacement of ``batch_shell_nonlinear_response``.

    Same inputs/outputs (global element dofs, without the qualified ``Cr``
    term) plus the leading-axis ``offsets``.  The last output holds the
    compatible (kinematic) layer strain, the layer stress and, with
    ``return_geometric``, the stress-Hessian part of the tangent.
    """

    from .vectorized_nonlinear import batch_shell_nonlinear_response

    u_elem_batch = np.asarray(u_elem_batch, dtype=float)
    n_elem, n_dof = u_elem_batch.shape
    L = int(num_layers)
    n_gp = int(np.asarray(detw).shape[1])
    points = n_gp * L
    committed_plastic = np.asarray(committed_plastic, dtype=float).reshape(n_elem, points, 3)
    committed_alpha = np.asarray(committed_alpha, dtype=float).reshape(n_elem, points)
    F, K, _ep, _alpha, compatible_layers = batch_shell_nonlinear_response(
        u_elem_batch, T0, B_m, B_b, B_d, Gw, detw, B_s, detw_shear,
        float(E), float(nu), float(G_mod), float(h), float(drilling_stabilization),
        bool(tangent), None, committed_plastic, committed_alpha, L,
    )
    F = np.array(F, dtype=float, copy=True)
    K = np.array(K, dtype=float, copy=True) if tangent else None
    offsets = np.asarray(offsets, dtype=float)
    T0 = np.asarray(T0, dtype=float)
    B_m = np.asarray(B_m, dtype=float)
    B_b = np.asarray(B_b, dtype=float)
    Gw = np.asarray(Gw, dtype=float)
    detw = np.asarray(detw, dtype=float)
    u_loc = np.einsum("eij,ej->ei", T0, u_elem_batch)
    z, w = lobatto_layers(L, float(h))
    C = plane_stress_elastic_matrix(float(E), float(nu))
    A = float(h) * C

    theta = np.einsum("egki,ei->egk", Gw, u_loc)
    B_nl = np.empty((n_elem, n_gp, 3, n_dof), dtype=float)
    B_nl[:, :, 0, :] = theta[:, :, 0, None] * Gw[:, :, 0, :]
    B_nl[:, :, 1, :] = theta[:, :, 1, None] * Gw[:, :, 1, :]
    B_nl[:, :, 2, :] = theta[:, :, 0, None] * Gw[:, :, 1, :] + theta[:, :, 1, None] * Gw[:, :, 0, :]
    eps_nl = np.empty((n_elem, n_gp, 3), dtype=float)
    eps_nl[:, :, 0] = 0.5 * theta[:, :, 0] ** 2
    eps_nl[:, :, 1] = 0.5 * theta[:, :, 1] ** 2
    eps_nl[:, :, 2] = theta[:, :, 0] * theta[:, :, 1]
    d6 = np.einsum("egki,ei->egk", offsets, u_loc)
    N_offset = d6[:, :, :3] @ A.T
    dE_m = offsets[:, :, :3, :]
    F_loc = np.einsum("eg,egki,egk->ei", detw, dE_m, eps_nl @ A.T) + np.einsum(
        "eg,egki,egk->ei", detw, B_nl, N_offset
    )
    K_loc = None
    if tangent:
        cross = np.einsum("egki,egkj->eij", dE_m * detw[:, :, None, None], A @ B_nl)
        K_loc = cross + _transpose_last(cross) + _geometric(detw, Gw, N_offset)

    compatible = np.asarray(compatible_layers, dtype=float).reshape(n_elem, n_gp, L, 3)
    layers = (
        compatible
        + d6[:, :, None, :3]
        + z[None, None, :, None] * d6[:, :, None, 3:]
    )
    N_initial = np.zeros((n_elem, n_gp, 3), dtype=float)
    if initial_layer_offsets is not None:
        if curve is not None:
            raise ValueError("initial layer offsets are supported on the elastic route only")
        initial = np.asarray(initial_layer_offsets, dtype=float).reshape(n_elem, n_gp, L, 3)
        layers = layers + initial
        stress_0 = initial @ C.T
        N_initial = np.einsum("l,egli->egi", w, stress_0)
        M_initial = np.einsum("l,l,egli->egi", w, z, stress_0)
        B_all = np.concatenate((B_m + B_nl, B_b), axis=2) + offsets
        F_loc += np.einsum("egki,egk->ei", B_all * detw[:, :, None, None], np.concatenate((N_initial, M_initial), axis=2))
        if tangent:
            K_loc += _geometric(detw, Gw, N_initial)
    layer_strain = layers.reshape(n_elem * points, 3)
    if curve is None:
        # Single-law prototype: elastic material, no return map.
        sigma = layer_strain @ C.T
        C_t = None
        ep_new, alpha_new = committed_plastic.copy(), committed_alpha.copy()
        tangent_gate = False
    else:
        sigma, C_t, ep_new, alpha_new = plane_stress_return_map(
            layer_strain,
            committed_plastic.reshape(n_elem * points, 3),
            committed_alpha.reshape(n_elem * points),
            float(E),
            float(nu),
            curve,
            compute_tangent=bool(tangent),
        )
        tangent_gate = bool(tangent)
    ep_new = np.asarray(ep_new, dtype=float).reshape(n_elem, points, 3)
    alpha_new = np.asarray(alpha_new, dtype=float).reshape(n_elem, points)
    # The force correction is driven by the plastic strain and its derivative
    # by the algorithmic tangent.  Reversed flow can return a point to
    # eps_p == 0 on the plastic branch (C_t != C), so either criterion makes
    # an element active.
    active_rows = np.any(ep_new.reshape(n_elem, -1) != 0.0, axis=1)
    if tangent_gate:
        C_t = np.asarray(C_t, dtype=float).reshape(n_elem, n_gp, L, 3, 3)
        active_rows = active_rows | np.any((C_t != C).reshape(n_elem, -1), axis=1)
    active = np.flatnonzero(active_rows)
    N_p = np.zeros((n_elem, n_gp, 3), dtype=float)
    if active.size:
        stress_p = ep_new[active].reshape(active.size, n_gp, L, 3) @ C.T
        N_p[active] = np.einsum("l,egli->egi", w, stress_p)
        M_p = np.einsum("l,l,egli->egi", w, z, stress_p)
        gen = np.concatenate((N_p[active], M_p), axis=2)
        B_hat = np.concatenate((B_m[active] + B_nl[active], B_b[active]), axis=2) + offsets[active]
        F_loc[active] -= np.einsum("eg,egki,egk->ei", detw[active], B_hat, gen)
        if tangent:
            dC = C[None, None, None] - C_t[active]
            D0 = np.einsum("l,eglij->egij", w, dC)
            D1 = np.einsum("l,l,eglij->egij", w, z, dC)
            D2 = np.einsum("l,l,l,eglij->egij", w, z, z, dC)
            Dp = np.empty((active.size, n_gp, 6, 6), dtype=float)
            Dp[:, :, :3, :3] = D0
            Dp[:, :, :3, 3:] = D1
            Dp[:, :, 3:, :3] = D1
            Dp[:, :, 3:, 3:] = D2
            K_loc[active] -= np.einsum("egki,egkj->eij", B_hat * detw[active][:, :, None, None], Dp @ B_hat)
            K_loc[active] -= _geometric(detw[active], Gw[active], N_p[active])
    T0t = _transpose_last(T0)
    F += np.einsum("eij,ej->ei", T0t, F_loc)
    extras: Dict[str, Any] = {
        "kinematic_layer_strain": compatible.reshape(n_elem, points, 3),
        "layer_stress": np.asarray(sigma, dtype=float).reshape(n_elem, points, 3),
        "geometric": None,
    }
    if tangent:
        K_loc = 0.5 * (K_loc + _transpose_last(K_loc))
        K += T0t @ K_loc @ T0
        if return_geometric:
            membrane = (np.einsum("egki,ei->egk", B_m, u_loc) + eps_nl) @ A.T + N_offset + N_initial - N_p
            extras["geometric"] = T0t @ _geometric(detw, Gw, membrane) @ T0
    return F, K, ep_new, alpha_new, layer_strain, extras
