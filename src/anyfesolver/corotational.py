"""Element-independent corotational kinematics for large rigid rotations.

This module implements the corotational (CR) formulation for the nonlinear
static solver.  Each element's rigid-body rotation is extracted from the
current deformed geometry, the nodal displacements are pulled back to the
reference configuration (removing the rigid motion), the element's own
nonlinear local response (layered J2 shells, fiber beams, local von Karman
coupling) acts on the small deformational part, and the resulting forces are
rotated forward to the current configuration:

    u_d,i (translation) = R_rig^T (x_i - x_c) - (X_i - X_c)
    u_d,i (rotation)    = rotvec(R_rig^T exp(skew(theta_i)))
    f_global            = E f_local(u_d),  E = blockdiag(R_rig, R_rig, ...)
    K_tangent           = E k_local E^T                         (rotated mode)
                        = d(E f_local(u_d)) / d u                (consistent mode)

Scope and validity:

- small strains and small *deformational* rotations per element; rigid
  rotations of any magnitude (verified by rigid-rotation invariance);
- shells extract the rigid rotation from the element midsurface frame at the
  element center; beams from axis alignment plus mean axial twist;
- the deformational displacements are routed through the elements' own
  nonlinear local responses, so layered shell J2 plasticity, beam fiber
  plasticity and the local von Karman coupling are active in the corotated
  frame (plastic state is objective under rigid rotation); fracture/erosion
  remains unsupported in corotational mode;
- ``tangent_mode="consistent"`` applies the full chain rule through the
  deformational pull-back, nodal rotation coordinates, extracted element
  frame, and rotate-forward force map.  The exact Jacobian is generally
  nonsymmetric; frame sensitivity is evaluated by centered differences while
  the remaining derivatives are analytical.  ``"rotated"`` retains the
  inexpensive historical ``E k_local E.T`` approximation;
- the shell pull-back uses displacement differences to reduce cancellation
  at small loads. Beam pull-back and geometric-frame extraction can still
  impose a roundoff floor; demanding convergence below it can stall an
  increment;
- admitted nonlinear beam-shell attachments rotate their eccentricity through
  the rotating MPC map and its connected derivatives; ordinary linear MPC
  constraints retain their linear-coordinate meaning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .fe_core import FEModel

_SMALL = 1.0e-12


def resolve_corotational_tangent_mode(
    kinematics: str,
    tangent_mode: str = "auto",
    *,
    follower_pressure: bool = False,
) -> str:
    """Validate and resolve the nonlinear solver's corotational tangent mode.

    ``"auto"`` preserves the lower-cost rotated tangent for ordinary
    corotational solves and selects the consistent tangent when follower
    pressure makes the complete, generally nonsymmetric equilibrium Jacobian
    mandatory.  Explicit ``"consistent"`` remains available for demanding
    large-rotation Newton solves.
    """

    kinematics_name = str(kinematics).strip().lower()
    requested = str(tangent_mode).strip().lower()
    if requested not in {"auto", "rotated", "consistent", "work_conjugate"}:
        raise ValueError(
            "corotational_tangent must be 'auto', 'rotated', 'consistent', or 'work_conjugate'"
        )
    if kinematics_name != "corotational":
        if requested != "auto":
            raise ValueError(
                "corotational_tangent is only applicable when "
                "kinematics='corotational'"
            )
        return "not_applicable"
    if requested == "auto":
        return "consistent" if follower_pressure else "rotated"
    return requested


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def rotation_matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map: rotation vector -> rotation matrix."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    a2 = float(np.dot(vector, vector))
    angle = float(np.sqrt(a2))
    K = _skew(vector)
    if angle < 1.0e-2:
        c1 = 1.0 - a2 * (1.0 / 6.0 - a2 * (1.0 / 120.0 - a2 / 5040.0))
        c2 = 0.5 - a2 * (1.0 / 24.0 - a2 * (1.0 / 720.0 - a2 / 40320.0))
        return np.eye(3) + c1 * K + c2 * (K @ K)
    axis = vector / angle
    K_axis = _skew(axis)
    half_sin = np.sin(0.5 * angle)
    return np.eye(3) + np.sin(angle) * K_axis + (2.0 * half_sin * half_sin) * (K_axis @ K_axis)


def rotation_vector_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Logarithmic map: rotation matrix -> rotation vector (robust near 0 and pi)."""
    R = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(R))
    v = 0.5 * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    sin_angle = float(np.linalg.norm(v))
    cos_angle = min(max(0.5 * (trace - 1.0), -1.0), 1.0)
    angle = float(np.arctan2(sin_angle, cos_angle))
    if angle < 1.0e-4:
        s2 = sin_angle * sin_angle
        return (1.0 + s2 * (1.0 / 6.0 + 0.075 * s2)) * v
    if angle > np.pi - 1.0e-6:
        # near pi: extract axis from the symmetric part
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # fix signs from off-diagonal terms using the largest component
        k = int(np.argmax(axis))
        if axis[k] > _SMALL:
            for j in range(3):
                if j != k:
                    axis[j] = A[j, k] / axis[k]
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm > _SMALL else np.array([1.0, 0.0, 0.0])
        return angle * axis
    return (angle / sin_angle) * v


def _minimal_rotation(from_direction: np.ndarray, to_direction: np.ndarray) -> np.ndarray:
    """Smallest rotation mapping one unit vector onto another."""
    a = np.asarray(from_direction, dtype=float).reshape(3)
    b = np.asarray(to_direction, dtype=float).reshape(3)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm > _SMALL:
        a = a / a_norm
    if b_norm > _SMALL:
        b = b / b_norm
    cross = np.cross(a, b)
    dot = float(a @ b)
    if dot < -1.0 + 1.0e-12:
        # antiparallel: rotate pi about any axis orthogonal to a
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, seed)
        axis = axis / max(float(np.linalg.norm(axis)), _SMALL)
        return rotation_matrix_from_vector(np.pi * axis)
    K = _skew(cross)
    return np.eye(3) + K + (K @ K) / (1.0 + dot)


def _shell_center_frame(element: Any, coords: np.ndarray) -> np.ndarray:
    """Element midsurface frame (columns = local axes) at the element center."""
    native_frame = getattr(element, "corotational_reference_frame", None)
    if callable(native_frame):
        frame = np.asarray(native_frame(coords), dtype=float)
        if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
            raise ValueError(
                "formulation-native corotational frame must be finite 3x3"
            )
        orthogonality = frame.T @ frame
        if not np.allclose(
            orthogonality, np.eye(3), rtol=0.0, atol=64.0 * np.finfo(float).eps
        ) or float(np.linalg.det(frame)) <= 0.0:
            raise ValueError(
                "formulation-native corotational frame must be proper orthonormal"
            )
        return frame
    if element.num_nodes in (3, 6):
        xi, eta = 1.0 / 3.0, 1.0 / 3.0
    else:
        xi, eta = 0.0, 0.0
    _N, dN_dxi, dN_deta = element.compute_shape_functions(xi, eta)
    R, _dN_dx, _dN_dy, _det_j = element._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
    return np.asarray(R, dtype=float)


def _beam_rigid_rotation(reference: np.ndarray, deformed: np.ndarray, node_rotations: np.ndarray) -> np.ndarray:
    """SO(3)-equivariant beam rigid rotation: convected axis alignment and mean axial twist."""
    t_ref = reference[-1] - reference[0]
    t_def = deformed[-1] - deformed[0]
    t_ref = t_ref / max(float(np.linalg.norm(t_ref)), _SMALL)
    t_def = t_def / max(float(np.linalg.norm(t_def)), _SMALL)
    if len(node_rotations) == 0:
        return _minimal_rotation(t_ref, t_def)
    aligned_frames = []
    for theta in node_rotations:
        R_node = rotation_matrix_from_vector(theta)
        R_tilt = _minimal_rotation(R_node @ t_ref, t_def)
        aligned_frames.append(R_tilt @ R_node)
    R0 = aligned_frames[0]
    twists = [0.0]
    for R_aligned in aligned_frames[1:]:
        M = R0.T @ R_aligned
        sin_psi = 0.5 * (
            (M[2, 1] - M[1, 2]) * t_ref[0]
            + (M[0, 2] - M[2, 0]) * t_ref[1]
            + (M[1, 0] - M[0, 1]) * t_ref[2]
        )
        cos_psi = 0.5 * (float(np.trace(M)) - float(t_ref @ M @ t_ref))
        twists.append(float(np.arctan2(sin_psi, cos_psi)))
    mean_twist = float(np.mean(twists))
    return R0 @ rotation_matrix_from_vector(mean_twist * t_ref)


def _element_category(element: Any) -> Optional[str]:
    from .elements import BeamElement, ShellElement

    if isinstance(element, ShellElement):
        return "shell"
    if isinstance(element, BeamElement):
        return "beam"
    return None


class _CorotationalReference:
    """Reference-only geometry and frame data per element."""

    __slots__ = ("coordinates", "centroid", "frame", "category", "center_dN")

    def __init__(self, element: Any, model: "FEModel", *, centered_shell: bool = False):
        coordinates = np.asarray(element.get_node_coordinates(model.mesh), dtype=float).copy()
        self.category = _element_category(element)
        if centered_shell and self.category == "shell":
            coordinates -= coordinates.mean(axis=0)
        self.coordinates = coordinates
        self.centroid = coordinates.mean(axis=0)
        self.center_dN: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if self.category == "shell":
            self.frame = _shell_center_frame(element, self.coordinates)
            if not callable(getattr(element, "corotational_reference_frame", None)):
                xi, eta = (
                    (1.0 / 3.0, 1.0 / 3.0)
                    if element.num_nodes in (3, 6)
                    else (0.0, 0.0)
                )
                _N, dN_dxi, dN_deta = element.compute_shape_functions(xi, eta)
                self.center_dN = (
                    np.asarray(dN_dxi, dtype=float),
                    np.asarray(dN_deta, dtype=float),
                )
        else:
            self.frame = None


def _corotational_cache(
    model: "FEModel", *, centered_shell: bool = False
) -> Dict[int, _CorotationalReference]:
    mesh = model.mesh
    revisions = mesh.revision_signature()
    signature = (
        int(revisions.get("topology", 0)),
        int(revisions.get("geometry", 0)),
    )
    cache_name = (
        "_corotational_centered_shell_cache" if centered_shell else "_corotational_cache"
    )
    cached = getattr(mesh, cache_name, None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    cache: Dict[int, _CorotationalReference] = {}
    for element_id, element in mesh.elements.items():
        if _element_category(element) is not None:
            cache[int(element_id)] = _CorotationalReference(
                element, model, centered_shell=centered_shell)
    setattr(mesh, cache_name, (signature, cache))
    return cache


def _centered_nodal_vectors(values: np.ndarray) -> np.ndarray:
    """Remove a common translation without subtracting an absolute centroid.

    For a two-node beam the centered pair is exactly plus/minus half its
    difference. This retains small displacement differences independently of
    the reference-coordinate origin. Neither input representation is mutated.
    """
    if len(values) == 2:
        half_difference = 0.5 * (values[1] - values[0])
        return np.array([-half_difference, half_difference])
    relative = values - values[0]
    return relative - relative.mean(axis=0)


def _corotational_deformation_state(
    reference: _CorotationalReference,
    element: Any,
    u_element: np.ndarray,
    *,
    stable_shell_rotation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deformational DOFs, rigid rotation, and current coordinates.

    Keeping this pull-back in one place ensures that constitutive recovery uses
    exactly the same objective element state as nonlinear force assembly. Beam
    current geometry is represented up to its irrelevant common translation;
    the frame, B/G factors and geometric stencils all use this same local gauge.
    """

    num_nodes = int(element.num_nodes)
    u = np.asarray(u_element, dtype=float).reshape(num_nodes, 6)
    translations = u[:, :3]
    rotations = u[:, 3:]
    if reference.category == "beam":
        reference_arm = _centered_nodal_vectors(reference.coordinates)
        translation_offset = _centered_nodal_vectors(translations)
        deformed = reference_arm + translation_offset
    else:
        deformed = reference.coordinates + translations
    if reference.category == "shell":
        R_rig = _shell_center_frame(element, deformed) @ reference.frame.T
    else:
        R_rig = _beam_rigid_rotation(reference.coordinates, deformed, rotations)

    u_d = np.zeros((num_nodes, 6), dtype=float)
    if reference.category == "shell":
        # The equivalent difference of order-element-size coordinates loses
        # small shell deformation when the applied load is small.  Form the
        # translational pull-back from displacement differences instead.
        translation_offset = translations - translations.mean(axis=0)
        reference_arm = reference.coordinates - reference.centroid
        rotation_difference = (_stable_shell_rotation_difference(R_rig)
                               if stable_shell_rotation else R_rig.T - np.eye(3))
    else:
        # R.T@(X+u-centroid(X+u))-(X-centroid(X)) is the
        # algebraically identical expression below. Its absolute-coordinate
        # subtraction erases axial deformation at small loads, leaving a force
        # floor that Newton corrections below a coordinate ulp cannot remove.
        rotation_difference = _stable_shell_rotation_difference(R_rig)
    for node in range(num_nodes):
        if reference.category == "shell":
            u_d[node, :3] = (
                R_rig.T @ translation_offset[node]
                + rotation_difference @ reference_arm[node]
            )
        else:
            u_d[node, :3] = (
                R_rig.T @ translation_offset[node]
                + rotation_difference @ reference_arm[node]
            )
        R_node = rotation_matrix_from_vector(rotations[node])
        u_d[node, 3:] = rotation_vector_from_matrix(R_rig.T @ R_node)
    return u_d.reshape(-1), R_rig, deformed


def _stable_shell_rotation_difference(rotation: np.ndarray) -> np.ndarray:
    """Form R.T-I without subtracting nearly equal diagonal entries.

    For a proper rotation A=(R-R.T)/2=sin(theta)*hat(axis), so
    R-I=A+A@A/(1+cos(theta)). The skew entries retain small rotations and
    their quadratic strain contribution. Away from identity use the historical
    subtraction, avoiding the near-pi denominator. Callers already validate the
    shell frame; this is not a projection/admission rule for arbitrary matrices.
    """
    cosine = .5 * (float(np.trace(rotation)) - 1.)
    if (not np.isfinite(rotation).all() or cosine <= .5 or
            not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.,
                            atol=64. * np.finfo(float).eps) or
            float(np.linalg.det(rotation)) <= 0.):
        return rotation.T - np.eye(3)
    skew = .5 * (rotation - rotation.T)
    return (skew + (skew @ skew) / (1. + cosine)).T


def _dense_block_rotation(rotation: np.ndarray, num_nodes: int) -> np.ndarray:
    """Dense block rotation retained only by the consistent tangent oracle."""

    E = np.zeros((num_nodes * 6, num_nodes * 6), dtype=float)
    for node in range(num_nodes):
        E[node * 6 : node * 6 + 3, node * 6 : node * 6 + 3] = rotation
        E[node * 6 + 3 : node * 6 + 6, node * 6 + 3 : node * 6 + 6] = rotation
    return E


def corotational_element_response(
    model: "FEModel",
    element_id: int,
    element: Any,
    u_element: np.ndarray,
    tangent: bool,
    committed_state: Optional[Any] = None,
    num_layers: int = 5,
    tangent_mode: str = "rotated",
    centered_shell_reference: bool = False,
    stable_shell_rotation: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
    """Corotational internal force, tangent and trial state for one element.

    The deformational displacement ``u_d`` is a small-deformation field on the
    reference geometry, so it is routed through the element's own nonlinear
    local response — layered J2 plasticity for shells and fiber sections for
    beams — and the resulting force/tangent are rotated forward with the rigid
    frame.  Plastic state therefore lives in the corotated frame and is
    objective under arbitrary rigid rotation.

    Returns ``(None, None, None)`` for element types outside the corotational
    scope so the caller can fall back to the element's own response.

    ``stable_shell_rotation`` is used by the experimental mixed point owner's
    checkpoint-bound numerical policy. Ordinary and legacy callers retain the
    historical arithmetic unless they explicitly request it.
    """
    if bool(getattr(element, "formulation_native_total_lagrangian", False)):
        raise ValueError(
            "qualified S3 uses formulation-native total-Lagrangian mechanics "
            "and cannot be evaluated through the generic corotational wrapper"
        )
    tangent_mode = str(tangent_mode).strip().lower()
    if tangent_mode not in {"rotated", "consistent", "work_conjugate"}:
        raise ValueError(
            "corotational tangent_mode must be 'rotated', 'consistent', or 'work_conjugate'"
        )

    if type(centered_shell_reference) is not bool:
        raise ValueError("centered shell reference flag must be boolean")
    if type(stable_shell_rotation) is not bool:
        raise ValueError("stable shell rotation flag must be boolean")
    if tangent_mode == "work_conjugate":
        return corotational_work_conjugate_response(
            model,
            element_id,
            element,
            u_element,
            tangent=tangent,
            committed_state=committed_state,
            num_layers=num_layers,
            centered_shell_reference=centered_shell_reference,
            stable_shell_rotation=stable_shell_rotation,
        )
    reference = _corotational_cache(
        model, centered_shell=centered_shell_reference).get(int(element_id))
    if reference is None or reference.category is None:
        return None, None, None
    num_nodes = int(element.num_nodes)
    u = np.asarray(u_element, dtype=float).reshape(num_nodes, 6)
    rotations = u[:, 3:]
    u_d, R_rig, deformed = _corotational_deformation_state(
        reference,
        element,
        u_element,
        stable_shell_rotation=stable_shell_rotation,
    )

    material = model.get_material(element.material_name)
    f_ref, k_ref, trial_state = element.compute_nonlinear_response(
        model.mesh, material, u_d, committed_state, int(num_layers), tangent
    )
    f_ref = np.asarray(f_ref, dtype=float).reshape(-1)
    if not tangent:
        from .corotational_performance import rotate_corotational_force_blocks

        f_global = rotate_corotational_force_blocks(f_ref, R_rig)
        return f_global, None, trial_state
    k_ref = np.asarray(k_ref, dtype=float)
    if tangent_mode == "consistent":
        from .corotational_performance import (
            note_dense_consistent_rotation,
            rotate_corotational_force_blocks,
        )

        f_global = rotate_corotational_force_blocks(f_ref, R_rig)
        E = _dense_block_rotation(R_rig, num_nodes)
        note_dense_consistent_rotation()
        k_global = _consistent_corotational_tangent(
            reference,
            element,
            deformed,
            rotations,
            R_rig,
            E,
            f_global,
            k_ref,
        )
    else:
        # The rotated local tangent remains the robust default for large load
        # steps and preserves the historical convergence behavior.
        from .corotational_performance import (
            rotate_corotational_force_tangent_blocks,
        )

        f_global, k_global = rotate_corotational_force_tangent_blocks(
            f_ref,
            k_ref,
            R_rig,
        )
    return f_global, k_global, trial_state


def corotational_element_response_components(
    model: "FEModel",
    element_id: int,
    element: Any,
    u_element: np.ndarray,
    committed_state: Optional[Any] = None,
    num_layers: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Return the symmetric material/geometric split for eigen workflows.

    This is the read-only energy-Hessian counterpart of
    :func:`corotational_element_response`.  It evaluates the same element
    response and exact frame sensitivity, then returns the symmetric parts of
    the constitutive/pull-back and force/frame contributions separately.
    Additive rotation coordinates can make the Newton Jacobian nonsymmetric;
    modal and buckling pencils require the symmetric energy Hessian instead.
    """

    if bool(getattr(element, "formulation_native_total_lagrangian", False)):
        raise ValueError(
            "formulation-native total-Lagrangian elements cannot use the "
            "element-independent corotational component route"
        )
    reference = _corotational_cache(model).get(int(element_id))
    if reference is None or reference.category is None:
        raise ValueError("element is outside the corotational shell/beam scope")
    num_nodes = int(element.num_nodes)
    u = np.asarray(u_element, dtype=float).reshape(num_nodes, 6)
    rotations = u[:, 3:]
    u_d, rigid_rotation, deformed = _corotational_deformation_state(
        reference,
        element,
        u_element,
    )
    material = model.get_material(element.material_name)
    force_reference, tangent_reference, trial_state = element.compute_nonlinear_response(
        model.mesh,
        material,
        u_d,
        committed_state,
        int(num_layers),
        True,
    )
    force_reference = np.asarray(force_reference, dtype=float).reshape(-1)
    tangent_reference = np.asarray(tangent_reference, dtype=float)
    from .corotational_performance import rotate_corotational_force_blocks

    force_global = rotate_corotational_force_blocks(
        force_reference,
        rigid_rotation,
    )
    block_rotation = _dense_block_rotation(rigid_rotation, num_nodes)
    material_tangent, geometric_tangent = (
        _consistent_corotational_tangent_components(
            reference,
            element,
            deformed,
            rotations,
            rigid_rotation,
            block_rotation,
            force_global,
            tangent_reference,
        )
    )
    material_symmetric = 0.5 * (material_tangent + material_tangent.T)
    geometric_symmetric = 0.5 * (geometric_tangent + geometric_tangent.T)
    return (
        np.asarray(force_global, dtype=np.float64),
        np.asarray(material_symmetric, dtype=np.float64),
        np.asarray(geometric_symmetric, dtype=np.float64),
        trial_state,
    )


def _rotation_right_jacobian(vector: np.ndarray) -> np.ndarray:
    """Right Jacobian of the exponential map: d exp(theta + d) ~ exp(theta) skew(Jr d)."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    a2 = float(np.dot(vector, vector))
    angle = float(np.sqrt(a2))
    K = _skew(vector)
    if angle < 1.0e-2:
        c1 = 0.5 - a2 * (1.0 / 24.0 - a2 * (1.0 / 720.0 - a2 / 40320.0))
        c2 = (1.0 / 6.0) - a2 * (1.0 / 120.0 - a2 * (1.0 / 5040.0 - a2 / 362880.0))
        return np.eye(3) - c1 * K + c2 * (K @ K)
    half_sin = np.sin(0.5 * angle)
    return (
        np.eye(3)
        - (2.0 * half_sin * half_sin) / a2 * K
        + (angle - np.sin(angle)) / (a2 * angle) * (K @ K)
    )


def _rotation_right_jacobian_inverse(vector: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian for additive rotation-vector increments."""

    vector = np.asarray(vector, dtype=float).reshape(3)
    a2 = float(np.dot(vector, vector))
    angle = float(np.sqrt(a2))
    K = _skew(vector)
    if angle < 1.0e-2:
        c2 = (1.0 / 12.0) + a2 * (
            1.0 / 720.0 + a2 * (1.0 / 30240.0 + a2 / 1209600.0)
        )
        return np.eye(3) + 0.5 * K + c2 * (K @ K)
    coefficient = 1.0 / a2 - 0.5 / angle * (
        np.cos(0.5 * angle) / np.sin(0.5 * angle)
    )
    return np.eye(3) + 0.5 * K + coefficient * (K @ K)


def _rotation_left_jacobian_inverse(vector: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian for an SO(3) logarithmic-map coordinate."""

    vector = np.asarray(vector, dtype=float).reshape(3)
    a2 = float(np.dot(vector, vector))
    angle = float(np.sqrt(a2))
    K = _skew(vector)
    if angle < 1.0e-2:
        c2 = (1.0 / 12.0) + a2 * (
            1.0 / 720.0 + a2 * (1.0 / 30240.0 + a2 / 1209600.0)
        )
        return np.eye(3) - 0.5 * K + c2 * (K @ K)
    coefficient = 1.0 / a2 - 0.5 / angle * (
        np.cos(0.5 * angle) / np.sin(0.5 * angle)
    )
    return np.eye(3) - 0.5 * K + coefficient * (K @ K)


def _rigid_rotation_for_state(
    reference: "_CorotationalReference", element: Any, deformed: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    if reference.category == "shell":
        return _shell_center_frame(element, deformed) @ reference.frame.T
    return _beam_rigid_rotation(reference.coordinates, deformed, rotations)


def _rotation_sensitivity(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
) -> np.ndarray:
    """G = d(omega)/du (3 x 6n): rigid-rotation sensitivity.

    ``omega`` is the left-increment rotation vector of the extracted rigid
    rotation, ``R_rig(u + du) ~ exp(skew(G du)) R_rig(u)``.  Shell frames
    depend only on the nodal translations via the center surface tangents;
    beam frames additionally pick up twist from the nodal rotations.
    """
    num_nodes = deformed.shape[0]
    if reference.category == "shell" and reference.center_dN is not None:
        dN_dxi, dN_deta = reference.center_dN
        t_xi = dN_dxi @ deformed
        t_eta = dN_deta @ deformed
        R = R_rig @ reference.frame
        e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
        L1 = float(t_xi @ e1)
        alpha = float(t_eta @ e1)
        beta = float(t_eta @ e2)
        if (
            L1 > _SMALL
            and abs(beta) > _SMALL
            and abs(float(t_xi @ e3)) <= 1.0e-9 * L1
        ):
            J_loc_xi = np.vstack(
                [
                    (-alpha / (L1 * beta)) * e3,
                    (-1.0 / L1) * e3,
                    (1.0 / L1) * e2,
                ]
            )
            J_loc_eta = np.vstack(
                [(1.0 / beta) * e3, np.zeros(3), np.zeros(3)]
            )
            J_xi = R @ J_loc_xi
            J_eta = R @ J_loc_eta
            G = np.zeros((3, num_nodes * 6), dtype=float)
            for node in range(num_nodes):
                G[:, node * 6 : node * 6 + 3] = (
                    dN_dxi[node] * J_xi + dN_deta[node] * J_eta
                )
            trans_blocks = G.reshape(3, num_nodes, 6)[:, :, :3]
            trans_blocks -= np.mean(trans_blocks, axis=1, keepdims=True)
            return G

    G = np.zeros((3, num_nodes * 6), dtype=float)
    spread = float(np.max(np.linalg.norm(reference.coordinates - reference.centroid, axis=1)))
    step_translation = 1.0e-6 * max(spread, 1.0e-3)
    step_rotation = 1.0e-6
    include_rotations = reference.category == "beam"
    Rt = R_rig.T
    two_node_beam = include_rotations and num_nodes == 2
    for node in range(num_nodes):
        if two_node_beam and node == 1:
            G[:, 6:9] = -G[:, 0:3]
        else:
            for axis in range(3):
                perturbed = deformed.copy()
                perturbed[node, axis] += step_translation
                R_plus = _rigid_rotation_for_state(reference, element, perturbed, rotations)
                perturbed[node, axis] -= 2.0 * step_translation
                R_minus = _rigid_rotation_for_state(reference, element, perturbed, rotations)
                delta = rotation_vector_from_matrix(R_plus @ Rt) - rotation_vector_from_matrix(R_minus @ Rt)
                G[:, node * 6 + axis] = delta / (2.0 * step_translation)
        if include_rotations:
            for axis in range(3):
                perturbed_rotations = rotations.copy()
                perturbed_rotations[node, axis] += step_rotation
                R_plus = _rigid_rotation_for_state(reference, element, deformed, perturbed_rotations)
                perturbed_rotations[node, axis] -= 2.0 * step_rotation
                R_minus = _rigid_rotation_for_state(reference, element, deformed, perturbed_rotations)
                delta = rotation_vector_from_matrix(R_plus @ Rt) - rotation_vector_from_matrix(R_minus @ Rt)
                G[:, node * 6 + 3 + axis] = delta / (2.0 * step_rotation)
    trans_blocks = G.reshape(3, num_nodes, 6)[:, :, :3]
    trans_blocks -= np.mean(trans_blocks, axis=1, keepdims=True)
    return G


def _pullback_jacobian_factors(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(D, U, G)`` such that ``B(u) = du_d/du = D + U @ G``."""
    num_nodes = deformed.shape[0]
    n_dofs = num_nodes * 6
    Rt = R_rig.T
    centroid_def = deformed.mean(axis=0)

    D = np.zeros((n_dofs, n_dofs), dtype=float)
    U = np.zeros((n_dofs, 3), dtype=float)
    average = 1.0 / num_nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = (1.0 if i == j else 0.0) - average
            D[i * 6 : i * 6 + 3, j * 6 : j * 6 + 3] = weight * Rt
        R_node = rotation_matrix_from_vector(rotations[i])
        deformational_rotation = rotation_vector_from_matrix(Rt @ R_node)
        D[i * 6 + 3 : i * 6 + 6, i * 6 + 3 : i * 6 + 6] = (
            _rotation_right_jacobian_inverse(deformational_rotation)
            @ _rotation_right_jacobian(rotations[i])
        )
        U[i * 6 : i * 6 + 3, :] = Rt @ _skew(deformed[i] - centroid_def)
        U[i * 6 + 3 : i * 6 + 6, :] = (
            -_rotation_left_jacobian_inverse(deformational_rotation) @ Rt
        )

    G = _rotation_sensitivity(reference, element, deformed, rotations, R_rig)
    return D, U, G


def corotational_deformation_jacobian(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
) -> np.ndarray:
    """Exact deformation Jacobian ``B(u) = du_d / du = D(u) + U(u) G(u)``."""
    D, U, G = _pullback_jacobian_factors(
        reference, element, deformed, rotations, R_rig
    )
    return D + U @ G


def spatial_moment_from_rotation_covector(
    rotation_vector: np.ndarray,
    rotation_covector: np.ndarray,
) -> np.ndarray:
    """Convert an additive rotation-vector covector ``p_rot`` to spatial moment ``t``.

    Under a spatial spin increment ``delta_psi`` (``exp(skew(delta_psi)) R(theta)``),
    the additive rotation vector increments by ``delta_theta = J_l(theta)^{-1} delta_psi``.
    Virtual-work equivalence ``delta_theta^T p_rot = delta_psi^T t`` therefore gives
    ``t = J_l(theta)^{-T} p_rot``.
    """
    theta = np.asarray(rotation_vector, dtype=float).reshape(3)
    p_rot = np.asarray(rotation_covector, dtype=float).reshape(3)
    return _rotation_left_jacobian_inverse(theta).T @ p_rot


def spatial_force_and_moment_resultants(
    deformed_coords: np.ndarray,
    rotations: np.ndarray,
    generalized_force: np.ndarray,
    origin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return total spatial force and moment about ``origin`` in deformed geometry."""
    coords = np.asarray(deformed_coords, dtype=float).reshape(-1, 3)
    num_nodes = coords.shape[0]
    rots = np.asarray(rotations, dtype=float).reshape(num_nodes, 3)
    p = np.asarray(generalized_force, dtype=float).reshape(num_nodes, 6)
    ref_origin = (
         np.zeros(3, dtype=float)
        if origin is None
        else np.asarray(origin, dtype=float).reshape(3)
    )
    total_force = np.zeros(3, dtype=float)
    total_moment = np.zeros(3, dtype=float)
    for i in range(num_nodes):
        f_i = p[i, :3]
        t_i = spatial_moment_from_rotation_covector(rots[i], p[i, 3:])
        total_force += f_i
        total_moment += np.cross(coords[i] - ref_origin, f_i) + t_i
    return total_force, total_moment


def _corotational_pullback_force_precomputed(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_nodes: np.ndarray,
    Jr_nodes: np.ndarray,
    R_rig: np.ndarray,
    f_trans: np.ndarray,
    f_rot: np.ndarray,
) -> np.ndarray:
    """Evaluate ``B(u)^T f_ref`` directly using precomputed nodal rotation factors."""
    num_nodes = deformed.shape[0]
    Rt = R_rig.T
    f_trans_rot = f_trans @ Rt
    p_trans = f_trans_rot - np.mean(f_trans_rot, axis=0, keepdims=True)
    rel_def = deformed - np.mean(deformed, axis=0, keepdims=True)
    u_f = np.sum(np.cross(f_trans_rot, rel_def), axis=0)

    p_rot = np.empty((num_nodes, 3), dtype=float)
    for i in range(num_nodes):
        def_rot = rotation_vector_from_matrix(Rt @ R_nodes[i])
        D_rot = _rotation_right_jacobian_inverse(def_rot) @ Jr_nodes[i]
        U_rot = -_rotation_left_jacobian_inverse(def_rot) @ Rt
        f_rot_i = f_rot[i]
        p_rot[i] = D_rot.T @ f_rot_i
        u_f = u_f + U_rot.T @ f_rot_i

    G = _rotation_sensitivity(reference, element, deformed, rotations, R_rig)
    p = np.empty((num_nodes, 6), dtype=float)
    p[:, :3] = p_trans
    p[:, 3:] = p_rot
    return p.reshape(-1) + G.T @ u_f


def _fast_shell_rigid_rotation_from_tangents(
    t_xi: np.ndarray, t_eta: np.ndarray, ref_frame_T: np.ndarray
) -> np.ndarray:
    c = np.cross(t_xi, t_eta)
    e3 = c / float(np.linalg.norm(c))
    e1_raw = t_xi - float(t_xi @ e3) * e3
    e1 = e1_raw / float(np.linalg.norm(e1_raw))
    e2 = np.cross(e3, e1)
    return np.column_stack((e1, e2, e3)) @ ref_frame_T


def _scalar_work_conjugate_corotational_tangent(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    f_ref: np.ndarray,
    k_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(material_tangent, geometric_tangent)`` for work-conjugate pullback.

    With ``p(u) = B(u)^T f_ref(u_d(u))``, the exact Jacobian is:
        ``K(u) = B(u)^T K_ref(u_d) B(u) + sum_a f_ref,a d^2 u_d,a / du^2``.
    The second term is the exact Hessian of the smooth scalar
    ``phi(u) = f_ref^T u_d(u)`` at frozen ``f_ref``, so it is symmetric by
    equality of mixed partial derivatives.
    """
    num_nodes = deformed.shape[0]
    n_dofs = num_nodes * 6
    material_tangent = B.T @ k_ref @ B
    if float(np.linalg.norm(f_ref)) <= _SMALL:
        return material_tangent, np.zeros((n_dofs, n_dofs), dtype=float)

    spread = float(
        np.max(np.linalg.norm(reference.coordinates - reference.centroid, axis=1))
    )
    step_translation = 1.0e-5 * max(spread, 1.0e-3)
    step_rotation = 1.0e-5
    K_geom = np.zeros((n_dofs, n_dofs), dtype=float)
    is_shell = reference.category == "shell"
    two_node_beam = reference.category == "beam" and num_nodes == 2
    Rt = R_rig.T
    f_blocks = f_ref.reshape(num_nodes, 6)
    f_trans = f_blocks[:, :3]
    f_rot = f_blocks[:, 3:]
    R_nodes = np.array(
        [rotation_matrix_from_vector(rotations[i]) for i in range(num_nodes)],
        dtype=float,
    )
    Jr_nodes = np.array(
        [_rotation_right_jacobian(rotations[i]) for i in range(num_nodes)],
        dtype=float,
    )
    inv_2h_trans = 1.0 / (2.0 * step_translation)
    inv_2h_rot = 1.0 / (2.0 * step_rotation)

    use_fast_shell_frame = False
    if is_shell and reference.center_dN is not None:
        dN_dxi, dN_deta = reference.center_dN
        t_xi_base = dN_dxi @ deformed
        t_eta_base = dN_deta @ deformed
        R_base = R_rig @ reference.frame
        L1_base = float(t_xi_base @ R_base[:, 0])
        beta_base = float(t_eta_base @ R_base[:, 1])
        if (
            L1_base > _SMALL
            and abs(beta_base) > _SMALL
            and abs(float(t_xi_base @ R_base[:, 2])) <= 1.0e-9 * L1_base
        ):
            use_fast_shell_frame = True
            ref_frame_T = reference.frame.T

    for node in range(num_nodes):
        if two_node_beam and node == 1:
            K_geom[:, 6:9] = -K_geom[:, 0:3]
        else:
            for axis in range(3):
                col = node * 6 + axis
                def_plus = deformed.copy()
                def_plus[node, axis] += step_translation
                if use_fast_shell_frame:
                    t_xi_p = t_xi_base.copy()
                    t_eta_p = t_eta_base.copy()
                    t_xi_p[axis] += step_translation * dN_dxi[node]
                    t_eta_p[axis] += step_translation * dN_deta[node]
                    R_plus = _fast_shell_rigid_rotation_from_tangents(
                        t_xi_p, t_eta_p, ref_frame_T
                    )
                else:
                    R_plus = _rigid_rotation_for_state(
                        reference, element, def_plus, rotations
                    )
                p_plus = _corotational_pullback_force_precomputed(
                    reference,
                    element,
                    def_plus,
                    rotations,
                    R_nodes,
                    Jr_nodes,
                    R_plus,
                    f_trans,
                    f_rot,
                )

                def_minus = deformed.copy()
                def_minus[node, axis] -= step_translation
                if use_fast_shell_frame:
                    t_xi_m = t_xi_base.copy()
                    t_eta_m = t_eta_base.copy()
                    t_xi_m[axis] -= step_translation * dN_dxi[node]
                    t_eta_m[axis] -= step_translation * dN_deta[node]
                    R_minus = _fast_shell_rigid_rotation_from_tangents(
                        t_xi_m, t_eta_m, ref_frame_T
                    )
                else:
                    R_minus = _rigid_rotation_for_state(
                        reference, element, def_minus, rotations
                    )
                p_minus = _corotational_pullback_force_precomputed(
                    reference,
                    element,
                    def_minus,
                    rotations,
                    R_nodes,
                    Jr_nodes,
                    R_minus,
                    f_trans,
                    f_rot,
                )
                K_geom[:, col] = (p_plus - p_minus) * inv_2h_trans

        if is_shell:
            r0 = node * 6 + 3
            r1 = r0 + 3
            f_rot_node = f_rot[node]
            if float(np.linalg.norm(f_rot_node)) <= _SMALL:
                continue
            theta_base = rotations[node]
            for axis in range(3):
                col = r0 + axis
                theta_plus = theta_base.copy()
                theta_plus[axis] += step_rotation
                def_rot_plus = rotation_vector_from_matrix(
                    Rt @ rotation_matrix_from_vector(theta_plus)
                )
                D_rot_plus = _rotation_right_jacobian_inverse(
                    def_rot_plus
                ) @ _rotation_right_jacobian(theta_plus)
                U_rot_plus = -_rotation_left_jacobian_inverse(def_rot_plus) @ Rt

                theta_minus = theta_base.copy()
                theta_minus[axis] -= step_rotation
                def_rot_minus = rotation_vector_from_matrix(
                    Rt @ rotation_matrix_from_vector(theta_minus)
                )
                D_rot_minus = _rotation_right_jacobian_inverse(
                    def_rot_minus
                ) @ _rotation_right_jacobian(theta_minus)
                U_rot_minus = -_rotation_left_jacobian_inverse(def_rot_minus) @ Rt

                K_geom[:, col] = G.T @ (
                    ((U_rot_plus - U_rot_minus).T @ f_rot_node) * inv_2h_rot
                )
                K_geom[r0:r1, col] += (
                    (D_rot_plus - D_rot_minus).T @ f_rot_node
                ) * inv_2h_rot
        else:
            for axis in range(3):
                col = node * 6 + 3 + axis
                rot_plus = rotations.copy()
                rot_plus[node, axis] += step_rotation
                R_plus = _rigid_rotation_for_state(
                    reference, element, deformed, rot_plus
                )
                R_nodes_plus = R_nodes.copy()
                Jr_nodes_plus = Jr_nodes.copy()
                R_nodes_plus[node] = rotation_matrix_from_vector(rot_plus[node])
                Jr_nodes_plus[node] = _rotation_right_jacobian(rot_plus[node])
                p_plus = _corotational_pullback_force_precomputed(
                    reference,
                    element,
                    deformed,
                    rot_plus,
                    R_nodes_plus,
                    Jr_nodes_plus,
                    R_plus,
                    f_trans,
                    f_rot,
                )

                rot_minus = rotations.copy()
                rot_minus[node, axis] -= step_rotation
                R_minus = _rigid_rotation_for_state(
                    reference, element, deformed, rot_minus
                )
                R_nodes_minus = R_nodes.copy()
                Jr_nodes_minus = Jr_nodes.copy()
                R_nodes_minus[node] = rotation_matrix_from_vector(rot_minus[node])
                Jr_nodes_minus[node] = _rotation_right_jacobian(rot_minus[node])
                p_minus = _corotational_pullback_force_precomputed(
                    reference,
                    element,
                    deformed,
                    rot_minus,
                    R_nodes_minus,
                    Jr_nodes_minus,
                    R_minus,
                    f_trans,
                    f_rot,
                )
                K_geom[:, col] = (p_plus - p_minus) * inv_2h_rot

    K_geom = 0.5 * (K_geom + K_geom.T)
    return material_tangent, K_geom


def _work_conjugate_corotational_tangent(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    f_ref: np.ndarray,
    k_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the unchanged geometric stencils together in numerical batches."""
    from ._corotational_batch import work_conjugate_tangent

    return work_conjugate_tangent(
        reference, element, deformed, rotations, R_rig, G, B, f_ref, k_ref
    )


def corotational_work_conjugate_response(
    model: "FEModel",
    element_id: int,
    element: Any,
    u_element: np.ndarray,
    tangent: bool = True,
    committed_state: Optional[Any] = None,
    num_layers: int = 5,
    centered_shell_reference: bool = False,
    stable_shell_rotation: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
    """Exact work-conjugate corotational internal force, tangent, and trial state.

    Computes ``p(u) = B(u)^T f_ref(u_d)`` and
    ``K(u) = B(u)^T K_ref(u_d) B(u) + sum_a f_ref,a nabla^2 u_d,a(u)``, where
    ``B(u) = du_d / du``.  Because ``u_d(u)`` is strictly invariant under
    arbitrary global rigid-body motions in ``SE(3)``, ``p(u)`` is an exact
    potential gradient when ``f_ref`` is conservative, ``K(u)`` is symmetric
    whenever ``K_ref`` is symmetric, and spatial force/moment equilibrium
    holds identically via ``t_i = J_l(theta_i)^{-T} p_{rot,i}``.
    """
    if bool(getattr(element, "formulation_native_total_lagrangian", False)):
        raise ValueError(
            "qualified S3 uses formulation-native total-Lagrangian mechanics "
            "and cannot be evaluated through the generic corotational wrapper"
        )
    reference = _corotational_cache(
        model, centered_shell=centered_shell_reference
    ).get(int(element_id))
    if reference is None or reference.category is None:
        return None, None, None

    num_nodes = int(element.num_nodes)
    u = np.asarray(u_element, dtype=float).reshape(num_nodes, 6)
    rotations = u[:, 3:]
    u_d, R_rig, deformed = _corotational_deformation_state(
        reference,
        element,
        u_element,
        stable_shell_rotation=stable_shell_rotation,
    )
    material = model.get_material(element.material_name)
    f_ref, k_ref, trial_state = element.compute_nonlinear_response(
        model.mesh, material, u_d, committed_state, int(num_layers), tangent
    )
    f_ref = np.asarray(f_ref, dtype=float).reshape(-1)
    D, U, G = _pullback_jacobian_factors(
        reference, element, deformed, rotations, R_rig
    )
    B = D + U @ G
    p_global = B.T @ f_ref
    if not tangent:
        return p_global, None, trial_state

    k_ref = np.asarray(k_ref, dtype=float)
    material_tangent, geometric_tangent = _work_conjugate_corotational_tangent(
        reference, element, deformed, rotations, R_rig, G, B, f_ref, k_ref
    )
    return p_global, material_tangent + geometric_tangent, trial_state


def _consistent_corotational_tangent_components(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
    E: np.ndarray,
    f_global: np.ndarray,
    k_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the consistent corotational tangent into material and geometric parts.

    Chain rule of ``f = E(omega) K_ref u_d(u, omega)`` through the three
    dependency paths:

    - ``D`` — pull-back derivative at fixed frame: centered translations and
      right-Jacobian-corrected nodal rotation increments;
    - ``U`` — pull-back sensitivity to the rigid rotation;
    - ``S`` — rotation of the internal force with the frame.

    ``K = E K_ref D + (S + E K_ref U) G`` with ``G`` the frame sensitivity
    from :func:`_rotation_sensitivity`.  The result is deliberately not
    symmetrized: additive rotation-vector coordinates and frame effects
    produce a generally nonsymmetric exact Jacobian.
    """
    num_nodes = deformed.shape[0]
    n_dofs = num_nodes * 6
    D, U, G = _pullback_jacobian_factors(
        reference, element, deformed, rotations, R_rig
    )

    S = np.zeros((n_dofs, 3), dtype=float)
    for i in range(num_nodes):
        S[i * 6 : i * 6 + 3, :] = -_skew(f_global[i * 6 : i * 6 + 3])
        S[i * 6 + 3 : i * 6 + 6, :] = -_skew(f_global[i * 6 + 3 : i * 6 + 6])

    EK = E @ k_ref
    material_tangent = EK @ D + (EK @ U) @ G
    geometric_tangent = S @ G
    return material_tangent, geometric_tangent


def _consistent_corotational_tangent(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
    E: np.ndarray,
    f_global: np.ndarray,
    k_ref: np.ndarray,
) -> np.ndarray:
    """Return the exact additive-coordinate Jacobian used by Newton solves."""

    material, geometric = _consistent_corotational_tangent_components(
        reference,
        element,
        deformed,
        rotations,
        R_rig,
        E,
        f_global,
        k_ref,
    )
    return material + geometric


def validate_corotational_scope(model: "FEModel") -> None:
    """Validate corotational applicability.

    Since Phase 4 the corotational path routes the deformational displacements
    through the elements' own nonlinear local responses, so layered shell J2
    plasticity and beam fiber plasticity are supported.  The function is kept
    as the hook for future scope checks.
    """
    for element in model.mesh.elements.values():
        if bool(getattr(element, "formulation_native_total_lagrangian", False)):
            raise ValueError(
                "qualified S3 uses formulation-native total-Lagrangian mechanics "
                "and cannot be selected with generic corotational kinematics"
            )
    return None


def _nonlinear_minus_linear_rotation(theta: np.ndarray) -> np.ndarray:
    """Return ``(R(theta) - I) - skew(theta)`` without small-angle cancellation."""
    w = np.asarray(theta, dtype=float).reshape(3)
    alpha = float(np.linalg.norm(w))
    K = _skew(w)
    if alpha < 1.0e-4:
        a2 = alpha * alpha
        c1 = -a2 * (1.0 - a2 / 20.0) / 6.0
        c2 = 0.5 * (1.0 - a2 / 12.0)
    else:
        c1 = float(np.sin(alpha) / alpha - 1.0)
        c2 = float((1.0 - np.cos(alpha)) / (alpha * alpha))
    return c1 * K + c2 * (K @ K)


class SO3MeanDomainError(ValueError):
    """An intrinsic SO(3) mean input left its supported convex/chart domain.

    Raised by the domain guards of :func:`_so3_weighted_mean` (principal
    rotation-vector chart, unique two-master geodesic, convex anchor ball,
    mean chart boundary, singular tangent operator).  It is a ``ValueError``
    for backward compatibility, but carries a distinct type so callers can
    classify a *state-domain* refusal of a trial point (for example a
    line-search trial whose master rotations left the chart) instead of
    aborting on an arbitrary exception.  Input-contract violations (finite
    rotations/weights, nonnegative partition weights) remain plain
    ``ValueError``.
    """


def _so3_weighted_mean(
    master_thetas,
    weights,
    label,
):
    """Intrinsic weighted SO(3) mean with exact tangent derivatives.

    Returns ``(theta_m, (G_i, ...))`` where ``R(theta_m)`` solves the
    Karcher-type condition ``sum_i w_i Log(R_m^T R_i) = 0`` and
    ``G_i = dtheta_m/dtheta_i`` is its exact differential in additive
    rotation-vector coordinates:

        Ghat_i = R_m B^{-1} w_i J_l(psi_i)^{-1} R_m^T      (spatial),
        G_i    = J_l(theta_m)^{-1} Ghat_i J_l(theta_i),
        B      = sum_i w_i J_l(psi_i)^{-1},  psi_i = Log(R_m^T R_i).

    The spatial blocks satisfy ``sum_i Ghat_i = I``, which makes the mean
    left-equivariant: for a common superposed rotation ``Q``,
    ``mean(Q R_i, w) = Q mean(R_i, w)``, and the parameter tangents satisfy
    the differential rigid-spin identity
    ``sum_i G_i J_l(theta_i)^{-1} = J_l(theta_m)^{-1}``, the covector
    counterpart of physical wrench preservation.

    Two-master means use the unique shortest-geodesic closed form.  Larger
    means are restricted to an open pi/4 ball about the first master, with
    the returned mean inside that same ball.  Multi-master input/output
    rotation vectors must remain inside the supported principal chart.
    Nonnegative partition weights are required; their common scale cancels
    from the mean equation and its differential.  The direct route keeps
    its existing rotation coordinates.
    """
    thetas = [np.asarray(t, dtype=float).reshape(3).copy() for t in master_thetas]
    ws = [float(w) for w in weights]
    if not thetas or len(thetas) != len(ws):
        raise ValueError(f"{label}: SO(3) mean requires at least one master")
    if not all(np.all(np.isfinite(t)) for t in thetas) or not np.all(np.isfinite(ws)):
        raise ValueError(f"{label}: SO(3) mean requires finite rotations and weights")
    if any(w < 0.0 for w in ws):
        raise ValueError(
            f"{label}: the intrinsic SO(3) mean on the active rotating-offset "
            f"route requires nonnegative weights; got {ws}"
        )
    weight_sum = float(sum(ws))
    if abs(weight_sum - 1.0) > 1.0e-8:
        raise ValueError(
            f"{label}: the intrinsic SO(3) mean requires weights forming a "
            f"partition of unity; got sum {weight_sum!r}"
        )
    if len(thetas) == 1:
        return thetas[0], (np.eye(3),)
    ws = [w / weight_sum for w in ws]
    chart_limit = np.pi - 1.0e-6
    if any(float(np.linalg.norm(t)) >= chart_limit for t in thetas):
        raise SO3MeanDomainError(f"{label}: interpolated SO(3) mean is outside the principal rotation-vector chart")
    first = thetas[0]
    if all(np.array_equal(t, first) for t in thetas[1:]):
        return first.copy(), tuple(w * np.eye(3) for w in ws)

    rotations = [rotation_matrix_from_vector(t) for t in thetas]
    anchor = rotations[0]
    if len(thetas) == 2:
        relative = rotation_vector_from_matrix(anchor.T @ rotations[1])
        if float(np.linalg.norm(relative)) >= chart_limit:
            raise SO3MeanDomainError(f"{label}: ambiguous shortest-geodesic SO(3) mean")
        fraction = ws[1]
        R_m = anchor @ rotation_matrix_from_vector(fraction * relative)
        psis = [-fraction * relative, (1.0 - fraction) * relative]
    else:
        radius = np.pi / 4.0
        if any(float(np.linalg.norm(rotation_vector_from_matrix(anchor.T @ r))) >= radius
               for r in rotations):
            raise SO3MeanDomainError(f"{label}: SO(3) mean requires a unique convex anchor ball of radius pi/4")
        R_m = anchor.copy()
        for _ in range(100):
            psis = [rotation_vector_from_matrix(R_m.T @ r) for r in rotations]
            f = sum((w * psi for w, psi in zip(ws, psis)), np.zeros(3))
            if float(np.linalg.norm(f)) <= 8.0 * np.finfo(float).eps:
                break
            R_m = R_m @ rotation_matrix_from_vector(f)
        else:
            raise SO3MeanDomainError(f"{label}: SO(3) mean failed to converge in the convex anchor ball")
        if float(np.linalg.norm(rotation_vector_from_matrix(anchor.T @ R_m))) >= radius:
            raise SO3MeanDomainError(f"{label}: SO(3) mean left the unique convex anchor ball")
    theta_m = rotation_vector_from_matrix(R_m)
    if float(np.linalg.norm(theta_m)) >= chart_limit:
        raise SO3MeanDomainError(f"{label}: SO(3) mean reached the principal rotation-vector chart boundary")
    A = [_rotation_left_jacobian_inverse(psi) for psi in psis]
    B = np.zeros((3, 3), dtype=float)
    for a, w in zip(A, ws):
        B += w * a
    if float(np.min(np.linalg.eigvalsh(0.5 * (B + B.T)))) <= 0.0:
        raise SO3MeanDomainError(
            f"{label}: intrinsic SO(3) mean tangent operator is singular; "
            "ambiguous mean refused"
        )
    B_inv = np.linalg.inv(B)
    Jl_m_inv = _rotation_left_jacobian_inverse(theta_m)
    G = []
    for t, a, w in zip(thetas, A, ws):
        Ghat = R_m @ (B_inv @ (w * a)) @ R_m.T
        Jl_i = _rotation_right_jacobian(t).T
        G.append(Jl_m_inv @ Ghat @ Jl_i)
    return theta_m, tuple(G)


def rotating_mpc_lever_arm_jacobian(
    theta_bar: np.ndarray,
    eccentricity: np.ndarray,
) -> np.ndarray:
    """Return ``A(theta_bar, r_0) = d((R(theta_bar) - I) r_0) / d(theta_bar)``.

    Exact closed-form derivative:
        A(theta_bar, r_0) = -R(theta_bar) @ skew(r_0) @ J_r(theta_bar)
    which reduces at ``theta_bar = 0`` to ``-skew(r_0)`` (i.e. ``theta x r_0``).
    """
    w = np.asarray(theta_bar, dtype=float).reshape(3)
    r0 = np.asarray(eccentricity, dtype=float).reshape(3)
    R = rotation_matrix_from_vector(w)
    Jr = _rotation_right_jacobian(w)
    return -(R @ _skew(r0) @ Jr)


def rotating_mpc_geometric_hessian(
    theta_bar: np.ndarray,
    eccentricity: np.ndarray,
    slave_force: np.ndarray,
) -> np.ndarray:
    """Return the symmetric 3x3 Hessian ``d(A(theta_bar, r_0)^T f_b) / d(theta_bar)``."""
    f_b = np.asarray(slave_force, dtype=float).reshape(3)
    if float(np.linalg.norm(f_b)) <= 1.0e-30:
        return np.zeros((3, 3), dtype=float)
    w = np.asarray(theta_bar, dtype=float).reshape(3)
    r0 = np.asarray(eccentricity, dtype=float).reshape(3)
    if float(np.linalg.norm(r0)) <= 1.0e-30:
        return np.zeros((3, 3), dtype=float)
    H = np.zeros((3, 3), dtype=float)
    h = 1.0e-6
    for j in range(3):
        wp = w.copy()
        wm = w.copy()
        wp[j] += h
        wm[j] -= h
        gp = rotating_mpc_lever_arm_jacobian(wp, r0).T @ f_b
        gm = rotating_mpc_lever_arm_jacobian(wm, r0).T @ f_b
        H[:, j] = (gp - gm) / (2.0 * h)
    return 0.5 * (H + H.T)


class NonlinearRotatingMPCEntry:
    """One finite-rotation eccentric beam-shell coupling descriptor."""

    def __init__(
        self,
        *,
        element_id: int,
        slave_u_dofs: Tuple[int, int, int],
        slave_theta_dofs: Tuple[int, int, int],
        masters: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], float], ...],
        eccentricity: np.ndarray,
        theta_proj: Any,
        label_prefix: str,
        master_theta_projs: Tuple[Any, ...] = (),
        director_trace_geometry: Any = None,
    ) -> None:
        self.element_id = int(element_id)
        if director_trace_geometry is not None:
            positions,slave,offset,xi,nodes = director_trace_geometry
            copied = [np.array(a,dtype=float,copy=True) for a in (positions,slave,offset)]
            for array in copied: array.setflags(write=False)
            director_trace_geometry = (*copied,float(xi),tuple(nodes))
        self.director_trace_geometry = director_trace_geometry
        self.slave_u_dofs = np.asarray(slave_u_dofs, dtype=np.intp)
        self.slave_theta_dofs = np.asarray(slave_theta_dofs, dtype=np.intp)
        self.masters = tuple(
            (
                np.asarray(m_u, dtype=np.intp),
                np.asarray(m_theta, dtype=np.intp),
                float(w),
            )
            for m_u, m_theta, w in masters
        )
        self.eccentricity = np.asarray(eccentricity, dtype=float).reshape(3).copy()
        self.r_skew = _skew(self.eccentricity)
        self.theta_proj = theta_proj.tocsr()
        self.label_prefix = str(label_prefix)
        self.master_theta_projs = tuple(p.tocsr() for p in master_theta_projs)

    def master_rotation(self, u: np.ndarray) -> np.ndarray:
        """Linear average ``theta_lin = sum_i w_i theta_i`` (legacy relation)."""
        theta_bar = np.zeros(3, dtype=float)
        for _m_u, m_theta, w in self.masters:
            theta_bar += w * u[m_theta]
        return theta_bar

    def master_rotations(self, u: np.ndarray) -> list:
        """Individual master rotation vectors ``theta_i``."""
        return [
            np.asarray(u[m_theta], dtype=float).reshape(3).copy()
            for _m_u, m_theta, _w in self.masters
        ]

    def mean_state(self, u: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """Intrinsic weighted SO(3) mean of the master rotations and tangents."""
        thetas = self.master_rotations(u)
        weights = [w for _m_u, _m_theta, w in self.masters]
        return _so3_weighted_mean(thetas, weights, self.label_prefix)

    def master_translation(self, u: np.ndarray) -> np.ndarray:
        u_bar = np.zeros(3, dtype=float)
        for m_u, _m_theta, w in self.masters:
            u_bar += w * u[m_u]
        return u_bar

    def director_state(self, u):
        from ._director_trace import director_trace_state, DirectorTraceDomainError
        x = np.concatenate([np.r_[u[m_u],u[m_theta]] for m_u,m_theta,_w in self.masters])
        cached = getattr(self, "_director_cache", None)
        if cached is not None and np.array_equal(x,cached[0]):
            return cached[1]
        positions,slave,offset,xi,_nodes = self.director_trace_geometry
        try:
            state = director_trace_state(positions,slave,offset,xi,x)
        except DirectorTraceDomainError as exc:
            raise SO3MeanDomainError(f"{self.label_prefix}: {exc}") from exc
        if not all(np.all(np.isfinite(a)) for a in state):
            raise SO3MeanDomainError(f"{self.label_prefix}: nonfinite director trace")
        self._director_cache = (x.copy(),state)
        return state

    def director_reference(self):
        from ._director_trace import director_trace_state
        cached = getattr(self,"_director_reference",None)
        if cached is None:
            positions,slave,offset,xi,_nodes = self.director_trace_geometry
            cached = director_trace_state(positions,slave,offset,xi,np.zeros(12))[1]
            self._director_reference = cached
        return cached

    def director_dofs(self):
        return np.concatenate([np.r_[m_u,m_theta] for m_u,m_theta,_w in self.masters])


class NonlinearRotatingMPCMap:
    """State-dependent master-slave kinematic map for rotating eccentricities.

    Given the reference linear constraint transformation ``T_linear`` (which
    encodes ``u_b,lin = sum_i w_i u_{s,i} + theta_lin x r_0`` and
    ``theta_b = theta_lin := sum_i w_i theta_{s,i}``), the nonlinear
    kinematic map replaces the linearized lever-arm offset and the
    non-objective averaged orientation with the intrinsic weighted SO(3)
    mean ``R_m = mean(R_i, w_i)`` on the active rotating-offset route:

        u_b(q)     = sum_i w_i u_{s,i}(q)
                     + (R_m(q) - I) r_0
        theta_b(q) = theta_m(q) = Log(R_m(q))

    Under a common superposed rigid rotation ``Q`` the weighted deformed
    attachment point and the slave orientation transform equivariantly,
    ``x_b -> Q x_b`` and ``R_b -> Q R_b`` (adopted rigid-spin contract),
    The correction to the linear displacement therefore subtracts
    ``theta_lin x r_0``. The direct single-master route and identical-rotation
    limit retain the legacy physical map. Zero Lagrange multipliers are used.
    """

    def __init__(
        self,
        T_linear: Any,
        entries: Tuple[NonlinearRotatingMPCEntry, ...],
    ) -> None:
        from scipy import sparse

        self.T_linear = sparse.csr_matrix(T_linear, dtype=float)
        self.entries = tuple(entries)
        self.total_dofs = int(self.T_linear.shape[0])
        self.n_red = int(self.T_linear.shape[1])

    @property
    def active(self) -> bool:
        return bool(self.entries)

    def _slave_corrections(self, u: np.ndarray) -> list:
        """Per-entry slave translation/rotation corrections ``(d_u, d_theta)``.

        Both corrections are evaluated from the supplied state only: master
        rotations are never corrected by the map, so the result is a proper
        function of the input state and forward/inverse application is
        order-independent across chained entries.
        """
        corrections = []
        for entry in self.entries:
            if entry.director_trace_geometry is not None:
                x=u[entry.director_dofs()]
                y,_B,_H=entry.director_state(u)
                correction=y-entry.director_reference()@x
                corrections.append((entry,correction[:3],correction[3:]))
                continue
            theta_lin = entry.master_rotation(u)
            theta_m, _G = entry.mean_state(u)
            d_theta = theta_m - theta_lin
            d_u = _nonlinear_minus_linear_rotation(theta_m) @ entry.eccentricity
            d_u = d_u + np.cross(d_theta, entry.eccentricity)
            corrections.append((entry, d_u, d_theta))
        return corrections

    def apply_kinematics(self, u_linear: np.ndarray) -> np.ndarray:
        """Map a linear MPC displacement state ``u_lin`` to the nonlinear manifold."""
        u = np.asarray(u_linear, dtype=float).reshape(-1).copy()
        if not self.entries:
            return u
        for entry, d_u, d_theta in self._slave_corrections(u):
            u[entry.slave_u_dofs] += d_u
            u[entry.slave_theta_dofs] += d_theta
        return u

    def to_linear_displacement(self, u_nonlinear: np.ndarray) -> np.ndarray:
        """Exact inverse diffeomorphism mapping ``u`` back to linear MPC coordinates."""
        u_lin = np.asarray(u_nonlinear, dtype=float).reshape(-1).copy()
        if not self.entries:
            return u_lin
        for entry, d_u, d_theta in self._slave_corrections(u_lin):
            u_lin[entry.slave_u_dofs] -= d_u
            u_lin[entry.slave_theta_dofs] -= d_theta
        return u_lin

    def jacobian(self, u_current: np.ndarray) -> Any:
        """Return the state-dependent kinematic Jacobian ``J(u) = du / dq``."""
        from scipy import sparse

        if not self.entries:
            return self.T_linear
        u = np.asarray(u_current, dtype=float).reshape(-1)
        row_blocks = []
        col_blocks = []
        data_blocks = []
        for entry in self.entries:
            if entry.director_trace_geometry is not None:
                _y,B,_H=entry.director_state(u)
                block=sparse.csr_matrix(B-entry.director_reference())@self.T_linear[entry.director_dofs(),:]
                rows=np.r_[entry.slave_u_dofs,entry.slave_theta_dofs]
                coo=block.tocoo()
                row_blocks.append(rows[coo.row]); col_blocks.append(coo.col); data_blocks.append(coo.data)
                continue
            if entry.theta_proj.nnz == 0 and all(
                p.nnz == 0 for p in entry.master_theta_projs
            ):
                continue
            theta_m, G_list = entry.mean_state(u)
            A = rotating_mpc_lever_arm_jacobian(theta_m, entry.eccentricity)
            # M_proj = dtheta_m/dq = sum_i G_i dtheta_i/dq
            M = None
            for G, P in zip(G_list, entry.master_theta_projs):
                if P.nnz == 0:
                    continue
                term = sparse.csr_matrix(G) @ P
                M = term if M is None else (M + term)
            if M is None:
                continue
            # slave translations: A @ M + skew(r_0) @ theta_proj
            C_u = sparse.csr_matrix(A) @ M + sparse.csr_matrix(entry.r_skew) @ entry.theta_proj
            # slave rotations: M - theta_proj
            C_theta = M - entry.theta_proj
            for rows, block in ((entry.slave_u_dofs, C_u), (entry.slave_theta_dofs, C_theta)):
                coo = block.tocoo()
                if coo.nnz and np.any(coo.data):
                    row_blocks.append(np.asarray(rows)[coo.row])
                    col_blocks.append(coo.col)
                    data_blocks.append(coo.data)
        if not data_blocks:
            return self.T_linear
        delta_T = sparse.coo_matrix(
            (
                np.concatenate(data_blocks),
                (np.concatenate(row_blocks), np.concatenate(col_blocks)),
            ),
            shape=(self.total_dofs, self.n_red),
            dtype=float,
        ).tocsr()
        return (self.T_linear + delta_T).tocsr()

    def reduce_force(self, u_current: np.ndarray, full_force: np.ndarray) -> np.ndarray:
        """Return ``J(u)^T full_force`` without materializing ``J(u)``."""
        f = np.asarray(full_force, dtype=float).reshape(-1)
        if any(e.director_trace_geometry is not None for e in self.entries):
            return np.asarray(self.jacobian(u_current).T@f,dtype=float).reshape(-1)
        if not self.entries:
            return np.asarray(self.T_linear.T @ f, dtype=float).reshape(-1)
        u = np.asarray(u_current, dtype=float).reshape(-1)
        f_mod = f.copy()
        for entry in self.entries:
            f_b = f[entry.slave_u_dofs]
            m_b = f[entry.slave_theta_dofs]
            if float(np.linalg.norm(f_b)) <= 1.0e-30 and float(np.linalg.norm(m_b)) <= 1.0e-30:
                continue
            theta_m, G_list = entry.mean_state(u)
            A = rotating_mpc_lever_arm_jacobian(theta_m, entry.eccentricity)
            v = A.T @ f_b + m_b
            const = entry.r_skew.T @ f_b - m_b
            for (_m_u, m_theta, w), G in zip(entry.masters, G_list):
                f_mod[m_theta] += G.T @ v + w * const
        return np.asarray(self.T_linear.T @ f_mod, dtype=float).reshape(-1)

    def geometric_tangent(
        self,
        u_current: np.ndarray,
        full_imbalance: np.ndarray,
    ) -> Any:
        """Return ``K_mpc,geom(u, F_int - F_ext) = d(J^T F)/dq`` (F frozen).

        The complete second derivative of the slave correction is included:
        second derivatives of the intrinsic mean and of the slave
        orientation contracted with the full slave rotation imbalance
        ``m_b``, and offset-translation derivatives contracted with ``f_b``.
        It is evaluated by finite differences of the exact
        per-master covector correction of :meth:`reduce_force`, so it stays
        consistent with the forward map. Central differences are preferred;
        near an admitted-domain boundary a second-order one-sided stencil
        stays inside that domain. The stencil is reduced if neither side is
        usable; an unrepresentable local stencil is explicitly refused.
        """
        from scipy import sparse

        if not self.entries or self.n_red == 0:
            return sparse.csr_matrix((self.n_red, self.n_red), dtype=float)
        u = np.asarray(u_current, dtype=float).reshape(-1)
        imbalance = np.asarray(full_imbalance, dtype=float).reshape(-1)
        terms = []
        h = 1.0e-6
        for entry in self.entries:
            if entry.director_trace_geometry is not None:
                _y,_B,H=entry.director_state(u)
                force=imbalance[np.r_[entry.slave_u_dofs,entry.slave_theta_dofs]]
                P=self.T_linear[entry.director_dofs(),:]
                terms.append(P.T@sparse.csr_matrix(np.einsum("i,ijk->jk",force,H))@P)
                continue
            projs = entry.master_theta_projs
            if all(p.nnz == 0 for p in projs):
                continue
            f_b = imbalance[entry.slave_u_dofs]
            m_b = imbalance[entry.slave_theta_dofs]
            if float(np.linalg.norm(f_b)) <= 1.0e-30 and float(np.linalg.norm(m_b)) <= 1.0e-30:
                continue
            thetas = entry.master_rotations(u)
            weights = [w for _m_u, _m_theta, w in entry.masters]

            def covectors(state_thetas):
                theta_m, G_list = _so3_weighted_mean(
                    state_thetas, weights, entry.label_prefix
                )
                A = rotating_mpc_lever_arm_jacobian(theta_m, entry.eccentricity)
                vv = A.T @ f_b + m_b
                return [G.T @ vv for G in G_list]

            n_m = len(projs)
            centre = covectors(thetas)
            D = [[np.zeros((3, 3), dtype=float) for _ in range(n_m)] for _ in range(n_m)]
            for j in range(n_m):
                for c in range(3):
                    def probe(distance):
                        shifted = [t.copy() for t in thetas]
                        shifted[j][c] += distance
                        if shifted[j][c] == thetas[j][c]:
                            return None
                        try:
                            return covectors(shifted)
                        except SO3MeanDomainError:
                            return None

                    step = h
                    for _ in range(32):
                        cp, cm = probe(step), probe(-step)
                        if cp is not None and cm is not None:
                            derivative = [(p - m) / (2.0 * step) for p, m in zip(cp, cm)]
                            break
                        c2p = probe(2.0 * step) if cp is not None else None
                        c2m = probe(-2.0 * step) if cm is not None else None
                        if cp is not None and c2p is not None:
                            derivative = [(-3.0 * z + 4.0 * p - p2) / (2.0 * step)
                                          for z, p, p2 in zip(centre, cp, c2p)]
                            break
                        if cm is not None and c2m is not None:
                            derivative = [(3.0 * z - 4.0 * m + m2) / (2.0 * step)
                                          for z, m, m2 in zip(centre, cm, c2m)]
                            break
                        step *= 0.5
                    else:
                        raise SO3MeanDomainError(
                            f"{entry.label_prefix}: no representable admitted SO(3) tangent stencil"
                        )
                    for i in range(n_m):
                        D[i][j][:, c] = derivative[i]
            for i in range(n_m):
                for j in range(n_m):
                    if not np.any(D[i][j]):
                        continue
                    term = projs[i].T @ sparse.csr_matrix(D[i][j]) @ projs[j]
                    if term.nnz:
                        terms.append(term)
        if not terms:
            return sparse.csr_matrix((self.n_red, self.n_red), dtype=float)
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        total = (total + total.T).tocsr()
        return 0.5 * total

    def reduce_tangent(
        self,
        u_current: np.ndarray,
        K_T: Any,
        full_imbalance: Optional[np.ndarray] = None,
        K_ext: Optional[Any] = None,
    ) -> Any:
        """Return ``J(u)^T (K_T - K_ext) J(u) + K_mpc,geom(u, full_imbalance)``."""
        J = self.jacobian(u_current)
        K_red = (J.T @ K_T @ J).tocsr()
        if K_ext is not None and getattr(K_ext, "nnz", 0):
            K_red = (K_red - (J.T @ K_ext @ J)).tocsr()
        if full_imbalance is not None:
            K_geom = self.geometric_tangent(u_current, full_imbalance)
            if K_geom.nnz:
                K_red = (K_red + K_geom).tocsr()
        return K_red

    def transfer_slave_imbalance_to_masters(
        self,
        u_current: np.ndarray,
        full_imbalance: np.ndarray,
    ) -> np.ndarray:
        """Transfer slave forces/moments to master shell DOFs for support reactions.

        The slave force ``f_b`` is distributed with the linear weights to the
        master translations; the mean-chart slave moment covector
        ``m_b + A(theta_m)^T f_b`` is distributed through the transposed mean
        tangents ``G_i^T`` to the master rotation covectors.  This is the
        exact adjoint of the nonlinear slave orientation map, so the total
        physical spatial wrench is preserved (the old common-covector
        shortcut left a moment defect ``D^T (m_b + A^T f_b)``).
        """
        imbalance = np.asarray(full_imbalance, dtype=float).reshape(-1).copy()
        if not self.entries:
            return imbalance
        u = np.asarray(u_current, dtype=float).reshape(-1)
        for entry in self.entries:
            f_b = imbalance[entry.slave_u_dofs].copy()
            m_b = imbalance[entry.slave_theta_dofs].copy()
            if entry.director_trace_geometry is not None:
                _y,B,_H=entry.director_state(u)
                imbalance[entry.director_dofs()] += B.T@np.r_[f_b,m_b]
                imbalance[entry.slave_u_dofs]=0.
                imbalance[entry.slave_theta_dofs]=0.
                continue
            if float(np.linalg.norm(f_b)) <= 1.0e-30 and float(np.linalg.norm(m_b)) <= 1.0e-30:
                imbalance[entry.slave_u_dofs] = 0.0
                imbalance[entry.slave_theta_dofs] = 0.0
                continue
            theta_m, G_list = entry.mean_state(u)
            A = rotating_mpc_lever_arm_jacobian(theta_m, entry.eccentricity)
            moment_covector = m_b + A.T @ f_b
            for (m_u, m_theta, w), G in zip(entry.masters, G_list):
                imbalance[m_u] += w * f_b
                imbalance[m_theta] += G.T @ moment_covector
            imbalance[entry.slave_u_dofs] = 0.0
            imbalance[entry.slave_theta_dofs] = 0.0
        return imbalance


def has_nonlinear_rotating_mpc(model: "FEModel") -> bool:
    """Return True if ``model`` contains active rotating-arm beam-shell MPCs."""
    mesh = getattr(model, "mesh", None)
    elements = getattr(mesh, "elements", None)
    if not elements:
        return False
    for element in elements.values():
        getter = getattr(element, "get_nonlinear_mpc_descriptor", None)
        if callable(getter) and getter(mesh) is not None:
            return True
    return False


def _reject_downstream_rotating_slave_translation_dependencies(
    model: "FEModel",
    entries: Tuple[NonlinearRotatingMPCEntry, ...],
) -> None:
    """Fail closed when an admitted affine equation consumes a rotating slave DOF.

    The rotating map corrects each descriptor's own slave translations, and
    the intrinsic SO(3) mean makes the slave rotation of every multi-master
    coupling a nonlinear function of the master rotations.  Any other
    admitted affine equation that uses one of those DOFs as a nonzero
    master would silently retain the linearized value in the returned
    physical displacement, so such compositions are refused before
    constitutive activation instead of being solved on the wrong manifold.
    The check uses the actual normalized audit equations, including their
    coefficient filtering.  Ordinary linear chains without a rotating
    descriptor and upstream shell row ties remain supported; single-master
    direct slave rotations stay linear under the map and are not
    restricted.  A currently identical multi-master state does not admit a
    dependency that becomes nonlinear later.
    """
    rotating_slave_translations: set = set()
    nonlinear_slave_rotations: set = set()
    for entry in entries:
        rotating_slave_translations.update(int(d) for d in entry.slave_u_dofs)
        if len(entry.masters) > 1:
            nonlinear_slave_rotations.update(int(d) for d in entry.slave_theta_dofs)
    blocked = rotating_slave_translations | nonlinear_slave_rotations
    if not blocked:
        return
    from .constraint_audit import require_valid_constraints

    report = require_valid_constraints(model)
    for equation in report.equations:
        if str(equation.kind) == "support":
            continue
        dependent = int(equation.dependent_dof)
        for dof, coefficient in equation.coefficients:
            master = int(dof)
            if master == dependent or float(coefficient) == 0.0:
                continue
            if master in blocked:
                raise ValueError(
                    f"Rotating-MPC route: affine equation '{equation.origin}' "
                    f"(dependent DOF {dependent}) uses rotating slave "
                    f"DOF {master} as a master. Downstream dependencies on "
                    "rotating slave translations, or on the nonlinearly "
                    "interpolated slave rotations of multi-master couplings, "
                    "are not supported; remove the downstream equation or the "
                    "rotating coupling."
                )


def build_nonlinear_rotating_mpc_map(
    model: "FEModel",
    T_linear: Any,
) -> Optional[NonlinearRotatingMPCMap]:
    """Build a :class:`NonlinearRotatingMPCMap` if rotating-arm MPCs are present."""
    from scipy import sparse

    mesh = getattr(model, "mesh", None)
    elements = getattr(mesh, "elements", None)
    if not elements:
        return None
    T_csr = sparse.csr_matrix(T_linear, dtype=float)
    entries = []
    for _elem_id, element in sorted(elements.items(), key=lambda item: int(item[0])):
        getter = getattr(element, "get_nonlinear_mpc_descriptor", None)
        if not callable(getter):
            continue
        desc = getter(mesh)
        if desc is None:
            continue
        masters = tuple(desc["masters"])
        if len(masters) > 1 and any(w < 0.0 for _m_u, _m_theta, w in masters):
            raise ValueError(
                f"{desc['label_prefix']}: the intrinsic SO(3) mean on the "
                "active rotating-offset route requires nonnegative shape "
                f"weights; got {[w for _m_u, _m_theta, w in masters]}"
            )
        slave_theta = tuple(int(d) for d in desc["slave_theta_dofs"])
        theta_proj = T_csr[list(slave_theta), :]
        master_theta_projs = tuple(
            T_csr[[int(d) for d in m_theta], :]
            for _m_u, m_theta, _w in masters
        )
        entries.append(
            NonlinearRotatingMPCEntry(
                element_id=int(desc["element_id"]),
                slave_u_dofs=tuple(int(d) for d in desc["slave_u_dofs"]),
                slave_theta_dofs=slave_theta,
                masters=masters,
                eccentricity=np.asarray(desc["eccentricity"], dtype=float),
                theta_proj=theta_proj,
                label_prefix=str(desc["label_prefix"]),
                master_theta_projs=master_theta_projs,
                director_trace_geometry=desc.get("director_trace_geometry"),
            )
        )
    if not entries:
        return None
    _reject_downstream_rotating_slave_translation_dependencies(model, tuple(entries))
    return NonlinearRotatingMPCMap(T_csr, tuple(entries))
