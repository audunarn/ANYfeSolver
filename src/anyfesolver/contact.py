"""Limited rigid-sphere structural impact dynamics.

The contact scope is one rigid sphere with frictionless normal penalty contact
against shell midsurface/thickness-offset targets and opt-in beam-axis segment
targets. Optional linear or material-nonlinear structural response and several
engineering damage/erosion screens are available. This is not arbitrary
contact, profile-resolved beam contact, or crack propagation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .assembly import build_constraint_transformation, reconstruct_full_solution
from .boundary import BoundaryCondition, LoadCase
from .cases import make_result_case
from .constraint_audit import constraint_residual_summary
from .contact_performance import ContactWorkBuffer, ContactWorkCounters
from .control import CancellationToken, ProgressCallback, SolveCancelled, cancellation_safe_point, emit_progress
from .damage_matrix_performance import (
    DamageMatrixPlan,
    DamageMatrixPlanFallback,
    DamageMatrixPlanGate,
)
from .dynamics import (
    TransientConfig,
    _full_initial_vector,
    _node_dof_indices,
    _reduced_load,
    _saved_step_count,
    _time_grid,
    _translation_peak,
)
from .elements import ShellElement, create_shell_element
from .fracture import (
    DeletedElementRecord,
    ImpactDamageConfig,
    ImpactFractureConfig,
    PlasticImpactDamageConfig,
    element_fracture_category,
    element_measure,
    filtered_load_case_for_deleted_elements,
    fracture_summary,
    state_equivalent_plastic_strain,
    state_rtcl_increment,
)
from .linalg import MatrixClass, factorize
from .impact_performance import ImpactTangentReuseController
from .materials import beam_material_properties, material_symmetry, shell_characteristic_modulus
from .matrix_assembly import assemble_load_vector, assemble_mass_matrix, assemble_stiffness_matrix
from .recovery import enforce_memory_limit, estimate_model_memory, recovery_metadata
from .validation import ProductionValidationIssue, ProductionValidationReport, load_vector_resultant
from .threading_policy import resource_threaded

if TYPE_CHECKING:
    from .fe_core import FEModel


@dataclass(frozen=True)
class RigidSphereImpact:
    """Input state for a rigid sphere impact."""

    name: str
    radius: float
    mass: float
    start_point: Sequence[float]
    travel_direction: Sequence[float]
    speed: float
    t_start: float = 0.0

    def __post_init__(self) -> None:
        # Comparisons with NaN are false, so finiteness is checked explicitly.
        if not (np.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("sphere radius must be positive and finite")
        if not (np.isfinite(self.mass) and self.mass > 0.0):
            raise ValueError("sphere mass must be positive and finite")
        if not (np.isfinite(self.speed) and self.speed >= 0.0):
            raise ValueError("sphere speed must be non-negative and finite")
        if not (np.isfinite(self.t_start) and self.t_start >= 0.0):
            raise ValueError("sphere t_start must be non-negative and finite")
        start = np.asarray(self.start_point, dtype=float).reshape(-1)
        direction = np.asarray(self.travel_direction, dtype=float).reshape(-1)
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(direction))):
            raise ValueError("sphere start_point and travel_direction must be finite")
        if start.shape != (3,):
            raise ValueError("sphere start_point must contain three coordinates")
        if direction.shape != (3,):
            raise ValueError("sphere travel_direction must contain three coordinates")
        if float(np.linalg.norm(direction)) <= 0.0:
            raise ValueError("sphere travel_direction must be non-zero")

    @property
    def direction_unit(self) -> np.ndarray:
        direction = np.asarray(self.travel_direction, dtype=float)
        return direction / float(np.linalg.norm(direction))

    @property
    def initial_position(self) -> np.ndarray:
        return np.asarray(self.start_point, dtype=float).copy()

    @property
    def travel_velocity(self) -> np.ndarray:
        return self.direction_unit * float(self.speed)


@dataclass(frozen=True)
class SphereContactConfig:
    """Penalty-contact controls for rigid-sphere impact."""

    penalty_stiffness: Optional[float] = None
    contact_damping: float = 0.0
    search_margin: float = 0.0
    max_contact_iterations: int = 25
    penetration_tolerance: float = 1.0e-8
    force_tolerance: float = 1.0e-6
    side_mode: str = "both"
    contact_surface: str = "midsurface"
    signed_surface_offset: float = 0.0
    target_penetration_fraction: float = 0.01
    auto_penalty_safety_factor: float = 1.0
    max_sphere_travel_fraction: float = 0.25
    max_event_substeps: int = 16
    max_active_contacts: int = 1
    load_patch_radius_factor: float = 1.25
    min_load_patch_nodes: int = 4
    max_load_patch_nodes: int = 12
    save_contact_history: bool = True
    post_separation_time: float = 0.0
    contact_relaxation: str = "aitken"
    beam_contact: bool = False

    def __post_init__(self) -> None:
        if self.penalty_stiffness is not None and not (
                np.isfinite(self.penalty_stiffness) and self.penalty_stiffness > 0.0):
            raise ValueError("penalty_stiffness must be positive and finite")
        if not (np.isfinite(self.contact_damping) and self.contact_damping >= 0.0):
            raise ValueError("contact_damping must be non-negative and finite")
        if not (np.isfinite(self.search_margin) and self.search_margin >= 0.0):
            raise ValueError("search_margin must be non-negative and finite")
        for name in ("penetration_tolerance", "force_tolerance", "signed_surface_offset",
                     "target_penetration_fraction", "auto_penalty_safety_factor",
                     "max_sphere_travel_fraction", "load_patch_radius_factor",
                     "post_separation_time"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.max_contact_iterations <= 0:
            raise ValueError("max_contact_iterations must be positive")
        if self.penetration_tolerance <= 0.0:
            raise ValueError("penetration_tolerance must be positive")
        if self.force_tolerance <= 0.0:
            raise ValueError("force_tolerance must be positive")
        if self.side_mode != "both":
            raise NotImplementedError("SphereContactConfig v1 supports side_mode='both' only")
        if self.contact_surface not in {"midsurface", "top", "bottom", "signed"}:
            raise ValueError("contact_surface must be 'midsurface', 'top', 'bottom', or 'signed'")
        if self.contact_surface == "signed" and abs(float(self.signed_surface_offset)) > 1.0:
            raise ValueError("signed_surface_offset must be in [-1, 1] as a fraction of half thickness")
        if self.target_penetration_fraction <= 0.0:
            raise ValueError("target_penetration_fraction must be positive")
        if self.auto_penalty_safety_factor <= 0.0:
            raise ValueError("auto_penalty_safety_factor must be positive")
        if self.max_sphere_travel_fraction <= 0.0:
            raise ValueError("max_sphere_travel_fraction must be positive")
        if self.max_event_substeps <= 0:
            raise ValueError("max_event_substeps must be positive")
        if self.max_active_contacts <= 0:
            raise ValueError("max_active_contacts must be positive")
        if self.load_patch_radius_factor <= 0.0:
            raise ValueError("load_patch_radius_factor must be positive")
        if self.min_load_patch_nodes <= 0:
            raise ValueError("min_load_patch_nodes must be positive")
        if self.max_load_patch_nodes < self.min_load_patch_nodes:
            raise ValueError("max_load_patch_nodes must be greater than or equal to min_load_patch_nodes")
        if self.post_separation_time < 0.0:
            raise ValueError("post_separation_time must be non-negative")
        if self.contact_relaxation not in {"aitken", "none"}:
            raise ValueError("contact_relaxation must be 'aitken' or 'none'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "penalty_stiffness": None if self.penalty_stiffness is None else float(self.penalty_stiffness),
            "contact_damping": float(self.contact_damping),
            "search_margin": float(self.search_margin),
            "max_contact_iterations": int(self.max_contact_iterations),
            "penetration_tolerance": float(self.penetration_tolerance),
            "force_tolerance": float(self.force_tolerance),
            "side_mode": self.side_mode,
            "contact_surface": self.contact_surface,
            "signed_surface_offset": float(self.signed_surface_offset),
            "target_penetration_fraction": float(self.target_penetration_fraction),
            "auto_penalty_safety_factor": float(self.auto_penalty_safety_factor),
            "max_sphere_travel_fraction": float(self.max_sphere_travel_fraction),
            "max_event_substeps": int(self.max_event_substeps),
            "max_active_contacts": int(self.max_active_contacts),
            "load_patch_radius_factor": float(self.load_patch_radius_factor),
            "min_load_patch_nodes": int(self.min_load_patch_nodes),
            "max_load_patch_nodes": int(self.max_load_patch_nodes),
            "save_contact_history": bool(self.save_contact_history),
            "post_separation_time": float(self.post_separation_time),
            "contact_relaxation": self.contact_relaxation,
            "beam_contact": bool(self.beam_contact),
        }


@dataclass(frozen=True)
class NonlinearTransientConfig:
    """Material/geometric nonlinear implicit Newmark controls for impact."""

    enabled: bool = False
    num_layers: int = 5
    max_iterations: int = 20
    residual_tolerance: float = 2.0e-3
    displacement_tolerance: float = 1.0e-8
    contact_force_tolerance: float = 2.0e-3
    line_search: bool = True
    min_line_search_factor: float = 0.125
    max_cutbacks: int = 12
    min_dt: float = 1.0e-8
    tangent_reuse_iterations: int = 0
    record_element_state_history: bool = True
    kinematics: str = "von_karman"
    # Energy-based stagnation acceptance: when the Newton iterate is a fixed
    # point (displacement increment and contact change below their limits)
    # and the residual's energy content |delta . r| is a negligible fraction
    # of the impact energy, the step is accepted instead of burning cutbacks
    # on a residual criterion that cannot be met (e.g. force/tangent noise
    # floors on rotational equations).  Set to 0 to disable.
    stagnation_energy_tolerance: float = 1.0e-8
    # Start the transient from the static equilibrium of the base load case
    # instead of applying it as a sudden step (which rings the structure at
    # its axial period through the whole impact).
    equilibrate_base_load: bool = True
    # Opt-in multisecant or small-system numerical-contact Newton iteration;
    # contact_relaxation='none'
    # takes precedence. The default retains the legacy staggered iteration.
    coupled_contact_acceleration: str = "none"

    def __post_init__(self) -> None:
        if self.coupled_contact_acceleration not in {"none", "anderson", "contact_newton"}:
            raise ValueError("coupled_contact_acceleration must be 'none', 'anderson' or 'contact_newton'")
        if str(self.kinematics).lower() not in {"von_karman", "corotational"}:
            raise ValueError("NonlinearTransientConfig.kinematics must be 'von_karman' or 'corotational'")
        object.__setattr__(self, "kinematics", str(self.kinematics).lower())
        if self.num_layers <= 0:
            raise ValueError("NonlinearTransientConfig.num_layers must be positive")
        if self.max_iterations <= 0:
            raise ValueError("NonlinearTransientConfig.max_iterations must be positive")
        if self.residual_tolerance <= 0.0:
            raise ValueError("NonlinearTransientConfig.residual_tolerance must be positive")
        if self.displacement_tolerance <= 0.0:
            raise ValueError("NonlinearTransientConfig.displacement_tolerance must be positive")
        if self.contact_force_tolerance <= 0.0:
            raise ValueError("NonlinearTransientConfig.contact_force_tolerance must be positive")
        if not (0.0 < self.min_line_search_factor <= 1.0):
            raise ValueError("NonlinearTransientConfig.min_line_search_factor must be in (0, 1]")
        if self.max_cutbacks < 0:
            raise ValueError("NonlinearTransientConfig.max_cutbacks must be non-negative")
        if self.min_dt <= 0.0:
            raise ValueError("NonlinearTransientConfig.min_dt must be positive")
        if self.tangent_reuse_iterations < 0:
            raise ValueError("NonlinearTransientConfig.tangent_reuse_iterations must be non-negative")
        if self.stagnation_energy_tolerance < 0.0:
            raise ValueError("NonlinearTransientConfig.stagnation_energy_tolerance must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "num_layers": int(self.num_layers),
            "max_iterations": int(self.max_iterations),
            "residual_tolerance": float(self.residual_tolerance),
            "displacement_tolerance": float(self.displacement_tolerance),
            "contact_force_tolerance": float(self.contact_force_tolerance),
            "line_search": bool(self.line_search),
            "min_line_search_factor": float(self.min_line_search_factor),
            "max_cutbacks": int(self.max_cutbacks),
            "min_dt": float(self.min_dt),
            "tangent_reuse_iterations": int(self.tangent_reuse_iterations),
            "record_element_state_history": bool(self.record_element_state_history),
            "kinematics": self.kinematics,
            "stagnation_energy_tolerance": float(self.stagnation_energy_tolerance),
            "equilibrate_base_load": bool(self.equilibrate_base_load),
            "coupled_contact_acceleration": self.coupled_contact_acceleration,
        }


@dataclass(frozen=True)
class SphereContactRecord:
    """One active contact point at a saved or iterated state."""

    element_id: int
    local_coordinates: Tuple[float, float]
    contact_point: np.ndarray
    normal: np.ndarray
    penetration: float
    normal_force: float
    sphere_force: np.ndarray
    structure_force: np.ndarray
    contact_classification: str = "face"
    nodal_forces: Mapping[int, np.ndarray] = field(default_factory=dict)
    nodal_moments: Mapping[int, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "element_id": int(self.element_id),
            "local_coordinates": [float(self.local_coordinates[0]), float(self.local_coordinates[1])],
            "contact_point": self.contact_point.tolist(),
            "normal": self.normal.tolist(),
            "penetration": float(self.penetration),
            "normal_force": float(self.normal_force),
            "sphere_force": self.sphere_force.tolist(),
            "structure_force": self.structure_force.tolist(),
            "contact_classification": self.contact_classification,
            "nodal_forces": {int(node_id): value.tolist() for node_id, value in self.nodal_forces.items()},
        }
        if self.nodal_moments:
            payload["nodal_moments"] = {
                int(node_id): value.tolist()
                for node_id, value in self.nodal_moments.items()
            }
        return payload


@dataclass(frozen=True)
class SphereImpactResult:
    """Structural transient histories plus rigid-sphere contact histories."""

    times: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    node_histories: Dict[int, np.ndarray]
    sphere_positions: np.ndarray
    sphere_velocities: np.ndarray
    sphere_accelerations: np.ndarray
    contact_force_history: np.ndarray
    active_contact_history: Tuple[Tuple[Dict[str, Any], ...], ...]
    load_impulse: np.ndarray
    force_impulse: np.ndarray
    moment_impulse: np.ndarray
    sphere_impulse: np.ndarray
    max_penetration: float
    max_penetration_ratio: float
    peak_contact_force: float
    contact_duration: float
    sphere_momentum_balance_error: float
    peak_displacement: float
    peak_displacement_node: Optional[int]
    status: str
    diagnostics: Dict[str, Any]
    result_case: Optional[Dict[str, Any]] = None

    @property
    def quantity_metadata(self) -> Tuple[Any, ...]:
        from .quantities import describe_result_quantities

        return describe_result_quantities(self)


def _project_local_coordinates(
    element: ShellElement, coords: np.ndarray, point: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangular = element.num_nodes in (3, 6)
    if triangular:
        local = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        local = np.zeros(2, dtype=float)
    # Closest-point Newton projection.  A closed-form 2x2 solve and a
    # contact-appropriate tolerance (1e-9 in natural coordinates) keep this
    # inner loop cheap -- it is the dominant cost of penalty contact assembly.
    N, dN_dxi, dN_deta = element.compute_shape_functions(float(local[0]), float(local[1]))
    for _ in range(8):
        surface_point = N @ coords
        j0 = dN_dxi @ coords
        j1 = dN_deta @ coords
        residual = surface_point - point
        a = float(j0 @ j0)
        b = float(j0 @ j1)
        d = float(j1 @ j1)
        r0 = float(j0 @ residual)
        r1 = float(j1 @ residual)
        det = a * d - b * b
        if abs(det) <= 1.0e-30:
            break
        delta0 = (d * r0 - b * r1) / det
        delta1 = (a * r1 - b * r0) / det
        local[0] -= delta0
        local[1] -= delta1
        if triangular:
            local[0] = max(float(local[0]), 0.0)
            local[1] = max(float(local[1]), 0.0)
            total = float(local[0] + local[1])
            if total > 1.0:
                local /= total
        else:
            local[0] = min(max(float(local[0]), -1.0), 1.0)
            local[1] = min(max(float(local[1]), -1.0), 1.0)
        N, dN_dxi, dN_deta = element.compute_shape_functions(float(local[0]), float(local[1]))
        if delta0 * delta0 + delta1 * delta1 < 1.0e-18:
            break
    surface_point = N @ coords
    return local, N, dN_dxi, dN_deta, surface_point


def _project_qualified_s3_contact_coordinates(
    element: ShellElement,
    coordinates: np.ndarray,
    point: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the numbering-invariant closest point on a three-corner facet.

    The generic contact projector clips the two natural coordinates after
    each Newton correction.  That box/simplex clipping path depends on which
    physical corner happens to own each natural coordinate and therefore is
    not D3 covariant for edge and corner contact.  A flat qualified S3 facet
    permits a compact exact active-set solve: test the plane projection and
    all three edge projections, then select the physical closest point.
    """

    vertices = np.asarray(coordinates, dtype=float)
    target = np.asarray(point, dtype=float).reshape(3)
    if vertices.shape != (3, 3) or not np.all(np.isfinite(vertices)):
        raise ValueError("Qualified S3 contact coordinates must have shape (3, 3)")
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    edge_01 = vertices[1] - vertices[0]
    edge_02 = vertices[2] - vertices[0]
    gram = np.asarray(
        (
            (float(edge_01 @ edge_01), float(edge_01 @ edge_02)),
            (float(edge_01 @ edge_02), float(edge_02 @ edge_02)),
        ),
        dtype=float,
    )
    determinant = float(np.linalg.det(gram))
    if determinant <= np.finfo(float).tiny:
        raise ValueError("Qualified S3 contact facet is singular")
    natural = np.linalg.solve(
        gram,
        np.asarray(
            (
                float(edge_01 @ (target - vertices[0])),
                float(edge_02 @ (target - vertices[0])),
            ),
            dtype=float,
        ),
    )
    plane_weights = np.asarray(
        (1.0 - natural[0] - natural[1], natural[0], natural[1]), dtype=float
    )
    if float(np.min(plane_weights)) >= -64.0 * np.finfo(float).eps:
        plane_weights = np.maximum(plane_weights, 0.0)
        plane_weights /= float(np.sum(plane_weights))
        projected = plane_weights @ vertices
        candidates.append(
            (float((target - projected) @ (target - projected)), plane_weights, projected)
        )
    for first, second in ((0, 1), (1, 2), (2, 0)):
        edge = vertices[second] - vertices[first]
        length_squared = float(edge @ edge)
        if length_squared <= np.finfo(float).tiny:
            raise ValueError("Qualified S3 contact facet has a zero-length edge")
        parameter = min(
            max(float((target - vertices[first]) @ edge) / length_squared, 0.0),
            1.0,
        )
        weights = np.zeros(3, dtype=float)
        weights[first] = 1.0 - parameter
        weights[second] = parameter
        projected = weights @ vertices
        candidates.append(
            (float((target - projected) @ (target - projected)), weights, projected)
        )
    _distance_squared, selected_weights, surface_point = min(
        candidates,
        key=lambda item: (
            item[0],
            float(item[2][0]),
            float(item[2][1]),
            float(item[2][2]),
        ),
    )
    local = np.asarray((selected_weights[1], selected_weights[2]), dtype=float)
    # Both qualified three-corner formulations use the same exact
    # barycentric contact interpolation.  Keep this orchestration-owned so a
    # strict candidate never enters the generic ShellElement dispatch.
    N = np.asarray(selected_weights, dtype=float)
    dN_dxi = np.asarray((-1.0, 1.0, 0.0), dtype=float)
    dN_deta = np.asarray((-1.0, 0.0, 1.0), dtype=float)
    return local, N, dN_dxi, dN_deta, np.asarray(surface_point, dtype=float)


def _contact_classification(element: ShellElement, local: np.ndarray, tol: float = 1.0e-7) -> str:
    if element.num_nodes in (3, 6):
        bary = (float(local[0]), float(local[1]), float(1.0 - local[0] - local[1]))
        near = sum(1 for value in bary if abs(value) <= tol)
        if near >= 2:
            return "corner"
        if near == 1:
            return "edge"
        return "face"
    on_x = abs(abs(float(local[0])) - 1.0) <= tol
    on_y = abs(abs(float(local[1])) - 1.0) <= tol
    if on_x and on_y:
        return "corner"
    if on_x or on_y:
        return "edge"
    return "face"


def _surface_offset(element: ShellElement, config: SphereContactConfig) -> float:
    thickness = float(getattr(element, "thickness", 0.0) or 0.0)
    normalized_coordinate = 0.0
    if config.contact_surface == "top":
        normalized_coordinate = 1.0
    elif config.contact_surface == "bottom":
        normalized_coordinate = -1.0
    elif config.contact_surface == "signed":
        normalized_coordinate = float(config.signed_surface_offset)
    if _is_qualified_s3_contact_target(element):
        provider = getattr(element, "material_surface_offset_from_reference", None)
        if not callable(provider):
            raise TypeError(
                "Qualified S3 contact target does not expose its material "
                "surface/reference-surface offset policy"
            )
        return float(provider(normalized_coordinate))
    return normalized_coordinate * 0.5 * thickness


_QUALIFIED_S3_FORMULATION_IDS = frozenset(
    {
        "E4_PL_QUALIFIED_S3_COMPANION_V1",
        "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
    }
)


def _is_qualified_s3_contact_target(element: ShellElement) -> bool:
    """Return whether ``element`` uses the formulation-native S3 contract."""

    return bool(
        int(getattr(element, "num_nodes", 0)) == 3
        and str(getattr(element, "formulation_id", ""))
        in _QUALIFIED_S3_FORMULATION_IDS
    )


def _qualified_s3_reference_directors(
    model: "FEModel", element: ShellElement
) -> np.ndarray:
    """Return the element-owned physical reference directors.

    ``reference_normal`` is an owner-orientation hint and need not itself be
    perpendicular to the facet.  The formulation's model-bound API resolves
    the actual facet directors, including that orientation, and is therefore
    the sole linear-contact authority.
    """

    provider = getattr(element, "native_reference_directors", None)
    if not callable(provider):
        raise TypeError(
            "Qualified S3 contact target does not expose its model-bound "
            "reference directors"
        )
    directors = np.asarray(provider(model.mesh), dtype=float)
    if directors.shape != (3, 3) or not np.all(np.isfinite(directors)):
        raise ValueError(
            "Qualified S3 contact requires three finite physical reference "
            "directors"
        )
    norms = np.linalg.norm(directors, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Qualified S3 contact requires unit physical reference directors"
        )
    return directors.copy()


def _qualified_s3_contact_directors(
    model: "FEModel",
    element: ShellElement,
    current_coordinates: np.ndarray,
    native_rotation_context: Optional[Any],
) -> np.ndarray:
    """Resolve authoritative qualified-S3 directors for contact kinematics.

    A missing context denotes the linear/reference configuration.  A supplied
    context must be the active solver-owned nonlinear state transaction.  Its
    generation-checked native view is consumed directly, and the view must be
    bound to the exact translational configuration used by contact.  Contact
    never reconstructs multiplicative rotations from element DOFs.
    """

    reference = _qualified_s3_reference_directors(model, element)
    if native_rotation_context is None:
        return reference
    bound_reference = reference
    active_token_provider = getattr(
        native_rotation_context, "active_trial_token", None
    )
    view_provider = getattr(
        native_rotation_context, "native_element_rotation_view", None
    )
    if not callable(active_token_provider) or not callable(view_provider):
        raise TypeError(
            "Qualified S3 nonlinear contact requires an active solver-owned "
            "NonlinearStateStore context"
        )
    token = active_token_provider()
    view = view_provider(
        token,
        int(element.element_id),
        tuple(int(node_id) for node_id in element.node_ids),
        bound_reference,
    )
    view_nodes = tuple(int(node_id) for node_id in getattr(view, "node_ids", ()))
    expected_nodes = tuple(int(node_id) for node_id in element.node_ids)
    view_coordinates = np.asarray(
        getattr(view, "trial_coordinates", ()), dtype=float
    )
    coordinates = np.asarray(current_coordinates, dtype=float)
    if view_nodes != expected_nodes or (
        view_coordinates.shape != coordinates.shape
        or not np.all(np.isfinite(view_coordinates))
        or not np.array_equal(view_coordinates, coordinates)
    ):
        raise ValueError(
            "Qualified S3 contact coordinates disagree with the active "
            "solver-owned native trial"
        )
    directors = np.asarray(getattr(view, "trial_directors", ()), dtype=float)
    if directors.shape != (3, 3) or not np.all(np.isfinite(directors)):
        raise ValueError(
            "Qualified S3 native contact view has incompatible trial directors"
        )
    norms = np.linalg.norm(directors, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "Qualified S3 native contact view has non-unit trial directors"
        )
    return directors.copy()


def _shell_contact_candidates(model: "FEModel") -> List[ShellElement]:
    return [element for element in model.mesh.elements.values() if isinstance(element, ShellElement)]


def _beam_contact_candidates(model: "FEModel") -> List[Any]:
    from .elements import BeamElement
    from .ge_beam3_element import GeometricallyExactBeam3D3NElement

    # GE-B3 deliberately does not inherit legacy BeamElement.  Admit only the
    # exact qualified public class; private candidates and subclasses cannot
    # become contact targets by structural duck typing.
    return [
        element
        for element in model.mesh.elements.values()
        if isinstance(element, BeamElement)
        or type(element) is GeometricallyExactBeam3D3NElement
    ]


def _beam_contact_radius(element: Any) -> float:
    """Circular contact-radius proxy for a beam section.

    ``cross_section["contact_radius"]`` takes precedence; the default is the
    equivalent-area circle radius, an engineering proxy for the physical
    stiffener/girder profile.
    """
    section = getattr(element, "cross_section", {}) or {}
    explicit = section.get("contact_radius")
    if explicit is not None:
        return max(float(explicit), 0.0)
    area = float(section.get("area", 0.0) or 0.0)
    return float(np.sqrt(max(area, 0.0) / np.pi))


class _ContactGeometry:
    """Mesh-constant shell contact geometry arrays, cached on the mesh revision.

    The contact load assembly runs inside contact/Newton iterations, so anything
    that only depends on the undeformed mesh (node coordinates, translation DOF
    indices, element connectivity, representative edge length) is precomputed
    once and reused until the mesh revision changes.
    """

    __slots__ = (
        "elements",
        "element_ids",
        "qualified_s3_mask",
        "node_ids",
        "node_id_to_slot",
        "node_base",
        "node_dofs",
        "node_rotational_dofs",
        "element_node_slots",
        "element_node_counts",
        "node_mask",
        "representative_edge_length",
        "beam_segment_slots",
        "beam_segment_radii",
        "beam_segment_element_ids",
        "contact_work_local",
    )

    def __init__(self, model: "FEModel"):
        elements = _shell_contact_candidates(model)
        beams = _beam_contact_candidates(model)
        self.elements = elements
        self.element_ids = np.asarray([int(element.element_id) for element in elements], dtype=np.int64)
        self.qualified_s3_mask = np.asarray(
            [_is_qualified_s3_contact_target(element) for element in elements],
            dtype=bool,
        )
        node_id_to_slot: Dict[int, int] = {}
        for element in elements:
            for node_id in element.node_ids:
                node_id_to_slot.setdefault(int(node_id), len(node_id_to_slot))
        for beam in beams:
            for node_id in beam.node_ids:
                node_id_to_slot.setdefault(int(node_id), len(node_id_to_slot))
        self.node_id_to_slot = node_id_to_slot
        self.node_ids = np.asarray(list(node_id_to_slot), dtype=np.int64)
        n_nodes = len(node_id_to_slot)
        self.node_base = np.zeros((max(n_nodes, 1), 3), dtype=float)
        self.node_dofs = np.zeros((max(n_nodes, 1), 3), dtype=np.intp)
        self.node_rotational_dofs = np.zeros((max(n_nodes, 1), 3), dtype=np.intp)
        for node_id, slot in node_id_to_slot.items():
            node = model.mesh.get_node(node_id)
            self.node_base[slot] = (node.x, node.y, node.z)
            self.node_dofs[slot] = np.asarray(node.dofs[:3], dtype=np.intp)
            self.node_rotational_dofs[slot] = np.asarray(
                node.dofs[3:6], dtype=np.intp
            )
        n_elem = len(elements)
        max_nodes = max((element.num_nodes for element in elements), default=1)
        self.element_node_slots = np.zeros((max(n_elem, 1), max_nodes), dtype=np.intp)
        self.element_node_counts = np.zeros(max(n_elem, 1), dtype=np.intp)
        lengths: List[float] = []
        for index, element in enumerate(elements):
            slots = [node_id_to_slot[int(node_id)] for node_id in element.node_ids]
            self.element_node_counts[index] = len(slots)
            self.element_node_slots[index, : len(slots)] = slots
            self.element_node_slots[index, len(slots):] = slots[0]
            corner_count = 3 if element.num_nodes in (3, 6) else 4
            corners = self.node_base[slots[:corner_count]]
            for corner in range(corner_count):
                lengths.append(float(np.linalg.norm(corners[(corner + 1) % corner_count] - corners[corner])))
        self.node_mask = np.arange(max_nodes)[None, :] < self.element_node_counts[:, None]
        self.representative_edge_length = float(np.median(lengths)) if lengths else 0.0
        # Beam contact segments: 2-node beams contribute one segment, 3-node
        # quadratic beams two (end-mid, mid-end).  Only used when
        # SphereContactConfig.beam_contact is enabled.
        segment_slots: List[Tuple[int, int]] = []
        segment_radii: List[float] = []
        segment_element_ids: List[int] = []
        for beam in beams:
            radius = _beam_contact_radius(beam)
            slots = [node_id_to_slot[int(node_id)] for node_id in beam.node_ids]
            if len(slots) == 2:
                pairs = [(slots[0], slots[1])]
            elif len(slots) == 3:
                pairs = [(slots[0], slots[1]), (slots[1], slots[2])]
            else:
                continue
            for pair in pairs:
                segment_slots.append(pair)
                segment_radii.append(radius)
                segment_element_ids.append(int(beam.element_id))
        self.beam_segment_slots = np.asarray(segment_slots, dtype=np.intp).reshape(-1, 2)
        self.beam_segment_radii = np.asarray(segment_radii, dtype=float)
        self.beam_segment_element_ids = np.asarray(segment_element_ids, dtype=np.int64)
        self.contact_work_local = threading.local()

    def deformed_node_positions(self, displacement: Optional[np.ndarray]) -> np.ndarray:
        if displacement is None:
            return self.node_base
        return self.node_base + np.asarray(displacement, dtype=float)[self.node_dofs]

    def element_centroids_and_radii(self, node_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-element deformed node blocks, centroids and bounding radii."""
        element_nodes = node_positions[self.element_node_slots]
        counts = np.maximum(self.element_node_counts, 1)
        sums = np.where(self.node_mask[..., None], element_nodes, 0.0).sum(axis=1)
        centroids = sums / counts[:, None]
        deltas = np.where(self.node_mask[..., None], element_nodes - centroids[:, None, :], 0.0)
        radii = np.sqrt((deltas**2).sum(axis=2)).max(axis=1)
        return element_nodes, centroids, radii

    def active_element_mask(self, deleted_element_ids: Sequence[int]) -> np.ndarray:
        if not deleted_element_ids:
            return np.ones(self.element_ids.shape[0], dtype=bool)
        deleted = np.asarray(sorted({int(element_id) for element_id in deleted_element_ids}), dtype=np.int64)
        return ~np.isin(self.element_ids, deleted)


def _contact_geometry(model: "FEModel") -> _ContactGeometry:
    mesh = model.mesh
    signature = mesh.revision_signature()
    cached = getattr(mesh, "_contact_geometry_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    geometry = _ContactGeometry(model)
    mesh._contact_geometry_cache = (signature, geometry)
    return geometry


def _contact_patch_nodal_weights(
    geometry: _ContactGeometry,
    node_positions: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
    active_mask: np.ndarray,
    contact_point: np.ndarray,
    contact_config: SphereContactConfig,
    sphere: RigidSphereImpact,
    penetration: float,
) -> Dict[int, float]:
    """Return smooth local nodal weights (keyed by node slot) for one contact force."""
    patch_radius = max(
        np.sqrt(max(float(sphere.radius) * float(penetration), 0.0)),
        float(contact_config.load_patch_radius_factor) * max(float(geometry.representative_edge_length), 1.0e-12),
        1.0e-12,
    )
    point = np.asarray(contact_point, dtype=float).reshape(3)
    near = active_mask & (np.linalg.norm(centroids - point[None, :], axis=1) <= patch_radius + radii)
    if not near.any():
        return {}
    slots = np.unique(geometry.element_node_slots[near][geometry.node_mask[near]])
    distances = np.linalg.norm(node_positions[slots] - point[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    slots = slots[order]
    distances = distances[order]
    within = int(np.searchsorted(distances, patch_radius, side="right"))
    min_nodes = min(int(contact_config.min_load_patch_nodes), slots.shape[0])
    selected_count = max(within, min_nodes)
    max_nodes = max(int(contact_config.max_load_patch_nodes), min_nodes)
    selected_count = min(selected_count, max_nodes)
    if selected_count <= 0:
        return {}
    slots = slots[:selected_count]
    selected_distances = distances[:selected_count]
    # Compactly supported kernel: the weight decays to zero at the selection
    # boundary, so a node entering or leaving the patch as the deformed
    # geometry changes does not discontinuously redistribute the contact load.
    # A hard-cutoff kernel (e.g. a truncated Gaussian) makes the contact
    # fixed-point map discontinuous and produces non-converging chatter cycles.
    support_radius = max(patch_radius, float(selected_distances[-1]) * (1.0 + 1.0e-6), 1.0e-12)
    ratios = np.minimum(selected_distances / support_radius, 1.0)
    raw = (1.0 - ratios * ratios) ** 2
    total = float(raw.sum())
    if total <= 0.0:
        raw = np.full(selected_count, 1.0 / selected_count)
        total = float(selected_count) * (1.0 / selected_count)
    weights = raw / total
    return {int(slot): float(weight) for slot, weight in zip(slots, weights)}


def _representative_shell_edge_length(model: "FEModel") -> float:
    return _contact_geometry(model).representative_edge_length


def material_characteristic_modulus(material: Any, structural_kind: str = "shell") -> float:
    """Return a numerical stiffness scale, not a constitutive modulus.

    Orthotropic shell contact uses ``max(E1, E2)`` and orthotropic beam
    numerics use ``E1``.  Isotropic materials retain ``E``.  General
    anisotropy is deliberately rejected until an arbitrary constitutive
    scaling policy exists.
    """

    kind = str(structural_kind).strip().lower()
    if kind not in {"shell", "beam"}:
        raise ValueError("structural_kind must be 'shell' or 'beam'")
    value = (
        shell_characteristic_modulus(material)
        if kind == "shell"
        else beam_material_properties(material).characteristic_modulus
    )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{kind} characteristic modulus must be finite and positive")
    return value


def _shell_element_characteristic_modulus(model: "FEModel", element: ShellElement) -> float:
    """Return the shell's numerical modulus scale, including section laws."""

    section = getattr(element, "shell_section", None)
    if section is None:
        return material_characteristic_modulus(
            model.get_material(element.material_name),
            "shell",
        )
    membrane = np.asarray(getattr(section, "A"), dtype=float)
    stiffness = float(np.max(np.linalg.eigvalsh(0.5 * (membrane + membrane.T))))
    thickness = float(getattr(element, "thickness", 0.0))
    value = stiffness / max(thickness, 1.0e-30)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Generalized shell-section characteristic modulus must be finite "
            "and positive"
        )
    return value


def _representative_shell_stiffness(model: "FEModel") -> float:
    """Median shell membrane stiffness scale ``E_char * t`` in N/m."""
    values: List[float] = []
    for element in _shell_contact_candidates(model):
        values.append(
            _shell_element_characteristic_modulus(model, element)
            * float(element.thickness)
        )
    return float(np.median(values)) if values else 0.0


def recommend_sphere_contact_penalty(
    model: "FEModel",
    sphere: RigidSphereImpact,
    target_penetration_fraction: float = 0.01,
    safety_factor: float = 1.0,
) -> float:
    """Recommend a conservative normal penalty stiffness [N/m] for v1 contact.

    Two dimensionally consistent stiffness scales are combined:

    - impact energy: ``k = m v^2 / delta^2`` stops the sphere's kinetic energy
      within the target penetration ``delta = R * target_penetration_fraction``
      (from ``1/2 k delta^2 = 1/2 m v^2``);
    - structural: the representative shell membrane stiffness ``E t`` [N/m]
      divided by the target penetration fraction, so the penalty spring is much
      stiffer than the contacted structure and the penetration stays a small
      fraction of the structural response.
    """

    if target_penetration_fraction <= 0.0:
        raise ValueError("target_penetration_fraction must be positive")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive")
    target_penetration = max(float(sphere.radius) * float(target_penetration_fraction), 1.0e-12)
    impact_stiffness = float(sphere.mass) * max(float(sphere.speed), 1.0e-9) ** 2 / target_penetration**2
    shell_stiffness = _representative_shell_stiffness(model) / max(float(target_penetration_fraction), 1.0e-12)
    return float(safety_factor) * max(impact_stiffness, shell_stiffness, 1.0)


def _resolved_contact_config(model: "FEModel", sphere: RigidSphereImpact, config: Optional[SphereContactConfig]) -> SphereContactConfig:
    raw = config or SphereContactConfig()
    if raw.penalty_stiffness is not None:
        return raw
    return SphereContactConfig(
        penalty_stiffness=recommend_sphere_contact_penalty(
            model,
            sphere,
            target_penetration_fraction=raw.target_penetration_fraction,
            safety_factor=raw.auto_penalty_safety_factor,
        ),
        contact_damping=raw.contact_damping,
        search_margin=raw.search_margin,
        max_contact_iterations=raw.max_contact_iterations,
        penetration_tolerance=raw.penetration_tolerance,
        force_tolerance=raw.force_tolerance,
        side_mode=raw.side_mode,
        contact_surface=raw.contact_surface,
        signed_surface_offset=raw.signed_surface_offset,
        target_penetration_fraction=raw.target_penetration_fraction,
        auto_penalty_safety_factor=raw.auto_penalty_safety_factor,
        max_sphere_travel_fraction=raw.max_sphere_travel_fraction,
        max_event_substeps=raw.max_event_substeps,
        max_active_contacts=raw.max_active_contacts,
        load_patch_radius_factor=raw.load_patch_radius_factor,
        min_load_patch_nodes=raw.min_load_patch_nodes,
        max_load_patch_nodes=raw.max_load_patch_nodes,
        save_contact_history=raw.save_contact_history,
        post_separation_time=raw.post_separation_time,
        contact_relaxation=raw.contact_relaxation,
        beam_contact=raw.beam_contact,
    )


def _contact_issue(
    code: str,
    severity: str,
    entity_type: str,
    entity_id: Optional[int],
    message: str,
    measured: Optional[float] = None,
    limit: Optional[float] = None,
    suggestion: str = "",
) -> ProductionValidationIssue:
    return ProductionValidationIssue(code, severity, entity_type, entity_id, message, measured, limit, suggestion)


def validate_contact_configuration(
    model: "FEModel",
    sphere: RigidSphereImpact,
    contact_config: Optional[SphereContactConfig],
    transient_config: TransientConfig,
) -> ProductionValidationReport:
    """Validate limited rigid-sphere contact inputs before production use."""

    issues: List[ProductionValidationIssue] = []
    shells = _shell_contact_candidates(model)
    beam_targets = bool(contact_config is not None and contact_config.beam_contact) and bool(_beam_contact_candidates(model))
    if not shells and not beam_targets:
        issues.append(
            _contact_issue(
                "CONTACT001",
                "error",
                "model",
                None,
                "Sphere contact requires at least one shell target element.",
                suggestion="Add shell elements or use a different load/contact model.",
            )
        )
    used_materials = {element.material_name for element in model.mesh.elements.values() if hasattr(element, "material_name")}
    for name in sorted(used_materials):
        material = model.get_material(name)
        if float(material.density) <= 0.0:
            issues.append(
                _contact_issue(
                    "CONTACT002",
                    "error",
                    "material",
                    None,
                    f"Material {name!r} has non-positive density for transient contact.",
                    measured=float(material.density),
                    limit=0.0,
                    suggestion="Set density on all structural materials used by the impact model.",
                )
            )
    if contact_config is not None and contact_config.side_mode != "both":
        issues.append(
            _contact_issue(
                "CONTACT003",
                "error",
                "contact_config",
                None,
                "Only side_mode='both' is supported for limited sphere-shell contact.",
                suggestion="Use side_mode='both'.",
            )
        )
    config = _resolved_contact_config(model, sphere, contact_config)
    if transient_config.dt <= 0.0:
        issues.append(_contact_issue("CONTACT004", "error", "transient_config", None, "Transient dt must be positive."))
    else:
        travel = float(sphere.speed) * float(transient_config.dt)
        limit = 0.5 * float(sphere.radius)
        effective_travel = travel / max(int(config.max_event_substeps), 1)
        if effective_travel > limit:
            issues.append(
                _contact_issue(
                    "CONTACT005",
                    "error",
                    "transient_config",
                    None,
                    "Sphere travel per time step is too large for production contact without event substepping.",
                    measured=effective_travel,
                    limit=limit,
                    suggestion="Reduce dt or increase event substepping controls.",
                )
            )
    period_dt = 0.2 * np.sqrt(float(sphere.mass) / float(config.penalty_stiffness))
    if transient_config.dt > period_dt:
        issues.append(
            _contact_issue(
                "CONTACT006",
                "warning",
                "transient_config",
                None,
                "Time step is large relative to the contact penalty period; the linear "
                "impact solver auto-substeps during contact to compensate (capped by "
                "max_event_substeps).",
                measured=float(transient_config.dt),
                limit=float(period_dt),
                suggestion="No action needed unless max_event_substeps is reached; then reduce dt or lower penalty stiffness.",
            )
        )
    mesh_quality = {
        "shell_contact_targets": len(shells),
        "representative_shell_edge_length": _representative_shell_edge_length(model),
        "recommended_penalty_stiffness": float(config.penalty_stiffness),
        "sphere_travel_per_step": float(sphere.speed) * float(transient_config.dt),
        "sphere_travel_radius_ratio": float(sphere.speed) * float(transient_config.dt) / max(float(sphere.radius), 1.0e-12),
    }
    status = "invalid" if any(issue.severity == "error" for issue in issues) else ("warning" if issues else "ok")
    return ProductionValidationReport(status, tuple(issues), mesh_quality, model.mesh.revision_signature())


def _assemble_sphere_contact_work_buffer(
    model: "FEModel",
    sphere: RigidSphereImpact,
    contact_config: SphereContactConfig,
    sphere_position: np.ndarray,
    sphere_velocity: np.ndarray,
    structural_displacement: Optional[np.ndarray] = None,
    structural_velocity: Optional[np.ndarray] = None,
    deleted_element_ids: Sequence[int] = (),
    contact_scale_by_element: Optional[Mapping[int, float]] = None,
    preferred_element_ids: Sequence[int] = (),
    work_buffer: Optional[ContactWorkBuffer] = None,
    work_counters: Optional[ContactWorkCounters] = None,
    native_rotation_context: Optional[Any] = None,
) -> ContactWorkBuffer:
    """Assemble contact into reusable compact work storage.

    ``preferred_element_ids`` stabilizes the ``max_active_contacts`` reduction:
    adjacent coplanar elements report near-identical penetrations for one
    physical contact, and a bare deepest-penetration argmax then flips between
    them from iteration to iteration, which makes the contact fixed-point map
    discontinuous and can lock the iteration into a non-converging chatter
    cycle.  Within a small penetration tie band the previously selected
    elements are kept instead.
    """

    total_dofs = model.mesh.dof_manager.total_dofs
    u = None if structural_displacement is None else np.asarray(structural_displacement, dtype=float)
    v = np.zeros(total_dofs, dtype=float) if structural_velocity is None else np.asarray(structural_velocity, dtype=float)
    center = np.asarray(sphere_position, dtype=float).reshape(3)
    velocity = np.asarray(sphere_velocity, dtype=float).reshape(3)
    contact_work = work_buffer or ContactWorkBuffer(total_dofs, counters=work_counters)
    contact_work.reset(total_dofs)
    contact_scales = {} if contact_scale_by_element is None else {int(k): float(v) for k, v in contact_scale_by_element.items()}
    deleted_ids = {int(element_id) for element_id in deleted_element_ids}
    element_activity = getattr(model.mesh, "element_activity", None)
    if element_activity is not None:
        activity_ids = np.asarray(
            element_activity.element_ids, dtype=np.int64
        ).reshape(-1)
        activity_scales = np.asarray(
            element_activity.contact_scales(activity_ids), dtype=float
        ).reshape(-1)
        for element_id, scale in zip(activity_ids, activity_scales):
            identifier = int(element_id)
            contact_scales[identifier] = (
                float(contact_scales.get(identifier, 1.0)) * float(scale)
            )
        deleted_ids.update(
            int(element_id)
            for element_id in activity_ids[
                np.asarray(element_activity.hard_deleted_mask, dtype=bool)
            ]
        )
    deleted_element_ids = tuple(sorted(deleted_ids))
    geometry = _contact_geometry(model)
    has_beam_targets = bool(contact_config.beam_contact) and geometry.beam_segment_slots.shape[0] > 0
    if geometry.element_ids.shape[0] == 0 and not has_beam_targets:
        contact_work.select_and_scatter(
            max_active_contacts=int(contact_config.max_active_contacts),
            preferred_element_ids=preferred_element_ids,
            node_dofs=geometry.node_dofs,
            node_rotational_dofs=geometry.node_rotational_dofs,
        )
        return contact_work
    node_positions = geometry.deformed_node_positions(u)
    element_nodes, centroids, radii = geometry.element_centroids_and_radii(node_positions)
    active_mask = geometry.active_element_mask(deleted_element_ids)
    if geometry.element_ids.shape[0] == 0:
        near_mask = np.zeros(0, dtype=bool)
    else:
        qualified_s3_offset_allowance: float | np.ndarray = 0.0
        if np.any(geometry.qualified_s3_mask):
            qualified_s3_offset_allowance = np.zeros_like(radii)
            qualified_indices = np.flatnonzero(geometry.qualified_s3_mask)
            qualified_s3_offset_allowance[qualified_indices] = np.asarray(
                [
                    abs(_surface_offset(geometry.elements[index], contact_config))
                    for index in qualified_indices
                ],
                dtype=float,
            )
        near_mask = active_mask & (
            np.linalg.norm(centroids - center[None, :], axis=1)
            <= sphere.radius
            + radii
            + contact_config.search_margin
            + qualified_s3_offset_allowance
        )

    for element_index in np.flatnonzero(near_mask):
        element = geometry.elements[element_index]
        contact_scale = float(contact_scales.get(int(element.element_id), 1.0))
        if contact_scale <= 0.0:
            continue
        node_count = int(geometry.element_node_counts[element_index])
        element_slots = geometry.element_node_slots[element_index, :node_count]
        coords = element_nodes[element_index, :node_count]
        offset = _surface_offset(element, contact_config)
        qualified_s3 = _is_qualified_s3_contact_target(element)
        qualified_s3_directors: Optional[np.ndarray] = None
        projection_coordinates = coords
        if qualified_s3:
            qualified_s3_directors = _qualified_s3_contact_directors(
                model,
                element,
                coords,
                native_rotation_context,
            )
            if offset != 0.0:
                # The MITC3+ physical surface is interpolated from the three
                # offset corner positions.  Keeping the director
                # interpolation unnormalised makes its variation exactly
                # conjugate to the nodal rotational moments scattered below.
                projection_coordinates = (
                    coords + offset * qualified_s3_directors
                )
        if qualified_s3:
            local, N, dN_dxi, dN_deta, surface_point = (
                _project_qualified_s3_contact_coordinates(
                    element, projection_coordinates, center
                )
            )
        else:
            local, N, dN_dxi, dN_deta, surface_point = (
                _project_local_coordinates(element, projection_coordinates, center)
            )
        gap_vector = center - surface_point
        distance = float(np.linalg.norm(gap_vector))
        if distance > sphere.radius + contact_config.search_margin:
            continue
        tangent_xi = dN_dxi @ projection_coordinates
        tangent_eta = dN_deta @ projection_coordinates
        fallback_normal = np.cross(tangent_xi, tangent_eta)
        fallback_norm = float(np.linalg.norm(fallback_normal))
        if fallback_norm <= 1.0e-14:
            continue
        surface_normal = fallback_normal / fallback_norm
        if qualified_s3:
            assert qualified_s3_directors is not None
            physical_director = np.asarray(N, dtype=float) @ qualified_s3_directors
            physical_director_norm = float(np.linalg.norm(physical_director))
            if physical_director_norm <= 1.0e-14:
                raise ValueError(
                    "Qualified S3 contact has a singular interpolated physical director"
                )
            physical_director /= physical_director_norm
            if float(surface_normal @ physical_director) < 0.0:
                surface_normal = -surface_normal
        if offset != 0.0 and not qualified_s3:
            surface_point = surface_point + offset * surface_normal
            gap_vector = center - surface_point
            distance = float(np.linalg.norm(gap_vector))
            if distance > sphere.radius + contact_config.search_margin:
                continue
        if distance > 1.0e-14:
            normal = gap_vector / distance
        elif fallback_norm > 1.0e-14:
            normal = surface_normal
        else:
            continue
        penetration = max(float(sphere.radius - distance), 0.0)
        if penetration <= 0.0:
            continue
        surface_velocity = (
            np.asarray(N, dtype=float)
            @ v[geometry.node_dofs[element_slots]]
        )
        if qualified_s3 and offset != 0.0:
            assert qualified_s3_directors is not None
            angular_velocities = v[
                geometry.node_rotational_dofs[element_slots]
            ]
            surface_velocity = surface_velocity + offset * np.sum(
                np.asarray(N, dtype=float)[:, None]
                * np.cross(angular_velocities, qualified_s3_directors),
                axis=0,
            )
        relative_normal_velocity = float(np.dot(velocity - surface_velocity, normal))
        normal_force = max(
            float(contact_config.penalty_stiffness) * penetration - float(contact_config.contact_damping) * relative_normal_velocity,
            0.0,
        )
        normal_force *= min(max(contact_scale, 0.0), 1.0)
        if normal_force <= 0.0:
            continue
        sphere_force = normal_force * normal
        structure_force = -sphere_force
        if qualified_s3:
            patch_weights = {int(slot): float(N[local_index]) for local_index, slot in enumerate(element_slots)}
        else:
            patch_weights = _contact_patch_nodal_weights(
                geometry,
                node_positions,
                centroids,
                radii,
                active_mask,
                surface_point,
                contact_config,
                sphere,
                penetration,
            )
            if not patch_weights:
                patch_weights = {
                    int(slot): float(N[local_index])
                    for local_index, slot in enumerate(element_slots)
                }
        nodal_slots = tuple(int(slot) for slot in patch_weights)
        nodal_forces = np.asarray(
            [float(patch_weights[slot]) * structure_force for slot in nodal_slots],
            dtype=float,
        )
        nodal_moments = np.zeros_like(nodal_forces)
        if qualified_s3 and offset != 0.0:
            assert qualified_s3_directors is not None
            local_index_by_slot = {
                int(slot): int(local_index)
                for local_index, slot in enumerate(element_slots)
            }
            nodal_moments = np.asarray(
                [
                    offset
                    * np.cross(
                        qualified_s3_directors[local_index_by_slot[slot]],
                        nodal_forces[index],
                    )
                    for index, slot in enumerate(nodal_slots)
                ],
                dtype=float,
            )
        contact_work.append(
            element_id=int(element.element_id),
            local_coordinates=(float(local[0]), float(local[1])),
            contact_point=surface_point,
            normal=normal,
            penetration=penetration,
            normal_force=normal_force,
            sphere_force=sphere_force,
            structure_force=structure_force,
            contact_classification=_contact_classification(element, local),
            nodal_slots=nodal_slots,
            nodal_forces=nodal_forces,
            nodal_moments=nodal_moments,
        )

    if bool(contact_config.beam_contact) and geometry.beam_segment_slots.shape[0]:
        deleted = {int(element_id) for element_id in deleted_element_ids}
        seg_a = node_positions[geometry.beam_segment_slots[:, 0]]
        seg_b = node_positions[geometry.beam_segment_slots[:, 1]]
        axis = seg_b - seg_a
        length_sq = np.maximum(np.einsum("ij,ij->i", axis, axis), 1.0e-30)
        t_param = np.clip(np.einsum("ij,ij->i", center[None, :] - seg_a, axis) / length_sq, 0.0, 1.0)
        closest = seg_a + t_param[:, None] * axis
        gap_vectors = center[None, :] - closest
        distances = np.linalg.norm(gap_vectors, axis=1)
        reach = sphere.radius + geometry.beam_segment_radii + contact_config.search_margin
        for segment_index in np.flatnonzero(distances <= reach):
            beam_id = int(geometry.beam_segment_element_ids[segment_index])
            if beam_id in deleted:
                continue
            contact_scale = float(contact_scales.get(beam_id, 1.0))
            if contact_scale <= 0.0:
                continue
            distance = float(distances[segment_index])
            penetration = float(sphere.radius + geometry.beam_segment_radii[segment_index] - distance)
            if penetration <= 0.0 or distance <= 1.0e-14:
                continue
            normal = gap_vectors[segment_index] / distance
            t_value = float(t_param[segment_index])
            slot_a = int(geometry.beam_segment_slots[segment_index, 0])
            slot_b = int(geometry.beam_segment_slots[segment_index, 1])
            surface_velocity = (1.0 - t_value) * v[geometry.node_dofs[slot_a]] + t_value * v[geometry.node_dofs[slot_b]]
            relative_normal_velocity = float(np.dot(velocity - surface_velocity, normal))
            normal_force = max(
                float(contact_config.penalty_stiffness) * penetration
                - float(contact_config.contact_damping) * relative_normal_velocity,
                0.0,
            )
            normal_force *= min(max(contact_scale, 0.0), 1.0)
            if normal_force <= 0.0:
                continue
            sphere_force = normal_force * normal
            structure_force = -sphere_force
            contact_work.append(
                element_id=beam_id,
                local_coordinates=(2.0 * t_value - 1.0, 0.0),
                contact_point=closest[segment_index],
                normal=normal,
                penetration=penetration,
                normal_force=normal_force,
                sphere_force=sphere_force,
                structure_force=structure_force,
                contact_classification="beam",
                nodal_slots=(slot_a, slot_b),
                nodal_forces=np.asarray(
                    [
                        (1.0 - t_value) * structure_force,
                        t_value * structure_force,
                    ],
                    dtype=float,
                ),
            )

    contact_work.select_and_scatter(
        max_active_contacts=int(contact_config.max_active_contacts),
        preferred_element_ids=preferred_element_ids,
        node_dofs=geometry.node_dofs,
        node_rotational_dofs=geometry.node_rotational_dofs,
    )
    return contact_work


def assemble_sphere_contact_load_vector(
    model: "FEModel",
    sphere: RigidSphereImpact,
    contact_config: SphereContactConfig,
    sphere_position: np.ndarray,
    sphere_velocity: np.ndarray,
    structural_displacement: Optional[np.ndarray] = None,
    structural_velocity: Optional[np.ndarray] = None,
    deleted_element_ids: Sequence[int] = (),
    contact_scale_by_element: Optional[Mapping[int, float]] = None,
    preferred_element_ids: Sequence[int] = (),
    native_rotation_context: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[SphereContactRecord, ...]]:
    """Assemble contact loads and materialize the stable public records.

    ``native_rotation_context`` is reserved for a currently active
    solver-owned nonlinear-state transaction.  Omitting it selects linear
    reference-director kinematics; passing any element-owned or stale
    rotation surrogate is rejected by the qualified-S3 resolver.
    """

    geometry = _contact_geometry(model)
    public_work_buffer = getattr(geometry.contact_work_local, "buffer", None)
    if public_work_buffer is None:
        public_work_buffer = ContactWorkBuffer(model.mesh.dof_manager.total_dofs)
        geometry.contact_work_local.buffer = public_work_buffer
    contact_work = _assemble_sphere_contact_work_buffer(
        model,
        sphere,
        contact_config,
        sphere_position,
        sphere_velocity,
        structural_displacement=structural_displacement,
        structural_velocity=structural_velocity,
        deleted_element_ids=deleted_element_ids,
        contact_scale_by_element=contact_scale_by_element,
        preferred_element_ids=preferred_element_ids,
        work_buffer=public_work_buffer,
        native_rotation_context=native_rotation_context,
    )
    records = contact_work.materialize_records(SphereContactRecord, geometry.node_ids)
    return contact_work.load.copy(), contact_work.sphere_force.copy(), records


def _contact_default_config(sphere: RigidSphereImpact) -> SphereContactConfig:
    stiffness = max(float(sphere.mass) * max(float(sphere.speed), 1.0) ** 2 / max(float(sphere.radius) * 0.01, 1.0e-9), 1.0)
    return SphereContactConfig(penalty_stiffness=stiffness)


LinearElementMatrixTerms = Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _contact_erodible_element_count(model: "FEModel") -> int:
    """Count elements that impact fracture can erode (shell midsurfaces + beams)."""
    return sum(1 for element in model.mesh.elements.values() if element_fracture_category(element) is not None)


def _linear_element_matrix_terms(model: "FEModel") -> Tuple[int, Tuple[LinearElementMatrixTerms, ...]]:
    """Cache unscaled element K/M contributions for repeated damage rescaling."""
    terms: List[LinearElementMatrixTerms] = []
    total_dofs = model.mesh.dof_manager.total_dofs
    for element_id, element in model.mesh.elements.items():
        if not hasattr(element, "get_dof_mapping"):
            continue
        mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp).reshape(-1)
        if mapping.size == 0:
            continue
        material = model.get_material(element.material_name)
        k_elem = np.asarray(element.compute_stiffness_matrix(model.mesh, material), dtype=float)
        m_elem = np.asarray(element.compute_mass_matrix(model.mesh, material), dtype=float)
        if k_elem.shape != (mapping.size, mapping.size) or m_elem.shape != (mapping.size, mapping.size):
            continue
        row = np.repeat(mapping, mapping.size)
        col = np.tile(mapping, mapping.size)
        terms.append((int(element_id), row, col, k_elem.reshape(-1), m_elem.reshape(-1)))
    return total_dofs, tuple(terms)


def _assemble_damaged_linear_matrices(
    model: "FEModel",
    element_scales: Mapping[int, float],
    cached_terms: Optional[Tuple[int, Tuple[LinearElementMatrixTerms, ...]] | DamageMatrixPlan] = None,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Assemble K and M with per-element damage stiffness/mass scales."""
    if isinstance(cached_terms, DamageMatrixPlan):
        try:
            return cached_terms.update(model, element_scales)
        except DamageMatrixPlanFallback:
            # Revisions can change through a progress callback.  Preserve the
            # scalar helper contract and exact current-model semantics instead
            # of using a stale contribution map.
            cached_terms = None
    if cached_terms is None:
        cached_terms = _linear_element_matrix_terms(model)
    total_dofs, terms = cached_terms
    scales = {int(element_id): float(scale) for element_id, scale in element_scales.items()}
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    k_data: List[np.ndarray] = []
    m_data: List[np.ndarray] = []
    for element_id, row, col, k_flat, m_flat in terms:
        scale = min(max(float(scales.get(int(element_id), 1.0)), 0.0), 1.0)
        rows.append(row)
        cols.append(col)
        k_data.append(scale * k_flat)
        m_data.append(scale * m_flat)
    if not rows:
        empty = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        return empty, empty
    row_all = np.concatenate(rows)
    col_all = np.concatenate(cols)
    K = sparse.coo_matrix((np.concatenate(k_data), (row_all, col_all)), shape=(total_dofs, total_dofs)).tocsr()
    M = sparse.coo_matrix((np.concatenate(m_data), (row_all, col_all)), shape=(total_dofs, total_dofs)).tocsr()
    # Preserve added point masses across erosion-driven matrix rebuilds.
    point_masses = getattr(model.mesh, "point_masses", None)
    if point_masses:
        diagonal = np.zeros(total_dofs, dtype=float)
        for node_id, mass in point_masses.items():
            node = model.mesh.get_node(int(node_id))
            if node is None or float(mass) == 0.0:
                continue
            for axis in range(3):
                diagonal[node.dofs[axis]] += float(mass)
        if diagonal.any():
            M = (M + sparse.diags(diagonal, 0, shape=(total_dofs, total_dofs), format="csr")).tocsr()
    return K, M


def _assemble_damaged_linear_matrices_with_gate(
    model: "FEModel",
    element_scales: Mapping[int, float],
    gate: DamageMatrixPlanGate,
    *,
    opportunity_index: int,
    cached_terms: Optional[Tuple[int, Tuple[LinearElementMatrixTerms, ...]]] = None,
    plan: Optional[DamageMatrixPlan] = None,
) -> Tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    Tuple[int, Tuple[LinearElementMatrixTerms, ...]],
    Optional[DamageMatrixPlan],
]:
    """Select the incremental plan only after its production gates pass."""

    cached_terms_stale = (
        cached_terms is not None
        and gate.cached_terms_registered
        and not gate.cached_terms_match(model)
    )
    if cached_terms is None or cached_terms_stale:
        cached_terms = _linear_element_matrix_terms(model)
        gate.register_cached_terms(model, replacing_stale=cached_terms_stale)
        if cached_terms_stale:
            # A model revision invalidates both representations.  The freshly
            # generated terms retain exact current-model scalar semantics; do
            # not pay another incremental-plan setup in this analysis.
            plan = None
    elif not gate.cached_terms_registered:
        gate.register_cached_terms(model)

    use_plan = gate.consider(cached_terms, opportunity_index=int(opportunity_index))
    if use_plan and plan is None:
        try:
            candidate = DamageMatrixPlan.build(model, cached_terms)
        except MemoryError:
            gate.reject_plan_setup("plan_setup_memory_error")
        else:
            if gate.accept_built_plan(candidate):
                plan = candidate

    if plan is not None:
        try:
            stiffness, mass = plan.update(model, element_scales)
        except DamageMatrixPlanFallback:
            revision_changed = not plan.is_valid(model)
            gate.record_plan_update_fallback(plan.fallback_reason or "plan_update_fallback")
            if revision_changed:
                cached_terms = _linear_element_matrix_terms(model)
                gate.register_cached_terms(model, replacing_stale=True)
            plan = None
            stiffness, mass = _assemble_damaged_linear_matrices(
                model,
                element_scales,
                cached_terms=cached_terms,
            )
            gate.record_update(used_plan=False)
        else:
            gate.record_update(used_plan=True)
    else:
        stiffness, mass = _assemble_damaged_linear_matrices(
            model,
            element_scales,
            cached_terms=cached_terms,
        )
        gate.record_update(used_plan=False)
    return stiffness, mass, cached_terms, plan


def _assemble_eroded_linear_matrices(
    model: "FEModel",
    deleted_element_ids: Sequence[int],
    residual_fraction: float,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Assemble K and M with residual contribution from deleted elements."""
    scales = {int(element_id): float(residual_fraction) for element_id in deleted_element_ids}
    return _assemble_damaged_linear_matrices(model, scales)


def _impact_contact_patch_area(
    record: SphereContactRecord,
    element: ShellElement,
    config: ImpactDamageConfig,
    sphere: RigidSphereImpact,
) -> float:
    """Estimate the local shell contact patch area used by damage checks."""
    penetration_radius = np.sqrt(max(float(sphere.radius) * float(record.penetration), 0.0))
    fraction_radius = max(float(sphere.radius) * float(config.contact_area_radius_fraction), 0.0)
    thickness_radius = max(float(getattr(element, "thickness", 0.0)), 0.0)
    min_radius = np.sqrt(float(config.min_contact_area) / np.pi)
    patch_radius = max(penetration_radius, fraction_radius, thickness_radius, min_radius)
    return max(float(np.pi * patch_radius**2), float(config.min_contact_area))


def _impact_material_capacity(model: "FEModel", element: ShellElement, config: ImpactDamageConfig) -> Tuple[float, str]:
    """Return yield/ultimate/user stress capacity in Pa for impact damage."""
    if config.capacity_basis == "user":
        return float(config.user_capacity), "user"
    if getattr(element, "shell_section", None) is not None:
        raise ValueError(
            "Generalized shell-section linear impact pressure damage requires "
            "ImpactDamageConfig(capacity_basis='user', user_capacity=...)."
        )
    material = model.get_material(element.material_name)
    symmetry = material_symmetry(material)
    if symmetry == "orthotropic":
        raise ValueError(
            "Orthotropic linear impact pressure damage requires "
            "ImpactDamageConfig(capacity_basis='user', user_capacity=...)."
        )
    curve = getattr(material, "hardening_curve", None)
    if config.capacity_basis == "yield":
        if curve is not None and hasattr(curve, "sigma_yield"):
            return float(curve.sigma_yield), "hardening_curve.sigma_yield"
        if float(getattr(material, "yield_stress", 0.0)) > 0.0:
            return float(material.yield_stress), "material.yield_stress"
    else:
        if curve is not None:
            candidates = []
            for name in ("sigma_yield_2", "sigma_yield"):
                value = getattr(curve, name, None)
                if value is not None:
                    candidates.append(float(value))
            if hasattr(curve, "flow_stress"):
                try:
                    candidates.append(float(np.asarray(curve.flow_stress(np.asarray([0.05], dtype=float))).reshape(-1)[0]))
                except Exception:
                    pass
            if candidates:
                return max(candidates), "hardening_curve.ultimate_proxy"
        if float(getattr(material, "yield_stress", 0.0)) > 0.0:
            return 1.15 * float(material.yield_stress), "material.yield_stress_ultimate_proxy"
    elastic_proxy = max(0.002 * material_characteristic_modulus(material, "shell"), 1.0)
    return elastic_proxy, "elastic_0.2pct_proxy"


def _impact_damage_metrics(
    model: "FEModel",
    record: SphereContactRecord,
    config: ImpactDamageConfig,
    sphere: RigidSphereImpact,
    dt: float,
) -> Dict[str, Any]:
    element = model.mesh.get_element(int(record.element_id))
    if not isinstance(element, ShellElement):
        return {}
    area = _impact_contact_patch_area(record, element, config, sphere)
    capacity, capacity_source = _impact_material_capacity(model, element, config)
    pressure = float(record.normal_force) / max(area, 1.0e-30)
    impulse_density = float(record.normal_force) * float(dt) / max(area, 1.0e-30)
    pressure_utilization = pressure / max(capacity, 1.0e-30)
    impulse_utilization = impulse_density / max(capacity * float(config.impulse_reference_time), 1.0e-30)
    elastic_modulus = _shell_element_characteristic_modulus(model, element)
    equivalent_plastic_strain_estimate = max(pressure - capacity, 0.0) / elastic_modulus * float(config.strain_scale)
    strain_utilization = equivalent_plastic_strain_estimate / max(float(config.plastic_strain_capacity), 1.0e-30)
    components = {
        "contact_pressure": float(pressure_utilization),
        "impulse_per_area": float(impulse_utilization),
        "equivalent_plastic_strain_estimate": float(strain_utilization),
    }
    governing_component = max(components, key=components.get)
    combined = max(float(value) for value in components.values())
    return {
        "element_id": int(record.element_id),
        "contact_patch_area": float(area),
        "contact_pressure": float(pressure),
        "impulse_density_increment": float(impulse_density),
        "capacity": float(capacity),
        "capacity_source": capacity_source,
        "equivalent_plastic_strain_estimate": float(equivalent_plastic_strain_estimate),
        "utilizations": components,
        "combined_utilization": float(combined),
        "governing_component": governing_component,
    }


def _impact_damage_scale(damage: float, config: ImpactDamageConfig) -> float:
    if damage < float(config.softening_start):
        return 1.0
    if damage >= float(config.delete_at):
        return float(config.residual_stiffness_fraction)
    span = max(float(config.delete_at) - float(config.softening_start), 1.0e-12)
    fraction = (float(damage) - float(config.softening_start)) / span
    return float(1.0 - fraction * (1.0 - float(config.residual_stiffness_fraction)))


def _update_impact_damage_states(
    model: "FEModel",
    records: Sequence[SphereContactRecord],
    config: ImpactDamageConfig,
    sphere: RigidSphereImpact,
    states: Dict[int, Dict[str, Any]],
    deleted_element_ids: Sequence[int],
    *,
    step_index: int,
    time_value: float,
    dt: float,
) -> Tuple[Tuple[DeletedElementRecord, ...], float, Tuple[Dict[str, Any], ...], bool]:
    deleted = {int(element_id) for element_id in deleted_element_ids}
    new_deleted: List[DeletedElementRecord] = []
    diagnostics: List[Dict[str, Any]] = []
    changed = False
    max_utilization = 0.0
    for record in records:
        element_id = int(record.element_id)
        if element_id in deleted:
            continue
        element = model.mesh.get_element(element_id)
        # Capacity-based impact damage is an area/contact-pressure screening
        # model, which maps to shell midsurface contact only.  Beam line
        # contact (force per length) is out of scope here; use
        # ImpactFractureConfig (contact force/penetration) or the nonlinear
        # PlasticImpactDamageConfig path for beam erosion.
        if element is None or element_fracture_category(element) != "shell":
            continue
        metrics = _impact_damage_metrics(model, record, config, sphere, dt)
        if not metrics:
            continue
        utilization = float(metrics["combined_utilization"])
        max_utilization = max(max_utilization, utilization)
        state = states.setdefault(
            element_id,
            {
                "damage": 0.0,
                "max_utilization": 0.0,
                "governing_component": "",
                "accumulated_impulse_density": 0.0,
                "max_contact_patch_area": 0.0,
                "contact_count": 0,
                "history": [],
                "deleted_time": None,
            },
        )
        old_damage = float(state["damage"])
        old_scale = float(state.get("scale", _impact_damage_scale(old_damage, config)))
        state["contact_count"] = int(state.get("contact_count", 0)) + 1
        state["max_utilization"] = max(float(state.get("max_utilization", 0.0)), utilization)
        if utilization >= float(state.get("max_utilization", 0.0)) - 1.0e-12:
            state["governing_component"] = str(metrics["governing_component"])
        state["accumulated_impulse_density"] = float(state.get("accumulated_impulse_density", 0.0)) + float(
            metrics["impulse_density_increment"]
        )
        state["max_contact_patch_area"] = max(float(state.get("max_contact_patch_area", 0.0)), float(metrics["contact_patch_area"]))
        if config.mode == "instant_threshold":
            new_damage = max(old_damage, utilization / float(config.damage_threshold))
        else:
            increment = max(utilization / float(config.damage_threshold), 0.0) * float(dt) / float(config.impulse_reference_time)
            new_damage = old_damage + increment
        state["damage"] = float(new_damage)
        scale = _impact_damage_scale(new_damage, config)
        state["scale"] = float(scale)
        diag = {
            "element_id": element_id,
            "time": float(time_value),
            "damage": float(new_damage),
            "scale": float(scale),
            **metrics,
        }
        diagnostics.append(diag)
        if bool(config.record_history):
            history = state.setdefault("history", [])
            history.append(diag)
        if abs(scale - old_scale) > 1.0e-12:
            changed = True
        deletion_allowed = True
        if bool(config.neighbor_smoothing) and int(state.get("contact_count", 0)) < 2:
            deletion_allowed = False
            diag["neighbor_smoothing_hold"] = True
        if new_damage >= float(config.delete_at) and deletion_allowed:
            state["deleted_time"] = float(time_value)
            new_deleted.append(
                DeletedElementRecord(
                    element_id=element_id,
                    element_type="shell",
                    step_index=int(step_index),
                    load_factor=float(time_value),
                    trigger_name=f"impact_damage:{metrics['governing_component']}",
                    trigger_value=float(new_damage),
                    threshold=float(config.delete_at),
                    location=f"contact:{record.contact_classification}",
                    measure=element_measure(model.mesh, element),
                )
            )
    return tuple(new_deleted), max_utilization, tuple(diagnostics), changed


def _impact_damage_summary(
    model: "FEModel",
    config: Optional[ImpactDamageConfig],
    states: Mapping[int, Mapping[str, Any]],
    deleted_element_ids: Sequence[int],
    *,
    deletion_records: Sequence[DeletedElementRecord] = (),
    warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    if config is None:
        return {"enabled": False, "records": [], "warnings": list(warnings)}
    deleted = {int(element_id) for element_id in deleted_element_ids}
    shell_count = sum(1 for element in model.mesh.elements.values() if element_fracture_category(element) == "shell")
    active_states = {int(element_id): state for element_id, state in states.items()}
    softened = [
        element_id
        for element_id, state in active_states.items()
        if float(state.get("damage", 0.0)) >= float(config.softening_start) and element_id not in deleted
    ]
    records = []
    for element_id, state in sorted(active_states.items()):
        history = state.get("history", []) if bool(config.record_history) else []
        records.append(
            {
                "element_id": int(element_id),
                "damage": float(state.get("damage", 0.0)),
                "scale": float(state.get("scale", _impact_damage_scale(float(state.get("damage", 0.0)), config))),
                "max_utilization": float(state.get("max_utilization", 0.0)),
                "governing_component": str(state.get("governing_component", "")),
                "accumulated_impulse_density": float(state.get("accumulated_impulse_density", 0.0)),
                "max_contact_patch_area": float(state.get("max_contact_patch_area", 0.0)),
                "contact_count": int(state.get("contact_count", 0)),
                "deleted_time": state.get("deleted_time"),
                "history": history,
            }
        )
    return {
        "enabled": True,
        "config": config.to_dict(),
        "deleted_count": len(deleted),
        "softened_count": len(softened),
        "deleted_fraction": float(len(deleted) / max(shell_count, 1)),
        "deleted_element_ids": sorted(deleted),
        "softened_element_ids": sorted(softened),
        "max_damage": max((float(state.get("damage", 0.0)) for state in active_states.values()), default=0.0),
        "max_utilization": max((float(state.get("max_utilization", 0.0)) for state in active_states.values()), default=0.0),
        "records": records,
        "deletion_records": [record.to_dict() for record in deletion_records],
        "warnings": list(warnings),
    }


def _impact_damage_preflight_warnings(
    model: "FEModel",
    config: Optional[ImpactDamageConfig],
    sphere: RigidSphereImpact,
    *,
    fracture_config: Optional[ImpactFractureConfig] = None,
    beam_contact: bool = False,
) -> Tuple[str, ...]:
    if config is None:
        return ()
    warnings: List[str] = []
    if fracture_config is not None:
        warnings.append(
            "IMPACT_DAMAGE010: both ImpactFractureConfig and ImpactDamageConfig are active; "
            "either path may erode shell elements, and summaries report their own trigger ownership."
        )
    if beam_contact and _beam_contact_candidates(model):
        warnings.append(
            "IMPACT_DAMAGE014: beam contact targets are active but capacity-based ImpactDamageConfig "
            "screens shell midsurface contact only; beam erosion needs ImpactFractureConfig "
            "(contact force/penetration) or the nonlinear PlasticImpactDamageConfig path."
        )
    shell_elements = [element for element in model.mesh.elements.values() if element_fracture_category(element) == "shell"]
    if not shell_elements:
        return tuple(warnings)
    fallback_capacity = False
    for element in shell_elements:
        if not isinstance(element, ShellElement):
            continue
        _capacity, source = _impact_material_capacity(model, element, config)
        if source == "elastic_0.2pct_proxy":
            fallback_capacity = True
            break
    if fallback_capacity and config.capacity_basis != "user":
        warnings.append(
            "IMPACT_DAMAGE011: one or more shell materials lack yield stress or RP-C208 hardening curve; "
            "impact damage capacity is using a 0.2% elastic proxy."
        )
    representative_edge = _representative_shell_edge_length(model)
    if representative_edge > 0.0:
        representative_area = representative_edge**2
        if float(config.min_contact_area) < 1.0e-5 * representative_area:
            warnings.append(
                "IMPACT_DAMAGE012: min_contact_area is very small relative to representative shell area; "
                "damage may be mesh sensitive unless calibrated."
            )
    if float(config.contact_area_radius_fraction) * float(sphere.radius) < 0.05 * max(representative_edge, 1.0e-12):
        warnings.append(
            "IMPACT_DAMAGE013: contact patch radius fraction is small relative to mesh edge length; "
            "consider a larger minimum area or finer mesh for production screening."
        )
    return tuple(warnings)


def _require_user_capacity_for_orthotropic_linear_damage(
    model: "FEModel",
    config: Optional[ImpactDamageConfig],
) -> None:
    if config is None or config.capacity_basis == "user":
        return
    affected = []
    for element in _shell_contact_candidates(model):
        material = model.get_material(element.material_name)
        if (
            getattr(element, "shell_section", None) is not None
            or material_symmetry(material) == "orthotropic"
        ):
            affected.append(
                str(
                    getattr(
                        getattr(element, "shell_section", None),
                        "name",
                        "",
                    )
                    or element.material_name
                )
            )
    if affected:
        names = ", ".join(sorted(set(affected)))
        raise ValueError(
            "Orthotropic or generalized-section linear impact pressure damage requires "
            "ImpactDamageConfig(capacity_basis='user', user_capacity=...); "
            f"affected shell material/section name(s): {names}."
        )


def _warnings_with_prefixes(warnings: Sequence[str], prefixes: Sequence[str]) -> Tuple[str, ...]:
    prefix_tuple = tuple(str(prefix) for prefix in prefixes)
    return tuple(str(warning) for warning in warnings if str(warning).startswith(prefix_tuple))


def _impact_fracture_trigger_value(
    record: SphereContactRecord,
    config: ImpactFractureConfig,
    sphere: RigidSphereImpact,
) -> float:
    if config.trigger == "contact_force":
        return float(record.normal_force)
    if config.trigger == "penetration_ratio":
        return float(record.penetration) / max(float(sphere.radius), 1.0e-12)
    contact_radius = max(float(sphere.radius) * float(config.contact_area_radius_fraction), 1.0e-12)
    area = np.pi * contact_radius**2
    return float(record.normal_force) / area


def _detect_impact_fracture_records(
    model: "FEModel",
    records: Sequence[SphereContactRecord],
    config: ImpactFractureConfig,
    sphere: RigidSphereImpact,
    deleted_element_ids: Sequence[int],
    *,
    step_index: int,
    time_value: float,
) -> Tuple[Tuple[DeletedElementRecord, ...], float]:
    if float(time_value) + 1.0e-12 < config.min_time:
        return (), 0.0
    deleted = {int(element_id) for element_id in deleted_element_ids}
    new_records: List[DeletedElementRecord] = []
    max_utilization = 0.0
    for contact_record in records:
        element_id = int(contact_record.element_id)
        if element_id in deleted:
            continue
        element = model.mesh.get_element(element_id)
        if element is None:
            continue
        category = element_fracture_category(element)
        # Contact-observable fracture triggers (force, penetration ratio,
        # sphere-area pressure proxy) are geometry-agnostic, so any erodible
        # contact target -- shell midsurface or beam segment -- can fracture.
        if category is None:
            continue
        trigger_value = _impact_fracture_trigger_value(contact_record, config, sphere)
        utilization = trigger_value / config.threshold
        max_utilization = max(max_utilization, float(utilization))
        if trigger_value >= config.threshold:
            new_records.append(
                DeletedElementRecord(
                    element_id=element_id,
                    element_type=category,
                    step_index=int(step_index),
                    load_factor=float(time_value),
                    trigger_name=str(config.trigger),
                    trigger_value=float(trigger_value),
                    threshold=float(config.threshold),
                    location=f"contact:{contact_record.contact_classification}",
                    measure=element_measure(model.mesh, element),
                )
            )
    return tuple(new_records), max_utilization


def _plastic_damage_scale(utilization: float, config: PlasticImpactDamageConfig) -> float:
    value = max(float(utilization), 0.0)
    if value < float(config.softening_start):
        return 1.0
    if value >= float(config.delete_at):
        return float(config.residual_stiffness_fraction)
    span = max(float(config.delete_at) - float(config.softening_start), 1.0e-12)
    fraction = (value - float(config.softening_start)) / span
    return 1.0 - fraction * (1.0 - float(config.residual_stiffness_fraction))


def _plastic_impact_damage_update(
    model: "FEModel",
    committed_states: Mapping[int, Any],
    config: Optional[PlasticImpactDamageConfig],
    deleted_element_ids: set[int],
    damage_states: Dict[int, Dict[str, Any]],
    *,
    step_index: int,
    time_value: float,
) -> Tuple[List[DeletedElementRecord], bool, float]:
    if config is None:
        return [], False, 0.0
    changed = False
    new_records: List[DeletedElementRecord] = []
    max_utilization = 0.0
    criterion = str(getattr(config, "criterion", "fixed"))
    material_constants: Dict[str, Tuple[float, float, str]] = {}
    for element_id, state in committed_states.items():
        element = model.mesh.elements.get(int(element_id))
        if element is None:
            continue
        category = element_fracture_category(element)
        if category not in set(config.element_scope):
            continue
        trigger_value, location = state_equivalent_plastic_strain(state)

        threshold = float(config.threshold)
        if criterion in {"mesh_scaled_gl", "rtcl", "rtcl_modified"}:
            # GL/RP-C208 mesh scaling of the critical strain: coarse shells
            # cannot resolve necking, so the limit shrinks with element size.
            if category == "shell":
                thickness = float(getattr(element, "thickness", 0.0))
                area = float(element_measure(model.mesh, element))
                l_e = float(np.sqrt(area)) if area > 0.0 else 0.0
                if l_e > 1.0e-6 and thickness > 1.0e-6:
                    threshold = 0.056 + 0.54 * (thickness / l_e)
            elif category == "beam":
                thickness = float(getattr(element, "cross_section", {}).get("web_thickness", 0.0))
                l_e = float(element_measure(model.mesh, element))
                if l_e > 1.0e-6 and thickness > 1.0e-6:
                    threshold = 0.056 + 0.54 * (thickness / l_e)
            if criterion == "rtcl_modified":
                # Calibrate the critical strain so plane-strain tension
                # (eta = 1/sqrt(3), the governing state for plate necking and
                # the state the GL curve was derived from) reaches utilization
                # 1 exactly at the mesh-scaled limit: eps_cr is scaled by the
                # RTCL weight at plane strain, w_ps = exp(sqrt(3)/2 - 1/2).
                weight_ps = float(np.exp(0.5 * np.sqrt(3.0) - 0.5))
                threshold = threshold * weight_ps

        previous = damage_states.get(int(element_id), {})
        previous_damage = float(previous.get("damage", 0.0) or 0.0)
        if criterion in {"rtcl", "rtcl_modified"}:
            # RTCL: accumulate triaxiality-weighted plastic strain per
            # integration point, D = sum(w(eta) d_alpha) / eps_cr.  Damage is
            # zero in compression and reduced in shear, which keeps erosion
            # away from the compressed side of a dent (the classic source of
            # spurious deletions with plain plastic-strain thresholds).
            name = str(element.material_name)
            if name not in material_constants:
                material = model.get_material(name)
                symmetry = material_symmetry(material)
                structural_kind = "shell" if category == "shell" else "beam"
                characteristic_modulus = material_characteristic_modulus(material, structural_kind)
                poisson_ratio = (
                    float(material.poisson_ratio)
                    if symmetry == "isotropic"
                    else float(getattr(material, "poisson_ratio_12", 0.0))
                )
                material_constants[name] = (characteristic_modulus, poisson_ratio, symmetry)
            E, nu, symmetry = material_constants[name]
            rtcl = state_rtcl_increment(
                state,
                previous.get("rtcl_alpha"),
                E,
                nu,
                elastic_symmetry=symmetry,
            )
            accumulated = np.asarray(previous.get("rtcl_damage", ()), dtype=float).reshape(-1)
            if rtcl is None:
                utilization = previous_damage
                rtcl_alpha = previous.get("rtcl_alpha")
            else:
                alpha_now, weighted_delta = rtcl
                if accumulated.size != alpha_now.size:
                    accumulated = np.zeros_like(alpha_now)
                accumulated = accumulated + weighted_delta / max(threshold, 1.0e-30)
                utilization = float(np.max(accumulated)) if accumulated.size else previous_damage
                rtcl_alpha = alpha_now
                location = f"rtcl[{int(np.argmax(accumulated))}]" if accumulated.size else location
        else:
            utilization = float(trigger_value) / max(threshold, 1.0e-30)
        max_utilization = max(max_utilization, utilization)
        damage = max(previous_damage, utilization)
        scale = _plastic_damage_scale(damage, config)
        history = list(previous.get("history", [])) if bool(config.record_history) else []
        if bool(config.record_history):
            history.append(
                {
                    "time": float(time_value),
                    "step_index": int(step_index),
                    "equivalent_plastic_strain": float(trigger_value),
                    "utilization": float(utilization),
                    "damage": float(damage),
                    "scale": float(scale),
                    "location": str(location),
                }
            )
        if damage > previous_damage + 1.0e-15 or abs(scale - float(previous.get("scale", 1.0) or 1.0)) > 1.0e-15:
            changed = True
        entry = {
            "element_id": int(element_id),
            "damage": float(damage),
            "scale": float(scale),
            "max_equivalent_plastic_strain": max(float(previous.get("max_equivalent_plastic_strain", 0.0) or 0.0), float(trigger_value)),
            "max_utilization": max(float(previous.get("max_utilization", 0.0) or 0.0), float(utilization)),
            "location": str(location),
            "history": history,
        }
        if criterion in {"rtcl", "rtcl_modified"}:
            entry["rtcl_damage"] = accumulated
            entry["rtcl_alpha"] = rtcl_alpha
        damage_states[int(element_id)] = entry
        if int(element_id) not in deleted_element_ids and damage >= float(config.delete_at) - 1.0e-15:
            record = DeletedElementRecord(
                element_id=int(element_id),
                element_type=type(element).__name__,
                step_index=int(step_index),
                load_factor=float(time_value),
                trigger_name="rtcl_damage" if criterion in {"rtcl", "rtcl_modified"} else "max_equivalent_plastic_strain",
                trigger_value=float(damage if criterion in {"rtcl", "rtcl_modified"} else trigger_value),
                threshold=float(threshold),
                location=str(location),
                measure=element_measure(model.mesh, element),
            )
            new_records.append(record)
            deleted_element_ids.add(int(element_id))
            changed = True
    return new_records, changed, max_utilization


def _plastic_impact_damage_summary(
    model: "FEModel",
    config: Optional[PlasticImpactDamageConfig],
    damage_states: Mapping[int, Mapping[str, Any]],
    deleted_element_ids: Sequence[int],
    deletion_records: Sequence[DeletedElementRecord],
    *,
    warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    if config is None:
        return {"enabled": False, "records": [], "warnings": list(warnings)}
    scope = set(config.element_scope)
    scoped_count = sum(
        1
        for element in model.mesh.elements.values()
        if element_fracture_category(element) in scope
    )
    deleted = sorted(int(element_id) for element_id in deleted_element_ids)
    records = []
    for element_id, state in sorted((int(k), v) for k, v in damage_states.items()):
        records.append(
            {
                "element_id": int(element_id),
                "damage": float(state.get("damage", 0.0) or 0.0),
                "scale": float(state.get("scale", 1.0) or 1.0),
                "max_equivalent_plastic_strain": float(state.get("max_equivalent_plastic_strain", 0.0) or 0.0),
                "max_utilization": float(state.get("max_utilization", 0.0) or 0.0),
                "location": str(state.get("location", "")),
                "history": list(state.get("history", []) or []),
            }
        )
    return {
        "enabled": True,
        "mode": "material_nonlinear_plastic_strain",
        "config": config.to_dict(),
        "deleted_count": len(deleted),
        "deleted_fraction": float(len(deleted) / max(scoped_count, 1)),
        "deleted_element_ids": deleted,
        "softened_element_ids": sorted(
            int(element_id)
            for element_id, state in damage_states.items()
            if int(element_id) not in set(deleted) and float(state.get("scale", 1.0) or 1.0) < 1.0
        ),
        "max_damage": max((float(state.get("damage", 0.0) or 0.0) for state in damage_states.values()), default=0.0),
        "max_utilization": max((float(state.get("max_utilization", 0.0) or 0.0) for state in damage_states.values()), default=0.0),
        "max_equivalent_plastic_strain": max(
            (float(state.get("max_equivalent_plastic_strain", 0.0) or 0.0) for state in damage_states.values()),
            default=0.0,
        ),
        "records": records,
        "deletion_records": [record.to_dict() for record in deletion_records],
        "warnings": list(warnings),
    }


def _solve_transient_sphere_impact_nonlinear(
    model: "FEModel",
    transient_config: TransientConfig,
    sphere: RigidSphereImpact,
    config: SphereContactConfig,
    validation: ProductionValidationReport,
    *,
    base_load_case: Optional[LoadCase] = None,
    nonlinear_config: Optional[NonlinearTransientConfig] = None,
    plastic_damage_config: Optional[PlasticImpactDamageConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> SphereImpactResult:
    """Implicit Newmark nonlinear impact with material/geometric element response."""
    from .algebraic_dynamics import (
        AlgebraicDynamicsError,
        build_algebraic_static_reduction,
        build_declared_algebraic_basis,
        declared_algebraic_mass_elements,
    )
    from .nonlinear_static import (
        _activate_nonlinear_state_storage,
        _assemble_nonlinear_system,
        _nonlinear_state_summary,
        states_von_mises_map,
    )
    from .nonlinear_state import (
        begin_state_evaluation,
        commit_state_candidate,
        discard_active_state_candidate,
        materialize_state_mapping,
    )

    full_coordinate_assembly_count = 0

    def assemble_nonlinear(
        displacements: np.ndarray,
        states: Mapping[int, Any],
        *,
        tangent: bool,
        scales: Optional[Mapping[int, float]] = None,
    ) -> Tuple[np.ndarray, Any, Mapping[int, Any]]:
        nonlocal full_coordinate_assembly_count
        kwargs = {
            "deleted_element_ids": tuple(deleted_element_ids),
            "residual_stiffness_fraction": float(
                plastic_damage_config.residual_stiffness_fraction if plastic_damage_config is not None else 1.0
            ),
            "kinematics": str(nl.kinematics),
        }
        if scales:
            kwargs["element_stiffness_scales"] = scales
        try:
            result = _assemble_nonlinear_system(
                model,
                displacements,
                states,
                int(nl.num_layers),
                tangent=tangent,
                **kwargs,
            )
            full_coordinate_assembly_count += 1
            return result
        except TypeError as exc:
            if "element_stiffness_scales" not in str(exc):
                raise
            kwargs.pop("element_stiffness_scales", None)
            result = _assemble_nonlinear_system(
                model,
                displacements,
                states,
                int(nl.num_layers),
                tangent=tangent,
                **kwargs,
            )
            full_coordinate_assembly_count += 1
            return result

    cancellation_safe_point(cancellation_token, "sphere_impact.nonlinear.start")
    nl = nonlinear_config or NonlinearTransientConfig(enabled=True)
    model.apply_boundary_conditions()
    total_dofs = model.mesh.dof_manager.total_dofs
    needs_linear_stiffness = abs(float(transient_config.rayleigh_beta)) > 0.0
    if needs_linear_stiffness:
        K0, stiffness_info = assemble_stiffness_matrix(model)
    else:
        K0 = sparse.eye(total_dofs, format="csr", dtype=float)
        stiffness_info = {
            "assembly_skipped": True,
            "reason": "nonlinear_impact_no_stiffness_proportional_damping",
        }
    M, mass_info = assemble_mass_matrix(model)
    base_load, base_load_info = assemble_load_vector(model, base_load_case)
    zero_load = np.zeros(total_dofs, dtype=float)
    _K_red0, _zero_red, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K0, zero_load, model)
    K_red0 = (T.T @ K0 @ T).tocsr()
    K_energy_red = K_red0 if needs_linear_stiffness else None
    M_red = (T.T @ M @ T).tocsr()
    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red0).tocsr()
    descriptor_elements = declared_algebraic_mass_elements(model)
    descriptor_basis = None
    descriptor_reduction = None
    descriptor_partition_operator = None
    descriptor_certificate: Dict[str, Any] = {
        "enabled": bool(descriptor_elements),
        "element_ids": [int(value) for value in descriptor_elements],
    }
    if descriptor_elements:
        descriptor_basis = build_declared_algebraic_basis(
            model,
            M,
            M_red,
            T,
            independent_dofs,
            dense_size_limit=512,
        )
        descriptor_certificate["mass_basis"] = descriptor_basis.diagnostics
    if float(np.linalg.norm(M_red.diagonal())) <= 0.0 and M_red.nnz == 0:
        raise ValueError("Nonlinear sphere impact requires a non-zero structural mass matrix; set material density values.")

    times = _time_grid(transient_config)
    recovery = transient_config.recovery
    output_node_ids = tuple(int(node_id) for node_id in (transient_config.output_nodes or ()))
    if recovery is not None and not output_node_ids and recovery.node_ids is not None:
        output_node_ids = tuple(int(node_id) for node_id in recovery.node_ids)
    history_storage_mode = "full" if recovery is None else str(recovery.history_mode)
    if recovery is not None and history_storage_mode == "full" and not recovery.store_full_histories:
        history_storage_mode = "selected"
    if history_storage_mode == "selected" and not output_node_ids and (recovery is None or recovery.include_displacements):
        output_node_ids = tuple(int(node_id) for node_id in model.mesh.nodes)
    history_dof_indices = _node_dof_indices(model, output_node_ids) if history_storage_mode == "selected" else None
    estimated_saved_steps = _saved_step_count(times, transient_config.save_every)
    preflight_memory = estimate_model_memory(
        model,
        transient_saved_steps=estimated_saved_steps,
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(preflight_memory, transient_config.resource_config, context="solve_transient_sphere_impact.nonlinear")
    damage_matrix_gate = DamageMatrixPlanGate(
        total_opportunities=max(int(len(times) - 1), 0),
        preflight_memory_bytes=int(preflight_memory.total_bytes_estimate),
        configured_memory_limit_bytes=(
            None
            if transient_config.resource_config is None
            or transient_config.resource_config.memory_limit_bytes is None
            else int(transient_config.resource_config.memory_limit_bytes)
        ),
    )

    q = _full_initial_vector(transient_config.initial_displacement, total_dofs)
    v_full = _full_initial_vector(transient_config.initial_velocity, total_dofs)
    q_red = np.asarray((q - u0)[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    v_red = np.asarray(v_full[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    impact_energy_reference = max(0.5 * float(sphere.mass) * float(sphere.speed) ** 2, 1.0)
    base_load_equilibrated = False
    committed_states: Mapping[int, Any] = {}
    state_storage_info: Dict[str, Any] = {}
    committed_states = _activate_nonlinear_state_storage(
        model,
        committed_states,
        int(nl.num_layers),
        state_storage_info,
        kinematics=str(nl.kinematics),
        committed_displacements=reconstruct_full_solution(T, q_red, u0),
    )
    deleted_element_ids: set[int] = set()
    damage_deleted_element_ids: set[int] = set()
    plastic_damage_records: List[DeletedElementRecord] = []
    plastic_damage_states: Dict[int, Dict[str, Any]] = {}
    damage_scale_by_element: Dict[int, float] = {}
    linear_matrix_terms: Optional[Tuple[int, Tuple[LinearElementMatrixTerms, ...]]] = None
    damage_matrix_plan: Optional[DamageMatrixPlan] = None
    if (
        bool(nl.equilibrate_base_load)
        and transient_config.initial_displacement is None
        and float(np.linalg.norm(base_load)) > 0.0
    ):
        # Start from the NONLINEAR static equilibrium of the base load
        # instead of applying it as a step at t = 0: a stepped preload rings
        # the structure at its axial period through the whole impact,
        # distorting stresses and degrading Newton convergence.  A short
        # Newton solve with residual backtracking finds the equilibrium; if
        # it cannot converge (e.g. the preload alone is beyond a limit
        # point), the solver falls back to the stepped start.
        base_reference = max(float(np.linalg.norm(np.asarray(T.T @ base_load, dtype=float))), 1.0e-30)
        q_static = np.zeros_like(q_red)
        try:
            for static_iteration in range(25):
                cancellation_safe_point(
                    cancellation_token,
                    f"sphere_impact.nonlinear.preload:{static_iteration + 1}",
                )
                F_int_static, K_static, _trial_unused = assemble_nonlinear(
                    reconstruct_full_solution(T, q_static, u0),
                    committed_states,
                    tangent=True,
                )
                residual_static = np.asarray(T.T @ (base_load - F_int_static), dtype=float).reshape(-1)
                residual_static_norm = float(np.linalg.norm(residual_static))
                if residual_static_norm <= 1.0e-3 * base_reference:
                    base_load_equilibrated = True
                    break
                K_static_red = (T.T @ K_static @ T).tocsr()
                static_handle = factorize(
                    K_static_red,
                    MatrixClass.SYMMETRIC_INDEFINITE,
                    signature=f"sphere_impact.nl.base_equilibrium:{static_iteration}",
                )
                dq_static = np.asarray(static_handle.solve(residual_static), dtype=float).reshape(-1)
                if not np.all(np.isfinite(dq_static)):
                    break
                accepted_q = None
                step_factor = 1.0
                for _backtrack in range(8):
                    q_candidate = q_static + step_factor * dq_static
                    F_candidate, _K_unused, _states_unused = assemble_nonlinear(
                        reconstruct_full_solution(T, q_candidate, u0),
                        committed_states,
                        tangent=False,
                    )
                    candidate_norm = float(
                        np.linalg.norm(np.asarray(T.T @ (base_load - F_candidate), dtype=float))
                    )
                    if candidate_norm < residual_static_norm:
                        accepted_q = q_candidate
                        break
                    step_factor *= 0.5
                if accepted_q is None:
                    break
                q_static = accepted_q
        except Exception:
            base_load_equilibrated = False
        if base_load_equilibrated:
            q_red = q_static
    descriptor_active = bool(
        descriptor_basis is not None
        and descriptor_basis.reduced_basis.shape[1] > 0
    )
    initial_algebraic_iterations = 0
    initial_algebraic_relative_residual = 0.0
    F_int0 = np.zeros(total_dofs, dtype=float)
    K_initial = None
    trial0: Mapping[int, Any] = committed_states
    try:
        for algebraic_iteration in range(1, 6 if descriptor_active else 2):
            initial_algebraic_iterations = algebraic_iteration
            initial_full_u = reconstruct_full_solution(T, q_red, u0)
            F_int0, K_initial, trial0 = assemble_nonlinear(
                initial_full_u,
                committed_states,
                tangent=descriptor_active,
            )
            if not descriptor_active:
                break
            assert descriptor_basis is not None and K_initial is not None
            K_initial_red = (T.T @ K_initial @ T).tocsr()
            descriptor_reduction = build_algebraic_static_reduction(
                K_initial_red,
                M_red,
                descriptor_basis.reduced_basis,
                dense_size_limit=512,
            )
            physical_velocity, _supplied_algebraic_velocity = (
                descriptor_reduction.split_k_orthogonal_state(v_red)
            )
            v_red = descriptor_reduction.reconstruct_k_orthogonal_state(
                physical_velocity,
                np.zeros(descriptor_reduction.algebraic_dimension, dtype=float),
            )
            initial_residual_red = np.asarray(
                T.T @ (base_load - F_int0), dtype=float
            ).reshape(-1) - np.asarray(C_red @ v_red, dtype=float).reshape(-1)
            algebraic_residual = np.asarray(
                descriptor_reduction.basis.T @ initial_residual_red,
                dtype=float,
            ).reshape(-1)
            algebraic_scale = np.asarray(
                abs(descriptor_reduction.basis.T)
                @ (
                    np.abs(np.asarray(T.T @ base_load, dtype=float).reshape(-1))
                    + np.abs(np.asarray(T.T @ F_int0, dtype=float).reshape(-1))
                    + np.abs(np.asarray(C_red @ v_red, dtype=float).reshape(-1))
                ),
                dtype=float,
            ).reshape(-1)
            initial_algebraic_relative_residual = float(
                np.linalg.norm(algebraic_residual)
                / max(np.linalg.norm(algebraic_scale), 1.0)
            )
            if initial_algebraic_relative_residual <= 2.0e-10:
                break
            algebraic_correction = descriptor_reduction.solve_algebraic(
                algebraic_residual
            )
            q_red = q_red + np.asarray(
                descriptor_reduction.basis @ algebraic_correction,
                dtype=float,
            ).reshape(-1)
        else:
            raise AlgebraicDynamicsError(
                "nonlinear impact initial algebraic equilibrium did not converge"
            )
        if descriptor_active and initial_algebraic_relative_residual > 2.0e-10:
            raise AlgebraicDynamicsError(
                "nonlinear impact initial algebraic equilibrium is unresolved"
            )
        accepted_initial_full_u = reconstruct_full_solution(T, q_red, u0)
        committed_states = commit_state_candidate(
            committed_states,
            trial0,
            accepted_full_displacement=accepted_initial_full_u,
            model=model,
        )
    except BaseException:
        discard_active_state_candidate(committed_states)
        raise
    if descriptor_reduction is not None:
        physical_rows = np.asarray(
            descriptor_reduction.physical_rows, dtype=np.intp
        )
        physical_selector = sparse.csc_matrix(
            (
                np.ones(physical_rows.size, dtype=float),
                (physical_rows, np.arange(physical_rows.size, dtype=np.intp)),
            ),
            shape=(M_red.shape[0], physical_rows.size),
        )
        descriptor_partition_operator = sparse.hstack(
            (physical_selector, descriptor_reduction.basis),
            format="csc",
        )
        if descriptor_partition_operator.shape != M_red.shape:
            raise AlgebraicDynamicsError(
                "nonlinear impact algebraic partition is not square"
            )
        descriptor_certificate["static_reduction"] = (
            descriptor_reduction.diagnostics
        )
        descriptor_certificate["initial_algebraic_iterations"] = int(
            initial_algebraic_iterations
        )
        descriptor_certificate["initial_algebraic_relative_residual"] = float(
            initial_algebraic_relative_residual
        )
        descriptor_certificate["partition_dimension"] = int(
            descriptor_partition_operator.shape[0]
        )
    elif descriptor_elements:
        descriptor_certificate["constrained_algebraic_dimension"] = 0
    F_int_prev = np.asarray(F_int0, dtype=float).copy()
    F0_red = np.asarray(T.T @ (base_load - F_int0), dtype=float).reshape(-1)
    from .impact_reduced_assembly import prepare_impact_reduced_assembly

    impact_reduced_assembly = prepare_impact_reduced_assembly(
        model,
        T,
        u0,
        num_layers=int(nl.num_layers),
        kinematics=str(nl.kinematics),
        plastic_damage_enabled=plastic_damage_config is not None,
        num_steps=max(int(len(times) - 1), 0),
        max_iterations=int(nl.max_iterations),
    )
    base_load_red = np.asarray(T.T @ base_load, dtype=float).reshape(-1)
    F_int_prev_red = np.asarray(T.T @ F_int_prev, dtype=float).reshape(-1)
    initial_inertial_rhs = np.asarray(
        F0_red - C_red @ v_red, dtype=float
    ).reshape(-1)
    if descriptor_reduction is None:
        mass_handle = factorize(
            M_red,
            MatrixClass.SYMMETRIC_SEMIDEFINITE,
            signature="sphere_impact.nl.initial_mass",
        )
        a_red = np.asarray(
            mass_handle.solve(initial_inertial_rhs), dtype=float
        ).reshape(-1)
    else:
        mass_handle = factorize(
            descriptor_reduction.mass_physical,
            MatrixClass.SPD,
            signature="sphere_impact.nl.initial_descriptor_physical_mass",
        )
        if mass_handle.status != "ok":
            raise AlgebraicDynamicsError(
                "nonlinear impact descriptor physical mass factorization failed"
            )
        physical_acceleration = np.asarray(
            mass_handle.solve(
                initial_inertial_rhs[descriptor_reduction.physical_rows]
            ),
            dtype=float,
        ).reshape(-1)
        a_red = descriptor_reduction.reconstruct_k_orthogonal_state(
            physical_acceleration,
            np.zeros(descriptor_reduction.algebraic_dimension, dtype=float),
        )

    sphere_position = sphere.initial_position
    sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    sphere_acceleration = np.zeros(3, dtype=float)

    saved_times: List[float] = []
    saved_u: List[np.ndarray] = []
    saved_v: List[np.ndarray] = []
    saved_a: List[np.ndarray] = []
    saved_sphere_x: List[np.ndarray] = []
    saved_sphere_v: List[np.ndarray] = []
    saved_sphere_a: List[np.ndarray] = []
    saved_contact_force: List[np.ndarray] = []
    saved_contacts: List[Tuple[Dict[str, Any], ...]] = []
    node_history_values: Dict[int, list[np.ndarray]] = {int(node_id): [] for node_id in output_node_ids}
    element_state_history: List[Dict[str, Any]] = []
    state_von_mises_history: List[Dict[int, float]] = []
    peak_displacement = 0.0
    peak_displacement_node = None
    max_penetration = 0.0
    peak_contact_force = 0.0
    contact_step_count = 0
    active_contact_duration = 0.0
    iteration_counts: List[int] = []
    energy_kinetic: List[float] = []
    energy_strain: List[float] = []
    sphere_kinetic: List[float] = []
    plastic_work_history: List[float] = []

    def save_state(
        time: float,
        q_state: np.ndarray,
        v_state: np.ndarray,
        a_state: np.ndarray,
        sphere_x: np.ndarray,
        sphere_v: np.ndarray,
        sphere_a: np.ndarray,
        sphere_force: np.ndarray,
        records: Tuple[SphereContactRecord, ...],
        state_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal peak_displacement, peak_displacement_node, max_penetration, peak_contact_force
        full_u = reconstruct_full_solution(T, q_state, u0)
        full_v = np.asarray(T @ v_state, dtype=float).reshape(-1)
        full_a = np.asarray(T @ a_state, dtype=float).reshape(-1)
        saved_times.append(float(time))
        if history_storage_mode == "full":
            saved_u.append(full_u)
            saved_v.append(full_v)
            saved_a.append(full_a)
        elif history_storage_mode == "selected":
            indices = np.asarray(history_dof_indices if history_dof_indices is not None else (), dtype=np.intp)
            saved_u.append(full_u[indices])
            saved_v.append(full_v[indices])
            saved_a.append(full_a[indices])
        for node_id in output_node_ids:
            node = model.mesh.get_node(int(node_id))
            if node is not None:
                node_history_values[int(node_id)].append(full_u[np.asarray(node.dofs, dtype=np.intp)])
        current_peak, current_node = _translation_peak(model, full_u)
        if current_peak > peak_displacement:
            peak_displacement = current_peak
            peak_displacement_node = current_node
        if records:
            max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
            peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
        saved_sphere_x.append(np.asarray(sphere_x, dtype=float).copy())
        saved_sphere_v.append(np.asarray(sphere_v, dtype=float).copy())
        saved_sphere_a.append(np.asarray(sphere_a, dtype=float).copy())
        saved_contact_force.append(np.asarray(sphere_force, dtype=float).copy())
        saved_contacts.append(tuple(record.to_dict() for record in records) if config.save_contact_history else tuple())
        energy_kinetic.append(0.5 * float(v_state @ (M_red @ v_state)))
        sphere_kinetic.append(0.5 * float(sphere.mass) * float(sphere_v @ sphere_v))
        if K_energy_red is not None:
            energy_strain.append(0.5 * float(q_state @ (K_energy_red @ q_state)))
        else:
            # internal work measure from the committed nonlinear internal
            # force: exact strain energy for the elastic response, and an
            # internal-work proxy (elastic + dissipated) once plasticity is
            # active.
            if impact_reduced_assembly.active:
                # Direct assembly is eligible only for u0 == 0, hence
                # full_u @ F_int == q @ (T.T @ F_int) exactly.
                energy_strain.append(0.5 * float(q_state @ F_int_prev_red))
            else:
                energy_strain.append(0.5 * float(full_u @ F_int_prev))
        # One summary pass per saved step: it walks every element state, so
        # repeating it for the plastic-work history, the state history and
        # the progress payload triples a cost that rivals the Newton solves.
        # The stepping loop passes its own summary in to avoid recomputing.
        summary = state_summary if state_summary is not None else _nonlinear_state_summary(committed_states)
        max_alpha = float(summary.get("max_equivalent_plastic_strain", 0.0) or 0.0)
        plastic_work_history.append(max_alpha)
        if bool(nl.record_element_state_history):
            element_state_history.append({"time": float(time), **summary})
            # True (return-mapped) stress envelope per element at this saved
            # step: an elastic recovery from the saved displacements ignores
            # the plastic strain and overshoots the material curve.
            state_von_mises_history.append(states_von_mises_map(model, committed_states))
        if progress_callback is not None and history_storage_mode == "full":
            emit_progress(
                progress_callback,
                "sphere_impact_live_step",
                "sphere_impact.nonlinear.integration",
                completed=float(time),
                total=float(transient_config.t_end),
                time_s=float(time),
                step_index=int(len(saved_times) - 1),
                displacement=full_u.copy(),
                sphere_position=np.asarray(sphere_x, dtype=float).copy(),
                sphere_radius=float(sphere.radius),
                contact_force=np.asarray(sphere_force, dtype=float).copy(),
                active_contacts=tuple(record.to_dict() for record in records) if config.save_contact_history else tuple(),
                nonlinear=True,
                max_equivalent_plastic_strain=max_alpha,
            )

    contact_work_counters = ContactWorkCounters()
    contact_work_buffer = ContactWorkBuffer(total_dofs, counters=contact_work_counters)
    contact_geometry = _contact_geometry(model)
    native_contact_state = bool(
        getattr(committed_states, "has_native_rotations", False)
    )
    initial_full_u = reconstruct_full_solution(T, q_red, u0)
    try:
        if native_contact_state:
            # Contact consumes the same generation-checked native view as the
            # element response.  Open an exact candidate for the already
            # committed initial configuration, then reject it after contact has
            # copied the required kinematics; no material or rotation history is
            # advanced by a contact search.
            begin_state_evaluation(
                committed_states,
                model=model,
                displacements=initial_full_u,
            )
        initial_contact_work = _assemble_sphere_contact_work_buffer(
            model,
            sphere,
            config,
            sphere_position,
            sphere_velocity,
            structural_displacement=initial_full_u,
            structural_velocity=np.asarray(T @ v_red, dtype=float).reshape(-1),
            deleted_element_ids=tuple(deleted_element_ids),
            contact_scale_by_element=damage_scale_by_element,
            work_buffer=contact_work_buffer,
            native_rotation_context=(
                committed_states if native_contact_state else None
            ),
        )
    finally:
        if native_contact_state:
            discard_active_state_candidate(committed_states)
    initial_load = initial_contact_work.load.copy()
    initial_sphere_force = initial_contact_work.sphere_force.copy()
    initial_records = initial_contact_work.materialize_records(
        SphereContactRecord,
        contact_geometry.node_ids,
    )
    save_state(float(times[0]), q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, initial_sphere_force, initial_records)
    sticky_contact_ids: Tuple[int, ...] = tuple(int(record.element_id) for record in initial_records)
    tangent_reuse = ImpactTangentReuseController(int(nl.tangent_reuse_iterations))
    tangent_reuse.set_initial_contact(initial_records)

    load_prev = base_load + initial_load
    load_prev_red = np.asarray(T.T @ load_prev, dtype=float).reshape(-1)
    sphere_force_prev = initial_sphere_force.copy()
    impulse = np.zeros(total_dofs, dtype=float)
    sphere_impulse = np.zeros(3, dtype=float)
    status = "completed"
    stop_reason = "completed"
    warnings: List[str] = [f"{issue.code}: {issue.message}" for issue in validation.warnings]
    warnings.append(
        "NONLINEAR_IMPACT001: material-nonlinear impact uses implicit Newmark with structural nonlinear tangent "
        "and iterative penalty contact; exact contact tangent, friction, spin, tearing paths and FSI are unsupported."
    )
    factorization_count = 0
    solve_count = 0
    total_substep_count = 0
    event_substep_count = 0
    cutback_count = 0
    step_diagnostics: List[Dict[str, Any]] = []
    nonlinear_failure_summary: Dict[str, Any] = {}
    alpha_h, beta, gamma = transient_config.integration_parameters()
    one_plus_alpha = 1.0 + alpha_h
    coupling_fallback = None
    if nl.coupled_contact_acceleration == "none":
        coupling_fallback = "not_requested"
    elif config.contact_relaxation == "none":
        coupling_fallback = "contact_relaxation_disabled"
    elif native_contact_state:
        coupling_fallback = "native_rotations"
    elif descriptor_reduction is not None:
        coupling_fallback = "descriptor_coordinates"
    elif T[np.asarray([dof for node_id in model.mesh.nodes
                        for dof in model.mesh.get_node(node_id).dofs[3:6]], dtype=np.intp), :].nnz:
        coupling_fallback = "rotational_coordinates"
    coupled_active = coupling_fallback is None
    coupled_safeguard_count = 0
    coupling_method = nl.coupled_contact_acceleration if coupled_active else "none"
    if coupling_method == "contact_newton" and q_red.size + 3 > 64:
        coupling_method = "anderson"
        coupling_fallback = "coupled_dimension_exceeds_64"
    contact_newton_active = coupling_method == "contact_newton"
    contact_jacobian_evaluation_count = 0
    separation_elapsed = 0.0
    separation_stop_time = float(config.post_separation_time)

    segments: List[Tuple[int, float, float, bool]] = [
        (int(index), float(times[index - 1]), float(times[index]), True)
        for index in range(1, len(times))
    ]
    travel_scale = max(float(sphere.radius) * float(config.max_sphere_travel_fraction), 1.0e-12)
    pending_save: Optional[Tuple[float, np.ndarray, Tuple[SphereContactRecord, ...]]] = None
    # Convergence-distress carryover: after a cutback, upcoming base steps start
    # pre-halved (up to quartered) instead of retrying the full dt, then the
    # subdivision decays as Newton recovers.  This avoids paying a full failed
    # Newton budget on every step of a difficult (yielding/limit-point) phase.
    distress_halvings = 0
    preemptive_substep_count = 0
    while segments:
        cancellation_safe_point(
            cancellation_token,
            "sphere_impact.nonlinear.segment",
        )
        step_index, segment_start, sub_time, needs_travel_check = segments.pop(0)
        dt = float(sub_time - segment_start)
        if dt <= 0.0:
            continue
        if needs_travel_check:
            predicted_travel = max(float(np.linalg.norm(sphere_velocity)) * dt, float(sphere.speed) * dt)
            n_travel = int(np.ceil(predicted_travel / travel_scale))
            if distress_halvings > 0:
                n_travel = max(n_travel, 2 ** distress_halvings)
            n_travel = min(max(n_travel, 1), int(config.max_event_substeps))
            if n_travel > 1:
                pieces: List[Tuple[int, float, float, bool]] = []
                piece_start = float(segment_start)
                for piece in range(1, n_travel + 1):
                    piece_end = float(sub_time) if piece == n_travel else float(segment_start) + piece * dt / n_travel
                    pieces.append((step_index, piece_start, piece_end, False))
                    piece_start = piece_end
                segments[:0] = pieces
                event_substep_count += n_travel - 1
                if distress_halvings > 0:
                    preemptive_substep_count += n_travel - 1
                continue
        total_substep_count += 1
        pre_state = (
            q_red.copy(),
            v_red.copy(),
            a_red.copy(),
            sphere_position.copy(),
            sphere_velocity.copy(),
            sphere_acceleration.copy(),
            committed_states,
            set(deleted_element_ids),
            dict(plastic_damage_states),
            dict(damage_scale_by_element),
            load_prev.copy(),
            load_prev_red.copy(),
            sphere_force_prev.copy(),
            F_int_prev.copy(),
            F_int_prev_red.copy(),
        )
        if sub_time - dt < sphere.t_start <= sub_time and np.linalg.norm(sphere_velocity) == 0.0:
            sphere_velocity = sphere.travel_velocity.copy()
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        tangent_reuse.begin_substep(dt, damage_scale_by_element, deleted_element_ids)
        q_trial = q_red.copy()
        contact_load = np.zeros(total_dofs, dtype=float)
        sphere_force = np.zeros(3, dtype=float)
        records: Tuple[SphereContactRecord, ...] = tuple()
        last_contact_work: Optional[ContactWorkBuffer] = None
        converged = False
        residual_norm = float("inf")
        displacement_increment = float("inf")
        residual_energy_increment = float("inf")
        contact_change = float("inf")
        reference = 1.0
        contact_scale = 1.0
        trial_states: Mapping[int, Any] = committed_states
        q_next = q_red.copy()
        v_next = v_red.copy()
        a_next = a_red.copy()
        sphere_x_next = sphere_position.copy()
        sphere_v_next = sphere_velocity.copy()
        sphere_a_next = sphere_acceleration.copy()
        factor_diagnostics: Dict[str, Any] = {}
        convergence_reason = "standard"
        max_newton_corrections = int(nl.max_iterations)
        from ._impact_coupling import ImpactCoupledIteration, numerical_contact_jacobian
        coupled_iteration = ImpactCoupledIteration() if coupling_method == "anderson" else None
        coupled_signature = None
        sphere_x_pred = sphere_position + dt * sphere_velocity + dt**2 * (0.5 - beta) * sphere_acceleration
        sphere_v_pred = sphere_velocity + dt * (1.0 - gamma) * sphere_acceleration
        sphere_x_iterate = sphere_x_pred.copy()
        for iteration in range(1, max_newton_corrections + 2):
            cancellation_safe_point(
                cancellation_token,
                f"sphere_impact.nonlinear.time:{sub_time:.16g}.iteration:{iteration}",
            )
            if status_callback:
                status_callback(f"\r  Time {sub_time:.4f} s, Iteration {iteration}: Res {residual_norm:.2e}")
            full_u = reconstruct_full_solution(T, q_trial, u0)
            refresh_tangent, refresh_reasons = tangent_reuse.refresh_decision()
            if contact_newton_active:
                refresh_tangent = True
                refresh_reasons = tuple(refresh_reasons) + ("coupled_contact_jacobian",)
            if impact_reduced_assembly.active:
                F_int_red, K_T_red, trial_states = impact_reduced_assembly.assemble(
                    full_u,
                    committed_states,
                    tangent=refresh_tangent,
                )
                F_int = None
                K_T = None
            else:
                F_int, K_T, trial_states = assemble_nonlinear(
                    full_u,
                    committed_states,
                    tangent=refresh_tangent,
                    scales=damage_scale_by_element,
                )
                F_int_red = None
                K_T_red = None
            a_trial = a0 * (q_trial - q_red) - a2 * v_red - a3 * a_red
            v_trial = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_trial)
            if impact_reduced_assembly.active:
                contact_load_red = np.asarray(T.T @ contact_load, dtype=float).reshape(-1)
                external_load_red = base_load_red + contact_load_red
                if alpha_h != 0.0:
                    weighted_force_red = (
                        one_plus_alpha * (external_load_red - F_int_red)
                        - alpha_h * (load_prev_red - F_int_prev_red)
                    )
                    residual = (
                        weighted_force_red
                        - one_plus_alpha * (C_red @ v_trial)
                        + alpha_h * (C_red @ v_red)
                        - M_red @ a_trial
                    )
                else:
                    residual = (
                        external_load_red
                        - F_int_red
                        - C_red @ v_trial
                        - M_red @ a_trial
                    )
                reference = max(
                    float(np.linalg.norm(external_load_red)),
                    float(np.linalg.norm(F_int_red)),
                    1.0,
                )
            else:
                if alpha_h != 0.0:
                    weighted_force = one_plus_alpha * (base_load + contact_load - F_int) - alpha_h * (load_prev - F_int_prev)
                    residual = (
                        np.asarray(T.T @ weighted_force, dtype=float).reshape(-1)
                        - one_plus_alpha * (C_red @ v_trial)
                        + alpha_h * (C_red @ v_red)
                        - M_red @ a_trial
                    )
                else:
                    residual = np.asarray(T.T @ (base_load + contact_load - F_int), dtype=float).reshape(-1) - C_red @ v_trial - M_red @ a_trial
                reference = max(float(np.linalg.norm(np.asarray(T.T @ (base_load + contact_load), dtype=float))), float(np.linalg.norm(np.asarray(T.T @ F_int, dtype=float))), 1.0)
            residual_norm = float(np.linalg.norm(residual))
            trial_refresh_reasons = tangent_reuse.assess_trial_state(residual_norm, trial_states)
            if not refresh_tangent and trial_refresh_reasons:
                # A force-only trial exposed a stall or a material-state
                # transition.  Reassemble the tangent at the same iterate so
                # the invalidation takes effect before solving the correction.
                if impact_reduced_assembly.active:
                    F_int_red, K_T_red, trial_states = impact_reduced_assembly.assemble(
                        full_u,
                        committed_states,
                        tangent=True,
                    )
                else:
                    F_int, K_T, trial_states = assemble_nonlinear(
                        full_u,
                        committed_states,
                        tangent=True,
                        scales=damage_scale_by_element,
                    )
                refresh_tangent = True
                refresh_reasons = tuple(sorted(set(refresh_reasons).union(trial_refresh_reasons)))
            elif refresh_tangent and trial_refresh_reasons:
                refresh_reasons = tuple(sorted(set(refresh_reasons).union(trial_refresh_reasons)))

            # Convergence belongs to the configuration that was just
            # assembled.  The previous implementation tested the residual
            # at ``q_trial`` only after advancing to ``q_next``, then stored
            # the old trial material state while accepting the new
            # displacement.  Apart from corrupting path-dependent history,
            # that is incompatible with solver-owned multiplicative nodal
            # rotations, whose commit is bound to the exact accepted global
            # vector.  Use the previous correction/contact changes as the
            # fixed-point measures and accept only this evaluated iterate.
            displacement_limit = float(nl.displacement_tolerance) * max(
                float(np.linalg.norm(q_trial)), 1.0
            )
            residual_limit = float(nl.residual_tolerance) * reference
            force_limit = max(
                float(nl.contact_force_tolerance) * contact_scale,
                float(config.force_tolerance),
            )
            completed_corrections = iteration - 1
            standard_converged = (
                completed_corrections > 0
                and residual_norm <= residual_limit
                and displacement_increment <= displacement_limit
                and contact_change <= force_limit
            )
            contact_stall_converged = (
                completed_corrections >= max(3, int(nl.max_iterations) // 2)
                and residual_norm <= 1.0e-2 * reference
                and displacement_increment
                <= max(10.0 * displacement_limit, 1.0e-6)
                and contact_change
                <= max(
                    5.0e-3 * contact_scale,
                    10.0 * float(config.force_tolerance),
                )
            )
            stagnation_converged = (
                float(nl.stagnation_energy_tolerance) > 0.0
                and completed_corrections >= 5
                and displacement_increment <= displacement_limit
                and contact_change <= force_limit
                and residual_energy_increment
                <= float(nl.stagnation_energy_tolerance)
                * impact_energy_reference
            )
            if standard_converged or contact_stall_converged or stagnation_converged:
                converged = True
                q_next = q_trial.copy()
                v_next = np.asarray(v_trial, dtype=float).copy()
                a_next = np.asarray(a_trial, dtype=float).copy()
                # The accepted acceleration is conjugate to the contact
                # force evaluated at the accepted sphere configuration.
                if not coupled_active:
                    sphere_a_next = np.asarray(
                        sphere_force / float(sphere.mass), dtype=float
                    ).copy()
                if standard_converged:
                    convergence_reason = "standard"
                elif contact_stall_converged:
                    convergence_reason = "contact_stall"
                else:
                    convergence_reason = "stagnation_energy"
                iteration_counts.append(completed_corrections)
                if completed_corrections <= max(4, int(nl.max_iterations) // 3):
                    distress_halvings = max(distress_halvings - 1, 0)
                elif completed_corrections >= int(nl.max_iterations) - 2:
                    distress_halvings = min(distress_halvings + 1, 2)
                break
            if completed_corrections >= max_newton_corrections:
                # The final response evaluation is a confirmation only; the
                # configured limit counts Newton corrections, not a hidden
                # unevaluated candidate beyond that limit.
                break
            use_contact_newton = contact_newton_active and iteration > 1
            sphere_delta = None
            try:
                if refresh_tangent:
                    if impact_reduced_assembly.active:
                        if descriptor_reduction is not None:
                            raise AlgebraicDynamicsError(
                                "nonlinear impact direct reduction cannot bypass "
                                "declared algebraic coordinates"
                            )
                        K_eff = (one_plus_alpha * K_T_red + a0 * M_red + one_plus_alpha * a1 * C_red).tocsr()
                    else:
                        K_tangent_red = (T.T @ K_T @ T).tocsr()
                        K_eff = (
                            one_plus_alpha * K_tangent_red
                            + a0 * M_red
                            + one_plus_alpha * a1 * C_red
                        ).tocsr()
                        if descriptor_reduction is not None:
                            assert descriptor_basis is not None
                            current_descriptor_reduction = (
                                build_algebraic_static_reduction(
                                    K_tangent_red,
                                    M_red,
                                    descriptor_basis.reduced_basis,
                                    dense_size_limit=512,
                                )
                            )
                            if not np.array_equal(
                                current_descriptor_reduction.physical_rows,
                                descriptor_reduction.physical_rows,
                            ):
                                raise AlgebraicDynamicsError(
                                    "nonlinear impact algebraic partition changed "
                                    "during Newton iteration"
                                )
                            assert descriptor_partition_operator is not None
                            K_eff = (
                                descriptor_partition_operator.T
                                @ K_eff
                                @ descriptor_partition_operator
                            ).tocsr()
                    if use_contact_newton:
                        def evaluate_contact_coordinates(coordinates):
                            nonlocal contact_jacobian_evaluation_count
                            q_probe, x_probe = coordinates[:-3], coordinates[-3:]
                            a_probe = a0 * (q_probe - q_red) - a2 * v_red - a3 * a_red
                            v_probe = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_probe)
                            sphere_v_probe = sphere_v_pred + a1 * (x_probe - sphere_x_pred)
                            # Fresh scratch storage: probes never alter the actual
                            # iterate's work buffer, sticky IDs or material trial.
                            work = _assemble_sphere_contact_work_buffer(
                                model, sphere, config, x_probe, sphere_v_probe,
                                structural_displacement=reconstruct_full_solution(T, q_probe, u0),
                                structural_velocity=np.asarray(T @ v_probe, dtype=float).reshape(-1),
                                deleted_element_ids=tuple(deleted_element_ids),
                                contact_scale_by_element=damage_scale_by_element,
                                preferred_element_ids=sticky_contact_ids,
                            )
                            contact_jacobian_evaluation_count += 1
                            return np.concatenate((np.asarray(T.T @ work.load).reshape(-1), work.sphere_force))
                        contact_jacobian = numerical_contact_jacobian(
                            evaluate_contact_coordinates,
                            np.concatenate((q_trial, sphere_x_iterate)), float(sphere.radius),
                            safe_point=lambda: cancellation_safe_point(
                                cancellation_token, "sphere_impact.nonlinear.contact_jacobian"),
                        )
                        n_q = q_trial.size
                        # Negative derivative of the two physical residuals.
                        contact_jacobian[:n_q, :] *= one_plus_alpha
                        base_operator = sparse.block_diag((K_eff, float(sphere.mass) * a0 * sparse.eye(3)))
                        K_eff = (base_operator - sparse.csr_matrix(contact_jacobian)).tocsr()
                    tangent_reuse.record_tangent_assembly(refresh_reasons)
                    handle = factorize(
                        K_eff,
                        MatrixClass.GENERAL if use_contact_newton else MatrixClass.SYMMETRIC_INDEFINITE,
                        signature=(
                            f"sphere_impact.nl.effective:{dt:.16g}:{iteration}:"
                            f"descriptor={descriptor_reduction is not None}"
                        ),
                    )
                else:
                    handle = tangent_reuse.cached_handle
                    if handle is None:  # defensive; refresh_decision guards this
                        raise RuntimeError("impact tangent reuse cache is unexpectedly empty")
                factor_diagnostics = handle.diagnostics()
                solve_rhs = (
                    descriptor_reduction.effective_rhs(residual)
                    if descriptor_reduction is not None
                    else residual
                )
                if use_contact_newton:
                    solve_rhs = np.concatenate((solve_rhs,
                        sphere_force - float(sphere.mass) * a0 * (sphere_x_iterate - sphere_x_pred)))
                solved_delta = np.asarray(
                    handle.solve(solve_rhs), dtype=float
                ).reshape(-1)
                if use_contact_newton:
                    sphere_delta = solved_delta[-3:]
                    solved_delta = solved_delta[:-3]
                delta = (
                    descriptor_reduction.reconstruct_partition_state(
                        solved_delta[: descriptor_reduction.physical_dimension],
                        solved_delta[descriptor_reduction.physical_dimension :],
                    )
                    if descriptor_reduction is not None
                    else solved_delta
                )
                if refresh_tangent:
                    tangent_reuse.record_factorization(handle)
                    factorization_count += 1
                else:
                    tangent_reuse.record_reuse()
                solve_count += 1
            except SolveCancelled:
                raise
            except Exception:
                status = "nonlinear_tangent_failed"
                stop_reason = "nonlinear_tangent_factorization_failed"
                break
            if np.any(~np.isfinite(delta)) or (sphere_delta is not None and np.any(~np.isfinite(sphere_delta))):
                status = "nonlinear_iteration_failed"
                stop_reason = "nonfinite_nonlinear_increment"
                break
            factor = 1.0
            if bool(nl.line_search):
                factor = 1.0
                while factor > float(nl.min_line_search_factor) and float(np.linalg.norm(factor * delta)) > 10.0 * max(float(np.linalg.norm(q_trial)), 1.0):
                    factor *= 0.5
            tangent_reuse.observe_line_search(factor)
            q_next = q_trial + factor * delta
            sphere_a_next = sphere_force / float(sphere.mass)
            sphere_x_next = sphere_x_pred + beta * dt**2 * sphere_a_next
            if sphere_delta is not None:
                sphere_x_next = sphere_x_iterate + factor * sphere_delta
            sphere_v_next = sphere_v_pred + gamma * dt * sphere_a_next
            displacement_increment = float(np.linalg.norm(factor * delta))
            residual_energy_increment = abs(float(np.dot(factor * delta, residual)))
            if coupled_iteration is not None:
                # The first map uses the legacy zero contact-load initialization,
                # not a physical residual at x_pred: never store its secant.
                if iteration > 1:
                    previous_safeguards = coupled_iteration.globalization_count
                    coupled_next = coupled_iteration.update(
                        np.concatenate((q_trial, sphere_x_iterate)),
                        np.concatenate((q_next, sphere_x_next)),
                        signature=tuple(zip(last_contact_work.active_element_ids,
                                            last_contact_work.active_classifications)),
                    )
                    coupled_safeguard_count += coupled_iteration.globalization_count - previous_safeguards
                    if not np.all(np.isfinite(coupled_next)):
                        status = "nonlinear_iteration_failed"
                        stop_reason = "nonfinite_nonlinear_increment"
                        break
                    q_next = coupled_next[:-3]
                    sphere_x_next = coupled_next[-3:]
                actual_delta = q_next - q_trial
                # A tiny extrapolated update must never make a physical Newton
                # correction pass a displacement/energy stagnation criterion.
                displacement_increment = max(displacement_increment, float(np.linalg.norm(actual_delta)))
                residual_energy_increment = max(residual_energy_increment, abs(float(np.dot(actual_delta, residual))))
            if coupled_active:
                sphere_a_next = a0 * (sphere_x_next - sphere_x_pred)
                sphere_v_next = sphere_v_pred + gamma * dt * sphere_a_next
                sphere_x_iterate = sphere_x_next.copy()
            a_next = a0 * (q_next - q_red) - a2 * v_red - a3 * a_red
            v_next = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_next)
            full_u_next = reconstruct_full_solution(T, q_next, u0)
            full_v_next = np.asarray(T @ v_next, dtype=float).reshape(-1)
            try:
                if native_contact_state:
                    # The internal response above owns a trial for ``q_trial``;
                    # contact is evaluated at ``q_next``.  Replace it with an
                    # exact q_next candidate before resolving the native
                    # directors.  This candidate is contact-only and is always
                    # discarded; the next Newton residual re-evaluates q_next and
                    # owns the sole candidate that may be committed.
                    begin_state_evaluation(
                        committed_states,
                        model=model,
                        displacements=full_u_next,
                    )
                last_contact_work = _assemble_sphere_contact_work_buffer(
                    model,
                    sphere,
                    config,
                    sphere_x_next,
                    sphere_v_next,
                    structural_displacement=full_u_next,
                    structural_velocity=full_v_next,
                    deleted_element_ids=tuple(deleted_element_ids),
                    contact_scale_by_element=damage_scale_by_element,
                    preferred_element_ids=sticky_contact_ids,
                    work_buffer=contact_work_buffer,
                    native_rotation_context=(
                        committed_states if native_contact_state else None
                    ),
                )
            finally:
                if native_contact_state:
                    discard_active_state_candidate(committed_states)
            if last_contact_work.selected_indices.size:
                sticky_contact_ids = last_contact_work.active_element_ids
            contact_scale = max(float(np.linalg.norm(last_contact_work.sphere_force)), 1.0)
            contact_change = float(np.linalg.norm(last_contact_work.sphere_force -
                (float(sphere.mass) * sphere_a_next if coupled_active else sphere_force)))
            if coupled_iteration is not None:
                next_signature = tuple(zip(last_contact_work.active_element_ids, last_contact_work.active_classifications))
                if next_signature and coupled_signature and next_signature != coupled_signature:
                    coupled_iteration = ImpactCoupledIteration()
                if next_signature:
                    coupled_signature = next_signature
            np.copyto(contact_load, last_contact_work.load)
            np.copyto(sphere_force, last_contact_work.sphere_force)
            tangent_reuse.observe_contact_signature(
                tuple(
                    zip(
                        last_contact_work.active_element_ids,
                        last_contact_work.active_classifications,
                    )
                )
            )
            q_trial = q_next
        if last_contact_work is not None:
            records = last_contact_work.materialize_records(
                SphereContactRecord,
                contact_geometry.node_ids,
            )
        if not converged:
            if status in {"nonlinear_tangent_failed", "nonlinear_iteration_failed"}:
                pass
            elif cutback_count < int(nl.max_cutbacks) and dt * 0.5 >= float(nl.min_dt):
                discard_active_state_candidate(committed_states)
                (
                    q_red,
                    v_red,
                    a_red,
                    sphere_position,
                    sphere_velocity,
                    sphere_acceleration,
                    committed_states,
                    deleted_element_ids,
                    plastic_damage_states,
                    damage_scale_by_element,
                    load_prev,
                    load_prev_red,
                    sphere_force_prev,
                    F_int_prev,
                    F_int_prev_red,
                ) = pre_state
                midpoint = 0.5 * (float(segment_start) + float(sub_time))
                segments.insert(0, (step_index, midpoint, float(sub_time), False))
                segments.insert(0, (step_index, float(segment_start), midpoint, False))
                cutback_count += 1
                event_substep_count += 1
                distress_halvings = min(distress_halvings + 1, 2)
                step_diagnostics.append(
                    {
                        "step_index": int(step_index),
                        "time": float(sub_time),
                        "dt": float(dt),
                        "status": "nonlinear_cutback_retry",
                        "residual_norm": float(residual_norm),
                        "effective_residual_tolerance": float(nl.residual_tolerance) * float(reference),
                        "displacement_increment": float(displacement_increment),
                        "contact_force_change": float(contact_change),
                        "effective_force_tolerance": max(float(nl.contact_force_tolerance) * float(contact_scale), float(config.force_tolerance)),
                    }
                )
                nonlinear_failure_summary = dict(step_diagnostics[-1])
                nonlinear_failure_summary["suggestion"] = (
                    "The nonlinear contact/plasticity iteration recovered by cutting the time step. "
                    "If failures persist, reduce dt/penalty, allow more cutbacks, or loosen the nonlinear contact tolerance."
                )
                continue
            else:
                status = "nonlinear_iteration_failed"
                stop_reason = "maximum_iterations_or_cutbacks_reached"
                active_ids = [int(record.element_id) for record in records]
                # Residual decomposition at the failed iterate: which physics
                # carries the unconverged force imbalance.  Essential for
                # diagnosing stagnation (tiny displacement increments with a
                # stuck residual) versus genuine divergence.
                decomposition: Dict[str, float] = {}
                try:
                    full_u_fail = reconstruct_full_solution(T, q_trial, u0)
                    if impact_reduced_assembly.active:
                        F_int_fail_red, _K_unused, _states_unused = impact_reduced_assembly.assemble(
                            full_u_fail,
                            committed_states,
                            tangent=False,
                        )
                    else:
                        F_int_fail, _K_unused, _states_unused = assemble_nonlinear(
                            full_u_fail, committed_states, tangent=False, scales=damage_scale_by_element
                        )
                        F_int_fail_red = np.asarray(T.T @ F_int_fail, dtype=float).reshape(-1)
                    a_fail = a0 * (q_trial - q_red) - a2 * v_red - a3 * a_red
                    v_fail = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_fail)
                    decomposition = {
                        "residual_contact_norm": float(np.linalg.norm(np.asarray(T.T @ contact_load, dtype=float))),
                        "residual_base_load_norm": float(np.linalg.norm(np.asarray(T.T @ base_load, dtype=float))),
                        "residual_internal_norm": float(np.linalg.norm(F_int_fail_red)),
                        "residual_damping_norm": float(np.linalg.norm(C_red @ v_fail)),
                        "residual_inertia_norm": float(np.linalg.norm(M_red @ a_fail)),
                    }
                except Exception:
                    decomposition = {}
                nonlinear_failure_summary = {
                    "step_index": int(step_index),
                    "time": float(sub_time),
                    "dt": float(dt),
                    "status": str(status),
                    "stop_reason": str(stop_reason),
                    "iterations": int(nl.max_iterations),
                    "cutbacks": int(cutback_count),
                    "residual_norm": float(residual_norm),
                    "effective_residual_tolerance": float(nl.residual_tolerance) * float(reference),
                    "displacement_increment": float(displacement_increment),
                    "residual_energy_increment": float(residual_energy_increment),
                    "impact_energy_reference": float(impact_energy_reference),
                    "contact_force_change": float(contact_change),
                    "effective_force_tolerance": max(float(nl.contact_force_tolerance) * float(contact_scale), float(config.force_tolerance)),
                    "active_element_ids": active_ids,
                    "max_penetration": float(max((record.penetration for record in records), default=0.0)),
                    **decomposition,
                    "suggestion": (
                        "The nonlinear impact solve exhausted its time-step cutbacks. "
                        "Use a smaller manual dt, increase NL cutbacks, reduce contact penalty/target penetration stiffness, "
                        "or run the linear collision path first to confirm the contact setup."
                    ),
                }
            if not iteration_counts:
                iteration_counts.append(int(nl.max_iterations))
            discard_active_state_candidate(committed_states)
            break

        accepted_full_u = reconstruct_full_solution(T, q_next, u0)
        try:
            committed_states = commit_state_candidate(
                committed_states,
                trial_states,
                accepted_full_displacement=accepted_full_u,
                model=model,
            )
        except BaseException:
            discard_active_state_candidate(committed_states)
            raise
        contact_norm = float(np.linalg.norm(sphere_force))
        if contact_norm > 0.0:
            contact_step_count += 1
            active_contact_duration += dt
            separation_elapsed = 0.0
        elif contact_step_count > 0 and separation_stop_time > 0.0:
            separation_elapsed += dt
        if records:
            max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
            peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
        impulse += 0.5 * (load_prev + base_load + contact_load) * dt
        sphere_impulse += 0.5 * (sphere_force_prev + sphere_force) * dt
        load_prev = base_load + contact_load
        load_prev_red = np.asarray(T.T @ load_prev, dtype=float).reshape(-1)
        if impact_reduced_assembly.active:
            # The reduced plan owns a persistent force buffer, so retain an
            # accepted-step copy for HHT history and internal work.
            F_int_prev_red = np.asarray(F_int_red, dtype=float).copy()
        else:
            F_int_prev = F_int
            F_int_prev_red = np.asarray(T.T @ F_int_prev, dtype=float).reshape(-1)
        sphere_force_prev = sphere_force.copy()
        q_red, v_red, a_red = q_next, v_next, a_next
        sphere_position, sphere_velocity, sphere_acceleration = sphere_x_next, sphere_v_next, sphere_a_next
        new_damage_records, damage_changed, max_damage_util = _plastic_impact_damage_update(
            model,
            committed_states,
            plastic_damage_config,
            deleted_element_ids,
            plastic_damage_states,
            step_index=int(step_index),
            time_value=float(sub_time),
        )
        if new_damage_records:
            plastic_damage_records.extend(new_damage_records)
            damage_deleted_element_ids.update(record.element_id for record in new_damage_records)
        if plastic_damage_config is not None:
            damage_scale_by_element = {
                int(element_id): float(state.get("scale", 1.0) or 1.0)
                for element_id, state in plastic_damage_states.items()
                if float(state.get("scale", 1.0) or 1.0) < 1.0
            }
            for element_id in deleted_element_ids:
                damage_scale_by_element[int(element_id)] = float(plastic_damage_config.residual_stiffness_fraction)
            if damage_changed:
                filtered_base = filtered_load_case_for_deleted_elements(base_load_case, deleted_element_ids)
                base_load, base_load_info = assemble_load_vector(model, filtered_base)
                base_load_red = np.asarray(T.T @ base_load, dtype=float).reshape(-1)
                _K_scaled, M_scaled, linear_matrix_terms, damage_matrix_plan = (
                    _assemble_damaged_linear_matrices_with_gate(
                        model,
                        damage_scale_by_element,
                        damage_matrix_gate,
                        opportunity_index=int(step_index),
                        cached_terms=linear_matrix_terms,
                        plan=damage_matrix_plan,
                    )
                )
                M_red = (T.T @ M_scaled @ T).tocsr()
                C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red0).tocsr()
                scoped_total = sum(
                    1
                    for element in model.mesh.elements.values()
                    if element_fracture_category(element) in set(plastic_damage_config.element_scope)
                )
                if len(deleted_element_ids) / max(scoped_total, 1) > float(plastic_damage_config.max_deleted_fraction) + 1.0e-12:
                    status = "max_deleted_fraction_reached"
                    stop_reason = "plastic_impact_damage_max_deleted_fraction_reached"
        state_summary = _nonlinear_state_summary(committed_states)
        step_diagnostics.append(
            {
                "step_index": int(step_index),
                "time": float(sub_time),
                "dt": float(dt),
                "status": "converged",
                "iterations": int(iteration_counts[-1]),
                "residual_norm": float(residual_norm),
                "displacement_increment": float(displacement_increment),
                "contact_force_change": float(contact_change),
                "convergence_reason": str(convergence_reason),
                "active_element_ids": [int(record.element_id) for record in records],
                "max_penetration": float(max((record.penetration for record in records), default=0.0)),
                "max_equivalent_plastic_strain": float(state_summary.get("max_equivalent_plastic_strain", 0.0) or 0.0),
                "max_damage_utilization": float(max_damage_util),
            }
        )
        step_complete = float(sub_time) >= float(times[step_index]) - 1.0e-12 * max(abs(float(times[step_index])), 1.0)
        if step_complete and (step_index % int(transient_config.save_every) == 0 or step_index == len(times) - 1):
            save_state(sub_time, q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, sphere_force, records, state_summary=state_summary)
            pending_save = None
        else:
            pending_save = (float(sub_time), sphere_force.copy(), records)
        if status == "max_deleted_fraction_reached":
            break
        if status == "completed" and contact_step_count > 0 and separation_stop_time > 0.0 and separation_elapsed >= separation_stop_time:
            stop_reason = "completed_after_contact_separation"
            break

    if pending_save is not None:
        save_state(
            pending_save[0], q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, pending_save[1], pending_save[2]
        )

    if contact_step_count == 0 and status == "completed":
        status = "no_contact"
        stop_reason = "no_contact"
    impulse_resultant = load_vector_resultant(model, impulse)
    max_penetration_ratio = float(max_penetration) / max(float(sphere.radius), 1.0e-12)
    contact_duration = float(active_contact_duration)
    initial_sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    final_sphere_velocity = np.asarray(saved_sphere_v[-1], dtype=float) if saved_sphere_v else sphere_velocity
    sphere_momentum_change = float(sphere.mass) * (final_sphere_velocity - initial_sphere_velocity)
    sphere_momentum_balance_error = float(np.linalg.norm(sphere_impulse - sphere_momentum_change))
    history_width = total_dofs if history_storage_mode == "full" else int(0 if history_dof_indices is None else len(history_dof_indices))
    saved_u_array = np.vstack(saved_u) if saved_u else np.zeros((0, history_width), dtype=float)
    saved_v_array = np.vstack(saved_v) if saved_v else np.zeros((0, history_width), dtype=float)
    saved_a_array = np.vstack(saved_a) if saved_a else np.zeros((0, history_width), dtype=float)
    node_histories: Dict[int, np.ndarray] = {}
    for node_id in output_node_ids:
        values = node_history_values.get(int(node_id), [])
        node_histories[int(node_id)] = np.vstack(values) if values else np.zeros((0, 6), dtype=float)
    recovery_memory = estimate_model_memory(
        model,
        transient_saved_steps=len(saved_times),
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(recovery_memory, transient_config.resource_config, context="solve_transient_sphere_impact.nonlinear.recovery")
    policy_metadata = recovery_metadata(recovery, transient_config.resource_config, recovery_memory)
    assembly_info = {"stiffness": stiffness_info, "mass": mass_info, "load": base_load_info}
    plastic_damage_summary = _plastic_impact_damage_summary(
        model,
        plastic_damage_config,
        plastic_damage_states,
        damage_deleted_element_ids,
        plastic_damage_records,
        warnings=[warning for warning in warnings if str(warning).startswith("NONLINEAR_IMPACT")],
    )
    erosion_summary = {
        "all_eroded_element_ids": sorted(int(element_id) for element_id in deleted_element_ids),
        "damage_triggered_element_ids": sorted(int(element_id) for element_id in damage_deleted_element_ids),
        "active_softened_element_ids": sorted(
            int(element_id)
            for element_id, scale in damage_scale_by_element.items()
            if int(element_id) not in deleted_element_ids and float(scale) < 1.0
        ),
        "residual_stiffness_model": "nonlinear element force/tangent and mass scaling; topology, nodes, MPCs retained",
    }
    result_case = make_result_case(
        name=f"sphere_impact_nonlinear:{sphere.name}",
        analysis_type="sphere_impact_nonlinear_transient",
        load_cases=() if base_load_case is None else (base_load_case,),
        assembly_info=assembly_info,
        solver_info={"backend": factor_diagnostics, "convergence_info": {"status": status, "stop_reason": stop_reason}},
        recovery={
            "displacement_history": True,
            "velocity_history": True,
            "acceleration_history": True,
            "sphere_history": True,
            "contact_history": bool(config.save_contact_history),
            "element_state_history": bool(nl.record_element_state_history),
            "history_storage_mode": history_storage_mode,
            **policy_metadata["recovery"],
        },
        settings={
            "dt": transient_config.dt,
            "t_end": transient_config.t_end,
            "beta": beta,
            "gamma": gamma,
            "hht_alpha": alpha_h,
            "rayleigh_alpha": transient_config.rayleigh_alpha,
            "rayleigh_beta": transient_config.rayleigh_beta,
            "sphere": {
                "name": sphere.name,
                "radius": float(sphere.radius),
                "mass": float(sphere.mass),
                "start_point": sphere.initial_position.tolist(),
                "travel_direction": sphere.direction_unit.tolist(),
                "speed": float(sphere.speed),
                "t_start": float(sphere.t_start),
            },
            "contact": config.to_dict(),
            "nonlinear": nl.to_dict(),
            "plastic_damage": None if plastic_damage_config is None else plastic_damage_config.to_dict(),
        },
        metadata={
            "resources": policy_metadata.get("resources"),
            "memory_estimate": policy_metadata.get("memory_estimate"),
            "contact_scope": "single rigid sphere to shell target; frictionless normal penalty with iterative active-set updates",
            "nonlinear_scope": "implicit Newmark; current shell/beam nonlinear element formulations; approximate contact tangent",
            "solution_control": "implicit_newmark_time_domain",
            "arc_length_applicability": "not_applicable_to_dynamic_impact",
            "solver_convergence": {"status": status, "stop_reason": stop_reason},
            "verification_gate": "nonlinear_contact",
        },
    ).to_dict()
    total_energy = (
        np.asarray(energy_kinetic, dtype=float)
        + np.asarray(energy_strain, dtype=float)
        + np.asarray(sphere_kinetic, dtype=float)
    )
    nonzero_energy = total_energy[np.abs(total_energy) > 1.0e-30]
    energy_drift = 0.0
    if nonzero_energy.size:
        energy_drift = float((np.max(nonzero_energy) - np.min(nonzero_energy)) / max(abs(nonzero_energy[0]), 1.0e-30))
    impact_reduced_diagnostics = impact_reduced_assembly.diagnostics()
    impact_reduced_diagnostics["full_coordinate_assembly_count"] = int(
        full_coordinate_assembly_count
    )
    public_committed_states = materialize_state_mapping(committed_states)
    diagnostics = {
        "coupled_contact_acceleration": {"requested": nl.coupled_contact_acceleration,
            "active": coupled_active, "method": coupling_method,
            "fallback_reason": coupling_fallback, "depth": 4,
            "scalar_bracket_safeguards": coupled_safeguard_count,
            "contact_jacobian_force_evaluations": contact_jacobian_evaluation_count},
        "method": "nonlinear_newmark_sphere_penalty_contact",
        "solution_control": "implicit_newmark_time_domain",
        "arc_length_applicability": "not_applicable_to_dynamic_impact",
        "status": status,
        "stop_reason": stop_reason,
        "warnings": warnings,
        "num_steps": max(int(len(times) - 1), 0),
        "num_substeps": int(total_substep_count),
        "event_substep_count": int(event_substep_count),
        "preemptive_substep_count": int(preemptive_substep_count),
        "cutback_count": int(cutback_count),
        "num_saved_steps": len(saved_times),
        "num_reduced_dofs": int(M_red.shape[0]),
        "num_dynamic_physical_dofs": int(
            M_red.shape[0]
            if descriptor_reduction is None
            else descriptor_reduction.physical_dimension
        ),
        "declared_algebraic_contact_dynamics": descriptor_certificate,
        "contact_step_count": int(contact_step_count),
        "active_contact_duration": float(contact_duration),
        "separation_stop_time": float(separation_stop_time),
        "post_contact_separation_time": float(separation_elapsed if contact_step_count > 0 else 0.0),
        "max_penetration_ratio": float(max_penetration_ratio),
        "sphere_momentum_balance_error": float(sphere_momentum_balance_error),
        "iteration_counts": iteration_counts,
        "contact_step_diagnostics": step_diagnostics,
        "nonlinear_failure_summary": nonlinear_failure_summary,
        "factorization_count": int(factorization_count),
        "solve_count": int(solve_count),
        "direct_reduced_assembly_count": int(
            impact_reduced_assembly.direct_reduced_assembly_count
        ),
        "full_coordinate_assembly_count": int(full_coordinate_assembly_count),
        "impact_reduced_assembly": impact_reduced_diagnostics,
        "tangent_assembly_count": int(tangent_reuse.tangent_assembly_count),
        "tangent_reuse_count": int(tangent_reuse.tangent_reuse_count),
        "factorization_reuse_count": int(tangent_reuse.factorization_reuse_count),
        "refresh_reason_counts": dict(sorted(tangent_reuse.refresh_reason_counts.items())),
        "active_contact_set_changes": int(tangent_reuse.active_contact_set_changes),
        "tangent_reuse": tangent_reuse.diagnostics(),
        "contact_public_materialization_count": int(
            contact_work_counters.public_materialization_count
        ),
        "contact_direct_full_scatter_count": int(
            contact_work_counters.direct_full_scatter_count
        ),
        "contact_work_buffer": {
            "path": "compact_internal_candidates_lazy_public_records",
            "lazy_public_materialization": True,
            **contact_work_counters.diagnostics(),
        },
        "linear_matrix_terms_cached": linear_matrix_terms is not None,
        "damage_matrix_plan_selection": damage_matrix_gate.diagnostics(),
        "damage_matrix_plan": None if damage_matrix_plan is None else damage_matrix_plan.diagnostics(),
        "initial_mass_factorization": mass_handle.diagnostics(),
        "effective_stiffness_factorization": factor_diagnostics,
        "constraint_info": constraint_info,
        "constraint_postcheck": constraint_residual_summary(
            model,
            np.asarray(T @ q_red + u0, dtype=float).reshape(-1),
        ),
        "stiffness": stiffness_info,
        "mass": mass_info,
        "base_load": base_load_info,
        "contact_config": config.to_dict(),
        "nonlinear_config": nl.to_dict(),
        "base_load_equilibrated": bool(base_load_equilibrated),
        "strain_summary": _nonlinear_state_summary(public_committed_states),
        "element_states": public_committed_states,
        "nonlinear_state_storage": state_storage_info.get(
            "nonlinear_state_storage", {}
        ),
        "element_state_history": element_state_history,
        "state_von_mises_history": tuple(state_von_mises_history),
        "plastic_impact_damage_summary": plastic_damage_summary,
        "impact_damage_summary": plastic_damage_summary,
        "erosion_summary": erosion_summary,
        "contact_validation": validation.to_dict(),
        "sphere": result_case["analysis_case"]["settings"]["sphere"],
        "kinetic_energy": energy_kinetic,
        "strain_energy": energy_strain,
        "sphere_kinetic_energy": sphere_kinetic,
        "plastic_work_proxy": plastic_work_history,
        "max_relative_energy_drift": energy_drift,
        "result_case": result_case,
    }
    return SphereImpactResult(
        times=np.asarray(saved_times, dtype=float),
        displacements=saved_u_array,
        velocities=saved_v_array,
        accelerations=saved_a_array,
        node_histories=node_histories,
        sphere_positions=np.vstack(saved_sphere_x) if saved_sphere_x else np.zeros((0, 3), dtype=float),
        sphere_velocities=np.vstack(saved_sphere_v) if saved_sphere_v else np.zeros((0, 3), dtype=float),
        sphere_accelerations=np.vstack(saved_sphere_a) if saved_sphere_a else np.zeros((0, 3), dtype=float),
        contact_force_history=np.vstack(saved_contact_force) if saved_contact_force else np.zeros((0, 3), dtype=float),
        active_contact_history=tuple(saved_contacts),
        load_impulse=impulse,
        force_impulse=impulse_resultant.force,
        moment_impulse=impulse_resultant.moment,
        sphere_impulse=sphere_impulse,
        max_penetration=float(max_penetration),
        max_penetration_ratio=float(max_penetration_ratio),
        peak_contact_force=float(peak_contact_force),
        contact_duration=float(contact_duration),
        sphere_momentum_balance_error=float(sphere_momentum_balance_error),
        peak_displacement=float(peak_displacement),
        peak_displacement_node=peak_displacement_node,
        status=status,
        diagnostics=diagnostics,
        result_case=result_case,
    )


@resource_threaded
def solve_transient_sphere_impact(
    model: "FEModel",
    transient_config: TransientConfig,
    sphere: RigidSphereImpact,
    contact_config: Optional[SphereContactConfig] = None,
    base_load_case: Optional[LoadCase] = None,
    fracture_config: Optional[ImpactFractureConfig] = None,
    damage_config: Optional[ImpactDamageConfig] = None,
    nonlinear_config: Optional[NonlinearTransientConfig] = None,
    plastic_damage_config: Optional[PlasticImpactDamageConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> SphereImpactResult:
    """Solve a limited rigid-sphere-to-shell impact transient."""

    cancellation_safe_point(cancellation_token, "sphere_impact.start")
    from .element_capabilities import require_model_element_capabilities

    require_model_element_capabilities(
        model,
        ("contact_state", "transient_algebraic_dynamics", "nonlinear_geometry"),
        context="sphere-impact analysis",
    )
    config = _resolved_contact_config(model, sphere, contact_config)
    if fracture_config is not None and not isinstance(fracture_config, ImpactFractureConfig):
        raise TypeError("fracture_config must be an ImpactFractureConfig or None")
    if damage_config is not None and not isinstance(damage_config, ImpactDamageConfig):
        raise TypeError("damage_config must be an ImpactDamageConfig or None")
    if nonlinear_config is not None and not isinstance(nonlinear_config, NonlinearTransientConfig):
        raise TypeError("nonlinear_config must be a NonlinearTransientConfig or None")
    if plastic_damage_config is not None and not isinstance(plastic_damage_config, PlasticImpactDamageConfig):
        raise TypeError("plastic_damage_config must be a PlasticImpactDamageConfig or None")
    nonlinear_enabled = nonlinear_config is not None and bool(nonlinear_config.enabled)
    if not nonlinear_enabled:
        _require_user_capacity_for_orthotropic_linear_damage(model, damage_config)
    validation = validate_contact_configuration(model, sphere, config, transient_config)
    if validation.errors:
        codes = ", ".join(issue.code for issue in validation.errors)
        raise ValueError(f"Invalid sphere contact configuration: {codes}")
    if nonlinear_enabled:
        from .e4_pl_element import _elastic_batch_baseline_scope
        with _elastic_batch_baseline_scope():
            return _solve_transient_sphere_impact_nonlinear(
                model,
                transient_config,
                sphere,
                config,
                validation,
                base_load_case=base_load_case,
                nonlinear_config=nonlinear_config,
                plastic_damage_config=plastic_damage_config,
                progress_callback=progress_callback,
                status_callback=status_callback,
                cancellation_token=cancellation_token,
            )
    model.apply_boundary_conditions()
    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    total_dofs = model.mesh.dof_manager.total_dofs
    deleted_element_ids: set[int] = set()
    fracture_deleted_element_ids: set[int] = set()
    damage_deleted_element_ids: set[int] = set()
    fracture_records: List[DeletedElementRecord] = []
    damage_records: List[DeletedElementRecord] = []
    max_fracture_utilization = 0.0
    impact_damage_states: Dict[int, Dict[str, Any]] = {}
    max_damage_utilization = 0.0
    damage_scale_by_element: Dict[int, float] = {}
    linear_matrix_terms: Optional[Tuple[int, Tuple[LinearElementMatrixTerms, ...]]] = None
    damage_matrix_plan: Optional[DamageMatrixPlan] = None
    base_load, base_load_info = assemble_load_vector(model, base_load_case)
    zero_load = np.zeros(total_dofs, dtype=float)
    K_red, _zero_red, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, zero_load, model)
    M_red = (T.T @ M @ T).tocsr()
    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red).tocsr()
    if float(np.linalg.norm(M_red.diagonal())) <= 0.0 and M_red.nnz == 0:
        raise ValueError("Sphere impact transient requires a non-zero structural mass matrix; set material density values.")

    times = _time_grid(transient_config)
    recovery = transient_config.recovery
    output_node_ids = tuple(int(node_id) for node_id in (transient_config.output_nodes or ()))
    if recovery is not None and not output_node_ids and recovery.node_ids is not None:
        output_node_ids = tuple(int(node_id) for node_id in recovery.node_ids)
    history_storage_mode = "full" if recovery is None else str(recovery.history_mode)
    if recovery is not None and history_storage_mode == "full" and not recovery.store_full_histories:
        history_storage_mode = "selected"
    if history_storage_mode == "selected" and not output_node_ids and (recovery is None or recovery.include_displacements):
        output_node_ids = tuple(int(node_id) for node_id in model.mesh.nodes)
    history_dof_indices = _node_dof_indices(model, output_node_ids) if history_storage_mode == "selected" else None
    estimated_saved_steps = _saved_step_count(times, transient_config.save_every)
    preflight_memory = estimate_model_memory(
        model,
        transient_saved_steps=estimated_saved_steps,
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(preflight_memory, transient_config.resource_config, context="solve_transient_sphere_impact")
    damage_matrix_gate = DamageMatrixPlanGate(
        total_opportunities=max(int(len(times) - 1), 0),
        preflight_memory_bytes=int(preflight_memory.total_bytes_estimate),
        configured_memory_limit_bytes=(
            None
            if transient_config.resource_config is None
            or transient_config.resource_config.memory_limit_bytes is None
            else int(transient_config.resource_config.memory_limit_bytes)
        ),
    )

    q = _full_initial_vector(transient_config.initial_displacement, total_dofs)
    v_full = _full_initial_vector(transient_config.initial_velocity, total_dofs)
    q_red = np.asarray((q - u0)[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    v_red = np.asarray(v_full[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    F0_red = _reduced_load(T, K, u0, base_load)
    mass_handle = factorize(M_red, MatrixClass.SYMMETRIC_SEMIDEFINITE, signature="sphere_impact.initial_mass")
    a_red = np.asarray(mass_handle.solve(F0_red - C_red @ v_red - K_red @ q_red), dtype=float).reshape(-1)

    sphere_position = sphere.initial_position
    sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    sphere_acceleration = np.zeros(3, dtype=float)

    saved_times: List[float] = []
    saved_u: List[np.ndarray] = []
    saved_v: List[np.ndarray] = []
    saved_a: List[np.ndarray] = []
    saved_sphere_x: List[np.ndarray] = []
    saved_sphere_v: List[np.ndarray] = []
    saved_sphere_a: List[np.ndarray] = []
    saved_contact_force: List[np.ndarray] = []
    saved_contacts: List[Tuple[Dict[str, Any], ...]] = []
    node_history_values: Dict[int, list[np.ndarray]] = {int(node_id): [] for node_id in output_node_ids}
    peak_displacement = 0.0
    peak_displacement_node = None
    max_penetration = 0.0
    peak_contact_force = 0.0
    contact_step_count = 0
    active_contact_duration = 0.0
    iteration_counts: List[int] = []
    energy_kinetic: List[float] = []
    energy_strain: List[float] = []
    sphere_kinetic: List[float] = []

    def save_state(
        time: float,
        q_state: np.ndarray,
        v_state: np.ndarray,
        a_state: np.ndarray,
        sphere_x: np.ndarray,
        sphere_v: np.ndarray,
        sphere_a: np.ndarray,
        sphere_force: np.ndarray,
        records: Tuple[SphereContactRecord, ...],
    ) -> None:
        nonlocal peak_displacement, peak_displacement_node, max_penetration, peak_contact_force
        full_u = reconstruct_full_solution(T, q_state, u0)
        full_v = np.asarray(T @ v_state, dtype=float).reshape(-1)
        full_a = np.asarray(T @ a_state, dtype=float).reshape(-1)
        saved_times.append(float(time))
        if history_storage_mode == "full":
            saved_u.append(full_u)
            saved_v.append(full_v)
            saved_a.append(full_a)
        elif history_storage_mode == "selected":
            indices = np.asarray(history_dof_indices if history_dof_indices is not None else (), dtype=np.intp)
            saved_u.append(full_u[indices])
            saved_v.append(full_v[indices])
            saved_a.append(full_a[indices])
        for node_id in output_node_ids:
            node = model.mesh.get_node(int(node_id))
            if node is not None:
                node_history_values[int(node_id)].append(full_u[np.asarray(node.dofs, dtype=np.intp)])
        current_peak, current_node = _translation_peak(model, full_u)
        if current_peak > peak_displacement:
            peak_displacement = current_peak
            peak_displacement_node = current_node
        if records:
            max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
            peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
        saved_sphere_x.append(np.asarray(sphere_x, dtype=float).copy())
        saved_sphere_v.append(np.asarray(sphere_v, dtype=float).copy())
        saved_sphere_a.append(np.asarray(sphere_a, dtype=float).copy())
        saved_contact_force.append(np.asarray(sphere_force, dtype=float).copy())
        saved_contacts.append(tuple(record.to_dict() for record in records) if config.save_contact_history else tuple())
        energy_kinetic.append(0.5 * float(v_state @ (M_red @ v_state)))
        energy_strain.append(0.5 * float(q_state @ (K_red @ q_state)))
        sphere_kinetic.append(0.5 * float(sphere.mass) * float(sphere_v @ sphere_v))
        if progress_callback is not None and history_storage_mode == "full":
            emit_progress(
                progress_callback,
                "sphere_impact_live_step",
                "sphere_impact.linear.integration",
                completed=float(time),
                total=float(transient_config.t_end),
                time_s=float(time),
                step_index=int(len(saved_times) - 1),
                displacement=full_u.copy(),
                sphere_position=np.asarray(sphere_x, dtype=float).copy(),
                sphere_radius=float(sphere.radius),
                contact_force=np.asarray(sphere_force, dtype=float).copy(),
                active_contacts=tuple(record.to_dict() for record in records) if config.save_contact_history else tuple(),
            )

    initial_load, initial_sphere_force, initial_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        sphere_position,
        sphere_velocity,
        structural_displacement=reconstruct_full_solution(T, q_red, u0),
        structural_velocity=np.asarray(T @ v_red, dtype=float).reshape(-1),
        deleted_element_ids=tuple(deleted_element_ids),
        contact_scale_by_element=damage_scale_by_element,
    )
    save_state(float(times[0]), q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, initial_sphere_force, initial_records)
    sticky_contact_ids: Tuple[int, ...] = tuple(int(record.element_id) for record in initial_records)
    load_prev = base_load + initial_load
    sphere_force_prev = initial_sphere_force.copy()
    impulse = np.zeros(total_dofs, dtype=float)
    sphere_impulse = np.zeros(3, dtype=float)

    factorization_count = 0
    solve_count = 0
    cached_dt = None
    cached_solver = None
    cached_solver_diagnostics: Dict[str, Any] = {}
    status = "completed"
    stop_reason = "completed"
    warnings: List[str] = []
    for issue in validation.warnings:
        warnings.append(f"{issue.code}: {issue.message}")
    warnings.extend(
        _impact_damage_preflight_warnings(
            model, damage_config, sphere, fracture_config=fracture_config, beam_contact=bool(config.beam_contact)
        )
    )
    if fracture_config is not None:
        warnings.append(
            "IMPACT_FRACTURE001: impact fracture uses contact-observable thresholds and residual element erosion; "
            "it is not ductile crack propagation or material nonlinear impact."
        )
    if damage_config is not None:
        warnings.append(
            "IMPACT_DAMAGE001: impact damage uses engineering contact-demand utilization and element softening/erosion; "
            "it is not crack propagation, material nonlinear impact, or validated fracture mechanics."
        )

    alpha_h, beta, gamma = transient_config.integration_parameters()
    one_plus_alpha = 1.0 + alpha_h
    F_red_prev = _reduced_load(T, K, u0, load_prev)
    total_substep_count = 0
    event_substep_count = 0
    step_diagnostics: List[Dict[str, Any]] = []
    fracture_limit_warning_emitted = False
    damage_deletion_warning_emitted = False
    damage_limit_warning_emitted = False
    eroded_matrix_rebuild_count = 0
    damage_state_update_count = 0
    separation_elapsed = 0.0
    separation_stop_time = float(config.post_separation_time)
    stop_after_separation = False
    # Automatic contact-period substepping: while contact is active a coarse
    # step under-resolves the penalty oscillation and produces spurious forces
    # or non-convergence (the CONTACT006 warning condition).  Refine the step
    # to keep dt below a fraction of the penalty period, capped by
    # max_event_substeps, and only while contact is active so free flight and
    # post-separation stay fast.
    contact_period_dt = 0.2 * np.sqrt(float(sphere.mass) / max(float(config.penalty_stiffness), 1.0e-30))
    contact_active_last_step = bool(initial_records)
    for step_index in range(1, len(times)):
        cancellation_safe_point(
            cancellation_token,
            f"sphere_impact.linear.step:{step_index}",
        )
        target_time = float(times[step_index])
        dt_total = float(times[step_index] - times[step_index - 1])
        if dt_total <= 0.0:
            continue
        travel_scale = max(float(sphere.radius) * float(config.max_sphere_travel_fraction), 1.0e-12)
        predicted_travel = max(float(np.linalg.norm(sphere_velocity)) * dt_total, float(sphere.speed) * dt_total)
        n_travel = max(int(np.ceil(predicted_travel / travel_scale)), 1)
        n_contact = int(np.ceil(dt_total / contact_period_dt)) if contact_active_last_step and contact_period_dt > 0.0 else 1
        n_substeps = min(max(n_travel, n_contact, 1), int(config.max_event_substeps))
        contact_active_this_step = False
        total_substep_count += n_substeps
        if n_substeps > 1:
            event_substep_count += n_substeps - 1
        last_records: Tuple[SphereContactRecord, ...] = tuple()
        last_sphere_force = np.zeros(3, dtype=float)
        last_step_diag: Dict[str, Any] = {}
        for substep in range(n_substeps):
            cancellation_safe_point(
                cancellation_token,
                f"sphere_impact.linear.step:{step_index}.substep:{substep + 1}",
            )
            sub_time = float(times[step_index - 1]) + (substep + 1) * dt_total / n_substeps
            dt = dt_total / n_substeps
            if sub_time - dt < sphere.t_start <= sub_time and np.linalg.norm(sphere_velocity) == 0.0:
                sphere_velocity = sphere.travel_velocity.copy()
            a0 = 1.0 / (beta * dt**2)
            a1 = gamma / (beta * dt)
            a2 = 1.0 / (beta * dt)
            a3 = 1.0 / (2.0 * beta) - 1.0
            a4 = gamma / beta - 1.0
            a5 = dt * (gamma / (2.0 * beta) - 1.0)
            if cached_solver is None or cached_dt is None or not np.isclose(dt, cached_dt):
                K_eff = (one_plus_alpha * K_red + a0 * M_red + one_plus_alpha * a1 * C_red).tocsr()
                cached_solver = factorize(K_eff, MatrixClass.SYMMETRIC_INDEFINITE, signature=f"sphere_impact.effective:{dt:.16g}")
                cached_solver_diagnostics = cached_solver.diagnostics()
                cached_dt = dt
                factorization_count += 1

            sphere_x_pred = sphere_position + dt * sphere_velocity + dt**2 * (0.5 - beta) * sphere_acceleration
            sphere_v_pred = sphere_velocity + dt * (1.0 - gamma) * sphere_acceleration
            contact_load = np.zeros(total_dofs, dtype=float)
            sphere_force = np.zeros(3, dtype=float)
            records: Tuple[SphereContactRecord, ...] = tuple()
            converged = False
            q_next = q_red.copy()
            v_next = v_red.copy()
            a_next = a_red.copy()
            sphere_x_next = sphere_x_pred.copy()
            sphere_v_next = sphere_v_pred.copy()
            sphere_a_next = np.zeros(3, dtype=float)
            force_change = 0.0
            penetration_change = 0.0
            use_aitken = str(config.contact_relaxation) == "aitken"
            relaxation = 1.0
            fixed_point_residual_prev: Optional[np.ndarray] = None
            for iteration in range(1, int(config.max_contact_iterations) + 1):
                cancellation_safe_point(
                    cancellation_token,
                    f"sphere_impact.linear.step:{step_index}.substep:{substep + 1}.iteration:{iteration}",
                )
                load_next = base_load + contact_load
                F_red = _reduced_load(T, K, u0, load_next)
                rhs = (
                    one_plus_alpha * F_red
                    - alpha_h * F_red_prev
                    + M_red @ (a0 * q_red + a2 * v_red + a3 * a_red)
                    + one_plus_alpha * (C_red @ (a1 * q_red + a4 * v_red + a5 * a_red))
                )
                if alpha_h != 0.0:
                    rhs += alpha_h * (K_red @ q_red) + alpha_h * (C_red @ v_red)
                q_next = np.asarray(cached_solver.solve(rhs), dtype=float).reshape(-1)
                solve_count += 1
                a_next = a0 * (q_next - q_red) - a2 * v_red - a3 * a_red
                v_next = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_next)
                sphere_a_next = sphere_force / float(sphere.mass)
                sphere_x_next = sphere_x_pred + beta * dt**2 * sphere_a_next
                sphere_v_next = sphere_v_pred + gamma * dt * sphere_a_next
                full_u = reconstruct_full_solution(T, q_next, u0)
                full_v = np.asarray(T @ v_next, dtype=float).reshape(-1)
                # Linear impact fracture and capacity damage consume public
                # records below.  Keep eager materialization on this path;
                # only the nonlinear state-driven loop uses lazy records.
                new_contact_load, new_sphere_force, new_records = assemble_sphere_contact_load_vector(
                    model,
                    sphere,
                    config,
                    sphere_x_next,
                    sphere_v_next,
                    structural_displacement=full_u,
                    structural_velocity=full_v,
                    deleted_element_ids=tuple(deleted_element_ids),
                    contact_scale_by_element=damage_scale_by_element,
                    preferred_element_ids=sticky_contact_ids,
                )
                if new_records:
                    sticky_contact_ids = tuple(int(record.element_id) for record in new_records)
                force_change = float(np.linalg.norm(new_sphere_force - sphere_force))
                penetration_change = abs(
                    (max((record.penetration for record in new_records), default=0.0))
                    - (max((record.penetration for record in records), default=0.0))
                )
                scale = max(float(np.linalg.norm(new_sphere_force)), 1.0)
                fixed_point_residual = new_sphere_force - sphere_force
                if use_aitken and fixed_point_residual_prev is not None:
                    residual_difference = fixed_point_residual - fixed_point_residual_prev
                    denominator = float(residual_difference @ residual_difference)
                    if denominator > 1.0e-30:
                        relaxation = -relaxation * float(fixed_point_residual_prev @ residual_difference) / denominator
                        relaxation = float(min(max(relaxation, 1.0e-4), 1.5))
                    else:
                        relaxation = 1.0
                fixed_point_residual_prev = fixed_point_residual.copy()
                if use_aitken and relaxation != 1.0:
                    contact_load = contact_load + relaxation * (new_contact_load - contact_load)
                    sphere_force = sphere_force + relaxation * (new_sphere_force - sphere_force)
                    records = new_records
                else:
                    contact_load, sphere_force, records = new_contact_load, new_sphere_force, new_records
                if force_change <= config.force_tolerance * scale and penetration_change <= config.penetration_tolerance:
                    converged = True
                    iteration_counts.append(iteration)
                    break
            if not converged:
                status = "contact_iteration_failed"
                stop_reason = "contact_iteration_failed"
                iteration_counts.append(int(config.max_contact_iterations))

            contact_norm = float(np.linalg.norm(sphere_force))
            if contact_norm > 0.0:
                contact_step_count += 1
                active_contact_duration += dt
                separation_elapsed = 0.0
                contact_active_this_step = True
            elif contact_step_count > 0 and separation_stop_time > 0.0:
                separation_elapsed += dt
            if records:
                max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
                peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
            load_next = base_load + contact_load
            impulse += 0.5 * (load_prev + load_next) * dt
            sphere_impulse += 0.5 * (sphere_force_prev + sphere_force) * dt
            load_prev = load_next
            F_red_prev = F_red
            sphere_force_prev = sphere_force.copy()
            q_red, v_red, a_red = q_next, v_next, a_next
            sphere_position, sphere_velocity, sphere_acceleration = sphere_x_next, sphere_v_next, sphere_a_next
            last_records = records
            last_sphere_force = sphere_force.copy()
            last_step_diag = {
                "step_index": int(step_index),
                "substep_index": int(substep),
                "time": float(sub_time),
                "contact_iterations": int(iteration_counts[-1]),
                "contact_relaxation_factor": float(relaxation),
                "force_change_norm": float(force_change),
                "penetration_change": float(penetration_change),
                "active_element_ids": [int(record.element_id) for record in records],
                "max_penetration": float(max((record.penetration for record in records), default=0.0)),
                "status": "converged" if converged else "contact_iteration_failed",
            }
            if fracture_config is not None and records and status != "contact_iteration_failed":
                new_fracture_records, substep_utilization = _detect_impact_fracture_records(
                    model,
                    records,
                    fracture_config,
                    sphere,
                    tuple(deleted_element_ids),
                    step_index=int(step_index),
                    time_value=float(sub_time),
                )
                max_fracture_utilization = max(max_fracture_utilization, substep_utilization)
                if new_fracture_records:
                    fracture_records.extend(new_fracture_records)
                    deleted_element_ids.update(record.element_id for record in new_fracture_records)
                    fracture_deleted_element_ids.update(record.element_id for record in new_fracture_records)
                    filtered_base = filtered_load_case_for_deleted_elements(base_load_case, deleted_element_ids)
                    base_load, base_load_info = assemble_load_vector(model, filtered_base)
                    fracture_scales = {int(element_id): float(scale) for element_id, scale in damage_scale_by_element.items()}
                    for element_id in deleted_element_ids:
                        fracture_scales[int(element_id)] = min(
                            fracture_scales.get(int(element_id), 1.0),
                            float(fracture_config.residual_stiffness_fraction),
                        )
                    K, M, linear_matrix_terms, damage_matrix_plan = (
                        _assemble_damaged_linear_matrices_with_gate(
                            model,
                            fracture_scales,
                            damage_matrix_gate,
                            opportunity_index=int(step_index),
                            cached_terms=linear_matrix_terms,
                            plan=damage_matrix_plan,
                        )
                    )
                    eroded_matrix_rebuild_count += 1
                    K_red = (T.T @ K @ T).tocsr()
                    M_red = (T.T @ M @ T).tocsr()
                    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red).tocsr()
                    cached_solver = None
                    cached_dt = None
                    mass_handle = factorize(
                        M_red,
                        MatrixClass.SYMMETRIC_SEMIDEFINITE,
                        signature=f"sphere_impact.fracture_mass:{len(deleted_element_ids)}",
                    )
                    current_red_load = _reduced_load(T, K, u0, base_load + contact_load)
                    F_red_prev = current_red_load
                    a_red = np.asarray(mass_handle.solve(current_red_load - C_red @ v_red - K_red @ q_red), dtype=float).reshape(-1)
                    scoped_total = _contact_erodible_element_count(model)
                    deleted_fraction = len(deleted_element_ids) / max(scoped_total, 1)
                    last_step_diag["fracture_new_deleted_element_ids"] = [record.element_id for record in new_fracture_records]
                    last_step_diag["fracture_deleted_fraction"] = float(deleted_fraction)
                    last_step_diag["fracture_max_utilization"] = float(max_fracture_utilization)
                    if deleted_fraction > fracture_config.max_deleted_fraction + 1.0e-12:
                        status = "max_deleted_fraction_reached"
                        stop_reason = "impact_fracture_max_deleted_fraction_reached"
                        if not fracture_limit_warning_emitted:
                            warnings.append("IMPACT_FRACTURE002: maximum deleted contact-target fraction reached; transient stopped at last eroded state.")
                            fracture_limit_warning_emitted = True
            if damage_config is not None and records and status not in {"contact_iteration_failed", "max_deleted_fraction_reached"}:
                new_damage_deletions, substep_damage_utilization, damage_diags, damage_changed = _update_impact_damage_states(
                    model,
                    records,
                    damage_config,
                    sphere,
                    impact_damage_states,
                    tuple(deleted_element_ids),
                    step_index=int(step_index),
                    time_value=float(sub_time),
                    dt=float(dt),
                )
                max_damage_utilization = max(max_damage_utilization, substep_damage_utilization)
                if damage_diags:
                    damage_state_update_count += len(damage_diags)
                if damage_diags:
                    last_step_diag["impact_damage"] = list(damage_diags)
                    last_step_diag["impact_damage_max_utilization"] = float(max_damage_utilization)
                if new_damage_deletions:
                    damage_records.extend(new_damage_deletions)
                    deleted_element_ids.update(record.element_id for record in new_damage_deletions)
                    damage_deleted_element_ids.update(record.element_id for record in new_damage_deletions)
                    for deleted_id in deleted_element_ids:
                        state = impact_damage_states.setdefault(int(deleted_id), {})
                        state["damage"] = max(float(state.get("damage", 0.0)), float(damage_config.delete_at))
                        state["scale"] = float(damage_config.residual_stiffness_fraction)
                    last_step_diag["impact_damage_new_deleted_element_ids"] = [record.element_id for record in new_damage_deletions]
                    if not damage_deletion_warning_emitted:
                        warnings.append("IMPACT_DAMAGE002: one or more shell elements were eroded by impact damage.")
                        damage_deletion_warning_emitted = True
                    damage_changed = True
                if damage_changed:
                    damage_scale_by_element = {
                        int(element_id): _impact_damage_scale(float(state.get("damage", 0.0)), damage_config)
                        for element_id, state in impact_damage_states.items()
                    }
                    for element_id in deleted_element_ids:
                        damage_scale_by_element[int(element_id)] = min(
                            damage_scale_by_element.get(int(element_id), 1.0),
                            float(damage_config.residual_stiffness_fraction),
                        )
                    filtered_base = filtered_load_case_for_deleted_elements(base_load_case, deleted_element_ids)
                    base_load, base_load_info = assemble_load_vector(model, filtered_base)
                    K, M, linear_matrix_terms, damage_matrix_plan = (
                        _assemble_damaged_linear_matrices_with_gate(
                            model,
                            damage_scale_by_element,
                            damage_matrix_gate,
                            opportunity_index=int(step_index),
                            cached_terms=linear_matrix_terms,
                            plan=damage_matrix_plan,
                        )
                    )
                    eroded_matrix_rebuild_count += 1
                    K_red = (T.T @ K @ T).tocsr()
                    M_red = (T.T @ M @ T).tocsr()
                    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red).tocsr()
                    cached_solver = None
                    cached_dt = None
                    mass_handle = factorize(
                        M_red,
                        MatrixClass.SYMMETRIC_SEMIDEFINITE,
                        signature=f"sphere_impact.damage_mass:{len(damage_scale_by_element)}:{len(deleted_element_ids)}",
                    )
                    current_red_load = _reduced_load(T, K, u0, base_load + contact_load)
                    F_red_prev = current_red_load
                    a_red = np.asarray(mass_handle.solve(current_red_load - C_red @ v_red - K_red @ q_red), dtype=float).reshape(-1)
                    scoped_total = sum(1 for element in model.mesh.elements.values() if element_fracture_category(element) == "shell")
                    deleted_fraction = len(deleted_element_ids) / max(scoped_total, 1)
                    last_step_diag["impact_damage_deleted_fraction"] = float(deleted_fraction)
                    if deleted_fraction > float(damage_config.max_deleted_fraction) + 1.0e-12:
                        status = "max_deleted_fraction_reached"
                        stop_reason = "impact_damage_max_deleted_fraction_reached"
                        if not damage_limit_warning_emitted:
                            warnings.append(
                                "IMPACT_DAMAGE003: maximum configured shell damage deletion fraction reached; "
                                "transient stopped at last eroded state."
                            )
                            damage_limit_warning_emitted = True
            step_diagnostics.append(last_step_diag)
            if status in {"contact_iteration_failed", "max_deleted_fraction_reached"}:
                break
            if status == "completed" and contact_step_count > 0 and separation_stop_time > 0.0 and separation_elapsed >= separation_stop_time:
                stop_reason = "completed_after_contact_separation"
                stop_after_separation = True
                break

        contact_active_last_step = contact_active_this_step
        save_time = float(last_step_diag.get("time", target_time)) if last_step_diag else target_time
        if step_index % int(transient_config.save_every) == 0 or step_index == len(times) - 1 or stop_after_separation:
            save_state(save_time, q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, last_sphere_force, last_records)
        if status in {"contact_iteration_failed", "max_deleted_fraction_reached"} or stop_after_separation:
            break

    if contact_step_count == 0 and status == "completed":
        status = "no_contact"
        stop_reason = "no_contact"
    impulse_resultant = load_vector_resultant(model, impulse)
    total_energy = np.asarray(energy_kinetic, dtype=float) + np.asarray(energy_strain, dtype=float) + np.asarray(sphere_kinetic, dtype=float)
    nonzero_energy = total_energy[np.abs(total_energy) > 1.0e-30]
    energy_drift = 0.0
    if nonzero_energy.size:
        energy_drift = float((np.max(nonzero_energy) - np.min(nonzero_energy)) / max(abs(nonzero_energy[0]), 1.0e-30))
    max_penetration_ratio = float(max_penetration) / max(float(sphere.radius), 1.0e-12)
    contact_duration = float(active_contact_duration)
    initial_sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    if saved_sphere_v:
        final_sphere_velocity = np.asarray(saved_sphere_v[-1], dtype=float)
    else:
        final_sphere_velocity = sphere_velocity
    sphere_momentum_change = float(sphere.mass) * (final_sphere_velocity - initial_sphere_velocity)
    sphere_momentum_balance_error = float(np.linalg.norm(sphere_impulse - sphere_momentum_change))

    history_width = total_dofs if history_storage_mode == "full" else int(0 if history_dof_indices is None else len(history_dof_indices))
    saved_u_array = np.vstack(saved_u) if saved_u else np.zeros((0, history_width), dtype=float)
    saved_v_array = np.vstack(saved_v) if saved_v else np.zeros((0, history_width), dtype=float)
    saved_a_array = np.vstack(saved_a) if saved_a else np.zeros((0, history_width), dtype=float)
    node_histories: Dict[int, np.ndarray] = {}
    for node_id in output_node_ids:
        values = node_history_values.get(int(node_id), [])
        node_histories[int(node_id)] = np.vstack(values) if values else np.zeros((0, 6), dtype=float)

    recovery_memory = estimate_model_memory(
        model,
        transient_saved_steps=len(saved_times),
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(recovery_memory, transient_config.resource_config, context="solve_transient_sphere_impact.recovery")
    policy_metadata = recovery_metadata(recovery, transient_config.resource_config, recovery_memory)
    assembly_info = {"stiffness": stiffness_info, "mass": mass_info, "load": base_load_info}
    result_case = make_result_case(
        name=f"sphere_impact:{sphere.name}",
        analysis_type="sphere_impact_transient",
        load_cases=() if base_load_case is None else (base_load_case,),
        assembly_info=assembly_info,
        solver_info={"backend": cached_solver_diagnostics, "convergence_info": {"status": status, "stop_reason": stop_reason}},
        recovery={
            "displacement_history": True,
            "velocity_history": True,
            "acceleration_history": True,
            "sphere_history": True,
            "contact_history": bool(config.save_contact_history),
            "history_storage_mode": history_storage_mode,
            **policy_metadata["recovery"],
        },
        settings={
            "dt": transient_config.dt,
            "t_end": transient_config.t_end,
            "beta": beta,
            "gamma": gamma,
            "hht_alpha": alpha_h,
            "rayleigh_alpha": transient_config.rayleigh_alpha,
            "rayleigh_beta": transient_config.rayleigh_beta,
            "sphere": {
                "name": sphere.name,
                "radius": float(sphere.radius),
                "mass": float(sphere.mass),
                "start_point": sphere.initial_position.tolist(),
                "travel_direction": sphere.direction_unit.tolist(),
                "speed": float(sphere.speed),
                "t_start": float(sphere.t_start),
            },
            "contact": config.to_dict(),
            "fracture": None if fracture_config is None else fracture_config.to_dict(),
            "damage": None if damage_config is None else damage_config.to_dict(),
        },
        metadata={
            "resources": policy_metadata.get("resources"),
            "memory_estimate": policy_metadata.get("memory_estimate"),
            "contact_scope": "single rigid sphere to shell midsurface/thickness-offset surface; frictionless normal penalty",
            "solution_control": "implicit_newmark_time_domain",
            "arc_length_applicability": "not_applicable_to_dynamic_impact",
            "target_element_types": ["S3", "S4", "S6", "S8", "S8R"],
            "beam_response": "beams respond only through existing shell-beam coupling; direct beam contact is unsupported",
            "impact_fracture_scope": (
                "disabled"
                if fracture_config is None
                else "contact-observable shell erosion after converged contact substeps"
            ),
            "impact_damage_scope": (
                "disabled"
                if damage_config is None
                else "engineering capacity-based shell softening/erosion after converged contact substeps"
            ),
            "erosion_scope": (
                "disabled"
                if not deleted_element_ids and not damage_scale_by_element
                else "residual stiffness/mass/contact scaling; no node, MPC, beam-shell coupling or topology deletion"
            ),
            "solver_convergence": {"status": status, "stop_reason": stop_reason},
            "verification_gate": "contact",
        },
    ).to_dict()
    impact_fracture_summary = fracture_summary(
        model,
        fracture_config,
        fracture_records,
        fracture_deleted_element_ids,
        max_utilization=max_fracture_utilization,
        warnings=_warnings_with_prefixes(warnings, ("IMPACT_FRACTURE",)) if fracture_config is not None else (),
    )
    impact_damage_summary = _impact_damage_summary(
        model,
        damage_config,
        impact_damage_states,
        damage_deleted_element_ids,
        deletion_records=damage_records,
        warnings=_warnings_with_prefixes(warnings, ("IMPACT_DAMAGE",)) if damage_config is not None else (),
    )
    erosion_summary = {
        "all_eroded_element_ids": sorted(int(element_id) for element_id in deleted_element_ids),
        "fracture_triggered_element_ids": sorted(int(element_id) for element_id in fracture_deleted_element_ids),
        "damage_triggered_element_ids": sorted(int(element_id) for element_id in damage_deleted_element_ids),
        "active_softened_element_ids": sorted(
            int(element_id)
            for element_id, scale in damage_scale_by_element.items()
            if int(element_id) not in deleted_element_ids and float(scale) < 1.0
        ),
        "residual_stiffness_model": "element stiffness/mass scaling; topology, nodes, MPCs and beam-shell couplings retained",
    }
    diagnostics = {
        "method": "newmark_sphere_penalty_contact",
        "solution_control": "implicit_newmark_time_domain",
        "arc_length_applicability": "not_applicable_to_dynamic_impact",
        "status": status,
        "stop_reason": stop_reason,
        "warnings": warnings,
        "num_steps": max(int(len(times) - 1), 0),
        "num_substeps": int(total_substep_count),
        "event_substep_count": int(event_substep_count),
        "num_saved_steps": len(saved_times),
        "num_reduced_dofs": int(K_red.shape[0]),
        "contact_step_count": int(contact_step_count),
        "active_contact_duration": float(contact_duration),
        "separation_stop_time": float(separation_stop_time),
        "post_contact_separation_time": float(separation_elapsed if contact_step_count > 0 else 0.0),
        "max_penetration_ratio": float(max_penetration_ratio),
        "sphere_momentum_balance_error": float(sphere_momentum_balance_error),
        "iteration_counts": iteration_counts,
        "contact_step_diagnostics": step_diagnostics,
        "factorization_count": int(factorization_count),
        "solve_count": int(solve_count),
        "eroded_matrix_rebuild_count": int(eroded_matrix_rebuild_count),
        "damage_state_update_count": int(damage_state_update_count),
        "linear_matrix_terms_cached": linear_matrix_terms is not None,
        "damage_matrix_plan_selection": damage_matrix_gate.diagnostics(),
        "damage_matrix_plan": None if damage_matrix_plan is None else damage_matrix_plan.diagnostics(),
        "contact_work_buffer": {
            "path": "compact_candidates_eager_public_records",
            "lazy_public_materialization": False,
            "fallback_reason": "linear_fracture_damage_record_semantics",
        },
        "initial_mass_factorization": mass_handle.diagnostics(),
        "effective_stiffness_factorization": cached_solver_diagnostics,
        "constraint_info": constraint_info,
        "constraint_postcheck": constraint_residual_summary(
            model,
            np.asarray(T @ q_red + u0, dtype=float).reshape(-1),
        ),
        "stiffness": stiffness_info,
        "mass": mass_info,
        "base_load": base_load_info,
        "contact_config": config.to_dict(),
        "impact_fracture_summary": impact_fracture_summary,
        "impact_damage_summary": impact_damage_summary,
        "erosion_summary": erosion_summary,
        "contact_validation": validation.to_dict(),
        "sphere": result_case["analysis_case"]["settings"]["sphere"],
        "kinetic_energy": energy_kinetic,
        "strain_energy": energy_strain,
        "sphere_kinetic_energy": sphere_kinetic,
        "max_relative_energy_drift": energy_drift,
        "result_case": result_case,
    }
    return SphereImpactResult(
        times=np.asarray(saved_times, dtype=float),
        displacements=saved_u_array,
        velocities=saved_v_array,
        accelerations=saved_a_array,
        node_histories=node_histories,
        sphere_positions=np.vstack(saved_sphere_x) if saved_sphere_x else np.zeros((0, 3), dtype=float),
        sphere_velocities=np.vstack(saved_sphere_v) if saved_sphere_v else np.zeros((0, 3), dtype=float),
        sphere_accelerations=np.vstack(saved_sphere_a) if saved_sphere_a else np.zeros((0, 3), dtype=float),
        contact_force_history=np.vstack(saved_contact_force) if saved_contact_force else np.zeros((0, 3), dtype=float),
        active_contact_history=tuple(saved_contacts),
        load_impulse=impulse,
        force_impulse=impulse_resultant.force,
        moment_impulse=impulse_resultant.moment,
        sphere_impulse=sphere_impulse,
        max_penetration=float(max_penetration),
        max_penetration_ratio=float(max_penetration_ratio),
        peak_contact_force=float(peak_contact_force),
        contact_duration=float(contact_duration),
        sphere_momentum_balance_error=float(sphere_momentum_balance_error),
        peak_displacement=float(peak_displacement),
        peak_displacement_node=peak_displacement_node,
        status=status,
        diagnostics=diagnostics,
        result_case=result_case,
    )


def _verification_contact_panel(stiffener: bool = False) -> "FEModel":
    from .elements import BeamElement, CoupledBeamShellElement
    from .fe_core import FEModel

    model = FEModel("contact_verification_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "soft", thickness=0.05))
    model.add_boundary_condition(
        BoundaryCondition(
            "restrain_shell_nonimpact_modes",
            [1, 2, 3, 4],
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    if stiffener:
        section = {"area": 0.05, "Iy": 1.0e-3, "Iz": 1.0e-3, "J": 1.0e-3}
        model.add_node(10, 0.5, 0.0, 0.05)
        model.add_node(11, 0.5, 1.0, 0.05)
        model.add_element(10, BeamElement(10, [10, 11], "soft", section))
        model.add_element(20, CoupledBeamShellElement(20, beam_node_id=10, shell_node_id=1, material_name="soft"))
        model.add_element(21, CoupledBeamShellElement(21, beam_node_id=11, shell_node_id=4, material_name="soft"))
    return model


def _two_shell_contact_verification_panel() -> "FEModel":
    from .fe_core import FEModel

    model = FEModel("two_shell_contact_verification_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    for node_id, xyz in {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (2.0, 0.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (1.0, 1.0, 0.0),
        6: (2.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, *xyz)
    model.add_element(1, create_shell_element(1, [1, 2, 5, 4], "soft", thickness=0.05))
    model.add_element(2, create_shell_element(2, [2, 3, 6, 5], "soft", thickness=0.05))
    return model


def contact_verification_metrics() -> Dict[str, Dict[str, Any]]:
    """Return compact metrics for the rigid-sphere contact verification ledger."""

    metrics: Dict[str, Dict[str, Any]] = {}
    no_contact = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.01, t_end=0.03),
        RigidSphereImpact("miss", radius=0.1, mass=2.0, start_point=(0.5, 0.5, 2.0), travel_direction=(1.0, 0.0, 0.0), speed=3.0),
        SphereContactConfig(penalty_stiffness=1000.0),
    )
    metrics["CONTACT-001"] = {
        "status": no_contact.status,
        "trajectory_error": float(np.linalg.norm(no_contact.sphere_positions[-1] - np.array([0.59, 0.5, 2.0]))),
        "peak_contact_force": float(no_contact.peak_contact_force),
    }

    sphere = RigidSphereImpact("unit", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    config = SphereContactConfig(penalty_stiffness=1000.0)
    contact_model = _verification_contact_panel()
    vector, sphere_force, records = assemble_sphere_contact_load_vector(
        contact_model,
        sphere,
        config,
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    resultant = load_vector_resultant(contact_model, vector)
    expected_force = 100.0
    metrics["CONTACT-002"] = {
        "normal_force": float(records[0].normal_force if records else 0.0),
        "expected_normal_force": expected_force,
        "relative_error": abs(float(records[0].normal_force if records else 0.0) - expected_force) / expected_force,
    }
    metrics["CONTACT-003"] = {
        "sphere_force": sphere_force.tolist(),
        "structure_force_resultant": resultant.force.tolist(),
        "balance_error": float(np.linalg.norm(sphere_force + resultant.force)),
        "moment_norm": float(resultant.moment_norm),
    }

    hit = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12, output_nodes=[1]),
        RigidSphereImpact("hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        SphereContactConfig(penalty_stiffness=4000.0, contact_damping=0.0, max_contact_iterations=40),
    )
    momentum_change = hit.sphere_velocities[-1] - np.array([0.0, 0.0, -2.0])
    metrics["CONTACT-004"] = {
        "impulse_error": float(np.linalg.norm(hit.sphere_impulse - momentum_change)),
        "impulse_norm": float(np.linalg.norm(hit.sphere_impulse)),
        "status": hit.status,
    }
    metrics["CONTACT-005"] = {
        "status": hit.status,
        "peak_contact_force": float(hit.peak_contact_force),
        "max_penetration": float(hit.max_penetration),
        "peak_displacement": float(hit.peak_displacement),
        "contact_step_count": int(hit.diagnostics["contact_step_count"]),
    }
    stiffened = solve_transient_sphere_impact(
        _verification_contact_panel(stiffener=True),
        TransientConfig(dt=0.0025, t_end=0.11, output_nodes=[10, 11]),
        RigidSphereImpact("stiffened", radius=0.1, mass=1.0, start_point=(0.25, 0.25, 0.22), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        SphereContactConfig(penalty_stiffness=3000.0, max_contact_iterations=40),
    )
    metrics["CONTACT-006"] = {
        "status": stiffened.status,
        "peak_contact_force": float(stiffened.peak_contact_force),
        "beam_node_10_peak_uz": float(np.max(np.abs(stiffened.node_histories[10][:, 2]))),
        "beam_node_11_peak_uz": float(np.max(np.abs(stiffened.node_histories[11][:, 2]))),
    }
    projection_cases = [
        (_verification_contact_panel(), np.array([0.5, 0.5, 0.08]), "face"),
        (_verification_contact_panel(), np.array([1.0, 0.5, 0.08]), "edge"),
        (_verification_contact_panel(), np.array([1.0, 1.0, 0.08]), "corner"),
    ]
    projection_ok = True
    projection_rows = []
    for model, position, expected in projection_cases:
        sphere_case = RigidSphereImpact("projection", radius=0.1, mass=1.0, start_point=position, travel_direction=(0.0, 0.0, -1.0), speed=0.0)
        _vector, _force, case_records = assemble_sphere_contact_load_vector(
            model,
            sphere_case,
            SphereContactConfig(penalty_stiffness=1000.0),
            sphere_position=position,
            sphere_velocity=np.zeros(3),
        )
        actual = case_records[0].contact_classification if case_records else "none"
        projection_ok = projection_ok and actual == expected
        projection_rows.append({"expected": expected, "actual": actual})
    metrics["CONTACT-007"] = {"projection_classification_ok": projection_ok, "rows": projection_rows}

    top_sphere = RigidSphereImpact("surface", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    _mid_vector, _mid_force, mid_records = assemble_sphere_contact_load_vector(
        _verification_contact_panel(),
        top_sphere,
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface="midsurface"),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    _top_vector, _top_force, top_records = assemble_sphere_contact_load_vector(
        _verification_contact_panel(),
        top_sphere,
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface="top"),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    metrics["CONTACT-008"] = {
        "midsurface_records": len(mid_records),
        "top_penetration": float(top_records[0].penetration if top_records else 0.0),
        "expected_top_penetration": 0.025,
    }

    two_shell = _two_shell_contact_verification_panel()
    shared = RigidSphereImpact("shared_edge", radius=0.2, mass=1.0, start_point=(1.0, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    _shared_vector, shared_force, shared_records = assemble_sphere_contact_load_vector(
        two_shell,
        shared,
        SphereContactConfig(penalty_stiffness=1000.0, max_active_contacts=1),
        sphere_position=np.array([1.0, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    metrics["CONTACT-009"] = {
        "active_records": len(shared_records),
        "force_norm": float(np.linalg.norm(shared_force)),
        "expected_force_norm": 100.0,
    }

    auto_sphere = RigidSphereImpact("auto", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    auto = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.0015, t_end=0.11),
        auto_sphere,
        SphereContactConfig(target_penetration_fraction=0.05, auto_penalty_safety_factor=5.0, max_contact_iterations=40),
    )
    metrics["CONTACT-010"] = {
        "status": auto.status,
        "max_penetration_ratio": float(auto.max_penetration_ratio),
        "resolved_penalty": float(auto.diagnostics["contact_config"]["penalty_stiffness"]),
    }

    fast = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.05, t_end=0.05),
        RigidSphereImpact("fast", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.45), travel_direction=(0.0, 0.0, -1.0), speed=20.0),
        SphereContactConfig(penalty_stiffness=4000.0, max_event_substeps=64, max_contact_iterations=40),
    )
    metrics["CONTACT-011"] = {
        "status": fast.status,
        "event_substep_count": int(fast.diagnostics["event_substep_count"]),
        "peak_contact_force": float(fast.peak_contact_force),
    }

    invalid = _verification_contact_panel()
    invalid.materials["soft"].density = 0.0
    invalid_report = validate_contact_configuration(
        invalid,
        RigidSphereImpact("bad_config", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 1.0), travel_direction=(0.0, 0.0, -1.0), speed=10.0),
        SphereContactConfig(penalty_stiffness=1000.0, max_event_substeps=1),
        TransientConfig(dt=0.1, t_end=0.1),
    )
    metrics["CONTACT-012"] = {
        "validation_status": invalid_report.status,
        "issue_codes": [issue.code for issue in invalid_report.issues],
    }
    return metrics
