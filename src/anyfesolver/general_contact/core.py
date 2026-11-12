"""The first opt-in mortar contact profile: qualified planar Q4 and a plane.

The reference-area pressure basis is the element's nonnegative Q1 basis.  At
fixed reference support weights, its penalty potential is the sum of the
positive-part augmented potentials with zero multipliers.  The assembled
force and tangent are derivatives of that same discrete potential.  This is
an intentionally narrow coupled-solver slice, not a general-contact claim.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np
from scipy import sparse

from .runtime import _ACTIVE_CONTACT, active_contact_manager, suspend_contact


_Q4_PLANE_PROFILE = "PLANAR_Q4_RIGID_PLANE_PENALTY_MORTAR_V1"
_S3_PLANE_PROFILE = "PLANAR_Q4_S3_RIGID_PLANE_PENALTY_MORTAR_V0"
_MOVING_PLANE_PROFILE = "PLANAR_Q4_S3_TRANSLATING_PLANE_PENALTY_MORTAR_FORCE_STATIC_DEV_V0"


_Q4_SOURCE_OVERLAP_DUAL_REVISION = 'constant_nonnegative_joint_bernstein_overlap_dual_v2'
_CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION = 'declared_macro_path_physical_substeps_v1'
_INITIAL_CONTACT_SEED_REVISION = 'origin_linearized_kkt_nonnegative_pressure_v1'
_INITIAL_CONTACT_SEED_GLOBALIZATION_REVISION = 'origin_linearized_kkt_seed_merit_globalization_v2'
_INITIAL_CONTACT_SEED_SELECTIONS = ('none', 'linearized_kkt_dev',
                                    'linearized_kkt_merit_globalization_dev')


def _contact_settings_contract(settings):
    """Retain legacy settings contracts when the optional predictor is absent."""
    value = asdict(settings)
    if value.get('q4_source_overlap_dual_dev') is False:
        value.pop('q4_source_overlap_dual_dev')
    if value.get('path_continuation_maximum_halvings') == 0:
        # Explicit OFF omits the whole continuation block so prior
        # serialized settings shapes are preserved byte-for-byte.
        for name in ('path_continuation_maximum_halvings',
                     'path_stagnation_patience', 'path_stagnation_fraction',
                     'path_stagnation_merit_rate'):
            value.pop(name, None)
    if value.get("path_predictor") == "none":
        value.pop("path_predictor")
    if value.get("initial_contact_seed") == "none":
        value.pop("initial_contact_seed")
    if value.get("shell_pair_orientation") == "legacy":
        value.pop("shell_pair_orientation")
    if value.get("shell_skin_kinematics") == "reference_linear":
        value.pop("shell_skin_kinematics")
    if value.get("finite_frame_seed") == "common_reference_frame_dev":
        value.pop("finite_frame_seed")
    return value

if TYPE_CHECKING:
    from anyfesolver.fe_core import FEModel


@dataclass(frozen=True)
class ContactSurface:
    name: str
    element_ids: tuple[int, ...]
    side: str = "midsurface"
    beam_radius: float | None = None
    beam_end_caps: str = "none"

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in self.element_ids):
            raise ValueError("contact element IDs must be positive integers")
        ids = tuple(self.element_ids)
        if not self.name or not ids or len(ids) != len(set(ids)):
            raise ValueError("contact surface requires a name and unique element IDs")
        if self.side not in {"midsurface", "top", "bottom"}:
            raise ValueError("shell side must be midsurface, top, or bottom")
        if self.beam_radius is not None:
            radius = float(self.beam_radius)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("beam contact radius must be finite and positive")
            if self.side != "midsurface":
                raise ValueError("round beam contact does not use a shell side")
            object.__setattr__(self, "beam_radius", radius)
        if self.beam_end_caps not in {"none", "hemisphere"}:
            raise ValueError("beam_end_caps must be none or hemisphere")
        if self.beam_radius is None and self.beam_end_caps != "none":
            raise ValueError("beam end caps require an explicit beam radius")
        object.__setattr__(self, "element_ids", ids)


@dataclass(frozen=True)
class RigidContactBody:
    name: str
    kind: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float] | None = None
    radius: float | None = None
    path_translation: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=float)
        if not self.name or self.kind not in {"plane", "sphere", "cylinder"}:
            raise ValueError("rigid contact body must name a plane, sphere or cylinder")
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("rigid contact point must contain three finite values")
        object.__setattr__(self, "point", tuple(float(x) for x in point))
        if self.path_translation is not None:
            translation = np.asarray(self.path_translation, dtype=float)
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                raise ValueError("rigid contact path translation must contain three finite values")
            object.__setattr__(self, "path_translation", (
                None if not np.any(translation) else
                tuple(float(x) for x in translation)
            ))
        if self.kind == "sphere":
            if self.normal is not None or self.radius is None:
                raise ValueError("rigid sphere requires a radius and no plane normal")
            radius = float(self.radius)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("rigid sphere radius must be finite and positive")
            object.__setattr__(self, "radius", radius)
            return
        if self.kind == "cylinder":
            if self.normal is None or self.radius is None:
                raise ValueError("rigid cylinder requires an axis and radius")
            radius = float(self.radius)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("rigid cylinder radius must be finite and positive")
            axis = np.asarray(self.normal, dtype=float)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise ValueError("rigid cylinder axis must contain three finite values")
            length = float(np.linalg.norm(axis))
            if length <= 0.0:
                raise ValueError("rigid cylinder axis must be nonzero")
            object.__setattr__(self, "radius", radius)
            object.__setattr__(self, "normal", tuple(float(x) for x in axis / length))
            return
        if self.radius is not None or self.normal is None:
            raise ValueError("rigid plane requires a normal and no sphere radius")
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("plane normal must contain three finite values")
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise ValueError("plane normal must be nonzero")
        object.__setattr__(self, "normal", tuple(float(x) for x in normal / length))


@dataclass(frozen=True)
class ContactInteraction:
    name: str
    surface: str
    target: str
    normal_stiffness: float | None
    gap_tolerance: float
    normal_law: str = "penalty"
    line_stiffness: float | None = None
    endpoint_stiffness: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.surface or not self.target:
            raise ValueError("contact interaction needs a name and two participants")
        if self.normal_law not in {"penalty", "augmented", "projected_penalty"}:
            raise ValueError("normal_law must be penalty, augmented or projected_penalty")
        for name in ("normal_stiffness", "line_stiffness", "endpoint_stiffness"):
            if getattr(self, name) is None:
                continue
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        tolerance = float(self.gap_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("gap_tolerance must be finite and positive")
        object.__setattr__(self, "gap_tolerance", tolerance)


@dataclass(frozen=True)
class ContactDefinition:
    surfaces: tuple[ContactSurface, ...]
    rigid_bodies: tuple[RigidContactBody, ...]
    interactions: tuple[ContactInteraction, ...]

    def __post_init__(self) -> None:
        for name in ("surfaces", "rigid_bodies", "interactions"):
            values = tuple(getattr(self, name))
            if (name != "rigid_bodies" and not values) or len({value.name for value in values}) != len(values):
                raise ValueError(f"{name} must have unique names and be nonempty when required")
            object.__setattr__(self, name, values)
        participant_names = [item.name for item in (*self.surfaces, *self.rigid_bodies)]
        if len(participant_names) != len(set(participant_names)):
            raise ValueError("surface and rigid-body names must be distinct")


@dataclass(frozen=True)
class ContactSolverSettings:
    """Explicit resource and restricted-pair controls, versioned with the profile."""

    maximum_candidate_pairs: int = 100_000
    pair_motion: str = "guided"
    sliding_difference_step: float | None = None
    maximum_gap_triangles: int = 10_000
    maximum_patch_normal_dofs: int = 256
    maximum_augmentations: int = 12
    pressure_tolerance: float | None = None
    maximum_shell_tilt: float | None = None
    line_integration_relative_tolerance: float | None = None
    maximum_line_subdivisions: int = 8
    maximum_swept_subdivisions: int = 14
    normal_discretization: str = "primal_mortar"
    pointwise_integration_subdivisions: int = 2
    maximum_pointwise_triangles: int = 10_000
    beam_shell_rim_clearance: float | None = None
    beam_shell_difference_step: float | None = None
    warped_q4_plane: bool = False
    warped_q4_quadrature_order: int | None = None
    warped_q4_control_order: int | None = None
    warped_q4_integration_tolerance: float | None = None
    warped_q4_gap_accuracy: float | None = None
    warped_q4_maximum_gap_cells: int | None = None
    warped_q4_maximum_gap_depth: int | None = None
    beam_pair_gap_accuracy: float | None = None
    maximum_beam_pair_gap_cells: int | None = None
    maximum_beam_pair_gap_depth: int | None = None
    general_s3_difference_step: float | None = None
    general_s3_differentiation: str = "existing"
    q4_s3_shared_edge_tie_step_m: float | None = None
    q4_s3_shared_edge_tie_direction: tuple[float, float, float] | None = None
    q4_s3_shared_edge_tie_relative_tolerance: float | None = None
    maximum_general_s3_energy_evaluations: int = 1500
    q4_target_weighted_gap_tolerance: float | None = None
    maximum_q4_target_integration_triangles: int | None = None
    q4_source_area_tolerance: float | None = None
    q4_source_weighted_gap_tolerance: float | None = None
    maximum_q4_source_integration_triangles: int | None = None
    q4_source_gap_certificate: str = "whole_face"
    q4_source_pressure_basis: str = "fine_nodal_q1"
    q4_source_pressure_knot_columns: tuple[int, ...] | None = None
    augmentation_acceleration: str = "none"
    augmentation_pivot_max_pressure_pa: float | None = None
    augmentation_pivot_difference_step_m: float | None = None
    augmentation_equilibrated_pressure_step_pa: float | None = None
    augmentation_schur_maximum_modes: int | None = None
    augmentation_schur_maximum_pressure_pa: float | None = None
    augmentation_schur_maximum_step_pa: float | None = None
    augmentation_schur_difference_step_m: float | None = None
    augmentation_schur_maximum_attempts: int = 1
    augmentation_schur_maximum_relinearizations: int = 0
    augmentation_schur_nonlinear_backtrack_dev: bool = False
    augmentation_support_resolution: str = "exact"
    augmentation_support_force_budget_n: float | None = None
    enforcement_solver: str = "nested_augmented"
    complementarity_force_scale_n: float | None = None
    monolithic_maximum_pivots: int = 12
    monolithic_linear_solver: str = "direct"
    q4_source_support_gradient_tolerance: float | None = None
    q4_source_support_hessian_tolerance: float | None = None
    q4_source_weighted_gap_gradient_tolerance: float | None = None
    q4_source_weighted_gap_hessian_tolerance: float | None = None
    d1_interior_profile: bool = False
    d1_interior_coverage_tolerance_m2: float | None = None
    d1_interior_boundary_force_tolerance_n: float | None = None
    beam_shell_geometry: str = "translated_planar"
    beam_shell_maximum_swept_intervals: int | None = None
    beam_chain_tangent: str = "finite_difference"
    beam_end_cap_work: str = "owned_point"
    path_predictor: str = "none"
    shell_pair_orientation: str = "legacy"
    shell_skin_kinematics: str = "reference_linear"
    finite_frame_seed: str = "common_reference_frame_dev"
    q4_source_overlap_dual_dev: bool = False
    path_continuation_maximum_halvings: int = 0
    path_stagnation_patience: int = 2
    path_stagnation_fraction: float = .125
    path_stagnation_merit_rate: float = .05
    initial_contact_seed: str = "none"

    def __post_init__(self) -> None:
        if self.finite_frame_seed not in {"common_reference_frame_dev", "per_surface_reference_frames_dev"}:
            raise ValueError("unsupported finite_frame_seed")
        if self.shell_skin_kinematics not in {"reference_linear", "corotational_nodal_director_dev"}:
            raise ValueError("unsupported shell_skin_kinematics")
        if self.shell_pair_orientation not in {"legacy", "physical_sides_dev"}:
            raise ValueError("unsupported shell_pair_orientation")
        if self.path_predictor not in {"none", "control_projection_dev"}:
            raise ValueError("unsupported path_predictor")
        if self.initial_contact_seed not in _INITIAL_CONTACT_SEED_SELECTIONS:
            raise ValueError("unsupported initial_contact_seed")
        if self.initial_contact_seed != "none" and not (
                self.d1_interior_profile
                and self.q4_source_gap_certificate == 'overlap_local_dev'
                and self.general_s3_differentiation == 'jet2_q4_s3_dev'
                and self.q4_source_pressure_basis == 'fine_nodal_q1'
                and self.enforcement_solver == 'monolithic_active_set'
                and self.normal_discretization == 'primal_mortar'
                and self.shell_skin_kinematics == 'reference_linear'
                and self.shell_pair_orientation == 'legacy'):
            raise ValueError('initial contact seed requires the D1 reference-linear fine-Q1 monolithic Q4/S3 overlap-local route')
        if self.beam_end_cap_work not in {"owned_point", "hemisphere_support_dev", "hemisphere_coupled_tool_dev"}:
            raise ValueError("unsupported beam_end_cap_work")
        if self.beam_chain_tangent not in {"finite_difference", "consistent_dev"}:
            raise ValueError("unsupported beam_chain_tangent")
        if self.beam_shell_geometry not in {"translated_planar", "jet2_deformable_s3_dev"}:
            raise ValueError("unsupported beam_shell_geometry")
        if self.beam_shell_maximum_swept_intervals is not None and (type(self.beam_shell_maximum_swept_intervals) is not int or self.beam_shell_maximum_swept_intervals <= 0):
            raise ValueError("beam_shell_maximum_swept_intervals must be a positive integer")
        if type(self.d1_interior_profile) is not bool:
            raise TypeError("d1_interior_profile must be bool")
        if type(self.q4_source_overlap_dual_dev) is not bool:
            raise TypeError('q4_source_overlap_dual_dev must be bool')
        if self.q4_source_overlap_dual_dev and not (
                self.d1_interior_profile
                and self.q4_source_gap_certificate == 'overlap_local_dev'
                and self.general_s3_differentiation == 'jet2_q4_s3_dev'
                and self.q4_source_pressure_basis == 'fine_nodal_q1'
                and self.enforcement_solver == 'monolithic_active_set'
                and self.normal_discretization == 'primal_mortar'
                and self.shell_skin_kinematics == 'reference_linear'
                and self.shell_pair_orientation == 'legacy'):
            raise ValueError('overlap dual requires the D1 reference-linear fine-Q1 monolithic Q4/S3 overlap-local route')
        if (type(self.path_continuation_maximum_halvings) is not int
                or not 0 <= self.path_continuation_maximum_halvings <= 12):
            raise ValueError("path_continuation_maximum_halvings must be an integer in [0, 12]")
        if (type(self.path_stagnation_patience) is not int
                or self.path_stagnation_patience < 1):
            raise ValueError("path_stagnation_patience must be an integer >= 1")
        if (type(self.path_stagnation_fraction) is not float
                or not np.isfinite(self.path_stagnation_fraction)
                or not 0.0 < self.path_stagnation_fraction <= 1.0):
            raise ValueError("path_stagnation_fraction must be finite in (0, 1]")
        if (type(self.path_stagnation_merit_rate) is not float
                or not np.isfinite(self.path_stagnation_merit_rate)
                or not 0.0 < self.path_stagnation_merit_rate < 1.0):
            raise ValueError("path_stagnation_merit_rate must be finite in (0, 1)")
        if self.path_continuation_maximum_halvings == 0 and (
                self.path_stagnation_patience != 2
                or self.path_stagnation_fraction != .125
                or self.path_stagnation_merit_rate != .05):
            raise ValueError("path stagnation controls require path_continuation_maximum_halvings > 0")
        if self.path_continuation_maximum_halvings and self.path_predictor != "none":
            raise ValueError("path continuation is incompatible with the path predictor")
        if self.path_continuation_maximum_halvings and not (
                self.d1_interior_profile
                and self.q4_source_gap_certificate == 'overlap_local_dev'
                and self.general_s3_differentiation == 'jet2_q4_s3_dev'
                and self.q4_source_pressure_basis == 'fine_nodal_q1'
                and self.enforcement_solver == 'monolithic_active_set'
                and self.normal_discretization == 'primal_mortar'
                and self.shell_skin_kinematics == 'reference_linear'
                and self.shell_pair_orientation == 'legacy'):
            raise ValueError('path continuation requires the D1 reference-linear fine-Q1 monolithic Q4/S3 overlap-local route')
        interior_controls = (self.d1_interior_coverage_tolerance_m2,
                             self.d1_interior_boundary_force_tolerance_n)
        if self.d1_interior_profile:
            if any(value is None or not np.isfinite(value) or value <= 0.0
                   for value in interior_controls):
                raise ValueError("D1 interior profile needs positive dimensional coverage and boundary-force tolerances")
        elif any(value is not None for value in interior_controls):
            raise ValueError("D1 interior controls require d1_interior_profile")
        if self.enforcement_solver not in {"nested_augmented", "monolithic_active_set"}:
            raise ValueError("enforcement_solver must be nested_augmented or monolithic_active_set")
        if (self.enforcement_solver == "monolithic_active_set"
                and self.augmentation_acceleration != "none"):
            raise ValueError("monolithic_active_set does not use augmentation acceleration")
        if self.enforcement_solver == "monolithic_active_set":
            if (self.complementarity_force_scale_n is None
                    or not np.isfinite(self.complementarity_force_scale_n)
                    or self.complementarity_force_scale_n <= 0.0):
                raise ValueError("monolithic_active_set requires a positive complementarity force scale")
        elif self.complementarity_force_scale_n is not None:
            raise ValueError("complementarity force scale requires monolithic_active_set")
        if (type(self.monolithic_maximum_pivots) is not int
                or not 1 <= self.monolithic_maximum_pivots <= 64):
            raise ValueError("monolithic_maximum_pivots must be an integer in [1, 64]")
        if (self.enforcement_solver != "monolithic_active_set"
                and self.monolithic_maximum_pivots != 12):
            raise ValueError("monolithic pivot limit requires monolithic_active_set")
        if self.monolithic_linear_solver not in {"direct", "block_response_dev"}:
            raise ValueError("unsupported monolithic_linear_solver")
        if (self.enforcement_solver != "monolithic_active_set"
                and self.monolithic_linear_solver != "direct"):
            raise ValueError("monolithic linear solver requires monolithic_active_set")
        if (self.q4_target_weighted_gap_tolerance is not None
                and (not np.isfinite(self.q4_target_weighted_gap_tolerance)
                     or self.q4_target_weighted_gap_tolerance <= 0.0)):
            raise ValueError("q4_target_weighted_gap_tolerance must be finite and positive")
        if (self.maximum_q4_target_integration_triangles is not None
                and (type(self.maximum_q4_target_integration_triangles) is not int
                     or self.maximum_q4_target_integration_triangles < 1)):
            raise ValueError("maximum_q4_target_integration_triangles must be a positive integer")
        for name in ("q4_source_area_tolerance",
                     "q4_source_weighted_gap_tolerance",
                     "q4_source_support_gradient_tolerance",
                     "q4_source_support_hessian_tolerance",
                     "q4_source_weighted_gap_gradient_tolerance",
                     "q4_source_weighted_gap_hessian_tolerance"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if (self.maximum_q4_source_integration_triangles is not None
                and (type(self.maximum_q4_source_integration_triangles) is not int
                     or self.maximum_q4_source_integration_triangles < 1)):
            raise ValueError("maximum_q4_source_integration_triangles must be a positive integer")
        if self.q4_source_gap_certificate not in {
                "whole_face", "projected_matching", "fixed_projection_graph",
                "overlap_local_dev"}:
            raise ValueError("q4_source_gap_certificate must be whole_face, projected_matching, fixed_projection_graph or overlap_local_dev")
        if self.q4_source_pressure_basis not in {
                "fine_nodal_q1", "macro_q1_pair_dev",
                "selected_q1_knots_dev"}:
            raise ValueError("unsupported q4_source_pressure_basis")
        knots = self.q4_source_pressure_knot_columns
        if self.q4_source_pressure_basis == "selected_q1_knots_dev":
            if (type(knots) is not tuple or len(knots) < 2
                    or any(type(column) is not int for column in knots)):
                raise ValueError("selected Q1 pressure needs immutable integer knot columns")
        elif knots is not None:
            raise ValueError("pressure knot columns require selected_q1_knots_dev")
        if type(self.maximum_candidate_pairs) is not int or self.maximum_candidate_pairs < 1:
            raise ValueError("maximum_candidate_pairs must be a positive integer")
        if self.pair_motion not in {"guided", "restricted_sliding", "general"}:
            raise ValueError("pair_motion must be guided, restricted_sliding or general")
        if self.normal_discretization not in {"primal_mortar", "pointwise_penalty"}:
            raise ValueError("normal_discretization must be primal_mortar or pointwise_penalty")
        if type(self.warped_q4_plane) is not bool:
            raise ValueError("warped_q4_plane must be a boolean")
        for name in ("warped_q4_quadrature_order", "warped_q4_control_order",
                     "warped_q4_maximum_gap_cells"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if (self.warped_q4_maximum_gap_depth is not None
                and (type(self.warped_q4_maximum_gap_depth) is not int
                     or self.warped_q4_maximum_gap_depth < 0)):
            raise ValueError("warped_q4_maximum_gap_depth must be a nonnegative integer")
        for name in ("warped_q4_integration_tolerance", "warped_q4_gap_accuracy"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if (self.beam_pair_gap_accuracy is not None
                and (not np.isfinite(self.beam_pair_gap_accuracy)
                     or self.beam_pair_gap_accuracy <= 0.0)):
            raise ValueError("beam_pair_gap_accuracy must be finite and positive")
        if (self.maximum_beam_pair_gap_cells is not None
                and (type(self.maximum_beam_pair_gap_cells) is not int
                     or self.maximum_beam_pair_gap_cells < 1)):
            raise ValueError("maximum_beam_pair_gap_cells must be a positive integer")
        if (self.maximum_beam_pair_gap_depth is not None
                and (type(self.maximum_beam_pair_gap_depth) is not int
                     or self.maximum_beam_pair_gap_depth < 0)):
            raise ValueError("maximum_beam_pair_gap_depth must be a nonnegative integer")
        if (type(self.pointwise_integration_subdivisions) is not int
                or not 0 <= self.pointwise_integration_subdivisions <= 20):
            raise ValueError("pointwise_integration_subdivisions must be an integer from 0 to 20")
        if type(self.maximum_pointwise_triangles) is not int or self.maximum_pointwise_triangles < 1:
            raise ValueError("maximum_pointwise_triangles must be a positive integer")
        if self.beam_shell_rim_clearance is not None and (
            not np.isfinite(self.beam_shell_rim_clearance)
            or self.beam_shell_rim_clearance <= 0.0
        ):
            raise ValueError("beam_shell_rim_clearance must be finite and positive")
        if self.beam_shell_difference_step is not None and (
            not np.isfinite(self.beam_shell_difference_step)
            or self.beam_shell_difference_step <= 0.0
        ):
            raise ValueError("beam_shell_difference_step must be finite and positive")
        if self.sliding_difference_step is not None and (
            not np.isfinite(self.sliding_difference_step)
            or self.sliding_difference_step <= 0.0
        ):
            raise ValueError("sliding_difference_step must be finite and positive")
        if self.general_s3_difference_step is not None and (
            not np.isfinite(self.general_s3_difference_step)
            or self.general_s3_difference_step <= 0.0
        ):
            raise ValueError("general_s3_difference_step must be finite and positive")
        if self.general_s3_differentiation not in {
            "existing", "jet2_s3_dev", "jet2_q4_s3_dev", "jet2_s3_q4_dev", "jet2_q4_q4_dev",
        }:
            raise ValueError("general_s3_differentiation is not an admitted development mode")
        tie_values = (self.q4_s3_shared_edge_tie_step_m,
                      self.q4_s3_shared_edge_tie_direction,
                      self.q4_s3_shared_edge_tie_relative_tolerance)
        if any(value is not None for value in tie_values):
            if any(value is None for value in tie_values):
                raise ValueError("Q4/S3 shared-edge tie needs step, direction and relative tolerance")
            if self.general_s3_differentiation != "jet2_q4_s3_dev":
                raise ValueError("Q4/S3 shared-edge tie needs Jet2 Q4/S3 development differentiation")
            if (not np.isfinite(self.q4_s3_shared_edge_tie_step_m)
                    or self.q4_s3_shared_edge_tie_step_m <= 0.0
                    or not np.isfinite(self.q4_s3_shared_edge_tie_relative_tolerance)
                    or not 0.0 < self.q4_s3_shared_edge_tie_relative_tolerance <= 1.0e-3):
                raise ValueError("Q4/S3 shared-edge tie step and tolerance must be finite and positive")
            direction = np.asarray(self.q4_s3_shared_edge_tie_direction, dtype=float)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)) or not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1.0e-12):
                raise ValueError("Q4/S3 shared-edge tie direction must be a unit 3-vector")
        if (type(self.maximum_general_s3_energy_evaluations) is not int
                or self.maximum_general_s3_energy_evaluations < 1):
            raise ValueError("maximum_general_s3_energy_evaluations must be positive")
        if type(self.maximum_gap_triangles) is not int or self.maximum_gap_triangles < 1:
            raise ValueError("maximum_gap_triangles must be a positive integer")
        if type(self.maximum_patch_normal_dofs) is not int or self.maximum_patch_normal_dofs < 1:
            raise ValueError("maximum_patch_normal_dofs must be a positive integer")
        if type(self.maximum_augmentations) is not int or self.maximum_augmentations < 1:
            raise ValueError("maximum_augmentations must be a positive integer")
        if (type(self.augmentation_acceleration) is not str
                or self.augmentation_acceleration not in {
                    "none", "active_secant", "active_secant_pivot",
                    "supported_schur_dev",
                }):
            raise ValueError("augmentation_acceleration must be none, active_secant, active_secant_pivot or supported_schur_dev")
        pivot_bounds = (self.augmentation_pivot_max_pressure_pa,
                        self.augmentation_pivot_difference_step_m)
        if self.augmentation_acceleration == "active_secant_pivot":
            if any(value is None or not np.isfinite(value) or value <= 0.0
                   for value in pivot_bounds):
                raise ValueError("active_secant_pivot requires positive pressure and difference-step bounds")
            if self.maximum_augmentations < 15:
                raise ValueError("active_secant_pivot requires at least 15 augmentations")
        elif any(value is not None for value in pivot_bounds):
            raise ValueError("augmentation pivot bounds require active_secant_pivot")
        pressure_step = self.augmentation_equilibrated_pressure_step_pa
        if pressure_step is not None:
            if self.augmentation_acceleration != "active_secant_pivot":
                raise ValueError("equilibrated pressure step requires active_secant_pivot")
            if not np.isfinite(pressure_step) or pressure_step <= 0.0:
                raise ValueError("equilibrated pressure step must be finite and positive")
        schur_bounds = (
            self.augmentation_schur_maximum_modes,
            self.augmentation_schur_maximum_pressure_pa,
            self.augmentation_schur_maximum_step_pa,
            self.augmentation_schur_difference_step_m,
        )
        if self.augmentation_acceleration == "supported_schur_dev":
            if (type(schur_bounds[0]) is not int or schur_bounds[0] < 1
                    or any(value is None or not np.isfinite(value) or value <= 0.0
                           for value in schur_bounds[1:])):
                raise ValueError("supported_schur_dev needs explicit positive mode, pressure, step and geometry-difference bounds")
            if self.maximum_augmentations < 4:
                raise ValueError("supported_schur_dev needs at least four augmentations")
            if (type(self.augmentation_schur_maximum_attempts) is not int
                    or self.augmentation_schur_maximum_attempts < 1
                    or self.augmentation_schur_maximum_attempts
                    > self.maximum_augmentations - 3):
                raise ValueError("Schur attempt cap must be positive and fit the augmentation budget")
            if (type(self.augmentation_schur_maximum_relinearizations) is not int
                    or self.augmentation_schur_maximum_relinearizations not in {0, 1}):
                raise ValueError("Schur relinearization cap must be zero or one")
            if type(self.augmentation_schur_nonlinear_backtrack_dev) is not bool:
                raise ValueError("Schur nonlinear backtrack opt-in must be boolean")
            if (self.augmentation_schur_nonlinear_backtrack_dev
                    and self.augmentation_schur_maximum_attempts < 2):
                raise ValueError("Schur nonlinear backtrack needs at least two attempts")
        elif any(value is not None for value in schur_bounds):
            raise ValueError("Schur bounds require supported_schur_dev")
        elif self.augmentation_schur_maximum_attempts != 1:
            raise ValueError("Schur attempts require supported_schur_dev")
        elif self.augmentation_schur_maximum_relinearizations != 0:
            raise ValueError("Schur relinearization requires supported_schur_dev")
        elif self.augmentation_schur_nonlinear_backtrack_dev:
            raise ValueError("Schur nonlinear backtrack requires supported_schur_dev")
        if self.augmentation_support_resolution not in {"exact", "roundoff"}:
            raise ValueError("augmentation_support_resolution must be exact or roundoff")
        if self.augmentation_support_resolution == "roundoff":
            budget = self.augmentation_support_force_budget_n
            if budget is None or not np.isfinite(budget) or budget <= 0.0:
                raise ValueError("roundoff support requires a positive force budget")
        elif self.augmentation_support_force_budget_n is not None:
            raise ValueError("support force budget requires roundoff support")
        if self.pressure_tolerance is not None and (
            not np.isfinite(self.pressure_tolerance) or self.pressure_tolerance <= 0.0
        ):
            raise ValueError("pressure_tolerance must be finite and positive")
        if self.maximum_shell_tilt is not None and (
            not np.isfinite(self.maximum_shell_tilt)
            or self.maximum_shell_tilt <= 0.0
        ):
            raise ValueError("maximum_shell_tilt must be finite and positive")
        if self.line_integration_relative_tolerance is not None and (
            not np.isfinite(self.line_integration_relative_tolerance)
            or self.line_integration_relative_tolerance <= 0.0
        ):
            raise ValueError("line_integration_relative_tolerance must be finite and positive")
        for name in ("maximum_line_subdivisions", "maximum_swept_subdivisions"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.pair_motion == "restricted_sliding" and self.sliding_difference_step is None:
            raise ValueError("restricted sliding requires an explicit sliding_difference_step")
        if self.pair_motion == "guided" and self.sliding_difference_step is not None:
            raise ValueError("sliding_difference_step requires restricted_sliding pair_motion")


@dataclass(frozen=True)
class ContactValidationReport:
    status: str
    issues: tuple[str, ...]
    profile: str = "PLANAR_Q4_RIGID_PLANE_PENALTY_MORTAR_V1"

    @property
    def valid(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class ContactResults:
    profile: str
    minimum_gap: float
    penalty_energy: float
    shell_force: tuple[float, float, float]
    rigid_force: tuple[float, float, float]
    shell_moment: tuple[float, float, float]
    rigid_moment: tuple[float, float, float]
    active_pressure_dofs: int
    gap_tolerance_met: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "minimum_gap": self.minimum_gap,
            "penalty_energy": self.penalty_energy,
            "shell_force": list(self.shell_force),
            "rigid_force": list(self.rigid_force),
            "shell_moment": list(self.shell_moment),
            "rigid_moment": list(self.rigid_moment),
            "active_pressure_dofs": self.active_pressure_dofs,
            "gap_tolerance_met": self.gap_tolerance_met,
        }


def _exact_single_q4_pair(model: FEModel,
                          definition: ContactDefinition) -> bool:
    """Route only exact qualified one-Q4-per-side definitions to the patch owner."""
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement

    return (len(definition.surfaces) == 2
            and all(len(surface.element_ids) == 1
                    and type(model.mesh.elements.get(surface.element_ids[0]))
                    is QualifiedE4PLShellElement
                    for surface in definition.surfaces))


def _exact_q4_s3_patch(model: FEModel,
                       definition: ContactDefinition) -> bool:
    """Admit only one Q4 pressure face against two qualified S3 target faces."""
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    if len(definition.surfaces) != 2 or len(definition.interactions) != 1:
        return False
    names = {surface.name: surface for surface in definition.surfaces}
    interaction = definition.interactions[0]
    source = names.get(interaction.surface)
    target = names.get(interaction.target)
    if source is None or target is None or source is target:
        return False
    return (len(source.element_ids) == 1
            and len(target.element_ids) == 2
            and type(model.mesh.elements.get(source.element_ids[0]))
            is QualifiedE4PLShellElement
            and all(type(model.mesh.elements.get(element_id))
                    is NativeParityE4PLS3V2DShellElement
                    for element_id in target.element_ids))


def _refined_q4_s3_patch(model: FEModel,
                         definition: ContactDefinition) -> bool:
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    if len(definition.surfaces) != 2 or len(definition.interactions) != 1:
        return False
    names = {surface.name: surface for surface in definition.surfaces}
    interaction = definition.interactions[0]
    source = names.get(interaction.surface)
    target = names.get(interaction.target)
    return bool(source is not None and target is not None and source is not target
                and len(source.element_ids) >= 2 and len(target.element_ids) >= 1
                and all(type(model.mesh.elements.get(element_id))
                        is QualifiedE4PLShellElement for element_id in source.element_ids)
                and all(type(model.mesh.elements.get(element_id))
                        is NativeParityE4PLS3V2DShellElement for element_id in target.element_ids))


def _exact_monolithic_s3_pair(model, definition, settings):
    """Named full-face hard S3 row; historical single-pair routing stays intact."""
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    return (settings is not None and settings.pair_motion == 'general'
            and settings.general_s3_differentiation == 'jet2_s3_dev'
            and settings.enforcement_solver == 'monolithic_active_set'
            and settings.shell_pair_orientation == 'physical_sides_dev'
            and not definition.rigid_bodies and len(definition.surfaces) == 2
            and len(definition.interactions) == 1
            and definition.interactions[0].normal_law == 'augmented'
            and all(len(s.element_ids) == 1 for s in definition.surfaces)
            and all(type(model.mesh.elements.get(e)) is NativeParityE4PLS3V2DShellElement
                    for s in definition.surfaces for e in s.element_ids))


def validate_general_contact(model: FEModel, definition: ContactDefinition,
                             settings: ContactSolverSettings | None = None,
                             *, analysis_control: str = "force") -> ContactValidationReport:
    """Report every unsupported feature before contact changes the solve."""

    issues: list[str] = []
    if type(definition) is not ContactDefinition:
        return ContactValidationReport("invalid", ("ContactDefinition required",))
    if settings is not None and type(settings) is not ContactSolverSettings:
        return ContactValidationReport("invalid", ("ContactSolverSettings required",))
    if settings is not None and settings.finite_frame_seed != "common_reference_frame_dev":
        if (analysis_control != "contact_path"
                or settings.shell_skin_kinematics != "corotational_nodal_director_dev"):
            return ContactValidationReport("invalid", (
                "per_surface_reference_frames_dev requires the finite Q4 contact_path_program",))
        from .finite_rotation_predictor import finite_surface_partitions
        try:
            finite_surface_partitions(model, definition)
        except ValueError as exc:
            return ContactValidationReport("invalid", (str(exc),))
    if settings is not None and settings.shell_skin_kinematics != "reference_linear":
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
        from .finite_mixed_state import admitted_definition
        finite_mixed = analysis_control=='contact_path' and admitted_definition(model,definition,settings)
        from .finite_v2d_state import admitted_definition as admitted_v2d
        finite_mixed = finite_mixed or (analysis_control=='contact_path' and admitted_v2d(model,definition,settings))
        from .finite_rigid_patch import admitted_definition as admitted_finite_rigid
        finite_mixed = finite_mixed or (analysis_control in {'force','displacement'}
                                       and admitted_finite_rigid(model,definition,settings))
        if not finite_mixed and (analysis_control != "contact_path" or definition.rigid_bodies
                or settings.general_s3_differentiation != "jet2_q4_q4_dev"
                or settings.enforcement_solver != "monolithic_active_set"
                or settings.q4_source_gap_certificate != "whole_face"
                or settings.path_predictor != "none" or settings.d1_interior_profile
                or len(definition.surfaces) != 2 or len(definition.interactions) != 1
                or definition.interactions[0].normal_law != "augmented"
                or set(model.mesh.elements) != {eid for surface in definition.surfaces for eid in surface.element_ids}
                or not all(surface.side in {"top", "bottom"} and surface.element_ids
                           and all(type(model.mesh.elements.get(eid)) is QualifiedE4PLShellElement
                                   for eid in surface.element_ids)
                           for surface in definition.surfaces)):
            return ContactValidationReport("invalid", (
                "corotational_nodal_director_dev requires the explicit frictionless Q4/Q4 monolithic whole-face contact path",))
    if settings is not None and settings.general_s3_differentiation == "jet2_q4_q4_dev":
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
        if (analysis_control not in {"force", "displacement", "contact_path"}
                or definition.rigid_bodies or len(definition.surfaces) != 2
                or len(definition.interactions) != 1
                or not (definition.interactions[0].normal_law == "projected_penalty"
                        or (definition.interactions[0].normal_law == "augmented"
                            and settings.enforcement_solver == "monolithic_active_set"))
                or not all(surface.element_ids and surface.beam_radius is None
                           and all(type(model.mesh.elements.get(eid)) is QualifiedE4PLShellElement
                                   for eid in surface.element_ids) for surface in definition.surfaces)):
            return ContactValidationReport("invalid", (
                "jet2_q4_q4_dev requires a static deformable Q4/Q4 projected-penalty or monolithic augmented pair",))
    if settings is not None and settings.shell_pair_orientation != "legacy":
        if analysis_control != "contact_path":
            return ContactValidationReport("invalid", ("shell_pair_orientation requires contact_path_program",))
        from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
        physical_s3 = (settings.general_s3_differentiation == "jet2_s3_dev"
                       and definition.interactions
                       and ((settings.enforcement_solver == "nested_augmented"
                             and definition.interactions[0].normal_law == "projected_penalty")
                            or (settings.enforcement_solver == "monolithic_active_set"
                                and definition.interactions[0].normal_law == "augmented"
                                and all(s.element_ids for s in definition.surfaces)))
                       and all(all(type(model.mesh.elements.get(eid)) is NativeParityE4PLS3V2DShellElement
                                   for eid in surface.element_ids) for surface in definition.surfaces))
        physical_q4 = (settings.general_s3_differentiation == "jet2_q4_q4_dev"
                       and settings.enforcement_solver == "monolithic_active_set"
                       and definition.interactions
                       and definition.interactions[0].normal_law == "augmented"
                       and all(all(type(model.mesh.elements.get(eid)) is QualifiedE4PLShellElement
                                   for eid in surface.element_ids) for surface in definition.surfaces))
        from .finite_mixed_state import admitted_definition
        physical_mixed = admitted_definition(model,definition,settings)
        if (not (physical_s3 or physical_q4 or physical_mixed)
                or settings.q4_source_gap_certificate != "whole_face"
                or definition.rigid_bodies or len(definition.surfaces) != 2
                or len(definition.interactions) != 1
                or not all(surface.side in {"top", "bottom"} for surface in definition.surfaces)):
            return ContactValidationReport("invalid", (
                "physical_sides_dev requires the S3 projected-penalty or Q4/Q4 monolithic fixed-map contact path",))
    if (settings is not None and settings.path_predictor != "none"
            and analysis_control != "contact_path"):
        return ContactValidationReport("invalid", ("path_predictor requires contact_path_program",))
    if settings is not None and settings.path_predictor != "none":
        from dataclasses import replace
        from .s3_patch_static import (REFINED_Q4_S3_PROJECTED_PENALTY_PROFILE,
                                      S3_PROJECTED_PENALTY_PROFILE, REFINED_Q4_Q4_PROJECTED_PENALTY_PROFILE, S3_Q4_PROJECTED_PENALTY_PROFILE, S3_MESH_MONOLITHIC_PROFILE)
        baseline = validate_general_contact(model, definition,
            replace(settings, path_predictor="none"),
            analysis_control="contact_path" if settings.shell_pair_orientation != "legacy" else "force")
        geometry = ((baseline.profile == REFINED_Q4_S3_PROJECTED_PENALTY_PROFILE
                     and settings.general_s3_differentiation == "jet2_q4_s3_dev"
                     and settings.q4_source_gap_certificate == "overlap_local_dev")
                    or (baseline.profile == S3_PROJECTED_PENALTY_PROFILE
                        and settings.general_s3_differentiation == "jet2_s3_dev"
                        and settings.q4_source_gap_certificate == "whole_face")
                    or (baseline.profile == S3_Q4_PROJECTED_PENALTY_PROFILE
                        and settings.general_s3_differentiation == "jet2_s3_q4_dev"
                        and settings.q4_source_gap_certificate == "whole_face")
                    or (baseline.profile == REFINED_Q4_Q4_PROJECTED_PENALTY_PROFILE
                        and settings.general_s3_differentiation == "jet2_q4_q4_dev"
                        and settings.q4_source_gap_certificate == "whole_face"))
        # This report admits only the normal prerequisite. The public path
        # additionally requires the exact mesh material-friction owner.
        mesh_lift = (baseline.profile == S3_MESH_MONOLITHIC_PROFILE
                     and settings.enforcement_solver == "monolithic_active_set"
                     and settings.shell_pair_orientation == "physical_sides_dev"
                     and settings.general_s3_differentiation == "jet2_s3_dev"
                     and settings.q4_source_gap_certificate == "whole_face")
        if (not baseline.valid or not (mesh_lift or (geometry
                and settings.enforcement_solver == "nested_augmented"))
                or definition.rigid_bodies):
            return ContactValidationReport("invalid", (
                "control_projection_dev requires the refined Q4/S3 or S3/S3 projected-penalty fixed-map contact path",))
        return baseline
    if (settings is not None and settings.path_continuation_maximum_halvings
            and analysis_control != "contact_path"):
        # Enabled continuation declares macro physical substeps;
        # only the contact_path driver implements them. Refuse
        # before the internal contact_path->force conversion so
        # no other driver can silently ignore the requested
        # controls after validation passes.
        return ContactValidationReport("invalid", (
            "path_continuation requires contact_path_program",))
    if (settings is not None and settings.initial_contact_seed != "none"
            and analysis_control != "contact_path"):
        # The seed is an initial-only cold-start initializer of the
        # explicit contact path; no other driver may silently ignore it.
        return ContactValidationReport("invalid", (
            "initial_contact_seed requires contact_path_program",))
    if analysis_control == "contact_path":
        analysis_control = "force"
    if analysis_control == 'arc_length':
        from .arc_contact import validate_arc_contact
        return validate_arc_contact(model, definition, settings)
    from .triangulated_rigid import TriangulatedRigidContactBody
    if any(type(body) is TriangulatedRigidContactBody for body in definition.rigid_bodies):
        if settings is not None and settings.shell_skin_kinematics == 'corotational_nodal_director_dev':
            from .finite_rigid_patch import validate_finite_rigid_patch
            return validate_finite_rigid_patch(model, definition, settings, analysis_control=analysis_control)
        from .triangulated_rigid_static import validate_triangulated_rigid
        return validate_triangulated_rigid(model, definition, settings, analysis_control=analysis_control)
    if settings is not None and settings.beam_end_cap_work == 'hemisphere_coupled_tool_dev':
        from .beam_hemisphere_coupled_tool import validate_coupled_hemisphere_tool
        return validate_coupled_hemisphere_tool(model, definition, settings,
                                                analysis_control=analysis_control)
    if settings is not None and settings.beam_end_cap_work == "hemisphere_support_dev":
        if (len(definition.interactions) > 1
                and any(s.beam_radius is None for s in definition.surfaces)):
            from .multiple_beam_s3_caps import validate_multiple_beam_s3_caps
            return validate_multiple_beam_s3_caps(model, definition, settings,
                                                 analysis_control=analysis_control)
        if (not definition.rigid_bodies and len(definition.surfaces) == 2
                and sum(s.beam_radius is not None for s in definition.surfaces) == 1):
            from .deformable_beam_s3_caps import validate_deformable_beam_s3_caps
            return validate_deformable_beam_s3_caps(model, definition, settings,
                                                   analysis_control=analysis_control)
        if len(definition.interactions) > 1:
            from .multiple_beam_rigid import validate_multiple_beam_rigid
            return validate_multiple_beam_rigid(model, definition, settings,
                                                analysis_control=analysis_control)
        if len(definition.rigid_bodies) == 1 and definition.rigid_bodies[0].kind == 'cylinder':
            if definition.rigid_bodies[0].path_translation is not None:
                from .beam_hemisphere_cylinder_motion import validate_translating_hemisphere_cylinder
                return validate_translating_hemisphere_cylinder(model, definition, settings,
                                                               analysis_control=analysis_control)
            from .beam_hemisphere_cylinder import validate_hemisphere_cylinder_contact
            return validate_hemisphere_cylinder_contact(model, definition, settings,
                                                       analysis_control=analysis_control)
        if len(definition.rigid_bodies) == 1 and definition.rigid_bodies[0].kind == 'sphere':
            from .beam_hemisphere_sphere import validate_hemisphere_sphere_contact
            return validate_hemisphere_sphere_contact(model, definition, settings,
                                                     analysis_control=analysis_control)
        from .beam_hemisphere_plane import validate_hemisphere_plane_contact
        return validate_hemisphere_plane_contact(model, definition, settings,
                                                 analysis_control=analysis_control)
    if settings is not None and settings.beam_chain_tangent == "consistent_dev":
        if (definition.rigid_bodies or len(definition.surfaces) != 2
                or len(definition.interactions) != 1
                or not all(surface.beam_radius is not None for surface in definition.surfaces)
                or not any(len(surface.element_ids) > 1 for surface in definition.surfaces)):
            return ContactValidationReport("invalid", ("consistent beam-chain tangent requires one explicit multi-cell beam pair",))
    if settings is not None and settings.beam_shell_geometry == "jet2_deformable_s3_dev":
        if (definition.rigid_bodies or len(definition.surfaces) != 2
                or sum(surface.beam_radius is not None for surface in definition.surfaces) != 1
                or len(definition.interactions) != 1):
            return ContactValidationReport("invalid", ("deformable beam-S3 selection requires one explicit beam/shell pair",))
    if len(definition.interactions) > 1:
        if definition.rigid_bodies and all(surface.beam_radius is not None
                                          for surface in definition.surfaces):
            from .multiple_beam_rigid import validate_multiple_beam_rigid
            return validate_multiple_beam_rigid(
                model, definition, settings, analysis_control=analysis_control)
        if definition.rigid_bodies and all(body.kind == 'plane' for body in definition.rigid_bodies):
            from .multiple_shell_planes import validate_multiple_shell_planes
            return validate_multiple_shell_planes(
                model, definition, settings, analysis_control=analysis_control)
        from .multiple_projected_pairs import validate_multiple_projected_pairs
        return validate_multiple_projected_pairs(
            model, definition, settings, analysis_control=analysis_control)
    if any(item.normal_law == "projected_penalty" for item in definition.interactions):
        from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

        s3_geometry = (not definition.rigid_bodies and len(definition.surfaces) == 2
                       and all(surface.element_ids for surface in definition.surfaces)
                       and sum(len(surface.element_ids) for surface in definition.surfaces) >= 3
                       and all(type(model.mesh.elements.get(element_id))
                               is NativeParityE4PLS3V2DShellElement
                               for surface in definition.surfaces
                               for element_id in surface.element_ids))
        s3_route = (s3_geometry and settings is not None
                    and settings.general_s3_differentiation == "jet2_s3_dev"
                    and settings.q4_source_gap_certificate == "whole_face"
                    and settings.q4_s3_shared_edge_tie_step_m is None)
        q4_route = (settings is not None and _refined_q4_s3_patch(model, definition)
                    and settings.general_s3_differentiation == "jet2_q4_s3_dev"
                    and settings.q4_source_gap_certificate == "overlap_local_dev")
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
        q4_q4_route = (settings is not None
                       and settings.general_s3_differentiation == "jet2_q4_q4_dev"
                       and settings.q4_source_gap_certificate == "whole_face"
                       and not definition.rigid_bodies and len(definition.surfaces) == 2
                       and all(type(model.mesh.elements.get(eid)) is QualifiedE4PLShellElement
                               for surface in definition.surfaces for eid in surface.element_ids))
        s3_q4_route = (settings is not None
                       and settings.general_s3_differentiation == "jet2_s3_q4_dev"
                       and settings.q4_source_gap_certificate == "whole_face")
        if (settings is None or settings.pair_motion != "general"
                or not (s3_route or q4_route or s3_q4_route or q4_q4_route)
                or len(definition.interactions) != 1
                or settings.q4_source_pressure_basis != "fine_nodal_q1"
                or settings.augmentation_support_resolution != "exact"
                or settings.augmentation_acceleration != "none"
                or settings.enforcement_solver != "nested_augmented"
                or settings.d1_interior_profile):
            return ContactValidationReport("invalid", (
                ("projected_penalty S3/S3 requires the separate S3 Jet2 whole-face profile with exact nodal support and no augmentation, tie or D1 admission"
                 if s3_geometry else
                 "projected_penalty requires the separate refined Q4/S3 fine-Q1 Jet2 overlap-local profile without augmentation or D1 admission"),
            ))
    if settings is not None and settings.d1_interior_profile:
        if (not _refined_q4_s3_patch(model, definition)
                or definition.interactions[0].normal_law != "augmented"
                or settings.pair_motion != "general"
                or settings.normal_discretization != "primal_mortar"
                or settings.q4_source_pressure_basis != "fine_nodal_q1"
                or settings.general_s3_differentiation != "jet2_q4_s3_dev"):
            return ContactValidationReport("invalid", (
                "D1 interior profile requires a refined augmented Q4/S3 fine-Q1 Jet2 general patch",
            ))
    if (settings is not None
            and settings.q4_source_pressure_basis == "macro_q1_pair_dev"
            and (not _refined_q4_s3_patch(model, definition)
                 or settings.general_s3_differentiation != "jet2_q4_s3_dev"
                 or settings.normal_discretization != "primal_mortar"
                 or settings.augmentation_support_resolution != "exact")):
        return ContactValidationReport("invalid", (
            "macro Q1 pressure basis needs refined Q4-source/S3-target Jet2 augmented mortar with exact support",
        ))
    if settings is not None and settings.general_s3_differentiation == "jet2_s3_dev":
        from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

        selected = (settings.pair_motion == "general"
                    and not definition.rigid_bodies
                    and len(definition.surfaces) == 2
                    and len(definition.interactions) == 1
                    and definition.interactions[0].normal_law in {"augmented", "projected_penalty"}
                    and (sum(len(surface.element_ids) for surface in definition.surfaces) >= 3
                         or (settings.enforcement_solver == "monolithic_active_set"
                             and definition.interactions[0].normal_law == "augmented"
                             and settings.shell_pair_orientation == "physical_sides_dev"
                             and all(len(surface.element_ids) == 1 for surface in definition.surfaces)))
                    and all(type(model.mesh.elements.get(element_id))
                            is NativeParityE4PLS3V2DShellElement
                            for surface in definition.surfaces
                            for element_id in surface.element_ids))
        if not selected:
            return ContactValidationReport("invalid", (
                "Jet2 S3 development derivative requires an augmented general S3/S3 patch of at least three faces",
            ))
    if settings is not None and settings.general_s3_differentiation == "jet2_q4_s3_dev":
        derivative_tolerances = (
            settings.q4_source_support_gradient_tolerance,
            settings.q4_source_support_hessian_tolerance,
            settings.q4_source_weighted_gap_gradient_tolerance,
            settings.q4_source_weighted_gap_hessian_tolerance,
        )
        if (settings.pair_motion != "general" or definition.rigid_bodies
                or len(definition.interactions) != 1
                or definition.interactions[0].normal_law not in {"augmented", "projected_penalty"}
                or not _refined_q4_s3_patch(model, definition)
                or any(value is None for value in derivative_tolerances)):
            return ContactValidationReport("invalid", (
                "Jet2 Q4/S3 development derivative requires a refined augmented general patch and four derivative tolerances",
            ))
    if settings is not None and settings.augmentation_acceleration != "none":
        if (definition.rigid_bodies or len(definition.interactions) != 1
                or definition.interactions[0].normal_law != "augmented"
                or settings.pair_motion not in {"general", "restricted_sliding"}):
            return ContactValidationReport("invalid", (
                "active secant acceleration requires one augmented deformable shell patch pair",
            ))
        if (settings.augmentation_acceleration == "active_secant_pivot"
                and not _refined_q4_s3_patch(model, definition)):
            return ContactValidationReport("invalid", (
                "active-set pivot requires the refined Q4/S3 augmented patch profile",
            ))
        if settings.augmentation_acceleration == "supported_schur_dev" and (
                not _refined_q4_s3_patch(model, definition)
                or settings.augmentation_support_resolution != "exact"):
            return ContactValidationReport("invalid", (
                "supported Schur development route requires a refined Q4/S3 augmented patch with exact pressure support",
            ))
    if settings is not None and settings.augmentation_support_resolution == "roundoff":
        if (definition.rigid_bodies or len(definition.interactions) != 1
                or definition.interactions[0].normal_law != "augmented"
                or settings.pair_motion != "general"
                or not _refined_q4_s3_patch(model, definition)):
            return ContactValidationReport("invalid", (
                "roundoff pressure support requires the refined Q4/S3 augmented patch profile",
            ))
    if settings is not None and settings.q4_source_gap_certificate in {
            "projected_matching", "overlap_local_dev"}:
        if not (settings.pair_motion == "general"
                and not definition.rigid_bodies
                and len(definition.interactions) == 1
                and definition.interactions[0].normal_law in {"augmented", "projected_penalty"}
                and (_exact_single_q4_pair(model, definition)
                     or _exact_q4_s3_patch(model, definition)
                     or _refined_q4_s3_patch(model, definition))):
            return ContactValidationReport("invalid", (
                "projected-matching Q4 gap certificate requires an explicit augmented Q4/Q4 or Q4/S3 general pair",
            ))
    if settings is not None and settings.q4_source_gap_certificate == "overlap_local_dev":
        if not _refined_q4_s3_patch(model, definition):
            return ContactValidationReport("invalid", (
                "overlap-local development gap requires a refined Q4/S3 patch",
            ))
    if settings is not None and settings.q4_source_gap_certificate == "fixed_projection_graph":
        if not (settings.pair_motion == "general"
                and not definition.rigid_bodies
                and len(definition.interactions) == 1
                and definition.interactions[0].normal_law == "augmented"
                and settings.augmentation_support_resolution == "exact"
                and settings.augmentation_acceleration == "none"
                and _refined_q4_s3_patch(model, definition)):
            return ContactValidationReport("invalid", (
                "fixed-projection graph gap requires an explicit refined augmented Q4/S3 general patch with exact support and no acceleration",
            ))
    if settings is not None and settings.normal_discretization == "pointwise_penalty":
        general_s3 = (settings.pair_motion == "general"
                      and not definition.rigid_bodies
                      and len(definition.surfaces) == 2
                      and all(item.beam_radius is None for item in definition.surfaces)
                      and all(len(item.element_ids) == 1 for item in definition.surfaces)
                      and all(item.normal_law == "penalty" for item in definition.interactions))
        if (not general_s3 and
                (settings.pair_motion != "restricted_sliding" or definition.rigid_bodies
                or len(definition.surfaces) != 2
                or not any(len(item.element_ids) > 1 for item in definition.surfaces)
                or any(item.normal_law != "penalty" for item in definition.interactions))):
            return ContactValidationReport("invalid", (
                "pointwise penalty needs a multi-face restricted shell pair or the single-face general S3 verification profile, with penalty normal law",
            ))
    if analysis_control not in {"force", "displacement"}:
        return ContactValidationReport("invalid", ("analysis_control must be force or displacement",))
    if settings is not None and settings.warped_q4_plane:
        from .warped_q4_static import validate_warped_q4_plane

        return validate_warped_q4_plane(model, definition, settings,
                                        analysis_control=analysis_control)
    if any(surface.beam_radius is not None for surface in definition.surfaces):
        if definition.rigid_bodies:
            if len(definition.rigid_bodies) == 1 and definition.rigid_bodies[0].kind == "capped_cylinder":
                from .capped_cylinder import validate_capped_cylinder_beam
                return validate_capped_cylinder_beam(model, definition, settings)
            if len(definition.rigid_bodies) == 1 and definition.rigid_bodies[0].kind in {
                "sphere", "cylinder",
            }:
                if any(surface.beam_end_caps=='hemisphere' for surface in definition.surfaces):
                    from .beam_radial_ends import validate_radial_beam_ends
                    return validate_radial_beam_ends(model,definition,settings)
                from .beam_radial import validate_beam_radial_contact

                return validate_beam_radial_contact(model, definition, settings)
            from .beam_plane import validate_beam_plane_contact

            return validate_beam_plane_contact(model, definition, settings)
        if any(surface.beam_radius is None for surface in definition.surfaces):
            from .beam_shell import validate_beam_shell_contact

            return validate_beam_shell_contact(model, definition, settings)
        if any(surface.beam_end_caps == 'hemisphere' for surface in definition.surfaces):
            from .beam_pair_ends import validate_beam_pair_ends
            return validate_beam_pair_ends(model, definition, settings)
        from .beam_pair import validate_beam_pair_contact

        return validate_beam_pair_contact(model, definition, settings)
    if any(item.line_stiffness is not None or item.endpoint_stiffness is not None
           for item in definition.interactions):
        return ContactValidationReport("invalid", (
            "line and endpoint stiffness are admitted only for declared round beam contact",
        ))
    if any(item.normal_stiffness is None for item in definition.interactions):
        return ContactValidationReport("invalid", (
            "shell contact requires explicit normal_stiffness in pressure-law units",
        ))
    if any(item.normal_law == "augmented" for item in definition.interactions):
        general_patch = (settings is not None and settings.pair_motion == "general"
                         and not definition.rigid_bodies
                         and len(definition.surfaces) == 2
                         and (any(len(item.element_ids) > 1
                                  for item in definition.surfaces)
                              or _exact_single_q4_pair(model, definition)
                              or _exact_monolithic_s3_pair(model, definition, settings)))
        restricted_patch = (settings is not None
                            and settings.pair_motion == "restricted_sliding"
                            and not definition.rigid_bodies
                            and len(definition.surfaces) == 2
                            and any(len(item.element_ids) > 1 for item in definition.surfaces))
        if (not (general_patch or restricted_patch)
                or settings.pressure_tolerance is None
                or len(definition.interactions) != 1):
            return ContactValidationReport("invalid", (
                "augmented law requires an admitted multi-face shell path and explicit pressure_tolerance",
            ))
    if not definition.rigid_bodies:
        if settings is not None and settings.pair_motion == "general":
            if (any(len(surface.element_ids) > 1 for surface in definition.surfaces)
                    or _exact_single_q4_pair(model, definition)
                    or _exact_monolithic_s3_pair(model, definition, settings)):
                from .s3_patch_static import validate_general_s3_patch

                return validate_general_s3_patch(model, definition, settings)
            from .s3_pair_static import validate_general_s3_pair

            return validate_general_s3_pair(model, definition, settings)
        if settings is not None and settings.pair_motion == "restricted_sliding":
            if len(definition.surfaces) == 2 and any(
                len(surface.element_ids) > 1 for surface in definition.surfaces
            ):
                from .patch_sliding import validate_restricted_sliding_patch

                return validate_restricted_sliding_patch(model, definition, settings)
            from .sliding import validate_restricted_sliding_pair

            return validate_restricted_sliding_pair(model, definition)
        from .pair import validate_planar_shell_pair

        return validate_planar_shell_pair(model, definition)
    if (len(definition.rigid_bodies) == 1
            and definition.rigid_bodies[0].kind in {"sphere", "cylinder"}):
        from .sphere import validate_sphere_shell_contact

        report = validate_sphere_shell_contact(model, definition, settings)
        if (analysis_control == "displacement"
                and definition.rigid_bodies[0].path_translation is not None):
            return ContactValidationReport(
                report.status, report.issues,
                report.profile.replace("_FORCE_STATIC_DEV_V0", "_DISPLACEMENT_STATIC_DEV_V0"),
            )
        return report
    if settings is not None and settings.pair_motion != "guided":
        return ContactValidationReport("invalid", ("rigid plane requires guided pair_motion",))
    surface_by_name = {surface.name: surface for surface in definition.surfaces}
    rigid_by_name = {body.name: body for body in definition.rigid_bodies}
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (1, 1, 1):
        issues.append("this profile requires exactly one surface, one plane, and one interaction")
    from anyfesolver.e4_pl_element import FORMULATION_ID as q4_formulation
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_formulation

    contains_s3 = False

    for surface in definition.surfaces:
        for element_id in surface.element_ids:
            element = model.mesh.elements.get(element_id)
            if element is None:
                issues.append(f"surface {surface.name}: missing element {element_id}")
                continue
            formulation = getattr(element, "formulation_id", None)
            if formulation not in {q4_formulation, s3_formulation}:
                issues.append(f"surface {surface.name}: element {element_id} is not admitted Q4/S3")
                continue
            if formulation == s3_formulation:
                contains_s3 = True
                try:
                    from .overlap import reference_shell_face

                    reference_shell_face(model, 1, element_id, surface.side)
                except ValueError as exc:
                    issues.append(f"surface {surface.name}: {exc}")
                continue
            coordinates = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            diagonal = max(float(np.linalg.norm(coordinates[i] - coordinates[j]))
                           for i in range(4) for j in range(i))
            normal = np.cross(coordinates[1] - coordinates[0], coordinates[3] - coordinates[0])
            area = float(np.linalg.norm(normal))
            if area <= 1.0e-14 * diagonal * diagonal:
                issues.append(f"surface {surface.name}: element {element_id} is degenerate")
            elif abs(float((coordinates[2] - coordinates[0]) @ normal)) / area > 1.0e-10 * diagonal:
                issues.append(f"surface {surface.name}: warped Q4 element {element_id} is not yet admitted")
    for interaction in definition.interactions:
        if interaction.surface not in surface_by_name or interaction.target not in rigid_by_name:
            issues.append(f"interaction {interaction.name}: missing surface or rigid body")
    moving = (len(definition.rigid_bodies) == 1 and
              definition.rigid_bodies[0].path_translation is not None)
    profile = (_MOVING_PLANE_PROFILE if moving else
               _S3_PLANE_PROFILE if contains_s3 else _Q4_PLANE_PROFILE)
    if moving and analysis_control == "displacement":
        profile = profile.replace("_FORCE_STATIC_DEV_V0", "_DISPLACEMENT_STATIC_DEV_V0")
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


class _PlanarQ4PlaneManager:
    def __init__(self, model: FEModel, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        report = validate_general_contact(model, definition)
        if not report.valid or not definition.rigid_bodies:
            raise ValueError("unsupported general contact: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        settings = ContactSolverSettings() if settings is None else settings
        self.settings = settings
        interaction = definition.interactions[0]
        surface = next(item for item in definition.surfaces if item.name == interaction.surface)
        if len(surface.element_ids) > settings.maximum_candidate_pairs:
            raise MemoryError("contact candidate limit exceeded; no pair was discarded")
        body = next(item for item in definition.rigid_bodies if item.name == interaction.target)
        self.stiffness = interaction.normal_stiffness
        self.gap_tolerance = interaction.gap_tolerance
        self.swept_trial_cutbacks = 0
        self.normal = np.asarray(body.normal, dtype=float)
        self.reference_point = np.asarray(body.point, dtype=float)
        self.path_translation = (np.zeros(3) if body.path_translation is None
                                 else np.asarray(body.path_translation, dtype=float))
        self.path_factor = 0.0
        self.point = self.reference_point.copy()
        node_ids = sorted({int(node_id) for element_id in surface.element_ids
                           for node_id in model.mesh.elements[element_id].node_ids})
        pressure_index = {node_id: index for index, node_id in enumerate(node_ids)}
        self.output_nodes = tuple(
            (np.asarray(model.mesh.get_node(node_id).coords(), dtype=float),
             np.asarray(model.mesh.dof_manager.get_node_dofs(node_id)[:6], dtype=np.intp))
            for node_id in node_ids
        )
        n_dof = model.mesh.dof_manager.total_dofs
        support = np.zeros(len(node_ids), dtype=float)
        reference_weighted_gap = np.zeros(len(node_ids), dtype=float)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        self.patches = []
        contact_length_scale = 0.0
        from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_formulation

        for element_id in surface.element_ids:
            element = model.mesh.elements[element_id]
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            contact_length_scale = max(
                contact_length_scale,
                float(np.max(np.linalg.norm(reference - reference[0], axis=1))),
            )
            dofs = np.asarray(
                [model.mesh.dof_manager.get_node_dofs(int(node_id))[:6]
                 for node_id in element.node_ids], dtype=np.intp
            )
            normal = np.cross(reference[1] - reference[0], reference[-1] - reference[0])
            normal /= np.linalg.norm(normal)
            director = element.physical_reference_director
            if director is not None:
                normal = np.asarray(director, dtype=float)
            coordinate = {"midsurface": 0.0, "top": 1.0, "bottom": -1.0}[surface.side]
            if element.formulation_id == s3_formulation:
                offset = float(element.material_surface_offset_from_reference(coordinate))
            else:
                offset = 0.5 * coordinate * float(element.thickness)
            local_gap_gradient = np.r_[self.normal, offset * np.cross(normal, self.normal)]
            weights = []
            shapes = []
            for local, weight in zip(element.gauss_points, element.gauss_weights):
                if element.formulation_id == s3_formulation:
                    N, dxi, deta = element.compute_native_surface_shape_functions(
                        float(local[0]), float(local[1])
                    )
                else:
                    N, dxi, deta = element.compute_shape_functions(
                        float(local[0]), float(local[1])
                    )
                jacobian = float(np.linalg.norm(np.cross(dxi @ reference, deta @ reference)))
                if not np.isfinite(jacobian) or jacobian <= 0.0:
                    raise ValueError(f"contact element {element_id} has invalid reference measure")
                weights.append(float(weight) * jacobian)
                shapes.append(N)
            N = np.asarray(shapes, dtype=float)
            w = np.asarray(weights, dtype=float)
            element_support = N.T @ w
            if np.any(element_support <= 0.0):
                raise ValueError(f"contact element {element_id} has nonpositive pressure support")
            overlap = N.T @ (w[:, None] * N)
            reference_gap = (reference + offset * normal - self.point) @ self.normal
            for a, node_id_a in enumerate(element.node_ids):
                pressure_row = pressure_index[int(node_id_a)]
                support[pressure_row] += float(element_support[a])
                reference_weighted_gap[pressure_row] += float(overlap[a] @ reference_gap)
                for i in range(len(element.node_ids)):
                    for component in range(6):
                        coefficient = float(overlap[a, i] * local_gap_gradient[component])
                        if coefficient:
                            rows.append(pressure_row)
                            cols.append(int(dofs[i, component]))
                            data.append(coefficient)
            self.patches.append((reference, dofs, N, offset * normal,
                                 local_gap_gradient))
        if np.any(support <= 0.0):
            raise ValueError("contact pressure support must be positive")
        self.support = support
        reference_weighted_gap[
            np.abs(reference_weighted_gap)
            <= 256.0 * np.finfo(float).eps * contact_length_scale * support
        ] = 0.0
        self.reference_weighted_gap = reference_weighted_gap
        self.gap_gradient = sparse.coo_matrix(
            (data, (rows, cols)), shape=(len(node_ids), n_dof)
        ).tocsr()
        initial_u = (np.zeros(model.mesh.dof_manager.total_dofs)
                     if initial_displacement is None else initial_displacement)
        initial = self.evaluate(initial_u, tangent=False)[2]
        reference_scale = max(
            1.0,
            float(np.linalg.norm(self.point)),
            *(float(np.linalg.norm(reference)) for reference, _ in self.output_nodes),
        )
        roundoff = 64.0 * np.finfo(float).eps * reference_scale
        if initial.minimum_gap < -roundoff:
            raise ValueError(
                f"initial contact overclosure {initial.minimum_gap:.6g} m is not "
                "admitted; resolve initial geometry before solving"
            )

    def restart_contract(self) -> dict[str, object]:
        """Bind the state-free C0 penalty law to a versioned static checkpoint."""

        return {
            "profile": self.profile,
            "surfaces": [
                {"name": item.name, "element_ids": list(item.element_ids),
                 "side": item.side}
                for item in self.definition.surfaces
            ],
            "rigid_bodies": [
                {"name": item.name, "kind": item.kind, "point": list(item.point),
                 "normal": list(item.normal),
                 **({"path_translation": list(item.path_translation)}
                    if item.path_translation is not None else {})}
                for item in self.definition.rigid_bodies
            ],
            "interactions": [
                {"name": item.name, "surface": item.surface,
                 "target": item.target, "normal_stiffness": item.normal_stiffness,
                 "gap_tolerance": item.gap_tolerance,
                 "normal_law": item.normal_law}
                for item in self.definition.interactions
            ],
            "settings": {
                "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
            },
        }

    def point_at(self, factor: float) -> np.ndarray:
        value = float(factor)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("rigid-plane path factor must be finite and nonnegative")
        point = self.reference_point + value * self.path_translation
        if not np.all(np.isfinite(point)):
            raise ValueError("rigid-plane path position is nonfinite")
        return point

    def set_path_factor(self, factor: float) -> None:
        self.point = self.point_at(factor)
        self.path_factor = float(factor)

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        n_dof = self.model.mesh.dof_manager.total_dofs
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != n_dof or not np.all(np.isfinite(u)):
            raise ValueError("contact displacement has invalid size or nonfinite values")
        plane_shift = -float((self.point - self.reference_point) @ self.normal)
        weighted_gap = self.reference_weighted_gap + plane_shift * self.support + np.asarray(
            self.gap_gradient @ u, dtype=float
        ).reshape(-1)
        normalized_gap = weighted_gap / self.support
        active_gap = np.minimum(normalized_gap, 0.0)
        # At exact first touch, choose the closed-side generalized tangent so
        # a free rigid approach mode can enter contact without a singular
        # initial Newton system.  Zero-gap nodes carry no pressure yet.
        active_mask = normalized_gap <= 0.0
        energy = 0.5 * self.stiffness * float(
            self.support @ (active_gap * active_gap)
        )
        force = np.asarray(
            self.gap_gradient.T @ (self.stiffness * active_gap), dtype=float
        ).reshape(-1)
        if tangent:
            active_gradient = self.gap_gradient[active_mask]
            stiffness = (
                self.stiffness * active_gradient.T
                @ sparse.diags(1.0 / self.support[active_mask])
                @ active_gradient
            ).tocsr()
        else:
            stiffness = None
        minimum_gap = float("inf")
        shell_force = np.zeros(3, dtype=float)
        shell_moment = np.zeros(3, dtype=float)
        for reference, dofs, N, skin_offset, gradient in self.patches:
            nodal_gap = (
                (reference + u[dofs[:, :3]] + skin_offset - self.point) @ self.normal
                + u[dofs[:, 3:]] @ gradient[3:]
            )
            sample_gap = N @ nodal_gap
            # A bilinear Q4 gap against a fixed plane has its minimum at a
            # corner; checking only Gauss points can hide edge penetration.
            minimum_gap = min(minimum_gap, float(np.min(sample_gap)),
                              float(np.min(nodal_gap)))
        for reference, dofs in self.output_nodes:
            local_force = force[dofs]
            shell_force += local_force[:3]
            shell_moment += np.cross(reference + u[dofs[:3]], local_force[:3])
            shell_moment += local_force[3:]
        # ``force`` is the internal residual dPi/du.  Physical contact force
        # on the shell is its negative; the plane receives the opposite.
        shell_tuple = tuple(float(-x) for x in shell_force)
        rigid_tuple = tuple(float(x) for x in shell_force)
        moment_tuple = tuple(float(-x) for x in shell_moment)
        rigid_moment_tuple = tuple(float(x) for x in shell_moment)
        results = ContactResults(
            self.profile,
            minimum_gap,
            energy,
            shell_tuple,
            rigid_tuple,
            moment_tuple,
            rigid_moment_tuple,
            int(np.count_nonzero(normalized_gap < 0.0)),
            minimum_gap >= -self.gap_tolerance,
        )
        return force, stiffness, results

    def admissible_step_fraction(self, start: np.ndarray, end: np.ndarray) -> float:
        """Bound a linear trial path by the physical corner gap tolerance.

        For this fixed-plane, affine Q4 profile each corner gap is affine in
        the trial coordinates.  The endpoint bound is therefore a complete
        swept certificate for the represented skin, including its edges.
        """

        start_u = np.asarray(start, dtype=float).reshape(-1)
        end_u = np.asarray(end, dtype=float).reshape(-1)
        fraction = 1.0
        for reference, dofs, _N, skin_offset, gradient in self.patches:
            base = (reference + skin_offset - self.point) @ self.normal
            first = base + start_u[dofs] @ gradient
            last = base + end_u[dofs] @ gradient
            crossing = last < -self.gap_tolerance
            if np.any(crossing):
                allowed = (first[crossing] + self.gap_tolerance) / (
                    first[crossing] - last[crossing]
                )
                fraction = min(fraction, float(np.min(allowed)))
        if fraction >= 1.0:
            return 1.0
        self.swept_trial_cutbacks += 1
        return max(0.0, 0.98 * fraction)

    def swept_increment_admissible(self, start: np.ndarray, end: np.ndarray,
                                   start_factor: float,
                                   end_factor: float) -> bool:
        if not np.any(self.path_translation):
            return True
        first_point = self.point_at(start_factor)
        last_point = self.point_at(end_factor)
        for reference, dofs, _N, skin_offset, gradient in self.patches:
            first_gap = ((reference + skin_offset - first_point) @ self.normal
                         + np.asarray(start)[dofs] @ gradient)
            last_gap = ((reference + skin_offset - last_point) @ self.normal
                        + np.asarray(end)[dofs] @ gradient)
            if np.min(first_gap) < -self.gap_tolerance:
                raise ValueError("committed rigid-plane pose exceeds the gap tolerance")
            if np.min(last_gap) < -self.gap_tolerance:
                self.swept_trial_cutbacks += 1
                return False
        return True

    def tool_work_increment(self, start: np.ndarray, end: np.ndarray,
                            start_factor: float, end_factor: float) -> float:
        if not np.any(self.path_translation):
            return 0.0
        previous_factor = self.path_factor
        try:
            self.set_path_factor(start_factor)
            first_force = np.asarray(self.evaluate(start, tangent=False)[2].rigid_force)
            self.set_path_factor(end_factor)
            last_force = np.asarray(self.evaluate(end, tangent=False)[2].rigid_force)
        finally:
            self.set_path_factor(previous_factor)
        motion = (float(end_factor) - float(start_factor)) * self.path_translation
        work = float(0.5 * (first_force + last_force) @ motion)
        if not np.isfinite(work):
            raise ValueError("rigid-plane contact work is nonfinite")
        return work


@contextmanager
def contact_scope(model: FEModel, definition: ContactDefinition,
                   settings: ContactSolverSettings | None = None,
                   initial_displacement: np.ndarray | None = None,
                   *, allow_augmented: bool = False,
                   analysis_control: str = "force") -> Iterator[object]:
    settings = ContactSolverSettings() if settings is None else settings
    from .triangulated_rigid import TriangulatedRigidContactBody
    if any(type(body) is TriangulatedRigidContactBody for body in definition.rigid_bodies):
        from .triangulated_rigid_static import TriangulatedRigidManager
        report = validate_general_contact(model,definition,settings,analysis_control=analysis_control)
        if not report.valid:
            raise ValueError('unsupported rigid mesh contact: '+'; '.join(report.issues))
        if settings.shell_skin_kinematics == 'corotational_nodal_director_dev':
            from .finite_rigid_patch import FiniteRigidPatchManager
            manager = FiniteRigidPatchManager(model, definition, settings, initial_displacement)
        else:
            manager = TriangulatedRigidManager(model, definition, settings, initial_displacement)
        token = _ACTIVE_CONTACT.set(manager)
        try:
            yield manager
        finally:
            _ACTIVE_CONTACT.reset(token)
        return
    if settings.beam_end_cap_work == 'hemisphere_coupled_tool_dev':
        from .beam_hemisphere_coupled_tool import CoupledHemisphereToolManager
        report = validate_general_contact(model, definition, settings,
                                          analysis_control=analysis_control)
        if not report.valid:
            raise ValueError('unsupported coupled tool contact: ' + '; '.join(report.issues))
        manager = CoupledHemisphereToolManager(model, definition, settings, initial_displacement)
        token = _ACTIVE_CONTACT.set(manager)
        try:
            yield manager
        finally:
            _ACTIVE_CONTACT.reset(token)
        return
    if settings.beam_end_cap_work == "hemisphere_support_dev":
        from .beam_hemisphere_plane import HemispherePlaneManager
        report = validate_general_contact(model, definition, settings,
                                          analysis_control=analysis_control)
        if not report.valid:
            raise ValueError('unsupported hemisphere-plane contact: ' + '; '.join(report.issues))
        if (len(definition.interactions) > 1
                and any(s.beam_radius is None for s in definition.surfaces)):
            from .multiple_beam_s3_caps import MultipleBeamS3CapManager
            manager = MultipleBeamS3CapManager(model, definition, settings, initial_displacement,
                                               analysis_control=analysis_control)
        elif not definition.rigid_bodies:
            from .deformable_beam_s3_caps import DeformableBeamS3CapManager
            manager = DeformableBeamS3CapManager(model, definition, settings, initial_displacement)
        elif len(definition.interactions) > 1:
            from .multiple_beam_rigid import MultipleBeamRigidManager
            manager = MultipleBeamRigidManager(model, definition, settings, initial_displacement,
                                               analysis_control=analysis_control)
        elif definition.rigid_bodies[0].kind == 'cylinder':
            if definition.rigid_bodies[0].path_translation is not None:
                from .beam_hemisphere_cylinder_motion import TranslatingHemisphereCylinderManager
                manager = TranslatingHemisphereCylinderManager(model, definition, settings, initial_displacement)
            else:
                from .beam_hemisphere_cylinder import HemisphereCylinderManager
                manager = HemisphereCylinderManager(model, definition, settings, initial_displacement)
        elif definition.rigid_bodies[0].kind == 'sphere':
            from .beam_hemisphere_sphere import HemisphereSphereManager
            manager = HemisphereSphereManager(model, definition, settings, initial_displacement)
        else:
            manager = HemispherePlaneManager(model, definition, settings, initial_displacement)
        token = _ACTIVE_CONTACT.set(manager)
        try:
            yield manager
        finally:
            _ACTIVE_CONTACT.reset(token)
        return
    if len(definition.interactions) > 1:
        if definition.rigid_bodies and all(surface.beam_radius is not None
                                          for surface in definition.surfaces):
            from .multiple_beam_rigid import MultipleBeamRigidManager
            manager = MultipleBeamRigidManager(model, definition, settings, initial_displacement,
                                              analysis_control=analysis_control)
        elif definition.rigid_bodies and all(body.kind == 'plane' for body in definition.rigid_bodies):
            from .multiple_shell_planes import MultipleShellPlaneManager
            manager = MultipleShellPlaneManager(model, definition, settings, initial_displacement,
                                                analysis_control=analysis_control)
        else:
            from .multiple_projected_pairs import MultipleProjectedPairManager
            manager = MultipleProjectedPairManager(model, definition, settings, initial_displacement,
                                                  analysis_control=analysis_control)
        token = _ACTIVE_CONTACT.set(manager)
        try:
            yield manager
        finally:
            _ACTIVE_CONTACT.reset(token)
        return
    if settings.warped_q4_plane:
        report = validate_general_contact(
            model, definition, settings, analysis_control=analysis_control,
        )
        if not report.valid:
            raise ValueError("unsupported warped Q4 plane contact: "
                             + "; ".join(report.issues))
    if any(item.normal_law == "projected_penalty" for item in definition.interactions):
        report = validate_general_contact(
            model, definition, settings, analysis_control=analysis_control,
        )
        if not report.valid:
            raise ValueError("unsupported projected penalty contact: " + "; ".join(report.issues))
    if any(item.normal_law == "augmented" for item in definition.interactions):
        if not allow_augmented:
            raise ValueError("augmented contact requires the contact path analysis owner")
        report = validate_general_contact(model, definition, settings,
                                          analysis_control=analysis_control)
        if not report.valid:
            raise ValueError("unsupported augmented contact: " + "; ".join(report.issues))
    if not definition.rigid_bodies and any(
        surface.beam_radius is not None for surface in definition.surfaces
    ):
        if any(surface.beam_radius is None for surface in definition.surfaces):
            if settings.beam_shell_geometry == "jet2_deformable_s3_dev":
                from .deformable_beam_s3_static import DeformableBeamS3Manager
                manager = DeformableBeamS3Manager(model, definition, settings, initial_displacement)
            elif any(surface.beam_radius is None and len(surface.element_ids) > 1
                   for surface in definition.surfaces):
                from .beam_shell_patch import _BeamShellPatchManager

                manager = _BeamShellPatchManager(model, definition, settings, initial_displacement)
            else:
                from .beam_shell import _BeamShellManager

                manager = _BeamShellManager(model, definition, settings, initial_displacement)
        else:
            if any(surface.beam_end_caps == 'hemisphere' for surface in definition.surfaces):
                if any(len(surface.element_ids)>1 for surface in definition.surfaces):
                    from .beam_chain_ends import BeamChainEndManager
                    manager = BeamChainEndManager(model, definition, settings, initial_displacement)
                else:
                    from anyfesolver.elements import QuadraticBeamElement
                    if any(type(model.mesh.elements.get(element)) is QuadraticBeamElement
                           for surface in definition.surfaces for element in surface.element_ids):
                        from .beam_curve_ends import BeamCurveEndManager
                        manager = BeamCurveEndManager(model, definition, settings, initial_displacement)
                    else:
                        from .beam_pair_ends import BeamPairEndManager
                        manager = BeamPairEndManager(model, definition, settings, initial_displacement)
            elif any(len(surface.element_ids) > 1 for surface in definition.surfaces):
                from .beam_chain_pair import _BeamChainPairManager

                manager = _BeamChainPairManager(model, definition, settings, initial_displacement)
            else:
                from anyfesolver.elements import BeamElement, QuadraticBeamElement

                element_types = tuple(
                    type(model.mesh.elements.get(surface.element_ids[0]))
                    for surface in definition.surfaces
                )
                if (len(element_types) == 2
                        and all(kind in (BeamElement, QuadraticBeamElement)
                                for kind in element_types)
                        and QuadraticBeamElement in element_types):
                    from .beam_curve_pair_static import B3PairManager

                    manager = B3PairManager(model, definition, settings,
                                            initial_displacement)
                else:
                    from .beam_pair import _BeamPairManager

                    manager = _BeamPairManager(model, definition, settings,
                                               initial_displacement)
    elif definition.rigid_bodies:
        if settings.warped_q4_plane:
            from .warped_q4_static import WarpedQ4PlaneManager

            manager = WarpedQ4PlaneManager(model, definition, settings,
                                           initial_displacement)
        elif any(surface.beam_radius is not None for surface in definition.surfaces):
            if len(definition.rigid_bodies) == 1 and definition.rigid_bodies[0].kind == "capped_cylinder":
                from .capped_cylinder import CappedCylinderBeamManager
                manager = CappedCylinderBeamManager(model, definition, settings, initial_displacement)
            elif len(definition.rigid_bodies) == 1 and definition.rigid_bodies[0].kind in {
                "sphere", "cylinder",
            }:
                if any(surface.beam_end_caps=='hemisphere' for surface in definition.surfaces):
                    from .beam_radial_ends import BeamRadialEndManager
                    manager=BeamRadialEndManager(model,definition,settings,initial_displacement)
                else:
                    from .beam_radial import _BeamRadialManager
                    manager = _BeamRadialManager(model, definition, settings, initial_displacement)
            else:
                from .beam_plane import _BeamPlaneManager

                manager = _BeamPlaneManager(model, definition, settings, initial_displacement)
        elif (len(definition.rigid_bodies) == 1
                and definition.rigid_bodies[0].kind in {"sphere", "cylinder"}):
            from .sphere import _SphereShellManager

            manager = _SphereShellManager(model, definition, settings,
                                          initial_displacement)
        else:
            if settings.pair_motion != "guided":
                raise ValueError("rigid plane requires guided pair_motion")
            manager = _PlanarQ4PlaneManager(model, definition, settings, initial_displacement)
    else:
        if settings.pair_motion == "general":
            if (any(len(surface.element_ids) > 1 for surface in definition.surfaces)
                    or _exact_single_q4_pair(model, definition)
                    or _exact_monolithic_s3_pair(model, definition, settings)):
                from .s3_patch_static import GeneralS3PatchManager

                manager = GeneralS3PatchManager(
                    model, definition, settings, initial_displacement,
                )
            else:
                from .s3_pair_static import GeneralS3PairManager

                manager = GeneralS3PairManager(
                    model, definition, settings, initial_displacement,
                )
        elif settings.pair_motion == "restricted_sliding":
            if len(definition.surfaces) == 2 and any(
                len(surface.element_ids) > 1 for surface in definition.surfaces
            ):
                from .patch_sliding import _RestrictedSlidingPatchFEManager as manager_type
            else:
                from .sliding import _RestrictedSlidingPairFEManager as manager_type

            manager = manager_type(
                model, definition, settings, initial_displacement=initial_displacement,
            )
        else:
            from .pair import _PlanarShellPairManager

            manager = _PlanarShellPairManager(model, definition, settings, initial_displacement)
    if analysis_control == "displacement" and definition.rigid_bodies:
        report = validate_general_contact(
            model, definition, settings, analysis_control=analysis_control,
        )
        if not report.valid:
            raise ValueError("unsupported displacement contact: " + "; ".join(report.issues))
        manager.profile = report.profile
    token = _ACTIVE_CONTACT.set(manager)
    try:
        yield manager
    finally:
        try:
            discard = getattr(manager, "discard_geometry_preparation", None)
            if discard is not None:
                discard()
        finally:
            _ACTIVE_CONTACT.reset(token)
