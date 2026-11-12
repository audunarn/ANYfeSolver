"""Narrowly admitted nonlinear Newmark shell/beam contact development profiles.

The coupled residual is evaluated by the ordinary nonlinear structural
assembler inside one contact scope. The Q4 profile uses positive-definite
reduced mass; a separate guided single-S3 V2D profile retains declared
massless rotations and recovers their rates from algebraic equilibrium.
A distinct S3 history profile scales the same dead nodal load by a
piecewise-linear factor in time; its complete base load must lie in the
physical mass range so the scalar history never forces the algebraic
manifold. No profile here qualifies the broader C6 impact capability.
An explicit classical B2/B3 opt-in reuses line-contact work and actual physical
mass with a complete tensor-Bernstein curve/time gap enclosure.
"""

from __future__ import annotations

from dataclasses import dataclass,field
from bisect import bisect_right
from functools import wraps
from copy import deepcopy
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from anymaterial.isotropic import IsotropicMaterial

from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
from anyfesolver.assembly import assemble_mass_matrix, build_constraint_transformation
from anyfesolver.boundary import LoadCase
from anyfesolver.control import CancellationToken
from anyfesolver.dynamics import TransientConfig
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.matrix_assembly import assemble_load_vector, assemble_stiffness_matrix
from anyfesolver.nonlinear_state import (
    commit_state_candidate, discard_active_state_candidate,
    _run_with_nonlinear_state_cleanup,
)
from anyfesolver.nonlinear_restart import (
    analysis_configuration_fingerprint, model_configuration_fingerprint,
)
from anyfesolver.nonlinear_static import (
    _assemble_nonlinear_system, _is_known_cancellation_checkpoint,
)

from .core import (
    ContactDefinition,
    ContactResults,
    ContactSolverSettings,
    _PlanarQ4PlaneManager,
    contact_scope,
    suspend_contact,
    validate_general_contact,
)
from .beam_plane import _BeamPlaneResults
from .beam_radial import _RadialResults
from .multiple_beam_rigid import MultipleBeamRigidResults
from .multiple_shell_planes import MultipleShellPlaneResults
from .pair import _PairResults
from .s3_patch_static import _PatchResults
from .rigid_translation import ContactRigidTranslationHistory, _PlaneTranslations
from .free_rigid import (ContactFreeRigidBody, FreeRigidBodyState, _FreeSphereManager,
                        FREE_SPHERE_PROFILE, FREE_SPHERE_HISTORY_PROFILE,
                        FREE_SHELL_SPHERE_PROFILE, FREE_SHELL_SPHERE_HISTORY_PROFILE)
from .descriptor_rates import directional_descriptor_rate
from .algebraic_path import affine_descriptor_path
from .event_subdivision import EventNeedsCutback, localize_branch_event


_PROFILE = "PLANAR_Q4_FIXED_PLANE_NEWMARK_PENALTY_DEV_V1"
_Q4_PLASTIC_PROFILE = "PLANAR_Q4_FIXED_PLANE_NEWMARK_DNV_PLASTIC_DEV_V1"
_Q4_PLASTIC_RESTART_SCHEMA = "ANYSOLVER_Q4_PLASTIC_TRANSIENT_CONTACT_DEV_CHECKPOINT_V7"
_Q4_EVENT_PROFILE = "PLANAR_Q4_FIXED_PLANE_NEWMARK_EVENT_DEV_V1"
_Q4_PLASTIC_EVENT_PROFILE = "PLANAR_Q4_FIXED_PLANE_NEWMARK_DNV_PLASTIC_EVENT_DEV_V1"
_Q4_EVENT_RESTART_SCHEMA = "ANYSOLVER_Q4_PLANE_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V8"
_Q4_PLASTIC_EVENT_RESTART_SCHEMA = "ANYSOLVER_Q4_PLASTIC_PLANE_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V9"
_MULTIPLE_Q4_EVENT_PROFILE = "MULTIPLE_Q4_FIXED_PLANES_NEWMARK_EVENT_DEV_V1"
_MULTIPLE_Q4_EVENT_RESTART_SCHEMA = "ANYSOLVER_MULTIPLE_Q4_PLANE_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V10"
_GUIDED_SHELL_PAIR_EVENT_PROFILE = 'GUIDED_Q4_S3_SHELL_PAIR_NEWMARK_EVENT_DEV_V1'
_GUIDED_SHELL_PAIR_RESTART_SCHEMA = 'ANYSOLVER_GUIDED_SHELL_PAIR_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V11'
_FIXED_PROJECTION_S3_EVENT_PROFILE = 'FIXED_PROJECTION_S3_TARGET_NEWMARK_EVENT_DEV_V1'
_FIXED_PROJECTION_S3_RESTART_SCHEMA = 'ANYSOLVER_FIXED_PROJECTION_S3_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V12'
_NORMAL_S3_SKIN_PROFILE = 'NORMAL_NODAL_S3_MATCHED_OVERLAP_NEWMARK_EVENT_DEV_V3'
_NORMAL_S3_SKIN_RESTART_SCHEMA = 'ANYSOLVER_NORMAL_S3_MATCHED_OVERLAP_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V15'
_NORMAL_S3_POLYGON_PROFILE = 'NORMAL_NODAL_S3_POLYGON_PRESSURE_NEWMARK_EVENT_DEV_V4'
_NORMAL_S3_POLYGON_RESTART_SCHEMA = 'ANYSOLVER_NORMAL_S3_POLYGON_PRESSURE_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V16'
_S3_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_DESCRIPTOR_DEV_V0"
_S3_HISTORY_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_DESCRIPTOR_HISTORY_DEV_V1"
_S3_DIRECTIONAL_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_DIRECTIONAL_RATES_DEV_V1"
_S3_DIRECTIONAL_HISTORY_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_DIRECTIONAL_HISTORY_DEV_V1"
_S3_ALGEBRAIC_PATH_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_ALGEBRAIC_PATH_DEV_V1"
_S3_ALGEBRAIC_PATH_HISTORY_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_ALGEBRAIC_PATH_HISTORY_DEV_V1"
_S3_EVENT_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_EVENT_SUBDIVISION_DEV_V1"
_S3_EVENT_HISTORY_PROFILE = "PLANAR_S3_V2D_FIXED_PLANE_NEWMARK_EVENT_SUBDIVISION_HISTORY_DEV_V1"
_BEAM_PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_VON_KARMAN_FIXED_PLANE_NEWMARK_DEV_V1"
_BEAM_HISTORY_PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_VON_KARMAN_FIXED_PLANE_NEWMARK_HISTORY_DEV_V1"
_MULTIPLE_BEAM_PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_MULTIPLE_FIXED_PLANES_NEWMARK_DEV_V1"
_MULTIPLE_BEAM_HISTORY_PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_MULTIPLE_FIXED_PLANES_NEWMARK_HISTORY_DEV_V1"
_BEAM_TOOL_PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_PRESCRIBED_TRANSLATION_NEWMARK_DEV_V1"
_BEAM_TOOL_HISTORY_PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_PRESCRIBED_TRANSLATION_NEWMARK_HISTORY_DEV_V1"
_TOOL_RESTART_SCHEMA = "ANYSOLVER_TRANSLATING_PLANE_TRANSIENT_CONTACT_DEV_CHECKPOINT_V5"
_FREE_RESTART_SCHEMA = "ANYSOLVER_FREE_SPHERE_TRANSIENT_CONTACT_DEV_CHECKPOINT_V6"
_RESTART_SCHEMA = "ANYSOLVER_Q4_PLANE_TRANSIENT_CONTACT_DEV_CHECKPOINT_V1"
_HISTORY_RESTART_SCHEMA = "ANYSOLVER_Q4_PLANE_TRANSIENT_CONTACT_DEV_CHECKPOINT_V2"
_ALGEBRAIC_PATH_RESTART_SCHEMA = "ANYSOLVER_S3_ALGEBRAIC_PATH_TRANSIENT_CONTACT_DEV_CHECKPOINT_V3"
_EVENT_RESTART_SCHEMA = "ANYSOLVER_S3_EVENT_SUBDIVISION_TRANSIENT_CONTACT_DEV_CHECKPOINT_V4"


@dataclass(frozen=True)
class ContactNodalLoadHistory:
    """Piecewise-linear scale of one fixed nodal LoadCase in physical time."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        try:
            points = tuple(tuple(point) for point in self.points)
        except TypeError as exc:
            raise ValueError("nodal load history needs time-factor pairs") from exc
        if (len(points) < 2 or any(len(point) != 2 for point in points)
                or any(not all(
                    type(value) is not bool
                    and isinstance(value, (int, float, np.integer, np.floating))
                    and np.isfinite(value)
                    for value in point)
                       for point in points)
                or points[0][0] != 0.0
                or any(right[0] <= left[0]
                       for left, right in zip(points, points[1:]))):
            raise ValueError("nodal load history needs finite increasing time-factor pairs from zero")
        object.__setattr__(self, "points", tuple((float(t), float(f))
                                                 for t, f in points))

    def factor(self, time_s: float) -> float:
        if time_s < 0.0 or time_s > self.points[-1][0]:
            raise ValueError("nodal load history does not cover the requested time")
        if time_s == self.points[-1][0]:
            return float(self.points[-1][1])
        index = max(0, bisect_right(self.points, (time_s, float("inf")))-1)
        first, last = self.points[index:index+2]
        fraction = (time_s-first[0])/(last[0]-first[0])
        return float(first[1]+fraction*(last[1]-first[1]))


def _hermite_linear_load_work(
    start: np.ndarray, end: np.ndarray,
    start_velocity: np.ndarray, end_velocity: np.ndarray,
    start_force: np.ndarray, end_force: np.ndarray, dt: float,
) -> float:
    """Exact work of linear dead load on the declared cubic Hermite step path."""
    total = 0.0
    for s in (0.5-0.5/np.sqrt(3.0), 0.5+0.5/np.sqrt(3.0)):
        derivative = (
            (6*s*s-6*s)*start + (3*s*s-4*s+1)*dt*start_velocity
            + (-6*s*s+6*s)*end + (3*s*s-2*s)*dt*end_velocity
        )
        force = (1-s)*start_force+s*end_force
        total += 0.5*float(force @ derivative)
    return total


@dataclass(frozen=True)
class ContactTransientSolverControls:
    """Nonlinear iteration limits for the developing contact driver."""

    maximum_newton_iterations: int = 20
    maximum_cutbacks: int = 8
    minimum_dt: float = 1.0e-8
    equilibrium_tolerance: float = 1.0e-8
    allow_s3_descriptor_dev: bool = False
    s3_descriptor_difference_step_s: float = 1.0e-5
    s3_one_sided_rates_dev: bool = False
    s3_algebraic_path_dev: bool = False
    s3_path_position_tolerance_m: float = 1.0e-10
    s3_path_rotation_tolerance_rad: float = 1.0e-9
    s3_event_subdivision_dev: bool = False
    s3_event_time_tolerance_s: float = 1.0e-10
    s3_event_gap_tolerance_m: float = 1.0e-10
    maximum_event_localizations: int = 12
    maximum_event_root_iterations: int = 64
    allow_classical_beam_dev: bool = False
    beam_swept_gap_tolerance_m: float = 1.0e-10
    maximum_beam_swept_subdivisions: int = 512
    allow_multiple_beam_planes_dev: bool = False
    tool_work_absolute_tolerance_j: float = 1e-10
    tool_work_relative_tolerance: float = 1e-8
    maximum_tool_work_subdivisions: int = 64
    allow_free_sphere_dev: bool = False
    allow_free_shell_sphere_dev: bool = False
    allow_q4_plastic_dev: bool = False
    save_q4_material_history_dev: bool = False
    q4_plane_event_subdivision_dev: bool = False
    plane_event_time_tolerance_s: float = 1.e-10
    plane_event_gap_tolerance_m: float = 1.e-10
    maximum_plane_event_endpoint_evaluations: int = 64
    maximum_plane_event_stage_evaluations: int = 128
    save_constraint_reactions_dev: bool = False
    allow_multiple_shell_planes_dev: bool = False
    batch_plane_event_fronts_dev: bool = False
    allow_guided_shell_pair_dev: bool = False
    allow_fixed_projection_s3_target_dev: bool = False
    allow_normal_s3_skin_hermite_dev: bool = False
    normal_s3_polygon_pressure_bounds_dev: bool = False

    allow_deformable_beam_s3_dev: bool = False
    allow_rotational_beam_s3_dev: bool = False
    allow_directional_beam_s3_dev: bool = False
    allow_equilibrated_directional_beam_s3_dev: bool = False

    equilibrated_newmark_acceleration_unknown_dev: bool = False
    allow_multiple_deformable_beam_s3_dev: bool = False

    def __post_init__(self) -> None:
        if type(self.allow_multiple_deformable_beam_s3_dev) is not bool:
            raise ValueError('multiple deformable beam-S3 selection must be boolean')
        if self.allow_multiple_deformable_beam_s3_dev and any(value for name,value in vars(self).items()
                if name.endswith('_dev') and name not in ('allow_multiple_deformable_beam_s3_dev','save_constraint_reactions_dev')):
            raise ValueError('multiple deformable beam-S3 requires its separate fixed-director profile')
        if type(self.equilibrated_newmark_acceleration_unknown_dev) is not bool:
            raise ValueError("equilibrated acceleration selection must be boolean")
        if self.equilibrated_newmark_acceleration_unknown_dev and not self.allow_equilibrated_directional_beam_s3_dev:
            raise ValueError("equilibrated acceleration requires the equilibrated directional profile")
        if type(self.allow_equilibrated_directional_beam_s3_dev) is not bool:
            raise ValueError('equilibrated directional selection must be boolean')
        if self.allow_equilibrated_directional_beam_s3_dev and any(value for name,value in vars(self).items()
                if name.endswith('_dev') and name.startswith(('allow_', 's3_', 'q4_'))
                and name!='allow_equilibrated_directional_beam_s3_dev'):
            raise ValueError('equilibrated directional requires its separate profile')
        if type(self.allow_directional_beam_s3_dev) is not bool:
            raise ValueError('directional beam-S3 selection must be boolean')
        if self.allow_directional_beam_s3_dev and any(value for name,value in vars(self).items()
                if name.endswith('_dev') and name.startswith(('allow_', 's3_', 'q4_'))
                and name!='allow_directional_beam_s3_dev'):
            raise ValueError('directional beam-S3 requires its separate descriptor profile')
        if type(self.allow_rotational_beam_s3_dev) is not bool:
            raise ValueError('rotational beam-S3 selection must be boolean')
        if self.allow_rotational_beam_s3_dev and any(
                value for name,value in vars(self).items()
                if name.endswith('_dev') and name.startswith(('allow_', 's3_', 'q4_'))
                and name not in ('allow_rotational_beam_s3_dev',)):
            raise ValueError('rotational beam-S3 requires its separate descriptor profile')
        if type(self.allow_deformable_beam_s3_dev) is not bool:
            raise ValueError('deformable beam-S3 selection must be boolean')
        if self.allow_deformable_beam_s3_dev and any((self.allow_classical_beam_dev,self.allow_s3_descriptor_dev,self.allow_guided_shell_pair_dev,self.allow_fixed_projection_s3_target_dev,self.allow_normal_s3_skin_hermite_dev,self.allow_multiple_shell_planes_dev,self.allow_free_sphere_dev,self.allow_free_shell_sphere_dev,self.allow_q4_plastic_dev,self.q4_plane_event_subdivision_dev)):
            raise ValueError('deformable beam-S3 requires its separate elastic positive-mass profile')
        if type(self.normal_s3_polygon_pressure_bounds_dev) is not bool:
            raise ValueError('normal S3 polygon pressure selection must be boolean')
        if self.normal_s3_polygon_pressure_bounds_dev and not self.allow_normal_s3_skin_hermite_dev:
            raise ValueError('polygon pressure bounds require the normal S3 skin profile')
        if type(self.allow_normal_s3_skin_hermite_dev) is not bool:
            raise ValueError('normal S3 skin selection must be boolean')
        if self.allow_normal_s3_skin_hermite_dev and (
                self.q4_plane_event_subdivision_dev or self.allow_guided_shell_pair_dev
                or self.allow_fixed_projection_s3_target_dev or self.allow_multiple_shell_planes_dev
                or self.allow_classical_beam_dev or self.allow_q4_plastic_dev
                or self.allow_s3_descriptor_dev or self.allow_free_sphere_dev):
            raise ValueError('normal S3 skin paths require their separate elastic non-affine profile')
        if type(self.allow_fixed_projection_s3_target_dev) is not bool:
            raise ValueError('fixed-projection S3 target selection must be boolean')
        if self.allow_fixed_projection_s3_target_dev and (
                not self.q4_plane_event_subdivision_dev or self.allow_guided_shell_pair_dev
                or self.allow_multiple_shell_planes_dev or self.allow_classical_beam_dev
                or self.allow_q4_plastic_dev or self.allow_s3_descriptor_dev
                or self.allow_free_sphere_dev):
            raise ValueError('fixed-projection S3 targets require the separate elastic pressure-event profile')
        if type(self.allow_guided_shell_pair_dev) is not bool:
            raise ValueError('guided shell pair selection must be boolean')
        if self.allow_guided_shell_pair_dev and (not self.q4_plane_event_subdivision_dev
                or self.allow_multiple_shell_planes_dev or self.allow_classical_beam_dev
                or self.allow_q4_plastic_dev or self.allow_s3_descriptor_dev
                or self.allow_free_sphere_dev):
            raise ValueError('guided shell pairs require the separate elastic pressure-event profile')
        if type(self.batch_plane_event_fronts_dev) is not bool:
            raise ValueError('batched plane event front selection must be boolean')
        if self.batch_plane_event_fronts_dev and (not self.q4_plane_event_subdivision_dev
                or not (self.allow_multiple_shell_planes_dev or self.allow_guided_shell_pair_dev
                        or self.allow_fixed_projection_s3_target_dev)):
            raise ValueError('batched plane event fronts require an explicit multiple-Q4 or guided-shell-pair event profile')
        if type(self.allow_multiple_shell_planes_dev) is not bool:
            raise ValueError('multiple-shell-plane selection must be boolean')
        if self.allow_multiple_shell_planes_dev and (not self.q4_plane_event_subdivision_dev
                or self.allow_q4_plastic_dev or self.allow_s3_descriptor_dev
                or self.allow_classical_beam_dev or self.allow_free_sphere_dev):
            raise ValueError('multiple shell planes require the separate elastic Q4 plane-event profile')
        if type(self.save_constraint_reactions_dev) is not bool:
            raise ValueError('constraint-reaction selection must be boolean')
        if type(self.q4_plane_event_subdivision_dev) is not bool:
            raise ValueError('Q4 plane event selection must be boolean')
        if self.q4_plane_event_subdivision_dev and (self.allow_s3_descriptor_dev
                or self.allow_classical_beam_dev or self.allow_free_sphere_dev):
            raise ValueError('Q4 plane events require the separate guided Q4 profile')
        for name in ('plane_event_time_tolerance_s','plane_event_gap_tolerance_m'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or value<=0.:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('maximum_plane_event_endpoint_evaluations','maximum_plane_event_stage_evaluations'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if self.maximum_plane_event_endpoint_evaluations<2:
            raise ValueError('plane event localization requires at least two endpoint evaluations')
        if type(self.allow_q4_plastic_dev) is not bool or type(self.save_q4_material_history_dev) is not bool:
            raise ValueError('Q4 plastic/material-history selections must be boolean')
        if self.save_q4_material_history_dev and not self.allow_q4_plastic_dev:
            raise ValueError('material history requires the Q4 plastic development profile')
        if self.allow_q4_plastic_dev and (self.allow_free_sphere_dev or self.allow_classical_beam_dev
                or self.allow_s3_descriptor_dev):
            raise ValueError('Q4 plastic development requires its separate fixed-plane profile')
        if type(self.allow_free_sphere_dev) is not bool:
            raise ValueError("allow_free_sphere_dev must be boolean")
        if type(self.allow_free_shell_sphere_dev) is not bool:
            raise ValueError('allow_free_shell_sphere_dev must be boolean')
        if self.allow_free_shell_sphere_dev and (not self.allow_free_sphere_dev
                or self.allow_classical_beam_dev or self.allow_s3_descriptor_dev):
            raise ValueError('free shell sphere requires its separate Q4/free-body opt-ins')
        if self.allow_free_sphere_dev and ((not self.allow_classical_beam_dev and not self.allow_free_shell_sphere_dev)
                or self.allow_multiple_beam_planes_dev):
            raise ValueError("free sphere requires one classical beam interaction")
        for name in ("tool_work_absolute_tolerance_j", "tool_work_relative_tolerance"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0.:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.maximum_tool_work_subdivisions) is not int or self.maximum_tool_work_subdivisions < 0:
            raise ValueError("maximum_tool_work_subdivisions must be a nonnegative integer")
        if type(self.allow_multiple_beam_planes_dev) is not bool:
            raise ValueError("allow_multiple_beam_planes_dev must be boolean")
        if self.allow_multiple_beam_planes_dev and not self.allow_classical_beam_dev:
            raise ValueError("multiple beam planes require the classical beam opt-in")
        if type(self.allow_classical_beam_dev) is not bool:
            raise ValueError("allow_classical_beam_dev must be boolean")
        if self.allow_classical_beam_dev and self.allow_s3_descriptor_dev:
            raise ValueError("classical beam and S3 descriptor opt-ins are separate profiles")
        if (isinstance(self.beam_swept_gap_tolerance_m,(bool,np.bool_))
                or not np.isfinite(self.beam_swept_gap_tolerance_m)
                or self.beam_swept_gap_tolerance_m <= 0.):
            raise ValueError("beam_swept_gap_tolerance_m must be finite and positive")
        if (type(self.maximum_beam_swept_subdivisions) is not int
                or self.maximum_beam_swept_subdivisions < 1):
            raise ValueError("maximum_beam_swept_subdivisions must be a positive integer")
        if type(self.maximum_newton_iterations) is not int or self.maximum_newton_iterations < 1:
            raise ValueError("maximum_newton_iterations must be a positive integer")
        if type(self.maximum_cutbacks) is not int or self.maximum_cutbacks < 0:
            raise ValueError("maximum_cutbacks must be a nonnegative integer")
        if not np.isfinite(self.minimum_dt) or self.minimum_dt <= 0.0:
            raise ValueError("minimum_dt must be finite and positive")
        if not np.isfinite(self.equilibrium_tolerance) or self.equilibrium_tolerance <= 0.0:
            raise ValueError("equilibrium_tolerance must be finite and positive")
        if type(self.allow_s3_descriptor_dev) is not bool:
            raise ValueError("allow_s3_descriptor_dev must be boolean")
        if type(self.s3_one_sided_rates_dev) is not bool:
            raise ValueError("s3_one_sided_rates_dev must be boolean")
        if self.s3_one_sided_rates_dev and not self.allow_s3_descriptor_dev:
            raise ValueError("one-sided rates require the S3 descriptor opt-in")
        if type(self.s3_algebraic_path_dev) is not bool:
            raise ValueError("s3_algebraic_path_dev must be boolean")
        if self.s3_algebraic_path_dev and not self.s3_one_sided_rates_dev:
            raise ValueError("algebraic path requires the S3 one-sided-rate opt-in")
        for name in ("s3_path_position_tolerance_m", "s3_path_rotation_tolerance_rad"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0.:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.s3_event_subdivision_dev) is not bool:
            raise ValueError("s3_event_subdivision_dev must be boolean")
        if self.s3_event_subdivision_dev and not self.s3_algebraic_path_dev:
            raise ValueError("event subdivision requires the S3 algebraic-path opt-in")
        for name in ("s3_event_time_tolerance_s", "s3_event_gap_tolerance_m"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0.:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("maximum_event_localizations", "maximum_event_root_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not np.isfinite(self.s3_descriptor_difference_step_s)
                or self.s3_descriptor_difference_step_s <= 0.0):
            raise ValueError("s3_descriptor_difference_step_s must be finite and positive")


@dataclass(frozen=True)
class TransientContactSettings:
    """Compatibility adapter for the original fixed-plane development call."""

    dt: float
    t_end: float
    initial_velocity: np.ndarray
    maximum_newton_iterations: int = 20
    maximum_cutbacks: int = 8
    minimum_dt: float = 1.0e-8
    equilibrium_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.t_end) or self.t_end < 0.0:
            raise ValueError("t_end must be finite and nonnegative")
        ContactTransientSolverControls(
            self.maximum_newton_iterations, self.maximum_cutbacks,
            self.minimum_dt, self.equilibrium_tolerance,
        )
        velocity = np.array(self.initial_velocity, dtype=float, copy=True).reshape(-1)
        if not np.all(np.isfinite(velocity)):
            raise ValueError("initial_velocity must be finite")
        velocity.setflags(write=False)
        object.__setattr__(self, "initial_velocity", velocity)


def _resolve_transient_inputs(transient, solver_controls):
    if type(transient) is TransientContactSettings:
        if solver_controls is not None:
            raise ValueError("legacy transient settings already contain solver controls")
        controls = ContactTransientSolverControls(
            transient.maximum_newton_iterations, transient.maximum_cutbacks,
            transient.minimum_dt, transient.equilibrium_tolerance,
        )
        transient = TransientConfig(
            dt=transient.dt, t_end=transient.t_end,
            initial_velocity=transient.initial_velocity,
        )
    elif type(transient) is TransientConfig:
        controls = (ContactTransientSolverControls() if solver_controls is None
                    else solver_controls)
        if type(controls) is not ContactTransientSolverControls:
            raise TypeError("ContactTransientSolverControls required")
    else:
        raise TypeError("TransientConfig required")
    if (not np.isfinite(transient.dt) or not np.isfinite(transient.t_end)
            or transient.beta != 0.25 or transient.gamma != 0.5
            or transient.hht_alpha != 0.0
            or transient.rayleigh_alpha != 0.0
            or transient.rayleigh_beta != 0.0):
        raise ValueError("fixed-plane contact admits only undamped average-acceleration Newmark")
    if (transient.output_nodes is not None or transient.output_elements is not None
            or transient.include_stress_history or transient.recovery is not None
            or transient.resource_config is not None):
        raise ValueError("selected recovery and resource options are not admitted in this contact profile")
    if type(transient.save_every) is not int or transient.save_every < 1:
        raise ValueError("save_every must be a positive integer")
    return transient, controls


@dataclass(frozen=True)
class TransientContactResults:
    """Accepted dynamic states; the profile remains developmental."""

    profile: str
    times: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    contact: tuple[ContactResults | _PairResults | _PatchResults | _BeamPlaneResults | MultipleBeamRigidResults | MultipleShellPlaneResults | _RadialResults, ...]
    swept_minimum_gaps: np.ndarray
    equilibrium_residuals: np.ndarray
    cutbacks: int
    newton_iterations: tuple[int, ...]
    external_nodal_work_j: np.ndarray
    restart_checkpoint: dict | None = None
    represented_path_transition_times: tuple[float, ...] = ()
    localized_contact_event_times: tuple[float, ...] = ()
    event_endpoint_evaluations: int = 0
    prescribed_tool_work_j: tuple[dict[str, float], ...] = ()
    prescribed_tool_work_error_j: tuple[dict[str, float], ...] = ()
    free_rigid_bodies: tuple[dict[str, FreeRigidBodyState], ...] = ()
    element_states: Mapping = field(default_factory=dict)
    material_state_history: tuple[Mapping,...] = ()
    plane_event_stage_evaluations: int = 0
    cutback_diagnostics: tuple[dict[str, object], ...] = ()
    # Generalized structural constraint forces M a + f_internal/contact - f_load.
    # These require formulation-owned maps before interpretation as physical moments.
    constraint_reactions: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pair_swept_minimum_gaps: tuple[dict[str,float], ...] = ()

    def to_archive_bytes(self, *, max_bytes=16 * 1024 * 1024):
        """Export an independent strict-data snapshot; never admit a restart."""
        from .transient_results_archive import serialize_transient_contact_results
        return serialize_transient_contact_results(self,max_bytes=max_bytes)


def _restart_analysis_contract(transient, controls, manager, profile,
                               base_load_case, external_load,
                               load_history, tool_histories=()) -> dict:
    # The requested end time is a continuation target, not a definition of
    # the already accepted state. All other integration inputs are bound.
    contract = {
        "profile": profile,
        "dt": float(transient.dt),
        "beta": float(transient.beta),
        "gamma": float(transient.gamma),
        "save_every": int(transient.save_every),
        "initial_displacement": (
            None if transient.initial_displacement is None
            else np.asarray(transient.initial_displacement).tolist()
        ),
        "initial_velocity": np.asarray(transient.initial_velocity).tolist(),
        "controls": {
            "maximum_newton_iterations": controls.maximum_newton_iterations,
            "maximum_cutbacks": controls.maximum_cutbacks,
            "minimum_dt": controls.minimum_dt,
            "equilibrium_tolerance": controls.equilibrium_tolerance,
            "allow_s3_descriptor_dev": controls.allow_s3_descriptor_dev,
            "s3_descriptor_difference_step_s": controls.s3_descriptor_difference_step_s,
        },
        "contact": manager.restart_contract(),
    }
    if base_load_case is not None:
        contract["constant_nodal_load"] = {
            "name": base_load_case.name,
            "generalized_vector_si": np.asarray(external_load).tolist(),
        }
    if load_history is not None:
        contract["nodal_load_history"] = [list(point)
                                          for point in load_history.points]
    if controls.equilibrated_newmark_acceleration_unknown_dev:
        contract['controls'].update(equilibrated_newmark_acceleration_unknown_dev=True,
            newmark_iteration_policy='ACCELERATION_UNKNOWN_V1',
            endpoint_roundoff_policy='ULP_MATCHED_NOMINAL_SOLVED_STEP_V1')
    if controls.s3_one_sided_rates_dev:
        # Absent for historical profiles, preserving their checkpoint contract.
        contract["controls"]["s3_one_sided_rates_dev"] = True
    if controls.save_constraint_reactions_dev:
        contract['controls']['save_constraint_reactions_dev'] = True
    if controls.batch_plane_event_fronts_dev:
        contract['controls']['batch_plane_event_fronts_dev'] = True
        contract['controls']['plane_event_front_revision'] = 'SHARED_MINIMUM_ORIENTED_GAP_V1'
    if controls.allow_multiple_shell_planes_dev:
        contract['controls']['allow_multiple_shell_planes_dev'] = True
    if controls.allow_guided_shell_pair_dev:
        contract['controls']['allow_guided_shell_pair_dev'] = True
        contract['controls']['guided_shell_pair_revision'] = 'FIXED_OVERLAP_PHYSICAL_TRANSLATIONS_V1'
    if controls.allow_fixed_projection_s3_target_dev:
        contract['controls']['allow_fixed_projection_s3_target_dev'] = True
        contract['controls']['fixed_projection_s3_revision'] = 'EXACT_NORMAL_MOTION_PHYSICAL_HERMITE_V1'
        contract['controls']['physical_sweep_maximum_panels'] = manager.settings.maximum_gap_triangles
        contract['controls']['physical_sweep_maximum_depth'] = manager.settings.maximum_swept_subdivisions
    if controls.allow_normal_s3_skin_hermite_dev:
        contract['controls']['allow_normal_s3_skin_hermite_dev'] = True
        contract['controls']['normal_s3_skin_revision'] = 'REFERENCE_P1_JOINT_PHASE_MATCHED_OVERLAP_ENDPOINT_LOCALIZATION_V3'
        if controls.normal_s3_polygon_pressure_bounds_dev:
            contract['controls']['normal_s3_polygon_pressure_bounds_dev']=True
            contract['controls']['normal_s3_skin_revision']='REFERENCE_P1_SIGNED_POLYGON_PHASE_MATCHED_OVERLAP_LOCALIZATION_V4'
        contract['controls']['physical_sweep_maximum_panels'] = manager.settings.maximum_gap_triangles
        contract['controls']['physical_sweep_maximum_depth'] = manager.settings.maximum_swept_subdivisions
        contract['controls']['hidden_physical_event_tolerance_m'] = controls.plane_event_gap_tolerance_m
        for name in ('plane_event_time_tolerance_s','plane_event_gap_tolerance_m',
                'maximum_plane_event_endpoint_evaluations','maximum_plane_event_stage_evaluations',
                'maximum_event_localizations'):
            contract['controls'][name]=getattr(controls,name)
        contract['controls']['pressure_sweep_maximum_cells']=manager.settings.maximum_gap_triangles
    if controls.allow_multiple_deformable_beam_s3_dev:
        contract['controls'].update(allow_multiple_deformable_beam_s3_dev=True,
            physical_path_policy='ALL_PAIR_COMPLETE_CUBIC_HULL_FIXED_DIRECTOR_V1',
            maximum_path_intervals=controls.maximum_beam_swept_subdivisions,
            maximum_path_depth=manager.settings.maximum_swept_subdivisions)
    if controls.allow_deformable_beam_s3_dev:
        contract['controls'].update(allow_deformable_beam_s3_dev=True,
            physical_path_policy='both_body_complete_cubic_hull_moving_normal_v1',
            maximum_path_intervals=controls.maximum_beam_swept_subdivisions,
            maximum_path_depth=manager.settings.maximum_swept_subdivisions)
    if _directional_selected(controls):
        contract['controls'].update(allow_directional_beam_s3_dev=True,
            directional_policy='EXACT_FULL_LINE_RATIONAL_WORK_V1',
            descriptor_rate_policy='DIRECTIONAL_VELOCITY_NULL_SECANT_ACCELERATION_V1',
            represented_path='COMPLETE_AFFINE_SKIN_ENDPOINT_HERMITE_V1',
            maximum_path_intervals=controls.maximum_beam_swept_subdivisions,
            maximum_path_depth=manager.settings.maximum_swept_subdivisions)
    if controls.allow_rotational_beam_s3_dev:
        contract['controls'].update(allow_rotational_beam_s3_dev=True,
            descriptor_rate_policy='TOTAL_TANGENT_VELOCITY_NULL_SECANT_ACCELERATION_V1',
            represented_path='COMPLETE_AFFINE_SKIN_ENDPOINT_HERMITE_V1',
            maximum_path_intervals=controls.maximum_beam_swept_subdivisions,
            maximum_path_depth=manager.settings.maximum_swept_subdivisions)
    if controls.allow_classical_beam_dev:
        contract["controls"].update(
            allow_classical_beam_dev=True,
            beam_swept_gap_tolerance_m=controls.beam_swept_gap_tolerance_m,
            maximum_beam_swept_subdivisions=controls.maximum_beam_swept_subdivisions,
        )
    if controls.allow_multiple_beam_planes_dev:
        contract["controls"]["allow_multiple_beam_planes_dev"] = True
    if controls.allow_free_sphere_dev:
        contract["controls"]["allow_free_sphere_dev"] = True
    if controls.allow_free_shell_sphere_dev:
        contract['controls'].update(allow_free_shell_sphere_dev=True,
            beam_swept_gap_tolerance_m=controls.beam_swept_gap_tolerance_m,
            maximum_beam_swept_subdivisions=controls.maximum_beam_swept_subdivisions)
    if tool_histories:
        contract["rigid_translation_histories"] = [
            {"body": h.body_name, "points": [list(point) for point in h.points]}
            for h in tool_histories]
        for name in ("tool_work_absolute_tolerance_j", "tool_work_relative_tolerance", "maximum_tool_work_subdivisions"):
            contract["controls"][name] = getattr(controls, name)
    if controls.allow_q4_plastic_dev:
        contract['controls'].update(allow_q4_plastic_dev=True,num_layers=3,
            save_q4_material_history_dev=controls.save_q4_material_history_dev)
    if controls.q4_plane_event_subdivision_dev:
        for name in ('q4_plane_event_subdivision_dev','plane_event_time_tolerance_s',
                'plane_event_gap_tolerance_m','maximum_plane_event_endpoint_evaluations',
                'maximum_plane_event_stage_evaluations','maximum_event_localizations'):
            contract['controls'][name]=getattr(controls,name)
    if controls.s3_algebraic_path_dev:
        contract["controls"]["s3_algebraic_path_dev"] = True
        contract["controls"]["s3_path_position_tolerance_m"] = controls.s3_path_position_tolerance_m
        contract["controls"]["s3_path_rotation_tolerance_rad"] = controls.s3_path_rotation_tolerance_rad
    if controls.s3_event_subdivision_dev:
        for name in ("s3_event_subdivision_dev", "s3_event_time_tolerance_s",
                     "s3_event_gap_tolerance_m", "maximum_event_localizations",
                     "maximum_event_root_iterations"):
            contract["controls"][name] = getattr(controls, name)
    if controls.allow_equilibrated_directional_beam_s3_dev:
        contract['controls'].update(allow_equilibrated_directional_beam_s3_dev=True,
            initial_policy='ELASTIC_COMPLETE_EQUILIBRIUM_CONSTANT_DEADLOAD_V1')
    return contract


def _new_restart_checkpoint(model_fingerprint, analysis_fingerprint,
                            current_time, q, velocity, acceleration,
                            accepted_steps, external_work_j=None, path_terminal_active=None,
                            event_boundary_support=None, tool_work_j=None, tool_work_error_j=None,
                            free_body_name=None,material_checkpoint=None,plane_event_state=None,
                            plane_pressure_features=None,guided_shell_pair=False,
                            fixed_projection_s3_target=False,normal_s3_skin_hermite=False,
                            normal_s3_polygon_pressure=False,beam_s3=False,rotational_beam_s3=False,
                            rotational_previous_velocity=None,rotational_last_dt=0.,directional_beam_s3=False,equilibrated_directional=False,multiple_beam_s3=False) -> dict:
    payload = {
        "schema": (_HISTORY_RESTART_SCHEMA if external_work_j is not None
                   else _RESTART_SCHEMA),
        "version": 2 if external_work_j is not None else 1,
        "model_fingerprint": model_fingerprint,
        "analysis_fingerprint": analysis_fingerprint,
        "state": {
            "time": float(current_time),
            "q": np.asarray(q).tolist(),
            "velocity": np.asarray(velocity).tolist(),
            "acceleration": np.asarray(acceleration).tolist(),
            "accepted_steps": int(accepted_steps),
        },
    }
    if external_work_j is not None:
        payload["state"]["external_work_j"] = float(external_work_j)
    if path_terminal_active is not None:
        payload["schema"] = _ALGEBRAIC_PATH_RESTART_SCHEMA
        payload["version"] = 3
        payload["state"]["path_terminal_active"] = list(path_terminal_active)
    if event_boundary_support is not None:
        payload["schema"] = _EVENT_RESTART_SCHEMA
        payload["version"] = 4
        payload["state"]["event_boundary_support"] = list(event_boundary_support)
    if tool_work_j is not None:
        payload["schema"], payload["version"] = _TOOL_RESTART_SCHEMA, 5
        payload["state"]["tool_work_j"] = dict(tool_work_j)
        payload["state"]["tool_work_error_j"] = dict(tool_work_error_j)
    if free_body_name is not None:
        payload["schema"], payload["version"] = _FREE_RESTART_SCHEMA, 6
        payload["state"]["free_rigid_body_names"] = [free_body_name]
    if material_checkpoint is not None:
        payload['schema'],payload['version']=_Q4_PLASTIC_RESTART_SCHEMA,7
        payload['state']['material_checkpoint']=deepcopy(material_checkpoint)
    if plane_event_state is not None:
        payload['schema'],payload['version']=((_Q4_PLASTIC_EVENT_RESTART_SCHEMA,9)
            if material_checkpoint is not None else (_Q4_EVENT_RESTART_SCHEMA,8))
        payload['state']['plane_event_state']=deepcopy(plane_event_state)
    if plane_pressure_features is not None:
        payload['schema'],payload['version']=_MULTIPLE_Q4_EVENT_RESTART_SCHEMA,10
        payload['state']['plane_pressure_features']=[list(feature) for feature in plane_pressure_features]
        if guided_shell_pair:
            payload['schema'],payload['version']=_GUIDED_SHELL_PAIR_RESTART_SCHEMA,11
        if fixed_projection_s3_target:
            payload['schema'],payload['version']=_FIXED_PROJECTION_S3_RESTART_SCHEMA,12
        if normal_s3_skin_hermite:
            payload['schema'],payload['version']=_NORMAL_S3_SKIN_RESTART_SCHEMA,15
            if normal_s3_polygon_pressure:
                payload['schema'],payload['version']=_NORMAL_S3_POLYGON_RESTART_SCHEMA,16
    if beam_s3:
        from .beam_s3_transient import RESTART_SCHEMA
        payload['schema'],payload['version']=RESTART_SCHEMA,17
    if rotational_beam_s3:
        from .beam_s3_rotational_transient import RESTART_SCHEMA
        payload['schema'],payload['version']=RESTART_SCHEMA,18
        payload['state']['previous_velocity']=np.asarray(rotational_previous_velocity).tolist()
        payload['state']['last_dt']=float(rotational_last_dt)
    if directional_beam_s3:
        from .beam_s3_directional_rates import RESTART_SCHEMA
        payload['schema'],payload['version']=RESTART_SCHEMA,19
    if equilibrated_directional:
        from .beam_s3_directional_rates import EQUILIBRATED_RESTART_SCHEMA
        payload['schema'],payload['version']=EQUILIBRATED_RESTART_SCHEMA,20
    if multiple_beam_s3:
        from .multiple_beam_s3_transient import RESTART_SCHEMA
        payload['schema'],payload['version']=RESTART_SCHEMA,21
    payload["checkpoint_sha256"] = analysis_configuration_fingerprint(payload)
    return payload


def _plane_event_state(reference, gradient, position, velocity, gap_tolerance):
    """Numerical boundary evidence; it never changes the pressure law."""
    gaps=np.asarray(reference+gradient@position).reshape(-1)
    rates=np.asarray(gradient@velocity).reshape(-1)
    phase=np.zeros(gaps.size,dtype=int)
    phase[gaps < -gap_tolerance]=-1
    phase[gaps > gap_tolerance]=1
    boundary=np.abs(gaps)<=gap_tolerance
    phase[boundary & (rates<0.)]=-1
    phase[boundary & (rates>0.)]=1
    return dict(pressure_dofs=int(gaps.size),
        support_indices=np.flatnonzero(boundary).tolist(),
        normalized_gaps_m=gaps.tolist(),gap_rates_m_s=rates.tolist(),
        departing_phase=phase.tolist())


def _validate_plane_event_state(value,expected):
    if (type(value) is not dict or set(value)!=set(expected)
            or type(value['pressure_dofs']) is not int
            or value['pressure_dofs']!=expected['pressure_dofs']):
        raise ValueError('transient plane event checkpoint boundary is malformed')
    for name in ('support_indices','departing_phase'):
        if (type(value[name]) is not list or any(type(i) is not int for i in value[name])
                or value[name]!=expected[name]):
            raise ValueError('transient plane event checkpoint phase/support is incompatible')
    for name in ('normalized_gaps_m','gap_rates_m_s'):
        if (type(value[name]) is not list
                or any(type(i) not in (int,float) or not np.isfinite(i) for i in value[name])
                or not np.array_equal(value[name],expected[name])):
            raise ValueError('transient plane event checkpoint gaps/rates are incompatible')


def _read_restart_checkpoint(value, model_fingerprint, analysis_fingerprint,
                             reduced_size, t_end, *, load_history, path_pressure_dofs=0,
                             event_subdivision=False, tool_body_names=(), free_body_name=None,
                             q4_material_state=False,plane_event_state=False,
                             plane_pressure_features=None,guided_shell_pair=False,
                             fixed_projection_s3_target=False,normal_s3_skin_hermite=False,
                             normal_s3_polygon_pressure=False,beam_s3=False,rotational_beam_s3=False,directional_beam_s3=False,equilibrated_directional=False,multiple_beam_s3=False):
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "version", "model_fingerprint", "analysis_fingerprint",
        "state", "checkpoint_sha256",
    }:
        raise ValueError("transient contact checkpoint schema is incompatible")
    payload = dict(value)
    signature = payload.pop("checkpoint_sha256")
    if (type(signature) is not str
            or signature != analysis_configuration_fingerprint(payload)):
        raise ValueError("transient contact checkpoint integrity mismatch")
    expected_version = 2 if load_history is not None else 1
    expected_schema = (_HISTORY_RESTART_SCHEMA if load_history is not None
                       else _RESTART_SCHEMA)
    if path_pressure_dofs:
        expected_version, expected_schema = 3, _ALGEBRAIC_PATH_RESTART_SCHEMA
    if event_subdivision:
        expected_version, expected_schema = 4, _EVENT_RESTART_SCHEMA
    if tool_body_names:
        expected_version, expected_schema = 5, _TOOL_RESTART_SCHEMA
    if free_body_name is not None:
        expected_version, expected_schema = 6, _FREE_RESTART_SCHEMA
    if q4_material_state:
        expected_version,expected_schema=7,_Q4_PLASTIC_RESTART_SCHEMA
    if plane_event_state:
        expected_version,expected_schema=((9,_Q4_PLASTIC_EVENT_RESTART_SCHEMA)
            if q4_material_state else (8,_Q4_EVENT_RESTART_SCHEMA))
    if plane_pressure_features is not None:
        expected_version,expected_schema=10,_MULTIPLE_Q4_EVENT_RESTART_SCHEMA
        if guided_shell_pair:
            expected_version,expected_schema=11,_GUIDED_SHELL_PAIR_RESTART_SCHEMA
        if fixed_projection_s3_target:
            expected_version,expected_schema=12,_FIXED_PROJECTION_S3_RESTART_SCHEMA
        if normal_s3_skin_hermite:
            expected_version,expected_schema=15,_NORMAL_S3_SKIN_RESTART_SCHEMA
            if normal_s3_polygon_pressure:
                expected_version,expected_schema=16,_NORMAL_S3_POLYGON_RESTART_SCHEMA
    if beam_s3:
        from .beam_s3_transient import RESTART_SCHEMA
        expected_schema,expected_version=RESTART_SCHEMA,17
    if rotational_beam_s3:
        from .beam_s3_rotational_transient import RESTART_SCHEMA
        expected_schema,expected_version=RESTART_SCHEMA,18
    if directional_beam_s3:
        from .beam_s3_directional_rates import RESTART_SCHEMA
        expected_schema,expected_version=RESTART_SCHEMA,19
    if equilibrated_directional:
        from .beam_s3_directional_rates import EQUILIBRATED_RESTART_SCHEMA
        expected_schema,expected_version=EQUILIBRATED_RESTART_SCHEMA,20
    if multiple_beam_s3:
        from .multiple_beam_s3_transient import RESTART_SCHEMA
        expected_schema,expected_version=RESTART_SCHEMA,21
    if payload["schema"] != expected_schema or payload["version"] != expected_version:
        raise ValueError("transient contact checkpoint version is incompatible")
    if payload["model_fingerprint"] != model_fingerprint:
        raise ValueError("transient contact checkpoint model is incompatible")
    if payload["analysis_fingerprint"] != analysis_fingerprint:
        raise ValueError("transient contact checkpoint analysis is incompatible")
    state = payload["state"]
    expected_state_keys = {
        "time", "q", "velocity", "acceleration", "accepted_steps",
    }
    if rotational_beam_s3:
        expected_state_keys.update(('previous_velocity','last_dt'))
    if load_history is not None:
        expected_state_keys.add("external_work_j")
    if path_pressure_dofs:
        expected_state_keys.add("path_terminal_active")
    if event_subdivision:
        expected_state_keys.add("event_boundary_support")
    if tool_body_names:
        expected_state_keys.update(("tool_work_j", "tool_work_error_j"))
    if free_body_name is not None:
        expected_state_keys.add("free_rigid_body_names")
    if q4_material_state:
        expected_state_keys.add('material_checkpoint')
    if plane_event_state:
        expected_state_keys.add('plane_event_state')
    if plane_pressure_features is not None:
        expected_state_keys.add('plane_pressure_features')
    if not isinstance(state, Mapping) or set(state) != expected_state_keys:
        raise ValueError("transient contact checkpoint state is malformed")
    if plane_pressure_features is not None:
        features=state['plane_pressure_features']
        if (type(features) is not list
                or any(type(feature) is not list or len(feature)!=2
                    or type(feature[0]) is not str or type(feature[1]) is not int
                    for feature in features)
                or features != [list(feature) for feature in plane_pressure_features]):
            raise ValueError('transient contact checkpoint pressure-feature identity is incompatible')
    if free_body_name is not None and state["free_rigid_body_names"] != [free_body_name]:
        raise ValueError("transient contact checkpoint free body identity is invalid")
    time = float(state["time"])
    steps = state["accepted_steps"]
    if (not np.isfinite(time) or time <= 0.0 or time >= t_end
            or type(steps) is not int or steps < 1):
        raise ValueError("transient contact checkpoint time or step is invalid")
    arrays = []
    for name in ("q", "velocity", "acceleration"):
        array = np.asarray(state[name], dtype=float)
        if array.shape != (reduced_size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"transient contact checkpoint {name} is invalid")
        arrays.append(array.copy())
    work = float(state["external_work_j"]) if load_history is not None else None
    if work is not None and not np.isfinite(work):
        raise ValueError("transient contact checkpoint external work is invalid")
    terminal = None
    if path_pressure_dofs:
        terminal = state["path_terminal_active"]
        if (type(terminal) is not list or len(terminal) != path_pressure_dofs
                or any(type(flag) is not bool for flag in terminal)):
            raise ValueError("transient contact checkpoint path activity is invalid")
        terminal = tuple(terminal)
    values = (time, steps, *arrays, work, terminal)
    if rotational_beam_s3:
        if type(state['previous_velocity']) is not list or any(type(x) not in (int,float) or not np.isfinite(x) for x in state['previous_velocity']):
            raise ValueError('rotational checkpoint previous velocity is malformed')
        previous=np.asarray(state['previous_velocity'],dtype=float)
        duration=state['last_dt']
        if (previous.shape!=(reduced_size,) or not np.isfinite(previous).all()
                or type(duration) not in (int,float) or not np.isfinite(duration)
                or not 0.<duration<=time):
            raise ValueError('rotational checkpoint secant history is malformed')
        values+=(previous.copy(),float(duration))
    if event_subdivision:
        boundary = state["event_boundary_support"]
        if (type(boundary) is not list or any(type(i) is not int
                or not 0 <= i < path_pressure_dofs for i in boundary)
                or boundary != sorted(set(boundary))):
            raise ValueError("transient contact checkpoint event boundary is invalid")
        values += (tuple(boundary),)
    if tool_body_names:
        for key in ("tool_work_j", "tool_work_error_j"):
            data = state[key]
            if (not isinstance(data, Mapping) or set(data) != set(tool_body_names)
                    or any(isinstance(v, (bool, np.bool_))
                           or not isinstance(v, (int, float, np.integer, np.floating))
                           or not np.isfinite(v) or (key == "tool_work_error_j" and v < 0.)
                           for v in data.values())):
                raise ValueError("transient contact checkpoint tool work is invalid")
            values += (dict(data),)
    if q4_material_state:
        values+=(state['material_checkpoint'],)
    if plane_event_state:
        values+=(state['plane_event_state'],)
    return values


def _analytic_vk_curvature(model, full_position, full_velocity):
    """Exact F''(x)[v,v] when every element is an elastic qualified S3 V2D shell.

    The von Karman force is cubic in the displacements, so the central finite
    difference of the tangent is exact in exact arithmetic but carries
    cancellation noise; the closed form removes it. None means the model has
    another element type or a non-elastic section: the finite difference stays.
    """
    mesh = getattr(model, "mesh", None)
    if mesh is None or not hasattr(mesh, "elements"):
        return None  # no assembled model (e.g. manufactured structures): keep the finite difference
    elements = tuple(mesh.elements.values())
    if not elements or any(type(element) is not NativeParityE4PLS3V2DShellElement
                           for element in elements):
        return None
    total = np.zeros(full_position.shape[0], dtype=float)
    for element in elements:
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        part = element.von_karman_directional_curvature(
            model.mesh, model.get_material(element.material_name),
            full_position[dofs], full_velocity[dofs],
        )
        if part is None:
            return None
        np.add.at(total, dofs, part)
    return total


def _consistent_s3_descriptor_rates(
    model, T, Mred, basis, full_position, reduced_velocity,
    reduced_acceleration, full_tangent, committed_states, difference_step_s,
    *, one_sided_manager=None, maximum_iterations=20,
    event_touch_indices=(), event_gap_tolerance=None,
):
    """Recover massless rates from the current nonlinear algebraic manifold.

    The physical mass equation fixes rates modulo the certified null basis Z.
    For G(q)=Z.T*F(q)=0, smooth-branch rates satisfy
    Z.T*K(q)*v=0 and Z.T*(K(q)*a+F''(q)[v,v])=0. The contact-plane
    penalty tangent is constant on a fixed active branch, so the directional
    second derivative comes only from the smooth structural force here.
    """
    Z = basis.reduced_basis.toarray()
    gram = Z.T @ Z
    K = (T.T @ full_tangent @ T).toarray()
    algebraic = Z.T @ K @ Z
    if one_sided_manager is not None:
        manager = one_sided_manager
        _, contact_tangent, _ = manager.evaluate(full_position, tangent=True)
        structural = K-(T.T @ contact_tangent @ T).toarray()
        B = (sparse.diags(1./manager.support) @ manager.gap_gradient @ T).toarray()
        shift = -float((manager.point-manager.reference_point) @ manager.normal)
        reference_gap = manager.reference_weighted_gap/manager.support+shift
        normalized_gap = (reference_gap
                          +np.asarray(manager.gap_gradient @ full_position).reshape(-1)/manager.support)
        # Tie budget follows arithmetic operands, not the physical gap budget.
        operand_scale = (np.abs(reference_gap)
                         +np.asarray(abs(manager.gap_gradient) @ np.abs(full_position)).reshape(-1)/manager.support)
        touch = np.abs(normalized_gap) <= 64*np.finfo(float).eps*operand_scale
        if event_touch_indices:
            indices = list(event_touch_indices)
            if (event_gap_tolerance is None
                    or np.any(np.abs(normalized_gap[indices]) > event_gap_tolerance)):
                raise ValueError("resolved S3 event touch exceeds its gap budget")
            touch[indices] = True
        closed = (normalized_gap < 0.) & ~touch
        weights = manager.stiffness*manager.support

    def physical_part(value):
        return value - Z @ np.linalg.solve(gram, Z.T @ value)

    try:
        if one_sided_manager is None:
            physical_velocity = physical_part(reduced_velocity)
            velocity = physical_velocity + Z @ np.linalg.solve(
                algebraic, -Z.T @ K @ physical_velocity,
            )
        else:
            velocity = directional_descriptor_rate(
                structural, Z, reduced_velocity, B, weights, closed, touch,
                maximum_iterations=maximum_iterations,
            )
    except np.linalg.LinAlgError as exc:
        raise ValueError("S3 descriptor algebraic velocity solve is singular") from exc
    full_velocity = np.asarray(T @ velocity).reshape(-1)
    analytic = _analytic_vk_curvature(model, full_position, full_velocity)
    if analytic is not None:
        structural_curvature = np.asarray(T.T @ analytic).reshape(-1)
    else:
        h = float(difference_step_s)
        with suspend_contact():
            _, plus, _ = _assemble_nonlinear_system(
                model, full_position+h*full_velocity, committed_states, 3,
                tangent=True, kinematics="von_karman",
            )
            discard_active_state_candidate(committed_states)
            _, minus, _ = _assemble_nonlinear_system(
                model, full_position-h*full_velocity, committed_states, 3,
                tangent=True, kinematics="von_karman",
            )
            discard_active_state_candidate(committed_states)
        structural_curvature = np.asarray(
            T.T @ ((plus-minus) @ full_velocity)
        ).reshape(-1)/(2.0*h)
    try:
        if one_sided_manager is None:
            physical_acceleration = physical_part(reduced_acceleration)
            acceleration = physical_acceleration + Z @ np.linalg.solve(
                algebraic,
                -Z.T @ (K @ physical_acceleration+structural_curvature),
            )
        else:
            gap_rate = B @ velocity
            rate_roundoff = 64*np.finfo(float).eps*(np.abs(B) @ np.abs(velocity))
            stationary_touch = touch & (np.abs(gap_rate) <= rate_roundoff)
            closing_touch = touch & (gap_rate < -rate_roundoff)
            acceleration = directional_descriptor_rate(
                structural, Z, reduced_acceleration, B, weights,
                closed | closing_touch, stationary_touch,
                offset=structural_curvature, maximum_iterations=maximum_iterations,
            )
    except np.linalg.LinAlgError as exc:
        raise ValueError("S3 descriptor algebraic acceleration solve is singular") from exc
    if (not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(acceleration))
            or np.linalg.norm(Mred @ (acceleration-reduced_acceleration))
            > 1.0e-8*max(1.0, np.linalg.norm(Mred @ reduced_acceleration))):
        raise ValueError("S3 descriptor rate recovery changed physical inertia")
    return velocity, acceleration


_CHEBYSHEV_7 = .5*(1.-np.cos((2*np.arange(7)+1)*np.pi/14.))


def _lebesgue_constant(nodes):
    grid = np.linspace(0., 1., 4001)
    basis = np.ones((nodes.size, grid.size))
    for i, node in enumerate(nodes):
        for j, other in enumerate(nodes):
            if i != j:
                basis[i] *= (grid-other)/(node-other)
    return float(np.max(np.sum(np.abs(basis), axis=0)))*(1.+1e-6)


_LEBESGUE_7 = _lebesgue_constant(_CHEBYSHEV_7)


@dataclass(frozen=True)
class _S3AffinePathAdapter:
    structural: np.ndarray
    null_basis: np.ndarray
    gradient: np.ndarray
    reference_gap: np.ndarray
    weights: np.ndarray
    force_offset: np.ndarray
    corner_reference: np.ndarray
    corner_gradient: np.ndarray
    model: object = field(default=None, compare=False, repr=False)
    transform: object = field(default=None, compare=False, repr=False)
    committed_states: object = field(default=None, compare=False, repr=False)
    memo: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def prepare(cls, model, T, basis, manager, committed_states):
        # The admitted exact elastic S3 local operator is reference-linear.
        # This would be a tangent approximation for a generic nonlinear owner;
        # _admit deliberately does not permit that substitution.
        with suspend_contact():
            force, stiffness, _ = _assemble_nonlinear_system(
                model, np.zeros(T.shape[0]), committed_states, 3,
                tangent=True, kinematics="von_karman",
            )
            discard_active_state_candidate(committed_states)
        structural = (T.T @ stiffness @ T).toarray()
        Z = basis.reduced_basis.toarray()
        try:
            np.linalg.cholesky(Z.T @ structural @ Z)
        except np.linalg.LinAlgError as exc:
            raise ValueError("S3 affine path requires positive definite algebraic work") from exc
        B = (sparse.diags(1./manager.support) @ manager.gap_gradient @ T).toarray()
        shift = -float((manager.point-manager.reference_point) @ manager.normal)
        g = manager.reference_weighted_gap/manager.support+shift
        corners, gradients = [], []
        for reference, dofs, _shape, skin_offset, gradient in manager.patches:
            corners.extend((reference+skin_offset-manager.point) @ manager.normal)
            for corner_dofs in dofs:
                full_gradient = np.zeros(T.shape[0])
                full_gradient[corner_dofs] = gradient
                gradients.append(np.asarray(full_gradient @ T).reshape(-1))
        return cls(structural, Z, B, g, manager.stiffness*manager.support,
                   np.asarray(T.T @ force).reshape(-1), np.asarray(corners), np.asarray(gradients),
                   model, T, committed_states)

    def nonlinear_part(self, position):
        """Von Karman excess of the structural force over the reference-linear one.

        Exactly zero (not roundoff) when the response is reference-linear to
        within 512 epsilon of the force scale, so linear models are unchanged.
        """
        with suspend_contact():
            force, _, _ = _assemble_nonlinear_system(
                self.model, np.asarray(self.transform @ position).reshape(-1),
                self.committed_states, 3, tangent=False, kinematics="von_karman",
            )
            discard_active_state_candidate(self.committed_states)
        actual = np.asarray(self.transform.T @ force).reshape(-1)
        linear = self.structural @ position+self.force_offset
        excess = actual-linear
        noise = 512*np.finfo(float).eps*max(1., float(np.linalg.norm(actual)),
                                            float(np.linalg.norm(linear)))
        return excess if float(np.max(np.abs(excess), initial=0.)) > noise else np.zeros_like(excess)

    def _path(self, coefficients, load, force):
        rate = curvature = None
        if force is not None:
            rate, curvature = -force[1], -force[2]
        return affine_descriptor_path(
            self.structural, self.null_basis, coefficients, self.gradient,
            self.reference_gap, self.weights,
            force=load-self.force_offset-(0. if force is None else force[0]),
            force_rate=rate, force_curvature=curvature,
        )

    def chord(self, start, end):
        """Von Karman excess force at both endpoints (start, end); None when linear."""
        key = (np.asarray(start).tobytes(), np.asarray(end).tobytes())
        if key not in self.memo:
            first, last = self.nonlinear_part(start), self.nonlinear_part(end)
            self.memo.clear()
            self.memo[key] = None if not (first.any() or last.any()) else (first, last)
        return self.memo[key]

    def evaluate(self, start, end, velocity, dt, load):
        ends = self.chord(start, end)
        self.memo["force"] = None
        coefficients = np.stack((start, dt*velocity, end-start-dt*velocity))
        path = self._path(coefficients, load, None)
        if ends is None:
            # Zero endpoint excess does not make the path linear: along the lifted path the
            # excess is a polynomial of degree <= 6 in s (cubic force, quadratic path), so zero
            # at both ends and at five further distinct samples means identically zero.
            for s in (1./6., 1./3., .5, 2./3., 5./6.):
                segment = next(g for g in path.segments if g.start <= s <= g.end)
                if self.nonlinear_part(segment.position(s)).any():
                    ends = (np.zeros(start.shape[0]), np.zeros(start.shape[0]))
                    break
        if ends is not None:
            # Von Karman: the algebraic balance carries the excess force as a
            # polynomial in the path parameter fixed by its exact values at s=0,
            # 1 and at the lifted midpoint; two fixed-point passes refine the
            # midpoint. check_interior then certifies the result.
            force = self.force_model(ends, ends[0]+.5*(ends[1]-ends[0]))
            for _ in range(2):
                path = self._path(coefficients, load, force)
                segment = next(g for g in path.segments if g.start <= .5 <= g.end)
                force = self.force_model(ends, self.nonlinear_part(segment.position(.5)))
            path = self._path(coefficients, load, force)
            self.memo["force"] = force
            self.memo["load"] = load
        return path, path.minimum_gap(self.corner_reference, self.corner_gradient)

    @staticmethod
    def force_model(ends, middle):
        """(c0, c1, c2) of the quadratic through the excess force at s=0, 1/2, 1."""
        first, last = ends
        curvature = 2.*(last+first-2.*middle)
        return first, last-first-curvature, curvature

    def check_interior(self, path, controls):
        """Certified position error of the lifted path against the true von Karman force.

        On each segment the error e(s) = A^-1 Z'(F_vk(x(s)) - model(s)) (A = Z'(K+B'WB)Z of the
        active branch) is a vector polynomial of degree <= 6 in s (cubic force, quadratic path,
        quadratic model), hence equals its degree-6 interpolant on 7 Chebyshev nodes and
        |e(s)| <= Lambda_7 * max_i |e(s_i)| with Lambda_7 the Lebesgue constant of the nodes
        (_LEBESGUE_7). That certified bound, mapped through Z, must stay within the path
        position budget.
        """
        force = self.memo["force"]
        for segment in path.segments:
            active = np.asarray(segment.active, dtype=bool)
            tangent = self.structural+self.gradient[active].T @ (self.weights[active, None]*self.gradient[active])
            algebraic = self.null_basis.T @ tangent @ self.null_basis
            worst = 0.
            for node in _CHEBYSHEV_7:
                s = segment.start+(segment.end-segment.start)*node
                exact = self.nonlinear_part(segment.position(s))
                model = force[0]+s*(force[1]+s*force[2])
                worst = max(worst, float(np.linalg.norm(self.null_basis @ np.linalg.solve(
                    algebraic, self.null_basis.T @ (exact-model)))))
            if _LEBESGUE_7*worst > controls.s3_path_position_tolerance_m:
                raise ValueError("S3 lifted path interior disagrees with von Karman work")

    def check_endpoints(self, path, start, end, transform, controls, scale=1.):
        """Strict endpoint budgets on the raw mismatch between lifted and committed endpoints.

        No residual- or tolerance-derived allowance is added to a budget. For a von Karman step
        the committed endpoint must additionally be a converged equilibrium of the model balance:
        |Z'F(actual)| <= equilibrium_tolerance*scale, else the step is refused.
        """
        force = self.memo.get("force")
        for side, segment, fraction, actual in ((0, path.segments[0], 0., start),
                                                (1, path.segments[-1], 1., end)):
            if force is not None:
                c = force[0] if side == 0 else force[0]+force[1]+force[2]
                r = self.null_basis.T @ self.endpoint_force(actual, self.memo["load"], c)
                if float(np.linalg.norm(r)) > controls.equilibrium_tolerance*scale:
                    raise ValueError("S3 lifted path endpoint residual exceeds the Newton tolerance")
            error = np.asarray(transform @ (segment.position(fraction)-actual)).reshape(-1, 6)
            if (np.max(np.abs(error[:, :3])) > controls.s3_path_position_tolerance_m
                    or np.max(np.abs(error[:, 3:])) > controls.s3_path_rotation_tolerance_rad):
                raise ValueError("S3 lifted path endpoint mismatch exceeds its numerical budget")

    def endpoint_force(self, position, load, chord_force=None):
        gap = self.reference_gap+self.gradient @ position
        force = (self.structural @ position+self.force_offset-load
                 +self.gradient.T @ (self.weights*np.minimum(gap, 0.)))
        return force if chord_force is None else force+chord_force


def _accepted_path_transition_times(previous_active, path, start_time, step):
    times = []
    if previous_active is not None and previous_active != path.segments[0].active:
        times.append(float(start_time))
    times.extend(float(start_time+step*s) for s in path.transition_fractions)
    return tuple(times)


def _event_continuation_segments(path, step, boundary_support, start_gaps,
                                 time_tolerance, gap_tolerance):
    """Coalesce only contiguous transitions of the committed numerical boundary.

    The original path remains authoritative for force/work and interval minima.
    This changes which transition still requires a new physical event step.
    """
    index = 0
    resolved = frozenset(boundary_support)
    while resolved and index+1 < len(path.segments):
        left,right = path.segments[index:index+2]
        changed = tuple(i for i,(a,b) in enumerate(zip(left.active,right.active)) if a != b)
        if (left.end*step > time_tolerance or not changed
                or any(i not in resolved for i in changed)
                or np.any(np.abs(start_gaps[list(changed)]) > gap_tolerance)):
            break
        index += 1
    return path.segments[index:]


def _minimum_hermite_gap(
    manager: _PlanarQ4PlaneManager,
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    dt: float,
) -> float:
    """Exact corner minimum on the declared cubic step path.

    Q4 skin gap against a fixed plane is bilinear over each face and cubic
    along the Hermite path defined by Newmark endpoint positions/velocities.
    Its face minimum is attained at a corner for each path parameter.
    """

    minimum = float("inf")
    for reference, dofs, _shape, skin_offset, gradient in manager.patches:
        base = (reference + skin_offset - manager.point) @ manager.normal
        g0 = base + start[dofs] @ gradient
        g1 = base + end[dofs] @ gradient
        dg0 = dt * (start_velocity[dofs] @ gradient)
        dg1 = dt * (end_velocity[dofs] @ gradient)
        a = 2.0 * g0 - 2.0 * g1 + dg0 + dg1
        b = -3.0 * g0 + 3.0 * g1 - 2.0 * dg0 - dg1
        c = dg0
        for i in range(len(g0)):
            values = [float(g0[i]), float(g1[i])]
            roots = np.roots([3.0 * a[i], 2.0 * b[i], c[i]])
            for root in roots:
                if abs(root.imag) <= 1.0e-12 and 0.0 < root.real < 1.0:
                    s = float(root.real)
                    values.append(float(((a[i] * s + b[i]) * s + c[i]) * s + g0[i]))
            minimum = min(minimum, *values)
    return minimum


def _multiple_shell_plane_path_gaps(manager,start,end,start_velocity,end_velocity,dt,check_cancel):
    """Outward corner bounds for every pair's represented complete cubic path.

    Each fixed planar Q4 gap is bilinear in its face coordinates, so bounding
    every corner over time bounds the whole represented face. Upstream floating
    gap/rate arithmetic is distinct from the exact supplied-float cubic bounds.
    """
    from .plane_newmark_events import hermite_branch_extrema
    gaps={}
    admissible=True
    for name,child in manager.pairs:
        minimum=float('inf')
        for reference,dofs,_shape,skin_offset,gradient in child.patches:
            check_cancel()
            base=(reference+skin_offset-child.point)@child.normal
            bounds=hermite_branch_extrema(base+start[dofs]@gradient,
                base+end[dofs]@gradient,start_velocity[dofs]@gradient,
                end_velocity[dofs]@gradient,dt)
            minimum=min(minimum,float(np.min(bounds.minimum_gaps)))
        gaps[name]=minimum
        admissible=admissible and minimum>=-child.gap_tolerance
    return min(gaps.values()),admissible,gaps


def _free_shell_sphere_path_on_selected_side(manager,start,end,start_velocity,
                                             end_velocity,dt,maximum_subdivisions,
                                             check_cancel):
    """Certify the radial owner's centroid-side domain over the full cubic path.

    The admitted shell has fixed directors and zero rotational coordinates.
    Exact binary-input Bernstein controls enclose each face's signed centre
    separation. Negative/zero feasible values reject the path; unresolved
    positivity fails explicitly. Subdivisions share the distance-bound budget.
    """
    from .beam_temporal_bounds import _fraction,_split_curve,BeamHermiteGapBoundsError
    sign=1 if manager.child.side=='top' else -1
    step=_fraction(dt)
    count=0
    for face in manager.child.faces:
        gaps=[];rates=[]
        for state,velocity in ((start,start_velocity),(end,end_velocity)):
            gap=rate=0
            for axis,director in enumerate(face.director):
                d=_fraction(director)
                centre=_fraction(float(manager.reference_point[axis]+state[-3+axis]))
                position=sum((_fraction(face.reference[i,axis])
                    +_fraction(state[face.dofs[i,axis]]) for i in range(4)),start=0)/4
                position+=_fraction(face.offset)*d
                gap+=sign*d*(centre-position)
                shell_rate=sum((_fraction(velocity[face.dofs[i,axis]])
                                for i in range(4)),start=0)/4
                rate+=sign*d*(_fraction(velocity[-3+axis])-shell_rate)
            gaps.append(gap);rates.append(rate)
        stack=[(gaps[0],gaps[0]+step*rates[0]/3,
                gaps[1]-step*rates[1]/3,gaps[1])]
        while stack:
            check_cancel('free shell sphere complete-path physical side')
            controls=stack.pop()
            if min(controls)>0:continue
            left,right=_split_curve(controls)
            if min(controls[0],controls[-1],left[-1])<=0:return False,count
            if count>=maximum_subdivisions:
                raise BeamHermiteGapBoundsError('free shell sphere selected-side path unresolved')
            stack.extend((left,right));count+=1
    return True,count


def _directional_selected(controls):
    return controls.allow_directional_beam_s3_dev or controls.allow_equilibrated_directional_beam_s3_dev


def _admit_free_shell_coordinates(model, transformation, prescribed):
    """Guided Q4 profile: fixed in-plane geometry and zero shell rotations.

    Independent normal nodal coordinates remain deformable. Their cubic path
    cannot invert its projection onto the planar reference surface. This is a
    physical work-map check of the actual constraint transformation.
    """
    for element in model.mesh.elements.values():
        reference=np.asarray(element.get_node_coordinates(model.mesh),dtype=float)
        normal=np.cross(reference[1]-reference[0],reference[3]-reference[0])
        length=np.linalg.norm(normal)
        if length<=0:raise ValueError('free shell sphere requires nondegenerate planar Q4 reference')
        normal/=length
        padding=64*np.finfo(float).eps*max(np.linalg.norm(reference-reference[0]),np.finfo(float).tiny)
        if np.max(np.abs((reference-reference[0])@normal))>padding:
            raise ValueError('free shell sphere requires planar Q4 reference geometry')
        projector=np.eye(3)-np.outer(normal,normal)
        for node in element.node_ids:
            dofs=np.asarray(model.mesh.dof_manager.get_node_dofs(node),dtype=int)
            if transformation[dofs[3:6]].nnz or np.any(prescribed[dofs[3:6]]):
                raise ValueError('free shell sphere requires constrained zero shell rotations')
            work=transformation[dofs[:3]].toarray()
            column_scale=np.max(np.abs(work),axis=0)
            active=column_scale>0
            normalized=work[:,active]/column_scale[active]
            if np.any(np.linalg.norm(projector@normalized,axis=0)>64*np.finfo(float).eps):
                raise ValueError('free shell sphere requires constrained in-plane shell motion')
            if np.linalg.norm(projector@prescribed[dofs[:3]])>padding:
                raise ValueError('free shell sphere requires zero prescribed in-plane shell displacement')


def _admit(model, definition: ContactDefinition, settings: ContactSolverSettings,
           controls: ContactTransientSolverControls) -> str:
    if (type(definition) is ContactDefinition and type(settings) is ContactSolverSettings
            and settings.finite_frame_seed != 'common_reference_frame_dev'):
        raise ValueError('finite_frame_seed requires contact_path_program; unsupported in transient contact')
    if (type(definition) is ContactDefinition and type(settings) is ContactSolverSettings
            and settings.shell_skin_kinematics != "reference_linear"):
        raise ValueError("shell_skin_kinematics requires contact_path_program; unsupported in transient contact")
    # Every transient route, including direct multi-beam admission, must
    # reject the static-only selector before inventory/manager allocation.
    # Preserve each existing wrong-type admission contract.
    if (type(definition) is ContactDefinition and type(settings) is ContactSolverSettings
            and settings.path_predictor != "none"):
        raise ValueError("path_predictor requires contact_path_program; unsupported in transient contact")
    if (type(definition) is ContactDefinition and type(settings) is ContactSolverSettings
            and settings.general_s3_differentiation == "jet2_q4_q4_dev"):
        raise ValueError("jet2_q4_q4_dev is unsupported in transient contact")
    if (type(definition) is ContactDefinition and type(settings) is ContactSolverSettings
            and settings.shell_pair_orientation != "legacy"):
        raise ValueError("shell_pair_orientation requires contact_path_program; unsupported in transient contact")
    if controls.allow_multiple_deformable_beam_s3_dev:
        from .multiple_beam_s3_transient import pair_definitions,PROFILE
        pair_definitions(model,definition,settings)
        return PROFILE
    if controls.allow_deformable_beam_s3_dev or controls.allow_rotational_beam_s3_dev or _directional_selected(controls):
        from .beam_s3_transient import PROFILE
        if controls.allow_rotational_beam_s3_dev:
            from .beam_s3_rotational_transient import PROFILE
        if _directional_selected(controls):
            from .beam_s3_directional_rates import PROFILE
            if controls.allow_equilibrated_directional_beam_s3_dev:
                from .beam_s3_directional_rates import EQUILIBRATED_PROFILE as PROFILE
        from .deformable_beam_s3_static import PROFILE as STATIC_PROFILE
        report=validate_general_contact(model,definition,settings)
        elements=tuple(model.mesh.elements.values())
        if (not report.valid or report.profile!=STATIC_PROFILE or len(elements)!=2 or
                set(eid for surface in definition.surfaces for eid in surface.element_ids)!=set(model.mesh.elements)):
            raise ValueError('beam-S3 impact requires one explicitly selected elastic B2/B3 and S3 interior')
        for element in elements:
            material=model.get_material(element.material_name)
            if type(material) is not IsotropicMaterial or material.yield_stress!=0. or material.hardening_curve is not None:
                raise ValueError('beam-S3 impact requires exact isotropic elastic material')
        return PROFILE
    if controls.allow_fixed_projection_s3_target_dev or controls.allow_normal_s3_skin_hermite_dev:
        if (settings.pair_motion != 'general' or settings.warped_q4_plane
                or settings.normal_discretization != 'primal_mortar'
                or settings.general_s3_differentiation != 'jet2_s3_dev'
                or len(definition.surfaces) != 2 or definition.rigid_bodies
                or any(surface.beam_radius is not None for surface in definition.surfaces)
                or len(definition.interactions) != 1
                or definition.interactions[0].normal_law != 'projected_penalty'):
            raise ValueError('fixed-projection S3 impact requires two explicit general Jet2 surfaces and one projected-penalty pair')
        elements = tuple(model.mesh.elements.values())
        if (not elements or any(type(element) is not NativeParityE4PLS3V2DShellElement
                                for element in elements)
                or set(eid for surface in definition.surfaces for eid in surface.element_ids)
                    != set(model.mesh.elements)):
            raise ValueError('fixed-projection S3 impact requires exact qualified S3 V2D on the declared surfaces')
        for element in elements:
            material = model.get_material(element.material_name)
            if (type(material) is not IsotropicMaterial or material.yield_stress != 0.
                    or material.hardening_curve is not None):
                raise ValueError('fixed-projection S3 impact admits only isotropic elastic material')
        report = validate_general_contact(model, definition, settings)
        if not report.valid:
            raise ValueError('unsupported fixed-projection S3 impact: ' + '; '.join(report.issues))
        return ((_NORMAL_S3_POLYGON_PROFILE if controls.normal_s3_polygon_pressure_bounds_dev
                 else _NORMAL_S3_SKIN_PROFILE) if controls.allow_normal_s3_skin_hermite_dev
                else _FIXED_PROJECTION_S3_EVENT_PROFILE)
    if controls.allow_guided_shell_pair_dev:
        if (settings.pair_motion!='guided' or settings.warped_q4_plane
                or settings.normal_discretization!='primal_mortar'
                or len(definition.surfaces)!=2 or definition.rigid_bodies
                or any(surface.beam_radius is not None for surface in definition.surfaces)
                or len(definition.interactions)!=1
                or definition.interactions[0].normal_law!='penalty'):
            raise ValueError('guided shell pair impact requires two explicit surfaces, no rigid body and one penalty pair')
        elements=tuple(model.mesh.elements.values())
        if (not elements or any(type(element) not in (QualifiedE4PLShellElement,
                NativeParityE4PLS3V2DShellElement) for element in elements)
                or set(eid for surface in definition.surfaces for eid in surface.element_ids)!=set(model.mesh.elements)):
            raise ValueError('guided shell pair impact requires exact qualified Q4/S3 V2D on the declared surfaces')
        for element in elements:
            material=model.get_material(element.material_name)
            if (type(material) is not IsotropicMaterial or material.yield_stress!=0.
                    or material.hardening_curve is not None):
                raise ValueError('guided shell pair impact admits only isotropic elastic material')
        pair=definition.interactions[0]
        selections={surface.name:surface for surface in definition.surfaces}
        if pair.surface in selections and pair.target in selections:
            source,target=selections[pair.surface],selections[pair.target]
            if len(source.element_ids)*len(target.element_ids)>settings.maximum_candidate_pairs:
                raise MemoryError('guided shell pair candidate preflight limit exceeded; no contacts discarded')
        report=validate_general_contact(model,definition,settings)
        if not report.valid:
            raise ValueError('unsupported guided shell pair impact: '+'; '.join(report.issues))
        return _GUIDED_SHELL_PAIR_EVENT_PROFILE
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError("unsupported transient contact: " + "; ".join(report.issues))
    multiple = controls.allow_multiple_beam_planes_dev or controls.allow_multiple_shell_planes_dev
    required_motion = "general" if controls.allow_classical_beam_dev or controls.allow_free_shell_sphere_dev else "guided"
    if (settings.pair_motion != required_motion or settings.warped_q4_plane
            or (not multiple and (len(definition.surfaces) != 1
                or len(definition.rigid_bodies) != 1 or len(definition.interactions) != 1))
            or (multiple and len(definition.interactions) < 2)
            or any(body.kind != ("sphere" if controls.allow_free_sphere_dev else "plane") or body.path_translation is not None
                   for body in definition.rigid_bodies)
            or any(pair.normal_law != "penalty" for pair in definition.interactions)):
        raise ValueError(
            "free sphere transient development requires one admitted surface and one unprescribed penalty sphere"
            if controls.allow_free_sphere_dev else
            "multiple beam transient development requires at least two interactions against fixed penalty planes"
            if controls.allow_multiple_beam_planes_dev else
            "multiple shell transient development requires at least two fixed penalty-plane interactions"
            if multiple else "transient development profile requires one shell surface and one fixed penalty plane")
    surfaces = definition.surfaces
    if set(element_id for surface in surfaces for element_id in surface.element_ids) != set(model.mesh.elements):
        raise ValueError(
            "multiple beam transient development requires every structural element on declared surfaces"
            if multiple else "transient development profile requires every shell element on the surface")
    elements = tuple(model.mesh.elements.values())
    if controls.allow_classical_beam_dev:
        from anyfesolver.elements import BeamElement, QuadraticBeamElement
        if (not elements or any(type(element) not in (BeamElement,QuadraticBeamElement)
                                for element in elements)
                or any(surface.beam_radius is None or surface.beam_end_caps != "none"
                       for surface in surfaces)):
            raise ValueError("classical beam impact requires exact elastic B2/B3 round tubes without end caps")
        profile = (FREE_SPHERE_PROFILE if controls.allow_free_sphere_dev
                   else _MULTIPLE_BEAM_PROFILE if multiple else _BEAM_PROFILE)
    elif all(type(element) is QualifiedE4PLShellElement for element in elements):
        if controls.allow_s3_descriptor_dev:
            raise ValueError("S3 descriptor opt-in requires one qualified S3 V2D element")
        profile = (_Q4_PLASTIC_PROFILE if controls.allow_q4_plastic_dev
                   else FREE_SHELL_SPHERE_PROFILE if controls.allow_free_shell_sphere_dev else _PROFILE)
    elif (len(elements) == 1
          and type(elements[0]) is NativeParityE4PLS3V2DShellElement
          and controls.allow_s3_descriptor_dev):
        profile = _S3_PROFILE
    else:
        raise ValueError("transient development profile requires qualified Q4 or opted-in single S3 V2D")
    for element in model.mesh.elements.values():
        material = model.get_material(element.material_name)
        if controls.allow_q4_plastic_dev:
            from anymaterial import DNVC208MaterialCurve
            if type(material) is not IsotropicMaterial or type(material.hardening_curve) is not DNVC208MaterialCurve:
                raise ValueError('Q4 plastic transient development requires exact isotropic DNV material')
        elif (type(material) is not IsotropicMaterial
                or float(material.yield_stress) != 0.0 or material.hardening_curve is not None):
            raise ValueError("transient development profile admits only isotropic elastic material")
    if controls.q4_plane_event_subdivision_dev:
        if profile not in (_PROFILE,_Q4_PLASTIC_PROFILE):
            raise ValueError('Q4 plane events require exact qualified Q4 elements')
        profile=(_Q4_PLASTIC_EVENT_PROFILE if controls.allow_q4_plastic_dev else _Q4_EVENT_PROFILE)
        if controls.allow_multiple_shell_planes_dev:
            profile=_MULTIPLE_Q4_EVENT_PROFILE
    return profile


def _beam_complete_path_gaps(manager, start, end, start_velocity, end_velocity,
                             step, controls, check_cancel, translations=None, time_s=0.):
    """Every explicit pair retains its own gap gate and shares one split cap."""
    from .beam_temporal_bounds import beam_hermite_gap_bounds
    from .multiple_beam_rigid import MultipleBeamRigidManager

    pairs = manager.pairs if type(manager) is MultipleBeamRigidManager else ((None, manager),)
    remaining = controls.maximum_beam_swept_subdivisions
    minimum = float("inf")
    admissible = True
    for _, child in pairs:
        motion = ({} if translations is None else dict(
            plane_start_point=translations.point(child, time_s),
            plane_end_point=translations.point(child, time_s+step)))
        bounds = beam_hermite_gap_bounds(
            child, start, end, start_velocity, end_velocity, step,
            tolerance_m=controls.beam_swept_gap_tolerance_m,
            maximum_subdivisions=remaining, check_cancel=check_cancel,
            **motion,
        )
        remaining -= bounds.subdivisions
        minimum = min(minimum, bounds.lower_gap)
        admissible = admissible and bounds.lower_gap >= -child.gap_tolerance
    return minimum, admissible


def _with_transient_state_cleanup(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        return _run_with_nonlinear_state_cleanup(operation, *args, **kwargs)
    return wrapped


@_with_transient_state_cleanup
def solve_transient_contact(
    model,
    definition: ContactDefinition,
    transient: TransientConfig | TransientContactSettings,
    *,
    contact_settings: ContactSolverSettings | None = None,
    base_load_case: LoadCase | None = None,
    nodal_load_history: ContactNodalLoadHistory | None = None,
    rigid_translation_histories: tuple[ContactRigidTranslationHistory, ...] = (),
    free_rigid_bodies: tuple[ContactFreeRigidBody, ...] = (),
    solver_controls: ContactTransientSolverControls | None = None,
    restart_checkpoint: Mapping | None = None,
    emit_restart_checkpoint: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> TransientContactResults:
    """Solve bounded elastic shell or classical round-beam contact impact.

    Every Newton evaluation starts from the last committed material state.
    A rejected step leaves both the displacement and material state at the
    last accepted time. The fixed-plane gap is checked independently at each
    accepted endpoint and over its declared cubic Hermite integration path.
    S3 descriptor admission additionally requires explicit opt-in and a
    strictly open, stress-free rigid-normal initial state.
    The S3 descriptor profiles admit a piecewise-linear nodal load history
    only when the complete base load has zero algebraic projection; the
    distinct history profile is bound into restart contracts.
    The separate S3 algebraic-path opt-in instead lifts the physical Newmark
    quadratic through reference-linear elastic/penalty algebraic equilibrium.
    Classical B2/B3 require a separate explicit opt-in, scalar elastic von Karman
    sections, positive physical mass and round centreline geometry without caps.
    Their full Hermite curve/time path is bounded before physical commit.
    The free-sphere opt-in additionally requires explicit mass and isotropic
    centre inertia. Three centre coordinates join the same coupled equations;
    torque-free spin is exact and does not expand friction/rotation admission.
    The separate free-shell-sphere opt-in admits planar-reference Q4 skins,
    constrained zero shell rotations/in-plane motion, and the same supplied
    isotropic body. Its nodal-pressure radial law and complete two-space/time
    Hermite bound have their own development profile and restart identity.
    Full endpoint agreement and corner minima are checked; reported transitions
    describe this discrete path, not exact continuum event times.
    The guided-shell-pair opt-in couples two finite FE bodies on fixed projected
    overlap, admitting only uniform face-normal translation, common tangential
    motion and zero rotations. Exact qualified planar Q4/S3 V2D, elastic material
    and positive physical reduced mass are required. Both-body work, complete
    relative Hermite gaps and feature-bound V11 restart use the shared driver.
    This development profile does not admit general deforming/sliding targets.
    Cancellation admits only the solver's exact unmodified token. It raises
    SolveCancelled at a safe boundary, discards solve-owned pending trials and
    returns no partial result or checkpoint. Completed increments remain committed.
    """

    def check_cancel(stage):
        if not _is_known_cancellation_checkpoint(cancellation_token):
            raise TypeError("transient contact requires an exact unmodified CancellationToken")
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled(stage)

    check_cancel("transient contact entry")
    transient, controls = _resolve_transient_inputs(transient, solver_controls)
    if (restart_checkpoint is not None or emit_restart_checkpoint) and transient.save_every != 1:
        raise ValueError("transient contact development restart requires save_every=1")
    if emit_restart_checkpoint and transient.t_end <= 0.0:
        raise ValueError("transient contact checkpoint needs one accepted time step")
    settings = ContactSolverSettings() if contact_settings is None else contact_settings
    profile = _admit(model, definition, settings, controls)
    free_bodies = tuple(free_rigid_bodies)
    if any(type(body) is not ContactFreeRigidBody for body in free_bodies):
        raise TypeError("free_rigid_bodies requires ContactFreeRigidBody entries")
    if bool(free_bodies) != controls.allow_free_sphere_dev:
        raise ValueError("free body definitions require explicit free-sphere opt-in")
    if free_bodies and (len(free_bodies) != 1
            or free_bodies[0].body_name != definition.rigid_bodies[0].name
            or rigid_translation_histories):
        raise ValueError("free sphere requires one matching body and no prescribed translation")
    free_body = free_bodies[0] if free_bodies else None
    if free_body is not None and not np.all(np.isfinite(transient.t_end*np.asarray(free_body.initial_angular_velocity_rad_s))):
        raise ValueError("free sphere angular travel is nonfinite")
    tool_histories = tuple(rigid_translation_histories)
    if any(type(h) is not ContactRigidTranslationHistory for h in tool_histories):
        raise TypeError("rigid_translation_histories requires ContactRigidTranslationHistory entries")
    tool_histories = tuple(sorted(tool_histories, key=lambda h: h.body_name))
    if tool_histories:
        if not controls.allow_classical_beam_dev:
            raise ValueError("prescribed translation currently requires the classical beam profile")
        if (len({h.body_name for h in tool_histories}) != len(tool_histories)
                or any(h.body_name not in {body.name for body in definition.rigid_bodies} for h in tool_histories)):
            raise ValueError("rigid translation body identities must be unique and declared")
        if any(h.points[-1][0] < transient.t_end for h in tool_histories):
            raise ValueError("rigid translation histories must cover t_end")
        if any(b[0]-a[0] < max(controls.minimum_dt, 16*np.finfo(float).eps*max(1., abs(b[0])))
               for h in tool_histories for a, b in zip(h.points, h.points[1:])):
            raise ValueError("rigid translation knots are closer than admitted time resolution")
    if nodal_load_history is not None:
        if controls.q4_plane_event_subdivision_dev and not (controls.allow_multiple_shell_planes_dev or controls.allow_guided_shell_pair_dev or controls.allow_fixed_projection_s3_target_dev):
            raise ValueError('Q4 plane event development does not admit load histories')
        if type(nodal_load_history) is not ContactNodalLoadHistory:
            raise TypeError("nodal_load_history must be ContactNodalLoadHistory")
        if base_load_case is None or profile not in (_PROFILE, _S3_PROFILE, _BEAM_PROFILE, _MULTIPLE_BEAM_PROFILE, FREE_SPHERE_PROFILE, FREE_SHELL_SPHERE_PROFILE, _MULTIPLE_Q4_EVENT_PROFILE,_GUIDED_SHELL_PAIR_EVENT_PROFILE,_FIXED_PROJECTION_S3_EVENT_PROFILE,_NORMAL_S3_SKIN_PROFILE,_NORMAL_S3_POLYGON_PROFILE):
            raise ValueError("nodal load history requires one Q4 plane nodal LoadCase or opted-in S3 plane nodal LoadCase")
        if nodal_load_history.points[-1][0] < transient.t_end:
            raise ValueError("nodal load history must cover t_end")
        if any(
            right[0]-left[0] < max(
                controls.minimum_dt,
                16.0*np.finfo(float).eps*max(1.0, abs(right[0])),
            )
            for left, right in zip(nodal_load_history.points,
                                   nodal_load_history.points[1:])
        ):
            raise ValueError("nodal load history knots are closer than the admitted minimum_dt or time resolution")
        if profile == _S3_PROFILE:
            profile = _S3_HISTORY_PROFILE
        elif profile == _BEAM_PROFILE:
            profile = _BEAM_HISTORY_PROFILE
        elif profile == _MULTIPLE_BEAM_PROFILE:
            profile = _MULTIPLE_BEAM_HISTORY_PROFILE
        elif profile == FREE_SPHERE_PROFILE:
            profile = FREE_SPHERE_HISTORY_PROFILE
        elif profile == FREE_SHELL_SPHERE_PROFILE:
            profile = FREE_SHELL_SPHERE_HISTORY_PROFILE
    if tool_histories or free_body is not None:
        if tool_histories:
            profile = _BEAM_TOOL_HISTORY_PROFILE if nodal_load_history is not None else _BEAM_TOOL_PROFILE
        all_knots = sorted({p[0] for h in tool_histories for p in h.points}
                           | ({p[0] for p in nodal_load_history.points} if nodal_load_history is not None else set()))
        if any(b-a < max(controls.minimum_dt, 16*np.finfo(float).eps*max(1., abs(b)))
               for a, b in zip(all_knots, all_knots[1:])):
            raise ValueError("combined tool/load history knots are closer than admitted time resolution")
        reached_knots = [t for t in all_knots if t <= transient.t_end]
        for a, b in zip(reached_knots, reached_knots[1:]):
            roundoff = 128*np.finfo(float).eps*max(abs(a), abs(b), transient.dt, controls.minimum_dt)
            count = max(1, int(np.ceil(max(0., b-a-roundoff)/transient.dt)))
            if (b-a)/count < controls.minimum_dt-roundoff:
                raise ValueError("tool/load knot intervals cannot fit dt and minimum_dt")
    rotational_descriptor = controls.allow_rotational_beam_s3_dev or _directional_selected(controls)
    s3_descriptor = profile in (_S3_PROFILE, _S3_HISTORY_PROFILE)
    if controls.s3_one_sided_rates_dev:
        profile = (_S3_DIRECTIONAL_HISTORY_PROFILE if nodal_load_history is not None
                   else _S3_DIRECTIONAL_PROFILE)
    if controls.s3_algebraic_path_dev:
        profile = (_S3_ALGEBRAIC_PATH_HISTORY_PROFILE if nodal_load_history is not None
                   else _S3_ALGEBRAIC_PATH_PROFILE)
    if controls.s3_event_subdivision_dev:
        profile = (_S3_EVENT_HISTORY_PROFILE if nodal_load_history is not None
                   else _S3_EVENT_PROFILE)
    if base_load_case is not None:
        if (type(base_load_case) is not LoadCase
                or not base_load_case.nodal_loads
                or base_load_case.element_loads
                or base_load_case.pressure_loads
                or base_load_case.gravity is not None
                or base_load_case.added_node_masses
                or base_load_case.follower_pressure):
            raise ValueError("transient development loads require one nodal-only dead LoadCase")
    model.apply_boundary_conditions()
    size = model.mesh.dof_manager.total_dofs
    external_load, _load_info = assemble_load_vector(model, base_load_case)
    if external_load.shape != (size,) or not np.all(np.isfinite(external_load)):
        raise ValueError("transient development load vector is invalid")
    if transient.initial_velocity is None:
        raise ValueError("initial_velocity is required for transient contact")
    if controls.allow_normal_s3_skin_hermite_dev:
        for supplied in (transient.initial_velocity,transient.initial_displacement):
            if supplied is not None and np.asarray(supplied).dtype.kind not in 'fiu':
                raise ValueError('normal S3 skin initial conditions require real numeric vectors')
    initial_velocity = np.asarray(transient.initial_velocity, dtype=float)
    if initial_velocity.shape != (size,) or not np.all(np.isfinite(initial_velocity)):
        raise ValueError(f"initial_velocity must have {size} structural components")
    initial_displacement = (
        np.zeros(size, dtype=float) if transient.initial_displacement is None
        else np.asarray(transient.initial_displacement, dtype=float)
    )
    if (initial_displacement.shape != (size,)
            or not np.all(np.isfinite(initial_displacement))):
        raise ValueError(f"initial_displacement must have {size} structural components")
    K0, _ = assemble_stiffness_matrix(model)
    M, _ = assemble_mass_matrix(model)
    _, _, T, u0, independent_dofs, _ = build_constraint_transformation(
        K0, np.zeros(size, dtype=float), model,
    )
    if controls.allow_free_shell_sphere_dev:
        _admit_free_shell_coordinates(model, T, u0)
    if free_body is not None:
        M = sparse.block_diag((M, free_body.mass_kg*sparse.eye(3)), format='csr')
        T = sparse.block_diag((T, sparse.eye(3)), format='csr')
        u0 = np.r_[u0, np.zeros(3)]
        independent_dofs = np.r_[independent_dofs, np.arange(size, size+3)]
        initial_displacement = np.r_[initial_displacement, np.zeros(3)]
        initial_velocity = np.r_[initial_velocity, free_body.initial_velocity_m_s]
        external_load = np.r_[external_load, np.zeros(3)]
    external_reduced = np.asarray(T.T @ external_load).reshape(-1)
    if np.linalg.norm(u0, ord=np.inf) > 1.0e-13:
        raise ValueError("transient development profile requires zero prescribed displacements")
    if not len(independent_dofs):
        raise ValueError("transient contact requires at least one free coordinate")
    if s3_descriptor:
        element = next(iter(model.mesh.elements.values()))
        expected_free = sorted(
            int(model.mesh.dof_manager.get_node_dofs(node)[local])
            for node in element.node_ids for local in (2, 3, 4, 5)
        )
        if (sorted(map(int, independent_dofs)) != expected_free
                or (T - sparse.eye(size, format="csr")[:, independent_dofs]).nnz):
            raise ValueError(
                "S3 descriptor development requires each node guided in-plane "
                "with free normal displacement and rotations"
            )
    Mred = (T.T @ M @ T).tocsr()
    if Mred.shape[0] > 512:
        raise ValueError("transient development profile admits at most 512 free coordinates")
    if controls.allow_multiple_deformable_beam_s3_dev and not np.isfinite(Mred.data).all():
        raise ValueError('multiple beam-S3 requires finite positive physical reduced mass')
    descriptor_basis = None
    if controls.allow_deformable_beam_s3_dev or controls.allow_multiple_deformable_beam_s3_dev or profile in (_PROFILE,_BEAM_PROFILE,_BEAM_HISTORY_PROFILE,_MULTIPLE_Q4_EVENT_PROFILE,_GUIDED_SHELL_PAIR_EVENT_PROFILE,_FIXED_PROJECTION_S3_EVENT_PROFILE,_NORMAL_S3_SKIN_PROFILE,_NORMAL_S3_POLYGON_PROFILE,
                   _Q4_PLASTIC_PROFILE,_Q4_EVENT_PROFILE,_Q4_PLASTIC_EVENT_PROFILE,
                   _MULTIPLE_BEAM_PROFILE,_MULTIPLE_BEAM_HISTORY_PROFILE,_BEAM_TOOL_PROFILE,_BEAM_TOOL_HISTORY_PROFILE,
                   FREE_SPHERE_PROFILE,FREE_SPHERE_HISTORY_PROFILE,
                   FREE_SHELL_SPHERE_PROFILE,FREE_SHELL_SPHERE_HISTORY_PROFILE):
        try:
            np.linalg.cholesky(Mred.toarray())
        except np.linalg.LinAlgError as exc:
            raise ValueError("reduced mass is not positive definite; algebraic dynamics are not admitted") from exc
    else:
        descriptor_basis = build_declared_algebraic_basis(
            model, M, Mred, T, independent_dofs, dense_size_limit=512,
        )
        nullity = descriptor_basis.reduced_basis.shape[1]
        spectrum = np.linalg.eigvalsh(Mred.toarray())
        mass_scale = float(np.max(np.abs(spectrum)))
        if not np.isfinite(mass_scale) or mass_scale <= 0.0:
            raise ValueError("S3 descriptor needs positive physical mass")
        threshold = 1.0e-10*mass_scale
        if (nullity < 1 or int(np.count_nonzero(spectrum <= threshold)) != nullity
                or np.any(spectrum < -threshold)):
            raise ValueError("S3 descriptor mass nullity disagrees with declared algebraic basis")
        if s3_descriptor and nodal_load_history is not None:
            # The complete base load must satisfy Z.T f_base = 0 so the scalar
            # history cannot force the algebraic manifold; a zero initial
            # scale must not hide an algebraic base load.
            algebraic_base_force = np.asarray(
                descriptor_basis.reduced_basis.T @ external_reduced
            ).reshape(-1)
            # History amplitude and slope are not bounded. A tolerance-sized
            # nonzero algebraic base force could be amplified later, and its
            # time derivatives are absent from the admitted rate equations.
            if np.any(algebraic_base_force != 0.0):
                raise ValueError(
                    "S3 descriptor base load has unresolved algebraic force"
                )
    q = np.asarray(initial_displacement[independent_dofs], dtype=float).copy()
    full_initial = np.asarray(T @ q).reshape(-1)
    if not np.allclose(full_initial, initial_displacement,
                       atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("initial_displacement does not satisfy structural constraints")
    vfull = initial_velocity
    vred = np.asarray(vfull[independent_dofs], dtype=float)
    if not np.allclose(np.asarray(T @ vred).reshape(-1), vfull,
                       atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("initial_velocity does not satisfy structural constraints")
    if s3_descriptor:
        # This first descriptor impact admits a stress-free rigid normal
        # approach. General DAE initial states require a separate consistency
        # solve for algebraic displacement, velocity and acceleration.
        if np.linalg.norm(full_initial, ord=np.inf) > 1.0e-13:
            raise ValueError("S3 descriptor development needs zero initial displacement")
        normal = np.asarray(definition.rigid_bodies[0].normal, dtype=float)
        nodal_velocity = np.asarray([
            initial_velocity[model.mesh.dof_manager.get_node_dofs(node)[:6]]
            for node in model.mesh.elements[
                definition.surfaces[0].element_ids[0]
            ].node_ids
        ])
        speed = float(nodal_velocity[0, :3] @ normal)
        expected_velocity = np.r_[speed*normal, np.zeros(3)]
        if not np.allclose(nodal_velocity, expected_velocity[None, :],
                           rtol=0.0, atol=1.0e-12):
            raise ValueError("S3 descriptor development needs uniform rigid-normal initial velocity")
    committed_states = {}
    material_packet = None
    if controls.allow_q4_plastic_dev:
        from .q4_material_checkpoint import prepare_q4_material_checkpoint,restore_q4_material_checkpoint

    def sealed_material_candidate(full_displacement,candidate_states):
        packet=prepare_q4_material_checkpoint(model,full_displacement,candidate_states,num_layers=3)
        owned={record['element_id']:record['state'] for record in packet['element_states']}
        return packet,owned
    # Legacy managers check the configured fresh pose. The normal-S3 driver
    # checks that pose separately with its matched-overlap gap certificate;
    # its static manager retains the original Cartesian guard at reference.
    # Resumed physical poses are checked against the accepted gap budget.
    selected_scope=contact_scope
    if rotational_descriptor:
        from .beam_s3_rotational_transient import rotational_scope
        selected_scope=rotational_scope
        if _directional_selected(controls):
            from .beam_s3_directional_rates import directional_scope
            selected_scope=directional_scope
            if controls.allow_equilibrated_directional_beam_s3_dev:
                from .beam_s3_directional_rates import equilibrated_directional_scope
                selected_scope=equilibrated_directional_scope
    if controls.allow_deformable_beam_s3_dev:
        from .beam_s3_transient import beam_s3_scope
        selected_scope=beam_s3_scope
    if controls.allow_multiple_deformable_beam_s3_dev:
        from .multiple_beam_s3_transient import multiple_beam_s3_scope
        selected_scope=multiple_beam_s3_scope
    with selected_scope(model, definition, settings,
                       initial_displacement=(np.zeros(size) if controls.allow_normal_s3_skin_hermite_dev
                                             else full_initial[:size])) as manager:
        if rotational_descriptor:
            from .beam_s3_rotational_transient import RotationalBeamS3Manager, recover_rates
            expected_manager=RotationalBeamS3Manager
            if _directional_selected(controls):
                from .beam_s3_directional_rates import DirectionalBeamS3Manager,recover_directional_rates
                expected_manager=DirectionalBeamS3Manager
                if controls.allow_equilibrated_directional_beam_s3_dev:
                    from .beam_s3_directional_rates import EquilibratedDirectionalBeamS3Manager
                    expected_manager=EquilibratedDirectionalBeamS3Manager
            if controls.allow_equilibrated_directional_beam_s3_dev:
                if base_load_case is None or nodal_load_history is not None or transient.initial_displacement is None or not np.any(initial_displacement):
                    raise ValueError('equilibrated directional requires constant nodal LoadCase and supplied nonzero elastic initial pose')
                from .beam_s3_directional_rates import exact_gap_polynomial
                if not exact_gap_polynomial(manager,initial_displacement)[0]:
                    raise ValueError('equilibrated directional initial geometry needs exact full-line touch')
            else:
                if base_load_case is not None or nodal_load_history is not None:
                    raise ValueError('rotational descriptor currently requires unforced impact')
                if np.any(initial_displacement != 0.):
                    raise ValueError('rotational descriptor requires a virgin zero initial pose')
        elif controls.allow_deformable_beam_s3_dev or controls.allow_multiple_deformable_beam_s3_dev:
            from .beam_s3_transient import BeamS3TransientManager
            expected_manager=BeamS3TransientManager
            if controls.allow_multiple_deformable_beam_s3_dev:
                from .multiple_beam_s3_transient import MultipleBeamS3TransientManager
                expected_manager=MultipleBeamS3TransientManager
            if (np.any(initial_velocity[manager.shell_dofs[:,3:]]!=0.)
                    or np.any(initial_displacement[manager.shell_dofs[:,3:]]!=0.)):
                raise ValueError('beam-S3 initial target rotations and rates must be exactly zero')
            if np.any(T[manager.shell_dofs[:,3:].ravel(),:].toarray()!=0.):
                raise ValueError('beam-S3 shell rotations must be explicitly constrained zero')
        elif controls.allow_classical_beam_dev:
            from .beam_plane import _BeamPlaneManager
            from .multiple_beam_rigid import MultipleBeamRigidManager
            from .beam_radial import _BeamRadialManager
            expected_manager = (_BeamRadialManager if free_body is not None
                                else MultipleBeamRigidManager if controls.allow_multiple_beam_planes_dev
                                else _BeamPlaneManager)
        elif controls.allow_multiple_shell_planes_dev:
            from .multiple_shell_planes import MultipleShellPlaneManager
            expected_manager = MultipleShellPlaneManager
        elif controls.allow_guided_shell_pair_dev:
            from .pair import _PlanarShellPairManager
            expected_manager = _PlanarShellPairManager
        elif controls.allow_fixed_projection_s3_target_dev or controls.allow_normal_s3_skin_hermite_dev:
            from .s3_patch_static import GeneralS3PatchManager
            expected_manager = GeneralS3PatchManager
        elif controls.allow_free_shell_sphere_dev:
            from .sphere import _SphereShellManager
            expected_manager = _SphereShellManager
        else:
            expected_manager = _PlanarQ4PlaneManager
        if type(manager) is not expected_manager:
            raise ValueError("transient development profile selected an unsupported contact owner")
        plane_state=None
        pressure_contract = None
        normal_skin_path = None
        pressure_features=(manager.pressure_features if controls.allow_multiple_shell_planes_dev else None)
        if controls.allow_guided_shell_pair_dev:
            from .guided_pair_transient_geometry import (prepare_guided_shell_pair_transient,
                guided_shell_pair_path_minimum_gap)
            pressure_features=prepare_guided_shell_pair_transient(manager,T)
        if controls.allow_fixed_projection_s3_target_dev:
            from .fixed_projection_s3_events import prepare_fixed_projection_s3_events
            pressure_contract = prepare_fixed_projection_s3_events(model, manager, T, prescribed=u0)
            pressure_features = pressure_contract.features
            manager.restrict_tangent_to_reduced_variations(T)
        if controls.allow_normal_s3_skin_hermite_dev:
            from .normal_s3_skin_path import prepare_normal_s3_skin_path
            from .plane_newmark_events import localize_endpoint_event
            normal_skin_path = prepare_normal_s3_skin_path(model,manager,T)
            pressure_features = normal_skin_path.features
            manager.restrict_tangent_to_reduced_variations(T)
        if controls.q4_plane_event_subdivision_dev:
            from .plane_newmark_events import (hermite_branch_extrema,
                branch_path_admissible,localize_endpoint_event)
            if pressure_contract is not None:
                pressure_gradient = sparse.csr_matrix(pressure_contract.reduced_gradient)
                pressure_reference = pressure_contract.reference_gap
            else:
                normalized_gradient=sparse.diags(1./manager.support)@manager.gap_gradient
                pressure_gradient=(normalized_gradient@T).tocsr()
                pressure_reference=(manager.reference_weighted_gap
                    +manager.gap_gradient@u0)/manager.support
        if free_body is not None:
            manager = _FreeSphereManager(manager, free_body,allow_shell=controls.allow_free_shell_sphere_dev)
            manager.profile = profile
        translations = _PlaneTranslations(manager, tool_histories) if tool_histories else None
        path_adapter = (_S3AffinePathAdapter.prepare(
            model, T, descriptor_basis, manager, committed_states,
        ) if controls.s3_algebraic_path_dev else None)

        if restart_checkpoint is not None or emit_restart_checkpoint:
            model_fingerprint = model_configuration_fingerprint(model)
            analysis_fingerprint = analysis_configuration_fingerprint(
                _restart_analysis_contract(
                    transient, controls, manager, profile,
                    base_load_case, external_load, nodal_load_history, tool_histories,
                )
            )
        if restart_checkpoint is None:
            current_time = 0.0
            accepted_steps = 0
            resumed_acceleration = None
            rotational_previous_velocity=vred.copy()
            rotational_last_dt=0.
            resumed_work = None
            last_path_active = None
            event_boundary_support = ()
            tool_work = {h.body_name: 0. for h in tool_histories}
            tool_error = tool_work.copy()
        else:
            resumed = _read_restart_checkpoint(
                    restart_checkpoint, model_fingerprint,
                    analysis_fingerprint, Mred.shape[0], transient.t_end,
                    load_history=nodal_load_history,
                    path_pressure_dofs=(manager.support.size if path_adapter is not None else 0),
                    event_subdivision=controls.s3_event_subdivision_dev,
                    tool_body_names=tuple(h.body_name for h in tool_histories),
                    free_body_name=free_body.body_name if free_body is not None else None,
                    q4_material_state=controls.allow_q4_plastic_dev,
                    plane_event_state=controls.q4_plane_event_subdivision_dev,
                    plane_pressure_features=pressure_features,
                    guided_shell_pair=controls.allow_guided_shell_pair_dev,
                    fixed_projection_s3_target=controls.allow_fixed_projection_s3_target_dev,
                    normal_s3_skin_hermite=controls.allow_normal_s3_skin_hermite_dev,
                    normal_s3_polygon_pressure=controls.normal_s3_polygon_pressure_bounds_dev,
                    beam_s3=controls.allow_deformable_beam_s3_dev,
                    rotational_beam_s3=rotational_descriptor,
                    directional_beam_s3=_directional_selected(controls),
                    equilibrated_directional=controls.allow_equilibrated_directional_beam_s3_dev,
                    multiple_beam_s3=controls.allow_multiple_deformable_beam_s3_dev,
                )
            current_time, accepted_steps, q, vred, resumed_acceleration, resumed_work, last_path_active = resumed[:7]
            if rotational_descriptor:
                rotational_previous_velocity,rotational_last_dt=resumed[-2:]
                Z=descriptor_basis.reduced_basis.toarray()
                expected=Z.T@(vred-rotational_previous_velocity)/rotational_last_dt
                if not np.allclose(Z.T@resumed_acceleration,expected,rtol=0.,atol=64*np.finfo(float).eps*max(1.,np.max(abs(expected)))):
                    raise ValueError('rotational checkpoint null acceleration disagrees with secant history')
            if controls.allow_q4_plastic_dev:
                material_packet=resumed[7]
                committed_states=restore_q4_material_checkpoint(model,u0+np.asarray(T@q).reshape(-1),material_packet,num_layers=3)
            if controls.q4_plane_event_subdivision_dev:
                expected=_plane_event_state(pressure_reference,pressure_gradient,q,vred,
                    controls.plane_event_gap_tolerance_m)
                _validate_plane_event_state(resumed[-1],expected)
            event_boundary_support = resumed[7] if controls.s3_event_subdivision_dev else ()
            if tool_histories:
                tool_work, tool_error = resumed[-2:]
            else:
                tool_work, tool_error = {}, {}
            if event_boundary_support:
                boundary_gaps = path_adapter.reference_gap+path_adapter.gradient @ q
                if np.any(np.abs(boundary_gaps[list(event_boundary_support)])
                          > controls.s3_event_gap_tolerance_m):
                    raise ValueError("transient contact checkpoint event boundary exceeds its gap budget")

        def load_at(time_s: float) -> np.ndarray:
            if nodal_load_history is None:
                return external_reduced
            return nodal_load_history.factor(time_s)*external_reduced

        reaction_candidate = None

        def capture_reaction(full, force, time_s):
            nonlocal reaction_candidate
            if controls.save_constraint_reactions_dev:
                factor = 1.0 if nodal_load_history is None else nodal_load_history.factor(time_s)
                reaction_candidate = (full.copy(), float(time_s),
                                      np.asarray(force).reshape(-1).copy() - factor*external_load)

        def accepted_reaction(full, time_s, candidate_acceleration):
            if (reaction_candidate is None or reaction_candidate[1] != float(time_s)
                    or not np.array_equal(reaction_candidate[0], full)):
                raise RuntimeError('constraint reaction does not match the accepted candidate')
            reaction = np.asarray(M @ (T @ candidate_acceleration)).reshape(-1) + reaction_candidate[2]
            if not np.all(np.isfinite(reaction)):
                raise RuntimeError('nonfinite accepted constraint reaction')
            return reaction[:size].copy()

        def evaluate(position: np.ndarray, time_s: float, *, tangent: bool):
            if translations is not None:
                translations.set_pose(time_s)
            full = np.asarray(T @ position).reshape(-1)
            if normal_skin_path is not None:
                normal_skin_path.require_motion(full)
            if pressure_contract is not None:
                # Preserve full general both-body work; the sidecar only certifies
                # the exact represented subspace and supplies event coordinates.
                with suspend_contact():
                    force, stiffness, trial = _assemble_nonlinear_system(
                        model, full, committed_states, 3, tangent=tangent,
                        kinematics='von_karman')
                contact_force, contact_tangent, contact_result = manager.evaluate(full, tangent=tangent)
                pressure_contract.require_candidate(full, position, contact_result)
                force = force + contact_force
                stiffness = stiffness + contact_tangent if tangent else None
                if controls.save_constraint_reactions_dev:
                    capture_reaction(full, force, time_s)
                return full, np.asarray(T.T @ force).reshape(-1)-load_at(time_s), stiffness, trial
            if free_body is not None:
                # Structural work is evaluated on its own coordinates; contact
                # then supplies complete structural/centre residual and blocks.
                with suspend_contact():
                    force, stiffness, trial = _assemble_nonlinear_system(
                        model, full[:size], committed_states, 3, tangent=tangent,
                        kinematics="von_karman")
                contact_force, contact_tangent, _ = manager.evaluate(full, tangent=tangent)
                force = np.r_[force, np.zeros(3)]+contact_force
                stiffness = (sparse.block_diag((stiffness, sparse.csr_matrix((3, 3))), format='csr')
                             +contact_tangent if tangent else None)
                if controls.save_constraint_reactions_dev:
                    capture_reaction(full, force, time_s)
                return full, np.asarray(T.T @ force).reshape(-1)-load_at(time_s), stiffness, trial
            force, stiffness, trial = _assemble_nonlinear_system(
                model, full, committed_states, 3, tangent=tangent,
                kinematics="von_karman",
            )
            if controls.save_constraint_reactions_dev:
                capture_reaction(full, force, time_s)
            return (full, np.asarray(T.T @ force).reshape(-1)
                    - load_at(time_s), stiffness, trial)

        check_cancel("transient contact initial evaluation")
        full0, force0, initial_tangent, trial0 = evaluate(
            q, current_time,
            tangent=(rotational_descriptor or (s3_descriptor and restart_checkpoint is None)
                     or controls.allow_q4_plastic_dev),
        )
        contact0 = manager.evaluate(full0, tangent=False)[2]
        if normal_skin_path is not None:
            contact0,failure=normal_skin_path.overlap_endpoint(manager,full0,contact0,
                cancel=lambda:check_cancel('normal S3 initial matched gap'))
            if failure is not None:raise ValueError(failure)
            if restart_checkpoint is None:
                scale=max(1.,*(float(np.max(np.abs(value))) for value in manager._positions(full0)))
                if contact0.minimum_gap<-256*np.finfo(float).eps*scale:
                    raise ValueError(f'normal S3 initial matched penetration unresolved: {contact0.minimum_gap:.6g} m')
        normal_committed_gaps=(np.asarray(contact0.normalized_weighted_gap_m,dtype=float).copy()
            if normal_skin_path is not None else None)
        if not contact0.gap_tolerance_met:
            raise ValueError("initial contact gap exceeds the registered tolerance")
        if (s3_descriptor and restart_checkpoint is None
                and contact0.minimum_gap <= 1.0e-10):
            raise ValueError("S3 descriptor development requires a strictly open initial gap")
        if rotational_descriptor:
            # Check total algebraic equilibrium and supplied velocity, including
            # on resume. Null acceleration is a numerical endpoint-rate record.
            if _directional_selected(controls):
                recovered,_=recover_directional_rates(manager,full0,T,Mred,descriptor_basis,
                    initial_tangent,vred,np.zeros_like(vred),vred,1.,contact0)
            else:
                recovered,_ = recover_rates(T,Mred,descriptor_basis,initial_tangent,
                    vred,np.zeros_like(vred),vred,1.,contact0)
            Z=descriptor_basis.reduced_basis
            if np.linalg.norm(Z.T@force0)>controls.equilibrium_tolerance:
                raise ValueError('rotational descriptor initial algebraic force is inconsistent')
            if not np.allclose(recovered,vred,rtol=0.,atol=1.e-10):
                raise ValueError('rotational descriptor initial algebraic velocity is inconsistent')
        if resumed_acceleration is None:
            if rotational_descriptor:
                if np.linalg.norm(force0)>controls.equilibrium_tolerance:
                    raise ValueError('equilibrated directional initial total force is inconsistent' if controls.allow_equilibrated_directional_beam_s3_dev else 'rotational descriptor needs a stress-free initial pose')
                acceleration=np.zeros_like(q) # declared numerical null-rate seed
            elif s3_descriptor:
                basis = descriptor_basis.reduced_basis
                reduced_tangent = (T.T @ initial_tangent @ T).tocsr()
                algebraic_velocity_work = np.asarray(
                    basis.T @ (reduced_tangent @ vred)
                ).reshape(-1)
                if float(np.linalg.norm(algebraic_velocity_work)) > controls.equilibrium_tolerance:
                    raise ValueError("S3 descriptor initial algebraic state is inconsistent")
                if base_load_case is None:
                    if float(np.linalg.norm(force0)) > controls.equilibrium_tolerance:
                        raise ValueError("S3 descriptor initial algebraic state is inconsistent")
                    # Rigid-normal free motion has zero initial acceleration.
                    acceleration = np.zeros_like(q)
                else:
                    force_scale = max(1.0, float(np.linalg.norm(force0)))
                    algebraic_force = np.asarray(basis.T @ force0).reshape(-1)
                    if (float(np.linalg.norm(algebraic_force))
                            > controls.equilibrium_tolerance*force_scale):
                        raise ValueError("S3 descriptor initial load has unresolved algebraic force")
                    physical_acceleration = -np.linalg.pinv(
                        Mred.toarray(), rcond=1.0e-10,
                    ) @ force0
                    mass_balance = Mred @ physical_acceleration + force0
                    if (not np.all(np.isfinite(physical_acceleration))
                            or float(np.linalg.norm(mass_balance))
                            > controls.equilibrium_tolerance*force_scale):
                        raise ValueError("S3 descriptor initial load is outside the physical mass range")
                    recovered_velocity, acceleration = _consistent_s3_descriptor_rates(
                        model, T, Mred, descriptor_basis, full0, vred,
                        physical_acceleration, initial_tangent,
                        committed_states, controls.s3_descriptor_difference_step_s,
                        one_sided_manager=(manager if controls.s3_one_sided_rates_dev else None),
                        maximum_iterations=controls.maximum_newton_iterations,
                    )
                    if not np.allclose(recovered_velocity, vred,
                                       rtol=0.0, atol=1.0e-10):
                        raise ValueError("S3 descriptor initial loaded velocity is inconsistent")
            else:
                acceleration = np.asarray(spsolve(Mred, -force0)).reshape(-1)
        else:
            acceleration = resumed_acceleration
            balance = Mred @ acceleration + force0
            scale = max(1.0, float(np.linalg.norm(Mred @ acceleration)),
                        float(np.linalg.norm(force0)))
            if float(np.linalg.norm(balance)) > controls.equilibrium_tolerance*scale:
                raise ValueError("transient contact checkpoint acceleration is inconsistent")
        check_cancel("transient contact initial acceptance")
        initial_body_state = (manager.state(current_time, full0, np.asarray(T @ vred).reshape(-1),
            np.asarray(T @ acceleration).reshape(-1), contact0) if free_body is not None else None)
        if controls.q4_plane_event_subdivision_dev:
            plane_state=_plane_event_state(pressure_reference,pressure_gradient,q,vred,
                controls.plane_event_gap_tolerance_m)
        constraint_reactions = ([accepted_reaction(full0, current_time, acceleration)]
                               if controls.save_constraint_reactions_dev else [])
        if controls.allow_q4_plastic_dev:
            if restart_checkpoint is None:
                if any(np.any(np.asarray(state['alpha'])!=0.) for state in trial0.values()):
                    raise ValueError('Q4 plastic development requires an initially nonplastic pose')
                material_packet,trial0=sealed_material_candidate(full0,trial0)
                committed_states=commit_state_candidate(committed_states,trial0)
            else:
                # The equilibrium observation must not commit another return map.
                discard_active_state_candidate(committed_states)
        else:
            committed_states = commit_state_candidate(committed_states, trial0)
        times = [current_time]
        positions = [full0.copy()]
        velocities = [np.asarray(T @ vred).reshape(-1)]
        accelerations = [np.asarray(T @ acceleration).reshape(-1)]
        contacts = [contact0]
        accumulated_work = (
            float(resumed_work) if resumed_work is not None
            else float(external_load @ (full0 - full_initial))
            if nodal_load_history is None else 0.0
        )
        external_work = [accumulated_work]
        tool_works, tool_errors = [tool_work.copy()], [tool_error.copy()]
        body_states = [{free_body.body_name: initial_body_state}] if free_body is not None else []
        material_history=[committed_states] if controls.save_q4_material_history_dev else []
        swept_gaps = [contact0.minimum_gap]
        pair_swept_history=([{name:result.minimum_gap for name,result in contact0.pairs}]
                            if controls.allow_multiple_shell_planes_dev or controls.allow_multiple_deformable_beam_s3_dev else [])
        pending_pair_gaps=({name:float('inf') for name,_ in manager.pairs}
                           if controls.allow_multiple_shell_planes_dev or controls.allow_multiple_deformable_beam_s3_dev else {})
        residuals = [float(np.linalg.norm(Mred @ acceleration + force0))]
        iterations = []
        path_transitions = []
        localized_events = []
        event_evaluations = 0
        plane_total_stage_evaluations = int(controls.q4_plane_event_subdivision_dev)
        plane_total_endpoint_requests = 0
        cutbacks = 0
        cutback_diagnostics = []
        previous_full = full0.copy()
        previous_velocity = np.asarray(T @ vred).reshape(-1)
        pending_swept_gap = float("inf")
        pending_residual = 0.0
        last_step_was_end_truncated = False
        _alpha, beta, gamma = transient.integration_parameters()
        while (current_time < transient.t_end if tool_histories or free_body is not None else
               current_time < transient.t_end - 1.0e-14 * max(1.0, transient.t_end)):
            remaining = transient.t_end - current_time
            step = min(transient.dt, remaining)
            if controls.equilibrated_newmark_acceleration_unknown_dev:
                # Match the nominal solved family when only endpoint arithmetic differs.
                allowance=128*np.finfo(float).eps*max(abs(current_time),abs(transient.t_end),abs(transient.dt),np.finfo(float).tiny)
                if abs(remaining-transient.dt)<=allowance:
                    step=transient.dt
            if nodal_load_history is not None and not tool_histories and free_body is None:
                knot_index = bisect_right(
                    nodal_load_history.points, (current_time, float("inf")),
                )
                next_knot = (nodal_load_history.points[knot_index][0]
                             if knot_index < len(nodal_load_history.points)
                             else transient.t_end)
                if (knot_index < len(nodal_load_history.points)
                        and next_knot-current_time < controls.minimum_dt):
                    raise ValueError(
                        "nodal load history knot is too close to the current time"
                    )
                knot_step = next_knot-current_time
                # Integrate to the actual knot when the nominal step misses
                # it only through arithmetic rounding. Without this, repeated
                # dt addition can leave a sub-minimum phantom step before a
                # perfectly resolved knot. Do not advance time without solving.
                knot_roundoff = 128*np.finfo(float).eps*max(
                    abs(current_time),abs(next_knot),abs(step),np.finfo(float).tiny,
                )
                step = (knot_step if next_knot <= transient.t_end
                        and abs(step-knot_step) <= knot_roundoff
                        else min(step,knot_step))
            if tool_histories or free_body is not None:
                index = bisect_right(all_knots, current_time)
                if index < len(all_knots) and all_knots[index] <= transient.t_end:
                    next_knot = all_knots[index]
                    distance = next_knot-current_time
                    roundoff = 128*np.finfo(float).eps*max(abs(current_time), abs(next_knot), abs(step), controls.minimum_dt)
                    if distance < controls.minimum_dt-roundoff:
                        raise ValueError("rigid translation knot is too close to current time")
                    # Snap only to the nearest combined knot; processing later
                    # histories must never enlarge a step across an earlier one.
                    step = distance if abs(step-distance) <= roundoff else min(step, distance)
                    if step < distance and distance-step < controls.minimum_dt:
                        count = max(1, int(np.ceil(max(0., distance-roundoff)/transient.dt)))
                        step = distance/count
                        if step < controls.minimum_dt-roundoff:
                            raise ValueError("tool/load knot remainder cannot fit dt and minimum_dt")
                else:
                    # End time need not be a registered history knot. Land on
                    # it with the same arithmetic allowance, rather than send
                    # an accumulation-roundoff-sized final step to Newton.
                    roundoff = 128*np.finfo(float).eps*max(abs(current_time), abs(transient.t_end), abs(step), controls.minimum_dt)
                    if abs(step-remaining) <= roundoff:
                        step = remaining
                if current_time+step <= current_time:
                    raise ValueError("prescribed-tool step cannot advance representable physical time")
            end_truncated = (
                remaining < transient.dt
                - 128.0*np.finfo(float).eps*max(1.0, abs(transient.t_end))
                and not any(transient.t_end == point[0] for h in tool_histories for point in h.points)
                and (nodal_load_history is None or not any(
                    transient.t_end == point[0]
                    for point in nodal_load_history.points
                ))
            )
            local_cutbacks = 0
            event_targets = ()
            event_seed = None
            event_localizations = 0
            last_event_failure = None
            plane_stage_evaluations=0
            plane_endpoint_requests=0
            plane_localized_boundary=False
            plane_endpoint_cache={}

            def consume_plane_stage():
                nonlocal plane_stage_evaluations,plane_total_stage_evaluations
                if plane_stage_evaluations>=controls.maximum_plane_event_stage_evaluations:
                    raise RuntimeError(f'plane Newmark event stage-work budget exhausted at t={current_time:.9g}')
                plane_stage_evaluations+=1
                plane_total_stage_evaluations+=1

            def plane_endpoint_gaps(duration):
                """Actual equilibrated endpoint, always from the committed origin."""
                nonlocal plane_endpoint_requests,plane_total_endpoint_requests
                plane_endpoint_requests+=1
                plane_total_endpoint_requests+=1
                if plane_endpoint_requests>controls.maximum_plane_event_endpoint_evaluations:
                    raise RuntimeError(f'plane Newmark event endpoint budget exhausted at t={current_time:.9g}')
                if duration in plane_endpoint_cache:
                    return plane_endpoint_cache[duration].copy()
                predictor_h=q+duration*vred+duration*duration*(.5-beta)*acceleration
                a_h=np.zeros_like(q)
                try:
                    for _ in range(controls.maximum_newton_iterations):
                        check_cancel('plane Newmark event stage evaluation')
                        consume_plane_stage()
                        q_h=predictor_h+beta*duration*duration*a_h
                        full_h,f_h,k_h,_=evaluate(q_h,current_time+duration,tangent=True)
                        r_h=Mred@a_h+f_h
                        scale_h=max(1.,float(np.linalg.norm(Mred@a_h)),float(np.linalg.norm(f_h)))
                        if np.linalg.norm(r_h)<=controls.equilibrium_tolerance*scale_h:
                            gaps_h=(np.asarray(manager.evaluate(full_h,tangent=False)[2].normalized_weighted_gap_m)
                                if normal_skin_path is not None else
                                np.asarray(pressure_reference+pressure_gradient@q_h).reshape(-1))
                            plane_endpoint_cache[duration]=gaps_h.copy()
                            return gaps_h
                        jac_h=(Mred+beta*duration*duration*(T.T@k_h@T)).tocsr()
                        delta_a=np.asarray(spsolve(jac_h,-r_h)).reshape(-1)
                        if not np.isfinite(delta_a).all():
                            break
                        a_h+=delta_a
                        discard_active_state_candidate(committed_states)
                finally:
                    discard_active_state_candidate(committed_states)
                raise EventNeedsCutback('plane Newmark endpoint family did not equilibrate')

            while True:
                last_event_failure = None
                predictor = q + step * vred + step * step * (0.5 - beta) * acceleration
                velocity_predictor = vred + step * (1.0 - gamma) * acceleration
                # In event/tool profiles iterate on acceleration as well as its
                # Newmark position. Subtracting two positions after each solve
                # amplifies endpoint roundoff by 1/h² at short event steps.
                # Always start from the same predictor, including at a requested
                # checkpoint boundary; the localized position is not a separate
                # numerical initialization for the same physical endpoint.
                candidate = predictor.copy() if controls.s3_event_subdivision_dev or controls.q4_plane_event_subdivision_dev or event_seed is None else event_seed.copy()
                event_acceleration = np.zeros_like(candidate)
                event_seed = None
                converged = False
                requested_event_step = None
                for iteration in range(1, controls.maximum_newton_iterations + 1):
                    check_cancel("transient contact Newton evaluation")
                    if controls.q4_plane_event_subdivision_dev or controls.allow_normal_s3_skin_hermite_dev:
                        consume_plane_stage()
                        candidate=predictor+beta*step*step*event_acceleration
                    if controls.equilibrated_newmark_acceleration_unknown_dev:
                        candidate=predictor+beta*step*step*event_acceleration
                    full, internal, tangent, trial = evaluate(
                        candidate, current_time+step, tangent=True,
                    )
                    candidate_acceleration = (event_acceleration.copy() if controls.equilibrated_newmark_acceleration_unknown_dev or controls.s3_event_subdivision_dev or controls.q4_plane_event_subdivision_dev or controls.allow_normal_s3_skin_hermite_dev or tool_histories or free_body is not None
                                              else (candidate - predictor) / (beta * step * step))
                    residual = Mred @ candidate_acceleration + internal
                    scale = max(1.0, float(np.linalg.norm(Mred @ candidate_acceleration)),
                                float(np.linalg.norm(internal)))
                    if float(np.linalg.norm(residual)) <= controls.equilibrium_tolerance * scale:
                        contact_result = manager.evaluate(full, tangent=False)[2]
                        if normal_skin_path is not None:
                            contact_result,failure=normal_skin_path.overlap_endpoint(manager,full,contact_result,
                                cancel=lambda:check_cancel('normal S3 candidate matched gap'))
                            if failure is not None:
                                last_event_failure=failure
                                break
                        candidate_velocity = velocity_predictor + gamma * step * candidate_acceleration
                        if s3_descriptor:
                            rate_touch_targets = event_targets
                            candidate_velocity, candidate_acceleration = (
                                _consistent_s3_descriptor_rates(
                                    model, T, Mred, descriptor_basis, full,
                                    candidate_velocity, candidate_acceleration,
                                    tangent, committed_states,
                                    controls.s3_descriptor_difference_step_s,
                                    one_sided_manager=(manager if controls.s3_one_sided_rates_dev else None),
                                    maximum_iterations=controls.maximum_newton_iterations,
                                    event_touch_indices=(event_targets if controls.s3_event_subdivision_dev else ()),
                                    event_gap_tolerance=controls.s3_event_gap_tolerance_m,
                                )
                            )
                        if _directional_selected(controls):
                            candidate_velocity,candidate_acceleration=recover_directional_rates(
                                manager,full,T,Mred,descriptor_basis,tangent,candidate_velocity,
                                candidate_acceleration,vred,step,contact_result)
                        elif rotational_descriptor:
                            candidate_velocity,candidate_acceleration=recover_rates(
                                T,Mred,descriptor_basis,tangent,candidate_velocity,
                                candidate_acceleration,vred,step,contact_result)
                        full_velocity = np.asarray(T @ candidate_velocity).reshape(-1)
                        if path_adapter is None:
                            if controls.allow_multiple_deformable_beam_s3_dev:
                                from .multiple_beam_s3_transient import multiple_hermite_gap_bounds
                                swept_gap,beam_path_admissible,_visited,pair_swept_gaps=multiple_hermite_gap_bounds(manager,
                                    previous_full,full,previous_velocity,full_velocity,step,
                                    maximum_intervals=controls.maximum_beam_swept_subdivisions,
                                    maximum_depth=settings.maximum_swept_subdivisions,
                                    check_cancel=lambda:check_cancel('multiple beam-S3 complete physical paths'))
                            elif controls.allow_deformable_beam_s3_dev or rotational_descriptor:
                                from .beam_s3_transient import hermite_gap_bounds
                                swept_gap,beam_path_admissible,_visited=hermite_gap_bounds(manager,
                                    previous_full,full,previous_velocity,full_velocity,step,
                                    maximum_intervals=controls.maximum_beam_swept_subdivisions,
                                    maximum_depth=settings.maximum_swept_subdivisions,
                                    check_cancel=lambda:check_cancel('beam-S3 complete physical path'))
                            elif controls.allow_free_shell_sphere_dev:
                                from .shell_sphere_temporal_bounds import q4_sphere_hermite_gap_bounds
                                side_admissible,side_splits=_free_shell_sphere_path_on_selected_side(
                                    manager,previous_full,full,previous_velocity,full_velocity,step,
                                    controls.maximum_beam_swept_subdivisions,check_cancel)
                                bounds=q4_sphere_hermite_gap_bounds(manager.child,
                                    previous_full[:size],full[:size],previous_velocity[:size],full_velocity[:size],step,
                                    sphere_start_point=manager.reference_point+previous_full[-3:],
                                    sphere_end_point=manager.reference_point+full[-3:],
                                    sphere_start_velocity=previous_velocity[-3:],sphere_end_velocity=full_velocity[-3:],
                                    tolerance_m=controls.beam_swept_gap_tolerance_m,
                                    maximum_subdivisions=controls.maximum_beam_swept_subdivisions-side_splits,
                                    check_cancel=lambda:check_cancel('free shell sphere complete-path bounds'))
                                swept_gap=bounds.lower_gap
                                beam_path_admissible=side_admissible and bounds.lower_gap>=-manager.gap_tolerance
                            elif controls.allow_classical_beam_dev:
                                if free_body is not None:
                                    from .sphere_temporal_bounds import beam_sphere_hermite_gap_bounds
                                    bounds = beam_sphere_hermite_gap_bounds(manager.child,
                                        previous_full[:size], full[:size], previous_velocity[:size], full_velocity[:size], step,
                                        sphere_start_point=manager.child.reference_point+previous_full[-3:],
                                        sphere_end_point=manager.child.reference_point+full[-3:],
                                        sphere_start_velocity=previous_velocity[-3:], sphere_end_velocity=full_velocity[-3:],
                                        tolerance_m=controls.beam_swept_gap_tolerance_m,
                                        maximum_subdivisions=controls.maximum_beam_swept_subdivisions,
                                        check_cancel=lambda: check_cancel("free sphere complete-path bounds"))
                                    swept_gap, beam_path_admissible = bounds.lower_gap, bounds.lower_gap >= -manager.gap_tolerance
                                else:
                                    swept_gap, beam_path_admissible = _beam_complete_path_gaps(
                                        manager,previous_full,full,previous_velocity,full_velocity,step,
                                        controls, lambda: check_cancel("transient beam complete-path bounds"),
                                        translations, current_time,
                                    )
                            elif controls.allow_multiple_shell_planes_dev:
                                swept_gap,beam_path_admissible,pair_swept_gaps=_multiple_shell_plane_path_gaps(
                                    manager,previous_full,full,previous_velocity,full_velocity,step,
                                    lambda:check_cancel('multiple shell planes path'))
                            elif controls.allow_guided_shell_pair_dev:
                                swept_gap=guided_shell_pair_path_minimum_gap(manager,
                                    previous_full,full,previous_velocity,full_velocity,step,
                                    lambda:check_cancel('guided shell pair physical path'))
                            elif normal_skin_path is not None:
                                swept_gap, failure = normal_skin_path.overlap_sweep(manager,
                                    previous_full,full,previous_velocity,full_velocity,step,
                                    cancel=lambda:check_cancel('normal S3 skin physical path'),
                                    event_gap_tolerance_m=controls.plane_event_gap_tolerance_m)
                                if failure is not None:
                                    last_event_failure = failure
                                    break
                            elif pressure_contract is not None:
                                from .shell_pair_hermite_sweep import shell_pair_hermite_lower
                                skins = pressure_contract.skin_path(manager, previous_full, full,
                                    previous_velocity, full_velocity)
                                target_ids = manager.law.target_face_indices[0]
                                lowers = []
                                for source_ids in manager.law.source_face_indices:
                                    args = [skin[source_ids if i in (0,1,4,5) else target_ids]
                                            for i, skin in enumerate(skins)]
                                    sweep = shell_pair_hermite_lower(*args, step,
                                        maximum_panels=settings.maximum_gap_triangles,
                                        maximum_depth=settings.maximum_swept_subdivisions,
                                        gap_tolerance_m=manager.gap_tolerance,
                                        cancel=lambda:check_cancel('fixed-projection S3 physical path'))
                                    if not sweep.success:
                                        last_event_failure = 'fixed-projection S3 physical path unresolved: ' + sweep.detail
                                        break
                                    lowers.append(sweep.lower_gap)
                                if len(lowers) != len(manager.law.source_face_indices):
                                    break # Bounded physical-proof failure requests ordinary step cutback.
                                swept_gap = min(lowers)
                            else:
                                swept_gap = _minimum_hermite_gap(
                                    manager, previous_full, full,
                                    previous_velocity, full_velocity, step,
                                )
                        else:
                            vk_chord = path_adapter.chord(q, candidate)
                            if (np.linalg.norm(path_adapter.endpoint_force(
                                    candidate, load_at(current_time+step),
                                    None if vk_chord is None else vk_chord[1])-internal)
                                    > controls.equilibrium_tolerance*scale):
                                raise ValueError("S3 reference-linear path disagrees with endpoint work")
                            candidate_path, swept_gap = path_adapter.evaluate(
                                q, candidate, vred, step, load_at(current_time),
                            )
                            if path_adapter.memo.get('force') is not None:
                                path_adapter.check_interior(candidate_path, controls)
                            path_adapter.check_endpoints(candidate_path, q, candidate, T, controls, scale)
                            if controls.s3_event_subdivision_dev:
                                event_gaps = path_adapter.reference_gap+path_adapter.gradient @ candidate
                                start_gaps = path_adapter.reference_gap+path_adapter.gradient @ q
                                continuation = _event_continuation_segments(
                                    candidate_path,step,event_boundary_support,start_gaps,
                                    controls.s3_event_time_tolerance_s,controls.s3_event_gap_tolerance_m,
                                )
                                transitions = tuple(segment.end for segment in continuation[:-1])
                                early = tuple(s for s in transitions
                                              if (1.-s)*step > controls.s3_event_time_tolerance_s)
                                event_matches = (event_targets and np.all(
                                    np.abs(event_gaps[list(event_targets)]) <= controls.s3_event_gap_tolerance_m))
                                if transitions and (early or not event_matches):
                                    if event_localizations >= controls.maximum_event_localizations:
                                        raise RuntimeError("S3 event localization budget exhausted before commit")
                                    event_localizations += 1
                                    first, second = continuation[:2]
                                    changed = tuple(i for i,(left,right) in enumerate(zip(first.active, second.active))
                                                    if left != right)
                                    try:
                                        event = localize_branch_event(
                                            path_adapter.structural, Mred.toarray(), path_adapter.gradient,
                                            path_adapter.reference_gap, path_adapter.weights,
                                            path_adapter.force_offset, q, vred, acceleration,
                                            load_at(current_time), load_at(current_time+step), step,
                                            first.active, changed, minimum_dt=controls.minimum_dt,
                                            time_tolerance=controls.s3_event_time_tolerance_s,
                                            gap_tolerance=controls.s3_event_gap_tolerance_m,
                                            maximum_iterations=controls.maximum_event_root_iterations,
                                            check_cancel=check_cancel,
                                            null_basis=path_adapter.null_basis,
                                        )
                                    except EventNeedsCutback as exc:
                                        event_evaluations += exc.endpoint_evaluations
                                        last_event_failure = str(exc)
                                        if exc.diagnostics is not None:
                                            last_event_failure += f"; {exc.diagnostics}"
                                        break
                                    event_evaluations += event.endpoint_evaluations
                                    event_targets = event.support_indices
                                    if step-event.duration > 64*np.finfo(float).eps*step:
                                        requested_event_step = event.duration
                                        event_seed = event.position
                                        break
                                    if early:
                                        break
                                if event_targets and not np.all(
                                        np.abs(event_gaps[list(event_targets)]) <= controls.s3_event_gap_tolerance_m):
                                    raise RuntimeError("reequilibrated S3 event exceeds its dimensional gap budget")
                                if not event_targets:
                                    gap_rate = path_adapter.gradient @ candidate_velocity
                                    rate_budget = 64*np.finfo(float).eps*(np.abs(path_adapter.gradient) @ np.abs(candidate_velocity))
                                    touched = np.abs(event_gaps) <= controls.s3_event_gap_tolerance_m
                                    approaching = np.sign(start_gaps)*gap_rate < -rate_budget
                                    near_boundary = tuple(np.flatnonzero(touched & approaching
                                        & (np.abs(start_gaps) > controls.s3_event_gap_tolerance_m)).tolist())
                                    if near_boundary:
                                        if event_localizations >= controls.maximum_event_localizations:
                                            raise RuntimeError("S3 event localization budget exhausted before commit")
                                        event_localizations += 1
                                        try:
                                            boundary = localize_branch_event(
                                                path_adapter.structural,Mred.toarray(),path_adapter.gradient,
                                                path_adapter.reference_gap,path_adapter.weights,
                                                path_adapter.force_offset,q,vred,acceleration,
                                                load_at(current_time),load_at(current_time+step),step,
                                                continuation[0].active,near_boundary,
                                                minimum_dt=controls.minimum_dt,
                                                time_tolerance=controls.s3_event_time_tolerance_s,
                                                gap_tolerance=controls.s3_event_gap_tolerance_m,
                                                maximum_iterations=controls.maximum_event_root_iterations,
                                                check_cancel=check_cancel,null_basis=path_adapter.null_basis,
                                            )
                                        except EventNeedsCutback as exc:
                                            # A nearby endpoint is not a resolved event. Preserve
                                            # ordinary truncation truth when the time bracket fails.
                                            event_evaluations += exc.endpoint_evaluations
                                        else:
                                            event_evaluations += boundary.endpoint_evaluations
                                            event_targets = boundary.support_indices
                                            if step-boundary.duration > controls.s3_event_time_tolerance_s:
                                                requested_event_step = boundary.duration
                                                event_seed = boundary.position
                                                break
                        if controls.q4_plane_event_subdivision_dev:
                            start_state=_plane_event_state(pressure_reference,pressure_gradient,q,vred,
                                controls.plane_event_gap_tolerance_m)
                            g_start=np.asarray(start_state['normalized_gaps_m'])
                            rate_start=np.asarray(start_state['gap_rates_m_s'])
                            g_end=np.asarray(pressure_reference+pressure_gradient@candidate).reshape(-1)
                            rate_end=np.asarray(pressure_gradient@candidate_velocity).reshape(-1)
                            phases=np.asarray(start_state['departing_phase'])
                            extrema=hermite_branch_extrema(g_start,g_end,rate_start,rate_end,step)
                            tolerance=controls.plane_event_gap_tolerance_m
                            unresolved=phases==0
                            active=phases<0
                            active[unresolved]=(extrema.maximum_gaps[unresolved]<=tolerance)
                            near=np.abs(g_start)<=tolerance
                            only_closed=(extrema.maximum_gaps<=tolerance)&(extrema.minimum_gaps<-tolerance)
                            only_open=(extrema.minimum_gaps>=-tolerance)&(extrema.maximum_gaps>tolerance)
                            # A tiny initial rate is not a certificate of the
                            # first departure outside the declared gap band.
                            active[near&only_closed]=True
                            active[near&only_open]=False
                            bounds=branch_path_admissible(g_start,g_end,rate_start,rate_end,step,
                                active=active,gap_tolerance=tolerance)
                            if event_targets and np.any(np.abs(g_end[list(event_targets)])>tolerance):
                                last_event_failure='reequilibrated plane endpoint exceeds event gap tolerance'
                                break
                            if not bounds.admissible:
                                required=tuple(int(i) for i in np.flatnonzero(np.where(active,-g_end,g_end)<-tolerance))
                                if not required:
                                    last_event_failure='hidden pressure-mode crossing has no endpoint bracket'
                                    break
                                event_localizations+=1
                                if event_localizations>controls.maximum_event_localizations:
                                    last_event_failure='plane event localization count exhausted'
                                    break
                                remaining_queries=controls.maximum_plane_event_endpoint_evaluations-plane_endpoint_requests
                                if remaining_queries<2:
                                    raise RuntimeError('plane Newmark event endpoint budget exhausted')
                                if step<=controls.minimum_dt or step<=controls.plane_event_time_tolerance_s:
                                    last_event_failure='plane event is below the resolved duration budget'
                                    break
                                try:
                                    boundary=localize_endpoint_event(plane_endpoint_gaps,step,active,required,
                                        minimum_dt=controls.minimum_dt,
                                        time_tolerance=controls.plane_event_time_tolerance_s,
                                        gap_tolerance=tolerance,maximum_evaluations=remaining_queries,
                                        batch_front=controls.batch_plane_event_fronts_dev,
                                        check_cancel=lambda context:check_cancel(context))
                                except EventNeedsCutback as exc:
                                    event_evaluations+=exc.endpoint_evaluations
                                    last_event_failure=str(exc)
                                    break
                                event_evaluations+=boundary.endpoint_evaluations
                                if not boundary.duration<step:
                                    last_event_failure='localized plane endpoint lacks complete branch certificate'
                                    break
                                requested_event_step=boundary.duration
                                event_targets=boundary.support_indices
                                plane_localized_boundary=True
                                break
                            arrivals=tuple(int(i) for i in np.flatnonzero((np.abs(g_end)<=tolerance)&(np.abs(g_start)>tolerance)))
                            if not event_targets:
                                event_targets=arrivals
                        if normal_skin_path is not None:
                            g_start=normal_committed_gaps
                            g_end=np.asarray(contact_result.normalized_weighted_gap_m)
                            tolerance=controls.plane_event_gap_tolerance_m
                            active=(g_start<-tolerance)
                            near=np.abs(g_start)<=tolerance
                            active[near]=g_end[near]<0.
                            if event_targets and np.any(np.abs(g_end[list(event_targets)])>tolerance):
                                last_event_failure='normal S3 reequilibrated endpoint exceeds pressure event tolerance'
                                break
                            required=tuple(int(i) for i in np.flatnonzero(
                                ((g_start>tolerance)&(g_end<-tolerance))
                                |((g_start<-tolerance)&(g_end>tolerance))))
                            if required:
                                event_localizations+=1
                                if event_localizations>controls.maximum_event_localizations:
                                    last_event_failure='normal S3 pressure event localization count exhausted'
                                    break
                                remaining_queries=controls.maximum_plane_event_endpoint_evaluations-plane_endpoint_requests
                                if remaining_queries<2 or step<=max(controls.minimum_dt,controls.plane_event_time_tolerance_s):
                                    last_event_failure='normal S3 pressure event duration/query budget exhausted'
                                    break
                                try:
                                    boundary=localize_endpoint_event(plane_endpoint_gaps,step,active,required,
                                        minimum_dt=controls.minimum_dt,
                                        time_tolerance=controls.plane_event_time_tolerance_s,
                                        gap_tolerance=tolerance,maximum_evaluations=remaining_queries,
                                        batch_front=True,check_cancel=check_cancel)
                                except EventNeedsCutback as exc:
                                    last_event_failure=str(exc)
                                    break
                                if not boundary.duration<step:
                                    last_event_failure='normal S3 pressure endpoint lacks complete phase certificate'
                                    break
                                requested_event_step=boundary.duration
                                event_targets=boundary.support_indices
                                break
                            pressure_proof=normal_skin_path.pressure_sweep(manager,
                                previous_full,full,previous_velocity,full_velocity,step,
                                active=active,gap_tolerance_m=tolerance,
                                cancel=lambda:check_cancel('normal S3 mortar pressure path'),
                                polygon_bounds=controls.normal_s3_polygon_pressure_bounds_dev)
                            if not pressure_proof.success:
                                last_event_failure='normal S3 mortar pressure path '+pressure_proof.status+': '+pressure_proof.detail
                                break
                            if not event_targets:
                                event_targets=tuple(int(i) for i in np.flatnonzero(
                                    (np.abs(g_end)<=tolerance)&(np.abs(g_start)>tolerance)))
                        if (contact_result.gap_tolerance_met
                                and (beam_path_admissible if rotational_descriptor or controls.allow_deformable_beam_s3_dev or controls.allow_multiple_deformable_beam_s3_dev or controls.allow_classical_beam_dev or controls.allow_free_shell_sphere_dev or controls.allow_multiple_shell_planes_dev
                                     else swept_gap >= -manager.gap_tolerance)):
                            converged = True
                        break
                    if controls.equilibrated_newmark_acceleration_unknown_dev or controls.q4_plane_event_subdivision_dev or controls.allow_normal_s3_skin_hermite_dev:
                        jacobian=(Mred+beta*step*step*(T.T@tangent@T)).tocsr()
                        delta_acceleration=np.asarray(spsolve(jacobian,-residual)).reshape(-1)
                        if not np.isfinite(delta_acceleration).all():
                            break
                        event_acceleration+=delta_acceleration
                        discard_active_state_candidate(committed_states)
                        continue
                    jacobian = (Mred / (beta * step * step) + T.T @ tangent @ T).tocsr()
                    correction = np.asarray(spsolve(jacobian, -residual)).reshape(-1)
                    if not np.all(np.isfinite(correction)):
                        break
                    candidate += correction
                    if controls.s3_event_subdivision_dev or tool_histories or free_body is not None:
                        event_acceleration += correction / (beta * step * step)
                    discard_active_state_candidate(committed_states)
                if requested_event_step is not None:
                    discard_active_state_candidate(committed_states)
                    step = requested_event_step
                    end_truncated = False
                    continue
                if converged:
                    if (controls.s3_event_subdivision_dev and event_targets
                            and event_targets != rate_touch_targets):
                        candidate_velocity,candidate_acceleration = _consistent_s3_descriptor_rates(
                            model,T,Mred,descriptor_basis,full,candidate_velocity,candidate_acceleration,
                            tangent,committed_states,controls.s3_descriptor_difference_step_s,
                            one_sided_manager=manager,maximum_iterations=controls.maximum_newton_iterations,
                            event_touch_indices=event_targets,
                            event_gap_tolerance=controls.s3_event_gap_tolerance_m,
                        )
                    step_work = 0.0
                    if nodal_load_history is not None:
                        start_force = (nodal_load_history.factor(current_time)
                                       * external_load)
                        end_force = (nodal_load_history.factor(current_time+step)
                                     * external_load)
                        if path_adapter is None:
                            step_work = _hermite_linear_load_work(
                                previous_full, full, previous_velocity,
                                full_velocity, start_force, end_force, step,
                            )
                        else:
                            step_work = candidate_path.linear_load_work(
                                load_at(current_time), load_at(current_time+step),
                            )
                    check_cancel("transient contact step acceptance")
                    candidate_body_state = (manager.state(current_time+step, full, full_velocity,
                        np.asarray(T @ candidate_acceleration).reshape(-1), contact_result)
                        if free_body is not None else None)
                    if translations is not None:
                        added_work, added_error = translations.work(
                            previous_full, full, previous_velocity, full_velocity,
                            current_time, step, controls, lambda: check_cancel("prescribed tool work"))
                        check_cancel("transient contact tool-work acceptance")
                        next_tool_work = {name: tool_work[name]+added_work[name] for name in tool_work}
                        next_tool_error = {name: tool_error[name]+added_error[name] for name in tool_error}
                        if not all(np.isfinite(value) for value in (*next_tool_work.values(), *next_tool_error.values())):
                            raise RuntimeError("nonfinite accumulated prescribed tool work")
                    if controls.allow_q4_plastic_dev:
                        next_material_packet,trial=sealed_material_candidate(full,trial)
                        check_cancel('Q4 plastic material-state acceptance')
                    if controls.q4_plane_event_subdivision_dev:
                        next_plane_state=_plane_event_state(pressure_reference,pressure_gradient,
                            candidate,candidate_velocity,controls.plane_event_gap_tolerance_m)
                        check_cancel('plane Newmark event-state acceptance')
                    candidate_reaction = (accepted_reaction(full, current_time+step, candidate_acceleration)
                                          if controls.save_constraint_reactions_dev else None)
                    committed_states = commit_state_candidate(committed_states, trial)
                    if controls.allow_q4_plastic_dev:
                        material_packet=next_material_packet
                    if translations is not None:
                        tool_work, tool_error = next_tool_work, next_tool_error
                    if controls.s3_event_subdivision_dev:
                        event_boundary_support = event_targets
                    if controls.s3_event_subdivision_dev and event_targets:
                        localized_events.append(float(current_time+step))
                        end_truncated = False
                    if controls.q4_plane_event_subdivision_dev:
                        plane_state=next_plane_state
                        if event_targets:
                            localized_events.append(float(current_time+step))
                            if plane_localized_boundary:
                                end_truncated=False
                    if normal_skin_path is not None:
                        normal_committed_gaps=np.asarray(contact_result.normalized_weighted_gap_m).copy()
                        if event_targets:
                            localized_events.append(float(current_time+step))
                    if path_adapter is not None:
                        for event_time in _accepted_path_transition_times(
                                last_path_active, candidate_path, current_time, step):
                            if not path_transitions or event_time != path_transitions[-1]:
                                path_transitions.append(event_time)
                        last_path_active = candidate_path.segments[-1].active
                    accumulated_work += step_work
                    if rotational_descriptor:
                        rotational_previous_velocity=vred.copy()
                        rotational_last_dt=step
                    q = candidate
                    vred = candidate_velocity
                    acceleration = candidate_acceleration
                    current_time += step
                    last_step_was_end_truncated = end_truncated
                    previous_full = full.copy()
                    previous_velocity = np.asarray(T @ vred).reshape(-1)
                    accepted_steps += 1
                    pending_swept_gap = min(pending_swept_gap, swept_gap)
                    if controls.allow_multiple_shell_planes_dev or controls.allow_multiple_deformable_beam_s3_dev:
                        pending_pair_gaps={name:min(value,pair_swept_gaps[name])
                                           for name,value in pending_pair_gaps.items()}
                    pending_residual = max(pending_residual,
                                           float(np.linalg.norm(residual)))
                    if (accepted_steps % transient.save_every == 0
                            or current_time >= transient.t_end - 1.0e-14
                            * max(1.0, transient.t_end)):
                        times.append(current_time)
                        positions.append(full.copy())
                        velocities.append(np.asarray(T @ vred).reshape(-1))
                        accelerations.append(np.asarray(T @ acceleration).reshape(-1))
                        if controls.save_constraint_reactions_dev:
                            constraint_reactions.append(candidate_reaction)
                        contacts.append(contact_result)
                        if nodal_load_history is None:
                            # Constant dead-load path integral is exact.
                            accumulated_work = float(
                                external_load @ (full - full_initial)
                            )
                        external_work.append(accumulated_work)
                        tool_works.append(tool_work.copy())
                        tool_errors.append(tool_error.copy())
                        if free_body is not None:
                            body_states.append({free_body.body_name: candidate_body_state})
                        if controls.save_q4_material_history_dev:
                            material_history.append(committed_states)
                        swept_gaps.append(pending_swept_gap)
                        if controls.allow_multiple_shell_planes_dev or controls.allow_multiple_deformable_beam_s3_dev:
                            pair_swept_history.append(pending_pair_gaps.copy())
                            pending_pair_gaps={name:float('inf') for name in pending_pair_gaps}
                        residuals.append(pending_residual)
                        pending_swept_gap = float("inf")
                        pending_residual = 0.0
                    iterations.append(iteration)
                    break
                discard_active_state_candidate(committed_states)
                event_targets = ()
                plane_localized_boundary=False
                cutback_diagnostics.append(dict(time_s=float(current_time),
                    attempted_dt_s=float(step),reason=(last_event_failure
                        or 'equilibrium or physical-gap acceptance failed'),
                    plane_endpoint_requests=plane_endpoint_requests
                        if controls.q4_plane_event_subdivision_dev or controls.allow_normal_s3_skin_hermite_dev else 0,
                    plane_stage_evaluations=plane_stage_evaluations
                        if controls.q4_plane_event_subdivision_dev or controls.allow_normal_s3_skin_hermite_dev else 0))
                event_seed = None
                local_cutbacks += 1
                cutbacks += 1
                step *= 0.5
                if local_cutbacks > controls.maximum_cutbacks or step < controls.minimum_dt:
                    raise RuntimeError(
                        f"transient contact step failed at t={current_time:.9g} s "
                        f"after {local_cutbacks} cutbacks; committed state retained"
                        +(f"; event: {last_event_failure}" if last_event_failure else "")
                    )
    if emit_restart_checkpoint and (accepted_steps == 0 or last_step_was_end_truncated):
        raise ValueError(
            "transient contact checkpoint requires an accepted untruncated final time step"
        )
    check_cancel("transient contact result export")
    checkpoint = (
        _new_restart_checkpoint(
            model_fingerprint, analysis_fingerprint, current_time,
            q, vred, acceleration, accepted_steps,
            accumulated_work if nodal_load_history is not None else None,
            last_path_active if path_adapter is not None else None,
            event_boundary_support if controls.s3_event_subdivision_dev else None,
            tool_work if translations is not None else None,
            tool_error if translations is not None else None,
            free_body.body_name if free_body is not None else None,
            material_packet,
            plane_state,
            pressure_features,
            controls.allow_guided_shell_pair_dev,
            controls.allow_fixed_projection_s3_target_dev,
            controls.allow_normal_s3_skin_hermite_dev,
            controls.normal_s3_polygon_pressure_bounds_dev,
            controls.allow_deformable_beam_s3_dev,
            rotational_descriptor,
            rotational_previous_velocity if rotational_descriptor else None,
            rotational_last_dt if rotational_descriptor else 0.,
            _directional_selected(controls),
            controls.allow_equilibrated_directional_beam_s3_dev,
            controls.allow_multiple_deformable_beam_s3_dev,
        ) if emit_restart_checkpoint else None
    )
    return TransientContactResults(
        profile, np.asarray(times), np.asarray(positions)[:, :size],
        np.asarray(velocities)[:, :size], np.asarray(accelerations)[:, :size], tuple(contacts),
        np.asarray(swept_gaps), np.asarray(residuals), cutbacks, tuple(iterations),
        np.asarray(external_work), checkpoint, tuple(path_transitions),
        tuple(localized_events), (plane_total_endpoint_requests
            if controls.q4_plane_event_subdivision_dev or controls.allow_normal_s3_skin_hermite_dev else event_evaluations),
        tuple(tool_works) if translations is not None else (),
        tuple(tool_errors) if translations is not None else (),
        tuple(body_states),
        committed_states if controls.allow_q4_plastic_dev else {},tuple(material_history),
        plane_total_stage_evaluations,
        tuple(cutback_diagnostics),
        np.asarray(constraint_reactions) if controls.save_constraint_reactions_dev else np.empty((0, 0)),
        tuple(pair_swept_history),
    )
