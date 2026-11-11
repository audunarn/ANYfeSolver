"""Small elastic multi-Q4/multi-B2 panel with existing F10 point joints.

This is an experimental reference-linear owner. Shell and beam node identities
are explicit and separate. Reference modes require explicit physical mass.
Explicit nodal footprints have a distinct reference-linear law; virgin supported
modes retain its normal-spin locking and incomplete mechanical qualification.
No line, geometric stiffness, nonlinear state or qualification is inferred.
"""

from __future__ import annotations

from functools import wraps
from dataclasses import dataclass, field
from hashlib import sha256
from math import fsum, isfinite
from numbers import Real
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, ArpackError
from scipy.sparse.csgraph import maximum_bipartite_matching

from ._b2_f10_f11_point_attachment import LAW_ID, compile_reference_point
from ._stiffened_shell_b2_nodal_footprint import (
    LAW_ID as FOOTPRINT_LAW_ID, B2NodalRigidFootprint,
    B2NodalRigidFootprintSelection, B2NodalRigidFootprintWrench,
    compile_footprints, checked_selection, selected_wrenches,
)
from ._stiffened_shell_b2_physical_rigid_footprint import (
    LAW_ID as PHYSICAL_FOOTPRINT_LAW_ID, B2PhysicalRigidFootprint,
    B2PhysicalRigidFootprintSelection, B2PhysicalRigidFootprintWrench,
    compile_physical_footprints, checked_selection as checked_physical_selection,
    selected_wrenches as physical_selected_wrenches,
)
from ._b2_physical_linear_definition import (
    PROFILE_ID, CompiledPhysicalB2Member, PhysicalB2Member,
    PhysicalB2OperatorCache,
)
from ._b2_physical_linear_batch_dev import (
    _load_value_snapshot, _checked_affine_projection as _project_affine_work,
    _accurate_sparse_residual, B2NumericalSolveFailure,
)
from .control import CancellationToken, SolveCancelled
from ._stiffened_shell_physical_q4_mass import physical_mass, POLICY as Q4_MASS_POLICY
from .e4_pl_s3_v2d_element import (
    MASS_POLICY_ID as S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1,
    FORMULATION_ID as S3_V2D_FORMULATION_ID,
)
from ._b2_physical_flexibility_linear import RepresentabilityError
from ._b2_loaded_point_trace_dev import LoadedPointMember, checked_sparse_action
from ._stiffened_shell_b2_affine_resolution import certify_affine
from ._b2_attachment_elimination_dev import prepare_attachment_elimination
from .e4_pl_element import _retire_q4_private_mesh_caches
from ._stiffened_shell_b2_assembly_diagnostics import active_capture, DiagnosticRefusal
from .elements import create_shell_element
from .fe_core import FEModel
from .linalg import MatrixClass, factorize_cached

B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV = "B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV"

class ExperimentalB2Q4PanelError(RuntimeError):
    """The experimental panel cannot be prepared or used."""


class _NumericalFailure(RuntimeError):
    """A checked sparse panel solution failed numerically."""


_MAX_DENSE_AUDIT_ROWS = 48
_MAX_DENSE_AUDIT_VALUES = 16_384


def _bounded_accurate_residual(matrix, solution, rhs):
    """Stored-equation residual and outward bound using exact products.

    Only selected explicitly loaded rows use this exceptional path. Fraction
    products/summation avoid FMA/fsum portability and underflow assumptions.
    Conversion to binary64 is correctly rounded; a full ULP and outward final
    rounding conservatively enclose the absolute rational residual. This does
    not certify element/assembly rounding or forward displacement accuracy.
    """
    if (not np.isfinite(matrix.data).all() or not np.isfinite(solution).all() or
            not np.isfinite(rhs).all()):
        raise B2NumericalSolveFailure("physical B2 residual is not representable")
    defect = _accurate_sparse_residual(matrix, solution, rhs, exact_products=True)
    magnitude = np.abs(defect)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rounding = np.nextafter(magnitude, np.inf)-magnitude
        upper = np.nextafter(magnitude+rounding, np.inf)
    return defect, upper


def _checked_affine_projection(rows, multi, singles, values: np.ndarray) -> np.ndarray:
    """Keep the panel's failure identity while sharing checked affine work."""
    return _project_affine_work(rows, multi, singles, values,
                                failure_type=_NumericalFailure, label="panel")


def _scaled_point_kkt(reduced_k: sparse.spmatrix,
                      reduced_c: sparse.spmatrix,
                      multiplier_compliance: sparse.spmatrix | None = None) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Representable congruence for the elastic point-constrained KKT.

    A zero-energy primal coordinate is valid when a constraint anchors it;
    its scale then follows that constraint column. This owner admits no
    negative-energy primal diagonal or unanchored zero-energy coordinate.
    """
    diagonal = reduced_k.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal < 0.):
        raise ExperimentalB2Q4PanelError(
            "panel factorization requires nonnegative finite primal diagonal")
    primal_scale = np.empty(len(diagonal))
    positive = diagonal > 0.
    with np.errstate(over="ignore", under="ignore", divide="ignore",
                     invalid="ignore"):
        primal_scale[positive] = 1./np.sqrt(diagonal[positive])
        if np.any(~positive):
            # For a positive-semidefinite elastic K, a zero diagonal has a
            # zero row. An off-diagonal term instead signals invalid data.
            zero_rows = reduced_k.tocsr()[np.flatnonzero(~positive)]
            if np.any(zero_rows.data != 0.):
                raise ExperimentalB2Q4PanelError(
                    "zero-energy primal row has nonzero stiffness coupling")
            column_max = np.asarray(
                abs(reduced_c).max(axis=0).toarray()).reshape(-1)
            if (not np.isfinite(column_max[~positive]).all() or
                    np.any(column_max[~positive] <= 0.)):
                raise ExperimentalB2Q4PanelError(
                    "unanchored zero-energy primal coordinate")
            primal_scale[~positive] = 1./column_max[~positive]
        if (not np.isfinite(primal_scale).all() or
                np.any(primal_scale <= 0.)):
            raise ExperimentalB2Q4PanelError(
                "panel factorization scaling is not representable")
        scaled_rows = reduced_c @ sparse.diags(primal_scale)
        row_max = np.asarray(abs(scaled_rows).max(axis=1).toarray()).reshape(-1)
        if multiplier_compliance is not None:
            diagonal_h = multiplier_compliance.diagonal()
            if (multiplier_compliance.shape != (reduced_c.shape[0],)*2 or
                    not np.isfinite(multiplier_compliance.data).all() or
                    np.any(diagonal_h < 0.)):
                raise ExperimentalB2Q4PanelError("finite nonnegative point compliance required")
            row_max = np.maximum(row_max, np.sqrt(diagonal_h))
        multiplier_scale = 1./row_max
        factor_scale = np.r_[primal_scale, multiplier_scale]
        kkt = sparse.bmat(
            ((reduced_k, reduced_c.T), (reduced_c, None if multiplier_compliance is None
                                      else -multiplier_compliance)), format="csc")
        scaling = sparse.diags(factor_scale, format="csc")
        scaled_kkt = (scaling @ kkt @ scaling).tocsc()
    kkt.sort_indices()
    scaled_kkt.sort_indices()
    if (not np.isfinite(factor_scale).all() or
            np.any(factor_scale <= 0.) or
            not np.isfinite(scaled_kkt.data).all() or
            kkt.nnz != scaled_kkt.nnz or
            not np.array_equal(kkt.indptr, scaled_kkt.indptr) or
            not np.array_equal(kkt.indices, scaled_kkt.indices) or
            np.any((kkt.data != 0.) & (scaled_kkt.data == 0.))):
        raise ExperimentalB2Q4PanelError(
            "panel factorization scaling is not representable")
    return scaled_kkt, factor_scale


def _require_independent_supported_point_rows(reduced_c: sparse.spmatrix) -> None:
    """Bound the dense audit; defer large numerical rank to KKT factorization.

    A matching proves only structural rank. The subsequent sparse KKT
    factorization must still reject numerical row dependence for larger models.
    """
    rows, columns = reduced_c.shape
    if rows > columns or not np.isfinite(reduced_c.data).all():
        raise ValueError("independent supported point rows required")
    if rows <= _MAX_DENSE_AUDIT_ROWS and rows*columns <= _MAX_DENSE_AUDIT_VALUES:
        if np.linalg.matrix_rank(reduced_c.toarray()) != rows:
            raise ValueError("independent supported point rows required")
        return
    pattern = reduced_c.tocsr(copy=True)
    pattern.eliminate_zeros()
    matching = maximum_bipartite_matching(pattern, perm_type="column")
    if np.any(matching < 0):
        raise ValueError("structurally independent supported point rows required")


@dataclass(frozen=True)
class Q4PanelShell:
    shell_id: str
    node_ids: tuple[int, int, int, int]
    E: float
    nu: float
    thickness_m: float
    reference_normal: tuple[float, float, float] = (0., 0., 1.)
    density_kg_m3: float | None = None


@dataclass(frozen=True)
class Q4B2Point:
    point_id: str
    shell_id: str
    member_id: str
    station: float
    offset_local_m: tuple[float, float, float] = (0., 0., 0.)


@dataclass(frozen=True)
class S3V2DPanelShell:
    """Explicit native S3 V2D elastic facet; reference mass requires modal opt-in."""

    shell_id: str
    node_ids: tuple[int, int, int]
    E: float
    nu: float
    thickness_m: float
    reference_normal: tuple[float, float, float]
    density_kg_m3: float | None = None


@dataclass(frozen=True)
class ExperimentalB2Q4PanelCapabilities:
    profile_id: str = PROFILE_ID
    point_law_id: str = LAW_ID
    status: str = "experimental"
    production_qualified: bool = False
    shell_domain: str = "multiple flat elastic Q4"
    beam_domain: str = "multiple physical B2 members"
    attachment: str = "F10 reference points only"
    operations: tuple[str, ...] = ("solve_linear", "solve_linear_many", "close")
    spectral_domain: str = "unavailable without explicit physical mass"
    body_load_domain: str = "unavailable without complete physical density/mass"
    point_trace_policy: str | None = None


@dataclass(frozen=True)
class ExperimentalB2LoadedPointCapabilities(ExperimentalB2Q4PanelCapabilities):
    point_trace_policy: str = B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV
    spectral_domain: str = "unsupported: loaded point trace inertia unverified"


@dataclass(frozen=True, eq=False)
class B2ReferenceBodyLoad:
    """Owned reference-linear body load; six global coordinates per node.

    Shell nodes precede beam nodes, each in the returned ID order. Acceleration
    is global in m/s2. This is a reference load, not finite-configuration gravity.
    """

    acceleration_m_s2: tuple[float, float, float]
    shell_node_ids: tuple[int, ...]
    beam_node_ids: tuple[int, ...]
    load: np.ndarray
    model_identity: str = ""
    policy: str = field(default="B2_E4_PL_REFERENCE_BODY_ACCELERATION_V1", init=False)
    production_qualified: bool = field(default=False, init=False)
    member_recovery: str = field(default="pass typed load for particular-load equilibrated fields; .load alone retains nodal-state constitutive recovery", init=False)

    def __post_init__(self):
        # Bytes-backed storage cannot be made writable, and never aliases the
        # owner's compiled map or the caller's acceleration.
        values = np.asarray(self.load, dtype=np.float64)
        object.__setattr__(self, "load", np.frombuffer(values.tobytes(), dtype=np.float64))
        object.__setattr__(self, "acceleration_m_s2", tuple(self.acceleration_m_s2))
        object.__setattr__(self, "shell_node_ids", tuple(self.shell_node_ids))
        object.__setattr__(self, "beam_node_ids", tuple(self.beam_node_ids))


@dataclass(frozen=True)
class B2LoadedMemberSections:
    """Static admissible section fields; not a new loaded kinematic state."""

    member_id: str
    stations: tuple[float, ...]
    resultants_left: tuple[tuple[float, ...], ...]
    resultants_right: tuple[tuple[float, ...], ...]
    strains_left: tuple[tuple[float, ...], ...]
    strains_right: tuple[tuple[float, ...], ...]
    endpoint_tractions_local: tuple[float, ...]
    policy: str = field(default="B2_REFERENCE_EQUILIBRATED_MEMBER_FIELDS_V1_DEV", init=False)
    production_qualified: bool = field(default=False, init=False)


def _beam_body_map(member: CompiledPhysicalB2Member, frame: np.ndarray) -> np.ndarray:
    """Integrate physical section action before any rotary-mass cancellation."""
    section = member.definition.mass.matrix_at_reference_axis()[:, :3]
    abscissae, weights = np.polynomial.legendre.leggauss(4)
    stations = tuple(float((x+1.)/2.) for x in abscissae)
    trace = member.operator.trace(np.zeros(12), stations=stations)
    local = np.zeros((12, 3))
    with np.errstate(over="ignore", invalid="ignore"):
        for weight, jacobian in zip(weights, trace.jacobian):
            local += member.definition.length_m*(weight/2.)*(jacobian.T@section)
        result = frame.T@local@frame[:3, :3]
    if not np.isfinite(result).all():
        raise RepresentabilityError("physical B2 body-load map is not representable")
    return result


@dataclass(frozen=True)
class ExperimentalB2Q4FootprintCapabilities(ExperimentalB2Q4PanelCapabilities):
    footprint_law_id: str = FOOTPRINT_LAW_ID
    footprints: tuple[B2NodalRigidFootprint, ...] = ()

@dataclass(frozen=True)
class B2PhysicalRigidFootprintCapabilities(ExperimentalB2Q4PanelCapabilities):
    footprint_law_id: str = PHYSICAL_FOOTPRINT_LAW_ID
    footprints: tuple[B2PhysicalRigidFootprint, ...] = ()
    normal_spin: str = "independent; normal torque through noncollinear translation arms"


@dataclass(frozen=True)
class ExperimentalB2MixedPanelCapabilities(ExperimentalB2Q4PanelCapabilities):
    shell_domain: str = "explicit flat elastic Q4/native S3 V2D panel"
    point_law_id: tuple[str, ...] = ()
    shell_formulations: tuple[tuple[str, str], ...] = ()
    point_laws: tuple[tuple[str, str], ...] = ()
    footprint_law_id: str | None = None
    footprints: tuple[B2NodalRigidFootprint, ...] = ()


@dataclass(frozen=True)
class ExperimentalB2MixedReferenceModalCapabilities(ExperimentalB2MixedPanelCapabilities):
    inertia_policy: str = S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1
    shell_mass_policies: tuple[tuple[str, str], ...] = ()
    s3_rotary_inertia: str = "omitted by S3 facet contributions; shared nodes may retain Q4 director inertia"
    beam_mass_policy: str = "declared physical B2 sectional inertia through the station trace"
    boundary_policy: str | None = None


B2_FREE_FREE_MODAL_ONLY_V1 = "B2_FREE_FREE_MODAL_ONLY_V1"


@dataclass(frozen=True)
class B2ReferenceModesRequest:
    """Elastic finite reference pairs; boundary policy belongs to the definition."""

    count: int = 3
    shift_hz: float = 1.
    max_iterations: int = 600

    def __post_init__(self):
        if (type(self.count) is not int or self.count <= 0 or
                type(self.max_iterations) is not int or self.max_iterations <= 0 or
                isinstance(self.shift_hz, (bool, np.bool_)) or
                not isinstance(self.shift_hz, (int, float, np.floating)) or
                not np.isfinite(self.shift_hz) or self.shift_hz <= 0.):
            raise ValueError("positive finite reference-mode controls required")


@dataclass(frozen=True)
class B2ReferenceModesResult:
    status: str
    frequencies_hz: tuple[float, ...] = ()
    physical_modes: tuple[tuple[float, ...], ...] = ()
    point_multiplier_modes: tuple[tuple[float, ...], ...] = ()
    equation_residuals: tuple[float, ...] = ()
    constraint_residuals: tuple[float, ...] = ()
    factorization_count: int = 0
    solve_count: int = 0
    failure_reason: str | None = None
    production_qualified: bool = False


@dataclass(frozen=True)
class B2FreeReferenceModesResult:
    """Six verified physical rigid fields and selected finite flexible pairs.

    This does not certify complete nullity or an eigenvalue error bound.
    All returned numerical collections own immutable scalar tuples.
    """

    status: str
    flexible: B2ReferenceModesResult
    rigid_origin_m: tuple[float, ...] = ()
    canonical_rigid_modes: tuple[tuple[float, ...], ...] = ()
    rigid_modes: tuple[tuple[float, ...], ...] = ()
    rigid_multiplier_modes: tuple[tuple[float, ...], ...] = ()
    rigid_kinetic_gram: tuple[tuple[float, ...], ...] = ()
    rigid_equation_residuals: tuple[float, ...] = ()
    rigid_constraint_residuals: tuple[float, ...] = ()
    rigid_flexible_mass_overlap: float = 0.
    spectral_shift_rad2_s2: float = 0.
    failure_reason: str | None = None
    policy: str = B2_FREE_FREE_MODAL_ONLY_V1
    production_qualified: bool = False


@dataclass(frozen=True)
class B2ModalConstraintRow:
    """Immutable provenance for one actual physical constraint/multiplier row.

    Rows follow C. Footprint gaps are global; point translation gaps use the
    master shell reference frame and point rotational gaps use the beam
    reference frame. Multipliers are work-conjugate covectors, not display-ready
    global wrenches. Frame columns give the actual basis in global coordinates.
    """

    row_index: int
    component: str
    point_id: str | None = None
    footprint_id: str | None = None
    shell_node_id: int | None = None
    beam_node_id: int | None = None
    coordinate_frame: str = "global"
    reference_frame_columns: tuple[tuple[float, ...], ...] = (
        (1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


@dataclass(frozen=True)
class B2FootprintReferenceModesResult(B2ReferenceModesResult):
    """Virgin descriptor modes of the reference-linear, normal-spin locked law.

    All multipliers follow constraint_rows. Inherited point_multiplier_modes
    contain only the true point prefix, never footprint multipliers.
    """

    footprint_law_id: str = FOOTPRINT_LAW_ID
    footprint_ids: tuple[str, ...] = ()
    constraint_multiplier_modes: tuple[tuple[float, ...], ...] = ()
    constraint_rows: tuple[B2ModalConstraintRow, ...] = ()
    normal_spin: str = "locked"
    mechanical_qualification: str = "incomplete"


@dataclass(frozen=True)
class ExperimentalB2Q4PanelResult:
    status: str
    displacements: tuple[float, ...] = ()
    point_multipliers: tuple[tuple[str, tuple[float, ...]], ...] = ()
    affine_reactions: tuple[float, ...] = ()
    point_gaps: tuple[tuple[str, tuple[float, ...]], ...] = ()
    beam_resultants_local: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...] = ()
    internal_strain_energy: float | None = None
    factorization_count: int | None = None
    solve_count: int | None = None
    failure_reason: str | None = None
    production_qualified: bool = False


@dataclass(frozen=True)
class ExperimentalB2Q4PanelSelectedCase:
    case_id: str
    selected_displacements: tuple[float, ...]
    selected_affine_reactions: tuple[float, ...]
    point_multipliers: tuple[tuple[str, tuple[float, ...]], ...]
    point_gaps: tuple[tuple[str, tuple[float, ...]], ...]
    beam_resultants_local: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...]
    internal_strain_energy: float


@dataclass(frozen=True)
class ExperimentalB2Q4PanelResponseEnvelope:
    """One selected physical coordinate's signed complete-case extrema.

    Displacements retain m/rad and affine reactions N/Nm according to the
    selected DOF. Equal extrema are governed by the first submitted case.
    """

    selected_dof: int
    minimum: float
    minimum_case_id: str
    maximum: float
    maximum_case_id: str


@dataclass(frozen=True)
class ExperimentalB2Q4PanelMemberResultantEnvelope:
    """Signed local sectional-resultant extrema at one requested station.

    Components are N, Qy, Qz, Mx, My, Mz in the member's local frame.
    Station indices distinguish repeated normalized station coordinates.
    """

    member_id: str
    station_index: int
    station_normalized: float
    component: str
    minimum: float
    minimum_case_id: str
    maximum: float
    maximum_case_id: str


@dataclass(frozen=True)
class ExperimentalB2Q4PanelBatchResult:
    status: str
    selected_dofs: tuple[int, ...]
    selected_points: tuple[str, ...]
    cases: tuple[ExperimentalB2Q4PanelSelectedCase, ...]
    factorization_count: int
    solve_count: int
    maximum_rhs_chunk: int = 1
    failure_reason: str | None = None
    production_qualified: bool = False
    displacement_envelopes: tuple[ExperimentalB2Q4PanelResponseEnvelope, ...] = ()
    affine_reaction_envelopes: tuple[ExperimentalB2Q4PanelResponseEnvelope, ...] = ()
    member_resultant_envelopes: tuple[ExperimentalB2Q4PanelMemberResultantEnvelope, ...] = ()


@dataclass(frozen=True)
class ExperimentalB2Q4FootprintResult(ExperimentalB2Q4PanelResult):
    footprint_ids: tuple[str, ...] = ()
    footprint_wrenches: tuple[B2NodalRigidFootprintWrench, ...] = ()
    footprint_law_id: str = FOOTPRINT_LAW_ID

@dataclass(frozen=True)
class B2PhysicalRigidFootprintResult(ExperimentalB2Q4PanelResult):
    footprint_ids: tuple[str, ...] = ()
    footprint_wrenches: tuple[B2PhysicalRigidFootprintWrench, ...] = ()
    constraint_multipliers: tuple = ()
    constraint_gaps: tuple = ()
    constraint_rows: tuple = ()
    footprint_law_id: str = PHYSICAL_FOOTPRINT_LAW_ID


@dataclass(frozen=True)
class B2PhysicalRigidFootprintSelectedCase(ExperimentalB2Q4PanelSelectedCase):
    """Only requested physical outputs from one complete batch case."""
    footprint_wrenches: tuple[B2PhysicalRigidFootprintWrench, ...] = ()
    constraint_multipliers: tuple[float, ...] = ()
    constraint_gaps: tuple[float, ...] = ()
    footprint_law_id: str = PHYSICAL_FOOTPRINT_LAW_ID


@dataclass(frozen=True)
class B2PhysicalRigidFootprintBatchResult(ExperimentalB2Q4PanelBatchResult):
    selected_footprints: tuple[str, ...] = ()
    constraint_rows: tuple = ()
    footprint_law_id: str = PHYSICAL_FOOTPRINT_LAW_ID


@dataclass(frozen=True)
class ExperimentalB2Q4FootprintSelectedCase(ExperimentalB2Q4PanelSelectedCase):
    footprint_wrenches: tuple[B2NodalRigidFootprintWrench, ...] = ()


@dataclass(frozen=True)
class ExperimentalB2Q4FootprintBatchResult(ExperimentalB2Q4PanelBatchResult):
    selected_footprints: tuple[str, ...] = ()
    footprint_law_id: str = FOOTPRINT_LAW_ID


@dataclass(frozen=True)
class B2BodyLoadedPanelResult(ExperimentalB2Q4PanelResult):
    body_model_identity: str = ""
    body_acceleration_m_s2: tuple[float, ...] = ()
    equilibrated_sections: tuple[B2LoadedMemberSections, ...] = ()


@dataclass(frozen=True)
class B2BodyLoadedFootprintResult(ExperimentalB2Q4FootprintResult):
    body_model_identity: str = ""
    body_acceleration_m_s2: tuple[float, ...] = ()
    equilibrated_sections: tuple[B2LoadedMemberSections, ...] = ()


@dataclass(frozen=True)
class B2BodyLoadedPanelSelectedCase(ExperimentalB2Q4PanelSelectedCase):
    body_model_identity: str = ""
    body_acceleration_m_s2: tuple[float, ...] = ()
    equilibrated_sections: tuple[B2LoadedMemberSections, ...] = ()


@dataclass(frozen=True)
class B2BodyLoadedFootprintSelectedCase(ExperimentalB2Q4FootprintSelectedCase):
    body_model_identity: str = ""
    body_acceleration_m_s2: tuple[float, ...] = ()
    equilibrated_sections: tuple[B2LoadedMemberSections, ...] = ()


@dataclass(frozen=True)
class B2LoadedMemberPoses:
    member_id: str
    stations: tuple[float, ...]
    poses_local: tuple[tuple[float, ...], ...]
    point_trace_policy: str = field(default=B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV, init=False)
    production_qualified: bool = field(default=False, init=False)


@dataclass(frozen=True)
class B2LoadedPointPanelResult(B2BodyLoadedPanelResult):
    loaded_member_poses: tuple[B2LoadedMemberPoses, ...] = ()
    loaded_strain_energy: float | None = None
    point_trace_policy: str = field(default=B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV, init=False)


@dataclass(frozen=True)
class B2LoadedPointPanelSelectedCase(B2BodyLoadedPanelSelectedCase):
    loaded_member_poses: tuple[B2LoadedMemberPoses, ...] = ()
    loaded_strain_energy: float | None = None
    point_trace_policy: str = field(default=B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV, init=False)


def _unique_ids(items, name):
    ids = [getattr(item, name) for item in items]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"nonempty {name} required")
    if len(set(ids)) != len(ids):
        raise ValueError(f"unique {name} required")


def _checked_recovery(recover_members, available):
    if recover_members is None:
        return None
    if not isinstance(recover_members, Mapping):
        raise ValueError("selected physical B2 recovery mapping required")
    selected = {}
    for member_id, requested in recover_members.items():
        if member_id not in available:
            raise ValueError("selected physical B2 member is not compiled")
        try:
            stations = np.asarray(requested, dtype=float).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("finite normalized recovery stations required") from exc
        if (stations.ndim != 1 or not len(stations) or
                not np.isfinite(stations).all() or
                np.any((stations < 0.) | (stations > 1.))):
            raise ValueError("finite normalized recovery stations required")
        selected[member_id] = tuple(float(value) for value in stations)
    return selected


def _flat_vertices(vertices):
    base = vertices[0]
    normal = np.cross(vertices[1]-base, vertices[3]-base)
    diameter = max(np.linalg.norm(a-b) for a in vertices for b in vertices)
    magnitude = np.linalg.norm(normal)
    if (not isfinite(diameter) or diameter <= 0. or
            magnitude <= 1e-12*diameter**2 or
            abs((vertices[2]-base) @ normal) > 1e-10*diameter*magnitude):
        raise ValueError("nondegenerate flat Q4 shell required")


def _require_connected_panel(shells, members, points, footprints=()):
    """Check physical incidence without forming a numerical constraint matrix."""
    neighbours = {(0, shell.shell_id): set() for shell in shells}
    neighbours.update({(1, member.member_id): set() for member in members})

    def join(a, b):
        neighbours[a].add(b)
        neighbours[b].add(a)

    for kind, entities in ((0, shells), (1, members)):
        owners = {}
        for entity in entities:
            key = (kind, entity.shell_id if kind == 0 else entity.member_id)
            shell_count = 3 if type(entity) is S3V2DPanelShell else 4
            if (kind == 0 and (type(entity.node_ids) is not tuple or
                    len(entity.node_ids) != shell_count or
                    any(type(node_id) is not int for node_id in entity.node_ids))):
                raise ValueError("three integer S3 V2D nodes required" if shell_count == 3
                                 else "four integer Q4 nodes required")
            for node_id in entity.node_ids:
                previous = owners.setdefault(node_id, key)
                join(key, previous)
    for point in points:
        shell_key, member_key = (0, point.shell_id), (1, point.member_id)
        if shell_key not in neighbours or member_key not in neighbours:
            raise ValueError("point host IDs must identify declared elements")
        join(shell_key, member_key)
    for group in footprints:
        shell_hosts = [s for s in shells if set(s.node_ids) & set(group.shell_nodes)]
        beam_hosts = [m for m in members if group.beam_node in m.node_ids]
        if (not beam_hosts or set(group.shell_nodes) -
                {n for s in shell_hosts for n in s.node_ids}):
            raise ValueError("known shell nodes and physical B2 endpoint required")
        for shell in shell_hosts:
            for member in beam_hosts:
                join((0, shell.shell_id), (1, member.member_id))
    first = next(iter(neighbours))
    visited = {first}
    pending = [first]
    while pending:
        for other in neighbours[pending.pop()]-visited:
            visited.add(other)
            pending.append(other)
    if len(visited) != len(neighbours):
        raise ValueError("connected shell/beam point panel required")


def _supports_all_panel_rigid_fields(transformation, shell_positions, beam_positions):
    """Detect all six physical rigid fields using only fixed coordinate rows.

    The constructor requires one connected physical incidence graph, so this
    is its one component's necessary support check, not a stiffness/kernel or
    mechanical qualification claim. u=a+w cross (X-origin), theta=w. Translate
    and scale its origin/length to keep rank independent of coordinate units.
    Stream at most six support rows through a 6x6 QR; no global dense rigid,
    stiffness, constraint or nullspace matrix is assembled.
    """
    selector = transformation.copy()
    selector.eliminate_zeros()
    fixed = np.flatnonzero(np.diff(selector.indptr) == 0)
    if len(fixed) < 6:
        return False
    xyz = tuple(shell_positions[n] for n in sorted(shell_positions))+tuple(
        beam_positions[n] for n in sorted(beam_positions))
    supported_nodes = np.unique(fixed//6)
    origin = np.asarray(xyz[int(supported_nodes[0])], dtype=float)
    length = 0.
    with np.errstate(over="ignore", invalid="ignore"):
        for node in supported_nodes:
            arm = np.asarray(xyz[int(node)], dtype=float)-origin
            if not np.isfinite(arm).all():
                return False
            length = max(length, float(np.max(abs(arm))))
    length = length or 1.
    reduced = np.zeros((6, 6))
    start = 0
    for node in supported_nodes:
        stop = start
        while stop < len(fixed) and fixed[stop]//6 == node:
            stop += 1
        arm = (np.asarray(xyz[int(node)], dtype=float)-origin)/length
        x, y, z = arm
        cross = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
        rigid = np.block([[np.eye(3), -cross], [np.zeros((3, 3)), np.eye(3)]])
        reduced = np.linalg.qr(np.vstack((reduced, rigid[fixed[start:stop] % 6])),
                               mode="r")
        start = stop
    scales = np.linalg.norm(reduced, axis=0)
    if not np.isfinite(reduced).all() or np.any(scales == 0.):
        return False
    singular = np.linalg.svd(reduced/scales, compute_uv=False)
    return bool(singular[-1] > 64*np.finfo(float).eps*max(6, len(fixed))*singular[0])


def _retire_private_q4_mesh(owner):
    mesh = owner._private_q4_mesh
    for element in owner._private_q4_elements:
        _retire_q4_private_mesh_caches(element, mesh)
    owner._private_q4_elements.clear()
    owner._private_q4_mesh = None


def _private_q4_constructor_lifetime(initialize):
    @wraps(initialize)
    def guarded(owner, *args, **kwargs):
        owner._private_q4_mesh = None
        owner._private_q4_elements = []
        try:
            return initialize(owner, *args, **kwargs)
        except BaseException as original:
            try:
                _retire_private_q4_mesh(owner)
            except BaseException as cleanup:
                original.add_note("private Q4 retirement failed: " + repr(cleanup))
            raise
    return guarded


class ExperimentalB2Q4PanelAnalysis:
    """Own one connected reference-linear panel and its sparse factorization.

    The coordinate order is sorted shell node IDs, then sorted beam node IDs,
    six global translations/rotations for each. Affine ``T/u0`` is supplied
    in that order. Reactions are affine reactions; they are support reactions
    only when the affine plan fixes coordinates without other MPCs.
    """

    _mixed_shells = False

    @_private_q4_constructor_lifetime
    def __init__(self, shell_nodes_m: Mapping[int, Sequence[float]],
                 shells: Sequence[Q4PanelShell],
                 members: Sequence[PhysicalB2Member | CompiledPhysicalB2Member],
                 points: Sequence[Q4B2Point], *,
                 transformation: sparse.spmatrix, prescribed_offset,
                 footprints: Sequence[B2NodalRigidFootprint] = (),
                 reference_inertia_policy: str | None = None,
                 reference_boundary_policy: str | None = None,
                 point_trace_policy: str | None = None,
                 physical_footprints: Sequence[B2PhysicalRigidFootprint] = ()):
        diagnostic = active_capture()
        if diagnostic is not None:
            diagnostic.begin()
            if not physical_footprints or points or footprints or reference_inertia_policy is not None or reference_boundary_policy is not None or point_trace_policy is not None:
                raise DiagnosticRefusal("capture admits physical static footprint compilation only")
        shells = tuple(shells)
        supplied_members = tuple(members)
        points = tuple(points)
        footprints = tuple(footprints)
        physical_footprints = tuple(physical_footprints)
        self._physical_footprint_law = bool(physical_footprints)
        if physical_footprints:
            if points or footprints or reference_inertia_policy is not None or reference_boundary_policy is not None or point_trace_policy is not None:
                raise ValueError("separate physical footprint static family required")
            footprints = physical_footprints
        modal_mass = reference_inertia_policy is not None
        if reference_boundary_policy is not None and (
                type(reference_boundary_policy) is not str or
                reference_boundary_policy != B2_FREE_FREE_MODAL_ONLY_V1 or
                not modal_mass):
            raise ValueError("explicit mixed free-free modal-only boundary policy required")
        self._modal_only = reference_boundary_policy is not None
        self._reference_boundary_policy = reference_boundary_policy
        if modal_mass and (not self._mixed_shells or
                type(reference_inertia_policy) is not str or
                reference_inertia_policy != S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1 or
                footprints):
            raise ValueError("explicit native-S3 reference inertia policy without footprints required")
        has_s3 = any(type(shell) is S3V2DPanelShell for shell in shells)
        if point_trace_policy is not None and (
                type(point_trace_policy) is not str or
                point_trace_policy != B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV or
                self._mixed_shells or has_s3 or footprints or modal_mass):
            raise ValueError("explicit Q4-only fixed-end loaded point trace without footprints or modes required")
        self._point_trace_policy = point_trace_policy
        if has_s3 and footprints and not self._physical_footprint_law:
            raise ValueError("S3 mixed-panel nodal footprints are unsupported")
        admitted_shells = ((Q4PanelShell, S3V2DPanelShell) if self._mixed_shells
                           else (Q4PanelShell,))
        footprint_type = B2PhysicalRigidFootprint if self._physical_footprint_law else B2NodalRigidFootprint
        if any(type(g) is not footprint_type for g in footprints):
            raise ValueError("typed experimental B2 nodal footprints required")
        if (len(shells) < 2 or len(supplied_members) < 2 or not (points or footprints) or
                any((type(shell) not in admitted_shells if self._mixed_shells else
                     not isinstance(shell, Q4PanelShell)) for shell in shells) or
                any(not isinstance(member, (PhysicalB2Member,
                                            CompiledPhysicalB2Member))
                    for member in supplied_members) or
                any(not isinstance(point, Q4B2Point) for point in points)):
            raise ValueError("multi-Q4/S3/multi-B2 physical point panel required" if
                             self._mixed_shells else
                             "multi-Q4/multi-B2 physical point panel required")
        _unique_ids(shells, "shell_id")
        _unique_ids(points, "point_id")
        definitions = tuple(member.definition if isinstance(
            member, CompiledPhysicalB2Member) else member
            for member in supplied_members)
        if diagnostic is not None and (any(shell.density_kg_m3 is not None for shell in shells) or
                                       any(member.mass is not None for member in definitions)):
            raise DiagnosticRefusal("capture of physical mass/body-map storage is not admitted")
        if modal_mass:
            for shell in shells:
                density = shell.density_kg_m3
                if isinstance(density, (bool, np.bool_)) or not isinstance(density, Real):
                    raise ValueError("mixed reference modes require complete positive finite shell density")
                try:
                    density_value = float(density)
                except OverflowError as exc:
                    raise ValueError("mixed reference modes require representable shell density") from exc
                if not isfinite(density_value) or density_value <= 0.:
                    raise ValueError("mixed reference modes require complete positive finite shell density")
            if any(member.mass is None for member in definitions):
                raise ValueError("mixed reference modes require complete declared physical B2 mass")
            if any(member.geometric is not None for member in definitions):
                raise ValueError("mixed reference modes are virgin elastic; prestress/geometric metadata unsupported")
        _unique_ids(definitions, "member_id")
        _require_connected_panel(shells, definitions, points, footprints)
        cache = PhysicalB2OperatorCache()
        if diagnostic is not None: diagnostic.mark("member_compilation")
        compiled = tuple(
            member if isinstance(member, CompiledPhysicalB2Member)
            else member.compile_local(operator_cache=cache)
            for member in supplied_members)
        if diagnostic is not None: diagnostic.mark("topology_mapping")
        shell_ids = tuple(sorted(shell_nodes_m))
        if any(type(node_id) is not int for node_id in shell_ids):
            raise ValueError("integer shell node IDs required")
        positions = {}
        for node_id in shell_ids:
            vector = np.array(shell_nodes_m[node_id], dtype=float, copy=True)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("finite shell node coordinates required")
            positions[node_id] = vector
        used_shell_nodes = {node_id for shell in shells for node_id in shell.node_ids}
        if used_shell_nodes != set(shell_ids):
            raise ValueError("every declared shell node must belong to a Q4/S3" if self._mixed_shells
                             else "every declared shell node must belong to a Q4")
        shell_index = {node_id: i for i, node_id in enumerate(shell_ids)}
        beam_positions = {}
        for member in definitions:
            for node_id, point in zip(member.node_ids, member.reference_points_m):
                old = beam_positions.setdefault(node_id, point)
                if old != point:
                    raise ValueError("shared B2 node has inconsistent position")
        beam_ids = tuple(sorted(beam_positions))
        beam_index = {node_id: i+len(shell_ids)
                      for i, node_id in enumerate(beam_ids)}
        footprint_compiler = compile_physical_footprints if self._physical_footprint_law else compile_footprints
        footprints, footprint_incidence = footprint_compiler(
            footprints, positions, shells, beam_positions,
            shell_index, beam_index, points)
        total = 6*(len(shell_ids)+len(beam_ids))
        for shell in shells:
            density = shell.density_kg_m3
            if density is not None:
                if isinstance(density, (bool, np.bool_)) or not isinstance(density, Real):
                    raise ValueError("positive finite shell density required when supplied")
                try:
                    density_value = float(density)
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError("representable positive shell density required") from exc
                if not isfinite(density_value) or density_value <= 0.:
                    raise ValueError("positive finite shell density required when supplied")
        complete_body_mass = (all(shell.density_kg_m3 is not None for shell in shells)
                              and all(member.definition.mass is not None for member in compiled))
        body_map = np.zeros((total, 3)) if complete_body_mass else None
        body_masses = []
        if diagnostic is not None: diagnostic.mark("shell_model_setup")
        model = FEModel("experimental-multi-Q4-B2-point")
        self._private_q4_mesh = model.mesh
        for node_id in shell_ids:
            model.add_node(node_id, *positions[node_id])
        rows, columns, values = [], [], []
        mass_rows, mass_columns, mass_values = [], [], []
        complete_mass = (not has_s3 or modal_mass) and all(shell.density_kg_m3 is not None for shell in shells)
        complete_mass = complete_mass and all(member.definition.mass is not None
                                             for member in compiled)
        shell_lookup = {}
        pieces = []
        for index, shell in enumerate(shells, 1):
            count = 3 if type(shell) is S3V2DPanelShell else 4
            if (len(shell.node_ids) != count or len(set(shell.node_ids)) != count or
                    any(node_id not in shell_index for node_id in shell.node_ids)):
                raise ValueError("distinct declared Q4/S3 nodes required" if self._mixed_shells
                                 else "four declared Q4 nodes required")
            vertices = np.array([positions[node_id] for node_id in shell.node_ids])
            if count == 4:
                _flat_vertices(vertices)
            material_id = f"panel-material-{index}"
            if modal_mass:
                model.add_material(material_id, shell.E, shell.nu, density=shell.density_kg_m3)
            else:
                model.add_material(material_id, shell.E, shell.nu)
            if self._mixed_shells:
                element = create_shell_element(
                    index, list(shell.node_ids), material_id,
                    formulation="e4-pl-s3-v2d" if count == 3 else "e4-pl",
                    thickness=shell.thickness_m,
                    reference_normal=np.asarray(shell.reference_normal, dtype=float))
            else:
                element = create_shell_element(
                    index, shell.node_ids, material_id, formulation="e4-pl", thickness=shell.thickness_m,
                    reference_normal=np.asarray(shell.reference_normal, dtype=float))
            if count == 4:
                self._private_q4_elements.append(element)
            model.add_element(index, element)
            pieces.append((shell, vertices, element, material_id, count))
        # Native S3 scope validation must see the complete shell registry.
        if diagnostic is not None: diagnostic.mark("shell_response_scatter")
        for shell, vertices, element, material_id, count in pieces:
            local = element.compute_stiffness_matrix(
                model.mesh, model.get_material(material_id))
            if local.shape != (6*count, 6*count) or not np.isfinite(local).all():
                raise ExperimentalB2Q4PanelError("finite Q4/S3 stiffness required" if self._mixed_shells
                                                else "finite Q4 stiffness required")
            slots = np.array([6*shell_index[node_id]+component
                              for node_id in shell.node_ids
                              for component in range(6)])
            _scatter(rows, columns, values, slots, local)
            if count == 4 and shell.density_kg_m3 is not None:
                local_mass = physical_mass(
                    vertices, density_kg_m3=shell.density_kg_m3,
                    thickness_m=shell.thickness_m,
                    reference_normal=shell.reference_normal)
                if complete_mass:
                    _scatter(mass_rows, mass_columns, mass_values, slots, local_mass)
                if body_map is not None:
                    translations = np.zeros((24, 3))
                    for i in range(4):
                        translations[6*i:6*i+3] = np.eye(3)
                    local_body = local_mass@translations
                    body_map[slots] += local_body
                    body_masses.append(fsum(float(v) for v in local_body[::6, 0]))
            elif count == 3 and modal_mass:
                local_mass = element.compute_mass_matrix(model.mesh, model.get_material(material_id))
                angular = np.array([6*i+j for i in range(3) for j in (3, 4, 5)])
                translation = np.array([6*i+j for i in range(3) for j in (0, 1, 2)])
                if (local_mass.shape != (18, 18) or not np.isfinite(local_mass).all()
                        or np.any(np.diag(local_mass)[translation] <= 0.)
                        or np.any(local_mass[angular, :] != 0.)
                        or np.any(local_mass[:, angular] != 0.)):
                    raise ValueError("native S3 translational mass is not representable with zero angular inertia")
                _scatter(mass_rows, mass_columns, mass_values, slots, local_mass)
            if count == 3 and body_map is not None:
                density = shell.density_kg_m3
                area = .5*float(np.linalg.norm(np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])))
                weight = float(density)*shell.thickness_m*area/3.
                if not isfinite(weight) or weight <= 0.:
                    raise RepresentabilityError("S3 reference body load is not representable")
                for i in range(3):
                    body_map[slots[6*i:6*i+3]] += weight*np.eye(3)
                body_masses.append(3.*weight)
            shell_lookup[shell.shell_id] = (vertices, slots)
        if diagnostic is not None: diagnostic.mark("beam_response_scatter")
        member_lookup = {}
        for member in compiled:
            frame = member.local_dof_map()
            local = frame.T @ member.operator.evaluate(np.zeros(12)).tangent @ frame
            if local.shape != (12, 12) or not np.isfinite(local).all():
                raise ExperimentalB2Q4PanelError("finite physical B2 stiffness required")
            slots = np.array([6*beam_index[node_id]+component
                              for node_id in member.definition.node_ids
                              for component in range(6)])
            _scatter(rows, columns, values, slots, local)
            if complete_mass:
                local_mass = frame.T@member.local_candidate_mass_matrix()@frame
                if not np.isfinite(local_mass).all():
                    raise ValueError("physical B2 mass assembly is not representable")
                if modal_mass and np.any(np.diag(local_mass)[[0, 1, 2, 6, 7, 8]] <= 0.):
                    raise ValueError("physical B2 translational mass is not representable")
                _scatter(mass_rows, mass_columns, mass_values, slots, local_mass)
            if body_map is not None:
                body_map[slots] += _beam_body_map(member, frame)
                body_masses.append(member.definition.length_m*member.definition.mass.matrix_per_m[0][0])
            member_lookup[member.definition.member_id] = (member, slots)
        if diagnostic is not None: diagnostic.mark("K_CSR")
        K = sparse.coo_matrix((values, (rows, columns)),
                              shape=(total, total)).tocsr()
        _release_coo_scratch(rows, columns, values, diagnostic, "K_CSR")
        if diagnostic is not None: diagnostic.mark("attachment_rows")
        constraint_rows, constraint_cols, constraint_values = [], [], []
        point_reference_frames = []
        point_laws = []
        buckling_points = []
        section_point_actions = {}
        loaded_point_entries = {}
        for point_index, point in enumerate(points):
            if (point.shell_id not in shell_lookup or
                    point.member_id not in member_lookup):
                raise ValueError("point host IDs must identify declared elements")
            vertices, shell_slots = shell_lookup[point.shell_id]
            member, beam_slots = member_lookup[point.member_id]
            law = compile_reference_point(
                member.definition, vertices, station=point.station,
                offset_local_m=point.offset_local_m)
            point_laws.append((point.point_id, law.law_id))
            buckling_points.append((point, law, vertices.copy(), member,
                                    np.r_[shell_slots, beam_slots]))
            if point_trace_policy is not None:
                port_constraint = (law.beam_constraint_differential @
                    np.kron(np.eye(2), member.definition.local_frame_rows().T))
                loaded_point_entries.setdefault(point.member_id, []).append(
                    (point_index, float(point.station), tuple(point.offset_local_m), port_constraint))
            if 0. < point.station < 1.:
                d = np.array(point.offset_local_m, dtype=float, copy=True)
                shift = np.eye(6)
                shift[3:, :3] = ((0., -d[2], d[1]), (d[2], 0., -d[0]), (-d[1], d[0], 0.))
                pullback = shift@np.kron(np.eye(2), member.definition.local_frame_rows())@(-law.beam_constraint_differential.T)
                pullback.setflags(write=False)
                section_point_actions.setdefault(point.member_id, []).append(
                    (point_index, float(point.station), pullback))
            if footprints:
                point_reference_frames.append(tuple(
                    tuple(tuple(map(float, row)) for row in frame)
                    for frame in law.reference_frames))
            slots = np.r_[shell_slots, beam_slots]
            for local_row in range(6):
                for local_col, slot in enumerate(slots):
                    constraint_rows.append(6*point_index+local_row)
                    constraint_cols.append(int(slot))
                    constraint_values.append(law.constraint[local_row, local_col])
        for item in footprint_incidence:
            slots = item.shell_slots+item.beam_slots
            width = getattr(item, "row_count", 6)
            local = np.column_stack((item.shell_matrix, getattr(item, "beam_matrix", np.eye(6))))
            for local_row in range(width):
                for local_col, slot in enumerate(slots):
                    constraint_rows.append(item.row_start+local_row)
                    constraint_cols.append(slot)
                    constraint_values.append(local[local_row, local_col])
        if diagnostic is not None: diagnostic.mark("C_CSR")
        C = sparse.coo_matrix(
            (constraint_values, (constraint_rows, constraint_cols)),
            shape=(6*len(points)+sum(getattr(item,"row_count",6) for item in footprint_incidence), total)).tocsr()
        _release_coo_scratch(constraint_rows, constraint_cols, constraint_values, diagnostic, "C_CSR")
        if diagnostic is not None: diagnostic.mark("support_reduction_scaling_elimination")
        loaded_members = {}
        H = None
        if point_trace_policy is not None:
            h_rows, h_columns, h_values = [], [], []
            for member_id, (member, _) in member_lookup.items():
                compiled_load = LoadedPointMember(member, loaded_point_entries.get(member_id, ()),
                                                  body_enabled=body_map is not None)
                loaded_members[member_id] = compiled_load
                slots = np.array([6*index+channel for index in compiled_load.point_indices
                                  for channel in range(6)])
                _scatter(h_rows, h_columns, h_values, slots, compiled_load.constraint_compliance)
            H = sparse.coo_matrix((h_values, (h_rows, h_columns)), shape=(C.shape[0],)*2).tocsr()
            # Stored analytical zeros carry no numerical information. Remove
            # them before congruence so underflow checks compare nonzero patterns.
            H.eliminate_zeros()
        try:
            T = transformation.tocsr(copy=True)
            u0 = np.array(prescribed_offset, dtype=float, copy=True)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("finite sparse panel affine plan required") from exc
        minimum_columns = C.shape[0] if point_trace_policy is None else 0
        if (T.shape[0] != total or T.shape[1] <= minimum_columns or
                u0.shape != (total,) or not np.isfinite(T.data).all() or
                not np.isfinite(u0).all()):
            raise ValueError("finite sparse panel affine plan required")
        if modal_mass:
            if np.any(u0 != 0.):
                raise ValueError("virgin mixed reference modes require zero prescribed offset")
            selector = T.copy()
            selector.eliminate_zeros()
            if (np.any(np.diff(selector.tocsc().indptr) != 1) or
                    np.any(np.diff(selector.indptr) > 1)):
                raise ValueError("mixed reference modes require coordinate supports without MPCs")
            if self._modal_only:
                delta = (T-sparse.eye(total, format="csr")) if T.shape == (total, total) else None
                if delta is None or np.any(delta.data != 0.):
                    raise ValueError("free-free modal-only policy requires identity transformation without supports")
            elif not _supports_all_panel_rigid_fields(T, positions, beam_positions):
                raise ValueError("mixed reference modes require supports detecting all six rigid fields; free-free unsupported")
        reduced_c = (C @ T).tocsc()
        if point_trace_policy is None:
            _require_independent_supported_point_rows(reduced_c)
        elif not np.isfinite(reduced_c.data).all():
            raise ValueError("finite supported loaded-point rows required")
        # Loaded fixed-end compliance can anchor a multiplier row even when
        # its free-coordinate trace vanishes. Independence is therefore a
        # property of the complete [K, C.T; C, -H] factorization on this route.
        reduced_k = (T.T @ K @ T).tocsc()
        # Congruence scaling changes factorization coordinates only. The
        # physical displacement and multiplier are both restored after solve.
        # Without this, a valid slender B2 section can leave point gaps above
        # the physical acceptance tolerance despite a sound local operator.
        handle = factor_scale = None
        attachment_maps = None
        attachment_reason = "not_physical_static_family"
        if not self._modal_only:
            scaled_kkt, factor_scale = _scaled_point_kkt(reduced_k, reduced_c, H)
            if self._physical_footprint_law and not points and point_trace_policy is None and H is None:
                attachment_maps, attachment_reason = prepare_attachment_elimination(
                    scaled_kkt, reduced_k.shape[0], T, footprint_incidence)
            # Ineligible attachment layouts use the exact existing factor path.
            # An eligible path factors once and restores the full scaled answer
            # before original physical checks/recovery; spectral operators are
            # still prepared from original K/C/T, not these static embeddings.
            if diagnostic is not None:
                diagnostic.prefactor(
                    dict(K=K,C=C,T=T,u0=u0,H=H,reduced_k=reduced_k,reduced_c=reduced_c,
                         scaled_kkt=scaled_kkt,factor_scale=factor_scale,attachment_maps=attachment_maps,
                         model=model,positions=positions,beam_positions=beam_positions,shell_index=shell_index,
                         beam_index=beam_index,shell_lookup=shell_lookup,member_lookup=member_lookup,
                         footprint_incidence=footprint_incidence,compiled=compiled,cache=cache,
                         private_q4_elements=self._private_q4_elements),
                    dict(rows=rows,columns=columns,values=values,constraint_rows=constraint_rows,
                         constraint_cols=constraint_cols,constraint_values=constraint_values,
                         pieces=pieces,local=local,slots=slots,frame=frame),
                    (*shells,*definitions,*footprints,*(m.section for m in definitions),
                     *(m for m in supplied_members if isinstance(m,CompiledPhysicalB2Member))))
            handle = (attachment_maps.factorize() if attachment_maps is not None else
                      factorize_cached(scaled_kkt, MatrixClass.SYMMETRIC_INDEFINITE))
            if handle.status != "ok":
                raise ExperimentalB2Q4PanelError(
                    "panel factorization failed: " + str(handle.failure_reason))
        if diagnostic is not None:
            try: diagnostic.mark("owner_publication")
            except BaseException:
                if handle is not None:
                    try: handle.close()
                    except BaseException: pass
                raise
        self._modal_positions = (np.array(
            [positions[i] for i in shell_ids]+[beam_positions[i] for i in beam_ids],
            dtype=float, copy=True) if self._modal_only else None)
        self.shell_node_ids = shell_ids
        self.beam_node_ids = beam_ids
        self.point_ids = tuple(point.point_id for point in points)
        self._has_s3 = has_s3
        self._reference_inertia_policy = reference_inertia_policy
        self._shell_formulations = tuple((shell.shell_id, element.formulation_id)
                                        for shell, _, element, _, _ in pieces)
        self._point_laws = tuple(point_laws)
        self._point_reference_frames = tuple(point_reference_frames)
        self.footprints = footprints
        self.footprint_ids = tuple(g.footprint_id for g in footprints)
        self._footprint_incidence = footprint_incidence
        self._members = tuple(member_lookup.items())
        self._section_point_actions = {key: tuple(value) for key, value in section_point_actions.items()}
        self._loaded_members = loaded_members
        self._H = H
        self._abs_H = None if H is None else abs(H)
        self._K = K
        # Spectral-only captures reuse the exact prepared reference operators;
        # no accepted state or nonlinear material history is manufactured.
        self._buckling_model = model
        self._buckling_shells = tuple((shell, element, material_id,
            shell_lookup[shell.shell_id][1]) for shell, _, element, material_id, _ in pieces)
        self._buckling_points = tuple(buckling_points)
        if body_map is not None and not np.isfinite(body_map).all():
            raise RepresentabilityError("reference body-load map is not representable")
        self._body_map = body_map
        self._body_mass = None
        if self._body_map is not None:
            try:
                self._body_mass = fsum(body_masses)
            except OverflowError as exc:
                raise RepresentabilityError("total reference body mass is not representable") from exc
            if not isfinite(self._body_mass) or self._body_mass <= 0.:
                raise RepresentabilityError("total reference body mass is not representable")
            self._body_map.setflags(write=False)
        self._M = (sparse.coo_matrix((mass_values, (mass_rows, mass_columns)),
                                    shape=(total, total)).tocsr()
                   if complete_mass else None)
        if self._M is not None and not np.isfinite(self._M.data).all():
            raise ValueError("physical panel mass assembly is not representable")
        self._C = C
        # Freeze a translation scale from the physical reference geometry;
        # rotation constraints remain in radians. Origin shifts do not alter it.
        self._compatibility_length = None
        if footprints:
            reference = np.asarray(tuple(positions.values())+tuple(beam_positions.values()))
            relative = reference-reference[0]
            radius = np.linalg.norm(relative-np.mean(relative, axis=0), axis=1)
            self._compatibility_length = 2.*float(np.max(radius))
            if not isfinite(self._compatibility_length) or self._compatibility_length <= 0.:
                raise ValueError("representable physical reference length required")
        self._T = T
        self._Tt = T.T.tocsr()
        self._projection_multi = np.flatnonzero(
            np.diff(self._Tt.indptr) > 1)
        self._projection_single = np.flatnonzero(
            np.diff(self._Tt.indptr) == 1)
        # Reuse the immutable CSR patterns; only the absolute values need
        # separate storage for repeated RHS validation.
        self._abs_K = sparse.csr_matrix(
            (np.abs(K.data), K.indices, K.indptr), shape=K.shape, copy=False)
        self._abs_C = sparse.csr_matrix(
            (np.abs(C.data), C.indices, C.indptr), shape=C.shape, copy=False)
        self._abs_Ct = abs(C.T.tocsr())
        self._abs_Tt = abs(T.T.tocsr())
        self._u0 = u0
        # One compilation-time binding covers the numerical model and physical
        # member data used by body action/recovery. Never hash the model per case.
        identity = sha256(b"B2_REFERENCE_BODY_LOAD_BINDING_V1_DEV")
        if self._modal_only:
            identity.update(reference_boundary_policy.encode("ascii"))
        identity.update(repr((shell_ids, beam_ids, tuple((i, tuple(positions[i])) for i in shell_ids), self._shell_formulations,
                              tuple(point_laws), reference_inertia_policy)).encode("utf-8"))
        for matrix in (K, C, T):
            data = matrix.tocsr()
            identity.update(repr(data.shape).encode("ascii"))
            for array in (data.indptr, data.indices):
                identity.update(np.asarray(array, dtype="<i8").tobytes())
            identity.update(np.asarray(data.data, dtype="<f8").tobytes())
        identity.update(np.asarray(u0, dtype="<f8").tobytes())
        if body_map is not None:
            identity.update(np.asarray(body_map, dtype="<f8").tobytes())
        for member_id, (member, _) in self._members:
            identity.update(repr((member_id, member.definition.reference_points_m,
                member.definition.local_y_global, member.definition.section,
                member.definition.mass)).encode("utf-8"))
        for key, actions in sorted(self._section_point_actions.items()):
            identity.update(repr((key, tuple((i, s) for i, s, _ in actions))).encode("utf-8"))
            for _, _, pullback in actions:
                identity.update(np.asarray(pullback, dtype="<f8").tobytes())
        if point_trace_policy is not None:
            identity.update(point_trace_policy.encode("ascii"))
            for matrix in (H,):
                identity.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
                identity.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
                identity.update(np.asarray(matrix.data, dtype="<f8").tobytes())
            for member_id, compiled_load in sorted(loaded_members.items()):
                identity.update(repr((member_id, compiled_load.point_indices,
                    compiled_load.stations, compiled_load.offsets,
                    compiled_load.quadrature_stations)).encode("utf-8"))
                for value in (compiled_load.maps.uniform_pose_map, compiled_load.maps.point_pose_map,
                        compiled_load.maps.point_compliance, compiled_load.body_wrench_map,
                        compiled_load.body_constraint_map, compiled_load.quadrature_weights,
                        *compiled_load.port_constraints):
                    identity.update(repr(value.shape).encode("ascii"))
                    identity.update(np.asarray(value, dtype="<f8").tobytes())
        self._body_model_identity = identity.hexdigest()
        self._footprint_rigid_supported = (not footprints or
            (self._coordinate_modal_supports() and
             _supports_all_panel_rigid_fields(T, positions, beam_positions)))
        self._ku0 = np.asarray(K @ u0).reshape(-1)
        self._cu0 = np.asarray(C @ u0).reshape(-1)
        # Exact homogeneous singleton constraints prescribe a reduced
        # coordinate to zero. Store those substitutions once, rather than
        # allowing factor roundoff to turn a mathematically zero channel into
        # its own nonzero relative compatibility scale.
        self._physical_singletons = ()
        if self._physical_footprint_law and not self._modal_only:
            self._physical_singletons = self._structural_physical_singletons(C, T, u0)
        self._factor_scale = factor_scale
        self._handle = handle
        self._attachment_elimination = attachment_maps
        self._attachment_elimination_info = ({"status": "eligible", **attachment_maps.info}
                                             if attachment_maps is not None else
                                             {"status": "legacy_fallback", "reason": attachment_reason})
        # Retain the full original scaled operator on the physical route.
        # Eligible handles factorize its attachment reduction and recover full
        # answers; original checks/correction still use the same reused factors.
        self._physical_scaled_kkt = (scaled_kkt if self._physical_footprint_law
                                     and not self._modal_only else None)
        self._resolution_kkt = None
        self._affine_resolution_cache = None
        self._closed = False
        if diagnostic is not None:
            try: diagnostic.finish()
            except BaseException:
                try: self.close()
                except BaseException: pass
                raise

    def _physical_equilibrium_ratio(self, answer, applied):
        """Measure the existing per-row equilibrium check before recovery."""
        q, lam = answer[:-len(self._cu0)], answer[-len(self._cu0):]
        u = np.asarray(self._T @ q).reshape(-1)+self._u0
        force = np.asarray(self._K @ u).reshape(-1)
        constraint_force = np.asarray(self._C.T @ lam).reshape(-1)
        projected = self._project(force+constraint_force-applied)
        physical_force_scale = np.asarray(
            self._abs_K @ np.abs(u)+self._abs_Ct @ np.abs(lam)+
            np.abs(applied)).reshape(-1)
        equilibrium_scale = np.asarray(self._abs_Tt @ physical_force_scale).reshape(-1)
        by_node = physical_force_scale.reshape(-1, 6)
        physical_roundoff = np.tile(
            (float(np.max(by_node[:, :3])),)*3+
            (float(np.max(by_node[:, 3:])),)*3, len(by_node))
        projected_roundoff = np.asarray(self._abs_Tt @ physical_roundoff).reshape(-1)
        tolerance = (1e-9*equilibrium_scale+
                     len(answer)*np.finfo(float).eps*projected_roundoff)
        if (not np.isfinite(projected).all() or
                not np.isfinite(tolerance).all() or np.any(tolerance < 0.)):
            raise _NumericalFailure("panel equilibrium correction is not representable")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.divide(np.abs(projected), tolerance,
                               out=np.full_like(projected, np.inf),
                               where=tolerance > 0.)
        ratios[(tolerance == 0.) & (projected == 0.)] = 0.
        return float(np.max(ratios))

    def _physical_work_ratio(self, answer, applied):
        """Check one explicit physical load against homogeneous-constraint work."""
        if np.any(self._u0 != 0.):
            return 0.
        q = answer[:-len(self._cu0)]
        u = np.asarray(self._T @ q).reshape(-1)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            internal = float(u @ (self._K @ u))
            external = float(u @ applied)
        if not isfinite(internal) or not isfinite(external):
            raise _NumericalFailure("panel physical footprint work is not representable")
        defect = abs(internal-external)
        # The physical-footprint family admits no points, body-load driver or
        # nonzero constraint RHS. Keep the registered work absolute allowance.
        tolerance = 1e-12 + 1e-8*abs(external)
        return defect/tolerance

    @staticmethod
    def _structural_physical_singletons(C, T, u0):
        """Admit only a single product with individually zero offset terms.

        Rounded cancellation in C@T or C@u0 never establishes eligibility.
        Coupled rows and even cancellation within one reduced column remain
        on the ordinary factor/guard route.
        """
        constraints = C.tocsr(copy=True)
        transform = T.tocsr(copy=True)
        constraints.eliminate_zeros()
        transform.eliminate_zeros()
        singletons = []
        for row in range(constraints.shape[0]):
            start, end = constraints.indptr[row:row+2]
            physical = constraints.indices[start:end]
            if np.any(u0[physical] != 0.):
                continue
            products = []
            for index, coefficient in zip(physical, constraints.data[start:end]):
                first, last = transform.indptr[index:index+2]
                for column, value in zip(transform.indices[first:last], transform.data[first:last]):
                    products.append((int(column), float(coefficient*value)))
                    if len(products) > 1:
                        break
                if len(products) > 1:
                    break
            if len(products) == 1 and np.isfinite(products[0][1]) and products[0][1] != 0.:
                singletons.append((row, *products[0]))
        return tuple(singletons)

    def _restore_physical_singletons(self, answer):
        """Enforce exact zero coordinates only for roundoff-sized residues.

        This changes the candidate, never a compatibility tolerance. Material
        errors remain for the existing equilibrium/correction/refusal checks.
        Translation and rotation amplitudes retain their separate units.
        """
        if not self._physical_singletons:
            return answer
        q = answer[:-len(self._cu0)]
        u = np.asarray(self._T @ q).reshape(-1)+self._u0
        poses = np.abs(u).reshape(-1, 6)
        amplitude = np.tile((float(np.max(poses[:, :3])),)*3+
                            (float(np.max(poses[:, 3:])),)*3, len(poses))
        bounds = len(answer)*np.finfo(float).eps*np.asarray(
            self._abs_C @ amplitude).reshape(-1)
        if not np.isfinite(bounds).all():
            raise _NumericalFailure("physical singleton substitution is not representable")
        restored = answer.copy()
        for row, column, coefficient in self._physical_singletons:
            if abs(coefficient*q[column]) <= bounds[row]:
                restored[column] = 0.
        return restored

    def _physical_original_residual(self, answer, applied):
        """Original affine/constraint covectors at the actual scored candidate.

        Use the same original K/C/T and ordinary arithmetic as recovery; no
        rounded reduced/congruence matrix supplies this target. This private
        residual is used only by the physical footprint reference-static route.
        """
        # Constructor admission excludes points and point_trace_policy, so H
        # and body/fixed-end constraint RHS d are absent on this route. Typed
        # body loads are refused before the sole solve_linear caller; batches
        # delegate to that caller. Refuse a corrupted/foreign private owner
        # rather than silently evaluating Cu=0 for [C,-H] or nonzero d.
        if (not self._physical_footprint_law or self.point_ids or
                self._point_trace_policy is not None or
                (self._H is not None and self._H.nnz != 0)):
            raise _NumericalFailure("original correction requires the separate physical footprint static family")
        if (answer.shape != self._factor_scale.shape or
                applied.shape != self._u0.shape or
                not np.isfinite(answer).all() or not np.isfinite(applied).all()):
            raise _NumericalFailure("finite original physical correction inputs required")
        nq = self._T.shape[1]
        q, lam = answer[:nq], answer[nq:]
        u = np.asarray(self._T @ q).reshape(-1)+self._u0
        reaction = np.asarray(self._K @ u).reshape(-1)+np.asarray(self._C.T @ lam).reshape(-1)-applied
        target = np.r_[self._project(reaction), np.asarray(self._C @ u).reshape(-1)]
        if not np.isfinite(u).all() or not np.isfinite(target).all():
            raise _NumericalFailure("original physical correction target is not representable")
        return target

    def _correct_physical_equilibrium(self, scaled_answer, scaled_rhs, applied):
        """One original-operator correction using the existing factorization.

        ``scaled_rhs`` remains in this private calling convention, but cannot
        define the original target. Restoration precedes both scoring and RHS
        formation; the correction is added to that same restored candidate.
        Physical equation/work/compatibility acceptance remains unchanged.
        """
        answer = self._restore_physical_singletons(self._factor_scale*scaled_answer)
        original_ratio = self._physical_equilibrium_ratio(answer, applied)
        original_work_ratio = self._physical_work_ratio(answer, applied)
        original_score = max(original_ratio, original_work_ratio)
        if original_score <= 1.:
            return answer
        original_target = self._physical_original_residual(answer, applied)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            correction_rhs = -self._factor_scale*original_target
        if (not np.isfinite(correction_rhs).all() or
                np.any((original_target != 0.) & (correction_rhs == 0.))):
            raise _NumericalFailure("panel equilibrium correction is not representable")
        try:
            delta = np.asarray(self._handle.solve(correction_rhs), dtype=float)
        except RuntimeError as exc:
            raise _NumericalFailure("panel equilibrium correction failed: " + str(exc)) from exc
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            candidate = answer+self._factor_scale*delta
        if candidate.shape != answer.shape or not np.isfinite(candidate).all():
            raise _NumericalFailure("finite panel equilibrium correction required")
        candidate = self._restore_physical_singletons(candidate)
        candidate_score = max(self._physical_equilibrium_ratio(candidate, applied),
                              self._physical_work_ratio(candidate, applied))
        if candidate_score >= original_score:
            raise _NumericalFailure("panel equilibrium correction did not improve")
        return candidate

    def _explicit_load_resolved(self, answer, rhs, explicit_rhs, *, applied=None, reconstructed=None) -> bool:
        """Certify active stored KKT rows without a model-wide force floor.

        Pure unit selectors retain the stored KKT certificate. Zero-offset
        footprint affine maps certify exact physical virtual equilibrium and
        the actual reconstruction error from original sparse operators/load.
        Nonzero offsets and loaded-point affine maps remain ineligible. This
        resolves net projected loads only, not cancelling individual actions.
        All existing physical equilibrium/compatibility checks still apply.
        """
        if (np.any(self._u0 != 0.) or not np.isfinite(answer).all() or
                not np.isfinite(rhs).all()):
            return False
        selector = self._T.tocsc(copy=True)
        selector.eliminate_zeros()
        if (np.any(np.diff(selector.indptr) != 1) or
                np.any(selector.data != 1.) or
                len(np.unique(selector.indices)) != selector.shape[1]):
            if not self.footprints or self._point_trace_policy is not None or applied is None or reconstructed is None:
                return False
            if self._affine_resolution_cache is None:self._affine_resolution_cache = {}
            try:
                return certify_affine(self._K,self._C,self._T,answer,applied,reconstructed,self._affine_resolution_cache)
            except B2NumericalSolveFailure:
                return False
        active = np.flatnonzero(explicit_rhs > 0.)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            cap = np.nextafter(1e-9*explicit_rhs[active], 0.)
        if not len(active) or np.any(cap <= 0.) or not np.isfinite(cap).all():
            return False
        if self._resolution_kkt is None:
            columns = selector.indices
            rk = self._K.tocsr()[columns][:, columns]
            rc = self._C.tocsr()[:, columns]
            self._resolution_kkt = sparse.bmat(
                ((rk, rc.T), (rc, None if self._point_trace_policy is None else -self._H)), format="csr")
        try:
            _, upper = _bounded_accurate_residual(
                self._resolution_kkt[active], answer[:, None], rhs[active, None])
        except B2NumericalSolveFailure:
            return False
        return bool(np.isfinite(upper).all() and np.all(upper[:, 0] <= cap))

    def _project(self, values: np.ndarray) -> np.ndarray:
        return _checked_affine_projection(
            self._Tt, self._projection_multi, self._projection_single, values)

    def _footprint_gap_tolerance(self, u, gap_scale, roundoff_fraction):
        """Quiet-channel solve admission, bounded by reference compatibility.

        Called only after explicit-load certification for unit selectors or
        proved general-affine footprint maps, with zero offsets. This is a
        solve/compatibility convention, not a bound on
        rounding in C@u. Translation and rotation amplitudes stay separate;
        actual row coefficients include eccentric lever arms. Historical point
        rows retain their original check. Every footprint row also stays below
        the independent reference-geometry 1e-9 compatibility cap.
        """
        tolerance = 1e-9*gap_scale
        poses = np.abs(u).reshape(-1, 6)
        amplitude = np.tile((float(np.max(poses[:, :3])),)*3+
                            (float(np.max(poses[:, 3:])),)*3, len(poses))
        allowance = roundoff_fraction*np.asarray(self._abs_C@amplitude).reshape(-1)
        if not np.isfinite(allowance).all():
            raise _NumericalFailure("footprint compatibility allowance is not representable")
        for item in self._footprint_incidence:
            width = getattr(item,"row_count",6)
            rows = slice(item.row_start, item.row_start+width)
            cap = 1e-9*np.array((self._compatibility_length,)*3+(1.,)*(width-3))
            tolerance[rows] = np.minimum(tolerance[rows]+allowance[rows], cap)
        return tolerance

    @property
    def capabilities(self) -> ExperimentalB2Q4PanelCapabilities:
        if self._physical_footprint_law:
            return B2PhysicalRigidFootprintCapabilities(
                attachment="physical rigid-region translations and director tangents; five rows per selected node",
                operations=(("solve_linear","solve_linear_many","close") if self._has_s3 else ("solve_linear","solve_linear_many","buckling","close")),
                spectral_domain=("S3/mixed physical footprint modes and buckling unsupported" if self._has_s3 else "explicit physical-footprint conservative reference buckling request; modes unavailable without mass contract"),
                footprints=self.footprints)
        if self._modal_only:
            return ExperimentalB2Q4PanelCapabilities(
                operations=("modes", "close"),
                body_load_domain="unsupported by free-free modal-only policy",
                spectral_domain="experimental connected free-free reference descriptor; six verified physical rigid fields and selected finite flexible pairs; completeness unqualified")
        body_operations = (("body_acceleration_load",) if self._body_map is not None else ())
        body_domain = ("experimental reference-linear uniform global acceleration; typed loads give separate equilibrated section fields including interior point actions; raw vectors retain nodal constitutive recovery; loaded interior pose and finite gravity unsupported"
                       if self._body_map is not None else "unavailable without complete physical density/mass")
        if self._point_trace_policy is not None:
            return ExperimentalB2LoadedPointCapabilities(
                attachment="explicit fixed-end loaded finite-vertex point trace",
                operations=("solve_linear", "solve_linear_many", "close") + body_operations,
                body_load_domain=("typed reference body acceleration and multiplier-dependent fixed-end bubbles; "
                    "raw arrays are nodal loads; finite gravity unsupported" if self._body_map is not None
                    else body_domain))
        if self.footprints:
            admitted = (self._M is not None and self._coordinate_modal_supports()
                        and not np.any(self._u0 != 0.) and self._footprint_rigid_supported)
            return ExperimentalB2Q4FootprintCapabilities(
                attachment="explicit reference-linear selected-node rigid footprints and finite-vertex points",
                operations=(("solve_linear", "solve_linear_many", "modes", "close")
                            if admitted else ("solve_linear", "solve_linear_many", "close")) + body_operations,
                body_load_domain=body_domain,
                spectral_domain=("experimental virgin supported reference modes; normal-spin locked; "
                                 "mechanical qualification incomplete; prestress unsupported"
                                 if admitted else
                                 "reference modes require complete physical mass, coordinate supports detecting all six rigid fields and zero offset; "
                                 "normal-spin locked; mechanical qualification incomplete; prestress unsupported"),
                footprints=self.footprints)
        if self._M is None:
            return ExperimentalB2Q4PanelCapabilities(
                operations=("solve_linear", "solve_linear_many", "close") + body_operations,
                body_load_domain=body_domain)
        return ExperimentalB2Q4PanelCapabilities(
            operations=("solve_linear", "solve_linear_many", "modes", "close") + body_operations,
            body_load_domain=body_domain,
            spectral_domain="experimental reference B2/Q4 finite pairs with coordinate supports; completeness unqualified")

    def body_acceleration_load(self, acceleration_m_s2) -> B2ReferenceBodyLoad:
        """Prepare an owned dead-load vector without assembling or factoring.

        f = integral D.T rho a uses the compiled physical trace and mass origin.
        Pass ``result.load`` to the existing static or compatible batch route.
        """
        if self._closed:
            raise ExperimentalB2Q4PanelError("experimental panel is closed")
        if self._modal_only:
            raise ValueError("free-free modal-only policy does not admit body loads")
        if self._body_map is None:
            raise ValueError("complete physical density/mass required for reference body load")
        raw = np.asarray(acceleration_m_s2, dtype=object)
        if (raw.shape != (3,) or any(isinstance(v, (bool, np.bool_)) or
                not isinstance(v, Real) for v in raw.flat)):
            raise ValueError("three finite real global acceleration components required")
        try:
            acceleration = np.array(raw, dtype=float, copy=True)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("three finite real global acceleration components required") from exc
        if not np.isfinite(acceleration).all():
            raise ValueError("three finite real global acceleration components required")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            applied = self._body_map@acceleration
        if not np.isfinite(applied).all():
            raise RepresentabilityError("reference body acceleration load is not representable")
        # A finite vector can still lose a required force through underflow or
        # cancellation. Test rigid-translation work against integrated physical
        # mass without adding an absolute load floor or changing solver criteria.
        for axis, component in enumerate(acceleration):
            if component == 0.:
                continue
            expected = self._body_mass*float(component)
            try:
                actual = fsum(float(v) for v in applied[axis::6])
            except OverflowError as exc:
                raise RepresentabilityError("reference body force is not representable") from exc
            if (not isfinite(expected) or expected == 0. or not isfinite(actual)
                    or abs(actual/expected-1.) > 1e-9):
                raise RepresentabilityError("required reference body force is not representable")
        return B2ReferenceBodyLoad(tuple(float(v) for v in acceleration),
                                   self.shell_node_ids, self.beam_node_ids, applied,
                                   self._body_model_identity)

    def _body_load_snapshot(self, load):
        if isinstance(load, B2ReferenceBodyLoad):
            if (load.model_identity != self._body_model_identity or
                    load.shell_node_ids != self.shell_node_ids or load.beam_node_ids != self.beam_node_ids):
                raise ValueError("body load belongs to a different compiled model")
            expected = self.body_acceleration_load(load.acceleration_m_s2)
            if not np.array_equal(load.load, expected.load):
                raise ValueError("body load values disagree with its physical acceleration")
            snapshot, fingerprint = _load_value_snapshot(load.load, self._u0.shape)
            metadata = sha256(fingerprint)
            metadata.update(load.model_identity.encode("utf-8"))
            metadata.update(repr((load.shell_node_ids, load.beam_node_ids)).encode("utf-8"))
            metadata.update(np.asarray(load.acceleration_m_s2, dtype="<f8").tobytes())
            return snapshot, metadata.digest()
        return _load_value_snapshot(load, self._u0.shape)

    def _coordinate_modal_supports(self) -> bool:
        selector = self._T.copy()
        selector.eliminate_zeros()
        return not (np.any(np.diff(selector.tocsc().indptr) != 1) or
                    np.any(np.diff(selector.indptr) > 1))

    def _reference_modes_result(self, status, frequencies=(), physical=(),
                                multipliers=(), equations=(), constraints=(),
                                factorization_count=0, solve_count=0,
                                failure_reason=None):
        args = (status, frequencies, physical, multipliers, equations, constraints,
                factorization_count, solve_count, failure_reason)
        if not self.footprints:
            return B2ReferenceModesResult(*args)
        components = ("ux", "uy", "uz", "rx", "ry", "rz")
        provenance = [B2ModalConstraintRow(
                          6*i+j, component, point_id=point_id,
                          coordinate_frame=("point_master_shell_reference" if j < 3
                                            else "point_beam_reference"),
                          reference_frame_columns=self._point_reference_frames[i][0 if j < 3 else 1])
                      for i, point_id in enumerate(self.point_ids)
                      for j, component in enumerate(components)]
        for item in self._footprint_incidence:
            provenance.extend(B2ModalConstraintRow(
                item.row_start+j, component, footprint_id=item.footprint_id,
                shell_node_id=item.shell_node, beam_node_id=item.beam_node)
                for j, component in enumerate(components))
        point_modes = (tuple(mode[:6*len(self.point_ids)] for mode in multipliers)
                       if self.point_ids else ())
        return B2FootprintReferenceModesResult(
            status, frequencies, physical, point_modes, equations, constraints,
            factorization_count, solve_count, failure_reason,
            footprint_ids=self.footprint_ids,
            constraint_multiplier_modes=multipliers, constraint_rows=tuple(provenance))

    @property
    def physical_footprint_buckling_capabilities(self):
        from ._stiffened_shell_b2_reference_buckling import physical_footprint_capabilities
        return physical_footprint_capabilities(self)

    @property
    def reference_buckling_capabilities(self):
        from ._stiffened_shell_b2_reference_buckling import capabilities
        return capabilities(self)

    def buckling(self, request, *, cancellation=None):
        """Explicit experimental reference-preload conservative spectral route."""
        from ._stiffened_shell_b2_reference_buckling import solve
        return solve(self, request, cancellation=cancellation)

    def modes(self, request: B2ReferenceModesRequest, *,
              cancellation: CancellationToken | None = None
              ) -> B2ReferenceModesResult | B2FreeReferenceModesResult:
        """Sparse descriptor modes with physical inertia and massless multipliers.

        Reuse compiled K/C/T and mass, never a global nullspace or dense inverse.
        A shifted general inverse-product is not Euclidean symmetric, so use
        eigs rather than eigsh. Explicit free-free owners project verified rigid
        fields in the physical mass metric; loaded/prestressed modes remain out.
        """
        if self._physical_footprint_law:
            raise ValueError("physical footprint reference modes require a separately adopted mass contract")
        if self._closed:
            raise ExperimentalB2Q4PanelError("experimental panel is closed")
        if self._point_trace_policy is not None:
            raise ValueError("loaded point trace reference modes unsupported: shape inertia unverified")
        if self._has_s3 and self._reference_inertia_policy is None:
            raise ValueError("native S3 V2D physical mass and reference modes are deferred")
        if type(request) is not B2ReferenceModesRequest:
            raise ValueError("typed reference-mode request required")
        if self._M is None:
            raise ValueError("explicit shell density and beam physical mass required")
        if not self._coordinate_modal_supports():
            raise ValueError("reference modes currently require coordinate supports without MPCs")
        if self.footprints and np.any(self._u0 != 0.):
            raise ValueError("virgin nodal-footprint reference modes require zero prescribed offset")
        if self.footprints and not self._footprint_rigid_supported:
            raise ValueError("nodal-footprint reference modes require coordinate supports detecting all six physical rigid fields; free-free and under-supported modes unsupported")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise ValueError("CancellationToken required")
        n = self._T.shape[1]
        rows = self._C.shape[0]
        if request.count >= min(n-rows-(6 if self._modal_only else 0), n+rows-1):
            raise ValueError("reference mode count exceeds supported sparse spectrum")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            sigma = -float(np.square(2.*np.pi*np.float64(request.shift_hz)))
        if not np.isfinite(sigma) or sigma == 0.:
            raise ValueError("reference mode shift is not representable")

        def safe_point(stage):
            if cancellation is not None:
                cancellation.checkpoint(stage)

        handle = None
        rigid = None
        overlap = 0.

        def finish(*args, **kwargs):
            flexible = self._reference_modes_result(*args, **kwargs)
            if not self._modal_only:
                return flexible
            data = {}
            if flexible.status == "ok":
                data = dict(rigid_origin_m=rigid["origin"],
                    canonical_rigid_modes=tuple(tuple(map(float, x)) for x in rigid["canonical"].T),
                    rigid_modes=tuple(tuple(map(float, x)) for x in rigid["normalized"].T),
                    rigid_multiplier_modes=tuple((0.,)*rows for _ in range(6)),
                    rigid_kinetic_gram=tuple(tuple(map(float, x)) for x in rigid["gram"]),
                    rigid_equation_residuals=rigid["equations"],
                    rigid_constraint_residuals=rigid["gaps"],
                    rigid_flexible_mass_overlap=overlap)
            return B2FreeReferenceModesResult(flexible.status, flexible,
                spectral_shift_rad2_s2=sigma, failure_reason=flexible.failure_reason, **data)

        try:
            safe_point("reference modes before factorization")
            k = (self._T.T@self._K@self._T).tocsc()
            m = (self._T.T@self._M@self._T).tocsc()
            c = (self._C@self._T).tocsc()
            a = sparse.bmat(((k, c.T), (c, None)), format="csc")
            b = sparse.block_diag((m, sparse.csc_matrix((rows, rows))), format="csc")
            try:
                shifted, scale = _scaled_point_kkt(k-sigma*m, c)
            except ExperimentalB2Q4PanelError as exc:
                raise _NumericalFailure("reference modal scaling failed: "+str(exc)) from exc
            scaling = sparse.diags(scale, format="csc")
            bs = (scaling@b@scaling).tocsc()
            b.eliminate_zeros(); b.sort_indices()
            bs.eliminate_zeros(); bs.sort_indices()
            if (not np.isfinite(bs.data).all() or b.nnz != bs.nnz or
                    not np.array_equal(b.indptr, bs.indptr) or
                    not np.array_equal(b.indices, bs.indices) or
                    np.any((b.data != 0.) & (bs.data == 0.))):
                raise _NumericalFailure("reference modal mass scaling is not representable")
            if self._modal_only:
                rigid = _free_b2_rigid_space(self, scale, bs, safe_point)

            def project(vector):
                if not self._modal_only:
                    return vector
                return vector-rigid["scaled"]@(rigid["scaled"].T@(bs@vector))

            handle = factorize_cached(shifted, MatrixClass.GENERAL)
            safe_point("reference modes after factorization")
            if handle.status != "ok":
                raise _NumericalFailure("reference modal factorization failed: "+str(handle.failure_reason))

            def action(vector):
                safe_point("reference modal iteration")
                try:
                    answer = np.asarray(handle.solve(bs@project(vector)), dtype=float)
                except RuntimeError as exc:
                    # These are explicit numerical failures of the shared
                    # backend. Unclassified/programming errors still propagate.
                    if (str(exc) == "Sparse solve produced NaN/Inf values" or
                            str(exc).startswith("Cannot solve with failed factorization:")):
                        raise _NumericalFailure("reference modal sparse solve failed: "+str(exc)) from exc
                    raise
                if not np.isfinite(answer).all():
                    raise _NumericalFailure("finite reference modal inverse action required")
                safe_point("reference modal iteration completed")
                return project(answer)

            operator = LinearOperator(a.shape, matvec=action, dtype=float)
            values, vectors = eigs(operator, k=request.count, which="LM",
                                  v0=np.sin(np.arange(a.shape[0])+.23),
                                  tol=1e-11, maxiter=request.max_iterations)
            safe_point("reference modes after eigensolve")
            abs_a, abs_b, abs_c = abs(a), abs(b), abs(c)
            recovered = []
            for mu, y in zip(values, vectors.T):
                if (not np.isfinite(mu) or abs(mu.imag) > 1e-9*abs(mu.real) or
                        mu.real <= 0. or not np.isfinite(y).all() or
                        np.linalg.norm(y.imag) > 1e-9*np.linalg.norm(y.real)):
                    raise _NumericalFailure("nonphysical or unresolved finite modal pair")
                lam = sigma+1./mu.real
                if self._modal_only:
                    # Resolution screen, not a certified eigenvalue error bound.
                    screen = 64*np.finfo(float).eps*a.shape[0]*(abs(sigma)+abs(1./mu.real))
                    if not np.isfinite(screen) or lam <= screen:
                        raise _NumericalFailure("extra mechanism or unresolved near-zero flexible pair")
                x = scale*y.real
                mass_norm = float(x@b@x)
                if not np.isfinite(lam) or lam <= 0. or not np.isfinite(mass_norm) or mass_norm <= 0.:
                    raise _NumericalFailure("positive finite supported mode and physical inertia required")
                x /= np.sqrt(mass_norm)
                physical = np.asarray(self._T@x[:n])
                if not np.isfinite(x).all() or not np.isfinite(physical).all():
                    raise _NumericalFailure("physical mode normalization is not representable")
                residual = a@x-lam*(b@x)
                bound = abs_a@abs(x)+abs(lam)*(abs_b@abs(x))
                equation = _modal_scaled_residual(residual, bound, scale)
                constraint = _modal_scaled_residual(c@x[:n], abs_c@abs(x[:n]), scale[n:])
                physical_gap = _modal_scaled_residual(
                    self._C@physical, abs(self._C)@abs(physical), scale[n:])
                physical_balance = self._K@physical+self._C.T@x[n:]-lam*(self._M@physical)
                physical_bound = (abs(self._K)@abs(physical)+abs(self._C.T)@abs(x[n:])+
                                  abs(lam)*(abs(self._M)@abs(physical)))
                physical_equation = _modal_scaled_residual(
                    self._project(physical_balance), abs(self._T.T)@physical_bound, scale[:n])
                equation = max(equation, physical_equation)
                constraint = max(constraint, physical_gap)
                if equation > 1e-8 or constraint > 1e-9:
                    raise _NumericalFailure("reference modal equation or constraint residual failed")
                recovered.append((float(np.sqrt(lam)/(2.*np.pi)),
                                  tuple(physical), tuple(x[n:]),
                                  equation, constraint))
            recovered.sort(key=lambda mode: mode[0])
            primal = np.array([mode[1] for mode in recovered]).T
            modal_gram = primal.T@self._M@primal
            if (not np.isfinite(modal_gram).all() or
                    np.max(abs(modal_gram-np.eye(request.count))) > 1e-7):
                raise _NumericalFailure("modal physical-mass orthogonality failed")
            if self._modal_only:
                overlap = float(np.max(abs(rigid["normalized"].T@self._M@primal)))
                if not np.isfinite(overlap) or overlap > 1e-7:
                    raise _NumericalFailure("rigid/flexible physical-mass overlap failed")
            safe_point("reference modes before publication")
            return finish(
                "ok", tuple(mode[0] for mode in recovered),
                tuple(mode[1] for mode in recovered), tuple(mode[2] for mode in recovered),
                tuple(mode[3] for mode in recovered), tuple(mode[4] for mode in recovered),
                handle.factorization_count, handle.solve_count)
        except SolveCancelled as exc:
            return finish(
                "cancelled", factorization_count=handle.factorization_count if handle else 0,
                solve_count=handle.solve_count if handle else 0, failure_reason=str(exc))
        except (ArpackError, _NumericalFailure) as exc:
            return finish(
                "failed", factorization_count=handle.factorization_count if handle else 0,
                solve_count=handle.solve_count if handle else 0, failure_reason=str(exc))
        finally:
            # Do not retain a numerical factor through an exception traceback.
            handle = None

    def _panel_result(self, *args, **kwargs):
        if self._physical_footprint_law:
            return B2PhysicalRigidFootprintResult(*args,footprint_ids=self.footprint_ids,**kwargs)
        if self._point_trace_policy is not None:
            return B2LoadedPointPanelResult(*args, **kwargs)
        if "body_model_identity" in kwargs:
            result_type = B2BodyLoadedFootprintResult if self.footprints else B2BodyLoadedPanelResult
            if self.footprints:
                kwargs["footprint_ids"] = self.footprint_ids
            return result_type(*args, **kwargs)
        if self.footprints:
            return ExperimentalB2Q4FootprintResult(
                *args, footprint_ids=self.footprint_ids, **kwargs)
        return ExperimentalB2Q4PanelResult(*args, **kwargs)

    def solve_linear(self, load, *, recover_members=None,
                     selected_footprints: B2NodalRigidFootprintSelection | B2PhysicalRigidFootprintSelection | None = None
                     ) -> ExperimentalB2Q4PanelResult:
        if self._closed:
            raise ExperimentalB2Q4PanelError("experimental panel is closed")
        if self._modal_only:
            raise ValueError("free-free modal-only policy does not admit static solves")
        selector = checked_physical_selection if self._physical_footprint_law else checked_selection
        chosen_footprints = selector(selected_footprints, self.footprints)
        if self._physical_footprint_law and isinstance(load,B2ReferenceBodyLoad):
            raise ValueError("physical footprint body-load driver not admitted")
        recovery = _checked_recovery(
            recover_members, {member_id for member_id, _ in self._members})
        try:
            applied, _ = self._body_load_snapshot(load)
        except (TypeError, ValueError, OverflowError) as exc:
            if isinstance(load, B2ReferenceBodyLoad):
                raise
            raise ValueError("finite panel load vector required") from exc
        if applied.shape != self._u0.shape or not np.isfinite(applied).all():
            raise ValueError("finite panel load vector required")
        acceleration = load.acceleration_m_s2 if isinstance(load, B2ReferenceBodyLoad) else None
        body_gap = body_gap_scale = None
        try:
            rhs = np.r_[self._project(applied-self._ku0), -self._cu0]
            if self._point_trace_policy is not None:
                body_gap = np.zeros(len(self._cu0))
                body_gap_scale = np.zeros(len(self._cu0))
                for compiled_load in self._loaded_members.values():
                    values, scales = compiled_load.body_constraint(acceleration)
                    slots = [6*index+channel for index in compiled_load.point_indices for channel in range(6)]
                    body_gap[slots] = values
                    body_gap_scale[slots] = scales
                rhs[-len(self._cu0):] -= body_gap
        except (_NumericalFailure, RepresentabilityError) as exc:
            return self._panel_result(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count, failure_reason=str(exc))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled_rhs = self._factor_scale*rhs
        if (not np.isfinite(scaled_rhs).all() or
                np.any((rhs != 0.) & (scaled_rhs == 0.))):
            return self._panel_result(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count,
                failure_reason="panel right-hand side scaling is not representable")
        try:
            scaled_answer = np.asarray(self._handle.solve(scaled_rhs), dtype=float)
            answer = self._factor_scale*scaled_answer
            if self._physical_footprint_law and answer.shape == rhs.shape and np.isfinite(answer).all():
                answer = self._correct_physical_equilibrium(
                    scaled_answer, scaled_rhs, applied)
                if self._physical_work_ratio(answer, applied) > 1.:
                    raise _NumericalFailure("panel physical footprint work consistency failed")
        except (RuntimeError, _NumericalFailure) as exc:
            return self._panel_result(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count, failure_reason=str(exc))
        try:
            if answer.shape != rhs.shape or not np.isfinite(answer).all():
                raise _NumericalFailure("finite panel solution required")
            q, lam = answer[:-len(self._cu0)], answer[-len(self._cu0):]
            u = np.asarray(self._T @ q).reshape(-1)+self._u0
            if not np.isfinite(u).all():
                raise _NumericalFailure("finite panel solution required")
            gap = np.asarray(self._C @ u).reshape(-1)
            compliance_gap_scale = None
            if self._point_trace_policy is not None:
                compliance_gap, compliance_gap_scale = checked_sparse_action(self._H, lam)
                gap = gap+body_gap-compliance_gap
            force = np.asarray(self._K @ u).reshape(-1)
            constraint_force = np.asarray(self._C.T @ lam).reshape(-1)
            reaction = force+constraint_force-applied
            projected = self._project(reaction)
            # Each row uses the absolute contribution sum in its own physical
            # work unit. A 1 N floor would mask a failed micro-/nano-load;
            # a single mixed force/moment norm would depend on chosen units.
            gap_scale = np.asarray(self._abs_C @ np.abs(u)).reshape(-1)
            if self._point_trace_policy is not None:
                gap_scale += body_gap_scale+compliance_gap_scale
            physical_force_scale = np.asarray(
                self._abs_K @ np.abs(u)+self._abs_Ct @ np.abs(lam)+
                np.abs(applied)).reshape(-1)
            equilibrium_scale = np.asarray(
                self._abs_Tt @ physical_force_scale).reshape(-1)
            # Sparse factorization can leave roundoff in a locally quiet DOF.
            # Its allowance follows the current force and moment magnitudes,
            # separately, then maps through the affine transpose; it has no
            # fixed N, Nm, m or rad floor.
            by_node = physical_force_scale.reshape(-1, 6)
            force_roundoff = float(np.max(by_node[:, :3]))
            moment_roundoff = float(np.max(by_node[:, 3:]))
            physical_roundoff = np.tile(
                (force_roundoff,)*3+(moment_roundoff,)*3,
                len(by_node))
            projected_roundoff = np.asarray(
                self._abs_Tt @ physical_roundoff).reshape(-1)
            roundoff_fraction = len(rhs)*np.finfo(float).eps
            if roundoff_fraction >= 1e-9:
                raise _NumericalFailure("panel equilibrium resolution insufficient")
            explicit_rhs = np.abs(self._project(applied))
            resolution_floor = roundoff_fraction*projected_roundoff
            # Prescribed offsets can induce reduced loads much smaller than
            # the model-wide reaction scale. Their physical effect is checked
            # by the full equilibrium/compatibility test below. This extra
            # small-load resolution guard applies to explicit applied work,
            # so a superposed tiny external load cannot disappear silently.
            active = explicit_rhs > 0.
            certified = False
            if np.any(resolution_floor[active] >
                      1e-9*explicit_rhs[active]):
                certified = (bool(self.footprints) or self._point_trace_policy is not None) and self._explicit_load_resolved(
                    answer, rhs, explicit_rhs, applied=applied, reconstructed=u)
                if not certified:
                    raise _NumericalFailure("panel mixed-load resolution insufficient")
            equilibrium_tolerance = (1e-9*equilibrium_scale+
                                     roundoff_fraction*projected_roundoff)
            gap_tolerance = 1e-9*gap_scale
            if (self.footprints and np.any(np.abs(gap) > gap_tolerance) and
                    (certified or self._explicit_load_resolved(answer, rhs, explicit_rhs, applied=applied, reconstructed=u))):
                gap_tolerance = self._footprint_gap_tolerance(u, gap_scale, roundoff_fraction)
            for item in self._footprint_incidence:
                width=getattr(item,"row_count",6)
                rows = slice(item.row_start, item.row_start+width)
                cap = 1e-9*np.array((self._compatibility_length,)*3+(1.,)*(width-3))
                gap_tolerance[rows] = np.minimum(gap_tolerance[rows], cap)
            if (not np.isfinite(reaction).all() or
                    not np.isfinite(gap).all() or not np.isfinite(projected).all() or
                    not np.isfinite(gap_scale).all() or
                    not np.isfinite(equilibrium_tolerance).all() or
                    not np.isfinite(gap_tolerance).all() or
                    np.any(np.abs(gap) > gap_tolerance) or
                    np.any(np.abs(projected) > equilibrium_tolerance)):
                raise _NumericalFailure("panel point gap or equilibrium failed")
        except (_NumericalFailure, RepresentabilityError) as exc:
            return self._panel_result(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count, failure_reason=str(exc))
        recovered = []
        equilibrated = []
        loaded_poses = []
        loaded_states = {}
        if self._point_trace_policy is not None:
            try:
                loaded_states = {member_id: compiled_load.bubble_state(acceleration, lam)
                                 for member_id, compiled_load in self._loaded_members.items()}
            except RepresentabilityError as exc:
                return self._panel_result("failed", factorization_count=self._handle.factorization_count,
                    solve_count=self._handle.solve_count, failure_reason=str(exc))
        for member_id, (member, slots) in self._members:
            if recovery is not None and member_id not in recovery:
                continue
            local = member.local_dof_map() @ u[slots]
            if not np.isfinite(local).all():
                return self._panel_result(
                    "failed", factorization_count=self._handle.factorization_count,
                    solve_count=self._handle.solve_count,
                    failure_reason="finite physical B2 recovery coordinates required")
            result = member.operator.recover_sections(
                local, stations=(0., .5, 1.) if recovery is None
                else recovery[member_id])
            if (not np.isfinite(result.resultants).all() or
                    not np.isfinite(result.strains).all()):
                return self._panel_result(
                    "failed", factorization_count=self._handle.factorization_count,
                    solve_count=self._handle.solve_count,
                    failure_reason="finite physical B2 recovery required")
            recovered.append((member_id, tuple(
                tuple(map(float, row)) for row in result.resultants)))
            if isinstance(load, B2ReferenceBodyLoad) or self._point_trace_policy is not None:
                try:
                    if self._point_trace_policy is not None:
                        compiled_load = self._loaded_members[member_id]
                        uniform, point_wrenches, _ = loaded_states[member_id]
                        actions = compiled_load.section_actions(point_wrenches)
                        poses = compiled_load.recover_pose(local, result.stations, uniform, point_wrenches)
                        loaded_poses.append(B2LoadedMemberPoses(member_id, tuple(map(float, result.stations)),
                            tuple(tuple(map(float, row)) for row in poses)))
                    else:
                        with np.errstate(over="ignore", invalid="ignore"):
                            uniform = (member.definition.mass.matrix_at_reference_axis()[:, :3]@
                                       member.definition.local_frame_rows()@np.asarray(load.acceleration_m_s2))
                            actions = tuple((station, pullback@lam[6*i:6*i+6]) for i, station, pullback in
                                            self._section_point_actions.get(member_id, ()))
                    if (not np.isfinite(uniform).all() or
                            any(not np.isfinite(wrench).all() for _, wrench in actions)):
                        raise RepresentabilityError("physical section actions are not representable")
                    fields = member.operator.recover_equilibrated_sections(local,
                        uniform_wrench=uniform, point_actions=actions, stations=result.stations)
                except RepresentabilityError as exc:
                    return self._panel_result("failed", factorization_count=self._handle.factorization_count,
                        solve_count=self._handle.solve_count, failure_reason=str(exc))
                rows = lambda values: tuple(tuple(map(float, row)) for row in values)
                equilibrated.append(B2LoadedMemberSections(member_id, tuple(map(float, fields.stations)),
                    rows(fields.resultants_left), rows(fields.resultants_right),
                    rows(fields.strains_left), rows(fields.strains_right),
                    tuple(map(float, fields.endpoint_tractions))))
        energy = .5*float(u @ force)
        if not isfinite(energy):
            return self._panel_result(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count,
                failure_reason="finite panel strain energy required")
        extra = {}
        if self._physical_footprint_law:
            extra.update(constraint_multipliers=tuple(map(float,lam)),constraint_gaps=tuple(map(float,gap)),
                constraint_rows=tuple((item.row_start+j,item.footprint_id,item.shell_node,item.beam_node,component)
                    for item in self._footprint_incidence for j,component in enumerate(("translation_x","translation_y","translation_z","director_1","director_2"))))
        if isinstance(load, B2ReferenceBodyLoad):
            extra.update(body_model_identity=load.model_identity,
                         body_acceleration_m_s2=load.acceleration_m_s2,
                         equilibrated_sections=tuple(equilibrated))
        if self._point_trace_policy is not None:
            try:
                loaded_energy = fsum((energy, *(state[2] for state in loaded_states.values())))
            except OverflowError:
                loaded_energy = float("inf")
            if not isfinite(loaded_energy) or loaded_energy < 0.:
                return self._panel_result("failed", factorization_count=self._handle.factorization_count,
                    solve_count=self._handle.solve_count, failure_reason="loaded strain energy is not representable")
            extra.update(equilibrated_sections=tuple(equilibrated), loaded_member_poses=tuple(loaded_poses),
                         loaded_strain_energy=loaded_energy)
        if chosen_footprints:
            try:
                wrench_recovery = physical_selected_wrenches if self._physical_footprint_law else selected_wrenches
                extra["footprint_wrenches"] = wrench_recovery(
                    self._footprint_incidence, chosen_footprints, lam, gap)
            except (ValueError, OverflowError) as exc:
                return self._panel_result(
                    "failed", factorization_count=self._handle.factorization_count,
                    solve_count=self._handle.solve_count, failure_reason=str(exc))
        return self._panel_result(
            "completed", tuple(map(float, u)),
            tuple((point_id, tuple(map(float, lam[6*i:6*i+6])))
                  for i, point_id in enumerate(self.point_ids)),
            tuple(map(float, reaction)),
            tuple((point_id, tuple(map(float, gap[6*i:6*i+6])))
                  for i, point_id in enumerate(self.point_ids)),
            tuple(recovered), energy, self._handle.factorization_count,
            self._handle.solve_count, **extra)

    def solve_linear_many(
            self, cases: Sequence[tuple[str, object]], *,
            selected_dofs: Sequence[int],
            selected_points: Sequence[str] = (),
            selected_footprints: B2NodalRigidFootprintSelection | B2PhysicalRigidFootprintSelection | None = None,
            recover_members: Mapping[str, Sequence[float]] | None = None,
            cancellation_token: CancellationToken | None = None,
            on_case: Callable[[str], None] | None = None,
            ) -> ExperimentalB2Q4PanelBatchResult:
        """Solve ordered cases with one factorization and bounded case storage.

        Each result is complete before it joins the returned prefix. Numerical
        failure, cooperative cancellation or owner closure stops further cases.
        Closing from a completion callback preserves the complete prefix; a
        final-case closure still returns completed. Callback and programming
        errors propagate. This development route uses one RHS at a
        time, avoiding full displacement/reaction batches for large panels.
        Selected displacement, reaction and local member-resultant envelopes
        describe only complete cases in the returned prefix, with the first
        submitted case governing ties.
        """
        if self._closed:
            raise ExperimentalB2Q4PanelError("experimental panel is closed")
        if self._modal_only:
            raise ValueError("free-free modal-only policy does not admit static solves")
        selector = checked_physical_selection if self._physical_footprint_law else checked_selection
        chosen_footprints = selector(selected_footprints, self.footprints)
        try:
            chosen = tuple(selected_dofs)
            chosen_points = tuple(selected_points)
            submitted = tuple(cases)
        except TypeError as exc:
            raise ValueError("ordered B2 panel cases and selection required") from exc
        if (not chosen or any(type(value) is not int or
                              not 0 <= value < len(self._u0)
                              for value in chosen) or
                len(set(chosen)) != len(chosen)):
            raise ValueError("unique selected panel coordinate indices required")
        if (any(type(point_id) is not str or
                point_id not in self.point_ids for point_id in chosen_points) or
                len(set(chosen_points)) != len(chosen_points)):
            raise ValueError("unique selected panel point IDs required")
        if (not submitted or
                any(type(item) not in (tuple, list) or len(item) != 2 or
                    type(item[0]) is not str or not item[0].strip()
                    for item in submitted) or
                len({item[0] for item in submitted}) != len(submitted)):
            raise ValueError("unique nonempty ordered B2 panel case IDs required")
        # Caller-owned list pairs may be changed by a completion callback.
        # Snapshot their inventory/references now; later load-value mutation
        # remains checked by the existing bounded fingerprints below.
        submitted = tuple((case_id, load) for case_id, load in submitted)
        if self._physical_footprint_law and any(isinstance(load,B2ReferenceBodyLoad) for _,load in submitted):
            raise ValueError("physical footprint body-load driver not admitted")
        selected_rows = tuple((item.row_start+j,item.footprint_id,item.shell_node,item.beam_node,component)
            for item in self._footprint_incidence if item.footprint_id in chosen_footprints
            for j,component in enumerate(("translation_x","translation_y","translation_z","director_1","director_2"))) if self._physical_footprint_law else ()
        if cancellation_token is not None and not isinstance(
                cancellation_token, CancellationToken):
            raise TypeError("B2 panel cancellation token required")
        if on_case is not None and not callable(on_case):
            raise TypeError("B2 panel case callback required")
        recovery = _checked_recovery(
            {} if recover_members is None else recover_members,
            {member_id for member_id, _ in self._members})
        # Counts outlive a callback closing the owner, without retaining its
        # factor in this operation or a propagated callback-error traceback.
        factorization_count = self._handle.factorization_count
        solve_count = self._handle.solve_count
        fingerprints = []
        for _case_id, load in submitted:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                break
            snapshot, fingerprint = self._body_load_snapshot(load)
            fingerprints.append(fingerprint)
            del snapshot
        completed = []

        def envelopes(attribute):
            if not completed:
                return ()
            extrema = []
            for index, dof in enumerate(chosen):
                # min/max preserve the first case on exact ties. No rounding,
                # absolute envelope or physical-unit conversion is introduced.
                minimum = min(completed, key=lambda case: getattr(case, attribute)[index])
                maximum = max(completed, key=lambda case: getattr(case, attribute)[index])
                extrema.append(ExperimentalB2Q4PanelResponseEnvelope(
                    dof, getattr(minimum, attribute)[index], minimum.case_id,
                    getattr(maximum, attribute)[index], maximum.case_id))
            return tuple(extrema)

        def member_envelopes():
            if not completed or not recovery:
                return ()
            sections = tuple((case.case_id, dict(case.beam_resultants_local))
                             for case in completed)
            extrema = []
            for member_id, stations in recovery.items():
                if any(member_id not in rows or len(rows[member_id]) != len(stations)
                       for _, rows in sections):
                    raise RuntimeError("incomplete selected B2 member recovery")
                for station_index, station in enumerate(stations):
                    for component_index, component in enumerate(
                            ("N", "Qy", "Qz", "Mx", "My", "Mz")):
                        values = tuple((case_id, rows[member_id][station_index][component_index])
                                       for case_id, rows in sections)
                        minimum = min(values, key=lambda item: item[1])
                        maximum = max(values, key=lambda item: item[1])
                        extrema.append(ExperimentalB2Q4PanelMemberResultantEnvelope(
                            member_id, station_index, station, component,
                            minimum[1], minimum[0], maximum[1], maximum[0]))
            return tuple(extrema)

        def finish(status, failure_reason=None):
            result_type = (ExperimentalB2Q4FootprintBatchResult if self.footprints
                           else ExperimentalB2Q4PanelBatchResult)
            extra = {"selected_footprints": chosen_footprints} if self.footprints else {}
            if self._physical_footprint_law:
                result_type = B2PhysicalRigidFootprintBatchResult
                extra["constraint_rows"] = selected_rows
            return result_type(
                status, chosen, chosen_points, tuple(completed),
                factorization_count, solve_count,
                failure_reason=failure_reason,
                displacement_envelopes=envelopes("selected_displacements"),
                affine_reaction_envelopes=envelopes("selected_affine_reactions"),
                member_resultant_envelopes=member_envelopes(), **extra)

        for index, (case_id, load) in enumerate(submitted):
            if self._closed or (cancellation_token is not None and
                                cancellation_token.is_cancelled):
                break
            snapshot, fingerprint = self._body_load_snapshot(load)
            if fingerprint != fingerprints[index]:
                raise ValueError("B2 panel load case changed after submission")
            result = self.solve_linear(load if isinstance(load, B2ReferenceBodyLoad) else snapshot, recover_members=recovery,
                                       selected_footprints=selected_footprints)
            factorization_count = result.factorization_count
            solve_count = result.solve_count
            if result.status != "completed":
                return finish("failed", f"{case_id}: {result.failure_reason}")
            point_multipliers = dict(result.point_multipliers)
            point_gaps = dict(result.point_gaps)
            case_type = (ExperimentalB2Q4FootprintSelectedCase if self.footprints
                         else ExperimentalB2Q4PanelSelectedCase)
            extra = {"footprint_wrenches": result.footprint_wrenches} if self.footprints else {}
            if self._physical_footprint_law:
                case_type = B2PhysicalRigidFootprintSelectedCase
                row_indices = tuple(row[0] for row in selected_rows)
                extra.update(constraint_multipliers=tuple(result.constraint_multipliers[i] for i in row_indices),
                             constraint_gaps=tuple(result.constraint_gaps[i] for i in row_indices))
            if isinstance(load, B2ReferenceBodyLoad):
                case_type = B2BodyLoadedFootprintSelectedCase if self.footprints else B2BodyLoadedPanelSelectedCase
                extra.update(body_model_identity=result.body_model_identity,
                             body_acceleration_m_s2=result.body_acceleration_m_s2,
                             equilibrated_sections=result.equilibrated_sections)
            if self._point_trace_policy is not None:
                case_type = B2LoadedPointPanelSelectedCase
                extra.update(equilibrated_sections=result.equilibrated_sections,
                             loaded_member_poses=result.loaded_member_poses,
                             loaded_strain_energy=result.loaded_strain_energy)
            completed.append(case_type(
                case_id,
                tuple(result.displacements[index] for index in chosen),
                tuple(result.affine_reactions[index] for index in chosen),
                tuple((point_id, point_multipliers[point_id])
                      for point_id in chosen_points),
                tuple((point_id, point_gaps[point_id])
                      for point_id in chosen_points),
                result.beam_resultants_local,
                result.internal_strain_energy, **extra))
            if on_case is not None:
                on_case(case_id)
        return finish(
            "cancelled" if len(completed) != len(submitted) else "completed",
            failure_reason=("experimental panel closed" +
                            (" after " + completed[-1].case_id if completed else "")
                            if self._closed and len(completed) != len(submitted)
                            else None))

    def close(self) -> None:
        self._closed = True
        try:
            _retire_private_q4_mesh(self)
        finally:
            self._handle = None
            self._physical_scaled_kkt = None
            self._attachment_elimination = None
            if self._point_trace_policy is not None:
                for compiled_load in self._loaded_members.values():
                    compiled_load._recovery_maps.clear()
                self._loaded_members.clear()
                self._H = self._abs_H = self._resolution_kkt = None
            self._resolution_kkt = None
            self._affine_resolution_cache = None
            self._buckling_model = None
            self._buckling_shells = ()


def _release_coo_scratch(rows, columns, values, diagnostic, phase):
    # CSR owns its arrays; these compiler lists have no later mechanics use.
    counts = tuple(map(len, (rows, columns, values))) if diagnostic is not None else None
    rows.clear()
    columns.clear()
    values.clear()
    if diagnostic is not None:
        diagnostic.record_coo_release(phase, counts)


def _scatter(rows, columns, values, slots, local):
    width = len(slots)
    rows.extend(np.repeat(slots, width))
    columns.extend(np.tile(slots, width))
    values.extend(np.asarray(local).ravel())


def _modal_scaled_residual(residual, bound, row_scale):
    """Infinity backward error in the fixed congruence row representation.

    Near-zero components of an uncoupled mode are not separate relative-error
    denominators. Scaling the rows before the norm keeps their units consistent
    with the factorization coordinates without an arbitrary physical floor.
    """
    residual, bound = np.asarray(residual), np.asarray(bound)
    if not np.isfinite(residual).all() or not np.isfinite(bound).all():
        return float("inf")
    numerator = np.max(abs(residual)*row_scale, initial=0.)
    denominator = np.max(bound*row_scale, initial=0.)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("inf")
    if denominator == 0.:
        return 0. if numerator == 0. else float("inf")
    return float(numerator/denominator)


def _free_b2_rigid_space(owner, scale, scaled_mass, safe_point):
    """Common physical velocities, with only a bounded six-by-six Gram solve."""
    origin = owner._modal_positions[:len(owner.shell_node_ids)].mean(axis=0)
    canonical = np.zeros((owner._K.shape[0], 6))
    for i, position in enumerate(owner._modal_positions):
        slots = slice(6*i, 6*i+3)
        canonical[slots, :3] = np.eye(3)
        for j in range(3):
            canonical[slots, 3+j] = np.cross(np.eye(3)[j], position-origin)
        canonical[6*i+3:6*i+6, 3:] = np.eye(3)
    n = owner._K.shape[0]
    equations, gaps = [], []
    for column in canonical.T:
        safe_point("reference modes rigid field")
        equation = _modal_scaled_residual(owner._K@column,
            abs(owner._K)@abs(column), scale[:n])
        gap = _modal_scaled_residual(owner._C@column,
            abs(owner._C)@abs(column), scale[n:])
        if equation > 1e-8 or gap > 1e-9:
            raise _NumericalFailure("canonical free-free rigid field violates physical K or point C")
        equations.append(equation); gaps.append(gap)
    gram = canonical.T@owner._M@canonical
    if (not np.isfinite(gram).all() or
            np.max(abs(gram-gram.T)) > 1e-10*np.max(abs(gram))):
        raise _NumericalFailure("canonical rigid kinetic Gram symmetry unresolved")
    try:
        lower = np.linalg.cholesky(gram)
        normalized = canonical@np.linalg.solve(lower.T, np.eye(6))
    except np.linalg.LinAlgError as exc:
        raise _NumericalFailure("six physical rigid fields require positive kinetic Gram") from exc
    if (not np.isfinite(normalized).all() or
            np.max(abs(normalized.T@owner._M@normalized-np.eye(6))) > 1e-7):
        raise _NumericalFailure("rigid kinetic normalization unresolved")
    scaled = np.vstack((normalized, np.zeros((owner._C.shape[0], 6))))/scale[:, None]
    if (not np.isfinite(scaled).all() or
            np.max(abs(scaled.T@scaled_mass@scaled-np.eye(6))) > 1e-7):
        raise _NumericalFailure("rigid congruence reconstruction unresolved")
    return dict(origin=tuple(map(float, origin)), canonical=canonical,
        normalized=normalized, scaled=scaled, gram=gram,
        equations=tuple(equations), gaps=tuple(gaps))


class ExperimentalB2MixedPanelAnalysis(ExperimentalB2Q4PanelAnalysis):
    """Explicit mixed-shell admission using the existing static sparse owner."""

    _mixed_shells = True

    @property
    def capabilities(self) -> ExperimentalB2MixedPanelCapabilities:
        existing = super().capabilities
        capability_type = (ExperimentalB2MixedReferenceModalCapabilities if
                           self._reference_inertia_policy is not None else ExperimentalB2MixedPanelCapabilities)
        extra = ({"shell_mass_policies": tuple((shell_id,
                  S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1 if formulation ==
                  S3_V2D_FORMULATION_ID else Q4_MASS_POLICY)
                  for shell_id, formulation in self._shell_formulations)}
                 if self._reference_inertia_policy is not None else {})
        if self._reference_inertia_policy is not None:
            extra["boundary_policy"] = self._reference_boundary_policy
        return capability_type(
            point_law_id=tuple(dict.fromkeys(law for _, law in self._point_laws)),
            attachment=existing.attachment,
            operations=existing.operations,
            body_load_domain=existing.body_load_domain,
            spectral_domain=(existing.spectral_domain if self._modal_only or self._physical_footprint_law else
                             "experimental virgin supported finite descriptor pairs; heterogeneous Q4 director and native S3 lumped translation inertia; S3 rotary inertia omitted; completeness unqualified; prestress unsupported"
                             if self._reference_inertia_policy is not None else
                             "native S3 V2D physical mass and modes deferred" if self._has_s3
                             else existing.spectral_domain),
            shell_formulations=self._shell_formulations,
            point_laws=self._point_laws,
            footprint_law_id=existing.footprint_law_id if self._physical_footprint_law else FOOTPRINT_LAW_ID if self.footprints else None,
            footprints=self.footprints, **extra)


def create_experimental_b2_q4_panel_analysis(*args, **kwargs):
    return ExperimentalB2Q4PanelAnalysis(*args, **kwargs)
