"""Explicit experimental entry for the developing stiffened-shell profiles.

Capabilities report the exact implemented subset. G0-G6 qualification and
release are separate decisions tied to final installed artifacts.
"""

from __future__ import annotations

from ._stiffened_shell_b2_reference_buckling import (
    B2ReferenceBucklingRequest, B2ReferenceBucklingResult,
    B2ReferenceBucklingCapabilities, B2SectionTractionQuadrature,
    B2PhysicalFootprintBucklingRequest, B2PhysicalFootprintBucklingResult,
)

from ._stiffened_shell_b2_physical_rigid_footprint import (
    B2PhysicalRigidFootprint, B2PhysicalRigidFootprintSelection,
    B2PhysicalRigidFootprintWrench, LAW_ID as B2_PHYSICAL_RIGID_FOOTPRINT_DIRECTOR_REFERENCE_V1_EXPERIMENTAL,
)

from dataclasses import dataclass
from ._stiffened_shell_b2_definition_ownership import OwnedB2Definition as _OwnedB2Definition
from ._stiffened_shell_selected_recovery import GERecoverySelection, SelectedGERecovery
from ._stiffened_shell_ge_distributed_load import GEBeamLineAction, GEBeamDistributedLoad, GEDistributedRecovery
from ._stiffened_shell_ge_distributed_programme import GEBeamDistributedProgramme
from ._stiffened_shell_ge_pressure_load import (
    GEShellPressureAction, GEShellPressureLoad, GEPressureRecovery,
    VECTOR_EQUILIBRIUM_POLICY as GE_PRESSURE_VECTOR_EQUILIBRIUM_V1_DEV,
)
from ._stiffened_shell_ge_body_load import GEBodyLoad, GEBodyRecovery
from ._stiffened_shell_staged_load_experimental import StagedDeadLoadProgramme
from ._stiffened_shell_ge_reference_modal import (
    GEReferenceModesRequest, GEReferenceModesResult, GEFreeReferenceModesResult,
    GEMixedReferenceModesRequest, GEMixedReferenceModesResult,
    GE_MIXED_FREE_FREE_MODAL_ONLY_V1,
    GE_FREE_FREE_MODAL_ONLY_V1,
)
from ._stiffened_shell_ge_point_buckling import GEPointBucklingRequest, GEPointBucklingResult, GELineBucklingRequest, GELineBucklingResult
from ._stiffened_shell_ge_point_buckling import GEMixedBucklingRequest, GEMixedBucklingResult
from ._stiffened_shell_ge_point_buckling import GENativeLineBucklingRequest, GENativeLineBucklingResult
from ._stiffened_shell_bl5_native_elastic import BL5NativeElasticDomainError

from ._b2_physical_linear_definition import (
    PROFILE_ID as B2_E4_PL_FAST_LINEAR_V1,
    PhysicalB2Member, CompiledPhysicalB2Member, MassMetadata,
)
from ._b2_physical_flexibility_linear import (
    LinearSection, RepresentabilityError, OutputRepresentationError,
    OutputRepresentationCertificate,
    OUTPUT_REPRESENTATION_ID as B2_BOUNDED_OUTPUT_REPRESENTATION_V1,
)
from ._stiffened_shell_b2_experimental import ExperimentalB2Analysis
from ._stiffened_shell_point_recovery import PointTransferWrench
from ._stiffened_shell_ge_distributing_finite import (
    FiniteDistributingFootprintDefinition as GEFiniteDistributingFootprint,
    FiniteDistributingFootprintWrench as GEFiniteDistributingFootprintWrench,
    LAW as GE_DISTRIBUTING_FINITE_DIRECTOR_POLAR_V1_EXPERIMENTAL,
)
from ._stiffened_shell_ge_distributing_reference import (
    DistributingFootprintDefinition as GEReferenceDistributingFootprint,
    DistributingFootprintSelection as GEReferenceDistributingFootprintSelection,
    DistributingFootprintWrench as GEReferenceDistributingFootprintWrench,
    LAW as GE_DISTRIBUTING_REFERENCE_LINEAR_V1_EXPERIMENTAL,
)
from ._stiffened_shell_nodal_footprint import (
    INCREMENT_ORIGIN_CHART_POLICY as GE_NODAL_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2,
)
from ._stiffened_shell_increment_control import (
    AdaptiveIncrementControl, IncrementAttempt, FiniteIncrementFailure,
    AdaptiveIncrementExhausted,
)
from ._stiffened_shell_b2_q4_point_experimental import (
    ExperimentalB2Q4PointAnalysis,
)
from ._stiffened_shell_b2_q4_panel_experimental import (
    ExperimentalB2Q4PanelAnalysis, Q4PanelShell, Q4B2Point,
    ExperimentalB2MixedPanelAnalysis, S3V2DPanelShell,
    ExperimentalB2MixedPanelCapabilities,
    ExperimentalB2MixedReferenceModalCapabilities,
    S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1,
    ExperimentalB2Q4PanelResponseEnvelope,
    ExperimentalB2Q4PanelMemberResultantEnvelope,
    B2ReferenceModesRequest, B2ReferenceModesResult,
    B2FreeReferenceModesResult, B2_FREE_FREE_MODAL_ONLY_V1,
    B2ReferenceBodyLoad,
    B2LoadedMemberSections, B2BodyLoadedPanelResult, B2BodyLoadedFootprintResult,
    B2LoadedMemberPoses, B2LoadedPointPanelResult, B2LoadedPointPanelSelectedCase,
    ExperimentalB2LoadedPointCapabilities, B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV,
    B2BodyLoadedPanelSelectedCase, B2BodyLoadedFootprintSelectedCase,
    B2FootprintReferenceModesResult, B2ModalConstraintRow,
    B2PhysicalRigidFootprintSelectedCase, B2PhysicalRigidFootprintBatchResult,
    ExperimentalB2Q4FootprintResult, ExperimentalB2Q4FootprintSelectedCase,
    ExperimentalB2Q4FootprintBatchResult, ExperimentalB2Q4FootprintCapabilities,
)
from ._stiffened_shell_physical_q4_mass import POLICY as Q4_DIRECTOR_MASS_POLICY
from ._stiffened_shell_b2_nodal_footprint import (
    B2NodalRigidFootprint, B2NodalRigidFootprintSelection,
    B2NodalRigidFootprintWrench, B2NodalFootprintNodeAction,
)

from ._stiffened_shell_ge_experimental import (
    PROFILE as B3_GE_E4_PL_STIFFENED_SHELL_V1,
    ExperimentalGEAnalysis, ExperimentalL1Line, ExperimentalBL5Line,
    FixedCoordinate,
    LinearDefinition, LinearResult,
    NATIVE_SHELL_COORDINATE_SUPPORT_V1, HISTORICAL_SHELL_SUPPORT_POLICY,
    NodalAction, PointConnection, ShellPanel,
    NodalRigidFootprint, NodalRigidFootprintSelection, NodalRigidFootprintWrench,
    Q4_ALGORITHMIC_PULLBACK_V1, BL5_GENERALIZED_MATERIAL_V1, BL5_COUPLED_J2_MATERIAL_V1, BL5_ELASTIC_NATIVE_FIBRE_V1, BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1, BL5_NATIVE_FIBRE_MATERIAL_V1,
    GE_FINITE_DISTRIBUTING_GENERALIZED_MATERIAL_V1, GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1,
    GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1, GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1,
    GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1, GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1,
)


@dataclass(frozen=True)
class B2LinearDefinition(_OwnedB2Definition):
    """Experimental physical B2 graph with an explicit audited affine plan.

    Shell attachment is not yet implemented by this definition.  The separate
    profile identity is retained for the intended complete workflow; runtime
    capabilities state the smaller currently admitted operation set.
    """

    profile: str
    members: tuple[PhysicalB2Member | CompiledPhysicalB2Member, ...]
    transformation: object
    prescribed_offset: object
    selected_dofs: object
    max_rhs_chunk: int = 8


@dataclass(frozen=True)
class B2Q4PointDefinition(_OwnedB2Definition):
    """One elastic Q4 and one physical B2 member with a reference point tie."""

    profile: str
    member: PhysicalB2Member | CompiledPhysicalB2Member
    shell_vertices_m: object
    shell_E: float
    shell_nu: float
    shell_thickness_m: float
    station: float
    transformation: object
    prescribed_offset: object
    offset_local_m: tuple[float, float, float] = (0., 0., 0.)
    reference_normal: tuple[float, float, float] = (0., 0., 1.)


@dataclass(frozen=True)
class B2Q4PanelDefinition(_OwnedB2Definition):
    """Experimental reference-linear Q4/B2 points or explicit nodal footprints.

    Empty footprints preserve the historical finite-vertex point route.
    Nonempty footprints admit virgin supported reference modes with complete
    physical mass and coordinate supports. They lock normal spin and retain
    incomplete mechanical qualification.
    """

    profile: str
    shell_nodes_m: object
    shells: tuple[Q4PanelShell, ...]
    members: tuple[PhysicalB2Member | CompiledPhysicalB2Member, ...]
    points: tuple[Q4B2Point, ...]
    transformation: object
    prescribed_offset: object
    footprints: tuple[B2NodalRigidFootprint, ...] = ()
    point_trace_policy: str | None = None


@dataclass(frozen=True)
class B2PhysicalRigidFootprintDefinition(_OwnedB2Definition):
    """Five-row physical director footprint reference-static Q4/S3 owner."""
    profile: str
    shell_nodes_m: object
    shells: tuple
    members: tuple
    physical_footprints: tuple
    transformation: object
    prescribed_offset: object


@dataclass(frozen=True)
class B2MixedPanelDefinition(_OwnedB2Definition):
    """Opt-in elastic Q4/native S3 V2D/B2 finite-vertex point panel.

    Any S3 facet restricts this definition to static solves. S3 inertia,
    spectral analysis and nodal footprints remain deferred.
    """

    profile: str
    shell_nodes_m: object
    shells: tuple[Q4PanelShell | S3V2DPanelShell, ...]
    members: tuple[PhysicalB2Member | CompiledPhysicalB2Member, ...]
    points: tuple[Q4B2Point, ...]
    transformation: object
    prescribed_offset: object
    footprints: tuple[B2NodalRigidFootprint, ...] = ()


@dataclass(frozen=True)
class B2MixedReferenceModalDefinition(_OwnedB2Definition):
    """Virgin reference modes with explicitly declared discrete inertia.

    Q4 facets retain director mass; native S3 facets use translational lumped
    mass and omit all rotary inertia. This does not qualify continuum dynamics.
    Default boundary behavior requires supports. B2_FREE_FREE_MODAL_ONLY_V1
    explicitly admits an identity transformation and disables static operations.
    """

    profile: str
    shell_nodes_m: object
    shells: tuple[Q4PanelShell | S3V2DPanelShell, ...]
    members: tuple[PhysicalB2Member | CompiledPhysicalB2Member, ...]
    points: tuple[Q4B2Point, ...]
    transformation: object
    prescribed_offset: object
    footprints: tuple[B2NodalRigidFootprint, ...] = ()
    inertia_policy: str = S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1
    boundary_policy: str | None = None


def create_analysis(*, definition, resources=None):
    """Create the explicitly selected experimental owner for an admitted profile."""
    if type(definition) is LinearDefinition:
        return ExperimentalGEAnalysis(definition, resources=resources)
    if type(definition) is B2LinearDefinition:
        if definition.profile != B2_E4_PL_FAST_LINEAR_V1 or resources is not None:
            raise ValueError("explicit experimental physical B2 definition required")
        return ExperimentalB2Analysis(
            definition.members, definition.transformation,
            definition.prescribed_offset, definition.selected_dofs,
            max_rhs_chunk=definition.max_rhs_chunk)
    if type(definition) is B2Q4PointDefinition:
        if definition.profile != B2_E4_PL_FAST_LINEAR_V1 or resources is not None:
            raise ValueError("explicit experimental B2/Q4 point definition required")
        return ExperimentalB2Q4PointAnalysis(
            definition.member, definition.shell_vertices_m,
            shell_E=definition.shell_E, shell_nu=definition.shell_nu,
            shell_thickness_m=definition.shell_thickness_m,
            station=definition.station,
            transformation=definition.transformation,
            prescribed_offset=definition.prescribed_offset,
            offset_local_m=definition.offset_local_m,
            reference_normal=definition.reference_normal)
    if type(definition) is B2Q4PanelDefinition:
        if definition.profile != B2_E4_PL_FAST_LINEAR_V1 or resources is not None:
            raise ValueError("explicit experimental B2/Q4 panel definition required")
        return ExperimentalB2Q4PanelAnalysis(
            definition.shell_nodes_m, definition.shells,
            definition.members, definition.points,
            transformation=definition.transformation,
            prescribed_offset=definition.prescribed_offset,
            footprints=definition.footprints,
            point_trace_policy=definition.point_trace_policy)
    if type(definition) is B2MixedPanelDefinition:
        if definition.profile != B2_E4_PL_FAST_LINEAR_V1 or resources is not None:
            raise ValueError("explicit experimental B2 mixed-panel definition required")
        return ExperimentalB2MixedPanelAnalysis(
            definition.shell_nodes_m, definition.shells,
            definition.members, definition.points,
            transformation=definition.transformation,
            prescribed_offset=definition.prescribed_offset,
            footprints=definition.footprints)
    if type(definition) is B2PhysicalRigidFootprintDefinition:
        if definition.profile != B2_E4_PL_FAST_LINEAR_V1 or resources is not None or not definition.physical_footprints:
            raise ValueError("explicit physical rigid footprint definition required")
        owner_type = (ExperimentalB2MixedPanelAnalysis if any(type(shell) is S3V2DPanelShell
                      for shell in definition.shells) else ExperimentalB2Q4PanelAnalysis)
        return owner_type(definition.shell_nodes_m, definition.shells,
            definition.members, (), transformation=definition.transformation,
            prescribed_offset=definition.prescribed_offset, physical_footprints=definition.physical_footprints)
    if type(definition) is B2MixedReferenceModalDefinition:
        if definition.profile != B2_E4_PL_FAST_LINEAR_V1 or resources is not None:
            raise ValueError("explicit experimental B2 mixed reference-modal definition required")
        if (type(definition.inertia_policy) is not str or
                definition.inertia_policy != S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1):
            raise ValueError("explicit native-S3 reference inertia policy required")
        return ExperimentalB2MixedPanelAnalysis(
            definition.shell_nodes_m, definition.shells,
            definition.members, definition.points,
            transformation=definition.transformation,
            prescribed_offset=definition.prescribed_offset,
            footprints=definition.footprints,
            reference_inertia_policy=definition.inertia_policy,
            reference_boundary_policy=definition.boundary_policy)
    raise ValueError("unsupported stiffened-shell definition/profile")


__all__ = [
    "B2PhysicalRigidFootprintDefinition", "B2PhysicalRigidFootprint", "B2PhysicalRigidFootprintSelection",
    "B2PhysicalRigidFootprintSelectedCase", "B2PhysicalRigidFootprintBatchResult",
    "B2PhysicalRigidFootprintWrench", "B2PhysicalFootprintBucklingRequest", "B2PhysicalFootprintBucklingResult",
    "B2_PHYSICAL_RIGID_FOOTPRINT_DIRECTOR_REFERENCE_V1_EXPERIMENTAL",
    "AdaptiveIncrementControl", "IncrementAttempt", "FiniteIncrementFailure",
    "AdaptiveIncrementExhausted",
    "Q4_ALGORITHMIC_PULLBACK_V1", "BL5_GENERALIZED_MATERIAL_V1", "BL5_COUPLED_J2_MATERIAL_V1",
    "GE_FINITE_DISTRIBUTING_GENERALIZED_MATERIAL_V1", "GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1",
    "GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1", "GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1",
    "GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1", "GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1",
    "B3_GE_E4_PL_STIFFENED_SHELL_V1", "B2_E4_PL_FAST_LINEAR_V1",
    "ShellPanel", "PointConnection", "PointTransferWrench",
    "NodalRigidFootprint", "NodalRigidFootprintSelection", "NodalRigidFootprintWrench",
    "GEReferenceDistributingFootprint", "GEReferenceDistributingFootprintSelection",
    "GEFiniteDistributingFootprint", "GEFiniteDistributingFootprintWrench",
    "GE_DISTRIBUTING_FINITE_DIRECTOR_POLAR_V1_EXPERIMENTAL",
    "GEReferenceDistributingFootprintWrench", "GE_DISTRIBUTING_REFERENCE_LINEAR_V1_EXPERIMENTAL",
    "FixedCoordinate", "NodalAction", "LinearDefinition", "LinearResult",
    "StagedDeadLoadProgramme",
    "GEReferenceModesRequest", "GEReferenceModesResult",
    "GEMixedReferenceModesRequest", "GEMixedReferenceModesResult",
    "GE_MIXED_FREE_FREE_MODAL_ONLY_V1",
    "GEFreeReferenceModesResult", "GE_FREE_FREE_MODAL_ONLY_V1",
    "GE_NODAL_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2",
    "B2LinearDefinition", "B2Q4PointDefinition", "B2Q4PanelDefinition",
    "B2MixedPanelDefinition", "S3V2DPanelShell", "ExperimentalB2MixedPanelCapabilities",
    "B2MixedReferenceModalDefinition", "ExperimentalB2MixedReferenceModalCapabilities",
    "B2FreeReferenceModesResult", "B2_FREE_FREE_MODAL_ONLY_V1",
    "S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1",
    "Q4_DIRECTOR_MASS_POLICY",
    "PhysicalB2Member", "LinearSection", "Q4PanelShell", "Q4B2Point",
    "MassMetadata", "B2ReferenceModesRequest", "B2ReferenceModesResult",
    "B2ReferenceBucklingRequest", "B2ReferenceBucklingResult",
    "B2ReferenceBucklingCapabilities", "B2SectionTractionQuadrature",
    "B2ReferenceBodyLoad",
    "B2LoadedMemberSections", "B2BodyLoadedPanelResult", "B2BodyLoadedFootprintResult",
    "B2LoadedMemberPoses", "B2LoadedPointPanelResult", "B2LoadedPointPanelSelectedCase",
    "ExperimentalB2LoadedPointCapabilities", "B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV",
    "B2BodyLoadedPanelSelectedCase", "B2BodyLoadedFootprintSelectedCase",
    "B2FootprintReferenceModesResult", "B2ModalConstraintRow",
    "B2NodalRigidFootprint", "B2NodalRigidFootprintSelection",
    "B2NodalRigidFootprintWrench", "B2NodalFootprintNodeAction",
    "ExperimentalB2Q4FootprintResult", "ExperimentalB2Q4FootprintSelectedCase",
    "ExperimentalB2Q4FootprintBatchResult", "ExperimentalB2Q4FootprintCapabilities",
    "B2_BOUNDED_OUTPUT_REPRESENTATION_V1", "OutputRepresentationCertificate",
    "OutputRepresentationError", "RepresentabilityError",
    "create_analysis",
]

# Explicit experimental controller; legacy solve defaults are unchanged.
from ._stiffened_shell_displacement_control import DisplacementControl, DisplacementResult
__all__ += ["DisplacementControl", "DisplacementResult"]

from ._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure, DisplacementFailureDetail
__all__ += ["DisplacementCorrectorFailure", "DisplacementFailureDetail"]

from ._stiffened_shell_displacement_increment import DisplacementAttempt, DisplacementIncrementLimit
__all__ += ["DisplacementAttempt", "DisplacementIncrementLimit"]

from ._stiffened_shell_displacement_control import GENERALIZED_BRANCH_EVENT_TRIAL,PAIRED_FIXED_WORK_QUADRATIC_V1
__all__ += ["GENERALIZED_BRANCH_EVENT_TRIAL","PAIRED_FIXED_WORK_QUADRATIC_V1"]

from ._stiffened_shell_generalized_recovery import GeneralizedBeamElementRecovery, GeneralizedBeamStationRecovery
__all__ += ["GeneralizedBeamElementRecovery", "GeneralizedBeamStationRecovery"]

from ._stiffened_shell_arc_control import ArcLengthControl
from ._stiffened_shell_arc_metric import PhysicalChordMetric, NativeDirectorChordMetric
from ._stiffened_shell_arc_publication import ArcResult
from ._stiffened_shell_arc_trial import ArcSelectionFailure, ArcAttempt
__all__ += ["ArcLengthControl", "PhysicalChordMetric", "NativeDirectorChordMetric", "ArcResult", "ArcSelectionFailure", "ArcAttempt"]

from ._stiffened_shell_ge_experimental import CURVED_Q4_POINT_REFERENCE_V1, CURVED_Q4_POINT_MATERIAL_REFERENCE_V1, CURVED_Q4_S3_POINT_REFERENCE_V1
__all__ += ["CURVED_Q4_POINT_REFERENCE_V1", "CURVED_Q4_POINT_MATERIAL_REFERENCE_V1", "CURVED_Q4_S3_POINT_REFERENCE_V1"]

from ._stiffened_shell_stress_free_reference import (
    ReferencePosePerturbation, StressFreePointReference, create_stress_free_point_reference,
)
__all__ += ["ReferencePosePerturbation", "StressFreePointReference", "create_stress_free_point_reference"]
__all__ += ["GEPointBucklingRequest", "GEPointBucklingResult"]
__all__ += ["GELineBucklingRequest", "GELineBucklingResult"]
__all__ += ["GEMixedBucklingRequest", "GEMixedBucklingResult"]
__all__ += ["GENativeLineBucklingRequest", "GENativeLineBucklingResult", "BL5_ELASTIC_NATIVE_FIBRE_V1", "BL5NativeElasticDomainError"]
__all__ += ["BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1"]
__all__ += ["BL5_NATIVE_FIBRE_MATERIAL_V1"]
from ._stiffened_shell_ge_experimental import BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1
__all__ += ["BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1"]


def capabilities_record(analysis):
    """Return detached JSON capability metadata for an existing GE or B2 owner.

    The owner's original report and typed request admission remain authoritative.
    This record neither executes requests nor establishes qualification.
    """
    from ._stiffened_shell_consumer_capabilities import capabilities_record as record
    return record(analysis)


__all__ += ["capabilities_record"]
