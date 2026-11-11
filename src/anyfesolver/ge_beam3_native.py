"""Explicit native GE-B3 workflow interface; never a topology/default alias.

Artifact acceptance is separate from this immutable interface policy. Historical
implementation IDs and ``production_qualified=False`` checkpoint fields are
preserved; importing this module does not declare qualification. See the
GE_BEAM3_NATIVE_WORKFLOWS documentation and the external artifact acceptance.
"""
from hashlib import sha256

from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as ReferenceGeometry
from ._ge_beam3_fibre_section import Fibre, FlowCurve, PhysicalFibreSection
from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from ._ge_beam3_native_definition import NativeBeamDefinition as BeamDefinition
from ._ge_beam3_native_analysis import NativeBeamAnalysis as _Analysis
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement as _Generalized
from ._ge_beam3_native_fibre_static_element import NativeFibreStaticElement as _Fibre
from ._ge_beam3_durable_coupled import DurableCoupledBeamAnalysis as _Coupled
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_retained_generalized_state import Program as DistributedProgram
from ._ge_beam3_retained_nodal_loading import NodalDeadForces, Program as NodalProgram
from ._ge_beam3_retained_translation_control import Program as TranslationProgram
from ._ge_beam3_retained_fibre_control import TranslationProgram as FibreTranslationProgram
from ._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from ._ge_beam3_g6_domain import (
    ActivityEpochLease, ConsumerPolicy, GeneralizedInitialField, evaluate_initial_field,
)
from ._ge_beam3_pose_joint import RigidPoseJoint as ObjectivePoseJoint
from ._ge_beam3_p5_seeded.core import canonical as _canonical

SELECTOR = 'ge-beam3-native'
PROFILE_ID = 'GE_BEAM3_NATIVE_OWNED_WORKFLOWS_V1'
PROFILE_BYTES = _canonical(dict(
    schema=PROFILE_ID, selector=SELECTOR,
    acceptance='REQUIRES_SEPARATE_EXACT_ARTIFACT_ACCEPTANCE',
    geometry='REGULAR_STRAIGHT_OR_PIECEWISE_Q2_EXPLICIT_NODAL_TRIADS',
    section_families=('RESULTANT_ELLIPSOID', 'PHYSICAL_AXIAL_BIAXIAL_FIBRE'),
    standalone='EXACT_NATIVE_DEFINITION_AND_MODEL_OWNED_WORKFLOWS',
    coupled='DURABLE_SINGLE_ELASTIC_Q4_OR_S3_V2D_GENERALIZED_BEAM_V2_VARIATIONAL',
    dynamics='CONSERVATIVE_CURRENT_REST_PHYSICAL_RETAINED_INERTIA_ONLY',
    buckling='FROZEN_CURRENT_CONSERVATIVE_PREDICTIONS_NOT_LIMIT_LOADS',
    exclusions=('ARBITRARY_SECTION_LAWS', 'FINITE_VELOCITY_DYNAMICS',
                'GENERIC_MIXED_FEMODEL', 'PLASTIC_SHELL_COUPLING',
                'PHYSICAL_FIBRE_SHELL_COUPLING', 'STABLE_POSTBUCKLING_CLAIM',
                'DEFAULT_OR_LEGACY_ALIAS_REPLACEMENT'),
    historical_checkpoint_fields='UNCHANGED_IMPLEMENTATION_CONTRACT_NOT_RELEASE_STATUS',
))
PROFILE_SHA256 = sha256(PROFILE_BYTES).hexdigest()


def _select(formulation):
    if type(formulation) is not str or formulation != SELECTOR:
        raise ValueError('explicit ge-beam3-native workflow required; no alias or legacy fallback')


def define_beam(formulation, element_id, node_ids, coordinates, nodal_triads,
                section, section_inertia, *, quadrature=4, arithmetic_policy=None):
    """Capture a reconstructible definition, not an accepted material state.

Three nodes are [end, midpoint-parameter, end]. All physical nodal triads and
the positive physical 6x6 section inertia must be supplied. Admitted quadrature
and optional precise arithmetic retain the exact existing definition policies.
"""
    _select(formulation)
    if type(section) not in (EllipsoidalGeneralizedSection, PhysicalFibreSection):
        raise ValueError('exact associated resultant or physical axial/biaxial fibre section required')
    if type(quadrature) is not int or quadrature not in (4, 8):
        raise ValueError('registered four- or eight-point native quadrature required')
    if type(section) is PhysicalFibreSection and arithmetic_policy is not None:
        raise ValueError('generalized precise arithmetic is not a physical-fibre policy')
    reference = ReferenceGeometry(coordinates, nodal_triads)
    if type(section) is EllipsoidalGeneralizedSection:
        element = _Generalized(element_id, node_ids, reference, section,
                               order=quadrature, arithmetic_policy=arithmetic_policy)
    else:
        element = _Fibre(element_id, node_ids, reference, section, order=quadrature)
    return BeamDefinition.capture(element, section_inertia)


def create_analysis(formulation, definitions, boundaries, *, retained_refinement=False):
    """Return the exact reviewed standalone owner, with all admission guards.

Use its solve/recover/reference_modes/current_modes/translation/buckling
methods. Each workflow has its own unchanged checkpoint and material scope.
No subclass, dynamic dispatch, or generic FEModel conversion is introduced.
"""
    _select(formulation)
    return _Analysis(definitions, boundaries, retained_refinement=retained_refinement)


def create_coupled_analysis(formulation, definitions, boundaries, program, *,
                            shell, targets, shell_fixed, nodal_forces,
                            shell_density=None, max_iterations=24, max_backtracks=8):
    """Single elastic Q4/S3 V2D plus generalized beam, exact V2 pose work map.

Every operation constructs a fresh bounded owner and mechanically replays the
authenticated checkpoint. No inner deadline or state identity is renewed.
"""
    _select(formulation)
    return _Coupled(definitions, boundaries, program, shell=shell, targets=targets,
                    shell_fixed=shell_fixed, nodal_forces=nodal_forces,
                    shell_density=shell_density, max_iterations=max_iterations,
                    max_backtracks=max_backtracks)


def workflow_provenance(analysis):
    """Describe the owned definition and delivery policy, not state acceptance.

Keep this alongside the actual run's unchanged checkpoint and digest. This
record does not validate a checkpoint or replace mechanical restart replay.
"""
    if type(analysis) not in (_Analysis, _Coupled):
        raise ValueError('exact native standalone or durable coupled owner required')
    analysis._guard()
    return _canonical(dict(profile_id=PROFILE_ID, profile_sha256=PROFILE_SHA256,
        definition_sha256=analysis.identity,
        workflow='STANDALONE' if type(analysis) is _Analysis else 'COUPLED_V2',
        artifact_acceptance='EXTERNAL_RECORD_REQUIRED', checkpoint_validated=False))


__all__ = ['SELECTOR', 'PROFILE_ID', 'PROFILE_BYTES', 'PROFILE_SHA256',
           'ReferenceGeometry', 'Fibre', 'FlowCurve', 'PhysicalFibreSection',
           'EllipsoidalGeneralizedSection', 'BeamDefinition', 'LinePattern',
           'DistributedPattern', 'DistributedProgram', 'NodalDeadForces',
           'NodalProgram', 'TranslationProgram', 'FibreTranslationProgram',
           'SpatialNodalMoments', 'define_beam', 'create_analysis',
           'create_coupled_analysis', 'workflow_provenance',
           'ActivityEpochLease', 'GeneralizedInitialField',
           'ConsumerPolicy', 'ObjectivePoseJoint', 'evaluate_initial_field']
