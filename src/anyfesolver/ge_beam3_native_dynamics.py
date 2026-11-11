"""Opt-in contact-free finite-velocity native beam DEVELOPMENT interface.

No C7 acceptance, released finite-speed contract or contact admission is claimed.
The original :mod:`ge_beam3_native` selector and static/rest guards are unchanged.
Definitions may be built with its existing ``define_beam``; activation requires
this separate selector. Issued objects cannot be transferred between analyses.
"""
from hashlib import sha256
from ._ge_beam3_p5_seeded.core import canonical
from ._ge_beam3_native_dynamic_analysis import NativeDynamicAnalysis
from ._ge_beam3_native_midpoint import (
    MidpointState, MidpointCandidate, CandidateStageAudit,
    EndpointAccounting, PhysicalLoadAccounting)

SELECTOR = 'ge-beam3-native-contact-free-midpoint-dev'
PROFILE_ID = 'GE_BEAM3_NATIVE_CONTACT_FREE_FINITE_VELOCITY_DEV_V2'
PROFILE_BYTES = canonical(dict(profile_id=PROFILE_ID, status='DEVELOPMENT',
    scope='ONE_CENTERED_ELASTIC_ELLIPSOID_ELEMENT_FREE_OR_HOMOGENEOUS_FIRST_NODE_CLAMP',
    mechanics='EXISTING_DYNAMIC_ELASTIC_CENTERED_FINITE_INERTIA_MIDPOINT',
    restart='FRESH_OWNER_COMMITTED_MIDPOINT_DEV_V1', contact='UNSUPPORTED', acceptance='EXTERNAL_FULL_C7_OPEN'))
PROFILE_SHA256 = sha256(PROFILE_BYTES).hexdigest()


def create_analysis(formulation, definitions, boundaries, *, inertia_order=24,
                    max_iterations=12, max_backtracks=8,
                    momentum_origin_high=(0.,0.,0.), momentum_origin_low=(0.,0.,0.),
                    load_pattern=None, stage_residual_tolerance=1e-11,
                    stage_coordinates='increment', runtime_limit_seconds=120.):
    """Create one explicit development owner; default solve tolerance unchanged.

    Boundary tuples may be empty or a homogeneous full clamp at the first node.
    Constant reference-measure physical loads use the existing signed pattern
    contract. Time/state/history/restart authority remains in one midpoint owner.
    Optional acceleration coordinates change the Newton chart only; the default
    increment route, physical equations and convergence gates remain unchanged.
    A positive finite runtime limit up to900s is captured at construction; default
    120s is preserved. Extended owners exclude optional local validation scopes.
    This resource setting does not authorize or renew an evidence run.
    """
    if type(formulation) is not str or formulation != SELECTOR:
        raise ValueError('explicit native contact-free dynamics development selector required')
    return NativeDynamicAnalysis(definitions, boundaries, profile_sha256=PROFILE_SHA256,
        inertia_order=inertia_order, max_iterations=max_iterations, max_backtracks=max_backtracks,
        momentum_origin_high=momentum_origin_high, momentum_origin_low=momentum_origin_low,
        load_pattern=load_pattern, stage_residual_tolerance=stage_residual_tolerance,
        stage_coordinates=stage_coordinates, runtime_limit_seconds=runtime_limit_seconds)


def workflow_provenance(analysis):
    if type(analysis) is not NativeDynamicAnalysis:
        raise ValueError('exact native dynamic development analysis required')
    return dict(profile_id=PROFILE_ID, profile_sha256=PROFILE_SHA256,
                analysis_sha256=analysis.identity, status='DEVELOPMENT',
                production_qualified=False, restart_supported=True,
                acceptance='EXTERNAL_FULL_C7_OPEN')


def solve_transient(analysis, config, *, checkpoint=None, expected_sha256=None,
                    max_steps=10000, max_cutbacks=8, minimum_dt=None,
                    cancellation_token=None):
    """Opt-in contact-free native midpoint trajectory DEVELOPMENT workflow.

    Uses existing TransientConfig time/physical15 initial/output controls;
    Newmark/HHT/damping/recovery are unsupported. No C7 acceptance is implied.
    """
    from ._ge_beam3_native_transient import solve_native_transient
    return solve_native_transient(analysis, config, checkpoint=checkpoint,
        expected_sha256=expected_sha256, max_steps=max_steps,
        max_cutbacks=max_cutbacks, minimum_dt=minimum_dt,
        cancellation_token=cancellation_token)
