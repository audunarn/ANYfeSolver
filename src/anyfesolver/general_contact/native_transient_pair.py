"""Additive public factory for common-stage native beam/beam contact DEVELOPMENT.

Exact selector only; not a generic FEModel admission or engineering acceptance.
"""
from anyfesolver._ge_beam3_native_pair_contact import (
    NativePairContactAnalysis, PROFILE, ENDPOINT_POLICY,
)
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from .native_retained_whole_pair import NativeRetainedWholeCurvePair

SELECTOR = 'ge_beam3_native_whole_round_pair_transient_dev'


def create_native_transient_pair_analysis(formulation, definitions, boundaries, *,
        radius_a_m, radius_b_m, endpoint_policy, line_stiffness_n_per_m2,
        distance_tolerance_m, feature_tolerance_m, search_parameter_tolerance,
        maximum_search_depth, maximum_search_cells, energy_tolerance_j,
        force_tolerance_n, tangent_tolerance_n_per_m, maximum_integration_intervals,
        maximum_penetration_m, maximum_swept_intervals, inertia_order=24,
        max_iterations=12, max_backtracks=8):
    """Build fresh exact native operators and complete physical two-cell pairing.

    All dimensional normal-law/search/error/event controls are explicit. Endpoint
    distance participates in symmetric line measure; no cap-area force is added.
    """
    if formulation != SELECTOR:
        raise ValueError('explicit native pair transient development selector required')
    if (type(definitions) is not tuple or len(definitions) != 2
            or any(type(d) is not NativeBeamDefinition for d in definitions)):
        raise ValueError('two exact immutable native beam definitions required')
    if type(boundaries) is not tuple or boundaries:
        raise ValueError('two free native bodies and empty boundary tuple required')
    a, _ = definitions[0].instantiate(); b, _ = definitions[1].instantiate()
    contact = NativeRetainedWholeCurvePair(
        a.element_id, a.operator.reference, a.node_ids, radius_a_m,
        b.element_id, b.operator.reference, b.node_ids, radius_b_m,
        line_stiffness_n_per_m2=line_stiffness_n_per_m2,
        distance_tolerance_m=distance_tolerance_m, feature_tolerance_m=feature_tolerance_m,
        search_parameter_tolerance=search_parameter_tolerance,
        maximum_search_depth=maximum_search_depth, maximum_search_cells=maximum_search_cells,
        energy_tolerance_j=energy_tolerance_j, force_tolerance_n=force_tolerance_n,
        tangent_tolerance_n_per_m=tangent_tolerance_n_per_m,
        maximum_integration_intervals=maximum_integration_intervals,
        certify_curved_ownership=True, pooled_quadrature_budget=True)
    return NativePairContactAnalysis(definitions, boundaries, contact,
        endpoint_policy=endpoint_policy, maximum_penetration_m=maximum_penetration_m,
        maximum_swept_intervals=maximum_swept_intervals, inertia_order=inertia_order,
        max_iterations=max_iterations, max_backtracks=max_backtracks)


def solve_native_transient_pair(analysis, config, *, checkpoint=None,
        expected_sha256=None, max_steps=10000, max_cutbacks=8, minimum_dt=None,
        cancellation_token=None):
    """One TransientConfig controller and one uncondensed retained midpoint solve."""
    if type(analysis) is not NativePairContactAnalysis:
        raise TypeError('exact native pair transient development owner required')
    from anyfesolver._ge_beam3_connected_trajectory import _solve_retained_transient
    return _solve_retained_transient(analysis, config, checkpoint=checkpoint,
        expected_sha256=expected_sha256, max_steps=max_steps,
        max_cutbacks=max_cutbacks, minimum_dt=minimum_dt,
        cancellation_token=cancellation_token)
