"""Opt-in connected centered elastic native dynamics DEVELOPMENT API.

Source candidate: no qualification, contact or generic FEModel admission.
"""
from ._ge_beam3_connected_dynamic import (ConnectedDynamicAnalysis, ConnectedState,
    ConnectedCandidate, PROFILE, SCHEMA)

SELECTOR='ge_beam3_connected_centered_elastic_dynamic_dev'


def create_analysis(formulation,definitions,boundaries, *, inertia_order=24,
                    max_iterations=12,max_backtracks=8):
    if formulation!=SELECTOR:raise ValueError('explicit connected dynamic selector required')
    return ConnectedDynamicAnalysis(definitions,boundaries,inertia_order=inertia_order,
        max_iterations=max_iterations,max_backtracks=max_backtracks)


def solve_transient(analysis,config, *, checkpoint=None,expected_sha256=None,
                    max_steps=10000,max_cutbacks=8,minimum_dt=None,cancellation_token=None):
    """Shared native midpoint controller; physical3N+6E reference initial data."""
    from ._ge_beam3_connected_trajectory import solve_connected_transient
    return solve_connected_transient(analysis,config,checkpoint=checkpoint,
        expected_sha256=expected_sha256,max_steps=max_steps,max_cutbacks=max_cutbacks,
        minimum_dt=minimum_dt,cancellation_token=cancellation_token)
