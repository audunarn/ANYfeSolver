"""Model-owned combined force/couple integration; no state-owner conversion."""
import numpy as np

from .control import cancellation_safe_point
from ._ge_beam3_native_analysis import COMBINED_WORKFLOW, NativeBeamRun, _controls, _require
from ._ge_beam3_native_generalized_combined_couples import solve_combined_static
from ._ge_beam3_native_generalized_combined_restart import (
    LoadPoint, capture_moments, encode_checkpoint, decode_checkpoint,
)
from ._ge_beam3_native_generalized_restart import LoadPoint as DistributedPoint
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_native_force_adaptation import describe, require_capacity
from ._ge_beam3_p5_seeded.core import canonical


def _decode(analysis, checkpoint, expected):
    raw, digest = analysis._backend(checkpoint, expected, workflow=COMBINED_WORKFLOW)
    return raw, digest, decode_checkpoint(analysis.model, raw, expected_sha256=digest)


def solve(analysis, nodal_moments, *, distributed_pattern=None, steps=2, max_iterations=24,
          line_search=True, step_policy=None, checkpoint=None, expected_sha256=None,
          cancellation_token=None, progress_callback=None):
    """Continue with the specified load increments from the last accepted point.

    Unlike the fibre fixed-force-pattern target-factor API, these patterns
    are increments. The existing native protocol owns every effective load,
    origin, rotation update and accepted predecessor in the complete chain.
    """
    cancellation_safe_point(cancellation_token, 'owned-combined.capture')
    _controls(steps, max_iterations)
    _require(type(line_search) is bool, 'explicit combined line-search flag required')
    _require(progress_callback is None or callable(progress_callback), 'callable combined observer required')
    controls = describe(step_policy, steps, max_iterations, line_search)
    analysis._family_required('GENERALIZED_DISTRIBUTED')
    _require((checkpoint is None) == (expected_sha256 is None), 'combined checkpoint/hash pair required')
    empty = DistributedPattern(LinePattern(()), ())
    pattern = empty if distributed_pattern is None else distributed_pattern
    _require(type(pattern) is DistributedPattern, 'exact combined distributed pattern required')
    with analysis._operation():
        _require(bool(analysis._admit_boundaries()), 'supported combined force model required')
        moments = capture_moments(nodal_moments, analysis.model)
        pattern.require(analysis.model.mesh)
        patterns_before = canonical((moments, pattern))
        raw = digest = None
        constant_distributed, constant_moments = empty, None
        if checkpoint is None:
            # Capacity and all caller controls are checked before mechanics.
            require_capacity(step_policy, steps, 0)
            chain = (dict(load_point=LoadPoint(DistributedPoint(0., empty, pattern), None, moments),
                displacements=np.zeros(analysis.model.mesh.dof_manager.total_dofs),
                states=analysis._initial(cancellation_token)),)
        else:
            raw, digest, chain = _decode(analysis, checkpoint, expected_sha256)
            accepted = chain[-1]['load_point']
            constant_distributed = accepted.distributed.effective(analysis.model)
            constant_moments = accepted.effective_moments(analysis.model)
            require_capacity(step_policy, steps, len(chain)-1)
            if step_policy is None:
                _require(len(chain)-1+steps <= 64, 'combined continuation exceeds complete-chain capacity')
        cancellation_safe_point(cancellation_token, 'owned-combined.captured')
        result, _ = solve_combined_static(analysis.model, moments, distributed_pattern=pattern,
            steps=steps, max_iterations=max_iterations, line_search=line_search, step_policy=step_policy,
            initial_checkpoint=raw, expected_sha256=digest,
            cancellation_token=cancellation_token, progress_callback=progress_callback)
        chain += tuple(dict(load_point=LoadPoint(DistributedPoint(float(s.load_factor),
            constant_distributed, pattern), constant_moments, moments),
            displacements=s.displacements, states=s.element_states) for s in result.snapshots)
        _require(describe(step_policy, steps, max_iterations, line_search) == controls
                 and canonical((capture_moments(nodal_moments, analysis.model), pattern)) == patterns_before,
                 'combined caller controls or patterns changed')
        cancellation_safe_point(cancellation_token, 'owned-combined.checkpoint')
        envelope = analysis._envelope(encode_checkpoint(analysis.model, chain), workflow=COMBINED_WORKFLOW)
        cancellation_safe_point(cancellation_token, 'owned-combined.complete')
        return NativeBeamRun(result.status, envelope, result)


def recover(analysis, checkpoint, *, expected_sha256):
    from ._ge_beam3_native_generalized_recovery import recover_native_fields
    analysis._family_required('GENERALIZED_DISTRIBUTED')
    with analysis._operation():
        _, _, chain = _decode(analysis, checkpoint, expected_sha256)
        state = chain[-1]
        before = canonical(state)
        result = {}
        for element in analysis._elements:
            mapping = list(element.get_dof_mapping(analysis.model.mesh))
            result[element.element_id] = recover_native_fields(element, analysis.model.mesh,
                state['states'][element.element_id], expected_committed_total_u=state['displacements'][mapping])
        _require(canonical(state) == before, 'combined recovery changed accepted history')
        return result


def prefix(analysis, checkpoint, accepted_steps, *, expected_sha256):
    _require(type(accepted_steps) is int and accepted_steps >= 0, 'explicit combined prefix step count required')
    analysis._family_required('GENERALIZED_DISTRIBUTED')
    with analysis._operation():
        _, _, chain = _decode(analysis, checkpoint, expected_sha256)
        _require(accepted_steps < len(chain), 'combined prefix exceeds accepted chain')
        return analysis._envelope(encode_checkpoint(analysis.model, chain[:accepted_steps+1]),
                                  workflow=COMBINED_WORKFLOW)
