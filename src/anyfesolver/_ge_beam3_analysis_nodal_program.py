"""Owned generalized nodal dead-force programme; preserves its retained state.

Targets are the complete frozen absolute load-factor schedule. Continuation
resumes a prefix of that same schedule, never relabels a condensed force state.
"""
from dataclasses import dataclass, field
from hashlib import sha256

from . import _ge_beam3_retained_nodal_loading as owner
from ._ge_beam3_native_analysis import NativeBeamRun, _require
from ._ge_beam3_analysis_translation import _strict, _operation
from ._ge_beam3_analysis_translation_modal import _controls
from ._ge_beam3_p5_seeded.core import canonical
from .control import cancellation_safe_point

SCHEMA = 'GE_BEAM3_MODEL_OWNED_NODAL_PROGRAM_CHECKPOINT_V1'
KEYS = {'schema', 'definition_graph_sha256', 'program', 'backend',
        'backend_sha256', 'production_qualified'}


@dataclass(frozen=True)
class NativeNodalProgramModes:
    definition_graph_sha256: str
    checkpoint_sha256: str
    packet: object
    modes: object
    production_qualified: bool = field(default=False, init=False)


def _program(analysis, program):
    analysis._family_required('GENERALIZED_DISTRIBUTED')
    _require(type(program) is owner.Program, 'exact retained nodal programme required')
    program.__post_init__()
    program.nodal_forces.require(analysis.model.mesh)
    return canonical(program)


def _envelope(analysis, program, backend):
    analysis._guard()
    raw = canonical(dict(schema=SCHEMA, definition_graph_sha256=analysis.identity,
        program=program, backend=backend.decode('ascii'),
        backend_sha256=sha256(backend).hexdigest(), production_qualified=False))
    _strict(raw)
    return raw


def _header(backend, digest, program):
    _require(type(backend) is bytes and type(digest) is str
        and sha256(backend).hexdigest() == digest, 'external nodal backend hash mismatch')
    data = _strict(backend)
    _require(type(data) is dict and data.get('schema') == owner.SCHEMA
        and canonical(data.get('program')) == canonical(program), 'nodal backend owner/program mismatch')


def _backend(analysis, program, raw, digest):
    _require(type(raw) is bytes and type(digest) is str
        and sha256(raw).hexdigest() == digest, 'external nodal checkpoint hash mismatch')
    data = _strict(raw)
    _require(type(data) is dict and set(data) == KEYS, 'complete nodal envelope required')
    _require(data['schema'] == SCHEMA and data['definition_graph_sha256'] == analysis.identity
        and canonical(data['program']) == canonical(program)
        and data['production_qualified'] is False and type(data['backend']) is str,
        'nodal definition/program/owner mismatch')
    backend = data['backend'].encode('ascii')
    _header(backend, data['backend_sha256'], program)
    return backend, data['backend_sha256']


def solve(analysis, program, *, checkpoint=None, expected_sha256=None, stop_after=None,
          cancellation_token=None, progress=None):
    cancellation_safe_point(cancellation_token, 'owned-nodal-program.start')
    frozen = _program(analysis, program)
    _require((checkpoint is None) == (expected_sha256 is None), 'checkpoint/hash pair required')
    _require(progress is None or callable(progress), 'callable nodal observer required')
    end = len(program.targets) if stop_after is None else stop_after
    _require(type(end) is int and 0 <= end <= len(program.targets), 'bounded nodal stop target')
    with _operation(analysis, cancellation_token):
        backend = digest = None
        if checkpoint is not None:
            backend, digest = _backend(analysis, program, checkpoint, expected_sha256)
        def observed(row):
            analysis._guard()
            if progress is not None:
                progress(row)
            analysis._guard()
            _require(canonical(program) == frozen, 'nodal programme changed')
        result = owner.solve(analysis.model, program, checkpoint=backend, expected_sha256=digest,
            stop_after=stop_after, cancellation_token=cancellation_token, progress=observed)
        _require(canonical(program) == frozen, 'nodal programme changed')
        # A cancelled/failed owner returns only its actual last accepted capsule.
        # Preserve that capsule for resumption; never issue an unaccepted state.
        return NativeBeamRun(result.status, _envelope(analysis, program, result.checkpoint), result)


def adopt(analysis, program, backend, *, expected_sha256):
    _program(analysis, program)
    _header(backend, expected_sha256, program)
    with _operation(analysis):
        context = owner.Context(analysis.model, program)
        state, records = context.restore(backend, expected_sha256=expected_sha256)
        _require(context.checkpoint(records) == backend, 'nodal adoption changed accepted chain')
        return _envelope(analysis, program, backend)


def recover(analysis, program, checkpoint, *, expected_sha256):
    _program(analysis, program)
    with _operation(analysis):
        backend, digest = _backend(analysis, program, checkpoint, expected_sha256)
        context = owner.Context(analysis.model, program)
        state, records = context.restore(backend, expected_sha256=digest)
        before = canonical(state)
        result = context.recover(state)
        _require(canonical(state) == before and context.checkpoint(records) == backend,
            'nodal recovery changed accepted state')
        return result


def prefix(analysis, program, checkpoint, accepted_steps, *, expected_sha256):
    _program(analysis, program)
    _require(type(accepted_steps) is int and accepted_steps >= 0, 'exact nonnegative nodal cursor')
    with _operation(analysis):
        backend, digest = _backend(analysis, program, checkpoint, expected_sha256)
        context = owner.Context(analysis.model, program)
        state, records = context.restore(backend, expected_sha256=digest)
        _require(accepted_steps <= len(records), 'nodal prefix exceeds accepted history')
        return _envelope(analysis, program, context.checkpoint(records[:accepted_steps]))


def modes(analysis, program, checkpoint, *, expected_sha256, bounds, num_modes=6,
          root_width=1e-10, relative_width=1e-12, cancellation_token=None):
    cancellation_safe_point(cancellation_token, 'owned-nodal-modes.start')
    _program(analysis, program)
    _controls(analysis, bounds, num_modes, root_width, relative_width)
    with _operation(analysis, cancellation_token):
        backend, digest = _backend(analysis, program, checkpoint, expected_sha256)
        packet, result = owner.solve_modes(analysis.model, program, backend, analysis._inertias,
            expected_sha256=digest, bounds=bounds, num_modes=num_modes,
            root_width=root_width, relative_width=relative_width, cancellation_token=cancellation_token)
        cancellation_safe_point(cancellation_token, 'owned-nodal-modes.complete')
        return NativeNodalProgramModes(analysis.identity, expected_sha256, packet, result)
