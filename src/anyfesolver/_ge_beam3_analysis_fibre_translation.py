"""Model-owned physical-fibre continuation, using its original retained owner.

Only immutable definitions cross the static/retained adapter boundary. No
accepted static state is converted to a retained material-history checkpoint.
"""
from copy import deepcopy

from .fe_core import FEModel
from ._ge_beam3_native_definition import NativeBeamDefinition
from ._ge_beam3_native_analysis import NativeBeamRun, NativeBeamWorkflowError, _require
from ._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
from ._ge_beam3_p5_seeded.core import canonical
from . import _ge_beam3_retained_fibre_control as owner
from ._ge_beam3_analysis_translation import _envelope, _backend, _check_backend_header

WORKFLOW = 'PHYSICAL_FIBRE_RETAINED_FROM_REFERENCE'


def _program(analysis, program):
    analysis._family_required('PHYSICAL_FIBRE_NODAL')
    if type(program) is not owner.TranslationProgram:
        raise NativeBeamWorkflowError('exact physical-fibre translation program required')
    program.__post_init__()
    return program.descriptor()


def _model(analysis):
    """Reconstruct a fresh driver container; preserve the exact physical operator."""
    analysis._guard()
    _require(bool(analysis._admit_boundaries()), 'supported fibre translation model required')
    _require(analysis.model.mesh.dof_manager.total_dofs+24*len(analysis._elements) <= 256,
             'physical-fibre translation exceeds the retained driver coordinate bound')
    made = FEModel('model-owned-retained-physical-fibre')
    for node, value in sorted(analysis.model.mesh.nodes.items()):
        made.add_node(node, *value.coords())
    for raw, original in zip(analysis._definitions, analysis._elements):
        definition, _ = NativeBeamDefinition(raw).instantiate()
        element = NativeRetainedFibreElement(definition.element_id, tuple(definition.node_ids),
            definition.operator.reference, definition.section, order=definition.operator.order)
        _require(element.operator.identity == original.operator.identity == definition.operator.identity,
                 'fibre driver reconstruction changed physical operator')
        made.add_element(element.element_id, element)
        made.materials[element.material_name] = element.section
    for boundary in analysis.model.boundary_conditions:
        made.add_boundary_condition(deepcopy(boundary))
    analysis._guard()
    return made


def _decode(analysis, data, raw, expected):
    backend, digest = _backend(analysis, data, raw, expected, WORKFLOW, None)
    _check_backend_header(backend, digest, data, owner, None)
    return backend, digest


def solve(analysis, program, *, checkpoint=None, expected_sha256=None, stop_after=None,
          cancellation_token=None, progress=None):
    data = _program(analysis, program)
    _require((checkpoint is None) == (expected_sha256 is None), 'checkpoint/hash pair required')
    end = len(program.targets) if stop_after is None else stop_after
    _require(type(end) is int and 0 <= end <= len(program.targets), 'bounded fibre translation stop target')
    _require(progress is None or callable(progress), 'callable fibre translation observer required')
    with analysis._operation():
        backend = digest = None
        if checkpoint is not None:
            backend, digest = _decode(analysis, data, checkpoint, expected_sha256)
        model = _model(analysis)
        def observed(row):
            analysis._guard()
            if progress is not None:
                progress(row)
            analysis._guard()
        result = owner.solve_translation_program(model, program, checkpoint=backend,
            expected_checkpoint_sha256=digest, stop_after=stop_after,
            cancellation_token=cancellation_token, progress=observed)
        return NativeBeamRun(result.status, _envelope(analysis, data, result.checkpoint, WORKFLOW, None), result)


def adopt(analysis, program, backend, *, expected_sha256):
    data = _program(analysis, program)
    _check_backend_header(backend, expected_sha256, data, owner, None)
    with analysis._operation():
        context = owner.Context(_model(analysis), program)
        _, records = context.restore(backend, expected_sha256=expected_sha256)
        _require(context.checkpoint(records) == backend, 'fibre adoption changed accepted checkpoint')
        return _envelope(analysis, data, backend, WORKFLOW, None)


def recover(analysis, program, checkpoint, *, expected_sha256):
    data = _program(analysis, program)
    with analysis._operation():
        backend, digest = _decode(analysis, data, checkpoint, expected_sha256)
        context = owner.Context(_model(analysis), program)
        state, records = context.restore(backend, expected_sha256=digest)
        before = canonical(state)
        result = context.recover(state)
        _require(canonical(state) == before and context.checkpoint(records) == backend,
                 'fibre recovery changed accepted material history')
        return result


def prefix(analysis, program, checkpoint, accepted_steps, *, expected_sha256):
    data = _program(analysis, program)
    _require(type(accepted_steps) is int and accepted_steps >= 0, 'exact nonnegative fibre prefix cursor')
    with analysis._operation():
        backend, digest = _decode(analysis, data, checkpoint, expected_sha256)
        context = owner.Context(_model(analysis), program)
        _, records = context.restore(backend, expected_sha256=digest)
        _require(accepted_steps <= len(records), 'fibre prefix exceeds accepted chain')
        return _envelope(analysis, data, context.checkpoint(records[:accepted_steps]), WORKFLOW, None)


def modes(analysis, program, checkpoint, *, expected_sha256, bounds, num_modes=6,
          root_width=1e-10, relative_width=1e-12, cancellation_token=None):
    from ._ge_beam3_analysis_translation_modal import _controls, NativeTranslationModes
    from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes
    from ._ge_beam3_retained_fibre_modal import prepare
    from .control import cancellation_safe_point
    data = _program(analysis, program)
    _controls(analysis, bounds, num_modes, root_width, relative_width)
    with analysis._operation():
        cancellation_safe_point(cancellation_token, 'fibre-modal.model-capture')
        backend, digest = _decode(analysis, data, checkpoint, expected_sha256)
        packet, guard = prepare(_model(analysis), program, backend, analysis._inertias,
            expected_sha256=digest, cancellation_token=cancellation_token)
        result = solve_paired_factor_chain_modes(packet.left, packet.right, packet.geometric, packet.kinetic,
            packet.free_dofs, packet.algebraic_dofs, bounds=bounds, num_modes=num_modes,
            root_width=root_width, relative_width=relative_width, cancellation_token=cancellation_token)
        guard()
        return NativeTranslationModes(analysis.identity, expected_sha256, packet, result)
