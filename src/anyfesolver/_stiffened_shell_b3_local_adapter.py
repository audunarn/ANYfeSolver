"""Private native-B3 42-coordinate sparse trial extraction for G0 F08.

This adapter consumes the existing generalized or physical-fibre owner layout
without invoking its global dense assembler.  It is an extraction seam, not a
production mixed owner or a substitute for independent material oracles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_generalized_static_boundary import (
    cell_couple_load, spatial_jacobian,
)
from ._ge_beam3_operator_validation_scope import operator_call


@dataclass(frozen=True)
class NativeB3LocalTrial:
    element_id: object
    slots: np.ndarray
    residual: np.ndarray
    tangent: np.ndarray
    candidate_history: object
    family: str
    production_qualified: bool = False


@dataclass(frozen=True)
class NativeB3SparseTrial:
    residual: np.ndarray
    tangent: sparse.csr_matrix
    candidate_histories: tuple[object, ...]
    family: str
    production_qualified: bool = False


def _validated_inputs(context, state, origin, parameter, family):
    """Apply the bound owner's model, state and history admission once."""
    if family not in ("generalized", "physical_fibre"):
        raise ValueError("native B3 family must be generalized or physical_fibre")
    if not np.isfinite(parameter):
        raise ValueError("finite native B3 load parameter required")
    fibre = family == "physical_fibre"
    layout = context.layout if fibre else context
    context.guard()
    state = (layout.make(state.descriptor()) if fibre else
             context.make(state.descriptor()))
    origin = context.history(origin) if fibre else context.histories(origin)
    return layout, state, origin, fibre


def _evaluate_local_validated(context, layout, state, origin, parameter,
                              family, fibre, element_index):
    if type(element_index) is not int or not 0 <= element_index < len(layout.elements):
        raise ValueError("valid native B3 element index required")
    eid, element = layout.elements[element_index]
    nodes = layout.nodes[element_index]
    operator = element.operator
    response = (operator.evaluate(
        state.positions[nodes], state.position_low[nodes],
        state.nodal_frames[nodes], state.cell_rotations[element_index],
        state.resultants[element_index], origin=origin[element_index],
        check=context.check)
        if fibre else operator_call(
            operator, "evaluate", context.guard, context.started,
            state.positions[nodes], state.position_low[nodes],
            state.nodal_frames[nodes], state.cell_rotations[element_index],
            state.resultants[element_index], origin=origin[element_index]))
    if fibre:
        local_r = response.residual
        # The fibre operator returns the symmetric left-Exp energy Hessian.
        # The mixed sparse seam consumes a spatial residual Jacobian, just
        # like the generalized branch below.
        local_k = spatial_jacobian(
            local_r, response.hessian+response.hessian_low)
    else:
        work = line_work(
            operator.reference, state.positions[nodes],
            state.position_low[nodes], state.cell_rotations[element_index],
            parameter*context.program.pattern.force(eid),
            order=operator.order)
        corrected = response.residual-work.gradient
        local_r = corrected-cell_couple_load(
            operator, parameter*context.program.pattern.density(eid))
        local_k = spatial_jacobian(
            corrected, response.hessian+response.hessian_low-
            work.hessian)
    slots = np.asarray(layout.slots[element_index], dtype=int)
    if (slots.shape != (42,) or local_r.shape != (42,) or
            local_k.shape != (42, 42) or
            not np.isfinite(local_r).all() or
            not np.isfinite(local_k).all()):
        raise ValueError("finite 42-coordinate native B3 local response required")
    return NativeB3LocalTrial(eid, slots.copy(), np.array(local_r, copy=True),
                              np.array(local_k, copy=True), response.history,
                              family)


def evaluate_local_trial(context, state, origin, parameter: float, *,
                         family: str, element_index: int) -> NativeB3LocalTrial:
    """Return one owned 42-coordinate response from an accepted predecessor."""
    layout, state, origin, fibre = _validated_inputs(
        context, state, origin, parameter, family)
    local = _evaluate_local_validated(
        context, layout, state, origin, parameter, family, fibre, element_index)
    context.guard()
    return local


def evaluate_sparse_trial(context, state, origin, parameter: float, *,
                          family: str) -> NativeB3SparseTrial:
    """Scatter bounded local trials without invoking a global dense owner."""
    layout, state, origin, fibre = _validated_inputs(
        context, state, origin, parameter, family)
    residual = np.zeros(layout.count)
    row, column, data = [], [], []
    candidates = []
    for i in range(len(layout.elements)):
        local = _evaluate_local_validated(
            context, layout, state, origin, parameter, family, fibre, i)
        np.add.at(residual, local.slots, local.residual)
        row.extend(np.repeat(local.slots, 42))
        column.extend(np.tile(local.slots, 42))
        data.extend(local.tangent.ravel())
        candidates.append(local.candidate_history)
    if fibre:
        residual[:layout.nodal_count] -= parameter*layout.force
    if not np.isfinite(residual).all():
        raise ValueError("finite native B3 assembled residual required")
    tangent = sparse.coo_matrix(
        (data, (row, column)), shape=(layout.count, layout.count)).tocsr()
    tangent.eliminate_zeros()
    if not np.isfinite(tangent.data).all():
        raise ValueError("finite native B3 assembled tangent required")
    context.guard()
    return NativeB3SparseTrial(residual, tangent, tuple(candidates), family)
