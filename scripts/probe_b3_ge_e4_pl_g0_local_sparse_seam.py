"""G0 F08-S bounded local native-beam sparse-gather diagnostic, not product code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from anyfesolver._ge_beam3_fibre_line_work import evaluate as line_work
from anyfesolver._ge_beam3_generalized_static_boundary import cell_couple_load, spatial_jacobian
from anyfesolver._ge_beam3_operator_validation_scope import operator_call
from anyfesolver._ge_beam3_retained_fibre_state import Context as FibreContext
from anyfesolver._ge_beam3_retained_generalized_state import Context as GeneralizedContext, Program
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from test_ge_beam3_native_generalized_restart import make as generalized_model, pattern
from test_ge_beam3_retained_fibre import make_model as fibre_model


def gather(count, slots, local_residuals, local_tangents):
    row = []
    col = []
    data = []
    residual = np.zeros(count)
    for ids, local_r, local_k in zip(slots, local_residuals, local_tangents, strict=True):
        ids = np.asarray(ids, dtype=int)
        assert ids.ndim == 1 and len(ids) == len(local_r) == local_k.shape[0] == local_k.shape[1]
        assert len(ids) == len(set(ids)) and np.all((0 <= ids) & (ids < count))
        np.add.at(residual, ids, local_r)
        row.extend(np.repeat(ids, len(ids)))
        col.extend(np.tile(ids, len(ids)))
        data.extend(local_k.ravel())
    tangent = sparse.coo_matrix((data, (row, col)), shape=(count, count)).tocsr()
    tangent.eliminate_zeros()
    return residual, tangent


def check(kind, context, residual, dense, metrics, responses, local_r, local_k, parameter, slots, external):
    sparse_r, tangent = gather(context.count if kind == "generalized" else context.layout.count,
                               slots, local_r, local_k)
    sparse_r[:len(external)] -= external
    difference_r = sparse_r - residual
    difference_k = tangent.toarray() - dense
    absolute_r = float(np.max(np.abs(difference_r)))
    absolute_k = float(np.max(np.abs(difference_k)))
    relative_r = float(np.linalg.norm(difference_r) / max(1., np.linalg.norm(residual)))
    relative_k = float(np.linalg.norm(difference_k) / max(1., np.linalg.norm(dense)))
    all_data = (sparse_r, tangent.data, residual, dense)
    assert all(np.isfinite(x).all() for x in all_data)
    assert max(absolute_r, absolute_k) <= 1e-9, (kind, absolute_r, absolute_k)
    assert max(relative_r, relative_k) <= 1e-11, (kind, relative_r, relative_k)
    assert len(local_r) == len(local_k) == len(slots) == len(responses) == 2
    assert all(len(ids) == 42 for ids in slots)
    assert len(set(slots[0]) & set(slots[1])) == 6
    return dict(family=kind, parameter=parameter, elements=2, coordinates=len(residual),
                retained_coordinates=48, nnz=int(tangent.nnz),
                residual_absolute=absolute_r, tangent_absolute=absolute_k,
                residual_normalized=relative_r, tangent_normalized=relative_k,
                equilibrium_compatibility=metrics, local_response_count=len(responses),
                shared_physical_coordinates=len(set(slots[0]) & set(slots[1])))


def generalized():
    model = generalized_model("connected-plastic")
    program = Program((.5, 1.), pattern(model))
    context = GeneralizedContext(model, program)
    state = context.initial.mechanical
    origins = context.initial.origins
    parameter = .5
    residual, dense, metrics, responses, works = context.assemble(state, parameter, origins)
    local_r = []
    local_k = []
    response_checks = []
    work_checks = []
    for i, (eid, element) in enumerate(context.elements):
        nodes = context.nodes[i]
        op = element.operator
        response = operator_call(op, "evaluate", context.guard, context.started,
            state.positions[nodes], state.position_low[nodes], state.nodal_frames[nodes],
            state.cell_rotations[i], state.resultants[i], origin=origins[i])
        work = line_work(op.reference, state.positions[nodes], state.position_low[nodes],
            state.cell_rotations[i], parameter * program.pattern.force(eid), order=op.order)
        rc = response.residual - work.gradient
        local_r.append(rc - cell_couple_load(op, parameter * program.pattern.density(eid)))
        local_k.append(spatial_jacobian(rc, response.hessian + response.hessian_low - work.hessian))
        response_checks.append(np.array_equal(response.residual, responses[i].residual))
        work_checks.append(work.value == works[i])
    assert all(response_checks + work_checks)
    result = check("generalized", context, residual, dense, metrics, responses,
                   local_r, local_k, parameter, context.slots, np.zeros(0))
    result["local_response_equal"] = all(response_checks)
    result["line_work_equal"] = all(work_checks)
    return result


def fibre():
    model = fibre_model()
    program = ForceProgram((.25, .5), ((5, .4, -.005, 0.),), max_iterations=24)
    context = FibreContext(model, program)
    layout = context.layout
    state = context.initial.mechanical
    origins = context.initial.origins
    parameter = .25
    residual, dense, metrics, responses = context.assemble(state, parameter, origins)
    local_r = []
    local_k = []
    response_checks = []
    for i, probe in enumerate(context.probes):
        nodes = layout.nodes[i]
        response = probe.evaluate(state.positions[nodes], state.position_low[nodes],
            state.nodal_frames[nodes], state.cell_rotations[i], state.resultants[i],
            origin=origins[i], check=context.check)
        local_r.append(response.residual)
        local_k.append(response.hessian + response.hessian_low)
        response_checks.append(np.array_equal(response.residual, responses[i].residual))
    assert all(response_checks)
    result = check("physical-fibre", context, residual, dense, metrics, responses,
                   local_r, local_k, parameter, layout.slots, parameter * layout.force)
    result["local_response_equal"] = all(response_checks)
    return result


if __name__ == "__main__":
    print(json.dumps(dict(profile="G0-F08-S-v1", status="diagnostic_pass",
                          results=[generalized(), fibre()]), sort_keys=True))
