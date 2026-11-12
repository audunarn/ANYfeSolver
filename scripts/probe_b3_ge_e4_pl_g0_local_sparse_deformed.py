"""G0 F08-D: local CSR parity at a genuine accepted, deformed predecessor."""

from __future__ import annotations

import json

import numpy as np

import probe_b3_ge_e4_pl_g0_local_sparse_seam as seam
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_retained_fibre_program import solve_force_program
from anyfesolver._ge_beam3_retained_generalized_program import solve


def case(kind):
    if kind == "generalized":
        model = seam.generalized_model("connected-plastic")
        program = seam.Program((.5, 1.), seam.pattern(model))
        completed = solve(model, program, stop_after=1)
        context = seam.GeneralizedContext(model, program)
        layout = context
        target = .5
    else:
        model = seam.fibre_model()
        program = seam.ForceProgram((.25, .5), ((5, .4, -.005, 0.),), max_iterations=24)
        completed = solve_force_program(model, program, stop_after=1)
        context = seam.FibreContext(model, program)
        layout = context.layout
        target = .25
    if completed.status != "paused" or completed.completed_targets != 1:
        raise RuntimeError(f"{kind}: actual accepted target unavailable: {completed.status}: {completed.failure}")
    accepted = completed.state
    assert accepted.model_sha256 == context.identity
    origin_bytes = canonical((accepted, accepted.histories))
    direction = np.zeros(layout.count)
    direction[layout.free] = .0002 * np.sin(np.array(layout.free) + .3)
    trial = layout.advance(accepted.mechanical, direction)
    origins = accepted.histories
    assert float(np.linalg.norm(trial.nodal_frames - layout.reference_frames)) > 1e-7
    assert float(np.linalg.norm(trial.cell_rotations - np.eye(3))) > 1e-7
    assert float(np.linalg.norm(trial.resultants)) > 1e-7

    if kind == "generalized":
        residual, dense, metrics, responses, work_values = context.assemble(trial, target, origins)
        external = np.zeros(0)
        operators = [element.operator for _, element in context.elements]
    else:
        residual, dense, metrics, responses = context.assemble(trial, target, origins)
        external = target * layout.force
        operators = context.probes
    assert metrics[1] > 1e-7, (kind, metrics)
    local_r = []
    local_k = []
    histories = []
    materials = []
    works = []
    for i, operator in enumerate(operators):
        nodes = layout.nodes[i]
        args = (trial.positions[nodes], trial.position_low[nodes], trial.nodal_frames[nodes],
                trial.cell_rotations[i], trial.resultants[i])
        if kind == "generalized":
            response = seam.operator_call(operator, "evaluate", context.guard, context.started,
                                          *args, origin=origins[i])
            eid = context.elements[i][0]
            work = seam.line_work(operator.reference, trial.positions[nodes],
                trial.position_low[nodes], trial.cell_rotations[i],
                target * program.pattern.force(eid), order=operator.order)
            reduced = response.residual - work.gradient
            local_r.append(reduced - seam.cell_couple_load(
                operator, target * program.pattern.density(eid)))
            local_k.append(seam.spatial_jacobian(
                reduced, response.hessian + response.hessian_low - work.hessian))
            works.append(work.value == work_values[i])
        else:
            response = operator.evaluate(*args, origin=origins[i], check=context.check)
            local_r.append(response.residual)
            local_k.append(response.hessian + response.hessian_low)
        histories.append(canonical(response.history) == canonical(responses[i].history))
        materials.append(response.material == responses[i].material)
    assert all(histories + materials + works)
    assert all(np.linalg.norm(k[24:, 24:]) > 0 and np.linalg.norm(k[24:, :24]) > 0
               for k in local_k)
    assert len(set(layout.slots[0]) & set(layout.slots[1])) == 6
    sparse_r, csr = seam.gather(layout.count, layout.slots, local_r, local_k)
    sparse_r[:len(external)] -= external
    dr = sparse_r - residual
    dk = csr.toarray() - dense
    absolute_r = float(np.max(np.abs(dr)))
    absolute_k = float(np.max(np.abs(dk)))
    normalized_r = float(np.linalg.norm(dr) / max(1., np.linalg.norm(residual)))
    normalized_k = float(np.linalg.norm(dk) / max(1., np.linalg.norm(dense)))
    assert max(absolute_r, absolute_k) <= 1e-9
    assert max(normalized_r, normalized_k) <= 1e-11
    scale = layout.scale if kind == "generalized" else layout.scales
    scaled = sparse_r / scale
    sparse_metrics = (float(np.linalg.norm(scaled[layout.equilibrium])),
                      float(np.linalg.norm(scaled[layout.compatibility])))
    np.testing.assert_allclose(sparse_metrics, metrics, atol=1e-11, rtol=1e-11)
    assert origin_bytes == canonical((accepted, accepted.histories))
    return dict(family=kind, status=completed.status, accepted_targets=1,
                coordinates=layout.count, retained_coordinates=layout.count-layout.nodal_count,
                nnz=int(csr.nnz), local_blocks_present=True, shared_physical_coordinates=6,
                equilibrium_compatibility=metrics, residual_absolute=absolute_r,
                tangent_absolute=absolute_k, residual_normalized=normalized_r,
                tangent_normalized=normalized_k, history_equal=True,
                material_response_equal=True, accepted_unchanged=True)


if __name__ == "__main__":
    print(json.dumps(dict(profile="G0-F08-D-v1", status="diagnostic_pass",
                          results=[case("generalized"), case("physical-fibre")]), sort_keys=True))
