"""G0 F08-S-v2: complete the registered material and metric parity checks."""

from __future__ import annotations

import json

import numpy as np

import probe_b3_ge_e4_pl_g0_local_sparse_seam as v1
from anyfesolver._ge_beam3_p5_seeded.core import canonical


original_check = v1.check


def check(kind, context, residual, dense, metrics, responses, local_r, local_k,
          parameter, slots, external):
    result = original_check(kind, context, residual, dense, metrics, responses,
                            local_r, local_k, parameter, slots, external)
    count = context.count if kind == "generalized" else context.layout.count
    sparse_r, tangent = v1.gather(count, slots, local_r, local_k)
    sparse_r[:len(external)] -= external
    layout = context if kind == "generalized" else context.layout
    scale = layout.scale if kind == "generalized" else layout.scales
    scaled = sparse_r / scale
    sparse_metrics = [float(np.linalg.norm(scaled[layout.equilibrium])),
                      float(np.linalg.norm(scaled[layout.compatibility]))]
    np.testing.assert_allclose(sparse_metrics, metrics, atol=1e-11, rtol=1e-11)
    assert tangent.shape == dense.shape

    mechanical = context.initial.mechanical
    origins = context.initial.origins
    histories_equal = []
    materials_equal = []
    for i, old in enumerate(responses):
        nodes = layout.nodes[i]
        if kind == "generalized":
            operator = context.elements[i][1].operator
            new = v1.operator_call(operator, "evaluate", context.guard, context.started,
                mechanical.positions[nodes], mechanical.position_low[nodes],
                mechanical.nodal_frames[nodes], mechanical.cell_rotations[i],
                mechanical.resultants[i], origin=origins[i])
        else:
            operator = context.probes[i]
            new = operator.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes],
                mechanical.nodal_frames[nodes], mechanical.cell_rotations[i],
                mechanical.resultants[i], origin=origins[i], check=context.check)
        histories_equal.append(canonical(new.history) == canonical(old.history))
        materials_equal.append(new.material == old.material)
    assert all(histories_equal + materials_equal)
    result.update(history_equal=True, material_response_equal=True,
                  sparse_metrics=sparse_metrics)
    return result


v1.check = check

if __name__ == "__main__":
    print(json.dumps(dict(profile="G0-F08-S-v2", status="diagnostic_pass",
                          results=[v1.generalized(), v1.fibre()]), sort_keys=True))
