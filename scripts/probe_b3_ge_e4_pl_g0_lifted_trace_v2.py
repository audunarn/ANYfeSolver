"""G0 F03-B-v2: normalize the fixture's compensated coordinate pairs."""

from __future__ import annotations

import json

import numpy as np

import probe_b3_ge_e4_pl_g0_lifted_trace as prior
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair


original_native = prior.native_work


def normalized_native(reference, positions, position_low, cells, force, *, order):
    high = np.empty_like(positions)
    low = np.empty_like(position_low)
    for i in range(3):
        for j in range(3):
            high[i, j], low[i, j] = split_sum((
                float(positions[i, j]), float(position_low[i, j])))
            validate_pair(float(high[i, j]), float(low[i, j]))
    assert np.allclose(high+low, positions+position_low, rtol=0., atol=1e-15)
    return original_native(reference, high, low, cells, force, order=order)


prior.native_work = normalized_native

if __name__ == "__main__":
    results = [prior.row(False), prior.row(True)]
    passed = all(case["status"] == "diagnostic_pass" for case in results)
    print(json.dumps(dict(profile="G0-F03-B-v2",
                          status="diagnostic_pass" if passed else "failed",
                          results=results), sort_keys=True))
    raise SystemExit(0 if passed else 1)
