"""Independent high-precision work derivatives at a captured physical edge."""

from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.q4_s3_overlap_derivatives import (
    q4_s3_overlap_derivatives,
)
from scripts.general_contact_d1_wedge_decimal_oracle import (
    add, d, scale, work,
)


FIXTURE = (Path(__file__).parent / "fixtures" /
           "general_contact_d1_physical_edge_wedge.json")


def test_captured_physical_edge_work_hessian_has_independent_decimal_oracle():
    captured = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with localcontext() as context:
        context.prec = 80
        reference = [tuple(map(d, row)) for row in captured["source_reference"]]
        source = [tuple(map(d, row)) for row in captured["local_source"]]
        target = [tuple(map(d, row)) for row in captured["local_target"]]
        pressure = tuple(map(d, captured["local_pressure_pa"]))
        direction = [tuple(map(d, row)) for row in captured["direction_per_m"]]

        def value(step, order):
            q = [add(point, scale(rate, step))
                 for point, rate in zip(source, direction[:4])]
            t = [add(point, scale(rate, step))
                 for point, rate in zip(target, direction[4:])]
            return work(reference, q, t, pressure, order)

        answers = []
        for order in (4, 8):
            center, center_area, signature = value(Decimal(0), order)
            assert center_area > 0
            for step in (Decimal("2e-10"), Decimal("5e-11")):
                plus, plus_area, plus_signature = value(step, order)
                minus, minus_area, minus_signature = value(-step, order)
                assert min(plus_area, minus_area) > 0
                assert plus_signature == signature == minus_signature
                answers.append((
                    float((plus-minus)/(2*step)),
                    float((plus-2*center+minus)/(step*step)),
                ))

    for gradient, hessian in answers[1:]:
        np.testing.assert_allclose(gradient, answers[0][0], rtol=1e-8)
        np.testing.assert_allclose(hessian, answers[0][1], rtol=1e-8)

    jet = q4_s3_overlap_derivatives(
        np.asarray(captured["source_reference"]),
        np.asarray(captured["local_source"]),
        np.asarray(captured["local_target"]),
        area_tolerance=1e-9, weighted_gap_tolerance=1e-10,
        maximum_triangles=300, allow_warped_source=True,
        support_gradient_tolerance=1e-5,
        support_hessian_tolerance=1e-5,
        weighted_gap_gradient_tolerance=1e-5,
        weighted_gap_hessian_tolerance=1e-5,
    )
    weights = np.asarray(captured["local_pressure_pa"])
    direction_flat = np.asarray(captured["direction_per_m"]).ravel()
    gradient = float(weights @ (jet.weighted_gap_gradient @ direction_flat))
    hessian = float(sum(
        weights[i]*direction_flat @ jet.weighted_gap_hessian[i]
        @ direction_flat for i in range(4)
    ))
    np.testing.assert_allclose(gradient, answers[0][0], rtol=1e-5)
    np.testing.assert_allclose(hessian, answers[0][1], rtol=1e-5)
