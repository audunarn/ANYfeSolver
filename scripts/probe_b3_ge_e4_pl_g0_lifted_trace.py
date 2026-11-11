"""G0 F03-B: independent station integration versus native lifted B3 line work."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from anyfesolver._ge_beam3_fibre_line_work import evaluate as native_work
from test_ge_beam3_generalized_slenderness_diagnostic import make


FORCE = np.array((.2, -.3, .4))


def independent(reference, positions, position_low, cells, delta, *, order=4):
    """Integrate native physical station position without importing its work formula."""
    x = positions + position_low
    perturbed = x + delta[:18].reshape(3, 6)[:, :3]
    rotations = [Rotation.from_rotvec(delta[18+3*c:21+3*c]).as_matrix() @ cells[c]
                 for c in range(2)]
    roots, weights = np.polynomial.legendre.leggauss(order)
    total = 0.
    max_reference_error = 0.
    for c in (0, 1):
        for root, weight in zip(roots, weights):
            t = float((root+1.)/2.)
            xi = c-1+t
            lift = reference.half_cell_lift(c, t)
            ref = reference.position_pair(xi)
            ref_position = ref[0] + ref[1]
            chord = (1.-t)*reference.coordinates[c]+t*reference.coordinates[c+1]
            max_reference_error = max(max_reference_error,
                float(np.linalg.norm(chord + lift - ref_position)))
            current = (1.-t)*perturbed[c]+t*perturbed[c+1]+rotations[c] @ lift
            total += float(weight*reference.jacobian(xi)/2.)*float(FORCE @ (current-ref_position))
    return total, max_reference_error


def row(curved):
    model = make(100., curved, np.eye(3))[0]
    reference = model.mesh.elements[1].operator.reference
    nominal = reference.coordinates
    positions = nominal + .01*np.sin(np.arange(9).reshape(3, 3)+.4)
    low = 1e-12*np.sin(np.arange(9).reshape(3, 3)+.9)
    cells = np.array([Rotation.from_rotvec(v).as_matrix() for v in
                      ((.04,-.02,.03), (-.03,.05,.02))])
    actual = native_work(reference, positions, low, cells, FORCE, order=4)
    zero = np.zeros(24)
    value, reference_error = independent(reference, positions, low, cells, zero)
    h = 2e-6
    derivative = np.zeros(24)
    for i in range(24):
        direction = np.zeros(24)
        direction[i] = 1.
        plus = independent(reference, positions, low, cells, h*direction)[0]
        minus = independent(reference, positions, low, cells, -h*direction)[0]
        derivative[i] = (plus-minus)/(2*h)
    gradient_error = float(np.linalg.norm(derivative-actual.gradient[:24]) /
                           max(1., np.linalg.norm(derivative)))
    direction = np.sin(np.arange(24)+.31)
    direction /= np.linalg.norm(direction)
    expected = float(direction @ actual.hessian[:24, :24] @ direction)
    hessian_errors = []
    for step in (2e-4, 1e-4):
        plus = independent(reference, positions, low, cells, step*direction)[0]
        minus = independent(reference, positions, low, cells, -step*direction)[0]
        observed = (plus-2*value+minus)/step**2
        hessian_errors.append(abs(observed-expected)/max(1., abs(expected)))
    nodal_rotation = np.array([3,4,5,9,10,11,15,16,17])
    zero_rows = float(np.linalg.norm(actual.gradient[nodal_rotation]))
    stress_rows = float(np.linalg.norm(actual.gradient[24:]))
    rotational_work = float(np.linalg.norm(actual.gradient[18:24]))
    value_error = abs(value-actual.value)
    passed = (value_error <= 1e-11 and gradient_error <= 1e-7
              and max(hessian_errors) <= 1e-5 and reference_error <= 1e-12
              and zero_rows == stress_rows == 0.
              and (rotational_work > 1e-7 if curved else rotational_work <= 1e-12))
    return dict(curved=curved, status="diagnostic_pass" if passed else "failed",
                value_error=value_error, gradient_error=gradient_error,
                directional_hessian_errors=hessian_errors,
                reference_reconstruction_error=reference_error,
                nodal_rotation_gradient=zero_rows, retained_stress_gradient=stress_rows,
                cell_rotation_gradient=rotational_work, native_work=actual.value)


if __name__ == "__main__":
    results = [row(False), row(True)]
    passed = all(case["status"] == "diagnostic_pass" for case in results)
    print(json.dumps(dict(profile="G0-F03-B-v1",
                          status="diagnostic_pass" if passed else "failed",
                          results=results), sort_keys=True))
    raise SystemExit(0 if passed else 1)
