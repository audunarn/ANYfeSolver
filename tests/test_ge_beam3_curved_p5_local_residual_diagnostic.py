"""Reproduce an open stopping-rule weakness; passing is NOT qualification.

These deliberately assert the current defect so that the diagnostic remains
reproducible. A future correction must replace the defect assertions with its
reviewed acceptance contract, not treat these tests as correctness evidence.
"""

import numpy as np
import pytest

from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation
from docs.reference_cases.ge_beam3_curved_p5_nonlinear_mixed_probe import NonlinearMixedBeamProbe
from docs.reference_cases.ge_beam3_curved_p5_section_probe import DirectedHardeningSectionProbe
from test_ge_beam3_curved_p5_algebra_probe import reference, perturbed


@pytest.fixture(scope='module')
def witness():
    base = reference(.1)
    scale = 1/32
    ref = CurvedBeam3ReferenceGeometry(base.coordinates*scale, base.nodal_triads)
    x, q, _ = perturbed(ref)
    x = ref.coordinates+(x-ref.coordinates)*scale*.1
    law = DirectedHardeningSectionProbe(
        np.diag([1000., 400., 400., .02, .01, .02]),
        [1., 0., 0., 0., 0., 0.], 1e6, 1.)
    model = NonlinearMixedBeamProbe(ref, law, order=8)
    root = model.solve(x, q)
    original = model.evaluate(x, q, root.local_rotations, root.moments)
    h = original.hessian
    # dr_external / dr_internal at fixed external coordinates. This is a
    # diagnostic sensitivity, never a replacement residual or acceptance rule.
    sensitivity = np.linalg.solve(h[18:, 18:], h[18:, :18]).T
    column = int(np.argmax(np.linalg.norm(sensitivity, axis=0)))
    intended = np.eye(18)[column]*4e-12
    step = np.linalg.solve(h[18:, 18:], intended)
    rotations = np.array([rotation(step[3*c:3*c+3]) @ root.local_rotations[c]
                          for c in (0, 1)])
    moments = root.moments+step[6:].reshape(2, 2, 3)
    saved = [a.copy() for a in (x, q, rotations, moments)]
    accepted = model.solve(x, q, initial_rotations=rotations,
                           initial_moments=moments)
    evaluation = model.evaluate(x, q, accepted.local_rotations, accepted.moments)
    return model, root, original, accepted, evaluation, sensitivity, saved, (x, q, rotations, moments)


def test_current_local_acceptance_does_not_bound_external_force_error(witness):
    _, root, _, accepted, _, _, _, _ = witness
    assert root.local_residual_norm < 1e-13
    assert 3e-12 < accepted.local_residual_norm < 5e-12
    assert accepted.iterations == 0 and accepted.evaluations == 1
    assert np.linalg.norm(accepted.residual-root.residual) > 1e-10


def test_internal_to_external_sensitivity_explains_manufactured_witness(witness):
    _, root, original, accepted, evaluation, sensitivity, _, _ = witness
    actual = accepted.residual-root.residual
    predicted = sensitivity @ (evaluation.residual[18:]-original.residual[18:])
    assert np.max(np.linalg.norm(sensitivity, axis=0)) > 100
    assert np.linalg.norm(actual-predicted) < .01*np.linalg.norm(actual)


def test_diagnostic_preserves_external_coordinates_and_station_origins(witness):
    model, root, _, accepted, _, _, saved, inputs = witness
    for before, after in zip(saved, inputs):
        np.testing.assert_array_equal(before, after)
    assert len(root.stations) == len(accepted.stations) == 16
    assert tuple(s.response.origin for s in root.stations) == model.origins
    assert tuple(s.response.origin for s in accepted.stations) == model.origins
    assert not any(s.response.plastic_active for s in accepted.stations)
