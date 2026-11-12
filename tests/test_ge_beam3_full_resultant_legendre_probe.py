"""Local numerical equivalence and analytic-variation checks, not qualification."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement, DirectedHardeningSection
from docs.reference_cases.ge_beam3_full_resultant_legendre_probe import ElasticResultantProbe
from test_ge_beam3_curved_contrast_probe import make_curved


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_legendre_force_elimination_recovers_uncondensed_v5(slenderness):
    model, _ = make_curved(slenderness); e = model.mesh.elements[1]
    # This is an elastic Legendre identity check, not the registered force
    # fixture or a high-strain plasticity qualification.
    section = DirectedHardeningSection(e.core.section._elastic,
        np.array([1., .2, -.1, .3, -.4, .5]), 1e30, 1.)
    e = NativeP5BeamElement(1, e.node_ids, e.core.reference, section, line_force=np.zeros(3))
    probe = ElasticResultantProbe(e); ref = e.core.reference
    low = np.zeros((3, 3)); x = ref.coordinates.copy()
    x += np.array([[0., 0., 0.], [.001, -.002, .003], [.002, .001, -.001]])
    q = np.array([rotation(v)@frame for v, frame in zip(
        ([.01, -.02, .03], [-.02, .03, .01], [.03, .01, -.02]), ref.nodal_triads)])
    u = np.array([rotation([.005, -.003, .002]), rotation([-.004, .002, .006])])
    m = np.linspace(-.02, .03, 12)
    trial = probe.evaluate(x, low, q, u, np.r_[np.zeros(6), m])
    a = probe.a@trial.kinematics[:6]+probe.b@m
    full = probe.evaluate(x, low, q, u, np.r_[a, m])
    old = CenteredStationaryBeam(ref, e.core.section, position_low=low, order=e.core.order)
    reference = old.evaluate(x, q, u, m.reshape(2, 2, 3))
    retained = list(range(24))+list(range(30, 42)); internal = list(range(24, 30))
    h = full.hessian
    schur = h[np.ix_(retained, retained)]-h[np.ix_(retained, internal)]@np.linalg.solve(
        h[np.ix_(internal, internal)], h[np.ix_(internal, retained)])
    def relative(a, b): return np.linalg.norm(a-b)/max(1., np.linalg.norm(b))
    assert relative(full.residual[retained], reference.residual) < 1e-11
    assert relative(schur, reference.hessian) < 1e-11
    assert abs(full.potential-reference.potential) < 1e-11*max(1., abs(reference.potential))
    assert np.linalg.norm(full.residual[internal]) < 1e-11


def test_full_resultant_analytic_variations_and_state_ownership():
    model, _ = make_curved(100.); e = model.mesh.elements[1]
    probe = ElasticResultantProbe(e); ref = e.core.reference
    x = ref.coordinates.copy(); low = np.zeros((3, 3))
    q = ref.nodal_triads.copy(); u = np.repeat(np.eye(3)[None], 2, axis=0)
    p = np.linspace(-.2, .3, 18); frozen = (x.copy(), q.copy(), u.copy(), p.copy())
    centre = np.linspace(-.02, .03, 42); direction = np.cos(np.arange(42))*0.2
    evaluated = probe.evaluate(x, low, q, u, p, increment=centre)
    epsilon = 1e-5
    plus = probe.evaluate(x, low, q, u, p, increment=centre+epsilon*direction)
    minus = probe.evaluate(x, low, q, u, p, increment=centre-epsilon*direction)
    np.testing.assert_allclose((plus.residual-minus.residual)/(2*epsilon),
        evaluated.hessian@direction, rtol=1e-7, atol=1e-7)
    assert abs((plus.potential-minus.potential)/(2*epsilon)-evaluated.residual@direction) < 1e-7
    np.testing.assert_allclose(evaluated.hessian, evaluated.hessian.T, rtol=1e-11, atol=1e-11)
    for before, after in zip(frozen, (x, q, u, p)): np.testing.assert_array_equal(before, after)
    assert not evaluated.production_qualified


def test_elastic_probe_rejects_nonlinear_branch_and_rotation_chart():
    model, _ = make_curved(100.); e = model.mesh.elements[1]
    probe = ElasticResultantProbe(e); ref = e.core.reference
    args = (ref.coordinates, np.zeros((3, 3)), ref.nodal_triads,
            np.tile(np.eye(3), (2, 1, 1)))
    with pytest.raises(ValueError, match='nonlinear section branch'):
        probe.evaluate(*args, np.full(18, 1e12))
    step = np.zeros(42); step[3] = np.pi
    with pytest.raises(ValueError, match='rotation chart'):
        probe.evaluate(*args, np.zeros(18), increment=step)
