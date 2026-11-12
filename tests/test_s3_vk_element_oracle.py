"""Validation of the first-principles S3 von Karman oracle and its agreement with production."""
import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from s3_vk_element_oracle import S3VonKarmanExcess

# Complex-step: force = dU/du and tangent = dF/du to 1e-12 of their scale (observed ~1e-15).
CS_RTOL = 1e-12
# Production agreement on random states: 1e-12 of the largest excess entry plus cancellation
# roundoff of the two responses (observed max relative error ~1e-12 only where the excess is tiny).
PROD_RTOL = 1e-12


def _random_case(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q if np.linalg.det(q) > 0 else -q
    local = np.array(((0., 0., 0.), (rng.uniform(.05, .3), 0., 0.), (rng.uniform(-.02, .02), rng.uniform(.05, .25), 0.)))
    local[1, 1] = rng.uniform(0., .02)
    points = (q @ local.T).T+rng.normal(size=3)
    normal = q[:, 2]*rng.choice([-1., 1.])
    return dict(points=points, normal=normal, E=rng.uniform(1e4, 2e5), nu=rng.uniform(.1, .45),
                thickness=rng.uniform(.02, .08), offset=rng.uniform(-.02, .02),
                polarity=int(rng.choice([-1, 1])))


def _production(case):
    model = FEModel("oracle")
    model.add_material("elastic", case["E"], case["nu"], density=1.)
    for i, p in enumerate(case["points"], 1):
        model.add_node(i, *p)
    element = create_shell_element(
        1, (1, 2, 3), "elastic", formulation="e4-pl-s3-v2d", thickness=case["thickness"],
        reference_normal=tuple(case["normal"]), reference_surface_offset=case["offset"],
        director_polarity=case["polarity"])
    model.add_element(1, element)
    return model, element, model.materials["elastic"]


@pytest.mark.parametrize("seed", range(6))
def test_oracle_force_and_tangent_are_energy_derivatives_by_complex_step(seed):
    rng = np.random.default_rng(seed)
    case = _random_case(rng)
    oracle = S3VonKarmanExcess(**case)
    u = rng.normal(size=18)*1e-2
    h = 1e-30
    gradient = np.array([oracle.energy(u+1j*h*np.eye(18)[k]).imag/h for k in range(18)])
    force = oracle.force(u)
    assert np.max(np.abs(gradient-force)) <= CS_RTOL*np.max(np.abs(force))
    tangent = np.column_stack([oracle.force(u+1j*h*np.eye(18)[k]).imag/h for k in range(18)])
    assert np.max(np.abs(tangent-oracle.tangent(u))) <= CS_RTOL*np.max(np.abs(tangent))
    assert np.max(np.abs(tangent-tangent.T)) <= CS_RTOL*np.max(np.abs(tangent))


@pytest.mark.parametrize("seed", range(12))
def test_oracle_matches_production_vk_excess_on_random_states(seed):
    rng = np.random.default_rng(100+seed)
    case = _random_case(rng)
    model, element, material = _production(case)
    oracle = S3VonKarmanExcess(**case)
    u = rng.normal(size=18)*rng.choice([1e-3, 1e-2])
    layers = int(rng.choice([3, 5]))
    f1, t1, _ = element.compute_nonlinear_response(model.mesh, material, u, None, layers, True, von_karman=True)
    f0, t0, _ = element.compute_nonlinear_response(model.mesh, material, u, None, layers, True, von_karman=False)
    excess_f, excess_t = f1-f0, t1-t0
    assert np.max(np.abs(excess_f)) > 0.
    # The excess is the difference of two large responses: allow their cancellation roundoff.
    slack_f, slack_t = 64*np.finfo(float).eps*np.max(np.abs(f0)), 64*np.finfo(float).eps*np.max(np.abs(t0))
    assert np.max(np.abs(excess_f-oracle.force(u))) <= PROD_RTOL*np.max(np.abs(excess_f))+slack_f
    assert np.max(np.abs(excess_t-oracle.tangent(u))) <= PROD_RTOL*np.max(np.abs(excess_t))+slack_t
    # rotation dofs are covered: the excess must load them when the offset is nonzero
    if abs(case["offset"]) > 5e-3:
        rotations = np.concatenate([np.arange(6*i+3, 6*i+6) for i in range(3)])
        assert np.max(np.abs(excess_f[rotations])) > 0.


@pytest.mark.parametrize("seed", range(8))
def test_analytic_directional_curvature_matches_oracle_and_finite_difference(seed):
    rng = np.random.default_rng(300+seed)
    case = _random_case(rng)
    model, element, material = _production(case)
    oracle = S3VonKarmanExcess(**case)
    u, v = rng.normal(size=18)*1e-2, rng.normal(size=18)
    curvature = element.von_karman_directional_curvature(model.mesh, material, u, v)
    # independent: d/dh T_vk(u+h v) v by complex step on the first-principles tangent
    h = 1e-30
    expected = oracle.tangent(u+1j*h*v).imag @ v/h
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(curvature-expected)) <= 1e-12*scale
    # the old production finite difference agrees to its own accuracy (the law is cubic:
    # the central difference is exact up to cancellation noise)
    step = 1e-5
    tangents = [element.compute_nonlinear_response(model.mesh, material, u+s*step*v, None, 5, True,
                                                   von_karman=True)[1] for s in (1., -1.)]
    difference = (tangents[0]-tangents[1]) @ v/(2*step)
    assert np.max(np.abs(curvature-difference)) <= 1e-6*scale
