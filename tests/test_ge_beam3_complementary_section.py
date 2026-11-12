"""Stress-control nonlinear section identities; not beam qualification."""
from fractions import Fraction as F
from math import fsum
import numpy as np
import pytest
from anyfesolver._ge_beam3_complementary_section import ComplementaryDirectedSection
from anyfesolver._ge_beam3_p5.section import DirectedHardeningSection, SectionHistory
from anyfesolver._ge_beam3_p5_seeded.core import canonical


def section(contrast=1.):
    base = np.array([[1., .05, -.08, .12, -.04, .03], [0., 1., .06, -.05, .08, .04],
        [0., 0., 1., .07, .02, -.05], [0., 0., 0., 1., .1, .05],
        [0., 0., 0., 0., 1., -.1], [0., 0., 0., 0., 0., 1.]])
    factor = base@np.diag(np.sqrt([30*contrast, 10*contrast, 12*contrast, 2., 1., 1.5]))
    return DirectedHardeningSection(factor.T@factor, np.array([1., .2, -.1, .3, -.4, .5]), .25, 2.)


@pytest.mark.parametrize('stress', [np.zeros(6), np.array([.1, -.05, .02, .01, .02, -.01]),
    np.array([1., -.2, .1, .3, -.4, .5]), np.array([-1., .2, -.1, -.3, .4, -.5])])
@pytest.mark.parametrize('origin', [SectionHistory(), SectionHistory(.15, .3)])
def test_complementary_response_reproduces_primal_return(stress, origin):
    source = section(); dual = ComplementaryDirectedSection(source).response(stress, origin)
    primal = source.strain_response(dual.total_strain, origin)
    np.testing.assert_allclose(primal.resultants, stress, rtol=1e-11, atol=1e-11)
    assert abs(primal.history.plastic_coordinate-dual.history.plastic_coordinate) <= 1e-11
    assert abs(primal.history.accumulated-dual.history.accumulated) <= 1e-11
    np.testing.assert_allclose(primal.tangent@dual.compliance, np.eye(6), rtol=1e-11, atol=1e-11)
    assert abs(primal.incremental_potential-dual.primal_incremental_potential) <= 1e-11
    work = fsum(float(s)*float(v) for s, v in zip(stress, dual.total_strain))
    work += fsum(float(s)*float(v) for s, v in zip(stress, dual.total_strain_low))
    assert abs(work-dual.dual_potential-dual.primal_incremental_potential) <= 1e-11
    assert dual.dissipation >= 0 and dual.history.accumulated >= origin.accumulated


@pytest.mark.parametrize('sign', [-1., 1.])
def test_dual_first_second_variations_on_plastic_branches(sign):
    law = ComplementaryDirectedSection(section())
    s = sign*np.array([1., -.2, .1, .3, -.4, .5]); origin = SectionHistory(.03, .1)
    direction = np.array([.3, -.2, .4, -.1, .2, .15]); eps = 1e-5
    centre = law.response(s, origin)
    plus = law.response(s+eps*direction, origin); minus = law.response(s-eps*direction, origin)
    derivative = (plus.total_strain-minus.total_strain)/(2*eps)
    np.testing.assert_allclose(derivative, centre.compliance@direction, rtol=1e-7, atol=1e-7)
    assert abs((plus.dual_potential-minus.dual_potential)/(2*eps)-centre.total_strain@direction) <= 1e-7
    assert centre.derivative_kind == 'CLASSICAL_SMOOTH_BRANCH'


def test_yield_boundary_is_explicit_not_claimed_classically_smooth():
    law = ComplementaryDirectedSection(section())
    value = law.response([.25, 0., 0., 0., 0., 0.])
    assert value.branch == 'YIELD_BOUNDARY'
    assert value.derivative_kind == 'SEMISMOOTH_ELASTIC_SELECTION'
    assert value.plastic_increment == 0


@pytest.mark.parametrize('contrast', [1., 10000., 1000000000000.])
def test_retained_elastic_strain_survives_plastic_offset(contrast, tmp_path):
    law = ComplementaryDirectedSection(section(contrast))
    stress = np.array([1., -.2, .1, .3, -.4, .5])
    value = law.response(stress, SectionHistory(.1, .2))
    # Verify the full represented elastic strain by independent exact rational
    # products, not by dropping its low part or subtracting plastic strain.
    exact_error = [sum((F(float(c))*(F(float(x))+F(float(y))) for c, x, y in
        zip(row, value.elastic_strain, value.elastic_strain_low)), F(0))-F(float(s))
        for row, s in zip(law.elastic, stress)]
    assert max(abs(float(x)) for x in exact_error) <= 1e-11
    assert value.elastic_residual_inf <= 1e-11
    work = fsum(float(s)*float(v) for s, v in zip(stress, value.total_strain))
    work += fsum(float(s)*float(v) for s, v in zip(stress, value.total_strain_low))
    assert abs(work-value.dual_potential-value.primal_incremental_potential) <= 1e-11
    with (tmp_path/'complementary.json').open('xb') as stream: stream.write(canonical(value))
    for array in (value.resultants, value.elastic_strain, value.elastic_strain_low,
                  value.total_strain, value.total_strain_low, value.compliance, value.compliance_factor):
        with pytest.raises(ValueError): array.setflags(write=True)


def test_exact_scalar_conjugate_stationarity_and_fenchel_identity():
    # Independent rational scalar minimization, not the production return-map.
    c, a, y, h, z0, p0 = map(F, (7, 2, 3, 5, 1, 2))
    for s in map(F, (-20, -2, 0, 2, 20)):
        drive = a*s; radius = y+h*p0
        dz = (max(drive-radius, F(0))-max(-drive-radius, F(0)))/h
        z = z0+dz; p = p0+abs(dz); e = s/c+a*z
        primal = c*(e-a*z)**2/2+h*(p*p-p0*p0)/2+y*abs(dz)
        dual = s*s/(2*c)+z0*drive+max(abs(drive)-radius, F(0))**2/(2*h)
        assert s*e == primal+dual
        for alternative in (dz-F(1, 7), F(0), dz+F(1, 7)):
            alt = c*(e-a*(z0+alternative))**2/2+h*((p0+abs(alternative))**2-p0*p0)/2+y*abs(alternative)
            assert alt >= primal


def test_trials_do_not_advance_their_origin_and_replay_is_deterministic(tmp_path):
    law = ComplementaryDirectedSection(section()); origin = SectionHistory()
    states = []
    for stress in ([1., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [-2., 0., 0., 0., 0., 0.]):
        before = origin
        trial = law.response(stress, origin)
        assert origin == before
        assert canonical(law.response(stress, origin)) == canonical(trial)
        states.append(trial); origin = trial.history
    assert states[-1].history.plastic_coordinate < states[0].history.plastic_coordinate
    assert states[-1].history.accumulated > states[0].history.accumulated
    with (tmp_path/'history.json').open('xb') as stream: stream.write(canonical(states))


def test_section_capture_owns_coefficients_and_forbids_mutation():
    source = section(); law = ComplementaryDirectedSection(source)
    before = canonical(law.response(np.ones(6)))
    source._yield = 100.
    assert canonical(law.response(np.ones(6))) == before
    for name in ('elastic', 'root', 'hardening', 'yield_force', 'direction', 'identity'):
        with pytest.raises(AttributeError): setattr(law, name, None)
    for array in (law.elastic, law.direction, law.root, law.elastic_compliance, law.elastic_factor):
        with pytest.raises(ValueError): array.setflags(write=True)


@pytest.mark.parametrize('origin', [SectionHistory(True, 1.), SectionHistory(2., 1.),
    SectionHistory(float('nan'), 0.), SectionHistory(0., float('inf'))])
def test_invalid_history_rejected_without_advance(origin):
    with pytest.raises(ValueError): ComplementaryDirectedSection(section()).response(np.ones(6), origin)


@pytest.mark.parametrize('stress', [[1., 2.], [float('nan')]*6, [float('inf')]*6])
def test_invalid_resultants_rejected(stress):
    with pytest.raises(ValueError): ComplementaryDirectedSection(section()).response(stress)


@pytest.mark.parametrize('stress', [np.zeros(6), np.ones(6), -np.ones(6)])
def test_compliance_factor_preserves_smooth_branch_work(stress):
    law = ComplementaryDirectedSection(section())
    value = law.response(stress)
    np.testing.assert_allclose(value.compliance_factor.T@value.compliance_factor, value.compliance,
        rtol=1e-11, atol=1e-11)
    direction = np.array([.3, -.2, .1, .2, -.1, .4])
    assert np.linalg.norm(value.compliance_factor@direction) > 0.
    if value.plastic_increment:
        expected = (abs(law.direction@stress)-law.yield_force)/law.hardening
        assert abs(value.plastic_increment-expected) < 1e-14
