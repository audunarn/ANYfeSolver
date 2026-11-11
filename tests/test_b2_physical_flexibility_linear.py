"""Local, preparatory oracles for the distinct physical B2 linear operator."""

from fractions import Fraction

import numpy as np
import pytest

from anyfesolver._b2_physical_flexibility_linear import (
    ConditioningAdmissionError, LinearOperator, LinearSection,
    OPERATOR_ID, RepresentabilityError,
)


def scalar(area=1e-14):
    return LinearOperator(1., LinearSection.isotropic(
        E=1., nu=0., area=area, Iy=1e-12, Iz=2e-12, J=3e-12,
        ky=.8, kz=.6,
    ))


@pytest.mark.parametrize("area", [1e-14, 1e-12, 1e-10, 2.**-80])
@pytest.mark.parametrize("direction, bending, shear", [(8, 1e-12, .6), (7, 2e-12, .8)])
def test_scalar_unfloored_independent_transverse_energy(area, direction, bending, shear):
    operator = scalar(area)
    u = np.zeros(12)
    u[direction] = .01
    response = operator.evaluate(u)
    G = Fraction(1, 2)
    EI = Fraction(bending)
    GA = G*Fraction(area)*Fraction(shear)
    expected = float(6*EI*GA/(GA+12*EI) * Fraction(.01)**2)
    assert response.energy > 0
    assert response.energy == pytest.approx(expected, rel=2e-15)
    assert response.force @ u == pytest.approx(2*expected, rel=2e-15)
    assert min(operator.modal_stiffness) > 0
    assert response.operator_id == OPERATOR_ID
    assert not response.production_qualified


def test_scalar_rigid_modes_reversal_work_and_station_equilibrium():
    operator = scalar(1e-6)
    B = operator.basic_matrix
    rigid = np.zeros((12, 6))
    for axis in range(3):
        rigid[axis, axis] = rigid[axis+6, axis] = 1
        rigid[axis+3, axis+3] = rigid[axis+9, axis+3] = 1
    rigid[8, 4] = -1
    rigid[7, 5] = 1
    assert np.linalg.matrix_rank(B) == 6
    np.testing.assert_allclose(B @ rigid, 0, atol=0)
    u = np.array([.02, -.04, .03, .01, -.02, .005, -.01, .03, -.05, .015, .02, -.008])
    d = np.array([.5, -.7, .2, .4, -.6, .3, -.1, .8, -.4, .2, .9, -.5])
    r = operator.evaluate(u, stations=(0., .5, 1.))
    np.testing.assert_allclose(r.force, r.tangent @ u, rtol=1e-14, atol=1e-24)
    np.testing.assert_allclose(operator.evaluate(d).force, r.tangent @ d, rtol=1e-14, atol=1e-24)
    assert r.energy == pytest.approx(.5*u @ r.force, rel=2e-15)
    np.testing.assert_allclose(r.resultants[:, 2], r.resultants[0, 2])
    np.testing.assert_allclose(r.resultants[:, 1], r.resultants[0, 1])
    assert r.resultants[0, 4] != r.resultants[-1, 4]
    assert r.resultants[0, 5] != r.resultants[-1, 5]
    section = np.diag(np.diag(np.asarray(operator.section.stiffness)))
    np.testing.assert_allclose(r.resultants, r.strains @ section, rtol=1e-14, atol=1e-24)
    assert np.linalg.norm(operator.evaluate(rigid[:, 4]).force) == 0


def test_extreme_shear_soft_mode_retained_without_dense_eigenvalue_subtraction():
    A = 2.**-80
    operator = LinearOperator(1., LinearSection.isotropic(
        E=1., nu=0., area=A, Iy=1., Iz=1., J=1., ky=1., kz=1.))
    u = np.zeros(12)
    u[8] = .01
    r = operator.evaluate(u)
    shear = Fraction(A)/2
    expected = float(6*shear/(shear+12) * Fraction(.01)**2)
    assert r.energy == pytest.approx(expected, rel=2e-15)
    assert r.energy > 0 and operator.modal_stiffness[2] > 0
    assert r.force[8] > 0 and r.resultants[1, 2] > 0


def test_generalized_exact_polynomial_station_work_and_coupling():
    S = np.diag([10., 7., 9., 3., 2., 4.])
    S[0, 4] = S[4, 0] = .4
    S[1, 5] = S[5, 1] = .3
    S[2, 3] = S[3, 2] = -.2
    operator = LinearOperator(2., LinearSection.generalized(S))
    u = np.array([.02, -.04, .03, .01, -.02, .005, -.01, .03, -.05, .015, .02, -.008])
    d = np.array([.5, -.7, .2, .4, -.6, .3, -.1, .8, -.4, .2, .9, -.5])
    r = operator.evaluate(u, stations=(0., .5, 1.))
    dr = operator.evaluate(d, stations=(0., .5, 1.))
    quadrature = 2./6 * sum(w * (a @ b) for w, a, b in zip(
        (1., 4., 1.), r.resultants, dr.strains))
    assert quadrature == pytest.approx(r.force @ d, rel=2e-14)
    np.testing.assert_allclose(r.resultants, r.strains @ S, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(r.force, r.tangent @ u, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(r.tangent, r.tangent.T, rtol=2e-14, atol=1e-14)
    assert r.resultants[0, 0] != 0 and r.resultants[0, 4] != 0


def test_owned_operator_definition_cannot_change_cached_mechanics():
    source = np.diag([10., 7., 9., 3., 2., 4.])
    section = LinearSection.generalized(source)
    operator = LinearOperator(2., section)
    displacement = np.arange(12, dtype=float) / 100
    before = operator.evaluate(displacement)
    source[0, 0] = 1000.
    with pytest.raises(AttributeError, match="immutable"):
        operator.length = Fraction(3)
    with pytest.raises(AttributeError, match="immutable"):
        operator.section = LinearSection.generalized(np.eye(6))
    with pytest.raises(AttributeError, match="immutable"):
        del operator.length
    with pytest.raises(AttributeError, match="immutable"):
        del operator.section
    with pytest.raises(AttributeError, match="immutable"):
        del operator._sealed
    with pytest.raises(TypeError):
        operator._B[0][0] = 7
    with pytest.raises(TypeError):
        operator._compliance[0][0] = 7
    after = operator.evaluate(displacement)
    np.testing.assert_array_equal(after.force, before.force)
    np.testing.assert_array_equal(after.resultants, before.resultants)
    assert operator.section.stiffness[0][0] == 10.


def test_scalar_provenance_is_owned_and_cannot_relabel_diagonal_stiffness():
    scalar_section = scalar().section
    assert scalar_section.scalar_parameters == (1., 0., 1e-14, 1e-12,
                                                2e-12, 3e-12, .8, .6)
    params = list(scalar_section.scalar_parameters)
    direct = LinearSection(scalar_section.stiffness, "isotropic_scalar",
                           scalar_parameters=params)
    params[2] = 1.
    assert direct.scalar_parameters == scalar_section.scalar_parameters
    with pytest.raises(ValueError, match="source parameters required"):
        LinearSection(scalar_section.stiffness, "isotropic_scalar")
    with pytest.raises(ValueError, match="does not match"):
        LinearSection(scalar_section.stiffness, "isotropic_scalar",
                      scalar_parameters=(1., 0., 1., 1e-12, 2e-12, 3e-12, .8, .6))
    with pytest.raises(ValueError, match="no scalar source"):
        LinearSection(tuple(tuple(row) for row in np.eye(6)), "elastic_generalized",
            scalar_parameters=scalar_section.scalar_parameters)


def test_explicit_input_and_representability_negatives():
    with pytest.raises(ValueError, match="positive"):
        scalar(0)
    with pytest.raises(ValueError, match="Poisson"):
        LinearSection.isotropic(E=1, nu=.5, area=1, Iy=1, Iz=1, J=1, ky=1, kz=1)
    with pytest.raises(ValueError, match="positive-definite"):
        LinearSection.generalized(np.diag([1., 1., 1., 1., 1., -1.]))
    S = np.eye(6)
    S[0, 1] = S[1, 0] = 1 - 1e-14
    # The exactly SPD input law is valid; its emitted tangent must still satisfy
    # the output certificate. An exact inverse needs no floating pivot veto.
    section = LinearSection.generalized(S)
    with pytest.raises(ConditioningAdmissionError, match="output representation"):
        LinearOperator(2., section)
    with pytest.raises(RepresentabilityError, match="underflow"):
        scalar(2.**-1074).evaluate([0., 0., 0., 0., 0., 0., 0., 0., 1e-100, 0., 0., 0.])
    with pytest.raises(RepresentabilityError, match="overflow"):
        LinearSection.isotropic(E=1e308, nu=0., area=1e308, Iy=1., Iz=1., J=1., ky=1., kz=1.)


def test_nonzero_subnormal_rounding_with_large_relative_error_is_rejected():
    minimum = float(np.nextafter(0., 1.))
    exact_axial_mode = Fraction(3, 2) * Fraction.from_float(minimum)
    rounded = float(exact_axial_mode)
    assert rounded == 2 * minimum
    # Independent rational oracle: the finite, nonzero rounded mode is 1/3
    # above its physical value.  A zero-only underflow check would admit it.
    assert abs(Fraction.from_float(rounded) - exact_axial_mode) / exact_axial_mode == Fraction(1, 3)
    section = LinearSection.isotropic(
        E=1., nu=-.5, area=3 * minimum, Iy=1e-12, Iz=2e-12,
        J=3e-12, ky=1., kz=1.,
    )
    with pytest.raises(RepresentabilityError, match="relative precision"):
        LinearOperator(2., section)

    # Nearby exactly representable physical modes retain the old route.
    retained = LinearOperator(2., LinearSection.isotropic(
        E=1., nu=-.5, area=4 * minimum, Iy=1e-12, Iz=2e-12,
        J=3e-12, ky=1., kz=1.,
    ))
    assert retained.modal_stiffness[0] == 2 * minimum


def test_generalized_coupled_operator_has_equivalent_numerical_length_units():
    """Private operator identity; public physical definitions still require SI."""
    section = np.diag([22., 13., 17., 9., 7., 11.])
    section[0, 4] = section[4, 0] = 1.5
    section[1, 5] = section[5, 1] = -.7
    factor = 1000.
    wrench_scale = np.diag([1., 1., 1., factor, factor, factor])
    pose_scale = np.diag([factor, factor, factor, 1., 1., 1.])
    nodal_scale = np.kron(np.eye(2), pose_scale)
    nodal_inverse = np.linalg.inv(nodal_scale)
    original = LinearOperator(2., LinearSection.generalized(section))
    converted = LinearOperator(
        2.*factor,
        LinearSection.generalized(wrench_scale@section@wrench_scale))
    displacement = np.array((.01, -.02, .03, .004, -.006, .008,
                             -.013, .025, -.017, .003, .005, -.007))
    stations = (.0, .23, .77, 1.)
    a = original.evaluate(displacement, stations=stations)
    b = converted.evaluate(nodal_scale@displacement, stations=stations)

    assert b.energy == pytest.approx(factor*a.energy, rel=2e-12)
    np.testing.assert_allclose(
        b.force, factor*nodal_inverse.T@a.force, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(
        b.tangent, factor*nodal_inverse.T@a.tangent@nodal_inverse,
        rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(
        b.resultants, a.resultants@wrench_scale.T,
        rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(
        b.strains, a.strains@np.linalg.inv(wrench_scale).T,
        rtol=2e-12, atol=2e-11)

    offset = np.array((.02, .015, -.01))
    trace_a = original.trace(displacement, stations=stations,
                             offset_local_m=offset)
    trace_b = converted.trace(nodal_scale@displacement, stations=stations,
                              offset_local_m=factor*offset)
    np.testing.assert_allclose(
        trace_b.reference_points, factor*trace_a.reference_points,
        rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(
        trace_b.pose, trace_a.pose@pose_scale.T,
        rtol=2e-12, atol=2e-11)
    for jac_a, jac_b in zip(trace_a.jacobian, trace_b.jacobian):
        np.testing.assert_allclose(
            jac_b, pose_scale@jac_a@nodal_inverse,
            rtol=2e-12, atol=2e-11)
