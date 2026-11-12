"""Independent complete line/surface work and swept-path controls."""
import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver.general_contact.deformable_beam_s3_line import DeformableBeamS3Line


def fixture(count=3):
    skin = np.array(((-2., -2., 0.), (2., -2., 0.), (0., 2., 0.)))
    beam = np.column_stack((np.linspace(-.3, .3, count), np.zeros(count), np.full(count, .02)))
    kernel = DeformableBeamS3Line(beam, skin, radius=.01, line_stiffness=100.,
                                side_normal=(0., 0., 1.), rim_clearance=.02)
    return kernel, beam, skin


def test_uniform_b2_force_complete_target_work_and_wrench():
    kernel, beam, skin = fixture(2)
    beam[:, 2] = .009
    result = kernel.evaluate(beam, skin)
    assert result.potential == pytest.approx(.5*100.*.001**2*.6, rel=1.e-12)
    forces = -result.gradient.reshape(5, 3)
    np.testing.assert_allclose(forces[:2].sum(axis=0), (0., 0., .06), atol=1.e-13)
    np.testing.assert_allclose(forces[2:, 2], (-.015, -.015, -.03), atol=1.e-13)
    np.testing.assert_allclose(forces.sum(axis=0), 0., atol=1.e-13)
    np.testing.assert_allclose(np.cross(np.r_[beam, skin], forces).sum(axis=0), 0., atol=1.e-13)


def independent_energy(beam, skin):
    n = np.cross(skin[1]-skin[0], skin[2]-skin[0])
    n /= np.linalg.norm(n)
    def integrand(s):
        N = np.array((2*(s-.5)*(s-1), 4*s*(1-s), 2*s*(s-.5)))
        gap = n@(N@beam-skin[0])-.01
        return .5*100.*min(gap, 0.)**2*.6
    # Independent scalar roots partition quadrature at activation.
    values = np.array([n@(p-skin[0])-.01 for p in beam])
    coeff = np.array((2*values[0]-4*values[1]+2*values[2],
                      -3*values[0]+4*values[1]-values[2], values[0]))
    roots = [float(z.real) for z in np.roots(coeff) if abs(z.imag)<1.e-12 and 0<z.real<1]
    return quad(integrand, 0., 1., points=roots, epsabs=1.e-15, epsrel=1.e-12)[0]


def test_partial_b3_deforming_triangle_independent_energy_work_and_tangent():
    kernel, beam, skin = fixture()
    beam[:, 2] = (.0102, .0095, .0103)
    skin[2, 2] = .0003
    x = np.r_[beam.ravel(), skin.ravel()]
    direction = np.array((.2, -.1, .3, -.2, .1, -.4, .1, .2, .1,
                          -.1, .2, -.2, .2, -.2, .3, -.2, .1, -.1))
    result = kernel.evaluate(beam, skin)
    assert abs(result.potential-independent_energy(beam, skin)) < 1.e-14
    for h in (1.e-6, 3.e-7):
        plus, minus = x+h*direction, x-h*direction
        independent_work = (independent_energy(plus[:9].reshape(3,3), plus[9:].reshape(3,3))
                            -independent_energy(minus[:9].reshape(3,3), minus[9:].reshape(3,3)))/(2*h)
        assert abs(independent_work-result.gradient@direction)/abs(independent_work) < 1.e-5
        p = kernel.evaluate(plus[:9].reshape(3,3), plus[9:].reshape(3,3))
        m = kernel.evaluate(minus[:9].reshape(3,3), minus[9:].reshape(3,3))
        derivative = (p.gradient-m.gradient)/(2*h)
        assert np.linalg.norm(derivative-result.hessian@direction)/np.linalg.norm(derivative) < 1.e-5
    forces = -result.gradient.reshape(6,3)
    np.testing.assert_allclose(forces.sum(axis=0), 0., atol=1.e-12)
    np.testing.assert_allclose(np.cross(np.r_[beam,skin], forces).sum(axis=0), 0., atol=1.e-12)


def test_common_coordinate_rotation_preserves_physical_work():
    kernel, beam, skin = fixture()
    beam[:,2] = .009
    angle = .7
    Q = np.array(((np.cos(angle), 0., np.sin(angle)), (0.,1.,0.),
                  (-np.sin(angle),0.,np.cos(angle))))
    shifted = np.array((.3, -.2, .1))
    other = DeformableBeamS3Line(beam@Q.T+shifted, skin@Q.T+shifted,
                                radius=.01, line_stiffness=100., side_normal=Q@(0.,0.,1.),
                                rim_clearance=.02)
    first, second = kernel.evaluate(beam,skin), other.evaluate(beam@Q.T+shifted,skin@Q.T+shifted)
    assert second.potential == pytest.approx(first.potential, rel=1.e-11)
    np.testing.assert_allclose(second.gradient.reshape(6,3), first.gradient.reshape(6,3)@Q.T,
                               rtol=1.e-10, atol=1.e-12)


def test_affine_deforming_clear_path_and_explicit_budget_exhaustion():
    kernel, beam, skin = fixture()
    changed = skin.copy()
    changed[2,2] = 1.e-5
    assert kernel.swept_admissible(beam,skin,beam,changed,gap_tolerance=1.e-5,maximum_intervals=64)
    assert kernel.swept_admissible(beam,skin,beam,skin,gap_tolerance=1.e-5,maximum_intervals=1)
    large = skin.copy()
    large[2,2] = .1
    assert not kernel.swept_admissible(beam,skin,beam,large,gap_tolerance=1.e-5,maximum_intervals=1)


def rotation(axis, angle):
    if axis == 'x':
        return np.array(((1.,0.,0.),(0.,np.cos(angle),-np.sin(angle)),
                         (0.,np.sin(angle),np.cos(angle))))
    return np.array(((np.cos(angle),-np.sin(angle),0.),(np.sin(angle),np.cos(angle),0.),(0.,0.,1.)))


def test_clear_endpoint_gaps_do_not_hide_rotating_plane_crossing():
    kernel, beam, skin = fixture(2)
    start, end = skin@rotation('x',np.pi/3).T, skin@rotation('x',-np.pi/3).T
    b0, b1 = beam.copy(), beam.copy()
    for b, pose in ((b0,start),(b1,end)):
        n = np.cross(pose[1]-pose[0],pose[2]-pose[0]);n /= np.linalg.norm(n)
        b[:,1:] = (.01002*n)[1:]
        assert kernel.geometry(b,pose)[1][-1] > 0
    assert not kernel.swept_admissible(b0,start,b1,end,gap_tolerance=1.e-5,maximum_intervals=64)


def test_clear_endpoint_rims_do_not_hide_interior_rim_crossing():
    kernel, beam, skin = fixture(2)
    start, end = skin@rotation('z',np.pi/3).T, skin@rotation('z',-np.pi/3).T
    beam[:,0] = (.74,.76)
    for pose in (start,end):
        assert kernel.geometry(beam,pose)[2] > kernel.rim_clearance
    assert not kernel.swept_admissible(beam,start,beam,end,gap_tolerance=1.e-5,maximum_intervals=128)


def test_invalid_geometry_and_physical_rim_are_explicit_non_success():
    kernel, beam, skin = fixture()
    with pytest.raises(ValueError, match='inverted|collapsed|unresolved'):
        kernel.evaluate(beam,skin[[0,2,1]])
    beam[:,0] += 10.
    with pytest.raises(ValueError, match='rim'):
        kernel.evaluate(beam,skin)
