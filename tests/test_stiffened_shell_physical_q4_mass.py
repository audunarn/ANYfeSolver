"""Independent kinetic-volume controls for the new flat director plate mass."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._stiffened_shell_physical_q4_mass import physical_mass


VERTICES = np.array(((0., 0., 0.), (1.2, 0., 0.),
                     (1.2, .8, 0.), (0., .8, 0.)))


def test_mass_matches_analytic_rectangle_without_normal_spin_inertia():
    rho, h = 7850., .018
    matrix = physical_mass(VERTICES, density_kg_m3=rho, thickness_m=h)
    nodal = 1.2*.8/36.*np.array(((4.,2.,1.,2.), (2.,4.,2.,1.),
                               (1.,2.,4.,2.), (2.,1.,2.,4.)))
    expected = np.kron(nodal, np.diag((rho*h,)*3+(rho*h**3/12.,)*2+(0.,)))
    np.testing.assert_allclose(matrix, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(matrix[5::6, :], 0.)
    np.testing.assert_array_equal(matrix[:, 5::6], 0.)
    changed = matrix.copy()
    changed[0, 0] = 0.
    assert physical_mass(VERTICES, density_kg_m3=rho, thickness_m=h)[0, 0] > 0.


def test_mass_matches_independent_resolved_thickness_particle_work():
    rho, h = 7850., .018
    # Affine in-plane mapping; a different 4x4x3 quadrature integrates the
    # physical volume velocities directly, without producer Jacobians/blocks.
    q = np.random.default_rng(92912).normal(size=(4, 6))
    points, weights = np.polynomial.legendre.leggauss(4)
    depth, depth_weights = np.polynomial.legendre.leggauss(3)
    direct = 0.
    normal = np.array((0., 0., 1.))
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            shape = np.array(((1-xi)*(1-eta), (1+xi)*(1-eta),
                              (1+xi)*(1+eta), (1-xi)*(1+eta)))/4.
            velocity, angular = shape@q[:, :3], shape@q[:, 3:]
            for z, wz in zip(depth, depth_weights):
                physical = velocity+(h*z/2.)*np.cross(angular, normal)
                direct += rho*1.2*.8*h*wx*wy*wz/8.*(physical@physical)
    matrix = physical_mass(VERTICES, density_kg_m3=rho, thickness_m=h)
    assert q.ravel()@matrix@q.ravel() == pytest.approx(direct, rel=2e-14)


def test_mass_preserves_oblique_frame_director_reversal_and_rigid_box_work():
    rho, h = 7850., .018
    rotation = Rotation.from_rotvec((.5, -.7, .3)).as_matrix()
    vertices = VERTICES@rotation.T+(.2, -.3, .4)
    normal = rotation[:, 2]
    matrix = physical_mass(vertices, density_kg_m3=rho, thickness_m=h,
                           reference_normal=normal)
    transform = np.kron(np.eye(8), rotation)
    original = physical_mass(VERTICES, density_kg_m3=rho, thickness_m=h)
    np.testing.assert_allclose(matrix, transform@original@transform.T,
                               rtol=1e-12, atol=2e-13)
    reversed_mass = physical_mass(vertices, density_kg_m3=rho, thickness_m=h,
                                  reference_normal=-normal)
    np.testing.assert_array_equal(reversed_mass, matrix)
    velocity, angular = np.array((.1, -.2, .3)), np.array((.3, .2, -.1))
    q = np.zeros((4, 6))
    q[:, :3] = velocity+np.cross(angular, vertices-vertices.mean(axis=0))
    q[:, 3:] = angular
    mass = rho*h*1.2*.8
    inertia_local = mass/12.*np.diag((.8**2+h**2, 1.2**2+h**2, 1.2**2+.8**2))
    expected = mass*(velocity@velocity)+angular@(rotation@inertia_local@rotation.T)@angular
    assert q.ravel()@matrix@q.ravel() == pytest.approx(expected, rel=2e-13)
    spin = np.zeros((4, 6)); spin[:, 3:] = normal
    assert np.linalg.norm(matrix@spin.ravel()) <= 1e-14*np.linalg.norm(matrix)


@pytest.mark.parametrize("change", ("density", "thickness", "normal", "fold", "warped"))
def test_physical_mass_rejects_invalid_definitions(change):
    vertices = VERTICES.copy()
    kwargs = dict(density_kg_m3=7850., thickness_m=.018)
    if change == "density": kwargs["density_kg_m3"] = True
    if change == "thickness": kwargs["thickness_m"] = 0.
    if change == "normal": kwargs["reference_normal"] = (1., 0., 0.)
    if change == "fold": vertices[[1, 2]] = vertices[[2, 1]]
    if change == "warped": vertices[2, 2] = .02
    with pytest.raises(ValueError):
        physical_mass(vertices, **kwargs)


def test_folded_corner_cannot_hide_between_gauss_locations():
    vertices = np.array(((0.,0.,0.), (1.,0.,0.), (.4,.4,0.), (0.,1.,0.)))
    with pytest.raises(ValueError, match="nonfolded"):
        physical_mass(vertices, density_kg_m3=7850., thickness_m=.018)


def test_admitted_normal_roundoff_is_canonicalized_without_added_inertia():
    matrix = physical_mass(VERTICES, density_kg_m3=7850., thickness_m=.018,
                           reference_normal=(0., 0., 1.+5e-11))
    expected = physical_mass(VERTICES, density_kg_m3=7850., thickness_m=.018)
    np.testing.assert_array_equal(matrix, expected)


def test_assembly_rejects_lost_rotary_inertia_even_when_translation_survives():
    vertices = np.array(((0., 0., 0.), (1e-20, 0., 0.),
                         (1e-20, 1e-20, 0.), (0., 1e-20, 0.)))
    with pytest.raises(ValueError, match="mass assembly is not representable"):
        physical_mass(vertices, density_kg_m3=1., thickness_m=1e-100)


def test_section_rejects_lost_tilted_rotary_diagonal():
    vertices = np.array(((0., 0., 0.), (1., 0., -1e-12),
                         (1., 1., -1e-12), (0., 1., 0.)))
    with pytest.raises(ValueError, match="plate inertia is not representable"):
        physical_mass(vertices, density_kg_m3=1., thickness_m=1e-100,
                      reference_normal=(1e-12, 0., 1.))
