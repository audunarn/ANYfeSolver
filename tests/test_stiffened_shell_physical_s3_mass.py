"""Independent barycentric/thickness kinetic work for the declared S3 field."""
import numpy as np
import pytest
from anyfesolver._stiffened_shell_physical_s3_mass import physical_mass


def test_barycentric_thickness_kinetic_work_and_exact_normal_spin():
    x = np.array(((0., 0., 0.), (2., 0., 0.), (.2, 3., 0.)))
    rho, h = 6., .2
    mass = physical_mass(x, density_kg_m3=rho, thickness_m=h)
    q = np.arange(18, dtype=float).reshape(3, 6)/7.
    work = 0.
    # Degree2 triangular rule and degree2 thickness Gauss rule, independent
    # direct integration of v(z), with no tensor-product mass construction.
    for shape in ((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)):
        v, omega = np.asarray(shape)@q[:, :3], np.asarray(shape)@q[:, 3:]
        for z in (-h/(2*np.sqrt(3)), h/(2*np.sqrt(3))):
            velocity = v+z*np.cross(omega, (0., 0., 1.))
            work += rho*3./3.*h/2.*(velocity@velocity)
    np.testing.assert_allclose(q.ravel()@mass@q.ravel(), work, rtol=1e-14)
    assert not np.any(mass[[5, 11, 17], :])
    assert np.linalg.matrix_rank(mass) == 15
    np.testing.assert_array_equal(mass, mass.T)
    np.testing.assert_array_equal(mass, physical_mass(x, density_kg_m3=rho,
        thickness_m=h, reference_normal=(0., 0., -1.)))
    for permutation in ((1,2,0),(2,1,0)):
        slots=np.array([6*node+axis for node in permutation for axis in range(6)])
        permuted=physical_mass(x[list(permutation)],density_kg_m3=rho,thickness_m=h)
        np.testing.assert_allclose(permuted,mass[np.ix_(slots,slots)],rtol=1e-14,atol=0.)
    for axis in (0,3,5):
        constant=np.zeros((3,6));constant[:,axis]=1.
        expected = rho*3.*(h if axis == 0 else h**3/12 if axis == 3 else 0.)
        np.testing.assert_allclose(constant.ravel()@mass@constant.ravel(),expected,
            rtol=1e-14,atol=0.)


def test_covariance_and_rigid_kinetic_work():
    x = np.array(((0., 0., 0.), (1., 0., 0.), (0., 2., 0.)))
    rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
    mass = physical_mass(x, density_kg_m3=2., thickness_m=.3)
    transformed = physical_mass(x@rotation.T, density_kg_m3=2., thickness_m=.3,
        reference_normal=rotation[:, 2])
    transform = np.kron(np.eye(6), rotation)
    np.testing.assert_allclose(transformed, transform@mass@transform.T, atol=0., rtol=1e-14)
    v = np.tile((2., 3., 4., 0., 0., 0.), 3)
    np.testing.assert_allclose(v@mass@v, 2.*.3*1.*29., rtol=1e-14)


@pytest.mark.parametrize("vertices,rho,h,n", [
    (np.zeros((3,3)), 1., 1., (0.,0.,1.)),
    (((0,0,0),(1,0,0),(0,1,.1)), 1., 1., (0.,0.,1.)),
    (((0,0,0),(1,0,0),(0,1,0)), 0., 1., (0.,0.,1.)),
    (((0,0,0),(1,0,0),(0,1,0)), 1., 0., (0.,0.,1.)),
    (((0,0,0),(1,0,0),(0,1,0)), 1., 1., (0.,0.,2.)),
    (((0,0,0),(1,0,0),(0,1,0)), 1e-300, 1e-300, (0.,0.,1.)),
])
def test_refuses_nonphysical_or_unrepresentable_mass(vertices,rho,h,n):
    with pytest.raises(ValueError):
        physical_mass(vertices,density_kg_m3=rho,thickness_m=h,reference_normal=n)
