"""Private work-law feasibility checks; these do not admit finite S3 pressure."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._stiffened_shell_ge_experimental import ShellPanel, _shell_reference
from anyfesolver._stiffened_shell_s3_dead_work_candidate import evaluate


def fixture(pressure=2.5):
    xyz = np.array(((0., 0., 0.), (2., 0., 0.), (.4, 1., 0.)))
    panel = ShellPanel((1, 2, 3), tuple(map(tuple, xyz)), (),
                      .1, 2.e9, .3, 3., ((1, 2, 3),))
    model, _, _, _, parts = _shell_reference(panel)
    element = parts[0][1]
    native = element.compute_dead_transverse_pressure_load(model.mesh, pressure)
    return xyz, np.array((0., 0., 1.)), native


def state():
    return np.array(((.03, -.02, .08, .1, -.2, .3),
                     (-.04, .03, .12, -.2, .14, -.1),
                     (.06, .02, -.07, .07, .05, -.13))).ravel()


def independent_work(xyz, normal, native, q):
    """Value-only SVD reconstruction, independent of quaternion/Jet2 derivatives."""
    nodes = q.reshape(3, 6)
    current = xyz+nodes[:, :3]
    u, _, vt = np.linalg.svd((current-current.mean(0)).T@(xyz-xyz.mean(0)))
    fitted = u@np.diag((1., 1., np.linalg.det(u@vt)))@vt
    relative = np.array([director_tilt(normal, fitted.T@Rotation.from_rotvec(a).as_matrix()@normal)
                         for a in nodes[:, 3:]])
    f = native.reshape(3, 6)
    return np.sum(f[:, :3]*nodes[:, :3])+(normal@fitted@normal)*np.sum(f[:, 3:]*relative)


def director_tilt(normal, director):
    cross = np.cross(normal, director)
    sine = np.linalg.norm(cross)
    return cross if sine < 1e-14 else np.arctan2(sine, normal@director)*cross/sine


def quadrature_work(xyz, normal, pressure, q):
    """Reconstruct the proposed surface; no native pressure vector is used."""
    current = xyz+q.reshape(3, 6)[:, :3]
    u, _, vt = np.linalg.svd((current-current.mean(0)).T@(xyz-xyz.mean(0)))
    fitted = u@np.diag((1., 1., np.linalg.det(u@vt)))@vt
    x_axis = (xyz[1]-xyz[0])/np.linalg.norm(xyz[1]-xyz[0])
    frame = np.column_stack((x_axis, np.cross(normal, x_axis), normal))
    xy = (xyz-xyz[0])@frame[:, :2]
    a, b = np.empty(3), np.empty(3)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        a[i], b[i] = xy[k, 0]-xy[j, 0], xy[j, 1]-xy[k, 1]
    rho = np.array([director_tilt(normal, fitted.T@Rotation.from_rotvec(v).as_matrix()@normal)
                    for v in q.reshape(3, 6)[:, 3:]])@frame
    area = np.linalg.norm(np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0]))/2
    result = 0.
    for xi, eta in ((1/6, 1/6), (2/3, 1/6), (1/6, 2/3)):
        shape = np.array((1-xi-eta, xi, eta))
        height = 0.
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            l_value = .5*shape[i]*(b[k]*shape[j]-b[j]*shape[k])
            m_value = .5*shape[i]*(a[j]*shape[k]-a[k]*shape[j])
            height += -l_value*rho[i, 0]+m_value*rho[i, 1]
        surface = shape@current+(fitted@normal)*height
        result += pressure*area/3*(normal@(surface-shape@xyz))
    return result


@pytest.mark.parametrize('pressure', [2.5, -2.5])
def test_reference_native_covector_and_independent_finite_work(pressure):
    xyz, normal, native = fixture(pressure)
    work, gradient, hessian, force, _ = evaluate(xyz, normal, native, np.zeros(18))
    assert abs(work) < 1e-14
    np.testing.assert_allclose(gradient, native, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(force, native, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-13)
    q = state()
    assert evaluate(xyz, normal, native, q)[0] == pytest.approx(
        independent_work(xyz, normal, native, q), rel=1e-12, abs=1e-14)
    assert evaluate(xyz, normal, native, q)[0] == pytest.approx(
        quadrature_work(xyz, normal, pressure, q), rel=1e-12, abs=1e-14)


def test_actual_coordinate_derivatives_and_spatial_row_conversion():
    xyz, normal, native = fixture()
    q = state()
    work, gradient, hessian, force, tangent = evaluate(xyz, normal, native, q)
    direction = np.linspace(-.3, .4, 18)
    for step in (1e-4, 3e-5, 1e-5):
        plus = evaluate(xyz, normal, native, q+step*direction)
        minus = evaluate(xyz, normal, native, q-step*direction)
        assert (independent_work(xyz, normal, native, q+step*direction)
                -independent_work(xyz, normal, native, q-step*direction))/(2*step) == pytest.approx(
                    gradient@direction, rel=1e-6, abs=1e-10)
        for numerical, expected in (((plus[1]-minus[1])/(2*step), hessian@direction),
                                     ((plus[3]-minus[3])/(2*step), tangent@direction)):
            assert np.linalg.norm(numerical-expected)/np.linalg.norm(expected) < 1e-6
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-13)
    assert np.linalg.norm(tangent-tangent.T) > 1e-5


def test_rigid_motion_retains_dead_work_and_physical_resultants():
    xyz, normal, native = fixture()
    angle = np.array((.4, -.3, .2))
    rotation = Rotation.from_rotvec(angle).as_matrix()
    current = xyz@rotation.T+np.array((.2, -.1, .4))
    q = np.column_stack((current-xyz, np.tile(angle, (3, 1)))).ravel()
    work, _, _, force, _ = evaluate(xyz, normal, native, q)
    expected = native.reshape(3, 6)[:, :3]
    assert work == pytest.approx(np.sum(expected*(current-xyz)), abs=1e-13)
    nodal = force.reshape(3, 6)
    np.testing.assert_allclose(nodal[:, :3].sum(0), expected.sum(0), atol=1e-13)
    np.testing.assert_allclose((np.cross(current, nodal[:, :3])+nodal[:, 3:]).sum(0),
                               np.cross(current, expected).sum(0), atol=1e-13)


def test_chart_rebase_preserves_work_and_spatial_operator():
    xyz, normal, native = fixture()
    q = state()
    original = evaluate(xyz, normal, native, q)
    anchors = Rotation.from_rotvec(q.reshape(3, 6)[:, 3:]).as_matrix()
    rebased = q.copy().reshape(3, 6)
    rebased[:, 3:] = 0.
    result = evaluate(xyz, normal, native, rebased.ravel(), anchors=anchors)
    assert result[0] == pytest.approx(original[0], abs=1e-13)
    np.testing.assert_allclose(result[3], original[3], atol=2e-13)
    chart = np.eye(18)
    for node in range(3):
        slots = slice(6*node+3, 6*node+6)
        chart[slots, slots] = _exp_terms(q[slots])[1]
    np.testing.assert_allclose(result[4]@chart, original[4], atol=2e-12)


def test_reference_frame_covariance_and_origin_shift():
    xyz, normal, native = fixture()
    q = state()
    a = Rotation.from_rotvec((.5, .3, -.2)).as_matrix()
    transform = np.kron(np.eye(6), a)
    original = evaluate(xyz, normal, native, q)
    moved = evaluate(xyz@a.T+np.array((3., -2., 1.)), a@normal,
                     transform@native, transform@q)
    assert moved[0] == pytest.approx(original[0], abs=1e-13)
    np.testing.assert_allclose(moved[3], transform@original[3], atol=2e-13)
    np.testing.assert_allclose(moved[4], transform@original[4]@transform.T, atol=2e-12)


def test_director_preserving_nodal_drill_has_no_pressure_work():
    xyz, normal, native = fixture()
    q = state()
    original = Rotation.from_rotvec(q.reshape(3, 6)[:, 3:]).as_matrix()
    spun = original.copy()
    spun[0] = original[0]@Rotation.from_rotvec(.4*normal).as_matrix()
    np.testing.assert_allclose(spun@normal, original@normal, atol=1e-15)
    changed = q.copy().reshape(3, 6)
    changed[:, 3:] = Rotation.from_matrix(spun).as_rotvec()
    before = evaluate(xyz, normal, native, q)[0]
    after = evaluate(xyz, normal, native, changed.ravel())[0]
    assert abs(after-before) <= 1e-10*max(1., abs(before), abs(after)), (before, after)
    moment = evaluate(xyz, normal, native, q)[3].reshape(3, 6)[:, 3:]
    np.testing.assert_allclose(np.einsum('ij,ij->i', moment, original@normal),
                               np.zeros(3), atol=1e-13)
    np.testing.assert_allclose(evaluate(xyz, normal, native, changed.ravel())[3],
                               evaluate(xyz, normal, native, q)[3], atol=1e-13)
    # Preserve the rejected full-log lift's counterexample explicitly. This
    # passing diagnostic assertion does NOT accept that failed physical law.
    rejected_before = evaluate(xyz, normal, native, q, director_only=False)[0]
    rejected_after = evaluate(xyz, normal, native, changed.ravel(), director_only=False)[0]
    assert abs(rejected_after-rejected_before) > .008


def test_uniform_director_tilt_keeps_constant_shear_admissible():
    xyz, normal, native = fixture()
    q = np.zeros((3, 6))
    q[:, 3:5] = (.35, -.2)
    # Flat surface, common tilted director: a finite shear configuration,
    # not a constraint forcing the director to the surface normal.
    assert abs(quadrature_work(xyz, normal, 2.5, q.ravel())) < 1e-14
    assert abs(evaluate(xyz, normal, native, q.ravel())[0]) < 1e-14


def test_antipodal_director_chart_rejected():
    xyz, normal, native = fixture()
    q = np.zeros(18)
    q[3] = np.pi
    with pytest.raises(ValueError):
        evaluate(xyz, normal, native, q)


@pytest.mark.parametrize('fraction', [.89, .901])
def test_inherited_relative_tilt_domain(fraction):
    xyz, normal, native = fixture()
    q = np.zeros(18)
    q[3] = fraction*np.pi
    if fraction < .9:
        assert np.isfinite(evaluate(xyz, normal, native, q)[0])
    else:
        with pytest.raises(ValueError):
            evaluate(xyz, normal, native, q)


def test_nonuniform_cylindrical_bending_surface_and_shear_refinement():
    """Manufactured kinematic screen, not an external shell-response oracle."""
    errors = []
    for length in (.2, .1, .05):
        xyz = length*np.array(((0., 0., 0.), (1., 0., 0.), (.2, 1., 0.)))
        current = np.column_stack((np.sin(xyz[:, 0]), xyz[:, 1], 1-np.cos(xyz[:, 0])))
        directors = np.column_stack((-np.sin(xyz[:, 0]), np.zeros(3), np.cos(xyz[:, 0])))
        u, _, vt = np.linalg.svd((current-current.mean(0)).T@(xyz-xyz.mean(0)))
        fitted = u@np.diag((1., 1., np.linalg.det(u@vt)))@vt
        normal = np.array((0., 0., 1.))
        rho = np.array([director_tilt(normal, fitted.T@d) for d in directors])
        a, b = np.empty(3), np.empty(3)
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            a[i], b[i] = xyz[k, 0]-xyz[j, 0], xyz[j, 1]-xyz[k, 1]
        gradients = np.array(((-1., -1.), (1., 0.), (0., 1.)))@np.linalg.inv(
            np.array((xyz[1, :2]-xyz[0, :2], xyz[2, :2]-xyz[0, :2])).T)
        np.testing.assert_allclose(xyz[:, :2].T@gradients, np.eye(2), atol=1e-15)
        position_errors, shear_errors = [], []
        for xi, eta in ((1/6, 1/6), (2/3, 1/6), (1/6, 2/3)):
            shape = np.array((1-xi-eta, xi, eta))
            height, dh = 0., np.zeros(2)
            for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                l = .5*shape[i]*(b[k]*shape[j]-b[j]*shape[k])
                m = .5*shape[i]*(a[j]*shape[k]-a[k]*shape[j])
                dl = .5*(gradients[i]*(b[k]*shape[j]-b[j]*shape[k])
                         +shape[i]*(b[k]*gradients[j]-b[j]*gradients[k]))
                dm = .5*(gradients[i]*(a[j]*shape[k]-a[k]*shape[j])
                         +shape[i]*(a[j]*gradients[k]-a[k]*gradients[j]))
                height += -l*rho[i, 0]+m*rho[i, 1]
                dh += -dl*rho[i, 0]+dm*rho[i, 1]
            surface = shape@current+(fitted@normal)*height
            tangents = current.T@gradients+np.outer(fitted@normal, dh)
            point = shape@xyz
            exact = np.array((np.sin(point[0]), point[1], 1-np.cos(point[0])))
            d = shape@directors
            d /= np.linalg.norm(d)
            position_errors.append(np.linalg.norm(surface-exact))
            shear_errors.append(np.linalg.norm(d@tangents))
        errors.append((max(position_errors), max(shear_errors)))
    # Geometry/interpolation consistency under fixed-radius refinement. This
    # does not impose director-normal equality in the numerical model.
    errors = np.asarray(errors)
    assert np.all(errors[1:] < .6*errors[:-1]), errors
