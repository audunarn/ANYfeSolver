"""G1a material/complementary checks; independent station KKT reconstruction."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_g1_elastic import ElasticSection
from anyfesolver._ge_beam3_g1_operator import ElasticOperator, schur, local_solve
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_p5.algebra import rotation


def section(dense=False):
    if dense:
        L = np.diag([10., 9., 8., 7., 6., 5.])+np.tril(np.full((6, 6), .2), -1)
        return ElasticSection(L @ L.T, "dense")
    return ElasticSection.isotropic(E=100., G=40., area=1., Iy=.2, Iz=.3,
                                    J=.1, shear_y=.8, shear_z=.7)


def reference(start=0., roll=0.):
    return Reference(np.array([[start, 0., 0.], [start+.5, 0., 0.], [start+1., 0., 0.]]),
                     np.tile(rotation([roll, 0., 0.]), (3, 1, 1)))


def test_exact_elastic_modes_and_empty_history():
    law = section()
    C = np.diag([100., 32., 28., 4., 20., 30.])
    for e in np.eye(6):
        r = law.response(e)
        np.testing.assert_array_equal(r["resultants"], C @ e)
        assert r["potential"] == e @ C @ e/2 and r["history"] == ()
    # No hidden yield limit: the same constant law applies at large strain.
    np.testing.assert_array_equal(law.response(np.ones(6)*1e6)["tangent"], C)
    with pytest.raises(ValueError): law.response(np.zeros(6), history=(0.,))
    with pytest.raises(ValueError): law.stiffness[0, 0] = 0.


@pytest.mark.parametrize("dense", [False, True])
def test_independent_full_station_kkt_and_recovery(dense):
    ref = reference(roll=.37); law = section(dense); op = ElasticOperator(ref, law)
    # Reconstruct from quadrature and C; do not use producer station maps/lift/S.
    n = 8; W = np.zeros((6*n, 6*n)); B = np.zeros((6+3*n, 6*n)); D = np.zeros((6+3*n, 18))
    points, weights = np.polynomial.legendre.leggauss(4)
    compliance = np.linalg.solve(law.stiffness, np.eye(6))
    for c in range(2):
        for j in range(4):
            k = 4*c+j; t = (points[j]+1)/2; xi = c-1+t
            weight = weights[j]*ref.jacobian(xi)/2
            W[6*k:6*k+6, 6*k:6*k+6] = weight*compliance
            B[3*c:3*c+3, 6*k:6*k+3] = weight*ref.frame(xi)/ref.jacobian(xi)
            B[6+3*k:9+3*k, 6*k+3:6*k+6] = np.eye(3)
            D[6+3*k:9+3*k, 6+6*c:9+6*c] = (1-t)*np.eye(3)
            D[6+3*k:9+3*k, 9+6*c:12+6*c] = t*np.eye(3)
    D[:6, :6] = np.eye(6)
    A = np.block([[W, B.T], [B, np.zeros((len(B), len(B)))]])
    lift = np.linalg.solve(A, np.vstack([np.zeros((6*n, 18)), D]))[:6*n]
    expected = lift.T @ W @ lift
    np.testing.assert_allclose(op.cell.compliance, expected, atol=1e-11, rtol=1e-11)
    p = np.linspace(-.03, .04, 18)
    v, g, h = op.cell.response(p)
    assert abs(v-p @ expected @ p/2) < 1e-11
    np.testing.assert_allclose(g, expected @ p, atol=1e-11)
    for i, row in enumerate(op.cell.recover(p)):
        np.testing.assert_allclose(row["resultants"], (lift @ p)[6*i:6*i+6], atol=1e-11)
        np.testing.assert_allclose(row["strain"], compliance @ row["resultants"], atol=1e-11)


def test_schur_including_nonzero_internal_residual():
    op = ElasticOperator(reference(), section(True)); ref = op.reference
    data = op.evaluate(ref.coordinates, np.zeros((3, 3)), ref.nodal_triads,
                       np.tile(np.eye(3), (2, 1, 1)), np.linspace(-.01, .02, 18),
                       line=(.01, -.02, .005), couple=(.001, .002, -.001))
    r, K, lift, correction = schur(data["residual"], data["jacobian"])
    H = data["jacobian"]; full = data["residual"]; dq = np.linspace(-.001, .002, 18)
    da = correction+lift @ dq
    np.testing.assert_allclose(H[18:, :18] @ dq+H[18:, 18:] @ da+full[18:], 0., atol=1e-11)
    np.testing.assert_allclose(H[:18, :18] @ dq+H[:18, 18:] @ da+full[:18], r+K @ dq, atol=1e-11)


def test_operator_rigid_objectivity():
    ref = reference(roll=.37); op = ElasticOperator(ref, section(True))
    Q = rotation([.7, -.3, .5]); translation = np.array([.2, -.1, .3])
    base = op.evaluate(ref.coordinates, np.zeros((3, 3)), ref.nodal_triads,
                       np.tile(np.eye(3), (2, 1, 1)), np.zeros(18))
    value = op.evaluate(ref.coordinates @ Q.T+translation, np.zeros((3, 3)), Q @ ref.nodal_triads,
                        np.tile(Q, (2, 1, 1)), np.zeros(18))
    assert abs(value["potential"]-base["potential"]) < 1e-11
    assert np.linalg.norm(value["residual"]) < 1e-11


def test_external_section_capture_is_immutable():
    class External:
        name = "external SPD"
        C = section(True).stiffness.copy()
        def generalized_stiffness_matrix(self): return self.C
    original = External(); law = ElasticSection.capture(original)
    identity = law.identity; original.C[0, 0] *= 2
    assert law.identity == identity


def test_mutation_and_invalid_material_rejected():
    with pytest.raises(ValueError): ElasticSection(np.eye(5))
    with pytest.raises(ValueError): ElasticSection(np.full((6, 6), np.nan))
    with pytest.raises(np.linalg.LinAlgError): ElasticSection(np.ones((6, 6)))
    op = ElasticOperator(reference(), section()); op.cell.compliance = np.eye(18)
    with pytest.raises(ValueError): op.guard()


def test_reversal_work_and_condensed_operator():
    ref = reference(roll=.37); law = section(True)
    D = np.diag([-1., 1., -1.]); T = np.diag([1., -1., 1., 1., -1., 1.])
    reversed_ref = Reference(ref.coordinates[::-1], ref.nodal_triads[::-1] @ D)
    original = ElasticOperator(ref, law)
    reverse = ElasticOperator(reversed_ref, ElasticSection(T @ law.stiffness @ T.T, "reversed"))
    perturbation = np.array([[0., 0., 0.], [.0001, .0002, -.0001], [.0003, -.0002, .0004]])
    x = ref.coordinates+perturbation
    frames = np.array([rotation(v) @ R for v, R in zip([[0., 0., 0.], [.002, -.001, .003], [-.001, .002, .001]], ref.nodal_triads)])
    args = dict(line=np.zeros(3), couple=np.zeros(3))
    a = local_solve(original, x, np.zeros((3, 3)), frames,
                    np.tile(np.eye(3), (2, 1, 1)), np.zeros(18), **args)
    b = local_solve(reverse, x[::-1], np.zeros((3, 3)), frames[::-1] @ D,
                    np.tile(np.eye(3), (2, 1, 1)), np.zeros(18), **args)
    perm = np.arange(18).reshape(3, 6)[::-1].ravel()
    np.testing.assert_allclose(a["residual"][perm], b["residual"], atol=1e-11, rtol=1e-11)
    assert np.linalg.norm(a["tangent"][np.ix_(perm, perm)]-b["tangent"]) / max(1., np.linalg.norm(a["tangent"])) < 1e-11
    assert abs(a["full"]["potential"]-b["full"]["potential"]) < 1e-11


def test_isotropic_adapter_boolean_and_overflow_rejection():
    with pytest.raises(ValueError):
        ElasticSection.isotropic(E=True, G=40., area=1., Iy=.2, Iz=.3, J=.1, shear_y=.8, shear_z=.7)
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        section().response(np.ones(6)*1e200)


def test_unhealthy_factorization_is_not_warning_only():
    from anyfesolver._ge_beam3_g1_elastic import solve
    with pytest.raises(np.linalg.LinAlgError):
        solve(np.diag([1., 1e-30]), np.ones(2))
