"""Original bilinear target and independent simplex-moment/derivative controls."""
import math

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.overlap import PlanarFace
from anyfesolver.general_contact.q4_target_overlap import q4_target_overlap_integrals
from anyfesolver.general_contact.q4_s3_overlap_derivatives import UnsupportedQ4S3Branch
from anyfesolver.general_contact.s3_q4_overlap_derivatives import s3_q4_overlap_derivatives


CONTROLS = dict(support_tolerance=1e-12, weighted_gap_tolerance=1e-13,
    support_gradient_tolerance=1e-10, weighted_gap_gradient_tolerance=1e-11,
    support_hessian_tolerance=1e-8, weighted_gap_hessian_tolerance=1e-9,
    maximum_triangles=200)


def positions(*, partial=False):
    source=np.array(((.1,.1,.001),(.8,.2,.001),(.2,.7,.001)))
    if partial:
        source[1,0]=1.3
    target=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,.0002),(0.,1.,0.)))
    return source,target


def test_original_bilinear_target_matches_independent_simplex_moments():
    source,target=positions()
    result=s3_q4_overlap_derivatives(source,source,target,**CONTROLS)
    area=.5*np.linalg.norm(np.cross(source[1]-source[0],source[2]-source[0]))
    expected=[]
    for i in range(3):
        xy_integral=0.
        for j in range(3):
            for k in range(3):
                multiplicities=[(i,j,k).count(node) for node in range(3)]
                moment=area*math.prod(math.factorial(n) for n in multiplicities)/60
                xy_integral+=source[j,0]*source[k,1]*moment
        expected.append(.001*area/3-.0002*xy_integral)
    np.testing.assert_allclose(result.reference_support,area/3,rtol=2e-13,atol=1e-15)
    np.testing.assert_allclose(result.reference_weighted_gap,expected,rtol=2e-13,atol=1e-17)
    assert all(error<=control for error,control in zip(result.measured_errors,
        (1e-12,1e-13,1e-10,1e-11,1e-8,1e-9)))


@pytest.mark.parametrize("partial",(False,True))
def test_scalar_projection_parity_and_complete_coordinate_directional_derivatives(partial):
    source,target=positions(partial=partial)
    result=s3_q4_overlap_derivatives(source,source,target,**CONTROLS)
    scalar=q4_target_overlap_integrals(PlanarFace((1,1),source),target,
        reference_area_ratio=1.,weighted_gap_tolerance=1e-13,maximum_triangles=200)
    np.testing.assert_allclose(result.reference_support,scalar.support,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(result.reference_weighted_gap,scalar.weighted_gap,rtol=1e-11,atol=1e-16)
    direction=np.random.default_rng(510+int(partial)).normal(size=21)
    direction/=np.linalg.norm(direction)
    coordinates=np.concatenate((source,target)).reshape(-1)
    for step in (1e-6,3e-7):
        plus=(coordinates+step*direction).reshape(7,3)
        minus=(coordinates-step*direction).reshape(7,3)
        a=s3_q4_overlap_derivatives(source,plus[:3],plus[3:],**CONTROLS)
        b=s3_q4_overlap_derivatives(source,minus[:3],minus[3:],**CONTROLS)
        for value,gradient,hessian in (
            ("reference_support","support_gradient","support_hessian"),
            ("reference_weighted_gap","weighted_gap_gradient","weighted_gap_hessian")):
            for analytic,difference in (
                (getattr(result,gradient)@direction,(getattr(a,value)-getattr(b,value))/(2*step)),
                (getattr(result,hessian)@direction,(getattr(a,gradient)-getattr(b,gradient))/(2*step))):
                if value=="reference_support" and not partial:
                    # A fully covered P1 source has exactly A_ref/3 support,
                    # independent of its current coordinates: derivatives zero.
                    assert np.linalg.norm(analytic)<1e-12
                    assert np.linalg.norm(difference)<1e-9
                else:
                    assert np.linalg.norm(analytic-difference)/np.linalg.norm(analytic)<1e-5


def test_common_rigid_transform_and_both_body_translation():
    source,target=positions(partial=True)
    result=s3_q4_overlap_derivatives(source,source,target,**CONTROLS)
    rotation=Rotation.from_rotvec((.3,-.2,.1)).as_matrix()
    shift=np.array((1.2,-.7,.4))
    moved_source=source@rotation.T+shift
    moved_target=target@rotation.T+shift
    moved=s3_q4_overlap_derivatives(moved_source,moved_source,moved_target,**CONTROLS)
    transform=block_diag(*([rotation]*7))
    np.testing.assert_allclose(result.reference_support,moved.reference_support,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(result.reference_weighted_gap,moved.reference_weighted_gap,rtol=1e-10,atol=1e-16)
    for name in ("support_gradient","weighted_gap_gradient"):
        np.testing.assert_allclose(getattr(moved,name),getattr(result,name)@transform.T,rtol=1e-9,atol=1e-12)
    for name in ("support_hessian","weighted_gap_hessian"):
        expected=np.einsum('ij,njk,lk->nil',transform,getattr(result,name),transform)
        np.testing.assert_allclose(getattr(moved,name),expected,rtol=1e-8,atol=1e-11)
    for axis in range(3):
        direction=np.tile(np.eye(3)[axis],7)
        np.testing.assert_allclose(result.support_gradient@direction,0.,atol=1e-12)
        np.testing.assert_allclose(result.weighted_gap_gradient@direction,0.,atol=1e-12)


def test_explicit_resource_and_geometry_failures():
    source,target=positions()
    with pytest.raises(RuntimeError,match="budget exhausted"):
        s3_q4_overlap_derivatives(source,source,target,**dict(CONTROLS,maximum_triangles=1))
    with pytest.raises(ValueError,match="positive dimensioned"):
        s3_q4_overlap_derivatives(source,source,target,**dict(CONTROLS,support_tolerance=0.))
    with pytest.raises(UnsupportedQ4S3Branch,match="reversed"):
        s3_q4_overlap_derivatives(source,source,target[::-1],**CONTROLS)


def test_clipping_resolution_is_retained_without_dropping_pressure_supports():
    source,target=positions(partial=True)
    result=s3_q4_overlap_derivatives(source,source,target,**CONTROLS)
    assert 0.<result.overlap_area_resolution_m2<result.overlap_area
    assert result.reference_support.shape==(3,) and np.all(result.reference_support>0.)
    # An intentionally unresolved input uncertainty fails the whole candidate;
    # it does not convert a positive overlap to zero or erase a pressure mode.
    with pytest.raises(UnsupportedQ4S3Branch):
        s3_q4_overlap_derivatives(source,source,target,**dict(CONTROLS,relative_branch_tolerance=.1))
