"""Independent reference moments and conservative supplied-path containment."""
from fractions import Fraction as Q
import numpy as np
import pytest
from anyfesolver.general_contact.s3_pressure_hermite_bounds import (
    _ROOT,_prepare_overlap,_factor,s3_pressure_hermite_bounds)
from anyfesolver.general_contact.s3_pressure_polygon_bounds import _integral,_signed_integral,_polygon_cell

SOURCE=np.array([[0.,0.,0.],[.6,0.,0.],[0.,.4,0.]])
TARGET=np.array([[-.2,-.2,0.],[.3,-.2,0.],[.3,.9,0.]])
ZERO=np.zeros((3,3))


def oracle(a,b,reference):
    # Direct planar solve/float clipping; no production projection or integrator.
    normal=np.cross(a[1]-a[0],a[2]-a[0]);normal/=np.linalg.norm(normal)
    solved=np.linalg.solve(np.column_stack((b[1]-b[0],b[2]-b[0],-normal)),(a-b[0]).T).T
    gap=-solved[:,2];uv=solved[:,:2]
    coefficients=np.column_stack((1-uv.sum(axis=1),uv))
    polygon=list(np.eye(3))
    for edge in coefficients.T:
        clipped=[]
        for a0,a1 in zip(polygon,polygon[1:]+polygon[:1]):
            f0,f1=a0@edge,a1@edge
            if f0>=0:clipped.append(a0)
            if (f0<0 and f1>0) or (f0>0 and f1<0):
                clipped.append(a0+(a1-a0)*f0/(f0-f1))
        polygon=clipped
    area=.5*np.linalg.norm(np.cross(reference[1]-reference[0],reference[2]-reference[0]))
    S=np.zeros(3);W=np.zeros(3)
    for k in range(1,len(polygon)-1):
        triangle=np.array([polygon[0],polygon[k],polygon[k+1]])
        jac=np.linalg.det(triangle[1:,1:]-triangle[0,1:]);subarea=area*abs(jac)
        g=triangle@gap
        S+=subarea*triangle.sum(axis=0)/3
        W+=subarea*(triangle.sum(axis=0)*g.sum()+triangle.T@g)/12
    return S,W


def test_original_basis_exact_moments_and_zero_factor_branch():
    for j in range(3):
        coefficients=tuple(Q(int(i==j)) for i in range(3))
        values=_integral(_ROOT,coefficients)
        assert values==tuple(Q(1,6) if i==j else Q(1,12) for i in range(3))
    for positive in (False,True):
        shift=Q(1,100) if positive else Q(-1,100)
        assert _signed_integral(_ROOT,(Q(0),)*3,(Q(1,2),Q(3,2)),shift,
            lower=True,positive=positive)==(shift/3,)*3


@pytest.mark.parametrize('seed',range(6))
def test_joint_support_and_gap_bounds_contain_independent_moving_geometry(seed):
    rng=np.random.default_rng(seed)
    a0=SOURCE+rng.uniform(-.018,.018,(3,3));a1=SOURCE+rng.uniform(-.018,.018,(3,3))
    b0=TARGET+rng.uniform(-.025,.025,(3,3));b1=TARGET+rng.uniform(-.025,.025,(3,3))
    av0,av1,bv0,bv1=(rng.uniform(-.08,.08,(3,3)) for _ in range(4));dt=.1;tau=Q(1,10000)
    prepared=_prepare_overlap(SOURCE,TARGET,((0,1,2),),a0,a1,b0,b1,av0,av1,bv0,bv1,Q(dt),3)
    face=prepared[0];factor,status=_factor(face);assert status is None
    supports,opens,closed,_=_polygon_cell(face,_ROOT,tau,factor)
    def position(p0,p1,v0,v1,t):
        return (2*t**3-3*t*t+1)*p0+(-2*t**3+3*t*t)*p1+(t**3-2*t*t+t)*dt*v0+(t**3-t*t)*dt*v1
    for t in np.linspace(0,1,31):
        S,W=oracle(position(a0,a1,av0,av1,t),position(b0,b1,bv0,bv1,t),SOURCE)
        for i in range(3):
            assert float(supports[i][0])-1.e-15<=S[i]<=float(supports[i][1])+1.e-15
            assert float(opens[i][0])-1.e-15<=W[i]+float(tau)*S[i]<=float(opens[i][1])+1.e-15
            assert float(closed[i][0])-1.e-15<=W[i]-float(tau)*S[i]<=float(closed[i][1])+1.e-15


def proof(a,target,active,**controls):
    return s3_pressure_hermite_bounds(SOURCE,target,((0,1,2),),a,a,target,target,
        ZERO,ZERO,ZERO,ZERO,1.,active=np.array(active,dtype=bool),gap_tolerance_m=1.e-10,
        _polygon_bounds=True,**controls)


def test_partial_constant_geometry_preserves_physical_reference_measure():
    a=SOURCE.copy();a[:,2]=.003
    result=proof(a,TARGET,[False]*3)
    assert result.success
    S,W=oracle(a,TARGET,SOURCE)
    for i in range(3):
        assert result.support_lower[i]-1.e-16<=S[i]<=result.support_upper[i]+1.e-16
        assert result.open_joint_lower[i]<=W[i]+1.e-10*S[i]+1.e-16
        assert result.closed_joint_upper[i]>=W[i]-1.e-10*S[i]-1.e-16


def test_tiny_positive_support_rejects_wrong_phase_and_absent_support_is_zero():
    tiny=np.array([[0.,0.,.001],[1.e-12,0.,.001],[0.,1.e-12,.001]])
    result=proof(SOURCE,tiny,[False]*3)
    assert not result.success and result.status=='pressure_phase_violation'
    absent=proof(SOURCE,TARGET+[20.,0.,.001],[True]*3)
    assert absent.success and absent.support_lower==absent.support_upper==(0.,)*3


def test_hidden_weak_phase_is_not_certified_from_open_endpoints():
    target=np.array([[-2.,-2.,-.001],[5.,-2.,-.001],[-2.,5.,-.001]])
    a=SOURCE;v0=ZERO.copy();v1=ZERO.copy();v0[:,2]=-.012;v1[:,2]=.012
    result=s3_pressure_hermite_bounds(SOURCE,target,((0,1,2),),a,a,target,target,v0,v1,ZERO,ZERO,1.,
        active=np.zeros(3,dtype=bool),gap_tolerance_m=1.e-10,_polygon_bounds=True)
    assert not result.success and result.status=='pressure_phase_violation'


def test_limits_cancellation_and_invalid_selection_remain_explicit():
    a=SOURCE.copy();a[:,2]=.003
    result=proof(a,TARGET,[False]*3,maximum_spatial_cells=1)
    assert not result.success and result.status=='spatial_budget_exhausted'
    cancelled=proof(a,TARGET,[False]*3,cancel=lambda:True)
    assert not cancelled.success and cancelled.status=='cancelled'
    with pytest.raises(ValueError,match='boolean'):
        s3_pressure_hermite_bounds(SOURCE,TARGET,((0,1,2),),a,a,TARGET,TARGET,
            ZERO,ZERO,ZERO,ZERO,1.,active=np.zeros(3,dtype=bool),gap_tolerance_m=1.e-10,_polygon_bounds=1)
