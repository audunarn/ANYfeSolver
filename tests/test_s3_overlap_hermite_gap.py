"""Independent affine-plane extrema and mapped cubic gap discriminators."""
import numpy as np
import pytest
from anyfesolver.general_contact.s3_overlap_hermite_gap import s3_overlap_hermite_gap_bounds
from anyfesolver.general_contact.s3_pressure_hermite_bounds import s3_pressure_hermite_bounds
from anyfesolver.general_contact.shell_pair_hermite_sweep import shell_pair_hermite_lower

SOURCE=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
TARGET=np.array([[-1.,0.,0.],[2.,0.,0.],[-1.,3.,0.]])


def inputs(source=SOURCE,target=TARGET):
    rs=SOURCE.copy();rt=target.copy();rt[:,2]=-.001
    z=np.zeros_like(source);zt=np.zeros_like(target)
    return (rs,rt,((0,1,2),),source,source,target,target,z,z,zt,zt,1.)


def proof(source=SOURCE,target=TARGET,**kwargs):
    return s3_overlap_hermite_gap_bounds(*inputs(source,target),
        **{'gap_tolerance_m':.001,**kwargs})


def test_remote_cartesian_corner_cannot_reject_matched_open_contact():
    target=TARGET.copy();target[:,2]=target[:,0]-1.1
    # Direct target plane is z=x-1.1. The exact projected source domain is the
    # entire source triangle; source-normal gap 1.1-x has minimum 0.1 m.
    old=shell_pair_hermite_lower(SOURCE,SOURCE,target,target,
        np.zeros_like(SOURCE),np.zeros_like(SOURCE),np.zeros_like(target),np.zeros_like(target),
        1.,gap_tolerance_m=.001)
    assert not old.success
    result=proof(target=target)
    assert result.success and result.status=='resolved'
    assert .09999999999999<result.lower_gap_m<.10000000000001


def test_partial_overlap_exact_halfplane_removes_only_remote_penetration():
    target=np.array([[-2.,-2.,0.],[.5,-2.,0.],[.5,3.,0.]])
    target[:,2]=target[:,0]-.3
    # Direct polygon is 0<=x<=.5, 0<=y<=1-x; g=.3-x. Minimum is -.2,
    # whereas the same source plane extended outside target reaches -.7.
    result=proof(target=target,gap_tolerance_m=.25)
    assert result.success
    np.testing.assert_allclose(result.lower_gap_m,-.2,rtol=0.,atol=1.e-14)
    rejected=proof(target=target,gap_tolerance_m=.19)
    assert not rejected.success and rejected.status=='gap_budget_violation'


def test_crossed_interior_is_retained_when_no_source_vertex_is_in_target():
    target=np.array([[.2,.2,.0001],[.4,.2,.0001],[.2,.4,.0001]])
    result=proof(target=target)
    assert result.success
    np.testing.assert_allclose(result.lower_gap_m,-.0001,rtol=0.,atol=1.e-18)
    assert not proof(target=target,gap_tolerance_m=1.e-5).success


def test_absent_overlap_is_explicit_and_not_a_zero_gap():
    target=np.array([[2.,2.,.01],[3.,2.,.01],[2.,3.,.01]])
    result=proof(target=target)
    assert result.success and result.status=='no_overlap'
    assert result.lower_gap_m is None


@pytest.mark.parametrize('xy',(
    [[-1.,0.],[0.,-1.],[0.,0.]], # true point closure
    [[0.,0.],[0.,-1.],[1.,0.]], # true segment closure
))
def test_lower_dimensional_closure_is_not_discarded(xy):
    target=np.column_stack((xy,np.full(3,.01)))
    result=proof(target=target)
    assert not result.success and result.status=='gap_budget_violation'
    target[:,2]=-.01
    result=proof(target=target)
    assert result.success and result.lower_gap_m<=.01


def hidden_inputs():
    target=np.array([[-1.,-1.,0.],[3.,-1.,0.],[-1.,3.,0.]])
    source=SOURCE.copy();source[:,2]=[1.e-6,4.e-6,4.e-6]
    data=list(inputs(source,target))
    data[7]=np.array([[0.,0.,-8.e-6],[0.,0.,0.],[0.,0.,0.]])
    data[8]=-data[7]
    return tuple(data)


def test_hidden_point_approach_cannot_be_inferred_from_open_weak_modes():
    data=hidden_inputs()
    # Endpoint gaps are positive. At t=.5 the first nodal gap is -1e-6,
    # while exact P1 averages remain positive: (2*(-1)+4+4)/4=1.5e-6.
    weak=s3_pressure_hermite_bounds(*data,active=np.zeros(3,dtype=bool),gap_tolerance_m=1.e-10)
    assert weak.success,weak
    unguarded=s3_overlap_hermite_gap_bounds(*data,gap_tolerance_m=2.e-5)
    assert unguarded.success,unguarded
    guarded=s3_overlap_hermite_gap_bounds(*data,gap_tolerance_m=2.e-5,
        exclude_open_endpoint_approach_tolerance_m=1.e-10,maximum_depth=4)
    assert not guarded.success and guarded.status=='hidden_approach_unresolved'
    assert guarded.lower_gap_m is None


def test_original_polarity_is_checked_even_for_remote_absent_overlap():
    target=np.array([[2.,2.,.01],[3.,2.,.01],[2.,3.,.01]])
    data=list(inputs(target=target));data[5]=target[[0,2,1]];data[6]=data[5]
    result=s3_overlap_hermite_gap_bounds(*data,gap_tolerance_m=.001)
    assert not result.success and result.status=='orientation_invalid'


def test_second_source_face_failure_cannot_accept_the_first_resolved_face():
    source=np.array([[0.,0.,.001],[1.,0.,.001],[1.,1.,.001],[0.,1.,-.01]])
    rs=source.copy();rs[:,2]=0.
    target=np.array([[-1.,-1.,0.],[3.,-1.,0.],[-1.,3.,0.]])
    rt=target.copy();rt[:,2]=-.001
    zeros=np.zeros_like(source);zt=np.zeros_like(target)
    result=s3_overlap_hermite_gap_bounds(rs,rt,((0,1,2),(0,2,3)),
        source,source,target,target,zeros,zeros,zt,zt,1.,gap_tolerance_m=.001)
    assert not result.success and result.status=='gap_budget_violation'
    assert result.lower_gap_m is None


@pytest.mark.parametrize('limits,status',(
    ({'maximum_spatial_cells':1},'spatial_budget_exhausted'),
    ({'maximum_time_panels':1},'time_budget_exhausted'),
))
def test_unresolved_resource_limits_do_not_return_partial_success(limits,status):
    result=s3_overlap_hermite_gap_bounds(*hidden_inputs(),gap_tolerance_m=2.e-5,
        exclude_open_endpoint_approach_tolerance_m=1.e-10,**limits)
    assert not result.success and result.status==status
    assert result.lower_gap_m is None


def test_owned_cancellation_returns_no_completed_bound():
    result=proof(cancel=lambda:True)
    assert not result.success and result.status=='cancelled'
    assert result.lower_gap_m is None and result.visited_time_panels==0


@pytest.mark.parametrize('change',(
    {'dt':False},{'dt':0.},{'gap_tolerance_m':-1.},
    {'maximum_depth':False},{'exclude_open_endpoint_approach_tolerance_m':complex(1.)},
))
def test_invalid_values_are_rejected(change):
    data=list(inputs())
    if 'dt' in change:data[-1]=change['dt'];change={key:value for key,value in change.items() if key!='dt'}
    with pytest.raises(ValueError):
        s3_overlap_hermite_gap_bounds(*data,**{'gap_tolerance_m':.001,**change})
