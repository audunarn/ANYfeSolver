"""Independent P1 integrals, overlap controls and hidden pressure-phase paths."""
from fractions import Fraction as Q
from math import comb
from pathlib import Path
import runpy
import numpy as np
import pytest
from anyfesolver.general_contact.s3_pressure_hermite_bounds import s3_pressure_hermite_bounds
from anyfesolver.general_contact.normal_s3_skin_path import prepare_normal_s3_skin_path
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager

SOURCE=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
LARGE=np.array([[-2.,-2.,0.],[5.,-2.,0.],[-2.,5.,0.]])
ZERO=np.zeros((3,3))


def bound(source,target,*,active,rate0=ZERO,rate1=ZERO,reference=SOURCE,**kwargs):
    return s3_pressure_hermite_bounds(reference,LARGE,((0,1,2),),
        source,source,target,target,rate0,rate1,ZERO,ZERO,1.,active=np.array(active,dtype=bool),
        gap_tolerance_m=1.e-10,**kwargs)


@pytest.mark.parametrize('active',(False,True))
def test_full_reference_measure_and_pressure_phase(active):
    source=SOURCE.copy();source[:,2]=-.001 if active else .001
    result=bound(source,LARGE,active=[active]*3)
    assert result.success
    for lower,upper in zip(result.support_lower,result.support_upper):
        assert lower<=1/6<=upper
    joint=(-.001-1.e-10 if active else .001+1.e-10)/6
    values=result.closed_joint_upper if active else result.open_joint_lower
    np.testing.assert_allclose(values,joint,rtol=0.,atol=1.e-18)


def test_partial_overlap_uses_common_edge_and_parent_basis():
    target=np.array([[-2.,-2.,0.],[.5,-2.,0.],[.5,3.,0.]])
    target[:,2]=.0003-.001*target[:,0]
    # Original domain: 0<=x<=1/2, 0<=y<=1-x. Analytic monomial integration.
    def integral(a,b):
        return sum(Q((-1)**k*comb(b+1,k),b+1)*Q(1,2)**(a+k+1)/Q(a+k+1)
                   for k in range(b+2))
    shapes=({(0,0):Q(1),(1,0):Q(-1),(0,1):Q(-1)},
            {(1,0):Q(1)},{(0,1):Q(1)})
    # Derive the actual supplied binary-float gap plane exactly.
    g=tuple(-Q(float(z)) for z in target[:,2])
    bx=(g[1]-g[0])/Q(5,2);by=(g[2]-g[1])/5
    offset=g[0]+2*bx+2*by
    gap={(0,0):offset,(1,0):bx,(0,1):by}
    support=[sum(v*integral(*powers) for powers,v in shape.items()) for shape in shapes]
    weighted=[sum(v*w*integral(a+c,b+d) for (a,b),v in shape.items()
                  for (c,d),w in gap.items()) for shape in shapes]
    phase=[value<0 for value in weighted]
    assert phase==[True,False,True]
    result=bound(SOURCE,target,active=phase,maximum_spatial_cells=10000)
    assert result.success,result
    assert result.visited_spatial_cells>3
    for i in range(3):
        assert Q(result.support_lower[i])<=support[i]<=Q(result.support_upper[i])
        assert Q(result.open_joint_lower[i])<=weighted[i]+Q(1.e-10)*support[i]
        assert Q(result.closed_joint_upper[i])>=weighted[i]-Q(1.e-10)*support[i]


def test_different_outside_edges_do_not_discard_crossed_overlap():
    # No source vertex lies inside this target, but its interior is covered.
    target=np.array([[.2,.2,.001],[.4,.2,.001],[.2,.4,.001]])
    result=bound(SOURCE,target,active=[False]*3,maximum_spatial_cells=512)
    assert not result.success
    assert result.status in ('spatial_budget_exhausted','pressure_phase_violation')
    closed=bound(SOURCE,target,active=[True]*3)
    assert closed.success
    assert len(closed.support_upper)==3 and all(v>0 for v in closed.support_upper)


def test_small_positive_support_is_not_deleted_to_pass_open_phase():
    target=np.array([[0.,0.,.001],[1.e-12,0.,.001],[0.,1.e-12,.001]])
    result=bound(SOURCE,target,active=[False]*3,maximum_spatial_cells=128)
    assert not result.success
    assert result.status=='spatial_budget_exhausted'


def test_certified_absent_overlap_retains_all_modes_without_division():
    target=LARGE+[20.,0.,.001]
    for active in (False,True):
        result=bound(SOURCE,target,active=[active]*3)
        assert result.success
        assert result.support_lower==result.support_upper==(0.,0.,0.)
        assert result.open_joint_lower==result.closed_joint_upper==(0.,0.,0.)


def test_actual_pressure_excursion_that_corner_guard_misses():
    example=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_deforming_s3_impact_dev.py'))
    model,definition,modes=example['build_model']()
    fixture=example['_fixture']
    manager=GeneralS3PatchManager(model,definition,fixture['SETTINGS'])
    owner=prepare_normal_s3_skin_path(model,manager,modes)
    q=np.r_[np.array([1.e-7,1.1e-6,1.1e-6,1.e-7])-fixture['GAP'],np.zeros(3)]
    rate=np.r_[np.full(4,-4.e-6),np.zeros(3)]
    lower,failure=owner.sweep(manager,modes@q,modes@q,modes@rate,-modes@rate,1.,
                             event_gap_tolerance_m=1.e-10)
    assert failure is None and lower>=-2.e-5 # Historical corner-only proof passes.
    first=manager.evaluate(modes@q,tangent=False)[2]
    middle=q.copy();middle[:4]-=1.e-6
    mid=manager.evaluate(modes@middle,tangent=False)[2]
    np.testing.assert_allclose(first.normalized_weighted_gap_m,
        np.array([.475,.85,.725,.35])*1.e-6,rtol=0.,atol=1.e-16)
    assert np.all(np.array(mid.normalized_weighted_gap_m)<0.)
    skins=owner.skins(manager,modes@q,modes@q,modes@rate,-modes@rate)
    result=s3_pressure_hermite_bounds(manager.reference[0],manager.reference[1],
        manager.law.source_face_indices,*skins,1.,active=np.zeros(4,dtype=bool),gap_tolerance_m=1.e-10)
    assert not result.success
    assert result.status=='pressure_phase_violation'


@pytest.mark.parametrize('change',({'maximum_time_panels':True},{'maximum_spatial_cells':0},
    {'maximum_depth':-1},{'gap_tolerance_m':complex(1.e-10)}, {'dt':False}))
def test_invalid_controls_do_not_invent_geometry(change):
    args=dict(active=np.zeros(3,dtype=bool),gap_tolerance_m=1.e-10)
    duration=change.pop('dt',1.) if 'dt' in change else 1.
    args.update(change)
    with pytest.raises(ValueError):
        s3_pressure_hermite_bounds(SOURCE,LARGE,((0,1,2),),SOURCE,SOURCE,LARGE,LARGE,
            ZERO,ZERO,ZERO,ZERO,duration,**args)


def test_cancelled_and_inverted_pressure_paths_are_non_success():
    result=bound(SOURCE,LARGE,active=[False]*3,cancel=lambda:True)
    assert not result.success and result.status=='cancelled'
    inverted=SOURCE[[0,2,1]]
    result=bound(inverted,LARGE,active=[False]*3)
    assert not result.success and result.status=='orientation_invalid'


@pytest.mark.parametrize('q',([-0.003,.001,.002,-.001,.002,-.001,-.004],
                              [-.00023,-.000229,-.000231,-.000228,-1.e-5,-9.e-6,-1.1e-5]))
def test_shared_node_aggregates_different_source_face_factors(q):
    # Independent closed-form covered-triangle oracle from the driver checks.
    oracle=runpy.run_path(str(Path(__file__).parent/'test_general_contact_transient_deforming_s3.py'))
    q=np.asarray(q)
    model,definition,modes=oracle['build']()
    manager=GeneralS3PatchManager(model,definition,oracle['example']['SETTINGS'])
    owner=prepare_normal_s3_skin_path(model,manager,modes)
    expected=oracle['gaps'](q)*oracle['SUPPORT']
    proof=owner.pressure_sweep(manager,modes@q,modes@q,np.zeros(42),np.zeros(42),1.,
        active=expected<0.,gap_tolerance_m=1.e-10)
    assert proof.success,proof
    for index in range(4):
        assert proof.open_joint_lower[index]<=expected[index]+1.e-10*oracle['SUPPORT'][index]+1.e-17
        assert proof.closed_joint_upper[index]>=expected[index]-1.e-10*oracle['SUPPORT'][index]-1.e-17
