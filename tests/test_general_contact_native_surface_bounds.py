"""Independent curved material geometry/search controls, not solve admission."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver.general_contact.native_surface_bounds import native_round_patch_bounds
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.contact_search import ContactAABBIndex
from test_general_contact_native_retained_station import _reference


def _setup(twist=False, axial_order=1):
    base = _reference(True)
    frames = base.nodal_triads.copy()
    if twist:
        for i,roll in enumerate((.1,.4,-.2)):
            frames[i] = frames[i]@Rotation.from_rotvec((roll,0.,0.)).as_matrix()
    reference = CenteredCurvedBeam3ReferenceGeometry(base.coordinates,frames)
    rule = NativeRoundSurfaceRule(7,reference,.005,xi_range=(-.95,.8),
        angle_range=(.12,5.9),axial_order=axial_order,angular_order=2)
    U = np.stack((Rotation.from_rotvec((.8,-.1,.2)).as_matrix(),
                  Rotation.from_rotvec((-.6,.3,-.2)).as_matrix()))
    p = reference.coordinates.copy()
    p[1] += (.005,-.002,.004)
    return rule,p,np.zeros_like(p),U


def _independent_point(xi, angle, radius, p, low, U, twist):
    # Analytic parabola and planar tangent: no production station/bounds kernel.
    cell = 0 if xi<0 else 1
    t = xi+1 if cell==0 else xi
    tangent = np.array((1.,-.4*xi,0.));tangent/=np.linalg.norm(tangent)
    transverse = np.array((-tangent[1],tangent[0],0.))
    frame = np.column_stack((tangent,transverse,(0.,0.,1.)))
    rolls = (.1,.4,-.2) if twist else (0.,0.,0.)
    roll = (1-t)*rolls[cell]+t*rolls[cell+1]
    frame = frame@Rotation.from_rotvec((roll,0.,0.)).as_matrix()
    lift = np.array((0.,.2*t*(1-t),0.))
    arm = lift+radius*(np.cos(angle)*frame[:,1]+np.sin(angle)*frame[:,2])
    return (1-t)*(p[cell]+low[cell])+t*(p[cell+1]+low[cell+1])+U[cell]@arm


@pytest.mark.parametrize('twist',(False,True))
def test_curved_twisted_whole_patch_bounds_and_order_independence(twist):
    rule,p,low,U = _setup(twist)
    bounds = rule.current_bounds(p,low,U)
    assert bounds.cells==(0,1)
    assert not bounds.lower.flags.writeable and not bounds.upper.flags.writeable
    radius=rule.definition['radius']
    for xi in np.r_[np.linspace(-.95,-1e-8,77),np.linspace(1e-8,.8,67)]:
        for angle in np.linspace(.12,5.9,19):
            point=_independent_point(xi,angle,radius,p,low,U,twist)
            assert np.all(bounds.lower<=point) and np.all(point<=bounds.upper)
    refined,_,_,_=_setup(twist,axial_order=6)
    other=refined.current_bounds(p,low,U)
    np.testing.assert_array_equal(bounds.as_search_box(),other.as_search_box())


def test_fixed_nodes_retained_curved_lift_expands_bounds_and_search():
    rule,p,low,_=_setup()
    p=rule.samples[0].station.reference.coordinates
    rest=np.tile(np.eye(3),(2,1,1))
    first=rule.current_bounds(p,low,rest)
    moved=rest.copy();moved[0]=Rotation.from_rotvec((.8,0.,0.)).as_matrix()
    second=rule.current_bounds(p,low,moved)
    hidden=_independent_point(-.5,.12,.005,p,low,moved,False)
    assert hidden[2] > np.max(p[:,2])+.005
    assert hidden[2] > first.upper[2]
    assert second.upper[2] > first.upper[2]
    pytest.importorskip('anygeometry')
    index=ContactAABBIndex((('native-material',7),),np.array([second.as_search_box()]))
    box=np.r_[hidden-1e-8,hidden+1e-8]
    assert index.query(box)==(('native-material',7),)
    index.refit(np.array([first.as_search_box()]))
    assert index.query(box)==()


def test_restricted_curved_box_transform_and_compensated_low_parts():
    rule,p,low,U=_setup(True)
    G=Rotation.from_rotvec((.3,-.4,.2)).as_matrix()
    offset=np.array((2.**30,-2.**29,2.**28))
    # Preserve the given binary values through normalized high/low pairs.
    from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
    actual=p@G.T
    high=np.empty_like(p);small=np.empty_like(p)
    for i in range(3):
        for j in range(3):
            high[i,j],small[i,j]=split_sum((float(offset[j]),float(actual[i,j])))
    bounds=rule.current_bounds(high,small,G@U)
    for xi in (-.95,-.5,-.01,.01,.4,.8):
        for angle in (.12,1.,3.,5.9):
            point=G@_independent_point(xi,angle,.005,p,low,U,True)+offset
            assert np.all(bounds.lower<=point) and np.all(point<=bounds.upper)
    assert bounds.roundoff_padding_m>1e-4


def test_curved_bounds_definition_and_rotation_rejections():
    rule,p,low,U=_setup()
    with pytest.raises(TypeError,match='exact native'):
        native_round_patch_bounds(object(),p,low,U)
    wrong=U.copy();wrong[0,0,0]+=1e-5
    with pytest.raises(ValueError,match='proper rotations'):
        rule.current_bounds(p,low,wrong)
    low[0,0]=.1
    with pytest.raises(ValueError,match='normalized coordinate'):
        rule.current_bounds(p,low,U)
    low[:]=0.;rule.definition['radius']*=2
    with pytest.raises(ValueError,match='definition changed'):
        rule.current_bounds(p,low,U)


@pytest.mark.parametrize('twist',(False,True))
def test_swept_curved_patch_contains_entire_declared_affine_exp_path(twist):
    rule,p,low,U=_setup(twist)
    move=np.array(((.02,-.04,.05),(-.03,.06,-.02),(.01,-.02,.08)))
    spin=np.array(((.5,-.4,.3),(-.3,.2,.6)))
    bounds=rule.swept_bounds(p,low,U,move,spin)
    assert 'SWEEP' in bounds.profile
    for alpha in np.linspace(0,1,17):
        moved=np.stack([Rotation.from_rotvec(alpha*w).as_matrix()@u for w,u in zip(spin,U)])
        for xi in (-.95,-.73,-.3,-.01,.01,.2,.6,.8):
            for angle in (.12,1.4,3.2,5.9):
                point=_independent_point(xi,angle,.005,p+alpha*move,low,moved,twist)
                assert np.all(bounds.lower<=point) and np.all(point<=bounds.upper)


def test_swept_search_detects_hidden_interior_rotation_and_keeps_path_limit():
    base=_reference(True)
    rule=NativeRoundSurfaceRule(7,base,.005,xi_range=(-.55,-.45),
        angle_range=(.12,5.9),axial_order=1,angular_order=2)
    p=base.coordinates;low=np.zeros((3,3));U=np.tile(np.eye(3),(2,1,1))
    move=np.zeros((3,3));spin=np.array(((2.6,0.,0.),(0.,0.,0.)))
    swept=rule.swept_bounds(p,low,U,move,spin)
    end=U.copy();end[0]=Rotation.from_rotvec(spin[0]).as_matrix()
    start_box=rule.current_bounds(p,low,U).as_search_box()
    end_box=rule.current_bounds(p,low,end).as_search_box()
    middle=U.copy();middle[0]=Rotation.from_rotvec(spin[0]/2).as_matrix()
    hidden=_independent_point(-.5,.12,.005,p,low,middle,False)
    assert hidden[2]>max(start_box[5],end_box[5])
    pytest.importorskip('anygeometry')
    box=np.r_[hidden-1e-8,hidden+1e-8]
    index=ContactAABBIndex((('native-material',7),),np.array([swept.as_search_box()]))
    assert index.query(box)==(('native-material',7),)
    for endpoint in (start_box,end_box):
        index.refit(np.array([endpoint]))
        assert index.query(box)==()
    spin[0]=(2.9,0.,0.)
    with pytest.raises(ValueError,match='angular increment'):
        rule.swept_bounds(p,low,U,move,spin)


@pytest.mark.parametrize('stretch',(-2e-12,2e-12))
def test_reference_transport_gain_covers_admitted_nonorthogonal_nodes(stretch):
    from anyfesolver.general_contact.native_surface_bounds import _reference_radial_gain
    reference=_reference(True)
    frames=reference.nodal_triads.copy()
    frames[:,:,0]*=1+stretch
    made=CenteredCurvedBeam3ReferenceGeometry(reference.coordinates,frames)
    gain=_reference_radial_gain(made)
    # SVD independently computes the actual interpolated operator norm.
    for xi in np.linspace(-1,1,401):
        assert np.linalg.svd(made.frame(float(xi)),compute_uv=False)[0]<=gain


@pytest.mark.parametrize('stretch',(-1e-13,1e-13))
def test_transport_gain_near_frozen_turn_with_nonorthogonal_nodes(stretch):
    from anyfesolver.general_contact.native_surface_bounds import _reference_radial_gain
    reference=_near_turn_reference(stretch)
    gain=_reference_radial_gain(reference)
    for xi in np.linspace(-1,1,2001):
        assert np.linalg.svd(reference.frame(float(xi)),compute_uv=False)[0]<=gain
    rule=NativeRoundSurfaceRule(7,reference,1e-5,axial_order=1,angular_order=2)
    result=rule.current_bounds(reference.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)))
    assert result.cells==(0,1)


def _near_turn_reference(stretch):
    turn=.9*np.pi-1e-6
    a=np.array((1.,0.,0.));b=np.array((2.,np.tan(np.pi-turn),0.))
    coordinates=np.array((-a+b/2,np.zeros(3),a+b/2))
    frames=[]
    for xi in (-1.,0.,1.):
        t=a+xi*b;t/=np.linalg.norm(t)
        frames.append(np.column_stack((t*(1+stretch),(-t[1],t[0],0.),(0.,0.,1.))))
    return CenteredCurvedBeam3ReferenceGeometry(coordinates,np.array(frames))


@pytest.mark.parametrize('stretch',(-2e-12,2e-12))
def test_near_turn_distorted_frames_keep_original_owner_rejection(stretch):
    from anyfesolver.ge_beam3_curved_reference import GeBeam3CurvedFrameError
    with pytest.raises(GeBeam3CurvedFrameError,match='frozen shortest-transport roll map'):
        _near_turn_reference(stretch)


def test_q2_restriction_retains_sub_ulp_material_parameter_shift():
    from fractions import Fraction as F
    from anyfesolver.general_contact.native_surface_bounds import _restrict_q2
    # xi near zero maps to t just below one; float(xi+1) would erase the shift.
    left=1-F(1,2**60);right=1-F(1,2**61)
    restricted=_restrict_q2(((F(0),F(1),F(2)),),left,right)[0]
    assert restricted[0]==2*left and restricted[-1]==2*right
    assert restricted[-1]<2
