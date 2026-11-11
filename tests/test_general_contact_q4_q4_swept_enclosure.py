"""Independent exact polynomial proof and real new-profile controller checks."""
from fractions import Fraction as F
from pathlib import Path
import json, os
import numpy as np
import pytest
from anyfesolver.general_contact.q4_q4_swept_bound import q4_q4_whole_pair_swept_lower as lower


def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def scale(a,s):return tuple(x*s for x in a)
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def exact(a):return [tuple(F(float(v)) for v in row) for row in a]
def quad(z=0.):return np.array([[0.,0.,z],[1.,0.,z],[1.,1.,z],[0.,1.,z]])
def controls(a0,a1,b0,b1):
    a0,a1,b0,b1=map(exact,(a0,a1,b0,b1));normals=[]
    for k in range(4):
        e0,f0=sub(a0[(k+1)%4],a0[k]),sub(a0[(k-1)%4],a0[k])
        e1,f1=sub(a1[(k+1)%4],a1[k]),sub(a1[(k-1)%4],a1[k])
        normals.append((cross(e0,f0),scale(add(cross(e0,f1),cross(e1,f0)),F(1,2)),cross(e1,f1)))
    nums=[]
    for i in range(4):
        for j in range(4):
            d0,d1=sub(a0[i],b0[j]),sub(a1[i],b1[j])
            for n0,nm,n1 in normals:
                nums.extend((dot(d0,n0),(2*dot(d0,nm)+dot(d1,n0))/3,
                             (dot(d0,n1)+2*dot(d1,nm))/3,dot(d1,n1)))
    axis=max(range(3),key=lambda j:abs(normals[0][0][j]));sign=1 if normals[0][0][axis]>=0 else -1
    den=min(sign*n[axis] for ns in normals for n in ns)
    assert den>0
    return min(F(0),min(nums))/den,nums,normals


def save(data):
    p=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if p:Path(p).write_text(json.dumps(data,indent=2),encoding='utf-8')


def test_exact_whole_time_controls_and_errors():
    reference=quad();cases=[]
    a0=quad(-.000005);a1=quad(.015);b0=quad();b1=quad()
    cases.append(('planar_release',a0,a1,b0,b1))
    a0=quad(.002);a0[2,2]+=.0002;a1=a0.copy();a1[:,0]+=.02;a1[1,2]+=.0001
    b0=quad();b0[2,2]=.0003;b1=b0.copy();b1[3,2]=-.0001
    cases.append(('both_moving_warped_Q1',a0,a1,b0,b1))
    a0=quad(-.000005);a1=a0.copy();a1[:,0]+=.02;b0=quad();b1=quad();b1[2,2]=-.000001
    for x in (a0,a1,b0,b1):x[:,:2]+=2.**30
    cases.append(('large_translation_cancellation',a0,a1,b0,b1))
    report=[]
    for name,a0,a1,b0,b1 in cases:
        bound=lower(a0,a1,b0,b1,reference,reference)
        oracle,nums,normals=controls(a0,a1,b0,b1)
        assert np.isfinite(bound) and F(bound)<=oracle
        report.append(dict(case=name,directed_lower=bound,exact_sufficient_lower=str(oracle),numerator_controls=len(nums)))
    a=quad();b=quad();e=2.**-20
    bound=lower(a,a,b,b,reference,reference,endpoint_errors=(e,e,e,e))
    # An exact admitted box member translates every source down and target up.
    assert np.isfinite(bound) and F(bound)<=-2*F(e)
    # All-coordinate errors also allow tilt; a vertical-only tightness claim
    # would be unjustified. The required inequality above stays exact.
    with pytest.raises(ValueError):lower(a,a,b,b,reference,reference,endpoint_errors=(-e,0,0,0))
    assert lower(a,a,b,b,reference,reference,endpoint_errors=(1.,)*4)==-np.inf
    reversed_target=b[[0,3,2,1]]
    assert lower(a,a,reversed_target,reversed_target,reference,reference)==-np.inf
    degenerate=np.zeros((4,3));assert lower(degenerate,degenerate,b,b,reference,reference)==-np.inf
    # Endpoint gap .1m, genuine interior crossing -.15m at target x=0,t=1/2.
    a0=np.array([[0.,-1.,.1],[2.,-1.,.1],[2.,1.,.1],[0.,1.,.1]])
    a1=np.array([[-1.,-1.,-.9],[1.,-1.,1.1],[1.,1.,1.1],[-1.,1.,-.9]])
    target=np.array([[0.,-.5,0.],[.01,-.5,0.],[.01,.5,0.],[0.,.5,0.]])
    # Independent exact affine plane: z=.1+t*(x-(1-t)).
    assert F(1,10)+F(0)*(0-(1-F(0)))>0 and F(1,10)+F(1)*(0-(1-F(1)))>0
    interior=F(1,10)-F(1,2)*F(1,2);assert interior==F(-3,20)
    assert lower(a0,a1,target,target,reference,target)<-5e-5
    # Near-cancelling reference cross: exact binary polarity is negative even
    # though the rounded cross loses the eps^2 term. Unresolved must refuse.
    eps=2.**-26
    ref=np.array([[0.,0.,0.],[1.+eps,1.,0.],[2.,2.,0.],[1.,1.-eps,0.]])
    refaxis=cross(sub(exact(ref)[1],exact(ref)[0]),sub(exact(ref)[3],exact(ref)[0]))
    assert refaxis[2]==-F(eps)**2
    from anyfesolver.general_contact.q4_overlap_local_bound import _normal_controls_interval
    enclosure=_normal_controls_interval(ref,ref,0)
    assert F(enclosure[0][0,2])<=refaxis[2]<=F(enclosure[1][0,2])
    from anyfesolver.general_contact.q4_q4_overlap_derivatives import q4_q4_swept_chart_admissible
    assert not q4_q4_swept_chart_admissible(quad(),quad(),quad(),quad(),ref,reference)
    # Rounded midpoint encloses original exact binary arithmetic, not a chord reset.
    from anyfesolver.general_contact.s3_patch_static import _affine_midpoint_with_error
    left=(np.array([[2.**30,1.,-1.]]),);right=(np.nextafter(left[0],np.inf),)
    error=(np.full((1,3),2.**-24),)
    middle,me=_affine_midpoint_with_error(left,right,error,error)
    for j in range(3):
        exact_mid=(F(left[0][0,j])+F(right[0][0,j]))/2
        assert F(middle[0][0,j])-F(me[0][0,j])<=exact_mid-F(error[0][0,j])
        assert F(middle[0][0,j])+F(me[0][0,j])>=exact_mid+F(error[0][0,j])
    save(dict(cases=report,endpoint_error_member_gap_m=-2*e,ideal_decimal_plane_interior_gap=str(interior),
              midpoint_original_path_enclosed=True,law='unchanged'))


def test_actual_manager_errors_cutback_and_endpoint_guard(monkeypatch):
    import anyfesolver.general_contact.q4_q4_swept_bound as module
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact import resolve_automatic_contact_domain
    from examples.general_contact_automatic_q4_pair import build
    model,domain,settings,resolution,program=build()
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    calls=[];original=module.q4_q4_whole_pair_swept_lower
    def observed(*args,**kwargs):
        errors=kwargs.get('endpoint_errors')
        calls.append(None if errors is None else max(float(np.max(e)) for e in errors))
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'q4_q4_whole_pair_swept_lower',observed)
    zero=np.zeros(manager.map.shape[1]);assert manager._swept_admissible(zero,zero)
    assert any(x is not None and x>0 for x in calls)
    unsafe=zero.copy()
    for node in range(1,7):unsafe[model.mesh.dof_manager.get_node_dofs(node)[2]]=-.03
    assert not manager._swept_admissible(zero,unsafe)
    fraction=manager.admissible_step_fraction(zero,unsafe);assert 0<fraction<1
    with pytest.raises(ValueError,match='committed Q4/Q4 gap'):manager._swept_admissible(unsafe,unsafe)
    # Endpoints/chart cannot route acceptance through the historic float cone.
    import anyfesolver.general_contact.s3_patch_static as controller
    def trap(*args,**kwargs):raise AssertionError('old Q4 gap branch used')
    monkeypatch.setattr(controller,'q4_source_swept_gap_lower',trap)
    monkeypatch.setattr(controller,'q4_source_gap_lower',trap)
    assert manager._swept_admissible(zero,zero)
    # Force one inconclusive sufficient root bound, preserving endpoint checks,
    # so the REAL manager propagates enclosures through rounded child paths.
    forced=[False];children=[]
    def subdivided(*args,**kwargs):
        moving=not np.array_equal(args[0],args[1])
        if moving and not forced[0]:
            forced[0]=True
            return -np.inf
        if moving:
            children.append(max(float(np.max(e)) for e in kwargs['endpoint_errors']))
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'q4_q4_whole_pair_swept_lower',subdivided)
    safe=zero.copy()
    for node in range(1,7):safe[model.mesh.dof_manager.get_node_dofs(node)[2]]=-.001
    assert manager._swept_admissible(zero,safe)
    assert forced[0] and len(children)>=2 and min(children)>0
    save(dict(actual_positive_error_calls=sum(x is not None and x>0 for x in calls),
              maximum_endpoint_error_m=max(x or 0 for x in calls),safe_fraction=fraction,
              unsafe_committed_refused=True,legacy_gap_branch_not_used=True,
              forced_inconclusive_root_child_calls=len(children),child_error_max_m=max(children)))


def test_exact_binary_interior_crossing_with_safe_endpoints():
    a0=np.array([[0.,-1.,.1],[2.,-1.,.1],[2.,1.,.1],[0.,1.,.1]])
    a1=np.array([[-1.,-1.,-.9],[1.,-1.,1.1],[1.,1.,1.1],[-1.,1.,-.9]])
    target=np.array([[0.,-.5,0.],[.01,-.5,0.],[.01,.5,0.],[0.,.5,0.]])
    reference=quad()
    assert lower(a0,a0,target,target,reference,target)>=-5e-5
    assert lower(a1,a1,target,target,reference,target)>=-5e-5
    midpoint=[scale(add(x,y),F(1,2)) for x,y in zip(exact(a0),exact(a1))]
    # Q1 at u=1/4,v=1/2: same XY as target left-edge midpoint.
    weights=(F(3,8),F(1,8),F(1,8),F(3,8))
    point=tuple(sum(w*q[j] for w,q in zip(weights,midpoint)) for j in range(3))
    target_point=scale(add(exact(target)[0],exact(target)[3]),F(1,2))
    assert point[:2]==target_point[:2]
    normal=cross(sub(midpoint[1],midpoint[0]),sub(midpoint[3],midpoint[0]))
    numerator=dot(sub(point,target_point),normal);norm_squared=dot(normal,normal)
    assert numerator<0 and numerator**2>F(float(5e-5))**2*norm_squared
    assert lower(a0,a1,target,target,reference,target)<-5e-5
    save(dict(binary_material_point=[str(x) for x in point],signed_numerator=str(numerator),
              norm_squared=str(norm_squared),unsafe_squared_inequality=True,both_endpoint_bounds_pass=True))
