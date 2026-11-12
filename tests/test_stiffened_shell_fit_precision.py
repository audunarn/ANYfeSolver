"""Physical small-rotation precision and differentiated normalization."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_variational_shell import _quaternion_jets,deformation

@pytest.mark.parametrize('angle',[0.,1e-12,-1e-12,1e-7,-1e-7,.5,np.pi/2,2.])
def test_stable_fit_analytic_rotation_and_normalization(angle):
    ref=np.array([[-1.,-.5,0.],[1.,-.5,0.],[1.,.5,0.],[-1.,.5,0.]])
    s=np.sin(angle);cminus=-2*np.sin(angle/2)**2
    increment=np.array([[cminus,0.,s],[0.,0.,0.],[-s,0.,cminus]])
    u=ref@increment.T
    jets=_quaternion_jets(ref,None,displacement=u)
    q=np.array([v.value for v in jets]);dq=np.array([v.gradient for v in jets]);ddq=np.array([v.hessian for v in jets])
    expected=np.array([np.cos(angle/2),0.,np.sin(angle/2),0.])
    if q@expected<0:expected=-expected
    np.testing.assert_allclose(q,expected,rtol=2e-14,atol=max(1e-28,32*np.finfo(float).eps*abs(angle)))
    np.testing.assert_allclose(q@dq,0.,atol=3e-16)
    np.testing.assert_allclose(np.einsum('i,ijk->jk',q,ddq)+dq.T@dq,0.,atol=2e-15)

@pytest.mark.parametrize('count',[3,4])
def test_stable_fit_derivatives_at_finite_rotation_and_deformation(count):
    ref=np.array([[0.,0.,0.],[1.2,.1,0.],[.9,1.1,0.],[-.1,.9,0.]])[:count]
    u=.035*np.sin(np.arange(6*count)+.4)
    direction=np.cos(np.arange(len(u))+.6)
    value,d,second=deformation(ref,u,stable_translation=True)
    first=d@direction;expected=np.einsum('ijk,k->ij',second,direction)
    errors=[]
    for h in (2e-5,1e-5,5e-6):
        plus=deformation(ref,u+h*direction,stable_translation=True)
        minus=deformation(ref,u-h*direction,stable_translation=True)
        errors.append((np.linalg.norm((plus[0]-minus[0])/(2*h)-first)/np.linalg.norm(first),
                       np.linalg.norm((plus[1]-minus[1])/(2*h)-expected)/np.linalg.norm(expected)))
    # Registered directional checks use normalized errors over an FD plateau,
    # not relative accuracy of individually near-zero Hessian entries.
    assert max(max(pair) for pair in errors)<1e-6,errors

@pytest.mark.parametrize('width',[1.,1e-5,1e-6])
def test_stable_fit_repeated_lower_spectrum_and_existing_top_gap(width):
    from anyfesolver._ge_beam3_variational_shell import _davenport
    ref=np.array([[-1.,-width,0.],[1.,-width,0.],[1.,width,0.],[-1.,width,0.]])
    if width==1e-6:
        with pytest.raises(ValueError,match='nonunique'):_quaternion_jets(ref,None,displacement=np.zeros_like(ref))
        return
    jets=_quaternion_jets(ref,None,displacement=np.zeros_like(ref))
    q=np.array([v.value for v in jets]);r=ref/np.linalg.norm(ref);matrix=_davenport(r.T@r)
    assert np.linalg.norm(matrix@q-(q@matrix@q)*q)<=8*np.finfo(float).eps*np.linalg.norm(matrix)
    assert all(np.isfinite(v.gradient).all() and np.isfinite(v.hessian).all() for v in jets)

@pytest.mark.parametrize('shift',[-1e-6,0.,1e-6])
def test_stable_fit_general_axis_pivot_transition(shift):
    from scipy.spatial.transform import Rotation
    ref=np.array([[-1.,-.5,0.],[1.,-.5,0.],[1.,.5,0.],[-1.,.5,0.]])
    axis=np.array([1.,.3,-.2]);axis/=np.linalg.norm(axis)
    angle=2*np.arctan(1/axis[0])+shift
    rot=Rotation.from_rotvec(angle*axis);u=np.zeros((4,6))
    u[:,:3]=rot.apply(ref)-ref;u[:,3:]=angle*axis
    value,d,hessian=deformation(ref,u.ravel(),stable_translation=True)
    assert np.linalg.norm(value)<1e-13
    direction=np.sin(np.arange(24)+.2);step=1e-5
    plus=deformation(ref,u.ravel()+step*direction,stable_translation=True)
    minus=deformation(ref,u.ravel()-step*direction,stable_translation=True)
    expected=np.einsum('ijk,k->ij',hessian,direction)
    assert np.linalg.norm((plus[1]-minus[1])/(2*step)-expected)/np.linalg.norm(expected)<1e-6
@pytest.mark.parametrize('angle',[1e-12,-1e-12])
def test_tiny_general_axis_rotation_components(angle):
    ref=np.array([[-1.,-.5,0.],[1.,-.5,0.],[1.,.5,0.],[-1.,.5,0.]])
    axis=np.array([1.,.3,-.2]);axis/=np.linalg.norm(axis)
    x,y,z=axis;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    increment=np.sin(angle)*cross+2*np.sin(angle/2)**2*(cross@cross)
    jets=_quaternion_jets(ref,None,displacement=ref@increment.T)
    q=np.array([v.value for v in jets]);expected=np.r_[np.cos(angle/2),np.sin(angle/2)*axis]
    if q@expected<0:q=-q
    np.testing.assert_allclose(q[1:],expected[1:],rtol=1e-13,atol=1e-28)
