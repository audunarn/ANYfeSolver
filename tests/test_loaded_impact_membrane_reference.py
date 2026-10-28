"""Independent CST von Karman membrane term of the loaded-impact energy oracle."""
import numpy as np
import pytest
from examples.general_contact_beam_s3_transient_directional_tilted import build_pose
from examples.general_contact_beam_s3_transient_directional_loaded_impact import independent_membrane,independent_reference


def _pose():
    m,_,_,u,refs,dofs,*_=build_pose()
    return m,u,dofs


def _skin_direction(dofs,values=(.3,-.7,.4)):
    d=np.zeros(max(max(v) for v in dofs.values())+1)
    for n,value in zip((3,4,5),values):d[dofs[n][2]]=value
    return d


def test_membrane_closed_form_for_planar_tilt():
    m,u,dofs=_pose();u=np.zeros_like(u);a,b=.01,-.004
    for n in (3,4,5):
        x,y,_=m.mesh.nodes[n].coords();u[dofs[n][2]]=a*x+b*y
        u[dofs[n][3]]=.2;u[dofs[n][4]]=-.1  # rotations do not enter membrane strain
    energy,gradient=independent_membrane(m,u)
    xy=np.array([m.mesh.nodes[n].coords()[:2] for n in (3,4,5)])
    area=abs(np.linalg.det(np.c_[np.ones(3),xy]))/2;E,nu,t=1.e5,.3,1/64
    ex,ey,gxy=a*a/2,b*b/2,a*b
    expected=.5*area*E*t/(1-nu**2)*(ex*ex+ey*ey+2*nu*ex*ey+(1-nu)/2*gxy*gxy)
    assert energy==pytest.approx(expected,rel=1e-13)
    # Rigid w translation leaves slopes unchanged: zero net transverse force.
    assert abs(gradient.sum())<=1e-13*np.abs(gradient).max()


@pytest.mark.parametrize('values',[(.3,-.7,.4),(1.,0.,0.),(0.,-1.,.5)])
def test_membrane_directional_derivative(values):
    m,u,dofs=_pose();d=_skin_direction(dofs,values)
    energy,gradient=independent_membrane(m,u)
    assert energy>0.
    slope=gradient@np.array([d[dofs[n][2]] for n in (3,4,5)])
    # Quartic energy: central difference error is exactly h^2/6 E'''[d,d,d].
    h=1.e-6
    numeric=(independent_membrane(m,u+h*d)[0]-independent_membrane(m,u-h*d)[0])/(2*h)
    assert numeric==pytest.approx(slope,rel=1e-6,abs=1e-14)


def test_reference_structural_force_includes_membrane_gradient():
    m,u,dofs=_pose();zero=np.zeros_like(u)
    d=np.zeros_like(u)
    for n,value in zip((1,6,2),(.2,-.1,.15)):d[dofs[n][2]]=value;d[dofs[n][4]]=.05
    for n,value in zip((3,4,5),(-.1,.3,.2)):d[dofs[n][2]]=value
    d[dofs[3][3]]=.02;d[dofs[3][4]]=-.03
    def structural(q):
        ref,force,_=independent_reference(m,q,zero)
        return ref['beam']+ref['shell']+ref['membrane'],force
    ref,_,_=independent_reference(m,u,zero)
    assert ref['energy']==pytest.approx(ref['beam']+ref['shell']+ref['membrane']+ref['contact']+ref['kinetic'],rel=1e-15)
    assert ref['membrane']==pytest.approx(independent_membrane(m,u)[0],rel=0)
    _,force=structural(u);h=1.e-6
    numeric=(structural(u+h*d)[0]-structural(u-h*d)[0])/(2*h)
    assert numeric==pytest.approx(force@d,rel=1e-6)
    # The membrane gradient is a nonzero part of the skin w force at the tilted pose.
    membrane_force=independent_membrane(m,u)[1]
    assert np.linalg.norm(membrane_force)>0.
