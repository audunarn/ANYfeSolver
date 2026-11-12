"""Nonzero curved-force derivative through actual mixed-owner coordinates."""
from dataclasses import replace
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_distributed_load import owner, load


def test_curved_force_jacobian_scatter_and_actual_update(request):
    base=_panel();beams=[]
    for index,raw in enumerate(base.beams):
        element,inertia=raw.instantiate();ref=element.operator.reference
        xyz=np.array(ref.coordinates);xyz[1,1]+=.04*(-1)**index
        frames=[]
        for derivative in ((-1.5,2.,-.5),(-.5,0.,.5),(.5,-2.,1.5)):
            tangent=np.asarray(derivative)@xyz;tangent/=np.linalg.norm(tangent)
            normal=np.array((0.,0.,1.))
            frames.append(np.column_stack((tangent,np.cross(normal,tangent),normal)))
        if index:frames[0]=shared_frame
        shared_frame=frames[-1]
        beams.append(b3_ge.define_beam('b3-ge',element.element_id,element.node_ids,
            xyz,np.array(frames),element.section,inertia,quadrature=element.operator.order))
    a=owner(request,replace(base,beams=tuple(beams),q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1))
    driver=FinitePointDriver(a)
    pattern,_=driver._prepare_load(load(a,force=(17.,23.,-100.)),(.1,))
    state=replace(driver._state,cells=np.tile(rotation([.1,-.08,.12]),(2,2,1,1)))
    f,j,_=driver._external.evaluate(state)
    assert np.linalg.norm(j.data)>.1
    r1,k1,*_=driver._assemble(state,1.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    r0,k0,*_=driver._assemble(state,0.,driver._shell_origins,driver._beam_origins,pattern=pattern)
    n=len(f)
    np.testing.assert_allclose(r1[:n]-r0[:n],-f,rtol=1e-7,atol=1e-6)
    np.testing.assert_allclose((k1-k0)[:n,:n].toarray(),-j.toarray(),rtol=1e-6,atol=1e-6)
    direction=np.zeros(len(r1))
    for _,slots,_ in a._beam_parts:
        direction[np.asarray(slots[18:24])+a._shell_count]=(.3,-.2,.1,.2,.4,-.1)
    h=1e-5
    fp=driver._external.evaluate(driver._advance(state,direction,h))[0]
    fm=driver._external.evaluate(driver._advance(state,direction,-h))[0]
    np.testing.assert_allclose((fp-fm)/(2*h),j@direction[:n],rtol=1e-7,atol=1e-8)
