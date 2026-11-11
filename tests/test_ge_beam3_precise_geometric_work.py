from copy import deepcopy
from decimal import Decimal as D, localcontext, getcontext, ROUND_UP
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import _ge_beam3_precise_geometric_work as precise
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from docs.reference_cases.ge_beam3_refined_controlled_case import model


@pytest.mark.parametrize('angle',(0.,1e-12,1e-7,.2,.51,1.,1.8,2.7,.9*np.pi-1e-7))
def test_independent_scipy_so3(angle):
    axis=np.array([.3,-.4,.5]);axis/=np.linalg.norm(axis);v=axis*angle
    with localcontext() as ctx:
        ctx.prec=80
        a=precise._exp([D.from_float(x) for x in v])
        np.testing.assert_allclose(np.array(a,dtype=float),Rotation.from_rotvec(v).as_matrix(),rtol=0,atol=1e-14)
        recovered=np.array(precise._log(a),dtype=float)
        np.testing.assert_allclose(recovered,v,rtol=0,atol=1e-14)
        # The oracle uses quaternion extraction, not this trace/atan algorithm.
        np.testing.assert_allclose(recovered,Rotation.from_matrix(np.array(a,dtype=float)).as_rotvec(),rtol=0,atol=1e-13)


def arguments():
    ref=np.array([[0.,0.,0.],[.5,.07,-.01],[1.,0.,0.]])
    return dict(reference_coordinates=ref,reference_triads=np.tile(np.eye(3),(3,1,1)),
        positions=ref.copy(),position_low=np.zeros((3,3)),nodal_frames=np.tile(np.eye(3),(3,1,1)),
        cell_rotations=np.tile(np.eye(3),(2,1,1)),increment=np.zeros(42),
        resultants=np.arange(18,dtype=float)/20,material_potential=[.125,2.**-58])


def test_exact_zero_and_simple_translation_work():
    args=arguments();args['material_potential']=[0.,0.]
    assert precise.potential(**args)==0.
    args['increment'][12]=.125;args['resultants'][3]=2.
    assert precise.potential(**args)==.25


@pytest.mark.parametrize('angle',(.01,1.2,3.14,4.7,6.28))
def test_arbitrary_common_rigid_motion(angle):
    args=arguments();args['positions'][1]+=[.002,-.003,.001]
    args['nodal_frames'][1]=Rotation.from_rotvec([.12,-.03,.04]).as_matrix()
    args['cell_rotations'][1]=Rotation.from_rotvec([-.03,.08,.02]).as_matrix()
    before=precise.potential(**args)
    axis=np.array([1.,-2.,3.]);axis*=angle/np.linalg.norm(axis);q=Rotation.from_rotvec(axis).as_matrix()
    args['positions']=args['positions']@q.T+[1.,-2.,.5]
    args['nodal_frames']=q@args['nodal_frames'];args['cell_rotations']=q@args['cell_rotations']
    assert abs(precise.potential(**args)-before)<1e-11


@pytest.mark.parametrize('kind',('nan','inf','shape','bool','reflection','bad_frame','increment','relative'))
def test_invalid_inputs_fail_closed(kind):
    args=arguments()
    if kind=='nan':args['positions'][0,0]=float('nan')
    elif kind=='inf':args['material_potential'][1]=float('inf')
    elif kind=='shape':args['increment']=np.zeros(41)
    elif kind=='bool':args['material_potential'][0]=True
    elif kind=='reflection':args['nodal_frames'][0,0,0]=-1.
    elif kind=='bad_frame':args['cell_rotations'][0,0,0]=1.001
    elif kind=='increment':args['increment'][3]=.91*np.pi
    else:args['nodal_frames'][0]=Rotation.from_rotvec([.91*np.pi,0.,0.]).as_matrix()
    with pytest.raises(ValueError):precise.potential(**args)


def test_cancellation_state_context_and_determinism():
    args=arguments();before=canonical(args);context=getcontext().copy()
    assert precise.potential(**args)==precise.potential(**args)
    expected=precise.potential(**args)
    with localcontext() as ctx:
        ctx.prec=9;ctx.rounding=ROUND_UP
        assert precise.potential(**args)==expected
        assert ctx.prec==9 and ctx.rounding==ROUND_UP
    assert canonical(args)==before and getcontext().prec==context.prec
    calls=[]
    def cancel():
        calls.append(True)
        if len(calls)==2:raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):precise.potential(**args,check=cancel)
    assert canonical(args)==before and getcontext().prec==context.prec


@pytest.mark.parametrize('plastic',(False,True))
def test_native_finite_variations_and_material_origin(plastic):
    source=model(2).mesh.elements[1].operator
    mixing=np.eye(6);mixing[0,3]=.13;mixing[1,5]=-.09;mixing[2,4]=.08
    section=EllipsoidalGeneralizedSection(mixing.T@np.diag([20.,12.,13.,2.,3.,4.])@mixing,
        np.eye(6),.001 if plastic else 1e6,1.)
    core=RetainedGeneralizedOperator(source.reference,section,order=4)
    successor=RetainedGeneralizedOperator(source.reference,section,order=4,arithmetic_policy=precise.POLICY)
    assert successor.identity!=core.identity and successor.cell.identity==core.cell.identity
    ref=core.reference;positions=ref.coordinates.copy();positions[1]+=[.003,-.004,.002]
    low=np.zeros((3,3));q=np.array([Rotation.from_rotvec(v).as_matrix()@f for v,f in
        zip(([.02,-.01,.03],[.13,.07,-.02],[-.04,.02,.09]),ref.nodal_triads)])
    u=np.array([Rotation.from_rotvec(v).as_matrix() for v in ([.02,.01,-.03],[-.03,.04,.02])])
    p=.01*np.sin(np.arange(18)+.7);origin=core.cell.virgin();before=canonical(origin)
    direction=.01*np.cos(np.arange(42)+.2);values={};residuals={};base=None
    for h in (0.,1e-4,-1e-4,5e-5,-5e-5):
        delta=h*direction;r=core.evaluate(positions,low,q,u,p,origin=origin,increment=delta)
        material=json.loads(r.material)['potential']
        values[h]=precise.potential(ref.coordinates,ref.nodal_triads,positions,low,q,u,delta,
            p+delta[24:],[material[0][0],material[1][0]])
        corrected=successor.evaluate(positions,low,q,u,p,origin=origin,increment=delta)
        assert corrected.potential==values[h]
        for key in ('residual','hessian','hessian_low','kinematics'):
            np.testing.assert_array_equal(getattr(corrected,key),getattr(r,key))
        assert corrected.material==r.material and canonical(corrected.history)==canonical(r.history)
        assert abs(values[h]-r.potential)<1e-11
        residuals[h]=r.residual
        if h==0.:base=r
        assert canonical(origin)==before
    work=float(direction@(base.hessian+base.hessian_low)@direction)
    for h in (1e-4,5e-5):
        first=(values[h]-values[-h])/(2*h)
        second=(values[h]+values[-h]-2*values[0.])/(h*h)
        assert abs(first-float(base.residual@direction))<1e-7
        assert abs(second-work)<1e-7
        assert np.linalg.norm((residuals[h]-residuals[-h])/(2*h)-(base.hessian+base.hessian_low)@direction)<1e-7
    assert any(sum(s.accumulated)>0 for s in base.history.stations)==plastic


def test_explicit_arithmetic_identity_and_mutation():
    source=model(2).mesh.elements[1]
    from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement as Element
    old=Element(1,source.node_ids,source.operator.reference,source.section)
    new=Element(1,source.node_ids,source.operator.reference,source.section,arithmetic_policy=precise.POLICY)
    assert old.identity==source.identity and old.to_dict()==source.to_dict()
    assert new.identity!=old.identity and new.to_dict()['arithmetic_policy']==precise.POLICY
    with pytest.raises(ValueError):Element(1,source.node_ids,source.operator.reference,source.section,arithmetic_policy='unregistered')
    object.__setattr__(new.operator,'arithmetic_policy',None)
    with pytest.raises(ValueError,match='arithmetic policy'):new.operator.guard()
