"""Supervised ordinary Log and connected material/state controls."""
import os,json,numpy as np,pytest
from pathlib import Path
pytestmark=pytest.mark.skipif('F06_LOG_OUTPUT' not in os.environ,reason='fresh ordinary-domain supervisor only')

@pytest.mark.parametrize('angle',[.0006,.4,1.7,2.7])
def test_control_full_coefficients_independent_series(angle):
    from scripts.r8_f06_log_domain_route_dev import coefficient_oracle
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import Pair,full_log_coefficients
    high=float(np.cos(angle));low=2.**-60
    got=full_log_coefficients(Pair(high,low));want=coefficient_oracle(high,low)
    np.testing.assert_allclose([p.scalar() for p in got],want,rtol=5e-13,atol=5e-13)

@pytest.mark.parametrize('low',[2.**-80,0.,-2.**-80])
def test_control_physical_pair_branch_and_cutoff(low,monkeypatch):
    import anyfesolver._stiffened_shell_f06_increment_arithmetic as a
    from anyfesolver._ge_beam3_mixed_ad import math_cos_limit,RotationDomainError
    c=a.Pair(1.-1e-7,low);calls=[];original=a.full_log_coefficients
    def capture(c):calls.append(c);return original(c)
    monkeypatch.setattr(a,'full_log_coefficients',capture)
    zero=a.Jet.constant(0.,1);m=[[zero,zero,zero],[zero,zero,zero],[zero,zero,zero]]
    m[0][0]=a.Jet.constant(2*c+1.,1);m[2][1]=a.Jet.constant(1.,1)
    a.log(m);assert bool(calls)==(low<=0.)
    edge=a.Pair(math_cos_limit(),low)
    if low>0.:original(edge)
    else:
        with pytest.raises(RotationDomainError):original(edge)

@pytest.mark.parametrize('angle',[.0003,.0006,.7,2.7])
def test_control_axis_equivalence_and_left_jacobian(angle):
    from scipy.spatial.transform import Rotation
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import Jet,exp,mm,const,log
    axis=np.array([2.,-1.,3.]);axis/=np.linalg.norm(axis);phi=angle*axis
    r=Rotation.from_rotvec(phi).as_matrix();w=[Jet.variable(0.,i,3) for i in range(3)]
    values=log(mm(exp(w),const(r,np.zeros((3,3)),3)))
    S=np.array([[0.,-phi[2],phi[1]],[phi[2],0.,-phi[0]],[-phi[1],phi[0],0.]])
    coefficient=(1.-.5*angle/np.tan(.5*angle))/angle**2
    expected=np.eye(3)-.5*S+coefficient*S@S
    np.testing.assert_allclose([v.value.scalar() for v in values],phi,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose([v.gradient.rounded() for v in values],expected,rtol=1e-10,atol=1e-11)
    opposite=Rotation.from_rotvec((-angle)*(-axis)).as_matrix();np.testing.assert_allclose(opposite,r,rtol=0.,atol=1e-15)

def test_control_off_so3_affine_gradient_hessian():
    from scripts.r8_f06_log_domain_route_dev import coefficient_oracle
    from scipy.spatial.transform import Rotation
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import Jet,Pair,log
    high=Rotation.from_rotvec([.4,-.2,.3]).as_matrix();high[1,1]+=3e-13
    low=np.zeros((3,3));low[0,0]=2.**-58
    D=np.array([[.7,.2,-.3],[-.1,-.4,.8],[.4,-.6,.2]])
    m=[[Jet.constant(Pair(high[i,j],low[i,j]),1)+Jet.variable(0.,0,1)*D[i,j] for j in range(3)] for i in range(3)]
    vals=log(m);c=(np.trace(high)-1.)/2.;cl=np.trace(low)/2.;f,fp,fpp=coefficient_oracle(c,cl)
    v=np.array([high[2,1]-high[1,2],high[0,2]-high[2,0],high[1,0]-high[0,1]])/2
    dv=np.array([D[2,1]-D[1,2],D[0,2]-D[2,0],D[1,0]-D[0,1]])/2;dc=np.trace(D)/2
    np.testing.assert_allclose([v.gradient.rounded()[0] for v in vals],fp*dc*v+f*dv,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose([v.hessian.rounded()[0,0] for v in vals],fpp*dc*dc*v+2*fp*dc*dv,rtol=1e-11,atol=1e-12)

def test_control_native_active_fixed_origin_spatial_work_and_objectivity():
    from scripts.r8_f06_log_domain_route_dev import definition
    from scipy.spatial.transform import Rotation
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import add_state,updated_rotation
    from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
    e,_=definition().beams[0].instantiate();op=e.operator;x=op.reference.coordinates.copy();x[2]+=[.03,.02,.01]
    q=Rotation.from_rotvec([[.4,-.1,.2],[.2,.3,-.1],[.7,.1,.2]]).as_matrix()
    u=Rotation.from_rotvec([[.1,.15,-.05],[.2,-.1,.15]]).as_matrix()
    p=np.array([20.,7.,3.,25.,-4.,2.,8.,5.,1.,9.,-6.,2.,10.,3.,-4.,12.,7.,2.]);pl=np.zeros(18)
    origin=op.cell.response((p*.4).tolist())['history'];captured=origin
    def evaluate(x,q,u,p,xl=None,ql=None,ul=None,pl=None):
        return op.evaluate(x,np.zeros_like(x) if xl is None else xl,q,u,p,origin=origin,
            vertices_low=ql,rotations_low=ul,resultants_low=pl)
    base=evaluate(x,q,u,p);assert origin==captured
    assert set(s['branch'] for s in json.loads(base.material)['stations'])=={'PLASTIC'}
    direction=np.linspace(-.03,.04,42);h=1e-5;ends=[]
    for sign in (-1,1):
        xx,xl=add_state(x,np.zeros_like(x),sign*h*direction[:18].reshape(3,6)[:,:3]);qs=[];qls=[];us=[];uls=[]
        for n in range(3):a,b=updated_rotation(q[n],np.zeros((3,3)),sign*h*direction[6*n+3:6*n+6]);qs.append(a);qls.append(b)
        for n in range(2):a,b=updated_rotation(u[n],np.zeros((3,3)),sign*h*direction[18+3*n:21+3*n]);us.append(a);uls.append(b)
        pp,pll=add_state(p,pl,sign*h*direction[24:]);r=evaluate(xx,np.array(qs),np.array(us),pp,xl,np.array(qls),np.array(uls),pll)
        assert set(s['branch'] for s in json.loads(r.material)['stations'])=={'PLASTIC'};ends.append(r.residual)
    predicted=spatial_jacobian(base.residual,base.hessian+base.hessian_low)@direction
    error=np.linalg.norm((ends[1]-ends[0])/(2*h)-predicted)/np.linalg.norm(predicted);assert error<=1e-6,error
    G=Rotation.from_rotvec([.5,-.2,.3]).as_matrix();moved=evaluate(x@G.T+np.array([2.,-1.,.3]),G@q,G@u,p)
    T=np.eye(42)
    for start in (0,3,6,9,12,15,18,21):T[start:start+3,start:start+3]=G
    np.testing.assert_allclose(moved.residual,T@base.residual,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(moved.kinematics,base.kinematics,rtol=1e-10,atol=1e-12)
    rows=op.recover(u,p,origin=origin,rotations_low=np.zeros_like(u),resultants_low=np.zeros_like(p));assert rows
    with pytest.raises(ValueError):op.recover(u,p,origin=origin,rotations_low=np.ones_like(u),resultants_low=np.zeros_like(p))
    assert origin==captured
    Path(os.environ['F06_LOG_OUTPUT'],'native-metrics.json').write_text(json.dumps(dict(error=error,branch='PLASTIC',ordinary_rotations=True,fixed_origin=True,objectivity=True)))

def test_route_public_active_ordinary_lifecycle():
    from scripts.r8_f06_log_domain_route_dev import public_route
    public_route(Path(os.environ['F06_LOG_OUTPUT']))


def test_control_point_ordinary_spatial_derivative_work_objectivity():
    from scipy.spatial.transform import Rotation
    from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import add_state,updated_rotation
    x0=np.array([[0.,0.,0.],[.3,-.2,.1]]);q0=np.tile(np.eye(3),(2,1,1))
    joint=RigidPoseJoint(x0,q0);x=x0+np.array([[.01,.02,0.],[-.01,.03,.04]])
    q=Rotation.from_rotvec([[.3,-.2,.1],[-.2,.4,.5]]).as_matrix();lam=np.array([2.,-3.,4.,7.,-5.,6.])
    base=joint.evaluate(x,q,lam,frames_low=np.zeros_like(q));d=np.linspace(-.2,.3,18);h=1e-5;ends=[]
    for sign in (-1,1):
        xx,xl=add_state(x,np.zeros_like(x),sign*h*d[:12].reshape(2,6)[:,:3]);qs=[];qls=[]
        for n in range(2):
            a,b=updated_rotation(q[n],np.zeros((3,3)),sign*h*d[6*n+3:6*n+6]);qs.append(a);qls.append(b)
        ends.append(joint.evaluate(xx,np.array(qs),lam+sign*h*d[12:],position_low=xl,frames_low=np.array(qls)).residual)
    predicted=base.spatial_jacobian@d
    error=np.linalg.norm((ends[1]-ends[0])/(2*h)-predicted)/np.linalg.norm(predicted)
    assert error<=1e-6,error
    assert np.linalg.norm(base.chart_hessian-base.spatial_jacobian)>1e-3
    assert abs(base.residual[:12]@d[:12]-lam@(base.constraint_jacobian[:,:12]@d[:12]))<1e-12
    G=Rotation.from_rotvec([.5,.3,-.4]).as_matrix();shift=np.array([2.,-1.,3.])
    moved=joint.evaluate(x@G.T+shift,G@q,lam,frames_low=np.zeros_like(q));T=np.eye(18)
    for start in (0,3,6,9):T[start:start+3,start:start+3]=G
    np.testing.assert_allclose(moved.constraints,base.constraints,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(moved.residual,T@base.residual,rtol=1e-11,atol=1e-12)
    Path(os.environ['F06_LOG_OUTPUT'],'point-metrics.json').write_text(json.dumps(dict(error=error,ordinary_rotation=True,moving_row_correction=True,objectivity=True)))


@pytest.mark.parametrize('angle',[.4,1.7,2.7])
def test_control_paired_coefficient_low_parts(angle):
    from decimal import Decimal,localcontext
    from scripts.r8_f06_log_domain_route_dev import coefficient_oracle
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import Pair,full_log_coefficients
    high=float(np.cos(angle));low=float(np.spacing(abs(high)))/4.
    expected=coefficient_oracle(high,low,decimal=True)
    actual=full_log_coefficients(Pair(high,low))
    with localcontext() as ctx:
        ctx.prec=80
        errors=[abs(Decimal.from_float(float(v.high))+Decimal.from_float(float(v.low))-e) for v,e in zip(actual,expected)]
        bound=64*Decimal.from_float(float(np.finfo(float).eps))**2
        assert all(e<=bound*max(Decimal(1),abs(v)) for e,v in zip(errors,expected)),errors
        dropped=full_log_coefficients(Pair(high))
        omitted=[abs(Decimal.from_float(float(v.high))+Decimal.from_float(float(v.low))-e) for v,e in zip(dropped,expected)]
        assert all(e>1000*bound*max(Decimal(1),abs(v)) for e,v in zip(omitted,expected)),omitted


def test_control_historical_compensated_checkpoint_encoding():
    # Serialization impact control only: this does not claim mechanical acceptance.
    from dataclasses import replace
    from types import SimpleNamespace
    from scripts.r8_f06_log_domain_route_dev import definition
    from anyfesolver import stiffened_shell as ss
    from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_compensated,decode
    from anyfesolver._stiffened_shell_q4_algorithmic_pullback import HISTORICAL_POLICY
    owner=ss.create_analysis(definition=replace(definition(),q4_pullback_policy=HISTORICAL_POLICY))
    try:
        driver=FinitePointDriver(owner);assert driver._compensated and driver._stable_q4
        owner._generation=1;driver._accepted=SimpleNamespace();driver._accepted_factors=(.1,);driver._pattern=np.zeros(owner._K.shape[0])
        raw=driver.checkpoint();state=driver._state
        shapes=tuple((n,getattr(state,n).shape) for n in ('q_low','frames_low','cells_low','resultants_low'))
        inner,_=decode_compensated(raw,owner._identity,shapes)
        base_shapes=tuple((n,v.shape) for n,v in (('q',state.q),('frames',state.frames),('cells',state.cells),
            ('resultants',state.resultants),('multipliers',state.multipliers),('pattern',driver._pattern),
            ('accepted_factors',np.asarray(driver._accepted_factors))))
        kind,generation,values=decode(inner,owner._identity,lambda k,g:base_shapes)
        assert kind=='finite_stable_q4' and generation==1
    finally:owner.close()
