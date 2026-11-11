"""Finite area work/actual-update consequences; no owner qualification."""
import numpy as np
import pytest
from scipy import sparse

from anyfesolver import _stiffened_shell_distributing_footprint as reference
from anyfesolver._stiffened_shell_distributing_pose import (
    FiniteDistributingPose, DistributingPoseDomainError,
)
from anyfesolver._ge_beam3_mixed_ad import rotation_exponential as exp
from anyfesolver._ge_beam3_mixed_ad import RotationDomainError
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver.control import CancellationToken, SolveCancelled


def fixture(nx=2,ny=1):
    xyz=np.array([(2*i/nx,j/ny,0.) for j in range(ny+1) for i in range(nx+1)])
    hosts=tuple(reference.CartesianQ4Host(1+j*nx+i,(1+j*(nx+1)+i,2+j*(nx+1)+i,
        2+(j+1)*(nx+1)+i,1+(j+1)*(nx+1)+i)) for j in range(ny) for i in range(nx)
        if min(2*(i+1)/nx,1.5)>max(2*i/nx,.5) and min((j+1)/ny,.75)>max(j/ny,.25))
    selected=sorted({node for host in hosts for node in host.node_ids})
    xyz=xyz[np.array(selected)-1]
    nodes=tuple(reference.FlatQ4Node(node,tuple(x),6*i) for i,(node,x) in enumerate(zip(selected,xyz)))
    beam=reference.PhysicalBeamNodePort(1001,(1.3,.3,.5),6*len(nodes),reference.IDENTITY_AXES)
    definition=reference.DistributingFootprintDefinition("pad",tuple(h.element_id for h in hosts),
        1001,(.5,1.5,.25,.75))
    compiled=reference.compile_footprint(definition,nodes,hosts,beam)
    pose=FiniteDistributingPose(compiled)
    return compiled,pose,xyz,np.tile(np.eye(3),(len(xyz),1,1)),np.array(beam.coordinates_m),np.eye(3)


def deform(x,Q,rb,Qb):
    F=np.array(((1.03,.12,.02),(-.07,.98,-.03),(.14,-.09,1.)))
    x=x@F.T+.03*np.array((.2,-.3,.8))
    Q=np.array([exp(np.array((.15+.02*i,-.09+.01*i,.12)))@q for i,q in enumerate(Q)])
    return x,Q,rb+np.array((.01,-.02,.03)),exp(np.array((.07,-.08,.16)))@Qb


def advance(state,delta):
    x,Q,rb,Qb=state;n=len(x);delta=np.asarray(delta).reshape(n+1,6)
    return x+delta[:n,:3],np.array([exp(d[3:])@q for d,q in zip(delta[:n],Q)]),\
        rb+delta[n,:3],exp(delta[n,3:])@Qb


def norm_error(a,b):
    return np.linalg.norm(a-b)/max(1.,np.linalg.norm(b))


@pytest.mark.parametrize("grid",[(2,1),(4,2),(8,4)])
def test_reference_value_and_all_columns_reproduce_existing_area_map(grid):
    c,p,*state=fixture(*grid);lam=np.array((1.,-2.,3.,-.4,.5,.6))
    r=p.evaluate(*state,lam)
    assert np.max(np.abs(r.constraints))<1e-14
    assert norm_error(r.constraint_jacobian,c.constraint_matrix)<1e-14
    assert norm_error(r.residual,c.transpose_work(lam))<1e-14
    assert not r.owner_binding_authorized
    assert p.descriptor()['macro_ad_variables']==18
    assert sparse.isspmatrix_csr(r.chart_hessian)


@pytest.mark.parametrize("angle",[(.7,-.4,.2),(2.4,.6,-.8),(-1.2,1.3,.9)])
def test_common_large_rotation_translation_objectivity_and_spatial_wrenches(angle):
    c,p,*initial=fixture();state=deform(*initial);lam=np.array((2.,-3.,1.,.4,-.7,.2))
    r=p.evaluate(*state,lam);G=exp(angle);shift=np.array((10.,-7.,3.))
    x,Q,rb,Qb=state
    moved=(x@G.T+shift,np.einsum('ij,njk->nik',G,Q),G@rb+shift,G@Qb)
    s=p.evaluate(*moved,lam)
    assert norm_error(s.constraints,r.constraints)<2e-13
    assert norm_error(s.shell_frame,G@r.shell_frame)<1e-13
    expected=r.residual.reshape(-1,6).copy()
    expected[:,:3]=expected[:,:3]@G.T;expected[:,3:]=expected[:,3:]@G.T
    assert norm_error(s.residual,expected.ravel())<3e-13
    q=s.residual.reshape(-1,6);positions=np.vstack((moved[0],moved[2]))
    assert np.linalg.norm(q[:,:3].sum(axis=0))<1e-12
    assert np.linalg.norm((np.cross(positions,q[:,:3])+q[:,3:]).sum(axis=0))<1e-12


def test_physical_drill_invariance_at_tilted_directors_and_null_director_couples():
    c,p,*initial=fixture();state=deform(*initial);x,Q,rb,Qb=state
    lam=np.array((1.,2.,-1.,.7,-.3,.9));r=p.evaluate(*state,lam)
    drilled=np.array([q@exp((0.,0.,.3+.1*i)) for i,q in enumerate(Q)])
    s=p.evaluate(x,drilled,rb,Qb,lam)
    assert np.array_equal(s.constraints,r.constraints)
    assert np.array_equal(s.residual,r.residual)
    assert np.max(np.abs(np.einsum('ni,ni->n',r.residual.reshape(-1,6)[:-1,3:],Q[:,:,2])))<1e-13
    # Global rz at a tilted state is not rotation about the current director.
    assert np.max(np.abs(r.residual.reshape(-1,6)[:-1,5]))>1e-4


def test_independent_director_and_transverse_surface_slope_remain_distinct():
    c,p,x,Q,rb,Qb=fixture()
    sloped=x.copy();sloped[:,2]=.2*(x[:,0]-1.)+.3*(x[:,1]-.5)
    r=p.evaluate(sloped,Q,rb,Qb,np.ones(6))
    assert np.allclose(r.shell_frame,np.eye(3),rtol=0,atol=1e-14)
    tilted=np.tile(exp((.2,-.1,0.)),(len(x),1,1))
    s=p.evaluate(x,tilted,rb,Qb,np.ones(6))
    expected=tilted[0,:,2]
    assert np.allclose(s.shell_frame[:,2],expected,rtol=0,atol=1e-14)
    assert abs(float(s.shell_frame[2,0]))>.05
    assert not np.allclose(s.shell_frame,r.shell_frame)


def test_polar_frame_matches_independent_thin_polar_and_handles_repeated_singular_values():
    c,p,*initial=fixture();state=deform(*initial)
    r=p.evaluate(*state,np.ones(6));x,Q,rb,Qb=state
    h=c.area_mean@Q[:,:,2];n=h/np.linalg.norm(h)
    B=x.T@c.planar_gradients.T;T=B-n[:,None]*(n@B)
    values,vectors=np.linalg.eigh(T.T@T)
    thin=T@(vectors@np.diag(values**-.5)@vectors.T)
    assert np.allclose(r.shell_frame,np.column_stack((thin,n)),atol=2e-14,rtol=0)
    identity=p.evaluate(*initial,np.ones(6)).shell_frame
    assert np.allclose(identity,np.eye(3),atol=1e-14,rtol=0)


@pytest.mark.parametrize('epsilon',[1e-10,1e-12])
def test_normal_dominated_gradient_pair_is_rejected_even_with_unit_projected_condition(epsilon):
    # T has equal singular values, but projection relative to the input loses
    # significance. A relative T condition number alone cannot admit this case.
    c,p,x,Q,rb,Qb=fixture()
    G=exp((.5,-.7,.3));n=G[:,2]
    a=n+epsilon*G[:,0];b=n+epsilon*G[:,1]
    x=x[:,0,None]*a+x[:,1,None]*b
    Q[:]=G
    with pytest.raises(DistributingPoseDomainError,match='projected'):
        p.evaluate(x,Q,rb,Qb,np.ones(6))


def test_tiny_but_representable_normalization_does_not_overflow_constant_reciprocal_powers():
    c,p,x,Q,rb,Qb=fixture();x[:,:2]*=1e-140
    r=p.evaluate(x,Q,rb,Qb,np.ones(6))
    assert np.isfinite(r.constraint_jacobian).all()
    assert np.isfinite(r.chart_hessian.data).all()
    assert np.allclose(r.shell_frame,np.eye(3),atol=1e-14,rtol=0)


def test_unrepresentable_normalization_and_aggregate_fail_with_typed_domain_details():
    c,p,x,Q,rb,Qb=fixture();tiny=x.copy();tiny[:,:2]*=1e-310
    with pytest.raises(DistributingPoseDomainError,match='normalization'):
        p.evaluate(tiny,Q,rb,Qb,np.ones(6))
    huge=x.copy();huge[0,0]=np.finfo(float).max;huge[1,0]=-np.finfo(float).max
    with pytest.raises(DistributingPoseDomainError,match='aggregate'):
        p.evaluate(huge,Q,rb,Qb,np.ones(6))
    shifted=x.copy();shifted[:,0]=-np.finfo(float).max
    low=x.copy();low[:,0]=x[:,0]
    far=rb.copy();far[0]=np.finfo(float).max
    with pytest.raises(DistributingPoseDomainError,match='gap') as error:
        p.evaluate(shifted,Q,far,Qb,np.ones(6),shell_position_low=low)
    assert isinstance(error.value.__cause__,OverflowError)


def test_admitted_normal_dominated_gradients_still_produce_a_proper_physical_frame():
    c,p,x,Q,rb,Qb=fixture();G=exp((.5,-.7,.3));n=G[:,2];epsilon=1e-7
    x=x[:,0,None]*(n+epsilon*G[:,0])+x[:,1,None]*(n+epsilon*G[:,1]);Q[:]=G
    r=p.evaluate(x,Q,rb,Qb,np.ones(6))
    assert np.linalg.norm(r.shell_frame.T@r.shell_frame-np.eye(3))<1e-13
    assert abs(np.linalg.det(r.shell_frame)-1.)<1e-13
    assert np.linalg.norm(r.shell_frame-G)<1e-8


def test_existing_relative_log_chart_boundary_is_preserved():
    c,p,x,Q,rb,Qb=fixture()
    with pytest.raises(RotationDomainError):p.evaluate(x,Q,rb,exp((0.,0.,.91*np.pi)),np.ones(6))


def test_nonzero_multiplier_constraints_work_hessian_and_spatial_residual_derivatives():
    c,p,*initial=fixture();state=deform(*initial)
    lam=np.array((1.2,-.8,2.1,.7,-.4,.9));r=p.evaluate(*state,lam)
    rng=np.random.default_rng(917);direction=rng.normal(size=r.residual.size);direction/=np.linalg.norm(direction)
    def pulled_gradient(value,increment):
        q=value.residual.reshape(-1,6).copy()
        for i,d in enumerate(increment.reshape(-1,6)):q[i,3:]=_exp_terms(d[3:])[1].T@q[i,3:]
        return q.ravel()
    for h in (2e-4,2e-5,2e-6):
        s=p.evaluate(*advance(state,h*direction),lam);t=p.evaluate(*advance(state,-h*direction),lam)
        assert norm_error((s.constraints-t.constraints)/(2*h),r.constraint_jacobian@direction)<1e-7
        assert abs((s.potential-t.potential)/(2*h)-float(r.residual@direction))<1e-8
        assert norm_error((s.residual-t.residual)/(2*h),r.spatial_jacobian@direction)<1e-7
        difference=(pulled_gradient(s,h*direction)-pulled_gradient(t,-h*direction))/(2*h)
        assert norm_error(difference,r.chart_hessian@direction)<1e-7
    h=2e-5;s=p.evaluate(*advance(state,h*direction),lam);t=p.evaluate(*advance(state,-h*direction),lam)
    second=(s.potential-2*r.potential+t.potential)/h**2
    assert abs(second-float(direction@(r.chart_hessian@direction)))<2e-5
    assert sparse.linalg.norm(r.chart_hessian-r.chart_hessian.T)<1e-12
    assert sparse.linalg.norm(r.spatial_jacobian-r.spatial_jacobian.T)>1e-2


def test_actual_additive_shell_chart_columns_and_curvature_pullback():
    c,p,x,Q,rb,Qb=fixture();n=len(x)
    theta=np.array([(.2+.01*i,-.15,.12) for i in range(n)])
    anchors=np.array([exp((-.04,.02,.08)) for i in range(n)])
    state=(x,np.array([_exp_terms(t)[0]@a for t,a in zip(theta,anchors)]),rb,exp((.1,-.05,.08)))
    lam=np.array((1.,2.,3.,.2,-.3,.4));r=p.evaluate(*state,lam);size=r.residual.size
    A=np.eye(size);extra=np.zeros((size,size))
    coordinates=np.vstack((theta,np.zeros(3)))
    for i,t in enumerate(coordinates):
        _,a,da=_exp_terms(t);start=6*i+3;A[start:start+3,start:start+3]=a
        for k in range(3):extra[start:start+3,start+k]=da[:,:,k].T@r.residual[start:start+3]
    H=A.T@r.spatial_jacobian@A+extra
    assert np.linalg.norm(H-H.T)<1e-12
    d=np.random.default_rng(51).normal(size=size);d/=np.linalg.norm(d);step=1e-5
    def chart_residual(alpha):
        dx=(alpha*d).reshape(n+1,6);frames=np.array([_exp_terms(t+u[3:])[0]@a for t,u,a in zip(theta,dx,anchors)])
        z=p.evaluate(x+dx[:n,:3],frames,rb+dx[n,:3],exp(dx[n,3:])@state[3],lam)
        current=np.eye(size)
        for i,t in enumerate(theta+dx[:n,3:]):current[6*i+3:6*i+6,6*i+3:6*i+6]=_exp_terms(t)[1]
        current[6*n+3:6*n+6,6*n+3:6*n+6]=_exp_terms(dx[n,3:])[1]
        return current.T@z.residual
    assert norm_error((chart_residual(step)-chart_residual(-step))/(2*step),H@d)<1e-8
    # The existing owner retains spatial rows and pulls only increment columns.
    mixed=r.spatial_jacobian@A
    def mixed_rows(alpha):
        dx=(alpha*d).reshape(n+1,6);frames=np.array([_exp_terms(t+u[3:])[0]@a for t,u,a in zip(theta,dx,anchors)])
        return p.evaluate(x+dx[:n,:3],frames,rb+dx[n,:3],exp(dx[n,3:])@state[3],lam).residual
    assert norm_error((mixed_rows(step)-mixed_rows(-step))/(2*step),mixed@d)<1e-8


@pytest.mark.parametrize("kind",['director_cancel','rank_collapse','reverse','ninety_degree_shear'])
def test_explicit_physical_domain_failures(kind):
    c,p,x,Q,rb,Qb=fixture()
    if kind=='director_cancel':
        Q[[0,1,2]]=exp((np.pi,0.,0.))
    elif kind=='rank_collapse':x[:,1]=0.
    elif kind=='reverse':x[:,1]*=-1.
    else:Q[:]=exp((0.,np.pi/2,0.))
    with pytest.raises(DistributingPoseDomainError):p.evaluate(x,Q,rb,Qb,np.ones(6))


def test_compensated_positions_ownership_and_cancellation():
    c,p,x,Q,rb,Qb=fixture();lam=np.ones(6)
    from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
    shift=np.array((1e12,-2e12,3e12));correction=np.array((1e-9,-2e-9,3e-9))
    low=np.empty_like(x);bl=np.empty_like(rb)
    for i,j in np.ndindex(x.shape):x[i,j],low[i,j]=split_sum((float(x[i,j]),float(shift[j])))
    for j in range(3):rb[j],bl[j]=split_sum((float(rb[j]),float(shift[j]),float(correction[j])))
    r=p.evaluate(x,Q,rb,Qb,lam,shell_position_low=low,beam_position_low=bl)
    assert np.max(np.abs(r.constraints[:3]-correction))<1e-14
    x[:]=0.;Q[:]=0.;rb[:]=0.;lam[:]=0.;low[:]=0.;bl[:]=0.
    for v in (r.constraints,r.residual,r.shell_frame,r.chart_hessian.data):
        with pytest.raises(ValueError):v.setflags(write=True)
    with pytest.raises(AttributeError):p.identity='changed'
    _,_,x,Q,rb,Qb=fixture();token=CancellationToken();token.cancel('private candidate only')
    with pytest.raises(SolveCancelled):p.evaluate(x,Q,rb,Qb,np.ones(6),cancellation_token=token)
    with pytest.raises(ValueError):p.evaluate(x.astype(np.float32),Q,rb,Qb,np.ones(6))
