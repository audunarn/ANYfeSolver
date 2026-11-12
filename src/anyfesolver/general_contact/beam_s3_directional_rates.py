"""Certified full-line one-sided penalty velocity for the affine skin profile."""
from fractions import Fraction as Q
from dataclasses import replace
from contextlib import contextmanager
import numpy as np
from .beam_s3_rotational_transient import RotationalBeamS3Manager, recover_rates
from .runtime import _ACTIVE_CONTACT

PROFILE='ELASTIC_B2_B3_AFFINE_S3_DIRECTIONAL_LINE_HERMITE_DEV_V1'
RESTART_SCHEMA='ANYSOLVER_BEAM_S3_DIRECTIONAL_LINE_DEV_CHECKPOINT_V19'


class DirectionalBeamS3Manager(RotationalBeamS3Manager):
    def restart_contract(self):
        data=super().restart_contract()
        data.update(profile=PROFILE,revision='EXACT_FULL_LINE_DIRECTIONAL_PENALTY_V1')
        return data

    def evaluate(self,*args,**kwargs):
        r,K,summary=super().evaluate(*args,**kwargs)
        return r,K,replace(summary,profile=PROFILE)


@contextmanager
def directional_scope(model,definition,settings,initial_displacement=None):
    manager=DirectionalBeamS3Manager(model,definition,settings,initial_displacement)
    token=_ACTIVE_CONTACT.set(manager)
    try:yield manager
    finally:_ACTIVE_CONTACT.reset(token)


def exact_gap_polynomial(manager,u):
    """Frozen binary chart in exact rational arithmetic; no nearzero snapping."""
    manager.guard()
    A=manager.work_map.toarray()
    refs=np.r_[manager.beam_reference.ravel(),manager.shell.reference_positions.ravel()]
    values=np.asarray([Q(float(ref))+sum(Q(float(a))*Q(float(x)) for a,x in zip(row,u)) for ref,row in zip(refs,A)],object).reshape(-1,3)
    nb=len(manager.beam_ids);beam=values[:nb];skin=values[nb:]
    C=np.cross(skin[1]-skin[0],skin[2]-skin[0]);C2=C@C
    if C2<=0 or C@np.asarray([Q(float(x)) for x in manager.kernel.reference_normal])<=0:
        raise ValueError('directional face orientation unresolved')
    if nb==2:poly=np.array((beam[0]-skin[0],beam[1]-beam[0],np.zeros(3,dtype=object)),object)
    else:poly=np.array((beam[0]-skin[0],-3*beam[0]+4*beam[1]-beam[2],2*beam[0]-4*beam[1]+2*beam[2]),object)
    orientation=Q(float(manager.kernel.orientation));heights=np.array([orientation*(p@C) for p in poly],object)
    radius=Q(float(manager.kernel.radius))
    def zero(value):return value>0 and value*value==radius*radius*C2
    full=heights[1]==0 and heights[2]==0 and zero(heights[0])
    candidates=[Q(0),Q(1)]
    if heights[2]>0:
        vertex=-heights[1]/(2*heights[2])
        if 0<vertex<1:candidates.append(vertex)
    minimum=min(sum(c*s**i for i,c in enumerate(heights)) for s in candidates)
    isolated=not full and zero(minimum)
    return full,isolated,A,skin,poly,C,C2,orientation


def directional_coefficients(manager,u):
    """Return rational numerator B(s) and epsilon*L/|C|^2.

    delta g = orientation/|C| * (delta e.C + e.delta C
    - (e.C)*(C.delta C)/|C|^2). This includes normal variation and the
    formulation-owned eccentric rotation map. The common sqrt cancels in D/H.
    """
    full,_,A,skin,poly,C,C2,orientation=exact_gap_polynomial(manager,u)
    if not full:raise ValueError('directional branch needs genuinely exact full-line zero gap')
    nb=len(manager.beam_ids);n=A.shape[1];B=np.empty((3,n),object)
    maps=np.asarray([[Q(float(x)) for x in row] for row in A],object).reshape(nb+3,3,n)
    for j in range(n):
        sb=maps[nb:,:,j];bb=maps[:nb,:,j]
        dC=np.cross(sb[1]-sb[0],skin[2]-skin[0])+np.cross(skin[1]-skin[0],sb[2]-sb[0])
        if nb==2:de=np.array((bb[0]-sb[0],bb[1]-bb[0],np.zeros(3,dtype=object)),object)
        else:de=np.array((bb[0]-sb[0],-3*bb[0]+4*bb[1]-bb[2],2*bb[0]-4*bb[1]+2*bb[2]),object)
        for i in range(3):B[i,j]=orientation*(de[i]@C+poly[i]@dC-(poly[i]@C)*(C@dC)/C2)
    return B,Q(float(manager.kernel.stiffness))*Q(float(manager.kernel.length))/C2


def _negative_intervals(coefficients):
    """Complete quadratic inventory with finite exact sign/root brackets."""
    c=list(coefficients)
    f=lambda s:c[0]+s*(c[1]+s*c[2])
    cuts=[Q(0),Q(1)]
    if c[2]:
        vertex=-c[1]/(2*c[2])
        if 0<vertex<1:cuts.insert(1,vertex)
    roots=[];uncertainty=Q(0)
    for left,right in zip(cuts,cuts[1:]):
        a,b=f(left),f(right)
        if a==0:roots.append(left)
        if b==0:roots.append(right)
        if a*b<0:
            lo,hi=left,right
            for _ in range(56):
                mid=(lo+hi)/2;value=f(mid)
                if value==0:lo=hi=mid;break
                if f(lo)*value<0:hi=mid
                else:lo=mid
            roots.append((lo+hi)/2);uncertainty+=hi-lo
    stations=sorted(set([Q(0),Q(1)]+roots))
    return [(a,b) for a,b in zip(stations,stations[1:]) if f((a+b)/2)<0],uncertainty


def directional_operator(B,factor,velocity):
    """Degree-four polynomial integral; bounded root-strip error returned."""
    B=np.asarray(B,object);v=np.array([Q(float(x)) for x in velocity],object)
    coefficients=B@v;intervals,width=_negative_intervals(coefficients)
    moments=np.array([[sum((b**(i+j+1)-a**(i+j+1))/Q(i+j+1) for a,b in intervals) for j in range(3)] for i in range(3)],object)
    exactH=factor*(B.T@moments@B);exactD=exactH@v
    boundB=max(sum(abs(x) for x in B[:,j]) for j in range(B.shape[1]))
    boundRate=sum(abs(x) for x in coefficients)
    errorD=abs(factor)*width*boundB*boundRate
    errorH=abs(factor)*width*boundB*boundB
    D=np.array(exactD,float);H=np.array(exactH,float)
    # Account conversion before the float Newton residual is evaluated.
    errorD=float(errorD)+16*np.finfo(float).eps*max(1.,np.max(abs(D)))
    errorH=float(errorH)+16*np.finfo(float).eps*max(1.,np.max(abs(H)))
    return D,H,errorD,errorH


def _null_error_factor(Z):
    """Outward Euclidean bound for Z.T times a component error box."""
    sums=[sum(abs(Q(float(x))) for x in column) for column in np.asarray(Z).T]
    square=sum(x*x for x in sums)
    result=float(np.sqrt(float(square)))
    if not np.isfinite(result):raise ValueError('directional null error amplification nonfinite')
    while Q(result)*Q(result)<square:result=float(np.nextafter(result,np.inf))
    return result


def solve_directional_velocity(K,Z,physical,B,factor,*,maximum_iterations=20):
    """Safeguarded strongly monotone null-coordinate solve; no inertia change."""
    A=Z.T@K@Z
    np.linalg.cholesky((A+A.T)/2)
    coordinate=np.linalg.solve(A,-Z.T@K@physical)
    error_factor=_null_error_factor(Z)
    for _ in range(maximum_iterations):
        v=physical+Z@coordinate;D,H,eD,eH=directional_operator(B,factor,v)
        residual=Z.T@(K@v+D);scale=max(1.,np.linalg.norm(K@v),np.linalg.norm(D))
        error=float(np.nextafter(error_factor*eD,np.inf))
        if np.linalg.norm(residual)+error<=1.e-9*scale:return v
        change=np.linalg.solve(A+Z.T@H@Z,-residual)
        accepted=False
        for cut in range(12):
            trial=coordinate+2.**(-cut)*change;vt=physical+Z@trial
            Dt,_,et,_=directional_operator(B,factor,vt)
            norm=np.linalg.norm(Z.T@(K@vt+Dt))+float(np.nextafter(error_factor*et,np.inf))
            if norm<np.linalg.norm(residual)+error:
                coordinate=trial;accepted=True;break
        if not accepted:break
        last=residual
    raise ValueError('directional null velocity finite solve budget exhausted')


def recover_directional_rates(manager,full,T,M,basis,tangent,velocity,acceleration,previous_velocity,dt,summary):
    full_touch,isolated,*_=exact_gap_polynomial(manager,full)
    if not full_touch and not isolated:
        return recover_rates(T,M,basis,tangent,velocity,acceleration,previous_velocity,dt,summary)
    # The old floating kernel can select a closed interval at exact geometric
    # touch. Remove that H before using the certified one-sided law (or zero H
    # for isolated point touch, whose distributed penalty tangent is continuous).
    _,contactH,_=manager.evaluate(full)
    structural=tangent-contactH
    Z=basis.reduced_basis.toarray();gram=Z.T@Z
    physical=lambda x:x-Z@np.linalg.solve(gram,Z.T@x)
    K=(T.T@structural@T).toarray();pv=physical(velocity)
    if full_touch:
        B,factor=directional_coefficients(manager,full)
        reduced=np.asarray(B@np.asarray([[Q(float(x)) for x in row] for row in T.toarray()],object),object)
        v=solve_directional_velocity(K,Z,pv,reduced,factor)
    else:v=pv+Z@np.linalg.solve(Z.T@K@Z,-Z.T@K@pv)
    a=physical(acceleration)+Z@np.linalg.solve(gram,Z.T@(v-previous_velocity)/dt)
    if not np.isfinite(v).all() or not np.isfinite(a).all() or np.linalg.norm(M@(a-acceleration))>1.e-10*max(1.,np.linalg.norm(M@acceleration)):
        raise ValueError('directional rates changed physical inertia or became nonfinite')
    return v,a


EQUILIBRATED_PROFILE='ELASTIC_B2_B3_AFFINE_S3_EQUILIBRATED_DIRECTIONAL_HERMITE_DEV_V1'
EQUILIBRATED_RESTART_SCHEMA='ANYSOLVER_BEAM_S3_EQUILIBRATED_DIRECTIONAL_DEV_CHECKPOINT_V20'


class EquilibratedDirectionalBeamS3Manager(DirectionalBeamS3Manager):
    def restart_contract(self):
        data=super().restart_contract()
        data.update(profile=EQUILIBRATED_PROFILE,initial_policy='ELASTIC_COMPLETE_EQUILIBRIUM_CONSTANT_DEADLOAD_V1')
        return data

    def evaluate(self,*args,**kwargs):
        r,K,summary=super().evaluate(*args,**kwargs)
        return r,K,replace(summary,profile=EQUILIBRATED_PROFILE)


@contextmanager
def equilibrated_directional_scope(model,definition,settings,initial_displacement=None):
    manager=EquilibratedDirectionalBeamS3Manager(model,definition,settings,initial_displacement)
    token=_ACTIVE_CONTACT.set(manager)
    try:yield manager
    finally:_ACTIVE_CONTACT.reset(token)
