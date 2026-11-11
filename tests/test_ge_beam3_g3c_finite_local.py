"""Remaining B2/B3 local evidence, never a replacement for graph qualification."""
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
import pytest
from anyfesolver._ge_beam3_g3c_local_beam import LocalBeam
from test_ge_beam3_g3c_local_beam import SECTION, exp, err, equal, state

ROOT=Path(__file__).resolve().parents[1]
PROGRAM=json.loads((ROOT/'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json').read_bytes())['programs']

def zeros(m,n): return np.full((m,n),F(0),dtype=object)
def rref(a):
    a=a.copy(); row=0; pivots=[]
    for col in range(a.shape[1]):
        pivot=next((i for i in range(row,len(a)) if a[i,col]),None)
        if pivot is None: continue
        a[[row,pivot]]=a[[pivot,row]]; a[row]=a[row]/a[row,col]
        for i in range(len(a)):
            if i!=row: a[i]=a[i]-a[i,col]*a[row]
        pivots.append(col); row+=1
        if row==len(a): break
    return a,pivots
def inverse(a):
    size=len(a); identity=zeros(size,size)
    for i in range(size): identity[i,i]=F(1)
    reduced,pivots=rref(np.column_stack((a,identity)))
    assert pivots==list(range(size))
    return reduced[:,size:]

def exact_reference(n,length):
    """Rational source equations; no production stiffness/recovery helper."""
    L=F(length); S=[F(100),F(32),F(28),F(4),F(20),F(30)]
    if n==2:
        A=zeros(6,12); A[0,0]=-1; A[0,6]=1; A[1,3]=-1; A[1,9]=1
        for row,rotation in ((2,4),(3,10)):
            A[row,2]=-1/L; A[row,8]=1/L; A[row,rotation]=F(1)
        for row,rotation in ((4,5),(5,11)):
            A[row,1]=1/L; A[row,7]=-1/L; A[row,rotation]=F(1)
        H0=zeros(6,6); H1=zeros(6,6); H0[0,0]=H0[3,1]=F(1)
        H0[1,4:6]=-1/L; H0[2,2:4]=1/L
        H0[4,2:4]=[F(-1,2),F(1,2)]; H0[5,4:6]=[F(-1,2),F(1,2)]
        H1[4,2:4]=[F(1,2),F(1,2)]; H1[5,4:6]=[F(1,2),F(1,2)]
        compliance=np.diag([1/s for s in S])
        flex=L*(H0.T@compliance@H0+F(1,3)*H1.T@compliance@H1)
        K=A.T@inverse(flex)@A
    else:
        shape=np.array([[F(0),F(1),F(0)],[F(-1,2),F(0),F(1,2)],
                        [F(1,2),F(-1),F(1,2)]],dtype=object)
        derivative=np.array([2*shape[1]/L,4*shape[2]/L,[F(0)]*3],dtype=object)
        B=[zeros(6,18) for _ in range(3)]
        for power in range(3):
            for i in range(3):
                b=6*i; d=derivative[power,i]; s=shape[power,i]
                for j in range(6): B[power][j,b+j]=d
                B[power][1,b+5]=-s; B[power][2,b+4]=s
        K=zeros(18,18)
        for p in range(3):
            for q in range(3):
                if (p+q)%2==0: K+=L/F(p+q+1)*(B[p].T@np.diag(S)@B[q])
    R=zeros(6*n,6)
    for i in range(n):
        x=L*F(i,n-1)
        for j in range(3): R[6*i+j,j]=F(1); R[6*i+3+j,3+j]=F(1)
        R[6*i+1,5]=x; R[6*i+2,4]=-x
    return K,R

def make(n,length):
    ids=tuple(range(101,101+n)); coords=np.zeros((n,3)); coords[:,0]=np.linspace(0,length,n)
    return LocalBeam('B'+str(n),ids,coords,ids[0 if n==2 else 1],100,.25,SECTION)

@pytest.mark.parametrize('n',(2,3))
@pytest.mark.parametrize('length',(1,2))
def test_exact_reference_rank_rigids_complement_and_actual_operator(n,length):
    K,R=exact_reference(n,length); expected=6*n-6
    assert all(isinstance(v,(F,int)) for v in K.flat)
    assert np.array_equal(K,K.T) and not np.any(K@R)
    assert len(rref(K)[1])==expected and len(rref(R)[1])==6
    reduced,pivots=rref(R.T); free=[j for j in range(6*n) if j not in pivots]
    Z=zeros(6*n,len(free))
    for j,col in enumerate(free):
        Z[col,j]=F(1)
        for row,pivot in enumerate(pivots): Z[pivot,j]=-reduced[row,col]
    assert not np.any(R.T@Z)
    Q=Z.T@K@Z; lower=zeros(expected,expected); diagonal=[]
    for i in range(expected):
        lower[i,i]=F(1)
        pivot=Q[i,i]-sum(lower[i,k]**2*diagonal[k] for k in range(i))
        assert pivot>0; diagonal.append(pivot)
        for j in range(i+1,expected):
            lower[j,i]=(Q[j,i]-sum(lower[j,k]*lower[i,k]*diagonal[k] for k in range(i)))/pivot
    assert np.array_equal(lower@np.diag(diagonal)@lower.T,Q)
    trial=make(n,length).evaluate(np.zeros(6*n),np.tile(np.eye(3),(n,1,1)))
    equal(trial.chart_hessian,np.asarray(K,dtype=float)); equal(trial.local_tangent,np.asarray(K,dtype=float))
    equal(trial.chart_hessian@np.asarray(R,dtype=float),np.zeros((6*n,6)))

@pytest.mark.parametrize('n',(2,3))
@pytest.mark.parametrize('length',(1,2))
@pytest.mark.parametrize('component',range(6))
def test_component_energy_work_and_material_geometric_terms(n,length,component):
    p=make(n,length); x=np.array(p.descriptor()['coordinates'])[:,0]
    u=np.zeros((n,6)); q=np.tile(np.eye(3),(n,1,1))
    if component<3: u[:,component]=.025*x
    else:
        anchor=x[0 if n==2 else 1]
        q=np.array([exp(np.eye(3)[component-3]*(.025*(xi-anchor))) for xi in x])
    out=p.evaluate(u.ravel(),q); s=out.station_diagnostics
    energies=.5*np.einsum('s,si,si->i',s.weights,s.strains,s.resultants)
    assert np.isfinite(energies).all() and np.all(energies>=0)
    equal(sum(energies),s.energy)
    forces=np.einsum('s,sij,si->ij',s.weights,s.strain_differential,s.resultants)
    equal(forces.sum(axis=0),out.local_force)
    assert s.energy>0 and np.linalg.norm(out.local_force)>0

@pytest.mark.parametrize('n',(2,3))
@pytest.mark.parametrize('length',(1,2))
def test_nonzero_force_weighted_hessian_is_required(n,length):
    p=make(n,length); u,q=state(n); u=u.copy(); u.reshape(n,6)[:,:3]*=length
    out=p.evaluate(u,q); D=out.kinematics.differential
    omitted=D.T@out.local_tangent@D; full=out.chart_hessian
    assert err(omitted,full)>1e-7
    v=np.cos(np.arange(6*n)+.7); v/=np.linalg.norm(v)
    for h in PROGRAM['directional_steps']:
        fd=(p.evaluate(u+h*v,q).chart_force-p.evaluate(u-h*v,q).chart_force)/(2*h)
        assert err(full@v,fd)<=1e-7
    equal(full-omitted,np.einsum('i,ijk->jk',out.local_force,out.kinematics.second))

@pytest.mark.parametrize('n',(2,3))
@pytest.mark.parametrize('length',(1,2))
@pytest.mark.parametrize('scale',PROGRAM['force_scales'])
def test_registered_dead_force_scales_and_rigid_work(n,length,scale):
    # Local external-work check only. The graph's 5-stage equilibrium histories
    # remain required later; these tests do not relabel scaled poses as solves.
    p=make(n,length); u,q=state(n); ref=np.array(p.descriptor()['coordinates'])
    force=np.zeros((n,6)); force[-1,:3]=np.array(PROGRAM['force'])*scale
    v=np.sin(np.arange(6*n)+.8); v/=np.linalg.norm(v)
    def energy(z): return p.evaluate(z,q).station_diagnostics.energy-float(force.ravel()@z)
    gradient=p.evaluate(u,q).chart_force-force.ravel()
    for h in PROGRAM['directional_steps']:
        assert err((energy(u+h*v)-energy(u-h*v))/(2*h),gradient@v)<=1e-7
    current=np.array([exp(row[3:])@qi for row,qi in zip(u.reshape(n,6),q)])
    baseline=p.evaluate(u,q)
    for vector in PROGRAM['common_rotation_vectors']:
        W=exp(np.array(vector)); z=np.zeros((n,6))
        z[:,:3]=(ref+u.reshape(n,6)[:,:3])@W.T+[2,-3,1]-ref
        transformed=p.evaluate(z.ravel(),np.array([W@qi for qi in current]))
        equal(transformed.station_diagnostics.energy,baseline.station_diagnostics.energy)
        transformed_force=force[:,:3]@W.T; virtual=v.reshape(n,6)[:,:3]@W.T
        equal(np.sum(transformed_force*virtual),np.sum(force[:,:3]*v.reshape(n,6)[:,:3]))
