"""Frozen private matrix-adapter inventory; bounded reviewed runner only."""
from dataclasses import fields
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
import pytest
import anyfesolver._ge_beam3_g3c_b2_physical_adapter as module

CONTRACT = json.loads((Path(__file__).resolve().parents[1]/
    'docs/reference_cases/ge_beam3_g3c_b2_physical_contract_v1.json').read_text())
SCIENTIFIC_RECORDS = []


def data(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if hasattr(value, '__dataclass_fields__'):
        return {f.name:data(getattr(value,f.name)) for f in fields(value)}
    return value


def evaluate(adapter, u, q):
    result = adapter.evaluate(u, q)
    SCIENTIFIC_RECORDS.append(dict(definition=adapter.descriptor(), displacement=u.tolist(),
                                   accepted_rotations=q.tolist(), result=data(result)))
    return result


def compare(a, b, tolerance=1e-11, relative=True):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    scale = max(float(np.max(abs(a))), float(np.max(abs(b))))
    if not relative: scale = max(1., scale)
    if scale == 0: return
    a, b = a/scale, b/scale
    assert np.linalg.norm(a-b) <= tolerance*max(1., np.linalg.norm(a), np.linalg.norm(b))


def exp(v):
    t = np.linalg.norm(v); k = np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
    if t == 0: return np.eye(3)
    return np.eye(3)+np.sin(t)/t*k+(1-np.cos(t))/(t*t)*(k@k)


def log(r):
    c = np.clip((np.trace(r)-1)/2, -1., 1.); t = np.arccos(c)
    v = np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])/2
    return v*(1+t*t/6) if t < 1e-6 else v*t/np.sin(t)


def fixtures():
    for row in CONTRACT['fixtures']:
        x = {k:float(F(v)) for k,v in row.items() if k != 'id'}
        section = dict(area=x['A'], Iy=x['Iy'], Iz=x['Iz'], J=x['J'],
                       shear_factor_y=x['ky'], shear_factor_z=x['kz'], orientation=[0.,0.,1.])
        ref = np.array([[0.,0.,0.],[x['L'],0.,0.]])
        yield row['id'], module.PhysicalB2Adapter((101,102),ref,101,x['E'],x['nu'],section)


def state():
    u = .013*np.sin(np.arange(12)+.4)
    q = np.array([exp(np.array([.4,-.2,.3])+.01*i*np.array([1.,2.,-1.])) for i in range(2)])
    return u, q


def oracle(adapter, u, qa):
    """Independent Rodrigues/Log values and rational six-force flexibility solve."""
    definition = adapter.descriptor(); ref = np.array(definition['coordinates'])
    x = ref+u.reshape(2,6)[:,:3]
    q = np.array([exp(row[3:])@a for row,a in zip(u.reshape(2,6),qa)])
    anchor = q[definition['node_ids'].index(definition['anchor_node'])]
    d = np.c_[(x-x.mean(axis=0))@anchor-(ref-ref.mean(axis=0)),
              np.array([log(anchor.T@a) for a in q])].ravel()
    axis = ref[1]-ref[0]; length = np.linalg.norm(axis); axis /= length
    z = np.array(definition['section']['orientation']); z -= (z@axis)*axis; z /= np.linalg.norm(z)
    frame = np.column_stack((axis,np.cross(z,axis),z)); T = np.kron(np.eye(4),frame.T)
    v = list(map(F,T@d)); l = F(float(length)); z0 = F(0)
    section = definition['section']; E=F(definition['E']); G=E/(2*(1+F(definition['nu'])))
    # Match the defined physical binary64 rigidity inputs, not target core internals.
    s = list(map(F,map(float,(E*F(section['area']),G*F(section['area'])*F(section['shear_factor_y']),
        G*F(section['area'])*F(section['shear_factor_z']),G*F(section['J']),
        E*F(section['Iy']),E*F(section['Iz'])))))
    H = [[z0]*6 for _ in range(6)]; H[0][0]=l/s[0]; H[1][1]=l/s[3]
    for a,ei,shear in ((2,s[4],s[2]),(4,s[5],s[1])):
        for i in range(2):
            for j in range(2): H[a+i][a+j]=l/ei*(F(1,3) if i==j else -F(1,6))+1/(shear*l)
    B = [[z0]*12 for _ in range(6)]; B[1][3]=-1; B[1][9]=1
    for i,r in ((2,4),(3,10)): B[i][2]=-1/l; B[i][8]=1/l; B[i][r]=1
    for i,r in ((4,5),(5,11)): B[i][1]=1/l; B[i][7]=-1/l; B[i][r]=1
    rhs=[sum(a*b for a,b in zip(row,v)) for row in B]
    rows=[row[:]+[b]+br[:] for row,b,br in zip(H,rhs,B)]
    for i in range(6):
        p=rows[i][i]; rows[i]=[a/p for a in rows[i]]
        for j in range(6):
            if j!=i:
                p=rows[j][i]; rows[j]=[a-p*b for a,b in zip(rows[j],rows[i])]
    basic=[r[6] for r in rows]; dbasic=[r[7:] for r in rows]
    du,dv,dw=[(v[i+6]-v[i])/l for i in range(3)]; strain=du+(dv*dv+dw*dw)/2
    de=[z0]*12; de[:3]=[-1/l,-dv/l,-dw/l]; de[6:9]=[1/l,dv/l,dw/l]
    he=[[z0]*12 for _ in range(12)]
    for i in (1,2):
        for a,sa in ((i,-1),(i+6,1)):
            for b,sb in ((i,-1),(i+6,1)):he[a][b]=F(sa*sb)/(l*l)
    force=[sum(B[i][j]*basic[i] for i in range(6))+l*s[0]*strain*de[j] for j in range(12)]
    tangent=[[sum(B[k][i]*dbasic[k][j] for k in range(6))+
              l*s[0]*(de[i]*de[j]+strain*he[i][j]) for j in range(12)] for i in range(12)]
    energy=sum(a*b for a,b in zip(rhs,basic))/2+s[0]*l*strain*strain/2
    station_fields=[]; differentials=[]
    for xi in (-np.sqrt(3/5),0.,np.sqrt(3/5)):
        xi=F(float(xi)); interpolation=[[z0]*6 for _ in range(6)]
        interpolation[0][0]=1;interpolation[3][1]=1
        interpolation[1][4]=interpolation[1][5]=-1/l
        interpolation[2][2]=interpolation[2][3]=1/l
        interpolation[4][2:4]=[(xi-1)/2,(xi+1)/2]
        interpolation[5][4:6]=[(xi-1)/2,(xi+1)/2]
        qforce=basic[:];qforce[0]=s[0]*strain
        qdiff=[r[:] for r in dbasic];qdiff[0]=[s[0]*a for a in de]
        station_fields.append([float(sum(a*b for a,b in zip(row,qforce))) for row in interpolation])
        differentials.append([[float(sum(row[k]*qdiff[k][j] for k in range(6))/s[i])
                              for j in range(12)] for i,row in enumerate(interpolation)])
    resultants=np.array(station_fields)
    return dict(d=d,energy=float(energy),force=T.T@np.array(list(map(float,force))),
        tangent=T.T@np.array([[float(a) for a in row] for row in tangent])@T,
        resultants=resultants,strains=resultants/np.array(list(map(float,s))),
        differential=np.array(differentials)@T,frame=frame,current_frame=anchor@frame)


def test_noncommuting_chart_pullback():
    u,q=state();v=np.cos(np.arange(12)+.7);v/=np.linalg.norm(v)
    for name,adapter in fixtures():
        out=evaluate(adapter,u,q);expected=oracle(adapter,u,q)
        compare(out.kinematics.deformation,expected['d']);compare(out.energy,expected['energy'])
        compare(out.local_force,expected['force']);compare(out.local_tangent,expected['tangent'])
        for i in range(6):
            compare(out.resultants[:,i],expected['resultants'][:,i])
            compare(out.strains[:,i],expected['strains'][:,i])
            compare(out.strain_differential[:,i],expected['differential'][:,i])
        compare(out.chart_hessian,out.chart_hessian.T)
        recovered=np.einsum('s,sij,si->j',out.weights,out.chart_strain_differential,out.resultants)
        compare(out.chart_force,recovered)
        compare(out.energy,.5*np.einsum('s,si,si->',out.weights,out.strains,out.resultants))
        compare(out.reference_frame,expected['frame']);compare(out.current_frame,expected['current_frame'])
        for h in (1e-4,1e-5,1e-6):
            plus=evaluate(adapter,u+h*v,q);minus=evaluate(adapter,u-h*v,q)
            compare(out.chart_hessian@v,(plus.chart_force-minus.chart_force)/(2*h),1e-7)
            compare(out.spatial_row_chart_tangent@v,(plus.spatial_force-minus.spatial_force)/(2*h),1e-7)
            compare(out.chart_force@v,(plus.energy-minus.energy)/(2*h),1e-7)
            compare(out.kinematics.differential@v,(plus.kinematics.deformation-minus.kinematics.deformation)/(2*h),1e-7)
            compare(np.einsum('ijk,k->ij',out.kinematics.second,v),
                    (plus.kinematics.differential-minus.kinematics.differential)/(2*h),1e-7)
            compare(out.chart_strain_differential@v,(plus.strains-minus.strains)/(2*h),1e-7)
        assert np.linalg.norm(out.chart_hessian-out.kinematics.differential.T@out.local_tangent@
                              out.kinematics.differential)>0
        if name == 'LOCAL':
            reference=evaluate(adapter,np.zeros(12),np.tile(np.eye(3),(2,1,1)))
            L=adapter.descriptor()['length'];R=np.zeros((12,6))
            R[:3,:3]=np.eye(3);R[6:9,:3]=np.eye(3)
            R[3:6,3:]=np.eye(3);R[9:12,3:]=np.eye(3)
            R[7,5]=L;R[8,4]=-L
            compare(reference.chart_hessian@R,np.zeros((12,6)),relative=False)
            spectrum=np.linalg.eigvalsh(reference.chart_hessian)
            assert np.count_nonzero(spectrum>1e-11*max(abs(spectrum)))==6
            assert min(spectrum)>=-1e-11*max(abs(spectrum))


def test_common_motion_passive_reversal_rebase():
    u,q=state();order=np.r_[np.arange(6,12),np.arange(6)];signs=np.array([1,1,-1,1,1,-1])
    for _,adapter in fixtures():
        desc=adapter.descriptor();ref=np.array(desc['coordinates']);a=evaluate(adapter,u,q)
        Q=a.kinematics.rotations;rebased=u.copy();rebased.reshape(2,6)[:,3:]=0
        b=evaluate(adapter,rebased,Q)
        compare(a.energy,b.energy);compare(a.spatial_force,b.spatial_force)
        compare(a.spatial_input_tangent,b.spatial_input_tangent)
        compare(a.strains,b.strains);compare(a.resultants,b.resultants)
        for angle in ([np.pi,0.,0.],[0.,1.4*np.pi,0.],[.4,-.3,.2]):
            W=exp(np.array(angle));moved=rebased.copy()
            moved.reshape(2,6)[:,:3]=(ref+rebased.reshape(2,6)[:,:3])@W.T+[2,-3,1]-ref
            c=evaluate(adapter,moved,np.array([W@qi for qi in Q]));map12=np.kron(np.eye(4),W)
            compare(c.energy,b.energy);compare(c.strains,b.strains);compare(c.resultants,b.resultants)
            compare(c.current_frame,W@b.current_frame);compare(c.spatial_force,map12@b.spatial_force)
            compare(c.spatial_input_tangent,map12@b.spatial_input_tangent@map12.T)
            compare(c.chart_strain_differential,b.chart_strain_differential@map12.T)
        reverse=module.PhysicalB2Adapter(tuple(desc['node_ids'][::-1]),ref[::-1].copy(),101,
                                        desc['E'],desc['nu'],desc['section'])
        c=evaluate(reverse,u[order].copy(),q[::-1].copy())
        compare(c.energy,a.energy);compare(c.chart_force,a.chart_force[order])
        compare(c.chart_hessian,a.chart_hessian[np.ix_(order,order)])
        compare(c.strains,a.strains[::-1]*signs);compare(c.resultants,a.resultants[::-1]*signs)
        compare(c.chart_strain_differential,a.chart_strain_differential[::-1][:,:,order]*signs[None,:,None])
        compare(c.reference_frame,a.reference_frame@np.diag([-1.,-1.,1.]))
        U=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);map12=np.kron(np.eye(4),U)
        section=dict(desc['section'],orientation=(U@np.array(desc['section']['orientation'])).tolist())
        other=module.PhysicalB2Adapter(tuple(desc['node_ids']),ref@U.T+[2,-3,1],101,desc['E'],desc['nu'],section)
        c=evaluate(other,map12@u,np.array([U@qi@U.T for qi in q]))
        compare(c.energy,a.energy);compare(c.strains,a.strains);compare(c.resultants,a.resultants)
        compare(c.reference_frame,U@a.reference_frame);compare(c.current_frame,U@a.current_frame)
        compare(c.chart_force,map12@a.chart_force);compare(c.chart_hessian,map12@a.chart_hessian@map12.T)
        compare(c.chart_strain_differential,a.chart_strain_differential@map12.T)


def test_definition_and_old_identity_rejection(monkeypatch):
    _,adapter=next(fixtures());desc=adapter.descriptor();u,q=state()
    args=((101,102),np.array(desc['coordinates']),101,desc['E'],desc['nu'],desc['section'])
    original=module.deformation;called=[]
    def never(*args,**kwargs): called.append(True);raise AssertionError('numerical entry before identity rejection')
    with monkeypatch.context() as patch:
        patch.setattr(module,'deformation',never);patch.setattr(module,'core_evaluate',never)
        for kw in ({'policy':'GE_BEAM3_G3C_ANCHOR_DIRECTOR_LOCAL_POTENTIAL_V1'},
                   {'operator_id':'B2'}, {'schema':'GE_BEAM3_G3C_STABLE_GRAPH_RESTART_V1'}):
            with pytest.raises(ValueError):module.PhysicalB2Adapter(*args,**kw)
    assert not called
    for ids,anchor in (((101,101),101),((True,102),True),((101,102),103)):
        with pytest.raises(ValueError):module.PhysicalB2Adapter(ids,args[1],anchor,*args[3:])
    for section in (dict(desc['section'],orientation=[1.,0.,0.]),dict(desc['section'],fiber_plasticity=True),
                    dict(desc['section'],orientation=[False,0.,1.])):
        with pytest.raises(ValueError):module.PhysicalB2Adapter(*args[:5],section)
    class Derived(module.PhysicalB2Adapter): pass
    with pytest.raises(ValueError):Derived(*args)
    for bad in (np.ones(11),np.ones(12,dtype=int),np.full(12,np.nan)):
        before=q.copy()
        with pytest.raises(ValueError):adapter.evaluate(bad,q)
        assert np.array_equal(q,before)
    for bad in (np.ones((2,3,3)),np.full((2,3,3),np.nan),np.tile(np.diag([-1.,1.,1.]),(2,1,1))):
        before=u.copy()
        with pytest.raises(ValueError):adapter.evaluate(u,bad)
        assert np.array_equal(u,before)
    cut=u.copy();cut[3]=.9*np.pi
    with pytest.raises(ValueError):adapter.evaluate(cut,q)
    pairs=np.array([np.eye(3),exp(np.array([.91*np.pi,0.,0.]))])
    with pytest.raises(ValueError):adapter.evaluate(np.zeros(12),pairs)
    expected=evaluate(adapter,u,q);before_u=u.copy();before_q=q.copy()
    def changed(*a,**kw):
        result=original(*a,**kw);u[:]=.2;q[:]=np.eye(3);return result
    with monkeypatch.context() as patch:
        patch.setattr(module,'deformation',changed);actual=adapter.evaluate(u,q)
    compare(actual.chart_hessian,expected.chart_hessian);compare(actual.spatial_input_tangent,expected.spatial_input_tangent)
    u,q=before_u,before_q
    for value in (expected,expected.kinematics):
        for field in fields(value):
            array=getattr(value,field.name)
            if isinstance(array,np.ndarray):
                assert not array.flags.writeable
                with pytest.raises(ValueError):array.setflags(write=True)
    detached=adapter.descriptor();detached['E']=123
    assert adapter.descriptor()['E']==desc['E']
    other=module.PhysicalB2Adapter(*args[:3],2*desc['E'],*args[4:])
    observed=module.PhysicalB2Adapter(*args)
    original_descriptor=module.PhysicalB2Adapter.descriptor
    def swap_after_descriptor(self):
        description=original_descriptor(self)
        if self is observed:
            object.__setattr__(self,'_body',other._body)
            object.__setattr__(self,'_seal',other._seal)
        return description
    with monkeypatch.context() as patch:
        patch.setattr(module.PhysicalB2Adapter,'descriptor',swap_after_descriptor)
        patch.setattr(module,'deformation',never)
        patch.setattr(module,'core_evaluate',never)
        with pytest.raises(module.AdmissionError,match='entry observation'):
            observed.evaluate(u,q)
    assert not called
    def swap(*a,**kw):
        result=original(*a,**kw)
        object.__setattr__(adapter,'_body',other._body);object.__setattr__(adapter,'_seal',other._seal)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module,'deformation',swap)
        with pytest.raises(ValueError):adapter.evaluate(u,q)
    assert not expected.production_qualified and not expected.state_committed
    assert not hasattr(adapter,'commit') and not hasattr(adapter,'checkpoint')
