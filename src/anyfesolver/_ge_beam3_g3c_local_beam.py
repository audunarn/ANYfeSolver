"""Private, unqualified G3c direct-anchor local trial. No graph/state owner."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import numpy as np

from ._ge_beam3_mixed_ad import (Jet2, constant_matrix, matmul, matvec,
    transpose, so3_exp, so3_log, rotation_log)
from ._ge_beam3_pose_joint import _exp_terms

POLICY='GE_BEAM3_G3C_ANCHOR_DIRECTOR_LOCAL_POTENTIAL_V1'

def owned(value):
    a=np.ascontiguousarray(value,dtype=float)
    if not np.isfinite(a).all(): raise ValueError('nonfinite local computed output')
    return np.frombuffer(a.tobytes(),dtype=float).reshape(a.shape)

def array(value,shape):
    if (type(value) is not np.ndarray or value.dtype!=np.dtype('float64')
            or value.shape!=shape or not np.isfinite(value).all()):
        raise ValueError('exact finite binary64 array required')
    return owned(value)

def rotations(value,n):
    q=array(value,(n,3,3))
    for v in q:
        if np.linalg.norm(v.T@v-np.eye(3))>1e-11 or abs(np.linalg.det(v)-1)>1e-11:
            raise ValueError('proper rotation required')
    return q

@dataclass(frozen=True)
class Kinematics:
    deformation: np.ndarray
    differential: np.ndarray
    second: np.ndarray
    frame: np.ndarray
    rotations: np.ndarray

def deformation(reference,displacement,accepted_rotations,anchor_index):
    """Differentiate all positions and Exp(eta)Qaccepted, including the anchor."""
    n=len(reference)
    if n not in (2,3) or type(anchor_index) is not int or not 0<=anchor_index<n:
        raise ValueError('registered beam topology and exact anchor required')
    ref=array(reference,(n,3)); u=array(displacement,(6*n,))
    qa=rotations(accepted_rotations,n)
    length=float(np.linalg.norm(ref[-1]-ref[0]))
    if not math.isfinite(length) or length<1e-12 or (n==3 and not np.array_equal(ref[1],(ref[0]+ref[2])/2)):
        raise ValueError('straight nondegenerate exact-midpoint reference required')
    if n==3 and anchor_index!=1: raise ValueError('B3 physical midpoint anchor required')
    if np.any(np.linalg.norm(u.reshape(n,6)[:,3:],axis=1)>=.9*np.pi):
        raise ValueError('trial chart increment requires cutback')
    count=6*n
    z=[[Jet2.variable(u[6*i+j],6*i+j,count) for j in range(6)] for i in range(n)]
    x=[[z[i][j]+ref[i,j] for j in range(3)] for i in range(n)]
    q=[matmul(so3_exp(z[i][3:]),constant_matrix(qa[i],count)) for i in range(n)]
    qv=np.array([[[v.value for v in row] for row in qi] for qi in q])
    # Anchor-only checks are insufficient: enforce the actual all-pair rule.
    for i in range(n):
        for j in range(i): rotation_log(qv[j].T@qv[i])
    rt=transpose(q[anchor_index]); centre=[sum(row[j] for row in x)/n for j in range(3)]
    refc=ref-ref.mean(axis=0); values=[]
    for i in range(n):
        local=matvec(rt,[x[i][j]-centre[j] for j in range(3)])
        values.extend(local[j]-refc[i,j] for j in range(3))
        values.extend(so3_log(matmul(rt,q[i])))
    return Kinematics(owned([v.value for v in values]),owned([v.gradient for v in values]),
        owned([v.hessian for v in values]),owned(qv[anchor_index]),owned(qv))

@dataclass(frozen=True)
class LocalTrial:
    kinematics: Kinematics
    local_force: np.ndarray
    local_tangent: np.ndarray
    chart_force: np.ndarray
    chart_hessian: np.ndarray
    spatial_force: np.ndarray
    spatial_row_chart_tangent: np.ndarray
    definition_sha256: str
    station_diagnostics: object = field(default=None)
    production_qualified: bool=field(default=False,init=False)
    recovery_complete: bool=field(default=False,init=False)
    state_committed: bool=field(default=False,init=False)

def scalar(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('finite nonboolean scalar required')
    return float(value)

class LocalBeam:
    """Immutable definition, fresh isolated family/cache per pure trial.

    No input material/model/state object is borrowed; no commit/restart API.
    Station recovery and global graph ownership remain separate unfinished gates.
    """
    __slots__=('_body','_seal')
    def __setattr__(self,name,value):
        if hasattr(self,name): raise AttributeError('write-once local definition')
        object.__setattr__(self,name,value)
    def __delattr__(self,name):
        raise AttributeError('local definition cannot be deleted')
    def __init__(self,family,node_ids,coordinates,anchor_node,E,nu,section):
        if family not in ('B2','B3'): raise ValueError('exact B2/B3 family required')
        n=2 if family=='B2' else 3
        if (type(node_ids) is not tuple or len(node_ids)!=n or
            any(type(i) is not int or i<=0 for i in node_ids) or len(set(node_ids))!=n or
            type(anchor_node) is not int or anchor_node not in node_ids):
            raise ValueError('exact physical node IDs required')
        if n==3 and anchor_node!=node_ids[1]: raise ValueError('midpoint anchor required')
        ref=array(coordinates,(n,3))
        length=float(np.linalg.norm(ref[-1]-ref[0]))
        if not math.isfinite(length) or length<1e-12 or (n==3 and not np.array_equal(ref[1],(ref[0]+ref[2])/2)):
            raise ValueError('straight exact-midpoint reference required')
        required={'area','Iy','Iz','J','shear_factor_y','shear_factor_z','orientation'}
        if type(section) is not dict or set(section)!=required: raise ValueError('closed scalar section required')
        s={k:scalar(section[k]) for k in required-{'orientation'}}
        if min(s.values())<=0: raise ValueError('positive section values required')
        o=np.asarray(section['orientation'])
        if o.shape!=(3,) or any(type(v) is bool for v in section['orientation']):
            raise ValueError('explicit physical local-z direction required')
        o=np.array([scalar(v) for v in section['orientation']])
        scale=float(np.max(np.abs(o)))
        if scale==0: raise ValueError('nonzero physical local-z direction required')
        o=o/scale; o=o/np.linalg.norm(o)
        a=(ref[-1]-ref[0])/length
        if not np.linalg.norm(o-a*(a@o))>1e-6*np.linalg.norm(o):
            raise ValueError('legacy orientation fallback forbidden')
        s['orientation']=o.tolist(); E=scalar(E); nu=scalar(nu)
        if E<=0 or not -1<nu<.5: raise ValueError('positive elastic material required')
        body=dict(policy=POLICY,family=family,node_ids=list(node_ids),coordinates=ref.tolist(),
                  anchor_node=anchor_node,E=E,nu=nu,section=s)
        self._body=(json.dumps(body,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode('ascii')
        self._seal=sha256(self._body).hexdigest()

    def descriptor(self):
        if type(self) is not LocalBeam or type(self._body) is not bytes or sha256(self._body).hexdigest()!=self._seal:
            raise ValueError('frozen local definition changed')
        return json.loads(self._body)

    def evaluate(self,displacement,accepted_rotations):
        entry_body,entry_seal=self._body,self._seal
        d=self.descriptor(); n=len(d['node_ids']); ref=np.array(d['coordinates'])
        u=array(displacement,(6*n,)); q=rotations(accepted_rotations,n)
        kin=deformation(ref,u,q,d['node_ids'].index(d['anchor_node']))
        # New objects every call isolate legacy B2's mutable nonlinear cache.
        from .fe_core import FEModel, Material
        from .elements import BeamElement, QuadraticBeamElement
        model=FEModel('private G3c direct anchor local trial')
        for node,position in zip(d['node_ids'],ref): model.add_node(node,*position)
        section=dict(d['section'],geometric_nonlinearity='von_karman')
        cls=BeamElement if d['family']=='B2' else QuadraticBeamElement
        element=cls(1,list(d['node_ids']),cross_section=section)
        axis=(ref[-1]-ref[0])/np.linalg.norm(ref[-1]-ref[0])
        normal=np.array(d['section']['orientation']); normal-=axis*(axis@normal)
        normal/=np.linalg.norm(normal)
        expected_frame=np.column_stack((axis,np.cross(normal,axis),normal))
        _,transform=element._beam_frame_and_transform(ref)
        if np.linalg.norm(transform[:3,:3].T-expected_frame)>1e-11:
            raise ValueError('actual local section frame differs from physical authority')
        material=Material('G3c virgin scalar',d['E'],d['nu'])
        if (element.generalized_section is not None or element._fiber_plasticity_config(material) is not None
            or element._geometric_nonlinearity!='von_karman' or material.yield_stress!=0
            or material.hardening_curve is not None):
            raise ValueError('unregistered local dispatch')
        f,k,state=element.compute_nonlinear_response(model.mesh,material,kin.deformation,None,3,True)
        if state is not None: raise ValueError('unexpected local history')
        f=array(np.asarray(f),(6*n,)); k=array(np.asarray(k),(6*n,6*n))
        from ._ge_beam3_g3c_recovery import recover
        recovery=recover(d,kin,transform,element,material,entry_seal)
        recovered_force=owned(np.einsum('s,sij,si->j',recovery.weights,
                                 recovery.strain_differential,recovery.resultants))
        if np.linalg.norm(recovered_force-f)>1e-11*max(1.,np.linalg.norm(f),np.linalg.norm(recovered_force)):
            raise ValueError('station virtual work differs from actual local force')
        if np.linalg.norm(k-k.T)>1e-11*max(1.,np.linalg.norm(k)):
            raise ValueError('nonconservative local tangent')
        D,S=kin.differential,kin.second
        g=D.T@f; H=D.T@k@D+np.einsum('i,ijk->jk',f,S)
        P=np.eye(6*n); terms=[]
        for i,row in enumerate(u.reshape(n,6)):
            _,J,dJ=_exp_terms(row[3:]); sl=slice(6*i+3,6*i+6)
            P[sl,sl]=J; terms.append((sl,dJ))
        r=np.linalg.solve(P.T,g); correction=np.zeros_like(H)
        for sl,dJ in terms: correction[sl,sl]=np.einsum('ijk,i->jk',dJ,r[sl])
        spatial_J=np.linalg.solve(P.T,H-correction)
        self.descriptor()
        if self._body!=entry_body or self._seal!=entry_seal:
            raise ValueError('local definition replaced during evaluation')
        return LocalTrial(kin,f,k,owned(g),owned(H),owned(r),owned(spatial_J),entry_seal,recovery)
