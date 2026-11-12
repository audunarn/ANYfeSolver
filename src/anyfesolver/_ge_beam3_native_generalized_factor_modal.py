"""Private current-material factor pencil; no production or dynamic activation."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from scipy.linalg import block_diag,solve_triangular
from . import _ge_beam3_native_generalized_modal as dense
from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._ge_beam3_p5_centered.mass import _CenteredRestInertia,current_rest_mass
from ._native_reference_modal import _owned
from ._dyadic_factor_chain import reassemble_chain_exact_binary64
from ._native_relative_factor_chain_modes import solve_relative_factor_chain_modes
from ._ge_beam3_p5_seeded.core import sha

POLICY='GE_BEAM3_NATIVE_GENERALIZED_CURRENT_REST_FACTOR_CHAIN_V1'

def compliance_factor(value,check=lambda:None):
    s=_owned(value)
    if s.shape!=(18,18):raise ValueError('eighteen-resultant compliance required')
    symmetric=np.empty_like(s)
    for i in range(18):
        check()
        for j in range(i,18):
            # Preserve equal entries, including signed zeros, bit-for-bit.
            a,b=float(s[i,j]),float(s[j,i])
            v=a if a==b else float((Fraction(a)+Fraction(b))/2)
            symmetric[i,j]=symmetric[j,i]=v
    root=np.linalg.cholesky(symmetric)
    left=solve_triangular(root,np.eye(18),lower=True)
    # A compliance-energy-scaled witness, computed without intermediate
    # binary64 products. No explicit ill-conditioned matrix inverse is made.
    normalized,_=reassemble_chain_exact_binary64(np.zeros((1,1)),np.zeros((1,18)),
        s,np.zeros((1,18)),left.T,check)
    error=float(np.linalg.norm(normalized-np.eye(18)))
    if error>1e-11:raise ValueError('original compliance energy witness failed')
    return _owned(left),error

def kinetic_factor(reference,inertia,rotations,check=lambda:None):
    root=np.linalg.cholesky(inertia).T;rows=[]
    for cell,left,right,t,r0,offset,measure in _CenteredRestInertia(reference,inertia,order=24)._stations:
        check();u=rotations[cell];q=u@r0;d=u@offset;x,y,z=d
        skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        b=np.zeros((6,24));slot=slice(18+3*cell,21+3*cell)
        b[:3,6*left:6*left+3]=(1-t)*np.eye(3);b[:3,6*right:6*right+3]=t*np.eye(3)
        b[:3,slot]=-skew;b[3:,slot]=np.eye(3)
        rows.append(np.sqrt(measure)*root@(np.kron(np.eye(2),q).T@b))
    return _owned(np.vstack(rows))

@dataclass(frozen=True)
class FactorOperator:
    stiffness: np.ndarray  # Rounded preconditioner/diagnostic, not spectrum authority.
    mass: np.ndarray
    net_residual: np.ndarray
    state_identity: str
    left: np.ndarray
    right: np.ndarray
    geometric: np.ndarray
    kinetic: np.ndarray
    compliance_energy_residual: float
    nodal_jacobian_error: float
    history_unchanged: bool=True

def _operator(element,state,inertia,check):
    core=element.operator;response=state['response']
    h,force=dense._elastic_replay(element,state,check)
    if not np.array_equal(h[:24,24:].T,h[24:,:24]):
        raise ValueError('work-conjugate generalized kinematic blocks required')
    left,error=compliance_factor(-h[24:,24:],check);right=_owned(h[24:,:24])
    geometric=_owned(h[:24,:24])
    if dense._relative(geometric,geometric.T)>1e-11:
        raise ValueError('symmetric conservative geometric Hessian required')
    expanded=left@right;k=expanded.T@expanded+geometric
    j=spatial_jacobian(force,h)
    nodal=j[:18,:18]-j[:18,18:]@np.linalg.solve(j[18:,18:],j[18:,:18])
    nodal_error=dense._relative(nodal,response.spatial_jacobian)
    if nodal_error>1e-11:raise ValueError('factor capture changed accepted nodal Jacobian')
    kinetic=kinetic_factor(core.reference,inertia,response.rotations,check)
    mass=current_rest_mass(core.reference,inertia,24,state['positions'],state['position_low'],response.rotations)
    if dense._relative(kinetic.T@kinetic,mass)>1e-11:
        raise ValueError('factor changed physical current-rest inertia')
    check()
    return FactorOperator(_owned(k),_owned(mass),_owned(force[:24]),state['state_sha256'],
        left,right,geometric,kinetic,error,nodal_error)

@dataclass(frozen=True)
class FactorPencil:
    base: dense.Pencil
    left: np.ndarray
    right: np.ndarray
    geometric: np.ndarray
    kinetic: np.ndarray
    policy: str=POLICY
    production_qualified: bool=False
    buckling_factor_authorized: bool=False
    finite_velocity_dynamics_authorized: bool=False

def prepare(model,element_states,displacement,section_inertias,nodal_spatial_dead_forces,*,cancellation_token=None):
    base,old_guard=dense._prepare_with_operator(model,element_states,displacement,section_inertias,
        nodal_spatial_dead_forces,operator_factory=_operator,policy=POLICY,cancellation_token=cancellation_token)
    size=len(base.mass);lefts=[];rights=[];kinetics=[];geometric=np.zeros((size,size))
    layout=dict(base.internal_layout)
    for eid,op in base.operators:
        old_guard();slots=tuple(model.mesh.elements[eid].get_dof_mapping(model.mesh))+layout[eid]
        right=np.zeros((18,size));right[:,slots]=op.right
        kinetic=np.zeros((len(op.kinetic),size));kinetic[:,slots]=op.kinetic
        lefts.append(op.left);rights.append(right);kinetics.append(kinetic)
        geometric[np.ix_(slots,slots)]+=op.geometric
    result=FactorPencil(base,_owned(block_diag(*lefts)),_owned(np.vstack(rights)),
        _owned(geometric),_owned(np.vstack(kinetics)))
    identity=sha(result)
    def guard():
        old_guard()
        if sha(result)!=identity:raise ValueError('native generalized factor packet changed')
    guard();return result,guard

def solve_modes(model,element_states,displacement,section_inertias,nodal_spatial_dead_forces,*,
        bounds,num_modes=6,root_width=1e-10,relative_width=1e-12,cancellation_token=None):
    packet,guard=prepare(model,element_states,displacement,section_inertias,nodal_spatial_dead_forces,
        cancellation_token=cancellation_token)
    result=solve_relative_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
        packet.base.free_dofs,packet.base.algebraic_dofs,bounds=bounds,num_modes=num_modes,
        root_width=root_width,relative_width=relative_width,cancellation_token=cancellation_token)
    guard();return packet,result
