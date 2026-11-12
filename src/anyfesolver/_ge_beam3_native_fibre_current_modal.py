"""Conservative modes from the actual physical-fibre FORCE checkpoint owner.

No translation-state conversion, fictitious displacement controller or state
advance. The two physical cell rotations retain their complete inertia; only
the eighteen inertia-free resultants are eliminated before the modal solve.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag
from . import _ge_beam3_native_generalized_modal as capture
from ._ge_beam3_native_generalized_factor_modal import compliance_factor,kinetic_factor,FactorPencil
from ._ge_beam3_retained_fibre_modal import elastic_interior
from ._ge_beam3_p5_centered.mass import current_rest_mass
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._native_reference_modal import _owned
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes

POLICY='GE_BEAM3_NATIVE_PHYSICAL_FIBRE_FORCE_CURRENT_REST_FACTOR_CHAIN_V1'


@dataclass(frozen=True)
class FibreForceOperator:
    stiffness: np.ndarray
    mass: np.ndarray
    net_residual: np.ndarray
    state_identity: str
    left: np.ndarray
    right: np.ndarray
    geometric: np.ndarray
    kinetic: np.ndarray
    compliance_energy_residual: float
    static_hessian_error: float
    history_unchanged: bool=True


def _operator(element,state,inertia,check):
    core=element.operator;accepted=state['response'];origin=accepted.history
    before=canonical(state)
    replay=core.evaluate(state['positions'],state['position_low'],
        state['committed_nodal_rotation_matrices']@core.reference.nodal_triads,
        accepted.rotations,accepted.resultants,origin=origin,check=check)
    elastic_interior(core,replay,origin)
    h=replay.hessian+replay.hessian_low;force=replay.residual
    if capture._relative(force,accepted.full_residual)>1e-11:
        raise ValueError('fibre force-state modal replay changed equilibrium')
    if not np.array_equal(h[:24,24:].T,h[24:,:24]):
        raise ValueError('work-conjugate fibre force-state blocks required')
    if capture._relative(h[:24,:24],h[:24,:24].T)>1e-11:
        raise ValueError('symmetric conservative fibre force-state Hessian required')
    left,error=compliance_factor(-h[24:,24:],check);right=_owned(h[24:,:24]);geometric=_owned(h[:24,:24])
    expanded=left@right;stiffness=expanded.T@expanded+geometric
    # This owner's static response is the symmetric local second variation,
    # not the generalized owner's spatial_jacobian field or state convention.
    nodal=h[:18,:18]-h[:18,18:]@np.linalg.solve(h[18:,18:],h[18:,:18])
    nodal_error=capture._relative(nodal,accepted.hessian)
    if nodal_error>1e-11:raise ValueError('fibre current elastic static Hessian changed')
    kinetic=kinetic_factor(core.reference,inertia,accepted.rotations,check)
    mass=current_rest_mass(core.reference,inertia,24,state['positions'],state['position_low'],accepted.rotations)
    if capture._relative(kinetic.T@kinetic,mass)>1e-11:
        raise ValueError('fibre force-state current-rest inertia changed')
    check()
    if canonical(state)!=before:raise ValueError('fibre force-state material history changed')
    return FibreForceOperator(_owned(stiffness),_owned(mass),_owned(force[:24]),state['state_sha256'],
        left,right,geometric,kinetic,error,nodal_error)


def prepare(model,states,displacements,inertias,external,*,cancellation_token=None):
    base,owner_guard=capture._prepare_with_operator(model,states,displacements,inertias,external,
        operator_factory=_operator,policy=POLICY,section_family='PHYSICAL_FIBRE',cancellation_token=cancellation_token)
    size=len(base.mass);lefts=[];rights=[];kinetics=[];geometric=np.zeros((size,size));layout=dict(base.internal_layout)
    for eid,op in base.operators:
        owner_guard();slots=tuple(model.mesh.elements[eid].get_dof_mapping(model.mesh))+layout[eid]
        right=np.zeros((18,size));right[:,slots]=op.right
        kinetic=np.zeros((len(op.kinetic),size));kinetic[:,slots]=op.kinetic
        lefts.append(op.left);rights.append(right);kinetics.append(kinetic)
        geometric[np.ix_(slots,slots)]+=op.geometric
    packet=FactorPencil(base,_owned(block_diag(*lefts)),_owned(np.vstack(rights)),
        _owned(geometric),_owned(np.vstack(kinetics)),policy=POLICY)
    identity=sha(packet)
    def guard():
        owner_guard()
        if sha(packet)!=identity:raise ValueError('native fibre force-state modal packet changed')
    guard();return packet,guard


def solve(analysis,checkpoint,*,expected_sha256,bounds,num_modes=6,cancellation_token=None):
    from ._ge_beam3_analysis_translation_modal import _controls
    from ._ge_beam3_native_fibre_restart import decode_checkpoint,_forces
    from .control import cancellation_safe_point
    cancellation_safe_point(cancellation_token,'fibre-force-modal.capture')
    analysis._family_required('PHYSICAL_FIBRE_NODAL')
    _controls(analysis,bounds,num_modes,1e-10,1e-12)
    with analysis._operation():
        raw,digest=analysis._backend(checkpoint,expected_sha256)
        forces,chain=decode_checkpoint(analysis.model,raw,expected_sha256=digest)
        state=chain[-1]
        external=state['load_factor']*_forces(forces,tuple(sorted(analysis.model.mesh.nodes)),
            analysis.model.mesh.dof_manager.total_dofs)
        packet,guard=prepare(analysis.model,state['states'],state['displacements'],analysis._inertias,external,
            cancellation_token=cancellation_token)
        result=solve_paired_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
            packet.base.free_dofs,packet.base.algebraic_dofs,bounds=bounds,num_modes=num_modes,
            cancellation_token=cancellation_token)
        guard();return packet,result
