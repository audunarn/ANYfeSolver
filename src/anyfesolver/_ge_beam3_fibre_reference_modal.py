"""Physical reference modes from immutable native fibre definitions.

This is a virgin reference operator, not a fabricated accepted-state capsule.
Free-body and supported spectra retain the physical cell-rotation inertia.
"""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from scipy.linalg import block_diag
from ._ge_beam3_native_analysis import _require
from ._ge_beam3_analysis_translation_modal import _controls
from ._ge_beam3_retained_fibre_modal import elastic_interior, MATERIAL_POLICY
from ._ge_beam3_native_generalized_factor_modal import compliance_factor, kinetic_factor
from ._ge_beam3_native_generalized_modal import _relative, MASS_POLICY
from ._ge_beam3_p5_centered.mass import current_rest_mass
from ._native_reference_modal import _owned
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes
from ._ge_beam3_p5_seeded.core import sha
from .control import cancellation_safe_point
from ._ge_beam3_effective_deadline import effective_deadline_seconds

POLICY='GE_BEAM3_PHYSICAL_FIBRE_REFERENCE_FACTOR_CHAIN_V1'


@dataclass(frozen=True)
class ReferencePencil:
    left: np.ndarray
    right: np.ndarray
    geometric: np.ndarray
    kinetic: np.ndarray
    stiffness: np.ndarray
    mass: np.ndarray
    free_dofs: tuple
    algebraic_dofs: tuple
    internal_layout: tuple
    definition_graph_sha256: str
    identity: str
    policy: str=POLICY
    mass_policy: str=MASS_POLICY
    material_policy: str=MATERIAL_POLICY
    production_qualified: bool=False


def prepare(analysis, cancellation_token=None):
    analysis._family_required('PHYSICAL_FIBRE_NODAL')
    started=monotonic()
    def check():
        cancellation_safe_point(cancellation_token,'fibre-reference-modal.guard')
        if monotonic()-started>effective_deadline_seconds(120.):raise RuntimeError('physical-fibre reference modal deadline')
        analysis._guard()
    check()
    nodal=analysis.model.mesh.dof_manager.total_dofs;size=nodal+6*len(analysis._elements)
    _require(size<=256,'physical-fibre reference modal coordinate bound')
    lefts=[];rights=[];kinetics=[];layout=[];frames={}
    geometric=np.zeros((size,size));mass=np.zeros_like(geometric)
    for i,element in enumerate(analysis._elements):
        check();operator=element.operator;reference=operator.reference
        for node,frame in zip(element.node_ids,reference.nodal_triads,strict=True):
            if node in frames:
                _require(np.array_equal(frame,frames[node]),'shared-node physical fibre reference frames disagree')
            frames[node]=frame
        rotations=np.tile(np.eye(3),(2,1,1));origin=operator.cell.virgin()
        response=operator.evaluate(reference.coordinates,np.zeros((3,3)),reference.nodal_triads,
            rotations,np.zeros(18),origin=origin,check=check)
        elastic_interior(operator,response,origin)
        _require(np.linalg.norm(response.residual)<=1e-11,'stress-free physical-fibre reference required')
        h=response.hessian+response.hessian_low
        _require(not np.any(h[:24,:24]) and np.array_equal(h[:24,24:].T,h[24:,:24]),
                 'reference fibre work-conjugate Hessian required')
        left,_=compliance_factor(-h[24:,24:],check)
        internal=tuple(range(nodal+6*i,nodal+6*i+6))
        slots=tuple(element.get_dof_mapping(analysis.model.mesh))+internal
        right=np.zeros((18,size));right[:,slots]=h[24:,:24]
        inertia=analysis._inertias[element.element_id]
        kinetic=kinetic_factor(reference,inertia,rotations,check)
        physical_mass=current_rest_mass(reference,inertia,24,reference.coordinates,np.zeros((3,3)),rotations)
        _require(_relative(kinetic.T@kinetic,physical_mass)<=1e-11,'physical fibre reference kinetic identity')
        lifted=np.zeros((len(kinetic),size));lifted[:,slots]=kinetic
        lefts.append(left);rights.append(right);kinetics.append(lifted)
        mass[np.ix_(slots,slots)]+=physical_mass;layout.append((element.element_id,internal))
    fixed=set(analysis._admit_boundaries())
    free=tuple(d for d in range(size) if d not in fixed)
    rotations=tuple(d for d in range(nodal) if d%6 in (3,4,5))
    algebraic=tuple(d for d in rotations if d not in fixed)
    _require(not np.any(mass[:,rotations]) and not np.any(mass[rotations,:]),'zero algebraic trace inertia required')
    left=block_diag(*lefts);right=np.vstack(rights);kinetic=np.vstack(kinetics);expanded=left@right
    body=dict(left=_owned(left),right=_owned(right),geometric=_owned(geometric),kinetic=_owned(kinetic),
        stiffness=_owned(expanded.T@expanded),mass=_owned(mass),free_dofs=free,algebraic_dofs=algebraic,
        internal_layout=tuple(layout),definition_graph_sha256=analysis.identity)
    packet=ReferencePencil(**body,identity=sha(dict(policy=POLICY,**body)))
    identity=sha(packet)
    def guard():
        check()
        _require(sha(packet)==identity,'physical fibre reference packet changed')
    guard();return packet,guard


def solve(analysis,*,bounds,num_modes=6,cancellation_token=None):
    _controls(analysis,bounds,num_modes,1e-10,1e-12)
    with analysis._operation():
        packet,guard=prepare(analysis,cancellation_token)
        modes=solve_paired_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
            packet.free_dofs,packet.algebraic_dofs,bounds=bounds,num_modes=num_modes,cancellation_token=cancellation_token)
        guard();return packet,modes
