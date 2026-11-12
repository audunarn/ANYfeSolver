"""Private authenticated retained-state conservative current-rest factors.

No condensed-state fabrication, history advance, trace inertia or public route.
The rounded stiffness is a diagnostic, never paired-spectrum authority.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import numpy as np
from scipy.linalg import block_diag
from .control import cancellation_safe_point
from ._ge_beam3_retained_generalized_state import Context, Program
from ._ge_beam3_native_generalized_factor_modal import compliance_factor, kinetic_factor
from ._ge_beam3_native_generalized_modal import _inertia, _relative, MATERIAL_POLICY, MASS_POLICY
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_p5_centered.mass import current_rest_mass
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._native_reference_modal import _owned
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes

POLICY='GE_BEAM3_RETAINED_ACCEPTED_CURRENT_REST_FACTOR_CHAIN_V1'


def _elastic_interior(core, response, origin):
    if canonical(response.history)!=canonical(origin):
        raise ValueError('current-rest perturbation must not advance material history')
    with localcontext() as ctx:
        ctx.prec=80
        for station,history in zip(json.loads(response.material)['stations'],origin.stations,strict=True):
            if station['branch']!='ELASTIC':
                raise ValueError('current-rest spectra require elastic-interior material branch')
            high,low=station['resultants']
            force=[Decimal.from_float(a)+Decimal.from_float(b) for a,b in zip(high,low,strict=True)]
            radius=Decimal.from_float(core.section.yield_force)+Decimal.from_float(core.section.hardening)*core.section._origin(history)[1]
            q=sum(force[i]*core.section._m[i][j]*force[j] for i in range(6) for j in range(6)).sqrt()
            margin=Decimal.from_float(64*np.finfo(float).eps)*max(Decimal(1),q,radius)
            if radius-q<=margin:
                raise ValueError('current-rest material branch is too close to yield boundary')


@dataclass(frozen=True)
class Pencil:
    left: np.ndarray
    right: np.ndarray
    geometric: np.ndarray
    kinetic: np.ndarray
    stiffness: np.ndarray
    mass: np.ndarray
    net_residual: np.ndarray
    free_dofs: tuple
    algebraic_dofs: tuple
    internal_layout: tuple
    compliance_errors: tuple
    checkpoint_sha256: str
    model_sha256: str
    identity: str
    policy: str=POLICY
    mass_policy: str=MASS_POLICY
    material_policy: str=MATERIAL_POLICY
    history_unchanged: bool=True
    production_qualified: bool=False
    buckling_factor_authorized: bool=False
    finite_velocity_dynamics_authorized: bool=False


def prepare(model,program,checkpoint,section_inertias,*,expected_sha256,cancellation_token=None):
    return _prepare(Context,POLICY,model,program,checkpoint,section_inertias,
        expected_sha256=expected_sha256,cancellation_token=cancellation_token)


def _prepare(context_type,policy,model,program,checkpoint,section_inertias,*,expected_sha256,cancellation_token=None):
    """Authenticate the complete accepted chain before current-material replay."""
    cancellation_safe_point(cancellation_token,'retained-modal.capture')
    if type(program) is not context_type.program_type:raise ValueError('exact retained program required')
    program.__post_init__();program.pattern.require(model.mesh)
    if any(np.any(program.pattern.density(eid)) for eid in model.mesh.elements):
        raise ValueError('nonconservative distributed couples forbidden in conservative spectra')
    if type(checkpoint) is not bytes or type(expected_sha256) is not str or sha256(checkpoint).hexdigest()!=expected_sha256:
        raise ValueError('external checkpoint authority mismatch')
    ids=set(model.mesh.elements)
    if type(section_inertias) is not dict or set(section_inertias)!=ids or any(type(i) is not int for i in section_inertias):
        raise ValueError('complete exact section inertia map required')
    inertias={eid:_inertia(section_inertias[eid]) for eid in sorted(ids)}
    input_identity=sha(section_inertias)
    context=context_type(model,program)
    state,records=context.restore(checkpoint,expected_sha256=expected_sha256)
    issued=context._require_issued(state)
    before=canonical(state)

    def check():
        cancellation_safe_point(cancellation_token,'retained-modal.guard')
        context._require_issued(state)
        if canonical(state)!=before or sha(section_inertias)!=input_identity:
            raise ValueError('retained-modal inputs changed')

    check()
    nodal=context.nodal_count;size=nodal+6*len(context.elements)
    if size>256:raise ValueError('bounded retained spectral model required')
    lefts=[];rights=[];kinetics=[];errors=[];layout=[]
    geometric=np.zeros((size,size));mass=np.zeros_like(geometric)
    force=np.zeros(size);full_force=np.zeros(context.count)
    parameter=0. if state.completed_targets==0 else program.targets[state.completed_targets-1]
    mechanical=state.mechanical
    for i,(eid,element) in enumerate(context.elements):
        check();core=element.operator;nodes=context.nodes[i]
        response=core.evaluate(mechanical.positions[nodes],mechanical.position_low[nodes],
            mechanical.nodal_frames[nodes],mechanical.cell_rotations[i],mechanical.resultants[i],
            origin=state.histories[i],check=check)
        _elastic_interior(core,response,state.histories[i])
        work=line_work(core.reference,mechanical.positions[nodes],mechanical.position_low[nodes],
            mechanical.cell_rotations[i],parameter*program.pattern.force(eid),order=core.order)
        h=response.hessian+response.hessian_low-work.hessian
        residual=response.residual-work.gradient
        if not np.array_equal(h[:24,24:].T,h[24:,:24]):
            raise ValueError('work-conjugate generalized kinematic blocks required')
        if _relative(h[:24,:24],h[:24,:24].T)>1e-11:
            raise ValueError('symmetric conservative geometric Hessian required')
        left,error=compliance_factor(-h[24:,24:],check)
        internal=tuple(range(nodal+6*i,nodal+6*i+6))
        slots=tuple(element.get_dof_mapping(model.mesh))+internal
        right=np.zeros((18,size));right[:,slots]=h[24:,:24]
        kinetic=kinetic_factor(core.reference,inertias[eid],mechanical.cell_rotations[i],check)
        physical_mass=current_rest_mass(core.reference,inertias[eid],24,
            mechanical.positions[nodes],mechanical.position_low[nodes],mechanical.cell_rotations[i])
        if _relative(kinetic.T@kinetic,physical_mass)>1e-11:
            raise ValueError('factor changed physical current-rest inertia')
        lifted=np.zeros((len(kinetic),size));lifted[:,slots]=kinetic
        lefts.append(left);rights.append(right);kinetics.append(lifted);errors.append(error)
        geometric[np.ix_(slots,slots)]+=h[:24,:24]
        mass[np.ix_(slots,slots)]+=physical_mass
        force[list(slots)]+=residual[:24];full_force[context.slots[i]]+=residual
        layout.append((eid,internal))
    external=context.nodal_external(parameter)
    if np.any(external):
        force[:nodal]-=external;full_force[:nodal]-=external
    accepted_residual=np.array(json.loads(issued)['residual'])
    if np.linalg.norm((full_force-accepted_residual)/context.scale)>1e-11:
        raise ValueError('current-rest replay changed accepted equilibrium')
    if max(np.linalg.norm((full_force/context.scale)[context.equilibrium]),
           np.linalg.norm((full_force/context.scale)[context.compatibility]))>1e-11:
        raise ValueError('free compatible conservative equilibrium required')
    free=tuple(int(d) for d in context.free if d<nodal)+tuple(range(nodal,size))
    rotations=tuple(6*i+j for i in range(len(context.node_ids)) for j in (3,4,5))
    algebraic=tuple(d for d in free if d in rotations)
    if np.any(mass[:,rotations]) or np.any(mass[rotations,:]):
        raise ValueError('nodal trace inertia must be exactly zero')
    left=block_diag(*lefts);right=np.vstack(rights);kinetic=np.vstack(kinetics)
    expanded=left@right
    body=dict(left=_owned(left),right=_owned(right),geometric=_owned(geometric),kinetic=_owned(kinetic),
        stiffness=_owned(expanded.T@expanded+geometric),mass=_owned(mass),net_residual=_owned(force),
        free_dofs=free,algebraic_dofs=algebraic,internal_layout=tuple(layout),compliance_errors=tuple(errors),
        checkpoint_sha256=expected_sha256,model_sha256=context.identity)
    packet=Pencil(**body,identity=sha(dict(policy=policy,**body)),policy=policy)
    identity=sha(packet)

    def guard():
        check()
        if sha(packet)!=identity:raise ValueError('retained-modal factor packet changed')

    guard();return packet,guard


def solve_modes(model,program,checkpoint,section_inertias,*,expected_sha256,bounds,
                num_modes=6,root_width=1e-10,relative_width=1e-12,cancellation_token=None):
    packet,guard=prepare(model,program,checkpoint,section_inertias,expected_sha256=expected_sha256,
        cancellation_token=cancellation_token)
    result=solve_paired_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
        packet.free_dofs,packet.algebraic_dofs,bounds=bounds,num_modes=num_modes,
        root_width=root_width,relative_width=relative_width,cancellation_token=cancellation_token)
    guard();return packet,result
