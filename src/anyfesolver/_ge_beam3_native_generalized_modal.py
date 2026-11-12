"""Private generalized native beam pencil at conservative current rest.

Retain physical cell inertia. No static cell condensation, trace mass floor,
plastic-frequency claim, public selector or history advance.
"""
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, localcontext
from time import monotonic
import json
import numpy as np
from scipy import sparse
from .assembly import build_constraint_transformation
from .control import cancellation_safe_point
from ._native_reference_modal import _owned
from ._native_stationary_spectrum import solve_stationary_spectrum
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_p5_centered.mass import current_rest_mass
from ._ge_beam3_p5_seeded.core import canonical, sha

POLICY = 'GE_BEAM3_NATIVE_GENERALIZED_CURRENT_REST_CONSERVATIVE_PENCIL_V1'
MASS_POLICY = 'CENTERED_LIFTED_CELL_INERTIA_24_POINT_NO_STATIC_REDUCTION'
MATERIAL_POLICY = 'COMMITTED_HISTORY_ELASTIC_INTERIOR_NO_ADVANCE'


@dataclass(frozen=True)
class Operator:
    stiffness: np.ndarray
    mass: np.ndarray
    net_residual: np.ndarray
    state_identity: str
    moment_inverse_residual: float
    nodal_jacobian_error: float
    history_unchanged: bool


@dataclass(frozen=True)
class Pencil:
    stiffness: np.ndarray
    mass: np.ndarray
    net_residual: np.ndarray
    nodal_spatial_dead_forces: np.ndarray
    free_dofs: tuple
    algebraic_dofs: tuple
    internal_layout: tuple
    operators: tuple
    identity: str
    policy: str = POLICY
    mass_policy: str = MASS_POLICY
    material_policy: str = MATERIAL_POLICY
    buckling_factor_authorized: bool = False
    finite_velocity_dynamics_authorized: bool = False
    production_qualified: bool = False


def _relative(a, b):
    return float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b)))


def _inertia(value):
    matrix = _owned(value)
    if matrix.shape != (6,6) or not np.array_equal(matrix,matrix.T):
        raise ValueError('exact symmetric six-by-six physical section inertia required')
    np.linalg.cholesky(matrix)
    return matrix


def _elastic_replay(element, state, check):
    core = element.operator
    response = state['response']
    # New perturbations begin at committed history, not the previous increment.
    origin = response.history
    replay = core.evaluate(state['positions'],state['position_low'],
        state['committed_nodal_rotation_matrices']@core.reference.nodal_triads,
        response.rotations,response.resultants,origin=origin,check=check)
    material = json.loads(replay.material)
    if canonical(replay.history) != canonical(origin):
        raise ValueError('current-rest perturbation must not advance material history')
    with localcontext() as ctx:
        ctx.prec = 80
        for station,history in zip(material['stations'],origin.stations,strict=True):
            if station['branch'] != 'ELASTIC':
                raise ValueError('current-rest spectra require elastic-interior material branch')
            high,low = station['resultants']
            force = [Decimal.from_float(a)+Decimal.from_float(b) for a,b in zip(high,low)]
            radius = Decimal.from_float(core.section.yield_force)+Decimal.from_float(core.section.hardening)*core.section._origin(history)[1]
            q = sum(force[i]*core.section._m[i][j]*force[j] for i in range(6) for j in range(6)).sqrt()
            margin = Decimal.from_float(64*np.finfo(float).eps)*max(Decimal(1),q,radius)
            if radius-q <= margin:
                raise ValueError('current-rest material branch is too close to yield boundary')
    h = replay.hessian+replay.hessian_low
    force = replay.residual.copy()
    load = state['load_pattern'].force(element.element_id)
    if np.any(load):
        work = line_work(core.reference,state['positions'],state['position_low'],
            response.rotations,load,order=core.order)
        force -= work.gradient
        h = h-work.hessian
    if _relative(force,response.full_residual)>1e-11:
        raise ValueError('current-rest replay changed accepted equilibrium')
    return h,force


def _operator(element, state, inertia, check):
    core=element.operator;response=state['response']
    h,force=_elastic_replay(element,state,check)
    np.linalg.cholesky(-h[24:,24:])
    inverse = np.linalg.solve(h[24:,24:],np.eye(18))
    inverse_error = max(_relative(h[24:,24:]@inverse,np.eye(18)),
        _relative(inverse@h[24:,24:],np.eye(18)))
    if inverse_error>1e-11:
        raise ValueError('generalized resultant inverse witness failed')
    k = h[:24,:24]-h[:24,24:]@inverse@h[24:,:24]
    # Validate static consistency with the spatial connection terms retained.
    j = spatial_jacobian(force,h)
    nodal = j[:18,:18]-j[:18,18:]@np.linalg.solve(j[18:,18:],j[18:,:18])
    nodal_error = _relative(nodal,response.spatial_jacobian)
    if nodal_error>1e-11:
        raise ValueError('current elastic nodal Jacobian differs from accepted operator')
    mass = current_rest_mass(core.reference,inertia,24,state['positions'],
        state['position_low'],response.rotations)
    check()
    return Operator(_owned(k),_owned(mass),_owned(force[:24]),state['state_sha256'],
        inverse_error,nodal_error,True)


def prepare(model, element_states, displacement, section_inertias,
        nodal_spatial_dead_forces, *, cancellation_token=None):
    """Capture a supported conservative equilibrium without registering a store."""
    return _prepare_with_operator(model,element_states,displacement,section_inertias,
        nodal_spatial_dead_forces,operator_factory=_operator,policy=POLICY,
        cancellation_token=cancellation_token)


def _prepare_with_operator(model,element_states,displacement,section_inertias,
        nodal_spatial_dead_forces,*,operator_factory,policy,cancellation_token=None,section_family='GENERALIZED'):
    """Internal shared state/ownership capture; callers bind their numeric policy."""
    started = monotonic()
    if section_family not in ('GENERALIZED','PHYSICAL_FIBRE'):
        raise ValueError('registered native spectral section family required')
    fibre = section_family=='PHYSICAL_FIBRE'
    element_type=NativeGeneralizedStaticElement
    material_policy=MATERIAL_POLICY
    def material_guard():pass
    if fibre:
        from ._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
        from ._ge_beam3_retained_fibre_modal import _section_guard,MATERIAL_POLICY as FIBRE_MATERIAL_POLICY
        element_type=NativeFibreStaticElement;material_policy=FIBRE_MATERIAL_POLICY
        def material_guard():_section_guard(model)
    elements = tuple(sorted(model.mesh.elements.items()))
    nodal = model.mesh.dof_manager.total_dofs
    if not elements or not 1 <= nodal+6*len(elements) <= 256:
        raise ValueError('bounded native generalized spectral model required')
    if any(type(e) is not element_type for _,e in elements):
        raise ValueError('exact native spectral section-family elements required')
    material_guard()
    ids = {i for i,_ in elements}
    if any(type(v) is not dict or set(v)!=ids or any(type(i) is not int for i in v)
        for v in (element_states,section_inertias)):
        raise ValueError('complete exact state and inertia maps required')
    inertias = {i:_inertia(section_inertias[i]) for i,_ in elements}
    total,external = _owned(displacement),_owned(nodal_spatial_dead_forces)
    if total.shape!=(nodal,) or external.shape!=(nodal,):
        raise ValueError('complete finite displacement and actual nodal force required')
    node_maps = [tuple(model.mesh.dof_manager.get_node_dofs(i)) for i in sorted(model.mesh.nodes)]
    if any(len(row)!=6 for row in node_maps) or sorted(d for row in node_maps for d in row)!=list(range(nodal)):
        raise ValueError('complete disjoint six-DOF node maps required')
    rotations = tuple(d for row in node_maps for d in row[3:])
    if np.any(external[list(rotations)]):
        raise ValueError('nodal moments require a separate conservative load contract')
    if model.constraint_equations or model.mesh.element_activity is not None or model.mesh.point_masses:
        raise ValueError('MPC/activity/point-mass spectral extension not yet qualified')
    for eid,e in elements:
        e._check(model.mesh)
        if model.materials.get(e.material_name) is not e.section:
            raise ValueError('native section ownership changed')
        if not fibre and np.any(element_states[eid]['load_pattern'].density(eid)):
            raise ValueError('nonconservative distributed couples forbidden in conservative spectra')
    _,_,transform,offset,independent,info = build_constraint_transformation(
        sparse.eye(nodal,format='csr'),np.zeros(nodal),model)
    fixed = set(range(nodal))-set(independent)
    if info['num_mpc_constraints'] or np.any(offset) or not np.array_equal(transform.toarray(),np.eye(nodal)[:,independent]):
        raise ValueError('homogeneous support selector required')
    if np.any(total[list(fixed)]):
        raise ValueError('accepted displacement violates spectral supports')
    if any(len(set(row[3:])&fixed) not in (0,3) for row in node_maps):
        raise ValueError('partial rotational constraints need a separate chart contract')

    def snapshot():
        return sha(dict(elements=[(i,e.to_dict(),tuple(e.get_dof_mapping(model.mesh))) for i,e in elements],
            nodes=[(i,n.coords()) for i,n in sorted(model.mesh.nodes.items())],
            states=element_states,inertias=section_inertias,displacement=displacement,
            forces=nodal_spatial_dead_forces,boundaries=[vars(b) for b in model.boundary_conditions],
            constraints=model.constraint_equations,activity=model.mesh.element_activity,
            point_masses=model.mesh.point_masses,constrained=sorted(model.mesh.dof_manager._constrained_dofs)))
    identity = snapshot()

    def activity():
        cancellation_safe_point(cancellation_token,'native_generalized_modal.guard')
        if monotonic()-started>600:
            raise TimeoutError('native generalized spectral construction deadline')

    def check():
        activity()
        material_guard()
        if tuple(sorted(model.mesh.elements.items()))!=elements or snapshot()!=identity:
            raise ValueError('native generalized spectral inputs changed')
        for _,e in elements:
            e._check(model.mesh)
            if model.materials.get(e.material_name) is not e.section:
                raise ValueError('native generalized spectral section ownership changed')

    check()
    owned = deepcopy(element_states)
    shared = {}; signatures = set()
    for eid,e in elements:
        check()
        dofs = list(e.get_dof_mapping(model.mesh))
        owned[eid] = e.validate_model_bound_nonlinear_state(model.mesh,e.section,
            owned[eid],1,expected_committed_total_u=total[dofs])
        if not fibre:signatures.add(owned[eid]['load_pattern'].signature)
        for index,node in enumerate(e.node_ids):
            frame = owned[eid]['committed_nodal_rotation_matrices'][index]
            if node in shared and not np.array_equal(shared[node],frame):
                raise ValueError('shared physical nodal rotation authority differs')
            shared[node] = frame
    if not fibre and len(signatures)!=1:
        raise ValueError('one complete accepted distributed load pattern required')
    size = nodal+6*len(elements)
    k=np.zeros((size,size));m=np.zeros_like(k);force=np.zeros(size)
    layout=[];operators=[]
    for index,(eid,e) in enumerate(elements):
        check();op=operator_factory(e,owned[eid],inertias[eid],activity)
        check()
        internal=tuple(range(nodal+6*index,nodal+6*index+6))
        slots=tuple(e.get_dof_mapping(model.mesh))+internal
        k[np.ix_(slots,slots)]+=op.stiffness;m[np.ix_(slots,slots)]+=op.mass
        force[list(slots)]+=op.net_residual
        layout.append((eid,internal));operators.append((eid,op))
    free=tuple(int(i) for i in independent)+tuple(range(nodal,size))
    algebraic=tuple(i for i in free if i in rotations)
    residual=force-np.r_[external,np.zeros(size-nodal)]
    length=max(float(np.linalg.norm(e.operator.reference.coordinates[-1]-e.operator.reference.coordinates[0])) for _,e in elements)
    residual[list(rotations)+list(range(nodal,size))]/=length
    if np.linalg.norm(residual[list(free)])>1e-11*max(1.,np.linalg.norm(external),np.linalg.norm(force)):
        raise ValueError('free conservative equilibrium required for current-rest spectrum')
    if _relative(k,k.T)>1e-11:
        raise ValueError('assembled conservative Hessian is not symmetric')
    if np.any(m[:,rotations]) or np.any(m[rotations,:]):
        raise ValueError('nodal trace inertia must be exactly zero')
    body=dict(stiffness=_owned(k),mass=_owned(m),net_residual=_owned(force),
        nodal_spatial_dead_forces=external,free_dofs=free,algebraic_dofs=algebraic,
        internal_layout=tuple(layout),operators=tuple(operators))
    packet=Pencil(**body,identity=sha(dict(policy=policy,inputs=identity,**body)),policy=policy,material_policy=material_policy)
    packet_hash=sha(packet)

    def guard():
        check()
        if sha(packet)!=packet_hash:
            raise ValueError('native generalized spectral packet changed')
    guard()
    return packet,guard


def solve_modes(model, element_states, displacement, section_inertias,
        nodal_spatial_dead_forces, *, num_modes=6,cancellation_token=None):
    packet,guard=prepare(model,element_states,displacement,section_inertias,
        nodal_spatial_dead_forces,cancellation_token=cancellation_token)
    result=solve_stationary_spectrum(packet.stiffness,packet.mass,packet.free_dofs,
        packet.algebraic_dofs,num_modes=num_modes,cancellation_token=cancellation_token)
    guard()
    return packet,result
