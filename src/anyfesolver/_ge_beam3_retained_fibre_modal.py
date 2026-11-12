"""Physical-fibre-owned conservative current-rest modal factors.

Retains the six physical cell-rotation coordinates, eliminates only the
eighteen inertia-free stress resultants, and leaves nodal rotation traces
massless. Accepted material history is never advanced for a modal probe.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import numpy as np
from scipy.linalg import block_diag

from . import _ge_beam3_retained_fibre_control as control
from ._ge_beam3_retained_generalized_modal import Pencil
from ._ge_beam3_native_generalized_factor_modal import compliance_factor, kinetic_factor
from ._ge_beam3_native_generalized_modal import _inertia, _relative
from ._ge_beam3_p5_centered.mass import current_rest_mass
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._native_reference_modal import _owned
from .control import cancellation_safe_point

POLICY = 'GE_BEAM3_PHYSICAL_FIBRE_OWNED_CURRENT_REST_FACTOR_CHAIN_V1'
MATERIAL_POLICY = 'PHYSICAL_FIBRE_COMMITTED_HISTORY_STRICT_ELASTIC_INTERIOR_V1'


def _section_guard(model):
    from ._ge_beam3_fibre_section import canonical as section_bytes, POLICY as section_policy, MEASURE
    for element in model.mesh.elements.values():
        section=element.section
        actual=sha256(section_bytes(dict(policy=section_policy,measure=MEASURE,
            fibres=section.fibres,background_factor=section.background_factor))).hexdigest()
        if actual!=section.identity:
            raise ValueError('physical-fibre section content/identity mismatch')


def elastic_interior(operator, response, origin):
    """Every physical fibre must admit a two-sided elastic perturbation."""
    if canonical(response.history) != canonical(origin):
        raise ValueError('fibre current-rest probe must not advance committed history')
    data = json.loads(response.material)
    if data['derivative_kind'] != 'CLASSICAL_SMOOTH_BRANCH':
        raise ValueError('fibre modal probe requires a smooth elastic-interior branch')
    with localcontext() as ctx:
        ctx.prec = 80
        if len(data['stations']) != len(origin.stations):
            raise ValueError('complete physical-fibre station history required')
        for station, history in zip(data['stations'], origin.stations, strict=True):
            origins = operator.section._origins(history)
            for field, fibre, (_, accumulated) in zip(station['fibres'], operator.section.fibres, origins, strict=True):
                if field['fibre_id'] != fibre.fibre_id or field['branch'] != 'ELASTIC':
                    raise ValueError('physical-fibre modes require every fibre strictly inside yield')
                stress = sum(map(Decimal.from_float, field['stress']), Decimal(0))
                increment = sum(map(Decimal.from_float, field['plastic_increment']), Decimal(0))
                radius = fibre.curve._at(accumulated)[0]
                margin = Decimal.from_float(64*np.finfo(float).eps)*max(Decimal(1), abs(stress), radius)
                if increment != 0 or radius-abs(stress) <= margin:
                    raise ValueError('physical-fibre modal state is too close to a yield boundary')


@dataclass(frozen=True)
class FibrePencil(Pencil):
    parameter: float = 0.
    completed_targets: int = 0
    displacement_target: float = 0.
    control_constraint_in_physical_stiffness: bool = False


def prepare(model, program, checkpoint, section_inertias, *, expected_sha256, cancellation_token=None):
    cancellation_safe_point(cancellation_token, 'fibre-modal.capture')
    if type(program) is not control.TranslationProgram:
        raise ValueError('exact physical-fibre translation programme required')
    program.__post_init__()
    if type(checkpoint) is not bytes or type(expected_sha256) is not str or sha256(checkpoint).hexdigest() != expected_sha256:
        raise ValueError('external fibre checkpoint authority mismatch')
    ids = set(model.mesh.elements)
    if type(section_inertias) is not dict or set(section_inertias) != ids or any(type(i) is not int for i in section_inertias):
        raise ValueError('complete physical-fibre section inertia map required')
    inertias = {eid:_inertia(section_inertias[eid]) for eid in sorted(ids)}
    inertia_identity = sha(section_inertias)
    _section_guard(model)
    context = control.Context(model, program,
        check=lambda: cancellation_safe_point(cancellation_token, 'fibre-modal.material'))
    state, records = context.restore(checkpoint, expected_sha256=expected_sha256)
    if context.checkpoint(records) != checkpoint:
        raise ValueError('physical-fibre replay changed checkpoint')
    physical = context.physical; layout = context.layout
    before = canonical(state)
    identity = context.identity

    def check():
        cancellation_safe_point(cancellation_token, 'fibre-modal.guard')
        context.guard()
        _section_guard(model)
        if (context.identity != identity or canonical(state) != before
                or sha(section_inertias) != inertia_identity):
            raise ValueError('physical-fibre modal inputs changed')

    check()
    nodal = layout.nodal_count; size = nodal+6*len(layout.elements)
    if not 1 <= size <= 256:
        raise ValueError('physical-fibre modal coordinate bound')
    lefts=[]; rights=[]; kinetics=[]; errors=[]; internal_layout=[]
    geometric=np.zeros((size,size)); mass=np.zeros_like(geometric)
    force=np.zeros(size); full_force=np.zeros(layout.count)
    mechanical=state.mechanical
    for i,(eid,element) in enumerate(layout.elements):
        check(); operator=element.operator; nodes=layout.nodes[i]
        response=operator.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes],
            mechanical.nodal_frames[nodes], mechanical.cell_rotations[i], mechanical.resultants[i],
            origin=state.histories[i], check=check)
        elastic_interior(operator,response,state.histories[i])
        h=response.hessian+response.hessian_low
        if not np.array_equal(h[:24,24:].T,h[24:,:24]) or _relative(h[:24,:24],h[:24,:24].T)>1e-11:
            raise ValueError('symmetric work-conjugate fibre Hessian required')
        left,error=compliance_factor(-h[24:,24:],check)
        internal=tuple(range(nodal+6*i,nodal+6*i+6))
        slots=tuple(element.get_dof_mapping(model.mesh))+internal
        right=np.zeros((18,size));right[:,slots]=h[24:,:24]
        kinetic=kinetic_factor(operator.reference,inertias[eid],mechanical.cell_rotations[i],check)
        physical_mass=current_rest_mass(operator.reference,inertias[eid],24,
            mechanical.positions[nodes],mechanical.position_low[nodes],mechanical.cell_rotations[i])
        if _relative(kinetic.T@kinetic,physical_mass)>1e-11:
            raise ValueError('factor changed physical fibre current-rest inertia')
        lifted=np.zeros((len(kinetic),size));lifted[:,slots]=kinetic
        lefts.append(left);rights.append(right);kinetics.append(lifted);errors.append(error)
        geometric[np.ix_(slots,slots)]+=h[:24,:24];mass[np.ix_(slots,slots)]+=physical_mass
        force[list(slots)]+=response.residual[:24];full_force[layout.slots[i]]+=response.residual
        internal_layout.append((eid,internal))
    external=state.parameter*layout.force
    force[:nodal]-=external;full_force[:nodal]-=external
    original=json.loads(records[-1] if records else context.genesis)['residual']
    if np.linalg.norm((full_force-np.array(original))/layout.scales)>1e-11:
        raise ValueError('fibre current-rest replay changed accepted equilibrium')
    if max(np.linalg.norm((full_force/layout.scales)[layout.equilibrium]),
           np.linalg.norm((full_force/layout.scales)[layout.compatibility]))>1e-11:
        raise ValueError('equilibrated compatible physical-fibre state required')
    free=tuple(int(d) for d in layout.free if d<nodal)+tuple(range(nodal,size))
    rotations=tuple(6*i+j for i in range(len(layout.node_ids)) for j in (3,4,5))
    algebraic=tuple(d for d in free if d in rotations)
    if np.any(mass[:,rotations]) or np.any(mass[rotations,:]):
        raise ValueError('physical-fibre nodal trace inertia must be zero')
    left=block_diag(*lefts);right=np.vstack(rights);kinetic=np.vstack(kinetics);expanded=left@right
    body=dict(left=_owned(left),right=_owned(right),geometric=_owned(geometric),kinetic=_owned(kinetic),
        stiffness=_owned(expanded.T@expanded+geometric),mass=_owned(mass),net_residual=_owned(force),
        free_dofs=free,algebraic_dofs=algebraic,internal_layout=tuple(internal_layout),compliance_errors=tuple(errors),
        checkpoint_sha256=expected_sha256,model_sha256=context.identity,parameter=state.parameter,
        completed_targets=state.completed_targets,
        displacement_target=program.targets[state.completed_targets-1] if records else 0.,material_policy=MATERIAL_POLICY)
    packet=FibrePencil(**body,identity=sha(dict(policy=POLICY,**body)),policy=POLICY)
    packet_identity=sha(packet)
    def guard():
        check()
        if sha(packet)!=packet_identity:raise ValueError('physical-fibre factor packet changed')
    guard();return packet,guard
