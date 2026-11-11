"""Private translation-owned conservative current-rest factor capture.

The displacement controller locates an equilibrium; its constraint is not a
physical stiffness term or modal support. Never reinterpret another state type.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import numpy as np
from scipy.linalg import block_diag
from . import _ge_beam3_retained_translation_control as control
from .control import cancellation_safe_point
from ._ge_beam3_retained_generalized_modal import Pencil, _elastic_interior
from ._ge_beam3_native_generalized_factor_modal import compliance_factor, kinetic_factor
from ._ge_beam3_native_generalized_modal import _inertia, _relative
from ._ge_beam3_p5_centered.mass import current_rest_mass
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._native_reference_modal import _owned
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes

POLICY = 'GE_BEAM3_RETAINED_TRANSLATION_OWNED_CURRENT_REST_FACTOR_CHAIN_V1'


def _factor_size(nodal, elements):
    from ._ge_beam3_refinement_capacity import retained_limits
    maximum,_,full=retained_limits()
    if (type(nodal) is not int or type(elements) is not int or not 1<=elements<=maximum
            or not 6<=nodal or nodal%6 or nodal+6*elements>full):
        raise ValueError('bounded retained spectral model required')
    return nodal+6*elements


@dataclass(frozen=True)
class TranslationPencil(Pencil):
    parameter: float = 0.
    completed_targets: int = 0
    displacement_target: float = 0.
    control_constraint_in_physical_stiffness: bool = False


def prepare(model, program, checkpoint, section_inertias, *, expected_sha256, cancellation_token=None):
    cancellation_safe_point(cancellation_token, 'translation-modal.capture')
    if type(program) is not control.Program: raise ValueError('exact translation program required')
    program.__post_init__(); program.nodal_forces.require(model.mesh)
    if type(checkpoint) is not bytes or type(expected_sha256) is not str or sha256(checkpoint).hexdigest() != expected_sha256:
        raise ValueError('external translation checkpoint authority mismatch')
    ids = set(model.mesh.elements)
    if type(section_inertias) is not dict or set(section_inertias) != ids or any(type(i) is not int for i in section_inertias):
        raise ValueError('complete exact section inertia map required')
    inertias = {eid:_inertia(section_inertias[eid]) for eid in sorted(ids)}
    inertia_identity = sha(section_inertias)
    context = control.Context(model, program)
    state, records = context.restore(checkpoint, expected_sha256=expected_sha256)
    issued = context._require_issued(state); before = canonical(state)
    if context.checkpoint(records) != checkpoint: raise ValueError('translation replay changed checkpoint')
    physical = context.physical

    def check():
        cancellation_safe_point(cancellation_token, 'translation-modal.guard')
        context._require_issued(state, expected_snapshot=before)
        if sha(section_inertias) != inertia_identity:
            raise ValueError('translation-modal inputs changed')

    check()
    nodal = physical.nodal_count; size = _factor_size(nodal,len(physical.elements))
    lefts = []; rights = []; kinetics = []; errors = []; layout = []
    geometric = np.zeros((size,size)); mass = np.zeros_like(geometric)
    force = np.zeros(size); full_force = np.zeros(physical.count)
    mechanical = state.mechanical
    for i, (eid, element) in enumerate(physical.elements):
        check(); core = element.operator; nodes = physical.nodes[i]
        response = core.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes],
            mechanical.nodal_frames[nodes], mechanical.cell_rotations[i], mechanical.resultants[i],
            origin=state.histories[i], check=check)
        _elastic_interior(core, response, state.histories[i])
        # Only spatial dead nodal loads are admitted. Their Hessian is zero;
        # neither the numerical control row nor its load column enters K.
        h = response.hessian+response.hessian_low; residual = response.residual
        if not np.array_equal(h[:24,24:].T, h[24:,:24]):
            raise ValueError('work-conjugate retained kinematic blocks required')
        if _relative(h[:24,:24], h[:24,:24].T) > 1e-11:
            raise ValueError('symmetric conservative geometric Hessian required')
        left, error = compliance_factor(-h[24:,24:], check)
        internal = tuple(range(nodal+6*i,nodal+6*i+6))
        slots = tuple(element.get_dof_mapping(model.mesh))+internal
        right = np.zeros((18,size)); right[:,slots] = h[24:,:24]
        kinetic = kinetic_factor(core.reference, inertias[eid], mechanical.cell_rotations[i], check)
        physical_mass = current_rest_mass(core.reference, inertias[eid], 24,
            mechanical.positions[nodes], mechanical.position_low[nodes], mechanical.cell_rotations[i])
        if _relative(kinetic.T@kinetic, physical_mass) > 1e-11:
            raise ValueError('factor changed physical current-rest inertia')
        lifted = np.zeros((len(kinetic),size)); lifted[:,slots] = kinetic
        lefts.append(left); rights.append(right); kinetics.append(lifted); errors.append(error)
        geometric[np.ix_(slots,slots)] += h[:24,:24]; mass[np.ix_(slots,slots)] += physical_mass
        force[list(slots)] += residual[:24]; full_force[physical.slots[i]] += residual
        layout.append((eid,internal))
    external = physical.nodal_external(state.parameter)
    force[:nodal] -= external; full_force[:nodal] -= external
    accepted_residual = np.array(json.loads(issued)['residual'])
    if np.linalg.norm((full_force-accepted_residual)/physical.scale) > 1e-11:
        raise ValueError('current-rest replay changed accepted controlled equilibrium')
    if max(np.linalg.norm((full_force/physical.scale)[physical.equilibrium]),
           np.linalg.norm((full_force/physical.scale)[physical.compatibility])) > 1e-11:
        raise ValueError('free compatible conservative controlled equilibrium required')
    # The control node remains physically free. Only model boundary conditions
    # constrain perturbations, not the auxiliary displacement equation.
    free = tuple(int(d) for d in physical.free if d < nodal)+tuple(range(nodal,size))
    rotations = tuple(6*i+j for i in range(len(physical.node_ids)) for j in (3,4,5))
    algebraic = tuple(d for d in free if d in rotations)
    if np.any(mass[:,rotations]) or np.any(mass[rotations,:]):
        raise ValueError('nodal trace inertia must be exactly zero')
    left = block_diag(*lefts); right = np.vstack(rights); kinetic = np.vstack(kinetics)
    expanded = left@right
    target = 0. if state.completed_targets == 0 else program.targets[state.completed_targets-1]
    body = dict(left=_owned(left), right=_owned(right), geometric=_owned(geometric), kinetic=_owned(kinetic),
        stiffness=_owned(expanded.T@expanded+geometric), mass=_owned(mass), net_residual=_owned(force),
        free_dofs=free, algebraic_dofs=algebraic, internal_layout=tuple(layout), compliance_errors=tuple(errors),
        checkpoint_sha256=expected_sha256, model_sha256=context.identity, parameter=state.parameter,
        completed_targets=state.completed_targets, displacement_target=target)
    packet = TranslationPencil(**body, identity=sha(dict(policy=POLICY,**body)), policy=POLICY)
    identity = sha(packet)

    def guard():
        check()
        if sha(packet) != identity: raise ValueError('translation-modal factor packet changed')

    guard(); return packet, guard


def solve_modes(model, program, checkpoint, section_inertias, *, expected_sha256, bounds,
                num_modes=6, root_width=1e-10, relative_width=1e-12, cancellation_token=None):
    packet, guard = prepare(model,program,checkpoint,section_inertias,expected_sha256=expected_sha256,
                            cancellation_token=cancellation_token)
    result = solve_paired_factor_chain_modes(packet.left,packet.right,packet.geometric,packet.kinetic,
        packet.free_dofs,packet.algebraic_dofs,bounds=bounds,num_modes=num_modes,
        root_width=root_width,relative_width=relative_width,cancellation_token=cancellation_token)
    guard(); return packet, result
