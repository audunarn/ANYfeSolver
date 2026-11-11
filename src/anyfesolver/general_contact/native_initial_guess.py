"""Coordinates-only first-target initialization for the retained fibre owner.

This is neither checkpoint continuation nor a material-state transfer. The fine
owner reconstructs the trial using its ordinary update and admission contracts.
"""
from dataclasses import dataclass
from math import fsum
import numpy as np
from scipy.spatial.transform import Rotation
from anyfesolver.control import cancellation_safe_point
from anyfesolver._ge_beam3_p5_seeded.core import sha,canonical
from .native_retained_station import _readonly


@dataclass(frozen=True)
class NativeFirstTargetGuess:
    owner_identity: str
    structure_identity: str
    source_owner_identity: str
    source_checkpoint_sha256: str
    increment: np.ndarray
    fingerprint: str

    def descriptor(self):
        return {key:getattr(self,key) for key in self.__dataclass_fields__ if key!='fingerprint'}


def structure_identity(owner):
    model=owner.shell_model
    return sha(dict(native=owner.context.identity,
        shell_stiffness=owner.shell_stiffness,free=owner.shell_free,fixed=owner.shell_fixed,
        nodes=[(node,model.mesh.nodes[node].coords(),model.mesh.dof_manager.get_node_dofs(node))
               for node in sorted(model.mesh.nodes)],
        skins=[dict(element=p.work.shell.element_id,body=p.work.shell.body_id,
                    nodes=p.work.shell.node_ids,side=p.work.shell.side,
                    reference=p.work.shell.reference_positions,director=p.work.shell.director,
                    offset=p.work.shell.physical_offset,work_map=p.work.shell.work_map.toarray())
               for p in owner.pairs]))


def _require_virgin_initial(owner):
    native=owner.context.initial
    if (owner.initial.completed_targets!=0 or np.any(owner.initial.shell_displacement)
            or canonical(owner.initial.mechanical.descriptor())!=canonical(native.mechanical.descriptor())
            or canonical(owner.initial.native_histories)!=canonical(native.histories)):
        raise ValueError('first-target guess requires the bound virgin initial state')


def make_guess(owner,source,checkpoint,expected_sha256):
    if type(source) is not type(owner) or not owner._fibre_mode or not owner._curved_mode:
        raise ValueError('first-target guess requires exact curved physical-fibre owners')
    owner._guard();source._guard()
    if not source._fibre_mode or not source._curved_mode:
        raise ValueError('first-target guess source formulation')
    _require_virgin_initial(owner);_require_virgin_initial(source)
    structure=structure_identity(owner)
    if structure_identity(source)!=structure:
        raise ValueError('first-target guess structural identity mismatch')
    # Replay the SOURCE definition. No source checkpoint is restored in owner.
    with source.context.operation(None):
        state,_=source._restore(checkpoint,expected_sha256)
    if state.completed_targets!=1:
        raise ValueError('first-target guess requires a first-target source checkpoint')
    target=owner.context.make(state.mechanical.descriptor())
    initial=owner.initial.mechanical
    step=np.zeros(owner.context.count+len(owner.initial.shell_displacement))
    for node in range(len(owner.context.node_ids)):
        for axis in range(3):
            step[6*node+axis]=fsum((target.positions[node,axis],target.position_low[node,axis],
                                  -initial.positions[node,axis],-initial.position_low[node,axis]))
        if not np.array_equal(target.nodal_frames[node],initial.nodal_frames[node]):
            step[6*node+3:6*node+6]=Rotation.from_matrix(
                target.nodal_frames[node]@initial.nodal_frames[node].T).as_rotvec()
    for beam in range(len(owner.context.elements)):
        first=owner.context.nodal_count+24*beam
        for cell in (0,1):
            if not np.array_equal(target.cell_rotations[beam,cell],initial.cell_rotations[beam,cell]):
                step[first+3*cell:first+3*cell+3]=Rotation.from_matrix(
                    target.cell_rotations[beam,cell]@initial.cell_rotations[beam,cell].T).as_rotvec()
        step[first+6:first+24]=target.resultants[beam]-initial.resultants[beam]
    step[owner.context.count:]=state.shell_displacement-owner.initial.shell_displacement
    body=dict(owner_identity=owner.identity,structure_identity=structure,
              source_owner_identity=source.identity,source_checkpoint_sha256=expected_sha256,
              increment=_readonly(step))
    owner._guard()
    _require_virgin_initial(owner)
    return NativeFirstTargetGuess(**body,fingerprint=sha(body))


def prepare_trial(owner,guess,cancellation_token):
    cancellation_safe_point(cancellation_token,'native first-target guess')
    owner._guard()
    _require_virgin_initial(owner)
    if type(guess) is not NativeFirstTargetGuess:
        raise ValueError('first-target guess binding mismatch')
    step=np.array(guess.increment,dtype=float,copy=True)
    descriptor={**guess.descriptor(),'increment':step}
    if (guess.owner_identity!=owner.identity
            or guess.structure_identity!=structure_identity(owner)
            or sha(descriptor)!=guess.fingerprint):
        raise ValueError('first-target guess binding mismatch')
    n=owner.context.count
    if step.shape!=(n+len(owner.initial.shell_displacement),) or not np.isfinite(step).all():
        raise ValueError('first-target guess finite coordinate shape')
    if np.any(step[n+owner.shell_fixed]):
        raise ValueError('first-target guess violates shell supports')
    angular=[step[6*i+3:6*i+6] for i in range(len(owner.context.node_ids))]
    angular.extend(step[owner.context.nodal_count+24*b+3*c:owner.context.nodal_count+24*b+3*c+3]
                   for b in range(len(owner.context.elements)) for c in (0,1))
    if any(np.linalg.norm(v)>=.45*np.pi for v in angular):
        raise ValueError('first-target guess requires a bounded rotation chart')
    start=owner.initial.mechanical;u0=owner.initial.shell_displacement
    trial=owner.context.advance(start,step[:n]);u=u0+step[n:]
    if not owner._candidate_geometry_admissible(trial,u):
        raise ValueError('first-target guess endpoint geometry rejected')
    contact=owner._contact(trial,u)
    if not owner._contact_admissible(contact):
        raise ValueError('first-target guess endpoint gap rejected')
    if not owner._swept_cancellable(start,u0,step,trial,u,cancellation_token=cancellation_token):
        raise ValueError('first-target guess swept path rejected')
    cancellation_safe_point(cancellation_token,'native first-target guess admitted')
    return trial,u
