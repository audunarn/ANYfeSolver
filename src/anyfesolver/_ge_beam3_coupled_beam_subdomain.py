"""Retained generalized beam subdomain whose support may come from the shell.

Reuses the exact retained assembly/update functions, not standalone accepted
state issuance. The reference seed is not an accepted checkpoint. Only the
global coupled owner may test equilibrium and commit it. Standalone support
requirements and checkpoint schemas remain unchanged.
"""
from dataclasses import dataclass, field
from time import monotonic
import numpy as np
from scipy import sparse

from .assembly import build_constraint_transformation
from .constraint_audit import require_valid_constraints
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from ._ge_beam3_retained_generalized_state import Context, Program, _retained_size
from ._ge_beam3_refinement_capacity import retained_limits
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._native_reference_modal import _owned


@dataclass(frozen=True)
class ReferenceSeed:
    mechanical: object
    histories: tuple
    accepted_state: bool = field(default=False, init=False)


def _model_identity(model):
    elements = tuple(sorted(model.mesh.elements.items())); n = model.mesh.dof_manager.total_dofs
    if not 1 <= len(elements) <= retained_limits()[0] or n != 6*len(model.mesh.nodes) or n > 512:
        raise ValueError('bounded coupled beam subdomain required')
    if model.constraint_equations or model.mesh.point_masses or model.mesh.element_activity is not None:
        raise ValueError('subdomain MPC/activity/point-mass policies not admitted')
    for eid, element in elements:
        if (type(element) is not NativeGeneralizedStaticElement or eid != element.element_id
                or model.materials.get(element.material_name) is not element.section):
            raise ValueError('exact coupled beam element/section authority required')
        element._check(model.mesh)
    if set(model.mesh.nodes) != {node for _, e in elements for node in e.node_ids}:
        raise ValueError('unconnected coupled beam nodes')
    fixed = set()
    for boundary in model.boundary_conditions:
        for dof, value in boundary.get_constrained_dofs(model.mesh.dof_manager):
            if value != 0.: raise ValueError('homogeneous coupled beam supports required')
            fixed.add(int(dof))
    for row, node in enumerate(sorted(model.mesh.nodes)):
        dofs = tuple(model.mesh.dof_manager.get_node_dofs(node))
        if dofs != tuple(range(6*row, 6*row+6)):
            raise ValueError('coupled beam node/DOF ordering')
        if len(set(dofs[3:]) & fixed) not in (0, 3):
            raise ValueError('partial coupled beam rotation support not admitted')
    return sha(dict(elements=[(i, e.to_dict()) for i, e in elements],
        nodes=[(i, node.coords(), list(model.mesh.dof_manager.get_node_dofs(i)))
            for i, node in sorted(model.mesh.nodes.items())],
        boundaries=[vars(b) for b in model.boundary_conditions], fixed=sorted(fixed)))


class CoupledBeamSubdomain:
    # Exact existing kernels, not copied mechanical implementations.
    make = Context.make
    histories = Context.histories
    assemble = Context.assemble
    advance = Context.advance

    def __init__(self, model, program):
        if type(program) is not Program:
            raise ValueError('exact distributed subdomain program required')
        program.__post_init__(); program.pattern.require(model.mesh)
        self.started = monotonic(); self.model, self.program = model, program
        self.model_identity = _model_identity(model)
        self.program_bytes = canonical(program)
        self.support_identity = canonical(require_valid_constraints(model))
        self.elements = tuple(sorted(model.mesh.elements.items()))
        self.node_ids = tuple(sorted(model.mesh.nodes)); self.index = {node: i for i, node in enumerate(self.node_ids)}
        self.nodal_count = model.mesh.dof_manager.total_dofs
        _, _, transform, offset, free, info = build_constraint_transformation(
            sparse.eye(self.nodal_count, format='csr'), np.zeros(self.nodal_count), model)
        if info['slave_dofs'] or np.any(offset) or transform.nnz != len(free) or np.any(transform.data != 1.):
            raise ValueError('exact subdomain supported selection required')
        self.fixed = tuple(i for i in range(self.nodal_count) if i not in free)
        self.count = _retained_size(self.nodal_count, len(self.elements))
        self.free = tuple(map(int, free))+tuple(range(self.nodal_count, self.count))
        self.nodes = []; self.slots = []; frames = [None]*len(self.node_ids)
        self.scale = np.ones(self.count)
        self.length = max(float(np.linalg.norm(e.operator.reference.coordinates[-1]-e.operator.reference.coordinates[0]))
            for _, e in self.elements)
        self.equilibrium = list(map(int, free)); self.compatibility = []
        for i, (_, element) in enumerate(self.elements):
            nodes = [self.index[n] for n in element.node_ids]; first = self.nodal_count+24*i
            self.nodes.append(nodes)
            self.slots.append(list(element.get_dof_mapping(model.mesh))+list(range(first, first+24)))
            self.equilibrium.extend(range(first, first+6)); self.compatibility.extend(range(first+6, first+24))
            self.scale[first:first+12] = self.length
            for j, node in enumerate(nodes):
                q = element.operator.reference.nodal_triads[j]
                if frames[node] is not None and not np.array_equal(frames[node], q):
                    raise ValueError('coupled shared reference frames disagree')
                frames[node] = q
        for i in range(len(self.node_ids)): self.scale[6*i+3:6*i+6] = self.length
        self.scale = _owned(self.scale)
        self.reference_positions = _owned([model.mesh.nodes[i].coords() for i in self.node_ids])
        self.reference_frames = _owned(frames)
        self.identity = sha(dict(policy='GE_BEAM3_RETAINED_GLOBAL_COUPLED_SUBDOMAIN_V1',
            model=self.model_identity, program=program, reference_positions=self.reference_positions,
            reference_frames=self.reference_frames, free=self.free))
        mechanical = self.make(dict(positions=self.reference_positions, position_low=np.zeros_like(self.reference_positions),
            nodal_frames=self.reference_frames, cell_rotations=np.tile(np.eye(3), (len(self.elements), 2, 1, 1)),
            resultants=np.zeros((len(self.elements), 18))))
        self.initial = ReferenceSeed(mechanical, tuple(e.operator.cell.virgin() for _, e in self.elements))
        self._compiled = canonical(self._layout())
        self.guard()

    def _layout(self):
        return dict(identity=self.identity, count=self.count, nodal_count=self.nodal_count,
            node_ids=self.node_ids, index=self.index, nodes=self.nodes, slots=self.slots,
            fixed=self.fixed, free=self.free, scale=self.scale, length=self.length,
            equilibrium=self.equilibrium, compatibility=self.compatibility,
            reference_positions=self.reference_positions, reference_frames=self.reference_frames, initial=self.initial)

    def guard(self):
        if monotonic()-self.started > 120:
            raise RuntimeError('coupled beam subdomain deadline')
        if (_model_identity(self.model) != self.model_identity
                or canonical(require_valid_constraints(self.model)) != self.support_identity
                or canonical(self.program) != self.program_bytes or canonical(self._layout()) != self._compiled):
            raise ValueError('coupled beam subdomain authority changed')
        self.program.pattern.require(self.model.mesh)
