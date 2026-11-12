"""Private FEModel container for the retained-fibre driver, not a public alias."""
import numpy as np
from .elements import Element
from ._ge_beam3_retained_fibre import RetainedFibreOperator, POLICY
from ._ge_beam3_p5_seeded.core import sha


class NativeRetainedFibreElement(Element):
    num_nodes = 3
    dofs_per_node = 6
    formulation_id = POLICY
    production_qualified = False

    def __init__(self, element_id, node_ids, reference, section, *, order):
        if (type(element_id) is not int or element_id <= 0 or type(node_ids) is not tuple
                or len(node_ids) != 3 or any(type(i) is not int or i <= 0 for i in node_ids)
                or len(set(node_ids)) != 3):
            raise ValueError('explicit positive element ID and three distinct node IDs required')
        super().__init__(element_id, node_ids, 'private-retained-fibre-'+str(element_id))
        self.operator = RetainedFibreOperator(reference, section, order=order)
        self.section = section
        self.driver_identity = sha(self.to_dict())

    def to_dict(self):
        return dict(formulation_id=POLICY, element_id=self.element_id, node_ids=self.node_ids,
            material_name=self.material_name, operator=self.operator.identity,
            reference=self.operator.reference.fingerprint(), section=self.section.identity,
            order=self.operator.order, production_qualified=False)

    def check(self, mesh):
        self.operator.guard()
        if type(self) is not NativeRetainedFibreElement or mesh.elements.get(self.element_id) is not self:
            raise ValueError('retained fibre element ownership mismatch')
        if (self.formulation_id != POLICY or self.production_qualified is not False
                or type(self.num_nodes) is not int or self.num_nodes != 3
                or type(self.dofs_per_node) is not int or self.dofs_per_node != 6):
            raise ValueError('retained fibre formulation/provenance traits changed')
        if self.section is not self.operator.section or sha(self.to_dict()) != self.driver_identity:
            raise ValueError('retained fibre element/section identity changed')
        coordinates = self.get_node_coordinates(mesh)
        if not np.array_equal(coordinates, self.operator.reference.coordinates):
            raise ValueError('retained fibre node/reference mismatch')
        slots = self.get_dof_mapping(mesh)
        if len(slots) != 18 or len(set(slots)) != 18: raise ValueError('eighteen distinct beam nodal DOFs required')

    def get_node_coordinates(self, mesh):
        if any(i not in mesh.nodes for i in self.node_ids): raise ValueError('retained fibre node missing')
        return np.array([mesh.nodes[i].coords() for i in self.node_ids])

    def _unsupported(self, *args, **kwargs):
        raise ValueError('private physical-fibre element requires the retained-fibre driver; this route is not qualified')

    compute_stiffness_matrix = _unsupported
    compute_mass_matrix = _unsupported
    compute_geometric_stiffness_matrix = _unsupported
    compute_internal_forces = _unsupported
    compute_nonlinear_response = _unsupported
    compute_stresses = _unsupported
