"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

import numpy as np

from anyfesolver._native_reference_modal import ReferenceModalBlock, solve_reference_modal
from anyfesolver.control import cancellation_safe_point
from anyfesolver._ge_beam3_p5.element import NativeP5BeamElement
from anyfesolver._ge_beam3_p5.core import sha
from anyfesolver._ge_beam3_p5.mass import reference_kinetic_factors
from anyfesolver._ge_beam3_p5.algebra import validate_section


def prepare(model, section_inertias, *, cancellation_token=None):
    """Capture explicit elastic-origin inertia inputs; do not alter the elements."""
    elements = tuple(sorted(model.mesh.elements.items()))
    if model.mesh.dof_manager.total_dofs + 6*len(elements) > 256:
        raise ValueError("bounded native model required before factor construction")
    if (not elements or type(section_inertias) is not dict
            or set(section_inertias) != {i for i, _ in elements}
            or any(type(e) is not NativeP5BeamElement for _, e in elements)):
        raise ValueError("complete explicit inertia map and exact private P5 elements required")
    inertias = {i: validate_section(section_inertias[i]).copy() for i, _ in elements}

    def snapshot():
        cancellation_safe_point(cancellation_token, "p5_reference_modal.capture")
        if (tuple(sorted(model.mesh.elements.items())) != elements
                or model.mesh.element_activity is not None or model.constraint_equations
                or model.mesh.point_masses):
            raise ValueError("reference modal element ownership/activity/constraint mismatch")
        for _, element in elements:
            element._check(model.mesh)
            if model.materials.get(element.material_name) is not element.core.section:
                raise ValueError("reference section ownership mismatch")
        return sha(dict(elements=[(i, e.to_dict(), list(e.get_dof_mapping(model.mesh))) for i, e in elements],
            nodes=[(i, n.coords()) for i, n in sorted(model.mesh.nodes.items())],
            inertia=[(i, inertias[i]) for i, _ in elements],
            boundaries=[vars(bc) for bc in model.boundary_conditions],
            dofs=model.mesh.dof_manager.total_dofs))

    identity = snapshot()

    frozen_blocks = None

    def guard(stage, observed_blocks=None):
        if snapshot() != identity:
            raise ValueError("reference modal frozen inputs changed at " + stage)
        if observed_blocks is not None and sha(observed_blocks) != frozen_blocks:
            raise ValueError("reference modal block binding mismatch")

    blocks = []
    for i, element in elements:
        guard("before_factor")
        factors = reference_kinetic_factors(element.core.reference, element.core.section._elastic,
                                            inertias[i], order=element.core.order)
        guard("after_factor")
        elastic, kinetic = factors.uncondensed_stiffness_factor, factors.full
        binding = sha(dict(model=identity, element=i, elastic=elastic, kinetic=kinetic,
                           policy="FULL_CELL_INERTIA_NO_GUYAN"))
        blocks.append(ReferenceModalBlock(i, tuple(int(d) for d in element.get_dof_mapping(model.mesh)),
            (3, 4, 5, 9, 10, 11, 15, 16, 17), elastic, kinetic, binding))
    blocks = tuple(blocks)
    frozen_blocks = sha(blocks)
    return blocks, guard


def solve(model, section_inertias, *, num_modes=6, cancellation_token=None):
    blocks, guard = prepare(model, section_inertias, cancellation_token=cancellation_token)
    return solve_reference_modal(model, blocks, check_inputs=guard, num_modes=num_modes,
                                 cancellation_token=cancellation_token)
