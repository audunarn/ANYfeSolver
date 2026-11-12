"""Virgin elastic Q4 reference operator for mixed-model compilation only.

The qualified zero-state baseline is pulled through the existing consistent
corotational map. No constitutive trial, tangent lagging or runtime cache is
introduced. Noneligible states retain the ordinary full response.
"""
from __future__ import annotations

import numpy as np

from .e4_pl_element import QualifiedE4PLShellElement
from .corotational import (
    _corotational_cache, _corotational_deformation_state,
    _consistent_corotational_tangent, _dense_block_rotation,
)
from .corotational_performance import (
    note_dense_consistent_rotation, rotate_corotational_force_blocks,
)


def elastic_reference_response(model, element_id, element, predecessor):
    """Return an owned reference response, or None to require full evaluation.

    Elastic inherited moduli use closed-form thickness integration. Therefore
    replacing ``legacy_3 + (qualified - legacy_5)`` by ``qualified`` changes
    only binary64 addition/subtraction rounding, not the zero-state operator.
    J2/Hill histories deliberately keep their original constitutive evaluation.
    """
    if (type(element) is not QualifiedE4PLShellElement or
            element.total_dofs != 24 or element.shell_section is not None or
            type(predecessor) is not dict or
            set(predecessor) != {"plastic_strain", "alpha"}):
        return None
    points = len(element.gauss_points)*3
    for name, shape in (("plastic_strain", (points, 3)), ("alpha", (points,))):
        value = predecessor[name]
        if (type(value) is not np.ndarray or value.shape != shape or
                value.dtype != np.dtype(float) or np.any(value != 0.)):
            return None
    material = model.get_material(element.material_name)
    if element._requires_algorithmic_return_map_origin(material):
        return None
    reference = _corotational_cache(model).get(int(element_id))
    if reference is None or reference.category != "shell":
        return None
    # This seam is strictly the flat mixed-owner reference, not a warped-shell
    # replacement for its intentionally retained inherited nonlinear baseline.
    if np.any(reference.coordinates[:, 2] != reference.coordinates[0, 2]):
        return None
    zero = np.zeros(24)
    deformation, rotation, deformed = _corotational_deformation_state(
        reference, element, zero)
    if np.any(deformation != 0.):
        return None
    # The public stiffness view borrows a guarded cache. Own it before the
    # subsequent frame evaluations; no borrowed buffer survives that operation.
    local = np.array(element.compute_stiffness_matrix(model.mesh, material), copy=True)
    force = rotate_corotational_force_blocks(np.zeros(24), rotation)
    note_dense_consistent_rotation()
    tangent = _consistent_corotational_tangent(
        reference, element, deformed, np.zeros((4, 3)), rotation,
        _dense_block_rotation(rotation, 4), force, local)
    # The elastic candidate equals its virgin parent but must not alias it.
    candidate = {name: value.copy() for name, value in predecessor.items()}
    return force, tangent, candidate
