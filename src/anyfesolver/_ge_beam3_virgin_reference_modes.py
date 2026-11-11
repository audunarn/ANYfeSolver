"""Private V4 virgin-reference modal adapter; never a prestress fallback."""

from copy import deepcopy

import numpy as np

from ._ge_beam3_loaded_modal import prepare
from ._ge_beam3_p5_loads import NativeP5BeamElement
from ._ge_beam3_p5_loads.core import canonical
from ._ge_beam3_p5_centered.mass import reference_kinetic_factors
from ._native_reference_factor_spectrum import solve_reference_factor_spectrum
from .control import cancellation_safe_point


def solve_virgin_reference_modes(model, element_states, displacement, section_inertias, *,
                                 num_modes=6, cancellation_token=None):
    """Require exact fresh native states and zero displacement before factors.

    The preserved loaded-state capture validates model/section ownership,
    complete six-DOF maps, reference pose and homogeneous supports. Its dense
    matrix is checked for consistency, but NOT used for the eigenproblem.
    Explicitly rejects loaded, rotated, history-bearing or post-commit states;
    no hidden fallback from a failed signed/prestressed solve is installed.
    """
    cancellation_safe_point(cancellation_token, 'virgin_reference_modes.start')
    total = np.array(displacement, dtype=float, copy=True)
    if total.shape != (model.mesh.dof_manager.total_dofs,) or not np.isfinite(total).all() or np.any(total):
        raise ValueError('exact zero virgin-reference displacement required')
    elements = tuple(sorted(model.mesh.elements.items()))
    if not elements or total.size+6*len(elements) > 256:
        raise ValueError('bounded virgin-reference model required before mechanics')
    if type(element_states) is not dict or set(element_states) != {eid for eid, _ in elements}:
        raise ValueError('complete virgin state map required')
    states, inertias = deepcopy(element_states), deepcopy(section_inertias)
    for eid, element in elements:
        if type(element) is not NativeP5BeamElement:
            raise ValueError('exact native V4 reference elements required')
        fresh = element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
        if canonical(states[eid]) != canonical(fresh):
            raise ValueError('exact fresh virgin state required; no prestress/history fallback')
    packet, guard = prepare(model, states, total, inertias, load_parameter=0.,
        cancellation_token=cancellation_token)
    elastic_rows, kinetic_rows = [], []
    for eid, internal in packet.internal_layout:
        guard(); element = model.mesh.elements[eid]
        factor = reference_kinetic_factors(element.core.reference, element.core.section._elastic,
            inertias[eid], order=element.core.order)
        slots = tuple(int(i) for i in element.get_dof_mapping(model.mesh))+internal
        for local, rows in ((factor.uncondensed_stiffness_factor, elastic_rows), (factor.full, kinetic_rows)):
            full = np.zeros((local.shape[0], len(packet.mass)))
            full[:, slots] = local; rows.append(full)
    elastic, kinetic = np.vstack(elastic_rows), np.vstack(kinetic_rows)
    for made, saved in ((elastic.T@elastic, packet.stiffness), (kinetic.T@kinetic, packet.mass)):
        if np.linalg.norm(made-saved) > 1e-11*max(1., np.linalg.norm(saved)):
            raise ValueError('virgin factors disagree with native reference operator')
    guard()
    modes = solve_reference_factor_spectrum(elastic, kinetic, packet.free_dofs, packet.algebraic_dofs,
        num_modes=num_modes, cancellation_token=cancellation_token)
    guard()
    return packet, modes
