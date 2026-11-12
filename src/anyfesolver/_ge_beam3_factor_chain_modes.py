# V5 private successor: preserved mechanical split, retained kinetic factor.
"""Private signed modes at accepted native V5 elastic-interior states.

Preserved V5 equations retain their common kinematic maps through constitutive factors.
No state advancement, positive reference fallback, public routing or buckling
factor qualification. Cell inertia and both signs of squared frequency remain.
"""
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic

from ._ge_beam3_seeded_loaded_modal import prepare
from ._native_reference_modal import _owned
from ._native_factor_chain_modes import solve_factor_chain_modes
from ._ge_beam3_lifted_kinetic_factor import current_lifted_kinetic_factor
from ._ge_beam3_p5_seeded.core import sha
from ._ge_beam3_p5_seeded.state import LOAD_POLICY
from .control import cancellation_safe_point

import numpy as np



from ._ge_beam3_shared_kinematic_split import shared_kinematic_chain
from scipy.linalg import block_diag


@dataclass(frozen=True)
class SignedLoadedModes:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    numerical_brackets: np.ndarray
    spectral_residual: float
    full_backward_residual: float
    original_ritz_residual: float
    operator_identity: str
    external_work_identity: str
    nodal_spatial_dead_forces: np.ndarray
    policy: str = 'GE_BEAM3_V5_SHARED_KINEMATIC_SIGNED_LOADED_MODES_V1'
    mass_policy: str = 'CURRENT_LIFTED_KINETIC_FIELD_LINEARIZED_AT_REST'
    material_policy: str = 'ACCEPTED_ELASTIC_INTERIOR_NO_HISTORY_ADVANCE'
    negative_eigenvalues_retained: bool = True
    certified_intervals: bool = False
    buckling_factor_authorized: bool = False
    production_qualified: bool = False


def solve_signed_loaded_modes(model, element_states, displacement, section_inertias,
        nodal_spatial_dead_forces, *, load_parameter, bounds, num_modes=6,
        root_width=1e-10, cancellation_token=None):
    """Detached native snapshot. Actual nodal forces, not a unit load pattern."""
    started = monotonic()
    def checkpoint(stage):
        cancellation_safe_point(cancellation_token, 'signed_loaded_modes.'+stage)
        if monotonic()-started > 600.: raise ValueError('signed loaded modes time limit')
    checkpoint('capture')
    states = deepcopy(element_states)
    inertias = deepcopy(section_inertias)
    packet, guard = prepare(model, states, displacement, inertias,
        load_parameter=load_parameter, cancellation_token=cancellation_token)
    if not all(op.elastic_interior for _, op in packet.operators):
        raise ValueError('plastic or yield-boundary vibration branch is not authorized')
    force = _owned(nodal_spatial_dead_forces); nodal = len(packet.displacement)
    if force.shape != (nodal,): raise ValueError('complete actual nodal spatial dead force required')
    rotations = tuple(int(d) for i in sorted(model.mesh.nodes) for d in model.mesh.dof_manager.get_node_dofs(i)[3:])
    if np.any(force[list(rotations)]): raise ValueError('nodal moments need a separate load tangent')
    residual = packet.net_residual-np.r_[force, np.zeros(len(packet.net_residual)-nodal)]
    for dof in rotations+tuple(range(nodal, len(residual))):
        residual[dof] /= max(e.core.length for e in model.mesh.elements.values())
    if np.linalg.norm(residual[list(packet.free_dofs)]) > 1e-11*max(1., np.linalg.norm(force)):
        raise ValueError('loaded free equilibrium required for modal interpretation')
    left_rows = []; right_rows = []; kinetic_rows = []; geometric = np.zeros_like(packet.stiffness)
    for eid, internal in packet.internal_layout:
        checkpoint('element.capture'); guard()
        element = model.mesh.elements[eid]
        dofs = tuple(int(i) for i in element.get_dof_mapping(model.mesh)); slots = dofs+internal
        state = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[eid], 1,
            expected_committed_total_u=packet.displacement[list(dofs)])
        left, right, local_g = shared_kinematic_chain(element, state['material_state'], checkpoint)
        row = np.zeros((right.shape[0], len(geometric))); row[:, slots] = right
        left_rows.append(left); right_rows.append(row); geometric[np.ix_(slots, slots)] += local_g
        local_kinetic = current_lifted_kinetic_factor(element.core.reference, inertias[eid],
            state['material_state']['response'].local_rotations, order=element.core.order)
        kinetic_row = np.zeros((len(local_kinetic), len(geometric))); kinetic_row[:, slots] = local_kinetic
        kinetic_rows.append(kinetic_row)
        guard()
    left_factor = block_diag(*left_rows); right_factor = np.vstack(right_rows)
    factor = left_factor@right_factor
    error = np.linalg.norm(factor.T@factor+geometric-packet.stiffness)
    if not np.isfinite(error) or error > 1e-11*max(1., np.linalg.norm(packet.stiffness)):
        raise ValueError('signed native Hessian reconstruction failed')
    checkpoint('kernel'); guard()
    kinetic = np.vstack(kinetic_rows)
    if np.linalg.norm(kinetic.T@kinetic-packet.mass) > 1e-11*max(1., np.linalg.norm(packet.mass)):
        raise ValueError('current kinetic factor reconstruction failed')
    result = solve_factor_chain_modes(left_factor, right_factor, geometric, kinetic, packet.free_dofs,
        packet.algebraic_dofs, bounds=bounds, num_modes=num_modes, root_width=root_width,
        cancellation_token=cancellation_token)
    checkpoint('output'); guard()
    work = sha(dict(policy=LOAD_POLICY, parameter=packet.load_parameter, nodal_forces=force,
        element_line_patterns=[(i, e.core.line_force) for i, e in sorted(model.mesh.elements.items())]))
    return packet, SignedLoadedModes(result.eigenvalues, result.full_modes, result.numerical_brackets,
        result.spectral_residual, result.full_backward_residual, result.original_ritz_residual, packet.identity, work, force)
