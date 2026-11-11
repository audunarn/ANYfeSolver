# V5 private successor: preserved mechanical split, retained kinetic factor.
"""Private signed modes at accepted native V5 elastic-interior states.

Split equations are copied from the hash-preserved development reconstruction.
No state advancement, positive reference fallback, public routing or buckling
factor qualification. Cell inertia and both signs of squared frequency remain.
"""
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic

from ._ge_beam3_seeded_loaded_modal import prepare
from ._native_reference_modal import _owned
from ._native_signed_kinetic_factor_modes import solve_signed_kinetic_factor_modes
from ._ge_beam3_lifted_kinetic_factor import current_lifted_kinetic_factor
from ._ge_beam3_p5_seeded.core import sha
from ._ge_beam3_p5_seeded.state import LOAD_POLICY
from .control import cancellation_safe_point

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log
from anyfesolver._ge_beam3_p5.compensated import compensated_strain
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_p5.algebra import HALVES


def _split(element, inner, checkpoint):
    core = element.core; reference = core.reference; response = inner['response']
    mixed = CenteredStationaryBeam(reference, core.section, position_low=inner['committed_position_low'],
        order=core.order, origins=inner['origins'], line_force=inner['load_parameter']*core.line_force)
    variables = [Jet2.variable(0., i, 24) for i in range(24)]
    vertices = inner['committed_nodal_rotation_matrices']@reference.nodal_triads
    made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(vertices[n], 24)) for n in range(3)]
    rows = []; coupling = np.zeros((12, 24)); compliance = np.zeros((12, 12)); geometric = np.zeros((24, 24))
    for cell, (left, right) in enumerate(HALVES):
        checkpoint('split.cell')
        made_u = matmul(so3_exp(variables[18+3*cell:21+3*cell]), constant_matrix(response.local_rotations[cell], 24))
        z = compensated_strain(made_u, reference.coordinates, inner['committed_positions'],
            inner['committed_position_low'], variables, left, right)
        for endpoint, node, sign in ((0, left, -1), (1, right, 1)):
            frame = matmul(made_u, constant_matrix(reference.nodal_triads[node], 24))
            ell = so3_log(matmul(transpose(frame), made_q[node]))
            for i, value in enumerate(ell):
                coupling[6*cell+3*endpoint+i] += sign*value.gradient
                geometric += sign*response.moments[cell, endpoint, i]*value.hessian
        for index, t, xi, measure, v, offset, _ in mixed._stations[cell]:
            checkpoint('split.station')
            gamma = matvec(constant_matrix(v, 24), z)
            moment = (1-t)*response.moments[cell, 0]+t*response.moments[cell, 1]
            density = core.section.mixed_response([x.value for x in gamma], moment, mixed.origins[cell*core.order+index])
            if density.section.plastic_active:
                raise ValueError('elastic accepted-increment split only')
            jacobian = np.array([x.gradient for x in gamma])
            interpolation = np.zeros((3, 12))
            interpolation[:, 6*cell:6*cell+3] = (1-t)*np.eye(3)
            interpolation[:, 6*cell+3:6*cell+6] = t*np.eye(3)
            h = density.hessian
            rows.append(np.sqrt(measure)*np.linalg.cholesky(h[:3, :3]).T@jacobian)
            compliance -= measure*interpolation.T@h[3:, 3:]@interpolation
            coupling += measure*interpolation.T@h[3:, :3]@jacobian
            for i, value in enumerate(gamma):
                geometric += measure*density.gradient[i]*value.hessian
            # Linear nodal displacement work has zero Hessian. The lifted
            # spatial-dead load contributes only through the cell rotation.
            for i in range(3):
                for k in range(3):
                    geometric -= measure*mixed.line_force[i]*offset[k]*made_u[i][k].hessian
    rows.append(np.linalg.solve(np.linalg.cholesky(compliance), coupling))
    factor = np.vstack(rows)
    if not np.isfinite(factor).all() or not np.isfinite(geometric).all():
        raise ValueError('nonfinite loaded split')
    return factor, geometric



@dataclass(frozen=True)
class SignedLoadedModes:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    numerical_brackets: np.ndarray
    spectral_residual: float
    full_backward_residual: float
    operator_identity: str
    external_work_identity: str
    nodal_spatial_dead_forces: np.ndarray
    policy: str = 'GE_BEAM3_V5_SIGNED_FACTOR_LOADED_MODES_V1'
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
    rows = []; kinetic_rows = []; geometric = np.zeros_like(packet.stiffness)
    for eid, internal in packet.internal_layout:
        checkpoint('element.capture'); guard()
        element = model.mesh.elements[eid]
        dofs = tuple(int(i) for i in element.get_dof_mapping(model.mesh)); slots = dofs+internal
        state = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[eid], 1,
            expected_committed_total_u=packet.displacement[list(dofs)])
        factor, local_g = _split(element, state['material_state'], checkpoint)
        row = np.zeros((factor.shape[0], len(geometric))); row[:, slots] = factor
        rows.append(row); geometric[np.ix_(slots, slots)] += local_g
        local_kinetic = current_lifted_kinetic_factor(element.core.reference, inertias[eid],
            state['material_state']['response'].local_rotations, order=element.core.order)
        kinetic_row = np.zeros((len(local_kinetic), len(geometric))); kinetic_row[:, slots] = local_kinetic
        kinetic_rows.append(kinetic_row)
        guard()
    factor = np.vstack(rows)
    error = np.linalg.norm(factor.T@factor+geometric-packet.stiffness)
    if not np.isfinite(error) or error > 1e-11*max(1., np.linalg.norm(packet.stiffness)):
        raise ValueError('signed native Hessian reconstruction failed')
    checkpoint('kernel'); guard()
    kinetic = np.vstack(kinetic_rows)
    if np.linalg.norm(kinetic.T@kinetic-packet.mass) > 1e-11*max(1., np.linalg.norm(packet.mass)):
        raise ValueError('current kinetic factor reconstruction failed')
    result = solve_signed_kinetic_factor_modes(factor, geometric, kinetic, packet.free_dofs,
        packet.algebraic_dofs, bounds=bounds, num_modes=num_modes, root_width=root_width,
        cancellation_token=cancellation_token)
    checkpoint('output'); guard()
    work = sha(dict(policy=LOAD_POLICY, parameter=packet.load_parameter, nodal_forces=force,
        element_line_patterns=[(i, e.core.line_force) for i, e in sorted(model.mesh.elements.items())]))
    return packet, SignedLoadedModes(result.eigenvalues, result.full_modes, result.numerical_brackets,
        result.spectral_residual, result.full_backward_residual, packet.identity, work, force)
