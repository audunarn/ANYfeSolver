"""Centered-reference native successor to the frozen P5 coordinate package.

Private development candidate: no public selector or qualification authority.
Unchanged physical operators are imported from the preserved P5 package.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver._native_reference_modal import _owned
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.control import cancellation_safe_point
from anyfesolver.nonlinear_state import create_model_native_rotation_store
from anyfesolver._ge_beam3_p5_centered.reference_modal import prepare as prepare_reference
from anyfesolver._ge_beam3_p5_centered.core import sha
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam as CompensatedStationaryBeam
from .positions import validate_total_pair, rest_mass


POLICY = 'P5_CENTERED_ACCEPTED_HESSIAN_CURRENT_REST_MASS_V3'


@dataclass(frozen=True)
class CommittedOperator:
    stiffness: np.ndarray
    mass: np.ndarray
    internal_force: np.ndarray
    nodal_condensed_tangent: np.ndarray
    state_identity: str
    elastic_interior: bool


@dataclass(frozen=True)
class CommittedPencil:
    stiffness: np.ndarray
    mass: np.ndarray
    internal_force: np.ndarray
    free_dofs: tuple
    algebraic_dofs: tuple
    internal_layout: tuple
    operators: tuple
    displacement: np.ndarray
    identity: str
    policy: str = POLICY
    production_qualified: bool = False


@dataclass(frozen=True)
class CommittedModes:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    dynamic_map: np.ndarray
    normalized_residual: float
    operator_identity: str
    spatial_dead_forces: np.ndarray
    external_work_identity: str
    mass_policy: str = 'CURRENT_LIFTED_KINETIC_FIELD_LINEARIZED_AT_REST'
    material_policy: str = 'ACCEPTED_ELASTIC_INTERIOR_NO_HISTORY_ADVANCE'
    production_qualified: bool = False


def _close(actual, expected, label):
    if np.linalg.norm(actual-expected) > 1e-11*max(1., np.linalg.norm(expected)):
        raise ValueError(label+' mismatch')


def _operator(element, inner, inertia):
    core = element.core; ref = core.reference; response = inner['response']
    high, low = validate_total_pair(ref.coordinates, inner['committed_total_u'],
        inner['committed_positions'], inner['committed_position_low'], inner['coordinate_policy'])
    mixed = CompensatedStationaryBeam(ref, core.section, order=core.order,
        position_low=low, origins=inner['origins'])
    evaluation = mixed.evaluate(high,
        inner['committed_nodal_rotation_matrices']@ref.nodal_triads,
        response.local_rotations, response.moments)
    if np.linalg.norm(evaluation.residual[18:], np.inf) > 1e-11:
        raise ValueError('accepted internal stationary equilibrium failed')
    h = evaluation.hessian
    np.linalg.cholesky(-h[24:, 24:])
    # Endpoint moments only. Cell rotations remain in the dynamic operator.
    k = h[:24, :24]-h[:24, 24:]@np.linalg.solve(h[24:, 24:], h[24:, :24])
    condensed = k[:18, :18]-k[:18, 18:]@np.linalg.solve(k[18:, 18:], k[18:, :18])
    _close(condensed, response.tangent, 'accepted nodal Schur tangent')
    _close(evaluation.residual[:18], response.residual, 'accepted nodal force')
    if sha(evaluation.stations) != sha(response.stations):
        raise ValueError('accepted station replay changed at retained coordinates')
    mass = rest_mass(ref, inertia, core.order, high, low, response.local_rotations)
    elastic = True
    for station in evaluation.stations:
        state = station.response
        drive = abs(float(core.section._direction@state.resultants))
        threshold = core.section._yield+core.section._hardening*state.history.accumulated
        margin = threshold-drive
        # An explicit roundoff guard around a nonsmooth constitutive boundary,
        # not a changed yield law or softened mechanics tolerance.
        elastic &= (not state.plastic_active and
                    margin > 64*np.finfo(float).eps*max(1., drive, threshold))
    return CommittedOperator(_owned(k), _owned(mass), _owned(evaluation.residual[:24]),
        _owned(condensed), inner['state_sha256'], bool(elastic))


def prepare(model, element_states, displacement, section_inertias, *, cancellation_token=None):
    if type(section_inertias) is not dict:
        raise ValueError('explicit section inertia map required')
    inertias = {i: _owned(value) for i, value in section_inertias.items()}
    reference_blocks, reference_guard = prepare_reference(model, inertias,
                                                          cancellation_token=cancellation_token)
    nodal = model.mesh.dof_manager.total_dofs
    count = len(model.mesh.elements)
    if not nodal+6*count <= 256:
        raise ValueError('bounded committed native model required')
    if type(element_states) is not dict or set(element_states) != set(model.mesh.elements):
        raise ValueError('complete accepted native state map required')
    total_u = _owned(displacement)
    if total_u.shape != (nodal,): raise ValueError('complete accepted displacement required')
    owned_states = deepcopy(element_states)
    # This detached store validates shared nodal rotation authority. It starts
    # no trial and never attaches to or commits a caller-owned material store.
    create_model_native_rotation_store(model, owned_states, total_u)
    stiffness = np.zeros((nodal+6*count,)*2); mass = np.zeros_like(stiffness)
    force = np.zeros(len(stiffness)); algebraic = set(); internal_layout = []; operators = []
    for index, (eid, element) in enumerate(sorted(model.mesh.elements.items())):
        reference_guard('before_committed_operator', reference_blocks)
        dofs = tuple(int(i) for i in element.get_dof_mapping(model.mesh))
        validated = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section,
            owned_states[eid], 1, expected_committed_total_u=total_u[list(dofs)])
        operator = _operator(element, validated['material_state'], inertias[eid])
        local = tuple(range(nodal+6*index, nodal+6*index+6)); slots = dofs+local
        stiffness[np.ix_(slots, slots)] += operator.stiffness
        mass[np.ix_(slots, slots)] += operator.mass
        force[list(slots)] += operator.internal_force
        algebraic.update(dofs[i] for i in (3,4,5,9,10,11,15,16,17))
        internal_layout.append((eid, local)); operators.append((eid, operator))
        reference_guard('after_committed_operator', reference_blocks)
    _, _, transform, origin, independent, info = build_constraint_transformation(
        sparse.csr_matrix(stiffness[:nodal, :nodal]), np.zeros(nodal), model)
    if (info['num_mpc_constraints'] or not np.array_equal(transform.toarray(), np.eye(nodal)[:, independent])
            or np.any(origin != 0.) or np.any(total_u[info['fixed_dofs']] != 0.)):
        raise ValueError('homogeneous supports and compatible accepted displacement required')
    free = tuple(int(i) for i in independent)+tuple(range(nodal, len(stiffness)))
    body = dict(stiffness=_owned(stiffness), mass=_owned(mass), internal_force=_owned(force),
        free_dofs=free, algebraic_dofs=tuple(i for i in free if i in algebraic),
        internal_layout=tuple(internal_layout), operators=tuple(operators), displacement=total_u)
    identity = sha(dict(policy=POLICY, reference=tuple(b.identity for b in reference_blocks), **body))
    packet = CommittedPencil(**body, identity=identity)
    frozen = sha(packet)
    def guard():
        cancellation_safe_point(cancellation_token, 'committed_modal.guard')
        reference_guard('committed_modal', reference_blocks)
        if sha(packet) != frozen: raise ValueError('committed operator packet changed')
    guard()
    return packet, guard


def solve_elastic_modes(model, element_states, displacement, section_inertias,
                        spatial_dead_forces, *, num_modes=6, cancellation_token=None):
    """Modes about an equilibrated elastic-interior state at rest.

    Explicit nodal spatial dead forces only: no rotational moments, distributed
    loads, followers or hidden external-load tangents. The supplied force is
    verified against free assembled equilibrium; constrained residuals are
    reactions. No station history is advanced or re-originated.
    """
    packet, guard = prepare(model, element_states, displacement, section_inertias,
                             cancellation_token=cancellation_token)
    if not all(op.elastic_interior for _, op in packet.operators):
        raise ValueError('plastic or yield-boundary vibration branch is not authorized')
    load = _owned(spatial_dead_forces)
    nodal = len(packet.displacement)
    if load.shape != (nodal,): raise ValueError('explicit complete spatial dead force required')
    rotations = {int(i) for e in model.mesh.elements.values()
                 for i in np.array(e.get_dof_mapping(model.mesh)).reshape(3,6)[:, 3:].flat}
    if np.any(load[sorted(rotations)] != 0.):
        raise ValueError('nodal moments require a separately authorized load tangent')
    external = np.r_[load, np.zeros(len(packet.internal_force)-nodal)]
    residual = packet.internal_force-external
    for dof in tuple(rotations)+tuple(range(nodal, len(residual))):
        residual[dof] /= max(e.core.length for e in model.mesh.elements.values())
    scale = max(1., np.linalg.norm(load))
    if np.linalg.norm(residual[list(packet.free_dofs)])/scale > 1e-11:
        raise ValueError('committed free equilibrium required for modal interpretation')
    guard()
    result = solve_stationary_spectrum(packet.stiffness, packet.mass, packet.free_dofs,
        packet.algebraic_dofs, num_modes=num_modes, cancellation_token=cancellation_token)
    guard()
    return packet, CommittedModes(result.eigenvalues, result.full_modes, result.dynamic_map,
        result.normalized_residual, packet.identity, load,
        sha(dict(policy='SPATIAL_DEAD_NODAL_TRANSLATION_WORK', forces=load)))
