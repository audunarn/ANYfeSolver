"""Private loaded-state conservative pencil for the native V4 beam.

Retain cell inertia; eliminate endpoint moments locally and exactly massless
nodal rotation traces globally. No history advance, load omission or mass floor.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.control import cancellation_safe_point
from anyfesolver.nonlinear_state import create_model_native_rotation_store
from anyfesolver._native_reference_modal import _owned
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_p5.algebra import validate_section
from anyfesolver._ge_beam3_p5_centered.positions import validate_total_pair, rest_mass
from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement
from anyfesolver._ge_beam3_p5_loads.core import sha
from anyfesolver._ge_beam3_p5_loads.state import parameter_value, LOAD_POLICY


POLICY = 'GE_BEAM3_V4_LOADED_HESSIAN_CURRENT_REST_MASS_V1'


@dataclass(frozen=True)
class LoadedOperator:
    stiffness: np.ndarray
    mass: np.ndarray
    net_residual: np.ndarray
    nodal_condensed_tangent: np.ndarray
    state_identity: str
    load_parameter: float
    elastic_interior: bool


@dataclass(frozen=True)
class LoadedPencil:
    stiffness: np.ndarray
    mass: np.ndarray
    net_residual: np.ndarray
    free_dofs: tuple
    algebraic_dofs: tuple
    internal_layout: tuple
    operators: tuple
    displacement: np.ndarray
    load_parameter: float
    identity: str
    policy: str = POLICY
    production_qualified: bool = False


@dataclass(frozen=True)
class LoadedModes:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    dynamic_map: np.ndarray
    normalized_residual: float
    operator_identity: str
    nodal_spatial_dead_forces: np.ndarray
    external_work_identity: str
    mass_policy: str = 'CURRENT_LIFTED_KINETIC_FIELD_LINEARIZED_AT_REST'
    material_policy: str = 'ACCEPTED_ELASTIC_INTERIOR_NO_HISTORY_ADVANCE'
    negative_eigenvalues_retained: bool = True
    buckling_factor_authorized: bool = False
    production_qualified: bool = False


def _close(actual, expected, label):
    if np.linalg.norm(actual-expected) > 1e-11*max(1., np.linalg.norm(expected)):
        raise ValueError(label+' mismatch')


def _operator(element, inner, inertia):
    core = element.core; ref = core.reference; response = inner['response']
    p = parameter_value(inner['load_parameter'])
    high, low = validate_total_pair(ref.coordinates, inner['committed_total_u'],
        inner['committed_positions'], inner['committed_position_low'], inner['coordinate_policy'])
    mixed = CenteredStationaryBeam(ref, core.section, order=core.order, position_low=low,
        origins=inner['origins'], line_force=p*core.line_force)
    evaluation = mixed.evaluate(high, inner['committed_nodal_rotation_matrices']@ref.nodal_triads,
        response.local_rotations, response.moments)
    if np.linalg.norm(evaluation.residual[18:], np.inf) > 1e-11:
        raise ValueError('accepted loaded internal equilibrium failed')
    h = evaluation.hessian
    np.linalg.cholesky(-h[24:, 24:])
    # Moment multipliers have no inertia. Cell rotations DO have inertia and
    # are retained; the second Schur operation below is a validation only.
    k = h[:24, :24]-h[:24, 24:]@np.linalg.solve(h[24:, 24:], h[24:, :24])
    condensed = k[:18, :18]-k[:18, 18:]@np.linalg.solve(k[18:, 18:], k[18:, :18])
    _close(condensed, response.tangent, 'accepted loaded nodal Schur tangent')
    _close(evaluation.residual[:18], response.residual, 'accepted loaded net force')
    if sha(evaluation.stations) != sha(response.stations):
        raise ValueError('accepted loaded stations changed on replay')
    mass = rest_mass(ref, inertia, core.order, high, low, response.local_rotations)
    elastic = True
    for station in evaluation.stations:
        state = station.response
        drive = abs(float(core.section._direction@state.resultants))
        threshold = core.section._yield+core.section._hardening*state.history.accumulated
        # Guard a nonsmooth branch boundary, not a modified yield condition.
        elastic &= (not state.plastic_active and
            threshold-drive > 64*np.finfo(float).eps*max(1., drive, threshold))
    return LoadedOperator(_owned(k), _owned(mass), _owned(evaluation.residual[:24]),
        _owned(condensed), inner['state_sha256'], p, bool(elastic))


def prepare(model, element_states, displacement, section_inertias, *, load_parameter,
        cancellation_token=None):
    """Detached accepted-state snapshot; no caller material-store transaction."""
    p = parameter_value(load_parameter)
    elements = tuple(sorted(model.mesh.elements.items())); nodal = model.mesh.dof_manager.total_dofs
    if not elements or not 1 <= nodal+6*len(elements) <= 256:
        raise ValueError('bounded loaded native model required before mechanics')
    if any(type(e) is not NativeP5BeamElement for _, e in elements):
        raise ValueError('exact load-aware native elements required')
    ids = {i for i, _ in elements}
    for values in (element_states, section_inertias):
        if type(values) is not dict or set(values) != ids or any(type(i) is not int for i in values):
            raise ValueError('complete exact state and inertia maps required')
    inertias = {i: _owned(validate_section(section_inertias[i])) for i, _ in elements}
    total = _owned(displacement)
    if total.shape != (nodal,): raise ValueError('complete accepted displacement required')

    def snapshot():
        cancellation_safe_point(cancellation_token, 'loaded_modal.capture')
        if (tuple(sorted(model.mesh.elements.items())) != elements or model.mesh.element_activity is not None
                or model.constraint_equations or model.mesh.point_masses
                or model.mesh.dof_manager.total_dofs != nodal):
            raise ValueError('loaded model ownership/activity/constraints changed or unsupported')
        for _, element in elements:
            element._check(model.mesh)
            if model.materials.get(element.material_name) is not element.core.section:
                raise ValueError('loaded section ownership changed')
        return sha(dict(elements=[(i, e.to_dict(), list(e.get_dof_mapping(model.mesh))) for i, e in elements],
            nodes=[(i, node.coords()) for i, node in sorted(model.mesh.nodes.items())],
            inertias=inertias, boundaries=[vars(bc) for bc in model.boundary_conditions],
            dofs=nodal, constrained_dofs=sorted(model.mesh.dof_manager._constrained_dofs), parameter=p))

    # Build the support cache before its identity is captured.
    _, _, transform, offset, independent, info = build_constraint_transformation(
        sparse.eye(nodal, format='csr'), np.zeros(nodal), model)
    fixed = set(range(nodal))-set(independent)
    if (info['num_mpc_constraints'] or np.any(offset) or np.any(total[list(fixed)])
            or not np.array_equal(transform.toarray(), np.eye(nodal)[:, independent])):
        raise ValueError('homogeneous support selector and compatible displacement required')
    node_maps = [tuple(model.mesh.dof_manager.get_node_dofs(i)) for i in sorted(model.mesh.nodes)]
    if any(len(row) != 6 for row in node_maps) or sorted(d for row in node_maps for d in row) != list(range(nodal)):
        raise ValueError('complete disjoint six-DOF node maps required')
    for row in node_maps:
        if len(set(row[3:]) & fixed) not in (0, 3):
            raise ValueError('partial rotational supports require a separate contract')
    identity = snapshot()

    def model_guard():
        if snapshot() != identity: raise ValueError('loaded modal frozen inputs changed')

    owned_states = deepcopy(element_states); validated = {}
    for eid, element in elements:
        model_guard()
        dofs = list(element.get_dof_mapping(model.mesh))
        state = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section,
            owned_states[eid], 1, expected_committed_total_u=total[dofs])
        if state['material_state']['load_parameter'] != p:
            raise ValueError('accepted state/load parameter mismatch')
        validated[eid] = state
    # Validate shared authoritative rotations, without opening/committing a trial.
    create_model_native_rotation_store(model, validated, total)
    stiffness = np.zeros((nodal+6*len(elements),)*2); mass = np.zeros_like(stiffness)
    force = np.zeros(len(stiffness)); algebraic = set(); layout = []; operators = []
    for index, (eid, element) in enumerate(elements):
        model_guard()
        operator = _operator(element, validated[eid]['material_state'], inertias[eid])
        dofs = tuple(int(i) for i in element.get_dof_mapping(model.mesh))
        internal = tuple(range(nodal+6*index, nodal+6*index+6)); slots = dofs+internal
        stiffness[np.ix_(slots, slots)] += operator.stiffness
        mass[np.ix_(slots, slots)] += operator.mass
        force[list(slots)] += operator.net_residual
        algebraic.update(dofs[i] for i in (3, 4, 5, 9, 10, 11, 15, 16, 17))
        layout.append((eid, internal)); operators.append((eid, operator))
    free = tuple(int(i) for i in independent)+tuple(range(nodal, len(stiffness)))
    body = dict(stiffness=_owned(stiffness), mass=_owned(mass), net_residual=_owned(force), free_dofs=free,
        algebraic_dofs=tuple(i for i in free if i in algebraic), internal_layout=tuple(layout),
        operators=tuple(operators), displacement=total, load_parameter=p)
    packet = LoadedPencil(**body, identity=sha(dict(policy=POLICY, model=identity, **body)))
    frozen = sha(packet)
    def guard():
        model_guard()
        if sha(packet) != frozen: raise ValueError('loaded modal operator packet changed')
    guard()
    return packet, guard


def solve_elastic_modes(model, element_states, displacement, section_inertias,
        nodal_spatial_dead_forces, *, load_parameter, num_modes=6, cancellation_token=None):
    """Signed spectrum at a loaded equilibrium on an elastic-interior branch.

    Nodal forces are actual forces at this state, NOT the unit load pattern.
    The bound distributed load is already in the net potential derivatives.
    No nodal moments, followers, plastic branch selection or critical factor.
    """
    packet, guard = prepare(model, element_states, displacement, section_inertias,
        load_parameter=load_parameter, cancellation_token=cancellation_token)
    if not all(op.elastic_interior for _, op in packet.operators):
        raise ValueError('plastic or yield-boundary vibration branch is not authorized')
    force = _owned(nodal_spatial_dead_forces); nodal = len(packet.displacement)
    if force.shape != (nodal,): raise ValueError('complete actual nodal spatial dead force required')
    rotations = tuple(int(d) for i in sorted(model.mesh.nodes) for d in model.mesh.dof_manager.get_node_dofs(i)[3:])
    if np.any(force[list(rotations)]): raise ValueError('nodal moments need a separately authorized load tangent')
    residual = packet.net_residual-np.r_[force, np.zeros(len(packet.net_residual)-nodal)]
    for dof in rotations+tuple(range(nodal, len(residual))):
        residual[dof] /= max(e.core.length for e in model.mesh.elements.values())
    if np.linalg.norm(residual[list(packet.free_dofs)]) > 1e-11*max(1., np.linalg.norm(force)):
        raise ValueError('loaded free equilibrium required for modal interpretation')
    guard()
    result = solve_stationary_spectrum(packet.stiffness, packet.mass, packet.free_dofs,
        packet.algebraic_dofs, num_modes=num_modes, cancellation_token=cancellation_token)
    guard()
    work = sha(dict(policy=LOAD_POLICY, parameter=packet.load_parameter, nodal_forces=force,
        element_line_patterns=[(i, e.core.line_force) for i, e in sorted(model.mesh.elements.items())]))
    return packet, LoadedModes(result.eigenvalues, result.full_modes, result.dynamic_map,
        result.normalized_residual, packet.identity, force, work)
