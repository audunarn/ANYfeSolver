"""Private conservative coupled current-rest factor pencil and signed modes.

No static elimination of physical beam cell rotations, fictitious joint mass,
mass-eigenvalue cutoff, active-yield spectra, or public/default routing.
"""
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from hashlib import sha256
import numpy as np
from scipy.linalg import block_diag

from ._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis, decode
from ._ge_beam3_variational_shell import response as shell_response
from ._ge_beam3_pose_joint import _exp_terms
from ._ge_beam3_p5.algebra import skew
from ._ge_beam3_p5_seeded.core import canonical
from ._ge_beam3_retained_generalized_modal import _elastic_interior
from ._ge_beam3_native_generalized_modal import _inertia, _relative
from ._ge_beam3_native_generalized_factor_modal import compliance_factor, kinetic_factor
from ._ge_beam3_p5_centered.mass import current_rest_mass
from ._native_reference_modal import _owned
from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes
from .control import cancellation_safe_point

POLICY = 'GE_BEAM3_VARIATIONAL_COUPLED_CURRENT_REST_FACTOR_PENCIL_V1'


@dataclass(frozen=True)
class CoupledPencil:
    left: np.ndarray
    right: np.ndarray
    geometric: np.ndarray
    kinetic: np.ndarray
    lift: np.ndarray
    full_mass: np.ndarray | None
    full_stiffness: np.ndarray  # Rounded diagnostic; not factor-spectrum authority.
    constraints: np.ndarray
    full_kinematic_dofs: tuple
    algebraic_dofs: tuple
    shell_mass_rank: int
    internal_cell_coordinates: int
    definition_sha256: str
    state_sha256: str
    inertia_sha256: str
    witnesses: tuple
    mass_captured: bool
    policy: str = field(default=POLICY, init=False)
    production_qualified: bool = field(default=False, init=False)
    buckling_factor_authorized: bool = field(default=False, init=False)


@contextmanager
def _locked(owner):
    if type(owner) is not CoupledShellBeamAnalysis:
        raise ValueError('actual global coupled owner required')
    if not owner._lock.acquire(blocking=False):
        raise RuntimeError('concurrent coupled spectral use')
    held = False
    try:
        if not owner.assembly._lock.acquire(blocking=False):
            raise RuntimeError('concurrent coupled assembly use')
        held = True
        yield
    finally:
        if held: owner.assembly._lock.release()
        owner._lock.release()


def _shell_factor(material, positions):
    """Factor on the known six-rigid-motion complement, not eigenvalue clipping."""
    x = positions-positions.mean(axis=0)
    length = np.linalg.norm(x)
    if not length > 0.: raise ValueError('degenerate shell rigid complement')
    rigid = np.vstack([np.block([[np.eye(3), -skew(row/length)],
        [np.zeros((3, 3)), np.eye(3)/length]]) for row in x])
    q, _ = np.linalg.qr(rigid, mode='complete')
    if _relative(material@q[:, :6], np.zeros((len(material), 6))) > 1e-11*max(1., np.linalg.norm(material)):
        raise ValueError('shell material factor lost rigid nulls')
    z = q[:, 6:]
    root = np.linalg.cholesky(z.T@material@z).T
    factor = root@z.T
    error = _relative(factor.T@factor, material)
    if error > 1e-11: raise ValueError('original shell material factor witness')
    return factor, error


def _capture(owner, state, section_inertias, shell_density, token, *, with_mass=True):
    a = owner.assembly; beam = a.beam
    issued = owner._require(state); before = canonical(state.descriptor())
    if not a.variational_shell:
        raise ValueError('V1 rotate-only shell work is not conservative spectral authority')
    if with_mass and (type(shell_density) is not float or not np.isfinite(shell_density) or shell_density <= 0.):
        raise ValueError('explicit positive shell density required')
    if with_mass and (type(section_inertias) is not dict or set(section_inertias) != set(beam.model.mesh.elements)
            or any(type(eid) is not int for eid in section_inertias)):
        raise ValueError('complete explicit physical beam inertia map required')
    inertias = {eid: _inertia(section_inertias[eid]) for eid in sorted(section_inertias)} if with_mass else {}
    inertia_bytes = canonical(dict(beam=section_inertias, shell_density=shell_density))
    def check():
        cancellation_safe_point(token, 'coupled-modes.guard')
        if owner._require(state) != issued or canonical(state.descriptor()) != before:
            raise ValueError('coupled spectral state changed')
        if canonical(dict(beam=section_inertias, shell_density=shell_density)) != inertia_bytes:
            raise ValueError('coupled spectral inertia changed')
    check()
    if any(np.any(beam.program.pattern.density(eid)) for eid in beam.model.mesh.elements):
        raise ValueError('nonconservative beam couples forbidden in coupled spectra')
    trial, residual, _, _ = owner._evaluate(state, state.mechanical, state.shell_u,
        state.multipliers, state.cursor, token)
    if np.linalg.norm((residual/owner.scale)[list(owner.free)]) > 1e-11:
        raise ValueError('actual free conservative coupled equilibrium required')
    # Current-rest material replay, never reusing an active plastic tangent.
    parameter = 0. if state.cursor == 0 else owner.targets[state.cursor-1]
    _, _, _, responses, _ = beam.assemble(state.mechanical, parameter, state.beam_histories)
    for (_, element), response_, origin in zip(beam.elements, responses, state.beam_histories, strict=True):
        _elastic_interior(element.operator, response_, origin)
    sh = a.shell_count; bn = beam.nodal_count; ne = len(beam.elements)
    full = tuple(range(sh+bn))+tuple(sh+bn+24*i+j for i in range(ne) for j in range(6))
    size = len(full)
    if size > 256: raise ValueError('bounded coupled spectral extent exceeded')
    inverse_p = np.eye(a.count)
    for i, row in enumerate(state.shell_u.reshape(-1, 6)):
        inverse_p[6*i+3:6*i+6, 6*i+3:6*i+6] = np.linalg.solve(_exp_terms(row[3:])[1], np.eye(3))
    spatial = trial.tangent@inverse_p
    constraints = spatial[-6:, list(full)]
    _, _, candidate, shell_material, shell_geometric = shell_response(a.model, a.element,
        state.shell_u, decode(state.shell_history), a.layers, split=True)
    if canonical(candidate) != trial.shell_candidate:
        raise ValueError('split shell replay changed trial state')
    sf, shell_error = _shell_factor(shell_material, a.coordinates+state.shell_u.reshape(-1, 6)[:, :3])
    lefts = [np.eye(len(sf))]; sr = np.zeros((len(sf), size)); sr[:, :sh] = sf
    rights = [sr]; witnesses = [('shell-material', shell_error)]
    geometric = np.zeros((size, size)); geometric[:sh, :sh] = shell_geometric
    geometric[sh:, sh:] = spatial[np.ix_(full[sh:], full[sh:])]
    # The beam diagonal block above includes its own joint port Hessian;
    # add only the shell/cross joint blocks to avoid double counting.
    jj = np.zeros((a.count, a.count)); jj[np.ix_(a.slots, a.slots)] = trial.joint_tangent
    jj = (jj@inverse_p)[np.ix_(full, full)]
    geometric[:sh, :] += jj[:sh, :]; geometric[sh:, :sh] += jj[sh:, :sh]
    kinetics = []; mass = np.zeros((size, size))
    for i, (eid, element) in enumerate(beam.elements):
        check()
        stresses = tuple(range(sh+bn+24*i+6, sh+bn+24*i+24))
        left, error = compliance_factor(-spatial[np.ix_(stresses, stresses)], check)
        right = spatial[np.ix_(stresses, full)]
        lefts.append(left); rights.append(right); witnesses.append(('compliance-'+str(eid), error))
        local = tuple(sh+int(d) for d in element.get_dof_mapping(beam.model.mesh))+tuple(range(sh+bn+6*i, sh+bn+6*i+6))
        if with_mass:
            kinetic = kinetic_factor(element.operator.reference, inertias[eid], state.mechanical.cell_rotations[i], check)
            lifted = np.zeros((len(kinetic), size)); lifted[:, local] = kinetic; kinetics.append(lifted)
            nodes = beam.nodes[i]
            actual_mass = current_rest_mass(element.operator.reference, inertias[eid], 24,
                state.mechanical.positions[nodes], state.mechanical.position_low[nodes], state.mechanical.cell_rotations[i])
            if _relative(kinetic.T@kinetic, actual_mass) > 1e-11:
                raise ValueError('coupled beam physical inertia witness')
            mass[np.ix_(local, local)] += actual_mass
    dynamic_shell = tuple(range(sh)) if a.topology == 'Q4' else tuple(6*i+j for i in range(sh//6) for j in range(3))
    if with_mass:
        material = copy(a.model.get_material('joint-shell')); material.density = shell_density
        sm = a.element.compute_mass_matrix(a.model.mesh, material)
        sk = np.zeros((len(dynamic_shell), size))
        sk[:, list(dynamic_shell)] = np.linalg.cholesky(sm[np.ix_(dynamic_shell, dynamic_shell)]).T
        if _relative(sk[:, :sh].T@sk[:, :sh], sm) > 1e-11:
            raise ValueError('unchanged shell inertia witness')
        kinetics.append(sk); mass[:sh, :sh] = sm
    # Eliminate the six exact pose constraints, not physical beam cell inertia.
    free = tuple(i for i, dof in enumerate(full) if dof in owner.free)
    slave = tuple(sh+6*beam.index[a.beam_node]+j for j in range(6))
    if not set(slave) <= set(free):
        raise ValueError('spectral joint requires six free slave beam coordinates')
    keep = tuple(i for i in free if i not in slave)
    lift = np.zeros((size, len(keep))); lift[list(keep)] = np.eye(len(keep))
    master = tuple(6*a.shell_node+j for j in range(6))
    rotation = _exp_terms(state.shell_u.reshape(-1, 6)[a.shell_node, 3:])[0]@a.frame
    offset = rotation@a.joint._offset
    lift[list(slave[:3])] = lift[list(master[:3])]-skew(offset)@lift[list(master[3:])]
    lift[list(slave[3:])] = lift[list(master[3:])]
    algebraic = [i for i, dof in enumerate(keep) if sh <= dof < sh+bn and (dof-sh)%6 >= 3]
    if a.topology == 'S3-V2D':
        algebraic += [i for i, dof in enumerate(keep) if dof < sh and dof%6 >= 3 and dof not in master[3:]]
        attached = [keep.index(dof) for dof in master[3:] if dof in keep]
        if len(attached) not in (0, 3):
            raise ValueError('partial attachment shell rotation supports need a separate kinetic basis')
        if attached:
            scale = np.max(np.abs(offset))
            if scale == 0.: algebraic.extend(attached)
            else:
                axis = offset/scale; axis /= np.linalg.norm(axis)
                seed = np.eye(3)[int(np.argmin(np.abs(axis)))]; second = np.cross(axis, seed); second /= np.linalg.norm(second)
                basis = np.column_stack((axis, second, np.cross(axis, second)))
                lift[:, attached] = lift[:, attached]@basis
                # Exact identity skew(offset) offset=0, not inertia clipping.
                lift[list(slave[:3]), attached[0]] = 0.
                algebraic.append(attached[0])
    constraint_error = _relative(constraints@lift, np.zeros((6, len(keep))))
    if constraint_error > 1e-11: raise ValueError('exact joint tangent lift witness')
    algebraic = tuple(sorted(algebraic))
    left = block_diag(*lefts); right = np.vstack(rights)
    kinetic = np.vstack(kinetics) if with_mass else np.zeros((0, size))
    mapped_kinetic = kinetic@lift
    if np.any(mapped_kinetic[:, algebraic] != 0.):
        raise ValueError('declared geometric massless coordinates have nonzero inertia')
    g = lift.T@geometric@lift
    if _relative(g, g.T) > 1e-11: raise ValueError('coupled conservative geometric symmetry witness')
    expanded = left@right
    stiffness = expanded.T@expanded+geometric
    raw_kk = spatial[np.ix_(full, full)].copy()
    for i, _ in enumerate(beam.elements):
        stresses = tuple(range(sh+bn+24*i+6, sh+bn+24*i+24))
        raw_kk -= spatial[np.ix_(full, stresses)]@np.linalg.solve(
            spatial[np.ix_(stresses, stresses)], spatial[np.ix_(stresses, full)])
    assembly_error = _relative(lift.T@stiffness@lift, lift.T@raw_kk@lift)
    if assembly_error > 1e-11: raise ValueError('original coupled saddle Schur witness')
    witnesses += [('joint-lift', constraint_error), ('original-saddle', assembly_error)]
    check()
    return CoupledPencil(_owned(left), _owned(right@lift), _owned(g), _owned(mapped_kinetic),
        _owned(lift), _owned(mass) if with_mass else None, _owned(stiffness), _owned(constraints),
        full, algebraic if with_mass else (), len(dynamic_shell) if with_mass else 0, 6*ne,
        owner.identity, sha256(before).hexdigest(), sha256(inertia_bytes).hexdigest(), tuple(witnesses), with_mass)


def capture(owner, state, *, section_inertias, shell_density, cancellation_token=None):
    with _locked(owner):
        return _capture(owner, state, section_inertias, shell_density, cancellation_token)


def modes(owner, state, *, section_inertias, shell_density, bounds, num_modes=6,
          root_width=1e-10, relative_width=1e-12, cancellation_token=None):
    with _locked(owner):
        packet = _capture(owner, state, section_inertias, shell_density, cancellation_token)
        result = solve_paired_factor_chain_modes(packet.left, packet.right, packet.geometric,
            packet.kinetic, tuple(range(packet.lift.shape[1])), packet.algebraic_dofs,
            bounds=bounds, num_modes=num_modes, root_width=root_width,
            relative_width=relative_width, cancellation_token=cancellation_token)
        owner._require(state)
        if sha256(canonical(dict(beam=section_inertias, shell_density=shell_density))).hexdigest() != packet.inertia_sha256:
            raise ValueError('coupled spectral inertia changed during solution')
        return packet, result


def buckling(owner, state, *, bounds, num_modes=6, cancellation_token=None):
    """Frozen-current material/stress multipliers; no inertia or static cell lift.

These are conservative linearized predictions, not a nonlinear critical load
or stable postbuckling certificate. Every reduced kinematic coordinate remains
in the multiplier problem, including the massless traces.
"""
    from ._ge_beam3_native_buckling import factor_buckling
    with _locked(owner):
        packet = _capture(owner, state, None, None, cancellation_token, with_mass=False)
        result = factor_buckling(packet.left, packet.right, packet.geometric,
            tuple(range(packet.lift.shape[1])), bounds=bounds, num_modes=num_modes,
            cancellation_token=cancellation_token, check=lambda: owner._require(state))
        owner._require(state)
        return packet, result
