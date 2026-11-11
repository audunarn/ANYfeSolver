"""Frozen-current-state, conservative native beam buckling predictions.

Solve F.T F v = lambda (-G) v in the complete supported kinematic space,
where F = compliance_factor @ compatibility and G is the stress Hessian minus
the conservative external-work Hessian. Only inertia-free stress
resultants have been eliminated. Cell rotations and nodal rotation traces are
NOT condensed at lambda=1, and mass is not a buckling operator.

These are local linearized multipliers of that complete captured operator. They
are not nonlinear continuation loads, a guarantee of the first bifurcation,
or proof of a stable postbuckling path. A nonsmooth active material state,
nonconservative load or singular supported material operator fails closed.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, solve_triangular

from .control import cancellation_safe_point
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import sha


POLICY = 'GE_BEAM3_FROZEN_CURRENT_MATERIAL_VS_NEGATIVE_EFFECTIVE_GEOMETRIC_HESSIAN_V1'
TOLERANCE = 1e-11


class NativeBucklingError(ValueError):
    """The current state cannot supply the declared buckling prediction."""


def _require(ok, message):
    if not ok:
        raise NativeBucklingError(message)


def _controls(bounds, num_modes):
    _require(type(bounds) is tuple and len(bounds) == 2
             and all(type(v) in (int, float) and np.isfinite(v) for v in bounds)
             and 0 <= bounds[0] < bounds[1], 'explicit finite positive multiplier window required')
    _require(type(num_modes) is int and 1 <= num_modes <= 512,
             'bounded positive buckling mode count required')


@dataclass(frozen=True)
class BucklingSpectrum:
    status: str
    multipliers: np.ndarray
    modes: np.ndarray
    free_dofs: tuple
    requested_modes: int
    positive_multipliers_in_window: int
    bounds: tuple
    original_factor_residuals: tuple
    material_orthogonality_error: float
    whitening_error: float
    original_symmetry_error: float
    projected_symmetry_error: float
    input_sha256: str
    policy: str = field(default=POLICY, init=False)
    nonlinear_critical_load_authorized: bool = field(default=False, init=False)
    production_qualified: bool = field(default=False, init=False)


def factor_buckling(left, right, geometric, free_dofs, *, bounds, num_modes=6,
                    check=lambda: None, cancellation_token=None):
    """Material-energy QR whitening; never form an ill-scaled F.T F pencil.

The normalization and final residuals are checked against the original two
material factors, not only against a rounded material stiffness matrix.
All modes are expressed in the original nodal-plus-cell coordinates.
"""
    def guard():
        cancellation_safe_point(cancellation_token, 'native-buckling.factor')
        check()
    guard()
    _controls(bounds, num_modes)
    _require(all(type(a) is np.ndarray and a.ndim == 2 and a.dtype == np.dtype(float)
                 and np.isfinite(a).all() for a in (left, right, geometric)),
             'finite binary64 native factors required')
    size = right.shape[1]
    _require(1 <= size <= 512 and 1 <= left.shape[0] <= 512
             and left.shape[1] == right.shape[0]
             and geometric.shape == (size, size), 'bounded compatible native factor shapes required')
    _require(type(free_dofs) is tuple and free_dofs
             and all(type(i) is int and 0 <= i < size for i in free_dofs)
             and tuple(sorted(set(free_dofs))) == free_dofs,
             'ordered unique free kinematic coordinates required')
    original_symmetry = float(np.linalg.norm(geometric-geometric.T)/max(1., float(np.linalg.norm(geometric))))
    _require(np.isfinite(original_symmetry) and original_symmetry <= TOLERANCE,
             'symmetric conservative stress Hessian required')
    identity = sha(dict(left=left, right=right, geometric=geometric, free=free_dofs,
                        bounds=bounds, num_modes=num_modes, policy=POLICY))
    columns = list(free_dofs)
    r = right[:, columns]
    g = geometric[np.ix_(columns, columns)]
    # Column scaling removes coordinate-unit differences before QR. It does
    # not drop any column or regularize a mechanism.
    material = left @ r
    scales = np.max(np.abs(material), axis=0)
    _require(material.shape[0] >= len(columns) and np.isfinite(material).all()
             and np.all(scales > 0), 'supported material operator is singular')
    _, triangular = np.linalg.qr(material / scales, mode='reduced')
    _require(np.all(np.diag(triangular) != 0), 'supported material operator is singular')
    try:
        lift = solve_triangular(triangular, np.eye(len(columns)), lower=False) / scales[:, None]
    except (ValueError, np.linalg.LinAlgError) as error:
        raise NativeBucklingError('supported material whitening failed') from error
    guard()
    original_whitened = left @ (r @ lift)
    whitening_error = float(np.linalg.norm(original_whitened.T @ original_whitened - np.eye(len(columns)), ord=2))
    _require(np.isfinite(lift).all() and np.isfinite(whitening_error)
             and whitening_error <= TOLERANCE, 'original material-energy whitening witness failed')
    projected = -(lift.T @ g @ lift)
    _require(np.isfinite(projected).all(), 'nonfinite projected stress Hessian')
    scale = max(1., float(np.linalg.norm(projected, ord=2)))
    symmetry_error = float(np.linalg.norm(projected-projected.T, ord=2)/scale)
    _require(np.isfinite(projected).all() and symmetry_error <= TOLERANCE,
             'projected stress Hessian symmetry witness failed')
    # Average the two rounded evaluations of the conservative congruence.
    # Record both input and projected defects, and check every returned mode
    # against the ORIGINAL unsymmetrized stress Hessian below.
    values, vectors = eigh(.5*projected + .5*projected.T, check_finite=True)
    guard()
    selected = []
    for index in range(len(values)-1, -1, -1):
        if values[index] <= 0:
            continue
        multiplier = 1./float(values[index])
        if np.isfinite(multiplier) and bounds[0] < multiplier <= bounds[1]:
            selected.append((multiplier, index))
    selected.sort(key=lambda item: item[0])
    count = len(selected)
    chosen = selected[:num_modes]
    modes = np.zeros((size, len(chosen)))
    residuals = []
    for column, (multiplier, index) in enumerate(chosen):
        guard()
        mode = lift @ vectors[:, index]
        energy = left @ (r @ mode)
        norm = float(np.linalg.norm(energy))
        _require(np.isfinite(norm) and norm > 0, 'nonpositive material mode energy')
        mode /= norm
        if mode[int(np.argmax(np.abs(mode)))] < 0:
            mode = -mode
        material_force = r.T @ (left.T @ (left @ (r @ mode)))
        stress_force = multiplier * (g @ mode)
        denominator = float(np.linalg.norm(material_force) + np.linalg.norm(stress_force))
        residual = float(np.linalg.norm(material_force+stress_force)/denominator) if denominator > 0 else np.inf
        _require(np.isfinite(residual) and residual <= TOLERANCE,
                 'original supported buckling residual witness failed')
        modes[columns, column] = mode
        residuals.append(residual)
    energy_modes = left @ (right @ modes)
    orthogonality = float(np.linalg.norm(energy_modes.T @ energy_modes - np.eye(len(chosen))))
    _require(np.isfinite(orthogonality) and orthogonality <= TOLERANCE,
             'original material mode orthogonality witness failed')
    guard()
    _require(sha(dict(left=left, right=right, geometric=geometric, free=free_dofs,
                      bounds=bounds, num_modes=num_modes, policy=POLICY)) == identity,
             'native buckling factors changed during solution')
    return BucklingSpectrum('COMPLETED' if chosen else 'NO_MULTIPLIERS_IN_REQUESTED_WINDOW',
        _owned(np.array([v for v, _ in chosen])), _owned(modes), free_dofs, num_modes,
        count, bounds, tuple(residuals), orthogonality, whitening_error, original_symmetry, symmetry_error, identity)


@dataclass(frozen=True)
class NativeBucklingRun:
    definition_graph_sha256: str
    checkpoint_sha256: str
    section_owner: str
    pencil: object
    spectrum: BucklingSpectrum
    policy: str = field(default=POLICY, init=False)
    production_qualified: bool = field(default=False, init=False)


def solve(analysis, checkpoint, *, expected_sha256, bounds, num_modes=6, cancellation_token=None):
    """Capture through the real force checkpoint owner, without state advance."""
    cancellation_safe_point(cancellation_token, 'native-buckling.capture')
    _controls(bounds, num_modes)
    with analysis._operation():
        raw, digest = analysis._backend(checkpoint, expected_sha256)
        if analysis._family == 'GENERALIZED_DISTRIBUTED':
            from ._ge_beam3_native_generalized_restart import decode_checkpoint
            from ._ge_beam3_native_generalized_factor_modal import prepare
            state = decode_checkpoint(analysis.model, raw, expected_sha256=digest)[-1]
            # This native owner already includes ALL distributed load work.
            external = np.zeros(analysis.model.mesh.dof_manager.total_dofs)
        elif analysis._family == 'PHYSICAL_FIBRE_NODAL':
            from ._ge_beam3_native_fibre_restart import decode_checkpoint, _forces
            from ._ge_beam3_native_fibre_current_modal import prepare
            forces, chain = decode_checkpoint(analysis.model, raw, expected_sha256=digest)
            state = chain[-1]
            external = state['load_factor'] * _forces(forces, tuple(sorted(analysis.model.mesh.nodes)),
                analysis.model.mesh.dof_manager.total_dofs)
        else:
            raise NativeBucklingError('unsupported native buckling state owner')
        cancellation_safe_point(cancellation_token, 'native-buckling.captured')
        packet, guard = prepare(analysis.model, state['states'], state['displacements'],
            analysis._inertias, external, cancellation_token=cancellation_token)
        spectrum = factor_buckling(packet.left, packet.right, packet.geometric,
            packet.base.free_dofs, bounds=bounds, num_modes=num_modes,
            check=guard, cancellation_token=cancellation_token)
        guard()
        return NativeBucklingRun(analysis.identity, expected_sha256, analysis._family, packet, spectrum)
