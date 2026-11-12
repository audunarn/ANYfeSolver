# Private V2 successor: preserve the full kinetic factor through whitening.
"""Private signed factor-spectrum numerical kernel; not qualification.

Derived from the preserved research reconstruction, with owned inputs,
bounded cancellation and actual signed mode-shape extraction. No mechanics,
public routing, state transaction or independent review is provided here.
"""

from dataclasses import dataclass
from time import monotonic
from ._native_reference_modal import _owned
from .control import cancellation_safe_point

import numpy as np
from scipy import linalg
from ._relative_reference_svd import relative_reference_svd


def _reduce(factor, geometric, mass, free_dofs, algebraic_dofs, checkpoint, kinetic):
    from ._native_modal_capacity import limits
    coordinates,rows,_=limits()
    checkpoint('reduction.start')
    f, g, m = (np.array(x, dtype=float, copy=True) for x in (factor, geometric, mass))
    if (f.ndim != 2 or not 1 <= f.shape[1] <= coordinates or f.shape[0] > rows
            or g.shape != (f.shape[1],)*2 or m.shape != g.shape
            or not all(np.isfinite(x).all() for x in (f, g, m))):
        raise ValueError('bounded finite signed split required')
    for a in (g, m):
        if np.linalg.norm(a-a.T) > 1e-11*max(1., np.linalg.norm(a)):
            raise ValueError('symmetric split required')
    for slots in (free_dofs, algebraic_dofs):
        if (type(slots) is not tuple or len(set(slots)) != len(slots)
                or any(type(i) is not int or not 0 <= i < f.shape[1] for i in slots)):
            raise ValueError('exact distinct in-range DOFs required')
    if not set(algebraic_dofs) <= set(free_dofs):
        raise ValueError('algebraic DOFs must be free')
    if np.any(kinetic[:, algebraic_dofs] != 0.):
        raise ValueError('algebraic kinetic factor must be exactly zero')
    if np.any(m[:, algebraic_dofs] != 0.) or np.any(m[algebraic_dofs, :] != 0.):
        raise ValueError('algebraic inertia must be exactly zero')
    physical = tuple(i for i in free_dofs if i not in algebraic_dofs)
    if not physical: raise ValueError('physical coordinates required')
    mapping = np.zeros((f.shape[1], len(physical)))
    mapping[list(physical)] = np.eye(len(physical))
    strain = f[:, physical]
    if algebraic_dofs:
        count = len(algebraic_dofs)
        q, r = linalg.qr(f[:, algebraic_dofs], mode='full')
        r = r[:count]
        if r.shape != (count, count): raise ValueError('insufficient trace rows')
        singular = linalg.svdvals(r)
        if singular[-1] <= np.finfo(float).eps*count*singular[0]:
            raise ValueError('unresolved material trace rank')
        mapping[list(algebraic_dofs)] = -linalg.solve_triangular(r, q[:, :count].T@strain)
        # Retain the original physical strain rows. Material trace columns
        # vanish exactly in the strong axial/shear station rows. Rotating
        # their entire row space by a completed Q unnecessarily mixes strong
        # and weak rows. This evaluates the same trace-stationary map without
        # forming or subtracting normal equations.
        strain = f@mapping
        trace = r.T@r+g[np.ix_(algebraic_dofs, algebraic_dofs)]
        cholesky = linalg.cho_factor(trace)
        cross = g[list(algebraic_dofs)]@mapping
        correction = linalg.cho_solve(cholesky, cross)
        reduced_g = mapping.T@g@mapping-cross.T@correction
        mapping[list(algebraic_dofs)] -= correction
    else:
        reduced_g = mapping.T@g@mapping
    # Avoid squaring kinetic conditioning in a dense mass Cholesky.
    # Algebraic columns are exactly zero, so the signed trace correction
    # does not change the retained kinetic factor.
    _, mass_r = linalg.qr(kinetic[:, physical], mode='economic')
    singular_mass = linalg.svdvals(mass_r)
    if (mass_r.shape != (len(physical), len(physical))
            or singular_mass[-1] <= np.finfo(float).eps*len(physical)*singular_mass[0]):
        raise ValueError('unresolved physical kinetic rank')
    normalized = linalg.solve_triangular(mass_r.T, strain.T, lower=True).T
    left_g = linalg.solve_triangular(mass_r.T, reduced_g, lower=True)
    normalized_g = linalg.solve_triangular(mass_r.T, left_g.T, lower=True).T
    checkpoint('reduction.svd')
    _, singular, right = relative_reference_svd(normalized)
    with np.errstate(over='raise', invalid='raise', under='raise'):
        diagonal = singular**2
    signed = right@normalized_g@right.T
    vectors = mapping@linalg.solve_triangular(mass_r, right.T)
    return diagonal, signed, vectors


def _shifted_matrix(diagonal, correction, shift):
    """Congruently scale the shifted pencil without modifying its inertia."""
    d, g = np.asarray(diagonal), np.asarray(correction)
    if (d.ndim != 1 or not 1 <= len(d) <= 256 or g.shape != (len(d), len(d))
            or not np.isfinite(d).all() or not np.isfinite(g).all()
            or not np.isfinite(shift) or np.any(d < 0.)):
        raise ValueError('finite positive-factor diagonal and signed correction required')
    if np.linalg.norm(g-g.T) > 1e-11*max(1., np.linalg.norm(g)):
        raise ValueError('symmetric signed correction required')
    # Form the shift before summing the stress correction. No global stiffness
    # scale enters the sign threshold; congruence preserves inertia, not roots.
    delta = d-shift
    # Diagonal-based congruence matters: using the largest off-diagonal entry
    # in each row can bury a small Schur root even after a valid congruence.
    # Off-diagonal row scale is used only for an exactly absent diagonal.
    size = np.maximum(np.abs(delta), np.abs(np.diag(g)))
    size = np.where(size == 0., np.max(np.abs(g), axis=1), size)
    scale = np.sqrt(size)
    if np.any(scale == 0.): raise ValueError('unresolved shifted sign')
    scaled = (g/scale[:, None])/scale[None, :]
    scaled[np.diag_indices(len(d))] += (delta/scale)/scale
    return scaled, scale


def _inertia(diagonal, correction, shift):
    scaled, _ = _shifted_matrix(diagonal, correction, shift)
    values = linalg.eigvalsh(scaled)
    margin = 64*np.finfo(float).eps*len(diagonal)*max(1., np.max(np.abs(values)))
    if np.min(np.abs(values)) <= margin: raise ValueError('unresolved shifted sign')
    return int(np.count_nonzero(values < 0.))


def _bracket(diagonal, correction, bounds, count, width, checkpoint):
    """Bounded same-author numerical search; caller freezes the search bounds."""
    if (type(count) is not int or not 1 <= count <= len(diagonal)
            or len(bounds) != 2 or not np.isfinite(bounds).all()
            or not bounds[0] < bounds[1] or not np.isfinite(width) or width <= 0.):
        raise ValueError('finite ordered search bounds and positive width required')
    low_count = _inertia(diagonal, correction, bounds[0])
    high_count = _inertia(diagonal, correction, bounds[1])
    if low_count != 0 or high_count < count: raise ValueError('bounds do not enclose lowest requested roots')
    result = []
    for index in range(count):
        lo, hi = bounds
        for _ in range(96):
            checkpoint('bracket.shift')
            if hi-lo <= width: break
            middle = lo+(hi-lo)/2
            try:
                negative = _inertia(diagonal, correction, middle)
            except ValueError as exc:
                if str(exc) != 'unresolved shifted sign': raise
                left, right = max(lo, middle-width/4), min(hi, middle+width/4)
                a = _inertia(diagonal, correction, left)
                b = _inertia(diagonal, correction, right)
                if a <= index < b:
                    lo, hi = left, right
                    break
                raise ValueError('shifted signs unresolved at requested root width') from exc
            if negative <= index: lo = middle
            else: hi = middle
        else: raise ValueError('signed search iteration limit')
        if hi-lo > width: raise ValueError('signed search width not achieved')
        result.append((lo, hi))
    return np.array(result)

@dataclass(frozen=True)
class SignedFactorModes:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    numerical_brackets: np.ndarray
    spectral_residual: float
    full_backward_residual: float
    policy: str = 'PRIVATE_SIGNED_MATERIAL_KINETIC_FACTOR_MODES_V2'
    negative_eigenvalues_retained: bool = True
    certified_intervals: bool = False
    production_qualified: bool = False


def _orthonormalize(columns):
    # Householder Q reconstruction can lose an O(1e-18) leading component
    # through subtraction from one. Retain that component by direct scaling
    # and twice-applied modified Gram-Schmidt on the small requested cluster.
    result = []
    for column in columns.T:
        v = column.copy()
        maximum = np.max(np.abs(v))
        if not np.isfinite(maximum) or maximum == 0.: raise ValueError('empty signed mode column')
        v /= maximum
        for _ in range(2):
            for previous in result: v -= previous*(previous@v)
        size = np.linalg.norm(v)
        if not np.isfinite(size) or size <= 64*np.finfo(float).eps:
            raise ValueError('unresolved signed mode subspace rank')
        result.append(v/size)
    return np.column_stack(result)


def solve_signed_kinetic_factor_modes(factor, geometric, kinetic_factor, free_dofs, algebraic_dofs, *,
        bounds, num_modes=6, root_width=1e-10, cancellation_token=None):
    """Finite bounded kernel, no fallback. Caller binds the mechanical split.

    Full-coordinate residual is a backward check ONLY: it is not sufficient
    evidence of accuracy of low bending modes. The separate signed-basis
    residual uses the ACTION of the material diagonal and prestress, never
    the norm of the largest unrelated stiffness.
    """
    started = monotonic()
    def checkpoint(stage):
        cancellation_safe_point(cancellation_token, 'signed_factor_modes.'+stage)
        if monotonic()-started > 600.: raise ValueError('signed factor modes time limit')
    checkpoint('capture')
    # Bytes-owned copies survive caller mutation and are not writable views.
    # Dense mass below is for checks only, never used for whitening.
    f, g, kinetic = (_owned(x) for x in (factor, geometric, kinetic_factor))
    if (kinetic.ndim != 2 or not 1 <= kinetic.shape[0] <= 8192
            or not 1 <= kinetic.shape[1] <= 256 or f.ndim != 2
            or kinetic.shape[1] != f.shape[1]):
        raise ValueError('bounded matching material and kinetic factors required')
    m = kinetic.T@kinetic
    if (type(bounds) is not tuple or len(bounds) != 2
            or any(type(x) not in (int, float) or not np.isfinite(x) for x in bounds)
            or bounds[0] >= bounds[1] or type(num_modes) is not int or num_modes < 1
            or type(root_width) is not float or not np.isfinite(root_width) or root_width <= 0.):
        raise ValueError('immutable finite explicit root bounds, count and width required')
    d, correction, mapping = _reduce(f, g, m, free_dofs, algebraic_dofs, checkpoint, kinetic)
    intervals = _bracket(d, correction, bounds, num_modes, root_width, checkpoint)
    coordinates = []; values = []
    index = 0
    while index < num_modes:
        end = index+1
        while end < num_modes and intervals[end, 0]-intervals[end-1, 1] <= 8*root_width:
            end += 1
        # Do not return an arbitrary portion of a repeated eigenvalue.
        if _inertia(d, correction, intervals[index, 0]) != index or (
                _inertia(d, correction, intervals[end-1, 1]) != end):
            raise ValueError('requested modes truncate or fail to resolve a cluster')
        checkpoint('cluster.eigenvectors')
        # Stay off the root: a pure-material mode can have exactly zero
        # diagonal and correction at its root, with no scaling available.
        shift = float(intervals[index, 0]-2*root_width)
        scaled, scale = _shifted_matrix(d, correction, shift)
        roots, vectors = linalg.eigh(scaled)
        raw_all = vectors/scale[:, None]
        # Congruence does NOT preserve eigenvalue distance. For a diagonal
        # pencil every scaled root can be +/-1. Include the transformed mass
        # in the selection score, rather than selecting an unrelated mode.
        scores = np.abs(roots)/np.sum(raw_all*raw_all, axis=0)
        if not np.isfinite(scores).all(): raise ValueError('unrepresentable signed mode selection')
        selected = np.argsort(scores, kind='stable')[:end-index]
        raw = raw_all[:, selected]
        basis = _orthonormalize(raw)
        # A fixed shift strictly outside this cluster avoids attempting a
        # singular solve at its Ritz root. Two bounded block inverse steps
        # restore small strong-coordinate components through force balance.
        inverse_scale = scale
        lu = linalg.lu_factor(scaled)
        for _ in range(2):
            checkpoint('cluster.inverse_iteration')
            basis = _orthonormalize(linalg.lu_solve(lu, basis/inverse_scale[:, None])/inverse_scale[:, None])
        # Rayleigh-Ritz on a small cluster includes the SIGNED correction.
        ritz = basis.T@(d[:, None]*basis)+basis.T@correction@basis
        rv, ru = linalg.eigh(ritz)
        actual = basis@ru
        if np.any(rv < intervals[index:end, 0]-root_width) or np.any(rv > intervals[index:end, 1]+root_width):
            raise ValueError('signed Ritz values leave numerical brackets')
        coordinates.append(actual); values.extend(rv)
        index = end
    coordinates = np.column_stack(coordinates); values = np.array(values)
    if np.linalg.norm(coordinates.T@coordinates-np.eye(num_modes)) > 1e-11:
        raise ValueError('signed spectral orthogonality failed')
    diagonal_action = d[:, None]*coordinates; stress_action = correction@coordinates
    residue = diagonal_action+stress_action-coordinates*values
    action_scale = np.maximum(1., np.maximum(np.linalg.norm(diagonal_action, axis=0),
        np.maximum(np.linalg.norm(stress_action, axis=0), np.abs(values))))
    spectral_error = float(np.max(np.linalg.norm(residue, axis=0)/action_scale))
    if not np.isfinite(spectral_error) or spectral_error > 1e-11:
        raise ValueError('signed action-scaled eigenpair residual failed')
    modes = mapping@coordinates
    if np.linalg.norm(modes.T@m@modes-np.eye(num_modes)) > 1e-11:
        raise ValueError('signed physical mass normalization failed')
    for j in range(num_modes):
        if modes[int(np.argmax(np.abs(modes[:, j]))), j] < 0.: modes[:, j] *= -1.
    residue = (f.T@(f@modes)+g@modes-(m@modes)*values)[list(free_dofs)]
    scale = max(1., (np.linalg.norm(f)**2+np.linalg.norm(g))*np.linalg.norm(modes),
                np.linalg.norm(m)*np.linalg.norm(modes*values))
    backward = float(np.linalg.norm(residue)/scale)
    if not np.isfinite(backward) or backward > 1e-11:
        raise ValueError('signed full backward residual failed')
    checkpoint('output')
    return SignedFactorModes(_owned(values), _owned(modes), _owned(intervals), spectral_error, backward)
