"""Development column batching of the original finite retained-cell product rules.

Consumes original prepared section/station data. Geometry, quadrature, physical
centroid and compensated momentum-pair recovery remain with existing owners.
The station accumulation order and base force work are unchanged. No dispatch.
"""
import numpy as np

from ._ge_beam3_p5.algebra import HALVES, skew, validate_section
from ._ge_beam3_p5.arrays import _array, _frames, _readonly
from ._ge_beam3_p5.inertia import FiniteInertiaResponse

IMPLEMENTATION_ID = 'DEVELOPMENT_C7_CENTERED_BATCHED_FINITE_INERTIA_V1'


def _prepared(section_mass, stations):
    """Validate bounded prepared inputs; no reference geometry is reconstructed."""
    section = validate_section(section_mass)
    if type(stations) is not tuple or not 4 <= len(stations) <= 128:
        raise ValueError('bounded original prepared station tuple required')
    result = []
    counts = [0, 0]
    for row in stations:
        if type(row) is not tuple or len(row) != 7:
            raise ValueError('complete original prepared station required')
        cell, left, right, t, frame, offset, measure = row
        if (type(cell) is not int or cell not in (0, 1)
                or type(left) is not int or type(right) is not int
                or (left, right) != HALVES[cell]):
            raise ValueError('original two-cell station connectivity required')
        if (type(t) is not float or not np.isfinite(t) or not 0. < t < 1.
                or type(measure) is not float or not np.isfinite(measure) or measure <= 0.):
            raise ValueError('finite interior station/positive measure required')
        frame = _frames([frame], 1, 'prepared station frame')[0]
        offset = _array(offset, (3,), 'prepared station offset')
        result.append((cell, left, right, t, frame, offset, measure))
        counts[cell] += 1
    if counts[0] != counts[1] or not 2 <= counts[0] <= 64:
        raise ValueError('matching original bounded two-cell station counts required')
    return section, result


def _cross_columns(left, right):
    """Cross product on 3-by-C columns; a length-three operand broadcasts."""
    a = left.T if np.ndim(left) == 2 else left
    b = right.T if np.ndim(right) == 2 else right
    return np.cross(a, b).T


def evaluate_batched_finite_inertia(section_mass, stations, positions,
        local_rotations, velocity, acceleration):
    """Return the original complete physical24 response from prepared owners.

    Configuration derivatives are left spatial cell increments at fixed supplied
    spatial velocity/acceleration. Batching changes contraction granularity only;
    dv/dq remain generally nonsymmetric force derivatives. Nodal-frame trace
    rows/columns remain exact zeros. Existing centered owner owns the42 extension
    and compensated linear/angular momentum recovery about its stated origin.
    """
    j, prepared = _prepared(section_mass, stations)
    x = _array(positions, (3, 3), 'nodal positions')
    cells = _frames(local_rotations, 2, 'cell rotations')
    speed = _array(velocity, (24,), 'spatial velocity')
    accel = _array(acceleration, (24,), 'spatial acceleration')
    mass = np.zeros((24, 24))
    inertia, momentum = np.zeros(24), np.zeros(24)
    dv, dq = np.zeros_like(mass), np.zeros_like(mass)
    linear, angular, energy = np.zeros(3), np.zeros(3), 0.
    for cell, left, right, t, r0, d0, measure in prepared:
        slot = slice(18+3*cell, 21+3*cell)
        q, d = cells[cell]@r0, cells[cell]@d0
        b = np.zeros((6, 24))
        b[:3, 6*left:6*left+3] = (1-t)*np.eye(3)
        b[:3, 6*right:6*right+3] = t*np.eye(3)
        b[:3, slot], b[3:, slot] = -skew(d), np.eye(3)
        transform = np.kron(np.eye(2), q)
        body_map = transform.T@b
        field = b@speed
        v, omega = field[:3], field[3:]
        acceleration_field = b@accel
        alpha = acceleration_field[3:]
        a = acceleration_field[:3]+np.cross(omega, np.cross(omega, d))
        u, w = q.T@v, q.T@omega
        body_velocity = np.r_[u, w]
        body_acceleration = np.r_[q.T@a-np.cross(w, u), q.T@alpha]
        p, h = np.split(j@body_velocity, 2)
        force_body, moment_body = np.split(j@body_acceleration, 2)
        force_body += np.cross(w, p)
        moment_body += np.cross(w, h)+np.cross(u, p)
        f, n = q@force_body, q@moment_body
        station_force = np.r_[f, n]
        energy += measure*float(body_velocity@(j@body_velocity))/2
        mass += measure*(body_map.T@j@body_map)
        momentum += measure*(body_map.T@(j@body_velocity))
        inertia += measure*(b.T@station_force)
        linear += measure*(q@p)
        angular += measure*(np.cross((1-t)*x[left]+t*x[right]+d, q@p)+q@h)

        # First C columns are active velocity variations; final3 are eta=e_k.
        columns = np.flatnonzero(np.any(b != 0., axis=0))
        count = len(columns)
        eta = np.zeros((3, count+3))
        eta[:, count:] = np.eye(3)
        dd = np.zeros_like(eta)
        dd[:, count:] = _cross_columns(eta[:, count:], d)
        delta_v, delta_omega = np.zeros_like(eta), np.zeros_like(eta)
        delta_v[:, :count], delta_omega[:, :count] = b[:3, columns], b[3:, columns]
        delta_v[:, count:] = _cross_columns(omega, dd[:, count:])
        delta_a = np.zeros_like(eta)
        delta_a[:, :count] = (_cross_columns(delta_omega[:, :count], np.cross(omega, d))
            +_cross_columns(omega, _cross_columns(delta_omega[:, :count], d)))
        delta_a[:, count:] = (_cross_columns(alpha, dd[:, count:])
            +_cross_columns(omega, _cross_columns(omega, dd[:, count:])))
        # All delta_alpha columns are zero in these two derivative families.
        du = q.T@(delta_v-_cross_columns(eta, v))
        dw = q.T@(delta_omega-_cross_columns(eta, omega))
        da = (q.T@(delta_a-_cross_columns(eta, a))
              -_cross_columns(dw, u)-_cross_columns(w, du))
        dalpha = q.T@(-_cross_columns(eta, alpha))
        dp, dh = np.split(j@np.vstack((du, dw)), 2)
        df, dn = np.split(j@np.vstack((da, dalpha)), 2)
        df += _cross_columns(dw, p)+_cross_columns(w, dp)
        dn += (_cross_columns(dw, h)+_cross_columns(w, dh)
               +_cross_columns(du, p)+_cross_columns(u, dp))
        changed = b.T@np.vstack((q@df+_cross_columns(eta, f), q@dn+_cross_columns(eta, n)))
        changed[slot] += _cross_columns(dd, f)
        dv[:, columns] += measure*changed[:, :count]
        dq[:, slot] += measure*changed[:, count:]
    arrays = (mass, momentum, inertia, dv, dq, linear, angular)
    if not np.isfinite(energy) or energy < 0. or any(not np.isfinite(value).all() for value in arrays):
        raise ValueError('nonfinite kinetic response')
    return FiniteInertiaResponse(float(energy), *(_readonly(value) for value in arrays))
