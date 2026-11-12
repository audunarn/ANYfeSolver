"""Separate development owner for centered finite retained-cell inertia.

Preserves the original finite physical kernel; public/rest admission is unchanged.
Positions enter compensated angular momentum recovery only. No time integrator.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import numpy as np

from ._ge_beam3_p5.arrays import _array, _frames, _readonly
from ._ge_beam3_p5.compensated_coordinates import validate_pair
from ._ge_beam3_p5.inertia import LiftedInertia, FiniteInertiaResponse
from ._ge_beam3_p5_centered.mass import _CenteredRestInertia
from ._ge_beam3_p5_centered.positions import station_position
from ._ge_beam3_p5_coordinates.positions import _rounded_pair

IMPLEMENTATION_ID = "DEVELOPMENT_C7_CENTERED_FINITE_INERTIA_V1"
PHYSICAL_DOFS = (0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 22, 23)
ALGEBRAIC_DOFS = (3, 4, 5, 9, 10, 11, 15, 16, 17, *range(24, 42))


class _CenteredFiniteKernel(LiftedInertia):
    """Finite owner, initialized exclusively from the unchanged centered cache.

    This is not a rest-owner instance and never calls its evaluate method.
    The reviewed column batching supplies the original finite force product
    rules. LiftedInertia.evaluate remains the unchanged scalar verification route.
    """
    def __init__(self, reference, section_inertia, order):
        centered = _CenteredRestInertia(reference, section_inertia, order=order)
        self.section_mass = centered.section_mass
        self._stations = centered._stations

    def evaluate(self, positions, local_rotations, velocity, acceleration):
        from ._ge_beam3_centered_batched_finite_inertia import evaluate_batched_finite_inertia
        return evaluate_batched_finite_inertia(self.section_mass, self._stations,
            positions, local_rotations, velocity, acceleration)


@dataclass(frozen=True)
class CenteredFiniteInertiaResponse:
    physical: FiniteInertiaResponse
    linear_momentum_low: np.ndarray
    angular_momentum_low: np.ndarray
    momentum_origin_high: np.ndarray
    momentum_origin_low: np.ndarray
    mixed_mass: np.ndarray
    mixed_generalized_momentum: np.ndarray
    mixed_inertia: np.ndarray
    mixed_velocity_derivative: np.ndarray
    mixed_configuration_derivative: np.ndarray
    kinetic_station_count: int
    physical_dofs: tuple = PHYSICAL_DOFS
    algebraic_dofs: tuple = ALGEBRAIC_DOFS
    implementation_id: str = IMPLEMENTATION_ID
    production_qualified: bool = False

    @property
    def kinetic_energy(self): return self.physical.kinetic_energy

    @property
    def mass(self): return self.physical.mass

    @property
    def generalized_momentum(self): return self.physical.generalized_momentum

    @property
    def inertia(self): return self.physical.inertia

    @property
    def velocity_derivative(self): return self.physical.velocity_derivative

    @property
    def configuration_derivative(self): return self.physical.configuration_derivative

    @property
    def linear_momentum(self): return self.physical.linear_momentum

    @property
    def angular_momentum(self): return self.physical.angular_momentum


def _pair(high, low, shape, label):
    high = _array(high, shape, label+" high")
    low = _array(low, shape, label+" low")
    for h, l in zip(high.flat, low.flat):
        validate_pair(float(h), float(l))
    return high, low


def _momentum_pairs(kernel, high, low, cells, velocity, origin_high, origin_low):
    """Recover physical P and H about O from the preserved station velocity map.

    The six-component section-momentum contraction is repeated only to recover
    station momenta, which the original aggregate response does not expose.
    Finite force work/M/derivatives are exclusively inherited from LiftedInertia.
    """
    linear = [F(0)]*3
    angular = [F(0)]*3
    origin = [F(float(h))+F(float(l)) for h, l in zip(origin_high, origin_low)]
    for cell, left, right, t, frame, offset, measure in kernel._stations:
        rotation = cells[cell]
        q = rotation @ frame
        lift = rotation @ offset
        omega = velocity[18+3*cell:21+3*cell]
        # Exact same represented B and field evaluation as the finite owner.
        x, y, z = lift
        skew = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
        b = np.zeros((6, 24))
        b[:3, 6*left:6*left+3] = (1-t)*np.eye(3)
        b[:3, 6*right:6*right+3] = t*np.eye(3)
        b[:3, 18+3*cell:21+3*cell] = -skew
        b[3:, 18+3*cell:21+3*cell] = np.eye(3)
        field = b @ velocity
        body_momentum = kernel.section_mass @ np.r_[q.T @ field[:3], q.T @ omega]
        p, h = q @ body_momentum[:3], q @ body_momentum[3:]
        station_high, station_low = station_position(high, low, cell, float(t), rotation, offset)
        lever = [F(float(a))+F(float(b))-o for a, b, o in zip(station_high, station_low, origin)]
        fp, fh, weight = [F(float(v)) for v in p], [F(float(v)) for v in h], F(float(measure))
        cross = (lever[1]*fp[2]-lever[2]*fp[1],
                 lever[2]*fp[0]-lever[0]*fp[2],
                 lever[0]*fp[1]-lever[1]*fp[0])
        for axis in range(3):
            linear[axis] += weight*fp[axis]
            angular[axis] += weight*(cross[axis]+fh[axis])
    lp = np.array([_rounded_pair(v) for v in linear])
    hp = np.array([_rounded_pair(v) for v in angular])
    return _readonly(lp[:, 0]), _readonly(lp[:, 1]), _readonly(hp[:, 0]), _readonly(hp[:, 1])


def _mixed(value):
    """Zero extension in nodal18 / cell6 / algebraic-resultants18 ordering."""
    shape = (42,) if value.ndim == 1 else (42, 42)
    result = np.zeros(shape)
    if value.ndim == 1:
        result[:24] = value
    else:
        result[:24, :24] = value
    return _readonly(result)


def evaluate_centered_finite_inertia(
        reference, section_inertia, positions_high, positions_low, cell_rotations,
        velocity, acceleration, *, order=24, momentum_origin_high=(0., 0., 0.),
        momentum_origin_low=(0., 0., 0.)):
    """Pure finite physical response with compensated momentum origin.

    Spatial velocity/acceleration order: nodal18 + two cell spins6. Left cell
    configuration increments hold supplied spatial velocity/acceleration fixed.
    Returned physical arrays have size24; mixed arrays have size42 and exact
    zero algebraic inertia. No symmetric-potential tangent transformation.
    """
    high, low = _pair(positions_high, positions_low, (3, 3), "positions")
    origin_high, origin_low = _pair(momentum_origin_high, momentum_origin_low, (3,), "momentum origin")
    cells = _frames(cell_rotations, 2, "cell rotations")
    speed = _array(velocity, (24,), "spatial velocity")
    accel = _array(acceleration, (24,), "spatial acceleration")
    kernel = _CenteredFiniteKernel(reference, section_inertia, order)
    # Positions affect only the base kernel's temporary angular recovery.
    # Subtraction of full pairs avoids overflow/loss from a moving common origin.
    relative = np.array([[float(F(float(high[i, j]))+F(float(low[i, j]))
                                -F(float(origin_high[j]))-F(float(origin_low[j])))
                          for j in range(3)] for i in range(3)])
    physical = kernel.evaluate(relative, cells, speed, accel)
    lp, ll, hp, hl = _momentum_pairs(kernel, high, low, cells, speed, origin_high, origin_low)
    physical = replace(physical, linear_momentum=lp, angular_momentum=hp)
    return CenteredFiniteInertiaResponse(
        physical, ll, hl, _readonly(origin_high), _readonly(origin_low),
        _mixed(physical.mass), _mixed(physical.generalized_momentum),
        _mixed(physical.inertia), _mixed(physical.velocity_derivative),
        _mixed(physical.configuration_derivative), len(kernel._stations))
