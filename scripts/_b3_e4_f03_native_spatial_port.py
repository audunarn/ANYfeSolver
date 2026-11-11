"""Bounded six-channel native B3 station port for G0 F03 derivation checks.

The output rows are spatial translation and left-multiplicative angular
velocity. This is a private kinematic proof seam, not a coupled owner or a
selected line-attachment law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from anyfesolver._ge_beam3_retained_generalized_state import Context, Mechanical
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def _affine_pair(left_high: float, left_low: float,
                 right_high: float, right_low: float,
                 weight: float, lift: float) -> tuple[float, float]:
    """Split the exact affine combination of the supplied binary64 inputs.

    This bounded derivation port uses exact rationals to avoid rounding each
    weighted absolute coordinate before cancellation at a large origin.
    """
    try:
        left = Fraction(float(left_high))+Fraction(float(left_low))
        right = Fraction(float(right_high))+Fraction(float(right_low))
        value = left+Fraction(float(weight))*(right-left)+Fraction(float(lift))
        high = float(value)
        low = float(value-Fraction(high))
    except (OverflowError, ValueError) as error:
        raise ValueError("native station coordinate overflow") from error
    if not math.isfinite(high) or not math.isfinite(low):
        raise ValueError("native station coordinate overflow")
    return split_sum((high, low))


@dataclass(frozen=True)
class NativeHalfCellSpatialPort:
    """One explicit side of one actual native retained half-cell."""

    context: Context
    element_index: int
    cell: int
    weight: float

    def __post_init__(self) -> None:
        if (not isinstance(self.context, Context) or
                type(self.element_index) is not int or
                not 0 <= self.element_index < len(self.context.elements) or
                type(self.cell) is not int or self.cell not in (0, 1) or
                isinstance(self.weight, (bool, str)) or
                not isinstance(self.weight, (int, float, np.floating)) or
                not np.isfinite(self.weight) or
                not 0. <= self.weight <= 1.):
            raise ValueError("valid native element, half-cell and station required")

    def _parts(self, state: Mechanical):
        if not isinstance(state, Mechanical):
            raise TypeError("retained native mechanical state required")
        self.context.guard()
        # Trial states need not be issued accepted states, but their complete
        # mechanical payload must satisfy this context's ownership contract.
        state = self.context.make(state.descriptor())
        nodes = self.context.nodes[self.element_index]
        left, right = nodes[self.cell:self.cell+2]
        reference = self.context.elements[self.element_index][1].operator.reference
        lift = reference.half_cell_lift(self.cell, self.weight)
        frame = reference.frame(self.cell-1+self.weight)
        cell_rotation = state.cell_rotations[self.element_index, self.cell]
        return state, left, right, lift, frame, cell_rotation

    def pose_pair(self, state: Mechanical
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return owned high/low position and current material frame."""
        state, left, right, lift, frame, rotation = self._parts(state)
        displacement = rotation@lift
        pairs = [_affine_pair(
            state.positions[left, axis], state.position_low[left, axis],
            state.positions[right, axis], state.position_low[right, axis],
            self.weight, displacement[axis]) for axis in range(3)]
        high, low = np.asarray(pairs, dtype=float).T
        return high.copy(), low.copy(), rotation@frame

    def pose(self, state: Mechanical) -> tuple[np.ndarray, np.ndarray]:
        """Return display/high position and current material frame."""
        high, _, frame = self.pose_pair(state)
        return high, frame

    def tangent(self, state: Mechanical) -> np.ndarray:
        """Map Context.advance increments to [delta r, spatial delta theta]."""
        _, left, right, lift, _, rotation = self._parts(state)
        result = np.zeros((6, self.context.count))
        result[:3, 6*left:6*left+3] = (1-self.weight)*np.eye(3)
        result[:3, 6*right:6*right+3] = self.weight*np.eye(3)
        cell_rotation = self.context.nodal_count+24*self.element_index+3*self.cell
        result[:3, cell_rotation:cell_rotation+3] = -_skew(rotation@lift)
        result[3:, cell_rotation:cell_rotation+3] = np.eye(3)
        return result

    def pullback(self, state: Mechanical, spatial_wrench: np.ndarray) -> np.ndarray:
        """Pull a [force, physical moment] wrench into retained coordinates."""
        wrench = np.asarray(spatial_wrench, dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("finite six-component spatial wrench required")
        return self.tangent(state).T@wrench

    def second_variation(self, state: Mechanical, first: np.ndarray,
                         second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric mixed derivative of (position, frame matrix) in one chart.

        Both directions are applied through one ``Context.advance`` exponential
        from the same state. This is not the Hessian of a relative-pose Log
        constraint; that constraint needs its own second-variation chain rule.
        """
        _, _, _, lift, frame, rotation = self._parts(state)
        u, v = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        if (u.shape != (self.context.count,) or
                v.shape != (self.context.count,) or
                not np.isfinite(u).all() or not np.isfinite(v).all()):
            raise ValueError("finite retained variation directions required")
        column = self.context.nodal_count+24*self.element_index+3*self.cell
        a, b = _skew(u[column:column+3]), _skew(v[column:column+3])
        mixed = .5*(a@b+b@a)
        return mixed@(rotation@lift), mixed@(rotation@frame)


@dataclass(frozen=True)
class NativeNodalSpatialPort:
    """Unique physical nodal pose at a native member junction or termination.

    A half-cell material frame is one-sided at an internal node. Its endpoint
    value must not silently stand in for this shared nodal frame when a point
    attachment or a footprint is declared at the node.
    """

    context: Context
    node_index: int

    def __post_init__(self) -> None:
        if (not isinstance(self.context, Context) or
                type(self.node_index) is not int or
                not 0 <= self.node_index < self.context.nodal_count // 6):
            raise ValueError("valid native physical node required")

    def pose_pair(self, state: Mechanical
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(state, Mechanical):
            raise TypeError("retained native mechanical state required")
        self.context.guard()
        owned = self.context.make(state.descriptor())
        node = self.node_index
        return (owned.positions[node].copy(), owned.position_low[node].copy(),
                owned.nodal_frames[node].copy())

    def tangent(self, state: Mechanical) -> np.ndarray:
        self.pose_pair(state)
        result = np.zeros((6, self.context.count))
        start = 6 * self.node_index
        result[:, start:start+6] = np.eye(6)
        return result

    def pullback(self, state: Mechanical, spatial_wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(spatial_wrench, dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("finite six-component spatial wrench required")
        return self.tangent(state).T @ wrench

    def second_variation(self, state: Mechanical, first: np.ndarray,
                         second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, frame = self.pose_pair(state)
        u, v = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        if (u.shape != (self.context.count,) or
                v.shape != (self.context.count,) or
                not np.isfinite(u).all() or not np.isfinite(v).all()):
            raise ValueError("finite retained variation directions required")
        start = 6 * self.node_index + 3
        a, b = _skew(u[start:start+3]), _skew(v[start:start+3])
        return np.zeros(3), .5 * (a @ b + b @ a) @ frame
