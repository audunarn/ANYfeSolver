"""Exact quadratic physical half-cell and work pullback for retained B3-GE."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

import numpy as np

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from anyfesolver._ge_beam3_p5.compensated_coordinates import validate_pair
from anyfesolver._ge_beam3_p5_seeded.core import sha

from .native_retained_station import _cross_matrix, _readonly


@dataclass(frozen=True)
class NativeRetainedCurveCellState:
    positions: np.ndarray
    geometric_jacobian: np.ndarray
    rotated_midpoint_lift: np.ndarray
    cell: int

    def pullback(self, force: np.ndarray,
                 tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map a Q2 curve-position potential to all 42 native coordinates."""
        gradient = _array(force, (9,), "native curve-position gradient")
        hessian = _array(tangent, (9, 9), "native curve-position tangent")
        jacobian = self.geometric_jacobian
        residual = jacobian.T @ gradient
        coupled = jacobian.T @ hessian @ jacobian
        first = 18+3*self.cell
        mid_force = gradient[3:6]
        lift = self.rotated_midpoint_lift
        coupled[first:first+3, first:first+3] += (
            0.5*(np.outer(mid_force, lift)+np.outer(lift, mid_force))
            - float(mid_force @ lift)*np.eye(3)
        )
        return _readonly(residual), _readonly(coupled)


class NativeRetainedCurveCell:
    """One physical half-cell; no contact law or pair ownership is implied."""

    def __init__(self, element_id: int,
                 reference: CenteredCurvedBeam3ReferenceGeometry,
                 cell: int):
        if type(element_id) is not int or element_id <= 0:
            raise ValueError("native curve cell needs positive element identity")
        if type(reference) is not CenteredCurvedBeam3ReferenceGeometry:
            raise TypeError("native curve cell needs exact centered B3-GE reference")
        if type(cell) is not int or cell not in (0, 1):
            raise ValueError("native curve cell needs exact half-cell identity")
        self.element_id = element_id
        self.reference = reference
        self.cell = cell
        self._reference_digest = sha(reference.canonical_data())
        self.identity = sha(dict(
            profile="NATIVE_RETAINED_Q2_CURVE_CELL_DEV_V0",
            element=element_id, reference=self._reference_digest, cell=cell,
        ))

    def evaluate(self, positions: np.ndarray, position_low: np.ndarray,
                 cell_rotations: np.ndarray) -> NativeRetainedCurveCellState:
        if sha(self.reference.canonical_data()) != self._reference_digest:
            raise ValueError("native curve cell reference identity changed")
        high = _array(positions, (3, 3), "native curve nodal positions")
        low = _array(position_low, (3, 3), "native curve nodal low parts")
        rotations = _frames(cell_rotations, 2, "native curve cell rotations")
        for first, second in zip(high.flat, low.flat):
            validate_pair(float(first), float(second))
        left, right = self.cell, self.cell+1
        endpoints = np.array([
            [fsum((float(high[node, axis]), float(low[node, axis])))
             for axis in range(3)]
            for node in (left, right)
        ])
        lift = rotations[self.cell] @ self.reference.half_cell_lift(
            self.cell, 0.5,
        )
        midpoint = np.array([
            fsum((0.5*float(high[left, axis]),
                  0.5*float(low[left, axis]),
                  0.5*float(high[right, axis]),
                  0.5*float(low[right, axis]), float(lift[axis])))
            for axis in range(3)
        ])
        points = np.vstack((endpoints[0], midpoint, endpoints[1]))
        jacobian = np.zeros((9, 42))
        jacobian[0:3, 6*left:6*left+3] = np.eye(3)
        jacobian[3:6, 6*left:6*left+3] = 0.5*np.eye(3)
        jacobian[3:6, 6*right:6*right+3] = 0.5*np.eye(3)
        jacobian[6:9, 6*right:6*right+3] = np.eye(3)
        angular = 18+3*self.cell
        jacobian[3:6, angular:angular+3] = -_cross_matrix(lift)
        return NativeRetainedCurveCellState(
            _readonly(points), _readonly(jacobian), _readonly(lift),
            self.cell,
        )
