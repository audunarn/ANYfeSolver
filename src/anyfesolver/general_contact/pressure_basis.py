"""Topology-only candidate bases for a refined Q4 source pressure strip.

This builds pressure interpolation data. It does not change the contact law,
integration side, virtual work, tangent, or solver admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np


MACRO_Q1_PAIR_VERSION = "q4_source_macro_q1_pair_v1"
SELECTED_Q1_KNOTS_VERSION = "q4_source_selected_q1_knots_v1"
MAX_STRIP_CELLS = 32
MAX_SELECTED_STRIP_CELLS = 128


@dataclass(frozen=True)
class PressureBasis:
    """Immutable fine-node to coarse-pressure interpolation on one Q4 strip.

    ``B[i][a]`` is the value of coarse basis function ``a`` at fine source
    node ``i``. The node and mode orders are explicit and both use stable
    material IDs, never current coordinates.
    """

    version: str
    fine_source_node_ids: tuple[int, ...]
    mode_ids: tuple[int, ...]
    B: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        try:
            fine = tuple(self.fine_source_node_ids)
            modes = tuple(self.mode_ids)
            matrix = tuple(tuple(float(value) for value in row)
                           for row in self.B)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pressure-basis IDs and rows must be finite sequences") from exc
        object.__setattr__(self, "fine_source_node_ids", fine)
        object.__setattr__(self, "mode_ids", modes)
        object.__setattr__(self, "B", matrix)
        if self.version not in {MACRO_Q1_PAIR_VERSION,
                                SELECTED_Q1_KNOTS_VERSION}:
            raise ValueError("unsupported Q4 source pressure-basis version")
        if (not fine or not modes
                or any(type(node) is not int or node <= 0 for node in (*fine, *modes))
                or len(set(fine)) != len(fine)
                or len(set(modes)) != len(modes)
                or not set(modes).issubset(fine)
                or len(matrix) != len(fine)):
            raise ValueError("pressure-basis IDs or matrix dimensions are invalid")
        for row in matrix:
            if (len(row) != len(modes)
                    or any(not isfinite(value) or value < 0.0 for value in row)
                    or abs(sum(row)-1.0) > 1.0e-14):
                raise ValueError("pressure basis must be finite, positive and partition unity")
        fine_index = {node: index for index, node in enumerate(fine)}
        for column, node in enumerate(modes):
            anchor = matrix[fine_index[node]]
            if any(value != (1.0 if index == column else 0.0)
                   for index, value in enumerate(anchor)):
                raise ValueError("each pressure mode must have its own nodal anchor")

    def as_array(self) -> np.ndarray:
        """Return a detached numeric copy of the immutable interpolation."""
        return np.asarray(self.B, dtype=float).copy()

    def transform_integrals(
        self,
        fine_support: np.ndarray,
        fine_weighted_gap: np.ndarray,
        fine_gram: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform assembled reference integrals before gap normalization.

        The caller must assemble all fine-face contributions using the exact
        ``fine_source_node_ids`` order. This operation neither projects a
        pressure nor changes the physical gap certificate.
        """
        count = len(self.fine_source_node_ids)
        support = np.asarray(fine_support, dtype=float)
        weighted_gap = np.asarray(fine_weighted_gap, dtype=float)
        gram = np.asarray(fine_gram, dtype=float)
        if (support.shape != (count,) or weighted_gap.shape != (count,)
                or gram.shape != (count, count)
                or not np.all(np.isfinite(support))
                or not np.all(np.isfinite(weighted_gap))
                or not np.all(np.isfinite(gram))
                or np.any(support < 0.0)):
            raise ValueError("fine mortar integrals have invalid shape or values")
        scale = float(np.max(np.abs(gram), initial=0.0))
        symmetry_tolerance = 64.0*np.finfo(float).eps*scale
        if np.max(np.abs(gram-gram.T), initial=0.0) > symmetry_tolerance:
            raise ValueError("fine pressure Gram must be symmetric")
        symmetric_gram = 0.5*(gram+gram.T)
        eigenvalues = np.linalg.eigvalsh(symmetric_gram)
        if eigenvalues[0] < -count*symmetry_tolerance:
            raise ValueError("fine pressure Gram must be positive semidefinite")
        B = self.as_array()
        return B.T @ support, B.T @ weighted_gap, B.T @ gram @ B

    def transform_derivatives(
        self,
        fine_support_gradient: np.ndarray,
        fine_weighted_gap_gradient: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform assembled integral gradients with unchanged coordinates."""
        count = len(self.fine_source_node_ids)
        support = np.asarray(fine_support_gradient, dtype=float)
        weighted_gap = np.asarray(fine_weighted_gap_gradient, dtype=float)
        if (support.ndim != 2 or support.shape[0] != count
                or weighted_gap.shape != support.shape
                or not np.all(np.isfinite(support))
                or not np.all(np.isfinite(weighted_gap))):
            raise ValueError("fine mortar gradients have invalid shape or values")
        B = self.as_array()
        return B.T @ support, B.T @ weighted_gap


def build_q4_source_macro_q1_pair_basis(
    ordered_faces: Iterable[tuple[int, int, int, int]],
) -> PressureBasis:
    """Build one macro Q1 pressure cell per adjacent pair of Q4 faces.

    Faces must be declared in strip order as (lower-left, lower-right,
    upper-right, upper-left), all with the same combinatorial orientation.
    Incidence alone cannot certify physical Jacobian sign or element shape;
    the geometry owner retains those checks.
    """
    try:
        faces = tuple(tuple(face) for face in ordered_faces)
    except TypeError as exc:
        raise ValueError("Q4 strip must declare ordered four-node faces") from exc
    cells = len(faces)
    if cells < 2 or cells % 2 or cells > MAX_STRIP_CELLS:
        raise ValueError("macro Q1 pressure basis needs 2–32 even ordered Q4 cells")
    for face in faces:
        if (len(face) != 4
                or any(type(node) is not int or node <= 0 for node in face)
                or len(set(face)) != 4):
            raise ValueError("each Q4 face needs four distinct positive material node IDs")

    lower = [faces[0][0]]
    upper = [faces[0][3]]
    for face in faces:
        if face[0] != lower[-1] or face[3] != upper[-1]:
            raise ValueError("Q4 faces must share one oriented cross-strip edge")
        lower.append(face[1])
        upper.append(face[2])
    rows = (tuple(lower), tuple(upper))
    if len(set(lower + upper)) != 2*(cells+1):
        raise ValueError("Q4 strip reuses a node outside its shared edges")

    fine = tuple(sorted(lower + upper))
    anchors = tuple(row[column] for row in rows
                    for column in range(0, cells+1, 2))
    modes = tuple(sorted(anchors))
    mode_index = {node: index for index, node in enumerate(modes)}
    weights: dict[int, tuple[float, ...]] = {}
    for row in rows:
        for column, node in enumerate(row):
            values = [0.0]*len(modes)
            if column % 2 == 0:
                values[mode_index[node]] = 1.0
            else:
                values[mode_index[row[column-1]]] = 0.5
                values[mode_index[row[column+1]]] = 0.5
            weights[node] = tuple(values)
    return PressureBasis(
        MACRO_Q1_PAIR_VERSION, fine, modes,
        tuple(weights[node] for node in fine),
    )


def build_q4_source_selected_q1_basis(
    ordered_faces: Iterable[tuple[int, int, int, int]],
    selected_columns: Iterable[int],
) -> PressureBasis:
    """Build a fixed nonnegative Q1 pressure subspace from structural columns.

    Selected knots are material-node columns, including both strip ends.
    Between knots, pressure is linear in the ordered reference face index.
    The contact owner still integrates physical work on the original Q4 faces;
    this transform introduces no new contact geometry or multiplier on a
    quadrature subdivision. Its topology and mode IDs remain fixed during
    the analysis.
    """
    try:
        faces = tuple(tuple(face) for face in ordered_faces)
        knots = tuple(selected_columns)
    except TypeError as exc:
        raise ValueError("selected Q1 basis needs faces and knot columns") from exc
    cells = len(faces)
    if cells < 2 or cells > MAX_SELECTED_STRIP_CELLS:
        raise ValueError("selected Q1 basis needs 2–128 ordered Q4 cells")
    if (len(knots) < 2 or knots[0] != 0 or knots[-1] != cells
            or any(type(column) is not int or column < 0 or column > cells
                   for column in knots)
            or any(right <= left for left, right in zip(knots, knots[1:]))):
        raise ValueError("selected pressure columns must strictly span the Q4 strip")
    for face in faces:
        if (len(face) != 4
                or any(type(node) is not int or node <= 0 for node in face)
                or len(set(face)) != 4):
            raise ValueError("each Q4 face needs four distinct positive material node IDs")
    lower = [faces[0][0]]
    upper = [faces[0][3]]
    for face in faces:
        if face[0] != lower[-1] or face[3] != upper[-1]:
            raise ValueError("Q4 faces must share one oriented cross-strip edge")
        lower.append(face[1])
        upper.append(face[2])
    rows = (tuple(lower), tuple(upper))
    if len(set(lower+upper)) != 2*(cells+1):
        raise ValueError("Q4 strip reuses a node outside its shared edges")
    fine = tuple(sorted(lower+upper))
    modes = tuple(sorted(row[column] for row in rows for column in knots))
    mode_index = {node: index for index, node in enumerate(modes)}
    weights = {}
    for row in rows:
        for column, node in enumerate(row):
            values = [0.0]*len(modes)
            if column in knots:
                values[mode_index[node]] = 1.0
            else:
                right_index = next(index for index, knot in enumerate(knots)
                                   if knot > column)
                left, right = knots[right_index-1:right_index+1]
                fraction = (column-left)/(right-left)
                values[mode_index[row[left]]] = 1.0-fraction
                values[mode_index[row[right]]] = fraction
            weights[node] = tuple(values)
    return PressureBasis(
        SELECTED_Q1_KNOTS_VERSION, fine, modes,
        tuple(weights[node] for node in fine),
    )
