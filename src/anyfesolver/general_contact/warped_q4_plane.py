"""Local primal-pressure work for a warped Q4 skin against an analytic plane.

This linearized small-rotation kernel is an unadmitted ingredient. A solver
owner must still certify current/swept physical gap and coordinate lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .warped_q4_skin import WarpedQ4Skin


@dataclass(frozen=True)
class WarpedQ4PlaneState:
    weighted_gap: np.ndarray
    normalized_gap: np.ndarray
    energy: float
    residual: np.ndarray
    tangent: np.ndarray
    pressure_coefficients: np.ndarray


class WarpedQ4PlaneMortar:
    """Four nonnegative Q1 pressure supports on one physical Q4 side."""

    def __init__(self, skin: WarpedQ4Skin, plane_point: np.ndarray,
                 plane_normal: np.ndarray, stiffness: float, *,
                 quadrature_order: int, control_order: int,
                 integration_relative_tolerance: float):
        if type(skin) is not WarpedQ4Skin:
            raise TypeError("warped Q4 plane mortar needs a physical Q4 skin")
        point = np.asarray(plane_point, dtype=float)
        normal = np.asarray(plane_normal, dtype=float)
        if (point.shape != (3,) or normal.shape != (3,)
                or not np.all(np.isfinite(point))
                or not np.all(np.isfinite(normal))
                or float(np.linalg.norm(normal)) <= 0.0):
            raise ValueError("plane needs a finite point and nonzero normal")
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("normal stiffness must be finite and positive")
        if (type(quadrature_order) is not int or quadrature_order < 1
                or type(control_order) is not int
                or control_order <= quadrature_order or control_order > 32):
            raise ValueError("warped contact needs increasing finite quadrature orders")
        if (not np.isfinite(integration_relative_tolerance)
                or integration_relative_tolerance <= 0.0):
            raise ValueError("integration relative tolerance must be positive")
        self.skin = skin
        self.point = point.copy()
        self.normal = normal / np.linalg.norm(normal)
        self.stiffness = float(stiffness)
        self.quadrature_order = quadrature_order
        self.control_order = control_order
        self.integration_relative_tolerance = float(integration_relative_tolerance)
        low = self._integrals(quadrature_order)
        high = self._integrals(control_order)
        support_scale = float(np.sum(high[0]))
        reference_span = float(np.max(np.ptp(skin.reference, axis=0)))
        length_scale = max(reference_span, abs(skin.offset),
                           abs(float(self.normal @ (skin.reference[0] - point))),
                           np.finfo(float).tiny)
        errors = (
            float(np.max(np.abs(low[0] - high[0]))) / support_scale,
            float(np.max(np.abs(low[1] - high[1]))) / (support_scale * length_scale),
            float(np.max(np.abs(low[2][:, :, :3] - high[2][:, :, :3]))) / support_scale,
            float(np.max(np.abs(low[2][:, :, 3:] - high[2][:, :, 3:])))
            / (support_scale * max(abs(skin.offset), np.finfo(float).tiny)),
        )
        self.integration_relative_error = max(errors)
        if self.integration_relative_error > self.integration_relative_tolerance:
            raise ValueError(
                "warped Q4 contact integration tolerance exceeded: "
                f"{self.integration_relative_error:.6g} > "
                f"{self.integration_relative_tolerance:.6g}"
            )
        self.support, self.reference_weighted_gap, local_gradient = high
        if np.any(self.support <= 0.0):
            raise ValueError("warped Q4 pressure support must be positive")
        self.gap_gradient = local_gradient.reshape(4, 24)

    def _integrals(self, order: int):
        positions, weights = np.polynomial.legendre.leggauss(order)
        support = np.zeros(4)
        weighted_gap = np.zeros(4)
        gradient = np.zeros((4, 4, 6))
        for xi, weight_xi in zip(positions, weights):
            for eta, weight_eta in zip(positions, weights):
                shape, _dx, _dy = self.skin.element.compute_shape_functions(
                    float(xi), float(eta),
                )
                shape = np.asarray(shape, dtype=float)
                sample = self.skin.sample(float(xi), float(eta))
                measure = float(weight_xi * weight_eta * sample.reference_measure)
                reference_gap = float(self.normal @ (sample.position - self.point))
                work = np.r_[self.normal,
                             self.skin.offset * np.cross(sample.director, self.normal)]
                support += measure * shape
                weighted_gap += measure * shape * reference_gap
                gradient += measure * shape[:, None, None] * shape[None, :, None] * work
        return support, weighted_gap, gradient

    def evaluate(self, nodal_coordinates: np.ndarray, *,
                 plane_normal_translation: float = 0.0) -> WarpedQ4PlaneState:
        coordinates = np.asarray(nodal_coordinates, dtype=float)
        if coordinates.shape != (4, 6) or not np.all(np.isfinite(coordinates)):
            raise ValueError("warped Q4 contact needs four finite six-DOF rows")
        if not np.isfinite(plane_normal_translation):
            raise ValueError("plane translation must be finite")
        return _evaluate_penalty(
            self.support, self.reference_weighted_gap, self.gap_gradient,
            self.stiffness, coordinates, plane_normal_translation,
        )


def _evaluate_penalty(support, reference_weighted_gap, gap_gradient,
                      stiffness, coordinates, plane_normal_translation):
    weighted = (reference_weighted_gap
                + gap_gradient @ coordinates.reshape(-1)
                - float(plane_normal_translation) * support)
    normalized = weighted / support
    closed_gap = np.minimum(normalized, 0.0)
    closed = normalized <= 0.0
    energy = 0.5 * stiffness * float(support @ (closed_gap * closed_gap))
    residual = gap_gradient.T @ (stiffness * closed_gap)
    tangent = (stiffness * gap_gradient[closed].T
               @ np.diag(1.0 / support[closed])
               @ gap_gradient[closed])
    return WarpedQ4PlaneState(
        weighted, normalized, energy, residual.reshape(coordinates.shape), tangent,
        np.maximum(0.0, -stiffness * closed_gap),
    )


class WarpedQ4PatchPlaneMortar:
    """Continuous Q1 pressure supports over two edge-sharing Q4 skins."""

    def __init__(self, skins, plane_point, plane_normal, stiffness, *,
                 quadrature_order, control_order,
                 integration_relative_tolerance):
        if len(skins) != 2 or any(type(skin) is not WarpedQ4Skin for skin in skins):
            raise ValueError("warped Q4 patch law needs exactly two physical skins")
        self.skins = tuple(skins)
        self.node_ids = tuple(sorted({node for skin in skins for node in skin.node_ids}))
        index = {node: i for i, node in enumerate(self.node_ids)}
        n_nodes = len(self.node_ids)
        self.support = np.zeros(n_nodes)
        self.reference_weighted_gap = np.zeros(n_nodes)
        gradient = np.zeros((n_nodes, n_nodes, 6))
        local_laws = tuple(WarpedQ4PlaneMortar(
            skin, plane_point, plane_normal, stiffness,
            quadrature_order=quadrature_order,
            control_order=control_order,
            integration_relative_tolerance=integration_relative_tolerance,
        ) for skin in skins)
        for skin, law in zip(skins, local_laws):
            local_gradient = law.gap_gradient.reshape(4, 4, 6)
            for a, node_a in enumerate(skin.node_ids):
                row = index[node_a]
                self.support[row] += law.support[a]
                self.reference_weighted_gap[row] += law.reference_weighted_gap[a]
                for b, node_b in enumerate(skin.node_ids):
                    gradient[row, index[node_b]] += local_gradient[a, b]
        if np.any(self.support <= 0.0):
            raise ValueError("warped Q4 patch pressure support must be positive")
        self.gap_gradient = gradient.reshape(n_nodes, 6 * n_nodes)
        self.stiffness = float(stiffness)
        self.integration_relative_error = max(
            law.integration_relative_error for law in local_laws
        )

    def evaluate(self, nodal_coordinates: np.ndarray, *,
                 plane_normal_translation: float = 0.0) -> WarpedQ4PlaneState:
        coordinates = np.asarray(nodal_coordinates, dtype=float)
        if (coordinates.shape != (len(self.node_ids), 6)
                or not np.all(np.isfinite(coordinates))):
            raise ValueError("warped Q4 patch needs finite six-DOF nodal rows")
        if not np.isfinite(plane_normal_translation):
            raise ValueError("plane translation must be finite")
        return _evaluate_penalty(
            self.support, self.reference_weighted_gap, self.gap_gradient,
            self.stiffness, coordinates, plane_normal_translation,
        )
