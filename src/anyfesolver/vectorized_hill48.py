"""Compiled common path for batched Hill-48 plane-stress plasticity.

The public constitutive contract and validation remain in :mod:`plasticity`.
This module only owns the flattened numerical kernel, canonical hardening-curve
packing, and observable activation/fallback diagnostics.  Unsupported curve
protocols deliberately stay on the scalar oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .jit_compiler import (
    JIT_BACKEND,
    JIT_DISABLED_REASON,
    JIT_ENABLED,
    njit,
)
from .material_curves import (
    DNVC208MaterialCurve,
    LinearHardeningCurve,
    PiecewiseLinearCurve,
    PowerLawHardeningCurve,
)
from .nonlinear_analysis_diagnostics import record_hill48_analysis_execution


_CURVE_PERFECT = 0
_CURVE_LINEAR = 1
_CURVE_PIECEWISE = 2
_CURVE_POWER = 3
_CURVE_DNV_C208 = 4

_STATUS_OK = 0
_STATUS_BRACKET_FAILED = 1
_STATUS_NOT_CONVERGED = 2
_STATUS_NUMERICAL_FAILURE = 3

_CURVE_NAMES = {
    _CURVE_PERFECT: "perfect_plasticity",
    _CURVE_LINEAR: "linear",
    _CURVE_PIECEWISE: "piecewise_linear",
    _CURVE_POWER: "power_law",
    _CURVE_DNV_C208: "dnv_c208",
}
_STATUS_NAMES = {
    _STATUS_BRACKET_FAILED: "compiled_bracket_failure",
    _STATUS_NOT_CONVERGED: "compiled_nonconverged",
    _STATUS_NUMERICAL_FAILURE: "compiled_numerical_failure",
}


@dataclass(frozen=True, slots=True)
class Hill48CurvePack:
    """Fixed numerical representation of one canonical hardening curve."""

    kind: int
    parameters: np.ndarray
    plastic_strain: np.ndarray
    flow_stress: np.ndarray

    @property
    def name(self) -> str:
        return _CURVE_NAMES[int(self.kind)]


def _readonly_contiguous(values: Any) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def pack_hill48_curve(
    curve: Any | None,
) -> Tuple[Optional[Hill48CurvePack], Optional[str]]:
    """Pack a canonical ANYmaterial curve or return its scalar fallback reason.

    Exact type checks are intentional.  A subclass may override either curve
    method, so treating it as its parent would silently narrow the arbitrary
    curve protocol supported by :func:`hill48_plane_stress_return_map`.
    """

    empty = _readonly_contiguous(np.empty(0, dtype=np.float64))
    if curve is None:
        return Hill48CurvePack(
            _CURVE_PERFECT,
            _readonly_contiguous(np.zeros(8, dtype=np.float64)),
            empty,
            empty,
        ), None
    curve_type = type(curve)
    if curve_type is LinearHardeningCurve:
        parameters = np.zeros(8, dtype=np.float64)
        parameters[0] = float(curve.sigma_yield)
        parameters[1] = float(curve.hardening_modulus_value)
        return Hill48CurvePack(
            _CURVE_LINEAR,
            _readonly_contiguous(parameters),
            empty,
            empty,
        ), None
    if curve_type is PiecewiseLinearCurve:
        return Hill48CurvePack(
            _CURVE_PIECEWISE,
            _readonly_contiguous(np.zeros(8, dtype=np.float64)),
            _readonly_contiguous(curve.plastic_strain),
            _readonly_contiguous(curve.flow_stress_values),
        ), None
    if curve_type is PowerLawHardeningCurve:
        parameters = np.zeros(8, dtype=np.float64)
        parameters[0] = float(curve.K)
        parameters[1] = float(curve.n)
        parameters[2] = float(curve.eps_0)
        return Hill48CurvePack(
            _CURVE_POWER,
            _readonly_contiguous(parameters),
            empty,
            empty,
        ), None
    if curve_type is DNVC208MaterialCurve:
        parameters = np.asarray(
            [
                float(curve.sigma_prop),
                float(curve.sigma_yield),
                float(curve.sigma_yield_2),
                float(curve.eps_p_y1),
                float(curve.eps_p_y2),
                float(curve.K),
                float(curve.n),
                float(curve._power_offset),
            ],
            dtype=np.float64,
        )
        return Hill48CurvePack(
            _CURVE_DNV_C208,
            _readonly_contiguous(parameters),
            empty,
            empty,
        ), None
    return None, "custom_curve_protocol"


@njit(cache=True)
def _curve_flow_and_modulus(
    kind: int,
    alpha: float,
    parameters: np.ndarray,
    curve_strain: np.ndarray,
    curve_stress: np.ndarray,
) -> Tuple[float, float, bool]:
    value = max(alpha, 0.0)
    if kind == _CURVE_PERFECT:
        return 1.0, 0.0, True
    if kind == _CURVE_LINEAR:
        return parameters[0] + parameters[1] * value, parameters[1], True
    if kind == _CURVE_PIECEWISE:
        count = curve_strain.shape[0]
        if count < 2:
            return np.nan, np.nan, False
        if value >= curve_strain[count - 1]:
            return curve_stress[count - 1], 0.0, True
        interval = 0
        for index in range(count - 1):
            if value < curve_strain[index + 1]:
                interval = index
                break
        denominator = curve_strain[interval + 1] - curve_strain[interval]
        if denominator <= 0.0 or not np.isfinite(denominator):
            return np.nan, np.nan, False
        slope = (
            curve_stress[interval + 1] - curve_stress[interval]
        ) / denominator
        flow = curve_stress[interval] + slope * (
            value - curve_strain[interval]
        )
        return flow, slope, np.isfinite(flow) and np.isfinite(slope)
    if kind == _CURVE_POWER:
        base = value + parameters[2]
        if base <= 0.0 or not np.isfinite(base):
            return np.nan, np.nan, False
        flow = parameters[0] * np.power(base, parameters[1])
        hardening = (
            parameters[0]
            * parameters[1]
            * np.power(base, parameters[1] - 1.0)
        )
        return flow, hardening, np.isfinite(flow) and np.isfinite(hardening)
    if kind == _CURVE_DNV_C208:
        if value <= parameters[3]:
            slope = (parameters[1] - parameters[0]) / parameters[3]
            flow = parameters[0] + slope * value
            return flow, slope, np.isfinite(flow) and np.isfinite(slope)
        if value <= parameters[4]:
            slope = (parameters[2] - parameters[1]) / (
                parameters[4] - parameters[3]
            )
            flow = parameters[1] + slope * (value - parameters[3])
            return flow, slope, np.isfinite(flow) and np.isfinite(slope)
        base = max(value + parameters[7], 1.0e-12)
        flow = parameters[5] * np.power(base, parameters[6])
        hardening = (
            parameters[5]
            * parameters[6]
            * np.power(base, parameters[6] - 1.0)
        )
        return flow, hardening, np.isfinite(flow) and np.isfinite(hardening)
    return np.nan, np.nan, False


@njit(cache=True)
def _scaled_curve_flow_and_modulus(
    kind: int,
    alpha: float,
    parameters: np.ndarray,
    curve_strain: np.ndarray,
    curve_stress: np.ndarray,
    reference_strength: float,
    reference_flow: float,
) -> Tuple[float, float, bool]:
    if kind == _CURVE_PERFECT:
        return reference_strength, 0.0, True
    flow, hardening, valid = _curve_flow_and_modulus(
        kind,
        alpha,
        parameters,
        curve_strain,
        curve_stress,
    )
    scale = reference_strength / reference_flow
    flow *= scale
    hardening *= scale
    hardening_scale = max(abs(flow), reference_strength, 1.0)
    valid = bool(
        valid
        and np.isfinite(flow)
        and np.isfinite(hardening)
        and flow > 0.0
        and hardening >= -1.0e-12 * hardening_scale
    )
    return flow, max(hardening, 0.0), valid


@njit(cache=True)
def _projection_state(
    trial_0: float,
    trial_1: float,
    trial_2: float,
    elastic_metric: np.ndarray,
    metric: np.ndarray,
    gamma: float,
) -> Tuple[
    bool,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Evaluate the scalar oracle's 3-by-3 projection without LAPACK calls."""

    a00 = 1.0 + gamma * elastic_metric[0]
    a01 = gamma * elastic_metric[1]
    a02 = gamma * elastic_metric[2]
    a10 = gamma * elastic_metric[3]
    a11 = 1.0 + gamma * elastic_metric[4]
    a12 = gamma * elastic_metric[5]
    a20 = gamma * elastic_metric[6]
    a21 = gamma * elastic_metric[7]
    a22 = 1.0 + gamma * elastic_metric[8]

    i00 = a11 * a22 - a12 * a21
    i01 = a02 * a21 - a01 * a22
    i02 = a01 * a12 - a02 * a11
    i10 = a12 * a20 - a10 * a22
    i11 = a00 * a22 - a02 * a20
    i12 = a02 * a10 - a00 * a12
    i20 = a10 * a21 - a11 * a20
    i21 = a01 * a20 - a00 * a21
    i22 = a00 * a11 - a01 * a10
    determinant = a00 * i00 + a01 * i10 + a02 * i20
    matrix_scale = max(
        abs(a00), abs(a01), abs(a02),
        abs(a10), abs(a11), abs(a12),
        abs(a20), abs(a21), abs(a22), 1.0,
    )
    if (
        not np.isfinite(determinant)
        or abs(determinant) <= 1.0e-300 * matrix_scale**3
    ):
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    inverse_determinant = 1.0 / determinant
    i00 *= inverse_determinant
    i01 *= inverse_determinant
    i02 *= inverse_determinant
    i10 *= inverse_determinant
    i11 *= inverse_determinant
    i12 *= inverse_determinant
    i20 *= inverse_determinant
    i21 *= inverse_determinant
    i22 *= inverse_determinant

    stress_0 = i00 * trial_0 + i01 * trial_1 + i02 * trial_2
    stress_1 = i10 * trial_0 + i11 * trial_1 + i12 * trial_2
    stress_2 = i20 * trial_0 + i21 * trial_1 + i22 * trial_2
    metric_stress_0 = (
        metric[0] * stress_0 + metric[1] * stress_1 + metric[2] * stress_2
    )
    metric_stress_1 = (
        metric[3] * stress_0 + metric[4] * stress_1 + metric[5] * stress_2
    )
    metric_stress_2 = (
        metric[6] * stress_0 + metric[7] * stress_1 + metric[8] * stress_2
    )
    equivalent_squared = (
        stress_0 * metric_stress_0
        + stress_1 * metric_stress_1
        + stress_2 * metric_stress_2
    )
    if not np.isfinite(equivalent_squared) or equivalent_squared <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    equivalent = np.sqrt(equivalent_squared)
    normal_0 = metric_stress_0 / equivalent
    normal_1 = metric_stress_1 / equivalent
    normal_2 = metric_stress_2 / equivalent

    rhs_0 = (
        elastic_metric[0] * stress_0
        + elastic_metric[1] * stress_1
        + elastic_metric[2] * stress_2
    )
    rhs_1 = (
        elastic_metric[3] * stress_0
        + elastic_metric[4] * stress_1
        + elastic_metric[5] * stress_2
    )
    rhs_2 = (
        elastic_metric[6] * stress_0
        + elastic_metric[7] * stress_1
        + elastic_metric[8] * stress_2
    )
    derivative_0 = -(i00 * rhs_0 + i01 * rhs_1 + i02 * rhs_2)
    derivative_1 = -(i10 * rhs_0 + i11 * rhs_1 + i12 * rhs_2)
    derivative_2 = -(i20 * rhs_0 + i21 * rhs_1 + i22 * rhs_2)
    equivalent_derivative = (
        normal_0 * derivative_0
        + normal_1 * derivative_1
        + normal_2 * derivative_2
    )
    valid = bool(
        np.isfinite(stress_0)
        and np.isfinite(stress_1)
        and np.isfinite(stress_2)
        and np.isfinite(equivalent_derivative)
    )
    return (
        valid,
        stress_0,
        stress_1,
        stress_2,
        equivalent,
        normal_0,
        normal_1,
        normal_2,
        equivalent_derivative,
    )


@njit(cache=True)
def _solve_4x4_3rhs(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    solution: np.ndarray,
) -> bool:
    """Small pivoted elimination used only for the analytical tangent."""

    augmented = np.empty((4, 7), dtype=np.float64)
    scale = 1.0
    for row in range(4):
        for column in range(4):
            value = matrix[row, column]
            augmented[row, column] = value
            scale = max(scale, abs(value))
        for column in range(3):
            augmented[row, 4 + column] = right_hand_side[row, column]
    for pivot_column in range(4):
        pivot_row = pivot_column
        pivot_size = abs(augmented[pivot_row, pivot_column])
        for candidate in range(pivot_column + 1, 4):
            candidate_size = abs(augmented[candidate, pivot_column])
            if candidate_size > pivot_size:
                pivot_size = candidate_size
                pivot_row = candidate
        if not np.isfinite(pivot_size) or pivot_size <= 1.0e-300 * scale:
            return False
        if pivot_row != pivot_column:
            for column in range(pivot_column, 7):
                temporary = augmented[pivot_column, column]
                augmented[pivot_column, column] = augmented[pivot_row, column]
                augmented[pivot_row, column] = temporary
        pivot = augmented[pivot_column, pivot_column]
        for column in range(pivot_column, 7):
            augmented[pivot_column, column] /= pivot
        for row in range(4):
            if row == pivot_column:
                continue
            factor = augmented[row, pivot_column]
            for column in range(pivot_column, 7):
                augmented[row, column] -= factor * augmented[pivot_column, column]
    for row in range(4):
        for column in range(3):
            value = augmented[row, 4 + column]
            if not np.isfinite(value):
                return False
            solution[row, column] = value
    return True


@njit(cache=True)
def _jit_hill48_return_map_flat(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    elastic: np.ndarray,
    metric: np.ndarray,
    reference_strength: float,
    reference_flow: float,
    curve_kind: int,
    curve_parameters: np.ndarray,
    curve_strain: np.ndarray,
    curve_stress: np.ndarray,
    max_iterations: int,
    tolerance: float,
    upper_seed: float,
    tangent_amplification_limit: float,
    compute_tangent: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Flattened deterministic batch kernel matching the safeguarded oracle."""

    point_count = alpha.shape[0]
    stress = np.empty(point_count * 3, dtype=np.float64)
    tangent = np.zeros(point_count * 9, dtype=np.float64)
    new_plastic = plastic_strain.copy()
    new_alpha = alpha.copy()
    status = np.zeros(point_count, dtype=np.int8)
    scaled_residual = np.zeros(point_count, dtype=np.float64)
    tangent_valid = np.ones(point_count, dtype=np.bool_)

    elastic_metric = np.empty(9, dtype=np.float64)
    for row in range(3):
        for column in range(3):
            value = 0.0
            for inner in range(3):
                value += elastic[row * 3 + inner] * metric[inner * 3 + column]
            elastic_metric[row * 3 + column] = value
    elastic_norm_squared = 0.0
    for index in range(9):
        elastic_norm_squared += elastic[index] * elastic[index]
    elastic_norm = max(np.sqrt(elastic_norm_squared), 1.0)

    for point in range(point_count):
        offset = point * 3
        delta_0 = strain[offset] - plastic_strain[offset]
        delta_1 = strain[offset + 1] - plastic_strain[offset + 1]
        delta_2 = strain[offset + 2] - plastic_strain[offset + 2]
        trial_0 = (
            elastic[0] * delta_0 + elastic[1] * delta_1 + elastic[2] * delta_2
        )
        trial_1 = (
            elastic[3] * delta_0 + elastic[4] * delta_1 + elastic[5] * delta_2
        )
        trial_2 = (
            elastic[6] * delta_0 + elastic[7] * delta_1 + elastic[8] * delta_2
        )
        stress[offset] = trial_0
        stress[offset + 1] = trial_1
        stress[offset + 2] = trial_2
        if compute_tangent:
            tangent_offset = point * 9
            for index in range(9):
                tangent[tangent_offset + index] = elastic[index]

        flow_n, _hardening_n, curve_valid = _scaled_curve_flow_and_modulus(
            curve_kind,
            alpha[point],
            curve_parameters,
            curve_strain,
            curve_stress,
            reference_strength,
            reference_flow,
        )
        metric_trial_0 = (
            metric[0] * trial_0 + metric[1] * trial_1 + metric[2] * trial_2
        )
        metric_trial_1 = (
            metric[3] * trial_0 + metric[4] * trial_1 + metric[5] * trial_2
        )
        metric_trial_2 = (
            metric[6] * trial_0 + metric[7] * trial_1 + metric[8] * trial_2
        )
        equivalent_trial_squared = max(
            trial_0 * metric_trial_0
            + trial_1 * metric_trial_1
            + trial_2 * metric_trial_2,
            0.0,
        )
        equivalent_trial = np.sqrt(equivalent_trial_squared)
        if not curve_valid or not np.isfinite(equivalent_trial):
            status[point] = _STATUS_NUMERICAL_FAILURE
            continue
        if (
            equivalent_trial - flow_n
            <= tolerance * max(flow_n, equivalent_trial, 1.0)
        ):
            continue

        lower = 0.0
        upper = upper_seed
        upper_bracketed = False
        for _ in range(128):
            (
                projection_valid,
                returned_0,
                returned_1,
                returned_2,
                equivalent,
                normal_0,
                normal_1,
                normal_2,
                equivalent_derivative,
            ) = _projection_state(
                trial_0,
                trial_1,
                trial_2,
                elastic_metric,
                metric,
                upper,
            )
            if not projection_valid:
                break
            plastic_increment = upper * equivalent
            flow, hardening, curve_valid = _scaled_curve_flow_and_modulus(
                curve_kind,
                alpha[point] + plastic_increment,
                curve_parameters,
                curve_strain,
                curve_stress,
                reference_strength,
                reference_flow,
            )
            if not curve_valid:
                break
            residual = equivalent - flow
            if residual <= 0.0:
                upper_bracketed = True
                break
            upper *= 2.0
        if not upper_bracketed:
            status[point] = (
                _STATUS_BRACKET_FAILED if projection_valid and curve_valid
                else _STATUS_NUMERICAL_FAILURE
            )
            continue

        gamma = lower
        converged = False
        projection_valid = True
        curve_valid = True
        residual = np.inf
        residual_scale = 1.0
        for _ in range(max_iterations):
            (
                projection_valid,
                returned_0,
                returned_1,
                returned_2,
                equivalent,
                normal_0,
                normal_1,
                normal_2,
                equivalent_derivative,
            ) = _projection_state(
                trial_0,
                trial_1,
                trial_2,
                elastic_metric,
                metric,
                gamma,
            )
            if not projection_valid:
                break
            plastic_increment = gamma * equivalent
            flow, hardening, curve_valid = _scaled_curve_flow_and_modulus(
                curve_kind,
                alpha[point] + plastic_increment,
                curve_parameters,
                curve_strain,
                curve_stress,
                reference_strength,
                reference_flow,
            )
            if not curve_valid:
                break
            residual = equivalent - flow
            residual_scale = max(abs(equivalent), abs(flow), 1.0)
            if abs(residual) <= tolerance * residual_scale:
                converged = True
                break
            if residual > 0.0:
                lower = gamma
            else:
                upper = gamma
            plastic_increment_derivative = (
                equivalent + gamma * equivalent_derivative
            )
            residual_derivative = (
                equivalent_derivative
                - hardening * plastic_increment_derivative
            )
            candidate = np.nan
            if np.isfinite(residual_derivative) and residual_derivative < 0.0:
                candidate = gamma - residual / residual_derivative
            if (
                not np.isfinite(candidate)
                or candidate <= lower
                or candidate >= upper
            ):
                candidate = 0.5 * (lower + upper)
            gamma = candidate

        if not converged and projection_valid and curve_valid:
            (
                projection_valid,
                returned_0,
                returned_1,
                returned_2,
                equivalent,
                normal_0,
                normal_1,
                normal_2,
                equivalent_derivative,
            ) = _projection_state(
                trial_0,
                trial_1,
                trial_2,
                elastic_metric,
                metric,
                gamma,
            )
            if projection_valid:
                plastic_increment = gamma * equivalent
                flow, hardening, curve_valid = _scaled_curve_flow_and_modulus(
                    curve_kind,
                    alpha[point] + plastic_increment,
                    curve_parameters,
                    curve_strain,
                    curve_stress,
                    reference_strength,
                    reference_flow,
                )
                residual = abs(equivalent - flow)
                residual_scale = max(abs(equivalent), abs(flow), 1.0)
                converged = bool(
                    curve_valid and residual <= tolerance * residual_scale
                )
        scaled_residual[point] = abs(residual) / residual_scale
        if not projection_valid or not curve_valid:
            status[point] = _STATUS_NUMERICAL_FAILURE
            continue
        if not converged:
            status[point] = _STATUS_NOT_CONVERGED
            continue

        stress[offset] = returned_0
        stress[offset + 1] = returned_1
        stress[offset + 2] = returned_2
        new_plastic[offset] = plastic_strain[offset] + plastic_increment * normal_0
        new_plastic[offset + 1] = (
            plastic_strain[offset + 1] + plastic_increment * normal_1
        )
        new_plastic[offset + 2] = (
            plastic_strain[offset + 2] + plastic_increment * normal_2
        )
        new_alpha[point] = alpha[point] + plastic_increment

        if not compute_tangent:
            continue
        hessian = np.empty((3, 3), dtype=np.float64)
        normals = np.asarray([normal_0, normal_1, normal_2])
        for row in range(3):
            for column in range(3):
                hessian[row, column] = (
                    metric[row * 3 + column]
                    - normals[row] * normals[column]
                ) / equivalent
        jacobian = np.zeros((4, 4), dtype=np.float64)
        right_hand_side = np.zeros((4, 3), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                elastic_hessian = 0.0
                for inner in range(3):
                    elastic_hessian += (
                        elastic[row * 3 + inner] * hessian[inner, column]
                    )
                jacobian[row, column] = (
                    (1.0 if row == column else 0.0)
                    + plastic_increment * elastic_hessian
                )
                right_hand_side[row, column] = elastic[row * 3 + column]
            jacobian[row, 3] = (
                elastic[row * 3] * normal_0
                + elastic[row * 3 + 1] * normal_1
                + elastic[row * 3 + 2] * normal_2
            )
            jacobian[3, row] = normals[row]
        jacobian[3, 3] = -hardening
        solution = np.empty((4, 3), dtype=np.float64)
        valid_solution = _solve_4x4_3rhs(
            jacobian,
            right_hand_side,
            solution,
        )
        tangent_offset = point * 9
        tangent_norm_squared = 0.0
        if valid_solution:
            for row in range(3):
                for column in range(3):
                    local_value = 0.5 * (
                        solution[row, column] + solution[column, row]
                    )
                    tangent[tangent_offset + row * 3 + column] = local_value
                    tangent_norm_squared += local_value * local_value
        tangent_valid[point] = bool(
            valid_solution
            and np.isfinite(tangent_norm_squared)
            and np.sqrt(tangent_norm_squared)
            <= tangent_amplification_limit * elastic_norm
        )
        if not tangent_valid[point]:
            for index in range(9):
                tangent[tangent_offset + index] = np.nan

    return (
        stress,
        tangent,
        new_plastic,
        new_alpha,
        status,
        scaled_residual,
        tangent_valid,
    )


def compiled_hill48_return_map(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    elastic_matrix: np.ndarray,
    metric: np.ndarray,
    reference_strength: float,
    reference_flow: float,
    curve_pack: Hill48CurvePack,
    max_iterations: int,
    tolerance: float,
    tangent_amplification_limit: float,
    compute_tangent: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run the compiled flattened kernel on already validated inputs."""

    if not JIT_ENABLED:
        raise RuntimeError(JIT_DISABLED_REASON or "Numba JIT is unavailable")
    point_count = int(np.asarray(alpha).shape[0])
    upper_seed = 1.0 / max(
        float(np.linalg.norm(elastic_matrix, ord=2)),
        1.0,
    )
    result = _jit_hill48_return_map_flat(
        np.ascontiguousarray(strain, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(plastic_strain, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.ascontiguousarray(elastic_matrix, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(metric, dtype=np.float64).reshape(-1),
        float(reference_strength),
        float(reference_flow),
        int(curve_pack.kind),
        curve_pack.parameters,
        curve_pack.plastic_strain,
        curve_pack.flow_stress,
        int(max_iterations),
        float(tolerance),
        float(upper_seed),
        float(tangent_amplification_limit),
        bool(compute_tangent),
    )
    stress, tangent, new_plastic, new_alpha, status, residual, tangent_valid = result
    return (
        stress.reshape(point_count, 3),
        tangent.reshape(point_count, 3, 3),
        new_plastic.reshape(point_count, 3),
        new_alpha,
        status,
        residual,
        tangent_valid,
    )


_DIAGNOSTIC_LOCK = RLock()
_DIAGNOSTICS: Dict[str, Any] = {}


def reset_hill48_vectorized_diagnostics() -> None:
    """Reset bounded process diagnostics without touching compiled code."""

    with _DIAGNOSTIC_LOCK:
        _DIAGNOSTICS.clear()
        _DIAGNOSTICS.update(
            {
                "public_call_count": 0,
                "point_count": 0,
                "compiled_call_count": 0,
                "compiled_point_count": 0,
                "scalar_fallback_call_count": 0,
                "scalar_fallback_point_count": 0,
                "row_fallback_count": 0,
                "numerical_tangent_row_count": 0,
                "fallback_reason_counts": {},
                "last_call": None,
            }
        )


reset_hill48_vectorized_diagnostics()


def record_hill48_execution(
    *,
    point_count: int,
    curve_name: str,
    compiled: bool,
    scalar_fallback_points: int = 0,
    fallback_reason_counts: Optional[Mapping[str, int]] = None,
    numerical_tangent_rows: int = 0,
) -> None:
    """Record one public constitutive execution from :mod:`plasticity`."""

    points = int(point_count)
    scalar_points = int(scalar_fallback_points)
    reasons = {
        str(reason): int(count)
        for reason, count in (fallback_reason_counts or {}).items()
        if int(count) > 0
    }
    with _DIAGNOSTIC_LOCK:
        _DIAGNOSTICS["public_call_count"] += 1
        _DIAGNOSTICS["point_count"] += points
        if compiled:
            _DIAGNOSTICS["compiled_call_count"] += 1
            _DIAGNOSTICS["compiled_point_count"] += points
        if scalar_points:
            _DIAGNOSTICS["scalar_fallback_call_count"] += 1
            _DIAGNOSTICS["scalar_fallback_point_count"] += scalar_points
        row_fallbacks = sum(reasons.values())
        _DIAGNOSTICS["row_fallback_count"] += row_fallbacks
        _DIAGNOSTICS["numerical_tangent_row_count"] += int(
            numerical_tangent_rows
        )
        stored_reasons = _DIAGNOSTICS["fallback_reason_counts"]
        for reason, count in reasons.items():
            stored_reasons[reason] = int(stored_reasons.get(reason, 0)) + count
        _DIAGNOSTICS["last_call"] = {
            "point_count": points,
            "curve": str(curve_name),
            "path": "compiled" if compiled else "scalar_reference",
            "scalar_fallback_points": scalar_points,
            "fallback_reason_counts": dict(sorted(reasons.items())),
            "numerical_tangent_rows": int(numerical_tangent_rows),
        }
    record_hill48_analysis_execution(
        point_count=points,
        curve_name=curve_name,
        compiled=compiled,
        scalar_fallback_points=scalar_points,
        fallback_reason_counts=reasons,
        numerical_tangent_rows=numerical_tangent_rows,
    )


def hill48_vectorized_diagnostics() -> Dict[str, Any]:
    """Return observable fast-path activation and row-fallback counters."""

    with _DIAGNOSTIC_LOCK:
        snapshot = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in _DIAGNOSTICS.items()
        }
        if isinstance(_DIAGNOSTICS.get("last_call"), dict):
            snapshot["last_call"] = dict(_DIAGNOSTICS["last_call"])
            snapshot["last_call"]["fallback_reason_counts"] = dict(
                _DIAGNOSTICS["last_call"].get("fallback_reason_counts", {})
            )
    snapshot.update(
        {
            "fast_path": "hill48_flattened_numba_return_map",
            "jit_enabled": bool(JIT_ENABLED),
            "jit_backend": JIT_BACKEND,
            "jit_disabled_reason": JIT_DISABLED_REASON,
            "supported_curve_types": tuple(
                _CURVE_NAMES[index]
                for index in (
                    _CURVE_PERFECT,
                    _CURVE_LINEAR,
                    _CURVE_PIECEWISE,
                    _CURVE_POWER,
                    _CURVE_DNV_C208,
                )
            ),
        }
    )
    return snapshot


def status_fallback_reason(status: int) -> str:
    return _STATUS_NAMES.get(int(status), "compiled_unknown_failure")


__all__ = (
    "Hill48CurvePack",
    "compiled_hill48_return_map",
    "hill48_vectorized_diagnostics",
    "pack_hill48_curve",
    "record_hill48_execution",
    "reset_hill48_vectorized_diagnostics",
    "status_fallback_reason",
)
