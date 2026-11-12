"""Second-order SO(3) automatic differentiation for the private GE-B3 mixed core."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import numpy as np


class RotationDomainError(ValueError):
    """A requested principal logarithm is outside the admitted SO(3) chart."""


class Jet2:
    """Scalar value with a dense gradient and Hessian."""

    __slots__ = ("value", "gradient", "hessian")

    def __init__(self, value: float, gradient: np.ndarray, hessian: np.ndarray):
        self.value = float(value)
        self.gradient = gradient
        self.hessian = hessian

    @classmethod
    def constant(cls, value: float, size: int) -> "Jet2":
        return cls(float(value), np.zeros(size), np.zeros((size, size)))

    @classmethod
    def variable(cls, value: float, index: int, size: int) -> "Jet2":
        gradient = np.zeros(size)
        gradient[index] = 1.0
        return cls(float(value), gradient, np.zeros((size, size)))

    def _coerce(self, other: Any) -> "Jet2":
        return other if isinstance(other, Jet2) else Jet2.constant(float(other), self.gradient.size)

    def __add__(self, other: Any) -> "Jet2":
        made = self._coerce(other)
        return Jet2(self.value + made.value, self.gradient + made.gradient, self.hessian + made.hessian)

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other: Any) -> "Jet2":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> "Jet2":
        return self._coerce(other) - self

    def __mul__(self, other: Any) -> "Jet2":
        if type(self) is Jet2 and not isinstance(other, Jet2):
            scalar = float(other)
            if _scalar_layout(self) and isfinite(self.value):
                return _scalar_parts_product(self, scalar, 0.0, 0.0)
            # Preserve one observable scalar conversion even on fallback.
            other = scalar
        made = self._coerce(other)
        return Jet2(
            self.value * made.value,
            self.gradient * made.value + made.gradient * self.value,
            self.hessian * made.value
            + made.hessian * self.value
            + np.outer(self.gradient, made.gradient)
            + np.outer(made.gradient, self.gradient),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "Jet2":
        if self.value == 0.0:
            raise ZeroDivisionError("jet reciprocal is singular")
        inverse = 1.0 / self.value
        return unary(self, inverse, -(inverse**2), 2.0 * inverse**3)

    def __truediv__(self, other: Any) -> "Jet2":
        if type(self) is Jet2 and not isinstance(other, Jet2):
            scalar = float(other)
            if _scalar_layout(self) and isfinite(self.value):
                if scalar == 0.0:
                    raise ZeroDivisionError("jet reciprocal is singular")
                inverse = 1.0 / scalar
                # Same powers/order as reciprocal, including overflow errors.
                first, second = -(inverse**2), 2.0 * inverse**3
                if isfinite(first) and isfinite(second):
                    return _scalar_parts_product(
                        self, inverse, first*0.0, first*0.0 + second*0.0)
            other = scalar
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: Any) -> "Jet2":
        return self._coerce(other) * self.reciprocal()


def _scalar_layout(jet: Jet2) -> bool:
    """Fast arithmetic requires the ordinary owned binary64 derivative layout."""
    gradient, hessian = jet.gradient, jet.hessian
    return (type(jet.value) is float and type(gradient) is np.ndarray and
            type(hessian) is np.ndarray and gradient.dtype == np.float64 and
            hessian.dtype == np.float64 and gradient.ndim == 1 and
            hessian.shape == (gradient.size, gradient.size))


def _scalar_parts_product(jet: Jet2, value: float, gradient_zero: float,
                          hessian_zero: float) -> Jet2:
    """Original product order, with constant derivatives broadcast as vectors.

    Do not drop the zero products: they retain signed zero and propagation of
    nonfinite derivatives. Avoid only the constant NxN buffers/outer products.
    Every returned derivative array is independently owned.
    """
    result = jet.value * value
    gradient = jet.gradient * value + gradient_zero * jet.value
    hessian = jet.hessian * value + hessian_zero * jet.value
    hessian = hessian + (jet.gradient * gradient_zero)[:, None]
    hessian = hessian + (gradient_zero * jet.gradient)[None, :]
    return Jet2(result, gradient, hessian)


def unary(argument: Jet2, value: float, first: float, second: float) -> Jet2:
    return Jet2(
        value,
        first * argument.gradient,
        first * argument.hessian + second * np.outer(argument.gradient, argument.gradient),
    )


def sin(argument: Jet2) -> Jet2:
    return unary(argument, np.sin(argument.value), np.cos(argument.value), -np.sin(argument.value))


def cos(argument: Jet2) -> Jet2:
    return unary(argument, np.cos(argument.value), -np.sin(argument.value), -np.cos(argument.value))


def sqrt(argument: Jet2) -> Jet2:
    if argument.value <= 0.0:
        raise ValueError("jet square root requires a positive value")
    root = float(np.sqrt(argument.value))
    return unary(argument, root, 0.5 / root, -0.25 / root**3)


def zeros(rows: int, columns: int, size: int) -> list[list[Jet2]]:
    return [[Jet2.constant(0.0, size) for _ in range(columns)] for _ in range(rows)]


def constant_matrix(values: Any, size: int) -> list[list[Jet2]]:
    array = np.asarray(values, dtype=np.float64)
    return [[Jet2.constant(array[row, column], size) for column in range(array.shape[1])] for row in range(array.shape[0])]


def identity(size: int) -> list[list[Jet2]]:
    return constant_matrix(np.eye(3), size)


def transpose(matrix: Sequence[Sequence[Jet2]]) -> list[list[Jet2]]:
    return [[matrix[row][column] for row in range(len(matrix))] for column in range(len(matrix[0]))]


def matmul(left: Sequence[Sequence[Jet2]], right: Sequence[Sequence[Jet2]]) -> list[list[Jet2]]:
    size = left[0][0].gradient.size
    result = zeros(len(left), len(right[0]), size)
    for row in range(len(left)):
        for column in range(len(right[0])):
            result[row][column] = sum(
                (left[row][index] * right[index][column] for index in range(len(right))),
                start=Jet2.constant(0.0, size),
            )
    return result


def matvec(matrix: Sequence[Sequence[Jet2]], vector: Sequence[Jet2]) -> list[Jet2]:
    size = vector[0].gradient.size
    return [
        sum((coefficient * entry for coefficient, entry in zip(row, vector)), start=Jet2.constant(0.0, size))
        for row in matrix
    ]


def dot(left: Sequence[Jet2], right: Sequence[Jet2]) -> Jet2:
    return sum((a * b for a, b in zip(left, right)), start=Jet2.constant(0.0, left[0].gradient.size))


def skew(vector: Sequence[Jet2]) -> list[list[Jet2]]:
    x, y, z = vector
    zero = Jet2.constant(0.0, x.gradient.size)
    return [[zero, -z, y], [z, zero, -x], [-y, x, zero]]


def _exp_coefficients(argument: Jet2) -> tuple[Jet2, Jet2]:
    one = Jet2.constant(1.0, argument.gradient.size)
    if abs(argument.value) < 1.0e-8:
        x = argument
        x2 = x * x
        x3 = x2 * x
        x4 = x3 * x
        return (
            one - x / 6.0 + x2 / 120.0 - x3 / 5040.0 + x4 / 362880.0,
            one / 2.0 - x / 24.0 + x2 / 720.0 - x3 / 40320.0 + x4 / 3628800.0,
        )
    theta = sqrt(argument)
    return sin(theta) / theta, (one - cos(theta)) / argument


def so3_exp(vector: Sequence[Jet2]) -> list[list[Jet2]]:
    size = vector[0].gradient.size
    argument = sum((entry * entry for entry in vector), start=Jet2.constant(0.0, size))
    a, b = _exp_coefficients(argument)
    cross = skew(vector)
    cross2 = matmul(cross, cross)
    result = identity(size)
    for row in range(3):
        for column in range(3):
            result[row][column] = result[row][column] + a * cross[row][column] + b * cross2[row][column]
    return result


def _log_factor(cosine: Jet2) -> Jet2:
    if cosine.value > 1.0 - 1.0e-7:
        delta = 1.0 - cosine
        return 1.0 + delta / 3.0 + 2.0 * delta * delta / 15.0 + 2.0 * delta * delta * delta / 35.0
    if not math_cos_limit() < cosine.value < 1.0:
        raise RotationDomainError("relative rotation must remain below 0.9*pi; refine or cut back")
    angle = float(np.arccos(cosine.value))
    sine = float(np.sqrt(1.0 - cosine.value**2))
    value = angle / sine
    first = -1.0 / sine**2 + angle * cosine.value / sine**3
    second = -3.0 * cosine.value / sine**4 + angle / sine**3 + 3.0 * angle * cosine.value**2 / sine**5
    return unary(cosine, value, first, second)


def math_cos_limit() -> float:
    """Cosine threshold for the frozen relative-rotation chart."""

    return float(np.cos(0.9 * np.pi))


def so3_log(matrix: Sequence[Sequence[Jet2]]) -> list[Jet2]:
    cosine = (matrix[0][0] + matrix[1][1] + matrix[2][2] - 1.0) / 2.0
    factor = _log_factor(cosine)
    axial = [
        (matrix[2][1] - matrix[1][2]) / 2.0,
        (matrix[0][2] - matrix[2][0]) / 2.0,
        (matrix[1][0] - matrix[0][1]) / 2.0,
    ]
    return [factor * entry for entry in axial]


def rotation_exponential(vector: Any) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("rotation vector must contain three finite values")
    theta2 = float(values @ values)
    cross = np.array(((0.0, -values[2], values[1]), (values[2], 0.0, -values[0]), (-values[1], values[0], 0.0)))
    if theta2 < 1.0e-16:
        a = 1.0 - theta2 / 6.0 + theta2**2 / 120.0
        b = 0.5 - theta2 / 24.0 + theta2**2 / 720.0
    else:
        theta = float(np.sqrt(theta2))
        a = float(np.sin(theta) / theta)
        b = float((1.0 - np.cos(theta)) / theta2)
    return np.eye(3) + a * cross + b * (cross @ cross)


def rotation_log(matrix: Any) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    axial = 0.5 * np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]))
    if angle < 1.0e-8:
        return (1.0 + angle**2 / 6.0 + 7.0 * angle**4 / 360.0) * axial
    if angle >= 0.9 * np.pi:
        raise RotationDomainError("relative rotation must remain below 0.9*pi; refine or cut back")
    return angle / np.sin(angle) * axial


def proper_rotation(values: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must have shape (3, 3) and be finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=1.0e-10) or abs(float(np.linalg.det(matrix)) - 1.0) > 1.0e-10:
        raise ValueError(f"{label} is not a proper rotation")
    return np.array(matrix, copy=True)
