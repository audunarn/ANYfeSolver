"""Scalar integration for literal 2025 Eq. 21/Eqs. 24-25 MITC4+/D data."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mitc4_plus_d_reference import (
    generalized_strain_operators,
    local_strain_operator,
)
from .protocol import LinearShellSection, Q4ReferenceData


FloatArray = NDArray[np.float64]
ConstitutiveProvider = Callable[[float, float, float], ArrayLike]
DensityProvider = Callable[[float, float, float], float]


def isotropic_plane_stress_constitutive(
    youngs_modulus: float, poisson_ratio: float
) -> FloatArray:
    """Return homogeneous local ``C`` in the five-component shell order.

    Transverse shear is the physical isotropic shear modulus; no shear or
    drilling correction factor is used.
    """

    youngs_modulus = float(youngs_modulus)
    poisson_ratio = float(poisson_ratio)
    if not np.isfinite(youngs_modulus) or youngs_modulus <= 0.0:
        raise ValueError("youngs_modulus must be finite and positive")
    if not np.isfinite(poisson_ratio) or not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must lie strictly between -1 and 0.5")
    factor = youngs_modulus / (1.0 - poisson_ratio * poisson_ratio)
    shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    constitutive = np.zeros((5, 5), dtype=np.float64)
    constitutive[0, 0] = factor
    constitutive[0, 1] = factor * poisson_ratio
    constitutive[1, 0] = factor * poisson_ratio
    constitutive[1, 1] = factor
    constitutive[2, 2] = shear
    constitutive[3, 3] = shear
    constitutive[4, 4] = shear
    return constitutive


def _validated_constitutive(value: ArrayLike, name: str = "constitutive") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (5, 5):
        raise ValueError(f"{name} must have shape (5, 5), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def linear_stiffness(
    reference: Q4ReferenceData,
    constitutive: ArrayLike | ConstitutiveProvider,
) -> FloatArray:
    """Integrate homogeneous/layered continuum stiffness with full 2 x 2 x 2.

    A callable is a scalar-oracle convenience and is invoked as
    ``constitutive(r, s, zeta)``.  Production dispatch stacks constitutive
    arrays before entering compiled loops.
    """

    stiffness = np.zeros((24, 24), dtype=np.float64)
    constant = None if callable(constitutive) else _validated_constitutive(constitutive)

    for surface_index in range(4):
        for thickness_index in range(2):
            r, s, zeta = reference.volume_points[surface_index, thickness_index]
            material = (
                _validated_constitutive(
                    constitutive(float(r), float(s), float(zeta)),
                    "constitutive provider result",
                )
                if callable(constitutive)
                else constant
            )
            operator = reference.volume_strain_operators[
                surface_index, thickness_index
            ]
            weight = reference.volume_weights[surface_index, thickness_index]
            stiffness += weight * (operator.T @ material @ operator)
    return stiffness


def linear_stiffness_generalized(
    reference: Q4ReferenceData,
    section: LinearShellSection,
) -> FloatArray:
    """Integrate a generalized ``A/B/D/As`` section over four surface points.

    The upper-right coupling block is ``section.coupling`` and the lower-left
    block is its transpose, preserving a supplied nonsymmetric ``B`` exactly.
    """

    if not isinstance(section, LinearShellSection):
        raise TypeError("section must be LinearShellSection")
    stiffness = np.zeros((24, 24), dtype=np.float64)
    for station in range(4):
        membrane = reference.generalized_membrane_operators[station]
        bending = reference.generalized_bending_operators[station]
        shear = reference.generalized_shear_operators[station]
        weight = reference.surface_weights[station]
        stiffness += weight * (
            membrane.T @ section.membrane @ membrane
            + membrane.T @ section.coupling @ bending
            + bending.T @ section.coupling.T @ membrane
            + bending.T @ section.bending @ bending
            + shear.T @ section.shear @ shear
        )
    return stiffness


def consistent_mass(
    reference: Q4ReferenceData,
    density: float | DensityProvider,
) -> FloatArray:
    """Integrate consistent translational, rotary, and physical `/D` inertia."""

    constant_density: float | None
    if callable(density):
        constant_density = None
    else:
        constant_density = float(density)
        if not np.isfinite(constant_density) or constant_density < 0.0:
            raise ValueError("density must be finite and non-negative")

    mass = np.zeros((24, 24), dtype=np.float64)
    for surface_index in range(4):
        for thickness_index in range(2):
            r, s, zeta = reference.volume_points[surface_index, thickness_index]
            rho = (
                float(density(float(r), float(s), float(zeta)))
                if callable(density)
                else constant_density
            )
            if not np.isfinite(rho) or rho < 0.0:
                raise ValueError("density provider must return a finite non-negative value")
            operator = reference.volume_displacement_operators[
                surface_index, thickness_index
            ]
            weight = reference.volume_weights[surface_index, thickness_index]
            mass += rho * weight * (operator.T @ operator)
    return mass


def linear_residual_tangent(
    reference: Q4ReferenceData,
    displacement: ArrayLike,
    constitutive: ArrayLike | ConstitutiveProvider,
) -> tuple[FloatArray, FloatArray]:
    """Return the exact scalar linear residual and analytical tangent."""

    vector = np.asarray(displacement, dtype=np.float64)
    if vector.shape != (24,) or not np.all(np.isfinite(vector)):
        raise ValueError("displacement must be a finite vector with shape (24,)")
    tangent = linear_stiffness(reference, constitutive)
    return tangent @ vector, tangent


def linear_residual_tangent_generalized(
    reference: Q4ReferenceData,
    displacement: ArrayLike,
    section: LinearShellSection,
) -> tuple[FloatArray, FloatArray]:
    """Return exact generalized-section residual and analytical tangent."""

    vector = np.asarray(displacement, dtype=np.float64)
    if vector.shape != (24,) or not np.all(np.isfinite(vector)):
        raise ValueError("displacement must be a finite vector with shape (24,)")
    tangent = linear_stiffness_generalized(reference, section)
    return tangent @ vector, tangent


def strain_at(
    reference: Q4ReferenceData,
    displacement: ArrayLike,
    r: float,
    s: float,
    zeta: float,
) -> FloatArray:
    """Evaluate local engineering strain from the authoritative operator."""

    vector = np.asarray(displacement, dtype=np.float64)
    if vector.shape != (24,) or not np.all(np.isfinite(vector)):
        raise ValueError("displacement must be a finite vector with shape (24,)")
    operator, _ = local_strain_operator(reference, r, s, zeta)
    return operator @ vector


def generalized_strain_at(
    reference: Q4ReferenceData,
    displacement: ArrayLike,
    r: float,
    s: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate ``(epsilon0, kappa, gamma)`` at one surface point."""

    vector = np.asarray(displacement, dtype=np.float64)
    if vector.shape != (24,) or not np.all(np.isfinite(vector)):
        raise ValueError("displacement must be a finite vector with shape (24,)")
    membrane, bending, shear = generalized_strain_operators(reference, r, s)
    return membrane @ vector, bending @ vector, shear @ vector
