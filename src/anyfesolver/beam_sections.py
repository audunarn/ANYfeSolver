"""Generalized beam-section contracts.

The constitutive law is written in beam-local axes as

``resultants = stiffness @ generalized_strains``

with the documented orders

``[eps_x, gamma_xy, gamma_xz, kappa_x, kappa_y, kappa_z]`` and
``[N, V_y, V_z, T, M_y, M_z]``.

Entries in a sectional stiffness matrix therefore intentionally carry mixed
units.  The matrix is validated after diagonal normalization so a physically
reasonable disparity between axial, shear, torsional, and bending stiffnesses
does not look numerically indefinite merely because their units differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


GENERALIZED_BEAM_STRAIN_ORDER: Tuple[str, ...] = (
    "eps_x",
    "gamma_xy",
    "gamma_xz",
    "kappa_x",
    "kappa_y",
    "kappa_z",
)
GENERALIZED_BEAM_RESULTANT_ORDER: Tuple[str, ...] = (
    "N",
    "V_y",
    "V_z",
    "T",
    "M_y",
    "M_z",
)
GENERALIZED_BEAM_VELOCITY_ORDER: Tuple[str, ...] = (
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
)


@runtime_checkable
class GeneralizedBeamSectionContract(Protocol):
    """Solver-facing contract for a coupled beam section.

    Third-party section objects only need to expose a name and the stiffness
    method.  ``generalized_mass_matrix_per_length`` is optional and is queried
    by capability rather than being part of the required protocol.
    """

    name: str

    def generalized_stiffness_matrix(self) -> np.ndarray:
        """Return the local 6x6 sectional stiffness matrix."""


def _validated_matrix(
    value: Any,
    *,
    label: str,
    positive_definite: bool,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"{label} must have shape (6, 6); got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain only finite values")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise ValueError(f"{label} must have a strictly positive diagonal")
    inverse_sqrt = 1.0 / np.sqrt(diagonal)
    normalized = inverse_sqrt[:, None] * matrix * inverse_sqrt[None, :]
    if not np.allclose(
        normalized,
        normalized.T,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(f"{label} must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    normalized = 0.5 * (normalized + normalized.T)
    eigenvalues = np.linalg.eigvalsh(normalized)
    tolerance = 1.0e-12 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if positive_definite and float(eigenvalues[0]) <= tolerance:
        raise ValueError(f"{label} must be positive definite")
    result = np.array(matrix, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class GeneralizedBeamSection:
    """Homogeneous linear generalized beam section.

    Parameters
    ----------
    stiffness:
        Symmetric positive-definite 6x6 section stiffness in the orders stated
        by :data:`GENERALIZED_BEAM_STRAIN_ORDER` and
        :data:`GENERALIZED_BEAM_RESULTANT_ORDER`.
    mass_matrix:
        Optional symmetric positive-definite 6x6 inertia matrix per unit beam
        length in :data:`GENERALIZED_BEAM_VELOCITY_ORDER`.  When omitted, beam
        elements retain their historical ``density * area`` and rotary-inertia
        mass construction.  Dynamics accepts any such generalized inertia.
        Scalar mass/center-of-mass diagnostics additionally require the
        physical spatial-inertia block form
        ``[[mu*I, -skew(h)], [skew(h), J]]``, where ``h = mu*c`` is the
        sectional first moment about the beam reference axis.
    name:
        Stable descriptive name used in diagnostics and generated geometry.
    """

    stiffness: Any
    mass_matrix: Optional[Any] = None
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stiffness",
            _validated_matrix(
                self.stiffness,
                label="generalized beam-section stiffness",
                positive_definite=True,
            ),
        )
        if self.mass_matrix is not None:
            object.__setattr__(
                self,
                "mass_matrix",
                _validated_matrix(
                    self.mass_matrix,
                    label="generalized beam-section mass matrix per unit length",
                    positive_definite=True,
                ),
            )
        if not isinstance(self.name, str):
            raise TypeError("generalized beam-section name must be a string")

    def generalized_stiffness_matrix(self) -> np.ndarray:
        """Return a defensive copy of the sectional stiffness."""

        return np.array(self.stiffness, dtype=float, copy=True)

    def generalized_mass_matrix_per_length(self) -> Optional[np.ndarray]:
        """Return a defensive copy of the sectional inertia, when supplied."""

        if self.mass_matrix is None:
            return None
        return np.array(self.mass_matrix, dtype=float, copy=True)


def generalized_beam_stiffness(section: Any) -> np.ndarray:
    """Validate and return a generalized section's 6x6 stiffness."""

    method = getattr(section, "generalized_stiffness_matrix", None)
    if not callable(method):
        raise TypeError(
            "A generalized beam section must define "
            "generalized_stiffness_matrix()."
        )
    return _validated_matrix(
        method(),
        label="generalized beam-section stiffness",
        positive_definite=True,
    )


def generalized_beam_mass_matrix(section: Any) -> Optional[np.ndarray]:
    """Validate and return the optional local 6x6 inertia per unit length."""

    method = getattr(section, "generalized_mass_matrix_per_length", None)
    if not callable(method):
        return None
    value = method()
    if value is None:
        return None
    return _validated_matrix(
        value,
        label="generalized beam-section mass matrix per unit length",
        positive_definite=True,
    )


def coerce_generalized_beam_section(
    value: Any,
    *,
    name: str = "",
    mass_matrix: Any = None,
) -> GeneralizedBeamSectionContract:
    """Coerce arrays/mappings or validate an external section contract."""

    if isinstance(value, GeneralizedBeamSection):
        if mass_matrix is not None:
            raise ValueError(
                "mass_matrix cannot be supplied separately when value is "
                "already a GeneralizedBeamSection"
            )
        return value
    if isinstance(value, Mapping):
        stiffness = value.get("stiffness", value.get("generalized_stiffness"))
        if stiffness is None:
            raise ValueError(
                "Generalized beam-section mapping requires 'stiffness' or "
                "'generalized_stiffness'."
            )
        mapped_mass = value.get(
            "mass_matrix",
            value.get(
                "mass_per_length",
                value.get("generalized_mass_per_length", mass_matrix),
            ),
        )
        mapped_name = value.get("name", name)
        return GeneralizedBeamSection(
            stiffness=stiffness,
            mass_matrix=mapped_mass,
            name=str(mapped_name),
        )
    if callable(getattr(value, "generalized_stiffness_matrix", None)):
        if mass_matrix is not None:
            raise ValueError(
                "An external generalized beam section must provide its own "
                "generalized_mass_matrix_per_length() method."
            )
        if not isinstance(getattr(value, "name", None), str):
            raise TypeError(
                "An external generalized beam section must define a string "
                "name attribute."
            )
        generalized_beam_stiffness(value)
        generalized_beam_mass_matrix(value)
        return value
    return GeneralizedBeamSection(
        stiffness=value,
        mass_matrix=mass_matrix,
        name=name,
    )


def resolve_generalized_beam_section(
    cross_section: Mapping[str, Any],
    explicit_section: Any = None,
) -> Optional[GeneralizedBeamSectionContract]:
    """Resolve the opt-in generalized section from element constructor data."""

    inline_section = cross_section.get(
        "generalized_section",
        cross_section.get("beam_section"),
    )
    inline_stiffness = cross_section.get("generalized_stiffness")
    if explicit_section is not None and (
        inline_section is not None or inline_stiffness is not None
    ):
        raise ValueError(
            "Specify a generalized beam section either with section= or in "
            "cross_section, not both."
        )
    if inline_section is not None and inline_stiffness is not None:
        raise ValueError(
            "cross_section cannot contain both 'generalized_section' and "
            "'generalized_stiffness'."
        )
    value = (
        explicit_section
        if explicit_section is not None
        else inline_section
        if inline_section is not None
        else inline_stiffness
    )
    if value is None:
        return None
    mass = cross_section.get(
        "generalized_mass_matrix",
        cross_section.get(
            "mass_per_length",
            cross_section.get("generalized_mass_per_length"),
        ),
    )
    name = str(cross_section.get("section_name", ""))
    return coerce_generalized_beam_section(value, name=name, mass_matrix=mass)
