"""Solver-owned generalized shell and beam section contracts.

The protocols in this module are structural: future section objects from
ANYmaterial or another repository can satisfy them without importing or
subclassing ANYfeSolver.  The concrete dataclasses are convenience containers,
not an inheritance requirement.
"""

from __future__ import annotations

from typing import Any

from .beam_sections import (
    GENERALIZED_BEAM_RESULTANT_ORDER,
    GENERALIZED_BEAM_STRAIN_ORDER,
    GENERALIZED_BEAM_VELOCITY_ORDER,
    GeneralizedBeamSection,
    GeneralizedBeamSectionContract,
    coerce_generalized_beam_section,
    generalized_beam_mass_matrix,
    generalized_beam_stiffness,
    resolve_generalized_beam_section,
)
from .shell_sections import (
    SHELL_MEMBRANE_VOIGT_ORDER,
    SHELL_TRANSVERSE_SHEAR_ORDER,
    GeneralizedShellSection,
    GeneralizedShellSectionProtocol,
    coerce_generalized_shell_section,
    validate_generalized_shell_section,
)


def validate_generalized_beam_section(
    section: Any,
) -> GeneralizedBeamSectionContract:
    """Validate and return a generalized beam-section contract.

    External objects are retained after their stiffness and optional
    mass-per-length matrices have been qualified. Arrays and mappings are
    converted to the solver-owned convenience dataclass.
    """

    validated = coerce_generalized_beam_section(section)
    generalized_beam_stiffness(validated)
    generalized_beam_mass_matrix(validated)
    return validated


__all__ = [
    "GENERALIZED_BEAM_RESULTANT_ORDER",
    "GENERALIZED_BEAM_STRAIN_ORDER",
    "GENERALIZED_BEAM_VELOCITY_ORDER",
    "GeneralizedBeamSection",
    "GeneralizedBeamSectionContract",
    "GeneralizedShellSection",
    "GeneralizedShellSectionProtocol",
    "SHELL_MEMBRANE_VOIGT_ORDER",
    "SHELL_TRANSVERSE_SHEAR_ORDER",
    "coerce_generalized_beam_section",
    "coerce_generalized_shell_section",
    "generalized_beam_mass_matrix",
    "generalized_beam_stiffness",
    "resolve_generalized_beam_section",
    "validate_generalized_beam_section",
    "validate_generalized_shell_section",
]
