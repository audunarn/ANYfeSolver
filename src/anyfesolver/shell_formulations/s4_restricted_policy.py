"""Cold release metadata for the dormant improved-S4 research formulation.

This module deliberately contains no formulation selector, runtime classifier,
assembly hook, constraint, or numerical operator.  It records the release-safe
interpretation of the accepted S4 evidence while the legacy shell element
remains the only production default.
"""

from __future__ import annotations

from dataclasses import dataclass


RELEASE_CONTRACT_SCHEMA = "anysolver.s4.restricted-release"
RELEASE_CONTRACT_VERSION = 1

# These are descriptive release identities, not serialization or dispatch tokens.
LEGACY_DEFAULT_ID = "anysolver.shell_element.legacy_s4"
IMPROVED_RESEARCH_ID = "mitc4_plus_d_published_2025_eq21_eq25_reference_v2"

SQUARE_NULLSPACE = (
    ("rank_B", 16),
    ("N", 8),
    ("G", 1),
    ("P", 7),
    ("R", 6),
    ("R_N", 6),
    ("R_G", 0),
    ("RQ", 6),
    ("Z", 1),
)

RESTRICTED_REASON_CODES = (
    "s4_improved.research_only",
    "s4_improved.positive_mass_zero_stiffness_z",
    "s4_improved.threshold_sensitive_geometry",
    "s4_improved.coupling_unqualified",
    "s4_improved.nonlinear_unqualified",
    "s4_improved.geometric_stiffness_unqualified",
    "s4_improved.buckling_unqualified",
    "s4_improved.recovery_unqualified",
    "s4_improved.optimized_batches_unqualified",
    "s4_improved.provenance_unavailable_or_unqualified",
)


@dataclass(frozen=True, slots=True)
class S4RestrictedReleaseStatus:
    """Immutable, passive description of the restricted release state."""

    default_formulation_id: str
    improved_research_id: str
    production_activation_available: bool
    gauge_constraint_applied: bool
    square_nullspace: tuple[tuple[str, int], ...]
    restricted_reason_codes: tuple[str, ...]


RESTRICTED_RELEASE_STATUS = S4RestrictedReleaseStatus(
    default_formulation_id=LEGACY_DEFAULT_ID,
    improved_research_id=IMPROVED_RESEARCH_ID,
    production_activation_available=False,
    gauge_constraint_applied=False,
    square_nullspace=SQUARE_NULLSPACE,
    restricted_reason_codes=RESTRICTED_REASON_CODES,
)


__all__ = [
    "IMPROVED_RESEARCH_ID",
    "LEGACY_DEFAULT_ID",
    "RELEASE_CONTRACT_SCHEMA",
    "RELEASE_CONTRACT_VERSION",
    "RESTRICTED_REASON_CODES",
    "RESTRICTED_RELEASE_STATUS",
    "S4RestrictedReleaseStatus",
    "SQUARE_NULLSPACE",
]
