"""Shared initial-field state predicates for nonlinear assembly paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


INITIAL_FIELD_STATE_KEYS = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
    "initial_fiber_stress",
    "initial_fiber_prestrain",
)


def state_has_active_initial_fields(
    state: Any,
    field_keys: Sequence[str] = INITIAL_FIELD_STATE_KEYS,
) -> bool:
    """Return whether state contains an active or malformed initial field.

    Qualified S3 committed state self-describes all four shell fields, even
    when they are identically zero.  Presence alone therefore cannot select a
    scalar initial-field path.  Explicit provenance remains active even for a
    zero field, while malformed values conservatively count as active so they
    reach the strict validator instead of a fast path.
    """

    if not isinstance(state, Mapping):
        return False
    if "initial_field_provenance" in state:
        provenance = state["initial_field_provenance"]
        if not isinstance(provenance, Mapping) or bool(provenance):
            return True
    for key in field_keys:
        if key not in state:
            continue
        try:
            values = np.asarray(state[key], dtype=float)
        except (TypeError, ValueError, OverflowError):
            return True
        if values.size == 0 or np.any(~np.isfinite(values)):
            return True
        if np.any(values != 0.0):
            return True
    return False


__all__ = ["INITIAL_FIELD_STATE_KEYS", "state_has_active_initial_fields"]
