"""Generalized initial-field offsets for the qualified S3 companion.

The public :class:`~anyfesolver.nonlinear_static.ShellInitialField` convention
describes a uniform membrane stress, an antisymmetric *surface* bending
stress, and membrane/curvature eigenstrains in the numbered shell frame.  A
pre-integrated generalized section has no layers from which those inputs can
be reconstructed.  This module therefore performs only the analytic
through-thickness integration that is defined by that public convention; it
never creates ply or material-point data.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


GENERALIZED_INITIAL_FIELD_POLICY_ID = (
    "SHELL_INITIAL_FIELD_ANALYTIC_N_H_SIGMA_M_M_H2_OVER6_SIGMA_B_V1"
)
GENERALIZED_INITIAL_FIELD_COMPONENTS = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
)


def integrate_generalized_initial_fields(
    fields: Mapping[str, Any],
    thickness: Any,
    *,
    station_count: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return generalized eigenstrain and initial-resultant station rows.

    Inputs must already be normalized to one ``(station_count, 3)`` array per
    public shell field.  The returned eight-component rows use the qualified
    S3 generalized strain/resultant ordering.  Transverse-shear offsets are
    exact zero because ``ShellInitialField`` declares no transverse-shear
    stress or eigenstrain component.
    """

    if not isinstance(fields, Mapping):
        raise TypeError("qualified S3 generalized initial fields must be a mapping")
    if isinstance(station_count, (bool, np.bool_)) or not isinstance(
        station_count, (int, np.integer)
    ):
        raise TypeError("station_count must be an integer")
    count = int(station_count)
    if count <= 0:
        raise ValueError("station_count must be positive")
    if isinstance(thickness, (bool, np.bool_)) or not isinstance(
        thickness, (int, float, np.integer, np.floating)
    ):
        raise TypeError("thickness must be a finite positive scalar")
    made_thickness = float(thickness)
    if not math.isfinite(made_thickness) or made_thickness <= 0.0:
        raise ValueError("thickness must be a finite positive scalar")

    arrays: dict[str, np.ndarray] = {}
    missing = set(GENERALIZED_INITIAL_FIELD_COMPONENTS) - set(fields)
    unknown = set(fields) - set(GENERALIZED_INITIAL_FIELD_COMPONENTS)
    if missing or unknown:
        raise ValueError(
            "qualified S3 generalized initial-field keys mismatch; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for name in GENERALIZED_INITIAL_FIELD_COMPONENTS:
        try:
            value = np.asarray(fields[name], dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values") from exc
        if value.shape != (count, 3):
            raise ValueError(f"{name} must have shape ({count}, 3)")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite values")
        arrays[name] = np.array(value, dtype=np.float64, order="C", copy=True)

    eigenstrain = np.zeros((count, 8), dtype=np.float64)
    eigenstrain[:, :3] = arrays["initial_membrane_prestrain"]
    eigenstrain[:, 3:6] = arrays["initial_curvature_prestrain"]

    resultant = np.zeros((count, 8), dtype=np.float64)
    resultant[:, :3] = (
        made_thickness * arrays["initial_membrane_stress"]
    )
    resultant[:, 3:6] = (
        (made_thickness * made_thickness / 6.0)
        * arrays["initial_bending_stress"]
    )
    return eigenstrain, resultant


__all__ = [
    "GENERALIZED_INITIAL_FIELD_COMPONENTS",
    "GENERALIZED_INITIAL_FIELD_POLICY_ID",
    "integrate_generalized_initial_fields",
]
