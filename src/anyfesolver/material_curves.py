"""Compatibility facade for material curves now owned by ANYmaterial.

Only :class:`FiberSectionPlasticityConfig` is solver-owned: it configures the
beam formulation's integration grid, not material behaviour.  The remaining
names are canonical ANYmaterial objects re-exported through the 0.2.x line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anymaterial import (
    DNVC208MaterialCurve,
    HardeningCurve,
    LinearHardeningCurve,
    PiecewiseLinearCurve,
    PowerLawHardeningCurve,
    curve_from_properties,
    dnv_c208_steel_curve,
    dnv_c208_steel_properties,
)


@dataclass(frozen=True)
class FiberSectionPlasticityConfig:
    """Opt-in beam fiber integration configuration.

    The equivalent rectangular grid is a numerical choice made by the solver;
    the curve placed on that grid is supplied by ANYmaterial.
    """

    num_y: int = 5
    num_z: int = 5
    material_curve: Optional[HardeningCurve] = None

    def __post_init__(self) -> None:
        if self.num_y <= 0 or self.num_z <= 0:
            raise ValueError("num_y and num_z must be positive")


__all__ = [
    "DNVC208MaterialCurve",
    "FiberSectionPlasticityConfig",
    "HardeningCurve",
    "LinearHardeningCurve",
    "PiecewiseLinearCurve",
    "PowerLawHardeningCurve",
    "curve_from_properties",
    "dnv_c208_steel_curve",
    "dnv_c208_steel_properties",
]
