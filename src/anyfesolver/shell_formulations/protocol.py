"""Internal contracts shared by scalar and compiled shell formulations.

The objects in this module contain numeric finite-element data only.  They do
not retain geometry-kernel objects, material objects, or solver state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _immutable_f64(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class Q4QualityMetrics:
    """Deterministic scale-free and dimensional Q4 validity metrics."""

    minimum_volume_jacobian: float
    maximum_volume_jacobian: float
    volume_jacobian_ratio: float
    minimum_surface_jacobian: float
    maximum_surface_jacobian: float
    surface_jacobian_ratio: float
    center_plane_condition: float
    distortion_norm: float
    maximum_director_angle_degrees: float


@dataclass(frozen=True, slots=True)
class Q4ReferenceData:
    """Immutable scalar reference data for one full MITC4+/D element.

    Quadrature arrays are deliberately operator-sized rather than matrix-sized:
    no local 24 x 24 stiffness or mass matrix is retained.  The leading
    quadrature dimensions are four surface stations and two natural-thickness
    stations.
    """

    coordinates: FloatArray
    directors: FloatArray
    thickness: FloatArray
    drill_direction: FloatArray
    center_covariant: FloatArray
    center_dual: FloatArray
    distortion_vector: FloatArray
    distortion_scalars: FloatArray
    mitc4_plus_coefficients: FloatArray
    mitc4_plus_qrs_coefficients: FloatArray
    drill_edge_coefficients: FloatArray
    surface_points: FloatArray
    surface_weights: FloatArray
    generalized_membrane_operators: FloatArray
    generalized_bending_operators: FloatArray
    generalized_shear_operators: FloatArray
    volume_points: FloatArray
    volume_weights: FloatArray
    volume_strain_operators: FloatArray
    volume_displacement_operators: FloatArray
    quality: Q4QualityMetrics
    signature: str

    def __post_init__(self) -> None:
        shapes = {
            "coordinates": (4, 3),
            "directors": (4, 3),
            "thickness": (4,),
            "drill_direction": (3,),
            "center_covariant": (2, 3),
            "center_dual": (2, 3),
            "distortion_vector": (3,),
            "distortion_scalars": (3,),
            "mitc4_plus_coefficients": (5,),
            "mitc4_plus_qrs_coefficients": (10,),
            "drill_edge_coefficients": (4, 2),
            "surface_points": (4, 2),
            "surface_weights": (4,),
            "generalized_membrane_operators": (4, 3, 24),
            "generalized_bending_operators": (4, 3, 24),
            "generalized_shear_operators": (4, 2, 24),
            "volume_points": (4, 2, 3),
            "volume_weights": (4, 2),
            "volume_strain_operators": (4, 2, 5, 24),
            "volume_displacement_operators": (4, 2, 3, 24),
        }
        for name, shape in shapes.items():
            object.__setattr__(self, name, _immutable_f64(getattr(self, name), shape, name))
        if not isinstance(self.quality, Q4QualityMetrics):
            raise TypeError("quality must be Q4QualityMetrics")
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("signature must be a non-empty string")

    @property
    def B_surface(self) -> FloatArray:
        """Batch-ready local strain operators, shape ``(4, 2, 5, 24)``."""

        return self.volume_strain_operators

    @property
    def H_surface(self) -> FloatArray:
        """Batch-ready displacement operators, shape ``(4, 2, 3, 24)``."""

        return self.volume_displacement_operators

    @property
    def weighted_volume_jacobians(self) -> FloatArray:
        """Physical quadrature weights ``det(J) * w_r * w_s * w_zeta``."""

        return self.volume_weights


@dataclass(frozen=True, slots=True)
class LinearShellSection:
    """Linear generalized shell section in engineering component order.

    ``coupling`` is the upper-right membrane-to-curvature block.  Its transpose
    is used as the lower-left block; it is not symmetrized on input.
    """

    membrane: FloatArray
    coupling: FloatArray
    bending: FloatArray
    shear: FloatArray
    mass_per_area: float = 0.0
    rotary_inertia_per_area: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "membrane", _immutable_f64(self.membrane, (3, 3), "membrane")
        )
        object.__setattr__(
            self, "coupling", _immutable_f64(self.coupling, (3, 3), "coupling")
        )
        object.__setattr__(
            self, "bending", _immutable_f64(self.bending, (3, 3), "bending")
        )
        object.__setattr__(self, "shear", _immutable_f64(self.shear, (2, 2), "shear"))
        for name in ("mass_per_area", "rotary_inertia_per_area"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @classmethod
    def isotropic(
        cls,
        youngs_modulus: float,
        poisson_ratio: float,
        thickness: float,
        *,
        density: float = 0.0,
    ) -> "LinearShellSection":
        """Create an uncorrected Reissner-Mindlin isotropic section.

        The transverse-shear block is ``G * thickness``.  No empirical shear
        or drilling factor is introduced.
        """

        youngs_modulus = float(youngs_modulus)
        poisson_ratio = float(poisson_ratio)
        thickness = float(thickness)
        density = float(density)
        if not np.isfinite(youngs_modulus) or youngs_modulus <= 0.0:
            raise ValueError("youngs_modulus must be finite and positive")
        if not np.isfinite(poisson_ratio) or not (-1.0 < poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie strictly between -1 and 0.5")
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("thickness must be finite and positive")
        if not np.isfinite(density) or density < 0.0:
            raise ValueError("density must be finite and non-negative")

        factor = youngs_modulus / (1.0 - poisson_ratio * poisson_ratio)
        plane_stress = factor * np.array(
            [
                [1.0, poisson_ratio, 0.0],
                [poisson_ratio, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - poisson_ratio)],
            ],
            dtype=np.float64,
        )
        shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
        return cls(
            membrane=thickness * plane_stress,
            coupling=np.zeros((3, 3), dtype=np.float64),
            bending=(thickness**3 / 12.0) * plane_stress,
            shear=thickness * shear_modulus * np.eye(2, dtype=np.float64),
            mass_per_area=density * thickness,
            rotary_inertia_per_area=density * thickness**3 / 12.0,
        )


@runtime_checkable
class ShellFormulation(Protocol):
    """Internal formulation dispatch boundary; dispatch occurs outside loops."""

    def build_reference_data(self, *args: Any, **kwargs: Any) -> Q4ReferenceData: ...

    def linear_stiffness(self, reference: Q4ReferenceData, *args: Any, **kwargs: Any) -> FloatArray: ...

    def consistent_mass(self, reference: Q4ReferenceData, *args: Any, **kwargs: Any) -> FloatArray: ...

    def nonlinear_response(self, reference: Q4ReferenceData, *args: Any, **kwargs: Any) -> Any: ...

    def recover_results(self, reference: Q4ReferenceData, *args: Any, **kwargs: Any) -> Any: ...

    def quality_metrics(self, reference: Q4ReferenceData) -> Q4QualityMetrics: ...

    def batch_eligibility(self, reference: Q4ReferenceData) -> bool: ...
