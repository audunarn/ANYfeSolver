"""Compatibility imports for the extracted :mod:`anymaterial` package.

Material contracts, elastic reductions and yield criteria are owned by
ANYmaterial as of ANYfeSolver 0.2.  This module remains for the 0.2.x line so
existing ``anyfesolver.materials`` imports keep resolving to the canonical
objects rather than to duplicated implementations.
"""

from anymaterial import (
    BeamMaterialProperties,
    ENGINEERING_VOIGT_ORDER,
    SUPPORTED_ELASTIC_SYMMETRIES,
    Hill48Yield,
    OrthotropicMaterial,
    StructuralMaterial,
    beam_material_properties,
    elastic_compliance_matrix,
    is_isotropic_material,
    is_orthotropic_material,
    material_symmetry,
    material_validation_errors,
    shell_characteristic_modulus,
    shell_material_matrices,
    validate_material,
)

__all__ = [
    "BeamMaterialProperties",
    "ENGINEERING_VOIGT_ORDER",
    "SUPPORTED_ELASTIC_SYMMETRIES",
    "Hill48Yield",
    "OrthotropicMaterial",
    "StructuralMaterial",
    "beam_material_properties",
    "elastic_compliance_matrix",
    "is_isotropic_material",
    "is_orthotropic_material",
    "material_symmetry",
    "material_validation_errors",
    "shell_characteristic_modulus",
    "shell_material_matrices",
    "validate_material",
]
