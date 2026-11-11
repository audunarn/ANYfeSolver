"""Compatibility aliases for the ANYfileio SESAM document layer."""

from anyfileio.sesam.document import (
    FemBoundary,
    FemConceptRecord,
    FemCoordinate,
    FemCoordinateTransform,
    FemDependency,
    FemElement,
    FemElementReference,
    FemHeader,
    FemLoadRecord,
    FemMaterial,
    FemNode,
    FemSection,
    FemUnitVector,
    SesamFemDocument,
    parse_sesam_fem_records,
    read_sesam_fem_document,
)

__all__ = [
    "FemBoundary",
    "FemConceptRecord",
    "FemCoordinate",
    "FemCoordinateTransform",
    "FemDependency",
    "FemElement",
    "FemElementReference",
    "FemHeader",
    "FemLoadRecord",
    "FemMaterial",
    "FemNode",
    "FemSection",
    "FemUnitVector",
    "SesamFemDocument",
    "parse_sesam_fem_records",
    "read_sesam_fem_document",
]
