"""Compatibility aliases for the ANYfileio SESAM schema registry."""

from anyfileio.sesam.schema import (
    BOUNDARY_LOAD_RECORDS,
    CONCEPT_RECORDS,
    DOF_NAMES,
    GEOMETRY_RECORDS,
    KNOWN_UNSUPPORTED_STRUCTURAL_RECORDS,
    MATERIAL_SECTION_RECORDS,
    METADATA_RECORDS,
    SESAM_ELEMENT_REGISTRY,
    SUPPORTED_RECORDS,
    TEXT_RECORDS,
    TRANSFORM_RECORDS,
    SesamElementSpec,
    classify_record,
    get_element_spec,
)

__all__ = [
    "BOUNDARY_LOAD_RECORDS",
    "CONCEPT_RECORDS",
    "DOF_NAMES",
    "GEOMETRY_RECORDS",
    "KNOWN_UNSUPPORTED_STRUCTURAL_RECORDS",
    "MATERIAL_SECTION_RECORDS",
    "METADATA_RECORDS",
    "SESAM_ELEMENT_REGISTRY",
    "SUPPORTED_RECORDS",
    "TEXT_RECORDS",
    "TRANSFORM_RECORDS",
    "SesamElementSpec",
    "classify_record",
    "get_element_spec",
]
