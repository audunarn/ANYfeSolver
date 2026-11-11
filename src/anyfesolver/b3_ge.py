"""Production opt-in facade for the accepted native B3-GE workflow.

``b3-ge`` is an explicit workflow selector.  It is intentionally not an
element-factory alias: ordinary B3 construction continues to select the
legacy :class:`QuadraticBeamElement`, and historical ``ge-beam3`` records keep
their original mechanics.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from . import ge_beam3_native as _native


SELECTOR = "b3-ge"
NAME = "B3-GE"
DESCRIPTION = "B3-GE — geometrically exact Simo–Reissner beam"
ADMISSION_SCHEMA = "ANYSOLVER_B3_GE_PRODUCTION_OPT_IN_V1"
NATIVE_PROFILE_ID = _native.PROFILE_ID
NATIVE_PROFILE_SHA256 = _native.PROFILE_SHA256
G6_CONFIRMATION_FILE_SHA256 = (
    "b1e6c726f36bb5d08f6bfd407abf7de8b04866dc1958b02b3f6f63610022958f"
)
G6_CONFIRMATION_PAYLOAD_SHA256 = (
    "ac900402bfa247fc12e0ee931d0f8a724da29bc363a7ae401dc6744fea1071ff"
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


ADMISSION_BYTES = _canonical(
    {
        "default_changed": False,
        "description": DESCRIPTION,
        "g6_confirmation_file_sha256": G6_CONFIRMATION_FILE_SHA256,
        "g6_confirmation_payload_sha256": G6_CONFIRMATION_PAYLOAD_SHA256,
        "historical_ge_beam3_redirected": False,
        "legacy_b3_default": True,
        "name": NAME,
        "native_profile_id": NATIVE_PROFILE_ID,
        "native_profile_sha256": NATIVE_PROFILE_SHA256,
        "schema": ADMISSION_SCHEMA,
        "selection": "EXPLICIT_OPT_IN_ONLY",
        "selector": SELECTOR,
    }
)
ADMISSION_SHA256 = sha256(ADMISSION_BYTES).hexdigest()


def _select(formulation: str) -> None:
    if type(formulation) is not str or formulation != SELECTOR:
        raise ValueError(
            "explicit b3-ge workflow required; legacy B3 remains the default "
            "and no alias or fallback is permitted"
        )


def define_beam(
    formulation: str,
    element_id: int,
    node_ids: Any,
    coordinates: Any,
    nodal_triads: Any,
    section: Any,
    section_inertia: Any,
    *,
    quadrature: int = 4,
    arithmetic_policy: Any = None,
):
    """Create one reconstructible native definition under explicit B3-GE policy."""

    _select(formulation)
    return _native.define_beam(
        _native.SELECTOR,
        element_id,
        node_ids,
        coordinates,
        nodal_triads,
        section,
        section_inertia,
        quadrature=quadrature,
        arithmetic_policy=arithmetic_policy,
    )


def create_analysis(
    formulation: str,
    definitions: Any,
    boundaries: Any,
    *,
    retained_refinement: bool = False,
):
    """Create the accepted standalone owner; never a generic FEModel element."""

    _select(formulation)
    return _native.create_analysis(
        _native.SELECTOR,
        definitions,
        boundaries,
        retained_refinement=retained_refinement,
    )


def create_coupled_analysis(formulation: str, definitions: Any, boundaries: Any, program: Any, **kwargs: Any):
    """Create the admitted single-shell/native-beam coupled owner."""

    _select(formulation)
    return _native.create_coupled_analysis(
        _native.SELECTOR, definitions, boundaries, program, **kwargs
    )


def create_stiffened_shell_analysis(*, formulation: str, definition: Any,
                                    resources: Any = None):
    """Select the developing mixed owner explicitly, without altering legacy routes."""
    _select(formulation)
    from .stiffened_shell import create_analysis

    return create_analysis(definition=definition, resources=resources)


def workflow_provenance(analysis: Any) -> bytes:
    """Return canonical public-selection and immutable native provenance."""

    native = json.loads(_native.workflow_provenance(analysis))
    return _canonical(
        {
            "admission_schema": ADMISSION_SCHEMA,
            "admission_sha256": ADMISSION_SHA256,
            "default_changed": False,
            "legacy_b3_default": True,
            "name": NAME,
            "native": native,
            "selection": "EXPLICIT_OPT_IN_ONLY",
            "selector": SELECTOR,
        }
    )


ReferenceGeometry = _native.ReferenceGeometry
Fibre = _native.Fibre
FlowCurve = _native.FlowCurve
PhysicalFibreSection = _native.PhysicalFibreSection
EllipsoidalGeneralizedSection = _native.EllipsoidalGeneralizedSection
BeamDefinition = _native.BeamDefinition
LinePattern = _native.LinePattern
DistributedPattern = _native.DistributedPattern
DistributedProgram = _native.DistributedProgram
NodalDeadForces = _native.NodalDeadForces
NodalProgram = _native.NodalProgram
TranslationProgram = _native.TranslationProgram
FibreTranslationProgram = _native.FibreTranslationProgram
SpatialNodalMoments = _native.SpatialNodalMoments
ActivityEpochLease = _native.ActivityEpochLease
GeneralizedInitialField = _native.GeneralizedInitialField
ConsumerPolicy = _native.ConsumerPolicy
ObjectivePoseJoint = _native.ObjectivePoseJoint
evaluate_initial_field = _native.evaluate_initial_field


__all__ = [
    "SELECTOR", "NAME", "DESCRIPTION", "ADMISSION_SCHEMA", "ADMISSION_BYTES",
    "ADMISSION_SHA256", "NATIVE_PROFILE_ID", "NATIVE_PROFILE_SHA256",
    "ReferenceGeometry", "Fibre", "FlowCurve", "PhysicalFibreSection",
    "EllipsoidalGeneralizedSection", "BeamDefinition", "LinePattern",
    "DistributedPattern", "DistributedProgram", "NodalDeadForces", "NodalProgram",
    "TranslationProgram", "FibreTranslationProgram", "SpatialNodalMoments",
    "define_beam", "create_analysis", "create_coupled_analysis",
    "create_stiffened_shell_analysis",
    "workflow_provenance", "ActivityEpochLease", "GeneralizedInitialField",
    "ConsumerPolicy", "ObjectivePoseJoint", "evaluate_initial_field",
]
