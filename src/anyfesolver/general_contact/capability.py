"""Conservative general-contact profile reporting.

Validation describes an executable development profile. This report is
descriptive, not an exact model or source-bound qualification certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import (
    ContactDefinition, ContactSolverSettings, ContactValidationReport,
    validate_general_contact,
)


@dataclass(frozen=True)
class ContactCapabilityReport:
    """Separate executable implementation from unproven evidence claims.

    Geometry/material fields summarize the supplied selection. They do not
    contain its full mesh, interactions, settings or source identity. Local
    profile-family tests in the ledger are not exact-model qualification.
    Evidence flags cannot be set through the constructor or dataclass replace.
    Applications must use ``admitted_for_application``; it remains false until
    a separate source-bound admission record is implemented and verified.
    """

    validation: ContactValidationReport
    analysis_mode: str
    surface_profiles: tuple[tuple[str, str, str], ...]
    rigid_profiles: tuple[tuple[str, str], ...]
    formulations: tuple[str, ...]
    materials: tuple[tuple[str, str], ...]
    friction: str
    lifecycle: str
    implemented: bool
    tested: bool = field(default=False, init=False)
    independently_accepted: bool = field(default=False, init=False)
    released: bool = field(default=False, init=False)

    @property
    def admitted_for_application(self) -> bool:
        return False


def report_general_contact_capability(
    model, definition: ContactDefinition,
    settings: ContactSolverSettings | None = None,
    *, analysis_control: str = "force",
) -> ContactCapabilityReport:
    """Describe a static contact selection without broadening admission.

    The validator remains the sole implementation gate. Missing element or
    material identities are reported explicitly rather than inferred from
    element node counts or substituted with another formulation.
    """

    validation = validate_general_contact(
        model, definition, settings, analysis_control=analysis_control,
    )
    if type(definition) is not ContactDefinition:
        surfaces = ()
        rigids = ()
        formulations = ()
        materials = ()
    else:
        surfaces = tuple(sorted((
            surface.name,
            "round_centreline" if surface.beam_radius is not None else "shell",
            (f"radius={surface.beam_radius:.17g};caps={surface.beam_end_caps}"
             if surface.beam_radius is not None else surface.side),
        ) for surface in definition.surfaces))
        rigids = tuple(sorted((body.name, body.kind)
                              for body in definition.rigid_bodies))
        formulation_set: set[str] = set()
        material_set: set[tuple[str, str]] = set()
        for surface in definition.surfaces:
            for element_id in surface.element_ids:
                element = model.mesh.elements.get(element_id)
                if element is None:
                    formulation_set.add(f"missing_element:{element_id}")
                    continue
                formulation_set.add(str(getattr(element, "formulation_id", "unidentified")))
                material_name = str(getattr(element, "material_name", "unidentified"))
                material = model.materials.get(material_name)
                material_set.add((material_name, type(material).__name__
                                  if material is not None else "missing_material"))
        formulations = tuple(sorted(formulation_set))
        materials = tuple(sorted(material_set))
    return ContactCapabilityReport(
        validation=validation,
        analysis_mode=f"nonlinear_static_{analysis_control}",
        surface_profiles=surfaces,
        rigid_profiles=rigids,
        formulations=formulations,
        materials=materials,
        friction="none_declared",
        lifecycle="model_validation_only",
        implemented=validation.valid,
    )
