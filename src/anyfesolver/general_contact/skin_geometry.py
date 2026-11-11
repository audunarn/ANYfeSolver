"""Optional paired-shell skin recovery from the contact owner's work maps."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..current_state_tangent import require_exact_qualified_component_lifecycle_api as _guard
from ..fe_core import FEModel
from .core import ContactDefinition, ContactSolverSettings, validate_general_contact
from .s3_patch_static import _adapters, _selected, _unique_work_map, validate_general_s3_patch


class ContactSkinGeometryError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ContactShellSkinFace:
    surface_name: str
    role: str
    side: str
    element_id: int
    formulation_id: str
    node_ids: tuple[int, ...]
    reference_positions_m: tuple[tuple[float, float, float], ...]
    current_positions_m: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ContactShellSkinGeometry:
    interaction_name: str
    profile: str
    faces: tuple[ContactShellSkinFace, ...]

    def to_dict(self):
        return {"schema": "ANYSOLVER_CONTACT_SHELL_SKINS_V1",
                "kinematics": "reference_skin_linearized_director_work_map_v1",
                "interaction_name": self.interaction_name, "profile": self.profile,
                "faces": [asdict(face) for face in self.faces],
                "admitted_for_application": False}


def _require_orientation(reference, current):
    # At the four Q4 parametric corners these are its bilinear Jacobians.
    # The represented patch uses the same reference normal as the work owner.
    normal = np.cross(reference[1]-reference[0], reference[-1]-reference[0])
    for i in range(len(current)):
        cross = np.cross(current[(i+1) % len(current)]-current[i],
                         current[(i-1) % len(current)]-current[i])
        if not np.all(np.isfinite(cross)) or not float(cross @ normal) > 0.:
            raise ContactSkinGeometryError("invalid_geometry", "skin has degenerate or inverted orientation")


def recover_general_contact_shell_skins(model, definition, settings, displacements):
    """Recover original paired Q4/S3 faces at explicit structural coordinates.

    This is the existing small-rotation patch owner's represented skin, using
    physical reference offsets and its shared nodal work maps. No finite-rotation
    geometry, current-area traction, contact activation or admission is inferred.
    It does not certify result/model correspondence; the live solve adapter owns
    that association. Unsupported formulations and invalid poses refuse.
    """
    if (type(model) is not FEModel or type(definition) is not ContactDefinition
            or type(settings) is not ContactSolverSettings):
        raise ContactSkinGeometryError("invalid_request", "exact model, contact definitions and settings required")
    _guard(model, context="skin recovery before observation")
    if (type(displacements) is not np.ndarray or displacements.dtype.kind not in "iuf"
            or displacements.shape != (model.mesh.dof_manager.total_dofs,)
            or not np.all(np.isfinite(displacements))):
        raise ContactSkinGeometryError("invalid_displacement", "finite real authoritative displacement vector required")
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ContactSkinGeometryError("unsupported_profile", "; ".join(report.issues))
    report = validate_general_s3_patch(model, definition, settings)
    if not report.valid:
        raise ContactSkinGeometryError("unsupported_profile", "; ".join(report.issues))
    source, target, interaction = _selected(definition)
    adapters = _adapters(model, (source, target))
    faces = []
    with np.errstate(over="raise", invalid="raise"):
        for surface, role, group in zip((source, target), ("source", "target"), adapters):
            nodes = tuple(sorted({node for adapter in group for node in adapter.node_ids}))
            reference, mapping = _unique_work_map(group, nodes)
            current = reference + np.asarray(mapping @ displacements).reshape(-1, 3)
            if not np.all(np.isfinite(current)):
                raise ContactSkinGeometryError("invalid_geometry", "nonfinite physical skin coordinates")
            lookup = {node: index for index, node in enumerate(nodes)}
            for adapter in group:
                indices = [lookup[node] for node in adapter.node_ids]
                ref, pose = reference[indices], current[indices]
                _require_orientation(ref, pose)
                faces.append(ContactShellSkinFace(
                    surface.name, role, surface.side, adapter.element_id,
                    str(model.mesh.elements[adapter.element_id].formulation_id),
                    adapter.node_ids,
                    tuple(tuple(map(float, row)) for row in ref),
                    tuple(tuple(map(float, row)) for row in pose)))
    _guard(model, context="skin recovery after observation")
    return ContactShellSkinGeometry(interaction.name, report.profile, tuple(faces))


__all__ = ["ContactSkinGeometryError", "ContactShellSkinFace", "ContactShellSkinGeometry",
           "recover_general_contact_shell_skins"]
