"""Model mass properties and rigid-body inertia diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from .beam_sections import generalized_beam_mass_matrix
from .elements import BeamElement, QuadraticBeamElement, ShellElement
from .matrix_assembly import assemble_mass_matrix

if TYPE_CHECKING:
    from .fe_core import FEModel


@dataclass
class MassProperties:
    """Integrated and assembled mass diagnostics for an FE model."""

    total_mass: float
    center_of_mass: np.ndarray
    first_moment: np.ndarray
    inertia_tensor_origin: np.ndarray
    inertia_tensor_center_of_mass: np.ndarray
    rigid_body_mass_matrix: np.ndarray
    assembled_translation_masses: Dict[str, float]
    num_mass_points: int
    skipped_elements: List[int]
    assembly_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mass": float(self.total_mass),
            "center_of_mass": self.center_of_mass.tolist(),
            "first_moment": self.first_moment.tolist(),
            "inertia_tensor_origin": self.inertia_tensor_origin.tolist(),
            "inertia_tensor_center_of_mass": self.inertia_tensor_center_of_mass.tolist(),
            "rigid_body_mass_matrix": self.rigid_body_mass_matrix.tolist(),
            "assembled_translation_masses": dict(self.assembled_translation_masses),
            "num_mass_points": int(self.num_mass_points),
            "skipped_elements": [int(eid) for eid in self.skipped_elements],
            "assembly_info": self.assembly_info,
        }


def _spatial_inertia_components(
    matrix: Any,
    *,
    label: str,
    reference_length: float | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Extract scalar mass, first moment and origin inertia from a 6x6 inertia.

    With generalized velocity order ``[v, omega]``, a physical spatial
    inertia has the block form

    ``[[m I, -[h]x], [[h]x, J_origin]]``

    where ``h = m * center_of_mass``.  An arbitrary symmetric positive-
    definite generalized inertia need not have that form: in particular, an
    anisotropic translational block has no unique scalar mass or center of
    mass.  Such matrices remain usable by element dynamics, but this scalar
    mass-properties diagnostic rejects them rather than inventing a value.
    """

    spatial = np.asarray(matrix, dtype=float)
    if spatial.shape != (6, 6):
        raise ValueError(f"{label} must have shape (6, 6); got {spatial.shape}")
    if not np.all(np.isfinite(spatial)):
        raise ValueError(f"{label} must contain only finite values")
    spatial = 0.5 * (spatial + spatial.T)

    translation = spatial[:3, :3]
    mass = float(np.trace(translation) / 3.0)
    translation_scale = max(
        float(np.max(np.abs(translation))),
        abs(mass),
        np.finfo(float).tiny,
    )
    if mass < -1.0e-10 * translation_scale:
        raise ValueError(f"{label} has a negative scalar mass")
    if abs(mass) <= 1.0e-13 * translation_scale:
        mass = 0.0
    if not np.allclose(
        translation,
        mass * np.eye(3),
        rtol=1.0e-9,
        atol=1.0e-10 * translation_scale,
    ):
        raise ValueError(
            f"{label} is not a physical spatial inertia: its translational "
            "3x3 block must equal scalar mass times the identity. An "
            "anisotropic generalized translational inertia has no unique "
            "total_mass or center_of_mass."
        )

    translation_rotation = spatial[:3, 3:]
    coupling_scale = max(
        float(np.max(np.abs(translation_rotation))),
        np.finfo(float).tiny,
    )
    if reference_length is not None:
        # Near the COM the whole coupling block cancels. Measure projection
        # roundoff against mass*length using the same relative tolerance.
        # Existing callers without an explicit length retain their behavior.
        coupling_scale = max(coupling_scale, abs(mass) * reference_length)
    if not np.allclose(
        translation_rotation,
        -translation_rotation.T,
        rtol=1.0e-9,
        atol=1.0e-10 * coupling_scale,
    ):
        raise ValueError(
            f"{label} is not a physical spatial inertia: its "
            "translation-rotation block must be skew-symmetric so it "
            "defines a unique center_of_mass."
        )
    coupling = 0.5 * (
        translation_rotation - translation_rotation.T
    )
    first_moment = np.array(
        (coupling[1, 2], coupling[2, 0], coupling[0, 1]),
        dtype=float,
    )
    if mass == 0.0 and not np.allclose(
        first_moment,
        0.0,
        rtol=0.0,
        atol=1.0e-12 * max(coupling_scale, 1.0),
    ):
        raise ValueError(
            f"{label} has zero scalar mass but a non-zero first moment"
        )
    return mass, first_moment, np.array(spatial[3:, 3:], copy=True)


def _rigid_body_modes(model: "FEModel", reference: np.ndarray) -> np.ndarray:
    total_dofs = model.mesh.dof_manager.total_dofs
    modes = np.zeros((total_dofs, 6), dtype=float)
    for node in model.mesh.nodes.values():
        x, y, z = node.coords() - reference
        ux, uy, uz, rx, ry, rz = node.dofs[:6]
        modes[ux, 0] = 1.0
        modes[uy, 1] = 1.0
        modes[uz, 2] = 1.0

        modes[uy, 3] = -z
        modes[uz, 3] = y
        modes[rx, 3] = 1.0

        modes[ux, 4] = z
        modes[uz, 4] = -x
        modes[ry, 4] = 1.0

        modes[ux, 5] = -y
        modes[uy, 5] = x
        modes[rz, 5] = 1.0
    return modes


def _project_rigid_inertia(
    model: "FEModel", matrix: Any, reference: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project an already assembled mass without another element observation."""
    modes = _rigid_body_modes(model, reference)
    rigid_mass = np.asarray(modes.T @ (matrix @ modes), dtype=float)
    return modes, rigid_mass


def _validated_point_masses(model: "FEModel") -> List[Tuple[Any, float]]:
    """Validate public dictionary entries as well as add_point_mass inputs."""
    entries = []
    for node_id, mass in getattr(model.mesh, "point_masses", {}).items():
        node = model.mesh.get_node(int(node_id))
        if node is None:
            raise ValueError(f"Point mass references missing node {node_id}")
        mass_value = float(mass)
        if not np.isfinite(mass_value) or mass_value < 0.0:
            raise ValueError(f"Point mass at node {node_id} must be finite and non-negative")
        entries.append((node, mass_value))
    return entries


def _shell_mass_points(model: "FEModel", element: ShellElement) -> List[Tuple[float, np.ndarray]]:
    material = model.get_material(element.material_name)
    coords = element.get_node_coordinates(model.mesh)
    mass_per_area = (
        float(element.shell_section.mass_per_area)
        if element.shell_section is not None
        and element.shell_section.mass_per_area is not None
        else float(material.density) * float(element.thickness)
    )
    points: List[Tuple[float, np.ndarray]] = []
    for (xi, eta), weight in zip(element.gauss_points, element.gauss_weights):
        N, dN_dxi, dN_deta = element.compute_shape_functions(float(xi), float(eta))
        _, _, _, det_j = element._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
        mass = mass_per_area * float(det_j) * float(weight)
        points.append((mass, np.asarray(N @ coords, dtype=float)))
    return points


def _beam_mass_points(model: "FEModel", element: BeamElement) -> List[Tuple[float, np.ndarray]]:
    material = model.get_material(element.material_name)
    coords = element.get_node_coordinates(model.mesh)
    length, transform = element._beam_frame_and_transform(coords)
    section_mass = (
        None
        if element.generalized_section is None
        else generalized_beam_mass_matrix(element.generalized_section)
    )
    if section_mass is not None:
        line_mass, local_first_moment, _ = _spatial_inertia_components(
            section_mass,
            label=(
                f"Generalized beam-section mass matrix on element "
                f"{element.element_id}"
            ),
        )
        local_offset = (
            local_first_moment / line_mass
            if line_mass > 0.0
            else np.zeros(3, dtype=float)
        )
        # The element transform maps global vectors to beam-local vectors.
        # Its transpose therefore maps the sectional COM offset back to the
        # global frame.
        global_offset = transform[:3, :3].T @ local_offset
    else:
        line_mass = float(material.density) * float(element._A)
        global_offset = np.zeros(3, dtype=float)
    if isinstance(element, QuadraticBeamElement):
        points = []
        for xi, weight in zip(element.GAUSS_POINTS, element.GAUSS_WEIGHTS):
            N, _ = element.compute_shape_functions(float(xi))
            mass = line_mass * length / 2.0 * float(weight)
            points.append(
                (
                    mass,
                    np.asarray(N @ coords, dtype=float) + global_offset,
                )
            )
        return points
    xi_values = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    points = []
    for xi in xi_values:
        N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0], dtype=float)
        mass = line_mass * length / 2.0
        points.append(
            (
                mass,
                np.asarray(N @ coords, dtype=float) + global_offset,
            )
        )
    return points


def element_mass_points(model: "FEModel") -> Tuple[List[Tuple[float, np.ndarray]], List[int]]:
    """Return scalar quadrature mass points for physically interpretable elements.

    Generalized shell ``mass_per_area`` overrides material density.  A
    generalized beam spatial inertia supplies both line mass and its
    center-of-mass offset.  Elements whose generalized inertia has no scalar
    mass interpretation are listed in ``skipped``.
    """
    points: List[Tuple[float, np.ndarray]] = []
    skipped: List[int] = []
    for elem_id, element in model.mesh.elements.items():
        try:
            if isinstance(element, ShellElement):
                points.extend(_shell_mass_points(model, element))
            elif isinstance(element, BeamElement):
                points.extend(_beam_mass_points(model, element))
            else:
                skipped.append(int(elem_id))
        except Exception:
            skipped.append(int(elem_id))
    return points, skipped


def calculate_mass_properties(model: "FEModel", reference_point: Any = None) -> MassProperties:
    """Calculate integrated mass properties and assembled rigid-body mass.

    Scalar mass, first moment and inertia tensors are extracted from rigid-body
    projections of the assembled mass matrix.  Consequently they use exactly
    the same shell rotary inertia, generalized beam sectional inertia,
    orientations, and point masses as dynamics.  Element quadrature remains
    available through :func:`element_mass_points` for integration diagnostics
    and supplies ``num_mass_points``.

    A coupled generalized beam mass matrix must have physical spatial-inertia
    blocks to admit scalar mass properties.  Arbitrary positive-definite
    generalized inertia remains valid for dynamics, but this routine raises
    when its translational block is anisotropic or its translation-rotation
    block cannot define a unique center of mass.
    """
    points, skipped = element_mass_points(model)
    for node, mass_value in _validated_point_masses(model):
        if mass_value > 0.0:
            points.append((mass_value, node.coords()))

    from .concentrated_mass import entries
    points.extend((v.mass, np.asarray(n.coords()) + v.offset) for n, v in entries(model.mesh) if v.mass > 0.)
    M, assembly_info = assemble_mass_matrix(model)
    origin = np.zeros(3, dtype=float)
    origin_modes, rbm_origin = _project_rigid_inertia(model, M, origin)
    total_mass, first_moment, inertia_origin = (
        _spatial_inertia_components(
            rbm_origin,
            label="Assembled rigid-body mass matrix",
        )
    )
    center = (
        first_moment / total_mass
        if total_mass > 0.0
        else np.zeros(3, dtype=float)
    )
    _, rbm_center = _project_rigid_inertia(model, M, center)
    inertia_center = np.array(
        0.5 * (rbm_center[3:, 3:] + rbm_center[3:, 3:].T),
        dtype=float,
    )

    if reference_point is None:
        rbm = rbm_center
    else:
        reference = np.asarray(reference_point, dtype=float).reshape(3)
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference_point must contain only finite values")
        _, rbm = _project_rigid_inertia(model, M, reference)
    tx = origin_modes[:, 0]
    ty = origin_modes[:, 1]
    tz = origin_modes[:, 2]
    assembled_translation_masses = {
        "x": float(tx @ (M @ tx)),
        "y": float(ty @ (M @ ty)),
        "z": float(tz @ (M @ tz)),
    }

    return MassProperties(
        total_mass=total_mass,
        center_of_mass=center,
        first_moment=first_moment,
        inertia_tensor_origin=inertia_origin,
        inertia_tensor_center_of_mass=inertia_center,
        rigid_body_mass_matrix=rbm,
        assembled_translation_masses=assembled_translation_masses,
        num_mass_points=len(points),
        skipped_elements=skipped,
        assembly_info=assembly_info,
    )
