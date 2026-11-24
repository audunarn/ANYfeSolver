"""Report where default beam local axes switch on orientation-sensitive sections.

``BeamElement`` (and ``QuadraticBeamElement``) without ``cross_section["orientation"]``
take their local axes from a fixed trial vector: local y is global Y projected
perpendicular to the member axis, unless ``|e1 . Y| > 0.95``, in which case global Z
is used instead.  Along a member line that passes near global Y this can turn the
section by 90 degrees between neighbouring elements.  For a section whose
stiffness differs between its local y and z axes, the result then depends on
that switch rather than on the intended section orientation (for example the
MacNeal-Harder curved beam out of plane: -21.7 % without ``orientation``).

:func:`check_default_beam_axes` only inspects a model.  It changes no element,
result or default.  Pass ``orientation`` on the reported elements to fix their
axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .beam_sections import resolve_generalized_beam_section
from .elements import BeamElement, _beam_material_properties, _beam_rotation_matrix

__all__ = [
    "DEFAULT_AXIS_SWITCH_THRESHOLD",
    "BeamAxisRecord",
    "DefaultBeamAxesReport",
    "check_default_beam_axes",
    "section_is_orientation_sensitive",
]

#: ``|e1 . global Y|`` above which the default rule uses global Z as trial vector.
DEFAULT_AXIS_SWITCH_THRESHOLD = 0.95

@dataclass(frozen=True)
class BeamAxisRecord:
    """Default-axis data of one beam element without ``orientation``."""

    element_id: int
    node_ids: Tuple[int, ...]
    axis: Tuple[float, float, float]
    alignment_with_global_y: float
    trial_vector: str  # "global_Y" or "global_Z"
    local_z: Tuple[float, float, float]
    orientation_sensitive: bool
    sensitivity_reason: str


@dataclass(frozen=True)
class DefaultBeamAxesReport:
    """Result of :func:`check_default_beam_axes`.

    ``switch_nodes`` maps a node id to the element ids that meet there with
    orientation-sensitive sections and different default trial vectors.  A
    trial-vector switch makes the local axes depend on which side of the
    threshold each element lies; it is a turn of the section only when the
    neighbouring local axes actually differ (``local_z`` per record shows it).
    ``z_trial_elements`` lists orientation-sensitive elements that use the
    global-Z trial; that alone is a consistent convention (for example a
    straight member along global Y) and is information, not a finding.
    """

    records: Tuple[BeamAxisRecord, ...]
    switch_nodes: Mapping[int, Tuple[int, ...]]
    z_trial_elements: Tuple[int, ...]
    oriented_elements: Tuple[int, ...]
    other_elements: Tuple[int, ...] = field(default=())

    @property
    def has_findings(self) -> bool:
        return bool(self.switch_nodes)


def _pair_difference(pairs) -> Tuple[bool, str]:
    for name, y_value, z_value in pairs:
        if y_value is None and z_value is None:
            continue
        if y_value is None or z_value is None:
            return True, f"only one of {name} given"
        y_value, z_value = float(y_value), float(z_value)
        if abs(y_value - z_value) > 1.0e-12 * max(abs(y_value), abs(z_value)):
            return True, f"{name}: {y_value:.6g} != {z_value:.6g}"
    return False, "y and z section values agree"


def section_is_orientation_sensitive(cross_section: Mapping[str, Any]) -> Tuple[bool, str]:
    """Return ``(sensitive, reason)`` for a ``BeamElement`` cross-section dict alone.

    Fails closed: a generalized section (resolved with the solver's own
    precedence), fibre plasticity or an unresolvable section counts as
    sensitive, because its symmetry is not checked here.  A scalar section is
    insensitive only when its y and z values agree (Iy/Iz, shear factors, and
    c_y/c_z when given) to a relative 1e-12.  The material is not seen here;
    :func:`check_default_beam_axes` also checks the element's resolved section
    and its material's G12/G13.
    """

    try:
        generalized = resolve_generalized_beam_section(cross_section)
    except Exception as error:  # conflicting or invalid section data
        return True, f"section not resolvable ({error})"
    if generalized is not None:
        return True, "generalized section (symmetry not checked)"
    if cross_section.get("fiber_plasticity") is not None:
        return True, "fiber_plasticity (symmetry not checked)"
    return _pair_difference(
        (
            ("Iy/Iz", cross_section.get("Iy", 1.0e-8), cross_section.get("Iz", 1.0e-8)),
            ("shear_factor_y/z", cross_section.get("shear_factor_y", 5.0 / 6.0),
             cross_section.get("shear_factor_z", 5.0 / 6.0)),
            ("c_y/c_z", cross_section.get("c_y"), cross_section.get("c_z")),
        )
    )


def _element_sensitivity(element: BeamElement, material: Any) -> Tuple[bool, str]:
    """Sensitivity from the element's resolved state, as the solver uses it."""

    if getattr(element, "generalized_section", None) is not None:
        return True, "generalized section (symmetry not checked)"
    if getattr(element, "_fiber_plasticity", None) is not None:
        return True, "fiber_plasticity (symmetry not checked)"
    sensitive, reason = _pair_difference(
        (
            ("Iy/Iz", element._Iy, element._Iz),
            ("shear_factor_y/z", element._ky, element._kz),
            ("c_y/c_z", element._c_y, element._c_z),
        )
    )
    if sensitive:
        return sensitive, reason
    if material is None:
        return True, f"material {element.material_name!r} not found (not checked)"
    try:
        _E, g12, g13, _gj = _beam_material_properties(material, element.cross_section)
    except Exception as error:
        return True, f"material not checked ({error})"
    return _pair_difference((("G12/G13", g12, g13),))


def _element_axis(element: BeamElement, mesh: Any) -> np.ndarray:
    coords = np.asarray(element.get_node_coordinates(mesh), dtype=float)
    chord = coords[-1] - coords[0]
    length = float(np.linalg.norm(chord))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError(f"beam element {element.element_id} has zero or non-finite length")
    return chord / length


def check_default_beam_axes(model: Any) -> DefaultBeamAxesReport:
    """Report beam elements whose default local axes depend on the global-Y switch.

    Inspects every ``BeamElement`` (including ``QuadraticBeamElement``) in
    ``model``; other element types are ignored.  It reads the state each
    element resolved at construction (orientation, section) and its material's
    G12/G13, not the possibly edited ``cross_section`` dict.  Elements whose
    orientation the solver uses are listed in ``oriented_elements`` and not checked further; an orientation
    (nearly) parallel to the member axis falls back to the default rule and is
    checked like a missing one.  Read only.
    """

    mesh = model.mesh
    records: List[BeamAxisRecord] = []
    oriented: List[int] = []
    other: List[int] = []
    for element_id in sorted(mesh.elements):
        element = mesh.elements[element_id]
        if not isinstance(element, BeamElement):
            other.append(int(element_id))
            continue
        e1 = _element_axis(element, mesh)
        # The orientation stored at construction is the one the solver uses.
        orientation = getattr(element, "_orientation", None)
        if orientation is not None:
            # Same admission as elements._beam_rotation_matrix: an orientation
            # (nearly) parallel to the axis falls back to the default rule.
            candidate = orientation - np.dot(orientation, e1) * e1
            if float(np.linalg.norm(candidate)) > 1.0e-6 * float(np.linalg.norm(orientation)):
                oriented.append(int(element_id))
                continue
        alignment = abs(float(e1[1]))
        trial = "global_Z" if alignment > DEFAULT_AXIS_SWITCH_THRESHOLD else "global_Y"
        local_z = _beam_rotation_matrix(e1, None)[:, 2]
        sensitive, reason = _element_sensitivity(
            element, model.materials.get(element.material_name)
        )
        if orientation is not None:
            reason += "; orientation parallel to the axis, default rule used"
        records.append(
            BeamAxisRecord(
                element_id=int(element_id),
                node_ids=tuple(int(n) for n in element.node_ids),
                axis=tuple(float(v) for v in e1),
                alignment_with_global_y=alignment,
                trial_vector=trial,
                local_z=tuple(float(v) for v in local_z),
                orientation_sensitive=sensitive,
                sensitivity_reason=reason,
            )
        )

    by_node: Dict[int, List[BeamAxisRecord]] = {}
    for record in records:
        if record.orientation_sensitive:
            for node_id in (record.node_ids[0], record.node_ids[-1]):
                by_node.setdefault(node_id, []).append(record)
    switch_nodes = {
        node_id: tuple(sorted(r.element_id for r in members))
        for node_id, members in sorted(by_node.items())
        if len({r.trial_vector for r in members}) > 1
    }
    z_trial = tuple(
        r.element_id for r in records if r.orientation_sensitive and r.trial_vector == "global_Z"
    )
    return DefaultBeamAxesReport(
        records=tuple(records),
        switch_nodes=switch_nodes,
        z_trial_elements=z_trial,
        oriented_elements=tuple(oriented),
        other_elements=tuple(other),
    )
