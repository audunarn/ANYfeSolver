"""Normalized generated-geometry FEM mode.

The functions in this module consume geometry that a client has already
generated.  They do not parse IFC or external FEM files.  The workflow is:

1. normalize generated geometry and idealize tagged member plates as beams;
2. convert generated nodes/shells/beams into ``FEModel``;
3. generate symmetric design loads from normalized client inputs;
4. solve the full static model;
5. recover prestress from the static result;
6. solve linear buckling from that recovered prestress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .assembly import compute_stresses, solve_linear
from .beam_sections import (
    GeneralizedBeamSectionContract,
    coerce_generalized_beam_section,
)
from .boundary import BoundaryCondition, LoadCase
from .buckling import BucklingResult, solve_eigenvalue_buckling
from .elements import (
    BeamElement,
    QuadraticBeamElement,
    ShellElement,
    _normalized_shell_formulation,
    create_shell_element,
)
from .fe_core import FEModel
from .mesh_gen import InterpolatedBeamShellMPCElement, RigidLidMPCElement
from .shell_sections import (
    GeneralizedShellSection,
    coerce_generalized_shell_section,
)
from .validation import LoadResultant, load_case_resultant


@dataclass(frozen=True)
class AnyStructureFEMConfig:
    """Configuration for the generated-geometry full FEM workflow."""

    geometry_scale: float = 1.0
    include_shells: bool = True
    include_beams: bool = True
    idealize_stiffeners_as_beams: bool = True
    idealize_girders_as_beams: bool = True
    auto_idealize_member_plates_as_beams: bool = True
    exclude_idealized_member_plates: bool = True
    require_idealized_member_beams: bool = True
    default_material: str = "steel"
    elastic_modulus: float = 210.0e9
    poisson_ratio: float = 0.3
    density: float = 7850.0
    yield_stress: float = 355.0e6
    pressure_pa: Optional[float] = None
    pressure_sign: float = 1.0
    load_scale: float = 1.0
    num_buckling_modes: int = 5
    solver_type: str = "direct"
    stress_percentile: float = 95.0
    add_inplane_edge_loads: bool = True


@dataclass
class AnyStructureFEMResult:
    """Result from the generated-geometry FEM mode."""

    valid: bool
    status: str
    invalid_reason: Optional[str] = None
    static_solver_status: str = "not_run"
    buckling_solver_status: str = "not_run"
    critical_load_factor: Optional[float] = None
    buckling_load_factors: List[float] = field(default_factory=list)
    max_displacement: float = 0.0
    max_translation: float = 0.0
    stress_max: float = 0.0
    stress_percentile: float = 0.0
    node_count: int = 0
    element_count: int = 0
    shell_element_count: int = 0
    beam_element_count: int = 0
    prestress_summary: Dict[str, Any] = field(default_factory=dict)
    load_resultant: Optional[LoadResultant] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    displacements: Optional[np.ndarray] = None
    buckling_result: Optional[BucklingResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "invalid_reason": self.invalid_reason,
            "static_solver_status": self.static_solver_status,
            "buckling_solver_status": self.buckling_solver_status,
            "critical_load_factor": self.critical_load_factor,
            "buckling_load_factors": self.buckling_load_factors,
            "max_displacement": self.max_displacement,
            "max_translation": self.max_translation,
            "stress_max": self.stress_max,
            "stress_percentile": self.stress_percentile,
            "node_count": self.node_count,
            "element_count": self.element_count,
            "shell_element_count": self.shell_element_count,
            "beam_element_count": self.beam_element_count,
            "prestress_summary": self.prestress_summary,
            "load_resultant": None if self.load_resultant is None else self.load_resultant.to_dict(),
            "diagnostics": self.diagnostics,
        }


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _collection(obj: Any, *names: str) -> List[Any]:
    value = _value(obj, *names, default=[])
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [{"id": key, **val} if isinstance(val, Mapping) else {"id": key, "value": val} for key, val in value.items()]
    return list(value)


def _collection_by_name(obj: Any, name: str) -> List[Any]:
    if obj is None:
        return []
    missing = object()
    if isinstance(obj, Mapping):
        value = obj.get(name, missing)
    else:
        value = getattr(obj, name, missing)
    if value is missing or value is None:
        return []
    if isinstance(value, Mapping):
        return [{"id": key, **val} if isinstance(val, Mapping) else {"id": key, "value": val} for key, val in value.items()]
    return list(value)


def _combined_collections(obj: Any, *names: str) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    for name in names:
        items.extend((name, item) for item in _collection_by_name(obj, name))
    return items


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coords_from_node(node: Any, scale: float) -> Tuple[int, float, float, float]:
    if isinstance(node, Mapping):
        node_id = int(_value(node, "id", "node_id"))
        if "value" in node:
            coords = np.asarray(node["value"], dtype=float).reshape(-1)
            if coords.size >= 3:
                return node_id, float(coords[0]) * scale, float(coords[1]) * scale, float(coords[2]) * scale
        if "coords" in node:
            coords = np.asarray(node["coords"], dtype=float).reshape(-1)
            return node_id, float(coords[0]) * scale, float(coords[1]) * scale, float(coords[2]) * scale
        return (
            node_id,
            _as_float(_value(node, "x")) * scale,
            _as_float(_value(node, "y")) * scale,
            _as_float(_value(node, "z")) * scale,
        )
    if isinstance(node, (list, tuple)) and len(node) >= 4:
        return int(node[0]), float(node[1]) * scale, float(node[2]) * scale, float(node[3]) * scale
    node_id = int(_value(node, "id", "node_id"))
    return node_id, _as_float(_value(node, "x")) * scale, _as_float(_value(node, "y")) * scale, _as_float(_value(node, "z")) * scale


def _node_ids(item: Any) -> List[int]:
    ids = _value(item, "node_ids", "nodes", "connectivity", default=[])
    return [int(node_id) for node_id in ids]


def _cross_section(item: Any) -> Dict[str, Any]:
    section = _value(item, "cross_section", "section", default=None)
    if isinstance(section, Mapping):
        source = section
    else:
        source = item
    section = {
        "area": _as_float(_value(source, "area", "A"), 0.01),
        "Iy": _as_float(_value(source, "Iy", "iy"), 1.0e-8),
        "Iz": _as_float(_value(source, "Iz", "iz"), 1.0e-8),
        "J": _as_float(_value(source, "J", "torsion_constant"), 1.0e-8),
        "shear_factor_y": _as_float(_value(source, "shear_factor_y", "ky"), 5.0 / 6.0),
        "shear_factor_z": _as_float(_value(source, "shear_factor_z", "kz"), 5.0 / 6.0),
    }
    orientation = _value(source, "orientation", "section_orientation", "web_direction", default=None)
    if orientation is not None:
        vector = [float(component) for component in orientation]
        if len(vector) >= 3 and any(abs(component) > 0.0 for component in vector[:3]):
            section["orientation"] = tuple(vector[:3])
    if bool(_value(source, "consistent_mass", default=False)):
        section["consistent_mass"] = True
    contact_radius = _value(source, "contact_radius", default=None)
    if contact_radius is not None and float(contact_radius) > 0.0:
        section["contact_radius"] = float(contact_radius)
    torsional_rigidity = _value(source, "torsional_rigidity", "GJ", "gj", default=None)
    if torsional_rigidity is not None:
        # Preserve an explicitly supplied invalid value so orthotropic beam
        # validation can fail closed instead of silently substituting G * J.
        section["torsional_rigidity"] = float(torsional_rigidity)
    geometric_polar_radius_squared = _value(
        source,
        "geometric_polar_radius_squared",
        default=None,
    )
    if geometric_polar_radius_squared is not None:
        section["geometric_polar_radius_squared"] = float(
            geometric_polar_radius_squared
        )
    for key, aliases in (
        ("c_y", ("c_y", "cy", "fiber_distance_y")),
        ("c_z", ("c_z", "cz", "fiber_distance_z")),
        ("torsion_modulus", ("torsion_modulus", "Wt", "torsional_section_modulus")),
        # Physical profile dimensions: the fiber-section plasticity model uses
        # them to build a profile-true web/flange fiber layout.
        ("web_height", ("web_height", "web_h", "hw")),
        ("web_thickness", ("web_thickness", "web_thk", "tw")),
        ("flange_width", ("flange_width", "flange_w", "bf")),
        ("flange_thickness", ("flange_thickness", "flange_thk", "tf")),
    ):
        value = _value(source, *aliases, default=None)
        if value is not None and float(value) > 0.0:
            section[key] = float(value)
    return section


def _generalized_beam_section_definitions(
    generated_geometry: Any,
) -> Dict[str, GeneralizedBeamSectionContract]:
    """Parse named generalized beam sections from generated geometry."""

    sections: Dict[str, GeneralizedBeamSectionContract] = {}
    for record in _collection(
        generated_geometry,
        "beam_sections",
        "generalized_beam_sections",
    ):
        name = str(_value(record, "name", "id", default="")).strip()
        if not name:
            raise ValueError("generalized-beam-section-name-must-be-nonempty")
        if name in sections:
            raise ValueError(f"duplicate-generalized-beam-section-{name}")
        value = _value(record, "value", default=record)
        try:
            sections[name] = coerce_generalized_beam_section(value, name=name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid-generalized-beam-section-{name}: {exc}") from exc
    return sections


def _generalized_beam_section(
    item: Any,
    definitions: Mapping[str, GeneralizedBeamSectionContract],
) -> Optional[GeneralizedBeamSectionContract]:
    """Resolve an inline or named generalized section for one beam."""

    source = _value(item, "cross_section", "section", default=None)
    raw = _value(
        item,
        "generalized_section",
        "generalized_beam_section",
        "beam_section",
        default=None,
    )
    if raw is None and source is not None:
        raw = _value(
            source,
            "generalized_section",
            "generalized_beam_section",
            "beam_section",
            default=None,
        )
    if raw is None:
        stiffness = _value(
            item,
            "generalized_stiffness",
            "section_stiffness",
            default=None,
        )
        if stiffness is None and source is not None:
            stiffness = _value(
                source,
                "generalized_stiffness",
                "section_stiffness",
                default=None,
            )
        if stiffness is None:
            return None
        raw = {
            "stiffness": stiffness,
            "mass_matrix": _value(
                item,
                "generalized_mass_matrix",
                "generalized_mass_per_length",
                "mass_per_length",
                default=_value(
                    source,
                    "generalized_mass_matrix",
                    "generalized_mass_per_length",
                    "mass_per_length",
                    default=None,
                ),
            ),
        }
    if isinstance(raw, str):
        name = raw.strip()
        if name not in definitions:
            raise ValueError(f"unknown-generalized-beam-section-{name}")
        return definitions[name]
    try:
        return coerce_generalized_beam_section(raw)
    except (TypeError, ValueError) as exc:
        element_id = _value(item, "id", "element_id", default="<unnamed>")
        raise ValueError(
            f"invalid-generalized-beam-section-for-element-{element_id}: {exc}"
        ) from exc


def _has_cross_section(item: Any) -> bool:
    section = _value(item, "cross_section", "section", default=None)
    if isinstance(section, Mapping):
        return True
    return any(_value(item, name, default=None) is not None for name in ("area", "A", "Iy", "iy", "Iz", "iz", "J", "torsion_constant"))


def _material_name(item: Any, config: AnyStructureFEMConfig) -> str:
    return str(_value(item, "material", "material_name", default=config.default_material))


def _material_property(item: Any, *names: str, default: Any = None) -> Any:
    """Read a material value from flat or common nested material records."""

    missing = object()
    value = _value(item, *names, default=missing)
    if value is not missing:
        return value
    for container_name in ("elastic", "engineering_constants", "orthotropic"):
        container = _value(item, container_name, default=None)
        value = _value(container, *names, default=missing)
        if value is not missing:
            return value
    return default


def _normalized_elastic_symmetry(item: Any) -> str:
    raw = _material_property(
        item,
        "elastic_symmetry",
        "symmetry",
        "material_symmetry",
        default="",
    )
    symmetry = str(raw or "").strip().lower().replace("-", "_")
    if symmetry in {"orthotropic", "orthotropy", "ortho"}:
        return "orthotropic"
    if symmetry in {"anisotropic", "anisotropy", "general_anisotropic"}:
        return "anisotropic"
    if symmetry in {"isotropic", "isotropy", "iso"}:
        return "isotropic"
    if symmetry:
        # Preserve an explicit unknown declaration so the caller rejects it
        # instead of silently interpreting a future symmetry as isotropic.
        return symmetry
    orthotropic_markers = (
        "elastic_modulus_1",
        "elastic_modulus_2",
        "elastic_modulus_3",
        "E1",
        "E2",
        "E3",
        "shear_modulus_12",
        "G12",
    )
    if any(_material_property(item, marker, default=None) is not None for marker in orthotropic_markers):
        return "orthotropic"
    return "isotropic"


def _orthotropic_constant(item: Any, descriptive_name: str, alias: str) -> float:
    value = _material_property(item, descriptive_name, alias, alias.lower(), default=None)
    if value is None:
        raise ValueError(f"orthotropic-material-missing-{descriptive_name}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"orthotropic-material-invalid-{descriptive_name}") from exc


def _hill48_from_material_record(item: Any) -> Any:
    """Build optional Hill-48 strengths from nested or flat geometry metadata."""

    hill = _material_property(
        item,
        "hill_yield",
        "hill48",
        "hill_48",
        "yield_surface",
        default=None,
    )
    if hill is None:
        flat_names = (
            "X",
            "Y",
            "Z",
            "S12",
            "S13",
            "S23",
            "yield_strength_1",
            "yield_strength_2",
            "yield_strength_3",
        )
        if any(
            _material_property(item, name, default=None) is not None
            for name in flat_names
        ):
            hill = item
    if hill is None:
        return None
    # Keep the solver contract concrete even when geometry supplies a
    # structurally similar object from another package.
    from .materials import Hill48Yield

    if isinstance(hill, Hill48Yield):
        return hill

    def strength(*names: str) -> float:
        value = _material_property(hill, *names, default=None)
        if value is None:
            raise ValueError(f"hill48-missing-{names[0]}")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"hill48-invalid-{names[0]}") from exc

    return Hill48Yield(
        X=strength("X", "x", "yield_strength_1", "longitudinal_strength"),
        Y=strength("Y", "y", "yield_strength_2", "transverse_strength_2"),
        Z=strength("Z", "z", "yield_strength_3", "transverse_strength_3"),
        S12=strength("S12", "s12", "shear_strength_12"),
        S13=strength("S13", "s13", "shear_strength_13"),
        S23=strength("S23", "s23", "shear_strength_23"),
    )


def _hardening_curve_from_material_record(item: Any) -> Any:
    hardening = _material_property(
        item,
        "hardening_curve",
        "hardening",
        default=None,
    )
    if hardening is None or hasattr(hardening, "flow_stress"):
        return hardening
    if not isinstance(hardening, Mapping):
        raise ValueError("material-hardening-curve-must-be-an-object-or-mapping")
    from .material_curves import curve_from_properties

    properties = dict(hardening)
    properties.pop("type", None)
    properties.pop("model", None)
    try:
        return curve_from_properties(properties)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid-material-hardening-curve") from exc


def _shell_material_kwargs(shell: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    direction = _value(
        shell,
        "material_direction",
        "material_axis_1",
        "orthotropic_direction",
        default=None,
    )
    if direction is not None:
        try:
            components = np.asarray(direction, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("shell-material-direction-must-be-a-vector") from exc
        if components.size != 3:
            raise ValueError("shell-material-direction-must-have-three-components")
        kwargs["material_direction"] = tuple(float(value) for value in components)
    angle = _value(
        shell,
        "material_angle_deg",
        "material_angle",
        "orthotropic_angle_deg",
        default=None,
    )
    if angle is not None:
        try:
            kwargs["material_angle_deg"] = float(angle)
        except (TypeError, ValueError) as exc:
            raise ValueError("shell-material-angle-must-be-numeric") from exc
    return kwargs


def _shell_formulation_policy(
    shell: Any,
    node_count: int,
) -> tuple[Optional[str], str]:
    """Return the authored request and centrally resolved shell policy."""

    formulation = _value(shell, "formulation", default=None)
    shell_formulation = _value(shell, "shell_formulation", default=None)
    if formulation is not None and shell_formulation is not None:
        first = _normalized_shell_formulation(int(node_count), str(formulation))
        second = _normalized_shell_formulation(int(node_count), str(shell_formulation))
        if first != second:
            raise ValueError("conflicting-shell-formulation-fields")
    request = formulation if formulation is not None else shell_formulation
    requested = None if request is None else str(request)
    selected = _normalized_shell_formulation(int(node_count), requested)
    supplied_id = _value(shell, "formulation_id", default=None)
    expected_ids = {
        "e4-pl": "E4_PL_QUALIFIED_Q4_HYBRID_V2",
        "e4-pl-s3-v2d": "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
        "legacy-s3": "LEGACY_SHELL_ELEMENT_TRI3",
        "legacy": "LEGACY_SHELL_ELEMENT",
    }
    expected_id = expected_ids.get(selected)
    if supplied_id is not None and (
        expected_id is None or str(supplied_id) != expected_id
    ):
        raise ValueError("shell-formulation-id-does-not-match-selected-policy")
    if selected == "e4-pl-s3-v2d":
        if supplied_id is None:
            raise ValueError("qualified-s3-generated-record-requires-formulation-id")
        if _value(shell, "reference_normal", default=None) is None:
            raise ValueError("qualified-s3-generated-record-requires-owner-normal")
        if _value(shell, "owner_normal_authority", default=None) != (
            "PHYSICAL_SURFACE_OWNER_NORMAL_V2D_V1"
        ):
            raise ValueError(
                "qualified-s3-generated-record-requires-physical-owner-normal-authority"
            )
    return requested, selected


def _shell_director_kwargs(
    shell: Any,
    selected_formulation: str,
) -> Dict[str, Any]:
    """Forward an explicitly authored shell-director authority.

    A generalized membrane/bending coupling matrix has a physical thickness
    direction, so qualified Q4 construction must not infer that direction from
    connectivity winding.  Normalized generated geometry may supply the exact
    authority as ``reference_normal`` (and, optionally, ``director_polarity``).
    The authority is valid only for the qualified Q4 or qualified S3 routes.
    A legacy request fails closed instead of discarding or interpreting
    formulation-specific state.
    """

    reference_normal = _value(shell, "reference_normal", default=None)
    director_polarity = _value(shell, "director_polarity", default=None)
    if reference_normal is None and director_polarity is None:
        return {}
    if selected_formulation not in {"e4-pl", "e4-pl-s3", "e4-pl-s3-v2d"}:
        raise ValueError(
            "shell-director-authority-requires-qualified-shell-formulation"
        )

    kwargs: Dict[str, Any] = {}
    if reference_normal is not None:
        try:
            components = np.asarray(reference_normal, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("shell-reference-normal-must-be-a-vector") from exc
        if components.size != 3:
            raise ValueError("shell-reference-normal-must-have-three-components")
        if not np.all(np.isfinite(components)):
            raise ValueError("shell-reference-normal-must-be-finite")
        component_scale = float(np.max(np.abs(components)))
        if component_scale <= 0.0:
            raise ValueError("shell-reference-normal-must-be-nonzero")
        kwargs["reference_normal"] = tuple(float(value) for value in components)
    if director_polarity is not None:
        kwargs["director_polarity"] = director_polarity
    return kwargs


def _generalized_shell_section_definitions(
    generated_geometry: Any,
) -> Dict[str, GeneralizedShellSection]:
    """Parse named A/B/D/As shell sections from generated geometry."""

    sections: Dict[str, GeneralizedShellSection] = {}
    for record in _collection(
        generated_geometry,
        "shell_sections",
        "generalized_shell_sections",
    ):
        name = str(_value(record, "name", "id", default="")).strip()
        if not name:
            raise ValueError("generalized-shell-section-name-must-be-nonempty")
        if name in sections:
            raise ValueError(f"duplicate-generalized-shell-section-{name}")
        value = _value(record, "value", default=record)
        if isinstance(value, Mapping) and "name" not in value:
            value = {**value, "name": name}
        try:
            section = coerce_generalized_shell_section(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid-generalized-shell-section-{name}: {exc}") from exc
        if section is None:  # Defensive: named section records cannot be null.
            raise ValueError(f"invalid-generalized-shell-section-{name}: section is null")
        sections[name] = section
    return sections


def _generalized_shell_section(
    item: Any,
    definitions: Mapping[str, GeneralizedShellSection],
) -> Optional[GeneralizedShellSection]:
    """Resolve an inline or named generalized section for one shell."""

    raw = _value(
        item,
        "shell_section",
        "generalized_shell_section",
        default=None,
    )
    if raw is None:
        return None
    if isinstance(raw, str):
        name = raw.strip()
        if name not in definitions:
            raise ValueError(f"unknown-generalized-shell-section-{name}")
        return definitions[name]
    try:
        return coerce_generalized_shell_section(raw)
    except (TypeError, ValueError) as exc:
        element_id = _value(item, "id", "element_id", default="<unnamed>")
        raise ValueError(
            f"invalid-generalized-shell-section-for-element-{element_id}: {exc}"
        ) from exc


def _structural_member_role(item: Any, source_name: str = "") -> Optional[str]:
    raw = " ".join(
        str(value)
        for value in (
            source_name,
            _value(item, "role", "structural_role", "member_role", "kind", "category", "type", default=""),
        )
        if value is not None
    ).lower()
    if any(token in raw for token in ("stiffener", "longitudinal", "stringer")):
        return "stiffener"
    if any(token in raw for token in ("girder", "frame", "transverse")):
        return "girder"
    return None


def _is_idealized_member_plate(item: Any, config: AnyStructureFEMConfig) -> Optional[str]:
    role = _structural_member_role(item)
    if role == "stiffener" and config.idealize_stiffeners_as_beams:
        return role
    if role == "girder" and config.idealize_girders_as_beams:
        return role
    return None


def _add_model_element(model: FEModel, elem_id: int, element: Any) -> None:
    if model.mesh.get_element(elem_id) is not None:
        raise ValueError(f"duplicate-element-id-{elem_id}")
    model.add_element(elem_id, element)


def _item_copy(item: Any) -> Any:
    if isinstance(item, Mapping):
        return dict(item)
    return item


def _beam_item_copy(source_name: str, item: Any) -> Any:
    copied = _item_copy(item)
    role = _structural_member_role(copied, source_name)
    if isinstance(copied, Mapping):
        copied.setdefault("role", role or "beam")
        copied.setdefault("geometry_source", source_name)
    return copied


def _polygon_area(coords: np.ndarray) -> float:
    if coords.shape[0] < 3:
        return 0.0
    centroid = np.mean(coords, axis=0)
    area = 0.0
    for i in range(coords.shape[0]):
        a = coords[i] - centroid
        b = coords[(i + 1) % coords.shape[0]] - centroid
        area += 0.5 * float(np.linalg.norm(np.cross(a, b)))
    return area


def _infer_member_centerline(plates: List[Any], node_coords: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    coords = []
    for plate in plates:
        for node_id in _node_ids(plate):
            if node_id in node_coords:
                coords.append(node_coords[node_id])
    if len(coords) < 2:
        raise ValueError("member-plate-has-insufficient-nodes")
    points = np.asarray(coords, dtype=float)
    center = np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1.0e-12:
        raise ValueError("member-plate-centerline-has-zero-length")
    direction = vh[0]
    projections = (points - center) @ direction
    p0 = center + float(np.min(projections)) * direction
    p1 = center + float(np.max(projections)) * direction
    if np.linalg.norm(p1 - p0) <= 1.0e-12:
        raise ValueError("member-plate-centerline-has-zero-length")
    return p0, p1


def _section_from_member_plates(
    plates: List[Any],
    node_coords: Dict[int, np.ndarray],
    p0: np.ndarray,
    p1: np.ndarray,
    scale: float,
) -> Dict[str, float]:
    for plate in plates:
        if _has_cross_section(plate):
            return _cross_section(plate)

    length = float(np.linalg.norm(p1 - p0))
    if length <= 0.0:
        raise ValueError("member-plate-centerline-has-zero-length")

    area = 0.0
    iy = 0.0
    iz = 0.0
    max_section_width = 0.0
    max_thickness = 0.0
    for plate in plates:
        node_ids = _node_ids(plate)
        coords = np.asarray([node_coords[node_id] for node_id in node_ids if node_id in node_coords], dtype=float)
        if coords.shape[0] < 3:
            continue
        thickness = _as_float(_value(plate, "thickness", "t"), 0.0)
        if thickness <= 0.0:
            continue
        face_area = _polygon_area(coords[:4] if coords.shape[0] >= 4 else coords)
        section_width = face_area / length if length > 0.0 else 0.0
        if section_width <= 0.0:
            continue
        area += thickness * section_width
        iy += thickness * section_width**3 / 12.0
        iz += section_width * thickness**3 / 12.0
        max_section_width = max(max_section_width, section_width)
        max_thickness = max(max_thickness, thickness)

    if area <= 0.0:
        raise ValueError("member-plate-section-could-not-be-inferred")

    scaled_area = area * scale**2
    scaled_iy = max(iy * scale**4, 1.0e-12)
    scaled_iz = max(iz * scale**4, 1.0e-12)
    section = {
        "area": scaled_area,
        "Iy": scaled_iy,
        "Iz": scaled_iz,
        "J": max((iy + iz) * scale**4, 1.0e-12),
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
    }
    if max_section_width > 0.0 and max_thickness > 0.0:
        # Strip section: width (web) along local z, thickness along local y.
        section["c_z"] = 0.5 * max_section_width * scale
        section["c_y"] = 0.5 * max_thickness * scale
        section["torsion_modulus"] = section["J"] / (max_thickness * scale)

    # The inferred strip section has its width (web) along the in-plane
    # direction transverse to the centerline.  Recover that direction so the
    # beam local z axis matches the Iy/Iz meaning used above.
    points = []
    for plate in plates:
        for node_id in _node_ids(plate):
            if node_id in node_coords:
                points.append(node_coords[node_id])
    if len(points) >= 3:
        points_arr = np.asarray(points, dtype=float)
        axis = (p1 - p0) / length
        offsets = points_arr - np.mean(points_arr, axis=0)
        offsets -= np.outer(offsets @ axis, axis)
        _, singular_values, vh = np.linalg.svd(offsets, full_matrices=False)
        if singular_values.size and singular_values[0] > 1.0e-9 * length:
            section["orientation"] = tuple(float(component) for component in vh[0])
    return section


def _member_group_key(item: Any, role: str) -> Tuple[str, str]:
    raw_id = _value(item, "member_id", "parent_id", "stiffener_id", "girder_id", "name", "id", default="member")
    return role, str(raw_id)


def _next_available_id(used: set, start: int) -> int:
    value = int(start)
    while value in used:
        value += 1
    used.add(value)
    return value


def idealize_generated_geometry_members(
    generated_geometry: Any,
    config: Optional[AnyStructureFEMConfig] = None,
) -> Dict[str, Any]:
    """Return generated geometry with tagged stiffener/girder plates collapsed to beams.

    A client may produce member geometry as plate surfaces. For the default
    beam-shell solver idealization, tagged stiffener and girder plates are
    removed from the shell set and represented by equivalent line beams.
    """
    config = config or AnyStructureFEMConfig()
    if generated_geometry is None:
        raise ValueError("missing-generated-geometry")

    nodes = [_item_copy(node) for node in _collection(generated_geometry, "nodes")]
    node_coords = {node_id: np.array([x, y, z], dtype=float) for node_id, x, y, z in (_coords_from_node(node, 1.0) for node in nodes)}
    used_node_ids = set(node_coords)

    output: Dict[str, Any] = {
        "name": str(_value(generated_geometry, "name", default="ANYfeSolverGeneratedGeometry")),
        "nodes": nodes,
        "shells": [],
        "beams": [_beam_item_copy(source, beam) for source, beam in _combined_collections(
            generated_geometry,
            "beams",
            "beam_members",
            "line_members",
            "members",
            "stiffeners",
            "girders",
            "frames",
            "longitudinals",
            "transverses",
        )],
        "couplings": [_item_copy(item) for _source, item in _combined_collections(generated_geometry, "couplings", "mpcs")],
        "rigid_lids": [_item_copy(item) for _source, item in _combined_collections(generated_geometry, "rigid_lids", "diaphragms")],
        "supports": [_item_copy(item) for _source, item in _combined_collections(generated_geometry, "supports", "boundary_conditions")],
        "materials": [_item_copy(item) for item in _collection(generated_geometry, "materials")],
        "shell_sections": [
            _item_copy(item)
            for item in _collection(
                generated_geometry,
                "shell_sections",
                "generalized_shell_sections",
            )
        ],
        "beam_sections": [
            _item_copy(item)
            for item in _collection(
                generated_geometry,
                "beam_sections",
                "generalized_beam_sections",
            )
        ],
        "idealization": {"auto_member_beams": [], "excluded_member_plates": []},
    }

    member_plate_groups: Dict[Tuple[str, str], List[Any]] = {}
    excluded_member_roles: List[str] = []
    for _source_name, shell in _combined_collections(generated_geometry, "shells", "shell_faces", "faces", "plates"):
        role = _is_idealized_member_plate(shell, config)
        if role and config.exclude_idealized_member_plates:
            excluded_member_roles.append(role)
            output["idealization"]["excluded_member_plates"].append({"id": _value(shell, "id", "element_id", default=None), "role": role})
            if config.auto_idealize_member_plates_as_beams:
                member_plate_groups.setdefault(_member_group_key(shell, role), []).append(shell)
            continue
        output["shells"].append(_item_copy(shell))

    existing_member_keys = {
        _member_group_key(beam, role)
        for beam in output["beams"]
        for role in [_structural_member_role(beam)]
        if role is not None
    }
    existing_roles = {role for role, _member_id in existing_member_keys}
    if excluded_member_roles and config.require_idealized_member_beams and not config.auto_idealize_member_plates_as_beams:
        missing_roles = sorted({role for role in excluded_member_roles if role not in existing_roles})
        if missing_roles:
            raise ValueError(f"idealized-member-plates-require-beam-members-{','.join(missing_roles)}")

    used_element_ids = {
        int(element_id)
        for element_id in (
            _value(item, "id", "element_id", default=None)
            for item in output["shells"] + output["beams"] + output["couplings"] + output["rigid_lids"]
        )
        if element_id is not None
    }

    for key, plates in member_plate_groups.items():
        role, member_id = key
        if key in existing_member_keys:
            continue
        p0, p1 = _infer_member_centerline(plates, node_coords)
        section = _section_from_member_plates(plates, node_coords, p0, p1, config.geometry_scale)
        n0 = _next_available_id(used_node_ids, 900_000)
        n1 = _next_available_id(used_node_ids, n0 + 1)
        output["nodes"].extend(
            [
                {"id": n0, "coords": p0.tolist()},
                {"id": n1, "coords": p1.tolist()},
            ]
        )
        node_coords[n0] = p0
        node_coords[n1] = p1
        element_id = _next_available_id(used_element_ids, 20_000)
        beam = {
            "id": element_id,
            "node_ids": [n0, n1],
            "cross_section": section,
            "role": role,
            "member_id": member_id,
            "material": _material_name(plates[0], config),
            "generated_from": "member_plates",
        }
        output["beams"].append(beam)
        output["idealization"]["auto_member_beams"].append(
            {"id": element_id, "role": role, "member_id": member_id, "source_plate_count": len(plates)}
        )

    return output


def _add_materials(model: FEModel, generated_geometry: Any, config: AnyStructureFEMConfig) -> None:
    model.add_material(
        config.default_material,
        elastic_modulus=config.elastic_modulus,
        poisson_ratio=config.poisson_ratio,
        density=config.density,
        yield_stress=config.yield_stress,
    )
    model.current_material = config.default_material
    for item in _collection(generated_geometry, "materials"):
        name = str(_value(item, "name", "id", default=config.default_material))
        symmetry = _normalized_elastic_symmetry(item)
        if symmetry == "anisotropic":
            raise NotImplementedError(
                f"Generated material {name!r} requests general anisotropic elasticity; "
                "ANYfeSolver currently supports isotropic and orthotropic materials only."
            )
        if symmetry not in {"isotropic", "orthotropic"}:
            raise NotImplementedError(
                f"Generated material {name!r} requests unsupported elastic symmetry "
                f"{symmetry!r}; ANYfeSolver currently supports isotropic and "
                "orthotropic materials only."
            )
        if symmetry == "orthotropic":
            model.add_orthotropic_material(
                name,
                elastic_modulus_1=_orthotropic_constant(item, "elastic_modulus_1", "E1"),
                elastic_modulus_2=_orthotropic_constant(item, "elastic_modulus_2", "E2"),
                elastic_modulus_3=_orthotropic_constant(item, "elastic_modulus_3", "E3"),
                poisson_ratio_12=_orthotropic_constant(item, "poisson_ratio_12", "nu12"),
                poisson_ratio_13=_orthotropic_constant(item, "poisson_ratio_13", "nu13"),
                poisson_ratio_23=_orthotropic_constant(item, "poisson_ratio_23", "nu23"),
                shear_modulus_12=_orthotropic_constant(item, "shear_modulus_12", "G12"),
                shear_modulus_13=_orthotropic_constant(item, "shear_modulus_13", "G13"),
                shear_modulus_23=_orthotropic_constant(item, "shear_modulus_23", "G23"),
                density=_as_float(_material_property(item, "density"), config.density),
                hill_yield=_hill48_from_material_record(item),
                hardening_curve=_hardening_curve_from_material_record(item),
            )
        else:
            model.add_material(
                name,
                elastic_modulus=_as_float(_material_property(item, "elastic_modulus", "E"), config.elastic_modulus),
                poisson_ratio=_as_float(_material_property(item, "poisson_ratio", "nu"), config.poisson_ratio),
                density=_as_float(_material_property(item, "density"), config.density),
                yield_stress=_as_float(_value(item, "yield_stress", "fy"), config.yield_stress),
                hardening_curve=_hardening_curve_from_material_record(item),
            )


def build_fe_model_from_generated_geometry(
    generated_geometry: Any,
    config: Optional[AnyStructureFEMConfig] = None,
) -> FEModel:
    """Convert normalized generated geometry into an ``FEModel``."""
    config = config or AnyStructureFEMConfig()
    if generated_geometry is None:
        raise ValueError("missing-generated-geometry")
    generated_geometry = idealize_generated_geometry_members(generated_geometry, config)

    model = FEModel(str(_value(generated_geometry, "name", default="ANYfeSolverGeneratedGeometry")))
    _add_materials(model, generated_geometry, config)
    generalized_shell_sections = _generalized_shell_section_definitions(
        generated_geometry
    )
    generalized_beam_sections = _generalized_beam_section_definitions(
        generated_geometry
    )

    nodes = _collection(generated_geometry, "nodes")
    if not nodes:
        raise ValueError("generated-geometry-has-no-nodes")
    for node in nodes:
        node_id, x, y, z = _coords_from_node(node, config.geometry_scale)
        model.add_node(node_id, x, y, z)

    element_count = 0
    skipped_member_plate_roles: List[str] = []
    if config.include_shells:
        for _source_name, shell in _combined_collections(generated_geometry, "shells", "shell_faces", "faces", "plates"):
            member_plate_role = _is_idealized_member_plate(shell, config)
            if member_plate_role and config.exclude_idealized_member_plates:
                skipped_member_plate_roles.append(member_plate_role)
                continue
            node_ids = _node_ids(shell)
            if len(node_ids) not in {3, 4, 6, 8}:
                raise ValueError(f"unsupported-shell-topology-{len(node_ids)}")
            elem_id = int(_value(shell, "id", "element_id", default=element_count + 1))
            thickness = _as_float(_value(shell, "thickness", "t"), 0.0) * config.geometry_scale
            if thickness <= 0.0:
                raise ValueError("shell-thickness-must-be-positive")
            elem_type = str(_value(shell, "type", default="")).upper()
            formulation, selected_formulation = _shell_formulation_policy(
                shell,
                len(node_ids),
            )
            _add_model_element(
                model,
                elem_id,
                create_shell_element(
                    elem_id,
                    node_ids,
                    _material_name(shell, config),
                    formulation=formulation,
                    thickness=thickness,
                    shell_section=_generalized_shell_section(
                        shell,
                        generalized_shell_sections,
                    ),
                    **(
                        {}
                        if selected_formulation == "e4-pl-s3-v2d"
                        else {"reduced_integration": elem_type == "S8R"}
                    ),
                    **_shell_material_kwargs(shell),
                    **_shell_director_kwargs(shell, selected_formulation),
                ),
            )
            element_count += 1

    beam_roles: List[str] = []
    if config.include_beams:
        beam_collections = (
            "beams",
            "beam_members",
            "line_members",
            "members",
            "stiffeners",
            "girders",
            "frames",
            "longitudinals",
            "transverses",
        )
        for source_name, beam in _combined_collections(generated_geometry, *beam_collections):
            node_ids = _node_ids(beam)
            elem_id = int(_value(beam, "id", "element_id", default=20_000 + element_count + 1))
            section = _cross_section(beam)
            generalized_section = _generalized_beam_section(
                beam,
                generalized_beam_sections,
            )
            material_name = _material_name(beam, config)
            role = _structural_member_role(beam, source_name)
            if len(node_ids) == 2:
                element = BeamElement(
                    elem_id,
                    node_ids,
                    material_name,
                    section,
                    section=generalized_section,
                )
            elif len(node_ids) == 3:
                element = QuadraticBeamElement(
                    elem_id,
                    node_ids,
                    material_name,
                    section,
                    section=generalized_section,
                )
            else:
                raise ValueError(f"unsupported-beam-topology-{len(node_ids)}")
            element.structural_role = role or "beam"
            element.geometry_source = source_name
            _add_model_element(model, elem_id, element)
            if role:
                beam_roles.append(role)
            element_count += 1

    if skipped_member_plate_roles and config.require_idealized_member_beams:
        missing_roles = sorted({role for role in skipped_member_plate_roles if role not in beam_roles})
        if missing_roles:
            raise ValueError(f"idealized-member-plates-require-beam-members-{','.join(missing_roles)}")

    for coupling in _collection(generated_geometry, "couplings", "mpcs"):
        elem_id = int(_value(coupling, "id", "element_id", default=30_000 + element_count + 1))
        beam_node_id = int(_value(coupling, "beam_node_id", "slave_node"))
        shell_node_ids = [int(node_id) for node_id in _value(coupling, "shell_node_ids", "master_nodes", default=[])]
        shape_weights = np.asarray(_value(coupling, "shape_weights", "weights", default=[]), dtype=float)
        eccentricity = np.asarray(_value(coupling, "eccentricity", default=[0.0, 0.0, 0.0]), dtype=float)
        if len(shell_node_ids) == 0 or shape_weights.size != len(shell_node_ids):
            raise ValueError("invalid-beam-shell-coupling")
        _add_model_element(
            model,
            elem_id,
            InterpolatedBeamShellMPCElement(
                elem_id,
                beam_node_id,
                shell_node_ids,
                shape_weights,
                eccentricity,
                material_name=config.default_material,
            ),
        )
        element_count += 1

    for lid in _collection(generated_geometry, "rigid_lids", "diaphragms"):
        elem_id = int(_value(lid, "id", "element_id", default=40_000 + element_count + 1))
        center_node_id = int(_value(lid, "center_node_id", "reference_node", "master_node"))
        ring_node_ids = [int(node_id) for node_id in _value(lid, "ring_node_ids", "node_ids", "nodes", default=[])]
        if len(ring_node_ids) < 3:
            raise ValueError("invalid-rigid-lid-ring")
        _add_model_element(
            model,
            elem_id,
            RigidLidMPCElement(
                elem_id,
                center_node_id,
                ring_node_ids,
                material_name=config.default_material,
            ),
        )
        element_count += 1

    for support in _collection(generated_geometry, "supports", "boundary_conditions"):
        node_ids = [int(node_id) for node_id in _value(support, "node_ids", "nodes", default=[])]
        constraints = dict(_value(support, "dof_constraints", "constraints", default={}))
        if node_ids and constraints:
            model.add_boundary_condition(BoundaryCondition(str(_value(support, "name", default="generated_support")), node_ids, constraints))

    if model.mesh.num_elements == 0:
        raise ValueError("generated-geometry-has-no-supported-elements")
    return model


def _calc_part(calc_object: Any, name: str) -> Any:
    root = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object
    return _value(root, name, default=None)


def _pressure_pa(calc_object: Any, lat_press: float, config: AnyStructureFEMConfig) -> float:
    if config.pressure_pa is not None:
        return float(config.pressure_pa) * float(config.load_scale)
    if lat_press:
        return float(lat_press) * 1000.0 * float(config.load_scale)
    for part_name in ("Plate", "Stiffener"):
        part = _calc_part(calc_object, part_name)
        pressure = _value(part, "pressure", "lat_press", default=None)
        if pressure is not None:
            return float(pressure) * 1.0e6 * float(config.load_scale)
    return 0.0


def _design_stress_pa(calc_object: Any, attr1: str, attr2: Optional[str] = None) -> float:
    part = _calc_part(calc_object, "Stiffener") or _calc_part(calc_object, "Plate")
    if part is None:
        return 0.0
    first = _as_float(_value(part, attr1), 0.0)
    second = _as_float(_value(part, attr2), first) if attr2 else first
    value = first if abs(first) >= abs(second) else second
    return value * 1.0e6


def _shell_elements(model: FEModel) -> List[ShellElement]:
    return [element for element in model.mesh.elements.values() if isinstance(element, ShellElement)]


def _edge_weights(nodes: List[Any], axis: int) -> Dict[int, float]:
    if not nodes:
        return {}
    coords = np.asarray([node.coords() for node in nodes], dtype=float)
    if len(nodes) == 1:
        return {int(nodes[0].id): 1.0}
    order = np.argsort(coords[:, axis])
    sorted_nodes = [nodes[int(i)] for i in order]
    sorted_coords = coords[order, axis]
    weights = {}
    for i, node in enumerate(sorted_nodes):
        if i == 0:
            length = abs(sorted_coords[1] - sorted_coords[0])
        elif i == len(sorted_nodes) - 1:
            length = abs(sorted_coords[-1] - sorted_coords[-2])
        else:
            length = 0.5 * abs(sorted_coords[i + 1] - sorted_coords[i - 1])
        weights[int(node.id)] = float(length)
    return weights


def _add_rectangular_edge_loads(load_case: LoadCase, calc_object: Any, model: FEModel, config: AnyStructureFEMConfig) -> Dict[str, float]:
    shell_elements = _shell_elements(model)
    if not shell_elements:
        return {"axial_stress_pa": 0.0, "transverse_stress_pa": 0.0, "shear_stress_pa": 0.0}

    shell_node_ids = sorted({node_id for element in shell_elements for node_id in element.node_ids})
    nodes = [model.mesh.get_node(node_id) for node_id in shell_node_ids]
    nodes = [node for node in nodes if node is not None]
    coords = np.asarray([node.coords() for node in nodes], dtype=float)
    span = np.ptp(coords, axis=0)
    if span[0] <= 0.0 or span[1] <= 0.0:
        return {"axial_stress_pa": 0.0, "transverse_stress_pa": 0.0, "shear_stress_pa": 0.0}

    thickness = float(np.mean([element.thickness for element in shell_elements]))
    sx = _design_stress_pa(calc_object, "sigma_x1", "sigma_x2") * config.load_scale
    sy = _design_stress_pa(calc_object, "sigma_y1", "sigma_y2") * config.load_scale
    txy = _design_stress_pa(calc_object, "tau_xy") * config.load_scale
    xmin, xmax = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    ymin, ymax = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    tol = 1.0e-8 * max(float(np.max(span)), 1.0)
    x0 = [node for node in nodes if abs(node.x - xmin) <= tol]
    x1 = [node for node in nodes if abs(node.x - xmax) <= tol]
    y0 = [node for node in nodes if abs(node.y - ymin) <= tol]
    y1 = [node for node in nodes if abs(node.y - ymax) <= tol]

    def add_edge(edge_nodes: List[Any], weights_axis: int, force_per_length: np.ndarray) -> None:
        weights = _edge_weights(edge_nodes, weights_axis)
        for node_id, length in weights.items():
            load_case.add_nodal_load(node_id, forces=force_per_length * length)

    # Existing normalized stress inputs are treated as compression-positive.
    add_edge(x0, 1, np.array([sx * thickness, txy * thickness, 0.0]))
    add_edge(x1, 1, np.array([-sx * thickness, -txy * thickness, 0.0]))
    add_edge(y0, 0, np.array([txy * thickness, sy * thickness, 0.0]))
    add_edge(y1, 0, np.array([-txy * thickness, -sy * thickness, 0.0]))
    return {"axial_stress_pa": sx, "transverse_stress_pa": sy, "shear_stress_pa": txy}


def build_symmetric_load_case(
    calc_object: Any,
    model: FEModel,
    config: Optional[AnyStructureFEMConfig] = None,
    lat_press: float = 0.0,
) -> LoadCase:
    """Create symmetric pressure and in-plane loads from normalized inputs."""
    config = config or AnyStructureFEMConfig()
    load_case = LoadCase("anystructure_symmetric_load")
    pressure = _pressure_pa(calc_object, lat_press, config)
    if pressure != 0.0:
        for element in _shell_elements(model):
            load_case.add_pressure_load(int(element.element_id), float(config.pressure_sign) * pressure)
    if config.add_inplane_edge_loads:
        _add_rectangular_edge_loads(load_case, calc_object, model, config)
    return load_case


def recover_prestress_from_static_result(
    model: FEModel,
    displacements: np.ndarray,
    *,
    nonlinear_result: Any = None,
    recovered_stresses: Optional[Mapping[int, Any]] = None,
    recovery_provenance: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Recover buckling prestress, retaining Gauss-point and material history.

    Linear calls use displacement-based stress recovery.  When a matching
    nonlinear result is supplied, the unified recovery path reconstructs
    stresses from committed shell-layer and beam-fiber states and records its
    provenance.
    """

    states: Dict[int, Dict[str, Any]] = {}
    provenance_payload: Dict[str, Any] | None = (
        None
        if recovery_provenance is None
        else dict(recovery_provenance)
    )
    if recovered_stresses is not None:
        stresses = recovered_stresses
    elif nonlinear_result is None:
        stresses = compute_stresses(model, displacements)
    else:
        from .recovery import recover_stress_result

        recovery = recover_stress_result(
            model,
            displacements,
            nonlinear_result=nonlinear_result,
            copy_committed_states=False,
        )
        stresses = recovery.element_stresses
        provenance_payload = recovery.provenance.to_dict()
    shell_count = 0
    beam_count = 0
    shell_compression = []
    beam_compression = []
    for element_id, element in model.mesh.elements.items():
        stress = stresses.get(element_id)
        if isinstance(element, ShellElement) and stress:
            expected_stations = len(np.asarray(element.gauss_points))
            if expected_stations <= 0:
                raise ValueError("shell prestress recovery requires integration stations")
            station_weights = np.asarray(element.gauss_weights, dtype=float).reshape(-1)
            if (
                station_weights.shape != (expected_stations,)
                or not np.all(np.isfinite(station_weights))
                or np.any(station_weights <= 0.0)
                or not np.isfinite(float(np.sum(station_weights)))
                or float(np.sum(station_weights)) <= 0.0
            ):
                raise ValueError(
                    "shell prestress recovery requires positive finite integration weights"
                )
            has_membrane_resultants = "membrane_resultants" in stress
            has_bending_resultants = "bending_resultants" in stress
            if has_membrane_resultants != has_bending_resultants:
                raise ValueError(
                    "shell resultant recovery requires paired membrane and "
                    "bending fields"
                )
            if has_membrane_resultants:
                membrane_resultants = np.asarray(
                    stress["membrane_resultants"],
                    dtype=float,
                )
                bending_resultants = np.asarray(
                    stress["bending_resultants"],
                    dtype=float,
                )
                if (
                    membrane_resultants.shape != (expected_stations, 3)
                    or bending_resultants.shape != (expected_stations, 3)
                    or not np.all(np.isfinite(membrane_resultants))
                    or not np.all(np.isfinite(bending_resultants))
                ):
                    raise ValueError(
                        "inconsistent or non-finite generalized shell resultant "
                        f"recovery shape; expected {(expected_stations, 3)}"
                    )
                mean_membrane = np.average(
                    membrane_resultants,
                    axis=0,
                    weights=station_weights,
                )
            else:
                sx_gp = np.asarray(stress.get("membrane_xx", np.zeros(1)), dtype=float).reshape(-1)
                sy_gp = np.asarray(stress.get("membrane_yy", np.zeros_like(sx_gp)), dtype=float).reshape(-1)
                txy_gp = np.asarray(stress.get("membrane_xy", np.zeros_like(sx_gp)), dtype=float).reshape(-1)
                if (
                    sx_gp.size != expected_stations
                    or sy_gp.size != expected_stations
                    or txy_gp.size != expected_stations
                    or not np.all(np.isfinite(sx_gp))
                    or not np.all(np.isfinite(sy_gp))
                    or not np.all(np.isfinite(txy_gp))
                ):
                    raise ValueError(
                        "inconsistent or non-finite shell membrane stress "
                        f"recovery shape; expected {expected_stations} stations"
                    )
                bx_gp = np.asarray(stress.get("bending_xx", np.zeros_like(sx_gp)), dtype=float).reshape(-1)
                by_gp = np.asarray(stress.get("bending_yy", np.zeros_like(sx_gp)), dtype=float).reshape(-1)
                bxy_gp = np.asarray(stress.get("bending_xy", np.zeros_like(sx_gp)), dtype=float).reshape(-1)
                if (
                    bx_gp.size != expected_stations
                    or by_gp.size != expected_stations
                    or bxy_gp.size != expected_stations
                    or not np.all(np.isfinite(bx_gp))
                    or not np.all(np.isfinite(by_gp))
                    or not np.all(np.isfinite(bxy_gp))
                ):
                    raise ValueError(
                        "inconsistent or non-finite shell bending stress "
                        f"recovery shape; expected {expected_stations} stations"
                    )

                thickness = float(element.thickness)
                membrane_resultants = np.column_stack((sx_gp, sy_gp, txy_gp)) * thickness
                bending_resultants = (
                    np.column_stack((bx_gp, by_gp, bxy_gp))
                    * thickness
                    * thickness
                    / 6.0
                )
                mean_membrane = np.array(
                    (
                        float(np.average(sx_gp, weights=station_weights)) * thickness,
                        float(np.average(sy_gp, weights=station_weights)) * thickness,
                        float(np.average(txy_gp, weights=station_weights)) * thickness,
                    ),
                    dtype=float,
                )
            if getattr(element, "formulation_id", None) == (
                "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
            ):
                # The qualified V2D reference-buckling surface consumes only
                # one uniform, compression-positive membrane resultant.
                # Recovery is tension-positive, hence the sign conversion.
                state = {"membrane_compression": (-mean_membrane).tolist()}
            else:
                state = {
                    # Legacy scalar summaries remain for serialized/downstream
                    # compatibility, but are now quadrature-weighted.  The
                    # enhanced Mindlin operator consumes the full station fields.
                    "membrane_force_x": float(mean_membrane[0]),
                    "membrane_force_y": float(mean_membrane[1]),
                    "membrane_force_xy": float(mean_membrane[2]),
                    "membrane_forces_at_gauss": membrane_resultants.tolist(),
                    "bending_moments_at_gauss": bending_resultants.tolist(),
                    "resultant_summary_policy": (
                        "QUADRATURE_WEIGHTED_INTEGRATION_STATION_MEAN_V1"
                    ),
                }
            for provenance_key in (
                "bubble_linearization_policy",
                "through_thickness_stress_profile",
            ):
                if provenance_key in stress:
                    state[provenance_key] = stress[provenance_key]
            states[int(element_id)] = state
            shell_compression.append(
                float(max(np.max(-membrane_resultants[:, :2]), 0.0))
            )
            shell_count += 1
        elif isinstance(element, (BeamElement, QuadraticBeamElement)) and stress:
            if "generalized_resultant" in stress:
                generalized_resultant = np.asarray(
                    stress["generalized_resultant"],
                    dtype=float,
                )
                if generalized_resultant.ndim == 1:
                    if generalized_resultant.shape != (6,):
                        raise ValueError(
                            "inconsistent generalized beam resultant recovery shape"
                        )
                    axial_force = float(generalized_resultant[0])
                elif (
                    generalized_resultant.ndim == 2
                    and generalized_resultant.shape[1] == 6
                ):
                    axial_force = float(np.mean(generalized_resultant[:, 0]))
                else:
                    raise ValueError(
                        "inconsistent generalized beam resultant recovery shape"
                    )
            else:
                axial_force = float(stress.get("axial_stress", 0.0)) * float(getattr(element, "_A", 0.0))
            states[int(element_id)] = {"axial_force": axial_force}
            beam_compression.append(max(-axial_force, 0.0))
            beam_count += 1
    summary = {
        "shell_elements": shell_count,
        "beam_elements": beam_count,
        "state_count": len(states),
        "max_shell_compression_resultant": float(max(shell_compression) if shell_compression else 0.0),
        "max_beam_compression_force": float(max(beam_compression) if beam_compression else 0.0),
    }
    if provenance_payload is not None:
        summary["stress_recovery"] = provenance_payload
    return states, summary


def _stress_statistics(model: FEModel, displacements: np.ndarray, percentile: float) -> Dict[str, float]:
    values = []
    for stress in compute_stresses(model, displacements).values():
        if "von_mises" not in stress:
            continue
        values.extend(np.asarray(stress["von_mises"], dtype=float).reshape(-1).tolist())
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "max": 0.0, "percentile": 0.0}
    return {
        "count": int(arr.size),
        "max": float(np.max(arr)),
        "percentile": float(np.percentile(arr, percentile)),
    }


def _max_translation(model: FEModel, displacements: np.ndarray) -> float:
    value = 0.0
    for node in model.mesh.nodes.values():
        value = max(value, float(np.linalg.norm(displacements[node.dofs[:3]])))
    return value


def _beam_role_counts(model: FEModel) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for element in model.mesh.elements.values():
        if not isinstance(element, (BeamElement, QuadraticBeamElement)):
            continue
        role = str(getattr(element, "structural_role", "beam") or "beam")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _invalid_result(reason: str, diagnostics: Optional[Dict[str, Any]] = None) -> AnyStructureFEMResult:
    return AnyStructureFEMResult(valid=False, status="invalid", invalid_reason=reason, diagnostics=diagnostics or {})


def run_anystructure_fem_mode(
    calc_object: Any,
    generated_geometry: Any,
    config: Optional[AnyStructureFEMConfig] = None,
    lat_press: float = 0.0,
) -> AnyStructureFEMResult:
    """Run static full-geometry analysis and buckling from recovered prestress."""
    config = config or AnyStructureFEMConfig()
    try:
        model = build_fe_model_from_generated_geometry(generated_geometry, config)
        load_case = build_symmetric_load_case(calc_object, model, config, lat_press=lat_press)
    except Exception as exc:
        return _invalid_result(str(exc))

    resultant = load_case_resultant(model, load_case)
    displacements, solver_info = solve_linear(model, load_case, solver_type=config.solver_type, constraint_mode="auto")
    static_status = str((solver_info.get("convergence_info") or {}).get("status", "unknown"))
    if static_status != "converged":
        return AnyStructureFEMResult(
            valid=False,
            status="static_failed",
            invalid_reason=static_status,
            static_solver_status=static_status,
            node_count=model.mesh.num_nodes,
            element_count=model.mesh.num_elements,
            shell_element_count=len(_shell_elements(model)),
            beam_element_count=sum(isinstance(e, (BeamElement, QuadraticBeamElement)) for e in model.mesh.elements.values()),
            load_resultant=resultant,
            diagnostics={"solver_info": solver_info},
            displacements=displacements,
        )

    prestress_states, prestress_summary = recover_prestress_from_static_result(model, displacements)
    if not prestress_states:
        return AnyStructureFEMResult(
            valid=False,
            status="prestress_failed",
            invalid_reason="failed-prestress-recovery",
            static_solver_status=static_status,
            node_count=model.mesh.num_nodes,
            element_count=model.mesh.num_elements,
            load_resultant=resultant,
            diagnostics={"solver_info": solver_info},
            displacements=displacements,
        )

    buckling = solve_eigenvalue_buckling(model, prestress_states, num_modes=config.num_buckling_modes)
    stress_stats = _stress_statistics(model, displacements, config.stress_percentile)
    mode_factors = [float(mode.load_factor) for mode in buckling.modes]
    valid = buckling.solver_status == "ok" and bool(mode_factors)
    status = "ok" if valid else "buckling_failed"
    invalid_reason = None if valid else buckling.solver_status
    return AnyStructureFEMResult(
        valid=valid,
        status=status,
        invalid_reason=invalid_reason,
        static_solver_status=static_status,
        buckling_solver_status=buckling.solver_status,
        critical_load_factor=buckling.critical_load_factor,
        buckling_load_factors=mode_factors,
        max_displacement=float(np.max(np.abs(displacements))) if displacements.size else 0.0,
        max_translation=_max_translation(model, displacements),
        stress_max=stress_stats["max"],
        stress_percentile=stress_stats["percentile"],
        node_count=model.mesh.num_nodes,
        element_count=model.mesh.num_elements,
        shell_element_count=len(_shell_elements(model)),
        beam_element_count=sum(isinstance(e, (BeamElement, QuadraticBeamElement)) for e in model.mesh.elements.values()),
        prestress_summary=prestress_summary,
        load_resultant=resultant,
        diagnostics={
            "solver_info": solver_info,
            "stress_statistics": stress_stats,
            "buckling": buckling.to_dict(),
            "beam_role_counts": _beam_role_counts(model),
        },
        displacements=displacements,
        buckling_result=buckling,
    )


# Generic solver-owned names.  Historical names remain available for
# downstream ANYstructure compatibility.
GeneratedGeometryFEMConfig = AnyStructureFEMConfig
GeneratedGeometryFEMResult = AnyStructureFEMResult
run_generated_geometry_fem = run_anystructure_fem_mode

__all__ = [
    "AnyStructureFEMConfig",
    "AnyStructureFEMResult",
    "GeneratedGeometryFEMConfig",
    "GeneratedGeometryFEMResult",
    "build_fe_model_from_generated_geometry",
    "build_symmetric_load_case",
    "idealize_generated_geometry_members",
    "recover_prestress_from_static_result",
    "run_anystructure_fem_mode",
    "run_generated_geometry_fem",
]
