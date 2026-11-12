"""Analytical composite-strip checks for shell/beam MPC verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

from .assembly import solve_linear
from .boundary import BoundaryCondition, LoadCase
from .buckling import solve_eigenvalue_buckling
from .elements import BeamElement, ShellElement
from .fe_core import FEModel
from .mesh_gen import MeshConfig, PanelGeometry, StiffenerCrossSection, generate_stiffened_panel_mesh
from .modal import solve_free_vibration


@dataclass(frozen=True)
class CompositeStripSpec:
    """Independent analytical reference definition for a stiffened strip."""

    length: float = 1.0
    width: float = 0.08
    thickness: float = 0.001
    stiffener_area_ratio: float = 0.4
    eccentricity_to_thickness: float = 5.0
    elastic_modulus: float = 210.0e9
    density: float = 7850.0
    poisson_ratio: float = 0.3

    @property
    def eccentricity(self) -> float:
        return float(self.eccentricity_to_thickness) * float(self.thickness)

    @property
    def stiffener_area(self) -> float:
        return float(self.stiffener_area_ratio) * float(self.width) * float(self.thickness)


@dataclass(frozen=True)
class CompositeStripProperties:
    area: float
    neutral_axis_z: float
    bending_inertia_y: float
    mass_per_length: float
    beam_area: float
    beam_inertia_y: float


def composite_strip_properties(spec: CompositeStripSpec) -> CompositeStripProperties:
    """Return composite EA/EI properties from the parallel-axis theorem."""

    plate_area = float(spec.width) * float(spec.thickness)
    beam_area = float(spec.stiffener_area)
    # The production fixture below uses a flatbar because it gives an explicit
    # A and section Iy without embedding the equation under test in the mesh.
    flatbar_width = beam_area / max(float(spec.thickness), 1.0e-30)
    section = StiffenerCrossSection.from_geometry(
        "Flatbar",
        float(spec.eccentricity),
        float(spec.thickness),
        flatbar_width,
        float(spec.thickness),
    )
    total_area = plate_area + beam_area
    zbar = beam_area * float(spec.eccentricity) / max(total_area, 1.0e-30)
    inertia = (
        float(spec.width) * float(spec.thickness) ** 3 / 12.0
        + plate_area * zbar**2
        + section.Iy
        + beam_area * (float(spec.eccentricity) - zbar) ** 2
    )
    return CompositeStripProperties(
        area=total_area,
        neutral_axis_z=zbar,
        bending_inertia_y=inertia,
        mass_per_length=float(spec.density) * total_area,
        beam_area=beam_area,
        beam_inertia_y=section.Iy,
    )


def build_composite_strip_model(
    spec: CompositeStripSpec,
    *,
    element_type: str = "Q4",
    # The Q8/Q8R coupled strip converges more slowly than Q4 in the slender
    # cantilever reference; these defaults keep the analytical check in the
    # fine-mesh regime without adding unnecessary transverse DOFs.
    shell_divisions_x: int = 48,
    shell_divisions_y: int = 2,
    beam_divisions: int = 48,
) -> FEModel:
    """Build the production shell/beam/MPC model for the analytical strip."""

    props = composite_strip_properties(spec)
    flatbar_width = props.beam_area / max(float(spec.thickness), 1.0e-30)
    panel = PanelGeometry(
        length=float(spec.length),
        width=float(spec.width),
        plate_thickness=float(spec.thickness),
        stiffener_type="Flatbar",
        stiffener_spacing=float(spec.width) / 2.0,
        stiffener_height=float(spec.eccentricity),
        stiffener_web_thickness=float(spec.thickness),
        stiffener_flange_width=flatbar_width,
        stiffener_flange_thickness=float(spec.thickness),
        num_stiffeners=1,
        in_plane_support="free",
        rotational_support="",
    )
    normalized = str(element_type).upper()
    config = MeshConfig(
        shell_num_divisions_x=int(shell_divisions_x),
        shell_num_divisions_y=int(shell_divisions_y),
        beam_num_divisions=int(beam_divisions),
        use_coupling_elements=True,
        align_mesh_to_stiffeners=True,
        use_8node_shells=normalized in {"Q8", "Q8R", "S8", "S8R"},
    )
    model = generate_stiffened_panel_mesh(panel, config)
    material = model.get_material("steel")
    material.elastic_modulus = float(spec.elastic_modulus)
    material.poisson_ratio = float(spec.poisson_ratio)
    material.density = float(spec.density)
    if normalized in {"Q8R", "S8R"}:
        for element in model.mesh.elements.values():
            if isinstance(element, ShellElement) and len(element.node_ids) == 8:
                element.reduced_integration = True
    model.clear_boundary_conditions()
    root_shell_nodes = [
        int(node_id)
        for node_id, node in model.mesh.nodes.items()
        if int(node_id) < 10000 and abs(float(node.x)) <= 1.0e-12 * max(float(spec.length), 1.0)
    ]
    model.add_boundary_condition(
        BoundaryCondition(
            "composite_strip_fixed_shell_root",
            root_shell_nodes,
            {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _tip_beam_node(model: FEModel, spec: CompositeStripSpec) -> int:
    candidates = [
        int(node_id)
        for node_id, node in model.mesh.nodes.items()
        if int(node_id) >= 10000
        and abs(float(node.x) - float(spec.length)) <= 1.0e-12 * max(float(spec.length), 1.0)
        and abs(float(node.y) - 0.5 * float(spec.width)) <= 1.0e-12 * max(float(spec.width), 1.0)
        and abs(float(node.z) - float(spec.eccentricity)) <= 1.0e-12 * max(abs(float(spec.eccentricity)), 1.0)
    ]
    if not candidates:
        raise ValueError("composite-strip tip beam node not found")
    return min(candidates)


def _element_counts(model: FEModel) -> Dict[str, int]:
    shell4 = shell8 = beam = mpc = 0
    for element in model.mesh.elements.values():
        if isinstance(element, ShellElement):
            if len(element.node_ids) == 8:
                shell8 += 1
            else:
                shell4 += 1
        elif isinstance(element, BeamElement):
            beam += 1
        elif hasattr(element, "get_mpc_constraints"):
            mpc += 1
    return {"shell4": shell4, "shell8": shell8, "beam2": beam, "mpc": mpc}


def composite_static_tip_metric(spec: CompositeStripSpec, element_type: str) -> Dict[str, Any]:
    """Compare production static tip compliance with composite beam theory."""

    model = build_composite_strip_model(spec, element_type=element_type)
    props = composite_strip_properties(spec)
    force = -1.0
    load = LoadCase("composite_strip_tip_force")
    tip_node = _tip_beam_node(model, spec)
    load.add_nodal_load(tip_node, [0.0, 0.0, force, 0.0, 0.0, 0.0])
    displacements, solver_info = solve_linear(model, load)
    tip = model.mesh.get_node(tip_node)
    if tip is None:
        raise ValueError("missing tip node")
    measured = float(displacements[tip.dofs[2]])
    reference = force * float(spec.length) ** 3 / (
        3.0 * float(spec.elastic_modulus) * max(float(props.bending_inertia_y), 1.0e-30)
    )
    relative_error = abs(measured - reference) / max(abs(reference), 1.0e-30)
    return {
        "element_type": str(element_type).upper(),
        "eccentricity_to_thickness": float(spec.eccentricity_to_thickness),
        "tip_displacement_z": measured,
        "reference_tip_displacement_z": reference,
        "relative_error": relative_error,
        "solver_status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
        "properties": props.__dict__,
        "mesh": {"nodes": len(model.mesh.nodes), "elements": len(model.mesh.elements), **_element_counts(model)},
    }


def composite_modal_metric(spec: CompositeStripSpec, element_type: str, num_modes: int = 6) -> Dict[str, Any]:
    """Compare first vertical bending frequency with composite beam theory."""

    model = build_composite_strip_model(spec, element_type=element_type)
    props = composite_strip_properties(spec)
    result = solve_free_vibration(model, num_modes=int(num_modes), dense_size_limit=20000)
    beta1 = 1.875104068711961
    omega = beta1**2 * math.sqrt(
        float(spec.elastic_modulus) * props.bending_inertia_y
        / max(props.mass_per_length * float(spec.length) ** 4, 1.0e-30)
    )
    reference_hz = omega / (2.0 * math.pi)
    measured_hz = float(result.frequencies_hz[0]) if result.frequencies_hz.size else math.nan
    relative_error = abs(measured_hz - reference_hz) / max(abs(reference_hz), 1.0e-30)
    return {
        "element_type": str(element_type).upper(),
        "eccentricity_to_thickness": float(spec.eccentricity_to_thickness),
        "frequency_hz": measured_hz,
        "reference_frequency_hz": reference_hz,
        "relative_error": relative_error,
        "solver_status": str(result.solver_status),
        "frequencies_hz": result.frequencies_hz[: min(6, result.frequencies_hz.size)].tolist(),
        "diagnostics": dict(result.diagnostics or {}),
        "properties": props.__dict__,
        "mesh": {"nodes": len(model.mesh.nodes), "elements": len(model.mesh.elements), **_element_counts(model)},
    }


def _unit_compression_states(model: FEModel, spec: CompositeStripSpec) -> Dict[int, Dict[str, float]]:
    props = composite_strip_properties(spec)
    plate_force_share = float(spec.width) * float(spec.thickness) / max(props.area, 1.0e-30)
    beam_force_share = props.beam_area / max(props.area, 1.0e-30)
    states: Dict[int, Dict[str, float]] = {}
    for element_id, element in model.mesh.elements.items():
        if isinstance(element, ShellElement):
            states[int(element_id)] = {"membrane_compression_x": plate_force_share / max(float(spec.width), 1.0e-30)}
        elif isinstance(element, BeamElement):
            states[int(element_id)] = {"axial_compression": beam_force_share}
        else:
            states[int(element_id)] = {}
    return states


def composite_buckling_metric(spec: CompositeStripSpec, element_type: str) -> Dict[str, Any]:
    """Compare fixed-free Euler buckling with production shell/beam KG."""

    model = build_composite_strip_model(spec, element_type=element_type)
    props = composite_strip_properties(spec)
    states = _unit_compression_states(model, spec)
    result = solve_eigenvalue_buckling(model, states, num_modes=3, dense_size_limit=20000)
    reference = math.pi**2 * float(spec.elastic_modulus) * props.bending_inertia_y / (4.0 * float(spec.length) ** 2)
    measured = float(result.critical_load_factor) if result.critical_load_factor is not None else math.nan
    relative_error = abs(measured - reference) / max(abs(reference), 1.0e-30)
    return {
        "element_type": str(element_type).upper(),
        "eccentricity_to_thickness": float(spec.eccentricity_to_thickness),
        "critical_load_factor": measured,
        "reference_critical_load": reference,
        "relative_error": relative_error,
        "solver_status": str(result.solver_status),
        "load_factors": [float(mode.load_factor) for mode in result.modes[:3]],
        "diagnostics": {
            key: value
            for key, value in dict(result.diagnostics or {}).items()
            if key not in {"rejected_roots"}
        },
        "properties": props.__dict__,
        "mesh": {"nodes": len(model.mesh.nodes), "elements": len(model.mesh.elements), **_element_counts(model)},
    }


def composite_strip_metric_rows(
    metric: str,
    *,
    eccentricity_to_thickness: float = 5.0,
    element_types: Iterable[str] = ("Q4", "Q8", "Q8R"),
) -> List[Dict[str, Any]]:
    """Run a compact production-model composite-strip metric set."""

    spec = CompositeStripSpec(eccentricity_to_thickness=float(eccentricity_to_thickness))
    rows: List[Dict[str, Any]] = []
    for element_type in element_types:
        if metric == "static":
            rows.append(composite_static_tip_metric(spec, element_type))
        elif metric == "modal":
            rows.append(composite_modal_metric(spec, element_type))
        elif metric == "buckling":
            rows.append(composite_buckling_metric(spec, element_type))
        else:
            raise ValueError(f"unsupported composite-strip metric: {metric}")
    return rows
