"""Generated and executable CalculiX reference cases.

Deck generation is deliberately separate from validation.  A generated deck
has status ``not_executed``.  It only becomes a numerical validation result
after CalculiX has run, the requested FRD/DAT observables have been parsed, and
every declared comparison has met its tolerance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from anyfileio import (
    CalculixParsedResults,
    DeckModel,
    DeckSupport,
    merge_results as merge_calculix_results,
    parse_dat as parse_calculix_dat,
    parse_frd as parse_calculix_frd,
    write_deck,
)
from anymaterial import material_symmetry
from anymesher import Mesh as NeutralMesh

from .boundary import BoundaryCondition, LoadCase
from .elements import BeamElement, ShellElement, create_shell_element
from .fe_core import FEModel
from .mesh_gen import generate_simple_panel_mesh


DEFAULT_EXTERNAL_REFERENCE_PATH = Path("reports/external_references/external_reference_report.json")
DEFAULT_CALCULIX_RUN_PATH = Path("reports/external_references/runs")
CALCULIX_EXECUTABLE_ENV = "ANYSOLVER_CALCULIX_EXECUTABLE"
_COMPONENT_INDEX = {
    "x": 0,
    "ux": 0,
    "u1": 0,
    "fx": 0,
    "rf1": 0,
    "y": 1,
    "uy": 1,
    "u2": 1,
    "fy": 1,
    "rf2": 1,
    "z": 2,
    "uz": 2,
    "u3": 2,
    "fz": 2,
    "rf3": 2,
}


@dataclass(frozen=True)
class ExternalReferenceCase:
    """One generated external-reference input deck."""

    name: str
    kind: str
    inp_path: Path
    metadata_path: Path
    model_summary: Mapping[str, Any]
    load_summary: Mapping[str, Any]
    assumptions: Tuple[str, ...] = ()
    comparisons: Tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inp_path": str(self.inp_path),
            "metadata_path": str(self.metadata_path),
            "model_summary": dict(self.model_summary),
            "load_summary": dict(self.load_summary),
            "assumptions": list(self.assumptions),
            "comparisons": [dict(item) for item in self.comparisons],
        }


def _resolved_shell_material_orientation(
    model: FEModel,
    element: ShellElement,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve solver shell material axes as global CalculiX vectors."""

    coords = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
    frame = np.asarray(element._center_frame(coords), dtype=float)
    local_x = frame[:, 0]
    normal = frame[:, 2]
    direction = getattr(element, "material_direction", None)
    if direction is None:
        axis_1 = local_x
    else:
        supplied = np.asarray(direction, dtype=float).reshape(-1)
        if supplied.size != 3 or not np.all(np.isfinite(supplied)):
            raise ValueError(
                f"Shell element {element.element_id} has an invalid material_direction"
            )
        projected = supplied - float(supplied @ normal) * normal
        norm = float(np.linalg.norm(projected))
        if norm <= 1.0e-12:
            raise ValueError(
                f"Shell element {element.element_id} material_direction has no in-plane projection"
            )
        axis_1 = projected / norm
    angle = math.radians(float(getattr(element, "material_angle_deg", 0.0)))
    axis_2_base = np.cross(normal, axis_1)
    axis_1 = math.cos(angle) * axis_1 + math.sin(angle) * axis_2_base
    axis_1 /= max(float(np.linalg.norm(axis_1)), 1.0e-30)
    axis_2 = np.cross(normal, axis_1)
    axis_2 /= max(float(np.linalg.norm(axis_2)), 1.0e-30)
    return axis_1, axis_2


def _neutral_calculix_model(
    model: FEModel,
    load_case: Optional[LoadCase],
) -> DeckModel:
    """Flatten native solver objects into ANYfileio's neutral deck contract."""

    neutral = NeutralMesh()
    for node_id, node in model.mesh.nodes.items():
        neutral.nodes[int(node_id)] = np.asarray(node.coords(), dtype=float)

    material_of_element: Dict[int, str] = {}
    thickness_of_element: Dict[int, float] = {}
    beam_sections: Dict[int, Mapping[str, Any]] = {}
    shell_orientations: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for element_id, element in model.mesh.elements.items():
        if getattr(element, "shell_section", None) is not None:
            raise NotImplementedError(
                "CalculiX reference export does not yet map generalized shell "
                f"section resultants for element {int(element_id)}. A, B, D and "
                "As (including membrane-bending coupling) must be validated "
                "analytically or with a dedicated section-capable reference deck."
            )
        if getattr(element, "generalized_section", None) is not None:
            raise NotImplementedError(
                "CalculiX reference export cannot faithfully map the coupled "
                f"6x6 generalized beam section on element {int(element_id)} to "
                "the current equivalent RECT beam deck. Use analytical or "
                "dedicated section-stiffness validation."
            )

        node_ids = tuple(int(node_id) for node_id in getattr(element, "node_ids", ()))
        if isinstance(element, ShellElement):
            target = neutral.tris if len(node_ids) in (3, 6) else neutral.quads
            target[int(element_id)] = node_ids
            thickness_of_element[int(element_id)] = float(element.thickness)
            material = model.get_material(element.material_name)
            if material_symmetry(material) == "orthotropic":
                shell_orientations[int(element_id)] = _resolved_shell_material_orientation(
                    model, element
                )
        elif isinstance(element, BeamElement):
            material = model.get_material(element.material_name)
            if material_symmetry(material) == "orthotropic":
                element_type = "B32" if len(node_ids) == 3 else "B31"
                raise NotImplementedError(
                    "CalculiX reference export cannot faithfully represent orthotropic beam "
                    f"element set {element_type}/{element.material_name}: the solver's explicit "
                    "cross_section['torsional_rigidity'] is independent of the equivalent RECT "
                    "section. Use analytical orthotropic beam validation."
                )
            neutral.beams[int(element_id)] = node_ids
            beam_sections[int(element_id)] = dict(element.cross_section)
        else:
            continue
        material_of_element[int(element_id)] = str(element.material_name)

    supports: List[DeckSupport] = []
    for boundary_condition in model.boundary_conditions:
        dofs = tuple(
            str(name)
            for name, value in boundary_condition.dof_constraints.items()
            if abs(float(value)) == 0.0
        )
        for node_id in boundary_condition.node_ids:
            if dofs:
                supports.append(DeckSupport(int(node_id), dofs))

    nodal_loads: Dict[int, Sequence[float]] = {}
    pressure_of_element: Dict[int, float] = {}
    gravity: Optional[Tuple[float, float, float]] = None
    if load_case is not None:
        nodal_loads = {
            int(node_id): np.asarray(values, dtype=float)
            for node_id, values in load_case.nodal_loads.items()
        }
        pressure_of_element = {
            int(element_id): float(value)
            for element_id, value in load_case.pressure_loads.items()
        }
        if load_case.gravity is not None:
            gravity = tuple(
                float(value) for value in np.asarray(load_case.gravity, dtype=float).reshape(3)
            )

    return DeckModel(
        mesh=neutral,
        name=model.name,
        materials=dict(model.materials),
        material_of_element=material_of_element,
        thickness_of_element=thickness_of_element,
        beam_section_of_element=beam_sections,
        shell_orientation_of_element=shell_orientations,
        supports=tuple(supports),
        nodal_loads=nodal_loads,
        pressure_of_element=pressure_of_element,
        gravity=gravity,
    )


def write_calculix_input_deck(
    model: FEModel,
    load_case: Optional[LoadCase],
    output_path: Path | str,
    *,
    analysis: str = "static",
    metadata: Optional[Mapping[str, Any]] = None,
    comparisons: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ExternalReferenceCase:
    """Write through ANYfileio while preserving the solver verification sidecar."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.apply_boundary_conditions()
    report = write_deck(
        _neutral_calculix_model(model, load_case),
        output,
        analysis=analysis,
        num_modes=5,
        overwrite=True,
    )

    if load_case is None:
        load_summary = {"name": None, "nodal_loads": 0, "pressure_loads": 0}
    else:
        load_summary = {
            "name": load_case.name,
            "nodal_loads": len(load_case.nodal_loads),
            "pressure_loads": len(load_case.pressure_loads),
            "has_gravity": load_case.gravity is not None,
        }
    model_summary = {
        "name": model.name,
        "nodes": len(model.mesh.nodes),
        "elements": len(model.mesh.elements),
        "materials": sorted(model.materials),
        "analysis": analysis,
    }
    assumptions = tuple(
        item
        for item in report.assumptions
        if item != "No loads were supplied, so the static step is unloaded."
    )
    sidecar = report.path.with_suffix(".json")
    payload = {
        "name": report.path.stem,
        "kind": analysis,
        "model_summary": model_summary,
        "load_summary": load_summary,
        "assumptions": list(assumptions),
        "metadata": dict(metadata or {}),
        "comparisons": [dict(item) for item in comparisons or ()],
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ExternalReferenceCase(
        name=report.path.stem,
        kind=analysis,
        inp_path=report.path,
        metadata_path=sidecar,
        model_summary=model_summary,
        load_summary=load_summary,
        assumptions=assumptions,
        comparisons=tuple(dict(item) for item in comparisons or ()),
    )


def _navier_plate_observables(
    *,
    length: float,
    width: float,
    thickness: float,
    pressure: float,
    elastic_modulus: float,
    poisson_ratio: float,
    series_terms: int = 99,
) -> Tuple[float, float]:
    """Return center deflection and surface von Mises stress for an SSSS plate."""

    rigidity = elastic_modulus * thickness**3 / (12.0 * (1.0 - poisson_ratio**2))
    displacement_sum = 0.0
    moment_x = 0.0
    moment_y = 0.0
    for m in range(1, int(series_terms) + 1, 2):
        for n in range(1, int(series_terms) + 1, 2):
            center_shape = math.sin(m * math.pi / 2.0) * math.sin(n * math.pi / 2.0)
            wave = (m / length) ** 2 + (n / width) ** 2
            denominator = m * n * wave**2
            displacement_sum += center_shape / denominator
            moment_scale = 16.0 * pressure * center_shape / (math.pi**4 * denominator)
            moment_x += moment_scale * ((m / length) ** 2 + poisson_ratio * (n / width) ** 2)
            moment_y += moment_scale * ((n / width) ** 2 + poisson_ratio * (m / length) ** 2)
    displacement = 16.0 * pressure * displacement_sum / (math.pi**6 * rigidity)
    stress_x = 6.0 * moment_x / thickness**2
    stress_y = 6.0 * moment_y / thickness**2
    von_mises = math.sqrt(stress_x**2 - stress_x * stress_y + stress_y**2)
    return float(abs(displacement)), float(von_mises)


def _analytical_comparison(
    name: str,
    quantity: str,
    expected: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float = 0.0,
    component: Optional[str] = None,
    absolute: bool = False,
    description: str,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {
        "name": name,
        "quantity": quantity,
        "expected": float(expected),
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "absolute": bool(absolute),
        "reference": {
            "kind": "analytical",
            "description": description,
            "parameters": dict(parameters),
        },
    }
    if component is not None:
        comparison["component"] = component
    return comparison


def _pressure_plate_case(output_dir: Path) -> ExternalReferenceCase:
    length = 2.0
    width = 1.0
    thickness = 0.01
    pressure = 1000.0
    elastic_modulus = 210.0e9
    poisson_ratio = 0.3
    # The original 2x1 mesh put every node on a supported edge and therefore
    # had identically zero transverse displacement.  The 24x12 mesh includes
    # the analytical center point and resolves CalculiX S4 bending without the
    # severe coarse-mesh stiffness seen in the original deck.
    model = generate_simple_panel_mesh(
        length,
        width,
        thickness,
        num_divisions_x=24,
        num_divisions_y=12,
    )
    model.name = "external_pressure_plate_s4"
    load_case = LoadCase("pressure")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, pressure)
    center_displacement, center_von_mises = _navier_plate_observables(
        length=length,
        width=width,
        thickness=thickness,
        pressure=pressure,
        elastic_modulus=elastic_modulus,
        poisson_ratio=poisson_ratio,
    )
    reference_parameters = {
        "length_m": length,
        "width_m": width,
        "thickness_m": thickness,
        "pressure_pa": pressure,
        "elastic_modulus_pa": elastic_modulus,
        "poisson_ratio": poisson_ratio,
        "series_max_odd_index": 99,
    }
    comparisons = [
        _analytical_comparison(
            "plate_max_abs_uz",
            "max_abs_displacement",
            center_displacement,
            component="z",
            relative_tolerance=0.10,
            absolute_tolerance=1.0e-12,
            description="Navier double-sine series at the center of a simply supported thin rectangular plate",
            parameters=reference_parameters,
        ),
        _analytical_comparison(
            "plate_max_von_mises",
            "max_von_mises_stress",
            center_von_mises,
            relative_tolerance=0.15,
            absolute_tolerance=1.0,
            description="Navier plate center moments converted to top/bottom surface plane-stress von Mises stress",
            parameters=reference_parameters,
        ),
        _analytical_comparison(
            "plate_nodal_force_balance_z",
            "sum_nodal_force_balance",
            0.0,
            component="z",
            relative_tolerance=0.0,
            absolute_tolerance=0.1,
            description="Static vertical equilibrium from the complete FRD nodal applied-plus-reaction force field",
            parameters={"pressure_pa": pressure, "loaded_area_m2": length * width},
        ),
    ]
    return write_calculix_input_deck(
        model,
        load_case,
        output_dir / "pressure_plate_s4.inp",
        analysis="static",
        metadata={"purpose": "S4 simply supported pressure-plate external reference"},
        comparisons=comparisons,
    )


def _beam_buckling_case(output_dir: Path) -> ExternalReferenceCase:
    model = FEModel("external_beam_column_buckling")
    elastic_modulus = 210.0e9
    length = 4.0
    second_moment = 5.0e-6
    # The generic writer emits an equivalent square RECT section.  Choose its
    # area from I=A^2/12 so that both the generated deck and the analytical
    # Euler reference have exactly the same section stiffness.
    area = math.sqrt(12.0 * second_moment)
    num_elements = 10
    model.add_material("steel", elastic_modulus, 0.3, density=7850.0)
    section = {"area": area, "Iy": second_moment, "Iz": second_moment, "J": 2.0 * second_moment}
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    for i in range(num_elements):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(model.mesh.nodes)
    # Keep the axial field free except for node 1.  The previous deck fixed ux
    # at every node and applied the compression to a constrained node, so it
    # could not establish a geometric-stiffness reference state.
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("axial_reference", [1], {"ux": 0.0}))
    end_node = num_elements + 1
    model.add_boundary_condition(BoundaryCondition("pins", [1, end_node], {"uy": 0.0}))
    load_case = LoadCase("unit_compression")
    load_case.add_nodal_load(end_node, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    critical_load = math.pi**2 * elastic_modulus * second_moment / length**2
    comparisons = [
        _analytical_comparison(
            "column_first_buckling_factor",
            "first_buckling_factor",
            critical_load,
            relative_tolerance=0.05,
            absolute_tolerance=1.0,
            description="Euler pinned-pinned column critical load divided by the unit reference compression",
            parameters={
                "length_m": length,
                "elastic_modulus_pa": elastic_modulus,
                "second_moment_m4": second_moment,
                "reference_compression_n": 1.0,
            },
        )
    ]
    return write_calculix_input_deck(
        model,
        load_case,
        output_dir / "beam_column_buckling.inp",
        analysis="buckling",
        metadata={"purpose": "Pinned-pinned Euler beam-column buckling external reference"},
        comparisons=comparisons,
    )


def _cylindrical_shell_case(output_dir: Path) -> ExternalReferenceCase:
    model = FEModel("external_cylinder_s4_pressure")
    elastic_modulus = 210.0e9
    poisson_ratio = 0.3
    thickness = 0.02
    pressure = 1000.0
    model.add_material("steel", elastic_modulus, poisson_ratio, density=7850.0)
    radius = 1.0
    height = 2.0
    n_circ = 32
    n_z = 8
    node_id = 1
    grid: Dict[Tuple[int, int], int] = {}
    for iz in range(n_z + 1):
        z = height * iz / n_z
        for it in range(n_circ):
            theta = 2.0 * np.pi * it / n_circ
            grid[(iz, it)] = node_id
            model.add_node(node_id, radius * np.cos(theta), radius * np.sin(theta), z)
            node_id += 1
    elem_id = 1
    for iz in range(n_z):
        for it in range(n_circ):
            n1 = grid[(iz, it)]
            n2 = grid[(iz, (it + 1) % n_circ)]
            n3 = grid[(iz + 1, (it + 1) % n_circ)]
            n4 = grid[(iz + 1, it)]
            model.add_element(elem_id, create_shell_element(elem_id, [n1, n2, n3, n4], "steel", thickness=thickness))
            elem_id += 1
    bottom = [grid[(0, it)] for it in range(n_circ)]
    top = [grid[(n_z, it)] for it in range(n_circ)]
    model.add_boundary_condition(BoundaryCondition("bottom_uz", bottom, {"uz": 0.0}))
    model.add_boundary_condition(BoundaryCondition("top_uz", top, {"uz": 0.0}))
    # Radially restrain both end rings to eliminate every shell rigid-body and
    # ovalization mechanism.  Comparisons use only the central gauge region,
    # away from these deliberately conservative end restraints.
    model.add_boundary_condition(BoundaryCondition("bottom_xy", bottom, {"ux": 0.0, "uy": 0.0}))
    model.add_boundary_condition(BoundaryCondition("top_xy", top, {"ux": 0.0, "uy": 0.0}))
    load_case = LoadCase("internal_pressure")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, pressure)
    hoop_stress = pressure * radius / thickness
    axial_stress = poisson_ratio * hoop_stress
    von_mises = math.sqrt(hoop_stress**2 - hoop_stress * axial_stress + axial_stress**2)
    radial_displacement = pressure * radius**2 * (1.0 - poisson_ratio**2) / (elastic_modulus * thickness)
    reference_parameters = {
        "radius_m": radius,
        "thickness_m": thickness,
        "pressure_pa": pressure,
        "elastic_modulus_pa": elastic_modulus,
        "poisson_ratio": poisson_ratio,
        "axial_condition": "zero axial strain",
        "central_gauge_z_m": [0.75, 1.25],
    }
    radial_comparison = _analytical_comparison(
            "cylinder_mean_abs_radial_displacement",
            "mean_radial_displacement",
            radial_displacement,
            absolute=True,
            relative_tolerance=0.12,
            absolute_tolerance=1.0e-12,
            description="Thin closed-cylinder membrane displacement with axially restrained ends",
            parameters=reference_parameters,
        )
    radial_comparison.update(
        {
            "reference_radius": radius,
            "radial_coordinate_tolerance": 0.015,
            "z_min": 0.75,
            "z_max": 1.25,
        }
    )
    stress_comparison = _analytical_comparison(
            "cylinder_median_von_mises",
            "median_von_mises_stress",
            von_mises,
            relative_tolerance=0.15,
            absolute_tolerance=1.0,
            description="Thin-cylinder membrane hoop stress with Poisson axial stress from zero axial strain",
            parameters=reference_parameters,
        )
    stress_comparison.update(
        {
            "reference_radius": radius,
            "radial_coordinate_tolerance": 0.015,
            "z_min": 0.75,
            "z_max": 1.25,
        }
    )
    comparisons = [radial_comparison, stress_comparison]
    return write_calculix_input_deck(
        model,
        load_case,
        output_dir / "cylinder_s4_pressure.inp",
        analysis="static",
        metadata={"purpose": "Axially restrained cylindrical shell membrane reference"},
        comparisons=comparisons,
    )


def _orthotropic_membrane_case(output_dir: Path) -> ExternalReferenceCase:
    """Exact constant-stress S4 patch with the weak axis aligned globally x."""

    model = FEModel("external_orthotropic_membrane_s4")
    length = 2.0
    width = 1.0
    thickness = 0.01
    total_force = 100.0e3
    E1 = 150.0e9
    E2 = 10.0e9
    E3 = 8.0e9
    nu12 = 0.25
    material = model.add_orthotropic_material(
        "orthotropic",
        elastic_modulus_1=E1,
        elastic_modulus_2=E2,
        elastic_modulus_3=E3,
        poisson_ratio_12=nu12,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=4.0e9,
        shear_modulus_23=3.0e9,
        density=1600.0,
    )
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (length, 0.0, 0.0),
            (length, width, 0.0),
            (0.0, width, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        create_shell_element(
            1,
            [1, 2, 3, 4],
            material.name,
            thickness=thickness,
            material_angle_deg=90.0,
        ),
    )
    model.add_boundary_condition(
        BoundaryCondition("left_edge_x", [1, 4], {"ux": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition("in_plane_anchor", [1], {"uy": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "membrane_only",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load_case = LoadCase("constant_x_traction")
    load_case.add_nodal_load(
        2,
        [0.5 * total_force, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    load_case.add_nodal_load(
        3,
        [0.5 * total_force, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    stress_x = total_force / (width * thickness)
    displacement_x = stress_x * length / E2
    displacement_y = nu12 * stress_x * width / E1
    parameters = {
        "length_m": length,
        "width_m": width,
        "thickness_m": thickness,
        "total_edge_force_n": total_force,
        "material_angle_deg": 90.0,
        "E1_pa": E1,
        "E2_pa": E2,
        "nu12": nu12,
        "global_x_material_axis": 2,
    }
    comparisons = [
        _analytical_comparison(
            "orthotropic_max_abs_ux",
            "max_abs_displacement",
            displacement_x,
            component="x",
            relative_tolerance=0.03,
            absolute_tolerance=1.0e-12,
            description="Constant-stress orthotropic membrane extension along material axis 2",
            parameters=parameters,
        ),
        _analytical_comparison(
            "orthotropic_max_abs_uy",
            "max_abs_displacement",
            displacement_y,
            component="y",
            relative_tolerance=0.05,
            absolute_tolerance=1.0e-12,
            description="Reciprocal-Poisson transverse contraction of the constant-stress patch",
            parameters=parameters,
        ),
        _analytical_comparison(
            "orthotropic_max_von_mises",
            "max_von_mises_stress",
            stress_x,
            relative_tolerance=0.03,
            absolute_tolerance=1.0,
            description="Uniaxial physical stress invariant in the constant-stress membrane patch",
            parameters=parameters,
        ),
        _analytical_comparison(
            "orthotropic_nodal_force_balance_x",
            "sum_nodal_force_balance",
            0.0,
            component="x",
            relative_tolerance=0.0,
            absolute_tolerance=0.1,
            description="Static global-x equilibrium of applied and reaction nodal forces",
            parameters={"total_edge_force_n": total_force},
        ),
    ]
    return write_calculix_input_deck(
        model,
        load_case,
        output_dir / "orthotropic_membrane_s4.inp",
        analysis="static",
        metadata={
            "purpose": (
                "Analytical and CalculiX S4 orthotropic membrane reference"
            )
        },
        comparisons=comparisons,
    )


def generate_external_reference_cases(output_dir: Path | str = Path("reports/external_references/decks")) -> List[ExternalReferenceCase]:
    """Generate the default external reference decks."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return [
        _pressure_plate_case(root),
        _beam_buckling_case(root),
        _cylindrical_shell_case(root),
        _orthotropic_membrane_case(root),
    ]


def _component_index(component: Any) -> int:
    key = str(component or "").strip().lower()
    if key not in _COMPONENT_INDEX:
        raise ValueError(f"Unsupported vector component '{component}'")
    return _COMPONENT_INDEX[key]


def _von_mises(stress: Sequence[float]) -> float:
    if len(stress) < 6:
        raise ValueError("A six-component stress tensor is required")
    sxx, syy, szz, sxy, syz, szx = (float(value) for value in stress[:6])
    return math.sqrt(
        max(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + szx**2),
            0.0,
        )
    )


def _filtered_result_nodes(
    results: CalculixParsedResults,
    specification: Mapping[str, Any],
    candidates: Iterable[int],
) -> List[int]:
    node_ids = sorted(int(node_id) for node_id in candidates)
    uses_coordinate_filter = any(
        key in specification
        for key in ("z_min", "z_max", "reference_radius", "radial_coordinate_tolerance")
    )
    if not uses_coordinate_filter:
        return node_ids
    if not results.coordinates:
        raise ValueError("FRD coordinates are required by the observable filter")
    z_min = float(specification.get("z_min", -math.inf))
    z_max = float(specification.get("z_max", math.inf))
    reference_radius = specification.get("reference_radius")
    radial_tolerance = float(specification.get("radial_coordinate_tolerance", math.inf))
    filtered: List[int] = []
    for node_id in node_ids:
        coordinates = results.coordinates.get(node_id)
        if coordinates is None:
            continue
        x, y, z = coordinates
        if not (z_min <= z <= z_max):
            continue
        if reference_radius is not None and abs(math.hypot(x, y) - float(reference_radius)) > radial_tolerance:
            continue
        filtered.append(node_id)
    if not filtered:
        raise ValueError("No result nodes satisfy the declared coordinate filter")
    return filtered


def _extract_observable(results: CalculixParsedResults, specification: Mapping[str, Any]) -> float:
    quantity = str(specification.get("quantity", "")).strip().lower()
    if quantity == "max_abs_displacement":
        if not results.displacements:
            raise ValueError("FRD displacement results are missing")
        component = specification.get("component")
        if component is None or str(component).lower() in {"all", "magnitude"}:
            return max(float(np.linalg.norm(row[:3])) for row in results.displacements.values())
        index = _component_index(component)
        return max(abs(float(row[index])) for row in results.displacements.values())
    if quantity in {"max_von_mises_stress", "median_von_mises_stress"}:
        if not results.stresses:
            raise ValueError("FRD stress results are missing")
        node_ids = _filtered_result_nodes(results, specification, results.stresses)
        values = sorted(_von_mises(results.stresses[node_id]) for node_id in node_ids)
        if quantity == "max_von_mises_stress":
            return float(values[-1])
        middle = len(values) // 2
        return float(values[middle]) if len(values) % 2 else float(0.5 * (values[middle - 1] + values[middle]))
    if quantity == "sum_reaction":
        index = _component_index(specification.get("component"))
        if results.reaction_total is not None:
            return float(results.reaction_total[index])
        if not results.reaction_forces:
            raise ValueError("FRD/DAT reaction results are missing")
        return float(sum(row[index] for row in results.reaction_forces.values()))
    if quantity == "sum_nodal_force_balance":
        index = _component_index(specification.get("component"))
        if not results.reaction_forces:
            raise ValueError("FRD nodal force results are missing")
        return float(sum(row[index] for row in results.reaction_forces.values()))
    if quantity == "mean_radial_displacement":
        shared_nodes = _filtered_result_nodes(
            results,
            specification,
            set(results.coordinates) & set(results.displacements),
        )
        if not shared_nodes:
            raise ValueError("FRD coordinates and displacements are required for radial displacement")
        center_x = float(specification.get("center_x", 0.0))
        center_y = float(specification.get("center_y", 0.0))
        radial_values: List[float] = []
        for node_id in shared_nodes:
            x, y, _z = results.coordinates[node_id]
            ux, uy, _uz = results.displacements[node_id]
            dx = x - center_x
            dy = y - center_y
            radius = math.hypot(dx, dy)
            if radius > 1.0e-15:
                radial_values.append((ux * dx + uy * dy) / radius)
        if not radial_values:
            raise ValueError("No off-axis nodes are available for radial displacement")
        return float(sum(radial_values) / len(radial_values))
    if quantity == "first_buckling_factor":
        if not results.buckling_factors:
            raise ValueError("DAT buckling factors are missing")
        return float(results.buckling_factors[0])
    if quantity == "first_frequency_hz":
        if not results.frequencies_hz:
            raise ValueError("DAT eigenfrequencies are missing")
        return float(results.frequencies_hz[0])
    raise ValueError(f"Unsupported external-reference quantity '{quantity}'")


def evaluate_calculix_comparisons(
    results: CalculixParsedResults,
    specifications: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate parsed observables against explicit absolute/relative tolerances."""

    comparisons: List[Dict[str, Any]] = []
    for specification in specifications:
        row = dict(specification)
        expected = float(specification["expected"])
        relative_tolerance = float(specification.get("relative_tolerance", 0.0))
        absolute_tolerance = float(specification.get("absolute_tolerance", 0.0))
        try:
            actual = _extract_observable(results, specification)
            if bool(specification.get("absolute", False)):
                actual = abs(actual)
            error = abs(actual - expected)
            limit = absolute_tolerance + relative_tolerance * abs(expected)
            row.update(
                {
                    "status": "passed" if math.isfinite(actual) and error <= limit else "failed",
                    "actual": float(actual),
                    "absolute_error": float(error),
                    "relative_error": None if expected == 0.0 else float(error / abs(expected)),
                    "tolerance_limit": float(limit),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            row.update({"status": "missing", "error": str(exc)})
        comparisons.append(row)
    return comparisons


def resolve_calculix_executable(executable: Optional[Path | str] = None) -> Path:
    """Resolve an explicit executable, environment override, or PATH command."""

    requested: Optional[str]
    if executable is not None:
        requested = str(executable)
    else:
        requested = os.environ.get(CALCULIX_EXECUTABLE_ENV)
    if requested:
        requested = requested.strip().strip('"')
        candidate = Path(requested).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        located = shutil.which(requested)
        if located:
            return Path(located).resolve()
        raise FileNotFoundError(f"CalculiX executable was not found: {requested}")

    candidate_names = ["ccx", "ccx.exe"]
    for minor in range(30, 9, -1):
        candidate_names.extend(
            [
                f"ccx_2.{minor}",
                f"ccx_2.{minor}.exe",
                f"ccx_2.{minor}_MT",
                f"ccx_2.{minor}_MT.exe",
            ]
        )
    for candidate_name in candidate_names:
        located = shutil.which(candidate_name)
        if located:
            return Path(located).resolve()
    raise FileNotFoundError(
        f"CalculiX executable was not found on PATH; pass an explicit path or set {CALCULIX_EXECUTABLE_ENV}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        if "calculix" in line.lower() or re.search(r"\bccx\b", line, re.IGNORECASE):
            match = re.search(r"(?:version\s*)?(\d+\.\d+(?:\.\d+)?)", line, re.IGNORECASE)
            return match.group(1) if match else line.strip()[:200]
    return None


def calculix_solver_provenance(
    executable: Path | str,
    *,
    executable_args: Sequence[str] = (),
    probe_timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    """Capture executable identity and a best-effort non-blocking version probe."""

    resolved = resolve_calculix_executable(executable)
    stat = resolved.stat()
    probe_command = [str(resolved), *[str(arg) for arg in executable_args], "--version"]
    probe: Dict[str, Any] = {"command": probe_command}
    try:
        completed = subprocess.run(
            probe_command,
            cwd=str(resolved.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(probe_timeout_seconds),
            check=False,
        )
        combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        probe.update({"returncode": int(completed.returncode), "output": combined[:4000]})
        version = _version_from_text(combined)
    except (OSError, subprocess.TimeoutExpired) as exc:
        probe.update({"error": str(exc)})
        version = None
    return {
        "name": "CalculiX CrunchiX",
        "executable": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": int(stat.st_size),
        "version": version,
        "version_probe": probe,
    }


def _safe_case_directory_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "case"


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_calculix_reference_case(
    case: ExternalReferenceCase,
    *,
    executable: Path | str,
    run_root: Path | str = DEFAULT_CALCULIX_RUN_PATH,
    executable_args: Sequence[str] = (),
    timeout_seconds: float = 300.0,
    solver_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one deck in an isolated case directory and validate its results."""

    if float(timeout_seconds) <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    resolved = resolve_calculix_executable(executable)
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    work_dir = root / _safe_case_directory_name(case.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    job_name = case.name
    for suffix in (".frd", ".dat", ".sta", ".cvg", ".12d", ".eig", ".stdout.log", ".stderr.log"):
        stale = work_dir / f"{job_name}{suffix}"
        if stale.is_file():
            stale.unlink()
    input_path = work_dir / f"{job_name}.inp"
    metadata_path = work_dir / f"{job_name}.json"
    shutil.copy2(case.inp_path, input_path)
    shutil.copy2(case.metadata_path, metadata_path)

    command = [str(resolved), *[str(arg) for arg in executable_args], "-i", job_name]
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    returncode: Optional[int] = None
    execution_error: Optional[str] = None
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_seconds),
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        execution_error = f"CalculiX exceeded the {float(timeout_seconds):g} s timeout"
    except OSError as exc:
        execution_error = str(exc)
    duration = time.perf_counter() - started
    stdout_path = work_dir / f"{job_name}.stdout.log"
    stderr_path = work_dir / f"{job_name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    result: Dict[str, Any] = {
        "status": "execution_failed",
        "executed": True,
        "command": command,
        "working_directory": str(work_dir),
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": float(timeout_seconds),
        "duration_seconds": float(duration),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "solver_version": (solver_provenance or {}).get("version") or _version_from_text(stdout + "\n" + stderr),
        "comparisons": [],
    }
    if execution_error is not None:
        result["error"] = execution_error
        return result
    if returncode != 0:
        result["error"] = f"CalculiX exited with code {returncode}"
        return result

    parsed_parts: List[CalculixParsedResults] = []
    parse_errors: List[str] = []
    frd_path = work_dir / f"{job_name}.frd"
    dat_path = work_dir / f"{job_name}.dat"
    if frd_path.is_file():
        try:
            parsed_parts.append(parse_calculix_frd(frd_path))
        except (OSError, UnicodeError, ValueError) as exc:
            parse_errors.append(f"{frd_path.name}: {exc}")
    if dat_path.is_file():
        try:
            parsed_parts.append(parse_calculix_dat(dat_path))
        except (OSError, UnicodeError, ValueError) as exc:
            parse_errors.append(f"{dat_path.name}: {exc}")
    parsed = merge_calculix_results(*parsed_parts)
    comparisons = evaluate_calculix_comparisons(parsed, case.comparisons)
    result.update(
        {
            "output_files": {
                "frd": str(frd_path) if frd_path.is_file() else None,
                "dat": str(dat_path) if dat_path.is_file() else None,
            },
            "parsed_results": parsed.summary(),
            "parse_errors": parse_errors,
            "comparisons": comparisons,
        }
    )
    if parse_errors:
        result["status"] = "parse_failed"
    elif not parsed.has_results:
        result["status"] = "results_missing"
        result["error"] = "CalculiX returned success but no recognized FRD/DAT result was parsed"
    elif not comparisons:
        result["status"] = "not_validated"
        result["error"] = "No numerical comparison specifications were declared"
    elif all(item.get("status") == "passed" for item in comparisons):
        result["status"] = "passed"
    elif any(item.get("status") == "missing" for item in comparisons):
        result["status"] = "incomplete_results"
    else:
        result["status"] = "failed"
    return result


def generate_external_reference_report(
    output_dir: Path | str = Path("reports/external_references/decks"),
    *,
    execute: bool = False,
    calculix_executable: Optional[Path | str] = None,
    calculix_args: Sequence[str] = (),
    run_dir: Path | str = DEFAULT_CALCULIX_RUN_PATH,
    timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Generate decks and optionally execute tolerance-controlled validation."""

    cases = generate_external_reference_cases(output_dir)
    case_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "status": "not_executed",
        "schema_version": 2,
        "validation_performed": False,
        "execution_mode": "calculix" if execute else "deck_only",
        "cases": case_rows,
        "known_limitations": [
            "Deck-only mode generates reproducible inputs but is explicitly not numerical validation.",
            "The reference pack covers linear static shell response and linear eigenvalue buckling; nonlinear external comparisons remain out of scope.",
        ],
    }
    if not execute:
        for case in cases:
            row = case.to_dict()
            row["validation"] = {"status": "not_executed", "executed": False, "comparisons": []}
            case_rows.append(row)
        return report

    try:
        resolved = resolve_calculix_executable(calculix_executable)
        provenance = calculix_solver_provenance(resolved, executable_args=calculix_args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        report["status"] = "solver_unavailable"
        report["error"] = str(exc)
        for case in cases:
            row = case.to_dict()
            row["validation"] = {
                "status": "solver_unavailable",
                "executed": False,
                "error": str(exc),
                "comparisons": [],
            }
            case_rows.append(row)
        return report

    report["solver"] = provenance
    report["validation_performed"] = True
    for case in cases:
        validation = run_calculix_reference_case(
            case,
            executable=resolved,
            run_root=run_dir,
            executable_args=calculix_args,
            timeout_seconds=timeout_seconds,
            solver_provenance=provenance,
        )
        row = case.to_dict()
        row["validation"] = validation
        case_rows.append(row)
        if provenance.get("version") is None and validation.get("solver_version"):
            provenance["version"] = validation["solver_version"]
    report["status"] = "passed" if cases and all(
        row.get("validation", {}).get("status") == "passed" for row in case_rows
    ) else "failed"
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# External FE Reference Report", "", f"- Status: {report.get('status')}"]
    lines.append(f"- Execution mode: {report.get('execution_mode')}")
    solver = report.get("solver")
    if isinstance(solver, Mapping):
        lines.append(f"- Solver: {solver.get('name')} {solver.get('version') or '(version not reported)'}")
        lines.append(f"- Executable: `{solver.get('executable')}`")
        lines.append(f"- Executable SHA-256: `{solver.get('sha256')}`")
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases", []):
        summary = case.get("model_summary", {})
        validation = case.get("validation", {})
        lines.append(f"### {case.get('name')}")
        lines.append(f"- Kind: {case.get('kind')}")
        lines.append(f"- Input: `{case.get('inp_path')}`")
        lines.append(f"- Nodes: {summary.get('nodes')}")
        lines.append(f"- Elements: {summary.get('elements')}")
        lines.append(f"- Validation: {validation.get('status')}")
        for comparison in validation.get("comparisons", []):
            lines.append(
                f"- `{comparison.get('name')}`: {comparison.get('status')} "
                f"(actual={comparison.get('actual')}, expected={comparison.get('expected')})"
            )
        if validation.get("error"):
            lines.append(f"- Error: {validation.get('error')}")
        lines.append("")
    lines.extend(["## Known Limitations", ""])
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_external_reference_report(
    output: Path | str = DEFAULT_EXTERNAL_REFERENCE_PATH,
    *,
    deck_dir: Path | str = Path("reports/external_references/decks"),
    markdown: Optional[Path | str] = None,
    execute: bool = False,
    calculix_executable: Optional[Path | str] = None,
    calculix_args: Sequence[str] = (),
    run_dir: Path | str = DEFAULT_CALCULIX_RUN_PATH,
    timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Generate decks and write a JSON/Markdown report."""

    report = generate_external_reference_report(
        deck_dir,
        execute=execute,
        calculix_executable=calculix_executable,
        calculix_args=calculix_args,
        run_dir=run_dir,
        timeout_seconds=timeout_seconds,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown_path = Path(markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report
