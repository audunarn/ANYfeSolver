"""Focused mesh, load and boundary-condition verification report."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .anystructure_fem_mode import AnyStructureFEMConfig, build_fe_model_from_generated_geometry, build_symmetric_load_case
from .assembly import build_constraint_transformation, solve_linear
from .boundary import BoundaryCondition, FixedSupport, LoadCase, PinnedSupport, RollerSupport, SymmetryBC
from .dynamics import PressurePatch, assemble_pressure_patch_load_vector
from .elements import BeamElement, CoupledBeamShellElement, ShellElement, create_shell_element
from .fe_core import FEModel
from .validation import load_case_resultant, load_vector_resultant, validate_production_model

DEFAULT_MESH_LOAD_BC_VERIFICATION_PATH = Path("reports/mesh_load_bc_verification/mesh_load_bc_verification_report.json")


@dataclass(frozen=True)
class MeshLoadBCCase:
    case_id: str
    category: str
    title: str
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "title": self.title,
            "required": bool(self.required),
        }


@dataclass(frozen=True)
class MeshLoadBCCaseResult:
    case_id: str
    status: str
    category: str
    title: str
    checks: Mapping[str, Any] = field(default_factory=dict)
    measured: Mapping[str, Any] = field(default_factory=dict)
    tolerance: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "status": self.status,
            "category": self.category,
            "title": self.title,
            "checks": dict(self.checks),
            "measured": dict(self.measured),
            "tolerance": dict(self.tolerance),
            "diagnostics": dict(self.diagnostics),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


CASE_ROWS: Tuple[MeshLoadBCCase, ...] = (
    MeshLoadBCCase("MLBC-001", "mesh", "Flat stiffened mesh topology and member-line alignment"),
    MeshLoadBCCase("MLBC-002", "mesh", "Cylinder seam closure, ring-frame topology and shell orientation"),
    MeshLoadBCCase("MLBC-003", "mesh", "Q8 midside placement and distorted-mesh guardrails"),
    MeshLoadBCCase("MLBC-004", "load", "Pressure direction and resultant sign conventions"),
    MeshLoadBCCase("MLBC-005", "load", "Pressure patch area selection and resultant"),
    MeshLoadBCCase("MLBC-006", "load", "Edge load and nodal moment resultant balance"),
    MeshLoadBCCase("MLBC-007", "boundary", "Fixed, pinned, roller and symmetry support DOF semantics"),
    MeshLoadBCCase("MLBC-008", "mpc", "MPC duplicate ownership and fixed-slave rejection"),
    MeshLoadBCCase("MLBC-009", "nullspace", "Self-equilibrated free-free load nullspace consistency"),
)


def mesh_load_bc_manifest_cases() -> List[MeshLoadBCCase]:
    return list(CASE_ROWS)


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _pass(case: MeshLoadBCCase, **kwargs: Any) -> MeshLoadBCCaseResult:
    return MeshLoadBCCaseResult(case.case_id, "PASS", case.category, case.title, **kwargs)


def _fail(case: MeshLoadBCCase, reason: str, **kwargs: Any) -> MeshLoadBCCaseResult:
    return MeshLoadBCCaseResult(case.case_id, "FAIL", case.category, case.title, reason=reason, **kwargs)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _node_coords(model: FEModel, node_ids: Sequence[int]) -> np.ndarray:
    return np.asarray([model.mesh.get_node(int(node_id)).coords() for node_id in node_ids], dtype=float)


def _shell_normal(model: FEModel, shell: ShellElement) -> np.ndarray:
    coords = _node_coords(model, shell.node_ids[:4])
    normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return np.zeros(3)
    return normal / norm


def _coords_key(coords: Sequence[float]) -> Tuple[int, int, int]:
    return tuple(int(round(float(value) * 1.0e9)) for value in coords)


def _flat_stiffened_geometry() -> Dict[str, Any]:
    nodes = [
        {"id": 1, "coords": [0.0, 0.0, 0.0]},
        {"id": 2, "coords": [1.0, 0.0, 0.0]},
        {"id": 3, "coords": [2.0, 0.0, 0.0]},
        {"id": 4, "coords": [0.0, 1.0, 0.0]},
        {"id": 5, "coords": [1.0, 1.0, 0.0]},
        {"id": 6, "coords": [2.0, 1.0, 0.0]},
        {"id": 100, "coords": [0.0, 0.5, 0.2]},
        {"id": 101, "coords": [1.0, 0.5, 0.2]},
        {"id": 102, "coords": [2.0, 0.5, 0.2]},
        {"id": 200, "coords": [1.0, 0.0, 0.3]},
        {"id": 201, "coords": [1.0, 1.0, 0.3]},
    ]
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    return {
        "nodes": nodes,
        "shells": [
            {"id": 1, "node_ids": [1, 2, 5, 4], "thickness": 0.02, "role": "skin"},
            {"id": 2, "node_ids": [2, 3, 6, 5], "thickness": 0.02, "role": "skin"},
        ],
        "stiffeners": [{"id": 20, "node_ids": [100, 101, 102], "cross_section": section}],
        "girders": [{"id": 21, "node_ids": [200, 201], "cross_section": section}],
        "couplings": [
            {"id": 30, "beam_node_id": 101, "shell_node_ids": [1, 2, 5, 4], "shape_weights": [0.25] * 4},
            {"id": 31, "beam_node_id": 200, "shell_node_ids": [1, 2, 5, 4], "shape_weights": [0.25] * 4},
        ],
    }


def _cylinder_geometry(num_circ: int = 12) -> Dict[str, Any]:
    radius = 1.0
    height = 1.5
    nodes = []
    for iz, z in enumerate([0.0, 0.75, height]):
        for itheta in range(num_circ):
            theta = 2.0 * math.pi * itheta / num_circ
            nodes.append({"id": iz * num_circ + itheta + 1, "coords": [radius * math.cos(theta), radius * math.sin(theta), z]})
    shells = []
    for iz in range(2):
        for itheta in range(num_circ):
            nxt = (itheta + 1) % num_circ
            shells.append(
                {
                    "id": iz * num_circ + itheta + 1,
                    "node_ids": [
                        iz * num_circ + itheta + 1,
                        iz * num_circ + nxt + 1,
                        (iz + 1) * num_circ + nxt + 1,
                        (iz + 1) * num_circ + itheta + 1,
                    ],
                    "thickness": 0.02,
                    "role": "skin",
                }
            )
    section = {"area": 0.004, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    return {
        "plot_type": "cylinder",
        "radius_m": radius,
        "nodes": nodes,
        "shells": shells,
        "girders": [
            {"id": 1000 + i, "node_ids": [num_circ + i + 1, num_circ + ((i + 1) % num_circ) + 1], "cross_section": section, "role": "girder"}
            for i in range(num_circ)
        ],
    }


def _q8_model(midside_offset: float = 0.0, warp: float = 0.0) -> FEModel:
    model = FEModel("q8_mesh_check")
    model.add_material("steel", 210.0e9, 0.3)
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, warp),
        5: (0.5 + midside_offset, 0.0, 0.0),
        6: (1.0, 0.5, 0.0),
        7: (0.5, 1.0, 0.0),
        8: (0.0, 0.5, 0.0),
    }
    for node_id, xyz in coords.items():
        model.add_node(node_id, *xyz)
    model.add_element(1, create_shell_element(1, list(range(1, 9)), "steel", thickness=0.01))
    return model


def _run_mlbc_001(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    geometry = _flat_stiffened_geometry()
    model = build_fe_model_from_generated_geometry(geometry)
    shell_count = sum(1 for element in model.mesh.elements.values() if isinstance(element, ShellElement))
    beam_roles = sorted(getattr(element, "structural_role", "") for element in model.mesh.elements.values() if isinstance(element, BeamElement))
    coords = [_coords_key(node.coords()) for node in model.mesh.nodes.values()]
    duplicate_count = len(coords) - len(set(coords))
    stiffener_nodes = [model.mesh.get_node(node_id) for node_id in (100, 101, 102)]
    girder_nodes = [model.mesh.get_node(node_id) for node_id in (200, 201)]
    stiffener_y_spread = max(node.y for node in stiffener_nodes) - min(node.y for node in stiffener_nodes)
    girder_x_spread = max(node.x for node in girder_nodes) - min(node.x for node in girder_nodes)
    validation = validate_production_model(model, allow_free_mechanisms=True)

    _assert(shell_count == 2, "flat mesh shell count changed")
    _assert(beam_roles == ["girder", "stiffener"], "stiffener/girder roles were not preserved")
    _assert(duplicate_count == 0, "flat generated mesh contains duplicate coordinates")
    _assert(stiffener_y_spread < 1.0e-12 and girder_x_spread < 1.0e-12, "member lines are not aligned with shell coordinates")
    _assert(validation.status in {"ok", "warning"}, "flat generated mesh failed production validation")
    return _pass(
        case,
        measured={
            "shell_count": shell_count,
            "beam_roles": beam_roles,
            "duplicate_coordinate_count": duplicate_count,
            "stiffener_y_spread": stiffener_y_spread,
            "girder_x_spread": girder_x_spread,
        },
        diagnostics={"validation_status": validation.status, "mesh_quality": validation.mesh_quality},
    )


def _run_mlbc_002(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    geometry = _cylinder_geometry()
    model = build_fe_model_from_generated_geometry(geometry)
    shells = [element for element in model.mesh.elements.values() if isinstance(element, ShellElement)]
    beams = [element for element in model.mesh.elements.values() if isinstance(element, BeamElement)]
    seam_shells = [shell for shell in geometry["shells"] if {1, 12} <= set(shell["node_ids"]) or {25, 36} <= set(shell["node_ids"])]
    normal_dots = []
    for shell in shells:
        coords = _node_coords(model, shell.node_ids[:4])
        centroid = np.mean(coords, axis=0)
        radial = np.array([centroid[0], centroid[1], 0.0], dtype=float)
        radial /= max(float(np.linalg.norm(radial)), 1.0e-15)
        normal_dots.append(float(np.dot(_shell_normal(model, shell), radial)))
    validation = validate_production_model(model, allow_free_mechanisms=True)

    _assert(len(shells) == 24, "cylinder shell count changed")
    _assert(len(beams) == 12, "ring-frame beam count changed")
    _assert(len(seam_shells) >= 1, "cylinder seam is not closed by wraparound connectivity")
    _assert(min(normal_dots) > 0.95, "one or more cylinder shell normals are not outward")
    _assert(validation.status in {"ok", "warning"}, "cylinder mesh failed production validation")
    return _pass(
        case,
        measured={"shell_count": len(shells), "ring_beam_count": len(beams), "min_outward_normal_dot": min(normal_dots)},
        diagnostics={"validation_status": validation.status, "mesh_quality": validation.mesh_quality},
    )


def _run_mlbc_003(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    good = validate_production_model(_q8_model(), allow_free_mechanisms=True)
    distorted = validate_production_model(_q8_model(midside_offset=0.3, warp=0.2), allow_free_mechanisms=True)
    codes = {issue.code for issue in distorted.issues}
    _assert(good.status == "ok", "well-formed Q8 element should pass mesh guardrails")
    _assert({"MESH003", "MESH004"} <= codes, "distorted Q8 did not trigger warp and midside warnings")
    return _pass(
        case,
        measured={
            "good_status": good.status,
            "distorted_status": distorted.status,
            "max_q8_midside_deviation": distorted.mesh_quality["max_q8_midside_deviation"],
            "max_warp": distorted.mesh_quality["max_warp"],
        },
        diagnostics={"distorted_issue_codes": sorted(codes)},
    )


def _run_mlbc_004(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    flat = build_fe_model_from_generated_geometry(
        {
            "nodes": [
                {"id": 1, "coords": [0.0, 0.0, 0.0]},
                {"id": 2, "coords": [2.0, 0.0, 0.0]},
                {"id": 3, "coords": [2.0, 1.0, 0.0]},
                {"id": 4, "coords": [0.0, 1.0, 0.0]},
            ],
            "shells": [{"id": 1, "node_ids": [1, 2, 3, 4], "thickness": 0.02}],
        }
    )
    positive = LoadCase("positive_normal")
    positive.add_pressure_load(1, 5.0)
    negative = LoadCase("negative_normal")
    negative.add_pressure_load(1, -5.0)
    pos_res = load_case_resultant(flat, positive)
    neg_res = load_case_resultant(flat, negative)

    cylinder = build_fe_model_from_generated_geometry(_cylinder_geometry())
    external = LoadCase("external")
    for element in cylinder.mesh.elements.values():
        if isinstance(element, ShellElement):
            external.add_pressure_load(int(element.element_id), -100.0)
    external_vector = external.get_load_vector(cylinder.mesh, cylinder.mesh.dof_manager, cylinder.get_material)
    radial_work = 0.0
    for node in cylinder.mesh.nodes.values():
        radius = np.array([node.x, node.y, 0.0], dtype=float)
        norm = float(np.linalg.norm(radius))
        if norm > 0.0:
            radial_work += float(np.dot(external_vector[node.dofs[:3]], radius / norm))
    cyl_res = load_vector_resultant(cylinder, external_vector)

    _assert(np.allclose(pos_res.force, [0.0, 0.0, 10.0], atol=1.0e-12), "positive flat pressure resultant is wrong")
    _assert(np.allclose(neg_res.force, [0.0, 0.0, -10.0], atol=1.0e-12), "negative flat pressure resultant is wrong")
    _assert(radial_work < 0.0, "external cylinder pressure is not inward")
    _assert(np.linalg.norm(cyl_res.force) < 1.0e-10, "closed-cylinder pressure resultant should self-equilibrate")
    return _pass(
        case,
        measured={
            "flat_positive_force": pos_res.force.tolist(),
            "flat_negative_force": neg_res.force.tolist(),
            "cylinder_external_radial_force_sum": radial_work,
            "cylinder_resultant_force_norm": cyl_res.force_norm,
        },
    )


def _run_mlbc_005(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    model = build_fe_model_from_generated_geometry(_flat_stiffened_geometry())
    patch = PressurePatch("one_shell", pressure_time=1.0, element_ids=[1])
    _vector, info = assemble_pressure_patch_load_vector(model, patch, pressure=7.0)
    resultant = np.asarray(info["resultant_force"], dtype=float)
    _assert(info["selected_element_ids"] == [1], "explicit pressure patch selected wrong elements")
    _assert(np.allclose(resultant, [0.0, 0.0, 7.0], atol=1.0e-12), "pressure patch resultant does not match selected area")
    return _pass(case, measured={"selected_element_ids": info["selected_element_ids"], "resultant_force": resultant.tolist()})


class _Part:
    span = 1.0
    spacing = 1.0
    t = 0.01
    sigma_x1 = 5.0
    sigma_x2 = 5.0
    sigma_y1 = 3.0
    sigma_y2 = 3.0
    tau_xy = 1.0
    pressure = 0.0


class _Calc:
    Plate = _Part()
    Stiffener = None


def _run_mlbc_006(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    model = build_fe_model_from_generated_geometry(
        {
            "nodes": [
                {"id": 1, "coords": [0.0, 0.0, 0.0]},
                {"id": 2, "coords": [2.0, 0.0, 0.0]},
                {"id": 3, "coords": [2.0, 1.0, 0.0]},
                {"id": 4, "coords": [0.0, 1.0, 0.0]},
            ],
            "shells": [{"id": 1, "node_ids": [1, 2, 3, 4], "thickness": 0.02}],
        }
    )
    edge_load = build_symmetric_load_case(_Calc(), model, AnyStructureFEMConfig(pressure_pa=0.0, add_inplane_edge_loads=True))
    edge_resultant = load_case_resultant(model, edge_load)
    moment_load = LoadCase("pure_nodal_moment")
    moment_load.add_nodal_load(1, moments=np.array([0.0, 0.0, 12.0]))
    moment_resultant = load_case_resultant(model, moment_load)
    _assert(edge_resultant.force_norm < 1.0e-12, "balanced edge loads produced net force")
    _assert(edge_resultant.moment_norm < 1.0e-12, "balanced edge loads produced net moment")
    _assert(np.allclose(moment_resultant.moment, [0.0, 0.0, 12.0], atol=1.0e-12), "nodal moment resultant is wrong")
    return _pass(
        case,
        measured={
            "edge_force_norm": edge_resultant.force_norm,
            "edge_moment_norm": edge_resultant.moment_norm,
            "nodal_moment_resultant": moment_resultant.moment.tolist(),
        },
    )


def _support_model() -> FEModel:
    model = FEModel("support_semantics")
    model.add_material("steel", 210.0e9, 0.3)
    for node_id in range(1, 5):
        model.add_node(node_id, float(node_id), 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(PinnedSupport("pinned", [2]))
    model.add_boundary_condition(RollerSupport("roller", [3], ["uy", "uz"]))
    model.add_boundary_condition(SymmetryBC("symmetry", [4], "xy"))
    model.apply_boundary_conditions()
    return model


def _run_mlbc_007(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    model = _support_model()
    constrained = set(model.mesh.dof_manager._constrained_dofs)

    def names(node_id: int) -> List[str]:
        return [model.mesh.dof_manager.get_dof_info(dof)[2] for dof in model.mesh.get_node(node_id).dofs if dof in constrained]

    measured = {str(node_id): names(node_id) for node_id in range(1, 5)}
    _assert(measured["1"] == ["ux", "uy", "uz", "rx", "ry", "rz"], "fixed support DOFs changed")
    _assert(measured["2"] == ["ux", "uy", "uz"], "pinned support DOFs changed")
    _assert(measured["3"] == ["uy", "uz"], "roller support DOFs changed")
    _assert(measured["4"] == ["uz", "rx", "ry"], "symmetry support DOFs changed")
    return _pass(case, measured={"constrained_dof_names_by_node": measured})


def _run_mlbc_008(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    duplicate = FEModel("duplicate_mpc")
    duplicate.add_material("steel", 210.0e9, 0.3)
    duplicate.add_node(1, 0.0, 0.0, 0.0)
    duplicate.add_node(2, 0.1, 0.0, 0.0)
    duplicate.add_node(3, 0.2, 0.0, 0.0)
    duplicate.add_element(1, CoupledBeamShellElement(1, beam_node_id=2, shell_node_id=1, material_name="steel"))
    duplicate.add_element(2, CoupledBeamShellElement(2, beam_node_id=2, shell_node_id=3, material_name="steel"))
    duplicate_report = validate_production_model(duplicate, allow_free_mechanisms=True)

    fixed_slave = FEModel("fixed_slave")
    fixed_slave.add_material("steel", 210.0e9, 0.3)
    fixed_slave.add_node(1, 0.0, 0.0, 0.0)
    fixed_slave.add_node(2, 0.1, 0.0, 0.0)
    fixed_slave.add_element(1, CoupledBeamShellElement(1, beam_node_id=2, shell_node_id=1, material_name="steel"))
    fixed_slave.add_boundary_condition(BoundaryCondition("bad_fixed_slave", [2], {"ux": 0.0}))
    load = np.zeros(fixed_slave.mesh.dof_manager.total_dofs)
    fixed_slave_error = ""
    try:
        fixed_slave.apply_boundary_conditions()
        build_constraint_transformation(sparse.eye(fixed_slave.mesh.dof_manager.total_dofs, format="csr"), load, fixed_slave)
    except ValueError as exc:
        fixed_slave_error = str(exc)

    _assert("MPC001" in {issue.code for issue in duplicate_report.issues}, "duplicate MPC slave was not rejected")
    _assert(
        "CONSTRAINT002" in fixed_slave_error and "multiple dependent definitions" in fixed_slave_error,
        "fixed MPC slave was not rejected before assembly",
    )
    return _pass(
        case,
        measured={"duplicate_status": duplicate_report.status, "fixed_slave_error": fixed_slave_error},
        diagnostics={"duplicate_issue_codes": sorted(issue.code for issue in duplicate_report.issues)},
    )


def _beam_bar_model(fixed_left: bool) -> FEModel:
    model = FEModel("free_free_bar" if not fixed_left else "fixed_bar")
    model.add_material("steel", 100.0, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    if fixed_left:
        model.add_boundary_condition(BoundaryCondition("axial_only", [1, 2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
        model.add_boundary_condition(BoundaryCondition("fix_left", [1], {"ux": 0.0}))
    return model


def _run_mlbc_009(case: MeshLoadBCCase) -> MeshLoadBCCaseResult:
    free = _beam_bar_model(fixed_left=False)
    free_load = LoadCase("self_equilibrated")
    free_load.add_nodal_load(1, forces=np.array([-10.0, 0.0, 0.0]))
    free_load.add_nodal_load(2, forces=np.array([10.0, 0.0, 0.0]))
    free_u, free_info = solve_linear(free, free_load, constraint_mode="auto")
    free_extension = float(free_u[free.mesh.get_node(2).dofs[0]] - free_u[free.mesh.get_node(1).dofs[0]])

    fixed = _beam_bar_model(fixed_left=True)
    fixed_load = LoadCase("tip_load")
    fixed_load.add_nodal_load(2, forces=np.array([10.0, 0.0, 0.0]))
    fixed_u, fixed_info = solve_linear(fixed, fixed_load, constraint_mode="auto")
    fixed_extension = float(fixed_u[fixed.mesh.get_node(2).dofs[0]] - fixed_u[fixed.mesh.get_node(1).dofs[0]])

    _assert(free_info["convergence_info"]["status"] == "converged", "free-free self-equilibrated solve did not converge")
    _assert(free_info["constraint_method"].endswith("_nullspace"), "free-free solve did not use nullspace handling")
    _assert(abs(free_extension - fixed_extension) < 1.0e-10, "free-free nullspace extension does not match constrained reference")
    _assert(abs(free_extension - 0.1) < 1.0e-10, "bar extension does not match FL/EA")
    return _pass(
        case,
        measured={
            "free_extension": free_extension,
            "fixed_extension": fixed_extension,
            "expected_extension": 0.1,
            "nullspace_rank": free_info["nullspace_info"]["rank"],
        },
        diagnostics={
            "free_constraint_method": free_info["constraint_method"],
            "fixed_constraint_method": fixed_info["constraint_method"],
        },
    )


RUNNERS: Mapping[str, Callable[[MeshLoadBCCase], MeshLoadBCCaseResult]] = {
    "MLBC-001": _run_mlbc_001,
    "MLBC-002": _run_mlbc_002,
    "MLBC-003": _run_mlbc_003,
    "MLBC-004": _run_mlbc_004,
    "MLBC-005": _run_mlbc_005,
    "MLBC-006": _run_mlbc_006,
    "MLBC-007": _run_mlbc_007,
    "MLBC-008": _run_mlbc_008,
    "MLBC-009": _run_mlbc_009,
}


def run_mesh_load_bc_verification(selected_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    selected = None if selected_ids is None else {str(case_id) for case_id in selected_ids}
    results: List[MeshLoadBCCaseResult] = []
    for case in mesh_load_bc_manifest_cases():
        if selected is not None and case.case_id not in selected:
            continue
        runner = RUNNERS.get(case.case_id)
        if runner is None:
            results.append(_fail(case, "case is registered but not implemented"))
            continue
        try:
            results.append(runner(case))
        except Exception as exc:
            results.append(_fail(case, str(exc)))
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    required_failures = [result.case_id for result in results if result.status != "PASS"]
    status = "passed" if not required_failures else "failed"
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_sha(),
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__},
        "status": status,
        "counts": counts,
        "required_failures": required_failures,
        "manifest_cases": [case.to_dict() for case in mesh_load_bc_manifest_cases()],
        "results": [result.to_dict() for result in results],
        "known_limitations": [
            "This focused gate verifies supported ANYfeSolver mesh/load/boundary behavior only.",
            "Follower-pressure tangents, contact and arbitrary CAD topology are outside this focused gate.",
            "External pressure is negative element-normal direction; internal/outward pressure is positive normal.",
        ],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mesh, Load and Boundary Verification Report",
        "",
        f"- Status: {report.get('status')}",
        f"- Commit: {report.get('commit') or 'unknown'}",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted((report.get("counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Results", ""])
    for result in report.get("results", []):
        suffix = f" - {result.get('reason')}" if result.get("reason") and result.get("status") != "PASS" else ""
        lines.append(f"- {result.get('case_id')} {result.get('status')}: {result.get('title')}{suffix}")
    lines.extend(["", "## Known Limitations", ""])
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_mesh_load_bc_verification_report(
    path: Path | str = DEFAULT_MESH_LOAD_BC_VERIFICATION_PATH,
    *,
    markdown: Path | str | None = None,
    selected_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    report = run_mesh_load_bc_verification(selected_ids=selected_ids)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown is not None:
        md = Path(markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(_markdown(report), encoding="utf-8")
    return report


def mesh_load_bc_result_by_case(selected_ids: Optional[Iterable[str]] = None) -> Dict[str, MeshLoadBCCaseResult]:
    """Return case results keyed by id for reuse in other verification ledgers."""

    report = run_mesh_load_bc_verification(selected_ids=selected_ids)
    results: Dict[str, MeshLoadBCCaseResult] = {}
    for item in report["results"]:
        results[str(item["case_id"])] = MeshLoadBCCaseResult(
            case_id=str(item["case_id"]),
            status=str(item["status"]),
            category=str(item["category"]),
            title=str(item["title"]),
            checks=item.get("checks") or {},
            measured=item.get("measured") or {},
            tolerance=item.get("tolerance") or {},
            diagnostics=item.get("diagnostics") or {},
            reason=item.get("reason"),
        )
    return results
