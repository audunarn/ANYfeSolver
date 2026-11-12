"""Elastic sphere onto a clamped eccentric-beam-stiffened panel: initial state.

Declared benchmark case: a hollow deformable elastic sphere (1 m outside
diameter, 10 mm wall) is placed at rest above the impact centre of a clamped
5 x 5 m x 10 mm panel stiffened by four 150 x 10 mm upright flat bars on its
underside at 1 m spacing, with a nominal 10 m drop retained as the eventual
target.  Both bodies are purely elastic (E=210 GPa, nu=0.3, rho=7850 kg/m^3,
no yield, no hardening); gravity is 9.81 m/s^2.

This module delivers the reproducible case construction and the verified
initial state only:

- the panel is built from the canonical ANYmesher neutral stiffened-panel
  geometry through public solver APIs, with the generator's legacy 235 MPa
  material explicitly replaced by the elastic case material before assembly;
- the panel-only gravity equilibrium is solved with the qualified nonlinear
  static caller (corotational kinematics, work-conjugate rotating-MPC tangent,
  initial four-way gravity partition (existing adaptive stepping), max 20 Newton iterations, tolerance 1e-8);
- the sphere is the fixed 162-node/320-triangle faceted skin from
  ``examples.elastic_sphere_stiffened_panel_mesh`` built as elastic S3 shells
  with per-triangle outward reference normals.  It carries no supports, is
  never statically solved, starts stress-free at rest, and is placed with a
  10 m clearance between its lowest outer-surface point and the equilibrated
  panel top (sphere centre z = equilibrated panel top + 10.0 + 0.500 m).

The eventual gravity-driven impact with both bodies deformable is NOT
supported by the current solver and is refused clearly: no impact history,
transient checkpoint or restart is produced or implied.  The output status is
``initial_state_prepared`` with ``impact_status=unsupported``.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from anyfesolver import (
    FEModel,
    BeamElement,
    LoadCase,
    MeshConfig,
    PanelGeometry,
    StiffenerCrossSection,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    build_constraint_transformation,
    create_shell_element,
    generate_stiffened_panel_mesh,
    solve_static_nonlinear,
)
from anyfesolver.corotational import (
    build_nonlinear_rotating_mpc_map,
    corotational_element_response,
)
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement

from examples.elastic_sphere_stiffened_panel_mesh import (
    SPHERE_MIDSURFACE_RADIUS,
    SPHERE_NODES,
    SPHERE_THICKNESS,
    SPHERE_TRIANGLES,
)

SCHEMA = "anyfesolver.elastic-sphere-stiffened-panel-initial-state-v1"
STATUS = "initial_state_prepared"
IMPACT_STATUS = "unsupported"

# Declared case data (fixed; see docs/ELASTIC_SPHERE_STIFFENED_PANEL.md).
ELASTIC_MODULUS = 210.0e9
POISSON_RATIO = 0.3
DENSITY = 7850.0
GRAVITY = 9.81
PANEL_LENGTH = 5.0
PANEL_WIDTH = 5.0
PLATE_THICKNESS = 0.01
PLATE_TOP_Z = 0.005
BAR_HEIGHT = 0.15
BAR_THICKNESS = 0.01
BAR_CENTROID_Z = -0.080
BAR_BOTTOM_Z = -0.155
NUM_BARS = 4
BAR_SPACING = 1.0
SHELL_DIVISIONS = 10
BEAM_DIVISIONS = 10
DROP_CLEARANCE = 10.0

# Physical section properties of one upright 150 x 10 mm flat bar.
BAR_AREA = BAR_THICKNESS * BAR_HEIGHT
BAR_IY = BAR_THICKNESS * BAR_HEIGHT ** 3 / 12.0
BAR_IZ = BAR_HEIGHT * BAR_THICKNESS ** 3 / 12.0
BAR_J = BAR_HEIGHT * BAR_THICKNESS ** 3 / 3.0
BAR_TORSION_MODULUS = BAR_J / BAR_THICKNESS

# Outside radius of the declared sphere (midsurface radius + half wall).
SPHERE_OUTSIDE_RADIUS = SPHERE_MIDSURFACE_RADIUS + SPHERE_THICKNESS / 2.0

# Analytic reference masses (planning quantities, not simulated results).
PLATE_MASS = DENSITY * PANEL_LENGTH * PANEL_WIDTH * PLATE_THICKNESS
BARS_MASS = NUM_BARS * PANEL_LENGTH * BAR_AREA * DENSITY
TARGET_PANEL_MASS = PLATE_MASS + BARS_MASS
SMOOTH_MIDSURFACE_SHELL_MASS = 241.707384343
HOLLOW_SOLID_VOLUME_MASS = 241.715604844

# Prospective acceptance criteria (fixed before execution; do not weaken).
RADIUS_TOLERANCE = 1.0e-12
RELATIVE_MASS_TOLERANCE = 1.0e-10
FORCE_BALANCE_TOLERANCE = 1.0e-8 * max(1.0, TARGET_PANEL_MASS * GRAVITY)
MOMENT_BALANCE_TOLERANCE = 1.0e-8 * max(1.0, TARGET_PANEL_MASS * GRAVITY * 5.0)
# Agreement between the recovered and the solver-recorded named support
# resultants (same force law; only floating-point assembly order differs).
REACTION_AGREEMENT_TOLERANCE = 1.0e-9
# Declared bar layout: four bars at 1 m spacing, centred on the case origin.
BAR_Y_POSITIONS = (-1.5, -0.5, 0.5, 1.5)

SOLVE_SETTINGS = {
    "kinematics": "corotational",
    "corotational_tangent": "work_conjugate",
    "num_layers": 5,
    "num_steps": 4,
    "max_iterations": 20,
    "tolerance": 1.0e-8,
    "max_load_factor": 1.0,
}

IMPACT_REFUSAL = (
    "Impact execution is unsupported for this case: the current solver has "
    "no admitted transient route for a free deformable shell sphere with "
    "gravity acting on both bodies. No impact history, transient checkpoint "
    "or restart exists or is implied; only the verified initial state "
    "(status=initial_state_prepared) is delivered."
)


class ImpactUnsupportedError(RuntimeError):
    """Raised for any request to execute the unsupported impact."""


def request_impact(*_args, **_kwargs):
    """Refuse an impact run clearly instead of fabricating a response."""
    raise ImpactUnsupportedError(IMPACT_REFUSAL)


def _case_material(model: FEModel) -> None:
    """Install the declared elastic material, replacing any legacy default."""
    model.add_material(
        "steel",
        ELASTIC_MODULUS,
        POISSON_RATIO,
        density=DENSITY,
        yield_stress=0.0,
        hardening_curve=None,
    )


def _install_canonical_flatbar_section(model: FEModel) -> None:
    """Install the canonical zero-flange upright-web section on the bar beams.

    The mesher's negative centroid placement parameter (-0.080 m) locates only
    the neutral beam line and the tie eccentricity; it is not case geometry.
    The physical 150 x 10 mm upright flat bar is restored through the canonical
    public section constructor ``StiffenerCrossSection.from_geometry("T-bar",
    0.150, 0.010, 0.0, 0.0)`` -- a zero-flange rectangular upright web under
    the canonical thin-wall convention -- which provides the exact area,
    bending inertias, extreme-fibre distances and the canonical thin-wall
    open-section St Venant model J = h*t^3/3 = 5e-8 m^4 with its torsion
    modulus.  J is the canonical thin-wall section model for the open
    rectangular web, not the exact 3D rectangular-torsion value.  The beam
    elements are replaced through the public element API with this canonical
    section before any assembly; no section formula is overridden by hand and
    the reported physical height/thickness stay truthful.
    """
    canonical = StiffenerCrossSection.from_geometry(
        "T-bar", BAR_HEIGHT, BAR_THICKNESS, 0.0, 0.0
    ).as_dict()
    for name, expected in (
        ("area", BAR_AREA),
        ("Iy", BAR_IY),
        ("Iz", BAR_IZ),
        ("J", BAR_J),
        ("c_y", BAR_THICKNESS / 2.0),
        ("c_z", BAR_HEIGHT / 2.0),
        ("torsion_modulus", BAR_TORSION_MODULUS),
    ):
        value = float(canonical[name])
        if abs(value - expected) > 1.0e-15 * max(1.0, abs(expected)):
            raise RuntimeError(
                f"canonical flat-bar section property {name}={value!r} "
                f"disagrees with the declared physical value {expected!r}"
            )
    for element_id, element in sorted(model.mesh.elements.items()):
        if not isinstance(element, BeamElement):
            continue
        model.add_element(
            int(element_id),
            type(element)(
                int(element_id),
                list(element.node_ids),
                material_name=element.material_name,
                cross_section=dict(canonical),
            ),
        )


def _center_panel(model: FEModel) -> None:
    """Move the mesher's [0,L]x[0,W] panel to the case origin x=y=0."""
    for node_id, node in model.mesh.nodes.items():
        model.mesh.set_node_coordinates(
            int(node_id), node.x - PANEL_LENGTH / 2.0, node.y - PANEL_WIDTH / 2.0, node.z
        )


def build_panel_model() -> FEModel:
    """Build the clamped eccentric-beam-stiffened panel (no sphere)."""
    panel = PanelGeometry(
        length=PANEL_LENGTH,
        width=PANEL_WIDTH,
        plate_thickness=PLATE_THICKNESS,
        stiffener_type="Flatbar",
        stiffener_spacing=BAR_SPACING,
        stiffener_height=BAR_CENTROID_Z,
        stiffener_web_thickness=BAR_THICKNESS,
        stiffener_flange_width=BAR_THICKNESS,
        stiffener_flange_thickness=BAR_HEIGHT,
        num_stiffeners=NUM_BARS,
        in_plane_support="Integrated",
        rotational_support="CL",
    )
    config = MeshConfig(
        shell_num_divisions_x=SHELL_DIVISIONS,
        shell_num_divisions_y=SHELL_DIVISIONS,
        beam_num_divisions=BEAM_DIVISIONS,
        use_coupling_elements=True,
        rotating_eccentricity=True,
    )
    model = generate_stiffened_panel_mesh(panel, config)
    _case_material(model)
    _install_canonical_flatbar_section(model)
    _center_panel(model)
    return model


def _sphere_geometry() -> dict:
    """Geometry report and admission checks for the fixed sphere case data."""
    nodes = np.asarray(SPHERE_NODES, dtype=float)
    tris = np.asarray(SPHERE_TRIANGLES, dtype=int) - 1
    radii = np.linalg.norm(nodes, axis=1)
    max_radius_deviation = float(np.max(np.abs(radii - SPHERE_MIDSURFACE_RADIUS)))
    edge_counts: dict[tuple[int, int], int] = {}
    min_outward_cosine = 1.0
    area = 0.0
    for triangle in tris:
        a, b, c = nodes[triangle[0]], nodes[triangle[1]], nodes[triangle[2]]
        cross = np.cross(b - a, c - a)
        cross_norm = float(np.linalg.norm(cross))
        area += 0.5 * cross_norm
        centroid = (a + b + c) / 3.0
        cosine = float(np.dot(cross, centroid) / (cross_norm * np.linalg.norm(centroid)))
        min_outward_cosine = min(min_outward_cosine, cosine)
        for i in range(3):
            edge = (int(triangle[i]), int(triangle[(i + 1) % 3]))
            edge = (min(edge), max(edge))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    closed = all(count == 2 for count in edge_counts.values())
    return {
        "midsurface_radius_m": SPHERE_MIDSURFACE_RADIUS,
        "thickness_m": SPHERE_THICKNESS,
        "node_count": int(nodes.shape[0]),
        "triangle_count": int(tris.shape[0]),
        "edge_count": len(edge_counts),
        "edges_with_two_faces": int(sum(1 for c in edge_counts.values() if c == 2)),
        "closed": bool(closed),
        "outward_winding": bool(min_outward_cosine > 0.0),
        "min_outward_cosine": float(min_outward_cosine),
        "max_radius_deviation_m": max_radius_deviation,
        "faceted_area_m2": float(area),
    }


def build_sphere_model() -> tuple:
    """Build the unconstrained elastic sphere skin from the fixed case data."""
    geometry = _sphere_geometry()
    model = FEModel(name="elastic_sphere")
    _case_material(model)
    nodes = np.asarray(SPHERE_NODES, dtype=float)
    tris = np.asarray(SPHERE_TRIANGLES, dtype=int) - 1
    for node_id, (x, y, z) in enumerate(nodes, 1):
        model.add_node(int(node_id), float(x), float(y), float(z))
    for element_id, triangle in enumerate(tris, 1):
        a, b, c = nodes[triangle[0]], nodes[triangle[1]], nodes[triangle[2]]
        normal = np.cross(b - a, c - a)
        normal = normal / np.linalg.norm(normal)
        model.add_element(
            int(element_id),
            create_shell_element(
                int(element_id),
                [int(n) + 1 for n in triangle],
                "steel",
                thickness=SPHERE_THICKNESS,
                reference_normal=(float(normal[0]), float(normal[1]), float(normal[2])),
            ),
        )
    return model, geometry


def translation_masses(model: FEModel) -> list:
    """Raw unconstrained translation mass per axis (rigid-translation form)."""
    matrix, _ = assemble_mass_matrix(model)
    manager = model.mesh.dof_manager
    masses = []
    for axis in range(3):
        rigid = np.zeros(manager.total_dofs, dtype=float)
        for node in model.mesh.nodes.values():
            rigid[node.dofs[axis]] = 1.0
        masses.append(float(rigid @ (matrix @ rigid)))
    return masses


def _panel_inventory(model: FEModel) -> dict:
    shell_nodes = []
    beam_nodes = []
    shells = beams = couplings = 0
    nodes = {}
    elements = {}
    for node_id, node in sorted(model.mesh.nodes.items()):
        nodes[str(node_id)] = [float(node.x), float(node.y), float(node.z)]
        (beam_nodes if node_id >= 10000 else shell_nodes).append(int(node_id))
    for element_id, element in sorted(model.mesh.elements.items()):
        if isinstance(element, InterpolatedBeamShellMPCElement):
            couplings += 1
            body = "beam_shell_coupling"
            element_type = "InterpolatedBeamShellMPC"
        elif isinstance(element, BeamElement):
            beams += 1
            body = "stiffener_beam"
            element_type = "Beam"
        else:
            shells += 1
            body = "plate_shell"
            element_type = "Shell"
        elements[str(element_id)] = {
            "type": element_type,
            "body": body,
            "node_ids": [int(n) for n in element.node_ids],
        }
    supports = []
    for bc in model.boundary_conditions:
        supports.append(
            {
                "name": str(bc.name),
                "components": sorted(str(key) for key in bc.dof_constraints),
                "node_count": len(bc.node_ids),
                "node_ids": [int(n) for n in bc.node_ids],
            }
        )
    beam_nodes_z = sorted({round(node.z, 12) for node in model.mesh.nodes.values() if node.id >= 10000})
    bar_y_positions = sorted({round(node.y, 12) for node in model.mesh.nodes.values() if node.id >= 10000})
    plate_nodes_z = sorted({round(node.z, 12) for node in model.mesh.nodes.values() if node.id < 10000})
    return {
        "shell_node_count": len(shell_nodes),
        "beam_node_count": len(beam_nodes),
        "shell_element_count": shells,
        "beam_element_count": beams,
        "coupling_element_count": couplings,
        "supports": supports,
        "bar_centroid_z_values_m": beam_nodes_z,
        "bar_y_positions_m": bar_y_positions,
        "plate_node_z_values_m": plate_nodes_z,
        "nodes": nodes,
        "elements": elements,
    }


def solve_panel_gravity(model: FEModel):
    """Panel-only gravity equilibrium with the qualified nonlinear static caller."""
    load_case = LoadCase(name="panel_gravity", gravity=np.array([0.0, 0.0, -GRAVITY]))
    result = solve_static_nonlinear(
        model,
        load_case,
        num_steps=SOLVE_SETTINGS["num_steps"],
        max_iterations=SOLVE_SETTINGS["max_iterations"],
        tolerance=SOLVE_SETTINGS["tolerance"],
        max_load_factor=SOLVE_SETTINGS["max_load_factor"],
        kinematics=SOLVE_SETTINGS["kinematics"],
        corotational_tangent=SOLVE_SETTINGS["corotational_tangent"],
        num_layers=SOLVE_SETTINGS["num_layers"],
    )
    return load_case, result


def _corotational_internal_force(model: FEModel, displacements: np.ndarray) -> np.ndarray:
    """Full-coordinate internal force with the solve's own corotational law.

    This is the same per-element force route the nonlinear static solver uses
    for ``kinematics="corotational"`` with the work-conjugate tangent: the
    public ``corotational_element_response`` (which resolves
    ``tangent_mode="work_conjugate"`` to the exact work-conjugate force
    ``p(u) = B(u)^T f_ref(u_d)``) for every element in corotational scope,
    and the element's own public nonlinear response for elements outside it
    (the interpolated beam-shell MPC contributes no element force; its
    constraint force enters only through the rotating-MPC transfer below).
    The material is purely elastic, so the force is path-independent and
    evaluating without the solver's committed plastic state reproduces the
    accepted equilibrium force exactly.
    """
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    internal = np.zeros(total_dofs, dtype=float)
    for element_id, element in mesh.elements.items():
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        u_element = displacements[dof_mapping]
        f_element, _unused_tangent, _unused_state = corotational_element_response(
            model,
            int(element_id),
            element,
            u_element,
            False,
            committed_state=None,
            num_layers=SOLVE_SETTINGS["num_layers"],
            tangent_mode=SOLVE_SETTINGS["corotational_tangent"],
        )
        if f_element is None:
            material = model.get_material(element.material_name)
            f_element, _unused_k, _unused_state = element.compute_nonlinear_response(
                mesh, material, u_element, None, SOLVE_SETTINGS["num_layers"], False
            )
        np.add.at(internal, dof_mapping, np.asarray(f_element, dtype=float))
    return internal


def _named_support_resultants(model: FEModel, imbalance: np.ndarray) -> dict:
    """Sum the transferred imbalance by named support boundary condition.

    Uses the same public boundary-condition data the solver's own named
    support-reaction recording uses (BC name, node ids, constrained DOF
    names mapped to their local component index).
    """
    manager = model.mesh.dof_manager
    result: dict = {}
    for boundary in getattr(model, "boundary_conditions", ()) or ():
        name = str(boundary.name)
        values = result.setdefault(name, np.zeros(6, dtype=float))
        for node_id in boundary.node_ids:
            node_dofs = manager.get_node_dofs(int(node_id))
            for dof_name in boundary.dof_constraints:
                local = ["ux", "uy", "uz", "rx", "ry", "rz"].index(dof_name)
                if local >= len(node_dofs):
                    continue
                dof = int(node_dofs[local])
                if dof >= 0:
                    values[local] += float(imbalance[dof])
    return {name: [float(v) for v in values] for name, values in result.items()}


def support_reaction_recovery(model: FEModel, load_case: LoadCase, displacements: np.ndarray) -> dict:
    """Recover per-node support reactions with the rotating-MPC transfer.

    The internal force is assembled with the same public corotational
    work-conjugate force law as the actual solve (see
    ``_corotational_internal_force``).  The raw imbalance ``F_int - F_ext``
    carries the coupling constraint forces on slave and master DOFs; the
    solver's own rotating-MPC map transfers the slave imbalance to the
    masters (the exact adjoint of the nonlinear slave orientation map), after
    which the imbalance at fixed DOFs is the physical support reaction.  The
    external load is the same public assembled gravity vector the solver
    uses as its proportional load.
    """
    model.apply_boundary_conditions()
    external, _ = assemble_load_vector(model, load_case)
    internal = _corotational_internal_force(model, displacements)
    imbalance = np.asarray(internal, dtype=float) - np.asarray(external, dtype=float)
    stiffness, _ = assemble_stiffness_matrix(model)
    _, _, transformation, _, _, _ = build_constraint_transformation(
        stiffness, np.zeros(stiffness.shape[0]), model
    )
    mpc_map = build_nonlinear_rotating_mpc_map(model, transformation)
    if mpc_map is not None:
        imbalance = mpc_map.transfer_slave_imbalance_to_masters(displacements, imbalance)
    manager = model.mesh.dof_manager
    per_node: dict = {}
    for dof in range(manager.total_dofs):
        if not manager.is_constrained(dof):
            continue
        node_id, local, _ = manager.get_dof_info(dof)
        per_node.setdefault(int(node_id), np.zeros(6, dtype=float))[local] = imbalance[dof]
    return {
        "external": external,
        "per_node": per_node,
        "named_resultants": _named_support_resultants(model, imbalance),
    }


def _global_balance(model: FEModel, recovery: dict) -> dict:
    external = np.asarray(recovery["external"], dtype=float)
    manager = model.mesh.dof_manager
    reaction_force = np.zeros(3, dtype=float)
    reaction_moment = np.zeros(3, dtype=float)
    for node_id, values in recovery["per_node"].items():
        node = model.mesh.get_node(int(node_id))
        r = node.coords()
        force = values[:3]
        reaction_force += force
        reaction_moment += np.cross(r, force) + values[3:]
    applied_force = np.zeros(3, dtype=float)
    applied_moment = np.zeros(3, dtype=float)
    for dof in range(manager.total_dofs):
        node_id, local, _ = manager.get_dof_info(dof)
        if node_id < 0:
            continue
        r = model.mesh.get_node(int(node_id)).coords()
        if local < 3:
            applied_force[local] += external[dof]
            applied_moment += np.cross(r, np.eye(3)[local] * external[dof])
        else:
            applied_moment[local - 3] += external[dof]
    return {
        "reaction_force_N": [float(v) for v in reaction_force],
        "applied_force_N": [float(v) for v in applied_force],
        "force_balance_N": [float(a + b) for a, b in zip(reaction_force, applied_force)],
        "reaction_moment_Nm": [float(v) for v in reaction_moment],
        "applied_moment_Nm": [float(v) for v in applied_moment],
        "moment_balance_Nm": [float(a + b) for a, b in zip(reaction_moment, applied_moment)],
        "force_balance_norm_N": float(np.linalg.norm(reaction_force + applied_force)),
        "moment_balance_norm_Nm": float(np.linalg.norm(reaction_moment + applied_moment)),
        "force_balance_tolerance_N": FORCE_BALANCE_TOLERANCE,
        "moment_balance_tolerance_Nm": MOMENT_BALANCE_TOLERANCE,
    }


def _runtime_identity() -> dict:
    import anyfesolver
    import scipy

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "anyfesolver_version": str(anyfesolver.__version__),
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "command": list(sys.argv),
    }


def _verify_materials(model: FEModel) -> None:
    for element in model.mesh.elements.values():
        material = model.get_material(element.material_name)
        if (
            abs(material.elastic_modulus - ELASTIC_MODULUS) > 0.0
            or abs(material.poisson_ratio - POISSON_RATIO) > 0.0
            or abs(material.density - DENSITY) > 0.0
            or float(material.yield_stress) != 0.0
            or material.hardening_curve is not None
        ):
            raise RuntimeError(
                f"material {element.material_name!r} is not the declared elastic case material"
            )


def _verify_initial_state(payload: dict) -> None:
    """Fail closed on any prospective acceptance criterion (never weaken)."""
    sphere = payload["sphere"]
    geometry = sphere["geometry"]
    if geometry["node_count"] != 162 or geometry["triangle_count"] != 320:
        raise RuntimeError("fixed sphere case data has the wrong inventory")
    if not geometry["closed"] or geometry["edges_with_two_faces"] != geometry["edge_count"]:
        raise RuntimeError("fixed sphere skin is not closed with two faces per edge")
    if not geometry["outward_winding"]:
        raise RuntimeError("fixed sphere skin has inward-wound triangles")
    if geometry["max_radius_deviation_m"] > RADIUS_TOLERANCE:
        raise RuntimeError("fixed sphere radius deviation exceeds 1e-12 m")
    if sphere["supports"] != "none":
        raise RuntimeError("the released sphere must be unconstrained")
    faceted_mass = DENSITY * SPHERE_THICKNESS * geometry["faceted_area_m2"]
    for axis, mass in enumerate(sphere["masses_kg"]):
        if abs(mass - faceted_mass) > RELATIVE_MASS_TOLERANCE * faceted_mass:
            raise RuntimeError(
                f"sphere translation mass axis {axis}: {mass!r} disagrees with "
                f"rho*t*faceted_area {faceted_mass!r}"
            )
    panel = payload["panel"]
    for axis, mass in enumerate(panel["masses_kg"]):
        if abs(mass - TARGET_PANEL_MASS) > RELATIVE_MASS_TOLERANCE * TARGET_PANEL_MASS:
            raise RuntimeError(
                f"panel translation mass axis {axis}: {mass!r} disagrees with "
                f"target {TARGET_PANEL_MASS!r}"
            )
    if panel["gravity_solve"]["status"] != "completed":
        raise RuntimeError(
            f"panel gravity equilibrium did not complete: {panel['gravity_solve']['status']!r}"
        )
    mesh = panel["mesh"]
    if mesh["bar_y_positions_m"] != [float(v) for v in BAR_Y_POSITIONS]:
        raise RuntimeError(
            f"physical bars are not at the declared y positions {BAR_Y_POSITIONS}: "
            f"{mesh['bar_y_positions_m']!r}"
        )
    if mesh["bar_centroid_z_values_m"] != [BAR_CENTROID_Z]:
        raise RuntimeError(
            f"beam neutral lines are not at the declared z={BAR_CENTROID_Z}: "
            f"{mesh['bar_centroid_z_values_m']!r}"
        )
    if mesh["plate_node_z_values_m"] != [0.0]:
        raise RuntimeError(
            f"plate midsurface nodes are not at z=0: {mesh['plate_node_z_values_m']!r}"
        )
    node_count = mesh["shell_node_count"] + mesh["beam_node_count"]
    if len(mesh["nodes"]) != node_count:
        raise RuntimeError("exported panel node coordinates are incomplete")
    element_count = (
        mesh["shell_element_count"]
        + mesh["beam_element_count"]
        + mesh["coupling_element_count"]
    )
    if len(mesh["elements"]) != element_count:
        raise RuntimeError("exported panel element connectivity is incomplete")
    half_length = PANEL_LENGTH / 2.0
    half_width = PANEL_WIDTH / 2.0
    for support in mesh["supports"]:
        for node_id in support["node_ids"]:
            x, y, _z = mesh["nodes"][str(node_id)]
            on_edge = (
                abs(abs(x) - half_length) < 1.0e-12
                or abs(abs(y) - half_width) < 1.0e-12
            )
            if not on_edge:
                raise RuntimeError(
                    f"support node {node_id} at ({x}, {y}) is not on a plate edge"
                )
    sphere_mesh = sphere["mesh"]
    if len(sphere_mesh["nodes"]) != sphere["geometry"]["node_count"]:
        raise RuntimeError("exported sphere node coordinates are incomplete")
    if len(sphere_mesh["elements"]) != sphere["geometry"]["triangle_count"]:
        raise RuntimeError("exported sphere element connectivity is incomplete")
    for coordinates in sphere_mesh["nodes"].values():
        radius = (sum(value ** 2 for value in coordinates)) ** 0.5
        if abs(radius - SPHERE_MIDSURFACE_RADIUS) > RADIUS_TOLERANCE:
            raise RuntimeError(
                f"exported sphere node radius {radius!r} deviates from "
                f"{SPHERE_MIDSURFACE_RADIUS!r}"
            )
    reactions = panel["reactions"]
    if reactions["max_relative_component_deviation"] > reactions["agreement_tolerance"]:
        raise RuntimeError(
            "recovered support resultants disagree with the solver-recorded "
            "resultants beyond the declared agreement tolerance"
        )
    equilibrium = panel["equilibrium"]
    if equilibrium["force_balance_norm_N"] > FORCE_BALANCE_TOLERANCE:
        raise RuntimeError("global force balance exceeds the declared tolerance")
    if equilibrium["moment_balance_norm_Nm"] > MOMENT_BALANCE_TOLERANCE:
        raise RuntimeError("global moment balance exceeds the declared tolerance")


def case_acceptance(panel_model: FEModel, sphere_model: FEModel,
                    sphere_geometry: dict, load_case: LoadCase,
                    result, displacements: np.ndarray) -> dict:
    """The one fixed mass/reaction/global-balance acceptance of the case.

    Computes the acceptance measurements from the single accepted gravity
    solve and fails closed on every fixed criterion (never weakened): raw
    assembled translation masses against the faceted sphere mass and the
    declared panel target, recovered support resultants against the
    solver-recorded resultants, and the global force and moment balance.
    Both the initial-state workflow and the curved-skin geometry workflow
    invoke this same acceptance; no second gravity solve is performed.
    """
    panel_masses = translation_masses(panel_model)
    sphere_masses = translation_masses(sphere_model)

    faceted_mass = DENSITY * SPHERE_THICKNESS * sphere_geometry["faceted_area_m2"]
    for axis, mass in enumerate(sphere_masses):
        if abs(mass - faceted_mass) > RELATIVE_MASS_TOLERANCE * faceted_mass:
            raise RuntimeError(
                f"sphere translation mass axis {axis}: {mass!r} disagrees with "
                f"rho*t*faceted_area {faceted_mass!r}"
            )
    for axis, mass in enumerate(panel_masses):
        if abs(mass - TARGET_PANEL_MASS) > RELATIVE_MASS_TOLERANCE * TARGET_PANEL_MASS:
            raise RuntimeError(
                f"panel translation mass axis {axis}: {mass!r} disagrees with "
                f"target {TARGET_PANEL_MASS!r}"
            )

    recovery = support_reaction_recovery(panel_model, load_case, displacements)
    balance = _global_balance(panel_model, recovery)

    recorded = result.steps[-1].support_reactions if result.steps else {}
    solver_resultants = {
        str(name): [float(value) for value in values]
        for name, values in recorded.items()
    }
    recovered_resultants = {
        str(name): [float(value) for value in values]
        for name, values in recovery["named_resultants"].items()
    }
    if set(solver_resultants) != set(recovered_resultants):
        raise RuntimeError(
            "recovered support names do not match the solver-recorded supports: "
            f"{sorted(recovered_resultants)} vs {sorted(solver_resultants)}"
        )
    max_resultant_deviation = 0.0
    for name, recorded_values in solver_resultants.items():
        for component, (recorded_value, recovered_value) in enumerate(
            zip(recorded_values, recovered_resultants[name])
        ):
            deviation = abs(recovered_value - recorded_value)
            max_resultant_deviation = max(
                max_resultant_deviation,
                deviation / max(1.0, abs(recorded_value)),
            )
    if max_resultant_deviation > REACTION_AGREEMENT_TOLERANCE:
        raise RuntimeError(
            "recovered support resultants disagree with the solver-recorded "
            f"resultants (max relative deviation {max_resultant_deviation!r} "
            f"exceeds {REACTION_AGREEMENT_TOLERANCE!r})"
        )
    reaction_check = {
        "recovery_law": (
            "public corotational work-conjugate element force response "
            "(same law as the solve) with rotating-MPC slave-to-master "
            "imbalance transfer"
        ),
        "recovered_named_resultants": recovered_resultants,
        "solver_recorded_named_resultants": solver_resultants,
        "max_relative_component_deviation": float(max_resultant_deviation),
        "agreement_tolerance": REACTION_AGREEMENT_TOLERANCE,
    }
    if balance["force_balance_norm_N"] > FORCE_BALANCE_TOLERANCE:
        raise RuntimeError("global force balance exceeds the declared tolerance")
    if balance["moment_balance_norm_Nm"] > MOMENT_BALANCE_TOLERANCE:
        raise RuntimeError("global moment balance exceeds the declared tolerance")
    return {
        "panel_masses_kg": panel_masses,
        "sphere_masses_kg": sphere_masses,
        "sphere_faceted_mass_kg": faceted_mass,
        "reaction_check": reaction_check,
        "equilibrium": balance,
    }


def prepare_case_models():
    """Build both delivered models and solve the panel gravity state once.

    This is the single existing gravity/model preparation shared by the
    initial-state example and the curved-skin geometry example: it builds the
    clamped stiffened panel and the unconstrained elastic sphere, verifies
    the declared elastic material on both, solves the panel-only gravity
    equilibrium with the qualified nonlinear static caller and returns the
    models with the accepted displacement, retaining the reference
    coordinates.  The sphere is never solved.
    """
    panel_model = build_panel_model()
    sphere_model, sphere_geometry = build_sphere_model()

    _verify_materials(panel_model)
    _verify_materials(sphere_model)

    load_case, result = solve_panel_gravity(panel_model)
    if result.status != "completed":
        raise RuntimeError(
            f"panel gravity equilibrium did not complete: {result.status!r}; "
            f"failure_reason={result.failure_reason!r}"
        )
    displacements = np.asarray(result.displacements, dtype=float)
    return panel_model, sphere_model, sphere_geometry, load_case, result, displacements


def prepare_initial_state() -> dict:
    """Build the case, solve the panel-only gravity state and verify it."""
    panel_model, sphere_model, sphere_geometry, load_case, result, displacements = (
        prepare_case_models()
    )

    acceptance = case_acceptance(
        panel_model, sphere_model, sphere_geometry, load_case, result,
        displacements)
    panel_masses = acceptance["panel_masses_kg"]
    sphere_masses = acceptance["sphere_masses_kg"]
    balance = acceptance["equilibrium"]
    reaction_check = acceptance["reaction_check"]
    solver_resultants = reaction_check["solver_recorded_named_resultants"]

    inventory = _panel_inventory(panel_model)
    centre_node = min(
        panel_model.mesh.nodes.values(),
        key=lambda node: abs(node.x) + abs(node.y),
    )
    centre_top_z = PLATE_TOP_Z + float(displacements[centre_node.dofs[2]])
    sphere_centre_z = centre_top_z + DROP_CLEARANCE + SPHERE_OUTSIDE_RADIUS

    nodal_fields = {}
    for node_id, node in panel_model.mesh.nodes.items():
        nodal_fields[str(node_id)] = [
            float(displacements[dof]) for dof in node.dofs
        ]

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "impact_status": IMPACT_STATUS,
        "case": {
            "panel": {
                "length_m": PANEL_LENGTH,
                "width_m": PANEL_WIDTH,
                "plate_thickness_m": PLATE_THICKNESS,
                "plate_midsurface_z_m": 0.0,
                "plate_top_z_m": PLATE_TOP_Z,
                "support": "all six DOFs clamped on the four plate edges",
                "shell_divisions": [SHELL_DIVISIONS, SHELL_DIVISIONS],
            },
            "stiffeners": {
                "type": "upright flat bar represented by eccentric elastic beams",
                "count": NUM_BARS,
                "spacing_m": BAR_SPACING,
                "height_m": BAR_HEIGHT,
                "thickness_m": BAR_THICKNESS,
                "centroid_z_m": BAR_CENTROID_Z,
                "bottom_z_m": BAR_BOTTOM_Z,
                "beam_segments_per_bar": BEAM_DIVISIONS,
                "section_construction": (
                    'canonical StiffenerCrossSection.from_geometry("T-bar", '
                    "0.150, 0.010, 0.0, 0.0): zero-flange rectangular upright "
                    "web; physical 150 x 10 mm flat bar, not a flanged T"
                ),
                "placement_note": (
                    "the mesher centroid parameter -0.080 m locates only the "
                    "neutral beam line and tie eccentricity; the physical "
                    "height/thickness above are the case geometry"
                ),
                "section": {
                    "area_m2": BAR_AREA,
                    "Iy_m4": BAR_IY,
                    "Iz_m4": BAR_IZ,
                    "J_m4": BAR_J,
                    "torsion_modulus_m3": BAR_TORSION_MODULUS,
                },
                "connection": "interpolated beam-shell MPC, rotating eccentricity",
            },
            "sphere": {
                "outside_diameter_m": 2.0 * (SPHERE_MIDSURFACE_RADIUS + SPHERE_THICKNESS / 2.0),
                "wall_thickness_m": SPHERE_THICKNESS,
                "midsurface_radius_m": SPHERE_MIDSURFACE_RADIUS,
                "representation": "elastic S3 shell skin, all DOFs deformable",
            },
            "material": {
                "elastic_modulus_Pa": ELASTIC_MODULUS,
                "poisson_ratio": POISSON_RATIO,
                "density_kg_m3": DENSITY,
                "yield_stress_Pa": 0.0,
                "hardening_curve": None,
                "plasticity": "none",
            },
            "gravity_m_s2": GRAVITY,
            "drop_target": {
                "clearance_m": DROP_CLEARANCE,
                "executed": False,
                "note": "actual gravity impact retained as target; unsupported here",
            },
        },
        "panel": {
            "mesh": inventory,
            "masses_kg": panel_masses,
            "mass_reference_kg": {
                "plate": PLATE_MASS,
                "bars": BARS_MASS,
                "target_total": TARGET_PANEL_MASS,
            },
            "gravity_solve": {
                "settings": dict(SOLVE_SETTINGS),
                "status": str(result.status),
                "steps": [
                    {
                        "step_index": int(step.step_index),
                        "load_factor": float(step.load_factor),
                        "iterations": int(step.iterations),
                        "residual_norm": float(step.residual_norm),
                    }
                    for step in result.steps
                ],
                "final_load_factor": float(result.load_factor),
                "solver_support_reaction_resultants": solver_resultants,
            },
            "equilibrium": balance,
            "reactions": reaction_check,
            "nodal_fields": nodal_fields,
        },
        "sphere": {
            "mesh": {
                "node_count": sphere_geometry["node_count"],
                "triangle_count": sphere_geometry["triangle_count"],
                "edge_count": sphere_geometry["edge_count"],
                "element_formulation": "e4-pl-s3-v2d (default three-node shell)",
                "coordinate_convention": (
                    "local sphere coordinates about the sphere centre; the "
                    "pose below places the body in the case frame"
                ),
                "nodes": {
                    str(node_id): [float(x), float(y), float(z)]
                    for node_id, (x, y, z) in enumerate(
                        np.asarray(SPHERE_NODES, dtype=float), 1
                    )
                },
                "elements": {
                    str(element_id): {
                        "type": "Shell",
                        "body": "sphere_skin",
                        "node_ids": [int(n) for n in triangle],
                    }
                    for element_id, triangle in enumerate(
                        np.asarray(SPHERE_TRIANGLES, dtype=int), 1
                    )
                },
            },
            "geometry": sphere_geometry,
            "masses_kg": sphere_masses,
            "mass_reference_kg": {
                "faceted_rho_t_area": DENSITY * SPHERE_THICKNESS * sphere_geometry["faceted_area_m2"],
                "smooth_midsurface_shell": SMOOTH_MIDSURFACE_SHELL_MASS,
                "hollow_solid_volume": HOLLOW_SOLID_VOLUME_MASS,
                "note": (
                    "the faceted FE skin mass is the assembled model mass; the "
                    "smooth midsurface-shell and hollow-solid values are "
                    "distinct analytic references, not assembly targets"
                ),
            },
            "supports": "none",
            "pose": {
                "centre_m": [0.0, 0.0, float(sphere_centre_z)],
                "translational_velocity_m_s": [0.0, 0.0, 0.0],
                "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                "stress_state": "stress-free (never solved, zero displacement)",
            },
        },
        "placement": {
            "convention": (
                "10 m clearance between the lowest point of the sphere's "
                "outer surface (centre minus outside radius 0.500 m) and "
                "the equilibrated panel top surface at the impact centre "
                "(x=y=0); the midsurface skin sits 5 mm higher"
            ),
            "panel_top_z_at_impact_centre_m": float(centre_top_z),
            "clearance_m": DROP_CLEARANCE,
            "sphere_outside_radius_m": SPHERE_OUTSIDE_RADIUS,
            "sphere_centre_z_m": float(sphere_centre_z),
        },
        "references": {
            "analytic_planning_not_simulated": {
                "free_fall_time_s": 1.427843123,
                "impact_speed_m_s": 14.007141036,
                "smooth_shell_energy_kJ": 23.7115,
                "note": "ideal free fall through the 10 m clearance; not a solver result",
            }
        },
        "identity": _runtime_identity(),
    }
    _verify_initial_state(payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the initial-state JSON document to this path")
    parser.add_argument("--impact", action="store_true",
                        help="request the gravity impact (refused: unsupported)")
    args = parser.parse_args(argv)
    if args.impact:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "impact_refused",
            "impact_status": IMPACT_STATUS,
            "reason": IMPACT_REFUSAL,
        }), file=sys.stdout)
        print(IMPACT_REFUSAL, file=sys.stderr)
        return 3
    payload = prepare_initial_state()
    document = json.dumps(payload, indent=2)
    if args.output is None:
        print(document)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "impact_status": payload["impact_status"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
