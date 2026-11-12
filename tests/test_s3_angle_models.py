"""Assembled-model regression checks for reduced-angle S3 (V2D) meshes.

Small, fast versions of the families run by ``scripts/qualify_s3_angles.py``;
see docs/S3_ANGLE_EXTENSION.md for the full campaign and its evidence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qualify_s3_angles as campaign  # noqa: E402
from anyfesolver import LoadCase, assemble_load_vector  # noqa: E402
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement  # noqa: E402


@pytest.mark.parametrize("angle", (20.0, 15.0))
@pytest.mark.parametrize("family", ("right", "obtuse"))
def test_family_meshes_hit_the_requested_angle_exactly(family: str, angle: float) -> None:
    mesh = campaign.s3_family_mesh(family, angle, 0, target=(1.0, 1.0), base_count=4)
    stats = campaign.mesh_statistics(mesh)
    assert stats["s3_fraction"] == 1.0
    assert stats["s3_min_angle_deg"] == pytest.approx(angle, abs=1.0e-9)
    expected_max = 90.0 if family == "right" else 180.0 - 2.0 * angle
    assert stats["s3_max_angle_deg"] == pytest.approx(expected_max, abs=1.0e-9)
    model = campaign.build_model(mesh, 0.01)
    assert all(
        type(element) is NativeParityE4PLS3V2DShellElement
        for element in model.mesh.elements.values()
    )


@pytest.mark.parametrize("pattern", ("isolated", "chain", "boundary", "cluster", "quarter"))
def test_mixed_layouts_are_conformal_and_keep_the_angle(pattern: str) -> None:
    mesh = campaign.mixed_mesh(pattern, 15.0, 1, base_count=4)
    stats = campaign.mesh_statistics(mesh)
    assert 0.0 < stats["s3_fraction"] < 0.5
    assert stats["s3_min_angle_deg"] == pytest.approx(15.0, abs=1.0e-9)
    # Conformity: every interior edge is shared by exactly two elements.
    edges: dict[tuple[int, int], int] = {}
    for _kind, nodes in mesh.elements:
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) <= {1, 2}
    boundary = [key for key, count in edges.items() if count == 1]
    perimeter = sum(
        float(np.hypot(*np.subtract(mesh.nodes[a], mesh.nodes[b]))) for a, b in boundary
    )
    assert perimeter == pytest.approx(2.0 * (mesh.lx + mesh.ly), rel=1.0e-12)


def test_nodal_pressure_equivalent_matches_public_load_assembly() -> None:
    mesh = campaign.mixed_mesh("chain", 20.0, 0, base_count=4)
    model = campaign.build_model(mesh, 0.01)
    load = LoadCase("pressure")
    for element_id in model.mesh.elements:
        load.add_pressure_load(element_id, 1.0e3)
    public, _info = assemble_load_vector(model, load)
    direct = load.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material)
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(public))
    assert float(np.sum(np.asarray(public)[2::6])) == pytest.approx(1.0e3 * mesh.lx * mesh.ly, rel=1e-12)


def test_direct_buckling_matches_public_solver_route_at_low_angle() -> None:
    mesh = campaign.s3_family_mesh("right", 20.0, 0, target=(0.5, 0.5), base_count=2)
    direct = campaign.run_plate_buckling(mesh, 0.01)
    public = campaign.run_plate_buckling(mesh, 0.01, public_route=True)
    assert public["solver_status"] == "ok"
    assert direct["buckling_factor"] == pytest.approx(public["buckling_factor"], rel=1.0e-10)


@pytest.mark.parametrize("angle", (20.0, 15.0))
@pytest.mark.parametrize("family", ("right", "obtuse"))
def test_plate_bending_converges_at_reduced_angle(family: str, angle: float) -> None:
    errors = [
        campaign.run_plate_static(
            campaign.s3_family_mesh(family, angle, level, target=(1.0, 1.0), base_count=4), 0.01
        )
        for level in (0, 1)
    ]
    deflection = [row["w_centre_error"] for row in errors]
    moment = [row["s3_moment_l2_error"] for row in errors]
    # Second-order deflection and first-order raw moments, as at 45 deg.
    assert deflection[1] < deflection[0] / 3.0
    assert deflection[1] < 0.02
    assert moment[1] < 0.6 * moment[0]


@pytest.mark.parametrize("narrow", ("x", "y"))
def test_membrane_manufactured_solution_converges_at_15_degrees(narrow: str) -> None:
    rows = [
        campaign.run_membrane(
            campaign.s3_family_mesh(
                "right", 15.0, level, target=(4.0, 1.0), base_count=8 if narrow == "y" else 2, narrow=narrow
            )
        )
        for level in (0, 1)
    ]
    rate = np.log2(rows[0]["energy_norm_error"] / rows[1]["energy_norm_error"])
    assert rate > 0.9
    assert rows[1]["max_nodal_displacement_error"] < 1.0e-3


@pytest.mark.parametrize("layout", ("pure", "mixed"))
def test_direct_modal_matches_public_solver_route_at_low_angle(layout: str) -> None:
    mesh = (
        campaign.s3_family_mesh("obtuse", 15.0, 0, target=(0.5, 0.5), base_count=2)
        if layout == "pure"
        else campaign.mixed_mesh("chain", 15.0, 0, base_count=4)
    )
    direct = campaign.run_plate_modal(mesh, 0.01)
    public = campaign.run_plate_modal(mesh, 0.01, public_route=True)
    assert public["solver_status"] == "ok"
    np.testing.assert_allclose(direct["frequencies_hz"], public["frequencies_hz"], rtol=1.0e-8)


def test_corotational_restart_is_bitwise_equivalent_on_a_15_degree_mesh() -> None:
    """Restart/identity: V2D state carries no geometry-admission field, so a
    reduced-angle model must checkpoint and resume exactly like any other."""

    from anyfesolver import FEModel, create_shell_element
    from anyfesolver.boundary import FixedSupport
    from anyfesolver.e4_pl_s3_v2d_state import canonical_json_bytes
    from anyfesolver.nonlinear_static import solve_static_nonlinear

    # Two 15/75/90 triangles: restart checkpoint normalization currently runs
    # a model-wide lifecycle guard per JSON node (quadratic in model size, see
    # docs/S3_ANGLE_EXTENSION.md section 3), so keep this model minimal.
    mesh = campaign.s3_family_mesh(
        "right", 15.0, 0, target=(1.0, math.tan(math.radians(15.0))), base_count=1, narrow="y"
    )
    assert campaign.mesh_statistics(mesh)["s3_min_angle_deg"] == pytest.approx(15.0, abs=1e-9)

    def model_and_load():
        model = FEModel("s3-angle-restart")
        model.add_material("steel", campaign.E, 0.0, density=campaign.DENSITY)
        for node_id, (x, y) in mesh.nodes.items():
            model.add_node(node_id, x, y, 0.0)
        for element_id, (_kind, nodes) in enumerate(mesh.elements, start=1):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id, list(nodes), "steel", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)
                ),
            )
        sides = campaign._boundary_nodes(mesh)
        model.add_boundary_condition(FixedSupport("root", sides["x0"]))
        rigidity = campaign.E * mesh.ly * 0.01**3 / 12.0
        load = LoadCase("tip-moment")
        tip = sides["x1"]
        for node_id in tip:
            load.add_nodal_load(node_id, moments=[0.0, -0.3 * rigidity / mesh.lx / len(tip), 0.0])
        return model, load

    options = dict(kinematics="corotational", corotational_tangent="consistent", emit_restart_checkpoint=True)
    model, load = model_and_load()
    full = solve_static_nonlinear(model, load, num_steps=2, **options)
    model, load = model_and_load()
    first = solve_static_nonlinear(model, load, max_load_factor=0.5, num_steps=1, **options)
    model, load = model_and_load()
    resumed = solve_static_nonlinear(
        model,
        load,
        max_load_factor=1.0,
        num_steps=1,
        restart_checkpoint=first.restart_checkpoint_bytes(),
        kinematics="corotational",
        corotational_tangent="consistent",
        emit_restart_checkpoint=True,
    )
    assert full.status == first.status == resumed.status == "completed"
    np.testing.assert_array_equal(full.displacements, resumed.displacements)
    for element_id in full.element_states:
        assert canonical_json_bytes(full.element_states[element_id]) == canonical_json_bytes(
            resumed.element_states[element_id]
        )
    # Physical sanity on this one-cell strip: tip lifts, within 10% of the arc.
    dm = model.mesh.dof_manager
    tip = campaign._boundary_nodes(mesh)["x1"]
    w = float(np.mean([full.displacements[dm.get_node_dofs(n)[2]] for n in tip]))
    assert w == pytest.approx(mesh.lx / 0.3 * (1.0 - math.cos(0.3)), rel=0.1)
