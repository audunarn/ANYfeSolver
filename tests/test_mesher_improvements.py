"""ANYfeSolver integration tests for meshes now owned by ANYmesher."""

from __future__ import annotations

import numpy as np

import anymesher
import anyfesolver.mesh_gen as mesh_gen
from anyfesolver.elements import BeamElement, ShellElement


def _panel() -> mesh_gen.PanelGeometry:
    return mesh_gen.PanelGeometry(
        length=2.4,
        width=1.8,
        plate_thickness=0.012,
        stiffener_type="T-bar",
        stiffener_spacing=0.55,
        stiffener_height=0.18,
        stiffener_web_thickness=0.008,
        stiffener_flange_width=0.10,
        stiffener_flange_thickness=0.012,
        num_stiffeners=2,
        in_plane_support="Integrated",
        rotational_support="CL",
    )


def _neutral_panel(panel: mesh_gen.PanelGeometry) -> anymesher.StiffenedPanel:
    return anymesher.StiffenedPanel(
        length=panel.length,
        width=panel.width,
        plate_thickness=panel.plate_thickness,
        stiffener_type=panel.stiffener_type,
        stiffener_spacing=panel.stiffener_spacing,
        stiffener_height=panel.stiffener_height,
        stiffener_web_thickness=panel.stiffener_web_thickness,
        stiffener_flange_width=panel.stiffener_flange_width,
        stiffener_flange_thickness=panel.stiffener_flange_thickness,
        num_stiffeners=panel.num_stiffeners,
    )


def test_stiffened_panel_adapter_preserves_neutral_numbering_and_couplings() -> None:
    panel = _panel()
    config = mesh_gen.MeshConfig(
        shell_num_divisions_x=5,
        shell_num_divisions_y=4,
        beam_num_divisions=3,
        use_8node_shells=True,
        align_mesh_to_stiffeners=True,
    )
    neutral = anymesher.stiffened_panel_mesh(
        _neutral_panel(panel),
        anymesher.PanelMeshConfig(
            shell_num_divisions_x=config.shell_num_divisions_x,
            shell_num_divisions_y=config.shell_num_divisions_y,
            beam_num_divisions=config.beam_num_divisions,
            use_coupling_elements=config.use_coupling_elements,
            tolerance=config.tolerance,
            use_8node_shells=config.use_8node_shells,
            align_mesh_to_stiffeners=config.align_mesh_to_stiffeners,
        ),
    )
    model = mesh_gen.generate_stiffened_panel_mesh(panel, config)

    assert list(model.mesh.nodes) == list(neutral.nodes)
    for node_id, coordinates in neutral.nodes.items():
        np.testing.assert_array_equal(model.mesh.nodes[node_id].coords(), coordinates)

    shells = {
        element_id: tuple(element.node_ids)
        for element_id, element in model.mesh.elements.items()
        if isinstance(element, ShellElement)
    }
    beams = {
        element_id: tuple(element.node_ids)
        for element_id, element in model.mesh.elements.items()
        if isinstance(element, BeamElement)
    }
    assert shells == {**neutral.quads, **neutral.tris}
    assert beams == neutral.beams

    couplings = {
        element_id: element
        for element_id, element in model.mesh.elements.items()
        if isinstance(element, mesh_gen.InterpolatedBeamShellMPCElement)
    }
    assert list(couplings) == list(neutral.couplings)
    for element_id, record in neutral.couplings.items():
        element = couplings[element_id]
        assert element.beam_node_id == record.beam_node
        assert tuple(element.shell_node_ids) == record.plate_nodes
        np.testing.assert_array_equal(element.shape_weights, record.weights)
        np.testing.assert_array_equal(element.eccentricity, record.eccentricity)

    edges = anymesher.panel_edge_nodes(neutral)
    integrated = next(
        condition
        for condition in model.boundary_conditions
        if condition.name == "Integrated_edge_translations"
    )
    assert integrated.node_ids == edges["all"]


def test_simple_panel_adapter_calls_anymesher(monkeypatch) -> None:
    calls = 0
    original = mesh_gen.neutral_simple_panel_mesh

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mesh_gen, "neutral_simple_panel_mesh", counted)
    model = mesh_gen.generate_simple_panel_mesh(2.0, 1.0, 0.01, 4, 3)
    assert calls == 1
    assert len(model.mesh.nodes) == 20


def test_mesh_quality_keeps_the_legacy_dict_facade() -> None:
    model = mesh_gen.generate_simple_panel_mesh(10.0, 1.0, 0.01, 1, 1)
    report = mesh_gen.verify_mesh_quality(model)
    assert report["num_shell_elements"] == 1
    assert report["max_aspect_ratio"] == 10.0
    assert report["warnings"]
