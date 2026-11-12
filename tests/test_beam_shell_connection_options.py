"""Public connection selection; routine contract checks, not qualification."""
import copy
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import (BoundaryCondition, CoupledBeamShellElement, FEModel,
                       InterpolatedBeamShellMPCElement, LoadCase, MeshConfig,
                       PanelGeometry, generate_stiffened_panel_mesh, solve_static_nonlinear)
from anyfesolver.elements import BeamElement
import anyfesolver.mesh_gen as mesh_gen


def test_documented_complete_example_executes():
    document = Path(__file__).resolve().parents[1]/"docs"/"BEAM_SHELL_CONNECTION.md"
    code = document.read_text(encoding="utf-8").split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(code, str(document), "exec"), namespace)
    assert namespace["result"].status == "completed"


def _connection(kind, **options):
    if kind == "direct":
        return CoupledBeamShellElement(10, 3, 1, "steel", eccentricity=np.array([0., 0., .1]), **options)
    return InterpolatedBeamShellMPCElement(10, 3, [1, 2], np.array([.5, .5]),
                                          np.array([0., 0., .1]), "steel", **options)


@pytest.mark.parametrize("kind", ("direct", "interpolated"))
@pytest.mark.parametrize("setting", (None, True, False))
def test_constructor_default_and_explicit_connection_laws(kind, setting):
    model = FEModel("connection")
    for node, xyz in ((1, (0., 0., 0.)), (2, (1., 0., 0.)), (3, (.5, 0., .1))):
        model.add_node(node, *xyz)
    element = _connection(kind, **({} if setting is None else {"rotating_eccentricity": setting}))
    assert element.rotating_eccentricity is (setting is not False)
    assert (element.get_nonlinear_mpc_descriptor(model.mesh) is not None) is (setting is not False)
    assert len(element.get_mpc_constraints(model.mesh)) == 6
    element.eccentricity[:] = 0.
    assert element.get_nonlinear_mpc_descriptor(model.mesh) is None


@pytest.mark.parametrize("kind", ("direct", "interpolated"))
def test_legacy_missing_option_defaults_true_but_invalid_mutation_refused(kind):
    model = FEModel("legacy")
    for node, xyz in ((1,(0.,0.,0.)), (2,(1.,0.,0.)), (3,(.5,0.,.1))):
        model.add_node(node, *xyz)
    element = _connection(kind)
    del element.rotating_eccentricity
    assert element.get_nonlinear_mpc_descriptor(model.mesh) is not None
    element.rotating_eccentricity = False
    assert element.get_nonlinear_mpc_descriptor(model.mesh) is None
    element.rotating_eccentricity = 1
    with pytest.raises(TypeError, match="rotating_eccentricity must be a bool"):
        element.get_nonlinear_mpc_descriptor(model.mesh)


def test_legacy_positional_config_and_constructors_keep_argument_order():
    config = MeshConfig(2, 3, 4, True, 0., False, 1e-5, "a", "b", "c", False, True)
    assert config.align_mesh_to_stiffeners is True and config.rotating_eccentricity is True
    offset = np.array([0.,0.,.1])
    assert CoupledBeamShellElement(1, 2, 3, "steel", 0., offset).rotating_eccentricity is True
    assert InterpolatedBeamShellMPCElement(1, 2, [3], np.array([1.]), offset, "steel").rotating_eccentricity is True


@pytest.mark.parametrize("bad", (0, 1, "false", None, np.bool_(True)))
def test_non_bool_options_rejected_before_generation_without_mutation(monkeypatch, bad):
    weights = np.array([.5, .5]); offset = np.array([0., 0., .1])
    before = (weights.copy(), offset.copy())
    for factory in (lambda: CoupledBeamShellElement(10, 3, 1, eccentricity=offset, rotating_eccentricity=bad),
                    lambda: InterpolatedBeamShellMPCElement(10, 3, [1, 2], weights, offset, rotating_eccentricity=bad),
                    lambda: MeshConfig(rotating_eccentricity=bad)):
        with pytest.raises(TypeError, match="rotating_eccentricity must be a bool"):
            factory()
    np.testing.assert_array_equal(weights, before[0]); np.testing.assert_array_equal(offset, before[1])
    config = MeshConfig(); config.rotating_eccentricity = bad
    panel = PanelGeometry(length=1., width=.5, plate_thickness=.01)
    panel_before, config_before = copy.deepcopy(vars(panel)), copy.deepcopy(vars(config))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid solver option reached neutral meshing")
    monkeypatch.setattr(mesh_gen, "neutral_stiffened_panel_mesh", forbidden)
    with pytest.raises(TypeError, match="rotating_eccentricity must be a bool"):
        generate_stiffened_panel_mesh(panel, config)
    assert vars(panel) == panel_before and vars(config) == config_before


@pytest.mark.parametrize("setting", (True, False))
def test_generated_public_panel_forwards_solver_option_only(setting):
    panel = PanelGeometry(length=1., width=.5, plate_thickness=.01,
        stiffener_type="T-bar", stiffener_height=.1, stiffener_web_thickness=.005,
        stiffener_flange_width=.05, stiffener_flange_thickness=.005, num_stiffeners=1)
    config = MeshConfig(shell_num_divisions_x=2, shell_num_divisions_y=2,
                        rotating_eccentricity=setting)
    before = copy.deepcopy((vars(panel), vars(config)))
    model = generate_stiffened_panel_mesh(panel, config)
    connections = [e for e in model.mesh.elements.values() if isinstance(e, InterpolatedBeamShellMPCElement)]
    assert connections and all(e.rotating_eccentricity is setting for e in connections)
    assert all((e.get_nonlinear_mpc_descriptor(model.mesh) is not None) is setting for e in connections)
    assert not hasattr(mesh_gen._neutral_config(config), "rotating_eccentricity")
    assert (vars(panel), vars(config)) == before
    assert fields(MeshConfig)[-1].name == "rotating_eccentricity"


@pytest.mark.parametrize("kind", ("direct", "interpolated"))
@pytest.mark.parametrize("setting", (True, False))
def test_public_elastic_solve_obeys_selected_lever_arm(kind, setting):
    model = FEModel("prescribed_elastic_beam")
    model.add_material("steel", 210e9, .3)
    for nid, xyz in ((1,(0.,0.,0.)), (2,(1.,0.,0.)), (3,(0.,0.,.1)), (4,(1.,0.,.1))):
        model.add_node(nid, *xyz)
    section = dict(area=.001, Iy=1e-6, Iz=1e-6, J=1e-6)
    model.add_element(1, BeamElement(1, [3,4], "steel", section))
    angle = .3
    rotation = np.array([[np.cos(angle),0.,np.sin(angle)], [0.,1.,0.], [-np.sin(angle),0.,np.cos(angle)]])
    for master, slave in ((1,3), (2,4)):
        if kind == "direct":
            connection = CoupledBeamShellElement(10+slave, slave, master, "steel", eccentricity=np.array([0.,0.,.1]), rotating_eccentricity=setting)
        else:
            connection = InterpolatedBeamShellMPCElement(10+slave, slave, [master], np.array([1.]), np.array([0.,0.,.1]), "steel", rotating_eccentricity=setting)
        model.add_element(10+slave, connection)
        xyz = model.mesh.get_node(master).coords()
        displacement = (rotation-np.eye(3))@xyz
        conditions = dict(zip(("ux","uy","uz","rx","ry","rz"), (*displacement,0.,angle,0.)))
        if master == 2:
            del conditions["ux"]  # Retain one physical elastic equilibrium DOF.
        model.add_boundary_condition(BoundaryCondition(str(master), [master], conditions))
    model.apply_boundary_conditions()
    load = LoadCase("probe"); load.add_nodal_load(4, forces=(0.,0.,-1.)); model.add_load_case(load)
    result = solve_static_nonlinear(model, load, kinematics="corotational", corotational_tangent="auto", num_steps=1, max_iterations=10, emit_restart_checkpoint=False)
    assert result.status == "completed"
    if setting:
        assert result.info["corotational_tangent"] == "work_conjugate"
    assert np.all(np.isfinite(result.displacements))
    for master, slave in ((1,3), (2,4)):
        dofs = model.mesh.dof_manager.get_node_dofs
        expected = result.displacements[dofs(master)[:3]] + ((rotation-np.eye(3))@np.array([0.,0.,.1]) if setting else np.array([angle*.1,0.,0.]))
        np.testing.assert_allclose(result.displacements[dofs(slave)[:3]], expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.displacements[dofs(slave)[3:]], [0.,angle,0.], atol=1e-14)
