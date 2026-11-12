"""Independent work and ordinary static lifecycle for explicit beam-rigid sums."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
                       ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
                       RigidContactBody, solve_static_nonlinear, validate_general_contact)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.multiple_beam_rigid import MultipleBeamRigidManager, PROFILE


def fixture(*, cylinder=False):
    model = FEModel("one_round_b3_two_obstacles")
    model.add_material("elastic", 1.e3, .3, density=1.)
    for node, x in enumerate((-.5, 0., .5), 1):
        model.add_node(node, x, .015, .015)
    model.add_element(1, QuadraticBeamElement(
        1, [1, 2, 3], "elastic", {"area": 1.e-3, "Iy": 1.e-4, "Iz": 1.e-4, "J": 1.e-4}))
    model.add_boundary_condition(BoundaryCondition(
        "ends", (1, 3), {dof: 0. for dof in ("ux", "uy", "uz", "rx", "ry", "rz")}))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,), {dof: 0. for dof in ("ux", "rx", "ry", "rz")}))
    body_y = (RigidContactBody("y_tool", "cylinder", (0., -.005, .015),
                              normal=(1., 0., 0.), radius=.005) if cylinder else
              RigidContactBody("y_tool", "plane", (0., 0., 0.), normal=(0., 1., 0.)))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=.01, beam_end_caps="none"),),
        rigid_bodies=(body_y, RigidContactBody("z_tool", "plane", (0., 0., 0.), normal=(0., 0., 1.))),
        interactions=(ContactInteraction("y_contact", "tube", "y_tool", None, .001,
                                         line_stiffness=1.e5),
                      ContactInteraction("z_contact", "tube", "z_tool", None, .0008,
                                         line_stiffness=2.e5)))
    settings = ContactSolverSettings(pair_motion="general", line_integration_relative_tolerance=1.e-8)
    return model, definition, settings


def test_independent_two_plane_potential_work_and_both_tangents():
    model, definition, settings = fixture()
    manager = MultipleBeamRigidManager(model, definition, settings)
    dofs = model.mesh.dof_manager.get_node_dofs(2)
    q = np.zeros(model.mesh.dof_manager.total_dofs)
    q[dofs[1]], q[dofs[2]] = -.006, -.0057
    direction = np.zeros_like(q)
    direction[dofs[1]], direction[dofs[2]] = .7, -.3
    residual, tangent, summary = manager.evaluate(q)

    def independent_potential(displacement):
        energy = 0.
        for axis, stiffness in ((1, 1.e5), (2, 2.e5)):
            u = float(displacement[dofs[axis]])
            # Original B3 scalar field: Nmid=4s(1-s), reference length=1m.
            root = np.sqrt(1+.005/u)
            lo, hi = .5*(1-root), .5*(1+root)
            energy += quad(lambda s: .5*stiffness*(.005+4*s*(1-s)*u)**2,
                           lo, hi, epsabs=1.e-13, epsrel=1.e-12)[0]
        return energy

    assert abs(summary.to_dict()["algorithmic_contact_potential"]-independent_potential(q)) < 1.e-12
    h = 1.e-7
    virtual_work = (independent_potential(q+h*direction)-independent_potential(q-h*direction))/(2*h)
    assert abs(virtual_work-residual@direction)/abs(virtual_work) < 1.e-7
    plus = manager.evaluate(q+h*direction, tangent=False)[0]
    minus = manager.evaluate(q-h*direction, tangent=False)[0]
    np.testing.assert_allclose((plus-minus)/(2*h), tangent@direction, rtol=1.e-5, atol=1.e-7)
    data = summary.to_dict()
    np.testing.assert_allclose(np.array(data["beam_force"])+data["rigid_force"], 0., atol=1.e-12)
    np.testing.assert_allclose(np.array(data["beam_moment"])+data["rigid_moment"], 0., atol=1.e-12)
    assert len(data["pairs"]) == 2 and summary.gap_tolerance_met is False
    # The z gap passes its budget while the y pair independently fails its own.
    assert data["pairs"][0]["contact"]["gap_tolerance_met"] is False
    assert data["pairs"][1]["contact"]["gap_tolerance_met"] is True
    assert not manager.swept_increment_admissible(np.zeros_like(q), q, 0., 1.)


@pytest.mark.parametrize("bad", ["duplicate", "candidate_cap", "moving", "unused", "caps"])
def test_invalid_definitions_fail_without_dropping_contacts(bad):
    model, definition, settings = fixture()
    if bad == "duplicate":
        definition = replace(definition, interactions=definition.interactions+(
            replace(definition.interactions[0], name="duplicate_alias"),))
    elif bad == "candidate_cap":
        settings = replace(settings, maximum_candidate_pairs=1)
    elif bad == "moving":
        definition = replace(definition, rigid_bodies=(
            replace(definition.rigid_bodies[0], path_translation=(0., -.001, 0.)), definition.rigid_bodies[1]))
    elif bad == "unused":
        definition = replace(definition, surfaces=definition.surfaces+(
            replace(definition.surfaces[0], name="unreferenced"),))
    else:
        definition = replace(definition, surfaces=(replace(definition.surfaces[0], beam_end_caps="hemisphere"),))
    assert not validate_general_contact(model, definition, settings).valid
    with pytest.raises((ValueError, MemoryError)):
        MultipleBeamRigidManager(model, definition, settings)


def test_fixed_cylinder_and_plane_share_existing_kernels():
    model, definition, settings = fixture(cylinder=True)
    manager = MultipleBeamRigidManager(model, definition, settings)
    q = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        q[dofs[1]] = -.006
    residual, tangent, result = manager.evaluate(q)
    fields = result.to_dict()["pairs"]
    # Entire reference line has radius .014m; R+r=.015m, k=1e5N/m2.
    assert fields[0]["contact"]["beam_force"][1] == pytest.approx(100., rel=1.e-12)
    assert fields[0]["contact"]["line_contact_energy"] == pytest.approx(.05, rel=1.e-12)
    assert fields[1]["contact"]["beam_force"] == [0., 0., 0.]
    assert np.isfinite(residual).all() and np.isfinite(tangent.data).all()


def test_coupled_two_plane_force_control_split_restart_and_properties():
    model, definition, settings = fixture()
    load = LoadCase("two_component_dead_force")
    load.add_nodal_load(2, forces=(0., -10., -12.))
    kwargs = dict(contact=definition, contact_settings=settings,
                  num_steps=4, max_iterations=22, tolerance=1.e-8, emit_restart_checkpoint=True)
    full = solve_static_nonlinear(model, load, **kwargs)
    assert full.status == "completed", full.failure_reason
    summary = full.info["general_contact"]
    assert summary["profile"] == PROFILE and summary["gap_tolerance_met"]
    assert all(pair["contact"]["active_line_length"] > 0 for pair in summary["pairs"])
    assert [item["line_stiffness_N_per_m2"] for item in summary["pair_enforcement"]] == [1.e5, 2.e5]
    assert [item["gap_tolerance_m"] for item in summary["pair_enforcement"]] == [.001, .0008]
    partial_model, _, _ = fixture()
    partial = solve_static_nonlinear(partial_model, load, max_load_factor=.5, **kwargs)
    assert partial.status == "completed", partial.failure_reason
    fresh_model, _, _ = fixture()
    resumed = solve_static_nonlinear(fresh_model, load, restart_checkpoint=json.loads(
        json.dumps(partial.to_restart_checkpoint())), **kwargs)
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements, rtol=1.e-7, atol=1.e-10)
    changed = replace(definition, interactions=(replace(definition.interactions[0], line_stiffness=1.01e5),
                                                definition.interactions[1]))
    kwargs["contact"] = changed
    incompatible_model, _, _ = fixture()
    with pytest.raises(ValueError, match="restart|contract|incompatible"):
        solve_static_nonlinear(incompatible_model, load, restart_checkpoint=partial.to_restart_checkpoint(), **kwargs)
    print("MULTIPLE_BEAM_RIGID_RESPONSE "+json.dumps({"contact": summary,
          "full_steps": len(full.steps), "split_steps": len(partial.steps),
          "resumed_steps": len(resumed.steps)}, sort_keys=True), flush=True)


def test_displacement_control_preserves_two_active_pair_gates_and_reactions():
    model, definition, settings = fixture()
    load = LoadCase("two_component_control_reference")
    load.add_nodal_load(2, forces=(0., -10., -12.))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement", displacement_control=DisplacementControl(
            node_id=2, dof="uy", target_displacement=-.0058),
        num_steps=4, max_iterations=22, tolerance=1.e-8, emit_restart_checkpoint=True)
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["gap_tolerance_met"]
    assert all(pair["contact"]["active_line_length"] > 0 for pair in summary["pairs"])
    assert "normal_stiffness" not in summary
    assert result.steps[-1].control_value == pytest.approx(-.0058, abs=1.e-12)
    np.testing.assert_allclose(np.array(summary["beam_force"])+summary["rigid_force"], 0., atol=1.e-12)
    # Reported residual_norm has force units; the tolerance is relative to a
    # positive force scale. Use a fixed declared 10/12N reference here, stricter
    # than the driver's max(1, |lambda|)*reference at this final lambda > 1.
    assert result.steps[-1].residual_norm/np.hypot(10., 12.) <= 1.e-8
    print("MULTIPLE_BEAM_DISPLACEMENT_RESPONSE "+json.dumps(summary, sort_keys=True), flush=True)


def test_named_pair_order_is_deterministic_and_unused_external_features_rejected():
    model, definition, settings = fixture()
    manager = MultipleBeamRigidManager(model, definition, settings)
    reordered = replace(definition, interactions=tuple(reversed(definition.interactions)),
                        rigid_bodies=tuple(reversed(definition.rigid_bodies)))
    other = MultipleBeamRigidManager(model, reordered, settings)
    assert manager.restart_contract() == other.restart_contract()
    q = np.zeros(model.mesh.dof_manager.total_dofs)
    middle = model.mesh.dof_manager.get_node_dofs(2)
    q[middle[1]], q[middle[2]] = -.0054, -.0053
    a, K, summary = manager.evaluate(q)
    b, L, alternate = other.evaluate(q)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(K.toarray(), L.toarray())
    assert summary.to_dict() == alternate.to_dict()


def test_development_example_runs_with_the_declared_profile():
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, "examples/general_contact_multiple_round_beam_tools.py"],
                         cwd=root, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    record = json.loads(run.stdout)
    assert record["status"] == "completed" and record["profile"] == PROFILE
    assert record["contact"]["gap_tolerance_met"]
    assert len(record["contact"]["pairs"]) == 2
