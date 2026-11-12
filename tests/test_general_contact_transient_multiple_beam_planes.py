"""Shared Newmark with explicit additive beam-plane contributors."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
                       ContactSolverSettings, ContactSurface, LoadCase, RigidContactBody)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import active_contact_manager
from anyfesolver.general_contact.multiple_beam_rigid import MultipleBeamRigidManager
from anyfesolver.general_contact.transient import (
    ContactNodalLoadHistory, ContactTransientSolverControls,
    _beam_complete_path_gaps, solve_transient_contact,
)

CONTROLS = ContactTransientSolverControls(allow_classical_beam_dev=True,
                                        allow_multiple_beam_planes_dev=True)


def fixture(*, quadratic=False, orthogonal=False, reverse=False):
    model = FEModel("multiple_plane_impact")
    model.add_material("elastic", 1e3, .3, density=1.)
    xs = (-.5, 0., .5) if quadratic else (-.5, .5)
    for node, x in enumerate(xs, 1):
        model.add_node(node, x, .015, .015)
    element = QuadraticBeamElement if quadratic else BeamElement
    model.add_element(1, element(1, list(model.mesh.nodes), "elastic",
        {"area": 1e-3, "Iy": 1e-4, "Iz": 1e-4, "J": 1e-4}))
    model.add_boundary_condition(BoundaryCondition("guide", list(model.mesh.nodes),
        {name: 0. for name in (("ux", "rx", "ry", "rz") if orthogonal
                              else ("ux", "uy", "rx", "ry", "rz"))}))
    bodies = (RigidContactBody("first", "plane", (0., 0., 0.),
                             normal=(0., 1., 0.) if orthogonal else (0., 0., 1.)),
              RigidContactBody("second", "plane", (0., 0., 0.), normal=(0., 0., 1.)))
    pairs = (ContactInteraction("first_pair", "tube", "first", None, .002, line_stiffness=3e3),
             ContactInteraction("second_pair", "tube", "second", None, .002, line_stiffness=7e3))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=.01, beam_end_caps="none"),),
        rigid_bodies=bodies[::-1] if reverse else bodies,
        interactions=pairs[::-1] if reverse else pairs)
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes:
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        initial[dofs[2]] = -.25 if orthogonal else -.5
        if orthogonal:
            initial[dofs[1]] = -.5
    return model, definition, ContactSolverSettings(pair_motion="general"), initial


def impact(*, end=.013, checkpoint=None, emit=False, history=None, **options):
    model, definition, settings, initial = fixture(**options)
    load = None
    if history is not None:
        load = LoadCase("normal_load")
        weights = (1/6, 2/3, 1/6) if options.get("quadratic") else (.5, .5)
        for node, weight in zip(model.mesh.nodes, weights):
            load.add_nodal_load(node, forces=(0., 0., -1e-4*weight))
    result = solve_transient_contact(model, definition,
        TransientConfig(.0001, end, initial_velocity=initial), contact_settings=settings,
        solver_controls=CONTROLS, restart_checkpoint=checkpoint,
        emit_restart_checkpoint=emit, base_load_case=load, nodal_load_history=history)
    return model, result


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.parametrize("orthogonal", [False, True])
def test_multiple_plane_scalar_newmark_reactions_impulse_and_opening(quadratic, orthogonal):
    model, result = impact(quadratic=quadratic, orthogonal=orthogonal,
                           end=.023 if orthogonal else .013)
    assert "MULTIPLE_FIXED_PLANES" in result.profile
    # Independent translational modes: each total directional mass is rho*A*L=.001 kg.
    axes = ((1, -.5, 3e3), (2, -.25, 7e3)) if orthogonal else ((2, -.5, 1e4),)
    for axis, velocity, stiffness in axes:
        gap, acceleration, previous_force = .005, 0., 0.
        dofs = [model.mesh.dof_manager.get_node_dofs(node)[axis] for node in model.mesh.nodes]
        for i, dt in enumerate(np.diff(result.times), 1):
            predictor = gap+dt*velocity+.25*dt*dt*acceleration
            new_gap = predictor if predictor >= 0. else predictor/(1+.25*dt*dt*stiffness/.001)
            new_a = (new_gap-predictor)/(.25*dt*dt)
            new_v = velocity+.5*dt*(acceleration+new_a)
            np.testing.assert_allclose(.005+result.displacements[i, dofs], new_gap, atol=1e-10, rtol=0.)
            np.testing.assert_allclose(result.velocities[i, dofs], new_v, atol=1e-8, rtol=0.)
            force = max(-stiffness*new_gap, 0.)
            data = result.contact[i].to_dict()
            assert data["beam_force"][axis] == pytest.approx(force, abs=1e-7)
            assert .001*(new_v-velocity) == pytest.approx(.5*dt*(force+previous_force), abs=1e-11)
            for name, pair in result.contact[i].pairs:
                expected = max(-dict(first_pair=3e3, second_pair=7e3)[name]*new_gap, 0.)
                if not orthogonal or (name == "first_pair") == (axis == 1):
                    assert pair.beam_force[axis] == pytest.approx(expected, abs=1e-7)
                np.testing.assert_allclose(np.array(pair.beam_force)+pair.rigid_force, 0., atol=1e-12)
                np.testing.assert_allclose(np.array(pair.beam_moment)+pair.rigid_moment, 0., atol=1e-12)
            gap, velocity, acceleration, previous_force = new_gap, new_v, new_a, force
    assert any(any(pair.active_line_cells for _, pair in c.pairs) for c in result.contact)
    assert all(not pair.active_line_cells for _, pair in result.contact[-1].pairs)
    assert active_contact_manager() is None


def test_multiple_active_history_restart_order_and_definition_binding():
    history = ContactNodalLoadHistory(((0., 0.), (.004, 1.), (.013, 1.)))
    _, direct = impact(quadratic=True, history=history)
    _, prefix = impact(quadratic=True, history=history, end=.0102, emit=True)
    assert all(pair.active_line_cells for _, pair in prefix.contact[-1].pairs)
    _, resumed = impact(quadratic=True, history=history, checkpoint=prefix.restart_checkpoint, reverse=True)
    for field in ("times", "displacements", "velocities", "accelerations", "swept_minimum_gaps", "external_nodal_work_j"):
        np.testing.assert_allclose(np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])),
                                   getattr(direct, field), atol=1e-9, rtol=0., err_msg=field)
    model, definition, settings, initial = fixture(quadratic=True)
    changed = replace(definition, interactions=(replace(definition.interactions[0], line_stiffness=4e3),
                                                definition.interactions[1]))
    load = LoadCase("normal_load")
    for node, weight in zip(model.mesh.nodes, (1/6, 2/3, 1/6)):
        load.add_nodal_load(node, forces=(0., 0., -1e-4*weight))
    with pytest.raises(ValueError, match="checkpoint .*incompatible"):
        solve_transient_contact(model, changed, TransientConfig(.0001, .013, initial_velocity=initial),
            contact_settings=settings, solver_controls=CONTROLS, base_load_case=load,
            nodal_load_history=history, restart_checkpoint=prefix.restart_checkpoint)


def test_multiple_path_uses_pair_budgets_and_one_shared_split_cap():
    from anyfesolver.general_contact.beam_temporal_bounds import beam_hermite_gap_bounds, BeamHermiteGapBoundsError
    model, definition, settings, initial = fixture(orthogonal=True, quadratic=True)
    definition = replace(definition, interactions=(replace(definition.interactions[0], gap_tolerance=.001),
                                                   definition.interactions[1]))
    manager = MultipleBeamRigidManager(model, definition, settings)
    q = np.zeros_like(initial)
    for node in model.mesh.nodes:
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        q[dofs[1]], q[dofs[2]] = -.0055, -.0065
    minimum, admitted = _beam_complete_path_gaps(manager, q, q, q*0, q*0, .001, CONTROLS, lambda: None)
    assert minimum < -.001 and admitted  # larger penetration belongs to the larger budget
    for node in model.mesh.nodes:
        q[model.mesh.dof_manager.get_node_dofs(node)[1]] = -.0061
    assert not _beam_complete_path_gaps(manager, q, q, q*0, q*0, .001, CONTROLS, lambda: None)[1]
    model, definition, settings, initial = fixture(quadratic=True)
    manager = MultipleBeamRigidManager(model, definition, settings)
    q = np.zeros_like(initial)
    for node, u in zip(model.mesh.nodes, (-.004, -.006, -.003)):
        q[model.mesh.dof_manager.get_node_dofs(node)[2]] = u
    child = manager.pairs[0][1]
    bounds = beam_hermite_gap_bounds(child, q, q, q*0, q*0, .001, tolerance_m=1e-10, maximum_subdivisions=512)
    assert bounds.subdivisions > 0
    with pytest.raises(BeamHermiteGapBoundsError, match="unresolved"):
        _beam_complete_path_gaps(manager, q, q, q*0, q*0, .001,
                                replace(CONTROLS, maximum_beam_swept_subdivisions=2*bounds.subdivisions-1), lambda: None)
    assert _beam_complete_path_gaps(manager, q, q, q*0, q*0, .001,
        replace(CONTROLS, maximum_beam_swept_subdivisions=2*bounds.subdivisions), lambda: None)[1]


def test_multiple_admission_opt_in_duplicate_and_nonplane_rejection():
    model, definition, settings, initial = fixture()
    config = TransientConfig(.0001, .001, initial_velocity=initial)
    with pytest.raises(ValueError, match="one shell surface"):
        solve_transient_contact(model, definition, config, contact_settings=settings,
                               solver_controls=replace(CONTROLS, allow_multiple_beam_planes_dev=False))
    duplicate = replace(definition, interactions=(definition.interactions[0],
                                                 replace(definition.interactions[0], name="alias")))
    with pytest.raises(ValueError, match="duplicate"):
        solve_transient_contact(model, duplicate, config, contact_settings=settings, solver_controls=CONTROLS)
    radial = replace(definition, rigid_bodies=(replace(definition.rigid_bodies[0], kind="sphere", radius=.001, normal=None),
                                             definition.rigid_bodies[1]))
    with pytest.raises(ValueError, match="fixed penalty plane"):
        solve_transient_contact(model, radial, config,
            contact_settings=replace(settings, line_integration_relative_tolerance=1e-8), solver_controls=CONTROLS)
    with pytest.raises(ValueError, match="classical beam opt-in"):
        replace(CONTROLS, allow_classical_beam_dev=False)


@pytest.mark.parametrize("cancel", [False, True])
def test_second_pair_path_failure_does_not_commit_partial_candidate(monkeypatch, cancel):
    import anyfesolver.general_contact.transient as driver
    import anyfesolver.general_contact.beam_temporal_bounds as geometry
    from anyfesolver.control import CancellationToken, SolveCancelled
    token = CancellationToken()
    commits, calls = [], []
    commit, bounds = driver.commit_state_candidate, geometry.beam_hermite_gap_bounds
    def record(*args, **kwargs):
        commits.append(1)
        return commit(*args, **kwargs)
    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            if cancel:
                token.cancel()
                kwargs["check_cancel"]()
            raise RuntimeError("injected second-pair bound failure")
        return bounds(*args, **kwargs)
    monkeypatch.setattr(driver, "commit_state_candidate", record)
    monkeypatch.setattr(geometry, "beam_hermite_gap_bounds", fail)
    model, definition, settings, initial = fixture()
    with pytest.raises(SolveCancelled if cancel else RuntimeError):
        solve_transient_contact(model, definition, TransientConfig(.0001, .001, initial_velocity=initial),
            contact_settings=settings, solver_controls=CONTROLS, cancellation_token=token)
    assert len(commits) == 1  # initial consistency only; neither pair accepts the rejected step
    assert active_contact_manager() is None


def test_union_coverage_rejects_unrepresented_structural_beam():
    model, definition, settings, initial = fixture()
    model.add_node(3, -.5, .2, .015)
    model.add_node(4, .5, .2, .015)
    model.add_element(2, BeamElement(2, [3, 4], "elastic",
        {"area": 1e-3, "Iy": 1e-4, "Iz": 1e-4, "J": 1e-4}))
    with pytest.raises(ValueError, match="every structural element"):
        solve_transient_contact(model, definition, TransientConfig(.0001, .001,
            initial_velocity=np.zeros(model.mesh.dof_manager.total_dofs)),
            contact_settings=settings, solver_controls=CONTROLS)


def test_public_driver_rejects_a_pair_exceeding_its_own_gap_budget():
    model, definition, settings, initial = fixture()
    definition = replace(definition, interactions=(replace(definition.interactions[0], gap_tolerance=1e-8),
                                                   definition.interactions[1]))
    with pytest.raises(RuntimeError, match="step failed.*committed state retained"):
        solve_transient_contact(model, definition, TransientConfig(.001, .013, initial_velocity=initial),
            contact_settings=settings, solver_controls=replace(CONTROLS, maximum_cutbacks=0))
    assert active_contact_manager() is None


def test_disjoint_surfaces_cover_all_beams_without_duplicate_work():
    model, definition, settings, _ = fixture()
    for node, x in ((3, -.5), (4, .5)):
        model.add_node(node, x, .2, .015)
    model.add_element(2, BeamElement(2, [3, 4], "elastic",
        {"area": 1e-3, "Iy": 1e-4, "Iz": 1e-4, "J": 1e-4}))
    model.add_boundary_condition(BoundaryCondition("second_guide", [3, 4],
        {name: 0. for name in ("ux", "uy", "rx", "ry", "rz")}))
    definition = replace(definition,
        surfaces=(definition.surfaces[0], ContactSurface("other", (2,), beam_radius=.01, beam_end_caps="none")),
        interactions=(definition.interactions[0], replace(definition.interactions[1], surface="other")))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes:
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.5
    result = solve_transient_contact(model, definition, TransientConfig(.0001, .0102, initial_velocity=initial),
        contact_settings=settings, solver_controls=CONTROLS)
    pairs = dict(result.contact[-1].pairs)
    for name, nodes in (("first_pair", [1, 2]), ("second_pair", [3, 4])):
        gap = .005+result.displacements[-1, model.mesh.dof_manager.get_node_dofs(nodes[0])[2]]
        assert pairs[name].candidate_cells == 1
        assert pairs[name].beam_force[2] == pytest.approx(-dict(first_pair=3e3, second_pair=7e3)[name]*gap, abs=1e-7)
    assert result.displacements[-1, model.mesh.dof_manager.get_node_dofs(1)[2]] != pytest.approx(
        result.displacements[-1, model.mesh.dof_manager.get_node_dofs(3)[2]], abs=1e-7)
