"""Support balance and continuation for opt-in generalized dynamic reactions."""
from dataclasses import replace
import numpy as np
import pytest

from anyfesolver import (ContactTransientSolverControls, ContactNodalLoadHistory,
                       LoadCase, TransientConfig, solve_transient_contact)
from test_general_contact_transient_plane import _guided_plate, _constant_q4_load


def case(end=.014, *, reactions=True, checkpoint=None, emit=False, load=False):
    model, definition, velocity = _guided_plate()
    body = replace(definition.rigid_bodies[0], normal=(np.sin(1.e-4), 0., np.cos(1.e-4)))
    definition = replace(definition, rigid_bodies=(body,))
    result = solve_transient_contact(model, definition,
        TransientConfig(.001, end, initial_velocity=velocity),
        solver_controls=ContactTransientSolverControls(save_constraint_reactions_dev=reactions),
        base_load_case=_constant_q4_load() if load else None,
        restart_checkpoint=checkpoint, emit_restart_checkpoint=emit)
    return model, result


@pytest.fixture(scope='module')
def impact():
    return case()


def test_reactions_balance_oblique_contact_and_do_no_free_coordinate_work(impact):
    model, result = impact
    dofs = [model.mesh.dof_manager.get_node_dofs(n) for n in (1,2,3,4)]
    forces = result.constraint_reactions[:, [d[:3] for d in dofs]]
    assert result.constraint_reactions.shape == result.displacements.shape
    assert any(c.active_pressure_dofs for c in result.contact)
    rigid = np.array([c.rigid_force for c in result.contact])
    assert np.max(np.abs(rigid[:,0])) > 1.e-5
    np.testing.assert_allclose(forces[:,:,0].sum(axis=1), rigid[:,0], atol=2.e-10, rtol=1.e-8)
    np.testing.assert_allclose(forces[:,:,1].sum(axis=1), rigid[:,1], atol=2.e-10)
    # Each normal coordinate is free: its generalized support force must vanish.
    np.testing.assert_allclose(forces[:,:,2], 0., atol=2.e-8)


def test_reaction_output_preserves_dynamics_and_restarts(impact):
    _, full = impact
    _, historical = case(reactions=False)
    assert historical.constraint_reactions.shape == (0,0)
    np.testing.assert_array_equal(historical.displacements, full.displacements)
    np.testing.assert_array_equal(historical.accelerations, full.accelerations)
    _, first = case(.011, emit=True)
    _, resumed = case(checkpoint=first.restart_checkpoint)
    offset = len(first.times)-1
    np.testing.assert_allclose(resumed.constraint_reactions, full.constraint_reactions[offset:], atol=2.e-8)
    with pytest.raises(ValueError, match='checkpoint.*(analysis|incompatible)'):
        case(checkpoint=first.restart_checkpoint, reactions=False)


def test_external_nodal_load_is_subtracted_from_reactions():
    _, result = case(.004, load=True)
    assert all(c.active_pressure_dofs == 0 for c in result.contact)
    np.testing.assert_allclose(result.constraint_reactions, 0., atol=2.e-10)


def test_reaction_selection_requires_bool():
    with pytest.raises(ValueError, match='reaction.*boolean'):
        ContactTransientSolverControls(save_constraint_reactions_dev=1)


def test_mpc_constraint_reactions_balance_unequal_loads():
    model, definition, velocity = _guided_plate()
    z = [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (1,2,3,4)]
    for slave in z[1:]:
        model.add_constraint_equation(terms=((slave,1.),(z[0],-1.)))
    load = LoadCase('unequal normal load')
    load.add_nodal_load(1, forces=(0.,0.,-.005))
    result = solve_transient_contact(model, definition,
        TransientConfig(.001,.004,initial_velocity=np.zeros_like(velocity)),
        base_load_case=load,
        solver_controls=ContactTransientSolverControls(save_constraint_reactions_dev=True))
    # Total mass .05 kg, uniform acceleration -.1 m/s^2, row masses .0125 kg.
    expected = np.array([.00375,-.00125,-.00125,-.00125])
    np.testing.assert_allclose(result.constraint_reactions[:,z],
                               np.broadcast_to(expected,(len(result.times),4)), atol=2.e-11)
    np.testing.assert_allclose(result.constraint_reactions[:,z].sum(axis=1),0.,atol=2.e-11)


def test_load_history_and_sparse_output_retain_stage_support_forces():
    model, definition, velocity = _guided_plate()
    load = LoadCase('fixed tangential ramp')
    for node in (1,2,3,4):
        load.add_nodal_load(node, forces=(.005,0.,0.))
    history = ContactNodalLoadHistory(((0.,0.),(.004,1.)))
    result = solve_transient_contact(model, definition,
        TransientConfig(.001,.004,save_every=2,initial_velocity=np.zeros_like(velocity)),
        base_load_case=load, nodal_load_history=history,
        solver_controls=ContactTransientSolverControls(save_constraint_reactions_dev=True))
    x = [model.mesh.dof_manager.get_node_dofs(n)[0] for n in (1,2,3,4)]
    np.testing.assert_allclose(result.times,[0.,.002,.004],atol=1.e-14)
    expected = -.005*result.times/.004
    np.testing.assert_allclose(result.constraint_reactions[:,x],
                               np.broadcast_to(expected[:,None],(3,4)),atol=2.e-12)


def test_descriptor_reactions_preserve_free_algebraic_equilibrium():
    from test_general_contact_transient_plane import _guided_s3_descriptor_impact
    model, definition, velocity = _guided_s3_descriptor_impact(tilt=.002,span=.1)
    result = solve_transient_contact(model, definition,
        TransientConfig(.0002,.012,initial_velocity=velocity),
        solver_controls=ContactTransientSolverControls(allow_s3_descriptor_dev=True,
                                                        save_constraint_reactions_dev=True))
    dofs = [model.mesh.dof_manager.get_node_dofs(n) for n in (1,2,3)]
    free = [d[i] for d in dofs for i in (2,3,4,5)]
    assert result.contact[-1].active_pressure_dofs > 0
    np.testing.assert_allclose(result.constraint_reactions[:,free],0.,atol=1.e-7)
    rigid = np.asarray([c.rigid_force for c in result.contact])
    for coordinate in (0,1):
        np.testing.assert_allclose(result.constraint_reactions[:,[d[coordinate] for d in dofs]].sum(axis=1),
                                   rigid[:,coordinate],atol=1.e-8)


def test_free_body_reactions_export_only_structural_coordinates():
    from test_general_contact_transient_free_shell_sphere import impact, CONTROLS
    model, result = impact(end=.015,controls=replace(CONTROLS,save_constraint_reactions_dev=True))
    dofs = [model.mesh.dof_manager.get_node_dofs(n) for n in (1,2,3,4)]
    assert result.constraint_reactions.shape == result.displacements.shape
    assert any(c.active_pressure_dofs for c in result.contact)
    np.testing.assert_allclose(result.constraint_reactions[:,[d[2] for d in dofs]],0.,atol=1.e-9)
    assert np.max(np.abs(result.constraint_reactions[:,[d[0] for d in dofs]])) > 1.e-6
    for coordinate in (0,1):
        body_force = np.array([state['sphere'].contact_force_n[coordinate]
                               for state in result.free_rigid_bodies])
        np.testing.assert_allclose(result.constraint_reactions[:,[d[coordinate] for d in dofs]].sum(axis=1),
                                   body_force,atol=1.e-9)
