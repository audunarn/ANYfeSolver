"""Independent Q1 mortar work and explicit fixed-plane composition checks."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    ContactDefinition, ContactInteraction, ContactSolverSettings, ContactSurface,
    FEModel, RigidContactBody, create_shell_element, validate_general_contact,
)
from anyfesolver.general_contact.core import contact_scope
from anyfesolver.general_contact.multiple_shell_planes import (
    PROFILE, MultipleShellPlaneManager, validate_multiple_shell_planes,
)


def fixture(*, size=.02):
    model = FEModel('one-q4-two-fixed-planes')
    model.add_material('elastic', 1.e3, .3, density=1.)
    for node, (x, y) in enumerate(((-size/2, -size/2), (size/2, -size/2),
                                  (size/2, size/2), (-size/2, size/2)), 1):
        model.add_node(node, x, y, .1)
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], 'elastic', thickness=.004))
    n1, n2 = np.array([0., 0., 1.]), np.array([.6, 0., .8])
    centre_skin = np.array([0., 0., .098])
    definition = ContactDefinition(
        surfaces=(ContactSurface('skin', (1,), side='bottom'),),
        rigid_bodies=(RigidContactBody('floor', 'plane', tuple(centre_skin-.03*n1), tuple(n1)),
                      RigidContactBody('inclined', 'plane', tuple(centre_skin-.02*n2), tuple(n2))),
        interactions=(ContactInteraction('z_pair', 'skin', 'floor', 2.e5, .003),
                      ContactInteraction('inclined_pair', 'skin', 'inclined', 7.e5, .006)),
    )
    return model, definition, ContactSolverSettings()


def displacement(model):
    value = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3, 4):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        value[dofs[:3]] = [-.01, 0., -.035]
        value[dofs[4]] = .02
    return value


def independent(model, definition, value):
    # Closed bilinear Q1 mass/overlap matrix on the square. Each pressure
    # mode has support A/4; normalized mortar gaps are overlap@g / support.
    coordinates = np.array([model.mesh.nodes[node].coords() for node in (1, 2, 3, 4)])
    area = (coordinates[1, 0]-coordinates[0, 0])*(coordinates[3, 1]-coordinates[0, 1])
    overlap = area/36.*np.array([[4., 2., 1., 2.], [2., 4., 2., 1.],
                               [1., 2., 4., 2.], [2., 1., 2., 4.]])
    support = np.full(4, area/4.)
    n_dof = len(value)
    force, tangent = np.zeros(n_dof), np.zeros((n_dof, n_dof))
    energy = 0.
    physical_force, physical_moment = np.zeros(3), np.zeros(3)
    bodies = {body.name: body for body in definition.rigid_bodies}
    for pair in definition.interactions:
        body = bodies[pair.target]
        normal = np.array(body.normal)
        offset = -.002*np.array([0., 0., 1.])
        gradient = np.r_[normal, np.cross(offset, normal)]
        local_dofs = np.array([model.mesh.dof_manager.get_node_dofs(node)[:6] for node in (1, 2, 3, 4)])
        nodal_gap = (coordinates+offset-np.array(body.point)) @ normal+value[local_dofs] @ gradient
        normalized = overlap @ nodal_gap/support
        weighted_gradient = np.zeros((4, n_dof))
        for row in range(4):
            for node in range(4):
                weighted_gradient[row, local_dofs[node]] = overlap[row, node]*gradient
        closed = normalized < 0.
        force_pair = weighted_gradient.T @ (pair.normal_stiffness*np.minimum(normalized, 0.))
        force += force_pair
        tangent += weighted_gradient.T @ np.diag(pair.normal_stiffness*closed/support) @ weighted_gradient
        energy += .5*pair.normal_stiffness*np.dot(support, np.minimum(normalized, 0.)**2)
        for node in range(4):
            local = force_pair[local_dofs[node]]
            physical_force -= local[:3]
            physical_moment -= np.cross(coordinates[node]+value[local_dofs[node, :3]], local[:3])+local[3:]
    return force, tangent, energy, physical_force, physical_moment


def test_two_pair_independent_potential_tangent_and_wrench():
    model, definition, settings = fixture()
    manager = MultipleShellPlaneManager(model, definition, settings)
    value = displacement(model)
    force, tangent, result = manager.evaluate(value)
    expected_force, expected_tangent, energy, shell_force, shell_moment = independent(model, definition, value)
    np.testing.assert_allclose(force, expected_force, rtol=2.e-13, atol=1.e-13)
    np.testing.assert_allclose(tangent.toarray(), expected_tangent, rtol=3.e-13, atol=1.e-12)
    assert result.penalty_energy == pytest.approx(energy, rel=2.e-13, abs=1.e-15)
    np.testing.assert_allclose(result.shell_force, shell_force, rtol=2.e-13, atol=1.e-13)
    np.testing.assert_allclose(result.shell_moment, shell_moment, rtol=2.e-13, atol=1.e-13)
    np.testing.assert_allclose(np.array(result.shell_force)+result.rigid_force, 0., atol=1.e-14)
    np.testing.assert_allclose(np.array(result.shell_moment)+result.rigid_moment, 0., atol=1.e-14)
    assert result.active_pressure_dofs == 8
    assert not result.gap_tolerance_met
    assert [name for name, _ in result.pairs] == ['inclined_pair', 'z_pair']
    assert sparse.isspmatrix_csr(tangent)
    assert manager.evaluate(value, tangent=False)[1] is None
    fields = result.to_dict()
    assert fields['penalty_energy'] == result.penalty_energy
    assert fields['active_pressure_dofs'] == 8 and len(fields['pairs']) == 2
    json.dumps(fields, allow_nan=False)


def test_independent_virtual_work_and_full_tangent_with_both_pairs_active():
    model, definition, settings = fixture()
    manager = MultipleShellPlaneManager(model, definition, settings)
    value = displacement(model)
    direction = np.random.default_rng(231).normal(size=value.size)
    step = 1.e-7
    force, tangent, _ = manager.evaluate(value)
    plus_energy = independent(model, definition, value+step*direction)[2]
    minus_energy = independent(model, definition, value-step*direction)[2]
    assert (plus_energy-minus_energy)/(2*step) == pytest.approx(force @ direction, rel=2.e-8, abs=1.e-10)
    plus = manager.evaluate(value+step*direction, tangent=False)[0]
    minus = manager.evaluate(value-step*direction, tangent=False)[0]
    np.testing.assert_allclose((plus-minus)/(2*step), tangent @ direction, rtol=2.e-8, atol=1.e-8)


def test_pressure_row_features_and_restart_are_stable_under_definition_order():
    model, definition, settings = fixture()
    manager = MultipleShellPlaneManager(model, definition, settings)
    reordered = replace(definition, surfaces=tuple(reversed(definition.surfaces)),
        rigid_bodies=tuple(reversed(definition.rigid_bodies)), interactions=tuple(reversed(definition.interactions)))
    other = MultipleShellPlaneManager(model, reordered, settings)
    assert manager.pressure_features == tuple((name, node) for name in ('inclined_pair', 'z_pair') for node in (1, 2, 3, 4))
    np.testing.assert_array_equal(manager.support, other.support)
    np.testing.assert_array_equal(manager.reference_weighted_gap, other.reference_weighted_gap)
    np.testing.assert_array_equal(manager.gap_gradient.toarray(), other.gap_gradient.toarray())
    assert manager.restart_contract() == other.restart_contract()
    assert not hasattr(manager, 'stiffness') and not hasattr(manager, 'gap_tolerance')
    pair_settings = manager.summary_settings()['pair_enforcement']
    assert [record['surface_stiffness_N_per_m3'] for record in pair_settings] == [7.e5, 2.e5]
    assert [record['gap_tolerance_m'] for record in pair_settings] == [.006, .003]
    changed = MultipleShellPlaneManager(model, definition, replace(settings, maximum_swept_subdivisions=3))
    assert manager.restart_contract() != changed.restart_contract()
    json.dumps(manager.restart_contract(), allow_nan=False)


def test_all_small_positive_supports_are_retained_and_counted_per_pair():
    model, definition, settings = fixture(size=1.e-8)
    manager = MultipleShellPlaneManager(model, definition, replace(settings, maximum_patch_normal_dofs=8))
    assert manager.pressure_mode_count == 8 and manager.candidate_count == 2
    assert len(manager.support) == 8
    assert np.all(manager.support > 0.) and np.max(manager.support) < 3.e-17


def shared_face_fixture():
    model, definition, settings = fixture()
    model.add_node(5, .03, -.01, .1)
    model.add_node(6, .03, .01, .1)
    model.add_element(2, create_shell_element(2, [2, 5, 6, 3], 'elastic', thickness=.004))
    definition = replace(definition, surfaces=(ContactSurface('skin', (1, 2), side='bottom'),
        ContactSurface('single', (1,), side='bottom')), interactions=(
            definition.interactions[0], replace(definition.interactions[1], surface='single')))
    return model, definition, settings


def test_aggregate_face_and_distinct_per_pair_pressure_caps_are_exact():
    model, definition, settings = shared_face_fixture()
    accepted = replace(settings, maximum_candidate_pairs=3, maximum_patch_normal_dofs=10)
    manager = MultipleShellPlaneManager(model, definition, accepted)
    assert manager.candidate_count == 3 and manager.pressure_mode_count == 10
    assert manager.support.size == 10
    for controls in (replace(accepted, maximum_candidate_pairs=2),
                     replace(accepted, maximum_patch_normal_dofs=9)):
        assert not validate_multiple_shell_planes(model, definition, controls).valid
        with pytest.raises(MemoryError):
            MultipleShellPlaneManager(model, definition, controls)


@pytest.mark.parametrize('bad', ['duplicate', 'alias', 'unused_surface', 'unused_body',
    'moving', 'sphere', 'general', 'warped_setting', 'warped_geometry', 's3', 'beam_feature',
    'augmented', 'line_law', 'pointwise', 'one_pair', 'unsupported_control'])
def test_unsupported_definitions_fail_before_any_pair_is_dropped(bad):
    model, definition, settings = fixture()
    control = 'force'
    if bad == 'duplicate':
        definition = replace(definition, interactions=definition.interactions+(
            replace(definition.interactions[0], name='duplicate'),))
    elif bad == 'alias':
        definition = replace(definition, surfaces=definition.surfaces+(replace(definition.surfaces[0], name='alias'),),
            interactions=definition.interactions+(replace(definition.interactions[0], name='alias_pair', surface='alias'),))
    elif bad == 'unused_surface':
        definition = replace(definition, surfaces=definition.surfaces+(replace(definition.surfaces[0], name='unused'),))
    elif bad == 'unused_body':
        definition = replace(definition, rigid_bodies=definition.rigid_bodies+(replace(definition.rigid_bodies[0], name='unused'),))
    elif bad == 'moving':
        definition = replace(definition, rigid_bodies=(replace(definition.rigid_bodies[0], path_translation=(0., 0., .001)), definition.rigid_bodies[1]))
    elif bad == 'sphere':
        definition = replace(definition, rigid_bodies=(RigidContactBody('floor', 'sphere', (0., 0., 0.), radius=.01), definition.rigid_bodies[1]))
    elif bad == 'general':
        settings = replace(settings, pair_motion='general')
    elif bad == 'warped_setting':
        settings = replace(settings, warped_q4_plane=True)
    elif bad == 'warped_geometry':
        model.mesh.nodes[3].z += .001
    elif bad == 's3':
        model.mesh.elements[1] = create_shell_element(1, [1, 2, 3], 'elastic', thickness=.004,
                                                       reference_normal=(0., 0., 1.))
    elif bad == 'beam_feature':
        definition = replace(definition, surfaces=(ContactSurface('skin', (1,), beam_radius=.01),))
    elif bad == 'augmented':
        definition = replace(definition, interactions=(replace(definition.interactions[0], normal_law='augmented'), definition.interactions[1]))
    elif bad == 'line_law':
        definition = replace(definition, interactions=(replace(definition.interactions[0], normal_stiffness=None, line_stiffness=1.e5), definition.interactions[1]))
    elif bad == 'pointwise':
        settings = replace(settings, normal_discretization='pointwise_penalty')
    elif bad == 'one_pair':
        definition = replace(definition, interactions=(definition.interactions[0],))
    else:
        control = 'arc_length'
    assert not validate_multiple_shell_planes(model, definition, settings, analysis_control=control).valid
    with pytest.raises((ValueError, MemoryError)):
        MultipleShellPlaneManager(model, definition, settings, analysis_control=control)


def test_distinct_skin_sides_are_distinct_physical_pairs():
    model, definition, settings = fixture()
    top = ContactSurface('top_skin', (1,), side='top')
    definition = replace(definition, surfaces=definition.surfaces+(top,), interactions=definition.interactions+(
        replace(definition.interactions[0], name='top_pair', surface=top.name),))
    manager = MultipleShellPlaneManager(model, definition, settings)
    assert manager.candidate_count == 3 and manager.pressure_mode_count == 12


def test_fixed_linear_sweep_checks_all_pair_physical_gap_tolerances():
    model, definition, settings = fixture()
    manager = MultipleShellPlaneManager(model, definition, settings)
    start, end = np.zeros(manager.n_dof), displacement(model)
    assert not manager.swept_increment_admissible(start, end, 0., 1.)
    fraction = manager.admissible_step_fraction(start, end)
    assert 0. < fraction < 1.
    assert manager.swept_increment_admissible(start, start+fraction*(end-start), 0., 1.)
    assert all(child.evaluate(start+fraction*(end-start), tangent=False)[2].gap_tolerance_met for _, child in manager.pairs)
    with pytest.raises(ValueError, match='committed fixed-plane pose'):
        manager.swept_increment_admissible(end, start, 0., 1.)


def test_core_routes_explicit_multiple_shell_planes_and_rejects_initial_overclosure():
    model, definition, settings = fixture()
    assert validate_general_contact(model, definition, settings).profile == PROFILE
    with contact_scope(model, definition, settings) as manager:
        assert type(manager) is MultipleShellPlaneManager
    with pytest.raises(ValueError, match='initial contact overclosure'):
        MultipleShellPlaneManager(model, definition, settings, displacement(model))
