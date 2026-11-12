"""Independent shared-beam work and two actually deforming physical skins."""
from dataclasses import replace
import json, os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import root

from anyfesolver import (ContactSurface, ContactInteraction, RigidContactBody, DisplacementControl,
    CancellationToken, SolveCancelled, solve_static_nonlinear, validate_general_contact,
    report_general_contact_capability)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.general_contact.multiple_beam_s3_caps import MultipleBeamS3CapManager, PROFILE
from anyfesolver.general_contact.deformable_beam_s3_caps import DeformableBeamS3CapManager
from anyfesolver.general_contact.beam_hemisphere_plane import HemispherePlaneManager
from anyfesolver.general_contact.beam_hemisphere_sphere import HemisphereSphereManager
from examples.general_contact_multiple_beam_s3_caps import build, run
from test_general_contact_deformable_beam_s3_caps import scalar_total
from s3_von_karman_oracle import model_triangle_von_karman


def fixture(quadratic=False):
    model, definition, settings, load = build()
    if quadratic:
        model.add_node(9, .25, 0., .02005)
        model.mesh.elements[1] = QuadraticBeamElement(1, (1, 9, 2), 'beam',
            {'area': .001, 'Iy': 1.e-4, 'Iz': 1.e-4, 'J': 1.e-4})
    return model, definition, settings, load


def local_pose(model, quadratic=False):
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node, axis, value in ((1, 2, -.0002), (2, 2, -.0003), (5, 2, .0001),
                              (1, 1, -.00015), (2, 1, -.00025), (8, 1, -.00005)):
        u[model.mesh.dof_manager.get_node_dofs(node)[axis]] = value
    if quadratic:
        u[model.mesh.dof_manager.get_node_dofs(9)[1:3]] = (-.00018, -.00023)
    return u


def scalar_potential(manager, u):
    return sum(scalar_total(*child._positions(u), child) for _, _, _, child in manager.pairs)


def independent_four_potential(x):
    """Independent guided fixture: both moving normals include cross motion.

    Coordinates are tip-z, skin-z, tip-y, skin-y. Plane normals are
    (0,-wz,3)/Az and (0,3,wy)/Ay. Genuine root caps remain open; tip
    supports are on their spherical branches. No production geometry calls.
    """
    tz, wz, ty, wy = x
    Az, Ay = np.sqrt(9+wz*wz), np.sqrt(9+wy*wy)
    gz0 = (.03015-wz)/Az-.01; gz1 = gz0+(3*tz-wz*ty)/Az
    gy0 = (.03015-wy)/Ay-.01; gy1 = gy0+(3*ty+wy*tz)/Ay
    energy = 0.
    for a, b in ((gz0, gz1), (gy0, gy1)):
        if a.real < 0 and b.real < 0: integral = (a*a+a*b+b*b)/3
        elif b.real < 0: integral = b**3/(3*(b-a))
        elif a.real < 0: integral = a**3/(3*(a-b))
        else: integral = 0.
        energy += .5*1e5*.5*integral
        if b.real < 0: energy += .5*1e3*b*b
    return energy


def independent_four_gradient(x):
    # Complex step on the independent closed scalar integral, within fixed
    # smooth activation branches. This does not call Jet2 or the contact kernel.
    return np.array([independent_four_potential(np.asarray(x,complex)+1e-25j*np.eye(4)[i]).imag/1e-25
                     for i in range(4)])


def assert_reference_domain(x):
    tz, wz, ty, wy = x
    for A, w, shift in ((np.sqrt(9+wz*wz), wz, 3*tz-wz*ty),
                        (np.sqrt(9+wy*wy), wy, 3*ty+wy*tz)):
        root_gap = (.03015-w)/A-.01
        assert root_gap > 0 and shift < 0 and root_gap+shift/A < 0


@pytest.mark.parametrize('quadratic', (False, True))
def test_two_deforming_skins_independent_shared_beam_work_full_hessian_and_couples(quadratic):
    model, definition, settings, _ = fixture(quadratic)
    assert validate_general_contact(model, definition, settings).valid
    manager = MultipleBeamS3CapManager(model, definition, settings)
    u = local_pose(model, quadratic)
    force, K, result = manager.evaluate(u)
    fields = result.to_dict()
    assert fields['profile'] == PROFILE and fields['active_endpoints'] == 4
    assert len(fields['beam_features']) == 1 and len(fields['shell_features']) == 2
    assert fields['algorithmic_contact_potential'] == pytest.approx(scalar_potential(manager, u), rel=1e-11)
    d = np.zeros_like(u)
    for node in model.mesh.nodes:
        d[model.mesh.dof_manager.get_node_dofs(node)[:3]] = np.random.default_rng(node+902).normal(size=3)
    d /= np.linalg.norm(d)
    for h in (1e-7, 3e-8):
        fd = (scalar_potential(manager, u+h*d)-scalar_potential(manager, u-h*d))/(2*h)
        assert fd == pytest.approx(force@d, rel=1e-7, abs=1e-9)
        G = (manager.evaluate(u+h*d, tangent=False)[0]-manager.evaluate(u-h*d, tangent=False)[0])/(2*h)
        assert np.linalg.norm(G-K@d)/np.linalg.norm(K@d) < 1e-5
    features = fields['beam_features'] + fields['shell_features']
    np.testing.assert_allclose(sum(np.asarray(f['force']) for f in features), 0., atol=1e-10)
    np.testing.assert_allclose(sum(np.asarray(f['moment']) for f in features), 0., atol=1e-10)
    # The two distinct physical skin offsets retain their structural couples.
    for _, _, _, child in manager.pairs:
        G = child.evaluate(u)[0]
        for node in child.shell.node_ids:
            dofs = model.mesh.dof_manager.get_node_dofs(node)
            np.testing.assert_allclose(G[dofs[3:6]],
                np.cross(child.shell.physical_offset*child.shell.director, G[dofs[:3]]), atol=1e-12)
    assert not manager.swept_increment_admissible(np.zeros_like(u), u, 0., 1.)
    fraction = manager.admissible_step_fraction(np.zeros_like(u), u)
    assert 0 < fraction < 1 and manager.swept_increment_admissible(np.zeros_like(u), fraction*u, 0., fraction)


def test_actual_four_coordinate_response_both_targets_controls_and_split_restart():
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    model, definition, settings, load = fixture()
    common = dict(contact=definition, contact_settings=settings, num_steps=4,
        max_iterations=25, tolerance=1e-10, emit_restart_checkpoint=True)
    result = solve_static_nonlinear(model, load, **common)
    assert result.status == 'completed', result.failure_reason
    inds = np.array([model.mesh.dof_manager.get_node_dofs(n)[axis]
                     for n, axis in ((2, 2), (5, 2), (2, 1), (8, 1))])
    x = result.displacements[inds]
    assert x[1] < -1e-7 and x[3] < -1e-7
    manager = MultipleBeamS3CapManager(model, definition, settings)
    G, _, contact = manager.evaluate(result.displacements)
    independent = independent_four_gradient
    error = np.linalg.norm(G[inds]-independent(x))/np.linalg.norm(G[inds])
    assert error < 1e-6
    K, _ = assemble_stiffness_matrix(model)
    reduced = K[inds][:, inds].toarray()
    # Retarget to the adopted S3 VK law (precedent 706bf011c): add each skin's
    # independent membrane VK force (one normal coordinate free per skin:
    # node 5 uz on the z skin, node 8 uy on the y skin; fixture constants
    # E=1e5, nu=.3, t=.02) to the linear structural force.
    def membrane(y):
        z = model_triangle_von_karman(model, (3, 4, 5), (0., 0., 1.), (0., 0., y[1]), 1.e5, .3, .02)[1][2]
        w = model_triangle_von_karman(model, (6, 7, 8), (0., 1., 0.), (0., 0., y[3]), 1.e5, .3, .02)[1][2]
        return np.array((0., z, 0., w))
    reference = root(lambda y: reduced@y+membrane(y)+independent(y)-[-.001, 0., -.001, 0.], x, tol=1e-11)
    assert reference.success
    response_error = np.linalg.norm(reference.x-x)/np.linalg.norm(reference.x)
    assert response_error < 1e-3  # Bounded structural reference: linear plus independent VK membrane.
    assert_reference_domain(x); assert_reference_domain(reference.x)
    for pair in contact.to_dict()['pairs']:
        end_fields = pair['contact']['endpoint_fields']
        assert not end_fields[0]['active'] and end_fields[1]['active']
        assert end_fields[1]['branch'] == 'sphere'
    assert contact.gap_tolerance_met and result.info['general_contact']['candidate_pairs'] == 2
    assert len(result.info['general_contact']['pair_enforcement']) == 2
    record = os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:
        Path(record).with_name('multiple_s3_response.json').write_text(json.dumps({'scope': 'two fixed-director S3 interiors, bounded linear plus VK-membrane reference',
            'actual_m': x.tolist(), 'reference_m': reference.x.tolist(), 'response_relative_error': float(response_error),
            'contact_gradient_relative_error': float(error), 'minimum_gap_m': contact.minimum_gap,
            'gap_tolerance_m': 1e-5, 'contact': contact.to_dict()}, indent=2), encoding='utf-8')
    fresh, _, _, _ = fixture()
    controlled = solve_static_nonlinear(fresh, load, **{**common, 'num_steps': 32}, control='displacement',
        displacement_control=DisplacementControl(node_id=2, dof='uz', target_displacement=float(x[0])))
    assert controlled.status == 'completed', controlled.failure_reason
    np.testing.assert_allclose(controlled.displacements, result.displacements, rtol=1e-6, atol=1e-10)
    fresh, _, _, _ = fixture()
    prefix = solve_static_nonlinear(fresh, load, **{**common, 'num_steps': 2}, max_load_factor=.5)
    assert prefix.status == 'completed'
    checkpoint = json.loads(json.dumps(prefix.to_restart_checkpoint(), allow_nan=False))
    fresh, _, _, _ = fixture()
    resumed = solve_static_nonlinear(fresh, load, **{**common, 'num_steps': 2}, restart_checkpoint=checkpoint)
    assert resumed.status == 'completed', resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements, rtol=1e-6, atol=1e-10)
    for changed in (replace(settings, beam_shell_maximum_swept_intervals=127),
                    replace(settings, beam_end_cap_work='owned_point')):
        fresh, _, _, _ = fixture()
        with pytest.raises(ValueError, match='contact|restart|incompatible|unsupported'):
            solve_static_nonlinear(fresh, load, **{**common, 'contact_settings': changed}, restart_checkpoint=checkpoint)
    altered = replace(definition, interactions=definition.interactions[:1])
    fresh, _, _, _ = fixture()
    with pytest.raises(ValueError, match='contact|restart|incompatible|unsupported'):
        solve_static_nonlinear(fresh, load, **{**common, 'contact': altered}, restart_checkpoint=checkpoint)


@pytest.mark.parametrize('kind', ('plane', 'sphere'))
def test_matching_optional_rigid_factory_wrench_path_and_effective_restart(kind):
    model, definition, settings, _ = fixture()
    body = (RigidContactBody('tool', 'plane', (-.01005, 0., 0.), (1., 0., 0.)) if kind == 'plane'
            else RigidContactBody('tool', 'sphere', (-.11005, 0., .02005), radius=.1))
    definition = replace(definition, rigid_bodies=(body,), interactions=definition.interactions+
        (ContactInteraction('tool_pair', 'tube', 'tool', None, 1e-5, line_stiffness=1e5, endpoint_stiffness=1e3),))
    assert validate_general_contact(model, definition, settings).valid
    manager = MultipleBeamS3CapManager(model, definition, settings)
    child = next(m for _, _, k, m in manager.pairs if k == 'rigid')
    assert type(child) is (HemispherePlaneManager if kind == 'plane' else HemisphereSphereManager)
    u = local_pose(model)
    for node in (1, 2):
        u[model.mesh.dof_manager.get_node_dofs(node)[0]] = -.0002
    G, H, result = manager.evaluate(u)
    fields = result.to_dict()
    assert fields['candidate_endpoints'] == 6 and len(fields['rigid_bodies']) == 1
    rig = next(p['contact'] for p in fields['pairs'] if p['interaction'] == 'tool_pair')
    assert rig['active_endpoints'] == 1
    np.testing.assert_allclose(sum(np.asarray(f['force']) for f in fields['beam_features']+fields['shell_features'])+
                              fields['rigid_bodies'][0]['rigid_force'], 0., atol=1e-10)
    np.testing.assert_allclose(sum(np.asarray(f['moment']) for f in fields['beam_features']+fields['shell_features'])+
                              fields['rigid_bodies'][0]['rigid_moment'], 0., atol=1e-10)
    direction = np.random.default_rng(18).normal(size=len(u))
    for _, _, k, m in manager.pairs:
        if k == 'shell': direction[m.shell_dofs[:, 3:]] = 0.
    direction /= np.linalg.norm(direction)
    for h in (1e-7, 3e-8):
        fd = (manager.evaluate(u+h*direction, tangent=False)[0]-manager.evaluate(u-h*direction, tangent=False)[0])/(2*h)
        assert np.linalg.norm(fd-H@direction)/np.linalg.norm(H@direction) < 1e-5
    contract = manager.restart_contract()
    effective = next(p['effective_settings'] for p in contract['pairs'] if p['kind'] == 'rigid')
    assert effective['beam_shell_geometry'] == 'translated_planar'
    assert contract['settings']['beam_shell_geometry'] == 'jet2_deformable_s3_dev'
    json.dumps(contract, allow_nan=False)
    fraction = manager.admissible_step_fraction(np.zeros_like(u), u)
    assert 0 < fraction < 1 and manager.swept_increment_admissible(np.zeros_like(u), fraction*u, 0., fraction)
    # Exercise the real heterogeneous driver summary and checkpoint, not just
    # the standalone result codec. This root tool remains open under transverse press.
    model, _, _, load = fixture()
    result = solve_static_nonlinear(model, load, contact=definition, contact_settings=settings,
        num_steps=4, max_iterations=25, tolerance=1e-10, emit_restart_checkpoint=True)
    assert result.status == 'completed', result.failure_reason
    assert len(result.info['general_contact']['pair_enforcement']) == 3
    json.dumps(result.to_restart_checkpoint(), allow_nan=False)


def test_two_beams_share_one_skin_and_true_endpoint_midnode_junction():
    model, original, settings, _ = fixture()
    for node, p in ((9, (0., .3, .02005)), (10, (.5, .3, .02005))):
        model.add_node(node, *p)
    model.add_element(4, BeamElement(4, (9, 10), 'beam',
        {'area': .001, 'Iy': 1e-4, 'Iz': 1e-4, 'J': 1e-4}))
    other = ContactSurface('other', (4,), beam_radius=.01, beam_end_caps='hemisphere')
    definition = replace(original, surfaces=(original.surfaces[0], original.surfaces[1], other),
        interactions=(original.interactions[0], replace(original.interactions[0], name='other_pair', surface='other')))
    def pose():
        u = np.zeros(model.mesh.dof_manager.total_dofs)
        for node in (1, 2, 9, 10):
            u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.0002
        return u
    manager = MultipleBeamS3CapManager(model, definition, settings)
    fields = manager.evaluate(pose())[2].to_dict()
    assert len(fields['beam_features']) == 2 and len(fields['shell_features']) == 1
    assert fields['candidate_endpoints'] == 4
    np.testing.assert_allclose(sum(np.asarray(f['force']) for f in fields['beam_features']+fields['shell_features']),
                              0., atol=1e-10)
    # A separate material member uses the first tube's endpoint as its midnode.
    # That node is a true junction even though the member is not selected.
    for node, p in ((11, (-.25, 0., .02005)), (12, (.25, 0., .02005))): model.add_node(node, *p)
    model.add_element(5, QuadraticBeamElement(5, (11, 1, 12), 'beam',
        {'area': .001, 'Iy': 1e-4, 'Iz': 1e-4, 'J': 1e-4}))
    revised = MultipleBeamS3CapManager(model, definition, settings)
    source = next(m for _, p, _, m in revised.pairs if p.surfaces[0].name == 'tube')
    second = next(m for _, p, _, m in revised.pairs if p.surfaces[0].name == 'other')
    assert set(source.cap_endpoints) == {2} and set(second.cap_endpoints) == {9, 10}
    assert revised.evaluate(pose())[2].to_dict()['candidate_endpoints'] == 3


@pytest.mark.parametrize('cancel', (False, True))
def test_actual_second_deforming_child_exception_or_cancel_is_replay_safe(cancel, monkeypatch):
    model, definition, settings, load = fixture()
    common = dict(contact=definition, contact_settings=settings, num_steps=4,
        max_iterations=25, tolerance=1e-10, emit_restart_checkpoint=True)
    armed = [False]; seen = []
    original = DeformableBeamS3CapManager.evaluate
    def evaluate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if armed[0] and self.shell_surface.name == 'z_skin':
            seen.append(self.shell_surface.name); armed[0] = False
            raise RuntimeError('second skin candidate failure')
        return result
    monkeypatch.setattr(DeformableBeamS3CapManager, 'evaluate', evaluate)
    token = CancellationToken()
    def callback(progress):
        if progress.event_type == 'nonlinear_static_step' and progress.metadata['load_factor'] >= .75:
            if cancel: token.cancel('two skin interruption')
            else: armed[0] = True
    with pytest.raises(SolveCancelled if cancel else RuntimeError):
        solve_static_nonlinear(model, load, **common, progress_callback=callback, cancellation_token=token)
    if not cancel: assert seen == ['z_skin']
    replay = solve_static_nonlinear(model, load, **common)
    fresh, _, _, _ = fixture()
    reference = solve_static_nonlinear(fresh, load, **common)
    assert replay.status == reference.status == 'completed'
    np.testing.assert_array_equal(replay.displacements, reference.displacements)
    assert replay.to_restart_checkpoint() == reference.to_restart_checkpoint()


def test_duplicate_physical_alias_resource_and_geometry_fail_closed():
    model, definition, settings, _ = fixture()
    alias = ContactSurface('skin_alias', (2,), side='top')
    dup = replace(definition, surfaces=definition.surfaces+(alias,), interactions=definition.interactions+
        (replace(definition.interactions[0], name='alias_pair', target=alias.name),))
    assert 'duplicate' in str(validate_general_contact(model, dup, settings).issues)
    assert not validate_general_contact(model, definition, replace(settings, maximum_candidate_pairs=1)).valid
    assert not validate_general_contact(model, definition, settings, analysis_control='arc_length').valid
    conflict = replace(definition, surfaces=definition.surfaces+(ContactSurface('other_tube', (1,),
        beam_radius=.02, beam_end_caps='hemisphere'),), interactions=definition.interactions+
        (replace(definition.interactions[0], name='different_radius', surface='other_tube'),))
    assert not validate_general_contact(model, conflict, settings).valid
    manager = MultipleBeamS3CapManager(model, definition, settings)
    bad = np.zeros(model.mesh.dof_manager.total_dofs)
    bad[model.mesh.dof_manager.get_node_dofs(8)[3]] = 1e-8
    with pytest.raises(ValueError, match='rotational'): manager.evaluate(bad)
    # Admission does not turn an interior-only kernel into a face-edge contact.
    bad[:] = 0.; bad[model.mesh.dof_manager.get_node_dofs(2)[0]] = 2.
    assert not manager.swept_increment_admissible(np.zeros_like(bad), bad, 0., 1.)
    assert not report_general_contact_capability(model, definition, settings).admitted_for_application


def test_public_two_deforming_skin_example():
    result = run()
    assert result.status == 'completed' and result.info['general_contact']['profile'] == PROFILE
