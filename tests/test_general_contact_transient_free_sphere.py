"""Independent two-body mechanics and complete coupled free-sphere work."""
from dataclasses import replace
import math

import numpy as np
import pytest

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, RigidContactBody, LoadCase)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_radial import _BeamRadialManager
from anyfesolver.general_contact.core import active_contact_manager
from anyfesolver.general_contact.free_rigid import ContactFreeRigidBody, _FreeSphereManager
from anyfesolver.general_contact.transient import ContactNodalLoadHistory, ContactTransientSolverControls, solve_transient_contact


CONTROLS = ContactTransientSolverControls(allow_classical_beam_dev=True,
    allow_free_sphere_dev=True, maximum_newton_iterations=20)
BODY = ContactFreeRigidBody('sphere', .002, 8e-8, (0., 0., .2), (2., -3., 4.))


def model_data(quadratic=False, elastic_modulus=1e3):
    model = FEModel('coupled_free_sphere')
    model.add_material('elastic', elastic_modulus, .3, density=1.)
    for node, x in enumerate((-.01, 0., .01) if quadratic else (-.01, .01), 1):
        model.add_node(node, x, 0., .025)
    cls = QuadraticBeamElement if quadratic else BeamElement
    model.add_element(1, cls(1, list(model.mesh.nodes), 'elastic',
        dict(area=.05, Iy=1e-4, Iz=1e-4, J=1e-4)))
    definition = ContactDefinition(
        surfaces=(ContactSurface('beam', (1,), beam_radius=.01, beam_end_caps='none'),),
        rigid_bodies=(RigidContactBody('sphere', 'sphere', (0., 0., 0.), radius=.01),),
        interactions=(ContactInteraction('pair', 'beam', 'sphere', None, .002, line_stiffness=1e4),))
    settings = ContactSolverSettings(pair_motion='general', line_integration_relative_tolerance=1e-8)
    return model, definition, settings


def impact(*, quadratic=False, end=.025, checkpoint=None, emit=False, body=BODY, dt=.00025, token=None, history=None):
    model, definition, settings = model_data(quadratic, elastic_modulus=10. if quadratic else 1e3)
    model.add_boundary_condition(BoundaryCondition('guide', list(model.mesh.nodes),
        {name: 0. for name in ('ux', 'uy', 'rx', 'ry', 'rz')}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes:
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.2
    load = None
    if history is not None:
        load = LoadCase('normal_history')
        weights = (1/6, 2/3, 1/6) if quadratic else (.5, .5)
        for node, weight in zip(model.mesh.nodes, weights):
            load.add_nodal_load(node, forces=(0., 0., -1e-4*weight))
    result = solve_transient_contact(model, definition, TransientConfig(dt, end, initial_velocity=initial),
        contact_settings=settings, solver_controls=CONTROLS, free_rigid_bodies=(body,),
        restart_checkpoint=checkpoint, emit_restart_checkpoint=emit, cancellation_token=token,
        base_load_case=load, nodal_load_history=history)
    return model, result


def force_and_derivative(distance):
    """Closed integral over a straight horizontal line, independent of kernels."""
    radius, half_length, stiffness = .02, .01, 1e4
    if distance >= radius:
        return 0., 0.
    assert distance > 0.
    a = min(half_length, math.sqrt(radius*radius-distance*distance))
    root = math.sqrt(a*a+distance*distance)
    return (2*stiffness*distance*(radius*math.asinh(a/distance)-a),
            2*stiffness*(radius*math.asinh(a/distance)-a-radius*a/root))


def independent_two_body(times, history=None):
    mass = np.array([.001, BODY.mass_kg])
    q, v, a = np.zeros(2), np.array([-.2, .2]), np.zeros(2)
    positions, velocities = [q.copy()], [v.copy()]
    for index, h in enumerate(np.diff(times), 1):
        predictor = q+h*v+.25*h*h*a
        vp = v+.5*h*a
        acceleration = np.zeros(2)
        for _ in range(20):
            candidate = predictor+.25*h*h*acceleration
            force, derivative = force_and_derivative(.025+candidate[0]-candidate[1])
            external = -1e-4*float(np.interp(times[index], [p[0] for p in history.points], [p[1] for p in history.points])) if history is not None else 0.
            residual = mass*acceleration+np.array([-force-external, force])
            if np.linalg.norm(residual) < 1e-12:
                break
            jacobian = np.diag(mass)-.25*h*h*derivative*np.array([[1., -1.], [-1., 1.]])
            acceleration += np.linalg.solve(jacobian, -residual)
        else:
            raise AssertionError('independent two-body Newton did not converge')
        q, v, a = candidate, vp+.5*h*acceleration, acceleration
        positions.append(q.copy()); velocities.append(v.copy())
    return np.array(positions), np.array(velocities)


def test_two_body_response_force_impulse_spin_and_opening():
    model, result = impact()
    position, velocity = independent_two_body(result.times)
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    body = [sample['sphere'] for sample in result.free_rigid_bodies]
    np.testing.assert_allclose(result.displacements[:, dof], position[:, 0], atol=1e-10, rtol=0.)
    np.testing.assert_allclose([state.centre_of_mass_m[2] for state in body], position[:, 1], atol=1e-10, rtol=0.)
    np.testing.assert_allclose(result.velocities[:, dof], velocity[:, 0], atol=1e-8, rtol=0.)
    np.testing.assert_allclose([state.velocity_m_s[2] for state in body], velocity[:, 1], atol=1e-8, rtol=0.)
    omega = np.asarray(BODY.initial_angular_velocity_rad_s)
    momentum = .001*(-.2)+BODY.mass_kg*.2
    for i, state in enumerate(body):
        assert .001*result.velocities[i, dof]+state.linear_momentum_kg_m_s[2] == pytest.approx(momentum, abs=1e-12)
        expected_force = force_and_derivative(.025+position[i, 0]-position[i, 1])[0]
        assert result.contact[i].beam_force[2] == pytest.approx(expected_force, abs=1e-7)
        np.testing.assert_allclose(state.contact_force_n, -np.asarray(result.contact[i].beam_force), atol=1e-12)
        np.testing.assert_allclose(state.angular_momentum_about_origin_kg_m2_s, BODY.isotropic_inertia_kg_m2*omega, atol=1e-12)
        R = np.asarray(state.rotation)
        axis = omega/np.linalg.norm(omega)
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        angle = np.linalg.norm(omega)*result.times[i]
        independent_rotation = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew @ skew)
        np.testing.assert_allclose(R, independent_rotation, atol=1e-14)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1., abs=1e-14)
        assert state.kinetic_energy_j-.5*BODY.mass_kg*np.dot(state.velocity_m_s, state.velocity_m_s) == pytest.approx(.5*BODY.isotropic_inertia_kg_m2*(omega @ omega), abs=1e-14)
        if i:
            dt = result.times[i]-result.times[i-1]
            impulse = .5*dt*(np.asarray(body[i-1].contact_force_n)+state.contact_force_n)
            np.testing.assert_allclose(BODY.mass_kg*(np.asarray(state.velocity_m_s)-body[i-1].velocity_m_s), impulse, atol=1e-11)
    assert any(contact.active_line_cells for contact in result.contact)
    assert result.contact[-1].active_line_cells == 0
    assert np.min(result.swept_minimum_gaps) >= -.002
    assert result.displacements.shape[1] == model.mesh.dof_manager.total_dofs


@pytest.mark.parametrize('quadratic', [False, True])
def test_complete_coupled_virtual_work_tangent_and_centre_wrench(quadratic):
    model, definition, settings = model_data(quadratic)
    owner = _FreeSphereManager(_BeamRadialManager(model, definition, settings), BODY)
    u = np.zeros(model.mesh.dof_manager.total_dofs+3)
    for node in model.mesh.nodes:
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.009
    u[-3:] = [.001, -.0002, .0003]
    force, tangent, contact = owner.evaluate(u)
    direction = np.random.default_rng(7).normal(size=len(u))*.001
    h = 1e-4
    plus = owner.evaluate(u+h*direction, tangent=False)
    minus = owner.evaluate(u-h*direction, tangent=False)
    assert (plus[2].line_contact_energy-minus[2].line_contact_energy)/(2*h) == pytest.approx(force @ direction, abs=1e-10)
    fd = (plus[0]-minus[0])/(2*h)
    assert np.linalg.norm(fd-tangent @ direction)/np.linalg.norm(fd) < 1e-5
    np.testing.assert_allclose(force[-3:], -np.asarray(owner.translation_map.T @ force[:-3]).reshape(-1), atol=1e-12)
    centre = np.asarray(definition.rigid_bodies[0].point)+u[-3:]
    np.testing.assert_allclose(contact.rigid_moment, np.cross(centre, contact.rigid_force), atol=1e-10)


def test_active_restart_binds_inertia_spin_and_named_state():
    import copy
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    _, direct = impact()
    _, prefix = impact(end=.014, emit=True)
    assert prefix.restart_checkpoint['version'] == 6
    assert prefix.contact[-1].active_line_cells
    _, resumed = impact(checkpoint=prefix.restart_checkpoint)
    for field in ('times', 'displacements', 'velocities', 'accelerations', 'swept_minimum_gaps'):
        np.testing.assert_allclose(np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])), getattr(direct, field), atol=1e-9, rtol=0., err_msg=field)
    for attr in ('centre_of_mass_m', 'velocity_m_s', 'rotation', 'kinetic_energy_j'):
        np.testing.assert_allclose([getattr(s['sphere'], attr) for s in prefix.free_rigid_bodies+resumed.free_rigid_bodies[1:]], [getattr(s['sphere'], attr) for s in direct.free_rigid_bodies], atol=1e-9, rtol=0.)
    for changed in (replace(BODY, mass_kg=.003), replace(BODY, isotropic_inertia_kg_m2=9e-8), replace(BODY, initial_angular_velocity_rad_s=(0., 0., 0.))):
        with pytest.raises(ValueError, match='analysis is incompatible'):
            impact(body=changed, checkpoint=prefix.restart_checkpoint)
    malformed = copy.deepcopy(prefix.restart_checkpoint)
    malformed['state']['free_rigid_body_names'] = ['other']
    malformed.pop('checkpoint_sha256')
    malformed['checkpoint_sha256'] = analysis_configuration_fingerprint(malformed)
    with pytest.raises(ValueError, match='free body identity'):
        impact(checkpoint=malformed)


def test_deforming_b3_free_sphere_preserves_normal_momentum():
    model, result = impact(quadratic=True)
    dofs = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in model.mesh.nodes]
    weights = .001*np.array([1/6, 2/3, 1/6])
    initial = -.0002+BODY.mass_kg*.2
    for index, sample in enumerate(result.free_rigid_bodies):
        body = sample['sphere']
        assert weights @ result.velocities[index, dofs]+body.linear_momentum_kg_m_s[2] == pytest.approx(initial, abs=1e-12)
        if index:
            dt = result.times[index]-result.times[index-1]
            beam_impulse = .5*dt*(result.contact[index-1].beam_force[2]+result.contact[index].beam_force[2])
            assert weights @ (result.velocities[index, dofs]-result.velocities[index-1, dofs]) == pytest.approx(beam_impulse, abs=1e-11)
    assert np.max(np.ptp(result.displacements[:, dofs], axis=1)) > 1e-5
    assert any(contact.active_line_cells for contact in result.contact)
    assert np.min(result.swept_minimum_gaps) >= -.002


def test_two_body_elastic_energy_refinement_against_closed_line_potential():
    defects = []
    initial_energy = .5*.001*.2**2+.5*BODY.mass_kg*.2**2
    spin = .5*BODY.isotropic_inertia_kg_m2*np.dot(BODY.initial_angular_velocity_rad_s, BODY.initial_angular_velocity_rad_s)
    for dt in (.00025, .000125):
        model, result = impact(dt=dt)
        dof = model.mesh.dof_manager.get_node_dofs(1)[2]
        balances = []
        for index, sample in enumerate(result.free_rigid_bodies):
            body = sample['sphere']
            distance = .025+result.displacements[index, dof]-body.centre_of_mass_m[2]
            potential = 0.
            if distance < .02:
                a = min(.01, math.sqrt(.02**2-distance**2))
                potential = 1e4*(.02**2*a-.02*(a*math.sqrt(a*a+distance*distance)
                    +distance*distance*math.asinh(a/distance))+a**3/3+distance*distance*a)
            assert result.contact[index].line_contact_energy == pytest.approx(potential, abs=1e-10)
            balances.append(.5*.001*result.velocities[index, dof]**2+body.kinetic_energy_j-spin+potential)
        defects.append(float(np.max(np.abs(np.asarray(balances)-initial_energy))/initial_energy))
    assert defects[1] < defects[0]
    assert defects[1] <= .01


def test_free_sphere_nodal_history_and_work_survive_active_restart():
    history = ContactNodalLoadHistory(((0., 0.), (.004, 1.), (.025, .5)))
    model, direct = impact(history=history)
    position, velocity = independent_two_body(direct.times, history)
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    np.testing.assert_allclose(direct.displacements[:, dof], position[:, 0], atol=1e-10, rtol=0.)
    np.testing.assert_allclose(direct.velocities[:, dof], velocity[:, 0], atol=1e-8, rtol=0.)
    assert .004 in direct.times
    _, prefix = impact(end=.014, emit=True, history=history)
    _, resumed = impact(checkpoint=prefix.restart_checkpoint, history=history)
    for field in ('times', 'displacements', 'velocities', 'external_nodal_work_j'):
        np.testing.assert_allclose(np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])), getattr(direct, field), atol=1e-9, rtol=0.)
    work = 0.
    for index, h in enumerate(np.diff(direct.times), 1):
        q0, q1 = direct.displacements[index-1, dof], direct.displacements[index, dof]
        v0, v1 = direct.velocities[index-1, dof], direct.velocities[index, dof]
        derivative = np.array([h*v0, 2*(-3*q0-2*h*v0+3*q1-h*v1), 3*(2*q0+h*v0-2*q1+h*v1)])
        f0 = -1e-4*np.interp(direct.times[index-1], [p[0] for p in history.points], [p[1] for p in history.points])
        f1 = -1e-4*np.interp(direct.times[index], [p[0] for p in history.points], [p[1] for p in history.points])
        coefficients = np.polynomial.polynomial.polymul(derivative, [f0, f1-f0])
        work += sum(value/(power+1) for power, value in enumerate(coefficients))
    assert direct.external_nodal_work_j[-1] == pytest.approx(work, abs=1e-12)


@pytest.mark.parametrize('cancel', [False, True])
def test_free_body_candidate_failure_does_not_commit(monkeypatch, cancel):
    import anyfesolver.general_contact.transient as driver
    from anyfesolver.control import CancellationToken, SolveCancelled
    token = CancellationToken()
    commits, calls = [], []
    commit, evaluate = driver.commit_state_candidate, _FreeSphereManager.evaluate
    def recorded(*args, **kwargs):
        commits.append(1)
        return commit(*args, **kwargs)
    def injected(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            if cancel:
                token.cancel()
            else:
                raise RuntimeError('free body candidate failed')
        return evaluate(self, *args, **kwargs)
    monkeypatch.setattr(driver, 'commit_state_candidate', recorded)
    monkeypatch.setattr(_FreeSphereManager, 'evaluate', injected)
    with pytest.raises(SolveCancelled if cancel else RuntimeError):
        impact(token=token)
    assert len(commits) == 1
    assert active_contact_manager() is None


def test_explicit_inertia_ownership_and_capability_rejection():
    with pytest.raises(ValueError, match='positive'):
        replace(BODY, isotropic_inertia_kg_m2=0.)
    with pytest.raises(ValueError, match='positive'):
        replace(BODY, mass_kg=True)
    with pytest.raises(ValueError, match='three finite'):
        replace(BODY, initial_velocity_m_s=(0., float('nan'), 0.))
    model, definition, settings = model_data()
    config = TransientConfig(.001, .01, initial_velocity=np.zeros(model.mesh.dof_manager.total_dofs))
    with pytest.raises(ValueError, match='opt-in'):
        solve_transient_contact(model, definition, config, contact_settings=settings, solver_controls=CONTROLS)
    with pytest.raises(ValueError, match='matching body'):
        solve_transient_contact(model, definition, config, contact_settings=settings, solver_controls=CONTROLS,
            free_rigid_bodies=(replace(BODY, body_name='wrong'),))
