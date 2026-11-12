"""Analytical and transactional checks for guided S3 dead-load history."""

import math

import numpy as np
import pytest

from anyfesolver import (
    ContactNodalLoadHistory, ContactTransientSolverControls, LoadCase,
    TransientConfig, solve_transient_contact,
)
from anyfesolver.general_contact.core import active_contact_manager
from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
from anyfesolver.assembly import assemble_mass_matrix, build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from test_general_contact_transient_plane import (
    _guided_s3_descriptor_impact, _uniform_s3_normal_load,
)


CONTROLS = ContactTransientSolverControls(allow_s3_descriptor_dev=True)


def test_s3_history_uniform_ramp_free_flight_and_discrete_path_work():
    history = ContactNodalLoadHistory(((0., 0.), (.004, 1.)))
    force_rate = .02/.004
    acceleration_rate = force_rate/.025  # rho*A*t = 1*.5*.05
    errors = []
    for dt in (.001, .0005):
        model, definition, velocity = _guided_s3_descriptor_impact()
        velocity[:] = 0.
        result = solve_transient_contact(
            model, definition, TransientConfig(dt, .004, initial_velocity=velocity),
            base_load_case=_uniform_s3_normal_load(), nodal_load_history=history,
            solver_controls=CONTROLS,
        )
        dof = model.mesh.dof_manager.get_node_dofs(1)[2]
        times = result.times
        # Closed-form free motion under a linear force. Average-acceleration
        # displacement has the known accumulated ramp error b*t*dt^2/12.
        exact = -acceleration_rate*times**3/6
        discrete = exact-acceleration_rate*times*dt**2/12
        np.testing.assert_allclose(result.displacements[:, dof], discrete,
                                   rtol=0., atol=2e-12)
        np.testing.assert_allclose(result.velocities[:, dof],
                                   -acceleration_rate*times**2/2, atol=2e-12)
        np.testing.assert_allclose(result.accelerations[:, dof],
                                   -acceleration_rate*times, atol=2e-10)
        # Integration by parts of linear force and cubic endpoint Hermite
        # displacement, independent of the production Gauss work integration.
        expected_work = 0.
        for n in range(1, len(times)):
            q0, q1 = discrete[n-1:n+1]
            v0, v1 = -acceleration_rate*times[n-1:n+1]**2/2
            integral_q = dt*((q0+q1)/2+dt*(v0-v1)/12)
            expected_work += (-force_rate*times[n]*q1
                              +force_rate*times[n-1]*q0+force_rate*integral_q)
        assert result.external_nodal_work_j[-1] == pytest.approx(expected_work, abs=2e-12)
        assert all(c.active_pressure_dofs == 0 for c in result.contact)
        assert result.profile.endswith("DESCRIPTOR_HISTORY_DEV_V1")
        errors.append(abs(result.displacements[-1, dof]-exact[-1]))
    assert errors[0]/errors[1] == pytest.approx(4., rel=1e-5)
    assert active_contact_manager() is None


def test_s3_history_ramped_impact_against_independent_forced_oscillator():
    model, definition, velocity = _guided_s3_descriptor_impact()
    result = solve_transient_contact(
        model, definition, TransientConfig(.0001, .011, initial_velocity=velocity),
        base_load_case=_uniform_s3_normal_load(),
        nodal_load_history=ContactNodalLoadHistory(((0., 0.), (.012, 1.))),
        solver_controls=CONTROLS,
    )
    mass, stiffness, rate = .025, 50000., .02/.012
    b = rate/mass
    hit = .01
    for _ in range(6):
        hit -= (hit+b*hit**3/6-.01)/(1+b*hit**2/2)
    speed = 1+b*hit**2/2
    tau = .011-hit
    omega = math.sqrt(stiffness/mass)
    penetration = ((rate*hit+rate*tau)/stiffness
                   -rate*hit/stiffness*math.cos(omega*tau)
                   +(speed-rate/stiffness)/omega*math.sin(omega*tau))
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    assert result.contact[-1].active_pressure_dofs > 0
    assert result.displacements[-1, dof] == pytest.approx(-.01-penetration, abs=2e-5)
    assert result.external_nodal_work_j[-1] > 0
    assert all(c.gap_tolerance_met for c in result.contact)


def test_s3_history_knot_split_restart_binds_history_and_profile():
    history = ContactNodalLoadHistory(((0., 0.), (.0015, 1.), (.004, .5)))
    def solve(end, checkpoint=None, emit=False, supplied_history=history):
        model, definition, velocity = _guided_s3_descriptor_impact()
        velocity[:] = 0.
        return solve_transient_contact(
            model, definition, TransientConfig(.001, end, initial_velocity=velocity),
            base_load_case=_uniform_s3_normal_load(),
            nodal_load_history=supplied_history, solver_controls=CONTROLS,
            restart_checkpoint=checkpoint, emit_restart_checkpoint=emit,
        )
    direct = solve(.004)
    prefix = solve(.0015, emit=True)
    assert prefix.restart_checkpoint["version"] == 2
    resumed = solve(.004, prefix.restart_checkpoint)
    assert np.any(np.isclose(direct.times, .0015, atol=1e-15))
    np.testing.assert_allclose(np.r_[prefix.times, resumed.times[1:]], direct.times,
                               atol=1e-15, rtol=0.)
    for field in ("displacements", "velocities", "accelerations", "external_nodal_work_j"):
        np.testing.assert_allclose(
            np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])),
            getattr(direct, field), rtol=0., atol=1e-11,
        )
    with pytest.raises(ValueError, match="checkpoint version is incompatible"):
        solve(.004, prefix.restart_checkpoint, supplied_history=None)
    changed = ContactNodalLoadHistory(((0., 0.), (.0015, 1.), (.004, .6)))
    with pytest.raises(ValueError, match="checkpoint analysis is incompatible"):
        solve(.004, prefix.restart_checkpoint, supplied_history=changed)


@pytest.mark.parametrize("moment", [1., 1e-12])
def test_s3_history_zero_initial_scale_cannot_hide_algebraic_base_load(moment):
    model, definition, velocity = _guided_s3_descriptor_impact()
    load = LoadCase("massless_rotation")
    load.add_nodal_load(1, moments=(0., moment, 0.))
    with pytest.raises(ValueError, match="base load has unresolved algebraic force"):
        solve_transient_contact(
            model, definition, TransientConfig(.001, .002, initial_velocity=velocity),
            base_load_case=load,
            nodal_load_history=ContactNodalLoadHistory(((0., 0.), (.002, 1e12))),
            solver_controls=CONTROLS,
        )
    assert active_contact_manager() is None


def test_s3_history_nonuniform_load_retains_later_algebraic_rates():
    model, definition, velocity = _guided_s3_descriptor_impact()
    velocity[:] = 0.
    load = LoadCase("nonuniform_physical_ramp")
    for node, force in zip((1, 2, 3), (-.010, -.007, -.003)):
        load.add_nodal_load(node, forces=(0., 0., force))
    result = solve_transient_contact(
        model, definition, TransientConfig(.0005, .001, initial_velocity=velocity),
        base_load_case=load,
        nodal_load_history=ContactNodalLoadHistory(((0., 0.), (.001, 1.))),
        solver_controls=CONTROLS,
    )
    stiffness, _ = assemble_stiffness_matrix(model)
    mass, _ = assemble_mass_matrix(model)
    _, _, T, _, free, _ = build_constraint_transformation(
        stiffness, np.zeros(mass.shape[0]), model,
    )
    Z = build_declared_algebraic_basis(
        model, mass, (T.T @ mass @ T).tocsr(), T, free,
        dense_size_limit=512,
    ).reduced_basis.toarray()
    for u, v, a in zip(result.displacements[1:], result.velocities[1:], result.accelerations[1:]):
        assert np.linalg.norm(Z.T @ v[free]) > 1e-8
        f, K, _ = _assemble_nonlinear_system(model, u, {}, 3, tangent=True, kinematics="von_karman")
        assert np.linalg.norm(Z.T @ (T.T @ (K @ v))) < 1e-8
        # Differentiate structural force directly, independently of the
        # production tangent-difference rate reconstruction. Contact is open.
        h = .01
        plus, _, _ = _assemble_nonlinear_system(model, u+h*v, {}, 3, tangent=False, kinematics="von_karman")
        minus, _, _ = _assemble_nonlinear_system(model, u-h*v, {}, 3, tangent=False, kinematics="von_karman")
        curvature = (plus-2*f+minus)/h**2
        assert np.linalg.norm(Z.T @ (T.T @ (K @ a+curvature))) < 1e-8
    assert all(c.active_pressure_dofs == 0 for c in result.contact)
