"""Independent unilateral-manifold oracles; no production rate oracle calls."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar, root

from anyfesolver import ContactNodalLoadHistory, ContactTransientSolverControls, TransientConfig, solve_transient_contact
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
import anyfesolver.general_contact.transient as driver
from test_general_contact_transient_plane import _guided_s3_descriptor_impact, _uniform_s3_normal_load
from s3_vk_element_oracle import S3VonKarmanExcess

from anyfesolver.general_contact.algebraic_path import affine_descriptor_path
import anyfesolver.general_contact.algebraic_path as path_kernel


@pytest.mark.parametrize("physical", [[.1, -.2, 0.], [.1, -.8, .8]])
def test_scalar_crossing_and_reversal_match_exact_manifold_and_corner_minimum(physical):
    k, c = 2., 3.
    path = affine_descriptor_path(
        np.diag([0., k]), np.array([[0.], [1.]]), np.c_[physical, np.zeros(3)],
        [[1., 1.]], [0.], [c],
    )
    roots = (.5,) if physical[2] == 0 else ((1-1/np.sqrt(2))/2, (1+1/np.sqrt(2))/2)
    np.testing.assert_allclose(path.transition_fractions, roots, atol=2e-15)
    assert path.segments[0].start == 0. and path.segments[-1].end == 1.
    for segment in path.segments:
        for s in (segment.start, (segment.start+segment.end)/2, segment.end):
            x = physical[0]+s*(physical[1]+s*physical[2])
            expected = [x, -c*min(x, 0.)/(k+c)]
            actual = segment.position(s)
            np.testing.assert_allclose(actual, expected, atol=2e-16)
            assert abs(k*actual[1]+c*min(actual.sum(), 0.)) < 3e-16
    # A corner need not use the pressure weighted-gap map.
    assert path.minimum_gap([0.], [[1., .4]]) == pytest.approx(-.076, abs=2e-16)
    assert path.minimum_gap([0.], [[1., 1.]]) == pytest.approx(-.04, abs=2e-16)
    assert path.maximum_branch_gap_error < 1e-15


def test_two_separable_constraints_reference_shift_and_scaled_null_basis():
    K = np.diag([0., 0., 2., 5.])
    B = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    physical = np.array([[.1, -.1, 11., -7.], [-.2, .4, -3., 2.], [0., 0., 4., 5.]])
    original = physical.copy()
    path = affine_descriptor_path(K, np.eye(4)[:, 2:] @ np.diag([2., 3.]), physical,
                                 B, [0., .025], [3., 7.])
    np.testing.assert_allclose(path.transition_fractions, [.1875, .5], atol=2e-15)
    for segment in path.segments:
        s = (segment.start+segment.end)/2
        x, y = .1-.2*s, -.1+.4*s
        expected = [x, y, -3*min(x, 0.)/5, -7*min(y+.025, 0.)/12]
        np.testing.assert_allclose(segment.position(s), expected, atol=3e-16)
    np.testing.assert_array_equal(physical, original)
    with pytest.raises(ValueError):
        path.segments[0].coefficients[0, 0] = 3.


@pytest.mark.parametrize("force", [-1.3, 1.3])
def test_constant_algebraic_force_is_included(force):
    path = affine_descriptor_path(np.diag([0., 2.]), [[0.], [1.]],
                                 [[-.7, 0.], [1.4, 0.], [0., 0.]],
                                 [[1., 1.]], [0.], [3.], force=[0., force])
    assert path.transition_fractions[0] == pytest.approx((.7-force/2)/1.4)
    for segment in path.segments:
        s = (segment.start+segment.end)/2
        x = -.7+1.4*s
        expected_a = (force-3*x)/5 if 2*x+force < 0 else force/2
        np.testing.assert_allclose(segment.position(s), [x, expected_a], atol=2e-16)


def test_touch_has_continuous_position_and_distinct_incoming_outgoing_rates():
    path = affine_descriptor_path(np.diag([0., 2.]), [[0.], [1.]],
                                 [[.1, 0.], [-.2, 0.], [0., 0.]],
                                 [[1., 1.]], [0.], [3.])
    left, right = path.segments
    np.testing.assert_allclose(left.position(.5), right.position(.5), atol=1e-16)
    np.testing.assert_allclose(left.coefficients[1], [-.2, 0.])
    np.testing.assert_allclose(right.coefficients[1], [-.2, .12])
    # Smoothing these traces into one all-coordinate Hermite curve changes
    # the algebraic constraint inside the interval.
    midpoint_hermite = -.015
    assert abs(2*midpoint_hermite+3*min(midpoint_hermite, 0.)) > .01


def test_stationary_quadratic_touch_does_not_activate_or_discard_an_interval():
    path = affine_descriptor_path(np.diag([0., 2.]), [[0.], [1.]],
                                 [[.25, 0.], [-1., 0.], [1., 0.]],
                                 [[1., 1.]], [0.], [3.])
    assert path.transition_fractions == ()
    assert path.minimum_gap([0.], [[1., 1.]]) == pytest.approx(0., abs=2e-16)


def test_nonfinite_branch_solution_cannot_be_accepted():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="nonfinite.*branch solution"):
            affine_descriptor_path(np.diag([1e-308, 1e-308]), [[1.], [1.]],
                                   np.zeros((3, 2)), [[1., 1.]], [0.], [3.],
                                   force=[5e307, 5e307])


def test_large_finite_quadratic_extremum_does_not_overflow_vertex_denominator():
    path = affine_descriptor_path(np.diag([0., 2.]), [[0.], [1.]],
                                 [[0., 0.], [-5e307, 0.], [1e308, 0.]],
                                 [[0., 1.]], [1.], [3.])
    assert path.minimum_gap([0.], [[1., 0.]]) == pytest.approx(-6.25e306)


@pytest.mark.parametrize("K", [np.diag([0., -2.]), np.zeros((2, 2))])
def test_nonpositive_algebraic_work_fails_explicitly(K):
    with pytest.raises(ValueError, match="positive definite"):
        affine_descriptor_path(K, [[0.], [1.]], [[.1, 0.], [-.2, 0.], [0., 0.]],
                               [[1., 1.]], [0.], [3.])


def test_pressure_cap_does_not_drop_constraints():
    with pytest.raises(ValueError, match="over-cap"):
        affine_descriptor_path(np.diag([0., 2.]), [[0.], [1.]],
                               [[.1, 0.], [-.2, 0.], [0., 0.]],
                               [[1., 1.]]*4, [0.]*4, [3.]*4)


def test_missing_crossing_roots_fail_interval_coverage(monkeypatch):
    monkeypatch.setattr(path_kernel, "_roots", lambda polynomial: ())
    with pytest.raises(ValueError, match="unresolved.*interval"):
        affine_descriptor_path(np.diag([0., 2.]), [[0.], [1.]],
                               [[.1, 0.], [-.2, 0.], [0., 0.]],
                               [[1., 1.]], [0.], [3.])


def test_piecewise_work_matches_independent_scalar_integral():
    path = affine_descriptor_path(np.diag([0., 2.]), [[0.], [1.]],
                                 [[.1, 0.], [-.2, 0.], [0., 0.]],
                                 [[1., 1.]], [0.], [3.])
    # a'=.12 only on [.5,1]; integral .12*(2+3s) ds = .255.
    assert path.linear_load_work([0., 2.], [0., 5.]) == pytest.approx(.255, abs=1e-15)


def test_s3_algebraic_path_matches_independent_force_roots_and_physical_skin_minima(monkeypatch):
    model, definition, velocity = _guided_s3_descriptor_impact(tilt=.002, span=.1)
    controls = ContactTransientSolverControls(
        allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=True, s3_algebraic_path_dev=True,
    )
    observations = []
    original = driver._S3AffinePathAdapter.evaluate

    def observe(adapter, start, end, speed, dt, load):
        path, minimum = original(adapter, start, end, speed, dt, load)
        observations.append((start.copy(), end.copy(), speed.copy(), dt, path, minimum))
        return path, minimum

    monkeypatch.setattr(driver._S3AffinePathAdapter, "evaluate", observe)
    result = solve_transient_contact(
        model, definition, TransientConfig(.0002, .012, initial_velocity=velocity),
        solver_controls=controls,
    )
    assert result.profile.endswith("NEWMARK_ALGEBRAIC_PATH_DEV_V1")
    assert len(result.represented_path_transition_times) >= 1
    assert all(0. < t < .012 for t in result.represented_path_transition_times)
    assert all(g >= -.001 for g in result.swept_minimum_gaps)

    # Independent P1 work/skin maps, public structural stiffness, and nonlinear
    # force roots. No production path, pressure-manager, or rate oracle calls.
    K, _ = assemble_stiffness_matrix(model)
    K = K.toarray()
    element = model.mesh.elements[1]
    node_dofs = np.array([model.mesh.dof_manager.get_node_dofs(i) for i in (1, 2, 3)])
    physical = node_dofs[:, 2]
    algebraic = node_dofs[:, 3:].reshape(-1)
    free = node_dofs[:, 2:].reshape(-1)
    normal = np.array(definition.rigid_bodies[0].normal)
    director = np.array(element.reference_normal)
    offset = -.5*element.thickness-element.reference_surface_offset
    points = np.array([[model.mesh.nodes[i].x, model.mesh.nodes[i].y, model.mesh.nodes[i].z]
                       for i in (1, 2, 3)])
    corner_base = (points+offset*director-np.array(definition.rigid_bodies[0].point)) @ normal
    H = np.zeros((3, K.shape[0]))
    for i, dofs in enumerate(node_dofs):
        H[i, dofs[:3]] = normal
        H[i, dofs[3:]] = offset*np.cross(director, normal)
    pressure_average = (np.eye(3)+np.ones((3, 3)))/4
    B, g = pressure_average @ H, pressure_average @ corner_base
    area = .5*np.linalg.norm(np.cross(points[1]-points[0], points[2]-points[0]))
    w = definition.interactions[0].normal_stiffness*area/3
    Kaa, Kap, Ba = K[np.ix_(algebraic, algebraic)], K[np.ix_(algebraic, physical)], B[:, algebraic]
    # Independent first-principles von Karman excess of the S3 element (all 18 dofs),
    # validated against production in test_s3_vk_element_oracle.py.
    vk = S3VonKarmanExcess(points, director, 1.0e5, .3, element.thickness,
                           element.reference_surface_offset, element.director_polarity)

    for start, end, speed, dt, path, actual_minimum in observations:
        q0 = np.zeros(K.shape[0]); q0[free] = start
        q1 = np.zeros(K.shape[0]); q1[free] = end
        v0 = np.zeros(K.shape[0]); v0[free] = speed

        def independent_pose(s):
            q = np.zeros(K.shape[0])
            q[physical] = (q0[physical]+s*dt*v0[physical]
                           +s*s*(q1[physical]-q0[physical]-dt*v0[physical]))
            rigid_gap = g+B[:, physical] @ q[physical]
            right = Kap @ q[physical]
            def pose(a):
                trial = q.copy()
                trial[algebraic] = a
                return trial
            def equations(a):
                return (Kaa @ a+right+vk.force(pose(a))[algebraic]
                        +Ba.T @ (w*np.minimum(rigid_gap+Ba @ a, 0.)))
            def jacobian(a):
                active = rigid_gap+Ba @ a < 0.
                return (Kaa+vk.tangent(pose(a))[np.ix_(algebraic, algebraic)]
                        +w*Ba[active].T @ Ba[active])
            solve = root(equations, (1-s)*q0[algebraic]+s*q1[algebraic],
                         jac=jacobian, options={"xtol": 1e-10})
            assert np.linalg.norm(equations(solve.x)) < 1e-10
            q[algebraic] = solve.x
            return q

        reference_minimum = float("inf")
        for segment in path.segments:
            middle = (segment.start+segment.end)/2
            np.testing.assert_allclose(segment.position(middle), independent_pose(middle)[free],
                                       atol=1e-10, rtol=0.)
            # Each reported segment is quadratic; the separate force-root
            # reference supplies actual poses to a scalar optimizer and endpoints.
            for corner in range(3):
                def gap(s):
                    return corner_base[corner]+H[corner] @ independent_pose(s)
                optimum = minimize_scalar(gap, bounds=(segment.start, segment.end), method="bounded",
                                          options={"xatol": 1e-12})
                reference_minimum = min(reference_minimum, gap(segment.start), gap(segment.end), optimum.fun)
        assert actual_minimum == pytest.approx(reference_minimum, abs=1e-10)


def test_s3_algebraic_history_work_and_same_partition_restart():
    controls = ContactTransientSolverControls(
        allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=True, s3_algebraic_path_dev=True,
    )
    history = ContactNodalLoadHistory(((0., 0.), (.002, 1.)))
    def solve(end, checkpoint=None, emit=False, supplied_controls=controls):
        model, definition, velocity = _guided_s3_descriptor_impact()
        velocity[:] = 0.
        return solve_transient_contact(
            model, definition, TransientConfig(.001, end, initial_velocity=velocity),
            base_load_case=_uniform_s3_normal_load(), nodal_load_history=history,
            solver_controls=supplied_controls, emit_restart_checkpoint=emit, restart_checkpoint=checkpoint,
        )
    direct = solve(.002)
    assert direct.profile.endswith("NEWMARK_ALGEBRAIC_PATH_HISTORY_DEV_V1")
    # Independent scalar Newmark path: work=2e-9/3+8e-9 J.
    assert direct.external_nodal_work_j[-1] == pytest.approx(26e-9/3, abs=2e-12)
    prefix = solve(float(direct.times[1]), emit=True)
    resumed = solve(.002, prefix.restart_checkpoint)
    for field in ("displacements", "velocities", "accelerations", "external_nodal_work_j", "swept_minimum_gaps"):
        np.testing.assert_allclose(np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])),
                                   getattr(direct, field), atol=1e-11, rtol=0.)
    assert prefix.represented_path_transition_times+resumed.represented_path_transition_times == direct.represented_path_transition_times
    from dataclasses import replace
    with pytest.raises(ValueError, match="checkpoint .* is incompatible"):
        solve(.002, prefix.restart_checkpoint, supplied_controls=replace(controls, s3_algebraic_path_dev=False))


def test_grid_aligned_scalar_transition_survives_path_checkpoint():
    arguments = (np.diag([0., 2.]), [[0.], [1.]])
    open_path = affine_descriptor_path(*arguments, [[.1, 0.], [-.1, 0.], [0., 0.]],
                                       [[1., 1.]], [0.], [3.])
    closing_path = affine_descriptor_path(*arguments, [[0., 0.], [-.1, 0.], [0., 0.]],
                                          [[1., 1.]], [0.], [3.])
    assert open_path.transition_fractions == closing_path.transition_fractions == ()
    expected = driver._accepted_path_transition_times(open_path.segments[-1].active, closing_path, .5, .5)
    assert expected == (.5,)
    checkpoint = driver._new_restart_checkpoint("fixture", "fixture", .5, [0., 0.], [-.2, 0.],
                                                [0., 0.], 1, path_terminal_active=open_path.segments[-1].active)
    assert checkpoint["version"] == 3
    read = driver._read_restart_checkpoint(checkpoint, "fixture", "fixture", 2, 1.,
                                           load_history=None, path_pressure_dofs=1)
    assert driver._accepted_path_transition_times(read[-1], closing_path, read[0], .5) == expected


def test_public_s3_active_path_restart_and_failure_cleanup(monkeypatch):
    controls = ContactTransientSolverControls(
        allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=True, s3_algebraic_path_dev=True,
    )
    def solve(end, checkpoint=None, emit=False):
        model, definition, velocity = _guided_s3_descriptor_impact()
        return solve_transient_contact(
            model, definition, TransientConfig(.002, end, initial_velocity=velocity),
            solver_controls=controls, emit_restart_checkpoint=emit, restart_checkpoint=checkpoint,
        )
    direct = solve(.012)
    assert any(abs(t-.01) < 1e-12 for t in direct.represented_path_transition_times)
    prefix = solve(float(direct.times[5]), emit=True)
    resumed = solve(.012, prefix.restart_checkpoint)
    assert prefix.restart_checkpoint["version"] == 3
    for field in ("displacements", "velocities", "accelerations", "swept_minimum_gaps"):
        np.testing.assert_allclose(np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])),
                                   getattr(direct, field), rtol=0., atol=1e-11)
    np.testing.assert_allclose(prefix.represented_path_transition_times+resumed.represented_path_transition_times,
                               direct.represented_path_transition_times, rtol=0., atol=1e-12)
    from anyfesolver.general_contact.core import active_contact_manager
    commits = []
    original_commit = driver.commit_state_candidate
    def commit(*args, **kwargs):
        commits.append(True)
        return original_commit(*args, **kwargs)
    def fail(*args, **kwargs):
        raise ValueError("injected unresolved affine interval")
    with monkeypatch.context() as patch:
        patch.setattr(driver, "affine_descriptor_path", fail)
        patch.setattr(driver, "commit_state_candidate", commit)
        with pytest.raises(ValueError, match="injected unresolved affine interval"):
            solve(.004, emit=True)
    assert len(commits) == 1
    assert active_contact_manager() is None


def test_explicit_lifted_endpoint_budget_rejects_an_ill_resolved_trace():
    from scipy import sparse
    adapter = driver._S3AffinePathAdapter(
        np.zeros((6, 6)), np.eye(6)[:, 3:], np.zeros((1, 6)), np.ones(1), np.ones(1),
        np.zeros(6), np.ones(1), np.zeros((1, 6)),
    )
    path = affine_descriptor_path(np.diag([0., 0., 0., 2., 2., 2.]), np.eye(6)[:, 3:],
                                 np.zeros((3, 6)), np.zeros((1, 6)), [1.], [3.])
    wrong_endpoint = np.array([0., 0., 0., .001, 0., 0.])
    with pytest.raises(ValueError, match="endpoint mismatch"):
        adapter.check_endpoints(path, np.zeros(6), wrong_endpoint, sparse.eye(6), ContactTransientSolverControls())


def test_s3_algebraic_path_requires_explicit_directional_rate_profile():
    with pytest.raises(ValueError, match="one-sided-rate opt-in"):
        ContactTransientSolverControls(allow_s3_descriptor_dev=True, s3_algebraic_path_dev=True)
