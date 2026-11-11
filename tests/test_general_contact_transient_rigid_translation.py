"""Prescribed translation: independent rigid-mode equations and polynomial work."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import BoundaryCondition, LoadCase
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.beam_plane import _BeamPlaneManager
from anyfesolver.general_contact.core import active_contact_manager
from anyfesolver.general_contact.rigid_translation import ContactRigidTranslationHistory, _PlaneTranslations
from anyfesolver.general_contact.transient import (
    ContactNodalLoadHistory, ContactTransientSolverControls, solve_transient_contact,
)
from test_general_contact_beam_plane import _model

CONTROLS = ContactTransientSolverControls(allow_classical_beam_dev=True,
                                         tool_work_absolute_tolerance_j=1e-12)


def impact(*, quadratic=False, end=.013, points=None, checkpoint=None, emit=False, history=None, dt=.0001):
    model, definition, settings = _model(quadratic=quadratic)
    model.add_boundary_condition(BoundaryCondition("normal_guide", list(model.mesh.nodes),
        {name: 0. for name in ("ux", "uy", "rx", "ry", "rz")}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes:
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.3
    motion = ContactRigidTranslationHistory("plane", points or ((0., 0., 0., 0.), (.013, 0., 0., .0026)))
    load = None
    if history is not None:
        load = LoadCase("normal_load")
        for node, weight in zip(model.mesh.nodes, (1/6, 2/3, 1/6) if quadratic else (.5, .5)):
            load.add_nodal_load(node, forces=(0., 0., -1e-4*weight))
    result = solve_transient_contact(model, definition, TransientConfig(dt, end, initial_velocity=initial),
        contact_settings=settings, solver_controls=CONTROLS, rigid_translation_histories=(motion,),
        restart_checkpoint=checkpoint, emit_restart_checkpoint=emit, base_load_case=load, nodal_load_history=history)
    return model, result, motion


def test_moving_tool_elastic_energy_accounting_improves_with_time_refinement():
    defects = []
    for dt in (.0001, .00005):
        model, result, _ = impact(dt=dt)
        dof = model.mesh.dof_manager.get_node_dofs(1)[2]
        initial = .5*.001*.3**2
        final = .5*.001*result.velocities[-1, dof]**2+result.contact[-1].line_contact_energy
        defect = abs(final-initial-result.prescribed_tool_work_j[-1]["plane"])/initial
        defects.append(defect)
    assert defects[1] < defects[0]
    assert defects[1] <= .01


def polynomial_work(q0, q1, v0, v1, p0, p1, dt, stiffness):
    # Independently integrate clipped cubic gap, split at all real polynomial roots.
    gap = np.array([.005+q0-p0, dt*v0-(p1-p0),
        -3*q0-2*dt*v0+3*q1-dt*v1, 2*q0+dt*v0-2*q1+dt*v1])
    roots = np.polynomial.polynomial.polyroots(gap)
    cuts = sorted([0., 1.]+[float(r.real) for r in roots if abs(r.imag) < 1e-10 and 0. < r.real < 1.])
    primitive = np.polynomial.polynomial.polyint(gap)
    integral = 0.
    for a, b in zip(cuts, cuts[1:]):
        if np.polynomial.polynomial.polyval(.5*(a+b), gap) < 0.:
            integral += np.polynomial.polynomial.polyval(b, primitive)-np.polynomial.polynomial.polyval(a, primitive)
    return -stiffness*(p1-p0)*integral


@pytest.mark.parametrize("quadratic", [False, True])
def test_uniform_translating_tool_matches_forced_newmark_impulse_and_work(quadratic):
    model, result, motion = impact(quadratic=quadratic)
    assert "PRESCRIBED_TRANSLATION" in result.profile
    q, v, acceleration, previous_force, work = 0., -.3, 0., 0., 0.
    dofs = [model.mesh.dof_manager.get_node_dofs(n)[2] for n in model.mesh.nodes]
    for i, dt in enumerate(np.diff(result.times), 1):
        p0, p1 = .2*result.times[i-1], .2*result.times[i]
        predictor = q+dt*v+.25*dt*dt*acceleration
        new_q = predictor if .005+predictor-p1 >= 0. else (.001*predictor+.25*dt*dt*1e4*(p1-.005))/(.001+.25*dt*dt*1e4)
        new_a = (new_q-predictor)/(.25*dt*dt)
        new_v = v+.5*dt*(acceleration+new_a)
        force = max(-1e4*(.005+new_q-p1), 0.)
        np.testing.assert_allclose(result.displacements[i, dofs], new_q, atol=1e-10, rtol=0.)
        np.testing.assert_allclose(result.velocities[i, dofs], new_v, atol=1e-8, rtol=0.)
        assert result.contact[i].beam_force[2] == pytest.approx(force, abs=1e-7)
        assert .001*(new_v-v) == pytest.approx(.5*dt*(previous_force+force), abs=1e-11)
        work += polynomial_work(q, new_q, v, new_v, p0, p1, dt, 1e4)
        assert result.prescribed_tool_work_j[i]["plane"] == pytest.approx(work, abs=1e-10)
        q, v, acceleration, previous_force = new_q, new_v, new_a, force
    assert work > 0.
    assert np.all(result.external_nodal_work_j == 0.)
    assert result.prescribed_tool_work_error_j[-1]["plane"] >= 0.
    assert result.swept_minimum_gaps.min() >= -.002
    assert active_contact_manager() is None


def test_motion_knots_open_recontact_and_schema5_split_restart():
    points = ((0., 0., 0., 0.), (.012, 0., 0., .0024), (.017, 0., 0., -.001), (.024, 0., 0., .020))
    history = ContactNodalLoadHistory(((0., 0.), (.004, 1.), (.024, 1.)))
    model, direct, _ = impact(quadratic=True, end=.024, points=points, history=history)
    _, prefix, _ = impact(quadratic=True, end=.0102, points=points, emit=True, history=history)
    assert prefix.restart_checkpoint["version"] == 5
    _, resumed, _ = impact(quadratic=True, end=.024, points=points, checkpoint=prefix.restart_checkpoint, history=history)
    for field in ("times", "displacements", "velocities", "accelerations", "swept_minimum_gaps", "external_nodal_work_j"):
        np.testing.assert_allclose(np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])),
                                   getattr(direct, field), atol=1e-9, rtol=0., err_msg=field)
    for field in ("prescribed_tool_work_j", "prescribed_tool_work_error_j"):
        np.testing.assert_allclose([p["plane"] for p in getattr(prefix, field)+getattr(resumed, field)[1:]],
                                   [p["plane"] for p in getattr(direct, field)], atol=1e-10, rtol=0.)
    for time in (.004, .012, .017):
        assert np.any(direct.times == time)
    active = [bool(c.active_line_cells) for c in direct.contact]
    first = active.index(True)
    opening = active.index(False, first)
    assert any(active[opening+1:])
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    work = 0.
    for i, dt in enumerate(np.diff(direct.times), 1):
        work += polynomial_work(direct.displacements[i-1, dof], direct.displacements[i, dof],
            direct.velocities[i-1, dof], direct.velocities[i, dof],
            np.interp(direct.times[i-1], [p[0] for p in points], [p[3] for p in points]),
            np.interp(direct.times[i], [p[0] for p in points], [p[3] for p in points]), dt, 1e4)
    assert direct.prescribed_tool_work_j[-1]["plane"] == pytest.approx(work, abs=1e-10)


@pytest.mark.parametrize("grazing", [False, True])
def test_unsampled_contact_pulse_work_and_pose_restoration(grazing):
    model, definition, settings = _model()
    manager = _BeamPlaneManager(model, definition, settings)
    motion = ContactRigidTranslationHistory("plane", ((0., 0., 0., 0.), (1., 0., 0., .002)))
    owner = _PlaneTranslations(manager, (motion,))
    owner.set_pose(1.)
    saved = manager.point.copy()
    vectors = [np.zeros(model.mesh.dof_manager.total_dofs) for _ in range(4)]
    a, center, depth = .01, .31, 1e-6
    if grazing:
        gauss = .5+.5*np.polynomial.legendre.leggauss(5)[0][1]
        depth = a*(center-gauss+1e-14)**2
    for node in model.mesh.nodes:
        i = model.mesh.dof_manager.get_node_dofs(node)[2]
        vectors[0][i], vectors[1][i] = a*center**2-depth-.005, a*(1-center)**2-depth-.005+.002
        vectors[2][i], vectors[3][i] = -2*a*center+.002, 2*a*(1-center)+.002
    expected = 1e4*.002*(4/3)*depth*np.sqrt(depth/a)
    work, error = owner.work(*vectors, 0., 1., CONTROLS, lambda: None)
    assert work["plane"] == pytest.approx(expected, abs=1e-10)
    np.testing.assert_array_equal(manager.point, saved)
    with pytest.raises(RuntimeError, match="subdivision limit"):
        owner.work(*vectors, 0., 1., replace(CONTROLS, maximum_tool_work_subdivisions=0), lambda: None)
    np.testing.assert_array_equal(manager.point, saved)


def test_tool_knot_steps_fit_minimum_dt_and_impossible_partitions_reject():
    model, definition, settings = _model()
    model.add_boundary_condition(BoundaryCondition("normal_guide", list(model.mesh.nodes),
        {name: 0. for name in ("ux", "uy", "rx", "ry", "rz")}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    motion = ContactRigidTranslationHistory("plane", ((0., 0., 0., 0.), (.04, 0., 0., .0004)))
    controls = replace(CONTROLS, minimum_dt=.02)
    result = solve_transient_contact(model, definition, TransientConfig(.03, .04, initial_velocity=initial),
        contact_settings=settings, solver_controls=controls, rigid_translation_histories=(motion,))
    np.testing.assert_allclose(result.times, [0., .02, .04], atol=0., rtol=0.)
    bad = ContactRigidTranslationHistory("plane", ((0., 0., 0., 0.), (.03, 0., 0., .0003)))
    with pytest.raises(ValueError, match="cannot fit dt and minimum_dt"):
        solve_transient_contact(model, definition, TransientConfig(.025, .03, initial_velocity=initial),
            contact_settings=settings, solver_controls=controls, rigid_translation_histories=(bad,))


def test_decimal_tool_knot_intervals_share_runtime_roundoff_policy():
    model, definition, settings = _model()
    model.add_boundary_condition(BoundaryCondition("normal_guide", list(model.mesh.nodes),
        {name: 0. for name in ("ux", "uy", "rx", "ry", "rz")}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    motion = ContactRigidTranslationHistory("plane", ((0., 0., 0., 0.),
        (.11, 0., 0., 0.), (.17, 0., 0., 0.)))
    result = solve_transient_contact(model, definition,
        TransientConfig(.03, .17, initial_velocity=initial), contact_settings=settings,
        solver_controls=replace(CONTROLS, minimum_dt=.025), rigid_translation_histories=(motion,))
    assert result.times[-1] == .17
    assert .11 in result.times
    assert np.min(np.diff(result.times)) >= .025-128*np.finfo(float).eps
    assert np.max(np.diff(result.times)) <= .03+128*np.finfo(float).eps


@pytest.mark.parametrize("shifted", [False, True])
def test_tool_knots_outside_end_and_arithmetic_minimum_remainders(shifted):
    model, definition, settings = _model()
    model.add_boundary_condition(BoundaryCondition("normal_guide", list(model.mesh.nodes),
        {name: 0. for name in ("ux", "uy", "rx", "ry", "rz")}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    motion = ContactRigidTranslationHistory("plane", ((0., 0., 0., 0.), (.19, 0., 0., .0019), (.23, 0., 0., .0023))
        if shifted else ((0., 0., 0., 0.), (.04, 0., 0., .0004)))
    end = .23 if shifted else .039
    result = solve_transient_contact(model, definition, TransientConfig(.03, end, initial_velocity=initial),
        contact_settings=settings, solver_controls=replace(CONTROLS, minimum_dt=.02),
        rigid_translation_histories=(motion,))
    assert result.times[-1] == end
    if shifted:
        assert np.all(np.diff(result.times) >= .02-128*np.finfo(float).eps)


def test_deforming_b3_translation_preserves_normal_momentum_and_curved_gap():
    model, definition, settings = _model(quadratic=True)
    model.add_boundary_condition(BoundaryCondition("guide", list(model.mesh.nodes),
        {name: 0. for name in ("ux", "uy", "rx", "rz")}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    dofs = [model.mesh.dof_manager.get_node_dofs(n)[2] for n in model.mesh.nodes]
    initial[dofs] = [-.1, -.35, -.1]
    motion = ContactRigidTranslationHistory("plane", ((0., 0., 0., 0.), (.022, 0., 0., .0022)))
    result = solve_transient_contact(model, definition, TransientConfig(.0001, .022, initial_velocity=initial),
        contact_settings=settings, solver_controls=CONTROLS, rigid_translation_histories=(motion,))
    weights = .001*np.array([1/6, 2/3, 1/6])
    for i, dt in enumerate(np.diff(result.times), 1):
        impulse = .5*dt*(result.contact[i-1].beam_force[2]+result.contact[i].beam_force[2])
        assert weights @ (result.velocities[i, dofs]-result.velocities[i-1, dofs]) == pytest.approx(impulse, abs=1e-11)
        h = .005+result.displacements[i, dofs]-.1*result.times[i]
        a, b, c = 2*(h[0]+h[2]-2*h[1]), 4*h[1]-3*h[0]-h[2], h[0]
        minimum = min(h[0], h[2])
        if a > 0. and 0. < -b/(2*a) < 1.:
            minimum = min(minimum, c-b*b/(4*a))
        assert result.contact[i].minimum_gap == pytest.approx(minimum, abs=1e-10)
    assert any(0. < c.active_line_length < .99 for c in result.contact)
    assert np.max(np.abs(result.displacements[:, [model.mesh.dof_manager.get_node_dofs(n)[4] for n in model.mesh.nodes]])) > 0.


def test_translation_admission_and_restart_history_binding():
    import copy
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    _, prefix, _ = impact(end=.0102, emit=True)
    with pytest.raises(ValueError, match="checkpoint analysis is incompatible"):
        impact(checkpoint=prefix.restart_checkpoint,
               points=((0., 0., 0., 0.), (.013, 0., 0., .0025)))
    with pytest.raises(ValueError, match="zero offset"):
        ContactRigidTranslationHistory("plane", ((0., 0., 0., .001), (1., 0., 0., .002)))
    with pytest.raises(ValueError, match="increasing"):
        ContactRigidTranslationHistory("plane", ((0., 0., 0., 0.), (0., 0., 0., .002)))
    with pytest.raises(ValueError, match="cover t_end"):
        impact(points=((0., 0., 0., 0.), (.01, 0., 0., .002)))
    with pytest.raises(ValueError, match="combined tool/load history knots"):
        impact(history=ContactNodalLoadHistory(((0., 0.), (.012999999, 1.), (.02, 1.))))
    malformed = copy.deepcopy(prefix.restart_checkpoint)
    malformed["state"]["tool_work_error_j"]["plane"] = -1.
    malformed["checkpoint_sha256"] = analysis_configuration_fingerprint(
        {k: v for k, v in malformed.items() if k != "checkpoint_sha256"})
    with pytest.raises(ValueError, match="checkpoint tool work is invalid"):
        impact(checkpoint=malformed)


def test_translating_plane_bounds_reproduce_relative_polynomial_extrema():
    from anyfesolver.general_contact.beam_temporal_bounds import beam_hermite_gap_bounds
    model, definition, settings = _model(quadratic=True)
    manager = _BeamPlaneManager(model, definition, settings)
    q0, q1, v0, v1 = [np.zeros(model.mesh.dof_manager.total_dofs) for _ in range(4)]
    for node in model.mesh.nodes:
        i = model.mesh.dof_manager.get_node_dofs(node)[2]
        q0[i], q1[i], v0[i], v1[i] = -.003, -.0035, -.005, .003
    bounds = beam_hermite_gap_bounds(manager, q0, q1, v0, v1, 1., tolerance_m=1e-10,
        maximum_subdivisions=512, plane_start_point=(0., 0., 0.), plane_end_point=(0., 0., .002))
    coefficients = np.array([.002, -.007, .0055, -.001])
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
    probes = [0., 1.]+[r.real for r in roots if abs(r.imag) < 1e-12 and 0. < r.real < 1.]
    exact = min(np.polynomial.polynomial.polyval(s, coefficients) for s in probes)
    assert bounds.lower_gap <= exact <= bounds.upper_gap
    assert bounds.upper_gap-bounds.lower_gap <= 1e-10


def test_two_translating_planes_preserve_named_work_and_split_restart():
    from test_general_contact_transient_multiple_beam_planes import fixture
    histories = (ContactRigidTranslationHistory("first", ((0., 0., 0., 0.), (.018, 0., .0036, 0.))),
                 ContactRigidTranslationHistory("second", ((0., 0., 0., 0.), (.018, 0., 0., .0018))))
    def run(end, checkpoint=None, emit=False):
        model, definition, settings, initial = fixture(quadratic=True, orthogonal=True)
        result = solve_transient_contact(model, definition, TransientConfig(.0001, end, initial_velocity=initial),
            contact_settings=settings, solver_controls=replace(CONTROLS, allow_multiple_beam_planes_dev=True),
            rigid_translation_histories=histories[::-1], restart_checkpoint=checkpoint, emit_restart_checkpoint=emit)
        return model, result
    model, result = run(.018)
    _, prefix = run(.0146, emit=True)
    _, resumed = run(.018, checkpoint=prefix.restart_checkpoint)
    np.testing.assert_allclose(np.concatenate((prefix.displacements, resumed.displacements[1:])),
                               result.displacements, atol=1e-10, rtol=0.)
    for axis, body, speed, stiffness in ((1, "first", .2, 3e3), (2, "second", .1, 7e3)):
        dof = model.mesh.dof_manager.get_node_dofs(1)[axis]
        expected = sum(polynomial_work(result.displacements[i-1, dof], result.displacements[i, dof],
            result.velocities[i-1, dof], result.velocities[i, dof], speed*result.times[i-1], speed*result.times[i], dt, stiffness)
            for i, dt in enumerate(np.diff(result.times), 1))
        assert result.prescribed_tool_work_j[-1][body] == pytest.approx(expected, abs=1e-10)
        assert expected > 0.
        assert resumed.prescribed_tool_work_j[-1][body] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("near_terminal", [False, True])
def test_nearest_combined_knot_cannot_be_overridden_by_roundoff_snapping(near_terminal):
    from test_general_contact_transient_multiple_beam_planes import fixture
    model, definition, settings, initial = fixture(orthogonal=True)
    first, second, end = (1e8, 1e8+5e-7, 1e8+5e-7) if near_terminal else (1e8, 1e8+2e-6, 1e8+1e-4)
    motions = (ContactRigidTranslationHistory("first", ((0., 0., 0., 0.), (first, 0., .0001, 0.), (end, 0., 0., 0.))),
               ContactRigidTranslationHistory("second", ((0., 0., 0., 0.), (second, 0., 0., 0.))
                   if near_terminal else ((0., 0., 0., 0.), (second, 0., 0., 0.), (end, 0., 0., 0.))))
    result = solve_transient_contact(model, definition, TransientConfig(1e8, end, initial_velocity=initial*0),
        contact_settings=settings, solver_controls=replace(CONTROLS, allow_multiple_beam_planes_dev=True,
            minimum_dt=1e-8 if near_terminal else 1e-6),
        rigid_translation_histories=motions)
    assert first in result.times and second in result.times
    assert result.times[-1] == end


@pytest.mark.parametrize("cancel", [False, True])
def test_tool_work_failure_keeps_committed_state_and_cleans_scope(monkeypatch, cancel):
    import anyfesolver.general_contact.transient as driver
    from anyfesolver.control import CancellationToken, SolveCancelled
    token = CancellationToken()
    commits = []
    commit = driver.commit_state_candidate
    def record(*args, **kwargs):
        commits.append(1)
        return commit(*args, **kwargs)
    def fail(*args, **kwargs):
        if cancel:
            token.cancel()
            raise SolveCancelled("injected tool work cancellation")
        raise RuntimeError("injected tool work failure")
    monkeypatch.setattr(driver, "commit_state_candidate", record)
    monkeypatch.setattr(_PlaneTranslations, "work", fail)
    with pytest.raises(SolveCancelled if cancel else RuntimeError):
        impact()
    assert len(commits) == 1
    assert active_contact_manager() is None
