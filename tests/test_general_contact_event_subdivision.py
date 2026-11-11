"""Independent branch-time references and actual physical event commits."""
import math
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.event_subdivision import EventNeedsCutback, localize_branch_event
from anyfesolver.general_contact.transient import ContactNodalLoadHistory, ContactTransientSolverControls, solve_transient_contact
from test_general_contact_transient_plane import _guided_s3_descriptor_impact, _uniform_s3_normal_load


CONTROLS = ContactTransientSolverControls(
    allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=True,
    s3_algebraic_path_dev=True, s3_event_subdivision_dev=True,
)


@pytest.mark.parametrize('boundary,gap,elapsed,expected', [
    ((0,),1e-12,1e-12,1),
    ((),1e-12,1e-12,0),
    ((1,),1e-12,1e-12,0),
    ((0,),2e-10,1e-12,0),
    ((0,),1e-12,2e-10,0),
])
def test_resolved_boundary_requires_membership_and_both_dimensional_budgets(boundary,gap,elapsed,expected):
    from anyfesolver.general_contact.transient import _event_continuation_segments
    first = SimpleNamespace(end=elapsed,active=(False,False))
    second = SimpleNamespace(end=1.,active=(True,False))
    path = SimpleNamespace(segments=(first,second))
    original = path.segments
    remaining = _event_continuation_segments(path,1.,boundary,np.array([gap,0.]),1e-10,1e-10)
    assert remaining == original[expected:]
    assert path.segments is original  # force/work/minimum path is not replaced


@pytest.mark.parametrize('boundary', [[True],[0,0],[1,0],[-1],[3],'0'])
def test_restart_rejects_malformed_boundary_even_with_valid_integrity(boundary):
    from anyfesolver.general_contact.transient import _new_restart_checkpoint, _read_restart_checkpoint
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    checkpoint = _new_restart_checkpoint('model','analysis',.01,[0.],[0.],[0.],1,
        path_terminal_active=(False,False,False),event_boundary_support=())
    checkpoint['state']['event_boundary_support'] = boundary
    checkpoint['checkpoint_sha256'] = analysis_configuration_fingerprint(
        {k:v for k,v in checkpoint.items() if k != 'checkpoint_sha256'})
    with pytest.raises(ValueError,match='event boundary is invalid'):
        _read_restart_checkpoint(checkpoint,'model','analysis',1,.02,
            load_history=None,path_pressure_dofs=3,event_subdivision=True)


@pytest.mark.parametrize('force', [0., -2.])
def test_free_flight_event_matches_independent_accelerated_root(force):
    result = localize_branch_event(
        np.diag([0.,2.]), np.diag([1.,0.]), [[1.,1.]], [0.], [3.], [0.,0.],
        [.1,0.], [-1.,0.], [force,0.], [force,0.], [force,0.], .2, (False,), (0,),
        minimum_dt=1e-8, time_tolerance=1e-12, gap_tolerance=1e-12,
        null_basis=[[0.],[1.]],
    )
    expected = .1 if not force else .2/(1+math.sqrt(1-.2*force))
    assert result.duration == pytest.approx(expected, abs=1e-12)
    assert result.support_indices == (0,)
    assert result.endpoint_evaluations > 2


def test_release_event_matches_independent_condensed_newmark_root():
    # Algebraic a=-3x/5; physical stiffness=6/5, mass=1.
    event = localize_branch_event(
        np.diag([0.,2.]), np.diag([1.,0.]), [[1.,1.]], [0.], [3.], [0.,0.],
        [-.1,.06], [1.,-.6], [.12,-.072], [0.,0.], [0.,0.], .2, (True,), (0,),
        minimum_dt=1e-8, time_tolerance=1e-12, gap_tolerance=1e-12,
        null_basis=[[0.],[1.]],
    )
    expected = .2/(1+math.sqrt(1+.012))
    assert event.duration == pytest.approx(expected, abs=1e-12)


def test_double_crossing_has_isolated_duration_roots_despite_open_endpoints():
    # Open fixed-branch path 1-8s+8s² crosses twice although endpoints are open.
    event = localize_branch_event(
            [[0.]], [[1.]], [[1.]], [0.], [1.], [0.],
            [1.], [-4.], [0.], [0.], [8.], 2., (False,), (0,),
            minimum_dt=1e-8, time_tolerance=1e-12, gap_tolerance=1e-12,
        )
    reference = min(root.real for root in np.roots([1.,0.,-4.,1.])
                    if abs(root.imag) < 1e-12 and root.real > 0.)
    assert event.duration == pytest.approx(reference,abs=1e-12)


def test_multimode_endpoint_family_selects_earliest_root_not_broad_bracket_root():
    # Independent numerator has roots h²=.1,.2,25/18. Every selected root
    # would pass the reconstructed path 1-s²; duration roots must be isolated.
    K = np.diag([4.,16.,36.]); q = np.array([2.365,-4.956,3.591])
    event = localize_branch_event(
        K,np.eye(3),[[1.,1.,1.]],[0.],[1.],[0.,0.,0.],q,np.zeros(3),-K @ q,
        np.zeros(3),np.zeros(3),1.3,(False,),(0,), minimum_dt=1e-8,
        time_tolerance=1e-12,gap_tolerance=1e-10,
    )
    assert event.duration == pytest.approx(math.sqrt(.1),abs=1e-11)


def _solve(end, checkpoint=None, *, controls=CONTROLS, load=False, emit=False):
    model, definition, velocity = _guided_s3_descriptor_impact()
    result = solve_transient_contact(
        model, definition, TransientConfig(.0013, end, initial_velocity=velocity),
        base_load_case=_uniform_s3_normal_load() if load else None,
        solver_controls=controls, restart_checkpoint=checkpoint, emit_restart_checkpoint=emit,
    )
    return model, result


def test_public_s3_integrates_activation_and_release_with_scalar_energy_reference():
    model, result = _solve(.016)
    assert result.profile.endswith('NEWMARK_EVENT_SUBDIVISION_DEV_V1')
    assert len(result.localized_contact_event_times) >= 2
    assert result.localized_contact_event_times[0] == pytest.approx(.01, abs=1e-10)
    assert result.event_endpoint_evaluations > 0
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    # Independent rigid-mode Newmark endpoint equations, including events.
    m, k = .025, 50000.
    gap, velocity, acceleration = .01, -1., 0.
    energies = []
    for i,h in enumerate(np.diff(result.times),1):
        predictor = gap+h*velocity+.25*h*h*acceleration
        next_gap = predictor if predictor >= 0. else predictor/(1+.25*h*h*k/m)
        next_acceleration = (next_gap-predictor)/(.25*h*h)
        velocity += .5*h*(acceleration+next_acceleration)
        acceleration, gap = next_acceleration, next_gap
        assert .01+result.displacements[i,dof] == pytest.approx(gap, abs=1e-10)
        assert result.velocities[i,dof] == pytest.approx(velocity, abs=1e-8)
        assert result.accelerations[i,dof] == pytest.approx(acceleration, abs=1e-5)
        energies.append(.5*m*result.velocities[i,dof]**2
                        +.5*k*min(.01+result.displacements[i,dof],0.)**2)
    assert max(abs(e-.5*m) for e in energies)/(.5*m) < 1e-6
    assert all(c.gap_tolerance_met for c in result.contact)
    assert result.cutbacks <= CONTROLS.maximum_cutbacks
    for event in result.localized_contact_event_times:
        assert np.any(result.times == event), 'event must be a physical accepted endpoint'


def test_public_loaded_activation_uses_accelerated_flight():
    model, result = _solve(.011, load=True)
    expected = .02/(1+math.sqrt(1+.016))
    assert result.localized_contact_event_times[0] == pytest.approx(expected, abs=1e-10)
    index = int(np.flatnonzero(result.times == result.localized_contact_event_times[0])[0])
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    assert result.displacements[index,dof] == pytest.approx(-.01, abs=1e-10)
    assert result.velocities[index,dof] == pytest.approx(-1.-.8*expected, abs=1e-8)


def test_event_boundary_split_restart_and_incompatible_controls():
    _, direct = _solve(.016)
    split = direct.localized_contact_event_times[0]
    _, prefix = _solve(split, emit=True)
    _, resumed = _solve(.016, prefix.restart_checkpoint)
    for field in ('times','displacements','velocities','accelerations','external_nodal_work_j','swept_minimum_gaps'):
        np.testing.assert_allclose(np.concatenate((getattr(prefix,field),getattr(resumed,field)[1:])),
                                   getattr(direct,field), rtol=0., atol=1e-10,err_msg=field)
    assert prefix.localized_contact_event_times+resumed.localized_contact_event_times == direct.localized_contact_event_times
    with pytest.raises(ValueError, match='checkpoint .* incompatible'):
        _solve(.016, prefix.restart_checkpoint, controls=replace(CONTROLS, s3_event_subdivision_dev=False))


@pytest.mark.parametrize('case', ['release','tilted','history'])
def test_resolved_boundary_restart_preserves_full_trajectory(case):
    end = .012 if case == 'tilted' else .016
    history = ContactNodalLoadHistory(((0.,0.),(.009,1.),(.02,1.))) if case == 'history' else None
    def solve(stop,checkpoint=None,emit=False):
        model,definition,initial = _guided_s3_descriptor_impact(
            **(dict(tilt=.002,span=.1) if case == 'tilted' else {}))
        return solve_transient_contact(model,definition,
            TransientConfig(.0002 if case == 'tilted' else .0013,stop,initial_velocity=initial),
            base_load_case=_uniform_s3_normal_load() if history is not None else None,
            nodal_load_history=history,solver_controls=CONTROLS,
            restart_checkpoint=checkpoint,emit_restart_checkpoint=emit)
    direct = solve(end)
    split = direct.localized_contact_event_times[1]
    prefix = solve(split,emit=True)
    assert prefix.restart_checkpoint['version'] == 4
    assert prefix.restart_checkpoint['state']['event_boundary_support']
    resumed = solve(end,prefix.restart_checkpoint)
    for field in ('times','displacements','velocities','accelerations',
                  'external_nodal_work_j','swept_minimum_gaps'):
        np.testing.assert_allclose(np.concatenate((getattr(prefix,field),getattr(resumed,field)[1:])),
                                   getattr(direct,field),rtol=0.,atol=1e-10,err_msg=case+' '+field)


def test_restart_rejects_a_boundary_inconsistent_with_its_restored_geometry():
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    _,direct = _solve(.016)
    _,prefix = _solve(direct.localized_contact_event_times[0],emit=True)
    checkpoint = deepcopy(prefix.restart_checkpoint)
    checkpoint['state']['q'][0] += 1e-4
    checkpoint['checkpoint_sha256'] = analysis_configuration_fingerprint(
        {k:v for k,v in checkpoint.items() if k != 'checkpoint_sha256'})
    with pytest.raises(ValueError,match='event boundary exceeds its gap budget'):
        _solve(.016,checkpoint)


def test_event_option_requires_path_and_dimensioned_budgets():
    with pytest.raises(ValueError, match='algebraic-path opt-in'):
        ContactTransientSolverControls(s3_event_subdivision_dev=True)
    for name in ('s3_event_time_tolerance_s','s3_event_gap_tolerance_m'):
        with pytest.raises(ValueError, match='finite and positive'):
            replace(CONTROLS, **{name: float('nan')})


def test_exact_isolated_point_departure_is_not_discarded(monkeypatch):
    import anyfesolver.general_contact.event_subdivision as events
    # Independent cubic roots .5,.75,1; exercise the isolator's exact-point
    # interface without relying on accidental floating-point subdivision ties.
    monkeypatch.setattr(events,'_root_intervals',lambda *a: ((.5,.5),(.75,.75),(1.,1.)))
    event = localize_branch_event([[0.]],[[1.]],[[1.]],[0.],[1.],[0.],
        [.375],[-1.625],[4.5],[4.5],[.5],1.,(False,),(0,),
        minimum_dt=1e-8,time_tolerance=1e-12,gap_tolerance=1e-12)
    assert event.duration == pytest.approx(.5,abs=1e-12)


def test_endpoint_gap_proximity_cannot_make_an_arbitrary_truncated_checkpoint():
    model,definition,velocity = _guided_s3_descriptor_impact()
    velocity *= .001
    # True free-flight root=10s; endpoint is5e-8s early but within gap budget.
    with pytest.raises(ValueError,match='untruncated final time step'):
        solve_transient_contact(model,definition,
            TransientConfig(3.,10.-5e-8,initial_velocity=velocity),
            solver_controls=CONTROLS,emit_restart_checkpoint=True)


def test_tilted_s3_multiple_support_events_match_independent_skin_average():
    model,definition,initial = _guided_s3_descriptor_impact(tilt=.002,span=.1)
    result = solve_transient_contact(model,definition,
        TransientConfig(.0002,.012,initial_velocity=initial),solver_controls=CONTROLS)
    assert result.localized_contact_event_times
    element = model.mesh.elements[1]
    dofs = np.array([model.mesh.dof_manager.get_node_dofs(i) for i in (1,2,3)])
    normal = np.asarray(definition.rigid_bodies[0].normal)
    director = np.asarray(element.reference_normal)
    offset = -.5*element.thickness-element.reference_surface_offset
    points = np.array([[model.mesh.nodes[i].x,model.mesh.nodes[i].y,model.mesh.nodes[i].z] for i in (1,2,3)])
    corner0 = (points+offset*director-np.asarray(definition.rigid_bodies[0].point)) @ normal
    for time in result.localized_contact_event_times:
        index = int(np.flatnonzero(result.times == time)[0])
        displacements = result.displacements[index,dofs]
        gaps = corner0+displacements[:,:3] @ normal+displacements[:,3:] @ (offset*np.cross(director,normal))
        weighted = (np.eye(3)+np.ones((3,3))) @ gaps/4
        assert min(abs(weighted)) <= CONTROLS.s3_event_gap_tolerance_m
    assert all(c.gap_tolerance_met for c in result.contact)


def test_indefinite_endpoint_family_and_exhausted_root_budget_fail_explicitly():
    args = ([[0.]],[[1.]],[[1.]],[0.],[1.],[0.],[.1],[-1.],[0.],[0.],[0.],.2,(False,),(0,))
    limits = dict(minimum_dt=1e-8,time_tolerance=1e-12,gap_tolerance=1e-12)
    with pytest.raises(EventNeedsCutback,match='not positive'):
        localize_branch_event([[-1.]],*args[1:],**limits)
    with pytest.raises(EventNeedsCutback,match='localization exhausted'):
        localize_branch_event(*args,**limits,maximum_iterations=1)


def test_public_event_history_preserves_independent_newmark_and_affine_load_work():
    model,definition,initial = _guided_s3_descriptor_impact()
    history = ContactNodalLoadHistory(((0.,0.),(.009,1.),(.02,1.)))
    result = solve_transient_contact(
        model,definition,TransientConfig(.0013,.016,initial_velocity=initial),
        base_load_case=_uniform_s3_normal_load(),nodal_load_history=history,solver_controls=CONTROLS,
    )
    assert result.profile.endswith('EVENT_SUBDIVISION_HISTORY_DEV_V1')
    assert np.any(result.times == .009)
    assert len(result.localized_contact_event_times) >= 2
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    gap,velocity,acceleration,work = .01,-1.,0.,0.
    for i,h in enumerate(np.diff(result.times),1):
        f0 = -.02*min(result.times[i-1]/.009,1.)
        f1 = -.02*min(result.times[i]/.009,1.)
        predictor = gap+h*velocity+.25*h*h*acceleration
        free = predictor+.25*h*h*f1/.025
        next_gap = free if free >= 0. else free/(1+.25*h*h*50000./.025)
        delta = next_gap-gap
        work += f0*delta+(f1-f0)*((2/3)*delta-(1/6)*h*velocity)
        next_acceleration = (next_gap-predictor)/(.25*h*h)
        velocity += .5*h*(acceleration+next_acceleration)
        gap,acceleration = next_gap,next_acceleration
        assert result.displacements[i,dof] == pytest.approx(gap-.01,abs=1e-10)
        assert result.external_nodal_work_j[i] == pytest.approx(work,abs=1e-12)


@pytest.mark.parametrize('cancel',[False,True])
def test_event_failure_or_cancellation_keeps_pre_event_commit_boundary(monkeypatch,cancel):
    import anyfesolver.general_contact.transient as driver
    from anyfesolver.control import CancellationToken, SolveCancelled
    from anyfesolver.general_contact.core import active_contact_manager
    commits = []
    original = driver.commit_state_candidate
    def commit(*args,**kwargs):
        commits.append(True)
        return original(*args,**kwargs)
    token = CancellationToken()
    original_event = driver.localize_branch_event
    def localize(*args,**kwargs):
        if not cancel:
            raise RuntimeError('injected event localization failure')
        token.cancel('inside event localization')
        return original_event(*args,**kwargs)
    monkeypatch.setattr(driver,'commit_state_candidate',commit)
    monkeypatch.setattr(driver,'localize_branch_event',localize)
    model,definition,initial = _guided_s3_descriptor_impact()
    with pytest.raises(SolveCancelled if cancel else RuntimeError,
                       match='inside event localization' if cancel else 'injected event localization'):
        solve_transient_contact(model,definition,TransientConfig(.0013,.016,initial_velocity=initial),
                                solver_controls=CONTROLS,cancellation_token=token)
    assert len(commits) == 8  # initialization plus seven accepted open steps
    assert active_contact_manager() is None
