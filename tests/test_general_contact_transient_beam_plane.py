"""Classical beam impact through the shared Newmark contact owner."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import BoundaryCondition, LoadCase
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.core import active_contact_manager
from anyfesolver.general_contact.transient import (
    ContactNodalLoadHistory,ContactTransientSolverControls,solve_transient_contact,
)
from test_general_contact_beam_plane import _model


CONTROLS = ContactTransientSolverControls(allow_classical_beam_dev=True)


def _impact(quadratic=False,*,end=.013,checkpoint=None,emit=False,history=None):
    model,definition,settings = _model(quadratic=quadratic)
    nodes = list(model.mesh.nodes)
    model.add_boundary_condition(BoundaryCondition('normal_guide',nodes,
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in nodes:
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.5
    load = None
    if history is not None:
        # Exact uniform body acceleration load integrated on the beam's own
        # shape functions; independently specified 1/2 or 1/6,2/3,1/6 weights.
        load = LoadCase('uniform_normal_load')
        for node,weight in zip(nodes,(1/6,2/3,1/6) if quadratic else (.5,.5)):
            load.add_nodal_load(node,forces=(0.,0.,-.0001*weight))
    result = solve_transient_contact(model,definition,
        TransientConfig(.0001,end,initial_velocity=initial),contact_settings=settings,
        solver_controls=CONTROLS,base_load_case=load,nodal_load_history=history,
        restart_checkpoint=checkpoint,emit_restart_checkpoint=emit)
    return model,result


@pytest.mark.parametrize('quadratic',[False,True])
def test_uniform_beam_impact_matches_independent_scalar_newmark_and_impulse(quadratic):
    model,result = _impact(quadratic)
    assert result.profile.endswith('ROUND_TUBE_VON_KARMAN_FIXED_PLANE_NEWMARK_DEV_V1')
    # Independent translational rigid mode: rho*A*L=.001kg, k_line*L=1e4N/m.
    gap,velocity,acceleration,mass,stiffness = .005,-.5,0.,.001,1e4
    previous_force = 0.
    dofs = [model.mesh.dof_manager.get_node_dofs(n)[2] for n in model.mesh.nodes]
    for i,h in enumerate(np.diff(result.times),1):
        predictor = gap+h*velocity+.25*h*h*acceleration
        new_gap = predictor if predictor >= 0. else predictor/(1+.25*h*h*stiffness/mass)
        new_a = (new_gap-predictor)/(.25*h*h)
        new_v = velocity+.5*h*(acceleration+new_a)
        np.testing.assert_allclose(.005+result.displacements[i,dofs],new_gap,rtol=0.,atol=1e-10)
        np.testing.assert_allclose(result.velocities[i,dofs],new_v,rtol=0.,atol=1e-8)
        force = max(-stiffness*new_gap,0.)
        assert result.contact[i].beam_force[2] == pytest.approx(force,abs=1e-7)
        assert result.contact[i].rigid_force[2] == pytest.approx(-force,abs=1e-7)
        assert mass*(new_v-velocity) == pytest.approx(.5*h*(previous_force+force),abs=1e-11)
        gap,velocity,acceleration,previous_force = new_gap,new_v,new_a,force
    assert min(result.swept_minimum_gaps) >= -.002
    assert any(c.active_line_cells for c in result.contact)
    assert result.contact[-1].active_line_cells == 0
    assert active_contact_manager() is None


@pytest.mark.parametrize('quadratic',[False,True])
def test_classical_beam_active_restart_and_definition_bound_controls(quadratic):
    _,direct = _impact(quadratic)
    _,prefix = _impact(quadratic,end=.0102,emit=True)
    assert prefix.contact[-1].active_line_cells
    _,resumed = _impact(quadratic,checkpoint=prefix.restart_checkpoint)
    for field in ('times','displacements','velocities','accelerations','swept_minimum_gaps'):
        np.testing.assert_allclose(np.concatenate((getattr(prefix,field),getattr(resumed,field)[1:])),
                                   getattr(direct,field),rtol=0.,atol=1e-9,err_msg=field)
    model,definition,settings = _model(quadratic=quadratic)
    model.add_boundary_condition(BoundaryCondition('normal_guide',list(model.mesh.nodes),
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes:
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.5
    with pytest.raises(ValueError,match='checkpoint analysis is incompatible'):
        solve_transient_contact(model,definition,TransientConfig(.0001,.013,initial_velocity=initial),
            contact_settings=settings,solver_controls=replace(CONTROLS,beam_swept_gap_tolerance_m=2e-10),
            restart_checkpoint=prefix.restart_checkpoint)


def test_beam_impact_requires_explicit_opt_in_and_rejects_caps():
    model,definition,settings = _model()
    with pytest.raises(ValueError,match='one shell surface'):
        solve_transient_contact(model,definition,TransientConfig(.001,.002),contact_settings=settings)
    model,definition,settings = _model(end_caps='hemisphere')
    with pytest.raises(ValueError,match='without end caps'):
        solve_transient_contact(model,definition,TransientConfig(.001,.002),contact_settings=settings,
                                solver_controls=CONTROLS)
    with pytest.raises(ValueError,match='separate profiles'):
        replace(CONTROLS,allow_s3_descriptor_dev=True)


def test_classical_beam_zero_physical_mass_is_rejected():
    model,definition,settings = _model()
    model.materials['elastic'].density = 0.
    with pytest.raises(ValueError,match='mass is not positive definite'):
        solve_transient_contact(model,definition,TransientConfig(.0001,.002,
            initial_velocity=np.zeros(model.mesh.dof_manager.total_dofs)),
            contact_settings=settings,solver_controls=CONTROLS)


def test_classical_beam_history_work_and_active_restart():
    history = ContactNodalLoadHistory(((0.,0.),(.004,1.),(.013,1.)))
    model,direct = _impact(True,history=history)
    assert direct.profile.endswith('NEWMARK_HISTORY_DEV_V1')
    _,prefix = _impact(True,end=.0102,emit=True,history=history)
    _,resumed = _impact(True,checkpoint=prefix.restart_checkpoint,history=history)
    np.testing.assert_allclose(np.concatenate((prefix.external_nodal_work_j,resumed.external_nodal_work_j[1:])),
                               direct.external_nodal_work_j,rtol=0.,atol=1e-12)
    # Piecewise affine force on quadratic Newmark/Hermite rigid motion:
    # a Simpson rule on each exact cubic integrand is independent here.
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    work = 0.
    for i,h in enumerate(np.diff(direct.times),1):
        x0,x1 = direct.displacements[i-1,dof],direct.displacements[i,dof]
        v0,v1 = direct.velocities[i-1,dof],direct.velocities[i,dof]
        f0,f1 = -.0001*history.factor(direct.times[i-1]),-.0001*history.factor(direct.times[i])
        derivative_mid = 1.5*(x1-x0)-.25*h*(v0+v1)
        work += (f0*h*v0+4*.5*(f0+f1)*derivative_mid+f1*h*v1)/6
    assert direct.external_nodal_work_j[-1] == pytest.approx(work,abs=1e-12)


def test_history_knot_rounding_never_advances_beyond_requested_end():
    end = np.nextafter(.004,0.)
    history = ContactNodalLoadHistory(((0.,0.),(.004,1.),(.013,1.)))
    _,result = _impact(True,end=end,history=history)
    assert result.times[-1] <= end


def test_deformable_b3_partial_impact_retains_rotary_motion_and_normal_momentum():
    model,definition,settings = _model(quadratic=True)
    nodes = list(model.mesh.nodes)
    model.add_boundary_condition(BoundaryCondition('planar_guide',nodes,
        {name:0. for name in ('ux','uy','rx','rz')}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    zdofs = [model.mesh.dof_manager.get_node_dofs(n)[2] for n in nodes]
    rdofs = [model.mesh.dof_manager.get_node_dofs(n)[4] for n in nodes]
    initial[zdofs] = [-.1,-.35,-.1]
    result = solve_transient_contact(model,definition,
        TransientConfig(.0001,.023,initial_velocity=initial),contact_settings=settings,
        solver_controls=CONTROLS)
    assert np.max(abs(result.velocities[:,rdofs])) > 1e-5
    assert any(0. < c.active_line_length < .99 for c in result.contact)
    assert np.max(np.ptp(result.displacements[:,zdofs],axis=1)) > 1e-5
    # Independent total normal momentum: exact integral of quadratic nodal
    # translation with rho*A*L=.001 and weights1/6,2/3,1/6. No normal guides.
    momentum = .001*result.velocities[:,zdofs] @ np.array([1/6,2/3,1/6])
    for i,h in enumerate(np.diff(result.times),1):
        impulse = .5*h*(result.contact[i-1].beam_force[2]+result.contact[i].beam_force[2])
        assert momentum[i]-momentum[i-1] == pytest.approx(impulse,abs=1e-10)
    reference = np.array([model.mesh.nodes[n].z for n in nodes])-.01
    for i,displacement in enumerate(result.displacements):
        heights = reference+displacement[zdofs]
        a,b,c = 2*(heights[0]-2*heights[1]+heights[2]),-3*heights[0]+4*heights[1]-heights[2],heights[0]
        values = [heights[0],heights[2]]
        if a > 0. and 0. < -b/(2*a) < 1.:
            values.append(c-b*b/(4*a))
        assert result.contact[i].minimum_gap == pytest.approx(min(values),abs=1e-12)


@pytest.mark.parametrize('cancel',[False,True])
def test_beam_path_failure_or_cancellation_preserves_commit_boundary(monkeypatch,cancel):
    import anyfesolver.general_contact.transient as driver
    import anyfesolver.general_contact.beam_temporal_bounds as geometry
    from anyfesolver.control import CancellationToken, SolveCancelled
    token = CancellationToken()
    commits,calls = [],[]
    commit,bounds = driver.commit_state_candidate,geometry.beam_hermite_gap_bounds
    def record_commit(*args,**kwargs):
        commits.append(1)
        return commit(*args,**kwargs)
    def stop(*args,**kwargs):
        calls.append(1)
        if len(calls) == 3:
            if cancel:
                token.cancel()
                kwargs['check_cancel']()
            raise RuntimeError('injected beam path failure')
        return bounds(*args,**kwargs)
    monkeypatch.setattr(driver,'commit_state_candidate',record_commit)
    monkeypatch.setattr(geometry,'beam_hermite_gap_bounds',stop)
    model,definition,settings = _model()
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes:
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.5
    with pytest.raises(SolveCancelled if cancel else RuntimeError):
        solve_transient_contact(model,definition,TransientConfig(.0001,.002,initial_velocity=initial),
            contact_settings=settings,solver_controls=CONTROLS,cancellation_token=token)
    assert len(commits) == 3
    assert active_contact_manager() is None
