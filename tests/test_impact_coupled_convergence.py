"""Public-route checks against a same-step two-mass Newmark oracle."""
import numpy as np
import pytest
import anyfesolver as fs
from anyfesolver.contact import _verification_contact_panel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._impact_coupling import ImpactCoupledIteration


def run_case(method="anderson", stiffness=40000., damping=0., deform=False, relaxation="aitken", end=.03, dt=.01, max_iterations=25, max_cutbacks=0, height=.205, token=None, callback=None, plastic=False, hht_alpha=0.):
    model = _verification_contact_panel()
    if deform:
        model.add_boundary_condition(BoundaryCondition("pin-one-z", [1], {"uz": 0.}))
    if plastic:
        from anyfesolver.material_curves import DNVC208MaterialCurve
        model.materials["soft"].hardening_curve = DNVC208MaterialCurve(
            sigma_prop=.008, sigma_yield=.01, sigma_yield_2=.012,
            eps_p_y1=1.e-5, eps_p_y2=1.e-3, K=.02, n=.1)
    return fs.solve_transient_sphere_impact(
        model, fs.TransientConfig(dt=dt, t_end=end, hht_alpha=hht_alpha),
        fs.RigidSphereImpact("coupled", radius=.2, mass=1., start_point=(.5,.5,height), travel_direction=(0,0,-1), speed=1.),
        fs.SphereContactConfig(penalty_stiffness=stiffness, contact_damping=damping, contact_relaxation=relaxation),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True,max_iterations=max_iterations,max_cutbacks=max_cutbacks,
            coupled_contact_acceleration=method), cancellation_token=token, status_callback=callback)


def two_mass_oracle(stiffness, damping, steps=3):
    # Each mass is exactly 1: sphere and area*density*thickness=1*20*.05.
    x=np.array([.205,0.]); v=np.array([-1.,0.]); a=np.zeros(2)
    positions=[x.copy()]; velocities=[v.copy()]; forces=[0.]
    dt=.01; beta=.25; gamma=.5
    for _ in range(steps):
        xp=x+dt*v+dt*dt*(.5-beta)*a
        vp=v+dt*(1-gamma)*a
        force=max((stiffness*(.2-xp[0]+xp[1])-damping*(vp[0]-vp[1])) /
                  (1+2*(stiffness*beta*dt*dt+damping*gamma*dt)),0.)
        if .2-xp[0]+xp[1]-2*beta*dt*dt*force <= 0:
            force=0.
        a=np.array([force,-force]); x=xp+beta*dt*dt*a; v=vp+gamma*dt*a
        positions.append(x.copy()); velocities.append(v.copy()); forces.append(force)
    return np.asarray(positions),np.asarray(velocities),np.asarray(forces)


@pytest.mark.parametrize("damping", [0.,10.])
@pytest.mark.parametrize("stiffness", [40000.,400000.])
def test_stiff_contact_completes_against_same_discrete_equations(damping, stiffness):
    result=run_case(damping=damping, stiffness=stiffness)
    assert result.status == "completed"
    x,v,f=two_mass_oracle(stiffness,damping)
    np.testing.assert_allclose(result.sphere_positions[:,2],x[:,0],atol=2.e-8,rtol=0)
    np.testing.assert_allclose(result.sphere_velocities[:,2],v[:,0],atol=3.e-6,rtol=0)
    np.testing.assert_allclose(result.displacements[:,2::6],np.repeat(x[:,1,None],4,axis=1),atol=2.e-8,rtol=0)
    np.testing.assert_allclose(result.contact_force_history[:,2],f,atol=3.e-4,rtol=0)
    assert all(s['convergence_reason']=='standard' for s in result.diagnostics['contact_step_diagnostics'])


def test_original_failure_and_explicit_disable_are_preserved():
    legacy=run_case("none")
    disabled=run_case(relaxation="none")
    assert legacy.status == disabled.status == "nonlinear_iteration_failed"
    np.testing.assert_array_equal(legacy.displacements,disabled.displacements)
    assert disabled.diagnostics['coupled_contact_acceleration']['fallback_reason']=='contact_relaxation_disabled'


def test_deforming_panel_matches_converged_legacy():
    legacy=run_case("none",stiffness=500.,deform=True)
    candidate=run_case(stiffness=500.,deform=True)
    assert legacy.status == candidate.status == "completed"
    assert np.ptp(candidate.displacements[-1,2::6])>1.e-5
    np.testing.assert_allclose(candidate.displacements,legacy.displacements,atol=2.e-8,rtol=1.e-5)
    np.testing.assert_allclose(candidate.sphere_positions,legacy.sphere_positions,atol=2.e-8,rtol=1.e-5)


def test_accelerator_affine_mode_rank_deficiency_and_detached_history():
    accelerator=ImpactCoupledIteration()
    x=np.array([0.,0.]); mapped=np.array([1.,1.]); accelerator.update(x,mapped)
    x[:]=99.; mapped[:]=99.
    result=accelerator.update(np.ones(2),-np.ones(2))
    np.testing.assert_allclose(result,np.full(2,1/3),atol=1.e-14)
    assert np.all(np.isfinite(accelerator.update(result,result)))


def test_configuration_rejects_unknown_method():
    with pytest.raises(ValueError,match='coupled_contact_acceleration'):
        fs.NonlinearTransientConfig(coupled_contact_acceleration='unknown')



def test_optin_no_contact_preserves_free_flight():
    legacy=run_case("none",height=1.)
    candidate=run_case(height=1.)
    np.testing.assert_array_equal(candidate.displacements,legacy.displacements)
    np.testing.assert_allclose(candidate.sphere_positions,legacy.sphere_positions,atol=1.e-15,rtol=0)
    np.testing.assert_array_equal(candidate.contact_force_history,legacy.contact_force_history)


def test_optin_cancellation_propagates_and_fresh_solve_succeeds():
    token=fs.CancellationToken()
    seen=[]
    def cancel_on_second_iteration(message):
        seen.append(message)
        if 'Iteration 2:' in message:
            token.cancel('coupled iteration cancelled')
    with pytest.raises(fs.SolveCancelled,match='coupled iteration cancelled'):
        run_case(token=token,callback=cancel_on_second_iteration)
    assert len(seen)==2
    # A fresh model solves successfully after cancellation propagates.
    assert run_case(end=.01).status=='completed'


def test_optin_cutback_preserves_elastic_committed_kinematics():
    cutback=run_case(end=.01,max_iterations=4,max_cutbacks=4)
    refined=run_case(end=.01,dt=.005)
    assert cutback.status==refined.status=='completed'
    assert cutback.diagnostics['cutback_count']>0
    np.testing.assert_allclose(cutback.displacements[-1],refined.displacements[-1],atol=2.e-8,rtol=0)
    np.testing.assert_allclose(cutback.sphere_positions[-1],refined.sphere_positions[-1],atol=2.e-8,rtol=0)
    assert cutback.diagnostics['strain_summary']['max_equivalent_plastic_strain']==0.


def test_optin_genuine_yielding_matches_converged_legacy():
    legacy=run_case('none',stiffness=500.,deform=True,plastic=True)
    candidate=run_case(stiffness=500.,deform=True,plastic=True)
    assert legacy.status==candidate.status=='completed'
    assert candidate.diagnostics['strain_summary']['max_equivalent_plastic_strain']>0.
    np.testing.assert_allclose(candidate.displacements,legacy.displacements,atol=2.e-8,rtol=1.e-5)
    np.testing.assert_allclose(candidate.contact_force_history,legacy.contact_force_history,atol=3.e-4,rtol=1.e-5)
    assert candidate.diagnostics['strain_summary']['max_equivalent_plastic_strain']==pytest.approx(
        legacy.diagnostics['strain_summary']['max_equivalent_plastic_strain'],rel=1.e-4,abs=1.e-10)


def test_nonfinite_extrapolation_falls_back_to_finite_raw_map():
    accelerator=ImpactCoupledIteration()
    accelerator.update(np.array([-1.e308]),np.array([-1.e308]))
    result=accelerator.update(np.array([0.]),np.array([1.e308]))
    np.testing.assert_array_equal(result,np.array([1.e308]))



def test_scalar_bracket_enters_hard_active_branch_without_changing_root():
    iteration=ImpactCoupledIteration()
    x=np.array([0.])
    for _ in range(24):
        mapped=np.maximum(.05-20*x,0.)
        signature=('active',) if mapped[0]>0 else ()
        new=iteration.update(x,mapped,signature=signature)
        if np.linalg.norm(mapped-x)<1.e-10:
            break
        x=new
    np.testing.assert_allclose(x,[.05/21],atol=1.e-10,rtol=0)
    assert iteration.globalization_count>0


def test_scalar_bracket_is_invariant_to_rigid_coordinate_shift_and_scale():
    def solve(scale,shift):
        iteration=ImpactCoupledIteration(); x=np.array([shift])
        for _ in range(24):
            raw=(x-shift)/scale
            mapped=shift+scale*np.maximum(.05-20*raw,0.)
            x=iteration.update(x,mapped,signature=('active',) if mapped[0]>shift else ())
        return (x-shift)/scale
    np.testing.assert_allclose(solve(1.,0.),solve(100.,50.),atol=1.e-10,rtol=0)


def test_large_residual_cannot_hide_transverse_displacement():
    iteration=ImpactCoupledIteration()
    iteration._globalize(np.array([0.,0.]),np.array([1.e8,0.]),np.array([1.,0.]))
    proposed=np.array([3.,1.])
    actual=iteration._globalize(np.array([0.,1.]),np.array([-1.e8,0.]),proposed)
    np.testing.assert_array_equal(actual,proposed)
    assert not iteration.mode_bracket
    assert iteration.globalization_count==0


def test_bracket_handles_same_sign_zero_and_repeated_coordinates():
    iteration=ImpactCoupledIteration()
    for current,residual in [(0.,2.),(1.,1.)]:
        proposed=np.array([current+residual])
        np.testing.assert_array_equal(iteration._globalize(np.array([current]),np.array([residual]),proposed),proposed)
    assert iteration.globalization_count==0
    proposed=np.array([1.])
    np.testing.assert_array_equal(iteration._globalize(proposed,np.array([0.]),proposed),proposed)
    # Opposite residuals at an identical coordinate are not a usable bracket.
    repeated=ImpactCoupledIteration()
    repeated._globalize(np.array([1.]),np.array([1.]),np.array([2.]))
    np.testing.assert_array_equal(repeated._globalize(np.array([1.]),np.array([-1.]),np.array([0.])),np.array([0.]))
    assert not repeated.mode_bracket


def test_noncollinear_residual_preserves_general_anderson_proposal():
    iteration=ImpactCoupledIteration()
    iteration.update(np.zeros(2),np.array([1.,0.]))
    x=np.array([1.,0.]);mapped=np.array([1.,1.])
    dx=np.array([1.,0.]);df=np.array([-1.,1.]);f=mapped-x
    expected=mapped-(dx+df)*(np.dot(df,f)/np.dot(df,df))
    actual=iteration.update(x,mapped)
    np.testing.assert_allclose(actual,expected,atol=1.e-15,rtol=0)
    assert not iteration.mode_bracket


def test_transverse_proposal_drops_bracket_and_exact_root_stays_fixed():
    iteration=ImpactCoupledIteration()
    iteration._globalize(np.zeros(2),np.array([1.,0.]),np.array([1.,0.]))
    proposal=np.array([.5,1.])
    np.testing.assert_array_equal(iteration._globalize(np.array([1.,0.]),np.array([-1.,0.]),proposal),proposal)
    assert not iteration.mode_bracket
    root=np.array([.5,0.])
    np.testing.assert_array_equal(iteration._globalize(root,np.zeros(2),np.array([5.,0.])),root)


@pytest.mark.parametrize("stiffness",[40000.,400000.])
def test_numerical_contact_newton_completes_deforming_contact(stiffness):
    result=run_case('contact_newton',stiffness=stiffness,deform=True)
    assert result.status=='completed'
    assert result.diagnostics['cutback_count']==0
    assert max(result.diagnostics['iteration_counts'])<=5
    assert np.ptp(result.displacements[-1,2::6])>1.e-5
    assert all(s['convergence_reason']=='standard' for s in result.diagnostics['contact_step_diagnostics'])
    assert result.diagnostics['coupled_contact_acceleration']['contact_jacobian_force_evaluations']>0


def test_contact_newton_matches_same_step_deforming_anderson():
    prior=run_case('anderson',stiffness=40000.,deform=True)
    candidate=run_case('contact_newton',stiffness=40000.,deform=True)
    np.testing.assert_allclose(candidate.displacements,prior.displacements,atol=2.e-8,rtol=1.e-5)
    np.testing.assert_allclose(candidate.sphere_positions,prior.sphere_positions,atol=2.e-8,rtol=1.e-5)
    np.testing.assert_allclose(candidate.contact_force_history,prior.contact_force_history,atol=3.e-4,rtol=1.e-5)


def test_contact_newton_hht_damping_and_yielding_preserve_equations():
    options=dict(stiffness=500.,damping=10.,deform=True,plastic=True,hht_alpha=-.1)
    prior=run_case('anderson',**options)
    candidate=run_case('contact_newton',**options)
    assert prior.status==candidate.status=='completed'
    assert candidate.diagnostics['strain_summary']['max_equivalent_plastic_strain']>0.
    for actual,reference in [(candidate.displacements,prior.displacements),
                             (candidate.sphere_positions,prior.sphere_positions)]:
        np.testing.assert_allclose(actual,reference,atol=2.e-8,rtol=1.e-5)
    np.testing.assert_allclose(candidate.contact_force_history,prior.contact_force_history,atol=3.e-4,rtol=1.e-5)


def test_numerical_contact_jacobian_preserves_nonsymmetric_derivative():
    from anyfesolver._impact_coupling import numerical_contact_jacobian
    matrix=np.array([[3.,2.],[-7.,5.]])
    coordinates=np.array([.03,.2])
    original=coordinates.copy()
    actual=numerical_contact_jacobian(lambda x:matrix@x,coordinates,.2)
    np.testing.assert_allclose(actual,matrix,atol=1.e-9,rtol=0)
    np.testing.assert_array_equal(coordinates,original)
    with pytest.raises(ValueError,match='nonfinite'):
        numerical_contact_jacobian(lambda x:np.full(2,np.nan),coordinates,.2)


def test_contact_jacobian_cancellation_propagates(monkeypatch):
    import anyfesolver.contact as contact
    original=contact.cancellation_safe_point
    token=fs.CancellationToken()
    def cancel_during_probe(token_arg,phase):
        if phase=='sphere_impact.nonlinear.contact_jacobian':
            token.cancel('cancel numerical contact probes')
        return original(token_arg,phase)
    monkeypatch.setattr(contact,'cancellation_safe_point',cancel_during_probe)
    with pytest.raises(fs.SolveCancelled,match='cancel numerical contact probes'):
        run_case('contact_newton',deform=True,token=token)


def test_contact_newton_honors_contact_relaxation_disable():
    result=run_case('contact_newton',relaxation='none')
    assert result.status=='nonlinear_iteration_failed'
    assert result.diagnostics['coupled_contact_acceleration']['method']=='none'
    assert result.diagnostics['coupled_contact_acceleration']['fallback_reason']=='contact_relaxation_disabled'



def test_numerical_contact_derivative_uses_physical_scale_under_origin_shift():
    from anyfesolver._impact_coupling import numerical_contact_jacobian
    for origin in [0.,1.e9]:
        point=np.array([origin+.03])
        local=point[0]-origin
        derivative=numerical_contact_jacobian(lambda x:(x-origin)**3,point,.2)
        # The expected derivative uses the actually representable local point.
        np.testing.assert_allclose(derivative,[[3.*local**2]],atol=1.e-10,rtol=0)


def test_contact_newton_large_translation_system_falls_back_to_anderson():
    from anyfesolver.fe_core import FEModel
    from anyfesolver.elements import create_shell_element
    model=FEModel('contact-newton-dimension-cap')
    model.add_material('soft',1.e5,.3,density=20.)
    for j in range(5):
        for i in range(5):model.add_node(1+j*5+i,i/4,j/4,0.)
    for j in range(4):
        for i in range(4):
            n=1+j*5+i
            model.add_element(1+j*4+i,create_shell_element(1+j*4+i,[n,n+1,n+6,n+5],'soft',thickness=.05))
    model.add_boundary_condition(BoundaryCondition('fixed-rotations',list(range(1,26)),{'rx':0.,'ry':0.,'rz':0.}))
    result=fs.solve_transient_sphere_impact(model,fs.TransientConfig(dt=.01,t_end=.01),
        fs.RigidSphereImpact('away',radius=.2,mass=1.,start_point=(.5,.5,1.),travel_direction=(0,0,-1),speed=1.),
        fs.SphereContactConfig(penalty_stiffness=40000.),
        nonlinear_config=fs.NonlinearTransientConfig(enabled=True,coupled_contact_acceleration='contact_newton'))
    diagnostics=result.diagnostics['coupled_contact_acceleration']
    assert result.status in {'completed','no_contact'}
    assert diagnostics['requested']=='contact_newton'
    assert diagnostics['method']=='anderson'
    assert diagnostics['fallback_reason']=='coupled_dimension_exceeds_64'
    assert diagnostics['contact_jacobian_force_evaluations']==0


def test_contact_newton_cutback_preserves_elastic_committed_kinematics():
    options=dict(method='contact_newton',deform=True,end=.01)
    cutback=run_case(max_iterations=3,max_cutbacks=4,**options)
    refined=run_case(dt=.005,**options)
    assert cutback.status==refined.status=='completed'
    assert cutback.diagnostics['cutback_count']>0
    np.testing.assert_allclose(cutback.displacements[-1],refined.displacements[-1],atol=2.e-8,rtol=0)
    np.testing.assert_allclose(cutback.sphere_positions[-1],refined.sphere_positions[-1],atol=2.e-8,rtol=0)
