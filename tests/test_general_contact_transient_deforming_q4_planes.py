"""Independent affine shear/membrane and mortar work; bounded dynamic specimen."""
import json
import os
from pathlib import Path
import runpy
import numpy as np
import pytest
from dataclasses import replace
from anyfesolver import TransientConfig,solve_transient_contact
from anyfesolver.assembly import assemble_mass_matrix
from anyfesolver.general_contact.core import ContactSolverSettings
from anyfesolver.general_contact.multiple_shell_planes import MultipleShellPlaneManager

example=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_deforming_q4_planes_dev.py'))
H,RHO,E,NU=.05,1.,1.e5,.3
GAP,KN=1.e-5,1.e9
MASS=np.diag([RHO*H,RHO*H/12])
KS=(5/6)*E/(2*(1+NU))*H
Q4=E*H/(8*(1-NU**2))


def pressure_geometry(cells=4):
    # Integral N_i dx and integral N_i*x dx / support for uniform linear hats.
    dx=1/cells
    weights=np.full(cells+1,dx)
    weights[[0,-1]]*=.5
    weighted_x=np.linspace(-.5,.5,cells+1)
    weighted_x[0]+=dx/3
    weighted_x[-1]-=dx/3
    rows=np.vstack((np.column_stack((np.ones(cells+1),weighted_x)),
                    -np.column_stack((np.ones(cells+1),weighted_x))))
    return np.tile(weights,2),rows


def structural(q):
    return np.array([0.,KS*q[1]+4*Q4*q[1]**3]),np.diag([0.,KS+12*Q4*q[1]**2])


def contact(q,cells=4):
    weights,direction=pressure_geometry(cells)
    gaps=GAP+direction@q
    closed=np.minimum(gaps,0.)
    force=KN*direction.T@(weights*closed)
    tangent=KN*direction.T@((weights*(gaps<0.))[:,None]*direction)
    return force,tangent,.5*KN*np.dot(weights,closed**2)


def test_affine_deforming_shell_mass_work_and_tangent_match_independent_reduction():
    model,_,modes=example['build_model']()
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,MASS,rtol=2.e-13,atol=2.e-15)
    for q in (np.array([0.,0.]),np.array([3.e-4,.08]),np.array([-.007,-.13])):
        force=np.zeros(2);tangent=np.zeros((2,2))
        for element in model.mesh.elements.values():
            dofs=np.concatenate([model.mesh.dof_manager.get_node_dofs(n) for n in element.node_ids])
            local=modes[dofs]
            f,k,_=element.compute_nonlinear_response(model.mesh,model.materials['elastic'],local@q,None,3,True)
            force+=local.T@f;tangent+=local.T@k@local
        expected,kexpected=structural(q)
        np.testing.assert_allclose(force,expected,rtol=2.e-12,atol=2.e-11)
        np.testing.assert_allclose(tangent,kexpected,rtol=2.e-12,atol=2.e-10)
        # Independent derivative of strain energy, at a slope resolving the quartic.
        delta=np.array([.7,-.4]);epsilon=1.e-7
        def energy(z):return .5*KS*z[1]**2+Q4*z[1]**4
        difference=(energy(q+epsilon*delta)-energy(q-epsilon*delta))/(2*epsilon)
        assert abs(difference-expected@delta)<=2.e-7*max(abs(expected@delta),1.)


def test_opposing_physical_skin_work_and_pressure_resolution_before_impact():
    # Geometry alone rejects a one-element pressure resolution for this budget.
    for cells,penetration in ((1,2.e-5),(4,2.e-6)):
        _,d=pressure_geometry(cells)
        slope=GAP/np.max(d[:,1])
        np.testing.assert_allclose(.5*slope-GAP,penetration,rtol=2.e-15)
    assert 2.e-6<example['GAP_TOLERANCE']<2.e-5
    model,definition,modes=example['build_model']()
    manager=MultipleShellPlaneManager(model,definition,ContactSolverSettings())
    for q in (np.array([0.,0.]),np.array([1.6e-6,3.1e-5]),np.array([-2.e-6,-3.3e-5])):
        f,k,result=manager.evaluate(modes@q,tangent=True)
        force,tangent,storage=contact(q)
        np.testing.assert_allclose(modes.T@f,force,rtol=3.e-11,atol=2.e-10)
        np.testing.assert_allclose(modes.T@k@modes,tangent,rtol=3.e-12,atol=2.e-7)
        np.testing.assert_allclose(result.penalty_energy,storage,rtol=3.e-11,atol=2.e-15)
        assert result.minimum_gap==pytest.approx(GAP-abs(q[0])-.5*abs(q[1]),abs=3.e-18)
        # Smooth two-mode directional check; do not cross a pressure activation kink.
        if q[1]:
            delta=np.array([.7,-.4]);epsilon=1.e-10
            fp=manager.evaluate(modes@(q+epsilon*delta),tangent=False)[0]
            fm=manager.evaluate(modes@(q-epsilon*delta),tangent=False)[0]
            numerical=modes.T@(fp-fm)/(2*epsilon)
            assert np.linalg.norm(numerical-tangent@delta)/np.linalg.norm(tangent@delta)<1.e-5


def reference(times):
    # Bounded independent Newton solves a strictly convex two-mode discrete work.
    # No ANYfeSolver contact/material/geometry/equilibrium calls.
    q=np.zeros(2);v=np.array([.002,.05]);a=np.zeros(2)
    rows=[(q.copy(),v.copy(),a.copy())]
    for left,right in zip(times,times[1:]):
        h=right-left;beta=.25*h*h;predictor=q+h*v+beta*a
        acceleration=a.copy()
        def residual(value):
            z=predictor+beta*value
            fs,ks=structural(z);fc,kc,_=contact(z)
            return MASS@value+fs+fc,MASS+beta*(ks+kc)
        for _ in range(40):
            r,jacobian=residual(acceleration)
            if np.linalg.norm(r,np.inf)<1.e-12:break
            step=np.linalg.solve(jacobian,-r)
            for reduction in range(30):
                trial=acceleration+(2.**-reduction)*step
                rt,_=residual(trial)
                if np.linalg.norm(rt)<=np.linalg.norm(r) or np.linalg.norm(rt,np.inf)<1.e-12:
                    acceleration=trial;break
            else:raise AssertionError('independent Newton backtracking exhausted')
        else:raise AssertionError('independent Newton iteration limit')
        next_q=predictor+beta*acceleration
        v=v+.5*h*(a+acceleration);q,a=next_q,acceleration
        rows.append((q.copy(),v.copy(),a.copy()))
    return np.array(rows)


@pytest.fixture(scope='module')
def impact():
    model,result,modes=example['solve']()
    q,v,a=(example['modal'](value,modes) for value in
        (result.displacements,result.velocities,result.accelerations))
    record=dict(times_s=result.times.tolist(),q=q.tolist(),v=v.tolist(),a=a.tolist(),
        pair_results=[r.to_dict() for r in result.contact],pair_gaps=result.pair_swept_minimum_gaps,
        cutbacks=result.cutbacks,cutback_diagnostics=result.cutback_diagnostics,
        events_s=result.localized_contact_event_times,endpoint_requests=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations)
    (Path(os.environ['RUNNER_TEMP'])/'deforming_shell_impact.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return model,result,modes


def test_deforming_shell_impacts_opposing_planes_with_actual_structural_work(impact):
    _,result,modes=impact
    expected=reference(result.times)
    for values,index,tolerance in ((result.displacements,0,1.e-11),(result.velocities,1,1.e-7),
                                  (result.accelerations,2,2.e-3)):
        actual=example['modal'](values,modes)
        np.testing.assert_allclose(actual,expected[:,index],rtol=0.,atol=tolerance)
        np.testing.assert_allclose(values,actual@modes.T,rtol=0.,atol=tolerance)
    _,d=pressure_geometry();weights,_=pressure_geometry()
    for q,row in zip(expected[:,0],result.contact):
        gaps=GAP+d@q
        pressures=KN*np.maximum(-gaps,0.)
        pairs=dict(row.pairs)
        np.testing.assert_allclose(pairs['lower'].rigid_force[2],-np.dot(weights[:5],pressures[:5]),rtol=1.e-7,atol=1.e-7)
        np.testing.assert_allclose(pairs['upper'].rigid_force[2],np.dot(weights[5:],pressures[5:]),rtol=1.e-7,atol=1.e-7)
        np.testing.assert_allclose(row.penalty_energy,contact(q)[2],rtol=1.e-7,atol=1.e-13)
    for name in ('lower','upper'):
        assert any(dict(row.pairs)[name].active_pressure_dofs for row in result.contact)
    assert result.localized_contact_event_times
    assert max(abs(expected[:,0,1]))>2.e-5
    assert np.max(np.abs(result.constraint_reactions@modes))<1.e-7
    energy=.5*np.einsum('ij,jk,ik->i',expected[:,1],MASS,expected[:,1])+.5*KS*expected[:,0,1]**2+Q4*expected[:,0,1]**4
    energy+=np.array([r.penalty_energy for r in result.contact])
    assert np.max(np.abs(energy/energy[0]-1.))<.01
    for pair in result.pair_swept_minimum_gaps:
        assert all(gap>=-example['GAP_TOLERANCE'] for gap in pair.values())


def test_deforming_active_restart_continues_work_and_rejects_front_algorithm_change():
    # A nominal step after the first event; it is outside the touch tolerance band.
    _,prefix,modes=example['solve'](end=.00046302395080566463,emit=True)
    assert dict(prefix.contact[-1].pairs)['upper'].active_pressure_dofs
    assert prefix.restart_checkpoint['version']==10
    _,resumed,_=example['solve'](end=.00065,checkpoint=prefix.restart_checkpoint)
    times=np.r_[prefix.times[:-1],resumed.times]
    for name,index,tolerance in (('displacements',0,1.e-11),('velocities',1,1.e-7),
                                ('accelerations',2,2.e-3)):
        values=np.concatenate((getattr(prefix,name)[:-1],getattr(resumed,name)))
        expected=reference(times)[:,index]
        actual=example['modal'](values,modes)
        np.testing.assert_allclose(actual,expected,rtol=0.,atol=tolerance)
        np.testing.assert_allclose(values,actual@modes.T,rtol=0.,atol=tolerance)
    model,definition,modes=example['build_model']()
    with pytest.raises(ValueError,match='analysis'):
        solve_transient_contact(model,definition,
            TransientConfig(1.e-5,.00065,initial_velocity=modes@example['INITIAL_VELOCITY']),
            solver_controls=replace(example['CONTROLS'],batch_plane_event_fronts_dev=False),
            restart_checkpoint=prefix.restart_checkpoint)
