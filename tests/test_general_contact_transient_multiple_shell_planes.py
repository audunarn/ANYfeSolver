"""Independent coupled modal equations for simultaneous Q4 plane contacts."""
from copy import deepcopy
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pytest
from anyfesolver import (ContactNodalLoadHistory, ContactTransientSolverControls, LoadCase,
                       TransientConfig, solve_transient_contact)
from anyfesolver.assembly import assemble_mass_matrix
from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
from anyfesolver.general_contact.transient import _multiple_shell_plane_path_gaps

example=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_multiple_q4_planes_dev.py'))
D=np.array([[1.,0.],[0.,1.],[.5,.5]])
M=.05*D.T@D
B=np.array([.01,.008,.01])
KN=np.array([1.e5,2.e5,1.e5])


def reference(times,load=None):
    """Enumerate 8 affine pressure branches; no production arithmetic/calls."""
    if load is None:
        load=lambda time:np.zeros(2)
    q=np.zeros(2); v=np.array([-1.,-.8]); a=np.linalg.solve(M,load(times[0]))
    rows=[(q.copy(),v.copy(),a.copy())]
    for time,h in zip(times[1:],np.diff(times)):
        prediction=q+h*v+.25*h*h*a
        candidates=[]
        for selected in itertools.product((False,True),repeat=3):
            weight=KN*np.array(selected)
            stiffness=D.T@(weight[:,None]*D)
            force=D.T@(weight*B)
            next_a=np.linalg.solve(M+.25*h*h*stiffness,-stiffness@prediction-force+load(time))
            next_q=prediction+.25*h*h*next_a
            gaps=B+D@next_q
            if any((closed and gap>1.e-10) or (not closed and gap<-1.e-10)
                   for closed,gap in zip(selected,gaps)):
                continue
            residual=M@next_a+D.T@(KN*np.minimum(gaps,0.))-load(time)
            if np.linalg.norm(residual)>1.e-7:
                continue
            candidates.append((next_q,next_a))
        assert candidates, 'independent modal branch unresolved'
        next_q,next_a=candidates[0]
        v=v+.5*h*(a+next_a)
        q,a=next_q,next_a
        rows.append((q.copy(),v.copy(),a.copy()))
    return np.array(rows)


@pytest.fixture(scope='module')
def impact():
    model,result,modes=example['solve']()
    folder=Path(os.environ['RUNNER_TEMP'])
    q,v,a=(example['modal'](values,modes) for values in
           (result.displacements,result.velocities,result.accelerations))
    record=dict(times_s=result.times.tolist(),q=q.tolist(),v=v.tolist(),a=a.tolist(),
        cutbacks=result.cutbacks,cutback_diagnostics=result.cutback_diagnostics,
        events_s=result.localized_contact_event_times,endpoint_requests=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations,
        pair_gaps=result.pair_swept_minimum_gaps,
        pairs=[row.to_dict() for row in result.contact])
    (folder/'multiple_shell_impact.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return model,result,modes


def test_multiple_plane_impact_has_true_coupled_mass_and_pressure_work(impact):
    model,result,modes=impact
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,M,rtol=2.e-13,atol=1.e-13)
    assert M[0,1]>0.
    expected=reference(result.times)
    for values,index,tolerance in ((result.displacements,0,1.e-10),(result.velocities,1,1.e-7),
                                   (result.accelerations,2,1.e-4)):
        np.testing.assert_allclose(example['modal'](values,modes),expected[:,index],rtol=0.,atol=tolerance)
    for position,contact in zip(expected[:,0],result.contact):
        pressure=KN*np.maximum(-B-D@position,0.)
        pairs=dict(contact.pairs)
        np.testing.assert_allclose(pairs['first'].rigid_force[2],-pressure[[0,2]].sum(),atol=1.e-5)
        np.testing.assert_allclose(pairs['second'].rigid_force[2],-pressure[1],atol=1.e-5)
        np.testing.assert_allclose(contact.penalty_energy,.5*np.sum(KN*np.minimum(B+D@position,0.)**2),atol=1.e-8)
    assert any(all(child.active_pressure_dofs for _,child in row.pairs) for row in result.contact)
    assert result.localized_contact_event_times
    assert len(result.pair_swept_minimum_gaps)==len(result.times)
    for row in result.pair_swept_minimum_gaps:
        assert row['first']>=-.0015 and row['second']>=-.0009
    assert np.max(np.abs(result.constraint_reactions@modes))<1.e-6
    velocity=expected[:,1]
    energy=.5*np.einsum('ij,jk,ik->i',velocity,M,velocity)+np.array([row.penalty_energy for row in result.contact])
    assert np.max(np.abs(energy/energy[0]-1.))<.01


def test_multiple_plane_restart_retains_pair_features_and_rejects_rehashed_corruption(impact):
    _,whole,modes=impact
    index=next(i for i,row in enumerate(whole.contact) if all(c.active_pressure_dofs for _,c in row.pairs))
    # A saved accepted time can be event-limited; use the next nominal step.
    index=next(i for i in range(index+1,len(whole.times)) if
        abs((whole.times[i]-whole.times[i-1])-.0005)<1.e-12
        and all(c.active_pressure_dofs for _,c in whole.contact[i].pairs))
    _,prefix,_=example['solve'](end=float(whole.times[index]),emit=True)
    assert prefix.restart_checkpoint['version']==10
    assert len(prefix.restart_checkpoint['state']['plane_pressure_features'])==12
    _,resumed,_=example['solve'](checkpoint=prefix.restart_checkpoint)
    joined=np.concatenate((prefix.displacements[:-1],resumed.displacements))
    np.testing.assert_allclose(joined,whole.displacements,rtol=0.,atol=1.e-10)
    corrupted=deepcopy(prefix.restart_checkpoint)
    corrupted['state']['plane_pressure_features'][0][1]=True
    corrupted['checkpoint_sha256']=analysis_configuration_fingerprint({k:v for k,v in corrupted.items() if k!='checkpoint_sha256'})
    with pytest.raises(ValueError,match='pressure-feature'):
        example['solve'](checkpoint=corrupted)


def test_multiple_planes_require_explicit_event_profile():
    model,definition,modes=example['build_model']()
    with pytest.raises(ValueError,match='one shell surface'):
        solve_transient_contact(model,definition,TransientConfig(.001,.001,initial_velocity=modes@[-1.,-.8]))
    with pytest.raises(ValueError,match='separate elastic'):
        ContactTransientSolverControls(allow_multiple_shell_planes_dev=True)


def test_each_pair_complete_path_uses_its_own_dimensional_gap_gate():
    from types import SimpleNamespace
    def child(gap,tolerance):
        return SimpleNamespace(patches=((np.array([[0.,0.,gap]]),np.array([[0,1,2]]),None,
            np.zeros((1,3)),np.array([0.,0.,1.])),),point=np.zeros(3),normal=np.array([0.,0.,1.]),gap_tolerance=tolerance)
    # First pair tolerates .02; second's interior cubic gap dips below .001.
    manager=SimpleNamespace(pairs=(('first',child(.01,.02)),('second',child(.001,.001))))
    start=end=np.zeros(3)
    minimum,admissible,per_pair=_multiple_shell_plane_path_gaps(manager,start,end,
        np.array([0.,0.,-.02]),np.array([0.,0.,.02]),1.,lambda:None)
    assert per_pair['first']>0. and per_pair['second']<-.001
    assert per_pair['first']==pytest.approx(.005,abs=1.e-15)
    assert per_pair['second']==pytest.approx(-.004,abs=1.e-15)
    assert minimum==per_pair['second'] and not admissible


def test_multiple_plane_load_history_work_and_restart_use_actual_stage_time():
    history=ContactNodalLoadHistory(((0.,0.),(.025,1.),(.04,1.)))
    load=LoadCase('two-patch dead-load ramp')
    for node in range(1,9):
        load.add_nodal_load(node,forces=(0.,0.,-.0025 if node<5 else -.005))
    def run(end,checkpoint=None,emit=False):
        model,definition,modes=example['build_model']()
        result=solve_transient_contact(model,definition,
            TransientConfig(.0005,end,initial_velocity=modes@[-1.,-.8]),
            solver_controls=example['CONTROLS'],base_load_case=load,nodal_load_history=history,
            restart_checkpoint=checkpoint,emit_restart_checkpoint=emit)
        return result,modes
    whole,modes=run(.025)
    force=np.array([-.01,-.02])
    factor=lambda time: min(time/.025,1.)
    expected=reference(whole.times,lambda time:factor(time)*force)
    q=example['modal'](whole.displacements,modes)
    v=example['modal'](whole.velocities,modes)
    np.testing.assert_allclose(q,expected[:,0],rtol=0.,atol=1.e-10)
    np.testing.assert_allclose(v,expected[:,1],rtol=0.,atol=1.e-7)
    work=[0.]
    for i,h in enumerate(np.diff(whole.times)):
        f0,f1=factor(whole.times[i]),factor(whole.times[i+1])
        # Newmark's represented Hermite path has linear velocity. Integrate
        # its product with the linear load factor independently, exactly.
        added=.5*(f0+f1)*(force@(q[i+1]-q[i]))+(f1-f0)*h*(force@(v[i+1]-v[i]))/12.
        work.append(work[-1]+added)
    np.testing.assert_allclose(whole.external_nodal_work_j,work,rtol=1.e-9,atol=1.e-12)
    prefix,_=run(.003,emit=True)
    resumed,_=run(.025,checkpoint=prefix.restart_checkpoint)
    assert prefix.restart_checkpoint['version']==10
    np.testing.assert_allclose(resumed.displacements,whole.displacements[len(prefix.times)-1:],atol=1.e-10,rtol=0.)
    np.testing.assert_allclose(resumed.external_nodal_work_j,whole.external_nodal_work_j[len(prefix.times)-1:],atol=1.e-12,rtol=0.)
