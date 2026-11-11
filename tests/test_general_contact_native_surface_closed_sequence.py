"""Closed-ending multi-event point development; no analysis admission."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
from anyfesolver.general_contact.native_surface_event_sequence import evaluate_closed_event_sequence
from anyfesolver.general_contact.friction_point import trial_coulomb_point
from test_general_contact_native_surface_s3_friction import _case


def _recontact(mu=.3):
    _,old,start,history=_case()
    source=NativeRetainedRoundStation(7,old.source.reference,-.5,.03,.5*np.pi)
    pair=NativeS3MaterialPointFriction(source,old.target,reference_weight_m2=.01,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=mu,tangential_stiffness_Pa_per_m=1000.)
    positions=start.positions.copy();positions[:,2]=-.065
    cells=start.cell_rotations.copy();cells[0]=Rotation.from_rotvec((-.8,0.,0.)).as_matrix()
    start=replace(start,positions=positions,cell_rotations=cells)
    pressure=1000*(.025-.03*np.cos(.8))
    history=trial_coulomb_point(history,material_frame=history.material_frame,
        relative_increment=(.0001,0.,0.),normal_pressure=pressure,
        friction_coefficient=mu,tangential_stiffness=1000.,partner_key=pair.partner_key).candidate_state
    history=replace(history,cumulative_dissipation_density=2.,cumulative_plastic_slip=.2)
    return pair,start,history,np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18)


@pytest.mark.parametrize('mu,branch',[(.3,'slip'),(10.,'stick')])
def test_closed_open_closed_scalar_reaction_and_history(mu,branch):
    pair,start,history,t,w,s=_recontact(mu)
    value=evaluate_closed_event_sequence(pair,start,history,t,w,s)
    theta=np.arccos(.025/.03)
    pressure=1000*(.025-.03*np.cos(.8))
    slip=-.03*(np.sin(.8)-np.sin(theta))
    traction=min(1000*abs(slip),mu*pressure)
    assert value.event_fraction==pytest.approx((theta+.8)/1.6,abs=2e-13)
    assert value.trial.branch==branch
    assert sum(value.residual[6*i+1] for i in range(3))==pytest.approx(-.01*traction,rel=1e-10)
    assert sum(value.residual[6*i+2] for i in range(3))==pytest.approx(-.01*pressure,rel=1e-10)
    assert value.trial.candidate_state.cumulative_dissipation_density==pytest.approx(
        2.+value.trial.dissipated_increment_density)
    assert value.trial.candidate_state.cumulative_plastic_slip>=.2
    assert history.cumulative_dissipation_density==2.
    assert history.partner_key==pair.partner_key


@pytest.mark.parametrize('mu',(.3,10.))
def test_closed_sequence_both_body_tangent_from_original_origin(mu):
    pair,start,history,t,w,s=_recontact(mu)
    value=evaluate_closed_event_sequence(pair,start,history,t,w,s)
    direction=np.random.default_rng(135).normal(size=60)
    for h in (1e-5,2e-6):
        values=[]
        for sign in (1,-1):
            trans=t+sign*h*np.array([direction[6*i:6*i+3] for i in range(3)])
            angular=np.array([(Rotation.from_rotvec(sign*h*direction[18+3*i:21+3*i])
                *Rotation.from_rotvec(w[i])).as_rotvec() for i in range(2)])
            values.append(evaluate_closed_event_sequence(pair,start,history,trans,angular,s+sign*h*direction[42:]))
        assert values[0].trial.branch==values[1].trial.branch==value.trial.branch
        fd=(values[0].residual-values[1].residual)/(2*h)
        assert np.linalg.norm(value.spatial_tangent()@direction-fd)/np.linalg.norm(fd)<=1e-5


def test_closed_sequence_rejects_persistence_and_mismatched_history():
    pair,start,history,t,w,s=_recontact()
    with pytest.raises(ValueError,match='alternating interior'):
        evaluate_closed_event_sequence(pair,start,history,t,np.zeros_like(w),s)
    with pytest.raises(ValueError,match='history disagrees'):
        evaluate_closed_event_sequence(pair,start,replace(history,partner_key=None,elastic_slip=(0.,0.,0.)),t,w,s)


def test_closed_sequence_independent_discrete_work_and_deforming_target():
    pair,start,history,t,w,s=_recontact()
    value=evaluate_closed_event_sequence(pair,start,history,t,w,s)
    theta=np.arccos(.025/.03)
    elastic=np.array(history.elastic_slip[:2]);old_y=-.03*np.sin(-.8)
    dissipation=0.;plastic_length=0.;plastic_vector=np.zeros(2)
    for i in range(1,pair.event_history_substeps):
        angle=-.8+(.8-theta)*i/pair.event_history_substeps
        y=-.03*np.sin(angle)
        trial=elastic+np.array((0.,y-old_y))
        cap=.3*1000*(.025-.03*np.cos(angle))
        norm=np.linalg.norm(trial)
        next_elastic=trial*min(1.,cap/(1000*norm))
        plastic=trial-next_elastic
        dissipation+=cap*np.linalg.norm(plastic)
        plastic_length+=np.linalg.norm(plastic);plastic_vector+=plastic
        elastic=next_elastic;old_y=y
    released=500*float(elastic@elastic)
    trial=np.array((0.,-.03*(np.sin(.8)-np.sin(theta))))
    cap=.3*1000*(.025-.03*np.cos(.8))
    elastic=trial*min(1.,cap/(1000*np.linalg.norm(trial)))
    plastic=trial-elastic;dissipation+=cap*np.linalg.norm(plastic)
    plastic_length+=np.linalg.norm(plastic);plastic_vector+=plastic
    assert value.trial.dissipated_increment_density==pytest.approx(dissipation,rel=1e-10)
    assert value.trial.released_storage_density==pytest.approx(released,rel=1e-10)
    assert value.trial.candidate_state.cumulative_plastic_slip==pytest.approx(.2+plastic_length,rel=1e-10)
    np.testing.assert_allclose(value.trial.plastic_slip_increment,(*plastic_vector,0.),rtol=1e-10,atol=1e-14)
    # Both target edges and its material frame change along the full path.
    s[6:9]=(.002,-.001,.001);s[12:15]=(-.001,.002,-.0007)
    value=evaluate_closed_event_sequence(pair,start,history,t,w,s)
    direction=np.random.default_rng(327).normal(size=60);h=2e-6
    responses=[]
    for sign in (1,-1):
        trans=t+sign*h*np.array([direction[6*i:6*i+3] for i in range(3)])
        angular=np.array([(Rotation.from_rotvec(sign*h*direction[18+3*i:21+3*i])
            *Rotation.from_rotvec(w[i])).as_rotvec() for i in range(2)])
        responses.append(evaluate_closed_event_sequence(pair,start,history,trans,angular,s+sign*h*direction[42:]))
    fd=(responses[0].residual-responses[1].residual)/(2*h)
    assert np.linalg.norm(value.spatial_tangent()@direction-fd)/np.linalg.norm(fd)<=1e-5


def test_closed_sequence_reports_actual_history_calls(monkeypatch):
    import anyfesolver.general_contact.native_surface_activation as activation
    import anyfesolver.general_contact.native_surface_event_sequence as sequence
    original=activation.trial_coulomb_point;calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    pair,start,history,t,w,s=_recontact()
    monkeypatch.setattr(activation,'trial_coulomb_point',counted)
    monkeypatch.setattr(sequence,'trial_coulomb_point',counted)
    value=evaluate_closed_event_sequence(pair,start,history,t,w,s)
    assert value.history_evaluations==len(calls)==pair.event_history_substeps+2
    assert len(value.event_fractions)==2
    assert value.event_fractions[-1]==value.event_fraction


def test_initially_open_three_events_preserve_completed_window_and_final_force():
    from scipy.optimize import brentq
    from anyfesolver.general_contact.friction_point import initial_friction_point
    _,old,start,_=_case()
    source=NativeRetainedRoundStation(7,old.source.reference,-.5,.03,0.)
    pair=NativeS3MaterialPointFriction(source,old.target,reference_weight_m2=.01,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.)
    positions=start.positions.copy();positions[:,2]=-.04+.03*.85*1.2+.00001
    cells=start.cell_rotations.copy();cells[0]=Rotation.from_rotvec((-1.2,0.,0.)).as_matrix()
    start=replace(start,positions=positions,cell_rotations=cells)
    history=initial_friction_point(pair.target.physical_pose(start.shell_displacement).material_frame)
    t=np.tile((0.,0.,-.03*.85*2.4),(3,1));w=np.array(((2.4,0.,0.),(0.,0.,0.)));s=np.zeros(18)
    value=evaluate_closed_event_sequence(pair,start,history,t,w,s)
    gap=lambda angle:.03*(np.sin(angle)-.85*angle)+.00001
    roots=[brentq(gap,a,b,xtol=1e-14) for a,b in ((-1.2,-.5),(-.5,.5),(.5,1.2))]
    assert value.event_fractions==pytest.approx(tuple((a+1.2)/2.4 for a in roots),abs=2e-13)
    assert value.history_evaluations==pair.event_history_substeps+3
    elastic=0.;previous=.03*np.cos(roots[0]);dissipation=0.
    for i in range(1,pair.event_history_substeps):
        angle=roots[0]+(roots[1]-roots[0])*i/pair.event_history_substeps
        y=.03*np.cos(angle);trial=elastic+y-previous;cap=-.3*1000*gap(angle)
        next_elastic=np.clip(trial,-cap/1000,cap/1000)
        dissipation+=cap*abs(trial-next_elastic);elastic=next_elastic;previous=y
    released=500*elastic**2
    slip=.03*(np.cos(1.2)-np.cos(roots[-1]));cap=-.3*1000*gap(1.2)
    elastic=np.clip(slip,-cap/1000,cap/1000);dissipation+=cap*abs(slip-elastic)
    assert value.trial.dissipated_increment_density==pytest.approx(dissipation,rel=1e-9)
    assert value.trial.released_storage_density==pytest.approx(released,rel=1e-9)
    assert sum(value.residual[6*i+1] for i in range(3))==pytest.approx(.01*1000*elastic,rel=1e-10)
    assert sum(value.residual[6*i+2] for i in range(3))==pytest.approx(.01*1000*gap(1.2),rel=1e-10)
