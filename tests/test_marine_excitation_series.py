"""Native development integration; no physical source qualification."""
from dataclasses import replace
import json
import math
import numpy as np
import pytest
import importlib.util
if importlib.util.find_spec('anyhydro') is None:
    pytest.importorskip('anyhydro', reason='optional marine-excitation provider is not installed')
if importlib.util.find_spec('anyloads') is None:
    pytest.importorskip('anyloads', reason='optional marine load provider is not installed')
from anyhydro import ExcitationSeries, build_excitation_series, UNIT_INCIDENT_AMPLITUDE
from anyhydro.fixtures import make_two_body_fixture
from anyloads import (JonswapSpectrum, WaveRealization, DirectionalSpectrum,
    MorisonSection, HydrodynamicOwnership)
from anyfesolver.marine_potential_flow import (data_to_dict,data_from_dict,
    PotentialFlowDevelopmentContribution,MarinePotentialFlowError)
from anyfesolver.marine_morison import MemberEndpoint, MorisonMember, MorisonMemberSetContribution
from anyfesolver.marine_contributions import ContributionSet, StepInformation, MarineContributionsError
from anyfesolver.marine_mechanical import MarineMechanicalModel, MarineMechanicalRunSpec, solve_marine_mechanical
from test_marine_potential_flow import fixture,frame,body,states_for


def declared_series(data):
    n=len(data.dof_keys)
    real=np.zeros((2,n));imag=real.copy()
    real[:,0]=(.4,-.1);imag[:,0]=(.2,.3)
    return ExcitationSeries(data.dof_keys,[.7,1.1],real,imag,'a'*64,'b'*64,
        'development_fixture',UNIT_INCIDENT_AMPLITUDE,(0.,0.))


def test_materialized_series_native_discrete_recurrence():
    b=body();base=fixture(linear=True);s=declared_series(base)
    data=base.with_excitation_series(s,development_only=True)
    p=PotentialFlowDevelopmentContribution(data)
    result=solve_marine_mechanical(MarineMechanicalModel((b,),contributions=(p,)),
        {'a':b.state(position=(.2,0.,0.),velocity=(.3,0.,0.,0.,0.,0.))},MarineMechanicalRunSpec(.1,3))
    assert result.status=='completed',result.logs
    def wave(t):return .4*math.cos(.7*t)+.2*math.sin(.7*t)-.1*math.cos(1.1*t)+.3*math.sin(1.1*t)
    x,v,a=.2,.3,wave(0.)/6.;velocities=[v]
    for k,accepted in enumerate(result.accepted):
        if k:
            xp,vp=x+.1*v+.0025*a,v+.05*a
            length=min(k,2)
            memory=.1*sum((1. if j<length else .5)*(1.,.5)[j-1]*velocities[k-j] for j in range(1,length+1))
            a=(.6+wave(.1*k)-3.*xp-.1*vp-memory)/(6.+3.*.0025+.1*.05)
            x,v=xp+.0025*a,vp+.05*a;velocities.append(v)
        np.testing.assert_allclose((accepted.positions['a'][0],accepted.world_velocities['a'][0],
            accepted.world_accelerations['a'][0]),(x,v,a),rtol=1e-10,atol=1e-10)


def test_series_work_pullback_and_diagnostics():
    frames=(frame('a'),frame('b'));base=fixture(frames);s=declared_series(base)
    data=base.with_excitation_series(s,development_only=True);p=PotentialFlowDevelopmentContribution(data)
    states=states_for(frames,True);owner=ContributionSet(('a','b'),(p,))
    step=owner.prepare_step(StepInformation(0,.1,'initialization'))
    trial=owner.evaluate_trial(step,states);diag=trial.diagnostics[p.contribution_id]
    from anyfesolver.marine_hydro_coordinates import evaluate_coupled
    mapped=evaluate_coupled(frames,states)
    expected=np.zeros(12);expected[0]=.4-.1
    np.testing.assert_allclose(diag['components']['excitation'],mapped.jacobian.T@expected,rtol=1e-12,atol=1e-12)
    assert diag['wave_realization_hash']==s.wave_realization_hash
    assert diag['excitation_source_hash']==s.hydro_content_hash
    assert diag['excitation_series_hash']==s.content_hash


def test_legacy_and_series_data_roundtrip_no_alias():
    base=fixture();old=data_to_dict(base)
    assert 'excitation_series' not in old
    assert data_to_dict(data_from_dict(old))==old
    data=base.with_excitation_series(declared_series(base),development_only=True)
    saved=json.loads(json.dumps(data_to_dict(data)));restored=data_from_dict(saved)
    assert data_to_dict(restored)==saved
    saved['excitation_series']['weighted_real'][0][0]=99.
    assert restored.excitation_series.weighted_real[0,0]==.4
    with pytest.raises(ValueError):restored.excitation_series.weighted_real.setflags(write=True)


@pytest.mark.parametrize('case',['status','order','axes','legacy','unknown','boolean','authority','none'])
def test_native_series_refusals(case):
    base=fixture();s=declared_series(base)
    if case=='status':s=replace(s,source_qualification_status='unqualified')
    if case=='order':s=replace(s,dof_keys=tuple(reversed(s.dof_keys)))
    if case=='axes':base=fixture((frame('a',True),))
    if case=='unknown':s=s.to_dict();s['provider']='python'
    if case=='boolean':s=s.to_dict();s['weighted_real'][0][0]=True
    if case=='none':s=None
    with pytest.raises(MarinePotentialFlowError):
        if case=='legacy':replace(base,excitation_series=s)
        else:base.with_excitation_series(s,development_only=case!='authority')


def test_series_cancellation_preserves_accepted_memory():
    base=fixture(linear=True);data=base.with_excitation_series(declared_series(base),development_only=True)
    b=body();p=PotentialFlowDevelopmentContribution(data);flag={'cancel':False}
    owner=ContributionSet(('a',),(p,),cancel=lambda:flag['cancel'])
    states={'a':b.state()}
    step=owner.prepare_step(StepInformation(1,.1,'step',states));trial=owner.evaluate_trial(step,states)
    owner.commit(step,trial);accepted=owner.committed_history
    step=owner.prepare_step(StepInformation(2,.1,'step',states));trial=owner.evaluate_trial(step,states)
    flag['cancel']=True
    with pytest.raises(MarineContributionsError,match='cancellation'):owner.commit(step,trial)
    assert owner.committed_history is accepted


def shared_model():
    from anyfesolver.marine_hydro_coordinates import EquilibriumHydroFrame
    ds=make_two_body_fixture()
    frames=tuple(EquilibriumHydroFrame(str(b),tuple(ds.rotation_center.values[i]),
        np.eye(3),np.eye(3),(0.,)*3) for i,b in enumerate(ds.body.values))
    waves=WaveRealization.from_jonswap(JonswapSpectrum(.0001,8.),(.7,1.1),(.2,.2),
        seed=17,directions=DirectionalSpectrum((0.,np.pi/2),(.4,.6)))
    series=build_excitation_series(ds,waves,amplitude_normalization=UNIT_INCIDENT_AMPLITUDE,phase_origin_xy=(0.,0.))
    data=fixture(frames).with_excitation_series(series,development_only=True)
    bodies=tuple(body(f.body_id) for f in frames)
    member=MorisonMember('fixture-drag',MemberEndpoint(bodies[0].body_id,(0.,0.,-1.)),
        MemberEndpoint(bodies[0].body_id,(0.,0.,-2.)),MorisonSection(.03,1.,1.),
        HydrodynamicOwnership('potential_flow',True,True,'none'),quadrature_order=2)
    environment=waves.environment(20.)
    drag=MorisonMemberSetContribution('drag-fixture',bodies,(member,),environment)
    pf=PotentialFlowDevelopmentContribution(data)
    return MarineMechanicalModel(bodies,contributions=(pf,drag)),{
        b.body_id:b.state(position=f.equilibrium_position) for b,f in zip(bodies,frames)},waves,data


def test_public_common_wave_realization_native_route():
    model,states,waves,data=shared_model()
    assert model.contributions[1].environment.components==waves.components
    assert data.excitation_series.wave_realization_hash==waves.content_hash
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.1,3))
    assert result.status=='completed',result.logs
    assert len(result.accepted)==4
