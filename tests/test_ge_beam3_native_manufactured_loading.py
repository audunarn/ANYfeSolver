"""Prepared independent native force/torque loading DEVELOPMENT checks.

UNEXECUTED: requires separate manufactured-loading execution authority.
No existing finite-angle, temporal or native acceptance evidence is regraded.
"""
import json
import os
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.ge_beam3_native_dynamics import solve_transient
from examples.native_manufactured_dynamic_loading_dev import build, run
from examples.native_manufactured_loading_reference import (LoadingInputs, endpoint,
    interval, stage, scales, cross, dot)

ANALYTIC_RELATIVE_LIMIT = 1.e-3  # <=0.1%, frozen prospectively.
BALANCE_RELATIVE_LIMIT = 1.e-6
PHYSICAL = tuple(6*n+a for n in range(3) for a in range(3))+tuple(range(18,24))


def directory(tmp_path):
    path = Path(os.environ.get('NATIVE_MANUFACTURED_LOADING_RESULT', str(tmp_path)))
    path.mkdir(parents=True, exist_ok=True)
    return path


def peak(error, actual, reference, scale):
    error.append(float(np.linalg.norm(np.asarray(actual)-np.asarray(reference)))/scale)


def save(path, name, value):
    (path/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def audit(inputs, data, observed):
    sc = scales(inputs)
    assert all(value > 0 for value in sc.values())
    errors = {name:[] for name in ('position','cell_rotation','nodal_frame','velocity','omega',
        'acceleration','alpha','kinetic','elastic','mechanical_total',
        'derived_conservative_potential','derived_conservative_total','derived_conservative_balance','momentum','angular_momentum','resultant_force',
        'resultant_moment','endpoint_force','endpoint_moment','line_work','stage_position',
        'stage_rotation','stage_nodal_frame','stage_velocity','stage_omega','stage_acceleration','stage_alpha',
        'stage_force','stage_moment','force_impulse','moment_impulse','work','energy_balance',
        'linear_balance','angular_balance','physical_increment_translation','physical_increment_spin')}
    initial = data['initial']
    first = endpoint(inputs, 0.)
    for row in [initial]+[step['endpoint'] for step in data['steps']]:
        expected = endpoint(inputs, float(row['time']))
        m = row['state']['mechanical']; account = row['accounting']; load = row['physical_load']
        pos = np.asarray(m['positions'])+np.asarray(m['position_low'])
        peak(errors['position'],pos,expected['positions'],sc['length_m'])
        peak(errors['cell_rotation'],m['cell_rotations'][0],[expected['rotation']]*2,sc['angle_rad'])
        peak(errors['nodal_frame'],m['nodal_frames'],[expected['rotation']]*3,sc['angle_rad'])
        v = np.asarray(row['state']['physical_velocity24'])[list(PHYSICAL)]
        a = np.asarray(row['state']['physical_acceleration24'])[list(PHYSICAL)]
        peak(errors['velocity'],v[:9],expected['translation_velocity']*3,sc['velocity_m_s'])
        peak(errors['omega'],v[9:],expected['angular_velocity']*2,sc['omega_rad_s'])
        peak(errors['acceleration'],a[:9],expected['translation_acceleration']*3,sc['acceleration_m_s2'])
        peak(errors['alpha'],a[9:],expected['angular_acceleration']*2,sc['alpha_rad_s2'])
        peak(errors['kinetic'],account['kinetic_energy'],expected['kinetic_energy'],sc['energy_j'])
        peak(errors['elastic'],account['elastic_energy'],0.,sc['energy_j'])
        peak(errors['mechanical_total'],account['total_energy'],expected['kinetic_energy'],sc['energy_j'])
        momentum=np.asarray(account['linear_momentum'])+np.asarray(account['linear_momentum_low'])
        angular=np.asarray(account['angular_momentum'])+np.asarray(account['angular_momentum_low'])
        peak(errors['momentum'],momentum,expected['momentum'],sc['momentum_ns'])
        peak(errors['angular_momentum'],angular,expected['angular_momentum'],sc['angular_momentum_nms'])
        resultants=np.asarray(m['resultants'][0]).reshape(3,6)
        peak(errors['resultant_force'],resultants[:,:3],np.zeros((3,3)),sc['force_n'])
        peak(errors['resultant_moment'],resultants[:,3:],np.zeros((3,3)),sc['moment_nm'])
        force = tuple(inputs.length*f for f in inputs.force_per_length)
        moment = np.asarray(cross(expected['shift'], force))+np.asarray(inputs.couple_per_length)*inputs.length
        peak(errors['endpoint_force'],load['force'],force,sc['force_n'])
        peak(errors['endpoint_moment'],load['moment_about_origin'],moment,sc['moment_nm'])
        expected_line_work=dot(force,expected['shift'])
        peak(errors['line_work'],load['line_force_work'],expected_line_work,sc['energy_j'])
        # Raw mixed_potential is strain potential, NOT conservative_value.
        # Explicitly DERIVED conservative energies subtract verified line work.
        peak(errors['derived_conservative_potential'],account['elastic_energy']-load['line_force_work'],
             -expected_line_work,sc['energy_j'])
        peak(errors['derived_conservative_total'],account['total_energy']-load['line_force_work'],
             expected['kinetic_energy']-expected_line_work,sc['energy_j'])
        assert row['state']['histories'] == initial['state']['histories']
        assert load['has_nonpotential_couple'] is True
        assert load['momentum_origin_high'] == [0.,0.,0.] and load['momentum_origin_low'] == [0.,0.,0.]
    assert len(observed) == len(data['steps'])
    previous = initial
    for step,actual in zip(data['steps'],observed):
        begin,end = float(step['start_time']),float(step['end_time'])
        refstage = stage(inputs,begin,end); ref = interval(inputs,begin,end)
        expected_pos = np.array(endpoint(inputs,begin)['positions'])+np.asarray(refstage['shift'])-np.asarray(endpoint(inputs,begin)['shift'])
        peak(errors['stage_position'],actual['positions'],expected_pos,sc['length_m'])
        peak(errors['stage_rotation'],actual['cells'],[refstage['rotation']]*2,sc['angle_rad'])
        peak(errors['stage_nodal_frame'],actual['nodal_frames'],[refstage['rotation']]*3,sc['angle_rad'])
        peak(errors['stage_velocity'],actual['velocity'][:9],refstage['velocity']*3,sc['velocity_m_s'])
        peak(errors['stage_omega'],actual['velocity'][9:],refstage['omega']*2,sc['omega_rad_s'])
        exact = endpoint(inputs,end)
        peak(errors['stage_acceleration'],actual['acceleration'][:9],exact['translation_acceleration']*3,sc['acceleration_m_s2'])
        peak(errors['stage_alpha'],actual['acceleration'][9:],exact['angular_acceleration']*2,sc['alpha_rad_s2'])
        # Straight zero-lift TWO LINEAR HALVES: weights L/4,L/2,L/4, not a global P2 mass.
        external = np.asarray(step['midpoint_external_load42'])
        nodal = np.asarray([external[6*n:6*n+3] for n in range(3)])
        expect_nodal = np.asarray(inputs.force_per_length)[None,:]*np.asarray((.25,.5,.25))[:,None]*inputs.length
        peak(errors['stage_force'],nodal,expect_nodal,sc['force_n'])
        peak(errors['stage_moment'],np.asarray(external[18:24]).reshape(2,3),
             np.tile(np.asarray(inputs.couple_per_length)*inputs.length/2,(2,1)),sc['moment_nm'])
        assert np.count_nonzero(np.delete(external,list(PHYSICAL))) == 0
        peak(errors['force_impulse'],step['midpoint_force_impulse_ns'],ref['force_impulse'],sc['momentum_ns'])
        peak(errors['moment_impulse'],step['midpoint_moment_impulse_nms'],ref['moment_impulse'],sc['angular_momentum_nms'])
        peak(errors['work'],step['midpoint_load_work_j'],ref['work'],sc['energy_j'])
        increment = np.asarray(step['midpoint_physical_increment42'])[list(PHYSICAL)]
        startref = endpoint(inputs,begin)
        ds = tuple(b-a for a,b in zip(startref['shift'],exact['shift']))
        dphi = exact['angle']-startref['angle']
        peak(errors['physical_increment_translation'],increment[:9],ds*3,sc['length_m'])
        peak(errors['physical_increment_spin'],increment[9:],(dphi,0.,0.)*2,sc['angle_rad'])
        current=step['endpoint']['accounting']; prior=previous['accounting']
        peak(errors['energy_balance'],current['total_energy']-prior['total_energy'],ref['work'],sc['energy_j'])
        actual_line_delta=step['endpoint']['physical_load']['line_force_work']-previous['physical_load']['line_force_work']
        force=tuple(inputs.length*f for f in inputs.force_per_length)
        expected_line_delta=dot(force,ds)
        peak(errors['derived_conservative_balance'],
             current['total_energy']-prior['total_energy']-actual_line_delta,
             ref['work']-expected_line_delta,sc['energy_j'])
        p1=np.asarray(current['linear_momentum'])+np.asarray(current['linear_momentum_low'])
        p0=np.asarray(prior['linear_momentum'])+np.asarray(prior['linear_momentum_low'])
        h1=np.asarray(current['angular_momentum'])+np.asarray(current['angular_momentum_low'])
        h0=np.asarray(prior['angular_momentum'])+np.asarray(prior['angular_momentum_low'])
        peak(errors['linear_balance'],p1-p0,ref['force_impulse'],sc['momentum_ns'])
        peak(errors['angular_balance'],h1-h0,ref['moment_impulse'],sc['angular_momentum_nms'])
        support=step['midpoint_support']
        assert not support['physical_fixed_indices'] and not support['algebraic_fixed_indices']
        assert not np.any(support['physical_reaction42']) and not np.any(support['algebraic_reaction42'])
        assert step['stage_residual_norm'] <= 1.e-11 and step['endpoint_algebraic_norm'] <= 1.e-11
        previous=step['endpoint']
    return dict(inputs=asdict(inputs),scales=sc,max_errors={key:max(value,default=0.) for key,value in errors.items()},
        analytic_relative_limit=ANALYTIC_RELATIVE_LIMIT,balance_relative_limit=BALANCE_RELATIVE_LIMIT,
        expected_angle_change=endpoint(inputs,float(data['accepted_time']))['angle']-inputs.initial_angle,
        reference='independent primitive ellipse/integrals; actual discrete midpoint pose; globalzero moment origin',
        energy_semantics=dict(raw_elastic_energy='mixed_potential=response.potential (strain potential)',
            raw_total_energy='kinetic+strain potential',
            independently_checked_line_force_work='F dot shift',
            derived_conservative_potential='raw elastic minus line-force work',
            derived_conservative_total='raw total minus line-force work'),
        engineering_accepted=False)


@pytest.fixture(scope='module')
def manufactured_bundle(tmp_path_factory):
    # Reuse accepted DATA only, never a live state/history as a new solve origin.
    # Prefix/resume below creates fresh owners; full trajectory is solved once.
    inputs=LoadingInputs();owner,config=build(inputs);observed=[]
    original=type(owner).candidate_stage_audit
    def capture(self,candidate,**kwargs):
        result=original(self,candidate,**kwargs);s=result.stage;m=s.mechanical
        observed.append(dict(positions=(m.positions+m.position_low).tolist(),cells=m.cell_rotations[0].tolist(),
            nodal_frames=m.nodal_frames.tolist(),velocity=s.physical_velocity24[list(PHYSICAL)].tolist(),
            acceleration=s.physical_acceleration24[list(PHYSICAL)].tolist()))
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(owner),'candidate_stage_audit',capture)
        result=solve_transient(owner,config)
    data=result.to_dict();path=directory(tmp_path_factory.mktemp('manufactured_loading'))
    save(path,'manufactured_actual.json',dict(result=data,observed_stages=observed))
    (path/'manufactured.checkpoint').write_bytes(result.checkpoint)
    return inputs,result,data,observed,path


def test_public_manufactured_force_torque_finite_rotation(manufactured_bundle):
    inputs,result,data,observed,path=manufactured_bundle
    report=audit(inputs,data,observed);save(path,'manufactured_reference_report.json',report)
    assert result.success and data['accepted_steps'] == inputs.steps
    assert report['expected_angle_change'] >= .2
    for key,error in report['max_errors'].items():
        limit=BALANCE_RELATIVE_LIMIT if key in ('linear_balance','angular_balance','energy_balance',
            'derived_conservative_balance','endpoint_force','endpoint_moment','stage_force','stage_moment','force_impulse','moment_impulse','work') else ANALYTIC_RELATIVE_LIMIT
        assert error <= limit,(key,error,limit)


def test_public_manufactured_force_torque_exact_split_restart(manufactured_bundle):
    inputs,full,_,_,path=manufactured_bundle
    _,prefix=run(inputs,end_time=2*inputs.dt)
    _,resumed=run(inputs,checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    save(path,'manufactured_restart.json',dict(full=full.to_dict(),prefix=prefix.to_dict(),resumed=resumed.to_dict()))
    (path/'manufactured_prefix.checkpoint').write_bytes(prefix.checkpoint)
    assert full.checkpoint == resumed.checkpoint
    assert full.to_dict() == resumed.to_dict()


@pytest.mark.parametrize('start,end',[(0.,.01),(.0025,.005),(.005,.01)])
def test_independent_reference_work_impulse_and_discrete_midpoint_identities(start,end):
    inputs=LoadingInputs();a=endpoint(inputs,start);b=endpoint(inputs,end);load=interval(inputs,start,end)
    assert abs(b['kinetic_energy']-a['kinetic_energy']-load['work']) <= 1.e-12
    assert np.linalg.norm(np.asarray(b['momentum'])-a['momentum']-load['force_impulse']) <= 1.e-12
    assert np.linalg.norm(np.asarray(b['angular_momentum'])-a['angular_momentum']-load['moment_impulse']) <= 1.e-12
    mid=endpoint(inputs,float((start+end)/2));discrete=stage(inputs,start,end);h=end-start
    assert np.linalg.norm(np.asarray(discrete['shift'])-mid['shift']-np.asarray(a['translation_acceleration'])*h*h/8) <= 1.e-15
    assert abs(discrete['angle']-mid['angle']-a['angular_acceleration'][0]*h*h/8) <= 1.e-14
