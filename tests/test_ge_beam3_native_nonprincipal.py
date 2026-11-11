"""Retained executed4/8 NONPASS checks and prospective corrected16 checks."""
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from examples import native_nonprincipal_reference as ref
from examples.native_nonprincipal_dev import run,build,solve_transient
from anyfesolver._ge_beam3_p5.algebra import rotation as native_rotation
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum

PHYSICAL=[0,1,2,6,7,8,12,13,14,18,19,20,21,22,23]
ENERGY_SCALE=ref.account(*ref.initial()[:3])['total_energy']
MOMENTUM_SCALE=ref.RHO*ref.LENGTH**2*ref.SPIN_SCALE
ANGULAR_SCALE=(ref.RHO*ref.LENGTH**3/12+ref.LENGTH*max(ref.JDENSITY))*ref.SPIN_SCALE


def _save(tag,result,metrics):
    assert tag in {'work','coarse','fine','restart_full','restart_prefix','restart_resumed','corrective_prefix','corrective_full'}
    path=os.environ.get('NATIVE_TRANSIENT_RESULT_RECORD')
    if not path:return
    base=Path(path);base.parent.mkdir(parents=True,exist_ok=True)
    capsule=base.with_name(base.stem+'_'+tag+'.checkpoint');capsule.write_bytes(result.checkpoint)
    base.with_name(base.stem+'_'+tag+'.json').write_text(json.dumps(dict(result=result.to_dict(),metrics=metrics,
        checkpoint_sha256=sha256(result.checkpoint).hexdigest(),checkpoint_path=str(capsule)),
        sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def _unpack(record):
    state=record['state'];m=state['mechanical']
    x=np.asarray(m['positions'])+np.asarray(m['position_low']);u=np.asarray(m['cell_rotations']).reshape(2,3,3)
    v=np.asarray(state['physical_velocity24'])[PHYSICAL];a=np.asarray(state['physical_acceleration24'])[PHYSICAL]
    return x,u,v,a,np.asarray(m['nodal_frames']),np.asarray(m['resultants']).reshape(18)


def _errors(record,expected):
    x,u,v,a,frames,p=_unpack(record);work=ref.elastic(expected['x'],expected['u'])
    fields=work[2];wanted=np.r_[fields['forces'].ravel(),fields['moments'].ravel()]
    state=record['state'];algebraic=[3,4,5,9,10,11,15,16,17]
    return dict(algebraic_velocity=0. if not np.any(np.asarray(state['physical_velocity24'])[algebraic]) else 1.,
        algebraic_acceleration=0. if not np.any(np.asarray(state['physical_acceleration24'])[algebraic]) else 1.,position=float(np.max(np.abs(x-expected['x'])))/ref.LENGTH,
        frames=float(np.max(np.abs(u-expected['u']))),
        nodal_velocity=float(np.max(np.abs(v[:9]-expected['v'][:9])))/ref.TRANSLATION_VELOCITY_SCALE,
        cell_omega=float(np.max(np.abs(v[9:]-expected['v'][9:])))/ref.SPIN_SCALE,
        nodal_acceleration=float(np.max(np.abs(a[:9]-expected['a'][:9])))/(ref.LENGTH*ref.ACC_SCALE),
        cell_alpha=float(np.max(np.abs(a[9:]-expected['a'][9:])))/ref.ACC_SCALE,
        nodal_frames=float(np.max(np.abs(frames-fields['frames']))),
        retained_force=float(np.max(np.abs(p[:6]-wanted[:6])))/ref.FORCE_SCALE,
        retained_moment=float(np.max(np.abs(p[6:]-wanted[6:])))/ref.MOMENT_SCALE)


def _physics(record):
    x,u,v,a,frames,p=_unpack(record);energy,f,fields=ref.elastic(x,u)
    residual=ref.physical_mass(u)@a+ref.gyro(u,v)+f
    physical_force=residual[:9].reshape(3,3);physical_torque=residual[9:].reshape(2,3)
    accounted=ref.account(x,u,v);stored=record['accounting']
    return dict(force_balance=float(np.max(np.abs(physical_force)))/ref.FORCE_SCALE,
        moment_balance=float(np.max(np.abs(physical_torque)))/ref.MOMENT_SCALE,
        force_resultant=float(np.linalg.norm(np.sum(physical_force,axis=0)))/ref.FORCE_SCALE,
        moment_resultant=float(np.linalg.norm(np.sum(np.cross(x,physical_force),axis=0)+np.sum(physical_torque,axis=0)))/ref.MOMENT_SCALE,
        energy=float(abs(stored['total_energy']-accounted['total_energy']))/ENERGY_SCALE,
        elastic_energy=float(abs(stored['elastic_energy']-accounted['elastic_energy']))/ENERGY_SCALE,
        kinetic_energy=float(abs(stored['kinetic_energy']-accounted['kinetic_energy']))/ENERGY_SCALE,
        linear_momentum=float(np.max(np.abs(np.asarray(stored['linear_momentum'])+np.asarray(stored['linear_momentum_low'])-accounted['linear_momentum'])))/MOMENTUM_SCALE,
        angular_momentum=float(np.max(np.abs(np.asarray(stored['angular_momentum'])+np.asarray(stored['angular_momentum_low'])-accounted['angular_momentum'])))/ANGULAR_SCALE,
        algebraic_trace=float(np.max(np.abs(ref.full_trace_work(x,u,frames)[1])))/ref.MOMENT_SCALE)


def _native_stage_record(owner,mechanical,velocity,acceleration,inertia=None):
    context=owner._owner();cells=np.asarray(mechanical.cell_rotations).reshape(2,3,3)
    resultants=np.asarray(mechanical.resultants).reshape(18)
    work=context._dynamic.element.operator.evaluate(mechanical.positions,mechanical.position_low,
        mechanical.nodal_frames,cells,resultants)
    if inertia is None:inertia=context._inertia(mechanical,velocity,acceleration)
    fields={key:getattr(inertia,key) for key in ('linear_momentum','linear_momentum_low','angular_momentum','angular_momentum_low')}
    fields.update(kinetic_energy=inertia.kinetic_energy,elastic_energy=work.potential,
                  total_energy=inertia.kinetic_energy+work.potential)
    state=dict(mechanical=dict(positions=mechanical.positions,position_low=mechanical.position_low,
        nodal_frames=mechanical.nodal_frames,cell_rotations=cells,resultants=resultants),
        physical_velocity24=velocity,physical_acceleration24=acceleration)
    return json.loads(json.dumps(dict(state=state,accounting=fields),default=lambda value:value.tolist(),allow_nan=False))


def _run_captured(**kwargs):
    """Observe authentic issued audits; return every original audit unchanged."""
    steps=kwargs.pop('steps',ref.STEPS);dt=kwargs.pop('dt',ref.DT)
    owner,config=build(steps=steps,dt=dt);captures={};original=type(owner).candidate_stage_audit
    def observe(self,candidate,**audit_kwargs):
        audit=original(self,candidate,**audit_kwargs)
        if self is owner:
            stage=audit.stage
            record=_native_stage_record(owner,stage.mechanical,stage.physical_velocity24,
                                        stage.physical_acceleration24,stage.inertia)
            record.update(evidence_kind='CAPTURED_AUTHENTIC_NATIVE_STAGE_REEVALUATED_ACCOUNTING',
                stage_sha256=audit.stage_sha256,candidate_sha256=audit.candidate_sha256,
                predecessor_sha256=audit.predecessor_sha256,stage_material_sha256=stage.material_sha256,
                dt=audit.dt,endpoint_epoch=audit.endpoint_epoch,
                unknown42=audit.unknown42.tolist(),increment42=audit.increment42.tolist())
            captures[audit.endpoint_epoch]=record
        return audit
    with patch.object(type(owner),'candidate_stage_audit',observe):
        result=solve_transient(owner,config,**kwargs)
    return owner,result,captures


def _reconstructed_stage(owner,previous,row):
    """Reevaluate a native stage from the full saved mixed increment.

    This is reconstructed evidence, not a saved original stage. Physical alpha
    has a conservative binary64 rounding bound; algebraic data never uses ref.
    """
    old=previous['state'];m=old['mechanical'];h=row['dt']
    d=np.asarray(row['midpoint_physical_increment42']);v0=np.asarray(old['physical_velocity24'])
    velocity=np.zeros(24);acceleration=np.zeros(24)
    velocity[PHYSICAL]=d[PHYSICAL]/h
    acceleration[PHYSICAL]=2*(d[PHYSICAL]-h*v0[PHYSICAL])/h**2
    unit=2.**-53
    alpha_bound=4*unit*(np.abs(d[PHYSICAL])+np.abs(h*v0[PHYSICAL]))/h**2
    high=np.asarray(m['positions']).copy();low=np.asarray(m['position_low']).copy()
    for node in range(3):
        for axis in range(3):
            k=6*node+axis
            high[node,axis],low[node,axis]=split_sum((float(high[node,axis]),float(low[node,axis]),
                .5*h*float(v0[k]),.25*h*h*float(acceleration[k])))
    frames=np.array([native_rotation(d[6*n+3:6*n+6])@np.asarray(m['nodal_frames'])[n] for n in range(3)])
    oldcells=np.asarray(m['cell_rotations']).reshape(2,3,3)
    cells=np.array([native_rotation(.5*d[18+3*c:21+3*c])@oldcells[c] for c in (0,1)])
    resultants=np.asarray(m['resultants']).reshape(18)+d[24:]
    mechanical=SimpleNamespace(positions=high,position_low=low,nodal_frames=frames,
                              cell_rotations=cells[np.newaxis],resultants=resultants[np.newaxis])
    stage=_native_stage_record(owner,mechanical,velocity,acceleration)
    bound=dict(cell_alpha=float(np.max(alpha_bound[9:]))/ref.ACC_SCALE,
               nodal_acceleration=float(np.max(alpha_bound[:9]))/(ref.LENGTH*ref.ACC_SCALE),
               position=float(np.max(.25*h*h*alpha_bound[:9]))/ref.LENGTH)
    stage.update(evidence_kind='RECONSTRUCTED_REEVALUATED_NATIVE_STAGE',rounding_bound=bound,
                 alpha_rounding_bounds=alpha_bound.tolist())
    # Materialized JSON-compatible evidence retains all native algebraic fields.
    return json.loads(json.dumps(stage,default=lambda value:value.tolist(),allow_nan=False))


def _saved_coarse():
    path=os.environ.get('C7_NONPRINCIPAL_COARSE_RECORD')
    if not path:return run()
    raw=Path(path).read_bytes()
    assert sha256(raw).hexdigest()==os.environ['C7_NONPRINCIPAL_COARSE_SHA256']
    saved=json.loads(raw);capsule=Path(saved['checkpoint_path']).read_bytes()
    assert sha256(capsule).hexdigest()==saved['checkpoint_sha256']
    owner=run(steps=0)[0]  # Initialize a matching native operator; no step replay.
    data=saved['result']
    assert data['status']=='completed' and data['accepted_steps']==ref.STEPS
    assert len(data['steps'])==ref.STEPS and all(row['dt']==ref.DT for row in data['steps'])
    result=SimpleNamespace(to_dict=lambda:data,checkpoint=capsule,success=True)
    return owner,result


def test_nonprincipal_closed_spatial_work_inertia_and_geodesic_trace():
    owner,result=run(steps=0);_save('work',result,None)
    initial=result.to_dict()['initial'];x,u,v,a,_,_= _unpack(initial)
    # Actual arbitrary-axis, nonuniform finite pose is local operator evidence,
    # not a claimed force-consistent transient initial state.
    xx=ref.POINTS+np.array(((.01,-.02,.03),(-.01,.01,-.02),(.02,.03,-.01)))
    uu=np.array((ref.exp(np.array((.3,-.2,.1))),ref.exp(np.array((.31,-.18,.08)))))
    energy,force,fields=ref.elastic(xx,uu);full=ref.full_trace_work(xx,uu,fields['frames'])
    assert np.max(np.abs(full[1]))/ref.MOMENT_SCALE<=1.e-6
    assert np.max(np.abs(full[3]-fields['moments']))/ref.MOMENT_SCALE<=1.e-6
    direction=np.array((.2,-.1,.3,-.3,.2,.1,.1,.4,-.2,.3,-.2,.1,-.1,.2,.4))
    work=[]
    for h in (1.e-3,1.e-4,1.e-5):
        plus=ref.displaced(xx,uu,h*direction);minus=ref.displaced(xx,uu,-h*direction)
        fd=(ref.elastic(*plus)[0]-ref.elastic(*minus)[0])/(2*h)
        work.append(abs(fd-force@direction)/ENERGY_SCALE)
    # Compare actual retained mixed work with independent eliminated work;
    # only massless/algebraic traces are reduced in this local tangent check.
    op=owner._owner()._dynamic.element.operator
    p=np.r_[fields['forces'].ravel(),fields['moments'].ravel()]
    response=op.evaluate(xx,np.zeros((3,3)),fields['frames'],uu,p)
    residual=np.asarray(response.residual);k=np.asarray(response.hessian)+np.asarray(response.hessian_low)
    algebraic=[i for i in range(42) if i not in PHYSICAL]
    reduced=k[np.ix_(PHYSICAL,PHYSICAL)]-k[np.ix_(PHYSICAL,algebraic)]@np.linalg.solve(k[np.ix_(algebraic,algebraic)],k[np.ix_(algebraic,PHYSICAL)])
    gradient_error=max(float(np.max(np.abs(residual[PHYSICAL[:9]]-force[:9])))/ref.FORCE_SCALE,
                       float(np.max(np.abs(residual[PHYSICAL[9:]]-force[9:])))/ref.MOMENT_SCALE)
    tangent=[];directional=[]
    for h in (1.e-3,1.e-4,1.e-5):
        plus=ref.displaced(xx,uu,h*direction);minus=ref.displaced(xx,uu,-h*direction)
        fd=(ref.elastic(*plus)[0]-2*energy+ref.elastic(*minus)[0])/h**2
        tangent.append(abs(fd-direction@reduced@direction)/ENERGY_SCALE)
        fd_gradient=(ref.chart_gradient(xx,uu,h*direction)-ref.chart_gradient(xx,uu,-h*direction))/(2*h)
        error=fd_gradient-reduced@direction
        directional.append(dict(force=float(np.max(np.abs(error[:9])))/ref.FORCE_SCALE,
                                moment=float(np.max(np.abs(error[9:])))/ref.MOMENT_SCALE))
    # Test all inertia blocks at a finite pose, including exact algebraic/cross
    # zeros and genuine anisotropic gyro. This probe is not a transient state.
    pose=SimpleNamespace(positions=xx,position_low=np.zeros((3,3)),cell_rotations=uu[np.newaxis])
    velocity24=np.zeros(24);acceleration24=np.zeros(24)
    velocity24[PHYSICAL]=v;acceleration24[PHYSICAL]=direction*ref.ACC_SCALE
    actual=owner._owner()._inertia(pose,velocity24,acceleration24)
    expected_mass=ref.physical_mass(uu);actual_mass=np.asarray(actual.mixed_mass)[np.ix_(PHYSICAL,PHYSICAL)]
    mass_error=max(float(np.max(np.abs(actual_mass[:9,:9]-expected_mass[:9,:9])))/(ref.RHO*ref.LENGTH),
        float(np.max(np.abs(actual_mass[9:,9:]-expected_mass[9:,9:])))/(ref.LENGTH*min(ref.JDENSITY)))
    mass_zero_error=max(float(np.max(np.abs(actual_mass[:9,9:]))),float(np.max(np.abs(actual_mass[9:,:9]))),
        float(np.max(np.abs(np.asarray(actual.mixed_mass)[algebraic,:]))),
        float(np.max(np.abs(np.asarray(actual.mixed_mass)[:,algebraic]))))
    inertia_error=np.asarray(actual.mixed_inertia)[PHYSICAL]-(expected_mass@acceleration24[PHYSICAL]+ref.gyro(uu,v))
    inertia_error=max(float(np.max(np.abs(inertia_error[:9])))/ref.FORCE_SCALE,
                      float(np.max(np.abs(inertia_error[9:])))/ref.MOMENT_SCALE)
    initial_expected=ref.initial();initial_error=_errors(initial,dict(x=initial_expected[0],u=initial_expected[1],v=initial_expected[2],a=initial_expected[3]))
    algebraic_error=dict(trace=float(np.max(np.abs(residual[[3,4,5,9,10,11,15,16,17]])))/ref.MOMENT_SCALE,
        chord=float(np.max(np.abs(residual[24:30])))/ref.ELL,angle=float(np.max(np.abs(residual[30:42]))))
    metrics=dict(work=work,tangent=tangent,directional=directional,gradient=gradient_error,algebraic_residual=algebraic_error,mass=mass_error,mass_zero=mass_zero_error,inertia=inertia_error,initial=initial_error,gyro_norm=float(np.linalg.norm(ref.gyro(uu,v))))
    _save('work',result,metrics)
    assert metrics['gyro_norm']>0 and max(work)<=1.e-5 and mass_error<=1.e-5
    assert mass_zero_error==0. and inertia_error<=1.e-6
    assert max(initial_error.values())<=1.e-6
    assert gradient_error<=1.e-6 and max(tangent)<=1.e-5
    assert max(max(e.values()) for e in directional)<=1.e-5 and max(algebraic_error.values())<=1.e-6
    assert abs(ENERGY_SCALE-59/15)<=1.e-13


@pytest.fixture(scope="module")
def coarse_result():
    owner,result=_saved_coarse();_save('coarse',result,None)
    return owner,result


def _trajectory_metrics(owner,result,captures=None):
    data=result.to_dict();initial=ref.initial();x,u,v=initial[:3]
    errors=[_errors(data['initial'],dict(x=x,u=u,v=v,a=initial[3]))];balances=[_physics(data['initial'])]
    energy=[];pdefect=[];hdefect=[];strain=[];angles=[];gyros=[];relative=[];moments=[];commutators=[];discrete_impulse_error=[];impulse_rows=[]
    initial_cells=u.copy();actual_initial=_unpack(data['initial']);previous_h=ref.account(*actual_initial[:3])['angular_momentum']
    actual_initial_h=previous_h.copy();cumulative_impulse=np.zeros(3);cumulative_defect=np.zeros(3)
    initial_account=ref.account(x,u,v);previous_record=data['initial'];stage_bounds=[];reconstructed_stages=[];capture_checks=[]
    for row in data['steps']:
        stage=_reconstructed_stage(owner,previous_record,row)
        stage_bounds.append(stage['rounding_bound']);reconstructed_stages.append(stage)
        if captures is not None:
            actual=captures[row['endpoint']['epoch']]
            reconstructed=_unpack(stage);captured=_unpack(actual)
            scales=(ref.LENGTH,1.,ref.SPIN_SCALE,ref.ACC_SCALE,1.)
            check={str(index):float(np.max(np.abs(a-b)))/scale
                   for index,(a,b,scale) in enumerate(zip(reconstructed,captured,scales))}
            check['saved_increment_match']=0. if np.array_equal(np.asarray(actual['increment42']),np.asarray(row['midpoint_physical_increment42'])) else 1.
            check['dt_match']=0. if actual['dt']==row['dt'] else 1.
            check['endpoint_epoch_match']=0. if actual['endpoint_epoch']==row['endpoint']['epoch'] else 1.
            check['stage_material_match']=0. if actual['stage_material_sha256']==row['stage_material_sha256'] else 1.
            check['predecessor_match']=0. if actual['predecessor_sha256']==row['predecessor_sha256'] else 1.
            check['retained_force']=float(np.max(np.abs(reconstructed[5][:6]-captured[5][:6])))/ref.FORCE_SCALE
            check['retained_moment']=float(np.max(np.abs(reconstructed[5][6:]-captured[5][6:])))/ref.MOMENT_SCALE
            alpha_error=np.abs(reconstructed[3]-captured[3])
            alpha_enclosure=np.asarray(stage['alpha_rounding_bounds'])+8*2.**-53*np.abs(captured[3])
            check['alpha_rounding_bound_match']=0. if np.all(alpha_error<=alpha_enclosure) else 1.
            check['position_low']=float(np.max(np.abs(np.asarray(stage['state']['mechanical']['position_low'])-
                np.asarray(actual['state']['mechanical']['position_low']))))/ref.LENGTH
            capture_checks.append(check);stage=actual
        expected=ref.step(x,u,v,row['dt'])
        errors.append(_errors(stage,dict(x=expected['stage_x'],u=expected['stage_u'],v=expected['stage_v'],a=expected['stage_a'])))
        errors.append(_errors(row['endpoint'],expected));balances.extend((_physics(stage),_physics(row['endpoint'])))
        x,u,v,a,_,_= _unpack(row['endpoint']);value=ref.account(x,u,v)
        xs,us,vs,as_,_,_= _unpack(stage)
        predicted=ref.discrete_angular_increment(xs,us,vs,as_,row['dt'])
        orbital=np.sum(np.cross(xs,(ref.translation_mass()@as_[:9]).reshape(3,3)),axis=0)
        spin=(ref.physical_mass(us)@as_+ref.gyro(us,vs))[9:].reshape(2,3).sum(axis=0)
        stage_impulse=row['dt']*(orbital+spin);method_defect=predicted-stage_impulse
        cumulative_impulse+=predicted;cumulative_defect+=method_defect
        discrete_impulse_error.append(float(np.max(np.abs(value['angular_momentum']-previous_h-predicted)))/ANGULAR_SCALE)
        impulse_rows.append(dict(dt=row['dt'],exact_increment=predicted.tolist(),stage_linear_impulse=stage_impulse.tolist(),
            finite_so3_defect=method_defect.tolist(),cumulative_exact_increment=cumulative_impulse.tolist(),
            cumulative_finite_so3_defect=cumulative_defect.tolist(),
            cumulative_accounting_error=float(np.max(np.abs(value['angular_momentum']-actual_initial_h-cumulative_impulse)))/ANGULAR_SCALE))
        previous_h=value['angular_momentum'].copy()
        energy.append(abs(value['total_energy']-initial_account['total_energy'])/ENERGY_SCALE)
        pdefect.append(float(np.max(np.abs(value['linear_momentum']-initial_account['linear_momentum'])))/MOMENTUM_SCALE)
        hdefect.append(float(np.max(np.abs(value['angular_momentum']-initial_account['angular_momentum'])))/ANGULAR_SCALE)
        strain.append(value['elastic_energy']);gyros.append(float(np.linalg.norm(ref.gyro(u,v))))
        fields=ref.elastic(x,u)[2];relative.append(float(np.linalg.norm(fields['gamma'])))
        moments.append(float(np.max(np.abs(fields['moments']))))
        increments=np.array([u[c]@initial_cells[c].T for c in (0,1)])
        commutators.append(float(np.linalg.norm(increments[0]@increments[1]-increments[1]@increments[0],ord=np.inf)))
        angles.append(max(ref.rotation_angle(r) for r in u))
        # Continue the independent recurrence, not reseeded from actual state.
        x,u,v=expected['x'],expected['u'],expected['v']
        previous_record=row['endpoint']
    return dict(errors=errors,balances=balances,energy_defect=energy,linear_defect=pdefect,angular_defect=hdefect,
        strain_energy=strain,angles=angles,gyro_norm=gyros,relative_angle=relative,retained_moment=moments,
        accumulated_rotation_commutator=commutators,discrete_angular_impulse_error=discrete_impulse_error,
        angular_impulse_rows=impulse_rows,reconstructed_stages=reconstructed_stages,stage_rounding_bounds=stage_bounds,
        captured_stages=captures,capture_reconstruction_errors=capture_checks)


def _assert_trajectory(result,metrics,steps,*,require_full_exposure=True):
    assert result.success and result.to_dict()['accepted_steps']==steps
    assert len(metrics['angular_impulse_rows'])==steps
    assert max(max(row.values()) for row in metrics['stage_rounding_bounds'])<=1.e-9
    assert all(max(row.values())<=1.e-9 for row in metrics['capture_reconstruction_errors'])
    assert max(max(e.values()) for e in metrics['errors'])<=1.e-6
    assert max(max(e.values()) for e in metrics['balances'])<=1.e-6
    if require_full_exposure:
        assert max(metrics['angles'])>=.2 and max(metrics['strain_energy'])>0 and min(metrics['gyro_norm'])>0
        assert max(metrics['relative_angle'])>=.01 and max(metrics['retained_moment'])>=ref.EI*.01/ref.ELL
        assert max(metrics['accumulated_rotation_commutator'])>=1.e-5
    assert max(metrics['discrete_angular_impulse_error'])<=1.e-6
    assert max(r['cumulative_accounting_error'] for r in metrics['angular_impulse_rows'])<=1.e-6
    assert max(metrics['energy_defect'])<=.01 and max(metrics['linear_defect'])<=1.e-6 and max(metrics['angular_defect'])<=1.e-6


def test_public_nonprincipal_nonlinear_midpoint_and_finite_angle(coarse_result):
    owner,result=coarse_result;metrics=_trajectory_metrics(owner,result);_save('coarse',result,metrics)
    _assert_trajectory(result,metrics,ref.STEPS)


def test_public_nonprincipal_temporal_reference_and_refinement(coarse_result):
    _,coarse=coarse_result;fine_owner,fine,captures=_run_captured(steps=2*ref.STEPS,dt=ref.DT/2)
    _save('fine',fine,dict(captured_stages=captures))
    exact=ref.semidiscrete(subdivisions=64);finer=ref.semidiscrete(subdivisions=128)
    reference_rows=[ref.temporal_errors(a['x'],a['u'],a['v'],b) for a,b in zip(exact,finer)]
    reference_error=max(max(e.values()) for e in reference_rows)
    def comparison(result,stride):
        # Compare every common accepted endpoint at the fixed physical times;
        # the independent ODE integrations have the same frozen inventory.
        rows=[ref.temporal_errors(*_unpack(row['endpoint'])[:3],finer[index//stride+1])
              for index,row in enumerate(result.to_dict()['steps']) if (index+1)%stride==0]
        response={key:max(row[key] for row in rows) for key in rows[0]}
        x,u,v,_,_,_= _unpack(result.to_dict()['final'])
        defect=abs(ref.account(x,u,v)['total_energy']-ENERGY_SCALE)/ENERGY_SCALE
        return response,defect,rows
    c=comparison(coarse,1);f=comparison(fine,2);fine_metrics=_trajectory_metrics(fine_owner,fine,captures)
    metrics=dict(reference_refinement=reference_error,reference_response_rows=reference_rows,coarse=c,fine=f,
        fine_trajectory=fine_metrics,temporal_scales=dict(position_m=ref.LENGTH,rotation_rad=ref.ANGLE_SCALE,
            nodal_velocity_m_per_s=ref.TRANSLATION_VELOCITY_SCALE,cell_omega_rad_per_s=ref.SPIN_SCALE))
    _save('fine',fine,metrics)
    _assert_trajectory(fine,fine_metrics,2*ref.STEPS)
    assert reference_error<=1.e-7 and max(c[0].values())<=.02 and max(f[0].values())<=.02
    assert max(f[0].values())<max(c[0].values())
    assert c[1]<=.01 and f[1]<=.01 and f[1]<c[1]


def test_public_nonprincipal_fresh_four_equals_two_plus_two(coarse_result):
    _,full=coarse_result;_save('restart_full',full,None)
    _,prefix,captures=_run_captured(steps=ref.STEPS//2);_save('restart_prefix',prefix,dict(captured_stages=captures))
    _,resumed,captures=_run_captured(checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    _save('restart_resumed',resumed,dict(captured_stages=captures))
    assert full.success and prefix.success and resumed.success
    assert full.to_dict()==resumed.to_dict() and full.checkpoint==resumed.checkpoint


def _bound_record(environment):
    path=os.environ.get(environment)
    if not path:return None
    raw=Path(path).read_bytes()
    assert sha256(raw).hexdigest()==os.environ[environment+'_SHA256']
    record=json.loads(raw);capsule=Path(record['checkpoint_path']).read_bytes()
    assert sha256(capsule).hexdigest()==record['checkpoint_sha256']
    return record,capsule


def _common_response(data,references,stride):
    references=[{key:np.asarray(value) for key,value in row.items()} for row in references]
    rows=[ref.temporal_errors(*_unpack(row['endpoint'])[:3],references[index//stride+1])
          for index,row in enumerate(data['steps']) if (index+1)%stride==0]
    return {key:max(row[key] for row in rows) for key in rows[0]},rows


def _correction_baseline():
    # V3b saved errors and convergence but omitted full ODE state fields.
    # Regenerate only its unchanged128 reference once; never replay its8 solve
    # in the held execution. Ordinary tests remain self-contained and fresh.
    references=ref.semidiscrete(subdivisions=128)
    saved=_bound_record('C7_NONPRINCIPAL_FINE_RECORD')
    if saved:
        record,_=saved;data=record['result'];old=record['metrics']
        assert data['status']=='completed' and data['accepted_steps']==2*ref.STEPS
        assert all(row['dt']==ref.DT/2 for row in data['steps'])
        response=old['fine'][0];energy=old['fine'][1];convergence=old['reference_refinement']
        provenance=dict(source='BOUND_EXECUTED8_NONPASS; REFERENCE128_STATES_REGENERATED_ONCE',
                        record_sha256=os.environ['C7_NONPRINCIPAL_FINE_RECORD_SHA256'])
    else:
        _,fine=run(steps=2*ref.STEPS,dt=ref.DT/2);data=fine.to_dict();assert fine.success
        response,_=_common_response(data,references,2)
        x,u,v,_,_,_=_unpack(data['final']);energy=abs(ref.account(x,u,v)['total_energy']-ENERGY_SCALE)/ENERGY_SCALE
        lower=ref.semidiscrete(subdivisions=64)
        convergence=max(max(ref.temporal_errors(a['x'],a['u'],a['v'],b).values()) for a,b in zip(lower,references))
        provenance=dict(source='ORDINARY_FRESH_PUBLIC8_BASELINE')
    assert convergence<=1.e-7
    rows=[{key:value.tolist() for key,value in row.items()} for row in references]
    return dict(reference128=rows,reference_refinement=convergence,baseline8_response=response,
                baseline8_energy=energy,baseline8_initial=data['initial'],provenance=provenance)


def _prepare_corrective_prefix():
    owner,result,captures=_run_captured(steps=2*ref.STEPS,dt=ref.DT/4)
    _save('corrective_prefix',result,dict(captured_stages=captures))
    trajectory=_trajectory_metrics(owner,result,captures);baseline=_correction_baseline()
    metrics=dict(trajectory=trajectory,baseline=baseline,segment='PREFIX8_OF_CORRECTED16',
                 dt=ref.DT/4,full_horizon=ref.STEPS*ref.DT,prefix_horizon=2*ref.STEPS*(ref.DT/4))
    _save('corrective_prefix',result,metrics)
    _assert_trajectory(result,trajectory,2*ref.STEPS,require_full_exposure=False)
    assert len(captures)==2*ref.STEPS
    assert result.to_dict()['initial']==baseline['baseline8_initial']
    return result,metrics


def test_public_nonprincipal_corrective_prefix_eight():
    _prepare_corrective_prefix()


def test_public_nonprincipal_corrective_resumed_sixteen():
    bound=_bound_record('C7_NONPRINCIPAL_CORRECTIVE_PREFIX_RECORD')
    if bound:
        record,capsule=bound;prefix_data=record['result'];prefix_metrics=record['metrics']
    else:
        prefix,prefix_metrics=_prepare_corrective_prefix();capsule=prefix.checkpoint;prefix_data=prefix.to_dict()
    assert prefix_data['status']=='completed' and prefix_data['accepted_steps']==2*ref.STEPS
    assert prefix_metrics['segment']=='PREFIX8_OF_CORRECTED16'
    assert prefix_metrics['dt']==ref.DT/4 and all(row['dt']==ref.DT/4 for row in prefix_data['steps'])
    owner,result,captures=_run_captured(steps=4*ref.STEPS,dt=ref.DT/4,
        checkpoint=capsule,expected_sha256=sha256(capsule).hexdigest())
    _save('corrective_full',result,dict(new_captured_stages=captures))
    data=result.to_dict();previous_captures={int(key):value for key,value in prefix_metrics['trajectory']['captured_stages'].items()}
    overlap=bool(set(previous_captures)&set(captures));combined={**previous_captures,**captures}
    trajectory=_trajectory_metrics(owner,result,combined)
    baseline=prefix_metrics['baseline'];response,response_rows=_common_response(data,baseline['reference128'],4)
    energy=max(trajectory['energy_defect'])
    chain=dict(prefix_capsule_sha256=sha256(capsule).hexdigest(),prefix_steps=2*ref.STEPS,
        new_steps=len(captures),capture_overlap=overlap,
        initial_equal=data['initial']==prefix_data['initial'],
        prefix_steps_equal=data['steps'][:2*ref.STEPS]==prefix_data['steps'],
        prefix_outputs_equal=data['outputs'][:len(prefix_data['outputs'])]==prefix_data['outputs'],
        restored_predecessor_equal=data['steps'][2*ref.STEPS]['predecessor_sha256']==prefix_data['final']['state_sha256'])
    metrics=dict(trajectory=trajectory,baseline=baseline,response=response,response_rows=response_rows,
                 maximum_energy_defect=energy,checkpoint_chain=chain,
                 claim='ACTUAL_PUBLIC8_PLUS8_PREFIX_CONSISTENCY; NO_SEPARATE_FRESH16_RUN')
    _save('corrective_full',result,metrics)
    _assert_trajectory(result,trajectory,4*ref.STEPS)
    assert len(captures)==2*ref.STEPS and len(combined)==4*ref.STEPS and not overlap
    assert all(chain[key] for key in ('initial_equal','prefix_steps_equal','prefix_outputs_equal','restored_predecessor_equal'))
    assert baseline['reference_refinement']<=1.e-7
    assert max(response.values())<=.02 and max(response.values())<max(baseline['baseline8_response'].values())
    assert energy<=.01 and energy<baseline['baseline8_energy']
