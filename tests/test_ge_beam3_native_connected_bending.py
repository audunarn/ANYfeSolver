"""Prepared connected transverse/rotary reference checks; numerical UNEXECUTED."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import numpy as np
from examples import native_connected_bending_reference as ref
from examples.native_connected_bending_dev import run

TRANSLATIONS=[6*i+a for i in range(5) for a in range(3)]
NODAL_ROTATIONS=[6*i+3+a for i in range(5) for a in range(3)]
CELL_ROTATIONS=list(range(30,42))
PLANE=[6*i for i in range(5)]+[6*i+1 for i in range(5)]+[32,35,38,41]


def _save(tag,result,metrics=None):
    assert tag in {'response','work','split_full','split_prefix','split_resumed'}
    destination=os.environ.get('NATIVE_TRANSIENT_RESULT_RECORD')
    if not destination:return
    base=Path(destination);base.parent.mkdir(parents=True,exist_ok=True)
    capsule=base.with_name(base.stem+'_'+tag+'.checkpoint');capsule.write_bytes(result.checkpoint)
    report=base.with_name(base.stem+'_'+tag+'.json')
    report.write_text(json.dumps(dict(result=result.to_dict(),metrics=metrics,
        checkpoint_sha256=sha256(result.checkpoint).hexdigest(),checkpoint_path=str(capsule)),
        sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def _plane(record):
    state=record['state'];positions=np.asarray(state['positions'])+np.asarray(state['position_low'])
    baseline=ref.ORIGIN+np.arange(5)[:,None]*ref.ELL*np.array((1.,0.,0.))
    delta=positions-baseline;cells=np.asarray(state['cell_rotations']).reshape(4,3,3)
    q=np.r_[delta[:,0],delta[:,1],np.arctan2(cells[:,1,0],cells[:,0,0])]
    return q,np.asarray(state['velocity'])[PLANE],np.asarray(state['acceleration'])[PLANE]


def _rz(angle):
    c=math.cos(float(angle));s=math.sin(float(angle))
    return np.array(((c,-s,0.),(s,c,0.),(0.,0.,1.)))


def _state_errors(record,q,v,a,mode):
    actualq,actualv,actuala=_plane(record);scale=mode['coordinate_scale'];omega=mode['omega']
    state=record['state'];fields=ref.storage(q)[3]
    cells=np.asarray(state['cell_rotations']).reshape(4,3,3)
    nodal=np.asarray(state['nodal_frames']);retained=np.asarray(state['resultants']).reshape(2,18)
    error=dict(q=float(np.max(np.abs(actualq-q)/scale)),
        v=float(np.max(np.abs(actualv-v)/(omega*scale))),
        a=float(np.max(np.abs(actuala-a)/(omega**2*scale))))
    # Complete vector comparisons, including nominally zero transverse/spin axes.
    expected_v=np.zeros(78);expected_a=np.zeros(78);expected_v[PLANE]=v;expected_a[PLANE]=a
    translations=np.asarray(state['positions'])+np.asarray(state['position_low'])
    error['out_of_plane_position']=float(np.max(np.abs(translations[:,2]-ref.ORIGIN[2])))/ref.AMP
    for name,expected,factor in (('velocity',expected_v,omega),('acceleration',expected_a,omega**2)):
        observed=np.asarray(state[name]);err=observed-expected
        error[name+'_full_translation']=float(np.max(np.abs(err[TRANSLATIONS])))/(ref.AMP*factor)
        error[name+'_full_cell_spin']=float(np.max(np.abs(err[CELL_ROTATIONS])))/(max(scale[10:])*factor)
        # Saved algebraic rates are zero by owner convention, not theta derivatives.
        error[name+'_nodal_frame_rate']=float(np.max(np.abs(observed[NODAL_ROTATIONS])))/(max(scale[10:])*factor)
        # Retained force/moment rates have distinct units; owner stores exact zero.
        error[name+'_retained_exact_zero']=0. if not np.any(observed[42:]) else 1.
    error['cell_frames']=max(float(np.max(np.abs(r-_rz(p)))) for r,p in zip(cells,q[10:]))/max(scale[10:])
    error['nodal_frames']=max(float(np.max(np.abs(r-_rz(p)))) for r,p in zip(nodal,fields['theta']))/max(scale[10:])
    error['retained_force']=float(np.max(np.abs(retained[:,:6].reshape(4,3)-fields['forces'])))/ref.FORCE_SCALE
    error['retained_moment']=float(np.max(np.abs(retained[:,6:].reshape(4,2,3)-fields['moments'])))/ref.MOMENT_SCALE
    return error


def _account_errors(actual,q,v,mode):
    expected=ref.account(q,v);error={}
    for key in ('elastic_energy','kinetic_energy','total_energy'):
        error[key]=abs(actual[key]-expected[key])/mode['energy_scale']
    for key,scale in (('linear_momentum',ref.FORCE_SCALE*mode['dt']),('angular_momentum',ref.MOMENT_SCALE*mode['dt'])):
        value=np.asarray(actual[key])+np.asarray(actual[key+'_low'])
        error[key]=float(np.max(np.abs(value-expected[key])))/scale
    return error


def test_public_bending_stage_support_response_and_semidiscrete_bound():
    owner,result,mode=run(4);data=result.to_dict();_save('response',result)
    assert result.success and data['accepted_steps']==4
    q=mode['q'].copy();v=mode['v'].copy();bounds=ref.linear_remainder_bound(mode)
    errors=[_state_errors(data['initial'],q,v,ref.acceleration(q),mode),
            _account_errors(data['initial']['accounting'],q,v,mode)]
    nonlinear_linear=[];analytic=[];actual_analytic=[];remainder_checks=[];remainder_excess=[];support_errors=[];impulse_errors=[];energy_defects=[]
    angles=[];moments=[];couples=[];spinrates=[];stage_energy=[];actual_stage_energy=[]
    previous=data['initial']['accounting']
    for index,row in enumerate(data['steps'],1):
        expected=ref.step(q,v,row['dt']);stage=expected['stage_q'];stagev=expected['stage_v']
        errors.extend((_state_errors(row['stage'],stage,stagev,expected['stage_a'],mode),
            _state_errors(row['endpoint'],expected['q'],expected['v'],expected['a'],mode),
            _account_errors(row['stage']['accounting'],stage,stagev,mode),
            _account_errors(row['endpoint']['accounting'],expected['q'],expected['v'],mode)))
        reaction=np.asarray(row['midpoint_support']['reaction_global']);endpoint=np.asarray(row['endpoint_reaction_global'])
        endpoint_force=ref.mass_matrix()@expected['a']+ref.storage(expected['q'])[1]
        endpoint_couple=ref.storage(expected['q'])[3]['root_couple']
        support_errors.append(dict(
            force=float(np.max(np.abs(reaction[:3]-expected['support_force'])))/ref.FORCE_SCALE,
            couple=float(np.max(np.abs(reaction[3:6]-expected['support_couple'])))/ref.MOMENT_SCALE,
            force_impulse=float(np.max(np.abs(np.asarray(row['midpoint_support']['force_impulse_ns'])-expected['force_impulse'])))/(ref.FORCE_SCALE*row['dt']),
            moment_impulse=float(np.max(np.abs(np.asarray(row['midpoint_support']['moment_impulse_nms'])-expected['moment_impulse'])))/(ref.MOMENT_SCALE*row['dt']),
            endpoint_force=float(np.max(np.abs(endpoint[:3]-np.array((endpoint_force[0],endpoint_force[5],0.)))))/ref.FORCE_SCALE,
            endpoint_couple=float(np.max(np.abs(endpoint[3:6]-np.array((0.,0.,endpoint_couple)))))/ref.MOMENT_SCALE,
            work=abs(row['midpoint_support']['work_j'])/mode['energy_scale']))
        current=row['endpoint']['accounting'];balance={}
        for key,impulse,scale in (('linear_momentum','force_impulse_ns',ref.FORCE_SCALE),('angular_momentum','moment_impulse_nms',ref.MOMENT_SCALE)):
            change=np.asarray(current[key])+np.asarray(current[key+'_low'])-np.asarray(previous[key])-np.asarray(previous[key+'_low'])
            balance[key]=float(np.max(np.abs(change-np.asarray(row['midpoint_support'][impulse]))))/(scale*row['dt'])
        impulse_errors.append(balance);previous=current
        energy_defects.append(abs(current['total_energy']-data['initial']['accounting']['total_energy'])/mode['energy_scale'])
        stage_energy.append(ref.account(stage,stagev)['total_energy'])
        actual_stage_energy.append(row['stage']['accounting']['total_energy'])
        q,v=expected['q'],expected['v'];linear=ref.linear_midpoint_exact(index,mode);exact=ref.semidiscrete_exact(row['endpoint']['time'],mode)
        actualq,actualv,_=_plane(row['endpoint'])
        for observed,actual,rest,continuous,scale,bound in zip((q,v),(actualq,actualv),linear[:2],exact[:2],
                (mode['coordinate_scale'],mode['omega']*mode['coordinate_scale']), (bounds['q'],bounds['v'])):
            # Measured model difference is checked against a separately derived
            # conditional bound; numerical comparison allowance is explicitly separate.
            remainder_checks.append(bool(np.all(np.abs(observed-rest)<=bound+1.e-6*scale)))
            remainder_excess.append(float(np.max((np.abs(observed-rest)-bound)/scale)))
            nonlinear_linear.append(float(np.max(np.abs(observed-rest)/scale)))
            analytic.append(float(np.max((bound+np.abs(rest-continuous))/scale)))
            actual_analytic.append(float(np.max(np.abs(actual-continuous)/scale)))
        fields=ref.storage(q)[3];angles.append(float(np.max(np.abs(q[10:]))));spinrates.append(float(np.max(np.abs(v[10:]))))
        moments.append(float(np.max(np.abs(fields['moments']))));couples.append(abs(expected['support_couple'][2]))
    metrics=dict(state_errors=errors,support_errors=support_errors,impulse_errors=impulse_errors,
        maximum_energy_defect=max(energy_defects),model_discrepancy=max(nonlinear_linear),
        analytic_bound=max(analytic),actual_analytic_response=max(actual_analytic),
        remainder_checks=remainder_checks,maximum_bound_excess=max(remainder_excess),
        eigen_residual_upper=bounds['eigen_residual_upper'].tolist(),energy_box=bounds['energy_box'],stage_energy=stage_energy,actual_stage_energy=actual_stage_energy,
        modal_phase=4*mode['phase_per_step'],maximum_cell_angle=max(angles),maximum_spin_rate=max(spinrates),
        maximum_retained_bending_moment=max(moments),maximum_root_couple=max(couples))
    _save('response',result,metrics)
    assert max(max(x.values()) for x in errors)<=1.e-6
    assert max(max(x.values()) for x in support_errors+impulse_errors)<=1.e-6
    assert max(stage_energy+actual_stage_energy)<=bounds['energy_box']
    assert all(remainder_checks)
    assert max(actual_analytic)<=.001
    assert max(energy_defects)<=.02
    assert .199<metrics['modal_phase']<=.2
    assert min(metrics['maximum_cell_angle'],metrics['maximum_spin_rate'],metrics['maximum_retained_bending_moment'],metrics['maximum_root_couple'])>0.
    assert data['final']['accounting']['kinetic_energy']>0.


def test_independent_bending_mass_complete_virtual_work_and_tangent_interval():
    owner,result,mode=run(0);_save('work',result);state=owner.current;q,_,_=_plane(result.to_dict()['initial'])
    mass=ref.mass_matrix();physical_mass=ref.physical_mass_matrix(q);energy,gradient,hessian,fields=ref.storage(q)
    with owner._operation():
        residual,jacobian,_,actual_mass,*_=owner._evaluate(state)
        z=list(owner.z);condensed=jacobian-jacobian[:,z]@np.linalg.solve(jacobian[np.ix_(z,z)],jacobian[z,:])
    direction=mode['coordinate_scale']*np.array((0.,.2,-.1,.3,.4,0.,.3,-.2,.1,.4,.2,-.3,.4,-.1))
    actual_direction=np.zeros(owner.size);actual_direction[PLANE]=direction
    response=condensed@actual_direction;expected=gradient@direction;expected_h=hessian@direction
    measured_physical_mass=actual_mass[np.ix_(TRANSLATIONS+CELL_ROTATIONS,TRANSLATIONS+CELL_ROTATIONS)]
    mass_error=measured_physical_mass-physical_mass
    rotary_off=mass_error[15:,15:].copy();np.fill_diagonal(rotary_off,0.)
    # kg, kg m^2, kg m blocks never share a largest-entry denominator.
    translation_mass_scale=ref.RHO_A*ref.ELL
    rotary_mass_scale=ref.ELL*max(ref.JX,ref.JY,ref.JZ)
    cross_mass_scale=ref.RHO_A*ref.ELL**2
    rotated_off_scale=ref.ELL*abs(ref.JX-ref.JY)*max(abs(q[10:]))
    assert rotated_off_scale>0.
    errors=dict(mass_plane=float(np.linalg.norm(actual_mass[np.ix_(PLANE,PLANE)]-mass)/np.linalg.norm(mass)),
        mass_translation=float(np.max(np.abs(mass_error[:15,:15])))/translation_mass_scale,
        mass_rotary=float(np.max(np.abs(mass_error[15:,15:])))/rotary_mass_scale,
        mass_cross=float(max(np.max(np.abs(mass_error[:15,15:])),np.max(np.abs(mass_error[15:,:15]))))/cross_mass_scale,
        mass_rotated_off_diagonal=float(np.max(np.abs(rotary_off)))/rotated_off_scale,
        gradient_force=float(np.max(np.abs(residual[PLANE[:10]]-gradient[:10])))/ref.FORCE_SCALE,
        gradient_moment=float(np.max(np.abs(residual[PLANE[10:]]-gradient[10:])))/ref.MOMENT_SCALE,
        work=abs(float(residual@actual_direction)-float(expected))/mode['energy_scale'],
        tangent_force=float(np.max(np.abs(response[PLANE[:10]]-expected_h[:10])))/ref.FORCE_SCALE,
        tangent_moment=float(np.max(np.abs(response[PLANE[10:]]-expected_h[10:])))/ref.MOMENT_SCALE)
    fd=[]
    for h in (1.e-2,1.e-3,1.e-4):
        ep,gp,*_=ref.storage(q+h*direction);em,gm,*_=ref.storage(q-h*direction)
        hresponse=(gp-gm)/(2*h)
        fd.append(dict(h=h,work=abs((ep-em)/(2*h)-float(residual@actual_direction))/mode['energy_scale'],
            force=float(np.max(np.abs(hresponse[:10]-response[PLANE[:10]])))/ref.FORCE_SCALE,
            moment=float(np.max(np.abs(hresponse[10:]-response[PLANE[10:]])))/ref.MOMENT_SCALE))
    _save('work',result,dict(errors=errors,finite_difference=fd,
        physical_maximum_translation_increment=1.e-2*float(np.max(np.abs(direction[:10]))),
        physical_maximum_angle_increment=1.e-2*float(np.max(np.abs(direction[10:])))))
    assert not np.any(actual_mass[list(owner.algebraic),:]) and not np.any(actual_mass[:,list(owner.algebraic)])
    assert max(errors.values())<=1.e-5
    assert max(max(x[k] for k in ('work','force','moment')) for x in fd)<=1.e-5
    off_translation=[i for i in TRANSLATIONS if i not in PLANE]
    off_spin=[i for i in CELL_ROTATIONS if i not in PLANE]
    assert np.max(np.abs(response[off_translation]))/ref.FORCE_SCALE<=1.e-5
    assert np.max(np.abs(response[off_spin]))/ref.MOMENT_SCALE<=1.e-5
    free_frames=[i for i in NODAL_ROTATIONS if i not in owner.fixed]
    assert np.max(np.abs(residual[free_frames]))/ref.MOMENT_SCALE<=1.e-5
    dual=residual[42:].reshape(2,18)
    # Retained force equations are lengths, moment equations are angles.
    assert np.max(np.abs(dual[:,:6]))/ref.AMP<=1.e-5
    assert np.max(np.abs(dual[:,6:]))/max(mode['coordinate_scale'][10:])<=1.e-5


def test_fresh_bending_four_equals_two_plus_two_complete_restart():
    _,full,_=run(4);_save('split_full',full)
    _,prefix,_=run(2);_save('split_prefix',prefix)
    _,resumed,_=run(4,checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    _save('split_resumed',resumed)
    assert resumed.to_dict()==full.to_dict() and resumed.checkpoint==full.checkpoint


def test_linear_midpoint_modal_phase_refinement():
    mode=ref.initial_mode();coarse=ref.linear_midpoint_exact(4,mode)
    fine_mode=dict(mode,dt=mode['dt']/2);fine=ref.linear_midpoint_exact(8,fine_mode)
    exact=ref.semidiscrete_exact(4*mode['dt'],mode)
    scale=mode['coordinate_scale'];omega=mode['omega']
    error=lambda value:max(float(np.max(np.abs(value[0]-exact[0])/scale)),float(np.max(np.abs(value[1]-exact[1])/(omega*scale))))
    assert error(fine)<error(coarse)<=.001

def test_stable_so3_successor_zero_arbitrary_axis_branches_and_jets():
    from anyfesolver._ge_beam3_centered_so3_strain import base_delta,rotation_jets
    from anyfesolver._ge_beam3_mixed_ad import Jet2
    import pytest
    axis=np.array((1.,2.,-3.));axis/=np.linalg.norm(axis)
    cross=np.array(((0.,-axis[2],axis[1]),(axis[2],0.,-axis[0]),(-axis[1],axis[0],0.)))
    square=cross@cross;weights=np.array(((.2,-.3,.1),(.4,.1,-.2),(-.1,.3,.5)))
    generators=[]
    for i in range(3):
        basis=np.zeros(3);basis[i]=1.
        generators.append(np.array(((0.,-basis[2],basis[1]),(basis[2],0.,-basis[0]),(-basis[1],basis[0],0.))))
    for angle in (0.,2.**-20,.3,math.pi/3-2.**-20,math.pi/3+2.**-20,1.2):
        expected_d=math.sin(angle)*cross+2*math.sin(angle/2)**2*square
        rotation=np.eye(3)+expected_d;delta=base_delta(rotation)
        if angle==0.:assert not np.any(delta)
        else:
            diagonal_scale=max(abs(np.diag(expected_d)))
            assert np.max(np.abs(np.diag(delta-expected_d)))/diagonal_scale<=128*np.finfo(float).eps
            assert np.linalg.norm(delta-expected_d)/np.linalg.norm(expected_d)<=128*np.finfo(float).eps
        made,made_d,_=rotation_jets(rotation,[Jet2.variable(0.,i,3) for i in range(3)])
        value=sum(weights[i,j]*made_d[i][j] for i in range(3) for j in range(3))
        base=np.eye(3)+delta
        expected_g=np.array([np.sum(weights*(a@base)) for a in generators])
        expected_h=np.array([[np.sum(weights*((a@b+b@a)/2@base)) for b in generators] for a in generators])
        assert np.max(np.abs(value.gradient-expected_g))<=1.e-5
        assert np.max(np.abs(value.hessian-expected_h))<=1.e-5
        direction=np.array((.2,-.3,.4))
        def evaluate(eta):
            _,d,_=rotation_jets(rotation,[Jet2.variable(float(x),i,3) for i,x in enumerate(eta)])
            return sum(weights[i,j]*d[i][j] for i in range(3) for j in range(3))
        point=np.array((.1,-.2,.15));at=evaluate(point)
        for h in (1.e-3,1.e-4,1.e-5):
            plus=evaluate(point+h*direction);minus=evaluate(point-h*direction)
            assert abs((plus.value-minus.value)/(2*h)-at.gradient@direction)<=1.e-5
            assert np.max(np.abs((plus.gradient-minus.gradient)/(2*h)-at.hessian@direction))<=1.e-5
    distorted=np.eye(3);distorted[0,0]+=1.e-12
    with pytest.raises(ValueError,match='machine-roundoff'):base_delta(distorted)


def test_stable_so3_definition_identity_restore_and_legacy_policy_isolation():
    from anyfesolver._ge_beam3_native_definition import (NativeBeamDefinition,SCHEMA,PRECISE_SCHEMA,STABLE_SO3_SCHEMA)
    from anyfesolver._ge_beam3_precise_geometric_work import POLICY as OLD_PRECISE
    from anyfesolver._ge_beam3_centered_so3_strain import POLICY as STABLE
    from anyfesolver.ge_beam3_native import define_beam,SELECTOR as BEAM
    from examples.native_connected_bending_dev import build,config
    from anyfesolver.ge_beam3_native_connected_dynamics import solve_transient
    import pytest
    owner=build();element=owner.elements[0];inertia=owner.analysis._inertias[element.element_id]
    definitions={}
    for policy,schema in ((None,SCHEMA),(OLD_PRECISE,PRECISE_SCHEMA),(STABLE,STABLE_SO3_SCHEMA)):
        refgeom=element.operator.reference
        captured=define_beam(BEAM,element.element_id,element.node_ids,refgeom.coordinates,refgeom.nodal_triads,
            element.section,inertia,quadrature=4,arithmetic_policy=policy)
        data=json.loads(captured.raw);assert data['schema']==schema
        restored=NativeBeamDefinition.from_bytes(captured.raw,expected_sha256=captured.sha256)
        fresh,fresh_inertia=restored.instantiate()
        assert NativeBeamDefinition.capture(fresh,fresh_inertia).raw==captured.raw
        definitions[str(policy)]=captured.raw
    assert len(set(definitions.values()))==3
    # Bind old capture-byte comparisons to a separately produced old-artifact input
    # when supplied by the claim-bearing lead harness; normal CI remains portable.
    baseline=os.environ.get('NATIVE_BENDING_LEGACY_DEFINITIONS')
    if baseline:
        expected=json.loads(Path(baseline).read_text(encoding='utf-8'))
        for policy in (None,OLD_PRECISE):assert definitions[str(policy)].decode('ascii')==expected[str(policy)]
    _,prefix,mode=run(0)
    # Same physical geometry/settings with the old default work interpretation
    # must refuse the new explicitly bound arithmetic checkpoint.
    from anyfesolver.ge_beam3_native_connected_dynamics import create_analysis,SELECTOR
    old_definitions=[]
    for held in owner.elements:
        r=held.operator.reference
        old_definitions.append(define_beam(BEAM,held.element_id,held.node_ids,r.coordinates,r.nodal_triads,
            held.section,owner.analysis._inertias[held.element_id],quadrature=4))
    old_owner=create_analysis(SELECTOR,tuple(old_definitions),tuple(owner.analysis.model.boundary_conditions))
    before_state=old_owner.current
    assert before_state is None
    with pytest.raises(ValueError,match='contract|identity|analysis|configuration'):
        solve_transient(old_owner,config(old_owner,mode,0),checkpoint=prefix.checkpoint,
            expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert old_owner.current is before_state
