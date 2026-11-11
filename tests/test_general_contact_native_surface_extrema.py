"""Gap checks independent of contact quadrature and complete patch admission."""
from dataclasses import replace,asdict
import json,os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.native_surface_extrema import straight_patch_plane_extrema,straight_patch_s3_gap
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
from test_general_contact_native_surface_s3_friction import _case
from test_general_contact_native_surface_patch import _patch
from test_general_contact_native_retained_station import _reference


def test_whole_cylinder_patch_gap_detects_penetration_between_samples():
    _,pair,pose,_=_case()
    rule=NativeRoundSurfaceRule(7,pair.source.reference,.03,xi_range=(-.9,-.1),angle_range=(4.1,5.3),axial_order=2,angular_order=2)
    gap=straight_patch_s3_gap(rule,pair.target,pose)
    assert gap.minimum==pytest.approx(-.01,abs=1e-15)
    assert gap.minimum_location[2]==pytest.approx(1.5*np.pi,abs=1e-15)
    sample_min=min(s.station.evaluate(pose.positions,pose.position_low,pose.cell_rotations).surface_position[2]+.04 for s in rule.samples)
    assert gap.minimum<sample_min-1e-3


def test_tilted_sheared_patch_extremum_against_independent_station_optimizer():
    _,pair,pose,_=_case(); rule=NativeRoundSurfaceRule(7,pair.source.reference,.03,
        xi_range=(-.9,-.1),angle_range=(2.,5.7),axial_order=2,angular_order=3)
    nodes=pose.positions.copy();nodes[1]+=(.02,.04,-.03)
    cells=pose.cell_rotations.copy();cells[0]=Rotation.from_rotvec((.2,-.1,.3)).as_matrix()
    pose=replace(pose,positions=nodes,cell_rotations=cells)
    normal=np.array((.1,.2,1.));normal/=np.linalg.norm(normal);origin=np.array((.2,-.3,.1))
    result=straight_patch_plane_extrema(rule,pose,origin,normal)
    def value(q):
        s=NativeRetainedRoundStation(7,pair.source.reference,float(q[0]),.03,float(q[1]))
        return float(normal@(s.evaluate(pose.positions,pose.position_low,pose.cell_rotations).surface_position-origin))
    oracle=differential_evolution(value,[(-.9,-.1),(2.,5.7)],seed=15,tol=1e-11,polish=True)
    assert oracle.success
    assert result.minimum==pytest.approx(oracle.fun,abs=1e-10)


def test_continuous_patch_rejects_curved_reference_and_target_rim():
    _,pair,pose,_=_case()
    curved=_reference(True);rule=NativeRoundSurfaceRule(7,curved,.03,axial_order=1,angular_order=2)
    with pytest.raises(ValueError,match='straight untwisted'):
        straight_patch_plane_extrema(rule,replace(pose,positions=curved.coordinates),np.zeros(3),np.array((0.,0.,1.)))
    rule=NativeRoundSurfaceRule(7,pair.source.reference,.03,xi_range=(-.1,.9),angle_range=(4.1,5.3),axial_order=1,angular_order=1)
    with pytest.raises(ValueError,match='continuous patch target ownership'):
        straight_patch_s3_gap(rule,pair.target,pose)


def test_initial_penetration_between_open_samples_is_rejected():
    _,sampled=_patch(radius=.0511)
    # Four open quadrature points alone produce no initial force.
    NativeS3MaterialTransaction(sampled)
    _,continuous=_patch(radius=.0511,gap_check='straight_patch')
    with pytest.raises(ValueError,match='initial continuous'):
        NativeS3MaterialTransaction(continuous)


def test_continuous_gap_selection_binds_restart():
    from hashlib import sha256
    _,sampled=_patch()
    capsule=NativeS3MaterialTransaction(sampled).checkpoint()
    _,continuous=_patch(gap_check='straight_patch')
    with pytest.raises(ValueError,match='definition/settings'):
        NativeS3MaterialTransaction.restore(continuous,capsule,sha256(capsule).hexdigest())


@pytest.mark.skipif('CONTACT_CONTINUOUS_GAP_OUTPUT' not in os.environ,reason='bounded gap screen needs output destination')
def test_coupled_eight_by_eight_independent_gap_screen():
    owner,bridge=_patch(activation_policy='transverse_events',axial_order=8,angular_order=8)
    transaction=NativeS3MaterialTransaction(bridge);accepted=owner.solve_material_increment(transaction,1.)
    response=transaction.propose(np.zeros(42),np.zeros(18),1.).assembly
    config=response.material.configuration
    gap=straight_patch_s3_gap(bridge.pair.rule,bridge.pair.target,config)
    path=Path(os.environ['CONTACT_CONTINUOUS_GAP_OUTPUT']);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(sampled_gap=response.material.contribution.normal_gap_m,
        continuous_gap=asdict(gap),owner_gap_tolerance=owner.gap_tolerance,
        configuration={k:v.tolist() for k,v in vars(config).items()},
        checkpoint=transaction.checkpoint().decode('ascii')),indent=2))
    assert gap.lower_bound<=response.material.contribution.normal_gap_m


@pytest.mark.skipif('CONTACT_ENFORCEMENT_OUTPUT' not in os.environ,reason='bounded enforcement screen needs output destination')
def test_eight_by_eight_stronger_penalty_with_continuous_gap_gate():
    owner,bridge=_patch(activation_policy='transverse_events',axial_order=8,angular_order=8,
        gap_check='straight_patch',normal_stiffness=1e5)
    transaction=NativeS3MaterialTransaction(bridge)
    accepted=owner.solve_material_increment(transaction,1.)
    response=transaction.propose(np.zeros(42),np.zeros(18),1.).assembly
    result=response.material.contribution
    record=dict(normal_stiffness=1e5,normal_force=float(-sum(result.residual[[2,8,14]])),
        tip_z=float(accepted.mechanical.positions[2,2]+accepted.mechanical.position_low[2,2]),
        continuous_gap=asdict(result.continuous_gap),
        sample_gap=min(p.normal_gap_m for p in result.points),equilibrium=list(response.metrics),
        active=sum(p.normal_pressure_Pa>0 for p in result.points),
        configuration={k:v.tolist() for k,v in vars(response.material.configuration).items()},
        checkpoint=transaction.checkpoint().decode('ascii'))
    path=Path(os.environ['CONTACT_ENFORCEMENT_OUTPUT']);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2))
    assert result.normal_gap_m==result.continuous_gap.lower_bound
    assert result.normal_gap_m>=-owner.gap_tolerance


@pytest.mark.skipif('CONTACT_FROZEN_FORCE_OUTPUT' not in os.environ,reason='saved-pose comparison needs explicit files')
def test_frozen_pose_continuous_normal_force_discriminator():
    """Analytic axial pressure integral, independent adaptive angular control."""
    from scipy.integrate import quad
    records=[]
    for file,stiffness in ((os.environ['CONTACT_BASELINE_POSE'],1e4),(os.environ['CONTACT_STIFFER_POSE'],1e5)):
        data=json.loads(Path(file).read_text())
        config=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in data['configuration'].items()})
        _,bridge=_patch(activation_policy='transverse_events',axial_order=8,angular_order=8,normal_stiffness=stiffness)
        rule=bridge.pair.rule; target=bridge.pair.target
        pose=target.physical_pose(config.shell_displacement)
        normal=np.asarray(pose.material_frame)[:,2]; origin=np.asarray(pose.position)
        reference=rule.samples[0].station.reference
        lo,hi=rule.definition['xi_range'];alo,ahi=rule.definition['angle_range'];radius=rule.definition['radius']
        assert 0<lo<hi<1  # This frozen fixture occupies exactly the right cell.
        positions=config.positions+config.position_low
        slope=float(normal@(positions[2]-positions[1]))
        frame=config.cell_rotations[1]@reference.nodal_triads[1]
        a,b=radius*float(normal@frame[:,1]),radius*float(normal@frame[:,2])
        offset=float(normal@(positions[1]-origin))
        def axial(alpha):
            base=offset+a*np.cos(alpha)+b*np.sin(alpha)
            left,right=base+slope*lo,base+slope*hi
            if left>=0 and right>=0:return 0.
            if left<=0 and right<=0:return -.5*(left+right)*(hi-lo)
            return .5*min(left,right)**2/abs(slope)
        breaks=[];amplitude=np.hypot(a,b);phase=np.arctan2(b,a)
        for xi in (lo,hi):
            ratio=-(offset+slope*xi)/amplitude
            if abs(ratio)<=1:
                angle=np.arccos(ratio)
                for sign in (-1,1):
                    for turn in (-1,0,1):
                        root=phase+sign*angle+turn*2*np.pi
                        if alo<root<ahi:breaks.append(root)
        integral,error=quad(axial,alo,ahi,points=sorted(breaks),epsabs=1e-14,epsrel=1e-10,limit=100)
        scale=stiffness*radius*np.linalg.norm(reference.coordinates[2]-reference.coordinates[1])
        discrete=sum(s.reference_weight_m2*stiffness*max(0.,-float(normal@(
            s.station.evaluate(config.positions,config.position_low,config.cell_rotations).surface_position-origin)))
            for s in rule.samples)
        records.append(dict(stiffness=stiffness,continuous_normal_force=scale*integral,
            estimated_integration_error=scale*error,quadrature_normal_force=discrete,
            relative_quadrature_error=abs(discrete-scale*integral)/(scale*integral)))
        assert error<1e-10*integral
    path=Path(os.environ['CONTACT_FROZEN_FORCE_OUTPUT']);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(records,indent=2))
