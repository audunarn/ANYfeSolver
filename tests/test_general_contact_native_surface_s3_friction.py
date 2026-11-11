"""Native retained surface/S3 point work, before analysis admission."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.friction_point import initial_friction_point
from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
from anyfesolver.general_contact.native_surface_s3_friction import (
    NativeS3MaterialConfiguration,NativeS3MaterialPointFriction,
)
from test_general_contact_native_retained_station import _reference


def _case():
    model=FEModel('native surface S3 point work')
    model.add_material('elastic',1e5,.3,density=1.)
    for node,xyz in enumerate(((-2.,-2.,-.05),(2.,-2.,-.05),(-2.,2.,-.05)),1):
        model.add_node(node,*xyz)
    model.add_element(1,create_shell_element(1,(1,2,3),'elastic',thickness=.02,
        formulation='e4-pl-s3-v2d',reference_normal=(0,0,1)))
    target=S3SkinMaterialPoint.from_model(model,2,1,'top',(.125,.375,.5))
    reference=_reference(False)
    station=NativeRetainedRoundStation(7,reference,-.5,.03,1.5*np.pi)
    pair=NativeS3MaterialPointFriction(station,target,reference_weight_m2=.01,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.)
    positions=reference.coordinates.copy();positions[:,2]-=.02
    pose=NativeS3MaterialConfiguration(positions,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)),np.zeros(18))
    history=initial_friction_point(target.physical_pose(pose.shell_displacement).material_frame)
    return model,pair,pose,history


def _moved(pose,direction,h):
    positions=pose.positions.copy();cells=pose.cell_rotations.copy()
    for node in range(3):positions[node]+=h*direction[6*node:6*node+3]
    for cell in range(2):
        cells[cell]=Rotation.from_rotvec(h*direction[18+3*cell:21+3*cell]).as_matrix()@cells[cell]
    return NativeS3MaterialConfiguration(positions,pose.position_low,cells,pose.shell_displacement+h*direction[42:])


@pytest.mark.parametrize('angle,branch',[(.01,'stick'),(.2,'slip')])
def test_native_surface_actual_gap_pressure_spin_and_both_body_tangent(angle,branch):
    _model,pair,old,history=_case()
    spin=np.zeros(60);spin[18]=1
    pose=_moved(old,spin,angle)
    value=pair.evaluate(old,pose,history)
    gap=-.01+.03*(1-np.cos(angle));pressure=-1000*gap
    assert value.normal_gap_m==pytest.approx(gap,abs=1e-14)
    assert value.normal_pressure_Pa==pytest.approx(pressure,rel=1e-12)
    assert value.trial.branch==branch
    # Internal residual is minus physical force. Both actual gap torque and
    # friction torque act on the retained cell rotation, not a nodal trace.
    traction=min(1000*.03*np.sin(angle),.3*pressure)
    expected=.01*(.03*np.cos(angle)*traction-.03*np.sin(angle)*pressure)
    assert value.residual[18]==pytest.approx(expected,rel=1e-11,abs=1e-14)
    assert value.residual[18]>0
    direction=np.random.default_rng(185).normal(size=60)
    tangent=value.spatial_tangent()
    for h in (1e-5,2e-6):
        plus=pair.evaluate(old,_moved(pose,direction,h),history)
        minus=pair.evaluate(old,_moved(pose,direction,-h),history)
        assert plus.trial.branch==minus.trial.branch==branch
        fd=(plus.residual-minus.residual)/(2*h)
        assert np.linalg.norm(tangent@direction-fd)/np.linalg.norm(fd)<=1e-5
    np.testing.assert_array_equal(value.residual[24:42],0.)
    np.testing.assert_array_equal(tangent[24:42,:],0.)
    assert history==initial_friction_point(np.eye(3))


def _wrench(pair,pose,residual):
    force=np.zeros(3);moment=np.zeros(3)
    for node in range(3):
        f=residual[6*node:6*node+3];force+=f
        moment+=np.cross(pose.positions[node]+pose.position_low[node],f)+residual[6*node+3:6*node+6]
    moment+=residual[18:21]+residual[21:24]
    reference=pair.target.adapter.reference_positions-np.array((0,0,.01))
    for node in range(3):
        f=residual[42+6*node:45+6*node];force+=f
        point=reference[node]+pose.shell_displacement[6*node:6*node+3]
        moment+=np.cross(point,f)+residual[45+6*node:48+6*node]
    return force,moment


def test_native_surface_nonproportional_history_preserves_wrench_and_rigid_transport():
    _model,pair,origin,empty=_case()
    spin=np.zeros(60);spin[18]=1
    old=_moved(origin,spin,.01)
    history=pair.evaluate(origin,old,empty).trial.candidate_state
    direction=np.zeros(60)
    for node in range(3):direction[6*node]=1
    pose=_moved(old,direction,.0002)
    value=pair.evaluate(old,pose,history)
    assert value.trial.branch=='stick'
    for item in _wrench(pair,pose,value.residual):np.testing.assert_allclose(item,0.,atol=1e-12)
    rigid=Rotation.from_rotvec((.3,-.2,.1)).as_matrix();shift=np.array((.1,.2,-.1))
    corners=pair.target.adapter.current_positions(old.shell_displacement)
    u=np.zeros(18)
    for node in range(3):u[6*node:6*node+3]=rigid@corners[node]+shift-pair.target.adapter.reference_positions[node]
    transformed=NativeS3MaterialConfiguration(old.positions@rigid.T+shift,old.position_low@rigid.T,
        rigid@old.cell_rotations,u)
    rotated=pair.evaluate(old,transformed,history)
    baseline=pair.evaluate(old,old,history)
    np.testing.assert_allclose(rotated.trial.traction,rigid@np.asarray(baseline.trial.traction),atol=1e-12)
    assert rotated.trial.dissipated_increment_density==pytest.approx(0.,abs=1e-14)
    assert rotated.normal_pressure_Pa==pytest.approx(baseline.normal_pressure_Pa,rel=1e-12)
    for item in _wrench(pair,transformed,rotated.residual):np.testing.assert_allclose(item,0.,atol=2e-12)


@pytest.mark.parametrize('angle,branch',[(.02,'stick'),(.2,'slip')])
def test_native_surface_retained_history_tangent_with_changing_target_frame(angle,branch):
    _model,pair,origin,empty=_case()
    spin=np.zeros(60);spin[18]=1
    old=_moved(origin,spin,.01)
    history=pair.evaluate(origin,old,empty).trial.candidate_state
    assert np.linalg.norm(history.elastic_slip)>0
    pose=_moved(origin,spin,angle)
    positions=pose.positions.copy();positions[:,0]+=.0002
    shell=pose.shell_displacement.copy();shell[14]=.0004
    pose=replace(pose,positions=positions,shell_displacement=shell)
    value=pair.evaluate(old,pose,history)
    assert value.trial.branch==branch
    direction=np.random.default_rng(561).normal(size=60)
    for h in (1e-5,2e-6):
        plus=pair.evaluate(old,_moved(pose,direction,h),history)
        minus=pair.evaluate(old,_moved(pose,direction,-h),history)
        assert plus.trial.branch==minus.trial.branch==branch
        fd=(plus.residual-minus.residual)/(2*h)
        assert np.linalg.norm(value.spatial_tangent()@direction-fd)/np.linalg.norm(fd)<=1e-5
    assert history==pair.evaluate(origin,old,empty).trial.candidate_state


def test_native_surface_point_rejects_transfer_and_unresolved_first_touch():
    _model,pair,old,history=_case()
    direction=np.zeros(60)
    for node in range(3):direction[6*node]=1
    with pytest.raises(ValueError,match='interior target ownership'):
        pair.evaluate(old,_moved(old,direction,5.),history)
    opened=old.positions.copy();opened[:,2]+=.02
    opened=replace(old,positions=opened)
    with pytest.raises(ValueError,match='first-touch'):
        pair.evaluate(opened,_moved(old,direction,.001),history)
    value=pair.evaluate(old,opened,history)
    assert value.trial.branch=='open'
    np.testing.assert_array_equal(value.residual,0.)
    np.testing.assert_array_equal(value.chart_tangent,0.)
    assert history==initial_friction_point(np.eye(3))
    with pytest.raises(ValueError,match='partner history transfer'):
        pair.evaluate(old,old,replace(history,partner_key='different material pair'))
    pair.source.radius*=2
    with pytest.raises(ValueError,match='definition changed'):
        pair.evaluate(old,old,history)


def test_native_surface_point_resource_preflight():
    _model,pair,_old,_history=_case()
    with pytest.raises(MemoryError,match='coordinate cap'):
        NativeS3MaterialPointFriction(pair.source,pair.target,reference_weight_m2=.01,
            normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,
            tangential_stiffness_Pa_per_m=1000.,maximum_coupled_dofs=59)


@pytest.mark.parametrize('angle', [.01, .2])
def test_local_geometry_memo_and_exact_rotation_match_uncached_jet_oracle(monkeypatch, angle):
    import anyfesolver.general_contact.native_surface_s3_friction as module
    from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp, matvec
    _, pair, origin, history = _case()
    spin = np.zeros(60); spin[18] = 1.
    candidate = _moved(origin, spin, angle)
    actual = pair.evaluate(origin, candidate, history)
    def old_rotation(variations):
        delta = [Jet2.variable(0., 3+i, 15) for i in range(3)]
        return matvec(so3_exp(delta), [Jet2.constant(v,15) for v in variations.surface_arm])
    frame = module._skin_frame_jet.__wrapped__
    monkeypatch.setattr(module, '_rotated_arm_jet', old_rotation)
    monkeypatch.setattr(module, '_skin_frame_jet', frame)
    expected = pair.evaluate(origin, candidate, history)
    np.testing.assert_allclose(actual.residual, expected.residual, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(actual.chart_tangent, expected.chart_tangent, rtol=1e-13, atol=1e-13)
    assert actual.trial == expected.trial


def test_skin_frame_memo_is_bounded_readonly_and_invalid_pose_still_rejects():
    import anyfesolver.general_contact.native_surface_s3_friction as module
    _, pair, origin, history = _case()
    module._skin_frame_jet.cache_clear()
    for shift in range(40):
        direction=np.zeros(60); direction[42]=1.
        candidate=_moved(origin,direction,1e-7*shift)
        pair.evaluate(origin,candidate,history)
    assert module._skin_frame_jet.cache_info().currsize == 32
    corners=pair.target.adapter.current_positions(origin.shell_displacement)
    cached=module._skin_frame_jet(corners.tobytes())
    assert all(not j.gradient.flags.writeable and not j.hessian.flags.writeable
               for row in cached for j in row)
    u=origin.shell_displacement.copy(); u[6:9] = (-4.,0.,0.)
    invalid=NativeS3MaterialConfiguration(origin.positions,origin.position_low,origin.cell_rotations,u)
    with pytest.raises(ValueError,match='degenerate|inverted'):
        pair.evaluate(origin,invalid,history)



@pytest.mark.parametrize('angle', [.01,.2])
def test_batched_local_jets_match_scalar_jet2_work(monkeypatch, angle):
    import anyfesolver.general_contact.native_surface_s3_friction as module
    from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp, matvec, dot
    _,pair,origin,history=_case()
    spin=np.zeros(60); spin[18]=1.
    candidate=_moved(origin,spin,angle)
    actual=pair.evaluate(origin,candidate,history)
    def scalar(new,corners,barycentric,old_target,old,history,partner_key):
        delta=[Jet2.variable(0.,i,15) for i in range(15)]
        rotated=matvec(so3_exp(delta[3:6]),[Jet2.constant(v,15) for v in new.surface_arm])
        x=[delta[i]+rotated[i]+new.station.surface_position[i]-new.surface_arm[i] for i in range(3)]
        y=[[Jet2.constant(corners[a,i],15)+delta[6+3*a+i] for i in range(3)] for a in range(3)]
        first=[y[1][i]-y[0][i] for i in range(3)];second=[y[2][i]-y[0][i] for i in range(3)]
        t1=module._unit(first);n=module._unit(module._cross(first,second));t2=module._cross(n,t1)
        F=[[t1[i],t2[i],n[i]] for i in range(3)]
        point=[sum(barycentric[a]*y[a][i] for a in range(3)) for i in range(3)]
        separation=[x[i]-point[i] for i in range(3)]
        components=np.asarray(old_target.material_frame).T@(old.surface_position-np.asarray(old_target.position))
        transported=matvec(F,[Jet2.constant(v,15) for v in components])
        d=[separation[i]-transported[i] for i in range(3)]
        q=dot(n,d);d=[d[i]-n[i]*q for i in range(3)]
        retained=np.asarray(history.elastic_slip) if history.partner_key==partner_key else np.zeros(3)
        old_elastic=np.asarray(history.material_frame).T@retained
        transported=matvec(F,[Jet2.constant(v,15) for v in old_elastic])
        return dot(n,separation),[d[i]+transported[i] for i in range(3)]
    monkeypatch.setattr(module,'_local_contact_jets',scalar)
    expected=pair.evaluate(origin,candidate,history)
    np.testing.assert_allclose(actual.residual,expected.residual,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(actual.chart_tangent,expected.chart_tangent,rtol=1e-13,atol=1e-13)
