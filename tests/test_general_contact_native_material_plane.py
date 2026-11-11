"""Independent physical-section plane work and retained-owner integration."""
from hashlib import sha256
from math import cos,pi,sin
import json
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces,Program
from anyfesolver.general_contact.native_material_plane import NativeRoundMaterialPlane,PROFILE
from anyfesolver.general_contact.native_retained_static import NativeRetainedRoundPlaneAnalysis
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from test_general_contact_native_retained_station import _reference
from test_ge_beam3_native_generalized_modal import make


def _settings():return StraightNormalIntegrationSettings(1e-12,1e-10,1e-9,64,2.)


def _kernel(reference=None,element_id=7,**kwargs):
    reference=_reference(False) if reference is None else reference
    rule=NativeRoundSurfaceRule(element_id,reference,.03,xi_range=(-.9,-.1),
        angle_range=(4.1,5.3),axial_order=1,angular_order=1)
    return NativeRoundMaterialPlane(rule,plane_origin_m=(0.,0.,-.04),plane_normal=(0.,0.,1.),
        normal_stiffness_Pa_per_m=1000.,integration_settings=_settings(),**kwargs)


def _independent(kernel,positions,rotations,components=range(7)):
    """Direct surface quadrature; no production gap, jet or integration helper."""
    frame=rotations[kernel.cell]@kernel.reference.nodal_triads[0]
    lo,hi=kernel.rule.definition['xi_range'];lo+=1.;hi+=1.
    alo,ahi=kernel.rule.definition['angle_range']
    measure=kernel.radius*np.linalg.norm(kernel.reference.coordinates[1]-kernel.reference.coordinates[0])
    def point_at(t,angle):
        return (1-t)*positions[0]+t*positions[1]+kernel.radius*(frame[:,1]*cos(angle)+frame[:,2]*sin(angle))
    def gap_at(t,angle):return float((point_at(t,angle)-kernel.origin)@kernel.normal)
    def values(t,angle):
        point=point_at(t,angle)
        gap=(point-kernel.origin)@kernel.normal
        pressure=-kernel.stiffness*min(float(gap),0.)
        force=pressure*kernel.normal
        return np.r_[.5*kernel.stiffness*min(float(gap),0.)**2,force,np.cross(point,force)]*measure
    # Independent direct quadrature splits its affine axial zero. Three Gauss
    # points integrate the resulting degree-two physical work fields exactly.
    nodes,weights=np.polynomial.legendre.leggauss(3)
    def angle_integral(angle,index):
        first,last=gap_at(lo,angle),gap_at(hi,angle)
        left,right=lo,hi
        if first>=0. and last>=0.:return 0.
        if first*last<0.:
            root=lo-(hi-lo)*first/(last-first)
            if first<0.:right=root
            else:left=root
        midpoint=(left+right)/2;half=(right-left)/2
        return half*sum(weight*values(midpoint+half*node,angle)[index] for node,weight in zip(nodes,weights))
    # Bracket independently evaluated endpoint-gap zeros and extrema. This
    # reference is restricted to these short smooth arcs, not a search oracle.
    cuts={alo,ahi}; grid=np.linspace(alo,ahi,65)
    for t in (lo,hi):
        derivative=lambda angle:float(kernel.normal@(kernel.radius*(-frame[:,1]*sin(angle)+frame[:,2]*cos(angle))))
        extrema=[alo,ahi]
        for left,right in zip(grid[:-1],grid[1:]):
            if derivative(left)*derivative(right)<0.:
                extrema.append(brentq(derivative,left,right,xtol=1e-14))
        extrema.sort();cuts.update(extrema)
        for left,right in zip(extrema[:-1],extrema[1:]):
            if gap_at(t,left)*gap_at(t,right)<0.:
                cuts.add(brentq(lambda angle:gap_at(t,angle),left,right,xtol=1e-14))
    answers=[]
    for index in components:
        # Energy is differenced over 2e-6, so its absolute reference budget
        # bounds work uncertainty below the 1e-11 check. Force/moment are
        # compared directly at 2e-10; their 2e-13 reference budget is separate.
        absolute=2e-18 if index==0 else 2e-13
        value,error=quad(lambda angle:angle_integral(angle,index),alo,ahi,
            points=sorted(cuts)[1:-1],epsabs=absolute,epsrel=2e-13,limit=100)
        assert error<=absolute+2e-13*abs(value), (index,value,error)
        answers.append(value)
    return np.array(answers)


@pytest.mark.parametrize('partial',(False,True))
def test_independent_area_energy_wrench_and_complete_spatial_tangent(partial):
    kernel=_kernel();positions=kernel.reference.coordinates.copy();positions[:,2]-=.02
    if partial:positions[0,2]=-.01;positions[1,2]=-.025
    rotations=np.tile(np.eye(3),(2,1,1));rotations[0]=Rotation.from_rotvec((.02,-.03,.01)).as_matrix()
    low=np.zeros((3,3));result=kernel.evaluate(positions,low,rotations)
    oracle=_independent(kernel,positions,rotations)
    assert_allclose(result.potential_j,oracle[0],rtol=2e-8,atol=2e-12)
    assert_allclose(result.beam_force_n,oracle[1:4],rtol=2e-8,atol=2e-10)
    assert_allclose(result.beam_moment_nm,oracle[4:],rtol=2e-8,atol=2e-10)
    assert result.minimum_gap_m<=result.minimum_gap_estimate_m<0.
    assert np.linalg.norm(result.residual[18:21])>1e-6
    assert not np.any(result.residual[24:]) and not np.any(result.tangent[24:])
    direction=np.zeros(42);direction[:3]=(.003,.002,-.004);direction[6:9]=(-.002,.004,.003)
    direction[18:21]=(.005,-.003,.002)
    for step in (1e-5,2e-6):
        def at(sign):
            moved=positions.copy();moved[0]+=sign*step*direction[:3];moved[1]+=sign*step*direction[6:9]
            changed=rotations.copy();changed[0]=Rotation.from_rotvec(sign*step*direction[18:21]).as_matrix()@rotations[0]
            return kernel.evaluate(moved,low,changed)
        plus,minus=at(1),at(-1)
        derivative=(plus.residual-minus.residual)/(2*step)
        assert np.linalg.norm(result.tangent@direction-derivative)/np.linalg.norm(derivative)<1e-5
        # Independent work scalar: all source translation and section columns.
        moved=positions.copy();moved[0]+=step*direction[:3];moved[1]+=step*direction[6:9]
        changed=rotations.copy();changed[0]=Rotation.from_rotvec(step*direction[18:21]).as_matrix()@rotations[0]
        moved2=positions.copy();moved2[0]-=step*direction[:3];moved2[1]-=step*direction[6:9]
        changed2=rotations.copy();changed2[0]=Rotation.from_rotvec(-step*direction[18:21]).as_matrix()@rotations[0]
        work=(_independent(kernel,moved,changed,(0,))[0]-_independent(kernel,moved2,changed2,(0,))[0])/(2*step)
        assert_allclose(result.residual@direction,work,rtol=1e-5,atol=1e-11)


def test_material_patch_moves_at_fixed_external_nodes_and_rejects_wrong_geometry():
    kernel=_kernel();positions=kernel.reference.coordinates.copy();positions[:,2]-=.02
    rotations=np.tile(np.eye(3),(2,1,1));first=kernel.evaluate(positions,np.zeros((3,3)),rotations)
    rotations[0]=Rotation.from_rotvec((.08,-.02,0.)).as_matrix()
    second=kernel.evaluate(positions,np.zeros((3,3)),rotations)
    assert abs(second.potential_j-first.potential_j)>1e-7
    with pytest.raises(ValueError,match='exact straight'):_kernel(_reference(True))
    boundary=NativeRoundSurfaceRule(7,_reference(False),.03,xi_range=(-1.,-.1),axial_order=1,angular_order=1)
    with pytest.raises(ValueError,match='strictly interior'):
        NativeRoundMaterialPlane(boundary,plane_origin_m=(0,0,0),plane_normal=(0,0,1),
            normal_stiffness_Pa_per_m=1000.,integration_settings=_settings())
    kernel.rule.samples[0].station.radius=.031
    with pytest.raises(ValueError,match='changed'):kernel.evaluate(positions,np.zeros((3,3)),rotations)


def test_shared_s3_normal_route_preserves_both_body_work_and_tangent():
    from anyfesolver.general_contact.native_surface_normal import integrate_straight_normal
    from test_general_contact_native_surface_s3_friction import _case,_moved
    _,pair,pose,_=_case();kernel=_kernel()
    plane=kernel.evaluate(pose.positions,pose.position_low,pose.cell_rotations)
    original=integrate_straight_normal(kernel.rule,pair.target,pose,1000.,_settings())
    assert_allclose(original.normal_storage_j,plane.potential_j,rtol=0,atol=1e-16)
    assert_allclose(original.residual[:42],plane.residual,rtol=0,atol=1e-14)
    assert_allclose(original.tangent[:42,:42],plane.tangent,rtol=0,atol=1e-12)
    total=sum((original.residual[6*i:6*i+3]+original.residual[42+6*i:45+6*i]
               for i in range(3)),start=np.zeros(3))
    assert_allclose(total,0.,rtol=0,atol=1e-14)
    assert np.linalg.norm(original.tangent[18:24,42:])>0.
    direction=np.random.default_rng(753).normal(size=60)*.001
    for step in (1e-5,2e-6):
        plus=integrate_straight_normal(kernel.rule,pair.target,_moved(pose,direction,step),1000.,_settings())
        minus=integrate_straight_normal(kernel.rule,pair.target,_moved(pose,direction,-step),1000.,_settings())
        derivative=(plus.residual-minus.residual)/(2*step)
        assert np.linalg.norm(original.tangent@direction-derivative)/np.linalg.norm(derivative)<1e-5


def _analysis(targets=(.5,1.),normal_stiffness=1e6,*,radius=.05,
              angle_range=None,plane_origin=(0.,0.,-.0501),equilibrium_tolerance=1e-10):
    model,_,inertias=make(False,clamped=True,coupled=False)
    element=model.mesh.elements[1];definition=NativeBeamDefinition.capture(element,inertias[1])
    program=Program(targets,DistributedPattern(LinePattern(()),()),
        NodalDeadForces(((3,0.,0.,-.005),)),max_iterations=24)
    rule=NativeRoundSurfaceRule(1,element.operator.reference,radius,xi_range=(.1,.9),
        angle_range=(pi-.3,pi+.3) if angle_range is None else angle_range,axial_order=1,angular_order=1)
    kernel=NativeRoundMaterialPlane(rule,plane_origin_m=plane_origin,plane_normal=(0.,0.,1.),
        normal_stiffness_Pa_per_m=normal_stiffness,integration_settings=_settings())
    return NativeRetainedRoundPlaneAnalysis((definition,),tuple(model.boundary_conditions),program,kernel,
        gap_tolerance_m=1e-3*radius,equilibrium_tolerance=equilibrium_tolerance,maximum_swept_intervals=128)


def test_existing_retained_owner_coupled_response_and_split_restart():
    owner=_analysis();run=owner.solve()
    assert run.status=='completed',run.failure
    assert run.profile==PROFILE and run.accepted.contact.beam_force_n[2]>0.
    assert run.accepted.contact.minimum_gap_m>=-owner.gap_tolerance
    r,j,metrics,_,contact=owner._assemble(run.accepted.mechanical,1.,run.accepted.histories)
    assert max(metrics)<1e-10 and np.linalg.norm(r[owner.context.free])<1e-10
    force_scale=.005;moment_scale=.01
    load=np.array((0.,0.,-.005));state=run.accepted.mechanical
    balance=r[:3]+load+contact.beam_force_n
    wrench=r[3:6]+np.cross(state.positions[0],r[:3])+np.cross(state.positions[2],load)+contact.beam_moment_nm
    assert np.linalg.norm(balance)/force_scale<=1e-6
    assert np.linalg.norm(wrench)/moment_scale<=1e-6
    assert np.linalg.norm(contact.residual[18:24])>0.
    split=_analysis().solve(stop_after=1);assert split.status=='paused',split.failure
    resumed=_analysis().solve(checkpoint=split.checkpoint,expected_sha256=sha256(split.checkpoint).hexdigest())
    assert resumed.status=='completed',resumed.failure
    assert resumed.checkpoint==run.checkpoint
    changed=_analysis();changed.contact.stiffness*=2
    with pytest.raises(ValueError,match='changed'):
        changed.solve(checkpoint=split.checkpoint,expected_sha256=sha256(split.checkpoint).hexdigest())
    record=os.environ.get('CONTACT_MATERIAL_PLANE_RECORD')
    if record:Path(record).write_text(json.dumps(dict(profile=run.profile,
        force_n=contact.beam_force_n.tolist(),minimum_gap_m=contact.minimum_gap_m,
        storage_J=contact.potential_j,equilibrium_metrics=metrics,
        normalized_force_balance=float(np.linalg.norm(balance)/force_scale),
        normalized_moment_balance=float(np.linalg.norm(wrench)/moment_scale),
        internal_rotation_residual_norm=float(np.linalg.norm(contact.residual[18:24])),
        checkpoint_sha256=run.checkpoint_sha256),indent=2),encoding='utf8')


def test_swept_material_radius_detects_hidden_rotational_gap_pulse():
    owner=_analysis(radius=.03,angle_range=(pi-.31,pi-.29),plane_origin=(0.,0.,-.0299))
    origin=owner.initial.mechanical;increment=np.zeros(owner.context.count)
    increment[owner.context.nodal_count+3]=1.2 # second cell, physical spin
    end=owner.context.advance(origin,increment)
    nodes=owner.context.nodes[0]
    gap=lambda state:owner.contact.minimum_gap(state.positions[nodes],state.position_low[nodes],state.cell_rotations[0])
    first,last=gap(origin),gap(end)
    assert first>0. and last>0. and gap(owner.context.advance(origin,.5*increment))>0.
    assert gap(owner.context.advance(origin,.25*increment))<-owner.gap_tolerance
    # Endpoint/midpoint sampling misses this; the radius-dependent path bound
    # must subdivide the actual exponential stage and reject the hidden pulse.
    assert not owner._swept_admissible(origin,increment,end,first,last)


def test_unresolved_angular_grazing_fails_before_quadrature(monkeypatch):
    from fractions import Fraction
    from math import hypot
    import anyfesolver.general_contact.native_material_plane as module
    owner=_analysis();kernel=NativeRoundMaterialPlane(owner.contact.rule,
        plane_origin_m=(0.,0.,-.05),plane_normal=(0.,0.,1.),
        normal_stiffness_Pa_per_m=1e7,integration_settings=_settings())
    # At an angular extremum the rounded offset/amplitude ratio can be exactly
    # one. A vanishing or sub-roundoff active pocket must not be called open
    # solely because both quadrature rules sampled zero pressure.
    rotations=np.tile(np.eye(3),(2,1,1))
    rotations[1]=Rotation.from_rotvec((.1,0.,0.)).as_matrix()
    frame=rotations[1]@kernel.reference.nodal_triads[0]
    a=kernel.radius*float(frame[2,1]);b=kernel.radius*float(frame[2,2])
    assert .05/hypot(a,b)==1.
    # Exact arithmetic on the declared binary coefficients shows a negative
    # minimum, despite the rounded amplitude ratio collapsing both roots.
    assert Fraction(a)**2+Fraction(b)**2>Fraction(.05)**2
    def forbidden(*args):raise AssertionError('quadrature reached unresolved front')
    monkeypatch.setattr(module,'_integrate_material_normal_data',forbidden)
    with pytest.raises(ValueError,match='endpoint/extremum gap unresolved'):
        kernel.evaluate(kernel.reference.coordinates,np.zeros((3,3)),rotations)


def test_explicit_area_penalty_sensitivity_reduces_independent_gap():
    rows=[]
    for stiffness in (1e6,1e7):
        owner=_analysis(normal_stiffness=stiffness);run=owner.solve()
        if run.status!='completed' and os.environ.get('CONTACT_MATERIAL_PENALTY_RECORD'):
            from anyfesolver._ge_beam3_p5_seeded.core import canonical
            Path(os.environ['CONTACT_MATERIAL_PENALTY_RECORD']).write_bytes(canonical(dict(
                scope='failed declared penalty sensitivity; no acceptance',prior_rows=rows,
                status=run.status,failure=run.failure,normal_stiffness_Pa_per_m=stiffness,
                diagnostics=run.diagnostics)))
        assert run.status=='completed',run.failure
        contact=run.accepted.contact
        assert contact.minimum_gap_m>=-owner.gap_tolerance
        rows.append(dict(normal_stiffness_Pa_per_m=stiffness,minimum_gap_m=contact.minimum_gap_m,
                         force_n=contact.beam_force_n.tolist(),storage_J=contact.potential_j))
    assert abs(rows[1]['minimum_gap_m'])<abs(rows[0]['minimum_gap_m'])
    record=os.environ.get('CONTACT_MATERIAL_PENALTY_RECORD')
    if record:Path(record).write_text(json.dumps(dict(scope='declared penalty sensitivity, not continuum acceptance',
        rows=rows,force_change_relative=abs(rows[1]['force_n'][2]-rows[0]['force_n'][2])/abs(rows[1]['force_n'][2])),
        indent=2),encoding='utf8')


def test_common_rigid_transform_preserves_material_surface_work():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    original=_kernel();positions=original.reference.coordinates.copy();positions[:,2]-=.02
    rotations=np.tile(np.eye(3),(2,1,1));rotations[0]=Rotation.from_rotvec((.02,-.03,.01)).as_matrix()
    value=original.evaluate(positions,np.zeros((3,3)),rotations)
    Q=Rotation.from_rotvec((.3,-.2,.1)).as_matrix()
    reference=CenteredCurvedBeam3ReferenceGeometry(original.reference.coordinates@Q.T,
        np.array([Q@frame for frame in original.reference.nodal_triads]))
    rule=NativeRoundSurfaceRule(7,reference,.03,xi_range=(-.9,-.1),angle_range=(4.1,5.3),axial_order=1,angular_order=1)
    changed=NativeRoundMaterialPlane(rule,plane_origin_m=Q@original.origin,plane_normal=Q@original.normal,
        normal_stiffness_Pa_per_m=1000.,integration_settings=_settings())
    transformed=changed.evaluate(positions@Q.T,np.zeros((3,3)),np.array([Q@U@Q.T for U in rotations]))
    T=np.eye(42)
    for start in range(0,24,3):T[start:start+3,start:start+3]=Q
    assert_allclose(transformed.potential_j,value.potential_j,rtol=1e-13,atol=1e-16)
    assert_allclose(transformed.residual,T@value.residual,rtol=1e-11,atol=1e-13)
    assert_allclose(transformed.tangent,T@value.tangent@T.T,rtol=1e-10,atol=1e-11)
    assert_allclose(transformed.beam_moment_nm,Q@value.beam_moment_nm,rtol=1e-11,atol=1e-13)


@pytest.mark.parametrize('cancel',(False,True))
def test_precommit_exception_and_cancellation_preserve_accepted_material_origin(cancel):
    from anyfesolver.control import CancellationToken
    first=_analysis().solve(stop_after=1);assert first.status=='paused',first.failure
    owner=_analysis();record=owner._record;token=CancellationToken()
    def failing(target,*args):
        staged=record(target,*args)
        if target==2:
            if cancel:token.cancel('injected after staging')
            else:raise RuntimeError('injected after staging')
        return staged
    owner._record=failing
    result=owner.solve(checkpoint=first.checkpoint,expected_sha256=first.checkpoint_sha256,
                       cancellation_token=token)
    assert result.status==('cancelled' if cancel else 'failed')
    assert result.accepted.completed_targets==1 and result.checkpoint==first.checkpoint
    assert_allclose(result.accepted.mechanical.cell_rotations,first.accepted.mechanical.cell_rotations,rtol=0,atol=0)


def test_runnable_native_material_plane_example(capsys):
    import runpy
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/general_contact_native_material_plane.py'))
    namespace['main']()
    output=json.loads(capsys.readouterr().out)
    assert output['profile']==PROFILE and output['status']=='completed'
    assert output['qualification'].startswith('implemented/tested development')
    assert output['minimum_gap_lower_bound_m']>=-5e-5
    assert output['completed_targets']==2 and output['contact_force_n'][2]>0.
