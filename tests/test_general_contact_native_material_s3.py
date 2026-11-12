"""Physical material area / deformable S3 work and retained-owner workflow."""
import json,os
from pathlib import Path
from hashlib import sha256
from math import cos,sin,pi
import numpy as np
import pytest
import runpy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from numpy.testing import assert_allclose
from anyfesolver import create_shell_element
from anyfesolver.boundary import BoundaryCondition,FixedSupport
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces,Program
from anyfesolver.general_contact.native_material_s3 import (
    NativeMaterialS3Patch,NativeRetainedMaterialS3Work,PROFILE)
from anyfesolver.general_contact.native_retained_s3_static import NativeRetainedRoundS3Analysis
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
from test_ge_beam3_native_generalized_modal import make


def test_runnable_deformable_material_patch_example(capsys):
    example=Path(__file__).resolve().parents[1]/'examples/general_contact_native_material_s3.py'
    runpy.run_path(str(example),run_name='__main__')
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='completed' and result['completed_targets']==5
    assert result['profile']==PROFILE and result['minimum_gap_lower_bound_m']>=-5e-5
    assert result['shell_displacement_norm_m']>0.
    assert_allclose(np.asarray(result['beam_force_n'])+result['shell_force_n'],0.,atol=1e-12)


def _two_analysis(**kwargs):
    example=Path(__file__).resolve().parents[1]/'examples/general_contact_native_material_s3.py'
    return runpy.run_path(str(example))['analysis'](beams=2,**kwargs)


def test_multiple_material_facets_complete_work_and_retained_response():
    from anyfesolver.general_contact.native_multi_s3_static import MATERIAL_MULTIPATCH_PROFILE
    owner=_two_analysis(facets=2,targets=(.25,.5,.75,1.));n=owner.context.count
    assert n==84 and owner.shell_stiffness.shape==(36,36) and len(owner.pairs)==3
    step=np.zeros(n)
    for i,nodes in enumerate(owner.context.nodes):
        for j,node in enumerate(nodes[1:]):
            step[6*node+2]=-.001-.0002*i-.0002*j
            if i==0:step[6*node+1]=-.0012-.0002*j
    pose=owner.context.advance(owner.initial.mechanical,step);u=np.zeros(36)
    value=owner._contact(pose,u)
    def energy(mechanical,displacement):
        return sum(_reference(p.work,mechanical.positions[owner.context.nodes[i]]+
            mechanical.position_low[owner.context.nodes[i]],mechanical.cell_rotations[i],displacement)[0]
            for p,i in zip(owner.pairs,owner.beam_indices))
    assert value.tangent.shape==(120,120) and value.profile==MATERIAL_MULTIPATCH_PROFILE
    assert_allclose(value.contact.potential_j,energy(pose,u),rtol=1e-10,atol=1e-13)
    assert_allclose(value.contact.beam_force_n+value.contact.shell_force_n,0.,atol=1e-11)
    assert_allclose(value.contact.beam_moment_nm+value.contact.shell_moment_nm,0.,atol=1e-11)
    d=np.random.default_rng(510).normal(size=n+36)*.003;d[list(owner.context.fixed)]=0.
    errors=[]
    for h in (1e-5,2e-6):
        states=[owner.context.advance(pose,sign*h*d[:n]) for sign in (1,-1)]
        rows=[owner._contact(s,u+sign*h*d[n:]) for sign,s in zip((1,-1),states)]
        fd=(rows[0].residual-rows[1].residual)/(2*h)
        error=float(np.linalg.norm(value.tangent@d-fd)/np.linalg.norm(fd));assert error<=1e-5;errors.append(error)
        virtual=(energy(states[0],u+h*d[n:])-energy(states[1],u-h*d[n:]))/(2*h)
        assert_allclose(value.residual@d,virtual,rtol=1e-5,atol=1e-10)
    run=owner.solve();assert run.status=='completed',run.failure
    assert run.profile==MATERIAL_MULTIPATCH_PROFILE
    a=run.accepted
    r,j,metrics,_,contact=owner._assemble(a.mechanical,a.shell_displacement,1.,a.native_histories)
    assert max(metrics)<1e-10 and all(row.contact.minimum_gap_m>=-p.gap_tolerance_m for p,row in contact.pairs)
    assert np.all(np.abs(a.shell_displacement[owner.shell_free])>0.)
    assert_allclose(contact.contact.beam_force_n+contact.contact.shell_force_n,0.,atol=1e-12)
    assert_allclose(contact.contact.beam_moment_nm+contact.contact.shell_moment_nm,0.,atol=1e-12)
    shell_r=r[n:].reshape(6,6)
    mids=np.vstack([pair.work.shell.reference_positions-pair.work.shell.physical_offset*pair.work.shell.director
        for pair in (owner.pairs[0],owner.pairs[1])])+a.shell_displacement.reshape(6,6)[:,:3]
    force_balance=shell_r[:,:3].sum(axis=0)
    moment_balance=np.cross(mids,shell_r[:,:3]).sum(axis=0)+shell_r[:,3:].sum(axis=0)
    for i,nodes in enumerate(owner.context.nodes):
        root=nodes[0];force=np.array((0.,-.003 if i==0 else 0.,-.005 if i==0 else -.003))
        force_balance+=r[6*root:6*root+3]+force
        moment_balance+=(r[6*root+3:6*root+6]+np.cross(a.mechanical.positions[root],r[6*root:6*root+3])+
            np.cross(a.mechanical.positions[nodes[2]],force))
    normalized_force=float(np.linalg.norm(force_balance)/.009)
    normalized_moment=float(np.linalg.norm(moment_balance)/.018)
    assert normalized_force<=1e-6 and normalized_moment<=1e-6
    direction=np.random.default_rng(513).normal(size=n+36)*.001
    direction[list(owner.context.fixed)]=0.;direction[n+owner.shell_fixed]=0.
    owner_errors=[]
    for h in (1e-5,2e-6):
        states=[owner.context.advance(a.mechanical,sign*h*direction[:n]) for sign in (1,-1)]
        rows=[owner._assemble(s,a.shell_displacement+sign*h*direction[n:],1.,a.native_histories)[0]
            for sign,s in zip((1,-1),states)]
        fd=(rows[0]-rows[1])/(2*h)
        error=float(np.linalg.norm(j@direction-fd)/np.linalg.norm(fd))
        assert error<=1e-5;owner_errors.append(error)
    output=os.environ.get('CONTACT_MATERIAL_FACETS_S3_RECORD')
    if output:Path(output).write_text(json.dumps(dict(profile=run.profile,coordinates=120,
        metrics=metrics,contact_tangent_errors=errors,owner_tangent_errors=owner_errors,
        normalized_force_balance=normalized_force,normalized_moment_balance=normalized_moment,
        minimum_gap_m=contact.contact.minimum_gap_m,
        shell_displacement_norm=float(np.linalg.norm(a.shell_displacement)),
        pair_forces=[row.contact.beam_force_n.tolist() for _,row in contact.pairs],
        checkpoint_sha256=run.checkpoint_sha256),indent=2))


def test_one_native_two_material_facets_open_checkpoint_and_recontact():
    example=Path(__file__).resolve().parents[1]/'examples/general_contact_native_material_s3.py'
    maker=runpy.run_path(str(example))['analysis']
    owner=maker(beams=1,facets=2);run=owner.solve()
    assert run.status=='completed',run.failure
    assert owner.context.count==42 and owner.shell_stiffness.shape==(36,36)
    assert run.accepted.completed_targets==9
    records=json.loads(run.checkpoint)['records']
    assert records[3]['contact']['potential_j']>0.
    assert records[4]['contact']['potential_j']==0.
    assert records[8]['contact']['potential_j']>0.
    for record in (records[3],records[8]):
        assert len(record['contact']['pairs'])==2
        assert all(pair['potential_j']>0. for pair in record['contact']['pairs'])
    assert all(pair['minimum_gap_m']>0. and pair['potential_j']==0.
        for pair in records[4]['contact']['pairs'])
    opened=maker(beams=1,facets=2).solve(stop_after=5)
    assert opened.status=='paused' and opened.accepted.contact.contact.potential_j==0.
    continued=maker(beams=1,facets=2).solve(checkpoint=opened.checkpoint,
        expected_sha256=opened.checkpoint_sha256)
    assert continued.status=='completed' and continued.checkpoint==run.checkpoint


@pytest.mark.parametrize("uniform",(False,True))
def test_two_material_contact_independent_work_before_structural_globalization(uniform):
    owner=_two_analysis(targets=(.25,.5,.75,1.));n=owner.context.count
    step=np.zeros(n)
    for i,nodes in enumerate(owner.context.nodes):
        for j,node in enumerate(nodes[1:]):step[6*node+2]=-.001-.0002*i-(0. if uniform else .0002*j)
        step[owner.context.nodal_count+24*i+3]=.04+.01*i
    pose=owner.context.advance(owner.initial.mechanical,step);u=np.zeros(18)
    result=owner._contact(pose,u)
    def energy(mechanical,displacement):
        return sum(_reference(p.work,mechanical.positions[owner.context.nodes[i]]+
            mechanical.position_low[owner.context.nodes[i]],mechanical.cell_rotations[i],displacement)[0]
            for p,i in zip(owner.pairs,owner.beam_indices))
    assert_allclose(result.contact.potential_j,energy(pose,u),rtol=1e-10,atol=1e-14)
    d=np.random.default_rng(490).normal(size=n+18)*.003
    d[list(owner.context.fixed)]=0.
    checks=[]
    for h in (1e-5,2e-6):
        poses=[owner.context.advance(pose,sign*h*d[:n]) for sign in (1,-1)]
        values=[owner._contact(s,u+sign*h*d[n:]) for sign,s in zip((1,-1),poses)]
        fd=(values[0].residual-values[1].residual)/(2*h)
        error=float(np.linalg.norm(result.tangent@d-fd)/np.linalg.norm(fd))
        assert error<=1e-5
        independent=(energy(poses[0],u+h*d[n:])-energy(poses[1],u-h*d[n:]))/(2*h)
        assert_allclose(result.residual@d,independent,rtol=1e-5,atol=1e-10)
        checks.append(dict(h=h,directional_error=error,
            virtual_work_error=float(abs(result.residual@d-independent)),
            intervals=[value.contact.integration_intervals for value in values]))
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:Path(output+f'-uniform-{uniform}.json').write_text(json.dumps(dict(
        geometry='historical uniform pose' if uniform else 'nonuniform control',
        energy_error=float(abs(result.contact.potential_j-energy(pose,u))),
        checks=checks),indent=2))


def test_two_native_material_patches_shared_deformable_target():
    from anyfesolver.general_contact.native_multi_s3_static import MATERIAL_PROFILE
    owner=_two_analysis(targets=(.25,.5,.75,1.));run=owner.solve()
    assert run.status=='completed',run.failure
    assert run.profile==MATERIAL_PROFILE and owner.context.count==84
    a=run.accepted;n=owner.context.count
    residual,tangent,metrics,_,contact=owner._assemble(a.mechanical,a.shell_displacement,1.,a.native_histories)
    assert tangent.shape==(102,102) and max(metrics)<1e-10
    assert contact.profile==MATERIAL_PROFILE and contact.contact.active_intervals is None
    assert np.linalg.norm(a.shell_displacement)>0.
    assert_allclose(contact.contact.beam_force_n+contact.contact.shell_force_n,0.,atol=1e-12)
    assert_allclose(contact.contact.beam_moment_nm+contact.contact.shell_moment_nm,0.,atol=1e-12)
    shell_r=residual[n:].reshape(3,6)
    skin=owner.pairs[0].work.shell
    shell_mid=skin.reference_positions-skin.physical_offset*skin.director+a.shell_displacement.reshape(3,6)[:,:3]
    force_balance=shell_r[:,:3].sum(axis=0)
    moment_balance=np.cross(shell_mid,shell_r[:,:3]).sum(axis=0)+shell_r[:,3:].sum(axis=0)
    for i,nodes in enumerate(owner.context.nodes):
        root=nodes[0];force=np.array((0.,0.,-.005 if i==0 else -.003))
        force_balance+=residual[6*root:6*root+3]+force
        moment_balance+=(residual[6*root+3:6*root+6]+
            np.cross(a.mechanical.positions[root],residual[6*root:6*root+3])+
            np.cross(a.mechanical.positions[nodes[2]],force))
    normalized_force=float(np.linalg.norm(force_balance)/.008)
    normalized_moment=float(np.linalg.norm(moment_balance)/.016)
    assert normalized_force<=1e-6 and normalized_moment<=1e-6
    expected=0.
    for (pair,row),i in zip(contact.pairs,owner.beam_indices):
        nodes=owner.context.nodes[i]
        expected+=_reference(pair.work,a.mechanical.positions[nodes]+a.mechanical.position_low[nodes],
                             a.mechanical.cell_rotations[i],a.shell_displacement)[0]
        assert row.contact.minimum_gap_m>=-pair.gap_tolerance_m
        assert row.tangent[:42,42:].nnz and row.tangent[42:,:42].nnz
        assert row.contact.beam_force_n[2]>0.
    assert_allclose(contact.contact.potential_j,expected,rtol=1e-9,atol=1e-15)
    # Shared-target compliance couples the native solves; neither beam was
    # independently condensed before adding contact.
    d=np.random.default_rng(486).normal(size=102)*.001
    d[list(owner.context.fixed)]=0.;d[n+owner.shell_fixed]=0.
    for h in (1e-5,2e-6):
        states=[owner.context.advance(a.mechanical,sign*h*d[:n]) for sign in (1,-1)]
        rows=[owner._assemble(s,a.shell_displacement+sign*h*d[n:],1.,a.native_histories)[0]
              for sign,s in zip((1,-1),states)]
        fd=(rows[0]-rows[1])/(2*h)
        assert np.linalg.norm(tangent@d-fd)/np.linalg.norm(fd)<=1e-5
    split=_two_analysis(targets=(.25,.5,.75,1.)).solve(stop_after=1)
    resumed=_two_analysis(targets=(.25,.5,.75,1.)).solve(checkpoint=split.checkpoint,
        expected_sha256=split.checkpoint_sha256)
    assert resumed.status=='completed' and resumed.checkpoint==run.checkpoint
    output=os.environ.get('CONTACT_MATERIAL_MULTI_S3_RECORD')
    if output:Path(output).write_text(json.dumps(dict(profile=run.profile,
        coordinates=102,metrics=metrics,minimum_gap_m=contact.contact.minimum_gap_m,
        normalized_force_balance=normalized_force,normalized_moment_balance=normalized_moment,
        beam_force_n=contact.contact.beam_force_n.tolist(),shell_displacement_norm_m=float(np.linalg.norm(a.shell_displacement)),
        potential_j=contact.contact.potential_j,checkpoint_sha256=run.checkpoint_sha256),indent=2))


def test_second_material_patch_hidden_rotation_uses_its_own_retained_block():
    owner=_two_analysis(targets=(.5,1.));n=owner.context.count
    initial=np.zeros(n)
    for node in owner.context.nodes[1][1:]:initial[6*node+2]=-.0003
    index=owner.context.nodal_count+24+3
    initial[index]=-.6
    start=owner.context.advance(owner.initial.mechanical,initial)
    delta=np.zeros(n+18);delta[index]=1.2
    end=owner.context.advance(start,delta[:n]);u=np.zeros(18)
    first=owner._contact(start,u);last=owner._contact(end,u)
    assert all(r.contact.minimum_gap_m>0. for _,r in first.pairs)
    assert all(r.contact.minimum_gap_m>0. for _,r in last.pairs)
    middle=owner.context.advance(start,.5*delta[:n])
    assert owner._contact(middle,u).pairs[1][1].contact.minimum_gap_m < -5e-5
    assert not owner._swept_admissible(start,u,delta,end,u)


def test_material_multi_profile_explicit_activation_and_rejects_mixed_or_legacy_controller():
    from anyfesolver.general_contact import create_native_contact_analysis,NativeS3ContactPair
    from anyfesolver.general_contact.native_activation_static import NativeActivationMultiS3Analysis,NativeContactActivationSettings
    from test_general_contact_native_retained_s3_static import _analysis as legacy
    owner=_two_analysis(targets=(.5,1.));args=(tuple(NativeBeamDefinition(raw) for raw in owner.native._definitions),
        tuple(owner.native.model.boundary_conditions),owner.program,owner.shell_model)
    activation=NativeContactActivationSettings(1e-7,64)
    limited=create_native_contact_analysis(*args,owner.pairs,equilibrium_tolerance=1e-10,activation_settings=activation)
    assert type(limited) is type(owner) and limited.activation_settings is activation
    assert limited.identity!=owner.identity
    with pytest.raises(ValueError,match='without activation controller'):
        NativeActivationMultiS3Analysis(*args,owner.pairs,equilibrium_tolerance=1e-10,activation_settings=activation)
    mixed=(owner.pairs[0],NativeS3ContactPair('line',legacy().contact,5e-5,128))
    with pytest.raises(ValueError,match='mixed line and area'):
        create_native_contact_analysis(*args,mixed,equilibrium_tolerance=1e-10)


@pytest.mark.parametrize('cancel',(False,True))
def test_two_material_owner_staging_and_incompatible_restart(cancel):
    from dataclasses import replace
    from anyfesolver.control import CancellationToken
    from anyfesolver.general_contact import create_native_contact_analysis
    first=_two_analysis(targets=(.25,.5,.75,1.)).solve(stop_after=1)
    assert first.status=='paused',first.failure
    owner=_two_analysis(targets=(.25,.5,.75,1.));checkpoint=owner._checkpoint;token=CancellationToken()
    def staged(records):
        raw=checkpoint(records)
        if len(records)==2:
            if cancel:token.cancel('two material patches after staging')
            else:raise RuntimeError('two material patches after staging')
        return raw
    owner._checkpoint=staged
    run=owner.solve(checkpoint=first.checkpoint,expected_sha256=first.checkpoint_sha256,cancellation_token=token)
    assert run.status==('cancelled' if cancel else 'failed')
    assert run.checkpoint==first.checkpoint and run.accepted.completed_targets==1
    assert_allclose(run.accepted.mechanical.cell_rotations,first.accepted.mechanical.cell_rotations,rtol=0.,atol=0.)
    assert_allclose(run.accepted.shell_displacement,first.accepted.shell_displacement,rtol=0.,atol=0.)
    original=_two_analysis(targets=(.25,.5,.75,1.))
    changed=create_native_contact_analysis(tuple(NativeBeamDefinition(raw) for raw in original.native._definitions),
        tuple(original.native.model.boundary_conditions),original.program,original.shell_model,
        (original.pairs[0],replace(original.pairs[1],gap_tolerance_m=4e-5)),equilibrium_tolerance=1e-10)
    with pytest.raises(ValueError,match='binding'):
        changed.solve(checkpoint=first.checkpoint,expected_sha256=first.checkpoint_sha256)


def test_two_beam_material_example(capsys):
    example=Path(__file__).resolve().parents[1]/'examples/general_contact_native_material_s3.py'
    runpy.run_path(str(example))['main'](beams=2)
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='completed' and result['completed_targets']==9
    assert result['profile']=='NATIVE_B3_GE_TWO_MATERIAL_PATCHES_SHARED_S3_STATIC_DEV_V1'


def test_pre_area_multiple_line_checkpoint_remains_replayable():
    root=Path(__file__).resolve().parents[1]
    fixture=root/'tests/data/general_contact_native_multi_line_46ea65f6.checkpoint'
    provenance=json.loads(fixture.with_suffix('.json').read_text())
    raw=fixture.read_bytes();assert sha256(raw).hexdigest()==provenance['checkpoint_sha256']
    example=runpy.run_path(str(root/'examples/general_contact_native_multiple_s3_static.py'))
    resumed=example['analysis']().solve(checkpoint=raw,
        expected_sha256=provenance['checkpoint_sha256'],stop_after=1)
    assert resumed.status=='paused',resumed.failure
    assert resumed.checkpoint==raw and resumed.accepted.completed_targets==1
    assert resumed.profile=='NATIVE_B3_GE_MULTIPLE_ELASTIC_S3_INTERIORS_STATIC_DEV_V1'
    with pytest.raises(ValueError,match='binding'):
        _two_analysis(targets=(.25,.5,.75,1.)).solve(checkpoint=raw,
            expected_sha256=provenance['checkpoint_sha256'])


def _analysis(*,force=-.005,targets=(.5,1.),stiffness=1e6,radius=.05,
              shell_z=-.0501,angles=(pi-.3,pi+.3),triangle=None,shell_E=1e8,
              shell_node_order=(4,5,6),director_polarity=1,native_yield=1e6,activation_settings=None):
    native,_,inertias=make(False,clamped=True,coupled=False)
    element=native.mesh.elements[1]
    if native_yield!=1e6:
        from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
        from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
        law=EllipsoidalGeneralizedSection(element.section.elastic,element.section.metric,native_yield,
            element.section.hardening)
        element=NativeGeneralizedStaticElement(1,(1,2,3),element.operator.reference,law,order=4)
    definition=NativeBeamDefinition.capture(element,inertias[1])
    program=Program(targets,DistributedPattern(LinePattern(()),()),
        NodalDeadForces(((3,0.,0.,force),)),max_iterations=24)
    shell=FEModel('native_material_s3')
    shell.add_material('elastic',shell_E,.3,density=1.)
    triangle=((-5.,-5.),(5.,-5.),(0.,10.)) if triangle is None else triangle
    for node,(x,y) in zip((4,5,6),triangle):shell.add_node(node,x,y,shell_z-.025)
    shell.add_element(21,create_shell_element(21,shell_node_order,'elastic',thickness=.05,
        formulation='e4-pl-s3-v2d',reference_normal=(0,0,1),reference_surface_offset=0.,director_polarity=director_polarity))
    shell.add_boundary_condition(FixedSupport('edge',[4,6]))
    shell.add_boundary_condition(BoundaryCondition('moving vertex',[5],
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    skin=S3SkinWorkAdapter.from_model(shell,7,21,'top')
    rule=NativeRoundSurfaceRule(1,element.operator.reference,radius,xi_range=(.1,.9),
        angle_range=angles,axial_order=1,angular_order=1)
    patch=NativeMaterialS3Patch(rule,(1,2,3),normal_stiffness_Pa_per_m=stiffness,
        integration_settings=StraightNormalIntegrationSettings(1e-12,1e-10,1e-9,64,2.),
        geometry_tolerance_m=1e-10,rim_clearance_m=1e-6)
    return NativeRetainedRoundS3Analysis((definition,),tuple(native.boundary_conditions),program,
        shell,NativeRetainedMaterialS3Work(patch,skin),gap_tolerance_m=1e-3*radius,
        equilibrium_tolerance=1e-10,maximum_swept_intervals=128,activation_settings=activation_settings)


def _corners(work,u):
    # Independent linear physical-skin map on this fixture's known node order.
    values=np.asarray(u).reshape(-1,6)[[node-4 for node in work.shell.node_ids]]
    return work.shell.reference_positions+values[:,:3]+work.shell.physical_offset*np.cross(
        values[:,3:],work.shell.director)


def test_cyclic_shell_corner_order_preserves_work_and_physical_wrench():
    normal=_analysis();permuted=_analysis(shell_node_order=(5,6,4))
    step=np.zeros(42);step[8]=-.001;step[14]=-.002;step[normal.context.nodal_count+3]=.02
    pose=normal.context.advance(normal.initial.mechanical,step)
    u=np.sin(np.arange(18)+.4)*1e-5
    results=[o._contact(pose,u) for o in (normal,permuted)]
    a,b=results
    assert_allclose(a.residual,b.residual,rtol=1e-10,atol=1e-11)
    assert_allclose(a.tangent.toarray(),b.tangent.toarray(),rtol=1e-9,atol=1e-8)
    assert_allclose(b.contact.shell_vertex_force_n,a.contact.shell_vertex_force_n[[1,2,0]],rtol=1e-10,atol=1e-11)
    assert_allclose(a.contact.shell_moment_nm,b.contact.shell_moment_nm,rtol=1e-10,atol=1e-11)
    assert_allclose(a.contact.beam_moment_nm+a.contact.shell_moment_nm,0.,atol=1e-11)
    assert_allclose(b.contact.beam_moment_nm+b.contact.shell_moment_nm,0.,atol=1e-11)
    ref=_reference(permuted.contact,pose.positions,pose.cell_rotations[0],u)
    assert_allclose(b.contact.beam_moment_nm,ref[4:7],rtol=1e-9,atol=1e-11)


def test_material_skin_rejects_unadmitted_negative_director_polarity():
    with pytest.raises(ValueError,match='positive director/reference orientation'):
        _analysis(director_polarity=-1)


def _accumulated(history):
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    def values(value):
        if isinstance(value,dict):
            if 'accumulated' in value:return [sum(value['accumulated'])]
            return [x for child in value.values() for x in values(child)]
        if isinstance(value,list):return [x for child in value for x in values(child)]
        return []
    return values(json.loads(canonical(history)))


@pytest.mark.parametrize('stiffness',(1e6,1e7))
def test_yielding_fixture_failure_preserves_last_accepted_state(stiffness):
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    kwargs=dict(targets=(.25,.5,.75,1.),native_yield=1e-5,stiffness=stiffness)
    owner=_analysis(**kwargs);run=owner.solve()
    assert run.status=='failed' and 'line-search/gap limit' in run.failure
    prefix=_analysis(**kwargs).solve(stop_after=run.accepted.completed_targets)
    assert prefix.status=='paused' and prefix.checkpoint==run.checkpoint
    assert canonical(prefix.accepted.native_histories)==canonical(run.accepted.native_histories)
    assert canonical(prefix.accepted.mechanical.descriptor())==canonical(run.accepted.mechanical.descriptor())
    assert_allclose(prefix.accepted.mechanical.positions,run.accepted.mechanical.positions,rtol=0.,atol=0.)
    assert_allclose(prefix.accepted.shell_displacement,run.accepted.shell_displacement,rtol=0.,atol=0.)
    assert max(_accumulated(owner.initial.native_histories))==0.
    maximum_accumulated=max(_accumulated(run.accepted.native_histories))
    if stiffness==1e6:
        assert run.accepted.completed_targets==3 and maximum_accumulated>0.
        restored=_analysis(**kwargs).solve(checkpoint=run.checkpoint,
            expected_sha256=run.checkpoint_sha256,stop_after=3)
        assert restored.status=='paused' and restored.checkpoint==run.checkpoint
        assert canonical(restored.accepted.native_histories)==canonical(run.accepted.native_histories)
        assert canonical(restored.accepted.mechanical.descriptor())==canonical(run.accepted.mechanical.descriptor())
        assert_allclose(restored.accepted.shell_displacement,run.accepted.shell_displacement,rtol=0.,atol=0.)
        previous=np.asarray(_accumulated(owner.initial.native_histories))
        for row in json.loads(run.checkpoint)['records']:
            current=np.asarray(_accumulated(row['histories']))
            assert np.all(current>=previous);previous=current
    assert run.accepted.contact.contact.minimum_gap_m>=-owner.gap_tolerance
    output=os.environ.get('CONTACT_MATERIAL_PLASTIC_RECORD')
    if output:Path(output+'-'+str(int(stiffness))+'.json').write_text(json.dumps(dict(
        disposition='coupled yielding NONPASS; failed-trial state preservation checked',
        status=run.status,failure=run.failure,completed_targets=run.accepted.completed_targets,
        maximum_accumulated=maximum_accumulated,
        minimum_gap_m=run.accepted.contact.contact.minimum_gap_m,
        checkpoint_sha256=run.checkpoint_sha256),indent=2))


def test_plastic_branch_complete_owner_tangent_and_disposable_history():
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    owner=_analysis(native_yield=1e-5,shell_z=-.05001)
    origins=owner.initial.native_histories;captured=canonical(origins)
    step=np.zeros(42);step[8]=-2e-5;step[14]=-3e-5
    step[owner.context.nodal_count+6:]=.0002
    pose=owner.context.advance(owner.initial.mechanical,step);u=np.zeros(18)
    residual,tangent,_,responses,contact=owner._assemble(pose,u,1.,origins)
    trial=tuple(row.history for row in responses)
    assert max(_accumulated(trial))>0. and min(_accumulated(trial))>=0.
    assert captured==canonical(origins)==canonical(owner.initial.native_histories)
    assert contact.contact.potential_j>0. and contact.contact.minimum_gap_m>=-owner.gap_tolerance
    d=np.random.default_rng(527).normal(size=60)*.001;d[list(owner.context.fixed)]=0.
    d[42+owner.shell_fixed]=0.;errors=[]
    for h in (1e-5,2e-6):
        states=[owner.context.advance(pose,sign*h*d[:42]) for sign in (1,-1)]
        rows=[owner._assemble(s,u+sign*h*d[42:],1.,origins)[0]
            for sign,s in zip((1,-1),states)]
        fd=(rows[0]-rows[1])/(2*h)
        error=float(np.linalg.norm(tangent@d-fd)/np.linalg.norm(fd));assert error<=1e-5;errors.append(error)
    assert canonical(origins)==captured
    after=owner._assemble(owner.initial.mechanical,u,0.,origins)
    assert max(_accumulated(tuple(row.history for row in after[3])))==0.
    output=os.environ.get('CONTACT_MATERIAL_PLASTIC_RECORD')
    if output:Path(output+'-local.json').write_text(json.dumps(dict(
        disposition='local nonequilibrated plastic-branch code check; no coupled acceptance',
        maximum_accumulated=max(_accumulated(trial)),owner_tangent_errors=errors,
        minimum_gap_m=contact.contact.minimum_gap_m),indent=2))


def test_bounded_plastic_unload_reload_history_and_split_restart():
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    kwargs=dict(targets=(.25,.5,.75,0.,.25,.5,.75),native_yield=1e-5,stiffness=1e6)
    owner=_analysis(**kwargs);run=owner.solve()
    output=os.environ.get('CONTACT_MATERIAL_PLASTIC_RECORD')
    if output:Path(output+'-cycle.json').write_text(json.dumps(dict(status=run.status,
        failure=run.failure,completed_targets=run.accepted.completed_targets,
        maximum_accumulated=max(_accumulated(run.accepted.native_histories))),indent=2))
    assert run.status=='completed',run.failure
    records=json.loads(run.checkpoint)['records'];previous=np.asarray(_accumulated(owner.initial.native_histories))
    for row in records:
        now=np.asarray(_accumulated(row['histories']))
        assert np.all(now>=previous);previous=now
        assert row['contact']['minimum_gap_m']>=-owner.gap_tolerance
    assert max(_accumulated(records[2]['histories']))>0.
    assert max(_accumulated(records[3]['histories']))>=max(_accumulated(records[2]['histories']))
    prefix=_analysis(**kwargs).solve(stop_after=4)
    assert prefix.status=='paused' and max(_accumulated(prefix.accepted.native_histories))>0.
    resumed=_analysis(**kwargs).solve(checkpoint=prefix.checkpoint,expected_sha256=prefix.checkpoint_sha256)
    assert resumed.status=='completed' and resumed.checkpoint==run.checkpoint
    assert canonical(resumed.accepted.mechanical.descriptor())==canonical(run.accepted.mechanical.descriptor())
    assert canonical(resumed.accepted.native_histories)==canonical(run.accepted.native_histories)
    a=run.accepted;r,_,metrics,_,contact=owner._assemble(a.mechanical,a.shell_displacement,.75,a.native_histories)
    skin=owner.contact.shell;shell_r=r[42:].reshape(3,6)
    mids=skin.reference_positions-skin.physical_offset*skin.director+a.shell_displacement.reshape(3,6)[:,:3]
    force=np.array((0.,0.,-.005*.75));root=owner.context.nodes[0][0];tip=owner.context.nodes[0][2]
    balance=shell_r[:,:3].sum(axis=0)+r[6*root:6*root+3]+force
    moment=(np.cross(mids,shell_r[:,:3]).sum(axis=0)+shell_r[:,3:].sum(axis=0)+r[6*root+3:6*root+6]+
        np.cross(a.mechanical.positions[root],r[6*root:6*root+3])+np.cross(a.mechanical.positions[tip],force))
    normalized_force=float(np.linalg.norm(balance)/.00375);normalized_moment=float(np.linalg.norm(moment)/.0075)
    assert max(metrics)<1e-10 and normalized_force<=1e-6 and normalized_moment<=1e-6
    assert_allclose(contact.contact.beam_force_n+contact.contact.shell_force_n,0.,atol=1e-12)
    if output:Path(output+'-cycle.json').write_text(json.dumps(dict(status=run.status,
        completed_targets=run.accepted.completed_targets,peak_load_factor=.75,
        maximum_accumulated=max(_accumulated(run.accepted.native_histories)),
        maximum_penetration_m=-min(row['contact']['minimum_gap_m'] for row in records),
        storage_j=[row['contact']['potential_j'] for row in records],
        normalized_force_balance=normalized_force,normalized_moment_balance=normalized_moment,
        metrics=metrics,checkpoint_sha256=run.checkpoint_sha256),indent=2))


def _reference(work,positions,rotations,u):
    """Direct area quadrature, no production geometry/AD/integration helpers."""
    k=work.contact;corners=_corners(work,u)
    normal=np.cross(corners[1]-corners[0],corners[2]-corners[0]);normal/=np.linalg.norm(normal)
    frame=rotations[k.cell]@k.beam_reference.nodal_triads[0]
    lo,hi=k.rule.definition['xi_range'];alo,ahi=k.rule.definition['angle_range']
    measure=k.radius*np.linalg.norm(k.beam_reference.coordinates[k.cell+1]-k.beam_reference.coordinates[k.cell])
    def point(t,a):return (1-t)*positions[k.cell]+t*positions[k.cell+1]+k.radius*(frame[:,1]*cos(a)+frame[:,2]*sin(a))
    def gap(t,a):return float((point(t,a)-corners[0])@normal)
    x,w=np.polynomial.legendre.leggauss(3)
    def axial(a,index):
        gl,gr=gap(lo,a),gap(hi,a);left,right=lo,hi
        if gl>=0 and gr>=0:return 0.
        if gl*gr<0:
            root=lo-(hi-lo)*gl/(gr-gl)
            if gl<0:right=root
            else:left=root
        middle=(left+right)/2;half=(right-left)/2
        total=0.
        for node,weight in zip(x,w):
            t=middle+half*node;p=point(t,a);g=gap(t,a)
            f=-k.stiffness*min(g,0.)*normal
            value=np.r_[.5*k.stiffness*min(g,0.)**2,f,np.cross(p,f)]
            total+=weight*value[index]
        return half*measure*total
    cuts={alo,ahi};grid=np.linspace(alo,ahi,65)
    derivative=lambda a:float(normal@(k.radius*(-frame[:,1]*sin(a)+frame[:,2]*cos(a))))
    extrema=[alo,ahi]
    for l,r in zip(grid[:-1],grid[1:]):
        if derivative(l)*derivative(r)<0:extrema.append(brentq(derivative,l,r,xtol=1e-14))
    extrema.sort();cuts.update(extrema)
    for t in (lo,hi):
        for l,r in zip(extrema[:-1],extrema[1:]):
            if gap(t,l)*gap(t,r)<0:cuts.add(brentq(lambda a:gap(t,a),l,r,xtol=1e-14))
    values=[]
    for i in range(7):
        tolerance=2e-18 if i==0 else 2e-13
        v,error=quad(lambda a:axial(a,i),alo,ahi,points=sorted(cuts)[1:-1],
            epsabs=tolerance,epsrel=2e-13,limit=100)
        assert error<=tolerance+2e-13*abs(v),(i,v,error)
        values.append(v)
    return np.array(values)


def _moved(positions,rotations,u,direction,h):
    p=positions.copy();R=rotations.copy()
    for i in range(3):p[i]+=h*direction[6*i:6*i+3]
    for c in (0,1):R[c]=Rotation.from_rotvec(h*direction[18+3*c:21+3*c]).as_matrix()@R[c]
    return p,R,u+h*direction[42:]


@pytest.mark.parametrize('partial',(False,True))
def test_independent_area_work_both_body_wrench_and_complete_tangent(partial):
    owner=_analysis(stiffness=1000.);work=owner.contact;k=work.contact
    p=k.beam_reference.coordinates.copy();R=np.tile(np.eye(3),(2,1,1));u=np.zeros(18)
    R[1]=Rotation.from_rotvec((.12,-.04,.02)).as_matrix()
    p[:,2]-=.01
    if partial:p[1,2]=-.00002;p[2,2]=-.0003
    # Physical offset rotations and a moving target normal both enter the map.
    u[8]=1e-5;u[3:6]=(.00003,-.00002,.00001)
    value=work.evaluate(p,np.zeros((3,3)),R,u);oracle=_reference(work,p,R,u)
    assert_allclose(value.contact.potential_j,oracle[0],rtol=2e-8,atol=2e-12)
    assert_allclose(value.contact.beam_force_n,oracle[1:4],rtol=2e-8,atol=2e-10)
    assert_allclose(value.contact.beam_moment_nm,oracle[4:],rtol=2e-8,atol=2e-10)
    assert_allclose(value.contact.beam_force_n+value.contact.shell_force_n,0.,atol=1e-12)
    assert_allclose(value.contact.beam_moment_nm+value.contact.shell_moment_nm,0.,atol=1e-12)
    assert np.linalg.norm(value.tangent.toarray()[18:24,42:])>0.
    direction=np.random.default_rng(643).normal(size=60)*.001
    direction[24:42]=0.;direction[3:6]=direction[9:12]=direction[15:18]=0.
    for h in (1e-5,2e-6):
        plus=_moved(p,R,u,direction,h);minus=_moved(p,R,u,direction,-h)
        vp=work.evaluate(plus[0],np.zeros((3,3)),plus[1],plus[2])
        vm=work.evaluate(minus[0],np.zeros((3,3)),minus[1],minus[2])
        fd=(vp.residual-vm.residual)/(2*h)
        assert np.linalg.norm(value.tangent@direction-fd)/np.linalg.norm(fd)<=1e-5
        reference_work=(_reference(work,*plus)[0]-_reference(work,*minus)[0])/(2*h)
        assert_allclose(value.residual@direction,reference_work,rtol=1e-5,atol=1e-11)


def test_coupled_deformable_target_response_and_exact_split_restart():
    owner=_analysis();run=owner.solve();assert run.status=='completed',run.failure
    assert run.profile==PROFILE and np.linalg.norm(run.accepted.shell_displacement)>1e-10
    a=run.accepted;r,j,metrics,_,contact=owner._assemble(a.mechanical,a.shell_displacement,1.,a.native_histories)
    assert max(metrics)<1e-10
    assert contact.contact.minimum_gap_m>=-owner.gap_tolerance
    assert_allclose(contact.contact.beam_force_n+contact.contact.shell_force_n,0.,atol=1e-12)
    load=np.array((0.,0.,-.005));reactions=r[42:].reshape(3,6)
    shell_mid=np.array([owner.shell_model.mesh.nodes[n].coords() for n in (4,5,6)])
    shell_mid+=a.shell_displacement.reshape(3,6)[:,:3]
    force_balance=r[:3]+load+reactions[:,:3].sum(axis=0)
    moment_balance=(r[3:6]+np.cross(a.mechanical.positions[0],r[:3])
        +np.cross(a.mechanical.positions[2],load)
        +np.cross(shell_mid,reactions[:,:3]).sum(axis=0)+reactions[:,3:].sum(axis=0))
    normalized_force=float(np.linalg.norm(force_balance)/.005)
    normalized_moment=float(np.linalg.norm(moment_balance)/.01)
    assert normalized_force<=1e-6 and normalized_moment<=1e-6
    split=_analysis().solve(stop_after=1);assert split.status=='paused',split.failure
    resumed=_analysis().solve(checkpoint=split.checkpoint,expected_sha256=split.checkpoint_sha256)
    assert resumed.status=='completed',resumed.failure
    assert resumed.checkpoint==run.checkpoint
    changed=_analysis(stiffness=2e6)
    with pytest.raises(ValueError,match='binding'):
        changed.solve(checkpoint=split.checkpoint,expected_sha256=split.checkpoint_sha256)
    # Complete owner derivative through its actual native incremental map.
    d=np.random.default_rng(819).normal(size=60)*.001
    d[list(owner.context.fixed)]=0.;d[owner.context.count+owner.shell_fixed]=0.
    errors=[]
    for h in (1e-5,2e-6):
        states=[owner.context.advance(a.mechanical,sign*h*d[:42]) for sign in (1,-1)]
        rows=[owner._assemble(s,a.shell_displacement+sign*h*d[42:],1.,a.native_histories)[0]
              for sign,s in zip((1,-1),states)]
        fd=(rows[0]-rows[1])/(2*h);error=np.linalg.norm(j@d-fd)/np.linalg.norm(fd)
        assert error<=1e-5;errors.append(float(error))
    output=os.environ.get('CONTACT_MATERIAL_S3_RECORD')
    if output:Path(output).write_text(json.dumps(dict(profile=run.profile,metrics=metrics,
        minimum_gap_m=contact.contact.minimum_gap_m,force_n=contact.contact.beam_force_n.tolist(),
        shell_displacement_norm=float(np.linalg.norm(a.shell_displacement)),
        normalized_force_balance=normalized_force,normalized_moment_balance=normalized_moment,
        owner_tangent_errors=errors,checkpoint_sha256=run.checkpoint_sha256),indent=2))


def test_press_open_recontact_and_open_checkpoint():
    owner=_analysis(targets=(.5,1.,0.,.5,1.));run=owner.solve()
    assert run.status=='completed',run.failure
    rows=json.loads(run.checkpoint)['records']
    assert rows[1]['contact']['potential_j']>0.
    assert rows[2]['contact']['potential_j']==0. and rows[2]['contact']['minimum_gap_m']>0.
    assert rows[4]['contact']['potential_j']>0.
    assert_allclose(rows[1]['contact']['beam_force_n'],rows[4]['contact']['beam_force_n'],rtol=1e-6,atol=1e-12)
    opened=_analysis(targets=(.5,1.,0.,.5,1.)).solve(stop_after=3)
    assert opened.status=='paused' and opened.accepted.contact.contact.potential_j==0.
    continued=_analysis(targets=(.5,1.,0.,.5,1.)).solve(checkpoint=opened.checkpoint,
        expected_sha256=opened.checkpoint_sha256)
    assert continued.status=='completed' and continued.checkpoint==run.checkpoint


def test_deformable_target_explicit_area_penalty_sensitivity():
    rows=[]
    for k in (1e6,1e7):
        owner=_analysis(stiffness=k);run=owner.solve();assert run.status=='completed',run.failure
        c=run.accepted.contact.contact
        assert c.minimum_gap_m>=-owner.gap_tolerance
        rows.append(dict(normal_stiffness_Pa_per_m=k,gap_lower_m=c.minimum_gap_m,
            force_n=c.beam_force_n.tolist(),shell_displacement_norm=float(np.linalg.norm(run.accepted.shell_displacement))))
    assert abs(rows[1]['gap_lower_m'])<abs(rows[0]['gap_lower_m'])
    output=os.environ.get('CONTACT_MATERIAL_S3_PENALTY_RECORD')
    if output:Path(output).write_text(json.dumps(dict(scope='explicit sensitivity; no convergence acceptance',rows=rows,
        force_change_relative=float(np.linalg.norm(np.subtract(rows[1]['force_n'],rows[0]['force_n']))/np.linalg.norm(rows[1]['force_n']))),indent=2))


def test_fixed_nodes_hidden_rotational_gap_requires_material_radius_bound():
    owner=_analysis(radius=.03,shell_z=-.0299,angles=(pi-.31,pi-.29))
    start=owner.initial.mechanical;u=np.zeros(18);step=np.zeros(60);step[21]=1.2
    end=owner.context.advance(start,step[:42]);nodes=owner.context.nodes[0]
    query=lambda state:owner.contact.geometry_gap(state.positions[nodes],state.position_low[nodes],state.cell_rotations[0],u)
    g0,r0=query(start);g1,r1=query(end)
    assert g0>0. and g1>0. and query(owner.context.advance(start,.5*step[:42]))[0]>0.
    assert query(owner.context.advance(start,.25*step[:42]))[0]<-owner.gap_tolerance
    from anyfesolver.general_contact.native_s3_swept import native_s3_swept_admissible
    # Reproduce the zero-motion bound defect in an isolated helper call.
    assert native_s3_swept_admissible(owner.context,owner.contact,0,owner.gap_tolerance,128,
        owner._recoverable_trial_error,start,u,step,end,u,g0,g1,r0,r1,
        geometry_only=True,geometry_query=lambda s,v:query(s))
    assert not owner._swept_admissible(start,u,step,end,u,g0,g1,r0,r1)


def test_hidden_rotation_crosses_finite_target_edge_with_open_normal_gap():
    triangle=((-5.,.0299),(0.,-5.),(5.,.0299))
    owner=_analysis(radius=.03,shell_z=-.05,angles=(pi+1.27,pi+1.29),triangle=triangle)
    start=owner.initial.mechanical;u=np.zeros(18);step=np.zeros(60);step[21]=1.2
    end=owner.context.advance(start,step[:42]);nodes=owner.context.nodes[0]
    query=lambda s:owner.contact.geometry_gap(s.positions[nodes],s.position_low[nodes],s.cell_rotations[0],u)
    g0,r0=query(start);g1,r1=query(end)
    assert min(g0,g1)>0. and query(owner.context.advance(start,.5*step[:42]))[0]>0.
    with pytest.raises(ValueError,match='target ownership'):
        query(owner.context.advance(start,.25*step[:42]))
    assert not owner._swept_admissible(start,u,step,end,u,g0,g1,r0,r1)


def test_reference_orientation_hidden_inversion_is_not_start_normal_admission():
    from types import SimpleNamespace as NS
    from anyfesolver.general_contact.native_s3_swept import native_s3_swept_admissible
    first=np.array(((0.,0.,0.),(0.,1.,0.),(-3/16,0.,1.)))
    last=np.array(((0.,0.,0.),(1.,1.,0.),(13/16,1.,1.)))
    def cross(t):
        corners=(1-t)*first+t*last
        return np.cross(corners[1]-corners[0],corners[2]-corners[0])
    controls=(cross(0.),2*cross(.5)-.5*(cross(0.)+cross(1.)),cross(1.))
    assert all(c@cross(0.)>0. for c in controls)
    assert cross(0.)[2]>0. and cross(1.)[2]>0. and cross(.5)[2]<0.
    shell=NS(reference_positions=np.array(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))),
        current_positions=lambda u:u.reshape(3,3))
    kernel=NS(geometry_tolerance=1e-10,radius=.03,rim_clearance=1e-6,
        beam_reference=NS(half_cell_lift=lambda c,t:np.zeros(3)))
    work=NS(shell=shell,contact=kernel)
    context=NS(nodes=(np.arange(3),),nodal_count=18,count=42)
    state=NS(positions=np.array(((0.,0.,2.),(1.,0.,2.),(2.,0.,2.))),position_low=np.zeros((3,3)))
    args=(context,work,0,1e-5,128,lambda e:False,state,first.ravel(),np.zeros(51),
        state,last.ravel(),1e6,1e6,1e6,1e6)
    # Manufactured geometry bounds isolate orientation, without a structural
    # solve or a claim that these large shell strains are an admitted fixture.
    assert native_s3_swept_admissible(*args,geometry_only=True)
    assert not native_s3_swept_admissible(*args,geometry_only=True,material_radius=.03)


def test_normal_material_owner_rejects_existing_friction_bridge():
    from anyfesolver.general_contact.native_surface_contributor import NativeS3MaterialContributor
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
    from test_general_contact_native_retained_s3_static import _analysis as legacy
    owner=_analysis();work=owner.contact
    pair=NativeS3MaterialPointFriction(work.contact.rule.samples[0].station,work.target,
        reference_weight_m2=.01,normal_stiffness_Pa_per_m=1000.,
        friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.)
    with pytest.raises(ValueError,match='does not admit the existing friction transaction'):
        NativeS3MaterialContributor(owner,pair)
    with pytest.raises(ValueError,match='does not admit the existing friction transaction'):
        owner.solve_material_increment(None,1.)
    # Existing binding only: no friction history evaluation or solve/replay.
    old=legacy(shell_z=-.0501)
    assert NativeS3MaterialContributor(old,pair).owner is old


@pytest.mark.parametrize('cancel',(False,True))
def test_staging_failure_preserves_native_and_deformable_shell_prefix(cancel):
    from anyfesolver.control import CancellationToken
    first=_analysis().solve(stop_after=1);assert first.status=='paused',first.failure
    owner=_analysis();checkpoint=owner._checkpoint;token=CancellationToken()
    def staged(records):
        raw=checkpoint(records)
        if len(records)==2:
            if cancel:token.cancel('after checkpoint staging')
            else:raise RuntimeError('after checkpoint staging')
        return raw
    owner._checkpoint=staged
    run=owner.solve(checkpoint=first.checkpoint,expected_sha256=first.checkpoint_sha256,
                    cancellation_token=token)
    assert run.status==('cancelled' if cancel else 'failed')
    assert run.checkpoint==first.checkpoint and run.accepted.completed_targets==1
    assert_allclose(run.accepted.shell_displacement,first.accepted.shell_displacement,rtol=0.,atol=0.)
    assert_allclose(run.accepted.mechanical.cell_rotations,first.accepted.mechanical.cell_rotations,rtol=0.,atol=0.)
