"""Explicit opposite-role shell route through the existing static contributor."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import (FEModel, create_shell_element, BoundaryCondition, ContactSurface,
    ContactInteraction, ContactDefinition, ContactPathControl, ContactPathProgram,
    validate_general_contact, solve_static_nonlinear)
from anyfesolver.general_contact.s3_patch_static import (
    GeneralS3PatchManager, S3_Q4_PROJECTED_PENALTY_PROFILE,
)
from anyfesolver.general_contact.s3_q4_swept import s3_q4_affine_lower
from test_general_contact_projected_penalty import _candidate


def build(*, target_rigid_normal=True):
    model = FEModel('s3_source_original_q4_target')
    model.add_material('elastic', 1e5, .3, density=1.)
    xy = ((.1,.1),(.9,.1),(.9,.9),(.1,.9),(0,0),(1,0),(1,1),(0,1))
    for n,(x,y) in enumerate(xy,1):
        model.add_node(n,x,y,.0601 if n<=4 else 0.)
    for eid,nodes in ((1,(1,2,4)),(2,(2,3,4)),(3,(5,6,7,8))):
        options = dict(formulation='e4-pl-s3-v2d',reference_surface_offset=.01) if eid<=2 else {}
        model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,
            reference_normal=(0,0,1),**options))
    model.add_boundary_condition(BoundaryCondition('source_guides',(1,2,3,4),
        {name:0. for name in ('uy','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target_guides',(5,6,7,8),
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    groups=[(1,(2,3,4),(0,2))]
    if target_rigid_normal:
        groups.append((5,(6,7,8),(2,)))
    else:
        model.add_boundary_condition(BoundaryCondition('target_edge_support',(5,6),{'uz':0.}))
    for master,nodes,axes in groups:
        for node in nodes:
            for axis in axes:
                model.add_constraint_equation(terms=((int(model.mesh.dof_manager.get_node_dofs(node)[axis]),1.),
                    (int(model.mesh.dof_manager.get_node_dofs(master)[axis]),-1.)),source_id=f'tie_{node}_{axis}')
    definition = ContactDefinition((ContactSurface('s3',(1,2),'bottom'),ContactSurface('q4',(3,),'top')),
        (), (ContactInteraction('pair','s3','q4',1e7,5e-5,normal_law='projected_penalty'),))
    settings = replace(_candidate()[2], general_s3_differentiation='jet2_s3_q4_dev',
        q4_source_gap_certificate='whole_face',q4_target_weighted_gap_tolerance=1e-12,
        maximum_q4_target_integration_triangles=128)
    return model,definition,settings


def test_both_body_reaction_shared_pressure_tangent_and_scope():
    model,definition,settings=build()
    report=validate_general_contact(model,definition,settings)
    assert report.valid,report.issues
    assert report.profile==S3_Q4_PROJECTED_PENALTY_PROFILE
    manager=GeneralS3PatchManager(model,definition,settings)
    u=np.zeros(manager.n_dof); direction=np.zeros_like(u)
    for node in range(1,9):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        if node<=4: u[dofs[2]]=-130e-6
        direction[dofs[2]]=.03*(node-4.5)
        direction[dofs[0]]=.002*(node-4)
    residual,tangent,contact=manager.evaluate(u)
    np.testing.assert_allclose(contact.integration_force,(0,0,192),atol=1e-8)
    np.testing.assert_allclose(contact.target_force,(0,0,-192),atol=1e-8)
    np.testing.assert_allclose(contact.integration_moment,(96,-96,0),atol=1e-8)
    np.testing.assert_allclose(np.add(contact.integration_moment,contact.target_moment),0,atol=1e-8)
    assert len(contact.source_pressure_node_ids)==4
    assert contact.pressure_gram_rank==4
    for h in (1e-7,3e-8):
        fd=(manager.evaluate(u+h*direction,tangent=False)[0]-manager.evaluate(u-h*direction,tangent=False)[0])/(2*h)
        assert np.linalg.norm(fd-tangent@direction)/np.linalg.norm(tangent@direction)<1e-5
    assert manager.evaluate(np.zeros_like(u),tangent=False)[2].active_pressure_dofs==0
    assert not validate_general_contact(model,definition,replace(settings,
        general_s3_differentiation='existing')).valid
    assert not validate_general_contact(model,definition,replace(settings,
        q4_source_support_hessian_tolerance=None)).valid


def test_all_four_corner_and_original_affine_swept_certificate():
    a=np.array(((.1,.1,.0251),(.9,.1,.0251),(.1,.9,.0251)))
    b=np.array(((0.,0.,.025),(1.,0.,.025),(1.,1.,.025),(0.,1.,.025)))
    end=a+np.array((.02,0,-130e-6))
    bound=s3_q4_affine_lower(a,end,b,b)
    assert -30.00001e-6 < bound <= -30e-6
    # Three corners have positive polarity but the fourth inverts a Q4 corner.
    bad=b.copy();bad[3,:2]=(2.,.5)
    assert np.cross(bad[1]-bad[0],bad[2]-bad[0])[2]>0
    assert s3_q4_affine_lower(a,a,bad,bad)==-np.inf
    rotated=b.copy();rotated[:,:2]=1.-rotated[:,:2]
    assert np.isfinite(s3_q4_affine_lower(a,a,rotated,rotated))
    assert s3_q4_affine_lower(a,a,b,rotated)==-np.inf
    with pytest.raises(ValueError,match='enclosure'):
        s3_q4_affine_lower(a,a,b,b,endpoint_errors=(-1.,0.,0.,0.))


def test_existing_path_press_slide_open_recontact_and_target_motion(tmp_path):
    model,definition,settings=build()
    program=ContactPathProgram(controls=(ContactPathControl.node(1,'uz'),
        ContactPathControl.node(1,'ux'),ContactPathControl.node(5,'uz')),
        waypoints=((0.,0.,0.),(-130e-6,0.,0.),(-130e-6,.02,0.),(0.,.02,0.),
                   (-130e-6,.02,-10e-6)),steps_per_segment=1)
    result=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,emit_restart_checkpoint=True)
    assert result.converged,result
    history=result.info['force_displacement_history']
    np.testing.assert_allclose([item['contact']['integration_force'][2] for item in history],(192,192,0,128),atol=1e-7)
    assert all(item['contact']['gap_tolerance_met'] for item in history)
    assert result.restart_checkpoint is not None
    assert result.info['contact_path']['profile']=='S3_Q4_PROJECTED_FINITE_PENALTY_PATH_STATIC_DEV_V0'
    from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes, load_nonlinear_checkpoint
    model,definition,settings=build()
    opened=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,max_load_factor=.75,emit_restart_checkpoint=True)
    assert opened.converged
    checkpoint_path=tmp_path/'open.json'
    checkpoint_path.write_bytes(canonical_checkpoint_json_bytes(opened.restart_checkpoint))
    checkpoint=load_nonlinear_checkpoint(checkpoint_path.read_bytes())
    model,definition,settings=build()
    resumed=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,restart_checkpoint=checkpoint,emit_restart_checkpoint=True)
    assert resumed.converged
    np.testing.assert_array_equal(resumed.displacements,result.displacements)
    assert resumed.restart_checkpoint['element_states']==result.restart_checkpoint['element_states']
    assert resumed.info['force_displacement_history']==history


def test_rotated_negative_gap_bound_against_independent_decimal_corners():
    from decimal import Decimal,localcontext
    # Binary coordinates are the actual input. The oracle uses decimal80
    # cross products and normalization, not the production interval routine.
    v=np.array((.43,-.71,.38));v/=np.linalg.norm(v)
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_rotvec(v*.91).as_matrix()
    a=np.array(((.1,.1,-3e-5),(.9,.1,-3e-5),(.1,.9,-3e-5)))@rotation.T
    b=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))@rotation.T
    bound=s3_q4_affine_lower(a,a,b,b)
    with localcontext() as context:
        context.prec=80
        aa=[[Decimal.from_float(float(x)) for x in p] for p in a]
        bb=[[Decimal.from_float(float(x)) for x in p] for p in b]
        e=[aa[1][k]-aa[0][k] for k in range(3)]
        f=[aa[2][k]-aa[0][k] for k in range(3)]
        n=[e[j]*f[k]-e[k]*f[j] for j,k in ((1,2),(2,0),(0,1))]
        norm=sum(x*x for x in n).sqrt()
        minimum=min(sum((p[k]-q[k])*n[k] for k in range(3))/norm for p in aa for q in bb)
        assert Decimal.from_float(bound)<=minimum
    assert np.isfinite(bound)


@pytest.mark.parametrize('load',(.01,0.),ids=('mixed_load','pure_press'))
def test_structurally_deforming_target_against_independent_fixed_geometry_kkt(load):
    from itertools import product
    from math import factorial
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.general_contact.static_path import ContactPathNodalForce
    model,definition,settings=build(target_rigid_normal=False)
    # Reduce the registered clearance to1um and press1.3um. This limits the
    # von Karman vs linear reference discrepancy; no stiffness/gap relaxation.
    for node in range(1,5):
        model.mesh.get_node(node).z-=99e-6
    xy=np.array(((.1,.1),(.9,.1),(.9,.9),(.1,.9)))
    support=np.zeros(4);coupling=np.zeros((4,4))
    for nodes in ((0,1,3),(1,2,3)):
        points=xy[list(nodes)]
        area=abs(np.linalg.det(np.stack((points[1]-points[0],points[2]-points[0]))))*.5
        def moment(indices):
            powers=[indices.count(j) for j in range(3)]
            return 2*area*np.prod([factorial(p) for p in powers])/factorial(len(indices)+2)
        for i,node in enumerate(nodes):
            one=moment((i,))
            x=sum(points[j,0]*moment((i,j)) for j in range(3))
            y=sum(points[j,1]*moment((i,j)) for j in range(3))
            x_y=sum(points[j,0]*points[k,1]*moment((i,j,k)) for j in range(3) for k in range(3))
            support[node]+=one
            coupling[node]+=np.array((one-x-y+x_y,x-x_y,x_y,y-x_y))
    # Independent P1-Q1 polynomial moments + the existing structural linear
    # operator, never the production contact overlap, work or pressure kernel.
    h=coupling[:,2:]/support[:,None]
    stiffness=assemble_stiffness_matrix(model)[0]
    free=[model.mesh.dof_manager.get_node_dofs(node)[2] for node in (7,8)]
    structural=stiffness[np.ix_(free,free)].toarray()
    clearance=1e-6;press=-1.3e-6;kappa=1e7
    # Explicit upward target loads exercise mixed physical force/control work.
    # Pure pressing has zero contact-free reporting reaction; the exact new
    # profile permits it after the unchanged constrained mechanism checks.
    target_load=np.full(2,load)
    candidates=[]
    for flags in product((False,True),repeat=4):
        active=np.array(flags)
        weights=kappa*support*active
        displacement=np.linalg.solve(structural+h.T@(weights[:,None]*h),
            target_load+h.T@(weights*np.full(4,clearance+press)))
        gap=clearance+press-h@displacement
        if np.array_equal(active,gap<0.):
            candidates.append((displacement,kappa*np.maximum(0.,-gap)))
    assert len(candidates)==1
    reference,pressure=candidates[0]
    program=ContactPathProgram(controls=(ContactPathControl.node(1,'uz'),ContactPathControl.node(1,'ux')),
        waypoints=((0.,0.),(press,0.)),steps_per_segment=1,
        nodal_forces=tuple(ContactPathNodalForce(node,'uz',(0.,load)) for node in (7,8)) if load else ())
    result=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,tolerance=1e-10)
    assert result.converged,result.failure_reason
    actual=result.displacements[free]
    assert np.linalg.norm(actual-reference)/np.linalg.norm(reference)<1e-6
    assert np.all(actual<0.) and abs(actual[0])>1e-8
    reported=np.array(result.info['general_contact']['numerical_pressure_pa'])
    assert np.linalg.norm(reported-pressure)/np.linalg.norm(pressure)<1e-6
    expected_force=support@pressure
    np.testing.assert_allclose(result.info['general_contact']['integration_force'][2],expected_force,rtol=1e-6)
    if not load:
        assert result.info['contact_free_elastic_force_predictor_n']==0.
    # Retained by the bounded run's -s log; ordinary pytest still captures it.
    import json
    print('S3_Q4_KKT_RESPONSE',json.dumps({
        'target_load_per_node_n':load,
        'reference_displacement_m':reference.tolist(),
        'actual_displacement_m':actual.tolist(),
        'relative_displacement_error':float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
        'reference_pressure_pa':pressure.tolist(),
        'actual_pressure_pa':reported.tolist(),
        'relative_pressure_error':float(np.linalg.norm(reported-pressure)/np.linalg.norm(pressure)),
        'reference_source_force_n':float(expected_force),
        'actual_source_force_n':float(result.info['general_contact']['integration_force'][2]),
    },sort_keys=True))


def test_zero_predictor_keeps_legacy_and_uncontrolled_mechanism_rejection():
    from scipy import sparse
    from anyfesolver.general_contact.static_path import _elastic_displacement_force_predictor
    stiffness=sparse.diags((0.,2.),format='csr')
    transform=sparse.eye(2,format='csr');row=np.array(((1.,0.),));target=np.array(((.1,),))
    with pytest.raises(ValueError,match='unresolved'):
        _elastic_displacement_force_predictor(stiffness,transform,row,target)
    assert _elastic_displacement_force_predictor(stiffness,transform,row,target,allow_zero=True)==0.
    with pytest.raises(ValueError,match='unavailable'):
        _elastic_displacement_force_predictor(sparse.csr_matrix((2,2)),transform,row,target,allow_zero=True)


def test_zero_predictor_tiny_positive_controlled_null_and_injection_refusals(monkeypatch):
    from types import SimpleNamespace
    from scipy import sparse
    import anyfesolver.general_contact.static_path as path_driver
    from anyfesolver.general_contact.static_path import _elastic_displacement_force_predictor
    transform=sparse.eye(2,format='csr');row=np.array(((1.,0.),));target=np.array(((.5,),))
    # A tiny positive diagonal reaction stays positive and exact under both
    # allowance values; no floor or snapping replaces it.
    tiny=sparse.diags((2.**-30,2.**-30),format='csr')
    for allow in (False,True):
        assert _elastic_displacement_force_predictor(
            tiny,transform,row,target,allow_zero=allow)==2.**-31
    # A controlled null mode beside a stiff second control keeps its nonzero
    # reaction sum under the strict default.
    mixed=sparse.diags((0.,2.),format='csr')
    assert _elastic_displacement_force_predictor(mixed,transform,
        np.array(((1.,0.),(0.,1.))),np.array(((.1,.25),)))==.5
    # An uncontrolled null mode stays unavailable under both allowance values.
    for allow in (False,True):
        with pytest.raises(ValueError,match='unavailable'):
            _elastic_displacement_force_predictor(
                sparse.csr_matrix((2,2)),transform,row,target,allow_zero=allow)
    # Nonfinite and zero-vector responses stay unresolved even with the zero
    # allowance; the finiteness and residual checks run before it.
    stiff=sparse.diags((2.,2.),format='csr')
    monkeypatch.setattr(path_driver,'factorize',lambda *a,**k: SimpleNamespace(
        solve=lambda rhs: np.full(np.asarray(rhs).shape,np.nan)))
    with pytest.raises(ValueError,match='unresolved'):
        _elastic_displacement_force_predictor(stiff,transform,row,target,allow_zero=True)
    monkeypatch.setattr(path_driver,'factorize',lambda *a,**k: SimpleNamespace(
        solve=lambda rhs: np.zeros_like(np.asarray(rhs,dtype=float))))
    with pytest.raises(ValueError,match='unresolved'):
        _elastic_displacement_force_predictor(stiff,transform,row,target,allow_zero=True)
