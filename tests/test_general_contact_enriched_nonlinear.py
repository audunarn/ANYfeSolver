"""Development discriminators for the separately versioned nonlinear skin.

The last case is the ORIGINAL matched model, freshly built/cold, not a toy.
Scalar independent integration below shares only the specified stress space,
never the implementation's AD/loading/gradient/Hessian routines.
"""
import json
from pathlib import Path
import numpy as np
import pytest

from anyfesolver import FEModel,BoundaryCondition,create_shell_element
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.e4_pl_enriched_nonlinear import compile_nonlinear_cell,FORMULATION
from anyfesolver.e4_pl_element import _source_fields,_coefficients
from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy,EnrichedGeometryError
from anyfesolver.general_contact.enriched_nonlinear import (
    NonlinearEnrichedSystem,PairDefinition,MixedCriteria,IterationControl,
)


def small(*,paired=False,fixed=False,touch=False,target_half=2.,unequal_offset=False,
          target_yield_stress=0.,target_hardening_curve=None):
    model=FEModel('nonlinear_enriched_development')
    model.add_material('elastic',1e5,.3,density=1.)
    model.add_material('target',1e5,.3,density=1.,yield_stress=target_yield_stress,
                       hardening_curve=target_hardening_curve)
    source=np.array(((-.5,-.5,0.),(.5,-.5,0.),(.5,.5,0.),(-.5,.5,0.)))
    if paired:
        z=-.04 if touch else -.2
        target=np.array(((-target_half,-target_half,z),(target_half,-target_half,z),
                         (target_half,target_half,z),(-target_half,target_half,z)))
        faces=((5,6,8),(6,7,8));normal=(0.,0.,1.)
    else:
        target=np.array(((-3.,-3.,-.2),(0.,4.,-.2),(4.,-3.,-.2)))
        faces=((5,6,7),);normal=(0.,0.,-1.)
    for node,xyz in enumerate(np.vstack((source,target)),1):model.add_node(node,*xyz)
    model.add_element(1,create_shell_element(1,(1,2,3,4),'elastic',thickness=.05,
        reference_normal=(0.,0.,1.)))
    pairs=[]
    for i,nodes in enumerate(faces,2):
        model.add_element(i,create_shell_element(i,nodes,'target',thickness=.05,
            reference_normal=normal,reference_surface_offset=.01))
        pairs.append(PairDefinition(1,i,-.025,.015+(.001 if unequal_offset and i==3 else 0.)))
    if fixed:
        model.add_boundary_condition(BoundaryCondition('fixed_nodal',tuple(model.mesh.nodes),
            {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    system=NonlinearEnrichedSystem.from_model(model,pairs,normal_stiffness=1e4,
        quadrature=QuadraturePolicy(1e-10,1e-8,1e-5,14144))
    return model,system


def independent_loading(cell,q,order=7):
    """Scalar physical strain integration with independent tensor Gauss rule."""
    linear=cell.linear;frame=linear.frame;reference=linear.reference
    points=reference+q[:24].reshape(4,6)[:,:3]
    director=np.cross(points[1]-points[0],points[3]-points[0]);director/=np.linalg.norm(director)
    direction=director @ frame
    nodal=q[:24].reshape(4,6)[:,:3] @ frame
    local=(reference-reference.mean(axis=0)) @ frame[:,:2];c=_coefficients(local)
    result=linear.coupling.T @ q
    xx,ww=np.polynomial.legendre.leggauss(order)
    for r,wr in zip(xx,ww):
        for s,ws in zip(xx,ww):
            nr=np.array((-(1-s),1-s,1+s,-(1+s)))/4
            ns=np.array((-(1-r),-(1+r),1+r,1-r))/4
            jac=np.array((nr @ local,ns @ local))
            gradient=np.linalg.solve(jac,np.vstack((nr,ns)))
            b=(1-r*r)*(1-s*s)
            br=np.array((-2*r*(1-s*s), (1-3*r*r)*(1-s*s),
                         -2*r*s*(1-s*s),s*(1-3*r*r)*(1-s*s)))
            bs=np.array((-2*s*(1-r*r),-2*s*r*(1-r*r),
                         (1-3*s*s)*(1-r*r),r*(1-3*s*s)*(1-r*r)))
            bg=np.linalg.solve(jac,np.vstack((br,bs))) @ q[24:]
            du=gradient @ nodal
            physical=du+np.outer(bg,direction)
            actual=np.array((physical[0,0]+physical[0,2]**2/2,
                physical[1,1]+physical[1,2]**2/2,
                physical[1,0]+physical[0,1]+physical[0,2]*physical[1,2],0.,0.,0.,
                physical[0,2],physical[1,2]))
            reference_linear=np.array((du[0,0],du[1,1],du[1,0]+du[0,1],0.,0.,0.,
                                       du[0,2]+bg[0],du[1,2]+bg[1]))
            stress=np.zeros((8,43));stress[:,:14]=_source_fields(c,r,s)[0]
            stress[6,35:39]=np.linalg.solve(jac,np.vstack((br,bs)))[0]
            stress[7,35:39]=np.linalg.solve(jac,np.vstack((br,bs)))[1]
            result+=wr*ws*np.linalg.det(jac)*(stress.T @ (actual-reference_linear))
    return result


def independent_difference(cell,q):
    zero=q.copy();zero[24:]=0
    f,g=independent_loading(cell,q),independent_loading(cell,zero)
    return -.5*(f @ np.linalg.solve(cell.linear.stationary,f)-g @ np.linalg.solve(cell.linear.stationary,g))


def actual_assembly(model,u):
    from anyfesolver.nonlinear_static import (_prepare_initial_states,
        _prepare_qualified_q4_states_for_nonlinear_solve,_activate_nonlinear_state_storage,
        _assemble_nonlinear_system,_discard_nonlinear_state_candidate)
    zero=np.zeros(model.mesh.dof_manager.total_dofs);info={}
    states,_=_prepare_initial_states(model,None,None,5)
    states=_prepare_qualified_q4_states_for_nonlinear_solve(model,zero,states,5,info,
        supplied_element_ids=(),ordinary_restart=False)
    states=_activate_nonlinear_state_storage(model,states,5,info,kinematics='von_karman',committed_displacements=zero)
    try:
        f,k,_=_assemble_nonlinear_system(model,u,states,5,tangent=True,
            kinematics='von_karman',require_full_coordinates=True)
        return np.asarray(f).reshape(-1),k.toarray()if hasattr(k,'toarray')else np.asarray(k)
    finally:_discard_nonlinear_state_candidate(states)


def test_actual_mixed_assembly_beta_zero_preserves_all_nodal_force_and_tangent(record_property):
    model,system=small()
    u=np.zeros(system.offset.size);u[:system.physical_dofs]=np.linspace(-.001,.001,system.physical_dofs)
    expected_f,expected_k=actual_assembly(model,u[:system.physical_dofs])
    f,k,energy=system.structural_response(u)
    np.testing.assert_array_equal(f[:system.physical_dofs],expected_f)
    np.testing.assert_array_equal(k[:system.physical_dofs,:system.physical_dofs],expected_k)
    assert energy==0 and system._accepted is None
    assert np.linalg.norm(k[system.physical_dofs:,:system.physical_dofs])>0
    record_property('actual_dispatch','whole _assemble_nonlinear_system: qualified Q4 + public V2D offset=.01')


def test_nonlinear_stationary_energy_matches_independent_physical_integral(record_property):
    _,system=small();cell=system._nonlinear_cells[0]
    q=np.linspace(-.003,.004,28);q[24:]=(.004,-.003,.002,.001)
    expected=independent_difference(cell,q)
    value,force,_=cell.response(q)
    np.testing.assert_allclose(value,expected,rtol=1e-10,atol=1e-12)
    f=independent_loading(cell,q);amplitudes=cell.recover(q)
    np.testing.assert_allclose(cell.linear.stationary @ amplitudes+f,0,rtol=0,atol=1e-10)
    fields=cell.recover_fields(q,.2,-.3)
    np.testing.assert_array_equal(fields['stationary_amplitudes'],amplitudes)
    assert fields['generalized_stress'].shape==(8,) and fields['generalized_strain'].shape==(8,)
    direction=np.linspace(.2,-.3,28)
    errors=[]
    for h in (1e-6,3e-7):
        fd=(independent_difference(cell,q+h*direction)-independent_difference(cell,q-h*direction))/(2*h)
        errors.append(abs(fd-force @ direction)/max(abs(force @ direction),np.finfo(float).tiny))
    assert max(errors)<=1e-5
    record_property('independent_energy_directional_errors',json.dumps(errors))
    record_property('stationary_fields',43)


def test_all_four_bubbles_and_complete_current_normal_chain_tangent(record_property):
    _,system=small();cell=system._nonlinear_cells[0]
    q=np.linspace(-.002,.003,28);q[24:]=(.002,-.001,.0015,-.0008)
    _,_,k=cell.response(q)
    directions=[np.eye(28)[24+i]for i in range(4)]
    directions += [np.linspace(-.3,.2,28)]
    errors=[]
    for d in directions:
        expected=k @ d
        for h in (1e-6,3e-7):
            fd=(cell.response(q+h*d)[1]-cell.response(q-h*d)[1])/(2*h)
            errors.append(np.linalg.norm(fd-expected)/max(np.linalg.norm(expected),np.finfo(float).tiny))
    assert max(errors)<=1e-5
    np.testing.assert_allclose(k,k.T,rtol=1e-12,atol=1e-10)
    record_property('all_four_beta_and_complete_direction_errors',json.dumps(errors))
    reduced=np.linspace(-1e-4,2e-4,system.transform.shape[1]);z=np.full(4,.01)
    load=np.zeros(system.offset.size)
    base=system.equations(reduced,z,load)
    direction=np.linspace(-.2,.3,reduced.size+z.size)
    expected=base[4] @ direction;whole_errors=[]
    for h in (1e-7,3e-8):
        p=system.equations(reduced+h*direction[:reduced.size],z+h*direction[reduced.size:],load)
        m=system.equations(reduced-h*direction[:reduced.size],z-h*direction[reduced.size:],load)
        assert p[1].branches==m[1].branches==base[1].branches
        fd=(np.concatenate((p[2],p[3]))-np.concatenate((m[2],m[3])))/(2*h)
        whole_errors.append(np.linalg.norm(fd-expected)/np.linalg.norm(expected))
    assert max(whole_errors)<=1e-5
    record_property('actual_complete_smooth_mixed_operator_direction_errors',json.dumps(whole_errors))


def test_affine_physical_offset_map_matches_original_both_body_helpers(record_property):
    from anyfesolver.general_contact.q4_skin_work import Q4SkinWorkAdapter
    from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
    model,_=small()
    source=Q4SkinWorkAdapter.from_model(model,1,1,'bottom')
    target=S3SkinWorkAdapter.from_model(model,2,2,'top')
    # Exact original-map identity requires the ORIGINAL element-derived
    # physical offsets, not an independently rounded decimal literal.
    system=NonlinearEnrichedSystem.from_model(model,
        (PairDefinition(1,2,source.physical_offset,target.physical_offset),),
        normal_stiffness=1e4,quadrature=QuadraturePolicy(1e-10,1e-8,1e-5,14144))
    u=np.linspace(-.01,.015,system.offset.size)
    pair=system.pairs[0];x,a,h=pair.map(u)
    expected=target.current_positions(u[:system.physical_dofs])
    order=[target.node_ids.index(node)for node in pair.target_node_ids]
    np.testing.assert_allclose(x[:12].reshape(4,3),source.current_positions(u[:system.physical_dofs]),rtol=0,atol=1e-15)
    np.testing.assert_allclose(x[12:21].reshape(3,3),expected[order],rtol=0,atol=1e-15)
    full=np.zeros((37,system.offset.size));full[:,np.asarray(pair.indices)]=a
    np.testing.assert_array_equal(full[:12,:system.physical_dofs],source.work_map.toarray())
    rows=[3*j+k for j in order for k in range(3)]
    np.testing.assert_array_equal(full[12:21,:system.physical_dofs],target.work_map.toarray()[rows])
    assert not np.any(h)
    assert pair.definition.source_offset==source.physical_offset
    assert pair.definition.target_offset==target.physical_offset
    assert pair.definition.target_offset==model.mesh.elements[2].material_surface_offset_from_reference(1.0)
    assert system.definition['physical_offset_kinematics']=='ORIGINAL_AFFINE_OFFSET_EXACT_WORKMAP'
    record_property('both_body_nonzero_offset_finite_rotation_map','exact original additive workmap; not SO3')


def test_certified_paired_union_internal_ties_preserve_whole_physical_trace_and_work(record_property):
    model,system=small(paired=True,touch=False)
    u=np.zeros(system.offset.size);z=np.full(4,.01)  # exact .04N total
    # A finite current planar tilt gives a nonconstant physical gap along the
    # shared edge, so horizontal edge motion has nonzero physical work.
    for node in (5,6,7,8):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        u[dofs[2]]=.02*model.mesh.nodes[node].x
    cert=system.union_certificates(u)
    assert len(cert)==1 and cert['1']['projected_jacobian_lower_bound']>0
    raw=system.linearize(u,z)
    np.testing.assert_allclose(raw.support,.25,rtol=1e-10,atol=1e-12)
    assert np.all(raw.support>0)
    assert np.all(raw.normalized_gap>0) and np.ptp(raw.normalized_gap)>0
    hats=np.array(((1,-1,-1,1),(1,1,-1,-1),(1,1,1,1),(1,-1,1,-1)),float)
    work=(raw.support[:,None]*hats).T @ raw.gap_gradient[:,system.physical_dofs:]
    np.testing.assert_allclose(work,np.diag((4/9,4/45,4/45,4/225)),rtol=1e-8,atol=1e-10)
    errors=[]
    # The exact independent4x4 matrix above covers every beta basis. One
    # complete four-mode direction at both original h controls tests the two
    # physical sides without16 redundant complete union integrations.
    d=np.zeros(u.size);d[system.physical_dofs:]=(.02,-.03,.04,.01)
    # Move the ACTUAL projected shared edge and its physical P1 trace too,
    # exercising internal moving-domain flux cancellation at a vertex tie.
    shared=cert['1']['shared_nodes']
    for node in shared:
        dofs=system._nodal.model.mesh.dof_manager.get_node_dofs(node)
        d[np.asarray(dofs[:3])]=(.1,-.04,.02)
    target_xy=np.zeros(u.size)
    for node in shared:
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        target_xy[np.asarray(dofs[:2])]=(.1,-.04)
    target_xy_work=float(raw.force @ target_xy)
    assert target_xy_work!=0.
    for h in (1e-5,3e-6):
        p=system.linearize(u+h*d,z);m=system.linearize(u-h*d,z)
        assert p.branches!=m.branches  # actual selected internal chart changed
        pair=system.pairs[0];shared_index=pair.target_node_ids.index(shared[0])
        plus_edge=pair.map(u+h*d)[0][12:21].reshape(3,3)[shared_index,:2]
        minus_edge=pair.map(u-h*d)[0][12:21].reshape(3,3)[shared_index,:2]
        np.testing.assert_allclose(plus_edge-minus_edge,2*h*np.array((.1,-.04)),rtol=1e-8,atol=1e-15)
        for observed in (-(z @ p.normalized_gap-z @ raw.normalized_gap)/h,
                         -(z @ raw.normalized_gap-z @ m.normalized_gap)/h):
            errors.append(abs(observed-raw.force @ d)/.04)
    assert max(errors)<=1e-6
    record_property('C0_union_certificate',json.dumps(cert))
    record_property('both_sides_work_errors',json.dumps(errors))
    record_property('moving_shared_edge_direction',json.dumps(dict(nodes=shared,translation=(.1,-.04,.02),beta=(.02,-.03,.04,.01),target_z_slope_x=.02,base_gap_range=raw.normalized_gap.tolist(),target_xy_work=target_xy_work,actual_chart_change_required=True)))
    record_property('tie_matrix_claim','selected piecewise Newton matrix ONLY; no Hessian certificate')


def test_union_external_nonconforming_and_folded_domains_refuse():
    with pytest.raises(EnrichedProfileError,match='scalar elastic'):
        small(target_yield_stress=1.)
    with pytest.raises(EnrichedProfileError,match='scalar elastic'):
        small(target_hardening_curve=object())
    _,external=small(paired=True,target_half=.5)
    with pytest.raises(EnrichedGeometryError,match='clearance'):
        external.union_certificates(np.zeros(external.offset.size))
    _,nonconforming=small(paired=True,unequal_offset=True)
    with pytest.raises(EnrichedGeometryError,match='nonconforming'):
        nonconforming.union_certificates(np.zeros(nonconforming.offset.size))
    _,system=small(paired=True)
    u=np.zeros(system.offset.size);u[2]=.05;u[6+3]=.2;u[system.physical_dofs:]=1e6
    with pytest.raises(EnrichedGeometryError,match='positivity'):
        system.union_certificates(u)
    # A source edge exactly collinear with the common edge is explicitly NOT
    # covered by the vertex-only C0 argument. Construct it in data, not by
    # replacing a protected authority function or discarding a subreport.
    pairs=list(system.pairs)
    mapping=[p.map(np.zeros(system.offset.size))for p in pairs]
    common=set(pairs[0].target_node_ids)&set(pairs[1].target_node_ids)
    # Put the first source edge on the actual shared line x+y=0.
    for x,_,_ in mapping:
        x[:12]=np.array(((-.5,.5,-.025),(.5,-.5,-.025),(.5,.5,-.025),(-.5,1.5,-.025))).ravel()
    from anyfesolver.general_contact.enriched_union import certify_union
    with pytest.raises(EnrichedGeometryError,match='collinearity'):
        certify_union(pairs[0],pairs[1],mapping[0],mapping[1])


def test_nonlinear_profile_state_cancellation_refusal_and_mutation_are_atomic():
    model,system=small(fixed=True)
    criteria=MixedCriteria(.04,1e-6,1e-8)
    system.solve(np.zeros(system.offset.size),criteria=criteria,control=IterationControl(2,2))
    before=system.checkpoint();accepted=system._accepted
    assert json.loads(before)['profile']==FORMULATION
    observations=[]
    def cancel_after_actual_nodal_assembly():
        observations.append(len(observations)+1)
        # solve entry, baseline preflight, baseline post-assembly. The actual
        # FE trial must be discarded by its finally path before propagating.
        return len(observations)>=3
    with pytest.raises(SolveCancelled):
        system.solve(np.zeros(system.offset.size),criteria=criteria,control=IterationControl(2,2),
                     cancel=cancel_after_actual_nodal_assembly)
    assert observations==[1,2,3]
    assert system._accepted is accepted and system.checkpoint()==before
    malformed=json.loads(before);malformed['coordinates']=['not-a-number']
    with pytest.raises(EnrichedProfileError):
        system.restore(json.dumps(malformed,sort_keys=True,separators=(',',':')).encode(),criteria=criteria)
    nonfinite=json.loads(before);nonfinite['coordinates'][0]=float('nan')
    with pytest.raises(EnrichedProfileError):
        system.restore(json.dumps(nonfinite,sort_keys=True,separators=(',',':')).encode(),criteria=criteria)
    assert system._accepted is accepted and system.checkpoint()==before
    system.structural_response(np.zeros(system.offset.size))  # trial store is reusable
    assert system._accepted is accepted and system.checkpoint()==before
    bad=json.loads(before);bad['profile']='E4PL_PHYSICAL_B4_REFERENCE_LINEAR_PUBLIC_S3_V2'
    with pytest.raises(EnrichedProfileError,match='profile'):
        system.restore(json.dumps(bad,sort_keys=True,separators=(',',':')).encode(),criteria=criteria)
    assert system._accepted is accepted and system.checkpoint()==before
    model.mesh.nodes[1].z+=.001
    with pytest.raises(EnrichedProfileError,match='definition changed'):
        system.structural_response(np.zeros(system.offset.size))
    assert system._accepted is accepted


def test_original_matched_inventory_controls_load_and_current_nodal_dispatch(record_property):
    from examples.general_contact_enriched_matched import build
    model,system,load,descriptor=build()
    assert system._accepted is None and system.offset.size==238 and system.transform.shape==(238,184)
    assert descriptor['physical_coordinates']==174 and descriptor['added_coordinates']==64
    assert system.pressure_ids==tuple(range(1,26)) and len(system._nonlinear_cells)==16
    controls=np.asarray(descriptor['control_matrix'])
    np.testing.assert_allclose(controls @ system.transform[:174],0,rtol=0,atol=1e-14)
    assert len(controls)==2 and np.sum(load[2:174:6])==pytest.approx(-.04,rel=0,abs=1e-16)
    q=np.zeros(238);q[:174]=np.linspace(-1e-6,1e-6,174)
    expected_f,expected_k=actual_assembly(model,q[:174])
    f,k,e=system.structural_response(q)
    np.testing.assert_array_equal(f[:174],expected_f)
    np.testing.assert_array_equal(k[:174,:174],expected_k)
    assert e==0 and system._accepted is None
    record_property('original_matched_layout',json.dumps(descriptor))


def test_original_matched_nonzero_response_and_authentic_checkpoint(record_property,request,tmp_path_factory):
    from examples.general_contact_enriched_matched import build,sampled_physical_gap
    # Fresh cold owner, not a previous test's accepted state or checkpoint.
    _,system,load,descriptor=build()
    assert system._accepted is None
    criteria=MixedCriteria(.04,1e-6,1e-8)
    result=system.solve(load,criteria=criteria,control=IterationControl(18,8))
    a=system._accepted
    xmlpath=request.config.getoption('xmlpath',default=None)
    evidence=Path(xmlpath).parent if xmlpath else tmp_path_factory.mktemp('enriched_matched')
    raw_path=evidence/'MATCHED_ACCEPTED_RAW.json'
    with raw_path.open('x',encoding='utf-8')as stream:
        json.dump(dict(result,descriptor=descriptor,coordinates=a.coordinates.tolist(),
            nodal_forces=a.nodal_forces.tolist(),generation=a.generation,
            scientific_acceptance=False),stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush()
    record_property('raw_current_matched_accepted_evidence',str(raw_path))
    assert result['equilibrium']<=1e-6 and result['complementarity']<=1e-8
    raw=system.linearize(a.coordinates,a.nodal_forces)
    assert a.coordinates.shape==(238,) and a.nodal_forces.shape==(25,) and np.all(raw.support>0)
    assert np.linalg.norm(a.coordinates)>0 and np.all(a.nodal_forces>=0)
    assert np.max(a.nodal_forces)>0 and np.any(a.coordinates[174:]!=0)
    gap_witness=sampled_physical_gap(system._nodal.model,system,a.coordinates)
    record_property('independent_sampled_physical_gap',json.dumps(gap_witness))
    print(json.dumps(dict(independent_sampled_physical_gap=gap_witness)),flush=True)
    assert gap_witness['represented_source_points']==400
    assert gap_witness['original_target_plane_sample_minimum_m']>=-5e-5
    control_error=float(np.max(np.abs((np.asarray(descriptor['control_matrix']) @ a.coordinates[:174]
        -np.asarray(descriptor['control_values']))/np.asarray(descriptor['control_scales']))))
    assert control_error<=1e-6
    checkpoint=system.checkpoint();identity=system._accepted
    with pytest.raises(SolveCancelled):
        system.restore(checkpoint,criteria=criteria,cancel=lambda:True)
    assert system._accepted is identity and system.checkpoint()==checkpoint
    _,restored,_,_=build();restored.restore(checkpoint,criteria=criteria)
    assert restored.checkpoint()==checkpoint
    bad=json.loads(checkpoint);bad['definition']='foreign'
    with pytest.raises(EnrichedProfileError):
        restored.restore(json.dumps(bad,sort_keys=True,separators=(',',':')).encode(),criteria=criteria)
    assert restored.checkpoint()==checkpoint
    record_property('cold_original_matched_response',json.dumps(dict(result,descriptor=descriptor,
        coordinates=a.coordinates.tolist(),nodal_forces=a.nodal_forces.tolist(),
        normalized_gap=raw.normalized_gap.tolist(),pressure=raw.pressure.tolist(),
        support=raw.support.tolist(),element_states=system.recover_element_states(),
        reference_branch_matches=bool(a.nodal_forces[11]>0 and
            np.all(np.delete(a.nodal_forces,11)==0)),
        seven_independent_original_gates='REQUIRED_UNVERIFIED',scientific_acceptance=False)))
