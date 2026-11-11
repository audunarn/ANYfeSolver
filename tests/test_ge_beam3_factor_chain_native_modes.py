"""Native V5 full-map signed modes: real states, covariance and replay."""
import json
from copy import deepcopy
import numpy as np
import pytest
import anyfesolver._ge_beam3_factor_chain_modes as adapter
from anyfesolver._ge_beam3_seeded_loaded_modal import solve_elastic_modes
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement
from anyfesolver._ge_beam3_p5.algebra import rotation as proper_rotation
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_curved_contrast_probe import make_curved, ROTATION
from test_ge_beam3_seeded_load_program import problem
from test_ge_beam3_curved_p5_mass_probe import section_mass
from test_ge_beam3_signed_loaded_spectrum import make as old_straight
from docs.reference_cases.ge_beam3_straight_discrete_signed_reference import bending_roots


def accepted(model,targets=(.5,1.),nodal=()):
    result=solve_force_program(model,ForceProgram(targets,nodal))
    assert result.status=='completed',result.failure
    states={r['element_id']:r['state'] for r in json.loads(result.checkpoint)['element_states']}
    force=np.zeros(len(result.displacements))
    for node,*vector in nodal:
        force[list(model.mesh.dof_manager.get_node_dofs(node)[:3])]=result.parameter*np.array(vector)
    return result,states,force


@pytest.mark.parametrize('slenderness',[100.,10000.,1000000.])
def test_curved_coupled_modes_covary_under_signed_permutation_and_general_rotation(slenderness,tmp_path):
    made=[]
    for name,q in [('E',np.eye(3)),('R90',ROTATION),('GENERAL',proper_rotation([.4,-.3,.2]))]:
        model,inertias=make_curved(slenderness,q)
        states={i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1) for i,e in model.mesh.elements.items()}
        before=canonical(states)
        packet,result=adapter.solve_signed_loaded_modes(model,states,np.zeros(30),inertias,np.zeros(30),
            load_parameter=0.,bounds=(-10000.,100000000.))
        assert canonical(states)==before
        with (tmp_path/(name+'.json')).open('xb') as stream: stream.write(canonical(result))
        made.append((packet,result,q))
    first=made[0]
    for packet,result,q in made[1:]:
        np.testing.assert_allclose(result.eigenvalues,first[1].eigenvalues,rtol=1e-11,atol=1e-11)
        transform=np.kron(np.eye(14),q)
        correlation=(transform@first[1].full_modes).T@packet.mass@result.full_modes
        np.testing.assert_allclose(np.abs(correlation),np.eye(6),rtol=1e-11,atol=1e-11)


@pytest.mark.parametrize('axial',[-1.,.5])
def test_actual_slender_prestressed_modes_match_rational_reference(axial,tmp_path):
    model,inertia=old_straight(1e6)
    for i,old in tuple(model.mesh.elements.items()):
        new=NativeP5BeamElement(i,old.node_ids,old.core.reference,old.core.section,line_force=old.core.line_force,order=old.core.order)
        model.mesh.elements[i]=new; model.materials[new.material_name]=new.core.section
    result,states,force=accepted(model,nodal=((3,axial,0.,0.),)); before=canonical(states)
    packet,modes=adapter.solve_signed_loaded_modes(model,states,result.displacements,{1:inertia},force,
        load_parameter=1.,bounds=(-10.,100.),num_modes=4)
    expected=np.repeat(bending_roots(1000000,axial),2)
    np.testing.assert_allclose(modes.eigenvalues,expected,rtol=1e-11,atol=1e-9)
    assert np.array_equal(np.sign(modes.eigenvalues),np.sign(expected))
    np.testing.assert_allclose(modes.full_modes.T@packet.mass@modes.full_modes,np.eye(4),rtol=1e-11,atol=1e-11)
    assert canonical(states)==before and not modes.production_qualified and not modes.buckling_factor_authorized
    with (tmp_path/'prestressed.json').open('xb') as stream: stream.write(canonical(modes))


@pytest.fixture(scope='module')
def loaded():
    model=problem(); result,states,force=accepted(model,nodal=((55,.025,-.01,.005),))
    return model,result,states,force


def solve_loaded(model,result,states,force,**kwargs):
    return adapter.solve_signed_loaded_modes(model,states,result.displacements,
        {i:section_mass() for i in model.mesh.elements},force,load_parameter=result.parameter,
        bounds=(-10000.,1000000.),**kwargs)


def test_loaded_curved_coupled_modes_match_dense_moderate_reference_and_replay(loaded,tmp_path):
    model,result,states,force=loaded; before=canonical(states)
    packet,modes=solve_loaded(*loaded)
    old_packet,old=solve_elastic_modes(model,states,result.displacements,{1:section_mass()},force,load_parameter=1.)
    assert packet.identity==old_packet.identity==modes.operator_identity
    np.testing.assert_allclose(modes.eigenvalues,old.eigenvalues,rtol=1e-11,atol=1e-10)
    np.testing.assert_allclose(np.abs(old.full_modes.T@packet.mass@modes.full_modes),np.eye(6),rtol=1e-11,atol=1e-11)
    clone=problem(); replay=solve_force_program(clone,ForceProgram((.5,1.),((55,.025,-.01,.005),)),checkpoint=result.checkpoint)
    assert replay.status=='completed' and replay.checkpoint==result.checkpoint
    assert canonical(solve_loaded(clone,replay,states,force))==canonical((packet,modes))
    assert canonical(states)==before
    # Numerical representation only: prestress/load Hessians remain exactly
    # those of the preserved source split, not a positive-reference fallback.
    from anyfesolver._ge_beam3_seeded_signed_modes import _split as original_split
    from anyfesolver._ge_beam3_shared_kinematic_split import shared_kinematic_chain
    for eid,e in model.mesh.elements.items():
        decoded=e.validate_model_bound_nonlinear_state(model.mesh,e.core.section,states[eid],1)
        old_factor,old_g=original_split(e,decoded['material_state'],lambda stage:None)
        left,right,new_g=shared_kinematic_chain(e,decoded['material_state'])
        assert np.array_equal(old_g,new_g)
        assert np.linalg.norm(left@right-old_factor)<=1e-11*np.linalg.norm(old_factor)
    with (tmp_path/'loaded.json').open('xb') as stream: stream.write(canonical(modes))


@pytest.mark.parametrize('mutation',['force','moment','state','parameter'])
def test_invalid_load_state_is_rejected_before_kernel(loaded,monkeypatch,mutation):
    model,result,states,force=loaded; states=deepcopy(states); force=force.copy(); parameter=1.
    if mutation=='force': force[12]+=.1
    elif mutation=='moment': force[15]=.1
    elif mutation=='state': states={}
    else: parameter=.5
    monkeypatch.setattr(adapter,'solve_factor_chain_modes',lambda *a,**k:pytest.fail('invalid kernel entry'))
    with pytest.raises(ValueError): adapter.solve_signed_loaded_modes(model,states,result.displacements,
        {1:section_mass()},force,load_parameter=parameter,bounds=(-10000.,1000000.))


def test_cancel_and_mutated_model_are_fail_closed(loaded,monkeypatch):
    token=CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled): solve_loaded(*loaded,cancellation_token=token)
    model,_,_,_=loaded; original=adapter.solve_factor_chain_modes; count=model.mesh.dof_manager.total_dofs
    def mutate(*a,**k):
        output=original(*a,**k); model.mesh.dof_manager._total_dofs=count+1; return output
    monkeypatch.setattr(adapter,'solve_factor_chain_modes',mutate)
    try:
        with pytest.raises(ValueError,match='changed'): solve_loaded(*loaded)
    finally: model.mesh.dof_manager._total_dofs=count


def test_active_plastic_branch_rejected_and_elastic_unloading_retains_history():
    model=problem(plastic=True); data=accepted(model); before=canonical(data[1])
    with pytest.raises(ValueError,match='plastic or yield-boundary'): solve_loaded(model,*data)
    assert canonical(data[1])==before
    model=problem(plastic=True); data=accepted(model,targets=(.5,1.,0.)); before=canonical(data[1])
    packet,modes=solve_loaded(model,*data)
    assert all(op.elastic_interior for _,op in packet.operators)
    e=model.mesh.elements[1]; state=e.validate_model_bound_nonlinear_state(model.mesh,e.core.section,data[1][1],1)
    assert any(h.accumulated>0 for h in state['material_state']['histories'])
    assert canonical(data[1])==before and np.all(modes.eigenvalues>0)


def test_free_curved_member_has_six_rigid_modes_and_positive_seventh():
    model,inertias=make_curved(100.)
    model.boundary_conditions.clear()
    states={i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1) for i,e in model.mesh.elements.items()}
    packet,modes=adapter.solve_signed_loaded_modes(model,states,np.zeros(30),inertias,np.zeros(30),
        load_parameter=0.,bounds=(-10000.,100000000.),num_modes=7)
    np.testing.assert_allclose(modes.eigenvalues[:6],0.,atol=1e-11)
    assert modes.eigenvalues[6]>1.
    rigid=np.zeros((42,6))
    for i,node in model.mesh.nodes.items():
        dofs=list(model.mesh.dof_manager.get_node_dofs(i)); rigid[dofs[:3],:3]=np.eye(3)
        rigid[dofs[3:],3:]=np.eye(3)
        rigid[dofs[:3],3:]=np.column_stack([np.cross(axis,node.coords()) for axis in np.eye(3)])
    for _,internal in packet.internal_layout:
        for offset in (0,3): rigid[list(internal[offset:offset+3]),3:]=np.eye(3)
    gram=rigid.T@packet.mass@rigid; q=rigid@np.linalg.inv(np.linalg.cholesky(gram).T)
    cross=q.T@packet.mass@modes.full_modes[:,:6]
    np.testing.assert_allclose(cross@cross.T,np.eye(6),rtol=1e-11,atol=1e-11)


def test_finitely_loaded_curved_contrast_modes_covary_and_replay(tmp_path):
    outputs=[]
    for name,q in [('E',np.eye(3)),('GENERAL',proper_rotation([.4,-.3,.2]))]:
        model,inertias=make_curved(100.,q)
        vector=tuple(float(v) for v in q@np.array([.05,-.001,0.]))
        nodal=((5,*vector),); result,states,force=accepted(model,nodal=nodal)
        saved=canonical(states)
        packet,modes=adapter.solve_signed_loaded_modes(model,states,result.displacements,inertias,force,
            load_parameter=1.,bounds=(-10000.,100000000.))
        clone,clone_inertias=make_curved(100.,q)
        replay=solve_force_program(clone,ForceProgram((.5,1.),nodal),checkpoint=result.checkpoint)
        assert replay.status=='completed' and replay.checkpoint==result.checkpoint
        again=adapter.solve_signed_loaded_modes(clone,states,replay.displacements,clone_inertias,force,
            load_parameter=1.,bounds=(-10000.,100000000.))
        assert canonical(again)==canonical((packet,modes)) and canonical(states)==saved
        with (tmp_path/(name+'.json')).open('xb') as stream:
            stream.write(canonical(dict(modes=modes,accepted_program=json.loads(result.checkpoint))))
        outputs.append((packet,modes,q))
    a,b=outputs; transform=np.kron(np.eye(14),b[2])
    np.testing.assert_allclose(a[1].eigenvalues,b[1].eigenvalues,rtol=1e-11,atol=1e-11)
    cross=(transform@a[1].full_modes).T@b[0].mass@b[1].full_modes
    np.testing.assert_allclose(np.abs(cross),np.eye(6),rtol=1e-11,atol=1e-11)


def test_loaded_two_element_spectrum_is_byte_identical_under_large_translation(tmp_path):
    outputs=[]
    for shift in (0.,2.**40):
        model=problem(count=2,shift=shift); result,states,force=accepted(model)
        _,modes=solve_loaded(model,result,states,force)
        outputs.append(canonical(dict(eigenvalues=modes.eigenvalues,modes=modes.full_modes,
            residual=modes.spectral_residual,original_ritz=modes.original_ritz_residual)))
    assert outputs[0]==outputs[1]
    with (tmp_path/'translation.json').open('xb') as stream: stream.write(outputs[0])


@pytest.mark.parametrize('batched',[False,True])
def test_extreme_curved_modes_preserve_covariance_for_both_coordinate_orders(batched,tmp_path):
    from docs.reference_cases.ge_beam3_reassembly_runtime_diagnostic import contrast
    results=[]
    for case,q in [('E',np.eye(3)),('GENERAL',proper_rotation([.4,-.3,.2]))]:
        model,inertias=contrast(1e6,q,batched)
        states={i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1)
            for i,e in model.mesh.elements.items()}
        packet,modes=adapter.solve_signed_loaded_modes(model,states,np.zeros(30),inertias,np.zeros(30),
            load_parameter=0.,bounds=(-10000.,100000000.))
        with (tmp_path/(case+'.json')).open('xb') as stream: stream.write(canonical(modes))
        results.append((packet,modes,q))
    a,b=results;transform=np.kron(np.eye(14),b[2])
    np.testing.assert_allclose(a[1].eigenvalues,b[1].eigenvalues,rtol=1e-11,atol=1e-11)
    cross=(transform@a[1].full_modes).T@b[0].mass@b[1].full_modes
    np.testing.assert_allclose(np.abs(cross),np.eye(6),rtol=1e-11,atol=1e-11)
