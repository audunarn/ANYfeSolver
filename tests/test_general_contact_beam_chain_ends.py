"""Independent multi-cell bar response and endpoint/junction work checks."""
from dataclasses import replace
from pathlib import Path
import json,os

import numpy as np
import pytest
from scipy.optimize import root

from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,
    ContactSurface,ContactSolverSettings,LoadCase,DisplacementControl,
    solve_static_nonlinear,validate_general_contact,report_general_contact_capability)
from anyfesolver.elements import BeamElement,QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_chain_ends import BeamChainEndManager,PROFILE


def bars(cells=2,other=None):
    other=cells if other is None else other
    model=FEModel('coupled physical B2 chain ends');model.add_material('elastic',1e3,.3,density=1.)
    left=np.linspace(-.212,-.012,cells+1);right=np.linspace(.012,.212,other+1)
    for node,x in enumerate(np.r_[left,right],1):model.add_node(node,x,0.,0.)
    section={'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}
    groups=(tuple(range(1,cells+1)),tuple(range(cells+1,cells+other+1)))
    nodes=(tuple(range(1,cells+2)),tuple(range(cells+2,cells+other+3)))
    for group,lane in zip(groups,nodes):
        for element,a,b in zip(group,lane[:-1],lane[1:]):
            model.add_element(element,BeamElement(element,(a,b),'elastic',section))
    model.add_boundary_condition(BoundaryCondition('roots',(nodes[0][0],nodes[1][-1]),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('axial guides',(*nodes[0][1:],*nodes[1][:-1]),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    definition=ContactDefinition(tuple(ContactSurface(name,group,beam_radius=.01,
        beam_end_caps='hemisphere') for name,group in zip(('a','b'),groups)),(),
        (ContactInteraction('pair','a','b',None,1e-5,line_stiffness=1e4,endpoint_stiffness=1e5),))
    settings=ContactSolverSettings(pair_motion='general',beam_chain_tangent='consistent_dev',
        line_integration_relative_tolerance=1e-7,maximum_line_subdivisions=14,maximum_swept_subdivisions=16)
    load=LoadCase('symmetric closure');load.add_nodal_load(nodes[0][-1],forces=(.2,0.,0.))
    load.add_nodal_load(nodes[1][0],forces=(-.2,0.,0.))
    return model,definition,settings,load,nodes


def independent_bar_solution(left,right):
    # Own P1 axial strain energy and analytically integrated endpoint line work.
    x=np.r_[left,right];n=len(left);free=np.r_[np.arange(1,n),np.arange(n,len(x)-1)]
    F=np.zeros(len(x));F[n-1]=.2;F[n]=-.2
    def residual(v):
        u=np.zeros(len(x));u[free]=v;grad=np.zeros(len(x))
        for lane in (np.arange(n),np.arange(n,len(x))):
            for i,j in zip(lane[:-1],lane[1:]):
                force=(u[j]-u[i])/(x[j]-x[i]);grad[i]-=force;grad[j]+=force
        delta=max(.02-((x[n]+u[n])-(x[n-1]+u[n-1])),0.)
        grad[n-1]+=1e5*delta;grad[n]-=1e5*delta
        for i,j in ((n-2,n-1),(n,n+1)):
            length=x[j]-x[i];D=length+u[j]-u[i];C=1e4*length/12
            common=3*C*delta**2/D
            grad[n-1]+=common;grad[n]-=common
            grad[i]+=C*delta**3/D**2;grad[j]-=C*delta**3/D**2
        return (grad-F)[free]
    u0=np.r_[np.linspace(0.,.0021,n),np.linspace(-.0021,0.,len(right))]
    solution=root(residual,u0[free],tol=1e-11)
    assert solution.success and np.linalg.norm(residual(solution.x))<1e-9
    u=np.zeros(len(x));u[free]=solution.x
    return u


@pytest.mark.parametrize('cells,other',((2,2),(4,3),(8,8)))
def test_actual_nonmatching_two_bar_force_response_and_physical_cap_count(cells,other):
    model,definition,settings,load,nodes=bars(cells,other)
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1e-10,emit_restart_checkpoint=True)
    assert result.status=='completed',result.failure_reason
    reference=independent_bar_solution(np.linspace(-.212,-.012,cells+1),np.linspace(.012,.212,other+1))
    dofs=[model.mesh.dof_manager.get_node_dofs(node)[0] for node in (*nodes[0],*nodes[1])]
    np.testing.assert_allclose(result.displacements[dofs],reference,rtol=1e-6,atol=1e-10)
    info=result.info['general_contact'];assert info['active_endpoints']==1 and info['candidate_endpoints']==4
    assert info['gap_tolerance_met'] and info['profile']==PROFILE
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name(f'bar_{cells}_{other}.json').write_text(json.dumps({
        'independent_displacement_m':reference.tolist(),'solved_displacement_m':result.displacements[dofs].tolist(),
        'contact':info},indent=2))


def test_split_active_restart_displacement_and_changed_topology_settings_law():
    model,definition,settings,load,nodes=bars(2,3)
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=25,
        tolerance=1e-10,emit_restart_checkpoint=True)
    full=solve_static_nonlinear(model,load,**args);assert full.status=='completed',full.failure_reason
    model,_,_,_,_=bars(2,3)
    prefix=solve_static_nonlinear(model,load,**{**args,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_,_=bars(2,3)
    resumed=solve_static_nonlinear(fresh,load,**{**args,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1e-6,atol=1e-10)
    changed=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=1.01e5),))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**args,'contact':changed},restart_checkpoint=checkpoint)
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**args,'contact_settings':replace(settings,maximum_swept_subdivisions=15)},restart_checkpoint=checkpoint)
    joined,_,_,_,_=bars(2,3);node=nodes[1][0]
    joined.add_node(20,.012,-.1,0.);joined.add_node(21,.012,.1,0.)
    joined.add_element(20,QuadraticBeamElement(20,(20,node,21),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(joined,load,**args,restart_checkpoint=checkpoint)
    controlled,_,_,_,_=bars(2,3);end=nodes[0][-1]
    result=solve_static_nonlinear(controlled,load,**args,control='displacement',
        displacement_control=DisplacementControl(node_id=end,dof='ux',target_displacement=
            full.displacements[model.mesh.dof_manager.get_node_dofs(end)[0]]))
    assert result.status=='completed',result.failure_reason
    np.testing.assert_allclose(result.displacements,full.displacements,rtol=1e-6,atol=1e-10)


def test_sparse_both_body_point_tangent_and_complete_energy_with_chain_line_work():
    model,definition,settings,_,nodes=bars(2,3)
    manager=BeamChainEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for lane,end in ((nodes[0],.0021),(tuple(reversed(nodes[1])),-.0021)):
        for node,v in zip(lane,np.linspace(0.,end,len(lane))):u[model.mesh.dof_manager.get_node_dofs(node)[0]]=v
    f,K,r=manager.evaluate(u)
    assert r.active_endpoints==1 and r.endpoint_contact_energy==pytest.approx(.002,rel=1e-10)
    direction=np.zeros_like(u)
    for node,v in zip(model.mesh.nodes,np.random.default_rng(4001).normal(size=(len(model.mesh.nodes),3))):
        direction[model.mesh.dof_manager.get_node_dofs(node)[:3]]=v*.2
    for h in (1e-7,3e-8):
        fp,_,rp=manager.evaluate(u+h*direction,tangent=False)
        fm,_,rm=manager.evaluate(u-h*direction,tangent=False)
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
        assert (rp.contact_energy-rm.contact_energy)/(2*h)==pytest.approx(f@direction,rel=1e-5,abs=1e-8)
    np.testing.assert_allclose(K.toarray(),K.toarray().T,atol=1e-9)
    assert np.linalg.norm(np.array(r.beam_a_force)+r.beam_b_force)<1e-6
    assert np.linalg.norm(np.array(r.beam_a_moment)+r.beam_b_moment)<1e-6


def test_public_admission_unsupported_tangent_and_candidate_resource_fail_closed():
    model,definition,settings,_,_=bars(2,2)
    report=report_general_contact_capability(model,definition,settings)
    assert report.validation.profile==PROFILE and report.implemented and not report.admitted_for_application
    for changed in (replace(settings,beam_chain_tangent='finite_difference',sliding_difference_step=1e-7),
                    replace(settings,maximum_candidate_pairs=3)):
        assert not validate_general_contact(model,definition,changed).valid
        with pytest.raises(ValueError):BeamChainEndManager(model,definition,changed)


def junction():
    model,definition,settings,_,nodes=bars(2,2)
    for node,x in zip(nodes[0],np.linspace(-.224,-.024,3)):model.mesh.nodes[node].x=x
    for node,point in zip(nodes[1],((.2,-.1,0.),(0.,0.,0.),(.2,.1,0.))):
        model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z=point
    manager=BeamChainEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(nodes[0][-1])[0]]=.0042
    return model,definition,settings,manager,u,nodes


def test_shared_target_junction_has_one_point_map_and_both_body_hessian():
    model,definition,settings,manager,u,nodes=junction()
    f,K,E,fields,count=manager._endpoint_work(manager._positions(u),True)
    active=[p for p in fields if p['state']=='active']
    assert count==1 and active[0]['feature_kind']=='end_junction'
    assert active[0]['target_node_id']==nodes[1][1] and len(active[0]['target_element_ids'])==2
    assert E==pytest.approx(.002,rel=1e-10)
    for node in (nodes[1][0],nodes[1][-1]):np.testing.assert_array_equal(f[model.mesh.dof_manager.get_node_dofs(node)],0.)
    direction=np.zeros_like(u)
    for node,v in zip(model.mesh.nodes,np.random.default_rng(770).normal(size=(len(model.mesh.nodes),3))):
        direction[model.mesh.dof_manager.get_node_dofs(node)[:3]]=v*.2
    h=1e-7
    fp,_,ep,_,_=manager._endpoint_work(manager._positions(u+h*direction),False)
    fm,_,em,_,_=manager._endpoint_work(manager._positions(u-h*direction),False)
    assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
    assert (ep-em)/(2*h)==pytest.approx(f@direction,rel=1e-5)
    zero=np.zeros_like(u);small=u*(.004005/.0042)
    assert manager.swept_increment_admissible(zero,small,0.,1.)
    # Complete line + point evaluation also resolves the shared node without duplication.
    total=manager.evaluate(u)[2]
    assert total.active_endpoints==1 and total.endpoint_contact_energy==pytest.approx(E,rel=1e-12)


def test_body_exchange_cell_order_connectivity_and_material_junction_invariance():
    model,definition,settings,manager,u,_=junction()
    f,K,r=manager.evaluate(u)
    swapped=replace(definition,surfaces=tuple(reversed(definition.surfaces)),
        interactions=(replace(definition.interactions[0],surface='b',target='a'),))
    ef,eK,er=BeamChainEndManager(model,swapped,settings).evaluate(u)
    np.testing.assert_allclose(ef,f,atol=1e-9);np.testing.assert_allclose(eK.toarray(),K.toarray(),atol=1e-8)
    assert er.contact_energy==pytest.approx(r.contact_energy,rel=1e-12)
    for cell in model.mesh.elements.values():cell.node_ids=tuple(reversed(cell.node_ids))
    reordered=replace(definition,surfaces=tuple(replace(s,element_ids=tuple(reversed(s.element_ids))) for s in definition.surfaces))
    rf,rK,rr=BeamChainEndManager(model,reordered,settings).evaluate(u)
    np.testing.assert_allclose(rf,f,atol=1e-9);np.testing.assert_allclose(rK.toarray(),K.toarray(),atol=1e-8)
    assert rr.contact_energy==pytest.approx(r.contact_energy,rel=1e-12)


def test_nonphysical_selected_cut_and_nonreciprocal_cap_pairs_fail_explicitly():
    model,definition,settings,_,nodes=bars(2,2)
    node=nodes[1][0];model.add_node(20,.012,-.1,0.);model.add_node(21,.012,.1,0.)
    model.add_element(20,QuadraticBeamElement(20,(20,node,21),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    manager=BeamChainEndManager(model,definition,settings);u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(nodes[0][-1])[0]]=.0021
    u[model.mesh.dof_manager.get_node_dofs(node)[0]]=-.0021
    with pytest.raises(ValueError,match='nonphysical'):manager.evaluate(u)
    model,definition,settings,_,_=bars(3,1)
    manager=BeamChainEndManager(model,definition,settings)
    points=np.array(((.0197,-.1,0.),(.0197,.1,0.),(-.1,0.,0.),(0.,0.,0.)))
    source=tuple(points[i:i+2] for i in range(3));target=(np.array(((.0198,0.,0.),(.3,0.,0.))),)
    with pytest.raises(ValueError,match='reciprocal'):
        manager._endpoint_work((source,target),True)


def test_nearest_path_hidden_distinct_partner_switch_and_unrepresented_cut():
    from fractions import Fraction as F
    from anyfesolver.general_contact.beam_chain_end_geometry import (
        nearest_endpoint_feature,admissible_chain_endpoint_path)
    targets=(((F(-1,100),-1,0),(F(-1,100),1,0)),
             ((F(1,100),-1,0),(F(1,100),1,0)))
    nodes=((10,11),(12,13));source0=((0,0,1),(F(-1,1000),0,F(1,200)))
    source1=((0,0,1),(F(1,1000),0,F(1,200)))
    with pytest.raises(ValueError,match='ties'):
        nearest_endpoint_feature(((0,0,1),(0,0,F(1,200))),targets,nodes,1,F(1,50))
    assert not admissible_chain_endpoint_path(source0,source1,targets,targets,nodes,1,F(1,50),8)
    # Same-node clamps are physically identical; mismatched coordinates do not acquire that identity.
    capsource=((-1,0,0),(0,0,0));joint=(((F(1,100),0,0),(1,1,0)),
        ((1,-1,0),(F(1,100),0,0)))
    choice=nearest_endpoint_feature(capsource,joint,((7,8),(9,7)),1,F(1,50))
    assert choice.target_node==7 and len(choice.tied_owners)==2
    assert admissible_chain_endpoint_path(capsource,capsource,joint,joint,((7,8),(9,7)),1,F(1,50),0,
        allowed_target_nodes={7})
    assert not admissible_chain_endpoint_path(capsource,capsource,joint,joint,((7,8),(9,7)),1,F(1,50),4,
        allowed_target_nodes=set())


def test_swept_activation_rejects_nonreciprocal_cap_features():
    model,definition,settings,_,nodes=bars(2,2)
    points=((0.,.002,0.),(-2.,0.,0.),(0.,0.,0.),
            (.1,1.01,0.),(.1,3.01,0.),(.1,4.01,0.))
    for node,p in zip(model.mesh.nodes.values(),points):node.x,node.y,node.z=p
    definition=replace(definition,surfaces=tuple(replace(s,beam_radius=.5005) for s in definition.surfaces),
        interactions=(replace(definition.interactions[0],gap_tolerance=.01),))
    manager=BeamChainEndManager(model,definition,settings)
    first=np.zeros(model.mesh.dof_manager.total_dofs);last=first.copy()
    for node in nodes[1]:last[model.mesh.dof_manager.get_node_dofs(node)[1]]=-.015
    assert manager.evaluate(first)[2].active_endpoints==0
    assert manager._minimum_gap(last)>=-.01
    with pytest.raises(ValueError,match='reciprocal'):manager.evaluate(last)
    assert not manager.swept_increment_admissible(first,last,0.,1.)
    contract=manager.restart_contract();fraction=manager.admissible_step_fraction(first,last)
    assert 0.<fraction<1.
    assert manager.evaluate(first+fraction*(last-first))[2].gap_tolerance_met
    assert manager.restart_contract()==contract and manager.evaluate(first)[2].contact_energy==0.


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_two_chain_candidate_exception_or_cancel_keeps_origin_and_restart(cancel,monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled
    model,definition,settings,load,_=bars(2,2)
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=25,
        tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False];original=BeamChainEndManager.evaluate
    def interrupt(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.5:
            if cancel:token.cancel('chain end interruption')
            else:armed[0]=True
    def fail_after_candidate(self,*values,**options):
        candidate=original(self,*values,**options)
        if armed[0] and not raised[0]:
            raised[0]=True
            raise RuntimeError('chain end interruption')
        return candidate
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(BeamChainEndManager,'evaluate',fail_after_candidate)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='chain end interruption'):
            solve_static_nonlinear(model,load,**args,progress_callback=interrupt,cancellation_token=token)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**args)
    fresh,_,_,_,_=bars(2,2);reference=solve_static_nonlinear(fresh,load,**args)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_junction_work_is_covariant_under_common_rigid_transform():
    model,definition,settings,manager,u,_=junction()
    f,K,r=manager.evaluate(u)
    angle=.39;Q=np.array(((np.cos(angle),-np.sin(angle),0.),(np.sin(angle),np.cos(angle),0.),(0.,0.,1.)))
    shift=np.array((.7,-.4,.2));v=np.zeros_like(u)
    for node in model.mesh.nodes.values():
        node.x,node.y,node.z=Q@node.coords()+shift
        dofs=model.mesh.dof_manager.get_node_dofs(node.id)[:3];v[dofs]=Q@u[dofs]
    tf,tK,tr=BeamChainEndManager(model,definition,settings).evaluate(v)
    assert tr.contact_energy==pytest.approx(r.contact_energy,rel=1e-10)
    direction=np.zeros_like(u);rotated_direction=np.zeros_like(u)
    for node,z in zip(model.mesh.nodes,np.random.default_rng(884).normal(size=(len(model.mesh.nodes),3))):
        dofs=model.mesh.dof_manager.get_node_dofs(node)[:3]
        direction[dofs]=z;rotated_direction[dofs]=Q@z
        np.testing.assert_allclose(tf[dofs],Q@f[dofs],rtol=1e-10,atol=1e-10)
    original=K@direction;transformed=tK@rotated_direction
    for node in model.mesh.nodes:
        dofs=model.mesh.dof_manager.get_node_dofs(node)[:3]
        np.testing.assert_allclose(transformed[dofs],Q@original[dofs],rtol=1e-10,atol=1e-8)


def test_public_nonmatching_chain_endpoint_example():
    from examples.general_contact_beam_chain_ends import run
    result=run()
    assert result.status=='completed' and result.info['general_contact']['profile']==PROFILE
    assert result.info['general_contact']['active_endpoints']==1


def test_moving_line_owner_interface_and_endpoint_work_keep_all_cross_blocks():
    from test_general_contact_beam_chain_consistent import _v_case,_energy_oracle
    model,definition,_,old,u=_v_case()
    definition=replace(definition,surfaces=tuple(replace(s,beam_end_caps='hemisphere') for s in definition.surfaces),
        interactions=(replace(definition.interactions[0],endpoint_stiffness=1e5),))
    manager=BeamChainEndManager(model,definition,old.settings)
    f,K,r=manager.evaluate(u)
    assert manager.last_moving_owner_interfaces>0 and r.active_endpoints==2
    # Independent scalar cap potential on the fixture's known physical nodes.
    positions={n:np.array(model.mesh.nodes[n].coords())+u[model.mesh.dof_manager.get_node_dofs(n)[:3]]
               for n in model.mesh.nodes}
    E=0.
    for node,adjacent,targets in ((1,2,((4,5),(5,6))),(3,2,((4,5),(5,6))),
                                (4,5,((1,2),(2,3))),(6,5,((1,2),(2,3)))):
        point=positions[node];outward=point-positions[adjacent];choices=[]
        for a,b in targets:
            v=positions[b]-positions[a];t=np.clip((point-positions[a])@v/(v@v),0.,1.)
            d=point-(positions[a]+t*v);choices.append(d)
        d=min(choices,key=np.linalg.norm);g=np.linalg.norm(d)-.02
        if g<0. and d@outward<0.:E+=.5*1e5*g*g
    assert r.contact_energy==pytest.approx(_energy_oracle(manager,u)+E,rel=2e-7)
    direction=np.zeros_like(u)
    for node,v in zip(model.mesh.nodes,np.random.default_rng(415).normal(size=(len(model.mesh.nodes),3))):
        direction[model.mesh.dof_manager.get_node_dofs(node)[:3]]=v*.1
    for h in (1e-7,3e-8):
        fp=manager.evaluate(u+h*direction,tangent=False)[0]
        fm=manager.evaluate(u-h*direction,tangent=False)[0]
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
