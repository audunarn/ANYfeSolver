"""Independent small-motion partial-contact benchmark; development, not acceptance."""
from pathlib import Path
import json,os

import numpy as np
import pytest
from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,
    ContactSolverSettings,ContactSurface,LoadCase,solve_static_nonlinear)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def _oracle(n,k):
    # Independent closed-form Timoshenko beam blocks and exact active P1 work.
    # No production contact/structural kernel is used by this reference.
    h=1/n; EI=.01; GA=(1e6/2.6)*1e-3*(5/6)
    phi=12*EI/(GA*h*h)
    local=EI/(h**3*(1+phi))*np.array([
        [12,6*h,-12,6*h],[6*h,(4+phi)*h*h,-6*h,(2-phi)*h*h],
        [-12,-6*h,12,-6*h],[6*h,(2-phi)*h*h,-6*h,(4+phi)*h*h]])
    K=np.zeros((2*(n+1),2*(n+1)))
    for cell in range(n):
        d=np.arange(2*cell,2*cell+4);K[np.ix_(d,d)]+=local
    external=np.zeros(len(K));external[-2]=-1e-8
    free=np.arange(2,len(K)); q=np.zeros(len(K));gap=1e-8
    def contact(q):
        force=np.zeros_like(q);matrix=np.zeros_like(K);energy=0.
        for cell in range(n):
            d=np.array([2*cell,2*cell+2]);values=gap+q[d]
            cuts=[0.,1.]
            if values[0]*values[1]<0:cuts.insert(1,float(-values[0]/(values[1]-values[0])))
            for a,b in zip(cuts[:-1],cuts[1:]):
                if (1-(a+b)/2)*values[0]+(a+b)/2*values[1]>=0:continue
                for node,weight in zip((-np.sqrt(3/5),0.,np.sqrt(3/5)),(5/9,8/9,5/9)):
                    s=(a+b)/2+(b-a)/2*node;N=np.array([1-s,s]);g=N@values;w=h*(b-a)/2*weight
                    force[d]+=k*g*N*w;matrix[np.ix_(d,d)]+=k*np.outer(N,N)*w
                    energy+=.5*k*g*g*w
        return force,matrix,energy
    def merit(q):
        f,_,e=contact(q)
        return .5*q@K@q-external@q+e
    for iteration in range(60):
        force,block,energy=contact(q); residual=K@q+force-external
        if np.linalg.norm(residual[free])<=1e-20:break
        direction=np.zeros_like(q);direction[free]=np.linalg.solve((K+block)[np.ix_(free,free)],-residual[free])
        for power in range(30):
            trial=q+2.**(-power)*direction
            trial_force,_,_=contact(trial)
            trial_residual=K@trial+trial_force-external
            # At the final correction the energy difference can round away.
            # A strict independently calculated equilibrium decrease remains
            # a valid merit; the original 1e-20 N termination is unchanged.
            if (merit(trial)<=merit(q)+1e-4*2.**(-power)*(residual@direction)
                    or np.linalg.norm(trial_residual[free])<
                    (1-1e-4*2.**(-power))*np.linalg.norm(residual[free])):
                q=trial;break
        else:raise AssertionError('independent convex oracle backtracking failed')
    else:raise AssertionError('independent convex oracle did not converge')
    return q,float(-force.sum()),float(np.linalg.norm(residual[free])),energy


def _case(n,k):
    model=FEModel('small_motion_round_chain_foundation')
    model.add_material('elastic',1e6,.3,density=1.)
    section={'area':1e-3,'Iy':1e-8,'Iz':1e-8,'J':1e-8,'ky':5/6,'kz':5/6}
    for i,x in enumerate(np.linspace(0.,1.,n+1),1):model.add_node(i,x,.002+1e-8,0.)
    target_nodes=(n+2,n+3,n+4)
    for i,x in zip(target_nodes,(0.,.5,1.)):model.add_node(i,x,0.,0.)
    for i in range(1,n+1):model.add_element(i,BeamElement(i,(i,i+1),'elastic',section))
    for eid,nodes in zip((n+1,n+2),zip(target_nodes[:-1],target_nodes[1:])):
        model.add_element(eid,BeamElement(eid,nodes,'elastic',section))
    fixed={d:0. for d in ('ux','uy','uz','rx','ry','rz')}
    model.add_boundary_condition(BoundaryCondition('ground',target_nodes,fixed))
    model.add_boundary_condition(BoundaryCondition('root',(1,),fixed))
    model.add_boundary_condition(BoundaryCondition('guide',tuple(range(2,n+2)),{d:0. for d in ('ux','uz','rx','ry')}))
    definition=ContactDefinition((ContactSurface('a',tuple(range(1,n+1)),beam_radius=.001),
        ContactSurface('b',(n+1,n+2),beam_radius=.001)),(),
        (ContactInteraction('a_b','a','b',None,1e-7,line_stiffness=k),))
    settings=ContactSolverSettings(pair_motion='general',beam_chain_tangent='consistent_dev',
        line_integration_relative_tolerance=1e-7,maximum_line_subdivisions=12)
    load=LoadCase('tip');load.add_nodal_load(n+1,forces=(0.,-1e-8,0.))
    return model,definition,settings,load


def _compare(n,k):
    oracle,reaction,residual,energy=_oracle(n,k)
    model,definition,settings,load=_case(n,k)
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1e-15)
    row={'cells':n,'line_stiffness_n_per_m2':k,'status':result.status,'failure':result.failure_reason,
        'oracle_displacement':oracle.tolist(),'oracle_reaction_n':reaction,'oracle_equilibrium_n':residual,
        'oracle_energy_j':energy,'contact':result.info.get('general_contact'), 'solver_info':result.info}
    if result.status=='completed':
        actual=np.array([result.displacements[model.mesh.dof_manager.get_node_dofs(i)[d]]
            for i in range(1,n+2) for d in (1,5)])
        actual_force=result.info['general_contact']['beam_a_force'][1]
        row.update(displacement=actual.tolist(),reaction_n=actual_force,
            displacement_relative_error=float(np.linalg.norm(actual-oracle)/np.linalg.norm(oracle)),
            reaction_relative_error=float(abs(actual_force-reaction)/abs(reaction)))
        # With fixed horizontal coordinates and a straight target spanning
        # the source, the whole piecewise-linear minimum is at a source vertex.
        row['independent_minimum_gap_m']=float(np.min(.002+1e-8+actual[::2])-.002)
        row['normalized_contact_force_balance']=float(np.linalg.norm(
            np.asarray(row['contact']['beam_a_force'])+row['contact']['beam_b_force'])/1e-8)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path).with_name(f'foundation_{n}_{int(k)}.json').write_text(json.dumps(row,indent=2,default=lambda v:np.asarray(v).tolist()))
    assert result.status=='completed',result.failure_reason
    assert row['displacement_relative_error']<=1e-6,row
    assert row['reaction_relative_error']<=1e-6,row
    assert result.info['general_contact']['gap_tolerance_met']
    assert row['independent_minimum_gap_m']>=-1e-7
    assert abs(row['independent_minimum_gap_m']-row['contact']['minimum_gap'])<=1e-15
    assert row['normalized_contact_force_balance']<=1e-6
    return row


def test_independent_partial_line_foundation_discriminator():
    _compare(4,1000.)


@pytest.mark.parametrize('n,k',((2,1000.),(8,1000.),(2,10000.),(4,10000.),(8,10000.)))
def test_small_motion_mesh_and_line_penalty_reference(n,k):
    _compare(n,k)
