"""Integrated development checks for the explicitly bounded moving-normal profile."""
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import root

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface, ContactInteraction,
                       ContactSolverSettings, LoadCase, create_shell_element,
                       solve_static_nonlinear, validate_general_contact)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement
from anyfesolver.general_contact.deformable_beam_s3_static import DeformableBeamS3Manager, PROFILE
from anyfesolver.matrix_assembly import assemble_stiffness_matrix


def fixture():
    model=FEModel('deformable_line_skin_dev')
    model.add_material('beam',1.e3,.3,density=1.)
    model.add_material('skin',1.e5,.3,density=1.)
    for node,p in enumerate(((0.,0.,.02005),(.5,0.,.02005),(-1.,-1.,0.),(2.,-1.,0.),(-1.,2.,0.)),1):
        model.add_node(node,*p)
    model.add_element(1,BeamElement(1,[1,2],'beam',{'area':1.e-3,'Iy':1.e-4,'Iz':1.e-4,'J':1.e-4}))
    model.add_element(2,create_shell_element(2,(3,4,5),'skin',thickness=.02,
                      formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=0.))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01),
                         ContactSurface('skin',(2,),side='top')),(),
                         (ContactInteraction('line','tube','skin',None,1.e-5,line_stiffness=1.e5),))
    settings=ContactSolverSettings(pair_motion='general',beam_shell_rim_clearance=.02,
                     line_integration_relative_tolerance=1.e-8,
                     beam_shell_geometry='jet2_deformable_s3_dev',beam_shell_maximum_swept_intervals=128)
    return model,definition,settings


def constraints(model):
    names=('ux','uy','uz','rx','ry','rz')
    model.add_boundary_condition(BoundaryCondition('fixed',[1,3,4],dict.fromkeys(names,0.)))
    model.add_boundary_condition(BoundaryCondition('guides',[2,5],dict.fromkeys(('ux','uy','rx','ry','rz'),0.)))


def test_complete_sparse_work_map_deforming_target_and_rejected_candidate():
    m,d,s=fixture(); assert validate_general_contact(m,d,s).valid
    manager=DeformableBeamS3Manager(m,d,s)
    u=np.zeros(m.mesh.dof_manager.total_dofs)
    dof=lambda node,axis:m.mesh.dof_manager.get_node_dofs(node)[axis]
    u[dof(2,2)]=-.0004;u[dof(5,2)]=-.0002
    r,K,summary=manager.evaluate(u)
    assert K.format=='csr' and summary.active_line_length>0
    direction=np.zeros_like(u);direction[dof(2,2)]=.3;direction[dof(5,2)]=-.2
    for h in (1.e-6,3.e-7):
        rp,_,sp=manager.evaluate(u+h*direction);rm,_,sm=manager.evaluate(u-h*direction)
        numerical=(rp-rm)/(2*h)
        assert np.linalg.norm(numerical-K@direction)/np.linalg.norm(numerical)<1.e-5
        derivative=(sp.line_contact_energy-sm.line_contact_energy)/(2*h)
        assert abs(derivative-r@direction)/abs(derivative)<1.e-5
    np.testing.assert_allclose(np.array(summary.beam_force)+summary.shell_force,0.,atol=1.e-12)
    np.testing.assert_allclose(np.array(summary.beam_moment)+summary.shell_moment,0.,atol=1.e-12)
    bad=u.copy();bad[dof(5,3)]=1.e-7
    with pytest.raises(ValueError,match='rotational'):manager.evaluate(bad)
    rr,KK,ss=manager.evaluate(u)
    np.testing.assert_array_equal(rr,r);np.testing.assert_array_equal(KK.toarray(),K.toarray());assert ss==summary
    clear=u.copy();clear[dof(2,2)]=0.;clear[dof(5,2)]=-1.e-5
    assert manager.swept_increment_admissible(np.zeros_like(u),clear,0.,1.)
    with pytest.raises(ValueError,match='initial penetration'):DeformableBeamS3Manager(m,d,s,u)


def test_profile_admission_and_restart_settings_are_explicit():
    m,d,s=fixture();manager=DeformableBeamS3Manager(m,d,s)
    assert manager.profile==PROFILE
    assert manager.restart_contract()!=DeformableBeamS3Manager(m,d,replace(s,beam_shell_maximum_swept_intervals=127)).restart_contract()
    assert not validate_general_contact(m,d,replace(s,beam_shell_maximum_swept_intervals=None)).valid
    from test_general_contact_beam_shell import _coupled_q4_model
    q,qd,qs=_coupled_q4_model()
    assert not validate_general_contact(q,qd,replace(qs,beam_shell_geometry='jet2_deformable_s3_dev',beam_shell_maximum_swept_intervals=128)).valid
    with pytest.raises(ValueError):replace(s,beam_shell_geometry='unknown')


def independent_contact_gradient(x):
    """Two DOF contact oracle: own moving-plane formula and scalar integration."""
    tip,w=x;A=np.sqrt(9+w*w)
    def quantities(t):
        N=3*(.01005+t*tip)-w
        gap=N/A-.01
        return gap,np.array((3*t/A,-1/A-N*w/A**3))
    ends=[quantities(t)[0] for t in (0.,1.)]
    points=[]
    if ends[0]*ends[1]<0:points=[ends[0]/(ends[0]-ends[1])]
    return np.array([quad(lambda t:1.e5*.5*min(quantities(t)[0],0.)*quantities(t)[1][i],
                          0.,1.,points=points,epsabs=1.e-13,epsrel=1.e-12)[0] for i in range(2)])


def test_coupled_deforming_target_independent_contact_equilibrium_and_split_restart():
    m,d,s=fixture();constraints(m)
    load=LoadCase('press');load.add_nodal_load(2,forces=(0.,0.,-.001))
    kwargs=dict(contact=d,contact_settings=s,kinematics='von_karman',num_steps=4,
                max_iterations=20,tolerance=1.e-10)
    full=solve_static_nonlinear(m,load,**kwargs)
    assert full.status=='completed',full.failure_reason
    inds=np.array([m.mesh.dof_manager.get_node_dofs(n)[2] for n in (2,5)])
    x=full.displacements[inds]
    assert x[1]<-1.e-7 # Genuinely different target translations, not common motion.
    contact=full.info['general_contact'];assert contact['profile']==PROFILE and contact['gap_tolerance_met']
    manager=DeformableBeamS3Manager(m,d,s)
    r,_,_=manager.evaluate(full.displacements)
    oracle=independent_contact_gradient(x)
    np.testing.assert_allclose(r[inds],oracle,rtol=1.e-9,atol=1.e-12)
    # Keep the independent contact oracle and add the adopted S3 VK membrane
    # term to the existing linear structural comparison. All in-plane DOFs are
    # fixed: area=4.5, dw/dy=w5/3, eps_yy=w5**2/18, so differentiating membrane
    # energy gives area*E*t*w5**3/(162*(1-nu**2)). Constants are fixture inputs.
    # This remains a bounded oracle, not independent beam/shell qualification.
    K,_=assemble_stiffness_matrix(m);K2=K[inds][:,inds].toarray();f=np.array((-.001,0.))
    cubic_coefficient=4.5*1.e5*.02/(162*(1.-.3**2))
    reference=root(lambda y:K2@y+np.array((0.,cubic_coefficient*y[1]**3))
                   +independent_contact_gradient(y)-f,x,tol=1.e-11)
    assert reference.success
    assert np.linalg.norm(reference.x-x)/np.linalg.norm(reference.x)<1.e-3
    half=solve_static_nonlinear(m,load,max_load_factor=.5,emit_restart_checkpoint=True,**kwargs)
    assert half.status=='completed',half.failure_reason
    checkpoint=json.loads(json.dumps(half.to_restart_checkpoint()))
    resumed=solve_static_nonlinear(m,load,restart_checkpoint=checkpoint,**kwargs)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1.e-7,atol=1.e-10)
    with pytest.raises(ValueError,match='contact|restart|incompatible'):
        solve_static_nonlinear(m,load,restart_checkpoint=checkpoint,
              **dict(kwargs,contact_settings=replace(s,beam_shell_maximum_swept_intervals=127)))
    print('DEFORMABLE_BEAM_S3_RESPONSE '+json.dumps({'coordinates':x.tolist(),
          'von_karman_reference':reference.x.tolist(),'contact':contact},sort_keys=True))



def independent_contact_hessian(x):
    tip,w=x;A=np.sqrt(9+w*w)
    def quantities(t):
        N=3*(.01005+t*tip)-w;gap=N/A-.01
        gradient=np.array((3*t/A,-1/A-N*w/A**3))
        second=np.array(((0.,-3*t*w/A**3),(-3*t*w/A**3,2*w/A**3-N/A**3+3*N*w*w/A**5)))
        return gap,gradient,second
    ends=[quantities(t)[0] for t in (0.,1.)];points=[]
    if ends[0]*ends[1]<0:points=[ends[0]/(ends[0]-ends[1])]
    def entry(t,i,j):
        g,G,H=quantities(t)
        return 1.e5*.5*(G[i]*G[j]+g*H[i,j]) if g<0 else 0.
    return np.array([[quad(lambda t:entry(t,i,j),0.,1.,points=points,epsabs=1.e-12,epsrel=1.e-12)[0] for j in range(2)] for i in range(2)])


def test_independent_linearized_path_locates_swept_resource_limit():
    m,d,s=fixture();constraints(m)
    indices=np.array([m.mesh.dof_manager.get_node_dofs(n)[2] for n in (2,5)])
    K,_=assemble_stiffness_matrix(m);K2=K[indices][:,indices].toarray()
    reference=root(lambda y:K2@y+independent_contact_gradient(y)-np.array((-.001,0.)),(-.00102946,-.00290194),jac=lambda y:K2+independent_contact_hessian(y),tol=1.e-11)
    assert reference.success
    manager=DeformableBeamS3Manager(m,d,s);u=np.zeros(m.mesh.dof_manager.total_dofs);u[indices]=reference.x
    b0,s0=manager._positions(np.zeros_like(u));b1,s1=manager._positions(u)
    outcomes={str(cap):manager.kernel.swept_admissible(b0,s0,b1,s1,gap_tolerance=manager.gap_tolerance,maximum_intervals=cap) for cap in (128,512)}
    print('SWEPT_RESOURCE_DISCRIMINATOR '+json.dumps({'reference':reference.x.tolist(),'minimum_gap':manager.kernel.evaluate(b1,s1).minimum_gap,'budgets':outcomes},sort_keys=True))
    assert outcomes['512']


def test_line_units_and_public_source_example():
    import runpy
    m,d,s=fixture();manager=DeformableBeamS3Manager(m,d,s)
    assert not hasattr(manager,'stiffness') and manager.line_stiffness==1.e5
    result=runpy.run_path('examples/general_contact_deformable_round_beam_s3.py')['main']()
    contact=result.info['general_contact']
    assert 'normal_stiffness' not in contact
    assert contact['pair_identity']=='line'
    assert contact['physical_features']=={'beam':{'surface':'tube','element_id':1},'shell':{'surface':'skin','element_id':2}}
    assert contact['moment_origin_m']==[0.,0.,0.]
    assert contact['contact_integration_measure']=='reference_line_length'
    assert contact['line_stiffness_N_per_m2']==1.e5 and contact['gap_tolerance_met']
