"""Actual B3 bending/rotary inertia and independent polynomial work references."""
import json,os
from pathlib import Path
import numpy as np
from examples.general_contact_beam_s3_transient import build,run
from test_general_contact_beam_s3_transient import body_balance
from s3_von_karman_oracle import model_triangle_von_karman,p1_von_karman_inplane_force


def independent_beam_storage(model,u):
    """Q2 Timoshenko/VK section potential, own shape functions/material values."""
    nodes=(1,6,2);L=.5;dofs=np.array([model.mesh.dof_manager.get_node_dofs(n) for n in nodes])
    # Reference x is axial; global z displacement and global ry rotation have
    # shear gamma=w'+ry. All other fixture components are explicitly zero.
    w=u[dofs[:,2]];theta=u[dofs[:,4]];EA=1.;GA=(1.e3/2.6)*1.e-3*(5/6);EI=.1
    points,weights=np.polynomial.legendre.leggauss(3)
    energy=bending=shear=0.
    for xi,weight in zip(points,weights):
        N=np.array((xi*(xi-1)/2,1-xi**2,xi*(xi+1)/2));derivative=np.array((xi-.5,-2*xi,xi+.5))*2/L
        slope=derivative@w;curvature=derivative@theta;gamma=slope+N@theta;jac=weight*L/2
        bending+=.5*EI*curvature**2*jac;shear+=.5*GA*gamma**2*jac
        energy+=.5*(EA*(.5*slope**2)**2+GA*gamma**2+EI*curvature**2)*jac
    return energy,bending,shear


def independent_contact_storage(model,u):
    """Own physical positions, stationary polynomial roots and exact integral."""
    p=[]
    for node in (1,6,2):p.append(model.mesh.nodes[node].coords()+u[model.mesh.dof_manager.get_node_dofs(node)[:3]])
    skin=[]
    for node in (3,4,5):skin.append(model.mesh.nodes[node].coords()+u[model.mesh.dof_manager.get_node_dofs(node)[:3]]+np.array((0.,0.,.01)))
    p=np.array(p);skin=np.array(skin);n=np.cross(skin[1]-skin[0],skin[2]-skin[0]);n/=np.linalg.norm(n)
    coefficients=np.array((n@(p[0]-skin[0])-.01,n@(-3*p[0]+4*p[1]-p[2]),n@(2*p[0]-4*p[1]+2*p[2])))
    roots=np.polynomial.polynomial.polyroots(coefficients)
    cuts=sorted([0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1.e-12 and 0<r.real<1])
    square=np.polynomial.polynomial.polymul(coefficients,coefficients);primitive=np.r_[0.,square/np.arange(1,len(square)+1)]
    integral=0.
    for a,b in zip(cuts,cuts[1:]):
        if np.polynomial.polynomial.polyval((a+b)/2,coefficients)<0:
            integral+=np.polynomial.polynomial.polyval(b,primitive)-np.polynomial.polynomial.polyval(a,primitive)
    candidates=[0.,1.]
    if coefficients[2]!=0:
        s=-coefficients[1]/(2*coefficients[2])
        if 0<s<1:candidates.append(s)
    endpoints=np.polynomial.polynomial.polyval(np.array((0.,1.)),coefficients)
    return .5*1.e5*.5*integral,min(np.polynomial.polynomial.polyval(s,coefficients) for s in candidates),endpoints


def independent_shell_coefficients(model):
    """Direct MIN3 trace relaxation from shape gradients, not element operators."""
    xy=np.array([model.mesh.nodes[n].coords()[:2] for n in (3,4,5)])
    interpolation=np.c_[np.ones(3),xy];gradients=np.linalg.inv(interpolation)[1:,:].T
    area=abs(np.linalg.det(interpolation))/2;E=1.e5;nu=.3;t=.02
    D=E*t**3/(12*(1-nu**2))*np.array(((1,nu,0),(nu,1,0),(0,0,(1-nu)/2)))
    H=(5/6)*E*t/(2*(1+nu));bend_trace=0.;shear_trace=0.
    # kappa=(ry,x,-rx,y,ry,y-rx,x); trace over each real rx/ry coordinate.
    for gx,gy in gradients:
        for kappa in (np.array((0.,-gy,-gx)),np.array((gx,0.,gy))):bend_trace+=area*kappa@D@kappa
    a=np.array((xy[2,0]-xy[1,0],xy[0,0]-xy[2,0],xy[1,0]-xy[0,0]))
    b=np.array((xy[1,1]-xy[2,1],xy[2,1]-xy[0,1],xy[0,1]-xy[1,1]))
    # L_i=.5 Ni(b_k Nj-b_j Nk), M_i=.5 Ni(a_j Nk-a_k Nj).
    # Their spatial derivatives give MIN3's true rotational shear columns.
    for N in ((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3)):
        for i,j,k in ((0,1,2),(1,2,0),(2,0,1)):
            gradL=.5*(gradients[i]*(b[k]*N[j]-b[j]*N[k])+N[i]*(b[k]*gradients[j]-b[j]*gradients[k]))
            gradM=.5*(gradients[i]*(a[j]*N[k]-a[k]*N[j])+N[i]*(a[j]*gradients[k]-a[k]*gradients[j]))
            rx=-gradL-np.array((0.,N[i]));ry=gradM+np.array((N[i],0.))
            shear_trace+=area/3*H*(rx@rx+ry@ry)
    psi=bend_trace/shear_trace;phi=2*psi/(1+2*psi)
    return gradients,area*phi*H,dict(phi_squared=phi,bending_trace=bend_trace,unrelaxed_shear_trace=shear_trace)


def independent_shell_storage(model,u):
    gradients,coefficient,_=independent_shell_coefficients(model)
    w=np.array([u[model.mesh.dof_manager.get_node_dofs(n)[2]] for n in (3,4,5)])
    slope=gradients.T@w
    return .5*coefficient*(slope@slope)


def independent_structural_force(model,u):
    """Analytic potential gradient in the actual free bending/normal chart."""
    force=np.zeros_like(u);nodes=(1,6,2);dofs=np.array([model.mesh.dof_manager.get_node_dofs(n) for n in nodes]);w=u[dofs[:,2]];theta=u[dofs[:,4]]
    points,weights=np.polynomial.legendre.leggauss(3)
    for xi,weight in zip(points,weights):
        N=np.array((xi*(xi-1)/2,1-xi**2,xi*(xi+1)/2));derivative=np.array((xi-.5,-2*xi,xi+.5))*4
        slope=derivative@w;curvature=derivative@theta;gamma=slope+N@theta;GA=(1.e3/2.6)*1.e-3*(5/6)
        force[dofs[:,2]]+=weight*.25*(.5*slope**3*derivative+GA*gamma*derivative)
        force[dofs[:,4]]+=weight*.25*(.1*curvature*derivative+GA*gamma*N)
    gradients,coefficient,_=independent_shell_coefficients(model)
    ids=[model.mesh.dof_manager.get_node_dofs(n)[2] for n in (3,4,5)]
    force[ids]=coefficient*gradients@(gradients.T@u[ids])+independent_shell_von_karman(model,u)[1]
    return force


def independent_shell_von_karman(model,u):
    """Adopted S3 VK membrane term (precedent 706bf011c): skin in-plane
    translations and directors are guided, so only the membrane VK energy of
    the uz field is added; fixture constants E=1e5, nu=.3, t=.02."""
    w=[u[model.mesh.dof_manager.get_node_dofs(n)[2]] for n in (3,4,5)]
    return model_triangle_von_karman(model,(3,4,5),(0.,0.,1.),w,1.e5,.3,.02)


def independent_mass_chart():
    """rho*A and rho*Iy Q2 integrals plus actual S3 lumped translation mass."""
    mass=np.zeros((9,9));points,weights=np.polynomial.legendre.leggauss(3)
    for xi,weight in zip(points,weights):
        N=np.array((xi*(xi-1)/2,1-xi**2,xi*(xi+1)/2))
        for i in range(3):
            for j in range(3):
                mass[2*i,2*j]+=weight*.25*1.e-3*N[i]*N[j]
                mass[2*i+1,2*j+1]+=weight*.25*1.e-4*N[i]*N[j]
    mass[6:,6:]=np.eye(3)*(.02*4.5/3)
    return mass


def test_actual_flexural_impact_and_independent_storage():
    from anyfesolver.assembly import assemble_mass_matrix
    from anyfesolver.general_contact.beam_s3_transient import BeamS3TransientManager
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.general_contact.runtime import suspend_contact
    m,r=run(quadratic=True,unequal=True,flexural=True)
    M,_=assemble_mass_matrix(m);dofs=lambda n:m.mesh.dof_manager.get_node_dofs(n)
    shell=m.mesh.elements[2];sdofs=np.concatenate([dofs(n) for n in (3,4,5)])
    Ks=shell.compute_stiffness_matrix(m.mesh,m.materials['skin'])
    stores=[];contact_errors=[];bending=[];shear=[];min_gaps=[];hidden=[];shell_storage_errors=[];shell_stores=[]
    for u,v,contact in zip(r.displacements,r.velocities,r.contact):
        beam,be,sh=independent_beam_storage(m,u);line,gap,ends=independent_contact_storage(m,u)
        direct_shell=independent_shell_storage(m,u)
        shell_stores.append(direct_shell)
        shell_storage_errors.append(abs(direct_shell-.5*u[sdofs]@(Ks@u[sdofs])))
        # Total storage includes the adopted S3 VK membrane energy.
        stores.append(.5*v@(M@v)+beam+direct_shell+independent_shell_von_karman(m,u)[0]+line)
        contact_errors.append(abs(line-contact.line_contact_energy));bending.append(be);shear.append(sh);min_gaps.append(gap)
        if gap<0 and np.min(ends)>0:hidden.append(dict(index=len(stores)-1,minimum_gap=gap,endpoint_gaps=ends.tolist()))
    energy_error=float(np.max(abs(np.array(stores)-stores[0]))/stores[0])
    i=int(np.argmin(min_gaps));u=r.displacements[i];mean_curvature=float(np.max(abs(r.displacements[:,dofs(1)[4]]-r.displacements[:,dofs(2)[4]]))/.5)
    curve=float(np.max(abs(r.displacements[:,dofs(6)[2]]-.5*(r.displacements[:,dofs(1)[2]]+r.displacements[:,dofs(2)[2]]))))
    with suspend_contact():internal,K,_=_assemble_nonlinear_system(m,u,{},3,tangent=True,kinematics='von_karman')
    # Retarget to the adopted S3 VK law: linear shell force plus the
    # independent VK membrane force on the three skin uz coordinates.
    shell_expected=np.array(Ks@u[sdofs],dtype=float);shell_expected[2::6]+=independent_shell_von_karman(m,u)[1]
    points=np.array([m.mesh.nodes[n].coords() for n in (3,4,5)]);w=u[sdofs[2::6]]
    inplane=p1_von_karman_inplane_force(points,(0.,0.,1.),w,1.e5,.3,.02)
    for i in range(3):shell_expected[6*i:6*i+3]+=inplane[i]  # membrane reactions on guided ux,uy
    shell_error=float(np.linalg.norm(internal[sdofs]-shell_expected))
    direction=np.zeros_like(u);direction[dofs(6)[2]]=.3;direction[dofs(1)[4]]=.2;direction[dofs(2)[4]]=-.1
    contact_direction=direction.copy();contact_direction[dofs(4)[2]]=-.2
    chart=[dofs(n)[c] for n in (1,6,2) for c in (2,4)]+[dofs(n)[2] for n in (3,4,5)]
    mass_error=float(np.linalg.norm(M[np.ix_(chart,chart)].toarray()-independent_mass_chart())/np.linalg.norm(independent_mass_chart()))
    manager=BeamS3TransientManager(*build(quadratic=True,unequal=True,flexural=True)[:3]);cr,CK,summary=manager.evaluate(u)
    errors=[]
    for h in (1.e-7,3.e-8,1.e-8):
        ep=independent_beam_storage(m,u+h*direction)[0];em=independent_beam_storage(m,u-h*direction)[0]
        beam_work=float(abs((ep-em)/(2*h)-internal@direction)/max(abs(internal@direction),1.e-20))
        rp,_,_ =manager.evaluate(u+h*contact_direction);rm,_,_=manager.evaluate(u-h*contact_direction)
        cp=independent_contact_storage(m,u+h*contact_direction)[0];cm=independent_contact_storage(m,u-h*contact_direction)[0]
        contact_work=float(abs((cp-cm)/(2*h)-cr@contact_direction)/max(abs(cr@contact_direction),1.e-20))
        contact_H=float(np.linalg.norm((rp-rm)/(2*h)-CK@contact_direction)/max(np.linalg.norm(CK@contact_direction),1.e-20))
        gp=independent_structural_force(m,u+h*contact_direction);gm=independent_structural_force(m,u-h*contact_direction)
        structural_H=float(np.linalg.norm(((gp-gm)/(2*h)-K@contact_direction)[chart])/np.linalg.norm((K@contact_direction)[chart]))
        errors.append(dict(h=h,beam_work=beam_work,contact_work=contact_work,contact_H=contact_H,structural_H=structural_H))
    _,prefix=run(quadratic=True,unequal=True,flexural=True,end=float(r.times[110]))
    _,resume=run(quadratic=True,unequal=True,flexural=True,restart=prefix.restart_checkpoint)
    restart_equal=np.array_equal(resume.displacements,r.displacements[len(prefix.times)-1:]) and resume.restart_checkpoint==r.restart_checkpoint
    balance=body_balance(m,r)
    data=dict(states=len(r.times),active=sum(x.active_line_length>0 for x in r.contact),final_active=r.contact[-1].active_line_length,minimum_gap=min(min_gaps),energy_error=energy_error,
        maximum_bending_storage=max(bending),maximum_shear_storage=max(shear),maximum_mean_curvature=mean_curvature,maximum_midpoint_chord_difference=curve,shell_linear_force_error=shell_error,
        contact_potential_error=max(contact_errors),derivative_errors=errors,restart_equal=restart_equal,body_balance_errors=balance,
        interior_active_open_endpoints=hidden,maximum_beam_ry=float(np.max(abs(r.displacements[:,[dofs(n)[4] for n in (1,6,2)]]))),
        independent_shell_coefficients=independent_shell_coefficients(m)[2],shell_storage_error=max(shell_storage_errors),
        mass_chart_error=mass_error,maximum_shell_storage=max(shell_stores),
        min_rotation_mass_eigenvalue=float(np.min(np.linalg.eigvalsh(M[np.ix_([dofs(n)[4] for n in (1,6,2)],[dofs(n)[4] for n in (1,6,2)])].toarray()))))
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):Path(os.environ['CONTACT_APP_RESULT_RECORD']).write_text(json.dumps(data,indent=2))
    assert data['active']>0 and data['final_active']==0
    assert hidden
    assert max(bending)>1.e-12 and mean_curvature>1.e-6 and curve>1.e-7
    assert data['min_rotation_mass_eigenvalue']>0 and shell_error<1.e-10
    assert max(contact_errors)<1.e-12 and min(min_gaps)>=-1.e-5
    assert max(max(x['beam_work'],x['contact_work'],x['contact_H'],x['structural_H']) for x in errors)<=1.e-5
    assert max(shell_storage_errors)<1.e-15
    assert mass_error<=1.e-13 and max(shell_stores)>0
    assert np.max(balance)<=1.e-6 and energy_error<=.01 and restart_equal


def test_flexural_temporal_energy_refinement():
    """Same input mechanics, fine timestep vs preserved actual coarse metric."""
    # Actual 10us result above: unchanged fixture, independently checked mass
    # (relative matrix error3.47e-16), energies and all accepted poses. This
    # constant preserves the coarse evidence without repeating its three solves.
    coarse_error=.0002207940005306129
    m,r=run(quadratic=True,unequal=True,flexural=True,dt=5.e-6)
    dofs=lambda n:m.mesh.dof_manager.get_node_dofs(n)
    chart=[dofs(n)[c] for n in (1,6,2) for c in (2,4)]+[dofs(n)[2] for n in (3,4,5)]
    independent_mass=independent_mass_chart();energies=[];gaps=[]
    for u,v in zip(r.displacements,r.velocities):
        line,gap,_=independent_contact_storage(m,u);gaps.append(gap)
        energies.append(.5*v[chart]@(independent_mass@v[chart])+independent_beam_storage(m,u)[0]+independent_shell_storage(m,u)+line)
    error=float(np.max(abs(np.array(energies)-energies[0]))/energies[0])
    data=dict(dt=5.e-6,coarse_dt=1.e-5,coarse_energy_error=coarse_error,fine_energy_error=error,energy_error_ratio=error/coarse_error,
        states=len(r.times),active=sum(x.active_line_length>0 for x in r.contact),minimum_gap=min(gaps),final_active=r.contact[-1].active_line_length,
        initial_energy_j=energies[0],independent_mass=True)
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):Path(os.environ['CONTACT_APP_RESULT_RECORD']).write_text(json.dumps(data,indent=2))
    assert error<=.01 and error<coarse_error
    assert data['active']>0 and data['final_active']==0 and min(gaps)>=-1.e-5
