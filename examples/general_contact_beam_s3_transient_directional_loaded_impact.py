"""Loaded V20 development fixture; independent physical energy references."""
import numpy as np
from examples.general_contact_beam_s3_transient_directional_tilted import prepare

KINETIC_SCALE=3.219950866699218e-8
FORCE_SCALE=.5
MOMENT_SCALE=.25
DT=1.e-5
END=.0008

def independent_shell_coefficients(model):
    """Direct MIN3 trace relaxation from shape gradients, not element operators."""
    xy=np.array([model.mesh.nodes[n].coords()[:2] for n in (3,4,5)])
    interpolation=np.c_[np.ones(3),xy];gradients=np.linalg.inv(interpolation)[1:,:].T
    area=abs(np.linalg.det(interpolation))/2;E=1.e5;nu=.3;t=1/64
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

def independent_membrane(model,u):
    """Isotropic plane-stress CST von Karman membrane with in-plane motion held.

    Skin ux,uy,rz are supported, so the strain is e=(wx^2/2, wy^2/2, wx*wy)
    (engineering shear), constant over the flat triangle. Returns the energy
    and its gradient with respect to the three skin w coordinates.
    """
    xy=np.array([model.mesh.nodes[n].coords()[:2] for n in (3,4,5)])
    interpolation=np.c_[np.ones(3),xy];gradients=np.linalg.inv(interpolation)[1:,:].T
    area=abs(np.linalg.det(interpolation))/2;E=1.e5;nu=.3;t=1/64
    C=E*t/(1-nu**2)*np.array(((1,nu,0),(nu,1,0),(0,0,(1-nu)/2)))
    w=u[[model.mesh.dof_manager.get_node_dofs(n)[2] for n in (3,4,5)]]
    gx,gy=gradients.T@w;strain=np.array((.5*gx*gx,.5*gy*gy,gx*gy));resultant=C@strain
    energy=.5*area*strain@resultant
    # d strain/d(gx,gy) = ((gx,0),(0,gy),(gy,gx)); chain through constant shape gradients.
    slope_gradient=area*np.array(((gx,0.),(0.,gy),(gy,gx))).T@resultant
    return float(energy),gradients@slope_gradient

def independent_reference(model,u,v):
    ids={n:model.mesh.dof_manager.get_node_dofs(n) for n in model.mesh.nodes}
    L=float(model.mesh.nodes[2].x-model.mesh.nodes[1].x)
    bd=np.array([ids[n] for n in (1,6,2)]);w=u[bd[:,2]];ry=u[bd[:,4]]
    points,weights=np.polynomial.legendre.leggauss(3)
    beam=kinetic=0.;force=np.zeros_like(u);mass=np.zeros((len(u),len(u)))
    GA=1000/2.6*.001*(5/6)
    for x,weight in zip(points,weights):
        N=np.array((x*(x-1)/2,1-x*x,x*(x+1)/2));dN=np.array((x-.5,-2*x,x+.5))*2/L;jac=weight*L/2
        slope=dN@w;curvature=dN@ry;shear=slope+N@ry
        beam+=.5*jac*((.5*slope*slope)**2+.1*curvature*curvature+GA*shear*shear)
        force[bd[:,2]]+=jac*(.5*slope**3*dN+GA*shear*dN)
        force[bd[:,4]]+=jac*(.1*curvature*dN+GA*shear*N)
        for column,density in ((2,.001),(4,.0001)):
            block=density*jac*np.outer(N,N);mass[np.ix_(bd[:,column],bd[:,column])]+=block
    gradients,c,details=independent_shell_coefficients(model)
    sd=np.array([ids[n] for n in (3,4,5)]);gamma=gradients.T@u[sd[:,2]]+np.array((u[ids[3][4]],-u[ids[3][3]]))
    shell=.5*c*(gamma@gamma);force[sd[:,2]]+=c*gradients@gamma
    # Only the tied rotational chart is compared: distribute the total tied derivative once.
    force[ids[3][4]]+=c*gamma[0];force[ids[3][3]]-=c*gamma[1]
    membrane,membrane_force=independent_membrane(model,u);force[sd[:,2]]+=membrane_force
    xy=np.array([model.mesh.nodes[n].coords()[:2] for n in (3,4,5)]);area=abs(np.linalg.det(np.c_[np.ones(3),xy]))/2
    mass[sd[:,2],sd[:,2]]=area*(1/64)/3
    kinetic=.5*v@(mass@v)
    beam_positions=np.array([model.mesh.nodes[n].coords()+u[ids[n][:3]] for n in (1,6,2)])
    skin=np.array([model.mesh.nodes[n].coords()+u[ids[n][:3]]+(1/128)*(np.array((0.,0.,1.))+np.cross(u[ids[n][3:]],(0.,0.,1.))) for n in (3,4,5)])
    normal=np.cross(skin[1]-skin[0],skin[2]-skin[0]);normal/=np.linalg.norm(normal)
    b0,b1,b2=beam_positions
    poly=np.array((normal@(b0-skin[0])-1/128,normal@(-3*b0+4*b1-b2),normal@(2*b0-4*b1+2*b2)))
    roots=np.polynomial.polynomial.polyroots(poly);cuts=sorted([0.,1.]+[float(x.real) for x in roots if abs(x.imag)<1.e-12 and 0<x.real<1])
    square=np.polynomial.polynomial.polymul(poly,poly);primitive=np.r_[0.,square/np.arange(1,len(square)+1)];integral=0.
    for a,b in zip(cuts,cuts[1:]):
        if np.polynomial.polynomial.polyval((a+b)/2,poly)<0:integral+=np.polynomial.polynomial.polyval(b,primitive)-np.polynomial.polynomial.polyval(a,primitive)
    contact=.5*1.e5*L*integral
    candidates=[0.,1.]
    if poly[2]!=0:
        stationary=-poly[1]/(2*poly[2])
        if 0<stationary<1:candidates.append(stationary)
    minimum=min(float(np.polynomial.polynomial.polyval(x,poly)) for x in candidates)
    return dict(independent_min_gap=minimum,energy=float(beam+shell+membrane+contact+kinetic),beam=float(beam),shell=float(shell),membrane=membrane,contact=float(contact),kinetic=float(kinetic),details=details),force,mass

def run(*,dt=DT,end=END,restart=None,acceleration_unknown=True):
    from anysolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anysolver.dynamics import TransientConfig
    m,d,s,u,v,load,initial=prepare()
    result=solve_transient_contact(m,d,TransientConfig(dt,end,initial_displacement=u,initial_velocity=v),base_load_case=load,contact_settings=s,
        solver_controls=ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=True,save_constraint_reactions_dev=True,
            equilibrated_newmark_acceleration_unknown_dev=acceleration_unknown),restart_checkpoint=restart,emit_restart_checkpoint=True)
    return m,result,initial

if __name__=='__main__':
    m,r,initial=run(dt=DT,end=END)
    print(dict(states=len(r.times),final_gap=r.contact[-1].minimum_gap,profile=r.profile))
