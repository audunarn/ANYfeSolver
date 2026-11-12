"""Bounded supplied-operator conditioning diagnosis; no qualification claim."""
from decimal import Decimal,localcontext
import numpy as np
import pytest
from docs.reference_cases.ge_beam3_generalized_virgin_chain import virgin_chain
from docs.reference_cases.ge_beam3_decimal_chain_audit import factor_chain_modes
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver import _ge_beam3_native_generalized_modal as modal
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from test_ge_beam3_schur_line_program import save

@pytest.mark.parametrize('kind',('valid','stress','geometric','asymmetric','nonfinite','shape','indefinite'))
def test_virgin_chain_boundary(kind,tmp_path):
    j=np.arange(18*24,dtype=float).reshape(18,24)/100
    s=np.diag(np.arange(1.,19.));h=np.block([[np.zeros((24,24)),j.T],[j,-s]])
    r=np.zeros(42);low=np.zeros_like(h)
    if kind=='stress':r[0]=1.
    if kind=='geometric':h[0,0]=1.
    if kind=='asymmetric':h[0,24]+=1.
    if kind=='nonfinite':h[24,24]=np.inf
    if kind=='shape':h=h[:-1]
    if kind=='indefinite':h[24,24]=1.
    if kind=='valid':
        l,right,compliance=virgin_chain(h,low,r)
        assert np.array_equal(right,j) and np.array_equal(compliance,s)
        assert np.linalg.norm((l@right).T@(l@right)-j.T@np.linalg.solve(s,j))<1e-11
        assert not l.flags.writeable and not right.flags.writeable
    else:
        with pytest.raises((ValueError,np.linalg.LinAlgError)):virgin_chain(h,low,r)
    save(tmp_path/'boundary.json',dict(kind=kind,checked=True))

def make(rho,curved,transform):
    x=np.array([-1.,0.,1.]);points=[];frames=[]
    for t in x:
        tangent=np.array([1.,-.4*t if curved else 0.,0.]);tangent/=np.linalg.norm(tangent)
        second=np.array([0.,0.,1.])
        points.append(transform@np.array([t,.2*(1-t*t) if curved else 0.,0.]))
        frames.append(transform@np.column_stack((tangent,second,np.cross(tangent,second))))
    ea=3*rho*rho;d=np.diag(np.sqrt([ea,ea/3,ea/3.5,2.,1.,1.5]))
    b=np.eye(6);b[0,3]=.02;b[1,5]=.03
    f=b@d;c=f.T@f
    height=2/rho
    inertia=np.diag([2.,2.,2.,5*height*height/12,3*height*height/12,2*height*height/12])
    model=FEModel('generalized-slenderness-diagnostic')
    for i,p in enumerate(points,1):model.add_node(i,*p)
    reference=CenteredCurvedBeam3ReferenceGeometry(np.array(points),np.array(frames))
    section=EllipsoidalGeneralizedSection(c,np.eye(6),1e6,1.)
    element=NativeGeneralizedStaticElement(1,(1,2,3),reference,section,order=4)
    model.add_element(1,element);model.materials[element.material_name]=section
    model.add_boundary_condition(BoundaryCondition('clamped',[1],{k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    state=element.init_model_bound_nonlinear_state(model.mesh,section,1)
    return model,{1:state},{1:inertia}

def kinetic(reference,inertia):
    rows=[];g,w=np.polynomial.legendre.leggauss(24)
    chol=np.linalg.cholesky(inertia).T
    for cell in (0,1):
        for a,b in zip(g,w):
            tau=float((a+1)/2);xi=cell-1+tau
            frame=reference.frame(xi);lift=reference.half_cell_lift(cell,tau)
            x,y,z=lift;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            v=np.zeros((6,24));v[:3,6*cell:6*cell+3]=(1-tau)*frame.T
            v[:3,6*(cell+1):6*(cell+1)+3]=tau*frame.T
            v[:3,18+3*cell:21+3*cell]=-frame.T@skew
            v[3:,18+3*cell:21+3*cell]=frame.T
            rows.append(np.sqrt(float(b)*reference.jacobian(xi)/2)*chol@v)
    return np.vstack(rows)

@pytest.mark.parametrize('rho',(100.,10000.,1000000.))
def test_supplied_generalized_slenderness_diagnosis(rho,tmp_path):
    observations=[]
    for curved in (False,True):
        for name,q in (('E',np.eye(3)),('GENERAL',rotation([.4,-.3,.2]))):
            label=('curved' if curved else 'straight')+'-'+name
            print(dict(stage='initialization',rho=rho,case=label),flush=True)
            model,states,inertias=make(rho,curved,q);before=canonical(states)
            e=model.mesh.elements[1];state=states[1];response=state['response'];core=e.operator
            replay=core.evaluate(state['positions'],state['position_low'],
                state['committed_nodal_rotation_matrices']@core.reference.nodal_triads,
                response.rotations,response.resultants,origin=response.history)
            left,right,compliance=virgin_chain(replay.hessian,replay.hessian_low,replay.residual)
            mass_factor=kinetic(core.reference,inertias[1])
            free=list(range(6,24));algebraic=[9,10,11,15,16,17]
            save(tmp_path/(label+'-inputs.json'),dict(left=left,right=right,compliance=compliance,
                kinetic=mass_factor,hessian=replay.hessian,hessian_low=replay.hessian_low,
                free=free,algebraic=algebraic,section=e.section.elastic,inertia=inertias[1],rotation=q))
            audits=[]
            for digits in (80,100):
                print(dict(stage='decimal',rho=rho,case=label,digits=digits),flush=True)
                value=factor_chain_modes(left.tolist(),right.tolist(),mass_factor.tolist(),free,algebraic,digits=digits)
                save(tmp_path/(label+'-decimal-'+str(digits)+'.json'),value);audits.append(value)
            with localcontext() as ctx:
                ctx.prec=100
                old=[Decimal(x) for x in audits[0]['eigenvalues']]
                exact=[Decimal(x) for x in audits[1]['eigenvalues']]
                delta=max(abs(a-b)/max(Decimal(1),abs(b)) for a,b in zip(old,exact))
                assert delta<Decimal('1e-40') and min(exact)>0
            target=np.array(audits[-1]['eigenvalues'],dtype=float)[:6]
            record=dict(case=label,rho=rho,decimal_relative_difference=str(delta),reference_roots=target,
                native_status='NOT_EVALUATED',production_qualified=False,independent_review='PENDING')
            try:
                packet,check=modal.prepare(model,states,np.zeros(18),inertias,np.zeros(18))
                record['native_status']='PREPARED'
                record['matrix_factor_relative_difference']=float(np.linalg.norm(packet.stiffness-(left@right).T@(left@right))/max(1.,np.linalg.norm(packet.stiffness)))
                record['mass_factor_relative_difference']=float(np.linalg.norm(packet.mass-mass_factor.T@mass_factor)/max(1.,np.linalg.norm(packet.mass)))
                save(tmp_path/(label+'-packet.json'),packet)
                modes=solve_stationary_spectrum(packet.stiffness,packet.mass,packet.free_dofs,packet.algebraic_dofs,num_modes=6)
                check();save(tmp_path/(label+'-dense.json'),modes)
                errors=abs(modes.eigenvalues-target)/np.maximum(1.,abs(target))
                record.update(native_status='RETURNED',native_roots=modes.eigenvalues,
                    normalized_root_error=errors,meets_1e11_root_comparison=bool(np.max(errors)<=1e-11))
            except (ValueError,np.linalg.LinAlgError) as exc:
                record.update(native_status='REJECTED',exception_type=type(exc).__name__,exception=str(exc))
            assert canonical(states)==before
            save(tmp_path/(label+'-observation.json'),record);observations.append(record)
            print(dict(stage='observation',case=label,rho=rho,status=record['native_status']),flush=True)
    covariance=[]
    for base,moved in zip(observations[::2],observations[1::2]):
        error=float(np.max(abs(base['reference_roots']-moved['reference_roots'])/np.maximum(1.,abs(base['reference_roots']))))
        covariance.append(dict(base=base['case'],moved=moved['case'],reference_root_error=error,
            meets_1e11_comparison=bool(error<=1e-11)))
    save(tmp_path/'diagnosis.json',dict(rho=rho,observations=observations,covariance=covariance,
        status='DIAGNOSTIC_OBSERVATIONS_COMPLETE',engineering_qualification=False,
        production_qualified=False,independent_review='PENDING'))
