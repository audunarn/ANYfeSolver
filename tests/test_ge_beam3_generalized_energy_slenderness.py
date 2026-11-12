"""Explicit energy-only supplied-block diagnostic, not force-operator repair."""
from decimal import Decimal,localcontext
from fractions import Fraction
import numpy as np
import pytest
from docs.reference_cases.ge_beam3_generalized_energy_chain import energy_chain
from docs.reference_cases.ge_beam3_generalized_virgin_chain import virgin_chain
from docs.reference_cases.ge_beam3_decimal_chain_audit import factor_chain_modes
from anyfesolver import _ge_beam3_native_generalized_modal as modal
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
from test_ge_beam3_schur_line_program import save
from test_ge_beam3_generalized_slenderness_diagnostic import make,kinetic

@pytest.mark.parametrize('kind',('symmetric','skew-energy','work-pair'))
def test_explicit_energy_boundary(kind,tmp_path):
    j=np.arange(18*24,dtype=float).reshape(18,24)/100
    s=4*np.eye(18)
    if kind=='skew-energy':s[0,1]=.5;s[1,0]=-.25
    h=np.block([[np.zeros((24,24)),j.T],[j,-s]])
    if kind=='work-pair':
        h[0,24]+=1.
        with pytest.raises(ValueError):energy_chain(h,np.zeros_like(h),np.zeros(42))
        save(tmp_path/'boundary.json',dict(kind=kind,rejected=True));return
    l,right,symmetric,report=energy_chain(h,np.zeros_like(h),np.zeros(42))
    if kind=='symmetric':
        old=virgin_chain(h,np.zeros_like(h),np.zeros(42))
        assert all(np.array_equal(a,b) for a,b in zip((l,right,symmetric),old))
    else:
        with pytest.raises(ValueError):virgin_chain(h,np.zeros_like(h),np.zeros(42))
        x=list(range(18))
        def quadratic(a):return sum(Fraction(float(a[i,k]))*x[i]*x[k] for i in range(18) for k in range(18))
        assert quadratic(s)==quadratic(symmetric)
        assert report['maximum_compliance_skew']==.75
    save(tmp_path/'boundary.json',dict(kind=kind,report=report))

@pytest.mark.parametrize('rho',(100.,10000.,1000000.))
def test_supplied_generalized_energy_diagnosis(rho,tmp_path):
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
            save(tmp_path/(label+'-raw.json'),dict(high=replay.hessian,low=replay.hessian_low,residual=replay.residual))
            left,right,compliance,energy_audit=energy_chain(replay.hessian,replay.hessian_low,replay.residual)
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
                native_status='NOT_EVALUATED',energy_audit=energy_audit,production_qualified=False,independent_review='PENDING')
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
