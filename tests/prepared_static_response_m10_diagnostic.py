"""Conditional output enclosures including observed canonical-solve uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np
from prepared_static_response_m10_bounds import *
from prepared_static_response_m9_diagnostic import (
    _shell_model,_shell_case,mixed_free_model,free_cases,s3_setup,
    PatternCase,RecoveryConfig,prepare_static_response_evaluator,
    PreparedStaticEvaluator,_recover_owned_local,_derived_measure)

def run(family):
    if family=='supported':
        model=_shell_model();patterns={'p':_shell_case(0,pressure=2.).load_case}
        coefficients=[{'p':.3},{'p':3.7}]
        eid=1
    else:
        model,groups,balancing=mixed_free_model()
        patterns={name:load for name,time,load in free_cases(model,groups,balancing,[0,48,576])}
        coefficients=[{'case-048':.3},{'case-048':1.,'case-576':-.7}]
        eid=5
    rows=[]
    with prepare_static_response_evaluator(model,patterns,recovery_config=RecoveryConfig(element_ids=[eid],include_stresses=True,include_reactions=False)) as owner:
        snap=owner._snapshot;n=snap.reduced_stiffness.shape[0]
        A=owner._operator_matrix if owner._operator_augmented else snap.reduced_stiffness
        bound=SignedActionBound(A.toarray())
        result=dict(size=A.shape[0],bound_method='signed_streamed_defect_action',comparison_eta_upper=upward(bound.comparison_eta),eta_upper=upward(bound.eta),factor_certificate=bound.accepted,cases=rows)
        scaling=np.ones(A.shape[0])
        if not bound.accepted and family=='free':
            stiffness_size=float(np.max(np.abs(snap.reduced_stiffness.data)))
            exponent=int(np.rint(-.5*np.log2(stiffness_size)))
            scaling[:n]=np.ldexp(1.,exponent);scaling[n:]=np.ldexp(1.,-exponent)
            original=A.toarray();scaled=scaling[:,None]*original*scaling[None,:]
            assert np.all(np.isfinite(scaled))
            assert all(rational(scaled[i,j])==rational(original[i,j])*rational(scaling[i])*rational(scaling[j]) for i in range(len(scaling)) for j in range(len(scaling)))
            bound=SignedActionBound(scaled)
            result['power_two_congruence']={'q_exponent':exponent,'multiplier_exponent':-exponent,'eta_upper':upward(bound.eta),'factor_certificate':bound.accepted,'exact_operator_products':True}
        if not bound.accepted:return result
        beta=max(bound.apply([ONE]*A.shape[0]))/(ONE-bound.eta)
        normA=max(sum((abs(a) for a in row),ZERO) for row in bound.A)
        optimistic_alpha=rational(1e-10)*normA
        result['optimistic_gate_ignores_checker_rounding']={'inverse_action_norm_upper':upward(beta),'beta_alpha':upward(beta*optimistic_alpha),'contraction':beta*optimistic_alpha<ONE}
        assert np.count_nonzero(snap.prescribed_offset)==0
        recipe=snap.recovery_compilation.elements[eid].local_recipe
        mapping=snap.recovery_compilation.elements[eid].dof_mapping
        intervals_for_envelope=[];all_intervals=[];vm_intervals=[]
        for index,coef in enumerate(coefficients):
            _,_,packet=owner._ingest_case(PatternCase(str(index),coef))
            F=owner._combine(owner._force_columns,packet)
            rhs=np.asarray(snap.transformation.T@(F-snap.stiffness@snap.prescribed_offset))
            if owner._operator_augmented:rhs=np.concatenate((rhs,np.zeros(snap.rigid_modes.shape[1])))
            direct=PreparedStaticEvaluator._solve_block(owner,np.ascontiguousarray(rhs[:,None]),())[:,0]
            reconstructed=owner._combine(owner._solution_columns,packet)
            scaled_rhs=scaling*rhs;zc=reconstructed/scaling;zd=direct/scaling
            assert np.all(np.isfinite(scaled_rhs)) and np.all(np.isfinite(zc)) and np.all(np.isfinite(zd))
            assert all(rational(b)==rational(s)*rational(a) for a,b,s in zip(rhs,scaled_rhs,scaling))
            assert all(rational(x)==rational(z)*rational(s) for values,zvalues in ((reconstructed,zc),(direct,zd)) for x,z,s in zip(values,zvalues,scaling))
            ec=[rational(s)*e for s,e in zip(scaling,bound.error(scaled_rhs,zc))]
            ed=[rational(s)*e for s,e in zip(scaling,bound.error(scaled_rhs,zd))]
            qi=[(rational(x)-a-b,rational(x)+a+b) for x,a,b in zip(reconstructed,ec,ed)]
            assert all(contains(i,x) for i,x in zip(qi,direct))
            ui=matvec(snap.transformation.toarray(),qi[:n])
            ud=snap.transformation@direct[:n]+snap.prescribed_offset
            assert all(contains(i,x) for i,x in zip(ui,ud))
            native=_recover_owned_local(recipe,ud[mapping])
            if family=='supported':
                native_module,element,arrays,membrane,shear=s3_setup(recipe)
                local=matvec(element._local_dof_transform(arrays['frame']),[ui[j] for j in mapping])
                external=[local[j] for j in native_module.PHYSICAL_EXTERNAL_INDICES]
                coordinates=external+matvec(arrays['bubble_map'],external)
                fields=[];expected=[]
                for k,(r,s,_) in enumerate(native_module.TRIANGLE_QUADRATURE):
                    operator=element._director_generalized_transform()@native_module._kinematic_matrix(arrays['local_nodes'],r,s,arrays['assumed_shear_samples'])
                    strain=matvec(operator,coordinates)
                    resultants=matvec(arrays['constitutive'],strain)
                    fields.extend(matvec(membrane,strain[:3]))
                    expected.extend(native['membrane_'+name][k] for name in ('xx','yy','xy'))
                    fields.extend(divide(scale(value,6.),element.thickness*element.thickness) for value in resultants[3:6])
                    expected.extend(native['bending_'+name][k] for name in ('xx','yy','xy'))
                    fields.extend(matvec((5./6.)*shear,strain[6:]))
                    expected.extend(native['shear_'+name][k] for name in ('xz','yz'))
            else:
                from anyfesolver.elements import _beam_material_properties
                element,mesh,material=recipe.execution_owner
                L,T=element._beam_frame_and_transform(element.get_node_coordinates(mesh))
                local=matvec(T,[ui[j] for j in mapping])
                gamma=add(divide(add(local[8],scale(local[2],-1.)),L),scale(add(local[4],local[10]),.5))
                G=_beam_material_properties(material,element.cross_section)[2]
                fields=[scale(gamma,G*element._kz)];expected=[native['shear_stress_z']]
            failures=sum(not contains(i,x) for i,x in zip(fields,expected))
            assert failures==0
            widths=[upward(b-a) for a,b in fields]
            ratios=[upward((b-a)/(rational(1e-12)+rational(1e-12)*abs(rational(x)))) for (a,b),x in zip(fields,expected)]
            rows.append(dict(case=index,direct_reference_used=True,direct_error_radius=upward(max(ed)),reconstructed_error_radius=upward(max(ec)),observed_q_difference=float(np.max(np.abs(direct-reconstructed))),output_count=len(fields),containment_failures=failures,max_width=max(widths),max_width_over_diagnostic_tolerance=max(ratios)))
            if family=='supported':
                vm=[shell_vm(fields[k:k+8]) for k in range(0,len(fields),8)]
                reference=_derived_measure('von_mises',native)
                assert all(contains(i,x) for i,x in zip(vm,reference))
                rows[-1]['conditional_vm']={'count':len(vm),'containment_failures':0,'max_width':max(upward(b-a) for a,b in vm),'max_width_over_vm_tolerance':max(upward((b-a)/(rational(1e-12)+rational(1e-12)*abs(rational(x)))) for (a,b),x in zip(vm,reference))}
                vm_intervals.append(vm)
            all_intervals.append(fields)
            intervals_for_envelope.append(fields[0])
        a,b=intervals_for_envelope
        result['strict_max_exclusions']=int(strict_max_excluded(a,b))+int(strict_max_excluded(b,a))
        result['strict_min_exclusions']=int(strict_min_excluded(a,b))+int(strict_min_excluded(b,a))
        result['tie_excluded']=strict_max_excluded(a,a) or strict_min_excluded(a,a)
        assert not result['tie_excluded']
        def decisions(collection):
            first,second=collection
            assert len(first)==len(second)
            return {'paired_components':len(first),'observed_values':2*len(first),'max_separated_components':sum(strict_max_excluded(a,b) or strict_max_excluded(b,a) for a,b in zip(first,second)),'min_separated_components':sum(strict_min_excluded(a,b) or strict_min_excluded(b,a) for a,b in zip(first,second)),'self_tie_exclusions':sum(strict_max_excluded(a,a) or strict_min_excluded(a,a) for a in first+second)}
        result['all_component_decisions']=decisions(all_intervals)
        if vm_intervals:result['conditional_vm_decisions']=decisions(vm_intervals)
        return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    ill=np.array([[1.,1.],[1.,np.nextafter(1.,np.inf)]])
    p,l,u=lu(ill);u[-1,-1]=1e-30
    refused=ActionBound(ill,(p,l,u))
    assert not refused.accepted
    low=(Q(0),Q(1));high=(Q(2),Q(3))
    assert strict_max_excluded(low,high) and strict_min_excluded(high,low)
    assert not strict_max_excluded(low,low) and not strict_max_excluded(low,(Q(1),Q(2)))
    result={'status':'DIAGNOSTIC_ONLY_CONDITIONAL_NO_GO','controls':{'illconditioned_adverse_factor_eta':upward(refused.eta),'eta_refused':not refused.accepted,'strict_separation_resolved':True,'equal_and_touching_unresolved':True},'assumptions':['represented A and rhs are authoritative; assembly uncertainty not covered','direct reference residual is measured: no avoided direct solve','BLAS dots obey at most 2*n rounded operations, finite intermediates and gradual underflow','no principal eigensolver or whole-output certificate','zero-offset fixtures only'],'families':{family:run(family) for family in ('supported','free')}}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
