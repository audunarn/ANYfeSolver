"""Conditional unseen resident SuperLU solve and output enclosures."""
import argparse,json
from pathlib import Path
import numpy as np
from prepared_static_response_m10_bounds import *
from prepared_static_response_m12_resident import ResidentSolveBound
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
        A=owner._operator_matrix
        bound=ResidentSolveBound(owner._operator_handle._solver,A,owner._operator_scaling,backend_name=owner._operator_handle.backend_name)
        result=dict(bound.metadata,cases=rows,bound_uses_observed_direct=False)
        assert np.count_nonzero(snap.prescribed_offset)==0
        recipe=snap.recovery_compilation.elements[eid].local_recipe
        mapping=snap.recovery_compilation.elements[eid].dof_mapping
        intervals_for_envelope=[];all_intervals=[];vm_intervals=[]
        for index,coef in enumerate(coefficients):
            _,_,packet=owner._ingest_case(PatternCase(str(index),coef))
            F=owner._combine(owner._force_columns,packet)
            rhs=np.asarray(snap.transformation.T@(F-snap.stiffness@snap.prescribed_offset))
            if owner._operator_augmented:rhs=np.concatenate((rhs,np.zeros(snap.rigid_modes.shape[1])))
            qi=bound.enclose(rhs)
            ui=matvec(snap.transformation.toarray(),qi[:n])
            if family=='supported':
                native_module,element,arrays,membrane,shear=s3_setup(recipe)
                local=matvec(element._local_dof_transform(arrays['frame']),[ui[j] for j in mapping])
                external=[local[j] for j in native_module.PHYSICAL_EXTERNAL_INDICES]
                coordinates=external+matvec(arrays['bubble_map'],external)
                fields=[];expected_keys=[]
                for k,(r,s,_) in enumerate(native_module.TRIANGLE_QUADRATURE):
                    operator=element._director_generalized_transform()@native_module._kinematic_matrix(arrays['local_nodes'],r,s,arrays['assumed_shear_samples'])
                    strain=matvec(operator,coordinates)
                    resultants=matvec(arrays['constitutive'],strain)
                    fields.extend(matvec(membrane,strain[:3]))
                    expected_keys.extend(('membrane_'+name,k) for name in ('xx','yy','xy'))
                    fields.extend(divide(scale(value,6.),element.thickness*element.thickness) for value in resultants[3:6])
                    expected_keys.extend(('bending_'+name,k) for name in ('xx','yy','xy'))
                    fields.extend(matvec((5./6.)*shear,strain[6:]))
                    expected_keys.extend(('shear_'+name,k) for name in ('xz','yz'))
            else:
                from anyfesolver.elements import _beam_material_properties
                element,mesh,material=recipe.execution_owner
                L,T=element._beam_frame_and_transform(element.get_node_coordinates(mesh))
                local=matvec(T,[ui[j] for j in mapping])
                gamma=add(divide(add(local[8],scale(local[2],-1.)),L),scale(add(local[4],local[10]),.5))
                G=_beam_material_properties(material,element.cross_section)[2]
                fields=[scale(gamma,G*element._kz)];expected_keys=[('shear_stress_z',None)]
            vm=[shell_vm(fields[k:k+8]) for k in range(0,len(fields),8)] if family=='supported' else []
            # Both q and physical-output enclosures are complete before oracle.
            direct=PreparedStaticEvaluator._solve_block(owner,np.ascontiguousarray(rhs[:,None]),())[:,0]
            assert all(contains(interval,x) for interval,x in zip(qi,direct))
            ud=snap.transformation@direct[:n]+snap.prescribed_offset
            assert all(contains(interval,x) for interval,x in zip(ui,ud))
            native=_recover_owned_local(recipe,ud[mapping])
            expected=[native[name] if k is None else native[name][k] for name,k in expected_keys]
            failures=sum(not contains(i,x) for i,x in zip(fields,expected))
            assert failures==0
            widths=[upward(b-a) for a,b in fields]
            ratios=[upward((b-a)/(rational(1e-12)+rational(1e-12)*abs(rational(x)))) for (a,b),x in zip(fields,expected)]
            rows.append(dict(case=index,direct_reference_used_by_bound=False,post_bound_oracle=True,coordinate_max_width=max(upward(b-a) for a,b in qi),output_count=len(fields),containment_failures=failures,max_width=max(widths),max_width_over_diagnostic_tolerance=max(ratios)))
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
    result={'status':'CONDITIONAL_RESIDENT_SOLVE_NOT_QUALIFIED','assumptions':['actual resident SuperLU L/U exports and permutations equal native supernode solve data','gstrs/vendor triangular solves obey stated componentwise residual relation with diagonal-scaled underflow allowance','finite binary64 round-to-nearest arithmetic and gradual underflow','canonical represented RHS and stable factor/source/binary authority','frozen M10 recovery dot/square/sqrt assumptions remain conditional','zero-offset fixtures; no principal/reaction certification'],'families':{family:run(family) for family in ('supported','free')}}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
