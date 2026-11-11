"""M9 bounded stage diagnostic; no product route or scientific criteria changes."""
import argparse, json, sys
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'scripts')]
from anyfesolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator
from anyfesolver.prepared_static import PreparedStaticEvaluator
from anyfesolver.prepared_static_recovery import _recover_owned_local,_derived_measure
from anyfesolver.prepared_static_measures import SHELL_LINEAR_COMPONENTS,shell_surface_tensor
from anyfesolver.prepared_static_q4 import recover_q4_stages
from test_prepared_static_evaluator import _shell_model,_shell_case
from benchmark_static_batches import mixed_free_model,free_cases


def difference(a,b,rtol=1e-12,atol=1e-12):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==np.dtype('float64') and b.dtype==np.dtype('float64')
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    error=np.abs(a-b)
    payload_difference=a.view(np.uint64)!=b.view(np.uint64)
    signed_zero_difference=(a==0.) & (b==0.) & payload_difference
    return dict(max_abs=float(error.max(initial=0.)),failures=int(np.count_nonzero(error>atol+rtol*np.abs(b))),count=int(a.size),
        numerical_differences=int(np.count_nonzero(a!=b)),bit_differences=int(np.count_nonzero(payload_difference)),
        signed_zero_differences=int(np.count_nonzero(signed_zero_difference)))


def shell_metrics(a,b):
    result={}
    for surface in ('top','bottom'):
        result['principal_'+surface]=difference(np.linalg.eigvalsh(shell_surface_tensor(a,surface))[...,::-1],np.linalg.eigvalsh(shell_surface_tensor(b,surface))[...,::-1])
    result['von_mises']=difference(_derived_measure('von_mises',a),_derived_measure('von_mises',b))
    result['linear_components']=difference(np.concatenate([np.asarray(a[n]).ravel() for n in SHELL_LINEAR_COMPONENTS]),np.concatenate([np.asarray(b[n]).ravel() for n in SHELL_LINEAR_COMPONENTS]))
    return result


def combine(columns,c,extended=False):
    if extended:
        return np.asarray(np.sum(columns.astype(np.longdouble)*np.asarray(c,dtype=np.longdouble)[None,:],axis=1),dtype=float)
    active=np.flatnonzero(np.asarray(c)[1:])+1
    return columns[:,0]+(columns[:,active]@np.asarray(c)[active] if len(active) else 0.)


def s3_setup(recipe):
    from anyfesolver import e4_pl_s3_element as native
    element,mesh,material=recipe.execution_owner
    element._compute_stiffness_components(mesh,material,enforce_positive_winding=True)
    arrays=element._snapshot_qualified_component_arrays((("frame",(3,3)),("bubble_map",(2,15)),("local_nodes",(3,2)),("constitutive",(8,8)),("assumed_shear_samples",tuple((name,(2,17)) for name in ('A','B','C','D','E','F')))))
    membrane,shear,_,_=native._shell_material_matrices(material,element._material_angle(arrays['frame']),None)
    return native,element,arrays,membrane,shear


def s3_stages(setup,u=None,local=None,coordinates=None,strains=None):
    native,element,arrays,membrane,shear=setup
    if local is None:
        local=element._local_dof_transform(arrays['frame'])@u
    if coordinates is None:
        external=local[native.PHYSICAL_EXTERNAL_INDICES]
        coordinates=np.concatenate((external,arrays['bubble_map']@external))
    if strains is None:
        strains=np.zeros((len(native.TRIANGLE_QUADRATURE),8))
        transform=element._director_generalized_transform()
        for i,(r,s,_) in enumerate(native.TRIANGLE_QUADRATURE):
            strains[i]=transform@native._kinematic_matrix(arrays['local_nodes'],r,s,arrays['assumed_shear_samples'])@coordinates
    resultants=np.zeros_like(strains)
    for i in range(len(strains)):
        resultants[i]=arrays['constitutive']@strains[i]
    m=strains[:,:3]@membrane.T
    b=6.*resultants[:,3:6]/(element.thickness*element.thickness)
    t=strains[:,6:]@((5./6.)*shear).T
    fields={}
    for prefix,values in [('membrane_',m),('bending_',b)]:
        for j,name in enumerate(('xx','yy','xy')):
            fields[prefix+name]=values[:,j]
    fields.update(shear_xz=t[:,0],shear_yz=t[:,1])
    return fields,local,coordinates,strains


def run_family(family):
    if family.startswith('s3'):
        model=_shell_model(); patterns={'p':_shell_case(0,pressure=2.).load_case}
        cases=[{'p':c} for c in (1.,.1,.3,3.7,-1.25)]
        config=RecoveryConfig(include_stresses=True,include_reactions=False)
        if family=='s3_rotated':
            angle=np.pi/7
            rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
            for node in model.mesh.nodes.values():
                node.x,node.y,node.z=rotation@np.array([node.x,node.y,node.z])
            cases=[{'p':.3},{'p':3.7}]

    else:
        model,groups,balancing=mixed_free_model()
        patterns={name:load for name,time,load in free_cases(model,groups,balancing,[0,48,576])}
        cases=[{'case-048':.3},{'case-000':.25,'case-048':.75},{'case-048':1.,'case-576':-.7}]
        config=RecoveryConfig(element_ids=[1,5],include_stresses=True,include_reactions=False)
    result=[]
    with prepare_static_response_evaluator(model,patterns,recovery_config=config) as owner:
        snap=owner._snapshot; n=snap.reduced_stiffness.shape[0]
        assert np.count_nonzero(snap.prescribed_offset)==0, 'these diagnostic fixtures have zero prescribed offset'
        preparation_columns=[]
        for j in range(len(patterns)+1):
            source_rhs=np.asarray(snap.transformation.T@owner._force_columns[:,j])
            if j==0:source_rhs-=snap.transformation.T@(snap.stiffness@snap.prescribed_offset)
            if owner._operator_augmented:source_rhs=np.concatenate((source_rhs,np.zeros(snap.rigid_modes.shape[1])))
            canonical=PreparedStaticEvaluator._solve_block(owner,np.ascontiguousarray(source_rhs[:,None]),())[:,0]
            preparation_columns.append(difference(owner._solution_columns[:,j],canonical,0.,0.))
        full_columns=np.asarray(snap.transformation@owner._solution_columns[:n]);full_columns[:,0]+=snap.prescribed_offset
        for ci,coefficients in enumerate(cases):
            _,_,packet=owner._ingest_case(PatternCase(str(ci),coefficients))
            c=np.zeros(len(patterns)+1);c[0]=1.;c[packet.indices]=packet.coefficients
            F=owner._combine(owner._force_columns,packet)
            assert np.array_equal(F,combine(owner._force_columns,c))
            rhs=np.asarray(snap.transformation.T@(F-snap.stiffness@snap.prescribed_offset))
            if owner._operator_augmented: rhs=np.concatenate((rhs,np.zeros(snap.rigid_modes.shape[1])))
            direct=PreparedStaticEvaluator._solve_block(owner,np.ascontiguousarray(rhs[:,None]),())[:,0]
            qr=owner._combine(owner._solution_columns,packet)
            qe=combine(owner._solution_columns,c,True)
            ud=snap.transformation@direct[:n]+snap.prescribed_offset
            ur=snap.transformation@qr[:n]+snap.prescribed_offset
            ue=snap.transformation@qe[:n]+snap.prescribed_offset
            rtol,atol=(1e-11,1e-13) if family=='mixed' else (1e-12,1e-14)
            row=dict(case=ci,coefficients=coefficients,preparation_multi_vs_single_rhs=preparation_columns,
                reduced_coordinates_diagnostic=difference(qr[:n],direct[:n],rtol,atol),
                multiplier_roundoff=difference(qr[n:],direct[n:],0.,0.),
                physical_displacement_original_tolerance=difference(ur,ud,rtol,atol),
                backward_error=owner._backward_error(owner._operator_matrix if owner._operator_augmented else snap.reduced_stiffness,qr,rhs),elements={})
            for eid,compiled in snap.recovery_compilation.elements.items():
                recipe=compiled.local_recipe;mapping=compiled.dof_mapping
                canonical=_recover_owned_local(recipe,ud[mapping]);same_u=_recover_owned_local(recipe,ur[mapping]);extended=_recover_owned_local(recipe,ue[mapping])
                source=[_recover_owned_local(recipe,full_columns[mapping,j]) for j in range(len(c))]
                if family.startswith('s3') or recipe.staged_kernel is not None:
                    y={name:combine(np.column_stack([np.asarray(record[name]).ravel() for record in source]),c).reshape(np.asarray(source[0][name]).shape) for name in SHELL_LINEAR_COMPONENTS}
                    records={'q_native_vs_direct':shell_metrics(same_u,canonical),'q_alternate_accumulation_native_vs_direct':shell_metrics(extended,canonical),'field_sum_vs_same_u':shell_metrics(y,same_u)}
                    if family.startswith('s3'):
                        setup=s3_setup(recipe)
                        stage_source=[s3_stages(setup,u=full_columns[mapping,j]) for j in range(len(c))]
                        control,_,_,_=s3_stages(setup,u=ur[mapping])
                        records['staged_control_vs_same_u']=shell_metrics(control,same_u)
                        for stage,index in [('local',1),('coordinates',2),('strains',3)]:
                            values=combine(np.column_stack([values[index].ravel() for values in stage_source]),c).reshape(stage_source[0][index].shape)
                            kwargs={stage:values}
                            if stage!='local':kwargs['local']=stage_source[0][1]
                            proposed=s3_stages(setup,**kwargs)[0]
                            records[stage+'_sum_vs_same_u']=shell_metrics(proposed,same_u)
                    else:
                        stages=recipe.staged_kernel
                        local_cols=np.column_stack([stages.transform.T@full_columns[mapping,j] for j in range(len(c))])
                        local=combine(local_cols,c)
                        identity_stages=replace(stages,transform=np.eye(len(mapping)))
                        proposed=recover_q4_stages(identity_stages,local)
                        control=recover_q4_stages(identity_stages,stages.transform.T@ur[mapping])
                        records['staged_control_vs_same_u']=shell_metrics(control,same_u)
                        records['local_sum_vs_same_u']=shell_metrics(proposed,same_u)
                    row['elements'][eid]=records
                else:
                    from anyfesolver.elements import _beam_material_properties
                    element,mesh,material=recipe.execution_owner
                    L,Q=element._beam_frame_and_transform(element.get_node_coordinates(mesh))
                    G13=_beam_material_properties(material,element.cross_section)[2]
                    local_cols=np.column_stack([Q@full_columns[mapping,j] for j in range(len(c))])
                    local=combine(local_cols,c)
                    staged_tau=G13*element._kz*((local[8]-local[2])/L+.5*(local[4]+local[10]))
                    gamma_cols=(local_cols[8]-local_cols[2])/L+.5*(local_cols[4]+local_cols[10])
                    gamma_tau=G13*element._kz*float(combine(gamma_cols[None,:],c)[0])
                    row['elements'][eid]=dict(q_native_vs_direct=difference(same_u['shear_stress_z'],canonical['shear_stress_z']),local_sum_vs_same_u=difference(staged_tau,same_u['shear_stress_z']),gamma_sum_vs_same_u=difference(gamma_tau,same_u['shear_stress_z']),same_u_shear=float(same_u['shear_stress_z']),direct_shear=float(canonical['shear_stress_z']))
            result.append(row)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    results={'precision':{'float64_nmant':np.finfo(float).nmant,'longdouble_nmant':np.finfo(np.longdouble).nmant,
        'alternate_accumulation_is_wider':bool(np.finfo(np.longdouble).nmant>np.finfo(float).nmant)},
        'limitations':['zero prescribed offset fixtures','direct principal comparison applies corresponding field tolerance diagnostically; original principal qualification is same-u','reduced-coordinate tolerance is diagnostic; physical displacement checked separately','multiplier comparison records exact roundoff only'],
        'criteria':{'principal_same_u':[1e-12,1e-12],'von_mises':[1e-12,1e-12],'beam_shear':'conservative diagnostic only'},'families':{family:run_family(family) for family in ('s3','s3_rotated','mixed')}}
    for rows in results['families'].values():
        for row in rows:
            for measurements in row['elements'].values():
                for control in measurements.get('staged_control_vs_same_u',{}).values():
                    assert control['numerical_differences']==0, 'diagnostic stage control differs numerically from native same-u oracle'
    Path(args.output).write_text(json.dumps(results,indent=2))
    print(json.dumps({'output':args.output,'cases':sum(len(v) for v in results['families'].values())}))
