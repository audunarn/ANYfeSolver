"""Opt-in saved-data diagnostics; no solver state or friction law is changed.

Traction inversion checks reported material history, not an independent traction
law or nonlinear solution. Sampled pressure weighting is not a path integral.
"""
from fractions import Fraction
import copy
import math
import numpy as np

SCHEMA='S3_SAVED_MATERIAL_WORK_DIAGNOSTIC_V1'


def sliding_pressure_ramp_reference(*,area_m2,mu,stiffness_pa_per_m,motion_m,pressure_start_pa,pressure_end_pa,stages):
    """Exact binary-input algebra for a fixed-normal full positive-slip ramp."""
    if type(stages) is not int or stages<=0:raise ValueError('stages must be a positive exact integer')
    values=(area_m2,mu,stiffness_pa_per_m,motion_m,pressure_start_pa,pressure_end_pa)
    if any(isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in values):raise ValueError('positive finite ramp inputs required')
    A,m,k,S,p0,p1=map(lambda x:Fraction(float(x)),values);c0=m*p0;c1=m*p1;dc=c1-c0;gamma=S-dc/k
    if gamma<=0:raise ValueError('ramp is not in the declared full positive-slip regime')
    exact=A*(c0+c1)*gamma/2;bias=A*dc*gamma/(2*stages)
    return dict(exact_plastic_loss_j=float(exact),endpoint_plastic_loss_j=float(exact+bias),endpoint_bias_j=float(bias),
        exact_resisting_work_j=float(A*(c0+c1)*S/2),storage_change_j=float(A*(c1*c1-c0*c0)/(2*k)),plastic_motion_m=float(gamma),
        scope='fixednormal_linearpressure_full_positive_slip_not_coupled_path_integral')


def _array(value,shape):
    a=np.array(value,dtype=float,copy=True)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid finite diagnostic array')
    return a


def reconstruct_material_history(data):
    """Reconstruct anchors/chords from skins and traction; never divides D by p.

    Input samples carry physical source points, shared target nodal skins, stable
    facet/material IDs, endpoint pressure and traction. Coordinates and the
    coplanar-data tolerance belong to this diagnostic, not solver admission.
    Source-point/material-feature correspondence and shared context are trusted
    declared inputs; external evidence is required to authenticate them.
    """
    if data.get('schema')!=SCHEMA:raise ValueError('unknown material-work diagnostic schema')
    if any(isinstance(data[key],bool) for key in ('stiffness_pa_per_m','mu','coordinate_tolerance_m')):raise ValueError('boolean diagnostic scale')
    k=float(data['stiffness_pa_per_m']);mu=float(data['mu']);tol=float(data['coordinate_tolerance_m'])
    if not all(math.isfinite(v) and v>0 for v in (k,mu,tol)):raise ValueError('positive finite diagnostic scales required')
    if tol>1e-10:raise ValueError('diagnostic coordinate budget exceeds registered restart comparison budget')
    context=copy.deepcopy(data.get('declared_context'))
    if not isinstance(context,dict) or not context:raise ValueError('shared declared model/program/load context required')
    features=data['features'];N=len(features)
    if not N or len({x['id'] for x in features})!=N:raise ValueError('stable unique material feature IDs required')
    for feature in features:
        nodes=feature['source_node_ids'];bary=_array(feature['barycentric'],(3,))
        if type(feature['source_element_id']) is not int or len(nodes)!=3 or len(set(nodes))!=3 or np.any(bary<=0) or abs(sum(bary)-1)>128*np.finfo(float).eps:raise ValueError('invalid immutable material feature declaration')
    areas=_array([x['reference_area_m2'] for x in features],(N,))
    if np.any(areas<=0):raise ValueError('every positive material support must be retained')
    ids=data['target_node_ids'];faces=data['target_faces']
    if not ids or len(set(ids))!=len(ids) or any(type(n) is not int or n<=0 for n in ids):raise ValueError('unique positive exact target node IDs required')
    rows={node:i for i,node in enumerate(ids)};connectivity={int(e):tuple(nodes) for e,nodes in faces.items()}
    if not connectivity or any(len(ns)!=3 or len(set(ns))!=3 or any(n not in rows for n in ns) for ns in connectivity.values()):raise ValueError('invalid target material facet topology')
    if not data['samples']:raise ValueError('empty material history has no work diagnostic')
    previous=[None]*N;output=[];last_position=-math.inf;checkpoint_error=0.
    for index,sample in enumerate(data['samples']):
        if type(sample['step_index']) is not int or sample['step_index']!=index+1:raise ValueError('complete ordered physical sample inventory required')
        position=float(sample['user_position'])
        if not math.isfinite(position) or position<=last_position:raise ValueError('strictly ordered declared sample positions required')
        last_position=position
        x=_array(sample['source_points_m'],(N,3));nodes=_array(sample['target_nodes_m'],(len(ids),3))
        tractions=_array(sample['traction_pa'],(N,3));pressure=_array(sample['pressure_pa'],(N,))
        if np.any(pressure<0):raise ValueError('negative reported pressure')
        branches=sample['branch'];elements=sample['target_elements']
        if len(branches)!=N or len(elements)!=N:raise ValueError('incomplete material correspondence')
        corners={e:nodes[[rows[n] for n in ns]] for e,ns in connectivity.items()}
        raw=sum((np.cross(c[1]-c[0],c[2]-c[0]) for c in corners.values()),np.zeros(3));length=np.linalg.norm(raw)
        if not np.isfinite(length) or length<=0:raise ValueError('unresolved target normal')
        normal=raw/length;origin=next(iter(corners.values()))[0];P=np.eye(3)-np.outer(normal,normal)
        if max(abs((nodes-origin)@normal))>tol:raise ValueError('noncoplanar target diagnostic data')
        if any(np.dot(np.cross(c[1]-c[0],c[2]-c[0]),normal)<=0 for c in corners.values()):raise ValueError('inconsistent target material winding')
        dropped=int(np.argmax(abs(normal)));axes=[a for a in range(3) if a!=dropped]
        chords=[];anchors=[];transfers=[]
        for j,(branch,element) in enumerate(zip(branches,elements)):
            if branch not in ('open','activation','stick','slip'):raise ValueError('unknown reported material branch')
            if abs(tractions[j]@normal)>k*tol:raise ValueError('reported traction is not tangential')
            if branch=='open':
                if pressure[j]!=0 or np.linalg.norm(tractions[j])>k*tol or element is not None:raise ValueError('inconsistent inactive material record')
                previous[j]=None;anchors.append(None);chords.append(0.);transfers.append(False);continue
            if pressure[j]<=0 or type(element) is not int or element not in corners:raise ValueError('active material facet/pressure correspondence missing')
            c=corners[element];y=x[j]-normal*np.dot(normal,x[j]-origin)+tractions[j]/k
            matrix=(c[1:]-c[0])[:,axes].T
            try:uv=np.linalg.solve(matrix,(y-c[0])[axes])
            except np.linalg.LinAlgError as exc:raise ValueError('singular recorded material facet') from exc
            bary=np.array([1.-sum(uv),*uv]);reconstructed=bary@c
            if np.linalg.norm(reconstructed-y)>tol or np.min(bary)<-tol/max(np.linalg.norm(matrix),1e-300):raise ValueError('traction-inverted anchor is outside recorded facet')
            old=previous[j];transfer=old is not None and old[0]!=element
            if branch=='activation':
                if old is not None or np.linalg.norm(tractions[j])>k*tol:raise ValueError('inconsistent stress-free activation')
                chord=0.
            else:
                if old is None:raise ValueError('missing preceding committed material anchor')
                advected=old[1]@corners[old[0]]
                chord=float(np.linalg.norm(P@(reconstructed-advected)))
                if branch=='stick' and chord>tol:raise ValueError('stick anchor changed beyond diagnostic coordinate tolerance')
                if branch=='stick':chord=0.
            previous[j]=(element,bary);anchors.append(dict(element=element,barycentric=bary.tolist(),position_m=reconstructed.tolist()));chords.append(chord);transfers.append(transfer)
        loss=float(mu*np.dot(areas*pressure,chords))
        if 'checkpoint_anchors' in sample:
            saved=sample['checkpoint_anchors']
            if len(saved)!=N:raise ValueError('checkpoint material inventory differs')
            for actual,anchor in zip(saved,anchors):
                if anchor is None:
                    if actual['target_element'] is not None or actual['barycentric'] is not None:raise ValueError('checkpoint release correspondence differs')
                else:
                    if actual['target_element']!=anchor['element']:raise ValueError('checkpoint facet identity differs')
                    saved_bary=_array(actual['barycentric'],(3,))
                    if np.min(saved_bary)<-tol or abs(sum(saved_bary)-1)>128*np.finfo(float).eps:raise ValueError('invalid checkpoint barycentric coordinates')
                    saved_point=saved_bary@corners[anchor['element']]
                    error=float(np.linalg.norm(saved_point-anchor['position_m']));checkpoint_error=max(checkpoint_error,error)
                    if error>tol:raise ValueError('checkpoint anchor reconstruction exceeds coordinate budget')
        published=float(_array(sample['published_loss_j'],()))
        if published<0:raise ValueError('negative published loss')
        output.append(dict(step_index=sample['step_index'],user_position=position,pressure_pa=pressure.tolist(),plastic_chord_m=chords,
            reconstructed_loss_j=loss,published_loss_j=published,loss_error_j=loss-published,
            branch=list(branches),target_elements=list(elements),facet_transfer=transfers,anchors=anchors))
    return dict(schema=SCHEMA,feature_ids=[x['id'] for x in features],material_feature_manifest=copy.deepcopy(features),target_faces=copy.deepcopy(faces),reference_area_m2=areas.tolist(),mu=mu,stiffness_pa_per_m=k,target_node_ids=list(ids),declared_context=context,samples=output,
        checkpoint_anchor_max_error_m=checkpoint_error,total_reconstructed_loss_j=sum(x['reconstructed_loss_j'] for x in output),
        scope='traction_inversion_material_history_consistency_not_independent_traction_law_or_nonlinear_solution')


def compare_material_resolutions(coarse,fine):
    """Exact sampled-data difference split, not a continuum causal decomposition."""
    if any(coarse[key]!=fine[key] for key in ('stiffness_pa_per_m','target_node_ids','declared_context')):raise ValueError('paired law/target/context mismatch')
    if coarse['material_feature_manifest']!=fine['material_feature_manifest'] or coarse['target_faces']!=fine['target_faces'] or coarse['reference_area_m2']!=fine['reference_area_m2'] or coarse['mu']!=fine['mu']:raise ValueError('material feature/reference measure mismatch')
    area=np.array(coarse['reference_area_m2']);mu=coarse['mu'];intervals=[];last=0.;offset=0
    for row in coarse['samples']:
        end=row['user_position'];group=[]
        while offset<len(fine['samples']) and fine['samples'][offset]['user_position']<=end:
            group.append(fine['samples'][offset]);offset+=1
        if not group or group[-1]['user_position']!=end or group[0]['user_position']<=last:raise ValueError('resolutions lack exact common endpoint correspondence')
        pc=np.array(row['pressure_pa']);pf=np.array(group[-1]['pressure_pa']);gc=np.array(row['plastic_chord_m'])
        gf=sum((np.array(x['plastic_chord_m']) for x in group),np.zeros(len(area)))
        sampling=sum(mu*np.dot(area*(np.array(x['pressure_pa'])-pf),x['plastic_chord_m']) for x in group)
        endpoint=float(mu*np.dot(area*(pf-pc),gf));transport=float(mu*np.dot(area*pc,gf-gc))
        delta=sum(x['reconstructed_loss_j'] for x in group)-row['reconstructed_loss_j']
        intervals.append(dict(start=last,end=end,fine_minus_coarse_loss_j=delta,within_interval_pressure_sampling_j=float(sampling),
            endpoint_pressure_solution_difference_j=endpoint,chord_history_response_difference_j=transport,
            split_error_j=delta-sampling-endpoint-transport,coarse_branch=row['branch'],fine_branches=[x['branch'] for x in group],
            coarse_facet=row['target_elements'],fine_facets=[x['target_elements'] for x in group],
            scope='sampled_pressure_term_includes_activation_sampling_not_automatically_smooth_quadrature'))
        last=end
    if offset!=len(fine['samples']):raise ValueError('unmatched fine samples')
    keys=('fine_minus_coarse_loss_j','within_interval_pressure_sampling_j','endpoint_pressure_solution_difference_j','chord_history_response_difference_j','split_error_j')
    return dict(intervals=intervals,totals={k:sum(x[k] for x in intervals) for k in keys},
        scope='same_feature_sampled_response_split_not_continuum_integral_or_causal_solution_decomposition')


def diagnose_work_bundle(bundle):
    """Intended file-consumer entry; pure diagnostic with explicit correspondence."""
    if set(bundle)!= {'coarse','fine'}:raise ValueError('paired coarse/fine diagnostic inputs required')
    coarse=reconstruct_material_history(bundle['coarse']);fine=reconstruct_material_history(bundle['fine'])
    return dict(coarse=coarse,fine=fine,comparison=compare_material_resolutions(coarse,fine))
