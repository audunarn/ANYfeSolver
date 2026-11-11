"""Bounded pressure-loaded Q4/BEAS15 development slice; no syntax parsing."""
import numpy as np
from .sestra_verification import Unsupported,identity,_finite,COMPONENTS,RESULT_METADATA
from .sesam_fem import import_sesam_fem,read_sesam_fem_document

PROFILE='mixed_q4_beas15_constant_pressure'

def rectangular_serialization_admission(xyz):
    xyz=np.asarray(xyz,dtype=float);edges=np.roll(xyz,-1,axis=0)-xyz
    bound=4*np.finfo(np.float32).eps*float(np.max(np.abs(xyz)))
    closure=float(max(np.max(np.abs(edges[0]+edges[2])),np.max(np.abs(edges[1]+edges[3]))))
    orthogonal=float(abs(np.dot(edges[0],edges[1])))
    orthogonal_bound=bound*(np.linalg.norm(edges[0])+np.linalg.norm(edges[1]))+2*bound**2
    valid=np.all(xyz[:,2]==0) and np.cross(edges[0],edges[1])[2]>0 and closure<=bound and orthogonal<=orthogonal_bound
    return bool(valid),{'coordinate_bound_m':bound,'closure_residual_m':closure,'orthogonal_residual_m2':orthogonal,'orthogonal_bound_m2':float(orthogonal_bound)}

def validate_manifest(m,path):
    required={'schema_version','case_id','fem','reference_sif','reference_fem_sha256','load_case','units','frame','shell_ids','beam_ids','tolerances','beam_tolerances'}
    if not required<=m.keys() or m['schema_version']!=1 or m['units']!='SI' or m['frame']!='GLOBAL' or type(m['load_case']) is not int or m['load_case']<=0:
        raise Unsupported('incomplete mixed SI/GLOBAL manifest')
    for key in ('shell_ids','beam_ids'):
        ids=m[key]
        if not ids or len(ids)!=len(set(ids)) or any(type(i)is not int or i<=0 for i in ids):raise Unsupported('mixed IDs must be nonempty unique positive integers')
    if set(m['shell_ids'])&set(m['beam_ids']):raise Unsupported('overlapping mixed element IDs')
    if m['tolerances']!={'absolute_pa':1.,'relative':.001} or m['beam_tolerances']!={'absolute_n':.001,'absolute_nm':.001,'relative':.001}:
        raise Unsupported('mixed development tolerances must match adopted contract')
    for key in ('fem','reference_sif'):m[key]=str((path.parent/m[key]).resolve())
    m['manifest_identity']=identity(path)
    return m

def admit_mixed(m):
    imported=import_sesam_fem(m['fem'],strict=True);d=imported.document;model=imported.model
    allowed={'IDENT','DATE','UNITS','TDMATER','MISOSEL','TDSECT','GELTH','GBEAMG','GBARM','TDSCONC','SCONCEPT','SCONMESH','GUNIVEC','GECCEN','GNODE','GCOORD','BNBCD','GELMNT1','GELREF1','TDLOAD','BEUSLO','IEND'}
    if set(d.record_counts)-allowed or d.dependencies or d.coordinate_transforms:raise Unsupported('unadmitted mixed source semantics')
    if d.header.unit_values!=(5.,1.,1.,1.,1.):raise Unsupported('mixed source SI UNITS required')
    if set(d.elements)!=set(m['shell_ids'])|set(m['beam_ids']) or set(model.mesh.elements)!=set(d.elements):raise Unsupported('mixed model coverage differs from manifest/import')
    pressure={}
    for r in d.load_records:
        v=r.raw_values;_finite(v,'mixed load')
        if r.record_name=='TDLOAD':
            if len(v)!=4 or v[:2]!=(4.,float(m['load_case'])):raise Unsupported('wrong source case metadata')
            continue
        if r.record_name!='BEUSLO' or len(v)!=12 or v[0]!=m['load_case'] or v[1:4]!=(1.,0.,0.) or v[4]!=int(v[4]) or v[5:8]!=(4.,0.,2.) or any(x!=v[8] for x in v[8:]):
            raise Unsupported('only sole-case constant four-node real middle-plane pressures admitted')
        eid=d.element_internal_to_external.get(int(v[4]))
        if eid not in m['shell_ids'] or eid in pressure:raise Unsupported('pressure target/coverage ambiguous')
        pressure[eid]=v[8]
    if set(pressure)!=set(m['shell_ids']) or len(model.load_cases)!=1 or model.load_cases[0].pressure_loads!=pressure:
        raise Unsupported('native pressure coverage differs')
    for diagnostic in imported.diagnostics:
        if not (diagnostic.code=='FEM120' and diagnostic.record_name=='BEUSLO' and diagnostic.message=='BEUSLO is preserved but not translated into solver semantics yet'):
            raise Unsupported('unadmitted mixed importer diagnostic: '+str(diagnostic))
    for n in d.nodes.values():
        _finite(n.coordinates,'mixed coordinates')
        if n.coordinate_system_id!=0:raise Unsupported('mixed global coordinates required')
    for bc in d.boundaries:
        if len(bc.dof_flags)!=6 or any(x not in (0,1) for x in bc.dof_flags) or any(bc.prescribed_values):raise Unsupported('mixed binary zero supports required')
    for eid,e in d.elements.items():
        ref=d.element_references[eid];mat=d.materials[ref.material_id];sec=d.sections[ref.section_id];effective=model.get_material(model.mesh.elements[eid].material_name)
        if len(mat.raw_values)<4 or mat.raw_values[1:4]!=(mat.elastic_modulus,mat.poisson_ratio,mat.density) or (effective.elastic_modulus,effective.poisson_ratio,effective.density)!=mat.raw_values[1:4]:raise Unsupported('mixed material source/native mismatch')
        if any(ref.raw_values[2:8]) or ref.raw_values[9]!=0:raise Unsupported('mixed releases/dependencies/options unsupported')
        if eid in m['shell_ids']:
            if e.type_code!=24 or ref.transform_id or ref.raw_values[10] or sec.thickness is None or model.mesh.elements[eid].thickness!=sec.thickness:raise Unsupported('mixed explicit plain Q4 thickness required')
            xyz=np.asarray([d.nodes[n].coordinates for n in e.node_ids])
            edges=np.roll(xyz,-1,axis=0)-xyz
            if not np.all(xyz[:,2]==0) or any(np.cross(edges[i],edges[(i+1)%4])[2]<=0 for i in range(4)):
                raise Unsupported('mixed flat convex CCW nondegenerate XY Q4 required')
        else:
            if e.type_code!=15 or sec.kind!='beam_section' or not ref.transform_id or ref.transform_id not in d.unit_vectors or ref.nodal_transform_ids:raise Unsupported('mixed explicit BEAS15 section/local-z required')
            section=model.mesh.elements[eid].cross_section
            for key in ('area','Iy','Iz','J','shear_factor_y','shear_factor_z'):
                if key not in section or not np.isfinite(section[key]) or section[key]<=0:raise Unsupported('incomplete effective beam section')
    return imported

def compare_mixed(imported,m):
    from anyfileio.sesam.sif import read_sesam_sif_stress,read_sesam_sif_beam_forces
    from .assembly import solve_linear,compute_reactions
    source=imported.document;d=read_sesam_fem_document(m['reference_sif'],strict=True)
    if identity(m['fem'])['sha256']!=m['reference_fem_sha256']:raise Unsupported('mixed imported reference source binding mismatch')
    if any(r.name not in RESULT_METADATA for r in d.unknown_records):raise Unsupported('unadmitted mixed reference records')
    if d.header.unit_values!=source.header.unit_values or {k:v.coordinates for k,v in d.nodes.items()}!={k:v.coordinates for k,v in source.nodes.items()} or {k:(v.type_code,v.node_ids) for k,v in d.elements.items()}!={k:(v.type_code,v.node_ids) for k,v in source.elements.items()}:raise Unsupported('mixed reference model identity mismatch')
    def zero_tail(v):
        v=list(v)
        while v and v[-1]==0:v.pop()
        return tuple(v)
    def refs(doc):return {k:(v.material_id,v.section_id,v.transform_id,v.nodal_transform_ids,zero_tail(v.raw_values)) for k,v in doc.element_references.items()}
    if refs(d)!=refs(source) or d.eccentricities!=source.eccentricities or d.unit_vectors!=source.unit_vectors or d.boundaries!=source.boundaries:raise Unsupported('mixed reference mechanics/support identity mismatch')
    for key,mat in source.materials.items():
        for field in ('elastic_modulus','poisson_ratio','density'):
            if not np.isclose(getattr(mat,field),getattr(d.materials[key],field),rtol=np.finfo(np.float32).eps,atol=0):raise Unsupported('mixed reference material mismatch')
    if set(d.sections)!=set(source.sections):raise Unsupported('mixed reference section IDs differ')
    for sid,section in source.sections.items():
        result=d.sections[sid]
        if section.kind!=result.kind or zero_tail(section.raw_values)!=zero_tail(result.raw_values):raise Unsupported('mixed reference sections differ')
    def loads(doc):return sorted((r.record_name,r.raw_values) for r in doc.load_records if r.record_name!='TDLOAD')
    if loads(d)!=loads(source):raise Unsupported('mixed reference loads differ')
    shell=read_sesam_sif_stress(m['reference_sif'],m['load_case']);beam=read_sesam_sif_beam_forces(m['reference_sif'],m['load_case'])
    if set(shell.element_stress)!=set(m['shell_ids']) or set(beam.element_forces)!=set(m['beam_ids']):raise Unsupported('mixed result coverage differs')
    if any(v['external_result_case']!=m['load_case'] for v in list(shell.frame_provenance.values())+list(beam.frame_provenance.values())):raise Unsupported('wrong mixed external result case')
    model=imported.model;u,info=solve_linear(model,model.load_cases[0],constraint_mode='transformation')
    if info.get('convergence_info',{}).get('status')!='converged':raise Unsupported('mixed native solve failed')
    _finite(u,'mixed displacement');candidate_shell={};candidate_beam={};failures=[]
    def check(eid,kind,candidate,reference,absolute):
        a=np.asarray(candidate);b=np.asarray(reference)
        if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):raise Unsupported('mixed incomplete/nonfinite observations')
        tolerance=absolute+.001*np.abs(b);bad=np.argwhere(np.abs(a-b)>tolerance)
        for index in bad:
            idx=tuple(index);failures.append({'element':eid,'kind':kind,'index':index.tolist(),'native':float(a[idx]),'reference':float(b[idx]),'absolute_error':float(abs(a[idx]-b[idx])),'tolerance':float(np.broadcast_to(tolerance,a.shape)[idx])})
    for eid in m['shell_ids']:
        element=model.mesh.elements[eid];frame=shell.frame_provenance[eid]
        natural=np.asarray(frame['natural_point_coordinates'],dtype=float);physical=np.asarray(frame['physical_point_coordinates'],dtype=float)
        if natural.ndim!=2 or natural.shape[1]!=3 or physical.shape!=natural.shape or not np.all(np.isfinite(natural)) or np.any(np.abs(natural)>1):raise Unsupported('mixed shell requires resolved natural and physical station coordinates')
        groups={}
        for x,y,z in natural:groups.setdefault((x,y),[]).append(z)
        if any(sorted(z)!=[-1.,1.] for z in groups.values()):raise Unsupported('mixed shell requires symmetric top/bottom station pairs')
        coords=element.get_node_coordinates(model.mesh);coordinate_bound=4*np.finfo(np.float32).eps*float(max(np.max(np.abs(coords)),element.thickness))
        for nat,point in zip(natural,physical):
            N,_,_=element.compute_shape_functions(*nat[:2]);expected=N@coords+np.array([0,0,nat[2]*element.thickness/2])
            if np.any(np.abs(expected-point)>coordinate_bound):raise Unsupported('reported shell physical/natural stations disagree with exact source geometry')
        s=element.compute_stresses(model.mesh,u,model.get_material(element.material_name),return_global=True,natural_points=natural[:,:2])
        if 'global_membrane_resultant_tensors' not in s:raise Unsupported('matched membrane recovery must provide formulation-native global resultant tensors')
        mean=np.mean(s['global_membrane_resultant_tensors'],axis=0)/element.thickness
        candidate_shell[eid]=[float(mean[0,0]),float(mean[1,1]),float(mean[2,2]),float(mean[0,1]),float(mean[1,2]),float(mean[2,0])]
        check(eid,'shell_station_mean_pa',candidate_shell[eid],shell.element_stress[eid],1.)
    for eid in m['beam_ids']:
        element=model.mesh.elements[eid];a=element.recover_linear_section_forces(model.mesh,u,model.get_material(element.material_name),beam.station_coordinates[eid]);candidate_beam[eid]=a.tolist()
        check(eid,'beam_physical_station_force_moment',a,beam.element_forces[eid],.001)
    reactions=compute_reactions(model,u,model.load_cases[0]);reaction=np.zeros(6);load=np.zeros(6)
    for nid,value in reactions.items():
        value=np.asarray(value);reaction[:3]+=value[:3];reaction[3:]+=value[3:]+np.cross(source.nodes[nid].coordinates,value[:3])
    for eid,pressure in model.load_cases[0].pressure_loads.items():
        xyz=np.array([source.nodes[n].coordinates for n in source.elements[eid].node_ids]);nxt=np.roll(xyz,-1,axis=0);cross=xyz[:,0]*nxt[:,1]-nxt[:,0]*xyz[:,1];area=float(sum(cross)/2)
        centroid=np.array([sum((xyz[:,0]+nxt[:,0])*cross)/(6*area),sum((xyz[:,1]+nxt[:,1])*cross)/(6*area),0])
        f=np.array([0,0,pressure*area]);load[:3]+=f;load[3:]+=np.cross(centroid,f)
    balanced=bool(np.all(np.abs(reaction+load)<=1e-6+1e-8*np.abs(load)))
    effective_contract={'nodes':len(source.nodes),'elements':len(source.elements),'source_coordinate_identity':'exact FEM/SIF coordinates, no snapping','pressure_case':m['load_case'],'pressure_elements':len(model.load_cases[0].pressure_loads),'pressure_sign':'positive along CCW local+z','materials':{k:{'E_pa':v.elastic_modulus,'nu':v.poisson_ratio,'rho_kg_m3':v.density} for k,v in source.materials.items()},'sections':{k:{'kind':v.kind,'raw_values':v.raw_values} for k,v in source.sections.items()},'beam_effective_sections':{eid:model.mesh.elements[eid].cross_section for eid in m['beam_ids']},'global_eccentricities':source.eccentricities,'element_classes':sorted(set(type(e).__module__+'.'+type(e).__qualname__ for e in model.mesh.elements.values())),'metadata_normalization':'exact zero-tail padding only; material values rtol=float32epsilon/atol0; source/reference coordinate identity exact'}
    return {'status':'passed' if not failures and balanced else 'failed','reference':identity(m['reference_sif']),
            'effective_contract':effective_contract,
            'native_shell_tensors_pa':candidate_shell,'native_beam_physical_station_forces':candidate_beam,
            'reference_shell_tensors_pa':shell.element_stress,'reference_beam_physical_station_forces':beam.element_forces,
            'beam_frame_provenance':beam.frame_provenance,'failures':failures,'comparison_counts':{'shells':len(candidate_shell),'shell_components':6*len(candidate_shell),'beams':len(candidate_beam),'beam_station_components':sum(np.asarray(v).size for v in candidate_beam.values())},
            'balance':{'load_resultant':load.tolist(),'reaction_resultant':reaction.tolist(),'residual':(load+reaction).tolist(),'passed':balanced,'absolute':1e-6,'relative':1e-8},
            'native':{'solver_type':info['solver_type'],'constraint_mode':info['constraint_mode'],'displacement_norm':float(np.linalg.norm(u))},
            'shell_sampling_provenance':shell.frame_provenance,
            'observable':'global_inplane_membrane_matched_station_mean_and_physical_beam_station_forces',
            'sampling_limitation':'in-plane membrane tensor only: no Mindlin transverse shear; matched unweighted symmetric top/bottom stations, no bending-stress/qualification/beam-distributed-load claim'}
