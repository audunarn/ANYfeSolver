"""Ordered development decks and comparison for plate, flatbar and their union.

ANYfileIO owns record parsing/writing and neutral identities. No external
application is launched here. Existing source mechanics and criteria are reused.
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from .sestra_verification import identity,Unsupported

BASELINE_SHA='44f32e22475cfddcf2a67a855ca18b66acd234d4abb7b7c28dcc898f0ff23d6d'
ORDER=('plate','flatbar','combined')
CONTROL='''HEAD
COMM Generic reproducible linear-static development control.
CMAS    0.   1.   1.   0.   0.   0.   0.        0.
SOLM              0.                            0.        0.
ELOP                   1.   0.   1.
ITOP    1.
INAM  {prefix}
LNAM  {prefix} UNFORMATTED
RNAM  {prefix} FORMATTED
RSEL    1.   0.   0.   0.   0.   0.   1.   0.
RETR    3.
IDTY    1.
DYMA    1.   0.   0.
Z
'''

def prepare_three_cases(baseline,output):
    from anyfileio.sesam.records import FemRawRecord,canonical_record_lines
    from anyfileio.sesam.document import read_sesam_fem_document,parse_sesam_fem_records
    from anyfileio.sesam.exporter import write_sesam_fem_document
    if identity(baseline)['sha256']!=BASELINE_SHA:raise Unsupported('three-case generator requires immutable adopted 400-Q4/20-BEAS15 baseline hash')
    document=read_sesam_fem_document(baseline,strict=True)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    suite={'schema_version':1,'baseline':identity(baseline),'order':list(ORDER),'cases':{}}
    for case in ORDER:
        directory=root/case;directory.mkdir()
        selected={eid:e for eid,e in document.elements.items() if case=='combined' or (case=='plate' and e.type_code==24) or (case=='flatbar' and e.type_code==15)}
        used={n for e in selected.values() for n in e.node_ids}
        internal_elements={old:i for i,old in enumerate(sorted(document.element_internal_to_external),1) if document.element_internal_to_external[old] in selected}
        internal_elements={old:i for i,old in enumerate(sorted(internal_elements),1)}
        internal_nodes={old:i for i,old in enumerate(sorted(k for k,v in document.node_internal_to_external.items() if v in used),1)}
        records=[]
        allowed={'IDENT','UNITS','TDMATER','MISOSEL','GELTH','TDSECT','GBEAMG','GBARM','GUNIVEC','GECCEN','GNODE','GCOORD','BNBCD','GELMNT1','GELREF1','TDLOAD','BEUSLO','IEND'}
        for record in document.raw_records:
            name=record.name;v=list(record.numeric_fields)
            if name not in allowed:continue
            if case=='plate' and name in {'TDSECT','GBEAMG','GBARM','GUNIVEC','GECCEN'}:continue
            if case=='flatbar' and name in {'GELTH','GECCEN','BEUSLO'}:continue
            if name=='GNODE':
                if int(v[1]) not in internal_nodes:continue
                v[1]=internal_nodes[int(v[1])]
            elif name=='GCOORD':
                if int(v[0]) not in internal_nodes:continue
                v[0]=internal_nodes[int(v[0])]
                if case=='flatbar':v[3]=-.0500000007
            elif name=='BNBCD':
                if int(v[0]) not in internal_nodes:continue
                v[0]=internal_nodes[int(v[0])]
            elif name=='GELMNT1':
                if int(v[0]) not in selected:continue
                v[1]=internal_elements[int(v[1])]
                v[4:]=[internal_nodes[int(x)] for x in v[4:]]
            elif name=='GELREF1':
                if int(v[0]) not in internal_elements:continue
                v[0]=internal_elements[int(v[0])]
                if case=='flatbar':v[10]=0;v=v[:12]
            elif name=='BEUSLO':v[4]=internal_elements[int(v[4])]
            elif name=='IEND' and case=='flatbar':
                tip=max(used,key=lambda n:document.nodes[n].coordinates[1]);old=next(k for k,x in document.node_internal_to_external.items() if x==tip)
                loads=FemRawRecord('BNLOAD',(1.,1.,0.,0.,float(internal_nodes[old]),6.,0.,0.,-1000.,0.,0.,0.),(),0,0,())
                records.append(replace(loads,raw_lines=canonical_record_lines(loads)))
            numeric=replace(record,numeric_fields=tuple(map(float,v)),text_fields=(),raw_lines=())
            # Numeric fields changed; retain textual names without private DATE metadata.
            lines=canonical_record_lines(numeric)+tuple(' '*8+x for x in record.text_fields)
            records.append(replace(numeric,text_fields=record.text_fields,raw_lines=lines))
        prepared=parse_sesam_fem_records(records,strict=True)
        fem=directory/f'{case}_T1.FEM';write_sesam_fem_document(prepared,fem,mode='raw')
        (directory/'sestra_control.inp').write_bytes(CONTROL.replace('\n','\r\n').encode('ascii'))
        m={'schema_version':1,'case_id':case,'profile':'sequential_plate_flatbar_combined','fem':fem.name,'fem_sha256':identity(fem)['sha256'],'load_case':1,'units':'SI','frame':'GLOBAL','node_ids':sorted(used),'shell_ids':sorted(e for e,v in selected.items() if v.type_code==24),'beam_ids':sorted(e for e,v in selected.items() if v.type_code==15),'observables':['global_nodal_translations']+(['physical_beam_station_forces'] if case!='plate' else []),'translation_tolerances':{'absolute_m':1e-8,'relative':.001},'beam_tolerances':{'absolute_n':.001,'absolute_nm':.001,'relative':.001},'reference_sif':f'{case}_R1.SIF','reference_fem_sha256':identity(fem)['sha256'],'execution':{'argv':['C:/Program Files/DNV/Sestra V10.20-00/Bin/sestra.exe','/dsf'],'control_template':'sestra_control.inp','prefix_token':'{prefix}','prefix':case+'_','timeout_seconds':30}}
        (directory/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
        suite['cases'][case]={'manifest':str(directory/'manifest.json'),'fem':identity(fem),'nodes':len(used),'shells':len(m['shell_ids']),'beams':len(m['beam_ids'])}
    (root/'suite.json').write_text(json.dumps(suite,indent=2),encoding='utf-8')
    return suite

def read_manifest(path):
    path=Path(path).resolve();m=json.loads(path.read_text(encoding='utf-8'))
    if m.get('schema_version')!=1 or m.get('profile')!='sequential_plate_flatbar_combined' or m.get('case_id') not in ORDER or m.get('load_case')!=1 or m.get('units')!='SI' or m.get('frame')!='GLOBAL':raise Unsupported('unadmitted three-case manifest')
    if m.get('translation_tolerances')!={'absolute_m':1e-8,'relative':.001} or m.get('beam_tolerances')!={'absolute_n':.001,'absolute_nm':.001,'relative':.001}:raise Unsupported('three-case criteria differ from adopted fixed criteria')
    for key in ('node_ids','shell_ids','beam_ids'):
        ids=m[key]
        if any(type(x)is not int or x<=0 for x in ids) or len(ids)!=len(set(ids)):raise Unsupported('invalid/duplicate requested identities')
    if not m['node_ids'] or set(m['shell_ids'])&set(m['beam_ids']):raise Unsupported('empty/overlapping requested coverage')
    for key in ('fem','reference_sif'):m[key]=str((path.parent/m[key]).resolve())
    m['manifest_identity']=identity(path)
    if identity(m['fem'])['sha256']!=m['fem_sha256'] or m['reference_fem_sha256']!=m['fem_sha256']:raise Unsupported('source FEM hash/binding differs')
    return m

def admit_case(m):
    from .sesam_fem import import_sesam_fem
    from .sestra_mixed_verification import admit_mixed
    if m['case_id']!='flatbar':return admit_mixed(m)
    imported=import_sesam_fem(m['fem'],strict=True);d=imported.document;model=imported.model
    allowed={'IDENT','UNITS','TDMATER','MISOSEL','TDSECT','GBEAMG','GBARM','GUNIVEC','GNODE','GCOORD','BNBCD','GELMNT1','GELREF1','TDLOAD','BNLOAD','IEND'}
    if set(d.record_counts)-allowed or d.dependencies or d.coordinate_transforms or d.eccentricities:raise Unsupported('unadmitted isolated flatbar semantics')
    if d.header.unit_values!=(5.,1.,1.,1.,1.) or d.record_counts.get('TDLOAD')!=1 or any(r.raw_values[:2]!=(4.,1.) for r in d.load_records if r.record_name=='TDLOAD'):
        raise Unsupported('isolated flatbar requires explicit SI and sole TDLOAD case1')
    if set(d.nodes)!=set(m['node_ids']) or set(d.elements)!=set(m['beam_ids']) or m['shell_ids'] or any(e.type_code!=15 for e in d.elements.values()):raise Unsupported('isolated flatbar identity/type coverage mismatch')
    if len(d.nodes)!=21 or len(d.elements)!=20 or len(d.boundaries)!=1 or d.boundaries[0].dof_flags!=(1,)*6 or any(d.boundaries[0].prescribed_values):raise Unsupported('isolated flatbar requires 20 segments/21 nodes and one fully fixed root')
    loads=[r for r in d.load_records if r.record_name!='TDLOAD']
    if len(loads)!=1 or loads[0].record_name!='BNLOAD':raise Unsupported('isolated flatbar requires one tip BNLOAD')
    v=loads[0].raw_values
    if v[:4]!=(1.,1.,0.,0.) or v[5:]!=(6.,0.,0.,-1000.,0.,0.,0.):raise Unsupported('isolated beam requires real global centroid tip Fz=-1000N')
    tip=d.node_internal_to_external[int(v[4])]
    xyz=np.array([n.coordinates for n in d.nodes.values()])
    if not np.all(xyz[:,0]==1) or not np.all(xyz[:,2]==-.0500000007) or float(xyz[:,1].min())!=0 or float(xyz[:,1].max())!=2 or d.nodes[tip].coordinates[1]!=2 or d.nodes[d.boundaries[0].node_id].coordinates[1]!=0:raise Unsupported('isolated flatbar centerline/root/tip contract mismatch')
    if len(model.load_cases)!=1 or set(model.load_cases[0].nodal_loads)!={tip} or not np.array_equal(model.load_cases[0].nodal_loads[tip],[0,0,-1000,0,0,0]):raise Unsupported('native tip load differs from source BNLOAD')
    for diag in imported.diagnostics:
        if not (diag.code=='FEM120' and diag.record_name=='BNLOAD' and diag.message=='BNLOAD is preserved but not translated into solver semantics yet'):raise Unsupported('unadmitted isolated beam diagnostic: '+str(diag))
    for eid,e in d.elements.items():
        r=d.element_references[eid];mat=d.materials[r.material_id];effective=model.get_material(model.mesh.elements[eid].material_name);sec=d.sections[r.section_id]
        if r.raw_values[10]!=0 or any(r.raw_values[2:8]) or r.raw_values[9]!=0 or not r.transform_id:raise Unsupported('isolated beam requires explicit local-z, no release/offset')
        if model.mesh.elements[eid].cross_section.get('orientation')!=(0.,0.,1.):raise Unsupported('isolated upright local-z must match global-z')
        nominal={'web_height':.1,'web_thickness':.01,'area':.001,'iy':.01*.1**3/12,'iz':.1*.01**3/12}
        if sec.section_type!='FB' or any(getattr(sec,key) is None or not np.isclose(getattr(sec,key),value,rtol=np.finfo(np.float32).eps,atol=0) for key,value in nominal.items()):raise Unsupported('isolated flatbar must be nominal upright FB100x10 within serialized precision')
        section=model.mesh.elements[eid].cross_section
        if any(not np.isclose(section[key],value,rtol=np.finfo(np.float32).eps,atol=0) for key,value in {'area':.001,'Iy':nominal['iy'],'Iz':nominal['iz']}.items()):raise Unsupported('effective native flatbar area/inertias differ from nominal section')
        if mat.raw_values[1:4]!=(effective.elastic_modulus,effective.poisson_ratio,effective.density):raise Unsupported('native beam material differs from explicit source')
    return imported

def assert_reference_identity(source,result,m):
    from .sestra_verification import RESULT_METADATA
    if any(r.name not in RESULT_METADATA for r in result.unknown_records) or result.dependencies or result.coordinate_transforms:raise Unsupported('unsupported reference model semantics')
    if source.header.unit_values!=result.header.unit_values or {k:v.coordinates for k,v in source.nodes.items()}!={k:v.coordinates for k,v in result.nodes.items()} or {k:(v.type_code,v.node_ids) for k,v in source.elements.items()}!={k:(v.type_code,v.node_ids) for k,v in result.elements.items()}:raise Unsupported('reference node/topology/units identity differs')
    def tail(v):
        v=list(v)
        while v and v[-1]==0:v.pop()
        return tuple(v)
    def refs(d):return {k:(v.material_id,v.section_id,v.transform_id,v.nodal_transform_ids,tail(v.raw_values)) for k,v in d.element_references.items()}
    if refs(source)!=refs(result) or source.eccentricities!=result.eccentricities or source.unit_vectors!=result.unit_vectors or source.boundaries!=result.boundaries:raise Unsupported('reference source mechanics/support identity differs')
    if set(source.materials)!=set(result.materials) or set(source.sections)!=set(result.sections):raise Unsupported('reference property ID coverage differs')
    for key,mat in source.materials.items():
        for field in ('elastic_modulus','poisson_ratio','density'):
            a,b=getattr(mat,field),getattr(result.materials[key],field)
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a,b,rtol=np.finfo(np.float32).eps,atol=0):raise Unsupported('reference material property differs')
    if any(tail(v.raw_values)!=tail(result.sections[k].raw_values) for k,v in source.sections.items()):raise Unsupported('reference section values differ')
    def loads(d):return sorted((r.record_name,tail(r.raw_values)) for r in d.load_records if r.record_name!='TDLOAD')
    if loads(source)!=loads(result):raise Unsupported('reference physical loads differ from bound source')

def solve_case(m):
    from .assembly import solve_linear,compute_reactions
    imported=admit_case(m);model=imported.model
    if set(model.mesh.nodes)!=set(m['node_ids']):raise Unsupported('native node coverage differs')
    u,info=solve_linear(model,model.load_cases[0],constraint_mode='transformation')
    if info.get('convergence_info',{}).get('status')!='converged' or not np.all(np.isfinite(u)):raise Unsupported('native solve failed/nonfinite')
    translation={node:u[np.asarray(model.mesh.dof_manager.get_node_dofs(node))[:3]].tolist() for node in m['node_ids']}
    oracle=None
    if m['case_id']=='flatbar':
        tip=max(imported.document.nodes,key=lambda n:imported.document.nodes[n].coordinates[1]);element=model.mesh.elements[m['beam_ids'][0]];s=element.cross_section;mat=model.get_material(element.material_name)
        E=mat.elastic_modulus;G=E/(2*(1+mat.poisson_ratio));L=2.;P=1000.
        expected=-P*L**3/(3*E*s['Iy'])-P*L/(G*s['shear_factor_z']*s['area'])
        oracle={'tip_node':tip,'tip_uz_m':translation[tip][2],'analytic_timoshenko_uz_m':expected,'absolute_error_m':abs(translation[tip][2]-expected),'passed':bool(np.isclose(translation[tip][2],expected,rtol=1e-6,atol=1e-10)),'centerline':'global x=1,y=0..2,z=-.0500000007; upright localz=globalz; no attachment offsets'}
    reactions=compute_reactions(model,u,model.load_cases[0]);resultant=np.zeros(6);load=np.zeros(6)
    for nid,value in reactions.items():
        resultant[:3]+=value[:3];resultant[3:]+=value[3:]+np.cross(imported.document.nodes[nid].coordinates,value[:3])
    for nid,value in model.load_cases[0].nodal_loads.items():
        load[:3]+=value[:3];load[3:]+=value[3:]+np.cross(imported.document.nodes[nid].coordinates,value[:3])
    for eid,pressure in model.load_cases[0].pressure_loads.items():
        xy=np.array([imported.document.nodes[n].coordinates for n in imported.document.elements[eid].node_ids]);nxt=np.roll(xy,-1,axis=0);cross=xy[:,0]*nxt[:,1]-nxt[:,0]*xy[:,1];area=sum(cross)/2;centroid=[sum((xy[:,0]+nxt[:,0])*cross)/(6*area),sum((xy[:,1]+nxt[:,1])*cross)/(6*area),0.];force=np.array([0.,0.,pressure*area]);load[:3]+=force;load[3:]+=np.cross(centroid,force)
    balance={'load':load.tolist(),'reactions':resultant.tolist(),'residual':(load+resultant).tolist(),'passed':bool(np.all(np.abs(load+resultant)<=1e-6+1e-8*np.abs(load))),'absolute_n_nm':1e-6,'relative':1e-8}
    return imported,u,translation,oracle,balance,info

def compare_case(manifest,output):
    import sys,os,uuid,time
    from anyfileio.sesam import sif,semantics,document
    from . import assembly,elements,e4_pl_element
    from .sesam_fem import importer
    from threadpoolctl import threadpool_info
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);case=root/uuid.uuid4().hex;case.mkdir();start=time.perf_counter()
    report={'schema_version':1,'status':'not_executed','qualification':False,'python':sys.version,'sources':{x.__name__:identity(x.__file__) for x in (sif,semantics,document,assembly,elements,e4_pl_element,importer)},'harness':identity(__file__),'runtime':{'numba_disable_jit':os.environ.get('NUMBA_DISABLE_JIT'),'threadpools':threadpool_info()},'comparison_units':{'translations':'m','beam_forces':'N','beam_moments':'Nm'},'observable':'all global nodal translations; all physical beam force/moment stations','sampling_limit':'rotations and shell stress are not acceptance observables; translations resolve plate bending; no membrane-zero oracle'}
    try:
        m=read_manifest(manifest);report.update(case_id=m['case_id'],contract=m,input=identity(m['fem']))
        if not Path(m['reference_sif']).is_file():report['reason']='reference SIF not supplied; no external process launched'
        else:
            result_doc=document.read_sesam_fem_document(m['reference_sif'],strict=True);source=document.read_sesam_fem_document(m['fem'],strict=True);assert_reference_identity(source,result_doc,m)
            reference=sif.read_sesam_sif_translations(m['reference_sif'],m['load_case'],m['node_ids'])
            if reference.external_result_case!=m['load_case']:raise Unsupported('wrong external nodal result case')
            imported,u,native,oracle,balance,info=solve_case(m);failures=[]
            for nid in m['node_ids']:
                a=np.asarray(native[nid]);b=np.asarray(reference.translations[nid]);limit=1e-8+.001*np.abs(b)
                for i in np.flatnonzero(np.abs(a-b)>limit):failures.append({'kind':'translation','node':nid,'component':('UX','UY','UZ')[i],'native':float(a[i]),'reference':float(b[i]),'error':float(abs(a[i]-b[i])),'limit':float(limit[i])})
            beam_native={};beam_reference=None
            if m['beam_ids']:
                beam_reference=sif.read_sesam_sif_beam_forces(m['reference_sif'],m['load_case'])
                if set(beam_reference.element_forces)!=set(m['beam_ids']) or any(x['external_result_case']!=m['load_case'] for x in beam_reference.frame_provenance.values()):raise Unsupported('beam result coverage/case differs')
                for eid in m['beam_ids']:
                    e=imported.model.mesh.elements[eid];a=e.recover_linear_section_forces(imported.model.mesh,u,imported.model.get_material(e.material_name),beam_reference.station_coordinates[eid]);b=np.asarray(beam_reference.element_forces[eid]);beam_native[eid]=a.tolist()
                    if a.shape!=b.shape or not np.all(np.isfinite(a)):raise Unsupported('beam stations incomplete/nonfinite')
                    limit=.001+.001*np.abs(b)
                    for station,component in np.argwhere(np.abs(a-b)>limit):failures.append({'kind':'beam','element':eid,'station':int(station),'component':int(component),'native':float(a[station,component]),'reference':float(b[station,component]),'error':float(abs(a[station,component]-b[station,component])),'limit':float(limit[station,component])})
            report.update(status='passed' if not failures and balance['passed'] and (oracle is None or oracle['passed']) else 'failed',reference=identity(m['reference_sif']),native_translations_m=native,reference_translations_m=reference.translations,nodal_provenance=reference.provenance,native_beam_station_forces=beam_native,reference_beam_station_forces=None if beam_reference is None else beam_reference.element_forces,beam_provenance=None if beam_reference is None else beam_reference.frame_provenance,balance=balance,analytical_oracle=oracle,failures=failures,counts={'nodes':len(native),'translation_components':3*len(native),'beams':len(beam_native),'beam_station_components':sum(np.asarray(v).size for v in beam_native.values())},element_classes=sorted(set(type(e).__module__+'.'+type(e).__qualname__ for e in imported.model.mesh.elements.values())),solver={'type':info['solver_type'],'constraint_mode':info['constraint_mode']})
    except Unsupported as exc:report.update(status='unsupported',reason=str(exc))
    except Exception as exc:report.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
    report.update(process_wall_seconds=time.perf_counter()-start,report_path=str(case/'report.json'));(case/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');return report

def compare_suite(suite,output):
    suite=json.loads(Path(suite).read_text(encoding='utf-8'));results={}
    if (Path(output)/'suite_report.json').exists():raise Unsupported('suite output already contains evidence; choose a fresh directory')
    if suite['order']!=list(ORDER):raise Unsupported('fixed ordered three-case workflow required')
    for case in ORDER:results[case]=compare_case(suite['cases'][case]['manifest'],Path(output)/case)
    statuses={r['status'] for r in results.values()};status='passed' if statuses=={'passed'} else 'failed' if statuses&{'failed','unsupported'} else 'not_executed'
    summary={'schema_version':1,'order':list(ORDER),'status':status,'cases':{k:{'status':v['status'],'report_path':v['report_path'],'counts':v.get('counts'),'failure_count':len(v.get('failures',[]))} for k,v in results.items()},'qualification':False}
    Path(output).mkdir(parents=True,exist_ok=True);(Path(output)/'suite_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary
