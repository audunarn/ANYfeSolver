"""Discriminating source-based plate/attachment models and same-field probes."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from .sestra_verification import Unsupported,identity
from .sestra_three_case import BASELINE_SHA,CONTROL,assert_reference_identity

STAGE1=('plate_regular20','plate_regular20_thick','strip_nu0','strip_nu0_thick','combined_regular_zero','combined_regular_ecc')
STAGE2=('plate_regular40_thick','combined_regular40_ecc','combined_regular20_underplate')
NATIVE_STRIP=('strip_nu0_regular40_thick',)

def prepare_stage1(baseline,output,stage=1):
    from anyfileio.sesam import read_sesam_fem_document,parse_sesam_fem_records,write_sesam_fem_document
    from anyfileio.sesam.records import FemRawRecord,canonical_record_lines
    if identity(baseline)['sha256']!=BASELINE_SHA:raise Unsupported('deep model baseline identity differs from adopted immutable source')
    base=read_sesam_fem_document(baseline,strict=True);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    names=STAGE1 if stage==1 else STAGE2 if stage==2 else NATIVE_STRIP
    suite={'schema_version':1,'stage':'deep_stage'+str(stage),'baseline':identity(baseline),'order':list(names),'cases':{}}
    material=next(r.numeric_fields for r in base.raw_records if r.name=='MISOSEL')
    beam_sections=[r for r in base.raw_records if r.name in {'GBEAMG','GBARM','TDSECT','GUNIVEC'}]
    thickness=base.sections[1].thickness
    for name in names:
        strip=name.startswith('strip_nu0');combined=name.startswith('combined');ny=40 if '40' in name else 20;nx=1 if strip else ny;width=.1 if strip else 2.;length=2.;offset=-.0500000007 if name.endswith('_ecc') else -(thickness+base.sections[2].web_height)/2 if name.endswith('_underplate') else 0.
        directory=root/name;directory.mkdir();records=[]
        def add(name,values,text=()):
            r=FemRawRecord(name,tuple(map(float,values)),tuple(text),0,0,());numeric=replace(r,text_fields=());records.append(replace(r,raw_lines=canonical_record_lines(numeric)+tuple(' '*8+x for x in text)))
        add('IDENT',(1,1,3,0));add('UNITS',(5,1,1,1,1));values=list(material);values[2]=0. if strip else values[2];add('MISOSEL',values);add('GELTH',(1,thickness))
        if combined:
            for r in beam_sections:add(r.name,r.numeric_fields,r.text_fields)
            if offset:add('GECCEN',(1,0,0,offset))
        nodes={}
        for j in range(ny+1):
            for i in range(nx+1):
                nid=j*(nx+1)+i+1;xyz=(float(np.float32(width*i/nx)),float(np.float32(length*j/ny)),0.)
                add('GNODE',(nid,nid,6,123456));add('GCOORD',(nid,*xyz));nodes[nid]=xyz
                if j==0:add('BNBCD',(nid,6,1,1,1,1,1,1))
                elif strip:add('BNBCD',(nid,6,1,0,0,0,0,0))
        beam_ids=[];shell_ids=[];eid=0
        if combined:
            for j in range(ny):
                eid+=1;a=j*(nx+1)+nx//2+1;b=(j+1)*(nx+1)+nx//2+1
                add('GELMNT1',(eid,eid,15,0,a,b));add('GELREF1',(eid,1,0,0,0,0,0,0,2,0,1 if offset else 0,1));beam_ids.append(eid)
        pressures=[]
        for j in range(ny):
            for i in range(nx):
                eid+=1;a=j*(nx+1)+i+1;connectivity=(a,a+1,a+nx+2,a+nx+1)
                add('GELMNT1',(eid,eid,24,0,*connectivity));add('GELREF1',(eid,1,0,0,0,0,0,0,1,0,0,0));shell_ids.append(eid);pressures.append(eid)
        add('TDLOAD',(4,1,103,0),('LC1',))
        for eid in pressures:add('BEUSLO',(1,1,0,0,eid,4,0,2,-1000,-1000,-1000,-1000))
        add('IEND',(0,0,0,0));doc=parse_sesam_fem_records(records,strict=True);fem=directory/(name+'_T1.FEM');write_sesam_fem_document(doc,fem,mode='raw')
        (directory/'sestra_control.inp').write_bytes(CONTROL.replace('\n','\r\n').encode('ascii'))
        manifest={'schema_version':1,'profile':'deep_stage1','case_id':name,'fem':fem.name,'fem_sha256':identity(fem)['sha256'],'reference_fem_sha256':identity(fem)['sha256'],'reference_sif':name+'_R1.SIF','load_case':1,'units':'SI','frame':'GLOBAL','node_ids':sorted(nodes),'shell_ids':shell_ids,'beam_ids':beam_ids,'translation_tolerances':{'absolute_m':1e-8,'relative':.001},'beam_tolerances':{'absolute_n':.001,'absolute_nm':.001,'relative':.001},'model_contract':{'width_m':width,'length_m':length,'thickness_m':thickness,'nx':nx,'ny':ny,'poisson_ratio':values[2],'global_beam_offsets_m':offset,'pressure_pa':-1000,'root':'all six DOFs fixed y=0','strip_inplane_control':'UX=0 all nodes' if strip else None,'coordinates':'explicit float32 representable regular grid; no original source coordinates altered'},'question':'mesh perturbation' if name=='plate_regular20' else 'independent cylindrical strip limit' if strip else 'attachment eccentricity, matched regular geometry'}
        manifest['profile']='deep_stage'+str(stage)
        thick=name.endswith('_thick') or combined;control=directory/'sestra_control.inp'
        if thick:control.write_text(CONTROL.replace('ELOP                   1.   0.   1.','ELOP                   1.   1.   1.'),encoding='ascii')
        manifest['model_contract']['sestra_thck']=int(thick);manifest['control_template']=identity(control)
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');suite['cases'][name]={'manifest':str(directory/'manifest.json'),'fem':identity(fem),'nodes':len(nodes),'shells':len(shell_ids),'beams':len(beam_ids),'control_template':identity(control)}
    (root/'suite.json').write_text(json.dumps(suite,indent=2),encoding='utf-8');return suite

def prepare_stage2(baseline,output):return prepare_stage1(baseline,output,stage=2)

def assert_deep_reference_identity(source,result,m):
    a,b=source.eccentricities,result.eccentricities
    if set(a)!=set(b):raise Unsupported('offset ID coverage differs')
    records={}
    for key in a:
        x,y=np.asarray(a[key],dtype=float),np.asarray(b[key],dtype=float)
        if x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any((x==0)!=(y==0)) or np.any(np.signbit(x)!=np.signbit(y)) or not np.all(np.isclose(x,y,rtol=np.finfo(np.float32).eps,atol=0)):raise Unsupported('offset identity differs beyond float32 serialization bound')
        records[key]={'source_m':x.tolist(),'result_m':y.tolist(),'difference_m':(y-x).tolist(),'bound_m':(np.finfo(np.float32).eps*np.abs(x)).tolist()}
    assert_reference_identity(source,replace(result,eccentricities=a),m)
    return {'policy':'deep-only offset identity serialization rtol=float32epsilon, atol0, exact zeros/sign/IDs; native source arms unchanged','vectors':records}

def load_case_manifest(path):
    path=Path(path).resolve();m=json.loads(path.read_text(encoding='utf-8'))
    if m.get('profile') not in {'deep_stage1','deep_stage2','deep_stage3'} or m.get('case_id') not in STAGE1+STAGE2+NATIVE_STRIP or m.get('load_case')!=1 or m.get('units')!='SI' or m.get('frame')!='GLOBAL' or m.get('translation_tolerances')!={'absolute_m':1e-8,'relative':.001} or m.get('beam_tolerances')!={'absolute_n':.001,'absolute_nm':.001,'relative':.001}:raise Unsupported('unadmitted deep model contract/criteria')
    for key in ('fem','reference_sif'):m[key]=str((path.parent/m[key]).resolve())
    if identity(m['fem'])['sha256']!=m['fem_sha256'] or m['reference_fem_sha256']!=m['fem_sha256']:raise Unsupported('deep FEM source hash differs')
    if identity(m['control_template']['path'])['sha256']!=m['control_template']['sha256']:raise Unsupported('deep control source hash differs')
    for key in ('node_ids','shell_ids','beam_ids'):
        ids=m[key]
        if len(set(ids))!=len(ids) or any(type(v) is not int or v<=0 for v in ids):raise Unsupported('invalid diagnostic ID coverage')
    m['manifest_identity']=identity(path);return m

def solve_native(m):
    from .sestra_three_case import solve_case
    contract=dict(m);contract['case_id']='combined' if m['beam_ids'] else 'plate'
    imported,u,translations,_,balance,info=solve_case(contract)
    analytical=None
    if m['case_id'].startswith('strip_nu0'):
        mat=imported.model.get_material(next(iter(imported.model.mesh.elements.values())).material_name);h=m['model_contract']['thickness_m'];L=2.;p=1000.;D=mat.elastic_modulus*h**3/12;G=mat.elastic_modulus/2;k=5/6
        bending=-p*L**4/(8*D);shear=-p*L**2/(2*k*G*h);tips=[n for n,v in imported.document.nodes.items() if v.coordinates[1]==L];tip=float(np.mean([translations[n][2] for n in tips]))
        analytical={'formula':'-p L^4/(8D)-p L^2/(2 k G h), D=E h^3/12 for nu0, k=5/6','bending_tip_m':bending,'shear_tip_m':shear,'total_tip_m':bending+shear,'native_tip_m':tip,'relative_error':abs(tip-(bending+shear))/abs(bending+shear),'assumptions':'cylindrical uniform-pressure cantilever; nu0; shear-deformable continuum oracle, distinct from thin-plate result','effective_nu':mat.poisson_ratio}
    return imported,u,translations,balance,info,analytical

def beam_same_field_probe(imported,native_u,sif_path,load_case=1):
    from anyfileio.sesam import read_sesam_sif_displacements,read_sesam_sif_beam_forces
    model=imported.model;nodes=read_sesam_sif_displacements(sif_path,load_case,sorted(model.mesh.nodes));forces=read_sesam_sif_beam_forces(sif_path,load_case)
    expected={eid for eid,e in imported.document.elements.items() if e.type_code==15}
    if set(forces.element_forces)!=expected:raise Unsupported('same-field beam coverage differs')
    if nodes.external_result_case!=load_case or any(v['external_result_case']!=load_case for v in forces.frame_provenance.values()):raise Unsupported('wrong external case in same-field probe')
    reference_u=np.zeros_like(native_u)
    for nid,vector in nodes.displacements.items():reference_u[model.mesh.dof_manager.get_node_dofs(nid)]=vector
    def serialization_bound(values):
        a=np.asarray(values,dtype=float);exponent=np.floor(np.log10(np.maximum(np.abs(a),np.finfo(float).tiny)));text=.5*np.power(10.,exponent-8);text=np.where(a==0,0.,text)
        return .5*np.abs(np.spacing(a.astype(np.float32)).astype(float))+text,text
    bounds,_=serialization_bound(reference_u);float32_ok=0
    for vec in nodes.displacements.values():
        a=np.asarray(vec);_,text=serialization_bound(a);float32_ok+=int(np.count_nonzero(np.abs(a-a.astype(np.float32).astype(float))<=text+np.finfo(float).eps*np.abs(a)))
    from anyfileio.sesam.records import read_raw_records
    import re
    nodal_records=[r for r in read_raw_records(sif_path) if r.name=='RVNODDIS' and r.numeric_fields and r.numeric_fields[0]>0]
    printed=[token for r in nodal_records for line in r.raw_lines for token in line[8:].split() if 'E' in token.upper()]
    nine_digit=bool(printed) and all(re.fullmatch(r'[+-]?\d\.\d{8}[Ee][+-]\d+',v) for v in printed)
    vectors={};counts={'native_field_vs_sestra':0,'sestra_field_native_recovery_vs_sestra':0};max_errors={k:np.zeros(6) for k in counts}
    for eid,reported in forces.element_forces.items():
        e=model.mesh.elements[eid];mat=model.get_material(e.material_name);stations=forces.station_coordinates[eid];ref=np.asarray(reported)
        native=e.recover_linear_section_forces(model.mesh,native_u,mat,stations);same=e.recover_linear_section_forces(model.mesh,reference_u,mat,stations);limit=.001+.001*np.abs(ref)
        dofs=np.concatenate([model.mesh.dof_manager.get_node_dofs(n) for n in e.node_ids]);operator=[]
        for dof in dofs:
            basis=np.zeros_like(native_u);basis[dof]=1.;operator.append(e.recover_linear_section_forces(model.mesh,basis,mat,stations))
        propagated=np.sum(np.abs(np.stack(operator,axis=-1))*bounds[dofs],axis=-1);reported_bound,_=serialization_bound(ref);uncertainty=propagated+reported_bound
        for key,values in (('native_field_vs_sestra',native),('sestra_field_native_recovery_vs_sestra',same)):
            counts[key]+=int(np.count_nonzero(np.abs(values-ref)>limit));max_errors[key]=np.maximum(max_errors[key],np.max(np.abs(values-ref),axis=0))
        vectors[eid]={'native_field_native_recovery':native.tolist(),'sestra_field_native_recovery':same.tolist(),'sestra_reported':ref.tolist(),'native_minus_same_field':(native-same).tolist(),'same_field_minus_reported':(same-ref).tolist(),'stations_global_m':stations,'assumed_serialization_error_upper_bound':uncertainty.tolist(),'outside_assumed_serialization_bound_components':int(np.count_nonzero(np.abs(same-ref)>uncertainty))}
    return {'observable':'physical GLOBAL positive-local-x cut forces and moments','units':['N']*3+['Nm']*3,'criteria':{'absolute_n_nm':.001,'relative':.001},'compared_station_components':sum(np.asarray(v['sestra_reported']).size for v in vectors.values()),'failure_counts':counts,'maximum_absolute_errors_per_component':{k:v.tolist() for k,v in max_errors.items()},'elements':vectors,'precision_diagnostic':{'assumption':'independent nearest float32 rounding plus 9-significant-digit formatted text; |linear recovery operator| times nodal bounds plus reported-force serialization bound; excludes stiffness-serialization differences and floating point arithmetic','nine_significant_digit_raw_tokens_confirmed':nine_digit,'nodal_components_consistent_with_float32_plus_text':float32_ok,'nodal_components':6*len(nodes.displacements),'outside_assumed_bound_components':sum(v['outside_assumed_serialization_bound_components'] for v in vectors.values()),'fixed_criteria_unchanged':True},'interpretation':'same-field error tests kinematics/stiffness/recovery correspondence; native-minus-same-field isolates nodal-solution contribution; neither establishes plate-formulation cause alone','nodal_source':identity(sif_path)}

def run_diagnostics(input_path,output,mode='compare'):
    import os,time
    from threadpoolctl import threadpool_info
    from anyfileio.sesam import read_sesam_fem_document,read_sesam_sif_translations,sif,document,semantics
    from . import assembly,elements,e4_pl_element
    from .sestra_three_case import read_manifest,solve_case
    from .sesam_fem import importer
    started=time.perf_counter();root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);native_cache={}
    suite=json.loads(Path(input_path).read_text(encoding='utf-8'));results={}
    order=suite['order']
    for name in order:
        if mode=='probe' and name=='plate':continue
        entry=suite['cases'][name];mp=entry['manifest'] if isinstance(entry,dict) else entry
        case={'case_id':name,'qualification':False}
        try:
            m=read_manifest(mp) if mode=='probe' else load_case_manifest(mp)
            source=read_sesam_fem_document(m['fem'],strict=True);reference=read_sesam_fem_document(m['reference_sif'],strict=True)
            offset_identity=assert_deep_reference_identity(source,reference,m) if mode!='probe' else None
            if mode=='probe':assert_reference_identity(source,reference,m)
            cache_key=identity(m['fem'])['sha256']
            if mode=='probe':
                imported,u,translation,oracle,balance,info=solve_case(m);analytical=oracle
            elif cache_key in native_cache:imported,u,translation,balance,info,analytical=native_cache[cache_key]
            else:
                imported,u,translation,balance,info,analytical=solve_native(m);native_cache[cache_key]=(imported,u,translation,balance,info,analytical)
            ref=read_sesam_sif_translations(m['reference_sif'],m['load_case'],m['node_ids'])
            if ref.external_result_case!=m['load_case']:raise Unsupported('external nodal case differs')
            rows=[];failures=0;maximum=0.
            for nid in m['node_ids']:
                a=np.asarray(translation[nid]);b=np.asarray(ref.translations[nid]);error=np.abs(a-b);limit=1e-8+.001*np.abs(b);bad=error>limit;failures+=int(bad.sum());maximum=max(maximum,float(error.max()));rows.append({'node':nid,'native_m':a.tolist(),'reference_m':b.tolist(),'absolute_error_m':error.tolist(),'failed':bad.tolist()})
            probe=beam_same_field_probe(imported,u,m['reference_sif']) if m['beam_ids'] else None
            additional={}
            if mode=='probe' and m['beam_ids']:
                for label,path in suite.get('additional_references',{}).get(name,{}).items():
                    alternate=read_sesam_fem_document(path,strict=True);assert_reference_identity(source,alternate,m)
                    additional[label]=beam_same_field_probe(imported,u,path,m['load_case'])
            tipids=[nid for nid in m['node_ids'] if source.nodes[nid].coordinates[1]==max(v.coordinates[1] for v in source.nodes.values())]
            case.update(status='failed' if failures or not balance['passed'] or (probe and probe['failure_counts']['native_field_vs_sestra']) else 'passed',fem=identity(m['fem']),reference=identity(m['reference_sif']),model_contract=m.get('model_contract'),translation_criteria={'absolute_m':1e-8,'relative':.001},translation_components=len(rows)*3,translation_failures=failures,maximum_translation_absolute_error_m=maximum,translations=rows,balance=balance,analytical=analytical,beam_same_field_probe=probe,native_peak_uz_m=min(v[2] for v in translation.values()),reference_peak_uz_m=min(v[2] for v in ref.translations.values()),native_tip_mean_uz_m=float(np.mean([translation[n][2] for n in tipids])),reference_tip_mean_uz_m=float(np.mean([ref.translations[n][2] for n in tipids])))
            case['additional_same_field_probes']=additional
            case['offset_identity_normalization']=offset_identity
        except (Unsupported,ValueError,FileNotFoundError) as exc:case.update(status='unsupported',reason=str(exc))
        (root/(name+'.json')).write_text(json.dumps(case,indent=2),encoding='utf-8');results[name]={k:v for k,v in case.items() if k not in ('translations','beam_same_field_probe','additional_same_field_probes')}
        if case.get('beam_same_field_probe'):results[name]['probe_failure_counts']=case['beam_same_field_probe']['failure_counts']
    report={'status':'passed' if results and all(x['status']=='passed' for x in results.values()) else 'failed','mode':mode,'qualification':False,'cases':results,'runtime':{'NUMBA_DISABLE_JIT':os.environ.get('NUMBA_DISABLE_JIT'),'pools':threadpool_info(),'loaded_sources':{mod.__name__:identity(mod.__file__) for mod in (sif,document,semantics,assembly,elements,e4_pl_element,importer)}},'process_body_wall_seconds':time.perf_counter()-started,'interpretation':'Fixed development criteria retained. Diagnostic model changes identify mechanisms; they do not retroactively pass the original model.'}
    report['runtime']['loaded_sources'][__name__]=identity(__file__)
    (root/'diagnostics_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report
