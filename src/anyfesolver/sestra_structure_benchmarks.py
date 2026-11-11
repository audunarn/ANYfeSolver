"""Reproducible development structure benchmarks; no external executables."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from .sestra_verification import Unsupported,identity
from .sestra_three_case import CONTROL
from .sestra_deep_diagnostics import assert_deep_reference_identity,beam_same_field_probe

FAMILIES=('inclined_frame','crossing_stiffeners','folded_plate')
LEVELS=(4,8,16)
CRITERIA={'translation_absolute_m':1e-8,'relative':.001,'force_absolute_n':.001,'moment_absolute_nm':.001}
def f32(x):return float(np.float32(x))

def prepare_suite(output):
    from anyfileio.sesam.records import FemRawRecord,records_to_text,canonical_record_lines
    from anyfileio.sesam import read_sesam_fem_document,parse_sesam_fem_records,write_sesam_fem_document
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);suite={'schema_version':1,'profile':'structure_benchmarks','order':[],'cases':{}}
    for family in FAMILIES:
        for n in LEVELS:
            name=f'{family}_{n}';directory=root/name;directory.mkdir();records=[];nodes={};lookup={};shells=[];beams=[];supports=[];loads={};pressures={};eid=0
            def add(tag,v):records.append(FemRawRecord(tag,tuple(map(float,v)),(),0,0,()))
            add('IDENT',(1,1,3,0));add('UNITS',(5,1,1,1,1));add('MISOSEL',(1,210000003000,.300000012,7850,0,.000012,1,355));add('GELTH',(1,.0199999996))
            add('GBEAMG',(2,0,.00100000005,3.12333519e-8,8.33333331e-7,8.33333313e-9,0,3.12530210e-6,1.66666669e-5,1.66666666e-6,.00066666666,.00066666666,0,0,1.24999997e-5,1.24999997e-6));add('GBARM',(2,.100000001,.00999999978,.00999999978,1,1))
            add('GUNIVEC',(1,0,0,1));add('GUNIVEC',(2,f32(-.8),0,f32(.6)))
            offset=f32(-(.0199999996+.100000001)/2)
            if family=='crossing_stiffeners':add('GECCEN',(1,0,0,offset))
            def node(xyz):
                xyz=tuple(f32(x) for x in xyz)
                if xyz not in lookup:
                    key=len(nodes)+1;lookup[xyz]=key;nodes[key]=xyz;add('GNODE',(key,key,6,123456));add('GCOORD',(key,*xyz))
                return lookup[xyz]
            def beam(a,b,axis=1,off=False):
                nonlocal eid
                eid+=1;add('GELMNT1',(eid,eid,15,0,a,b));add('GELREF1',(eid,1,0,0,0,0,0,0,2,0,int(off),axis));beams.append(eid)
            def shell(ids,pressure=False):
                nonlocal eid
                eid+=1;add('GELMNT1',(eid,eid,24,0,*ids));add('GELREF1',(eid,1,0,0,0,0,0,0,1,0,0,0));shells.append(eid)
                if pressure:pressures[eid]=-1000.
            if family=='inclined_frame':
                a=np.array([0.,0.,0.]);joint=np.array([1.2,0.,1.6]);tip=np.array([1.2,2.,1.6]);chains=[]
                for start,end,axis in ((a,joint,2),(joint,tip,1)):
                    chain=[node(start+(end-start)*j/n) for j in range(n+1)];chains.append(chain)
                    for x,y in zip(chain,chain[1:]):beam(x,y,axis)
                supports=[chains[0][0]];loads[chains[1][-1]]=(.2,-.3,-1.,0.,.1,0.)
                model_contract={'members':[[nodes[chains[0][0]],nodes[chains[0][-1]]],[nodes[chains[1][0]],nodes[chains[1][-1]]]],'tip_node':chains[1][-1],'tip_wrench':list(loads[chains[1][-1]]),'assumptions':'rigid shared sixDOF joint; warping-free Saint-Venant torsion and Timoshenko shear'}
            elif family=='crossing_stiffeners':
                grid=[[node((2*i/n,2*j/n,0)) for i in range(n+1)] for j in range(n+1)]
                for j in range(n):beam(grid[j][n//2],grid[j+1][n//2],off=True)
                for i in range(n):beam(grid[n//2][i],grid[n//2][i+1],off=True)
                for j in range(n):
                    for i in range(n):shell((grid[j][i],grid[j][i+1],grid[j+1][i+1],grid[j+1][i]),True)
                supports=grid[0];model_contract={'global_offsets_m':[0,0,offset],'crossing_node':grid[n//2][n//2],'attachment':'bar top at plate underside; shared system node at crossing','nominal_pressure_pa':-1000}
            else:
                across=n//2;faces=[]
                for face in (0,1):
                    grid=[[node((i/across if face==0 else 0,2*j/n,0 if face==0 else i/across)) for i in range(across+1)] for j in range(n+1)];faces.append(grid)
                    for j in range(n):
                        for i in range(across):
                            a,b,c,d=grid[j][i],grid[j][i+1],grid[j+1][i+1],grid[j+1][i]
                            shell((a,b,c,d) if face==0 else (a,d,c,b))
                supports=sorted(set(faces[0][0]+faces[1][0]));loads[faces[0][-1][-1]]=(0,0,-500,0,0,0);loads[faces[1][-1][-1]]=(0,0,-500,0,0,0)
                model_contract={'faces':'XY and YZ, outward normals+Z/+X','fold':'shared global sixDOF nodes along y-axis','tip_load':'two global FZ=-500 N loads at outer tip corners','tip_corner_coordinates_m':[nodes[faces[0][-1][-1]],nodes[faces[1][-1][-1]]],'sampling_limit':'concentrated-load stress singularities; global displacement/energy are primary, no corner-stress convergence claim'}
            for key in supports:add('BNBCD',(key,6,1,1,1,1,1,1))
            add('TDLOAD',(4,1,103,0))
            records[-1]=replace(records[-1],text_fields=('LC1',),raw_lines=canonical_record_lines(records[-1])+('        LC1',))
            for key,v in loads.items():add('BNLOAD',(1,1,0,0,key,6,*v))
            for key,p in pressures.items():add('BEUSLO',(1,1,0,0,key,4,0,2,p,p,p,p))
            add('IEND',(0,0,0,0));records=[r if r.raw_lines else replace(r,raw_lines=canonical_record_lines(r)) for r in records];fem=directory/(name+'_T1.FEM');write_sesam_fem_document(parse_sesam_fem_records(tuple(records),strict=True),fem,mode='raw');doc=read_sesam_fem_document(fem,strict=True)
            control=directory/'sestra_control.inp';control.write_bytes(CONTROL.replace('ELOP                   1.   0.   1.','ELOP                   1.   1.   1.').replace('\n','\r\n').encode('ascii'))
            m={'schema_version':1,'profile':'structure_benchmarks','case_id':name,'family':family,'level':n,'units':'SI','frame':'GLOBAL','load_case':1,'fem':fem.name,'reference_sif':name+'_R1.SIF','fem_sha256':identity(fem)['sha256'],'reference_fem_sha256':identity(fem)['sha256'],'control_template':identity(control),'criteria':CRITERIA,'node_ids':sorted(doc.nodes),'shell_ids':shells,'beam_ids':beams,'support_nodes':supports,'nodal_loads':loads,'pressure_loads':pressures,'model_contract':model_contract}
            manifest=directory/'manifest.json';manifest.write_text(json.dumps(m,indent=2));suite['order'].append(name);suite['cases'][name]={'manifest':str(manifest),'fem':identity(fem),'control':identity(control),'nodes':len(nodes),'shells':len(shells),'beams':len(beams)}
    (root/'suite.json').write_text(json.dumps(suite,indent=2));return suite

def read_manifest(path):
    p=Path(path).resolve();m=json.loads(p.read_text())
    if m.get('profile')!='structure_benchmarks' or m.get('family') not in FAMILIES or m.get('level') not in LEVELS or m.get('criteria')!=CRITERIA or m.get('load_case')!=1 or m.get('units')!='SI' or m.get('frame')!='GLOBAL':raise Unsupported('unadmitted structure benchmark contract')
    for key in ('fem','reference_sif'):m[key]=str((p.parent/m[key]).resolve())
    if identity(m['fem'])['sha256']!=m['fem_sha256'] or m['reference_fem_sha256']!=m['fem_sha256'] or identity(m['control_template']['path'])['sha256']!=m['control_template']['sha256']:raise Unsupported('structure source/control binding differs')
    for key in ('node_ids','shell_ids','beam_ids'):
        if len(set(m[key]))!=len(m[key]) or any(type(x)is not int or x<=0 for x in m[key]):raise Unsupported('invalid structure requested IDs')
    m['manifest_identity']=identity(p);return m

def admit(m):
    from .sesam_fem import import_sesam_fem
    imported=import_sesam_fem(m['fem'],strict=True);d=imported.document;model=imported.model
    allowed={'IDENT','UNITS','MISOSEL','GELTH','GBEAMG','GBARM','GUNIVEC','GECCEN','GNODE','GCOORD','GELMNT1','GELREF1','BNBCD','TDLOAD','BNLOAD','BEUSLO','IEND'}
    if set(d.record_counts)-allowed or d.dependencies or d.coordinate_transforms or d.header.unit_values!=(5.,1.,1.,1.,1.):raise Unsupported('unadmitted structure record/units/transform semantics')
    if set(d.nodes)!=set(m['node_ids']) or set(d.elements)!=set(m['shell_ids'])|set(m['beam_ids']) or set(model.mesh.elements)!=set(d.elements) or len(model.load_cases)!=1:raise Unsupported('structure source/native coverage differs')
    for diagnostic in imported.diagnostics:
        if not(diagnostic.code=='FEM120' and diagnostic.record_name in {'BEUSLO','BNLOAD'} and diagnostic.message==diagnostic.record_name+' is preserved but not translated into solver semantics yet'):raise Unsupported('structure importer diagnostic: '+str(diagnostic))
    expected_nodal={int(k):tuple(v) for k,v in m['nodal_loads'].items()};expected_pressure={int(k):v for k,v in m['pressure_loads'].items()}
    if set(model.load_cases[0].nodal_loads)!=set(expected_nodal) or any(not np.array_equal(model.load_cases[0].nodal_loads[k],v) for k,v in expected_nodal.items()) or model.load_cases[0].pressure_loads!=expected_pressure:raise Unsupported('effective native load differs')
    for r in d.load_records:
        v=r.raw_values
        if not np.all(np.isfinite(v)):raise Unsupported('nonfinite structure load')
        if r.record_name=='TDLOAD':
            if v!=(4.,1.,103.,0.):raise Unsupported('structure source case metadata differs')
        elif r.record_name=='BNLOAD':
            if len(v)!=12 or v[:4]!=(1.,1.,0.,0.) or v[5]!=6 or d.node_internal_to_external.get(int(v[4])) not in expected_nodal or tuple(v[6:])!=expected_nodal[d.node_internal_to_external[int(v[4])]]:raise Unsupported('unsupported real global nodal load')
        elif r.record_name=='BEUSLO':
            if len(v)!=12 or v[:4]!=(1.,1.,0.,0.) or v[5:8]!=(4.,0.,2.) or any(x!=v[8] for x in v[8:]) or d.element_internal_to_external.get(int(v[4])) not in expected_pressure:raise Unsupported('unsupported constant planar pressure')
        else:raise Unsupported('unadmitted structure load')
    if {b.node_id for b in d.boundaries}!=set(m['support_nodes']) or any(b.dof_flags!=(1,1,1,1,1,1) or any(b.prescribed_values) for b in d.boundaries):raise Unsupported('structure supports differ')
    audit={};normals={}
    for eid,e in d.elements.items():
        xyz=np.asarray([d.nodes[n].coordinates for n in e.node_ids]);ref=d.element_references[eid];mat=d.materials[ref.material_id];native=model.mesh.elements[eid];effective=model.get_material(native.material_name)
        if not np.all(np.isfinite(xyz)) or mat.raw_values[1:4]!=(effective.elastic_modulus,effective.poisson_ratio,effective.density) or any(ref.raw_values[2:8]) or ref.raw_values[9]:raise Unsupported('source/native material or release semantics differ')
        if eid in m['shell_ids']:
            normal=np.cross(xyz[1]-xyz[0],xyz[3]-xyz[0]);area=np.linalg.norm(normal)
            if e.type_code!=24 or area<=0 or ref.transform_id or ref.raw_values[10] or abs(np.dot(xyz[2]-xyz[0],normal/area))>1e-7 or native.thickness!=d.sections[ref.section_id].thickness:raise Unsupported('only planar nondegenerate untransformed Q4 structure shells')
            normal/=area;edges=np.roll(xyz,-1,axis=0)-xyz
            if any(np.dot(np.cross(edges[i],edges[(i+1)%4]),normal)<=0 for i in range(4)):raise Unsupported('nonconvex structure Q4')
            normals[eid]=normal.tolist()
        else:
            if e.type_code!=15 or ref.transform_id not in d.unit_vectors or ref.nodal_transform_ids:raise Unsupported('explicit BEAS15 localz required')
            x=xyz[1]-xyz[0];x/=np.linalg.norm(x);z=np.asarray(d.unit_vectors[ref.transform_id].vector);z=z-x*np.dot(x,z)
            if not np.all(np.isfinite(z)) or np.linalg.norm(z)<1e-12:raise Unsupported('degenerate beam axis')
            z/=np.linalg.norm(z);y=np.cross(z,x);R=np.column_stack((x,y,z))
            if not np.allclose(R.T@R,np.eye(3),atol=1e-12) or np.linalg.det(R)<0:raise Unsupported('beam frame not right handed')
            for key in ('area','Iy','Iz','J','shear_factor_y','shear_factor_z'):
                if not np.isfinite(native.cross_section.get(key,float('nan'))) or native.cross_section[key]<=0:raise Unsupported('missing effective beam section')
            sec=d.sections[ref.section_id];expected={'area':sec.area,'Iy':sec.iy,'Iz':sec.iz,'J':sec.torsion,'shear_factor_y':sec.raw_values[10]/sec.area,'shear_factor_z':sec.raw_values[11]/sec.area}
            if any(native.cross_section[k]!=v for k,v in expected.items()):raise Unsupported('effective beam section differs from source')
            nz=np.asarray(native.cross_section['orientation']);nz=nz-x*np.dot(x,nz);nz/=np.linalg.norm(nz)
            if not np.allclose(nz,z,rtol=0,atol=1e-12):raise Unsupported('effective beam orientation differs')
            centroid=native.get_node_coordinates(model.mesh);offsets=np.asarray([d.eccentricities[int(ref.raw_values[10])]]*2) if ref.raw_values[10] else np.zeros((2,3))
            if not np.array_equal(centroid,xyz+offsets):raise Unsupported('effective beam offset differs')
            audit[eid]={'frame_columns':R.tolist(),'section':native.cross_section,'system_nodes_global_m':xyz.tolist()}
    return imported,{'beam_frames':audit,'shell_normals':normals,'nodes':len(d.nodes),'shells':len(normals),'beams':len(audit),'physical_contract':m['model_contract'],'source':identity(m['fem']),'element_classes':sorted({type(e).__module__+'.'+type(e).__qualname__ for e in model.mesh.elements.values()})}

def frame_oracle(imported,m):
    """Independent terminal-wrench compliance by exact energy quadrature."""
    model=imported.model;d=imported.document;tip=np.asarray(d.nodes[m['model_contract']['tip_node']].coordinates);C=np.zeros((6,6));mat=d.materials[1];E=mat.elastic_modulus;G=E/(2*(1+mat.poisson_ratio));sec=d.sections[2]
    D=np.diag([1/(E*sec.area),1/(G*sec.raw_values[10]),1/(G*sec.raw_values[11]),1/(G*sec.torsion),1/(E*sec.iy),1/(E*sec.iz)])
    for start,end in m['model_contract']['members']:
        a,b=np.asarray(start),np.asarray(end);delta=b-a;L=np.linalg.norm(delta);x=delta/L;z=np.array([-.8,0,.6]) if abs(x[1])<.5 else np.array([0.,0.,1.]);z-=x*np.dot(x,z);z/=np.linalg.norm(z);R=np.column_stack((x,np.cross(z,x),z))
        for t in ((1-1/np.sqrt(3))/2,(1+1/np.sqrt(3))/2):
            r=tip-(a+t*delta);B=np.zeros((6,6));B[:3,:3]=R.T
            skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]]);B[3:,:3]=R.T@skew;B[3:,3:]=R.T;C+=L/2*(B.T@D@B)
    w=np.asarray(m['model_contract']['tip_wrench']);return {'tip_global_sixdof':(C@w).tolist(),'units':['m']*3+['rad']*3,'strain_energy_j':float(w@C@w/2),'compliance':C.tolist(),'method':'Castigliano; terminal wrench and lever arms only; exact two-point integral of member quadratic energy; no native stiffness/recovery reuse'}

def assert_structure_reference_identity(source,result,m):
    """Audit result load serialization only, then delegate strict model identity."""
    def physical(records):
        table={}
        for r in records:
            if r.record_name=='TDLOAD':continue
            cut=6 if r.record_name=='BNLOAD' else 8 if r.record_name=='BEUSLO' else None
            if cut is None or len(r.raw_values)!=12:raise Unsupported('unsupported structural reference load layout')
            key=(r.record_name,r.load_case_id,r.target_id,tuple(r.raw_values[:cut]))
            if key in table:raise Unsupported('duplicate structural reference load identity')
            table[key]=(r,cut)
        return table
    a,b=physical(source.load_records),physical(result.load_records)
    if set(a)!=set(b):raise Unsupported('structural reference load metadata/IDs/case differ')
    normalization=[];replacements={}
    for key,(record,cut) in a.items():
        reference=b[key][0];x=np.asarray(record.raw_values[cut:]);y=np.asarray(reference.raw_values[cut:]);bound=np.finfo(np.float32).eps*np.abs(x)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any((x==0)!=(y==0)) or np.any(np.signbit(x)!=np.signbit(y)) or np.any(np.abs(y-x)>bound):raise Unsupported('structural reference load payload differs beyond float32 serialization')
        normalization.append({'record':record.record_name,'metadata':list(record.raw_values[:cut]),'source_payload':x.tolist(),'result_payload':y.tolist(),'signed_delta':(y-x).tolist(),'bound':bound.tolist(),'units':'N/Nm for BNLOAD; Pa for BEUSLO'})
        replacements[id(reference)]=replace(reference,raw_values=record.raw_values)
    audited=replace(result,load_records=tuple(replacements.get(id(r),r) for r in result.load_records))
    identity_report=assert_deep_reference_identity(source,audited,m)
    identity_report['load_serialization']={'policy':'structure-only rtol=float32epsilon, atol0; exact metadata/IDs/case/zeros/sign; source/native loads unchanged','records':normalization}
    return identity_report

def model_visual(imported,m,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    d=imported.document;fig=plt.figure(figsize=(7,5));ax=fig.add_subplot(111,projection='3d')
    polys=[np.asarray([d.nodes[n].coordinates for n in d.elements[e].node_ids]) for e in m['shell_ids']]
    if polys:ax.add_collection3d(Poly3DCollection(polys,facecolors='#a2bbd8',edgecolors='#687b91',linewidths=.4,alpha=.35))
    physical=[v.coordinates for v in d.nodes.values()]
    for e in m['beam_ids']:
        xyz=imported.model.mesh.elements[e].get_node_coordinates(imported.model.mesh);physical.extend(xyz.tolist());ax.plot(*xyz.T,color='#b63931',lw=2)
        system=np.asarray([d.nodes[n].coordinates for n in d.elements[e].node_ids])
        for a,b in zip(system,xyz):
            if not np.array_equal(a,b):ax.plot(*np.stack((a,b)).T,color='#999999',ls='--',lw=.6)
    xyz=np.asarray(physical);ax.auto_scale_xyz(xyz[:,0],xyz[:,1],xyz[:,2]);root=np.asarray([d.nodes[n].coordinates for n in m['support_nodes']]);ax.scatter(*root.T,color='#153471',marker='s',s=12,label='six-DOF clamps')
    for n,v in m['nodal_loads'].items():
        r=np.array(d.nodes[int(n)].coordinates);f=np.asarray(v[:3]);norm=np.linalg.norm(f)
        if norm:ax.quiver(*r,*(f/norm*.4),color='#238544',arrow_length_ratio=.25)
    annotation='Uniform global FZ pressure -1000 Pa' if m['pressure_loads'] else 'Tip global F='+str(m['model_contract']['tip_wrench'][:3])+' N; M='+str(m['model_contract']['tip_wrench'][3:])+' Nm' if m['family']=='inclined_frame' else 'Two global FZ=-500 N at fixed outer tip corners'
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));ax.zaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=8)
    if m['family']=='crossing_stiffeners':ax.set_zticks([float(m['model_contract']['global_offsets_m'][2]),0.]);ax.set_zticklabels(['centroid -0.06','plate 0'])
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=m['case_id']+'\nbeams: centroid lines; dashed lines: rigid arms');ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),.4));ax.legend(loc='upper left');fig.text(.03,.03,annotation+'; arrow length is illustrative',fontsize=8);fig.tight_layout(rect=(0,.20,1,1));fig.savefig(path,dpi=150);fig.savefig(Path(path).with_suffix('.svg'));plt.close(fig)

def native_solution(imported,m):
    from .assembly import solve_linear,compute_reactions
    model=imported.model;u,info=solve_linear(model,model.load_cases[0],constraint_mode='transformation')
    if info.get('convergence_info',{}).get('status')!='converged' or not np.all(np.isfinite(u)):raise Unsupported('structure native solve failed/nonfinite')
    fields={n:u[model.mesh.dof_manager.get_node_dofs(n)].tolist() for n in m['node_ids']};load=np.zeros(6);reaction=np.zeros(6)
    for n,v in model.load_cases[0].nodal_loads.items():load[:3]+=v[:3];load[3:]+=v[3:]+np.cross(imported.document.nodes[n].coordinates,v[:3])
    for e,p in model.load_cases[0].pressure_loads.items():
        xy=np.asarray([imported.document.nodes[n].coordinates for n in imported.document.elements[e].node_ids]);nxt=np.roll(xy,-1,axis=0);cr=xy[:,0]*nxt[:,1]-nxt[:,0]*xy[:,1];area=cr.sum()/2;centroid=np.array([sum((xy[:,0]+nxt[:,0])*cr)/(6*area),sum((xy[:,1]+nxt[:,1])*cr)/(6*area),0]);f=np.array([0,0,p*area]);load[:3]+=f;load[3:]+=np.cross(centroid,f)
    for n,v in compute_reactions(model,u,model.load_cases[0]).items():reaction[:3]+=v[:3];reaction[3:]+=v[3:]+np.cross(imported.document.nodes[n].coordinates,v[:3])
    balance={'load_n_nm':load.tolist(),'reaction_n_nm':reaction.tolist(),'residual_n_nm':(load+reaction).tolist(),'passed':bool(np.all(np.abs(load+reaction)<=1e-6+1e-8*np.abs(load)))}
    oracle=frame_oracle(imported,m) if m['family']=='inclined_frame' else None
    if oracle:
        tip=fields[m['model_contract']['tip_node']];expected=np.asarray(oracle['tip_global_sixdof']);oracle.update(native_tip_global_sixdof=tip,absolute_error_per_component=np.abs(np.asarray(tip)-expected).tolist(),passed=bool(np.allclose(tip,expected,rtol=1e-6,atol=1e-10)))
    work=load_energy(imported,m,fields)
    return fields,u,balance,oracle,{'nodal_load_energy_j':work,'solver':info['solver_type'],'constraint_mode':info['constraint_mode'],'maximum_translation_norm_m':float(max(np.linalg.norm(v[:3]) for v in fields.values())),'maximum_rotation_norm_rad':float(max(np.linalg.norm(v[3:]) for v in fields.values())),'amplitude_audit':'reported separately; algebraic linear agreement does not establish finite-displacement validity'}

def load_energy(imported,m,fields):
    value=sum(np.dot(np.asarray(v),fields[int(n)]) for n,v in m['nodal_loads'].items())
    for e,p in m['pressure_loads'].items():
        element=imported.document.elements[int(e)];xyz=np.asarray([imported.document.nodes[n].coordinates for n in element.node_ids]);area=np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[3]-xyz[0]));normal=np.cross(xyz[1]-xyz[0],xyz[3]-xyz[0])/area
        value+=sum(p*area/4*np.dot(normal,fields[n][:3]) for n in element.node_ids)
    return float(value/2)

def probe_locations(imported,m,fields):
    points=([m['model_contract']['members'][0][1],m['model_contract']['members'][1][1]] if m['family']=='inclined_frame' else [[1,1,0],[1,2,0],[0,2,0],[2,2,0]] if m['family']=='crossing_stiffeners' else [[1,2,0],[0,2,1],[0,2,0],[0,1,0]])
    found=[]
    for p in points:
        ids=[n for n,v in imported.document.nodes.items() if np.allclose(v.coordinates,p,rtol=0,atol=1e-8)]
        if len(ids)!=1:raise Unsupported('predefined physical probe coverage differs')
        found.append({'global_m':p,'node':ids[0],'translations_m':fields[ids[0]][:3]})
    return found

def convergence_summary(directory):
    """Read saved cases only; no solves and no new acceptance criteria."""
    root=Path(directory);result={'qualification':False,'criterion':'diagnostic mesh trends at fixed physical probes; counts are not convergence measures','families':{}}
    for family in FAMILIES:
        rows=[]
        for level in LEVELS:
            p=root/(f'{family}_{level}.json')
            if not p.is_file():continue
            r=json.loads(p.read_text());row={'level':level,'status':r['status'],'report':str(p),'visual':str(p.with_suffix('.png')),'native_energy_j':r.get('native',{}).get('nodal_load_energy_j'),'reference_energy_j':r.get('reference_load_energy_j'),'max_absolute_xyz_m':r.get('maximum_translation_absolute_errors_xyz_m'),'native_probes':r.get('native_probes'),'reference_probes':r.get('reference_probes'),'translation_failures':r.get('translation_failures'),'translation_components':r.get('translation_components')}
            if r.get('beam_comparison'):row.update(beam_failures=r['beam_comparison']['failure_counts']['native_field_vs_sestra'],beam_components=r['beam_comparison']['compared_station_components'])
            if rows:
                for origin in ('native','reference'):
                    old={tuple(x['global_m']):x for x in rows[-1].get(origin+'_probes') or []};current=row.get(origin+'_probes') or [];row[origin+'_probe_mesh_changes_m']=[{'global_m':x['global_m'],'signed_translation_change_m':(np.asarray(x['translations_m'])-np.asarray(old[tuple(x['global_m'])]['translations_m'])).tolist()} for x in current if tuple(x['global_m']) in old]
                    a,b=row.get(origin+'_energy_j'),rows[-1].get(origin+'_energy_j');row[origin+'_energy_mesh_change_j']=None if a is None or b is None else a-b
            rows.append(row)
        result['families'][family]=rows
    return result

def run_suite(input_path,output,mode='compare'):
    import os,time
    from threadpoolctl import threadpool_info
    from anyfileio.sesam import read_sesam_fem_document,read_sesam_sif_displacements,sif,document,semantics
    from . import assembly,elements,e4_pl_element
    from .sesam_fem import importer
    suite=json.loads(Path(input_path).read_text());root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);summary={};started=time.perf_counter()
    for name in suite['order']:
        report={'case_id':name,'qualification':False}
        try:
            m=read_manifest(suite['cases'][name]['manifest']);imp,audit=admit(m);report.update(audit=audit,contract=m,input=identity(m['fem']),control=m['control_template']);model_visual(imp,m,root/(name+'.png'))
            fields,u,balance,oracle,native=native_solution(imp,m);report.update(global_nodal_sixdof=fields,balance=balance,analytical_oracle=oracle,native=native,status='not_executed',reason='native evaluated; external reference comparison not requested')
            report['native_probes']=probe_locations(imp,m,fields)
            if not balance['passed'] or (oracle is not None and not oracle['passed']):report.update(status='failed',reason='native balance or independent oracle failed')
            if mode=='compare':
                refdoc=read_sesam_fem_document(m['reference_sif'],strict=True);report['reference_identity']=assert_structure_reference_identity(imp.document,refdoc,m);ref=read_sesam_sif_displacements(m['reference_sif'],1,m['node_ids'])
                if ref.external_result_case!=1:raise Unsupported('wrong external structure case')
                rows=[];fail=0
                for n in m['node_ids']:
                    a=np.asarray(fields[n][:3]);b=np.asarray(ref.displacements[n][:3]);delta=a-b;bad=np.abs(delta)>1e-8+.001*np.abs(b);fail+=int(bad.sum());rows.append({'node':n,'native_m':a.tolist(),'reference_m':b.tolist(),'signed_error_m':delta.tolist(),'failed':bad.tolist()})
                report['reference_probes']=probe_locations(imp,m,ref.displacements);report['reference_load_energy_j']=load_energy(imp,m,ref.displacements);report['maximum_translation_absolute_errors_xyz_m']=np.max(np.abs([v['signed_error_m'] for v in rows]),axis=0).tolist()
                beam=beam_same_field_probe(imp,u,m['reference_sif']) if m['beam_ids'] else None;report.update(status='passed' if not fail and balance['passed'] and (oracle is None or oracle['passed']) and (beam is None or not beam['failure_counts']['native_field_vs_sestra']) else 'failed',reason=None,translations=rows,translation_failures=fail,translation_components=3*len(rows),beam_comparison=beam,reference=identity(m['reference_sif']))
        except (Unsupported,ValueError,FileNotFoundError) as exc:report.update(status='unsupported',reason=str(exc))
        except Exception as exc:report.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        (root/(name+'.json')).write_text(json.dumps(report,indent=2));summary[name]={k:report.get(k) for k in ('status','reason','translation_failures','translation_components','balance','analytical_oracle')};summary[name]['report_path']=str(root/(name+'.json'));summary[name]['visual']=str(root/(name+'.png'))
        if report.get('beam_comparison'):summary[name]['beam_failures']=report['beam_comparison']['failure_counts']['native_field_vs_sestra'];summary[name]['beam_components']=report['beam_comparison']['compared_station_components']
    result={'schema_version':1,'qualification':False,'mode':mode,'status':'passed' if summary and all(v['status']=='passed' for v in summary.values()) else 'not_executed' if all(v['status']=='not_executed' for v in summary.values()) else 'failed','cases':summary,'runtime':{'NUMBA_DISABLE_JIT':os.environ.get('NUMBA_DISABLE_JIT'),'pools':threadpool_info()},'loaded_sources':{x.__name__:identity(x.__file__) for x in (sif,document,semantics,assembly,elements,e4_pl_element,importer)},'harness':identity(__file__),'body_wall_seconds':time.perf_counter()-started}
    (root/'convergence.json').write_text(json.dumps(convergence_summary(root),indent=2));result['convergence_path']=str(root/'convergence.json')
    (root/'suite_report.json').write_text(json.dumps(result,indent=2));return result
