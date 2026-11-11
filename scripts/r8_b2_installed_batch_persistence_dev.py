"""Installed experimental B2 input persistence and 111-case functional slice.

Preparation is stdlib-only. Every executing phase needs explicit root authority;
closed historical registrations are inputs/evidence, never execution authority.
No backend override, solver-state checkpoint, qualification or performance claim.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/agent_plans/development/r8-b2-installed-batch-persistence-20261010'
V2='--v2' in sys.argv
V3='--v3-beam' in sys.argv
OUT=BASE/'v3-beam' if V3 else BASE/'v2' if V2 else BASE
PRIOR=ROOT/'scripts/r8_b2_installed_original_target_dev.py'
BUILDER=ROOT/'docs/agent_plans/development/r8-b2-large-work-allocation-20261009/frozen_model_builder.py'
BEAM_BUILDER=ROOT/'scripts/b2_large_scale_development.py'
CONTROL=ROOT/'tests/test_r8_b2_installed_batch_persistence_dev.py'
SOLVER_SHA='0ed73724bf3c8ccffcc0134628d94720d640840f9592ac8890779f54285ee3d8'
CONSUMER_SHA='84b738061f0627da5c68fb2e8efb0a0882a2c19ce0f1750595050923f58bd66b'
ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS',
    'NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT')}
ENV['PYTHONOPTIMIZE']='0'
PHASES={'controls':20.,'beam':60.} if V3 else {'controls':20.,'full':240.,'beam':60.} if V2 else {'install':30.,'controls':30.,'full':240.,'beam':60.}
ALLOWANCE=180. if V3 else 420. if V2 else 600.


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(json_safe(data),indent=2,sort_keys=True,allow_nan=False,default=str)+'\n',encoding='utf-8')


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return repr(value)
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    return value


def extrema(values):
    """Independent ordered extrema; first submitted case governs exact ties."""
    require(values and all(math.isfinite(v) for _,v in values),'finite ordered extrema required')
    return min(values,key=lambda v:v[1]),max(values,key=lambda v:v[1])


def payload_bytes(payload):
    """Hash the canonical public consumer payload without double encoding."""
    if type(payload) is bytes:return payload
    if type(payload) is str:return payload.encode('utf-8')
    raise RuntimeError('canonical payload bytes/text required')


def builder_path(phase):
    if phase=='full':return BUILDER
    if phase=='beam':return BEAM_BUILDER
    raise RuntimeError('declared physical inventory phase required')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def work_predicate(internal,external):
    """Original relative criterion, separately from product absolute floor."""
    finite=math.isfinite(internal) and math.isfinite(external)
    defect=internal-external
    relative=1e-8*abs(external)
    return dict(internal_work=internal,external_work=external,work_defect=defect,
        original_work_allowance=relative,product_work_allowance=1e-12+relative,
        original_work_pass=finite and abs(defect)<=relative,
        product_work_pass=finite and abs(defect)<=1e-12+relative)


def counter_predicate(factors,primary,correction,cases,same_handle):
    return factors==1 and primary==cases and 0<=correction<=cases and same_handle


def package(wheel,prefix):
    with ZipFile(wheel) as z:
        names=[n for n in z.namelist() if n.startswith(prefix+'/') and not n.endswith('/')]
        require(names and all('..' not in Path(n).parts for n in names),'invalid wheel inventory')
        return {n:hashlib.sha256(z.read(n)).hexdigest() for n in names}


def prepare(solver,consumer):
    prior=module('b2_public_static_identity',PRIOR)
    require(sha(solver)==SOLVER_SHA and sha(consumer)==CONSUMER_SHA,'adopted wheel identity')
    require(sha(BUILDER)==json.loads((BUILDER.parent/'preflight.json').read_text())['builder_sha256'],
        'original builder drift')
    packages=package(solver,'anyfesolver')|package(consumer,'anyfem')
    for name in prior.B2_FILES:
        require(sha(ROOT/'src/anyfesolver'/name)==packages['anyfesolver/'+name],'B2 current-source drift '+name)
    closure=json.loads((BASE/'CONSUMER_SOURCE_CLOSURE.json').read_text())
    require(closure['wheel_sha256']==CONSUMER_SHA and closure['mismatches']==['anyfem/solve/stiffened_shell.py'],
        'consumer closure changed; root artifact decision needed')
    for name,data in closure['checked'].items():
        require(sha(Path(closure['source_root'])/name)==data['source'],'consumer review source drift '+name)
    paths=(Path(__file__),CONTROL,PRIOR,BUILDER,ROOT/'scripts/b2_large_scale_development.py',
        ROOT/'scripts/_b3_e4_f03_local_scientific_supervisor.py',ROOT/'scripts/_b3_e4_f03_local_scientific_cases.py',
        BASE/'CONSUMER_SOURCE_CLOSURE.json',BASE/'CONSUMER_SOURCE_DELTA.diff')
    for path in (Path(__file__),CONTROL):ast.parse(path.read_text())
    p=dict(status='PREPARED_SLOT_HELD',solver_wheel=str(Path(solver).resolve()),consumer_wheel=str(Path(consumer).resolve()),
        solver_sha256=SOLVER_SHA,consumer_sha256=CONSUMER_SHA,package=packages,
        site=str(BASE/'package/site'),source={str(p.relative_to(ROOT)):sha(p) for p in paths},
        current_b2_sources={name:sha(ROOT/'src/anyfesolver'/name) for name in prior.B2_FILES},
        consumer_choice_required='frozen84b738-B2-route-GE-delta-explicit',consumer_source_delta=closure['mismatches'],
        matrix_sha256=prior.EXPECTED_MATRICES,
        full=dict(nx=100,ny=100,shells=10000,members=1000,pads=100,incidences=400,rows=2000,
            dofs=67266,free=66660,batch_sizes=[1,10,100],primary_count=111,selected_dofs=[30902,30900],
            selected_pad='pad-0-5',selected_member='b2-0-0',stations=[0.,.5,1.],load_coordinate=30902),
        beam=dict(nx=20,ny=20,shells=400,members=1000,pads=100,incidences=400,rows=2000,
            batch_sizes=[100],primary_count=100),
        persistence='experimental physical sparse-v2 standalone AND nativeProject definition-only',
        policy='unchanged eligible current GENERAL/default; no factor/backend override',
        limits=dict(total_seconds=ALLOWANCE,reserve_seconds=60.,phase_seconds=PHASES,rss_bytes=4*1024**3,
            cpu=[0],threads=1,jit_disabled=True,optimize=0),qualification=False,performance=False,checkpoint=False)
    if V2 or V3:
        receipt=BASE/'attempt-install-01/receipt.json';result=BASE/'attempt-install-01/result.json'
        require(json.loads(receipt.read_text())['reason']=='completed' and json.loads(receipt.read_text())['exit_code']==0
            and json.loads(result.read_text())['status']=='PASS_INSTALLED_ARTIFACT','v1 installed artifact not reusable')
        for name,h in packages.items():require(sha(Path(p['site'])/name)==h,'reused installed bytes '+name)
        p['reused_install']=dict(receipt_sha256=sha(receipt),result_sha256=sha(result),
            binding_sha256=sha(BASE/'attempt-install-01/binding.json'),
            v1_prepared_sha256=sha(BASE/'PREPARED.json'),package_files=len(packages),site=p['site'])
    if V3:
        full=BASE/'v2/attempt-full-01';r=json.loads((full/'result.json').read_text())
        receipt=json.loads((full/'receipt.json').read_text())
        require(r['status']=='PASS_INSTALLED_FUNCTIONAL' and r['validated']==111 and r['primary_count']==111
            and receipt['reason']=='completed' and receipt['exit_code']==0,'unchanged v2 fullPASS prerequisite')
        require(json.loads((BASE/'v2/ledger.json').read_text())['status']=='CLOSED','v2 terminal ledger required')
        p['prior_full_pass']=dict(receipt_sha256=sha(full/'receipt.json'),result_sha256=sha(full/'result.json'),
            ledger_sha256=sha(BASE/'v2/ledger.json'),prepared_sha256=sha(BASE/'v2/PREPARED.json'),
            validated_cases=111,primary_count=111,correction_count=r['correction_count'])
    save(OUT/'PREPARED.json',p)
    return p


def assert_prepared(p):
    require(sha(p['solver_wheel'])==p['solver_sha256'] and sha(p['consumer_wheel'])==p['consumer_sha256'],'wheel drift')
    for name,value in p['source'].items():require(sha(ROOT/name)==value,'prepared source drift '+name)
    for name,value in p['current_b2_sources'].items():require(sha(ROOT/'src/anyfesolver'/name)==value,'B2 source drift '+name)
    if 'reused_install' in p:
        require(sha(BASE/'attempt-install-01/receipt.json')==p['reused_install']['receipt_sha256']
            and sha(BASE/'attempt-install-01/result.json')==p['reused_install']['result_sha256']
            and sha(BASE/'attempt-install-01/binding.json')==p['reused_install']['binding_sha256']
            and sha(BASE/'PREPARED.json')==p['reused_install']['v1_prepared_sha256'],'reused install binding drift')
    if 'prior_full_pass' in p:
        previous=p['prior_full_pass'];full=BASE/'v2/attempt-full-01'
        require(sha(full/'receipt.json')==previous['receipt_sha256'] and sha(full/'result.json')==previous['result_sha256']
            and sha(BASE/'v2/ledger.json')==previous['ledger_sha256']
            and sha(BASE/'v2/PREPARED.json')==previous['prepared_sha256'],'prior fullPASS drift')


def runtime(p):
    import numba
    import psutil
    from threadpoolctl import threadpool_info
    site=Path(p['site']).resolve();imports={}
    for name,m in tuple(sys.modules.items()):
        if name.split('.')[0] in ('anyfesolver','anyfem') and getattr(m,'__file__',None):
            file=Path(m.__file__).resolve();require(file.is_relative_to(site),'source import '+name)
            relative=file.relative_to(site).as_posix()
            require(relative in p['package'] and sha(file)==p['package'][relative],'actual installed hash '+name)
            imports[name]=dict(path=str(file),sha256=sha(file))
    r=dict(affinity=psutil.Process().cpu_affinity(),optimize=sys.flags.optimize,pools=threadpool_info(),
        numba=dict(disable_jit=int(numba.config.DISABLE_JIT),threads=numba.get_num_threads(),version=numba.__version__),
        environment={k:os.environ.get(k) for k in ENV},imports=imports,python=sys.version,executable=sys.executable)
    require(r['affinity']==[0] and r['optimize']==0 and r['environment']==ENV
        and r['numba']['disable_jit']==1 and r['numba']['threads']==1
        and r['pools'] and all(pool['num_threads']==1 for pool in r['pools']),'effective runtime refusal')
    return r


def matrix_identity(matrix,np):
    matrix=matrix.tocsr();h=hashlib.sha256()
    for value,dtype in ((matrix.shape,'<i8'),(matrix.indptr,'<i8'),(matrix.indices,'<i8'),(matrix.data,'<f8')):
        h.update(np.asarray(value,dtype=dtype).tobytes())
    return dict(shape=matrix.shape,nnz=matrix.nnz,sha256=h.hexdigest())


def execute_installed(phase,attempt,p):
    import psutil
    psutil.Process().cpu_affinity([0])
    site=Path(p['site']).resolve()
    sys.path[:]=[str(site)]+[s for s in sys.path if Path(s or '.').resolve()!=site
        and not Path(s or '.').resolve().is_relative_to(ROOT/'src')
        and not Path(s or '.').resolve().is_relative_to(Path('C:/Github/ANYfeSolver/src'))]
    for name,h in p['package'].items():require(sha(site/name)==h,'combined site mismatch '+name)
    import numpy as np
    import scipy
    from anyfesolver import stiffened_shell as api
    from anyfesolver.linalg import backend_operation_counts
    from anyfem.solve.stiffened_shell_b2 import (ExperimentalB2PhysicalRigidFootprintPanel,
        SCHEMA_PHYSICAL_FOOTPRINT_SPARSE_V2,save_case,load_case)
    from anyfem.solve.stiffened_shell_project import point_panel_from_analysis
    from anyfem.io.project_file import save_project,load_project
    from anyfem.model.project import Project
    from anyfem.model.records import AnalysisDefinition
    save(attempt/'runtime-before.json',runtime(p))
    record=dict(status='UNRESOLVED',phase=phase,qualification=False,performance=False,checkpoint=False)
    owner=None;start=time.perf_counter();state={'primary':0,'correction':0,'validated':0,'current':None}
    state['accepted_prefix']=[]
    stream=(attempt/'cases.jsonl').open('w',encoding='utf-8')
    try:
        builder=module('bound_b2_batch_model',builder_path(phase))
        config=builder.settings('large' if phase=='full' else 'beam_rich',100)
        definition,inventory,shell_id=builder.model(config);record['inventory']=inventory
        expected=p[phase]
        require(inventory['q4_shells']==expected['shells'] and inventory['physical_b2_members']==expected['members']
            and inventory['attached_patches']==100 and inventory['footprint_rows']==2000,'physical inventory')
        load_coordinate=6*(shell_id(config['nx'],config['ny']//2)-1)+2
        dofs=(load_coordinate,load_coordinate-2);pad=definition.physical_footprints[0].footprint_id
        member=definition.members[0].member_id;selection=api.B2PhysicalRigidFootprintSelection((pad,))
        loads=[]
        for i in range(100):
            load=np.zeros(inventory['dofs']);load[load_coordinate]=-(i+1.)
            loads.append((f'case-{i+1}',load))
        original_matrices={k:matrix_identity(v,np) for k,v in {'T':definition.transformation}.items()}
        generate_end=time.perf_counter();record['generation_seconds']=generate_end-start
        if phase=='full':
            panel=ExperimentalB2PhysicalRigidFootprintPanel.from_definition(definition,loads,selected_dofs=dofs,
                selected_footprints=selection,recover_members={member:(0.,.5,1.)},schema=SCHEMA_PHYSICAL_FOOTPRINT_SPARSE_V2)
            standalone=save_case(panel,attempt/'full.anyfem-b2.json');reopened=load_case(standalone)
            require(reopened.payload==panel.payload,'standalone canonical identity')
            project=Project('experimental full-size physical B2 inputs',mesh_only=True)
            analysis=AnalysisDefinition.experimental_point_panel('physical-B2',reopened);project.add_analysis(analysis)
            native=save_project(project,attempt/'full.anyfem');restored=load_project(native)
            reopened=point_panel_from_analysis(restored.analyses[analysis.id])
            require(reopened.payload==panel.payload,'native Project canonical physical identity')
            persisted=reopened.to_dict()['cases_sparse']
            require(len(persisted)==100,'persisted ordered load inventory')
            for index,row in enumerate(persisted):
                require(row==dict(id=f'case-{index+1}',load_sparse=dict(size=inventory['dofs'],
                    entries=[[load_coordinate,-float(index+1)]])),'persisted exact load identity')
                load=np.zeros(inventory['dofs'])
                for position,force in row['load_sparse']['entries']:load[position]=force
                loads[index]=(row['id'],load)
            definition2=reopened.definition();require(matrix_identity(definition2.transformation,np)==original_matrices['T'],
                'persisted CSR/support identity')
            require(len(definition2.physical_footprints)==100 and all(np.asarray(definition2.prescribed_offset)==0.),'persisted pads/offset')
            record['persistence']=dict(schema=reopened.schema,canonical_sha256=hashlib.sha256(payload_bytes(panel.payload)).hexdigest(),
                standalone_sha256=sha(standalone),native_sha256=sha(native),standalone_bytes=standalone.stat().st_size,
                native_bytes=native.stat().st_size,definition_only=True)
            del definition,definition2,panel,project,restored
            construct=lambda:reopened.create_analysis()
        else:construct=lambda:api.create_analysis(definition=definition)
        persistence_end=time.perf_counter();record['persistence_seconds']=persistence_end-generate_end
        before=backend_operation_counts();owner=construct();compiled=time.perf_counter()
        record['compile_seconds']=compiled-persistence_end;record['compiler']=dict(owner._attachment_elimination_info)
        record['backend_compile_delta']={k:backend_operation_counts()[k]-before[k] for k in before}
        delta=record['backend_compile_delta']
        require(delta['scipy_factorization_attempts']+delta['pardiso_factorization_attempts']==1
            and delta['pardiso_stale_rebuilds']==0,'extra factor/backend fallback')
        require(owner._handle.factorization_count==1 and owner._handle.solve_count==0,'fresh one factor')
        record['matrices']={k:matrix_identity(getattr(owner,'_'+k),np) for k in ('K','C','T')}
        if phase=='full':
            for k,h in p['matrix_sha256'].items():require(record['matrices'][k]['sha256']==h,'frozen '+k+' identity')
        require(len(owner._footprint_incidence)==400 and owner._C.shape[0]==2000 and np.all(owner._u0==0.),'actual trace/support')
        save(attempt/'result.json',record)
        identity=id(owner._handle);original_solve=owner._handle.solve
        solve_code=owner.solve_linear.__func__.__code__;cert_code=owner._explicit_load_resolved.__func__.__code__
        def capture_solve(rhs):
            current=state['current'];require(current is not None,'solve outside actual case')
            current['solves']+=1;require(current['solves']<=2,'more than one correction')
            role='primary' if current['solves']==1 else 'correction';state[role]+=1
            require(id(owner._handle)==identity,'factor handle changed')
            return original_solve(rhs)
        owner._handle.solve=capture_solve
        def observe(frame,event,value):
            if frame.f_code==solve_code and event=='call':
                state['current']=dict(ordinal=state['validated']+1,solves=0,certification=False,
                    batch_size=state['batch_size'],case_id=f"case-{state['validated']-state['batch_start']+1}")
            if event!='return':return
            if frame.f_code==cert_code:
                if state['current'] is not None:state['current']['certification']|=bool(value)
                return
            if frame.f_code!=solve_code:return
            current=state['current'];f=frame.f_locals
            arrays={k:np.asarray(v) for k,v in f.items() if isinstance(v,np.ndarray)}
            errors=[]
            def check(ok,label):
                if not ok:errors.append(label)
            current.update(status=getattr(value,'status',None),failure=getattr(value,'failure_reason',None))
            check(current['status']=='completed','actual public solve failed')
            needed=('u','q','lam','gap','reaction','projected','equilibrium_tolerance','gap_tolerance',
                'explicit_rhs','resolution_floor','gap_scale','applied')
            check(all(k in arrays for k in needed),'actual guard frame missing')
            if all(k in arrays for k in needed) and current['status']=='completed':
                current['selected']=dict(displacements=[value.displacements[j] for j in dofs],
                    affine_reactions=[value.affine_reactions[j] for j in dofs],
                    beam_resultants_local=value.beam_resultants_local,
                    footprint_wrenches=[asdict(v) for v in value.footprint_wrenches],
                    internal_strain_energy=value.internal_strain_energy)
                def ratio(r,t):
                    vals=np.full(r.shape,np.inf);np.divide(np.abs(r),t,out=vals,where=t>0.)
                    vals[(t==0.)&(r==0.)]=0.;return float(np.max(vals))
                eq=ratio(arrays['projected'],arrays['equilibrium_tolerance']);gap=ratio(arrays['gap'],arrays['gap_tolerance'])
                active=arrays['explicit_rhs']>0.
                need=bool(np.any(arrays['resolution_floor'][active]>1e-9*arrays['explicit_rhs'][active]))
                gap_need=bool(np.any(np.abs(arrays['gap'])>1e-9*arrays['gap_scale']))
                check(eq<=1. and gap<=1.,'original per-row physical guards')
                check(not need or bool(f.get('certified')) and current['certification'],'actual explicit certification')
                check(not gap_need or current['certification'],'actual gap certification')
                check(np.isfinite(value.affine_reactions).all() and len(value.constraint_multipliers)==2000
                    and len(value.constraint_rows)==2000 and len(value.displacements)==inventory['dofs'],'full recovery')
                check(value.production_qualified is False,'unexpected qualification')
                work=work_predicate(2.*value.internal_strain_energy,float(arrays['u']@arrays['applied']));current.update(work)
                rank=state['validated']-state['batch_start']+1
                check(np.count_nonzero(arrays['applied'])==1 and arrays['applied'][load_coordinate]==-float(rank),
                    'exact declared case/load identity')
                check(work['original_work_pass'] and work['product_work_pass'],'original relative work')
                check(len(value.footprint_wrenches)==1 and len(value.beam_resultants_local)==1,'selected recovery inventory')
                for wrench in value.footprint_wrenches:
                    check(wrench.footprint_id==pad and len(wrench.node_actions)==4 and not wrench.production_qualified,'pad identity')
                    for k in ('common_reference_origin_m','shell_force_N','shell_moment_Nm','beam_force_N','beam_moment_Nm'):
                        check(np.asarray(getattr(wrench,k)).shape==(3,) and np.isfinite(getattr(wrench,k)).all(),'finite wrench '+k)
                    for action in wrench.node_actions:
                        for k in ('shell_force_N','shell_direct_couple_Nm','beam_force_N','beam_direct_couple_Nm','gap'):
                            check(np.isfinite(getattr(action,k)).all(),'finite action '+k)
                for name,resultants in value.beam_resultants_local:
                    check(name==member and np.asarray(resultants).shape==(3,6) and np.isfinite(resultants).all(),'member resultants')
                section=f.get('result');check(hasattr(section,'strains') and np.asarray(section.strains).shape==(3,6)
                    and np.isfinite(section.strains).all() and np.isfinite(section.resultants).all(),'section material recovery')
                current.update(equilibrium_ratio=eq,gap_ratio=gap,certification_required=need,gap_certificate_required=gap_need)
            current['errors']=errors
            stream.write(json.dumps(json_safe(current),allow_nan=False,default=str)+'\n');stream.flush()
            if errors:
                # Complete caller arrays and published/failed candidate precede any assertion.
                np.savez(attempt/'first-failure-frame.npz',**arrays)
                if value is not None:
                    np.savez(attempt/'first-failure-public.npz',u=np.asarray(getattr(value,'displacements',())),
                        reaction=np.asarray(getattr(value,'affine_reactions',())),
                        lambda_=np.asarray(getattr(value,'constraint_multipliers',())),gap=np.asarray(getattr(value,'constraint_gaps',())))
                save(attempt/'first-failure.json',current)
        def on_case(case_id):
            current=state['current'];require(current['case_id']==case_id,'source published order')
            require(not current['errors'],'first per-case predicate failure '+','.join(current['errors']))
            require(current['solves'] in (1,2),'per-case counters')
            state['validated']+=1
            state['accepted_prefix'].append(dict(batch_size=current['batch_size'],case_id=case_id,
                validated_ordinal=state['validated'],selected=current['selected']))
            # This file exists even if the current batch never returns.
            save(attempt/'accepted-selected-prefix.json',state['accepted_prefix'])
        batches=[];selected=[];sys.setprofile(observe)
        try:
            for count in expected['batch_sizes']:
                batch_start=time.perf_counter();prefix_before=state['validated']
                state.update(batch_size=count,batch_start=prefix_before)
                input_hashes=[hashlib.sha256(v.tobytes()).hexdigest() for _,v in loads[:count]]
                batch=owner.solve_linear_many(tuple(loads[:count]),selected_dofs=dofs,selected_footprints=selection,
                    recover_members={member:(0.,.5,1.)},on_case=on_case)
                info=dict(count=count,status=batch.status,failure=batch.failure_reason,seconds=time.perf_counter()-batch_start,
                    factorization_count=batch.factorization_count,solve_count=batch.solve_count,maximum_rhs_chunk=batch.maximum_rhs_chunk,
                    validated_cases=state['validated']-prefix_before,
                    envelopes=[asdict(v) for v in batch.member_resultant_envelopes])
                batches.append(info);save(attempt/'batches.json',batches)
                require(input_hashes==[hashlib.sha256(v.tobytes()).hexdigest() for _,v in loads[:count]],'caller load mutation')
                require(batch.status=='completed' and len(batch.cases)==count and info['validated_cases']==count,'batch publication/prefix')
                require(batch.factorization_count==1 and batch.solve_count==state['primary']+state['correction']
                    and batch.maximum_rhs_chunk==1 and len(batch.member_resultant_envelopes)==18,'batch counters/bounds/envelopes')
                require(tuple(c.case_id for c in batch.cases)==tuple(f'case-{i+1}' for i in range(count)),'ordered case publication')
                for envelope in batch.member_resultant_envelopes:
                    j=('N','Qy','Qz','Mx','My','Mz').index(envelope.component)
                    values=[(c.case_id,dict(c.beam_resultants_local)[member][envelope.station_index][j]) for c in batch.cases]
                    low,high=extrema(values)
                    require(envelope.member_id==member and envelope.station_normalized==(0.,.5,1.)[envelope.station_index]
                        and (envelope.minimum_case_id,envelope.minimum)==low
                        and (envelope.maximum_case_id,envelope.maximum)==high,'independent member envelope governance')
                for field,items in (('selected_displacements',batch.displacement_envelopes),
                    ('selected_affine_reactions',batch.affine_reaction_envelopes)):
                    require(len(items)==2,'selected envelope count')
                    for index,envelope in enumerate(items):
                        low,high=extrema([(c.case_id,getattr(c,field)[index]) for c in batch.cases])
                        require(envelope.selected_dof==dofs[index] and (envelope.minimum_case_id,envelope.minimum)==low
                            and (envelope.maximum_case_id,envelope.maximum)==high,'independent selected envelope governance')
                for case in batch.cases:
                    require(np.isfinite(case.selected_displacements).all() and np.isfinite(case.selected_affine_reactions).all(),
                        'finite selected fields')
                    selected.append(dict(batch_size=count,case_id=case.case_id,selected_displacements=case.selected_displacements,
                        selected_affine_reactions=case.selected_affine_reactions,beam_resultants_local=case.beam_resultants_local,
                        internal_strain_energy=case.internal_strain_energy))
                save(attempt/'selected.json',selected)
        finally:sys.setprofile(None);owner._handle.solve=original_solve
        require(counter_predicate(owner._handle.factorization_count,state['primary'],state['correction'],
            expected['primary_count'],id(owner._handle)==identity),'total one-factor/correction count')
        require(state['validated']==expected['primary_count'],'all cases validated')
        record.update(status='PASS_INSTALLED_FUNCTIONAL',validated=state['validated'],primary_count=state['primary'],
            correction_count=state['correction'],batch_and_recovery_seconds=time.perf_counter()-compiled,
            selected_count=len(selected),batches=batches)
    except Exception as exc:
        import traceback
        (attempt/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        record.update(status='FAILED_OR_UNRESOLVED',error=str(exc),error_type=type(exc).__name__,validated=state['validated'],
            primary_count=state['primary'],correction_count=state['correction'])
    finally:
        sys.setprofile(None);stream.close()
        if owner is not None:record['terminal_factor']=owner._handle.diagnostics();owner.close()
        record['observed_seconds']=time.perf_counter()-start
        save(attempt/'result.json',record);save(attempt/'runtime-after.json',runtime(p))
    return record['status']=='PASS_INSTALLED_FUNCTIONAL'


def child(phase,attempt):
    p=json.loads((attempt/'binding.json').read_text())['preparation'];assert_prepared(p)
    import psutil
    psutil.Process().cpu_affinity([0])
    basic=dict(affinity=psutil.Process().cpu_affinity(),environment={k:os.environ.get(k) for k in ENV},
        optimize=sys.flags.optimize,executable=sys.executable,python=sys.version,phase=phase)
    require(basic['affinity']==[0] and basic['environment']==ENV and basic['optimize']==0,'basic child runtime refusal')
    save(attempt/'runtime-basic.json',basic)
    if phase in ('install','controls'):
        save(attempt/'runtime-before.json',basic)
    if phase=='install':
        site=Path(p['site']);require(not site.exists(),'fresh site required');site.parent.mkdir(exist_ok=True)
        command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--target',str(site),
            p['solver_wheel'],p['consumer_wheel']]
        save(attempt/'install-command.json',command);start=time.perf_counter()
        result=subprocess.run(command,check=False)
        require(result.returncode==0,'combined installation failed')
        for name,h in p['package'].items():require(sha(site/name)==h,'installed wheel bytes '+name)
        save(attempt/'result.json',dict(status='PASS_INSTALLED_ARTIFACT',seconds=time.perf_counter()-start,
            package_files=len(p['package']),source_choice='frozen84b738-B2-route-GE-delta-explicit'))
        save(attempt/'runtime-after.json',dict(basic,affinity=psutil.Process().cpu_affinity()))
        return True
    if phase=='controls':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(CONTROL.parent),pattern=CONTROL.name)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        save(attempt/'result.json',dict(status='PASS_CONTROLS' if result.wasSuccessful() else 'FAILED_CONTROLS',
            tests=result.testsRun,errors=len(result.errors),failures=len(result.failures)))
        save(attempt/'runtime-after.json',dict(basic,affinity=psutil.Process().cpu_affinity()))
        return result.wasSuccessful()
    return execute_installed(phase,attempt,p)


def run(phase,authority):
    a=json.loads(Path(authority).read_text());p=json.loads((OUT/'PREPARED.json').read_text());assert_prepared(p)
    require(a['status']=='AUTHORIZED' and a['numerical_slot_transferred'] is True
        and a['prepared_sha256']==sha(OUT/'PREPARED.json') and phase in a['phases'], 'root reviewed authority/slot required')
    require(a['consumer_choice']==p['consumer_choice_required'],'unadopted consumer artifact delta')
    if V2 or V3:require(a['reused_install_receipt_sha256']==p['reused_install']['receipt_sha256'],'root verified reused-install contract required')
    if V3:require(a['prior_full_receipt_sha256']==p['prior_full_pass']['receipt_sha256'],'root verified prior fullPASS required')
    ledger_path=OUT/'ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='OPEN',allowance_seconds=ALLOWANCE,
        reserve_seconds=60.,used_seconds=0.,attempts=[])
    require(ledger['status']=='OPEN' and not any(item['phase']==phase for item in ledger['attempts']),'closed/consumed phase')
    predecessors=(() if phase=='controls' else ('controls',)) if V3 else (() if phase=='controls' else ('controls',) if phase=='full' else ('full',)) if V2 else (
        ('install',) if phase=='controls' else ('install','controls') if phase=='full' else ('full',) if phase=='beam' else ())
    for previous in predecessors:
        require(any(x['phase']==previous and x['passed'] for x in ledger['attempts']),'missing predecessor '+previous)
    available=ALLOWANCE-60.-ledger['used_seconds'];limit=min(PHASES[phase],available)
    require(limit==PHASES[phase],'insufficient authorized phase budget/reserve')
    attempt=OUT/('attempt-'+phase+'-01');attempt.mkdir(exist_ok=False)
    temp=OUT/'package/temp';temp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env.update(ENV);env.update(PYTHONPATH=p['site'],TEMP=str(temp),TMP=str(temp),PIP_DISABLE_PIP_VERSION_CHECK='1')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',phase,str(attempt)]
    if V2:command.append('--v2')
    if V3:command.append('--v3-beam')
    save(attempt/'binding.json',dict(command=command,cwd=str(attempt),preparation=p,authority_sha256=sha(authority),
        environment={k:env[k] for k in (*ENV,'PYTHONPATH','TEMP','TMP')}))
    for name,h in p['source'].items():
        path=attempt/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        require(sha(path)==h,'snapshot drift')
    sys.path.insert(0,str(ROOT))
    from scripts._b3_e4_f03_local_scientific_supervisor import supervise,SupervisorLimits
    save(ledger_path,ledger)
    with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr:
        receipt=supervise(command,cwd=attempt,environment=env,stdout=stdout,stderr=stderr,
            limits=SupervisorLimits(limit,4*1024**3,.05))
    save(attempt/'receipt.json',asdict(receipt))
    result=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {}
    passed=receipt.reason=='completed' and receipt.exit_code==0 and result.get('status','').startswith('PASS')
    ledger['used_seconds']+=receipt.elapsed_seconds
    ledger['attempts'].append(dict(phase=phase,passed=passed,process_seconds=receipt.elapsed_seconds,
        reason=receipt.reason,exit_code=receipt.exit_code))
    if not passed or phase=='beam':
        ledger.update(status='CLOSED',discarded_unused_seconds=ALLOWANCE-ledger['used_seconds'])
    save(ledger_path,ledger)
    print(json.dumps(dict(status=ledger['status'],phase=phase,passed=passed,used_seconds=ledger['used_seconds'])))
    return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',nargs=2,metavar=('SOLVER_WHEEL','CONSUMER_WHEEL'))
    parser.add_argument('--v2',action='store_true')
    parser.add_argument('--v3-beam',action='store_true')
    parser.add_argument('--run',choices=tuple(PHASES));parser.add_argument('--authority');parser.add_argument('--child',nargs=2)
    args=parser.parse_args()
    if args.prepare:
        OUT.mkdir(parents=True,exist_ok=True);prepare(*args.prepare);print(json.dumps({'status':'PREPARED_SLOT_HELD','sha256':sha(OUT/'PREPARED.json')}))
    elif args.child:sys.exit(0 if child(args.child[0],Path(args.child[1])) else 1)
    elif args.run and args.authority:sys.exit(0 if run(args.run,args.authority) else 1)
    else:parser.error('explicit prepare or reviewed authority phase required')
