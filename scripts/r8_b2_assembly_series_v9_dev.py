"""Connected Q4 assembly series preparation; execution requires fresh root authority.

No numerical import occurs in preparation. Delivered inventory caps are immutable;
large inventory fit and resource authority remain unassessed after actual scratch
cleanup. The installed callable stays gated before any large model construction.
"""
import argparse
import ast
import base64
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import os
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/agent_plans/development/r8-b2-assembly-series-20261010'
OUT=BASE/'execution-v9'
PREVIOUS=BASE/'execution-v6'
LIFETIME=ROOT/'docs/agent_plans/development/r8-b2-q4-lifetime-20261010'
EXECUTION_AUTHORITY_NAME='ROOT_EXECUTION_AUTHORITY.json'
METRICS=ROOT/'docs/agent_plans/development/r8-b2-coo-scratch-cleanup-20261010'
G5=ROOT/'docs/agent_plans/development/r8-b2-g5-readiness-20261010'
FUNCTIONAL=ROOT/'docs/agent_plans/development/r8-b2-installed-batch-persistence-20261010'
FULL=FUNCTIONAL/'v2/attempt-full-01/full.anyfem-b2.json'
FULL_SHA='cf7970c30839256a2eabc0f9fc2a5521e61b641b92b4a6bca2ee68a8ab70f8b4'
SOLVER_SHA='00385aa8dcfaa472b1f0860257cf83baad263db958a250b0d0c22e9ebe0bff15'
CAPS=dict(objects=2000000,edges=8000000,buffers=500000,depth=40)
ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
 'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT')};ENV['PYTHONOPTIMIZE']='0';ENV['PYTHONDONTWRITEBYTECODE']='1'
PHASES=('controls','install','validate-install','public-smoke','low','middle','full-1','full-2','full-3')
SMOKE_CONFIG=dict(nx=3,ny=1,line_rows=[0],patch_x=[1],cases=1)
CONFIGS={2500:dict(nx=50,ny=50,line_rows=list(range(4,50,10)),patch_x=list(range(5,50,10)),cases=100),
 5000:dict(nx=100,ny=50,line_rows=list(range(4,50,10)),patch_x=list(range(5,100,10)),cases=100),
 10000:dict(nx=100,ny=100,line_rows=list(range(4,100,10)),patch_x=list(range(5,100,10)),cases=100)}

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def _contract_authority_gate(a,p,phase):
    require(a.get('status')=='PROSPECTIVELY_GRANTED_DEVELOPMENT_SCREEN' and phase in PHASES,'fresh series authority/phase')
    require(a['input_prepared_sha256']==p['predecessor_prepared_sha256'] and a['solver_wheel_sha256']==p['solver_sha256']==SOLVER_SHA
        and a['consumer_wheel_sha256']==p['consumer_sha256'] and a['full_definition_sha256']==FULL_SHA,'authority artifact/input binding')
    require((a['cumulative_seconds'],a['reserve_seconds'],a['process_cap_seconds'],a['controls_cap_seconds'],a['install_cap_seconds'],a['tree_rss_limit_bytes'])
        ==(3200,120,600,30,30,2147483648),'unchanged authority resource caps')
    require(a['cpu_affinity']==[0] and a['numerical_threads']==1 and a['jit_disabled'] is True and a['python_optimize']==0
        and a['one_numerical_job'] is True and a['inventories']==[2500,5000,10000] and a['full_warm_ordinal_groups']==[[1,2],[3,4],[5]],'authority runtime/inventory')
    smoke=a['public_smoke'];require(all(smoke[k]==v for k,v in dict(nx=3,ny=1,line_rows=[0],patch_x=[1],q4_shells=3,
        B2_members=3,pads=1,attachment_rows=20,dofs=72,fixed_dofs=12,loaded_dof=20,selected_pad='pad-0-1',
        selected_member='b2-0-0',public_cases=1,owners=1,process_cap_seconds=30).items()),'authority actual public smoke inventory')

def adjudicate_controls(result):
    require(result.testsRun==21 and not result.skipped and not result.errors and not result.failures and result.wasSuccessful(),'exact21 controls/zero skips/errors/failures')


def inventory(size):
    require(type(size) is int and size in CONFIGS,'registered connected size required')
    c=CONFIGS[size];nx,ny=c['nx'],c['ny'];lines=len(c['line_rows']);pads=lines*len(c['patch_x'])
    dofs=6*((nx+1)*(ny+1)+lines*(nx+1));slot=6*((ny//2)*(nx+1)+nx)+2
    return dict(shells=nx*ny,members=lines*nx,pads=pads,rows=20*pads,dofs=dofs,
        fixed_dofs=6*(ny+1),loaded_dof=slot,selected_dofs=[slot,slot-2],pad='pad-0-5',member='b2-0-0')

def smoke_inventory():
    return dict(shells=3,members=3,pads=1,rows=20,dofs=72,fixed_dofs=12,
        loaded_dof=20,selected_dofs=[20,18],pad='pad-0-1',member='b2-0-0')

def cap_obstruction(size):
    """Each extend(ndarray) materializes distinct NumPy scalar objects.

    _scatter emits full24x24/12x12 including zero entries. These are actual pre-clear
    entry counts; released lists now contribute zero scalar entries atpre-factor.
    Other model/cache/namespace storage is unassessed atlarge inventories.
    """
    i=inventory(size);n=3*(576*i['shells']+144*i['members'])
    return dict(preclear_scalar_entries=n,scalar_objects_lower_bound=0,traversal_edges_lower_bound=0,
        remaining_coo_entries=0,exceeds_objects=False,exceeds_edges=False,caps=CAPS,
        executable=False,reason='COO lifetime repaired; remaining large inventory/campaign authority not assessed')

def accounting_sample(capture,wall,physical_pass,factors):
    require(physical_pass is True and factors==1,'completed physical caller/one factor required')
    require(capture.get('status')=='COMPLETE' and not capture.get('timing_incomplete'),'complete capture required')
    phases=capture['phases'];names=[x['phase'] for x in phases]
    adopted=('shell_response_scatter','beam_response_scatter','K_CSR','attachment_rows','C_CSR')
    require(len(names)==len(set(names)) and all(n in names for n in adopted),'actual phase inventory')
    require(math.isfinite(wall) and wall>0 and all(math.isfinite(x['seconds']) and x['seconds']>=0 for x in phases),'negative/nonfinite timing')
    assembly=sum(x['seconds'] for x in phases if x['phase'] in adopted)
    require(assembly>0 and assembly<=wall and sum(x['seconds'] for x in phases)<=wall,'phase/wall accounting')
    storage=capture['storage'];require(storage['caps']==CAPS and type(storage['total_bytes']) is int and storage['total_bytes']>0,'declared complete storage/caps')
    require(capture['assembly_seconds']==assembly and capture.get('upper_bound') is False,'assembly semantics')
    return dict(assembly_seconds=assembly,selected_prefactor_bytes=storage['total_bytes'],constructor_wall=wall)

def summarize(groups):
    require(set(groups)==set(CONFIGS),'all three connected inventories required')
    summaries={}
    for size,rows in groups.items():
        require(len(rows)==5 and all(x.get('physical_pass') is True and x.get('status')=='PASS' and x.get('owner_close')==dict(private_mesh_released=True,private_elements_remaining=0,factor_handle_released=True) for x in rows),'exactly five physical PASS closed fresh owners required')
        times=[x['assembly_seconds'] for x in rows];memory=[x['selected_prefactor_bytes'] for x in rows]
        require(all(math.isfinite(x) and x>0 for x in times+memory),'positive finite samples')
        summaries[size]=dict(samples=len(rows),assembly_median=statistics.median(times),assembly_min=min(times),
            assembly_max=max(times),storage_median=statistics.median(memory),storage_min=min(memory),storage_max=max(memory))
    ratios=[]
    for low,high in ((2500,5000),(5000,10000)):
        t=summaries[high]['assembly_median']/summaries[low]['assembly_median']
        m=summaries[high]['storage_median']/summaries[low]['storage_median']
        ratios.append(dict(low=low,high=high,time_ratio=t,selected_storage_ratio=m,time_screen=t<=2.5,storage_screen=m<=2.3))
    return dict(summaries=summaries,ratios=ratios,screen_pass=all(x['time_screen'] and x['storage_screen'] for x in ratios),
        G5_acceptance=False,regression_comparison=None,qualification=False)

def installed_samples(attempt,plan,size,ordinals,*,smoke=False):
    """Future connected callable. Refuses unassessed authority before imports.

    Fresh root adoption and a remaining-inventory/resource assessment are required;
    this function cannot bypass the delivered source's caps or claim a failed sample.
    """
    require(plan.get('status')=='EXECUTABLE_DEVELOPMENT_SCREEN','reviewed executable preparation required')
    execution_authority(plan,'public-smoke' if smoke else 'low' if size==2500 else 'middle' if size==5000 else 'full-1')
    require((smoke is True and size==3 and ordinals==[]) or (smoke is False and len(ordinals)>0 and all(type(n) is int and 1<=n<=5 for n in ordinals) and len(set(ordinals))==len(ordinals)),'smoke or warm sample ordinals')
    site=Path(plan['package_site']).resolve()
    # Validate the actual installed graph before any product/scientific import.
    installed_before=installed_graph(plan);save(attempt/'installed-before-public.json',installed_before)
    sys.path[:]=[str(site)]+[x for x in sys.path if not Path(x or '.').resolve().is_relative_to(ROOT/'src')
        and not Path(x or '.').resolve().is_relative_to(Path('C:/Github/ANYfeSolver/src'))]
    # Imports and persistence are outside all measured owner constructions.
    import numpy as np
    from anyfesolver import stiffened_shell as api
    from anyfesolver._stiffened_shell_b2_assembly_diagnostics import capture_assembly
    from anyfem.solve.stiffened_shell_b2 import ExperimentalB2PhysicalRigidFootprintPanel,SCHEMA_PHYSICAL_FOOTPRINT_SPARSE_V2,save_case,load_case
    from anyfem.model.project import Project
    from anyfem.model.records import AnalysisDefinition
    from anyfem.io.project_file import save_project,load_project
    from anyfem.solve.stiffened_shell_project import point_panel_from_analysis
    helper=module('bound_series_functional',ROOT/'scripts/r8_b2_installed_batch_persistence_dev.py')
    validators=module('bound_series_warm',ROOT/'scripts/r8_b2_g5_readiness_dev.py')
    builder=module('bound_series_builder',ROOT/'scripts/b2_large_scale_development.py')
    save(attempt/'runtime-before.json',public_runtime(helper,plan))
    preparation_start=time.perf_counter();i=smoke_inventory() if smoke else inventory(size);config=SMOKE_CONFIG if smoke else CONFIGS[size];d,actual,_=builder.model(config)
    require(actual['q4_shells']==size and actual['physical_b2_members']==i['members'] and actual['attached_patches']==i['pads'],'connected inventory')
    require(actual['dofs']==i['dofs'] and actual['fixed_dofs']==i['fixed_dofs'] and actual['free_dofs']==i['dofs']-i['fixed_dofs'],'actual supported coordinate inventory')
    save(attempt/'preparation-started.json',dict(public_smoke=smoke,config=config,inventory=i,actual_inventory=actual))
    load=np.zeros(i['dofs']);load[i['loaded_dof']]=-1.
    # Full workload preserves the complete original100-load payload byte identity.
    if size==10000:
        require(sha(plan['full_fixture']['path'])==FULL_SHA,'full fixture drift');panel=load_case(plan['full_fixture']['path'])
    else:
        loads=[]
        for n in range(1,2 if smoke else 101):
            f=np.zeros(i['dofs']);f[i['loaded_dof']]=-float(n);loads.append((f'case-{n}',f))
        panel=ExperimentalB2PhysicalRigidFootprintPanel.from_definition(d,loads,selected_dofs=tuple(i['selected_dofs']),
            selected_footprints=api.B2PhysicalRigidFootprintSelection((i['pad'],)),recover_members={i['member']:(0.,.5,1.)},schema=SCHEMA_PHYSICAL_FOOTPRINT_SPARSE_V2)
    reopened=load_case(save_case(panel,attempt/'definition.anyfem-b2.json'));require(panel.payload==reopened.payload,'standalone persistence identity')
    project=Project('connected Q4 assembly inputs',mesh_only=True);analysis=AnalysisDefinition.experimental_point_panel('B2',reopened);project.add_analysis(analysis)
    restored=load_project(save_project(project,attempt/'definition.anyfem'));reopened=point_panel_from_analysis(restored.analyses[analysis.id])
    require(panel.payload==reopened.payload,'Project persistence identity')
    require(helper.matrix_identity(d.transformation,np)==helper.matrix_identity(reopened.definition().transformation,np),'support sparse identity')
    require(len(reopened.definition().shells)==i['shells'] and len(reopened.definition().members)==i['members'] and len(reopened.definition().physical_footprints)==i['pads'],'reopened physical inventory')
    save(attempt/'preparation-stages.json',dict(definition_generation_persistence_seconds=time.perf_counter()-preparation_start,
        canonical_sha256=hashlib.sha256(helper.payload_bytes(panel.payload)).hexdigest(),schema=reopened.schema,inventory=i))
    if size==10000:require(hashlib.sha256(helper.payload_bytes(panel.payload)).hexdigest()==FULL_SHA,'full canonical payload identity')
    del d,panel,project,analysis,restored
    if size!=10000:del loads
    records=[]
    for ordinal in [0]+list(ordinals):
        owner=None;captured=None;record=dict(ordinal=ordinal,cold=ordinal==0,public_smoke=smoke,physical_pass=False,status='CONSTRUCTION_STARTED')
        save(attempt/f'sample-{ordinal}-started.json',record);start=time.perf_counter()
        try:
            with capture_assembly() as captured:owner=reopened.create_analysis()
            wall=time.perf_counter()-start;record.update(capture=captured.report(),constructor_wall=wall,factor_before=owner._handle.diagnostics())
            save(attempt/f'sample-{ordinal}-before-physical.json',record)
            solve_start=time.perf_counter();batch=owner.solve_linear_many((('case-1',load),),selected_dofs=tuple(i['selected_dofs']),
                selected_footprints=api.B2PhysicalRigidFootprintSelection((i['pad'],)),recover_members={i['member']:(0.,.5,1.)})
            record['solve_recovery_seconds']=time.perf_counter()-solve_start
            save(attempt/f'sample-{ordinal}-returned.json',asdict(batch))
            checks=validators.validate_warm_reference(batch,[('case-1',-1.)])
            require(owner._handle.factorization_count==1 and 1<=owner._handle.solve_count<=2,'one primary/atmostone correction')
            record.update(accounting_sample(captured.report(),wall,True,owner._handle.factorization_count),physical_pass=True,status='PASS',work=checks,factor_after=owner._handle.diagnostics())
        except BaseException as exc:
            record.update(status='FAILED',error=str(exc),elapsed_since_construction_start=time.perf_counter()-start)
            (attempt/f'sample-{ordinal}-failure.txt').write_text(traceback.format_exc());raise
        finally:
            if captured is not None:record['capture']=captured.report()
            original=sys.exception()
            try:
                if owner is not None:
                    owner.close()
                    record['owner_close']=dict(private_mesh_released=owner._private_q4_mesh is None,private_elements_remaining=len(owner._private_q4_elements),factor_handle_released=owner._handle is None)
                    require(record['owner_close']==dict(private_mesh_released=True,private_elements_remaining=0,factor_handle_released=True),'actual private owner retirement incomplete')
            except BaseException as cleanup:
                record.update(status='FAILED',owner_close_error=repr(cleanup))
                if original is None:raise
                original.add_note('owner close failed: '+repr(cleanup))
            finally:save(attempt/f'sample-{ordinal}.json',record)
        records.append(record);save(attempt/'accepted-prefix.json',records)
    installed_after=installed_graph(plan);save(attempt/'installed-post-close.json',installed_after)
    require(installed_before==installed_after,'post-close installed namespace changed')
    save(attempt/'runtime-after.json',public_runtime(helper,plan))
    return records

def bytecode_guard():
    require(sys.dont_write_bytecode is True and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','effective bytecode disabled required')
    return dict(dont_write_bytecode=sys.dont_write_bytecode,pycache_prefix=sys.pycache_prefix)

def public_runtime(helper,plan):
    result=helper.runtime(dict(site=plan['package_site'],package=plan['package']))
    result.update(bytecode_guard());return result

def basic_runtime():
    import psutil
    r=dict(affinity=psutil.Process().cpu_affinity(),optimize=sys.flags.optimize,environment={k:os.environ.get(k) for k in ENV},executable=sys.executable,python=sys.version,**bytecode_guard())
    require(r['affinity']==[0] and r['optimize']==0 and r['environment']==ENV,'effective basic runtime');return r


def _execute_child_body(phase,attempt):
    import psutil
    psutil.Process().cpu_affinity([0]);p=json.loads((attempt/'binding.json').read_text())['preparation'];result=dict(status='FAILED')
    save(attempt/'runtime-basic-before.json',basic_runtime())
    try:
        if phase=='controls':
            import unittest
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_r8_b2_assembly_series_v9_dev.py')
            r=unittest.TextTestRunner(verbosity=2).run(suite);result.update(executed=r.testsRun,skips=len(r.skipped),errors=len(r.errors),failures=len(r.failures));adjudicate_controls(r)
            result['status']='PASS_CONTROLS'
        elif phase=='install':
            site=Path(p['package_site']);require(not site.exists(),'fresh isolated site already exists')
            require(sys.executable==p['launcher_provenance']['interpreter'],'actual installer interpreter drift')
            site.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--no-cache-dir','--target',str(site),p['solver_wheel'],p['consumer_wheel']]
            save(attempt/'install-command.json',command)
            require(subprocess.run(command,check=False).returncode==0,'fresh combined install failed')
            save(attempt/'installed-inventory.json',verify_site(p));result['status']='PASS_FRESH_INSTALL'
        elif phase=='validate-install':
            save(attempt/'installed-inventory.json',installed_graph(p));result['status']='PASS_FRESH_INSTALL_VALIDATION'
        elif phase=='public-smoke':
            result.update(inventory=smoke_inventory(),samples=installed_samples(attempt,p,3,[],smoke=True),status='PASS_PUBLIC_SMOKE')
        else:
            size,ordinals={'low':(2500,[1,2,3,4,5]),'middle':(5000,[1,2,3,4,5]),'full-1':(10000,[1,2]),'full-2':(10000,[3,4]),'full-3':(10000,[5])}[phase]
            result.update(size=size,samples=installed_samples(attempt,p,size,ordinals),status='PASS_SCREEN_SAMPLES')
        save(attempt/'runtime-basic-after.json',basic_runtime())
    except BaseException as exc:
        result.update(status='FAILED',error=str(exc));(attempt/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:save(attempt/'runtime-basic-final.json',basic_runtime())
        except BaseException as exc:
            result.update(status='FAILED',basic_runtime_after_error=str(exc));(attempt/'runtime-basic-final-failure.txt').write_text(traceback.format_exc())
        if phase not in ('controls','install','validate-install'):
            try:
                final_inventory=installed_graph(p);save(attempt/'installed-final-after-close.json',final_inventory)
                require(final_inventory==json.loads((OUT/'attempt-install-01/installed-inventory.json').read_text()),'success/failure-final candidate graph closure changed')
            except BaseException as exc:
                result.update(status='FAILED',closure_after_error=str(exc));(attempt/'closure-after-failure.txt').write_text(traceback.format_exc())
            try:
                helper=module('terminal_series_runtime',ROOT/'scripts/r8_b2_installed_batch_persistence_dev.py')
                save(attempt/'runtime-after.json',public_runtime(helper,p))
            except BaseException as exc:
                result.update(status='FAILED',runtime_after_error=str(exc));(attempt/'runtime-after-failure.txt').write_text(traceback.format_exc())
        save(attempt/'result.json',result)
    return result['status'].startswith('PASS')

def _supervised_run(phase):
    p=json.loads((OUT/'EXECUTABLE_PREPARED.json').read_text());a=execution_authority(p,phase)
    require(authority_contract_sha(a)==p['authority_contract_sha256'] and all(sha(ROOT/n)==h for n,h in p['source'].items()),'authority/source drift')
    ledger_path=OUT/'SERIES_LEDGER.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='OPEN',used_seconds=0.,attempts=[])
    require(ledger['status']=='OPEN' and len(ledger['attempts'])<len(PHASES) and phase==PHASES[len(ledger['attempts'])],'closed/consumed/out-of-order phase')
    require(all(x['passed'] for x in ledger['attempts']),'failed predecessor')
    cap=min(30 if phase in ('controls','install','validate-install','public-smoke') else 600,3200-120-ledger['used_seconds']);require(cap>0,'reserve exhausted')
    attempt=OUT/('attempt-'+phase+'-01');attempt.mkdir(exist_ok=False);temp=attempt/'temp';temp.mkdir();env=dict(os.environ);env.update(ENV);env.update(TEMP=str(temp),TMP=str(temp),PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_CACHE_DIR=str(temp/'pip-cache'),PIP_BUILD_TRACKER=str(temp/'pip-build-tracker'),PYTHONPATH=p['package_site'])
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',phase,str(attempt)]
    save(attempt/'binding.json',dict(command=command,cwd=str(attempt),preparation=p,environment={k:env[k] for k in (*ENV,'TEMP','TMP','PIP_CACHE_DIR','PIP_BUILD_TRACKER','PYTHONPATH')},frozen_grant_sha256=p['frozen_grant_sha256'],execution_authority_sha256=sha(OUT/EXECUTION_AUTHORITY_NAME)))
    for n,h in p['source'].items():
        dest=attempt/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/n).read_bytes());require(sha(dest)==h,'source snapshot')
    sys.path.insert(0,str(ROOT));from scripts._b3_e4_f03_local_scientific_supervisor import supervise,SupervisorLimits
    def launch(*args,**kwargs):
        process=subprocess.Popen(*args,**kwargs);save(attempt/'LIVE_HANDLE.json',dict(pid=process.pid,phase=phase,command=command));return process
    save(ledger_path,ledger)
    with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr:
        receipt=supervise(command,cwd=attempt,environment=env,stdout=stdout,stderr=stderr,popen=launch,limits=SupervisorLimits(cap,2*1024**3,.05))
    save(attempt/'receipt.json',asdict(receipt));r=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {}
    passed=receipt.reason=='completed' and receipt.exit_code==0 and r.get('status','').startswith('PASS')
    ledger['used_seconds']+=receipt.elapsed_seconds;ledger['attempts'].append(dict(phase=phase,passed=passed,receipt=asdict(receipt),ready_sha256=sha(OUT/'EXECUTABLE_PREPARED.json')))
    if not passed or phase=='full-3':ledger.update(status='CLOSED',unused_seconds=3200-ledger['used_seconds'],unused_discarded=True)
    save(ledger_path,ledger);print(json.dumps(dict(phase=phase,passed=passed,status=ledger['status'],used_seconds=ledger['used_seconds'])),flush=True);return passed


# V2 strict installed-provenance and entry gates are defined below.
import configparser
import io
import re
import textwrap
from pathlib import PurePosixPath,PureWindowsPath


def authority_contract_sha(a):
    stable={k:v for k,v in a.items() if k not in ('slot_status','ready_sha256','executable_prepared_sha256','root_preflight_sha256')}
    return hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def authority_gate(a,p,phase,*,require_slot=True):
    require(sha(OUT/'ROOT_DEVELOPMENT_AUTHORITY.json')==p.get('frozen_grant_sha256'),'frozen original grant drift')
    require(authority_contract_sha(a)==p.get('authority_contract_sha256')==authority_contract_sha(json.loads((OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').read_text())),'stable authority contract drift')
    _contract_authority_gate(a,p,phase)
    require(a.get('reuse_install') is False and Path(a['package_site']).resolve()==Path(p['package_site']).resolve()==(OUT/'package/site').resolve(),'only fresh isolated candidate graph')
    if require_slot:require(a.get('slot_status')=='TRANSFERRED','numerical slot not transferred')

def execution_authority(p,phase):
    # Frozen grant is never used as a mutable slot marker or execution substitute.
    require(sha(OUT/'ROOT_DEVELOPMENT_AUTHORITY.json')==p.get('frozen_grant_sha256'),'frozen original grant drift')
    path=OUT/EXECUTION_AUTHORITY_NAME
    require(path.is_file(),'separate execution authority missing')
    a=json.loads(path.read_text());authority_gate(a,p,phase)
    require(a.get('executable_prepared_sha256')==sha(OUT/'EXECUTABLE_PREPARED.json'),'live authority executable identity drift')
    return a

def safe_relative(name):
    require(type(name) is str and name and '\\' not in name and ':' not in name and not PureWindowsPath(name).is_absolute()
        and not PurePosixPath(name).is_absolute() and all(x not in ('','.','..') for x in name.split('/')),'unsafe installed path')
    return name

def declared_launchers(wheel):
    targets={}
    with ZipFile(wheel) as z:
        files=[n for n in z.namelist() if n.endswith('.dist-info/entry_points.txt')]
        require(len(files)<=1,'single entrypoint declaration')
        if not files:return targets
        c=configparser.ConfigParser(interpolation=None,strict=True);c.optionxform=str;c.read_string(z.read(files[0]).decode('utf-8'))
        for group,kind in (('console_scripts','t64'),('gui_scripts','w64')):
            for name,target in c.items(group) if c.has_section(group) else ():
                require(re.fullmatch(r'[A-Za-z0-9_-]+',name) is not None and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*',target) is not None,'safe declared entrypoint')
                path='bin/'+name+'.exe';require(path.casefold() not in {x.casefold() for x in targets},'duplicate declared launcher')
                targets[path]=dict(target=target,kind=kind,record=files[0].rsplit('/',1)[0]+'/RECORD')
    return targets

def launcher_payload(template,target):
    module,func=target.split(':');return (template%dict(module=module,import_name=func.split('.')[0],func=func)).encode('utf-8')

def verify_launcher_bytes(raw,declaration,provenance):
    kind=declaration['kind'];resource=provenance['resources'][kind]
    stub=Path(resource['path']).read_bytes();require(hashlib.sha256(stub).hexdigest()==resource['sha256'],'launcher resource drift')
    require(kind in ('t64','w64'),'declared launcher role required')
    interpreter=provenance['interpreter'];require(interpreter.endswith('python.exe'),'exact recorded console executable required')
    if kind=='w64':
        interpreter=interpreter[:-len('python.exe')]+'pythonw.exe';gui=provenance['gui_executable']
        require(gui['path']==interpreter and Path(gui['path']).is_file() and sha(gui['path'])==gui['sha256'],'declared GUI sibling executable drift')
    prefix=stub+b'#!'+interpreter.encode('utf-8')+b'\n'
    require(raw.startswith(prefix),'launcher stub/interpreter mismatch')
    require(len(raw)>len(prefix)+22 and raw[-22:-18]==b'PK\x05\x06' and raw[-2:]==b'\x00\x00','launcher trailing data/comment')
    with ZipFile(io.BytesIO(raw)) as z:
        entries=z.infolist();require(len(entries)==1 and entries[0].filename=='__main__.py' and entries[0].header_offset==len(prefix)
            and entries[0].compress_type==0 and not entries[0].flag_bits and not z.comment,'launcher ZIP structure')
        require(z.read('__main__.py')==launcher_payload(provenance['template'],declaration['target']),'launcher entrypoint payload mismatch')

def record_target(path,record,launchers):
    if path.startswith('../../bin/'):
        logical='bin/'+path[len('../../bin/'):]
        require(logical in launchers and launchers[logical]['record']==record and path=='../../'+logical,'undeclared installer script alias')
        # Logical alias only. Never resolve the raw dotdot RECORD path on disk.
        return logical
    logical=safe_relative(path)
    if logical in launchers:require(launchers[logical]['record']==record,'wrong launcher RECORD owner')
    return logical

def verify_site(plan):
    site=Path(plan['package_site']).resolve();expected={};launchers={}
    for wheel in (plan['solver_wheel'],plan['consumer_wheel']):
        with ZipFile(wheel) as z:
            for n in z.namelist():
                if n.endswith('/'):continue
                safe_relative(n);require(n.casefold() not in {x.casefold() for x in expected},'duplicate wheel member')
                expected[n]=hashlib.sha256(z.read(n)).hexdigest()
        for n,d in declared_launchers(wheel).items():require(n not in launchers,'overlapping launcher');launchers[n]=d
    records=[n for n in expected if n.endswith('.dist-info/RECORD')]
    extras={n.rsplit('/',1)[0]+'/'+x for n in records for x in ('INSTALLER','REQUESTED','direct_url.json')}
    actual=set()
    for q in site.rglob('*'):
        require(not q.is_symlink() and not getattr(q,'is_junction',lambda:False)(),'installed reparse path refused')
        if q.is_file():actual.add(safe_relative(q.relative_to(site).as_posix()))
    require(actual==set(expected)|extras|set(launchers),'combined installed file inventory')
    require(all(sha(site/n)==h for n,h in expected.items() if n not in records),'combined wheel bytes mismatch')
    for n,d in launchers.items():verify_launcher_bytes((site/n).read_bytes(),d,plan['launcher_provenance'])
    seen=set()
    for n in records:
        for row in csv.reader((site/n).read_text().splitlines()):
            require(len(row)==3,'RECORD row shape');path,digest,size=row;logical=record_target(path,n,launchers)
            require(logical in actual and logical not in seen,'installed RECORD path/duplicate');seen.add(logical)
            if logical==n:require(digest==size=='','RECORD self entry')
            else:
                data=(site/logical).read_bytes();h='sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                require(digest==h and size==str(len(data)),'installed RECORD digest/size')
    require(seen==actual,'installed RECORD completeness')
    return dict(files={n:sha(site/n) for n in sorted(actual)},launchers=launchers,reinstalled=False,
        interpretation='only exact declared installer logical aliases; no RECORD filesystem traversal')

def installed_graph(plan):
    require(plan['installation']==dict(reuse_install=False,bytecode_disabled=True),'fresh no-cache graph contract')
    receipt=json.loads((OUT/'attempt-install-01/receipt.json').read_text())
    result=json.loads((OUT/'attempt-install-01/result.json').read_text())
    require(receipt['exit_code']==0 and receipt['reason']=='completed' and result['status']=='PASS_FRESH_INSTALL','actual fresh installation PASS required')
    observed=verify_site(plan)
    require(observed==json.loads((OUT/'attempt-install-01/installed-inventory.json').read_text()),'exact installed candidate namespace/hash drift')
    return observed


def public_caller_contract(plan,runner_source=None):
    """Fixed production calls checked against actual frozen wheel signatures.

    This is static admission, not caller execution or a replacement for smoke.
    """
    with ZipFile(plan['solver_wheel']) as z:
        solver=ast.parse(z.read('anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py').decode())
    with ZipFile(plan['consumer_wheel']) as z:
        consumer=ast.parse(z.read('anyfem/solve/stiffened_shell_b2.py').decode())
        project_io=ast.parse(z.read('anyfem/io/project_file.py').decode())
        project_routes=ast.parse(z.read('anyfem/solve/stiffened_shell_project.py').decode())
    def method(tree,cls,name):
        c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    native='ExperimentalB2Q4PanelAnalysis';panel='ExperimentalB2PhysicalRigidFootprintPanel'
    methods={('owner','solve_linear_many'):method(solver,native,'solve_linear_many'),('owner','close'):method(solver,native,'close'),
        ('reopened','create_analysis'):method(consumer,panel,'create_analysis'),('reopened','definition'):method(consumer,panel,'definition'),
        ('ExperimentalB2PhysicalRigidFootprintPanel','from_definition'):method(consumer,panel,'from_definition')}
    globals_={'save_case':function(consumer,'save_case'),'load_case':function(consumer,'load_case'),
        'save_project':function(project_io,'save_project'),'load_project':function(project_io,'load_project'),
        'point_panel_from_analysis':function(project_routes,'point_panel_from_analysis')}
    tree=ast.parse(runner_source if runner_source is not None else Path(__file__).read_text())
    actual=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='installed_samples');seen=[]
    for call in (n for n in ast.walk(actual) if isinstance(n,ast.Call)):
        target=None;bound=False;label=None
        if isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name):
            key=(call.func.value.id,call.func.attr)
            if key[0]=='owner':require(key in methods,'unknown actual owner public method')
            if key in methods:target=methods[key];bound=True;label='.'.join(key)
        elif isinstance(call.func,ast.Name) and call.func.id in globals_:
            target=globals_[call.func.id];label=call.func.id
        if target is None:continue
        positional=[x.arg for x in (*target.args.posonlyargs,*target.args.args)][int(bound):]
        keywords={x.arg for x in target.args.kwonlyargs}|set(positional)
        require(not any(k.arg is None or k.arg not in keywords for k in call.keywords),'actual caller unsupported keyword '+label)
        require(not any(isinstance(x,ast.Starred) for x in call.args) and len(call.args)<=len(positional),'actual caller positional mismatch '+label)
        required=len(positional)-len(target.args.defaults)
        supplied=set(positional[:len(call.args)])|{k.arg for k in call.keywords}
        require(all(n in supplied for n in positional[:required]),'actual caller missing argument '+label)
        require(all(n.arg in supplied for n,d in zip(target.args.kwonlyargs,target.args.kw_defaults) if d is None),'actual caller missing keyword '+label)
        seen.append(label)
    require(all(x in seen for x in ('owner.solve_linear_many','owner.close','reopened.create_analysis','reopened.definition','ExperimentalB2PhysicalRigidFootprintPanel.from_definition',*globals_)),'complete connected public path missing')
    return dict(status='STATIC_FROZEN_WHEEL_CALLER_SIGNATURES',calls=seen,solver_sha256=sha(plan['solver_wheel']),consumer_sha256=sha(plan['consumer_wheel']),not_execution_evidence=True)


def wheel_package(path):
    with ZipFile(path) as z:
        return {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}


def ready():
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'SERIES_LEDGER.json').exists(),'consumed preparation is immutable')
    p=json.loads((PREVIOUS/'EXECUTABLE_PREPARED.json').read_text())
    artifact=json.loads((LIFETIME/'ARTIFACT.json').read_text())
    require(sha(artifact['wheel'])==artifact['sha256']==SOLVER_SHA,'verified repaired candidate wheel')
    p.update(solver_wheel=artifact['wheel'],solver_sha256=SOLVER_SHA,solver_package=artifact['package'],
        package_site=str(OUT/'package/site'),installation=dict(reuse_install=False,bytecode_disabled=True),
        expected_controls=21,public_smoke=smoke_inventory(),public_smoke_config=SMOKE_CONFIG)
    p['package']=dict(artifact['package']);p['package'].update(wheel_package(p['consumer_wheel']))
    for k in ('reused_install_receipt_sha256','historical_install_receipt_sha256','authority_sha256','frozen_grant_sha256','authority_contract_sha256'):p.pop(k,None)
    p['historical_v6']={n:sha(PREVIOUS/n) for n in ('EXECUTABLE_PREPARED.json','EXECUTION_SELECTED_PATHS.json','SERIES_LEDGER.json','EXECUTION_HANDOFF.json','EXECUTION_TERMINAL_TREE.json')}
    p['historical_v8']={n:sha(BASE/'execution-v8'/n) for n in ('EXECUTION_SELECTED_PATHS.json','EXECUTION_HANDOFF.json','SERIES_LEDGER.json','EXECUTION_TERMINAL_TREE.json','INSTALL_DIAGNOSIS.json')}
    p['historical_v7']={n:sha(BASE/'execution-v7'/n) for n in ('EXECUTION_SELECTED_PATHS.json','EXECUTION_HANDOFF.json','SERIES_LEDGER.json','EXECUTION_TERMINAL_TREE.json','PRELAUNCH_SEQUENCE_NOTE.json')}
    p['verified_lifetime']={n:sha(LIFETIME/n) for n in ('SELECTED_PATHS.json','HANDOFF.json','ledger.json','ARTIFACT.json','ROOT_ADOPTION.json','ROOT_ORIGINAL_SOURCE_RECONCILIATION.json','REPORT_COUNT_CORRECTION.md')}
    p['source'].update({n:sha(ROOT/n) for n in ('scripts/r8_b2_assembly_series_v9_dev.py','tests/test_r8_b2_assembly_series_v9_dev.py','scripts/r8_b2_assembly_series_v9_preflight_dev.py',
        'src/anyfesolver/e4_pl_element.py','src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py','src/anyfesolver/_stiffened_shell_b2_assembly_diagnostics.py')})
    p['question']='Does real private-owner retirement allow unchanged connected cold+fivewarm2500/5000/10000 under2GiB? Fresh-constructor/backend peak may still fail.'
    p['caller_contract']=public_caller_contract(p)
    p['predecessor_prepared_sha256']=sha(BASE/'PREPARED.json')
    p['execution_authority_name']=EXECUTION_AUTHORITY_NAME
    p['proposed_budget']=dict(cumulative_seconds=3200,reserve_seconds=120,process_cap_seconds=600,initial_phases_seconds=30,tree_rss_limit_bytes=2*1024**3,cpu_affinity=[0],numerical_threads=1,jit_disabled=True,python_optimize=0,one_numerical_job=True)
    p.update(status='INPUT_PREPARATION_AUTHORITY_PENDING',execution_authorized=False,numerical_runs=False)
    save(OUT/'INPUT_PREPARED.json',p)
    grant=OUT/'ROOT_DEVELOPMENT_AUTHORITY.json'
    if not grant.is_file():return p
    a=json.loads(grant.read_text());p.update(status='EXECUTABLE_DEVELOPMENT_SCREEN',frozen_grant_sha256=sha(grant),authority_sha256=sha(grant),authority_contract_sha256=authority_contract_sha(a))
    authority_gate(a,p,'controls',require_slot=False)
    require(all(sha(ROOT/n)==h for n,h in p['source'].items()),'source closure')
    immutable_inputs(p)
    save(OUT/'EXECUTABLE_PREPARED.json',p)
    selection_paths=[OUT/'INPUT_PREPARED.json',OUT/'EXECUTABLE_PREPARED.json',OUT/'READINESS.md',OUT/'GRANT_REQUEST.json',ROOT/'scripts/r8_b2_assembly_series_v9_dev.py',ROOT/'scripts/r8_b2_assembly_series_v9_preflight_dev.py',ROOT/'tests/test_r8_b2_assembly_series_v9_dev.py']
    require(all(q.is_file() for q in selection_paths),'connected readiness artifacts missing')
    save(OUT/'SELECTED_PATHS.json',dict(status='UNCONSUMED_EXECUTABLE_PREPARATION',prepared_sha256=sha(OUT/'EXECUTABLE_PREPARED.json'),files={q.relative_to(ROOT).as_posix():dict(bytes=q.stat().st_size,sha256=sha(q)) for q in selection_paths}))
    return p


def child(phase,attempt):
    binding=json.loads((attempt/'binding.json').read_text());p=binding['preparation'];a=execution_authority(p,phase)
    require(binding['frozen_grant_sha256']==p['frozen_grant_sha256'] and binding['execution_authority_sha256']==sha(OUT/EXECUTION_AUTHORITY_NAME),'child original/live authority binding drift')
    require(all(sha(ROOT/n)==h for n,h in p['source'].items()),'child source drift')
    ledger=json.loads((OUT/'SERIES_LEDGER.json').read_text());require(ledger['status']=='OPEN' and phase==PHASES[len(ledger['attempts'])],'child closed/consumed/out-of-order phase')
    immutable_inputs(p)
    public_caller_contract(p)
    return _execute_child_body(phase,attempt)

def immutable_inputs(p):
    require(sha(p['solver_wheel'])==p['solver_sha256']==SOLVER_SHA and sha(p['consumer_wheel'])==p['consumer_sha256'] and sha(FULL)==FULL_SHA,'child wheel/full input drift')
    proof=p['launcher_provenance'];require(sha(proof['template_source_path'])==proof['template_source_sha256'],'PipScriptMaker source drift')
    require(all(sha(x['path'])==x['sha256'] for x in proof['resources'].values()),'launcher PE resource drift')
    require(Path(proof['gui_executable']['path']).is_file() and sha(proof['gui_executable']['path'])==proof['gui_executable']['sha256'],'GUI sibling source drift')
    require(sha(proof['alternate_source']['path'])==proof['alternate_source']['sha256'],'GUI alternate generator source drift')

def run(phase):
    p=json.loads((OUT/'EXECUTABLE_PREPARED.json').read_text());execution_authority(p,phase) # Missing/untransferred live authority refuses before any ledger.
    immutable_inputs(p)
    public_caller_contract(p)
    return _supervised_run(phase)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ready',action='store_true');parser.add_argument('--run',choices=PHASES);parser.add_argument('--child',nargs=2);args=parser.parse_args()
    if args.ready:ready()
    elif args.child:raise SystemExit(0 if child(args.child[0],Path(args.child[1])) else 1)
    elif args.run:raise SystemExit(0 if run(args.run) else 1)
    else:parser.error('explicit readiness or transferred-slot phase required')
