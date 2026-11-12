"""One installed CURRENT PUBLIC B2 case1 gate, explicit adopted authority only.

No build/install/backend selection or historical campaign replay. Preparation
uses stdlib only; an installed child imports product modules from the supplied
isolated site and preserves first-failure evidence without numerical retries.
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
import sys
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/agent_plans/development/r8-b2-installed-original-target-20261010'
FROZEN=ROOT/'docs/agent_plans/development/r8-b2-large-work-allocation-20261009'
BUILDER=FROZEN/'frozen_model_builder.py'
EVALUATOR=ROOT/'scripts/r8_b2_physical_work_unblock_dev.py'
ENV={name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS',
    'NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT')}
ENV['PYTHONOPTIMIZE']='0'
B2_FILES=('__init__.py','stiffened_shell.py','_stiffened_shell_b2_q4_panel_experimental.py',
    '_stiffened_shell_b2_affine_resolution.py','_b2_physical_linear_batch_dev.py',
    '_b2_attachment_elimination_dev.py','_stiffened_shell_b2_physical_rigid_footprint.py',
    '_b2_physical_linear_definition.py','_b2_physical_flexibility_linear.py',
    '_b2_loaded_point_trace_dev.py','linalg.py','threading_policy.py','elements.py',
    'e4_pl_element.py','fe_core.py','control.py')
EXPECTED_MATRICES=dict(K='129ae611f49f2f85659b69638e69d6802ba3b9e669a6e68b6028835ab9dec708',
 C='ac1d7285d0abd0608ef3250669e3c24bf6161ac1e4ad5bc85a7c8480685e8808',
 T='6788e53a82f53b811fa8ab7eace3e4b71e5bad659baf30dfef964d990923f643')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,
        default=str)+'\n',encoding='utf-8')


def package_identity(site,wheel):
    site=Path(site).resolve();wheel=Path(wheel).resolve()
    require(site.is_dir() and wheel.is_file(),'existing isolated wheel/site required; no build')
    package={}
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('anyfesolver/') and not name.endswith('/'):
                require('..' not in Path(name).parts,'wheel path refusal')
                expected=hashlib.sha256(archive.read(name)).hexdigest()
                require(sha(site/name)==expected,'wheel/site mismatch '+name)
                package[name]=expected
    require(package and all('anyfesolver/'+name in package for name in B2_FILES),'package closure missing')
    for name in B2_FILES:
        require(sha(ROOT/'src/anyfesolver'/name)==package['anyfesolver/'+name],
            'current B2 source/site mismatch '+name)
    return dict(site=str(site),wheel=str(wheel),wheel_sha256=sha(wheel),package=package,
        relevant_current_sources={name:sha(ROOT/'src/anyfesolver'/name) for name in B2_FILES})


def preparation(site,wheel):
    previous=json.loads((FROZEN/'preflight.json').read_text())
    require(sha(BUILDER)==previous['builder_sha256'],'frozen model builder drift')
    require(previous['load_coordinate']==30902 and previous['selected_dofs']==[30902,30900]
        and previous['selected_pad']=='pad-0-5' and previous['selected_member']=='b2-0-0'
        and previous['stations']==[0,.5,1],'frozen case1 input drift')
    ast.parse(Path(__file__).read_text())
    artifact=package_identity(site,wheel)
    source={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),BUILDER,EVALUATOR,
        ROOT/'scripts/_b3_e4_f03_local_scientific_supervisor.py',
        ROOT/'scripts/_b3_e4_f03_local_scientific_cases.py')}
    return dict(status='PREPARED_SLOT_HELD',artifact=artifact,source=source,
        case=dict(nx=100,ny=100,line_rows=list(range(4,100,10)),patch_x=list(range(5,100,10)),
            definition_config_cases=100,submitted_cases=1,case_id='case-1',load_coordinate=30902,
            load_value=-1.,selected_dofs=[30902,30900],selected_pad='pad-0-5',
            selected_member='b2-0-0',stations=[0.,.5,1.]),
        matrix_sha256=EXPECTED_MATRICES,public_dispatch='stiffened_shell.create_analysis(definition=...)',
        factor_policy='unchanged current eligible attachment GENERAL/default; no override',
        limits=dict(allowance_seconds=360.,reserve_seconds=60.,process_seconds=120.,
            rss_bytes=2*1024**3,cpu=[0],threads=1,jit_disabled=True,optimize=0),
        scientific=dict(original_relative_work=1e-8,product_abs_work=1e-12,
            product_relative_work=1e-8,equilibrium_relative=1e-9,gap_relative=1e-9),
        qualification=False,full100_cases=False,scale=False)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def child(attempt):
    bound=json.loads((attempt/'binding.json').read_text());p=bound['preparation'];artifact=p['artifact']
    require(package_identity(artifact['site'],artifact['wheel'])==artifact,'artifact/source drift before import')
    for relative,expected in p['source'].items():require(sha(ROOT/relative)==expected,'development source drift')
    forbidden=(ROOT/'src',Path('C:/Github/ANYfeSolver/src'))
    sys.path[:]=[artifact['site']]+[entry for entry in sys.path
        if Path(entry or '.').resolve()!=Path(artifact['site']).resolve()
        and not any(Path(entry or '.').resolve().is_relative_to(base) for base in forbidden)]
    import psutil
    psutil.Process().cpu_affinity([0])
    import numpy as np
    import scipy
    import numba
    from threadpoolctl import threadpool_info
    import anyfesolver
    from anyfesolver import stiffened_shell as api
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as panel
    from anyfesolver.linalg import backend_operation_counts
    site=Path(artifact['site']).resolve()

    def runtime():
        imports={}
        for name,module in tuple(sys.modules.items()):
            if name=='anyfesolver' or name.startswith('anyfesolver.'):
                file=getattr(module,'__file__',None)
                if file:
                    file=Path(file).resolve();require(file.is_relative_to(site),'source product import '+name)
                    relative=file.relative_to(site).as_posix()
                    require(relative in artifact['package'] and sha(file)==artifact['package'][relative],
                        'actual product import identity '+name)
                    imports[name]={'path':str(file),'sha256':sha(file)}
        result=dict(affinity=psutil.Process().cpu_affinity(),pools=threadpool_info(),
            environment={k:os.environ.get(k) for k in ENV},optimize=sys.flags.optimize,
            numba=dict(version=numba.__version__,disable_jit=int(numba.config.DISABLE_JIT),threads=numba.get_num_threads()),
            numpy=np.__version__,scipy=scipy.__version__,python=sys.version,executable=sys.executable,imports=imports)
        require(result['affinity']==[0] and result['environment']==ENV and result['optimize']==0
            and result['numba']['disable_jit']==1 and result['numba']['threads']==1
            and result['pools'] and all(pool['num_threads']==1 for pool in result['pools']),
            'effective runtime refusal')
        return result

    save(attempt/'runtime-before.json',runtime())
    record=dict(status='UNRESOLVED',stage='model',public_gate_pass=False,qualification=False,
        saved_direction_prediction_applies=False)
    owner=None;full_results=[];call_frames={};correct_frames=[];solves=[];certification_events=[]
    started=time.perf_counter()
    try:
        builder=load_module('frozen_b2_public_case_builder',BUILDER)
        config=builder.settings('large',100)
        require(config['nx']==100 and config['ny']==100 and config['cases']==100,'frozen config')
        definition,inventory,shell_id=builder.model(config)
        require(inventory['q4_shells']==10000 and inventory['physical_b2_members']==1000
            and inventory['attached_patches']==100 and inventory['dofs']==67266
            and inventory['free_dofs']==66660 and inventory['fixed_dofs']==606,'physical inventory')
        operations_before=backend_operation_counts();generated=time.perf_counter()
        record.update(stage='public_compile',inventory=inventory,
            generation_wall_seconds=generated-started,backend_operations_before=operations_before)
        save(attempt/'result.json',record)
        owner=api.create_analysis(definition=definition)
        compiled=time.perf_counter()
        record.update(stage='compiled',compile_wall_seconds=compiled-generated,
            owner_type=type(owner).__name__,factor=owner._handle.diagnostics(),
            compiler_attachment=dict(owner._attachment_elimination_info))
        record['backend_operations_after_compile']=backend_operation_counts()
        changes={k:record['backend_operations_after_compile'][k]-operations_before[k] for k in operations_before}
        record['backend_operation_delta_compile']=changes
        require(changes['scipy_factorization_attempts']+changes['pardiso_factorization_attempts']==1
            and changes['pardiso_stale_rebuilds']==0,'extra/fallback factor attempt; stop before primary')
        for name in ('K','C','T'):
            matrix=getattr(owner,'_'+name).tocsr();digest=hashlib.sha256()
            for v,dtype in ((matrix.shape,'<i8'),(matrix.indptr,'<i8'),(matrix.indices,'<i8'),(matrix.data,'<f8')):
                digest.update(np.asarray(v,dtype=dtype).tobytes())
            record[name]=dict(shape=matrix.shape,nnz=matrix.nnz,sha256=digest.hexdigest())
            require(digest.hexdigest()==p['matrix_sha256'][name],name+' original compiler identity mismatch')
        require(owner._handle.factorization_count==1 and owner._handle.solve_count==0,'one compiled unused factor')
        require(len(owner._footprint_incidence)==400 and owner._C.shape[0]==2000,'actual physical rows')
        require(np.all(owner._u0==0.),'original homogeneous offset')
        save(attempt/'result.json',record)
        load=np.zeros(67266);load[30902]=-1.
        require(6*(shell_id(100,50)-1)+2==30902,'load reference mapping')
        load_hash=hashlib.sha256(load.tobytes()).hexdigest()
        factor_identity=id(owner._handle);original_solve=owner._handle.solve

        def captured_solve(rhs):
            ordinal=len(solves)+1;require(ordinal<=2,'extra primary/correction solve refusal')
            before=owner._handle.solve_count
            item=dict(ordinal=ordinal,before=before,role='primary' if ordinal==1 else 'original_target_correction')
            solves.append(item)
            np.savez(attempt/f'solve-{ordinal}-rhs.npz',rhs=np.asarray(rhs))
            answer=original_solve(rhs)
            np.savez(attempt/f'solve-{ordinal}-answer.npz',answer=np.asarray(answer))
            item.update(after=owner._handle.solve_count,same_handle=id(owner._handle)==factor_identity)
            save(attempt/'solve-events.json',solves)
            return answer

        owner._handle.solve=captured_solve
        solve_code=owner.solve_linear.__func__.__code__
        correction_code=owner._correct_physical_equilibrium.__func__.__code__
        certification_code=owner._explicit_load_resolved.__func__.__code__
        array_names=('answer','candidate','q','lam','u','gap','force','constraint_force','reaction',
            'projected','gap_scale','physical_force_scale','equilibrium_scale','equilibrium_tolerance',
            'gap_tolerance','explicit_rhs','resolution_floor','applied','rhs','scaled_rhs',
            'scaled_answer','original_target','correction_rhs','delta')

        def observe(frame,event,value):
            if event!='return':return
            if frame.f_code==certification_code:
                certification_events.append(dict(returned=value is not None,result=bool(value),
                    caller=frame.f_back.f_code.co_name if frame.f_back else None))
                return
            if frame.f_code not in (solve_code,correction_code):return
            locals_=frame.f_locals
            arrays={name:np.asarray(locals_[name]).copy() for name in array_names
                if name in locals_ and isinstance(locals_[name],np.ndarray)}
            if frame.f_code==correction_code:
                if isinstance(value,np.ndarray):arrays['returned_answer']=value.copy()
                index=len(correct_frames)+1
                np.savez(attempt/f'correction-frame-{index}.npz',**arrays)
                correct_frames.append({name:(float(locals_[name]) if math.isfinite(float(locals_[name]))
                    else repr(locals_[name])) for name in ('original_ratio','original_work_ratio','original_score','candidate_score')
                    if name in locals_ and isinstance(locals_[name],(int,float))})
            else:
                call_frames.update(arrays)
                np.savez(attempt/'actual-public-caller-frame.npz',**arrays)
                flags={name:(repr(locals_[name]) if isinstance(locals_[name],float)
                    and not math.isfinite(locals_[name]) else locals_[name])
                    for name in ('certified','roundoff_fraction','energy') if name in locals_}
                if 'certified' in flags:flags['certified']=bool(flags['certified'])
                record['actual_caller_flags']=flags
                section=locals_.get('result')
                if hasattr(section,'strains'):
                    np.savez(attempt/'actual-selected-member-material.npz',strains=section.strains,
                        resultants=section.resultants,local_coordinates=locals_.get('local'))
                full_results.append(value)

        sys.setprofile(observe)
        try:
            batch=owner.solve_linear_many((('case-1',load),),selected_dofs=(30902,30900),
                selected_footprints=api.B2PhysicalRigidFootprintSelection(('pad-0-5',)),
                recover_members={'b2-0-0':(0.,.5,1.)})
        finally:
            sys.setprofile(None);owner._handle.solve=original_solve
        solved=time.perf_counter()
        record.update(stage='public_return',solve_wall_seconds=solved-compiled,
            batch_status=batch.status,batch_failure=batch.failure_reason,
            case_count=len(batch.cases),factor=owner._handle.diagnostics(),
            solve_events=solves,correction_frames=correct_frames,
            explicit_load_certification_events=certification_events,
            backend_operations_after_solve=backend_operation_counts(),
            input_load_unchanged=hashlib.sha256(load.tobytes()).hexdigest()==load_hash,
            full_result_count=len(full_results))
        # Preserve actual returned candidate and all observed guards BEFORE
        # assertions. Failure has no subsequent reconstructed/scored replay.
        if len(full_results)==1 and full_results[0] is not None:
            returned=full_results[0]
            record['full_returned_status']=returned.status
            record['full_returned_failure']=returned.failure_reason
            np.savez(attempt/'public-full-state.npz',u=np.asarray(returned.displacements),
                q=call_frames.get('q',np.empty(0)),lambda_=np.asarray(getattr(returned,'constraint_multipliers',())),
                gap=np.asarray(getattr(returned,'constraint_gaps',())),
                reaction=np.asarray(returned.affine_reactions),load=load,
                projected=call_frames.get('projected',np.empty(0)))
            record['returned_recovery']=dict(beam_resultants=returned.beam_resultants_local,
                footprint_wrenches=[asdict(v) for v in getattr(returned,'footprint_wrenches',())],
                internal_strain_energy=returned.internal_strain_energy)
        save(attempt/'result.json',record)
        require(batch.status=='completed' and len(batch.cases)==1 and len(full_results)==1,
            'first public caller failure: '+str(batch.failure_reason))
        full=full_results[0];require(full.status=='completed','full returned failure '+str(full.failure_reason))
        require(full.production_qualified is False and batch.production_qualified is False,'unexpected qualification')
        require(record['input_load_unchanged'] and all(s['same_handle'] for s in solves),'mutation/factor reuse')
        require(batch.factorization_count==1 and batch.solve_count==len(solves) and len(solves) in (1,2),'counter mismatch')

        def ratio(residual,tolerance):
            residual=np.abs(residual);tolerance=np.asarray(tolerance)
            values=np.full(residual.shape,np.inf)
            np.divide(residual,tolerance,out=values,where=tolerance>0.)
            values[(tolerance==0.)&(residual==0.)]=0.
            return float(np.max(values))

        require(all(name in call_frames for name in ('u','q','lam','gap','reaction','projected',
            'equilibrium_tolerance','gap_tolerance','explicit_rhs','resolution_floor','gap_scale')),'actual guard frame missing')
        active=call_frames['explicit_rhs']>0.
        resolution_required=bool(np.any(call_frames['resolution_floor'][active]>
            1e-9*call_frames['explicit_rhs'][active]))
        actual_certified=record['actual_caller_flags'].get('certified') is True
        resolution_pass=(not resolution_required) or (actual_certified and any(e['result'] for e in certification_events))
        gap_certificate_required=bool(np.any(np.abs(call_frames['gap'])>1e-9*call_frames['gap_scale']))
        gap_certificate_pass=(not gap_certificate_required) or any(e['result'] for e in certification_events)
        record.update(explicit_load_resolution_required=resolution_required,
            explicit_load_resolution_pass=resolution_pass,gap_certificate_required=gap_certificate_required,
            gap_certificate_pass=gap_certificate_pass)
        require(resolution_pass and gap_certificate_pass,'actual explicit-load certification predicate')
        eq=ratio(call_frames['projected'],call_frames['equilibrium_tolerance'])
        gap=ratio(call_frames['gap'],call_frames['gap_tolerance'])
        u=np.asarray(full.displacements);lam=np.asarray(full.constraint_multipliers)
        require(u.shape==(67266,) and lam.shape==(2000,) and len(full.constraint_rows)==2000,
            'complete physical/constraint recovery')
        require(eq<=1. and gap<=1. and np.isfinite(full.affine_reactions).all(),'original equilibrium/gap/reactions')
        require(len(full.footprint_wrenches)==1 and len(full.beam_resultants_local)==1
            and len(batch.member_resultant_envelopes)==18,'requested public recovery inventory')
        for wrench in full.footprint_wrenches:
            require(wrench.footprint_id=='pad-0-5' and not wrench.production_qualified,
                'selected physical wrench identity')
            for name in ('common_reference_origin_m','shell_force_N','shell_moment_Nm','beam_force_N','beam_moment_Nm'):
                values=np.asarray(getattr(wrench,name))
                require(values.shape==(3,) and np.isfinite(values).all(),'finite actual footprint '+name)
            require(len(wrench.node_actions)==4,'actual selected node-action inventory')
            for action in wrench.node_actions:
                for name in ('shell_force_N','shell_direct_couple_Nm','beam_force_N','beam_direct_couple_Nm','gap'):
                    require(np.isfinite(getattr(action,name)).all(),'finite actual footprint node '+name)
        for member_id,resultants in full.beam_resultants_local:
            require(member_id=='b2-0-0' and np.asarray(resultants).shape==(3,6)
                and np.isfinite(resultants).all(),'finite actual selected member resultants')
        require(np.isfinite(batch.cases[0].selected_affine_reactions).all()
            and np.isfinite(batch.cases[0].selected_displacements).all(),'finite selected public reactions/displacements')
        material_path=attempt/'actual-selected-member-material.npz'
        require(material_path.exists(),'actual section strains missing')
        with np.load(material_path,allow_pickle=False) as material:
            require(material['strains'].shape==(3,6) and np.isfinite(material['strains']).all()
                and np.isfinite(material['resultants']).all(),'material recovery fields')
        internal=2.*full.internal_strain_energy;external=float(u@load);defect=internal-external
        relative=1e-8*abs(external);product=1e-12+relative
        require(math.isfinite(internal) and math.isfinite(external) and abs(defect)<=relative,
            'unchanged original relative-work criterion')
        require(abs(defect)<=product,'product work criterion')
        # Reuse the frozen tested stored-work evidence evaluator. This absolute
        # file import supplies development arithmetic only, never product code.
        evaluator=load_module('frozen_b2_work_evidence',EVALUATOR)
        exact=evaluator.ExactWork();eu=exact.vector(u);ef=exact.vector(load)
        W=exact.action(eu,owner._K.tocsr(),eu)-exact.dot(eu,ef)
        scorer_error=evaluator.Fraction.from_float(defect)-W
        sufficient=abs(W)+abs(scorer_error)<=evaluator.Fraction.from_float(1e-8)*abs(exact.dot(eu,ef))
        require(sufficient,'exact stored-work/scorer original-relative bound unresolved')
        record.update(stage='all_assigned_public_predicates',status='PASSED_INSTALLED_CASE1',public_gate_pass=True,
            equilibrium_max_row_ratio=eq,gap_max_row_ratio=gap,internal_work=internal,external_work=external,
            signed_work_defect=defect,original_relative_tolerance=relative,product_work_tolerance=product,
            exact_stored_work=evaluator.rational_record(W),exact_scorer_deviation=evaluator.rational_record(scorer_error),
            exact_work_operations=exact.operations,exact_stored_relative_bound_pass=sufficient,
            selected_case=asdict(batch.cases[0]),section_resultant_envelopes=[asdict(v) for v in batch.member_resultant_envelopes],
            footprint_wrenches=[asdict(v) for v in full.footprint_wrenches],
            fullscale_qualification=False,physical_assembly_uncertainty='not certified by stored-work evaluator')
    except Exception as error:
        import traceback
        record.update(status='FAILED_OR_UNRESOLVED',public_gate_pass=False,error_type=type(error).__name__,error=str(error))
        (attempt/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        sys.setprofile(None)
        if owner is not None:
            record['terminal_factor']=owner._handle.diagnostics();owner.close()
        save(attempt/'result.json',record)
        save(attempt/'runtime-after.json',runtime())
    print(json.dumps({'status':record['status'],'stage':record['stage'],'public_gate_pass':record['public_gate_pass']}))
    return 0


def run(authority_path):
    authority=json.loads(Path(authority_path).read_text())
    require(authority['status']=='AUTHORIZED' and authority['numerical_slot_released'] is True,
        'explicit root authority/slot required')
    p=json.loads((OUT/'PREPARED_GATE.json').read_text())
    require(sha(OUT/'PREPARED_GATE.json')==authority['prepared_sha256'],'prepared authority mismatch')
    require(package_identity(p['artifact']['site'],p['artifact']['wheel'])==p['artifact'],'prelaunch artifact drift')
    for relative,expected in p['source'].items():require(sha(ROOT/relative)==expected,'prelaunch source drift')
    require(not (OUT/'ledger.json').exists(),'one primary attempt only; ledger already exists')
    sys.path.insert(0,str(ROOT))
    from scripts._b3_e4_f03_local_scientific_supervisor import supervise,SupervisorLimits
    attempt=OUT/'attempt-installed-public-01';attempt.mkdir(exist_ok=False)
    env=dict(os.environ);env.update(ENV);env['PYTHONPATH']=p['artifact']['site']
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',str(attempt)]
    bound=dict(command=command,cwd=str(attempt),preparation=p,authority_sha256=sha(authority_path),
        environment={k:env[k] for k in (*ENV,'PYTHONPATH')})
    save(attempt/'binding.json',bound)
    for relative,expected in p['source'].items():
        target=attempt/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes());require(sha(target)==expected,'snapshot drift')
    save(OUT/'ledger.json',dict(status='OPEN',allowance_seconds=360.,reserve_seconds=60.,attempts=1,used_seconds=0.))
    with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr:
        receipt=supervise(command,cwd=attempt,environment=env,stdout=stdout,stderr=stderr,
            limits=SupervisorLimits(120.,2*1024**3,.05))
    save(attempt/'process-tree-receipt.json',asdict(receipt))
    ledger=dict(status='CLOSED',allowance_seconds=360.,reserve_seconds=60.,attempts=1,
        used_seconds=receipt.elapsed_seconds,discarded_unused_seconds=360.-receipt.elapsed_seconds,
        reason=receipt.reason,exit_code=receipt.exit_code)
    save(OUT/'ledger.json',ledger);print(json.dumps(ledger))
    return 0 if receipt.reason=='completed' and receipt.exit_code==0 else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--site',type=Path);parser.add_argument('--wheel',type=Path)
    parser.add_argument('--run',type=Path);parser.add_argument('--child',type=Path)
    args=parser.parse_args()
    if args.prepare:
        require(args.site is not None and args.wheel is not None,'existing site and wheel needed')
        save(OUT/'PREPARED_GATE.json',preparation(args.site,args.wheel))
        print(json.dumps({'status':'PREPARED_SLOT_HELD','prepared_sha256':sha(OUT/'PREPARED_GATE.json')}))
    elif args.child:sys.exit(child(args.child))
    elif args.run:sys.exit(run(args.run))
    else:parser.error('explicit preparation or adopted supervised run required')
