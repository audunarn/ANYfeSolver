"""Source-only B2 G5 preparation, plus unexecuted measured-route components.

Preparation reads bound source/metadata with stdlib only. Execution needs NEW
root authority and a transferred numerical slot; no historical ledger is reused.
Warm cached-RHS measurement is distinct from fresh-owner assembly/factor cost;
diagnostic tracing never establishes an unprofiled assembly acceptance median.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/agent_plans/development/r8-b2-g5-readiness-20261010'
MODELS=ROOT/'scripts/b2_large_scale_development.py'
FUNCTIONAL=ROOT/'docs/agent_plans/development/r8-b2-installed-batch-persistence-20261010'
PANEL=ROOT/'src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py'
CONTRACT=ROOT/'docs/agent_plans/B3_GE_E4_PL_B2_COMPLETE_PROGRAMME.md'
CONTROL_FILENAME='test_r8_b2_g5_readiness_dev.py'
EXPECTED_CONTROLS=8


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def source_settings(profile,cases=100):
    """Evaluate ONLY the existing stdlib settings function, never import model."""
    require(profile in ('small','medium','scale','large','beam_rich'),'declared existing profile required')
    tree=ast.parse(MODELS.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='settings')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(MODELS),'exec'),namespace)
    return namespace['settings'](profile,cases)


def inventory(config):
    nx,ny=config['nx'],config['ny'];lines=len(config['line_rows'])
    segments=nx*config.get('beam_segments_per_shell',1)
    pads=lines*len(config['patch_x']);shell_nodes=(nx+1)*(ny+1);beam_nodes=lines*(segments+1)
    return dict(q4_shells=nx*ny,physical_b2_members=lines*segments,pads=pads,
        incidences=4*pads,physical_rows=20*pads,dofs=6*(shell_nodes+beam_nodes),fixed_dofs=6*(ny+1),
        free_dofs=6*(shell_nodes+beam_nodes)-6*(ny+1),cases=config['cases'])


def selected_input(config):
    node=1+(config['ny']//2)*(config['nx']+1)+config['nx']
    slot=6*(node-1)+2
    return dict(loaded_node=node,loaded_dof=slot,selected_dofs=[slot,slot-2],
        selected_footprint='pad-0-'+str(config['patch_x'][0]),selected_member='b2-0-0',
        stations=[0.,.5,1.],fixed_shell_edge='x=0; all six coordinates; zero prescribed offset')


def proposed_doubling_configs():
    low=source_settings('scale');high=source_settings('large')
    middle=dict(nx=100,ny=50,line_rows=tuple(range(4,50,10)),patch_x=tuple(range(5,100,10)),cases=100)
    return [low,middle,high]


def validate_warm_reference(result,cases,loaded_dof_index=0):
    """Verify AFTER timed public batch using its original selected work fields.

    Caller supplies exact single-slot force values in submitted order. No physical
    norm is derived from stiffness; existing source status enforces its own guards.
    This does not expose hidden row residuals or a new material-strain oracle.
    """
    require(result.status=='completed' and not result.production_qualified,'unqualified complete reference batch required')
    require(tuple(c.case_id for c in result.cases)==tuple(name for name,_ in cases),'ordered complete prefix required')
    require(result.factorization_count==1 and result.maximum_rhs_chunk==1,'one unchanged factor/chunk1 required')
    checks=[]
    for case,(_,force) in zip(result.cases,cases):
        internal=2.*case.internal_strain_energy;external=force*case.selected_displacements[loaded_dof_index]
        require(math.isfinite(internal) and math.isfinite(external),'finite original-unit work required')
        defect=internal-external;relative=1e-8*abs(external)
        require(abs(defect)<=relative and abs(defect)<=1e-12+relative,'original relative work failure')
        require(all(math.isfinite(x) for x in (*case.selected_displacements,*case.selected_affine_reactions)),
            'finite selected physical fields required')
        require(len(case.footprint_wrenches)==1 and len(case.beam_resultants_local)==1,'selected recovery required')
        for wrench in case.footprint_wrenches:
            require(len(wrench.node_actions)==4 and not wrench.production_qualified,'physical footprint recovery required')
            for name in ('shell_force_N','shell_moment_Nm','beam_force_N','beam_moment_Nm'):
                require(all(math.isfinite(x) for x in getattr(wrench,name)),'finite original-unit wrench required')
            for action in wrench.node_actions:
                require(all(math.isfinite(x) for name in ('shell_force_N','shell_direct_couple_Nm',
                    'beam_force_N','beam_direct_couple_Nm','gap') for x in getattr(action,name)),
                    'finite physical node action required')
        for _,stations in case.beam_resultants_local:
            require(len(stations)==3 and all(len(v)==6 and all(math.isfinite(x) for x in v) for v in stations),
                'finite three-station resultants required')
        checks.append(dict(case_id=case.case_id,internal_work=internal,external_work=external,
            work_defect=defect,original_relative_allowance=relative))
    require(len(result.member_resultant_envelopes)==18,'selected envelope inventory required')
    components=('N','Qy','Qz','Mx','My','Mz')
    for envelope in result.member_resultant_envelopes:
        j=components.index(envelope.component);i=envelope.station_index
        values=[(c.case_id,dict(c.beam_resultants_local)[envelope.member_id][i][j]) for c in result.cases]
        require(envelope.station_normalized==(0.,.5,1.)[i]
            and (envelope.minimum_case_id,envelope.minimum)==min(values,key=lambda x:x[1])
            and (envelope.maximum_case_id,envelope.maximum)==max(values,key=lambda x:x[1]),'member envelope governance')
    return checks


def cached_rhs_series(owner,cases,selection,recovery,work_cases,*,repetitions=5,clock=time.perf_counter,
        emit=lambda event:None):
    """Prepared component: one prime and >=5 serial batches on SAME factor.

    No module import, factorization or execution is triggered by preparing this
    file. Future authorized runner must bind installed artifact/runtime/resources.
    Timed region is the public batch plus identical lightweight count callback;
    work/recovery checks occur after it. It measures cached RHS+recovery only.
    """
    require(type(repetitions) is int and repetitions>=5,'at least five warm repetitions required')
    factor=owner._handle;records=[]
    for ordinal in range(repetitions+1):
        before=factor.solve_count;previous=before;counts=[]
        def on_case(case_id):
            nonlocal previous
            counts.append((case_id,factor.solve_count-previous,id(owner._handle)==id(factor)))
            previous=factor.solve_count
        start=clock()
        result=owner.solve_linear_many(cases,selected_dofs=selection['dofs'],selected_footprints=selection['footprints'],
            recover_members=recovery,on_case=on_case)
        elapsed=clock()-start
        emit(dict(kind='returned_before_assertions',ordinal=ordinal,result=asdict(result),counts=counts))
        checks=validate_warm_reference(result,work_cases)
        require(tuple(name for name,_,_ in counts)==tuple(name for name,_ in work_cases)
            and all(n in (1,2) and same for _,n,same in counts),'per-case same-factor counts')
        require(result.solve_count-before==sum(n for _,n,_ in counts),'cumulative solve accounting')
        require(owner._handle is factor and factor.factorization_count==1,'unchanged factor required')
        records.append(dict(ordinal=ordinal,prime=ordinal==0,batch_seconds=elapsed,
            primary_count=len(counts),correction_count=sum(n-1 for _,n,_ in counts),checks=checks))
        emit(dict(kind='validated',record=records[-1]))
    return dict(quantity='cached_public_RHS_plus_selected_recovery',prime=records[0],warm=records[1:],
        median_seconds=statistics.median(r['batch_seconds'] for r in records[1:]),
        assembly_qualification=False,performance_acceptance=False)


def fresh_owner_series(factory,verify,*,repetitions=5,clock=time.perf_counter):
    """Prepared fresh-owner construction component, not a pure assembly timer.

    Caller performs a separately reported small cold warmup before this function.
    Exactly >=5 measured constructions occur in a single warm process.
    Caller retains same definition/artifact; factory is existing public constructor.
    Verification is outside construction timing; every owner closes on failure.
    Existing factor cache/backend selection remains unchanged and is reported.
    """
    require(type(repetitions) is int and repetitions>=5,'five warm fresh-owner constructions required')
    records=[]
    for ordinal in range(repetitions):
        owner=None
        try:
            start=clock();owner=factory();elapsed=clock()-start
            diagnostics=owner._handle.diagnostics()
            require(diagnostics['factorization_count']==1,'one factor per independently constructed owner')
            verify(owner)
            records.append(dict(ordinal=ordinal,construction_seconds=elapsed,
                backend=diagnostics['backend'],factorization_reported_seconds=diagnostics['factorization_time']))
        finally:
            if owner is not None:owner.close()
    return dict(quantity='complete_public_construction',warm=records,
        median_seconds=statistics.median(x['construction_seconds'] for x in records),
        assembly_qualification=False,performance_acceptance=False)


def cached_sequence_series(owner,loads,selection,recovery,*,repetitions=5,clock=time.perf_counter,
        emit=lambda event:None):
    """One prime plus five SERIAL 1/10/100 sequences on one reopened owner.

    Component timings exclude post-call checks/serialization. Sequence sum is
    cached public solve+selected recovery, not pure assembly or recovery time.
    No extra full-size constructor, load or numerical oracle is introduced.
    """
    require(len(loads)==100 and type(repetitions) is int and repetitions>=5,'100 matching loads and five repetitions')
    factor=owner._handle;sequences=[]
    # Reuse the same single-batch verifier without its prime/warm outer loop.
    for ordinal in range(repetitions+1):
        batches=[]
        for count in (1,10,100):
            before=factor.solve_count;previous=before;counts=[]
            def callback(name):
                nonlocal previous
                counts.append((name,factor.solve_count-previous,owner._handle is factor));previous=factor.solve_count
            start=clock()
            result=owner.solve_linear_many(tuple(loads[:count]),selected_dofs=selection['dofs'],
                selected_footprints=selection['footprints'],recover_members=recovery,on_case=callback)
            elapsed=clock()-start
            emit(dict(kind='returned_before_assertions',ordinal=ordinal,batch_size=count,result=asdict(result),counts=counts))
            expected=[(f'case-{i+1}',-float(i+1)) for i in range(count)]
            checks=validate_warm_reference(result,expected)
            require(tuple(x[0] for x in counts)==tuple(x[0] for x in expected)
                and all(n in (1,2) and same for _,n,same in counts),'ordered per-case same-factor counts')
            require(result.solve_count-before==sum(n for _,n,_ in counts)
                and owner._handle is factor and factor.factorization_count==1,'cumulative one-factor counts')
            record=dict(batch_size=count,seconds=elapsed,primary_count=count,
                correction_count=sum(n-1 for _,n,_ in counts),checks=checks)
            batches.append(record);emit(dict(kind='validated',ordinal=ordinal,record=record))
        sequences.append(dict(ordinal=ordinal,prime=ordinal==0,batches=batches,seconds=sum(x['seconds'] for x in batches)))
    return dict(quantity='cached_public_1_10_100_RHS_plus_selected_recovery',prime=sequences[0],warm=sequences[1:],
        median_seconds=statistics.median(x['seconds'] for x in sequences[1:]),
        primary_count=111*(repetitions+1),factorization_count=1,performance_acceptance=False,assembly_qualification=False)


def constructor_phase_map():
    """Source landmarks for separate diagnostic profiling; no product rewrite."""
    tree=ast.parse(PANEL.read_text());owner=next(n for n in tree.body if isinstance(n,ast.ClassDef)
        and n.name=='ExperimentalB2Q4PanelAnalysis')
    init=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    landmarks=[]
    assignments={'compiled':'local_member_setup','model':'shell_model_setup','K':'physical_stiffness_CSR',
        'constraint_rows':'constraint_assembly','C':'physical_constraint_CSR','T':'affine_preparation'}
    for node in ast.walk(init):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in assignments:
                    landmarks.append(dict(label=assignments[target.id],line=node.lineno))
        if isinstance(node,ast.For) and ast.unparse(node.iter)=='pieces':
            landmarks.append(dict(label='shell_element_and_scatter_assembly',line=node.lineno))
        if isinstance(node,ast.For) and ast.unparse(node.iter)=='compiled':
            landmarks.append(dict(label='beam_element_and_scatter_assembly',line=node.lineno))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='factorize':
            landmarks.append(dict(label='eligible_factorization',line=node.lineno))
    require({x['label'] for x in landmarks}>={'local_member_setup','physical_stiffness_CSR','physical_constraint_CSR',
        'affine_preparation','eligible_factorization','shell_element_and_scatter_assembly'},'source landmark drift')
    return dict(path=str(PANEL),source_sha256=sha(PANEL),class_name=owner.name,method='__init__',
        first_line=init.lineno,last_line=init.end_lineno,landmarks=sorted(landmarks,key=lambda x:x['line']),
        usage='separate source-bound diagnostic profiling only; NOT unprofiled assembly acceptance')


def execution_gate(authority,plan_hash):
    require(authority.get('status')=='AUTHORIZED' and authority.get('numerical_slot_transferred') is True
        and authority.get('plan_sha256')==plan_hash and authority.get('scope')=='cached-public-111-warm5',
        'fresh root authority/slot and exact preparation required')
    require(authority.get('allowance_seconds')==600 and authority.get('reserve_seconds')==60
        and authority.get('memory_bytes')==4*1024**3 and authority.get('controls_seconds')==20
        and authority.get('installed_seconds')==360,'unadopted resource protocol')


def control_suite():
    """Actual production selector: only the owned eight-control filename."""
    import unittest
    return unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=CONTROL_FILENAME)


def adjudicate_controls(result):
    """Discovery success alone never establishes the required control inventory."""
    counts=dict(executed=result.testsRun,skips=len(result.skipped),errors=len(result.errors),failures=len(result.failures))
    passed=(type(counts['executed']) is int and counts['executed']==EXPECTED_CONTROLS
        and counts['skips']==counts['errors']==counts['failures']==0 and result.wasSuccessful())
    return dict(status='PASS_CONTROLS' if passed else 'FAILED_CONTROLS',expected=EXPECTED_CONTROLS,**counts)


def assert_plan(plan):
    for name,h in plan['source_bindings'].items():require(sha(ROOT/name)==h,'bound source/input drift '+name)
    for name,h in plan['artifact']['package'].items():require(sha(Path(plan['artifact']['site'])/name)==h,'installed byte drift '+name)


def installed_child(attempt,plan):
    """Future authorized installed caller. Import/cold reopen/compile are separate.

    No profiler, full-vector copy or residual oracle is inside warm timed calls.
    Unchanged completed PUBLIC source enforces per-row/certification/material
    guards. Returned selected fields/envelopes/work/counts are checked outside.
    Every returned batch is saved before assertions; validated prefix is flushed.
    """
    import psutil
    psutil.Process().cpu_affinity([0])
    helper_path=ROOT/'scripts/r8_b2_installed_batch_persistence_dev.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('bound_b2_functional_helpers',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    site=Path(plan['artifact']['site']).resolve()
    sys.path[:]=[str(site)]+[x for x in sys.path if not Path(x or '.').resolve().is_relative_to(ROOT/'src')
        and not Path(x or '.').resolve().is_relative_to(Path('C:/Github/ANYfeSolver/src'))]
    owner=None;record=dict(status='UNRESOLVED',performance_acceptance=False,qualification=False)
    accepted=[];start=time.perf_counter()
    try:
        import numpy as np
        from anyfesolver import stiffened_shell as api
        from anyfem.solve.stiffened_shell_b2 import load_case
        p=dict(site=str(site),package=plan['artifact']['package'])
        helper.save(attempt/'runtime-before.json',helper.runtime(p));imported=time.perf_counter()
        reopened=load_case(ROOT/plan['frozen_case']['path'])
        require(hashlib.sha256(helper.payload_bytes(reopened.payload)).hexdigest()==plan['frozen_case']['canonical_sha256'],
            'frozen canonical definition drift')
        data=reopened.to_dict();definition=reopened.definition()
        expected=plan['existing_profiles']['large']['inventory'];selected=selected_input(source_settings('large'))
        require(len(definition.members)==1000 and len(definition.physical_footprints)==100
            and helper.matrix_identity(definition.transformation,np)['sha256']==plan['frozen_case']['transformation']['sha256'],
            'frozen original supports/member/footprint identity')
        loads=[]
        for i,row in enumerate(data['cases_sparse']):
            require(row==dict(id=f'case-{i+1}',load_sparse=dict(size=expected['dofs'],
                entries=[[selected['loaded_dof'],-float(i+1)]])),'exact ordered original load identity')
            load=np.zeros(expected['dofs']);load[selected['loaded_dof']]=-float(i+1);loads.append((row['id'],load))
        del definition,data
        reopened_at=time.perf_counter();owner=reopened.create_analysis();compiled=time.perf_counter()
        selection=dict(dofs=tuple(selected['selected_dofs']),
            footprints=api.B2PhysicalRigidFootprintSelection((selected['selected_footprint'],)))
        recovery={selected['selected_member']:tuple(selected['stations'])}
        require(owner._handle.factorization_count==1,'single unchanged factor required')
        record.update(import_seconds=imported-start,reopen_seconds=reopened_at-imported,compile_seconds=compiled-reopened_at,
            factor_before=owner._handle.diagnostics())
        helper.save(attempt/'stages.json',record)
        def emit(event):
            index=len(accepted)
            if event['kind']=='returned_before_assertions':
                helper.save(attempt/f"returned-{event['ordinal']}-{event['batch_size']}.json",event)
            else:
                accepted.append(event);helper.save(attempt/'accepted-prefix.json',accepted)
        series=cached_sequence_series(owner,loads,selection,recovery,emit=emit)
        require(series['primary_count']==666,'prime plus five warm111 sequences')
        record.update(status='PASS_CANDIDATE_WARM_BASELINE',series=series,accepted_batches=len(accepted),
            factor_after=owner._handle.diagnostics(),baseline_comparison=None)
    except Exception as exc:
        import traceback
        (attempt/'failure.txt').write_text(traceback.format_exc())
        record.update(status='FAILED_OR_UNRESOLVED',error=str(exc),accepted_batches=len(accepted))
    finally:
        if owner is not None:record['terminal_factor']=owner._handle.diagnostics();owner.close()
        helper.save(attempt/'result.json',record)
        helper.save(attempt/'runtime-after.json',helper.runtime(dict(site=str(site),package=plan['artifact']['package'])))
    return record['status']=='PASS_CANDIDATE_WARM_BASELINE'


def run(phase,authority_path):
    """Existing mechanics-blind supervisor; fresh authorized attempt only."""
    plan=json.loads((OUT/'RUN_PLAN.json').read_text());a=json.loads(Path(authority_path).read_text())
    execution_gate(a,sha(OUT/'RUN_PLAN.json'));assert_plan(plan)
    require(phase in ('controls','installed'),'declared phase')
    ledger_path=OUT/'ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='OPEN',used_seconds=0.,attempts=[])
    require(ledger['status']=='OPEN' and not any(x['phase']==phase for x in ledger['attempts']),'closed or consumed phase')
    if phase=='installed':require(any(x['phase']=='controls' and x['passed'] for x in ledger['attempts']),'controls PASS required')
    cap=20. if phase=='controls' else 360.
    require(600.-60.-ledger['used_seconds']>=cap,'preserve final reserve')
    attempt=OUT/f'attempt-{phase}-01';attempt.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT))
    from scripts._b3_e4_f03_local_scientific_supervisor import supervise,SupervisorLimits
    import importlib.util
    spec=importlib.util.spec_from_file_location('bound_runtime_helpers',ROOT/'scripts/r8_b2_installed_batch_persistence_dev.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    env=dict(os.environ);env.update(helper.ENV);env['PYTHONPATH']=plan['artifact']['site']
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',phase,str(attempt)]
    helper.save(attempt/'binding.json',dict(plan=plan,command=command,cwd=str(attempt),authority_path=str(Path(authority_path).resolve()),
        authority_sha256=sha(authority_path),
        environment={k:env[k] for k in (*helper.ENV,'PYTHONPATH')}))
    for name,h in plan['source_bindings'].items():
        dest=attempt/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        require(sha(dest)==h,'snapshot drift')
    helper.save(ledger_path,ledger)
    with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr:
        receipt=supervise(command,cwd=attempt,environment=env,stdout=stdout,stderr=stderr,limits=SupervisorLimits(cap,4*1024**3,.05))
    helper.save(attempt/'receipt.json',asdict(receipt))
    result=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {}
    passed=receipt.reason=='completed' and receipt.exit_code==0 and result.get('status','').startswith('PASS')
    if phase=='controls':
        passed=passed and result.get('expected')==EXPECTED_CONTROLS and result.get('executed')==EXPECTED_CONTROLS \
            and result.get('skips')==result.get('errors')==result.get('failures')==0
    ledger['used_seconds']+=receipt.elapsed_seconds;ledger['attempts'].append(dict(phase=phase,passed=passed,receipt=asdict(receipt)))
    if not passed or phase=='installed':ledger.update(status='CLOSED',discarded_unused_seconds=600.-ledger['used_seconds'])
    helper.save(ledger_path,ledger);return passed


def prepare():
    existing={name:dict(config=source_settings(name),inventory=inventory(source_settings(name)))
        for name in ('small','medium','scale','large','beam_rich')}
    doubling=proposed_doubling_configs()
    full=json.loads((FUNCTIONAL/'v2/attempt-full-01/result.json').read_text())
    beam=json.loads((FUNCTIONAL/'v3-beam/attempt-beam-01/result.json').read_text())
    require(full['status']==beam['status']=='PASS_INSTALLED_FUNCTIONAL','terminal functional evidence required')
    artifact=json.loads((FUNCTIONAL/'v3-beam/PREPARED.json').read_text())
    for name,h in artifact['package'].items():require(sha(Path(artifact['site'])/name)==h,'installed artifact drift '+name)
    refs=[Path(__file__),ROOT/'tests/test_r8_b2_g5_readiness_dev.py',MODELS,PANEL,CONTRACT,
        ROOT/'scripts/r8_b2_installed_batch_persistence_dev.py',FUNCTIONAL/'FINAL_REPORT.md',
        FUNCTIONAL/'FINAL_BINDINGS.json',FUNCTIONAL/'v2/attempt-full-01/result.json',
        FUNCTIONAL/'v2/attempt-full-01/receipt.json',FUNCTIONAL/'v3-beam/attempt-beam-01/result.json',
        FUNCTIONAL/'v3-beam/attempt-beam-01/receipt.json',FUNCTIONAL/'v3-beam/PREPARED.json']
    ast.parse(Path(__file__).read_text())
    for name,h in artifact['current_b2_sources'].items():require(sha(ROOT/'src/anyfesolver'/name)==h,'relevant B2 source/artifact drift')
    plan=dict(status='PREPARED_REQUIRED_INPUTS_OPEN',execution_authorized=False,numerical_runs=0,
        source_bindings={str(p.relative_to(ROOT)):sha(p) for p in refs},artifact=dict(site=artifact['site'],
            solver_sha256=artifact['solver_sha256'],consumer_sha256=artifact['consumer_sha256'],package=artifact['package']),
        existing_profiles=existing,doubling_proposal=[dict(config=c,inventory=inventory(c),
            selected_input=selected_input(c),root_adopted_preparation=True,executed=False,
            interpretation='fixed1m cells; doubling connected physical repetitions, not convergence')
            for i,c in enumerate(doubling)],constructor_profiling_map=constructor_phase_map(),
        contracts=dict(warm_repetitions_minimum=5,alternating_pairs_if_gain_claimed=7,
            controls=dict(filename=CONTROL_FILENAME,expected_executed=EXPECTED_CONTROLS,skips=0,errors=0,failures=0),
            assembly_doubling_time=2.5,assembly_state_memory_doubling=2.3,
            mechanically_identical_warm_regression_time=1.05,memory=1.10,
            original_relative_work=1e-8,physical_guards='unchanged installed PUBLIC source guards',
            factor_reuse='one per unchanged static operator; per-case <=one correction; separate owners factor separately'),
        baseline=dict(functional_reference='0ed73724 full111 + beam100 PASS',
            five_warm_mechanically_identical_series=None,comparison_authorized=False,
            excluded=['historical scalar-floor law','G0 B3-GE smoke','F13 small ownership oracle',
                'old3b6 refinement with unresolved original-work floor','single profiled full/beam observations']),
        observed_cost_screen=dict(full_construction_seconds=full['compile_seconds'],
            five_constructions_seconds=5*full['compile_seconds'],
            cached_prime_plus_five_screen_seconds=full['compile_seconds']+6*full['batch_and_recovery_seconds'],
            cached_batch_seconds_profiled=full['batch_and_recovery_seconds'],beam_construction_seconds=beam['compile_seconds'],
            prediction=False,old_child_ceiling_seconds=240.,old_memory_ceiling_bytes=4*1024**3),
        blockers=['fresh root authority/slot before any controls or installed measurement',
            'source-bound pure assembly/state-memory observable and metric definition for later doublings',
            'five-warm accepted mechanically identical baseline before any comparison'],
        next_ready_quantity='candidate-only cached PUBLIC111 warm5: one reopened owner, one prime then five1/10/100 sequences',
        prospective_protocol=dict(allowance_seconds=600,reserve_seconds=60,controls_seconds=20,installed_seconds=360,
            memory_bytes=4*1024**3,CPU=[0],numerical_threads=1,JIT=False,optimize=0,one_job=True,
            warm_timing='sum of three public calls; checks/JSON outside; callbacks identical; imports/reopen/compile/prime separate',
            scientific_source_guards='unchanged completed PUBLIC route, no new profiled residual oracle in warm measurement'),
        frozen_case=dict(path=str((FUNCTIONAL/'v2/attempt-full-01/full.anyfem-b2.json').relative_to(ROOT)),
            canonical_sha256=full['persistence']['canonical_sha256'],transformation=full['matrices']['T']),
        measurement_components=['cached_sequence_series','cached_rhs_series','fresh_owner_series','constructor_phase_map'],
        acceptance=False,qualification=False,performance_claim=False,mandatory_mistral_review='PENDING')
    plan['source_bindings'].update({str((ROOT/'src/anyfesolver'/name).relative_to(ROOT)):h
        for name,h in artifact['current_b2_sources'].items()})
    for path in (ROOT/plan['frozen_case']['path'],ROOT/'scripts/_b3_e4_f03_local_scientific_supervisor.py',
            ROOT/'scripts/_b3_e4_f03_local_scientific_cases.py'):
        plan['source_bindings'][str(path.relative_to(ROOT))]=sha(path)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'RUN_PLAN.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',choices=('controls','installed'));parser.add_argument('--authority');parser.add_argument('--child',nargs=2)
    args=parser.parse_args()
    if args.prepare:
        plan=prepare();print(json.dumps(dict(status=plan['status'],installed_package_files=len(plan['artifact']['package']),
            plan_sha256=sha(OUT/'RUN_PLAN.json'),numerical_runs=0)))
    elif args.run:
        if not args.authority:parser.error('NEW explicit root authority required')
        raise SystemExit(0 if run(args.run,args.authority) else 1)
    elif args.child:
        phase,folder=args.child;attempt=Path(folder);binding=json.loads((attempt/'binding.json').read_text())
        require(sha(binding['authority_path'])==binding['authority_sha256'],'authority binding drift')
        execution_gate(json.loads(Path(binding['authority_path']).read_text()),sha(OUT/'RUN_PLAN.json'))
        plan=binding['plan'];assert_plan(plan)
        if phase=='installed':raise SystemExit(0 if installed_child(attempt,plan) else 1)
        require(phase=='controls','declared child phase')
        import psutil,unittest
        psutil.Process().cpu_affinity([0])
        expected_env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT')}
        expected_env['PYTHONOPTIMIZE']='0'
        basic=dict(affinity=psutil.Process().cpu_affinity(),environment={k:os.environ.get(k) for k in
            expected_env},
            optimize=sys.flags.optimize,executable=sys.executable)
        require(basic['affinity']==[0] and basic['optimize']==0 and basic['environment']==expected_env,'effective basic control runtime')
        (attempt/'runtime-before.json').write_text(json.dumps(basic,indent=2))
        suite=control_suite()
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        adjudication=adjudicate_controls(result)
        (attempt/'result.json').write_text(json.dumps(adjudication))
        (attempt/'runtime-after.json').write_text(json.dumps(dict(basic,affinity=psutil.Process().cpu_affinity()),indent=2))
        raise SystemExit(0 if adjudication['status']=='PASS_CONTROLS' else 1)
    else:parser.error('prepare or NEW root-authorized execution only')
