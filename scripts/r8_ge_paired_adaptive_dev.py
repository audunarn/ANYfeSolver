"""Connected paired adaptive development route; fresh authority is mandatory."""
from pathlib import Path
from dataclasses import asdict,is_dataclass
import argparse,ast,hashlib,json,math,os,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE_DOC=ROOT/'docs/agent_plans/development/r8-ge-paired-adaptive-20261010'
DOC=BASE_DOC/'execution-v4'
PACKAGE=DOC/'package'
sys.path.insert(0,str(ROOT))
from scripts import r8_ge_coupled_cycle_dev as prior
PLAN={**prior.PLAN,'adaptive':True,'max_cutbacks':4,'max_attempts':96,
      'primary_accepts':'controller-dependent','cold_continuation_accepts':'same remaining controller',
      'rebased_continuation_accepts':'same remaining controller','rejected_probe_iterations':None,
      'rejected_probe_from':None,'rejected_probe_target':None}
RESOURCES=dict(budget_s=900,reserve_s=120,source_cap_s=30,wheel_cap_s=60,installed_cap_s=600,
               tree_rss_bytes=2147483648,cpu_affinity=[0],threads=1,jit=False,optimization=0,numerical_jobs=1)
SOURCE_NAMES=('test_source_paired_identity_and_legacy_header','test_source_digest_event_family_and_corruption_refusal',
 'test_source_seed_survives_rejected_wrapper','test_source_progress_direction_and_cursor_refusal',
 'test_source_typed_cutback_and_durable_limit','test_source_cancellation_and_untyped_failures_not_cutback',
 'test_source_postpublication_observer_failure_not_cutback','test_source_restore_atomic_cancel_stale_and_progress',
 'test_source_public_restore_dispatch_before_legacy','test_source_public_solve_adaptive_dispatch',
 'test_source_attempt_observer_persists_failure_and_restores','test_source_explicit_inventory_and_adoption_refusal')
sha,plain=prior.sha,prior.plain

def diagnostic_value(value):
    value=plain(value)
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':repr(value)}
    if isinstance(value,dict):return {k:diagnostic_value(v) for k,v in value.items()}
    if isinstance(value,list):return [diagnostic_value(v) for v in value]
    return value

def write(path,value):
    Path(path).write_text(json.dumps(diagnostic_value(value),indent=2,allow_nan=False)+'\n')

def require_adoption(value):
    if value.get('status')!='ADOPTED' or value.get('plan')!=PLAN:
        raise ValueError('exact root physical/controller adoption required')
    return PLAN

def bindings():
    names=['PREPARATION.md','PROSPECTIVE_PLAN.json','SOURCE_TREE.json','SOURCE_SPANS.json','IMPLEMENTATION_BINDINGS.json','READINESS.json']
    # Used authority bytes are bound separately by the supervisor receipt.
    # Excluding authority records here avoids a circular authority->manifest->authority hash.
    paths=[Path(__file__),ROOT/'tests/test_r8_ge_paired_adaptive_dev.py',ROOT/'examples/r8_ge_paired_adaptive_public_example.py',
        ROOT/'scripts/r8_ge_coupled_cycle_dev.py',ROOT/'scripts/r8_f06_paired_active_tangent_dev.py',
        ROOT/'scripts/r8_f06_cyclic_material_dev.py',ROOT/'scripts/r8_f06_native_material_breadth_dev.py',
        ROOT/'scripts/r8_f06_fibre_log_domain_dev.py',ROOT/'scripts/r8_f06_log_domain_check_dev.py',
        ROOT/'tests/conftest.py',*prior.IDENTITIES,*[BASE_DOC/x for x in names],
        *[DOC/('ADAPTIVE_'+x+'.json') for x in ('ADOPTION','SLOT_TRANSFER')],DOC/'EXECUTION_V4_READINESS.json']
    paths += [ROOT/'scripts/r8_ge_paired_adaptive_bootstrap_dev.py',
              BASE_DOC/'entrypoint-v2/ENTRYPOINT_READINESS.json']
    paths += list((BASE_DOC/'legacy-golden-v1').glob('*'))
    if PACKAGE.exists():
        paths += [p for p in (PACKAGE/'snapshot-bindings.json',PACKAGE/'metadata-bindings.json',PACKAGE/'artifact.json') if p.exists()]
        paths += list((PACKAGE/'wheels').glob('*.whl'))
    pending=[p for p in paths if p.parent==ROOT/'scripts'];seen=set(pending)
    while pending:
        for node in ast.walk(ast.parse(pending.pop().read_text(encoding='utf-8'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                modules=[node.module] if node.module!='scripts' else ['scripts.'+x.name for x in node.names]
            for module in modules:
                if module.startswith('scripts.'):
                    path=ROOT/(module.replace('.','/')+'.py')
                    if path.exists() and path not in seen:seen.add(path);paths.append(path);pending.append(path)
    value={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    prior.verify_bindings(value,value)
    return value

def source_tree():
    expected=json.loads((BASE_DOC/'SOURCE_TREE.json').read_text())
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'src/anyfesolver').rglob('*.py')}
    if actual!=expected:raise ValueError('current source tree changed after preparation')
    return expected

def artifact():
    expected=source_tree();record=json.loads((PACKAGE/'artifact.json').read_text())
    manifest=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
    wheels=list((PACKAGE/'wheels').glob('*.whl'))
    if len(wheels)!=1 or sha(wheels[0])!=record['sha256'] or manifest!=expected:
        raise ValueError('new candidate artifact identity mismatch')
    with zipfile.ZipFile(wheels[0]) as archive:
        for rel,digest in expected.items():
            entry=Path(rel).relative_to('src').as_posix()
            if not sha(PACKAGE/'snapshot'/rel)==sha(PACKAGE/'site'/entry)==hashlib.sha256(archive.read(entry)).hexdigest()==digest:
                raise ValueError('new candidate snapshot/site/wheel drift: '+rel)
    return dict(files=len(expected),wheel_sha256=sha(wheels[0]))

class AttemptObserver(prior.PublicObserver):
    """Observe the actual fixed call before adaptive failure-journal publication."""
    def __init__(self,out,*,atomic=None):
        super().__init__(out);self.merit=None;self.atomic=atomic;self.attempt=0
    def observe(self,frame,event,arg):
        if event=='return' and frame.f_code.co_name=='merit' and frame.f_code.co_filename.endswith('_stiffened_shell_displacement_corrector.py'):
            self.merit=float(arg)
        super().observe(frame,event,arg)
    def __enter__(self):
        from anyfesolver import _stiffened_shell_displacement_control as fixed
        if self.atomic is None:
            from scripts import r8_f06_paired_active_tangent_dev as atomic
            self.atomic=atomic
        self.fixed=fixed;self.original=fixed.solve
        def observed(owner,*args,**kwargs):
            before=self.atomic.capture(owner);self.merit=None;self.attempt+=1
            row=dict(attempt=self.attempt,generation=owner._generation,targets=plain(args[1]),status='STARTED')
            write(self.out/'live-attempt.json',row)
            try:
                value=self.original(owner,*args,**kwargs);row['status']='RETURNED';return value
            except BaseException as error:
                row.update(status='FAILED',flags=self.atomic.atomic_flags(owner,before),
                    exception=dict(type=type(error).__name__,message=str(error),detail=plain(getattr(error,'detail',None))),
                    actual_fixed_dual_work_merit=self.merit,
                    merit_provenance='last executed fixed-corrector merit call; unavailable before first merit call; not reconstructed final failure residual')
                from anyfesolver._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure
                if isinstance(error,DisplacementCorrectorFailure) and owner._generation==before['generation']:
                    prior.check_atomic(row['flags'])
                raise
            finally:
                with (self.out/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(diagnostic_value(row),allow_nan=False)+'\n')
        fixed.solve=observed
        return super().__enter__()
    def __exit__(self,*args):
        self.fixed.solve=self.original;return super().__exit__(*args)

def _connected_program(out,*,control_factory=None,control_kwargs=None):
    """Reusable mechanical/lifecycle program behind the historical connected stage.

    Control injection is explicit and single-route: control_factory returns the
    public displacement control, or control_kwargs extends the original default
    construction; the default route constructs the original control exactly.
    Historical stage/out/adoption/artifact guards remain in connected().
    """
    if control_factory is not None and control_kwargs is not None:
        raise ValueError('one explicit control injection route required')
    from anyfesolver import stiffened_shell as ss
    from anyfesolver.control import CancellationToken,SolveCancelled
    from scripts.r8_f06_cyclic_material_dev import definition,actions
    from scripts import r8_f06_paired_active_tangent_dev as atomic
    import numpy as np
    report=dict(status='INCOMPLETE',accepted=[],public_failures=[],atomic=[],parity={},plan=PLAN,
                independent_review='Mistral pending',acceptance=False)
    owners=[];primary=None;previous=None;observer=None
    def save(owner,label):
        (out/(label+'.bin')).write_bytes(owner.checkpoint());write(out/(label+'-recovery.json'),owner.recover())
    def make(raw):
        owner=ss.create_analysis(definition=definition('coupled'));owners.append(owner);owner.restore(raw);return owner
    def invoke(owner,label,**extra):
        before=atomic.capture(owner)
        try:return owner.solve(actions('coupled'),tuple(PLAN['targets']),displacement_control=control,increment_control=limits,
                               max_iterations=PLAN['max_iterations'],max_backtracks=PLAN['max_backtracks'],**extra)
        except BaseException as error:
            report['public_failures'].append(dict(label=label,flags=atomic.atomic_flags(owner,before),
                exception=dict(type=type(error).__name__,message=str(error),detail=plain(getattr(error,'detail',None))),
                controller=plain(owner._displacement_adaptive[0].state.payload()) if owner._displacement_adaptive else None))
            write(out/'cycle.json',report);raise
        finally:save(owner,label+'-terminal')
    try:
        for path,digest in prior.IDENTITIES.items():
            if sha(path)!=digest:raise ValueError('historical checkpoint drift')
        primary=make(prior.ACCEPTED.read_bytes());previous=prior.accumulation(primary)
        control=(control_factory() if control_factory is not None else
            ss.DisplacementControl('shell',2,0,force_seed_sha256=prior.IDENTITIES[prior.SEED],
                **(control_kwargs or {})))
        limits=ss.AdaptiveIncrementControl(PLAN['max_cutbacks'],PLAN['max_attempts'])
        token=CancellationToken();token.cancel('pretrial cancellation control')
        with AttemptObserver(out):
            try:invoke(primary,'cancel-pretrial',cancellation_token=token)
            except SolveCancelled:
                flags=report['public_failures'][-1]['flags'];prior.check_atomic(flags)
                report['atomic'].append(dict(label='pretrial',flags=flags))
        token=CancellationToken()
        with AttemptObserver(out) as observer:
            def accepted(result):
                nonlocal previous
                after=prior.accumulation(primary);mono=prior.monotone(previous,after);previous=after
                points=[x for x in observer.latest.get('contributions',[]) if x['kind']=='point']
                shells=observer.latest.get('shell_metadata',[])
                row=dict(target=result.target,amplitude=result.amplitude,generation=result.generation,
                    equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
                    point_work_max=max((x['absolute_error'] for x in points),default=float('inf')),point_count=len(points),
                    monotonic=mono,constitutive_valid=bool(shells) and all(all(x['converged']) for x in shells),
                    materials=prior.material_summary(result),history=plain(result.history),controller=plain(result.increment_attempts))
                nodal=primary._shell_count+6*len(primary._beam_ids)
                for name,value in (('actual_applied_translation_resultant',result.applied_dead_load),('actual_support_translation_resultant',result.support_reactions)):
                    row[name]=np.asarray(value)[:nodal].reshape(-1,6)[:,:3].sum(axis=0).tolist()
                report['accepted'].append(row);write(out/'cycle.json',report);save(primary,'accepted-'+str(len(report['accepted'])))
                prior.validate_metrics(row)
                if len(report['accepted'])==1:token.cancel('accepted adaptive prefix for durable continuation')
            try:invoke(primary,'primary-prefix',cancellation_token=token,on_accepted=accepted)
            except SolveCancelled:
                if len(report['accepted'])!=1:raise
            prefix=primary.checkpoint();cold=make(prefix);chart=make(prefix)
            if cold.checkpoint()!=prefix or chart.checkpoint()!=prefix:raise ValueError('cold prefix checkpoint words differ')
            rotations=[6*i+j for i in range(len(chart._shell_ids)) for j in (3,4,5)]
            selected=tuple(i for i in rotations if all(i+k in chart._free for k in range(3)) and i%6==3)
            if not selected:raise ValueError('no admissible shell chart rebase triple')
            chart.rebase_rotations(chart._generation)
            report['parity']['prefix-cold']=prior.parity(primary.recover(),cold.recover())
            report['parity']['prefix-chart']=prior.parity(primary.recover(),chart.recover())
            invoke(primary,'primary-complete',on_accepted=accepted)
            def continuation(owner,label):
                previous=prior.accumulation(owner)
                def record(result):
                    nonlocal previous
                    after=prior.accumulation(owner);mono=prior.monotone(previous,after);previous=after
                    points=[x for x in observer.latest.get('contributions',[]) if x['kind']=='point']
                    shells=observer.latest.get('shell_metadata',[])
                    row=dict(label=label,target=result.target,amplitude=result.amplitude,generation=result.generation,
                        equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
                        point_work_max=max((x['absolute_error'] for x in points),default=float('inf')),point_count=len(points),
                        monotonic=mono,constitutive_valid=bool(shells) and all(all(x['converged']) for x in shells),
                        materials=prior.material_summary(result),history=plain(result.history),controller=plain(result.increment_attempts))
                    report.setdefault('continuations',[]).append(row);write(out/'cycle.json',report)
                    save(owner,label+'-accepted-'+str(result.generation));prior.validate_metrics(row)
                return record
            invoke(cold,'cold-complete',on_accepted=continuation(cold,'cold'))
            invoke(chart,'chart-complete',on_accepted=continuation(chart,'chart'))
            report['parity']['cold']=prior.parity(primary.recover(),cold.recover())
            report['parity']['chart']=prior.parity(primary.recover(),chart.recover())
            report['parity']['cold-history']=prior.history_parity(primary,cold,exact=True)
            report['parity']['chart-history']=prior.history_parity(primary,chart,exact=False)
            if primary._displacement_adaptive[0].state!=cold._displacement_adaptive[0].state or primary._displacement_adaptive[0].state!=chart._displacement_adaptive[0].state:
                raise ValueError('controller restart cursor/history differs')
            if primary.recover().target!=PLAN['loaded_target'] or not primary._displacement_adaptive[0].state.completed:
                raise ValueError('required physical cycle incomplete')
        report['reversal']=prior.reversal([row['materials']['fibre_flow_signs'] for row in report['accepted']])
        report['status']='SCOPED_ADAPTIVE_CYCLE_PASS'
    except BaseException as error:
        report.update(status='UNRESOLVED_ADAPTIVE_CYCLE',exception=dict(type=type(error).__name__,message=str(error),detail=plain(getattr(error,'detail',None))))
        raise
    finally:
        if primary is not None:save(primary,'terminal')
        write(out/'cycle.json',report)
        for owner in reversed(owners):owner.close()

def connected(out):
    if (os.environ.get('GE_PAIRED_ADAPTIVE_STAGE')!='installed-cycle' or
            os.environ.get('GE_PAIRED_ADAPTIVE_OUTPUT')!=str(out) or
            out!=DOC/'attempt-installed-cycle1'):
        raise ValueError('sole installed child and exact execution-v4 output required')
    require_adoption(json.loads((DOC/'ADAPTIVE_ADOPTION.json').read_text()));artifact()
    return _connected_program(out)

def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0])
    if json.loads((out/'bindings.json').read_text())!=bindings():raise ValueError('changed child bindings')
    source_tree()
    if stage=='wheel':
        from scripts import r8_f06_fibre_log_domain_dev as helper
        helper.DOC=DOC;helper.bindings=bindings
        return helper.child(out,'wheel')
    selected=ROOT/'src' if stage=='source-controls' else PACKAGE/'site'
    if stage=='installed-cycle':write(out/'artifact-before.json',artifact())
    sys.path[:0]=[str(selected),str(ROOT/'tests')]
    os.environ.update(GE_PAIRED_ADAPTIVE_OUTPUT=str(out),GE_PAIRED_ADAPTIVE_STAGE=stage)
    from scripts.r8_f06_log_domain_check_dev import runtime
    runtime(out,'runtime-before',selected)
    import pytest
    nodes=[str(ROOT/'tests/test_r8_ge_paired_adaptive_dev.py')+'::'+name for name in SOURCE_NAMES] if stage=='source-controls' else [str(ROOT/'tests/test_r8_ge_paired_adaptive_dev.py')+'::test_installed_public_adaptive_cycle']
    try:
        code=int(pytest.main(nodes+['-q']+(['-x'] if stage=='installed-cycle' else [])+
            ['-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
        if code==0:write(out/'counts.json',prior.counts(out/'junit.xml',12 if stage=='source-controls' else 1))
        return code
    finally:
        rt=runtime(out,'runtime-after',selected);tree=source_tree()
        for name,path in rt['imports'].items():
            if sha(path)!=tree['src/'+Path(path).relative_to(selected).as_posix()]:raise ValueError('actual import drift')
        if stage=='installed-cycle':write(out/'artifact-after.json',artifact())

def launch(stage):
    require_adoption(json.loads((DOC/'ADAPTIVE_ADOPTION.json').read_text()));source_tree()
    slot=json.loads((DOC/'ADAPTIVE_SLOT_TRANSFER.json').read_text())
    if slot.get('status')!='TRANSFERRED' or slot.get('one_numerical_job') is not True:raise ValueError('fresh exclusive slot required')
    if list(DOC.glob('attempt-'+stage+'*')):raise ValueError('sole attempt already consumed')
    if stage!='source-controls':
        receipt=json.loads((DOC/'attempt-source-controls1/receipt.json').read_text())
        if receipt['status']!='PASS':raise ValueError('source controls required')
        prior.counts(DOC/'attempt-source-controls1/junit.xml',12)
    if stage=='installed-cycle':
        if json.loads((DOC/'attempt-wheel1/receipt.json').read_text())['status']!='PASS':raise ValueError('new artifact required')
        artifact()
    from scripts.r8_f06_fibre_log_domain_dev import run
    auth='ADAPTIVE_'+dict(**{'source-controls':'SOURCE','wheel':'WHEEL','installed-cycle':'INSTALLED'})[stage]+'_AUTHORITY.json'
    authority=json.loads((DOC/auth).read_text())
    cap=RESOURCES[{'source-controls':'source_cap_s','wheel':'wheel_cap_s','installed-cycle':'installed_cap_s'}[stage]]
    if (authority.get('budget_s')!=RESOURCES['budget_s'] or authority.get('reserve_s')!=RESOURCES['reserve_s'] or
            authority.get('process_cap_s')!=cap):raise ValueError('fresh authority must match adopted resource envelope')
    code=run(stage,doc=DOC,entry=__file__,bind=bindings,authority_file=auth)
    if code or stage=='installed-cycle':
        ledger=DOC/'ledger.json';value=json.loads(ledger.read_text());value.update(status='CLOSED',unused_discarded_s=value['budget_s']-value['used_s']);write(ledger,value)
    return code

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('source-controls','wheel','installed-cycle'),required=True)
    parser.add_argument('--child',action='store_true');parser.add_argument('--out');args=parser.parse_args()
    raise SystemExit(child(Path(args.out),args.stage) if args.child else launch(args.stage))
