"""One observed public fixed correction; no new evaluations or solver choices."""
from pathlib import Path
from dataclasses import asdict,is_dataclass
import argparse,ast,hashlib,json,math,os,struct,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/agent_plans/development/r8-ge-paired-adaptive-20261010/execution-v4'
DOC=BASE/'fixed-correction-diagnostic-v1'
PACKAGE=BASE/'package'
OUTER=BASE/'attempt-installed-cycle1/terminal.bin'
INNER=DOC/'gen11-fixed-inner.bin'
TARGET=.025006454108063153
SEED='970e0360f22ed348ca2eaa7eee0d2d168d6d686d0a91a750f3ccd31794865f91'
WHEEL='efc977a1818ddc32bc48b1a54b5c0a3896e4b924a9a132e939703c739e1a6f40'
PLAN=dict(entry='CANONICAL_INNER_FIXED_SNAPSHOT',generation=11,domain='shell',node=2,component=0,
 target=TARGET,max_iterations=20,max_backtracks=8,adaptive=False,force_seed_sha256=SEED)
SOURCE_NAMES=('test_source_inner_selection_and_corruption','test_source_outer_cannot_be_fixed_entry',
 'test_source_sparse_residual_exact_and_corrupt','test_source_predicate_and_unavailable_markers',
 'test_source_observer_restores_hooks_on_failure','test_source_public_caller_terminal_capture',
 'test_source_binding_inventory_and_adoption','test_source_trace_does_not_call_mechanics')
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(x):
    if is_dataclass(x):return plain(asdict(x))
    if hasattr(x,'tolist'):return plain(x.tolist())
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':repr(x)}
    if x is None or isinstance(x,(str,bool,int,float)):return x
    return {'unserialized_type':type(x).__name__}
def write(path,value):Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')
def unwrap(raw,magic):
    if not raw.startswith(magic) or hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError('exact checkpoint framing/checksum required')
    n=len(magic);size=int.from_bytes(raw[n:n+4],'little')
    if not 0<size<=8192 or n+4+size>=len(raw)-32:raise ValueError('bounded checkpoint header required')
    header=json.loads(raw[n+4:n+4+size]);return header,raw[n+4+size:-32]
def select_inner(raw):
    outer,inner=unwrap(raw,b'ANYSSDCP1')
    fixed,body=unwrap(inner,b'ANYSSPD1')
    if (outer.get('dof')!=6 or outer.get('force_seed_sha256')!=SEED or
        fixed.get('generation')!=11 or fixed.get('control_dof')!=6 or
        fixed.get('seed_sha256')!=SEED or outer.get('model')!=fixed.get('model') or
        fixed.get('policy')!='GE_PAIRED_MIXED_Q4_J2_POINT_DISPLACEMENT_V1_DEV'):
        raise ValueError('exact gen11 paired fixed mechanical entry required')
    return inner,outer,fixed
def require_fixed_entry(raw):
    if raw.startswith(b'ANYSSDCP1'):raise ValueError('pending adaptive wrapper is not a fixed diagnostic entry')
    return unwrap(raw,b'ANYSSPD1')
def require_adoption(value):
    if value.get('status')!='ADOPTED' or value.get('plan')!=PLAN:
        raise ValueError('root physical entry adoption required')
def sparse_copy(matrix):
    c=matrix.tocoo()
    return dict(shape=list(c.shape),row=plain(c.row),col=plain(c.col),data=plain(c.data))
def linear_residual(row):
    m=row['bordered'];x=row['solution'];rhs=row['rhs'];terms=[[] for _ in rhs]
    for i,j,v in zip(m['row'],m['col'],m['data']):terms[i].append(v*x[j])
    residual=[math.fsum(t)-v for t,v in zip(terms,rhs)]
    norm=math.hypot(*residual);den=math.hypot(*rhs)
    return dict(residual=residual,absolute_norm=norm,relative_norm=norm/den if den else None,
                rhs_norm=den,normalization='actual RHS norm; no floor or acceptance cutoff',
                interpretation='raw Euclidean residual of mixed-unit bordered rows; diagnostic only, not a scientific acceptance metric')
def predicate(value,old):
    return dict(trial_merit=value,current_merit=old,strict_decrease=value<old)
def selected_fields(locals_,names):
    return dict(values={k:plain(locals_[k]) for k in names if k in locals_},
                unavailable=[k for k in names if k not in locals_])
def require_expected_refusal(error,recognized_type,flags):
    if not isinstance(error,recognized_type):raise error
    if flags is None or len(flags)!=21 or not all(flags.values()):
        raise ValueError('all21 immediate fixed-call atomic flags required')

class Trace:
    """Read-only profile/exception observation of original production frames."""
    def __init__(self,out,atomic=None):
        self.out=Path(out);self.atomic=atomic;self.evaluations=0;self.solves=0;self.locals=[];self.material=[]
        self.eval_ids={};self.fixed_before=None;self.fixed_owner=None;self.fixed_error=None;self.fixed_flags=None
    def emit(self,kind,**value):
        with (self.out/'trace.jsonl').open('a') as s:s.write(json.dumps(plain(dict(kind=kind,**value)),allow_nan=False)+'\n')
    def __enter__(self):
        self.previous_profile=sys.getprofile();self.previous_trace=sys.gettrace()
        if self.previous_profile is not None or self.previous_trace is not None:raise ValueError('existing tracing hook refused')
        sys.settrace(self.exceptions);sys.setprofile(self.observe);return self
    def __exit__(self,*args):
        sys.setprofile(self.previous_profile);sys.settrace(self.previous_trace)
    def exceptions(self,frame,event,arg):
        file=frame.f_code.co_filename
        if not file.endswith(('_stiffened_shell_displacement_corrector.py','_stiffened_shell_displacement_control.py')):return None
        if event=='exception':
            type_,error,_=arg;f=frame.f_locals
            self.emit('exception',function=frame.f_code.co_name,line=frame.f_lineno,type=type_.__name__,message=str(error),
                      context=selected_fields(f,('iteration','backtrack','amount','amplitude','value','old')))
            if file.endswith('_stiffened_shell_displacement_control.py') and frame.f_code.co_name=='solve':
                self.fixed_error=dict(type=type_.__name__,message=str(error),detail=plain(getattr(error,'detail',None)))
        return self.exceptions
    def observe(self,frame,event,arg):
        code=frame.f_code;f=frame.f_locals;file=code.co_filename;name=code.co_name
        if event=='call' and file.endswith('_stiffened_shell_displacement_control.py') and name=='solve':
            self.fixed_owner=f['owner'];self.fixed_before=self.atomic.capture(self.fixed_owner)
            self.emit('fixed_entry',generation=self.fixed_owner._generation,targets=plain(f['targets']))
        if event!='return':return
        if file.endswith('_stiffened_shell_displacement_control.py') and name=='solve' and self.fixed_before is not None:
            self.fixed_flags=self.atomic.atomic_flags(self.fixed_owner,self.fixed_before)
            self.emit('fixed_return',returned=arg is not None,flags=self.fixed_flags,exception=self.fixed_error)
        if file.endswith('plasticity.py') and name=='plane_stress_return_map':
            self.material.append(dict(kind='J2',return_fields=selected_fields(f,('strain','plastic_strain','alpha','new_plastic','new_alpha',
                'sigma','converged','scaled_residual','fallback_mask','method','tolerance')),
                curve=selected_fields(vars(f['curve']) if hasattr(f.get('curve'),'__dict__') else {},
                    ('sigma_prop','sigma_yield','sigma_ultimate','epsilon_prop','epsilon_yield','epsilon_ultimate','E','n'))))
        if file.endswith('plasticity.py') and name.startswith('_jit_plane_stress'):
            # Exact executed internal masks/scalars, with names preserved. No
            # component-sign branch classification or invented missing marker.
            values={k:plain(v) for k,v in f.items() if hasattr(v,'tolist') or isinstance(v,(bool,int,float))}
            self.material.append(dict(kind='J2_executed_internal_selectors',function=name,values=values))
        if file.endswith('_stiffened_shell_q4_algorithmic_pullback.py') and name=='response':
            self.emit('Q4_actual_maps',maps=selected_fields(f,('u','local','differential','second','force','tangent','residual','spatial_tangent','rotation_anchors')))
        if name=='scatter' and frame.f_back is not None and frame.f_back.f_code.co_name=='_assemble':
            parent=frame.f_back.f_locals;owner=parent['owner'];ordinal=len(self.locals)
            item=dict(slots=plain(f['slots']),force=plain(f['force']),tangent=plain(f['tangent']))
            if ordinal<len(owner._shell_parts):item.update(kind='shell',candidate=plain(parent['candidate']))
            elif ordinal<len(owner._shell_parts)+len(owner._beam_parts):
                response=parent['response'];item.update(kind='native',material=json.loads(response.material),
                    residual=plain(response.residual),hessian=plain(response.hessian),hessian_low=plain(response.hessian_low),
                    spatial_tangent=plain(parent['tangent']))
            else:
                trial=parent['trial'];item.update(kind='POINT',coordinate_map=plain(trial.coordinate_map),
                    spatial_jacobian=plain(trial.pose.spatial_jacobian),constraint_jacobian=plain(trial.pose.constraint_jacobian))
            self.locals.append(item)
        if file.endswith('_stiffened_shell_ge_finite_experimental.py') and name=='_assemble':
            self.emit('assembly',local_contributions=self.locals,constitutive=self.material,
                      predecessor_shell_id=id(f.get('shell_origins')),predecessor_beam_id=id(f.get('beam_origins')))
            self.locals=[];self.material=[]
        if not file.endswith('_stiffened_shell_displacement_corrector.py'):return
        parent=frame.f_back.f_locals if frame.f_back is not None else {}
        if name=='evaluate':
            self.evaluations+=1
            if arg is not None:
                self.eval_ids[id(arg)]=self.evaluations;state=f['trial']
                self.emit('evaluation',evaluation=self.evaluations,value=f['value'],equilibrium=arg[2],gap=arg[3],
                    residual=plain(arg[0]),tangent=sparse_copy(arg[1]),state=selected_fields(
                    {k:getattr(state,k) for k in ('q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers','shell_anchors') if hasattr(state,k)},
                    ('q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers','shell_anchors')))
            else:self.emit('evaluation_declined',evaluation=self.evaluations,value=f.get('value'))
        elif name=='merit' and arg is not None:
            trial=f['trial'] is not parent.get('state');record=dict(value=arg,role='trial' if trial else 'current',
                evaluation=self.eval_ids.get(id(f['evaluation'])),iteration=parent.get('iteration'),
                backtrack=parent.get('backtrack'),amount=parent.get('amount'),amplitude=parent.get('value') if trial else parent.get('amplitude'))
            if trial and 'old' in parent:record['decision_predicate']=predicate(arg,parent['old'])
            self.emit('merit',**record)
        elif name=='bordered_path_correction' and arg is not None:
            self.solves+=1;free=f['free'];rhs=[-float(f['residual'][i]) for i in free]+[-float(f['gap'])]
            self.emit('bordered_solve',solve=self.solves,bordered=sparse_copy(f['bordered']),solution=plain(f['step']),rhs=rhs,
                full_correction=plain(arg[0]),amplitude_delta=arg[1],free=plain(free),gradient=plain(f['gradient']),
                direction=plain(f['direction']),matrix=sparse_copy(f['matrix']),iteration=parent.get('iteration'))

def invoke_once(owner,actions,control,trace,out,atomic):
    before=atomic.capture(owner);result=dict(status='STARTED',accepted=False)
    try:
        with trace:
            value=owner.solve(actions,(TARGET,),displacement_control=control,max_iterations=20,max_backtracks=8)
        result.update(status='RETURNED',accepted=True,physical=plain(vars(value)))
    except BaseException as error:
        result.update(status='FAILED',exception=dict(type=type(error).__name__,message=str(error),detail=plain(getattr(error,'detail',None))),
                      public_flags=atomic.atomic_flags(owner,before),fixed_flags=trace.fixed_flags)
        raise
    finally:
        result.update(observed_evaluations=trace.evaluations,observed_linear_solves=trace.solves)
        write(Path(out)/'outcome.json',result)
        (Path(out)/'terminal.bin').write_bytes(owner.checkpoint());write(Path(out)/'terminal-recovery.json',owner.recover())

def artifact():
    expected=json.loads((PACKAGE/'snapshot-bindings.json').read_text());wheel=next((PACKAGE/'wheels').glob('*.whl'))
    if len(expected)!=634 or sha(wheel)!=WHEEL:raise ValueError('exact installed efc artifact required')
    with zipfile.ZipFile(wheel) as z:
        for rel,h in expected.items():
            entry=Path(rel).relative_to('src').as_posix()
            if sha(PACKAGE/'snapshot'/rel)!=h or sha(PACKAGE/'site'/entry)!=h or hashlib.sha256(z.read(entry)).hexdigest()!=h:
                raise ValueError('installed artifact source closure changed: '+rel)
    return expected
def bindings():
    p=DOC/'DIAGNOSTIC_BINDINGS.json';expected=json.loads(p.read_text())
    if len({Path(k).name for k in expected})!=len(expected):raise ValueError('snapshot basename collision')
    for rel,h in expected.items():
        path=ROOT/rel
        if not path.resolve().is_relative_to(ROOT) or sha(path)!=h:raise ValueError('changed binding: '+rel)
    return expected
def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0])
    if json.loads((out/'bindings.json').read_text())!=bindings():raise ValueError('changed child binding')
    selected=PACKAGE/'site';sys.path[:0]=[str(selected),str(ROOT/'tests')]
    os.environ.update(GE_FIXED_DIAGNOSTIC_OUTPUT=str(out),GE_FIXED_DIAGNOSTIC_STAGE=stage)
    from scripts.r8_f06_log_domain_check_dev import runtime
    runtime(out,'runtime-before',selected)
    import pytest
    names=SOURCE_NAMES if stage=='source-controls' else ('test_installed_one_fixed_diagnostic',)
    try:
        code=int(pytest.main([str(ROOT/'tests/test_r8_ge_paired_fixed_diagnostic_dev.py')+'::'+n for n in names]+
            ['-q']+(['-x'] if stage=='installed-fixed' else [])+['-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
        from scripts.r8_ge_coupled_cycle_dev import counts
        if code==0:write(out/'counts.json',counts(out/'junit.xml',len(names)))
        return code
    finally:
        rt=runtime(out,'runtime-after',selected);tree=artifact()
        for _,path in rt['imports'].items():
            if sha(path)!=tree['src/'+Path(path).relative_to(selected).as_posix()]:raise ValueError('actual installed import drift')
def connected(out):
    if out!=DOC/'attempt-installed-fixed1' or os.environ.get('GE_FIXED_DIAGNOSTIC_STAGE')!='installed-fixed':
        raise ValueError('sole exact installed diagnostic output required')
    adoption=json.loads((DOC/'DIAGNOSTIC_ADOPTION.json').read_text())
    require_adoption(adoption)
    raw,outer,fixed=select_inner(OUTER.read_bytes())
    if raw!=INNER.read_bytes():raise ValueError('inner checkpoint is not exact saved mechanical prefix')
    artifact()
    from anyfesolver import stiffened_shell as ss
    from scripts.r8_f06_cyclic_material_dev import definition,actions
    from scripts import r8_f06_paired_active_tangent_dev as atomic
    from anyfesolver._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure
    owner=ss.create_analysis(definition=definition('coupled'))
    try:
        owner.restore(raw)
        if owner.checkpoint()!=raw or owner._generation!=11:raise ValueError('public fixed restore identity mismatch')
        write(out/'restored-entry.json',dict(fixed_header=fixed,outer_header=outer,recovery=owner.recover(),
            cursor_difference='disposable fixed owner has no pending adaptive cursor; archived outer unchanged'))
        control=ss.DisplacementControl('shell',2,0,force_seed_sha256=SEED)
        trace=Trace(out,atomic)
        try:invoke_once(owner,actions('coupled'),control,trace,out,atomic)
        except DisplacementCorrectorFailure as error:
            require_expected_refusal(error,DisplacementCorrectorFailure,trace.fixed_flags)
            if owner.checkpoint()!=raw or owner._generation!=11:
                raise ValueError('expected refused correction changed fixed predecessor')
            detail=error.detail
            if detail is None or detail.target!=TARGET or detail.accepted_generation!=11:
                raise ValueError('actual same-target refusal detail required')
            if trace.evaluations!=detail.evaluations or trace.solves!=detail.linear_solves:
                raise ValueError('observed evaluation/linear-solve counts disagree with actual failure detail')
            report=json.loads((out/'outcome.json').read_text())
            report.update(status='EXPECTED_FIXED_REFUSAL_DIAGNOSTICS_CAPTURED',physical_target_accepted=False,
                          diagnostic_capture=True,scientific_acceptance=False)
            write(out/'outcome.json',report)
    finally:
        try:
            path=out/'trace.jsonl'
            if path.exists():
                rows=[json.loads(x) for x in path.read_text().splitlines()]
                write(out/'linear-residuals.json',[dict(solve=x['solve'],**linear_residual(x)) for x in rows if x['kind']=='bordered_solve'])
        finally:owner.close()
def launch(stage):
    from scripts.r8_f06_fibre_log_domain_dev import run
    auth='DIAGNOSTIC_'+('SOURCE' if stage=='source-controls' else 'INSTALLED')+'_AUTHORITY.json'
    path=DOC/auth;captured=path.read_bytes();grant=json.loads(captured)
    slot=json.loads((DOC/'DIAGNOSTIC_SLOT_TRANSFER.json').read_text())
    if grant.get('status')!='AUTHORIZED' or slot.get('status')!='TRANSFERRED' or slot.get('one_numerical_job') is not True:
        raise ValueError('fresh root authority/slot required')
    if (sha(DOC/'DIAGNOSTIC_ADOPTION.json')!=grant.get('adoption_sha256') or
        sha(DOC/'DIAGNOSTIC_SLOT_TRANSFER.json')!=grant.get('slot_sha256')):
        raise ValueError('exact root diagnostic adoption/slot identities required')
    if (grant.get('budget_s'),grant.get('reserve_s'),grant.get('process_cap_s'))!=(180,60,20 if stage=='source-controls' else 90):
        raise ValueError('exact fresh diagnostic resource envelope required')
    if sha(DOC/'DIAGNOSTIC_BINDINGS.json')!=grant.get('bindings_sha256'):raise ValueError('exact prospective bindings required')
    if list(DOC.glob('attempt-'+stage+'*')):raise ValueError('sole attempt consumed')
    if stage=='installed-fixed':
        c=json.loads((DOC/'attempt-source-controls1/counts.json').read_text());r=json.loads((DOC/'attempt-source-controls1/receipt.json').read_text())
        if c!=dict(tests=8,errors=0,failures=0,skipped=0) or r['status']!='PASS':raise ValueError('exact8 source PASS required')
    out=DOC/('attempt-'+stage+'1')
    try:
        code=run(stage,doc=DOC,entry=__file__,bind=bindings,authority_file=auth)
        (out/'used-authority.json').write_bytes(captured)
        if hashlib.sha256(captured).hexdigest()!=json.loads((out/'receipt.json').read_text())['authority_sha256']:
            raise ValueError('authority changed during launch')
        return code
    finally:
        if out.exists() and (stage=='installed-fixed' or json.loads((out/'receipt.json').read_text()).get('status')!='PASS'):
            p=DOC/'ledger.json';v=json.loads(p.read_text());v.update(status='CLOSED',unused_discarded_s=v['budget_s']-v['used_s']);write(p,v)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('source-controls','installed-fixed'),required=True);p.add_argument('--child',action='store_true');p.add_argument('--out');a=p.parse_args()
    raise SystemExit(child(Path(a.out),a.stage) if a.child else launch(a.stage))
