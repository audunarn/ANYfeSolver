"""Prepared connected public GE cycle; explicit fresh authority required to run."""
from pathlib import Path
from decimal import Decimal, localcontext
from dataclasses import asdict, is_dataclass
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/agent_plans/development/r8-ge-coupled-cycle-20261010'
BASE=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/paired-displacement-v1'
LOADED=BASE/'loaded-x-v1'
PACKAGE=BASE/'package'
ACCEPTED=LOADED/'attempt-installed-history1/accepted.bin'
REBASED=LOADED/'attempt-installed-controls1/rebased.bin'
SEED=ROOT/'docs/agent_plans/development/r8-f06-log-domain-20261010/mixed-adaptive-v1/attempt-installed-resume1/terminal.bin'
IDENTITIES={ACCEPTED:'cbd6cddee612a1b2734b3a643f8f98b6b89391fc33247c4613b36b2fa05b4786',
    REBASED:'a824248c93cc8f2c91839d8a0b973ef9e5afc4ba99aaab925b7660aa32fe9a30',
    SEED:'970e0360f22ed348ca2eaa7eee0d2d168d6d686d0a91a750f3ccd31794865f91'}
TARGET=0.028578804694929322
PLAN=dict(control=['shell',2,0],control_dof=6,reference_translation=0.,loaded_target=TARGET,
    targets=[TARGET/2.,0.,TARGET/2.,TARGET],max_iterations=20,max_backtracks=8,
    rejected_probe_iterations=1,rejected_probe_from='completed-cycle-final',rejected_probe_target=TARGET/2.,adaptive=False,trial_policy=None,
    equilibrium_limit=1e-8,constraint_limit=1e-9,point_work_absolute_limit=1e-12,
    chart_invariant_limit=1e-11,monotonic_accumulation=True,
    primary_accepts=4,cold_continuation_accepts=2,rebased_continuation_accepts=2)
SOURCE_NAMES=('test_source_exact_path_and_limits','test_source_changed_adoption_refused',
    'test_source_binding_drift_refused','test_source_exact_junit_counts',
    'test_source_finite_acceptance_and_work_refusal','test_source_pair_monotonicity',
    'test_source_physical_reversal_reporting','test_source_production_public_sequence')
ATOMIC_KEYS=('checkpoint','generation','objects','state_object','state_words',
    'accepted_object','controller','driver_object','driver_accepted_object',
    'driver_accepted_words','applied_load_object','applied_load_bytes','snapshot_object',
    'shell_accepted_objects','shell_accepted_words','shell_predecessors_objects',
    'shell_predecessors_words','beam_accepted_objects','beam_accepted_words',
    'beam_predecessors_objects','beam_predecessors_words')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plain(value):
    if is_dataclass(value):return plain(asdict(value))
    if hasattr(value,'tolist'):return plain(value.tolist())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(x) for x in value]
    return value

def write(path,value):
    Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')

def require_adoption(value):
    if value.get('status')!='CYCLE_ADOPTED' or value.get('plan')!=PLAN:
        raise ValueError('root must adopt exact physical sequence and criteria')
    return PLAN

def verify_bindings(expected,actual):
    if expected!=actual or len({Path(p).name for p in actual})!=len(actual):
        raise ValueError('changed bindings or colliding snapshot basenames')

def counts(path,expected):
    suites=ET.parse(path).getroot().findall('testsuite')
    result={k:sum(int(x.get(k,'0')) for x in suites) for k in ('tests','errors','failures','skipped')}
    if result!=dict(tests=expected,errors=0,failures=0,skipped=0):
        raise ValueError('incomplete explicit control inventory: '+str(result))
    return result

def pair(value):
    if len(value)!=2 or any(not math.isfinite(float(x)) for x in value):
        raise ValueError('complete finite scalar high/low pair required')
    with localcontext() as context:
        context.prec=80
        return sum((Decimal.from_float(float(x)) for x in value),Decimal(0))

def monotone(before,after):
    if set(before)!=set(after) or any(after[k]<before[k] for k in before):
        raise ValueError('accumulated history decreased or changed layout')
    return True

def validate_metrics(value):
    fields=('equilibrium','gap','target','amplitude','point_work_max')
    if any(not math.isfinite(value[k]) for k in fields):raise ValueError('nonfinite physical endpoint')
    if value['equilibrium']>PLAN['equilibrium_limit'] or value['gap']>PLAN['constraint_limit']:
        raise ValueError('unchanged physical acceptance criterion failed')
    if value['point_work_max']>PLAN['point_work_absolute_limit'] or value['point_count']!=4:
        raise ValueError('POINT work identity failed or incomplete')
    if not value['constitutive_valid'] or not value['monotonic']:
        raise ValueError('constitutive/history acceptance unresolved')

def reversal(records):
    """Only observed physical uniaxial fibre flow qualifies as signed reversal."""
    signs={}
    for row in records:
        for key,sign in row.items():
            if sign:signs.setdefault(key,set()).add(sign)
    return sorted(key for key,values in signs.items() if values=={-1,1})

def check_atomic(flags):
    if set(flags)!=set(ATOMIC_KEYS) or not all(type(x) is bool and x for x in flags.values()):
        raise AssertionError('incomplete or false atomic flags')

def execute_targets(owner,loads,control,targets,record,*,offset=0,limits=None):
    """Production caller shared by installed cycle and fake-owner controls."""
    limits=PLAN if limits is None else limits
    for index,target in enumerate(targets,offset):
        observed=[]
        result=owner.solve(loads,(target,),displacement_control=control,
            max_iterations=limits['max_iterations'],max_backtracks=limits['max_backtracks'],
            on_accepted=lambda r:observed.append(r))
        def payload(value):return plain(value) if is_dataclass(value) else plain(vars(value))
        if len(observed)!=1 or payload(observed[0])!=payload(result):
            raise ValueError('exactly one public accepted callback required')
        record(owner,index,result)
    return owner

def artifact():
    import zipfile
    frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
    wheel=next((PACKAGE/'wheels').glob('*.whl'))
    if sha(wheel)!='0ed73724bf3c8ccffcc0134628d94720d640840f9592ac8890779f54285ee3d8':
        raise ValueError('changed immutable wheel')
    delta=[]
    with zipfile.ZipFile(wheel) as archive:
        for rel,digest in frozen.items():
            name=Path(rel).relative_to('src').as_posix()
            if not sha(PACKAGE/'snapshot'/rel)==sha(PACKAGE/'site'/name)==hashlib.sha256(archive.read(name)).hexdigest()==digest:
                raise ValueError('frozen snapshot/site/wheel drift: '+rel)
            current=sha(ROOT/rel)
            if current!=digest:delta.append(dict(path=rel,old_sha256=digest,new_sha256=current))
    decision=json.loads((DOC/'CYCLE_ARTIFACT_REUSE_DECISION.json').read_text())
    if decision.get('status')!='ARTIFACT_REUSE_ADOPTED' or decision.get('current_source_deltas')!=delta:
        raise ValueError('exact current source delta adoption required')
    if decision.get('wheel_sha256')!=sha(wheel) or decision.get('manifest_sha256')!=sha(PACKAGE/'snapshot-bindings.json'):
        raise ValueError('changed artifact decision identity')
    if any(x['path']!='src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py' for x in delta) or not decision.get('independence_justification'):
        raise ValueError('GE/current source closure drift')
    return dict(files=len(frozen),wheel_sha256=sha(wheel),manifest_sha256=sha(PACKAGE/'snapshot-bindings.json'),current_source_deltas=delta)

def bindings():
    helpers=('r8_f06_cyclic_material_dev.py','r8_f06_native_material_breadth_dev.py',
        'r8_f06_fibre_log_domain_dev.py','r8_f06_log_domain_check_dev.py','r8_f06_paired_active_tangent_dev.py')
    products=('_stiffened_shell_displacement_control.py','_stiffened_shell_displacement_corrector.py',
        '_stiffened_shell_displacement_checkpoint.py','_stiffened_shell_displacement_adaptive.py',
        '_stiffened_shell_ge_experimental.py','_stiffened_shell_ge_finite_experimental.py',
        '_stiffened_shell_generalized_recovery.py','_stiffened_shell_beam_history.py',
        '_stiffened_shell_algorithmic_pullback.py','_stiffened_shell_q4_algorithmic_pullback.py',
        '_stiffened_shell_point_recovery.py')
    paths=[Path(__file__),ROOT/'tests/test_r8_ge_coupled_cycle_dev.py',ROOT/'tests/conftest.py',ROOT/'examples/r8_ge_coupled_cycle_public_example.py',
        *IDENTITIES,LOADED/'TARGET_ADOPTION.json',PACKAGE/'snapshot-bindings.json',DOC/'PREPARATION.md',DOC/'PROSPECTIVE_PLAN.json']
    paths+=[ROOT/'scripts'/x for x in helpers]+[ROOT/'src/anyfesolver'/x for x in products]
    paths += [DOC/x for x in ('CYCLE_ADOPTION.json','CYCLE_ARTIFACT_REUSE_DECISION.json',
        'CYCLE_SOURCE_AUTHORITY.json','CYCLE_INSTALLED_AUTHORITY.json','CYCLE_SLOT_TRANSFER.json')]
    result={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    verify_bindings(result,result)
    return result

class PublicObserver:
    """Copy actual scatter/return metadata during public calls, no new evaluation."""
    def __init__(self,out):self.out=out;self.calls=0;self.latest={};self.rows=[];self.shell=[]
    def __enter__(self):
        self.previous=sys.getprofile();sys.setprofile(self.observe);return self
    def __exit__(self,*args):sys.setprofile(self.previous)
    def observe(self,frame,event,arg):
        if event!='return':return
        code=frame.f_code;f=frame.f_locals
        if code.co_name=='plane_stress_return_map' and code.co_filename.endswith('plasticity.py'):
            self.shell.append(plain({k:copy.deepcopy(f[k]) for k in ('converged','scaled_residual','fallback_mask','method','tolerance') if k in f}))
        if code.co_name=='scatter' and frame.f_back.f_code.co_name=='_assemble':
            parent=frame.f_back.f_locals
            owner=parent['owner'];ordinal=len(self.rows)
            if ordinal<len(owner._shell_parts):self.rows.append(dict(kind='shell',candidate=plain(copy.deepcopy(parent['candidate']))))
            elif ordinal<len(owner._shell_parts)+len(owner._beam_parts):self.rows.append(dict(kind='native',material=json.loads(parent['response'].material)))
            else:
                import numpy as np
                trial=parent['trial'];force=f['force'];local=np.arange(1,len(force)+1,dtype=float)*1e-4
                mapped=trial.coordinate_map@local
                a=float(force[:12]@mapped[:12]);b=float(parent['state'].multipliers[6*len([x for x in self.rows if x['kind']=='point']):6*(1+len([x for x in self.rows if x['kind']=='point']))]@(trial.pose.constraint_jacobian[:,:12]@mapped[:12]))
                self.rows.append(dict(kind='point',primal_work=a,constraint_work=b,absolute_error=abs(a-b)))
        if code.co_name=='_assemble' and code.co_filename.endswith('_stiffened_shell_ge_finite_experimental.py'):
            self.calls+=1
            self.latest=dict(assembly=self.calls,contributions=self.rows,shell_metadata=self.shell,
                predecessor_shell_id=id(f.get('shell_origins')),predecessor_beam_id=id(f.get('beam_origins')))
            with (self.out/'actual-assemblies.jsonl').open('a') as stream:stream.write(json.dumps(self.latest,allow_nan=False)+'\n')
            self.rows=[];self.shell=[]

def accumulation(owner):
    result={}
    d=owner._finite_driver
    for i,row in enumerate(d._shell_origins):
        for j,x in enumerate(row['alpha']):result[f'shell/{i}/{j}']=Decimal.from_float(float(x))
    for i,history in enumerate(d._beam_origins):
        for j,station in enumerate(history.stations):
            if hasattr(station,'rows'):
                for k,row in enumerate(station.rows):result[f'fibre/{i}/{j}/{k}']=pair(row[2:])
            else:result[f'generalized/{i}/{j}']=pair(station.accumulated)
    return result

def material_summary(result):
    fibre={};general=[];dissipation=[]
    for station in result.beam_fibre_material:
        for index,field in enumerate(station.fibres):
            field=plain(field);inc=pair(field['plastic_increment'])
            fibre[f'{station.element_id}/{station.station}/{index}']=int(inc>0)-int(inc<0)
    for item in result.beam_fibre_energy:dissipation.append(pair(item.incremental_dissipation_j))
    for item in result.beam_generalized_material:
        for row in item.stations:
            general.append(dict(element=item.element_id,station=row.station,branch=row.branch,increment=plain(row.plastic_increment)))
            dissipation.append(pair(row.dissipation_per_length))
    if not fibre or not general or not dissipation or any(x<0 for x in dissipation):
        raise ValueError('incomplete material recovery or negative valid incremental dissipation')
    return dict(fibre_flow_signs=fibre,generalized=general,dissipation=[str(x) for x in dissipation],
        dissipation_units='fibre integrated J; generalized per reference length J/m; last accepted increment only')

def numeric_layout(value):
    if isinstance(value,(list,tuple)):
        return tuple(numeric_layout(x) for x in value)
    if isinstance(value,(bool,str)) or not isinstance(value,(float,int)) or not math.isfinite(value):
        raise ValueError('finite physical numeric payload required')
    return None

def numeric_values(value):
    if isinstance(value,(list,tuple)):
        return [x for row in value for x in numeric_values(row)]
    return [float(value)]

def compare_physical_values(a,b):
    """Existing norm/max(1,norm(a),norm(b)) invariant, with exact layouts."""
    if set(a)!=set(b):raise ValueError('physical recovery fields differ')
    result={}
    for key in a:
        if numeric_layout(a[key])!=numeric_layout(b[key]):raise ValueError('physical recovery layout differs: '+key)
        x,y=numeric_values(a[key]),numeric_values(b[key])
        # Layout and lengths are checked before zip; omitted rows cannot pass.
        if len(x)!=len(y):raise ValueError('physical recovery length differs: '+key)
        norm=lambda values:math.sqrt(math.fsum(v*v for v in values))
        relative=norm([i-j for i,j in zip(x,y)])/max(1.,norm(x),norm(y))
        result[key]=relative
        if not math.isfinite(relative) or relative>PLAN['chart_invariant_limit']:
            raise ValueError('physical chart/snapshot continuation parity: '+key)
    return result

def physical_payload(result):
    keys=('shell_frames','beam_positions','beam_frames','cell_frames','cell_resultants','multipliers','applied_dead_load','support_reactions')
    payload={key:plain(getattr(result,key)) for key in keys}
    displacement=plain(result.shell_displacement)
    if len(displacement)%6:raise ValueError('complete shell displacement rows required')
    payload['shell_translations']=[displacement[start:start+3] for start in range(0,len(displacement),6)]
    # Q4 algorithmic and native/POINT assembly rows are already spatial
    # wrenches. prepare() selects support residual rows without chart pullback.
    # See source spans in PREPARATION; no second/ad-hoc dual transform belongs here.
    points=plain(result.point_wrenches)
    if len(points)!=4:raise ValueError('all four physical POINT recovery records required')
    identities=[]
    for row in points:
        identity=(row['point_index'],row['shell_node_id'],row['beam_node_id'],row['configuration'])
        identities.append(identity)
        name='point/'+str(row['point_index'])+'/'
        for key,size in (('shell_position_m',3),('beam_position_m',3),('shell_position_low_m',3),
                ('beam_position_low_m',3),('shell_action',6),('beam_action',6),('gap',6)):
            if len(row[key])!=size:raise ValueError('physical POINT field length differs')
            payload[name+key]=row[key]
    if len(set(identities))!=4 or {x[0] for x in identities}!=set(range(4)):
        raise ValueError('physical POINT identity layout incomplete')
    return payload,identities

def parity(a,b):
    x,ix=physical_payload(a);y,iy=physical_payload(b)
    if ix!=iy:raise ValueError('physical POINT incidence/configuration differs')
    return compare_physical_values(x,y)

def history_parity(a,b,*,exact):
    """Compare every accepted and fixed-predecessor history, preserving pairs."""
    import numpy as np
    aa=a._displacement_snapshot[2];bb=b._displacement_snapshot[2]
    names=('beam_plastic','beam_accumulated','beam_fibres','beam_origin_plastic',
        'beam_origin_accumulated','beam_origin_fibres','shell_plastic','shell_alpha',
        'shell_origin_plastic','shell_origin_alpha')
    result={}
    for name in names:
        x,y=aa[name],bb[name]
        if x.shape!=y.shape:raise ValueError('all-history array shape differs: '+name)
        words_equal=np.array_equal(x,y)
        if exact and not words_equal:raise ValueError('cold all-history words differ: '+name)
        if 'fibres' in name:
            px=[pair(row[:2]) for row in x]+[pair(row[2:]) for row in x]
            py=[pair(row[:2]) for row in y]+[pair(row[2:]) for row in y]
        elif name.startswith('beam_'):
            px=[pair(row) for row in x.reshape(-1,2)];py=[pair(row) for row in y.reshape(-1,2)]
        else:px=[Decimal.from_float(float(z)) for z in x.ravel()];py=[Decimal.from_float(float(z)) for z in y.ravel()]
        if len(px)!=len(py):raise ValueError('all-history pair layout differs: '+name)
        error=max((abs(i-j) for i,j in zip(px,py)),default=Decimal(0))
        scale=max([Decimal(1)]+list(map(abs,px))+list(map(abs,py)))
        relative=float(error/scale)
        result[name]=dict(words_equal=words_equal,physical_pair_invariant=relative)
        if relative>PLAN['chart_invariant_limit']:raise ValueError('chart all-history invariant failed: '+name)
    return result

def connected(out):
    if os.environ.get('GE_CYCLE_STAGE')!='installed-cycle' or os.environ.get('GE_CYCLE_OUTPUT')!=str(out):
        raise ValueError('connected cycle is installed-child-only')
    import numpy as np
    from scripts import r8_f06_paired_active_tangent_dev as atomic
    from scripts.r8_f06_cyclic_material_dev import definition,actions
    from anyfesolver import stiffened_shell as ss
    from anyfesolver.control import CancellationToken,SolveCancelled
    adoption=json.loads((DOC/'CYCLE_ADOPTION.json').read_text());require_adoption(adoption)
    for path,digest in IDENTITIES.items():
        if sha(path)!=digest:raise ValueError('changed saved endpoint')
    report=dict(status='INCOMPLETE',plan=PLAN,accepted=[],atomic=[],parity={},reversal=[],independent_review='Mistral pending',acceptance=False)
    owners=[];primary=None
    def make(raw):
        owner=ss.create_analysis(definition=definition('coupled'));owners.append(owner);owner.restore(raw);return owner
    def save(owner,label):
        (out/(label+'.bin')).write_bytes(owner.checkpoint())
        write(out/(label+'-recovery.json'),owner.recover())
    def atom(owner,label,before,*,require_unchanged=True):
        flags=atomic.atomic_flags(owner,before);report['atomic'].append(dict(label=label,flags=flags));write(out/'cycle.json',report)
        if require_unchanged:check_atomic(flags)
    class Cancel(CancellationToken):
        def __init__(self,stage,visit=1):super().__init__();self.stage=stage;self.visit=visit;self.count=0
        def raise_if_cancelled(self,stage=''):
            if stage==self.stage:
                self.count+=1
                if self.count==self.visit:self.cancel('cycle direct safe-point control')
            super().raise_if_cancelled(stage)
    try:
        primary=make(ACCEPTED.read_bytes());save(primary,'loaded')
        if primary._definition.q4_pullback_policy!='STIFFENED_SHELL_Q4_ALGORITHMIC_PULLBACK_V1':
            raise ValueError('declared physical spatial-row recovery route required')
        initial_chart=make(REBASED.read_bytes());report['parity']['saved_rebase']=parity(primary.recover(),initial_chart.recover())
        loads=actions('coupled');control=ss.DisplacementControl('shell',2,0,force_seed_sha256=IDENTITIES[SEED])
        report['initial_history']=plain(primary.recover().history)
        if primary.recover().target!=TARGET or primary._generation!=9 or initial_chart._generation!=10:
            raise ValueError('saved generation/coordinate mismatch')
        with PublicObserver(out) as observer:
            previous={id(primary):accumulation(primary)}
            def record(owner,index,result):
                label=('direct' if owner is primary else 'cold' if owner is cold else 'chart')+'-'+str(index)
                save(owner,label);write(out/(label+'-actual-local.json'),observer.latest)
                before=previous[id(owner)];after=accumulation(owner)
                mono=monotone(before,after);previous[id(owner)]=after
                points=[x for x in observer.latest.get('contributions',[]) if x['kind']=='point']
                shell=observer.latest.get('shell_metadata',[])
                summary=material_summary(result)
                row=dict(label=label,target=result.target,amplitude=result.amplitude,generation=result.generation,
                    history=plain(result.history),equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,
                    point_work_max=max((x['absolute_error'] for x in points),default=float('inf')),point_count=len(points),
                    monotonic=mono,constitutive_valid=bool(shell) and all(all(x['converged']) for x in shell),
                    materials=summary,accumulated={k:str(v) for k,v in after.items()})
                total=np.asarray(result.applied_dead_load)+np.asarray(result.support_reactions)
                nodal=owner._shell_count+6*len(owner._beam_ids)
                row['actual_applied_translation_resultant']=np.asarray(result.applied_dead_load)[:nodal].reshape(-1,6)[:,:3].sum(axis=0).tolist()
                row['actual_support_translation_resultant']=np.asarray(result.support_reactions)[:nodal].reshape(-1,6)[:,:3].sum(axis=0).tolist()
                row['translation_force_balance']=total[:nodal].reshape(-1,6)[:,:3].sum(axis=0).tolist()
                report['accepted'].append(row);write(out/'cycle.json',report);validate_metrics(row)
                expected=len(report['initial_history'])+index+1
                if len(result.history)!=expected or result.target!=PLAN['targets'][index]:raise ValueError('controller cursor/target mismatch')
            cold=chart=None
            execute_targets(primary,loads,control,PLAN['targets'][:2],record)
            zero=primary.checkpoint();save(primary,'zero-prefix');before=atomic.capture(primary)
            for label,target,kwargs,error in (
                ('nonfinite',float('nan'),{},ValueError),('cancel-trial1',PLAN['targets'][2],dict(cancellation_token=Cancel('stiffened-shell.displacement.trial')),SolveCancelled),
                ('cancel-trial2',PLAN['targets'][2],dict(cancellation_token=Cancel('stiffened-shell.displacement.trial',2)),SolveCancelled),
                ('cancel-publish',PLAN['targets'][2],dict(cancellation_token=Cancel('stiffened-shell.displacement.publish')),SolveCancelled)):
                try:primary.solve(loads,(target,),displacement_control=control,max_iterations=20,max_backtracks=8,**kwargs)
                except error as exc:write(out/(label+'-refusal.json'),dict(type=type(exc).__name__,message=str(exc)))
                else:raise AssertionError('expected direct refusal/cancellation absent: '+label)
                finally:save(primary,label+'-terminal');atom(primary,label,before)
            cold=make(zero);chart=make(zero);previous[id(cold)]=accumulation(cold);previous[id(chart)]=accumulation(chart)
            if cold.checkpoint()!=zero or chart.checkpoint()!=zero:raise ValueError('cold snapshot bytes differ')
            pre=chart.recover();chart.rebase_rotations(chart._generation);save(chart,'zero-rebased');report['parity']['zero_rebase']=parity(pre,chart.recover())
            execute_targets(primary,loads,control,PLAN['targets'][2:],record,offset=2)
            execute_targets(cold,loads,control,PLAN['targets'][2:],record,offset=2)
            execute_targets(chart,loads,control,PLAN['targets'][2:],record,offset=2)
            report['parity']['cold_final']=parity(primary.recover(),cold.recover())
            report['parity']['chart_final']=parity(primary.recover(),chart.recover())
            report['parity']['cold_all_histories']=history_parity(primary,cold,exact=True)
            report['parity']['chart_all_histories']=history_parity(primary,chart,exact=False)
            if primary.recover().history!=cold.recover().history:
                raise ValueError('cold controller history differs')
            report['controller_history']=dict(direct=plain(primary.recover().history),cold=plain(cold.recover().history),chart=plain(chart.recover().history))
            report['reversal']=reversal([x['materials']['fibre_flow_signs'] for x in report['accepted'] if x['label'].startswith('direct')])
            report['physical_reversal_observed']=bool(report['reversal'])
            report['assemblies']=observer.calls
            report['physical_cycle_and_restart_complete']=True;write(out/'cycle.json',report)
            # Deliberate rejection coverage follows all physical comparisons.
            # An unexpected publication remains visible and is terminal.
            before=atomic.capture(primary)
            rejected=False
            try:
                probe=primary.solve(loads,(PLAN['rejected_probe_target'],),displacement_control=control,max_iterations=1,max_backtracks=8)
            except ss.DisplacementCorrectorFailure as exc:
                rejected=True;report['required_rejection_observed']=True
                write(out/'rejected-prefix.json',dict(type=type(exc).__name__,message=str(exc),detail=plain(exc.detail)))
            else:
                report['required_rejection_observed']=False
                write(out/'unexpected-probe-publication.json',probe)
            finally:
                save(primary,'rejected-terminal');atom(primary,'rejected-prefix',before,require_unchanged=rejected)
            if not rejected:raise AssertionError('one-iteration probe accepted; required rejection coverage unresolved')
        report['status']='SCOPED_CYCLE_PASS';write(out/'cycle.json',report)
    except BaseException as exc:
        report['status']='UNRESOLVED_REQUIRED_REJECTION' if report.get('required_rejection_observed') is False else 'UNRESOLVED_CYCLE'
        report['failure']=dict(type=type(exc).__name__,message=str(exc));write(out/'cycle.json',report);raise
    finally:
        for i,owner in enumerate(owners):
            try:
                if owner._finite_driver is not None:save(owner,'terminal-owner-'+str(i))
            finally:owner.close()
    return report

def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0]);sys.path.insert(0,str(ROOT))
    verify_bindings(json.loads((out/'bindings.json').read_text()),bindings())
    require_adoption(json.loads((DOC/'CYCLE_ADOPTION.json').read_text()))
    write(out/'artifact-before.json',artifact())
    site=PACKAGE/'site';sys.path[:0]=[str(site),str(ROOT/'tests')]
    os.environ.update(GE_CYCLE_OUTPUT=str(out),GE_CYCLE_STAGE=stage)
    from scripts.r8_f06_log_domain_check_dev import runtime
    import pytest
    runtime(out,'runtime-before',site)
    nodes=[str(ROOT/'tests/test_r8_ge_coupled_cycle_dev.py')+'::'+name for name in SOURCE_NAMES] if stage=='source-controls' else [str(ROOT/'tests/test_r8_ge_coupled_cycle_dev.py')+'::test_installed_connected_cycle']
    try:
        code=int(pytest.main(nodes+['-q','-x','-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
        if code==0:write(out/'counts.json',counts(out/'junit.xml',8 if stage=='source-controls' else 1))
        return code
    finally:
        rt=runtime(out,'runtime-after',site);frozen=json.loads((PACKAGE/'snapshot-bindings.json').read_text())
        for name,path in rt['imports'].items():
            if sha(path)!=frozen['src/'+Path(path).relative_to(site).as_posix()]:raise ValueError('actual installed import drift')
        write(out/'artifact-after.json',artifact())

def launch(stage):
    # No preparation command creates authority or a ledger.
    require_adoption(json.loads((DOC/'CYCLE_ADOPTION.json').read_text()));artifact()
    slot=json.loads((DOC/'CYCLE_SLOT_TRANSFER.json').read_text())
    if slot.get('status')!='TRANSFERRED' or slot.get('one_numerical_job') is not True:raise ValueError('exclusive fresh slot not transferred')
    if list(DOC.glob('attempt-'+stage+'*')):raise ValueError('sole attempt already consumed')
    if stage=='installed-cycle':
        receipt=json.loads((DOC/'attempt-source-controls1/receipt.json').read_text())
        if receipt['status']!='PASS':raise ValueError('source controls not passed')
        counts(DOC/'attempt-source-controls1/junit.xml',8)
    sys.path.insert(0,str(ROOT))
    from scripts.r8_f06_fibre_log_domain_dev import run
    code=run(stage,doc=DOC,entry=__file__,bind=bindings,authority_file='CYCLE_SOURCE_AUTHORITY.json' if stage=='source-controls' else 'CYCLE_INSTALLED_AUTHORITY.json')
    ledger=DOC/'ledger.json'
    if code or stage=='installed-cycle':
        value=json.loads(ledger.read_text());value.update(status='CLOSED',unused_discarded_s=value['budget_s']-value['used_s']);write(ledger,value)
    return code

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('source-controls','installed-cycle'),required=True)
    parser.add_argument('--child',action='store_true');parser.add_argument('--out');args=parser.parse_args()
    raise SystemExit(child(Path(args.out),args.stage) if args.child else launch(args.stage))
