"""Prepared actual-update derivative discriminator; execution requires fresh authority."""
from pathlib import Path
import argparse, hashlib, json, os, sys
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_ge_paired_fixed_diagnostic_dev as saved
DOC=saved.DOC/'derivative-v1'
INPUT=DOC/'DERIVATIVE_INPUT.json'
STEPS=(1e-4,1e-5,1e-6)
PLAN=dict(entry='CANONICAL_INNER_FIXED_SNAPSHOT',generation=11,trial_evaluation=14,
          direction_solve=5,direction_scale=1.,steps=list(STEPS),difference='centered_actual_advance',
          work_plateau_limit=1e-5,shell_local_limit=1e-5,native_point_local_limit=1e-6,
          require_all_branches_stable=True,no_equilibrium_solve=True,
          prospective_budget_s=180,prospective_reserve_s=60,source_cap_s=20,installed_cap_s=90)
SOURCE_NAMES=('test_source_input_identity','test_source_complete_direction',
 'test_source_work_weight_formula','test_source_centered_step_pair','test_source_override_failure_restore',
 'test_source_actual_branch_refusal','test_source_inventory_and_no_new_solver','test_source_replay_refusal')
FIELDS=('q','q_low','frames','frames_low','cells','cells_low','resultants','resultants_low','multipliers','shell_anchors')
def sha(path):return saved.sha(path)
def write(path,value):saved.write(path,value)
def input_data():
    frozen=json.loads((DOC/'DERIVATIVE_INPUT_IDENTITY.json').read_text())
    if sha(INPUT)!=frozen['sha256']:raise ValueError('saved trial input drift')
    value=json.loads(INPUT.read_text())
    if value['evaluation']['evaluation']!=14 or value['solve']['solve']!=5:
        raise ValueError('exact current14/solve5 required')
    if set(value['evaluation']['state']['values'])!=set(FIELDS):raise ValueError('complete paired state required')
    if len(value['solve']['full_correction'])!=132 or len(value['solve']['direction'])!=132:
        raise ValueError('complete physical/retained/multiplier direction required')
    if value['trace_sha256']!=sha(saved.DOC/'attempt-installed-fixed1/trace.jsonl'):
        raise ValueError('trace identity drift')
    return value
def work_weights(scales,constraint_scales,energy):
    if energy<=0 or any(v<=0 for v in constraint_scales):raise ValueError('positive original work scales required')
    return [v/energy for v in scales]+[1/v for v in constraint_scales]
def step_pair(driver,state,direction,h):
    if h not in STEPS:raise ValueError('frozen inherited h required')
    return driver._advance(state,direction,h),driver._advance(state,direction,-h)
def evaluate_at(observer,label,state,amplitude,base,direction):
    previous=observer.applied
    try:
        observer.applied=base+amplitude*direction
        return observer.evaluate(label,state)
    finally:observer.applied=previous
def evaluate_atomic(observer,owner,atomic,label,state,amplitude,base,direction,out):
    before=atomic.capture(owner)
    try:return evaluate_at(observer,label,state,amplitude,base,direction)
    finally:
        flags=atomic.atomic_flags(owner,before)
        write(out/(label+'-atomic.json'),flags)
        atomic.checked(flags)
def require_replay(flags):
    if not flags or not all(type(v) is bool and v for v in flags.values()):
        raise ValueError('saved current trial replay mismatch')
def require_branches(original,base,plus,minus):
    return original.same_smooth_branch(base,plus) and original.same_smooth_branch(base,minus)
def cancellation_ratio(magnitude,net):
    return magnitude/net if net>0 else None
def prepare():
    if INPUT.exists():raise ValueError('preparation already frozen')
    DOC.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in (saved.DOC/'attempt-installed-fixed1/trace.jsonl').read_text().splitlines()]
    evaluation=next(x for x in rows if x['kind']=='evaluation' and x['evaluation']==14)
    solve=next(x for x in rows if x['kind']=='bordered_solve' and x['solve']==5)
    assembly=[x for x in rows if x['kind']=='assembly'][13]
    write(INPUT,dict(trace_sha256=sha(saved.DOC/'attempt-installed-fixed1/trace.jsonl'),
                     evaluation=evaluation,solve=solve,assembly=assembly))
    write(DOC/'DERIVATIVE_INPUT_IDENTITY.json',dict(sha256=sha(INPUT)))
    write(DOC/'DERIVATIVE_PLAN.json',PLAN)
def connected(out):
    import numpy as np
    from scipy import sparse
    from anyfesolver import stiffened_shell as ss
    from anyfesolver._stiffened_shell_ge_finite_experimental import _State
    from scripts.r8_f06_cyclic_material_dev import definition
    from scripts import r8_f06_paired_active_tangent_dev as original
    if out!=DOC/'attempt-installed-derivative1':raise ValueError('sole owned installed output required')
    data=input_data();raw,_,_=saved.select_inner(saved.OUTER.read_bytes());saved.artifact()
    owner=ss.create_analysis(definition=definition('coupled'));before=None
    report=dict(status='STARTED',scientific_acceptance=False,plan=PLAN,steps=[])
    try:
        owner.restore(raw)
        if owner.checkpoint()!=raw or owner._generation!=11:raise ValueError('exact gen11 public restore required')
        before=original.capture(owner);driver=owner._finite_driver
        state=_State(**{key:np.asarray(value,dtype=float) for key,value in data['evaluation']['state']['values'].items()})
        for value in vars(state).values():value.setflags(write=False)
        _,metadata,arrays=owner._displacement_snapshot
        base_load=arrays['base_load'];pattern=arrays['load_direction']
        direction=np.asarray(data['solve']['full_correction']);delta=float(data['solve']['amplitude_delta'])
        amplitude=float(data['evaluation']['value']);external=np.r_[pattern,np.zeros(len(state.multipliers))]
        if not np.array_equal(external,np.asarray(data['solve']['direction'])):raise ValueError('original external direction mismatch')
        observer=original.ObservedAssembly(owner,out)
        # Actual corrector uses CURRENT accepted origins, not prior accepted origins.
        observer.shell=driver._shell_origins;observer.beam=driver._beam_origins
        def evaluate(label,trial,a):
            return evaluate_atomic(observer,owner,original,label,trial,a,base_load,pattern,out)
        base=evaluate('current14-replay',state,amplitude)
        old=data['evaluation'];stored=old['tangent']
        expected=sparse.coo_matrix((stored['data'],(stored['row'],stored['col'])),shape=stored['shape']).tocsc()
        actual=base[0][1].tocsc();expected.sort_indices();actual.sort_indices()
        stored_locals=data['assembly']['local_contributions']
        if len(base[1])!=len(stored_locals):raise ValueError('saved replay local inventory mismatch')
        if any(not np.array_equal(item['slots'],old_item['slots']) for item,old_item in zip(base[1],stored_locals)):
            raise ValueError('saved replay local role/layout mismatch')
        selectors=[item['values'] for item in data['assembly']['constitutive'] if item['kind']=='J2_executed_internal_selectors']
        if len(selectors)<len(observer.shell):raise ValueError('saved shell predecessor observability incomplete')
        replay=dict(residual=bool(np.array_equal(base[0][0],old['residual'])),
            tangent=bool(actual.shape==expected.shape and np.array_equal(actual.indptr,expected.indptr) and
                         np.array_equal(actual.indices,expected.indices) and np.array_equal(actual.data,expected.data)),
            native_origins=all(item['material']['origin']==stored_item['material']['origin'] for item,stored_item in
                zip(base[1],stored_locals) if item['kind']=='native'),
            shell_origins=all(np.array_equal(origin['plastic_strain'],record['plastic_strain']) and
                             np.array_equal(origin['alpha'],record['alpha']) for origin,record in zip(observer.shell,selectors)))
        report['replay']=replay
        if not all(replay.values()):report['status']='UNRESOLVED_REPLAY'
        write(out/'assessment.json',report);require_replay(replay)
        count=owner._K.shape[0];beam_end=owner._shell_count+6*len(owner._beam_ids)
        groups={}
        for prefix,lo,hi in (('shell',0,owner._shell_count),('beam',owner._shell_count,beam_end)):
            for role,indices in (('translation',(0,1,2)),('rotation',(3,4,5))):
                groups[prefix+'_'+role]=np.array([i for i in owner._free if lo<=i<hi and i%6 in indices])
        for i,(element,_,_) in enumerate(owner._beam_parts):
            start=beam_end+24*i;family='fibre' if hasattr(element.section,'fibres') else 'generalized'
            groups[family+'_cell_rotation']=np.arange(start,start+6)
            groups[family+'_resultants']=np.arange(start+6,start+24)
        groups['POINT_constraints']=np.arange(count,len(direction))
        energy=float(np.max((abs(owner._K)@owner._coordinate_scale)*owner._coordinate_scale))
        weights=np.asarray(work_weights(owner._coordinate_scale,owner._constraint_scale,energy))
        action=base[0][1]@direction-external*delta
        magnitude=abs(base[0][1])@abs(direction)+abs(external*delta)
        evaluation_rows=np.r_[owner._free,np.arange(count,len(direction))]
        observed_groups=dict(all_free_mixed_rows=evaluation_rows,**groups)
        def norm(value,rows):return float(np.linalg.norm(value[rows]*weights[rows]))
        report['action_observability']={key:dict(work_action_norm=norm(action,rows),
            work_absolute_term_magnitude_norm=norm(magnitude,rows),
            magnitude_to_net_action=cancellation_ratio(norm(magnitude,rows),norm(action,rows)),
            interpretation='exact abs(K)*abs(d)+abs(p*delta) work magnitude; ratio only, no floor or cutoff')
            for key,rows in observed_groups.items()}
        np.savez(out/'actual-direction-work-map.npz',direction=direction,amplitude_delta=delta,external=external,
                 action=action,magnitude=magnitude,weights=weights,energy=energy,free=owner._free,**groups)
        errors={key:[] for key in groups};local_errors={str(i):[] for i in range(len(base[1]))}
        resolved=original.branch_smooth(base[2],base[3])
        for h in STEPS:
            plus_state,minus_state=step_pair(driver,state,direction,h)
            plus=evaluate('plus-'+str(h),plus_state,amplitude+h*delta)
            minus=evaluate('minus-'+str(h),minus_state,amplitude-h*delta)
            stable=require_branches(original,base[2],plus[2],minus[2]) and original.branch_smooth(plus[2],plus[3]) and original.branch_smooth(minus[2],minus[3])
            resolved=resolved and stable
            fd=(plus[0][0]-minus[0][0])/(2*h);error=fd-action
            metrics={key:original.metrics(error,action,rows,weights) for key,rows in groups.items()}
            local=[];local_terms=[[] for _ in action];absolute_local_terms=[[] for _ in action]
            if len(base[1])!=len(plus[1]) or len(base[1])!=len(minus[1]):raise ValueError('local contribution inventory changed')
            for i,(b,p,m) in enumerate(zip(base[1],plus[1],minus[1])):
                if not np.array_equal(b['slots'],p['slots']) or not np.array_equal(b['slots'],m['slots']):raise ValueError('local role/layout changed')
                predicted=b['tangent']@direction[b['slots']]
                observed=(p['residual']-m['residual'])/(2*h)
                metric=original.metrics(observed-predicted,predicted,np.arange(len(predicted)),weights[b['slots']])
                local.append(dict(index=i,kind=b['kind'],metrics=metric));local_errors[str(i)].append(metric['work_relative'])
                for slot,value in zip(b['slots'],observed-predicted):
                    local_terms[int(slot)].append(float(value));absolute_local_terms[int(slot)].append(abs(float(value)))
            scattered=np.asarray([math.fsum(row) for row in local_terms])
            local_magnitude=np.asarray([math.fsum(row) for row in absolute_local_terms])
            # Subtract the actually evaluated affine external derivative remainder;
            # do not pretend rounded base+a*p evaluations cancel exactly.
            load_plus=np.r_[base_load+(amplitude+h*delta)*pattern,np.zeros(len(state.multipliers))]
            load_minus=np.r_[base_load+(amplitude-h*delta)*pattern,np.zeros(len(state.multipliers))]
            external_error=(load_plus-load_minus)/(2*h)-external*delta
            local_predicted_assembled_error=scattered-external_error
            agreement=error-local_predicted_assembled_error
            observation={key:dict(work_error_absolute=norm(error,rows),work_action_norm=norm(action,rows),
                work_fd_norm=norm(fd,rows),work_plus_residual_norm=norm(plus[0][0],rows),
                work_minus_residual_norm=norm(minus[0][0],rows),
                work_local_scattered_error_norm=norm(scattered,rows),
                work_absolute_local_error_magnitude_norm=norm(local_magnitude,rows),
                local_error_magnitude_to_net=cancellation_ratio(norm(local_magnitude,rows),norm(scattered,rows)),
                work_external_rounding_error_norm=norm(external_error,rows),
                work_local_predicted_assembled_error_norm=norm(local_predicted_assembled_error,rows),
                work_local_vs_assembled_agreement_absolute=norm(agreement,rows))
                for key,rows in observed_groups.items()}
            np.savez(out/('errors-'+str(h)+'.npz'),fd=fd,error=error,local_scattered_error=scattered,
                absolute_local_error_magnitude=local_magnitude,external_rounding_error=external_error,
                local_predicted_assembled_error=local_predicted_assembled_error,local_vs_assembled_agreement=agreement)
            for key in errors:errors[key].append(metrics[key]['work_relative'])
            report['steps'].append(dict(h=h,branches_stable=stable,groups=metrics,local=local,observability=observation))
            write(out/'assessment.json',report)
        report.update(smooth_branch_resolved=resolved,common_adjacent_plateau=original.adjacent_plateau(errors),
            local_adjacent_plateaus={str(i):original.adjacent_plateau({str(i):local_errors[str(i)]},1e-5 if b['kind']=='shell' else 1e-6) for i,b in enumerate(base[1])})
        positive=all(m['exercised'] for row in report['steps'] for m in row['groups'].values()) and all(
            item['metrics']['exercised'] for row in report['steps'] for item in row['local'])
        report['positive_work_channels']=positive
        strict_pass=positive and report['common_adjacent_plateau'] and all(report['local_adjacent_plateaus'].values())
        report['strict_discriminator_pass']=bool(resolved and strict_pass)
        report['status']='UNRESOLVED_BRANCH' if not resolved else ('UNRESOLVED_EXCITATION' if not positive else ('PASS' if strict_pass else 'UNRESOLVED_DIRECTIONAL_CRITERION'))
        report['interpretation']='PASS is scoped derivative evidence on this actual direction only. Strict failure with weak net action/assembly cancellation remains unresolved, not proof of a tangent defect. Ratios introduce no new floor or acceptance cutoff.'
        write(out/'assessment.json',report)
        if report['status']!='PASS':raise AssertionError(report['status'])
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),detail=saved.plain(getattr(error,'detail',None)))
        write(out/'assessment.json',report);raise
    finally:
        if before is not None:
            report['atomic_flags']=original.atomic_flags(owner,before);write(out/'assessment.json',report)
        owner.close()
        if before is not None:original.checked(report['atomic_flags'])
def bindings():
    paths={ROOT/rel for rel in saved.bindings()}
    paths.update((Path(__file__).resolve(),ROOT/'tests/test_r8_ge_paired_trial_derivative_dev.py',INPUT,
        DOC/'DERIVATIVE_INPUT_IDENTITY.json',DOC/'DERIVATIVE_PLAN.json',DOC/'DERIVATIVE_READINESS.json',
        DOC/'DERIVATIVE_PREPARATION.md',saved.DOC/'attempt-installed-fixed1/trace.jsonl',saved.DOC/'TERMINAL_SELECTED.json'))
    if len({p.name for p in paths})!=len(paths):raise ValueError('supervisor snapshot basename collision')
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0]);sys.path[:0]=[str(saved.PACKAGE/'site'),str(ROOT/'tests')]
    if json.loads((out/'bindings.json').read_text())!=bindings():raise ValueError('child binding drift')
    os.environ.update(GE_TRIAL_DERIVATIVE_STAGE=stage,GE_TRIAL_DERIVATIVE_OUTPUT=str(out))
    from scripts.r8_f06_log_domain_check_dev import runtime
    runtime(out,'runtime-before',saved.PACKAGE/'site')
    import pytest
    names=SOURCE_NAMES if stage=='source-controls' else ('test_installed_actual_update_direction',)
    try:
        code=int(pytest.main([str(ROOT/'tests/test_r8_ge_paired_trial_derivative_dev.py')+'::'+n for n in names]+['-q','-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
        from scripts.r8_ge_coupled_cycle_dev import counts
        result=counts(out/'junit.xml',len(names));write(out/'counts.json',result)
        return code
    finally:
        rt=runtime(out,'runtime-after',saved.PACKAGE/'site');tree=saved.artifact()
        for path in rt['imports'].values():
            if sha(path)!=tree['src/'+Path(path).relative_to(saved.PACKAGE/'site').as_posix()]:raise ValueError('installed import drift')
def launch(stage):
    from scripts.r8_f06_fibre_log_domain_dev import run
    path=DOC/('DERIVATIVE_'+('SOURCE' if stage=='source-controls' else 'INSTALLED')+'_AUTHORITY.json')
    grant=json.loads(path.read_text());slot=DOC/'DERIVATIVE_SLOT_TRANSFER.json'
    if grant.get('status')!='AUTHORIZED' or json.loads(slot.read_text()).get('status')!='TRANSFERRED':raise ValueError('fresh derivative authority/slot required')
    if grant.get('plan')!=PLAN or grant.get('bindings_sha256')!=sha(DOC/'DERIVATIVE_BINDINGS.json') or grant.get('slot_sha256')!=sha(slot):raise ValueError('exact prospective plan/bindings/slot required')
    if (grant.get('budget_s'),grant.get('reserve_s'),grant.get('process_cap_s'))!=(180,60,20 if stage=='source-controls' else 90):raise ValueError('exact fresh resource envelope required')
    if list(DOC.glob('attempt-'+stage+'*')):raise ValueError('sole attempt already consumed')
    if stage=='installed-derivative':
        if json.loads((DOC/'attempt-source-controls1/counts.json').read_text())!=dict(tests=8,errors=0,failures=0,skipped=0):raise ValueError('exact8 source PASS required')
        if json.loads((DOC/'attempt-source-controls1/receipt.json').read_text())['status']!='PASS':raise ValueError('source receipt PASS required')
    captured=path.read_bytes()
    try:
        return run(stage,doc=DOC,entry=__file__,bind=bindings,authority_file=path.name)
    finally:
        out=DOC/('attempt-'+stage+'1')
        if out.exists():
            (out/'used-authority.json').write_bytes(captured)
            if stage=='installed-derivative' or json.loads((out/'receipt.json').read_text())['status']!='PASS':
                lp=DOC/'ledger.json';ledger=json.loads(lp.read_text());ledger.update(status='CLOSED',unused_discarded_s=ledger['budget_s']-ledger['used_s']);write(lp,ledger)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',choices=('source-controls','installed-derivative'));args=p.parse_args()
    if args.prepare:prepare()
    else:raise SystemExit(child(Path(args.out),args.stage) if args.child else launch(args.stage))
