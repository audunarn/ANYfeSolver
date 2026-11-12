"""One supervised installed assessment of the adopted whole-direction candidate.

Preparation is stdlib-only. No launch without fresh root authority/slot transfer.
The original assessment implementation and its failed evidence are immutable.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_f06_paired_active_tangent_dev as original
from scripts import r8_f06_paired_branch_domain_dev as domain

DOC=original.DOC/'branch-domain-v1/role-direction-v1/assessment-v1'
PREP=original.DOC/'branch-domain-v1/role-direction-v1'
SCALE=0.00698115606742955
RUNNER_SHA='8e5adbfd7a86e9ee0b0367c283ce37c5a467be6f40dac00d0609bd45b98c4129'
CANDIDATE_SHA='83a751cf33a518c0d44487b7292f718f4b4203213a70f543831e528d0c4928ba'
PREPARATION_SHA='26f4869189a838eb4a2775895e47080b71d8818db092c682da7f1260d8d14e97'
TEST=ROOT/'tests/test_r8_f06_paired_role_assessment_dev.py'
SOURCE_COUNT=17
SCALES=dict(physical=1.,cell_rotation=1.,multipliers=1.,native_retained=SCALE)
ROLES=dict(physical=list(range(60)),cell_rotation=list(range(60,66))+list(range(84,90)),
    native_retained=list(range(66,84))+list(range(90,108)),multipliers=list(range(108,132)))
MANIFEST_SHA='3288a782763b64c2d12fbf923ff0239abc3017dfcc68c234e0b5cef76480b5d9'
B2_DELTA_PATH='src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py'
ATOMIC_KEYS=('checkpoint','generation','objects','state_object','state_words','accepted_object','controller',
    'driver_object','driver_accepted_object','driver_accepted_words','applied_load_object','applied_load_bytes','snapshot_object',
    'shell_accepted_objects','shell_accepted_words','shell_predecessors_objects','shell_predecessors_words',
    'beam_accepted_objects','beam_accepted_words','beam_predecessors_objects','beam_predecessors_words')


def check_artifact_delta(delta,decision):
    """Root must adopt exact old/new words; never wildcard or implicit waiver."""
    if decision.get('status')!='ARTIFACT_REUSE_ADOPTED' or decision.get('manifest_sha256')!=MANIFEST_SHA or decision.get('wheel_sha256')!=original.WHEEL_SHA:
        raise ValueError('root frozen-artifact reuse adoption required')
    if decision.get('current_source_deltas')!=delta:
        raise ValueError('unadopted current source drift')
    if not isinstance(decision.get('independence_justification'),str) or not decision['independence_justification'].strip():
        raise ValueError('root artifact independence justification required')
    for row in delta:
        if row.get('path')!=B2_DELTA_PATH or set(row)!= {'path','old_sha256','new_sha256'}:
            raise ValueError('only exact independently justified B2 diagnostic module delta permitted')
        if any(not isinstance(row[k],str) or len(row[k])!=64 or any(c not in '0123456789abcdef' for c in row[k]) for k in ('old_sha256','new_sha256')):
            raise ValueError('exact SHA256 delta words required')
    if len(delta)>1:raise ValueError('duplicate artifact delta')
    return True


def artifact(*,preparation=False):
    package=original.PACKAGE;manifest=package/'snapshot-bindings.json';frozen=json.loads(manifest.read_text())
    wheels=list((package/'wheels').glob('*.whl'))
    if len(frozen)!=633 or sha(manifest)!=MANIFEST_SHA or len(wheels)!=1 or sha(wheels[0])!=original.WHEEL_SHA:
        raise ValueError('exact frozen 633-file artifact identity required')
    deltas=[]
    with zipfile.ZipFile(wheels[0]) as wheel:
        for rel,digest in sorted(frozen.items()):
            name=Path(rel).relative_to('src').as_posix()
            if sha(package/'snapshot'/rel)!=digest or sha(package/'site'/name)!=digest or hashlib.sha256(wheel.read(name)).hexdigest()!=digest:
                raise ValueError('frozen snapshot/site/wheel mismatch: '+rel)
            current=sha(ROOT/rel)
            if current!=digest:deltas.append(dict(path=rel,old_sha256=digest,new_sha256=current))
    adopted=False;decision=DOC/'BRANCH_ARTIFACT_REUSE_DECISION.json'
    if not preparation:
        if not decision.is_file():raise ValueError('root artifact-reuse decision missing')
        check_artifact_delta(deltas,json.loads(decision.read_text()));adopted=True
    return dict(files=633,wheel_sha256=original.WHEEL_SHA,manifest_sha256=MANIFEST_SHA,
        frozen_snapshot_site_wheel_exact=True,current_source_deltas=deltas,current_source_delta_adopted=adopted,
        legacy_allcurrent_contract_preserved=True)


def source_nodeids():
    """Frozen complete source inventory; names never interpreted as substrings."""
    inventory=(
        ('test_r8_f06_paired_active_tangent_dev.py',(
            'test_source_all_flags_negative_calibration',
            'test_source_group_norms_and_common_adjacent_plateau',
            'test_source_atomic_state_history_words_and_objects',
            'test_source_frozen_complete_direction',
            'test_source_paired_branch_sign_and_segment_crossing',
            'test_source_unfrozen_step_refused_before_evaluation',
            'test_source_vector_component_crossing_and_true_branch_transition')),
        ('test_r8_f06_paired_branch_domain_dev.py',(
            'test_affine_fibre_domain_uses_flow_boundary_not_tangent_errors',
            'test_crossing_or_zero_margin_refuses_saved_secant',
            'test_uniform_scale_preserves_all_nonzero_channels')),
        (TEST.name,(
            'test_source_adopted_scale_and_complete_nonzeros',
            'test_source_changed_scale_channel_and_metadata_refused',
            'test_source_changed_candidate_bytes_refused',
            'test_source_missing_branch_and_unexercised_work_refused',
            'test_source_override_refused_outside_installed_child',
            'test_source_caller_selects_exact_inventory_including_installed_name',
            'test_source_exact_artifact_delta_adoption_refuses_drift'))
    )
    targets=tuple(str(ROOT/'tests'/file)+'::'+name for file,names in inventory for name in names)
    if len(targets)!=SOURCE_COUNT or len(set(targets))!=SOURCE_COUNT:
        raise ValueError('frozen seventeen-node source inventory required')
    return targets


def run_tests(out,stage,main):
    if stage=='source-controls':
        targets=list(source_nodeids())
    elif stage=='installed-assessment':
        targets=[str(TEST)+'::test_installed_branch_domain_tangent']
    else:
        raise ValueError('unknown branch assessment stage')
    return int(main(targets+['-q','-x','-o','pythonpath=','-p','no:cacheprovider',
        '--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scaled_from_original(vector,candidate):
    if candidate['scale_by_role']!=SCALES or candidate['roles']!=ROLES or candidate['h']!=list(original.STEPS):
        raise ValueError('adopted role scales, complete roles and inherited h required')
    if len(vector)!=132 or len(candidate['direction'])!=132:
        raise ValueError('complete 132-component direction required')
    scaled=[SCALE*float(x) if i in ROLES['native_retained'] else float(x) for i,x in enumerate(vector)]
    if not all(math.isfinite(x) for x in scaled) or scaled!=candidate['direction']:
        raise ValueError('changed candidate or source direction')
    if sum(x!=0. for x in vector)!=120 or any((a==0.)!=(b==0.) for a,b in zip(vector,scaled)):
        raise ValueError('all 120 original nonzero channels required')
    return scaled


def load_candidate(candidate_path=None,preparation_path=None):
    candidate_path=PREP/'candidate-direction.json' if candidate_path is None else Path(candidate_path)
    preparation_path=PREP/'candidate.json' if preparation_path is None else Path(preparation_path)
    if sha(original.__file__)!=RUNNER_SHA:
        raise ValueError('original immutable assessment runner changed')
    if sha(candidate_path)!=CANDIDATE_SHA or sha(preparation_path)!=PREPARATION_SHA:
        raise ValueError('changed candidate/preparation bytes')
    candidate=json.loads(candidate_path.read_text());prepared=json.loads(preparation_path.read_text())
    if prepared['status']!='ROLE_DIRECTION_CANDIDATE_ONLY' or prepared['scale_by_role']!=SCALES:
        raise ValueError('bound development preparation required')
    decision=json.loads((PREP/'ROOT_DIRECTION_DECISION.json').read_text())
    if decision['status']!='PROSPECTIVE_DEVELOPMENT_DIRECTION_ADOPTED' or decision['candidate_sha256']!=PREPARATION_SHA or decision['direction_sha256']!=CANDIDATE_SHA or decision['scale_by_role']!=SCALES or decision['native_resultant_slots']!=ROLES['native_retained']:
        raise ValueError('exact root role direction adoption required')
    if prepared['accepted_sha256']!=original.INPUT_SHA or prepared['rebased_sha256']!=original.REBASED_SHA:
        raise ValueError('same accepted/rebased endpoint required')
    for rel,digest in prepared['bindings'].items():
        if sha(ROOT/rel)!=digest:
            raise ValueError('changed preparation dependency: '+rel)
    source=domain.SAVED/'direction-and-row-map.npz'
    if sha(source)!=prepared['source_vector_sha256']:
        raise ValueError('changed original frozen direction archive')
    scaled_from_original(domain.read_vector(source,'direction'),candidate)
    return candidate,prepared


def complete_metrics(report):
    """Require explicit observations; never turn missing branch/work data true."""
    steps=report.get('evaluations',[])
    if len(steps)!=3 or [x.get('h') for x in steps]!=list(original.STEPS):
        return dict(branches=False,positive_groups=False,positive_locals=False)
    groups=('shell','beam','constraints','fibre_retained','generalized_retained')
    positive_groups=all(set(s.get('groups',{}))==set(groups) and all(
        s['groups'][name].get('raw_action_norm',0.)>0. and
        s['groups'][name].get('work_action_norm',0.)>0. and
        s['groups'][name].get('exercised') is True for name in groups) for s in steps)
    local=report.get('local_derivatives',[])
    expected=len(report.get('local_adjacent_plateau',{}))
    positive_locals=expected>0 and len(local)==3*expected and all(
        x['metrics'].get('raw_action_norm',0.)>0. and x['metrics'].get('work_action_norm',0.)>0.
        and x['metrics'].get('exercised') is True for x in local)
    return dict(branches=report.get('base_smooth') is True and report.get('smooth_assessment_resolved') is True
        and all(s.get('branches_stable') is True for s in steps),
        positive_groups=positive_groups,positive_locals=positive_locals)


def assess(out):
    if os.environ.get('F06_BRANCH_STAGE')!='installed-assessment':
        raise ValueError('scaled_direction override is installed-child-only')
    candidate,prepared=load_candidate()
    provenance=dict(scale_by_role=SCALES,roles=ROLES,native_cap_hex=SCALE.hex(),
        source_direction_npz_sha256=prepared['source_vector_sha256'],candidate_sha256=CANDIDATE_SHA,
        preparation_sha256=PREPARATION_SHA,original_runner_sha256=RUNNER_SHA,
        steps=list(original.STEPS),rng_seed=818,component_count=132,nonzero_count=120,
        development_candidate=True,generalized_J2_certificate=False,independent_review='Mistral pending')
    original.write(out/'branch-direction-provenance.json',provenance)
    saved=original.frozen_direction
    def scaled_direction(owner,residual):
        import numpy as np
        return np.asarray(scaled_from_original(saved(owner,residual),candidate),dtype=float)
    original.frozen_direction=scaled_direction
    try:
        original.assess(out)
        result=json.loads((out/'assessment.json').read_text())
        flags=complete_metrics(result)
        result['branch_direction_provenance']=provenance;result['complete_observations']=flags
        if not all(flags.values()):
            result['status']='UNRESOLVED_BRANCH' if not flags['branches'] else 'UNRESOLVED_UNEXERCISED'
            original.write(out/'assessment.json',result)
            raise AssertionError(result['status'])
    finally:
        original.frozen_direction=saved
        path=out/'assessment.json'
        if path.exists():
            result=json.loads(path.read_text());result['branch_direction_provenance']=provenance
            result['complete_observations']=complete_metrics(result)
            terminal=out/'chart-terminal-atomic.json'
            if terminal.exists():
                result['chart_terminal_atomic']=json.loads(terminal.read_text())
            atomic_names=('atomic','chart_atomic','refusal_atomic','cancellation_atomic','terminal_atomic','chart_terminal_atomic')
            result['complete_atomic_observations']={name:bool(isinstance(result.get(name),dict) and set(result[name])==set(ATOMIC_KEYS) and all(type(v) is bool and v for v in result[name].values())) for name in atomic_names}
            original.write(path,result)
            if result['status']=='SCOPED_PASS' and not all(result['complete_atomic_observations'].values()):
                result['status']='UNRESOLVED_ATOMIC';result['failure']=dict(type='AssertionError',message='incomplete or failed 126 atomic flags')
                original.write(path,result)
                raise AssertionError('incomplete or failed 126 atomic flags')
    return result


def bindings(*,preparation=False):
    inherited=original.bindings(preparation=True)
    paths={ROOT/rel for rel in inherited if 'AUTHORITY' not in Path(rel).name and Path(rel).name!='SLOT_TRANSFER.json'}
    paths.update((Path(__file__).resolve(),TEST,ROOT/'scripts/r8_f06_paired_branch_domain_dev.py',
        ROOT/'tests/test_r8_f06_paired_branch_domain_dev.py',PREP/'candidate.json',PREP/'candidate-direction.json',PREP/'ROOT_DIRECTION_DECISION.json'))
    paths.update(DOC/name for name in ('BRANCH_SOURCE_AUTHORITY.json','BRANCH_INSTALLED_AUTHORITY.json','BRANCH_SLOT_TRANSFER.json'))
    paths.add(DOC/'BRANCH_ARTIFACT_REUSE_DECISION.json')
    paths.update(DOC.glob('BRANCH_*AUTHORITY*.json'))
    names={};result={}
    for path in sorted(paths):
        rel=path.relative_to(ROOT).as_posix();key=path.name.casefold()
        if key in names:
            raise ValueError('flat supervisor snapshot basename collision: '+names[key]+' and '+rel)
        names[key]=rel;result[rel]=None if preparation and not path.exists() else sha(path)
    return result


def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0])
    assert json.loads((out/'bindings.json').read_text())==bindings()
    load_candidate()
    original.write(out/'artifact-before.json',artifact())
    selected=original.PACKAGE/'site';sys.path[:0]=[str(selected),str(ROOT/'tests')]
    from scripts.r8_f06_log_domain_check_dev import runtime
    os.environ['F06_BRANCH_STAGE']=stage;os.environ['F06_BRANCH_ASSESSMENT_OUTPUT']=str(out)
    os.environ['F06_ACTIVE_TANGENT_OUTPUT']=str(out)
    runtime(out,'runtime-before',selected)
    import pytest
    try:
        return run_tests(out,stage,pytest.main)
    finally:
        rt=runtime(out,'runtime-after',selected);frozen=json.loads((original.PACKAGE/'snapshot-bindings.json').read_text())
        for name,path in rt['imports'].items():
            assert rt['import_sha256'][name]==frozen['src/'+Path(path).relative_to(selected).as_posix()]
        original.write(out/'artifact-after.json',artifact())


def launch(stage):
    assert (DOC/'BRANCH_SLOT_TRANSFER.json').is_file(), 'fresh root review and slot transfer required'
    transfer=json.loads((DOC/'BRANCH_SLOT_TRANSFER.json').read_text())
    assert transfer['status']=='TRANSFERRED' and transfer['only_numerical_job'] is True
    assert transfer['budget_s']==180. and transfer['reserve_s']==60.
    name='BRANCH_SOURCE_AUTHORITY.json' if stage=='source-controls' else 'BRANCH_INSTALLED_AUTHORITY.json'
    authority=json.loads((DOC/name).read_text())
    assert authority['status']=='AUTHORIZED' and authority['budget_s']==180. and authority['reserve_s']==60.
    assert 0.<authority['process_cap_s']<=(20. if stage=='source-controls' else 60.)
    assert authority['accepted_sha256']==original.INPUT_SHA and authority['rebased_sha256']==original.REBASED_SHA
    assert authority['wheel_sha256']==original.WHEEL_SHA and authority['scale_by_role']==SCALES
    assert authority['direction_sha256']==CANDIDATE_SHA
    assert not list(DOC.glob('attempt-'+stage+'*')), 'sole attempt; no automatic retry'
    if stage=='installed-assessment':
        receipt=DOC/'attempt-source-controls1/receipt.json'
        assert receipt.is_file() and json.loads(receipt.read_text())['status']=='PASS'
    load_candidate();bindings();artifact()
    from scripts import r8_f06_fibre_log_domain_dev as supervisor
    code=supervisor.run(stage,doc=DOC,entry=__file__,bind=bindings,authority_file=name)
    if stage=='source-controls' and not code:
        attempt=DOC/'attempt-source-controls1';receipt_path=attempt/'receipt.json'
        receipt=json.loads(receipt_path.read_text())
        try:
            cases=ET.parse(attempt/'junit.xml').getroot().findall('.//testcase')
            skipped=sum(x.find('skipped') is not None for x in cases)
            failed=sum(x.find('failure') is not None or x.find('error') is not None for x in cases)
            receipt['source_control_counts']=dict(total=len(cases),passed=len(cases)-skipped-failed,skipped=skipped,failed=failed)
            if len(cases)!=SOURCE_COUNT or skipped or failed:
                raise ValueError('required 17 source controls PASS, zero skipped')
        except Exception as exc:
            code=1;receipt.update(status='FAIL',failure='source control adjudication: '+repr(exc))
            ledger=json.loads((DOC/'ledger.json').read_text());ledger['attempts'][-1]['status']='FAIL'
            original.write(DOC/'ledger.json',ledger)
        original.write(receipt_path,receipt)
    if code or stage=='installed-assessment':
        ledger=json.loads((DOC/'ledger.json').read_text());ledger.update(status='CLOSED',unused_discarded_s=ledger['budget_s']-ledger['used_s'])
        original.write(DOC/'ledger.json',ledger)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('source-controls','installed-assessment'))
    parser.add_argument('--child',action='store_true');parser.add_argument('--out');parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        load_candidate();bound=bindings(preparation=True);DOC.mkdir(parents=True,exist_ok=True)
        original.write(DOC/'BRANCH_ARTIFACT_PREFLIGHT.json',artifact(preparation=True))
        original.write(DOC/'BRANCH_PREPARATION.json',dict(status='PREPARED_NOT_LAUNCHED',bindings=bound,
            unique_basenames=len(bound),scale_by_role=SCALES,source_count=SOURCE_COUNT,stage_caps_s=[20.,60.],budget_s=180.,reserve_s=60.,
            numerical_imports=[n for n in sys.modules if n.split('.')[0] in ('numpy','scipy','numba','anyfesolver','pytest')]))
        print(json.dumps(dict(bindings=len(bound),missing=[p for p,v in bound.items() if v is None])))
    elif args.child:
        raise SystemExit(child(Path(args.out),args.stage))
    else:
        assert args.stage in ('source-controls','installed-assessment')
        raise SystemExit(launch(args.stage))
