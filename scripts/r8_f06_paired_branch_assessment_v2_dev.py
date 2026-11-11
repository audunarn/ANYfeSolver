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
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_f06_paired_active_tangent_dev as original
from scripts import r8_f06_paired_branch_domain_dev as domain

DOC=original.DOC/'branch-domain-v1/assessment-v2'
PREP=original.DOC/'branch-domain-v1/preparation3'
SCALE=0.00698115606742955
RUNNER_SHA='8e5adbfd7a86e9ee0b0367c283ce37c5a467be6f40dac00d0609bd45b98c4129'
CANDIDATE_SHA='31478654c7ca8569054033f5b05c912ca443920d626f7d1c40a2af0f766cf27a'
PREPARATION_SHA='f722bdd60f1b4cb9b0b167f60c2e23b40a462a4e105584016317517053e49a5a'
TEST=ROOT/'tests/test_r8_f06_paired_branch_assessment_v2_dev.py'
SOURCE_COUNT=16


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
            'test_source_caller_selects_exact_inventory_including_installed_name'))
    )
    targets=tuple(str(ROOT/'tests'/file)+'::'+name for file,names in inventory for name in names)
    if len(targets)!=SOURCE_COUNT or len(set(targets))!=SOURCE_COUNT:
        raise ValueError('frozen sixteen-node source inventory required')
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
    if candidate['uniform_scale']!=SCALE or candidate['h']!=list(original.STEPS):
        raise ValueError('adopted scale and inherited h required')
    if len(vector)!=132 or len(candidate['direction'])!=132:
        raise ValueError('complete 132-component direction required')
    scaled=[SCALE*float(x) for x in vector]
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
    if prepared['status']!='AFFINE_FIBRE_DOMAIN_CANDIDATE_ONLY' or prepared['uniform_scale']!=SCALE:
        raise ValueError('bound development preparation required')
    if prepared['accepted_sha256']!=original.INPUT_SHA or prepared['rebased_sha256']!=original.REBASED_SHA:
        raise ValueError('same accepted/rebased endpoint required')
    for rel,digest in prepared['bindings'].items():
        if sha(ROOT/rel)!=digest:
            raise ValueError('changed preparation dependency: '+rel)
    source=domain.SAVED/'direction-and-row-map.npz'
    if sha(source)!=candidate['source_npz_sha256'] or sha(source)!=prepared['vector_sha256']:
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
    provenance=dict(uniform_scale=SCALE,uniform_scale_hex=SCALE.hex(),
        source_direction_npz_sha256=candidate['source_npz_sha256'],candidate_sha256=CANDIDATE_SHA,
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
        return result
    finally:
        original.frozen_direction=saved
        path=out/'assessment.json'
        if path.exists():
            result=json.loads(path.read_text());result['branch_direction_provenance']=provenance
            result['complete_observations']=complete_metrics(result)
            original.write(path,result)


def bindings(*,preparation=False):
    inherited=original.bindings(preparation=True)
    paths={ROOT/rel for rel in inherited if 'AUTHORITY' not in Path(rel).name and Path(rel).name!='SLOT_TRANSFER.json'}
    paths.update((Path(__file__).resolve(),TEST,ROOT/'scripts/r8_f06_paired_branch_domain_dev.py',
        ROOT/'tests/test_r8_f06_paired_branch_domain_dev.py',PREP/'candidate.json',PREP/'candidate-direction.json'))
    paths.update(DOC/name for name in ('BRANCH_SOURCE_AUTHORITY.json','BRANCH_INSTALLED_AUTHORITY.json','BRANCH_SLOT_TRANSFER.json'))
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
    original.write(out/'artifact-before.json',original.artifact())
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
        original.write(out/'artifact-after.json',original.artifact())


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
    assert authority['wheel_sha256']==original.WHEEL_SHA and authority['uniform_scale']==SCALE
    assert not list(DOC.glob('attempt-'+stage+'*')), 'sole attempt; no automatic retry'
    if stage=='installed-assessment':
        receipt=DOC/'attempt-source-controls1/receipt.json'
        assert receipt.is_file() and json.loads(receipt.read_text())['status']=='PASS'
    load_candidate();bindings()
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
                raise ValueError('required 16 source controls PASS, zero skipped')
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
        original.write(DOC/'BRANCH_PREPARATION.json',dict(status='PREPARED_NOT_LAUNCHED',bindings=bound,
            unique_basenames=len(bound),uniform_scale=SCALE,stage_caps_s=[20.,60.],budget_s=180.,reserve_s=60.,
            numerical_imports=[n for n in sys.modules if n.split('.')[0] in ('numpy','scipy','numba','anyfesolver','pytest')]))
        print(json.dumps(dict(bindings=len(bound),missing=[p for p,v in bound.items() if v is None])))
    elif args.child:
        raise SystemExit(child(Path(args.out),args.stage))
    else:
        assert args.stage in ('source-controls','installed-assessment')
        raise SystemExit(launch(args.stage))
