"""Bounded private smoke/restart waves; never full G3c qualification."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_ge_beam3_g3c_stable as inherited
import ge_beam3_g3c_history_contract as design
import ge_beam3_g3c_restart_preflight as preflight

environment = inherited.environment
canonical = inherited.canonical
write = inherited.write
BASE = 'f4cabebffa6507b98b74bbc3ca35e3d32646b1ab'
BASE_TREE = '1858bebd483869edc2f60fe8cc464ee91b301c4b'
SCOPE = 'G3C_HISTORY_WRAPPER_SMOKE_AND_PREFIX_V1'
TEST = 'tests/test_ge_beam3_g3c_history_smoke.py'
SMOKES = {name:[TEST+'::test_family_two_stage_smoke['+fixture+']'] for name,fixture in (
    ('smoke-b2','J_B2_PAIR'),('smoke-b3','J_B3_PAIR'),('smoke-q4','J_Q4_PAIR'),
    ('smoke-s3','J_S3_PAIR'),('smoke-loop','J_MULTIFAMILY_LOOP'))}
PREFIXES = {'restart-prefix'+str(i):[TEST+'::test_b2_authentic_prefix_replay_and_continuation[prefix'+str(i)+']'] for i in range(3)}
INVENTORIES = {**SMOKES,**PREFIXES}
ADDED = {
    'scripts/ge_beam3_g3c_history_owner.py','scripts/run_ge_beam3_g3c_history.py',
    'tests/test_ge_beam3_g3c_history_wrapper_static.py',TEST,
    'tests/test_ge_beam3_g3c_history_runner_static.py','docs/GE_BEAM3_G3C_HISTORY_WRAPPER_IMPLEMENTATION.md',
}


def verify_review(raw, expected, candidate, inputs):
    inherited.verify_review(raw,expected,candidate,inputs)
    review=environment.strict(raw)
    if review['scope'].get('scope_id') != SCOPE:
        raise ValueError('history wrapper scope not reviewed')
    return review


def authority(review_path, expected):
    candidate=inherited.clean_identity()
    if inherited.git('rev-parse',BASE+'^{tree}') != BASE_TREE: raise ValueError('history base tree')
    inherited.git('merge-base','--is-ancestor',BASE,'HEAD')
    changed={}
    for row in inherited.git('diff','--name-status','--no-renames',BASE,'HEAD').splitlines():
        kind,path=row.split('\t'); changed[path]=kind
    if set(changed)!=ADDED or any(v!='A' for v in changed.values()):
        raise ValueError('history implementation extent changed')
    design.audit(design.strict(design.normalized(ROOT/design.CONTRACT)))
    preflight.authorities()
    mapping=inherited.source_map.strict(inherited.source_map.normalized(ROOT/inherited.source_map.MAP))
    inherited.source_map.audit(mapping,implemented=True)
    inputs=inherited.inputs()
    raw=environment.regular(review_path).read_bytes()
    verify_review(raw,expected,candidate,inputs)
    environment.verify(inherited.infrastructure.CAPSULE,inherited.infrastructure.CAPSULE_SHA)
    return candidate,inputs,raw


def worker(out, expected):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('isolated worker required')
    raw=environment.regular(out/'lease.json').read_bytes()
    if sha256(raw).hexdigest()!=expected: raise ValueError('lease digest')
    lease=environment.strict(raw)
    candidate,inputs,review=authority(out/'review.json',lease['review_sha256'])
    if candidate!=lease['candidate'] or inputs!=lease['inputs'] or lease['scope_id']!=SCOPE:
        raise ValueError('leased authority changed')
    lane=lease['lane']
    if lane not in INVENTORIES or lease['inventory']!=INVENTORIES[lane]: raise ValueError('unknown lane')
    print('G3C CHECKPOINT history authority',flush=True)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(inherited.infrastructure.CAPSULE.parent/'site')]
    from anyfesolver._ge_beam3_g3c_stable import authority as runtime
    runtime.capture(raw)
    import pytest
    inventory=inherited.Inventory(1)
    code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),
                     *INVENTORIES[lane]],plugins=[inventory])
    final_candidate,final_inputs,final_review=authority(out/'review.json',lease['review_sha256'])
    if (final_candidate!=candidate or final_inputs!=inputs or final_review!=review or inventory.count!=1):
        raise ValueError('history final authority/inventory changed')
    completion=completion_expected(lease)
    completion.update(collected_nodes=inventory.count,passed_nodes=inventory.passed,
                      nonpassing=inventory.nonpassing,exit_code=int(code))
    write(out/'completion.json',completion)
    print('G3C CHECKPOINT history complete',flush=True)
    return int(code or inventory.nonpassing or inventory.passed!=1)


def fingerprint(path):
    raw=environment.regular(path).read_bytes()
    return dict(bytes=len(raw),sha256=sha256(raw).hexdigest())


def completion_expected(lease):
    return dict(kind='G3C_HISTORY_WORKER_COMPLETION',scope_id=SCOPE,lane=lease['lane'],
        candidate=lease['candidate'],inputs_sha256=sha256(canonical(lease['inputs'])).hexdigest(),
        inventory=lease['inventory'],collected_nodes=1,passed_nodes=1,nonpassing=False,exit_code=0,
        lease_sha256=sha256(canonical(lease)).hexdigest())


def verify_child_files(out, record, lease):
    if set(record['files'])!={'stdout.log','stderr.log','completion.json'}:
        raise ValueError('complete raw worker evidence required')
    for name in ('stdout.log','stderr.log','completion.json'):
        actual=fingerprint(out/name)
        if canonical(actual)!=canonical(record['files'][name]): raise ValueError('worker file hash/bytes mismatch')
        if name!='stderr.log' and actual['bytes']==0: raise ValueError('empty worker evidence')
    completion=environment.strict(environment.regular(out/'completion.json').read_bytes())
    if canonical(completion)!=canonical(completion_expected(lease)):
        raise ValueError('actual one-node pass/inventory evidence required')


def child(out, lease, review, deadline):
    out.mkdir()  # distinct exclusive directory; never resumed or reused
    with (out/'review.json').open('xb') as stream: stream.write(review)
    write(out/'lease.json',lease)
    lease_hash=sha256((out/'lease.json').read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('history_process_'+lease['lane'].replace('-','_'),ROOT/'docs/reference_cases/e4_pl_s3_v2_bounded_process.py')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'BLIS_NUM_THREADS','NUMBA_NUM_THREADS','TBB_NUM_THREADS'): env[key]='1'
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTHONDONTWRITEBYTECODE='1')
    job=module._ProcessJob(24*1024**3)
    start=time.monotonic(); last=start; previous=None; status='FAILED'; process=None
    try:
        with (out/'stdout.log').open('xb') as stdout, (out/'stderr.log').open('xb') as stderr:
            process=job.launch([sys.executable,'-I','-S','-B','-u',str(Path(__file__).resolve()),
                '--worker',str(out),lease_hash],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            while True:
                cpu,active,peak=job.accounting(); now=time.monotonic()
                progress=(cpu,(out/'stdout.log').stat().st_size,(out/'stderr.log').stat().st_size)
                if progress!=previous: previous=progress; last=now
                if now>=deadline or now-start>=600 or now-last>=120 or peak>24*1024**3:
                    status='RESOURCE_BLOCKED'; break
                if process.poll() is not None and active==0:
                    status='PASSED' if process.returncode==0 else 'FAILED'; break
                time.sleep(.1)
    finally:
        accounting=inherited.close_tree(job,process,lambda row:write(out/'cleanup-failure.json',row))
    record=dict(kind='G3C_HISTORY_CHILD_DIAGNOSTIC',scope_id=SCOPE,status=status,lane=lease['lane'],
        elapsed_seconds=time.monotonic()-start,returncode=process.returncode,active_processes=accounting[1],
        peak_tree_bytes=accounting[2],lease_sha256=lease_hash,
        files={name:fingerprint(out/name) for name in ('stdout.log','stderr.log','completion.json') if (out/name).exists()})
    if status=='PASSED':
        try: verify_child_files(out,record,lease)
        except (ValueError,OSError): record['status']='FAILED_EVIDENCE'
    write(out/'process.json',record)
    return record


def verify_prior_smoke(root, candidate, inputs, review_hash):
    wave=environment.strict(environment.regular(root/'wave.json').read_bytes())
    if (set(wave)!={'scope_id','kind','status','elapsed_seconds','results','wave'}
            or wave['kind']!='G3C_HISTORY_WAVE_DIAGNOSTIC' or type(wave['elapsed_seconds']) is not float
            or wave['scope_id']!=SCOPE or wave['wave']!='smoke' or wave['status']!='PASSED'
            or not 0.<=wave['elapsed_seconds']<=1800.):
        raise ValueError('complete accepted smoke diagnostics required')
    if set(wave['results'])!=set(SMOKES): raise ValueError('all five smoke families required')
    for lane in SMOKES:
        raw=environment.regular(root/lane/'lease.json').read_bytes(); lease=environment.strict(raw)
        result=environment.strict(environment.regular(root/lane/'process.json').read_bytes())
        verify_review(canonical(lease['implementation_review']),review_hash,candidate,inputs)
        if (set(result)!={'kind','scope_id','status','lane','elapsed_seconds','returncode',
                          'active_processes','peak_tree_bytes','lease_sha256','files'}
                or any(type(result[k]) is not int for k in ('returncode','active_processes','peak_tree_bytes'))
                or type(result['elapsed_seconds']) is not float or result['lane']!=lane or result['scope_id']!=SCOPE
                or result['kind']!='G3C_HISTORY_CHILD_DIAGNOSTIC'
                or canonical(lease['candidate'])!=canonical(candidate) or canonical(lease['inputs'])!=canonical(inputs) or lease['review_sha256']!=review_hash
                or lease['lane']!=lane or lease['inventory']!=SMOKES[lane] or lease['scope_id']!=SCOPE
                or canonical(result)!=canonical(wave['results'][lane]) or result['status']!='PASSED' or result['returncode']!=0
                or result['active_processes']!=0 or not 0.<=result['elapsed_seconds']<=600.
                or not 0<=result['peak_tree_bytes']<=24*1024**3 or result['lease_sha256']!=sha256(raw).hexdigest()):
            raise ValueError('smoke receipt authority/resource mismatch')
        verify_child_files(root/lane,result,lease)


def run_wave(out, lanes, lease_base, review):
    started=time.monotonic(); deadline=started+1800; remaining=iter(lanes); active={}; results={}; failed=False
    with ThreadPoolExecutor(max_workers=3) as pool:
        def launch():
            name=next(remaining,None)
            if name is None: return False
            lease=dict(lease_base,lane=name,inventory=INVENTORIES[name])
            active[pool.submit(child,out/name,lease,review,deadline)]=name
            return True
        for _ in range(min(3,len(lanes))): launch()
        while active:
            finished,_=wait(active,timeout=.25,return_when=FIRST_COMPLETED)
            if time.monotonic()>=deadline: failed=True
            for future in finished:
                name=active.pop(future)
                try: result=future.result()
                except BaseException as error:
                    result=dict(status='FAILED',exception_type=type(error).__name__,terminal_zero_proven=False)
                results[name]=result
                if result['status']!='PASSED': failed=True
                print('G3C WAVE '+name+' '+result['status'],flush=True)
            if not failed:
                while len(active)<3 and launch(): pass
        # Already launched workers always reach terminal/drain handling. A
        # failure prevents unlaunched work; it never causes an automatic retry.
        for name in remaining: results[name]=dict(status='NOT_LAUNCHED')
    if time.monotonic()>=deadline: failed=True
    return dict(scope_id=SCOPE,kind='G3C_HISTORY_WAVE_DIAGNOSTIC',
                status='FAILED' if failed else 'PASSED',elapsed_seconds=time.monotonic()-started,
                results={name:results[name] for name in lanes})


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker': return worker(Path(sys.argv[2]),sys.argv[3])
    parser=argparse.ArgumentParser()
    parser.add_argument('--wave',choices=('smoke','restart'),required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--prior-smoke',type=Path)
    args=parser.parse_args()
    if os.name!='nt' or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('Windows -I -S -B coordinator required')
    candidate,inputs,review=authority(args.review,args.review_sha256)
    if args.wave=='restart':
        if args.prior_smoke is None: raise ValueError('five-family smoke prerequisite')
        verify_prior_smoke(args.prior_smoke,candidate,inputs,args.review_sha256)
    out=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-history-'))
    print('DIAGNOSTICS '+str(out),flush=True)
    lease=dict(kind='G3C_STABLE_PRIVATE_DEVELOPMENT',scope_id=SCOPE,candidate=candidate,inputs=inputs,
        source_map_sha256=inherited.MAP_SHA,review_sha256=args.review_sha256,implementation_review=environment.strict(review))
    result=run_wave(out,SMOKES if args.wave=='smoke' else PREFIXES,lease,review)
    final_candidate,final_inputs,final_review=authority(args.review,args.review_sha256)
    if (final_candidate,final_inputs,final_review)!=(candidate,inputs,review): raise ValueError('wave authority changed')
    result['wave']=args.wave
    write(out/'wave.json',result)  # diagnostic status, never a scientific GO
    print(canonical(result).decode(),flush=True)
    return int(result['status']!='PASSED')


if __name__=='__main__': raise SystemExit(main())
