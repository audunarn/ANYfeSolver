"""Corrected consumer-only waves. Original producers retain their own authority."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_ge_beam3_g3c_history as old
import ge_beam3_g3c_rehearsal_contract as design
import ge_beam3_g3c_rehearsal_packets as packets
import ge_beam3_g3c_rehearsal_mutations as mutations
import ge_beam3_g3c_history_owner as history
import ge_beam3_g3c_rehearsal_bridge as bridge

inherited=old.inherited
environment=old.environment
canonical=old.canonical
write=packets.publish
BASE='3334574546007e835c37eaebf3e1438514517d9d'
BASE_TREE='8e10f224514c1aefbc4058a42c50cffe16460458'
SCOPE=bridge.contract.expected()['consumer_scope']
TEST=bridge.producer.TEST
ADDED=set(bridge.contract.expected()['implementation_add'])
MODIFIED=set(bridge.contract.expected()['correction_modify'])


def registry():
    assignments,waves=bridge.producer.registry()
    selected=bridge.contract.expected()['consumer_waves']
    waves={name:waves[name] for name in selected}
    lanes={name for names in waves.values() for name in names}
    return {name:assignments[name] for name in assignments if name in lanes},waves


ASSIGNMENTS,WAVES=registry()
INVENTORIES={name:[TEST] for name in ASSIGNMENTS}


def verify_review(raw,expected,candidate,inputs):
    inherited.verify_review(raw,expected,candidate,inputs)
    scope=environment.strict(raw)['scope']
    if (scope.get('scope_id')!=SCOPE or scope.get('bridge_contract_sha256')!=bridge.CONTRACT_SHA
            or scope.get('producer_bridge')!=bridge.identity()):
        raise ValueError('corrected consumer implementation scope required')


def check_deadline(deadline):
    if deadline is not None and time.monotonic()>=deadline:
        raise TimeoutError('complete coordinator wave deadline exceeded')


def verify_extent(changed):
    expected={**{path:'A' for path in ADDED},**{path:'M' for path in MODIFIED}}
    if changed!=expected: raise ValueError('exact consumer correction extent')


def authority(review_path,expected,deadline=None):
    check_deadline(deadline)
    candidate=inherited.clean_identity()
    if inherited.git('rev-parse',BASE+'^{tree}')!=BASE_TREE: raise ValueError('consumer base tree')
    inherited.git('merge-base','--is-ancestor',BASE,'HEAD')
    changed={}
    for row in inherited.git('diff','--name-status','--no-renames',BASE,'HEAD').splitlines():
        kind,path=row.split('\t'); changed[path]=kind
    verify_extent(changed)
    bridge.design_authority()
    design.inherited.audit(design.parent())
    history.packet.authorities()
    mapping=inherited.source_map.strict(inherited.source_map.normalized(ROOT/inherited.source_map.MAP))
    inherited.source_map.audit(mapping,implemented=True)
    inputs=inherited.inputs(); review=environment.regular(review_path).read_bytes()
    verify_review(review,expected,candidate,inputs)
    bridge.verify(deadline)
    check_deadline(deadline)
    environment.verify(inherited.infrastructure.CAPSULE,inherited.infrastructure.CAPSULE_SHA)
    check_deadline(deadline)
    return candidate,inputs,review


def completion_expected(lease):
    return dict(kind='G3C_REHEARSAL_WORKER_COMPLETION',scope_id=SCOPE,lane=lease['lane'],
        candidate=lease['candidate'],inputs_sha256=sha256(canonical(lease['inputs'])).hexdigest(),
        inventory=lease['inventory'],collected_nodes=1,passed_nodes=1,nonpassing=False,exit_code=0,
        lease_sha256=sha256(canonical(lease)).hexdigest())


fingerprint=old.fingerprint


def verify_child_files(out,record,lease):
    expected={'stdout.log','stderr.log','completion.json','artifacts.json'}
    if set(record['files'])!=expected: raise ValueError('complete rehearsal files required')
    for name in expected:
        if canonical(fingerprint(out/name))!=canonical(record['files'][name]): raise ValueError('rehearsal raw hash mismatch')
    if canonical(environment.strict(environment.regular(out/'completion.json').read_bytes()))!=canonical(completion_expected(lease)):
        raise ValueError('actual rehearsal test completion mismatch')
    artifact=environment.strict(environment.regular(out/'artifacts.json').read_bytes())
    assignment=ASSIGNMENTS[lease['lane']]
    if assignment['kind']=='history':
        row=assignment['case']
        packets.verify_directory(out/'packets',artifact,sha256(canonical(lease)).hexdigest(),row['case_id'],row['accepted_stages'],lease['runtime_sha256'])
    else:
        if set(artifact)!={'kind','lease_sha256','assignment','files','results'} or artifact['kind']!='G3C_REHEARSAL_ATTACK_DIAGNOSTICS':
            raise ValueError('attack artifact schema')
        if artifact['lease_sha256']!=sha256(canonical(lease)).hexdigest() or canonical(artifact['assignment'])!=canonical(assignment):
            raise ValueError('attack artifact authority')
        if {p.name for p in (out/'packets').iterdir()}!=set(artifact['files']): raise ValueError('attack file inventory')
        for name,bound in artifact['files'].items(): packets.read_bound(out/'packets',name,bound)
        expected_count=1 if assignment['kind']=='positive' else len(assignment['probes'])
        if len(artifact['results'])!=expected_count or any(r.get('passed') is not True for r in artifact['results']):
            raise ValueError('complete expected probe evidence')
        if assignment['kind']=='positive':
            if set(artifact['files'])!={'positive-replay.json'}: raise ValueError('positive replay raw evidence')
            result=artifact['results'][0]
            if (set(result)!={'passed','kind','origin','input_sha256'} or result['kind']!='GENUINE_ORIGIN_REPLAY'
                    or result['origin']!=assignment['origin'] or result['input_sha256']!=artifact['files']['positive-replay.json']['sha256']):
                raise ValueError('positive origin bytes mismatch')
        else:
            for index,(result,probe) in enumerate(zip(artifact['results'],assignment['probes'])):
                expected_keys={'category','member','rejection','expected_error','before_sha256','after_sha256','passed','origin','input_sha256'}
                if probe['category']=='R10_NORMAL_SOURCE': expected_keys.add('source_path')
                if (set(result)!=expected_keys or any(result[k]!=probe[k] for k in ('category','member','rejection','origin'))
                        or result['before_sha256']==result['after_sha256'] or type(result['expected_error']) is not str):
                    raise ValueError('exact negative member/phase evidence')
                suffix='json' if probe['member']=='changed_implementation_review' else 'bin'
                name=f'attack-{index:03d}.{suffix}'
                if name not in artifact['files'] or artifact['files'][name]['sha256']!=result['after_sha256']:
                    raise ValueError('negative changed bytes missing')
            if len(artifact['files'])!=expected_count: raise ValueError('negative file count mismatch')


def verify_exact_attacks(out,lease,assignment,origin):
    raw,digest=origin
    artifact=environment.strict(environment.regular(out/'artifacts.json').read_bytes())
    if assignment['kind']=='positive':
        if packets.read_bound(out/'packets','positive-replay.json',artifact['files']['positive-replay.json'])!=raw:
            raise ValueError('positive replay differs from bound producer bytes')
    else:
        for index,probe in enumerate(assignment['probes']):
            expected_raw,expected_record=mutations.attack_fixture(raw,probe['category'],probe['member'],lease['implementation_review'])
            expected_record.update(passed=True,origin=assignment['origin'],input_sha256=digest)
            if canonical(artifact['results'][index])!=canonical(expected_record):
                raise ValueError('registered attack identity/reason mismatch')
            suffix='json' if probe['member']=='changed_implementation_review' else 'bin'
            name=f'attack-{index:03d}.{suffix}'
            if packets.read_bound(out/'packets',name,artifact['files'][name])!=expected_raw:
                raise ValueError('registered attack bytes mismatch')


def prior_chain(reference,wave,candidate,inputs,review_hash,deadline=None):
    """Root externally bound; every predecessor and child checked, no cycles."""
    keys=list(WAVES); index=keys.index(wave); collected=bridge.verify(deadline)
    for expected_wave in reversed(keys[:index]):
        check_deadline(deadline)
        if type(reference) is not dict or set(reference)!={'path','bytes','sha256'}:
            raise ValueError('required predecessor wave reference')
        path=environment.regular(Path(reference['path']))
        raw=path.read_bytes()
        if canonical(packets.fingerprint(raw))!=canonical({k:reference[k] for k in ('bytes','sha256')}):
            raise ValueError('predecessor wave hash')
        value=environment.strict(raw)
        if (set(value)!={'kind','scope_id','wave','status','elapsed_seconds','results','previous'}
                or value['kind']!='G3C_REHEARSAL_WAVE_DIAGNOSTIC' or value['scope_id']!=SCOPE
                or value['wave']!=expected_wave or value['status']!='PASSED'
                or type(value['elapsed_seconds']) is not float or not 0<=value['elapsed_seconds']<1800
                or set(value['results'])!=set(WAVES[expected_wave])):
            raise ValueError('complete ordered prerequisite wave required')
        for lane in WAVES[expected_wave]:
            check_deadline(deadline)
            out=path.parent/lane
            lease_raw=environment.regular(out/'lease.json').read_bytes(); lease=environment.strict(lease_raw)
            process=environment.strict(environment.regular(out/'process.json').read_bytes())
            if (set(process)!={'kind','scope_id','status','lane','elapsed_seconds','returncode','active_processes','peak_tree_bytes','lease_sha256','files'}
                    or process['kind']!='G3C_REHEARSAL_CHILD_DIAGNOSTIC' or process['scope_id']!=SCOPE or process['lane']!=lane
                    or canonical(process)!=canonical(value['results'][lane]) or process['status']!='PASSED'
                    or type(process['returncode']) is not int or process['returncode']!=0
                    or type(process['active_processes']) is not int or process['active_processes']!=0
                    or type(process['elapsed_seconds']) is not float or not 0<=process['elapsed_seconds']<600
                    or type(process['peak_tree_bytes']) is not int or not 0<=process['peak_tree_bytes']<=24*1024**3
                    or process['lease_sha256']!=sha256(lease_raw).hexdigest()):
                raise ValueError('prerequisite terminal tree/resource evidence')
            verify_lease(lease,candidate,inputs,review_hash,lane,expected_wave)
            if canonical(lease['previous'])!=canonical(value['previous']): raise ValueError('predecessor lease linkage')
            verify_review(environment.regular(out/'review.json').read_bytes(),review_hash,candidate,inputs)
            verify_child_files(out,process,lease)
            collected[lane]=(out,lease)
        reference=value['previous']
    if reference is not None: raise ValueError('unexpected predecessor root')
    for lane,(out,lease) in collected.items():
        check_deadline(deadline)
        if lane not in ASSIGNMENTS: continue  # Already validated, original producer role.
        assignment=ASSIGNMENTS[lane]
        _,digest=origin_packet(assignment,collected)
        artifacts=environment.strict(environment.regular(out/'artifacts.json').read_bytes())
        if any(row['input_sha256']!=digest for row in artifacts['results']):
            raise ValueError('probe origin differs from producer manifest')
        verify_exact_attacks(out,lease,assignment,origin_packet(assignment,collected))
    check_deadline(deadline)
    return collected


def verify_lease(lease,candidate,inputs,review_hash,lane,wave):
    if (set(lease)!={'kind','scope_id','candidate','inputs','source_map_sha256','review_sha256',
                    'implementation_review','lane','inventory','runtime_sha256','previous','wave','producer_bridge'}
            or lease['kind']!='G3C_STABLE_PRIVATE_DEVELOPMENT' or lease['scope_id']!=SCOPE
            or canonical(lease['candidate'])!=canonical(candidate) or canonical(lease['inputs'])!=canonical(inputs)
            or lease['review_sha256']!=review_hash or lease['lane']!=lane or lease['wave']!=wave
            or lane not in WAVES[wave] or lease['inventory']!=INVENTORIES[lane]
            or lease['runtime_sha256']!=history.runtime_identity() or lease['source_map_sha256']!=inherited.MAP_SHA):
        raise ValueError('rehearsal lease identity/inventory')
    bridge.validate_identity(lease['producer_bridge'])
    verify_review(canonical(lease['implementation_review']),review_hash,candidate,inputs)


def origin_packet(assignment,collected):
    return bridge.producer.origin_packet(assignment,collected)


def worker(out,expected):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode): raise ValueError('isolated worker required')
    raw=environment.regular(out/'lease.json').read_bytes()
    if sha256(raw).hexdigest()!=expected: raise ValueError('external lease digest')
    lease=environment.strict(raw); candidate,inputs,review=authority(out/'review.json',lease['review_sha256'])
    lane=lease['lane']; verify_lease(lease,candidate,inputs,lease['review_sha256'],lane,lease['wave'])
    collected=prior_chain(lease['previous'],lease['wave'],candidate,inputs,lease['review_sha256'])
    assignment=ASSIGNMENTS[lane]
    origin=None if assignment['kind']=='history' else origin_packet(assignment,collected)
    print('G3C CHECKPOINT rehearsal authority',flush=True)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(inherited.infrastructure.CAPSULE.parent/'site')]
    from anyfesolver._ge_beam3_g3c_stable import authority as runtime
    runtime.capture(raw)
    import pytest
    class Context:
        @pytest.fixture
        def rehearsal_context(self): return dict(out=out,lease=lease,assignment=assignment,origin=origin)
    inventory=inherited.Inventory(1)
    code=pytest.main(['-vv','-s','-p','no:cacheprovider','--basetemp',str(out/'pytest'),TEST],plugins=[inventory,Context()])
    if authority(out/'review.json',lease['review_sha256'])!=(candidate,inputs,review): raise ValueError('final frozen authority')
    prior_chain(lease['previous'],lease['wave'],candidate,inputs,lease['review_sha256'])
    if code or inventory.nonpassing or inventory.count!=1 or inventory.passed!=1: return 1
    write(out/'completion.json',completion_expected(lease))
    print('G3C CHECKPOINT rehearsal complete',flush=True)
    return 0


def child(out, lease, review, deadline):
    out.mkdir()  # distinct exclusive directory; never resumed or reused
    with (out/'review.json').open('xb') as stream: stream.write(review)
    write(out/'lease.json',lease)
    lease_hash=sha256((out/'lease.json').read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('rehearsal_process_'+lease['lane'].replace('-','_'),ROOT/'docs/reference_cases/e4_pl_s3_v2_bounded_process.py')
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
    record=dict(kind='G3C_REHEARSAL_CHILD_DIAGNOSTIC',scope_id=SCOPE,status=status,lane=lease['lane'],
        elapsed_seconds=time.monotonic()-start,returncode=None if process is None else process.returncode,active_processes=accounting[1],
        peak_tree_bytes=accounting[2],lease_sha256=lease_hash,
        files={name:fingerprint(out/name) for name in ('stdout.log','stderr.log','completion.json','artifacts.json') if (out/name).exists()})
    if status=='PASSED':
        try:
            check_deadline(deadline)
            verify_child_files(out,record,lease)
            if ASSIGNMENTS[lease['lane']]['kind']!='history':
                collected=prior_chain(lease['previous'],lease['wave'],lease['candidate'],lease['inputs'],lease['review_sha256'],deadline)
                verify_exact_attacks(out,lease,ASSIGNMENTS[lease['lane']],origin_packet(ASSIGNMENTS[lease['lane']],collected))
            check_deadline(deadline)
        except (ValueError,OSError): record['status']='FAILED_EVIDENCE'
    write(out/'process.json',record)
    return record


def run_wave(out, lanes, lease_base, review,started=None):
    started=time.monotonic() if started is None else started
    deadline=started+1800; remaining=iter(lanes); active={}; results={}; failed=False
    check_deadline(deadline)
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
    return dict(scope_id=SCOPE,kind='G3C_REHEARSAL_WAVE_DIAGNOSTIC',
                status='FAILED' if failed else 'PASSED',elapsed_seconds=time.monotonic()-started,
                results={name:results[name] for name in lanes})


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker': return worker(Path(sys.argv[2]),sys.argv[3])
    started=time.monotonic(); deadline=started+1800
    parser=argparse.ArgumentParser()
    parser.add_argument('--wave',choices=WAVES,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--previous-wave',type=Path)
    parser.add_argument('--previous-sha256')
    args=parser.parse_args()
    if os.name!='nt' or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('Windows -I -S -B coordinator required')
    candidate,inputs,review=authority(args.review,args.review_sha256,deadline)
    previous=None
    if args.previous_wave is not None:
        raw=environment.regular(args.previous_wave).read_bytes()
        if sha256(raw).hexdigest()!=args.previous_sha256: raise ValueError('external predecessor digest')
        previous=dict(path=str(args.previous_wave.absolute()),**packets.fingerprint(raw))
    elif args.previous_sha256 is not None: raise ValueError('predecessor path missing')
    prior_chain(previous,args.wave,candidate,inputs,args.review_sha256,deadline)
    out=Path(tempfile.mkdtemp(prefix='anyfesolver-g3c-consumer-'))
    print('DIAGNOSTICS '+str(out),flush=True)
    lease=dict(kind='G3C_STABLE_PRIVATE_DEVELOPMENT',scope_id=SCOPE,candidate=candidate,inputs=inputs,
               source_map_sha256=inherited.MAP_SHA,review_sha256=args.review_sha256,
               implementation_review=environment.strict(review),runtime_sha256=history.runtime_identity(),
               previous=previous,wave=args.wave,producer_bridge=bridge.identity())
    result=run_wave(out,WAVES[args.wave],lease,review,started)
    if authority(args.review,args.review_sha256,deadline)!=(candidate,inputs,review): raise ValueError('wave final authority')
    prior_chain(previous,args.wave,candidate,inputs,args.review_sha256,deadline)
    check_deadline(deadline)
    result['elapsed_seconds']=time.monotonic()-started
    result.update(wave=args.wave,previous=previous)
    write(out/'wave.json',result,deadline)  # diagnostic only, never full G3c GO
    print(canonical(result).decode(),flush=True)
    return int(result['status']!='PASSED')


if __name__=='__main__': raise SystemExit(main())
