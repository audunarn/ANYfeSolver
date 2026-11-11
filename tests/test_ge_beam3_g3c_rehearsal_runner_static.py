"""Inert runner/scheduler guards; no scientific subprocesses."""
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import copy
import sys
import tempfile
import threading
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_ge_beam3_g3c_rehearsal as r
import test_ge_beam3_g3c_rehearsal_mutations_static as syntax


class RunnerStaticTests(unittest.TestCase):
    def test_rehashed_attack_reason_origin_and_bytes_rejected(self):
        raw=syntax.fake_origin('J_B2_PAIR','NONE',.01,2)
        digest=sha256(raw).hexdigest()
        assignment=r.ASSIGNMENTS['replay-00']; probe=assignment['probes'][0]
        attack,record=r.mutations.attack_fixture(raw,probe['category'],probe['member'],{})
        record.update(passed=True,origin=assignment['origin'],input_sha256=digest)
        for mutation in ('none','reason','before','attack'):
            with tempfile.TemporaryDirectory() as d:
                out=Path(d); (out/'packets').mkdir(); changed=copy.deepcopy(record); data=attack
                if mutation=='reason': changed['expected_error']='invented failure'
                if mutation=='before': changed['before_sha256']='0'*64
                if mutation=='attack':
                    data=attack+b' '; changed['after_sha256']=sha256(data).hexdigest()
                r.packets.exclusive(out/'packets'/'attack-000.bin',data)
                value=dict(files={'attack-000.bin':r.packets.fingerprint(data)},results=[changed])
                r.packets.publish(out/'artifacts.json',value)
                call=lambda:r.verify_exact_attacks(out,dict(implementation_review={}),assignment,(raw,digest))
                if mutation=='none': call()
                else:
                    with self.assertRaises(ValueError): call()

    def test_shared_wave_deadline_includes_preflight_and_finalization(self):
        with patch.object(r.time,'monotonic',return_value=2000.):
            with self.assertRaises(TimeoutError): r.check_deadline(1999.)
            with patch.object(r,'child') as child:
                with self.assertRaises(TimeoutError): r.run_wave(Path('.'),r.WAVES['none'],{},b'',started=0.)
                child.assert_not_called()
        # The same main start must be passed through both authority checks,
        # prior-chain checks and final atomic publication, not reset by a wave.
        import inspect
        source=inspect.getsource(r.main)
        self.assertEqual(source.count('started=time.monotonic()'),1)
        self.assertIn('lease,review,started)',source)
        self.assertIn("write(out/'wave.json',result,deadline)",source)

    def test_exact_separate_wave_inventories(self):
        self.assertEqual([len(v) for v in r.WAVES.values()],[5,5,2,2,4,6,6,6,6,2])
        probes=[p for a in r.ASSIGNMENTS.values() for p in a.get('probes',[])]
        self.assertEqual(len(probes),142)
        self.assertEqual({r.canonical(p) for p in probes},{r.canonical(p) for p in r.design.expected()['mutation_probes']})
        self.assertEqual(len(r.ASSIGNMENTS),44)
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('anyfesolver',sys.modules)

    def test_review_scope_and_hash_rejection(self):
        candidate=dict(commit='1'*40,tree='2'*40); inputs={}
        review=dict(decision='ACCEPTED_G3C_STABLE_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT',findings=[],
            reviewer=dict(independent=True),subject_commit=candidate['commit'],scope=dict(subject_tree=candidate['tree'],
            source_map_sha256=r.inherited.MAP_SHA,inputs_sha256=sha256(r.canonical(inputs)).hexdigest(),scope_id=r.SCOPE))
        raw=r.canonical(review); digest=sha256(raw).hexdigest()
        r.verify_review(raw,digest,candidate,inputs)
        review['scope']['scope_id']='old smoke only'; changed=r.canonical(review)
        with self.assertRaises(ValueError): r.verify_review(changed,sha256(changed).hexdigest(),candidate,inputs)
        with self.assertRaises(ValueError): r.verify_review(raw,'0'*64,candidate,inputs)

    def test_missing_predecessor_and_initial_extras_rejected(self):
        self.assertEqual(r.prior_chain(None,'none',{}, {}, '0'*64),{})
        with self.assertRaises(ValueError): r.prior_chain(None,'cm3',{}, {}, '0'*64)
        with self.assertRaises(ValueError): r.prior_chain(dict(path='unused',bytes=1,sha256='0'*64),'none',{}, {}, '0'*64)
        with self.assertRaises(ValueError): r.origin_packet(dict(origin=dict(case_id='missing',prefix=2)),{})

    def test_scheduler_overlap_cap_distinct_dirs_and_no_retry(self):
        active=0; peak=0; seen=[]; lock=threading.Lock()
        def child(out,lease,review,deadline):
            nonlocal active,peak
            with lock: active+=1; peak=max(peak,active); seen.append(out)
            time.sleep(.03)
            with lock: active-=1
            return dict(status='PASSED')
        with tempfile.TemporaryDirectory() as d, patch.object(r,'child',child):
            result=r.run_wave(Path(d),r.WAVES['none'],{},b'')
        self.assertEqual(result['status'],'PASSED'); self.assertEqual(peak,3)
        self.assertEqual(len(seen),5); self.assertEqual(len(set(seen)),5)
        seen=[]
        def failed(out,lease,review,deadline): seen.append(out); return dict(status='RESOURCE_BLOCKED')
        with tempfile.TemporaryDirectory() as d, patch.object(r,'child',failed):
            result=r.run_wave(Path(d),r.WAVES['none'],{},b'')
        self.assertEqual(result['status'],'FAILED'); self.assertEqual(len(seen),3)
        self.assertEqual(sum(v['status']=='NOT_LAUNCHED' for v in result['results'].values()),2)

    def test_reparse_free_raw_hash_receipt_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); lease=dict(lane=r.WAVES['none'][0],candidate={},inputs={},inventory=[r.TEST])
            for name in ('stdout.log','stderr.log','completion.json','artifacts.json'):
                r.packets.exclusive(out/name,b'not evidence\n')
            record=dict(files={name:r.fingerprint(out/name) for name in ('stdout.log','stderr.log','completion.json','artifacts.json')})
            with self.assertRaises(ValueError): r.verify_child_files(out,record,lease)
            record['files'].pop('artifacts.json')
            with self.assertRaises(ValueError): r.verify_child_files(out,record,lease)

    def test_special_read_rejections_before_constructor(self):
        raw=syntax.fake_origin('J_B2_PAIR','NONE',.01,2); value=r.history.packet.strict(raw)
        runtime=r.history.runtime_identity(); value['runtime_sha256']=runtime; raw=r.canonical(value)
        calls=[]
        def forbidden(*args,**kwargs): calls.append(True); raise AssertionError('construction')
        for source,expected in [(r.history.packet.DEFINITION,'external runtime mismatch'),
                ('docs/reference_cases/ge_beam3_g3c_fixtures_v1.json','frozen authority changed: docs/reference_cases/ge_beam3_g3c_fixtures_v1.json')]:
            target=ROOT/source; original=Path.read_bytes
            def changed(path): return original(path)+b'\nNEGATIVE\n' if path.absolute()==target.absolute() else original(path)
            with patch.object(Path,'read_bytes',changed),patch.object(r.history,'HistoryOwner',forbidden):
                with self.assertRaises(ValueError) as got:
                    r.history.resume(raw,sha256(raw).hexdigest(),expected_runtime_sha256=runtime)
                self.assertEqual(str(got.exception),expected)
        self.assertEqual(calls,[])


if __name__=='__main__': unittest.main()
