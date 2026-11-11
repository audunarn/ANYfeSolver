"""Inert consumer/bridge checks. Temporary fixtures are never numerical evidence."""
import ast
import copy
from hashlib import sha256
import inspect
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_ge_beam3_g3c_rehearsal_consumer as r
import test_ge_beam3_g3c_rehearsal_mutations_static as syntax
b=r.bridge


def review_fixture():
    candidate=dict(commit='1'*40,tree='2'*40); inputs={}
    value=dict(decision='ACCEPTED_G3C_STABLE_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT',findings=[],
        reviewer=dict(independent=True),subject_commit=candidate['commit'],scope=dict(subject_tree=candidate['tree'],
        source_map_sha256=r.inherited.MAP_SHA,inputs_sha256=sha256(r.canonical(inputs)).hexdigest(),scope_id=r.SCOPE,
        bridge_contract_sha256=b.CONTRACT_SHA,producer_bridge=b.identity()))
    return candidate,inputs,value


class ConsumerStaticTests(unittest.TestCase):
    def test_full_fixed_archive_and_original_inputs(self):
        collected=b.verify()
        self.assertEqual(len(collected),10)
        self.assertEqual(sum(b.producer.ASSIGNMENTS[k]['case']['accepted_stages']+1 for k in collected),80)
        for name in r.WAVES['positive']:
            raw,digest=r.origin_packet(r.ASSIGNMENTS[name],collected)
            self.assertEqual(sha256(raw).hexdigest(),digest)
        self.assertNotIn('numpy',sys.modules); self.assertNotIn('anyfesolver',sys.modules)

    def test_scope_review_hash_and_role_substitution(self):
        candidate,inputs,review=review_fixture()
        raw=r.canonical(review); r.verify_review(raw,sha256(raw).hexdigest(),candidate,inputs)
        for key,value in [('scope_id',b.producer.SCOPE),('bridge_contract_sha256','0'*64),('producer_bridge',{}),
                          ('subject_tree','3'*40),('inputs_sha256','0'*64)]:
            changed=copy.deepcopy(review); changed['scope'][key]=value; data=r.canonical(changed)
            with self.assertRaises(ValueError): r.verify_review(data,sha256(data).hexdigest(),candidate,inputs)
        with self.assertRaises(ValueError): r.verify_review(raw,'0'*64,candidate,inputs)

    def test_exact_extent_and_bridge_identity(self):
        expected={**dict.fromkeys(r.ADDED,'A'),**dict.fromkeys(r.MODIFIED,'M')}
        r.verify_extent(expected); b.validate_identity(b.identity())
        for path in expected:
            bad=dict(expected); bad.pop(path)
            with self.assertRaises(ValueError): r.verify_extent(bad)
        bad=dict(expected); bad['src/anyfesolver/_ge_beam3_g3c_stable/owner.py']='M'
        with self.assertRaises(ValueError): r.verify_extent(bad)
        for key in b.identity():
            bad=b.identity(); bad[key]='wrong'
            with self.assertRaises(ValueError): b.validate_identity(bad)

    def test_new_lease_binds_new_review_and_old_bridge_separately(self):
        candidate,inputs,review=review_fixture(); digest=sha256(r.canonical(review)).hexdigest()
        lane=r.WAVES['positive'][0]
        lease=dict(kind='G3C_STABLE_PRIVATE_DEVELOPMENT',scope_id=r.SCOPE,candidate=candidate,inputs=inputs,
            source_map_sha256=r.inherited.MAP_SHA,review_sha256=digest,implementation_review=review,
            lane=lane,inventory=r.INVENTORIES[lane],runtime_sha256=b.SPEC['runtime_sha256'],
            previous=None,wave='positive',producer_bridge=b.identity())
        r.verify_lease(lease,candidate,inputs,digest,lane,'positive')
        for key,value in [('scope_id',b.producer.SCOPE),('producer_bridge',{}),('runtime_sha256','0'*64),
                ('candidate',dict(commit=b.SPEC['commit'],tree=b.SPEC['tree'])),('wave','none')]:
            bad=copy.deepcopy(lease); bad[key]=value
            with self.assertRaises(ValueError): r.verify_lease(bad,candidate,inputs,digest,lane,'positive')
        self.assertEqual(r.completion_expected(lease)['scope_id'],r.SCOPE)
        self.assertNotEqual(r.completion_expected(lease),b.producer.completion_expected(lease))

    def test_original_lease_roles_and_terminal_failures(self):
        out=b.ARCHIVE/'none/none-j_b2_pair'; raw=(out/'lease.json').read_bytes(); lease=r.environment.strict(raw)
        process=r.environment.strict((out/'process.json').read_bytes())
        b.validate_process(process,process,lease['lane'],raw)
        for key,value in [('active_processes',1),('active_processes',False),('returncode',False),
                ('elapsed_seconds',600.),('peak_tree_bytes',24*1024**3+1),('status','FAILED'),('scope_id',r.SCOPE)]:
            bad=dict(process); bad[key]=value
            with self.assertRaises(ValueError): b.validate_process(bad,bad,lease['lane'],raw)
        for key,value in [('candidate',dict(commit='0'*40,tree=b.SPEC['tree'])),
                ('scope_id',r.SCOPE),('review_sha256','0'*64),('runtime_sha256','0'*64),('inputs',{})]:
            bad=copy.deepcopy(lease); bad[key]=value
            with self.assertRaises(ValueError):
                b.producer.verify_lease(bad,lease['candidate'],lease['inputs'],b.SPEC['implementation_review_sha256'],lease['lane'],'none')

    def test_unchanged_source_runtime_and_raw_packet_mutations(self):
        lease=r.environment.strict((b.ARCHIVE/'none/none-j_b2_pair/lease.json').read_bytes())
        original=Path.read_bytes
        target=ROOT/'src/anyfesolver/_ge_beam3_g1_analysis.py'
        def changed(path): return original(path)+b'\n' if path==target else original(path)
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaisesRegex(ValueError,'unchanged producer input changed'):
                b.original_inputs(lease['inputs'])
        target=b.ARCHIVE/'none/none-j_b2_pair/packets/prefix-002.json'
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaisesRegex(ValueError,'archived raw input changed'): b.archive_manifest()
        with patch.object(b.producer.history,'runtime_identity',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'original runtime changed'): b.verify()

    def test_exact_original_reference_mapping(self):
        manifest=b.archive_manifest()
        prior=r.environment.strict((b.ARCHIVE/'cm3/wave.json').read_bytes())['previous']
        b.previous_reference(None,'none',manifest); b.previous_reference(prior,'cm3',manifest)
        for key,value in [('path',str(b.ARCHIVE/'none/wave.json')),('bytes',True),('sha256','0'*64)]:
            bad=dict(prior); bad[key]=value
            with self.assertRaises(ValueError): b.previous_reference(bad,'cm3',manifest)
        with self.assertRaises(ValueError): b.previous_reference(prior,'none',manifest)

    def test_missing_failed_partial_and_old_scope_predecessor(self):
        with patch.object(b,'verify',return_value={}):
            self.assertEqual(r.prior_chain(None,'positive',{}, {}, '0'*64),{})
            with self.assertRaises(ValueError): r.prior_chain(None,'preflight',{}, {}, '0'*64)
            with self.assertRaises(ValueError): r.prior_chain(dict(path='unused',bytes=1,sha256='0'*64),'positive',{}, {}, '0'*64)
            for scope,status in [(r.SCOPE,'FAILED'),(r.SCOPE,'PASSED'),(b.producer.SCOPE,'PASSED')]:
                with tempfile.TemporaryDirectory() as d:
                    out=Path(d)/'wave.json'
                    value=dict(kind='G3C_REHEARSAL_WAVE_DIAGNOSTIC',scope_id=scope,wave='positive',status=status,
                               elapsed_seconds=1.,results={},previous=None)
                    r.packets.publish(out,value)
                    ref=dict(path=str(out),**r.packets.fingerprint(out.read_bytes()))
                    with self.assertRaises(ValueError): r.prior_chain(ref,'preflight',{}, {}, '0'*64)

    def test_all_probe_members_and_no_producer_routing(self):
        self.assertEqual(list(r.WAVES),b.contract.expected()['consumer_waves'])
        self.assertEqual([len(v) for v in r.WAVES.values()],[2,2,4,6,6,6,6,2])
        probes=[p for a in r.ASSIGNMENTS.values() for p in a['probes']]
        self.assertEqual(len(probes),142)
        self.assertEqual({r.canonical(p) for p in probes},{r.canonical(p) for p in r.design.expected()['mutation_probes']})
        self.assertFalse(any(a['kind']=='history' for a in r.ASSIGNMENTS.values()))

    def test_rehashed_wrong_r06_constant_rejected(self):
        raw=syntax.fake_origin('J_B2_PAIR','NONE',.01,2); digest=sha256(raw).hexdigest()
        assignment=copy.deepcopy(r.ASSIGNMENTS['preflight-0-preflight_before_construction'])
        assignment['probes']=[p for p in assignment['probes'] if p['category']=='R06_OLD_SCHEMA' and p['member']=='G1']
        probe=assignment['probes'][0]
        for wrong in (False,True):
            data,record=r.mutations.attack_fixture(raw,probe['category'],probe['member'],{})
            if wrong:
                value=r.environment.strict(data); value['schema']='GE_BEAM3_G1_ELASTIC_STATE_V1'; data=r.canonical(value)
                record['after_sha256']=sha256(data).hexdigest()
            record.update(passed=True,origin=assignment['origin'],input_sha256=digest)
            with tempfile.TemporaryDirectory() as d:
                out=Path(d); (out/'packets').mkdir(); r.packets.exclusive(out/'packets/attack-000.bin',data)
                r.packets.publish(out/'artifacts.json',dict(files={'attack-000.bin':r.packets.fingerprint(data)},results=[record]))
                call=lambda:r.verify_exact_attacks(out,dict(implementation_review={}),assignment,(raw,digest))
                if wrong:
                    with self.assertRaises(ValueError): call()
                else: call()

    def test_scheduler_overlap_unique_directories_and_no_retry(self):
        active=0; peak=0; seen=[]; lock=threading.Lock()
        def child(out,lease,review,deadline):
            nonlocal active,peak
            with lock: active+=1; peak=max(peak,active); seen.append(out)
            time.sleep(.02)
            with lock: active-=1
            return dict(status='PASSED')
        with tempfile.TemporaryDirectory() as d,patch.object(r,'child',child):
            result=r.run_wave(Path(d),r.WAVES['negative-0'],{},b'')
        self.assertEqual(result['status'],'PASSED'); self.assertEqual(peak,3)
        self.assertEqual(len(seen),6); self.assertEqual(len(set(seen)),6)
        seen=[]
        def failed(out,lease,review,deadline): seen.append(out); return dict(status='RESOURCE_BLOCKED')
        with tempfile.TemporaryDirectory() as d,patch.object(r,'child',failed):
            result=r.run_wave(Path(d),r.WAVES['negative-0'],{},b'')
        self.assertEqual(result['status'],'FAILED'); self.assertEqual(len(seen),3)

    def test_inherited_tree_bounds_and_shared_publication_deadline(self):
        # These implementations are unchanged; only module-owned scope/entrypoint
        # differs. Parent runner drains the actual Windows Job in finally.
        for name in ('child','run_wave'):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(r,name)))),
                             ast.dump(ast.parse(inspect.getsource(getattr(b.producer,name)))))
        with patch.object(r.time,'monotonic',return_value=2000.),patch.object(r,'child') as child:
            with self.assertRaises(TimeoutError): r.run_wave(Path('.'),r.WAVES['positive'],{},b'',started=0.)
            child.assert_not_called()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wave.json'
            with self.assertRaises(TimeoutError): r.write(path,{'synthetic':True},deadline=0.)
            self.assertFalse(path.exists()); self.assertTrue(path.with_name('wave.json.pending').exists())
        source=inspect.getsource(r.main)
        self.assertEqual(source.count('started=time.monotonic()'),1)
        self.assertIn('lease,review,started)',source)
        self.assertIn("write(out/'wave.json',result,deadline)",source)


if __name__=='__main__': unittest.main()
