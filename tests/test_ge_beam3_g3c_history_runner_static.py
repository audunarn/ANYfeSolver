"""Inert runner tests; worker stubs never execute numerical mechanics."""
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_ge_beam3_g3c_history as runner


class RunnerStaticTests(unittest.TestCase):
    def test_exact_separate_inventories_and_review_scope(self):
        self.assertEqual(len(runner.SMOKES),5)
        self.assertEqual(len(runner.PREFIXES),3)
        self.assertTrue(all(len(nodes)==1 for nodes in runner.INVENTORIES.values()))
        candidate=dict(commit='1'*40,tree='2'*40); inputs={'a':dict(bytes=1,sha256='3'*64)}
        review=dict(decision='ACCEPTED_G3C_STABLE_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT',findings=[],
            reviewer=dict(independent=True),subject_commit=candidate['commit'],
            scope=dict(subject_tree=candidate['tree'],source_map_sha256=runner.inherited.MAP_SHA,
                inputs_sha256=sha256(runner.canonical(inputs)).hexdigest(),scope_id=runner.SCOPE))
        raw=runner.canonical(review)
        runner.verify_review(raw,sha256(raw).hexdigest(),candidate,inputs)
        review['scope']['scope_id']='another_scope'; raw=runner.canonical(review)
        with self.assertRaises(ValueError): runner.verify_review(raw,sha256(raw).hexdigest(),candidate,inputs)

    def test_three_overlapping_workers_distinct_directories_no_retry(self):
        lock=threading.Lock(); barrier=threading.Barrier(3); live=0; maximum=0; names=[]; paths=[]
        first=set(list(runner.SMOKES)[:3])
        def child(out,lease,review,deadline):
            nonlocal live,maximum
            with lock:
                live+=1; maximum=max(maximum,live); names.append(lease['lane']); paths.append(out)
            if lease['lane'] in first: barrier.wait(timeout=5)
            with lock: live-=1
            return dict(status='PASSED')
        with patch.object(runner,'child',side_effect=child):
            result=runner.run_wave(Path('unused-inert-wave'),runner.SMOKES,{},b'')
        self.assertEqual(result['status'],'PASSED'); self.assertEqual(maximum,3)
        self.assertEqual(len(names),5); self.assertEqual(len(set(paths)),5)
        self.assertEqual(list(result['results']),list(runner.SMOKES))

    def test_failed_child_prevents_unstarted_work_without_retry(self):
        names=[]; hold=threading.Event()
        def child(out,lease,review,deadline):
            names.append(lease['lane'])
            if lease['lane']=='smoke-b2': return dict(status='FAILED')
            hold.wait(.2)
            return dict(status='PASSED')
        with patch.object(runner,'child',side_effect=child):
            result=runner.run_wave(Path('unused-inert-wave'),runner.SMOKES,{},b'')
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(len(names),3); self.assertEqual(len(set(names)),3)
        self.assertEqual(sum(r['status']=='NOT_LAUNCHED' for r in result['results'].values()),2)

    def test_inherited_cleanup_failure_always_closes_job(self):
        class Job:
            closed=False
            def accounting(self): return (0,1,1)
            def terminate(self): return False
            def close(self): self.closed=True
        job=Job(); failures=[]
        with self.assertRaises(RuntimeError): runner.inherited.close_tree(job,None,failures.append)
        self.assertTrue(job.closed); self.assertEqual(len(failures),1)
        self.assertFalse(failures[0]['terminal_zero_proven'])
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('anyfesolver',sys.modules)

    def test_prior_smoke_requires_logs_and_actual_completion_evidence(self):
        # Synthetic process receipts only; never offered to a real run.
        for mutation in ('none','missing_stdout','changed_stderr','truncated_stdout',
                         'rehashed_nonpass','rehashed_bool_count','missing_family'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='g3c-inert-receipt-') as directory:
                root=Path(directory); candidate=dict(commit='1'*40,tree='2'*40)
                inputs={'a':dict(bytes=1,sha256='3'*64)}
                review=dict(decision='ACCEPTED_G3C_STABLE_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT',findings=[],
                    reviewer=dict(independent=True),subject_commit=candidate['commit'],
                    scope=dict(subject_tree=candidate['tree'],source_map_sha256=runner.inherited.MAP_SHA,
                        inputs_sha256=sha256(runner.canonical(inputs)).hexdigest(),scope_id=runner.SCOPE))
                review_hash=sha256(runner.canonical(review)).hexdigest()
                results={}
                for lane,inventory in runner.SMOKES.items():
                    out=root/lane; out.mkdir()
                    lease=dict(kind='G3C_STABLE_PRIVATE_DEVELOPMENT',scope_id=runner.SCOPE,
                        candidate=candidate,inputs=inputs,source_map_sha256=runner.inherited.MAP_SHA,
                        review_sha256=review_hash,implementation_review=review,lane=lane,inventory=inventory)
                    runner.write(out/'lease.json',lease)
                    (out/'stdout.log').write_bytes(b'synthetic stdout, not a real pass\n')
                    (out/'stderr.log').write_bytes(b'synthetic stderr\n')
                    runner.write(out/'completion.json',runner.completion_expected(lease))
                    record=dict(kind='G3C_HISTORY_CHILD_DIAGNOSTIC',scope_id=runner.SCOPE,status='PASSED',
                        lane=lane,elapsed_seconds=1.,returncode=0,active_processes=0,peak_tree_bytes=1000,
                        lease_sha256=sha256(runner.canonical(lease)).hexdigest(),
                        files={name:runner.fingerprint(out/name) for name in ('stdout.log','stderr.log','completion.json')})
                    runner.write(out/'process.json',record); results[lane]=record
                wave=dict(scope_id=runner.SCOPE,kind='G3C_HISTORY_WAVE_DIAGNOSTIC',status='PASSED',
                          elapsed_seconds=5.,results=results,wave='smoke')
                runner.write(root/'wave.json',wave)
                runner.verify_prior_smoke(root,candidate,inputs,review_hash)
                out=root/'smoke-b2'
                if mutation=='missing_stdout': (out/'stdout.log').unlink()
                elif mutation=='changed_stderr': (out/'stderr.log').write_bytes(b'replaced')
                elif mutation=='truncated_stdout': (out/'stdout.log').write_bytes(b'')
                elif mutation in ('rehashed_nonpass','rehashed_bool_count'):
                    completion=runner.environment.strict((out/'completion.json').read_bytes())
                    completion['passed_nodes']=0 if mutation=='rehashed_nonpass' else True
                    (out/'completion.json').write_bytes(runner.canonical(completion))
                    results['smoke-b2']['files']['completion.json']=runner.fingerprint(out/'completion.json')
                    (out/'process.json').write_bytes(runner.canonical(results['smoke-b2']))
                    (root/'wave.json').write_bytes(runner.canonical(wave))
                elif mutation=='missing_family':
                    results.pop('smoke-loop'); (root/'wave.json').write_bytes(runner.canonical(wave))
                if mutation!='none':
                    with self.assertRaises((ValueError,OSError)):
                        runner.verify_prior_smoke(root,candidate,inputs,review_hash)
        self.assertNotIn('numpy',sys.modules)


if __name__=='__main__': unittest.main()
