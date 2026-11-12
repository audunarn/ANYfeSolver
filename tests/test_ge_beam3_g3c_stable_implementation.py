"""Inert implementation checks; intentionally imports no anyfesolver/NumPy."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import ge_beam3_g3c_stable_map as mapping
import run_ge_beam3_g3c_stable as runner


class ImplementationTests(unittest.TestCase):
    def test_exact_generated_sources(self):
        contract = mapping.strict(mapping.normalized(ROOT/mapping.MAP))
        self.assertEqual(mapping.audit(contract, implemented=True)['copies'], 13)

    def test_no_numerical_imports_during_preflight(self):
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('anyfesolver', sys.modules)
        for path in runner.EXTRA:
            if path.endswith('.py'):
                ast.parse(mapping.normalized(ROOT/path))

    def test_review_fail_closed(self):
        candidate = {'commit':'test', 'tree':'tree'}
        frozen = {'file': {'bytes':1, 'sha256':'0'*64}}
        review = dict(decision='ACCEPTED_G3C_STABLE_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT',
            findings=[], reviewer={'independent':True}, subject_commit='test',
            scope=dict(subject_tree='tree', source_map_sha256=runner.MAP_SHA,
                       inputs_sha256=runner.sha256(runner.canonical(frozen)).hexdigest()))
        raw = runner.canonical(review)
        runner.verify_review(raw, runner.sha256(raw).hexdigest(), candidate, frozen)
        for key, value in (('findings', ['unresolved']), ('subject_commit', 'foreign'),
                           ('decision', 'UNREVIEWED')):
            changed = copy.deepcopy(review); changed[key] = value
            raw = runner.canonical(changed)
            with self.assertRaises(ValueError):
                runner.verify_review(raw, runner.sha256(raw).hexdigest(), candidate, frozen)
        with self.assertRaises(ValueError):
            runner.verify_review(runner.canonical(review), '0'*64, candidate, frozen)

    def test_closed_inventory_and_limits(self):
        tree = ast.parse(mapping.normalized(ROOT/runner.GUARD))
        tests = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
        self.assertEqual(len(tests), runner.EXPECTED_NODES['guard'])
        self.assertEqual(runner.EXPECTED_NODES, {'guard':10, 'diagnostic':1, 'bridge':18})
        text = mapping.normalized(ROOT/'scripts/run_ge_beam3_g3c_stable.py').decode()
        for limit in ('now-start >= 600', 'now-last >= 120', '24*1024**3'):
            self.assertIn(limit, text)

    def test_complete_tree_cleanup_and_failure_diagnostics(self):
        for terminate_result, final_active in ((False, 1), (True, 1), (True, 0)):
            job, process = Mock(), Mock()
            job.accounting.side_effect = [(1, 1, 3), (2, final_active, 4)]
            job.terminate.return_value = terminate_result
            failures = []
            if not terminate_result or final_active:
                with self.assertRaises(RuntimeError):
                    runner.close_tree(job, process, failures.append)
                self.assertEqual(len(failures), 1)
                self.assertIs(failures[0]['terminal_zero_proven'], False)
            else:
                self.assertEqual(runner.close_tree(job, process, failures.append), (2, 0, 4))
                self.assertEqual(failures, [])
            job.close.assert_called_once()
            job.terminate.assert_called_once()

    def test_wait_and_diagnostic_failures_still_close_job(self):
        for diagnostic_fails in (False, True):
            job, process = Mock(), Mock()
            job.accounting.return_value = (1, 0, 3)
            process.wait.side_effect = TimeoutError('injected wait failure')
            errors = []
            def record(row):
                errors.append(row)
                if diagnostic_fails:
                    raise OSError('injected diagnostic failure')
            with self.assertRaises((TimeoutError, OSError)):
                runner.close_tree(job, process, record)
            self.assertEqual(len(errors), 1)
            job.close.assert_called_once()
            job.terminate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
