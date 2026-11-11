"""Focused stdlib-only controls for the current-artifact remaining scale harness.

Source/AST/hash admission only: no numeric imports, no model builds, no
installed execution and no physical phase runs here.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/r8_b2_remaining_scale_dev.py'
spec=importlib.util.spec_from_file_location('b2_remaining_scale_control',path)
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)

V9=Path(__file__).resolve().parents[1]/'docs/agent_plans/development/r8-b2-assembly-series-20261010/execution-v9'
INVENTORY=V9/'attempt-install-01/installed-inventory.json'


def original_target():
    p=Path(__file__).resolve().parents[1]/'scripts/r8_b2_installed_original_target_dev.py'
    s=importlib.util.spec_from_file_location('b2_original_target_control',p)
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
    return m


class Controls(unittest.TestCase):
    def test_artifact_and_canonical_bindings(self):
        plan=json.loads((V9/'EXECUTABLE_PREPARED.json').read_text())
        self.assertEqual(r.SOLVER_SHA,plan['solver_sha256'])
        self.assertEqual(r.CONSUMER_SHA,plan['consumer_sha256'])
        self.assertEqual(r.SOLVER_SHA,'00385aa8dcfaa472b1f0860257cf83baad263db958a250b0d0c22e9ebe0bff15')
        self.assertEqual(r.CONSUMER_SHA,'84b738061f0627da5c68fb2e8efb0a0882a2c19ce0f1750595050923f58bd66b')
        self.assertEqual(r.FULL_SHA,plan['full_fixture']['sha256'])
        self.assertEqual(r.FULL_SHA,'cf7970c30839256a2eabc0f9fc2a5521e61b641b92b4a6bca2ee68a8ab70f8b4')

    def test_budgets_and_runtime_contract(self):
        self.assertEqual(r.PHASES,{'controls':20.,'full':240.,'beam':60.})
        self.assertEqual(r.ALLOWANCE,420.);self.assertLessEqual(r.ALLOWANCE,600.)
        self.assertEqual(r.RSS_BYTES,2*1024**3)
        self.assertEqual(r.ENV['PYTHONOPTIMIZE'],'0')
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','NUMBA_DISABLE_JIT'):
            self.assertEqual(r.ENV[name],'1')
        self.assertIn('missing on 00385aa8',r.G5_NOTE)
        self.assertIn('does not close aggregate G5',r.G5_NOTE)

    def test_conditional_beam_predecessors(self):
        self.assertEqual(r.predecessors('controls'),())
        self.assertEqual(r.predecessors('full'),('controls',))
        self.assertEqual(r.predecessors('beam'),('full',))
        with self.assertRaises(KeyError):r.predecessors('install')

    def test_product_namespace_package_only(self):
        inventory=json.loads(INVENTORY.read_text())
        package=r.product_package(inventory)
        expected={n for n in inventory['files'] if n.startswith(('anyfesolver/','anyfem/'))}
        self.assertEqual(set(package),expected)
        self.assertTrue(package and all('.dist-info' not in n and not n.startswith('bin/') for n in package))
        original=original_target()
        for name in original.B2_FILES:self.assertIn('anyfesolver/'+name,package)
        self.assertIn('anyfesolver/stiffened_shell.py',package)
        self.assertIn('anyfem/solve/stiffened_shell_b2.py',package)

    def test_execute_installed_reused_not_duplicated(self):
        source=path.read_text();tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='execute_installed'
            for n in ast.walk(tree)),'execute_installed must be reused, not copied')
        self.assertIn('execute_installed',source)
        self.assertNotIn('import csv',source);self.assertNotIn("'RECORD'",source)
        self.assertNotIn("ROOT/'src",source,'worktree product sources must not be asserted')
        child=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='child')
        names={n.id for n in ast.walk(child) if isinstance(n,ast.Name)}
        self.assertIn('CONTROL',names);self.assertIn('TEST',names)
        calls=[call for call in ast.walk(child) if isinstance(call,ast.Call)
            and isinstance(call.func,ast.Name) and call.func.id=='verify_installed_graph']
        labels=[a.value for call in calls for a in call.args if isinstance(a,ast.Constant)]
        self.assertEqual(sorted(labels),['after','after','before'])
        before=[call for call in calls if call.args[1].value=='before']
        after=[call for call in calls if call.args[1].value=='after']
        body_col=child.body[0].col_offset
        # One unconditional before check ahead of every phase branch.
        self.assertEqual(len(before),1)
        self.assertEqual(before[0].col_offset,body_col)
        branches=[node for node in child.body if isinstance(node,ast.If)]
        self.assertTrue(branches)
        self.assertLess(before[0].lineno,branches[0].lineno)
        # One after check on the controls success path inside its branch, and
        # one tail after check outside every guard so a failed full/beam child
        # still verifies the installed graph.
        self.assertEqual(len(after),2)
        nested=[call for call in after if call.col_offset>body_col]
        tail=[call for call in after if call.col_offset==body_col]
        self.assertEqual(len(nested),1)
        self.assertEqual(len(tail),1)
        guarded=[node for node in ast.walk(child) if isinstance(node,ast.If)
            and any(isinstance(name,ast.Name) and name.id=='passed' for name in ast.walk(node.test))]
        self.assertTrue(guarded)
        self.assertGreater(tail[0].lineno,guarded[0].lineno)

    def test_graph_mismatch_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp);(site/'anyfesolver').mkdir();(site/'anyfesolver'/'__init__.py').write_text('x')
            inventory=dict(files={'anyfesolver/__init__.py':'sha','anyfesolver/extra.py':'sha'})
            self.assertEqual(sorted(r.graph_mismatch(site,inventory)),
                ['bytes anyfesolver/__init__.py','missing anyfesolver/extra.py'])
            (site/'anyfesolver'/'extra.py').write_text('y');(site/'stray.txt').write_text('z')
            problems=r.graph_mismatch(site,inventory)
            self.assertTrue(any(p.startswith('extra ') for p in problems))
            (site/'anyfesolver'/'extra.py').unlink();(site/'stray.txt').unlink()
            import hashlib
            good=hashlib.sha256(b'x').hexdigest()
            self.assertEqual(r.graph_mismatch(site,dict(files={'anyfesolver/__init__.py':good})),[])

    def test_matrix_contracts_adopted(self):
        contracts=r.matrix_contracts();original=original_target()
        self.assertEqual(contracts,dict(original.EXPECTED_MATRICES))
        self.assertEqual(sorted(contracts),['C','K','T'])

    def test_connected_caller_ast(self):
        caller=r.connected_caller_ast()
        self.assertEqual(caller['status'],'REUSED_CONNECTED_CALLER_AST')
        self.assertTrue(caller['not_execution_evidence'])
        for name in ('solve_linear_many','close','create_analysis','definition','from_definition',
                'save_case','load_case','save_project','load_project','point_panel_from_analysis'):
            self.assertIn(name,caller['required'])

    def test_prepared_bindings_if_present(self):
        prepared=r.OUT/'PREPARED.json'
        if not prepared.is_file():self.skipTest('PREPARED.json not yet generated')
        p=json.loads(prepared.read_text())
        self.assertEqual(p['status'],'PREPARED_SLOT_HELD')
        self.assertTrue(all(n.startswith(('anyfesolver/','anyfem/')) for n in p['package']))
        graph=p['installed_graph']
        self.assertEqual(graph['files'],761)
        self.assertTrue(graph['reused_v9_verified_graph'])
        self.assertFalse(graph['raw_wheel_record_recheck'])
        self.assertTrue(graph['launchers'])
        self.assertEqual(p['full']['canonical_sha256'],r.FULL_SHA)
        self.assertEqual(p['full']['batch_sizes'],[1,10,100]);self.assertEqual(p['full']['primary_count'],111)
        self.assertEqual(p['beam']['batch_sizes'],[100]);self.assertEqual(p['beam']['primary_count'],100)
        self.assertEqual(p['matrix_sha256'],r.matrix_contracts())
        limits=p['limits']
        self.assertLessEqual(limits['total_seconds'],600.);self.assertEqual(limits['rss_bytes'],2*1024**3)
        self.assertEqual(limits['cpu'],[0]);self.assertEqual(limits['threads'],1)
        self.assertTrue(limits['jit_disabled']);self.assertEqual(limits['optimize'],0)
        self.assertTrue(limits['one_job'])
        self.assertFalse(p['g5_aggregate']['closes_aggregate_g5'])
        self.assertEqual(p['prior_analogous_pass']['artifact_solver_sha256'],
            '0ed73724bf3c8ccffcc0134628d94720d640840f9592ac8890779f54285ee3d8')

    def test_no_numeric_or_physical_module_imports(self):
        tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.Import):
                self.assertFalse(any(a.name.split('.')[0] in ('numpy','scipy','numba','psutil',
                    'threadpoolctl') for a in node.names))
            if isinstance(node,ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0],
                    ('numpy','scipy','numba','psutil','threadpoolctl'))


if __name__=='__main__':unittest.main()
