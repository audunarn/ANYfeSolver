"""Prepared stdlib controls; not executed under preparation-only authority."""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('assembly_series',ROOT/'scripts/r8_b2_assembly_series_dev.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class Controls(unittest.TestCase):
    def test_geometry_counts(self):
        for size,members,pads,dofs in ((2500,250,25,17136),(5000,500,50,33936),(10000,1000,100,67266)):
            i=s.inventory(size);self.assertEqual((i['shells'],i['members'],i['pads'],i['dofs']),(size,members,pads,dofs))
    def test_deterministic_load_support(self):
        self.assertEqual(s.inventory(5000)['loaded_dof'],15752);self.assertEqual(s.inventory(5000)['fixed_dofs'],306)
        self.assertEqual(s.CONFIGS[10000]['line_rows'],list(range(4,100,10)))
    @unittest.skipUnless(s.FULL.exists(),
        'local assembly-series full fixture evidence '
        '(docs/agent_plans/development/r8-b2-installed-batch-persistence-20261010) is absent')
    def test_frozen_full_identity(self):self.assertEqual(s.sha(s.FULL),s.FULL_SHA)
    def test_source_scatter_proof(self):
        import ast
        tree=ast.parse((ROOT/'src/anyfesolver/_stiffened_shell_b2_q4_panel_experimental.py').read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_scatter')
        calls=[x for x in ast.walk(fn) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='extend']
        self.assertEqual(len(calls),3);self.assertEqual({x.func.value.id for x in calls},{'rows','columns','values'})
        release=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_release_coo_scratch')
        clears=[x for x in ast.walk(release) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='clear']
        self.assertEqual({x.func.value.id for x in clears},{'rows','columns','values'})
        caller=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='__init__' and any(isinstance(y,ast.Name) and y.id=='_release_coo_scratch' for y in ast.walk(x)))
        releases=[x for x in ast.walk(caller) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_release_coo_scratch']
        self.assertEqual(len(releases),2)
    def test_inventory_caps_no_silent_relaxation(self):
        self.assertEqual(s.cap_obstruction(2500)['preclear_scalar_entries'],4428000)
        self.assertEqual(s.cap_obstruction(2500)['scalar_objects_lower_bound'],0)
        for size in s.CONFIGS:self.assertFalse(s.cap_obstruction(size)['executable'])
        with self.assertRaises(RuntimeError):s.installed_samples(None,{},2500,[1])
        if not ((s.OUT/'PREPARED.json').exists() and (s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').exists()):
            self.skipTest('local assembly-series authority evidence '
                '(docs/agent_plans/development/r8-b2-assembly-series-20261010) is absent; '
                'tampered-authority refusals require the exact '
                'frozen grant bytes')
        import json,copy
        a=json.loads((s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').read_text());p=json.loads((s.OUT/'PREPARED.json').read_text())
        p['predecessor_prepared_sha256']=s.sha(s.OUT/'PREPARED.json');s.authority_gate(a,p,'low')
        for key,value in [('tree_rss_limit_bytes',4*1024**3),('input_prepared_sha256','wrong'),('jit_disabled',False)]:
            changed=copy.deepcopy(a);changed[key]=value
            with self.assertRaises(RuntimeError):s.authority_gate(changed,p,'low')
    def test_ratio_and_five_samples(self):
        groups={size:[dict(physical_pass=True,assembly_seconds=n,selected_prefactor_bytes=n*100) for _ in range(5)] for size,n in ((2500,1.),(5000,2.),(10000,4.))}
        self.assertTrue(s.summarize(groups)['screen_pass']);groups[10000][0]['physical_pass']=False
        with self.assertRaises(RuntimeError):s.summarize(groups)
    def test_empty_failed_samples_refuse(self):
        for g in ({},{size:[] for size in s.CONFIGS}):
            with self.assertRaises(RuntimeError):s.summarize(g)
        from types import SimpleNamespace
        for count,skips in ((0,[]),(8,[('test','skip')])):
            result=SimpleNamespace(testsRun=count,skipped=skips,errors=[],failures=[],wasSuccessful=lambda:True)
            with self.assertRaises(RuntimeError):s.adjudicate_controls(result)
    def test_negative_and_missing_phase_accounting(self):
        capture=dict(status='COMPLETE',phases=[dict(phase=n,seconds=1.) for n in ('shell_response_scatter','beam_response_scatter','K_CSR','attachment_rows','C_CSR')],
            assembly_seconds=5.,storage=dict(caps=s.CAPS,total_bytes=100),upper_bound=False)
        self.assertEqual(s.accounting_sample(capture,6.,True,1)['assembly_seconds'],5.)
        for wall in (-1.,4.,float('nan')):
            with self.assertRaises(RuntimeError):s.accounting_sample(capture,wall,True,1)
        capture['phases'].pop()
        with self.assertRaises(RuntimeError):s.accounting_sample(capture,6.,True,1)

if __name__=='__main__':unittest.main()
