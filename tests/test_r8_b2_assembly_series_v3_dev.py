"""Prepared stdlib controls; not executed under preparation-only authority."""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('assembly_series_v3',ROOT/'scripts/r8_b2_assembly_series_v3_dev.py')
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
        if not ((s.OUT/'EXECUTABLE_PREPARED.json').exists() and (s.OUT/'PREPARED.json').exists() and (s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').exists()):
            self.skipTest('local assembly-series authority evidence '
                '(docs/agent_plans/development/execution-v3) is absent; '
                'tampered-authority refusals require the exact '
                'frozen grant bytes')
        import json,copy
        a=json.loads((s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').read_text());p=json.loads((s.OUT/'EXECUTABLE_PREPARED.json').read_text())
        p['predecessor_prepared_sha256']=s.sha(s.OUT/'PREPARED.json');a['slot_status']='TRANSFERRED';s.authority_gate(a,p,'low')
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

import csv,base64,hashlib,io,json,tempfile,zipfile

class LauncherControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.site=self.root/'site';self.site.mkdir()
        stub=self.root/'stub.exe';stub.write_bytes(b'MZ-exact-test-stub')
        console=self.root/'python.exe';console.write_bytes(b'console-binary');gui=self.root/'pythonw.exe';gui.write_bytes(b'gui-binary')
        self.prov=dict(interpreter=str(console),gui_executable=dict(path=str(gui),sha256=s.sha(gui)),template='from %(module)s import %(import_name)s\n%(func)s()\n',resources={k:dict(path=str(stub),sha256=s.sha(stub)) for k in ('t64','w64')})
        self.declaration=dict(target='fixture.module:main',kind='t64',record='fixture.dist-info/RECORD')
        self.raw=self.make_launcher()
    def make_launcher(self,target=None,interpreter=None,kind='t64'):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:z.writestr('__main__.py',s.launcher_payload(self.prov['template'],target or self.declaration['target']))
        expected=self.prov['interpreter'] if kind=='t64' else self.prov['interpreter'][:-len('python.exe')]+'pythonw.exe'
        return Path(self.prov['resources'][kind]['path']).read_bytes()+b'#!'+(interpreter or expected).encode()+b'\n'+stream.getvalue()
    def fixture(self):
        wheels=[];files={};records=[]
        for package in ('solver','fixture'):
            record=package+'.dist-info/RECORD';records.append(record);content={package+'/product.py':b'unchanged',record:b''}
            if package=='fixture':content['fixture.dist-info/entry_points.txt']=b'[console_scripts]\nfixture = fixture.module:main\n'
            wheel=self.root/(package+'.whl')
            with zipfile.ZipFile(wheel,'w') as z:
                for n,data in content.items():z.writestr(n,data)
            wheels.append(str(wheel));files.update(content)
            for n in ('INSTALLER','REQUESTED','direct_url.json'):files[package+'.dist-info/'+n]=b'generated'
        files['bin/fixture.exe']=self.raw
        for n,data in files.items():q=self.site/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
        for record in records:
            package=record.split('.')[0];rows=[]
            for n,data in files.items():
                if not (n.startswith(package+'/') or n.startswith(package+'.dist-info/') or package=='fixture' and n.startswith('bin/')):continue
                if n==record:rows.append([n,'','']);continue
                digest='sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                rows.append(['../../'+n if n.startswith('bin/') else n,digest,str(len(data))])
            with (self.site/record).open('w',newline='') as f:csv.writer(f).writerows(rows)
        return dict(package_site=str(self.site),solver_wheel=wheels[0],consumer_wheel=wheels[1],launcher_provenance=self.prov)
    def test_declared_launcher_payload_valid(self):
        s.verify_launcher_bytes(self.raw,self.declaration,self.prov)
        s.verify_launcher_bytes(self.make_launcher(kind='w64'),dict(self.declaration,kind='w64'),self.prov)
    def test_wrong_stub_refuses(self):
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(b'BAD'+self.raw[3:],self.declaration,self.prov)
    def test_wrong_interpreter_refuses(self):
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(self.make_launcher(interpreter='other.exe'),self.declaration,self.prov)
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(self.make_launcher(kind='w64'),self.declaration,self.prov)
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(self.raw,dict(self.declaration,kind='w64'),self.prov)
        self.prov['gui_executable']['sha256']='wrong'
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(self.make_launcher(kind='w64'),dict(self.declaration,kind='w64'),self.prov)
        gui=Path(self.prov['gui_executable']['path']);self.prov['gui_executable']['sha256']=s.sha(gui);gui.unlink()
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(self.make_launcher(kind='w64'),dict(self.declaration,kind='w64'),self.prov)
    def test_changed_payload_refuses(self):
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(self.make_launcher(target='malicious.module:main'),self.declaration,self.prov)
    def test_trailing_executable_data_refuses(self):
        with self.assertRaises(RuntimeError):s.verify_launcher_bytes(self.raw+b'payload',self.declaration,self.prov)
    def test_site_and_declared_record_alias_valid(self):self.assertFalse(s.verify_site(self.fixture())['reinstalled'])
    def test_unrelated_extra_refuses(self):
        p=self.fixture();(self.site/'extra.py').write_text('x')
        with self.assertRaises(RuntimeError):s.verify_site(p)
    def test_changed_product_refuses(self):
        p=self.fixture();(self.site/'solver/product.py').write_text('changed')
        with self.assertRaises(RuntimeError):s.verify_site(p)
    def test_record_digest_refuses(self):
        p=self.fixture();r=self.site/'fixture.dist-info/RECORD';r.write_text(r.read_text().replace('sha256=','wrong='))
        with self.assertRaises(RuntimeError):s.verify_site(p)
    def test_path_traversal_and_wrong_alias_refuse(self):
        launchers={'bin/fixture.exe':self.declaration}
        self.assertEqual(s.record_target('../../bin/fixture.exe','fixture.dist-info/RECORD',launchers),'bin/fixture.exe')
        for path in ('../other','/absolute','C:/drive','bin\\backslash','../../bin/unknown.exe','bin/../fixture.exe'):
            with self.assertRaises(RuntimeError):s.record_target(path,'fixture.dist-info/RECORD',launchers)
        with self.assertRaises(RuntimeError):s.record_target('../../bin/fixture.exe','other.dist-info/RECORD',launchers)
        with self.assertRaises(RuntimeError):s.record_target('bin/fixture.exe','other.dist-info/RECORD',launchers)
    def test_untransferred_slot_refuses(self):
        if not ((s.OUT/'EXECUTABLE_PREPARED.json').exists() and (s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').exists()):
            self.skipTest('local assembly-series authority evidence '
                '(docs/agent_plans/development/execution-v3) is absent; '
                'the untransferred-slot refusal requires the exact '
                'frozen grant bytes')
        from unittest.mock import patch
        frozen=(s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').read_bytes()
        p=json.loads((s.OUT/'EXECUTABLE_PREPARED.json').read_text())
        temporary=self.root/'authority';temporary.mkdir()
        (temporary/'ROOT_DEVELOPMENT_AUTHORITY.json').write_bytes(frozen)
        with patch.object(s,'OUT',temporary):
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
            a=json.loads(frozen);a['slot_status']='NOT_TRANSFERRED'
            live=temporary/s.EXECUTION_AUTHORITY_NAME;live.write_text(json.dumps(a))
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
            a['slot_status']='TRANSFERRED';live.write_text(json.dumps(a))
            self.assertEqual(s.execution_authority(p,'controls')['slot_status'],'TRANSFERRED')
            a['reserve_seconds']=119;live.write_text(json.dumps(a))
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
            a=json.loads(frozen);a['slot_status']='TRANSFERRED';live.write_text(json.dumps(a))
            (temporary/'ROOT_DEVELOPMENT_AUTHORITY.json').write_bytes(frozen+b' ')
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
    def test_exact_twenty_adjudication(self):
        import ast
        from types import SimpleNamespace
        def result(**changes):
            fields=dict(testsRun=20,skipped=[],errors=[],failures=[],wasSuccessful=lambda:True)
            fields.update(changes);return SimpleNamespace(**fields)
        s.adjudicate_controls(result())
        for count in (0,8,19,21):
            with self.assertRaises(RuntimeError):s.adjudicate_controls(result(testsRun=count))
        for changes in (dict(skipped=[('test','reason')]),dict(errors=[('test','error')]),
                        dict(failures=[('test','failure')]),dict(wasSuccessful=lambda:False)):
            with self.assertRaises(RuntimeError):s.adjudicate_controls(result(**changes))
        # Bind the direct result control to the production execution path.
        tree=ast.parse(Path(s.__file__).read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(len(functions),sum(isinstance(n,ast.FunctionDef) for n in tree.body))
        for caller,callee in (('child','_execute_child_body'),('_execute_child_body','adjudicate_controls'),
                              ('run','_supervised_run'),('authority_gate','_contract_authority_gate')):
            self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==callee
                                for n in ast.walk(functions[caller])),(caller,callee))

if __name__=='__main__':unittest.main()


