"""Prepared stdlib controls; not executed under preparation-only authority."""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('assembly_series_v8',ROOT/'scripts/r8_b2_assembly_series_v8_dev.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class Controls(unittest.TestCase):
    def test_geometry_counts(self):
        for size,members,pads,dofs in ((2500,250,25,17136),(5000,500,50,33936),(10000,1000,100,67266)):
            i=s.inventory(size);self.assertEqual((i['shells'],i['members'],i['pads'],i['dofs']),(size,members,pads,dofs))
        self.assertEqual(s.smoke_inventory(),dict(shells=3,members=3,pads=1,rows=20,dofs=72,fixed_dofs=12,loaded_dof=20,selected_dofs=[20,18],pad='pad-0-1',member='b2-0-0'))
        self.assertEqual(s.SMOKE_CONFIG,dict(nx=3,ny=1,line_rows=[0],patch_x=[1],cases=1))
    def test_deterministic_load_support(self):
        self.assertEqual(s.inventory(5000)['loaded_dof'],15752);self.assertEqual(s.inventory(5000)['fixed_dofs'],306)
        self.assertEqual(s.CONFIGS[10000]['line_rows'],list(range(4,100,10)))
    @unittest.skipUnless(s.FULL.exists(),
        'local assembly-series full fixture evidence '
        '(docs/agent_plans/development/r8-b2-installed-batch-persistence-20261010) is absent')
    def test_frozen_full_identity(self):
        import ast,json
        self.assertEqual(s.sha(s.FULL),s.FULL_SHA)
        tree=ast.parse(Path(s.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='installed_samples')
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save_case']
        self.assertEqual(len(calls),1);path=calls[0].args[1]
        self.assertIsInstance(path,ast.BinOp);self.assertIsInstance(path.right,ast.Constant)
        self.assertEqual(path.right.value,'definition.anyfem-b2.json')
        self.assertEqual(s.PHASES[:5],('controls','install','validate-install','public-smoke','low'))
        self.assertEqual(s.ENV['PYTHONDONTWRITEBYTECODE'],'1')
        child=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_execute_child_body')
        literals={n.value for n in ast.walk(child) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertTrue({'--no-index','--no-deps','--no-compile','--no-cache-dir','--target'}<=literals)
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='exists' for n in ast.walk(child)))
        self.assertTrue(any(isinstance(n,ast.Attribute) and n.attr=='_private_q4_mesh' for n in ast.walk(fn)))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='collect' for n in ast.walk(fn)))

        sites=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='installed_graph']
        self.assertEqual(len(sites),2)
        self.assertTrue(any(isinstance(n,ast.Constant) and n.value=='installed-post-close.json' for n in ast.walk(fn)))
        plan=json.loads((s.OUT/'EXECUTABLE_PREPARED.json').read_text())
        self.assertIn('owner.solve_linear_many',s.public_caller_contract(plan)['calls'])
        source=Path(s.__file__).read_text()
        for broken in (source.replace('owner.solve_linear_many(', 'owner.solve_load_cases('),
                       source.replace('selected_dofs=tuple(i[', 'unknown_keyword=tuple(i[')):
            with self.assertRaises(RuntimeError):s.public_caller_contract(plan,broken)
        final=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_execute_child_body')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='installed_graph' for node in final.body if isinstance(node,ast.Try) for f in node.finalbody for n in ast.walk(f)))
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as name:
            attempt=Path(name);(attempt/'binding.json').write_text(json.dumps(dict(preparation=plan)))
            with patch.object(s,'basic_runtime',return_value={}),patch.object(s,'public_runtime',return_value={}),patch.object(s,'module',return_value=None),patch.object(s,'installed_samples',side_effect=RuntimeError('primary marker')),patch.object(s,'installed_graph',side_effect=RuntimeError('closure marker')) as closure:
                self.assertFalse(s._execute_child_body('public-smoke',attempt));self.assertEqual(closure.call_count,1)
            outcome=json.loads((attempt/'result.json').read_text())
            self.assertEqual(outcome['error'],'primary marker');self.assertEqual(outcome['closure_after_error'],'closure marker')
            self.assertIn('primary marker',(attempt/'failure.txt').read_text())
            self.assertTrue((attempt/'closure-after-failure.txt').is_file())


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
        if not ((s.BASE/'PREPARED.json').exists() and (s.OUT/'EXECUTABLE_PREPARED.json').exists() and (s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').exists()):
            self.skipTest('local assembly-series authority evidence '
                '(docs/agent_plans/development/execution-v8) is absent; '
                'tampered-authority refusals require the exact '
                'frozen grant bytes')
        import json,copy
        a=json.loads((s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').read_text());p=json.loads((s.OUT/'EXECUTABLE_PREPARED.json').read_text())
        p['predecessor_prepared_sha256']=s.sha(s.BASE/'PREPARED.json')
        # Relocated evidence is not an execution grant for this worktree.
        # Exercise the positive and tamper controls in a disposable coherent
        # unit namespace; never rewrite the original frozen authority.
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            temporary=Path(directory)
            a['package_site']=str(temporary/'package/site')
            frozen=json.dumps(a,sort_keys=True).encode()
            (temporary/'ROOT_DEVELOPMENT_AUTHORITY.json').write_bytes(frozen)
            p.update(package_site=a['package_site'],
                frozen_grant_sha256=hashlib.sha256(frozen).hexdigest(),
                authority_contract_sha256=s.authority_contract_sha(a))
            a['slot_status']='TRANSFERRED'
            with patch.object(s,'OUT',temporary):
                s.authority_gate(a,p,'low')
                for key,value in [('tree_rss_limit_bytes',4*1024**3),('input_prepared_sha256','wrong'),('jit_disabled',False)]:
                    changed=copy.deepcopy(a);changed[key]=value
                    with self.assertRaises(RuntimeError):s.authority_gate(changed,p,'low')
    def test_ratio_and_five_samples(self):
        groups={size:[dict(physical_pass=True,status='PASS',owner_close=dict(private_mesh_released=True,private_elements_remaining=0,factor_handle_released=True),assembly_seconds=n,selected_prefactor_bytes=n*100) for _ in range(5)] for size,n in ((2500,1.),(5000,2.),(10000,4.))}
        self.assertTrue(s.summarize(groups)['screen_pass']);groups[10000][0]['physical_pass']=False
        with self.assertRaises(RuntimeError):s.summarize(groups)
    def test_empty_failed_samples_refuse(self):
        for g in ({},{size:[] for size in s.CONFIGS}):
            with self.assertRaises(RuntimeError):s.summarize(g)
        failed_close={size:[dict(physical_pass=True,status='PASS',owner_close=dict(private_mesh_released=False,private_elements_remaining=2,factor_handle_released=True),assembly_seconds=1.,selected_prefactor_bytes=100) for _ in range(5)] for size in s.CONFIGS}
        with self.assertRaises(RuntimeError):s.summarize(failed_close)
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
        (self.site/'extra.py').unlink();cache=self.site/'fixture/__pycache__/product.cpython-313.pyc';cache.parent.mkdir();cache.write_bytes(b'cache')
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
                '(docs/agent_plans/development/execution-v8) is absent; '
                'the untransferred-slot refusal requires the exact '
                'frozen grant bytes')
        from unittest.mock import patch
        frozen=(s.OUT/'ROOT_DEVELOPMENT_AUTHORITY.json').read_bytes()
        p=json.loads((s.OUT/'EXECUTABLE_PREPARED.json').read_text())
        temporary=self.root/'authority';temporary.mkdir()
        # Rebind a coherent independent namespace; production freshness remains strict.
        grant=json.loads(frozen);grant['package_site']=str(temporary/'package/site')
        frozen=json.dumps(grant,sort_keys=True).encode()
        p.update(package_site=grant['package_site'],frozen_grant_sha256=__import__('hashlib').sha256(frozen).hexdigest(),
            authority_sha256=__import__('hashlib').sha256(frozen).hexdigest(),authority_contract_sha256=s.authority_contract_sha(grant))
        (temporary/'ROOT_DEVELOPMENT_AUTHORITY.json').write_bytes(frozen)
        (temporary/'EXECUTABLE_PREPARED.json').write_text(json.dumps(p))
        with patch.object(s,'OUT',temporary):
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
            a=json.loads(frozen);a['slot_status']='NOT_TRANSFERRED';a['executable_prepared_sha256']=s.sha(s.OUT/'EXECUTABLE_PREPARED.json')
            live=temporary/s.EXECUTION_AUTHORITY_NAME;live.write_text(json.dumps(a))
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
            a['slot_status']='TRANSFERRED';live.write_text(json.dumps(a))
            self.assertEqual(s.execution_authority(p,'controls')['slot_status'],'TRANSFERRED')
            a['reserve_seconds']=119;live.write_text(json.dumps(a))
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
            a=json.loads(frozen);a['slot_status']='TRANSFERRED';a['executable_prepared_sha256']=s.sha(s.OUT/'EXECUTABLE_PREPARED.json');live.write_text(json.dumps(a))
            (temporary/'ROOT_DEVELOPMENT_AUTHORITY.json').write_bytes(frozen+b' ')
            with self.assertRaises(RuntimeError):s.execution_authority(p,'controls')
    def test_exact_twenty_adjudication(self):
        import ast
        from types import SimpleNamespace
        def result(**changes):
            fields=dict(testsRun=20,skipped=[],errors=[],failures=[],wasSuccessful=lambda:True)
            fields.update(changes);return SimpleNamespace(**fields)
        s.adjudicate_controls(result())
        from unittest.mock import patch
        with patch.object(s.sys,'dont_write_bytecode',False):
            with self.assertRaises(RuntimeError):s.bytecode_guard()
        self.assertTrue(s.bytecode_guard()['dont_write_bytecode'])
        with patch.dict(s.os.environ,{'PYTHONDONTWRITEBYTECODE':'0'}):
            with self.assertRaises(RuntimeError):s.bytecode_guard()
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


