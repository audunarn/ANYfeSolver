"""Thin connected stages using the existing precise development supervisor."""
from pathlib import Path
from hashlib import sha256
import argparse,ast,json,os,sys,subprocess,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'docs/agent_plans/development/r8-ge-paired-adaptive-20261010/execution-v4/fixed-correction-diagnostic-v1/saved-merit-remainder-v1/interpolation-candidate-v1'
DOC=BASE/'execution-v1'
PACKAGE=DOC/'package'

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def tree():
    current={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'src/anyfesolver').rglob('*.py')}
    if current!=json.loads((DOC/'CURRENT_SOURCE_TREE.json').read_text()):raise ValueError('current source drift')
    return current
def bindings():
    from scripts.r8_ge_coupled_cycle_dev import IDENTITIES
    paths=[Path(__file__),DOC/'CURRENT_SOURCE_TREE.json',DOC/'APPLICATION.json',BASE/'READINESS.json',
           ROOT/'tests/test_r8_ge_merit_interpolation_candidate_dev.py',ROOT/'tests/test_r8_ge_paired_adaptive_dev.py',
           ROOT/'tests/test_r8_ge_merit_history_entrypoint_dev.py',ROOT/'tests/test_r8_f06_fibre_launch_seam_dev.py',
           ROOT/'tests/conftest.py',ROOT/'scripts/r8_f06_fibre_log_domain_dev.py',
           ROOT/'scripts/r8_f06_log_domain_check_dev.py',ROOT/'scripts/r8_ge_coupled_cycle_dev.py',
           ROOT/'scripts/r8_ge_paired_adaptive_dev.py',*IDENTITIES]
    paths+=list((ROOT/'docs/agent_plans/development/r8-ge-paired-adaptive-20261010/legacy-golden-v1').glob('*'))
    paths+=[ROOT/name for name in ('pyproject.toml','setup.py','setup.cfg','README.md','LICENSE','COPYRIGHT','THIRD_PARTY_NOTICES.md','MANIFEST.in') if (ROOT/name).exists()]
    pending=[p for p in paths if p.parent==ROOT/'scripts'];seen=set(paths)
    while pending:
        for node in ast.walk(ast.parse(pending.pop().read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                modules=[node.module] if node.module!='scripts' else ['scripts.'+x.name for x in node.names]
            for module in modules:
                p=ROOT/(module.replace('.','/')+'.py')
                if module.startswith('scripts.') and p.exists() and p not in seen:
                    seen.add(p);paths.append(p);pending.append(p)
    if PACKAGE.exists():
        paths += [p for p in (PACKAGE/'artifact.json',PACKAGE/'wheel-built.json',PACKAGE/'snapshot-bindings.json',PACKAGE/'metadata-bindings.json') if p.exists()]
        paths += list((PACKAGE/'wheels').glob('*.whl'))
    if len(set(p.name for p in paths))!=len(paths):raise ValueError('snapshot basename collision')
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def artifact():
    source=tree();record=json.loads((PACKAGE/'artifact.json').read_text())
    wheel=Path(record['wheel']);assert sha(wheel)==record['sha256']
    assert json.loads((PACKAGE/'snapshot-bindings.json').read_text())==source
    with zipfile.ZipFile(wheel) as z:
        for rel,h in source.items():
            entry=Path(rel).relative_to('src').as_posix()
            assert sha(PACKAGE/'snapshot'/rel)==sha(PACKAGE/'site'/entry)==sha256(z.read(entry)).hexdigest()==h,rel
    return record
def _history_slot():
    authority=json.loads((DOC/'INSTALLED_HISTORY_AUTHORITY.json').read_text())
    ledger=json.loads((DOC/'ledger.json').read_text())
    if (authority.get('status')!='AUTHORIZED' or authority.get('budget_s')!=ledger.get('budget_s')
            or authority.get('reserve_s')!=ledger.get('reserve_s') or ledger.get('status')!='OPEN'):
        raise ValueError('authorized open same-ledger installed-history slot required')
    return authority

def installed_history(out):
    """Current-candidate original material-history entrypoint; separately guarded.

    Owns the current-candidate source/artifact/authority/slot checks, then reuses
    the paired adaptive mechanical/lifecycle program with the public quadratic
    merit globalization control; the historical connected stage and its exact
    guards are untouched.
    """
    if (os.environ.get('GE_MERIT_HISTORY_STAGE')!='installed-history' or
            os.environ.get('GE_MERIT_HISTORY_OUTPUT')!=str(out)):
        raise ValueError('sole installed-history child and exact output required')
    tree();artifact();_history_slot()
    from anyfesolver._stiffened_shell_displacement_corrector import PAIRED_FIXED_WORK_QUADRATIC_V1
    from scripts import r8_ge_paired_adaptive_dev as adaptive
    adaptive._connected_program(out,control_kwargs=dict(merit_globalization=PAIRED_FIXED_WORK_QUADRATIC_V1))
def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0])
    assert json.loads((out/'bindings.json').read_text())==bindings()
    source=tree()
    if stage=='wheel':
        snapshot=PACKAGE/'snapshot';snapshot.mkdir(parents=True);wheels=PACKAGE/'wheels';wheels.mkdir()
        for rel,h in source.items():
            dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes());assert sha(dest)==h
        metadata={}
        for name in ('pyproject.toml','setup.py','setup.cfg','README.md','LICENSE','COPYRIGHT','THIRD_PARTY_NOTICES.md','MANIFEST.in'):
            p=ROOT/name
            if p.exists():shutil.copyfile(p,snapshot/name);metadata[name]=sha(p)
        write(PACKAGE/'snapshot-bindings.json',source);write(PACKAGE/'metadata-bindings.json',metadata)
        cmd=[sys.executable,'-m','pip','wheel',str(snapshot),'--no-index','--no-deps','--no-build-isolation','--no-cache-dir','-w',str(wheels)]
        write(out/'build-command.json',cmd);subprocess.run(cmd,check=True)
        found=list(wheels.glob('*.whl'));assert len(found)==1
        write(PACKAGE/'wheel-built.json',dict(wheel=str(found[0]),sha256=sha(found[0])))
        return 0
    if stage=='install':
        wheels=list((PACKAGE/'wheels').glob('*.whl'));assert len(wheels)==1
        built=json.loads((PACKAGE/'wheel-built.json').read_text());assert sha(wheels[0])==built['sha256']
        site=PACKAGE/'site';site.mkdir()
        cmd=[sys.executable,'-m','pip','install',str(wheels[0]),'--no-index','--no-deps','--no-compile','--no-cache-dir','--target',str(site)]
        write(out/'install-command.json',cmd);subprocess.run(cmd,check=True)
        write(PACKAGE/'artifact.json',dict(wheel=str(wheels[0]),sha256=sha(wheels[0]),site=str(site),files=len(source)))
        write(out/'artifact-verified.json',artifact());return 0
    if stage=='mock-seam':
        import pytest
        code=int(pytest.main([str(ROOT/'tests/test_r8_f06_fibre_launch_seam_dev.py'),'-q','-o','pythonpath=',
            '-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(Path(os.environ['TEMP'])/'pytest')]))
        if code==0:
            from scripts.r8_ge_coupled_cycle_dev import counts
            write(out/'counts.json',counts(out/'junit.xml',4))
        return code
    if stage=='installed-history':
        write(out/'artifact-before.json',artifact())
        sys.path[:0]=[str(PACKAGE/'site'),str(ROOT/'tests')]
        os.environ.update(GE_MERIT_HISTORY_OUTPUT=str(out),GE_MERIT_HISTORY_STAGE=stage)
        from scripts.r8_f06_log_domain_check_dev import runtime
        runtime(out,'runtime-before',PACKAGE/'site')
        try:
            installed_history(out);return 0
        finally:
            rt=runtime(out,'runtime-after',PACKAGE/'site')
            for name,path in rt['imports'].items():assert sha(path)==source['src/'+Path(path).relative_to(PACKAGE/'site').as_posix()],name
            write(out/'artifact-after.json',artifact())
    selected=ROOT/'src' if stage in ('source-controls','history-entrypoint-controls') else PACKAGE/'site'
    if stage=='installed-controls':write(out/'artifact-before.json',artifact())
    sys.path[:0]=[str(selected),str(ROOT/'tests')]
    from scripts.r8_f06_log_domain_check_dev import runtime
    runtime(out,'runtime-before',selected)
    import pytest
    nodes=([str(ROOT/n.split('::')[0])+'::'+n.split('::')[1] for n in
            json.loads((BASE/'READINESS.json').read_text())['source_controls']]
           if stage=='source-controls' else [str(ROOT/'tests/test_r8_ge_merit_history_entrypoint_dev.py')])
    try:
        code=int(pytest.main(nodes+[
            '-q','-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
        from scripts.r8_ge_coupled_cycle_dev import counts
        if code==0:write(out/'counts.json',counts(out/'junit.xml',12 if stage=='source-controls' else 4))
        return code
    finally:
        rt=runtime(out,'runtime-after',selected)
        for name,path in rt['imports'].items():assert sha(path)==source['src/'+Path(path).relative_to(selected).as_posix()],name
        if stage=='installed-controls':write(out/'artifact-after.json',artifact())
def launch(stage):
    tree()
    from scripts.r8_f06_fibre_log_domain_dev import run
    for prerequisite in {'source-controls':(),'mock-seam':(),'history-entrypoint-controls':(),
                         'wheel':('source-controls',),'install':('source-controls','wheel'),
                         'installed-controls':('source-controls','wheel','install'),
                         'installed-history':('source-controls','wheel','install','installed-controls',
                                              'history-entrypoint-controls')}[stage]:
        attempts=json.loads((DOC/'ledger.json').read_text())['attempts']
        last=[a for a in attempts if a['stage']==prerequisite][-1]
        assert last['status']=='PASS'
    return run(stage,doc=DOC,entry=__file__,bind=bindings,authority_file=stage.upper().replace('-','_')+'_AUTHORITY.json',
        temp_root=ROOT/'.development/ge-mi-temp')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',required=True,
        choices=('source-controls','mock-seam','history-entrypoint-controls','wheel','install',
                 'installed-controls','installed-history'));a=p.parse_args()
    raise SystemExit(child(Path(a.out),a.stage) if a.child else launch(a.stage))
