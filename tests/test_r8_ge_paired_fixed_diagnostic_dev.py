"""Focused stdlib contracts and one separately selected installed public call."""
import ast,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from scripts import r8_ge_paired_fixed_diagnostic_dev as d
SOURCE=pytest.mark.skipif(os.environ.get('GE_FIXED_DIAGNOSTIC_STAGE')!='source-controls',reason='supervised source stage required')
INSTALLED=pytest.mark.skipif(os.environ.get('GE_FIXED_DIAGNOSTIC_STAGE')!='installed-fixed',reason='sole installed stage required')

@SOURCE
def test_source_inner_selection_and_corruption():
    raw=d.OUTER.read_bytes();inner,outer,fixed=d.select_inner(raw)
    assert inner==d.INNER.read_bytes() and fixed['generation']==11 and outer['dof']==6
    with pytest.raises(ValueError,match='checksum'):d.select_inner(raw[:-1]+bytes([raw[-1]^1]))
    h,body=d.unwrap(inner,b'ANYSSPD1');h['generation']=10
    header=json.dumps(h,sort_keys=True,separators=(',',':')).encode();wrong=b'ANYSSPD1'+len(header).to_bytes(4,'little')+header+body
    wrong+=hashlib.sha256(wrong).digest()
    oh,_=d.unwrap(raw,b'ANYSSDCP1');header=json.dumps(oh,sort_keys=True,separators=(',',':')).encode()
    wrapper=b'ANYSSDCP1'+len(header).to_bytes(4,'little')+header+wrong;wrapper+=hashlib.sha256(wrapper).digest()
    with pytest.raises(ValueError,match='gen11'):d.select_inner(wrapper)

@SOURCE
def test_source_outer_cannot_be_fixed_entry():
    with pytest.raises(ValueError,match='pending adaptive'):d.require_fixed_entry(d.OUTER.read_bytes())
    h,_=d.require_fixed_entry(d.INNER.read_bytes());assert h['control_dof']==6 and h['seed_sha256']==d.SEED
    body=(d.ROOT/'src/anyfesolver/_stiffened_shell_displacement_control.py').read_text()
    assert 'pending adaptive displacement controls must match' in body

@SOURCE
def test_source_sparse_residual_exact_and_corrupt():
    row=dict(bordered=dict(shape=[2,2],row=[0,1],col=[0,1],data=[2.,4.]),solution=[3.,5.],rhs=[6.,20.])
    assert d.linear_residual(row)['absolute_norm']==0.
    row['solution']=[3.,6.];assert d.linear_residual(row)['residual']==[0.,4.]
    row.update(rhs=[0.,0.]);assert d.linear_residual(row)['relative_norm'] is None

@SOURCE
def test_source_predicate_and_unavailable_markers():
    assert d.predicate(1.,2.)['strict_decrease'] is True
    assert d.predicate(2.,2.)['strict_decrease'] is False
    assert d.predicate(3.,2.)['strict_decrease'] is False
    fields=d.selected_fields({'actual':0},('actual','missing'))
    assert fields==dict(values={'actual':0},unavailable=['missing'])

@SOURCE
def test_source_observer_restores_hooks_on_failure(tmp_path):
    before=(sys.getprofile(),sys.gettrace());assert before==(None,None)
    with pytest.raises(RuntimeError,match='fixture failure'):
        with d.Trace(tmp_path):raise RuntimeError('fixture failure')
    assert (sys.getprofile(),sys.gettrace())==before
    hook=lambda *a:None
    sys.setprofile(hook)
    try:
        with pytest.raises(ValueError,match='existing tracing'):d.Trace(tmp_path).__enter__()
        assert sys.getprofile() is hook
    finally:sys.setprofile(before[0]);sys.settrace(before[1])

class Atomic:
    def capture(self,owner):return owner.generation
    def atomic_flags(self,owner,before):return {'flag'+str(i):owner.generation==before for i in range(21)}

@SOURCE
def test_source_public_caller_terminal_capture(tmp_path):
    calls=[];namespace={'calls':calls}
    exec(compile("def solve(owner,actions,targets,control):\n calls.append((actions,targets,control))\n raise RuntimeError('fixed refusal')\n",
                 'fixture/_stiffened_shell_displacement_control.py','exec'),namespace)
    class Owner:
        generation=11
        _generation=11
        def solve(self,actions,targets,**kw):
            calls.append(kw);return namespace['solve'](self,actions,targets,kw['displacement_control'])
        def checkpoint(self):return b'unchanged fixed bytes'
        def recover(self):return {'generation':11}
    owner=Owner();atomic=Atomic();trace=d.Trace(tmp_path,atomic)
    with pytest.raises(RuntimeError,match='fixed refusal'):
        d.invoke_once(owner,('declared',),'control',trace,tmp_path,atomic)
    assert calls[0]==dict(displacement_control='control',max_iterations=20,max_backtracks=8)
    assert calls[1]==(('declared',),(d.TARGET,),'control')
    report=json.loads((tmp_path/'outcome.json').read_text())
    assert report['status']=='FAILED' and all(report['fixed_flags'].values()) and len(report['fixed_flags'])==21
    assert (tmp_path/'terminal.bin').read_bytes()==b'unchanged fixed bytes'
    assert report['observed_evaluations']==0 and report['observed_linear_solves']==0
    class RecognizedFixedError(Exception):pass
    d.require_expected_refusal(RecognizedFixedError(),RecognizedFixedError,report['fixed_flags'])
    with pytest.raises(RuntimeError,match='programming'):
        d.require_expected_refusal(RuntimeError('programming'),RecognizedFixedError,report['fixed_flags'])
    with pytest.raises(ValueError,match='all21'):
        d.require_expected_refusal(RecognizedFixedError(),RecognizedFixedError,{'partial':True})

@SOURCE
def test_source_binding_inventory_and_adoption():
    assert d.PLAN['entry']=='CANONICAL_INNER_FIXED_SNAPSHOT' and d.PLAN['adaptive'] is False
    assert len(d.SOURCE_NAMES)==8 and len(set(d.SOURCE_NAMES))==8
    inventory={n.name for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    assert set(d.SOURCE_NAMES)<=inventory and 'test_installed_one_fixed_diagnostic' not in d.SOURCE_NAMES
    bindings=d.bindings();assert len(bindings)==len({Path(p).name for p in bindings})
    assert d.sha(d.OUTER)==json.loads((d.DOC/'READINESS.json').read_text())['outer_sha256']
    d.require_adoption(dict(status='ADOPTED',plan=d.PLAN))
    with pytest.raises(ValueError,match='adoption'):
        d.require_adoption(dict(status='ADOPTED',plan={**d.PLAN,'max_backtracks':9}))
    with pytest.raises(ValueError,match='adoption'):
        d.require_adoption(dict(status='PROPOSED',plan=d.PLAN))

@SOURCE
def test_source_trace_does_not_call_mechanics(tmp_path):
    code=ast.parse(Path(d.__file__).read_text());trace=next(n for n in code.body if isinstance(n,ast.ClassDef) and n.name=='Trace')
    names={n.func.attr for n in ast.walk(trace) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not names & {'solve','correct','_advance','_assemble','evaluate','spsolve','restore'}
    sequence=[];namespace={'sequence':sequence}
    exec(compile("def merit(trial,evaluation):\n sequence.append('merit')\n return 2.\ndef correct():\n state=object()\n evaluation=object()\n old=merit(state,evaluation)\n candidate=object()\n changed=object()\n amount=.5\n backtrack=1\n iteration=0\n value=.1\n return merit(candidate,changed)<old\n",'fixture/_stiffened_shell_displacement_corrector.py','exec'),namespace)
    baseline=namespace['correct']();expected=list(sequence);sequence.clear()
    with d.Trace(tmp_path):observed=namespace['correct']()
    assert observed==baseline and sequence==expected==['merit','merit']
    records=[json.loads(x) for x in (tmp_path/'trace.jsonl').read_text().splitlines()]
    assert records[-1]['decision_predicate']['strict_decrease'] is False and records[-1]['amount']==.5

@INSTALLED
def test_installed_one_fixed_diagnostic():
    d.connected(Path(os.environ['GE_FIXED_DIAGNOSTIC_OUTPUT']))
