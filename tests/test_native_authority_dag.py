"""Differential contract checks against the unchanged boundary predicate."""
from importlib.util import find_spec
import types
import pytest
from anyfesolver.boundary import _callable_authority, _callable_authority_matches
from anyfesolver._native_authority_dag import intern_facts, matches
from anyfesolver import prepared_static_loads as loads


def fact(f):
    return intern_facts(((f,_callable_authority(f),()),))[0][1]


def compare(f,captured):
    assert matches(f,captured,{}) == _callable_authority_matches(f,captured)


@pytest.mark.parametrize('change',['code','defaults','keywords','closure','empty'])
def test_function_mutations(change):
    value=object()
    def f(x=value,*,y=value):return value
    captured=fact(f)
    compare(f,captured);assert matches(f,captured,{})
    if change=='code':
        def other(x=value,*,y=value):return (value,)
        f.__code__=other.__code__
    elif change=='defaults':f.__defaults__=(object(),)
    elif change=='keywords':f.__kwdefaults__={'y':object()}
    elif change=='closure':f.__closure__[0].cell_contents=object()
    else:del f.__closure__[0].cell_contents
    compare(f,captured);assert not matches(f,captured,{})


def test_each_guard_reads_live_facts_and_preserves_mismatched_expected_nodes():
    def f(x=object()):return x
    first=fact(f);original=f.__defaults__;f.__defaults__=(object(),);second=fact(f)
    normalized=intern_facts(((f,first,()),(f,second,())))
    assert normalized[0][1] is not normalized[1][1]
    memo={};assert not matches(f,normalized[0][1],memo);assert matches(f,normalized[1][1],memo)
    f.__defaults__=original
    assert matches(f,first,{}) and not matches(f,second,{})


def test_duplicate_expected_nodes_are_shared_without_arbitrary_equality():
    class Opaque:
        def __eq__(self,other):raise AssertionError('captured value equality')
        def __hash__(self):raise AssertionError('captured value hashing')
    obj=Opaque()
    def leaf(x=obj):return obj
    def root():return leaf
    rows=intern_facts(((root,_callable_authority(root),()),(root,_callable_authority(root),())))
    assert rows[0][1] is rows[1][1]
    assert matches(root,rows[0][1],{})


def test_cycles_and_descriptor_predicates():
    def cyclic():return cyclic
    captured=fact(cyclic)
    assert captured[4][0][1] == (cyclic,)
    assert captured[4][0][1] is not captured
    compare(cyclic,captured)
    for wrapper in (property(cyclic),staticmethod(cyclic),classmethod(cyclic)):
        compare(wrapper,fact(wrapper))
    assert not matches(cyclic,(lambda:None,),{})


def test_generation_counter_exception_matches_literal_type_rule():
    accepted_generation=1
    def require():return accepted_generation
    require=types.FunctionType(require.__code__,{'__name__':'anyfesolver._qualified_authority_epoch'},closure=require.__closure__)
    require.__qualname__='_AuthorityEpochManager.bind.<locals>.require'
    captured=fact(require)
    cell=require.__closure__[0]
    for value,expected in ((2,True),(True,False),(2.,False),(None,False)):
        cell.cell_contents=value;compare(require,captured);assert matches(require,captured,{}) is expected


@pytest.mark.parametrize('change',['alias','code','globals'])
def test_optimized_matcher_is_validated_before_use(monkeypatch,change):
    import anyfesolver._native_authority_dag as dag
    if change=='alias':monkeypatch.setattr(loads,'_native_authority_matches',lambda *a:True)
    elif change=='code':
        def forged(member,authority,memo):return True
        monkeypatch.setattr(dag.matches,'__code__',forged.__code__)
    else:monkeypatch.setattr(dag,'matches',lambda *a:True)
    with pytest.raises(loads.PreparedStaticCapabilityError):loads.require_native_source_authority()


def test_class_and_descriptor_mutation_is_still_checked(monkeypatch):
    from anyfesolver.elements import BeamElement
    loads.require_native_source_authority()
    monkeypatch.setattr(BeamElement,'native_dag_probe',property(lambda self:0),raising=False)
    with pytest.raises(loads.PreparedStaticCapabilityError,match='class executable'):loads.require_native_source_authority()


def test_plan_replacement_refused_by_existing_response_binding(monkeypatch):
    from anyfesolver.prepared_static_response import PreparedStaticResponseEvaluator,_require_response_authority
    owner=object.__new__(PreparedStaticResponseEvaluator)
    owner._numerical_affinity_fallback=owner._native_affinity_mode_binding=False
    owner._native_affinity_binding=None
    _require_response_authority(owner)
    monkeypatch.setattr(loads,'_NATIVE_HELPER_FACTS',tuple(list(loads._NATIVE_HELPER_FACTS)))
    with pytest.raises(loads.PreparedStaticCapabilityError):_require_response_authority(owner)


def test_globals_still_checked_each_guard(monkeypatch):
    function,authority,bindings=next(row for row in loads._NATIVE_HELPER_FACTS[1:] if row[2])
    name,value=next((name,value) for name,value in bindings if name!='__name__')
    monkeypatch.setitem(function.__globals__,name,object())
    with pytest.raises(loads.PreparedStaticCapabilityError):loads.require_native_source_authority()


@pytest.mark.parametrize('name',['id','type','all','len','zip','any'])
def test_absent_builtin_shadow_refused_before_memo_use(monkeypatch,name):
    import anyfesolver._native_authority_dag as dag
    function=next(row[0] for row in loads._NATIVE_HELPER_FACTS[1:] if not row[0].__closure__)
    def forged(*args,**kwargs):return True
    monkeypatch.setattr(dag,name,lambda *args:0,raising=False)
    monkeypatch.setattr(function,'__code__',forged.__code__)
    with pytest.raises(loads.PreparedStaticCapabilityError,match='helper executable'):
        loads.require_native_source_authority()


@pytest.mark.parametrize('name',['id','type','all','len','zip','any'])
def test_absent_builtin_shadow_alone_rejected_with_unchanged_function(monkeypatch,name):
    import anyfesolver._native_authority_dag as dag
    function,authority,_=loads._NATIVE_HELPER_FACTS[0]
    monkeypatch.setattr(dag,name,lambda *args:0,raising=False)
    assert _callable_authority_matches(function,authority)
    with pytest.raises(loads.PreparedStaticCapabilityError,match='native helper executable authority changed'):
        loads.require_native_source_authority()


def test_memo_retains_actual_and_expected_nodes():
    def f():return None
    captured=fact(f);memo={};assert matches(f,captured,memo)
    entry=memo[(id(f),id(captured))]
    assert entry[0] is f and entry[1] is captured and entry[2] is True


@pytest.mark.parametrize('restore_before_return',[False,True])
def test_callback_builtin_shadow_observed_at_original_guard(tmp_path,monkeypatch,restore_before_return):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    import anypost
    import anyfesolver._native_authority_dag as dag
    from anyfesolver import PatternCase,export_recovery_capsule_to_sink
    from test_native_affinity_fallback import prepare
    function=next(row[0] for row in loads._NATIVE_HELPER_FACTS[1:] if not row[0].__closure__)
    original_code=function.__code__
    def forged(*args,**kwargs):return True
    path=tmp_path/'callback.h5';path.write_bytes(b'previous')
    native=anypost.ArchiveSink(path)
    with prepare(numerical_affinity_fallback=True) as evaluator,monkeypatch.context() as patch:
        class Sink:
            capsule_begin=native.capsule_begin;capsule_stored=native.capsule_stored
            capsule_end=native.capsule_end;capsule_failed=native.capsule_failed
            def capsule_case(self,*args):
                native.capsule_case(*args)
                patch.setattr(dag,'id',lambda value:0,raising=False)
                patch.setattr(function,'__code__',forged.__code__)
                if restore_before_return:
                    patch.undo()
        if restore_before_return:
            with native:
                export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=Sink(),model_identity='m',mesh_identity='s')
        else:
            with pytest.raises(loads.PreparedStaticCapabilityError) as caught,native:
                export_recovery_capsule_to_sink(evaluator,[PatternCase('a',{})],sink=Sink(),model_identity='m',mesh_identity='s')
            assert str(caught.value)=='source response executable authority changed'
            assert '_guard_runtime' in [entry.name for entry in caught.traceback]
            assert '_require_response_authority' in [entry.name for entry in caught.traceback]
            assert path.read_bytes()==b'previous'
    if restore_before_return:
        with anypost.open_results(path) as reader:assert reader.cases()[0]['certified'] is True
    loads.require_native_source_authority()


def _identity_fact(a):
    if not a:return ()
    if len(a)==1:
        return ('wrapped',_identity_fact(a[0])) if isinstance(a[0],tuple) else ('leaf',id(a[0]))
    if len(a)==5:
        f,code,defaults,keywords,cells=a
        return ('function',id(f),id(code),id(defaults),None if keywords is None else tuple((id(k),id(v)) for k,v in keywords),
            tuple(('counter',g) if g else (id(v),_identity_fact(c),g) for v,c,g in cells))
    return tuple(_identity_fact(child) for child in a)


def test_real_native_facts_identity_structural_differential():
    for function,authority,_ in loads._NATIVE_HELPER_FACTS[1:]:
        assert _identity_fact(authority)==_identity_fact(_callable_authority(function)),function.__qualname__
