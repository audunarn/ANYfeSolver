"""Exact literal differential and optional native dispatch boundaries."""
import gc
import importlib
import sys
import types
import pytest
from anyfesolver import boundary,static_batch,prepared_static_loads as loads
from anyfesolver import _native_authority_dag as dag,_native_guard_dispatch as dispatch

@pytest.fixture
def native():
    return pytest.importorskip('anyfesolver._native_guard')

def plan(native,cls,functions=()):
    classes={cls:static_batch._class_state(cls)}
    helpers=tuple((f,boundary._callable_authority(f),()) for f in functions)
    roots=(static_batch._class_state,static_batch._callable_state,boundary._callable_authority_matches,dag.matches)
    value=native.make_plan(classes,helpers,boundary._MISSING,roots)
    assert value is not NotImplemented
    return value,classes,helpers

def literal(classes,helpers):
    return all(static_batch._class_state(c)==expected for c,expected in classes.items()) and all(
        boundary._callable_authority_matches(f,a) for f,a,_ in helpers)

def test_actual_all_original_facts(native):
    value=native.make_plan(loads._NATIVE_CLASS_FACTS,loads._NATIVE_HELPER_FACTS,dag._MISSING,
        (static_batch._class_state,static_batch._callable_state,boundary._callable_authority_matches,dag.matches))
    assert value is not NotImplemented
    before=native.statistics()
    assert native.compare(value,loads._NATIVE_CLASS_FACTS,loads._NATIVE_HELPER_FACTS) is True
    after=native.statistics()
    assert after['class_facts']-before['class_facts']==18
    assert after['helper_facts']-before['helper_facts']==len(loads._NATIVE_HELPER_FACTS)
    _assert_helper_facts_closed()

def _assert_helper_facts_closed():
    """The captured helper set is closed under the reachability rule that builds it.

    The exact fact count is not a constant: it depends on the Python version (3.12+
    inlines comprehensions, changing co_names) and on whether numba wraps the
    plasticity kernels (a numba dispatcher is not a FunctionType, so it is not
    captured). Measured on this tree: 3.11 with numba 1300 facts, 3.13/3.14 with
    numba 1302, 3.11 without numba 1307. Instead of pinning a number, check
    independently that every function reachable from a captured fact through a
    referenced global, a closure cell or a member of a root class is itself captured
    exactly once, that nothing outside the anyfesolver/anymaterial modules is, and
    that the five boundary helpers and the matcher are present."""
    facts=loads._NATIVE_HELPER_FACTS
    functions=[f for f,_,_ in facts]
    assert len(functions)==len({id(f) for f in functions}), 'duplicate helper facts'
    assert functions[0] is loads._native_authority_matches
    captured={id(f) for f in functions}
    def tracked(value):
        return type(value) is types.FunctionType and value.__module__.startswith(('anyfesolver','anymaterial'))
    assert all(tracked(f) for f in functions)
    for name in ('_load_model_access_is_builtin','_load_routing_classes_are_untouched','_numpy_error_routes_are_builtin',
                 '_builtin_acceleration_inputs','_exact_load_case_instance_namespace'):
        assert id(getattr(boundary,name)) in captured,name
    # Rebuild the reachable set and every binding tuple from the live functions, without
    # trusting the captured bindings, and require exact equality in both directions:
    # a dropped fact, a dropped binding or an extra (unreachable) fact all fail.
    def names(code):
        return set(code.co_names).union(*(names(c) for c in code.co_consts if type(c) is types.CodeType))
    def rebuilt_bindings(function):
        g=function.__globals__
        if function is loads._native_authority_matches:
            return tuple((n,n in g,g.get(n)) for n in sorted(names(function.__code__)))
        return tuple((n,g[n]) for n in function.__code__.co_names if n in g)
    roots=[loads._native_authority_matches]+[getattr(boundary,n) for n in (
        '_load_model_access_is_builtin','_load_routing_classes_are_untouched','_numpy_error_routes_are_builtin',
        '_builtin_acceleration_inputs','_exact_load_case_instance_namespace')]
    for cls in loads._NATIVE_CLASS_FACTS:
        for base in cls.__mro__:
            for member in vars(base).values():
                if type(member) in (staticmethod,classmethod): member=member.__func__
                roots.extend((member.fget,member.fset,member.fdel) if type(member) is property else (member,))
    reachable={};work=[f for f in roots if tracked(f)]
    while work:
        function=work.pop()
        if id(function) in reachable: continue
        reachable[id(function)]=function
        successors=[b[-1] for b in rebuilt_bindings(function)]
        for cell in function.__closure__ or ():
            try: successors.append(cell.cell_contents)
            except ValueError: pass
        work.extend(v for v in successors if tracked(v))
    missing=[reachable[i].__qualname__ for i in reachable.keys()-captured]
    extra=[f.__qualname__ for f in functions if id(f) not in reachable]
    assert not missing,('reachable helpers not captured',missing[:10])
    assert not extra,('captured helpers not reachable',extra[:10])
    for function,_,bindings in facts:
        expected=rebuilt_bindings(function)
        assert len(bindings)==len(expected) and all(
            len(a)==len(b) and a[0]==b[0] and all(x is y for x,y in zip(a[1:],b[1:]))
            for a,b in zip(bindings,expected)),('bindings differ',function.__qualname__)
    assert len(functions)>1000   # guards against an accidentally empty or truncated capture

def test_helper_facts_are_exactly_the_reachable_closure():
    # Runs without the optional native extension (the capture is pure Python).
    _assert_helper_facts_closed()

def test_helper_fact_closure_check_detects_a_dropped_fact(monkeypatch):
    facts=loads._NATIVE_HELPER_FACTS
    referenced={id(binding[-1]) for _,_,b in facts for binding in b}
    index=next(i for i in range(1,len(facts)) if id(facts[i][0]) in referenced)
    monkeypatch.setattr(loads,'_NATIVE_HELPER_FACTS',facts[:index]+facts[index+1:])
    with pytest.raises(AssertionError):
        _assert_helper_facts_closed()

def test_helper_fact_closure_check_detects_an_extra_unreachable_fact(monkeypatch):
    def stray(): return 1
    stray.__module__='anyfesolver.stray_probe'
    facts=loads._NATIVE_HELPER_FACTS
    monkeypatch.setattr(loads,'_NATIVE_HELPER_FACTS',facts+((stray,boundary._callable_authority(stray),()),))
    with pytest.raises(AssertionError,match='not reachable'):
        _assert_helper_facts_closed()

def test_helper_fact_closure_check_detects_a_fact_dropped_with_its_bindings(monkeypatch):
    """Drop a helper AND every captured binding that points to it (Codex r1 case)."""
    # boundary._identities_differ is reached only through global bindings (no closure
    # cell, no class member), so the old binding-trusting check accepted this removal.
    facts=loads._NATIVE_HELPER_FACTS
    target=boundary._identities_differ
    index=next(i for i,(f,_,_) in enumerate(facts) if f is target)
    pruned=tuple((f,a,tuple(b for b in bs if b[-1] is not target)) for f,a,bs in facts[:index]+facts[index+1:])
    monkeypatch.setattr(loads,'_NATIVE_HELPER_FACTS',pruned)
    with pytest.raises(AssertionError):
        _assert_helper_facts_closed()

@pytest.mark.parametrize('mutation',['code','defaults','keywords','closure','empty','method','property','mro'])
def test_mutable_metadata_literal_differential(native,mutation):
    token=object()
    def factory():
        value=token
        def function(x=token,*,a=token,b=None):return value,x,a,b
        return function
    function=factory()
    class Base:pass
    class A(Base):
        method=function
        prop=property(function)
        static=staticmethod(function)
        class_method=classmethod(function)
    class B(Base):pass
    value,classes,helpers=plan(native,A,(function,))
    assert native.compare(value,classes,helpers) is True
    if mutation=='code':function.__code__=function.__code__.replace()
    elif mutation=='defaults':function.__defaults__=(None,)
    elif mutation=='keywords':function.__kwdefaults__={'b':None,'a':token}
    elif mutation=='closure':function.__closure__[0].cell_contents=object()
    elif mutation=='empty':del function.__closure__[0].cell_contents
    elif mutation=='method':A.method=lambda *args:None
    elif mutation=='property':A.prop=property(function)
    elif mutation=='mro':A.__bases__=(B,)
    result=native.compare(value,classes,helpers)
    assert result is False or result is NotImplemented
    assert not literal(classes,helpers)

def test_code_equality_for_class_and_identity_for_helper(native):
    def function():return 1
    class A:method=function
    value,classes,helpers=plan(native,A)
    function.__code__=function.__code__.replace()
    assert literal(classes,helpers)
    assert native.compare(value,classes,helpers) is True

def test_kwdefault_helper_order_is_not_class_order(native):
    def function(*,a=1,b=2):return a,b
    class A:pass
    value,classes,helpers=plan(native,A,(function,))
    function.__kwdefaults__={'b':2,'a':1}
    assert literal(classes,helpers)
    assert native.compare(value,classes,helpers) is True

def test_class_inventory_uses_member_names_without_comparing_values(native):
    class Member:
        def __eq__(self,other):pytest.fail('member equality invoked')
        def __lt__(self,other):pytest.fail('member ordering invoked')
    class A:
        zeta=Member()
        alpha=Member()
    value,classes,helpers=plan(native,A)
    alpha=A.alpha
    zeta=A.zeta
    del A.zeta
    A.zeta=zeta
    assert native.compare(value,classes,helpers) is True
    A.alpha=Member()
    assert native.compare(value,classes,helpers) is False
    A.alpha=alpha
    A.extra=Member()
    assert native.compare(value,classes,helpers) is False
    del A.extra
    del A.zeta
    assert native.compare(value,classes,helpers) is False

def test_cycle_path_and_no_cache_across_guards(native):
    def factory():
        def f():return g
        def g():return f
        return f,g
    f,g=factory()
    class A:method=f
    value,classes,helpers=plan(native,A,(f,g))
    assert native.compare(value,classes,helpers) is True
    original=g.__code__
    g.__code__=g.__code__.replace()
    assert native.compare(value,classes,helpers) is False
    g.__code__=original
    assert native.compare(value,classes,helpers) is True

def test_unknown_tuple_dict_code_constant_and_meta_do_not_execute_custom_behavior(native):
    calls=[]
    class Defaults(tuple):
        def __bool__(self):calls.append('bool');return True
    class Keywords(dict):
        def items(self):calls.append('items');return super().items()
    class Constant:
        def __eq__(self,other):calls.append('eq');return True
    def function(x=1,*,k=2):return 3
    class A:method=function
    value,classes,helpers=plan(native,A,(function,))
    function.__defaults__=Defaults((1,))
    assert native.compare(value,classes,helpers) is NotImplemented
    assert calls==[]
    function.__defaults__=(1,)
    function.__kwdefaults__=Keywords(k=2)
    assert native.compare(value,classes,helpers) is NotImplemented
    assert calls==[]
    function.__kwdefaults__={'k':2}
    function.__code__=function.__code__.replace(co_consts=(None,Constant()))
    assert native.compare(value,classes,helpers) is NotImplemented
    assert calls==[]
    class Meta(type):
        def __getattribute__(self,name):calls.append(name);return super().__getattribute__(name)
    class C(metaclass=Meta):pass
    calls.clear()
    changed={C:()}
    assert native.make_plan(changed,(),boundary._MISSING,(static_batch._class_state,)) is not NotImplemented
    # The expected grammar is safe; actual custom metaclass is declined.
    cp=native.make_plan(changed,(),boundary._MISSING,(static_batch._class_state,))
    assert native.compare(cp,changed,()) is NotImplemented
    assert calls==[]

def test_plan_owner_order_and_readonly_type(native):
    class A:pass
    value,classes,helpers=plan(native,A)
    with pytest.raises(AttributeError):value.extra=True
    with pytest.raises(TypeError):type(value)()
    assert native.compare(value,dict(classes),helpers)==2
    classes[A]=()
    assert native.compare(value,classes,helpers)==2
    gc.collect()

def test_actual_native_binding_and_unrelated_unsupported_mapping(native):
    assert dispatch.diagnostics()['backend']=='native'
    before=dispatch.diagnostics()['native_passes']
    loads.require_native_source_authority()
    assert dispatch.diagnostics()['native_passes']==before+1
    sys.modules[123]=object()
    try:loads.require_native_source_authority()
    finally:del sys.modules[123]
    native.__dict__[123]=object()
    try:loads.require_native_source_authority()
    finally:del native.__dict__[123]

def test_admitted_entry_tamper_and_class_failure_priority(native,monkeypatch):
    original=native.compare
    monkeypatch.setattr(native,'compare',lambda *args:True)
    with pytest.raises(loads.PreparedStaticCapabilityError,match='native helper executable authority changed'):
        loads.require_native_source_authority()
    cls=next(iter(loads._NATIVE_CLASS_FACTS))
    monkeypatch.setattr(cls,'_cpp_test_mutation',object(),raising=False)
    with pytest.raises(loads.PreparedStaticCapabilityError,match='native class executable authority changed'):
        loads.require_native_source_authority()
    monkeypatch.delattr(cls,'_cpp_test_mutation')
    monkeypatch.setattr(native,'compare',original)

def test_dispatch_absent_builtin_shadow_refuses_without_body(native,monkeypatch):
    assert 'type' not in dispatch.try_compare.__globals__
    calls=[]
    monkeypatch.setitem(dispatch.try_compare.__globals__,'type',lambda _:calls.append('called'))
    with pytest.raises(loads.PreparedStaticCapabilityError,match='native helper executable authority changed'):
        loads.require_native_source_authority()
    assert calls==[]

@pytest.mark.parametrize('owner',['modules','native'])
def test_unknown_colliding_mapping_key_does_not_execute_equality(native,owner):
    calls=[]
    name='anyfesolver._native_guard' if owner=='modules' else 'compare'
    class Key:
        def __hash__(self):return hash(name)
        def __eq__(self,other):calls.append('eq');return False
    key=Key();mapping=sys.modules if owner=='modules' else native.__dict__
    mapping[key]=object();calls.clear()
    try:
        loads.require_native_source_authority()
        assert calls==[]
    finally:del mapping[key]

def test_owned_generation_current_classification_exact_int(native):
    def factory():
        accepted_generation=1
        def require():return accepted_generation
        require.__module__='anyfesolver._qualified_authority_epoch'
        require.__qualname__='_AuthorityEpochManager.bind.<locals>.require'
        return require
    function=factory()
    class A:method=function
    # Class-state's CURRENT name/module/type rule differs from the stronger
    # captured helper generation flag. This deliberately mimics only the class
    # exception; it does not manufacture a helper generation capability.
    helper_snapshot=boundary._callable_authority(function)
    value,classes,helpers=plan(native,A)
    function.__closure__[0].cell_contents=2
    assert literal(classes,helpers)
    assert not boundary._callable_authority_matches(function,helper_snapshot)
    assert native.compare(value,classes,helpers) is True
    function.__closure__[0].cell_contents=True
    assert not literal(classes,helpers)
    result=native.compare(value,classes,helpers)
    assert result is False or result is NotImplemented

def test_captured_generation_flag_is_exact_int_independent_of_current_name(native):
    def factory():
        epoch=1
        def function():return epoch
        return function
    function=factory()
    class A:pass
    classes={A:static_batch._class_state(A)}
    authority=(function,function.__code__,function.__defaults__,None,((1,(),True),))
    helpers=((function,authority,()),)
    value=native.make_plan(classes,helpers,boundary._MISSING,(static_batch._class_state,boundary._callable_authority_matches))
    assert value is not NotImplemented
    function.__closure__[0].cell_contents=2
    assert boundary._callable_authority_matches(function,authority)
    assert native.compare(value,classes,helpers) is True
    function.__closure__[0].cell_contents=True
    assert not boundary._callable_authority_matches(function,authority)
    assert native.compare(value,classes,helpers) is False

@pytest.mark.parametrize('layout',['no_record','editable','mismatch'])
def test_unadmitted_editable_fallback_and_recorded_tamper_refusal(native,monkeypatch,layout):
    import importlib.util,pathlib
    path=pathlib.Path(dispatch.__file__)
    spec=importlib.util.spec_from_file_location('native_guard_layout_probe',path)
    fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
    class Distribution:
        def read_text(self,name):
            if layout=='no_record':return None
            if layout=='editable':return 'editable_loader.pth,sha256=unused,10\n'
            mapped=native.binary_identity()[0]
            return mapped+',sha256=incorrect,0\n'
        def locate_file(self,member):return pathlib.Path(member)
    monkeypatch.setattr(fresh.importlib.metadata,'distribution',lambda *args:Distribution())
    fresh.initialise({},(),boundary._MISSING,(static_batch._class_state,))
    if layout=='mismatch':
        assert fresh.diagnostics()['backend']=='refusal'
        assert fresh.try_compare({},())==2
    else:
        assert fresh.diagnostics()['backend']=='fallback'
        assert fresh.try_compare({},()) is NotImplemented
        assert fresh.diagnostics()['reason']=='unadmitted_no_binary_record'

@pytest.mark.parametrize('field',['co_linetable','co_exceptiontable'])
def test_code_byte_table_subclasses_decline_before_custom_equality(native,field):
    calls=[]
    class Bytes(bytes):
        __hash__=bytes.__hash__
        def __eq__(self,other):calls.append('eq');return bytes.__eq__(self,other)
    def function():
        try:return 1
        except ValueError:return 2
    class A:method=function
    value,classes,helpers=plan(native,A)
    original=function.__code__
    function.__code__=original.replace(**{field:Bytes(getattr(original,field))})
    retained=type(getattr(function.__code__,field)) is Bytes
    result=native.compare(value,classes,helpers)
    assert calls==[]
    if retained:assert result is NotImplemented
    else:assert result is literal(classes,helpers)
    # Only the literal comparison may observe any reachable stock code-table
    # equality callback. Its verdict is recorded, not guessed from CPython.
    verdict=literal(classes,helpers)
    assert type(verdict) is bool

@pytest.mark.parametrize('pair',[('compare','make_plan'),('entry_identity','statistics')])
def test_cold_real_entry_swap_refuses_before_wrong_arity_api(native,monkeypatch,pair):
    import importlib.util,pathlib
    spec=importlib.util.spec_from_file_location('native_guard_entry_probe',pathlib.Path(dispatch.__file__))
    fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
    left,right=pair;first,second=getattr(native,left),getattr(native,right)
    monkeypatch.setattr(native,left,second);monkeypatch.setattr(native,right,first)
    fresh.initialise({},(),boundary._MISSING,(static_batch._class_state,))
    assert fresh.diagnostics()['backend']=='refusal'
    assert fresh.try_compare({},())==2
