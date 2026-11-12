"""Cold code grammar must never become a mutable authority proof."""
import gc
import types
import weakref
import pytest
from anyfesolver import boundary, static_batch
from test_native_guard_cpp import plan, literal

@pytest.fixture
def native():
    return pytest.importorskip('anyfesolver._native_guard')

@pytest.mark.parametrize('mutation', ['code', 'default', 'closure', 'global', 'alias'])
def test_admitted_code_still_observes_mutable_facts(native, mutation):
    token=object()
    def factory():
        value=token
        def method(x=token):
            def inner():return ('constant', 3, frozenset((1, 2)))
            return value, x, inner
        return method
    method=factory()
    class A: pass
    A.method=method
    value, classes, helpers=plan(native,A,(method,))
    assert native.compare(value,classes,helpers) is True
    original=method.__code__
    if mutation=='code':method.__code__=original.replace(co_consts=original.co_consts+(17,))
    elif mutation=='default':method.__defaults__=(None,)
    elif mutation=='closure':method.__closure__[0].cell_contents=None
    elif mutation=='alias':A.method=lambda:None
    else:
        # This explicitly captured global binding must remain observed every guard.
        helpers=((method,boundary._callable_authority(method),(('grammar_probe',token),)),)
        method.__globals__['grammar_probe']=token
        value=native.make_plan(classes,helpers,boundary._MISSING,
            (static_batch._class_state,static_batch._callable_state,boundary._callable_authority_matches))
        assert value is not NotImplemented
        assert native.compare(value,classes,helpers) is True
        method.__globals__['grammar_probe']=None
    try:
        result=native.compare(value,classes,helpers)
        assert result is False or result is NotImplemented
        if mutation!='global':assert not literal(classes,helpers)
    finally:
        method.__globals__.pop('grammar_probe',None)

@pytest.mark.parametrize('field',['co_linetable','co_exceptiontable'])
def test_cold_stock_code_does_not_admit_new_table_subclass(native,field):
    calls=[]
    class Bytes(bytes):
        __hash__=bytes.__hash__
        def __eq__(self,other):calls.append('eq');return bytes.__eq__(self,other)
    def method():
        try:return 1
        except ValueError:return 2
    class A:pass
    A.method=method
    value,classes,helpers=plan(native,A)
    assert native.compare(value,classes,helpers) is True
    original=method.__code__
    method.__code__=original.replace(**{field:Bytes(getattr(original,field))})
    result=native.compare(value,classes,helpers)
    assert calls==[]
    if type(getattr(method.__code__,field)) is Bytes:assert result is NotImplemented
    else:assert result is literal(classes,helpers)
    method.__code__=original
    assert native.compare(value,classes,helpers) is True

def test_plan_owns_nested_code_and_releases_gc_cycle(native):
    def method():
        def nested():return (1, 2)
        return nested
    class A:pass
    A.method=method
    # Clone the nested code so the test module constants do not own it.
    method.__code__=method.__code__.replace(co_consts=tuple(
        c.replace() if type(c) is types.CodeType else c for c in method.__code__.co_consts))
    original=method.__code__
    nested=next(c for c in original.co_consts if type(c) is types.CodeType)
    reference=weakref.ref(nested)
    value,classes,helpers=plan(native,A)
    del nested
    gc.collect()
    assert reference() is not None
    assert native.compare(value,classes,helpers) is True
    # Plan -> code -> stock constants and class -> plan cycle is collectible.
    owner_reference=weakref.ref(A)
    A.plan=value
    del value,classes,helpers,A,method,original
    gc.collect()
    assert owner_reference() is None
    assert reference() is None
