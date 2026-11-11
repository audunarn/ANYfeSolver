"""Solve-local exact baseline reuse and lifetime/authority regressions."""
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from anyfesolver import e4_pl_element as q4, vectorized_nonlinear as batch
from anyfesolver.control import SolveCancelled
from test_general_contact_c5_elastic_increment import case, pose

@contextmanager
def zero_calls():
    calls=[]
    code=batch.compensated_elastic_batch_zero.__code__
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is code: calls.append(1)
    previous=sys.getprofile();sys.setprofile(profile)
    try: yield calls
    finally: sys.setprofile(previous)

def evaluate(model,element,material):
    return element._elastic_increment_response(model.mesh,material,pose(),None,True,batch_baseline=True)

def test_scope_reuses_exact_baseline_and_nested_scope_is_independent():
    m,e,mat=case()
    with zero_calls() as calls:
        reference=evaluate(m,e,mat)
        with q4._elastic_batch_baseline_scope():
            cold=evaluate(m,e,mat);warm=evaluate(m,e,mat)
            assert len(calls)==2
            with q4._elastic_batch_baseline_scope(): evaluate(m,e,mat)
            evaluate(m,e,mat)
            assert len(calls)==3
        evaluate(m,e,mat)
        assert len(calls)==4
    for expected,a,b in zip(reference,cold,warm):
        np.testing.assert_array_equal(expected,a);np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize("error", [RuntimeError, SolveCancelled])
def test_scope_release_on_exception_and_copied_context(error):
    from contextvars import copy_context
    m,e,mat=case()
    with zero_calls() as calls:
        with pytest.raises(error,match='deliberate'):
            with q4._elastic_batch_baseline_scope():
                evaluate(m,e,mat)
                copied=copy_context()
                owned=next(value for key,value in copied.items()
                           if key.name=='qualified_q4_batch_baselines')
                assert not hasattr(owned,"records")
                raise error('deliberate')
        assert not hasattr(owned,"records")
        copied.run(evaluate,m,e,mat)
        copied.run(evaluate,m,e,mat)
        assert len(calls)==3

def test_warm_scope_rejects_changed_dependency(monkeypatch):
    m,e,mat=case()
    with q4._elastic_batch_baseline_scope():
        evaluate(m,e,mat)
        monkeypatch.setattr(batch,'compensated_elastic_batch_zero',lambda *a: np.zeros((24,24)))
        with pytest.raises((ValueError,RuntimeError),match='changed|authority'):
            evaluate(m,e,mat)

def test_material_change_invalidates_original_binding():
    m,e,mat=case()
    with zero_calls() as calls,q4._elastic_batch_baseline_scope():
        old=evaluate(m,e,mat)
        mat.elastic_modulus*=2
        changed=evaluate(m,e,mat)
        evaluate(m,e,mat)
        assert len(calls)==2
        assert not np.array_equal(old[0],changed[0])

def test_factory_observation_cannot_publish_inconsistent_material():
    m,e,mat=case()
    e.compute_stiffness_matrix(m.mesh,mat)
    snapshot=q4._capture_q4_fast_input_snapshot(e,m.mesh,mat)
    original=mat.elastic_modulus
    def mutate(): mat.elastic_modulus=original*2
    with q4._elastic_batch_baseline_scope():
        with pytest.raises((RuntimeError,ValueError),match='changed|stale'):
            q4._prepare_elastic_batch_baseline(e,m.mesh,mat,snapshot,mutate)
        mat.elastic_modulus=original
        with zero_calls() as calls:
            evaluate(m,e,mat);evaluate(m,e,mat)
            assert len(calls)==1

def test_geometry_aba_invalidates_even_restored_coordinates():
    m,e,mat=case()
    with zero_calls() as calls,q4._elastic_batch_baseline_scope():
        first=evaluate(m,e,mat)
        x=m.mesh.nodes[2].x
        m.mesh.nodes[2].x=x+.1
        m.mesh.nodes[2].x=x
        second=evaluate(m,e,mat)
        assert len(calls)==2
    for a,b in zip(first,second): np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('name',['_prepare_elastic_batch_baseline','_elastic_batch_baseline_scope'])
def test_runtime_preparation_binding_cannot_be_replaced(monkeypatch,name):
    m,e,mat=case()
    with q4._elastic_batch_baseline_scope():
        evaluate(m,e,mat)
        monkeypatch.setattr(q4,name,lambda *args,**kwargs: None)
        with pytest.raises((ValueError,RuntimeError),match='changed|authority'):
            evaluate(m,e,mat)
