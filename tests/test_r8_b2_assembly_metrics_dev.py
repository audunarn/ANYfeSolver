"""Small source/installed actual-owner diagnostics controls; no scale claim."""
from dataclasses import replace
import json,os
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver._stiffened_shell_b2_assembly_diagnostics import (
    Inventory,Capture,DiagnosticRefusal,capture_assembly,active_capture,ASSEMBLY)
from examples.stiffened_shell_b2_physical_rigid_footprint import definition,preload


def test_actual_owner_parity_and_factor_reuse():
    d=definition();plain=api.create_analysis(definition=d)
    try:
        with capture_assembly() as capture:measured=api.create_analysis(definition=d)
        try:
            assert capture.status=='COMPLETE' and capture.storage['total_bytes']>0
            if 'B2_DIAGNOSTIC_EVIDENCE' in os.environ:
                (Path(os.environ['B2_DIAGNOSTIC_EVIDENCE'])/'actual-capture.json').write_text(json.dumps(capture.report(),indent=2))
            assert set(ASSEMBLY)<=set(x['phase'] for x in capture.phases)
            for name in ('_K','_C','_T','_physical_scaled_kkt'):
                a,b=getattr(plain,name),getattr(measured,name)
                assert (a!=b).nnz==0
            a=plain.solve_linear(preload());b=measured.solve_linear(preload())
            assert a.status==b.status=='completed'
            assert a.displacements==b.displacements
            assert plain._handle.factorization_count==measured._handle.factorization_count==1
            assert plain._handle.solve_count==measured._handle.solve_count
        finally:measured.close()
    finally:plain.close()
    assert active_capture() is None


def test_fakeclock_failure_closes_phase_and_cleans_context():
    ticks=iter((0.,2.,5.))
    with pytest.raises(ValueError,match='original'):
        with capture_assembly(clock=lambda:next(ticks)) as c:
            c.begin();c.mark('shell_response_scatter');raise ValueError('original')
    assert c.status=='FAILED' and c.phases[-1]==dict(phase='shell_response_scatter',seconds=3.)
    assert active_capture() is None


def test_view_only_owner_and_alias_invariance():
    base=np.arange(80.);view=base[::2]
    a=Inventory();a.add('persistent',view)
    b=Inventory();b.add('persistent',(view,view,base))
    assert a.result()['groups']['persistent']['buffer_bytes']==base.nbytes
    assert a.result()['unique_buffers']==b.result()['unique_buffers']==1
    assert a.result()['unique_objects']>=2
    # Adding an already declared root must not change object or buffer bytes.
    before=a.result()['total_bytes'];a.add('persistent',view);assert a.result()['total_bytes']==before
    borrowed=Inventory(borrowed=(base,));borrowed.add('persistent',view)
    assert borrowed.result()['groups']['persistent']['buffer_bytes']==0


def test_sparse_namespace_and_extra_storage_refusal():
    matrix=sparse.eye(4,format='csr');i=Inventory();i.add('persistent',matrix)
    assert id(matrix.__dict__) in i.seen and id(matrix.data) in i.seen
    matrix.extra_array_cache=np.ones(3)
    with pytest.raises(DiagnosticRefusal,match='sparse retained'):Inventory().add('persistent',matrix)


def test_unknown_object_array_and_caps_refuse():
    for value in (object(),np.array([object()],dtype=object),np.array([(object(),)],dtype=[('v',object)])[0]):
        with pytest.raises(DiagnosticRefusal):Inventory().add('persistent',value)
    with pytest.raises(DiagnosticRefusal):Inventory(caps=dict(objects=1,edges=1,buffers=1,depth=1)).add('persistent',[1,2])


def test_nested_capture_and_failed_admission_no_inventory(monkeypatch):
    with capture_assembly():
        with pytest.raises(DiagnosticRefusal,match='nested'):
            with capture_assembly():pass
    assert active_capture() is None
    with pytest.raises(ValueError):
        with capture_assembly() as c:
            api.create_analysis(definition=replace(definition(),shells=()))
    assert c.status=='FAILED' and c.storage is None and active_capture() is None
    d=definition();mass_shell=replace(d.shells[0],density_kg_m3=1000.)
    with pytest.raises(DiagnosticRefusal,match='mass/body-map'):
        with capture_assembly() as c:
            api.create_analysis(definition=replace(d,shells=(mass_shell,*d.shells[1:])))
    assert c.storage is None and active_capture() is None


def test_default_avoids_inventory_and_precancel_counts(monkeypatch):
    def forbidden(*args):raise AssertionError('default inventory reached')
    monkeypatch.setattr(Inventory,'add',forbidden)
    owner=api.create_analysis(definition=definition())
    try:
        token=CancellationToken();token.cancel()
        before=owner._handle.solve_count
        result=owner.solve_linear_many((('one',preload()),),selected_dofs=(12,),cancellation_token=token)
        assert result.status=='cancelled' and result.cases==()
        assert owner._handle.solve_count==before and owner._handle.factorization_count==1
    finally:owner.close()


def test_inventory_error_is_not_mechanics_success(monkeypatch):
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as product
    def refuse(*args):raise DiagnosticRefusal('deliberate inventory')
    monkeypatch.setattr(Capture,'prefactor',refuse)
    with pytest.raises(DiagnosticRefusal,match='deliberate'):
        with capture_assembly() as c:api.create_analysis(definition=definition())
    assert c.status=='FAILED' and active_capture() is None
    monkeypatch.undo()
    closed=[];original_close=product.ExperimentalB2Q4PanelAnalysis.close
    def close(owner):closed.append(1);return original_close(owner)
    def fail_after_factor(capture):raise DiagnosticRefusal('after factor')
    monkeypatch.setattr(product.ExperimentalB2Q4PanelAnalysis,'close',close)
    monkeypatch.setattr(Capture,'finish',fail_after_factor)
    with pytest.raises(DiagnosticRefusal,match='after factor'):
        with capture_assembly():api.create_analysis(definition=definition())
    assert closed and active_capture() is None
