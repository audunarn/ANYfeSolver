"""Actual small caller and direct CSR lifetime controls; no large models."""
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import stiffened_shell as api
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import _scatter,_release_coo_scratch
from anyfesolver._stiffened_shell_b2_assembly_diagnostics import Capture,DiagnosticRefusal,capture_assembly
from examples.stiffened_shell_b2_physical_rigid_footprint import definition,preload

def test_direct_csr_order_values_survive_default_release():
    rows,cols,values=[],[],[];local=np.array([[2.,0.],[-3.,4.]])
    _scatter(rows,cols,values,np.array([1,0]),local)
    k=sparse.coo_matrix((values,(rows,cols)),shape=(2,2)).tocsr()
    before=(k.data.copy(),k.indices.copy(),k.indptr.copy())
    _release_coo_scratch(rows,cols,values,None,'K_CSR')
    assert rows==cols==values==[]
    for actual,expected in zip((k.data,k.indices,k.indptr),before):assert np.array_equal(actual,expected)
    assert np.array_equal(k.toarray(),np.array([[4.,-3.],[0.,2.]]))

def test_actual_lists_empty_before_factor_and_original_work(monkeypatch):
    original=Capture.prefactor;seen=[]
    def checked(self,persistent,transient,borrowed):
        names=('rows','columns','values','constraint_rows','constraint_cols','constraint_values')
        assert all(type(transient[n]) is list and not transient[n] for n in names)
        assert [r['phase'] for r in self.coo_releases]==['K_CSR','C_CSR']
        seen.append(True);return original(self,persistent,transient,borrowed)
    monkeypatch.setattr(Capture,'prefactor',checked);d=definition()
    with capture_assembly() as c:owner=api.create_analysis(definition=d)
    try:
        assert seen and c.coo_releases[0]['preclear_entries']['values']==576*len(d.shells)+144*len(d.members)
        result=owner.solve_linear(preload());assert result.status=='completed'
        u=np.asarray(result.displacements);f=np.asarray(preload());external=float(u@f)
        assert abs(2.*result.internal_strain_energy-external)<=1e-8*abs(external)
        assert owner._handle.factorization_count==1 and 1<=owner._handle.solve_count<=2
    finally:owner.close()

def test_release_counts_and_duration_inside_assembly():
    ticks=iter((0.,1.,2.));c=Capture(lambda:next(ticks));c.begin();c.mark('K_CSR')
    rows=[1,2];cols=[1,2];values=[0.,4.]
    _release_coo_scratch(rows,cols,values,c,'K_CSR');c.mark('attachment_rows')
    assert c.phases[-1]==dict(phase='K_CSR',seconds=1.)
    assert c.coo_releases[0]['preclear_entries']==dict(rows=2,columns=2,values=2)
    assert not rows and not cols and not values

def test_invalid_release_accounting_refuses():
    c=Capture();c.begin();c.mark('K_CSR')
    for phase,counts in [('C_CSR',(1,1,1)),('K_CSR',(1,2,1)),('K_CSR',(-1,-1,-1))]:
        with pytest.raises(DiagnosticRefusal):c.record_coo_release(phase,counts)
    c.record_coo_release('K_CSR',(1,1,1))
    with pytest.raises(DiagnosticRefusal):c.record_coo_release('K_CSR',(1,1,1))
