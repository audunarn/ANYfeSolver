import numpy as np
import pytest
from scipy import sparse
from anyfesolver.assembly import _backward_error
from prepared_static_response_m11_gate import *

def test_identity_accepted_solution_outside_output_tolerance_is_enclosed():
    A=sparse.eye(1,format='csr');b=np.ones(1);q=np.array([1.+1.5e-10])
    result=gate_enclosure(A,b,b)
    assert _backward_error(A,q,b)<=1e-10
    assert rational(q[0])-ONE>rational(2e-12)
    assert rational(q[0])-ONE<=result['radius']
    assert result['uses_observed_direct'] is False

def test_cancellation_and_subnormal_model_terms():
    A=sparse.csr_matrix([[1e16,1.,-1e16],[0.,1.,0.],[0.,0.,1.]])
    b=np.array([0.,1.,1.]);q=np.ones(3)
    assert _backward_error(A,q,b)==0.
    p=arithmetic_residual_parameters(A,b)
    assert ONE<=p['alpha']+p['zeta']  # true residual hidden by rounded dot
    sub=np.nextafter(0.,1.)
    unit=sparse.eye(1,format='csr')
    p=arithmetic_residual_parameters(unit,np.array([sub]))
    assert p['zeta']>=3*TINY
    result=gate_enclosure(unit,np.array([sub]),np.array([sub]))
    assert result['radius']>0 and _backward_error(unit,np.array([sub]),np.array([sub]))==0.

def test_refuse_nonpositive_nonfinite_overflow_and_illconditioning():
    unit=sparse.eye(1,format='csr')
    with pytest.raises(GateRefusal,match='binary64'):
        gate_enclosure(unit,np.array([1],dtype=np.float32),np.zeros(1))
    with pytest.raises(GateRefusal,match='zero_RHS'):
        gate_enclosure(unit,np.zeros(1),np.zeros(1))
    with pytest.raises(GateRefusal,match='nonfinite'):
        gate_enclosure(sparse.csr_matrix([[np.nan]]),np.ones(1),np.zeros(1))
    largest=np.finfo(float).max
    A=sparse.csr_matrix([[largest,largest],[0.,1.]])
    with pytest.raises(GateRefusal,match='row_sum_overflow'):
        gate_enclosure(A,np.ones(2),np.zeros(2))
    result=gate_enclosure(sparse.diags([1.,1e-20],format='csr'),np.ones(2),np.zeros(2))
    assert result['status']=='INCONCLUSIVE_GATE_CONTRACTION'
