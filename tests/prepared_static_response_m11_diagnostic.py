import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
from anyfesolver.assembly import _backward_error
from prepared_static_response_m11_gate import *
from prepared_static_response_m9_diagnostic import (
    _shell_model,_shell_case,mixed_free_model,free_cases,PatternCase,
    RecoveryConfig,prepare_static_response_evaluator,PreparedStaticEvaluator)

def controls():
    A=sparse.eye(1,format='csr');b=np.ones(1);q=np.array([1.+1.5e-10])
    accepted=_backward_error(A,q,b)
    bound=gate_enclosure(A,b,b)
    error=abs(rational(q[0])-ONE)
    assert accepted<=1e-10 and error>rational(1e-12)+rational(1e-12)
    assert error<=bound['radius']
    cancellation=sparse.csr_matrix([[1e16,1.,-1e16],[0.,1.,0.],[0.,0.,1.]])
    bc=np.array([0.,1.,1.]);qc=np.ones(3)
    params=arithmetic_residual_parameters(cancellation,bc)
    true_residual=ONE
    assert true_residual<=params['alpha']+params['zeta']
    zero=sparse.csr_matrix([[0.]])
    assert _backward_error(zero,np.ones(1),np.zeros(1))==0.
    try:gate_enclosure(A,np.zeros(1),np.zeros(1))
    except GateRefusal as exc:zero_refusal=str(exc)
    else:raise AssertionError('zero RHS branch incorrectly admitted')
    ill=gate_enclosure(sparse.diags([1.,1e-20],format='csr'),np.ones(2),np.zeros(2))
    assert ill['status']=='INCONCLUSIVE_GATE_CONTRACTION'
    # Exact constrained equation is unchanged by loads on a constrained DOF.
    K=np.diag([2.,3.]);T=np.array([[0.],[1.]])
    F1=np.array([5.,3.]);F2=np.array([9.,3.]);u=T[:,0]
    assert np.array_equal(T.T@F1,T.T@F2)
    assert not np.array_equal(K@u-F1,K@u-F2)
    assert _backward_error(sparse.csr_matrix(T.T@K@T),np.ones(1),T.T@F1)==0.
    return {'identity_false_pass':{'gate':accepted,'output_error':upward(error),'target':2e-12,'bound':report(bound)},
        'cancellation':{'gate':_backward_error(cancellation,qc,bc),'true_residual':1.,'rounding_bound':report(params)},
        'zero_denominator_native_gate':0.,'zero_RHS_bound_refusal':zero_refusal,
        'illconditioned':report(ill),'same_reduced_load_different_full_reactions':True}

def fixture(family):
    if family=='supported':
        model=_shell_model();patterns={'p':_shell_case(0,pressure=2.).load_case};coef={'p':.3}
    else:
        model,groups,balancing=mixed_free_model()
        patterns={name:load for name,time,load in free_cases(model,groups,balancing,[0,48,576])};coef={'case-048':.3}
    with prepare_static_response_evaluator(model,patterns,recovery_config=RecoveryConfig(include_stresses=False,include_reactions=False)) as owner:
        snap=owner._snapshot
        _,_,packet=owner._ingest_case(PatternCase('probe',coef))
        F=owner._combine(owner._force_columns,packet)
        rhs=np.asarray(snap.transformation.T@(F-snap.stiffness@snap.prescribed_offset))
        if owner._operator_augmented:rhs=np.concatenate((rhs,np.zeros(snap.rigid_modes.shape[1])))
        A=owner._operator_matrix if owner._operator_augmented else snap.reduced_stiffness
        reconstructed=owner._combine(owner._solution_columns,packet)
        result=gate_enclosure(A.tocsr(),rhs,reconstructed)
        # Oracle is executed strictly after the bound and never supplied to it.
        direct=PreparedStaticEvaluator._solve_block(owner,np.ascontiguousarray(rhs[:,None]),())[:,0]
        actual=max(abs(rational(a)-rational(b)) for a,b in zip(direct,reconstructed))
        if result['status']=='CONDITIONAL_ENCLOSURE':assert actual<=result['radius']
        return {'bound':report(result),'independent_direct_gate':_backward_error(A,direct,rhs),'observed_coordinate_difference':upward(actual)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result={'status':'DIAGNOSTIC_ONLY_NO_PROMOTION','scope':['actual finite canonical CSR and represented RHS; no source/assembly enclosure','1D accepted checker branch with positive RHS infinity norm','round-to-nearest binary64, gradual underflow and at most stated CSR dot/reduction operations are conditional assumptions','all relevant intermediates finite; native gate rejects nonfinite results','unknown canonical solution not read by bound; direct solve used only afterward as test oracle','no output or envelope certificate'],'controls':controls(),'fixtures':{f:fixture(f) for f in ('supported','free')}}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
