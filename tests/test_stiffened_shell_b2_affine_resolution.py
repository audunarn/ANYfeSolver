"""Exact affine proof controls and the actual zero-offset public support route."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import stiffened_shell as api
from anyfesolver import _stiffened_shell_b2_affine_resolution as proof
from anyfesolver import _stiffened_shell_b2_q4_panel_experimental as panel
from anyfesolver._b2_physical_linear_batch_dev import B2NumericalSolveFailure
from examples.stiffened_shell_b2_physical_rigid_footprint import definition


def certify(T,q,f,K=None,u=None,cache=None):
    T=sparse.csr_matrix(T);K=sparse.eye(T.shape[0],format='csr') if K is None else K
    return proof.certify_affine(K,sparse.csr_matrix((0,T.shape[0])),T,np.asarray(q,float),
        np.asarray(f,float),np.asarray(T@q if u is None else u,float),{} if cache is None else cache)


@pytest.mark.parametrize('T',([[2.,0.],[0.,.5]], [[-1.,0.],[0.,-2.]], [[1.,.25],[.5,-2.],[0.,1.]]))
def test_scaled_negative_mixed_virtual_coordinates_are_certified(T):
    T=np.array(T);q=np.array([.5,-.25]);f=T@q
    assert certify(T,q,f)


def test_exact_projected_cancellation_is_not_rounded_rhs_or_kkt_proof():
    T=np.ones((3,1));f=np.array([1e16,1.,-1e16]);cache={}
    assert sum(Fraction(float(x)) for x in f)==1 and float(np.dot(T[:,0],f))!=1.
    assert certify(T,[1/3],f,cache=cache)
    closure=cache['closure']
    assert not certify(T,[0.],f,cache=cache) # corrupt reduced rhs can report zero defect
    assert cache['closure'] is closure


def test_nonzero_net_tiny_load_cannot_be_omitted():
    T=np.array([[2.,0.],[0.,.5]]);q=np.array([1.,0.]);f=np.array([2.,1e-12])
    assert not certify(T,q,f)


def test_actual_reconstruction_error_is_amplified_by_original_stiffness():
    T=np.array([[2.],[0.]]);K=sparse.diags([1e16,1.],format='csr');f=np.array([2e16,0.])
    assert certify(T,[1.],f,K=K)
    assert not certify(T,[1.],f,K=K,u=[2.00000001,0.])


@pytest.mark.parametrize('f',([0.,0.],[np.nextafter(0.,1.),0.],[float('nan'),0.],[float('inf'),0.]))
def test_zero_subnormal_and_nonfinite_refuse(f):
    if not any(f):assert not certify([[2.,0.],[0.,.5]],[0.,0.],f)
    else:
        with pytest.raises(B2NumericalSolveFailure):certify([[2.,0.],[0.,.5]],[0.,0.],f)


def test_subnormal_projection_and_unrepresentable_reconstruction_refuse():
    # Subnormal g is refused before computing the relative cap.
    with pytest.raises(B2NumericalSolveFailure):certify([[1e-10]], [0.], [np.finfo(float).tiny])
    with pytest.raises(B2NumericalSolveFailure):certify([[1e308]],[2.],[1.],u=[0.])


def test_sparse_closure_work_and_storage_limits_refuse_without_dense(monkeypatch):
    monkeypatch.setattr(proof,'MAX_TERM_VISITS',1)
    with pytest.raises(B2NumericalSolveFailure,match='work limit'):certify([[2.,0.],[0.,.5]],[1.,1.],[2.,.5])
    monkeypatch.setattr(proof,'MAX_TERM_VISITS',200000);monkeypatch.setattr(proof,'MAX_CLOSURE_INDICES',1)
    with pytest.raises(B2NumericalSolveFailure,match='work limit'):certify([[2.,0.],[0.,.5]],[1.,1.],[2.,.5])


def test_current_coefficients_and_incidence_are_not_stale_cached_proof():
    T=sparse.csr_matrix([[2.],[.5]]);K=sparse.eye(2,format='csr');cache={};f=np.array([2.,.5])
    assert certify(T,[1.],f,K=K,cache=cache)
    K.data[0]=2.;assert not certify(T,[1.],f,K=K,cache=cache)
    K=sparse.eye(2,format='csr');T[1,0]=1.;assert not certify(T,[1.],f,K=K,cache=cache)


def test_multiplier_action_is_required_in_original_physical_residual():
    T=sparse.csr_matrix([[2.],[.5]]);K=sparse.eye(2,format='csr');C=sparse.csr_matrix([[1.,-1.]])
    f=np.array([3.,-.5]);u=np.array([2.,.5])
    assert proof.certify_affine(K,C,T,np.array([1.,1.]),f,u,{})
    assert not proof.certify_affine(K,C,T,np.array([1.,0.]),f,u,{})


def test_finite_rational_cancellation_cannot_hide_binary64_product_overflow():
    T=sparse.csr_matrix([[.5],[.5]]);K=sparse.csr_matrix([[4.,-4.],[-4.,4.]])
    C=sparse.csr_matrix([[1.,1.]]);q=np.finfo(float).max
    with pytest.raises(B2NumericalSolveFailure,match='intermediate not representable'):
        proof.certify_affine(K,C,T,np.array([q,1.]),np.ones(2),np.array([q*.5]*2),{})


def test_rational_bit_cost_limit_is_a_refusal(monkeypatch):
    monkeypatch.setattr(proof,'MAX_RATIONAL_BITS',8)
    with pytest.raises(B2NumericalSolveFailure,match='work limit'):
        certify([[2.]],[.1],[.2])


@pytest.mark.parametrize('failure',(B2NumericalSolveFailure,TypeError))
def test_general_caller_catches_only_typed_numerical_failure(monkeypatch,failure):
    owner=SimpleNamespace(_u0=np.zeros(2),_T=sparse.csr_matrix([[2.],[.5]]),_K=sparse.eye(2),
        _C=sparse.csr_matrix((0,2)),footprints=(object(),),_point_trace_policy=None,_affine_resolution_cache=None)
    def broken(*args):raise failure('injected affine failure')
    monkeypatch.setattr(panel,'certify_affine',broken)
    args=(np.array([1.]),np.ones(1),np.ones(1))
    if failure is B2NumericalSolveFailure:
        assert not panel.ExperimentalB2Q4PanelAnalysis._explicit_load_resolved(owner,*args,applied=np.ones(2),reconstructed=np.ones(2))
    else:
        with pytest.raises(TypeError,match='injected affine'):
            panel.ExperimentalB2Q4PanelAnalysis._explicit_load_resolved(owner,*args,applied=np.ones(2),reconstructed=np.ones(2))
    owner._point_trace_policy='loaded-point'
    assert not panel.ExperimentalB2Q4PanelAnalysis._explicit_load_resolved(owner,*args,applied=np.ones(2),reconstructed=np.ones(2))


def test_caller_ignores_corrupted_reduced_kkt_and_rhs_for_general_map():
    T=sparse.csr_matrix([[2.],[.5]]);owner=SimpleNamespace(_u0=np.zeros(2),_T=T,_K=sparse.eye(2,format='csr'),
        _C=sparse.csr_matrix((0,2)),footprints=(object(),),_point_trace_policy=None,_affine_resolution_cache=None,
        _resolution_kkt=sparse.csr_matrix([[0.]]))
    args=(np.array([0.]),np.array([0.]),np.array([0.]))
    assert not panel.ExperimentalB2Q4PanelAnalysis._explicit_load_resolved(owner,*args,applied=np.array([2.,.5]),reconstructed=np.zeros(2))
    assert panel.ExperimentalB2Q4PanelAnalysis._explicit_load_resolved(owner,np.array([1.]),np.array([99.]),np.array([99.]),
        applied=np.array([2.,.5]),reconstructed=np.array([2.,.5]))
    owner._u0[0]=1e-5
    assert not panel.ExperimentalB2Q4PanelAnalysis._explicit_load_resolved(owner,np.array([1.]),np.array([99.]),np.array([99.]),
        applied=np.array([2.,.5]),reconstructed=np.array([2.,.5]))


def test_actual_public_quarter_support_route_keeps_work_balance_gap_and_factor_cache(monkeypatch):
    d=definition();free=[i for i in range(54) if i not in range(36,42) and i!=49]
    T=sparse.eye(54,format='lil')[:,free];T[49,free.index(25)]=.25
    owner=api.create_analysis(definition=replace(d,transformation=T.tocsr()))
    observed=[];original=owner._explicit_load_resolved
    def record(*args,**kwargs):
        value=original(*args,**kwargs);observed.append(value);return value
    monkeypatch.setattr(owner,'_explicit_load_resolved',record)
    try:
        f=np.zeros(54);f[[13,14,48,49,50,51,52,53]]=[.017,-.025,.0315,.022,-.018,.004,-.006,.009]
        # Certification is on demand: a tiny superposed explicit load below
        # the roundoff-relative floor triggers the small-load resolution
        # guard, so this route genuinely exercises the affine certificate.
        f[12]=1e-8
        selected=api.B2PhysicalRigidFootprintSelection(('physical-left',))
        first=owner.solve_linear(f,selected_footprints=selected)
        assert first.status=='completed',first.failure_reason
        assert observed and all(observed) and owner._affine_resolution_cache is not None and owner._resolution_kkt is None
        u=np.array(first.displacements);r=np.array(first.affine_reactions);L=2.5
        F=np.linalg.norm(np.r_[f.reshape(-1,6)[:,:3].ravel(),f.reshape(-1,6)[:,3:].ravel()/L])
        assert np.max(np.abs(owner._C@u)/(abs(owner._C)@np.tile([L]*3+[1.]*3,9)))<=1e-9
        assert np.max(np.abs(T.T@r)/(abs(T.T)@np.tile([F]*3+[F*L]*3,9)))<=1e-8
        coordinates=np.array([d.shell_nodes_m[i] for i in sorted(d.shell_nodes_m)]+[(.5,.5,0.),(1.5,.5,0.),(2.5,.5,0.)])
        actions=(f+r).reshape(-1,6)
        assert np.linalg.norm(actions[:,:3].sum(axis=0))/F<=1e-8
        assert np.linalg.norm((np.cross(coordinates,actions[:,:3])+actions[:,3:]).sum(axis=0))/(F*L)<=1e-8
        work=float(u@(f+r));assert abs(2*first.internal_strain_energy-work)<=1e-8*abs(work)
        cache=owner._affine_resolution_cache;factor=owner._handle
        again=owner.solve_linear(f,selected_footprints=selected)
        assert again.status=='completed' and again.displacements==first.displacements
        assert owner._affine_resolution_cache is cache and owner._handle is factor and again.factorization_count==1
    finally:owner.close()
    assert owner._affine_resolution_cache is None
