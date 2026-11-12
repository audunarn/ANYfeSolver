"""Compensated full-map numerical kernel; not a qualification certificate."""
from fractions import Fraction
import numpy as np
import pytest
from scipy import linalg
from anyfesolver.control import CancellationToken, SolveCancelled
import anyfesolver._native_factor_chain_modes as kernel


def solve(f,g,b,free,algebraic,**kw):
    return kernel.solve_factor_chain_modes(f,np.eye(f.shape[1]),g,b,free,algebraic,**kw)


def test_fma_and_portable_product_residual_match_exact_rational_arithmetic(monkeypatch):
    from math import fsum
    rng=np.random.default_rng(34)
    pairs=[(float(x),float(y)) for x,y in rng.normal(size=(100,2))]
    pairs.extend([(1e150,1e-150),(1e-100,1e50),(.1,.3)])
    expected=[float(Fraction(x)*Fraction(y)-Fraction(x*y)) for x,y in pairs]
    assert [kernel.product_error(x,y,x*y) for x,y in pairs]==expected
    monkeypatch.setattr(kernel,'_fma',None)
    assert [kernel.product_error(x,y,x*y) for x,y in pairs]==expected
    assert fsum(kernel.dot_terms([1e16,1.,-1e16],[1.,1.,1.]))==1.


@pytest.mark.parametrize('x,y',[(1e308,2.),(1e-300,1e-300),(float('nan'),1.)])
def test_unrepresentable_products_are_not_silently_dropped(x,y):
    with pytest.raises(ValueError): kernel.dot_terms([x],[y])


def test_exact_rigid_cluster_and_partial_cluster_rejection():
    f=np.diag([1.,1.,1e12]); g=np.diag([-1.,-1.,0.])
    out=solve(f,g,np.eye(3),(0,1,2),(),bounds=(-1.,1.),num_modes=2)
    np.testing.assert_allclose(out.eigenvalues,0.,atol=1e-11)
    np.testing.assert_allclose(out.full_modes.T@out.full_modes,np.eye(2),atol=1e-11)
    with pytest.raises(ValueError,match='truncate'):
        solve(f,g,np.eye(3),(0,1,2),(),bounds=(-1.,1.),num_modes=1)


def test_signed_high_contrast_does_not_clip_negative_eigenvalue():
    f=np.diag([1.,1e12]); g=np.array([[-3.,1e12],[1e12,0.]])
    out=solve(f,g,np.eye(2),(0,1),(),bounds=(-5.,0.),num_modes=1)
    assert abs(out.eigenvalues[0]+3.)<1e-10
    assert abs(out.full_modes[1,0]/out.full_modes[0,0]+1e-12)<1e-23
    assert out.negative_eigenvalues_retained and not out.certified_intervals and not out.production_qualified
    for a in (out.eigenvalues,out.full_modes,out.numerical_brackets):
        with pytest.raises(ValueError): a.setflags(write=True)


def test_large_coupled_material_and_geometric_cancellation_retains_unit_stiffness():
    # Exact powers of two: F.T F + G = I in the supplied binary64 data.
    # Rounded G@V and F@V intermediates previously produced false roots.
    a=float(2**40)
    f=np.vstack((a*np.array([[1.,1.],[0.,1.]]),np.eye(2)))
    g=-a*a*np.array([[1.,1.],[1.,2.]])
    out=solve(f,g,np.eye(2),(0,1),(),bounds=(-2.,2.),num_modes=2)
    np.testing.assert_allclose(out.eigenvalues,[1.,1.],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(out.full_modes.T@out.full_modes,np.eye(2),rtol=1e-11,atol=1e-11)


def test_signed_trace_and_nondiagonal_mass_match_direct_moderate_pencil():
    rng=np.random.default_rng(53); f=rng.normal(size=(12,6))
    g=rng.normal(size=(6,6)); g=.1*(g+g.T)
    b=np.zeros((9,6)); b[:,:4]=rng.normal(size=(9,4))
    k=f.T@f+g; mass=b.T@b
    expected=linalg.eigvalsh(k[:4,:4]-k[:4,4:]@np.linalg.solve(k[4:,4:],k[4:,:4]),mass[:4,:4])
    out=solve(f,g,b,tuple(range(6)),(4,5),bounds=(-10.,100.),num_modes=4)
    np.testing.assert_allclose(out.eigenvalues,expected,rtol=1e-11,atol=1e-10)
    np.testing.assert_allclose(k@out.full_modes-(mass@out.full_modes)*out.eigenvalues,0.,atol=1e-10)


def test_portable_arithmetic_backend_has_byte_identical_small_modes(monkeypatch):
    rng=np.random.default_rng(75); f=rng.normal(size=(9,4)); raw=rng.normal(size=(4,4))
    g=(raw+raw.T)*.1; b=np.diag([1.,2.,1.5,.75]); kw=dict(bounds=(-10.,100.),num_modes=4)
    first=solve(f,g,b,(0,1,2,3),(),**kw)
    monkeypatch.setattr(kernel,'_fma',None)
    second=solve(f,g,b,(0,1,2,3),(),**kw)
    for name in ('eigenvalues','full_modes','numerical_brackets'):
        assert getattr(first,name).tobytes()==getattr(second,name).tobytes()


@pytest.mark.parametrize('kind',['geometric','kinetic','bounds','dofs','count','width'])
def test_invalid_inputs_fail_closed(kind):
    f=np.eye(2); g=np.zeros((2,2)); b=np.eye(2); free=(0,1)
    kw=dict(bounds=(-1.,2.),num_modes=2,root_width=1e-10)
    if kind=='geometric': g[0,0]=np.nan
    elif kind=='kinetic': b=np.zeros((0,2))
    elif kind=='bounds': kw['bounds']=(False,2.)
    elif kind=='dofs': free=(0,0)
    elif kind=='count': kw['num_modes']=True
    else: kw['root_width']=True
    with pytest.raises((ValueError,np.linalg.LinAlgError)): solve(f,g,b,free,(),**kw)


def test_precancel_and_deadline_run_no_reduction(monkeypatch):
    monkeypatch.setattr(kernel,'_reduce',lambda *a:pytest.fail('reduction after stop'))
    token=CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled): solve(np.eye(2),np.zeros((2,2)),np.eye(2),(0,1),(),
        bounds=(-1.,2.),num_modes=2,cancellation_token=token)
    times=iter((0.,601.)); monkeypatch.setattr(kernel,'monotonic',lambda:next(times))
    with pytest.raises(ValueError,match='deadline'):
        solve(np.eye(2),np.zeros((2,2)),np.eye(2),(0,1),(),bounds=(-1.,2.),num_modes=2)


def test_input_mutation_does_not_change_owned_snapshot(monkeypatch):
    f=np.diag([1.,2.]); g=np.zeros((2,2)); b=np.eye(2); original=kernel._reduce
    def change(*args):
        f[:]=100.; g[:]=-100.; b[:]=0.
        return original(*args)
    monkeypatch.setattr(kernel,'_reduce',change)
    out=solve(f,g,b,(0,1),(),bounds=(-1.,5.),num_modes=2)
    np.testing.assert_allclose(out.eigenvalues,[1.,4.],rtol=1e-11,atol=1e-11)


def test_original_bilinear_guard_detects_false_weak_root_hidden_by_global_backward_scale(monkeypatch):
    original=kernel.reassemble
    def corrupt(*args):
        h,m=original(*args); weak=int(np.argmin(np.diag(h))); h[weak,weak]+=.01
        return h,m
    monkeypatch.setattr(kernel,'reassemble',corrupt)
    with pytest.raises(ValueError,match='original signed bilinear Ritz'):
        solve(np.diag([1.,1e12]),np.zeros((2,2)),np.eye(2),(0,1),(),bounds=(-1.,2.),num_modes=1)


def test_cancellation_during_reassembly_exits_before_bracketing(monkeypatch):
    token=CancellationToken(); original=kernel.reassemble
    def change(*args):
        token.cancel('reassembly stop'); return original(*args)
    monkeypatch.setattr(kernel,'reassemble',change)
    monkeypatch.setattr(kernel,'brackets',lambda *a:pytest.fail('brackets after cancel'))
    with pytest.raises(SolveCancelled):
        solve(np.eye(2),np.zeros((2,2)),np.eye(2),(0,1),(),bounds=(-1.,5.),num_modes=2,cancellation_token=token)


def test_unrepresentable_physical_mode_from_extreme_chain_fails_closed():
    left=np.array([[1e16,1.],[1e16,0.]])
    right=np.array([[1.,1.],[0.,1.]])
    assert np.array_equal(left@right,np.full((2,2),1e16))
    # The necessary physical-component correction is below binary64 spacing.
    # An accurate reduced root alone is not authority to return a bad vector.
    with pytest.raises(ValueError,match='reassembled action residual'):
        kernel.solve_factor_chain_modes(left,right,np.zeros((2,2)),np.eye(2),(0,1),(),
            bounds=(-1.,1.),num_modes=1)
