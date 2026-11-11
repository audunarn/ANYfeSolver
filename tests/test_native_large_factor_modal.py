"""Exact reassociation, preserved-domain bytes and genuine repeated modes."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from anyfesolver import _native_modal_capacity as capacity
from anyfesolver import _native_large_factor_chain as large
from anyfesolver._dyadic_bilinear import _capture,_multiply,_transpose,_rounded
from anyfesolver._dyadic_factor_chain import reassemble_chain_exact_binary64 as chain
from anyfesolver._native_paired_factor_chain_modes import solve_paired_factor_chain_modes as old,apply_mode_map
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken,SolveCancelled


@pytest.mark.parametrize('seed',(1,23,44))
def test_sparse_integer_products_and_gram_are_exact(seed):
    rng=np.random.default_rng(seed);a=rng.normal(size=(9,7));b=rng.normal(size=(7,8))
    a[rng.uniform(size=a.shape)<.6]=0.;b[rng.uniform(size=b.shape)<.6]=0.
    aa=_capture(a,lambda:None);bb=_capture(b,lambda:None)
    assert large.multiply(aa,bb,lambda:None)==_multiply(aa,bb,lambda:None)
    assert large.gram(aa,lambda:None)==_multiply(_transpose(aa),aa,lambda:None)


@pytest.mark.parametrize('seed',(12,39,71))
def test_reassociation_preserves_every_rounded_entry(seed):
    rng=np.random.default_rng(seed)
    l=rng.normal(size=(9,7));r=rng.normal(size=(7,6));b=rng.normal(size=(11,6));v=rng.normal(size=(6,4))
    g=rng.normal(size=(6,6));g=g+g.T
    before=chain(l,r,g,b,v)
    with capacity.large_modal_capacity():after=chain(l,r,g,b,v)
    for a,c in zip(before,after):assert a.tobytes()==c.tobytes()


def test_reassociation_retains_small_cancellation_work():
    l=np.array([[1e16,1.],[1e16,0.]]);r=np.array([[1.,1.],[0.,1.]]);v=np.array([[-1.],[1.]])
    assert not np.any((l@r)@v)
    with capacity.large_modal_capacity():h,m=chain(l,r,np.zeros((2,2)),np.eye(2),v)
    np.testing.assert_array_equal(h,[[1.]]);np.testing.assert_array_equal(m,[[2.]])


@pytest.mark.parametrize('case',('repeated','signed-wide'))
def test_preserved_domain_mode_bytes_are_unchanged(case):
    left=np.diag([1.,1.,2.]) if case=='repeated' else np.diag([1.,2.,1e12])
    g=np.zeros((3,3)) if case=='repeated' else np.diag([-3.,-1.,0.])
    args=(left,np.eye(3),g,np.eye(3),(0,1,2),())
    before=old(*args,bounds=(-10.,2e24),num_modes=3)
    after=capacity.solve_large_factor_chain_modes(*args,bounds=(-10.,2e24),num_modes=3)
    assert canonical(before)==canonical(after)


def extended_problem():
    n=257;l=np.diag(np.r_[1.,1.,2.,3.,np.full(n-4,2.)]);b=np.zeros((n,n));b[:4,:4]=np.eye(4)
    return (l,np.eye(n),np.zeros((n,n)),b,tuple(range(n)),tuple(range(4,n)))


def test_actual_extended_coordinates_include_repeated_modes(tmp_path):
    args=extended_problem()
    with pytest.raises(ValueError,match='bounded'):old(*args,bounds=(-1.,10.),num_modes=4)
    result=capacity.solve_large_factor_chain_modes(*args,bounds=(-1.,10.),num_modes=4)
    np.testing.assert_allclose(result.eigenvalues,[1.,1.,4.,9.],rtol=1e-11,atol=1e-11)
    assert result.high_modes.shape==(257,4) and not result.production_qualified
    mapped=capacity.apply_large_mode_map(result,np.eye(257))
    np.testing.assert_allclose(mapped[:4].T@mapped[:4],np.eye(4),atol=1e-11,rtol=0.)
    with pytest.raises(ValueError):apply_mode_map(result,np.eye(257))
    (tmp_path/'extended-repeated.json').write_bytes(canonical(result))


def test_extended_truncated_cluster_rejected_and_scope_restored():
    with pytest.raises(ValueError,match='truncate'):
        capacity.solve_large_factor_chain_modes(*extended_problem(),bounds=(-1.,10.),num_modes=1)
    assert capacity.limits()==(256,8192,512) and not capacity.active()


def test_extended_kinetic_row_capacity():
    b=np.zeros((8193,3));b[:3]=np.eye(3)
    args=(np.diag([1.,2.,3.]),np.eye(3),np.zeros((3,3)),b,(0,1,2),())
    with pytest.raises(ValueError):old(*args,bounds=(-1.,10.),num_modes=3)
    result=capacity.solve_large_factor_chain_modes(*args,bounds=(-1.,10.),num_modes=3)
    np.testing.assert_allclose(result.eigenvalues,[1.,4.,9.],atol=1e-11,rtol=1e-11)


def test_capacity_nested_thread_isolation_and_exception_reset():
    assert capacity.limits()==(256,8192,512)
    with capacity.large_modal_capacity():
        assert capacity.limits()==(640,12288,640)
        with pytest.raises(ValueError):
            with capacity.large_modal_capacity():pass
        with ThreadPoolExecutor(max_workers=1) as pool:assert pool.submit(capacity.active).result() is False
    assert not capacity.active()
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):
        capacity.solve_large_factor_chain_modes(*extended_problem(),bounds=(-1.,10.),num_modes=4,cancellation_token=token)
    assert not capacity.active()


def test_larger_than_registered_capacity_rejected():
    with pytest.raises(ValueError,match='bounded'):
        capacity.solve_large_factor_chain_modes(np.eye(1),np.zeros((1,641)),np.eye(641),np.eye(641),
            tuple(range(641)),(),bounds=(-1.,10.),num_modes=1)
