"""Algebraic contracts for the recovered principal display-axis candidate."""
import threading
import numpy as np
import pytest
from anysolver_results.formulas import principal_stresses
from anysolver_results.principals import principal_axes


def test_rotated_axes_preserve_canonical_values_residuals_and_input():
    angle=.37
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],
                       [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    tensor=rotation@np.diag([7.,2.,-3.])@rotation.T
    before=tensor.copy()
    result=principal_axes(tensor)
    np.testing.assert_array_equal(result.values,principal_stresses(tensor))
    np.testing.assert_allclose(tensor@result.axes.T,result.axes.T*result.values,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(result.unique,[True,True,True])
    np.testing.assert_array_equal(tensor,before)
    for array in (result.values,result.axes,result.unique):
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.flat[0]=0


@pytest.mark.parametrize('diagonal,unique', [([2.,2.,1.],[False,False,True]),
    ([0.,0.,0.],[False,False,False]),
    ([1.+32*np.finfo(float).eps,1.,0.],[False,False,True]),
    ([1.+128*np.finfo(float).eps,1.,0.],[True,True,True])])
def test_existing_display_degeneracy_convention(diagonal,unique):
    tensor=np.diag(diagonal); result=principal_axes(tensor)
    np.testing.assert_array_equal(result.values,principal_stresses(tensor))
    np.testing.assert_array_equal(result.unique,unique)
    assert np.isnan(result.axes[~result.unique]).all()
    assert np.isfinite(result.axes[result.unique]).all()


def test_basis_columns_are_local_axes_in_global_coordinates():
    basis=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    before=basis.copy(); tensor=np.diag([3.,2.,1.])
    result=principal_axes(tensor,basis=basis)
    np.testing.assert_array_equal(result.values,principal_stresses(tensor))
    np.testing.assert_array_equal(result.axes,[[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    np.testing.assert_array_equal(basis,before)


@pytest.mark.parametrize('basis', [np.zeros((3,3)),np.diag([-1.,1.,1.]),np.eye(2)])
def test_invalid_basis_is_refused(basis):
    with pytest.raises(ValueError):principal_axes(np.eye(3),basis=basis)


@pytest.mark.parametrize('tensor', [np.ones((2,2)),np.full((3,3),np.nan),
    np.array([[1.,2.,0.],[0.,1.,0.],[0.,0.,1.]]),[[1.],[2.,3.]]])
def test_canonical_tensor_admission_is_preserved(tensor):
    with pytest.raises(ValueError):principal_axes(tensor)


def test_noncontiguous_input_and_broadcast_basis_are_not_modified():
    storage=np.tile(np.diag([3.,2.,1.]),(6,1,1))
    tensor=storage[::2]; before=storage.copy()
    result=principal_axes(tensor,basis=np.eye(3))
    np.testing.assert_array_equal(result.values,principal_stresses(tensor))
    np.testing.assert_array_equal(storage,before)
    assert result.axes.shape==(3,3,3)


def test_cancellation_before_admission_and_between_blocks(monkeypatch):
    import anysolver_results.principals as module
    event=threading.Event();event.set()
    with pytest.raises(InterruptedError):principal_axes(np.eye(3),cancel_event=event)
    class BetweenBlocks:
        calls=0
        def is_set(self):
            self.calls+=1
            return self.calls>=3
    blocks=[]; actual=module.principal_stresses
    def counted(value):
        blocks.append(len(value)); return actual(value)
    monkeypatch.setattr(module,'principal_stresses',counted)
    tensor=np.broadcast_to(np.diag([3.,2.,1.]),(4097,3,3))
    with pytest.raises(InterruptedError):principal_axes(tensor,cancel_event=BetweenBlocks())
    assert blocks==[4096]


@pytest.mark.parametrize('huge', [False,True])
def test_working_cap_precedes_conversion_even_for_huge_virtual_shape(monkeypatch,huge):
    import anysolver_results.principals as module
    calls=[]
    class HugeVirtualArray(np.ndarray):
        @property
        def shape(self):calls.append(True);return (3,3)
    tensor=(np.broadcast_to(np.eye(3),(2**22,2**22,3,3)).view(HugeVirtualArray)
            if huge else np.zeros((20,3,3)))
    def forbid(*args,**kwargs):raise AssertionError('conversion ran before working admission')
    monkeypatch.setattr(module.np,'asarray',forbid)
    with pytest.raises(MemoryError,match='working byte cap'):
        principal_axes(tensor,working_bytes=8192)
    assert calls==[]


@pytest.mark.parametrize('dimensions', [(np.int64(1),3,3),(-1,3,3),(True,3,3),('1',3,3)])
def test_unknown_or_negative_shape_overrides_are_ignored(dimensions):
    calls=[]
    class VirtualArray(np.ndarray):
        @property
        def shape(self):calls.append(True);return dimensions
    tensor=np.diag([3.,2.,1.]).view(VirtualArray)
    result=principal_axes(tensor)
    np.testing.assert_array_equal(result.values,[3.,2.,1.])
    assert calls==[]


@pytest.mark.parametrize('dimensions', [(1,3,3),(2,3),(3,2),(3,3,1)])
def test_malformed_or_oversized_basis_refuses_before_conversion(monkeypatch,dimensions):
    import anysolver_results.principals as module
    calls=[]
    class VirtualBasis(np.ndarray):
        @property
        def shape(self):calls.append(True);return (3,3)
    tensor=np.eye(3);basis=np.ones(dimensions).view(VirtualBasis)
    def forbid(*args,**kwargs):raise AssertionError('conversion ran before basis shape admission')
    monkeypatch.setattr(module.np,'asarray',forbid)
    with pytest.raises(ValueError):principal_axes(tensor,basis=basis,working_bytes=8192)
    assert calls==[]


def test_valid_basis_ignores_shape_property_without_calling_it():
    calls=[]
    class VirtualBasis(np.ndarray):
        @property
        def shape(self):calls.append(True);raise AssertionError('shape override executed')
    result=principal_axes(np.diag([3.,2.,1.]),basis=np.eye(3).view(VirtualBasis))
    np.testing.assert_array_equal(result.values,[3.,2.,1.])
    assert calls==[]


def test_invalid_actual_tensor_shape_refuses_before_conversion(monkeypatch):
    import anysolver_results.principals as module
    calls=[]
    class VirtualArray(np.ndarray):
        @property
        def shape(self):calls.append(True);return (3,3)
    tensor=np.zeros((3,2)).view(VirtualArray)
    def forbid(*args,**kwargs):raise AssertionError('conversion ran before actual-shape admission')
    monkeypatch.setattr(module.np,'asarray',forbid)
    with pytest.raises(ValueError):principal_axes(tensor)
    assert calls==[]
