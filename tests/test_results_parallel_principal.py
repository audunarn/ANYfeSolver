"""Independent old-expression oracle for station-parallel runtime scheduling."""
import threading
import numpy as np
import pytest
from anyfesolver_results import derive_field, DerivationError
from anyfesolver_results.formulas import principal_stresses, shell_surface_tensor


def original(tensor):
    values=np.asarray(tensor,dtype=float)
    if values.shape[-2:]!=(3,3) or not np.all(np.isfinite(values)):
        raise ValueError('principal stresses require finite 3 by 3 tensors')
    if not np.allclose(values,np.swapaxes(values,-1,-2),rtol=1e-12,atol=1e-12):
        raise ValueError('principal stresses require symmetric tensors')
    return np.linalg.eigvalsh(.5*(values+np.swapaxes(values,-1,-2)))[...,::-1].copy()


@pytest.mark.parametrize('shape',[(0,3,3),(3,3),(17,3,3),(4096,3,3),(2,2501,3,3)])
@pytest.mark.parametrize('order',['C','F'])
def test_station_workers_preserve_original_words_and_join(shape,order):
    rng=np.random.default_rng(7361)
    value=rng.normal(size=shape)
    value=np.array(value+np.swapaxes(value,-1,-2),order=order)
    if value.size:
        # Repeated eigenvalues and nearby principal ties, including signed zero.
        flat=value.reshape(-1,3,3)
        flat[0]=np.diag([1.,1.+np.spacing(1.),-0.])
    expected=original(value);before=value.tobytes(order='A');execution={}
    actual=principal_stresses(value,workers=2,execution=execution)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    assert value.tobytes(order='A')==before
    assert execution['workers']==(2 if expected.size>=4096*3 else 1)
    assert not any(t.name.startswith('anyfesolver-principal') for t in threading.enumerate())


def test_public_derived_parallel_field_matches_original_tensor():
    names=('membrane_xx','membrane_yy','membrane_xy','bending_xx','bending_yy','bending_xy','shear_xz','shear_yz')
    rng=np.random.default_rng(741)
    inputs={name:rng.normal(size=10001) for name in names}
    for surface in ('top','bottom'):
        expected=original(shell_surface_tensor(inputs,surface))
        for rank in (1,2,3):
            info={}
            actual=derive_field(dict(operation='shell_principal_stress',inputs={n:n for n in names},surface=surface,rank=rank),inputs,workers=2,execution=info)
            np.testing.assert_array_equal(actual.view(np.uint64),expected[:,rank-1].view(np.uint64))
            assert not actual.flags.writeable and info=={'workers':2,'tensors':10001}


@pytest.mark.parametrize('workers',[True,False,0,3,-1,None,2.0])
def test_invalid_worker_controls_are_typed_before_computation(workers):
    with pytest.raises(DerivationError,match='workers'):
        derive_field({}, {},workers=workers)


@pytest.mark.parametrize('kind',['nan','inf','asymmetric','wrong_shape'])
def test_parallel_refusals_preserve_original_validation(kind):
    value=np.zeros((5000,3,3))
    if kind=='nan':value[0,0,0]=np.nan
    elif kind=='inf':value[0,0,0]=np.inf
    elif kind=='asymmetric':value[0,0,1]=3.
    else:value=value[:,:2,:2]
    with pytest.raises(ValueError) as reference:original(value)
    with pytest.raises(ValueError,match=str(reference.value)):principal_stresses(value,workers=2)
    assert not any(t.name.startswith('anyfesolver-principal') for t in threading.enumerate())
