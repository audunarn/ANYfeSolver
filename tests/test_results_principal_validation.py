"""Exact old-predicate parity and typed/joined principal failure boundaries."""
from threading import Barrier,Event,current_thread,enumerate as threads
from concurrent.futures import CancelledError
import numpy as np
import pytest
from anyfesolver_results import derive_field,DerivationError
from anyfesolver_results.formulas import principal_stresses


def original(tensor):
    v=np.asarray(tensor,dtype=float)
    if v.shape[-2:]!=(3,3) or not np.all(np.isfinite(v)):raise ValueError('finite tensor')
    if not np.allclose(v,np.swapaxes(v,-1,-2),rtol=1e-12,atol=1e-12):raise ValueError('symmetry')
    return np.linalg.eigvalsh(.5*(v+np.swapaxes(v,-1,-2)))[...,::-1].copy()


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('off_diagonal',[1.,np.nextafter(1.,np.inf),np.nextafter(1.+2e-12,0.),1.+2e-12,np.nextafter(1.+2e-12,np.inf),1.+3e-12])
def test_near_tolerant_boundary_matches_original_predicate_and_words(off_diagonal,reverse):
    tensor=np.diag([2.,3.,4.]);tensor[0,1]=1.;tensor[1,0]=off_diagonal
    if reverse:tensor=tensor.T.copy()
    accepted=np.allclose(tensor,tensor.T,rtol=1e-12,atol=1e-12)
    if accepted:np.testing.assert_array_equal(principal_stresses(tensor).view('u8'),original(tensor).view('u8'))
    else:
        with pytest.raises(ValueError,match='symmetric'):principal_stresses(tensor)


@pytest.mark.parametrize('count',[4095,4096])
def test_exact_symmetric_signed_zero_shortcut_preserves_words_boundary_and_metadata(count,monkeypatch):
    tensor=np.broadcast_to(np.diag([1.,np.nextafter(1.,np.inf),-0.]),(count,3,3)).copy();tensor[:,0,1]=-0.;tensor[:,1,0]=0.
    expected=original(tensor);execution={}
    monkeypatch.setattr(np,'allclose',lambda *_args,**_kwargs:pytest.fail('exact finite symmetry should short circuit tolerant predicate'))
    actual=principal_stresses(tensor,workers=2,execution=execution)
    np.testing.assert_array_equal(actual.view('u8'),expected.view('u8'))
    assert execution=={'workers':2 if count==4096 else 1,'tensors':count}
    assert not any(t.name.startswith('anyfesolver-principal') for t in threads())


@pytest.mark.parametrize('value',[np.inf,np.nan,-np.inf])
def test_nonfinite_still_refuses_before_either_symmetry_predicate(value,monkeypatch):
    tensor=np.eye(3);tensor[0,0]=value
    monkeypatch.setattr(np,'array_equal',lambda *_args,**_kwargs:pytest.fail('finite check must remain first'))
    with pytest.raises(ValueError,match='finite'):principal_stresses(tensor)


def spec_inputs(count):
    names=('membrane_xx','membrane_yy','membrane_xy','bending_xx','bending_yy','bending_xy','shear_xz','shear_yz')
    return {'operation':'shell_principal_stress','surface':'top','rank':1,'inputs':{k:k for k in names}},{k:np.zeros(count) for k in names}


def test_worker_linalg_failure_joins_already_running_peer_and_is_typed(monkeypatch):
    spec,inputs=spec_inputs(4096);barrier=Barrier(2,timeout=2);peer_finished=Event();original_eig=np.linalg.eigvalsh
    def eigen(tensor):
        barrier.wait()
        if current_thread().name.endswith('_0'):raise np.linalg.LinAlgError('injected numerical failure')
        try:return original_eig(tensor)
        finally:peer_finished.set()
    monkeypatch.setattr(np.linalg,'eigvalsh',eigen);execution={}
    with pytest.raises(DerivationError,match='principal') as error:derive_field(spec,inputs,workers=2,execution=execution)
    assert isinstance(error.value.__cause__,np.linalg.LinAlgError) and peer_finished.is_set()
    assert execution=={'workers':2,'tensors':4096}
    assert not any(t.name.startswith('anyfesolver-principal') for t in threads())


@pytest.mark.parametrize('workers',[1,2])
def test_finite_input_overflow_is_typed_and_refusal_metadata_stays_serial(workers):
    spec,inputs=spec_inputs(4096);inputs['membrane_xx'].fill(1.7e308);inputs['bending_xx'].fill(1.7e308);execution={}
    with np.errstate(over='ignore'),pytest.raises(DerivationError) as error:derive_field(spec,inputs,workers=workers,execution=execution)
    assert isinstance(error.value.__cause__,ValueError) and execution=={'workers':1}
    assert not any(t.name.startswith('anyfesolver-principal') for t in threads())


@pytest.mark.parametrize('kind',[MemoryError,CancelledError,TypeError])
def test_resource_cancellation_programmer_errors_keep_original_type(monkeypatch,kind):
    spec,inputs=spec_inputs(1)
    def fail(*_args,**_kwargs):raise kind('original')
    monkeypatch.setattr('anyfesolver_results.registry.formulas.shell_principal_stresses',fail)
    with pytest.raises(kind,match='original'):derive_field(spec,inputs)
