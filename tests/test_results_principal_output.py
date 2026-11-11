"""Finite principal-output and ambient NumPy-error boundaries."""
from importlib.util import find_spec
from threading import enumerate as threads
import numpy as np
import pytest
from anyfesolver_results import derive_field,DerivationError
from anyfesolver_results.formulas import principal_stresses,SHELL_LINEAR_COMPONENTS


@pytest.mark.parametrize('workers',[1,2])
@pytest.mark.parametrize('layout',['diagonal','off_diagonal'])
def test_actual_extreme_finite_tensor_never_returns_nonfinite_eigenvalues(workers,layout):
    tensor=np.zeros((4096 if workers==2 else 1,3,3))
    if layout=='diagonal':tensor[:,range(3),range(3)]=1e308
    else:tensor[:,0,1]=tensor[:,1,0]=1e308
    with np.errstate(over='ignore',invalid='ignore'),pytest.raises((ValueError,np.linalg.LinAlgError)):
        principal_stresses(tensor,workers=workers)
    assert not any(t.name.startswith('anyfesolver-principal') for t in threads())


@pytest.mark.parametrize('workers',[1,2])
def test_nonfinite_lapack_return_is_explicit_value_error_and_workers_join(monkeypatch,workers):
    tensor=np.zeros((4096 if workers==2 else 1,3,3))
    monkeypatch.setattr(np.linalg,'eigvalsh',lambda value:np.full(value.shape[:-1],np.nan))
    with pytest.raises(ValueError,match='eigenvalues must be finite'):principal_stresses(tensor,workers=workers)
    assert not any(t.name.startswith('anyfesolver-principal') for t in threads())


@pytest.mark.parametrize('workers',[1,2])
@pytest.mark.parametrize('numpy_error',['ignore','raise'])
def test_derived_and_reader_nonfinite_or_floating_error_is_chained(tmp_path,workers,numpy_error):
    if find_spec("anypost") is None:
        pytest.importorskip("anypost")
    from anypost import write_results,open_results,RecoveryError
    from test_derived_caching import shell_capsule
    count=4096 if workers==2 else 1
    inputs={name:np.zeros(count) for name in SHELL_LINEAR_COMPONENTS};inputs['membrane_xy'].fill(1e308)
    spec={'operation':'shell_principal_stress','surface':'top','rank':1,'inputs':{name:name for name in inputs}}
    with np.errstate(over=numpy_error,invalid='ignore'),pytest.raises(DerivationError) as error:derive_field(spec,inputs,workers=workers)
    assert isinstance(error.value.__cause__,FloatingPointError if numpy_error=='raise' else (ValueError,np.linalg.LinAlgError))
    c=shell_capsule(count);c.update(version=2,runtime='anysolver-results/2',combination='ordered_float64/1',patterns=(),coefficients=np.empty((1,0)))
    c['cases']=({'case_id':'a','status':'converged','certified':True,'portable':True,'displacement_portable':True,'route':'basis','active_indices':[]},)
    c['responses']={name:value[:,None] for name,value in inputs.items()}
    path=write_results(tmp_path/'overflow.h5',c)
    with np.errstate(over=numpy_error,invalid='ignore'),open_results(path,derivation_workers=workers) as reader:
        with pytest.raises(RecoveryError) as public:reader.read_field('a','stress','principal')
        assert isinstance(public.value.__cause__,DerivationError)
        assert isinstance(public.value.__cause__.__cause__,FloatingPointError if numpy_error=='raise' else (ValueError,np.linalg.LinAlgError))
        assert public.value.case_id=='a' and public.value.component_key=='principal'
    assert not any(t.name.startswith('anyfesolver-principal') for t in threads())
