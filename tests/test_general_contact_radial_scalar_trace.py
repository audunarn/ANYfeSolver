"""Observers must preserve the pure law and restore bindings on failure."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from anyfesolver.general_contact import semismooth_path as driver
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.d1_radial_scalar_trace import radial_solver_trace


def test_scalar_trace_preserves_values_inputs_and_restores_after_exception(tmp_path):
    original = driver.complementarity
    preparation = GeneralS3PatchManager.__init__
    state = SimpleNamespace(normalized_gap=np.array([-.01,.02]), reference_support=np.ones(2))
    z = np.array([1.,0.]); c=np.array([10.,10.]); gradients=np.eye(2)
    expected = original(state,z,c,gradients)
    path=tmp_path/'trace.jsonl'
    with pytest.raises(RuntimeError, match='observer fixture'):
        with radial_solver_trace(path):
            actual=driver.complementarity(state,z,c,gradients)
            for a,b in zip(actual,expected,strict=True):
                np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(z,[1.,0.])
            raise RuntimeError('observer fixture')
    assert driver.complementarity is original
    assert GeneralS3PatchManager.__init__ is preparation
    records=[json.loads(row) for row in path.read_text().splitlines()]
    end=next(row for row in records if row['event']=='complementarity_end')
    assert end['active_modes']==1
    assert end['complementarity_l1_n']==.1
    assert records[-1]['event']=='trace_end'
    with pytest.raises(FileExistsError):
        with radial_solver_trace(path):
            pass
