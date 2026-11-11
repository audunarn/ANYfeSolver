"""Model dispatch must count distributed nodal/internal load work only once."""
import numpy as np
import pytest
from anyfesolver import _ge_beam3_native_generalized_modal as dense
from anyfesolver import _ge_beam3_native_generalized_paired_modal as paired
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern,nodal_force_vector
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken,SolveCancelled
from test_ge_beam3_native_analysis import analysis
from test_ge_beam3_native_generalized import problem,pattern


@pytest.mark.parametrize('curved',(False,True))
def test_actual_conservative_distributed_equilibrium_is_not_double_loaded(curved,tmp_path):
    made=analysis(problem(curved,False)[0]);load=DistributedPattern(LinePattern(((1,.02,-.007,.004),)),())
    run=made.solve_distributed(load,steps=1);assert run.status=='completed'
    state=made._decode(run.checkpoint,run.checkpoint_sha256)[-1];before=canonical(state)
    # Reproduce the old routing defect without changing any mechanics.
    doubled=nodal_force_vector(made.model,state['load_point'].effective(made.model).line)
    assert np.linalg.norm(doubled)>0
    with pytest.raises(ValueError,match='free conservative equilibrium'):
        dense.prepare(made.model,state['states'],state['displacements'],made._inertias,doubled)
    corrected=made.current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256)
    reference=dense.solve_modes(made.model,state['states'],state['displacements'],made._inertias,np.zeros(18))
    assert canonical(corrected)==canonical(reference)
    accurate=made.current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256,bounds=(-100.,1e6))
    direct=paired.solve_modes(made.model,state['states'],state['displacements'],made._inertias,np.zeros(18),
        bounds=(-100.,1e6))
    assert canonical(accurate)==canonical(direct)
    np.testing.assert_allclose(corrected[1].eigenvalues,accurate[1].eigenvalues,atol=1e-11,rtol=1e-11)
    assert canonical(state)==before and np.all(accurate[1].eigenvalues>0.)
    (tmp_path/'generalized-force-modes.json').write_bytes(canonical(dict(dense=corrected,paired=accurate)))
    (tmp_path/'generalized-force-checkpoint.json').write_bytes(run.checkpoint)


@pytest.mark.parametrize('bounds',(None,(-100.,1e6)))
def test_spatial_couples_never_gain_conservative_modal_authority(bounds):
    made=analysis(problem(True,False)[0]);run=made.solve_distributed(pattern(),steps=1)
    assert run.status=='completed'
    with pytest.raises(ValueError,match='nonconservative'):
        made.current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256,bounds=bounds)


def test_cancelled_generalized_request_does_not_decode(monkeypatch):
    made=analysis(problem(False,False)[0]);token=CancellationToken();token.cancel()
    monkeypatch.setattr(made,'_decode',lambda *a,**kw:pytest.fail('cancelled checkpoint decoded'))
    with pytest.raises(SolveCancelled):
        made.current_modes(b'invalid',expected_sha256='0'*64,cancellation_token=token)
