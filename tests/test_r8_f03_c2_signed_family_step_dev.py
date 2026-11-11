"""Source controls for the one-case 32x8 signed-family discriminator."""
import json

import numpy as np
import pytest
from scipy import linalg, sparse
from pathlib import Path
import anyfesolver

from scripts.b3_e4_bl5_coherent_aggregation_dev import model
from scripts import r8_f03_c2_signed_family_step_dev as step


def test_current_worktree_product_import():
    assert Path(anyfesolver.__file__).resolve().is_relative_to((step.WORKTREE/'src').resolve())
    provenance = step.product_provenance()
    assert all(Path(item['path']).is_file() for item in provenance.values())
    assert all(len(item['sha256'])==64 for item in provenance.values())


def test_exact_c2_support_inventory():
    definition, policy = model(32,8,64,2)
    assert definition.lines[0].basis == 'p0'
    assert policy['span_halfcells'] == 4
    assert policy['support_membership'] == [list(range(4*i,4*i+4)) for i in range(16)]
    assert len(definition.fixed) == 59


def test_sparse_null_basis_and_positive_certificate():
    c = sparse.csr_matrix([[1.,0.,0.,0.],[0.,1.,0.,0.]])
    z, inventory = step.null_basis(c)
    assert z.shape == (4,2)
    assert inventory['basis_nnz'] <= 4
    np.testing.assert_allclose(c@z.toarray(),0.,atol=1e-14)
    assert np.linalg.norm(z.toarray(),ord=2)**2 <= (
        sparse.linalg.norm(z,ord=1)*sparse.linalg.norm(z,ord=np.inf))
    projected = z.T@sparse.diags([2.,3.,4.,5.])@z
    certificate = step.cholesky_certificate(projected)
    assert certificate['lower_bound'] > 0.
    with pytest.raises(linalg.LinAlgError):
        step.cholesky_certificate(-sparse.eye(2))


def test_guard_accounts_both_kinds_and_reserve(tmp_path,monkeypatch):
    for name,seconds in [('attempt1',60.),('source1',16.)]:
        path=tmp_path/name
        path.mkdir()
        (path/'receipt.json').write_text(json.dumps({'process_wall_seconds':seconds}))
    assert step.prior_process_wall(tmp_path)=={'numerical':60.,'source_control':16.}
    monkeypatch.setattr(step.subprocess,'Popen',lambda *a,**k: pytest.fail('launched without allowance'))
    with pytest.raises(ValueError,match='insufficient cumulative allowance'):
        step.guarded_run(tmp_path/'attempt2',timeout=45.)
    assert not (tmp_path/'attempt2').exists()


def test_guard_caps_child_duration(tmp_path):
    with pytest.raises(ValueError,match='child cap'):
        step.guarded_run(tmp_path/'attempt1',timeout=45.1)
    with pytest.raises(ValueError,match='child cap'):
        step.guarded_run(tmp_path/'source1',kind='source_control',timeout=30.1)


def test_guard_refuses_unmetered_attempt(tmp_path):
    (tmp_path/'attempt1').mkdir()
    with pytest.raises(ValueError,match='unmetered prior attempt'):
        step.prior_process_wall(tmp_path)
    with pytest.raises(ValueError,match='unmetered prior attempt'):
        step.guarded_run(tmp_path/'attempt2',timeout=1.)
    assert not (tmp_path/'attempt2').exists()


def test_launch_failure_gets_accounted_receipt(tmp_path,monkeypatch):
    def fail_launch(*args,**kwargs):
        raise OSError('deliberate launch failure')
    monkeypatch.setattr(step.subprocess,'Popen',fail_launch)
    receipt=step.guarded_run(tmp_path/'attempt1',timeout=1.)
    assert receipt['exit_code'] is None
    assert receipt['stop_reason']=='SUPERVISOR_ERROR'
    assert 'deliberate launch failure' in receipt['supervisor_error']
    assert receipt['process_wall_seconds'] >= 0.
    assert step.prior_process_wall(tmp_path)['numerical']==receipt['process_wall_seconds']
