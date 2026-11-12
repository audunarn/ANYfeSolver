"""Numerical evaluation regressions; no new connection/material law."""
import numpy as np
import pytest
from anyfesolver import _ge_beam3_g3c_so3_numerics as stable
from anyfesolver import _ge_beam3_variational_shell as shell
from anyfesolver._ge_beam3_mixed_ad import Jet2
from test_ge_beam3_g3c_elementary_diagnostic import oracle


@pytest.mark.parametrize("angle", (.00045, .00062, .00125, .001288, .04))
def test_selected_log_factor_small_angle_value_and_both_derivatives(angle):
    cosine = float(np.cos(angle))
    expected = oracle("log", cosine, 60)
    actual = stable._log_factor(Jet2.variable(cosine, 0, 1))
    np.testing.assert_allclose((actual.value, actual.gradient[0], actual.hessian[0, 0]),
        expected, rtol=2e-14, atol=2e-15)


def test_material_deformation_dispatch_does_not_replace_historical_globals(monkeypatch):
    reference = np.array([[0., 0., 0.], [.1, 0., 0.], [.1, .1, 0.], [0., .1, 0.]])
    u = np.zeros(24)
    u.reshape(-1, 6)[:, 3:] = [.00062, .00125, .00045]
    historical = shell.deformation(reference, u, stable_translation=True)
    np.testing.assert_array_equal(historical[0], shell.deformation(reference, u,
        stable_translation=True, stable_rotation=False)[0])
    original_exp, original_log = shell.so3_exp, shell.so3_log
    def forbidden(*args):raise AssertionError("historical rotation dispatch")
    monkeypatch.setattr(shell, "so3_exp", forbidden)
    monkeypatch.setattr(shell, "so3_log", forbidden)
    selected = shell.deformation(reference, u, stable_translation=True, stable_rotation=True)
    np.testing.assert_allclose(selected[0].reshape(-1, 6)[:, 3:],
        np.tile(u[3:6], (4, 1)), rtol=2e-15, atol=2e-18)
    with pytest.raises(AssertionError, match="historical rotation dispatch"):
        shell.deformation(reference, u, stable_translation=True)
    assert stable.so3_exp is not original_exp and stable.so3_log is not original_log
    with pytest.raises(ValueError, match="typed shell rotation"):
        shell.deformation(reference, u, stable_rotation=1)


def test_selected_spatial_chart_preserves_historical_dispatch(monkeypatch):
    from anyfesolver import _stiffened_shell_algorithmic_pullback as pullback
    u = np.zeros(24);u[3:6] = [.00062, .00125, .00045]
    force = np.arange(24.)
    def forbidden(*args):raise AssertionError("historical chart dispatch")
    monkeypatch.setattr(pullback, "_exp_terms", forbidden)
    residual, tangent = pullback.spatial_response(u, force, np.eye(24),
        np.eye(24), np.zeros((24, 24, 24)), stable_rotation=True)
    assert np.isfinite(residual).all() and np.isfinite(tangent).all()
    with pytest.raises(AssertionError, match="historical chart dispatch"):
        pullback.spatial_response(u, force, np.eye(24), np.eye(24), np.zeros((24, 24, 24)))


def test_pre_stable_material_checkpoint_rejected_without_commit():
    from pathlib import Path
    from anyfesolver import stiffened_shell as ss
    from test_stiffened_shell_ge_distributing_material import definition
    owner = ss.create_analysis(definition=definition())
    try:
        assert owner.capabilities["shell_rotation_numerics"] == stable.POLICY
        assert owner._identity != "97a060696903febcf86c1741ccb2514e24c7d83c086dfb3e98208d82df4fe76e"
        before = owner._generation
        legacy = Path(__file__).with_name("data") / "stiffened_shell_material_pre_stable_rotation.bin"
        with pytest.raises(ValueError, match="incompatible experimental binary checkpoint"):
            owner.restore(legacy.read_bytes())
        assert owner._generation == before
    finally:owner.close()
