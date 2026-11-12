"""Stopping decisions must qualify the state actually returned to callers."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.plasticity import plane_stress_return_map


@pytest.mark.parametrize('max_iterations', [2, 30])
@pytest.mark.parametrize('compute_tangent', [False, True])
@pytest.mark.parametrize('axial_strain', [0.0015601375625240584, 0.0015601375625240587])
def test_near_tolerance_update_finishes_before_returning(max_iterations, compute_tangent, axial_strain):
    curve = DNVC208MaterialCurve(
        sigma_prop=320e6, sigma_yield=357e6, sigma_yield_2=363.3e6,
        eps_p_y1=.004, eps_p_y2=.015, K=740e6, n=.166,
    )
    # Frozen d04199 rejects this update: the stopping residual is
    # 9.999998620608125e-11, while the reconstructed residual exceeds1e-10.
    strain = np.array([[axial_strain, 0., 0.]])
    original = strain.copy()
    stress, tangent, plastic, alpha = plane_stress_return_map(
        strain, np.zeros_like(strain), np.zeros(1), 210e9, .3, curve,
        max_iterations=max_iterations, tolerance=1e-10,
        compute_tangent=compute_tangent,
    )
    np.testing.assert_array_equal(strain, original)
    assert np.all(np.isfinite(stress)) and np.all(np.isfinite(tangent))
    assert np.all(np.isfinite(plastic)) and alpha[0] > 0
    assert alpha[0] < curve.eps_p_y1
    # Independent 70-digit tensor invariant and first hardening branch,
    # evaluated from the returned state, not the implementation's residual.
    with localcontext() as context:
        context.prec = 70
        sx, sy, tau = map(Decimal.from_float, stress[0])
        flow = Decimal.from_float(curve.sigma_prop) + (
            (Decimal.from_float(curve.sigma_yield)-Decimal.from_float(curve.sigma_prop))
            / Decimal.from_float(curve.eps_p_y1) * Decimal.from_float(alpha[0])
        )
        phi2 = (sx*sx-sx*sy+sy*sy)/Decimal(3)+tau*tau
        residual = abs(phi2-flow*flow/Decimal(3))/(flow*flow)
        assert residual <= Decimal.from_float(1e-10)
