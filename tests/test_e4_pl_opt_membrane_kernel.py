"""OPT membrane kernel: frozen paper oracle (Felippa 2003, Fig. 11) and structural checks."""
import numpy as np
import pytest

from anyfesolver.e4_pl_opt_membrane import (
    isotropic_plane_stress, optimal_beta0, opt_membrane_stiffness, opt_membrane_stiffness_parts)

FIG11_XY = np.array([[0.0, 0.0], [4.08, -3.44], [3.4, 1.14]])
FIG11_EIG = [52.913, 43.834, 26.434, 11.181, 1.8722, 0.64900]


def _fig11_k():
    return opt_membrane_stiffness_parts(FIG11_XY, isotropic_plane_stress(120.0, 0.25), 0.125)


def test_beta0_isotropic_closed_form():
    for nu in (0.0, 0.25, 0.3, 0.45):
        assert optimal_beta0(isotropic_plane_stress(7.0, nu)) == pytest.approx(0.5 * (1 - 4 * nu * nu), abs=1e-12)
    assert optimal_beta0(isotropic_plane_stress(1.0, 0.499)) == 0.01  # floor


def test_paper_figure_11_matrices_and_eigenvalues():
    kb, kh = _fig11_k()
    assert kb[0, :3] == pytest.approx([10.350, 0.95258, 7.7327], abs=6e-4)
    assert kh[0, :3] == pytest.approx([0.041538, -0.27977, -0.63728], abs=6e-6)
    assert kb[8, 8] == pytest.approx(14.512, abs=6e-4) and kh[8, 8] == pytest.approx(20.204, abs=6e-4)
    k = kb + kh
    assert k[8, 8] == pytest.approx(34.716, abs=6e-4)
    ev = np.sort(np.linalg.eigvalsh(k))[::-1]
    assert ev[:6] == pytest.approx(FIG11_EIG, rel=1e-4, abs=6e-5)
    assert np.all(np.abs(ev[6:]) < 1e-12 * ev[0])


def test_rank_three_zero_modes_rigid_translation_and_rotation():
    xy = np.array([[0.1, 0.05], [1.3, 0.4], [0.5, 1.1]])
    k = opt_membrane_stiffness(xy, isotropic_plane_stress(1e7, 0.3), 0.1)
    ev = np.linalg.eigvalsh(k)
    assert int(np.sum(np.abs(ev) < 1e-10 * ev[-1])) == 3
    for mode in (np.tile([1, 0, 0], 3), np.tile([0, 1, 0], 3),
                 np.concatenate([[-0.01 * y, 0.01 * x, 0.01] for x, y in xy])):
        assert np.linalg.norm(k @ mode) < 1e-10 * ev[-1] * np.linalg.norm(mode)


@pytest.mark.parametrize("alpha", [0.25, 1.0, 4.0, 16.0])
@pytest.mark.parametrize("diag", [0, 1])
def test_pure_bending_exact_for_any_aspect_ratio(alpha, diag):
    nu = 0.25
    em = isotropic_plane_stress(1.0, nu)
    a, b = alpha, 1.0
    p = np.array([[-a / 2, -b / 2], [a / 2, -b / 2], [a / 2, b / 2], [-a / 2, b / 2]])
    tris = [(0, 1, 2), (0, 2, 3)] if diag == 0 else [(0, 1, 3), (1, 2, 3)]
    e = 0.0
    for t in tris:
        u = np.concatenate([[-x * y, 0.5 * (x * x + nu * y * y), x] for x, y in p[list(t)]])
        e += 0.5 * u @ opt_membrane_stiffness(p[list(t)], em, 1.0) @ u
    assert e / (0.5 * (b**3 / 12) * a) == pytest.approx(1.0, abs=1e-12)


def test_refuses_clockwise_degenerate_and_bad_material():
    em = isotropic_plane_stress(1.0, 0.3)
    with pytest.raises(ValueError):
        opt_membrane_stiffness(FIG11_XY[::-1], em, 1.0)
    with pytest.raises(ValueError):
        opt_membrane_stiffness([[0, 0], [1, 0], [2, 0]], em, 1.0)
    with pytest.raises(ValueError):
        opt_membrane_stiffness(FIG11_XY, -em, 1.0)
    with pytest.raises(ValueError):
        opt_membrane_stiffness(FIG11_XY, em, 0.0)


def test_strain_recovery_exact_for_constant_strain_and_rigid_rotation():
    xy = np.array([[0.1, 0.05], [1.3, 0.4], [0.5, 1.1]])
    from anyfesolver.e4_pl_opt_membrane import opt_membrane_strain_operator
    bary = np.array([[0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.7, 0.2, 0.1]])
    ops = opt_membrane_strain_operator(xy, bary, 0.1)
    exx, eyy, gxy, w = 1e-3, -2e-4, 5e-4, 3e-4  # constant strain + rigid rotation w (theta = w)
    u = np.concatenate([[exx * x + 0.5 * gxy * y - w * y, eyy * y + 0.5 * gxy * x + w * x, w] for x, y in xy])
    for op in ops:
        assert op @ u == pytest.approx([exx, eyy, gxy], abs=1e-14)


def test_recovery_operator_is_energy_consistent_with_the_higher_order_stiffness():
    """K_h = (3/4) (beta0/e0^2) V sum_k Bh(mid_k)^T E Bh(mid_k): ties recovery to the stiffness (catches a dropped
    higher-order term, wrong e0 handling or swapped midpoint stations)."""
    from anyfesolver.e4_pl_opt_membrane import opt_membrane_strain_operator
    xy = np.array([[0.1, 0.05], [1.3, 0.4], [0.5, 1.1]])
    h, em = 0.1, isotropic_plane_stress(1e7, 0.3)
    e0 = 1.5
    kb, kh = opt_membrane_stiffness_parts(xy, em, h)
    mids = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    basic = opt_membrane_strain_operator(xy, np.array([[1 / 3, 1 / 3, 1 / 3]]), h, 0.0)[0]
    area = 0.5 * ((xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1]) - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1]))
    ops = opt_membrane_strain_operator(xy, mids, h, e0) - basic
    beta0 = optimal_beta0(em)
    expected = 0.75 * (beta0 / e0**2) * h * area * sum(op.T @ em @ op for op in ops)
    assert expected == pytest.approx(kh, rel=1e-12, abs=1e-12 * np.abs(kh).max())
