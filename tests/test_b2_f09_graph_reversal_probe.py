"""Focused private F09 graph reversal development checks; no admission claim."""

import runpy
from pathlib import Path
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from anyfesolver._b2_physical_flexibility_linear import LinearSection


PROBE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts"
                           / "probe_b3_ge_e4_pl_g0_f09_graph_reversal.py"))


def test_coupled_section_reorientation_preserves_graph_and_physical_work():
    result = PROBE["evaluate"]()
    assert result["passed"]
    assert result["matrix_error"] < result["diagnostic_limits"]["component"]
    assert result["unchanged_section_matrix_difference"] > 1e-4
    assert len(result["cases"]) == 2
    for case in result["cases"]:
        assert case["physical_work"] > 0
        assert case["unchanged_section_work_difference"] > 1e-4
        assert case["work_error"] < result["diagnostic_limits"]["work"]


def test_reversed_member_station_trace_and_work_pullback():
    forward = PROBE["members"]()[1]
    reverse = PROBE["members"](True)[1]
    graph_u = np.array([.02, -.03, .04, .005, -.006, .007,
                        -.08, .09, -.01, -.012, .014, -.016])
    swap = np.block([[np.zeros((6, 6)), np.eye(6)],
                     [np.eye(6), np.zeros((6, 6))]])
    stations = (0., .25, .5, .75, 1.)
    a = forward.trace_global(graph_u, stations=stations)
    b = reverse.trace_global(swap @ graph_u, stations=stations)
    np.testing.assert_allclose(a.reference_points_m, b.reference_points_m[::-1],
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(a.pose, b.pose[::-1], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(a.jacobian,
                               b.jacobian[::-1] @ swap,
                               rtol=2e-12, atol=2e-12)
    station_force = np.array([.3, -.2, .4, -.05, .07, .09])
    np.testing.assert_allclose(a.jacobian[2].T @ station_force,
                               swap @ b.jacobian[2].T @ station_force,
                               rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("radius_m", (.005, .02, .08))
def test_physical_circular_section_elbow_scaled_sparse_work(radius_m):
    area = math.pi*radius_m**2
    inertia = math.pi*radius_m**4/4
    section = LinearSection.isotropic(
        E=210e9, nu=.3, area=area, Iy=inertia, Iz=inertia,
        J=2*inertia, ky=.9, kz=.9)
    members = tuple(replace(member, section_id=f"circle-{radius_m}",
                            section=section) for member in PROBE["members"]())
    matrix = PROBE["assemble"](members)
    free = np.arange(6, 18)
    submatrix = matrix[free, :][:, free].tocsc()
    tip = (.1, -.08, .06, .01, -.012, .008)
    load = np.zeros(18)
    load[12:] = tip
    rhs = load[free]
    diagonal = submatrix.diagonal()
    assert np.isfinite(diagonal).all() and np.all(diagonal > 0)
    scaling = 1/np.sqrt(diagonal)
    scaled = diags(scaling) @ submatrix @ diags(scaling)
    solution = scaling*splu(scaled.tocsc()).solve(scaling*rhs)
    exact = np.asarray(PROBE["exact_solve"](submatrix.toarray(), rhs),
                       dtype=float)
    assert np.linalg.norm(solution-exact)/np.linalg.norm(exact) <= 1e-9
    complete = np.zeros(18)
    complete[free] = solution
    work = .5*float(load@complete)
    independent = PROBE["physical_work"](members, tip)
    assert abs(work-independent)/abs(independent) <= 1e-9
    residual = matrix@complete-load
    assert np.max(np.abs(residual[free].reshape(2, 6)[:, :3])) <= 1e-9*max(abs(v) for v in tip[:3])
    assert np.max(np.abs(residual[free].reshape(2, 6)[:, 3:])) <= 1e-9*max(abs(v) for v in tip[3:])


@pytest.mark.parametrize("radius_m", (.005, .02, .08))
@pytest.mark.parametrize("tip", (
    (.1, -.08, .06, .01, -.012, .008),
    (-.03, .05, -.07, -.004, .006, -.009)))
def test_coherent_generalized_coupling_preserves_scaled_graph_and_cut_work(
        radius_m, tip):
    area = math.pi*radius_m**2
    inertia = math.pi*radius_m**4/4
    scalar = LinearSection.isotropic(
        E=210e9, nu=.3, area=area, Iy=inertia, Iz=inertia,
        J=2*inertia, ky=.9, kz=.9)
    diagonal = np.diag(np.asarray(scalar.stiffness))
    root = np.diag(np.sqrt(diagonal))
    correlation = np.eye(6)
    for i, j, value in ((0, 4, .12), (1, 5, -.08),
                        (2, 3, .06), (4, 5, .04)):
        correlation[i, j] = correlation[j, i] = value
    assert np.linalg.eigvalsh(correlation)[0] > .7
    sectional = root@correlation@root
    forward_section = LinearSection.generalized(sectional)
    forward = tuple(replace(member, section_id="coherent-coupled",
                            section=forward_section)
                    for member in PROBE["members"]())
    reversed_original = PROBE["members"](True)
    axis_reversal = np.diag([-1., 1., -1., -1., 1., -1.])
    reverse = (
        replace(reversed_original[0], section_id="coherent-coupled",
                section=forward_section),
        replace(reversed_original[1], section_id="coherent-coupled-reversed",
                section=LinearSection.generalized(
                    axis_reversal@sectional@axis_reversal)))
    matrices = [PROBE["assemble"](members)
                for members in (forward, reverse)]
    matrix_error = np.linalg.norm((matrices[0]-matrices[1]).toarray()) / np.linalg.norm(matrices[0].toarray())
    assert matrix_error <= 1e-10

    free = np.arange(6, 18)
    load = np.zeros(18)
    load[12:] = tip
    references = [PROBE["physical_work"](members, tip)
                  for members in (forward, reverse)]
    assert references[0] > 0.
    assert abs(references[0]-references[1])/references[0] <= 1e-9
    for matrix, reference in zip(matrices, references):
        reduced = matrix[free, :][:, free].tocsc()
        rhs = load[free]
        pivot = reduced.diagonal()
        assert np.isfinite(pivot).all() and np.all(pivot > 0.)
        scale = 1/np.sqrt(pivot)
        scaled = diags(scale)@reduced@diags(scale)
        solution = scale*splu(scaled.tocsc()).solve(scale*rhs)
        exact = np.asarray(PROBE["exact_solve"](
            reduced.toarray(), rhs), dtype=float)
        assert np.linalg.norm(solution-exact)/np.linalg.norm(exact) <= 1e-9
        complete = np.zeros(18)
        complete[free] = solution
        work = .5*float(load@complete)
        assert abs(work-reference)/reference <= 1e-9
        residual = matrix@complete-load
        rows = residual[free].reshape(2, 6)
        assert np.max(np.abs(rows[:, :3])) <= 1e-9*max(abs(v) for v in tip[:3])
        assert np.max(np.abs(rows[:, 3:])) <= 1e-9*max(abs(v) for v in tip[3:])
