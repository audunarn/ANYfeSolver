"""Small preparatory checks; these do not execute or qualify a P5 candidate."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "docs/reference_cases/ge_beam3_curved_p5_algebra_probe.py"
SPEC = importlib.util.spec_from_file_location("p5_algebra_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def reference(height=0.4, shift=0.0, twist=0.0):
    coords = np.array(((-1., 0., 0.), (shift, height, 0.), (1., 0., 0.)))
    frames = []
    for index, xi in enumerate((-1., 0., 1.)):
        tangent = np.array((xi-.5, -2*xi, xi+.5)) @ coords
        tangent /= np.linalg.norm(tangent)
        second = np.array((0., 0., 1.))
        frame = np.column_stack((tangent, second, np.cross(tangent, second)))
        frames.append(frame @ probe.rotation((twist*index, 0., 0.)))
    return CurvedBeam3ReferenceGeometry(coords, frames)


def section():
    factor = np.array(((2., .1, 0., .2, -.1, 0.), (0., 3., .2, 0., .3, .1),
                       (0., 0., 4., .1, 0., .2), (0., 0., 0., 1., .1, .2),
                       (0., 0., 0., 0., 1.5, .1), (0., 0., 0., 0., 0., 2.)))
    return factor.T @ factor


def perturbed(ref):
    positions = ref.coordinates + np.array(((.01, -.02, .03), (.04, .03, -.02),
                                            (-.01, .04, .02)))
    vertices = np.array([probe.rotation(v) @ frame for v, frame in zip(
        ((.04, -.02, .06), (-.03, .05, .02), (.02, .03, -.04)), ref.nodal_triads)])
    cells = np.array([probe.rotation((.02, -.01, .03)), probe.rotation((-.01, .03, .01))])
    return positions, vertices, cells


def assert_scaled_close(actual, expected, tolerance=1e-11):
    error = np.linalg.norm(np.asarray(actual)-np.asarray(expected))
    assert error <= tolerance * max(1., np.linalg.norm(expected)), error


def test_exact_force_constraint_rank_detects_extra_thin_limit_constraints():
    rows = probe.rank_diagnostic()["rows"]
    assert [(row["control_rank"], row["lift_rank"]) for row in rows] == [(8, 6), (9, 6), (9, 6)]
    assert all(row["lift_nullity"] == 9 for row in rows)
    assert rows[1]["control_nullity"] == 6


def test_exact_rank_is_invariant_under_rational_coordinate_changes():
    coords = np.array(((-1, 0, 0), (0, 1, 0), (1, 0, 0)))
    transform = np.array(((0, -1, 0), (0, 0, 1), (-1, 0, 0)))
    for made in (coords, 17*coords @ transform.T + (3, 8, -2), coords[::-1]):
        assert probe.force_constraint_rank(made, objective_lift=True) == 6
        assert probe.force_constraint_rank(made, objective_lift=False) == 9


@pytest.mark.parametrize("geometry", [(0., 0., 0.), (.4, 0., 0.), (.6, .2, .15)])
def test_reference_hessian_rigid_modes_rank_inertia_and_schur(geometry):
    ref = reference(*geometry)
    full, reduced, condensed = probe.reference_hessians(ref, section())
    eliminated_moments = full[:24, :24] - full[:24, 24:] @ np.linalg.solve(
        full[24:, 24:], full[24:, :24])
    assert_scaled_close(reduced, eliminated_moments)
    assert_scaled_close(condensed, condensed.T)
    assert_scaled_close(condensed @ probe.rigid_matrix(ref.coordinates), np.zeros((18, 6)))
    assert_scaled_close(reduced @ probe.rigid_matrix(ref.coordinates, True), np.zeros((24, 6)))
    assert np.linalg.matrix_rank(reduced, tol=1e-10) == 18
    assert np.linalg.matrix_rank(condensed, tol=1e-10) == 12
    for matrix, positive, negative, zero in ((full, 18, 12, 6), (reduced, 18, 0, 6),
                                            (condensed, 12, 0, 6)):
        values = np.linalg.eigvalsh(matrix)
        assert np.count_nonzero(values > 1e-10) == positive
        assert np.count_nonzero(values < -1e-10) == negative
        assert np.count_nonzero(abs(values) <= 1e-10) == zero


@pytest.mark.parametrize("geometry", [(0., 0., 0.), (.4, 0., 0.), (.6, .2, .15)])
def test_reference_state_and_arbitrary_common_rigid_motion_have_zero_energy(geometry):
    ref = reference(*geometry)
    for vector in ((0., 0., 0.), (2.1, -1.8, 2.5), (5., 4., -3.)):
        rotation = probe.rotation(vector)
        coords = ref.coordinates @ rotation.T + np.array((2., -7., 5.))
        frames = np.einsum("ij,njk->nik", rotation, ref.nodal_triads)
        cells = np.array((rotation, rotation))
        energy, moments = probe.reduced_energy(ref, section(), coords, frames, cells)
        assert energy < 1e-24
        assert_scaled_close(moments, np.zeros((2, 2, 3)), 1e-12)
        for cell in (0, 1):
            for fraction in (0., .25, .5, .75, 1.):
                point = probe.lift_position(ref, coords, cells[cell], cell, fraction)
                assert_scaled_close(point, rotation @ ref.position(cell-1+fraction)+(2., -7., 5.))


def test_finite_energy_and_lift_are_objective_in_deformed_configuration():
    ref = reference(.6, .2, .15)
    x, q, u = perturbed(ref)
    before, moments = probe.reduced_energy(ref, section(), x, q, u)
    g = probe.rotation((2.1, -1.8, 2.5))
    x2 = x @ g.T + (8., -3., 1.)
    q2 = np.einsum("ij,njk->nik", g, q)
    u2 = np.einsum("ij,njk->nik", g, u)
    after, moments2 = probe.reduced_energy(ref, section(), x2, q2, u2)
    assert_scaled_close(after, before)
    assert_scaled_close(moments2, moments)
    for cell in (0, 1):
        expected = g @ probe.lift_position(ref, x, u[cell], cell, .35)+(8., -3., 1.)
        assert_scaled_close(probe.lift_position(ref, x2, u2[cell], cell, .35), expected)


def test_reference_coordinate_reexpression_preserves_finite_energy():
    ref = reference(.6, .2, .15)
    x, q, u = perturbed(ref)
    g = probe.rotation((.4, .3, -.2))
    moved = ref.rigidly_transformed(g, (3., -1., 2.))
    x2 = x @ g.T+(3., -1., 2.)
    q2 = np.einsum("ij,njk->nik", g, q)
    u2 = np.array([g @ value @ g.T for value in u])
    assert_scaled_close(probe.reduced_energy(moved, section(), x2, q2, u2)[0],
                        probe.reduced_energy(ref, section(), x, q, u)[0])


def test_exact_moment_elimination_matches_station_functional_and_stationarity():
    ref = reference(.6, .2, .15)
    x, q, u = perturbed(ref)
    reduced, moments = probe.reduced_energy(ref, section(), x, q, u)
    mixed = probe.mixed_energy(ref, section(), x, q, u, moments)
    assert_scaled_close(mixed, reduced, 1e-13)
    # Complementary functional is a strict quadratic maximum in moments.
    delta = np.arange(12, dtype=float).reshape(2, 2, 3)/11
    plus = probe.mixed_energy(ref, section(), x, q, u, moments+.1*delta)
    minus = probe.mixed_energy(ref, section(), x, q, u, moments-.1*delta)
    assert_scaled_close(plus, minus, 1e-13)
    assert plus < mixed


def test_reversal_preserves_coupled_work_and_transforms_moments_and_hessian():
    ref = reference(.6, .2, .15)
    x, q, u = perturbed(ref)
    before, moments = probe.reduced_energy(ref, section(), x, q, u)
    reversed_ref = ref.reversed()
    c2 = probe.T6 @ section() @ probe.T6.T
    q2 = q[::-1] @ probe.S
    after, reversed_moments = probe.reduced_energy(reversed_ref, c2, x[::-1], q2, u[::-1])
    assert_scaled_close(after, before)
    assert_scaled_close(reversed_moments, np.einsum("ij,nej->nei", probe.T3, moments[::-1, ::-1]))
    k = probe.reference_hessians(ref, section())[2]
    k2 = probe.reference_hessians(reversed_ref, c2)[2]
    permutation = np.r_[12:18, 6:12, 0:6]
    assert_scaled_close(k2, k[np.ix_(permutation, permutation)])


def test_straight_limit_agrees_with_frozen_p3_uneliminated_functional():
    # Comparison target is imported only in this test, never by the probe.
    from anyfesolver.ge_beam3_mixed_element import GeometricallyExactBeam3D3NElement
    ref = reference(0.)
    x, q, u = perturbed(ref)
    _, moments = probe.reduced_energy(ref, section(), x, q, u)
    physical_cells = np.array([value @ ref.nodal_triads[cell] for cell, value in enumerate(u)])
    old = GeometricallyExactBeam3D3NElement._potential(
        x, q, 1., section(), physical_cells, moments, include_external=False)
    made = probe.mixed_energy(ref, section(), x, q, u, moments)
    assert_scaled_close(made, old.value, 1e-13)


def test_straight_reference_full_and_condensed_hessians_agree_with_p3():
    from anyfesolver.ge_beam3_mixed_element import GeometricallyExactBeam3D3NElement
    ref = reference(0.)
    physical_cells = ref.nodal_triads[:2]
    old = GeometricallyExactBeam3D3NElement._potential(
        ref.coordinates, ref.nodal_triads, 1., section(), physical_cells,
        np.zeros((2, 2, 3)), include_external=True)
    # P3 packs each local rotation with its moments; the probe groups rotations
    # before moments. Both perturb rotation matrices in the spatial chart.
    order = np.r_[0:18, 18:21, 27:30, 21:27, 30:36]
    old_full = old.hessian[np.ix_(order, order)]
    full, _, condensed = probe.reference_hessians(ref, section())
    assert_scaled_close(full, old_full)
    old_condensed = old.hessian[:18, :18]-old.hessian[:18, 18:] @ np.linalg.solve(
        old.hessian[18:, 18:], old.hessian[18:, :18])
    assert_scaled_close(condensed, old_condensed)


def test_reference_hessian_is_covariant_under_spatial_basis_change():
    ref = reference(.6, .2, .15)
    g = probe.rotation((.4, .3, -.2))
    moved = ref.rigidly_transformed(g, (3., -1., 2.))
    change = np.kron(np.eye(6), g)
    original = probe.reference_hessians(ref, section())[2]
    transformed = probe.reference_hessians(moved, section())[2]
    assert_scaled_close(transformed, change @ original @ change.T)


def test_reference_hessian_matches_second_variation_of_finite_functional():
    ref = reference(.4, .1, .15)
    _, hessian, _ = probe.reference_hessians(ref, section())
    direction = np.sin(np.arange(24)+.3)/10
    expected = direction @ hessian @ direction
    h = 1e-4
    energies = []
    for sign in (-1, 1):
        increment = sign*h*direction
        x = ref.coordinates+increment[:18].reshape(3, 6)[:, :3]
        q = np.array([probe.rotation(increment[n*6+3:n*6+6]) @ ref.nodal_triads[n]
                      for n in range(3)])
        u = np.array([probe.rotation(increment[18:21]), probe.rotation(increment[21:24])])
        energies.append(probe.reduced_energy(ref, section(), x, q, u)[0])
    assert_scaled_close(sum(energies)/h**2, expected, 1e-7)


def test_quadrature_comparison_is_diagnostic_and_serialization_is_deterministic():
    ref = reference(.6, .2, .15)
    for cell in (0, 1):
        before, after = probe.metrics(ref, section(), cell, 24), probe.metrics(ref, section(), cell, 48)
        for field in ("force", "compliance", "coupling"):
            assert_scaled_close(getattr(before, field), getattr(after, field))
    record = probe.rank_diagnostic()
    assert record["independent_review"] == "PENDING"
    assert record["formal_execution_authorized"] is False
    assert probe.canonical_bytes(record) == probe.canonical_bytes(probe.rank_diagnostic())
    with pytest.raises(ValueError):
        probe.canonical_bytes({"value": float("nan")})


@pytest.mark.parametrize("mutation", ["negative", "asymmetric", "nonfinite", "shape"])
def test_invalid_sections_are_rejected(mutation):
    c = section()
    if mutation == "negative":
        c[0, 0] = -1
    elif mutation == "asymmetric":
        c[0, 1] += 1
    elif mutation == "nonfinite":
        c[0, 0] = np.nan
    else:
        c = c[:5, :5]
    with pytest.raises(ValueError):
        probe.metrics(reference(), c, 0)


def test_relative_rotation_cutoff_and_reflection_fail_closed():
    for matrix in (probe.rotation((.91*np.pi, 0., 0.)), np.diag((-1., 1., 1.))):
        with pytest.raises(ValueError):
            probe.log_rotation(matrix)


def test_preparation_record_binds_inputs_and_recomputes_exact_rank_diagnostic():
    path = ROOT / "docs/reference_cases/ge_beam3_curved_p5_preparation.json"
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    record = json.loads(raw)
    assert raw == probe.canonical_bytes(record)
    assert record["diagnostic"] == probe.rank_diagnostic()
    assert record["evidence_scope"] == "AUTHOR_PREPARATORY_DIAGNOSTIC_NOT_INDEPENDENT_QUALIFICATION"
    assert record["production_boundary"] == {
        "candidate_freeze_authorized_by_this_record": False,
        "default_change_authorized": False,
        "formal_execution_authorized_by_this_record": False,
        "independent_review_complete": False,
        "production_activation_authorized": False,
        "public_selector_authorized": False,
        "publication_authorized": False,
    }
    assert len({row["path"] for row in record["inputs"]}) == len(record["inputs"]) == 4
    for row in record["inputs"]:
        content = (ROOT / row["path"]).read_bytes().replace(b"\r\n", b"\n")
        assert b"\r" not in content
        assert len(content) == row["bytes"]
        assert hashlib.sha256(content).hexdigest().upper() == row["sha256"]
    source = Path(record["source"]["path"])
    if source.is_file():
        content = source.read_bytes()
        assert len(content) == record["source"]["bytes"]
        assert hashlib.sha256(content).hexdigest().upper() == record["source"]["sha256"]
