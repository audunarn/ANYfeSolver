"""Opt-in raw nodal-force work at the artificial Q4/S3 target seam."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import ContactPathNodalForce, solve_static_nonlinear
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.q4_s3_overlap_derivatives import UnsupportedQ4S3Branch
from anyfesolver.general_contact import s3_patch_augmented
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from scripts.general_contact_d1_coupled_refinement import build_case


def _case():
    model, definition, settings, _program = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    saved = json.loads((Path(__file__).parent / "fixtures" /
                        "general_contact_q4_s3_neighbor_value_gate_pose.json").read_text())
    source = np.asarray(saved["source"])
    target = np.asarray(saved["target"])
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    _, _, transform, _, _, _ = build_constraint_transformation(
        stiffness, np.zeros(model.mesh.dof_manager.total_dofs), model,
    )
    owner.restrict_tangent_to_reduced_variations(transform)
    base = owner.law.geometry.evaluate(source, target)
    z = np.asarray(saved["multipliers"])*base.reference_support
    return owner, source, target, base, z


def test_raw_nodal_force_exact_seam_has_supported_gap_rows_and_tangent():
    owner, source, target, base, z = _case()
    supported = np.flatnonzero(base.reference_support > 0.0)
    assert len(supported) > 1
    z[supported[0]] = 0.0
    work = owner.law.nodal_force_linearization(source, target, z, tangent=True)
    first_order = owner.law.nodal_force_linearization(
        source, target, z, tangent=False,
    )
    assert work.energy_evaluations <= owner.law.geometry.maximum_energy_evaluations
    np.testing.assert_allclose(work.state.normalized_gap, base.normalized_gap,
                               rtol=0.0, atol=0.0)
    np.testing.assert_allclose(work.residual, -z @ work.gap_gradient,
                               rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(first_order.residual, work.residual,
                               rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(first_order.gap_gradient, work.gap_gradient,
                               rtol=1e-6, atol=1e-10)
    assert np.linalg.norm(work.gap_gradient[supported[0]]) > 0.0
    np.testing.assert_allclose(work.tangent, work.tangent.T,
                               rtol=1e-6, atol=1e-10)

    coordinates = np.concatenate((source, target)).reshape(-1)
    for coordinate in (3*1, 3*2+2, 3*(len(source)+1)):
        h = 7.5e-9
        observations = []
        for sign in (-1.0, 1.0):
            displaced = coordinates.copy()
            displaced[coordinate] += sign*h
            observations.append(owner.law.geometry.evaluate(
                displaced[:3*len(source)].reshape(-1, 3),
                displaced[3*len(source):].reshape(-1, 3),
            ).normalized_gap)
        observed = (observations[1]-observations[0])/(2*h)
        np.testing.assert_allclose(work.gap_gradient[supported, coordinate],
                                   observed[supported], rtol=1e-5, atol=1e-8)

    h = 1.0e-7
    shifted_minus = source.copy()
    shifted_plus = source.copy()
    shifted_minus[1, 0] -= h
    shifted_plus[1, 0] += h
    minus = owner.law.nodal_force_linearization(
        shifted_minus, target, z, tangent=False,
    )
    plus = owner.law.nodal_force_linearization(
        shifted_plus, target, z, tangent=False,
    )
    active = owner.law.tangent_active_coordinates
    observed_force_change = (plus.residual-minus.residual)/(2*h)
    predicted_force_change = work.tangent[:, 3]
    assert (np.linalg.norm((observed_force_change-predicted_force_change)[active])
            / np.linalg.norm(predicted_force_change[active]) < 1.0e-5)


def test_raw_nodal_force_seam_rejects_missing_opt_in_and_accepts_signed_trial(
        monkeypatch):
    owner, source, target, base, z = _case()
    supported = np.flatnonzero(base.reference_support > 0.0)
    saved_tie = owner.law.shared_edge_tie
    owner.law.shared_edge_tie = None
    with pytest.raises(UnsupportedQ4S3Branch):
        owner.law.nodal_force_linearization(source, target, z, tangent=False)
    owner.law.shared_edge_tie = saved_tie
    z[supported[0]] = -1.0e-4
    z[supported[1]] = 0.0
    tiny_wedge_calls = []
    original_verifier = s3_patch_augmented.verify_tiny_wedge_derivatives

    def checked_tiny_wedge(*args, **kwargs):
        tiny_wedge_calls.append(kwargs["force_scale_n"])
        return original_verifier(*args, **kwargs)

    monkeypatch.setattr(s3_patch_augmented, "verify_tiny_wedge_derivatives",
                        checked_tiny_wedge)
    work = owner.law.nodal_force_linearization(source, target, z, tangent=False)
    assert tiny_wedge_calls and all(scale > 0.0 for scale in tiny_wedge_calls)
    assert work.tangent is None
    np.testing.assert_allclose(work.residual, -z @ work.gap_gradient,
                               rtol=1e-6, atol=1e-10)
    assert np.linalg.norm(work.gap_gradient[supported[1]]) > 0.0


def test_raw_nodal_force_seam_rejects_unresolved_topology_and_resource_limit():
    owner, source, target, _base, z = _case()
    saved_tie = owner.law.shared_edge_tie
    smooth_edges = owner.law._reference_smooth_target_edges()
    assert smooth_edges is not None and len(smooth_edges[0]) == 1
    neighbor, shared = smooth_edges[0][0]
    assert neighbor == 1
    edge = target[shared[1]]-target[shared[0]]
    owner.law.shared_edge_tie = (saved_tie[0], edge/np.linalg.norm(edge),
                                 saved_tie[2])
    with pytest.raises(UnsupportedQ4S3Branch):
        owner.law.nodal_force_linearization(source, target, z, tangent=False)
    owner.law.shared_edge_tie = saved_tie
    original_faces = owner.law.target_face_indices
    owner.law.target_face_indices = original_faces + (original_faces[0],)
    with pytest.raises(UnsupportedQ4S3Branch):
        owner.law.nodal_force_linearization(source, target, z, tangent=False)
    owner.law.target_face_indices = original_faces
    owner.law.geometry.maximum_energy_evaluations = 100
    with pytest.raises(MemoryError, match="budget exhausted"):
        owner.law.nodal_force_linearization(source, target, z, tangent=False)


def test_raw_nodal_force_seam_resolves_gap_rows_at_zero_force():
    owner, source, target, base, _z = _case()
    work = owner.law.nodal_force_linearization(
        source, target, np.zeros_like(base.reference_support), tangent=True,
    )
    np.testing.assert_array_equal(work.residual, np.zeros_like(work.residual))
    np.testing.assert_array_equal(work.tangent, np.zeros_like(work.tangent))
    supported = np.flatnonzero(base.reference_support > 0.0)
    assert np.all(np.linalg.norm(work.gap_gradient[supported], axis=1) > 0.0)
    h = 7.5e-9
    shifted_minus = source.copy()
    shifted_plus = source.copy()
    shifted_minus[1, 0] -= h
    shifted_plus[1, 0] += h
    observed = (owner.law.geometry.evaluate(shifted_plus, target).normalized_gap
                - owner.law.geometry.evaluate(shifted_minus, target).normalized_gap)/(2*h)
    np.testing.assert_allclose(work.gap_gradient[supported, 3],
                               observed[supported], rtol=1e-5, atol=1e-8)


def test_raw_seam_rejects_biased_supported_zero_force_gap_row(monkeypatch):
    owner, source, target, base, z = _case()
    tiny_source = owner.law.source_face_indices[1]  # pair 2, two target faces
    local = next(index for index, node in enumerate(tiny_source)
                 if base.reference_support[node] > 0.0)
    node = tiny_source[local]
    z[node] = 0.0
    original_verifier = s3_patch_augmented.verify_tiny_wedge_derivatives
    calls = []

    def biased_zero_force_row(*args, **kwargs):
        used, scenarios = original_verifier(*args, **kwargs)
        calls.append(True)
        altered = []
        for gradient_error, hessian_error in scenarios:
            biased = gradient_error.copy()
            biased[1, local, 0] += 1.0e-4*base.reference_support[node]
            altered.append((biased, hessian_error))
        return used, tuple(altered)

    monkeypatch.setattr(s3_patch_augmented, "verify_tiny_wedge_derivatives",
                        biased_zero_force_row)
    with pytest.raises(UnsupportedQ4S3Branch,
                       match="raw supported gap derivative is unresolved"):
        owner.law.nodal_force_linearization(source, target, z, tangent=False)
    assert calls


def test_seam_ownership_ignores_sloped_face_without_tie():
    owner, _source, _target, _base, _z = _case()
    saved = json.loads((Path(__file__).parent / "fixtures" /
                        "general_contact_q4_s3_sloped_neighbor_seam_pose.json").read_text())
    source = np.asarray(saved["source"])
    target = np.asarray(saved["target"])
    z = np.asarray(saved["nodal_force_n"])
    faces = owner.law.source_face_indices
    _, unrelated_normal = s3_patch_augmented.projected_q4_source_face(
        source[faces[0]])
    _, tied_normal = s3_patch_augmented.projected_q4_source_face(
        source[faces[1]])
    assert abs(unrelated_normal[0]) > 1.0e-4
    assert abs(tied_normal[0]) < 1.0e-4
    assert owner.law._only_artificial_shared_edge_ties(
        source, target, failed_pair=3)
    raw = owner.law.nodal_force_linearization(source, target, z, tangent=True)
    np.testing.assert_allclose(raw.residual, -z @ raw.gap_gradient,
                               rtol=1.0e-6, atol=1.0e-10)
    h = 1.0e-7
    plus = source.copy()
    minus = source.copy()
    plus[1, 2] += h
    minus[1, 2] -= h
    observed = (owner.law.nodal_force_linearization(
        plus, target, z, tangent=False).residual
        - owner.law.nodal_force_linearization(
            minus, target, z, tangent=False).residual)/(2.0*h)
    predicted = raw.tangent[:, 5]
    active = owner.law.tangent_active_coordinates
    assert (np.linalg.norm((observed-predicted)[active])
            / np.linalg.norm(predicted[active]) < 1.0e-5)

    support = owner.law.geometry.evaluate(source, target).reference_support
    multipliers = np.zeros_like(support)
    np.divide(z, support, out=multipliers, where=support > 0.0)
    projected = owner.law.work(source, target, multipliers, tangent=True)
    np.testing.assert_allclose(projected.residual, raw.residual,
                               rtol=1.0e-6, atol=1.0e-10)
    assert np.all(np.isfinite(projected.tangent))


def test_monolithic_guided_slide_reaches_exact_s3_diagonal(monkeypatch):
    """Exercise the guarded seam derivative inside an accepted static step."""
    model, definition, settings, program = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    master = int(model.mesh.dof_manager.get_node_dofs(1)[0])
    slave = int(model.mesh.dof_manager.get_node_dofs(2)[0])
    model.add_constraint_equation(
        terms=((slave, 1.0), (master, -1.0)),
        dependent_dof=slave, source_id="guided_seam_node2_ux",
    )
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=0.3,
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
        q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    program = replace(
        program, waypoints=((0.0, 0.0), (0.0, -0.01504),
                            (0.05, -0.01504)),
        nodal_forces=(ContactPathNodalForce(16, "uz", (0.0, -0.3, -0.3)),),
    )
    seam_calls = []
    original = s3_patch_augmented.MovingS3PatchProjectedNormal._shared_edge_nodal_force_limit

    def counted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        seam_calls.append(True)
        return result

    monkeypatch.setattr(s3_patch_augmented.MovingS3PatchProjectedNormal,
                        "_shared_edge_nodal_force_limit", counted)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        tolerance=1.0e-6, emit_restart_checkpoint=False,
    )
    assert result.status == "completed", result.failure_reason
    assert len(result.steps) == 2
    assert len(seam_calls) >= 1
    assert abs(0.35 + result.displacements[slave] - 0.4) < 1.0e-12
    history = result.info["force_displacement_history"]
    assert history[-1]["constraint_error"] < 1.0e-10
    assert history[-1]["contact"]["minimum_gap"] >= -5.0e-5


def test_raw_seam_rejects_consistent_near_null_tangent_bias(monkeypatch):
    owner, _source, _target, _base, _z = _case()
    saved = json.loads((Path(__file__).parent / "fixtures" /
                        "general_contact_q4_s3_sloped_neighbor_seam_pose.json").read_text())
    source = np.asarray(saved["source"])
    target = np.asarray(saved["target"])
    z = np.asarray(saved["nodal_force_n"])
    direction = np.r_[np.tile((1.0, 0.0, 0.0), len(source)),
                      np.zeros(3*len(target))]
    direction /= np.linalg.norm(direction)
    original = owner.law._nodal_force_unresolved

    def biased(*args, **kwargs):
        work = original(*args, **kwargs)
        if work.tangent is None or kwargs.get("tiny_wedge_pair") is None:
            return work
        tangent = work.tangent + 1.0e-4*np.outer(direction, direction)
        return replace(work, tangent=tangent)

    monkeypatch.setattr(owner.law, "_nodal_force_unresolved", biased)
    with pytest.raises(UnsupportedQ4S3Branch,
                       match="raw shared-edge tie tangent direction or symmetry disagrees"):
        owner.law.nodal_force_linearization(source, target, z, tangent=True)


def test_raw_nodal_force_tangent_completion_matches_unrestricted_hessian():
    """The restricted-column raw tangent maps the same symmetric work."""
    owner, source, target, base, z = _case()
    supported = np.flatnonzero(base.reference_support > 0.0)
    z[supported[0]] = 0.0
    law = owner.law
    active = np.asarray(law.tangent_active_coordinates, dtype=np.intp)
    assert len(active) < law.n_coordinates
    shifted = np.asarray(source, dtype=float).copy()
    shifted[1, 0] += 1.0e-7
    restricted = law._nodal_force_unresolved(shifted, target, z, tangent=True)
    saved = law.tangent_active_coordinates
    law.tangent_active_coordinates = None
    try:
        unrestricted = law._nodal_force_unresolved(shifted, target, z, tangent=True)
    finally:
        law.tangent_active_coordinates = saved
    np.testing.assert_allclose(restricted.residual, unrestricted.residual,
                               rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_allclose(restricted.gap_gradient, unrestricted.gap_gradient,
                               rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_allclose(restricted.tangent, restricted.tangent.T,
                               rtol=1.0e-10, atol=1.0e-14)
    np.testing.assert_allclose(
        restricted.tangent[np.ix_(active, active)],
        unrestricted.tangent[np.ix_(active, active)], rtol=1.0e-12, atol=1.0e-15)
    inactive = np.setdiff1d(np.arange(law.n_coordinates, dtype=np.intp), active)
    np.testing.assert_allclose(
        restricted.tangent[np.ix_(active, inactive)],
        unrestricted.tangent[np.ix_(active, inactive)], rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_allclose(
        restricted.tangent[np.ix_(inactive, active)],
        unrestricted.tangent[np.ix_(inactive, active)], rtol=1.0e-12, atol=1.0e-15)


def test_raw_seam_rejects_asymmetric_tangent_bias(monkeypatch):
    """A skew tangent perturbation must still fail the seam symmetry gate."""
    owner, _source, _target, _base, _z = _case()
    saved = json.loads((Path(__file__).parent / "fixtures" /
                        "general_contact_q4_s3_sloped_neighbor_seam_pose.json").read_text())
    source = np.asarray(saved["source"])
    target = np.asarray(saved["target"])
    z = np.asarray(saved["nodal_force_n"])
    original = owner.law._nodal_force_unresolved

    def biased(*args, **kwargs):
        work = original(*args, **kwargs)
        if work.tangent is None:
            return work
        pattern = np.triu(np.ones_like(work.tangent), 1)
        skew = pattern-pattern.T
        return replace(work, tangent=work.tangent
                       + 1.0e-4*np.linalg.norm(work.tangent)*skew
                       / np.linalg.norm(skew))

    monkeypatch.setattr(owner.law, "_nodal_force_unresolved", biased)
    with pytest.raises(UnsupportedQ4S3Branch,
                       match="raw shared-edge tie tangent direction or symmetry disagrees"):
        owner.law.nodal_force_linearization(source, target, z, tangent=True)
