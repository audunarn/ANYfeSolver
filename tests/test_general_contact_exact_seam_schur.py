"""Bounded pressure-response checks at an opt-in artificial Q4/S3 seam."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact import linearized_pivot
from anyfesolver.general_contact.linearized_pivot import (
    _exact_artificial_seam_gap_gradient, _jet2_integral_gradients,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_coupled_refinement import build_case


def _saved_pose():
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "general_contact_q4_s3_neighbor_value_gate_pose.json").read_text())
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    source = np.asarray(fixture["source"])
    target = np.asarray(fixture["target"])
    owner.set_normal_multipliers(np.asarray(fixture["multipliers"]))
    base = owner.law.geometry.evaluate(source, target)
    selected = np.flatnonzero(base.reference_support > 0.0)
    return owner, source, target, base, selected


def test_exact_artificial_seam_response_has_stable_two_sided_gap_derivative():
    owner, source, target, base, selected = _saved_pose()
    diagnostics = {}
    derivative = _exact_artificial_seam_gap_gradient(
        owner, source, target, base, selected, diagnostics=diagnostics,
    )
    assert derivative.shape == (len(selected), 3*(len(source)+len(target)))
    assert diagnostics["schur_exact_seam_generalized_gradient"]
    assert diagnostics["schur_exact_seam_one_sided_error"] < 1.0e-7
    assert diagnostics["schur_exact_seam_step_error"] < 1.0e-7
    coordinates = np.concatenate((source, target)).reshape(-1)
    for coordinate in (3*1, 3*2+2, 3*(len(source)+1)):
        h = 7.5e-9  # independent finer step than the candidate's two levels
        observed = []
        for sign in (-1.0, 1.0):
            shifted = coordinates.copy()
            shifted[coordinate] += sign*h
            state = owner.law.geometry.evaluate(
                shifted[:3*len(source)].reshape(-1, 3),
                shifted[3*len(source):].reshape(-1, 3),
            )
            observed.append(state.normalized_gap[selected])
        np.testing.assert_allclose(
            derivative[:, coordinate], (observed[1]-observed[0])/(2*h),
            atol=1.0e-8, rtol=1.0e-5,
        )


def test_exact_seam_response_requires_opt_in_and_declared_geometry_budget():
    owner, source, target, base, selected = _saved_pose()
    saved_tie = owner.law.shared_edge_tie
    owner.law.shared_edge_tie = None
    assert _exact_artificial_seam_gap_gradient(
        owner, source, target, base, selected,
    ) is None
    owner.law.shared_edge_tie = saved_tie
    original_faces = owner.law.target_face_indices
    owner.law.target_face_indices = original_faces + (original_faces[0],)
    assert _exact_artificial_seam_gap_gradient(
        owner, source, target, base, selected,
    ) is None
    owner.law.target_face_indices = original_faces
    owner.settings = replace(
        owner.settings, maximum_general_s3_energy_evaluations=100,
    )
    with pytest.raises(ValueError, match="geometry budget exceeded"):
        _exact_artificial_seam_gap_gradient(
            owner, source, target, base, selected,
        )


def test_smooth_near_diagonal_wedge_bounds_inactive_gap_response():
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "general_contact_q4_s3_schur_tiny_wedge_pose.json").read_text())
    owner, _, _, _, _ = _saved_pose()
    source = np.asarray(fixture["source_positions_m"])
    target = np.asarray(fixture["target_positions_m"])
    owner.set_normal_multipliers(
        np.asarray(fixture["normal_multipliers_pa"]),
    )
    base = owner.law.geometry.evaluate(source, target)
    selected = np.flatnonzero(base.reference_support > 0.0)
    assert owner.law.near_d1_diagonal_tiny_wedge(source, target, 2)
    assert not owner.law._only_artificial_shared_edge_ties(source, target)
    diagnostics = {}
    weighted, support = _jet2_integral_gradients(
        owner.law.geometry, source, target, selected,
        3*(len(source)+len(target)),
        owner.settings.maximum_general_s3_energy_evaluations,
        manager=owner, base=base, diagnostics=diagnostics,
    )
    waiver = diagnostics.get("schur_tiny_wedge_gap_response")
    if waiver is not None:
        assert waiver["pair"] == 2
        assert waiver["maximum_gradient_error"] < 1.0e-7
    gap_gradient = (
        weighted-base.normalized_gap[selected, None]*support
    )/base.reference_support[selected, None]
    coordinates = np.concatenate((source, target)).reshape(-1)
    for coordinate in (3*1, 3*2+2, 3*(len(source)+1)):
        h = 7.5e-9
        observed = []
        for sign in (-1.0, 1.0):
            shifted = coordinates.copy()
            shifted[coordinate] += sign*h
            state = owner.law.geometry.evaluate(
                shifted[:3*len(source)].reshape(-1, 3),
                shifted[3*len(source):].reshape(-1, 3),
            )
            observed.append(state.normalized_gap[selected])
        np.testing.assert_allclose(
            gap_gradient[:, coordinate], (observed[1]-observed[0])/(2*h),
            atol=1.0e-7, rtol=1.0e-5,
        )
    owner.law.shared_edge_tie = None
    if waiver is not None:
        with pytest.raises(ValueError, match="pair=2"):
            _jet2_integral_gradients(
                owner.law.geometry, source, target, selected,
                3*(len(source)+len(target)),
                owner.settings.maximum_general_s3_energy_evaluations,
                manager=owner, base=base,
            )
    else:
        untied_weighted, untied_support = _jet2_integral_gradients(
            owner.law.geometry, source, target, selected,
            3*(len(source)+len(target)),
            owner.settings.maximum_general_s3_energy_evaluations,
            manager=owner, base=base,
        )
        np.testing.assert_array_equal(untied_weighted, weighted)
        np.testing.assert_array_equal(untied_support, support)


def test_gap_operand_roundoff_does_not_admit_material_value_error(monkeypatch):
    owner, source, target, base, selected = _saved_pose()
    original = linearized_pivot.q4_s3_overlap_derivatives

    def biased_derivative(*args, **kwargs):
        derivative = original(*args, **kwargs)
        return replace(
            derivative,
            reference_weighted_gap=derivative.reference_weighted_gap+1e-8,
        )

    monkeypatch.setattr(linearized_pivot, "q4_s3_overlap_derivatives",
                        biased_derivative)
    with pytest.raises(ValueError, match="pressure response derivative disagrees"):
        _jet2_integral_gradients(
            owner.law.geometry, source, target, selected,
            3*(len(source)+len(target)),
            owner.settings.maximum_general_s3_energy_evaluations,
            manager=owner, base=base,
        )
