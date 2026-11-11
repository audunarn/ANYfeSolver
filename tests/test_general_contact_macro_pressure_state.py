"""Development admission boundary for a macro pressure state on fine Q4 geometry."""

import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace
from scipy import sparse

from anyfesolver import validate_general_contact
from anyfesolver.general_contact import linearized_pivot
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_s3_jet2_static_owner import _settings
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case

from anyfesolver.general_contact.pressure_basis import (
    build_q4_source_macro_q1_pair_basis,
)
from anyfesolver.general_contact.s3_patch_augmented import (
    MovingS3PatchProjectedNormal,
)
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


def test_macro_projected_state_uses_aggregate_gap_and_keeps_physical_nodes():
    source_faces = (
        S3PatchFace((1, 2, 5, 4), np.array((
            (0., 0., 0.02), (1., 0., 0.02),
            (1., 1., 0.02), (0., 1., 0.02),
        ))),
        S3PatchFace((2, 3, 6, 5), np.array((
            (1., 0., 0.02), (2., 0., 0.02),
            (2., 1., 0.02), (1., 1., 0.02),
        ))),
    )
    target_faces = (S3PatchFace((7, 8, 9), np.array((
        (0.15, 0.1, 0.), (1.9, 0.1, 0.), (0.4, 0.9, 0.),
    ))),)
    geometry = MovingS3PatchPrimalMortarPenalty(
        source_faces, target_faces, 10_000., difference_step=1e-6,
        maximum_energy_evaluations=10_000,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
        differentiation="pair_local", maximum_candidate_pairs=2,
    )
    basis = build_q4_source_macro_q1_pair_basis(
        tuple(face.node_ids for face in source_faces))
    law = MovingS3PatchProjectedNormal(geometry, pressure_basis=basis)
    assert law.source_node_ids == geometry.source_node_ids
    assert law.pressure_mode_ids == basis.mode_ids
    assert len(law.source_node_ids) == 6
    assert len(law.pressure_mode_ids) == 4

    source = geometry.reference_source.copy()
    source[:, 2] -= 0.02003
    fine = geometry.evaluate(source, geometry.reference_target)
    multiplier = np.array((0.2, 0.1, 0.3, 0.4))
    macro = law.evaluate(source, geometry.reference_target, multiplier)
    B = basis.as_array()
    support = B.T @ fine.reference_support
    weighted = B.T @ fine.reference_weighted_gap
    gram = B.T @ fine.pressure_gram @ B
    np.testing.assert_allclose(macro.reference_support, support)
    np.testing.assert_allclose(macro.reference_weighted_gap, weighted)
    np.testing.assert_allclose(macro.pressure_gram, gram)
    np.testing.assert_allclose(macro.normalized_gap, weighted/support)
    np.testing.assert_allclose(
        macro.pressure,
        np.maximum(0., multiplier-10_000.*weighted/support),
    )
    assert macro.minimum_physical_gap == fine.minimum_physical_gap
    with pytest.raises(ValueError, match="needs the Q4/S3 Jet2 derivative"):
        law.work(source, geometry.reference_target, multiplier)
    with pytest.raises(ValueError, match="multipliers"):
        law.evaluate(source, geometry.reference_target, np.zeros(6))


def test_macro_q4_s3_jet2_work_matches_frozen_integral_variation_and_tangent():
    source_faces = (
        S3PatchFace((1, 2, 5, 4), np.array((
            (0., 0., 0.02), (1., 0., 0.02),
            (1., 1., 0.02), (0., 1., 0.02),
        ))),
        S3PatchFace((2, 3, 6, 5), np.array((
            (1., 0., 0.02), (2., 0., 0.02),
            (2., 1., 0.02), (1., 1., 0.02),
        ))),
    )
    target_faces = (S3PatchFace((7, 8, 9), np.array((
        (0.15, 0.11, 0.), (1.88, 0.14, 0.), (0.37, 0.91, 0.),
    ))),)
    geometry = MovingS3PatchPrimalMortarPenalty(
        source_faces, target_faces, 10_000., difference_step=1e-6,
        maximum_energy_evaluations=10_000,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
        q4_source_derivative_tolerances=(1e-5, 1e-5, 1e-5, 1e-5),
        differentiation="jet2_q4_s3", maximum_candidate_pairs=2,
    )
    basis = build_q4_source_macro_q1_pair_basis(
        tuple(face.node_ids for face in source_faces))
    law = MovingS3PatchProjectedNormal(geometry, pressure_basis=basis)
    source = geometry.reference_source.copy()
    source[:, 2] -= 0.02003
    target = geometry.reference_target.copy()
    multipliers = np.zeros(len(basis.mode_ids))
    work = law.work(source, target, multipliers, tangent=True)
    assert work.state.active_pressure_dofs > 0

    coordinates = np.concatenate((source, target)).reshape(-1)
    source_count = len(source)
    direction = np.random.default_rng(29).normal(size=len(coordinates))
    direction /= np.linalg.norm(direction)
    step = 2.0e-7

    def pose(sign):
        moved = (coordinates+sign*step*direction).reshape(-1, 3)
        return moved[:source_count], moved[source_count:]

    plus_source, plus_target = pose(1.0)
    minus_source, minus_target = pose(-1.0)
    plus = law.work(plus_source, plus_target, multipliers, tangent=False)
    minus = law.work(minus_source, minus_target, multipliers, tangent=False)
    directional = (plus.residual-minus.residual)/(2.0*step)
    prediction = work.tangent @ direction
    assert np.linalg.norm(directional-prediction)/np.linalg.norm(prediction) < 1e-5

    pressure = work.state.pressure.copy()
    gap = work.state.normalized_gap.copy()
    def frozen_work(sign):
        moved_source, moved_target = pose(sign)
        fine = geometry.evaluate(moved_source, moved_target)
        support, weighted, _gram = basis.transform_integrals(
            fine.reference_support, fine.reference_weighted_gap,
            fine.pressure_gram,
        )
        return float(pressure @ (weighted-gap*support))

    independent_virtual_work = -(frozen_work(1.0)-frozen_work(-1.0))/(2.0*step)
    assert float(work.residual @ direction) == pytest.approx(
        independent_virtual_work, rel=1e-5, abs=1e-8,
    )


def test_macro_manager_binds_mode_identity_and_restart_without_changing_geometry():
    model, definition, settings = _two_q4_two_s3()
    settings = replace(_settings(settings), q4_source_pressure_basis="macro_q1_pair_dev")
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.law.source_node_ids == (1, 2, 3, 4, 5, 6)
    assert manager.law.pressure_mode_ids == (1, 3, 4, 6)
    assert manager.normal_multipliers.shape == (4,)
    assert manager.map.shape[0] == 3*(6+4)
    displacement = np.zeros(manager.n_dof)
    for node in range(1, 7):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02003
    _force, _tangent, result = manager.evaluate(displacement, tangent=False)
    assert result.source_pressure_node_ids == manager.law.pressure_mode_ids
    assert len(result.normalized_weighted_gap_m) == 4
    assert result.gap_tolerance_met
    assert result.to_dict()["pressure_basis_version"] == "q4_source_macro_q1_pair_v1"
    assert result.to_dict()["source_geometry_node_ids"] == [1, 2, 3, 4, 5, 6]
    contract = manager.restart_contract()
    assert contract["settings"]["pressure_basis_mode_ids"] == [1, 3, 4, 6]
    assert len(contract["settings"]["pressure_basis_matrix"]) == 6


def test_macro_manager_rejects_nonstrip_source_topology():
    model, definition, settings = _two_q4_two_s3()
    settings = replace(_settings(settings), q4_source_pressure_basis="macro_q1_pair_dev")
    source = replace(definition.surfaces[0], element_ids=(2, 1))
    reordered = replace(definition, surfaces=(source, definition.surfaces[1]))
    report = validate_general_contact(model, reordered, settings)
    assert not report.valid
    assert any("macro Q1 pressure topology" in issue for issue in report.issues)


def test_selected_knot_manager_maps_fine_geometry_and_binds_restart():
    model, definition, settings, _program = build_case(4)
    settings = replace(
        settings,
        q4_source_pressure_basis="selected_q1_knots_dev",
        q4_source_pressure_knot_columns=(0, 1, 3, 4),
    )
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert len(manager.law.source_node_ids) == 10
    assert len(manager.law.pressure_mode_ids) == 8
    assert manager.normal_multipliers.shape == (8,)
    contract = manager.restart_contract()
    assert contract["settings"]["q4_source_pressure_knot_columns"] == [0, 1, 3, 4]
    assert contract["settings"]["pressure_basis_version"] == (
        "q4_source_selected_q1_knots_v1"
    )
    changed = replace(settings, q4_source_pressure_knot_columns=(0, 2, 4))
    other = GeneralS3PatchManager(model, definition, changed)
    assert other.restart_contract() != contract

    displacement = np.zeros(manager.n_dof)
    for node in range(1, 11):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -6.0e-5
    force, tangent, result = manager.evaluate(displacement, tangent=True)
    assert result.active_pressure_dofs == 8
    assert result.source_pressure_node_ids == manager.law.pressure_mode_ids
    assert result.gap_tolerance_met
    assert np.all(np.isfinite(force))
    assert np.all(np.isfinite(tangent.data))
    direction = np.zeros(manager.n_dof)
    direction[model.mesh.dof_manager.get_node_dofs(3)[2]] = 1.0
    step = 2.0e-7
    plus = manager.evaluate(displacement+step*direction, tangent=False)[0]
    minus = manager.evaluate(displacement-step*direction, tangent=False)[0]
    finite_difference = (plus-minus)/(2.0*step)
    predicted = tangent @ direction
    assert np.linalg.norm(finite_difference-predicted)/np.linalg.norm(predicted) < 1e-5


def test_selected_knot_admission_rejects_out_of_strip_mode_identity():
    model, definition, settings, _program = build_case(4)
    settings = replace(
        settings,
        q4_source_pressure_basis="selected_q1_knots_dev",
        q4_source_pressure_knot_columns=(0, 2, 5),
    )
    report = validate_general_contact(model, definition, settings)
    assert not report.valid
    assert any("selected Q1 pressure topology" in issue for issue in report.issues)


def test_macro_pressure_response_transforms_fine_derivatives_before_gap_law(monkeypatch):
    basis = build_q4_source_macro_q1_pair_basis(((1, 2, 5, 4), (2, 3, 6, 5)))
    B = basis.as_array()
    fine_support = np.array((1.0, 1.2, 0.8, 1.1, 0.9, 1.3))
    fine_weighted = -0.02*fine_support
    macro_support = B.T @ fine_support
    macro_gap = B.T @ fine_weighted/macro_support
    fine_ds = np.array((0.2, -0.1, 0.0, 0.15, -0.05, 0.1))
    fine_dw = np.array((0.3, 0.4, 0.2, 0.1, 0.5, 0.35))
    geometry = SimpleNamespace(
        differentiation="jet2_q4_s3", pair_laws=(object(),),
        evaluate=lambda _source, _target: SimpleNamespace(
            reference_support=fine_support,
            reference_weighted_gap=fine_weighted,
            overlap_signature=(1,),
        ),
    )
    law = SimpleNamespace(
        geometry=geometry, pressure_basis=basis,
        evaluate=lambda _source, _target, _multipliers: SimpleNamespace(
            reference_support=macro_support,
            normalized_gap=macro_gap,
        ),
    )
    manager = SimpleNamespace(
        _positions=lambda _u: (np.zeros((1, 3)), np.zeros((1, 3))),
        law=law, normal_multipliers=np.ones(4),
        map=sparse.csr_matrix(([1.0], ([0], [0])), shape=(6, 1)),
        settings=SimpleNamespace(maximum_general_s3_energy_evaluations=10),
    )
    monkeypatch.setattr(linearized_pivot, "_projected_clip_signature",
                        lambda *_args: (1,))
    monkeypatch.setattr(linearized_pivot, "_jet2_integral_gradients",
                        lambda *_args, **_kwargs: (
                            np.column_stack((fine_dw, np.zeros((6, 5)))),
                            np.column_stack((fine_ds, np.zeros((6, 5)))),
                        ))
    response = linearized_pivot._constrained_gap_response(
        manager, np.zeros(1), sparse.csr_matrix([[2.0]]),
        sparse.eye(1, format="csr"), np.zeros((0, 1)),
        np.arange(4), np.arange(4), difference_step=1e-6,
    )
    dS = B.T @ fine_ds
    dW = B.T @ fine_dw
    G = dW-macro_gap*dS
    expected = np.outer(G/macro_support, G)/2.0
    np.testing.assert_allclose(response, expected, rtol=1e-12, atol=1e-12)
