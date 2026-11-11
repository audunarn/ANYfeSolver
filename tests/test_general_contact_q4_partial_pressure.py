"""Independent Q1 pressure-space controls on partial Q4/Q4 overlap."""

from dataclasses import replace

import numpy as np

from anyfesolver.general_contact.q4_source_overlap import q4_source_overlap_integrals
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_q4_augmented import _augmented_case


def _reference_support_gram(start, y_end=1.0):
    abscissas, weights = np.polynomial.legendre.leggauss(6)
    x = start+(1-start)*(abscissas+1)/2
    x_weights = (1-start)*weights/2
    y = y_end*(abscissas+1)/2
    y_weights = y_end*weights/2
    support = np.zeros(4)
    gram = np.zeros((4, 4))
    for first, first_weight in zip(x, x_weights):
        for second, second_weight in zip(y, y_weights):
            basis = np.array(((1-first)*(1-second),
                              first*(1-second), first*second,
                              (1-first)*second))
            weight = first_weight*second_weight
            support += weight*basis
            gram += weight*np.outer(basis, basis)
    return support, gram


def test_q4_partial_overlap_retains_q1_support_and_gram_rank():
    source = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    for start in (0.5, 0.98):
        target = source.copy()
        target[:, 0] += start
        state = q4_source_overlap_integrals(
            source, source, target,
            area_tolerance=1e-11, weighted_gap_tolerance=1e-11,
            maximum_triangles=300,
        )
        expected_support, expected_gram = _reference_support_gram(start)
        np.testing.assert_allclose(state.support, expected_support,
                                   rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(state.pressure_gram, expected_gram,
                                   rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(state.pressure_gram.sum(axis=1),
                                   state.support, atol=1e-11)
        np.testing.assert_allclose(state.support.sum(),
                                   1-start, atol=1e-11)
        assert np.min(np.linalg.eigvalsh(state.pressure_gram)) > 0.0
        assert state.integration_triangles <= 300


def test_q4_partial_overlap_constant_augmented_pressure_balances_bodies():
    for shift, area, start in ((0.5, 0.54, 0.4),
                               (1.08, 0.018, 0.98)):
        model, definition, settings = _augmented_case()
        for node in (5, 6, 7, 8):
            model.mesh.get_node(node).x += shift
        settings = replace(settings, q4_source_gap_certificate="projected_matching")
        manager = GeneralS3PatchManager(model, definition, settings)
        manager.set_normal_multipliers(np.full(4, 10.0))
        displacement = np.zeros(manager.n_dof)
        for node in (1, 2, 3, 4):
            displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.015
        _force, _tangent, contact = manager.evaluate(displacement, tangent=False)
        assert contact.active_pressure_dofs == 4
        np.testing.assert_allclose(contact.numerical_pressure_pa, 10.0,
                                   rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(contact.current_overlap_area, area,
                                   atol=1e-10)
        np.testing.assert_allclose(contact.integration_force[2], 10.0*area,
                                   atol=1e-7)
        _, expected_gram = _reference_support_gram(start, y_end=0.9)
        assert contact.pressure_gram_rank == 4
        np.testing.assert_allclose(contact.pressure_gram_condition_estimate,
                                   np.linalg.cond(expected_gram), rtol=1e-8)
        diagonal = np.sqrt(np.diag(expected_gram))
        scaled_gram = expected_gram/diagonal[:, None]/diagonal[None, :]
        assert contact.pressure_gram_supported_modes == 4
        assert contact.pressure_gram_scaled_rank == 4
        np.testing.assert_allclose(
            contact.pressure_gram_scaled_condition_estimate,
            np.linalg.cond(scaled_gram), rtol=1e-8,
        )
        assert contact.to_dict()["pressure_gram_rank"] == 4
        np.testing.assert_allclose(
            np.asarray(contact.integration_force)+contact.target_force,
            0.0, atol=1e-7,
        )


def test_q4_projected_pressure_diagnostic_has_no_overlap_and_v0_is_unchanged():
    model, definition, settings = _augmented_case()
    projected = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="projected_matching"),
    )
    displacement = np.zeros(projected.n_dof)
    for node in (5, 6, 7, 8):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[0]] = 2.0
    _force, _tangent, contact = projected.evaluate(displacement, tangent=False)
    assert contact.pressure_gram_rank == 0
    assert contact.pressure_gram_condition_estimate is None
    assert contact.to_dict()["pressure_gram_condition_estimate"] is None

    original = GeneralS3PatchManager(model, definition, settings)
    _force, _tangent, old_contact = original.evaluate(
        np.zeros(original.n_dof), tangent=False,
    )
    assert "pressure_gram_rank" not in old_contact.to_dict()
    assert "pressure_gram_condition_estimate" not in old_contact.to_dict()
