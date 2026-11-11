"""Local S3-V2D projected finite penalty checks; no coupled shell solve.

The independent oracle integrates literal scalar P1 hats on declared polygons.
It does not call the production overlap, AD, penalty or contact functions.
"""
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import BoundaryCondition, create_shell_element, validate_general_contact
from anyfesolver.general_contact.s3_patch_static import (
    GeneralS3PatchManager, S3_PROJECTED_PENALTY_PROFILE,
)
from test_general_contact_s3_patch_static import _case


def _candidate():
    model, definition, settings = _case()
    # Use the target anti-diagonal: y=x in the old fixture passes exactly
    # through source (.1,.1), outside the inherited smooth Jet2 branch.
    for eid, ids in ((3, (5, 6, 8)), (4, (6, 7, 8))):
        model.mesh.elements[eid] = create_shell_element(eid, ids, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0., 0., 1.), reference_surface_offset=.01)
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], normal_law="projected_penalty"),))
    return model, definition, replace(settings, general_s3_differentiation="jet2_s3_dev")


def _p1_moments(polygons, hats, count):
    """Exact affine polynomial integrals using triangle vertex values."""
    support, mass = np.zeros(count), np.zeros((count, count))
    for polygon, shape in zip(polygons, hats, strict=True):
        vertices = np.asarray(polygon)
        for i in range(1, len(vertices)-1):
            triangle = vertices[[0, i, i+1]]
            a, b = triangle[1]-triangle[0], triangle[2]-triangle[0]
            area = .5*abs(a[0]*b[1]-a[1]*b[0])
            values = np.array([shape(*point) for point in triangle])
            sums = values.sum(axis=0)
            support += area*sums/3
            mass += area/12*(np.outer(sums, sums)+values.T@values)
    return support, mass


def _independent_planar_moments():
    # Source local coordinates xi=x-.1, eta=y-.1, split xi+eta=1.
    polygons = (((0., 0.), (.9, 0.), (.9, .1), (.1, .9), (0., .9)),
                ((.9, .1), (.9, .9), (.1, .9)))
    hats = (lambda x, y: [1-x-y, x, 0., y],
            lambda x, y: [0., 1-y, x+y-1, 1-x])
    source_support, source_mass = _p1_moments(polygons, hats, 4)
    # Target world coordinates, independently split x+y=1.
    target_polygons = (((.1, .1), (.9, .1), (.1, .9)),
                       ((.9, .1), (1., .1), (1., 1.), (.1, 1.), (.1, .9)))
    target_hats = (lambda x, y: [1-x-y, x, 0., y],
                   lambda x, y: [0., 1-y, x+y-1, 1-x])
    target_support, _ = _p1_moments(target_polygons, target_hats, 4)
    return source_support, source_mass, target_support


def _normal_pose(model, manager, distance=-.02):
    u = np.zeros(manager.n_dof)
    for node in range(1, 5):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = distance
    return u


def test_scalar_p1_plane_full_both_body_work_and_prescribed_reaction():
    model, definition, settings = _candidate()
    model.add_boundary_condition(BoundaryCondition("prescribed_source", (1,), {"uz": -.02}))
    model.apply_boundary_conditions()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == S3_PROJECTED_PENALTY_PROFILE
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.law.geometry.differentiation == "jet2_s3"
    u = _normal_pose(model, manager)
    # Exclude prescribed source-normal variation, retaining its residual row.
    excluded = model.mesh.dof_manager.get_node_dofs(1)[2]
    free = np.delete(np.arange(manager.n_dof), excluded)
    manager.restrict_tangent_to_reduced_variations(sparse.eye(manager.n_dof, format="csr")[:, free])
    assert manager.law.tangent_active_coordinates.size < manager.law.n_coordinates
    force, tangent, result = manager.evaluate(u)
    support, mass, target_support = _independent_planar_moments()
    gap, pressure, area = -.005, 5., .81
    np.testing.assert_allclose(mass.sum(axis=1), support, atol=1e-14)
    np.testing.assert_allclose(result.reference_support_m2, support, atol=1e-13)
    np.testing.assert_allclose(result.normalized_weighted_gap_m, gap, atol=1e-14)
    np.testing.assert_allclose(result.numerical_pressure_pa, pressure, atol=1e-12)
    expected = np.zeros(manager.n_dof)
    for node in range(1, 5):
        expected[model.mesh.dof_manager.get_node_dofs(node)[2]] = -pressure*support[node-1]
    for node in range(5, 9):
        expected[model.mesh.dof_manager.get_node_dofs(node)[2]] = pressure*target_support[node-5]
    np.testing.assert_allclose(force, expected, atol=2e-12)
    assert abs(force[excluded]) > 0
    np.testing.assert_allclose(result.integration_force, (0., 0., pressure*area), atol=1e-12)
    np.testing.assert_allclose(result.target_force, (0., 0., -pressure*area), atol=1e-12)
    np.testing.assert_allclose(np.add(result.integration_moment, result.target_moment), 0., atol=1e-12)
    # Independent full generalized virtual work, including both bodies and
    # offset-induced rotational skin variations (zero work on this flat pose).
    direction = np.linspace(-.2, .3, manager.n_dof)
    np.testing.assert_allclose(force@direction, expected@direction, atol=1e-12)
    paths = [Path(inspect.getsourcefile(item)).resolve() for item in (
        GeneralS3PatchManager, type(manager.law), validate_general_contact)]
    print(json.dumps(dict(check="scalar_p1_plane", python_executable=sys.executable,
        imported_source=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
        support_max_error_m2=float(np.max(abs(np.array(result.reference_support_m2)-support))),
        full_generalized_residual_max_error=float(np.max(abs(force-expected))),
        prescribed_source_normal_reaction_n=float(force[excluded]),
        both_body_work_error_j=float(force@direction-expected@direction))))
    assert result.pressure_gram_supported_modes == 4
    assert result.active_pressure_dofs == 4
    assert result.penalty_energy == result.augmented_merit_j == 0.
    np.testing.assert_allclose(result.regularization_storage_j, .5*1000*gap**2*area, atol=1e-14)
    assert result.to_dict()["contact_work_convention"] == "projected_weak_gap_nonpotential_v1"
    assert result.normal_multiplier_estimate_pa == ()
    assert not manager.normal_multipliers.flags.writeable
    np.testing.assert_array_equal(manager.normal_multipliers, 0.)
    with pytest.raises(ValueError):
        manager.normal_multipliers[0] = 1.
    with pytest.raises(ValueError, match="no normal multipliers"):
        manager.set_normal_multipliers(np.ones(4))
    opened = manager.evaluate(np.zeros(manager.n_dof), tangent=False)
    np.testing.assert_array_equal(opened[0], 0.)
    assert opened[2].regularization_storage_j == 0.


def test_affine_gap_pressure_uses_exact_p1_mass_on_flat_source_projection():
    model, definition, settings = _candidate()
    manager = GeneralS3PatchManager(model, definition, settings)
    u = _normal_pose(model, manager)
    source_gaps = []
    for node in range(1, 5):
        point = model.mesh.get_node(node)
        source_gaps.append(-.005+.001*(point.x-.6)+.0005*(point.y-.6))
    # S3 projection follows the actual source normal. Keep it horizontal so
    # this exact XY-domain scalar oracle applies; an affine tilted target
    # gives varying physical gap without altering projected XY support.
    for node in range(5, 9):
        point = model.mesh.get_node(node)
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.001*(point.x-.6)-.0005*(point.y-.6)
    force, _, result = manager.evaluate(u, tangent=False)
    support, mass, _ = _independent_planar_moments()
    gaps = mass@source_gaps/support
    pressures = 1000*np.maximum(-gaps, 0.)
    np.testing.assert_allclose(result.normalized_weighted_gap_m, gaps, atol=1e-14)
    np.testing.assert_allclose(result.numerical_pressure_pa, pressures, atol=1e-12)
    np.testing.assert_allclose(result.reference_support_m2, support, atol=1e-13)
    np.testing.assert_allclose(np.add(result.integration_force, result.target_force), 0., atol=1e-12)


def test_moving_cut_has_zero_constant_gap_work_but_storage_flux_and_nonsymmetric_tangent():
    model, definition, settings = _candidate()
    manager = GeneralS3PatchManager(model, definition, settings)
    u = _normal_pose(model, manager)
    force, tangent, result = manager.evaluate(u)
    direction = np.zeros(manager.n_dof)
    for node in range(5, 9):
        direction[model.mesh.dof_manager.get_node_dofs(node)[0]] = 1.
    assert abs(force@direction) < 1e-12
    step = 1e-6
    plus, minus = (manager.evaluate(u+sign*step*direction, tangent=False) for sign in (1., -1.))
    # Independent moving rectangle area derivative dA/d(target x)=.9.
    np.testing.assert_allclose((plus[2].regularization_storage_j-minus[2].regularization_storage_j)/(2*step),
        .5*1000*.005**2*.9, rtol=1e-8)
    numeric = (plus[0]-minus[0])/(2*step)
    assert np.linalg.norm(tangent@direction-numeric)/np.linalg.norm(numeric) <= 1e-5
    assert sparse.linalg.norm(tangent-tangent.T) > 1e-3


def test_both_body_small_rotation_skin_map_smooth_directional_tangent_interval():
    model, definition, settings = _candidate()
    manager = GeneralS3PatchManager(model, definition, settings)
    u, direction = np.zeros(manager.n_dof), np.zeros(manager.n_dof)
    for node in range(1, 5):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        x = model.mesh.get_node(node).x
        u[dofs[0]], u[dofs[1]], u[dofs[2]] = .013, .007, -.02+.001*(x-.6)
        direction[dofs[2]], direction[dofs[3]] = .2+.1*(x-.6), .03
    for node in range(5, 9):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        x = model.mesh.get_node(node).x
        u[dofs[2]] = .0003*x
        direction[dofs[0]], direction[dofs[2]], direction[dofs[4]] = .1, -.1+.05*x, -.02
    force, tangent, result = manager.evaluate(u)
    np.testing.assert_allclose(np.add(result.integration_force, result.target_force), 0., atol=1e-11)
    np.testing.assert_allclose(np.add(result.integration_moment, result.target_moment), 0., atol=1e-11)
    analytic = tangent@direction
    errors = []
    for step in (1e-6, 3e-7, 1e-7):
        plus = manager.evaluate(u+step*direction, tangent=False)
        minus = manager.evaluate(u-step*direction, tangent=False)
        assert plus[2].active_pressure_dofs == minus[2].active_pressure_dofs == result.active_pressure_dofs == 4
        numeric = (plus[0]-minus[0])/(2*step)
        error = float(np.linalg.norm(analytic-numeric)/np.linalg.norm(numeric))
        errors.append(dict(step=step, relative_error=error))
        assert error <= 1e-5
    print(json.dumps(dict(check="both_body_small_rotation_directional_tangent", perturbations=errors)))


@pytest.mark.parametrize("width", [.02, .002, .0002])
def test_positive_support_cutoff_limit_retains_tiny_p1_hat(width):
    model, definition, settings = _candidate()
    manager = GeneralS3PatchManager(model, definition, settings)
    u = _normal_pose(model, manager)
    for node in range(1, 5):
        u[model.mesh.dof_manager.get_node_dofs(node)[0]] = .9-width
    force, _, result = manager.evaluate(u, tangent=False)
    # Ω=[1-width,1]x[.1,1] lies entirely in source first triangle.
    support = np.array([.495*width-.45*width**2, .45*width**2, 0., .405*width])
    np.testing.assert_allclose(result.reference_support_m2, support, rtol=2e-9, atol=1e-15)
    assert result.source_pressure_node_ids == (1, 2, 3, 4)
    assert result.active_pressure_dofs == result.pressure_gram_supported_modes == 3
    np.testing.assert_allclose(np.array(result.numerical_pressure_pa)[support > 0], 5., atol=1e-10)
    np.testing.assert_allclose(result.integration_force[2], 5*.9*width, rtol=2e-9)
    # Force vanishes with overlap area, even though supported pressure stays 5Pa.
    assert np.linalg.norm(force) < 5*.9*width*2
    print(json.dumps(dict(check="positive_support_limit", width_m=width,
        tiny_hat_support_m2=result.reference_support_m2[1],
        source_contact_normal_force_n=result.integration_force[2])))


@pytest.mark.parametrize("change", [
    {"general_s3_differentiation": "existing"},
    {"general_s3_differentiation": "jet2_q4_s3_dev"},
    {"q4_source_gap_certificate": "overlap_local_dev"},
    {"pair_motion": "restricted_sliding", "sliding_difference_step": 1e-5},
    {"normal_discretization": "pointwise_penalty"},
    {"augmentation_acceleration": "active_secant"},
    {"augmentation_support_resolution": "roundoff", "augmentation_support_force_budget_n": 1e-6},
    {"enforcement_solver": "monolithic_active_set", "complementarity_force_scale_n": 1.},
    {"d1_interior_profile": True, "d1_interior_coverage_tolerance_m2": 1e-10,
     "d1_interior_boundary_force_tolerance_n": 1e-8},
    {"q4_s3_shared_edge_tie_step_m": 1e-7, "q4_s3_shared_edge_tie_direction": (1., 0., 0.),
     "q4_s3_shared_edge_tie_relative_tolerance": 1e-5},
])
def test_unsupported_enforcement_and_geometry_fail_closed(change):
    model, definition, settings = _candidate()
    if "q4_s3_shared_edge_tie_step_m" in change:
        with pytest.raises(ValueError, match="shared-edge tie needs Jet2 Q4/S3"):
            replace(settings, **change)
        return
    selected = replace(settings, **change)
    assert not validate_general_contact(model, definition, selected).valid
    with pytest.raises(ValueError, match="unsupported"):
        GeneralS3PatchManager(model, definition, selected)


def test_admission_rejects_single_pair_wrong_formulation_and_control():
    from anyfesolver.general_contact.core import contact_scope
    from anyfesolver import solve_static_nonlinear
    model, definition, settings = _candidate()
    single = replace(definition, surfaces=(replace(definition.surfaces[0], element_ids=(1,)),
        replace(definition.surfaces[1], element_ids=(3,))))
    assert not validate_general_contact(model, single, settings).valid
    with pytest.raises(ValueError, match="unsupported"):
        with contact_scope(model, definition, settings, analysis_control="arc_length"):
            pytest.fail("unsupported control admitted")
    with pytest.raises(ValueError, match="small-rotation kinematics"):
        solve_static_nonlinear(model, contact=definition, contact_settings=settings, kinematics="corotational")
    model.mesh.elements[3] = create_shell_element(3, (5, 6, 7, 8), "elastic", thickness=.05)
    assert not validate_general_contact(model, definition, settings).valid


def test_refined_four_s3_source_two_s3_target_uses_same_profile_and_resource_cap():
    model, definition, settings = _candidate()
    model.add_node(9, .6, .1, .065)
    model.add_node(10, .6, 1.1, .065)
    for eid, ids in ((1, (1, 9, 4)), (2, (9, 10, 4)), (5, (9, 2, 10)), (6, (2, 3, 10))):
        element = create_shell_element(eid, ids, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0., 0., 1.), reference_surface_offset=.01)
        if eid in model.mesh.elements:
            model.mesh.elements[eid] = element
        else:
            model.add_element(eid, element)
    definition = replace(definition, surfaces=(replace(definition.surfaces[0], element_ids=(1, 2, 5, 6)),
        definition.surfaces[1]))
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == S3_PROJECTED_PENALTY_PROFILE
    assert len(manager.law.pressure_mode_ids) == 6
    assert not validate_general_contact(model, definition, replace(settings, maximum_candidate_pairs=4)).valid


def test_inherited_exact_clipping_vertex_branch_fails_closed():
    from anyfesolver.general_contact.s3_overlap_derivatives import UnsupportedOverlapBranch
    model, definition, settings = _case()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], normal_law="projected_penalty"),))
    manager = GeneralS3PatchManager(model, definition, replace(settings, general_s3_differentiation="jet2_s3_dev"))
    with pytest.raises(UnsupportedOverlapBranch, match="vertex lies on a clipping boundary"):
        manager.evaluate(_normal_pose(model, manager))


def test_new_profile_restart_identity_and_old_law_payload_remain_distinct():
    model, definition, settings = _candidate()
    manager = GeneralS3PatchManager(model, definition, settings)
    contract = manager.restart_contract()
    assert contract["profile"] == S3_PROJECTED_PENALTY_PROFILE
    assert contract["projected_penalty_contract"] == {
        "revision": "finite_projected_weak_gap_v1", "multipliers": "identically_zero",
        "work": "projected_weak_gap_nonpotential_v1", "storage": "support_weighted_penalty_not_work"}
    assert "pressure_tolerance" not in contract["settings"]
    old_definition = replace(definition, interactions=(replace(definition.interactions[0], normal_law="augmented"),))
    old = GeneralS3PatchManager(model, old_definition, replace(settings, pressure_tolerance=1e-5))
    assert "projected_penalty_contract" not in old.restart_contract()
    payload = old.evaluate(np.zeros(old.n_dof), tangent=False)[2].to_dict()
    assert "regularization_storage_j" not in payload and "contact_work_convention" not in payload


def _roundoff_10nm_candidate():
    model, definition, settings = _candidate()
    manager = GeneralS3PatchManager(model, definition, settings)
    u = _normal_pose(model, manager, -.015-1e-8)
    for node in range(1, 5):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        u[dofs[0]], u[dofs[1]] = .013, .007
    return model, manager, u


def test_s3_roundoff_positive_perrow_budget_is_coordinate_scaled_not_gap_scaled():
    from anyfesolver.general_contact.s3_patch_augmented import _s3_gap_operand_roundoff
    source = np.array([[.1, .1, .015-1e-8], [1.1, .1, .015-1e-8], [.1, 1.1, .015-1e-8]])
    target = np.array([[0., 0., .015], [1., 0., .015], [0., 1., .015]])
    support = np.array([.03, .12, .21])
    allowance = _s3_gap_operand_roundoff(source, target, support)
    gamma = 64*np.finfo(float).eps/(1-64*np.finfo(float).eps)
    # Both right-triangle frames have condition1; each row retains its own S.
    distance_budget = gamma*np.max(np.sum(abs(np.vstack((source, target))), axis=1))*4
    np.testing.assert_allclose(allowance, support*distance_budget, rtol=1e-15)
    assert np.all(allowance > 0.) and distance_budget < 1e-9
    print(json.dumps(dict(check="s3_roundoff_direct_positive", perrow_allowance_m3=allowance.tolist(),
        distance_budget_m=float(distance_budget), physical_gap_m=-1e-8)))


def test_s3_roundoff_10nm_translated_full_projected_work():
    model, manager, u = _roundoff_10nm_candidate()
    force, tangent, result = manager.evaluate(u)
    area = (.9-.013)*(.9-.007)
    np.testing.assert_allclose(result.numerical_pressure_pa, 1000*1e-8, rtol=1e-8, atol=0.)
    np.testing.assert_allclose(result.reference_overlap_area, area, atol=1e-13)
    np.testing.assert_allclose(result.integration_force, (0., 0., 1000*1e-8*area), rtol=1e-8, atol=1e-16)
    np.testing.assert_allclose(np.add(result.integration_force, result.target_force), 0., atol=1e-16)
    np.testing.assert_allclose(np.add(result.integration_moment, result.target_moment), 0., atol=1e-16)
    assert result.active_pressure_dofs == 4 and all(s > 0 for s in result.reference_support_m2)
    assert np.all(np.isfinite(force)) and np.all(np.isfinite(tangent.data))
    assert not manager.normal_multipliers.flags.writeable
    direction = np.zeros(manager.n_dof)
    for node in range(1, 5):
        direction[model.mesh.dof_manager.get_node_dofs(node)[2]] = .3
    for node in range(5, 9):
        direction[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.2
    # Full both-body projected work, independent scalar normal traction.
    np.testing.assert_allclose(force@direction, -1000*1e-8*area*.5, rtol=1e-8, atol=1e-16)
    kernel_path = Path(inspect.getsourcefile(type(manager.law))).resolve()
    print(json.dumps(dict(check="s3_roundoff_10nm_full_work", source_contact_force_n=result.integration_force[2],
        physical_gap_m=result.sampled_overlap_gap, generalized_work_j=float(force@direction),
        kernel=dict(path=str(kernel_path), sha256=hashlib.sha256(kernel_path.read_bytes()).hexdigest()))))


def test_s3_roundoff_admitted_perrow_value_budget_propagates_to_shared_assembly(monkeypatch):
    import anyfesolver.general_contact.s3_patch_augmented as kernel
    _, manager, u = _roundoff_10nm_candidate()
    original_derivatives, original_bound = kernel.s3_overlap_derivatives, kernel._s3_gap_operand_roundoff
    admitted, retries = [], []
    def inject_within_bound(reference, source, target, **options):
        derivative = original_derivatives(reference, source, target, **options)
        offset = .25*original_bound(source, target, derivative.reference_support)
        # Above the old cancelled-W-relative comparison; below the explicit
        # Cartesian operand allowance. No derivative/hessian values changed.
        assert np.max(offset) > 1e-9*np.max(abs(derivative.reference_weighted_gap))
        admitted.append(offset.tolist())
        return replace(derivative, reference_weighted_gap=derivative.reference_weighted_gap+offset)
    def track_retry(source, target, support):
        value = original_bound(source, target, support)
        retries.append(value.tolist())
        return value
    monkeypatch.setattr(kernel, "s3_overlap_derivatives", inject_within_bound)
    monkeypatch.setattr(kernel, "_s3_gap_operand_roundoff", track_retry)
    force, tangent, result = manager.evaluate(u)
    assert admitted and len(retries) == len(admitted)
    assert result.active_pressure_dofs == 4
    assert np.all(np.isfinite(force)) and np.all(np.isfinite(tangent.data))
    print(json.dumps(dict(check="s3_roundoff_shared_assembly_admitted", pair_count=len(retries),
        synthetic_pair_value_offsets_m3=admitted, perrow_retry_allowances_m3=retries)))


def test_s3_roundoff_material_weighted_gap_error_still_rejects(monkeypatch):
    import anyfesolver.general_contact.s3_patch_augmented as kernel
    _, manager, u = _roundoff_10nm_candidate()
    original_derivatives, original_bound = kernel.s3_overlap_derivatives, kernel._s3_gap_operand_roundoff
    retries = []
    def inject_material_error(reference, source, target, **options):
        derivative = original_derivatives(reference, source, target, **options)
        return replace(derivative, reference_weighted_gap=derivative.reference_weighted_gap+1e-6*derivative.reference_support)
    def track_retry(source, target, support):
        value = original_bound(source, target, support)
        retries.append(float(np.max(value/support)))
        return value
    monkeypatch.setattr(kernel, "s3_overlap_derivatives", inject_material_error)
    monkeypatch.setattr(kernel, "_s3_gap_operand_roundoff", track_retry)
    with pytest.raises(ValueError, match="differentiated S3 overlap disagrees with mortar geometry"):
        manager.evaluate(u)
    assert retries and max(retries) < 1e-9
    print(json.dumps(dict(check="s3_roundoff_material_error_rejected", injected_normalized_gap_error_m=1e-6,
        retry_distance_budgets_m=retries)))


@pytest.mark.parametrize("case", ["far_origin", "ill_source", "ill_target", "degenerate_source", "degenerate_target"])
def test_s3_roundoff_unresolved_origin_or_frame_rejects_directly(case):
    from anyfesolver.general_contact.s3_patch_augmented import _s3_gap_operand_roundoff
    from anyfesolver.general_contact.s3_overlap_derivatives import UnsupportedOverlapBranch
    source = np.array([[.1, .1, .015-1e-8], [1.1, .1, .015-1e-8], [.1, 1.1, .015-1e-8]])
    target = np.array([[0., 0., .015], [1., 0., .015], [0., 1., .015]])
    if case == "far_origin":
        source += (1e8, 1e8, 0.)
        target += (1e8, 1e8, 0.)
    elif case == "ill_source":
        source[2] = source[1]+(0., 1e-12, 0.)
    elif case == "ill_target":
        target[2] = target[1]+(0., 1e-12, 0.)
    elif case == "degenerate_source":
        source[2] = source[0]+2*(source[1]-source[0])
    else:
        target[2] = target[0]+(0., 0., 1.)
    with pytest.raises(UnsupportedOverlapBranch, match="frame|projection|local resolution"):
        _s3_gap_operand_roundoff(source, target, np.array([.03, .12, .21]))
