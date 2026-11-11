"""Independent source-bin geometry/work checks; no coupled solver run."""

from dataclasses import replace

import numpy as np
import pytest

from docs.d1_review_support.d1_radial_source_bin_work import (
    PhysicalSkinFace, check_guided_source_bins, conversion_bound,
    source_bin_geometry_work, vertical_gap_factor,
)


def _faces(warped=False):
    coordinates = np.array([(x, y, .04) for y in (.1, .6, 1.1)
                            for x in (.1, .6, 1.1)])
    current = coordinates.copy()
    current[:, 2] -= .01
    if warped:
        current[:, 2] += .02*current[:, 0]+.03*current[:, 0]*current[:, 1]
    source = []
    for j in range(2):
        for i in range(2):
            indices = np.array((3*j+i, 3*j+i+1, 3*(j+1)+i+1, 3*(j+1)+i))
            source.append(PhysicalSkinFace(tuple(indices+1), coordinates[indices], current[indices]))
    target_reference = np.array(((0., -.13, 0.), (1.3, -.13, 0.),
                                 (1.3, 1.3, 0.), (0., 1.3, 0.), (.63, .57, 0.)))
    target_current = target_reference.copy()
    if warped:
        target_current[:, 2] = .01+.015*target_current[:, 0]-.02*target_current[:, 1]
        target_current[4, 2] += .002
    target = tuple(PhysicalSkinFace(tuple(101+np.array(indices)), target_reference[list(indices)],
                                    target_current[list(indices)])
                   for indices in ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    support = dict.fromkeys(range(1, 10), 0.)
    for face in source:
        for node in face.node_ids:
            support[node] += .25/4
    force = {node: support[node]*(1.+x+.2*y) for node, (x, y, _) in enumerate(coordinates, 1)}
    return tuple(source), target, force


def _check(source, target, force, **options):
    return source_bin_geometry_work(source, target, force,
                                    quadrature_tolerance_n=1e-11,
                                    maximum_triangles=2048, **options)


def test_flat_force_bins_equal_independent_affine_pressure_integrals():
    result = _check(*_faces())
    expected = np.array([[.25*(1.+x+.2*y) for x in (.35, .85)] for y in (.35, .85)])
    np.testing.assert_allclose(result["reference_pressure_bins_n"], expected, rtol=1e-14)
    np.testing.assert_allclose(result["independent_geometry_work_bins_n"], expected, rtol=1e-13)
    assert result["production_comparison"]["status"] == "not_checked"


@pytest.mark.parametrize("warped", (False, True))
def test_projection_gap_factor_matches_independent_line_plane_difference(warped):
    # Use an actual bilinear surface, with its corner and interior normals.
    q = np.array(((.1, .1, 0.), (.6, .1, 0.), (.6, .6, 0.), (.1, .6, 0.)))
    q[:, 2] = .08*q[:, 0]+.12*q[:, 1]
    if warped:
        q[:, 2] += .2*q[:, 0]*q[:, 1]
    u, v = .72, .38
    cross = q[2]-q[1]-q[3]+q[0]
    ju, jv = q[1]-q[0]+v*cross, q[3]-q[0]+u*cross
    ns, n0 = np.cross(ju, jv), np.cross(q[1]-q[0], q[3]-q[0])
    ns /= np.linalg.norm(ns); n0 /= np.linalg.norm(n0)
    st = np.array((.2, -.1)); nt = np.r_[-st, 1.]
    point = ((1-u)*(1-v)*q[0]+u*(1-v)*q[1]+u*v*q[2]+(1-u)*v*q[3])
    def gap(height):
        moved = point+np.array((0., 0., height))
        # Find the target hit independently by intersecting the n0 ray
        # with a target plane, then measure along the physical ns.
        hit = moved-(nt @ moved)/(nt @ n0)*n0
        return ns @ (moved-hit)
    h = 1e-5
    observed = (gap(h)-gap(-h))/(2*h)
    ss, s0 = -ns[:2]/ns[2], -n0[:2]/n0[2]
    factor = vertical_gap_factor(ss, s0, st)
    assert factor == pytest.approx(observed, rel=2e-12)
    if warped:
        proposed_planar_factor = np.sqrt(1.+ss@ss)/(1.+st@ss)
        assert abs(proposed_planar_factor-factor) > 1e-5


def test_corrected_bound_covers_both_sides_and_old_upper_only_bound_does_not():
    bound = conversion_bound(.3, .2)
    angles = np.linspace(0., 2*np.pi, 17)
    slopes = [.3*np.array((np.cos(a), np.sin(a))) for a in angles]
    targets = [.2*np.array((np.cos(a), np.sin(a))) for a in angles]
    values = [vertical_gap_factor(ss, s0, st) for ss in slopes for s0 in slopes for st in targets]
    assert min(values) >= bound["kappa_lower"]
    assert max(values) <= bound["kappa_upper"]
    lower = vertical_gap_factor((-.3, 0.), (.3, 0.), (0., 0.))
    old_alpha = np.sqrt(1.+.3**2)-1.
    assert 1.-lower > old_alpha
    assert 1.-lower <= conversion_bound(.3, 0.)["relative_conversion_bound"]


def test_warped_geometry_bins_obey_the_independent_two_sided_conversion_bound():
    result = _check(*_faces(warped=True))
    difference = abs(np.array(result["independent_geometry_work_bins_n"])
                     -result["reference_pressure_bins_n"])
    assert np.all(difference <= np.array(result["conversion_allowance_bins_n"])
                  +result["estimated_quadrature_error_bins_n"])
    assert result["slope_conversion_bound"]["source_slope_max"] > 0.


def test_warped_independent_bins_match_production_pair_work_before_assembly():
    from anyfesolver.general_contact.s3_patch_mortar_energy import (
        MovingS3PatchPrimalMortarPenalty, S3PatchFace,
    )
    source, target, force = _faces(warped=True)
    geometry = MovingS3PatchPrimalMortarPenalty(
        tuple(S3PatchFace(face.node_ids, face.reference_positions) for face in source),
        tuple(S3PatchFace(face.node_ids, face.reference_positions) for face in target),
        1000., difference_step=1e-6, differentiation="jet2_q4_s3",
        maximum_candidate_pairs=16, maximum_energy_evaluations=1500,
        q4_source_area_tolerance=1e-10, q4_source_weighted_gap_tolerance=1e-11,
        maximum_q4_source_integration_triangles=256,
        q4_source_derivative_tolerances=(1e-8,)*4)
    result = _check(source, target, force, production_geometry=geometry)
    comparison = result["production_comparison"]
    assert comparison["status"] == "measured", comparison
    np.testing.assert_allclose(comparison["gradient_minus_independent_bins_n"], 0., atol=2e-10)
    np.testing.assert_allclose(comparison["moving_normalization_term_bins_n"], 0., atol=2e-10)
    assert comparison["pair_energy_evaluations"] <= geometry.maximum_energy_evaluations
    assert comparison["straight_jet2_work_branch_validated"]
    assert abs(comparison["sum_of_bins_minus_validated_total_n"]) < 1e-12


@pytest.mark.parametrize("change, message", (("xy", "fixed physical XY"),
                                            ("reference", "flat-reference"),
                                            ("negative", "nonnegative"),
                                            ("cut", "outside target union|boundary cuts"),
                                            ("bin", "principal edges")))
def test_invalid_identity_scope_is_rejected(change, message):
    source, target, force = _faces()
    options = {}
    if change == "xy":
        q = source[0].current_positions.copy(); q[0, 0] += .01
        source = (replace(source[0], current_positions=q), *source[1:])
    elif change == "reference":
        q = [face.reference_positions.copy() for face in source]
        for points in q:
            points[:, 2] += .01*points[:, 0]
        source = tuple(replace(face, reference_positions=points) for face, points in zip(source, q))
    elif change == "negative":
        force[1] = -1.
    elif change == "cut":
        # Positive cut below the area-integration tolerance still fails scope.
        target = tuple(replace(face, reference_positions=face.reference_positions+[.100000000001, 0., 0.],
                               current_positions=face.current_positions+[.100000000001, 0., 0.])
                       for face in target)
    else:
        options["bin_edges"] = (.1, .5, 1.1)
    with pytest.raises(ValueError, match=message):
        _check(source, target, force, **options)


def test_scope_rejects_triangle_budget_exhaustion():
    source, target, force = _faces()
    with pytest.raises(MemoryError, match="triangle budget"):
        source_bin_geometry_work(source, target, force,
                                 quadrature_tolerance_n=1e-11, maximum_triangles=1)


def test_target_overlap_cannot_mask_missing_coverage():
    source, target, force = _faces()
    with pytest.raises(ValueError, match="target P1 faces overlap"):
        _check(source, (*target, target[0]), force)


def test_discontinuous_shared_target_pose_is_rejected():
    source, target, force = _faces()
    points = target[0].current_positions.copy(); points[2, 2] += .001
    target = (replace(target[0], current_positions=points), *target[1:])
    with pytest.raises(ValueError, match="discontinuous"):
        _check(source, target, force)


def test_missing_production_gradient_check_is_reported_explicitly():
    from types import SimpleNamespace
    result = _check(*_faces(), production_geometry=SimpleNamespace(differentiation="pair_local"))
    assert result["production_comparison"]["status"] == "not_checked"
    assert "requires Q4/S3 Jet2" in result["production_comparison"]["reason"]
    assert "gradient_minus_independent_bins_n" not in result["production_comparison"]


@pytest.mark.parametrize("source, target", ((1., 0.), (.5, 2.), (np.nan, 0.)))
def test_nonfinite_or_unresolved_conversion_bound_is_rejected(source, target):
    with pytest.raises(ValueError, match="slope bound requires"):
        conversion_bound(source, target)


def test_saved_model_wrapper_reconstructs_skins_and_rejects_director_motion():
    from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case
    model, definition, _settings, _program = build_case(
        4, source_rows=4, full_width_target=True, target_right_x_m=1.3)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    result = check_guided_source_bins(model, definition, displacement,
                                     dict.fromkeys(range(1, 26), .001),
                                     quadrature_tolerance_n=1e-11, maximum_triangles=2048)
    assert sum(map(sum, result["reference_pressure_bins_n"])) == pytest.approx(.025)
    displacement[model.mesh.dof_manager.get_node_dofs(1)[3]] = 1e-5
    with pytest.raises(ValueError, match="fixed XY and directors"):
        check_guided_source_bins(model, definition, displacement,
                                 dict.fromkeys(range(1, 26), .001),
                                 quadrature_tolerance_n=1e-11, maximum_triangles=2048)
