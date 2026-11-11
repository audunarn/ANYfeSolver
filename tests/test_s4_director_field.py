"""Per-element-corner S4 director validation and reconstruction checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

import anyfesolver.shell_formulations.director_field as director_field_module
from anyfesolver.shell_formulations.director_field import (
    DIRECTOR_PROVENANCE_NUMERIC_CODE,
    DirectorProvenanceCode,
    DirectorValidationLimits,
    PreparedDirectorField,
    SourceCornerSamples,
    prepare_supplied_corner_directors,
    reconstruct_corner_directors,
    validate_element_corner_directors,
)
from anyfesolver.shell_formulations.mitc4_plus_d_quality import (
    numeric_payload_fingerprint,
    q4_geometry_quality,
)


def _two_quad_plane() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        )
    )
    connectivity = np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=np.int64)
    return coordinates, connectivity


def _right_angle_fold() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        )
    )
    connectivity = np.asarray(((0, 1, 2, 3), (1, 0, 4, 5)), dtype=np.int64)
    return coordinates, connectivity


def _rotation_matrix() -> np.ndarray:
    axis = np.asarray((1.0, -2.0, 0.7), dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.71
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _reference_center_director_jacobian(
    corners: np.ndarray,
    directors: np.ndarray,
) -> float:
    """Midsurface determinant factor used by the reference continuum map."""

    derivative_xi = 0.25 * np.asarray((-1.0, 1.0, 1.0, -1.0))
    derivative_eta = 0.25 * np.asarray((-1.0, -1.0, 1.0, 1.0))
    covariant_xi = derivative_xi @ corners
    covariant_eta = derivative_eta @ corners
    center_director = np.mean(directors, axis=0)
    return float(np.dot(np.cross(covariant_xi, covariant_eta), center_director))


def test_supplied_directors_are_strictly_numeric_normalized_and_immutable() -> None:
    coordinates, connectivity = _two_quad_plane()
    corners = coordinates[connectivity]
    directors = np.zeros_like(corners)
    directors[:, :, 2] = 1.0
    prepared = prepare_supplied_corner_directors(
        corners,
        directors,
        provenance=DirectorProvenanceCode.SOURCE_SURFACE_EXACT,
    )
    assert prepared.directors.shape == (2, 4, 3)
    assert not prepared.directors.flags.writeable
    assert set(np.unique(prepared.provenance_codes)) == {
        DIRECTOR_PROVENANCE_NUMERIC_CODE[DirectorProvenanceCode.SOURCE_SURFACE_EXACT]
    }
    assert prepared.provenance_labels[0] == ("source_surface_exact",) * 4
    with pytest.raises(ValueError):
        prepared.directors[0, 0, 0] = 2.0

    invalid = directors.copy()
    invalid[0, 0] *= 2.0
    with pytest.raises(ValueError, match="normalized"):
        prepare_supplied_corner_directors(corners, invalid)
    invalid = directors.copy()
    invalid[0, 0] *= -1.0
    with pytest.raises(ValueError, match="orientation"):
        prepare_supplied_corner_directors(corners, invalid)


def test_reconstruction_is_translation_rotation_and_fe_numbering_invariant() -> None:
    coordinates, connectivity = _two_quad_plane()
    baseline = reconstruct_corner_directors(
        coordinates,
        connectivity,
        part_indices=(4, 4),
        sheet_indices=(8, 8),
    )
    rotation = _rotation_matrix()
    translated_rotated = coordinates @ rotation.T + np.asarray((1.0e5, -2.0e5, 3.0e5))
    transformed = reconstruct_corner_directors(
        translated_rotated,
        connectivity,
        part_indices=(4, 4),
        sheet_indices=(8, 8),
    )
    np.testing.assert_allclose(transformed.directors, baseline.directors @ rotation.T, atol=2.0e-11)

    element_order = np.asarray((1, 0))
    reordered = reconstruct_corner_directors(
        coordinates,
        connectivity[element_order],
        part_indices=np.asarray((4, 4))[element_order],
        sheet_indices=np.asarray((8, 8))[element_order],
    )
    np.testing.assert_allclose(reordered.directors[np.argsort(element_order)], baseline.directors, atol=1.0e-14)

    new_to_old = np.asarray((2, 0, 5, 1, 4, 3))
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(new_to_old.size)
    renumbered = reconstruct_corner_directors(
        coordinates[new_to_old],
        old_to_new[connectivity],
        part_indices=(4, 4),
        sheet_indices=(8, 8),
    )
    np.testing.assert_allclose(renumbered.directors, baseline.directors, atol=1.0e-14)


def test_sharp_fold_keeps_distinct_directors_at_one_shared_global_node() -> None:
    coordinates, connectivity = _right_angle_fold()
    field = reconstruct_corner_directors(coordinates, connectivity)
    np.testing.assert_allclose(field.directors[0, 0], (0.0, 0.0, 1.0), atol=1.0e-14)
    np.testing.assert_allclose(field.directors[1, 1], (0.0, 1.0, 0.0), atol=1.0e-14)
    assert field.quality.angular_crease_edge_count == 1
    assert not np.allclose(field.directors[0, 0], field.directors[1, 1])


@pytest.mark.parametrize(
    ("part_indices", "sheet_indices", "continuity_indices"),
    [
        ((1, 2), (3, 3), (4, 4)),
        ((1, 1), (3, 9), (4, 4)),
        ((1, 1), (3, 3), (4, 7)),
    ],
)
def test_part_sheet_and_declared_continuity_boundaries_never_average(
    part_indices: tuple[int, int],
    sheet_indices: tuple[int, int],
    continuity_indices: tuple[int, int],
) -> None:
    coordinates, connectivity = _right_angle_fold()
    permissive = DirectorValidationLimits(crease_angle_degrees=180.0)
    field = reconstruct_corner_directors(
        coordinates,
        connectivity,
        part_indices=part_indices,
        sheet_indices=sheet_indices,
        continuity_indices=continuity_indices,
        limits=permissive,
    )
    np.testing.assert_allclose(field.directors[0, 0], (0.0, 0.0, 1.0), atol=1.0e-14)
    np.testing.assert_allclose(field.directors[1, 1], (0.0, 1.0, 0.0), atol=1.0e-14)
    assert field.quality.region_boundary_edge_count == 1


def test_smooth_fan_averages_only_when_no_crease_or_source_boundary_blocks_it() -> None:
    coordinates, connectivity = _right_angle_fold()
    limits = DirectorValidationLimits(
        maximum_corner_spread_degrees=90.0,
        crease_angle_degrees=180.0,
    )
    field = reconstruct_corner_directors(coordinates, connectivity, limits=limits)
    expected = np.asarray((0.0, 1.0, 1.0)) / math.sqrt(2.0)
    np.testing.assert_allclose(field.directors[0, 0], expected, atol=1.0e-14)
    np.testing.assert_allclose(field.directors[1, 1], expected, atol=1.0e-14)
    assert field.quality.smooth_interior_edge_count == 1
    assert field.quality.maximum_smoothing_component_size == 2

    blocked = reconstruct_corner_directors(
        coordinates,
        connectivity,
        crease_edges=((0, 1),),
        limits=limits,
    )
    np.testing.assert_allclose(blocked.directors[0, 0], (0.0, 0.0, 1.0), atol=1.0e-14)
    np.testing.assert_allclose(blocked.directors[1, 1], (0.0, 1.0, 0.0), atol=1.0e-14)
    assert blocked.quality.declared_crease_edge_count == 1


@pytest.mark.parametrize(
    ("connectivity", "source_face_use_orientation"),
    [
        ((0, 1, 2, 3), 1),
        ((3, 2, 1, 0), -1),
    ],
)
def test_every_prepared_field_is_directly_consumable_by_reference_jacobian_convention(
    connectivity: tuple[int, int, int, int],
    source_face_use_orientation: int,
) -> None:
    coordinates = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    corners = coordinates[np.asarray(connectivity)]
    reconstructed = reconstruct_corner_directors(
        coordinates,
        (connectivity,),
        source_face_use_orientation_signs=(source_face_use_orientation,),
    )
    supplied = prepare_supplied_corner_directors(
        corners,
        reconstructed.directors[0],
        source_face_use_orientation_signs=(source_face_use_orientation,),
        provenance=DirectorProvenanceCode.SOURCE_SURFACE_SAMPLED,
    )
    connectivity_normal = np.asarray(q4_geometry_quality(corners).unit_normal)
    for prepared in (reconstructed, supplied):
        np.testing.assert_allclose(prepared.directors[0], np.tile(connectivity_normal, (4, 1)))
        jacobian = _reference_center_director_jacobian(corners, prepared.directors[0])
        assert jacobian > 0.0
        assert prepared.element_quality[0].center_director_jacobian == pytest.approx(jacobian)

    with pytest.raises(ValueError, match="positive center reference Jacobian"):
        prepare_supplied_corner_directors(
            corners,
            -reconstructed.directors[0],
            source_face_use_orientation_signs=(source_face_use_orientation,),
        )


def test_degenerate_geometry_and_excessive_director_spread_fail_closed() -> None:
    with pytest.raises(ValueError, match="characteristic length|zero or numerically unresolved"):
        reconstruct_corner_directors(
            ((0.0, 0.0, 0.0),) * 4,
            ((0, 1, 2, 3),),
        )

    corners = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    directors = np.asarray(((0.0, 0.0, 1.0), (0.5, 0.0, math.sqrt(0.75)), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="angular variation"):
        validate_element_corner_directors(
            corners,
            directors,
            limits=DirectorValidationLimits(maximum_corner_spread_degrees=20.0),
        )


def test_numeric_reference_fingerprint_is_deterministic_and_revision_sensitive() -> None:
    coordinates, connectivity = _two_quad_plane()
    field = reconstruct_corner_directors(coordinates, connectivity)
    assert field.numeric_fingerprint == reconstruct_corner_directors(coordinates, connectivity).numeric_fingerprint
    changed = field.directors.copy()
    changed[0, 0, 0] = 1.0e-12
    assert numeric_payload_fingerprint(coordinates, field.directors) != numeric_payload_fingerprint(
        coordinates, changed
    )


def test_numeric_fingerprint_preserves_integer_signedness_and_rejects_nonfinite_float() -> None:
    unsigned_maximum = np.asarray((np.iinfo(np.uint64).max,), dtype=np.uint64)
    signed_sentinel = np.asarray((-1,), dtype=np.int64)
    assert numeric_payload_fingerprint(unsigned_maximum) != numeric_payload_fingerprint(
        signed_sentinel
    )
    with pytest.raises(ValueError, match="must remain finite"):
        numeric_payload_fingerprint(np.asarray((np.nan,), dtype=np.float64))


def _internal_prepared_copy(
    valid: PreparedDirectorField,
    **changes: object,
) -> PreparedDirectorField:
    values: dict[str, object] = {
        "directors": valid.directors,
        "provenance_codes": valid.provenance_codes,
        "element_quality": valid.element_quality,
        "quality": valid.quality,
        "diagnostics": valid.diagnostics,
        "numeric_fingerprint": valid.numeric_fingerprint,
    }
    values.update(changes)
    return PreparedDirectorField(
        **values,
        _factory_token=director_field_module._PREPARED_DIRECTOR_FACTORY_TOKEN,
    )


def test_direct_prepared_field_construction_is_factory_only() -> None:
    coordinates, connectivity = _two_quad_plane()
    valid = reconstruct_corner_directors(coordinates, connectivity)
    with pytest.raises(TypeError, match="direct construction is unsupported"):
        PreparedDirectorField(
            valid.directors,
            valid.provenance_codes,
            valid.element_quality,
            valid.quality,
            valid.diagnostics,
            valid.numeric_fingerprint,
        )


def test_internal_prepared_field_validation_rejects_hostile_records() -> None:
    coordinates, connectivity = _two_quad_plane()
    valid = reconstruct_corner_directors(coordinates, connectivity)

    for invalid_value in (np.nan, np.inf, -np.inf):
        invalid_directors = valid.directors.copy()
        invalid_directors[0, 0, 0] = invalid_value
        with pytest.raises(ValueError, match="must remain finite"):
            _internal_prepared_copy(valid, directors=invalid_directors)

    zero_directors = valid.directors.copy()
    zero_directors[0, 0] = 0.0
    with pytest.raises(ValueError, match="finite nonzero length"):
        _internal_prepared_copy(valid, directors=zero_directors)

    for narrowed in (
        np.full(valid.provenance_codes.shape, 256, dtype=np.uint16),
        np.full(valid.provenance_codes.shape, -1, dtype=np.int16),
    ):
        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            _internal_prepared_copy(valid, provenance_codes=narrowed)

    unregistered = np.full(valid.provenance_codes.shape, 6, dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown numeric director provenance"):
        _internal_prepared_copy(valid, provenance_codes=unregistered)

    for changes, message in (
        ({"element_quality": (object(),) * len(valid.element_quality)}, "CornerDirectorQuality"),
        ({"quality": object()}, "DirectorFieldQuality"),
        ({"diagnostics": (object(),)}, "nonempty strings"),
        ({"numeric_fingerprint": object()}, "canonical lowercase"),
        ({"numeric_fingerprint": "A" * 64}, "canonical lowercase"),
        ({"numeric_fingerprint": "0" * 63}, "canonical lowercase"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            _internal_prepared_copy(valid, **changes)


def test_director_validation_limits_reject_boolean_and_nonscalar_values() -> None:
    fields = (
        "normalization_atol",
        "minimum_geometry_alignment",
        "maximum_corner_spread_degrees",
        "crease_angle_degrees",
    )
    for field_name in fields:
        for hostile in (True, np.bool_(False), np.asarray(1.0), np.asarray([1.0]), object(), "1"):
            with pytest.raises(TypeError, match="real numeric scalar"):
                DirectorValidationLimits(**{field_name: hostile})


def test_invalid_limit_type_cannot_relax_director_normalization() -> None:
    coordinates, connectivity = _two_quad_plane()
    corners = coordinates[connectivity[:1]]
    directors = np.tile((0.0, 0.0, 2.0), (1, 4, 1))
    with pytest.raises(ValueError, match="maximum norm error"):
        prepare_supplied_corner_directors(corners, directors)
    with pytest.raises(TypeError, match="real numeric scalar"):
        prepare_supplied_corner_directors(
            corners,
            directors,
            limits=DirectorValidationLimits(normalization_atol=True),
        )


def test_optional_source_uv_and_tangent_samples_are_cold_numeric_evidence() -> None:
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    tangent_1 = np.tile((1.0, 0.0, 0.0), (4, 1))
    tangent_2 = np.tile((0.0, 1.0, 0.0), (4, 1))
    samples = SourceCornerSamples(uv, tangent_1, tangent_2)
    assert samples.element_count == 1
    assert not samples.source_uv.flags.writeable
    assert len(samples.numeric_fingerprint) == 64
    with pytest.raises(ValueError, match="supplied together"):
        SourceCornerSamples(source_tangent_1=tangent_1)
    with pytest.raises(ValueError, match="linearly independent"):
        SourceCornerSamples(source_tangent_1=tangent_1, source_tangent_2=tangent_1)


def test_source_corner_tangent_validation_is_overflow_safe() -> None:
    huge = np.finfo(np.float64).max
    tangent_1 = np.tile((huge, huge, 0.0), (4, 1))
    tangent_2 = np.tile((0.0, huge, huge), (4, 1))
    samples = SourceCornerSamples(
        source_tangent_1=tangent_1,
        source_tangent_2=tangent_2,
    )
    assert samples.element_count == 1
    assert np.all(np.isfinite(samples.source_tangent_1))

    with pytest.raises(ValueError, match="linearly independent"):
        SourceCornerSamples(
            source_tangent_1=tangent_1,
            source_tangent_2=tangent_1,
        )


def test_q4_quality_rejects_concave_or_orientation_reversing_mapping() -> None:
    concave = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.2, 0.0), (0.0, 1.0, 0.0))
    )
    with pytest.raises(ValueError, match="Jacobian"):
        q4_geometry_quality(concave)

    huge = np.finfo(np.float64).max
    overflow_geometry = np.asarray(
        ((huge, 0.0, 0.0), (-huge, 0.0, 0.0), (-huge, 1.0, 0.0), (huge, 1.0, 0.0))
    )
    with pytest.raises(ValueError, match="overflowed"):
        q4_geometry_quality(overflow_geometry)

    overflow_scale = np.asarray(
        ((0.0, 0.0, 0.0), (1.0e154, 0.0, 0.0), (2.0e154, 0.0, 0.0), (1.0e154, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="tolerance scale overflowed"):
        q4_geometry_quality(overflow_scale)


def test_mesh_handoff_indices_must_be_integer_arrays() -> None:
    coordinates, connectivity = _two_quad_plane()
    with pytest.raises(ValueError, match="compact integer"):
        reconstruct_corner_directors(coordinates, connectivity, sheet_indices=(1.0, 1.0))
    with pytest.raises(ValueError, match="integer node indices"):
        reconstruct_corner_directors(coordinates, connectivity.astype(float))
    with pytest.raises(ValueError, match="-1.*or nonnegative"):
        reconstruct_corner_directors(coordinates, connectivity, sheet_indices=(-2, -2))
    unsigned_maximum = np.full(2, np.iinfo(np.uint64).max, dtype=np.uint64)
    with pytest.raises(ValueError, match="compact integer"):
        reconstruct_corner_directors(
            coordinates,
            connectivity,
            sheet_indices=unsigned_maximum,
        )
    with pytest.raises(ValueError, match="compact integer"):
        reconstruct_corner_directors(
            coordinates,
            connectivity,
            sheet_indices=(1 << 100, 1 << 100),
        )
    hostile_connectivity = connectivity.astype(np.uint64)
    hostile_connectivity[0, 0] = np.iinfo(np.uint64).max
    with pytest.raises(ValueError, match="nonnegative integer node indices"):
        reconstruct_corner_directors(coordinates, hostile_connectivity)
    with pytest.raises(ValueError, match="true integer node indices"):
        reconstruct_corner_directors(coordinates, connectivity, crease_edges=((0.5, 1),))
    with pytest.raises(ValueError, match="not booleans"):
        reconstruct_corner_directors(coordinates, connectivity, crease_edges=((False, 1),))
    with pytest.raises(ValueError, match=r"lie in \[-1, 1\]"):
        reconstruct_corner_directors(
            coordinates,
            connectivity,
            source_face_use_orientation_signs=unsigned_maximum,
        )
