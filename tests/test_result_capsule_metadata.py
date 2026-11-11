"""Focused tests for the shared once-per-model recovery metadata export."""
import math

import numpy as np
import pytest

from anyfesolver import (BoundaryCondition, FixedSupport, LoadCase, PatternCase,
                       RecoveryConfig, export_recovery_capsule,
                       export_recovery_capsule_to_sink, prepare_static_response_evaluator)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver_results import CapsuleValidationError, validate_capsule
from test_prepared_static_evaluator import _bar, _case, _shell_case, _shell_model
from test_result_capsule_export import _rotated_shell_model


def _shell_evaluator(model=None, *, stress_execution='certified_library', pressure=2.0):
    model = _shell_model() if model is None else model
    pattern = _shell_case(1, pressure=pressure).load_case
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    return prepare_static_response_evaluator(
        model, {'p': pattern}, recovery_config=recovery,
        stress_execution=stress_execution)


def _owner_of(evaluator, element_id):
    recipe = evaluator._snapshot.recovery_compilation.elements[element_id].local_recipe
    return recipe.execution_owner


def test_rotated_s3_local_frame_orthonormal_and_source_exact():
    model = _rotated_shell_model()
    with _shell_evaluator(model) as evaluator:
        capsule = export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.25})],
                                          model_identity='rotated-shell',
                                          mesh_identity='rotated-shell-mesh')
        from anyfesolver.e4_pl_s3_element import triangle_frame
        expected = {}
        for element_id in (1, 2):
            element, mesh, _material = _owner_of(evaluator, element_id)
            frame, _local, _quality = triangle_frame(
                element.get_node_coordinates(mesh), element.reference_normal)
            expected[element_id] = np.asarray(frame, dtype=np.float64)
    validate_capsule(capsule)
    frames = capsule['model_recovery_metadata']['frames']
    assert frames['convention'] == 'local_axes_as_columns_in_global_coordinates'
    assert sorted(frames['elements']) == ['1', '2']
    for element_id, source_frame in expected.items():
        exported = np.asarray(frames['elements'][str(element_id)], dtype=np.float64)
        # Source-exact: bitwise the frame the element's own recovery uses.
        np.testing.assert_array_equal(exported, source_frame)
        # Orthonormal, proper rotation.
        np.testing.assert_allclose(exported.T @ exported, np.eye(3), rtol=0.0, atol=1e-12)
        assert np.linalg.det(exported) > 0.0
        # The rotated fixture really rotates: not the identity frame.
        assert not np.allclose(exported, np.eye(3))


def test_s3_native_sampling_coordinates_source_exact_and_order():
    with _shell_evaluator() as evaluator:
        capsule = export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.25})],
                                          model_identity='shell-model',
                                          mesh_identity='shell-mesh')
        element, _mesh, _material = _owner_of(evaluator, 1)
        points = np.asarray(element.gauss_points, dtype=np.float64)
        weights = np.asarray(element.gauss_weights, dtype=np.float64)
    validate_capsule(capsule)
    metadata = capsule['model_recovery_metadata']
    assert set(metadata['sampling_rules']) == {'rule-0'}
    rule = metadata['sampling_rules']['rule-0']
    assert rule['kind'] == 'triangle_gauss_stations'
    assert rule['coordinate_system'] == 'natural_triangle_r_s'
    assert rule['order'] == 'source_native_station_order'
    assert rule['station_count'] == 7
    # Actual native coordinates and weights, source-exact and in source order.
    np.testing.assert_array_equal(np.asarray(rule['coordinates'], dtype=np.float64), points)
    np.testing.assert_array_equal(np.asarray(rule['weights'], dtype=np.float64), weights)
    # Shared once per model: both elements reference the same deduplicated rule.
    bindings = metadata['element_bindings']
    assert bindings['1']['sampling_rule'] == bindings['2']['sampling_rule'] == 'rule-0'
    assert bindings['1']['sample_count'] == bindings['2']['sample_count'] == 7
    assert bindings['1']['frame'] is True and bindings['2']['frame'] is True
    # Descriptors carry compact references into the shared tables.
    descriptors = {descriptor['key']: descriptor for descriptor in capsule['descriptors']}
    assert descriptors['membrane_xx']['metadata']['sampling_rules'] == ['rule-0']
    assert descriptors['membrane_xx']['metadata']['local_frames'] == 'model_recovery_metadata.frames'


def test_beam2_frame_and_explicit_section_scalar_rule():
    model = _bar()
    pattern = _case(1).load_case
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    cases = [PatternCase('a', {'p': 1.0})]
    with prepare_static_response_evaluator(model, {'p': pattern},
                                           recovery_config=recovery) as evaluator:
        capsule = export_recovery_capsule(evaluator, cases, model_identity='bar',
                                          mesh_identity='bar-mesh')
        element, mesh, _material = _owner_of(evaluator, 1)
        from anyfesolver.elements import _beam_rotation_matrix
        coordinates = np.asarray(element.get_node_coordinates(mesh), dtype=np.float64)
        axis = coordinates[1] - coordinates[0]
        expected = _beam_rotation_matrix(axis / np.linalg.norm(axis), element._orientation)
    validate_capsule(capsule)
    metadata = capsule['model_recovery_metadata']
    frame = np.asarray(metadata['frames']['elements']['1'], dtype=np.float64)
    np.testing.assert_array_equal(frame, np.asarray(expected, dtype=np.float64))
    rule = metadata['sampling_rules']['rule-0']
    assert rule['kind'] == 'element_section_scalar'
    # No native station coordinates exist for the beam2 envelope: the reason
    # is explicit, never an invented centroid.
    assert rule['coordinates'] is None
    assert rule['coordinates_unavailable_reason']
    assert metadata['element_bindings']['1']['sample_count'] == 1


def _beam_capsule_and_owner(element_type, *, section=None, positions=None):
    """Export a small solvable beam model through the real evaluator.

    Uses the actual private prepared owner captured by the snapshot, never
    the mutable live model.  Default geometry comes from the shared
    generalized-beam test fixture.
    """
    if positions is None:
        from test_generalized_beam_sections import _model
        model, _element = _model(element_type, section=section)
    else:
        model = FEModel('beam-metadata')
        model.add_material('dummy', 210.0e9, 0.3, density=7850.0)
        for node_id, position in enumerate(positions, start=1):
            model.add_node(node_id, *position)
        model.add_element(1, element_type(1, list(range(1, len(positions) + 1)),
                                          'dummy', {}, section=section))
    if element_type is QuadraticBeamElement:
        model.add_boundary_condition(FixedSupport('fixed', [1, 3]))
        load = LoadCase('beam-load')
        load.add_nodal_load(2, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        model.add_boundary_condition(FixedSupport('fixed', [1]))
        model.add_boundary_condition(BoundaryCondition(
            'slide', [2], {'uy': 0.0, 'uz': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0}))
        load = LoadCase('beam-load')
        load.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    with prepare_static_response_evaluator(model, {'p': load},
                                           recovery_config=recovery) as evaluator:
        capsule = export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.0})],
                                          model_identity='beam',
                                          mesh_identity='beam-mesh')
        owner = _owner_of(evaluator, 1)
    validate_capsule(capsule)
    return capsule, owner


def test_quadratic_beam_local_frame_is_end_chord_source_exact():
    # Admitted middle-node imperfection (within the source straightness
    # tolerance): the frame must be oriented on the end-node chord exactly
    # as the element's own recovery does, not on the middle node.
    positions = ((0.0, 0.0, 0.0), (0.5, 5.0e-9, 0.0), (1.0, 0.0, 0.0))
    capsule, (element, mesh, _material) = _beam_capsule_and_owner(
        QuadraticBeamElement, positions=positions)
    coordinates = np.asarray(element.get_node_coordinates(mesh), dtype=np.float64)
    _length, transform = element._beam_frame_and_transform(coordinates)
    frame = np.asarray(capsule['model_recovery_metadata']['frames']['elements']['1'],
                       dtype=np.float64)
    # Source-exact: the exported frame is the transpose of the recovery
    # transform's leading 3x3 block (local axes as columns in global
    # coordinates), built from coords[2] - coords[0].
    np.testing.assert_array_equal(
        frame, np.asarray(transform[:3, :3], dtype=np.float64).T)
    # The chord axis is exactly global x; the imperfection must not leak.
    np.testing.assert_array_equal(frame[:, 0], np.array([1.0, 0.0, 0.0]))


def test_beam_sampling_rules_match_native_recovery_field_shapes():
    from anyfesolver.beam_sections import GeneralizedBeamSection
    from anyfesolver.elements import BeamElement
    from test_generalized_beam_sections import _stiffness

    configurations = [
        (BeamElement, None, 'element_section_scalar', None),
        (QuadraticBeamElement, None, 'element_section_scalar', None),
        (BeamElement, GeneralizedBeamSection(_stiffness()),
         'beam_natural_xi_stations', (-1.0, 0.0, 1.0)),
        (QuadraticBeamElement, GeneralizedBeamSection(_stiffness()),
         'beam_natural_xi_stations', 'source_gauss_points'),
    ]
    for element_type, section, kind, expected_coordinates in configurations:
        capsule, (element, mesh, material) = _beam_capsule_and_owner(
            element_type, section=section)
        metadata = capsule['model_recovery_metadata']
        binding = metadata['element_bindings']['1']
        assert binding['frame'] is True
        rule = metadata['sampling_rules'][binding['sampling_rule']]
        assert rule['kind'] == kind
        assert rule['order'] == 'source_native_station_order'
        if expected_coordinates is None:
            # Ordinary beam recovery is one scalar section envelope per
            # element; no native station coordinates exist and the reason
            # is explicit (the three Gauss points are not recovery
            # stations, not even for the quadratic beam).
            assert rule['coordinates'] is None
            assert rule['coordinates_unavailable_reason']
            assert binding['sample_count'] == 1
        else:
            if expected_coordinates == 'source_gauss_points':
                expected_coordinates = tuple(
                    float(station) for station in element.GAUSS_POINTS)
            np.testing.assert_array_equal(
                np.asarray(rule['coordinates'], dtype=np.float64),
                np.asarray(expected_coordinates, dtype=np.float64))
            assert rule['station_count'] == 3
            assert binding['sample_count'] == 3
        # The rule matches the native recovery field shapes of the same
        # captured owner: scalar envelopes for ordinary beams, three
        # station-resolved rows for generalized sections.
        fields = element.compute_stresses(
            mesh, np.zeros(element.total_dofs), material)
        numeric = [np.asarray(value) for value in fields.values()
                   if isinstance(value, (np.ndarray, float, np.floating))]
        assert numeric
        if section is None:
            assert all(value.shape == () for value in numeric)
        else:
            assert all(value.shape[0] == 3 for value in numeric
                       if value.ndim >= 1)


def test_original_model_mutation_does_not_change_captured_metadata():
    model = _shell_model()
    with _shell_evaluator(model) as evaluator:
        before = export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.0})],
                                         model_identity='m', mesh_identity='x')
        # Mutate the live original model after preparation; the metadata is
        # captured from the snapshot's private owners, not this model.
        model.mesh.nodes[1].x = 999.0
        model.mesh.nodes[2].y = -999.0
        model.mesh.elements[1].thickness = 0.5
        after = export_recovery_capsule(evaluator, [PatternCase('b', {'p': 1.0})],
                                        model_identity='m', mesh_identity='x')
    assert after['model_recovery_metadata'] == before['model_recovery_metadata']
    frames = np.asarray(after['model_recovery_metadata']['frames']['elements']['1'])
    assert frames[2, 2] == pytest.approx(1.0)  # the original normal, not the mutation


def test_canonical_and_fallback_routes_retain_recovery_metadata():
    # Canonical route: stored element fields keep their source shapes and the
    # shared metadata is present.
    with _shell_evaluator(stress_execution='canonical') as evaluator:
        capsule = export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.25})],
                                          model_identity='m', mesh_identity='x')
    validate_capsule(capsule)
    metadata = capsule['model_recovery_metadata']
    assert metadata['frames']['elements']
    assert metadata['sampling_rules']
    descriptors = {descriptor['key']: descriptor for descriptor in capsule['descriptors']}
    assert descriptors['membrane_xx']['element_shapes'] == {'1': [7], '2': [7]}
    assert descriptors['membrane_xx']['availability'] == 'stored'
    assert descriptors['membrane_xx']['metadata']['sampling_rules'] == ['rule-0']

    # Certification fallback route: the metadata is retained there too.
    model = _bar()
    pattern = _case(1, 1).load_case
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    fallback_cases = [PatternCase('same', {'p': 1.0}, 2.0,
                                  canonical_load_case=_case(3, 7).load_case)]
    with prepare_static_response_evaluator(model, {'p': pattern}, recovery_config=recovery,
                                           stress_execution='certified_library') as evaluator:
        capsule = export_recovery_capsule(evaluator, fallback_cases, model_identity='bar',
                                          mesh_identity='bar-mesh')
    validate_capsule(capsule)
    assert capsule['cases'][0]['route'] == 'prepared_static_response/canonical_fallback'
    assert capsule['model_recovery_metadata']['frames']['elements']['1']


def test_stream_header_metadata_consistent_with_final_capsule():
    seen = {}

    class RecordingSink:
        def capsule_begin(self, header):
            seen['header'] = dict(header)

        def capsule_case(self, record, coefficients):
            pass

        def capsule_stored(self, key, case_indices, values):
            pass

        def capsule_end(self, final):
            seen['final'] = dict(final)

    with _shell_evaluator() as evaluator:
        export_recovery_capsule_to_sink(evaluator, [PatternCase('a', {'p': 1.25})],
                                        sink=RecordingSink(),
                                        model_identity='m', mesh_identity='x')
        memory = export_recovery_capsule(evaluator, [PatternCase('b', {'p': 1.0})],
                                         model_identity='m', mesh_identity='x')
    assert seen['header']['model_recovery_metadata'] == memory['model_recovery_metadata']
    assert seen['final']['descriptors']  # the final payload completed


def test_model_recovery_metadata_is_finite_plain_json():
    with _shell_evaluator() as evaluator:
        capsule = export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.25})],
                                          model_identity='m', mesh_identity='x')

    def plain(value):
        if value is None or type(value) in (bool, str):
            return True
        if type(value) is int:
            return True
        if type(value) is float:
            return math.isfinite(value)
        if type(value) is list:
            return all(plain(item) for item in value)
        if type(value) is dict:
            return all(type(key) is str and plain(item)
                       for key, item in value.items())
        return False

    # Emitted snapshots refuse mutation but serialize to the same finite
    # JSON-only contract; validate the persisted data representation too.
    import json
    metadata = json.loads(json.dumps(capsule['model_recovery_metadata'], allow_nan=False))
    assert plain(metadata)
    # Closed key set: no factorisations, stiffness, executables or owners.
    assert set(metadata) == {'capture', 'frames', 'sampling_rules', 'element_bindings'}
    assert set(metadata['frames']) == {'convention', 'transform', 'elements'}
    for frame in metadata['frames']['elements'].values():
        assert len(frame) == 3 and all(len(row) == 3 for row in frame)
    for binding in metadata['element_bindings'].values():
        assert set(binding) <= {'element_type', 'frame', 'frame_unavailable_reason',
                                'sampling_rule', 'sampling_unavailable_reason',
                                'sample_count'}
    # The consumer admits the mapping and refuses non-JSON content.
    validate_capsule(capsule)
    from copy import deepcopy
    corrupted = deepcopy(capsule)
    corrupted['model_recovery_metadata']['frames']['elements']['1'][0][0] = float('nan')
    with pytest.raises(CapsuleValidationError):
        validate_capsule(corrupted)
