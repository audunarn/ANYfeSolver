"""Focused development tests for the solver-owned recovery capsule exporter."""
import numpy as np
import pytest

from anyfesolver import (FixedSupport, PatternCase, RecoveryConfig, RecoveryCapsuleError,
                       export_recovery_capsule, prepare_static_evaluator,
                       prepare_static_response_evaluator)
from anyfesolver_results import derive_field, reconstruct_response, validate_capsule
from test_prepared_static_evaluator import _bar, _case, _shell_case, _shell_model


@pytest.mark.parametrize('callback_fails', [False, True])
def test_source_export_failure_reports_original_diagnostics_and_never_ends(callback_fails):
    import json
    from anyfesolver import export_recovery_capsule_to_sink, PreparedStaticError
    cases = [PatternCase('a', {'p': 1.}), PatternCase('b', {'missing': 2.})]
    class Sink:
        def __init__(self):
            self.events = []
            self.failure = None
        def capsule_begin(self, header): self.events.append('begin')
        def capsule_case(self, record, coefficients): self.events.append(record['case_id'])
        def capsule_stored(self, *args): pass
        def capsule_end(self, final): self.events.append('end')
        def capsule_failed(self, record):
            self.failure = record
            if callback_fails:
                raise OSError('diagnostic destination refused')
    sink = Sink()
    with prepare_static_response_evaluator(
            _bar(), {'p': _case(0).load_case},
            recovery_config=RecoveryConfig(include_stresses=True),
            stress_execution='certified_library') as evaluator:
        with pytest.raises(PreparedStaticError) as captured:
            export_recovery_capsule_to_sink(evaluator, cases, sink=sink,
                model_identity='bar', mesh_identity='bar-mesh', batch_size=1)
    primary = captured.value
    assert 'unknown source response pattern' in str(primary)
    assert sink.events == ['begin', 'a']
    failure = json.loads(json.dumps(sink.failure, allow_nan=False))
    assert failure['schema'] == 'anysolver.recovery_capsule_failure'
    assert failure['input_case_ids'] == ['a', 'b']
    assert failure['diagnostics']['pattern'] == 'missing'
    if callback_fails:
        assert any('diagnostic destination refused' in note for note in primary.__notes__)


def test_failure_diagnostics_keep_nonfinite_values_as_explicit_json_tags():
    import json
    from anyfesolver import PreparedStaticError
    from anyfesolver.result_capsule import _report_export_failure
    errors = []
    class Sink:
        def capsule_failed(self, record): errors.append(record)
    primary = PreparedStaticError('nonfinite certification', diagnostics={
        'relative_backward_error': np.float64(np.inf),
        'trace': np.array([np.nan, -np.inf])})
    _report_export_failure(Sink(), primary, [PatternCase('b', {})], 'bar', 'mesh')
    record = json.loads(json.dumps(errors[0], allow_nan=False))
    assert record['diagnostics']['relative_backward_error'] == {
        'numeric_kind': 'nonfinite_float', 'value': '+Infinity'}
    assert record['diagnostics']['trace'] == [
        {'numeric_kind': 'nonfinite_float', 'value': 'NaN'},
        {'numeric_kind': 'nonfinite_float', 'value': '-Infinity'}]


def test_public_capsule_is_an_immutable_owned_snapshot():
    import json
    from copy import deepcopy
    positions = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float64)
    with prepare_static_response_evaluator(
            _bar(), {'p': _case(0).load_case},
            recovery_config=RecoveryConfig(include_stresses=True),
            stress_execution='certified_library') as evaluator:
        capsule = export_recovery_capsule(
            evaluator, [PatternCase('case', {'p': 1.})],
            model_identity='bar', mesh_identity='bar-mesh',
            geometry={'positions': positions})
    positions[1, 0] = 99.
    assert positions.flags.writeable
    assert capsule['geometry']['positions'][1, 0] == 1.
    with pytest.raises(TypeError, match='read-only'):
        capsule['model_identity'] = 'changed'
    with pytest.raises(TypeError, match='read-only'):
        capsule['cases'][0]['certified'] = False
    with pytest.raises(TypeError, match='read-only'):
        capsule['cases'][0]['active_indices'].append(2)
    with pytest.raises(TypeError, match='read-only'):
        capsule['derived'].clear()
    for array in (capsule['coefficients'], capsule['geometry']['positions'],
                  next(iter(capsule['responses'].values())),
                  next(iter(capsule['associations'].values()))):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 42
        with pytest.raises(ValueError):
            array.setflags(write=True)
    # Snapshot metadata still has ordinary JSON/list semantics. Explicit
    # deep cloning permits independent construction without shared buffers.
    assert json.loads(json.dumps(capsule['cases']))[0]['active_indices'] == [1]
    clone = deepcopy(capsule)
    clone['model_identity'] = 'clone'
    clone['geometry']['positions'][1, 0] = 77.
    assert capsule['model_identity'] == 'bar'
    assert capsule['geometry']['positions'][1, 0] == 1.


def test_failed_record_keeps_source_diagnostics_without_certifying_fields():
    from anyfesolver.result_capsule import _CapsuleExport
    cases = [PatternCase('failed-case', {'p': 1.0})]
    failed = {
        'status': 'failed',
        'diagnostics': {'stage': 'recovery', 'element_id': np.int64(17)},
        'error': 'element recovery failed',
        'solver_info': {'status': 'failed', 'message': 'source failure',
                        'diagnostics': {'input_index': 0}},
    }
    with prepare_static_response_evaluator(
            _bar(), {'p': _case(0).load_case},
            recovery_config=RecoveryConfig(include_stresses=True),
            stress_execution='certified_library') as evaluator:
        exporter = _CapsuleExport(evaluator, cases, model_identity='bar',
                                  mesh_identity='bar-mesh', batch_size=1)
        record, coefficients = exporter._process_case(0, failed)
        assert record['diagnostics'] == {'stage': 'recovery', 'element_id': 17}
        assert record['error'] == failed['error']
        assert record['solver_info'] == failed['solver_info']
        assert not record['certified'] and not record['portable']
        assert record['displacement_portable'] is None
        assert not exporter.stored_rows
        np.testing.assert_array_equal(coefficients, [1.0])
    # Exported diagnostic metadata does not alias its mutable source.
    failed['diagnostics']['stage'] = 'mutated'
    failed['solver_info']['diagnostics']['input_index'] = 99
    assert record['diagnostics']['stage'] == 'recovery'
    assert record['solver_info']['diagnostics']['input_index'] == 0


def _reference_fields(case_result):
    """Flattened per-element float field records of one evaluator case result."""
    stresses = case_result['element_stresses']
    fields = {}
    for element_id in sorted(int(eid) for eid in stresses):
        record = stresses[element_id]
        for name, value in record.items():
            if isinstance(value, np.ndarray) and value.dtype.kind == 'f':
                fields.setdefault(name, {})[element_id] = value.reshape(-1)
            elif isinstance(value, (int, float, np.generic)) and not isinstance(value, (bool, np.bool_)):
                fields.setdefault(name, {})[element_id] = np.asarray(float(value)).reshape(1)
    return fields


def _samples_for(associations, element_id):
    return np.nonzero(associations == element_id)[0]


def test_library_route_export_reconstructs_evaluator_records():
    model = _shell_model()
    pattern = _shell_case(1, pressure=3).load_case
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    cases = [PatternCase('a', {'p': 1.5}), PatternCase('b', {'p': -0.5})]
    with prepare_static_response_evaluator(model, {'p': pattern}, recovery_config=recovery,
                                          stress_execution='certified_library') as evaluator:
        reference = next(evaluator.iter_batches(list(cases)))
        capsule = export_recovery_capsule(evaluator, cases, model_identity='shell-model',
                                         mesh_identity='shell-mesh')
    validate_capsule(capsule)
    assert capsule['schema'] == 'anysolver.recovery_capsule'
    assert capsule['version'] == 2 and capsule['runtime'] == 'anysolver-results/2'
    assert capsule['patterns'] == ('p',)
    np.testing.assert_array_equal(capsule['coefficients'], np.array([[1.5], [-0.5]]))
    assert [case['case_id'] for case in capsule['cases']] == ['a', 'b']
    displacement_keys = {f'displacement:{c}' for c in ('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')}
    reaction_keys = {f'reaction:{c}' for c in ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')}
    for case in capsule['cases']:
        assert case['status'] == 'converged' and case['certified'] and case['portable']
        assert case['route'] == 'prepared_static_response/certified_library'
        assert case['active_indices'] == [1]
        assert case['displacement_portable'] is True
        assert case['provenance']['relative_backward_error_disposition'] == 'CERTIFIED'
    # Element fields and the nodal displacement basis are reconstructable; the
    # actual reaction components (bitwise Ku-F) are the only stored rows on
    # this route, and every displacement component is in responses.
    assert set(capsule['stored']) == reaction_keys
    assert displacement_keys <= set(capsule['responses'])

    descriptors = {descriptor['key']: descriptor for descriptor in capsule['descriptors']}
    assert descriptors['von_mises']['availability'] == 'reconstructable'
    assert capsule['derived']['von_mises']['operation'] == 'shell_von_mises'
    assert set(capsule['derived']['von_mises']['inputs'].values()) == {
        'membrane_xx', 'membrane_yy', 'membrane_xy', 'bending_xx', 'bending_yy',
        'bending_xy', 'shear_xz', 'shear_yz'}
    for component in ('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ'):
        descriptor = descriptors[f'displacement:{component}']
        assert descriptor['availability'] == 'reconstructable'
        assert descriptor['quantity_id'] == 'displacement'
        assert descriptor['location'] == 'node' and descriptor['basis'] == 'global'
        assert descriptor['portable_case_count'] == 2
    for component in ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ'):
        descriptor = descriptors[f'reaction:{component}']
        assert descriptor['availability'] == 'stored'
        assert descriptor['quantity_id'] == 'reaction'
        assert descriptor['metadata']['source'] == 'actual_full_Ku_minus_F'

    for index, case_result in enumerate(reference.case_results):
        case = capsule['cases'][index]
        expected = _reference_fields(case_result)
        reconstructed = {}
        for name, responses in capsule['responses'].items():
            reconstructed[name] = reconstruct_response(
                responses, capsule['coefficients'][index],
                active_indices=case['active_indices'])
        for name in capsule['responses']:
            if name.startswith('displacement:'):
                continue
            assert name in expected, name
            for element_id, values in expected[name].items():
                samples = _samples_for(capsule['associations'][name], element_id)
                np.testing.assert_array_equal(reconstructed[name][samples], values)
        # The fake zero Hill utilisation is not an exported claim.
        descriptor_keys = {descriptor['key'] for descriptor in capsule['descriptors']}
        assert 'hill_utilization' not in descriptor_keys
        assert 'hill_utilization' not in capsule['derived']
        # Derived fields reproduce the evaluator's records from the same inputs.
        von_mises = derive_field(capsule['derived']['von_mises'],
                                 {name: reconstructed[name] for name in
                                  capsule['derived']['von_mises']['inputs']})
        for element_id, values in expected['von_mises'].items():
            samples = _samples_for(capsule['associations']['von_mises'], element_id)
            np.testing.assert_allclose(von_mises[samples], values, rtol=0, atol=0)
        if 'equivalent_stress' in capsule['derived']:
            equivalent = derive_field(capsule['derived']['equivalent_stress'],
                                      {'von_mises': von_mises})
            for element_id, values in expected['equivalent_stress'].items():
                samples = _samples_for(capsule['associations']['equivalent_stress'], element_id)
                np.testing.assert_array_equal(equivalent[samples], values)
        # The nodal displacement basis reproduces the evaluator's actual
        # recorded nodal vectors bitwise, component by component.
        recorded = case_result['node_displacements']
        for position, component in enumerate(('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')):
            key = f'displacement:{component}'
            association = capsule['associations'][key]
            actual = np.asarray([recorded[node][position] for node in association])
            np.testing.assert_array_equal(reconstructed[key], actual)
        # The stored reaction rows are the evaluator's actual Ku-F values.
        reactions = case_result['reactions']
        for position, component in enumerate(('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')):
            key = f'reaction:{component}'
            stored = capsule['stored'][key]
            row = list(stored['case_indices']).index(index)
            association = capsule['associations'][key]
            actual = np.asarray([reactions[node][position] for node in association])
            np.testing.assert_array_equal(stored['values'][row], actual)


@pytest.mark.parametrize('replay_version', [1, 2])
def test_canonical_route_exports_explicit_stored_arrays(replay_version):
    model = _shell_model()
    pattern = _shell_case(2, pressure=1.0).load_case
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    cases = [PatternCase('a', {'p': 2.0})]
    with prepare_static_response_evaluator(model, {'p': pattern},
                                           recovery_config=recovery) as evaluator:
        reference = next(evaluator.iter_batches(list(cases)))
        capsule = export_recovery_capsule(evaluator, cases, model_identity='shell-model',
                                         mesh_identity='shell-mesh', replay_version=replay_version)
    validate_capsule(capsule)
    displacement_keys = {f'displacement:{c}' for c in ('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')}
    # The canonical route legitimately emits the bitwise-certified nodal
    # displacement basis; element fields stay explicit stored arrays.
    assert set(capsule['responses']) == displacement_keys
    assert capsule['derived'] == {}
    case = capsule['cases'][0]
    assert case['certified'] and not case['portable']
    assert case['route'] == 'prepared_static_response/canonical'
    assert case['displacement_portable'] is (replay_version == 1)
    descriptors = {descriptor['key']: descriptor for descriptor in capsule['descriptors']}
    expected = _reference_fields(reference.case_results[0])
    # The fake zero Hill utilisation is never an exported claim.
    assert 'hill_utilization' not in descriptors
    for name in expected:
        if name == 'hill_utilization':
            continue
        assert descriptors[name]['availability'] == 'stored'
        assert name in capsule['stored']
        stored = capsule['stored'][name]
        np.testing.assert_array_equal(stored['case_indices'], [0])
        for element_id, values in expected[name].items():
            samples = _samples_for(capsule['associations'][name], element_id)
            np.testing.assert_array_equal(stored['values'][0][samples], values)
    # The emitted nodal basis reproduces the evaluator's actual recorded
    # nodal vectors bitwise, component by component.
    recorded = reference.case_results[0]['node_displacements']
    for position, component in enumerate(('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')):
        key = f'displacement:{component}'
        field = (capsule['stored'][key]['values'][0] if replay_version == 2 else
                 reconstruct_response(capsule['responses'][key], capsule['coefficients'][0],
                                      active_indices=case['active_indices']))
        association = capsule['associations'][key]
        actual = np.asarray([recorded[node][position] for node in association])
        np.testing.assert_array_equal(field, actual)
        assert descriptors[key]['availability'] == 'reconstructable'


def test_certification_fallback_case_is_stored_not_portable():
    model = _bar()
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    pattern = _case(1, 1).load_case
    cases = [PatternCase('same', {'p': 1.0}, 2.0, canonical_load_case=_case(3, 7).load_case),
             PatternCase('plain', {'p': 1.0})]
    with prepare_static_response_evaluator(model, {'p': pattern}, recovery_config=recovery,
                                          stress_execution='certified_library') as evaluator:
        reference = next(evaluator.iter_batches(list(cases)))
        capsule = export_recovery_capsule(evaluator, cases, model_identity='bar',
                                         mesh_identity='bar-mesh')
    validate_capsule(capsule)
    fallback, portable = capsule['cases']
    assert fallback['certified'] and not fallback['portable']
    assert fallback['route'] == 'prepared_static_response/canonical_fallback'
    assert portable['certified'] and portable['portable']
    expected = _reference_fields(reference.case_results[0])
    assert 'von_mises' in capsule['stored']
    stored = capsule['stored']['von_mises']
    assert list(stored['case_indices']) == [0]
    for element_id, values in expected['von_mises'].items():
        samples = _samples_for(capsule['associations']['von_mises'], element_id)
        np.testing.assert_allclose(stored['values'][0][samples], values, rtol=1e-12, atol=1e-12)
    # The beam von Mises operation is exported for the portable case.
    assert capsule['derived']['von_mises']['operation'] == 'beam_von_mises'


def test_two_pattern_sparse_and_reversed_coefficients_reconstruct_bitwise():
    """Sparse active subsets and reversed mapping orders use their own weights.

    A case activating one pattern of two must not multiply the full
    coefficient row against the active columns, and a mapping given in
    reversed order must weight the active columns in the original active
    order.  Both are checked bitwise against the evaluator's own records.
    """
    model = _bar()
    patterns = {'p': _case(1, 2.0).load_case, 'q': _case(2, -1.0).load_case}
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    cases = [PatternCase('sparse', {'p': 2.0}),
             PatternCase('reversed', {'q': 3.0, 'p': 2.0})]
    with prepare_static_response_evaluator(model, patterns, recovery_config=recovery,
                                           baseline_load_case=_case(0, 0.5).load_case,
                                           stress_execution='certified_library') as evaluator:
        reference = next(evaluator.iter_batches(list(cases)))
        capsule = export_recovery_capsule(evaluator, cases, model_identity='bar',
                                          mesh_identity='bar-mesh')
    validate_capsule(capsule)
    assert capsule['patterns'] == ('p', 'q')
    np.testing.assert_array_equal(capsule['coefficients'],
                                  np.array([[2.0, 0.0], [2.0, 3.0]]))
    assert [case['active_indices'] for case in capsule['cases']] == [[1], [2, 1]]
    # The exact original request position is exported on every case record,
    # and every available descriptor declares the source envelope tie policy.
    assert [case['input_index'] for case in capsule['cases']] == [0, 1]
    for descriptor in capsule['descriptors']:
        if descriptor['availability'] != 'unavailable':
            assert descriptor['tie_policy'] == 'exact_earliest_input_index'
    for case in capsule['cases']:
        assert case['certified'] and case['portable']
        assert case['displacement_portable'] is True
    for index, case_result in enumerate(reference.case_results):
        case = capsule['cases'][index]
        active = case['active_indices']
        weights = capsule['coefficients'][index][np.asarray(active) - 1]
        expected = _reference_fields(case_result)
        for name, responses in capsule['responses'].items():
            if name.startswith('displacement:'):
                continue
            # A per-case stored override is present whenever the runtime's
            # field-level accumulation does not reproduce the certified
            # record bitwise; the certified values are then observed from
            # the stored row instead of the basis.
            stored = capsule['stored'].get(name)
            override = stored is not None and index in list(stored['case_indices'])
            if override:
                row = list(stored['case_indices']).index(index)
                for element_id, values in expected[name].items():
                    samples = _samples_for(capsule['associations'][name], element_id)
                    np.testing.assert_array_equal(stored['values'][row][samples], values)
                continue
            field = reconstruct_response(responses, weights, active_indices=active)
            for element_id, values in expected[name].items():
                samples = _samples_for(capsule['associations'][name], element_id)
                np.testing.assert_array_equal(field[samples], values)
        # The nonzero baseline is part of the admitted arithmetic: the
        # baseline-only displacement of a zero-coefficient case is exact.
        recorded = case_result['node_displacements']
        for position, component in enumerate(('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')):
            key = f'displacement:{component}'
            field = reconstruct_response(capsule['responses'][key], weights,
                                        active_indices=active)
            association = capsule['associations'][key]
            actual = np.asarray([recorded[node][position] for node in association])
            np.testing.assert_array_equal(field, actual)
        reactions = case_result['reactions']
        for position, component in enumerate(('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')):
            key = f'reaction:{component}'
            stored = capsule['stored'][key]
            row = list(stored['case_indices']).index(index)
            association = capsule['associations'][key]
            actual = np.asarray([reactions[node][position] for node in association])
            np.testing.assert_array_equal(stored['values'][row], actual)


def test_source_envelope_ties_keep_exact_earliest_input_index():
    """The declared tie policy is the unchanged source contract.

    Both source envelope reducers retain, for equal minima, maxima and
    equal absolute maxima, the exact earliest original input_index,
    irrespective of update, reduction or subset traversal order.  The
    capsule's ``tie_policy`` descriptor label and per-case ``input_index``
    refer to exactly this contract.
    """
    from anyfesolver.static_envelopes import EnvelopeReducer
    reducer = EnvelopeReducer()
    for index in (3, 1, 2, 0):  # reversed traversal order
        reducer.update({'v': -5.0}, case_id=f'c{index}', input_index=index)
    extrema = reducer.extrema['v']
    assert extrema.minimum.input_index == 0
    assert extrema.maximum.input_index == 0
    assert extrema.absolute_maximum.input_index == 0
    # Equal absolute values with opposite signs also keep the earliest case.
    signed = EnvelopeReducer()
    for index in (2, 0):
        signed.update({'v': 5.0 if index else -5.0}, case_id='c', input_index=index)
    assert signed.extrema['v'].absolute_maximum.input_index == 0
    assert signed.extrema['v'].absolute_maximum.value == -5.0

    # The prepared canonical accumulator keeps the same policy across tile
    # folding and reduced-subset merging, in either traversal order.
    from types import SimpleNamespace
    from anyfesolver.prepared_static_envelopes import EnvelopeAccumulator

    def accumulator():
        recipe = SimpleNamespace(staged_kernel=None, staged_fallback_reason='tie-test')
        element = SimpleNamespace(element_id=1, dof_mapping=None, recipe=recipe,
                                  components=None, schema=(('s', ()),))
        plan = SimpleNamespace(elements=(element,), node_ids=(),
                               include_displacements=False, include_reactions=False,
                               retained_bytes=0, tile_bytes=0)
        return EnvelopeAccumulator(plan)

    def governing_indices(state):
        return [state.entries[('stress', 1, 's')][1][mode]['input_indices'][0]
                for mode in range(3)]

    # Reversed tile traversal: the earlier original index wins the tie.
    folded = accumulator()
    folded.fold('stress', 1, {'s': np.asarray(5.0)}, 'late', None, 7)
    folded.fold('stress', 1, {'s': np.asarray(5.0)}, 'early', None, 2)
    assert governing_indices(folded) == [2, 2, 2]
    # A reduced subset carrying a later index never displaces the retained
    # earliest one; a reduced subset carrying an earlier one does.
    retained = accumulator()
    retained.fold('stress', 1, {'s': np.asarray(5.0)}, 'early', None, 2)
    late_subset = retained.empty_subset(('stress',))
    late_subset.fold('stress', 1, {'s': np.asarray(5.0)}, 'late', None, 7)
    retained.merge_reduced(late_subset)
    assert governing_indices(retained) == [2, 2, 2]
    late_main = accumulator()
    late_main.fold('stress', 1, {'s': np.asarray(5.0)}, 'late', None, 7)
    early_subset = late_main.empty_subset(('stress',))
    early_subset.fold('stress', 1, {'s': np.asarray(5.0)}, 'early', None, 2)
    late_main.merge_reduced(early_subset)
    assert governing_indices(late_main) == [2, 2, 2]
    # The prepared result diagnostics declare the same policy label.
    assert folded.result('identity').diagnostics['tie_policy'] == 'exact_earliest_input_index'


def _rotated_shell_model():
    """The shell fixture rotated 30 degrees about the global x axis."""
    angle = np.deg2rad(30.0)
    cos, sin = np.cos(angle), np.sin(angle)

    def rotated(x, y, z):
        return (x, y * cos - z * sin, y * sin + z * cos)

    from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from anyfesolver.fe_core import FEModel
    model = FEModel('prepared-static-shell-rotated')
    model.add_material('steel', 210.0e9, 0.3, density=100.0)
    coordinates = {1: rotated(0.0, 0.0, 0.0), 2: rotated(1.0, 0.0, 0.0),
                   3: rotated(0.0, 1.0, 0.0), 4: rotated(1.0, 1.0, 0.0)}
    for node_id, (x, y, z) in coordinates.items():
        model.add_node(node_id, x, y, z)
    normal = (0.0, -sin, cos)
    model.add_element(1, QualifiedE4PLS3ShellElement(
        1, [1, 2, 3], 'steel', thickness=0.01, reference_normal=normal))
    model.add_element(2, QualifiedE4PLS3ShellElement(
        2, [2, 4, 3], 'steel', thickness=0.01, reference_normal=normal))
    model.add_boundary_condition(FixedSupport('fixed', [1, 3]))
    return model


def test_rotated_shell_displacement_basis_and_reactions():
    """A rotated real shell: nodal basis parity and actual reactions survive."""
    model = _rotated_shell_model()
    pattern = _shell_case(1, pressure=2.0).load_case
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    cases = [PatternCase('a', {'p': 1.25})]
    with prepare_static_response_evaluator(model, {'p': pattern}, recovery_config=recovery,
                                           stress_execution='certified_library') as evaluator:
        reference = next(evaluator.iter_batches(list(cases)))
        capsule = export_recovery_capsule(evaluator, cases, model_identity='rotated-shell',
                                          mesh_identity='rotated-shell-mesh')
    validate_capsule(capsule)
    case = capsule['cases'][0]
    assert case['certified'] and case['portable'] and case['displacement_portable'] is True
    case_result = reference.case_results[0]
    recorded = case_result['node_displacements']
    # Nonzero real nodal values on the rotated mesh.
    assert np.any(recorded[4][:3] != 0.0)
    for position, component in enumerate(('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')):
        key = f'displacement:{component}'
        field = reconstruct_response(capsule['responses'][key],
                                     capsule['coefficients'][0],
                                     active_indices=case['active_indices'])
        association = capsule['associations'][key]
        actual = np.asarray([recorded[node][position] for node in association])
        np.testing.assert_array_equal(field, actual)
    reactions = case_result['reactions']
    assert reactions
    for position, component in enumerate(('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')):
        key = f'reaction:{component}'
        stored = capsule['stored'][key]
        association = capsule['associations'][key]
        actual = np.asarray([reactions[node][position] for node in association])
        np.testing.assert_array_equal(stored['values'][0], actual)


@pytest.mark.parametrize('replay_version', [1, 2])
def test_principal_stress_ranks_queryable_with_explicit_fallback(replay_version):
    """Principal stresses export as per-rank scalars with the surface kept.

    Each rank is its own queryable component with exact per-case shape and
    an explicit stored fallback; the runtime operation reproduces the stored
    ranks from the reconstructed shell inputs.
    """
    config = RecoveryConfig(components=['principal_stresses_top',
                                        'principal_stresses_bottom'],
                            include_stresses=True, include_reactions=True)
    cases = [PatternCase('a', {'p': 1.5})]
    for route in ('canonical', 'certified_library'):
        model = _shell_model()
        pattern = _shell_case(1, pressure=2.0).load_case
        kwargs = {'recovery_config': config}
        if route == 'certified_library':
            kwargs['stress_execution'] = route
        with prepare_static_response_evaluator(model, {'p': pattern}, **kwargs) as evaluator:
            reference = next(evaluator.iter_batches(list(cases)))
            capsule = export_recovery_capsule(evaluator, cases, model_identity='shell-model',
                                              mesh_identity='shell-mesh', replay_version=replay_version)
        validate_capsule(capsule)
        descriptors = {descriptor['key']: descriptor for descriptor in capsule['descriptors']}
        shared = capsule['model_recovery_metadata']
        rule_id = shared['element_bindings']['1']['sampling_rule']
        case_result = reference.case_results[0]
        for surface in ('top', 'bottom'):
            record = case_result['element_stresses'][1][f'principal_stresses_{surface}']
            for rank in (1, 2, 3):
                key = f'principal_stress_{rank}:{surface}'
                descriptor = descriptors[key]
                assert descriptor['quantity_id'] == 'stress'
                assert descriptor['unit'] == 'Pa'
                assert descriptor['basis'] == 'principal'
                # The per-rank metadata plus the compact shared-table refs.
                expected_metadata = {
                    'surface': surface, 'rank': rank,
                    'sampling_rules': [rule_id],
                    'local_frames': 'model_recovery_metadata.frames'}
                if replay_version == 2:
                    expected_metadata['portable_replay_refusal'] = 'spectral_replay_not_certified_portable'
                    assert key not in capsule['derived']
                assert descriptor['metadata'] == expected_metadata
                assert descriptor['availability'] == 'stored'
                stored = capsule['stored'][key]
                # Exact shape: one scalar per element sample, per case.
                spatial = capsule['associations'][key].shape[0]
                assert stored['values'].shape == (1, spatial)
                expected = np.concatenate([
                    case_result['element_stresses'][eid][f'principal_stresses_{surface}'][:, rank - 1]
                    for eid in (1, 2)])
                np.testing.assert_array_equal(stored['values'][0], expected)
            # Descending eigenvalues per sample on the stored ranks.
            first = capsule['stored'][f'principal_stress_1:{surface}']['values'][0]
            second = capsule['stored'][f'principal_stress_2:{surface}']['values'][0]
            third = capsule['stored'][f'principal_stress_3:{surface}']['values'][0]
            assert np.all(first >= second) and np.all(second >= third)
        if route == 'certified_library' and replay_version == 1:
            # The runtime rank operation reproduces the stored scalars from
            # the reconstructed shell inputs on the portable route.
            case = capsule['cases'][0]
            inputs = {}
            for name in ('membrane_xx', 'membrane_yy', 'membrane_xy',
                         'bending_xx', 'bending_yy', 'bending_xy',
                         'shear_xz', 'shear_yz'):
                inputs[name] = reconstruct_response(
                    capsule['responses'][name], capsule['coefficients'][0],
                    active_indices=case['active_indices'])
            for surface in ('top', 'bottom'):
                for rank in (1, 2, 3):
                    field = derive_field({'operation': 'shell_principal_stress',
                                          'surface': surface, 'rank': rank,
                                          'inputs': {name: name for name in inputs}}, inputs)
                    stored = capsule['stored'][f'principal_stress_{rank}:{surface}']['values'][0]
                    np.testing.assert_array_equal(field, stored)


def test_export_refusals():
    model = _bar()
    pattern = _case(1).load_case
    cases = [PatternCase('a', {'p': 1.0})]
    with prepare_static_evaluator(model) as plain:
        with pytest.raises(RecoveryCapsuleError, match='response evaluator'):
            export_recovery_capsule(plain, cases, model_identity='m', mesh_identity='x')
    with prepare_static_response_evaluator(model, {'p': pattern}) as evaluator:
        with pytest.raises(RecoveryCapsuleError, match='model_identity'):
            export_recovery_capsule(evaluator, cases, model_identity='', mesh_identity='x')
        with pytest.raises(RecoveryCapsuleError, match='PatternCase'):
            export_recovery_capsule(evaluator, [_case(0, 1)], model_identity='m', mesh_identity='x')
        with pytest.raises(RecoveryCapsuleError, match='unique'):
            export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.0}),
                                                PatternCase('a', {'p': 2.0})],
                                    model_identity='m', mesh_identity='x')
    with pytest.raises(RecoveryCapsuleError, match='open'):
        export_recovery_capsule(evaluator, cases, model_identity='m', mesh_identity='x')


def test_geometry_and_solver_identity_passthrough():
    model = _bar()
    pattern = _case(1).load_case
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    with prepare_static_response_evaluator(model, {'p': pattern},
                                           recovery_config=RecoveryConfig(include_stresses=True)) as evaluator:
        capsule = export_recovery_capsule(evaluator, [PatternCase('a', {'p': 1.0})],
                                         model_identity='m', mesh_identity='x',
                                         geometry={'positions': positions},
                                         solver_identity={'solver': 'ANYfeSolver', 'build': 'test'})
    np.testing.assert_array_equal(capsule['geometry']['positions'], positions)
    assert capsule['solver_identity']['build'] == 'test'
    assert capsule['solver_identity']['stress_execution'] == 'canonical'
