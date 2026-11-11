"""Focused development tests for the NumPy-only anyfesolver-results runtime."""
import numpy as np
import pytest

import anyfesolver_results
from anyfesolver_results import (CapsuleError, CapsuleValidationError, DerivationError,
                               PreparedResponse, ReconstructionError, derive_field,
                               prepare_response, reconstruct_prepared_response,
                               reconstruct_response, validate_capsule)
from anyfesolver_results.capsule import COMPLETED_STATUSES


def _responses():
    return np.array([[1.0, 2.0, -1.0],
                     [4.0, 0.5, 3.0]])


def test_runtime_identity():
    assert anyfesolver_results.RUNTIME_ID == 'anysolver-results/2'
    assert anyfesolver_results.SUPPORTED_RUNTIME_IDS == ('anysolver-results/1','anysolver-results/2')


def test_reconstruct_adds_baseline_once():
    values = _responses()
    out = reconstruct_response(values, np.array([0.5, 2.0]))
    expected = values[:, 0] + values[:, 1:] @ np.array([0.5, 2.0])
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.float64
    assert not out.flags.writeable


def test_reconstruct_active_subset_matches_solver_combination():
    values = _responses()
    out = reconstruct_response(values, np.array([2.0]), active_indices=[2])
    np.testing.assert_array_equal(out, values[:, 0] + values[:, [2]] @ np.array([2.0]))
    out = reconstruct_response(values, np.zeros(0), active_indices=np.zeros(0, dtype=np.intp))
    np.testing.assert_array_equal(out, values[:, 0])


def test_reconstruct_refuses_misaligned_and_nonfinite():
    values = _responses()
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0, 2.0]), active_indices=[1])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=[0])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=[3])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([np.nan]), active_indices=[1])
    with pytest.raises(ReconstructionError):
        reconstruct_response(np.zeros((2, 2)), np.array([np.inf, 1.0]))
    with pytest.raises(ReconstructionError):
        reconstruct_response(np.zeros(4), np.array([1.0]))


def _shell_inputs(seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(-10.0, 10.0, 5)
            for name in ('membrane_xx', 'membrane_yy', 'membrane_xy',
                         'bending_xx', 'bending_yy', 'bending_xy',
                         'shear_xz', 'shear_yz')}


def _oracle_shell_von_mises(inputs):
    # The solver's reviewed per-sample expression (results._von_mises_surface
    # applied per sample with the shared transverse shear), used here as an
    # independent reference for the shared formula.
    out = []
    for i in range(len(next(iter(inputs.values())))):
        mx, my, mxy = (inputs[k][i] for k in ('membrane_xx', 'membrane_yy', 'membrane_xy'))
        bx, by, bxy = (inputs[k][i] for k in ('bending_xx', 'bending_yy', 'bending_xy'))
        sxz, syz = inputs['shear_xz'][i], inputs['shear_yz'][i]
        surfaces = []
        for sign in (1.0, -1.0):
            sxx, syy, szz = mx + sign * bx, my + sign * by, 0.0
            sxy, syz_s, sxz_s = mxy + sign * bxy, syz, sxz
            surfaces.append(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                                    + 3.0 * (sxy ** 2 + syz_s ** 2 + sxz_s ** 2)))
        out.append(max(surfaces))
    return np.asarray(out)


def test_shell_von_mises_matches_oracle_expression_bitwise():
    inputs = _shell_inputs(3)
    spec = {'operation': 'shell_von_mises',
            'inputs': {name: name for name in inputs}}
    out = derive_field(spec, inputs)
    np.testing.assert_array_equal(out, _oracle_shell_von_mises(inputs))


def test_beam_von_mises_and_equivalent_stress():
    rng = np.random.default_rng(5)
    inputs = {name: rng.uniform(-5.0, 5.0, 4)
              for name in ('axial_stress', 'bending_stress_y', 'bending_stress_z',
                           'shear_stress_y', 'shear_stress_z', 'torsional_stress')}
    out = derive_field({'operation': 'beam_von_mises',
                        'inputs': {name: name for name in inputs}}, inputs)
    sigma_x = inputs['axial_stress'] + inputs['bending_stress_y'] + inputs['bending_stress_z']
    expected = np.sqrt(sigma_x * sigma_x + 3.0 * (
        inputs['shear_stress_y'] ** 2 + inputs['shear_stress_z'] ** 2
        + inputs['torsional_stress'] ** 2))
    np.testing.assert_array_equal(out, expected)
    equivalent = derive_field({'operation': 'equivalent_stress', 'inputs': {'von_mises': 'vm'}},
                               {'von_mises': out})
    np.testing.assert_array_equal(equivalent, out)


def test_principal_stress_scalar_ranks_and_surface_parameter():
    inputs = _shell_inputs(7)
    ranks = {}
    for surface in ('top', 'bottom'):
        for rank in (1, 2, 3):
            ranks[(surface, rank)] = derive_field(
                {'operation': 'shell_principal_stress', 'surface': surface, 'rank': rank,
                 'inputs': {name: name for name in inputs}}, inputs)
            # Scalar per sample: the spatial shape never changes with the rank.
            assert ranks[(surface, rank)].shape == (5,)
    for surface in ('top', 'bottom'):
        values = [ranks[(surface, rank)] for rank in (1, 2, 3)]
        # Descending eigenvalues per sample.
        assert np.all(values[0] >= values[1]) and np.all(values[1] >= values[2])
        assert np.all(np.isfinite(values))
    with pytest.raises(DerivationError):
        derive_field({'operation': 'shell_principal_stress', 'surface': 'middle', 'rank': 1,
                      'inputs': {name: name for name in inputs}}, inputs)
    with pytest.raises(DerivationError):
        derive_field({'operation': 'shell_principal_stress', 'surface': 'top', 'rank': 0,
                      'inputs': {name: name for name in inputs}}, inputs)
    with pytest.raises(DerivationError):
        derive_field({'operation': 'shell_principal_stress', 'surface': 'top', 'rank': True,
                      'inputs': {name: name for name in inputs}}, inputs)


def test_registry_is_closed_and_hill_is_not_a_claim():
    assert 'hill_utilization' not in anyfesolver_results.OPERATIONS
    with pytest.raises(DerivationError):
        derive_field({'operation': 'hill_utilization', 'inputs': {}}, {})
    with pytest.raises(DerivationError):
        derive_field({'operation': 'invented_measure', 'inputs': {}}, {})


def test_derive_field_input_validation():
    inputs = _shell_inputs(11)
    spec = {'operation': 'shell_von_mises', 'inputs': {name: name for name in inputs}}
    with pytest.raises(DerivationError):
        derive_field(spec, {k: v for k, v in inputs.items() if k != 'shear_xz'})
    with pytest.raises(DerivationError):
        derive_field(spec, {**inputs, 'shear_xz': np.zeros((5, 1))})
    wrong = dict(inputs)
    wrong['shear_xz'] = wrong['shear_xz'][:3]
    with pytest.raises(DerivationError):
        derive_field(spec, wrong)
    with pytest.raises(DerivationError):
        derive_field(spec, {**inputs, 'shear_xz': np.array([np.nan, 0, 0, 0, 0])})


def _minimal_capsule():
    return {
        'schema': 'anysolver.recovery_capsule',
        'version': 1,
        'runtime': 'anysolver-results/1',
        'model_identity': 'model-a',
        'mesh_identity': 'mesh-a',
        'solver_identity': {'solver': 'ANYfeSolver'},
        'geometry': {'positions': np.zeros((0, 3))},
        'patterns': ('p',),
        'coefficients': np.array([[1.5], [-2.0]], dtype=np.float64),
        'cases': ({'case_id': 'a', 'status': 'converged', 'certified': True,
                   'portable': True, 'route': 'prepared_static_response/certified_library'},
                  {'case_id': 'b', 'status': 'failed', 'certified': False,
                   'portable': False, 'route': 'prepared_static_response/canonical'}),
        'descriptors': ({'key': 'von_mises', 'quantity_id': 'von_mises', 'component': 'von_mises',
                         'location': 'element', 'unit': 'native', 'basis': 'compiled',
                         'reconstruction': 'compiled_affine_pattern_basis',
                         'sampling': 'element_sample', 'availability': 'reconstructable'},),
        'associations': {'von_mises': np.array([1, 1, 2], dtype=np.int64)},
        'responses': {'von_mises': np.array([[0.0, 1.0], [0.0, 2.0], [0.0, -1.0]],
                                            dtype=np.float64)},
        'stored': {},
        'derived': {},
    }


def test_validate_accepts_minimal_capsule_with_failed_case():
    capsule = _minimal_capsule()
    validate_capsule(capsule)
    # The admitted capsule reconstructs: failed cases stay discoverable but
    # only the certified case's arithmetic is replayed.
    field = reconstruct_response(capsule['responses']['von_mises'],
                                 np.array([1.5]), active_indices=[1])
    np.testing.assert_array_equal(field, np.array([1.5, 3.0, -1.5]))


def test_validate_refuses_admission_counterexamples():
    """Runtime admission contract: malformed capsules are refused, not coerced."""
    def broken(**changes):
        capsule = _minimal_capsule()
        capsule.update(changes)
        return capsule

    def case_change(**changes):
        capsule = _minimal_capsule()
        case = dict(capsule['cases'][0])
        case.update(changes)
        capsule['cases'] = (case, capsule['cases'][1])
        return capsule

    # A failed case can never carry a certified flag.
    with pytest.raises(CapsuleValidationError):
        validate_capsule(case_change(status='failed'))
    # Every case needs its actual route label.
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        case = dict(capsule['cases'][0])
        del case['route']
        capsule['cases'] = (case, capsule['cases'][1])
        validate_capsule(capsule)
    # NaN is not plain JSON data anywhere in the identity or provenance.
    with pytest.raises(CapsuleValidationError):
        validate_capsule(broken(solver_identity={'solver': 'ANYfeSolver', 'value': float('nan')}))
    # NaN coefficient tables and NaN response bases are refused.
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['coefficients'] = np.array([[np.nan], [-2.0]], dtype=np.float64)
        validate_capsule(capsule)
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['responses']['von_mises'][0, 1] = np.inf
        validate_capsule(capsule)
    # Duplicate stored case indices would silently disagree per row.
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['stored'] = {'von_mises': {
            'case_indices': np.array([0, 0], dtype=np.int64),
            'values': np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])}}
        validate_capsule(capsule)
    # Derived dependencies must be closed over the declared descriptors.
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['derived'] = {'von_mises': {
            'operation': 'shell_von_mises',
            'inputs': {'membrane_xx': 'absent'}}}
        validate_capsule(capsule)
    # Derived dependencies must not form a cycle.
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['descriptors'] = capsule['descriptors'] + (
            {'key': 'other', 'quantity_id': 'other', 'component': 'other',
             'location': 'element', 'unit': 'native', 'basis': 'compiled',
             'reconstruction': 'derived_from_reconstructable_inputs',
             'sampling': 'element_sample', 'availability': 'reconstructable'},)
        capsule['derived'] = {
            'von_mises': {'operation': 'shell_von_mises', 'inputs': {'x': 'other'}},
            'other': {'operation': 'shell_von_mises', 'inputs': {'x': 'von_mises'}}}
        validate_capsule(capsule)
    # A portable case that is not certified is contradictory.
    with pytest.raises(CapsuleValidationError):
        validate_capsule(case_change(certified=False))


def test_reconstruct_refuses_coerced_active_indices():
    values = np.array([[10.0, 2.0, 3.0]])
    # Fractional, boolean and duplicate indices are refused, never coerced.
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=[1.9])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=[True])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0, 2.0]), active_indices=[1, 1])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=np.array([1.0]))
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=[[1]])
    # Invalid conversion paths raise the documented typed error.
    with pytest.raises(ReconstructionError):
        reconstruct_response('not an array', np.array([1.0]))
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, 'not an array')
    # Valid sparse order is preserved exactly.
    out = reconstruct_response(values, np.array([2.0, 1.0]), active_indices=[2, 1])
    np.testing.assert_array_equal(out, np.array([10.0 + 3.0 * 2.0 + 2.0 * 1.0]))
    out = reconstruct_response(values, np.zeros(0), active_indices=[])
    np.testing.assert_array_equal(out, np.array([10.0]))


def test_validate_refuses_broken_capsules():
    def broken(**changes):
        capsule = _minimal_capsule()
        capsule.update(changes)
        return capsule

    with pytest.raises(CapsuleValidationError):
        validate_capsule(broken(schema='other'))
    with pytest.raises(CapsuleValidationError):
        validate_capsule(broken(version=2))
    with pytest.raises(CapsuleValidationError):
        validate_capsule(broken(runtime='anysolver-results/2'))
    with pytest.raises(CapsuleValidationError):
        validate_capsule(broken(model_identity=''))
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['cases'] = (capsule['cases'][0], capsule['cases'][0])
        validate_capsule(capsule)
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['responses']['von_mises'] = capsule['responses']['von_mises'][:, :1]
        validate_capsule(capsule)
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['stored'] = {'von_mises': {'case_indices': np.array([5], dtype=np.int64),
                                          'values': np.zeros((1, 3))}}
        validate_capsule(capsule)
    with pytest.raises(CapsuleValidationError):
        capsule = _minimal_capsule()
        capsule['descriptors'] = ()
        validate_capsule(capsule)
    with pytest.raises(CapsuleValidationError):
        validate_capsule('not a mapping')


def test_runtime_has_no_solver_or_scipy_dependency():
    import pathlib
    import re
    package = pathlib.Path(anyfesolver_results.__file__).parent
    for source in package.glob('*.py'):
        for line in source.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if re.match(r'(import|from)\s+', stripped):
                assert 'scipy' not in stripped, (source.name, line)
                assert 'anyfesolver.' not in stripped and 'anyfesolver ' not in stripped.replace('anyfesolver_results', ''), (source.name, line)
                assert 'ANYfem' not in stripped and 'ANYpost' not in stripped, (source.name, line)


def test_errors_are_typed():
    for error in (CapsuleValidationError(), ReconstructionError(), DerivationError()):
        assert isinstance(error, CapsuleError)


def _parity_matrix(seed=17, rows=6, columns=5):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-5.0, 10.0, (rows, columns))
    values[:, 0] += 3.0
    return values


def test_prepared_parity_dense_sparse_reversed_and_empty():
    values = _parity_matrix()
    for active in (None, [1, 2, 3, 4], [3, 1], [4, 2, 1, 3], [],
                   np.array([2, 4], dtype=np.intp)):
        prepared = prepare_response(values, active_indices=active)
        weights = np.linspace(-2.0, 2.0, prepared.selected.shape[1])
        ordinary = reconstruct_response(values, weights, active_indices=active)
        replay = reconstruct_prepared_response(prepared, weights)
        np.testing.assert_array_equal(replay, ordinary)
        assert replay.dtype == np.float64
        assert not replay.flags.writeable


def test_prepared_parity_c_and_f_input_layouts():
    base = _parity_matrix(seed=23)
    for order in ('C', 'F'):
        values = np.array(base, order=order)
        assert values.flags['C_CONTIGUOUS'] == (order == 'C')
        assert values.flags['F_CONTIGUOUS'] == (order == 'F')
        for active in ([2, 1], [1, 3], []):
            prepared = prepare_response(values, active_indices=active)
            weights = np.linspace(-1.5, 2.5, len(active) if active else 0)
            np.testing.assert_array_equal(
                reconstruct_prepared_response(prepared, weights),
                reconstruct_response(values, weights, active_indices=active))


def test_prepared_parity_zero_and_negative_weights():
    values = _parity_matrix(seed=29)
    for weights in (np.zeros(4), np.array([-1.5, 2.5, 0.0, -3.0]),
                    np.array([0.0, 0.0, 0.0, 0.0])):
        prepared = prepare_response(values)
        np.testing.assert_array_equal(
            reconstruct_prepared_response(prepared, weights),
            reconstruct_response(values, weights))
    empty = prepare_response(values, active_indices=[])
    np.testing.assert_array_equal(
        reconstruct_prepared_response(empty, np.zeros(0)), values[:, 0])


def test_prepared_owns_immutable_arrays_and_exact_nbytes():
    values = _parity_matrix(seed=31)
    prepared = prepare_response(values, active_indices=[2, 1])
    assert prepared.baseline.dtype == np.float64
    assert prepared.selected.dtype == np.float64
    assert not prepared.baseline.flags.writeable
    assert not prepared.selected.flags.writeable
    # The retained selected matrix is exactly the ordinary expression's array:
    # original column order and the same memory layout (advanced indexing may
    # legitimately yield an F-contiguous result; the prepared copy preserves
    # whatever layout the ordinary expression materializes).
    ordinary_selected = values[:, [2, 1]]
    assert prepared.selected.strides == ordinary_selected.strides
    assert prepared.selected.flags.c_contiguous == ordinary_selected.flags.c_contiguous
    assert prepared.selected.flags.f_contiguous == ordinary_selected.flags.f_contiguous
    np.testing.assert_array_equal(prepared.selected, ordinary_selected)
    np.testing.assert_array_equal(prepared.baseline, values[:, 0])
    assert type(prepared.nbytes) is int
    assert prepared.nbytes == prepared.baseline.nbytes + prepared.selected.nbytes
    with pytest.raises(ValueError):
        prepared.baseline[0] = 1.0
    with pytest.raises(ValueError):
        prepared.selected[0, 0] = 1.0
    # An empty active order retains the baseline only.
    empty = prepare_response(values, active_indices=[])
    assert empty.selected.shape == (values.shape[0], 0)
    assert empty.nbytes == empty.baseline.nbytes
    assert empty.nbytes == values.shape[0] * 8


def test_prepared_replay_is_independent_of_input_mutation():
    values = _parity_matrix(seed=37)
    pristine = values.copy()
    prepared = prepare_response(values, active_indices=[4, 2, 1, 3])
    values[:] = np.nan
    weights = np.array([1.5, -2.0, 0.0, 3.0])
    replay = reconstruct_prepared_response(prepared, weights)
    np.testing.assert_array_equal(
        replay, reconstruct_response(pristine, weights, active_indices=[4, 2, 1, 3]))
    np.testing.assert_array_equal(
        replay, pristine[:, 0] + pristine[:, [4, 2, 1, 3]] @ weights)


def test_prepare_response_refuses_invalid_admissions():
    values = _parity_matrix(seed=41)
    for bad_active in ([0], [5], [1, 1], [1.5], [True], [[1]], 'x'):
        with pytest.raises(ReconstructionError):
            prepare_response(values, active_indices=bad_active)
    with pytest.raises(ReconstructionError):
        prepare_response(np.zeros(4))
    with pytest.raises(ReconstructionError):
        prepare_response(np.zeros((2, 0)))
    with pytest.raises(ReconstructionError):
        prepare_response('not an array')
    nonfinite = values.copy()
    nonfinite[0, 2] = np.inf
    with pytest.raises(ReconstructionError):
        prepare_response(nonfinite)


def test_reconstruct_prepared_refuses_wrong_type_and_weights():
    values = _parity_matrix(seed=43)
    prepared = prepare_response(values, active_indices=[1, 2])
    for bad in (None, {}, values, [prepared.baseline], 'prepared',
                type('Spoof', (), {'baseline': prepared.baseline,
                                   'selected': prepared.selected,
                                   'nbytes': 0})()):
        with pytest.raises(ReconstructionError):
            reconstruct_prepared_response(bad, np.array([1.0, 2.0]))
    with pytest.raises(ReconstructionError):
        reconstruct_prepared_response(prepared, np.array([1.0]))
    with pytest.raises(ReconstructionError):
        reconstruct_prepared_response(prepared, np.array([[1.0, 2.0]]))
    with pytest.raises(ReconstructionError):
        reconstruct_prepared_response(prepared, np.array([np.nan, 1.0]))
    with pytest.raises(ReconstructionError):
        reconstruct_prepared_response(prepared, 'not an array')
    # Direct construction cannot invent a prepared object.
    with pytest.raises(ReconstructionError):
        PreparedResponse(prepared.baseline, prepared.selected)
    out = reconstruct_prepared_response(prepared, np.zeros(2))
    np.testing.assert_array_equal(out, values[:, 0])
    assert not out.flags.writeable


def test_descriptor_reconstruction_extension_admission():
    """Absent reconstruction is admitted; a present value must be a string."""
    missing = object()

    def descriptor_with(reconstruction):
        capsule = _minimal_capsule()
        descriptor = dict(capsule['descriptors'][0])
        if reconstruction is missing:
            descriptor.pop('reconstruction', None)
        else:
            descriptor['reconstruction'] = reconstruction
        capsule['descriptors'] = (descriptor,)
        return capsule

    validate_capsule(descriptor_with(missing))
    validate_capsule(descriptor_with('compiled_affine_pattern_basis'))
    validate_capsule(descriptor_with('explicit_per_case_values'))
    # A present None is not a string: the adopted extension refuses it.
    with pytest.raises(CapsuleValidationError):
        validate_capsule(descriptor_with(None))
    with pytest.raises(CapsuleValidationError):
        validate_capsule(descriptor_with(5))


def test_completed_statuses_match_consumer_contract():
    """The certified-status gate is exactly the consumer's completed set."""
    assert COMPLETED_STATUSES == ('completed', 'complete', 'success',
                                  'succeeded', 'ok', 'converged')
    for status in COMPLETED_STATUSES:
        capsule = _minimal_capsule()
        case = dict(capsule['cases'][0])
        case['status'] = status
        capsule['cases'] = (case, capsule['cases'][1])
        validate_capsule(capsule)
    for status in ('partial', 'in_progress', ''):
        capsule = _minimal_capsule()
        case = dict(capsule['cases'][0])
        case['status'] = status
        capsule['cases'] = (case, capsule['cases'][1])
        with pytest.raises(CapsuleValidationError):
            validate_capsule(capsule)
    # A non-certified case keeps its failure status and stays discoverable.
    capsule = _minimal_capsule()
    case = dict(capsule['cases'][0])
    case.update(status='failed', certified=False, portable=False)
    capsule['cases'] = (case, capsule['cases'][1])
    validate_capsule(capsule)


def test_duplicate_quantity_component_pairs_refused():
    """Reader queries by (quantity_id, component) stay unambiguous."""
    capsule = _minimal_capsule()
    duplicate = dict(capsule['descriptors'][0])
    duplicate['key'] = 'von_mises_copy'
    capsule['descriptors'] = capsule['descriptors'] + (duplicate,)
    with pytest.raises(CapsuleValidationError):
        validate_capsule(capsule)
    # The same component under a different quantity id is a distinct pair.
    distinct = dict(duplicate)
    distinct['quantity_id'] = 'other_quantity'
    capsule['descriptors'] = capsule['descriptors'][:1] + (distinct,)
    validate_capsule(capsule)


def test_oversized_indices_and_weights_refuse_typed():
    """2**100 indices and 10**400 weights are typed refusals, never OverflowError."""
    values = np.array([[10.0, 2.0, 3.0]])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=[2**100])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, np.array([1.0]), active_indices=np.array([2**100]))
    with pytest.raises(ReconstructionError):
        prepare_response(values, active_indices=[2**100])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, [10**400])
    with pytest.raises(ReconstructionError):
        reconstruct_response(values, 10**400, active_indices=[1])
    prepared = prepare_response(values, active_indices=[1])
    with pytest.raises(ReconstructionError):
        reconstruct_prepared_response(prepared, [10**400])
    with pytest.raises(ReconstructionError):
        reconstruct_prepared_response(prepared, 10**400)


def _two_pattern_capsule():
    capsule = _minimal_capsule()
    capsule['patterns'] = ('p', 'q')
    capsule['coefficients'] = np.array([[1.5, -0.5]], dtype=np.float64)
    capsule['cases'] = ({'case_id': 'a', 'status': 'converged', 'certified': True,
                         'portable': True,
                         'route': 'prepared_static_response/certified_library'},)
    capsule['responses']['von_mises'] = np.array([[0.0, 1.0, 2.0],
                                                  [0.0, 2.0, -1.0]], dtype=np.float64)
    capsule['associations']['von_mises'] = np.array([1, 1], dtype=np.int64)
    return capsule


def test_case_active_indices_admission():
    """Per-case active_indices validate like a reconstruction request."""
    def capsule_with_active(active):
        capsule = _two_pattern_capsule()
        case = dict(capsule['cases'][0])
        case['active_indices'] = active
        capsule['cases'] = (case,)
        return capsule

    # Valid sparse order and the empty active set are admitted as given.
    # (Cases are plain JSON, so only list/tuple forms are admissible here;
    # array-form indices are exercised by the reconstruction tests.)
    for active in ([2, 1], [1], []):
        validate_capsule(capsule_with_active(active))
    # Duplicates, fractional, boolean, out-of-range and oversized values are
    # typed capsule refusals.
    for bad in ([1, 1], [1.5], [0], [3], [True], [2**100], 'x', [[1]]):
        with pytest.raises(CapsuleValidationError):
            validate_capsule(capsule_with_active(bad))
