"""M6 canonical mass capture; code-event counts are correctness, not timings."""
import sys
from collections import Counter
import numpy as np
import pytest
from anyfesolver import (LoadCase, StaticCase, RecoveryConfig, prepare_static_evaluator,
    PreparedStaticCapabilityError, CancellationToken, SolveCancelled)
from anyfesolver.elements import BeamElement, CoupledBeamShellElement
from anyfesolver.matrix_assembly import assemble_load_vector
from anyfesolver.prepared_static_loads import compile_canonical_load_maps, ingest_static_case, evaluate_load_vector
from test_prepared_static_evaluator import _bar, _case, _shell_model


def test_acceleration_capture_has_two_native_mass_passes_and_one_oracle_assembly():
    model = _bar()
    model.apply_boundary_conditions()
    counts = Counter()
    codes = {BeamElement.compute_mass_matrix.__code__: 'mass', assemble_load_vector.__code__: 'assembly'}
    def observer(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1
    previous = sys.getprofile()
    try:
        sys.setprofile(observer)
        maps = compile_canonical_load_maps(model, families=('acceleration',))
    finally:
        sys.setprofile(previous)
    assert counts == {'mass': 2, 'assembly': 1}
    assert not hasattr(maps, 'acceleration_map')
    assert len(maps.mass_maps) == 1
    assert dict(maps.verification)['acceleration'] <= 1e-11
    assert sys.getprofile() is previous


@pytest.mark.parametrize('factory', [_bar, _shell_model])
def test_full_F_exact_axes_general_added_mass_activity_and_snapshot_isolation(factory):
    from anyfesolver.activity import ElementActivity, ElementActivityPolicy
    model = factory()
    ids = tuple(model.mesh.elements)
    activity = ElementActivity(ids, .375, policy=ElementActivityPolicy(load='activity', mass='delete_only'))
    model.set_element_activity(activity)
    model.add_point_mass(1, 1.23456789)
    accelerations = [(1.,0.,0.), (0.,-2.,0.), (0.,0.,3.), (.1723456,-.438901,2.761234)]
    cases = []
    for index, acceleration in enumerate(accelerations):
        load = LoadCase(str(index))
        load.gravity = np.asarray(acceleration)
        load.added_node_masses[1] = .37 * index
        load.added_node_masses[2] = 1.19 / (index + 1)
        cases.append(StaticCase(str(index), load))
    expected = [assemble_load_vector(model, case.load_case)[0] for case in cases]
    config = RecoveryConfig(include_stresses=False, include_reactions=False)
    with prepare_static_evaluator(model, recovery_config=config, load_families=['acceleration']) as owner:
        maps = owner._snapshot.load_maps
        assert dict(maps.verification)['acceleration'] <= 1e-11
        assert isinstance(maps.mass_maps, tuple)
        for mapping, scale, matrix in maps.mass_maps:
            assert scale == .375
            with pytest.raises(ValueError):
                mapping.setflags(write=True)
            with pytest.raises(ValueError):
                matrix.setflags(write=True)
        model.materials['steel'].density *= 11.
        model.mesh.point_masses[1] = 1000.
        activity.set_activity(ids, .125, reason='post-snapshot mutation')
        for case, oracle in zip(cases, expected):
            actual = evaluate_load_vector(maps, ingest_static_case(case)[2])[0]
            np.testing.assert_array_equal(actual, oracle)
            assert np.any(actual[:6] != 0.)  # full constrained-node load survives


def test_massless_coupling_is_captured_and_added_mass_applied_once():
    model = _bar()
    model.add_node(3, 0., 0., 0.)
    model.add_element(2, CoupledBeamShellElement(2, 3, 1, 'steel'))
    model.apply_boundary_conditions()
    maps = compile_canonical_load_maps(model, families=('acceleration',))
    assert len(maps.mass_maps) == 2
    for added in (0., 1.23, 4.56):
        case = _case(0, force=0.)
        case.load_case.nodal_loads.clear()
        case.load_case.gravity = np.array([.123, -2.34, 5.67])
        case.load_case.added_node_masses[3] = added
        expected = assemble_load_vector(model, case.load_case)[0]
        np.testing.assert_array_equal(evaluate_load_vector(maps, ingest_static_case(case)[2])[0], expected)


def test_mass_capture_failure_discards_partial_maps_and_keeps_nodal_family():
    model = _bar()
    model.add_material('missing-density', 100., .3, density=1.)
    model.materials['missing-density'].density = None
    model.add_element(2, BeamElement(2, [1,2], 'missing-density',
        {'area':1., 'Iy':.001, 'Iz':.001, 'J':.001}))
    model.apply_boundary_conditions()
    maps = compile_canonical_load_maps(model, families=('nodal', 'acceleration'))
    assert not maps.acceleration_available
    assert maps.mass_maps == ()
    assert 'TypeError' in maps.acceleration_note
    case = _case(0)
    np.testing.assert_array_equal(evaluate_load_vector(maps, ingest_static_case(case)[2])[0],
                                  assemble_load_vector(model, case.load_case)[0])
    case.load_case.gravity = np.array([1.,2.,3.])
    with pytest.raises(PreparedStaticCapabilityError, match='acceleration load family'):
        evaluate_load_vector(maps, ingest_static_case(case)[2])


def test_cancel_at_mass_phase_boundary_still_propagates():
    token = CancellationToken()
    def progress(event):
        if event.stage == 'prepared_static.prepare.loads':
            token.cancel('after mass capture')
    with pytest.raises(SolveCancelled, match='after mass capture'):
        prepare_static_evaluator(_bar(), recovery_config=RecoveryConfig(include_stresses=False),
            load_families=['acceleration'], cancellation_token=token, progress_callback=progress)


@pytest.mark.parametrize('density, elements', [(float('nan'), 1), (float('inf'), 1), (1.7e308, 3)])
def test_nonfinite_or_accumulated_overflow_mass_keeps_nodal_route(density, elements):
    model = _bar()
    model.materials['steel'].density = density
    for eid in range(2, elements + 1):
        model.add_element(eid, BeamElement(eid, [1,2], 'steel',
            {'area':1., 'Iy':.001, 'Iz':.001, 'J':.001}))
    model.apply_boundary_conditions()
    if elements > 1:
        # Every individual matrix is finite; only canonical full-F accumulation
        # overflows. Rejecting merely nonfinite matrices would miss this case.
        assert np.all(np.isfinite(model.mesh.elements[1].compute_mass_matrix(model.mesh, model.materials['steel'])))
    maps = compile_canonical_load_maps(model, families=('nodal', 'acceleration'))
    assert not maps.acceleration_available
    assert maps.mass_maps == ()
    assert 'nonfinite canonical unit-acceleration load' in maps.acceleration_note
    case = _case(0)
    np.testing.assert_array_equal(evaluate_load_vector(maps, ingest_static_case(case)[2])[0],
                                  assemble_load_vector(model, case.load_case)[0])
