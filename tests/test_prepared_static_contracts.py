"""Integration contracts for prepared ownership and production recovery."""
import numpy as np
import pytest
from test_prepared_static_evaluator import _bar, _case, _shell_model, _shell_case
from anyfesolver import (AnalysisSession, RecoveryConfig, StaticCase, prepare_static_evaluator,
                       solve_linear, PreparedStaticError, LoadCase)
from anyfesolver.recovery import recover_stress_result
from anyfesolver.prepared_static_loads import ingest_static_case, evaluate_load_vector
from anyfesolver.matrix_assembly import assemble_load_vector


def test_factory_session_selection_and_iterator_cleanup():
    model = _bar()
    session = AnalysisSession(model)
    evaluator = prepare_static_evaluator(model, session=session, output_dofs=[6])
    session.close()
    done = []
    def source():
        try:
            for i in range(5):
                yield _case(i)
        finally:
            done.append(True)
    stream = evaluator.iter_batches(source(), batch_size=1)
    block = next(stream)
    assert block.output_dofs.tolist() == [6]
    assert block.displacements.shape == (1, 1)
    stream.close()
    assert done == [True]
    evaluator.close()
    evaluator.close()
    assert evaluator.diagnostics()['counters']['close_count'] == 1


def test_provider_failure_closes_and_preserves_certified_block():
    model = _bar()
    done = []
    def source():
        try:
            yield _case(0)
            raise ValueError('provider broken')
        finally:
            done.append(True)
    with prepare_static_evaluator(model) as owner:
        stream = owner.iter_batches(source(), batch_size=1)
        assert next(stream).case_ids == ('0',)
        with pytest.raises(PreparedStaticError) as error:
            next(stream)
        assert error.value.diagnostics['input_index'] == 1
        assert owner.diagnostics()['cases_certified'] == 1
    assert done


def test_selected_derived_recovery_and_actual_full_load():
    model = _shell_model()
    recovery = RecoveryConfig(components=['von_mises'], include_stresses=True, include_reactions=True)
    case = _shell_case(1, pressure=2.5, gravity=True, added=4)
    actual_F, _ = assemble_load_vector(model, case.load_case)
    u, _ = solve_linear(model, case.load_case)
    oracle = recover_stress_result(model, u, recovery)
    with prepare_static_evaluator(model, recovery_config=recovery) as owner:
        _, _, owned = ingest_static_case(case)
        compiled_F, _ = evaluate_load_vector(owner._snapshot.load_maps, owned)
        np.testing.assert_allclose(compiled_F, actual_F, rtol=1e-13, atol=1e-12)
        result = next(owner.iter_batches([case])).case_results[0]
        for element, values in oracle.element_stresses.items():
            assert set(result['element_stresses'][element]) == set(values)
            np.testing.assert_allclose(result['element_stresses'][element]['von_mises'], values['von_mises'], rtol=1e-10)


def test_production_s3_v2d_full_load_and_recovery():
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    model = _shell_model()
    for eid, old in tuple(model.mesh.elements.items()):
        model.mesh.elements[eid] = NativeParityE4PLS3V2DShellElement(eid, old.node_ids, 'steel', thickness=.01, reference_normal=(0., 0., 1.))
    case = _shell_case(1, pressure=2.5, gravity=True, added=4)
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    u, _ = solve_linear(model, case.load_case)
    oracle = recover_stress_result(model, u, recovery)
    full_F, _ = assemble_load_vector(model, case.load_case)
    with prepare_static_evaluator(model, recovery_config=recovery) as owner:
        _, _, owned = ingest_static_case(case)
        prepared_F, _ = evaluate_load_vector(owner._snapshot.load_maps, owned)
        np.testing.assert_allclose(prepared_F, full_F, rtol=1e-12, atol=1e-12)
        block = next(owner.iter_batches([case]))
        np.testing.assert_allclose(block.displacements[:, 0], u, rtol=1e-10, atol=1e-13)
        for eid, fields in oracle.element_stresses.items():
            for name, expected in fields.items():
                actual = block.case_results[0]['element_stresses'][eid][name]
                if np.asarray(expected).dtype.kind in 'biufc':
                    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
                else:
                    assert actual == expected


def test_generalized_beam_availability_and_coupling_acceleration():
    from anyfesolver import FixedSupport, BoundaryCondition
    from anyfesolver.elements import CoupledBeamShellElement
    from test_generalized_state_recovery import _beam_model
    model, element = _beam_model()
    model.add_boundary_condition(FixedSupport('fixed', [1]))
    model.add_boundary_condition(BoundaryCondition('slide', [2], {'uy':0., 'uz':0., 'rx':0., 'ry':0., 'rz':0.}))
    # A native massless coupling participates in canonical gravity assembly.
    model.add_node(3, 0., 0., 0.)
    model.add_element(2, CoupledBeamShellElement(2, 3, 1, 'dummy'))
    recovery = RecoveryConfig(element_ids=[1], include_stresses=True, include_reactions=True)
    case = _case(0)
    u, _ = solve_linear(model, case.load_case)
    oracle = recover_stress_result(model, u, recovery)
    with prepare_static_evaluator(model, recovery_config=recovery, load_families=['nodal']) as owner:
        result = next(owner.iter_batches([case])).case_results[0]['element_stresses'][1]
        assert set(result) == set(oracle.element_stresses[1])
        for name, value in oracle.element_stresses[1].items():
            if np.asarray(value).dtype.kind in 'biufc':
                np.testing.assert_allclose(result[name], value, rtol=1e-12, atol=1e-12)
            else:
                assert result[name] == value


def test_native_massless_coupling_gravity_full_F():
    from anyfesolver.elements import CoupledBeamShellElement
    model = _bar()
    model.add_node(3, 0., 0., 0.)
    model.add_element(2, CoupledBeamShellElement(2, 3, 1, 'steel'))
    case = _case(0)
    case.load_case.set_gravity(1., -2., 3.)
    expected, _ = assemble_load_vector(model, case.load_case)
    with prepare_static_evaluator(model) as owner:
        _, _, owned = ingest_static_case(case)
        actual, _ = evaluate_load_vector(owner._snapshot.load_maps, owned)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_unknown_derived_component_is_explicitly_refused():
    from anyfesolver import PreparedStaticCapabilityError
    model = _bar()
    recovery = RecoveryConfig(include_stresses=True, components=['principal_1'])
    with pytest.raises(PreparedStaticCapabilityError, match='source-reviewed') as error:
        prepare_static_evaluator(model, recovery_config=recovery)
    assert 'unavailable' in error.value.diagnostics['refused'][0]['reason']


def test_reaction_diagnostic_metadata_is_detached():
    from anyfesolver.prepared_static_recovery import constraint_force_diagnostics
    model = _bar()
    with prepare_static_evaluator(model) as owner:
        snap = owner._snapshot
        nested = {'nested': {'value': [1, 2]}}
        force = constraint_force_diagnostics(snap.reaction_maps, snap.stiffness,
            snap.transformation, snap.rigid_modes, np.zeros(snap.total_dofs),
            np.zeros(snap.total_dofs), constraint_info=nested, nullspace_info=nested)
        force['constraint_info']['nested']['value'][0] = 99
        force['nullspace_info']['nested']['value'].append(3)
        assert nested == {'nested': {'value': [1, 2]}}


def test_finalizer_releases_abandoned_close_and_failure_once(monkeypatch):
    import gc
    import importlib
    import weakref
    module = importlib.import_module('anyfesolver.prepared_static')
    releases = []
    original = module.AutoSparseSolverBackend.release_pattern_slots
    def release(backend):
        releases.append(id(backend))
        return original(backend)
    monkeypatch.setattr(module.AutoSparseSolverBackend, 'release_pattern_slots', release)
    model = _bar()
    owner = prepare_static_evaluator(model)
    backend_id = id(owner._backend)
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is None
    assert releases.count(backend_id) == 1
    owner = prepare_static_evaluator(model)
    before = len(releases)
    owner.close()
    owner.close()
    del owner
    gc.collect()
    assert len(releases) == before + 1
    def fail(*args, **kwargs):
        raise ValueError('capture failed')
    monkeypatch.setattr(module, 'capture_prepared_static_snapshot', fail)
    before = len(releases)
    with pytest.raises(PreparedStaticError, match='snapshot capture'):
        prepare_static_evaluator(model)
    gc.collect()
    assert len(releases) == before + 1


def test_numpy_scalar_pressure_and_static_follower_full_F():
    model = _shell_model()
    case = _shell_case(1, pressure=1., follower=True)
    case.load_case.pressure_loads[1] = np.float64(3.25)
    expected, _ = assemble_load_vector(model, case.load_case)
    with prepare_static_evaluator(model) as owner:
        _, _, owned = ingest_static_case(case)
        actual, _ = evaluate_load_vector(owner._snapshot.load_maps, owned)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        assert owner.diagnostics()['snapshot']['phase_timings_seconds']['load_maps'] >= 0.


def test_in_place_native_recovery_code_mutation_refused(monkeypatch):
    from anyfesolver.elements import BeamElement
    from anyfesolver import PreparedStaticCapabilityError
    model = _bar()
    function = BeamElement.compute_stresses
    changed = tuple(.25 if type(value) is float and value == .5 else value for value in function.__code__.co_consts)
    monkeypatch.setattr(function, '__code__', function.__code__.replace(co_consts=changed))
    with pytest.raises(PreparedStaticCapabilityError, match='authority changed'):
        prepare_static_evaluator(model, recovery_config=RecoveryConfig(include_stresses=True))


def test_custom_material_not_qualified_by_random_probes():
    from anymaterial import IsotropicMaterial
    from anyfesolver.prepared_static_recovery import _recovery_admission
    class ForeignMaterial(IsotropicMaterial):
        pass
    model = _bar()
    model.materials['steel'] = ForeignMaterial(name='steel', elastic_modulus=100., poisson_ratio=.3, density=2.)
    admitted, _, reason = _recovery_admission(model, model.mesh.elements[1], RecoveryConfig())
    assert not admitted
    assert 'exact reviewed builtin' in reason
    with pytest.raises(PreparedStaticError):
        prepare_static_evaluator(model, recovery_config=RecoveryConfig(include_stresses=True))


def test_instance_helper_shadow_and_custom_numeric_refused():
    from anyfesolver.prepared_static_loads import exact_element_authority, native_material_section_authority
    model = _bar()
    element = model.mesh.elements[1]
    element._get_element_displacements = lambda *args: np.zeros(12)
    assert not exact_element_authority(element)[0]
    with pytest.raises(PreparedStaticError):
        prepare_static_evaluator(model, recovery_config=RecoveryConfig(include_stresses=True))
    del element._get_element_displacements
    class FloatProvider:
        def __float__(self):
            raise AssertionError('custom numeric callback must not run')
    model.materials['steel'].elastic_modulus = FloatProvider()
    assert not native_material_section_authority(model, element)[0]


def test_packet_memory_checked_before_numeric_copy():
    # Replacing a module function after import is outside the supported contract
    # (the response authority check refuses it), so the order is checked through
    # the policy's own 'before copy' refusal.
    from anyfesolver import ResourceConfig
    from anyfesolver.recovery import ResourcePolicyError
    model = _bar()
    case = _case(1)
    case.load_case.nodal_loads[99999] = np.zeros(200000)
    with prepare_static_evaluator(model, resource_config=ResourceConfig(memory_limit_bytes=100000)) as owner:
        with pytest.raises(ResourcePolicyError, match='before copy'):
            next(owner.iter_batches([case], batch_size=1))


def test_early_close_error_and_secondary_failure_notes():
    class Source:
        def __iter__(self): return self
        def __next__(self): return _case(0)
        def close(self): raise ValueError('cleanup exploded')
    with prepare_static_evaluator(_bar()) as owner:
        stream = owner.iter_batches(Source(), batch_size=1)
        next(stream)
        with pytest.raises(ValueError, match='cleanup exploded'):
            stream.close()
    class FailedSource(Source):
        def __next__(self): raise LookupError('provider exploded')
    with prepare_static_evaluator(_bar()) as owner:
        with pytest.raises(PreparedStaticError, match='provider failed') as error:
            next(owner.iter_batches(FailedSource()))
        assert any('cleanup exploded' in note for note in error.value.__notes__)


def test_close_during_input_or_progress_cannot_yield_success():
    model = _bar()
    owner = prepare_static_evaluator(model)
    def source():
        owner.close()
        yield StaticCase('zero', None)
    with pytest.raises(RuntimeError, match='closed'):
        next(owner.iter_batches(source()))
    owner = prepare_static_evaluator(model)
    with pytest.raises(RuntimeError, match='closed'):
        next(owner.iter_batches([StaticCase('zero', None)], progress_callback=lambda event: owner.close()))


def test_memory_policy_splits_individually_fitting_variable_packets():
    from anyfesolver import ResourceConfig
    model = _bar()
    cases = [_case(0), _case(1)]
    for case in cases:
        case.load_case.nodal_loads[99999] = np.zeros(3000)
    with prepare_static_evaluator(model, resource_config=ResourceConfig(memory_limit_bytes=80000)) as owner:
        blocks = list(owner.iter_batches(cases, batch_size=2))
    assert [block.case_ids for block in blocks] == [('0',), ('1',)]
    assert all(block.diagnostics['estimated_block_bytes'] <= 80000 for block in blocks)


def test_principal_measure_analytic_rotation_and_eigen_residual():
    from anyfesolver.prepared_static_measures import principal_stresses
    tensors = [np.diag([3., -2., 1.]), np.array([[0., 4., 0.], [4., 0., 0.], [0., 0., 0.]])]
    expected = [[3., 1., -2.], [4., 0., -4.]]
    angle = .713
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    for tensor, values in zip(tensors, expected):
        np.testing.assert_allclose(principal_stresses(tensor), values, rtol=1e-12, atol=1e-12)
        rotated = rotation @ tensor @ rotation.T
        np.testing.assert_allclose(principal_stresses(rotated), values, rtol=1e-12, atol=1e-12)
        eigenvalues, vectors = np.linalg.eigh(rotated)
        np.testing.assert_allclose(rotated @ vectors, vectors * eigenvalues, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(principal_stresses(rotated), eigenvalues[::-1], rtol=1e-12, atol=1e-12)


def test_principal_shell_public_path_and_beam_refusal():
    from anyfesolver.prepared_static_measures import SHELL_LINEAR_COMPONENTS
    from anyfesolver import PreparedStaticCapabilityError
    model = _shell_model()
    case = _shell_case(0, pressure=2.)
    selection = ['principal_stresses_top', 'principal_stresses_bottom']
    recovery = RecoveryConfig(components=selection + list(SHELL_LINEAR_COMPONENTS), include_stresses=True)
    with prepare_static_evaluator(model, recovery_config=recovery) as owner:
        block = next(owner.iter_batches([case]))
        # Isolate the new derived operator from the separately recorded affine
        # recovery roundoff gate: use exactly its returned physical linear fields.
        for eid, fields in block.case_results[0]['element_stresses'].items():
            for surface, sign in (('top', 1.), ('bottom', -1.)):
                xx = fields['membrane_xx'] + sign*fields['bending_xx']
                yy = fields['membrane_yy'] + sign*fields['bending_yy']
                xy = fields['membrane_xy'] + sign*fields['bending_xy']
                tensor = np.zeros(np.shape(xx) + (3, 3))
                tensor[...,0,0], tensor[...,1,1] = xx, yy
                tensor[...,0,1] = tensor[...,1,0] = xy
                tensor[...,0,2] = tensor[...,2,0] = fields['shear_xz']
                tensor[...,1,2] = tensor[...,2,1] = fields['shear_yz']
                expected = np.linalg.eigvalsh(tensor)[..., ::-1]
                np.testing.assert_allclose(fields['principal_stresses_'+surface], expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(PreparedStaticCapabilityError, match='source-reviewed'):
        prepare_static_evaluator(_bar(), recovery_config=recovery)


def test_preparation_cancellation_preserves_type_and_reports_phases():
    from anyfesolver import CancellationToken, SolveCancelled
    token = CancellationToken()
    token.cancel('before prepare')
    with pytest.raises(SolveCancelled):
        prepare_static_evaluator(_bar(), cancellation_token=token)
    events = []
    with prepare_static_evaluator(_bar(), progress_callback=events.append) as owner:
        assert owner.diagnostics()['snapshot']['phase_timings_seconds']
    assert events


def test_closure_during_iter_construction_still_closes_input():
    owner = prepare_static_evaluator(_bar())
    closed = []
    class Cases:
        def __iter__(self):
            owner.close()
            return self
        def __next__(self): return StaticCase('zero', None)
        def close(self): closed.append(True)
    with pytest.raises(RuntimeError, match='closed'):
        next(owner.iter_batches(Cases()))
    assert closed == [True]


def test_shadowed_model_material_provider_refused_before_callback():
    from anyfesolver import PreparedStaticCapabilityError
    model = _bar()
    def forbidden(*args):
        raise AssertionError('foreign provider must not run')
    model.get_material = forbidden
    with pytest.raises(PreparedStaticCapabilityError, match='FEModel providers'):
        prepare_static_evaluator(model, recovery_config=RecoveryConfig(include_stresses=True))


def test_pressure_fast_span_uses_canonical_callback_free_proof(monkeypatch):
    from anyfesolver.prepared_static_loads import _pressure_probes_are_package_only
    import anyfesolver.boundary as boundary
    model = _shell_model()
    assert _pressure_probes_are_package_only(model)
    monkeypatch.setattr(boundary, '_numpy_error_routes_are_builtin', lambda: False)
    assert not _pressure_probes_are_package_only(model)
    monkeypatch.undo()
    model.get_material = lambda name: (_ for _ in ()).throw(AssertionError('provider executed'))
    assert not _pressure_probes_are_package_only(model)


def test_pressure_fast_span_uses_full_guards_with_activity_provider():
    from anyfesolver.prepared_static_loads import _pressure_probes_are_package_only
    model = _shell_model()
    class Activity:
        def __getattr__(self, name):
            raise AssertionError('activity provider must not be observed by preflight')
    model.mesh.element_activity = Activity()
    assert not _pressure_probes_are_package_only(model)


def test_shadowed_mapping_items_never_runs_inside_pressure_trust():
    from anyfesolver.prepared_static_loads import _pressure_probes_are_package_only
    from anyfesolver.e4_pl_element import _q4_trusted_operation_accepts
    from anyfesolver.e4_pl_s3_element import _s3_trusted_operation_accepts
    model = _shell_model()
    observed = []
    def shadow_items():
        elements = tuple(dict.values(model.mesh.elements))
        observed.append(any(_q4_trusted_operation_accepts(e) or _s3_trusted_operation_accepts(e) for e in elements))
        return dict.items(model.mesh.elements)
    model.mesh.elements.items = shadow_items
    assert not _pressure_probes_are_package_only(model)
    # The admission decision reads raw mapping state and does not execute it.
    assert observed == []
    with pytest.raises(PreparedStaticError):
        prepare_static_evaluator(model)
    assert not any(observed)


def test_returned_dof_headers_cannot_change_future_labels():
    with prepare_static_evaluator(_bar(), output_dofs=[6]) as owner:
        stream = owner.iter_batches([_case(0), _case(1)], batch_size=1)
        first = next(stream)
        with pytest.raises(ValueError):
            first.output_dofs.setflags(write=True)
        first.output_dofs.shape = (1, 1)
        second = next(stream)
        assert second.output_dofs.shape == (1,)
        assert second.output_dofs.tolist() == [6]


def test_owned_local_refinement_multiple_scales_and_source_gc():
    import gc
    import weakref
    from anyfesolver.prepared_static_recovery import apply_compiled_recovery
    model = _shell_model()
    recovery = RecoveryConfig(include_stresses=True)
    owner = prepare_static_evaluator(model, recovery_config=recovery)
    rng = np.random.default_rng(2718)
    probes = []
    for scale in (1e-9, 1e-5, 1e-2):
        u = rng.normal(size=model.mesh.dof_manager.total_dofs) * scale
        # Rigid translation plus a small deformation tests cancellation/offsets.
        u[0::6] += .2
        expected = recover_stress_result(model, u, recovery).element_stresses
        probes.append((u, expected))
    reference = weakref.ref(model)
    model.mesh.nodes[1].x = 999.
    del model
    gc.collect()
    assert reference() is None
    for u, expected in probes:
        actual = apply_compiled_recovery(owner._snapshot.recovery_compilation, u)[0]
        for eid, fields in expected.items():
            for name, value in fields.items():
                if np.asarray(value).dtype.kind in 'biufc':
                    np.testing.assert_allclose(actual[eid][name], value, rtol=1e-12, atol=1e-12)
    owner.close()


def test_owned_refinement_refuses_post_preparation_class_patch(monkeypatch):
    from anyfesolver.prepared_static_recovery import _recover_owned_local
    from anyfesolver.prepared_static_loads import PreparedStaticCapabilityError
    with prepare_static_evaluator(_shell_model(), recovery_config=RecoveryConfig(include_stresses=True)) as owner:
        recipe = next(iter(owner._snapshot.recovery_compilation.elements.values())).local_recipe
        element = recipe.execution_owner[0]
        monkeypatch.setattr(type(element), 'compute_stresses', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('foreign code executed')))
        with pytest.raises(PreparedStaticCapabilityError, match='authority changed'):
            _recover_owned_local(recipe, np.zeros(6 * len(element.node_ids)))
