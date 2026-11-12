"""Focused correctness/activation checks; no timings or numerical campaign."""
import numpy as np
import pytest
from anyfesolver import RecoveryConfig, prepare_static_evaluator, PreparedStaticCapabilityError
from anyfesolver.fe_core import FEModel
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.prepared_static_recovery import _capture_owned_local_recovery, _recover_owned_local, owned_recovery_scope
from anyfesolver import prepared_static_q4 as staged


def _q4_model(sign=1, warped=False):
    model = FEModel('staged-q4')
    model.add_material('steel', 210e9, .3, density=7800.)
    ax, ay = .37, -.26
    rx = np.array([[1.,0.,0.],[0.,np.cos(ax),-np.sin(ax)],[0.,np.sin(ax),np.cos(ax)]])
    ry = np.array([[np.cos(ay),0.,np.sin(ay)],[0.,1.,0.],[-np.sin(ay),0.,np.cos(ay)]])
    rotation = ry @ rx
    coordinates = np.array([[0.,0.,0.],[2.4,.2,0.],[2.1,1.7,0.],[-.1,1.,0.]])
    if warped:
        coordinates[2,2] = .03
    coordinates = coordinates @ rotation.T
    for index, point in enumerate(coordinates, 1):
        model.add_node(index, *point)
    element = QualifiedE4PLShellElement(1, [1,2,3,4], 'steel', thickness=.02,
        reference_normal=sign * rotation[:,2])
    model.add_element(1, element)
    return model, element


def _equal(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize('sign', [1,-1])
def test_staged_q4_rotated_skewed_director_full_field_exact_parity(sign):
    model, element = _q4_model(sign)
    material = model.get_material('steel')
    recipe = _capture_owned_local_recovery(model, element)
    assert recipe.staged_kernel is not None
    rng = np.random.default_rng(631)
    for amplitude in (1e-10, 1e-5, .01):
        u = rng.normal(size=24) * amplitude
        u[0::6] += .3
        expected = element.compute_stresses(model.mesh, u, material, return_global=False)
        with owned_recovery_scope((recipe,)):
            actual = _recover_owned_local(recipe, u)
        _equal(actual, expected)
    for name in ('transform','solution','stationary','coupling'):
        with pytest.raises(ValueError):
            getattr(recipe.staged_kernel, name).setflags(write=True)


def test_staged_q4_arithmetic_does_not_dispatch_native_stationary_solve(monkeypatch):
    # Isolate the arithmetic kernel. Public execution independently validates
    # unchanged native authority, so monkeypatched natives are refused there.
    model, element = _q4_model()
    recipe = _capture_owned_local_recovery(model, element)
    assert recipe.staged_kernel is not None
    calls = []
    def unexpected(*args, **kwargs):
        calls.append(True)
        raise AssertionError('stationary solve repeated during staged arithmetic')
    monkeypatch.setattr(staged._native, '_solve_stationary_system', unexpected)
    for scale in (0., 1e-8, 1e-4):
        staged.recover_q4_stages(recipe.staged_kernel, np.arange(24.) * scale)
    assert calls == []
    with pytest.raises(PreparedStaticCapabilityError):
        with owned_recovery_scope((recipe,)):
            _recover_owned_local(recipe, np.zeros(24))


@pytest.mark.parametrize('mutation', ['binding','code'])
def test_staged_q4_runtime_kernel_authority_before_owned_buffer_dispatch(monkeypatch, mutation):
    model, element = _q4_model()
    recipe = _capture_owned_local_recovery(model, element)
    observed = []
    if mutation == 'binding':
        monkeypatch.setattr(staged, 'recover_q4_stages', lambda *args: observed.append(args))
    else:
        method = staged.recover_q4_stages
        monkeypatch.setattr(method, '__code__', method.__code__.replace(co_filename='changed-staged-kernel'))
    with pytest.raises(PreparedStaticCapabilityError, match='kernel executable authority'):
        with owned_recovery_scope((recipe,)):
            _recover_owned_local(recipe, np.zeros(24))
    assert observed == []


def test_staged_q4_warped_falls_back_without_narrowing_native_capability():
    model, element = _q4_model(warped=True)
    recipe = _capture_owned_local_recovery(model, element)
    assert recipe.staged_kernel is None
    assert recipe.staged_fallback_reason == 'warped_geometry'
    u = np.arange(24.) * 1e-7
    expected = element.compute_stresses(model.mesh, u, model.get_material('steel'), return_global=False)
    with owned_recovery_scope((recipe,)):
        _equal(_recover_owned_local(recipe, u), expected)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_staged_q4_public_activation_fallback_and_source_deletion(batch_size):
    import sys
    import gc
    import weakref
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from benchmark_static_batches import mixed_free_model, free_cases
    from anyfesolver import StaticCase, iter_linear_batches
    model, groups, balancing = mixed_free_model()
    cases = [StaticCase(name, load, instant) for name, instant, load in free_cases(model, groups, balancing, [0,48])] * 16
    config = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=False)
    expected_blocks = list(iter_linear_batches(model, cases, recovery_config=config, batch_size=batch_size))
    owner = prepare_static_evaluator(model, recovery_config=config, envelope_recovery_config=config)
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    blocks = list(owner.iter_batches(cases, batch_size=batch_size))
    for expected, actual in zip(expected_blocks, blocks):
        np.testing.assert_array_equal(actual.displacements, expected.displacements)
        for reference_result, prepared_result in zip(expected.case_results, actual.case_results):
            for eid, fields in reference_result["element_stresses"].items():
                _equal(prepared_result["element_stresses"][eid], fields)
    summary = owner.envelope_result()
    assert summary.complete
    assert summary.diagnostics['prepared_q4_staged_elements']
    assert summary.diagnostics['native_fallback_reasons']
    assert blocks[0].case_results[0]['solver_info']['recovery']['prepared_q4_staged_elements']
    owner.close()

@pytest.mark.parametrize('feature', ['initial_field', 'generalized', 'orthotropic'])
def test_staged_q4_explicit_feature_eligibility_fallback(feature):
    model, element = _q4_model()
    material = model.get_material('steel')
    if feature == 'initial_field':
        # Predicate test only: introducing committed-state loading is outside
        # this static prototype. Explicit initial input data disables staging.
        object.__setattr__(element, 'initial_field_provenance', {})
        expected_reason = 'initial_field_inputs'
    elif feature == 'generalized':
        from test_generalized_state_recovery import _shell_model
        model, element = _shell_model()
        material = model.get_material(element.material_name)
        expected_reason = 'generalized_section'
    else:
        from test_e4_pl_planar_physical_recovery import _orthotropic
        material = _orthotropic()
        expected_reason = 'nonisotropic_material'
    kernel, reason = staged.compile_q4_stages(element, model.mesh, material)
    assert kernel is None
    assert reason == expected_reason
