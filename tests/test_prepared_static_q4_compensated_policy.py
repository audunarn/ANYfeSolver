"""Exact prepared/native arithmetic policy and its transitive authority."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from anyfesolver import prepared_static_q4 as staged
from anyfesolver.prepared_static_loads import PreparedStaticCapabilityError
from anyfesolver.prepared_static_recovery import (
    _AFFINITY_RELATIVE_TOLERANCE, _capture_owned_local_recovery,
    _recover_owned_local, owned_recovery_scope, trusted_capture_scope,
)
from test_prepared_static_q4 import _q4_model, _equal


@pytest.mark.parametrize('coordinate_type,revision,eligible', ((float, 0, True), (np.float64, 0, True), (np.float32, 0, False), (float, np.int64(0), False)), ids=('float', 'float64', 'float32', 'revision-type'))
@pytest.mark.parametrize('trusted', (False, True))
def test_original_planar_coordinate_policy_parity_and_source_deletion(coordinate_type, revision, eligible, trusted):
    import gc
    import weakref
    from anyfesolver.fe_core import FEModel
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    model = FEModel('original-coordinate-policy')
    model.add_material('steel', 210e9, .3, density=7800.)
    for number, point in enumerate(((0., 0., 0.), (2.4, .2, 0.), (2.1, 1.7, 0.), (-.1, 1., 0.)), 1):
        model.add_node(number, *(coordinate_type(value) for value in point))
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], 'steel', thickness=.02, reference_normal=(0., 0., 1.))
    model.add_element(1, element)
    object.__setattr__(model.mesh.nodes[1], '_coordinate_revision', revision)
    material = model.get_material('steel')
    assert staged._capture_original_q4_policy(element, model.mesh, material) is eligible
    with trusted_capture_scope(trusted):
        recipe = _capture_owned_local_recovery(model, element)
    assert recipe.staged_kernel.compensated is eligible
    rng = np.random.default_rng(631)
    displacements = [rng.normal(size=24) * scale for scale in (1e-10, 1e-5, .01)]
    for displacement in displacements:
        displacement[0::6] += .3
    expected = [element.compute_stresses(model.mesh, displacement, material, return_global=False) for displacement in displacements]
    from anyfesolver import RecoveryConfig, prepare_static_evaluator
    from anyfesolver.prepared_static_loads import PreparedStaticError
    from anyfesolver.matrix_assembly import AssemblyError
    config = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=False)
    public_owner = None
    if type(revision) is not int:
        with pytest.raises(PreparedStaticError, match='prepared static snapshot capture failed') as refused:
            prepare_static_evaluator(model, recovery_config=config, envelope_recovery_config=config)
        assert type(refused.value.__cause__) is AssemblyError
        assert str(refused.value.__cause__) == 'stiffness assembly preflight found incompatible qualified shell authority'
        assert refused.value.diagnostics == {
            'stage': 'snapshot capture',
            'error': 'stiffness assembly preflight found incompatible qualified shell authority',
            'original_exception_type': 'AssemblyError',
        }
        # The captured exception traceback owns source frames until released.
        del refused
    else:
        public_owner = prepare_static_evaluator(model, recovery_config=config, envelope_recovery_config=config)
    reference = weakref.ref(model)
    del model, element, material
    gc.collect()
    assert reference() is None
    with owned_recovery_scope((recipe,)):
        for displacement, result in zip(displacements, expected):
            _equal(_recover_owned_local(recipe, displacement), result)
    if public_owner is not None:
        public_owner.close()


@pytest.mark.parametrize('mutation', ('binding', 'code'))
def test_original_policy_helper_is_authenticated_before_capture(monkeypatch, mutation):
    model, element = _q4_model()
    calls = []
    if mutation == 'binding':
        monkeypatch.setattr(staged, '_capture_original_q4_policy', lambda *args: calls.append(args))
    else:
        helper = staged._capture_original_q4_policy
        monkeypatch.setattr(helper, '__code__', helper.__code__.replace(co_filename='changed-original-policy'))
    with pytest.raises(PreparedStaticCapabilityError, match='kernel executable authority'):
        _capture_owned_local_recovery(model, element)
    assert calls == []


def test_float_convertible_coordinate_is_not_retained_or_made_eligible():
    class Coordinate:
        def __float__(self):
            return 0.
    model, element = _q4_model()
    provider = Coordinate()
    model.mesh.nodes[1].x = provider
    recipe = _capture_owned_local_recovery(model, element)
    assert recipe.staged_kernel.compensated is False
    assert all(type(getattr(node, axis)) is float for node in recipe.execution_owner[1].nodes.values() for axis in ('x', 'y', 'z'))


def test_original_policy_flip_during_capture_is_refused():
    model, element = _q4_model()
    node = model.mesh.nodes[1]
    class Coordinate:
        def __float__(self):
            node.x = 0.
            return 0.
    node.x = Coordinate()
    with pytest.raises(PreparedStaticCapabilityError, match='source Q4 arithmetic policy changed during capture'):
        _capture_owned_local_recovery(model, element)


@pytest.mark.parametrize('sign', (1, -1))
@pytest.mark.parametrize('eligible', (True, False))
def test_native_policy_is_frozen_with_scalar_parity_and_column_probes(sign, eligible):
    model, element = _q4_model(sign)
    material = model.get_material('steel')
    if not eligible:
        material.yield_stress = 1e8
    recipe = _capture_owned_local_recovery(model, element)
    kernel = recipe.staged_kernel
    assert kernel is not None and kernel.compensated is eligible
    with pytest.raises(FrozenInstanceError):
        kernel.compensated = not eligible
    rng = np.random.default_rng(631)
    columns = rng.normal(size=(24, 3)) * np.array((1e-10, 1e-5, .01))
    columns[0::6] += .3
    made = staged.recover_q4_stage_columns(kernel, columns)
    for index in range(columns.shape[1]):
        expected = element.compute_stresses(model.mesh, columns[:, index], material, return_global=False)
        with owned_recovery_scope((recipe,)):
            actual = _recover_owned_local(recipe, columns[:, index])
        _equal(actual, expected)
        if eligible:
            for name, values in made.items():
                observed = values[..., index]
                scale = max(float(np.linalg.norm(observed)), float(np.linalg.norm(expected[name])), 1.)
                assert np.linalg.norm(observed - expected[name]) / scale <= _AFFINITY_RELATIVE_TOLERANCE, name
    empty = staged.recover_q4_stage_columns(kernel, np.empty((24, 0)))
    assert empty and all(values.shape[-1] == 0 for values in empty.values())


@pytest.mark.parametrize('name', ('product', 'dot', '_product_parts', 'sum_arrays', 'weighted_sum'))
@pytest.mark.parametrize('mutation', ('binding', 'code'))
def test_transitive_contraction_mutation_is_refused_before_dispatch(monkeypatch, name, mutation):
    model, element = _q4_model()
    recipe = _capture_owned_local_recovery(model, element)
    function = getattr(staged._contractions, name)
    calls = []
    if mutation == 'binding':
        monkeypatch.setattr(staged._contractions, name, lambda *args: calls.append(args))
    else:
        monkeypatch.setattr(function, '__code__', function.__code__.replace(co_filename='changed-contraction'))
    with pytest.raises(PreparedStaticCapabilityError, match='contraction executable authority'):
        _recover_owned_local(recipe, np.zeros(24))
    assert calls == []


def test_transitive_scalar_global_mutation_is_refused(monkeypatch):
    model, element = _q4_model()
    recipe = _capture_owned_local_recovery(model, element)
    calls = []
    monkeypatch.setattr(staged._contractions, '_fsum', lambda values: calls.append(values))
    with pytest.raises(PreparedStaticCapabilityError, match='contraction executable authority'):
        _recover_owned_local(recipe, np.zeros(24))
    assert calls == []


@pytest.mark.parametrize('mutation', ('binding', 'code', 'global'))
def test_native_policy_authority_precedes_compilation(monkeypatch, mutation):
    model, element = _q4_model()
    material = model.get_material('steel')
    function = staged._NATIVE_POLICY
    calls = []
    if mutation == 'binding':
        monkeypatch.setattr(type(element), '_elastic_increment_admitted', lambda *args: calls.append(args))
    elif mutation == 'code':
        monkeypatch.setattr(function, '__code__', function.__code__.replace(co_filename='changed-native-policy'))
    else:
        monkeypatch.setitem(function.__globals__, '_capture_q4_fast_input_snapshot', lambda *args: calls.append(args))
    if mutation == 'global':
        # The existing native guard observes this helper before policy dispatch.
        with pytest.raises(ValueError, match='qualified Q4 formulation runtime authority changed: _capture_q4_fast_input_snapshot'):
            staged.compile_q4_stages(element, model.mesh, material)
    else:
        with pytest.raises(PreparedStaticCapabilityError, match='native arithmetic policy authority'):
            staged.compile_q4_stages(element, model.mesh, material)
    assert calls == []
