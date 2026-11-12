"""Exact reference identity equivalence and all-live-input mutation coverage."""
from dataclasses import replace
import ast, inspect
import numpy as np
import pytest
from anyfesolver import _ge_beam3_reference_identity as identity
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from test_ge_beam3_centered_reference import fixture
from test_ge_beam3_native_generalized_modal import make


def captured():
    x, q = fixture(spatial=True); r = Reference(x, q)
    return r, identity.CapturedReferenceIdentity(r)


def test_original_hash_and_live_snapshot_without_reconstruction(monkeypatch):
    calls = []; original = identity.source.CurvedBeam3ReferenceGeometry.fingerprint
    def counted(self):
        calls.append(1); return original(self)
    monkeypatch.setattr(identity.source.CurvedBeam3ReferenceGeometry, 'fingerprint', counted)
    ref, bound = captured(); expected = original(ref)
    for _ in range(100): assert bound.require(ref) == expected
    assert len(calls) == 1
    # Same contents in a new array are not a false mutation; signed zero has
    # the original canonical equivalence and may take the slow fallback.
    ref._coordinates = ref._coordinates.copy()
    assert bound.require(ref) == expected
    ref._coordinates[0, 1] = -0.
    assert bound.require(ref) == original(ref) == expected


@pytest.mark.parametrize('field', ('_coordinates', '_nodal_triads', '_coefficient_high', '_coefficient_low'))
@pytest.mark.parametrize('kind', ('value', 'inplace', 'shape', 'nonfinite'))
def test_live_array_mutations(field, kind):
    ref, bound = captured(); array = getattr(ref, field)
    if kind == 'inplace':
        array.setflags(write=True); array.flat[0] += .01
    else:
        array = array.copy()
        if kind == 'value': array.flat[0] += .01
        elif kind == 'shape': array = array[:1]
        else: array.flat[0] = float('nan')
        setattr(ref, field, array)
    with pytest.raises(ValueError): bound.require(ref)


@pytest.mark.parametrize('field', ('affine_constant', 'affine_linear', 'minimizer_xi',
    'minimum_jacobian_squared', 'characteristic_length', 'minimum_admissible_jacobian'))
def test_all_regularity_fields(field):
    ref, bound = captured(); value = getattr(ref._regularity, field)
    changed = (value[0]+.01, *value[1:]) if isinstance(value, tuple) else value+.01
    ref._regularity = replace(ref._regularity, **{field:changed})
    with pytest.raises(ValueError, match='reference changed'): bound.require(ref)


@pytest.mark.parametrize('field', ('_frame_tolerance', '_rotation_tolerance', '_regularity_relative_tolerance', '_half_frame_data'))
def test_tolerances_and_branch_data(field):
    ref, bound = captured()
    if field == '_half_frame_data':
        a,b = ref._half_frame_data; ref._half_frame_data = ((a[0], a[1]+.01), b)
    else: setattr(ref, field, getattr(ref, field)*2.)
    with pytest.raises(ValueError, match='reference changed'): bound.require(ref)


@pytest.mark.parametrize('field', ('GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID', 'GE_BEAM3_CURVED_REFERENCE_REVERSAL_ID',
    'GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID', 'GE_BEAM3_CURVED_REFERENCE_REGULARITY_ID', 'SCHEMA', 'EVALUATION_ID', '_NODE_PARAMETERS'))
def test_canonical_global_inputs(field, monkeypatch):
    ref, bound = captured()
    module = identity.centered if field in ('SCHEMA','EVALUATION_ID') else identity.source
    original = getattr(module, field)
    changed = original+'-changed' if isinstance(original, str) else original+.01
    monkeypatch.setattr(module, field, changed)
    # Centered canonical_data overwrites the base schema ID. Its original
    # fingerprint therefore stays equivalent for that ONE source-only change.
    if field == 'GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID':
        assert bound.require(ref) == ref.fingerprint()
    else:
        with pytest.raises(ValueError, match='reference changed'): bound.require(ref)


@pytest.mark.parametrize('field', ('_identity', '_snapshot', '_reference', '_methods'))
def test_capture_tampering_and_foreign_references(field):
    ref, bound = captured()
    with pytest.raises(AttributeError): setattr(bound, field, None)
    object.__setattr__(bound, field, None)
    with pytest.raises(ValueError, match='authority changed'): bound.require(ref)


def test_fingerprint_method_replacement_is_not_a_cache_hit(monkeypatch):
    ref, bound = captured()
    monkeypatch.setattr(identity.source.CurvedBeam3ReferenceGeometry, 'fingerprint', lambda self:'forged')
    with pytest.raises(ValueError, match='authority changed'): bound.require(ref)


@pytest.mark.parametrize('method', ('fingerprint', 'canonical_bytes'))
def test_derived_class_method_override_is_bound(method, monkeypatch):
    ref, bound = captured()
    monkeypatch.setattr(Reference, method, lambda self:'forged')
    with pytest.raises(ValueError, match='authority changed'): bound.require(ref)


def test_regularity_instance_serializer_override_is_checked():
    ref, bound = captured()
    object.__setattr__(ref._regularity, 'canonical_data', lambda:{'changed':'serializer'})
    with pytest.raises(ValueError, match='reference changed'): bound.require(ref)


def test_closed_world_canonical_attribute_map():
    # Fail this guard test if a future source schema consumes a new instance
    # field without extending the snapshot. Constants are separately enumerated.
    def fields(function):
        tree = ast.parse(__import__('textwrap').dedent(inspect.getsource(function)))
        return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == 'self'}
    assert fields(identity.source.CurvedBeam3ReferenceGeometry.canonical_data) == {
        '_coordinates','_nodal_triads','_half_frame_data','_regularity','_frame_tolerance',
        '_regularity_relative_tolerance','_rotation_tolerance'}
    assert fields(Reference.canonical_data) == {'_coefficient_high','_coefficient_low'}
    assert fields(identity.source.CurvedBeam3RegularityCertificate.canonical_data) == {
        'affine_constant','affine_linear','minimizer_xi','minimum_jacobian_squared',
        'characteristic_length','minimum_admissible_jacobian'}


def test_element_descriptor_identity_stays_original_and_mutation_fails():
    model, _, _ = make(True, 1, clamped=True)
    element = model.mesh.elements[1]; ref = element.operator.reference
    assert element.to_dict()['reference'] == ref.fingerprint()
    before = ref._half_frame_data; ref._half_frame_data = ((before[0][0], before[0][1]+.01), before[1])
    with pytest.raises(ValueError): element.to_dict()
    with pytest.raises(ValueError): element.operator.guard()
