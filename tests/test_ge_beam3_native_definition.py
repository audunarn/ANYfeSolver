"""Native construction/transaction parity; not independent qualification."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from anyfesolver import _ge_beam3_native_definition as codec
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver._ge_beam3_native_generalized_loading import assemble_distributed_trial
from test_ge_beam3_native_generalized import problem as generalized_problem, pattern
from test_ge_beam3_native_fibre_static_element import problem as fibre_problem, store_for, sample


def problem(family, curved=True, plastic=True):
    return (generalized_problem if family == 'generalized' else fibre_problem)(curved, plastic)


def inertia():
    # Physical section inertia, unrelated to numerical nodal trace coordinates.
    return np.diag([2., 2., 2., .07, .09, .11])


def clone_model(source, element):
    model = FEModel('native-definition-roundtrip')
    for key, node in source.mesh.nodes.items():
        model.add_node(key, *node.coords())
    model.add_element(element.element_id, element)
    model.materials[element.material_name] = element.section
    for bc in source.boundary_conditions:
        model.add_boundary_condition(deepcopy(bc))
    return model


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
@pytest.mark.parametrize('curved', (False, True))
@pytest.mark.parametrize('order', (4, 8))
def test_reconstructible_native_definition(family, curved, order, tmp_path):
    model, original = problem(family, curved)
    element = type(original)(original.element_id, tuple(original.node_ids),
                             original.operator.reference, original.section, order=order)
    definition = codec.NativeBeamDefinition.capture(element, inertia())
    recovered = codec.NativeBeamDefinition.from_bytes(definition.raw, expected_sha256=definition.sha256)
    fresh, physical_inertia = recovered.instantiate()
    assert type(fresh) is type(element) and fresh is not element
    assert fresh._mesh is None and fresh._mapping is None and fresh._validator is None
    assert fresh.operator is not element.operator and fresh.section is not element.section
    assert fresh.identity == element.identity and fresh.operator.identity == element.operator.identity
    assert fresh.section.identity == element.section.identity
    assert physical_inertia.tobytes() == inertia().tobytes() and not physical_inertia.flags.writeable
    assert codec.NativeBeamDefinition.capture(fresh, physical_inertia).raw == definition.raw
    assert np.array_equal(fresh.operator.reference.nodal_triads, element.operator.reference.nodal_triads)
    assert np.array_equal(fresh.operator.reference.coordinates, element.operator.reference.coordinates)
    assert fresh.production_qualified is False
    # Definition decoding does not pretend to implement the dynamic solver route.
    with pytest.raises(ValueError, match='not qualified'):
        fresh.compute_mass_matrix(None, None)
    (tmp_path/'definition.json').write_bytes(definition.raw)


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
@pytest.mark.parametrize('curved', (False, True))
def test_actual_native_transaction_after_reconstruction(family, curved, tmp_path):
    model, original = problem(family, curved, True)
    definition = codec.NativeBeamDefinition.capture(original, inertia())
    fresh, _ = codec.NativeBeamDefinition.from_bytes(definition.raw, expected_sha256=definition.sha256).instantiate()
    clone = clone_model(model, fresh)
    results = []
    for owner, element in ((model, original), (clone, fresh)):
        store = store_for(owner, element)
        virgin = canonical(store.materialize())
        if family == 'generalized':
            force, tangent, payload, external = assemble_distributed_trial(owner, sample(), store, pattern())
        else:
            force, tangent, payload = _assemble_nonlinear_system(owner, sample(), store, 1)
            external = np.zeros(18)
        assert canonical(store.materialize()) == virgin
        candidate = payload[1]
        token = store.active_trial_token()
        store.commit(token, accepted_full_displacement=sample(), accepted_full_coordinates=candidate['positions'])
        accepted = canonical(store.materialize())
        assert store.generation == 1
        if family == 'generalized':
            assert all(any(sum(history.plastic[i]) != 0 for history in candidate['response'].history.stations) for i in range(6))
            assemble_distributed_trial(owner, .4*sample(), store, pattern())
        else:
            assert any(row[2] > 0 for station in candidate['response'].history.stations for row in station.rows)
            _assemble_nonlinear_system(owner, .4*sample(), store, 1)
        store.discard_trial(store.active_trial_token())
        assert canonical(store.materialize()) == accepted
        element._validate(owner.mesh, store[1], sample())
        result = canonical(dict(state=store[1], force=force, tangent=tangent.toarray(),
                                external=external, generation=store.generation))
        results.append(result)
    assert results[0] == results[1]
    assert codec.NativeBeamDefinition.capture(original, inertia()).raw == definition.raw
    assert codec.NativeBeamDefinition.capture(fresh, inertia()).raw == definition.raw
    (tmp_path/'actual-native-transaction.json').write_bytes(results[0])


@pytest.mark.parametrize('mutation', ('schema', 'family', 'formulation', 'element_id', 'node_boolean',
    'node_duplicate', 'coordinates', 'triads', 'evaluation', 'tolerance', 'quadrature', 'quadrature_boolean',
    'elastic', 'metric', 'yield', 'section_policy', 'inertia', 'inertia_boolean', 'element_hash',
    'operator_hash', 'section_hash', 'static_policy', 'dynamic_policy', 'qualification', 'history'))
def test_rebound_mutation_rejected(mutation):
    _, element = generalized_problem()
    definition = codec.NativeBeamDefinition.capture(element, inertia())
    row = json.loads(definition.raw)
    if mutation == 'schema': row['schema'] = 'LEGACY_B3'
    elif mutation == 'family': row['family'] = 'ELASTIC_FIBRE_SURROGATE'
    elif mutation == 'formulation': row['formulation_id'] = 'GE_BEAM3_DC_MIXED_K1_MACRO_V2'
    elif mutation == 'element_id': row['element_id'] = True
    elif mutation == 'node_boolean': row['node_ids'][0] = True
    elif mutation == 'node_duplicate': row['node_ids'][1] = row['node_ids'][0]
    elif mutation == 'coordinates': row['reference']['coordinates'][1][0] += .001
    elif mutation == 'triads': row['reference']['nodal_triads'][1][0][0] += .1
    elif mutation == 'evaluation': row['reference']['evaluation_id'] = 'UNKNOWN'
    elif mutation == 'tolerance': row['reference']['frame_tolerance'] = .1
    elif mutation == 'quadrature': row['quadrature'] = 5
    elif mutation == 'quadrature_boolean': row['quadrature'] = True
    elif mutation == 'elastic': row['section']['elastic'][0][0] += .1
    elif mutation == 'metric': row['section']['metric'][0][0] += .1
    elif mutation == 'yield': row['section']['yield_force'] *= 2.
    elif mutation == 'section_policy': row['section']['policy'] = 'THREE_DIMENSIONAL_J2'
    elif mutation == 'inertia': row['section_inertia'][0][0] = -1.
    elif mutation == 'inertia_boolean': row['section_inertia'][0][0] = True
    elif mutation == 'element_hash': row['element_identity'] = '0'*64
    elif mutation == 'operator_hash': row['operator_identity'] = '0'*64
    elif mutation == 'section_hash': row['section_identity'] = '0'*64
    elif mutation == 'static_policy': row['static_policy'] = 'CONDENSE_ALL_INERTIA'
    elif mutation == 'dynamic_policy': row['dynamic_policy'] = 'STATIC_GUYAN_MASS'
    elif mutation == 'qualification': row['production_qualified'] = True
    else: row['committed_state'] = {}
    raw = canonical(row)
    # The caller's SHA is refreshed, so structural/derived checks must reject it.
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        codec.NativeBeamDefinition.from_bytes(raw, expected_sha256=sha256(raw).hexdigest())


@pytest.mark.parametrize('mutation', ('curve', 'area', 'factor', 'duplicate_fibre', 'family'))
def test_rebound_physical_fibre_mutation_rejected(mutation):
    _, element = fibre_problem()
    row = json.loads(codec.NativeBeamDefinition.capture(element, inertia()).raw)
    if mutation == 'curve': row['section']['fibres'][0]['curve']['flow_stress'][0] *= 1.1
    elif mutation == 'area': row['section']['fibres'][0]['area'] *= 1.1
    elif mutation == 'factor': row['section']['background_factor'][0][0] += .1
    elif mutation == 'duplicate_fibre': row['section']['fibres'].append(row['section']['fibres'][0])
    else: row['family'] = 'RESULTANT_ELLIPSOID'
    raw = canonical(row)
    with pytest.raises(ValueError):
        codec.NativeBeamDefinition.from_bytes(raw, expected_sha256=sha256(raw).hexdigest())


def test_external_hash_checked_before_construction(monkeypatch):
    def forbidden(*args):
        raise AssertionError('construction before external authority')
    monkeypatch.setattr(codec, '_build', forbidden)
    with pytest.raises(codec.DefinitionError, match='external'):
        codec.NativeBeamDefinition.from_bytes(b'{}\n', expected_sha256='0'*64)


def test_oversized_definition_rejected_before_hash(monkeypatch):
    def forbidden(*args):
        raise AssertionError('oversized input was hashed or constructed')
    monkeypatch.setattr(codec, 'sha256', forbidden)
    with pytest.raises(codec.DefinitionError):
        codec.NativeBeamDefinition.from_bytes(b'x'*(codec.MAX_BYTES+1), expected_sha256='0'*64)


@pytest.mark.parametrize('raw', (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":1e999}\n',
                                b'{}', b'{} \n', b'[]\n', b'\xff', b'['*10000))
def test_bad_json_rejected_before_construction(raw, monkeypatch):
    def forbidden(*args):
        raise AssertionError('element constructed for invalid JSON')
    monkeypatch.setattr(codec, '_build', forbidden)
    with pytest.raises(codec.DefinitionError):
        codec.NativeBeamDefinition.from_bytes(raw, expected_sha256=sha256(raw).hexdigest())
