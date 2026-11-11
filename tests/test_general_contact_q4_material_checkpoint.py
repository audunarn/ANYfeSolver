"""Real yielded Q4 history and adversarial material-packet round trips."""

import copy
import json

import numpy as np
import pytest

from anyfesolver import FEModel, create_shell_element
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_state import canonical_sha256
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.general_contact.q4_material_checkpoint import (
    prepare_q4_material_checkpoint, restore_q4_material_checkpoint,
)
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.nonlinear_restart import (
    analysis_configuration_fingerprint, canonical_checkpoint_json_bytes,
)


def _yielded(*, cells=1, layers=3):
    model = FEModel("q4-material-packet")
    material = model.add_material("plastic", 210.e9, .3, density=7850.,
        yield_stress=100.e6, hardening_curve=DNVC208MaterialCurve(
            sigma_prop=100.e6, sigma_yield=105.e6, sigma_yield_2=110.e6,
            eps_p_y1=.005, eps_p_y2=.010, K=400.e6, n=.20))
    for cell in range(cells):
        nodes = tuple(range(4*cell+1, 4*cell+5))
        for node, (x, y) in zip(nodes, ((0., 0.), (1., 0.), (1., 1.), (0., 1.))):
            model.add_node(node, x+2*cell, y, 0.)
        model.add_element(cell+1, create_shell_element(cell+1, nodes, "plastic", thickness=.02))
    full = np.zeros(model.mesh.dof_manager.total_dofs)
    states = {}
    for element_id, element in model.mesh.elements.items():
        reference = element.get_node_coordinates(model.mesh)
        dofs = np.asarray(element.get_dof_mapping(model.mesh))
        parent_pose = np.zeros(24)
        parent_pose[0::6] = .003*(reference[:, 0]-reference[0, 0])
        parent_pose[1::6] = -.003*reference[:, 1]
        _, _, parent = element.compute_nonlinear_response(
            model.mesh, material, parent_pose, None, layers, True)
        pose = parent_pose*1.5
        _, _, state = element.compute_nonlinear_response(
            model.mesh, material, pose, parent, layers, True)
        assert np.min(state["alpha"]) > 0
        assert np.min(state["qualified_q4_algorithmic_origin"]["parent_alpha"]) > 0
        full[dofs] = pose
        states[element_id] = state
    return model, full, states


def _rehash(packet):
    packet["packet_sha256"] = analysis_configuration_fingerprint(
        {key: value for key, value in packet.items() if key != "packet_sha256"})
    return packet


def _resign_state(state):
    # Deliberately repair every unkeyed hash so the test reaches the semantic
    # replay check. Cryptographic integrity alone cannot establish mechanics.
    binding = state["qualified_q4_committed_binding"]
    binding["state_payload_sha256"] = canonical_sha256({key: value for key, value in state.items()
        if key not in {"qualified_q4_committed_binding", "state_integrity_sha256"}})
    binding["accepted_algorithmic_update_sha256"] = canonical_sha256({
        "layout": "Q4_ACCEPTED_DISCRETE_CONSTITUTIVE_UPDATE_V1",
        "algorithmic_origin": state["qualified_q4_algorithmic_origin"],
        "accepted_core": {key: state[key] for key in ("plastic_strain", "alpha", "layer_strain")}})
    digest = canonical_sha256({key: value for key, value in binding.items()
                               if key != "state_integrity_sha256"})
    binding["state_integrity_sha256"] = digest
    state["state_integrity_sha256"] = digest


@pytest.mark.parametrize("cells", [1, 2])
def test_yielded_state_serialization_preserves_nonvirgin_parent_and_owned_history(cells):
    model, full, states = _yielded(cells=cells)
    original = canonical_checkpoint_json_bytes({str(key): value for key, value in states.items()})
    packet = prepare_q4_material_checkpoint(model, full, states)
    encoded = canonical_checkpoint_json_bytes(packet)
    assert json.dumps(packet, allow_nan=False)
    restored = restore_q4_material_checkpoint(model, full, json.loads(encoded))
    assert list(restored) == list(states)
    for element_id, expected in states.items():
        for name in ("plastic_strain", "alpha", "layer_strain", "layer_stress"):
            np.testing.assert_array_equal(restored[element_id][name], expected[name])
        origin = restored[element_id]["qualified_q4_algorithmic_origin"]
        np.testing.assert_array_equal(origin["parent_alpha"], expected["qualified_q4_algorithmic_origin"]["parent_alpha"])
        assert origin["parent_alpha"] != restored[element_id]["alpha"]
    assert canonical_checkpoint_json_bytes({str(key): value for key, value in states.items()}) == original
    again = prepare_q4_material_checkpoint(model, full, restored)
    assert canonical_checkpoint_json_bytes(again) == encoded
    restored[1]["plastic_strain"][0][0] += .125
    restored[1]["qualified_q4_algorithmic_origin"]["parent_alpha"][0] += .125
    assert canonical_checkpoint_json_bytes(packet) == encoded
    assert canonical_checkpoint_json_bytes({str(key): value for key, value in states.items()}) == original


def test_restored_history_produces_same_next_constitutive_update():
    model, full, states = _yielded()
    restored = restore_q4_material_checkpoint(model, full, prepare_q4_material_checkpoint(model, full, states))
    element = model.mesh.elements[1]
    material = model.get_material("plastic")
    for multiplier in (.8, 1.1):
        pose = full*multiplier
        original = element.compute_nonlinear_response(model.mesh, material, pose, states[1], 3, True)
        resumed = element.compute_nonlinear_response(model.mesh, material, pose, restored[1], 3, True)
        np.testing.assert_array_equal(original[0], resumed[0])
        np.testing.assert_array_equal(original[1], resumed[1])
        assert canonical_checkpoint_json_bytes(original[2]) == canonical_checkpoint_json_bytes(resumed[2])


def test_actual_assembler_compact_candidate_keeps_accepted_algorithmic_parent():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    model, full, parents = _yielded()
    next_pose = full*1.1
    _, _, candidate = _assemble_nonlinear_system(model, next_pose, parents, 3, tangent=True)
    packet = prepare_q4_material_checkpoint(model, next_pose, candidate)
    restored = restore_q4_material_checkpoint(model, next_pose, packet)
    assert canonical_checkpoint_json_bytes(restored[1]["qualified_q4_algorithmic_origin"]) == (
        canonical_checkpoint_json_bytes(candidate[1]["qualified_q4_algorithmic_origin"]))
    np.testing.assert_array_equal(restored[1]["qualified_q4_algorithmic_origin"]["parent_alpha"], parents[1]["alpha"])


def test_signed_zero_coordinates_remain_canonical_and_round_trip():
    model, full, states = _yielded()
    full[full == 0.] = -0.
    packet = prepare_q4_material_checkpoint(model, full, states)
    restored = restore_q4_material_checkpoint(model, full, json.loads(canonical_checkpoint_json_bytes(packet)))
    assert restored[1]["alpha"] == packet["element_states"][0]["state"]["alpha"]


@pytest.mark.parametrize("wrong", ["pose", "material", "geometry", "layers", "layout"])
def test_packet_rejects_changed_configuration(wrong):
    model, full, states = _yielded()
    packet = prepare_q4_material_checkpoint(model, full, states)
    options = {}
    if wrong == "pose":
        full[6] += .001
    elif wrong == "material":
        model.materials["plastic"].elastic_modulus *= .9
    elif wrong == "geometry":
        model.mesh.nodes[2].x += .01
    elif wrong == "layers":
        options["num_layers"] = 5
    else:
        full = full[:-1]
    with pytest.raises(ValueError):
        restore_q4_material_checkpoint(model, full, packet, **options)


@pytest.mark.parametrize("field", ["alpha", "plastic_strain", "layer_stress", "parent_alpha", "parent_plastic_strain"])
def test_recomputed_outer_and_inner_hashes_do_not_authorize_invalid_constitutive_history(field):
    model, full, states = _yielded()
    packet = prepare_q4_material_checkpoint(model, full, states)
    state = packet["element_states"][0]["state"]
    target = state["qualified_q4_algorithmic_origin"] if field.startswith("parent_") else state
    if field in ("alpha", "parent_alpha"):
        target[field][0] += 1.e-5
    else:
        target[field][0][0] += 1.e-5 if field != "layer_stress" else 1.
    _resign_state(state)
    _rehash(packet)
    with pytest.raises(ValueError, match="reproduce|replay|layer_stress"):
        restore_q4_material_checkpoint(model, full, packet)


@pytest.mark.parametrize("wrong", ["missing", "extra", "duplicate", "order", "float_id", "bool_id"])
def test_complete_ordered_exact_element_ids_are_required(wrong):
    model, full, states = _yielded(cells=2)
    packet = prepare_q4_material_checkpoint(model, full, states)
    records = packet["element_states"]
    if wrong == "missing":
        records.pop()
    elif wrong == "extra":
        records.append({"element_id": 3, "state": copy.deepcopy(records[0]["state"])})
    elif wrong == "duplicate":
        records[1]["element_id"] = 1
    elif wrong == "order":
        records.reverse()
    else:
        records[0]["element_id"] = 1. if wrong == "float_id" else True
    _rehash(packet)
    with pytest.raises(ValueError, match="element"):
        restore_q4_material_checkpoint(model, full, packet)


@pytest.mark.parametrize("wrong", ["version", "schema", "hash", "extra", "missing", "nan", "coordinate_bool", "record_keys", "numpy_id", "cycle"])
def test_packet_schema_and_integrity_are_strict(wrong):
    model, full, states = _yielded()
    packet = prepare_q4_material_checkpoint(model, full, states)
    if wrong == "version":
        packet["version"] = True
    elif wrong == "schema":
        packet["schema"] += "_other"
    elif wrong == "hash":
        packet["packet_sha256"] = "0"*64
    elif wrong == "extra":
        packet["extra"] = 0
    elif wrong == "missing":
        del packet["num_layers"]
    elif wrong == "nan":
        packet["element_states"][0]["state"]["alpha"][0] = float("nan")
    elif wrong == "coordinate_bool":
        packet["displacements"][0] = False
    elif wrong == "numpy_id":
        packet["element_states"][0]["element_id"] = np.int64(1)
    elif wrong == "cycle":
        packet["element_states"].append(packet)
    else:
        packet["element_states"][0]["extra"] = None
        _rehash(packet)
    with pytest.raises(ValueError):
        restore_q4_material_checkpoint(model, full, packet)


@pytest.mark.parametrize("wrong", ["initial_field", "failed", "deleted", "unknown", "missing_origin", "negative_alpha", "bool_alpha", "parent_layers", "shape", "nonfinite"])
def test_prepare_rejects_unsupported_or_invalid_accepted_states(wrong):
    model, full, states = _yielded()
    state = states[1]
    if wrong == "initial_field":
        state["initial_membrane_stress"] = np.zeros((4, 3))
    elif wrong in ("failed", "deleted"):
        state["qualified_q4_activity_disposition"] = {"status": wrong.upper()}
    elif wrong == "unknown":
        state["invented_history"] = 0
    elif wrong == "missing_origin":
        del state["qualified_q4_algorithmic_origin"]
    elif wrong == "negative_alpha":
        state["alpha"][0] = -1.
    elif wrong == "bool_alpha":
        state["alpha"] = [False]*12
    elif wrong == "shape":
        state["plastic_strain"] = state["plastic_strain"][:-1]
    elif wrong == "nonfinite":
        state["alpha"][0] = float("inf")
    else:
        state["qualified_q4_algorithmic_origin"]["num_layers"] = 3.
    with pytest.raises(ValueError):
        prepare_q4_material_checkpoint(model, full, states)


@pytest.mark.parametrize("layers", [True, 1, 2, 4, 3.])
def test_invalid_layer_identity_is_rejected(layers):
    model, full, states = _yielded()
    with pytest.raises(ValueError):
        prepare_q4_material_checkpoint(model, full, states, num_layers=layers)


def test_prepare_requires_complete_int_ids_and_rejects_existing_tampered_seal():
    model, full, states = _yielded()
    for invalid in ({}, {True: states[1]}, {"1": states[1]}, {1: states[1], 2: states[1]}):
        with pytest.raises(ValueError):
            prepare_q4_material_checkpoint(model, full, invalid)
    packet = prepare_q4_material_checkpoint(model, full, states)
    sealed = packet["element_states"][0]["state"]
    sealed["qualified_q4_committed_binding"]["num_layers"] = 5
    with pytest.raises(ValueError, match="binding"):
        prepare_q4_material_checkpoint(model, full, {1: sealed})


def test_nonfinite_coordinates_and_activity_state_are_rejected():
    model, full, states = _yielded()
    for invalid in (np.full(24, np.inf), np.zeros((4, 6)), np.zeros(23)):
        with pytest.raises(ValueError):
            prepare_q4_material_checkpoint(model, invalid, states)
    from anyfesolver.activity import ElementActivity
    model.set_element_activity(ElementActivity([1]))
    with pytest.raises(ValueError, match="activity"):
        prepare_q4_material_checkpoint(model, full, states)


@pytest.mark.parametrize("boundary", ["entry", "state_observation"])
def test_qualified_api_mutation_is_rejected(monkeypatch, boundary):
    model, full, states = _yielded()
    with monkeypatch.context() as patch:
        def change_api():
            patch.setattr(QualifiedE4PLShellElement, "seal_committed_current_tangent_state", lambda *_: None)
        if boundary == "entry":
            change_api()
        else:
            class EffectfulStates(dict):
                def items(self):
                    change_api()
                    return super().items()
            states = EffectfulStates(states)
        with pytest.raises(ElementCapabilityError, match="exact qualified"):
            prepare_q4_material_checkpoint(model, full, states)


def test_only_exact_q4_and_finite_reference_geometry_are_admitted():
    model, full, states = _yielded()
    model.mesh.nodes[2].x = float("nan")
    with pytest.raises(ValueError):
        prepare_q4_material_checkpoint(model, full, states)
    from anyfesolver.elements import ShellElement
    model, full, states = _yielded()
    model.add_element(1, ShellElement(1, (1, 2, 3, 4), "plastic", thickness=.02))
    with pytest.raises(ValueError, match="exact qualified Q4"):
        prepare_q4_material_checkpoint(model, full, states)
