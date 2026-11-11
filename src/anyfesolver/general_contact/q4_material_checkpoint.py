"""Owned, semantically validated Q4 material packets for transient development.

This is a material-state packet, not a static analysis checkpoint or a dynamic
profile admission. The caller owns the accepted time, velocity, acceleration
and contact transaction and must bind this packet to those quantities.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math

import numpy as np

from anyfesolver.current_state_tangent import (
    require_exact_qualified_component_lifecycle_api as _LIFECYCLE_GUARD,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.nonlinear_restart import (
    analysis_configuration_fingerprint,
    canonical_checkpoint_json_bytes,
    model_configuration_fingerprint,
)


_SCHEMA = "ANYSOLVER_Q4_MATERIAL_CHECKPOINT_V1"
_INTEGRITY = "SHA256_CANONICAL_JSON_EXCLUDING_SELF_V1"
_KEYS = {"schema", "version", "integrity_id", "model_fingerprint",
         "num_layers", "total_dofs", "displacements", "element_states", "packet_sha256"}
_CORE_KEYS = {"plastic_strain", "alpha", "layer_strain", "layer_strain_material",
              "kinematic_layer_strain", "layer_stress"}
_STATE_KEYS = _CORE_KEYS | {"qualified_q4_algorithmic_origin",
    "qualified_q4_committed_binding", "state_integrity_sha256"}
_INITIAL_KEYS = {"initial_membrane_stress", "initial_bending_stress",
    "initial_membrane_prestrain", "initial_curvature_prestrain", "initial_field_provenance"}


def _guard(model, context):
    _LIFECYCLE_GUARD(model, context="Q4 material checkpoint "+context)


def _canonical(model, value):
    encoded = canonical_checkpoint_json_bytes(
        value, _exact_guard=_LIFECYCLE_GUARD, _guard_model=model)
    _guard(model, "canonical encoding")
    return json.loads(encoded)


def _fingerprint(model, packet):
    return analysis_configuration_fingerprint(
        packet, _exact_guard=_LIFECYCLE_GUARD, _guard_model=model)


def _same_model(model, fingerprint):
    current = model_configuration_fingerprint(model, _exact_guard=_LIFECYCLE_GUARD)
    _guard(model, "final model observation")
    if current != fingerprint:
        raise ValueError("Q4 material checkpoint model changed during state observation")


def _plain_json(value, active=None, depth=0):
    if depth > 64:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (list, dict):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        if type(value) is list:
            valid = all(_plain_json(item, active, depth+1) for item in value)
        else:
            valid = all(type(key) is str and _plain_json(item, active, depth+1)
                        for key, item in value.items())
        active.remove(identity)
        return valid
    return False


def _context(model, full_displacement, num_layers):
    _guard(model, "preflight")
    if type(num_layers) is not int or num_layers not in {3, 5, 7, 9, 11}:
        raise ValueError("Q4 material checkpoint num_layers must be an admitted exact integer")
    elements = tuple(model.mesh.elements.items())
    _guard(model, "element mapping observation")
    if (not elements or any(type(key) is not int or type(element) is not QualifiedE4PLShellElement
                            for key, element in elements)):
        raise ValueError("Q4 material checkpoint requires only exact qualified Q4 elements")
    if getattr(model.mesh, "element_activity", None) is not None:
        raise ValueError("Q4 material checkpoint does not admit activity or deletion state")
    size = model.mesh.dof_manager.total_dofs
    if type(size) is not int or size < 1:
        raise ValueError("Q4 material checkpoint requires a valid full DOF layout")
    try:
        full = np.asarray(full_displacement, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        _guard(model, "failed full displacement observation")
        raise ValueError("Q4 material checkpoint requires finite complete displacements") from exc
    _guard(model, "full displacement observation")
    if full.shape != (size,) or not np.isfinite(full).all():
        raise ValueError("Q4 material checkpoint requires finite complete displacements")
    full = full.copy()
    # The canonical checkpoint encoder maps both floating zeros to +0.
    # Normalize before the Q4 binary-coordinate seal, not after its hash.
    full[full == 0.] = 0.
    owned_elements = []
    for element_id, element in sorted(elements):
        coordinates = np.asarray(element.get_node_coordinates(model.mesh), dtype=np.float64)
        dofs = np.asarray(element.get_dof_mapping(model.mesh))
        _guard(model, "element coordinates and DOFs")
        if coordinates.shape != (4, 3) or not np.isfinite(coordinates).all():
            raise ValueError("Q4 material checkpoint requires finite reference coordinates")
        if (dofs.shape != (24,) or not np.issubdtype(dofs.dtype, np.integer)
                or np.any(dofs < 0) or np.any(dofs >= size)):
            raise ValueError("Q4 material checkpoint element DOFs are incompatible")
        material = model.get_material(element.material_name)
        _guard(model, "material observation")
        if element.shell_section is not None:
            raise ValueError("Q4 material checkpoint does not admit generalized section states")
        owned_elements.append((element_id, element, material, dofs.copy()))
    fingerprint = model_configuration_fingerprint(model, _exact_guard=_LIFECYCLE_GUARD)
    _guard(model, "model fingerprint")
    return full, size, fingerprint, tuple(owned_elements)


def _state(model, state, points, *, sealed):
    if not isinstance(state, Mapping):
        raise ValueError("Q4 material checkpoint state must be a mapping")
    # Guarded canonical conversion also makes arrays and nested parent history
    # owned before inspecting them or invoking the public element boundary.
    value = _canonical(model, state)
    if type(value) is not dict:
        raise ValueError("Q4 material checkpoint state must be an object")
    if set(value) & _INITIAL_KEYS:
        raise ValueError("Q4 material checkpoint does not admit initial fields")
    if not {"plastic_strain", "alpha"} <= set(value) or not set(value) <= _STATE_KEYS:
        raise ValueError("Q4 material checkpoint has unsupported or incomplete material state")
    if sealed and not {"qualified_q4_committed_binding", "state_integrity_sha256"} <= set(value):
        raise ValueError("Q4 material checkpoint requires a sealed accepted state")

    def numeric_array(array, shape):
        def numbers(item):
            if type(item) is list:
                return all(numbers(child) for child in item)
            if type(item) not in (int, float):
                return False
            try:
                return math.isfinite(float(item))
            except OverflowError:
                return False
        try:
            valid = type(array) is list and numbers(array) and np.asarray(array).shape == shape
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Q4 material checkpoint material arrays are invalid")

    numeric_array(value["plastic_strain"], (points, 3))
    numeric_array(value["alpha"], (points,))
    if any(alpha < 0 for alpha in value["alpha"]):
        raise ValueError("Q4 material checkpoint alpha must be nonnegative")
    for name in _CORE_KEYS - {"plastic_strain", "alpha"}:
        if name in value:
            numeric_array(value[name], (points, 3))
    origin = value.get("qualified_q4_algorithmic_origin")
    if origin is not None:
        if (type(origin) is not dict or set(origin) != {"schema_id", "kind", "num_layers",
                                                     "parent_plastic_strain", "parent_alpha"}
                or type(origin["schema_id"]) is not str or type(origin["kind"]) is not str
                or type(origin["num_layers"]) is not int):
            raise ValueError("Q4 material checkpoint algorithmic origin is malformed")
        numeric_array(origin["parent_plastic_strain"], (points, 3))
        numeric_array(origin["parent_alpha"], (points,))
        if any(alpha < 0 for alpha in origin["parent_alpha"]):
            raise ValueError("Q4 material checkpoint parent alpha must be nonnegative")
    return value


def prepare_q4_material_checkpoint(model, full_displacement, committed_states, *, num_layers=3):
    """Seal complete accepted Q4 history into an owned canonical JSON packet.

    Public Q4 sealing replays the retained discrete-update parent; a state
    without that parent, or recovery from another pose, cannot be sealed.
    The input mapping and constitutive arrays are never committed or modified.
    """
    full, size, fingerprint, elements = _context(model, full_displacement, num_layers)
    if not isinstance(committed_states, Mapping):
        raise ValueError("Q4 material checkpoint requires a complete state mapping")
    items = tuple(committed_states.items())
    _guard(model, "committed state mapping observation")
    ids = [key for key, _ in items]
    expected = [key for key, *_ in elements]
    if any(type(key) is not int for key in ids) or len(set(ids)) != len(ids) or sorted(ids) != expected:
        raise ValueError("Q4 material checkpoint element IDs must be complete and exact")
    states = dict(items)
    records = []
    for element_id, element, material, dofs in elements:
        state = _state(model, states[element_id], len(element.gauss_points)*num_layers, sealed=False)
        has_binding = "qualified_q4_committed_binding" in state
        has_digest = "state_integrity_sha256" in state
        if has_binding != has_digest:
            raise ValueError("Q4 material checkpoint accepted seal is incomplete")
        if has_binding:
            # Existing identity is evidence too. Do not erase a stale or
            # tampered seal and bless it by producing a fresh outer hash.
            element.validate_committed_current_tangent_state(
                model.mesh, material, full[dofs], state, num_layers)
            _guard(model, "existing accepted seal validation")
        accepted = element.seal_committed_current_tangent_state(
            model.mesh, material, full[dofs], state, num_layers)
        _guard(model, "accepted state sealing")
        accepted = _state(model, accepted, len(element.gauss_points)*num_layers, sealed=True)
        element.validate_committed_current_tangent_state(
            model.mesh, material, full[dofs], accepted, num_layers)
        _guard(model, "accepted state validation")
        records.append({"element_id": element_id, "state": accepted})
    _same_model(model, fingerprint)
    packet = {"schema": _SCHEMA, "version": 1, "integrity_id": _INTEGRITY,
              "model_fingerprint": fingerprint, "num_layers": num_layers, "total_dofs": size,
              "displacements": full.tolist(), "element_states": records}
    packet["packet_sha256"] = _fingerprint(model, packet)
    return _canonical(model, packet)


def restore_q4_material_checkpoint(model, full_displacement, packet, *, num_layers=3):
    """Validate identity, integrity and constitutive semantics before restoration.

    This returns a fresh int-ID mapping. It does not re-seal or repair supplied
    history: recomputing the packet hash cannot authorize an invalid core or
    return-map origin. The caller must keep this accepted history through its
    resumed initial force-balance check.
    """
    full, size, fingerprint, elements = _context(model, full_displacement, num_layers)
    if type(packet) is not dict or set(packet) != _KEYS or not _plain_json(packet):
        raise ValueError("Q4 material checkpoint schema keys are incompatible")
    for name in ("schema", "integrity_id", "model_fingerprint", "packet_sha256"):
        if type(packet[name]) is not str:
            raise ValueError("Q4 material checkpoint identity must use exact string types")
    for name in ("version", "num_layers", "total_dofs"):
        if type(packet[name]) is not int:
            raise ValueError("Q4 material checkpoint identity must use exact integer types")
    if (packet["schema"] != _SCHEMA or packet["version"] != 1 or packet["integrity_id"] != _INTEGRITY
            or packet["num_layers"] != num_layers or packet["total_dofs"] != size
            or packet["model_fingerprint"] != fingerprint):
        raise ValueError("Q4 material checkpoint model, layers or identity is incompatible")
    if (type(packet["displacements"]) is not list
            or any(type(value) not in (int, float) for value in packet["displacements"])
            or type(packet["element_states"]) is not list):
        raise ValueError("Q4 material checkpoint coordinates or records are malformed")
    owned = _canonical(model, packet)
    signature = owned.pop("packet_sha256")
    if len(signature) != 64 or signature != _fingerprint(model, owned):
        raise ValueError("Q4 material checkpoint integrity mismatch")
    if not np.array_equal(np.asarray(owned["displacements"], dtype=np.float64), full):
        raise ValueError("Q4 material checkpoint displacement is incompatible")
    records = owned["element_states"]
    if any(type(record) is not dict or set(record) != {"element_id", "state"}
           or type(record["element_id"]) is not int for record in records):
        raise ValueError("Q4 material checkpoint element records are malformed")
    if [record["element_id"] for record in records] != [element_id for element_id, *_ in elements]:
        raise ValueError("Q4 material checkpoint element IDs must be complete, ordered and unique")
    restored = {}
    for record, (element_id, element, material, dofs) in zip(records, elements):
        state = _state(model, record["state"], len(element.gauss_points)*num_layers, sealed=True)
        element.validate_committed_current_tangent_state(
            model.mesh, material, full[dofs], state, num_layers)
        _guard(model, "restored constitutive validation")
        restored[element_id] = state
    _guard(model, "restoration complete")
    _same_model(model, fingerprint)
    return restored
