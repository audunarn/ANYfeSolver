"""Typed result-quantity descriptors for solver/UI interchange.

The descriptors intentionally describe *available* arrays rather than
inventing zero-valued fields.  They are small, serializable contracts that a
postprocessor can use to build its tree without hard-coding each result class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import import_module
from numbers import Integral
from types import GetSetDescriptorType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ResultQuantity:
    """Metadata for one result field or history."""

    quantity_id: str
    label: str
    location: str
    components: Tuple[str, ...]
    unit: str
    basis: str = "global"
    frame_count: int = 1
    has_history: bool = False
    data_path: str = ""
    recovery: str = "native"
    availability: str = "available"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.quantity_id:
            raise ValueError("quantity_id must not be empty")
        if self.frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        object.__setattr__(self, "components", tuple(str(value) for value in self.components))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["components"] = list(self.components)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ResolvedResultQuantity:
    """One available descriptor paired with its authoritative data."""

    descriptor: ResultQuantity
    data: Any


class QuantityUnavailableError(LookupError):
    """Raised when a result genuinely carries no requested quantity."""


@dataclass(frozen=True)
class ReactionFrame:
    """Nodal and support reactions at one committed result frame."""

    frame_index: int
    abscissa: float
    abscissa_kind: str
    reactions: Mapping[int, Any]
    support_resultants: Mapping[str, Any]


_REGISTERED_QUANTITY_IDS = (
    "displacement",
    "velocity",
    "acceleration",
    "stress",
    "stress_history",
    "reaction",
    "reaction_history",
    "equivalent_plastic_strain",
    "equivalent_plastic_strain_history",
    "load_factor",
    "mode_shape",
    "frequency",
    "buckling_factor",
    "time",
    "contact_force",
    "impactor_position",
    "force_impulse",
    "moment_impulse",
    "load_impulse",
    "kinetic_energy",
    "strain_energy",
    "internal_work",
    "impactor_kinetic_energy",
    "displacement_envelope",
    "velocity_envelope",
    "acceleration_envelope",
)


def registered_result_quantity_ids() -> Tuple[str, ...]:
    """Stable canonical IDs that callers may resolve fail-closed."""

    return _REGISTERED_QUANTITY_IDS


def _array_frames(value: Any, *, default: int = 1) -> int:
    array = _finite_numeric_array(value)
    if array is None:
        return 0
    if array.ndim >= 2:
        return int(array.shape[0])
    return int(default)


def _valid_integer_identifier(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Integral):
        return True
    if isinstance(value, str):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return False
        return value == str(parsed)
    return False


def _valid_nodal_array(value: Any) -> bool:
    array = _finite_numeric_array(value)
    return bool(
        array is not None
        and array.ndim in (1, 2)
        and array.shape[-1] > 0
        and array.shape[-1] % 6 == 0
    )


def _stress_components(result: Any) -> Tuple[str, ...]:
    payload = _physical_stress_frame(
        getattr(result, "element_stresses", None)
    )
    return () if payload is None else tuple(sorted(payload[1]))


def _snapshot_displacements(result: Any) -> Tuple[Any, ...] | None:
    """Return a complete finite displacement history, or no history.

    Some result families use ``snapshots`` for constitutive state only.  Such
    snapshots must not change the displacement descriptor away from the
    authoritative final ``displacements`` array.
    """

    snapshots = tuple(getattr(result, "snapshots", ()) or ())
    if not snapshots:
        return None
    final = _finite_numeric_array(getattr(result, "displacements", None))
    if final is None or final.ndim != 1 or final.size % 6 != 0:
        return None
    values = tuple(getattr(snapshot, "displacements", None) for snapshot in snapshots)
    arrays = tuple(_finite_numeric_array(value) for value in values)
    if any(
        array is None
        or array.ndim != 1
        or array.size == 0
        or array.size % 6 != 0
        for array in arrays
    ):
        return None
    shapes = {array.shape for array in arrays if array is not None}
    return values if shapes == {final.shape} else None


def describe_result_quantities(result: Any) -> Tuple[ResultQuantity, ...]:
    """Return typed descriptors for arrays actually available on ``result``.

    The function is deliberately duck-typed so sidecar readers and downstream
    adapters can participate without inheriting solver result classes.
    """

    quantities = []
    class_name = type(result).__name__

    displacements = getattr(result, "displacements", None)
    snapshot_displacements = _snapshot_displacements(result)
    if _valid_nodal_array(displacements):
        frames = (
            len(snapshot_displacements)
            if snapshot_displacements is not None
            else _array_frames(displacements)
        )
        history_mode = str(getattr(result, "history_storage_mode", "full"))
        quantities.append(
            ResultQuantity(
                "displacement",
                "Displacement",
                "node",
                ("UX", "UY", "UZ", "RX", "RY", "RZ"),
                "mixed:m,rad",
                frame_count=frames,
                has_history=frames > 1,
                data_path=(
                    "snapshots[].displacements"
                    if snapshot_displacements is not None
                    else "displacements"
                ),
                recovery=(
                    "committed_state"
                    if snapshot_displacements is not None
                    else "native"
                ),
                metadata={
                    "history_storage_mode": history_mode,
                    "committed_state_snapshots": snapshot_displacements is not None,
                },
            )
        )

    if _valid_nodal_array(getattr(result, "velocities", None)):
        frames = _array_frames(result.velocities)
        quantities.append(
            ResultQuantity(
                "velocity",
                "Velocity",
                "node",
                ("VX", "VY", "VZ", "WX", "WY", "WZ"),
                "mixed:m/s,rad/s",
                frame_count=frames,
                has_history=True,
                data_path="velocities",
            )
        )
    if _valid_nodal_array(getattr(result, "accelerations", None)):
        frames = _array_frames(result.accelerations)
        quantities.append(
            ResultQuantity(
                "acceleration",
                "Acceleration",
                "node",
                ("AX", "AY", "AZ", "ALPHAX", "ALPHAY", "ALPHAZ"),
                "mixed:m/s^2,rad/s^2",
                frame_count=frames,
                has_history=True,
                data_path="accelerations",
            )
        )

    stress_payload = _physical_stress_frame(
        getattr(result, "element_stresses", None)
    )
    if stress_payload is not None:
        quantities.append(_stress_frame_descriptor(stress_payload))
    stress_history = getattr(result, "stress_history", None)
    history_payload = _physical_stress_history(stress_history)
    if history_payload is not None:
        filtered_history, history_bases, excluded_by_frame = history_payload
        history_components = tuple(sorted(history_bases))
        quantities.append(
            ResultQuantity(
                "stress_history",
                "Physical stress history",
                "element",
                history_components,
                "Pa",
                basis="component_specific",
                frame_count=len(filtered_history),
                has_history=True,
                data_path="stress_history[][physical_stress_components]",
                recovery="recovered_physical_stress_filter",
                metadata={
                    "component_basis": {
                        component: history_bases[component]
                        for component in history_components
                    },
                    "component_units": {
                        component: "Pa" for component in history_components
                    },
                    "excluded_nonphysical_element_ids_by_frame": excluded_by_frame,
                },
            )
        )
    reactions = getattr(result, "reactions", None)
    if isinstance(reactions, Mapping) and reactions:
        quantities.append(
            ResultQuantity(
                "reaction",
                "Reaction",
                "node",
                ("FX", "FY", "FZ", "MX", "MY", "MZ"),
                "mixed:N,N*m",
                data_path="reactions",
            )
        )

    steps = getattr(result, "steps", None)
    if steps is not None and len(steps) > 0:
        quantities.append(
            ResultQuantity(
                "load_factor",
                "Load factor",
                "global",
                ("FACTOR",),
                "1",
                frame_count=len(steps),
                has_history=True,
                data_path="steps[].load_factor",
            )
        )

    modes = getattr(result, "modes", None)
    if modes is not None and len(modes) > 0:
        mode_components = ("UX", "UY", "UZ", "RX", "RY", "RZ")
        quantities.append(
            ResultQuantity(
                "mode_shape",
                "Mode shape",
                "node",
                mode_components,
                "normalized",
                frame_count=len(modes),
                has_history=len(modes) > 1,
                data_path="modes[].mode_shape",
            )
        )
        if class_name == "ModalResult":
            quantities.append(ResultQuantity("frequency", "Frequency", "global", ("F",), "Hz", frame_count=len(modes), has_history=True, data_path="modes[].frequency_hz"))
        elif class_name == "BucklingResult":
            quantities.append(ResultQuantity("buckling_factor", "Buckling factor", "global", ("FACTOR",), "1", frame_count=len(modes), has_history=True, data_path="modes[].load_factor"))

    times = getattr(result, "times", None)
    times_array = _finite_numeric_array(times)
    if times_array is not None and times_array.ndim == 1:
        quantities.append(
            ResultQuantity(
                "time",
                "Time",
                "global",
                ("T",),
                "s",
                frame_count=int(times_array.size),
                has_history=True,
                data_path="times",
            )
        )
    if getattr(result, "contact_force_history", None) is not None:
        quantities.append(
            ResultQuantity(
                "contact_force",
                "Contact force",
                "contact",
                ("FX", "FY", "FZ"),
                "N",
                frame_count=_array_frames(result.contact_force_history),
                has_history=True,
                data_path="contact_force_history",
            )
        )
    if getattr(result, "sphere_positions", None) is not None:
        quantities.append(ResultQuantity("impactor_position", "Impactor position", "contact", ("X", "Y", "Z"), "m", frame_count=_array_frames(result.sphere_positions), has_history=True, data_path="sphere_positions"))
    if getattr(result, "force_impulse", None) is not None:
        quantities.append(ResultQuantity("force_impulse", "Force impulse", "global", ("IX", "IY", "IZ"), "N*s", data_path="force_impulse"))
    if getattr(result, "moment_impulse", None) is not None:
        quantities.append(ResultQuantity("moment_impulse", "Moment impulse", "global", ("IX", "IY", "IZ"), "N*m*s", data_path="moment_impulse"))

    for attribute, quantity_id, label, unit in (
        ("displacement_envelope", "displacement_envelope", "Displacement envelope", "mixed:m,rad"),
        ("velocity_envelope", "velocity_envelope", "Velocity envelope", "mixed:m/s,rad/s"),
        ("acceleration_envelope", "acceleration_envelope", "Acceleration envelope", "mixed:m/s^2,rad/s^2"),
    ):
        values = getattr(result, attribute, None)
        if _valid_nodal_array(values):
            quantities.append(
                ResultQuantity(
                    quantity_id,
                    label,
                    "node",
                    ("X", "Y", "Z", "RX", "RY", "RZ"),
                    unit,
                    data_path=attribute,
                    recovery="envelope",
                )
            )

    quantities = [
        descriptor
        for descriptor in quantities
        if _valid_described_data(
            descriptor.quantity_id,
            _described_data(result, descriptor.quantity_id),
            result=result,
        )
    ]

    for quantity_id in (
        "reaction_history",
        "equivalent_plastic_strain",
        "equivalent_plastic_strain_history",
        "load_impulse",
        "kinetic_energy",
        "strain_energy",
        "internal_work",
        "impactor_kinetic_energy",
    ):
        try:
            quantities.append(resolve_result_quantity(result, quantity_id).descriptor)
        except QuantityUnavailableError:
            continue

    return tuple(quantities)


def _resolved(descriptor: ResultQuantity, data: Any) -> ResolvedResultQuantity:
    return ResolvedResultQuantity(descriptor=descriptor, data=data)


def _nonempty_array(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(np.asarray(value).size)
    except (TypeError, ValueError):
        return False


def _finite_numeric_array(value: Any) -> np.ndarray | None:
    """Return a finite native numeric array, rejecting coercible text/bools."""

    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        native = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if native.size == 0 or native.dtype.kind not in "iuf":
        return None
    try:
        values = np.asarray(native, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    return values if np.all(np.isfinite(values)) else None


def _finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_finite_numeric_tree(item) for item in value.values())
    return _finite_numeric_array(value) is not None


_LOCAL_PHYSICAL_STRESS_COMPONENTS = frozenset(
    {
        "membrane_xx",
        "membrane_yy",
        "membrane_xy",
        "bending_xx",
        "bending_yy",
        "bending_xy",
        "shear_xz",
        "shear_yz",
        "axial_stress",
        "bending_stress_y",
        "bending_stress_z",
        "shear_stress_y",
        "shear_stress_z",
        "torsional_stress",
        "fiber_stress_min",
        "fiber_stress_max",
    }
)
_INVARIANT_PHYSICAL_STRESS_COMPONENTS = frozenset(
    {
        "equivalent_stress",
        "von_mises",
        "in_plane_von_mises",
        "mixed_reconstruction_von_mises",
        "longitudinal_fiber_von_mises",
        "fiber_von_mises_max",
        "fiber_mixed_reconstruction_von_mises_max",
        "von_mises_top",
        "von_mises_bot",
    }
)
_SURFACE_STRESS_SUFFIXES = tuple(
    f"_{component}_{surface}"
    for surface in ("top", "bot")
    for component in ("xx", "yy", "zz", "xy", "yz", "xz")
)


def _physical_stress_component_basis(component: Any) -> str | None:
    """Classify only physical stress values with units of pascals."""

    if not isinstance(component, str) or not component:
        return None
    if component in _INVARIANT_PHYSICAL_STRESS_COMPONENTS:
        return "invariant"
    if component in _LOCAL_PHYSICAL_STRESS_COMPONENTS:
        return "element_local"
    if component.startswith("local_") and component.endswith(
        _SURFACE_STRESS_SUFFIXES
    ):
        return "element_local"
    if component.startswith("global_") and component.endswith(
        _SURFACE_STRESS_SUFFIXES
    ):
        return "global"
    return None


def _physical_stress_frame(
    value: Any,
) -> Tuple[Mapping[int, Mapping[str, Any]], Dict[str, str], Tuple[int, ...]] | None:
    """Return a homogeneous physical-stress view of one recovery frame.

    Recovery bundles may also carry strains, utilization values, section
    resultants and descriptive metadata.  Those values are useful diagnostics,
    but they must not inherit a blanket ``Pa``/local-basis descriptor.  Records
    that explicitly contain generalized section resultants only are excluded.
    """

    if not isinstance(value, Mapping) or not value:
        return None
    filtered: Dict[int, Dict[str, Any]] = {}
    component_bases: Dict[str, str] = {}
    excluded = []
    canonical_ids = set()
    unchanged = True
    for raw_element_id, record in value.items():
        if not _valid_integer_identifier(raw_element_id):
            return None
        element_id = int(raw_element_id)
        if element_id in canonical_ids:
            return None
        canonical_ids.add(element_id)
        if not isinstance(record, Mapping) or not record:
            return None
        if (
            record.get("physical_stress_available") is False
            or record.get("generalized_stress_scope")
            == "section_resultants_only"
            or record.get("recovery_scope") == "section_resultants_only"
        ):
            unchanged = False
            excluded.append(element_id)
            continue
        physical: Dict[str, Any] = {}
        for component, data in record.items():
            basis = _physical_stress_component_basis(component)
            if basis is None:
                unchanged = False
                continue
            if _finite_numeric_array(data) is None:
                return None
            physical[component] = data
            prior_basis = component_bases.get(component)
            if prior_basis is not None and prior_basis != basis:
                return None
            component_bases[component] = basis
        if physical:
            filtered[element_id] = physical
        else:
            unchanged = False
            excluded.append(element_id)
    if not filtered:
        return None
    return (
        value if unchanged else filtered,
        component_bases,
        tuple(sorted(excluded)),
    )


def _stress_frame_descriptor(payload: Any) -> ResultQuantity:
    _data, bases, excluded = payload
    components = tuple(sorted(bases))
    return ResultQuantity(
        "stress", "Physical stress", "element", components, "Pa",
        basis="component_specific",
        data_path="element_stresses[physical_stress_components]",
        recovery="recovered_physical_stress_filter",
        metadata={
            "component_basis": {component: bases[component] for component in components},
            "component_units": {component: "Pa" for component in components},
            "excluded_nonphysical_element_ids": excluded,
        },
    )


def _native_stress_frame_eligible(result: Any) -> bool:
    """Admit only inert native carriers to single-pass selected resolution.

    Duck-typed results, mapping/array subclasses and callback-bearing physical
    values keep the general resolver's repeated reads and validation semantics.
    Eligibility checks types only; the complete physical-frame validator still
    checks every identifier, exclusion, component and current numeric value.
    """
    if not _is_exact_result_type(result, "recovery", "StressRecoveryResult"):
        return False
    class_fields = vars(type(result))
    if (
        "__getattribute__" in class_fields
        or "__getattr__" in class_fields
        or type(class_fields.get("__dict__")) is not GetSetDescriptorType
    ):
        return False
    field_definitions = class_fields.get("__dataclass_fields__")
    if type(field_definitions) is not dict or any(
        type(name) is not str for name in field_definitions
    ):
        return False
    expected = frozenset({
        "element_stresses", "provenance", "committed_element_states",
        "execution_report", "nodal_stresses",
    })
    if field_definitions.keys() != expected:
        return False
    if any(name in class_fields and class_fields[name] is not None for name in expected):
        return False
    # New class inputs can execute during the general inventory (including
    # energy/history probes). Keep extended native layouts on that route.
    allowed_class_names = expected | {
        "get_element_stress", "history_aware", "provenance_dict"
    }
    if any(
        not name.startswith("__") and name not in allowed_class_names
        for name in class_fields
    ):
        return False
    native_fields = vars(result)
    # Even an exact dataclass can carry a replacement __dict__ or an injected
    # __dataclass_fields__. Inspect neither through caller-defined protocols.
    if type(native_fields) is not dict or any(
        type(name) is not str for name in native_fields
    ):
        return False
    if not expected <= native_fields.keys() or not native_fields.keys() <= expected:
        return False
    if type(result.committed_element_states) is not dict or result.committed_element_states:
        return False
    value = result.element_stresses
    if type(value) is not dict:
        return False
    for element_id, record in value.items():
        if type(element_id) not in (int, str) or type(record) is not dict:
            return False
        for component, data in record.items():
            if type(component) is not str:
                return False
            if component in {
                "physical_stress_available", "generalized_stress_scope", "recovery_scope"
            }:
                if type(data) not in (str, bool, int, float, type(None)):
                    return False
            elif _physical_stress_component_basis(component) is not None:
                if type(data) not in (
                    np.ndarray, int, float, bool,
                    np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64, np.longdouble,
                ):
                    return False
    return True


def _physical_stress_history(
    value: Any,
) -> Tuple[
    Tuple[Dict[int, Dict[str, Any]], ...],
    Dict[str, str],
    Tuple[Tuple[int, ...], ...],
] | None:
    if not isinstance(value, (tuple, list)) or not value:
        return None
    frames = []
    component_bases: Dict[str, str] = {}
    excluded_by_frame = []
    for raw_frame in value:
        frame = _physical_stress_frame(raw_frame)
        if frame is None:
            return None
        filtered, bases, excluded = frame
        for component, basis in bases.items():
            prior_basis = component_bases.get(component)
            if prior_basis is not None and prior_basis != basis:
                return None
            component_bases[component] = basis
        frames.append(filtered)
        excluded_by_frame.append(excluded)
    return tuple(frames), component_bases, tuple(excluded_by_frame)


def _valid_reaction_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    canonical_ids = set()
    for raw_node_id, vector in value.items():
        if not _valid_integer_identifier(raw_node_id):
            return False
        canonical_id = int(raw_node_id)
        if canonical_id in canonical_ids:
            return False
        canonical_ids.add(canonical_id)
        array = _finite_numeric_array(vector)
        if array is None or array.shape != (6,):
            return False
    return True


def _valid_stress_data(value: Any, *, history: bool) -> bool:
    return (
        _physical_stress_history(value) is not None
        if history
        else _physical_stress_frame(value) is not None
    )


def _time_frame_count(result: Any) -> int | None:
    times = _finite_numeric_array(getattr(result, "times", None))
    if not _valid_time_axis(times):
        return None
    return int(times.size)


def _valid_time_axis(value: Any) -> bool:
    times = value if isinstance(value, np.ndarray) else _finite_numeric_array(value)
    return bool(
        times is not None
        and times.ndim == 1
        and times.size > 0
        and (times.size == 1 or np.all(np.diff(times) > 0.0))
    )


def _history_matches_time(result: Any, frame_count: int, *, required: bool) -> bool:
    time_frames = _time_frame_count(result)
    if time_frames is None:
        return not required
    return int(frame_count) == time_frames


def _valid_described_data(
    quantity_id: str, data: Any, *, result: Any = None
) -> bool:
    if quantity_id in {"stress", "stress_history"}:
        history = quantity_id == "stress_history"
        if not _valid_stress_data(data, history=history):
            return False
        return not history or _history_matches_time(
            result, len(data), required=True
        )
    if quantity_id == "reaction":
        return _valid_reaction_mapping(data)
    array = _finite_numeric_array(data)
    if array is None:
        return False
    if quantity_id == "displacement":
        if not _valid_nodal_array(data):
            return False
        if array.ndim == 1:
            return True
        if _snapshot_displacements(result) is not None:
            return True
        return _history_matches_time(result, array.shape[0], required=False)
    if quantity_id in {"velocity", "acceleration"}:
        return bool(
            _valid_nodal_array(data)
            and array.ndim == 2
            and _history_matches_time(result, array.shape[0], required=True)
        )
    if quantity_id == "mode_shape":
        return _valid_nodal_array(data) and array.ndim == 2
    if quantity_id == "time":
        return _valid_time_axis(array)
    if quantity_id in {"load_factor", "frequency", "buckling_factor"}:
        return array.ndim == 1
    if quantity_id in {"contact_force", "impactor_position"}:
        return bool(
            array.ndim == 2
            and array.shape[1] == 3
            and _history_matches_time(result, array.shape[0], required=True)
        )
    if quantity_id in {"force_impulse", "moment_impulse"}:
        return array.ndim == 1 and array.size == 3
    if quantity_id in {
        "displacement_envelope",
        "velocity_envelope",
        "acceleration_envelope",
    }:
        return _valid_nodal_array(data) and array.ndim == 1
    return False


def _described_data(result: Any, quantity_id: str) -> Any:
    if quantity_id == "stress":
        payload = _physical_stress_frame(
            getattr(result, "element_stresses", None)
        )
        return None if payload is None else payload[0]
    if quantity_id == "stress_history":
        payload = _physical_stress_history(
            getattr(result, "stress_history", None)
        )
        return None if payload is None else payload[0]
    attribute_by_id = {
        "displacement": "displacements",
        "velocity": "velocities",
        "acceleration": "accelerations",
        "reaction": "reactions",
        "time": "times",
        "contact_force": "contact_force_history",
        "impactor_position": "sphere_positions",
        "force_impulse": "force_impulse",
        "moment_impulse": "moment_impulse",
        "displacement_envelope": "displacement_envelope",
        "velocity_envelope": "velocity_envelope",
        "acceleration_envelope": "acceleration_envelope",
    }
    attribute = attribute_by_id.get(quantity_id)
    if attribute is not None:
        if quantity_id == "displacement":
            snapshots = _snapshot_displacements(result)
            if snapshots is not None:
                return snapshots
        return getattr(result, attribute, None)
    if quantity_id == "load_factor":
        return tuple(getattr(step, "load_factor", None) for step in result.steps)
    if quantity_id == "mode_shape":
        return tuple(getattr(mode, "mode_shape", None) for mode in result.modes)
    if quantity_id == "frequency":
        return tuple(getattr(mode, "frequency_hz", None) for mode in result.modes)
    if quantity_id == "buckling_factor":
        return tuple(getattr(mode, "load_factor", None) for mode in result.modes)
    return None


def _state_peeq(state: Any) -> Tuple[float, str] | None:
    if not isinstance(state, Mapping):
        return None
    for key in (
        "equivalent_plastic_strain",
        "max_equivalent_plastic_strain",
        "peeq",
        "alpha",
    ):
        if key not in state:
            continue
        value = state[key]
        values = _finite_numeric_array(value)
        if values is None or np.any(values < 0.0):
            raise QuantityUnavailableError(
                "equivalent plastic strain contains invalid committed state"
            )
        return float(np.max(values)), key
    return None


def _element_peeq(
    states: Any,
) -> Tuple[Dict[int, float], Dict[int, str]]:
    if not isinstance(states, Mapping):
        return {}, {}
    resolved: Dict[int, float] = {}
    source_keys: Dict[int, str] = {}
    canonical_ids = set()
    for raw_element_id, state in states.items():
        if not _valid_integer_identifier(raw_element_id):
            raise QuantityUnavailableError(
                "equivalent plastic strain contains an invalid element id"
            )
        element_id = int(raw_element_id)
        if element_id in canonical_ids:
            raise QuantityUnavailableError(
                "equivalent plastic strain contains duplicate element ids"
            )
        canonical_ids.add(element_id)
        if not isinstance(state, Mapping):
            raise QuantityUnavailableError(
                "equivalent plastic strain contains an invalid element state"
            )
        value_and_key = _state_peeq(state)
        if value_and_key is not None:
            resolved[element_id], source_keys[element_id] = value_and_key
    return resolved, source_keys


def _plastic_strain_quantity(
    result: Any, quantity_id: str
) -> ResolvedResultQuantity:
    snapshots = tuple(getattr(result, "snapshots", ()) or ())
    if quantity_id == "equivalent_plastic_strain":
        data_path = "element_states"
        data, source_keys = _element_peeq(
            getattr(result, "element_states", None)
        )
        if not data:
            data, source_keys = _element_peeq(
                getattr(result, "committed_element_states", None)
            )
            data_path = "committed_element_states"
        if not data and snapshots:
            data, source_keys = _element_peeq(
                getattr(snapshots[-1], "element_states", None)
            )
            data_path = "snapshots[-1].element_states"
        diagnostics = getattr(result, "diagnostics", None)
        if not data and isinstance(diagnostics, Mapping):
            data, source_keys = _element_peeq(diagnostics.get("element_states"))
            data_path = "diagnostics.element_states"
        if not data:
            raise QuantityUnavailableError("equivalent plastic strain is unavailable")
        descriptor = ResultQuantity(
            quantity_id,
            "Equivalent plastic strain (PEEQ)",
            "element",
            ("PEEQ",),
            "1",
            basis="material",
            data_path=data_path,
            recovery="committed_state_max_material_point",
            metadata={
                "source": "committed_constitutive_state",
                "source_keys": {
                    str(element_id): source_keys[element_id]
                    for element_id in sorted(source_keys)
                },
                "reduction": "maximum integration point/layer/fibre",
            },
        )
        return _resolved(descriptor, data)

    frames = []
    frame_indices = []
    load_factors = []
    control_values = []
    source_keys_by_frame = []
    prior_frame_index = -1
    for snapshot in snapshots:
        data, source_keys = _element_peeq(
            getattr(snapshot, "element_states", None)
        )
        if not data:
            continue
        raw_frame_index = getattr(snapshot, "step_index", None)
        raw_load_factor = getattr(snapshot, "load_factor", None)
        raw_control = getattr(snapshot, "control_value", None)
        if not _valid_integer_identifier(raw_frame_index):
            raise QuantityUnavailableError(
                "equivalent plastic strain history has no valid frame index"
            )
        load_factor = _finite_numeric_array(raw_load_factor)
        if load_factor is None or load_factor.ndim != 0:
            raise QuantityUnavailableError(
                "equivalent plastic strain history has no finite load factor"
            )
        if raw_control is not None:
            control = _finite_numeric_array(raw_control)
            if control is None or control.ndim != 0:
                raise QuantityUnavailableError(
                    "equivalent plastic strain history has invalid control data"
                )
            control_value = float(control)
        else:
            control_value = None
        frame_index = int(raw_frame_index)
        load_factor_value = float(load_factor)
        if frame_index <= prior_frame_index:
            raise QuantityUnavailableError(
                "equivalent plastic strain history frame ordering is invalid"
            )
        frames.append(data)
        frame_indices.append(frame_index)
        load_factors.append(load_factor_value)
        control_values.append(control_value)
        source_keys_by_frame.append(
            {
                str(element_id): source_keys[element_id]
                for element_id in sorted(source_keys)
            }
        )
        prior_frame_index = frame_index
    if not frames:
        raise QuantityUnavailableError("equivalent plastic strain history is unavailable")
    descriptor = ResultQuantity(
        quantity_id,
        "Equivalent plastic strain history (PEEQ)",
        "element",
        ("PEEQ",),
        "1",
        basis="material",
        frame_count=len(frames),
        has_history=True,
        data_path="snapshots[].element_states",
        recovery="saved_committed_state_max_material_point",
        metadata={
            "source": "committed_constitutive_snapshots",
            "frame_indices": frame_indices,
            "load_factors": load_factors,
            "control_values": control_values,
            "source_keys_by_frame": source_keys_by_frame,
            "reduction": "maximum integration point/layer/fibre",
            "unavailable_snapshots_omitted": True,
        },
    )
    return _resolved(descriptor, tuple(frames))


def _is_exact_result_type(result: Any, module: str, name: str) -> bool:
    module_name = f"anyfesolver.{module}"
    result_type = type(result)
    if result_type.__module__ != module_name or result_type.__name__ != name:
        return False
    try:
        return result_type is getattr(import_module(module_name), name)
    except (AttributeError, ImportError):
        return False


def _derived_step_reaction_history(result: Any) -> Tuple[ReactionFrame, ...]:
    if not (
        _is_exact_result_type(result, "nonlinear_static", "NonlinearStaticResult")
        or _is_exact_result_type(result, "arc_length", "ArcLengthResult")
    ):
        return ()
    steps = tuple(getattr(result, "steps", ()) or ())
    if not steps:
        return ()
    controls = tuple(getattr(step, "control_value", None) for step in steps)
    use_control = all(value is not None for value in controls)
    frames = []
    for step, control in zip(steps, controls):
        support_resultants = getattr(step, "support_reactions", None)
        if not isinstance(support_resultants, Mapping) or not support_resultants:
            if frames:
                raise QuantityUnavailableError(
                    "committed nonlinear reaction history is incomplete"
                )
            continue
        frames.append(
            ReactionFrame(
                frame_index=getattr(step, "step_index", None),
                abscissa=(
                    control if use_control else getattr(step, "load_factor", None)
                ),
                abscissa_kind="control_value" if use_control else "load_factor",
                reactions={},
                support_resultants=support_resultants,
            )
        )
    if frames and len(frames) != len(steps):
        raise QuantityUnavailableError(
            "committed nonlinear reaction history is incomplete"
        )
    return tuple(frames)


def _reaction_history_quantity(result: Any) -> ResolvedResultQuantity:
    frames = tuple(getattr(result, "reaction_history", ()) or ())
    derived = False
    if not frames:
        frames = _derived_step_reaction_history(result)
        derived = bool(frames)
    if not frames:
        raise QuantityUnavailableError("reaction history is unavailable")
    prior_index = -1
    prior_abscissa = float("-inf")
    abscissa_kind = ""
    for frame in frames:
        if not isinstance(frame, ReactionFrame):
            raise QuantityUnavailableError(
                "reaction history contains a non-ReactionFrame record"
            )
        if not _valid_integer_identifier(frame.frame_index):
            raise QuantityUnavailableError("reaction frame index is invalid")
        index = int(frame.frame_index)
        raw_abscissa = _finite_numeric_array(frame.abscissa)
        if raw_abscissa is None or raw_abscissa.ndim != 0:
            raise QuantityUnavailableError("reaction frame abscissa is invalid")
        abscissa = float(raw_abscissa)
        kind = frame.abscissa_kind
        monotone_abscissa = kind in {"time", "arc_length", "path_length"}
        if (
            index <= prior_index
            or not np.isfinite(abscissa)
            or not isinstance(kind, str)
            or not kind
            or (abscissa_kind and kind != abscissa_kind)
            or (monotone_abscissa and abscissa <= prior_abscissa)
        ):
            raise QuantityUnavailableError("reaction history frame ordering is invalid")
        if not isinstance(frame.reactions, Mapping) or not isinstance(
            frame.support_resultants, Mapping
        ):
            raise QuantityUnavailableError("reaction frame mappings are invalid")
        if not frame.reactions and not frame.support_resultants:
            raise QuantityUnavailableError("reaction frame carries no resultants")
        canonical_node_ids = set()
        for raw_node_id, value in frame.reactions.items():
            if not _valid_integer_identifier(raw_node_id):
                raise QuantityUnavailableError("reaction node id is invalid")
            canonical_node_id = int(raw_node_id)
            if canonical_node_id in canonical_node_ids:
                raise QuantityUnavailableError("reaction node ids are duplicated")
            canonical_node_ids.add(canonical_node_id)
            array = _finite_numeric_array(value)
            if array is None:
                raise QuantityUnavailableError("reaction frame data is invalid")
            if array.shape != (6,) or not np.all(np.isfinite(array)):
                raise QuantityUnavailableError("reaction frame data is invalid")
        for raw_support, value in frame.support_resultants.items():
            if not isinstance(raw_support, str) or not raw_support:
                raise QuantityUnavailableError("reaction support id is invalid")
            array = _finite_numeric_array(value)
            if array is None:
                raise QuantityUnavailableError("support reaction data is invalid")
            if array.shape != (6,) or not np.all(np.isfinite(array)):
                raise QuantityUnavailableError("support reaction data is invalid")
        prior_index = index
        prior_abscissa = abscissa
        abscissa_kind = kind
    descriptor = ResultQuantity(
        "reaction_history",
        "Reaction history",
        (
            "support"
            if derived
            else "mixed"
            if any(frame.reactions for frame in frames)
            and any(frame.support_resultants for frame in frames)
            else "node"
            if any(frame.reactions for frame in frames)
            else "support"
        ),
        ("FX", "FY", "FZ", "MX", "MY", "MZ"),
        "mixed:N,N*m",
        frame_count=len(frames),
        has_history=True,
        data_path=("steps[].support_reactions" if derived else "reaction_history"),
        recovery=("committed_step_support_resultants" if derived else "native"),
        metadata={
            "abscissa_kind": abscissa_kind,
            "nodal_reactions_available": any(frame.reactions for frame in frames),
            "support_resultants_available": any(
                frame.support_resultants for frame in frames
            ),
        },
    )
    return _resolved(descriptor, frames)


def _energy_quantity(result: Any, quantity_id: str) -> ResolvedResultQuantity:
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise QuantityUnavailableError(f"{quantity_id} is unavailable")
    measure = str(diagnostics.get("strain_energy_measure", ""))
    if not measure:
        method = str(diagnostics.get("method", ""))
        if method in {
            "newmark",
            "hht_alpha",
            "newmark_sphere_penalty_contact",
        }:
            measure = "elastic_strain_energy"
        elif method == "nonlinear_newmark_sphere_penalty_contact":
            measure = "internal_work_proxy"
    if quantity_id == "kinetic_energy":
        source_key = "kinetic_energy"
        label = "Kinetic energy"
        descriptor_measure = "structural_kinetic_energy"
    elif quantity_id == "strain_energy":
        if measure != "elastic_strain_energy":
            raise QuantityUnavailableError("elastic strain energy is unavailable")
        source_key = "strain_energy"
        label = "Strain energy"
        descriptor_measure = "elastic_strain_energy"
    elif quantity_id == "internal_work":
        if measure != "internal_work_proxy":
            raise QuantityUnavailableError("internal work is unavailable")
        source_key = "strain_energy"
        label = "Internal work"
        descriptor_measure = "committed_internal_work_proxy"
    else:
        source_key = "sphere_kinetic_energy"
        label = "Impactor kinetic energy"
        descriptor_measure = "rigid_impactor_kinetic_energy"
    values = diagnostics.get(source_key)
    array = _finite_numeric_array(values)
    if array is None:
        raise QuantityUnavailableError(f"{quantity_id} is unavailable")
    if array.ndim != 1:
        raise QuantityUnavailableError(f"{quantity_id} must be a one-dimensional history")
    times = _finite_numeric_array(getattr(result, "times", None))
    if (
        not _valid_time_axis(times)
        or int(times.size) != int(array.size)
    ):
        raise QuantityUnavailableError(f"{quantity_id} is not aligned with result frames")
    descriptor = ResultQuantity(
        quantity_id,
        label,
        "global",
        ("ENERGY",),
        "J",
        frame_count=int(array.size),
        has_history=True,
        data_path=f"diagnostics.{source_key}",
        recovery=(
            "committed_internal_force_work_proxy"
            if quantity_id == "internal_work"
            else "elastic_energy"
            if quantity_id == "strain_energy"
            else "native"
        ),
        metadata={
            "measure": descriptor_measure,
            "abscissa_path": "times",
            "abscissa": "time",
            "abscissa_unit": "s",
        },
    )
    return _resolved(descriptor, values)


def resolve_result_quantity(result: Any, quantity_id: str) -> ResolvedResultQuantity:
    """Resolve one canonical quantity without inventing absent values."""

    key = str(quantity_id)
    if key not in _REGISTERED_QUANTITY_IDS:
        raise QuantityUnavailableError(f"unknown result quantity {key!r}")
    if key in {
        "equivalent_plastic_strain",
        "equivalent_plastic_strain_history",
    }:
        return _plastic_strain_quantity(result, key)
    if key == "reaction_history":
        return _reaction_history_quantity(result)
    if key in {
        "kinetic_energy",
        "strain_energy",
        "internal_work",
        "impactor_kinetic_energy",
    }:
        return _energy_quantity(result, key)
    if key == "load_impulse":
        value = getattr(result, "load_impulse", None)
        array = _finite_numeric_array(value)
        if array is None or array.ndim != 1 or array.size % 6 != 0:
            raise QuantityUnavailableError("load impulse is unavailable")
        descriptor = ResultQuantity(
            key,
            "Load impulse",
            "node",
            ("IX", "IY", "IZ", "IRX", "IRY", "IRZ"),
            "mixed:N*s,N*m*s",
            data_path="load_impulse",
        )
        return _resolved(descriptor, value)

    if key == "stress" and _native_stress_frame_eligible(result):
        # Query-local reuse: the same admitted payload supplies both metadata
        # and returned data. Nothing survives this call or bypasses validation.
        payload = _physical_stress_frame(result.element_stresses)
        if payload is None:
            raise QuantityUnavailableError(
                f"{type(result).__name__} carries no result quantity {key!r}"
            )
        return _resolved(_stress_frame_descriptor(payload), payload[0])

    descriptors = {
        descriptor.quantity_id: descriptor
        for descriptor in describe_result_quantities(result)
    }
    descriptor = descriptors.get(key)
    if descriptor is None:
        raise QuantityUnavailableError(
            f"{type(result).__name__} carries no result quantity {key!r}"
        )
    data = _described_data(result, key)
    if data is None:
        raise QuantityUnavailableError(
            f"{type(result).__name__} carries no data for result quantity {key!r}"
        )
    if not _valid_described_data(key, data, result=result):
        raise QuantityUnavailableError(
            f"{type(result).__name__} carries no valid data for result quantity {key!r}"
        )
    return _resolved(descriptor, data)
