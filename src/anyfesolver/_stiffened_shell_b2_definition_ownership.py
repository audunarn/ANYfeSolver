"""Owned payloads for the public B2 definition facade; no operator compilation."""
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from types import MappingProxyType
import copy
import numpy as np
from scipy import sparse


def _value(value):
    if isinstance(value, np.ndarray):
        return _value(value.item()) if value.ndim == 0 else tuple(_value(x) for x in value)
    if isinstance(value, (tuple, list)):
        return tuple(_value(x) for x in value)
    return copy.deepcopy(value)


def _vector(value):
    try:
        return tuple(_value(x) for x in value)
    except TypeError:
        return copy.deepcopy(value)


class _Coordinates(Mapping):
    __slots__ = ("_data",)

    def __init__(self, values):
        object.__setattr__(self, "_data", MappingProxyType(
            {key: tuple(_value(x) for x in value) for key, value in values.items()}))

    def __setattr__(self, name, value):
        raise AttributeError("owned B2 coordinates are immutable")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


class _Array:
    """Exact numeric snapshot; each public view has detached array metadata."""
    __slots__ = ("_bytes", "_dtype", "_shape")

    def __init__(self, value):
        if isinstance(value, _Array):
            # Accept an already-owned payload defensively;
            # adopt the immutable snapshot instead of nesting it in an
            # object array, which would corrupt every numeric consumer.
            object.__setattr__(self, "_bytes", value._bytes)
            object.__setattr__(self, "_dtype", value._dtype)
            object.__setattr__(self, "_shape", value._shape)
            return
        array = np.array(value, copy=True)
        object.__setattr__(self, "_bytes", tuple(copy.deepcopy(x) for x in array.flat) if array.dtype.hasobject else array.tobytes())
        object.__setattr__(self, "_dtype", array.dtype)
        object.__setattr__(self, "_shape", array.shape)

    def __setattr__(self, name, value):
        raise AttributeError("owned B2 array payload is immutable")

    def __copy__(self):
        # Byte snapshots are immutable; a shallow copy shares them, including
        # object payloads, exactly like the caller-supplied source array would.
        clone = _Array.__new__(_Array)
        object.__setattr__(clone, "_bytes", self._bytes)
        object.__setattr__(clone, "_dtype", self._dtype)
        object.__setattr__(clone, "_shape", self._shape)
        return clone

    def __deepcopy__(self, memo):
        # Object payloads are deep-copied so a copy's materialized views can
        # never mutate the original's object data; immutable byte snapshots
        # and dtype/shape metadata are shared.
        clone = _Array.__new__(_Array)
        object.__setattr__(clone, "_bytes", copy.deepcopy(self._bytes, memo))
        object.__setattr__(clone, "_dtype", self._dtype)
        object.__setattr__(clone, "_shape", self._shape)
        return clone

    def materialize(self):
        if self._dtype.hasobject:
            array = np.empty(self._shape, dtype=self._dtype)
            for i, value in enumerate(self._bytes):
                array.flat[i] = copy.deepcopy(value)
            array.setflags(write=False)
            return array
        return np.frombuffer(self._bytes, dtype=self._dtype).reshape(self._shape)


class _CSR:
    """Sparse buffers and shape are private; consumers receive a fresh CSR."""
    __slots__ = ("_data", "_indices", "_indptr", "_shape", "_type")

    def __init__(self, value):
        csr = value.tocsr(copy=True)
        for name in ("data", "indices", "indptr"):
            object.__setattr__(self, "_" + name, _Array(getattr(csr, name)))
        object.__setattr__(self, "_shape", tuple(csr.shape))
        object.__setattr__(self, "_type", type(csr))

    def __setattr__(self, name, value):
        raise AttributeError("owned B2 sparse payload is immutable")

    def __copy__(self):
        clone = _CSR.__new__(_CSR)
        for name in ("data", "indices", "indptr"):
            object.__setattr__(clone, "_" + name, getattr(self, "_" + name))
        object.__setattr__(clone, "_shape", self._shape)
        object.__setattr__(clone, "_type", self._type)
        return clone

    def __deepcopy__(self, memo):
        clone = _CSR.__new__(_CSR)
        for name in ("data", "indices", "indptr"):
            object.__setattr__(clone, "_" + name, copy.deepcopy(getattr(self, "_" + name), memo))
        object.__setattr__(clone, "_shape", self._shape)
        object.__setattr__(clone, "_type", self._type)
        return clone

    def materialize(self):
        # Deliberately writable detached buffers: even attribute replacement on
        # this scipy object cannot modify the definition's byte snapshots.
        return self._type((self._data.materialize().copy(),
                           self._indices.materialize().copy(),
                           self._indptr.materialize().copy()), shape=self._shape)


class OwnedB2Definition:
    """Dataclass hook shared by the six existing public definition identities."""
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if isinstance(value, (_Array, _CSR)):
            return value.materialize()
        return value

    def __post_init__(self):
        fields = vars(self)
        for name in ("members", "footprints", "physical_footprints"):
            if name in fields and fields[name] is not None:
                object.__setattr__(self, name, tuple(fields[name]))
        if "shell_nodes_m" in fields and isinstance(fields["shell_nodes_m"], Mapping):
            object.__setattr__(self, "shell_nodes_m", _Coordinates(fields["shell_nodes_m"]))
        for name in ("shell_vertices_m", "selected_dofs"):
            if name in fields and fields[name] is not None:
                object.__setattr__(self, name, _Array(fields[name]))
        for name in ("reference_normal", "offset_local_m"):
            if name in fields and fields[name] is not None:
                object.__setattr__(self, name, _vector(fields[name]))
        if "shells" in fields:
            object.__setattr__(self, "shells", tuple(replace(shell,
                **{name: _value(getattr(shell, name)) for name in
                   ("E", "nu", "thickness_m")},
                reference_normal=_vector(shell.reference_normal),
                node_ids=tuple(_Array(x).materialize() if isinstance(x, np.ndarray) else copy.deepcopy(x) for x in shell.node_ids),
                density_kg_m3=_Array(shell.density_kg_m3).materialize() if isinstance(shell.density_kg_m3, np.ndarray) else copy.deepcopy(shell.density_kg_m3))
                if is_dataclass(shell) and hasattr(shell, "node_ids") and hasattr(shell, "reference_normal") else copy.deepcopy(shell) for shell in fields["shells"]))
        if "points" in fields:
            object.__setattr__(self, "points", tuple(replace(
                point, offset_local_m=_vector(point.offset_local_m), station=_value(point.station)) if is_dataclass(point) and hasattr(point, "offset_local_m") else copy.deepcopy(point) for point in fields["points"]))
        for name in ("shell_E", "shell_nu", "shell_thickness_m", "station"):
            if name in fields:
                object.__setattr__(self, name, _value(fields[name]))
        if "max_rhs_chunk" in fields:
            value = fields["max_rhs_chunk"]
            object.__setattr__(self, "max_rhs_chunk", _Array(value) if isinstance(value, np.ndarray) else copy.deepcopy(value))
        value = fields["transformation"]
        if sparse.issparse(value):
            object.__setattr__(self, "transformation", _CSR(value))
        elif value is not None:
            object.__setattr__(self, "transformation", _Array(value) if isinstance(value, np.ndarray) else copy.deepcopy(value))
        value = fields["prescribed_offset"]
        if value is not None:
            object.__setattr__(self, "prescribed_offset", _Array(value))
        # Equality-admitted zero-dimensional profile/policy arrays retain their
        # original dtype and shape; strict type gates must see the original type.
        for name, value in tuple(fields.items()):
            if isinstance(value, np.ndarray):
                object.__setattr__(self, name, _Array(value))
