"""Immutable staged spatial translation dead forces for admitted GE owners.

This module owns descriptors, bounded compilation and checkpoint framing only.
The existing finite owner owns mechanics, supported scope and atomic publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import json
from math import floor, isfinite
from numbers import Real
import re
from typing import TYPE_CHECKING

import numpy as np

from ._stiffened_shell_ge_checkpoint_binary import MAX_CHECKPOINT_BYTES

if TYPE_CHECKING:
    from ._stiffened_shell_ge_experimental import NodalAction

POLICY = "B3_GE_E4_PL_STAGED_TRANSLATION_DEAD_LOAD_V1_DEV"
ZERO_REFERENCE_POLICY = "B3_GE_E4_PL_STAGED_ZERO_REFERENCE_V1_DEV"
MAGIC = b"ANYSGE_STAGE1\0"
MAX_STAGES = 64
MAX_ACTIONS_PER_STAGE = 4096
MAX_ACTIONS = 16384
MAX_METADATA_BYTES = 1024 * 1024
MAX_COMPILED_VALUES = 1_000_000


class StagedLoadResolutionError(ArithmeticError):
    """The prescribed stored-input interpolated force is not resolved."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _identity(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("64-hex model/programme identity required")
    return value.lower()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate staged-load JSON key")
        result[key] = value
    return result


def _json(raw):
    def invalid_constant(value):
        raise ValueError("nonfinite staged-load JSON constant")
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid staged-load JSON") from exc


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("finite real staged-load value required")
    try:
        made = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("finite binary64 staged-load value required") from exc
    if not isfinite(made) or (value != 0 and made == 0.):
        raise ValueError("finite resolved binary64 staged-load value required")
    return 0. if made == 0. else made


def _progress(stages, position):
    position = _number(position)
    if not 0. <= position <= stages:
        raise ValueError("staged-load position outside programme")
    cursor = min(floor(position), stages)
    return position, cursor, position-cursor


def _action(action):
    # Local import avoids a cycle when the owning GE module imports this seam.
    from ._stiffened_shell_ge_experimental import NodalAction
    if (type(action) is not NodalAction or type(action.domain) is not str or
        action.domain not in ("shell", "beam") or isinstance(action.node, (bool, np.bool_)) or
        not isinstance(action.node, (int, np.integer)) or action.node <= 0):
        raise ValueError("typed shell/beam staged nodal action required")
    if not isinstance(action.wrench, (tuple, list, np.ndarray)) or np.shape(action.wrench) != (6,):
        raise ValueError("six-component staged wrench required")
    wrench = tuple(_number(v) for v in action.wrench)
    if any(v != 0. for v in wrench[3:]):
        raise ValueError("staged dead loads admit translation forces only")
    return NodalAction(action.domain, int(action.node), wrench)


@dataclass(frozen=True)
class StagedDeadLoadProgramme:
    """Model-bound absolute stage endpoints; implicit first force F0=0.

    Integer programme positions are endpoints, not amplitude multipliers.
    Spatial forces remain dead forces as the mechanical configuration changes.
    """
    model_identity: str
    endpoints: tuple[tuple[NodalAction, ...], ...]
    zero_load_reference_endpoint: int | None = None
    identity: str = field(init=False)
    _serialized: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        model_identity = _identity(self.model_identity)
        if type(self.endpoints) is not tuple or not 1 <= len(self.endpoints) <= MAX_STAGES:
            raise ValueError("bounded nonempty tuple of staged endpoints required")
        if any(type(row) is not tuple or len(row) > MAX_ACTIONS_PER_STAGE for row in self.endpoints):
            raise ValueError("bounded tuple of actions at every stage required")
        if sum(map(len, self.endpoints)) > MAX_ACTIONS:
            raise ValueError("staged-load total action bound exceeded")
        endpoints = []
        for row in self.endpoints:
            normalized = tuple(_action(a) for a in row)
            nodes = [(a.domain, a.node) for a in normalized]
            if len(set(nodes)) != len(nodes):
                raise ValueError("duplicate same-node staged actions are not admitted")
            endpoints.append(normalized)
        object.__setattr__(self, "model_identity", model_identity)
        object.__setattr__(self, "endpoints", tuple(endpoints))
        reference = self.zero_load_reference_endpoint
        if reference is not None:
            if type(reference) is not int or not 1 <= reference <= len(endpoints):
                raise ValueError("zero-load reference must identify an existing endpoint")
            if not any(v != 0. for action in endpoints[reference-1] for v in action.wrench):
                raise ValueError("zero-load reference endpoint must be nonzero")
        core = self._core_payload()
        identity = sha256(_canonical(core)).hexdigest()
        object.__setattr__(self, "identity", identity)
        raw = _canonical({**core, "identity": identity})
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("staged-load descriptor metadata bound exceeded")
        object.__setattr__(self, "_serialized", raw)

    @property
    def stage_count(self):
        return len(self.endpoints)

    @property
    def policy(self):
        return POLICY if self.zero_load_reference_endpoint is None else ZERO_REFERENCE_POLICY

    @property
    def augmented_model_identity(self):
        return sha256(_canonical(dict(policy=self.policy, model_identity=self.model_identity,
                                      programme_identity=self.identity))).hexdigest()

    def _core_payload(self):
        value = dict(policy=self.policy, model_identity=self.model_identity,
                    endpoints=[[dict(domain=a.domain, node=a.node, wrench=list(a.wrench))
                                for a in row] for row in self.endpoints])
        if self.zero_load_reference_endpoint is not None:
            value["zero_load_reference_endpoint"] = self.zero_load_reference_endpoint
        return value

    def to_payload(self):
        return {**self._core_payload(), "identity": self.identity}

    def serialize(self):
        return self._serialized

    @classmethod
    def from_payload(cls, value):
        from ._stiffened_shell_ge_experimental import NodalAction
        if type(value) is not dict:
            raise ValueError("invalid staged-load descriptor fields/policy")
        required = {"policy", "model_identity", "endpoints", "identity"}
        if value.get("policy") == ZERO_REFERENCE_POLICY:
            required.add("zero_load_reference_endpoint")
        if (set(value) != required or value.get("policy") not in (POLICY, ZERO_REFERENCE_POLICY)
            or type(value["endpoints"]) is not list
            or not 1 <= len(value["endpoints"]) <= MAX_STAGES):
            raise ValueError("invalid staged-load descriptor fields/policy")
        rows = value["endpoints"]
        if (any(type(row) is not list or len(row) > MAX_ACTIONS_PER_STAGE for row in rows) or
            sum(map(len, rows)) > MAX_ACTIONS):
            raise ValueError("staged-load descriptor action bound exceeded")
        normalized = []
        for row in rows:
            actions = []
            for action in row:
                if type(action) is not dict or set(action) != {"domain", "node", "wrench"}:
                    raise ValueError("invalid staged-load action fields")
                if type(action["node"]) is not int or type(action["wrench"]) is not list:
                    raise ValueError("invalid staged-load action types")
                actions.append(NodalAction(action["domain"], action["node"], action["wrench"]))
            normalized.append(tuple(actions))
        programme = cls(value["model_identity"], tuple(normalized),
                        zero_load_reference_endpoint=value.get("zero_load_reference_endpoint"))
        if (_identity(value["identity"]) != programme.identity or
            _canonical(value) != programme.serialize()):
            raise ValueError("staged-load descriptor identity/canonical representation mismatch")
        return programme

    @classmethod
    def deserialize(cls, raw):
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_METADATA_BYTES:
            raise ValueError("bounded staged-load descriptor bytes required")
        programme = cls.from_payload(_json(raw))
        if raw != programme.serialize():
            raise ValueError("canonical staged-load descriptor bytes required")
        return programme

    def compile(self, owner, load):
        if getattr(owner, "_identity", None) != self.model_identity:
            raise ValueError("staged-load programme belongs to a different model")
        if not callable(load):
            raise TypeError("staged-load compiler requires the owner load callable")
        if self.zero_load_reference_endpoint is not None:
            definition = getattr(owner, "_definition", None)
            from ._stiffened_shell_ge_experimental import PROFILE
            if (definition is None or definition.profile != PROFILE or
                not definition.points or definition.lines or definition.footprints or
                definition.distributing_footprints or definition.finite_distributing_footprints or
                definition.shell_reference_policy != "flat"):
                raise ValueError("zero-load reference policy requires the admitted flat point GE owner")
        shape = getattr(getattr(owner, "_K", None), "shape", None)
        if (type(shape) is not tuple or len(shape) != 2 or shape[0] != shape[1] or
            not isinstance(shape[0], (int, np.integer)) or shape[0] <= 0 or
            (self.stage_count+1)*shape[0] > MAX_COMPILED_VALUES):
            raise ValueError("staged-load compiled vector shape/storage bound exceeded")
        # No compiler/loader exceptions are swallowed or converted to cutbacks.
        vectors = []
        for row in ((),)+self.endpoints:
            made = np.asarray(load(row), dtype=np.float64)
            if made.shape != (shape[0],) or not np.isfinite(made).all():
                raise ValueError("staged-load compiler returned invalid vectors")
            # Snapshot before the next call: loaders may reuse a work buffer.
            vectors.append(np.frombuffer(made.tobytes(), dtype=np.float64))
        if self.zero_load_reference_endpoint is not None:
            reference = vectors[self.zero_load_reference_endpoint][owner._free]
            scales = np.asarray(owner._coordinate_scale)[owner._free]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                weighted = reference*scales
            if (not np.isfinite(weighted).all() or
                np.any((reference != 0.) & (weighted == 0.)) or not np.any(weighted != 0.)):
                raise ValueError("representable nonzero free work required at the reference endpoint")
        return CompiledStagedDeadLoadProgramme(self, tuple(vectors))


@dataclass(frozen=True)
class CompiledStagedDeadLoadProgramme:
    programme: StagedDeadLoadProgramme
    vectors: tuple[np.ndarray, ...]

    def __post_init__(self):
        if (type(self.programme) is not StagedDeadLoadProgramme or type(self.vectors) is not tuple or
            len(self.vectors) != self.programme.stage_count+1):
            raise ValueError("complete typed staged-load compilation required")
        shape = np.shape(self.vectors[0])
        if len(shape) != 1 or shape[0] <= 0 or len(self.vectors)*shape[0] > MAX_COMPILED_VALUES:
            raise ValueError("staged-load compiled storage bound exceeded")
        owned = []
        for vector in self.vectors:
            made = np.asarray(vector, dtype=np.float64)
            if made.shape != shape or not np.isfinite(made).all():
                raise ValueError("finite consistent staged-load vectors required")
            owned.append(np.frombuffer(made.tobytes(), dtype=np.float64))
        if np.any(owned[0] != 0.):
            raise ValueError("implicit staged initial force must be zero")
        reference = self.programme.zero_load_reference_endpoint
        if reference is not None and not np.any(owned[reference] != 0.):
            raise ValueError("compiled zero-load reference endpoint must be nonzero")
        object.__setattr__(self, "vectors", tuple(owned))

    @property
    def identity(self):
        return self.programme.identity

    @property
    def physical_model_identity(self):
        return self.programme.model_identity

    @property
    def augmented_model_identity(self):
        return self.programme.augmented_model_identity

    @property
    def model_identity(self):
        return self.augmented_model_identity

    @property
    def stage_count(self):
        return self.programme.stage_count

    def equilibrium_load(self, applied):
        """Normalization only: exact zero loads use the selected physical endpoint.

        All nonzero loads retain their actual-load metric, including tiny loads.
        This does not replace the applied force or alter a residual/tangent.
        """
        reference = self.programme.zero_load_reference_endpoint
        if reference is not None and not np.any(applied != 0.):
            return self.vectors[reference]
        return applied

    def progress_at(self, position):
        _, cursor, fraction = _progress(self.stage_count, position)
        return cursor, fraction

    def load_at(self, position):
        _, cursor, fraction = _progress(self.stage_count, position)
        if fraction == 0.:
            return self.vectors[cursor]
        a, b = self.vectors[cursor], self.vectors[cursor+1]
        result = np.zeros_like(a)
        wb = Fraction(fraction)
        wa = 1-wb
        # Only active spatial components need a bounded exact two-term dot.
        # This avoids overflow-prone endpoint subtraction and false zero work.
        for index in np.flatnonzero((a != 0.) | (b != 0.)):
            if a[index] == b[index]:
                result[index] = a[index]
                continue
            exact = wa*Fraction(float(a[index])) + wb*Fraction(float(b[index]))
            try:
                value = float(exact)
            except OverflowError as exc:
                raise StagedLoadResolutionError("staged force interpolation overflow") from exc
            if (not isfinite(value) or (exact != 0 and value == 0.) or
                (0. < abs(value) < np.finfo(float).tiny and Fraction(value) != exact)):
                raise StagedLoadResolutionError("staged force interpolation is unresolved")
            result[index] = value
        return np.frombuffer(result.tobytes(), dtype=np.float64)

    def validate_targets(self, accepted_position, targets):
        accepted, _, _ = _progress(self.stage_count, accepted_position)
        if type(targets) is not tuple or not targets:
            raise ValueError("nonempty tuple of staged positions required")
        normalized = tuple(_progress(self.stage_count, p)[0] for p in targets)
        if any(b <= a for a, b in zip((accepted,)+normalized, normalized)):
            raise ValueError("staged positions must advance strictly forward")
        required = set(range(floor(accepted)+1, floor(normalized[-1])+1))
        if not required.issubset(normalized):
            raise ValueError("staged positions must include every crossed integer boundary")
        return normalized


def encode_checkpoint(compiled, position, inner):
    if type(compiled) is not CompiledStagedDeadLoadProgramme:
        raise TypeError("typed compiled staged programme required")
    position, cursor, fraction = _progress(compiled.stage_count, position)
    if type(inner) is not bytes or not inner or len(inner) > MAX_CHECKPOINT_BYTES:
        raise ValueError("bounded staged inner checkpoint bytes required")
    header = _canonical(dict(policy=compiled.programme.policy, programme=compiled.programme.to_payload(),
        position=position, cursor=cursor, fraction=fraction, inner_sha256=sha256(inner).hexdigest()))
    size = len(MAGIC)+4+len(header)+len(inner)+32
    if len(header) > MAX_METADATA_BYTES or size > MAX_CHECKPOINT_BYTES:
        raise ValueError("staged checkpoint metadata/total byte bound exceeded")
    content = MAGIC+len(header).to_bytes(4, "little")+header+inner
    return content+sha256(content).digest()


def decode_checkpoint(raw):
    if (type(raw) is not bytes or not raw.startswith(MAGIC) or
        not len(MAGIC)+4+32 < len(raw) <= MAX_CHECKPOINT_BYTES):
        raise ValueError("invalid bounded staged checkpoint framing")
    start = len(MAGIC)+4
    size = int.from_bytes(raw[len(MAGIC):start], "little")
    if not 0 < size <= MAX_METADATA_BYTES or start+size >= len(raw)-32:
        raise ValueError("invalid staged checkpoint metadata bound")
    if sha256(memoryview(raw)[:-32]).digest() != raw[-32:]:
        raise ValueError("staged checkpoint content digest mismatch")
    header_raw = raw[start:start+size]
    header = _json(header_raw)
    if (type(header) is not dict or set(header) != {"policy", "programme", "position", "cursor", "fraction", "inner_sha256"}
        or header["policy"] not in (POLICY, ZERO_REFERENCE_POLICY) or type(header["position"]) is not float
        or type(header["fraction"]) is not float or type(header["cursor"]) is not int
        or _canonical(header) != header_raw):
        raise ValueError("invalid staged checkpoint fields/types/canonical metadata")
    programme = StagedDeadLoadProgramme.from_payload(header["programme"])
    if header["policy"] != programme.policy:
        raise ValueError("staged checkpoint normalization policy mismatch")
    position, cursor, fraction = _progress(programme.stage_count, header["position"])
    if (header["cursor"] != cursor or header["fraction"] != fraction or
        _canonical({**header, "position": position, "cursor": cursor, "fraction": fraction}) != header_raw):
        raise ValueError("staged checkpoint progress derivation mismatch")
    inner = raw[start+size:-32]
    if header["inner_sha256"] != sha256(inner).hexdigest():
        raise ValueError("staged checkpoint inner digest mismatch")
    return programme, position, inner
