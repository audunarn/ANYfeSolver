"""Bounded deterministic load-factor cutbacks; no constitutive approximation."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isfinite

from ._stiffened_shell_ge_checkpoint_binary import MAX_CHECKPOINT_BYTES


MAGIC = b"ANYSGECP10\0"
SCHEMA = "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_INCREMENT_CONTROL_V10"
MAX_METADATA_BYTES = 256 * 1024


@dataclass(frozen=True)
class AdaptiveIncrementControl:
    """Halve failed intervals and hold that step until the requested target.

    Limits apply to each invocation, including work in its restored accepted
    checkpoint. Rejected attempts after the last acceptance are reported on
    exhaustion but are not durable; retrying that prefix may repeat them.
    Defaults do not activate this optional experimental controller.
    """
    max_cutbacks: int = 6
    max_attempts: int = 64

    def __post_init__(self):
        if (type(self.max_cutbacks) is not int or not 0 <= self.max_cutbacks <= 20 or
                type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 1024):
            raise ValueError("bounded integer adaptive increment controls required")


@dataclass(frozen=True)
class IncrementAttempt:
    factor: float
    accepted_from: float
    target: float
    status: str
    iterations: int
    evaluations: int
    linear_solves: int
    rejected_trials: int
    equilibrium_ratio: float
    constraint_max: float
    failure: str | None = None


class FiniteIncrementFailure(RuntimeError):
    """Specifically retryable global Newton/backtrack exhaustion only."""
    def __init__(self, attempt):
        super().__init__(attempt.failure)
        self.attempt = attempt


class AdaptiveIncrementExhausted(RuntimeError):
    """Requested sequence incomplete; physical accepted prefix remains owned."""
    def __init__(self, reason, state, accepted_factors):
        super().__init__(reason)
        self.attempts = state.attempts
        self.requested_targets = state.targets
        self.accepted_factors = accepted_factors


@dataclass(frozen=True)
class _AdaptiveState:
    control: AdaptiveIncrementControl
    targets: tuple[float, ...]
    max_iterations: int
    max_backtracks: int
    cursor: int
    origin: float
    depth: int = 0
    attempts: tuple[IncrementAttempt, ...] = ()

    @property
    def completed(self):
        return self.cursor == len(self.targets)

    def next_factor(self, accepted):
        target = self.targets[self.cursor]
        step = (target - self.origin) / (2**self.depth)
        remaining = target - accepted
        return target if abs(step) >= abs(remaining) else accepted + step

    def accepted(self, attempt):
        finished = attempt.factor == self.targets[self.cursor]
        return replace(self, attempts=self.attempts + (attempt,),
                       cursor=self.cursor + int(finished),
                       origin=attempt.factor if finished else self.origin,
                       depth=0 if finished else self.depth)

    def rejected(self, attempt):
        return replace(self, attempts=self.attempts + (attempt,), depth=self.depth + 1)


def encode_control(model, state, inner):
    header = json.dumps({"schema": SCHEMA, "model": model, "state": asdict(state)},
                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(header) > MAX_METADATA_BYTES:
        raise ValueError("adaptive checkpoint metadata exceeds resource envelope")
    content = MAGIC + len(header).to_bytes(4, "little") + header + inner
    if len(content) + 32 > MAX_CHECKPOINT_BYTES:
        raise ValueError("adaptive checkpoint exceeds resource envelope")
    return content + sha256(content).digest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate adaptive checkpoint metadata")
        value[key] = item
    return value


def decode_control(raw, model):
    if (type(raw) is not bytes or not raw.startswith(MAGIC) or
            not len(MAGIC) + 4 + 32 < len(raw) <= MAX_CHECKPOINT_BYTES or
            sha256(raw[:-32]).digest() != raw[-32:]):
        raise ValueError("invalid adaptive checkpoint framing")
    start = len(MAGIC) + 4
    size = int.from_bytes(raw[len(MAGIC):start], "little")
    if not 0 < size <= MAX_METADATA_BYTES or start + size >= len(raw) - 32:
        raise ValueError("invalid adaptive checkpoint metadata bound")
    try:
        header = json.loads(raw[start:start+size], object_pairs_hook=_unique)
        if (type(header) is not dict or set(header) != {"schema", "model", "state"} or
                header["schema"] != SCHEMA or header["model"] != model):
            raise ValueError("incompatible adaptive checkpoint")
        if json.dumps(header,sort_keys=True,separators=(",", ":"),
                      allow_nan=False).encode("ascii") != raw[start:start+size]:
            raise ValueError("noncanonical adaptive checkpoint metadata")
        value = header["state"]
        if type(value) is not dict or set(value) != set(_AdaptiveState.__dataclass_fields__):
            raise ValueError("invalid adaptive controller fields")
        if (type(value["control"]) is not dict or
                set(value["control"]) != {"max_cutbacks", "max_attempts"} or
                type(value["targets"]) is not list or type(value["attempts"]) is not list):
            raise ValueError("invalid adaptive controller layout")
        control = AdaptiveIncrementControl(**value["control"])
        state = _AdaptiveState(**{**value, "control": control,
            "targets": tuple(value["targets"]),
            "attempts": tuple(IncrementAttempt(**v) for v in value["attempts"])})
    except (TypeError, KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid adaptive checkpoint metadata") from error
    return state, raw[start+size:-32]


def validate_control(state, factors, *, _allow_fibre_predictor=False,
                     _allow_coupled_predictor=False):
    """Replay bounded controller decisions, never element/material history.

    Only the admitted compensated fibre reader may allow one extra predictor
    solve per iteration and rejected full trial. No extra trial evaluations.
    Only the admitted coupled BL5 reader may allow the same bounded extra
    predictor solves and at most one probe evaluation per rejected full trial; the
    evaluation floor tracks solves and its ceiling includes failed local probes so the
    actual counters are never hidden. Default validation preserves the
    historical one-solve-per-iteration rule.
    """
    if (type(_allow_fibre_predictor) is not bool or
            type(_allow_coupled_predictor) is not bool or
            (_allow_fibre_predictor and _allow_coupled_predictor) or
            type(state) is not _AdaptiveState or not state.targets or
            any(type(v) is not float or not isfinite(v) or v <= 0. for v in state.targets) or
            any(a == b for a, b in zip(state.targets, state.targets[1:])) or
            type(state.max_iterations) is not int or not 1 <= state.max_iterations <= 24 or
            type(state.max_backtracks) is not int or not 0 <= state.max_backtracks <= 12 or
            type(state.cursor) is not int or not 0 <= state.cursor <= len(state.targets) or
            type(state.depth) is not int or not 0 <= state.depth <= state.control.max_cutbacks or
            type(state.origin) is not float or not isfinite(state.origin) or state.origin < 0. or
            not 0 < len(state.attempts) <= state.control.max_attempts):
        raise ValueError("invalid accepted adaptive controller state")
    accepted = tuple(row.factor for row in state.attempts if row.status == "accepted")
    if not accepted or tuple(factors[-len(accepted):]) != accepted:
        raise ValueError("adaptive controller disagrees with accepted load history")
    origin = float(factors[-len(accepted)-1]) if len(factors) > len(accepted) else 0.
    replay = _AdaptiveState(state.control, state.targets,
                            state.max_iterations, state.max_backtracks, 0, origin)
    previous = origin
    for row in state.attempts:
        if (replay.completed or row.factor != replay.next_factor(previous) or
                row.accepted_from != previous or row.target != replay.targets[replay.cursor] or
                any(type(v) is not float or not isfinite(v) for v in (
                    row.factor, row.accepted_from, row.target, row.equilibrium_ratio, row.constraint_max)) or
                min(row.equilibrium_ratio, row.constraint_max) < 0. or
                any(type(v) is not int or v < 0 for v in (
                    row.iterations, row.evaluations, row.linear_solves, row.rejected_trials)) or
                (not row.iterations <= row.linear_solves <= row.iterations + min(
                    row.iterations, row.rejected_trials)
                    if (_allow_fibre_predictor or _allow_coupled_predictor) else
                 row.iterations != row.linear_solves) or row.iterations > state.max_iterations or
                not (row.linear_solves if _allow_coupled_predictor else
                     (row.iterations if _allow_fibre_predictor else row.linear_solves)) + 1 <= row.evaluations <=
                    1 + (state.max_backtracks + 2) * state.max_iterations + (
                    min(row.iterations, row.rejected_trials) if _allow_coupled_predictor else 0) or
                row.rejected_trials > (state.max_backtracks + 1) * (
                    row.iterations if (_allow_fibre_predictor or _allow_coupled_predictor)
                    else row.linear_solves)):
            raise ValueError("invalid adaptive attempt or interval")
        if row.status == "accepted":
            if row.failure is not None or row.equilibrium_ratio > 1e-8 or row.constraint_max > 1e-9:
                raise ValueError("invalid adaptive acceptance")
            replay = replay.accepted(row)
            previous = row.factor
        elif row.status == "rejected":
            if (row.failure not in ("finite mixed Newton iteration limit",
                                   "finite mixed backtrack budget exhausted") or
                    replay.depth >= state.control.max_cutbacks):
                raise ValueError("invalid adaptive rejection")
            replay = replay.rejected(row)
        else:
            raise ValueError("invalid adaptive attempt status")
    if (replay != state or state.attempts[-1].status != "accepted" or
            previous != factors[-1]):
        raise ValueError("adaptive cursor disagrees with physical checkpoint")
