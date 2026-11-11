"""Common solve termination contract for qualified analysis results.

Legacy result classes retain their analysis-specific ``status`` or
``solver_status`` strings.  :class:`SolveOutcome` adds one uniform layer for
job orchestration: the disposition says whether usable work completed, while
``termination`` preserves the precise machine-readable solver reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib import import_module
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable


class SolveDisposition(str, Enum):
    """High-level disposition shared by every qualified solve family."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SolveOutcome:
    """Uniform result/job interpretation of one solver termination.

    ``termination`` is an analysis-specific stable code such as
    ``target_load_factor_reached`` or ``minimum_arc_radius_reached``.
    ``disposition`` is deliberately coarse and suitable for a job manager.
    A partial outcome always carries usable results but did not reach the
    requested target.
    """

    disposition: SolveDisposition
    termination: str
    target_reached: bool
    converged: bool
    has_results: bool
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Added after the complete 0.2 constructor surface so direct positional
    # construction keeps ``metadata`` in its historical position.
    control_kind: str = ""
    requested_control: Optional[float] = None
    achieved_control: Optional[float] = None
    last_converged_frame: Optional[int] = None
    last_converged_increment: Optional[int] = None

    def __post_init__(self) -> None:
        disposition = SolveDisposition(self.disposition)
        if not isinstance(self.termination, str) or not self.termination:
            raise TypeError("SolveOutcome termination must be a non-empty string")
        termination = self.termination
        for name in ("target_reached", "converged", "has_results"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"SolveOutcome {name} must be bool")
        if not isinstance(self.message, str) or not isinstance(self.control_kind, str):
            raise TypeError("SolveOutcome message and control_kind must be strings")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("SolveOutcome metadata must be a mapping")
        if disposition is SolveDisposition.PARTIAL and not self.has_results:
            raise ValueError("a partial SolveOutcome must contain usable results")
        if disposition is SolveDisposition.COMPLETED and not (
            self.target_reached and self.converged and self.has_results
        ):
            raise ValueError(
                "a completed SolveOutcome must be converged, target-reached, and usable"
            )
        if disposition is SolveDisposition.PARTIAL and self.target_reached:
            raise ValueError("a partial SolveOutcome cannot reach the requested target")
        if disposition in {SolveDisposition.FAILED, SolveDisposition.CANCELLED} and self.target_reached:
            raise ValueError("failed or cancelled outcomes cannot reach the requested target")
        if disposition in {SolveDisposition.FAILED, SolveDisposition.CANCELLED} and self.converged:
            raise ValueError("failed or cancelled outcomes cannot be converged")
        if disposition is SolveDisposition.FAILED and self.has_results:
            raise ValueError("a failed SolveOutcome cannot contain usable results")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "termination", termination)
        for name in ("requested_control", "achieved_control"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"SolveOutcome {name} must be finite numeric or None")
            object.__setattr__(self, name, None if value is None else float(value))
        for name in ("last_converged_frame", "last_converged_increment"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
            ):
                raise TypeError(f"SolveOutcome {name} must be a non-negative integer or None")
            object.__setattr__(self, name, None if value is None else int(value))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def status(self) -> str:
        """String form used by JSON/job records."""

        return self.disposition.value

    @property
    def completed(self) -> bool:
        return self.disposition is SolveDisposition.COMPLETED

    @property
    def partial(self) -> bool:
        return self.disposition is SolveDisposition.PARTIAL

    @property
    def failed(self) -> bool:
        return self.disposition is SolveDisposition.FAILED

    @property
    def cancelled(self) -> bool:
        return self.disposition is SolveDisposition.CANCELLED

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["disposition"] = self.disposition.value
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolveOutcome":
        if not isinstance(value, Mapping):
            raise TypeError("SolveOutcome.from_dict requires a mapping")
        allowed = {
            "disposition",
            "status",
            "termination",
            "target_reached",
            "converged",
            "has_results",
            "message",
            "metadata",
            "control_kind",
            "requested_control",
            "achieved_control",
            "last_converged_frame",
            "last_converged_increment",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"unknown SolveOutcome fields: {sorted(unknown)}")
        if ("disposition" in value) == ("status" in value):
            raise ValueError("SolveOutcome requires exactly one of disposition or status")
        required = {"termination", "target_reached", "converged", "has_results"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"missing SolveOutcome fields: {sorted(missing)}")
        raw_disposition = value.get("disposition", value.get("status"))
        return cls(
            disposition=(
                raw_disposition
                if isinstance(raw_disposition, SolveDisposition)
                else SolveDisposition(str(raw_disposition))
            ),
            termination=value["termination"],
            target_reached=value["target_reached"],
            converged=value["converged"],
            has_results=value["has_results"],
            message=value.get("message", ""),
            control_kind=value.get("control_kind", ""),
            requested_control=value.get("requested_control"),
            achieved_control=value.get("achieved_control"),
            last_converged_frame=value.get("last_converged_frame"),
            last_converged_increment=value.get("last_converged_increment"),
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def success(
        cls,
        termination: str,
        *,
        has_results: bool = True,
        message: str = "",
        control_kind: str = "",
        requested_control: Optional[float] = None,
        achieved_control: Optional[float] = None,
        last_converged_frame: Optional[int] = None,
        last_converged_increment: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SolveOutcome":
        return cls(
            SolveDisposition.COMPLETED,
            termination,
            target_reached=True,
            converged=True,
            has_results=has_results,
            message=message,
            control_kind=control_kind,
            requested_control=requested_control,
            achieved_control=achieved_control,
            last_converged_frame=last_converged_frame,
            last_converged_increment=last_converged_increment,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def stopped(
        cls,
        termination: str,
        *,
        has_results: bool = True,
        converged: bool = True,
        message: str = "",
        control_kind: str = "",
        requested_control: Optional[float] = None,
        achieved_control: Optional[float] = None,
        last_converged_frame: Optional[int] = None,
        last_converged_increment: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SolveOutcome":
        return cls(
            SolveDisposition.PARTIAL,
            termination,
            target_reached=False,
            converged=converged,
            has_results=has_results,
            message=message,
            control_kind=control_kind,
            requested_control=requested_control,
            achieved_control=achieved_control,
            last_converged_frame=last_converged_frame,
            last_converged_increment=last_converged_increment,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def failure(
        cls,
        termination: str,
        *,
        has_results: bool = False,
        message: str = "",
        control_kind: str = "",
        requested_control: Optional[float] = None,
        achieved_control: Optional[float] = None,
        last_converged_frame: Optional[int] = None,
        last_converged_increment: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SolveOutcome":
        if has_results:
            return cls.stopped(
                termination,
                has_results=True,
                converged=False,
                message=message,
                control_kind=control_kind,
                requested_control=requested_control,
                achieved_control=achieved_control,
                last_converged_frame=last_converged_frame,
                last_converged_increment=last_converged_increment,
                metadata=metadata,
            )
        return cls(
            SolveDisposition.FAILED,
            termination,
            target_reached=False,
            converged=False,
            has_results=False,
            message=message,
            control_kind=control_kind,
            requested_control=requested_control,
            achieved_control=achieved_control,
            last_converged_frame=last_converged_frame,
            last_converged_increment=last_converged_increment,
            metadata={} if metadata is None else metadata,
        )


@runtime_checkable
class SupportsSolveOutcome(Protocol):
    """Structural protocol implemented by qualified result bundles."""

    @property
    def outcome(self) -> SolveOutcome: ...


def _is_result_type(result: Any, module: str, name: str) -> bool:
    result_type = type(result)
    module_name = f"anyfesolver.{module}"
    if result_type.__module__ != module_name or result_type.__name__ != name:
        return False
    try:
        declared_type = getattr(import_module(module_name), name)
    except (AttributeError, ImportError):
        return False
    return result_type is declared_type


def _result_settings(result: Any) -> Mapping[str, Any]:
    result_case = getattr(result, "result_case", None)
    if not isinstance(result_case, Mapping):
        for container_name in ("info", "diagnostics"):
            container = getattr(result, container_name, None)
            if isinstance(container, Mapping) and isinstance(
                container.get("result_case"), Mapping
            ):
                result_case = container["result_case"]
                break
    if not isinstance(result_case, Mapping):
        return {}
    analysis_case = result_case.get("analysis_case")
    if not isinstance(analysis_case, Mapping):
        return {}
    settings = analysis_case.get("settings")
    return settings if isinstance(settings, Mapping) else {}


def _sequence_size(value: Any) -> int:
    try:
        return len(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _last_numeric(value: Any) -> Optional[float]:
    if _sequence_size(value) == 0:
        return None
    try:
        return float(value[-1])
    except (TypeError, ValueError, OverflowError, IndexError) as error:
        raise TypeError("result control history is malformed") from error


def _adapt_nonlinear_static(result: Any) -> SolveOutcome:
    info = getattr(result, "info", None)
    info = info if isinstance(info, Mapping) else {}
    steps = tuple(getattr(result, "steps", ()) or ())
    snapshots = tuple(getattr(result, "snapshots", ()) or ())
    status = str(getattr(result, "status", "unknown"))
    failure_reason = info.get("failure_reason")
    termination = str(info.get("stop_reason") or failure_reason or status)
    settings = _result_settings(result)
    control = str(settings.get("control", "force")).strip().lower()
    if control == "displacement":
        history = info.get("force_displacement_history")
        history = history if isinstance(history, (list, tuple)) else ()
        final_history = history[-1] if history and isinstance(history[-1], Mapping) else {}
        requested = final_history.get("target_displacement")
        last_step = steps[-1] if steps else None
        achieved = None if last_step is None else getattr(last_step, "control_value", None)
        control_kind = "prescribed_displacement"
    else:
        requested = settings.get("max_load_factor")
        achieved = getattr(result, "load_factor", None) if steps else None
        control_kind = "load_factor"
    fields = {
        "control_kind": control_kind,
        "requested_control": requested,
        "achieved_control": achieved,
        "last_converged_frame": len(snapshots) - 1 if snapshots else None,
        "last_converged_increment": (
            getattr(steps[-1], "step_index", None) if steps else None
        ),
    }
    if status == "completed" and failure_reason is None:
        return SolveOutcome.success(
            termination,
            **fields,
            metadata={"load_factor": getattr(result, "load_factor", None)},
        )
    if steps:
        return SolveOutcome.stopped(
            termination,
            converged=status == "stopped_at_limit",
            **fields,
            metadata={
                "load_factor": getattr(result, "load_factor", None),
                "legacy_status": status,
            },
        )
    return SolveOutcome.failure(
        termination,
        message=str(failure_reason or ""),
        **fields,
    )


def _adapt_arc_length(result: Any) -> SolveOutcome:
    steps = tuple(getattr(result, "steps", ()) or ())
    snapshots = tuple(getattr(result, "snapshots", ()) or ())
    status = str(getattr(result, "status", "unknown"))
    info = getattr(result, "info", None)
    info = info if isinstance(info, Mapping) else {}
    fields = {
        "control_kind": "arc_length",
        "requested_control": None,
        "achieved_control": (
            sum(abs(float(getattr(step, "path_increment_norm"))) for step in steps)
            if steps
            else None
        ),
        "last_converged_frame": len(snapshots) - 1 if snapshots else None,
        "last_converged_increment": (
            getattr(steps[-1], "step_index", None) if steps else None
        ),
    }
    successful = {
        "peak_confirmed",
        "load_factor_limit_reached",
        "post_buckling_traced",
        "displacement_limit_reached",
    }
    if status in successful:
        return SolveOutcome.success(
            status,
            **fields,
            metadata={"load_factor": getattr(result, "load_factor", None)},
        )
    termination = str(info.get("failure_reason") or status)
    if steps:
        return SolveOutcome.stopped(
            termination,
            converged=bool(getattr(result, "converged", False)),
            **fields,
            metadata={"load_factor": getattr(result, "load_factor", None)},
        )
    return SolveOutcome.failure(termination, **fields)


def _time_control_fields(result: Any) -> Dict[str, Any]:
    times = getattr(result, "times", ())
    diagnostics = getattr(result, "diagnostics", None)
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    settings = _result_settings(result)
    steps = diagnostics.get("num_substeps", diagnostics.get("num_steps"))
    return {
        "control_kind": "time",
        "requested_control": settings.get("t_end"),
        "achieved_control": _last_numeric(times),
        "last_converged_frame": _sequence_size(times) - 1 if _sequence_size(times) else None,
        "last_converged_increment": steps,
    }


def _adapt_transient(result: Any, *, impact: bool) -> SolveOutcome:
    status = str(getattr(result, "status", "unknown"))
    diagnostics = getattr(result, "diagnostics", None)
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    termination = str(diagnostics.get("stop_reason", status))
    fields = _time_control_fields(result)
    successful = {"completed", "no_contact"} if impact else {"completed"}
    if status in successful:
        return SolveOutcome.success(
            "transient_end_time_reached" if not impact else termination,
            **fields,
        )
    if _sequence_size(getattr(result, "times", ())) > 1:
        return SolveOutcome.stopped(termination, converged=False, **fields)
    return SolveOutcome.failure(termination, **fields)


def _adapt_modes(result: Any, *, buckling: bool) -> SolveOutcome:
    modes = tuple(getattr(result, "modes", ()) or ())
    status = str(getattr(result, "solver_status", "unknown"))
    requested = getattr(result, "num_modes_requested", None)
    if (
        isinstance(requested, bool)
        or not isinstance(requested, Integral)
        or int(requested) <= 0
    ):
        raise TypeError("mode result num_modes_requested must be a positive integer")
    requested = int(requested)
    if len(modes) > requested:
        raise TypeError("mode result contains more modes than requested")
    fields = {
        "control_kind": "mode_count",
        "requested_control": requested,
        "achieved_control": len(modes),
        "last_converged_frame": len(modes) - 1 if modes else None,
        "last_converged_increment": None,
    }
    if status == "ok" and len(modes) == requested:
        return SolveOutcome.success(
            "positive_buckling_modes_extracted"
            if buckling
            else "requested_modes_extracted",
            **fields,
        )
    if status == "ok" and modes:
        return SolveOutcome.stopped(
            "partial_buckling_modes_extracted"
            if buckling
            else "partial_modes_extracted",
            converged=True,
            **fields,
            metadata={"num_modes_returned": len(modes)},
        )
    return SolveOutcome.failure(
        status,
        has_results=bool(modes),
        **fields,
        metadata={"num_modes_returned": len(modes)},
    )


def _adapt_fe_result(result: Any) -> SolveOutcome:
    solver_info = getattr(result, "solver_info", None)
    if not isinstance(solver_info, Mapping):
        raise TypeError("FEResult solver_info is not a mapping")
    stored = solver_info.get("outcome")
    if stored is not None:
        if not isinstance(stored, Mapping):
            raise TypeError("FEResult outcome is not a mapping")
        return SolveOutcome.from_dict(stored)
    convergence = solver_info.get("convergence_info")
    convergence = convergence if isinstance(convergence, Mapping) else {}
    status = str(convergence.get("status", solver_info.get("status", "unknown")))
    if status == "converged":
        return SolveOutcome.success(
            "linear_equilibrium_converged",
            control_kind="load_factor",
            requested_control=1.0,
            achieved_control=1.0,
        )
    return SolveOutcome.failure(
        status,
        message=str(convergence.get("error", "")),
        control_kind="load_factor",
        requested_control=1.0,
    )


def _adapt_known_result(result: Any) -> Optional[SolveOutcome]:
    """Adapt only concrete public ANYfeSolver result carriers.

    This deliberately uses an exact module/class allowlist.  Duck-typed status
    objects and third-party wrappers are not guessed; they remain fail-closed.
    """

    if _is_result_type(result, "nonlinear_static", "NonlinearStaticResult"):
        return _adapt_nonlinear_static(result)
    if _is_result_type(result, "arc_length", "ArcLengthResult"):
        return _adapt_arc_length(result)
    if _is_result_type(result, "dynamics", "TransientResult"):
        return _adapt_transient(result, impact=False)
    if _is_result_type(result, "contact", "SphereImpactResult"):
        return _adapt_transient(result, impact=True)
    if _is_result_type(result, "modal", "ModalResult"):
        return _adapt_modes(result, buckling=False)
    if _is_result_type(result, "buckling", "BucklingResult"):
        return _adapt_modes(result, buckling=True)
    if _is_result_type(result, "results", "FEResult"):
        return _adapt_fe_result(result)
    if _is_result_type(result, "capacity_workflow", "CapacityWorkflowResult"):
        nested = solve_outcome(getattr(result, "nonlinear_result"))
        fields = {
            "control_kind": nested.control_kind,
            "requested_control": nested.requested_control,
            "achieved_control": nested.achieved_control,
            "last_converged_frame": nested.last_converged_frame,
            "last_converged_increment": nested.last_converged_increment,
        }
        status = str(getattr(result, "status", "unknown"))
        capacity_factor = getattr(result, "capacity_factor", None)
        if status == "completed" and nested.completed:
            return SolveOutcome.success(
                "capacity_target_completed",
                **fields,
                metadata={"capacity_factor": capacity_factor},
            )
        if nested.has_results:
            return SolveOutcome.stopped(
                nested.termination or status,
                converged=nested.converged,
                **fields,
                metadata={
                    "workflow_status": status,
                    "capacity_factor": capacity_factor,
                },
            )
        return SolveOutcome.failure(nested.termination or status, **fields)
    if _is_result_type(result, "nonlinear", "NonlinearLimitPointResult"):
        status = str(getattr(result, "status", "unknown"))
        steps = tuple(getattr(result, "steps", ()) or ())
        fields = {
            "control_kind": "load_factor",
            "requested_control": None,
            "achieved_control": getattr(result, "last_load_factor", None),
            "last_converged_increment": (
                getattr(steps[-1], "step_index", None) if steps else None
            ),
        }
        if bool(getattr(result, "converged", False)):
            return SolveOutcome.success(status, **fields)
        if steps:
            return SolveOutcome.stopped(status, converged=False, **fields)
        return SolveOutcome.failure(status, **fields)
    if _is_result_type(result, "runtime", "LightweightFEMResult"):
        status = str(getattr(result, "status", "unknown"))
        if status == "ok":
            return SolveOutcome.success("runtime_analysis_completed")
        return SolveOutcome.failure(status or "runtime_analysis_failed")
    if _is_result_type(
        result, "anystructure_fem_mode", "AnyStructureFEMResult"
    ):
        status = str(getattr(result, "status", "unknown"))
        valid = getattr(result, "valid", None)
        if not isinstance(valid, bool):
            raise TypeError("AnyStructureFEMResult valid flag must be bool")
        displacements = getattr(result, "displacements", None)
        try:
            has_results = displacements is not None and len(displacements) > 0
        except (TypeError, ValueError, OverflowError):
            has_results = False
        metadata = {
            "node_count": getattr(result, "node_count", 0),
            "element_count": getattr(result, "element_count", 0),
            "static_solver_status": getattr(result, "static_solver_status", "not_run"),
            "buckling_solver_status": getattr(
                result, "buckling_solver_status", "not_run"
            ),
        }
        if valid and status == "ok":
            return SolveOutcome.success(
                "generated_geometry_fem_completed", metadata=metadata
            )
        return SolveOutcome.failure(
            str(getattr(result, "invalid_reason", None) or status),
            has_results=has_results,
            metadata=metadata,
        )
    if _is_result_type(
        result, "imperfections", "ImperfectionCalibrationResult"
    ):
        nested_result = getattr(result, "result", None)
        nested = solve_outcome(nested_result) if nested_result is not None else None
        iterations = getattr(result, "iterations", None)
        converged = getattr(result, "converged", None)
        if not isinstance(converged, bool):
            raise TypeError("ImperfectionCalibrationResult converged flag must be bool")
        fields = {
            "control_kind": "calibration_iterations",
            "requested_control": None,
            "achieved_control": iterations,
            "last_converged_increment": iterations,
        }
        metadata = {
            "amplitude": getattr(result, "amplitude", None),
            "capacity": getattr(result, "capacity", None),
        }
        if converged and nested is not None and nested.completed:
            return SolveOutcome.success(
                "imperfection_calibration_converged",
                **fields,
                metadata=metadata,
            )
        if nested is not None and nested.has_results:
            return SolveOutcome.stopped(
                "imperfection_calibration_not_converged",
                converged=False,
                **fields,
                metadata=metadata,
            )
        history = tuple(getattr(result, "history", ()) or ())
        if history:
            return SolveOutcome.stopped(
                "imperfection_calibration_not_converged",
                converged=False,
                **fields,
                metadata=metadata,
            )
        return SolveOutcome.failure(
            "imperfection_calibration_not_converged",
            **fields,
            metadata=metadata,
        )
    return None


def solve_outcome(result: Any) -> SolveOutcome:
    """Resolve the authoritative common outcome from a qualified result."""

    outcome = getattr(result, "outcome", None)
    if isinstance(outcome, SolveOutcome):
        return outcome
    if isinstance(outcome, Mapping):
        return SolveOutcome.from_dict(outcome)
    adapted = _adapt_known_result(result)
    if adapted is not None:
        return adapted
    raise TypeError(f"{type(result).__name__} does not provide a SolveOutcome")
