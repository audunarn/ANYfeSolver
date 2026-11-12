"""Analysis-local diagnostics for nonlinear solver acceleration paths.

The public performance status APIs intentionally expose process-wide counters.
Those counters are useful for service health, but subtracting two snapshots is
not safe when analyses overlap.  This module mirrors the small set of execution
events needed by a result into a :class:`contextvars.ContextVar` scope.  The
scope is inclusive: an outer arc-length or displacement-control analysis also
records work performed by its nested preload solve.
"""

from __future__ import annotations

import functools
import threading
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar, cast


F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class _AnalysisPerformanceRecorder:
    nested_analysis_count: int = 0
    assembly_path_counts: Counter[str] = field(default_factory=Counter)
    assembly_fallback_reason_counts: Counter[str] = field(default_factory=Counter)
    assembly_tangent_calls: int = 0
    assembly_residual_only_calls: int = 0
    assembly_seconds: float = 0.0
    plans: Dict[int, Any] = field(default_factory=dict)
    hill_public_call_count: int = 0
    hill_point_count: int = 0
    hill_compiled_call_count: int = 0
    hill_compiled_point_count: int = 0
    hill_scalar_fallback_call_count: int = 0
    hill_scalar_fallback_point_count: int = 0
    hill_row_fallback_count: int = 0
    hill_numerical_tangent_row_count: int = 0
    hill_fallback_reason_counts: Counter[str] = field(default_factory=Counter)
    hill_last_call: Optional[Dict[str, Any]] = None
    corotational_force_block_rotations: int = 0
    corotational_tangent_block_rotations: int = 0
    corotational_dense_consistent_rotations: int = 0
    solver_event_counts: Counter[str] = field(default_factory=Counter)
    solver_failure_reason_counts: Counter[str] = field(default_factory=Counter)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record_nested_analysis(self) -> None:
        with self._lock:
            self.nested_analysis_count += 1

    def record_assembly(
        self,
        *,
        path: str,
        tangent: bool,
        elapsed_seconds: float,
        plan: Any,
        fallback_reason: Optional[str],
    ) -> None:
        with self._lock:
            self.assembly_path_counts[str(path)] += 1
            if tangent:
                self.assembly_tangent_calls += 1
            else:
                self.assembly_residual_only_calls += 1
            self.assembly_seconds += max(float(elapsed_seconds), 0.0)
            if fallback_reason:
                self.assembly_fallback_reason_counts[str(fallback_reason)] += 1
            if plan is not None:
                self.plans.setdefault(id(plan), plan)

    def record_hill48(
        self,
        *,
        point_count: int,
        curve_name: str,
        compiled: bool,
        scalar_fallback_points: int,
        fallback_reason_counts: Mapping[str, int],
        numerical_tangent_rows: int,
    ) -> None:
        points = int(point_count)
        scalar_points = int(scalar_fallback_points)
        reasons = {
            str(reason): int(count)
            for reason, count in fallback_reason_counts.items()
            if int(count) > 0
        }
        with self._lock:
            self.hill_public_call_count += 1
            self.hill_point_count += points
            if compiled:
                self.hill_compiled_call_count += 1
                self.hill_compiled_point_count += points
            if scalar_points:
                self.hill_scalar_fallback_call_count += 1
                self.hill_scalar_fallback_point_count += scalar_points
            self.hill_row_fallback_count += sum(reasons.values())
            self.hill_numerical_tangent_row_count += int(numerical_tangent_rows)
            self.hill_fallback_reason_counts.update(reasons)
            self.hill_last_call = {
                "point_count": points,
                "curve": str(curve_name),
                "path": "compiled" if compiled else "scalar_reference",
                "scalar_fallback_points": scalar_points,
                "fallback_reason_counts": dict(sorted(reasons.items())),
                "numerical_tangent_rows": int(numerical_tangent_rows),
            }

    def record_corotational(
        self,
        *,
        force_blocks: int = 0,
        tangent_blocks: int = 0,
        dense_consistent: int = 0,
    ) -> None:
        with self._lock:
            self.corotational_force_block_rotations += int(force_blocks)
            self.corotational_tangent_block_rotations += int(tangent_blocks)
            self.corotational_dense_consistent_rotations += int(dense_consistent)

    def record_solver_event(
        self,
        *,
        event: str,
        count: int = 1,
        failure_reason: Optional[str] = None,
    ) -> None:
        made_count = int(count)
        if made_count <= 0:
            return
        with self._lock:
            self.solver_event_counts[str(event)] += made_count
            if failure_reason:
                self.solver_failure_reason_counts[str(failure_reason)] += made_count

    def _solver_payload(self) -> Dict[str, Any]:
        with self._lock:
            counts = dict(sorted(self.solver_event_counts.items()))
            failure_counts = dict(
                sorted(self.solver_failure_reason_counts.items())
            )
        return {
            "event_counts": counts,
            "failure_reason_counts": failure_counts,
            "linear_factorizations": int(counts.get("linear_factorization", 0)),
            "linear_solves": int(counts.get("linear_solve", 0)),
            "rejected_full_steps": int(counts.get("rejected_full_step", 0)),
            "backtracks": int(counts.get("backtrack", 0)),
            "promotion_evaluations": int(counts.get("promotion_evaluation", 0)),
            "recoverable_trial_failures": int(
                counts.get("recoverable_trial_failure", 0)
            ),
            "failed_increments": int(counts.get("failed_increment", 0)),
            "failed_work": {
                "increment_count": int(counts.get("failed_increment", 0)),
                "newton_iterations": int(
                    counts.get("failed_increment_iteration", 0)
                ),
                "rejected_trial_evaluations": int(counts.get("backtrack", 0)),
                "recoverable_trial_failures": int(
                    counts.get("recoverable_trial_failure", 0)
                ),
            },
            "reaction_force_reuse_count": int(
                counts.get("reaction_force_reuse", 0)
            ),
            "reaction_force_reassembly_count": int(
                counts.get("reaction_force_reassembly", 0)
            ),
        }

    def _assembly_payload(self) -> Dict[str, Any]:
        with self._lock:
            path_counts = dict(sorted(self.assembly_path_counts.items()))
            fallback_counts = dict(
                sorted(self.assembly_fallback_reason_counts.items())
            )
            plans = list(self.plans.values())
            tangent_calls = int(self.assembly_tangent_calls)
            residual_calls = int(self.assembly_residual_only_calls)
            elapsed = float(self.assembly_seconds)

        plan_payloads = []
        for plan in plans:
            try:
                diagnostics = dict(plan.diagnostics())
                diagnostics["timings_scope"] = "model_cache_cumulative"
            except Exception as exc:  # pragma: no cover - defensive observability
                diagnostics = {
                    "diagnostics_error": f"{type(exc).__name__}:{exc}",
                }
            plan_payloads.append(diagnostics)

        batched_calls = int(
            path_counts.get("persistent_full_coordinate", 0)
            + path_counts.get("direct_reduced", 0)
        )
        total_calls = tangent_calls + residual_calls
        if total_calls == 0:
            fallback_reason = "nonlinear_assembly_not_exercised"
        elif batched_calls == 0:
            fallback_reason = (
                next(iter(fallback_counts))
                if len(fallback_counts) == 1
                else "reference_full_coordinate_selected"
            )
        elif path_counts.get("reference_full_coordinate", 0):
            fallback_reason = "partial_reference_full_coordinate_fallback"
        else:
            fallback_reason = None
        return {
            "activated": bool(batched_calls),
            "fallback_reason": fallback_reason,
            "fallback_reason_counts": fallback_counts,
            "path_counts": path_counts,
            "calls": int(total_calls),
            "tangent_calls": tangent_calls,
            "residual_only_calls": residual_calls,
            "elapsed_seconds": elapsed,
            "unique_plan_count": int(len(plans)),
            "plan_reused": bool(plans and batched_calls > len(plans)),
            "plan_reuse_scope": "within_analysis",
            "plans": plan_payloads,
        }

    def _hill48_payload(self) -> Dict[str, Any]:
        from . import vectorized_hill48

        with self._lock:
            public_calls = int(self.hill_public_call_count)
            compiled_calls = int(self.hill_compiled_call_count)
            scalar_points = int(self.hill_scalar_fallback_point_count)
            fallback_counts = dict(sorted(self.hill_fallback_reason_counts.items()))
            last_call = None if self.hill_last_call is None else dict(self.hill_last_call)
            if last_call is not None:
                last_call["fallback_reason_counts"] = dict(
                    last_call.get("fallback_reason_counts", {})
                )
            payload = {
                "public_call_count": public_calls,
                "point_count": int(self.hill_point_count),
                "compiled_call_count": compiled_calls,
                "compiled_point_count": int(self.hill_compiled_point_count),
                "scalar_fallback_call_count": int(
                    self.hill_scalar_fallback_call_count
                ),
                "scalar_fallback_point_count": scalar_points,
                "row_fallback_count": int(self.hill_row_fallback_count),
                "numerical_tangent_row_count": int(
                    self.hill_numerical_tangent_row_count
                ),
                "fallback_reason_counts": fallback_counts,
                "last_call": last_call,
            }
        activated = compiled_calls > 0
        if public_calls == 0:
            fallback_reason = "hill48_not_exercised"
        elif not activated:
            fallback_reason = (
                next(iter(fallback_counts))
                if len(fallback_counts) == 1
                else "scalar_reference_only"
            )
        elif scalar_points or fallback_counts:
            fallback_reason = "partial_scalar_fallback"
        else:
            fallback_reason = None
        payload.update(
            {
                "eligible": bool(vectorized_hill48.JIT_ENABLED),
                "activated": activated,
                "fallback_reason": fallback_reason,
                "fast_path": "hill48_flattened_numba_return_map",
                "jit_backend": vectorized_hill48.JIT_BACKEND,
                "jit_disabled_reason": vectorized_hill48.JIT_DISABLED_REASON,
            }
        )
        return payload

    def _corotational_payload(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        from .corotational_performance import corotational_performance_status

        process_status = corotational_performance_status()
        with self._lock:
            force_blocks = int(self.corotational_force_block_rotations)
            tangent_blocks = int(self.corotational_tangent_block_rotations)
            dense_consistent = int(self.corotational_dense_consistent_rotations)
        requested = str(info.get("kinematics", "von_karman")) == "corotational"
        activated = bool(force_blocks or tangent_blocks)
        exercised = bool(activated or dense_consistent)
        tangent_mode = str(info.get("corotational_tangent", "not_applicable"))
        if not requested:
            fallback_reason = "kinematics_not_corotational"
        elif not exercised:
            fallback_reason = "corotational_response_not_exercised"
        elif tangent_mode == "consistent" and dense_consistent:
            fallback_reason = "consistent_tangent_requires_dense_chain_rule"
        elif process_status.get("fallback_reason"):
            fallback_reason = str(process_status["fallback_reason"])
        else:
            fallback_reason = None
        return {
            "eligible": requested,
            "activated": activated,
            "exercised": exercised,
            "fallback_reason": fallback_reason,
            "tangent_mode": tangent_mode,
            "force_block_rotations": force_blocks,
            "tangent_block_rotations": tangent_blocks,
            "dense_consistent_rotations": dense_consistent,
            "force_block_activated": force_blocks > 0,
            "rotated_tangent_block_activated": tangent_blocks > 0,
            "dense_consistent_tangent_activated": dense_consistent > 0,
            "fast_path_name": process_status.get("fast_path_name"),
            "backend": process_status.get("backend"),
            "jit_enabled": process_status.get("jit_enabled"),
            "backend_fallback_reason": process_status.get("fallback_reason"),
        }

    def payload(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            from .nonlinear_performance_batch_c import (
                current_batch_c_analysis_diagnostics,
            )

            direct_reduced = current_batch_c_analysis_diagnostics()
        except Exception as exc:  # pragma: no cover - defensive observability
            direct_reduced = {
                "activated": False,
                "fallback_reason": f"diagnostics_unavailable:{type(exc).__name__}:{exc}",
            }
        with self._lock:
            nested_analysis_count = int(self.nested_analysis_count)
        return {
            "scope": "analysis_local_inclusive",
            "nested_analysis_count": nested_analysis_count,
            "assembly": self._assembly_payload(),
            "direct_reduced_assembly": direct_reduced,
            "solver": self._solver_payload(),
            "hill48": self._hill48_payload(),
            "corotational": self._corotational_payload(info),
        }


_ACTIVE_RECORDERS: ContextVar[tuple[_AnalysisPerformanceRecorder, ...]] = (
    ContextVar("anyfesolver_active_nonlinear_analysis_recorders", default=())
)


def _active_recorders() -> tuple[_AnalysisPerformanceRecorder, ...]:
    return _ACTIVE_RECORDERS.get()


def record_nonlinear_assembly_execution(
    *,
    path: str,
    tangent: bool,
    elapsed_seconds: float,
    plan: Any = None,
    fallback_reason: Optional[str] = None,
) -> None:
    """Record one completed global nonlinear assembly in active analyses."""

    for recorder in _active_recorders():
        recorder.record_assembly(
            path=path,
            tangent=tangent,
            elapsed_seconds=elapsed_seconds,
            plan=plan,
            fallback_reason=fallback_reason,
        )


def record_hill48_analysis_execution(
    *,
    point_count: int,
    curve_name: str,
    compiled: bool,
    scalar_fallback_points: int = 0,
    fallback_reason_counts: Optional[Mapping[str, int]] = None,
    numerical_tangent_rows: int = 0,
) -> None:
    """Mirror one public Hill-48 execution into active analyses."""

    for recorder in _active_recorders():
        recorder.record_hill48(
            point_count=point_count,
            curve_name=curve_name,
            compiled=compiled,
            scalar_fallback_points=scalar_fallback_points,
            fallback_reason_counts=fallback_reason_counts or {},
            numerical_tangent_rows=numerical_tangent_rows,
        )


def record_corotational_analysis_execution(
    *,
    force_blocks: int = 0,
    tangent_blocks: int = 0,
    dense_consistent: int = 0,
) -> None:
    """Mirror corotational block/dense execution into active analyses."""

    for recorder in _active_recorders():
        recorder.record_corotational(
            force_blocks=force_blocks,
            tangent_blocks=tangent_blocks,
            dense_consistent=dense_consistent,
        )


def record_nonlinear_solver_event(
    event: str,
    *,
    count: int = 1,
    failure_reason: Optional[str] = None,
) -> None:
    """Record one solver/globalization event in active analyses."""

    for recorder in _active_recorders():
        recorder.record_solver_event(
            event=event,
            count=count,
            failure_reason=failure_reason,
        )


def capture_nonlinear_analysis_diagnostics(func: F) -> F:
    """Attach task/thread-local performance diagnostics to a solver result."""

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any):
        recorder = _AnalysisPerformanceRecorder()
        parent_recorders = _active_recorders()
        for parent in parent_recorders:
            parent.record_nested_analysis()
        token = _ACTIVE_RECORDERS.set((*parent_recorders, recorder))
        try:
            result = func(*args, **kwargs)
            info = getattr(result, "info", None)
            if isinstance(info, dict):
                try:
                    info["nonlinear_performance"] = recorder.payload(info)
                except Exception as exc:  # Diagnostics must never fail a solve.
                    info["nonlinear_performance"] = {
                        "scope": "analysis_local_inclusive",
                        "diagnostics_error": f"{type(exc).__name__}:{exc}",
                    }
            return result
        finally:
            _ACTIVE_RECORDERS.reset(token)

    return cast(F, wrapped)


__all__ = [
    "capture_nonlinear_analysis_diagnostics",
    "record_corotational_analysis_execution",
    "record_hill48_analysis_execution",
    "record_nonlinear_assembly_execution",
    "record_nonlinear_solver_event",
]
